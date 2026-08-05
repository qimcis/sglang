#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace state_protection {

namespace paged {

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kMaxSources = 4;
constexpr int64_t kValidMarker = 0x5041474550524f54LL;  // "PAGEPROT"

constexpr int32_t kInvalidRequest = 1 << 8;
constexpr int32_t kInvalidMapping = 1 << 9;
constexpr int32_t kInvalidSwaMapping = 1 << 10;
constexpr int32_t kUnsealedPayload = 1 << 11;
constexpr int32_t kPayloadMismatch = 1 << 12;
constexpr int32_t kInvalidWrite = 1 << 13;
constexpr int32_t kPositionMismatch = 1 << 14;
constexpr int32_t kCanaryFailure = 1 << 24;

struct ByteSource {
  const uint8_t* data;
  int64_t rows;
  int64_t row_bytes;
};

SGL_DEVICE uint64_t splitmix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

SGL_DEVICE uint64_t fold_sources_serial(
    const ByteSource* sources,
    int32_t num_sources,
    int64_t slot,
    uint64_t seed) {
  // The digest follows the payload across compaction, CPU offload, and PD
  // transfer, all of which may move it to a different physical slot. Slot
  // identity is checked independently by the request mapping/canary contract.
  uint64_t digest = splitmix64(seed);
  uint64_t coordinate = 0;
  for (int32_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    const ByteSource source = sources[source_idx];
    if (slot < 0 || slot >= source.rows) return 0;
    const uint8_t* row = source.data + slot * source.row_bytes;
    for (int64_t offset = 0; offset < source.row_bytes; ++offset) {
      digest ^= splitmix64(
          coordinate ^ static_cast<uint64_t>(row[offset]));
      coordinate += 1;
    }
    digest = splitmix64(digest ^ static_cast<uint64_t>(source.row_bytes));
  }
  return splitmix64(digest ^ coordinate ^ seed);
}

struct PagedParams {
  int64_t* sidecar;
  int64_t sidecar_stride;
  int64_t num_slots;
  int32_t* req_to_token;
  int64_t req_stride;
  int64_t num_request_slots;
  int64_t max_context_len;
  const int64_t* request_indices;
  const int32_t* prefix_lens;
  int32_t* failure_status;
  const int32_t* enabled;
  const int32_t* canary_violation_index;
  const int32_t* canary_forward_start;
  const int64_t* swa_lut;
  int64_t swa_lut_size;
  int32_t swa_window_size;
  ByteSource sources[kMaxSources];
  int32_t num_sources;
  ByteSource dsa_source;
  int32_t dsa_page_size;
  int32_t dsa_token_bytes;
  int32_t dsa_aux_bytes;
  bool has_dsa_source;
  uint64_t domain_seed;
};

SGL_DEVICE uint64_t fold_payload(
    const PagedParams& p,
    int64_t slot) {
  uint64_t digest = fold_sources_serial(
      p.sources, p.num_sources, slot, p.domain_seed);
  if (!p.has_dsa_source) return digest;
  const int64_t page = slot / p.dsa_page_size;
  const int64_t offset = slot % p.dsa_page_size;
  if (page < 0 || page >= p.dsa_source.rows) return 0;
  const uint8_t* row = p.dsa_source.data + page * p.dsa_source.row_bytes;
  const int64_t aux_base =
      static_cast<int64_t>(p.dsa_page_size) * p.dsa_token_bytes;
  for (int32_t byte = 0; byte < p.dsa_token_bytes; ++byte) {
    digest ^= splitmix64(
        static_cast<uint64_t>(byte) ^
        static_cast<uint64_t>(row[offset * p.dsa_token_bytes + byte]));
  }
  for (int32_t byte = 0; byte < p.dsa_aux_bytes; ++byte) {
    digest ^= splitmix64(
        static_cast<uint64_t>(p.dsa_token_bytes + byte) ^
        static_cast<uint64_t>(
            row[aux_base + offset * p.dsa_aux_bytes + byte]));
  }
  return splitmix64(digest ^ static_cast<uint64_t>(p.dsa_source.row_bytes));
}

SGL_DEVICE int64_t translate_slot(
    const PagedParams& p,
    int64_t full_slot,
    int32_t* status) {
  if (full_slot <= 0) {
    *status |= kInvalidMapping;
    return 0;
  }
  if (p.swa_lut == nullptr) {
    if (full_slot >= p.num_slots) {
      *status |= kInvalidMapping;
      return 0;
    }
    return full_slot;
  }
  if (full_slot >= p.swa_lut_size) {
    *status |= kInvalidSwaMapping;
    return 0;
  }
  const int64_t translated = p.swa_lut[full_slot];
  if (translated <= 0 || translated >= p.num_slots) {
    *status |= kInvalidSwaMapping;
    return 0;
  }
  return translated;
}

__global__ void validate_mapping_kernel(PagedParams p, int32_t batch_size) {
  if (p.enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size) return;
  const int64_t request = p.request_indices[row];
  if (request <= 0 && p.prefix_lens[row] <= 0) return;
  if (request <= 0 || request >= p.num_request_slots) {
    if (threadIdx.x == 0) atomicOr(p.failure_status, kInvalidRequest);
    return;
  }

  const int32_t prefix_len = p.prefix_lens[row];
  if (prefix_len < 0 || prefix_len > p.max_context_len) {
    if (threadIdx.x == 0) atomicOr(p.failure_status + request, kInvalidMapping);
    return;
  }
  const int32_t first_position =
      p.swa_lut == nullptr || p.swa_window_size <= 0
      ? 0
      : (prefix_len > p.swa_window_size ? prefix_len - p.swa_window_size : 0);
  for (int32_t position = first_position + threadIdx.x;
       position < prefix_len;
       position += blockDim.x) {
    int32_t status = 0;
    int32_t* mapping = p.req_to_token + request * p.req_stride + position;
    const int64_t full_slot = static_cast<int64_t>(*mapping);
    const int64_t slot = translate_slot(p, full_slot, &status);
    if (status == 0) {
      const int64_t* fields = p.sidecar + slot * p.sidecar_stride;
      if (fields[1] != kValidMarker) {
        status |= kUnsealedPayload;
      } else if (fields[2] != position) {
        status |= kPositionMismatch;
      }
    }
    if (status != 0) {
      // Slot zero is the reserved sink. This protects metadata built after the
      // preflight; an already-built consumer remains covered by the token gate.
      *mapping = 0;
      atomicOr(p.failure_status + request, status);
    }
  }
}

__global__ void validate_payload_kernel(PagedParams p, int32_t batch_size) {
  if (p.enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size) return;
  const int64_t request = p.request_indices[row];
  if (request <= 0 && p.prefix_lens[row] <= 0) return;
  if (request <= 0 || request >= p.num_request_slots) {
    if (threadIdx.x == 0) atomicOr(p.failure_status, kInvalidRequest);
    return;
  }

  const int32_t prefix_len = p.prefix_lens[row];
  if (prefix_len < 0 || prefix_len > p.max_context_len) {
    if (threadIdx.x == 0) atomicOr(p.failure_status + request, kInvalidMapping);
    return;
  }
  const int32_t first_position =
      p.swa_lut == nullptr || p.swa_window_size <= 0
      ? 0
      : (prefix_len > p.swa_window_size ? prefix_len - p.swa_window_size : 0);
  const bool canary_failed =
      p.canary_violation_index[0] != p.canary_forward_start[0];
  if (canary_failed && threadIdx.x == 0) {
    atomicOr(p.failure_status + request, kCanaryFailure);
  }
  for (int32_t position = first_position + threadIdx.x;
       position < prefix_len;
       position += blockDim.x) {
    int32_t status = canary_failed ? kCanaryFailure : 0;
    int32_t* mapping = p.req_to_token + request * p.req_stride + position;
    const int64_t full_slot = static_cast<int64_t>(*mapping);
    const int64_t slot = translate_slot(p, full_slot, &status);
    if (status == 0) {
      const int64_t* fields = p.sidecar + slot * p.sidecar_stride;
      if (fields[1] != kValidMarker) {
        status |= kUnsealedPayload;
      } else if (fields[2] != position) {
        status |= kPositionMismatch;
      } else {
        const uint64_t actual = fold_payload(p, slot);
        if (static_cast<uint64_t>(fields[0]) != actual) {
          status |= kPayloadMismatch;
        }
      }
    }
    if (status != 0) {
      // Validation is part of the accessor contract: the protected consumer
      // must never observe a mapping whose identity or payload check failed.
      *mapping = 0;
      atomicOr(p.failure_status + request, status);
    }
  }
}

template <typename IndexT>
__global__ void validate_write_slots_kernel(
    PagedParams p,
    IndexT* write_slots,
    const int32_t* write_lens,
    int32_t batch_size,
    int32_t num_writes) {
  if (p.enabled[0] == 0) return;
  const int32_t request_row = blockIdx.x;
  if (request_row >= batch_size) return;

  const int64_t request = p.request_indices[request_row];
  int64_t declared_writes = 0;
  bool invalid_layout = false;
  for (int32_t row = 0; row < batch_size; ++row) {
    const int32_t row_len = write_lens[row];
    invalid_layout |= row_len < 0;
    if (row_len > 0) declared_writes += row_len;
  }
  invalid_layout |= declared_writes != num_writes;
  if (invalid_layout) {
    if (request_row == 0) {
      for (int32_t offset = threadIdx.x; offset < num_writes; offset += blockDim.x) {
        write_slots[offset] = static_cast<IndexT>(0);
      }
      if (threadIdx.x == 0) atomicOr(p.failure_status, kInvalidWrite);
    }
    return;
  }

  const int32_t write_len = write_lens[request_row];

  int64_t write_start = 0;
  for (int32_t prior = 0; prior < request_row; ++prior) {
    const int32_t prior_len = write_lens[prior];
    if (prior_len > 0) write_start += prior_len;
  }
  if (write_start + write_len > num_writes) {
    if (threadIdx.x == 0) {
      const int64_t status_row =
          request > 0 && request < p.num_request_slots ? request : 0;
      atomicOr(p.failure_status + status_row, kInvalidWrite);
    }
    return;
  }

  if (request <= 0 || request >= p.num_request_slots) {
    for (int32_t offset = threadIdx.x; offset < write_len; offset += blockDim.x) {
      IndexT* write_slot = write_slots + write_start + offset;
      if (static_cast<int64_t>(*write_slot) > 0) {
        *write_slot = static_cast<IndexT>(0);
        atomicOr(p.failure_status, kInvalidRequest | kInvalidWrite);
      }
    }
    return;
  }

  for (int32_t offset = threadIdx.x; offset < write_len; offset += blockDim.x) {
    IndexT* write_slot = write_slots + write_start + offset;
    int32_t status = 0;
    translate_slot(p, static_cast<int64_t>(*write_slot), &status);
    if (status != 0) {
      *write_slot = static_cast<IndexT>(0);
      atomicOr(p.failure_status + request, kInvalidWrite | status);
    }
  }
}

template <typename IndexT>
__global__ void seal_payload_kernel(
    PagedParams p,
    const IndexT* write_slots,
    const int32_t* write_lens,
    int32_t batch_size,
    int32_t num_writes) {
  if (p.enabled[0] == 0) return;
  const int32_t request_row = blockIdx.x;
  if (request_row >= batch_size) return;

  const int64_t request = p.request_indices[request_row];
  const int32_t write_len = write_lens[request_row];
  if (write_len < 0) {
    if (threadIdx.x == 0) atomicOr(p.failure_status, kInvalidWrite);
    return;
  }

  int64_t write_start = 0;
  for (int32_t prior = 0; prior < request_row; ++prior) {
    const int32_t prior_len = write_lens[prior];
    if (prior_len > 0) write_start += prior_len;
  }
  if (write_start + write_len > num_writes) {
    if (threadIdx.x == 0) {
      const int64_t status_row =
          request > 0 && request < p.num_request_slots ? request : 0;
      atomicOr(p.failure_status + status_row, kInvalidWrite);
    }
    return;
  }

  if (request <= 0 || request >= p.num_request_slots) {
    for (int32_t offset = threadIdx.x; offset < write_len; offset += blockDim.x) {
      if (static_cast<int64_t>(write_slots[write_start + offset]) > 0) {
        atomicOr(p.failure_status, kInvalidRequest | kInvalidWrite);
      }
    }
    return;
  }

  for (int32_t offset = threadIdx.x; offset < write_len; offset += blockDim.x) {
    int32_t status = 0;
    const int64_t full_slot =
        static_cast<int64_t>(write_slots[write_start + offset]);
    const int64_t slot = translate_slot(p, full_slot, &status);
    if (status != 0) {
      atomicOr(p.failure_status + request, kInvalidWrite | status);
      continue;
    }
    const uint64_t digest = fold_payload(p, slot);
    int64_t* fields = p.sidecar + slot * p.sidecar_stride;
    fields[0] = static_cast<int64_t>(digest);
    fields[2] = static_cast<int64_t>(p.prefix_lens[request_row]) + offset;
    __threadfence();
    fields[1] = kValidMarker;
  }
}

PagedParams build_params(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    tvm::ffi::TensorView dsa_source,
    int64_t num_sources,
    int64_t domain_seed,
    int64_t swa_window_size,
    int64_t dsa_page_size,
    int64_t dsa_token_bytes,
    int64_t dsa_aux_bytes,
    bool has_swa_lut,
    bool has_dsa_source,
    DLDevice& launch_device) {
  using namespace host;
  SymbolicSize N_slots = {"num_slots"};
  SymbolicSize N_requests = {"num_request_slots"};
  SymbolicSize Max_context = {"max_context_len"};
  SymbolicSize B = {"batch_size"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();

  TensorMatcher({N_slots, 3}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(sidecar);
  TensorMatcher({N_requests, Max_context}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(req_to_token);
  TensorMatcher({B}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(request_indices);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(prefix_lens);
  TensorMatcher({N_requests}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(failure_status);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(enabled);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_violation_index);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_forward_start);
  RuntimeCheck(num_sources > 0 && num_sources <= kMaxSources, "num_sources must be in [1, 4]");

  auto verify_source = [&](tvm::ffi::TensorView source, const char* rows, const char* cols) {
    SymbolicSize Rows = {rows};
    SymbolicSize Cols = {cols};
    TensorMatcher({Rows, Cols}).with_dtype<uint8_t>().with_device<kDLGPU>(device).verify(source);
  };
  verify_source(source_0, "source_rows_0", "source_cols_0");
  verify_source(source_1, "source_rows_1", "source_cols_1");
  verify_source(source_2, "source_rows_2", "source_cols_2");
  verify_source(source_3, "source_rows_3", "source_cols_3");
  verify_source(dsa_source, "dsa_source_rows", "dsa_source_cols");
  if (has_swa_lut) {
    SymbolicSize Swa_lut_size = {"swa_lut_size"};
    TensorMatcher({Swa_lut_size}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(swa_lut);
  }

  PagedParams p{};
  p.sidecar = static_cast<int64_t*>(sidecar.data_ptr());
  p.sidecar_stride = sidecar.stride(0);
  p.num_slots = N_slots.unwrap();
  p.req_to_token = static_cast<int32_t*>(req_to_token.data_ptr());
  p.req_stride = req_to_token.stride(0);
  p.num_request_slots = N_requests.unwrap();
  p.max_context_len = Max_context.unwrap();
  p.request_indices = static_cast<const int64_t*>(request_indices.data_ptr());
  p.prefix_lens = static_cast<const int32_t*>(prefix_lens.data_ptr());
  p.failure_status = static_cast<int32_t*>(failure_status.data_ptr());
  p.enabled = static_cast<const int32_t*>(enabled.data_ptr());
  p.canary_violation_index =
      static_cast<const int32_t*>(canary_violation_index.data_ptr());
  p.canary_forward_start =
      static_cast<const int32_t*>(canary_forward_start.data_ptr());
  p.swa_lut = has_swa_lut ? static_cast<const int64_t*>(swa_lut.data_ptr()) : nullptr;
  p.swa_lut_size = has_swa_lut ? swa_lut.size(0) : 0;
  p.swa_window_size = static_cast<int32_t>(swa_window_size);
  tvm::ffi::TensorView source_views[kMaxSources] = {
      source_0, source_1, source_2, source_3};
  for (int32_t idx = 0; idx < kMaxSources; ++idx) {
    p.sources[idx] = ByteSource{
        static_cast<const uint8_t*>(source_views[idx].data_ptr()),
        source_views[idx].size(0),
        source_views[idx].size(1)};
  }
  p.num_sources = static_cast<int32_t>(num_sources);
  p.dsa_source = ByteSource{
      static_cast<const uint8_t*>(dsa_source.data_ptr()),
      dsa_source.size(0),
      dsa_source.size(1)};
  p.dsa_page_size = static_cast<int32_t>(dsa_page_size);
  p.dsa_token_bytes = static_cast<int32_t>(dsa_token_bytes);
  p.dsa_aux_bytes = static_cast<int32_t>(dsa_aux_bytes);
  p.has_dsa_source = has_dsa_source;
  if (has_dsa_source) {
    RuntimeCheck(dsa_page_size > 0, "DSA page size must be positive");
    RuntimeCheck(dsa_token_bytes > 0, "DSA token bytes must be positive");
    RuntimeCheck(dsa_aux_bytes >= 0, "DSA auxiliary bytes must be non-negative");
    RuntimeCheck(
        dsa_source.size(1) >= dsa_page_size * (dsa_token_bytes + dsa_aux_bytes),
        "DSA source row is smaller than its declared token layout");
  }
  p.domain_seed = static_cast<uint64_t>(domain_seed);
  launch_device = device.unwrap();
  return p;
}

void validate_mapping(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    tvm::ffi::TensorView dsa_source,
    int64_t num_sources,
    int64_t domain_seed,
    int64_t swa_window_size,
    int64_t dsa_page_size,
    int64_t dsa_token_bytes,
    int64_t dsa_aux_bytes,
    bool has_swa_lut,
    bool has_dsa_source) {
  DLDevice device;
  auto p = build_params(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, num_sources,
      domain_seed, swa_window_size, dsa_page_size, dsa_token_bytes,
      dsa_aux_bytes, has_swa_lut, has_dsa_source, device);
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  if (batch_size == 0) return;
  host::LaunchKernel(batch_size, kThreads, device)(validate_mapping_kernel, p, batch_size);
}

void validate_payload(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    tvm::ffi::TensorView dsa_source,
    int64_t num_sources,
    int64_t domain_seed,
    int64_t swa_window_size,
    int64_t dsa_page_size,
    int64_t dsa_token_bytes,
    int64_t dsa_aux_bytes,
    bool has_swa_lut,
    bool has_dsa_source) {
  DLDevice device;
  auto p = build_params(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, num_sources,
      domain_seed, swa_window_size, dsa_page_size, dsa_token_bytes,
      dsa_aux_bytes, has_swa_lut, has_dsa_source, device);
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  if (batch_size == 0) return;
  host::LaunchKernel(batch_size, kThreads, device)(validate_payload_kernel, p, batch_size);
}

template <typename IndexT>
void validate_write_slots(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots,
    tvm::ffi::TensorView write_lens,
    int64_t num_sources,
    int64_t domain_seed,
    int64_t swa_window_size,
    int64_t dsa_page_size,
    int64_t dsa_token_bytes,
    int64_t dsa_aux_bytes,
    bool has_swa_lut,
    bool has_dsa_source) {
  using namespace host;
  DLDevice device;
  auto p = build_params(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, num_sources,
      domain_seed, swa_window_size, dsa_page_size, dsa_token_bytes,
      dsa_aux_bytes, has_swa_lut, has_dsa_source, device);
  SymbolicSize W = {"num_writes"};
  SymbolicSize B = {"batch_size"};
  SymbolicDevice matcher_device;
  matcher_device.set_options<kDLGPU>();
  TensorMatcher({W}).with_dtype<IndexT>().with_device<kDLGPU>(matcher_device).verify(write_slots);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLGPU>(matcher_device).verify(write_lens);
  RuntimeCheck(B.unwrap() == request_indices.size(0), "write_lens batch size mismatch");
  const int32_t num_writes = static_cast<int32_t>(write_slots.size(0));
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  if (batch_size == 0) return;
  host::LaunchKernel(batch_size, kThreads, device)(
      validate_write_slots_kernel<IndexT>, p,
      static_cast<IndexT*>(write_slots.data_ptr()),
      static_cast<const int32_t*>(write_lens.data_ptr()),
      batch_size, num_writes);
}

template <typename IndexT>
void seal_payload(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots,
    tvm::ffi::TensorView write_lens,
    int64_t num_sources,
    int64_t domain_seed,
    int64_t swa_window_size,
    int64_t dsa_page_size,
    int64_t dsa_token_bytes,
    int64_t dsa_aux_bytes,
    bool has_swa_lut,
    bool has_dsa_source) {
  using namespace host;
  DLDevice device;
  auto p = build_params(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, num_sources,
      domain_seed, swa_window_size, dsa_page_size, dsa_token_bytes,
      dsa_aux_bytes, has_swa_lut, has_dsa_source, device);
  SymbolicSize W = {"num_writes"};
  SymbolicSize B = {"batch_size"};
  SymbolicDevice matcher_device;
  matcher_device.set_options<kDLGPU>();
  TensorMatcher({W}).with_dtype<IndexT>().with_device<kDLGPU>(matcher_device).verify(write_slots);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLGPU>(matcher_device).verify(write_lens);
  RuntimeCheck(B.unwrap() == request_indices.size(0), "write_lens batch size mismatch");
  const int32_t num_writes = static_cast<int32_t>(write_slots.size(0));
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  if (batch_size == 0) return;
  host::LaunchKernel(batch_size, kThreads, device)(
      seal_payload_kernel<IndexT>, p,
      static_cast<const IndexT*>(write_slots.data_ptr()),
      static_cast<const int32_t*>(write_lens.data_ptr()),
      batch_size, num_writes);
}

}  // namespace

void validate_write_slots_i32(
    tvm::ffi::TensorView sidecar, tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices, tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut, tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3, tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots, tvm::ffi::TensorView write_lens,
    int64_t num_sources, int64_t domain_seed, int64_t swa_window_size,
    int64_t dsa_page_size, int64_t dsa_token_bytes, int64_t dsa_aux_bytes,
    bool has_swa_lut, bool has_dsa_source) {
  validate_write_slots<int32_t>(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3, dsa_source,
      write_slots, write_lens, num_sources, domain_seed, swa_window_size,
      dsa_page_size, dsa_token_bytes, dsa_aux_bytes,
      has_swa_lut, has_dsa_source);
}

void validate_write_slots_i64(
    tvm::ffi::TensorView sidecar, tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices, tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut, tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3, tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots, tvm::ffi::TensorView write_lens,
    int64_t num_sources, int64_t domain_seed, int64_t swa_window_size,
    int64_t dsa_page_size, int64_t dsa_token_bytes, int64_t dsa_aux_bytes,
    bool has_swa_lut, bool has_dsa_source) {
  validate_write_slots<int64_t>(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3, dsa_source,
      write_slots, write_lens, num_sources, domain_seed, swa_window_size,
      dsa_page_size, dsa_token_bytes, dsa_aux_bytes,
      has_swa_lut, has_dsa_source);
}

void seal_payload_i32(
    tvm::ffi::TensorView sidecar, tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices, tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut, tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3, tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots,
    tvm::ffi::TensorView write_lens,
    int64_t num_sources, int64_t domain_seed, int64_t swa_window_size,
    int64_t dsa_page_size, int64_t dsa_token_bytes, int64_t dsa_aux_bytes,
    bool has_swa_lut, bool has_dsa_source) {
  seal_payload<int32_t>(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, write_slots, write_lens,
      num_sources, domain_seed, swa_window_size, dsa_page_size,
      dsa_token_bytes, dsa_aux_bytes, has_swa_lut, has_dsa_source);
}

void seal_payload_i64(
    tvm::ffi::TensorView sidecar, tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView request_indices, tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView swa_lut, tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3, tvm::ffi::TensorView dsa_source,
    tvm::ffi::TensorView write_slots,
    tvm::ffi::TensorView write_lens,
    int64_t num_sources, int64_t domain_seed, int64_t swa_window_size,
    int64_t dsa_page_size, int64_t dsa_token_bytes, int64_t dsa_aux_bytes,
    bool has_swa_lut, bool has_dsa_source) {
  seal_payload<int64_t>(
      sidecar, req_to_token, request_indices, prefix_lens, failure_status,
      enabled, canary_violation_index, canary_forward_start, swa_lut,
      source_0, source_1, source_2, source_3,
      dsa_source, write_slots, write_lens,
      num_sources, domain_seed, swa_window_size, dsa_page_size,
      dsa_token_bytes, dsa_aux_bytes, has_swa_lut, has_dsa_source);
}

}  // namespace paged

}  // namespace state_protection
