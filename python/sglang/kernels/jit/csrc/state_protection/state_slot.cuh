#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace state_protection {

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kMaxSources = 4;
constexpr int64_t kValidMarker = 0x53544750524f5431LL;  // "STGPROT1"

constexpr int32_t kInvalidRequest = 1 << 0;
constexpr int32_t kInvalidSlot = 1 << 1;
constexpr int32_t kMappingMismatch = 1 << 2;
constexpr int32_t kGenerationMismatch = 1 << 3;
constexpr int32_t kUnsealedState = 1 << 4;
constexpr int32_t kPayloadMismatch = 1 << 5;
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

SGL_DEVICE uint64_t fold_sources(
    const ByteSource* sources,
    int32_t num_sources,
    int64_t slot,
    uint64_t seed) {
  // The checksum is payload identity, not allocation identity. Recurrent
  // state and its sidecar can move to a different physical slot during copy,
  // offload restore, or PD transfer; mapping plus generation protect the slot.
  uint64_t lane_hash = threadIdx.x == 0 ? splitmix64(seed) : 0ULL;
  uint64_t logical_offset = 0;
  for (int32_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    const ByteSource source = sources[source_idx];
    if (slot >= 0 && slot < source.rows) {
      const uint8_t* row = source.data + slot * source.row_bytes;
      for (int64_t offset = threadIdx.x; offset < source.row_bytes; offset += blockDim.x) {
        const uint64_t coordinate = logical_offset + static_cast<uint64_t>(offset);
        lane_hash ^= splitmix64(coordinate ^ static_cast<uint64_t>(row[offset]));
      }
    }
    logical_offset += static_cast<uint64_t>(source.row_bytes);
  }

  __shared__ uint64_t reduction[kThreads];
  reduction[threadIdx.x] = lane_hash;
  __syncthreads();
  for (int32_t width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) reduction[threadIdx.x] ^= reduction[threadIdx.x + width];
    __syncthreads();
  }
  return splitmix64(reduction[0] ^ logical_offset ^ seed);
}

template <typename IndexT>
struct StateSlotParams {
  int64_t* sidecar;
  int64_t sidecar_stride;
  int64_t num_slots;
  const int64_t* allocation_generations;
  const int64_t* request_indices;
  IndexT* cache_indices;
  const int32_t* require_sealed;
  const int32_t* expected_slots;
  const int64_t* expected_generations;
  int64_t num_request_slots;
  int32_t* failure_status;
  const int32_t* enabled;
  const int32_t* canary_violation_index;
  const int32_t* canary_forward_start;
  ByteSource sources[kMaxSources];
  int32_t num_sources;
  uint64_t domain_seed;
};

template <typename IndexT>
__global__ void validate_state_slots_kernel(
    StateSlotParams<IndexT> p,
    int32_t batch_size) {
  if (p.enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size) return;

  const int64_t request = p.request_indices[row];
  const int64_t slot = static_cast<int64_t>(p.cache_indices[row]);
  // CUDA-graph padding uses request/slot zero or -1. It is inert, not a fault.
  if (request <= 0 && slot <= 0) return;

  const bool require_sealed = p.require_sealed[row] != 0;
  int32_t local_status = 0;
  if (p.canary_violation_index[0] != p.canary_forward_start[0]) {
    local_status |= kCanaryFailure;
  }
  if (request <= 0 || request >= p.num_request_slots) {
    local_status |= kInvalidRequest;
  }
  if (slot <= 0 || slot >= p.num_slots) {
    local_status |= kInvalidSlot;
  }

  if (local_status == 0) {
    if (static_cast<int64_t>(p.expected_slots[request]) != slot) {
      local_status |= kMappingMismatch;
    }
    if (p.allocation_generations[slot] != p.expected_generations[request]) {
      local_status |= kGenerationMismatch;
    }
  }

  uint64_t actual_digest = 0;
  if (local_status == 0 && require_sealed) {
    actual_digest = fold_sources(p.sources, p.num_sources, slot, p.domain_seed);
  }

  if (threadIdx.x == 0 && local_status == 0 && require_sealed) {
    const int64_t* fields = p.sidecar + slot * p.sidecar_stride;
    if (fields[1] != kValidMarker) {
      local_status |= kUnsealedState;
    } else if (static_cast<uint64_t>(fields[0]) != actual_digest) {
      local_status |= kPayloadMismatch;
    }
  }

  __shared__ int32_t block_status;
  if (threadIdx.x == 0) block_status = local_status;
  __syncthreads();
  if (threadIdx.x == 0 && block_status != 0) {
    // Slot zero is reserved and zero-initialized. The consumer sees only this
    // safe sink when mapping, generation, or payload validation fails.
    p.cache_indices[row] = static_cast<IndexT>(0);
    const int64_t status_row =
        request > 0 && request < p.num_request_slots ? request : 0;
    atomicOr(p.failure_status + status_row, block_status);
  }
}

template <typename IndexT>
__global__ void seal_state_slots_kernel(
    StateSlotParams<IndexT> p,
    int32_t batch_size) {
  if (p.enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size) return;
  const int64_t request = p.request_indices[row];
  const int64_t slot = static_cast<int64_t>(p.cache_indices[row]);
  if (request <= 0 || request >= p.num_request_slots || slot <= 0 || slot >= p.num_slots) {
    return;
  }
  // A consumer may only seal the request's currently committed allocation.
  if (static_cast<int64_t>(p.expected_slots[request]) != slot ||
      p.allocation_generations[slot] != p.expected_generations[request]) {
    if (threadIdx.x == 0) atomicOr(p.failure_status + request, kMappingMismatch);
    return;
  }
  const uint64_t digest = fold_sources(p.sources, p.num_sources, slot, p.domain_seed);
  if (threadIdx.x == 0) {
    int64_t* fields = p.sidecar + slot * p.sidecar_stride;
    fields[0] = static_cast<int64_t>(digest);
    __threadfence();
    fields[1] = kValidMarker;
  }
}

template <typename IndexT>
__global__ void validate_unbound_state_slots_kernel(
    const int64_t* allocation_generations,
    int64_t num_slots,
    const int64_t* request_indices,
    IndexT* cache_indices,
    const int32_t* active_mask,
    const int32_t* expected_slots,
    const int64_t* expected_generations,
    int64_t expected_stride,
    int64_t expected_width,
    int64_t num_request_slots,
    int32_t* failure_status,
    const int32_t* enabled,
    const int32_t* canary_violation_index,
    const int32_t* canary_forward_start,
    int32_t batch_size) {
  if (enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size || active_mask[row] == 0 || threadIdx.x != 0) return;

  const int64_t request = request_indices[row];
  const int64_t slot = static_cast<int64_t>(cache_indices[row]);
  int32_t status = 0;
  if (canary_violation_index[0] != canary_forward_start[0]) {
    status |= kCanaryFailure;
  }
  if (request <= 0 || request >= num_request_slots) {
    status |= kInvalidRequest;
  }
  if (slot <= 0 || slot >= num_slots) {
    status |= kInvalidSlot;
  }
  if (status == 0) {
    bool owns_slot = false;
    bool generation_matches = false;
    const int64_t base = request * expected_stride;
    for (int64_t idx = 0; idx < expected_width; ++idx) {
      if (static_cast<int64_t>(expected_slots[base + idx]) == slot) {
        owns_slot = true;
        generation_matches |=
            expected_generations[base + idx] == allocation_generations[slot];
      }
    }
    if (!owns_slot) {
      status |= kMappingMismatch;
    } else if (!generation_matches) {
      status |= kGenerationMismatch;
    }
  }
  if (status != 0) {
    cache_indices[row] = static_cast<IndexT>(0);
    const int64_t status_row =
        request > 0 && request < num_request_slots ? request : 0;
    atomicOr(failure_status + status_row, status);
  }
}

template <typename IndexT>
__global__ void seal_unbound_state_slots_kernel(
    StateSlotParams<IndexT> p,
    int32_t batch_size) {
  if (p.enabled[0] == 0) return;
  const int32_t row = blockIdx.x;
  if (row >= batch_size || p.require_sealed[row] == 0) return;

  const int64_t request = p.request_indices[row];
  const int64_t slot = static_cast<int64_t>(p.cache_indices[row]);
  int32_t status = 0;
  if (p.canary_violation_index[0] != p.canary_forward_start[0]) {
    status |= kCanaryFailure;
  }
  if (request <= 0 || request >= p.num_request_slots) {
    status |= kInvalidRequest;
  }
  if (slot <= 0 || slot >= p.num_slots) {
    status |= kInvalidSlot;
  }
  if (status != 0) {
    if (threadIdx.x == 0) {
      const int64_t status_row =
          request > 0 && request < p.num_request_slots ? request : 0;
      atomicOr(p.failure_status + status_row, status);
    }
    return;
  }

  const uint64_t digest = fold_sources(p.sources, p.num_sources, slot, p.domain_seed);
  if (threadIdx.x == 0) {
    int64_t* fields = p.sidecar + slot * p.sidecar_stride;
    fields[0] = static_cast<int64_t>(digest);
    __threadfence();
    fields[1] = kValidMarker;
  }
}

template <typename IndexT>
StateSlotParams<IndexT> build_common_params(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView allocation_generations,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView expected_slots,
    tvm::ffi::TensorView expected_generations,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    int64_t num_sources,
    int64_t domain_seed,
    DLDevice& launch_device) {
  using namespace host;
  SymbolicSize N_slots = {"num_state_slots"};
  SymbolicSize N_requests = {"num_request_slots"};
  SymbolicSize B = {"batch_size"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();

  TensorMatcher({N_slots, 2}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(sidecar);
  TensorMatcher({N_slots}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(allocation_generations);
  TensorMatcher({B}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(request_indices);
  TensorMatcher({B}).with_dtype<IndexT>().with_device<kDLGPU>(device).verify(cache_indices);
  TensorMatcher({N_requests}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(expected_slots);
  TensorMatcher({N_requests}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(expected_generations);
  TensorMatcher({N_requests}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(failure_status);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(enabled);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_violation_index);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_forward_start);

  RuntimeCheck(num_sources > 0 && num_sources <= kMaxSources, "num_sources must be in [1, 4]");
  tvm::ffi::TensorView source_views[kMaxSources] = {source_0, source_1, source_2, source_3};

  auto verify_source = [&](tvm::ffi::TensorView source, const char* rows, const char* cols) {
    SymbolicSize Rows = {rows};
    SymbolicSize Cols = {cols};
    TensorMatcher({Rows, Cols})
        .with_dtype<uint8_t>()
        .with_device<kDLGPU>(device)
        .verify(source);
  };
  verify_source(source_0, "source_rows_0", "source_cols_0");
  verify_source(source_1, "source_rows_1", "source_cols_1");
  verify_source(source_2, "source_rows_2", "source_cols_2");
  verify_source(source_3, "source_rows_3", "source_cols_3");

  StateSlotParams<IndexT> p{};
  p.sidecar = static_cast<int64_t*>(sidecar.data_ptr());
  p.sidecar_stride = sidecar.stride(0);
  p.num_slots = N_slots.unwrap();
  p.allocation_generations = static_cast<const int64_t*>(allocation_generations.data_ptr());
  p.request_indices = static_cast<const int64_t*>(request_indices.data_ptr());
  p.cache_indices = static_cast<IndexT*>(cache_indices.data_ptr());
  p.expected_slots = static_cast<const int32_t*>(expected_slots.data_ptr());
  p.expected_generations = static_cast<const int64_t*>(expected_generations.data_ptr());
  p.num_request_slots = N_requests.unwrap();
  p.failure_status = static_cast<int32_t*>(failure_status.data_ptr());
  p.enabled = static_cast<const int32_t*>(enabled.data_ptr());
  p.canary_violation_index =
      static_cast<const int32_t*>(canary_violation_index.data_ptr());
  p.canary_forward_start =
      static_cast<const int32_t*>(canary_forward_start.data_ptr());
  for (int32_t idx = 0; idx < kMaxSources; ++idx) {
    p.sources[idx] = ByteSource{
        static_cast<const uint8_t*>(source_views[idx].data_ptr()),
        source_views[idx].size(0),
        source_views[idx].size(1)};
  }
  p.num_sources = static_cast<int32_t>(num_sources);
  p.domain_seed = static_cast<uint64_t>(domain_seed);
  launch_device = device.unwrap();
  return p;
}

template <typename IndexT>
void validate_impl(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView allocation_generations,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView require_sealed,
    tvm::ffi::TensorView expected_slots,
    tvm::ffi::TensorView expected_generations,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    int64_t num_sources,
    int64_t domain_seed) {
  DLDevice device;
  auto p = build_common_params<IndexT>(
      sidecar, allocation_generations, request_indices, cache_indices,
      expected_slots, expected_generations, failure_status, enabled,
      canary_violation_index, canary_forward_start,
      source_0, source_1, source_2, source_3, num_sources, domain_seed, device);
  host::SymbolicSize B = {"batch_size"};
  host::SymbolicDevice matcher_device;
  matcher_device.set_options<kDLGPU>();
  host::TensorMatcher({B})
      .with_dtype<int32_t>()
      .with_device<kDLGPU>(matcher_device)
      .verify(require_sealed);
  host::RuntimeCheck(B.unwrap() == request_indices.size(0), "require_sealed batch size mismatch");
  p.require_sealed = static_cast<const int32_t*>(require_sealed.data_ptr());
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  host::LaunchKernel(batch_size, kThreads, device)(validate_state_slots_kernel<IndexT>, p, batch_size);
}

template <typename IndexT>
void seal_impl(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView allocation_generations,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView expected_slots,
    tvm::ffi::TensorView expected_generations,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    int64_t num_sources,
    int64_t domain_seed) {
  DLDevice device;
  auto p = build_common_params<IndexT>(
      sidecar, allocation_generations, request_indices, cache_indices,
      expected_slots, expected_generations, failure_status, enabled,
      canary_violation_index, canary_forward_start,
      source_0, source_1, source_2, source_3, num_sources, domain_seed, device);
  p.require_sealed = nullptr;
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  host::LaunchKernel(batch_size, kThreads, device)(seal_state_slots_kernel<IndexT>, p, batch_size);
}

template <typename IndexT>
void validate_unbound_impl(
    tvm::ffi::TensorView allocation_generations,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView active_mask,
    tvm::ffi::TensorView expected_slots,
    tvm::ffi::TensorView expected_generations,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start) {
  using namespace host;
  SymbolicSize N_slots = {"num_state_slots"};
  SymbolicSize N_requests = {"num_request_slots"};
  SymbolicSize B = {"batch_size"};
  SymbolicSize T = {"tracking_slots_per_request"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();
  TensorMatcher({N_slots}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(allocation_generations);
  TensorMatcher({B}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(request_indices);
  TensorMatcher({B}).with_dtype<IndexT>().with_device<kDLGPU>(device).verify(cache_indices);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(active_mask);
  TensorMatcher({N_requests, T}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(expected_slots);
  TensorMatcher({N_requests, T}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(expected_generations);
  TensorMatcher({N_requests}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(failure_status);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(enabled);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_violation_index);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(canary_forward_start);
  RuntimeCheck(T.unwrap() > 0, "tracking ownership table must be non-empty");
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  if (batch_size == 0) return;
  LaunchKernel(batch_size, kThreads, device.unwrap())(
      validate_unbound_state_slots_kernel<IndexT>,
      static_cast<const int64_t*>(allocation_generations.data_ptr()),
      N_slots.unwrap(),
      static_cast<const int64_t*>(request_indices.data_ptr()),
      static_cast<IndexT*>(cache_indices.data_ptr()),
      static_cast<const int32_t*>(active_mask.data_ptr()),
      static_cast<const int32_t*>(expected_slots.data_ptr()),
      static_cast<const int64_t*>(expected_generations.data_ptr()),
      expected_slots.stride(0), T.unwrap(), N_requests.unwrap(),
      static_cast<int32_t*>(failure_status.data_ptr()),
      static_cast<const int32_t*>(enabled.data_ptr()),
      static_cast<const int32_t*>(canary_violation_index.data_ptr()),
      static_cast<const int32_t*>(canary_forward_start.data_ptr()),
      batch_size);
}

template <typename IndexT>
void seal_unbound_impl(
    tvm::ffi::TensorView sidecar,
    tvm::ffi::TensorView allocation_generations,
    tvm::ffi::TensorView request_indices,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView active_mask,
    tvm::ffi::TensorView expected_slots,
    tvm::ffi::TensorView expected_generations,
    tvm::ffi::TensorView failure_status,
    tvm::ffi::TensorView enabled,
    tvm::ffi::TensorView canary_violation_index,
    tvm::ffi::TensorView canary_forward_start,
    tvm::ffi::TensorView source_0,
    tvm::ffi::TensorView source_1,
    tvm::ffi::TensorView source_2,
    tvm::ffi::TensorView source_3,
    int64_t num_sources,
    int64_t domain_seed) {
  DLDevice device;
  auto p = build_common_params<IndexT>(
      sidecar, allocation_generations, request_indices, cache_indices,
      expected_slots, expected_generations, failure_status, enabled,
      canary_violation_index, canary_forward_start,
      source_0, source_1, source_2, source_3, num_sources, domain_seed, device);
  host::SymbolicSize B = {"batch_size"};
  host::SymbolicDevice matcher_device;
  matcher_device.set_options<kDLGPU>();
  host::TensorMatcher({B})
      .with_dtype<int32_t>()
      .with_device<kDLGPU>(matcher_device)
      .verify(active_mask);
  host::RuntimeCheck(B.unwrap() == request_indices.size(0), "active_mask batch size mismatch");
  p.require_sealed = static_cast<const int32_t*>(active_mask.data_ptr());
  const int32_t batch_size = static_cast<int32_t>(request_indices.size(0));
  host::LaunchKernel(batch_size, kThreads, device)(
      seal_unbound_state_slots_kernel<IndexT>, p, batch_size);
}

}  // namespace

#define STATE_PROTECTION_DEFINE_ENTRYPOINTS(SUFFIX, INDEX_T)                                                \
  void validate_unbound_state_slots_##SUFFIX(                                                              \
      tvm::ffi::TensorView allocation_generations, tvm::ffi::TensorView request_indices,                   \
      tvm::ffi::TensorView cache_indices, tvm::ffi::TensorView active_mask,                                \
      tvm::ffi::TensorView expected_slots, tvm::ffi::TensorView expected_generations,                      \
      tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,                                   \
      tvm::ffi::TensorView canary_violation_index, tvm::ffi::TensorView canary_forward_start) {            \
    validate_unbound_impl<INDEX_T>(                                                                        \
        allocation_generations, request_indices, cache_indices, active_mask,                               \
        expected_slots, expected_generations, failure_status, enabled,                                     \
        canary_violation_index, canary_forward_start);                                                     \
  }                                                                                                        \
  void validate_state_slots_##SUFFIX(                                                                      \
      tvm::ffi::TensorView sidecar, tvm::ffi::TensorView allocation_generations,                           \
      tvm::ffi::TensorView request_indices, tvm::ffi::TensorView cache_indices,                            \
      tvm::ffi::TensorView require_sealed, tvm::ffi::TensorView expected_slots,                            \
      tvm::ffi::TensorView expected_generations, tvm::ffi::TensorView failure_status,                      \
      tvm::ffi::TensorView enabled, tvm::ffi::TensorView canary_violation_index,                           \
      tvm::ffi::TensorView canary_forward_start, tvm::ffi::TensorView source_0,                            \
      tvm::ffi::TensorView source_1,                                                                        \
      tvm::ffi::TensorView source_2, tvm::ffi::TensorView source_3, int64_t num_sources,                   \
      int64_t domain_seed) {                                                                                \
    validate_impl<INDEX_T>(                                                                                 \
        sidecar, allocation_generations, request_indices, cache_indices, require_sealed,                   \
        expected_slots, expected_generations, failure_status, enabled,                                     \
        canary_violation_index, canary_forward_start, source_0, source_1,                                  \
        source_2, source_3, num_sources, domain_seed);                                                      \
  }                                                                                                        \
  void seal_state_slots_##SUFFIX(                                                                          \
      tvm::ffi::TensorView sidecar, tvm::ffi::TensorView allocation_generations,                           \
      tvm::ffi::TensorView request_indices, tvm::ffi::TensorView cache_indices,                            \
      tvm::ffi::TensorView expected_slots, tvm::ffi::TensorView expected_generations,                      \
      tvm::ffi::TensorView failure_status, tvm::ffi::TensorView enabled,                                   \
      tvm::ffi::TensorView canary_violation_index, tvm::ffi::TensorView canary_forward_start,             \
      tvm::ffi::TensorView source_0, tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2,          \
      tvm::ffi::TensorView source_3, int64_t num_sources, int64_t domain_seed) {                           \
    seal_impl<INDEX_T>(                                                                                     \
        sidecar, allocation_generations, request_indices, cache_indices, expected_slots,                   \
        expected_generations, failure_status, enabled, canary_violation_index,                             \
        canary_forward_start, source_0, source_1, source_2, source_3,                                      \
        num_sources, domain_seed);                                                                          \
  }                                                                                                        \
  void seal_unbound_state_slots_##SUFFIX(                                                                  \
      tvm::ffi::TensorView sidecar, tvm::ffi::TensorView allocation_generations,                           \
      tvm::ffi::TensorView request_indices, tvm::ffi::TensorView cache_indices,                            \
      tvm::ffi::TensorView active_mask, tvm::ffi::TensorView expected_slots,                               \
      tvm::ffi::TensorView expected_generations, tvm::ffi::TensorView failure_status,                      \
      tvm::ffi::TensorView enabled, tvm::ffi::TensorView canary_violation_index,                           \
      tvm::ffi::TensorView canary_forward_start, tvm::ffi::TensorView source_0,                            \
      tvm::ffi::TensorView source_1, tvm::ffi::TensorView source_2, tvm::ffi::TensorView source_3,          \
      int64_t num_sources, int64_t domain_seed) {                                                           \
    seal_unbound_impl<INDEX_T>(                                                                             \
        sidecar, allocation_generations, request_indices, cache_indices, active_mask,                      \
        expected_slots, expected_generations, failure_status, enabled,                                     \
        canary_violation_index, canary_forward_start, source_0, source_1,                                  \
        source_2, source_3, num_sources, domain_seed);                                                      \
  }

STATE_PROTECTION_DEFINE_ENTRYPOINTS(i32, int32_t)
STATE_PROTECTION_DEFINE_ENTRYPOINTS(i64, int64_t)

#undef STATE_PROTECTION_DEFINE_ENTRYPOINTS

}  // namespace state_protection
