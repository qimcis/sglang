#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>

#include "pytorch_extension_utils.h"

namespace {

constexpr int kThreads = 256;
constexpr int32_t kFailureOutOfRange = 1 << 0;
constexpr int32_t kFailureMissingDigest = 1 << 1;
constexpr int32_t kFailureDigestMismatch = 1 << 2;
constexpr int32_t kFailureMissingMapping = 1 << 3;
constexpr int32_t kFailureMappingMismatch = 1 << 4;
constexpr int32_t kFailureGenerationMismatch = 1 << 5;
constexpr int32_t kFailureInvalidRequest = 1 << 6;

__device__ __forceinline__ uint64_t avalanche(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

__device__ __forceinline__ uint64_t hash_word(uint64_t word, uint64_t byte_offset, uint64_t seed) {
  return avalanche(word ^ seed ^ 0x445356344b564932ULL ^ (byte_offset + 1) * 0x9e3779b97f4a7c15ULL);
}

__device__ uint64_t block_hash_row(const uint8_t* row, int64_t row_bytes, uint64_t seed) {
  uint64_t local = 0;
  for (int64_t offset = static_cast<int64_t>(threadIdx.x) * 8; offset < row_bytes;
       offset += static_cast<int64_t>(blockDim.x) * 8) {
    uint64_t word = 0;
    const int64_t remaining = row_bytes - offset;
    const int count = remaining >= 8 ? 8 : static_cast<int>(remaining);
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
      if (byte < count) {
        word |= static_cast<uint64_t>(row[offset + byte]) << (byte * 8);
      }
    }
    local ^= hash_word(word, static_cast<uint64_t>(offset), seed);
  }
  __shared__ uint64_t reduction[kThreads];
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) reduction[threadIdx.x] ^= reduction[threadIdx.x + width];
    __syncthreads();
  }
  uint64_t value = avalanche(reduction[0] ^ static_cast<uint64_t>(row_bytes) ^ seed);
  return value == 0 ? 0x9e3779b97f4a7c15ULL : value;
}

__global__ void dsv4_page_digests_kernel(
    const uint8_t* __restrict__ buffer,
    const int32_t* __restrict__ page_indices,
    int64_t* __restrict__ output,
    int64_t num_pages,
    int64_t capacity,
    int64_t row_bytes,
    uint64_t seed) {
  const int64_t i = blockIdx.x;
  if (i >= num_pages) return;
  const int32_t page = page_indices[i];
  if (page < 0 || page >= capacity) {
    if (threadIdx.x == 0) output[i] = 0;
    return;
  }
  const uint64_t digest = block_hash_row(buffer + static_cast<int64_t>(page) * row_bytes, row_bytes, seed);
  if (threadIdx.x == 0) output[i] = static_cast<int64_t>(digest);
}

template <typename SlotT>
__global__ void dsv4_bind_pages_kernel(
    const SlotT* __restrict__ slots,
    const int64_t* __restrict__ logical_pages,
    const int64_t* __restrict__ request_indices,
    const int64_t* __restrict__ generations,
    int32_t* __restrict__ expected_pages,
    int64_t* __restrict__ expected_generations,
    int32_t* __restrict__ expected_valid,
    SlotT* __restrict__ output,
    int32_t* __restrict__ failure_status,
    int64_t num_slots,
    int64_t num_requests,
    int64_t request_capacity,
    int64_t physical_capacity,
    int64_t logical_capacity,
    int64_t slot_page_size,
    SlotT invalid_value,
    bool install_only) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;
  const SlotT slot = slots[i];
  const int64_t logical = logical_pages[i];
  if (logical < 0) {
    output[i] = slot;
    return;
  }
  const int64_t row_width = num_slots / num_requests;
  const int64_t req = request_indices[i / row_width];
  const int64_t page = static_cast<int64_t>(slot) / slot_page_size;
  if (install_only) {
    if (req > 0 && req < request_capacity && page > 0 && page < physical_capacity && logical < logical_capacity &&
        generations[page] != 0) {
      const int64_t expected_idx = req * logical_capacity + logical;
      auto* valid_ptr = expected_valid + expected_idx;
      if (atomicCAS(valid_ptr, 0, 2) == 0) {
        expected_pages[expected_idx] = static_cast<int32_t>(page);
        expected_generations[expected_idx] = generations[page];
        __threadfence();
        atomicExch(valid_ptr, 1);
      }
    }
    return;
  }
  int32_t failure = 0;
  if (req <= 0 || req >= request_capacity) {
    failure = kFailureInvalidRequest;
  } else if (page <= 0 || page >= physical_capacity || logical >= logical_capacity) {
    failure = kFailureOutOfRange;
  } else if (generations[page] == 0) {
    failure = kFailureGenerationMismatch;
  } else {
    const int64_t expected_idx = req * logical_capacity + logical;
    if (expected_valid[expected_idx] != 1) {
      failure = kFailureMissingMapping;
    } else if (expected_pages[expected_idx] != page) {
      failure = kFailureMappingMismatch;
    } else if (expected_generations[expected_idx] != generations[page]) {
      failure = kFailureGenerationMismatch;
    }
  }
  output[i] = failure == 0 ? slot : invalid_value;
  if (failure != 0 && req > 0 && req < request_capacity) {
    atomicOr(failure_status + req, failure);
  }
}

template <typename SlotT>
__global__ void dsv4_collect_validation_pages_kernel(
    const SlotT* __restrict__ slots,
    const int64_t* __restrict__ logical_pages,
    int32_t* __restrict__ validation_state,
    int32_t* __restrict__ validation_failure,
    int32_t* __restrict__ validation_pages,
    int32_t* __restrict__ validation_count,
    int64_t num_slots,
    int64_t physical_capacity,
    int64_t slot_page_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_slots || slots[i] < 0 || logical_pages[i] < 0) return;
  const int64_t page = static_cast<int64_t>(slots[i]) / slot_page_size;
  if (page <= 0 || page >= physical_capacity) return;
  if (atomicExch(validation_state + page, 1) != 1) {
    validation_failure[page] = 0;
    const int32_t queue_index = atomicAdd(validation_count, 1);
    validation_pages[queue_index] = static_cast<int32_t>(page);
  }
}

__global__ void dsv4_hash_referenced_pages_kernel(
    const uint8_t* __restrict__ buffer,
    const int32_t* __restrict__ validation_pages,
    const int32_t* __restrict__ validation_count,
    const int64_t* __restrict__ digests,
    const uint8_t* __restrict__ component_valid,
    int32_t* __restrict__ validation_state,
    int32_t* __restrict__ validation_failure,
    int64_t row_bytes,
    uint64_t seed,
    bool allow_missing_digest) {
  const int64_t i = blockIdx.x;
  if (i >= *validation_count) return;
  const int32_t page = validation_pages[i];

  int32_t failure = 0;
  uint64_t actual = 0;
  if (component_valid[page] == 0) {
    failure = allow_missing_digest ? 0 : kFailureMissingDigest;
  } else {
    actual = block_hash_row(buffer + page * row_bytes, row_bytes, seed);
    if (threadIdx.x == 0 && actual != static_cast<uint64_t>(digests[page])) {
      failure = kFailureDigestMismatch;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    validation_failure[page] = failure;
    __threadfence();
    atomicExch(validation_state + page, 2);
  }
}

template <typename SlotT>
__global__ void dsv4_finalize_validation_kernel(
    const SlotT* __restrict__ slots,
    const int64_t* __restrict__ logical_pages,
    const int64_t* __restrict__ request_indices,
    const int32_t* __restrict__ validation_state,
    const int32_t* __restrict__ validation_failure,
    const int64_t* __restrict__ generations,
    const int32_t* __restrict__ expected_pages,
    const int64_t* __restrict__ expected_generations,
    const int32_t* __restrict__ expected_valid,
    SlotT* __restrict__ output,
    int32_t* __restrict__ failure_status,
    int64_t num_slots,
    int64_t num_requests,
    int64_t request_capacity,
    int64_t physical_capacity,
    int64_t logical_capacity,
    int64_t slot_page_size,
    SlotT invalid_value) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;
  const SlotT slot = slots[i];
  const int64_t logical = logical_pages[i];
  if (logical < 0) {
    output[i] = slot;
    return;
  }
  const int64_t row_width = num_slots / num_requests;
  const int64_t req = request_indices[i / row_width];
  const int64_t page = static_cast<int64_t>(slot) / slot_page_size;
  int32_t failure = 0;
  if (req <= 0 || req >= request_capacity) {
    failure = kFailureInvalidRequest;
  } else if (page <= 0 || page >= physical_capacity || logical >= logical_capacity) {
    failure = kFailureOutOfRange;
  } else if (generations[page] == 0) {
    failure = kFailureGenerationMismatch;
  } else {
    const int64_t expected_idx = req * logical_capacity + logical;
    if (expected_valid[expected_idx] != 1) {
      failure = kFailureMissingMapping;
    } else if (expected_pages[expected_idx] != page) {
      failure = kFailureMappingMismatch;
    } else if (expected_generations[expected_idx] != generations[page]) {
      failure = kFailureGenerationMismatch;
    } else if (validation_state[page] != 2) {
      failure = kFailureMissingDigest;
    } else {
      failure = validation_failure[page];
    }
  }
  output[i] = failure == 0 ? slot : invalid_value;
  if (failure != 0 && req > 0 && req < request_capacity) {
    atomicOr(failure_status + req, failure);
  }
}

template <typename SlotT>
__global__ void
dsv4_mark_dirty_kernel(const SlotT* slots, int32_t* dirty, int64_t count, int64_t capacity, int64_t slot_page_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count || slots[i] <= 0) return;
  const int64_t page = static_cast<int64_t>(slots[i]) / slot_page_size;
  if (page > 0 && page < capacity) atomicExch(dirty + page, 1);
}

template <typename SlotT>
__global__ void dsv4_refresh_slots_kernel(
    const uint8_t* buffer,
    int64_t* digests,
    uint8_t* valid,
    int32_t* dirty,
    const SlotT* slots,
    int64_t count,
    int64_t capacity,
    int64_t row_bytes,
    uint64_t seed,
    int64_t slot_page_size) {
  const int64_t i = blockIdx.x;
  if (i >= count || slots[i] <= 0) return;
  const int64_t page = static_cast<int64_t>(slots[i]) / slot_page_size;
  if (page <= 0 || page >= capacity) return;
  __shared__ bool elected;
  if (threadIdx.x == 0) elected = atomicCAS(dirty + page, 1, 2) == 1;
  __syncthreads();
  if (!elected) return;
  const uint64_t digest = block_hash_row(buffer + page * row_bytes, row_bytes, seed);
  if (threadIdx.x == 0) {
    digests[page] = static_cast<int64_t>(digest);
    __threadfence();
    valid[page] = 1;
    atomicExch(dirty + page, 0);
  }
}

void check_byte_matrix(const at::Tensor& buffer) {
  CHECK_CUDA(buffer);
  CHECK_CONTIGUOUS(buffer);
  TORCH_CHECK(buffer.scalar_type() == at::kByte, "buffer must be uint8");
  TORCH_CHECK(buffer.dim() == 2, "buffer must be a [capacity, bytes] matrix");
}

void check_slots(const at::Tensor& slots) {
  CHECK_CUDA(slots);
  CHECK_CONTIGUOUS(slots);
  TORCH_CHECK(slots.scalar_type() == at::kInt || slots.scalar_type() == at::kLong, "slots must be int32 or int64");
}

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on ", reference.device());
}

}  // namespace

at::Tensor dsv4_page_digests(const at::Tensor buffer, const at::Tensor page_indices, int64_t seed) {
  check_byte_matrix(buffer);
  CHECK_CUDA(page_indices);
  CHECK_CONTIGUOUS(page_indices);
  check_same_device(buffer, page_indices, "page_indices");
  TORCH_CHECK(
      page_indices.scalar_type() == at::kInt && page_indices.dim() == 1, "page_indices must be one-dimensional int32");
  auto output = at::empty(page_indices.sizes(), buffer.options().dtype(at::kLong));
  if (page_indices.numel() == 0) return output;
  dsv4_page_digests_kernel<<<page_indices.numel(), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      buffer.data_ptr<uint8_t>(),
      page_indices.data_ptr<int32_t>(),
      output.data_ptr<int64_t>(),
      page_indices.numel(),
      buffer.size(0),
      buffer.size(1),
      static_cast<uint64_t>(seed));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void dsv4_bind_pages(
    const at::Tensor slots,
    const at::Tensor logical_pages,
    const at::Tensor request_indices,
    const at::Tensor generations,
    at::Tensor expected_pages,
    at::Tensor expected_generations,
    at::Tensor expected_valid,
    at::Tensor output,
    at::Tensor failure_status,
    int64_t slot_page_size,
    int64_t invalid_value,
    bool install_missing) {
  check_slots(slots);
  CHECK_CUDA(logical_pages);
  CHECK_CUDA(request_indices);
  CHECK_CUDA(generations);
  CHECK_CUDA(expected_pages);
  CHECK_CUDA(expected_generations);
  CHECK_CUDA(expected_valid);
  CHECK_CUDA(output);
  CHECK_CUDA(failure_status);
  CHECK_CONTIGUOUS(logical_pages);
  CHECK_CONTIGUOUS(request_indices);
  CHECK_CONTIGUOUS(generations);
  CHECK_CONTIGUOUS(expected_pages);
  CHECK_CONTIGUOUS(expected_generations);
  CHECK_CONTIGUOUS(expected_valid);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(failure_status);
  check_same_device(slots, logical_pages, "logical_pages");
  check_same_device(slots, request_indices, "request_indices");
  check_same_device(slots, generations, "generations");
  check_same_device(slots, expected_pages, "expected_pages");
  check_same_device(slots, expected_generations, "expected_generations");
  check_same_device(slots, expected_valid, "expected_valid");
  check_same_device(slots, output, "output");
  check_same_device(slots, failure_status, "failure_status");
  TORCH_CHECK(
      logical_pages.scalar_type() == at::kLong && logical_pages.sizes() == slots.sizes(),
      "logical_pages must be int64 and match slots");
  TORCH_CHECK(
      request_indices.scalar_type() == at::kLong && request_indices.dim() == 1,
      "request_indices must be one-dimensional int64");
  TORCH_CHECK(
      request_indices.numel() > 0 && slots.numel() % request_indices.numel() == 0,
      "slots must contain a fixed-width row per request");
  TORCH_CHECK(generations.scalar_type() == at::kLong, "generations must be int64");
  TORCH_CHECK(
      expected_pages.scalar_type() == at::kInt && expected_pages.dim() == 2,
      "expected_pages must be two-dimensional int32");
  TORCH_CHECK(
      expected_generations.scalar_type() == at::kLong && expected_generations.sizes() == expected_pages.sizes(),
      "expected generation shape mismatch");
  TORCH_CHECK(
      expected_valid.scalar_type() == at::kInt && expected_valid.sizes() == expected_pages.sizes(),
      "expected valid shape mismatch");
  TORCH_CHECK(
      output.scalar_type() == slots.scalar_type() && output.sizes() == slots.sizes(), "output must match slots");
  TORCH_CHECK(
      failure_status.scalar_type() == at::kInt && failure_status.numel() == expected_pages.size(0),
      "failure status shape mismatch");
  TORCH_CHECK(generations.numel() > 0 && slot_page_size > 0, "invalid mapping geometry");
  if (slots.numel() == 0) return;
  const int blocks = (slots.numel() + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (slots.scalar_type() == at::kInt) {
    if (install_missing) {
      dsv4_bind_pages_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
          slots.data_ptr<int32_t>(),
          logical_pages.data_ptr<int64_t>(),
          request_indices.data_ptr<int64_t>(),
          generations.data_ptr<int64_t>(),
          expected_pages.data_ptr<int32_t>(),
          expected_generations.data_ptr<int64_t>(),
          expected_valid.data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          failure_status.data_ptr<int32_t>(),
          slots.numel(),
          request_indices.numel(),
          expected_pages.size(0),
          generations.numel(),
          expected_pages.size(1),
          slot_page_size,
          static_cast<int32_t>(invalid_value),
          true);
    }
    dsv4_bind_pages_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int32_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        generations.data_ptr<int64_t>(),
        expected_pages.data_ptr<int32_t>(),
        expected_generations.data_ptr<int64_t>(),
        expected_valid.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_pages.size(0),
        generations.numel(),
        expected_pages.size(1),
        slot_page_size,
        static_cast<int32_t>(invalid_value),
        false);
  } else {
    if (install_missing) {
      dsv4_bind_pages_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
          slots.data_ptr<int64_t>(),
          logical_pages.data_ptr<int64_t>(),
          request_indices.data_ptr<int64_t>(),
          generations.data_ptr<int64_t>(),
          expected_pages.data_ptr<int32_t>(),
          expected_generations.data_ptr<int64_t>(),
          expected_valid.data_ptr<int32_t>(),
          output.data_ptr<int64_t>(),
          failure_status.data_ptr<int32_t>(),
          slots.numel(),
          request_indices.numel(),
          expected_pages.size(0),
          generations.numel(),
          expected_pages.size(1),
          slot_page_size,
          invalid_value,
          true);
    }
    dsv4_bind_pages_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int64_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        generations.data_ptr<int64_t>(),
        expected_pages.data_ptr<int32_t>(),
        expected_generations.data_ptr<int64_t>(),
        expected_valid.data_ptr<int32_t>(),
        output.data_ptr<int64_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_pages.size(0),
        generations.numel(),
        expected_pages.size(1),
        slot_page_size,
        invalid_value,
        false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_validate_pages(
    const at::Tensor buffer,
    const at::Tensor slots,
    const at::Tensor logical_pages,
    const at::Tensor request_indices,
    const at::Tensor digests,
    const at::Tensor component_valid,
    at::Tensor validation_state,
    at::Tensor validation_failure,
    at::Tensor validation_pages,
    at::Tensor validation_count,
    const at::Tensor generations,
    const at::Tensor expected_pages,
    const at::Tensor expected_generations,
    const at::Tensor expected_valid,
    at::Tensor output,
    at::Tensor failure_status,
    int64_t seed,
    int64_t slot_page_size,
    int64_t invalid_value,
    bool allow_missing_digest) {
  check_byte_matrix(buffer);
  check_slots(slots);
  CHECK_CUDA(logical_pages);
  CHECK_CUDA(request_indices);
  CHECK_CUDA(digests);
  CHECK_CUDA(component_valid);
  CHECK_CUDA(validation_state);
  CHECK_CUDA(validation_failure);
  CHECK_CUDA(validation_pages);
  CHECK_CUDA(validation_count);
  CHECK_CUDA(generations);
  CHECK_CUDA(expected_pages);
  CHECK_CUDA(expected_generations);
  CHECK_CUDA(expected_valid);
  CHECK_CUDA(output);
  CHECK_CUDA(failure_status);
  CHECK_CONTIGUOUS(logical_pages);
  CHECK_CONTIGUOUS(request_indices);
  CHECK_CONTIGUOUS(digests);
  CHECK_CONTIGUOUS(component_valid);
  CHECK_CONTIGUOUS(validation_state);
  CHECK_CONTIGUOUS(validation_failure);
  CHECK_CONTIGUOUS(validation_pages);
  CHECK_CONTIGUOUS(validation_count);
  CHECK_CONTIGUOUS(generations);
  CHECK_CONTIGUOUS(expected_pages);
  CHECK_CONTIGUOUS(expected_generations);
  CHECK_CONTIGUOUS(expected_valid);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(failure_status);
  check_same_device(buffer, slots, "slots");
  check_same_device(buffer, logical_pages, "logical_pages");
  check_same_device(buffer, request_indices, "request_indices");
  check_same_device(buffer, digests, "digests");
  check_same_device(buffer, component_valid, "component_valid");
  check_same_device(buffer, validation_state, "validation_state");
  check_same_device(buffer, validation_failure, "validation_failure");
  check_same_device(buffer, validation_pages, "validation_pages");
  check_same_device(buffer, validation_count, "validation_count");
  check_same_device(buffer, generations, "generations");
  check_same_device(buffer, expected_pages, "expected_pages");
  check_same_device(buffer, expected_generations, "expected_generations");
  check_same_device(buffer, expected_valid, "expected_valid");
  check_same_device(buffer, output, "output");
  check_same_device(buffer, failure_status, "failure_status");
  TORCH_CHECK(buffer.size(0) == digests.numel() && digests.scalar_type() == at::kLong, "digest capacity mismatch");
  TORCH_CHECK(
      component_valid.numel() == buffer.size(0) && component_valid.scalar_type() == at::kByte,
      "component valid capacity mismatch");
  TORCH_CHECK(
      validation_state.numel() == buffer.size(0) && validation_state.scalar_type() == at::kInt &&
          validation_failure.numel() == buffer.size(0) && validation_failure.scalar_type() == at::kInt,
      "validation scratch capacity mismatch");
  TORCH_CHECK(
      validation_pages.numel() == buffer.size(0) && validation_pages.scalar_type() == at::kInt &&
          validation_count.numel() == 1 && validation_count.scalar_type() == at::kInt,
      "validation queue shape mismatch");
  TORCH_CHECK(
      generations.scalar_type() == at::kLong && generations.dim() == 1 && generations.numel() >= buffer.size(0),
      "generation capacity mismatch");
  TORCH_CHECK(
      logical_pages.scalar_type() == at::kLong && logical_pages.sizes() == slots.sizes(),
      "logical_pages must match slots");
  TORCH_CHECK(
      request_indices.scalar_type() == at::kLong && request_indices.dim() == 1 && request_indices.numel() > 0 &&
          slots.numel() % request_indices.numel() == 0,
      "invalid request row geometry");
  TORCH_CHECK(expected_pages.scalar_type() == at::kInt && expected_pages.dim() == 2, "invalid expected pages");
  TORCH_CHECK(
      expected_generations.sizes() == expected_pages.sizes() && expected_generations.scalar_type() == at::kLong &&
          expected_valid.sizes() == expected_pages.sizes() && expected_valid.scalar_type() == at::kInt,
      "expected sidecar shape mismatch");
  TORCH_CHECK(
      output.scalar_type() == slots.scalar_type() && output.sizes() == slots.sizes(), "output must match slots");
  TORCH_CHECK(
      failure_status.scalar_type() == at::kInt && failure_status.dim() == 1 &&
          failure_status.numel() == expected_pages.size(0),
      "failure status shape mismatch");
  TORCH_CHECK(slot_page_size > 0, "slot_page_size must be positive");
  if (slots.numel() == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (slots.numel() + kThreads - 1) / kThreads;
  const int hash_blocks = std::min<int64_t>(slots.numel(), buffer.size(0));
  C10_CUDA_CHECK(cudaMemsetAsync(validation_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  if (slots.scalar_type() == at::kInt) {
    dsv4_collect_validation_pages_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int32_t>(),
        logical_pages.data_ptr<int64_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        validation_pages.data_ptr<int32_t>(),
        validation_count.data_ptr<int32_t>(),
        slots.numel(),
        buffer.size(0),
        slot_page_size);
    dsv4_hash_referenced_pages_kernel<<<hash_blocks, kThreads, 0, stream>>>(
        buffer.data_ptr<uint8_t>(),
        validation_pages.data_ptr<int32_t>(),
        validation_count.data_ptr<int32_t>(),
        digests.data_ptr<int64_t>(),
        component_valid.data_ptr<uint8_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        buffer.size(1),
        static_cast<uint64_t>(seed),
        allow_missing_digest);
    dsv4_finalize_validation_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int32_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        generations.data_ptr<int64_t>(),
        expected_pages.data_ptr<int32_t>(),
        expected_generations.data_ptr<int64_t>(),
        expected_valid.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_pages.size(0),
        buffer.size(0),
        expected_pages.size(1),
        slot_page_size,
        static_cast<int32_t>(invalid_value));
  } else {
    dsv4_collect_validation_pages_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int64_t>(),
        logical_pages.data_ptr<int64_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        validation_pages.data_ptr<int32_t>(),
        validation_count.data_ptr<int32_t>(),
        slots.numel(),
        buffer.size(0),
        slot_page_size);
    dsv4_hash_referenced_pages_kernel<<<hash_blocks, kThreads, 0, stream>>>(
        buffer.data_ptr<uint8_t>(),
        validation_pages.data_ptr<int32_t>(),
        validation_count.data_ptr<int32_t>(),
        digests.data_ptr<int64_t>(),
        component_valid.data_ptr<uint8_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        buffer.size(1),
        static_cast<uint64_t>(seed),
        allow_missing_digest);
    dsv4_finalize_validation_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int64_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        validation_state.data_ptr<int32_t>(),
        validation_failure.data_ptr<int32_t>(),
        generations.data_ptr<int64_t>(),
        expected_pages.data_ptr<int32_t>(),
        expected_generations.data_ptr<int64_t>(),
        expected_valid.data_ptr<int32_t>(),
        output.data_ptr<int64_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_pages.size(0),
        buffer.size(0),
        expected_pages.size(1),
        slot_page_size,
        invalid_value);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_refresh_slots(
    const at::Tensor buffer,
    at::Tensor digests,
    at::Tensor valid,
    at::Tensor dirty,
    const at::Tensor slots,
    int64_t seed,
    int64_t slot_page_size) {
  check_byte_matrix(buffer);
  check_slots(slots);
  CHECK_CUDA(digests);
  CHECK_CUDA(valid);
  CHECK_CUDA(dirty);
  CHECK_CONTIGUOUS(digests);
  CHECK_CONTIGUOUS(valid);
  CHECK_CONTIGUOUS(dirty);
  check_same_device(buffer, digests, "digests");
  check_same_device(buffer, valid, "valid");
  check_same_device(buffer, dirty, "dirty");
  check_same_device(buffer, slots, "slots");
  TORCH_CHECK(
      digests.scalar_type() == at::kLong && valid.scalar_type() == at::kByte && dirty.scalar_type() == at::kInt &&
          digests.numel() == buffer.size(0) && valid.numel() == buffer.size(0) && dirty.numel() == buffer.size(0),
      "invalid slot refresh sidecars");
  TORCH_CHECK(slot_page_size > 0, "slot_page_size must be positive");
  if (slots.numel() == 0) return;
  const int blocks = (slots.numel() + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (slots.scalar_type() == at::kInt) {
    dsv4_mark_dirty_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int32_t>(), dirty.data_ptr<int32_t>(), slots.numel(), dirty.numel(), slot_page_size);
    dsv4_refresh_slots_kernel<int32_t><<<slots.numel(), kThreads, 0, stream>>>(
        buffer.data_ptr<uint8_t>(),
        digests.data_ptr<int64_t>(),
        valid.data_ptr<uint8_t>(),
        dirty.data_ptr<int32_t>(),
        slots.data_ptr<int32_t>(),
        slots.numel(),
        buffer.size(0),
        buffer.size(1),
        static_cast<uint64_t>(seed),
        slot_page_size);
  } else {
    dsv4_mark_dirty_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int64_t>(), dirty.data_ptr<int32_t>(), slots.numel(), dirty.numel(), slot_page_size);
    dsv4_refresh_slots_kernel<int64_t><<<slots.numel(), kThreads, 0, stream>>>(
        buffer.data_ptr<uint8_t>(),
        digests.data_ptr<int64_t>(),
        valid.data_ptr<uint8_t>(),
        dirty.data_ptr<int32_t>(),
        slots.data_ptr<int64_t>(),
        slots.numel(),
        buffer.size(0),
        buffer.size(1),
        static_cast<uint64_t>(seed),
        slot_page_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
