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

__device__ __forceinline__ uint64_t mapping_tag(
    int64_t request,
    int64_t request_epoch,
    int64_t logical_page,
    int64_t physical_page,
    int64_t generation,
    uint64_t seed) {
  uint64_t value = seed ^ 0x445356344d415032ULL;
  value ^= static_cast<uint64_t>(request) * 0x9e3779b97f4a7c15ULL;
  value ^= static_cast<uint64_t>(request_epoch) * 0xbf58476d1ce4e5b9ULL;
  value ^= static_cast<uint64_t>(logical_page) * 0x94d049bb133111ebULL;
  value ^= static_cast<uint64_t>(physical_page) * 0xd6e8feb86659fd93ULL;
  value ^= static_cast<uint64_t>(generation) * 0xa0761d6478bd642fULL;
  const uint64_t tag = avalanche(value);
  return tag == 0 ? 1 : tag;
}

__device__ uint64_t block_hash_row(const uint8_t* row, int64_t row_bytes, uint64_t seed) {
  uint64_t local = 0;
  int64_t tail_start = 0;
  if ((reinterpret_cast<uintptr_t>(row) & 15) == 0) {
    const int64_t vector_bytes = row_bytes & ~static_cast<int64_t>(15);
    for (int64_t offset = static_cast<int64_t>(threadIdx.x) * 16; offset < vector_bytes;
         offset += static_cast<int64_t>(blockDim.x) * 16) {
      const uint4 value = *reinterpret_cast<const uint4*>(row + offset);
      const uint64_t word0 = static_cast<uint64_t>(value.x) | (static_cast<uint64_t>(value.y) << 32);
      const uint64_t word1 = static_cast<uint64_t>(value.z) | (static_cast<uint64_t>(value.w) << 32);
      local ^= hash_word(word0, static_cast<uint64_t>(offset), seed);
      local ^= hash_word(word1, static_cast<uint64_t>(offset + 8), seed);
    }
    tail_start = vector_bytes;
  }
  for (int64_t offset = tail_start + static_cast<int64_t>(threadIdx.x) * 8; offset < row_bytes;
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
  constexpr int kWarpSize = 32;
#pragma unroll
  for (int width = kWarpSize / 2; width > 0; width >>= 1) {
    local ^= __shfl_down_sync(0xffffffff, local, width);
  }
  __shared__ uint64_t warp_reduction[kThreads / kWarpSize];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) warp_reduction[warp] = local;
  __syncthreads();
  if (warp == 0) {
    local = lane < (kThreads / kWarpSize) ? warp_reduction[lane] : 0;
#pragma unroll
    for (int width = kWarpSize / 2; width > 0; width >>= 1) {
      local ^= __shfl_down_sync(0xffffffff, local, width);
    }
    if (lane == 0) warp_reduction[0] = local;
  }
  __syncthreads();
  uint64_t value = avalanche(warp_reduction[0] ^ static_cast<uint64_t>(row_bytes) ^ seed);
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

__global__ void dsv4_batched_page_digests_kernel(
    const int64_t* __restrict__ descriptors,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ group_offsets,
    int64_t* __restrict__ output,
    int64_t num_descriptors,
    int64_t output_stride,
    int64_t num_groups) {
  const int64_t descriptor_id = blockIdx.y;
  const int64_t local_page = blockIdx.x;
  if (descriptor_id >= num_descriptors) return;
  const int64_t* descriptor = descriptors + descriptor_id * 5;
  const int64_t group = descriptor[4];
  if (group < 0 || group >= num_groups) return;
  const int32_t begin = group_offsets[group];
  const int32_t end = group_offsets[group + 1];
  if (begin < 0 || end < begin || end > output_stride) return;
  if (local_page >= end - begin) return;
  const int64_t flat_index = static_cast<int64_t>(begin) + local_page;
  if (flat_index < 0 || flat_index >= output_stride) return;
  const int32_t page = page_indices[flat_index];
  const int64_t capacity = descriptor[1];
  if (page < 0 || page >= capacity) return;
  const auto* buffer = reinterpret_cast<const uint8_t*>(descriptor[0]);
  const int64_t row_bytes = descriptor[2];
  const uint64_t seed = static_cast<uint64_t>(descriptor[3]);
  const uint64_t digest = block_hash_row(buffer + static_cast<int64_t>(page) * row_bytes, row_bytes, seed);
  if (threadIdx.x == 0) output[descriptor_id * output_stride + local_page] = static_cast<int64_t>(digest);
}

template <typename SlotT>
__global__ void dsv4_bind_pages_kernel(
    const SlotT* __restrict__ slots,
    const int64_t* __restrict__ logical_pages,
    const int64_t* __restrict__ request_indices,
    const int64_t* __restrict__ generations,
    const int64_t* __restrict__ request_epochs,
    unsigned long long* __restrict__ expected_tags,
    SlotT* __restrict__ output,
    int32_t* __restrict__ failure_status,
    int64_t num_slots,
    int64_t num_requests,
    int64_t request_capacity,
    int64_t physical_capacity,
    int64_t logical_capacity,
    uint64_t mapping_seed,
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
        generations[page] != 0 && request_epochs[req] != 0) {
      const int64_t expected_idx = req * logical_capacity + logical;
      const uint64_t tag = mapping_tag(req, request_epochs[req], logical, page, generations[page], mapping_seed);
      atomicCAS(expected_tags + expected_idx, 0ULL, static_cast<unsigned long long>(tag));
    }
    output[i] = slot;
    return;
  }
  int32_t failure = 0;
  if (req <= 0 || req >= request_capacity) {
    failure = kFailureInvalidRequest;
  } else if (page <= 0 || page >= physical_capacity || logical >= logical_capacity) {
    failure = kFailureOutOfRange;
  } else if (generations[page] == 0 || request_epochs[req] == 0) {
    failure = kFailureGenerationMismatch;
  } else {
    const int64_t expected_idx = req * logical_capacity + logical;
    const uint64_t expected = expected_tags[expected_idx];
    const uint64_t actual = mapping_tag(req, request_epochs[req], logical, page, generations[page], mapping_seed);
    if (expected == 0) {
      failure = kFailureMissingMapping;
    } else if (expected != actual) {
      failure = kFailureMappingMismatch;
    }
  }
  output[i] = failure == 0 ? slot : invalid_value;
  if (failure != 0 && req > 0 && req < request_capacity) {
    atomicOr(failure_status + req, failure);
  }
}

template <typename SlotT>
__device__ __forceinline__ int32_t validate_mapping_tag(
    SlotT slot,
    int64_t logical,
    int64_t request,
    const int64_t* generations,
    const int64_t* request_epochs,
    const unsigned long long* expected_tags,
    int64_t request_capacity,
    int64_t physical_capacity,
    int64_t logical_capacity,
    uint64_t seed,
    int64_t slot_page_size) {
  if (logical < 0) return 0;
  const int64_t page = static_cast<int64_t>(slot) / slot_page_size;
  if (request <= 0 || request >= request_capacity) return kFailureInvalidRequest;
  if (page <= 0 || page >= physical_capacity || logical >= logical_capacity) return kFailureOutOfRange;
  if (generations[page] == 0 || request_epochs[request] == 0) return kFailureGenerationMismatch;
  const int64_t expected_idx = request * logical_capacity + logical;
  const uint64_t expected = expected_tags[expected_idx];
  if (expected == 0) return kFailureMissingMapping;
  const uint64_t actual = mapping_tag(request, request_epochs[request], logical, page, generations[page], seed);
  return expected == actual ? 0 : kFailureMappingMismatch;
}

template <typename OutSlotT>
__global__ void dsv4_validate_core_mappings_kernel(
    int32_t* full_slots,
    const int64_t* full_logical,
    OutSlotT* out_slots,
    const int64_t* out_logical,
    int32_t* swa_slots,
    const int64_t* swa_logical,
    const int64_t* request_indices,
    const int64_t* full_generations,
    const int64_t* swa_generations,
    const int64_t* request_epochs,
    const unsigned long long* full_tags,
    const unsigned long long* swa_tags,
    int32_t* failure_status,
    int64_t full_count,
    int64_t out_count,
    int64_t swa_count,
    int64_t num_requests,
    int64_t full_width,
    int64_t swa_width,
    int64_t request_capacity,
    int64_t full_physical_capacity,
    int64_t swa_physical_capacity,
    int64_t full_logical_capacity,
    int64_t swa_logical_capacity,
    uint64_t full_seed,
    uint64_t swa_seed,
    int64_t full_page_size,
    int64_t swa_page_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = full_count + out_count + swa_count;
  if (i >= total) return;

  int64_t request_row;
  int64_t request;
  int32_t failure;
  if (i < full_count) {
    request_row = i / full_width;
    request = request_indices[request_row];
    const int32_t slot = full_slots[i];
    failure = validate_mapping_tag(
        slot,
        full_logical[i],
        request,
        full_generations,
        request_epochs,
        full_tags,
        request_capacity,
        full_physical_capacity,
        full_logical_capacity,
        full_seed,
        1);
    if (failure != 0) full_slots[i] = 0;
  } else if (i < full_count + out_count) {
    const int64_t j = i - full_count;
    request_row = j * num_requests / out_count;
    request = request_indices[request_row];
    const OutSlotT slot = out_slots[j];
    failure = validate_mapping_tag(
        slot,
        out_logical[j],
        request,
        full_generations,
        request_epochs,
        full_tags,
        request_capacity,
        full_physical_capacity,
        full_logical_capacity,
        full_seed,
        full_page_size);
    if (failure != 0) out_slots[j] = 0;
  } else {
    const int64_t j = i - full_count - out_count;
    request_row = j / swa_width;
    request = request_indices[request_row];
    const int32_t slot = swa_slots[j];
    failure = validate_mapping_tag(
        slot,
        swa_logical[j],
        request,
        swa_generations,
        request_epochs,
        swa_tags,
        request_capacity,
        swa_physical_capacity,
        swa_logical_capacity,
        swa_seed,
        swa_page_size);
    if (failure != 0) swa_slots[j] = 0;
  }
  if (failure != 0 && request > 0 && request < request_capacity) {
    atomicOr(failure_status + request, failure);
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
    const int64_t* __restrict__ request_epochs,
    const unsigned long long* __restrict__ expected_tags,
    SlotT* __restrict__ output,
    int32_t* __restrict__ failure_status,
    int64_t num_slots,
    int64_t num_requests,
    int64_t request_capacity,
    int64_t physical_capacity,
    int64_t logical_capacity,
    uint64_t mapping_seed,
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
  } else if (generations[page] == 0 || request_epochs[req] == 0) {
    failure = kFailureGenerationMismatch;
  } else {
    const int64_t expected_idx = req * logical_capacity + logical;
    const uint64_t expected = expected_tags[expected_idx];
    const uint64_t actual = mapping_tag(req, request_epochs[req], logical, page, generations[page], mapping_seed);
    if (expected == 0) {
      failure = kFailureMissingMapping;
    } else if (expected != actual) {
      failure = kFailureMappingMismatch;
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

at::Tensor
dsv4_batched_page_digests(const at::Tensor descriptors, const at::Tensor page_indices, const at::Tensor group_offsets) {
  CHECK_CUDA(descriptors);
  CHECK_CUDA(page_indices);
  CHECK_CUDA(group_offsets);
  CHECK_CONTIGUOUS(descriptors);
  CHECK_CONTIGUOUS(page_indices);
  CHECK_CONTIGUOUS(group_offsets);
  check_same_device(descriptors, page_indices, "page_indices");
  check_same_device(descriptors, group_offsets, "group_offsets");
  TORCH_CHECK(
      descriptors.scalar_type() == at::kLong && descriptors.dim() == 2 && descriptors.size(1) == 5,
      "descriptors must be [N, 5] int64");
  TORCH_CHECK(
      page_indices.scalar_type() == at::kInt && page_indices.dim() == 1, "page_indices must be one-dimensional int32");
  TORCH_CHECK(
      group_offsets.scalar_type() == at::kInt && group_offsets.dim() == 1 && group_offsets.numel() >= 2,
      "group_offsets must be one-dimensional int32");
  auto output = at::zeros({descriptors.size(0), page_indices.numel()}, descriptors.options().dtype(at::kLong));
  if (descriptors.size(0) == 0 || page_indices.numel() == 0) return output;
  const dim3 grid(page_indices.numel(), descriptors.size(0));
  dsv4_batched_page_digests_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      descriptors.data_ptr<int64_t>(),
      page_indices.data_ptr<int32_t>(),
      group_offsets.data_ptr<int32_t>(),
      output.data_ptr<int64_t>(),
      descriptors.size(0),
      page_indices.numel(),
      group_offsets.numel() - 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void dsv4_bind_pages(
    const at::Tensor slots,
    const at::Tensor logical_pages,
    const at::Tensor request_indices,
    const at::Tensor generations,
    const at::Tensor request_epochs,
    at::Tensor expected_tags,
    at::Tensor output,
    at::Tensor failure_status,
    int64_t mapping_seed,
    int64_t slot_page_size,
    int64_t invalid_value,
    bool install_missing) {
  check_slots(slots);
  CHECK_CUDA(logical_pages);
  CHECK_CUDA(request_indices);
  CHECK_CUDA(generations);
  CHECK_CUDA(request_epochs);
  CHECK_CUDA(expected_tags);
  CHECK_CUDA(output);
  CHECK_CUDA(failure_status);
  CHECK_CONTIGUOUS(logical_pages);
  CHECK_CONTIGUOUS(request_indices);
  CHECK_CONTIGUOUS(generations);
  CHECK_CONTIGUOUS(request_epochs);
  CHECK_CONTIGUOUS(expected_tags);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(failure_status);
  check_same_device(slots, logical_pages, "logical_pages");
  check_same_device(slots, request_indices, "request_indices");
  check_same_device(slots, generations, "generations");
  check_same_device(slots, request_epochs, "request_epochs");
  check_same_device(slots, expected_tags, "expected_tags");
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
      request_epochs.scalar_type() == at::kLong && request_epochs.dim() == 1,
      "request_epochs must be one-dimensional int64");
  TORCH_CHECK(
      expected_tags.scalar_type() == at::kLong && expected_tags.dim() == 2 &&
          expected_tags.size(0) == request_epochs.numel(),
      "expected_tags must be [request_capacity, logical_capacity] int64");
  TORCH_CHECK(
      output.scalar_type() == slots.scalar_type() && output.sizes() == slots.sizes(), "output must match slots");
  TORCH_CHECK(
      failure_status.scalar_type() == at::kInt && failure_status.numel() == expected_tags.size(0),
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
          request_epochs.data_ptr<int64_t>(),
          reinterpret_cast<unsigned long long*>(expected_tags.data_ptr<int64_t>()),
          output.data_ptr<int32_t>(),
          failure_status.data_ptr<int32_t>(),
          slots.numel(),
          request_indices.numel(),
          expected_tags.size(0),
          generations.numel(),
          expected_tags.size(1),
          static_cast<uint64_t>(mapping_seed),
          slot_page_size,
          static_cast<int32_t>(invalid_value),
          true);
    }
    dsv4_bind_pages_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int32_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        generations.data_ptr<int64_t>(),
        request_epochs.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(expected_tags.data_ptr<int64_t>()),
        output.data_ptr<int32_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_tags.size(0),
        generations.numel(),
        expected_tags.size(1),
        static_cast<uint64_t>(mapping_seed),
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
          request_epochs.data_ptr<int64_t>(),
          reinterpret_cast<unsigned long long*>(expected_tags.data_ptr<int64_t>()),
          output.data_ptr<int64_t>(),
          failure_status.data_ptr<int32_t>(),
          slots.numel(),
          request_indices.numel(),
          expected_tags.size(0),
          generations.numel(),
          expected_tags.size(1),
          static_cast<uint64_t>(mapping_seed),
          slot_page_size,
          invalid_value,
          true);
    }
    dsv4_bind_pages_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        slots.data_ptr<int64_t>(),
        logical_pages.data_ptr<int64_t>(),
        request_indices.data_ptr<int64_t>(),
        generations.data_ptr<int64_t>(),
        request_epochs.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(expected_tags.data_ptr<int64_t>()),
        output.data_ptr<int64_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_tags.size(0),
        generations.numel(),
        expected_tags.size(1),
        static_cast<uint64_t>(mapping_seed),
        slot_page_size,
        invalid_value,
        false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsv4_validate_core_mappings(
    at::Tensor full_slots,
    const at::Tensor full_logical,
    at::Tensor out_slots,
    const at::Tensor out_logical,
    at::Tensor swa_slots,
    const at::Tensor swa_logical,
    const at::Tensor request_indices,
    const at::Tensor full_generations,
    const at::Tensor swa_generations,
    const at::Tensor request_epochs,
    const at::Tensor full_tags,
    const at::Tensor swa_tags,
    at::Tensor failure_status,
    int64_t full_seed,
    int64_t swa_seed,
    int64_t full_page_size,
    int64_t swa_page_size) {
  check_slots(full_slots);
  check_slots(out_slots);
  check_slots(swa_slots);
  TORCH_CHECK(full_slots.scalar_type() == at::kInt, "full_slots must be int32");
  TORCH_CHECK(swa_slots.scalar_type() == at::kInt, "swa_slots must be int32");
  TORCH_CHECK(full_slots.dim() == 2 && swa_slots.dim() == 2, "core page tables must be two-dimensional");
  TORCH_CHECK(
      full_logical.scalar_type() == at::kLong && full_logical.sizes() == full_slots.sizes(),
      "full logical shape mismatch");
  TORCH_CHECK(
      swa_logical.scalar_type() == at::kLong && swa_logical.sizes() == swa_slots.sizes(), "SWA logical shape mismatch");
  TORCH_CHECK(
      out_logical.scalar_type() == at::kLong && out_logical.sizes() == out_slots.sizes(),
      "output logical shape mismatch");
  TORCH_CHECK(
      request_indices.scalar_type() == at::kLong && request_indices.dim() == 1 && request_indices.numel() > 0,
      "request_indices must be non-empty one-dimensional int64");
  TORCH_CHECK(
      full_slots.size(0) == request_indices.numel() && swa_slots.size(0) == request_indices.numel() &&
          out_slots.numel() % request_indices.numel() == 0,
      "core mapping request geometry mismatch");
  TORCH_CHECK(
      full_generations.scalar_type() == at::kLong && full_generations.dim() == 1 &&
          swa_generations.scalar_type() == at::kLong && swa_generations.dim() == 1 &&
          request_epochs.scalar_type() == at::kLong && request_epochs.dim() == 1,
      "core generation sidecar mismatch");
  TORCH_CHECK(
      full_tags.scalar_type() == at::kLong && full_tags.dim() == 2 && swa_tags.scalar_type() == at::kLong &&
          swa_tags.dim() == 2 && full_tags.size(0) == request_epochs.numel() &&
          swa_tags.size(0) == request_epochs.numel(),
      "core mapping tag sidecar mismatch");
  TORCH_CHECK(
      failure_status.scalar_type() == at::kInt && failure_status.numel() == request_epochs.numel(),
      "core failure status mismatch");
  TORCH_CHECK(full_page_size > 0 && swa_page_size > 0, "core page sizes must be positive");

  const at::Tensor tensors[] = {
      full_logical,
      out_slots,
      out_logical,
      swa_slots,
      swa_logical,
      request_indices,
      full_generations,
      swa_generations,
      request_epochs,
      full_tags,
      swa_tags,
      failure_status};
  for (const auto& tensor : tensors) {
    CHECK_CUDA(tensor);
    CHECK_CONTIGUOUS(tensor);
    check_same_device(full_slots, tensor, "core mapping tensor");
  }

  const int64_t total = full_slots.numel() + out_slots.numel() + swa_slots.numel();
  if (total == 0) return;
  const int blocks = (total + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_CORE_MAPPING(OutT)                                                 \
  dsv4_validate_core_mappings_kernel<OutT><<<blocks, kThreads, 0, stream>>>(      \
      full_slots.data_ptr<int32_t>(),                                             \
      full_logical.data_ptr<int64_t>(),                                           \
      out_slots.data_ptr<OutT>(),                                                 \
      out_logical.data_ptr<int64_t>(),                                            \
      swa_slots.data_ptr<int32_t>(),                                              \
      swa_logical.data_ptr<int64_t>(),                                            \
      request_indices.data_ptr<int64_t>(),                                        \
      full_generations.data_ptr<int64_t>(),                                       \
      swa_generations.data_ptr<int64_t>(),                                        \
      request_epochs.data_ptr<int64_t>(),                                         \
      reinterpret_cast<const unsigned long long*>(full_tags.data_ptr<int64_t>()), \
      reinterpret_cast<const unsigned long long*>(swa_tags.data_ptr<int64_t>()),  \
      failure_status.data_ptr<int32_t>(),                                         \
      full_slots.numel(),                                                         \
      out_slots.numel(),                                                          \
      swa_slots.numel(),                                                          \
      request_indices.numel(),                                                    \
      full_slots.size(1),                                                         \
      swa_slots.size(1),                                                          \
      request_epochs.numel(),                                                     \
      full_generations.numel(),                                                   \
      swa_generations.numel(),                                                    \
      full_tags.size(1),                                                          \
      swa_tags.size(1),                                                           \
      static_cast<uint64_t>(full_seed),                                           \
      static_cast<uint64_t>(swa_seed),                                            \
      full_page_size,                                                             \
      swa_page_size)
  if (out_slots.scalar_type() == at::kInt) {
    LAUNCH_CORE_MAPPING(int32_t);
  } else {
    TORCH_CHECK(out_slots.scalar_type() == at::kLong, "out_slots must be int32 or int64");
    LAUNCH_CORE_MAPPING(int64_t);
  }
#undef LAUNCH_CORE_MAPPING
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
    const at::Tensor request_epochs,
    const at::Tensor expected_tags,
    at::Tensor output,
    at::Tensor failure_status,
    int64_t seed,
    int64_t mapping_seed,
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
  CHECK_CUDA(request_epochs);
  CHECK_CUDA(expected_tags);
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
  CHECK_CONTIGUOUS(request_epochs);
  CHECK_CONTIGUOUS(expected_tags);
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
  check_same_device(buffer, request_epochs, "request_epochs");
  check_same_device(buffer, expected_tags, "expected_tags");
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
  TORCH_CHECK(request_epochs.scalar_type() == at::kLong && request_epochs.dim() == 1, "invalid request epochs");
  TORCH_CHECK(
      expected_tags.scalar_type() == at::kLong && expected_tags.dim() == 2 &&
          expected_tags.size(0) == request_epochs.numel(),
      "expected tag shape mismatch");
  TORCH_CHECK(
      output.scalar_type() == slots.scalar_type() && output.sizes() == slots.sizes(), "output must match slots");
  TORCH_CHECK(
      failure_status.scalar_type() == at::kInt && failure_status.dim() == 1 &&
          failure_status.numel() == expected_tags.size(0),
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
        request_epochs.data_ptr<int64_t>(),
        reinterpret_cast<const unsigned long long*>(expected_tags.data_ptr<int64_t>()),
        output.data_ptr<int32_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_tags.size(0),
        buffer.size(0),
        expected_tags.size(1),
        static_cast<uint64_t>(mapping_seed),
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
        request_epochs.data_ptr<int64_t>(),
        reinterpret_cast<const unsigned long long*>(expected_tags.data_ptr<int64_t>()),
        output.data_ptr<int64_t>(),
        failure_status.data_ptr<int32_t>(),
        slots.numel(),
        request_indices.numel(),
        expected_tags.size(0),
        buffer.size(0),
        expected_tags.size(1),
        static_cast<uint64_t>(mapping_seed),
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
