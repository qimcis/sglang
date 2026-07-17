/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

constexpr int32_t kKVPageInvalidMapping = 0x01;
constexpr int32_t kKVPageOwnerMismatch = 0x02;
constexpr int32_t kKVPagePositionMismatch = 0x04;
constexpr int32_t kKVPageAttentionTagMismatch = 0x08;
constexpr int32_t kKVPageGenerationMismatch = 0x10;
constexpr int32_t kKVPageTransferTagMismatch = 0x20;

struct KVTopKProtectionParams {
  const int64_t* __restrict__ request_indices;  // [B]
  int32_t page_size;
  int32_t page_offset;
  int32_t num_physical_pages;
  int32_t num_request_slots;
  const int64_t* __restrict__ actual_tags;
  const int64_t* __restrict__ actual_generations;
  const int32_t* __restrict__ actual_transfer_tags;
  const int32_t* __restrict__ owner_request_indices;
  const int32_t* __restrict__ owner_page_positions;
  const int64_t* __restrict__ expected_tags;
  const int64_t* __restrict__ expected_generations;
  const int32_t* __restrict__ expected_transfer_tags;
  const int32_t* __restrict__ request_epochs;
  int32_t* __restrict__ validated_epochs;
  int32_t* __restrict__ status;
};

__device__ __forceinline__ int32_t validate_selected_token_slot(
    const KVTopKProtectionParams& protection,
    int32_t request_idx,
    int32_t logical_position,
    int32_t token_slot,
    int32_t length) {
  int32_t validation_status = 0;
  if (logical_position < 0 || logical_position >= length || token_slot <= 0) {
    return kKVPageInvalidMapping;
  }

  const int64_t physical_page = static_cast<int64_t>(token_slot) / protection.page_size;
  const int32_t physical_offset = token_slot % protection.page_size;
  const int32_t logical_page = logical_position / protection.page_size;
  const int32_t logical_offset = logical_position % protection.page_size;
  const int64_t sidecar_page = physical_page + protection.page_offset;
  if (physical_offset != logical_offset || sidecar_page <= 0 || sidecar_page >= protection.num_physical_pages) {
    return kKVPageInvalidMapping;
  }

  const auto page = static_cast<int32_t>(sidecar_page);
  if (protection.owner_request_indices[page] != request_idx) {
    validation_status |= kKVPageOwnerMismatch;
  }
  if (protection.owner_page_positions[page] != logical_page) {
    validation_status |= kKVPagePositionMismatch;
  }
  if (protection.actual_tags[page] != protection.expected_tags[page]) {
    validation_status |= kKVPageAttentionTagMismatch;
  }
  if (protection.actual_generations[page] != protection.expected_generations[page]) {
    validation_status |= kKVPageGenerationMismatch;
  }
  if (protection.actual_transfer_tags[page] != protection.expected_transfer_tags[page]) {
    validation_status |= kKVPageTransferTagMismatch;
  }
  return validation_status;
}

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // protected decode
    void topk_transform_decode_protected_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        int64_t src_stride,
        int32_t src_num_cols,
        int32_t score_num_cols,
        const KVTopKProtectionParams protection) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = params.lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = params.input + bid * params.input_stride;
  const auto request_idx_i64 = protection.request_indices[bid];
  const bool request_idx_valid =
      request_idx_i64 >= 0 && request_idx_i64 < static_cast<int64_t>(protection.num_request_slots);
  const auto request_idx = request_idx_valid ? static_cast<int32_t>(request_idx_i64) : -1;
  const bool graph_padding = request_idx == 0;

  __shared__ int32_t s_validation_status;
  const bool row_shape_valid =
      request_idx_valid && !graph_padding && length >= 0 && length <= src_num_cols && length <= score_num_cols;
  if (tid == 0) {
    s_validation_status = (graph_padding || row_shape_valid) ? 0 : kKVPageInvalidMapping;
  }
  __syncthreads();

  const auto validate_and_store = [&](int32_t output_position, int32_t logical_position) {
    if (logical_position < 0 || logical_position >= length || logical_position >= src_num_cols) {
      dst_page_entry[output_position] = 0;
      ::atomicOr(&s_validation_status, kKVPageInvalidMapping);
      return;
    }

    const auto token_slot = src_page_entry[logical_position];
    const auto entry_status =
        validate_selected_token_slot(protection, request_idx, logical_position, token_slot, length);
    dst_page_entry[output_position] = entry_status == 0 ? token_slot : 0;
    if (entry_status != 0) {
      ::atomicOr(&s_validation_status, entry_status);
    }
  };

  if (graph_padding || !row_shape_valid) {
    for (auto i = static_cast<int32_t>(tid); i < TopK; i += kThreadsPerBlock) {
      dst_page_entry[i] = 0;
    }
  } else if (length <= TopK) {
    for (auto i = static_cast<int32_t>(tid); i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        validate_and_store(i, i);
      } else {
        dst_page_entry[i] = -1;
      }
    }
  } else {
    __shared__ int32_t s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, 0, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    validate_and_store(tid, s_indices[tid]);
    validate_and_store(tid + kThreadsPerBlock, s_indices[tid + kThreadsPerBlock]);
  }

  __syncthreads();
  if (tid == 0 && request_idx > 0) {
    if (s_validation_status != 0) {
      ::atomicOr(protection.status + request_idx, s_validation_status);
    }
    // The post-forward status check may trust this marker only after all
    // sanitized output and status writes are globally visible.
    __threadfence();
    ::atomicExch(protection.validated_epochs + request_idx, protection.request_epochs[request_idx]);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

void check_protection_tensor(
    const at::Tensor& tensor, const at::Tensor& reference, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the score device");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be a contiguous 1D tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> protection_request_indices_opt,
    int64_t protection_page_size,
    int64_t protection_page_offset,
    std::optional<at::Tensor> protection_actual_tags_opt,
    std::optional<at::Tensor> protection_actual_generations_opt,
    std::optional<at::Tensor> protection_actual_transfer_tags_opt,
    std::optional<at::Tensor> protection_owner_request_indices_opt,
    std::optional<at::Tensor> protection_owner_page_positions_opt,
    std::optional<at::Tensor> protection_expected_tags_opt,
    std::optional<at::Tensor> protection_expected_generations_opt,
    std::optional<at::Tensor> protection_expected_transfer_tags_opt,
    std::optional<at::Tensor> protection_request_epochs_opt,
    std::optional<at::Tensor> protection_validated_epochs_opt,
    std::optional<at::Tensor> protection_status_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must be int32");
  TORCH_CHECK(dst_page_table.scalar_type() == at::kInt, "dst_page_table must be int32");
  TORCH_CHECK(src_page_table.scalar_type() == at::kInt, "src_page_table must be int32");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(lengths.device() == score.device(), "lengths must be on the score device");
  TORCH_CHECK(dst_page_table.device() == score.device(), "dst_page_table must be on the score device");
  TORCH_CHECK(src_page_table.device() == score.device(), "src_page_table must be on the score device");
  TORCH_CHECK(cu_seqlens_q.device() == score.device(), "cu_seqlens_q must be on the score device");
  if (row_starts_opt.has_value()) {
    TORCH_CHECK(row_starts_opt->scalar_type() == at::kInt, "row_starts must be int32");
    TORCH_CHECK(row_starts_opt->device() == score.device(), "row_starts must be on the score device");
  }
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  const bool any_protection =
      protection_request_indices_opt.has_value() || protection_page_size != 0 || protection_page_offset != 0 ||
      protection_actual_tags_opt.has_value() || protection_actual_generations_opt.has_value() ||
      protection_actual_transfer_tags_opt.has_value() || protection_owner_request_indices_opt.has_value() ||
      protection_owner_page_positions_opt.has_value() || protection_expected_tags_opt.has_value() ||
      protection_expected_generations_opt.has_value() || protection_expected_transfer_tags_opt.has_value() ||
      protection_request_epochs_opt.has_value() || protection_validated_epochs_opt.has_value() ||
      protection_status_opt.has_value();
  const bool complete_protection =
      protection_request_indices_opt.has_value() && protection_actual_tags_opt.has_value() &&
      protection_actual_generations_opt.has_value() && protection_actual_transfer_tags_opt.has_value() &&
      protection_owner_request_indices_opt.has_value() && protection_owner_page_positions_opt.has_value() &&
      protection_expected_tags_opt.has_value() && protection_expected_generations_opt.has_value() &&
      protection_expected_transfer_tags_opt.has_value() && protection_request_epochs_opt.has_value() &&
      protection_validated_epochs_opt.has_value() && protection_status_opt.has_value();
  TORCH_CHECK(!any_protection || complete_protection, "topk KV protection arguments must be provided together");

  if (complete_protection) {
    TORCH_CHECK(is_decode, "topk KV protection currently supports decode only");
    TORCH_CHECK(
        protection_page_size > 0 && protection_page_size <= std::numeric_limits<int32_t>::max(),
        "topk KV protection page size is out of range");
    TORCH_CHECK(
        protection_page_offset >= 0 && protection_page_offset <= std::numeric_limits<int32_t>::max(),
        "topk KV protection page offset is out of range");
    TORCH_CHECK(
        src_page_table.size(1) <= std::numeric_limits<int32_t>::max() &&
            score.size(1) <= std::numeric_limits<int32_t>::max(),
        "topk KV protection input is too wide");

    const auto& request_indices = protection_request_indices_opt.value();
    const auto& actual_tags = protection_actual_tags_opt.value();
    const auto& actual_generations = protection_actual_generations_opt.value();
    const auto& actual_transfer_tags = protection_actual_transfer_tags_opt.value();
    const auto& owner_request_indices = protection_owner_request_indices_opt.value();
    const auto& owner_page_positions = protection_owner_page_positions_opt.value();
    const auto& expected_tags = protection_expected_tags_opt.value();
    const auto& expected_generations = protection_expected_generations_opt.value();
    const auto& expected_transfer_tags = protection_expected_transfer_tags_opt.value();
    const auto& request_epochs = protection_request_epochs_opt.value();
    auto& validated_epochs = protection_validated_epochs_opt.value();
    auto& status = protection_status_opt.value();

    check_protection_tensor(request_indices, score, at::kLong, "protection_request_indices");
    check_protection_tensor(actual_tags, score, at::kLong, "protection_actual_tags");
    check_protection_tensor(actual_generations, score, at::kLong, "protection_actual_generations");
    check_protection_tensor(actual_transfer_tags, score, at::kInt, "protection_actual_transfer_tags");
    check_protection_tensor(owner_request_indices, score, at::kInt, "protection_owner_request_indices");
    check_protection_tensor(owner_page_positions, score, at::kInt, "protection_owner_page_positions");
    check_protection_tensor(expected_tags, score, at::kLong, "protection_expected_tags");
    check_protection_tensor(expected_generations, score, at::kLong, "protection_expected_generations");
    check_protection_tensor(expected_transfer_tags, score, at::kInt, "protection_expected_transfer_tags");
    check_protection_tensor(request_epochs, score, at::kInt, "protection_request_epochs");
    check_protection_tensor(validated_epochs, score, at::kInt, "protection_validated_epochs");
    check_protection_tensor(status, score, at::kInt, "protection_status");

    TORCH_CHECK(request_indices.size(0) == B, "topk KV protection request-index batch mismatch");
    const auto num_physical_pages = actual_tags.size(0);
    TORCH_CHECK(
        num_physical_pages > 0 && num_physical_pages <= std::numeric_limits<int32_t>::max(),
        "topk KV protection physical sidecar size is out of range");
    TORCH_CHECK(
        actual_generations.size(0) == num_physical_pages && actual_transfer_tags.size(0) == num_physical_pages &&
            owner_request_indices.size(0) == num_physical_pages && owner_page_positions.size(0) == num_physical_pages &&
            expected_tags.size(0) == num_physical_pages && expected_generations.size(0) == num_physical_pages &&
            expected_transfer_tags.size(0) == num_physical_pages,
        "topk KV protection physical sidecar length mismatch");
    const auto num_request_slots = request_epochs.size(0);
    TORCH_CHECK(
        num_request_slots > 0 && num_request_slots <= std::numeric_limits<int32_t>::max(),
        "topk KV protection request sidecar size is out of range");
    TORCH_CHECK(
        validated_epochs.size(0) == num_request_slots && status.size(0) == num_request_slots,
        "topk KV protection request sidecar length mismatch");

    const KVTopKProtectionParams protection{
        .request_indices = request_indices.data_ptr<int64_t>(),
        .page_size = static_cast<int32_t>(protection_page_size),
        .page_offset = static_cast<int32_t>(protection_page_offset),
        .num_physical_pages = static_cast<int32_t>(num_physical_pages),
        .num_request_slots = static_cast<int32_t>(num_request_slots),
        .actual_tags = actual_tags.data_ptr<int64_t>(),
        .actual_generations = actual_generations.data_ptr<int64_t>(),
        .actual_transfer_tags = actual_transfer_tags.data_ptr<int32_t>(),
        .owner_request_indices = owner_request_indices.data_ptr<int32_t>(),
        .owner_page_positions = owner_page_positions.data_ptr<int32_t>(),
        .expected_tags = expected_tags.data_ptr<int64_t>(),
        .expected_generations = expected_generations.data_ptr<int64_t>(),
        .expected_transfer_tags = expected_transfer_tags.data_ptr<int32_t>(),
        .request_epochs = request_epochs.data_ptr<int32_t>(),
        .validated_epochs = validated_epochs.data_ptr<int32_t>(),
        .status = status.data_ptr<int32_t>(),
    };
    setup_kernel_smem_once<topk_transform_decode_protected_kernel, kSmem>();
    topk_transform_decode_protected_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        static_cast<int32_t>(src_page_table.size(1)),
        static_cast<int32_t>(score.size(1)),
        protection);
  } else if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
