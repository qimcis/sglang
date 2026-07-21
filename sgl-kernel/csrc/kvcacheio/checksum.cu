// Direct-KV transfer checksum kernel for PD disaggregation.
//
// Hashes K/V cache bytes in *logical token order* directly from the per-layer
// KV-cache buffers, WITHOUT first materializing a `[selected_tokens, row_bytes]`
// tensor (which `gather_logical_kv_rows` does in the Torch reference path).
//
// Bit-for-bit parity with
// `sglang.srt.mem_cache.kv_page_tags.hash_rows_with_positions`:
//
//   acc = CKSUM_SEED
//   if positions: acc = splitmix64(acc ^ position)
//   for lane in concatenated_int64_lanes(K(l0),V(l0),K(l1),V(l1),...)[:num_lanes]:
//       acc = splitmix64(acc ^ lane)
//   out[row] = acc        # XOR-reduce + finishing mixes happen in Python
//
// The per-lane fold is a strict serial chain (splitmix64 is NOT associative),
// so parallelism is *across rows*: one warp per selected token, with coalesced
// vector loads fed to lane 0 (which carries the serial chain) in lane order.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace {

// splitmix64 finalizer over uint64 (wraps mod 2^64), matching
// `_splitmix64_tensor` / `_mix_tensor(acc, f) = splitmix64(acc ^ f)`.
__device__ __forceinline__ uint64_t cksum_splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

// Must match `_CKSUM_SEED` in kv_page_tags.py.
__device__ constexpr uint64_t kCksumSeed = 0x5347'4C41'4E47'4353ULL;

__device__ __forceinline__ uint64_t kv_checksum_hash_slot(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    int64_t loc,
    int64_t position,
    bool has_position,
    int B,
    int64_t cap,
    int lane) {
  uint64_t acc = kCksumSeed;
  if (has_position) {
    acc = cksum_splitmix64(acc ^ static_cast<uint64_t>(position));
  }

  const uint64_t lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
  uint64_t consumed = 0;  // warp-uniform (every lane steps it identically)

  for (int b = 0; b < B && consumed < lane_cap; ++b) {
    const int64_t nlanes = row_nbytes[b] >> 3;  // /8
    const char* base = reinterpret_cast<const char*>(buffer_ptrs[b]) + loc * row_strides[b];
    const long long* p = reinterpret_cast<const long long*>(base);
    const bool a16 = ((reinterpret_cast<uintptr_t>(base) & 15ULL) == 0ULL);

    int64_t j = 0;
    // 16B vectorized bulk: 32 threads * 2 lanes = 64 lanes / iteration.
    if (a16) {
      const longlong2* p2 = reinterpret_cast<const longlong2*>(base);
      for (; j + 64 <= nlanes && consumed < lane_cap; j += 64) {
        longlong2 v = p2[(j >> 1) + lane];  // lanes (j+2*lane, j+2*lane+1)
#pragma unroll
        for (int k = 0; k < 32; ++k) {
          long long lo = __shfl_sync(0xffffffffu, v.x, k);
          long long hi = __shfl_sync(0xffffffffu, v.y, k);
          if (consumed < lane_cap) {
            if (lane == 0) acc = cksum_splitmix64(acc ^ static_cast<uint64_t>(lo));
            consumed++;
          }
          if (consumed < lane_cap) {
            if (lane == 0) acc = cksum_splitmix64(acc ^ static_cast<uint64_t>(hi));
            consumed++;
          }
        }
      }
    }
    // 8B coalesced remainder: 32 lanes / iteration.
    for (; j < nlanes && consumed < lane_cap; j += 32) {
      const int64_t jj = j + lane;
      long long v = (jj < nlanes) ? p[jj] : 0;
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        long long vk = __shfl_sync(0xffffffffu, v, k);
        if ((j + k) < nlanes && consumed < lane_cap) {
          if (lane == 0) acc = cksum_splitmix64(acc ^ static_cast<uint64_t>(vk));
          consumed++;
        }
      }
    }
  }

  return acc;
}

__device__ __forceinline__ uint8_t load_strided_row_byte(
    const char* __restrict__ base,
    int64_t loc,
    int64_t byte_offset,
    int64_t stride0_b,
    int64_t elem_size,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides_b,
    int ndims) {
  int64_t elem = byte_offset / elem_size;
  const int64_t byte_in_elem = byte_offset - elem * elem_size;
  int64_t storage = loc * stride0_b + byte_in_elem;
  for (int d = ndims - 1; d >= 0; --d) {
    const int64_t size = sizes[d];
    const int64_t idx = elem % size;
    elem /= size;
    storage += idx * strides_b[d];
  }
  return *reinterpret_cast<const uint8_t*>(base + storage);
}

__device__ __forceinline__ uint64_t load_strided_lane(
    const char* __restrict__ base,
    int64_t loc,
    int64_t byte_offset,
    int64_t stride0_b,
    int64_t row_nbytes,
    int64_t elem_size,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides_b,
    int ndims) {
  uint64_t lane = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int64_t off = byte_offset + k;
    const uint64_t v = off < row_nbytes ? static_cast<uint64_t>(load_strided_row_byte(
                                              base, loc, off, stride0_b, elem_size, sizes, strides_b, ndims))
                                        : 0ULL;
    lane |= (v << (8 * k));
  }
  return lane;
}

__device__ __forceinline__ uint64_t kv_checksum_hash_slot_strided(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const int64_t* __restrict__ elem_sizes,
    const int64_t* __restrict__ meta_offsets,
    const int64_t* __restrict__ meta_ndims,
    const int64_t* __restrict__ inner_sizes,
    const int64_t* __restrict__ inner_strides,
    int64_t loc,
    int64_t position,
    int B,
    int64_t cap,
    int lane) {
  uint64_t acc = cksum_splitmix64(kCksumSeed ^ static_cast<uint64_t>(position));
  const uint64_t lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
  uint64_t consumed = 0;

  for (int b = 0; b < B && consumed < lane_cap; ++b) {
    const int64_t nlanes = (row_nbytes[b] + 7) >> 3;
    const char* base = reinterpret_cast<const char*>(buffer_ptrs[b]);
    const int64_t offset = meta_offsets[b];
    const int ndims = static_cast<int>(meta_ndims[b]);
    const int64_t* sizes = inner_sizes + offset;
    const int64_t* strides = inner_strides + offset;

    for (int64_t j = 0; j < nlanes && consumed < lane_cap; j += 32) {
      const int64_t jj = j + lane;
      const uint64_t v =
          jj < nlanes ? load_strided_lane(
                            base, loc, jj * 8, row_strides[b], row_nbytes[b], elem_sizes[b], sizes, strides, ndims)
                      : 0ULL;
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        const uint64_t vk = __shfl_sync(0xffffffffu, v, k);
        if ((j + k) < nlanes && consumed < lane_cap) {
          if (lane == 0) acc = cksum_splitmix64(acc ^ vk);
          consumed++;
        }
      }
    }
  }

  return acc;
}

// One warp per selected token; coalesced 16B (longlong2) loads with an 8B tail.
template <int BLOCK>
__global__ void kv_checksum_direct_kernel(
    const uint64_t* __restrict__ buffer_ptrs,  // [B] device pointers (as int64)
    const int64_t* __restrict__ row_strides,   // [B] bytes between dim-0 rows
    const int64_t* __restrict__ row_nbytes,    // [B] flattened bytes per row (%8==0)
    const int64_t* __restrict__ sel_loc,       // [N] physical slot per logical token
    const int64_t* __restrict__ positions,     // [N] or nullptr
    int B,
    int N,
    int64_t cap,  // max leading concatenated lanes, <0 => all
    int64_t* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int warps_per_block = BLOCK >> 5;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (row >= N) return;  // warp-uniform: whole warp returns or none does

  const int64_t loc = sel_loc[row];
  const uint64_t acc = kv_checksum_hash_slot(
      buffer_ptrs,
      row_strides,
      row_nbytes,
      loc,
      (positions == nullptr) ? 0 : positions[row],
      positions != nullptr,
      B,
      cap,
      lane);
  if (lane == 0) out[row] = static_cast<int64_t>(acc);
}

template <int BLOCK, typename ReqToTokenT>
__global__ void kv_checksum_direct_batched_kernel(
    const uint64_t* __restrict__ buffer_ptrs,      // [B] device pointers (as int64)
    const int64_t* __restrict__ row_strides,       // [B] bytes between dim-0 rows
    const int64_t* __restrict__ row_nbytes,        // [B] flattened bytes per row (%8==0)
    const ReqToTokenT* __restrict__ req_to_token,  // [num_req_rows, max_context_len]
    const int64_t* __restrict__ req_pool_indices,  // [R]
    const int64_t* __restrict__ selected_offsets,  // [R]
    const int64_t* __restrict__ selected_lengths,  // [R]
    const int64_t* __restrict__ selected_indices,  // [sum(selected_lengths)]
    const int64_t* __restrict__ elem_sizes,        // [B]
    const int64_t* __restrict__ meta_offsets,      // [B]
    const int64_t* __restrict__ meta_ndims,        // [B]
    const int64_t* __restrict__ inner_sizes,       // [sum(meta_ndims)]
    const int64_t* __restrict__ inner_strides,     // [sum(meta_ndims)] bytes
    int64_t req_to_token_stride0,
    int B,
    int64_t cap,
    int64_t* __restrict__ out) {  // [R] final request checksums
  constexpr int kWarpSize = 32;
  constexpr int kWarps = BLOCK / kWarpSize;
  __shared__ uint64_t warp_combined[kWarps];

  const int req = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t start = selected_offsets[req];
  const int64_t length = selected_lengths[req];
  const int64_t req_pool = req_pool_indices[req];
  const ReqToTokenT* req_row = req_to_token + req_pool * req_to_token_stride0;

  uint64_t combined = 0;
  for (int64_t j = warp; j < length; j += kWarps) {
    const int64_t logical_pos = selected_indices[start + j];
    const int64_t loc = static_cast<int64_t>(req_row[logical_pos]);
    const uint64_t row_acc = kv_checksum_hash_slot_strided(
        buffer_ptrs,
        row_strides,
        row_nbytes,
        elem_sizes,
        meta_offsets,
        meta_ndims,
        inner_sizes,
        inner_strides,
        loc,
        logical_pos,
        B,
        cap,
        lane);
    if (lane == 0) combined ^= row_acc;
  }

  if (lane == 0) warp_combined[warp] = combined;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t req_combined = 0;
#pragma unroll
    for (int i = 0; i < kWarps; ++i) {
      req_combined ^= warp_combined[i];
    }
    if (length == 0) {
      out[req] = static_cast<int64_t>(cksum_splitmix64(kCksumSeed));
      return;
    }
    uint64_t total = cksum_splitmix64(kCksumSeed ^ req_combined);
    total = cksum_splitmix64(total ^ static_cast<uint64_t>(length));
    out[req] = static_cast<int64_t>(total);
  }
}

}  // namespace

// See sgl_kernel_ops.h for the contract. `buffer_ptrs`/`row_strides`/`row_nbytes`
// are small [B] int64 CUDA tensors; `sel_loc`/`positions` are [N] int64 CUDA
// tensors; `out` is a preallocated [N] int64 CUDA tensor (per-row accumulators).
void kv_checksum_direct(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& sel_loc,
    const std::optional<at::Tensor>& positions,
    int64_t num_lanes,
    at::Tensor& out) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(sel_loc.scalar_type() == at::kLong, "sel_loc must be int64");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(sel_loc.is_cuda() && out.is_cuda(), "sel_loc/out must be CUDA tensors");
  TORCH_CHECK(
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda(),
      "buffer metadata tensors must be CUDA tensors");
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(sel_loc.is_contiguous(), "sel_loc must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  const int B = static_cast<int>(buffer_ptrs.numel());
  const int N = static_cast<int>(sel_loc.numel());
  TORCH_CHECK(row_strides.numel() == B && row_nbytes.numel() == B, "metadata length mismatch");
  TORCH_CHECK(out.numel() == N, "out must have N elements");
  if (N == 0 || B == 0) return;

  const int64_t* pos_ptr = nullptr;
  if (positions.has_value() && positions->numel() > 0) {
    TORCH_CHECK(positions->scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(positions->is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(positions->is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(positions->numel() == N, "positions must have N elements");
    pos_ptr = positions->data_ptr<int64_t>();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kBlock = 256;
  const int warps_per_block = kBlock / 32;
  const int grid = (N + warps_per_block - 1) / warps_per_block;
  kv_checksum_direct_kernel<kBlock><<<grid, kBlock, 0, stream> > >(
      reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
      row_strides.data_ptr<int64_t>(),
      row_nbytes.data_ptr<int64_t>(),
      sel_loc.data_ptr<int64_t>(),
      pos_ptr,
      B,
      N,
      num_lanes,
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

// Batched form: one CUDA launch computes one final checksum per request.  The
// physical slots are read from SGLang's req_to_token matrix, so the caller does
// not need to materialize a flat kv_loc/sel_loc list for every request.
void kv_checksum_direct_batched(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& selected_offsets,
    const at::Tensor& selected_lengths,
    const at::Tensor& selected_indices,
    const at::Tensor& elem_sizes,
    const at::Tensor& meta_offsets,
    const at::Tensor& meta_ndims,
    const at::Tensor& inner_sizes,
    const at::Tensor& inner_strides,
    int64_t num_lanes,
    at::Tensor& out) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct_batched is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(
      req_to_token.scalar_type() == at::kInt || req_to_token.scalar_type() == at::kLong,
      "req_to_token must be int32 or int64");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
  TORCH_CHECK(selected_offsets.scalar_type() == at::kLong, "selected_offsets must be int64");
  TORCH_CHECK(selected_lengths.scalar_type() == at::kLong, "selected_lengths must be int64");
  TORCH_CHECK(selected_indices.scalar_type() == at::kLong, "selected_indices must be int64");
  TORCH_CHECK(elem_sizes.scalar_type() == at::kLong, "elem_sizes must be int64");
  TORCH_CHECK(meta_offsets.scalar_type() == at::kLong, "meta_offsets must be int64");
  TORCH_CHECK(meta_ndims.scalar_type() == at::kLong, "meta_ndims must be int64");
  TORCH_CHECK(inner_sizes.scalar_type() == at::kLong, "inner_sizes must be int64");
  TORCH_CHECK(inner_strides.scalar_type() == at::kLong, "inner_strides must be int64");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(req_to_token.is_cuda() && out.is_cuda(), "req_to_token/out must be CUDA tensors");
  TORCH_CHECK(
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda() && req_pool_indices.is_cuda() &&
          selected_offsets.is_cuda() && selected_lengths.is_cuda() && selected_indices.is_cuda() &&
          elem_sizes.is_cuda() && meta_offsets.is_cuda() && meta_ndims.is_cuda() && inner_sizes.is_cuda() &&
          inner_strides.is_cuda(),
      "all metadata tensors must be CUDA tensors");
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(req_to_token.is_contiguous(), "req_to_token must be contiguous");
  TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
  TORCH_CHECK(selected_offsets.is_contiguous(), "selected_offsets must be contiguous");
  TORCH_CHECK(selected_lengths.is_contiguous(), "selected_lengths must be contiguous");
  TORCH_CHECK(selected_indices.is_contiguous(), "selected_indices must be contiguous");
  TORCH_CHECK(elem_sizes.is_contiguous(), "elem_sizes must be contiguous");
  TORCH_CHECK(meta_offsets.is_contiguous(), "meta_offsets must be contiguous");
  TORCH_CHECK(meta_ndims.is_contiguous(), "meta_ndims must be contiguous");
  TORCH_CHECK(inner_sizes.is_contiguous(), "inner_sizes must be contiguous");
  TORCH_CHECK(inner_strides.is_contiguous(), "inner_strides must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2D");

  const int B = static_cast<int>(buffer_ptrs.numel());
  const int R = static_cast<int>(req_pool_indices.numel());
  TORCH_CHECK(row_strides.numel() == B && row_nbytes.numel() == B, "metadata length mismatch");
  TORCH_CHECK(
      elem_sizes.numel() == B && meta_offsets.numel() == B && meta_ndims.numel() == B,
      "buffer layout metadata length mismatch");
  TORCH_CHECK(selected_offsets.numel() == R && selected_lengths.numel() == R, "request metadata length mismatch");
  TORCH_CHECK(out.numel() == R, "out must have one element per request");
  if (R == 0 || B == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kBlock = 256;
  const int64_t req_to_token_stride0 = req_to_token.stride(0);
  if (req_to_token.scalar_type() == at::kInt) {
    kv_checksum_direct_batched_kernel<kBlock, int32_t><<<R, kBlock, 0, stream> > >(
        reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
        row_strides.data_ptr<int64_t>(),
        row_nbytes.data_ptr<int64_t>(),
        req_to_token.data_ptr<int32_t>(),
        req_pool_indices.data_ptr<int64_t>(),
        selected_offsets.data_ptr<int64_t>(),
        selected_lengths.data_ptr<int64_t>(),
        selected_indices.data_ptr<int64_t>(),
        elem_sizes.data_ptr<int64_t>(),
        meta_offsets.data_ptr<int64_t>(),
        meta_ndims.data_ptr<int64_t>(),
        inner_sizes.data_ptr<int64_t>(),
        inner_strides.data_ptr<int64_t>(),
        req_to_token_stride0,
        B,
        num_lanes,
        out.data_ptr<int64_t>());
  } else {
    kv_checksum_direct_batched_kernel<kBlock, int64_t><<<R, kBlock, 0, stream> > >(
        reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
        row_strides.data_ptr<int64_t>(),
        row_nbytes.data_ptr<int64_t>(),
        req_to_token.data_ptr<int64_t>(),
        req_pool_indices.data_ptr<int64_t>(),
        selected_offsets.data_ptr<int64_t>(),
        selected_lengths.data_ptr<int64_t>(),
        selected_indices.data_ptr<int64_t>(),
        elem_sizes.data_ptr<int64_t>(),
        meta_offsets.data_ptr<int64_t>(),
        meta_ndims.data_ptr<int64_t>(),
        inner_sizes.data_ptr<int64_t>(),
        inner_strides.data_ptr<int64_t>(),
        req_to_token_stride0,
        B,
        num_lanes,
        out.data_ptr<int64_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}
