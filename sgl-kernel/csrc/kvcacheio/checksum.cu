// Direct-KV transfer checksum for PD disaggregation.
//
// One warp hashes one logical token row directly from K/V cache buffers. Each
// thread hashes independent 8-byte chunks salted by logical token position and
// chunk offset; Python XOR-reduces the per-row outputs into the final checksum.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace {

// Murmur/XXH-style uint32 finalizer, matching `_fmix32_*` in kv_page_tags.py.
__device__ __forceinline__ uint32_t cksum_fmix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

// Must match `_CKSUM32_*` constants in kv_page_tags.py.
__device__ constexpr uint32_t kCksumSeed = 0x4E474353u;
__device__ constexpr uint32_t kPosMul = 0x9E3779B1u;
__device__ constexpr uint32_t kLaneMul = 0x85EBCA77u;
__device__ constexpr uint32_t kHiMul = 0xC2B2AE3Du;

__device__ __forceinline__ uint32_t cksum_chunk32(uint64_t value, int64_t position, uint64_t lane_index) {
  uint32_t h = kCksumSeed;
  h ^= static_cast<uint32_t>(position) * kPosMul;
  h ^= static_cast<uint32_t>(lane_index) * kLaneMul;
  h ^= static_cast<uint32_t>(value);
  h ^= static_cast<uint32_t>(value >> 32) * kHiMul;
  return cksum_fmix32(h);
}

// One warp per selected token; lanes stride over 8-byte chunks in the row.
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
  const int64_t position = (positions != nullptr) ? positions[row] : 0;

  const uint64_t lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
  uint64_t consumed = 0;  // warp-uniform (every lane steps it identically)
  uint32_t local = 0;

  for (int b = 0; b < B && consumed < lane_cap; ++b) {
    const int64_t nlanes = row_nbytes[b] >> 3;  // /8
    const char* base = reinterpret_cast<const char*>(buffer_ptrs[b]) + loc * row_strides[b];
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base);

    for (int64_t j = lane; j < nlanes; j += 32) {
      const uint64_t global_lane = consumed + static_cast<uint64_t>(j);
      if (global_lane < lane_cap) {
        local ^= cksum_chunk32(p[j], position, global_lane);
      }
    }
    consumed += static_cast<uint64_t>(nlanes);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local ^= __shfl_xor_sync(0xffffffffu, local, offset);
  }

  if (lane == 0) out[row] = static_cast<int64_t>(local);
}

// Same checksum math as kv_checksum_direct_kernel, but reads physical slots from
// kv_loc[start:start+N] and salts each row with logical position start+row.
template <int BLOCK>
__global__ void kv_checksum_direct_range_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const int64_t* __restrict__ kv_loc,
    int B,
    int64_t start,
    int N,
    int64_t cap,
    int64_t* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int warps_per_block = BLOCK >> 5;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (row >= N) return;

  const int64_t position = start + row;
  const int64_t loc = kv_loc[position];

  const uint64_t lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
  uint64_t consumed = 0;
  uint32_t local = 0;

  for (int b = 0; b < B && consumed < lane_cap; ++b) {
    const int64_t nlanes = row_nbytes[b] >> 3;
    const char* base = reinterpret_cast<const char*>(buffer_ptrs[b]) + loc * row_strides[b];
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base);

    for (int64_t j = lane; j < nlanes; j += 32) {
      const uint64_t global_lane = consumed + static_cast<uint64_t>(j);
      if (global_lane < lane_cap) {
        local ^= cksum_chunk32(p[j], position, global_lane);
      }
    }
    consumed += static_cast<uint64_t>(nlanes);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local ^= __shfl_xor_sync(0xffffffffu, local, offset);
  }

  if (lane == 0) out[row] = static_cast<int64_t>(local);
}

template <int BLOCK, typename LocT>
__global__ void kv_checksum_direct_table_batched_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const LocT* __restrict__ req_to_token,
    const int64_t req_to_token_stride0,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    int B,
    int64_t cap,
    uint32_t* __restrict__ accum) {
  const int lane = threadIdx.x & 31;
  const int warps_per_block = BLOCK >> 5;
  const int req = blockIdx.y;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int64_t n = lengths[req];
  if (row >= n) return;

  const int64_t position = starts[req] + row;
  const int64_t req_pool_idx = req_pool_indices[req];
  const int64_t loc = static_cast<int64_t>(req_to_token[req_pool_idx * req_to_token_stride0 + position]);

  const uint64_t lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
  uint64_t consumed = 0;
  uint32_t local = 0;

  for (int b = 0; b < B && consumed < lane_cap; ++b) {
    const int64_t nlanes = row_nbytes[b] >> 3;
    const char* base = reinterpret_cast<const char*>(buffer_ptrs[b]) + loc * row_strides[b];
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base);

    for (int64_t j = lane; j < nlanes; j += 32) {
      const uint64_t global_lane = consumed + static_cast<uint64_t>(j);
      if (global_lane < lane_cap) {
        local ^= cksum_chunk32(p[j], position, global_lane);
      }
    }
    consumed += static_cast<uint64_t>(nlanes);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local ^= __shfl_xor_sync(0xffffffffu, local, offset);
  }

  if (lane == 0) {
    atomicXor(reinterpret_cast<unsigned int*>(accum + req), static_cast<unsigned int>(local));
  }
}

__global__ void kv_checksum_finalize_batched_kernel(
    const uint32_t* __restrict__ accum, const int64_t* __restrict__ lengths, int M, int64_t* __restrict__ out) {
  const int req = blockIdx.x * blockDim.x + threadIdx.x;
  if (req >= M) return;
  const uint32_t h = cksum_fmix32(kCksumSeed ^ accum[req] ^ static_cast<uint32_t>(lengths[req]));
  out[req] = static_cast<int64_t>(h);
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
  auto kernel = kv_checksum_direct_kernel<kBlock>;
  // clang-format off
  kernel<<<grid, kBlock, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
      row_strides.data_ptr<int64_t>(),
      row_nbytes.data_ptr<int64_t>(),
      sel_loc.data_ptr<int64_t>(),
      pos_ptr,
      B,
      N,
      num_lanes,
      out.data_ptr<int64_t>());
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void kv_checksum_direct_range(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& kv_loc,
    int64_t start,
    int64_t num_tokens,
    int64_t num_lanes,
    at::Tensor& out) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct_range is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(kv_loc.scalar_type() == at::kLong, "kv_loc must be int64");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(kv_loc.is_cuda() && out.is_cuda(), "kv_loc/out must be CUDA tensors");
  TORCH_CHECK(
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda(),
      "buffer metadata tensors must be CUDA tensors");
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(kv_loc.is_contiguous(), "kv_loc must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  const int B = static_cast<int>(buffer_ptrs.numel());
  const int N = static_cast<int>(num_tokens);
  TORCH_CHECK(start >= 0, "start must be non-negative");
  TORCH_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  TORCH_CHECK(start + num_tokens <= kv_loc.numel(), "range exceeds kv_loc length");
  TORCH_CHECK(row_strides.numel() == B && row_nbytes.numel() == B, "metadata length mismatch");
  TORCH_CHECK(out.numel() >= N, "out must have at least num_tokens elements");
  if (N == 0 || B == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kBlock = 256;
  const int warps_per_block = kBlock / 32;
  const int grid = (N + warps_per_block - 1) / warps_per_block;
  auto kernel = kv_checksum_direct_range_kernel<kBlock>;
  // clang-format off
  kernel<<<grid, kBlock, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
      row_strides.data_ptr<int64_t>(),
      row_nbytes.data_ptr<int64_t>(),
      kv_loc.data_ptr<int64_t>(),
      B,
      start,
      N,
      num_lanes,
      out.data_ptr<int64_t>());
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void kv_checksum_direct_table_batched(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& starts,
    const at::Tensor& lengths,
    int64_t max_num_tokens,
    int64_t num_lanes,
    at::Tensor& accum,
    at::Tensor& out) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct_table_batched is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
  TORCH_CHECK(starts.scalar_type() == at::kLong, "starts must be int64");
  TORCH_CHECK(lengths.scalar_type() == at::kLong, "lengths must be int64");
  TORCH_CHECK(accum.scalar_type() == at::kInt, "accum must be int32");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(
      req_to_token.is_cuda() && accum.is_cuda() && out.is_cuda(), "req_to_token/accum/out must be CUDA tensors");
  TORCH_CHECK(
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda() && req_pool_indices.is_cuda() &&
          starts.is_cuda() && lengths.is_cuda(),
      "all metadata tensors must be CUDA tensors");
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
  TORCH_CHECK(starts.is_contiguous(), "starts must be contiguous");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(accum.is_contiguous(), "accum must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be a 2D table");
  TORCH_CHECK(
      req_to_token.scalar_type() == at::kInt || req_to_token.scalar_type() == at::kLong,
      "req_to_token must be int32 or int64");

  const int B = static_cast<int>(buffer_ptrs.numel());
  const int M = static_cast<int>(lengths.numel());
  TORCH_CHECK(row_strides.numel() == B && row_nbytes.numel() == B, "metadata length mismatch");
  TORCH_CHECK(req_pool_indices.numel() == M && starts.numel() == M, "batch metadata length mismatch");
  TORCH_CHECK(accum.numel() >= M && out.numel() >= M, "accum/out must have at least M elements");
  if (M == 0 || B == 0) return;

  const int64_t max_len = max_num_tokens;
  if (max_len <= 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream> > >(
        reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()),
        lengths.data_ptr<int64_t>(),
        M,
        out.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kBlock = 256;
  const int warps_per_block = kBlock / 32;
  const int grid_x = (static_cast<int>(max_len) + warps_per_block - 1) / warps_per_block;
  const dim3 grid(grid_x, M);
  const int64_t stride0 = req_to_token.stride(0);
  if (req_to_token.scalar_type() == at::kInt) {
    kv_checksum_direct_table_batched_kernel<kBlock><<<grid, kBlock, 0, stream> > >(
        reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
        row_strides.data_ptr<int64_t>(),
        row_nbytes.data_ptr<int64_t>(),
        req_to_token.data_ptr<int32_t>(),
        stride0,
        req_pool_indices.data_ptr<int64_t>(),
        starts.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(),
        B,
        num_lanes,
        reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
  } else {
    kv_checksum_direct_table_batched_kernel<kBlock><<<grid, kBlock, 0, stream> > >(
        reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
        row_strides.data_ptr<int64_t>(),
        row_nbytes.data_ptr<int64_t>(),
        req_to_token.data_ptr<int64_t>(),
        stride0,
        req_pool_indices.data_ptr<int64_t>(),
        starts.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(),
        B,
        num_lanes,
        reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream> > >(
      reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()),
      lengths.data_ptr<int64_t>(),
      M,
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}
