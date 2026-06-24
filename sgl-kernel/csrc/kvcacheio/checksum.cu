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
  uint64_t acc = kCksumSeed;
  if (positions != nullptr) {
    acc = cksum_splitmix64(acc ^ static_cast<uint64_t>(positions[row]));
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

  if (lane == 0) out[row] = static_cast<int64_t>(acc);
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
