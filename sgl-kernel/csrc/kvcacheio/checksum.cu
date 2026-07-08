// Batched direct-KV transfer checksum for PD disaggregation.
//
// One warp hashes one logical token row directly from K/V cache buffers. Each
// thread hashes independent 8-byte chunks salted by logical token position and
// chunk offset; the batched serving op XOR-reduces rows into one checksum per
// request.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

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

// `seed_pos` pre-folds the seed and the warp-invariant position salt; XOR is
// commutative so the result is bit-identical to the Python reference.
__device__ __forceinline__ uint32_t cksum_chunk32(uint64_t value, uint32_t seed_pos, uint64_t lane_index) {
  uint32_t h = seed_pos;
  h ^= static_cast<uint32_t>(lane_index) * kLaneMul;
  h ^= static_cast<uint32_t>(value);
  h ^= static_cast<uint32_t>(value >> 32) * kHiMul;
  return cksum_fmix32(h);
}

// Per-buffer metadata staged once per block into shared memory so warps do not
// re-read the global metadata arrays for every (row, buffer) pair.
struct BufMeta {
  const char* base;
  int64_t stride;
  int32_t nlanes;  // 8-byte lanes per row
  int32_t flags;
};

constexpr int32_t kFlagSwa = 1;
// Set when base pointer and row stride are both 16-byte aligned, so every row
// of the buffer can be read with 128-bit vector loads.
constexpr int32_t kFlagVec16 = 2;

template <int BLOCK, bool kCapped, typename LocT>
__global__ void __launch_bounds__(BLOCK) kv_checksum_direct_table_batched_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const int64_t* __restrict__ swa_buffer_flags,
    const int64_t* __restrict__ full_to_swa_index_mapping,
    const bool has_swa,
    const LocT* __restrict__ req_to_token,
    const int64_t req_to_token_stride0,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    int B,
    int64_t cap,
    uint32_t* __restrict__ accum) {
  extern __shared__ char smem[];
  constexpr int warps_per_block = BLOCK / 32;
  // Dynamic shared memory layout: [BufMeta[B], uint32_t[warps_per_block]].
  BufMeta* smeta = reinterpret_cast<BufMeta*>(smem);
  uint32_t* warp_acc = reinterpret_cast<uint32_t*>(smem + static_cast<size_t>(B) * sizeof(BufMeta));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int req = blockIdx.y;
  const int64_t n = lengths[req];
  // Uniform whole-block exit before any barrier.
  if (static_cast<int64_t>(blockIdx.x) * warps_per_block >= n) return;

  for (int b = threadIdx.x; b < B; b += BLOCK) {
    BufMeta m;
    const uint64_t ptr = buffer_ptrs[b];
    const int64_t stride = row_strides[b];
    m.base = reinterpret_cast<const char*>(ptr);
    m.stride = stride;
    m.nlanes = static_cast<int32_t>(row_nbytes[b] >> 3);
    int32_t flags = (swa_buffer_flags[b] != 0) ? kFlagSwa : 0;
    if (((ptr | static_cast<uint64_t>(stride)) & 15u) == 0) flags |= kFlagVec16;
    m.flags = flags;
    smeta[b] = m;
  }
  if (threadIdx.x < warps_per_block) warp_acc[threadIdx.x] = 0;
  __syncthreads();

  const int row = blockIdx.x * warps_per_block + warp;
  if (row < n) {
    const int64_t position = starts[req] + row;
    const int64_t req_pool_idx = req_pool_indices[req];
    const int64_t full_loc = static_cast<int64_t>(req_to_token[req_pool_idx * req_to_token_stride0 + position]);
    // The SWA translation depends only on the token, not the buffer: look it
    // up once per row instead of once per (row, buffer).
    int64_t swa_loc = 0;
    if (has_swa) {
      swa_loc = full_to_swa_index_mapping[full_loc];
      // Unmapped SWA rows are out of the sliding window; slot 0 is the reserved
      // dummy row and is consistently zero on both source and destination.
      if (swa_loc < 0) swa_loc = 0;
    }
    const uint32_t seed_pos = kCksumSeed ^ (static_cast<uint32_t>(position) * kPosMul);

    uint64_t lane_cap = 0;
    if constexpr (kCapped) {
      lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
    }
    uint64_t consumed = 0;
    uint32_t local = 0;

    for (int b = 0; b < B; ++b) {
      if constexpr (kCapped) {
        if (consumed >= lane_cap) break;
      }
      const BufMeta m = smeta[b];
      const int64_t loc = (m.flags & kFlagSwa) ? swa_loc : full_loc;
      const char* base = m.base + loc * m.stride;
      int32_t limit = m.nlanes;
      if constexpr (kCapped) {
        const uint64_t remaining = lane_cap - consumed;
        if (remaining < static_cast<uint64_t>(m.nlanes)) {
          limit = static_cast<int32_t>(remaining);
        }
      }

      if (m.flags & kFlagVec16) {
        const ulonglong2* p2 = reinterpret_cast<const ulonglong2*>(base);
        const int32_t npairs = limit >> 1;
#pragma unroll 4
        for (int32_t t = lane; t < npairs; t += 32) {
          const ulonglong2 v = p2[t];
          const uint64_t gl = consumed + (static_cast<uint64_t>(t) * 2);
          local ^= cksum_chunk32(v.x, seed_pos, gl);
          local ^= cksum_chunk32(v.y, seed_pos, gl + 1);
        }
        if ((limit & 1) && lane == 0) {
          const uint64_t* p = reinterpret_cast<const uint64_t*>(base);
          local ^= cksum_chunk32(p[limit - 1], seed_pos, consumed + static_cast<uint64_t>(limit - 1));
        }
      } else {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(base);
#pragma unroll 4
        for (int32_t j = lane; j < limit; j += 32) {
          local ^= cksum_chunk32(p[j], seed_pos, consumed + static_cast<uint64_t>(j));
        }
      }
      consumed += static_cast<uint64_t>(m.nlanes);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local ^= __shfl_xor_sync(0xffffffffu, local, offset);
    }
    if (lane == 0) warp_acc[warp] = local;
  }
  __syncthreads();

  // XOR-reduce across warps so each block issues one atomic instead of one per
  // warp. Warp 0 is always active thanks to the whole-block exit above.
  if (warp == 0) {
    uint32_t block_acc = (lane < warps_per_block) ? warp_acc[lane] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_acc ^= __shfl_xor_sync(0xffffffffu, block_acc, offset);
    }
    if (lane == 0 && block_acc != 0) {
      atomicXor(reinterpret_cast<unsigned int*>(accum + req), static_cast<unsigned int>(block_acc));
    }
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

template <int kBlock, bool kCapped, typename LocT>
static void launch_kv_checksum_kernel(
    const dim3& grid,
    size_t smem_bytes,
    cudaStream_t stream,
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const int64_t* __restrict__ swa_buffer_flags,
    const int64_t* __restrict__ full_to_swa_index_mapping,
    bool has_swa,
    const LocT* __restrict__ req_to_token,
    int64_t req_to_token_stride0,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    int B,
    int64_t num_lanes,
    uint32_t* __restrict__ accum) {
  kv_checksum_direct_table_batched_kernel<kBlock, kCapped, LocT><<<grid, kBlock, smem_bytes, stream>>>(
      buffer_ptrs,
      row_strides,
      row_nbytes,
      swa_buffer_flags,
      full_to_swa_index_mapping,
      has_swa,
      req_to_token,
      req_to_token_stride0,
      req_pool_indices,
      starts,
      lengths,
      B,
      num_lanes,
      accum);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kv_checksum_direct_table_batched(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& swa_buffer_flags,
    const at::Tensor& full_to_swa_index_mapping,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& starts,
    const at::Tensor& lengths,
    int64_t max_num_tokens,
    int64_t num_lanes,
    bool has_swa,
    bool is_capped,
    at::Tensor& accum,
    at::Tensor& out) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct_table_batched is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(swa_buffer_flags.scalar_type() == at::kLong, "swa_buffer_flags must be int64");
  TORCH_CHECK(full_to_swa_index_mapping.scalar_type() == at::kLong, "full_to_swa_index_mapping must be int64");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
  TORCH_CHECK(starts.scalar_type() == at::kLong, "starts must be int64");
  TORCH_CHECK(lengths.scalar_type() == at::kLong, "lengths must be int64");
  TORCH_CHECK(accum.scalar_type() == at::kInt, "accum must be int32");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(
      req_to_token.is_cuda() && accum.is_cuda() && out.is_cuda(), "req_to_token/accum/out must be CUDA tensors");
  TORCH_CHECK(
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda() && swa_buffer_flags.is_cuda() &&
          full_to_swa_index_mapping.is_cuda() && req_pool_indices.is_cuda() && starts.is_cuda() && lengths.is_cuda(),
      "all metadata tensors must be CUDA tensors");
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(swa_buffer_flags.is_contiguous(), "swa_buffer_flags must be contiguous");
  TORCH_CHECK(full_to_swa_index_mapping.is_contiguous(), "full_to_swa_index_mapping must be contiguous");
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
  TORCH_CHECK(
      row_strides.numel() == B && row_nbytes.numel() == B && swa_buffer_flags.numel() == B, "metadata length mismatch");
  // has_swa/is_capped are computed by the Python caller; the mapping must be
  // present whenever SWA is requested so the kernel never dereferences a null
  // mapping.
  TORCH_CHECK(!has_swa || full_to_swa_index_mapping.numel() > 0, "SWA checksum requires full_to_swa mapping");
  TORCH_CHECK(req_pool_indices.numel() == M && starts.numel() == M, "batch metadata length mismatch");
  TORCH_CHECK(accum.numel() >= M && out.numel() >= M, "accum/out must have at least M elements");
  if (M == 0 || B == 0) return;

  const int64_t max_len = max_num_tokens;
  // clang-format splits CUDA launch delimiters in this file into `> > >`, which nvcc rejects.
  // clang-format off
  if (max_len <= 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
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
  const size_t smem_bytes = static_cast<size_t>(B) * sizeof(BufMeta) +
                            static_cast<size_t>(warps_per_block) * sizeof(uint32_t);
  TORCH_CHECK(smem_bytes <= 48 * 1024, "too many KV buffers for shared-memory metadata staging");
  if (req_to_token.scalar_type() == at::kInt) {
    if (is_capped) {
      launch_kv_checksum_kernel<kBlock, true, int32_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int32_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          B,
          num_lanes,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
    } else {
      launch_kv_checksum_kernel<kBlock, false, int32_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int32_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          B,
          num_lanes,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
    }
  } else {
    if (is_capped) {
      launch_kv_checksum_kernel<kBlock, true, int64_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int64_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          B,
          num_lanes,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
    } else {
      launch_kv_checksum_kernel<kBlock, false, int64_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int64_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          B,
          num_lanes,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()));
    }
  }
  kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()),
      lengths.data_ptr<int64_t>(),
      M,
      out.data_ptr<int64_t>());
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}
