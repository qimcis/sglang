// Batched direct-KV transfer checksum for PD disaggregation.
//
// One warp hashes one logical token row directly from K/V cache buffers. Each
// thread hashes independent 8-byte chunks salted by logical token position and
// chunk offset; the batched serving op XOR-reduces rows into one checksum per
// request.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

// Murmur/XXH-style uint32 finalizer, matching `_fmix32_*` in kv_protection.py.
__device__ __forceinline__ uint32_t cksum_fmix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

// Must match `_CKSUM32_*` constants in kv_protection.py.
__device__ constexpr uint32_t kCksumSeed = 0x4E474353u;
__device__ constexpr uint32_t kPosMul = 0x9E3779B1u;
__device__ constexpr uint32_t kLaneMul = 0x85EBCA77u;
__device__ constexpr uint32_t kHiMul = 0xC2B2AE3Du;
// Independent second stream used only by the 64-bit page digest. The legacy
// request checksum continues to use the constants above unchanged.
__device__ constexpr uint32_t kPageSeedHi = 0xC4A35A71u;
__device__ constexpr uint32_t kPagePosMulHi = 0x27D4EB2Fu;
__device__ constexpr uint32_t kPageLaneMulHi = 0x165667B1u;
__device__ constexpr uint32_t kPageValueMulHi = 0x9E3779B9u;

__device__ __forceinline__ uint32_t page_chunk_hi32(uint64_t value, uint32_t seed_pos, uint64_t lane_index) {
  uint32_t h = seed_pos;
  h ^= static_cast<uint32_t>(lane_index) * kPageLaneMulHi;
  h ^= static_cast<uint32_t>(value) * kPageValueMulHi;
  h ^= static_cast<uint32_t>(value >> 32);
  return cksum_fmix32(h);
}

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
  int64_t nrows;
  int32_t nlanes;  // 8-byte lanes per row
  int32_t flags;
};

constexpr int32_t kFlagSwa = 1;
// Set when base pointer and row stride are both 16-byte aligned, so every row
// of the buffer can be read with 128-bit vector loads.
constexpr int32_t kFlagVec16 = 2;

template <int BLOCK, bool kCapped, bool kPageDigests, bool kDeriveRootFromPages, typename LocT>
__global__ void __launch_bounds__(BLOCK) kv_checksum_direct_table_batched_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const int64_t* __restrict__ row_strides,
    const int64_t* __restrict__ row_nbytes,
    const int64_t* __restrict__ buffer_num_rows,
    const int64_t* __restrict__ swa_buffer_flags,
    const int64_t* __restrict__ full_to_swa_index_mapping,
    const bool has_swa,
    const LocT* __restrict__ req_to_token,
    const int64_t req_to_token_stride0,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ logical_starts,
    int64_t req_to_token_rows,
    int64_t req_to_token_columns,
    int64_t max_num_tokens,
    int64_t full_to_swa_mapping_size,
    int B,
    int64_t cap,
    int64_t page_size,
    int64_t max_num_pages,
    uint32_t* __restrict__ accum,
    int64_t* __restrict__ out,
    uint64_t* __restrict__ page_accum) {
  extern __shared__ char smem[];
  constexpr int warps_per_block = BLOCK / 32;
  // Dynamic shared memory layout: metadata, root warp accumulators, then the
  // optional page digest accumulators and page indices.
  BufMeta* smeta = reinterpret_cast<BufMeta*>(smem);
  uint32_t* warp_acc = reinterpret_cast<uint32_t*>(smem + static_cast<size_t>(B) * sizeof(BufMeta));
  uint64_t* warp_page_acc = reinterpret_cast<uint64_t*>(warp_acc + warps_per_block);
  int32_t* warp_page_idx = reinterpret_cast<int32_t*>(warp_page_acc + warps_per_block);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int req = blockIdx.y;
  const int64_t n = lengths[req];
  const int64_t start = starts[req];
  const int64_t req_pool_idx = req_pool_indices[req];
  if (n < 0 || n > max_num_tokens || req_pool_idx < 0 || req_pool_idx >= req_to_token_rows || start < 0 ||
      start > req_to_token_columns || n > req_to_token_columns - start) {
    return;
  }
  // Uniform whole-block exit before any barrier.
  if (static_cast<int64_t>(blockIdx.x) * warps_per_block >= n) return;

  for (int b = threadIdx.x; b < B; b += BLOCK) {
    BufMeta m;
    const uint64_t ptr = buffer_ptrs[b];
    const int64_t stride = row_strides[b];
    m.base = reinterpret_cast<const char*>(ptr);
    m.stride = stride;
    m.nrows = buffer_num_rows[b];
    m.nlanes = static_cast<int32_t>(row_nbytes[b] >> 3);
    int32_t flags = (swa_buffer_flags[b] != 0) ? kFlagSwa : 0;
    if (((ptr | static_cast<uint64_t>(stride)) & 15u) == 0) flags |= kFlagVec16;
    m.flags = flags;
    smeta[b] = m;
  }
  if (threadIdx.x < warps_per_block) warp_acc[threadIdx.x] = 0;
  if constexpr (kPageDigests) {
    if (threadIdx.x < warps_per_block) {
      warp_page_acc[threadIdx.x] = 0;
      warp_page_idx[threadIdx.x] = -1;
    }
  }
  __syncthreads();

  const int row = blockIdx.x * warps_per_block + warp;
  if (row < n) {
    const int64_t position = start + row;
    const int64_t full_loc = static_cast<int64_t>(req_to_token[req_pool_idx * req_to_token_stride0 + position]);
    bool valid_loc = full_loc >= 0 && (!has_swa || full_loc < full_to_swa_mapping_size);
    // The SWA translation depends only on the token, not the buffer: look it
    // up once per row instead of once per (row, buffer).
    int64_t swa_loc = 0;
    if (valid_loc && has_swa) {
      swa_loc = full_to_swa_index_mapping[full_loc];
      // Unmapped SWA rows are out of the sliding window; slot 0 is the reserved
      // dummy row and is consistently zero on both source and destination.
      if (swa_loc < 0) swa_loc = 0;
    }
    if (valid_loc) {
      const uint32_t seed_pos = kCksumSeed ^ (static_cast<uint32_t>(position) * kPosMul);
      const uint32_t page_seed_pos = kPageSeedHi ^ (static_cast<uint32_t>(position) * kPagePosMulHi);

      uint64_t lane_cap = 0;
      if constexpr (kCapped) {
        lane_cap = (cap < 0) ? ~0ULL : static_cast<uint64_t>(cap);
      }
      uint64_t consumed = 0;
      uint32_t local = 0;
      uint32_t page_local_hi = 0;

      for (int b = 0; b < B; ++b) {
        if constexpr (kCapped) {
          if (consumed >= lane_cap) break;
        }
        const BufMeta m = smeta[b];
        const int64_t loc = (m.flags & kFlagSwa) ? swa_loc : full_loc;
        if (loc < 0 || loc >= m.nrows) {
          valid_loc = false;
          break;
        }
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
            if constexpr (kPageDigests) {
              page_local_hi ^= page_chunk_hi32(v.x, page_seed_pos, gl);
              page_local_hi ^= page_chunk_hi32(v.y, page_seed_pos, gl + 1);
            }
          }
          if ((limit & 1) && lane == 0) {
            const uint64_t* p = reinterpret_cast<const uint64_t*>(base);
            local ^= cksum_chunk32(p[limit - 1], seed_pos, consumed + static_cast<uint64_t>(limit - 1));
            if constexpr (kPageDigests) {
              page_local_hi ^=
                  page_chunk_hi32(p[limit - 1], page_seed_pos, consumed + static_cast<uint64_t>(limit - 1));
            }
          }
        } else {
          const uint64_t* p = reinterpret_cast<const uint64_t*>(base);
#pragma unroll 4
          for (int32_t j = lane; j < limit; j += 32) {
            local ^= cksum_chunk32(p[j], seed_pos, consumed + static_cast<uint64_t>(j));
            if constexpr (kPageDigests) {
              page_local_hi ^= page_chunk_hi32(p[j], page_seed_pos, consumed + static_cast<uint64_t>(j));
            }
          }
        }
        consumed += static_cast<uint64_t>(m.nlanes);
      }

      if (valid_loc) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          local ^= __shfl_xor_sync(0xffffffffu, local, offset);
          if constexpr (kPageDigests) {
            page_local_hi ^= __shfl_xor_sync(0xffffffffu, page_local_hi, offset);
          }
        }
        if (lane == 0) {
          warp_acc[warp] = local;
          if constexpr (kPageDigests) {
            const int64_t logical_page_start = logical_starts[req] / page_size;
            warp_page_idx[warp] = static_cast<int32_t>(position / page_size - logical_page_start);
            warp_page_acc[warp] = (static_cast<uint64_t>(page_local_hi) << 32) | local;
          }
        }
      }
    }
    if (!valid_loc && lane == 0) {
      atomicExch(reinterpret_cast<unsigned long long*>(out + req), ~0ULL);
    }
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
    if constexpr (!kDeriveRootFromPages) {
      if (lane == 0 && block_acc != 0) {
        atomicXor(reinterpret_cast<unsigned int*>(accum + req), static_cast<unsigned int>(block_acc));
      }
    }
    if constexpr (kPageDigests) {
      if (lane == 0) {
        int32_t current_page = -1;
        uint64_t current_acc = 0;
        for (int w = 0; w < warps_per_block; ++w) {
          const int32_t page = warp_page_idx[w];
          if (page < 0) continue;
          if (page != current_page && current_page >= 0 && current_page < max_num_pages) {
            atomicXor(
                reinterpret_cast<unsigned long long*>(page_accum + req * max_num_pages + current_page),
                static_cast<unsigned long long>(current_acc));
            current_acc = 0;
          }
          current_page = page;
          current_acc ^= warp_page_acc[w];
        }
        if (current_page >= 0 && current_page < max_num_pages && current_acc != 0) {
          atomicXor(
              reinterpret_cast<unsigned long long*>(page_accum + req * max_num_pages + current_page),
              static_cast<unsigned long long>(current_acc));
        }
      }
    }
  }
}

__global__ void kv_checksum_finalize_batched_kernel(
    const uint32_t* __restrict__ accum, const int64_t* __restrict__ lengths, int M, int64_t* __restrict__ out) {
  const int req = blockIdx.x * blockDim.x + threadIdx.x;
  if (req >= M) return;
  if (out[req] == -1) return;
  const uint32_t h = cksum_fmix32(kCksumSeed ^ accum[req] ^ static_cast<uint32_t>(lengths[req]));
  out[req] = static_cast<int64_t>(h);
}

__global__ void kv_checksum_validate_metadata_kernel(
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    int M,
    int64_t req_to_token_rows,
    int64_t req_to_token_columns,
    int64_t max_num_tokens,
    const int64_t* __restrict__ page_output_offsets,
    const int64_t* __restrict__ logical_starts,
    int64_t page_size,
    int64_t max_num_pages,
    int64_t page_out_size,
    int64_t* __restrict__ out) {
  const int req = blockIdx.x * blockDim.x + threadIdx.x;
  if (req >= M) return;
  const int64_t req_pool_idx = req_pool_indices[req];
  const int64_t start = starts[req];
  const int64_t length = lengths[req];
  bool invalid = length < 0 || length > max_num_tokens || req_pool_idx < 0 || req_pool_idx >= req_to_token_rows ||
                 start < 0 || start > req_to_token_columns || length > req_to_token_columns - start;
  int64_t required_pages = 0;
  if (logical_starts != nullptr) {
    const int64_t logical_start = logical_starts[req];
    const int64_t logical_page_start = logical_start / page_size;
    required_pages = length > 0 ? (start + length - 1) / page_size - logical_page_start + 1 : 0;
    invalid =
        invalid || logical_start < 0 || logical_start > start || required_pages < 0 || required_pages > max_num_pages;
  }
  if (page_output_offsets != nullptr) {
    const int64_t output_begin = page_output_offsets[req];
    const int64_t output_end = page_output_offsets[req + 1];
    invalid = invalid || output_begin < 0 || output_end < output_begin || output_end > page_out_size ||
              output_end - output_begin != required_pages || (req == 0 && output_begin != 0) ||
              (req == M - 1 && output_end != page_out_size);
  }
  if (invalid) {
    // Valid checksums are uint32 values represented in int64, so -1 cannot be
    // mistaken for a successful result by an asynchronous caller.
    out[req] = -1;
  }
}

__global__ void kv_checksum_finalize_pages_batched_kernel(
    const uint64_t* __restrict__ page_accum,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ logical_starts,
    int M,
    int64_t page_size,
    int64_t max_num_pages,
    int64_t* __restrict__ page_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(M) * max_num_pages;
  if (idx >= total) return;
  const int req = static_cast<int>(idx / max_num_pages);
  const int64_t page_offset = idx - static_cast<int64_t>(req) * max_num_pages;
  const int64_t page_begin = (logical_starts[req] / page_size + page_offset) * page_size;
  const int64_t scan_begin = starts[req];
  const int64_t scan_end = scan_begin + lengths[req];
  const int64_t begin = max(page_begin, scan_begin);
  const int64_t end = min(page_begin + page_size, scan_end);
  if (end <= begin) {
    page_out[idx] = 0;
    return;
  }
  const uint64_t raw = page_accum[idx];
  const uint32_t page_tokens = static_cast<uint32_t>(end - begin);
  const uint32_t lo = cksum_fmix32(kCksumSeed ^ static_cast<uint32_t>(raw) ^ page_tokens);
  const uint32_t hi = cksum_fmix32(kPageSeedHi ^ static_cast<uint32_t>(raw >> 32) ^ page_tokens);
  page_out[idx] = static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo);
}

__global__ void kv_checksum_finalize_requests_with_pages_kernel(
    const uint64_t* __restrict__ page_accum,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ logical_starts,
    const int64_t* __restrict__ page_output_offsets,
    int M,
    int64_t page_size,
    int64_t max_num_pages,
    int64_t page_out_size,
    uint32_t* __restrict__ accum,
    int64_t* __restrict__ out,
    int64_t* __restrict__ page_out) {
  const int req = blockIdx.x;
  if (req >= M) return;
  if (out[req] == -1) return;
  const int64_t scan_begin = starts[req];
  const int64_t scan_end = scan_begin + lengths[req];
  const int64_t logical_page_start = logical_starts[req] / page_size;
  const int64_t raw_first_page_offset = scan_begin / page_size - logical_page_start;
  const int64_t first_page_offset = raw_first_page_offset > 0 ? raw_first_page_offset : 0;
  const int64_t last_page_offset =
      lengths[req] > 0 ? min(max_num_pages, (scan_end - 1) / page_size - logical_page_start + 1) : first_page_offset;
  uint32_t root_raw = 0;
  for (int64_t page_offset = first_page_offset + threadIdx.x; page_offset < last_page_offset;
       page_offset += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(req) * max_num_pages + page_offset;
    const int64_t page_begin = (logical_page_start + page_offset) * page_size;
    const int64_t begin = max(page_begin, scan_begin);
    const int64_t end = min(page_begin + page_size, scan_end);
    if (end <= begin) continue;
    const uint64_t raw = page_accum[idx];
    root_raw ^= static_cast<uint32_t>(raw);
    const uint32_t page_tokens = static_cast<uint32_t>(end - begin);
    const uint32_t lo = cksum_fmix32(kCksumSeed ^ static_cast<uint32_t>(raw) ^ page_tokens);
    const uint32_t hi = cksum_fmix32(kPageSeedHi ^ static_cast<uint32_t>(raw >> 32) ^ page_tokens);
    const int64_t output_begin =
        page_output_offsets != nullptr ? page_output_offsets[req] : static_cast<int64_t>(req) * max_num_pages;
    const int64_t output_end =
        page_output_offsets != nullptr ? page_output_offsets[req + 1] : output_begin + max_num_pages;
    const int64_t output_idx = output_begin + page_offset;
    if (output_idx >= 0 && output_idx >= output_begin && output_idx < output_end && output_idx < page_out_size) {
      page_out[output_idx] = static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo);
    }
  }

  __shared__ uint32_t root_partials[256];
  root_partials[threadIdx.x] = root_raw;
  __syncthreads();
#pragma unroll
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) root_partials[threadIdx.x] ^= root_partials[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const uint32_t raw = root_partials[0];
    accum[req] = raw;
    out[req] = static_cast<int64_t>(cksum_fmix32(kCksumSeed ^ raw ^ static_cast<uint32_t>(lengths[req])));
  }
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
    const int64_t* __restrict__ buffer_num_rows,
    const int64_t* __restrict__ swa_buffer_flags,
    const int64_t* __restrict__ full_to_swa_index_mapping,
    bool has_swa,
    const LocT* __restrict__ req_to_token,
    int64_t req_to_token_stride0,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ logical_starts,
    int64_t req_to_token_rows,
    int64_t req_to_token_columns,
    int64_t max_num_tokens,
    int64_t full_to_swa_mapping_size,
    int B,
    int64_t num_lanes,
    int64_t page_size,
    int64_t max_num_pages,
    uint32_t* __restrict__ accum,
    int64_t* __restrict__ out,
    uint64_t* __restrict__ page_accum,
    bool with_page_digests,
    bool derive_root_from_pages) {
#define LAUNCH_CHECKSUM(PAGES, DERIVE_ROOT)                                          \
  kv_checksum_direct_table_batched_kernel<kBlock, kCapped, PAGES, DERIVE_ROOT, LocT> \
      <<<grid, kBlock, smem_bytes, stream>>>(                                        \
          buffer_ptrs,                                                               \
          row_strides,                                                               \
          row_nbytes,                                                                \
          buffer_num_rows,                                                           \
          swa_buffer_flags,                                                          \
          full_to_swa_index_mapping,                                                 \
          has_swa,                                                                   \
          req_to_token,                                                              \
          req_to_token_stride0,                                                      \
          req_pool_indices,                                                          \
          starts,                                                                    \
          lengths,                                                                   \
          logical_starts,                                                            \
          req_to_token_rows,                                                         \
          req_to_token_columns,                                                      \
          max_num_tokens,                                                            \
          full_to_swa_mapping_size,                                                  \
          B,                                                                         \
          num_lanes,                                                                 \
          page_size,                                                                 \
          max_num_pages,                                                             \
          accum,                                                                     \
          out,                                                                       \
          page_accum)
  if (with_page_digests) {
    if (derive_root_from_pages) {
      LAUNCH_CHECKSUM(true, true);
    } else {
      LAUNCH_CHECKSUM(true, false);
    }
  } else {
    LAUNCH_CHECKSUM(false, false);
  }
#undef LAUNCH_CHECKSUM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void kv_checksum_direct_table_batched_impl(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& buffer_num_rows,
    const at::Tensor& swa_buffer_flags,
    const at::Tensor& full_to_swa_index_mapping,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& starts,
    const at::Tensor& lengths,
    const at::Tensor& logical_starts,
    const at::Tensor& page_output_offsets,
    int64_t max_num_tokens,
    int64_t num_lanes,
    int64_t page_size,
    int64_t max_num_pages,
    bool has_swa,
    bool is_capped,
    at::Tensor& accum,
    at::Tensor& out,
    at::Tensor& page_accum,
    at::Tensor& page_out,
    bool with_page_digests) {
#if defined(USE_ROCM) || defined(USE_MUSA)
  TORCH_CHECK(false, "kv_checksum_direct_table_batched is CUDA-only and is not supported on ROCm/MUSA");
#else
  TORCH_CHECK(buffer_ptrs.scalar_type() == at::kLong, "buffer_ptrs must be int64");
  TORCH_CHECK(row_strides.scalar_type() == at::kLong, "row_strides must be int64");
  TORCH_CHECK(row_nbytes.scalar_type() == at::kLong, "row_nbytes must be int64");
  TORCH_CHECK(buffer_num_rows.scalar_type() == at::kLong, "buffer_num_rows must be int64");
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
      buffer_ptrs.is_cuda() && row_strides.is_cuda() && row_nbytes.is_cuda() && buffer_num_rows.is_cuda() &&
          swa_buffer_flags.is_cuda() && full_to_swa_index_mapping.is_cuda() && req_pool_indices.is_cuda() &&
          starts.is_cuda() && lengths.is_cuda(),
      "all metadata tensors must be CUDA tensors");
  const auto device = req_to_token.device();
  TORCH_CHECK(
      buffer_ptrs.device() == device && row_strides.device() == device && row_nbytes.device() == device &&
          buffer_num_rows.device() == device && swa_buffer_flags.device() == device &&
          full_to_swa_index_mapping.device() == device && req_pool_indices.device() == device &&
          starts.device() == device && lengths.device() == device && accum.device() == device && out.device() == device,
      "all checksum tensors must be on the same CUDA device");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(req_to_token));
  TORCH_CHECK(buffer_ptrs.is_contiguous(), "buffer_ptrs must be contiguous");
  TORCH_CHECK(row_strides.is_contiguous(), "row_strides must be contiguous");
  TORCH_CHECK(row_nbytes.is_contiguous(), "row_nbytes must be contiguous");
  TORCH_CHECK(buffer_num_rows.is_contiguous(), "buffer_num_rows must be contiguous");
  TORCH_CHECK(swa_buffer_flags.is_contiguous(), "swa_buffer_flags must be contiguous");
  TORCH_CHECK(full_to_swa_index_mapping.is_contiguous(), "full_to_swa_index_mapping must be contiguous");
  TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
  TORCH_CHECK(starts.is_contiguous(), "starts must be contiguous");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(accum.is_contiguous(), "accum must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be a 2D table");
  TORCH_CHECK(req_to_token.stride(1) == 1, "req_to_token rows must be contiguous");
  TORCH_CHECK(
      req_to_token.scalar_type() == at::kInt || req_to_token.scalar_type() == at::kLong,
      "req_to_token must be int32 or int64");

  TORCH_CHECK(
      buffer_ptrs.numel() <= std::numeric_limits<int>::max() && lengths.numel() <= std::numeric_limits<int>::max(),
      "checksum metadata exceeds supported int32 indexing");
  const int B = static_cast<int>(buffer_ptrs.numel());
  const int M = static_cast<int>(lengths.numel());
  TORCH_CHECK(M <= 65535, "checksum batch exceeds CUDA grid-y limit");
  TORCH_CHECK(B > 0 || M == 0, "checksum requires at least one KV buffer");
  TORCH_CHECK(
      max_num_tokens >= 0 && max_num_tokens <= std::numeric_limits<int>::max(),
      "max_num_tokens exceeds supported range");
  TORCH_CHECK(
      row_strides.numel() == B && row_nbytes.numel() == B && buffer_num_rows.numel() == B &&
          swa_buffer_flags.numel() == B,
      "metadata length mismatch");
  // has_swa/is_capped are computed by the Python caller; the mapping must be
  // present whenever SWA is requested so the kernel never dereferences a null
  // mapping.
  TORCH_CHECK(!has_swa || full_to_swa_index_mapping.numel() > 0, "SWA checksum requires full_to_swa mapping");
  TORCH_CHECK(req_pool_indices.numel() == M && starts.numel() == M, "batch metadata length mismatch");
  TORCH_CHECK(accum.numel() >= M && out.numel() >= M, "accum/out must have at least M elements");
  if (with_page_digests) {
    TORCH_CHECK(logical_starts.scalar_type() == at::kLong, "logical_starts must be int64");
    TORCH_CHECK(logical_starts.is_cuda() && logical_starts.is_contiguous(), "logical_starts must be contiguous CUDA");
    TORCH_CHECK(logical_starts.numel() == M, "logical_starts length mismatch");
    if (page_output_offsets.defined()) {
      TORCH_CHECK(page_output_offsets.scalar_type() == at::kLong, "page_output_offsets must be int64");
      TORCH_CHECK(
          page_output_offsets.is_cuda() && page_output_offsets.is_contiguous(),
          "page_output_offsets must be contiguous CUDA");
      TORCH_CHECK(page_output_offsets.numel() == M + 1, "page_output_offsets length mismatch");
    }
    TORCH_CHECK(page_size > 0, "page_size must be positive");
    TORCH_CHECK(max_num_pages > 0, "max_num_pages must be positive");
    TORCH_CHECK(
        M == 0 || max_num_pages <= std::numeric_limits<int64_t>::max() / M,
        "page checksum output size overflows int64");
    TORCH_CHECK(page_accum.scalar_type() == at::kLong, "page_accum must be int64");
    TORCH_CHECK(page_out.scalar_type() == at::kLong, "page_out must be int64");
    TORCH_CHECK(page_accum.is_cuda() && page_out.is_cuda(), "page_accum/page_out must be CUDA tensors");
    TORCH_CHECK(
        logical_starts.device() == device && page_accum.device() == device && page_out.device() == device,
        "all page checksum tensors must be on the same CUDA device");
    TORCH_CHECK(
        !page_output_offsets.defined() || page_output_offsets.device() == device,
        "page_output_offsets must be on the same CUDA device");
    TORCH_CHECK(page_accum.is_contiguous() && page_out.is_contiguous(), "page_accum/page_out must be contiguous");
    TORCH_CHECK(page_accum.numel() >= static_cast<int64_t>(M) * max_num_pages, "page_accum is too small");
    TORCH_CHECK(
        page_output_offsets.defined() || page_out.numel() >= static_cast<int64_t>(M) * max_num_pages,
        "dense page_out is too small");
  }
  if (M == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(accum.data_ptr<int32_t>(), 0, static_cast<size_t>(M) * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<int64_t>(), 0, static_cast<size_t>(M) * sizeof(int64_t), stream));
  if (with_page_digests) {
    const int64_t page_slots = static_cast<int64_t>(M) * max_num_pages;
    C10_CUDA_CHECK(
        cudaMemsetAsync(page_accum.data_ptr<int64_t>(), 0, static_cast<size_t>(page_slots) * sizeof(int64_t), stream));
  }

  const int64_t max_len = max_num_tokens;
  // clang-format splits CUDA launch delimiters in this file into `> > >`, which nvcc rejects.
  // clang-format off
  if (max_len <= 0) {
    if (with_page_digests) {
      if (page_output_offsets.defined()) {
        kv_checksum_finalize_requests_with_pages_kernel<<<M, 256, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(page_accum.data_ptr<int64_t>()),
            starts.data_ptr<int64_t>(), lengths.data_ptr<int64_t>(), logical_starts.data_ptr<int64_t>(),
            page_output_offsets.data_ptr<int64_t>(), M, page_size, max_num_pages, page_out.numel(),
            reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()), out.data_ptr<int64_t>(), page_out.data_ptr<int64_t>());
      } else {
        kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()), lengths.data_ptr<int64_t>(), M,
            out.data_ptr<int64_t>());
        const int64_t total_pages = static_cast<int64_t>(M) * max_num_pages;
        kv_checksum_finalize_pages_batched_kernel<<<(total_pages + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(page_accum.data_ptr<int64_t>()), starts.data_ptr<int64_t>(),
            lengths.data_ptr<int64_t>(), logical_starts.data_ptr<int64_t>(), M, page_size, max_num_pages,
            page_out.data_ptr<int64_t>());
      }
    } else {
      kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()),
          lengths.data_ptr<int64_t>(),
          M,
          out.data_ptr<int64_t>());
    }
    kv_checksum_validate_metadata_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
        req_pool_indices.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), lengths.data_ptr<int64_t>(), M,
        req_to_token.size(0), req_to_token.size(1), max_num_tokens,
        page_output_offsets.defined() ? page_output_offsets.data_ptr<int64_t>() : nullptr,
        with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr, page_size, max_num_pages,
        page_output_offsets.defined() ? page_out.numel() : 0,
        out.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  constexpr int kBlock = 256;
  const int warps_per_block = kBlock / 32;
  const int grid_x = (static_cast<int>(max_len) + warps_per_block - 1) / warps_per_block;
  const dim3 grid(grid_x, M);
  const int64_t stride0 = req_to_token.stride(0);
  const size_t smem_bytes = static_cast<size_t>(B) * sizeof(BufMeta) +
                            static_cast<size_t>(warps_per_block) * sizeof(uint32_t) +
                            (with_page_digests
                                 ? static_cast<size_t>(warps_per_block) * (sizeof(uint64_t) + sizeof(int32_t))
                                 : 0);
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
          buffer_num_rows.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int32_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr,
          req_to_token.size(0),
          req_to_token.size(1),
          max_num_tokens,
          full_to_swa_index_mapping.numel(),
          B,
          num_lanes,
          page_size,
          max_num_pages,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()),
          out.data_ptr<int64_t>(),
          with_page_digests ? reinterpret_cast<uint64_t*>(page_accum.data_ptr<int64_t>()) : nullptr,
          with_page_digests,
          page_output_offsets.defined());
    } else {
      launch_kv_checksum_kernel<kBlock, false, int32_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          buffer_num_rows.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int32_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr,
          req_to_token.size(0),
          req_to_token.size(1),
          max_num_tokens,
          full_to_swa_index_mapping.numel(),
          B,
          num_lanes,
          page_size,
          max_num_pages,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()),
          out.data_ptr<int64_t>(),
          with_page_digests ? reinterpret_cast<uint64_t*>(page_accum.data_ptr<int64_t>()) : nullptr,
          with_page_digests,
          page_output_offsets.defined());
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
          buffer_num_rows.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int64_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr,
          req_to_token.size(0),
          req_to_token.size(1),
          max_num_tokens,
          full_to_swa_index_mapping.numel(),
          B,
          num_lanes,
          page_size,
          max_num_pages,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()),
          out.data_ptr<int64_t>(),
          with_page_digests ? reinterpret_cast<uint64_t*>(page_accum.data_ptr<int64_t>()) : nullptr,
          with_page_digests,
          page_output_offsets.defined());
    } else {
      launch_kv_checksum_kernel<kBlock, false, int64_t>(
          grid,
          smem_bytes,
          stream,
          reinterpret_cast<const uint64_t*>(buffer_ptrs.data_ptr<int64_t>()),
          row_strides.data_ptr<int64_t>(),
          row_nbytes.data_ptr<int64_t>(),
          buffer_num_rows.data_ptr<int64_t>(),
          swa_buffer_flags.data_ptr<int64_t>(),
          full_to_swa_index_mapping.data_ptr<int64_t>(),
          has_swa,
          req_to_token.data_ptr<int64_t>(),
          stride0,
          req_pool_indices.data_ptr<int64_t>(),
          starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(),
          with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr,
          req_to_token.size(0),
          req_to_token.size(1),
          max_num_tokens,
          full_to_swa_index_mapping.numel(),
          B,
          num_lanes,
          page_size,
          max_num_pages,
          reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()),
          out.data_ptr<int64_t>(),
          with_page_digests ? reinterpret_cast<uint64_t*>(page_accum.data_ptr<int64_t>()) : nullptr,
          with_page_digests,
          page_output_offsets.defined());
    }
  }
  if (with_page_digests) {
    if (page_output_offsets.defined()) {
      kv_checksum_finalize_requests_with_pages_kernel<<<M, 256, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(page_accum.data_ptr<int64_t>()), starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(), logical_starts.data_ptr<int64_t>(), page_output_offsets.data_ptr<int64_t>(), M,
          page_size, max_num_pages, page_out.numel(), reinterpret_cast<uint32_t*>(accum.data_ptr<int32_t>()),
          out.data_ptr<int64_t>(), page_out.data_ptr<int64_t>());
    } else {
      kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()), lengths.data_ptr<int64_t>(), M,
          out.data_ptr<int64_t>());
      const int64_t total_pages = static_cast<int64_t>(M) * max_num_pages;
      kv_checksum_finalize_pages_batched_kernel<<<(total_pages + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(page_accum.data_ptr<int64_t>()), starts.data_ptr<int64_t>(),
          lengths.data_ptr<int64_t>(), logical_starts.data_ptr<int64_t>(), M, page_size, max_num_pages,
          page_out.data_ptr<int64_t>());
    }
  } else {
    kv_checksum_finalize_batched_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(accum.data_ptr<int32_t>()),
        lengths.data_ptr<int64_t>(),
        M,
        out.data_ptr<int64_t>());
  }
  kv_checksum_validate_metadata_kernel<<<(M + 255) / 256, 256, 0, stream>>>(
      req_pool_indices.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), lengths.data_ptr<int64_t>(), M,
      req_to_token.size(0), req_to_token.size(1), max_num_tokens,
      page_output_offsets.defined() ? page_output_offsets.data_ptr<int64_t>() : nullptr,
      with_page_digests ? logical_starts.data_ptr<int64_t>() : nullptr, page_size, max_num_pages,
      page_output_offsets.defined() ? page_out.numel() : 0,
      out.data_ptr<int64_t>());
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void kv_checksum_direct_table_batched(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& buffer_num_rows,
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
  at::Tensor unused;
  kv_checksum_direct_table_batched_impl(
      buffer_ptrs,
      row_strides,
      row_nbytes,
      buffer_num_rows,
      swa_buffer_flags,
      full_to_swa_index_mapping,
      req_to_token,
      req_pool_indices,
      starts,
      lengths,
      unused,
      unused,
      max_num_tokens,
      num_lanes,
      0,
      0,
      has_swa,
      is_capped,
      accum,
      out,
      unused,
      unused,
      false);
}

void kv_checksum_direct_table_batched_with_pages(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& buffer_num_rows,
    const at::Tensor& swa_buffer_flags,
    const at::Tensor& full_to_swa_index_mapping,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& starts,
    const at::Tensor& lengths,
    const at::Tensor& logical_starts,
    int64_t max_num_tokens,
    int64_t num_lanes,
    int64_t page_size,
    int64_t max_num_pages,
    bool has_swa,
    bool is_capped,
    at::Tensor& accum,
    at::Tensor& out,
    at::Tensor& page_accum,
    at::Tensor& page_out) {
  at::Tensor unused;
  kv_checksum_direct_table_batched_impl(
      buffer_ptrs,
      row_strides,
      row_nbytes,
      buffer_num_rows,
      swa_buffer_flags,
      full_to_swa_index_mapping,
      req_to_token,
      req_pool_indices,
      starts,
      lengths,
      logical_starts,
      unused,
      max_num_tokens,
      num_lanes,
      page_size,
      max_num_pages,
      has_swa,
      is_capped,
      accum,
      out,
      page_accum,
      page_out,
      true);
}

void kv_checksum_direct_table_batched_with_pages_compact(
    const at::Tensor& buffer_ptrs,
    const at::Tensor& row_strides,
    const at::Tensor& row_nbytes,
    const at::Tensor& buffer_num_rows,
    const at::Tensor& swa_buffer_flags,
    const at::Tensor& full_to_swa_index_mapping,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& starts,
    const at::Tensor& lengths,
    const at::Tensor& logical_starts,
    const at::Tensor& page_output_offsets,
    int64_t max_num_tokens,
    int64_t num_lanes,
    int64_t page_size,
    int64_t max_num_pages,
    bool has_swa,
    bool is_capped,
    at::Tensor& accum,
    at::Tensor& out,
    at::Tensor& page_accum,
    at::Tensor& page_out) {
  kv_checksum_direct_table_batched_impl(
      buffer_ptrs,
      row_strides,
      row_nbytes,
      buffer_num_rows,
      swa_buffer_flags,
      full_to_swa_index_mapping,
      req_to_token,
      req_pool_indices,
      starts,
      lengths,
      logical_starts,
      page_output_offsets,
      max_num_tokens,
      num_lanes,
      page_size,
      max_num_pages,
      has_swa,
      is_capped,
      accum,
      out,
      page_accum,
      page_out,
      true);
}
