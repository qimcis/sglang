// Fused KV transfer checksum kernel for PD disaggregation.
//
// This op reproduces, bit-for-bit, the Python/Torch reference in
// python/sglang/srt/mem_cache/kv_page_tags.py
// (`hash_rows_with_positions` / `hash_kv_rows`):
//
//   splitmix64(x):
//     x += 0x9E3779B97F4A7C15
//     z  = x
//     z  = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
//     z  = (z ^ (z >> 27)) * 0x94D049BB133111EB
//     z  = z ^ (z >> 31)
//   mix(acc, field) = splitmix64(acc ^ field)
//
//   For each selected logical row i (= row_indices[t]):
//     acc = CKSUM_SEED
//     if positions:  acc = mix(acc, positions[t])
//     for lane j in [0, num_lanes): acc = mix(acc, lane_j(row i))
//   combined = XOR_t acc_t            <-- returned by this kernel
//
// The two scalar finishing mixes
//     total = mix(CKSUM_SEED, combined); total = mix(total, num_rows)
// are applied in the Python wrapper so the kernel only needs to return one
// 64-bit value and the wrapper owns all "scalar" parity arithmetic.
//
// Rows are reinterpreted as little-endian int64 lanes exactly like
// `_as_int64_lanes`: a row of `row_bytes` bytes yields ceil(row_bytes/8) lanes;
// the trailing partial lane is zero-padded to 8 bytes. Physical page ids are
// never read here -- the caller passes logical rows / logical positions only.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

namespace {

// splitmix64 constants (same bit patterns as the Python reference).
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

__device__ __forceinline__ uint64_t mix(uint64_t acc, uint64_t field) {
  return splitmix64(acc ^ field);
}

// _CKSUM_SEED = 0x5347_4C41_4E47_4353
constexpr uint64_t kCksumSeed = 0x53474C414E474353ULL;

// Load little-endian int64 lane `j` from a byte row of length `row_bytes`,
// zero-padding bytes beyond the row (matches the uint8->int64 view + pad).
__device__ __forceinline__ uint64_t load_lane(const uint8_t* row, int64_t row_bytes, int64_t j) {
  const int64_t off = j * 8;
  if (off + 8 <= row_bytes) {
    const uint8_t* p = row + off;
    if ((reinterpret_cast<uintptr_t>(p) & 7ULL) == 0) {
      // Aligned full lane: single 64-bit load.
      return *reinterpret_cast<const uint64_t*>(p);
    }
    uint64_t v = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      v |= static_cast<uint64_t>(p[k]) << (8 * k);
    }
    return v;
  }
  // Trailing partial lane: assemble in-row bytes, zero-pad the rest.
  uint64_t v = 0;
  for (int k = 0; off + k < row_bytes; ++k) {
    v |= static_cast<uint64_t>(row[off + k]) << (8 * k);
  }
  return v;
}

__global__ void kv_checksum_kernel(
    const uint8_t* __restrict__ rows,
    const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ positions,  // may be nullptr
    int64_t num_rows,
    int64_t row_bytes,
    int64_t num_lanes,
    unsigned long long* __restrict__ out) {
  extern __shared__ uint64_t sdata[];
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

  // Out-of-range threads contribute the XOR identity (0) so the block-level
  // reduction below is correct for partial (tail) blocks without special cases.
  uint64_t acc = 0;
  if (t < num_rows) {
    const int64_t idx = row_indices[t];
    const uint8_t* row = rows + idx * row_bytes;
    acc = kCksumSeed;
    if (positions != nullptr) {
      acc = mix(acc, static_cast<uint64_t>(positions[t]));
    }
    for (int64_t j = 0; j < num_lanes; ++j) {
      acc = mix(acc, load_lane(row, row_bytes, j));
    }
  }

  // XOR is commutative/associative, so a tree reduction in any order is exact.
  sdata[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] ^= sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicXor(out, static_cast<unsigned long long>(sdata[0]));
  }
}

}  // namespace

// Writes a 1-element int64 CUDA tensor holding `combined = XOR_t acc_t`.
void kv_checksum(
    const at::Tensor& rows,
    const at::Tensor& row_indices,
    const std::optional<at::Tensor>& positions,
    int64_t num_lanes,
    at::Tensor& out) {
  TORCH_CHECK(rows.is_cuda(), "kv_checksum: rows must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "kv_checksum: row_indices must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "kv_checksum: out must be a CUDA tensor");
  TORCH_CHECK(rows.is_contiguous(), "kv_checksum: rows must be contiguous");
  TORCH_CHECK(row_indices.is_contiguous(), "kv_checksum: row_indices must be contiguous");
  TORCH_CHECK(row_indices.scalar_type() == at::kLong, "kv_checksum: row_indices must be int64");
  TORCH_CHECK(out.scalar_type() == at::kLong && out.numel() == 1, "kv_checksum: out must be a 1-element int64 tensor");
  TORCH_CHECK(rows.dim() >= 1, "kv_checksum: rows must be at least 1D");

  const int64_t num_total_rows = rows.size(0);
  const int64_t num_rows = row_indices.numel();

  if (num_rows == 0 || num_total_rows == 0) {
    return;
  }

  // row_bytes = total bytes / number of rows (rows is contiguous).
  const int64_t row_bytes = rows.nbytes() / num_total_rows;
  const int64_t total_lanes = (row_bytes + 7) / 8;
  int64_t lanes = (num_lanes < 0) ? total_lanes : std::min<int64_t>(num_lanes, total_lanes);
  if (lanes < 0) {
    lanes = 0;
  }

  const int64_t* positions_ptr = nullptr;
  at::Tensor positions_c;
  if (positions.has_value() && positions->defined() && positions->numel() > 0) {
    positions_c = positions->contiguous();
    TORCH_CHECK(positions_c.scalar_type() == at::kLong, "kv_checksum: positions must be int64");
    TORCH_CHECK(positions_c.numel() == num_rows, "kv_checksum: positions/row_indices length mismatch");
    positions_ptr = positions_c.data_ptr<int64_t>();
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(), 0, sizeof(int64_t), stream));
  const int threads = 256;  // power of two: required by the tree reduction
  const int64_t blocks = (num_rows + threads - 1) / threads;
  const size_t smem = static_cast<size_t>(threads) * sizeof(uint64_t);

  kv_checksum_kernel<<<static_cast<unsigned int>(blocks), threads, smem, stream>>>(
      reinterpret_cast<const uint8_t*>(rows.data_ptr()),
      row_indices.data_ptr<int64_t>(),
      positions_ptr,
      num_rows,
      row_bytes,
      lanes,
      reinterpret_cast<unsigned long long*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}
