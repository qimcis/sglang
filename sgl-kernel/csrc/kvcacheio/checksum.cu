#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <cstdint>

namespace {

constexpr uint64_t kSplitmixAdd = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kSplitmixM1 = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t kSplitmixM2 = 0x94D049BB133111EBull;
constexpr uint64_t kChecksumSeed = 0x53474C414E474353ull;  // "SGLANGCS"

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += kSplitmixAdd;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * kSplitmixM1;
  z = (z ^ (z >> 27)) * kSplitmixM2;
  z = z ^ (z >> 31);
  return z;
}

__device__ __forceinline__ uint64_t mix(uint64_t acc, uint64_t field) {
  return splitmix64(acc ^ field);
}

__device__ __forceinline__ uint64_t load_lane_padded(const uint8_t* row, int64_t row_bytes, int64_t lane) {
  const int64_t offset = lane * 8;
  if (offset + 8 <= row_bytes) {
    return *reinterpret_cast<const uint64_t*>(row + offset);
  }
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    const int64_t pos = offset + i;
    if (pos < row_bytes) {
      value |= static_cast<uint64_t>(row[pos]) << (8 * i);
    }
  }
  return value;
}

__global__ void kv_checksum_kernel(
    const uint8_t* __restrict__ rows,
    const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ positions,
    int64_t num_selected,
    int64_t row_bytes,
    int64_t num_lanes,
    bool include_positions,
    unsigned long long* __restrict__ out) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_selected) {
    return;
  }

  const int64_t row_id = row_indices[i];
  const uint8_t* row = rows + row_id * row_bytes;
  uint64_t acc = kChecksumSeed;
  if (include_positions) {
    acc = mix(acc, static_cast<uint64_t>(positions[i]));
  }
  for (int64_t lane = 0; lane < num_lanes; ++lane) {
    acc = mix(acc, load_lane_padded(row, row_bytes, lane));
  }
  atomicXor(out, static_cast<unsigned long long>(acc));
}

__global__ void finalize_checksum_kernel(unsigned long long* __restrict__ out, int64_t num_selected) {
  uint64_t total = mix(kChecksumSeed, static_cast<uint64_t>(*out));
  total = mix(total, static_cast<uint64_t>(num_selected));
  *out = static_cast<unsigned long long>(total);
}

uint64_t splitmix64_host(uint64_t x) {
  x += kSplitmixAdd;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * kSplitmixM1;
  z = (z ^ (z >> 27)) * kSplitmixM2;
  z = z ^ (z >> 31);
  return z;
}

}  // namespace

torch::Tensor kv_checksum(
    const torch::Tensor& rows,
    const torch::Tensor& row_indices,
    const torch::Tensor& positions,
    int64_t num_lanes,
    bool include_positions) {
  TORCH_CHECK(rows.is_cuda(), "kv_checksum: rows must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "kv_checksum: row_indices must be a CUDA tensor");
  TORCH_CHECK(row_indices.scalar_type() == torch::kInt64, "kv_checksum: row_indices must be int64");
  TORCH_CHECK(rows.is_contiguous(), "kv_checksum: rows must be contiguous");
  TORCH_CHECK(rows.dim() >= 1, "kv_checksum: rows must have at least one dimension");
  if (include_positions) {
    TORCH_CHECK(positions.is_cuda(), "kv_checksum: positions must be a CUDA tensor when include_positions=true");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64, "kv_checksum: positions must be int64");
    TORCH_CHECK(positions.numel() == row_indices.numel(), "kv_checksum: positions length must match row_indices");
  }

  auto out = torch::empty({1}, rows.options().dtype(torch::kInt64));
  const int64_t num_selected = row_indices.numel();
  if (num_selected == 0) {
    out.fill_(static_cast<int64_t>(splitmix64_host(kChecksumSeed)));
    return out;
  }

  const int64_t num_rows = rows.size(0);
  TORCH_CHECK(num_rows > 0, "kv_checksum: rows must have at least one row when row_indices is non-empty");
  const int64_t row_bytes = rows.numel() / num_rows * rows.element_size();
  const int64_t max_lanes = (row_bytes + 7) / 8;
  if (num_lanes < 0 || num_lanes > max_lanes) {
    num_lanes = max_lanes;
  }
  TORCH_CHECK(num_lanes > 0, "kv_checksum: num_lanes must resolve to a positive value");

  auto stream = at::cuda::getCurrentCUDAStream(rows.device().index());
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<int64_t>(), 0, sizeof(int64_t), stream.stream()));

  const int threads = 256;
  const int blocks = static_cast<int>((num_selected + threads - 1) / threads);
  kv_checksum_kernel<<<blocks, threads, 0, stream.stream()>>>(
      reinterpret_cast<const uint8_t*>(rows.data_ptr()),
      row_indices.data_ptr<int64_t>(),
      include_positions ? positions.data_ptr<int64_t>() : nullptr,
      num_selected,
      row_bytes,
      num_lanes,
      include_positions,
      reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  finalize_checksum_kernel<<<1, 1, 0, stream.stream()>>>(
      reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()), num_selected);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
