#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace {

constexpr int32_t kInvalidMapping = 1 << 0;
constexpr int32_t kOwnerMismatch = 1 << 1;
constexpr int32_t kPositionMismatch = 1 << 2;
constexpr int32_t kAttentionTagMismatch = 1 << 3;
constexpr int32_t kGenerationMismatch = 1 << 4;
constexpr int32_t kTransferTagMismatch = 1 << 5;
constexpr int32_t kThreads = 256;

struct ProtectionParams {
  const int64_t* request_indices;
  int32_t* seqlens;
  int32_t* page_table;
  int64_t page_table_stride;
  int32_t page_table_cols;
  int32_t* page_table_2;
  int64_t page_table_2_stride;
  int32_t page_table_2_cols;
  int32_t page_table_page_offset;
  int32_t page_table_2_page_offset;
  int32_t page_table_expected_mapping_offset;
  int32_t page_table_2_expected_mapping_offset;
  int32_t page_table_2_window_size;
  int32_t page_size;
  bool cache_validated_epochs;
  int32_t num_sidecar_pages;
  int32_t num_request_slots;
  int32_t expected_mapping_stride;
  int32_t expected_mapping_namespace_stride;
  const int64_t* actual_tags;
  const int64_t* actual_generations;
  const int32_t* actual_transfer_tags;
  const int32_t* expected_physical_pages;
  const int64_t* expected_tags;
  const int64_t* expected_generations;
  const int32_t* expected_transfer_tags;
  const int32_t* request_epochs;
  int32_t* validated_epochs;
  int32_t* status;
};

__device__ int32_t validate_table(
    const ProtectionParams& p,
    int32_t row,
    int32_t request_idx,
    int64_t num_pages,
    int32_t* table,
    int64_t table_stride,
    int32_t table_cols,
    int32_t page_offset,
    int32_t expected_mapping_offset,
    int64_t first_page) {
  int32_t result = 0;
  if (first_page < 0 || first_page > num_pages || num_pages > table_cols) {
    return kInvalidMapping;
  }
  for (int64_t position = first_page + threadIdx.x; position < num_pages; position += blockDim.x) {
    const int32_t local_page = table[static_cast<int64_t>(row) * table_stride + position];
    const int64_t page = local_page > 0 ? static_cast<int64_t>(local_page) + page_offset : local_page;
    if (page <= 0 || page >= p.num_sidecar_pages) {
      result |= kInvalidMapping;
      continue;
    }
    if (position >= p.expected_mapping_namespace_stride) {
      result |= kPositionMismatch;
      continue;
    }
    const int64_t expected_idx =
        static_cast<int64_t>(request_idx) * p.expected_mapping_stride + expected_mapping_offset + position;
    if (p.expected_physical_pages[expected_idx] != page) result |= kOwnerMismatch;
    if (p.actual_tags[page] != p.expected_tags[expected_idx]) result |= kAttentionTagMismatch;
    if (p.actual_generations[page] != p.expected_generations[expected_idx]) result |= kGenerationMismatch;
    const int32_t expected_transfer = p.expected_transfer_tags[expected_idx];
    if (expected_transfer != 0 && p.actual_transfer_tags[page] != expected_transfer) {
      result |= kTransferTagMismatch;
    }
  }
  return result;
}

__device__ void sanitize_row(const ProtectionParams& p, int32_t row) {
  for (int32_t col = threadIdx.x; col < p.page_table_cols; col += blockDim.x) {
    p.page_table[static_cast<int64_t>(row) * p.page_table_stride + col] = 0;
  }
  if (p.page_table_2 != nullptr) {
    for (int32_t col = threadIdx.x; col < p.page_table_2_cols; col += blockDim.x) {
      p.page_table_2[static_cast<int64_t>(row) * p.page_table_2_stride + col] = 0;
    }
  }
  if (threadIdx.x == 0) {
    // FA4's zero-length behavior is not a safety boundary. Page 0 is reserved,
    // so a one-token row guarantees a bounded, harmless external-kernel read.
    p.seqlens[row] = 1;
  }
}

__global__ void kv_page_protection_preflight_kernel(ProtectionParams p, int32_t batch_size) {
  const int32_t row = blockIdx.x;
  if (row >= batch_size) return;

  const int64_t request_idx_64 = p.request_indices[row];
  if (request_idx_64 == 0) {
    // Replay can leave stale values in graph-padding rows. Slot 0 is outside
    // the epoch ABI, so restore its sentinel on every preflight.
    sanitize_row(p, row);
    return;
  }
  if (request_idx_64 < 0 || request_idx_64 >= p.num_request_slots) {
    sanitize_row(p, row);
    if (threadIdx.x == 0) atomicOr(p.status, kInvalidMapping);
    return;
  }
  const int32_t request_idx = static_cast<int32_t>(request_idx_64);
  const int32_t epoch = p.request_epochs[request_idx];
  if (p.cache_validated_epochs && p.validated_epochs[request_idx] == epoch) {
    // This row's forward buffers were either preserved or sanitized by the
    // first validation in the epoch. Later layers only pay this cached check.
    return;
  }

  __shared__ int32_t block_status;
  if (threadIdx.x == 0) block_status = 0;
  __syncthreads();

  const int32_t seqlen = p.seqlens[row];
  int32_t thread_status = 0;
  int64_t num_pages = 0;
  if (seqlen <= 0) {
    thread_status = kInvalidMapping;
  } else {
    num_pages = (static_cast<int64_t>(seqlen) + p.page_size - 1) / p.page_size;
    thread_status |= validate_table(
        p,
        row,
        request_idx,
        num_pages,
        p.page_table,
        p.page_table_stride,
        p.page_table_cols,
        p.page_table_page_offset,
        p.page_table_expected_mapping_offset,
        0);
    if (p.page_table_2 != nullptr) {
      const int64_t window_start = static_cast<int64_t>(seqlen) - p.page_table_2_window_size;
      const int64_t first_page =
          p.page_table_2_window_size > 0 ? (window_start > 0 ? window_start : 0) / p.page_size : 0;
      thread_status |= validate_table(
          p,
          row,
          request_idx,
          num_pages,
          p.page_table_2,
          p.page_table_2_stride,
          p.page_table_2_cols,
          p.page_table_2_page_offset,
          p.page_table_2_expected_mapping_offset,
          first_page);
    }
  }
  if (thread_status != 0) atomicOr(&block_status, thread_status);
  __syncthreads();

  if (block_status != 0) sanitize_row(p, row);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (block_status != 0) atomicOr(p.status + request_idx, block_status);
    if (p.cache_validated_epochs) {
      __threadfence();
      atomicExch(p.validated_epochs + request_idx, epoch);
    }
  }
}

void check_1d(const at::Tensor& tensor, const at::Tensor& reference, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(), name, " must be on the CUDA input device");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be contiguous and 1D");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

void check_table(const at::Tensor& tensor, const at::Tensor& reference, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(), name, " must be on the CUDA input device");
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
  TORCH_CHECK(tensor.dim() == 2 && tensor.stride(1) == 1, name, " must be a row-major 2D tensor");
}

int32_t checked_int(int64_t value, const char* name, bool positive = false) {
  TORCH_CHECK(
      value >= (positive ? 1 : 0) && value <= std::numeric_limits<int32_t>::max(), name, " is out of int32 range");
  return static_cast<int32_t>(value);
}

}  // namespace

bool kv_page_protection_preflight_supported() {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  if (properties == nullptr ||
      !((properties->major == 9 && properties->minor == 0) || (properties->major == 10 && properties->minor == 0) ||
        (properties->major == 10 && properties->minor == 3))) {
    return false;
  }
  cudaFuncAttributes attributes{};
  const auto result = cudaFuncGetAttributes(&attributes, kv_page_protection_preflight_kernel);
  if (result == cudaSuccess) return true;
  cudaGetLastError();
  return false;
}

void kv_page_protection_preflight(
    const at::Tensor& request_indices,
    at::Tensor& seqlens,
    at::Tensor& page_table,
    std::optional<at::Tensor> page_table_2_opt,
    int64_t page_table_page_offset,
    int64_t page_table_2_page_offset,
    int64_t page_table_expected_mapping_offset,
    int64_t page_table_2_expected_mapping_offset,
    int64_t page_table_2_window_size,
    int64_t page_size,
    bool cache_validated_epochs,
    const at::Tensor& actual_tags,
    const at::Tensor& actual_generations,
    const at::Tensor& actual_transfer_tags,
    const at::Tensor& expected_physical_pages,
    int64_t expected_mapping_stride,
    int64_t expected_mapping_namespace_stride,
    const at::Tensor& expected_tags,
    const at::Tensor& expected_generations,
    const at::Tensor& expected_transfer_tags,
    const at::Tensor& request_epochs,
    at::Tensor& validated_epochs,
    at::Tensor& status) {
  check_1d(request_indices, request_indices, at::kLong, "request_indices");
  check_1d(seqlens, request_indices, at::kInt, "seqlens");
  check_table(page_table, request_indices, "page_table");
  TORCH_CHECK(
      request_indices.numel() == seqlens.numel() && request_indices.numel() == page_table.size(0),
      "KV protection batch dimensions must match");
  const int32_t batch_size = checked_int(request_indices.numel(), "batch size");
  const int32_t primary_cols = checked_int(page_table.size(1), "page_table width", true);

  at::Tensor* page_table_2 = nullptr;
  int64_t secondary_stride = 0;
  int32_t secondary_cols = 0;
  if (page_table_2_opt.has_value()) {
    page_table_2 = &page_table_2_opt.value();
    check_table(*page_table_2, request_indices, "page_table_2");
    TORCH_CHECK(page_table_2->size(0) == batch_size, "page_table_2 batch dimension must match");
    secondary_stride = page_table_2->stride(0);
    secondary_cols = checked_int(page_table_2->size(1), "page_table_2 width", true);
  } else {
    TORCH_CHECK(
        page_table_2_page_offset == 0 && page_table_2_expected_mapping_offset == 0 && page_table_2_window_size == 0,
        "secondary KV protection arguments require page_table_2");
  }

  for (const auto& item :
       {std::pair<const at::Tensor*, const char*>{&actual_tags, "actual_tags"},
        {&actual_generations, "actual_generations"},
        {&expected_tags, "expected_tags"},
        {&expected_generations, "expected_generations"}}) {
    check_1d(*item.first, request_indices, at::kLong, item.second);
  }
  for (const auto& item :
       {std::pair<const at::Tensor*, const char*>{&actual_transfer_tags, "actual_transfer_tags"},
        {&expected_physical_pages, "expected_physical_pages"},
        {&expected_transfer_tags, "expected_transfer_tags"},
        {&request_epochs, "request_epochs"},
        {&validated_epochs, "validated_epochs"},
        {&status, "status"}}) {
    check_1d(*item.first, request_indices, at::kInt, item.second);
  }

  const int32_t sidecar_pages = checked_int(actual_tags.numel(), "physical sidecar size", true);
  TORCH_CHECK(
      actual_generations.numel() == sidecar_pages && actual_transfer_tags.numel() == sidecar_pages,
      "physical sidecar lengths must match");
  const int32_t request_slots = checked_int(request_epochs.numel(), "request sidecar size", true);
  TORCH_CHECK(
      validated_epochs.numel() == request_slots && status.numel() == request_slots,
      "request sidecar lengths must match");

  const int32_t mapping_stride = checked_int(expected_mapping_stride, "expected_mapping_stride", true);
  const int32_t namespace_stride =
      checked_int(expected_mapping_namespace_stride, "expected_mapping_namespace_stride", true);
  const int32_t primary_mapping_offset =
      checked_int(page_table_expected_mapping_offset, "page_table_expected_mapping_offset");
  const int32_t secondary_mapping_offset =
      checked_int(page_table_2_expected_mapping_offset, "page_table_2_expected_mapping_offset");
  TORCH_CHECK(
      static_cast<int64_t>(primary_mapping_offset) + namespace_stride <= mapping_stride &&
          (!page_table_2 || static_cast<int64_t>(secondary_mapping_offset) + namespace_stride <= mapping_stride),
      "expected mapping namespace is out of range");
  const int64_t expected_count = static_cast<int64_t>(request_slots) * mapping_stride;
  TORCH_CHECK(
      expected_physical_pages.numel() == expected_count && expected_tags.numel() == expected_count &&
          expected_generations.numel() == expected_count && expected_transfer_tags.numel() == expected_count,
      "expected mapping sidecar lengths must match");

  const int32_t checked_page_size = checked_int(page_size, "page_size", true);

  if (batch_size == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(request_indices.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const ProtectionParams params{
      .request_indices = request_indices.data_ptr<int64_t>(),
      .seqlens = seqlens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_table_stride = page_table.stride(0),
      .page_table_cols = primary_cols,
      .page_table_2 = page_table_2 ? page_table_2->data_ptr<int32_t>() : nullptr,
      .page_table_2_stride = secondary_stride,
      .page_table_2_cols = secondary_cols,
      .page_table_page_offset = checked_int(page_table_page_offset, "page_table_page_offset"),
      .page_table_2_page_offset = checked_int(page_table_2_page_offset, "page_table_2_page_offset"),
      .page_table_expected_mapping_offset = primary_mapping_offset,
      .page_table_2_expected_mapping_offset = secondary_mapping_offset,
      .page_table_2_window_size = checked_int(page_table_2_window_size, "page_table_2_window_size"),
      .page_size = checked_page_size,
      .cache_validated_epochs = cache_validated_epochs,
      .num_sidecar_pages = sidecar_pages,
      .num_request_slots = request_slots,
      .expected_mapping_stride = mapping_stride,
      .expected_mapping_namespace_stride = namespace_stride,
      .actual_tags = actual_tags.data_ptr<int64_t>(),
      .actual_generations = actual_generations.data_ptr<int64_t>(),
      .actual_transfer_tags = actual_transfer_tags.data_ptr<int32_t>(),
      .expected_physical_pages = expected_physical_pages.data_ptr<int32_t>(),
      .expected_tags = expected_tags.data_ptr<int64_t>(),
      .expected_generations = expected_generations.data_ptr<int64_t>(),
      .expected_transfer_tags = expected_transfer_tags.data_ptr<int32_t>(),
      .request_epochs = request_epochs.data_ptr<int32_t>(),
      .validated_epochs = validated_epochs.data_ptr<int32_t>(),
      .status = status.data_ptr<int32_t>(),
  };
  kv_page_protection_preflight_kernel<<<batch_size, kThreads, 0, stream>>>(params, batch_size);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "KV page protection preflight launch failed: ", cudaGetErrorString(error));
}
