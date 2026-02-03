#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <type_traits>

namespace {

template <typename T, int VEC_SIZE_IN_BYTE>
struct VecTypeTrait;

template <>
struct VecTypeTrait<bf16_t, 16> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<fp16_t, 16> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<bf16_t, 32> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct VecTypeTrait<fp16_t, 32> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <typename packed_t>
SGL_DEVICE packed_t rms(packed_t& val, packed_t& weight, float rsqrt_square_sum) {
  float2 valf = device::cast<fp32x2_t, packed_t>(val);
  float2 weightf = device::cast<fp32x2_t, packed_t>(weight);
  return device::cast<packed_t, fp32x2_t>(
      make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void qknorm_across_heads_reg_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    int vec_hidden_size,
    uint32_t num_tokens,
    float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;
  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  __shared__ float2 shared_memory[device::kWarpThreads];  // per-warp sums and rsqrt

  const uint32_t lane_id = threadIdx.x % device::kWarpThreads;
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t num_warps = blockDim.x / device::kWarpThreads;
  const float inv_hidden = 1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)));

  vec_t* q_vec = reinterpret_cast<vec_t*>(q);
  vec_t* k_vec = reinterpret_cast<vec_t*>(k);
  const vec_t* q_weight_vec = reinterpret_cast<const vec_t*>(q_weight);
  const vec_t* k_weight_vec = reinterpret_cast<const vec_t*>(k_weight);

  for (uint32_t token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    vec_t v_q;
    vec_t v_k;
    float2 acc_square_q = make_float2(0.0f, 0.0f);  // Sum of squares for q
    float2 acc_square_k = make_float2(0.0f, 0.0f);  // Sum of squares for k

    if (threadIdx.x < vec_hidden_size) {
      vec_t* p_q = q_vec + token_id * vec_hidden_size;
      vec_t* p_k = k_vec + token_id * vec_hidden_size;
      v_q = p_q[threadIdx.x];
      v_k = p_k[threadIdx.x];

#pragma unroll
      for (int i = 0; i < inner_loop; i++) {
        float2 val_q = device::cast<fp32x2_t, packed_t>(v_q[i]);
        float2 val_k = device::cast<fp32x2_t, packed_t>(v_k[i]);
        acc_square_q.x += val_q.x * val_q.x;
        acc_square_q.y += val_q.y * val_q.y;
        acc_square_k.x += val_k.x * val_k.x;
        acc_square_k.y += val_k.y * val_k.y;
      }
    }

    float warp_sum_q = device::warp::reduce_sum(acc_square_q.x + acc_square_q.y);
    float warp_sum_k = device::warp::reduce_sum(acc_square_k.x + acc_square_k.y);
    if (lane_id == 0) {
      shared_memory[warp_id] = make_float2(warp_sum_q, warp_sum_k);
    }

    __syncthreads();
    if (warp_id == 0) {
      const float local_q = threadIdx.x < num_warps ? shared_memory[threadIdx.x].x : 0.0f;
      const float local_k = threadIdx.x < num_warps ? shared_memory[threadIdx.x].y : 0.0f;
      const float cta_sum_q = device::warp::reduce_sum(local_q);
      const float cta_sum_k = device::warp::reduce_sum(local_k);
      if (lane_id == 0) {
        shared_memory[0] = make_float2(rsqrtf(eps + cta_sum_q * inv_hidden), rsqrtf(eps + cta_sum_k * inv_hidden));
      }
    }
    __syncthreads();

    if (threadIdx.x < vec_hidden_size) {
      const float2 rsqrt_qk = shared_memory[0];
      vec_t v_weight = q_weight_vec[threadIdx.x];
#pragma unroll
      for (int i = 0; i < inner_loop; i++) {
        v_q[i] = rms(v_q[i], v_weight[i], rsqrt_qk.x);
      }
      vec_t* p_q_out = q_vec + token_id * vec_hidden_size;
      p_q_out[threadIdx.x] = v_q;

      v_weight = k_weight_vec[threadIdx.x];
#pragma unroll
      for (int i = 0; i < inner_loop; i++) {
        v_k[i] = rms(v_k[i], v_weight[i], rsqrt_qk.y);
      }
      vec_t* p_k_out = k_vec + token_id * vec_hidden_size;
      p_k_out[threadIdx.x] = v_k;
    }
  }
}

template <typename DType>
struct QKNormAcrossHeadsKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // q
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, D})  // k
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // q_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight);
    TensorMatcher({D})  // k_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_weight);

    auto cc_major = host::runtime::get_cc_major(device.unwrap().device_id);
    int hidden_size = static_cast<int>(D.unwrap());
    if ((cc_major <= 9 && hidden_size <= 8192) || (cc_major >= 10 && hidden_size <= 12288)) {
      int max_vec_size_byte = cc_major >= 10 ? 32 : 16;
      int elements_in_vec = max_vec_size_byte / sizeof(DType);
      int vec_hidden_size = hidden_size / elements_in_vec;
      uint threads = (vec_hidden_size + device::kWarpThreads - 1) / device::kWarpThreads * device::kWarpThreads;
      const auto num_tokens = static_cast<uint32_t>(N.unwrap());

      // Runtime check
      host::RuntimeCheck(
          hidden_size % elements_in_vec == 0,
          "hidden_size",
          hidden_size,
          " can not align to elements_in_vec ",
          elements_in_vec);

      // Launch single kernel for both q and k
      auto kernel = max_vec_size_byte == 32 ? qknorm_across_heads_reg_kernel<DType, 32>
                                            : qknorm_across_heads_reg_kernel<DType, 16>;

      const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, threads);
      const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
      const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);

      LaunchKernel(num_blocks, threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(q.data_ptr()),
              reinterpret_cast<DType*>(k.data_ptr()),
              reinterpret_cast<DType*>(q_weight.data_ptr()),
              reinterpret_cast<DType*>(k_weight.data_ptr()),
              vec_hidden_size,
              num_tokens,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
