/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <string>

namespace {

#if defined(FLASHMLA_PROTECTED_KV_SM100) || defined(FLASHMLA_PROTECTED_KV_SM103)
__global__ void flashmla_protected_kv_image_probe() {}
#endif

#if defined(FLASHMLA_PROTECTED_SPARSE_SM100) || defined(FLASHMLA_PROTECTED_SPARSE_SM103)
__global__ void flashmla_protected_sparse_image_probe() {}
#endif

template <auto* Kernel>
bool image_is_loadable() {
  const auto result = cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
  if (result == cudaSuccess) return true;
  cudaGetLastError();
  return false;
}

}  // namespace

bool flashmla_protected_consumer_image_available(const std::string& consumer) {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  if (consumer == "flashmla_kv") {
#ifdef FLASHMLA_PROTECTED_KV_SM103
    if (properties->major == 10 && properties->minor == 3)
      return image_is_loadable<flashmla_protected_kv_image_probe>();
#endif
#ifdef FLASHMLA_PROTECTED_KV_SM100
    if (properties->major == 10 && properties->minor == 0)
      return image_is_loadable<flashmla_protected_kv_image_probe>();
#endif
    return false;
  }
  if (consumer == "flashmla_sparse") {
#ifdef FLASHMLA_PROTECTED_SPARSE_SM103
    if (properties->major == 10 && properties->minor == 3)
      return image_is_loadable<flashmla_protected_sparse_image_probe>();
#endif
#ifdef FLASHMLA_PROTECTED_SPARSE_SM100
    if (properties->major == 10 && properties->minor == 0)
      return image_is_loadable<flashmla_protected_sparse_image_probe>();
#endif
  }
  return false;
}
