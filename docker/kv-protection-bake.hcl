variable "REGISTRY" {
  default = "local"
}

variable "SOURCE_REVISION" {
  default = "unknown"
}

group "default" {
  targets = ["hopper", "b200", "b300"]
}

target "kv-protection" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  target     = "kv_protection_final"
  platforms  = ["linux/amd64"]
  args = {
    BRANCH_TYPE                = "local"
    SGLANG_BUILD_COMMIT        = SOURCE_REVISION
    SGL_KERNEL_SOURCE_REVISION = SOURCE_REVISION
  }
}

target "hopper" {
  inherits = ["kv-protection"]
  tags     = ["${REGISTRY}/sglang-kv-protection:hopper-${SOURCE_REVISION}"]
  args = {
    CUDA_VERSION            = "12.9.1"
    SGL_KERNEL_ARCH_PROFILE = "hopper-sm90"
  }
}

target "b200" {
  inherits = ["kv-protection"]
  tags     = ["${REGISTRY}/sglang-kv-protection:b200-${SOURCE_REVISION}"]
  args = {
    CUDA_VERSION            = "12.9.1"
    SGL_KERNEL_ARCH_PROFILE = "blackwell-sm100"
  }
}

target "b300" {
  inherits = ["kv-protection"]
  tags     = ["${REGISTRY}/sglang-kv-protection:b300-${SOURCE_REVISION}"]
  args = {
    CUDA_VERSION            = "13.0.1"
    SGL_KERNEL_ARCH_PROFILE = "blackwell-sm103"
  }
}
