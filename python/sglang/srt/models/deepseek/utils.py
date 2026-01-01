from __future__ import annotations

import math

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_nvidia_cublas_cu12_version_ge_12_9,
)

# Hardware feature flags used across DeepSeek modules
_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_is_cublas_ge_129 = is_nvidia_cublas_cu12_version_ge_12_9()

# Optional hardware-specific imports shared across mixins
if _use_aiter_gfx95:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import (
        aiter_dsv3_router_gemm,
        fused_qk_rope_cat_and_cache_mla,
        get_dsv3_gemm_output_zero_allocator_size,
    )
else:
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant = None
    fused_flatten_fp8_group_quant = None
    fused_rms_fp8_group_quant = None
    quark_post_load_weights = None
    batched_gemm_afp4wfp4_pre_quant = None
    fused_flatten_mxfp4_quant = None
    fused_rms_mxfp4_quant = None
    aiter_dsv3_router_gemm = None
    fused_qk_rope_cat_and_cache_mla = None
    get_dsv3_gemm_output_zero_allocator_size = None

if _is_cuda:
    from sgl_kernel import (
        awq_dequantize,
        bmm_fp8,
        concat_mla_k,
        dsv3_fused_a_gemm,
        dsv3_router_gemm,
        merge_state_v2,
    )
elif _is_cpu and _is_cpu_amx_available:
    awq_dequantize = None
    bmm_fp8 = None
    concat_mla_k = None
    dsv3_fused_a_gemm = None
    dsv3_router_gemm = None
    merge_state_v2 = None
elif _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )
    from sglang.srt.layers.quantization.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )

    bmm_fp8 = None
    concat_mla_k = None
    dsv3_fused_a_gemm = None
    dsv3_router_gemm = None
    merge_state_v2 = None
elif _is_npu:
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )
    from sglang.srt.layers.quantization.awq_triton import (
        awq_dequantize_decomposition as awq_dequantize,
    )

    bmm_fp8 = None
    concat_mla_k = None
    dsv3_fused_a_gemm = None
    dsv3_router_gemm = None
    merge_state_v2 = None
    decode_attention_fwd_grouped_rope = None
else:
    awq_dequantize = None
    bmm_fp8 = None
    concat_mla_k = None
    dsv3_fused_a_gemm = None
    dsv3_router_gemm = None
    merge_state_v2 = None
    decode_attention_fwd_grouped_rope = None

# Optional quantization for DeepSeek nvfp4 checkpoint
NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]


def enable_nextn_moe_bf16_cast_to_fp8(quant_config):
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


__all__ = [
    "_device_sm",
    "_get_llama_4_scaling",
    "_is_cpu",
    "_is_cpu_amx_available",
    "_is_cuda",
    "_is_cublas_ge_129",
    "_is_fp8_fnuz",
    "_is_gfx95_supported",
    "_is_hip",
    "_is_npu",
    "_use_aiter",
    "_use_aiter_gfx95",
    "awq_dequantize",
    "bmm_fp8",
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
    "batched_gemm_afp4wfp4_pre_quant",
    "concat_mla_k",
    "decode_attention_fwd_grouped_rope",
    "dsv3_fused_a_gemm",
    "dsv3_router_gemm",
    "fused_flatten_fp8_group_quant",
    "fused_flatten_mxfp4_quant",
    "fused_qk_rope_cat_and_cache_mla",
    "fused_rms_fp8_group_quant",
    "fused_rms_mxfp4_quant",
    "get_dsv3_gemm_output_zero_allocator_size",
    "merge_state_v2",
    "quark_post_load_weights",
    "forward_dsa_core_npu",
    "forward_dsa_prepare_npu",
    "forward_mha_core_npu",
    "forward_mha_prepare_npu",
    "forward_mla_core_npu",
    "forward_mla_prepare_npu",
    "NVFP4_CKPT_FP8_ATTN_QUANT_MODULES",
    "enable_nextn_moe_bf16_cast_to_fp8",
    "yarn_get_mscale",
]
