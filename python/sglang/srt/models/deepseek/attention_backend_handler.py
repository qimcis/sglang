from __future__ import annotations

import logging
from enum import IntEnum, auto
from typing import TYPE_CHECKING

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import use_intel_amx_backend

from .utils import _is_hip

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)

# Optional quantization for DeepSeek nvfp4 checkpoint
FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]


def add_forward_absorb_core_attention_backend(backend_name: str) -> None:
    if backend_name not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
        FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.")


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in one shot
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()


def _dispatch_mla_subtype(attn: "DeepseekV2AttentionMLA", forward_batch):
    if _is_hip:
        if attn.rocm_fused_decode_mla and forward_batch.forward_mode.is_decode():
            return AttnForwardMethod.MLA_FUSED_ROPE
        else:
            return AttnForwardMethod.MLA
    else:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(attn):
            return AttnForwardMethod.MLA_FUSED_ROPE_CPU
        else:
            return AttnForwardMethod.MLA


class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))


def handle_attention_ascend(attn, forward_batch):
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    ):
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU
    else:
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MLA_NPU


def _get_sum_extend_prefix_lens(forward_batch):
    return (
        sum(forward_batch.extend_prefix_lens_cpu)
        if forward_batch.extend_prefix_lens_cpu is not None
        else 0
    )


def _support_mha_one_shot(attn: "DeepseekV2AttentionMLA", forward_batch, backend_name):
    attn_supported = backend_name in ["fa3", "flashinfer", "flashmla"]
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()


def _handle_attention_backend(
    attn: "DeepseekV2AttentionMLA", forward_batch, backend_name
):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    disable_ragged = (
        backend_name in ["flashinfer", "flashmla"]
    ) and attn.flashinfer_mla_disable_ragged

    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_flashinfer(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashinfer")


def handle_attention_fa3(attn, forward_batch):
    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")


def handle_attention_flashmla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashmla")


def handle_attention_cutlass_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "cutlass_mla")


def handle_attention_fa4(attn, forward_batch):
    # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
    return AttnForwardMethod.MHA_CHUNKED_KV


def handle_attention_trtllm_mla(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if forward_batch.forward_mode.is_extend_without_speculative() and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_aiter(attn, forward_batch):
    if forward_batch.forward_mode.is_extend_without_speculative():
        return AttnForwardMethod.MHA
    else:
        return AttnForwardMethod.MLA


def handle_attention_nsa(attn, forward_batch):
    """
    Dispatch logic is centralized in NativeSparseAttnBackend.set_nsa_prefill_impl and executed
    in init_forward_metadata. Read the decision from backend.use_mha.
    """

    backend = forward_batch.attn_backend
    if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
        backend = backend.primary
    if hasattr(backend, "use_mha") and backend.use_mha:
        return AttnForwardMethod.MHA_ONE_SHOT
    return AttnForwardMethod.MLA


def handle_attention_triton(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)

    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and sum(forward_batch.extend_prefix_lens_cpu) == 0
    ):
        return AttnForwardMethod.MHA
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def register_default_attention_backends():
    AttentionBackendRegistry.register("ascend", handle_attention_ascend)
    AttentionBackendRegistry.register("flashinfer", handle_attention_flashinfer)
    AttentionBackendRegistry.register("fa3", handle_attention_fa3)
    AttentionBackendRegistry.register("flashmla", handle_attention_flashmla)
    AttentionBackendRegistry.register("cutlass_mla", handle_attention_cutlass_mla)
    AttentionBackendRegistry.register("fa4", handle_attention_fa4)
    AttentionBackendRegistry.register("trtllm_mla", handle_attention_trtllm_mla)
    AttentionBackendRegistry.register("aiter", handle_attention_aiter)
    AttentionBackendRegistry.register("nsa", handle_attention_nsa)
    AttentionBackendRegistry.register("triton", handle_attention_triton)


# Register default handlers on import so existing code paths remain unchanged.
register_default_attention_backends()

__all__ = [
    "AttentionBackendRegistry",
    "AttnForwardMethod",
    "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS",
    "add_forward_absorb_core_attention_backend",
    "handle_attention_aiter",
    "handle_attention_ascend",
    "handle_attention_cutlass_mla",
    "handle_attention_fa3",
    "handle_attention_fa4",
    "handle_attention_flashinfer",
    "handle_attention_flashmla",
    "handle_attention_nsa",
    "handle_attention_triton",
    "handle_attention_trtllm_mla",
]
