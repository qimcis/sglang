from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api

try:
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    _flash_attn_varlen_func = None
    _flash_attn_import_error = _e
else:
    _flash_attn_import_error = None


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@lru_cache(maxsize=None)
def _preflight_supported_on_device(device_index: int) -> bool:
    from sgl_kernel import kv_page_protection_preflight_supported

    with torch.cuda.device(device_index):
        return kv_page_protection_preflight_supported()


def fa4_kv_page_protection_supported(device: torch.device) -> bool:
    device = torch.device(device)
    if device.type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        return False
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return _preflight_supported_on_device(device_index)


def _preflight_kv_page_protection(
    protection: Optional[Dict[str, Any]],
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cache_page_size: Optional[int],
) -> None:
    if protection is None:
        return
    if page_table is None or seqused_k is None:
        raise RuntimeError(
            "FA4 KV page protection requires paged KV and sequence lengths"
        )
    if protection.get("validate_full_mapping", False):
        raise RuntimeError("FA4 KV protection requires an active page-table mapping")
    if seqused_k is not protection.get("seqlens"):
        raise RuntimeError(
            "FA4 KV protection sequence lengths must be consumed by attention"
        )
    if page_table is not protection.get(
        "page_table"
    ) and page_table is not protection.get("page_table_2"):
        raise RuntimeError(
            "FA4 KV protection must validate the page table consumed by attention"
        )
    if cache_page_size is None or protection.get("page_size") != cache_page_size:
        raise RuntimeError("FA4 KV protection page size must match the KV cache")
    from sgl_kernel import kv_page_protection_preflight

    if page_table.is_cuda and not fa4_kv_page_protection_supported(page_table.device):
        capability = torch.cuda.get_device_capability(page_table.device)
        raise RuntimeError(
            "FA4 KV protection requires a loadable sgl-kernel preflight image "
            f"for SM100 or SM103; got SM{capability[0]}{capability[1]}"
        )

    # This same-stream launch is intentionally retained on every layer. The
    # device kernel exits after epoch/status loads once a request was validated.
    kv_page_protection_preflight(protection)


@debug_kernel_api
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    kv_page_protection: Optional[Dict[str, Any]] = None,
):
    if _flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "Vendored FlashAttention CUTE is not available (cannot import "
            "flash_attn.cute). Please check your source tree."
        ) from _flash_attn_import_error

    q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
    cu_seqlens_q, cu_seqlens_k = [
        _maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
    page_table = _maybe_contiguous(page_table)

    if learnable_sink is None and sinks is not None:
        learnable_sink = sinks

    if window_size == (-1, -1):
        window_size = (None, None)

    cache_page_size = k.shape[1] if page_table is not None and k.dim() == 4 else None
    _preflight_kv_page_protection(
        kv_page_protection, page_table, seqused_k, cache_page_size
    )
    result = _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_softmax_lse,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result


@debug_kernel_api
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: Optional[int] = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    sinks: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    kv_page_protection: Optional[Dict[str, Any]] = None,
    **_: object,
):
    if k is not None or v is not None or qv is not None:
        raise NotImplementedError("FA4 does not support updating KV cache in-place.")
    if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
        raise NotImplementedError("FA4 path does not support rotary embedding.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError(
            "FA4 path does not support non-consecutive batch indices or left padding."
        )
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError("FA4 path does not support descale.")

    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    result = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap != 0.0 else None,
        window_size=window_size,
        num_splits=num_splits if num_splits != 0 else 1,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_softmax_lse=True,
        kv_page_protection=kv_page_protection,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result
