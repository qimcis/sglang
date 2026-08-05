from __future__ import annotations

import logging
from typing import Callable, Dict, Type

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patcher.adapters.dsv4 import attach_dsv4
from sglang.srt.kv_canary.pool_patcher.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patcher.adapters.mla import (
    attach_hybrid_linear,
    attach_mla,
)
from sglang.srt.kv_canary.pool_patcher.adapters.swa import attach_swa
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import resolve_real_kv_read_bytes
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    HybridLinearKVPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)

PoolAttacher = Callable[..., tuple[CanaryBufferGroup, ...]]

_POOL_ATTACHERS: Dict[Type, PoolAttacher] = {
    MHATokenToKVPool: attach_mha,
    MHATokenToKVPoolFP4: attach_mha,
    MLATokenToKVPool: attach_mla,
    HybridLinearKVPool: attach_hybrid_linear,
    SWAKVPool: attach_swa,
    DeepSeekV4TokenToKVPool: attach_dsv4,
}


def register_pool_attacher(pool_class: Type, attacher: PoolAttacher) -> None:
    _POOL_ATTACHERS[pool_class] = attacher


def resolve_pool_attacher(pool: KVCache) -> PoolAttacher:
    """Resolve exact and subclassed pool layouts, preferring the nearest MRO type."""
    exact = _POOL_ATTACHERS.get(type(pool))
    if exact is not None:
        return exact
    for base in type(pool).__mro__[1:]:
        if base in _POOL_ATTACHERS:
            return _POOL_ATTACHERS[base]
    raise NotImplementedError(
        f"kv-canary: no attacher registered for pool class {type(pool).__name__}; "
        f"supported bases: {sorted(cls.__name__ for cls in _POOL_ATTACHERS)}"
    )


def attach_canary_buffers(
    *,
    pool: KVCache,
    config: CanaryConfig,
    device: torch.device,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple.

    ``kv_token_id_vs_position_offset`` is propagated into every produced :class:`CanaryBufferGroup` (0 for target
    pools; 1 for draft pools where the input-ids rotation shifts the slot-to-token mapping by one).
    """
    attacher = resolve_pool_attacher(pool)

    read_bytes = resolve_real_kv_read_bytes(config)
    groups = attacher(
        pool=pool,
        device=device,
        read_bytes=read_bytes,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    logger.info(
        "attach_canary_buffers: pool=%s attacher=%s read_bytes=%d n_groups=%d kinds=%s "
        "kv_token_id_vs_position_offset=%d",
        type(pool).__name__,
        attacher.__name__,
        read_bytes,
        len(groups),
        [g.kind.name for g in groups],
        kv_token_id_vs_position_offset,
    )
    return groups
