from __future__ import annotations

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patcher.buf_info_splice import patch_buf_info_method
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import (
    alloc_canary_buf,
    make_row_source,
)


def attach_mla(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    """Attach the logical chain to MLA/DSA pools.

    MLA has one combined latent-KV buffer and no independent V half. The real
    payload mixin uses the first raw local layer when it is slot-major; the
    state-protection layer supplies per-consumer coverage for the remaining
    layer/indexer buffers.
    """

    num_slots = int(pool.size) + int(pool.page_size)
    head = alloc_canary_buf(num_slots=num_slots, device=device)
    tail = alloc_canary_buf(num_slots=num_slots, device=device)
    raw = pool.kv_buffer[0]
    real_sources = (
        make_row_source(layer_buffer=raw, read_bytes=read_bytes)
        if raw.shape[0] == num_slots
        else ()
    )
    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=head,
        k_tail=tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=real_sources,
        real_kv_sources_v=(),
        swa_index_lut=None,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    patch_buf_info_method(
        pool,
        method_name="get_contiguous_buf_infos",
        group=group,
        has_v_half=False,
        page_size=pool.page_size,
    )
    return (group,)


def attach_hybrid_linear(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    # The outer pool delegates contiguous KV registration and all full-attention
    # reads/writes to this sub-pool, so patching it preserves one source of truth.
    # A pure recurrent model legitimately has no paged-attention layer; its
    # state-protection accessor supplies mapping/generation/payload coverage
    # without fabricating a KV canary endpoint.
    if int(pool.full_layer_nums) == 0:
        return ()
    from sglang.srt.kv_canary.pool_patcher.api import resolve_pool_attacher

    attacher = resolve_pool_attacher(pool.full_kv_pool)
    return attacher(
        pool=pool.full_kv_pool,
        device=device,
        read_bytes=read_bytes,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
