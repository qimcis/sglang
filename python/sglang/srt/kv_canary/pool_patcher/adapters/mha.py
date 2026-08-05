from __future__ import annotations

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patcher.buf_info_splice import patch_buf_info_method
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import (
    alloc_canary_buf,
    make_row_source,
)


def _slot_major_source(buffer: torch.Tensor, *, num_slots: int, read_bytes: int):
    # Some MHA subclasses are page/HND-major. Their logical canary still works,
    # but their bytes are not a contiguous per-token row and must be protected by
    # a kernel-family accessor instead of being mis-described here.
    if buffer.shape[0] != num_slots:
        return ()
    return make_row_source(layer_buffer=buffer, read_bytes=read_bytes)


def attach_mha(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    num_slots = int(pool.size) + int(pool.page_size)
    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)
    v_head = alloc_canary_buf(num_slots=num_slots, device=device)
    v_tail = alloc_canary_buf(num_slots=num_slots, device=device)

    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=_slot_major_source(
            pool.k_buffer[0], num_slots=num_slots, read_bytes=read_bytes
        ),
        real_kv_sources_v=_slot_major_source(
            pool.v_buffer[0], num_slots=num_slots, read_bytes=read_bytes
        ),
        swa_index_lut=None,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    patch_buf_info_method(
        pool,
        method_name="get_contiguous_buf_infos",
        group=group,
        has_v_half=True,
        page_size=pool.page_size,
    )
    return (group,)
