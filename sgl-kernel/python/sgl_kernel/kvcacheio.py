from typing import List

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


def kv_checksum_direct_table_batched(
    buffer_ptrs: torch.Tensor,
    row_strides: torch.Tensor,
    row_nbytes: torch.Tensor,
    buffer_num_rows: torch.Tensor,
    swa_buffer_flags: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    max_num_tokens: int,
    num_lanes: int,
    has_swa: bool,
    is_capped: bool,
    accum: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Production batched direct-KV checksum over request rows in ``req_to_token``."""
    torch.ops.sgl_kernel.kv_checksum_direct_table_batched.default(
        buffer_ptrs,
        row_strides,
        row_nbytes,
        buffer_num_rows,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        req_to_token,
        req_pool_indices,
        starts,
        lengths,
        int(max_num_tokens),
        int(num_lanes),
        bool(has_swa),
        bool(is_capped),
        accum,
        out,
    )


def kv_checksum_direct_table_batched_with_pages(
    buffer_ptrs: torch.Tensor,
    row_strides: torch.Tensor,
    row_nbytes: torch.Tensor,
    buffer_num_rows: torch.Tensor,
    swa_buffer_flags: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    logical_starts: torch.Tensor,
    max_num_tokens: int,
    num_lanes: int,
    page_size: int,
    max_num_pages: int,
    has_swa: bool,
    is_capped: bool,
    accum: torch.Tensor,
    out: torch.Tensor,
    page_accum: torch.Tensor,
    page_out: torch.Tensor,
) -> None:
    """Batched direct-KV root checksum and uint64 logical-page digests."""
    torch.ops.sgl_kernel.kv_checksum_direct_table_batched_with_pages.default(
        buffer_ptrs,
        row_strides,
        row_nbytes,
        buffer_num_rows,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        req_to_token,
        req_pool_indices,
        starts,
        lengths,
        logical_starts,
        int(max_num_tokens),
        int(num_lanes),
        int(page_size),
        int(max_num_pages),
        bool(has_swa),
        bool(is_capped),
        accum,
        out,
        page_accum,
        page_out,
    )


def kv_checksum_direct_table_batched_with_pages_compact(
    buffer_ptrs: torch.Tensor,
    row_strides: torch.Tensor,
    row_nbytes: torch.Tensor,
    buffer_num_rows: torch.Tensor,
    swa_buffer_flags: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    logical_starts: torch.Tensor,
    page_output_offsets: torch.Tensor,
    max_num_tokens: int,
    num_lanes: int,
    page_size: int,
    max_num_pages: int,
    has_swa: bool,
    is_capped: bool,
    accum: torch.Tensor,
    out: torch.Tensor,
    page_accum: torch.Tensor,
    page_out: torch.Tensor,
) -> None:
    """Batched root checksums and compact uint64 logical-page digests."""
    torch.ops.sgl_kernel.kv_checksum_direct_table_batched_with_pages_compact.default(
        buffer_ptrs,
        row_strides,
        row_nbytes,
        buffer_num_rows,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        req_to_token,
        req_pool_indices,
        starts,
        lengths,
        logical_starts,
        page_output_offsets,
        int(max_num_tokens),
        int(num_lanes),
        int(page_size),
        int(max_num_pages),
        bool(has_swa),
        bool(is_capped),
        accum,
        out,
        page_accum,
        page_out,
    )


def kv_page_history_record(
    page_ids: torch.Tensor,
    operation: int,
    generations: torch.Tensor,
    generations_by_page: bool,
    bootstrap_room: int,
    page_positions: torch.Tensor,
    page_position: int,
    values: torch.Tensor,
    value: int,
    cursor: torch.Tensor,
    records: torch.Tensor,
) -> None:
    """Record one bounded page-history event per page with one CUDA launch."""
    torch.ops.sgl_kernel.kv_page_history_record.default(
        page_ids,
        int(operation),
        generations,
        bool(generations_by_page),
        int(bootstrap_room),
        page_positions,
        int(page_position),
        values,
        int(value),
        cursor,
        records,
    )


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer.default(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_direct.default(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_direct_pf_lf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla.default(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf.default(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla.default(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf.default(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def copy_all_layer_kv_cache_cpu(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    torch.ops.sgl_kernel.copy_all_layer_kv_cache_cpu(
        data_ptrs,
        strides,
        tgt_loc,
        src_loc,
    )
