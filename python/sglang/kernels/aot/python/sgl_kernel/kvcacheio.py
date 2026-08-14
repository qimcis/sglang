from typing import List

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


def dsv4_page_digests(
    buffer: torch.Tensor, page_indices: torch.Tensor, seed: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel.dsv4_page_digests.default(buffer, page_indices, seed)


def dsv4_batched_page_digests(
    descriptors: torch.Tensor,
    page_indices: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.dsv4_batched_page_digests.default(
        descriptors, page_indices, group_offsets
    )


def dsv4_bind_pages(
    slots: torch.Tensor,
    logical_pages: torch.Tensor,
    request_indices: torch.Tensor,
    generations: torch.Tensor,
    request_epochs: torch.Tensor,
    expected_tags: torch.Tensor,
    output: torch.Tensor,
    failure_status: torch.Tensor,
    mapping_seed: int,
    slot_page_size: int,
    invalid_value: int,
    install_missing: bool,
) -> None:
    torch.ops.sgl_kernel.dsv4_bind_pages.default(
        slots,
        logical_pages,
        request_indices,
        generations,
        request_epochs,
        expected_tags,
        output,
        failure_status,
        mapping_seed,
        slot_page_size,
        invalid_value,
        install_missing,
    )


def dsv4_validate_core_mappings(
    full_slots: torch.Tensor,
    full_logical: torch.Tensor,
    out_slots: torch.Tensor,
    out_logical: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_logical: torch.Tensor,
    request_indices: torch.Tensor,
    full_generations: torch.Tensor,
    swa_generations: torch.Tensor,
    request_epochs: torch.Tensor,
    full_tags: torch.Tensor,
    swa_tags: torch.Tensor,
    failure_status: torch.Tensor,
    full_seed: int,
    swa_seed: int,
    full_page_size: int,
    swa_page_size: int,
) -> None:
    torch.ops.sgl_kernel.dsv4_validate_core_mappings.default(
        full_slots,
        full_logical,
        out_slots,
        out_logical,
        swa_slots,
        swa_logical,
        request_indices,
        full_generations,
        swa_generations,
        request_epochs,
        full_tags,
        swa_tags,
        failure_status,
        full_seed,
        swa_seed,
        full_page_size,
        swa_page_size,
    )


def dsv4_validate_pages(
    buffer: torch.Tensor,
    slots: torch.Tensor,
    logical_pages: torch.Tensor,
    request_indices: torch.Tensor,
    digests: torch.Tensor,
    component_valid: torch.Tensor,
    validation_state: torch.Tensor,
    validation_failure: torch.Tensor,
    validation_pages: torch.Tensor,
    validation_count: torch.Tensor,
    generations: torch.Tensor,
    request_epochs: torch.Tensor,
    expected_tags: torch.Tensor,
    output: torch.Tensor,
    failure_status: torch.Tensor,
    seed: int,
    mapping_seed: int,
    slot_page_size: int,
    invalid_value: int,
    allow_missing_digest: bool = False,
) -> None:
    torch.ops.sgl_kernel.dsv4_validate_pages.default(
        buffer,
        slots,
        logical_pages,
        request_indices,
        digests,
        component_valid,
        validation_state,
        validation_failure,
        validation_pages,
        validation_count,
        generations,
        request_epochs,
        expected_tags,
        output,
        failure_status,
        seed,
        mapping_seed,
        slot_page_size,
        invalid_value,
        allow_missing_digest,
    )


def dsv4_refresh_slots(
    buffer: torch.Tensor,
    digests: torch.Tensor,
    valid: torch.Tensor,
    dirty: torch.Tensor,
    slots: torch.Tensor,
    seed: int,
    slot_page_size: int,
) -> None:
    torch.ops.sgl_kernel.dsv4_refresh_slots.default(
        buffer, digests, valid, dirty, slots, seed, slot_page_size
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
