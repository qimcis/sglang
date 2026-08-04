from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_len",
    ]
)
def _fused_dsa_decode_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_len,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        seq_i32 = seq.to(tl.int32)
        dsa_seq = tl.minimum(seq_i32, dsa_index_topk)

        cu = tl.cumsum(seq_i32, 0)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(cache_seqlens + offs_b, seq_i32, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)
        tl.store(dsa_cache_seqlens + offs_b, dsa_seq, mask=mask_b)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_b, dsa_cu, mask=mask_b)
        return

    num_col_blocks = tl.cdiv(max_len, BLOCK_N)
    page_pid = pid - 1
    row = page_pid // num_col_blocks
    col_block = page_pid - row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row < bs) & (offs_n < max_len)

    req_idx = tl.load(
        req_pool_indices + row * req_pool_indices_stride,
        mask=row < bs,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    # Write the wide page_size=1 table only when the caller provides it; the
    # fused decode CUDA graph drops it and consumes real_page_table alone.
    if HAS_PAGE_TABLE_1:
        tl.store(
            page_table_1 + row * page_table_stride_0 + offs_n * page_table_stride_1,
            vals,
            mask=mask,
        )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + row * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // real_page_size,
            mask=real_mask,
        )


@triton.jit(
    do_not_specialize=[
        "real_page_table_stride_0",
        "max_len",
    ]
)
def _fused_dsa_decode_metadata_protected_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    protected_request_indices,
    failure_status,
    local_failed,
    global_failed,
    actual_tags,
    actual_generations,
    actual_transfer_tags,
    expected_physical_pages,
    expected_tags,
    expected_generations,
    expected_transfer_tags,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_len,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    num_sidecar_pages: tl.constexpr,
    num_request_slots: tl.constexpr,
    expected_mapping_stride: tl.constexpr,
    expected_mapping_namespace_stride: tl.constexpr,
    REAL_PAGE_COLS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    """Snapshot and validate the compact DSA page table in one launch.

    The protected normal-decode graph consumes only ``real_page_table``.  By
    validating while that table is copied out of ``req_to_token``, every
    mutable global sidecar read finishes before graph replay starts.  A bad row
    is published as all-zero pages, so neither the indexer nor attention can
    dereference the rejected mapping.
    """
    pid = tl.program_id(0)
    if pid == 0:
        # Preserve the original metadata kernel's single-pass prefix work.
        # Shape-invalid rows are clamped consistently; mapping/tag failures are
        # made safe by redirecting their complete table to reserved page 0.
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        input_seq = tl.load(
            seq_lens + offs_b * seq_lens_stride,
            mask=mask_b,
            other=0,
        ).to(tl.int32)
        # Shape faults need a bounded non-zero length before any external
        # kernel runs. Mapping/tag faults keep their valid original length and
        # are made safe by the all-zero compact page table below.
        max_safe_seq = tl.minimum(max_len, REAL_PAGE_COLS * real_page_size)
        all_seq = tl.where((input_seq > 0) & (input_seq <= max_safe_seq), input_seq, 1)
        all_dsa_seq = tl.minimum(all_seq, dsa_index_topk)
        tl.store(cache_seqlens + offs_b, all_seq, mask=mask_b)
        tl.store(cu_seqlens_k, 0)
        tl.store(
            cu_seqlens_k + 1 + offs_b,
            tl.cumsum(all_seq, axis=0),
            mask=mask_b,
        )
        tl.store(dsa_cache_seqlens + offs_b, all_dsa_seq, mask=mask_b)
        tl.store(dsa_cu_seqlens_k, 0)
        tl.store(
            dsa_cu_seqlens_k + 1 + offs_b,
            tl.cumsum(all_dsa_seq, axis=0),
            mask=mask_b,
        )
        return

    row = pid - 1
    row_mask = row < bs
    request_idx_i64 = tl.load(
        req_pool_indices + row * req_pool_indices_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    seq = tl.load(
        seq_lens + row * seq_lens_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    tl.store(protected_request_indices + row, request_idx_i64, mask=row_mask)

    graph_padding = request_idx_i64 == 0
    request_valid = (request_idx_i64 > 0) & (request_idx_i64 < num_request_slots)
    safe_request_idx = tl.where(request_valid, request_idx_i64, 0)
    num_pages = (seq + real_page_size - 1) // real_page_size
    row_shape_valid = (
        request_valid & (seq > 0) & (seq <= max_len) & (num_pages <= REAL_PAGE_COLS)
    )

    page_lane = tl.arange(0, BLOCK_PAGES)
    invalid_any = tl.zeros((), tl.int32)
    owner_any = tl.zeros((), tl.int32)
    position_any = tl.zeros((), tl.int32)
    attention_tag_any = tl.zeros((), tl.int32)
    generation_any = tl.zeros((), tl.int32)
    transfer_tag_any = tl.zeros((), tl.int32)

    # Keep register pressure independent of context length and stop at the live
    # page count. On a clean row the compact table is produced during the same
    # read that validates it. Only a failed row pays a full write-only clear.
    scan_pages = tl.where(row_shape_valid, num_pages, 0)
    for scan_start in tl.range(0, scan_pages, BLOCK_PAGES):
        logical_page = scan_start + page_lane
        output_mask = logical_page < scan_pages
        active = output_mask
        token_slot = tl.load(
            req_to_token
            + safe_request_idx * req_to_token_stride_0
            + logical_page * real_page_size * req_to_token_stride_1,
            mask=active,
            other=0,
        ).to(tl.int64)
        physical_page = token_slot // real_page_size
        mapping_valid = (
            active
            & (token_slot > 0)
            & ((token_slot % real_page_size) == 0)
            & (physical_page > 0)
            & (physical_page < num_sidecar_pages)
        )
        expected_position_valid = logical_page < expected_mapping_namespace_stride
        sidecar_valid = mapping_valid & expected_position_valid
        safe_physical_page = tl.where(sidecar_valid, physical_page, 0)
        expected_index = safe_request_idx * expected_mapping_stride + logical_page

        expected_page = tl.load(
            expected_physical_pages + expected_index,
            mask=sidecar_valid,
            other=0,
        ).to(tl.int64)
        expected_tag = tl.load(
            expected_tags + expected_index,
            mask=sidecar_valid,
            other=0,
        )
        expected_generation = tl.load(
            expected_generations + expected_index,
            mask=sidecar_valid,
            other=0,
        )
        expected_transfer_tag = tl.load(
            expected_transfer_tags + expected_index,
            mask=sidecar_valid,
            other=0,
        )
        actual_tag = tl.load(
            actual_tags + safe_physical_page,
            mask=sidecar_valid,
            other=0,
        )
        actual_generation = tl.load(
            actual_generations + safe_physical_page,
            mask=sidecar_valid,
            other=0,
        )
        actual_transfer_tag = tl.load(
            actual_transfer_tags + safe_physical_page,
            mask=sidecar_valid,
            other=0,
        )

        invalid_mapping = active & ~mapping_valid
        position_mismatch = mapping_valid & ~expected_position_valid
        owner_mismatch = sidecar_valid & (expected_page != physical_page)
        attention_tag_mismatch = sidecar_valid & (actual_tag != expected_tag)
        generation_mismatch = sidecar_valid & (actual_generation != expected_generation)
        transfer_tag_mismatch = (
            sidecar_valid
            & (expected_transfer_tag != 0)
            & (actual_transfer_tag != expected_transfer_tag)
        )

        invalid_any |= tl.max(invalid_mapping.to(tl.int32), axis=0)
        owner_any |= tl.max(owner_mismatch.to(tl.int32), axis=0)
        position_any |= tl.max(position_mismatch.to(tl.int32), axis=0)
        attention_tag_any |= tl.max(attention_tag_mismatch.to(tl.int32), axis=0)
        generation_any |= tl.max(generation_mismatch.to(tl.int32), axis=0)
        transfer_tag_any |= tl.max(transfer_tag_mismatch.to(tl.int32), axis=0)

        output_page = tl.where(active, physical_page, 0).to(tl.int32)
        tl.store(
            real_page_table
            + row * real_page_table_stride_0
            + logical_page * real_page_table_stride_1,
            output_page,
            mask=row_mask & output_mask,
        )

    # Keep these values in sync with KV_PAGE_* in kv_protection.py.
    status = (
        invalid_any
        | (owner_any << 1)
        | (position_any << 2)
        | (attention_tag_any << 3)
        | (generation_any << 4)
        | (transfer_tag_any << 5)
    )
    status = tl.where((~graph_padding) & (~row_shape_valid), status | 0x01, status)
    status = tl.where(graph_padding, 0, status).to(tl.int32)
    failed = (status != 0).to(tl.int32)

    if failed != 0:
        # Use a distinct induction variable from the live-page scan above.
        # Triton otherwise merges both loop variables through the surrounding
        # dynamic branch and rejects the int64 live bound / int32 constexpr
        # bound mismatch during TTIR construction on SM100.
        for clear_start in tl.range(0, REAL_PAGE_COLS, BLOCK_PAGES):
            logical_page = clear_start + page_lane
            output_mask = logical_page < REAL_PAGE_COLS
            tl.store(
                real_page_table
                + row * real_page_table_stride_0
                + logical_page * real_page_table_stride_1,
                0,
                mask=row_mask & output_mask,
            )
        # Shape-invalid lengths were already clamped consistently by program 0.
        # Mapping/tag faults retain their valid length, but every page entry is
        # redirected to reserved page 0. TP-global failure consensus prevents
        # this row from emitting a token.
    elif graph_padding:
        # Padding rows have a graph sentinel length of one, so only the first
        # compact entry can be consumed.
        tl.store(real_page_table + row * real_page_table_stride_0, 0)
    tl.store(failure_status + row, status, mask=row_mask)
    tl.store(local_failed + row, failed, mask=row_mask)
    # TP=1 consumes this directly.  TP>1 overwrites it inside the existing
    # logits all-gather after reducing the per-rank sideband.
    tl.store(global_failed + row, failed, mask=row_mask)


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    real_page_size: int,
    protection: Optional[dict] = None,
) -> None:
    """Fill decode-graph DSA metadata (seqlens + page tables) from req_to_token.

    ``page_table_1`` (the wide page_size=1 table) is optional: pass ``None`` to
    skip materializing it and write only the compact ``real_page_table``
    (page_size=``real_page_size``). This is used by the fused decode CUDA graph,
    where the wide table is never read (attention uses topk_indices, the indexer
    uses real_page_table); ``real_page_size`` must be >1 in that case. When a
    tensor is passed, behavior is unchanged (both tables are written).
    """
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        # page_size==1: real IS page_table_1, so page_table_1 must be present.
        assert page_table_1 is not None
        real_page_table = page_table_1

    # page_table_1 (the wide page_size=1 table) may be dropped for the fused
    # decode CUDA graph; the kernel then writes only real_page_table.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table  # dummy pointer for stride args
    else:
        assert page_table_1.is_cuda

    block_bs = triton.next_power_of_2(bs)

    if protection is not None:
        # KV protection is intentionally scoped to the compact page-size=64
        # GLM/DSA normal-decode path.  Its graph consumes this snapshot and no
        # longer reads mutable global protection sidecars during replay.
        assert has_real_page_table and not has_page_table_1
        assert real_page_size == 64
        assert protection["page_table"] is real_page_table
        assert protection["request_indices"].shape[0] == bs
        max_pages = real_page_table.shape[1]
        assert max_pages > 0
        block_pages = min(256, triton.next_power_of_2(max_pages))
        _fused_dsa_decode_metadata_protected_kernel[(1 + bs,)](
            seq_lens,
            req_pool_indices,
            req_to_token,
            cache_seqlens,
            cu_seqlens_k,
            dsa_cache_seqlens,
            dsa_cu_seqlens_k,
            real_page_table,
            protection["result_request_indices"],
            protection["failure_status"],
            protection["local_failed"],
            protection["failed"],
            protection["actual_tags"],
            protection["actual_generations"],
            protection["actual_transfer_tags"],
            protection["expected_physical_pages"],
            protection["expected_tags"],
            protection["expected_generations"],
            protection["expected_transfer_tags"],
            seq_lens.stride(0),
            req_pool_indices.stride(0),
            req_to_token.stride(0),
            req_to_token.stride(1),
            real_page_table.stride(0),
            real_page_table.stride(1),
            bs,
            max_len,
            dsa_index_topk,
            real_page_size,
            protection["actual_tags"].numel(),
            protection["num_request_slots"],
            protection["expected_mapping_stride"],
            protection["expected_mapping_namespace_stride"],
            REAL_PAGE_COLS=max_pages,
            BLOCK_BS=block_bs,
            BLOCK_PAGES=block_pages,
            num_warps=8,
        )
        protection["validated_in_metadata"] = True
        return

    block_n = 128
    num_col_blocks = triton.cdiv(max_len, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        max_len,
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        has_page_table_1,
        BLOCK_BS=block_bs,
        BLOCK_N=block_n,
    )


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_seqlen_k",
    ]
)
def _fused_dsa_target_verify_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    paged_mqa_ctx_lens_2d,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    paged_mqa_ctx_lens_stride_0: tl.constexpr,
    paged_mqa_ctx_lens_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_seqlen_k,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    next_n: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGED_MQA_CTX_LENS: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    expanded_size: tl.constexpr = bs * next_n

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32) + next_n
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < expanded_size
        req_row = offs_e // next_n
        draft_off = offs_e - req_row * next_n
        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        expanded_seq = base_seq + draft_off + 1
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)

        if HAS_PAGED_MQA_CTX_LENS:
            tl.store(
                paged_mqa_ctx_lens_2d
                + req_row * paged_mqa_ctx_lens_stride_0
                + draft_off * paged_mqa_ctx_lens_stride_1,
                base_seq + next_n,
                mask=mask_e,
            )
        return

    num_col_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)
    page_pid = pid - 1
    out_row = page_pid // num_col_blocks
    col_block = page_pid - out_row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (out_row < expanded_size) & (offs_n < max_seqlen_k)

    req_row = out_row // next_n
    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=out_row < expanded_size,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    # Write the wide page_size=1 table only when the caller provides it (see
    # fused_dsa_decode_metadata for the optional-page_table_1 contract).
    if HAS_PAGE_TABLE_1:
        tl.store(
            page_table_1 + out_row * page_table_stride_0 + offs_n * page_table_stride_1,
            vals,
            mask=mask,
        )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + out_row * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // real_page_size,
            mask=real_mask,
        )


def fused_dsa_target_verify_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: torch.Tensor = None,
) -> None:
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert next_n > 0

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        assert page_table_1 is not None
        real_page_table = page_table_1

    # page_table_1 (the wide page_size=1 table) may be dropped for the fused
    # decode CUDA graph; the kernel then writes only real_page_table.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table  # dummy pointer for stride args
    else:
        assert page_table_1.is_cuda

    has_paged_mqa_ctx_lens = paged_mqa_ctx_lens_2d is not None
    if has_paged_mqa_ctx_lens:
        assert paged_mqa_ctx_lens_2d.is_cuda
        assert paged_mqa_ctx_lens_2d.dtype == torch.int32
        assert paged_mqa_ctx_lens_2d.dim() == 2
        assert paged_mqa_ctx_lens_2d.size(0) == bs
        assert paged_mqa_ctx_lens_2d.size(1) == next_n
    else:
        paged_mqa_ctx_lens_2d = page_table_1

    expanded_size = bs * next_n
    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(expanded_size)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    grid = (1 + expanded_size * num_col_blocks,)

    _fused_dsa_target_verify_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        paged_mqa_ctx_lens_2d,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        paged_mqa_ctx_lens_2d.stride(0) if has_paged_mqa_ctx_lens else 0,
        paged_mqa_ctx_lens_2d.stride(1) if has_paged_mqa_ctx_lens else 0,
        bs,
        max_seqlen_k,
        dsa_index_topk,
        real_page_size,
        next_n,
        has_real_page_table,
        has_paged_mqa_ctx_lens,
        has_page_table_1,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_N=block_n,
    )


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "total_len",
        "max_seqlen_k",
    ]
)
def _fused_dsa_draft_extend_metadata_kernel(
    seq_lens,
    extend_seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    extend_seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    total_len,
    max_seqlen_k,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGE_TABLE_1: tl.constexpr,
    STATIC_EXTEND_LEN: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32)
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < total_len
        if STATIC_EXTEND_LEN:
            static_qo_len = tl.load(extend_seq_lens).to(tl.int32)
            req_row = offs_e // static_qo_len
            local_off = offs_e - req_row * static_qo_len
            qo_len_for_row = tl.zeros((BLOCK_EXPANDED,), tl.int32) + static_qo_len
        else:
            req_row = tl.full((BLOCK_EXPANDED,), 0, tl.int32)
            local_off = tl.full((BLOCK_EXPANDED,), 0, tl.int32)
            qo_len_for_row = tl.full((BLOCK_EXPANDED,), 1, tl.int32)
            prefix = tl.full((), 0, tl.int32)

            for i in tl.range(0, bs):
                qo_len = tl.load(extend_seq_lens + i * extend_seq_lens_stride).to(
                    tl.int32
                )
                in_row = (offs_e >= prefix) & (offs_e < prefix + qo_len)
                req_row = tl.where(in_row, i, req_row)
                local_off = tl.where(in_row, offs_e - prefix, local_off)
                qo_len_for_row = tl.where(in_row, qo_len, qo_len_for_row)
                prefix += qo_len

        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        # Clamp to >= 0: DP-padded / idle-companion rows carry the CUDA-graph
        # seq_len fill value (1), which is smaller than qo_len, so the raw
        # per-row visible kv length goes negative. Consumers treat these
        # lengths as unsigned (the top-k v2 kernel reads them as uint32), so a
        # negative row becomes a ~4e9-token length and an illegal memory
        # access. 0 keeps padded rows on the trivial all-(-1) output path.
        expanded_seq = base_seq - qo_len_for_row + local_off + 1
        expanded_seq = tl.maximum(expanded_seq, 0)
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)
        return

    num_col_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)
    page_pid = pid - 1
    req_row = page_pid // num_col_blocks
    col_block = page_pid - req_row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    qo_len = tl.load(
        extend_seq_lens + req_row * extend_seq_lens_stride,
        mask=req_row < bs,
        other=0,
    ).to(tl.int32)
    if STATIC_EXTEND_LEN:
        prefix = req_row * qo_len
    else:
        prefix = tl.full((), 0, tl.int32)
        for i in tl.range(0, bs):
            prev_qo_len = tl.load(extend_seq_lens + i * extend_seq_lens_stride).to(
                tl.int32
            )
            prefix += tl.where(i < req_row, prev_qo_len, 0)
    offs_r = tl.arange(0, BLOCK_ROWS)
    out_rows = prefix + offs_r
    row_mask = (req_row < bs) & (offs_r < qo_len) & (out_rows < total_len)
    col_mask = offs_n < max_seqlen_k
    has_rows = (req_row < bs) & (qo_len > 0)
    mask = row_mask[:, None] & col_mask[None, :]

    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=has_rows,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=col_mask & has_rows,
        other=0,
    ).to(tl.int32)
    # Write the wide page_size=1 table only when the caller provides it (see
    # fused_dsa_decode_metadata for the optional-page_table_1 contract).
    if HAS_PAGE_TABLE_1:
        tl.store(
            page_table_1
            + out_rows[:, None] * page_table_stride_0
            + offs_n[None, :] * page_table_stride_1,
            vals[None, :],
            mask=mask,
        )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n[None, :] % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + out_rows[:, None] * real_page_table_stride_0
            + real_cols[None, :] * real_page_table_stride_1,
            (vals // real_page_size)[None, :],
            mask=real_mask,
        )


def fused_dsa_draft_extend_metadata(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    total_len: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
    max_extend_len: int,
    max_total_len: int,
    static_extend_len: bool = False,
) -> None:
    assert seq_lens.is_cuda
    assert extend_seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    if total_len == 0:
        cache = seq_lens.to(torch.int32)
        cache_seqlens.copy_(cache)
        cu_seqlens_k[:1].zero_()
        cu_seqlens_k[1 : bs + 1].copy_(torch.cumsum(cache, dim=0, dtype=torch.int32))
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert total_len <= max_total_len
    # Caller-owned graph metadata guarantees each request accepts at most
    # max_extend_len tokens. Avoid checking extend_seq_lens.max() here because
    # that would sync in the replay hot path.
    assert max_extend_len > 0
    assert total_len <= bs * max_extend_len

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        assert page_table_1 is not None
        real_page_table = page_table_1

    # page_table_1 (the wide page_size=1 table) may be dropped for the fused
    # decode CUDA graph; the kernel then writes only real_page_table.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        assert has_real_page_table
        page_table_1 = real_page_table  # dummy pointer for stride args
    else:
        assert page_table_1.is_cuda

    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(max_total_len)
    block_rows = triton.next_power_of_2(max_extend_len)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_draft_extend_metadata_kernel[grid](
        seq_lens,
        extend_seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        extend_seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        total_len,
        max_seqlen_k,
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        has_page_table_1,
        static_extend_len,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_ROWS=block_rows,
        BLOCK_N=block_n,
    )
