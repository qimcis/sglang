from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl


def _inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA inputs take the fused triton kernel; CPU inputs take the
    torch reference implementation (triton is CUDA-only, and CPU-side callers
    such as unit tests exercise the reference path)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda
    raise AssertionError("kernel dispatch requires at least one tensor argument")


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:
    @classmethod
    def execute(cls, *args, **kwargs) -> ExpandPrefillCausallyResult:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally_triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )


def expand_prefill_causally(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[list[int]],
    extend_seq_lens_cpu: Optional[list[int]],
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    device = req_pool_indices.device
    cuda_int32_kwargs = {"dtype": torch.int32, "device": device}

    if extend_start_loc is not None:
        repeats = extend_seq_lens.to(torch.int64)
        req_pool_indices_repeated = torch.repeat_interleave(
            req_pool_indices, repeats, output_size=num_tokens
        )
        start_positions = seq_lens.to(torch.int32) - extend_seq_lens.to(torch.int32) + 1
        start_positions_repeated = torch.repeat_interleave(
            start_positions, repeats, output_size=num_tokens
        )
        start_locs_repeated = torch.repeat_interleave(
            extend_start_loc.to(torch.int32), repeats, output_size=num_tokens
        )
        token_offsets = (
            torch.arange(num_tokens, **cuda_int32_kwargs) - start_locs_repeated
        )
        seq_lens_casual = start_positions_repeated + token_offsets

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual, (0, pad_size), value=1
            )
            req_pool_indices_repeated = torch.cat(
                (
                    req_pool_indices_repeated,
                    req_pool_indices_repeated[-1:].expand(pad_size),
                )
            )
        return ExpandPrefillCausallyResult(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

    assert seq_lens_cpu is not None and extend_seq_lens_cpu is not None
    seq_lens_casual = torch.empty(num_tokens, **cuda_int32_kwargs)
    idx_to_req_repeated = torch.empty(num_tokens, **cuda_int32_kwargs)
    offset = 0
    for i, (kv_len, qo_len) in enumerate(zip(seq_lens_cpu, extend_seq_lens_cpu)):
        out = seq_lens_casual[offset : offset + qo_len]
        offset += qo_len
        torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
        idx_to_req_repeated[offset - qo_len : offset].fill_(i)

    assert offset == num_tokens
    req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

    if padded_num_tokens is not None and padded_num_tokens > num_tokens:
        pad_size = padded_num_tokens - num_tokens
        seq_lens_casual = torch.nn.functional.pad(
            seq_lens_casual, (0, pad_size), value=1
        )
        req_pool_indices_repeated = torch.nn.functional.pad(
            req_pool_indices_repeated,
            (0, pad_size),
            value=req_pool_indices_repeated[-1].item(),
        )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


@triton.jit
def _expand_prefill_causally_kernel(
    req_pool_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    seq_lens_casual_ptr,
    req_pool_repeated_ptr,
    bs,
    num_tokens,
    total_tokens,
    BLOCK: tl.constexpr,
    BS_P2: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_tokens

    b = tl.arange(0, BS_P2)
    bmask = b < bs
    extend = tl.load(extend_seq_lens_ptr + b, mask=bmask, other=0).to(tl.int32)
    start_locs = tl.cumsum(extend, axis=0) - extend

    is_real = offs < num_tokens
    t = tl.where(is_real, offs, 0).to(tl.int32)
    started = (start_locs[None, :] <= t[:, None]) & bmask[None, :]
    r = tl.sum(started.to(tl.int32), axis=1) - 1
    r = tl.where(is_real, r, bs - 1).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    ext = tl.load(extend_seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    start_loc = tl.sum(tl.where(started, extend[None, :], 0).to(tl.int32), axis=1) - ext
    causal = (seq_len - ext + 1) + (t - start_loc)
    causal = tl.where(is_real, causal, 1)

    rp = tl.load(req_pool_ptr + r, mask=mask, other=0)
    tl.store(seq_lens_casual_ptr + offs, causal, mask=mask)
    tl.store(req_pool_repeated_ptr + offs, rp, mask=mask)


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )

    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=device
    )
    BLOCK = 256
    _expand_prefill_causally_kernel[(triton.cdiv(total_tokens, BLOCK),)](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        seq_lens_casual,
        req_pool_indices_repeated,
        bs,
        num_tokens,
        total_tokens,
        BLOCK=BLOCK,
        BS_P2=triton.next_power_of_2(max(bs, 1)),
    )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


class PageTablePositionsResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor
    page_table: torch.Tensor
    swa_topk_lengths: torch.Tensor


class BuildPageTablePositions:
    @classmethod
    def execute(cls, *args, **kwargs) -> PageTablePositionsResult:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
        integrity_args=None,
    ) -> PageTablePositionsResult:
        if integrity_args is not None:
            raise ValueError("DSV4 integrity metadata validation requires CUDA")
        return build_page_table_positions(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
        integrity_args=None,
    ) -> PageTablePositionsResult:
        return build_page_table_positions_triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
            integrity_args=integrity_args,
        )


def build_page_table_positions(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    positions_casual = seq_lens_casual - 1
    page_table = req_to_token[
        req_pool_indices_repeated.to(torch.int64), :max_seq_len:page_size
    ]
    page_table = (page_table // page_size).to(torch.int32)
    swa_topk_lengths = torch.clamp(seq_lens_casual, max=swa_window)
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_casual,
        positions_casual=positions_casual,
        page_table=page_table,
        swa_topk_lengths=swa_topk_lengths,
    )


@triton.jit
def _page_table_positions_kernel(
    req_to_token_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    seq_lens_out_ptr,
    positions_out_ptr,
    page_table_ptr,
    topk_out_ptr,
    generations_ptr,
    request_epochs_ptr,
    expected_tags_ptr,
    failure_status_ptr,
    rt_stride,
    num_pages,
    max_seq_len,
    page_size,
    swa_window,
    physical_capacity: tl.constexpr,
    request_capacity: tl.constexpr,
    logical_capacity: tl.constexpr,
    mapping_seed,
    PROTECT: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    raw_seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    trusted_seq_limit = tl.minimum(max_seq_len, rt_stride)
    seq_len = tl.maximum(0, tl.minimum(raw_seq_len, trusted_seq_limit))
    tl.store(seq_lens_out_ptr + row, seq_len)
    tl.store(positions_out_ptr + row, tl.maximum(seq_len - 1, 0))
    tl.store(topk_out_ptr + row, tl.minimum(seq_len, swa_window))

    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    if PROTECT:
        safe_rp = tl.where((rp > 0) & (rp < request_capacity), rp, 0)
    else:
        safe_rp = rp
    base = req_to_token_ptr + safe_rp * rt_stride
    out_base = page_table_ptr + row.to(tl.int64) * num_pages
    for p0 in range(0, num_pages, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        page_start = p.to(tl.int64) * page_size
        pmask = p < num_pages
        load_mask = pmask & (page_start < max_seq_len) & (page_start < rt_stride)
        tok = tl.load(base + page_start, mask=load_mask, other=0).to(tl.int32)
        page = tok // page_size
        if PROTECT:
            live = load_mask & (page_start < seq_len)
            request_in_range = (rp > 0) & (rp < request_capacity)
            page_in_range = (page > 0) & (page < physical_capacity)
            logical_in_range = p < logical_capacity
            safe_request = tl.where(request_in_range, rp, 0)
            safe_page = tl.where(page_in_range, page, 0)
            safe_logical = tl.where(logical_in_range, p, 0)
            generation = tl.load(generations_ptr + safe_page)
            request_epoch = tl.load(request_epochs_ptr + safe_request)
            expected_tag = tl.load(
                expected_tags_ptr
                + safe_request * logical_capacity
                + safe_logical.to(tl.int64)
            ).to(tl.uint64)

            value = mapping_seed.to(tl.uint64) ^ 0x445356344D415032
            value ^= safe_request.to(tl.uint64) * 0x9E3779B97F4A7C15
            value ^= request_epoch.to(tl.uint64) * 0xBF58476D1CE4E5B9
            value ^= safe_logical.to(tl.uint64) * 0x94D049BB133111EB
            value ^= safe_page.to(tl.uint64) * 0xD6E8FEB86659FD93
            value ^= generation.to(tl.uint64) * 0xA0761D6478BD642F
            value ^= value >> 30
            value *= 0xBF58476D1CE4E5B9
            value ^= value >> 27
            value *= 0x94D049BB133111EB
            actual_tag = value ^ (value >> 31)
            actual_tag = tl.where(actual_tag == 0, 1, actual_tag)

            failure = tl.where(request_in_range, 0, 1 << 6)
            failure = tl.where(
                request_in_range & (~page_in_range | ~logical_in_range),
                1 << 0,
                failure,
            )
            valid_geometry = request_in_range & page_in_range & logical_in_range
            failure = tl.where(
                valid_geometry & ((generation == 0) | (request_epoch == 0)),
                1 << 5,
                failure,
            )
            live_identity = valid_geometry & (generation != 0) & (request_epoch != 0)
            failure = tl.where(live_identity & (expected_tag == 0), 1 << 3, failure)
            failure = tl.where(
                live_identity & (expected_tag != 0) & (expected_tag != actual_tag),
                1 << 4,
                failure,
            )
            report = live & request_in_range & (failure != 0)
            tl.atomic_or(
                failure_status_ptr + safe_request + tl.zeros_like(failure).to(tl.int64),
                failure.to(tl.int32),
                mask=report,
            )
            page = tl.where(live & (failure != 0), 0, page)
        tl.store(out_base + p, page, mask=pmask)


def build_page_table_positions_triton(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
    integrity_args=None,
) -> PageTablePositionsResult:
    num_q = seq_lens_casual.shape[0]
    num_pages = (max_seq_len + page_size - 1) // page_size
    device = seq_lens_casual.device

    seq_lens_out = torch.empty(num_q, dtype=torch.int32, device=device)
    positions_out = torch.empty(num_q, dtype=torch.int32, device=device)
    page_table = torch.empty((num_q, num_pages), dtype=torch.int32, device=device)
    topk_out = torch.empty(num_q, dtype=torch.int32, device=device)
    protect = integrity_args is not None
    if protect:
        generations = integrity_args["full_generations"]
        request_epochs = integrity_args["request_epochs"]
        expected_tags = integrity_args["full_tags"]
        failure_status = integrity_args["failure_status"]
        mapping_seed = integrity_args["full_seed"]
    else:
        generations = req_to_token
        request_epochs = req_to_token
        expected_tags = req_to_token
        failure_status = req_to_token
        mapping_seed = 0
    BLOCK_P = 256
    _page_table_positions_kernel[(num_q,)](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        seq_lens_out,
        positions_out,
        page_table,
        topk_out,
        generations,
        request_epochs,
        expected_tags,
        failure_status,
        req_to_token.stride(0),
        num_pages,
        max_seq_len,
        page_size,
        swa_window,
        physical_capacity=generations.numel() if protect else 0,
        request_capacity=request_epochs.numel() if protect else 0,
        logical_capacity=expected_tags.shape[1] if protect else 0,
        mapping_seed=mapping_seed,
        PROTECT=protect,
        BLOCK_P=BLOCK_P,
    )
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_out,
        positions_casual=positions_out,
        page_table=page_table,
        swa_topk_lengths=topk_out,
    )


class BuildCausalSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
        integrity_args=None,
    ) -> torch.Tensor:
        if integrity_args is not None:
            raise ValueError("DSV4 integrity metadata validation requires CUDA")
        return build_causal_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
        integrity_args=None,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
            integrity_args=integrity_args,
        )


def build_causal_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    device = seq_lens_casual.device
    pos_causal = seq_lens_casual - 1
    num_qo_tokens = seq_lens_casual.size(0)
    offsets = pos_causal.unsqueeze(1) - torch.arange(
        swa_window, dtype=torch.int32, device=device
    ).unsqueeze(0)
    invalid_offset_mask = offsets < 0
    offsets.masked_fill_(invalid_offset_mask, 0)
    raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
    assert raw_indices.shape == (num_qo_tokens, swa_window)
    raw_indices.masked_fill_(invalid_offset_mask, -1)
    swa_indices = full_to_swa_mapping[raw_indices]
    swa_indices = swa_indices.to(torch.int32)

    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    if padded_width == swa_window:
        return swa_indices
    return torch.nn.functional.pad(
        swa_indices, (0, padded_width - swa_window), value=-1
    )


@triton.jit
def _causal_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    out_ptr,
    full_generations_ptr,
    swa_generations_ptr,
    request_epochs_ptr,
    full_tags_ptr,
    swa_tags_ptr,
    failure_status_ptr,
    rt_stride,
    full_to_swa_capacity,
    swa_window,
    padded_width,
    full_page_size: tl.constexpr,
    swa_page_size: tl.constexpr,
    full_physical_capacity: tl.constexpr,
    swa_physical_capacity: tl.constexpr,
    request_capacity: tl.constexpr,
    full_logical_capacity: tl.constexpr,
    swa_logical_capacity: tl.constexpr,
    full_seed,
    swa_seed,
    PROTECT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    raw_seq_len = tl.load(seq_lens_ptr + row).to(tl.int64)
    seq_len = tl.maximum(0, tl.minimum(raw_seq_len, rt_stride))
    pos = tl.maximum(seq_len - 1, 0)
    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    if PROTECT:
        safe_rp = tl.where((rp > 0) & (rp < request_capacity), rp, 0)
    else:
        safe_rp = rp
    base = req_to_token_ptr + safe_rp * rt_stride
    out_base = out_ptr + row.to(tl.int64) * padded_width

    for k0 in range(0, padded_width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < padded_width
        off = pos - k.to(tl.int64)
        valid = (
            (seq_len > 0) & (k < swa_window) & (off >= 0) & (off < rt_stride) & kmask
        )
        full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
            tl.int64
        )
        safe_full_loc = tl.where(
            valid & (full_loc >= 0) & (full_loc < full_to_swa_capacity),
            full_loc,
            0,
        )
        swa = tl.load(
            full_to_swa_ptr + safe_full_loc,
            mask=valid,
            other=-1,
        ).to(tl.int32)
        if PROTECT:
            request_in_range = (rp > 0) & (rp < request_capacity)
            full_page = full_loc // full_page_size
            swa_page = swa // swa_page_size
            full_logical = off // full_page_size
            swa_logical = off // swa_page_size
            full_page_in_range = (full_page > 0) & (full_page < full_physical_capacity)
            swa_page_in_range = (swa_page > 0) & (swa_page < swa_physical_capacity)
            full_logical_in_range = (full_logical >= 0) & (
                full_logical < full_logical_capacity
            )
            swa_logical_in_range = (swa_logical >= 0) & (
                swa_logical < swa_logical_capacity
            )
            full_loc_in_range = (full_loc >= 0) & (full_loc < full_to_swa_capacity)
            safe_request = tl.where(request_in_range, rp, 0)
            safe_full_page = tl.where(full_page_in_range, full_page, 0)
            safe_swa_page = tl.where(swa_page_in_range, swa_page, 0)
            safe_full_logical = tl.where(full_logical_in_range, full_logical, 0)
            safe_swa_logical = tl.where(swa_logical_in_range, swa_logical, 0)
            request_epoch = tl.load(request_epochs_ptr + safe_request)
            full_generation = tl.load(full_generations_ptr + safe_full_page)
            swa_generation = tl.load(swa_generations_ptr + safe_swa_page)
            full_expected = tl.load(
                full_tags_ptr + safe_request * full_logical_capacity + safe_full_logical
            ).to(tl.uint64)
            swa_expected = tl.load(
                swa_tags_ptr + safe_request * swa_logical_capacity + safe_swa_logical
            ).to(tl.uint64)

            full_value = full_seed.to(tl.uint64) ^ 0x445356344D415032
            full_value ^= safe_request.to(tl.uint64) * 0x9E3779B97F4A7C15
            full_value ^= request_epoch.to(tl.uint64) * 0xBF58476D1CE4E5B9
            full_value ^= safe_full_logical.to(tl.uint64) * 0x94D049BB133111EB
            full_value ^= safe_full_page.to(tl.uint64) * 0xD6E8FEB86659FD93
            full_value ^= full_generation.to(tl.uint64) * 0xA0761D6478BD642F
            full_value ^= full_value >> 30
            full_value *= 0xBF58476D1CE4E5B9
            full_value ^= full_value >> 27
            full_value *= 0x94D049BB133111EB
            full_actual = full_value ^ (full_value >> 31)
            full_actual = tl.where(full_actual == 0, 1, full_actual)

            swa_value = swa_seed.to(tl.uint64) ^ 0x445356344D415032
            swa_value ^= safe_request.to(tl.uint64) * 0x9E3779B97F4A7C15
            swa_value ^= request_epoch.to(tl.uint64) * 0xBF58476D1CE4E5B9
            swa_value ^= safe_swa_logical.to(tl.uint64) * 0x94D049BB133111EB
            swa_value ^= safe_swa_page.to(tl.uint64) * 0xD6E8FEB86659FD93
            swa_value ^= swa_generation.to(tl.uint64) * 0xA0761D6478BD642F
            swa_value ^= swa_value >> 30
            swa_value *= 0xBF58476D1CE4E5B9
            swa_value ^= swa_value >> 27
            swa_value *= 0x94D049BB133111EB
            swa_actual = swa_value ^ (swa_value >> 31)
            swa_actual = tl.where(swa_actual == 0, 1, swa_actual)

            full_geometry = (
                request_in_range
                & full_loc_in_range
                & full_page_in_range
                & full_logical_in_range
            )
            swa_geometry = request_in_range & swa_page_in_range & swa_logical_in_range
            failure = tl.where(request_in_range, 0, 1 << 6)
            failure = tl.where(
                request_in_range & (~full_geometry | ~swa_geometry),
                1 << 0,
                failure,
            )
            generations_valid = (
                (request_epoch != 0) & (full_generation != 0) & (swa_generation != 0)
            )
            failure = tl.where(
                full_geometry & swa_geometry & ~generations_valid,
                1 << 5,
                failure,
            )
            identity_valid = full_geometry & swa_geometry & generations_valid
            failure = tl.where(
                identity_valid & ((full_expected == 0) | (swa_expected == 0)),
                1 << 3,
                failure,
            )
            failure = tl.where(
                identity_valid
                & (full_expected != 0)
                & (swa_expected != 0)
                & ((full_expected != full_actual) | (swa_expected != swa_actual)),
                1 << 4,
                failure,
            )
            report = valid & request_in_range & (failure != 0)
            tl.atomic_or(
                failure_status_ptr + safe_request + tl.zeros_like(failure).to(tl.int64),
                failure.to(tl.int32),
                mask=report,
            )
            swa = tl.where(valid & (failure != 0), 0, swa)
        tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
    integrity_args=None,
) -> torch.Tensor:
    num_qo_tokens = seq_lens_casual.size(0)
    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    out = torch.empty(
        (num_qo_tokens, padded_width),
        dtype=torch.int32,
        device=seq_lens_casual.device,
    )
    protect = integrity_args is not None
    if protect:
        full_generations = integrity_args["full_generations"]
        swa_generations = integrity_args["swa_generations"]
        request_epochs = integrity_args["request_epochs"]
        full_tags = integrity_args["full_tags"]
        swa_tags = integrity_args["swa_tags"]
        failure_status = integrity_args["failure_status"]
        full_seed = integrity_args["full_seed"]
        swa_seed = integrity_args["swa_seed"]
        full_page_size = integrity_args["full_page_size"]
        swa_page_size = integrity_args["swa_page_size"]
    else:
        full_generations = req_to_token
        swa_generations = req_to_token
        request_epochs = req_to_token
        full_tags = req_to_token
        swa_tags = req_to_token
        failure_status = req_to_token
        full_seed = 0
        swa_seed = 0
        full_page_size = 1
        swa_page_size = 1
    BLOCK_K = 256
    _causal_swa_page_indices_kernel[(num_qo_tokens,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        out,
        full_generations,
        swa_generations,
        request_epochs,
        full_tags,
        swa_tags,
        failure_status,
        req_to_token.stride(0),
        full_to_swa_mapping.numel(),
        swa_window,
        padded_width,
        full_page_size=full_page_size,
        swa_page_size=swa_page_size,
        full_physical_capacity=full_generations.numel() if protect else 0,
        swa_physical_capacity=swa_generations.numel() if protect else 0,
        request_capacity=request_epochs.numel() if protect else 0,
        full_logical_capacity=full_tags.shape[1] if protect else 0,
        swa_logical_capacity=swa_tags.shape[1] if protect else 0,
        full_seed=full_seed,
        swa_seed=swa_seed,
        PROTECT=protect,
        BLOCK_K=BLOCK_K,
    )
    return out
