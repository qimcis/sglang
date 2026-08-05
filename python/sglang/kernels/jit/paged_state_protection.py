from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_PAGED_SOURCES = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class PagedSlotContext:
    sidecar: torch.Tensor
    req_to_token: torch.Tensor
    failure_status: torch.Tensor
    enabled: torch.Tensor
    canary_violation_index: torch.Tensor
    canary_forward_start: torch.Tensor
    sources: tuple[torch.Tensor, ...]
    domain_seed: int
    swa_lut: torch.Tensor | None = None
    swa_window_size: int = 0
    dsa_source: torch.Tensor | None = None
    dsa_page_size: int = 0
    dsa_token_bytes: int = 0
    dsa_aux_bytes: int = 0


def _canonical_sources(
    sources: Sequence[torch.Tensor], *, num_slots: int
) -> tuple[list[torch.Tensor], int]:
    if not 1 <= len(sources) <= MAX_PAGED_SOURCES:
        raise ValueError(
            f"paged state protection supports 1..{MAX_PAGED_SOURCES} source "
            f"tensors per layer, got {len(sources)}"
        )
    canonical: list[torch.Tensor] = []
    for source in sources:
        if source.shape[0] != num_slots or not source.is_contiguous():
            raise ValueError(
                "paged state protection requires contiguous slot-major cache "
                f"rows; got shape={tuple(source.shape)}, contiguous={source.is_contiguous()}"
            )
        canonical.append(source.view(torch.uint8).reshape(num_slots, -1))
    dummy = torch.empty((1, 1), dtype=torch.uint8, device=sources[0].device)
    canonical.extend(dummy for _ in range(MAX_PAGED_SOURCES - len(canonical)))
    return canonical, len(sources)


def _validate_common(
    *,
    context: PagedSlotContext,
    request_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> tuple[list[torch.Tensor], int, torch.Tensor, torch.Tensor]:
    if request_indices.dtype != torch.int64 or not request_indices.is_contiguous():
        raise ValueError("request_indices must be contiguous int64")
    if prefix_lens.dtype != torch.int32 or not prefix_lens.is_contiguous():
        raise ValueError("prefix_lens must be contiguous int32")
    if request_indices.shape != prefix_lens.shape:
        raise ValueError("request_indices and prefix_lens must have identical shapes")
    if context.req_to_token.dtype != torch.int32 or not context.req_to_token.is_contiguous():
        raise ValueError("req_to_token must be contiguous int32")
    if (
        context.sidecar.dtype != torch.int64
        or context.sidecar.ndim != 2
        or context.sidecar.shape[1] != 3
        or not context.sidecar.is_contiguous()
    ):
        raise ValueError("sidecar must be contiguous int64 with shape [num_slots, 3]")
    if (
        context.failure_status.dtype != torch.int32
        or context.failure_status.ndim != 1
        or not context.failure_status.is_contiguous()
    ):
        raise ValueError("failure_status must be contiguous one-dimensional int32")
    if (
        context.enabled.dtype != torch.int32
        or tuple(context.enabled.shape) != (1,)
        or not context.enabled.is_contiguous()
    ):
        raise ValueError("enabled must be contiguous int32 with shape [1]")
    for name, counter in (
        ("canary_violation_index", context.canary_violation_index),
        ("canary_forward_start", context.canary_forward_start),
    ):
        if (
            counter.dtype != torch.int32
            or tuple(counter.shape) != (1,)
            or not counter.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 with shape [1]")
    sources, num_sources = _canonical_sources(
        context.sources, num_slots=int(context.sidecar.shape[0])
    )
    swa_lut = (
        context.swa_lut
        if context.swa_lut is not None
        else torch.empty(1, dtype=torch.int64, device=context.sidecar.device)
    )
    dsa_source = (
        context.dsa_source.view(torch.uint8).reshape(context.dsa_source.shape[0], -1)
        if context.dsa_source is not None
        else torch.empty((1, 1), dtype=torch.uint8, device=context.sidecar.device)
    )
    if not dsa_source.is_contiguous():
        raise ValueError("DSA protected source must expose contiguous page rows")
    if context.dsa_source is not None:
        if context.dsa_page_size <= 0 or context.dsa_token_bytes <= 0:
            raise ValueError("DSA protected source has invalid page/token geometry")
        expected_rows = (
            int(context.sidecar.shape[0]) + context.dsa_page_size - 1
        ) // context.dsa_page_size
        if dsa_source.shape[0] < expected_rows:
            raise ValueError("DSA protected source has fewer rows than the paged cache")
        expected_row_bytes = context.dsa_page_size * (
            context.dsa_token_bytes + context.dsa_aux_bytes
        )
        if dsa_source.shape[1] < expected_row_bytes:
            raise ValueError("DSA protected source row is smaller than its declared layout")
    return sources, num_sources, swa_lut, dsa_source


def validate_paged_mapping(
    *,
    context: PagedSlotContext,
    request_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> None:
    sources, num_sources, swa_lut, dsa_source = _validate_common(
        context=context,
        request_indices=request_indices,
        prefix_lens=prefix_lens,
    )
    _paged_slot_module().validate_mapping(
        context.sidecar,
        context.req_to_token,
        request_indices,
        prefix_lens,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        swa_lut,
        *sources,
        dsa_source,
        num_sources,
        context.domain_seed,
        context.swa_window_size,
        context.dsa_page_size,
        context.dsa_token_bytes,
        context.dsa_aux_bytes,
        context.swa_lut is not None,
        context.dsa_source is not None,
    )


def validate_paged_payload(
    *,
    context: PagedSlotContext,
    request_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> None:
    sources, num_sources, swa_lut, dsa_source = _validate_common(
        context=context,
        request_indices=request_indices,
        prefix_lens=prefix_lens,
    )
    _paged_slot_module().validate_payload(
        context.sidecar,
        context.req_to_token,
        request_indices,
        prefix_lens,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        swa_lut,
        *sources,
        dsa_source,
        num_sources,
        context.domain_seed,
        context.swa_window_size,
        context.dsa_page_size,
        context.dsa_token_bytes,
        context.dsa_aux_bytes,
        context.swa_lut is not None,
        context.dsa_source is not None,
    )


def validate_paged_write_slots(
    *,
    context: PagedSlotContext,
    request_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    write_slots: torch.Tensor,
    write_lens: torch.Tensor,
) -> None:
    """Sanitize persistent write destinations before any cache kernel sees them."""

    sources, num_sources, swa_lut, dsa_source = _validate_common(
        context=context,
        request_indices=request_indices,
        prefix_lens=prefix_lens,
    )
    if write_slots.dtype not in (torch.int32, torch.int64) or not write_slots.is_contiguous():
        raise ValueError("write_slots must be contiguous int32/int64")
    if (
        write_lens.dtype != torch.int32
        or write_lens.shape != request_indices.shape
        or not write_lens.is_contiguous()
    ):
        raise ValueError(
            "write_lens must be contiguous int32 matching request_indices"
        )
    entrypoint = (
        _paged_slot_module().validate_write_slots_i32
        if write_slots.dtype == torch.int32
        else _paged_slot_module().validate_write_slots_i64
    )
    entrypoint(
        context.sidecar,
        context.req_to_token,
        request_indices,
        prefix_lens,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        swa_lut,
        *sources,
        dsa_source,
        write_slots,
        write_lens,
        num_sources,
        context.domain_seed,
        context.swa_window_size,
        context.dsa_page_size,
        context.dsa_token_bytes,
        context.dsa_aux_bytes,
        context.swa_lut is not None,
        context.dsa_source is not None,
    )


def seal_paged_payload(
    *,
    context: PagedSlotContext,
    request_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    write_slots: torch.Tensor,
    write_lens: torch.Tensor,
) -> None:
    sources, num_sources, swa_lut, dsa_source = _validate_common(
        context=context,
        request_indices=request_indices,
        prefix_lens=prefix_lens,
    )
    if write_slots.dtype not in (torch.int32, torch.int64) or not write_slots.is_contiguous():
        raise ValueError("write_slots must be contiguous int32/int64")
    if (
        write_lens.dtype != torch.int32
        or write_lens.shape != request_indices.shape
        or not write_lens.is_contiguous()
    ):
        raise ValueError(
            "write_lens must be contiguous int32 matching request_indices"
        )
    entrypoint = (
        _paged_slot_module().seal_payload_i32
        if write_slots.dtype == torch.int32
        else _paged_slot_module().seal_payload_i64
    )
    entrypoint(
        context.sidecar,
        context.req_to_token,
        request_indices,
        prefix_lens,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        swa_lut,
        *sources,
        dsa_source,
        write_slots,
        write_lens,
        num_sources,
        context.domain_seed,
        context.swa_window_size,
        context.dsa_page_size,
        context.dsa_token_bytes,
        context.dsa_aux_bytes,
        context.swa_lut is not None,
        context.dsa_source is not None,
    )


@cache_once
def _paged_slot_module() -> Module:
    return load_jit(
        "state_protection_paged_slot",
        cuda_files=["state_protection/paged_slot.cuh"],
        cuda_wrappers=[
            ("validate_mapping", "state_protection::paged::validate_mapping"),
            ("validate_payload", "state_protection::paged::validate_payload"),
            (
                "validate_write_slots_i32",
                "state_protection::paged::validate_write_slots_i32",
            ),
            (
                "validate_write_slots_i64",
                "state_protection::paged::validate_write_slots_i64",
            ),
            ("seal_payload_i32", "state_protection::paged::seal_payload_i32"),
            ("seal_payload_i64", "state_protection::paged::seal_payload_i64"),
        ],
    )
