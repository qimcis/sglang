from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_STATE_SOURCES: Final[int] = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class StateSlotContext:
    """Graph-stable tensors consumed by the recurrent-state accessor."""

    sidecar: torch.Tensor
    allocation_generations: torch.Tensor
    expected_slots: torch.Tensor
    expected_generations: torch.Tensor
    failure_status: torch.Tensor
    enabled: torch.Tensor
    canary_violation_index: torch.Tensor
    canary_forward_start: torch.Tensor
    domain_seed: int


def _canonical_source(tensor: torch.Tensor, *, num_slots: int) -> torch.Tensor:
    if tensor.shape[0] != num_slots:
        raise ValueError(
            "state-protection source must use slot as its first dimension: "
            f"expected {num_slots}, got shape={tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            "state-protection source must be contiguous; kernel adapters must "
            "expose a graph-stable contiguous slot-major view"
        )
    return tensor.view(torch.uint8).reshape(num_slots, -1)


def _source_abi(
    sources: Sequence[torch.Tensor], *, num_slots: int, device: torch.device
) -> tuple[list[torch.Tensor], int]:
    if not 1 <= len(sources) <= MAX_STATE_SOURCES:
        raise ValueError(
            f"state protection supports 1..{MAX_STATE_SOURCES} source tensors per "
            f"kernel family, got {len(sources)}"
        )
    out = [_canonical_source(t, num_slots=num_slots) for t in sources]
    dummy = torch.empty((1, 1), dtype=torch.uint8, device=device)
    out.extend(dummy for _ in range(MAX_STATE_SOURCES - len(out)))
    return out, len(sources)


def validate_state_slots(
    *,
    context: StateSlotContext,
    request_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    sources: Sequence[torch.Tensor],
    require_sealed: torch.Tensor,
) -> None:
    """Validate and sanitize recurrent-state slots before a consumer reads them.

    ``cache_indices`` is intentionally mutable. A failed row is redirected to
    reserved slot zero on the same CUDA stream before the wrapped kernel runs.
    """

    if request_indices.dtype != torch.int64 or not request_indices.is_contiguous():
        raise ValueError(
            "state-protection request_indices must be a graph-stable contiguous int64 tensor"
        )
    if request_indices.ndim != 1:
        raise ValueError("state-protection request_indices must be one-dimensional")
    if cache_indices.dtype not in (torch.int32, torch.int64) or not cache_indices.is_contiguous():
        raise ValueError(
            "state-protection cache_indices must be a graph-stable contiguous int32/int64 tensor"
        )
    if request_indices.shape != cache_indices.shape:
        raise ValueError("request_indices and cache_indices must have identical shapes")
    if (
        require_sealed.dtype != torch.int32
        or require_sealed.shape != cache_indices.shape
        or not require_sealed.is_contiguous()
    ):
        raise ValueError(
            "state-protection require_sealed must be a graph-stable contiguous "
            "int32 tensor matching cache_indices"
        )
    source_abi, num_sources = _source_abi(
        sources,
        num_slots=int(context.sidecar.shape[0]),
        device=context.sidecar.device,
    )
    entrypoint = (
        _state_slot_module().validate_state_slots_i32
        if cache_indices.dtype == torch.int32
        else _state_slot_module().validate_state_slots_i64
    )
    entrypoint(
        context.sidecar,
        context.allocation_generations,
        request_indices,
        cache_indices,
        require_sealed,
        context.expected_slots,
        context.expected_generations,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        *source_abi,
        num_sources,
        context.domain_seed,
    )


def seal_state_slots(
    *,
    context: StateSlotContext,
    request_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    sources: Sequence[torch.Tensor],
) -> None:
    """Seal recurrent state after a kernel has committed its persistent writes."""

    if request_indices.dtype != torch.int64 or not request_indices.is_contiguous():
        raise ValueError(
            "state-protection request_indices must be a graph-stable contiguous int64 tensor"
        )
    if request_indices.ndim != 1:
        raise ValueError("state-protection request_indices must be one-dimensional")
    if cache_indices.dtype not in (torch.int32, torch.int64) or not cache_indices.is_contiguous():
        raise ValueError(
            "state-protection cache_indices must be a graph-stable contiguous int32/int64 tensor"
        )
    source_abi, num_sources = _source_abi(
        sources,
        num_slots=int(context.sidecar.shape[0]),
        device=context.sidecar.device,
    )
    entrypoint = (
        _state_slot_module().seal_state_slots_i32
        if cache_indices.dtype == torch.int32
        else _state_slot_module().seal_state_slots_i64
    )
    entrypoint(
        context.sidecar,
        context.allocation_generations,
        request_indices,
        cache_indices,
        context.expected_slots,
        context.expected_generations,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        *source_abi,
        num_sources,
        context.domain_seed,
    )


def seal_unbound_state_slots(
    *,
    context: StateSlotContext,
    request_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    active_mask: torch.Tensor,
    sources: Sequence[torch.Tensor],
) -> None:
    """Seal radix tracking slots that are not the request's active allocation."""

    if request_indices.dtype != torch.int64 or not request_indices.is_contiguous():
        raise ValueError("tracking request_indices must be contiguous int64")
    if cache_indices.dtype not in (torch.int32, torch.int64) or not cache_indices.is_contiguous():
        raise ValueError("tracking cache_indices must be contiguous int32/int64")
    if (
        active_mask.dtype != torch.int32
        or not active_mask.is_contiguous()
        or active_mask.shape != cache_indices.shape
        or request_indices.shape != cache_indices.shape
    ):
        raise ValueError("tracking active_mask must be contiguous int32 matching indices")
    source_abi, num_sources = _source_abi(
        sources,
        num_slots=int(context.sidecar.shape[0]),
        device=context.sidecar.device,
    )
    entrypoint = (
        _state_slot_module().seal_unbound_state_slots_i32
        if cache_indices.dtype == torch.int32
        else _state_slot_module().seal_unbound_state_slots_i64
    )
    entrypoint(
        context.sidecar,
        context.allocation_generations,
        request_indices,
        cache_indices,
        active_mask,
        context.expected_slots,
        context.expected_generations,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
        *source_abi,
        num_sources,
        context.domain_seed,
    )


def validate_unbound_state_slots(
    *,
    context: StateSlotContext,
    request_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    active_mask: torch.Tensor,
    expected_slots: torch.Tensor,
    expected_generations: torch.Tensor,
) -> None:
    """Validate and sanitize graph-stable radix tracking destinations."""

    if request_indices.dtype != torch.int64 or not request_indices.is_contiguous():
        raise ValueError("tracking request_indices must be contiguous int64")
    if cache_indices.dtype not in (torch.int32, torch.int64) or not cache_indices.is_contiguous():
        raise ValueError("tracking cache_indices must be contiguous int32/int64")
    if (
        active_mask.dtype != torch.int32
        or not active_mask.is_contiguous()
        or active_mask.shape != cache_indices.shape
        or request_indices.shape != cache_indices.shape
    ):
        raise ValueError("tracking active_mask must be contiguous int32 matching indices")
    if (
        expected_slots.dtype != torch.int32
        or expected_generations.dtype != torch.int64
        or expected_slots.shape != expected_generations.shape
        or expected_slots.ndim != 2
        or expected_slots.shape[0] != context.failure_status.shape[0]
        or not expected_slots.is_contiguous()
        or not expected_generations.is_contiguous()
    ):
        raise ValueError("tracking ownership tables have an invalid layout")
    entrypoint = (
        _state_slot_module().validate_unbound_state_slots_i32
        if cache_indices.dtype == torch.int32
        else _state_slot_module().validate_unbound_state_slots_i64
    )
    entrypoint(
        context.allocation_generations,
        request_indices,
        cache_indices,
        active_mask,
        expected_slots,
        expected_generations,
        context.failure_status,
        context.enabled,
        context.canary_violation_index,
        context.canary_forward_start,
    )


@cache_once
def _state_slot_module() -> Module:
    return load_jit(
        "state_protection_state_slot",
        cuda_files=["state_protection/state_slot.cuh"],
        cuda_wrappers=[
            ("validate_state_slots_i32", "state_protection::validate_state_slots_i32"),
            ("validate_state_slots_i64", "state_protection::validate_state_slots_i64"),
            ("seal_state_slots_i32", "state_protection::seal_state_slots_i32"),
            ("seal_state_slots_i64", "state_protection::seal_state_slots_i64"),
            (
                "seal_unbound_state_slots_i32",
                "state_protection::seal_unbound_state_slots_i32",
            ),
            (
                "seal_unbound_state_slots_i64",
                "state_protection::seal_unbound_state_slots_i64",
            ),
            (
                "validate_unbound_state_slots_i32",
                "state_protection::validate_unbound_state_slots_i32",
            ),
            (
                "validate_unbound_state_slots_i64",
                "state_protection::validate_unbound_state_slots_i64",
            ),
        ],
    )
