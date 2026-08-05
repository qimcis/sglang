from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch

from sglang.kernels.jit.state_protection import (
    StateSlotContext,
    seal_state_slots,
    seal_unbound_state_slots,
    validate_unbound_state_slots,
    validate_state_slots,
)


def _domain_seed(name: str) -> int:
    # Stable FNV-1a; Python's hash is process-randomized and unsuitable for a
    # transfer-visible domain identity.
    value = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


@dataclass(frozen=True, slots=True, kw_only=True)
class RecurrentLayerGuard:
    request_indices: torch.Tensor
    cache_indices: torch.Tensor
    sources: tuple[torch.Tensor, ...]
    local_layer: int


class RecurrentStateProtection:
    """Protect one request-indexed conv/SSM state pool.

    Mapping identity is stored per request, allocation generations per physical
    slot, and payload digests per (layer, slot). This covers KDA, GDN, Mamba and
    short-convolution kernels without making the core aware of their equations.
    """

    def __init__(
        self,
        *,
        req_to_token_pool,
        failure_status: torch.Tensor,
        enabled: torch.Tensor,
        canary_violation_index: torch.Tensor,
        canary_forward_start: torch.Tensor,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.mamba_pool = req_to_token_pool.mamba_pool
        self.allocator = req_to_token_pool.mamba_allocator
        self.device = torch.device(req_to_token_pool.device)
        self.num_slots = int(self.mamba_pool.size) + 1
        self.num_layers = int(self.mamba_pool.num_mamba_layers)
        self.sidecars = torch.zeros(
            (self.num_layers, self.num_slots, 2),
            dtype=torch.int64,
            device=self.device,
        )
        self.allocation_generations = torch.zeros(
            self.num_slots, dtype=torch.int64, device=self.device
        )
        req_slots = int(req_to_token_pool.req_to_token.shape[0])
        self.expected_slots = torch.zeros(
            req_slots, dtype=torch.int32, device=self.device
        )
        self.expected_generations = torch.zeros(
            req_slots, dtype=torch.int64, device=self.device
        )
        tracking_mapping = getattr(
            req_to_token_pool,
            "req_index_to_mamba_ping_pong_track_buffer_mapping",
            None,
        )
        if tracking_mapping is None:
            self.expected_tracking_slots = None
            self.expected_tracking_generations = None
        else:
            self.expected_tracking_slots = torch.zeros(
                tracking_mapping.shape, dtype=torch.int32, device=self.device
            )
            self.expected_tracking_generations = torch.zeros(
                tracking_mapping.shape, dtype=torch.int64, device=self.device
            )
        self.failure_status = failure_status
        self.enabled = enabled
        self.canary_violation_index = canary_violation_index
        self.canary_forward_start = canary_forward_start
        self.seed = _domain_seed("recurrent-state")
        self._validate_layouts()
        self.allocator.attach_state_protection(self)
        req_to_token_pool.recurrent_state_protection = self
        self.mamba_pool.state_protection = self

    def _validate_layouts(self) -> None:
        for global_layer_id, local_layer in self.req_to_token_pool.mamba_map.items():
            if local_layer < 0 or local_layer >= self.num_layers:
                raise RuntimeError(
                    f"recurrent layer {global_layer_id} maps to invalid local "
                    f"layer {local_layer}"
                )
            for source in self.layer_sources(global_layer_id):
                if source.shape[0] != self.num_slots or not source.is_contiguous():
                    raise RuntimeError(
                        "--enable-state-protection requires graph-stable contiguous "
                        "slot-major recurrent state; "
                        f"layer={global_layer_id}, shape={tuple(source.shape)}, "
                        f"contiguous={source.is_contiguous()}"
                    )

    def on_allocate(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.long)
        if slots.numel() == 0:
            return
        self.allocation_generations[slots] += 1
        self.sidecars[:, slots, 1] = 0

    def on_free(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.long)
        if slots.numel() > 0:
            self.sidecars[:, slots, 1] = 0

    def clear(self) -> None:
        self.sidecars.zero_()
        self.allocation_generations.zero_()
        self.expected_slots.zero_()
        self.expected_generations.zero_()
        if self.expected_tracking_slots is not None:
            self.expected_tracking_slots.zero_()
            self.expected_tracking_generations.zero_()

    def commit_mapping(
        self, request_indices: Sequence[int] | torch.Tensor, slots: torch.Tensor
    ) -> None:
        requests = torch.as_tensor(
            request_indices, dtype=torch.long, device=self.device
        )
        slots_i32 = slots.to(device=self.device, dtype=torch.int32)
        physical = self.req_to_token_pool.translate_mamba_indices(slots_i32)
        physical_long = physical.to(torch.long)
        self.expected_slots[requests] = physical.to(torch.int32)
        self.expected_generations[requests] = self.allocation_generations[
            physical_long
        ]

    def commit_tracking_mapping(
        self,
        request_indices: Sequence[int] | torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        if self.expected_tracking_slots is None:
            return
        requests = torch.as_tensor(
            request_indices, dtype=torch.long, device=self.device
        )
        slots_i32 = slots.to(device=self.device, dtype=torch.int32)
        physical = self.req_to_token_pool.translate_mamba_indices(slots_i32)
        valid = physical.ge(0) & physical.lt(self.num_slots)
        safe_physical = physical.masked_fill(~valid, 0).to(torch.long)
        self.expected_tracking_slots[requests] = physical.to(torch.int32)
        generations = self.allocation_generations[safe_physical]
        self.expected_tracking_generations[requests] = generations.masked_fill(
            ~valid, 0
        )

    def copy_slots(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        src = src.to(device=self.device, dtype=torch.long)
        dst = dst.to(device=self.device, dtype=torch.long)
        self.sidecars[:, dst] = self.sidecars[:, src]

    def invalidate_slots(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.long)
        self.sidecars[:, slots, 1] = 0

    def get_cpu_copy(self, slots: torch.Tensor) -> torch.Tensor:
        slots = slots.to(device=self.device, dtype=torch.long)
        return self.sidecars[:, slots].to("cpu", non_blocking=True)

    def load_cpu_copy(self, sidecars_cpu: torch.Tensor, slots: torch.Tensor) -> None:
        slots = slots.to(device=self.device, dtype=torch.long)
        expected = (self.num_layers, slots.numel(), 2)
        if tuple(sidecars_cpu.shape) != expected:
            raise RuntimeError(
                "protected recurrent sidecar shape mismatch: "
                f"expected {expected}, got {tuple(sidecars_cpu.shape)}"
            )
        self.sidecars[:, slots] = sidecars_cpu.to(
            device=self.device, dtype=self.sidecars.dtype, non_blocking=True
        )

    def layer_sources(self, global_layer_id: int) -> tuple[torch.Tensor, ...]:
        if global_layer_id not in self.req_to_token_pool.mamba_map:
            raise RuntimeError(
                f"state protection cannot map recurrent layer {global_layer_id}"
            )
        state = self.req_to_token_pool.mamba2_layer_cache(global_layer_id)
        conv_sources = (
            tuple(state.conv)
            if isinstance(state.conv, (tuple, list))
            else (state.conv,)
        )
        sources = conv_sources + (state.temporal,)
        if len(sources) > 4:
            raise RuntimeError(
                f"recurrent layer {global_layer_id} exposes {len(sources)} state "
                "tensors; the generalized accessor supports at most four"
            )
        return sources

    def context_for_layer(self, global_layer_id: int) -> StateSlotContext:
        local_layer = self.req_to_token_pool.mamba_map[global_layer_id]
        return StateSlotContext(
            sidecar=self.sidecars[local_layer],
            allocation_generations=self.allocation_generations,
            expected_slots=self.expected_slots,
            expected_generations=self.expected_generations,
            failure_status=self.failure_status,
            enabled=self.enabled,
            canary_violation_index=self.canary_violation_index,
            canary_forward_start=self.canary_forward_start,
            domain_seed=self.seed ^ int(local_layer),
        )

    def validate_layer(
        self,
        *,
        global_layer_id: int,
        forward_batch,
        cache_indices: torch.Tensor,
    ) -> RecurrentLayerGuard:
        requests = forward_batch.req_pool_indices[: cache_indices.shape[0]]
        prefix_lens = self._prefix_lens(forward_batch, cache_indices.shape[0])
        sources = self.layer_sources(global_layer_id)
        validate_state_slots(
            context=self.context_for_layer(global_layer_id),
            request_indices=requests,
            cache_indices=cache_indices,
            sources=sources,
            require_sealed=(prefix_lens > 0).to(torch.int32).contiguous(),
        )
        return RecurrentLayerGuard(
            request_indices=requests,
            cache_indices=cache_indices,
            sources=sources,
            local_layer=self.req_to_token_pool.mamba_map[global_layer_id],
        )

    def seal_layer(self, guard: RecurrentLayerGuard) -> None:
        context = StateSlotContext(
            sidecar=self.sidecars[guard.local_layer],
            allocation_generations=self.allocation_generations,
            expected_slots=self.expected_slots,
            expected_generations=self.expected_generations,
            failure_status=self.failure_status,
            enabled=self.enabled,
            canary_violation_index=self.canary_violation_index,
            canary_forward_start=self.canary_forward_start,
            domain_seed=self.seed ^ int(guard.local_layer),
        )
        seal_state_slots(
            context=context,
            request_indices=guard.request_indices,
            cache_indices=guard.cache_indices,
            sources=guard.sources,
        )

    def seal_tracking_layer(
        self,
        global_layer_id: int,
        forward_batch,
        track_indices: Optional[torch.Tensor],
    ) -> None:
        track_mask = getattr(forward_batch, "mamba_track_mask", None)
        if track_indices is None or track_mask is None:
            return
        batch_size = forward_batch.batch_size
        requests = forward_batch.req_pool_indices[:batch_size]
        # The attention backend owns virtual-to-physical translation and exposes
        # the exact graph-stable destination tensor consumed by its tracking
        # kernel. Re-translating here would corrupt unified-pool identities.
        track_indices = track_indices[:batch_size].contiguous()
        active_mask = track_mask[:batch_size].to(torch.int32).contiguous()
        seal_unbound_state_slots(
            context=self.context_for_layer(global_layer_id),
            request_indices=requests,
            cache_indices=track_indices,
            active_mask=active_mask,
            sources=self.layer_sources(global_layer_id),
        )

    def validate_tracking_layer(
        self,
        global_layer_id: int,
        forward_batch,
        track_indices: Optional[torch.Tensor],
    ) -> None:
        track_mask = getattr(forward_batch, "mamba_track_mask", None)
        if track_indices is None or track_mask is None:
            return
        if self.expected_tracking_slots is None:
            raise RuntimeError(
                "state protection received recurrent tracking indices without "
                "an ownership table"
            )
        batch_size = forward_batch.batch_size
        validate_unbound_state_slots(
            context=self.context_for_layer(global_layer_id),
            request_indices=forward_batch.req_pool_indices[:batch_size],
            cache_indices=track_indices[:batch_size].contiguous(),
            active_mask=track_mask[:batch_size].to(torch.int32).contiguous(),
            expected_slots=self.expected_tracking_slots,
            expected_generations=self.expected_tracking_generations,
        )

    @contextlib.contextmanager
    def guard_layer(
        self,
        *,
        global_layer_id: int,
        forward_batch,
        cache_indices: torch.Tensor,
        track_indices: Optional[torch.Tensor] = None,
    ) -> Iterator[None]:
        self.validate_tracking_layer(global_layer_id, forward_batch, track_indices)
        guard = self.validate_layer(
            global_layer_id=global_layer_id,
            forward_batch=forward_batch,
            cache_indices=cache_indices,
        )
        try:
            yield
        except BaseException:
            raise
        else:
            self.seal_layer(guard)
            self.seal_tracking_layer(
                global_layer_id, forward_batch, track_indices
            )

    def _prefix_lens(self, forward_batch, batch_size: int) -> torch.Tensor:
        prefix = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix is not None:
            return prefix[:batch_size]
        # Decode consumes all state committed before the current token.
        seq_lens = forward_batch.seq_lens[:batch_size]
        return torch.clamp(seq_lens - 1, min=0)


def maybe_recurrent_guard(
    backend,
    *,
    layer,
    forward_batch,
) -> contextlib.AbstractContextManager:
    manager = getattr(backend, "state_protection_manager", None)
    recurrent: Optional[RecurrentStateProtection] = (
        None if manager is None else manager.recurrent
    )
    if recurrent is None:
        return contextlib.nullcontext()
    metadata = getattr(backend, "forward_metadata", None)
    cache_indices = getattr(metadata, "mamba_cache_indices", None)
    if cache_indices is None:
        raise RuntimeError(
            "state protection reached a recurrent consumer without mamba_cache_indices"
        )
    return recurrent.guard_layer(
        global_layer_id=layer.layer_id,
        forward_batch=forward_batch,
        cache_indices=cache_indices,
        track_indices=getattr(metadata, "mamba_track_indices", None),
    )


def recurrent_guard_for_layer(
    backend,
    *,
    global_layer_id: int,
    forward_batch,
    cache_indices: Optional[torch.Tensor] = None,
    track_indices: Optional[torch.Tensor] = None,
) -> contextlib.AbstractContextManager:
    manager = getattr(backend, "state_protection_manager", None)
    recurrent: Optional[RecurrentStateProtection] = (
        None if manager is None else manager.recurrent
    )
    if recurrent is None:
        return contextlib.nullcontext()
    if cache_indices is None:
        metadata = getattr(backend, "forward_metadata", None)
        cache_indices = getattr(metadata, "mamba_cache_indices", None)
    if cache_indices is None:
        raise RuntimeError(
            "state protection reached a recurrent consumer without cache indices"
        )
    return recurrent.guard_layer(
        global_layer_id=global_layer_id,
        forward_batch=forward_batch,
        cache_indices=cache_indices,
        track_indices=track_indices,
    )
