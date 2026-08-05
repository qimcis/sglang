from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from sglang.kernels.jit.paged_state_protection import (
    PagedSlotContext,
    seal_paged_payload,
    validate_paged_mapping,
    validate_paged_payload,
    validate_paged_write_slots,
)


def _domain_seed(name: str) -> int:
    value = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


@dataclass(frozen=True, slots=True, kw_only=True)
class PagedLayerState:
    global_layer_id: int
    sources: tuple[torch.Tensor, ...]
    sidecar: torch.Tensor
    seed: int
    transfer_kind: str
    transfer_aux: tuple[torch.Tensor, ...]
    page_size: int
    swa_lut: Optional[torch.Tensor] = None
    swa_window_size: int = 0
    dsa_source: Optional[torch.Tensor] = None
    dsa_page_size: int = 0
    dsa_token_bytes: int = 0
    dsa_aux_bytes: int = 0


class PagedStateProtection:
    """Generic slot-major paged-state accessor.

    The logical token-chain canary owns request/position identity. This accessor
    adds full local-shard payload validation at the common attention consumer
    boundary and seals the slots written by that consumer. Its ABI is deliberately
    independent of MHA, MLA, and SWA equations.
    """

    def __init__(
        self,
        *,
        model_runner,
        failure_status: torch.Tensor,
        enabled: torch.Tensor,
        canary_violation_index: torch.Tensor,
        canary_forward_start: torch.Tensor,
    ) -> None:
        self.model_runner = model_runner
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.kv_pool = model_runner.token_to_kv_pool
        self.failure_status = failure_status
        self.enabled = enabled
        self.canary_violation_index = canary_violation_index
        self.canary_forward_start = canary_forward_start
        self.device = torch.device(model_runner.device)
        self.layers: dict[int, PagedLayerState] = {}
        self._source_pools: list[object] = []
        self._discover_layers()
        if self.layers:
            attached = getattr(self.kv_pool, "paged_state_protection", None)
            if attached not in (None, self):
                raise RuntimeError("paged state protection was installed twice")
            self.kv_pool.paged_state_protection = self
            for source_pool in self._source_pools:
                attached = getattr(source_pool, "paged_state_protection", None)
                if attached not in (None, self):
                    raise RuntimeError(
                        "paged state protection found a source pool owned by "
                        "another accessor"
                    )
                source_pool.paged_state_protection = self

    @staticmethod
    def _raw_layer_sources(
        source_pool, source_layer_id: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return persistent storage, never a dequantized getter result."""

        local_layer_id = source_layer_id - int(getattr(source_pool, "start_layer", 0))
        if local_layer_id < 0:
            raise RuntimeError(
                f"protected layer {source_layer_id} precedes the cache's start layer"
            )

        kv_buffer = getattr(source_pool, "kv_buffer", None)
        if kv_buffer is not None:
            try:
                sources = [kv_buffer[local_layer_id]]
            except IndexError as exc:
                raise RuntimeError(
                    f"protected MLA layer {source_layer_id} has no raw cache buffer"
                ) from exc
            kv_scale_buffer = getattr(source_pool, "kv_scale_buffer", None)
            if kv_scale_buffer is not None:
                sources.append(kv_scale_buffer[local_layer_id])
            return tuple(sources), tuple(sources[1:])

        k_buffer = getattr(source_pool, "k_buffer", None)
        v_buffer = getattr(source_pool, "v_buffer", None)
        if k_buffer is None or v_buffer is None:
            raise RuntimeError(
                f"state protection has no raw slot-major adapter for "
                f"{type(source_pool).__name__}"
            )
        try:
            sources = [k_buffer[local_layer_id], v_buffer[local_layer_id]]
        except IndexError as exc:
            raise RuntimeError(
                f"protected MHA layer {source_layer_id} has no raw cache buffer"
            ) from exc
        k_scale_buffer = getattr(source_pool, "k_scale_buffer", None)
        v_scale_buffer = getattr(source_pool, "v_scale_buffer", None)
        if (k_scale_buffer is None) != (v_scale_buffer is None):
            raise RuntimeError(
                f"protected MHA layer {source_layer_id} exposes only one FP4 scale buffer"
            )
        if k_scale_buffer is not None:
            sources.extend(
                (k_scale_buffer[local_layer_id], v_scale_buffer[local_layer_id])
            )
        return tuple(sources), tuple(sources[2:])

    def _discover_layers(self) -> None:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(self.kv_pool, DeepSeekV4TokenToKVPool):
            raise RuntimeError(
                "--enable-state-protection does not yet cover DeepSeek-V4's "
                "heterogeneous compressor/indexer pools"
            )

        layer_start = int(self.model_runner.layer_info.start_layer)
        layer_end = int(self.model_runner.layer_info.end_layer)
        for layer_id in range(layer_start, layer_end):
            transfer_kind = "contiguous"
            swa_lut = None
            swa_window_size = 0
            source_pool = self.kv_pool
            source_layer_id = layer_id

            if isinstance(self.kv_pool, HybridLinearKVPool):
                if layer_id not in self.kv_pool.full_attention_layer_id_mapping:
                    continue
                source_pool = self.kv_pool.full_kv_pool
                source_layer_id = self.kv_pool.full_attention_layer_id_mapping[
                    layer_id
                ]
            elif isinstance(self.kv_pool, SWAKVPool):
                mapping = self.kv_pool.layers_mapping.get(layer_id)
                if mapping is None:
                    continue
                _, is_swa = mapping
                if is_swa:
                    transfer_kind = "state"
                    source_pool = self.kv_pool.swa_kv_pool
                    swa_lut = self.kv_pool.full_to_swa_index_mapping
                    if swa_lut is None:
                        raise RuntimeError(
                            "state protection requires the SWA full-to-window "
                            "mapping to be registered before installation"
                        )
                    swa_window_size = int(
                        getattr(self.model_runner, "sliding_window_size", 0) or 0
                    )
                    if swa_window_size <= 0:
                        raise RuntimeError(
                            "state protection could not resolve the SWA window size"
                        )
                else:
                    source_pool = self.kv_pool.full_kv_pool
                source_layer_id = mapping[0]

            sources, transfer_aux = self._raw_layer_sources(
                source_pool, source_layer_id
            )
            if all(pool is not source_pool for pool in self._source_pools):
                self._source_pools.append(source_pool)
            key = sources[0]

            if getattr(source_pool, "use_hnd", False) or getattr(
                source_pool, "kv_cache_layout", None
            ) == "vectorized_5d":
                raise RuntimeError(
                    "--enable-state-protection currently requires contiguous "
                    "slot-major paged KV rows; HND/vectorized layouts need a "
                    "layout-specific accessor"
                )
            if not key.is_contiguous() or key.shape[0] <= 1:
                raise RuntimeError(
                    f"protected layer {layer_id} has unsupported key-cache layout "
                    f"shape={tuple(key.shape)}, contiguous={key.is_contiguous()}"
                )

            for source in sources[1:]:
                if not source.is_contiguous() or source.shape[0] != key.shape[0]:
                    raise RuntimeError(
                        f"protected layer {layer_id} has an unsupported auxiliary "
                        f"cache layout shape={tuple(source.shape)}"
                    )

            num_slots = int(key.shape[0])
            dsa_source = None
            dsa_page_size = dsa_token_bytes = dsa_aux_bytes = 0
            index_buffers = getattr(source_pool, "index_k_with_scale_buffer", None)
            if index_buffers is not None:
                local_layer_id = source_layer_id - int(
                    getattr(source_pool, "start_layer", 0)
                )
                dsa_source = index_buffers[local_layer_id]
                dsa_page_size = int(source_pool.page_size)
                dsa_token_bytes = int(source_pool.index_head_dim) * int(
                    dsa_source.element_size()
                )
                dsa_aux_bytes = (
                    int(source_pool.index_head_dim)
                    // int(source_pool.quant_block_size)
                    * 4
                )
                expected_pages = (num_slots + dsa_page_size - 1) // dsa_page_size
                if (
                    not dsa_source.is_contiguous()
                    or dsa_source.ndim != 2
                    or dsa_source.shape[0] < expected_pages
                ):
                    raise RuntimeError(
                        f"protected DSA layer {layer_id} has an unsupported "
                        "indexer-cache layout"
                    )
            sidecar = torch.zeros(
                (num_slots, 3), dtype=torch.int64, device=self.device
            )
            self.layers[layer_id] = PagedLayerState(
                global_layer_id=layer_id,
                sources=sources,
                sidecar=sidecar,
                seed=_domain_seed(f"paged-layer-{layer_id}"),
                transfer_kind=transfer_kind,
                transfer_aux=transfer_aux,
                page_size=int(getattr(source_pool, "page_size", 1)),
                swa_lut=swa_lut,
                swa_window_size=swa_window_size,
                dsa_source=dsa_source,
                dsa_page_size=dsa_page_size,
                dsa_token_bytes=dsa_token_bytes,
                dsa_aux_bytes=dsa_aux_bytes,
            )

    def get_transfer_buf_infos(
        self, transfer_kind: str
    ) -> tuple[list[int], list[int], list[int]]:
        states = [
            state
            for state in self.layers.values()
            if state.transfer_kind == transfer_kind
        ]
        ptrs: list[int] = []
        lens: list[int] = []
        item_lens: list[int] = []
        for state in states:
            ptrs.append(state.sidecar.data_ptr())
            lens.append(state.sidecar.nbytes)
            item_lens.append(state.sidecar[0].nbytes * state.page_size)
        return ptrs, lens, item_lens

    def get_aux_transfer_buf_infos(
        self, transfer_kind: str
    ) -> tuple[list[int], list[int], list[int]]:
        ptrs: list[int] = []
        lens: list[int] = []
        item_lens: list[int] = []
        for state in self.layers.values():
            if state.transfer_kind != transfer_kind:
                continue
            for source in state.transfer_aux:
                ptrs.append(source.data_ptr())
                lens.append(source.nbytes)
                item_lens.append(source[0].nbytes * state.page_size)
        return ptrs, lens, item_lens

    def get_cpu_copy(self, indices: torch.Tensor) -> dict[int, dict[str, object]]:
        """Snapshot sidecars in logical-token order for decode retraction."""

        indices = indices.to(device=self.device, dtype=torch.long)
        snapshot: dict[int, dict[str, object]] = {}
        for layer_id, state in self.layers.items():
            sidecar_indices = indices
            mapped = None
            if state.swa_lut is not None:
                sidecar_indices = state.swa_lut.index_select(0, indices)
                mapped = sidecar_indices.gt(0)
                rows = torch.zeros(
                    (indices.numel(), 3),
                    dtype=state.sidecar.dtype,
                    device=self.device,
                )
                rows[mapped] = state.sidecar[sidecar_indices[mapped]]
            else:
                rows = state.sidecar.index_select(0, sidecar_indices)
            snapshot[layer_id] = {
                "rows": rows.to("cpu"),
                "mapped": None if mapped is None else mapped.to("cpu"),
            }
        return snapshot

    def load_cpu_copy(
        self,
        snapshot: dict[int, dict[str, object]],
        indices: torch.Tensor,
    ) -> None:
        """Restore sidecars after payload restore, preserving SWA position identity."""

        if set(snapshot) != set(self.layers):
            raise RuntimeError(
                "protected paged sidecar layer mismatch during CPU restore"
            )
        indices = indices.to(device=self.device, dtype=torch.long)
        for layer_id, state in self.layers.items():
            layer_snapshot = snapshot[layer_id]
            rows_cpu = layer_snapshot.get("rows")
            if not isinstance(rows_cpu, torch.Tensor) or tuple(rows_cpu.shape) != (
                indices.numel(),
                3,
            ):
                raise RuntimeError(
                    f"protected paged sidecar shape mismatch for layer {layer_id}"
                )
            rows = rows_cpu.to(
                device=self.device, dtype=state.sidecar.dtype, non_blocking=True
            )
            if state.swa_lut is None:
                state.sidecar[indices] = rows
                continue

            mapped_cpu = layer_snapshot.get("mapped")
            if (
                not isinstance(mapped_cpu, torch.Tensor)
                or tuple(mapped_cpu.shape) != (indices.numel(),)
            ):
                raise RuntimeError(
                    f"protected SWA sidecar mapping mismatch for layer {layer_id}"
                )
            translated = state.swa_lut.index_select(0, indices)
            new_mapped = translated.gt(0)
            # Invalidate every newly allocated SWA target before restoring the
            # positional intersection. A changed window mapping must not inherit
            # a valid marker from a prior allocation.
            state.sidecar[translated[new_mapped], 1] = 0
            restore = mapped_cpu.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            ) & new_mapped
            state.sidecar[translated[restore]] = rows[restore]

    def context(self, layer: PagedLayerState) -> PagedSlotContext:
        return PagedSlotContext(
            sidecar=layer.sidecar,
            req_to_token=self.req_to_token_pool.req_to_token,
            failure_status=self.failure_status,
            enabled=self.enabled,
            canary_violation_index=self.canary_violation_index,
            canary_forward_start=self.canary_forward_start,
            sources=layer.sources,
            domain_seed=layer.seed,
            swa_lut=layer.swa_lut,
            swa_window_size=layer.swa_window_size,
            dsa_source=layer.dsa_source,
            dsa_page_size=layer.dsa_page_size,
            dsa_token_bytes=layer.dsa_token_bytes,
            dsa_aux_bytes=layer.dsa_aux_bytes,
        )

    @staticmethod
    def prefix_lens(forward_batch) -> torch.Tensor:
        prefix = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix is None:
            prefix = torch.clamp(forward_batch.seq_lens - 1, min=0)
        return prefix[: forward_batch.batch_size].to(torch.int32).contiguous()

    def preflight_mapping(self, forward_batch) -> None:
        request_indices = forward_batch.req_pool_indices[: forward_batch.batch_size]
        prefix_lens = self.prefix_lens(forward_batch)
        write_lens = self.write_lens(forward_batch, prefix_lens)
        write_slots = self.write_slots(forward_batch)
        seen: set[tuple[int, int]] = set()
        for layer in self.layers.values():
            identity = (
                0 if layer.swa_lut is None else layer.swa_lut.data_ptr(),
                int(layer.sidecar.shape[0]),
            )
            if identity in seen:
                continue
            seen.add(identity)
            validate_paged_mapping(
                context=self.context(layer),
                request_indices=request_indices,
                prefix_lens=prefix_lens,
            )
            validate_paged_write_slots(
                context=self.context(layer),
                request_indices=request_indices,
                prefix_lens=prefix_lens,
                write_slots=write_slots,
                write_lens=write_lens,
            )

    @staticmethod
    def write_lens(forward_batch, prefix_lens: torch.Tensor) -> torch.Tensor:
        write_lens = getattr(forward_batch, "extend_seq_lens", None)
        if write_lens is None:
            return torch.ones_like(prefix_lens, dtype=torch.int32)
        return write_lens[: forward_batch.batch_size].to(torch.int32).contiguous()

    @staticmethod
    def write_slots(forward_batch) -> torch.Tensor:
        write_slots = forward_batch.out_cache_loc
        real_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if real_tokens is not None:
            write_slots = write_slots[: int(real_tokens)]
        return write_slots.contiguous()

    @contextlib.contextmanager
    def guard_layer(
        self,
        *,
        global_layer_id: int,
        forward_batch,
        save_kv_cache: bool,
    ) -> Iterator[None]:
        layer = self.layers.get(global_layer_id)
        if layer is None:
            yield
            return
        request_indices = forward_batch.req_pool_indices[: forward_batch.batch_size]
        prefix_lens = self.prefix_lens(forward_batch)
        context = self.context(layer)
        validate_paged_payload(
            context=context,
            request_indices=request_indices,
            prefix_lens=prefix_lens,
        )
        try:
            yield
        except BaseException:
            raise
        else:
            if save_kv_cache:
                seal_paged_payload(
                    context=context,
                    request_indices=request_indices,
                    prefix_lens=prefix_lens,
                    write_slots=self.write_slots(forward_batch),
                    write_lens=self.write_lens(forward_batch, prefix_lens),
                )


def maybe_paged_guard(
    backend,
    *,
    layer,
    forward_batch,
    save_kv_cache: bool,
) -> contextlib.AbstractContextManager:
    manager = getattr(backend, "state_protection_manager", None)
    paged: Optional[PagedStateProtection] = None if manager is None else manager.paged
    if (
        paged is None
        or layer is None
        or forward_batch.forward_mode.is_idle()
    ):
        return contextlib.nullcontext()
    return paged.guard_layer(
        global_layer_id=layer.layer_id,
        forward_batch=forward_batch,
        save_kv_cache=save_kv_cache,
    )
