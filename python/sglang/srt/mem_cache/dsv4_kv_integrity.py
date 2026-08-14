"""Fail-closed integrity metadata for DeepSeek-V4 Flash KV state.

The protected layout contains seven independently addressed component types.
This module owns their stable identity, the strict Mooncake wire manifest and
the per-physical-page digest sidecars used by the decode consumers.  CUDA byte
hashing lives in :mod:`sgl_kernel.kvcacheio`; protocol parsing deliberately has
no CUDA dependency so malformed-message tests can run on CPU.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import struct
from typing import Iterable, Mapping, Optional, Sequence

import torch

DSV4_INTEGRITY_MAGIC = b"DSV4KVI\0"
DSV4_INTEGRITY_PROTOCOL_VERSION = 1
DSV4_INTEGRITY_DIGEST_BYTES = 8
DSV4_INTEGRITY_INVALID_DIGEST = -1

_HEADER = struct.Struct("<8sHHQQ32s")
_ENTRY = struct.Struct("<B3xihHIIQQI4x")
_U64 = struct.Struct("<Q")
_MAX_ENTRIES = 4096
_MAX_DIGESTS = 1 << 20


class DSV4IntegrityError(RuntimeError):
    pass


class DSV4ManifestError(DSV4IntegrityError):
    pass


class DSV4Component(enum.IntEnum):
    SWA_KV = 1
    C4_ATTENTION_KV = 2
    C128_ATTENTION_KV = 3
    C4_INDEXER_KV = 4
    C4_ATTENTION_STATE = 5
    C4_INDEXER_STATE = 6
    C128_ATTENTION_STATE = 7


class DSV4TransferGroup(enum.IntEnum):
    KV = 1
    SWA = 2
    C128_STATE = 3


class DSV4IntegrityDomain(enum.IntEnum):
    """Address spaces shared by DSV4 component buffers."""

    FULL = 1
    SWA = 2
    C128_STATE = 3


@dataclasses.dataclass(frozen=True)
class DSV4ComponentDescriptor:
    component: DSV4Component
    layer_id: int
    compress_ratio: int
    transfer_group: DSV4TransferGroup
    page_size: int
    item_nbytes: int
    capacity: int
    buffer: torch.Tensor = dataclasses.field(compare=False, repr=False)

    @property
    def identity(self) -> tuple[int, int]:
        return int(self.component), int(self.layer_id)

    def validate(self) -> None:
        if self.layer_id < 0:
            raise ValueError("DSV4 component layer_id must be non-negative")
        if self.compress_ratio not in (0, 4, 128):
            raise ValueError("invalid DSV4 component compression ratio")
        if self.page_size <= 0 or self.item_nbytes <= 0 or self.capacity <= 0:
            raise ValueError("invalid DSV4 component geometry")
        if not isinstance(self.buffer, torch.Tensor) or self.buffer.ndim != 2:
            raise ValueError("DSV4 integrity buffers must be two-dimensional")
        if self.buffer.shape[0] != self.capacity:
            raise ValueError("DSV4 component capacity drift")
        if self.buffer[0].nbytes != self.item_nbytes:
            raise ValueError("DSV4 component item-size drift")


def layout_fingerprint(
    descriptors: Sequence[DSV4ComponentDescriptor],
) -> bytes:
    h = hashlib.sha256()
    for d in descriptors:
        d.validate()
        h.update(
            struct.pack(
                "<B3xihHIIQ",
                int(d.component),
                d.layer_id,
                d.compress_ratio,
                int(d.transfer_group),
                d.page_size,
                d.item_nbytes,
                d.capacity,
            )
        )
    return h.digest()


@dataclasses.dataclass(frozen=True)
class DSV4ManifestEntry:
    component: DSV4Component
    layer_id: int
    compress_ratio: int
    transfer_group: DSV4TransferGroup
    page_size: int
    item_nbytes: int
    logical_start: int
    logical_count: int
    digests: tuple[int, ...]

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (
            int(self.component),
            self.layer_id,
            self.logical_start,
            self.logical_count,
        )

    def validate(self) -> None:
        if self.layer_id < 0 or self.logical_start < 0 or self.logical_count < 0:
            raise DSV4ManifestError("negative DSV4 manifest field")
        if self.layer_id > 0x7FFFFFFF:
            raise DSV4ManifestError("manifest layer id is out of range")
        if self.compress_ratio not in (0, 4, 128):
            raise DSV4ManifestError("invalid manifest compression ratio")
        if self.page_size <= 0 or self.item_nbytes <= 0:
            raise DSV4ManifestError("invalid manifest component geometry")
        if self.page_size > 0xFFFFFFFF or self.item_nbytes > 0xFFFFFFFF:
            raise DSV4ManifestError("manifest component geometry is out of range")
        if self.logical_start > 0xFFFFFFFFFFFFFFFF:
            raise DSV4ManifestError("manifest logical start is out of range")
        if len(self.digests) != self.logical_count:
            raise DSV4ManifestError("manifest digest count mismatch")
        if self.logical_count > _MAX_DIGESTS:
            raise DSV4ManifestError("manifest entry is too large")
        for digest in self.digests:
            if not 0 <= int(digest) <= 0xFFFFFFFFFFFFFFFF:
                raise DSV4ManifestError("manifest digest is not uint64")


@dataclasses.dataclass(frozen=True)
class DSV4TransferManifest:
    bootstrap_room: int
    transfer_nonce: int
    layout_digest: bytes
    entries: tuple[DSV4ManifestEntry, ...]
    version: int = DSV4_INTEGRITY_PROTOCOL_VERSION

    def validate(self) -> None:
        if self.version != DSV4_INTEGRITY_PROTOCOL_VERSION:
            raise DSV4ManifestError(
                f"unsupported DSV4 integrity protocol version {self.version}"
            )
        if (
            self.bootstrap_room < 0
            or self.bootstrap_room > 0xFFFFFFFFFFFFFFFF
            or not self.transfer_nonce
            or self.transfer_nonce > 0xFFFFFFFFFFFFFFFF
        ):
            raise DSV4ManifestError("invalid manifest room or nonce")
        if len(self.layout_digest) != hashlib.sha256().digest_size:
            raise DSV4ManifestError("invalid layout fingerprint")
        if len(self.entries) > _MAX_ENTRIES:
            raise DSV4ManifestError("invalid manifest entry count")
        seen = set()
        total_digests = 0
        for entry in self.entries:
            entry.validate()
            total_digests += len(entry.digests)
            if total_digests > _MAX_DIGESTS:
                raise DSV4ManifestError("manifest contains too many digests")
            component_identity = (int(entry.component), entry.layer_id)
            if component_identity in seen:
                raise DSV4ManifestError("duplicate DSV4 manifest entry")
            seen.add(component_identity)

    def to_bytes(self) -> bytes:
        self.validate()
        chunks = [
            _HEADER.pack(
                DSV4_INTEGRITY_MAGIC,
                self.version,
                len(self.entries),
                self.bootstrap_room,
                self.transfer_nonce,
                self.layout_digest,
            )
        ]
        for entry in self.entries:
            chunks.append(
                _ENTRY.pack(
                    int(entry.component),
                    entry.layer_id,
                    entry.compress_ratio,
                    int(entry.transfer_group),
                    entry.page_size,
                    entry.item_nbytes,
                    entry.logical_start,
                    entry.logical_count,
                    len(entry.digests),
                )
            )
            chunks.extend(_U64.pack(int(value)) for value in entry.digests)
        return b"".join(chunks)

    @classmethod
    def from_bytes(cls, payload: bytes) -> DSV4TransferManifest:
        if len(payload) < _HEADER.size:
            raise DSV4ManifestError("truncated DSV4 manifest header")
        magic, version, count, room, nonce, fingerprint = _HEADER.unpack_from(payload)
        if magic != DSV4_INTEGRITY_MAGIC:
            raise DSV4ManifestError("invalid DSV4 manifest magic")
        if count > _MAX_ENTRIES:
            raise DSV4ManifestError("invalid DSV4 manifest entry count")
        offset = _HEADER.size
        entries = []
        total_digests = 0
        for _ in range(count):
            if offset + _ENTRY.size > len(payload):
                raise DSV4ManifestError("truncated DSV4 manifest entry")
            (
                component,
                layer_id,
                ratio,
                group,
                page_size,
                item_nbytes,
                logical_start,
                logical_count,
                digest_count,
            ) = _ENTRY.unpack_from(payload, offset)
            offset += _ENTRY.size
            if digest_count != logical_count or digest_count > _MAX_DIGESTS:
                raise DSV4ManifestError("invalid DSV4 digest count")
            total_digests += digest_count
            if total_digests > _MAX_DIGESTS:
                raise DSV4ManifestError("manifest contains too many digests")
            digest_bytes = digest_count * _U64.size
            if offset + digest_bytes > len(payload):
                raise DSV4ManifestError("truncated DSV4 digest vector")
            digests = tuple(
                _U64.unpack_from(payload, offset + i * _U64.size)[0]
                for i in range(digest_count)
            )
            offset += digest_bytes
            try:
                component_enum = DSV4Component(component)
                group_enum = DSV4TransferGroup(group)
            except ValueError as exc:
                raise DSV4ManifestError("unknown DSV4 component identity") from exc
            entries.append(
                DSV4ManifestEntry(
                    component_enum,
                    layer_id,
                    ratio,
                    group_enum,
                    page_size,
                    item_nbytes,
                    logical_start,
                    logical_count,
                    digests,
                )
            )
        if offset != len(payload):
            raise DSV4ManifestError("trailing bytes in DSV4 manifest")
        result = cls(room, nonce, fingerprint, tuple(entries), version)
        result.validate()
        return result


def _unsigned_digest_values(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) & 0xFFFFFFFFFFFFFFFF for value in values.tolist())


def compute_page_digests(
    buffer: torch.Tensor, page_indices: torch.Tensor, *, seed: int
) -> torch.Tensor:
    if buffer.device.type != "cuda":
        raise DSV4IntegrityError("DSV4 page digests require CUDA buffers")
    try:
        from sgl_kernel.kvcacheio import dsv4_page_digests
    except (ImportError, AttributeError) as exc:
        raise DSV4IntegrityError(
            "the sglang-kernel DSV4 integrity op is not installed"
        ) from exc
    return dsv4_page_digests(buffer, page_indices, seed)


class DSV4IntegritySidecar:
    """Expected byte digests for one component buffer."""

    def __init__(self, descriptor: DSV4ComponentDescriptor):
        descriptor.validate()
        self.descriptor = descriptor
        device = descriptor.buffer.device
        self.digest = torch.zeros(descriptor.capacity, dtype=torch.int64, device=device)
        self.valid = torch.zeros(descriptor.capacity, dtype=torch.uint8, device=device)
        self.dirty = torch.zeros(descriptor.capacity, dtype=torch.int32, device=device)
        # Per-call graph-stable scratch.  The validator hashes each referenced
        # physical item once even when many token slots point into that item.
        self.validation_state = torch.zeros(
            descriptor.capacity, dtype=torch.int32, device=device
        )
        self.validation_failure = torch.zeros(
            descriptor.capacity, dtype=torch.int32, device=device
        )
        self.validation_pages = torch.empty(
            descriptor.capacity, dtype=torch.int32, device=device
        )
        self.validation_count = torch.zeros(1, dtype=torch.int32, device=device)

    def install(self, page_indices: torch.Tensor, digests: torch.Tensor) -> None:
        pages = page_indices.to(device=self.digest.device, dtype=torch.long)
        if pages.numel() != digests.numel():
            raise DSV4IntegrityError("sidecar page/digest length mismatch")
        if pages.numel() == 0:
            return
        if bool(((pages < 0) | (pages >= self.descriptor.capacity)).any().item()):
            raise DSV4IntegrityError("sidecar page index is out of bounds")
        self.digest.index_copy_(0, pages, digests.to(torch.int64))
        self.valid.index_fill_(0, pages, 1)

    def refresh(self, page_indices: torch.Tensor) -> torch.Tensor:
        pages = torch.unique(
            page_indices.to(device=self.digest.device, dtype=torch.int32)
        )
        pages = pages[(pages >= 0) & (pages < self.descriptor.capacity)]
        values = compute_page_digests(
            self.descriptor.buffer,
            pages,
            seed=component_seed(self.descriptor),
        )
        self.install(pages, values)
        return values

    def invalidate(self, page_indices: torch.Tensor) -> None:
        pages = page_indices.to(device=self.digest.device, dtype=torch.long)
        pages = pages[(pages >= 0) & (pages < self.descriptor.capacity)]
        if pages.numel():
            self.valid.index_fill_(0, pages, 0)


class DSV4IntegrityAddressSpace:
    """Allocation generations and per-request expected page mappings."""

    def __init__(
        self,
        *,
        domain: DSV4IntegrityDomain,
        physical_capacity: int,
        request_capacity: int,
        logical_capacity: int,
        device: torch.device,
    ) -> None:
        self.domain = domain
        self.physical_capacity = physical_capacity
        self.logical_capacity = logical_capacity
        self.generation = torch.zeros(
            physical_capacity, dtype=torch.int64, device=device
        )
        self.expected_page = torch.zeros(
            (request_capacity, logical_capacity), dtype=torch.int32, device=device
        )
        self.expected_generation = torch.zeros(
            (request_capacity, logical_capacity), dtype=torch.int64, device=device
        )
        self.expected_valid = torch.zeros(
            (request_capacity, logical_capacity), dtype=torch.int32, device=device
        )

    def clear_requests(self, request_indices: torch.Tensor) -> None:
        if request_indices.numel():
            self.expected_valid.index_fill_(0, request_indices.to(torch.long), 0)


def component_seed(descriptor: DSV4ComponentDescriptor) -> int:
    value = (
        0x44535634
        ^ (int(descriptor.component) * 0x9E3779B1)
        ^ (descriptor.layer_id * 0x85EBCA77)
        ^ (descriptor.compress_ratio * 0xC2B2AE3D)
    )
    return value & 0x7FFFFFFFFFFFFFFF


def causal_swa_logical_pages(
    seq_lens: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    width: int,
    page_size: int,
) -> torch.Tensor:
    """Map newest-to-oldest causal SWA columns to logical pages."""
    if page_size <= 0 or width < 0:
        raise ValueError("invalid SWA logical-page geometry")
    if seq_lens.ndim != 1 or valid_lengths.shape != seq_lens.shape:
        raise ValueError("invalid SWA logical-page batch geometry")
    columns = torch.arange(width, dtype=torch.int64, device=seq_lens.device).view(1, -1)
    positions = seq_lens.to(torch.int64).view(-1, 1) - 1 - columns
    live = (columns < valid_lengths.to(torch.int64).view(-1, 1)) & (positions >= 0)
    return torch.where(
        live,
        torch.div(positions, page_size, rounding_mode="floor"),
        -1,
    ).contiguous()


class DSV4KVIntegrityManager:
    def __init__(
        self,
        descriptors: Sequence[DSV4ComponentDescriptor],
        *,
        request_capacity: int,
        max_context_len: int,
        full_page_size: int,
        swa_page_size: int,
    ):
        descriptors = tuple(descriptors)
        if not descriptors:
            raise ValueError("DSV4 integrity requires component descriptors")
        if max_context_len is None or max_context_len <= 0:
            raise ValueError("DSV4 integrity requires a resolved context length")
        if full_page_size <= 0 or swa_page_size <= 0:
            raise ValueError("DSV4 integrity page sizes must be positive")
        identities = [descriptor.identity for descriptor in descriptors]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate DSV4 component descriptor")
        self.descriptors = descriptors
        self.layout_digest = layout_fingerprint(descriptors)
        self.by_identity = {
            descriptor.identity: descriptor for descriptor in descriptors
        }
        self.sidecars = {
            descriptor.identity: DSV4IntegritySidecar(descriptor)
            for descriptor in descriptors
        }
        device = descriptors[0].buffer.device
        self.request_capacity = request_capacity
        self.max_context_len = max_context_len
        self.full_page_size = full_page_size
        self.swa_page_size = swa_page_size
        self.failure_status = torch.zeros(
            request_capacity, dtype=torch.int32, device=device
        )
        self.request_epochs = torch.zeros(
            request_capacity, dtype=torch.int64, device=device
        )
        full_logical_capacity = (max_context_len + full_page_size - 1) // full_page_size
        swa_logical_capacity = (max_context_len + swa_page_size - 1) // swa_page_size

        def group_capacity(group: DSV4TransferGroup) -> int:
            capacities = {
                descriptor.capacity
                for descriptor in descriptors
                if descriptor.transfer_group == group
            }
            if not capacities:
                raise ValueError(f"DSV4 {group.name} has no component descriptors")
            # State buffers can contain more transfer-sized rows than their KV
            # allocator exposes (for example C4 state versus SWA KV).  They still
            # share page identities over the allocator's live prefix, so retain
            # one generation domain sized for the largest component.  Unallocated
            # tail entries remain generation zero and therefore fail closed.
            return max(capacities)

        self.address_spaces = {
            DSV4IntegrityDomain.FULL: DSV4IntegrityAddressSpace(
                domain=DSV4IntegrityDomain.FULL,
                physical_capacity=group_capacity(DSV4TransferGroup.KV),
                request_capacity=request_capacity,
                logical_capacity=full_logical_capacity,
                device=device,
            ),
            DSV4IntegrityDomain.SWA: DSV4IntegrityAddressSpace(
                domain=DSV4IntegrityDomain.SWA,
                physical_capacity=group_capacity(DSV4TransferGroup.SWA),
                request_capacity=request_capacity,
                logical_capacity=swa_logical_capacity,
                device=device,
            ),
            DSV4IntegrityDomain.C128_STATE: DSV4IntegrityAddressSpace(
                domain=DSV4IntegrityDomain.C128_STATE,
                physical_capacity=group_capacity(DSV4TransferGroup.C128_STATE),
                request_capacity=request_capacity,
                logical_capacity=1,
                device=device,
            ),
        }
        self._req_pool = None
        self._full_to_swa_mapping = None

    def descriptors_for_group(
        self, group: DSV4TransferGroup
    ) -> tuple[DSV4ComponentDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.transfer_group == group)

    def attach_request_pool(self, req_pool) -> None:
        if self._req_pool is req_pool:
            return
        if self._req_pool is not None:
            raise DSV4IntegrityError("DSV4 integrity request pool changed")
        self._req_pool = req_pool
        req_pool.register_allocation_callback(self.register_requests)
        req_pool.register_write_callback(self.bind_request_table_write)

    def attach_full_to_swa_mapping(self, mapping: torch.Tensor) -> None:
        if mapping.ndim != 1 or mapping.device != self.failure_status.device:
            raise DSV4IntegrityError("invalid DSV4 full-to-SWA mapping")
        if (
            self._full_to_swa_mapping is not None
            and self._full_to_swa_mapping is not mapping
        ):
            raise DSV4IntegrityError("DSV4 full-to-SWA mapping changed")
        self._full_to_swa_mapping = mapping

    def register_requests(
        self, request_indices: Sequence[int], generations: Sequence[int]
    ) -> None:
        if not request_indices:
            return
        device = self.failure_status.device
        reqs = torch.as_tensor(request_indices, dtype=torch.long, device=device)
        epochs = torch.as_tensor(generations, dtype=torch.int64, device=device)
        self.request_epochs.index_copy_(0, reqs, epochs)
        self.failure_status.index_fill_(0, reqs, 0)
        for space in self.address_spaces.values():
            space.clear_requests(reqs)
        self.bump_allocations(DSV4IntegrityDomain.C128_STATE, reqs)
        state_slots = reqs.to(torch.int32).reshape(-1, 1)
        self.bind_pages(
            DSV4IntegrityDomain.C128_STATE,
            state_slots,
            torch.zeros_like(state_slots, dtype=torch.int64),
            reqs,
            slot_page_size=1,
        )

    @staticmethod
    def domain_for_group(group: DSV4TransferGroup) -> DSV4IntegrityDomain:
        return {
            DSV4TransferGroup.KV: DSV4IntegrityDomain.FULL,
            DSV4TransferGroup.SWA: DSV4IntegrityDomain.SWA,
            DSV4TransferGroup.C128_STATE: DSV4IntegrityDomain.C128_STATE,
        }[group]

    def bump_allocations(
        self, domain: DSV4IntegrityDomain, page_indices: torch.Tensor
    ) -> None:
        space = self.address_spaces[domain]
        pages = torch.unique(
            page_indices.to(device=space.generation.device, dtype=torch.long)
        )
        pages = pages[(pages > 0) & (pages < space.physical_capacity)]
        if pages.numel() == 0:
            return
        space.generation.index_add_(0, pages, torch.ones_like(pages))
        group = {
            DSV4IntegrityDomain.FULL: DSV4TransferGroup.KV,
            DSV4IntegrityDomain.SWA: DSV4TransferGroup.SWA,
            DSV4IntegrityDomain.C128_STATE: DSV4TransferGroup.C128_STATE,
        }[domain]
        for descriptor in self.descriptors_for_group(group):
            self.sidecars[descriptor.identity].invalidate(pages)

    def _ops(self):
        try:
            from sgl_kernel.kvcacheio import (
                dsv4_bind_pages,
                dsv4_refresh_slots,
                dsv4_validate_pages,
            )
        except (ImportError, AttributeError) as exc:
            raise DSV4IntegrityError(
                "the sglang-kernel DSV4 integrity ops are not installed"
            ) from exc
        return (
            dsv4_bind_pages,
            dsv4_refresh_slots,
            dsv4_validate_pages,
        )

    def _check_mapping_inputs(
        self,
        domain: DSV4IntegrityDomain,
        slots: torch.Tensor,
        logical_pages: torch.Tensor,
        request_indices: torch.Tensor,
    ) -> None:
        space = self.address_spaces[domain]
        if slots.shape != logical_pages.shape or slots.ndim not in (1, 2):
            raise DSV4IntegrityError("DSV4 mapping slot geometry mismatch")
        if (
            request_indices.ndim != 1
            or request_indices.numel() == 0
            or slots.numel() % request_indices.numel() != 0
        ):
            raise DSV4IntegrityError("DSV4 mapping request geometry mismatch")
        if space.logical_capacity <= 0:
            raise DSV4IntegrityError("DSV4 mapping address space is empty")

    def bind_pages(
        self,
        domain: DSV4IntegrityDomain,
        slots: torch.Tensor,
        logical_pages: torch.Tensor,
        request_indices: torch.Tensor,
        *,
        slot_page_size: int,
        invalid_value: int = 0,
    ) -> torch.Tensor:
        """Install an expected mapping once, then require exact reuse."""
        bind, _, _ = self._ops()
        space = self.address_spaces[domain]
        self._check_mapping_inputs(domain, slots, logical_pages, request_indices)
        out = torch.empty_like(slots)
        bind(
            slots,
            logical_pages.to(torch.int64),
            request_indices.to(torch.int64),
            space.generation,
            space.expected_page,
            space.expected_generation,
            space.expected_valid,
            out,
            self.failure_status,
            slot_page_size,
            invalid_value,
            True,
        )
        return out

    def verify_mapping(
        self,
        domain: DSV4IntegrityDomain,
        slots: torch.Tensor,
        logical_pages: torch.Tensor,
        request_indices: torch.Tensor,
        *,
        slot_page_size: int,
        invalid_value: int = 0,
    ) -> torch.Tensor:
        """Verify a mapping without learning missing request ownership."""
        bind, _, _ = self._ops()
        space = self.address_spaces[domain]
        self._check_mapping_inputs(domain, slots, logical_pages, request_indices)
        out = torch.empty_like(slots)
        bind(
            slots,
            logical_pages.to(torch.int64),
            request_indices.to(torch.int64),
            space.generation,
            space.expected_page,
            space.expected_generation,
            space.expected_valid,
            out,
            self.failure_status,
            slot_page_size,
            invalid_value,
            False,
        )
        return out

    def replace_pages(
        self,
        domain: DSV4IntegrityDomain,
        slots: torch.Tensor,
        logical_pages: torch.Tensor,
        request_indices: torch.Tensor,
        *,
        slot_page_size: int,
        invalid_value: int = 0,
    ) -> torch.Tensor:
        """Replace mappings from a trusted allocator or cache-table write."""
        self._check_mapping_inputs(domain, slots, logical_pages, request_indices)
        space = self.address_spaces[domain]
        live = logical_pages >= 0
        req_matrix = request_indices.view(-1, 1).expand_as(logical_pages)
        space.expected_valid[
            req_matrix[live].to(torch.long), logical_pages[live].to(torch.long)
        ] = 0
        return self.bind_pages(
            domain,
            slots,
            logical_pages,
            request_indices,
            slot_page_size=slot_page_size,
            invalid_value=invalid_value,
        )

    def bind_request_table_write(self, indices, values) -> None:
        """Bind FULL and SWA ownership from a trusted req-table write."""
        if self._full_to_swa_mapping is None:
            raise DSV4IntegrityError("DSV4 full-to-SWA mapping is not attached")
        if not isinstance(indices, tuple) or len(indices) != 2:
            raise DSV4IntegrityError("unsupported DSV4 request-table write indexing")
        row_index, position_index = indices
        device = self.failure_status.device
        slots = torch.as_tensor(values, dtype=torch.int32, device=device)
        if slots.numel() == 0:
            return

        if isinstance(position_index, slice):
            if not isinstance(row_index, int):
                raise DSV4IntegrityError("unsupported sliced DSV4 request-table write")
            start, stop, step = position_index.indices(self.max_context_len)
            if step != 1:
                raise DSV4IntegrityError(
                    "strided DSV4 request-table writes are unsupported"
                )
            positions = torch.arange(start, stop, dtype=torch.int64, device=device)
            if positions.numel() != slots.numel():
                raise DSV4IntegrityError("DSV4 request-table write length mismatch")
            slots = slots.reshape(1, -1).contiguous()
            positions = positions.reshape(1, -1)
            requests = torch.tensor([row_index], dtype=torch.int64, device=device)
        else:
            requests_raw = torch.as_tensor(row_index, dtype=torch.int64, device=device)
            positions_raw = torch.as_tensor(
                position_index, dtype=torch.int64, device=device
            )
            try:
                requests_raw, positions_raw, slots = torch.broadcast_tensors(
                    requests_raw, positions_raw, slots
                )
            except RuntimeError as exc:
                raise DSV4IntegrityError(
                    "DSV4 request-table advanced indexing mismatch"
                ) from exc
            if requests_raw.numel() == 1:
                requests = requests_raw.reshape(1)
                positions = positions_raw.reshape(1, -1).contiguous()
                slots = slots.reshape(1, -1).contiguous()
            else:
                requests = requests_raw.reshape(-1).contiguous()
                positions = positions_raw.reshape(-1, 1).contiguous()
                slots = slots.reshape(-1, 1).contiguous()

        full_logical = torch.div(
            positions, self.full_page_size, rounding_mode="floor"
        ).contiguous()
        self.replace_pages(
            DSV4IntegrityDomain.FULL,
            slots,
            full_logical,
            requests,
            slot_page_size=self.full_page_size,
        )

        lut_indices = slots.to(torch.long)
        lut_valid = (lut_indices >= 0) & (
            lut_indices < self._full_to_swa_mapping.numel()
        )
        safe_indices = lut_indices.clamp(
            min=0, max=self._full_to_swa_mapping.numel() - 1
        )
        swa_slots = torch.where(
            lut_valid,
            self._full_to_swa_mapping[safe_indices],
            -1,
        ).to(torch.int32)
        swa_logical = torch.where(
            swa_slots > 0,
            torch.div(positions, self.swa_page_size, rounding_mode="floor"),
            -1,
        ).contiguous()
        self.replace_pages(
            DSV4IntegrityDomain.SWA,
            swa_slots.contiguous(),
            swa_logical,
            requests,
            slot_page_size=self.swa_page_size,
        )

    def validate_pages(
        self,
        descriptor: DSV4ComponentDescriptor,
        slots: torch.Tensor,
        logical_pages: torch.Tensor,
        request_indices: torch.Tensor,
        *,
        slot_page_size: int,
        invalid_value: int = 0,
        allow_missing_digest: bool = False,
    ) -> torch.Tensor:
        """Fused mapping, generation and byte validation with sanitization."""
        _, _, validate = self._ops()
        space = self.address_spaces[self.domain_for_group(descriptor.transfer_group)]
        sidecar = self.sidecars[descriptor.identity]
        out = torch.empty_like(slots)
        validate(
            descriptor.buffer,
            slots,
            logical_pages.to(torch.int64),
            request_indices.to(torch.int64),
            sidecar.digest,
            sidecar.valid,
            sidecar.validation_state,
            sidecar.validation_failure,
            sidecar.validation_pages,
            sidecar.validation_count,
            space.generation,
            space.expected_page,
            space.expected_generation,
            space.expected_valid,
            out,
            self.failure_status,
            component_seed(descriptor),
            slot_page_size,
            invalid_value,
            allow_missing_digest,
        )
        return out

    def refresh_written_slots(
        self,
        descriptor: DSV4ComponentDescriptor,
        slots: torch.Tensor,
        *,
        slot_page_size: int,
    ) -> None:
        _, refresh_slots, _ = self._ops()
        sidecar = self.sidecars[descriptor.identity]
        refresh_slots(
            descriptor.buffer,
            sidecar.digest,
            sidecar.valid,
            sidecar.dirty,
            slots,
            component_seed(descriptor),
            slot_page_size,
        )

    def protect_compressor_plan(
        self,
        descriptor: DSV4ComponentDescriptor,
        plan,
        request_indices_repeated: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Validate state reads and sanitize state writes in one plan copy."""
        domain = self.domain_for_group(descriptor.transfer_group)
        if plan.is_decode:
            plan_d = plan[1].clone()
            raw = plan_d.view(torch.int32).reshape(-1, 4)
            count = raw.shape[0]
            reqs = request_indices_repeated[:count].to(torch.int64).contiguous()
            seq_lens = raw[:, 0].to(torch.int64)
            consumes_state = (seq_lens > 0) & (
                seq_lens.remainder(descriptor.compress_ratio) == 0
            )
            logical_1 = torch.where(
                consumes_state,
                torch.div(
                    seq_lens - 1,
                    self.swa_page_size,
                    rounding_mode="floor",
                ),
                -1,
            )
            logical_0 = torch.where(
                consumes_state
                & (descriptor.compress_ratio == 4)
                & (seq_lens > descriptor.compress_ratio),
                torch.div(
                    seq_lens - 1 - descriptor.compress_ratio,
                    self.swa_page_size,
                    rounding_mode="floor",
                ),
                -1,
            )
            if descriptor.transfer_group == DSV4TransferGroup.C128_STATE:
                logical_1 = torch.where(consumes_state, 0, -1)
            raw[:, 2].copy_(
                self.validate_pages(
                    descriptor,
                    raw[:, 2].contiguous(),
                    logical_0.contiguous(),
                    reqs,
                    slot_page_size=max(
                        descriptor.page_size // descriptor.compress_ratio, 1
                    ),
                )
            )
            raw[:, 3].copy_(
                self.validate_pages(
                    descriptor,
                    raw[:, 3].contiguous(),
                    logical_1.contiguous(),
                    reqs,
                    slot_page_size=max(
                        descriptor.page_size // descriptor.compress_ratio, 1
                    ),
                )
            )
            raw[:, 1].copy_(
                self.verify_mapping(
                    domain,
                    raw[:, 1].contiguous(),
                    logical_1.contiguous(),
                    reqs,
                    slot_page_size=descriptor.page_size,
                    invalid_value=(descriptor.capacity - 1) * descriptor.page_size,
                )
            )
            return plan._replace(plan_d=plan_d)

        plan_c = plan[1].clone()
        raw_c = plan_c.view(torch.int32).reshape(-1, 4)
        seq_lens = raw_c[:, 0].to(torch.int64)
        ragged = torch.bitwise_and(raw_c[:, 1].to(torch.int64), 0xFFFF)
        buffer_lens = torch.bitwise_right_shift(raw_c[:, 1].to(torch.int64), 16)
        valid_c = seq_lens >= 0
        safe_ragged = torch.where(valid_c, ragged, 0).clamp(
            max=max(request_indices_repeated.numel() - 1, 0)
        )
        reqs_c = request_indices_repeated.to(torch.int64)[safe_ragged].contiguous()
        reads_page_1 = valid_c & (
            (buffer_lens > 4) if descriptor.compress_ratio == 4 else (buffer_lens > 0)
        )
        reads_page_0 = (
            valid_c
            & (descriptor.compress_ratio == 4)
            & (seq_lens > descriptor.compress_ratio)
            & (buffer_lens > 0)
        )
        logical_1 = torch.where(
            reads_page_1,
            torch.div(
                torch.clamp(seq_lens - 1, min=0),
                self.swa_page_size,
                rounding_mode="floor",
            ),
            -1,
        )
        logical_0 = torch.where(
            reads_page_0,
            torch.div(
                torch.clamp(seq_lens - 1 - descriptor.compress_ratio, min=0),
                self.swa_page_size,
                rounding_mode="floor",
            ),
            -1,
        )
        if descriptor.transfer_group == DSV4TransferGroup.C128_STATE:
            logical_0.fill_(-1)
            logical_1 = torch.where(reads_page_1, 0, -1)
        raw_c[:, 2].copy_(
            self.validate_pages(
                descriptor,
                raw_c[:, 2].contiguous(),
                logical_0.contiguous(),
                reqs_c,
                slot_page_size=max(
                    descriptor.page_size // descriptor.compress_ratio, 1
                ),
            )
        )
        raw_c[:, 3].copy_(
            self.validate_pages(
                descriptor,
                raw_c[:, 3].contiguous(),
                logical_1.contiguous(),
                reqs_c,
                slot_page_size=max(
                    descriptor.page_size // descriptor.compress_ratio, 1
                ),
            )
        )

        plan_w = plan[2].clone()
        raw_w = plan_w.view(torch.int32).reshape(-1, 2)
        ragged_w = raw_w[:, 0].to(torch.int64)
        valid_w = ragged_w >= 0
        safe_w = torch.where(valid_w, ragged_w, 0).clamp(
            max=max(request_indices_repeated.numel() - 1, 0)
        )
        reqs_w = request_indices_repeated.to(torch.int64)[safe_w].contiguous()
        logical_w = torch.where(
            valid_w,
            torch.div(
                positions.to(torch.int64)[safe_w],
                self.swa_page_size,
                rounding_mode="floor",
            ),
            -1,
        )
        if descriptor.transfer_group == DSV4TransferGroup.C128_STATE:
            logical_w = torch.where(valid_w, 0, -1)
        raw_w[:, 1].copy_(
            self.verify_mapping(
                domain,
                raw_w[:, 1].contiguous(),
                logical_w.contiguous(),
                reqs_w,
                slot_page_size=descriptor.page_size,
                invalid_value=(descriptor.capacity - 1) * descriptor.page_size,
            )
        )
        return plan._replace(plan_c=plan_c, plan_w=plan_w)

    def refresh_compressor_writes(
        self, descriptor: DSV4ComponentDescriptor, plan
    ) -> None:
        writes = (
            plan[1].view(torch.int32).reshape(-1, 4)[:, 1]
            if plan.is_decode
            else plan[2].view(torch.int32).reshape(-1, 2)[:, 1]
        )
        self.refresh_written_slots(
            descriptor, writes.contiguous(), slot_page_size=descriptor.page_size
        )

    def build_manifest(
        self,
        *,
        bootstrap_room: int,
        transfer_nonce: int,
        indices_by_group: Mapping[DSV4TransferGroup, Sequence[int] | torch.Tensor],
        logical_starts: Optional[Mapping[DSV4TransferGroup, int]] = None,
    ) -> DSV4TransferManifest:
        entries = []
        logical_starts = logical_starts or {}
        for descriptor in self.descriptors:
            raw_indices = indices_by_group.get(descriptor.transfer_group, ())
            pages = torch.as_tensor(
                raw_indices, dtype=torch.int32, device=descriptor.buffer.device
            ).reshape(-1)
            if pages.numel() == 0:
                continue
            digests = compute_page_digests(
                descriptor.buffer, pages, seed=component_seed(descriptor)
            ).cpu()
            entries.append(
                DSV4ManifestEntry(
                    component=descriptor.component,
                    layer_id=descriptor.layer_id,
                    compress_ratio=descriptor.compress_ratio,
                    transfer_group=descriptor.transfer_group,
                    page_size=descriptor.page_size,
                    item_nbytes=descriptor.item_nbytes,
                    logical_start=int(logical_starts.get(descriptor.transfer_group, 0)),
                    logical_count=pages.numel(),
                    digests=_unsigned_digest_values(digests),
                )
            )
        return DSV4TransferManifest(
            bootstrap_room=bootstrap_room,
            transfer_nonce=transfer_nonce,
            layout_digest=self.layout_digest,
            entries=tuple(entries),
        )

    def verify_and_install(
        self,
        manifest: DSV4TransferManifest,
        *,
        bootstrap_room: int,
        transfer_nonce: int,
        indices_by_group: Mapping[DSV4TransferGroup, Sequence[int] | torch.Tensor],
        logical_starts: Mapping[DSV4TransferGroup, int],
        request_index: int,
    ) -> None:
        manifest.validate()
        if manifest.bootstrap_room != bootstrap_room:
            raise DSV4ManifestError("manifest bootstrap room mismatch")
        if manifest.transfer_nonce != transfer_nonce:
            raise DSV4ManifestError("manifest transfer nonce mismatch")
        if manifest.layout_digest != self.layout_digest:
            raise DSV4ManifestError("manifest DSV4 layout mismatch")

        expected_identities = set()
        for descriptor in self.descriptors:
            indices = indices_by_group.get(descriptor.transfer_group, ())
            if len(indices):
                expected_identities.add(descriptor.identity)
        entries = {(int(e.component), e.layer_id): e for e in manifest.entries}
        if set(entries) != expected_identities:
            missing = expected_identities - set(entries)
            extra = set(entries) - expected_identities
            raise DSV4ManifestError(
                f"manifest component set mismatch (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )

        installs = []
        for identity, entry in entries.items():
            descriptor = self.by_identity[identity]
            if (
                entry.compress_ratio != descriptor.compress_ratio
                or entry.transfer_group != descriptor.transfer_group
                or entry.page_size != descriptor.page_size
                or entry.item_nbytes != descriptor.item_nbytes
            ):
                raise DSV4ManifestError("manifest component geometry mismatch")
            pages = torch.as_tensor(
                indices_by_group[descriptor.transfer_group],
                dtype=torch.int32,
                device=descriptor.buffer.device,
            ).reshape(-1)
            if pages.numel() != entry.logical_count:
                raise DSV4ManifestError("manifest logical range mismatch")
            if entry.logical_start != int(logical_starts[descriptor.transfer_group]):
                raise DSV4ManifestError("manifest logical start mismatch")
            actual = compute_page_digests(
                descriptor.buffer, pages, seed=component_seed(descriptor)
            )
            expected = torch.tensor(
                [
                    value if value < (1 << 63) else value - (1 << 64)
                    for value in entry.digests
                ],
                dtype=torch.int64,
                device=actual.device,
            )
            if not torch.equal(actual, expected):
                mismatch = torch.nonzero(actual != expected).flatten()
                first = int(mismatch[0].item()) if mismatch.numel() else -1
                raise DSV4IntegrityError(
                    f"DSV4 destination digest mismatch for {identity} at logical item {first}"
                )
            installs.append((self.sidecars[identity], pages, actual))

        # Install only after every component verifies, so partial manifests can
        # never leave a request looking protected.
        for sidecar, pages, values in installs:
            sidecar.install(pages, values)
        reqs = torch.full(
            (1,), request_index, dtype=torch.int64, device=self.failure_status.device
        )
        for group, raw_pages in indices_by_group.items():
            pages = torch.as_tensor(
                raw_pages, dtype=torch.int32, device=self.failure_status.device
            ).reshape(1, -1)
            logical_start = int(logical_starts[group])
            logical = torch.arange(
                logical_start,
                logical_start + pages.shape[1],
                dtype=torch.int64,
                device=pages.device,
            ).reshape_as(pages)
            self.bind_pages(
                self.domain_for_group(group),
                pages,
                logical,
                reqs,
                slot_page_size=1,
            )
        self.assert_request_clean(request_index)

    def refresh_pages(
        self, descriptor: DSV4ComponentDescriptor, pages: torch.Tensor
    ) -> None:
        pages = pages.reshape(-1).to(device=descriptor.buffer.device, dtype=torch.int32)
        pages = pages[(pages >= 0) & (pages < descriptor.capacity)]
        if pages.numel():
            self.sidecars[descriptor.identity].refresh(pages)

    def descriptor(
        self, component: DSV4Component, layer_id: int
    ) -> DSV4ComponentDescriptor:
        try:
            return self.by_identity[(int(component), int(layer_id))]
        except KeyError as exc:
            raise DSV4IntegrityError(
                f"missing DSV4 component {component.name} for layer {layer_id}"
            ) from exc

    def assert_clean(self) -> None:
        failures = torch.nonzero(self.failure_status).flatten()
        if failures.numel():
            first = int(failures[0].item())
            status = int(self.failure_status[first].item())
            raise DSV4IntegrityError(
                "DSV4 protected consumer rejected KV state "
                f"(request={first}, status=0x{status:08x})"
            )

    def assert_request_clean(self, request_index: int) -> None:
        if not 0 < request_index < self.request_capacity:
            raise DSV4IntegrityError("DSV4 request index is out of range")
        status = int(self.failure_status[request_index].item())
        if status:
            raise DSV4IntegrityError(
                "DSV4 request mapping was rejected before admission "
                f"(request={request_index}, status=0x{status:08x})"
            )


def reshape_state_component(pool) -> torch.Tensor:
    tensor = pool.kv_score_buffer.kv_score
    rows_per_item = pool.ring_size
    # CompressStatePool carries a trailing sentinel/padding region which is not
    # addressable by the SWA allocator or Mooncake state indices.  Protect the
    # complete transfer items and deliberately exclude only that partial tail.
    item_count = tensor.shape[0] // rows_per_item
    if item_count <= 0:
        raise ValueError("DSV4 compression-state pool has no complete items")
    return (
        tensor.narrow(0, 0, item_count * rows_per_item)
        .reshape(item_count, -1)
        .view(torch.uint8)
    )


def build_dsv4_component_descriptors(pool) -> tuple[DSV4ComponentDescriptor, ...]:
    if getattr(pool, "_unified_kv", False):
        raise ValueError("DSV4 integrity does not support unified KV")

    result = []
    stage_start = pool._stage_start
    stage_end = pool._stage_end

    def append(component, layer_id, ratio, group, page_size, buffer):
        byte_buffer = (
            buffer if buffer.dtype == torch.uint8 else buffer.view(torch.uint8)
        )
        byte_buffer = byte_buffer.reshape(byte_buffer.shape[0], -1)
        result.append(
            DSV4ComponentDescriptor(
                component=component,
                layer_id=layer_id,
                compress_ratio=ratio,
                transfer_group=group,
                page_size=page_size,
                item_nbytes=byte_buffer[0].nbytes,
                capacity=byte_buffer.shape[0],
                buffer=byte_buffer,
            )
        )

    for local_id, layer_id in enumerate(range(stage_start, stage_end)):
        append(
            DSV4Component.SWA_KV,
            layer_id,
            0,
            DSV4TransferGroup.SWA,
            pool.swa_kv_pool.page_size,
            pool.swa_kv_pool.kv_buffer[local_id],
        )

    for layer_id in range(stage_start, stage_end):
        ratio, compressed_layer_id, compressed_pool = pool.layer_mapping[layer_id]
        if ratio == 4:
            append(
                DSV4Component.C4_ATTENTION_KV,
                layer_id,
                4,
                DSV4TransferGroup.KV,
                compressed_pool.page_size,
                compressed_pool.kv_buffer[compressed_layer_id],
            )
            append(
                DSV4Component.C4_INDEXER_KV,
                layer_id,
                4,
                DSV4TransferGroup.KV,
                pool.c4_indexer_kv_pool.page_size,
                pool.c4_indexer_kv_pool.index_k_with_scale_buffer[compressed_layer_id],
            )
            append(
                DSV4Component.C4_ATTENTION_STATE,
                layer_id,
                4,
                DSV4TransferGroup.SWA,
                pool.compress_state_pools[layer_id].ring_size,
                reshape_state_component(pool.compress_state_pools[layer_id]),
            )
            append(
                DSV4Component.C4_INDEXER_STATE,
                layer_id,
                4,
                DSV4TransferGroup.SWA,
                pool.indexer_compress_state_pools[layer_id].ring_size,
                reshape_state_component(pool.indexer_compress_state_pools[layer_id]),
            )
        elif ratio == 128:
            append(
                DSV4Component.C128_ATTENTION_KV,
                layer_id,
                128,
                DSV4TransferGroup.KV,
                compressed_pool.page_size,
                compressed_pool.kv_buffer[compressed_layer_id],
            )
            append(
                DSV4Component.C128_ATTENTION_STATE,
                layer_id,
                128,
                DSV4TransferGroup.C128_STATE,
                pool.compress_state_pools[layer_id].ring_size,
                reshape_state_component(pool.compress_state_pools[layer_id]),
            )
    return tuple(result)


def require_complete_component_kinds(
    descriptors: Iterable[DSV4ComponentDescriptor],
) -> None:
    present = {descriptor.component for descriptor in descriptors}
    required = set(DSV4Component)
    missing = required - present
    if missing:
        raise ValueError(
            "protected DSV4 layout is missing component types: "
            + ", ".join(sorted(component.name for component in missing))
        )
