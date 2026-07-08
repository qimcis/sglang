"""
KV Attention Tags + Transfer Checksums for PD Disaggregation.

This module protects PD (prefill/decode) disaggregated decoding from using
stale, wrong, or mid-decode KV pages, and optionally proves that the KV bytes
copied across the network (e.g. by Mooncake) are byte-for-byte correct.

Two independent (but related) mechanisms live here:

1. Attention tags
   A sidecar GPU buffer of ``uint64`` tags, one per physical KV page, stored
   separately from the KV tensors themselves.  A tag identifies the expected
   request/page owner of whatever attention is about to read:

       tag = hash(physical_page_id, page_position, bootstrap_room, generation)

   where ``generation`` is a per-physical-page allocation generation that is
   bumped every time the page is (re)allocated.  At the decode pre-attention
   boundary we compare the expected tags for each request's logical pages
   against the sidecar buffer.  A mismatch means the page ownership changed
   since transfer commit -> we fail *only* the affected request with
   :class:`KVAttentionTagMismatch`.

2. Transfer checksums
   An optional, full byte-level proof that the KV bytes copied from
   prefill to decode are identical.  The prefill side hashes its *source* KV
   bytes in a consistent *logical* order (token-by-token, never by physical
   page id); the decode side hashes its *destination* KV bytes in the same
   logical order and compares the two checksums.  Because the comparison is in
   logical token order, prefill/decode physical page layout differences never
   cause false failures.  A mismatch fails only the affected request with
   :class:`KVChecksumError`.

Design constraints (hard requirements):
  * The attention-tag fast path performs NO full KV byte reads -- it only
     touches the small sidecar tag buffer.
  * Verification is vectorized: a single gather + compare over the batch, never
    a per-page Python loop with ``.item()`` over full-sequence pages.
  * Transfer checksums NEVER hash node-local physical page ids.
  * The whole feature is gated; see :class:`KVProtectionConfig`.  When disabled
    (the default for non-PD serving) there is zero allocator/decode overhead.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tags are conceptually uint64; torch lacks uint64 arithmetic so we store the
# bit pattern in int64 and rely on two's-complement wrap-around for the mixing
# multiplies/adds.  ``TAG_DTYPE`` is therefore int64 everywhere.
TAG_DTYPE = torch.int64

# splitmix64 constants (the bit patterns are reinterpreted as signed int64).
_SPLITMIX_ADD = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB

_U64_MASK = (1 << 64) - 1
_U32_MASK = (1 << 32) - 1
_I64_SIGN = 1 << 63
_I32_SIGN = 1 << 31

# Transfer page tags are conceptually uint32; torch stores the bit pattern in int32.
TRANSFER_PAGE_TAG_DTYPE = torch.int32


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KVPageProtectionError(Exception):
    """Base class for KV page protection failures."""


class KVAttentionTagMismatch(KVPageProtectionError):
    """Raised/recorded when a decode attention tag does not match.

    Carries enough diagnostic detail to identify the offending request/page.
    """

    def __init__(
        self,
        *,
        rid: Optional[str] = None,
        bootstrap_room: Optional[int] = None,
        page_id: Optional[int] = None,
        page_position: Optional[int] = None,
        expected_tag: Optional[int] = None,
        actual_tag: Optional[int] = None,
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.page_id = page_id
        self.page_position = page_position
        self.expected_tag = expected_tag
        self.actual_tag = actual_tag
        super().__init__(
            f"KV attention tag mismatch (rid={rid}, bootstrap_room={bootstrap_room}, "
            f"page_id={page_id}, page_position={page_position}, "
            f"expected_tag={_u64(expected_tag)}, actual_tag={_u64(actual_tag)})"
        )


class KVChecksumError(KVPageProtectionError):
    """Raised/recorded when a transfer checksum does not match."""

    def __init__(
        self,
        *,
        rid: Optional[str] = None,
        bootstrap_room: Optional[int] = None,
        expected_checksum: Optional[int] = None,
        actual_checksum: Optional[int] = None,
        num_checked_tokens: Optional[int] = None,
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.num_checked_tokens = num_checked_tokens
        super().__init__(
            f"KV transfer checksum mismatch (rid={rid}, "
            f"bootstrap_room={bootstrap_room}, "
            f"expected={_u32(expected_checksum)}, actual={_u32(actual_checksum)}, "
            f"num_checked_tokens={num_checked_tokens})"
        )


class KVTransferPageTagMismatch(KVPageProtectionError):
    """Raised/recorded when a transfer-written physical-page tag mismatches."""

    def __init__(
        self,
        *,
        rid: Optional[str] = None,
        bootstrap_room: Optional[int] = None,
        page_id: Optional[int] = None,
        page_position: Optional[int] = None,
        expected_transfer_tag: Optional[int] = None,
        actual_transfer_tag: Optional[int] = None,
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.page_id = page_id
        self.page_position = page_position
        self.expected_transfer_tag = expected_transfer_tag
        self.actual_transfer_tag = actual_transfer_tag
        super().__init__(
            f"KV transfer page tag mismatch (rid={rid}, bootstrap_room={bootstrap_room}, "
            f"page_id={page_id}, page_position={page_position}, "
            f"expected_transfer_tag={_u32(expected_transfer_tag)}, "
            f"actual_transfer_tag={_u32(actual_transfer_tag)})"
        )


def _u64(value: Optional[int]) -> Optional[int]:
    """Render a stored int64 bit pattern as its unsigned uint64 value for logs."""
    if value is None:
        return None
    return int(value) & _U64_MASK


def _u32(value: Optional[int]) -> Optional[int]:
    """Render a transfer checksum as its unsigned uint32 value for logs."""
    if value is None:
        return None
    return int(value) & _U32_MASK


# ---------------------------------------------------------------------------
# Gating / configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVProtectionConfig:
    """Resolved configuration for KV attention tags + transfer checksums.

    The feature only activates for PD disaggregation *and* when explicitly
    enabled.  ``from_env`` returns a fully-disabled config for non-PD serving so
    there is no allocator/decode overhead in the default path.
    """

    enable_attention_tags: bool = False
    enable_transfer_checksum: bool = False

    @property
    def enabled(self) -> bool:
        return self.enable_attention_tags or self.enable_transfer_checksum

    @property
    def checksum_enabled(self) -> bool:
        return self.enable_transfer_checksum

    @staticmethod
    def disabled() -> KVProtectionConfig:
        return KVProtectionConfig()

    @classmethod
    def from_env(cls, *, is_pd_decode: bool) -> KVProtectionConfig:
        """Build the config from environment variables.

        ``is_pd_decode`` gates the whole feature: outside PD disaggregation the
        protection is always disabled regardless of env vars, guaranteeing no
        regression for non-PD serving.
        """
        # Imported lazily so this module stays importable (and unit-testable)
        # without the full server/env stack.
        from sglang.srt.environ import envs

        if not is_pd_decode:
            return cls.disabled()

        enable_attention_tags = envs.SGLANG_KV_PAGE_PROTECTION.get()
        enable_transfer_checksum = envs.SGLANG_KV_TRANSFER_CHECKSUM.get()

        if not enable_attention_tags and not enable_transfer_checksum:
            return cls.disabled()

        return cls(
            enable_attention_tags=enable_attention_tags,
            enable_transfer_checksum=enable_transfer_checksum,
        )


# ---------------------------------------------------------------------------
# Unsupported-layout fail-fast
# ---------------------------------------------------------------------------

# Allocator / pool class *names* that we know how to protect.  Anything else
# (SWA, HiSparse/DSA, Mamba state, ...) must fail-fast rather than silently
# disable protection.
SUPPORTED_ALLOCATOR_CLASSES = (
    "PagedTokenToKVPoolAllocator",
    "TokenToKVPoolAllocator",
    "SWATokenToKVPoolAllocator",
)

# Transfer backends for which the transfer-checksum manifest exchange is wired.
SUPPORTED_CHECKSUM_BACKENDS = ("mooncake",)


def assert_protection_supported(
    config: KVProtectionConfig,
    *,
    allocator: object = None,
    transfer_backend: Optional[str] = None,
    is_spec_decode: bool = False,
) -> None:
    """Fail-fast when protection is enabled on an unsupported configuration.

    Rather than silently disabling protection (which would falsely claim
    success), raise a clear ``RuntimeError`` so the operator can either turn the
    feature off or run a supported layout.
    """
    if not config.enabled:
        return

    if allocator is not None:
        name = type(allocator).__name__
        if name not in SUPPORTED_ALLOCATOR_CLASSES:
            raise RuntimeError(
                "KV attention tags / transfer checksums are enabled but the "
                f"active allocator {name!r} is not supported. Supported "
                f"allocators: {SUPPORTED_ALLOCATOR_CLASSES}. Disable the feature "
                "(SGLANG_KV_PAGE_PROTECTION=0, SGLANG_KV_TRANSFER_CHECKSUM=0) "
                "or run a supported layout (plain paged/token or SWA)."
            )

    if is_spec_decode and config.enable_attention_tags:
        raise RuntimeError(
            "KV attention-tag protection does not yet support speculative decoding "
            "(multiple tokens/pages committed per step). Disable "
            "SGLANG_KV_PAGE_PROTECTION or speculative decoding."
        )

    if config.checksum_enabled and transfer_backend is not None:
        backend = str(transfer_backend).lower()
        if backend not in SUPPORTED_CHECKSUM_BACKENDS:
            raise RuntimeError(
                "KV transfer checksums are enabled but transfer backend "
                f"{transfer_backend!r} does not support the checksum manifest "
                f"exchange. Supported backends: {SUPPORTED_CHECKSUM_BACKENDS}. "
                "Set SGLANG_KV_TRANSFER_CHECKSUM=0 or use a supported backend."
            )


# ---------------------------------------------------------------------------
# Hashing primitives (scalar reference + vectorized tensor)
# ---------------------------------------------------------------------------


def _splitmix64_scalar(x: int) -> int:
    """Reference splitmix64 finalizer on a python int, returning uint64."""
    x = (x + _SPLITMIX_ADD) & _U64_MASK
    z = x
    z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _U64_MASK
    z = z ^ (z >> 31)
    return z & _U64_MASK


def _to_i64(value: int) -> int:
    """Map a uint64 value to its signed int64 bit-pattern."""
    value &= _U64_MASK
    return value - (1 << 64) if value & _I64_SIGN else value


def _to_i32(value: int) -> int:
    """Map a uint32 value to its signed int32 bit-pattern."""
    value &= _U32_MASK
    return value - (1 << 32) if value & _I32_SIGN else value


def _lshr_i64(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical (unsigned) right shift on an int64 tensor."""
    if n <= 0:
        return x
    shifted = torch.bitwise_right_shift(x, n)
    mask = _to_i64((_U64_MASK >> n))
    return torch.bitwise_and(shifted, mask)


def _splitmix64_tensor(x: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 finalizer over an int64 tensor (wraps mod 2^64)."""
    add = _to_i64(_SPLITMIX_ADD)
    m1 = _to_i64(_SPLITMIX_M1)
    m2 = _to_i64(_SPLITMIX_M2)
    x = x + add  # int64 wraps (two's complement)
    z = x
    z = (z ^ _lshr_i64(z, 30)) * m1
    z = (z ^ _lshr_i64(z, 27)) * m2
    z = z ^ _lshr_i64(z, 31)
    return z


def _fmix32_scalar(x: int) -> int:
    """Murmur3-style finalizer on a python int, returning uint32."""
    x &= _U32_MASK
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & _U32_MASK
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & _U32_MASK
    x ^= x >> 16
    return x & _U32_MASK


def _fmix32_tensor(x: torch.Tensor) -> torch.Tensor:
    """Vectorized uint32 finalizer stored in int64 tensors."""
    x = torch.bitwise_and(x.to(TAG_DTYPE), _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.bitwise_and(x * 0x85EBCA6B, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    x = torch.bitwise_and(x * 0xC2B2AE35, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    return torch.bitwise_and(x, _U32_MASK)


def _mix_scalar(acc: int, field: int) -> int:
    return _splitmix64_scalar((acc ^ (field & _U64_MASK)) & _U64_MASK)


def _mix_tensor(acc: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    return _splitmix64_tensor(torch.bitwise_xor(acc, field))


# Domain-separation seeds for the two hash families.
_TAG_SEED = 0x5347_4C41_4E47_5447  # "SGLANGTG"-ish
_CKSUM32_SEED = 0x4E474353  # Low 32 bits of "SGLANGCS".
_CKSUM32_POS_MUL = 0x9E3779B1
_CKSUM32_LANE_MUL = 0x85EBCA77
_CKSUM32_HI_MUL = 0xC2B2AE3D
_TRANSFER_PAGE_TAG_SEED = 0x5047_5447  # "PGTG".


def compute_attention_tag_scalar(
    physical_page_id: int,
    page_position: int,
    bootstrap_room: int,
    generation: int,
) -> int:
    """Reference attention ownership tag hash; returns a uint64.

    This intentionally does not hash token ids or KV bytes.  Transfer checksums
    prove byte equality; attention tags prove that the physical page attention is
    about to read is still the page generation assigned to this logical owner.
    """
    acc = _TAG_SEED
    acc = _mix_scalar(acc, bootstrap_room)
    acc = _mix_scalar(acc, page_position)
    acc = _mix_scalar(acc, physical_page_id)
    acc = _mix_scalar(acc, generation)
    return acc


def tags_to_tensor(tags: Sequence[int], device: str = "cpu") -> torch.Tensor:
    """Convert a list of uint64 tag values to an int64 (bit-pattern) tensor."""
    return torch.tensor([_to_i64(int(t)) for t in tags], dtype=TAG_DTYPE, device=device)


def compute_attention_tags_tensor(
    physical_page_ids: torch.Tensor,
    page_positions: torch.Tensor,
    bootstrap_rooms: torch.Tensor,
    generations: torch.Tensor,
) -> torch.Tensor:
    """Vectorized attention ownership tag hash for a batch of pages.

    Args:
        physical_page_ids: int64 ``[num_pages]`` physical page id per page.
        page_positions: int64 ``[num_pages]`` logical page index per page.
        bootstrap_rooms: int64 ``[num_pages]`` request bootstrap room per page.
        generations: int64 ``[num_pages]`` physical-page allocation generation.

    Returns:
        int64 ``[num_pages]`` tag tensor (uint64 bit pattern).
    """
    assert physical_page_ids.dtype in (
        torch.int32,
        torch.int64,
    ), "page ids must be integer"
    num_pages = physical_page_ids.numel()
    acc = torch.full(
        (num_pages,),
        _to_i64(_TAG_SEED),
        dtype=TAG_DTYPE,
        device=physical_page_ids.device,
    )
    acc = _mix_tensor(acc, bootstrap_rooms.to(TAG_DTYPE))
    acc = _mix_tensor(acc, page_positions.to(TAG_DTYPE))
    acc = _mix_tensor(acc, physical_page_ids.to(TAG_DTYPE))
    acc = _mix_tensor(acc, generations.to(TAG_DTYPE))
    return acc


def compute_transfer_page_tag_scalar(
    physical_page_id: int,
    page_position: int,
    bootstrap_room: int,
    generation: int,
) -> int:
    """Reference per-page transfer tag hash; returns a non-zero uint32.

    The tag is generated by decode for the physical destination page and is sent
    to prefill.  Prefill writes it into decode's actual transfer-tag table after
    writing the KV page.  A late stale transfer therefore overwrites the actual
    transfer tag with the stale request's value, and the next page-use check fails.
    """
    acc = _TRANSFER_PAGE_TAG_SEED
    for field in (bootstrap_room, page_position, physical_page_id, generation):
        acc = _fmix32_scalar(acc ^ (int(field) & _U32_MASK))
    return acc or 1


def compute_transfer_page_tags_tensor(
    physical_page_ids: torch.Tensor,
    page_positions: torch.Tensor,
    bootstrap_rooms: torch.Tensor,
    generations: torch.Tensor,
) -> torch.Tensor:
    """Vectorized transfer page tag hash stored as int32 uint32 bit patterns."""
    assert physical_page_ids.dtype in (
        torch.int32,
        torch.int64,
    ), "page ids must be integer"
    num_pages = physical_page_ids.numel()
    acc = torch.full(
        (num_pages,),
        _TRANSFER_PAGE_TAG_SEED,
        dtype=TAG_DTYPE,
        device=physical_page_ids.device,
    )
    for field in (
        bootstrap_rooms,
        page_positions,
        physical_page_ids,
        generations,
    ):
        acc = _fmix32_tensor(torch.bitwise_xor(acc, field.to(TAG_DTYPE)))
    acc = torch.where(acc == 0, torch.ones_like(acc), acc)
    return acc.to(TRANSFER_PAGE_TAG_DTYPE)


def transfer_page_tags_to_tensor(
    tags: Sequence[int], device: str = "cpu"
) -> torch.Tensor:
    """Convert uint32 transfer page tags to int32 bit-pattern tensor values."""
    return torch.tensor(
        [_to_i32(int(tag)) for tag in tags],
        dtype=TRANSFER_PAGE_TAG_DTYPE,
        device=device,
    )


# ---------------------------------------------------------------------------
# Attention ownership tag manifest
# ---------------------------------------------------------------------------


@dataclass
class AttentionTagManifest:
    """Per-request logical page ownership consumed by attention tag checks."""

    bootstrap_room: int
    page_size: int
    # logical page position -> physical page id backing it
    physical_page_ids: List[int]
    # manifest entry -> logical page position.  This is usually 0..N-1, but SWA
    # tail manifests can cover only the live sliding-window pages.
    page_positions: List[int]
    # logical page position -> allocation generation captured at write time
    generations: List[int]
    # cached expected attention tag per logical page (uint64)
    expected_tags: List[int]
    # Cached device tensors consumed by the decode verification hot path.
    physical_page_ids_t: torch.Tensor
    page_positions_t: torch.Tensor
    generations_t: torch.Tensor
    expected_tags_t: torch.Tensor

    @classmethod
    def from_pages(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: Sequence[int],
        generations: Sequence[int],
        page_positions: Optional[Sequence[int]] = None,
    ) -> AttentionTagManifest:
        phys = list(physical_page_ids)
        gens = list(generations)
        positions = (
            list(range(len(phys))) if page_positions is None else list(page_positions)
        )
        assert len(phys) == len(gens), "physical_page_ids/generations length mismatch"
        assert len(phys) == len(
            positions
        ), "physical_page_ids/page_positions length mismatch"
        expected = [
            compute_attention_tag_scalar(phys[i], positions[i], bootstrap_room, gens[i])
            for i in range(len(phys))
        ]
        physical_page_ids_t = torch.tensor(phys, dtype=torch.long)
        page_positions_t = torch.tensor(positions, dtype=TAG_DTYPE)
        generations_t = torch.tensor(gens, dtype=TAG_DTYPE)
        expected_tags_t = tags_to_tensor(expected)
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            physical_page_ids=phys,
            page_positions=positions,
            generations=gens,
            expected_tags=expected,
            physical_page_ids_t=physical_page_ids_t,
            page_positions_t=page_positions_t,
            generations_t=generations_t,
            expected_tags_t=expected_tags_t,
        )

    @property
    def num_pages(self) -> int:
        return len(self.physical_page_ids)

    def expected_tags_tensor(self, device: str = "cpu") -> torch.Tensor:
        """int64 tensor of expected tags (uint64 bit patterns) for verification."""
        return self.expected_tags_t.to(device=device, dtype=TAG_DTYPE)

    def physical_pages_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.physical_page_ids_t.to(device=device, dtype=torch.long)

    def page_positions_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.page_positions_t.to(device=device, dtype=TAG_DTYPE)

    def generations_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.generations_t.to(device=device, dtype=TAG_DTYPE)

    def _ensure_tensor_len(self, length: int, device: torch.device) -> None:
        cur = int(self.physical_page_ids_t.numel())
        if cur >= length:
            if self.physical_page_ids_t.device != device:
                self.physical_page_ids_t = self.physical_page_ids_t.to(device=device)
                self.page_positions_t = self.page_positions_t.to(device=device)
                self.generations_t = self.generations_t.to(device=device)
                self.expected_tags_t = self.expected_tags_t.to(device=device)
            return
        pad = length - cur
        self.physical_page_ids_t = torch.cat(
            [
                self.physical_page_ids_t.to(device=device),
                torch.zeros(pad, dtype=torch.long, device=device),
            ]
        )
        self.page_positions_t = torch.cat(
            [
                self.page_positions_t.to(device=device, dtype=TAG_DTYPE),
                torch.zeros(pad, dtype=TAG_DTYPE, device=device),
            ]
        )
        self.expected_tags_t = torch.cat(
            [
                self.expected_tags_t.to(device=device, dtype=TAG_DTYPE),
                torch.zeros(pad, dtype=TAG_DTYPE, device=device),
            ]
        )
        self.generations_t = torch.cat(
            [
                self.generations_t.to(device=device, dtype=TAG_DTYPE),
                torch.zeros(pad, dtype=TAG_DTYPE, device=device),
            ]
        )

    def _entry_index_for_page_position(self, page_position: int) -> Optional[int]:
        if (
            0 <= page_position < len(self.page_positions)
            and self.page_positions[page_position] == page_position
        ):
            return page_position
        for i, pos in enumerate(self.page_positions):
            if pos == page_position:
                return i
        return None

    def refresh_page_tensor(
        self,
        *,
        logical_pos: int,
        physical_page_id: torch.Tensor,
        generation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refresh the logical page containing ``logical_pos`` and return tag tensors."""
        page_position = int(logical_pos) // self.page_size
        entry_index = self._entry_index_for_page_position(page_position)
        is_new_page = entry_index is None
        if is_new_page:
            entry_index = len(self.physical_page_ids)
            self.physical_page_ids.append(0)
            self.page_positions.append(page_position)
            self.generations.append(0)
            self.expected_tags.append(0)

        physical_page_id = physical_page_id.reshape(1).to(dtype=torch.long)
        generation = generation.reshape(1).to(
            dtype=TAG_DTYPE, device=physical_page_id.device
        )
        device = physical_page_id.device
        self._ensure_tensor_len(entry_index + 1, device)
        if is_new_page:
            tag_page_id = physical_page_id
            tag_generation = generation
            self.physical_page_ids_t[entry_index : entry_index + 1] = tag_page_id
            self.page_positions_t[entry_index : entry_index + 1] = page_position
            self.generations_t[entry_index : entry_index + 1] = tag_generation
        else:
            tag_page_id = self.physical_page_ids_t[entry_index : entry_index + 1]
            tag_generation = self.generations_t[entry_index : entry_index + 1]

        expected_t = compute_attention_tags_tensor(
            tag_page_id,
            torch.tensor([page_position], dtype=TAG_DTYPE, device=device),
            torch.tensor([self.bootstrap_room], dtype=TAG_DTYPE, device=device),
            tag_generation,
        )
        self.expected_tags_t[entry_index : entry_index + 1] = expected_t

        if device.type == "cpu":
            self.physical_page_ids[entry_index] = int(tag_page_id[0].item())
            self.generations[entry_index] = int(tag_generation[0].item())
            self.expected_tags[entry_index] = int(expected_t[0].item())
        return tag_page_id, expected_t


@dataclass
class AttentionTagManifestGroup:
    """Full/SWA manifests for one request, verified and failed as one owner."""

    manifests: Tuple[AttentionTagManifest, ...]

    @property
    def bootstrap_room(self) -> int:
        return self.manifests[0].bootstrap_room if self.manifests else 0

    @property
    def num_pages(self) -> int:
        return sum(manifest.num_pages for manifest in self.manifests)


def _iter_attention_manifests(manifest) -> Tuple[AttentionTagManifest, ...]:
    if manifest is None:
        return ()
    if isinstance(manifest, AttentionTagManifestGroup):
        return manifest.manifests
    if isinstance(manifest, AttentionTagManifest):
        return (manifest,)
    if isinstance(manifest, (list, tuple)):
        return tuple(m for m in manifest if m is not None)
    return (manifest,)


@dataclass
class TransferPageTagManifest:
    """Per-request expected transfer-written page tags."""

    bootstrap_room: int
    page_size: int
    physical_page_ids: List[int]
    page_positions: List[int]
    generations: List[int]
    expected_tags: List[int]
    physical_page_ids_t: torch.Tensor
    page_positions_t: torch.Tensor
    generations_t: torch.Tensor
    expected_tags_t: torch.Tensor

    @classmethod
    def from_pages(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: Sequence[int],
        generations: Sequence[int],
        page_positions: Optional[Sequence[int]] = None,
    ) -> TransferPageTagManifest:
        phys = list(physical_page_ids)
        gens = list(generations)
        positions = (
            list(range(len(phys))) if page_positions is None else list(page_positions)
        )
        assert len(phys) == len(gens), "physical_page_ids/generations length mismatch"
        assert len(phys) == len(
            positions
        ), "physical_page_ids/page_positions length mismatch"
        expected = [
            compute_transfer_page_tag_scalar(
                phys[i], positions[i], bootstrap_room, gens[i]
            )
            for i in range(len(phys))
        ]
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            physical_page_ids=phys,
            page_positions=positions,
            generations=gens,
            expected_tags=expected,
            physical_page_ids_t=torch.tensor(phys, dtype=torch.long),
            page_positions_t=torch.tensor(positions, dtype=TAG_DTYPE),
            generations_t=torch.tensor(gens, dtype=TAG_DTYPE),
            expected_tags_t=transfer_page_tags_to_tensor(expected),
        )

    @property
    def num_pages(self) -> int:
        return len(self.physical_page_ids)

    def physical_pages_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.physical_page_ids_t.to(device=device, dtype=torch.long)

    def expected_tags_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.expected_tags_t.to(device=device, dtype=TRANSFER_PAGE_TAG_DTYPE)

    def generations_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.generations_t.to(device=device, dtype=TAG_DTYPE)

    def _ensure_tensor_len(self, length: int, device: torch.device) -> None:
        cur = int(self.physical_page_ids_t.numel())
        if cur >= length:
            if self.physical_page_ids_t.device != device:
                self.physical_page_ids_t = self.physical_page_ids_t.to(device=device)
                self.page_positions_t = self.page_positions_t.to(device=device)
                self.generations_t = self.generations_t.to(device=device)
                self.expected_tags_t = self.expected_tags_t.to(device=device)
            return
        pad = length - cur
        self.physical_page_ids_t = torch.cat(
            [
                self.physical_page_ids_t.to(device=device),
                torch.zeros(pad, dtype=torch.long, device=device),
            ]
        )
        self.page_positions_t = torch.cat(
            [
                self.page_positions_t.to(device=device, dtype=TAG_DTYPE),
                torch.zeros(pad, dtype=TAG_DTYPE, device=device),
            ]
        )
        self.generations_t = torch.cat(
            [
                self.generations_t.to(device=device, dtype=TAG_DTYPE),
                torch.zeros(pad, dtype=TAG_DTYPE, device=device),
            ]
        )
        self.expected_tags_t = torch.cat(
            [
                self.expected_tags_t.to(device=device, dtype=TRANSFER_PAGE_TAG_DTYPE),
                torch.zeros(pad, dtype=TRANSFER_PAGE_TAG_DTYPE, device=device),
            ]
        )

    def _entry_index_for_page_position(self, page_position: int) -> Optional[int]:
        if (
            0 <= page_position < len(self.page_positions)
            and self.page_positions[page_position] == page_position
        ):
            return page_position
        for i, pos in enumerate(self.page_positions):
            if pos == page_position:
                return i
        return None

    def refresh_page_tensor(
        self,
        *,
        logical_pos: int,
        physical_page_id: torch.Tensor,
        generation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refresh the logical page containing ``logical_pos`` and return tag."""
        page_position = int(logical_pos) // self.page_size
        entry_index = self._entry_index_for_page_position(page_position)
        is_new_page = entry_index is None
        if is_new_page:
            entry_index = len(self.physical_page_ids)
            self.physical_page_ids.append(0)
            self.page_positions.append(page_position)
            self.generations.append(0)
            self.expected_tags.append(0)

        physical_page_id = physical_page_id.reshape(1).to(dtype=torch.long)
        generation = generation.reshape(1).to(
            dtype=TAG_DTYPE, device=physical_page_id.device
        )
        device = physical_page_id.device
        self._ensure_tensor_len(entry_index + 1, device)
        if is_new_page:
            tag_page_id = physical_page_id
            tag_generation = generation
            self.physical_page_ids_t[entry_index : entry_index + 1] = tag_page_id
            self.page_positions_t[entry_index : entry_index + 1] = page_position
            self.generations_t[entry_index : entry_index + 1] = tag_generation
        else:
            tag_page_id = self.physical_page_ids_t[entry_index : entry_index + 1]
            tag_generation = self.generations_t[entry_index : entry_index + 1]

        expected_t = compute_transfer_page_tags_tensor(
            tag_page_id,
            torch.tensor([page_position], dtype=TAG_DTYPE, device=device),
            torch.tensor([self.bootstrap_room], dtype=TAG_DTYPE, device=device),
            tag_generation,
        )
        self.expected_tags_t[entry_index : entry_index + 1] = expected_t

        if device.type == "cpu":
            self.physical_page_ids[entry_index] = int(tag_page_id[0].item())
            self.generations[entry_index] = int(tag_generation[0].item())
            self.expected_tags[entry_index] = int(expected_t[0].item())
        return tag_page_id, expected_t


@dataclass
class TransferPageTagManifestGroup:
    """Full/SWA transfer page tag manifests for one request."""

    manifests: Tuple[TransferPageTagManifest, ...]

    @property
    def bootstrap_room(self) -> int:
        return self.manifests[0].bootstrap_room if self.manifests else 0

    @property
    def num_pages(self) -> int:
        return sum(manifest.num_pages for manifest in self.manifests)


def _iter_transfer_page_tag_manifests(
    manifest,
) -> Tuple[TransferPageTagManifest, ...]:
    if manifest is None:
        return ()
    if isinstance(manifest, TransferPageTagManifestGroup):
        return manifest.manifests
    if isinstance(manifest, TransferPageTagManifest):
        return (manifest,)
    if isinstance(manifest, (list, tuple)):
        return tuple(m for m in manifest if m is not None)
    return (manifest,)


# ---------------------------------------------------------------------------
# Sidecar attention-tag table + vectorized verification
# ---------------------------------------------------------------------------


class KVAttentionTagTable:
    """Sidecar GPU buffer of per-physical-page attention tags + generations.

    Stored separately from KV tensors.  Allocated lazily and only when KV page
    protection is enabled, so the default serving path pays nothing.
    """

    def __init__(self, num_pages: int, device: str = "cpu"):
        # +1 so physical page ids (which are 1-based in the paged allocator) fit.
        self._size = num_pages + 1
        self.device = device
        self.tags = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)
        self.generations = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)
        self.transfer_page_tags = torch.zeros(
            self._size, dtype=TRANSFER_PAGE_TAG_DTYPE, device=device
        )

    @property
    def size(self) -> int:
        return self._size

    def bump_generations(self, page_ids: torch.Tensor) -> None:
        """Increment the allocation generation of the given physical pages.

        Called only on *newly allocated* physical pages, so reused pages get a
        fresh generation and any stale expected tag from a prior owner will no
        longer match.
        """
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        self.generations.index_add_(
            0, page_ids, torch.ones_like(page_ids, dtype=TAG_DTYPE)
        )

    def generation_of(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        return self.generations.index_select(0, page_ids)

    def write_tags(self, page_ids: torch.Tensor, tags: torch.Tensor) -> None:
        """Scatter attention ownership tags into the sidecar buffer."""
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        tags = tags.to(self.device, dtype=TAG_DTYPE).reshape(-1)
        self.tags.index_copy_(0, page_ids, tags)

    def read_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        return self.tags.index_select(0, page_ids)

    def write_transfer_page_tags(
        self, page_ids: torch.Tensor, tags: torch.Tensor
    ) -> None:
        """Scatter transfer-written page tags into the sidecar buffer."""
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        tags = tags.to(self.device, dtype=TRANSFER_PAGE_TAG_DTYPE).reshape(-1)
        self.transfer_page_tags.index_copy_(0, page_ids, tags)

    def read_transfer_page_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        return self.transfer_page_tags.index_select(0, page_ids)


def verify_attention_tags(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """Vectorized batch verification of attention ownership tags.

    A vectorized gather + compare over the whole batch -- no per-page Python
    loop or ``.item()`` over full-sequence pages.  When provided, allocation
    generations are compared in the same vectorized path so a page reuse cannot
    be hidden by refreshing a tail tag.  Returns ``(all_ok, mismatch_mask)``.
    Only one ``.item()`` (the ``.any()`` short-circuit) is performed;
    per-element diagnostics are extracted only on the rare mismatch path.
    """
    if page_ids.numel() == 0:
        return True, torch.zeros(0, dtype=torch.bool, device=table.device)
    actual = table.read_tags(page_ids)
    expected = expected_tags.to(table.device, dtype=TAG_DTYPE).reshape(-1)
    mismatch = actual != expected
    if expected_generations is not None:
        actual_generations = table.generation_of(page_ids)
        expected_generations = expected_generations.to(
            table.device, dtype=TAG_DTYPE
        ).reshape(-1)
        mismatch |= actual_generations != expected_generations
    all_ok = not bool(mismatch.any().item())
    return all_ok, mismatch


def verify_transfer_page_tags(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """Vectorized verification of transfer-written page tags."""
    if page_ids.numel() == 0:
        return True, torch.zeros(0, dtype=torch.bool, device=table.device)
    actual = table.read_transfer_page_tags(page_ids)
    expected = expected_tags.to(table.device, dtype=TRANSFER_PAGE_TAG_DTYPE).reshape(-1)
    mismatch = actual != expected
    if expected_generations is not None:
        actual_generations = table.generation_of(page_ids)
        expected_generations = expected_generations.to(
            table.device, dtype=TAG_DTYPE
        ).reshape(-1)
        mismatch |= actual_generations != expected_generations
    all_ok = not bool(mismatch.any().item())
    return all_ok, mismatch


# ---------------------------------------------------------------------------
# Transfer checksums (logical order, physical-page-id independent)
# ---------------------------------------------------------------------------


def select_checksum_token_indices(
    num_tokens: int,
    bootstrap_room: int,
    sample_rate: float,
) -> torch.Tensor:
    """Choose logical token indices to checksum.

    Checksums are always full-strength when enabled: every logical token is
    included.  ``bootstrap_room`` and ``sample_rate`` are retained in the helper
    signature only for call-site stability while mode-based sampling is removed.
    """
    del bootstrap_room, sample_rate
    if num_tokens <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(num_tokens, dtype=torch.long)


def select_checksum_byte_count(row_nbytes: int) -> int:
    """Number of leading int64 lanes per token row to hash.

    Checksums are always full-byte when enabled, so this is the whole row.
    """
    return max(1, row_nbytes // 8)


def hash_kv_rows(
    rows: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    num_lanes: Optional[int] = None,
    include_positions: bool = True,
) -> int:
    """Hash KV bytes in *logical* token order, independent of physical layout.

    Args:
        rows: ``[num_tokens, row_len]`` tensor of KV bytes for the request, in
            *logical* token order (token 0..N-1).  Any integer/byte dtype is
            accepted; it is reinterpreted as int64 lanes for hashing.  The
            caller is responsible for gathering rows in logical order -- this
            function never sees, and therefore cannot depend on, physical page
            ids.
        token_indices: logical token indices to include.
        num_lanes: optional cap on the number of leading int64 lanes/row to hash.
            ``None`` hashes the whole row.
        include_positions: fold the logical token index into the hash so that a
            reordering of tokens is detected.

    Returns:
        uint32 checksum stored as a Python int.
    """
    if token_indices.numel() == 0:
        return _fmix32_scalar(_CKSUM32_SEED)

    lanes = _as_int64_lanes(rows)
    sel = lanes.index_select(0, token_indices.to(lanes.device, dtype=torch.long))
    positions = token_indices if include_positions else None
    return hash_rows_with_positions(sel, positions=positions, num_lanes=num_lanes)


def hash_rows_with_positions(
    rows: torch.Tensor,
    *,
    positions: Optional[torch.Tensor] = None,
    num_lanes: Optional[int] = None,
) -> int:
    """Hash already-selected, logically-ordered KV rows.

    ``rows`` are the selected token rows in logical order; ``positions`` are
    their logical token indices (folded in for order-sensitivity).  Neither
    physical page ids nor physical slot indices ever enter the hash.
    """
    if rows.numel() == 0 or rows.shape[0] == 0:
        return _fmix32_scalar(_CKSUM32_SEED)
    lanes = _as_int64_lanes(rows)
    if num_lanes is not None:
        lanes = lanes[:, :num_lanes]
    if lanes.numel() == 0 or lanes.shape[1] == 0:
        return _fmix32_scalar(_CKSUM32_SEED)

    # Each 8-byte lane is an independent chunk salted by logical token position
    # and lane offset, so CUDA can parallelize inside a token row.
    lo = torch.bitwise_and(lanes, _U32_MASK)
    hi = torch.bitwise_and(_lshr_i64(lanes, 32), _U32_MASK)
    lane_offsets = torch.arange(lanes.shape[1], dtype=TAG_DTYPE, device=lanes.device)
    h = torch.full(lanes.shape, _CKSUM32_SEED, dtype=TAG_DTYPE, device=lanes.device)
    if positions is not None:
        pos = positions.to(TAG_DTYPE).to(lanes.device).reshape(-1, 1)
        h = torch.bitwise_xor(h, torch.bitwise_and(pos * _CKSUM32_POS_MUL, _U32_MASK))
    h = torch.bitwise_xor(
        h,
        torch.bitwise_and(lane_offsets.reshape(1, -1) * _CKSUM32_LANE_MUL, _U32_MASK),
    )
    h = torch.bitwise_xor(h, lo)
    h = torch.bitwise_xor(h, torch.bitwise_and(hi * _CKSUM32_HI_MUL, _U32_MASK))
    chunks = _fmix32_tensor(h)

    combined = _xor_reduce(chunks.reshape(-1))
    return _fmix32_scalar(_CKSUM32_SEED ^ combined ^ int(lanes.shape[0]))


def _xor_reduce(x: torch.Tensor) -> int:
    """XOR-reduce an int64 tensor to a single python int (one host sync).

    Implemented as a vectorized log-step halving fold so there is no per-element
    Python loop / ``.item()`` over the tokens.
    """
    y = x
    while y.numel() > 1:
        n = y.numel()
        half = n // 2
        merged = torch.bitwise_xor(y[:half], y[half : 2 * half])
        if n % 2:
            merged = torch.cat([merged, y[-1:]])
        y = merged
    return int(y.item())


def _as_int64_lanes(rows: torch.Tensor) -> torch.Tensor:
    """Reinterpret a 2D row tensor as int64 lanes, zero-padding to a multiple of 8 bytes."""
    if rows.dim() == 1:
        rows = rows.unsqueeze(1)
    assert rows.dim() == 2, "rows must be 1D or 2D"
    # Make contiguous and view as bytes.
    rows = rows.contiguous()
    byte_view = rows.view(torch.uint8).reshape(rows.shape[0], -1)
    row_bytes = byte_view.shape[1]
    pad = (-row_bytes) % 8
    if pad:
        byte_view = torch.nn.functional.pad(byte_view, (0, pad))
    lanes = byte_view.view(rows.shape[0], -1)
    # Reinterpret 8 bytes -> int64 lane.
    lanes = lanes.contiguous().view(torch.int64).reshape(rows.shape[0], -1)
    return lanes


@dataclass
class ChecksumPlan:
    """A request's transfer-checksum plan, exchanged prefill -> decode.

    Contains only logical, layout-independent information.  Crucially it does
    NOT contain physical page ids on either side.
    """

    bootstrap_room: int
    num_tokens: int
    checksum: int  # the prefill-side uint32 source checksum

    def to_payload(self) -> dict:
        return {
            "bootstrap_room": int(self.bootstrap_room),
            "num_tokens": int(self.num_tokens),
            "checksum": int(self.checksum) & _U32_MASK,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> ChecksumPlan:
        return cls(
            bootstrap_room=int(payload["bootstrap_room"]),
            num_tokens=int(payload["num_tokens"]),
            checksum=int(payload["checksum"]) & _U32_MASK,
        )


@dataclass
class _DirectKVChecksumCache:
    """Reusable CUDA tensors for direct KV checksum metadata and row output."""

    metadata_key: Optional[Tuple[int, str, int]] = None
    buffer_ptrs: Optional[torch.Tensor] = None
    row_strides: Optional[torch.Tensor] = None
    row_nbytes: Optional[torch.Tensor] = None
    swa_buffer_flags: Optional[torch.Tensor] = None
    full_to_swa_index_mapping: Optional[torch.Tensor] = None
    total_bytes: int = 0
    out: Optional[torch.Tensor] = None
    accum: Optional[torch.Tensor] = None
    final_out: Optional[torch.Tensor] = None
    req_pool_indices: Optional[torch.Tensor] = None
    starts: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    stream: Optional[torch.cuda.Stream] = None

    def out_slice(self, numel: int, device: torch.device) -> torch.Tensor:
        if self.out is None or self.out.device != device or self.out.numel() < numel:
            self.out = torch.empty((numel,), dtype=TAG_DTYPE, device=device)
        return self.out[:numel]

    def batch_slices(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.accum is None
            or self.accum.device != device
            or self.accum.numel() < batch_size
        ):
            self.accum = torch.empty((batch_size,), dtype=torch.int32, device=device)
        if (
            self.final_out is None
            or self.final_out.device != device
            or self.final_out.numel() < batch_size
        ):
            self.final_out = torch.empty((batch_size,), dtype=TAG_DTYPE, device=device)
        if (
            self.req_pool_indices is None
            or self.req_pool_indices.device != device
            or self.req_pool_indices.numel() < batch_size
        ):
            self.req_pool_indices = torch.empty(
                (batch_size,), dtype=TAG_DTYPE, device=device
            )
        if (
            self.starts is None
            or self.starts.device != device
            or self.starts.numel() < batch_size
        ):
            self.starts = torch.empty((batch_size,), dtype=TAG_DTYPE, device=device)
        if (
            self.lengths is None
            or self.lengths.device != device
            or self.lengths.numel() < batch_size
        ):
            self.lengths = torch.empty((batch_size,), dtype=TAG_DTYPE, device=device)
        return (
            self.accum[:batch_size],
            self.final_out[:batch_size],
            self.req_pool_indices[:batch_size],
            self.starts[:batch_size],
            self.lengths[:batch_size],
        )


@dataclass
class AsyncChecksumBatch:
    """GPU-side batched checksum result finalized once at metadata boundary."""

    bootstrap_rooms: List[int]
    num_tokens: List[int]
    checksums_t: torch.Tensor
    stream: Optional[torch.cuda.Stream]
    finalized: Optional[List[ChecksumPlan]] = None

    def finalize(self) -> List[ChecksumPlan]:
        if self.finalized is not None:
            return self.finalized
        if self.stream is not None:
            torch.cuda.current_stream(self.checksums_t.device).wait_stream(self.stream)
        checksums = self.checksums_t.detach().cpu().tolist()
        self.finalized = [
            ChecksumPlan(
                bootstrap_room=room,
                num_tokens=n,
                checksum=int(checksum) & _U32_MASK,
            )
            for room, n, checksum in zip(
                self.bootstrap_rooms, self.num_tokens, checksums, strict=True
            )
        ]
        return self.finalized


def _direct_metadata_from_pool(
    kv_pool: object,
    device: torch.device,
    cache: Optional[_DirectKVChecksumCache],
    unsupported,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    metadata_key = (id(kv_pool), str(device), int(kv_pool.layer_num))
    if cache is not None and cache.metadata_key == metadata_key:
        buffer_ptrs = cache.buffer_ptrs
        row_strides = cache.row_strides
        row_nbytes = cache.row_nbytes
        swa_buffer_flags = cache.swa_buffer_flags
        full_to_swa_index_mapping = cache.full_to_swa_index_mapping
        total_bytes = cache.total_bytes
        if (
            buffer_ptrs is None
            or row_strides is None
            or row_nbytes is None
            or swa_buffer_flags is None
            or full_to_swa_index_mapping is None
        ):
            return unsupported("cached direct checksum metadata is incomplete")
        return (
            buffer_ptrs,
            row_strides,
            row_nbytes,
            swa_buffer_flags,
            full_to_swa_index_mapping,
            total_bytes,
        )

    # Build the buffer list in the same logical byte order used by the Python
    # checksum reference: K(l0), V(l0)?, K(l1), V(l1)?, ...
    # V is skipped when the pool folds V into K.
    buffers: List[torch.Tensor] = []
    is_swa_buffer: List[int] = []
    layers_mapping = getattr(kv_pool, "layers_mapping", None)
    for layer_id in range(int(kv_pool.layer_num)):
        is_swa = 0
        if layers_mapping is not None:
            try:
                is_swa = 1 if bool(layers_mapping[layer_id][1]) else 0
            except KeyError:
                return unsupported(f"missing SWA layer mapping for layer {layer_id}")
        buffers.append(kv_pool.get_key_buffer(layer_id))
        is_swa_buffer.append(is_swa)
        try:
            buffers.append(kv_pool.get_value_buffer(layer_id))
            is_swa_buffer.append(is_swa)
        except (NotImplementedError, AttributeError):
            pass
    if not buffers:
        return unsupported("no KV buffers exposed")

    ptrs: List[int] = []
    strides: List[int] = []
    nbytes: List[int] = []
    total_bytes = 0
    for buf in buffers:
        if not isinstance(buf, torch.Tensor) or not buf.is_cuda:
            return unsupported("a KV buffer is not a CUDA tensor")
        if buf.device != device:
            return unsupported("KV buffer / kv_loc device mismatch")
        if buf.dim() < 1 or buf.shape[0] == 0:
            return unsupported("degenerate KV buffer shape")
        expected_stride = 1
        for dim in range(buf.dim() - 1, 0, -1):
            if int(buf.shape[dim]) != 1 and int(buf.stride(dim)) != expected_stride:
                return unsupported("a KV buffer row is not contiguous")
            expected_stride *= int(buf.shape[dim])
        itemsize = buf.element_size()
        per_row_elems = 1
        for s in buf.shape[1:]:
            per_row_elems *= int(s)
        row_b = per_row_elems * itemsize
        if row_b == 0 or row_b % 8 != 0:
            return unsupported("per-buffer row bytes not a positive multiple of 8")
        stride0_b = int(buf.stride(0)) * itemsize
        if stride0_b % 8 != 0:
            return unsupported("per-buffer dim-0 stride not a multiple of 8")
        base = int(buf.data_ptr())
        if base % 8 != 0:
            return unsupported("KV buffer base pointer not 8-byte aligned")
        ptrs.append(base)
        strides.append(stride0_b)
        nbytes.append(row_b)
        total_bytes += row_b

    buffer_ptrs = torch.tensor(ptrs, dtype=TAG_DTYPE, device=device)
    row_strides = torch.tensor(strides, dtype=TAG_DTYPE, device=device)
    row_nbytes = torch.tensor(nbytes, dtype=TAG_DTYPE, device=device)
    swa_buffer_flags = torch.tensor(is_swa_buffer, dtype=TAG_DTYPE, device=device)
    if any(is_swa_buffer):
        full_to_swa_index_mapping = getattr(kv_pool, "full_to_swa_index_mapping", None)
        if not isinstance(full_to_swa_index_mapping, torch.Tensor):
            return unsupported("SWA KV pool does not expose full-to-SWA mapping")
        if (
            not full_to_swa_index_mapping.is_cuda
            or full_to_swa_index_mapping.device != device
        ):
            return unsupported("SWA full-to-SWA mapping device mismatch")
        full_to_swa_index_mapping = full_to_swa_index_mapping.contiguous()
    else:
        full_to_swa_index_mapping = torch.empty(0, dtype=TAG_DTYPE, device=device)
    if cache is not None:
        cache.metadata_key = metadata_key
        cache.buffer_ptrs = buffer_ptrs
        cache.row_strides = row_strides
        cache.row_nbytes = row_nbytes
        cache.swa_buffer_flags = swa_buffer_flags
        cache.full_to_swa_index_mapping = full_to_swa_index_mapping
        cache.total_bytes = total_bytes
    return (
        buffer_ptrs,
        row_strides,
        row_nbytes,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        total_bytes,
    )


def compare_checksums(expected: ChecksumPlan, actual_checksum: int) -> bool:
    """Return True if the decode-side checksum matches the prefill-side plan."""
    return (int(expected.checksum) & _U32_MASK) == (int(actual_checksum) & _U32_MASK)


def _to_int_list(values: Sequence[int] | torch.Tensor) -> List[int]:
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in values]


# ---------------------------------------------------------------------------
# High-level manager (decode side)
# ---------------------------------------------------------------------------


class KVPageProtectionManager:
    """Owns the sidecar table and drives registration/verification for PD decode.

    Constructed only when protection is enabled for PD decode.  It fails fast on
    unsupported allocators/backends so we never silently disable protection
    while claiming success.  All verification is vectorized; per-element work
    happens only on the (rare) mismatch path for diagnostics.
    """

    def __init__(
        self,
        config: KVProtectionConfig,
        *,
        allocator: object,
        num_pages: int,
        page_size: int,
        device: str = "cpu",
        metrics_collector: object = None,
        transfer_backend: Optional[str] = None,
        is_spec_decode: bool = False,
    ):
        assert_protection_supported(
            config,
            allocator=allocator,
            transfer_backend=transfer_backend,
            is_spec_decode=is_spec_decode,
        )
        self.config = config
        self.page_size = page_size
        self.device = device
        self.metrics = metrics_collector
        self.table: Optional[KVAttentionTagTable] = None
        self._checksum_cache = _DirectKVChecksumCache()
        if config.enable_attention_tags:
            self.table = KVAttentionTagTable(num_pages, device=device)
            if allocator is not None and hasattr(
                allocator, "attach_attention_tag_table"
            ):
                allocator.attach_attention_tag_table(self.table)

    # -- attention ownership tags ------------------------------------------

    def register_attention_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        bootstrap_room: int,
        page_positions: Optional[Sequence[int]] = None,
    ) -> Optional[AttentionTagManifest]:
        """Write attention ownership tags for a freshly transferred request.

        Captures each physical page's current allocation generation, computes
        the expected ownership tag, scatters it into the sidecar buffer, and returns a
        manifest cached on the request for later (per-step) verification.
        """
        if not self.config.enable_attention_tags or self.table is None:
            return None
        page_ids = _to_int_list(page_physical_ids)
        positions = None if page_positions is None else _to_int_list(page_positions)
        pages_t = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        generations = self.table.generation_of(pages_t).tolist()
        manifest = AttentionTagManifest.from_pages(
            self.page_size,
            bootstrap_room,
            page_ids,
            generations,
            page_positions=positions,
        )
        manifest.physical_page_ids_t = pages_t[: manifest.num_pages]
        manifest.page_positions_t = manifest.page_positions_t.to(device=self.device)
        manifest.generations_t = manifest.generations_t.to(device=self.device)
        manifest.expected_tags_t = manifest.expected_tags_tensor(self.device)
        self.table.write_tags(
            manifest.physical_page_ids_t,
            manifest.expected_tags_t,
        )
        return manifest

    def verify_request(
        self,
        manifest: Optional[AttentionTagManifest],
        *,
        rid: Optional[str] = None,
    ) -> Optional[KVAttentionTagMismatch]:
        """Verify one request's pages; return an exception object on mismatch.

        Returns ``None`` when protection is disabled or all pages match.  Only a
        single ``.any()`` host sync occurs on the happy path.
        """
        if (
            not self.config.enable_attention_tags
            or manifest is None
            or self.table is None
        ):
            return None
        if isinstance(manifest, AttentionTagManifestGroup) or isinstance(
            manifest, (list, tuple)
        ):
            mismatches = self.verify_batch([(rid, manifest)])
            return mismatches[0] if mismatches else None
        if manifest.num_pages == 0:
            return None
        pages = manifest.physical_pages_tensor(self.device)
        expected = manifest.expected_tags_tensor(self.device)
        generations = manifest.generations_tensor(self.device)
        ok, mismatch = verify_attention_tags(self.table, pages, expected, generations)
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_checked_pages(int(pages.numel()))
        if ok:
            return None
        # Rare path: extract the first offending page for diagnostics.
        bad = int(torch.nonzero(mismatch).reshape(-1)[0].item())
        expected_tag = int(expected[bad].item())
        actual_tag = int(self.table.read_tags(pages[bad : bad + 1])[0].item())
        exc = KVAttentionTagMismatch(
            rid=rid,
            bootstrap_room=manifest.bootstrap_room,
            page_id=int(pages[bad].item()),
            page_position=manifest.page_positions[bad],
            expected_tag=expected_tag,
            actual_tag=actual_tag,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_mismatches()
        return exc

    def refresh_tail_page(
        self,
        manifest: Optional[AttentionTagManifest],
        *,
        logical_pos: int,
        physical_page_id: torch.Tensor,
        swa_physical_page_id: Optional[torch.Tensor] = None,
    ) -> None:
        """Refresh the logical page touched by a decode append.

        ``physical_page_id`` is a one-element device tensor derived from
        ``batch.out_cache_loc``.  Keeping it as a tensor avoids GPU-to-CPU scalar
        syncs in the decode hot path.
        """
        if (
            not self.config.enable_attention_tags
            or manifest is None
            or self.table is None
        ):
            return
        manifests = _iter_attention_manifests(manifest)
        if len(manifests) != 1:
            if manifests:
                self.refresh_tail_page(
                    manifests[0],
                    logical_pos=logical_pos,
                    physical_page_id=physical_page_id,
                )
            if len(manifests) > 1 and swa_physical_page_id is not None:
                self.refresh_tail_page(
                    manifests[1],
                    logical_pos=logical_pos,
                    physical_page_id=swa_physical_page_id,
                )
            return
        manifest = manifests[0]
        physical_page_id = physical_page_id.reshape(1).to(
            device=self.device, dtype=torch.long
        )
        generation = self.table.generation_of(physical_page_id)
        page_id_t, expected_t = manifest.refresh_page_tensor(
            logical_pos=logical_pos,
            physical_page_id=physical_page_id,
            generation=generation,
        )
        self.table.write_tags(page_id_t, expected_t)

    # -- transfer-written page tags ----------------------------------------

    def register_transfer_page_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        bootstrap_room: int,
        page_positions: Optional[Sequence[int]] = None,
        write_actual: bool = False,
    ) -> Optional[TransferPageTagManifest]:
        """Create expected transfer page tags for decode-owned pages.

        ``write_actual`` is False for PD-transferred prompt pages: prefill writes
        those actual tags after the KV transfer completes. It is True only for
        locally produced decode pages that never pass through prefill transfer.
        """
        if not self.config.enable_attention_tags or self.table is None:
            return None
        page_ids = _to_int_list(page_physical_ids)
        positions = None if page_positions is None else _to_int_list(page_positions)
        pages_t = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        generations = self.table.generation_of(pages_t).tolist()
        manifest = TransferPageTagManifest.from_pages(
            self.page_size,
            bootstrap_room,
            page_ids,
            generations,
            page_positions=positions,
        )
        manifest.physical_page_ids_t = pages_t[: manifest.num_pages]
        manifest.page_positions_t = manifest.page_positions_t.to(device=self.device)
        manifest.generations_t = manifest.generations_t.to(device=self.device)
        manifest.expected_tags_t = manifest.expected_tags_tensor(self.device)
        if write_actual:
            self.table.write_transfer_page_tags(
                manifest.physical_page_ids_t,
                manifest.expected_tags_t,
            )
        return manifest

    def write_transfer_page_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        transfer_page_tags: Sequence[int],
    ) -> None:
        """Write actual transfer page tags produced by the transfer path."""
        if not self.config.enable_attention_tags or self.table is None:
            return
        page_ids = _to_int_list(page_physical_ids)
        tags = _to_int_list(transfer_page_tags)
        if len(page_ids) != len(tags):
            raise RuntimeError("transfer page tag metadata length mismatch")
        if not page_ids:
            return
        pages_t = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        tags_t = transfer_page_tags_to_tensor(tags, device=self.device)
        self.table.write_transfer_page_tags(pages_t, tags_t)

    def refresh_transfer_page_tag_tail_page(
        self,
        manifest: Optional[TransferPageTagManifest],
        *,
        logical_pos: int,
        physical_page_id: torch.Tensor,
        swa_physical_page_id: Optional[torch.Tensor] = None,
    ) -> None:
        """Refresh transfer page tags for locally appended decode pages."""
        if (
            not self.config.enable_attention_tags
            or manifest is None
            or self.table is None
        ):
            return
        manifests = _iter_transfer_page_tag_manifests(manifest)
        if len(manifests) != 1:
            if manifests:
                self.refresh_transfer_page_tag_tail_page(
                    manifests[0],
                    logical_pos=logical_pos,
                    physical_page_id=physical_page_id,
                )
            if len(manifests) > 1 and swa_physical_page_id is not None:
                self.refresh_transfer_page_tag_tail_page(
                    manifests[1],
                    logical_pos=logical_pos,
                    physical_page_id=swa_physical_page_id,
                )
            return
        manifest = manifests[0]
        physical_page_id = physical_page_id.reshape(1).to(
            device=self.device, dtype=torch.long
        )
        generation = self.table.generation_of(physical_page_id)
        page_id_t, expected_t = manifest.refresh_page_tensor(
            logical_pos=logical_pos,
            physical_page_id=physical_page_id,
            generation=generation,
        )
        self.table.write_transfer_page_tags(page_id_t, expected_t)

    def verify_transfer_page_tag_batch(
        self,
        items: Sequence[Tuple[str, TransferPageTagManifest]],
    ) -> List[KVTransferPageTagMismatch]:
        """Vectorized verification of actual transfer page tags for a batch."""
        if not self.config.enable_attention_tags or self.table is None:
            return []
        page_tensors: List[torch.Tensor] = []
        expected_tensors: List[torch.Tensor] = []
        generation_tensors: List[torch.Tensor] = []
        owners: List[Tuple[str, TransferPageTagManifest]] = []
        offsets: List[int] = [0]
        for rid, manifest in items:
            for sub_manifest in _iter_transfer_page_tag_manifests(manifest):
                if sub_manifest is None:
                    continue
                pages = sub_manifest.physical_pages_tensor(self.device)
                if pages.numel() == 0:
                    continue
                expected = sub_manifest.expected_tags_tensor(self.device)
                generations = sub_manifest.generations_tensor(self.device)
                page_tensors.append(pages)
                expected_tensors.append(expected)
                generation_tensors.append(generations)
                owners.append((rid, sub_manifest))
                offsets.append(offsets[-1] + int(pages.numel()))
        if not page_tensors:
            return []
        pages_t = torch.cat(page_tensors)
        expected_t = torch.cat(expected_tensors)
        generations_t = torch.cat(generation_tensors)
        ok, mismatch = verify_transfer_page_tags(
            self.table, pages_t, expected_t, generations_t
        )
        if ok:
            return []
        actual_all = self.table.read_transfer_page_tags(pages_t)
        bad_idx = torch.nonzero(mismatch).reshape(-1).cpu().tolist()
        seen_rids = set()
        result: List[KVTransferPageTagMismatch] = []
        for i in bad_idx:
            owner_idx = bisect_right(offsets, int(i)) - 1
            rid, manifest = owners[owner_idx]
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            p = int(i) - offsets[owner_idx]
            result.append(
                KVTransferPageTagMismatch(
                    rid=rid,
                    bootstrap_room=manifest.bootstrap_room,
                    page_id=int(pages_t[i].item()),
                    page_position=manifest.page_positions[p],
                    expected_transfer_tag=int(expected_t[i].item()),
                    actual_transfer_tag=int(actual_all[i].item()),
                )
            )
        if self.metrics is not None and result:
            self.metrics.increment_kv_attention_tag_mismatches(len(result))
        return result

    def verify_batch(
        self,
        items: Sequence[Tuple[str, AttentionTagManifest]],
    ) -> List[KVAttentionTagMismatch]:
        """Vectorized verification across a whole decode batch.

        ``items`` is a sequence of ``(rid, manifest)``.  All pages and expected
        tag/generation tensors are concatenated and verified with vectorized
        gathers+compares; per-req diagnostics are produced only for the requests
        that actually mismatched.
        Returns the list of mismatches (empty when all pass).  Metrics are
        incremented for checked pages and per mismatch.
        """
        if not self.config.enable_attention_tags or self.table is None:
            return []
        page_tensors: List[torch.Tensor] = []
        expected_tensors: List[torch.Tensor] = []
        generation_tensors: List[torch.Tensor] = []
        owners: List[Tuple[str, AttentionTagManifest]] = []
        offsets: List[int] = [0]
        for rid, manifest in items:
            for sub_manifest in _iter_attention_manifests(manifest):
                if sub_manifest is None:
                    continue
                pages = sub_manifest.physical_pages_tensor(self.device)
                if pages.numel() == 0:
                    continue
                expected = sub_manifest.expected_tags_tensor(self.device)
                generations = sub_manifest.generations_tensor(self.device)
                page_tensors.append(pages)
                expected_tensors.append(expected)
                generation_tensors.append(generations)
                owners.append((rid, sub_manifest))
                offsets.append(offsets[-1] + int(pages.numel()))
        if not page_tensors:
            return []
        pages_t = torch.cat(page_tensors)
        expected_t = torch.cat(expected_tensors)
        generations_t = torch.cat(generation_tensors)
        ok, mismatch = verify_attention_tags(
            self.table, pages_t, expected_t, generations_t
        )
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_checked_pages(int(pages_t.numel()))
        if ok:
            return []
        actual_all = self.table.read_tags(pages_t)
        bad_idx = torch.nonzero(mismatch).reshape(-1).cpu().tolist()
        # Report at most one mismatch per request (the first offending page).
        seen_rids = set()
        result: List[KVAttentionTagMismatch] = []
        for i in bad_idx:
            owner_idx = bisect_right(offsets, int(i)) - 1
            rid, manifest = owners[owner_idx]
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            p = int(i) - offsets[owner_idx]
            result.append(
                KVAttentionTagMismatch(
                    rid=rid,
                    bootstrap_room=manifest.bootstrap_room,
                    page_id=int(pages_t[i].item()),
                    page_position=manifest.page_positions[p],
                    expected_tag=int(expected_t[i].item()),
                    actual_tag=int(actual_all[i].item()),
                )
            )
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_mismatches(len(result))
        return result

    # -- transfer checksums ------------------------------------------------

    def begin_transfer_checksums_from_table(
        self,
        kv_pool: object,
        req_to_token: torch.Tensor,
        *,
        req_pool_indices: Sequence[int],
        bootstrap_rooms: Sequence[int],
        num_tokens: Sequence[int],
        starts: Optional[Sequence[int]] = None,
    ) -> Optional[AsyncChecksumBatch]:
        """Launch batched direct checksums from the request-token table.

        The returned object keeps the finalized checksum tensor on GPU until the
        caller reaches the metadata publication/verification boundary, avoiding
        per-request `.item()` synchronization.
        """
        if not self.config.checksum_enabled:
            return None
        if not isinstance(req_to_token, torch.Tensor) or not req_to_token.is_cuda:
            raise RuntimeError("batched direct checksum requires CUDA req_to_token")
        if len(req_pool_indices) == 0:
            return AsyncChecksumBatch([], [], torch.empty(0, dtype=TAG_DTYPE), None)
        if not (len(req_pool_indices) == len(bootstrap_rooms) == len(num_tokens)):
            raise RuntimeError("batched checksum metadata length mismatch")
        starts_list = (
            list(starts) if starts is not None else [0] * len(req_pool_indices)
        )
        if len(starts_list) != len(req_pool_indices):
            raise RuntimeError("batched checksum starts length mismatch")

        try:
            from sgl_kernel.kvcacheio import kv_checksum_direct_table_batched as _op
        except Exception as e:  # pragma: no cover - depends on CUDA build
            raise RuntimeError(
                f"sgl_kernel kv_checksum_direct_table_batched unavailable ({e})"
            ) from e

        device = req_to_token.device

        def _unsupported(reason: str) -> None:
            raise RuntimeError(
                "Batched direct KV checksum kernel is required but cannot run: "
                f"{reason}. Run a supported contiguous CUDA KV layout with "
                "8-byte-aligned rows and a built batched checksum op."
            )

        (
            buffer_ptrs,
            row_strides,
            row_nbytes,
            swa_buffer_flags,
            full_to_swa_index_mapping,
            total_bytes,
        ) = _direct_metadata_from_pool(
            kv_pool, device, self._checksum_cache, _unsupported
        )
        num_lanes = select_checksum_byte_count(total_bytes)
        has_swa = full_to_swa_index_mapping.numel() > 0
        is_capped = (int(num_lanes) * 8) < int(total_bytes)
        batch_size = len(req_pool_indices)
        accum, final_out, req_pool_t, starts_t, lengths_t = (
            self._checksum_cache.batch_slices(batch_size, device)
        )

        req_pool_t.copy_(
            torch.as_tensor(req_pool_indices, dtype=TAG_DTYPE, device=device),
            non_blocking=True,
        )
        starts_t.copy_(
            torch.as_tensor(starts_list, dtype=TAG_DTYPE, device=device),
            non_blocking=True,
        )
        lengths_t.copy_(
            torch.as_tensor(num_tokens, dtype=TAG_DTYPE, device=device),
            non_blocking=True,
        )

        if (
            self._checksum_cache.stream is None
            or self._checksum_cache.stream.device != device
        ):
            self._checksum_cache.stream = torch.cuda.Stream(device=device)
        stream = self._checksum_cache.stream
        stream.wait_stream(torch.cuda.current_stream(device))
        max_num_tokens = max(int(n) for n in num_tokens)
        with torch.cuda.stream(stream):
            accum.zero_()
            _op(
                buffer_ptrs,
                row_strides,
                row_nbytes,
                swa_buffer_flags,
                full_to_swa_index_mapping,
                req_to_token,
                req_pool_t,
                starts_t,
                lengths_t,
                int(max_num_tokens),
                int(num_lanes),
                has_swa,
                is_capped,
                accum,
                final_out,
            )
            checksums_t = final_out.clone()
        return AsyncChecksumBatch(
            bootstrap_rooms=[int(x) for x in bootstrap_rooms],
            num_tokens=[int(x) for x in num_tokens],
            checksums_t=checksums_t,
            stream=stream,
        )

    def compare_destination_checksum(
        self, expected: ChecksumPlan, actual_checksum: int, *, rid: Optional[str] = None
    ) -> bool:
        """Compare a precomputed destination checksum and update metrics."""
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_checked_pages(
                int(expected.num_tokens)
            )
        ok = compare_checksums(expected, actual_checksum)
        if not ok and self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_mismatches()
        del rid
        return ok
