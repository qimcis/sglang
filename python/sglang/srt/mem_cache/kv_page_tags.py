"""
KV Page Token Tags + Transfer Checksums for PD Disaggregation.

This module protects PD (prefill/decode) disaggregated decoding from using
stale, wrong, or mid-decode KV pages, and optionally proves that the KV bytes
copied across the network (e.g. by Mooncake) are byte-for-byte correct.

Two independent (but related) mechanisms live here:

1. Page token tags
   A sidecar GPU buffer of ``uint64`` tags, one per physical KV page, stored
   separately from the KV tensors themselves.  A tag identifies the logical
   content/owner of whatever currently lives in a physical page:

       tag = hash(tokens_in_page, page_position, bootstrap_room, generation)

   where ``generation`` is a per-physical-page allocation generation that is
   bumped every time the page is (re)allocated.  Before decode attention reads
   a request's pages, we recompute/look up the expected tag for each logical
   page and compare it (vectorized) against the sidecar buffer.  A mismatch
   means the page was reallocated, overwritten by another request, or never
   correctly populated -> we fail *only* the affected request with
   :class:`KVPageTagMismatch`.

2. Transfer checksums
   An optional, sampled or full, byte-level proof that the KV bytes copied from
   prefill to decode are identical.  The prefill side hashes its *source* KV
   bytes in a consistent *logical* order (token-by-token, never by physical
   page id); the decode side hashes its *destination* KV bytes in the same
   logical order and compares the two checksums.  Because the comparison is in
   logical token order, prefill/decode physical page layout differences never
   cause false failures.  A mismatch fails only the affected request with
   :class:`KVChecksumError`.

Design constraints (hard requirements):
  * The page-tag fast path performs NO full KV byte reads -- it only touches the
    small sidecar tag buffer.
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
from enum import Enum
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

# Sentinel used to pad logical token slots in a not-yet-full page so that a
# partial page hashes differently from the eventual full page.
_TOKEN_PAD = -1

# splitmix64 constants (the bit patterns are reinterpreted as signed int64).
_SPLITMIX_ADD = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB

_U64_MASK = (1 << 64) - 1
_I64_SIGN = 1 << 63


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KVPageProtectionError(Exception):
    """Base class for KV page protection failures."""


class KVPageTagMismatch(KVPageProtectionError):
    """Raised/recorded when a decode page tag does not match the expected tag.

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
            f"KV page tag mismatch (rid={rid}, bootstrap_room={bootstrap_room}, "
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
            f"expected={_u64(expected_checksum)}, actual={_u64(actual_checksum)}, "
            f"num_checked_tokens={num_checked_tokens})"
        )


def _u64(value: Optional[int]) -> Optional[int]:
    """Render a stored int64 bit pattern as its unsigned uint64 value for logs."""
    if value is None:
        return None
    return int(value) & _U64_MASK


# ---------------------------------------------------------------------------
# Checksum modes
# ---------------------------------------------------------------------------


class ChecksumMode(Enum):
    """Transfer checksum verification strength.

    NONE          : no transfer checksums.
    SAMPLED_PARTIAL: hash a deterministic sample of tokens, partial bytes/token.
    SAMPLED_FULL  : hash a deterministic sample of tokens, all bytes/token.
    ALWAYS_FULL   : hash every token, all bytes/token.
    """

    NONE = "none"
    SAMPLED_PARTIAL = "sampled_partial"
    SAMPLED_FULL = "sampled_full"
    ALWAYS_FULL = "always_full"

    @property
    def enabled(self) -> bool:
        return self is not ChecksumMode.NONE

    @property
    def is_sampled(self) -> bool:
        return self in (ChecksumMode.SAMPLED_PARTIAL, ChecksumMode.SAMPLED_FULL)

    @property
    def is_partial_bytes(self) -> bool:
        return self is ChecksumMode.SAMPLED_PARTIAL


# Accept a few friendly aliases without silently misparsing typos.
_CHECKSUM_ALIASES = {
    "none": ChecksumMode.NONE,
    "off": ChecksumMode.NONE,
    "": ChecksumMode.NONE,
    "sample_partial": ChecksumMode.SAMPLED_PARTIAL,
    "sampled_partial": ChecksumMode.SAMPLED_PARTIAL,
    "partial": ChecksumMode.SAMPLED_PARTIAL,
    "sampled_full": ChecksumMode.SAMPLED_FULL,
    "sample_full": ChecksumMode.SAMPLED_FULL,
    "always_full": ChecksumMode.ALWAYS_FULL,
    "full": ChecksumMode.ALWAYS_FULL,
}


# Stable integer codes for transmitting the mode through a numeric side channel
# (e.g. the spare metadata-buffer slots).  Append-only; never renumber.
_CHECKSUM_MODE_CODES = {
    ChecksumMode.NONE: 0,
    ChecksumMode.SAMPLED_PARTIAL: 1,
    ChecksumMode.SAMPLED_FULL: 2,
    ChecksumMode.ALWAYS_FULL: 3,
}
_CHECKSUM_CODE_TO_MODE = {code: mode for mode, code in _CHECKSUM_MODE_CODES.items()}


def checksum_mode_to_code(mode: ChecksumMode) -> int:
    return _CHECKSUM_MODE_CODES[mode]


def checksum_code_to_mode(code: int) -> ChecksumMode:
    return _CHECKSUM_CODE_TO_MODE.get(int(code), ChecksumMode.NONE)


def parse_checksum_mode(value: Optional[str]) -> ChecksumMode:
    """Parse a checksum-mode string; raises ValueError on an unknown value."""
    if value is None:
        return ChecksumMode.NONE
    key = str(value).strip().lower()
    if key not in _CHECKSUM_ALIASES:
        valid = ", ".join(sorted({m.value for m in ChecksumMode}))
        raise ValueError(
            f"Invalid KV transfer checksum mode {value!r}; expected one of: {valid}"
        )
    return _CHECKSUM_ALIASES[key]


# ---------------------------------------------------------------------------
# Gating / configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVProtectionConfig:
    """Resolved configuration for KV page protection + transfer checksums.

    The feature only activates for PD disaggregation *and* when explicitly
    enabled.  ``from_env`` returns a fully-disabled config for non-PD serving so
    there is no allocator/decode overhead in the default path.
    """

    enable_page_tags: bool = False
    checksum_mode: ChecksumMode = ChecksumMode.NONE
    # Fraction of tokens sampled when ``checksum_mode.is_sampled``.
    checksum_sample_rate: float = 0.05
    # Fraction of each token's bytes hashed when partial-byte sampling.
    checksum_partial_byte_rate: float = 0.25

    @property
    def enabled(self) -> bool:
        return self.enable_page_tags or self.checksum_mode.enabled

    @property
    def checksum_enabled(self) -> bool:
        return self.checksum_mode.enabled

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

        enable_page_tags = envs.SGLANG_KV_PAGE_PROTECTION.get()
        try:
            checksum_mode = parse_checksum_mode(
                envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.get()
            )
        except ValueError as e:
            logger.warning("%s; disabling KV transfer checksums.", e)
            checksum_mode = ChecksumMode.NONE

        if not enable_page_tags and not checksum_mode.enabled:
            return cls.disabled()

        sample_rate = float(envs.SGLANG_KV_CHECKSUM_SAMPLE_RATE.get())
        sample_rate = min(max(sample_rate, 0.0), 1.0)
        byte_rate = float(envs.SGLANG_KV_CHECKSUM_PARTIAL_BYTE_RATE.get())
        byte_rate = min(max(byte_rate, 0.0), 1.0)

        return cls(
            enable_page_tags=enable_page_tags,
            checksum_mode=checksum_mode,
            checksum_sample_rate=sample_rate,
            checksum_partial_byte_rate=byte_rate,
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
                "KV page protection / transfer checksums are enabled but the "
                f"active allocator {name!r} is not supported. Supported "
                f"allocators: {SUPPORTED_ALLOCATOR_CLASSES}. Disable the feature "
                "(SGLANG_KV_PAGE_PROTECTION=0, SGLANG_KV_TRANSFER_CHECKSUM_MODE="
                "none) or run a supported layout (plain paged, non-SWA/non-DSA)."
            )

    if is_spec_decode and config.enable_page_tags:
        raise RuntimeError(
            "KV page protection does not yet support speculative decoding "
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
                "Set SGLANG_KV_TRANSFER_CHECKSUM_MODE=none or use a supported "
                "backend."
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


def _mix_scalar(acc: int, field: int) -> int:
    return _splitmix64_scalar((acc ^ (field & _U64_MASK)) & _U64_MASK)


def _mix_tensor(acc: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    return _splitmix64_tensor(torch.bitwise_xor(acc, field))


# Domain-separation seeds (uint64) for the two hash families.
_TAG_SEED = 0x5347_4C41_4E47_5447  # "SGLANGTG"-ish
_CKSUM_SEED = 0x5347_4C41_4E47_4353  # "SGLANGCS"-ish


def compute_page_tag_scalar(
    tokens_in_page: Sequence[int],
    page_position: int,
    bootstrap_room: int,
    generation: int,
    page_size: int,
) -> int:
    """Reference (python) page-tag hash; returns a uint64.

    The tokens are folded position-by-position, padding empty slots so that a
    partial page hashes differently from the eventual full page; the valid
    token count is folded in as well.
    """
    acc = _TAG_SEED
    acc = _mix_scalar(acc, bootstrap_room)
    acc = _mix_scalar(acc, page_position)
    acc = _mix_scalar(acc, generation)
    acc = _mix_scalar(acc, len(tokens_in_page))
    for i in range(page_size):
        tok = tokens_in_page[i] if i < len(tokens_in_page) else _TOKEN_PAD
        acc = _mix_scalar(acc, tok & _U64_MASK)
    return acc


def tags_to_tensor(tags: Sequence[int], device: str = "cpu") -> torch.Tensor:
    """Convert a list of uint64 tag values to an int64 (bit-pattern) tensor."""
    return torch.tensor([_to_i64(int(t)) for t in tags], dtype=TAG_DTYPE, device=device)


def compute_page_tags_tensor(
    tokens: torch.Tensor,
    page_positions: torch.Tensor,
    bootstrap_rooms: torch.Tensor,
    generations: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    """Vectorized page-tag hash for a batch of pages.

    Args:
        tokens: int64 ``[num_pages, page_size]`` token ids; pad empty slots with
            ``_TOKEN_PAD``.
        page_positions: int64 ``[num_pages]`` logical page index per page.
        bootstrap_rooms: int64 ``[num_pages]`` request bootstrap room per page.
        generations: int64 ``[num_pages]`` physical-page allocation generation.
        valid_counts: int64 ``[num_pages]`` number of valid tokens in the page.

    Returns:
        int64 ``[num_pages]`` tag tensor (uint64 bit pattern).
    """
    assert tokens.dtype == TAG_DTYPE, "tokens must be int64"
    num_pages, page_size = tokens.shape
    acc = torch.full(
        (num_pages,), _to_i64(_TAG_SEED), dtype=TAG_DTYPE, device=tokens.device
    )
    acc = _mix_tensor(acc, bootstrap_rooms.to(TAG_DTYPE))
    acc = _mix_tensor(acc, page_positions.to(TAG_DTYPE))
    acc = _mix_tensor(acc, generations.to(TAG_DTYPE))
    acc = _mix_tensor(acc, valid_counts.to(TAG_DTYPE))
    for i in range(page_size):
        acc = _mix_tensor(acc, tokens[:, i])
    return acc


# ---------------------------------------------------------------------------
# Logical page manifest
# ---------------------------------------------------------------------------


@dataclass
class PageManifest:
    """Per-request logical page layout used to compute/verify page tags.

    Stores everything in *logical* terms (token ids, logical page positions,
    bootstrap room) plus the *physical* page ids currently backing each logical
    page and the allocation generation captured when the tag was written.  This
    is cached on the request and only refreshed for pages whose logical content
    changed (e.g. a newly committed partial page), so the decode hot path never
    recomputes tags for all full-sequence pages each step.
    """

    bootstrap_room: int
    page_size: int
    # logical page position -> list of token ids in that page (<= page_size)
    pages: List[List[int]]
    # logical page position -> physical page id backing it
    physical_page_ids: List[int]
    # logical page position -> allocation generation captured at write time
    generations: List[int]
    # cached expected tag per logical page (uint64)
    expected_tags: List[int]
    # Cached device tensors consumed by the decode verification hot path.
    physical_page_ids_t: torch.Tensor
    generations_t: torch.Tensor
    expected_tags_t: torch.Tensor

    @classmethod
    def from_tokens(
        cls,
        token_ids: Sequence[int],
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: Sequence[int],
        generations: Sequence[int],
    ) -> PageManifest:
        pages: List[List[int]] = [
            list(token_ids[i : i + page_size])
            for i in range(0, max(len(token_ids), 1), page_size)
        ]
        if not token_ids:
            pages = []
        num_pages = len(pages)
        phys = list(physical_page_ids[:num_pages])
        gens = list(generations[:num_pages])
        assert len(phys) == num_pages, "physical_page_ids/pages length mismatch"
        assert len(gens) == num_pages, "generations/pages length mismatch"
        expected = [
            compute_page_tag_scalar(pages[p], p, bootstrap_room, gens[p], page_size)
            for p in range(num_pages)
        ]
        physical_page_ids_t = torch.tensor(phys, dtype=torch.long)
        generations_t = torch.tensor(gens, dtype=TAG_DTYPE)
        expected_tags_t = tags_to_tensor(expected)
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            pages=pages,
            physical_page_ids=phys,
            generations=gens,
            expected_tags=expected,
            physical_page_ids_t=physical_page_ids_t,
            generations_t=generations_t,
            expected_tags_t=expected_tags_t,
        )

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def expected_tags_tensor(self, device: str = "cpu") -> torch.Tensor:
        """int64 tensor of expected tags (uint64 bit patterns) for verification."""
        return self.expected_tags_t.to(device=device, dtype=TAG_DTYPE)

    def physical_pages_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.physical_page_ids_t.to(device=device, dtype=torch.long)

    def generations_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.generations_t.to(device=device, dtype=TAG_DTYPE)

    def _ensure_tensor_len(self, length: int, device: torch.device) -> None:
        cur = int(self.physical_page_ids_t.numel())
        if cur >= length:
            if self.physical_page_ids_t.device != device:
                self.physical_page_ids_t = self.physical_page_ids_t.to(device=device)
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

    def refresh_page(
        self,
        page_position: int,
        token_ids: Sequence[int],
        physical_page_id: int,
        generation: int,
    ) -> None:
        """Refresh a single logical page's expected tag (e.g. it just filled).

        Only the changed page is recomputed; all other pages keep their cached
        tags.  This keeps the per-decode-step refresh O(1), not O(seq_len).
        """
        while len(self.pages) <= page_position:
            self.pages.append([])
            self.physical_page_ids.append(0)
            self.generations.append(0)
            self.expected_tags.append(0)
        self.pages[page_position] = list(token_ids)
        self.physical_page_ids[page_position] = physical_page_id
        self.generations[page_position] = generation
        self.expected_tags[page_position] = compute_page_tag_scalar(
            list(token_ids),
            page_position,
            self.bootstrap_room,
            generation,
            self.page_size,
        )
        device = self.expected_tags_t.device
        self._ensure_tensor_len(page_position + 1, device)
        self.physical_page_ids_t[page_position] = int(physical_page_id)
        self.generations_t[page_position] = int(generation)
        self.expected_tags_t[page_position] = _to_i64(self.expected_tags[page_position])

    def refresh_token_tensor(
        self,
        *,
        logical_pos: int,
        token_id: int,
        physical_page_id: torch.Tensor,
        generation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refresh the changed tail page and return its expected tag tensor.

        The decode hot path updates only the page containing ``logical_pos``.
        Full request page lists/tensors are cached on this manifest and are not
        rebuilt.  Python lists are maintained only as page-sized bookkeeping for
        future tail updates and mismatch diagnostics.
        """
        page_position = int(logical_pos) // self.page_size
        offset = int(logical_pos) % self.page_size
        is_new_page = page_position >= len(self.pages)
        while len(self.pages) <= page_position:
            self.pages.append([])
            self.physical_page_ids.append(0)
            self.generations.append(0)
            self.expected_tags.append(0)

        page_tokens = self.pages[page_position]
        while len(page_tokens) < offset:
            page_tokens.append(_TOKEN_PAD)
        if len(page_tokens) == offset:
            page_tokens.append(int(token_id))
        else:
            page_tokens[offset] = int(token_id)
        if len(page_tokens) > self.page_size:
            del page_tokens[self.page_size :]

        physical_page_id = physical_page_id.reshape(1).to(dtype=torch.long)
        generation = generation.reshape(1).to(
            dtype=TAG_DTYPE, device=physical_page_id.device
        )
        device = physical_page_id.device
        self._ensure_tensor_len(page_position + 1, device)
        if is_new_page:
            tag_page_id = physical_page_id
            tag_generation = generation
            self.physical_page_ids_t[page_position : page_position + 1] = tag_page_id
            self.generations_t[page_position : page_position + 1] = tag_generation
        else:
            tag_page_id = self.physical_page_ids_t[page_position : page_position + 1]
            tag_generation = self.generations_t[page_position : page_position + 1]

        tokens_t = torch.full(
            (1, self.page_size), _TOKEN_PAD, dtype=TAG_DTYPE, device=device
        )
        tokens_t[0, : len(page_tokens)] = torch.tensor(
            page_tokens, dtype=TAG_DTYPE, device=device
        )
        expected_t = compute_page_tags_tensor(
            tokens_t,
            torch.tensor([page_position], dtype=TAG_DTYPE, device=device),
            torch.tensor([self.bootstrap_room], dtype=TAG_DTYPE, device=device),
            tag_generation,
            torch.tensor([len(page_tokens)], dtype=TAG_DTYPE, device=device),
        )
        self.expected_tags_t[page_position : page_position + 1] = expected_t

        if device.type == "cpu":
            self.physical_page_ids[page_position] = int(tag_page_id[0].item())
            self.generations[page_position] = int(tag_generation[0].item())
            self.expected_tags[page_position] = int(expected_t[0].item())
        return tag_page_id, expected_t


# ---------------------------------------------------------------------------
# Sidecar page-tag table + vectorized verification
# ---------------------------------------------------------------------------


class KVPageTagTable:
    """Sidecar GPU buffer of per-physical-page tags + allocation generations.

    Stored separately from KV tensors.  Allocated lazily and only when KV page
    protection is enabled, so the default serving path pays nothing.
    """

    def __init__(self, num_pages: int, device: str = "cpu"):
        # +1 so physical page ids (which are 1-based in the paged allocator) fit.
        self._size = num_pages + 1
        self.device = device
        self.tags = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)
        self.generations = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)

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
        """Scatter logical-content tags into the sidecar buffer."""
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        tags = tags.to(self.device, dtype=TAG_DTYPE).reshape(-1)
        self.tags.index_copy_(0, page_ids, tags)

    def read_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.device, dtype=torch.long).reshape(-1)
        return self.tags.index_select(0, page_ids)


def verify_page_tags(
    table: KVPageTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """Vectorized batch verification of page tags.

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


# ---------------------------------------------------------------------------
# Transfer checksums (logical order, physical-page-id independent)
# ---------------------------------------------------------------------------


def select_checksum_token_indices(
    num_tokens: int,
    bootstrap_room: int,
    mode: ChecksumMode,
    sample_rate: float,
) -> torch.Tensor:
    """Deterministically choose which logical token indices to checksum.

    The selection depends only on ``(num_tokens, bootstrap_room, mode,
    sample_rate)`` -- never on physical layout -- so prefill and decode pick the
    *same* logical tokens independently.
    """
    if num_tokens <= 0 or not mode.enabled:
        return torch.empty(0, dtype=torch.long)
    if not mode.is_sampled:
        return torch.arange(num_tokens, dtype=torch.long)

    k = max(1, int(round(num_tokens * sample_rate)))
    k = min(k, num_tokens)
    if k >= num_tokens:
        return torch.arange(num_tokens, dtype=torch.long)

    # Deterministic, layout-independent stride sampling seeded by bootstrap_room.
    start = _splitmix64_scalar(bootstrap_room) % num_tokens
    # Use a stride coprime-ish with num_tokens to spread samples out.
    stride = max(1, num_tokens // k)
    idx = (start + torch.arange(k, dtype=torch.long) * stride) % num_tokens
    return torch.unique(idx)


def select_checksum_byte_count(
    row_nbytes: int,
    mode: ChecksumMode,
    partial_byte_rate: float,
) -> int:
    """Number of leading int64 lanes per token row to hash for a given mode."""
    lanes = max(1, row_nbytes // 8)
    if mode.is_partial_bytes:
        lanes = max(1, int(round(lanes * partial_byte_rate)))
    return lanes


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
        token_indices: logical token indices to include (subset for sampling).
        num_lanes: optional cap on the number of leading int64 lanes/row to hash
            (partial-byte sampling).  ``None`` hashes the whole row.
        include_positions: fold the logical token index into the hash so that a
            reordering of tokens is detected.

    Returns:
        uint64 checksum.
    """
    if token_indices.numel() == 0:
        return _splitmix64_scalar(_CKSUM_SEED)

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

    ``rows`` are the (sampled) token rows in logical order; ``positions`` are
    their logical token indices (folded in for order-sensitivity).  Neither
    physical page ids nor physical slot indices ever enter the hash.
    """
    if rows.numel() == 0 or rows.shape[0] == 0:
        return _splitmix64_scalar(_CKSUM_SEED)
    lanes = _as_int64_lanes(rows)
    if num_lanes is not None:
        lanes = lanes[:, :num_lanes]

    # Fold each lane across tokens with a per-position twist so that both
    # content and order matter.  One ``_mix_tensor`` per int64 lane (lane count
    # is a small per-token constant, never O(seq_len)).
    acc = torch.full(
        (lanes.shape[0],), _to_i64(_CKSUM_SEED), dtype=TAG_DTYPE, device=lanes.device
    )
    if positions is not None:
        acc = _mix_tensor(acc, positions.to(TAG_DTYPE).to(lanes.device))
    for j in range(lanes.shape[1]):
        acc = _mix_tensor(acc, lanes[:, j])

    # Each per-token accumulator already includes its logical position, so a
    # plain XOR-reduce stays order-sensitive while requiring only one
    # device->host sync.
    combined = _xor_reduce(acc)
    total = _mix_scalar(_CKSUM_SEED, combined)
    total = _mix_scalar(total, int(lanes.shape[0]))
    return total


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
    mode: ChecksumMode
    num_lanes: Optional[int]
    checksum: int  # the prefill-side (source) checksum

    def to_payload(self) -> dict:
        return {
            "bootstrap_room": int(self.bootstrap_room),
            "num_tokens": int(self.num_tokens),
            "mode": self.mode.value,
            "num_lanes": (None if self.num_lanes is None else int(self.num_lanes)),
            "checksum": _u64(self.checksum),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> ChecksumPlan:
        return cls(
            bootstrap_room=int(payload["bootstrap_room"]),
            num_tokens=int(payload["num_tokens"]),
            mode=parse_checksum_mode(payload["mode"]),
            num_lanes=(
                None if payload["num_lanes"] is None else int(payload["num_lanes"])
            ),
            checksum=_to_i64(int(payload["checksum"])),
        )


def compute_transfer_checksum(
    rows: torch.Tensor,
    *,
    bootstrap_room: int,
    num_tokens: int,
    mode: ChecksumMode,
    config: KVProtectionConfig,
    row_nbytes: Optional[int] = None,
) -> ChecksumPlan:
    """Compute a (source or destination) transfer checksum over logical rows.

    Both prefill and decode call this with their own physically-gathered-but-
    logically-ordered ``rows``; identical KV bytes yield identical checksums
    regardless of physical page placement.
    """
    indices = select_checksum_token_indices(
        num_tokens, bootstrap_room, mode, config.checksum_sample_rate
    )
    if row_nbytes is None:
        # Infer from the tensor.
        row_nbytes = (
            rows.contiguous().view(torch.uint8).reshape(rows.shape[0], -1).shape[1]
            if rows.numel()
            else 0
        )
    num_lanes = (
        select_checksum_byte_count(row_nbytes, mode, config.checksum_partial_byte_rate)
        if mode.enabled
        else None
    )
    checksum = hash_kv_rows(rows, indices, num_lanes=num_lanes)
    return ChecksumPlan(
        bootstrap_room=bootstrap_room,
        num_tokens=num_tokens,
        mode=mode,
        num_lanes=num_lanes,
        checksum=checksum,
    )


def gather_logical_kv_rows(
    kv_pool: object,
    kv_loc: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather KV bytes for the selected logical tokens, in logical order.

    For each selected logical token we concatenate its K and V across all layers
    into one row.  Rows are ordered by logical token (the order of
    ``token_indices``), so the resulting tensor is independent of how the tokens
    are scattered across physical pages: we read the *bytes* living at each
    token's physical slot, but never fold the slot/page id into the data.

    Fails fast (raises) for KV pools that do not expose the standard
    per-layer ``get_key_buffer``/``get_value_buffer`` accessors, rather than
    silently skipping the checksum.
    """
    if not (
        hasattr(kv_pool, "get_key_buffer")
        and hasattr(kv_pool, "get_value_buffer")
        and hasattr(kv_pool, "layer_num")
    ):
        raise RuntimeError(
            "KV transfer checksum is enabled but the active KV cache "
            f"{type(kv_pool).__name__!r} does not expose a per-layer "
            "get_key_buffer/get_value_buffer gather. Disable checksums or run a "
            "supported (MHA/MLA contiguous) KV cache."
        )
    sel_loc = kv_loc.to(torch.long).index_select(
        0, token_indices.to(kv_loc.device, dtype=torch.long)
    )
    parts: List[torch.Tensor] = []
    for layer_id in range(int(kv_pool.layer_num)):
        k = kv_pool.get_key_buffer(layer_id).index_select(0, sel_loc)
        parts.append(k.reshape(k.shape[0], -1))
        try:
            v = kv_pool.get_value_buffer(layer_id).index_select(0, sel_loc)
            parts.append(v.reshape(v.shape[0], -1))
        except (NotImplementedError, AttributeError):
            # MLA-style caches may fold V into K; K alone still proves transfer.
            pass
    return torch.cat(parts, dim=1)


def compare_checksums(expected: ChecksumPlan, actual_checksum: int) -> bool:
    """Return True if the decode-side checksum matches the prefill-side plan."""
    return _to_i64(int(expected.checksum)) == _to_i64(int(actual_checksum))


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
        self.table: Optional[KVPageTagTable] = None
        if config.enable_page_tags:
            self.table = KVPageTagTable(num_pages, device=device)
            if allocator is not None and hasattr(allocator, "attach_page_tag_table"):
                allocator.attach_page_tag_table(self.table)

    # -- page tags ---------------------------------------------------------

    def register_pages(
        self,
        *,
        token_ids: Sequence[int],
        page_physical_ids: Sequence[int],
        bootstrap_room: int,
    ) -> Optional[PageManifest]:
        """Write logical-content tags for a freshly transferred request.

        Captures each physical page's current allocation generation, computes
        the logical tag, scatters it into the sidecar buffer, and returns a
        manifest cached on the request for later (per-step) verification.
        """
        if not self.config.enable_page_tags or self.table is None:
            return None
        pages_t = torch.tensor(
            list(page_physical_ids), dtype=torch.long, device=self.device
        )
        generations = self.table.generation_of(pages_t).tolist()
        manifest = PageManifest.from_tokens(
            token_ids,
            self.page_size,
            bootstrap_room,
            list(page_physical_ids),
            generations,
        )
        manifest.physical_page_ids_t = pages_t[: manifest.num_pages]
        manifest.generations_t = manifest.generations_t.to(device=self.device)
        manifest.expected_tags_t = manifest.expected_tags_tensor(self.device)
        self.table.write_tags(
            manifest.physical_page_ids_t,
            manifest.expected_tags_t,
        )
        return manifest

    def verify_request(
        self,
        manifest: Optional[PageManifest],
        *,
        rid: Optional[str] = None,
    ) -> Optional[KVPageTagMismatch]:
        """Verify one request's pages; return an exception object on mismatch.

        Returns ``None`` when protection is disabled or all pages match.  Only a
        single ``.any()`` host sync occurs on the happy path.
        """
        if not self.config.enable_page_tags or manifest is None or self.table is None:
            return None
        if manifest.num_pages == 0:
            return None
        pages = manifest.physical_pages_tensor(self.device)
        expected = manifest.expected_tags_tensor(self.device)
        generations = manifest.generations_tensor(self.device)
        ok, mismatch = verify_page_tags(self.table, pages, expected, generations)
        if self.metrics is not None:
            self.metrics.increment_kv_page_tag_checked_pages(int(pages.numel()))
        if ok:
            return None
        # Rare path: extract the first offending page for diagnostics.
        bad = int(torch.nonzero(mismatch).reshape(-1)[0].item())
        expected_tag = int(expected[bad].item())
        actual_tag = int(self.table.read_tags(pages[bad : bad + 1])[0].item())
        exc = KVPageTagMismatch(
            rid=rid,
            bootstrap_room=manifest.bootstrap_room,
            page_id=int(pages[bad].item()),
            page_position=bad,
            expected_tag=expected_tag,
            actual_tag=actual_tag,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_page_tag_mismatches()
        return exc

    def refresh_tail_token(
        self,
        manifest: Optional[PageManifest],
        *,
        logical_pos: int,
        token_id: int,
        physical_page_id: torch.Tensor,
    ) -> None:
        """Refresh the one logical page changed by a decode append.

        ``physical_page_id`` is a one-element device tensor derived from
        ``batch.out_cache_loc``.  Keeping it as a tensor avoids GPU-to-CPU scalar
        syncs in the decode hot path.
        """
        if not self.config.enable_page_tags or manifest is None or self.table is None:
            return
        physical_page_id = physical_page_id.reshape(1).to(
            device=self.device, dtype=torch.long
        )
        generation = self.table.generation_of(physical_page_id)
        page_id_t, expected_t = manifest.refresh_token_tensor(
            logical_pos=logical_pos,
            token_id=token_id,
            physical_page_id=physical_page_id,
            generation=generation,
        )
        self.table.write_tags(page_id_t, expected_t)

    def verify_batch(
        self,
        items: Sequence[Tuple[str, PageManifest]],
    ) -> List[KVPageTagMismatch]:
        """Vectorized verification across a whole decode batch.

        ``items`` is a sequence of ``(rid, manifest)``.  All pages and expected
        tag/generation tensors are concatenated and verified with vectorized
        gathers+compares; per-req diagnostics are produced only for the requests
        that actually mismatched.
        Returns the list of mismatches (empty when all pass).  Metrics are
        incremented for checked pages and per mismatch.
        """
        if not self.config.enable_page_tags or self.table is None:
            return []
        page_tensors: List[torch.Tensor] = []
        expected_tensors: List[torch.Tensor] = []
        generation_tensors: List[torch.Tensor] = []
        owners: List[Tuple[str, PageManifest]] = []
        offsets: List[int] = [0]
        for rid, manifest in items:
            if manifest is None:
                continue
            pages = manifest.physical_pages_tensor(self.device)
            if pages.numel() == 0:
                continue
            expected = manifest.expected_tags_tensor(self.device)
            generations = manifest.generations_tensor(self.device)
            page_tensors.append(pages)
            expected_tensors.append(expected)
            generation_tensors.append(generations)
            owners.append((rid, manifest))
            offsets.append(offsets[-1] + int(pages.numel()))
        if not page_tensors:
            return []
        pages_t = torch.cat(page_tensors)
        expected_t = torch.cat(expected_tensors)
        generations_t = torch.cat(generation_tensors)
        ok, mismatch = verify_page_tags(self.table, pages_t, expected_t, generations_t)
        if self.metrics is not None:
            self.metrics.increment_kv_page_tag_checked_pages(int(pages_t.numel()))
        if ok:
            return []
        actual_all = self.table.read_tags(pages_t)
        bad_idx = torch.nonzero(mismatch).reshape(-1).cpu().tolist()
        # Report at most one mismatch per request (the first offending page).
        seen_rids = set()
        result: List[KVPageTagMismatch] = []
        for i in bad_idx:
            owner_idx = bisect_right(offsets, int(i)) - 1
            rid, manifest = owners[owner_idx]
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            p = int(i) - offsets[owner_idx]
            result.append(
                KVPageTagMismatch(
                    rid=rid,
                    bootstrap_room=manifest.bootstrap_room,
                    page_id=int(pages_t[i].item()),
                    page_position=p,
                    expected_tag=int(expected_t[i].item()),
                    actual_tag=int(actual_all[i].item()),
                )
            )
        if self.metrics is not None:
            self.metrics.increment_kv_page_tag_mismatches(len(result))
        return result

    # -- transfer checksums ------------------------------------------------

    def compute_source_checksum(
        self,
        rows: torch.Tensor,
        *,
        bootstrap_room: int,
        num_tokens: int,
    ) -> Optional[ChecksumPlan]:
        """Prefill side: hash source KV bytes in logical order."""
        if not self.config.checksum_enabled:
            return None
        return compute_transfer_checksum(
            rows,
            bootstrap_room=bootstrap_room,
            num_tokens=num_tokens,
            mode=self.config.checksum_mode,
            config=self.config,
        )

    def _checksum_from_loc(
        self,
        kv_pool: object,
        kv_loc: torch.Tensor,
        *,
        bootstrap_room: int,
        num_tokens: int,
        mode: ChecksumMode,
    ) -> int:
        """Gather only the sampled logical-token rows and hash them."""
        indices = select_checksum_token_indices(
            num_tokens, bootstrap_room, mode, self.config.checksum_sample_rate
        )
        if indices.numel() == 0:
            return _splitmix64_scalar(_CKSUM_SEED)
        rows = gather_logical_kv_rows(kv_pool, kv_loc, indices)
        row_nbytes = (
            rows.contiguous().view(torch.uint8).reshape(rows.shape[0], -1).shape[1]
            if rows.numel()
            else 0
        )
        num_lanes = select_checksum_byte_count(
            row_nbytes, mode, self.config.checksum_partial_byte_rate
        )
        return hash_rows_with_positions(rows, positions=indices, num_lanes=num_lanes)

    def compute_source_checksum_from_loc(
        self,
        kv_pool: object,
        kv_loc: torch.Tensor,
        *,
        bootstrap_room: int,
        num_tokens: int,
    ) -> Optional[ChecksumPlan]:
        """Prefill side: gather + hash source KV bytes (logical order)."""
        if not self.config.checksum_enabled:
            return None
        mode = self.config.checksum_mode
        checksum = self._checksum_from_loc(
            kv_pool,
            kv_loc,
            bootstrap_room=bootstrap_room,
            num_tokens=num_tokens,
            mode=mode,
        )
        indices = select_checksum_token_indices(
            num_tokens, bootstrap_room, mode, self.config.checksum_sample_rate
        )
        return ChecksumPlan(
            bootstrap_room=bootstrap_room,
            num_tokens=num_tokens,
            mode=mode,
            num_lanes=None,
            checksum=checksum,
        )

    def verify_destination_checksum_from_loc(
        self,
        kv_pool: object,
        kv_loc: torch.Tensor,
        *,
        bootstrap_room: int,
        num_tokens: int,
        expected: Optional[ChecksumPlan],
        rid: Optional[str] = None,
    ) -> Optional[KVChecksumError]:
        """Decode side: gather + hash destination KV bytes and compare."""
        if not self.config.checksum_enabled or expected is None:
            return None
        actual = self._checksum_from_loc(
            kv_pool,
            kv_loc,
            bootstrap_room=bootstrap_room,
            num_tokens=num_tokens,
            mode=expected.mode,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_checked_pages(int(num_tokens))
        if compare_checksums(expected, actual):
            return None
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_mismatches()
        return KVChecksumError(
            rid=rid,
            bootstrap_room=bootstrap_room,
            expected_checksum=expected.checksum,
            actual_checksum=actual,
            num_checked_tokens=num_tokens,
        )

    def verify_destination_checksum(
        self,
        rows: torch.Tensor,
        *,
        bootstrap_room: int,
        num_tokens: int,
        expected: Optional[ChecksumPlan],
        rid: Optional[str] = None,
    ) -> Optional[KVChecksumError]:
        """Decode side: hash destination KV bytes (logical order) and compare."""
        if not self.config.checksum_enabled or expected is None:
            return None
        actual = compute_transfer_checksum(
            rows,
            bootstrap_room=bootstrap_room,
            num_tokens=num_tokens,
            mode=expected.mode,
            config=self.config,
            row_nbytes=None,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_checked_pages(
                int(actual.num_tokens)
            )
        if compare_checksums(expected, actual.checksum):
            return None
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_mismatches()
        return KVChecksumError(
            rid=rid,
            bootstrap_room=bootstrap_room,
            expected_checksum=expected.checksum,
            actual_checksum=actual.checksum,
            num_checked_tokens=actual.num_tokens,
        )
