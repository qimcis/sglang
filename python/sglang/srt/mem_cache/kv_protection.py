"""
KV Attention Tags + Transfer Checksums for PD Disaggregation.

This module protects PD (prefill/decode) disaggregated decoding from using
stale, wrong, or mid-decode KV pages, and optionally proves that the KV bytes
copied across the network (e.g. by Mooncake) are byte-for-byte correct.

Two independent (but related) mechanisms live here:

1. Attention tags
   A sidecar GPU buffer of ``uint64`` tags, one per physical KV page, stored
   separately from the KV tensors themselves. A tag identifies one allocation
   generation of a physical page and remains stable while RadixCache shares it:

       tag = hash(physical_page_id, generation)

   where ``generation`` is a per-physical-page allocation generation that is
   bumped every time the page is (re)allocated. At the decode pre-attention
   boundary an independent request/logical-page sidecar first validates the
   active physical mapping, then compares its expected identity and generation
   against the physical sidecar. A mismatch means the mapping or allocation
   changed since transfer commit -> we fail *only* the affected request with
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

import heapq
import json
import logging
import os
import struct
import time
import uuid
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
KV_ATTENTION_TAG_BYTES_PER_PAGE = 20
KV_EXPECTED_MAPPING_BYTES_PER_PAGE = 24
KV_REQUEST_PROTECTION_BYTES_PER_SLOT = 16
TRANSFER_PAGE_TAG_SKIP = 0
KV_EXPECTED_MAPPING_FULL = 0
KV_EXPECTED_MAPPING_SWA = 1
KV_EXPECTED_MAPPING_NAMESPACE_COUNT = 2
KV_CHECKSUM_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
KV_PAGE_INVALID_MAPPING = 1 << 0
KV_PAGE_OWNER_MISMATCH = 1 << 1
KV_PAGE_POSITION_MISMATCH = 1 << 2
KV_PAGE_ATTENTION_TAG_MISMATCH = 1 << 3
KV_PAGE_GENERATION_MISMATCH = 1 << 4
KV_PAGE_TRANSFER_TAG_MISMATCH = 1 << 5
KV_PAGE_VALIDATION_INCOMPLETE = 1 << 30
KV_PAGE_VALIDATION_REMOTE_FAILURE = 1 << 29


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KVPageProtectionError(Exception):
    """Base class for KV page protection failures."""


class KVFusedProtectionError(KVPageProtectionError):
    """Raised after fused validation fails and before token publication."""

    def __init__(
        self,
        *,
        batch_indices: Sequence[int],
        request_pool_indices: Sequence[int],
        statuses: Sequence[int],
    ):
        self.batch_indices = tuple(int(index) for index in batch_indices)
        self.request_pool_indices = tuple(int(index) for index in request_pool_indices)
        self.statuses = tuple(int(status) for status in statuses)
        super().__init__(
            "Fused KV page validation failed before token publication "
            f"(batch_indices={self.batch_indices}, statuses={self.statuses})"
        )


class KVProtectionBookkeepingError(KVPageProtectionError):
    """Raised when protection metadata cannot be committed safely."""

    def __init__(
        self,
        *,
        rid: Optional[str] = None,
        bootstrap_room: Optional[int] = None,
        cause: str = "bookkeeping_error",
        detail: Optional[str] = None,
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.cause = cause
        self.detail = detail
        self.incident = None
        super().__init__(
            f"KV protection bookkeeping failed (rid={rid}, "
            f"bootstrap_room={bootstrap_room}, cause={cause}, detail={detail})"
        )


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
        expected_generation: Optional[int] = None,
        actual_generation: Optional[int] = None,
        cause: str = "tag_value",
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.page_id = page_id
        self.page_position = page_position
        self.expected_tag = expected_tag
        self.actual_tag = actual_tag
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        self.cause = cause
        self.incident = None
        super().__init__(
            f"KV attention tag mismatch (rid={rid}, bootstrap_room={bootstrap_room}, "
            f"page_id={page_id}, page_position={page_position}, "
            f"expected_tag={_u64(expected_tag)}, actual_tag={_u64(actual_tag)}, "
            f"expected_generation={expected_generation}, actual_generation={actual_generation}, "
            f"cause={cause})"
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
        page_position: Optional[int] = None,
        expected_page_digest: Optional[int] = None,
        actual_page_digest: Optional[int] = None,
        cause: str = "digest_mismatch",
        detail: Optional[str] = None,
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.num_checked_tokens = num_checked_tokens
        self.page_position = page_position
        self.expected_page_digest = expected_page_digest
        self.actual_page_digest = actual_page_digest
        self.cause = cause
        self.detail = detail
        self.incident = None
        super().__init__(
            f"KV transfer checksum mismatch (rid={rid}, "
            f"bootstrap_room={bootstrap_room}, "
            f"expected={_u32(expected_checksum)}, actual={_u32(actual_checksum)}, "
            f"num_checked_tokens={num_checked_tokens}, page_position={page_position}, "
            f"expected_page_digest={_u64(expected_page_digest)}, "
            f"actual_page_digest={_u64(actual_page_digest)}, cause={cause}, detail={detail})"
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
        expected_generation: Optional[int] = None,
        actual_generation: Optional[int] = None,
        cause: str = "tag_value",
    ):
        self.rid = rid
        self.bootstrap_room = bootstrap_room
        self.page_id = page_id
        self.page_position = page_position
        self.expected_transfer_tag = expected_transfer_tag
        self.actual_transfer_tag = actual_transfer_tag
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        self.cause = cause
        self.incident = None
        super().__init__(
            f"KV transfer page tag mismatch (rid={rid}, bootstrap_room={bootstrap_room}, "
            f"page_id={page_id}, page_position={page_position}, "
            f"expected_transfer_tag={_u32(expected_transfer_tag)}, "
            f"actual_transfer_tag={_u32(actual_transfer_tag)}, "
            f"expected_generation={expected_generation}, actual_generation={actual_generation}, "
            f"cause={cause})"
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


@dataclass
class KVProtectionIncident:
    schema_version: int
    incident_id: str
    observed_at_ns: int
    kind: str
    cause: str
    phase: str
    rid: Optional[str]
    bootstrap_room: Optional[int]
    pod: Optional[str]
    rank: Optional[str]
    page_id: Optional[int] = None
    page_position: Optional[int] = None
    expected_tag: Optional[int] = None
    actual_tag: Optional[int] = None
    expected_generation: Optional[int] = None
    actual_generation: Optional[int] = None
    expected_checksum: Optional[int] = None
    actual_checksum: Optional[int] = None
    expected_page_digest: Optional[int] = None
    actual_page_digest: Optional[int] = None
    num_checked_tokens: Optional[int] = None
    detail: Optional[str] = None
    history: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and key != "history"
        }
        payload["history"] = list(self.history)
        for key in (
            "expected_tag",
            "actual_tag",
            "expected_page_digest",
            "actual_page_digest",
        ):
            if payload.get(key) is not None:
                payload[key] = f"0x{int(payload[key]) & _U64_MASK:016x}"
        for key in ("expected_checksum", "actual_checksum"):
            if payload.get(key) is not None:
                payload[key] = f"0x{int(payload[key]) & _U32_MASK:08x}"
        return payload


def attach_kv_protection_incident(
    error: KVPageProtectionError,
    *,
    kind: str,
    phase: str,
    history: Sequence[Dict[str, Any]] = (),
) -> KVProtectionIncident:
    incident = KVProtectionIncident(
        schema_version=1,
        incident_id=uuid.uuid4().hex,
        observed_at_ns=time.time_ns(),
        kind=kind,
        cause=getattr(error, "cause", "unknown"),
        phase=phase,
        rid=getattr(error, "rid", None),
        bootstrap_room=getattr(error, "bootstrap_room", None),
        pod=os.getenv("HOSTNAME"),
        rank=os.getenv("RANK", os.getenv("LOCAL_RANK")),
        page_id=getattr(error, "page_id", None),
        page_position=getattr(error, "page_position", None),
        expected_tag=getattr(
            error, "expected_tag", getattr(error, "expected_transfer_tag", None)
        ),
        actual_tag=getattr(
            error, "actual_tag", getattr(error, "actual_transfer_tag", None)
        ),
        expected_generation=getattr(error, "expected_generation", None),
        actual_generation=getattr(error, "actual_generation", None),
        expected_checksum=getattr(error, "expected_checksum", None),
        actual_checksum=getattr(error, "actual_checksum", None),
        expected_page_digest=getattr(error, "expected_page_digest", None),
        actual_page_digest=getattr(error, "actual_page_digest", None),
        num_checked_tokens=getattr(error, "num_checked_tokens", None),
        detail=getattr(error, "detail", None),
        history=tuple(history),
    )
    error.incident = incident
    return incident


def emit_kv_protection_incident(
    error: KVPageProtectionError, log: logging.Logger = logger
) -> None:
    incident = getattr(error, "incident", None)
    if incident is None:
        kind = {
            KVAttentionTagMismatch: "attention_tag",
            KVTransferPageTagMismatch: "transfer_page_tag",
            KVChecksumError: "transfer_checksum",
        }.get(type(error), "unknown")
        incident = attach_kv_protection_incident(error, kind=kind, phase="unknown")
    log.error(
        "KVProtectionIncident %s",
        json.dumps(incident.to_dict(), sort_keys=True, separators=(",", ":")),
    )


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
    enable_page_history: bool = False

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

        if envs.SGLANG_ENABLE_LEGACY_KV_PROTECTION_COMPLETION.get():
            raise RuntimeError(
                "Legacy nonce-less KV protection completion is not safe and is "
                "no longer supported. Upgrade prefill workers before enabling "
                "KV page protection or transfer checksums."
            )

        return cls(
            enable_attention_tags=enable_attention_tags,
            enable_transfer_checksum=enable_transfer_checksum,
            enable_page_history=(
                enable_attention_tags and envs.SGLANG_KV_PAGE_HISTORY.get()
            ),
        )


def should_use_fused_kv_page_protection(tag_table: object, *, supported: bool) -> bool:
    """Select the only supported attention-tag validation path."""
    return tag_table is not None and supported


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
SUPPORTED_ATTENTION_TAG_BACKENDS = ("mooncake",)


def assert_protection_supported(
    config: KVProtectionConfig,
    *,
    allocator: object = None,
    transfer_backend: Optional[str] = None,
    is_spec_decode: bool = False,
    supports_spec_target_verify: bool = False,
    pp_size: int = 1,
    enable_dp_attention: bool = False,
    is_cuda_device: Optional[bool] = None,
    device_capability_major: Optional[int] = None,
    device_capability_minor: Optional[int] = None,
    prefix_cache: object = None,
) -> None:
    """Fail-fast when protection is enabled on an unsupported configuration.

    Rather than silently disabling protection (which would falsely claim
    success), raise a clear ``RuntimeError`` so the operator can either turn the
    feature off or run a supported layout.
    """
    if not config.enabled:
        return

    if (
        config.enable_attention_tags
        and prefix_cache is not None
        and not prefix_cache.supports_kv_page_protection()
    ):
        raise RuntimeError(
            "KV page protection requires a prefix cache with protected mapping "
            f"lifecycle support; {type(prefix_cache).__name__} is unsupported"
        )

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

    if (
        is_spec_decode
        and config.enable_attention_tags
        and not supports_spec_target_verify
    ):
        raise RuntimeError(
            "KV attention-tag protection requires a speculative target backend "
            "with both pre-indexer and protected top-k validation. Disable "
            "SGLANG_KV_PAGE_PROTECTION or use audited DSA target verification."
        )

    if config.enable_attention_tags and pp_size > 1:
        raise RuntimeError(
            "KV attention-tag protection does not yet support pipeline "
            "parallelism. Set --pp-size 1 or disable SGLANG_KV_PAGE_PROTECTION."
        )

    if config.enable_attention_tags and enable_dp_attention:
        raise RuntimeError(
            "KV attention-tag protection does not yet support DP attention. "
            "Disable --enable-dp-attention or SGLANG_KV_PAGE_PROTECTION."
        )

    if config.enable_attention_tags and transfer_backend is not None:
        backend = str(transfer_backend).lower()
        if backend not in SUPPORTED_ATTENTION_TAG_BACKENDS:
            raise RuntimeError(
                "KV attention tags are enabled but transfer backend "
                f"{transfer_backend!r} does not transport page-tag metadata. "
                f"Supported backends: {SUPPORTED_ATTENTION_TAG_BACKENDS}. "
                "Set SGLANG_KV_PAGE_PROTECTION=0 or use a supported backend."
            )

    if config.enable_attention_tags:
        from sglang.srt.environ import envs

        if envs.SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION.get():
            raise RuntimeError(
                "This GLM-5.2 protection build requires fused DSA validation. "
                "Scheduler-only validation is not supported. Unset "
                "SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION or disable "
                "SGLANG_KV_PAGE_PROTECTION."
            )
        capability = (
            (device_capability_major, device_capability_minor)
            if device_capability_major is not None
            and device_capability_minor is not None
            else (
                (device_capability_major, -1)
                if device_capability_major is not None
                else None
            )
        )
        if is_cuda_device is False or (
            capability is not None and capability not in ((9, 0), (10, 0), (10, 3))
        ):
            raise RuntimeError(
                "Fused KV page protection requires an NVIDIA Hopper SM90 or "
                "Blackwell SM100/SM103 GPU. Disable SGLANG_KV_PAGE_PROTECTION "
                "on unsupported hardware."
            )

    if config.checksum_enabled and is_cuda_device is False:
        raise RuntimeError(
            "KV transfer checksums require the NVIDIA CUDA checksum kernel. "
            "Set SGLANG_KV_TRANSFER_CHECKSUM=0 or run on CUDA."
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

TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE = 64
_CHECKSUM_MANIFEST_MAGIC = b"SGLKVDG1"
_CHECKSUM_MANIFEST_VERSION = 1
_CHECKSUM_MANIFEST_ALGORITHM = 1
_CHECKSUM_MANIFEST_DIGEST_BYTES = 8
_CHECKSUM_MANIFEST_MAX_BYTES = 8 * 1024 * 1024
_CHECKSUM_MANIFEST_HEADER = struct.Struct("<8sBBHQQIIqII")


def compute_attention_tag_scalar(
    physical_page_id: int,
    page_position: int,
    bootstrap_room: int,
    generation: int,
) -> int:
    """Reference attention ownership tag hash; returns a uint64.

    This intentionally does not hash token ids or KV bytes.  Transfer checksums
    prove byte equality; attention tags prove that the physical page attention is
    about to read is still the expected allocation generation.
    """
    del page_position, bootstrap_room
    acc = _TAG_SEED
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
    del page_positions, bootstrap_rooms
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
    # manifest entry -> logical page position.  This is usually 0..N-1, but SWA
    # tail manifests can cover only the live sliding-window pages.
    page_positions: List[int]
    # Cached device tensors consumed by the decode verification hot path.
    physical_page_ids_t: torch.Tensor
    page_positions_t: torch.Tensor
    generations_t: torch.Tensor
    expected_tags_t: torch.Tensor
    mapping_namespace: int = KV_EXPECTED_MAPPING_FULL
    revision: int = 0
    max_num_pages: Optional[int] = None
    _replacement_heap: List[Tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_num_pages is None:
            return
        if self.max_num_pages <= 0 or self.num_pages > self.max_num_pages:
            raise ValueError("invalid bounded attention tag manifest size")
        self._replacement_heap = [
            (int(page_position), index)
            for index, page_position in enumerate(self.page_positions)
        ]
        heapq.heapify(self._replacement_heap)

    @classmethod
    def from_pages(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: Sequence[int],
        generations: Sequence[int],
        page_positions: Optional[Sequence[int]] = None,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
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
        if max_num_pages is not None:
            max_num_pages = int(max_num_pages)
            if max_num_pages <= 0:
                raise ValueError("max_num_pages must be positive")
            first = max(0, len(phys) - max_num_pages)
            phys = phys[first:]
            gens = gens[first:]
            positions = positions[first:]
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
            page_positions=positions,
            physical_page_ids_t=physical_page_ids_t,
            page_positions_t=page_positions_t,
            generations_t=generations_t,
            expected_tags_t=expected_tags_t,
            mapping_namespace=mapping_namespace,
            max_num_pages=max_num_pages,
        )

    @classmethod
    def from_tensors(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: torch.Tensor,
        generations: torch.Tensor,
        page_positions: Optional[Sequence[int] | torch.Tensor] = None,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
    ) -> AttentionTagManifest:
        pages_t = physical_page_ids.reshape(-1).to(dtype=torch.long)
        generations_t = generations.reshape(-1).to(
            device=pages_t.device, dtype=TAG_DTYPE
        )
        num_pages = int(pages_t.numel())
        if page_positions is None:
            positions = list(range(num_pages))
            positions_t = torch.arange(
                num_pages, dtype=TAG_DTYPE, device=pages_t.device
            )
        elif isinstance(page_positions, torch.Tensor):
            positions_t = page_positions.to(
                device=pages_t.device, dtype=TAG_DTYPE
            ).reshape(-1)
            positions = []
        else:
            positions = [int(x) for x in page_positions]
            positions_t = torch.tensor(
                positions, dtype=TAG_DTYPE, device=pages_t.device
            )
        if generations_t.numel() != num_pages or positions_t.numel() != num_pages:
            raise RuntimeError("attention tag tensor metadata length mismatch")
        if max_num_pages is not None:
            max_num_pages = int(max_num_pages)
            if max_num_pages <= 0:
                raise ValueError("max_num_pages must be positive")
            first = max(0, num_pages - max_num_pages)
            if first:
                pages_t = pages_t[first:].clone()
                generations_t = generations_t[first:].clone()
                positions_t = positions_t[first:].clone()
                positions = positions[first:] if positions else []
                num_pages -= first
        expected_tags_t = compute_attention_tags_tensor(
            pages_t,
            positions_t,
            torch.full_like(positions_t, bootstrap_room),
            generations_t,
        )
        if not positions and num_pages:
            positions = [int(x) for x in positions_t.detach().cpu().tolist()]
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            page_positions=positions,
            physical_page_ids_t=pages_t,
            page_positions_t=positions_t,
            generations_t=generations_t,
            expected_tags_t=expected_tags_t,
            mapping_namespace=mapping_namespace,
            max_num_pages=max_num_pages,
        )

    @property
    def num_pages(self) -> int:
        return int(self.physical_page_ids_t.numel())

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

    def _entry_index_for_new_page(self, page_position: int) -> int:
        if self.max_num_pages is not None and self.num_pages >= self.max_num_pages:
            _, entry_index = heapq.heappop(self._replacement_heap)
            self.page_positions[entry_index] = page_position
        else:
            entry_index = len(self.page_positions)
            self.page_positions.append(page_position)
        if self.max_num_pages is not None:
            heapq.heappush(self._replacement_heap, (page_position, entry_index))
        return entry_index

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
            entry_index = self._entry_index_for_new_page(page_position)

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

        self.revision += 1
        return tag_page_id, expected_t

    def replace_pages(
        self, physical_page_ids: torch.Tensor, generations: torch.Tensor
    ) -> None:
        """Replace this request's logical mapping after a radix dedup/remap."""
        pages = physical_page_ids.reshape(-1).to(
            device=self.physical_page_ids_t.device, dtype=torch.long
        )
        generations = generations.reshape(-1).to(device=pages.device, dtype=TAG_DTYPE)
        if pages.numel() != self.num_pages or generations.numel() != self.num_pages:
            raise RuntimeError("attention tag remap length mismatch")
        self.physical_page_ids_t.copy_(pages)
        self.generations_t.copy_(generations)
        self.expected_tags_t.copy_(
            compute_attention_tags_tensor(
                pages,
                self.page_positions_t.to(pages.device),
                torch.full_like(self.page_positions_t, self.bootstrap_room).to(
                    pages.device
                ),
                generations,
            )
        )
        self.revision += 1


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
    page_positions: List[int]
    physical_page_ids_t: torch.Tensor
    page_positions_t: torch.Tensor
    generations_t: torch.Tensor
    expected_tags_t: torch.Tensor
    mapping_namespace: int = KV_EXPECTED_MAPPING_FULL
    revision: int = 0
    max_num_pages: Optional[int] = None
    _replacement_heap: List[Tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_num_pages is None:
            return
        if self.max_num_pages <= 0 or self.num_pages > self.max_num_pages:
            raise ValueError("invalid bounded transfer page tag manifest size")
        self._replacement_heap = [
            (int(page_position), index)
            for index, page_position in enumerate(self.page_positions)
        ]
        heapq.heapify(self._replacement_heap)

    @classmethod
    def from_pages(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: Sequence[int],
        generations: Sequence[int],
        page_positions: Optional[Sequence[int]] = None,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
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
        if max_num_pages is not None:
            max_num_pages = int(max_num_pages)
            if max_num_pages <= 0:
                raise ValueError("max_num_pages must be positive")
            first = max(0, len(phys) - max_num_pages)
            phys = phys[first:]
            gens = gens[first:]
            positions = positions[first:]
        expected = [
            compute_transfer_page_tag_scalar(
                phys[i], positions[i], bootstrap_room, gens[i]
            )
            for i in range(len(phys))
        ]
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            page_positions=positions,
            physical_page_ids_t=torch.tensor(phys, dtype=torch.long),
            page_positions_t=torch.tensor(positions, dtype=TAG_DTYPE),
            generations_t=torch.tensor(gens, dtype=TAG_DTYPE),
            expected_tags_t=transfer_page_tags_to_tensor(expected),
            mapping_namespace=mapping_namespace,
            max_num_pages=max_num_pages,
        )

    @classmethod
    def from_tensors(
        cls,
        page_size: int,
        bootstrap_room: int,
        physical_page_ids: torch.Tensor,
        generations: torch.Tensor,
        page_positions: Optional[Sequence[int] | torch.Tensor] = None,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
    ) -> TransferPageTagManifest:
        pages_t = physical_page_ids.reshape(-1).to(dtype=torch.long)
        generations_t = generations.reshape(-1).to(
            device=pages_t.device, dtype=TAG_DTYPE
        )
        num_pages = int(pages_t.numel())
        if page_positions is None:
            positions = list(range(num_pages))
            positions_t = torch.arange(
                num_pages, dtype=TAG_DTYPE, device=pages_t.device
            )
        elif isinstance(page_positions, torch.Tensor):
            positions_t = page_positions.to(
                device=pages_t.device, dtype=TAG_DTYPE
            ).reshape(-1)
            positions = []
        else:
            positions = [int(x) for x in page_positions]
            positions_t = torch.tensor(
                positions, dtype=TAG_DTYPE, device=pages_t.device
            )
        if generations_t.numel() != num_pages or positions_t.numel() != num_pages:
            raise RuntimeError("transfer page tag tensor metadata length mismatch")
        if max_num_pages is not None:
            max_num_pages = int(max_num_pages)
            if max_num_pages <= 0:
                raise ValueError("max_num_pages must be positive")
            first = max(0, num_pages - max_num_pages)
            if first:
                pages_t = pages_t[first:].clone()
                generations_t = generations_t[first:].clone()
                positions_t = positions_t[first:].clone()
                positions = positions[first:] if positions else []
                num_pages -= first
        expected_tags_t = compute_transfer_page_tags_tensor(
            pages_t,
            positions_t,
            torch.full_like(positions_t, bootstrap_room),
            generations_t,
        )
        if not positions and num_pages:
            positions = [int(x) for x in positions_t.detach().cpu().tolist()]
        return cls(
            bootstrap_room=bootstrap_room,
            page_size=page_size,
            page_positions=positions,
            physical_page_ids_t=pages_t,
            page_positions_t=positions_t,
            generations_t=generations_t,
            expected_tags_t=expected_tags_t,
            mapping_namespace=mapping_namespace,
            max_num_pages=max_num_pages,
        )

    @property
    def num_pages(self) -> int:
        return int(self.physical_page_ids_t.numel())

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

    def _entry_index_for_new_page(self, page_position: int) -> int:
        if self.max_num_pages is not None and self.num_pages >= self.max_num_pages:
            _, entry_index = heapq.heappop(self._replacement_heap)
            self.page_positions[entry_index] = page_position
        else:
            entry_index = len(self.page_positions)
            self.page_positions.append(page_position)
        if self.max_num_pages is not None:
            heapq.heappush(self._replacement_heap, (page_position, entry_index))
        return entry_index

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
            entry_index = self._entry_index_for_new_page(page_position)

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

        self.revision += 1
        return tag_page_id, expected_t

    def replace_pages(
        self,
        physical_page_ids: torch.Tensor,
        generations: torch.Tensor,
        *,
        skip_changed: bool,
    ) -> None:
        """Replace remapped pages and skip transfer checks for shared pages."""
        pages = physical_page_ids.reshape(-1).to(
            device=self.physical_page_ids_t.device, dtype=torch.long
        )
        generations = generations.reshape(-1).to(device=pages.device, dtype=TAG_DTYPE)
        if pages.numel() != self.num_pages or generations.numel() != self.num_pages:
            raise RuntimeError("transfer tag remap length mismatch")
        changed = pages.ne(self.physical_page_ids_t)
        self.physical_page_ids_t.copy_(pages)
        self.generations_t.copy_(generations)
        if skip_changed:
            self.expected_tags_t.masked_fill_(changed, TRANSFER_PAGE_TAG_SKIP)
        self.revision += 1


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


class KVPageHistory:
    """Small device-resident operation ring, materialized only after a mismatch."""

    DEPTH = 8
    BYTES_PER_PAGE = 8 + DEPTH * 5 * 8
    ALLOC = 1
    FREE = 2
    FREE_DEFERRED = 3
    FREE_RELEASED = 4
    TRANSFER_EXPECTED = 5
    TRANSFER_WRITE = 6
    TAG_REFRESH = 7
    _OP_NAMES = {
        ALLOC: "alloc",
        FREE: "free",
        FREE_DEFERRED: "free_deferred",
        FREE_RELEASED: "free_released",
        TRANSFER_EXPECTED: "transfer_expected",
        TRANSFER_WRITE: "transfer_write",
        TAG_REFRESH: "tag_refresh",
    }

    def __init__(self, size: int, device: str):
        shape = (size, self.DEPTH)
        self.device = device
        self.cursor = torch.zeros(size, dtype=TAG_DTYPE, device=device)
        self.records = torch.zeros((*shape, 5), dtype=TAG_DTYPE, device=device)
        self.records[:, :, 3].fill_(-1)
        self.operations = self.records[:, :, 0]
        self.generations = self.records[:, :, 1]
        self.bootstrap_rooms = self.records[:, :, 2]
        self.page_positions = self.records[:, :, 3]
        self.values = self.records[:, :, 4]

    @staticmethod
    def _field_tensor(value, page_ids: torch.Tensor, default: int) -> torch.Tensor:
        if value is None:
            return torch.full_like(page_ids, default, dtype=TAG_DTYPE)
        value_t = torch.as_tensor(
            value, dtype=TAG_DTYPE, device=page_ids.device
        ).reshape(-1)
        if value_t.numel() == 1 and page_ids.numel() != 1:
            value_t = value_t.expand(page_ids.numel())
        if value_t.numel() != page_ids.numel():
            raise RuntimeError("KV page history metadata length mismatch")
        return value_t

    def record(
        self,
        page_ids,
        operation: int,
        *,
        generations=None,
        bootstrap_rooms=None,
        page_positions=None,
        values=None,
        generations_by_page: bool = False,
    ) -> None:
        page_ids_t = torch.as_tensor(
            page_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if page_ids_t.numel() == 0:
            return
        if page_ids_t.is_cuda:
            if generations is None:
                raise RuntimeError("CUDA KV page history requires generations")
            if not isinstance(bootstrap_rooms, (int, type(None))):
                raise RuntimeError("CUDA KV page history requires a scalar room")

            def field_arg(value, default):
                if value is None:
                    return page_ids_t[:0], default
                if isinstance(value, int):
                    return page_ids_t[:0], int(value)
                value_t = torch.as_tensor(value, device=self.device).reshape(-1)
                if value_t.numel() not in (1, page_ids_t.numel()):
                    raise RuntimeError("KV page history metadata length mismatch")
                return value_t.contiguous(), default

            generations_t = torch.as_tensor(
                generations, dtype=TAG_DTYPE, device=self.device
            ).reshape(-1)
            expected_generations = (
                self.cursor.numel() if generations_by_page else page_ids_t.numel()
            )
            if generations_t.numel() != expected_generations:
                raise RuntimeError("KV page history generation length mismatch")
            page_positions_t, page_position = field_arg(page_positions, -1)
            values_t, value = field_arg(values, 0)
            from sgl_kernel.kvcacheio import kv_page_history_record

            kv_page_history_record(
                page_ids_t,
                operation,
                generations_t.contiguous(),
                generations_by_page,
                int(bootstrap_rooms or 0),
                page_positions_t,
                page_position,
                values_t,
                value,
                self.cursor,
                self.records,
            )
            return
        cursors = self.cursor.index_select(0, page_ids_t)
        slots = torch.remainder(cursors, self.DEPTH).to(torch.long)
        flat_indices = page_ids_t * self.DEPTH + slots
        if generations_by_page:
            generations_t = torch.as_tensor(
                generations, dtype=TAG_DTYPE, device=page_ids_t.device
            ).reshape(-1)
            generations_t = generations_t.index_select(0, page_ids_t)
        else:
            generations_t = generations
        fields = torch.stack(
            (
                torch.full_like(page_ids_t, int(operation), dtype=TAG_DTYPE),
                self._field_tensor(generations_t, page_ids_t, 0),
                self._field_tensor(bootstrap_rooms, page_ids_t, 0),
                self._field_tensor(page_positions, page_ids_t, -1),
                self._field_tensor(values, page_ids_t, 0),
            ),
            dim=1,
        )
        self.records.view(-1, 5).index_copy_(0, flat_indices, fields)
        self.cursor.index_add_(
            0, page_ids_t, torch.ones_like(page_ids_t, dtype=TAG_DTYPE)
        )

    def materialize(self, page_id: int) -> List[Dict[str, Any]]:
        cursor = int(self.cursor[page_id].item())
        count = min(cursor, self.DEPTH)
        start = cursor - count
        result = []
        for sequence in range(start, cursor):
            slot = sequence % self.DEPTH
            operation = int(self.operations[page_id, slot].item())
            result.append(
                {
                    "sequence": sequence,
                    "operation": self._OP_NAMES.get(operation, f"unknown_{operation}"),
                    "generation": int(self.generations[page_id, slot].item()),
                    "bootstrap_room": int(self.bootstrap_rooms[page_id, slot].item()),
                    "page_position": int(self.page_positions[page_id, slot].item()),
                    "value": f"0x{int(self.values[page_id, slot].item()) & _U64_MASK:016x}",
                }
            )
        return result


class KVAttentionTagTable:
    """Sidecar GPU buffer of per-physical-page attention tags + generations.

    Stored separately from KV tensors.  Allocated lazily and only when KV page
    protection is enabled, so the default serving path pays nothing.
    """

    def __init__(
        self,
        num_pages: int,
        device: str = "cpu",
        *,
        num_request_slots: int = 2,
        num_logical_pages: Optional[int] = None,
        enable_history: bool = False,
    ):
        # +1 so physical page ids (which are 1-based in the paged allocator) fit.
        self._size = num_pages + 1
        self.device = device
        self.num_logical_pages = int(num_logical_pages or num_pages)
        if self.num_logical_pages <= 0:
            raise ValueError("num_logical_pages must be positive")
        self.tags = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)
        self.generations = torch.zeros(self._size, dtype=TAG_DTYPE, device=device)
        self.transfer_page_tags = torch.zeros(
            self._size, dtype=TRANSFER_PAGE_TAG_DTYPE, device=device
        )
        self.expected_mapping_namespace_stride = self.num_logical_pages
        self.expected_mapping_stride = (
            KV_EXPECTED_MAPPING_NAMESPACE_COUNT * self.num_logical_pages
        )
        expected_size = num_request_slots * self.expected_mapping_stride
        self.expected_physical_pages = torch.zeros(
            expected_size, dtype=torch.int32, device=device
        )
        self.expected_tags = torch.zeros(expected_size, dtype=TAG_DTYPE, device=device)
        self.expected_generations = torch.zeros(
            expected_size, dtype=TAG_DTYPE, device=device
        )
        self.expected_transfer_page_tags = torch.zeros(
            expected_size, dtype=TRANSFER_PAGE_TAG_DTYPE, device=device
        )
        self.request_epochs = torch.zeros(
            num_request_slots, dtype=torch.int32, device=device
        )
        self.validated_epochs = torch.full(
            (num_request_slots,), -1, dtype=torch.int32, device=device
        )
        # DSA validates its full per-forward page table before the indexer, then
        # independently publishes selected-slot validation from fused top-k.
        self.pre_indexer_validated_epochs = torch.full(
            (num_request_slots,), -1, dtype=torch.int32, device=device
        )
        self.validation_status = torch.zeros(
            num_request_slots, dtype=torch.int32, device=device
        )
        self.history = KVPageHistory(self._size, device) if enable_history else None
        self._history_warning_emitted = False

    @property
    def size(self) -> int:
        return self._size

    def _record_history(self, *args, **kwargs) -> None:
        if self.history is None:
            return
        try:
            self.history.record(*args, **kwargs)
        except Exception:
            if not self._history_warning_emitted:
                logger.warning(
                    "KV page history recording failed; protection remains active",
                    exc_info=True,
                )
                self._history_warning_emitted = True

    def materialize_history(self, page_id: int) -> List[Dict[str, Any]]:
        if self.history is None:
            return []
        try:
            return self.history.materialize(page_id)
        except Exception:
            logger.warning(
                "KV page history materialization failed for page_id=%s",
                page_id,
                exc_info=True,
            )
            return []

    def bump_generations(self, page_ids: torch.Tensor) -> None:
        """Increment the allocation generation of the given physical pages.

        Called only on *newly allocated* physical pages, so reused pages get a
        fresh generation and any stale expected tag from a prior owner will no
        longer match.
        """
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.generations.device, dtype=torch.long).reshape(-1)
        self.generations.index_add_(
            0, page_ids, torch.ones_like(page_ids, dtype=TAG_DTYPE)
        )
        generations = self.generations.index_select(0, page_ids)
        self.tags.index_copy_(
            0,
            page_ids,
            compute_attention_tags_tensor(
                page_ids,
                torch.zeros_like(page_ids),
                torch.zeros_like(page_ids),
                generations,
            ),
        )
        history_pages = torch.unique(page_ids)
        self._record_history(
            history_pages,
            KVPageHistory.ALLOC,
            generations=self.generations,
            generations_by_page=True,
        )

    def generation_of(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.generations.device, dtype=torch.long).reshape(-1)
        return self.generations.index_select(0, page_ids)

    def write_tags(self, page_ids: torch.Tensor, tags: torch.Tensor) -> None:
        """Scatter attention ownership tags into the sidecar buffer."""
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(self.tags.device, dtype=torch.long).reshape(-1)
        tags = tags.to(self.tags.device, dtype=TAG_DTYPE).reshape(-1)
        self.tags.index_copy_(0, page_ids, tags)

    def write_expected_owner(
        self,
        page_ids: torch.Tensor,
        *,
        request_pool_idx: int,
        page_positions: torch.Tensor,
        generations: torch.Tensor,
        attention_tags: Optional[torch.Tensor] = None,
        transfer_page_tags: Optional[torch.Tensor] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
    ) -> None:
        """Publish request-logical expectations consumed by fused kernels."""
        if page_ids.numel() == 0:
            return
        if request_pool_idx <= 0 or request_pool_idx >= self.request_epochs.numel():
            raise RuntimeError("protected DSA request-pool index is out of bounds")
        page_ids = page_ids.to(
            self.expected_physical_pages.device, dtype=torch.long
        ).reshape(-1)
        count = page_ids.numel()
        positions = page_positions.to(
            self.expected_physical_pages.device, dtype=torch.long
        ).reshape(-1)
        expected_generations = generations.to(
            self.expected_generations.device, dtype=TAG_DTYPE
        ).reshape(-1)
        if positions.numel() != count or expected_generations.numel() != count:
            raise RuntimeError("protected DSA owner metadata length mismatch")
        if not 0 <= mapping_namespace < KV_EXPECTED_MAPPING_NAMESPACE_COUNT:
            raise RuntimeError("protected mapping namespace is out of bounds")
        if not positions.is_cuda and (
            bool((positions < 0).any().item())
            or bool((positions >= self.num_logical_pages).any().item())
        ):
            raise RuntimeError("protected logical page position is out of bounds")
        logical_indices = (
            request_pool_idx * self.expected_mapping_stride
            + mapping_namespace * self.expected_mapping_namespace_stride
            + positions
        )
        self.expected_physical_pages.index_copy_(
            0, logical_indices, page_ids.to(torch.int32)
        )
        self.expected_generations.index_copy_(0, logical_indices, expected_generations)
        if attention_tags is not None:
            self.expected_tags.index_copy_(
                0,
                logical_indices,
                attention_tags.to(self.expected_tags.device, dtype=TAG_DTYPE).reshape(
                    -1
                ),
            )
        if transfer_page_tags is not None:
            self.expected_transfer_page_tags.index_copy_(
                0,
                logical_indices,
                transfer_page_tags.to(
                    self.expected_transfer_page_tags.device,
                    dtype=TRANSFER_PAGE_TAG_DTYPE,
                ).reshape(-1),
            )

    def clear_request_slot(self, request_pool_idx: int) -> None:
        """Invalidate one graph-stable request row before its slot is reused."""
        if request_pool_idx <= 0 or request_pool_idx >= self.request_epochs.numel():
            return
        start = int(request_pool_idx) * self.expected_mapping_stride
        end = start + self.expected_mapping_stride
        self.expected_physical_pages[start:end].zero_()
        self.expected_tags[start:end].zero_()
        self.expected_generations[start:end].zero_()
        self.expected_transfer_page_tags[start:end].zero_()
        self.validation_status[request_pool_idx] = 0
        self.request_epochs[request_pool_idx] += 1
        self.validated_epochs[request_pool_idx] = -1
        self.pre_indexer_validated_epochs[request_pool_idx] = -1

    def clear_expected_owner_positions(
        self,
        request_pool_idx: int,
        page_positions: torch.Tensor,
        mapping_namespace: int,
    ) -> None:
        """Invalidate selected logical positions that left an SWA live window."""
        if page_positions.numel() == 0:
            return
        positions = page_positions.to(
            self.expected_physical_pages.device, dtype=torch.long
        ).reshape(-1)
        indices = (
            request_pool_idx * self.expected_mapping_stride
            + mapping_namespace * self.expected_mapping_namespace_stride
            + positions
        )
        self.expected_physical_pages.index_fill_(0, indices, 0)
        self.expected_tags.index_fill_(0, indices, 0)
        self.expected_generations.index_fill_(0, indices, 0)
        self.expected_transfer_page_tags.index_fill_(0, indices, 0)
        self.validated_epochs[request_pool_idx] = -1
        self.pre_indexer_validated_epochs[request_pool_idx] = -1

    def begin_fused_forward(self, request_pool_indices: torch.Tensor) -> None:
        if request_pool_indices.numel() == 0:
            return
        indices = request_pool_indices.to(
            self.request_epochs.device, dtype=torch.long
        ).reshape(-1)
        if self.request_epochs.is_cuda:
            torch.ops.sgl_kernel.kv_page_protection_begin_forward(
                indices, self.request_epochs, self.validation_status
            )
            return
        valid = indices.gt(0) & indices.lt(self.request_epochs.numel())
        safe_indices = torch.where(valid, indices, 0)
        self.validation_status.index_fill_(0, safe_indices, 0)
        self.request_epochs.index_add_(0, safe_indices, valid.to(torch.int32))

    def fused_failure_status(
        self, request_pool_indices: torch.Tensor, *, return_failed: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        indices = request_pool_indices.to(
            self.request_epochs.device, dtype=torch.long
        ).reshape(-1)
        if self.request_epochs.is_cuda:
            failure_status = torch.empty_like(indices, dtype=torch.int32)
            failed = torch.empty_like(indices, dtype=torch.int32)
            torch.ops.sgl_kernel.kv_page_protection_failure_status(
                indices,
                self.request_epochs,
                self.validated_epochs,
                self.validation_status,
                failure_status,
                failed,
            )
            return (failure_status, failed) if return_failed else failure_status
        invalid = indices.lt(0) | indices.ge(self.request_epochs.numel())
        safe_indices = torch.where(invalid, 0, indices)
        status = self.validation_status.index_select(0, safe_indices)
        request_epochs = self.request_epochs.index_select(0, safe_indices)
        validated_epochs = self.validated_epochs.index_select(0, safe_indices)
        incomplete = validated_epochs.ne(request_epochs).to(torch.int32)
        failure_status = status | (incomplete * KV_PAGE_VALIDATION_INCOMPLETE)
        failure_status = torch.where(
            invalid,
            torch.full_like(failure_status, KV_PAGE_INVALID_MAPPING),
            failure_status,
        )
        failure_status = torch.where(indices.eq(0), 0, failure_status)
        if return_failed:
            return failure_status, failure_status.ne(0).to(torch.int32)
        return failure_status

    def fused_forward_args(
        self,
        *,
        request_indices: torch.Tensor,
        seqlens: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        page_size: int,
        page_table_2: Optional[torch.Tensor] = None,
        page_table_page_offset: int = 0,
        page_table_2_page_offset: int = 0,
        page_table_2_window_size: int = 0,
        validate_full_mapping: bool = False,
        pre_indexer_cache_by_request: bool = True,
    ) -> Dict[str, object]:
        has_secondary_table = page_table_2 is not None
        return {
            "request_indices": request_indices,
            "seqlens": seqlens,
            "page_table": page_table,
            "page_table_2": page_table_2,
            "page_table_page_offset": int(page_table_page_offset),
            "page_table_2_page_offset": int(
                page_table_2_page_offset if has_secondary_table else 0
            ),
            "page_table_2_window_size": int(
                page_table_2_window_size if has_secondary_table else 0
            ),
            "page_size": int(page_size),
            "validate_full_mapping": bool(validate_full_mapping),
            "pre_indexer_cache_by_request": bool(pre_indexer_cache_by_request),
            "actual_tags": self.tags,
            "actual_generations": self.generations,
            "actual_transfer_tags": self.transfer_page_tags,
            "expected_physical_pages": self.expected_physical_pages,
            "expected_mapping_stride": self.expected_mapping_stride,
            "expected_mapping_namespace_stride": self.expected_mapping_namespace_stride,
            "page_table_expected_mapping_offset": 0,
            "page_table_2_expected_mapping_offset": (
                self.expected_mapping_namespace_stride if has_secondary_table else 0
            ),
            "expected_tags": self.expected_tags,
            "expected_generations": self.expected_generations,
            "expected_transfer_tags": self.expected_transfer_page_tags,
            "request_epochs": self.request_epochs,
            "validated_epochs": self.validated_epochs,
            "pre_indexer_validated_epochs": self.pre_indexer_validated_epochs,
            "status": self.validation_status,
        }

    def read_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(self.tags.device, dtype=torch.long).reshape(-1)
        return self.tags.index_select(0, page_ids)

    def write_transfer_page_tags(
        self, page_ids: torch.Tensor, tags: torch.Tensor
    ) -> Optional[torch.cuda.Event]:
        """Scatter transfer-written page tags into the sidecar buffer."""
        if page_ids.numel() == 0:
            return
        page_ids = page_ids.to(
            self.transfer_page_tags.device, dtype=torch.long
        ).reshape(-1)
        tags = tags.to(
            self.transfer_page_tags.device, dtype=TRANSFER_PAGE_TAG_DTYPE
        ).reshape(-1)
        self.transfer_page_tags.index_copy_(0, page_ids, tags)

    def read_transfer_page_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(
            self.transfer_page_tags.device, dtype=torch.long
        ).reshape(-1)
        return self.transfer_page_tags.index_select(0, page_ids)

    def record_free(self, page_ids, *, deferred: bool = False) -> None:
        page_ids_t = torch.as_tensor(
            page_ids, dtype=torch.long, device=self.generations.device
        ).reshape(-1)
        if page_ids_t.numel() == 0:
            return
        self.tags.index_fill_(0, page_ids_t, 0)
        self.transfer_page_tags.index_fill_(0, page_ids_t, TRANSFER_PAGE_TAG_SKIP)
        self._record_history(
            page_ids_t,
            KVPageHistory.FREE_DEFERRED if deferred else KVPageHistory.FREE,
            generations=self.generations,
            generations_by_page=True,
        )

    def record_free_released(self, page_ids) -> None:
        page_ids_t = torch.as_tensor(
            page_ids, dtype=torch.long, device=self.generations.device
        )
        self._record_history(
            page_ids_t,
            KVPageHistory.FREE_RELEASED,
            generations=self.generations,
            generations_by_page=True,
        )


def _attention_tag_mismatch_mask(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if page_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=table.device)
    actual = table.read_tags(page_ids)
    expected = expected_tags.to(table.device, dtype=TAG_DTYPE).reshape(-1)
    mismatch = actual != expected
    if expected_generations is not None:
        actual_generations = table.generation_of(page_ids)
        expected_generations = expected_generations.to(
            table.device, dtype=TAG_DTYPE
        ).reshape(-1)
        mismatch |= actual_generations != expected_generations
    return mismatch


def verify_attention_tags(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """Vectorized batch verification of attention ownership tags."""
    mismatch = _attention_tag_mismatch_mask(
        table, page_ids, expected_tags, expected_generations
    )
    all_ok = not bool(mismatch.any().item())
    return all_ok, mismatch


def _transfer_page_tag_mismatch_mask(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if page_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=table.device)
    actual = table.read_transfer_page_tags(page_ids)
    expected = expected_tags.to(table.device, dtype=TRANSFER_PAGE_TAG_DTYPE).reshape(-1)
    validate = expected != TRANSFER_PAGE_TAG_SKIP
    mismatch = validate & (actual != expected)
    if expected_generations is not None:
        actual_generations = table.generation_of(page_ids)
        expected_generations = expected_generations.to(
            table.device, dtype=TAG_DTYPE
        ).reshape(-1)
        mismatch |= validate & (actual_generations != expected_generations)
    return mismatch


def verify_transfer_page_tags(
    table: KVAttentionTagTable,
    page_ids: torch.Tensor,
    expected_tags: torch.Tensor,
    expected_generations: Optional[torch.Tensor] = None,
) -> Tuple[bool, torch.Tensor]:
    """Vectorized verification of transfer-written page tags."""
    mismatch = _transfer_page_tag_mismatch_mask(
        table, page_ids, expected_tags, expected_generations
    )
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


def swa_checksum_evicted_len(
    seq_len: int, sliding_window: Optional[int], page_size: int
) -> int:
    """Page-aligned leading tokens omitted from transferred SWA state."""
    if (
        not sliding_window
        or int(sliding_window) <= 0
        or int(seq_len) <= 0
        or int(page_size) <= 0
    ):
        return 0
    window_start = max(0, int(seq_len) - int(sliding_window))
    return (window_start // int(page_size)) * int(page_size)


@dataclass
class ChecksumPlan:
    """A request's transfer-checksum plan, exchanged prefill -> decode.

    Contains only logical, layout-independent information.  Crucially it does
    NOT contain physical page ids on either side.
    """

    bootstrap_room: int
    num_tokens: int
    checksum: int  # the prefill-side uint32 source checksum
    page_size: int = 0
    logical_start: int = 0
    page_digests: Tuple[int, ...] = ()

    def to_payload(self) -> dict:
        payload = {
            "bootstrap_room": int(self.bootstrap_room),
            "num_tokens": int(self.num_tokens),
            "checksum": int(self.checksum) & _U32_MASK,
        }
        if self.page_size > 0 or self.page_digests:
            payload.update(
                {
                    "manifest_version": _CHECKSUM_MANIFEST_VERSION,
                    "page_size": int(self.page_size),
                    "logical_start": int(self.logical_start),
                    "page_digests": [
                        int(value) & _U64_MASK for value in self.page_digests
                    ],
                }
            )
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> ChecksumPlan:
        return cls(
            bootstrap_room=int(payload["bootstrap_room"]),
            num_tokens=int(payload["num_tokens"]),
            checksum=int(payload["checksum"]) & _U32_MASK,
            page_size=int(payload.get("page_size", 0)),
            logical_start=int(payload.get("logical_start", 0)),
            page_digests=tuple(
                int(value) & _U64_MASK for value in payload.get("page_digests", ())
            ),
        )

    def to_wire_bytes(self, *, transfer_nonce: int) -> bytes:
        """Serialize a bounded, versioned manifest for Mooncake completion."""
        num_pages = len(self.page_digests)
        if self.num_tokens <= 0:
            raise ValueError("checksum page manifest requires a positive token count")
        if self.page_size != TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE:
            raise ValueError(
                "checksum page manifest requires page_size="
                f"{TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE}"
            )
        if self.logical_start < 0:
            raise ValueError("checksum page manifest requires a non-negative start")
        if not transfer_nonce:
            raise ValueError("checksum page manifest requires a transfer nonce")
        expected_pages = (
            (self.logical_start % self.page_size + self.num_tokens + self.page_size - 1)
            // self.page_size
            if self.num_tokens > 0
            else 0
        )
        if num_pages != expected_pages:
            raise ValueError(
                f"checksum page digest count mismatch: expected={expected_pages}, actual={num_pages}"
            )
        payload_size = _CHECKSUM_MANIFEST_HEADER.size + num_pages * 8
        if payload_size > _CHECKSUM_MANIFEST_MAX_BYTES:
            raise ValueError("checksum page manifest exceeds the wire size limit")
        header = _CHECKSUM_MANIFEST_HEADER.pack(
            _CHECKSUM_MANIFEST_MAGIC,
            _CHECKSUM_MANIFEST_VERSION,
            _CHECKSUM_MANIFEST_ALGORITHM,
            _CHECKSUM_MANIFEST_DIGEST_BYTES,
            int(transfer_nonce) & _U64_MASK,
            int(self.bootstrap_room) & _U64_MASK,
            int(self.num_tokens),
            int(self.page_size),
            int(self.logical_start),
            int(self.checksum) & _U32_MASK,
            num_pages,
        )
        if not num_pages:
            return header
        return header + struct.pack(
            f"<{num_pages}Q", *(int(value) & _U64_MASK for value in self.page_digests)
        )

    @classmethod
    def wire_identity(cls, payload: bytes) -> Tuple[int, int]:
        """Read the nonce and room before accepting a completion payload."""
        if (
            not isinstance(payload, bytes)
            or len(payload) < _CHECKSUM_MANIFEST_HEADER.size
        ):
            raise ValueError("checksum page manifest is truncated")
        if len(payload) > _CHECKSUM_MANIFEST_MAX_BYTES:
            raise ValueError("checksum page manifest exceeds the wire size limit")
        header = _CHECKSUM_MANIFEST_HEADER.unpack_from(payload)
        if header[0] != _CHECKSUM_MANIFEST_MAGIC:
            raise ValueError("checksum page manifest has invalid magic")
        return int(header[4]), int(header[5])

    @classmethod
    def from_wire_bytes(
        cls,
        payload: bytes,
        *,
        expected_transfer_nonce: Optional[int] = None,
        expected_bootstrap_room: Optional[int] = None,
    ) -> ChecksumPlan:
        if (
            not isinstance(payload, bytes)
            or len(payload) < _CHECKSUM_MANIFEST_HEADER.size
        ):
            raise ValueError("checksum page manifest is truncated")
        if len(payload) > _CHECKSUM_MANIFEST_MAX_BYTES:
            raise ValueError("checksum page manifest exceeds the wire size limit")
        (
            magic,
            version,
            algorithm,
            digest_bytes,
            transfer_nonce,
            bootstrap_room,
            num_tokens,
            page_size,
            logical_start,
            checksum,
            num_pages,
        ) = _CHECKSUM_MANIFEST_HEADER.unpack_from(payload)
        if magic != _CHECKSUM_MANIFEST_MAGIC:
            raise ValueError("checksum page manifest has invalid magic")
        if version != _CHECKSUM_MANIFEST_VERSION:
            raise ValueError(f"unsupported checksum page manifest version {version}")
        if algorithm != _CHECKSUM_MANIFEST_ALGORITHM or digest_bytes != 8:
            raise ValueError("unsupported checksum page digest algorithm")
        if num_tokens <= 0:
            raise ValueError("checksum page manifest has invalid token count")
        if page_size != TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE:
            raise ValueError(
                "checksum page manifest has unsupported page_size " f"{page_size}"
            )
        if logical_start < 0:
            raise ValueError("checksum page manifest has invalid logical_start")
        if transfer_nonce == 0:
            raise ValueError("checksum page manifest has invalid transfer nonce")
        expected_size = _CHECKSUM_MANIFEST_HEADER.size + num_pages * digest_bytes
        if len(payload) != expected_size:
            raise ValueError("checksum page manifest payload length mismatch")
        expected_pages = (
            (logical_start % page_size + num_tokens + page_size - 1) // page_size
            if num_tokens > 0
            else 0
        )
        if num_pages != expected_pages:
            raise ValueError("checksum page manifest digest count is inconsistent")
        if expected_transfer_nonce is not None and transfer_nonce != (
            int(expected_transfer_nonce) & _U64_MASK
        ):
            raise ValueError("checksum page manifest transfer nonce mismatch")
        if expected_bootstrap_room is not None and bootstrap_room != (
            int(expected_bootstrap_room) & _U64_MASK
        ):
            raise ValueError("checksum page manifest bootstrap room mismatch")
        page_digests = (
            struct.unpack_from(
                f"<{num_pages}Q", payload, _CHECKSUM_MANIFEST_HEADER.size
            )
            if num_pages
            else ()
        )
        return cls(
            bootstrap_room=int(bootstrap_room),
            num_tokens=int(num_tokens),
            checksum=int(checksum),
            page_size=int(page_size),
            logical_start=int(logical_start),
            page_digests=tuple(int(value) for value in page_digests),
        )


@dataclass
class _DirectKVChecksumCache:
    """Reusable CUDA tensors for direct KV checksum metadata and row output."""

    metadata_key: Optional[Tuple[int, str, int]] = None
    buffer_ptrs: Optional[torch.Tensor] = None
    row_strides: Optional[torch.Tensor] = None
    row_nbytes: Optional[torch.Tensor] = None
    buffer_num_rows: Optional[torch.Tensor] = None
    swa_buffer_flags: Optional[torch.Tensor] = None
    full_to_swa_index_mapping: Optional[torch.Tensor] = None
    total_bytes: int = 0
    full_buffer_ptrs: Optional[torch.Tensor] = None
    full_row_strides: Optional[torch.Tensor] = None
    full_row_nbytes: Optional[torch.Tensor] = None
    full_buffer_num_rows: Optional[torch.Tensor] = None
    full_swa_buffer_flags: Optional[torch.Tensor] = None
    full_total_bytes: int = 0
    swa_buffer_ptrs_only: Optional[torch.Tensor] = None
    swa_row_strides: Optional[torch.Tensor] = None
    swa_row_nbytes: Optional[torch.Tensor] = None
    swa_buffer_num_rows: Optional[torch.Tensor] = None
    swa_buffer_flags_only: Optional[torch.Tensor] = None
    swa_total_bytes: int = 0
    accum: Optional[torch.Tensor] = None
    final_out: Optional[torch.Tensor] = None
    secondary_accum: Optional[torch.Tensor] = None
    secondary_out: Optional[torch.Tensor] = None
    page_accum: Optional[torch.Tensor] = None
    page_out: Optional[torch.Tensor] = None
    secondary_page_accum: Optional[torch.Tensor] = None
    secondary_page_out: Optional[torch.Tensor] = None
    metadata_host: Optional[torch.Tensor] = None
    metadata_device: Optional[torch.Tensor] = None
    batch_capacity: int = 0
    page_capacity: int = 0
    stream: Optional[torch.cuda.Stream] = None
    metadata_copy_event: Optional[torch.cuda.Event] = None
    workspace_event: Optional[torch.cuda.Event] = None

    def batch_slices(
        self,
        batch_size: int,
        max_num_pages: int,
        total_num_pages: int,
        device: torch.device,
        *,
        need_secondary: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        resize_primary = (
            self.accum is None
            or self.accum.device != device
            or self.batch_capacity < batch_size
            or self.page_capacity < max_num_pages
        )
        capacity = (
            max(8, 1 << (batch_size - 1).bit_length())
            if resize_primary
            else self.batch_capacity
        )
        page_capacity = (
            1 << (max_num_pages - 1).bit_length()
            if resize_primary
            else self.page_capacity
        )
        primary_workspace_bytes = (
            capacity * (4 + 8)
            + capacity * page_capacity * 8 * 2
            + (6 * capacity + 1) * 8
        )
        if need_secondary:
            secondary_workspace_bytes = capacity * (4 + 8) + (
                capacity * page_capacity * 8 * 2
            )
        else:
            secondary_workspace_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    self.secondary_accum,
                    self.secondary_out,
                    self.secondary_page_accum,
                    self.secondary_page_out,
                )
                if tensor is not None and tensor.device == device
            )
        packed_results_bytes = (batch_size + total_num_pages) * 8
        workspace_bytes = (
            primary_workspace_bytes + secondary_workspace_bytes + packed_results_bytes
        )
        if workspace_bytes > KV_CHECKSUM_MAX_WORKSPACE_BYTES:
            raise RuntimeError(
                "KV checksum workspace exceeds the 512 MiB safety limit "
                f"(required_bytes={workspace_bytes}, batch_capacity={capacity}, "
                f"page_capacity={page_capacity})"
            )

        if resize_primary:
            # Resize both dimensions from the active shape. Keeping each prior
            # high-water mark independently can retain their unused cross-product.
            self.accum = None
            self.final_out = None
            self.page_accum = None
            self.page_out = None
            self.metadata_host = None
            self.metadata_device = None
            self.accum = torch.empty((capacity,), dtype=torch.int32, device=device)
            self.final_out = torch.empty((capacity,), dtype=TAG_DTYPE, device=device)
            self.page_accum = torch.empty(
                (capacity, page_capacity), dtype=TAG_DTYPE, device=device
            )
            self.page_out = torch.empty_like(self.page_accum)
            self.metadata_host = torch.empty(
                (6 * capacity + 1,), dtype=TAG_DTYPE, device="cpu", pin_memory=True
            )
            self.metadata_device = torch.empty(
                (6 * capacity + 1,), dtype=TAG_DTYPE, device=device
            )
            self.batch_capacity = capacity
            self.page_capacity = page_capacity
        if need_secondary and (
            self.secondary_accum is None
            or self.secondary_accum.device != device
            or self.secondary_accum.numel() != self.batch_capacity
            or self.secondary_page_accum is None
            or self.secondary_page_accum.shape[0] != self.batch_capacity
            or self.secondary_page_accum.shape[1] != self.page_capacity
        ):
            self.secondary_accum = None
            self.secondary_out = None
            self.secondary_page_accum = None
            self.secondary_page_out = None
            self.secondary_accum = torch.empty(
                (self.batch_capacity,), dtype=torch.int32, device=device
            )
            self.secondary_out = torch.empty(
                (self.batch_capacity,), dtype=TAG_DTYPE, device=device
            )
            self.secondary_page_accum = torch.empty(
                (self.batch_capacity, self.page_capacity),
                dtype=TAG_DTYPE,
                device=device,
            )
            self.secondary_page_out = torch.empty_like(self.secondary_page_accum)
        secondary_accum = self.secondary_accum if need_secondary else self.accum
        secondary_out = self.secondary_out if need_secondary else self.final_out
        secondary_page_accum = (
            self.secondary_page_accum if need_secondary else self.page_accum
        )
        secondary_page_out = (
            self.secondary_page_out if need_secondary else self.page_out
        )
        active_page_slots = batch_size * max_num_pages

        def active_page_accum_view(workspace: torch.Tensor) -> torch.Tensor:
            # Slicing both dimensions of rounded storage leaves a widened row
            # stride. Pack the active flat prefix without allocating instead.
            return workspace.view(-1)[:active_page_slots].view(
                batch_size, max_num_pages
            )

        def active_page_output_view(workspace: torch.Tensor) -> torch.Tensor:
            return workspace.view(-1)[:total_num_pages]

        return (
            self.accum[:batch_size],
            self.final_out[:batch_size],
            secondary_accum[:batch_size],
            secondary_out[:batch_size],
            active_page_accum_view(self.page_accum),
            active_page_output_view(self.page_out),
            active_page_accum_view(secondary_page_accum),
            active_page_output_view(secondary_page_out),
            self.metadata_host,
            self.metadata_device,
        )


@dataclass
class AsyncChecksumBatch:
    """GPU-side batched checksum result finalized once at metadata boundary."""

    bootstrap_rooms: List[int]
    num_tokens: List[int]
    packed_results_t: torch.Tensor
    page_counts: List[int]
    page_size: int
    logical_starts: List[int]
    stream: Optional[torch.cuda.Stream]
    finalized: Optional[List[ChecksumPlan]] = None

    def finalize(self) -> List[ChecksumPlan]:
        if self.finalized is not None:
            return self.finalized
        if self.stream is not None:
            torch.cuda.current_stream(self.packed_results_t.device).wait_stream(
                self.stream
            )
        packed_results = self.packed_results_t.detach().cpu()
        batch_size = len(self.page_counts)
        checksums = packed_results[:batch_size].tolist()
        if any(int(checksum) < 0 for checksum in checksums):
            raise RuntimeError("native KV checksum rejected invalid metadata")
        page_values = packed_results[batch_size:]
        page_digests = []
        page_offset = 0
        for page_count in self.page_counts:
            page_digests.append(
                page_values[page_offset : page_offset + page_count].tolist()
            )
            page_offset += page_count
        self.finalized = [
            ChecksumPlan(
                bootstrap_room=room,
                num_tokens=n,
                checksum=int(checksum) & _U32_MASK,
                page_size=self.page_size,
                logical_start=logical_start,
                page_digests=tuple(
                    int(value) & _U64_MASK for value in digest_row[:page_count]
                ),
            )
            for room, n, checksum, logical_start, page_count, digest_row in zip(
                self.bootstrap_rooms,
                self.num_tokens,
                checksums,
                self.logical_starts,
                self.page_counts,
                page_digests,
                strict=True,
            )
        ]
        return self.finalized


def _direct_metadata_from_pool(
    kv_pool: object,
    device: torch.device,
    cache: Optional[_DirectKVChecksumCache],
    unsupported,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    metadata_key = (id(kv_pool), str(device), int(kv_pool.layer_num))
    if cache is not None and cache.metadata_key == metadata_key:
        buffer_ptrs = cache.buffer_ptrs
        row_strides = cache.row_strides
        row_nbytes = cache.row_nbytes
        buffer_num_rows = cache.buffer_num_rows
        swa_buffer_flags = cache.swa_buffer_flags
        full_to_swa_index_mapping = cache.full_to_swa_index_mapping
        total_bytes = cache.total_bytes
        if (
            buffer_ptrs is None
            or row_strides is None
            or row_nbytes is None
            or buffer_num_rows is None
            or swa_buffer_flags is None
            or full_to_swa_index_mapping is None
        ):
            return unsupported("cached direct checksum metadata is incomplete")
        return (
            buffer_ptrs,
            row_strides,
            row_nbytes,
            buffer_num_rows,
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
    start_layer = int(getattr(kv_pool, "start_layer", 0))
    for layer_id in range(start_layer, start_layer + int(kv_pool.layer_num)):
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
    num_rows: List[int] = []
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
        num_rows.append(int(buf.shape[0]))
        total_bytes += row_b

    buffer_ptrs = torch.tensor(ptrs, dtype=TAG_DTYPE, device=device)
    row_strides = torch.tensor(strides, dtype=TAG_DTYPE, device=device)
    row_nbytes = torch.tensor(nbytes, dtype=TAG_DTYPE, device=device)
    buffer_num_rows = torch.tensor(num_rows, dtype=TAG_DTYPE, device=device)
    swa_buffer_flags = torch.tensor(is_swa_buffer, dtype=TAG_DTYPE, device=device)
    full_indices = [i for i, flag in enumerate(is_swa_buffer) if not flag]
    swa_indices = [i for i, flag in enumerate(is_swa_buffer) if flag]

    def subset_tensor(values: List[int], indices: List[int]) -> torch.Tensor:
        return torch.tensor(
            [values[i] for i in indices], dtype=TAG_DTYPE, device=device
        )

    full_buffer_ptrs = subset_tensor(ptrs, full_indices)
    full_row_strides = subset_tensor(strides, full_indices)
    full_row_nbytes = subset_tensor(nbytes, full_indices)
    full_buffer_num_rows = subset_tensor(num_rows, full_indices)
    full_swa_buffer_flags = torch.zeros(
        len(full_indices), dtype=TAG_DTYPE, device=device
    )
    full_total_bytes = sum(nbytes[i] for i in full_indices)
    swa_buffer_ptrs_only = subset_tensor(ptrs, swa_indices)
    swa_row_strides = subset_tensor(strides, swa_indices)
    swa_row_nbytes = subset_tensor(nbytes, swa_indices)
    swa_buffer_num_rows = subset_tensor(num_rows, swa_indices)
    swa_buffer_flags_only = torch.ones(len(swa_indices), dtype=TAG_DTYPE, device=device)
    swa_total_bytes = sum(nbytes[i] for i in swa_indices)
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
        cache.buffer_num_rows = buffer_num_rows
        cache.swa_buffer_flags = swa_buffer_flags
        cache.full_to_swa_index_mapping = full_to_swa_index_mapping
        cache.total_bytes = total_bytes
        cache.full_buffer_ptrs = full_buffer_ptrs
        cache.full_row_strides = full_row_strides
        cache.full_row_nbytes = full_row_nbytes
        cache.full_buffer_num_rows = full_buffer_num_rows
        cache.full_swa_buffer_flags = full_swa_buffer_flags
        cache.full_total_bytes = full_total_bytes
        cache.swa_buffer_ptrs_only = swa_buffer_ptrs_only
        cache.swa_row_strides = swa_row_strides
        cache.swa_row_nbytes = swa_row_nbytes
        cache.swa_buffer_num_rows = swa_buffer_num_rows
        cache.swa_buffer_flags_only = swa_buffer_flags_only
        cache.swa_total_bytes = swa_total_bytes
    return (
        buffer_ptrs,
        row_strides,
        row_nbytes,
        buffer_num_rows,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        total_bytes,
    )


def compare_checksums(expected: ChecksumPlan, actual_checksum: int) -> bool:
    """Return True if the decode-side checksum matches the prefill-side plan."""
    return (int(expected.checksum) & _U32_MASK) == (int(actual_checksum) & _U32_MASK)


def first_checksum_page_mismatch(
    expected: ChecksumPlan, actual: ChecksumPlan
) -> Optional[int]:
    """Return the first mismatching logical page position, if any."""
    if (
        expected.page_size != actual.page_size
        or expected.logical_start != actual.logical_start
    ):
        return expected.logical_start // max(expected.page_size, 1)
    count = max(len(expected.page_digests), len(actual.page_digests))
    for index in range(count):
        expected_digest = (
            int(expected.page_digests[index]) & _U64_MASK
            if index < len(expected.page_digests)
            else None
        )
        actual_digest = (
            int(actual.page_digests[index]) & _U64_MASK
            if index < len(actual.page_digests)
            else None
        )
        if expected_digest != actual_digest:
            return expected.logical_start // max(expected.page_size, 1) + index
    return None


def _to_int_list(values: Sequence[int] | torch.Tensor) -> List[int]:
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in values]


@contextmanager
def defer_kv_frees_until_mapping_refresh(allocator):
    """Delay page invalidation/reuse until an active request row is refreshed."""
    if getattr(allocator, "attention_tag_table", None) is None:
        yield
        return
    if not hasattr(allocator, "free_group_begin") or not hasattr(
        allocator, "free_group_end"
    ):
        raise RuntimeError(
            "KV page protection requires allocator free-group support for radix dedup"
        )
    already_grouped = not getattr(allocator, "is_not_in_free_group", True)
    if not already_grouped:
        allocator.free_group_begin()
    free_group_start = len(allocator.free_group)
    operation_group = getattr(allocator, "_free_group_ops", None)
    operation_group_start = len(operation_group) if operation_group is not None else 0
    try:
        yield
    except Exception:
        # A bookkeeping failure must leak rather than release pages still named
        # by a graph-visible request row.
        del allocator.free_group[free_group_start:]
        if operation_group is not None:
            del operation_group[operation_group_start:]
        if not already_grouped:
            allocator.is_not_in_free_group = True
        raise
    else:
        if not already_grouped:
            allocator.free_group_end()


def _retain_live_manifest_entries(manifest, live: torch.Tensor) -> None:
    """Drop manifest entries that no longer have a live physical mapping."""
    live = live.to(device=manifest.physical_page_ids_t.device, dtype=torch.bool)
    if bool(live.all().item()):
        return
    indices = torch.nonzero(live, as_tuple=False).reshape(-1)
    manifest.physical_page_ids_t = manifest.physical_page_ids_t.index_select(0, indices)
    manifest.page_positions_t = manifest.page_positions_t.index_select(0, indices)
    manifest.generations_t = manifest.generations_t.index_select(0, indices)
    manifest.expected_tags_t = manifest.expected_tags_t.index_select(0, indices)
    manifest.page_positions = [
        int(position) for position in manifest.page_positions_t.detach().cpu().tolist()
    ]
    if manifest.max_num_pages is not None:
        manifest._replacement_heap = [
            (position, index) for index, position in enumerate(manifest.page_positions)
        ]
        heapq.heapify(manifest._replacement_heap)
    manifest.revision += 1


def refresh_request_expected_mappings(
    req,
    req_to_token: torch.Tensor,
    allocator,
    *,
    max_sequence_len: Optional[int] = None,
) -> None:
    """Refresh request rows after RadixCache changes physical pages."""
    table = getattr(allocator, "attention_tag_table", None)
    request_pool_idx = getattr(req, "req_pool_idx", None)
    if table is None or request_pool_idx is None:
        return

    attention_manifests = _iter_attention_manifests(
        getattr(req, "kv_attention_tag_manifest", None)
    )
    transfer_manifests = _iter_transfer_page_tag_manifests(
        getattr(req, "kv_transfer_page_tag_manifest", None)
    )

    if max_sequence_len is not None:
        for manifest in (*attention_manifests, *transfer_manifests):
            live = manifest.page_positions_t * manifest.page_size < max_sequence_len
            _retain_live_manifest_entries(manifest, live)

    def remapped_pages(manifest) -> torch.Tensor:
        positions = manifest.page_positions_t.to(req_to_token.device, dtype=torch.long)
        token_locs = req_to_token[request_pool_idx, positions * manifest.page_size]
        if manifest.mapping_namespace == KV_EXPECTED_MAPPING_SWA:
            if not hasattr(allocator, "translate_loc_from_full_to_swa") or not hasattr(
                allocator, "attention_tag_swa_page_ids"
            ):
                raise RuntimeError("SWA protected mapping requires an SWA allocator")
            token_locs = allocator.translate_loc_from_full_to_swa(token_locs)
            live = token_locs > 0
            table.clear_expected_owner_positions(
                request_pool_idx,
                manifest.page_positions_t.to(live.device)[~live],
                manifest.mapping_namespace,
            )
            _retain_live_manifest_entries(manifest, live)
            token_locs = token_locs[live]
            pages = allocator.attention_tag_swa_page_ids(
                token_locs // manifest.page_size
            )
        else:
            pages = token_locs // manifest.page_size
        return pages.to(table.device, dtype=torch.long)

    for manifest in attention_manifests:
        pages = remapped_pages(manifest)
        generations = table.generation_of(pages)
        manifest.replace_pages(pages, generations)
        table.write_expected_owner(
            pages,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t,
            generations=manifest.generations_t,
            attention_tags=manifest.expected_tags_t,
            mapping_namespace=manifest.mapping_namespace,
        )

    for manifest in transfer_manifests:
        pages = remapped_pages(manifest)
        generations = table.generation_of(pages)
        manifest.replace_pages(pages, generations, skip_changed=True)
        table.write_expected_owner(
            pages,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t,
            generations=manifest.generations_t,
            transfer_page_tags=manifest.expected_tags_t,
            mapping_namespace=manifest.mapping_namespace,
        )


@dataclass
class _FlattenedTagBatch:
    key: tuple
    pages: torch.Tensor
    expected_tags: torch.Tensor
    generations: torch.Tensor
    owners: List[Tuple[Optional[str], object]]
    offsets: List[int]


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
        num_request_slots: int = 2,
        num_logical_pages: Optional[int] = None,
        device: str = "cpu",
        metrics_collector: object = None,
        transfer_backend: Optional[str] = None,
        is_spec_decode: bool = False,
        supports_spec_target_verify: bool = False,
        is_cuda_device: Optional[bool] = None,
    ):
        assert_protection_supported(
            config,
            allocator=allocator,
            transfer_backend=transfer_backend,
            is_spec_decode=is_spec_decode,
            supports_spec_target_verify=supports_spec_target_verify,
            is_cuda_device=is_cuda_device,
        )
        self.config = config
        self.page_size = page_size
        self.device = device
        self.metrics = metrics_collector
        self.table: Optional[KVAttentionTagTable] = None
        self._checksum_cache = _DirectKVChecksumCache()
        self._attention_verification_cache: Optional[_FlattenedTagBatch] = None
        self._transfer_verification_cache: Optional[_FlattenedTagBatch] = None
        if config.enable_attention_tags:
            if config.enable_page_history:
                history_bytes = (num_pages + 1) * KVPageHistory.BYTES_PER_PAGE
                logger.warning(
                    "KV page history enabled; allocating %.2f MiB for %s pages",
                    history_bytes / (1024 * 1024),
                    num_pages + 1,
                )
            attached_table = getattr(allocator, "attention_tag_table", None)
            if attached_table is not None:
                if not isinstance(attached_table, KVAttentionTagTable):
                    raise RuntimeError("invalid preallocated KV attention-tag table")
                if attached_table.size != num_pages + 1:
                    raise RuntimeError(
                        "preallocated KV attention-tag table size mismatch"
                    )
                if attached_table.request_epochs.numel() != num_request_slots:
                    raise RuntimeError("preallocated KV request-slot count mismatch")
                if num_logical_pages is not None and (
                    attached_table.num_logical_pages != int(num_logical_pages)
                ):
                    raise RuntimeError("preallocated KV logical-page count mismatch")
                if config.enable_page_history and attached_table.history is None:
                    raise RuntimeError("preallocated KV page history is missing")
                self.table = attached_table
            else:
                self.table = KVAttentionTagTable(
                    num_pages,
                    device=device,
                    num_request_slots=num_request_slots,
                    num_logical_pages=num_logical_pages,
                    enable_history=config.enable_page_history,
                )
                if allocator is not None and hasattr(
                    allocator, "attach_attention_tag_table"
                ):
                    allocator.attach_attention_tag_table(self.table)

    # -- attention ownership tags ------------------------------------------

    def clear_request_slot(self, request_pool_idx: int) -> None:
        if self.table is not None:
            self.table.clear_request_slot(request_pool_idx)

    def register_attention_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        request_pool_idx: int = 1,
        bootstrap_room: int,
        page_positions: Optional[Sequence[int]] = None,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
    ) -> Optional[AttentionTagManifest]:
        """Write attention ownership tags for a freshly transferred request.

        Captures each physical page's current allocation generation, computes
        the expected ownership tag, scatters it into the sidecar buffer, and returns a
        manifest cached on the request for later (per-step) verification.
        """
        if not self.config.enable_attention_tags or self.table is None:
            return None
        pages_t = torch.as_tensor(
            page_physical_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        generations_t = self.table.generation_of(pages_t)
        manifest = AttentionTagManifest.from_tensors(
            self.page_size,
            bootstrap_room,
            pages_t,
            generations_t,
            page_positions=page_positions,
            max_num_pages=max_num_pages,
            mapping_namespace=mapping_namespace,
        )
        self.table.write_expected_owner(
            manifest.physical_page_ids_t,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t,
            generations=manifest.generations_t,
            attention_tags=manifest.expected_tags_t,
            mapping_namespace=manifest.mapping_namespace,
        )
        self.table._record_history(
            manifest.physical_page_ids_t,
            KVPageHistory.TAG_REFRESH,
            generations=self.table.generations,
            bootstrap_rooms=bootstrap_room,
            page_positions=manifest.page_positions_t,
            values=manifest.expected_tags_t,
            generations_by_page=True,
        )
        return manifest

    def refresh_tail_page(
        self,
        manifest: Optional[AttentionTagManifest],
        *,
        logical_pos: int,
        request_pool_idx: int = 1,
        physical_page_id: torch.Tensor,
        swa_physical_page_id: Optional[torch.Tensor] = None,
    ) -> Optional[torch.cuda.Event]:
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
            for sub_manifest in manifests:
                sub_page_id = (
                    swa_physical_page_id
                    if sub_manifest.mapping_namespace == KV_EXPECTED_MAPPING_SWA
                    else physical_page_id
                )
                if sub_page_id is None:
                    continue
                self.refresh_tail_page(
                    sub_manifest,
                    logical_pos=logical_pos,
                    request_pool_idx=request_pool_idx,
                    physical_page_id=sub_page_id,
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
        page_position = int(logical_pos) // manifest.page_size
        entry_index = manifest._entry_index_for_page_position(page_position)
        if entry_index is None:
            raise RuntimeError("attention tag tail page is missing from its manifest")
        self.table.write_expected_owner(
            page_id_t,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t[entry_index : entry_index + 1],
            generations=manifest.generations_t[entry_index : entry_index + 1],
            attention_tags=expected_t,
            mapping_namespace=manifest.mapping_namespace,
        )
        self.table._record_history(
            page_id_t,
            KVPageHistory.TAG_REFRESH,
            generations=self.table.generations,
            bootstrap_rooms=manifest.bootstrap_room,
            page_positions=page_position,
            values=expected_t,
            generations_by_page=True,
        )

    # -- transfer-written page tags ----------------------------------------

    def register_transfer_page_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        request_pool_idx: int = 1,
        bootstrap_room: int,
        page_positions: Optional[Sequence[int]] = None,
        write_actual: bool = False,
        max_num_pages: Optional[int] = None,
        mapping_namespace: int = KV_EXPECTED_MAPPING_FULL,
    ) -> Optional[TransferPageTagManifest]:
        """Create expected transfer page tags for decode-owned pages.

        ``write_actual`` is False for PD-transferred prompt pages: decode commits
        those actual tags after the KV transfer completes. It is True only for
        locally produced decode pages that never pass through prefill transfer.
        """
        if not self.config.enable_attention_tags or self.table is None:
            return None
        pages_t = torch.as_tensor(
            page_physical_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        generations_t = self.table.generation_of(pages_t)
        manifest = TransferPageTagManifest.from_tensors(
            self.page_size,
            bootstrap_room,
            pages_t,
            generations_t,
            page_positions=page_positions,
            max_num_pages=max_num_pages,
            mapping_namespace=mapping_namespace,
        )
        if write_actual:
            self.table.write_transfer_page_tags(
                manifest.physical_page_ids_t,
                manifest.expected_tags_t,
            )
        self.table.write_expected_owner(
            manifest.physical_page_ids_t,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t,
            generations=manifest.generations_t,
            transfer_page_tags=manifest.expected_tags_t,
            mapping_namespace=manifest.mapping_namespace,
        )
        self.table._record_history(
            manifest.physical_page_ids_t,
            KVPageHistory.TRANSFER_EXPECTED,
            generations=self.table.generations,
            bootstrap_rooms=bootstrap_room,
            page_positions=manifest.page_positions_t,
            values=manifest.expected_tags_t,
            generations_by_page=True,
        )
        return manifest

    def commit_transfer_page_tags(self, manifest) -> None:
        """Commit retained transfer tags after all KV/state writes have landed."""
        if not self.config.enable_attention_tags or self.table is None:
            return None
        for sub_manifest in _iter_transfer_page_tag_manifests(manifest):
            self.table.write_transfer_page_tags(
                sub_manifest.physical_page_ids_t,
                sub_manifest.expected_tags_t,
            )
            self.table._record_history(
                sub_manifest.physical_page_ids_t,
                KVPageHistory.TRANSFER_WRITE,
                generations=self.table.generations,
                bootstrap_rooms=sub_manifest.bootstrap_room,
                page_positions=sub_manifest.page_positions_t,
                values=sub_manifest.expected_tags_t,
                generations_by_page=True,
            )

    def write_transfer_page_tags(
        self,
        *,
        page_physical_ids: Sequence[int],
        transfer_page_tags: Sequence[int],
        bootstrap_room: int = 0,
    ) -> None:
        """Write actual transfer page tags produced by the transfer path."""
        if not self.config.enable_attention_tags or self.table is None:
            return
        page_ids = _to_int_list(page_physical_ids)
        tags = _to_int_list(transfer_page_tags)
        if len(page_ids) != len(tags):
            raise RuntimeError("transfer page tag metadata length mismatch")
        if not page_ids:
            return None
        pages_t = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        tags_t = transfer_page_tags_to_tensor(tags, device=self.device)
        self.table.write_transfer_page_tags(pages_t, tags_t)
        self.table._record_history(
            pages_t,
            KVPageHistory.TRANSFER_WRITE,
            generations=self.table.generations,
            bootstrap_rooms=bootstrap_room,
            values=tags_t,
            generations_by_page=True,
        )
        if pages_t.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(pages_t.device))
            return event
        return None

    def refresh_transfer_page_tag_tail_page(
        self,
        manifest: Optional[TransferPageTagManifest],
        *,
        logical_pos: int,
        request_pool_idx: int = 1,
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
            for sub_manifest in manifests:
                sub_page_id = (
                    swa_physical_page_id
                    if sub_manifest.mapping_namespace == KV_EXPECTED_MAPPING_SWA
                    else physical_page_id
                )
                if sub_page_id is None:
                    continue
                self.refresh_transfer_page_tag_tail_page(
                    sub_manifest,
                    logical_pos=logical_pos,
                    request_pool_idx=request_pool_idx,
                    physical_page_id=sub_page_id,
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
        page_position = int(logical_pos) // manifest.page_size
        entry_index = manifest._entry_index_for_page_position(page_position)
        if entry_index is None:
            raise RuntimeError("transfer tag tail page is missing from its manifest")
        expected_t.fill_(TRANSFER_PAGE_TAG_SKIP)
        manifest.expected_tags_t[entry_index] = TRANSFER_PAGE_TAG_SKIP
        self.table.write_expected_owner(
            page_id_t,
            request_pool_idx=request_pool_idx,
            page_positions=manifest.page_positions_t[entry_index : entry_index + 1],
            generations=manifest.generations_t[entry_index : entry_index + 1],
            transfer_page_tags=expected_t,
            mapping_namespace=manifest.mapping_namespace,
        )
        self.table._record_history(
            page_id_t,
            KVPageHistory.TRANSFER_WRITE,
            generations=self.table.generations,
            bootstrap_rooms=manifest.bootstrap_room,
            page_positions=page_position,
            values=expected_t,
            generations_by_page=True,
        )

    def _flatten_tag_batch(
        self,
        items: Sequence[Tuple[Optional[str], object]],
        *,
        transfer: bool,
    ) -> Optional[_FlattenedTagBatch]:
        iterator = (
            _iter_transfer_page_tag_manifests if transfer else _iter_attention_manifests
        )
        entries = [
            (rid, sub_manifest)
            for rid, manifest in items
            for sub_manifest in iterator(manifest)
            if sub_manifest is not None and sub_manifest.num_pages > 0
        ]
        cache_attr = (
            "_transfer_verification_cache"
            if transfer
            else "_attention_verification_cache"
        )
        if not entries:
            setattr(self, cache_attr, None)
            return None

        key = tuple(
            (rid, id(manifest), manifest.revision, manifest.num_pages)
            for rid, manifest in entries
        )
        cached = getattr(self, cache_attr)
        if cached is not None and cached.key == key:
            return cached

        page_tensors = [
            manifest.physical_pages_tensor(self.device) for _, manifest in entries
        ]
        expected_tensors = [
            manifest.expected_tags_tensor(self.device) for _, manifest in entries
        ]
        generation_tensors = [
            manifest.generations_tensor(self.device) for _, manifest in entries
        ]

        def cat_or_single(tensors: List[torch.Tensor]) -> torch.Tensor:
            return tensors[0] if len(tensors) == 1 else torch.cat(tensors)

        offsets = [0]
        for pages in page_tensors:
            offsets.append(offsets[-1] + int(pages.numel()))
        flattened = _FlattenedTagBatch(
            key=key,
            pages=cat_or_single(page_tensors),
            expected_tags=cat_or_single(expected_tensors),
            generations=cat_or_single(generation_tensors),
            owners=entries,
            offsets=offsets,
        )
        setattr(self, cache_attr, flattened)
        return flattened

    def clear_verification_cache(self) -> None:
        self._attention_verification_cache = None
        self._transfer_verification_cache = None

    def _transfer_mismatch_details(
        self, flattened: _FlattenedTagBatch, mismatch: torch.Tensor
    ) -> List[KVTransferPageTagMismatch]:
        actual_all = self.table.read_transfer_page_tags(flattened.pages)
        actual_generations = self.table.generation_of(flattened.pages)
        bad_idx = torch.nonzero(mismatch).reshape(-1).cpu().tolist()
        seen_rids = set()
        result = []
        for i in bad_idx:
            owner_idx = bisect_right(flattened.offsets, int(i)) - 1
            rid, manifest = flattened.owners[owner_idx]
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            p = int(i) - flattened.offsets[owner_idx]
            expected_tag = int(flattened.expected_tags[i].item())
            actual_tag = int(actual_all[i].item())
            expected_generation = int(flattened.generations[i].item())
            actual_generation = int(actual_generations[i].item())
            tag_differs = expected_tag != actual_tag
            generation_differs = expected_generation != actual_generation
            cause = (
                "both"
                if tag_differs and generation_differs
                else "tag_value" if tag_differs else "generation"
            )
            error = KVTransferPageTagMismatch(
                rid=rid,
                bootstrap_room=manifest.bootstrap_room,
                page_id=int(flattened.pages[i].item()),
                page_position=manifest.page_positions[p],
                expected_transfer_tag=expected_tag,
                actual_transfer_tag=actual_tag,
                expected_generation=expected_generation,
                actual_generation=actual_generation,
                cause=cause,
            )
            attach_kv_protection_incident(
                error,
                kind="transfer_page_tag",
                phase="pre_attention",
                history=self.table.materialize_history(error.page_id),
            )
            result.append(error)
        if self.metrics is not None and result:
            self.metrics.increment_kv_transfer_page_tag_mismatches(len(result))
        return result

    def _attention_mismatch_details(
        self, flattened: _FlattenedTagBatch, mismatch: torch.Tensor
    ) -> List[KVAttentionTagMismatch]:
        actual_all = self.table.read_tags(flattened.pages)
        actual_generations = self.table.generation_of(flattened.pages)
        bad_idx = torch.nonzero(mismatch).reshape(-1).cpu().tolist()
        seen_rids = set()
        result = []
        for i in bad_idx:
            owner_idx = bisect_right(flattened.offsets, int(i)) - 1
            rid, manifest = flattened.owners[owner_idx]
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            p = int(i) - flattened.offsets[owner_idx]
            expected_tag = int(flattened.expected_tags[i].item())
            actual_tag = int(actual_all[i].item())
            expected_generation = int(flattened.generations[i].item())
            actual_generation = int(actual_generations[i].item())
            tag_differs = expected_tag != actual_tag
            generation_differs = expected_generation != actual_generation
            cause = (
                "both"
                if tag_differs and generation_differs
                else "tag_value" if tag_differs else "generation"
            )
            error = KVAttentionTagMismatch(
                rid=rid,
                bootstrap_room=manifest.bootstrap_room,
                page_id=int(flattened.pages[i].item()),
                page_position=manifest.page_positions[p],
                expected_tag=expected_tag,
                actual_tag=actual_tag,
                expected_generation=expected_generation,
                actual_generation=actual_generation,
                cause=cause,
            )
            attach_kv_protection_incident(
                error,
                kind="attention_tag",
                phase="pre_attention",
                history=self.table.materialize_history(error.page_id),
            )
            result.append(error)
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_mismatches(len(result))
        return result

    def verify_transfer_page_tag_batch(
        self,
        items: Sequence[Tuple[str, TransferPageTagManifest]],
    ) -> List[KVTransferPageTagMismatch]:
        """Vectorized verification of actual transfer page tags for a batch."""
        if not self.config.enable_attention_tags or self.table is None:
            return []
        flattened = self._flatten_tag_batch(items, transfer=True)
        if flattened is None:
            return []
        mismatch = _transfer_page_tag_mismatch_mask(
            self.table,
            flattened.pages,
            flattened.expected_tags,
            flattened.generations,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_transfer_page_tag_checked_pages(
                int(flattened.pages.numel())
            )
        if not bool(mismatch.any().item()):
            return []
        return self._transfer_mismatch_details(flattened, mismatch)

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
        flattened = self._flatten_tag_batch(items, transfer=False)
        if flattened is None:
            return []
        mismatch = _attention_tag_mismatch_mask(
            self.table,
            flattened.pages,
            flattened.expected_tags,
            flattened.generations,
        )
        if self.metrics is not None:
            self.metrics.increment_kv_attention_tag_checked_pages(
                int(flattened.pages.numel())
            )
        if not bool(mismatch.any().item()):
            return []
        return self._attention_mismatch_details(flattened, mismatch)

    def verify_protection_batch(
        self,
        attention_items: Sequence[Tuple[str, AttentionTagManifest]],
        transfer_items: Sequence[Tuple[str, TransferPageTagManifest]],
    ) -> List[KVPageProtectionError]:
        """Verify both tag families with one host-visible mismatch decision."""
        if not self.config.enable_attention_tags or self.table is None:
            return []
        attention = self._flatten_tag_batch(attention_items, transfer=False)
        transfer = self._flatten_tag_batch(transfer_items, transfer=True)
        attention_mismatch = (
            None
            if attention is None
            else _attention_tag_mismatch_mask(
                self.table,
                attention.pages,
                attention.expected_tags,
                attention.generations,
            )
        )
        transfer_mismatch = (
            None
            if transfer is None
            else _transfer_page_tag_mismatch_mask(
                self.table,
                transfer.pages,
                transfer.expected_tags,
                transfer.generations,
            )
        )
        if attention is not None and self.metrics is not None:
            self.metrics.increment_kv_attention_tag_checked_pages(
                int(attention.pages.numel())
            )
        if transfer is not None and self.metrics is not None:
            self.metrics.increment_kv_transfer_page_tag_checked_pages(
                int(transfer.pages.numel())
            )

        reductions = [
            mismatch.any()
            for mismatch in (attention_mismatch, transfer_mismatch)
            if mismatch is not None
        ]
        if not reductions:
            return []
        any_mismatch = reductions[0]
        for reduction in reductions[1:]:
            any_mismatch = torch.logical_or(any_mismatch, reduction)
        if not bool(any_mismatch.item()):
            return []

        result: List[KVPageProtectionError] = []
        if attention is not None and bool(attention_mismatch.any().item()):
            result.extend(
                self._attention_mismatch_details(attention, attention_mismatch)
            )
        if transfer is not None and bool(transfer_mismatch.any().item()):
            result.extend(self._transfer_mismatch_details(transfer, transfer_mismatch))
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
        swa_evicted_lens: Optional[Sequence[int]] = None,
        checksum_page_size: int = TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE,
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
            return AsyncChecksumBatch(
                [],
                [],
                torch.empty(0, dtype=TAG_DTYPE),
                [],
                int(checksum_page_size),
                [],
                None,
            )
        if not (len(req_pool_indices) == len(bootstrap_rooms) == len(num_tokens)):
            raise RuntimeError("batched checksum metadata length mismatch")
        batch_size = len(req_pool_indices)
        starts_list = (
            [int(x) for x in starts] if starts is not None else [0] * batch_size
        )
        if len(starts_list) != batch_size:
            raise RuntimeError("batched checksum starts length mismatch")
        num_tokens_list = [int(x) for x in num_tokens]
        checksum_page_size = int(checksum_page_size)
        if checksum_page_size <= 0:
            raise RuntimeError("checksum_page_size must be positive")
        if any(n < 0 for n in num_tokens_list):
            raise RuntimeError("batched checksum lengths must be non-negative")
        table_rows, table_columns = req_to_token.shape
        if any(index < 0 or index >= table_rows for index in req_pool_indices):
            raise RuntimeError("batched checksum request-pool index is out of bounds")
        if any(
            start < 0 or start + n > table_columns
            for start, n in zip(starts_list, num_tokens_list, strict=True)
        ):
            raise RuntimeError("batched checksum token range is out of bounds")
        page_counts = [
            (
                (start % checksum_page_size + n + checksum_page_size - 1)
                // checksum_page_size
                if n > 0
                else 0
            )
            for start, n in zip(starts_list, num_tokens_list, strict=True)
        ]
        page_offsets = [0]
        for page_count in page_counts:
            page_offsets.append(page_offsets[-1] + page_count)
        total_num_pages = page_offsets[-1]
        max_num_pages = max(1, max(page_counts))
        evicted_list = (
            [int(x) for x in swa_evicted_lens]
            if swa_evicted_lens is not None
            else [0] * batch_size
        )
        if len(evicted_list) != batch_size:
            raise RuntimeError("batched checksum swa_evicted length mismatch")
        evicted_list = [
            min(max(0, evicted), num_tokens_list[i])
            for i, evicted in enumerate(evicted_list)
        ]

        try:
            from sgl_kernel.kvcacheio import (
                kv_checksum_direct_table_batched_with_pages_compact as _op,
            )
        except Exception as e:  # pragma: no cover - depends on CUDA build
            raise RuntimeError(
                f"sgl_kernel compact page checksum op unavailable ({e})"
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
            buffer_num_rows,
            swa_buffer_flags,
            full_to_swa_index_mapping,
            total_bytes,
        ) = _direct_metadata_from_pool(
            kv_pool, device, self._checksum_cache, _unsupported
        )
        num_lanes = select_checksum_byte_count(total_bytes)
        has_swa = full_to_swa_index_mapping.numel() > 0
        is_capped = (int(num_lanes) * 8) < int(total_bytes)
        # The representation must depend only on the pool layout, not on which
        # other requests happen to share the prefill/decode checksum batch.
        need_two_pass = self._checksum_cache.swa_total_bytes > 0
        swa_starts_list = [starts_list[i] + evicted_list[i] for i in range(batch_size)]
        swa_lengths_list = [
            num_tokens_list[i] - evicted_list[i] for i in range(batch_size)
        ]
        device_changed = (
            self._checksum_cache.accum is not None
            and self._checksum_cache.accum.device != device
        )
        workspace_event = self._checksum_cache.workspace_event
        workspace_resize = (
            self._checksum_cache.accum is None
            or device_changed
            or self._checksum_cache.batch_capacity < batch_size
            or self._checksum_cache.page_capacity < max_num_pages
            or (
                need_two_pass
                and (
                    self._checksum_cache.secondary_accum is None
                    or self._checksum_cache.secondary_accum.numel() < batch_size
                    or self._checksum_cache.secondary_page_accum is None
                    or self._checksum_cache.secondary_page_accum.shape[1]
                    < max_num_pages
                )
            )
        )
        if (
            workspace_resize
            and workspace_event is not None
            and not workspace_event.query()
        ):
            workspace_event.synchronize()
        metadata_copy_event = self._checksum_cache.metadata_copy_event
        if metadata_copy_event is not None and not metadata_copy_event.query():
            metadata_copy_event.synchronize()
        if device_changed:
            workspace_event = None
            self._checksum_cache.workspace_event = None
            metadata_copy_event = None
            self._checksum_cache.metadata_copy_event = None
        (
            accum,
            final_out,
            secondary_accum,
            secondary_out,
            page_accum,
            page_out,
            secondary_page_accum,
            secondary_page_out,
            metadata_host,
            metadata_device,
        ) = self._checksum_cache.batch_slices(
            batch_size,
            max_num_pages,
            total_num_pages,
            device,
            need_secondary=need_two_pass,
        )
        metadata_host[: 5 * batch_size].copy_(
            torch.tensor(
                [
                    req_pool_indices,
                    starts_list,
                    num_tokens_list,
                    swa_starts_list,
                    swa_lengths_list,
                ],
                dtype=TAG_DTYPE,
            ).reshape(-1)
        )
        active_metadata_size = 6 * batch_size + 1
        metadata_host[5 * batch_size : active_metadata_size].copy_(
            torch.tensor(page_offsets, dtype=TAG_DTYPE)
        )
        req_pool_t = metadata_device[:batch_size]
        starts_t = metadata_device[batch_size : 2 * batch_size]
        lengths_t = metadata_device[2 * batch_size : 3 * batch_size]
        swa_starts_t = metadata_device[3 * batch_size : 4 * batch_size]
        swa_lengths_t = metadata_device[4 * batch_size : 5 * batch_size]
        page_offsets_t = metadata_device[5 * batch_size : active_metadata_size]

        if (
            self._checksum_cache.stream is None
            or self._checksum_cache.stream.device != device
        ):
            self._checksum_cache.stream = torch.cuda.Stream(device=device)
        stream = self._checksum_cache.stream
        stream.wait_stream(torch.cuda.current_stream(device))
        max_num_tokens = max(num_tokens_list)
        max_swa_tokens = max(swa_lengths_list)
        with torch.cuda.stream(stream):
            packed_results_t = torch.empty(
                batch_size + total_num_pages,
                dtype=TAG_DTYPE,
                device=device,
            )
            checksums_t = packed_results_t[:batch_size]
            page_digests_t = packed_results_t[batch_size:]
            metadata_device[:active_metadata_size].copy_(
                metadata_host[:active_metadata_size], non_blocking=True
            )
            if metadata_copy_event is None:
                metadata_copy_event = torch.cuda.Event()
                self._checksum_cache.metadata_copy_event = metadata_copy_event
            metadata_copy_event.record(stream)
            if not need_two_pass:
                page_accum.zero_()
                _op(
                    buffer_ptrs,
                    row_strides,
                    row_nbytes,
                    buffer_num_rows,
                    swa_buffer_flags,
                    full_to_swa_index_mapping,
                    req_to_token,
                    req_pool_t,
                    starts_t,
                    lengths_t,
                    starts_t,
                    page_offsets_t,
                    int(max_num_tokens),
                    int(num_lanes),
                    checksum_page_size,
                    max_num_pages,
                    has_swa,
                    is_capped,
                    accum,
                    checksums_t,
                    page_accum,
                    page_digests_t,
                )
            else:
                empty_mapping = full_to_swa_index_mapping[:0]
                full_out = None
                if self._checksum_cache.full_total_bytes > 0:
                    page_accum.zero_()
                    full_num_lanes = select_checksum_byte_count(
                        self._checksum_cache.full_total_bytes
                    )
                    _op(
                        self._checksum_cache.full_buffer_ptrs,
                        self._checksum_cache.full_row_strides,
                        self._checksum_cache.full_row_nbytes,
                        self._checksum_cache.full_buffer_num_rows,
                        self._checksum_cache.full_swa_buffer_flags,
                        empty_mapping,
                        req_to_token,
                        req_pool_t,
                        starts_t,
                        lengths_t,
                        starts_t,
                        page_offsets_t,
                        int(max_num_tokens),
                        int(full_num_lanes),
                        checksum_page_size,
                        max_num_pages,
                        False,
                        False,
                        accum,
                        final_out,
                        page_accum,
                        page_out,
                    )
                    full_out = final_out

                secondary_page_accum.zero_()
                if full_out is not None:
                    secondary_page_out.zero_()
                else:
                    page_digests_t.zero_()
                swa_num_lanes = select_checksum_byte_count(
                    self._checksum_cache.swa_total_bytes
                )
                _op(
                    self._checksum_cache.swa_buffer_ptrs_only,
                    self._checksum_cache.swa_row_strides,
                    self._checksum_cache.swa_row_nbytes,
                    self._checksum_cache.swa_buffer_num_rows,
                    self._checksum_cache.swa_buffer_flags_only,
                    full_to_swa_index_mapping,
                    req_to_token,
                    req_pool_t,
                    swa_starts_t,
                    swa_lengths_t,
                    starts_t,
                    page_offsets_t,
                    int(max_swa_tokens),
                    int(swa_num_lanes),
                    checksum_page_size,
                    max_num_pages,
                    True,
                    False,
                    secondary_accum,
                    checksums_t if full_out is None else secondary_out,
                    secondary_page_accum,
                    page_digests_t if full_out is None else secondary_page_out,
                )
                if full_out is not None:
                    torch.bitwise_xor(full_out, secondary_out, out=checksums_t)
                    # Plain XOR would cancel two -1 rejection sentinels.
                    torch.bitwise_or(full_out, secondary_out, out=full_out)
                    torch.bitwise_right_shift(full_out, 63, out=full_out)
                    torch.bitwise_or(checksums_t, full_out, out=checksums_t)
                    torch.bitwise_xor(page_out, secondary_page_out, out=page_digests_t)
            if workspace_event is None:
                workspace_event = torch.cuda.Event()
                self._checksum_cache.workspace_event = workspace_event
            workspace_event.record(stream)
        return AsyncChecksumBatch(
            bootstrap_rooms=[int(x) for x in bootstrap_rooms],
            num_tokens=num_tokens_list,
            packed_results_t=packed_results_t,
            page_counts=page_counts,
            page_size=checksum_page_size,
            logical_starts=starts_list,
            stream=stream,
        )

    def compare_destination_checksum(
        self,
        expected: ChecksumPlan,
        actual_checksum: int | ChecksumPlan,
        *,
        rid: Optional[str] = None,
    ) -> bool:
        """Compare a precomputed destination checksum and update metrics."""
        if self.metrics is not None:
            checked_pages = (
                len(expected.page_digests)
                or (int(expected.num_tokens) + TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE - 1)
                // TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE
            )
            self.metrics.increment_kv_transfer_checksum_checked_pages(checked_pages)
        actual_plan = (
            actual_checksum
            if isinstance(actual_checksum, ChecksumPlan)
            else ChecksumPlan(
                bootstrap_room=expected.bootstrap_room,
                num_tokens=expected.num_tokens,
                checksum=int(actual_checksum),
            )
        )
        ok = compare_checksums(expected, actual_plan.checksum)
        if expected.page_digests:
            ok = ok and first_checksum_page_mismatch(expected, actual_plan) is None
        if not ok and self.metrics is not None:
            self.metrics.increment_kv_transfer_checksum_mismatches()
        del rid
        return ok
