"""Nonblocking target-side ingress contract for remote MTP candidates.

This module intentionally has no torch or networking dependency.  A transport
adapter may publish candidates from a background thread, while the target
worker performs only an O(batch-size) in-memory claim at the scheduling seal.
The target verifier remains the sole authority allowed to commit tokens.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class RemoteMTPContractError(ValueError):
    """Raised when an engine/adapter boundary object violates the ABI."""


def remote_mtp_output_token_count(
    *,
    prompt_token_count: int,
    req_output_token_count: int,
    kv_boundary: int | None,
) -> int:
    """Return the target-authoritative semantic output boundary.

    ``kv_boundary`` is SGLang's honest ``Req.kv_committed_len``: it excludes
    the relayed root token that is already part of the semantic output prefix.
    ``Req.output_ids`` is an independent CPU-settled lower bound.
    """

    for name, value in (
        ("prompt_token_count", prompt_token_count),
        ("req_output_token_count", req_output_token_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RemoteMTPContractError(f"{name} must be a non-negative integer")
    if kv_boundary is None:
        return req_output_token_count
    if isinstance(kv_boundary, bool) or not isinstance(kv_boundary, int):
        raise RemoteMTPContractError("kv_boundary must be an integer or None")
    return max(req_output_token_count, kv_boundary - prompt_token_count + 1)


def remote_mtp_linear_chain_layout(
    depth: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return native EAGLE topk=1 parent and selected-index metadata."""

    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise RemoteMTPContractError("depth must be a positive integer")
    parents = tuple(range(-1, depth - 1)) if depth > 1 else ()
    selected_indices = tuple(range(depth))
    return parents, selected_indices


def remote_mtp_rejoin_order(
    native_order: Sequence[tuple[str, str]],
    live_identities: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Restore surviving native rows and append genuinely new rows stably.

    The scheduler owns request liveness (finished/retracted/cancelled). This
    pure helper owns only identity ordering and duplicate suppression, which
    makes detach/rejoin lifecycle behavior exhaustively CPU-testable.
    """

    native = tuple(native_order)
    live = tuple(live_identities)
    for label, values in (("native", native), ("live", live)):
        for identity in values:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or any(not isinstance(value, str) or not value for value in identity)
            ):
                raise RemoteMTPContractError(
                    f"{label} rejoin identities must be non-empty string pairs"
                )
    if len(set(native)) != len(native):
        raise RemoteMTPContractError("native rejoin order repeats an identity")
    live_unique = tuple(dict.fromkeys(live))
    live_set = set(live_unique)
    native_set = set(native)
    return tuple(identity for identity in native if identity in live_set) + tuple(
        identity for identity in live_unique if identity not in native_set
    )


@dataclass(frozen=True, slots=True)
class RemoteMTPRequestView:
    """Target-authoritative identity visible at one scheduling seal.

    ``output_token_count`` binds a proposal to the current committed output
    boundary without hashing the full, possibly very long, agent transcript on
    the scheduler thread.  The external adapter is still responsible for the
    complete semantic-prefix digest and checkpoint/configuration fences.
    """

    request_id: str
    request_incarnation: str
    prompt_token_count: int
    output_token_count: int
    last_target_service_monotonic_ns: int | None = None
    max_service_gap_ns: int | None = None
    service_completion_guard_ns: int | None = None
    service_period_ns: int | None = None
    remaining_tokens: int | None = None

    def validate(self) -> RemoteMTPRequestView:
        if not self.request_id:
            raise RemoteMTPContractError("request_id must be non-empty")
        if not self.request_incarnation:
            raise RemoteMTPContractError("request_incarnation must be non-empty")
        for name in ("prompt_token_count", "output_token_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RemoteMTPContractError(f"{name} must be a non-negative integer")
        if self.remaining_tokens is not None and (
            isinstance(self.remaining_tokens, bool)
            or not isinstance(self.remaining_tokens, int)
            or self.remaining_tokens < 0
        ):
            raise RemoteMTPContractError(
                "remaining_tokens must be a non-negative integer or None"
            )
        gap_fields = (
            self.max_service_gap_ns,
            self.service_completion_guard_ns,
        )
        if any(value is not None for value in gap_fields):
            if self.last_target_service_monotonic_ns is None or not all(
                value is not None for value in gap_fields
            ):
                raise RemoteMTPContractError(
                    "request service-gap identity must be all-or-none"
                )
            for name, value in (
                (
                    "last_target_service_monotonic_ns",
                    self.last_target_service_monotonic_ns,
                ),
                ("max_service_gap_ns", self.max_service_gap_ns),
                (
                    "service_completion_guard_ns",
                    self.service_completion_guard_ns,
                ),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or (name == "max_service_gap_ns" and value == 0)
                ):
                    raise RemoteMTPContractError(
                        f"{name} must be a valid monotonic service bound"
                    )
            assert self.max_service_gap_ns is not None
            assert self.service_completion_guard_ns is not None
            if self.service_completion_guard_ns >= self.max_service_gap_ns:
                raise RemoteMTPContractError(
                    "service completion guard must be below maximum service gap"
                )
        if self.service_period_ns is not None:
            if self.last_target_service_monotonic_ns is None:
                raise RemoteMTPContractError(
                    "service period requires the last target service timestamp"
                )
            if (
                isinstance(self.service_period_ns, bool)
                or not isinstance(self.service_period_ns, int)
                or self.service_period_ns <= 0
            ):
                raise RemoteMTPContractError(
                    "service_period_ns must be a positive integer"
                )
        if self.last_target_service_monotonic_ns is not None and (
            isinstance(self.last_target_service_monotonic_ns, bool)
            or not isinstance(self.last_target_service_monotonic_ns, int)
            or self.last_target_service_monotonic_ns < 0
        ):
            raise RemoteMTPContractError(
                "last_target_service_monotonic_ns must be a valid timestamp"
            )
        return self


def remote_mtp_service_window_partition(
    requests: Sequence[RemoteMTPRequestView],
    *,
    now_monotonic_ns: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    """Partition a resident batch by request-level not-before windows."""

    if (
        isinstance(now_monotonic_ns, bool)
        or not isinstance(now_monotonic_ns, int)
        or now_monotonic_ns < 0
    ):
        raise RemoteMTPContractError("now_monotonic_ns must be a valid timestamp")
    eligible: list[int] = []
    deferred: list[int] = []
    next_deadline: int | None = None
    for index, request in enumerate(requests):
        request.validate()
        period = request.service_period_ns
        last = request.last_target_service_monotonic_ns
        if period is None or last is None or now_monotonic_ns >= last + period:
            eligible.append(index)
            continue
        deferred.append(index)
        deadline = last + period
        next_deadline = (
            deadline if next_deadline is None else min(next_deadline, deadline)
        )
    return tuple(eligible), tuple(deferred), next_deadline


@dataclass(frozen=True, slots=True)
class RemoteMTPSchedulerAdvice:
    """Exact decode-seal advice returned to the native scheduler.

    Observe-only advice carries no plan identity. Enforced advice is accepted
    only when it binds a coordinator plan generation/window and one exact
    candidate ID per preferred request. Engine code still revalidates the
    current seal and candidate claim before a forward can launch.
    """

    requests: tuple[RemoteMTPRequestView, ...]
    preferred_request_ids: tuple[str, ...]
    compatible_depth: int | None
    reason: str
    seal_monotonic_ns: int
    enforce: bool = False
    plan_id: str | None = None
    snapshot_generation: int | None = None
    window_id: str | None = None
    candidate_ids: tuple[str, ...] = ()

    def validate(self) -> RemoteMTPSchedulerAdvice:
        if not self.requests:
            raise RemoteMTPContractError("scheduler advice requires a decode seal")
        for request in self.requests:
            request.validate()
        request_ids = tuple(request.request_id for request in self.requests)
        if len(set(request_ids)) != len(request_ids):
            raise RemoteMTPContractError("decode seal repeats a request ID")
        if len(set(self.preferred_request_ids)) != len(self.preferred_request_ids):
            raise RemoteMTPContractError("scheduler advice repeats a preferred request")
        if not set(self.preferred_request_ids).issubset(request_ids):
            raise RemoteMTPContractError(
                "scheduler advice names a request outside the seal"
            )
        if self.compatible_depth is not None and (
            isinstance(self.compatible_depth, bool)
            or not isinstance(self.compatible_depth, int)
            or self.compatible_depth <= 0
        ):
            raise RemoteMTPContractError("scheduler advice depth must be positive")
        if not self.reason:
            raise RemoteMTPContractError(
                "scheduler advice requires an attribution reason"
            )
        if (
            isinstance(self.seal_monotonic_ns, bool)
            or not isinstance(self.seal_monotonic_ns, int)
            or self.seal_monotonic_ns < 0
        ):
            raise RemoteMTPContractError(
                "scheduler advice requires a valid monotonic seal time"
            )
        if len(set(self.candidate_ids)) != len(self.candidate_ids) or any(
            not value for value in self.candidate_ids
        ):
            raise RemoteMTPContractError(
                "scheduler advice candidate IDs must be unique and non-empty"
            )
        plan_fields = (self.plan_id, self.snapshot_generation, self.window_id)
        if any(value is not None for value in plan_fields) and not all(
            value is not None for value in plan_fields
        ):
            raise RemoteMTPContractError(
                "scheduler advice plan identity must be all-or-none"
            )
        if self.snapshot_generation is not None and (
            isinstance(self.snapshot_generation, bool)
            or not isinstance(self.snapshot_generation, int)
            or self.snapshot_generation < 0
        ):
            raise RemoteMTPContractError(
                "scheduler advice generation must be non-negative"
            )
        if self.plan_id is not None and (not self.plan_id or not self.window_id):
            raise RemoteMTPContractError(
                "scheduler advice plan and window IDs must be non-empty"
            )
        if self.enforce:
            if self.compatible_depth is None:
                raise RemoteMTPContractError(
                    "enforced scheduler advice requires a fixed batch depth"
                )
            if self.plan_id is None:
                raise RemoteMTPContractError(
                    "enforced scheduler advice requires a committed plan identity"
                )
            if len(self.candidate_ids) != len(self.preferred_request_ids):
                raise RemoteMTPContractError(
                    "enforced scheduler advice requires one candidate per request"
                )
        elif self.candidate_ids:
            raise RemoteMTPContractError(
                "observe-only scheduler advice cannot bind candidate ownership"
            )
        return self


class RemoteMTPSchedulerAdvisor(Protocol):
    """Hot-path advisor; implementations may inspect only process-local state."""

    def observe_decode_seal(
        self,
        requests: Sequence[RemoteMTPRequestView],
        *,
        seal_monotonic_ns: int,
    ) -> RemoteMTPSchedulerAdvice | None: ...


@dataclass(frozen=True, slots=True)
class RemoteMTPDecodeSlice:
    """Validated native-order partition for one enforced target execution."""

    selected_indices: tuple[int, ...]
    deferred_indices: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    plan_id: str
    snapshot_generation: int
    window_id: str


@dataclass(frozen=True, slots=True)
class RemoteMTPMixedBatchPlan:
    """Native-order remote/local row assignment for one fused verification."""

    remote_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    candidate_ids_by_row: tuple[str | None, ...]
    selected_depth: int
    plan_id: str
    snapshot_generation: int
    window_id: str


def build_remote_mtp_mixed_batch_plan(
    requests: Sequence[RemoteMTPRequestView],
    advice: RemoteMTPSchedulerAdvice,
    *,
    configured_depth: int,
) -> RemoteMTPMixedBatchPlan:
    """Bind an enforced remote subset without deferring its local-MTP peers.

    Every input row remains in the target batch.  Candidate ownership is
    represented in native row order so the hybrid worker can replace only the
    selected local proposals before launching one fixed-width verification.
    """

    request_tuple = tuple(request.validate() for request in requests)
    advice.validate()
    if not advice.enforce or advice.requests != request_tuple:
        raise RemoteMTPContractError(
            "mixed batching requires enforced advice for the exact current seal"
        )
    if (
        isinstance(configured_depth, bool)
        or not isinstance(configured_depth, int)
        or configured_depth <= 0
        or advice.compatible_depth is None
        or advice.compatible_depth > configured_depth
    ):
        raise RemoteMTPContractError(
            "mixed batch depth exceeds the configured verifier maximum"
        )
    request_ids = tuple(request.request_id for request in request_tuple)
    index_by_id = {request_id: index for index, request_id in enumerate(request_ids)}
    remote_indices = tuple(
        index_by_id[request_id] for request_id in advice.preferred_request_ids
    )
    if remote_indices != tuple(sorted(remote_indices)):
        raise RemoteMTPContractError(
            "mixed remote rows must preserve the native seal order"
        )
    candidate_ids_by_row: list[str | None] = [None] * len(request_tuple)
    for index, candidate_id in zip(remote_indices, advice.candidate_ids, strict=True):
        candidate_ids_by_row[index] = candidate_id
    remote_set = set(remote_indices)
    local_indices = tuple(
        index for index in range(len(request_tuple)) if index not in remote_set
    )
    assert advice.plan_id is not None
    assert advice.snapshot_generation is not None
    assert advice.window_id is not None
    return RemoteMTPMixedBatchPlan(
        remote_indices=remote_indices,
        local_indices=local_indices,
        candidate_ids_by_row=tuple(candidate_ids_by_row),
        selected_depth=advice.compatible_depth,
        plan_id=advice.plan_id,
        snapshot_generation=advice.snapshot_generation,
        window_id=advice.window_id,
    )


def build_remote_mtp_decode_slice(
    requests: Sequence[RemoteMTPRequestView],
    advice: RemoteMTPSchedulerAdvice,
    *,
    configured_depth: int,
) -> RemoteMTPDecodeSlice:
    """Convert exact advice into a total, native-order request partition."""

    request_tuple = tuple(request.validate() for request in requests)
    advice.validate()
    if not advice.enforce or advice.requests != request_tuple:
        raise RemoteMTPContractError(
            "decode slicing requires enforced advice for the exact current seal"
        )
    if not advice.preferred_request_ids:
        raise RemoteMTPContractError(
            "remote-only decode slicing requires a non-empty candidate subset"
        )
    if (
        isinstance(configured_depth, bool)
        or not isinstance(configured_depth, int)
        or configured_depth <= 0
        or advice.compatible_depth != configured_depth
    ):
        raise RemoteMTPContractError("decode slice depth differs from the verifier")
    request_ids = tuple(request.request_id for request in request_tuple)
    forced_request_ids = {
        request.request_id
        for request in request_tuple
        if request.max_service_gap_ns is not None
        and request.last_target_service_monotonic_ns is not None
        and request.service_completion_guard_ns is not None
        and advice.seal_monotonic_ns - request.last_target_service_monotonic_ns
        >= request.max_service_gap_ns - request.service_completion_guard_ns
    }
    if not forced_request_ids.issubset(advice.preferred_request_ids):
        raise RemoteMTPContractError(
            "decode slice omitted an engine-forced maximum-gap request"
        )
    preferred_set = set(advice.preferred_request_ids)
    native_preferred = tuple(
        request_id for request_id in request_ids if request_id in preferred_set
    )
    if native_preferred != advice.preferred_request_ids:
        raise RemoteMTPContractError("decode slice must preserve the native seal order")
    index_by_id = {request_id: index for index, request_id in enumerate(request_ids)}
    selected = tuple(index_by_id[request_id] for request_id in native_preferred)
    selected_set = set(selected)
    deferred = tuple(
        index for index in range(len(request_tuple)) if index not in selected_set
    )
    assert advice.plan_id is not None
    assert advice.snapshot_generation is not None
    assert advice.window_id is not None
    return RemoteMTPDecodeSlice(
        selected,
        deferred,
        advice.candidate_ids,
        advice.plan_id,
        advice.snapshot_generation,
        advice.window_id,
    )


@dataclass(frozen=True, slots=True)
class RemoteMTPCandidate:
    """One complete, already validated linear-chain proposal.

    ``prefix_index_digest`` is the digest of the standalone service's complete
    semantic prefix key.  SGLang treats it as opaque audit metadata; the
    installed adapter must compare it with its current target-owned PrefixKey
    before publishing this object.
    """

    candidate_id: str
    request: RemoteMTPRequestView
    prefix_index_digest: str
    token_ids: tuple[int, ...]

    def validate(self) -> RemoteMTPCandidate:
        self.request.validate()
        if not self.candidate_id:
            raise RemoteMTPContractError("candidate_id must be non-empty")
        if len(self.prefix_index_digest) != 64:
            raise RemoteMTPContractError(
                "prefix_index_digest must be a lowercase SHA-256 hex digest"
            )
        try:
            int(self.prefix_index_digest, 16)
        except ValueError as exc:
            raise RemoteMTPContractError(
                "prefix_index_digest must be a lowercase SHA-256 hex digest"
            ) from exc
        if self.prefix_index_digest != self.prefix_index_digest.lower():
            raise RemoteMTPContractError(
                "prefix_index_digest must be a lowercase SHA-256 hex digest"
            )
        if not self.token_ids:
            raise RemoteMTPContractError("token_ids must contain at least one token")
        for token_id in self.token_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise RemoteMTPContractError(
                    "token_ids must contain only non-negative integers"
                )
        return self

    @property
    def depth(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class RemoteMTPBatchClaim:
    """Atomic claim of one fixed-depth proposal for every request in a batch."""

    requests: tuple[RemoteMTPRequestView, ...]
    candidate_ids: tuple[str, ...]
    prefix_index_digests: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    depth: int

    def validate(self) -> RemoteMTPBatchClaim:
        if not self.requests:
            raise RemoteMTPContractError("a batch claim must not be empty")
        width = len(self.requests)
        if len(self.candidate_ids) != width:
            raise RemoteMTPContractError("candidate_ids do not cover the batch")
        if len(self.prefix_index_digests) != width:
            raise RemoteMTPContractError("prefix_index_digests do not cover the batch")
        if len(self.token_ids) != width:
            raise RemoteMTPContractError("token_ids do not cover the batch")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth <= 0
        ):
            raise RemoteMTPContractError("depth must be a positive integer")
        if any(len(tokens) != self.depth for tokens in self.token_ids):
            raise RemoteMTPContractError("every candidate must have the claimed depth")
        if len(set(self.candidate_ids)) != width:
            raise RemoteMTPContractError("candidate_ids must be unique within a batch")
        for request in self.requests:
            request.validate()
        return self


@dataclass(frozen=True, slots=True)
class RemoteMTPVerificationOutcome:
    """CPU-resolved target outcome for one previously claimed candidate."""

    candidate_id: str
    request: RemoteMTPRequestView
    prefix_index_digest: str
    candidate_token_ids: tuple[int, ...]
    accepted_draft_count: int
    committed_token_ids: tuple[int, ...]

    def validate(self) -> RemoteMTPVerificationOutcome:
        self.request.validate()
        if not self.candidate_id:
            raise RemoteMTPContractError("candidate_id must be non-empty")
        if len(self.prefix_index_digest) != 64:
            raise RemoteMTPContractError(
                "prefix_index_digest must be a SHA-256 hex digest"
            )
        try:
            int(self.prefix_index_digest, 16)
        except ValueError as exc:
            raise RemoteMTPContractError(
                "prefix_index_digest must be a SHA-256 hex digest"
            ) from exc
        if not self.candidate_token_ids:
            raise RemoteMTPContractError(
                "candidate_token_ids must contain at least one token"
            )
        if (
            isinstance(self.accepted_draft_count, bool)
            or not isinstance(self.accepted_draft_count, int)
            or not 0 <= self.accepted_draft_count <= len(self.candidate_token_ids)
        ):
            raise RemoteMTPContractError(
                "accepted_draft_count must be within the candidate depth"
            )
        for field_name, token_ids in (
            ("candidate_token_ids", self.candidate_token_ids),
            ("committed_token_ids", self.committed_token_ids),
        ):
            for token_id in token_ids:
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                ):
                    raise RemoteMTPContractError(
                        f"{field_name} must contain non-negative integers"
                    )
        return self


@dataclass(frozen=True, slots=True)
class RemoteMTPTargetSettlement:
    """CPU-authoritative committed run paired with one feature export row.

    Unlike ``RemoteMTPVerificationOutcome``, this record also covers prefill
    and AR fallback.  The background exporter needs every target commit to
    keep the remote NextN state synchronized even when no remote candidate was
    used.  Empty committed runs are not published: retracted/finished rows are
    discarded by the result processor before this boundary.
    """

    feature_batch_id: str
    phase: str
    request: RemoteMTPRequestView
    executed_source: str
    committed_token_ids: tuple[int, ...]
    candidate_id: str | None = None
    prefix_index_digest: str | None = None
    candidate_token_ids: tuple[int, ...] = ()
    verified_depth: int = 0
    accepted_draft_count: int = 0
    fallback_reason: str | None = None
    plan_id: str | None = None
    snapshot_generation: int | None = None
    window_id: str | None = None

    def validate(self) -> RemoteMTPTargetSettlement:
        if not self.feature_batch_id:
            raise RemoteMTPContractError("feature_batch_id must be non-empty")
        if self.phase not in ("prefill", "verify"):
            raise RemoteMTPContractError("settlement phase must be prefill or verify")
        self.request.validate()
        if self.executed_source not in (
            "remote_mtp",
            "local_mtp",
            "autoregressive",
        ):
            raise RemoteMTPContractError("settlement source is unsupported")
        if not self.committed_token_ids:
            raise RemoteMTPContractError(
                "settlement must contain target-committed tokens"
            )
        for field_name, token_ids in (
            ("candidate_token_ids", self.candidate_token_ids),
            ("committed_token_ids", self.committed_token_ids),
        ):
            for token_id in token_ids:
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                ):
                    raise RemoteMTPContractError(
                        f"{field_name} must contain non-negative integers"
                    )
        if (
            isinstance(self.verified_depth, bool)
            or not isinstance(self.verified_depth, int)
            or self.verified_depth < 0
            or isinstance(self.accepted_draft_count, bool)
            or not isinstance(self.accepted_draft_count, int)
            or self.accepted_draft_count < 0
            or self.accepted_draft_count > self.verified_depth
        ):
            raise RemoteMTPContractError(
                "settlement verification counts are inconsistent"
            )
        if len(self.committed_token_ids) != self.accepted_draft_count + 1:
            raise RemoteMTPContractError(
                "settlement commit must be accepted drafts plus one terminal token"
            )
        if self.executed_source == "remote_mtp":
            if not self.candidate_id or self.prefix_index_digest is None:
                raise RemoteMTPContractError(
                    "remote settlement requires candidate and prefix identities"
                )
            if not self.candidate_token_ids:
                raise RemoteMTPContractError(
                    "remote settlement requires the verified candidate chain"
                )
            if self.accepted_draft_count > len(self.candidate_token_ids):
                raise RemoteMTPContractError(
                    "accepted draft count exceeds the candidate depth"
                )
            if self.verified_depth != len(self.candidate_token_ids):
                raise RemoteMTPContractError(
                    "remote verified depth differs from its candidate chain"
                )
            if self.fallback_reason is not None:
                raise RemoteMTPContractError(
                    "remote settlement cannot carry a fallback reason"
                )
        elif self.executed_source == "local_mtp":
            if (
                self.candidate_id is not None
                or self.prefix_index_digest is not None
                or self.candidate_token_ids
                or self.verified_depth <= 0
                or self.fallback_reason is None
            ):
                raise RemoteMTPContractError(
                    "local MTP settlement has invalid local fallback evidence"
                )
        else:
            if (
                self.candidate_id is not None
                or self.prefix_index_digest is not None
                or self.candidate_token_ids
                or self.verified_depth != 0
                or self.accepted_draft_count != 0
            ):
                raise RemoteMTPContractError(
                    "autoregressive settlement cannot claim a remote candidate"
                )
        plan_identity = (self.plan_id, self.snapshot_generation, self.window_id)
        if any(value is not None for value in plan_identity):
            if not all(value is not None for value in plan_identity):
                raise RemoteMTPContractError(
                    "settlement must carry complete target plan identity"
                )
            if self.executed_source != "remote_mtp":
                raise RemoteMTPContractError(
                    "only a remote settlement can complete a target plan"
                )
            if (
                not self.plan_id
                or isinstance(self.snapshot_generation, bool)
                or not isinstance(self.snapshot_generation, int)
                or self.snapshot_generation < 0
                or not self.window_id
            ):
                raise RemoteMTPContractError(
                    "settlement target plan identity is invalid"
                )
        if self.prefix_index_digest is not None:
            if len(self.prefix_index_digest) != 64:
                raise RemoteMTPContractError(
                    "prefix_index_digest must be a lowercase SHA-256 digest"
                )
            try:
                int(self.prefix_index_digest, 16)
            except ValueError as exc:
                raise RemoteMTPContractError(
                    "prefix_index_digest must be a lowercase SHA-256 digest"
                ) from exc
            if self.prefix_index_digest != self.prefix_index_digest.lower():
                raise RemoteMTPContractError(
                    "prefix_index_digest must be a lowercase SHA-256 digest"
                )
        return self


@dataclass(frozen=True, slots=True)
class RemoteMTPTargetFeatureBatch:
    """GPU-resident target result offered to a bounded background exporter.

    Tensor fields are opaque here to keep the engine seam independent of a
    particular transport.  The installed bridge owns stream/event handling,
    pinned-host copies, accepted-path selection, and conversion to the
    standalone service's exact feature ABI.
    """

    feature_batch_id: str
    phase: str
    requests: tuple[RemoteMTPRequestView, ...]
    target_hidden_states: object
    next_token_ids: object
    accept_lens: object | None
    new_seq_lens: object
    claim: RemoteMTPBatchClaim | None
    fallback_reason: str | None
    input_token_ids: object | None
    extend_lens: tuple[int, ...] | None
    prefix_lens: tuple[int, ...] | None
    row_stride: int
    completion_event: object | None

    def validate(self) -> RemoteMTPTargetFeatureBatch:
        if not self.feature_batch_id:
            raise RemoteMTPContractError("feature_batch_id must be non-empty")
        if self.phase not in ("prefill", "verify"):
            raise RemoteMTPContractError("phase must be 'prefill' or 'verify'")
        if not self.requests:
            raise RemoteMTPContractError("feature export requests must not be empty")
        for request in self.requests:
            request.validate()
        if self.target_hidden_states is None:
            raise RemoteMTPContractError("target_hidden_states must be present")
        if self.next_token_ids is None:
            raise RemoteMTPContractError("next_token_ids must be present")
        if self.new_seq_lens is None:
            raise RemoteMTPContractError("new_seq_lens must be present")
        if self.phase == "verify" and self.accept_lens is None:
            raise RemoteMTPContractError("verify exports require accept_lens")
        if self.phase == "prefill" and self.accept_lens is not None:
            raise RemoteMTPContractError("prefill exports must not carry accept_lens")
        if (
            isinstance(self.row_stride, bool)
            or not isinstance(self.row_stride, int)
            or self.row_stride <= 0
        ):
            raise RemoteMTPContractError("row_stride must be a positive integer")
        if self.phase == "prefill":
            if self.input_token_ids is None:
                raise RemoteMTPContractError("prefill exports require input_token_ids")
            if (
                self.extend_lens is None
                or self.prefix_lens is None
                or len(self.extend_lens) != len(self.requests)
                or len(self.prefix_lens) != len(self.requests)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (*self.extend_lens, *self.prefix_lens)
                )
            ):
                raise RemoteMTPContractError(
                    "prefill exports require bounded per-request span metadata"
                )
        elif (
            self.input_token_ids is not None
            or self.extend_lens is not None
            or self.prefix_lens is not None
        ):
            raise RemoteMTPContractError(
                "verify exports cannot carry prefill-only span metadata"
            )
        if self.claim is not None:
            self.claim.validate()
            if self.claim.requests != self.requests:
                raise RemoteMTPContractError(
                    "feature export claim does not cover the exported requests"
                )
        return self


class RemoteMTPCandidateSource(Protocol):
    """Hot-path interface implemented by the standalone target adapter.

    Implementations must never wait for network, disk, a condition variable,
    or candidate production.  Returning ``None`` means immediate local
    fallback for this complete target batch.
    """

    def try_claim_batch(
        self,
        requests: Sequence[RemoteMTPRequestView],
        *,
        depth: int,
        candidate_ids: Sequence[str] | None = None,
        plan_id: str | None = None,
        snapshot_generation: int | None = None,
        window_id: str | None = None,
    ) -> RemoteMTPBatchClaim | None: ...


class RemoteMTPOutcomeSink(Protocol):
    """Optional nonblocking outcome method implemented by the target adapter."""

    def offer_verification_outcomes(
        self,
        outcomes: Sequence[RemoteMTPVerificationOutcome],
    ) -> bool: ...


class RemoteMTPFeatureSink(Protocol):
    """Optional nonblocking feature-export method on the target adapter."""

    def offer_target_feature_batch(
        self,
        feature_batch: RemoteMTPTargetFeatureBatch,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RemoteMTPMailboxStats:
    published: int
    claimed: int
    duplicate_rejected: int
    overload_rejected: int
    incompatible_batch_misses: int
    resident: int


class BoundedRemoteMTPMailbox(RemoteMTPCandidateSource):
    """Thread-safe, bounded, exact-match candidate ingress.

    Admission is explicit: the mailbox never evicts an older proposal to make
    room for a newer one.  Batch claim is all-or-nothing, so a partially ready
    target batch cannot consume candidates that would then be unavailable to a
    later candidate-aware batch.
    """

    def __init__(self, capacity: int = 4096):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise RemoteMTPContractError("capacity must be a positive integer")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._by_candidate_id: OrderedDict[str, RemoteMTPCandidate] = OrderedDict()
        self._by_request: dict[RemoteMTPRequestView, str] = {}
        self._published = 0
        self._claimed = 0
        self._duplicate_rejected = 0
        self._overload_rejected = 0
        self._incompatible_batch_misses = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def offer(self, candidate: RemoteMTPCandidate) -> bool:
        candidate.validate()
        with self._lock:
            if (
                candidate.candidate_id in self._by_candidate_id
                or candidate.request in self._by_request
            ):
                self._duplicate_rejected += 1
                return False
            if len(self._by_candidate_id) >= self._capacity:
                self._overload_rejected += 1
                return False
            self._by_candidate_id[candidate.candidate_id] = candidate
            self._by_request[candidate.request] = candidate.candidate_id
            self._published += 1
            return True

    def try_claim_batch(
        self,
        requests: Sequence[RemoteMTPRequestView],
        *,
        depth: int,
        candidate_ids: Sequence[str] | None = None,
        plan_id: str | None = None,
        snapshot_generation: int | None = None,
        window_id: str | None = None,
    ) -> RemoteMTPBatchClaim | None:
        request_tuple = tuple(request.validate() for request in requests)
        if not request_tuple:
            raise RemoteMTPContractError("requests must not be empty")
        if len(set(request_tuple)) != len(request_tuple):
            raise RemoteMTPContractError("requests must be unique within a batch")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise RemoteMTPContractError("depth must be a positive integer")
        plan_values = (plan_id, snapshot_generation, window_id)
        if any(value is not None for value in plan_values) and not all(
            value is not None for value in plan_values
        ):
            raise RemoteMTPContractError(
                "planned claim must carry complete plan identity"
            )
        if plan_id is not None and (
            not plan_id
            or isinstance(snapshot_generation, bool)
            or not isinstance(snapshot_generation, int)
            or snapshot_generation < 0
            or not window_id
        ):
            raise RemoteMTPContractError("planned claim identity is invalid")
        planned_ids = tuple(candidate_ids) if candidate_ids is not None else None
        if planned_ids is not None and (
            len(planned_ids) != len(request_tuple)
            or len(set(planned_ids)) != len(planned_ids)
            or any(not value for value in planned_ids)
        ):
            raise RemoteMTPContractError(
                "planned candidate IDs must uniquely cover the request batch"
            )

        with self._lock:
            candidates: list[RemoteMTPCandidate] = []
            for index, request in enumerate(request_tuple):
                candidate_id = (
                    planned_ids[index]
                    if planned_ids is not None
                    else self._by_request.get(request)
                )
                candidate = (
                    self._by_candidate_id.get(candidate_id)
                    if candidate_id is not None
                    else None
                )
                if (
                    candidate is None
                    or candidate.request != request
                    or candidate.depth != depth
                ):
                    self._incompatible_batch_misses += 1
                    return None
                candidates.append(candidate)

            # The complete coverage check above runs before the first mutation.
            for candidate in candidates:
                del self._by_candidate_id[candidate.candidate_id]
                del self._by_request[candidate.request]
            self._claimed += len(candidates)

            return RemoteMTPBatchClaim(
                requests=request_tuple,
                candidate_ids=tuple(item.candidate_id for item in candidates),
                prefix_index_digests=tuple(
                    item.prefix_index_digest for item in candidates
                ),
                token_ids=tuple(item.token_ids for item in candidates),
                depth=depth,
            ).validate()

    def discard_request(self, request: RemoteMTPRequestView) -> bool:
        request.validate()
        with self._lock:
            candidate_id = self._by_request.pop(request, None)
            if candidate_id is None:
                return False
            self._by_candidate_id.pop(candidate_id, None)
            return True

    def stats(self) -> RemoteMTPMailboxStats:
        with self._lock:
            return RemoteMTPMailboxStats(
                published=self._published,
                claimed=self._claimed,
                duplicate_rejected=self._duplicate_rejected,
                overload_rejected=self._overload_rejected,
                incompatible_batch_misses=self._incompatible_batch_misses,
                resident=len(self._by_candidate_id),
            )


RemoteMTPCandidateSourceFactory = Callable[[object, int], RemoteMTPCandidateSource]
RemoteMTPSchedulerAdvisorFactory = Callable[
    [object, int], RemoteMTPSchedulerAdvisor | None
]

_candidate_source_factory_lock = threading.Lock()
_candidate_source_factory: RemoteMTPCandidateSourceFactory | None = None
_scheduler_advisor_factory_lock = threading.Lock()
_scheduler_advisor_factory: RemoteMTPSchedulerAdvisorFactory | None = None


def install_remote_mtp_candidate_source_factory(
    factory: RemoteMTPCandidateSourceFactory,
) -> None:
    """Install the process-local adapter factory before scheduler startup.

    A second, different installation is rejected so two integrations cannot
    silently race for ownership of target candidate ingress.
    """

    if not callable(factory):
        raise RemoteMTPContractError("factory must be callable")
    global _candidate_source_factory
    with _candidate_source_factory_lock:
        if (
            _candidate_source_factory is not None
            and _candidate_source_factory is not factory
        ):
            raise RemoteMTPContractError(
                "a remote MTP candidate source factory is already installed"
            )
        _candidate_source_factory = factory


def create_remote_mtp_candidate_source(
    server_args: object,
    gpu_id: int,
) -> RemoteMTPCandidateSource:
    with _candidate_source_factory_lock:
        factory = _candidate_source_factory
    if factory is None:
        # Missing integration is a normal failure-independent mode: all
        # scheduling seals take the local fallback path.
        return BoundedRemoteMTPMailbox()
    source = factory(server_args, gpu_id)
    # Protocols with data methods are intentionally not runtime-checkable.
    # Validate the one hot-path method explicitly instead.
    if not callable(getattr(source, "try_claim_batch", None)):
        raise RemoteMTPContractError(
            "remote MTP candidate source lacks try_claim_batch"
        )
    return source


def install_remote_mtp_scheduler_advisor_factory(
    factory: RemoteMTPSchedulerAdvisorFactory,
) -> None:
    """Install one process-local, nonblocking scheduler advisor factory."""

    if not callable(factory):
        raise RemoteMTPContractError("scheduler advisor factory must be callable")
    global _scheduler_advisor_factory
    with _scheduler_advisor_factory_lock:
        if (
            _scheduler_advisor_factory is not None
            and _scheduler_advisor_factory is not factory
        ):
            raise RemoteMTPContractError(
                "a remote MTP scheduler advisor factory is already installed"
            )
        _scheduler_advisor_factory = factory


def create_remote_mtp_scheduler_advisor(
    server_args: object,
    gpu_id: int,
) -> RemoteMTPSchedulerAdvisor | None:
    with _scheduler_advisor_factory_lock:
        factory = _scheduler_advisor_factory
    if factory is None:
        return None
    advisor = factory(server_args, gpu_id)
    if advisor is not None and not callable(
        getattr(advisor, "observe_decode_seal", None)
    ):
        raise RemoteMTPContractError(
            "remote MTP scheduler advisor lacks observe_decode_seal"
        )
    return advisor


def _reset_remote_mtp_candidate_source_factory_for_test() -> None:
    global _candidate_source_factory, _scheduler_advisor_factory
    with _candidate_source_factory_lock:
        _candidate_source_factory = None
    with _scheduler_advisor_factory_lock:
        _scheduler_advisor_factory = None
