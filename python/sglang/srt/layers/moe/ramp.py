from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from time import time_ns
from typing import Any, Mapping, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

logger = logging.getLogger(__name__)


def ramp_enabled() -> bool:
    return envs.SGLANG_MOE_RAMP_ENABLE.get()


def ramp_log_interval() -> int:
    return envs.SGLANG_MOE_RAMP_LOG_INTERVAL.get()


def ramp_histogram_path() -> Optional[str]:
    return envs.SGLANG_MOE_RAMP_HISTOGRAM_PATH.get()


def ramp_histogram_interval() -> int:
    return envs.SGLANG_MOE_RAMP_HISTOGRAM_INTERVAL.get()


def ramp_profile_json() -> Optional[str]:
    return envs.SGLANG_MOE_RAMP_PROFILE_JSON.get()


def ramp_profile_path() -> Optional[str]:
    return envs.SGLANG_MOE_RAMP_PROFILE_PATH.get()


@dataclass
class RampProviderSelection:
    provider: Optional[str]
    source: str
    bucket: Optional[str] = None
    num_tokens: Optional[int] = None


@dataclass
class RampRoutingStats:
    """GPU-resident MoE routing summary for RaMP-style runner selection.

    The fields intentionally avoid CPU synchronization. Call ``to_log_dict`` only on
    throttled paths because it materializes scalar tensors for human-readable logs.
    """

    total_assignments: int
    active_experts: int
    tokens_per_expert: torch.Tensor
    max_tokens_per_expert: torch.Tensor
    mean_tokens_per_active_expert: float
    singleton_experts: torch.Tensor

    def to_log_dict(self, include_tokens_per_expert: bool = False) -> dict[str, Any]:
        max_tokens = int(self.max_tokens_per_expert.item())
        singleton_experts = int(self.singleton_experts.item())
        mean_tokens = self.mean_tokens_per_active_expert
        imbalance = max_tokens / mean_tokens if mean_tokens > 0 else 0.0
        singleton_frac = (
            singleton_experts / self.active_experts if self.active_experts > 0 else 0.0
        )
        fields = {
            "total_assignments": self.total_assignments,
            "active_experts": self.active_experts,
            "max_tokens_per_expert": max_tokens,
            "mean_tokens_per_active_expert": mean_tokens,
            "imbalance_ratio": imbalance,
            "singleton_frac": singleton_frac,
            "bucket": classify_ramp_bucket(
                total_assignments=self.total_assignments,
                active_experts=self.active_experts,
                max_tokens_per_expert=max_tokens,
                mean_tokens_per_active_expert=mean_tokens,
                singleton_frac=singleton_frac,
            ),
        }
        if include_tokens_per_expert:
            fields["tokens_per_expert"] = self.tokens_per_expert.detach().cpu().tolist()
        return fields


def classify_ramp_bucket(
    *,
    total_assignments: int,
    active_experts: int,
    max_tokens_per_expert: int,
    mean_tokens_per_active_expert: float,
    singleton_frac: float,
) -> str:
    """Coarse initial RaMP buckets.

    These buckets do not change execution yet. They establish stable labels for
    tracing/replay and can later map to concrete MoeRunnerCore variants.
    """

    if active_experts == 0 or total_assignments == 0:
        return "empty"

    imbalance = (
        max_tokens_per_expert / mean_tokens_per_active_expert
        if mean_tokens_per_active_expert > 0
        else 0.0
    )
    if total_assignments <= 64 and imbalance >= 4.0:
        return "decode_tiny_extreme_skew"
    if singleton_frac >= 0.5:
        return "many_tiny_experts"
    if total_assignments <= 256:
        return "decode_medium"
    return "prefill_or_large_batch"


@lru_cache(maxsize=1)
def _load_ramp_profile() -> dict[str, Any]:
    raw = ramp_profile_json()
    if raw:
        return json.loads(raw)

    path = ramp_profile_path()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _selection_with_provider_policy(
    profile: dict[str, Any],
    *,
    provider: Optional[str],
    source: str,
    bucket: Optional[str],
    num_tokens: Optional[int],
) -> RampProviderSelection:
    provider_fallbacks = profile.get("provider_fallbacks") or {}
    if provider in provider_fallbacks:
        return RampProviderSelection(
            provider=provider_fallbacks[provider],
            source=f"{source}_provider_fallback",
            bucket=bucket,
            num_tokens=num_tokens,
        )
    return RampProviderSelection(
        provider=provider,
        source=source,
        bucket=bucket,
        num_tokens=num_tokens,
    )


def select_ramp_provider(
    stats: Optional[RampRoutingStats],
    *,
    layer_id: Optional[int],
    top_k: Optional[int],
) -> RampProviderSelection:
    if stats is None:
        return RampProviderSelection(provider=None, source="no_stats")

    profile = _load_ramp_profile()
    entries = profile.get("entries") or []
    fallback_provider = profile.get("fallback_provider")
    if not entries and not fallback_provider:
        return RampProviderSelection(provider=None, source="no_entries")

    fields = stats.to_log_dict()
    bucket = fields["bucket"]
    num_tokens = stats.total_assignments // max(top_k or 1, 1)
    selector_keys = ("layer_id", "bucket", "top_k", "num_tokens")

    if not entries:
        return _selection_with_provider_policy(
            profile,
            provider=fallback_provider,
            source="fallback_provider",
            bucket=bucket,
            num_tokens=num_tokens,
        )

    # Exact match first.
    for entry in entries:
        winner = entry.get("winner")
        if not winner:
            continue
        if entry.get("layer_id", layer_id) != layer_id:
            continue
        if entry.get("bucket", bucket) != bucket:
            continue
        if entry.get("top_k", top_k) != top_k:
            continue
        if entry.get("num_tokens", num_tokens) != num_tokens:
            continue
        if all(key not in entry for key in selector_keys):
            source = "wildcard"
        elif all(key in entry for key in selector_keys):
            source = "exact"
        else:
            source = "profile_match"
        return _selection_with_provider_policy(
            profile,
            provider=winner,
            source=source,
            bucket=bucket,
            num_tokens=num_tokens,
        )

    # Coarse bucket-level fallback for initial RaMP-lite experiments.
    bucket_votes: dict[str, int] = {}
    for entry in entries:
        if entry.get("bucket") != bucket:
            continue
        winner = entry.get("winner")
        if winner:
            bucket_votes[winner] = bucket_votes.get(winner, 0) + 1
    if not bucket_votes:
        if fallback_provider:
            return _selection_with_provider_policy(
                profile,
                provider=fallback_provider,
                source="fallback_provider",
                bucket=bucket,
                num_tokens=num_tokens,
            )
        return RampProviderSelection(
            provider=None,
            source="miss",
            bucket=bucket,
            num_tokens=num_tokens,
        )
    return _selection_with_provider_policy(
        profile,
        provider=max(bucket_votes.items(), key=lambda item: item[1])[0],
        source="bucket_fallback",
        bucket=bucket,
        num_tokens=num_tokens,
    )


def maybe_log_ramp_selection(
    selection: RampProviderSelection,
    *,
    counters: Mapping[str, int],
    call_count: int,
    layer_id: Optional[int],
    runner_backend: str,
) -> None:
    interval = ramp_log_interval()
    if interval <= 0 or call_count % interval != 0:
        return

    logger.info(
        "RaMP MoE selector stats layer=%s backend=%s provider=%s source=%s "
        "bucket=%s num_tokens=%s counters=%s",
        layer_id,
        runner_backend,
        selection.provider,
        selection.source,
        selection.bucket,
        selection.num_tokens,
        dict(sorted(counters.items())),
    )


def extract_topk_ids(dispatch_output: Any) -> Optional[torch.Tensor]:
    topk_output = getattr(dispatch_output, "topk_output", None)
    if topk_output is not None:
        topk_ids = getattr(topk_output, "topk_ids", None)
        if isinstance(topk_ids, torch.Tensor):
            return topk_ids

    topk_ids = getattr(dispatch_output, "topk_ids", None)
    if isinstance(topk_ids, torch.Tensor):
        return topk_ids
    return None


def build_ramp_routing_stats(
    dispatch_output: Any, config: MoeRunnerConfig
) -> Optional[RampRoutingStats]:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None

    topk_ids = extract_topk_ids(dispatch_output)
    if topk_ids is None:
        return None

    flat_ids = topk_ids.reshape(-1)
    valid_ids = flat_ids[flat_ids >= 0]
    total_assignments = valid_ids.numel()
    if total_assignments == 0:
        return None

    num_experts = config.num_experts or config.num_local_experts
    if num_experts is None or num_experts <= 0:
        num_experts = int(valid_ids.max().item()) + 1

    counts = torch.bincount(valid_ids.to(torch.int64), minlength=num_experts)
    active_counts = counts[counts > 0]
    active_experts = active_counts.numel()
    if active_experts == 0:
        return None

    return RampRoutingStats(
        total_assignments=total_assignments,
        active_experts=active_experts,
        tokens_per_expert=counts,
        max_tokens_per_expert=active_counts.max(),
        mean_tokens_per_active_expert=total_assignments / active_experts,
        singleton_experts=(active_counts == 1).sum(),
    )


def maybe_log_ramp_stats(
    stats: Optional[RampRoutingStats],
    *,
    call_count: int,
    layer_id: Optional[int],
    runner_backend: str,
):
    if stats is None:
        return

    interval = ramp_log_interval()
    if interval <= 0 or call_count % interval != 0:
        return

    fields = stats.to_log_dict()
    logger.info(
        "RaMP MoE routing stats layer=%s backend=%s assignments=%s "
        "active_experts=%s max_tokens=%s mean_tokens=%.2f imbalance=%.2f "
        "singleton_frac=%.2f bucket=%s",
        layer_id,
        runner_backend,
        fields["total_assignments"],
        fields["active_experts"],
        fields["max_tokens_per_expert"],
        fields["mean_tokens_per_active_expert"],
        fields["imbalance_ratio"],
        fields["singleton_frac"],
        fields["bucket"],
    )


def _resolve_histogram_path(path_template: str, layer_id: Optional[int]) -> str:
    try:
        return path_template.format(
            pid=os.getpid(),
            layer_id=layer_id if layer_id is not None else "none",
        )
    except Exception:
        return path_template


def maybe_dump_ramp_histogram(
    stats: Optional[RampRoutingStats],
    *,
    call_count: int,
    layer_id: Optional[int],
    top_k: Optional[int],
    runner_backend: str,
    dispatch_format: str,
):
    """Append full expert histograms to JSONL when explicitly requested.

    This path performs CPU synchronization and file I/O, so it is intentionally
    guarded by ``SGLANG_MOE_RAMP_HISTOGRAM_PATH`` and should be used only for
    profiling/replay capture.
    """

    if stats is None:
        return

    path_template = ramp_histogram_path()
    if not path_template:
        return

    interval = ramp_histogram_interval()
    if interval <= 0 or call_count % interval != 0:
        return

    path = _resolve_histogram_path(path_template, layer_id)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fields = stats.to_log_dict(include_tokens_per_expert=True)
    record = {
        "time_ns": time_ns(),
        "pid": os.getpid(),
        "layer_id": layer_id,
        "top_k": top_k,
        "num_experts": len(fields["tokens_per_expert"]),
        "runner_backend": runner_backend,
        "dispatch_format": dispatch_format,
        **fields,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.warning("Failed to write RaMP histogram to %s: %s", path, exc)
