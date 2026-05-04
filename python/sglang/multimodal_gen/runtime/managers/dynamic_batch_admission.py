# SPDX-License-Identifier: Apache-2.0
"""Admission control for native diffusion batching.

Native diffusion batching is model, resolution, device, and implementation
dependent. The scheduler treats `--batching-max-size` as the public ceiling;
`--batching-config` can apply stricter caps for specific model and shape
combinations.
"""

from __future__ import annotations

import bisect
import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from difflib import get_close_matches
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

# Bounds per-profile state and defines the OOM recovery window.
_MEMORY_PROFILE_HISTORY_SIZE = 32

_BATCHING_RULE_KEYS = frozenset(
    {
        "model",
        "model_contains",
        "resolution",
        "device_memory_gb_min",
        "device_memory_gb_max",
        "offload",
        "max_batch_size",
        "max_cost",
        # Free-form provenance/benchmark metadata. It is intentionally ignored
        # by admission, but accepted so production configs can explain caps.
        "calibration",
    }
)


@dataclass(frozen=True)
class AdmissionLimit:
    """Effective batch size and cost caps after matching batching rules."""

    max_batch_size: int
    max_cost: float | None = None
    cap_reason: str | None = None

    def reject_reason(self, *, batch_size: int, batch_cost: float) -> str | None:
        if batch_size > self.max_batch_size:
            return self.cap_reason or f"config_cap:{self.max_batch_size}"
        if self.max_cost is not None and batch_cost > self.max_cost:
            return f"cost_budget:{batch_cost:.0f}>{self.max_cost:.0f}"
        return None

    def stop_reason_for_next_cost(self, next_batch_cost: float) -> str | None:
        if self.max_cost is not None and next_batch_cost > self.max_cost:
            return f"cost_budget_next:{next_batch_cost:.0f}>{self.max_cost:.0f}"
        return None


@dataclass(frozen=True)
class BatchingRule:
    """One user-provided batching admission rule loaded from batching config."""

    model: str | None = None
    model_contains: str | None = None
    resolution: str | None = None
    device_memory_gb_min: float | None = None
    device_memory_gb_max: float | None = None
    offload: bool | None = None
    max_batch_size: int = 1
    max_cost: float | None = None
    source: str = "user"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str) -> "BatchingRule":
        if not isinstance(data, dict):
            raise ValueError(
                f"batching config rule from {source} must be an object, "
                f"got {type(data).__name__}"
            )
        _validate_rule_keys(data, source=source)
        if "max_batch_size" not in data:
            raise ValueError("batching config rule requires max_batch_size")

        rule = cls(
            model=_optional_str(data.get("model")),
            model_contains=_optional_str(data.get("model_contains")),
            resolution=_optional_str(data.get("resolution")),
            device_memory_gb_min=_optional_float(data.get("device_memory_gb_min")),
            device_memory_gb_max=_optional_float(data.get("device_memory_gb_max")),
            offload=_optional_bool(data.get("offload")),
            max_batch_size=int(data["max_batch_size"]),
            max_cost=_optional_float(data.get("max_cost")),
            source=source,
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if self.model is not None and self.model_contains is not None:
            raise ValueError(
                "batching config rule cannot set both model and model_contains"
            )
        if self.model is None and self.model_contains is None:
            raise ValueError("batching config rule requires model or model_contains")
        if self.max_batch_size < 1:
            raise ValueError("batching config rule max_batch_size must be >= 1")
        if self.max_cost is not None and self.max_cost <= 0.0:
            raise ValueError("batching config rule max_cost must be > 0")
        if (
            self.device_memory_gb_min is not None
            and self.device_memory_gb_max is not None
            and self.device_memory_gb_min > self.device_memory_gb_max
        ):
            raise ValueError(
                "batching config rule device_memory_gb_min must be <= device_memory_gb_max"
            )

    def matches(
        self,
        *,
        model_path: str,
        resolution: str | None,
        device_memory_gb: float | None,
        offload: bool,
    ) -> bool:
        if self.model is not None and self.model != model_path:
            return False
        if self.model_contains is not None and self.model_contains not in model_path:
            return False
        if self.resolution not in (None, "*") and self.resolution != resolution:
            return False
        if self.offload is not None and self.offload != offload:
            return False
        if device_memory_gb is None:
            return True
        if (
            self.device_memory_gb_min is not None
            and device_memory_gb < self.device_memory_gb_min
        ):
            return False
        if (
            self.device_memory_gb_max is not None
            and device_memory_gb > self.device_memory_gb_max
        ):
            return False
        return True


@dataclass(frozen=True)
class MemoryObservation:
    batch_cost: float
    peak_memory_mb: float
    batch_size: int = 1


class MemoryProfile:
    """Observed peak memory for one runtime and request profile."""

    def __init__(self):
        self.successes: deque[MemoryObservation] = deque(
            maxlen=_MEMORY_PROFILE_HISTORY_SIZE
        )
        self.residual_factors: deque[float] = deque(maxlen=_MEMORY_PROFILE_HISTORY_SIZE)
        self._danger_cost: float | None = None
        self._recovery_cost: float | None = None
        self._recovery_successes = 0
        self._cache_dirty = True
        self._cached_distinct_cost_count = 0
        self._cached_max_batch_size = 0
        self._cached_residual_max = 1.0
        self._cached_monotone_points: list[tuple[float, float]] = []
        self._cached_monotone_costs: list[float] = []
        self._cached_regression: tuple[float, float] | None = None

    def observe_success(
        self, batch_cost: float, peak_memory_mb: float, *, batch_size: int = 1
    ) -> None:
        if batch_cost <= 0.0 or peak_memory_mb <= 0.0:
            return
        batch_size = max(1, int(batch_size))
        predicted_peak_mb = self._estimate_peak_memory_base(batch_cost)
        if predicted_peak_mb is not None and predicted_peak_mb > 0.0:
            self.residual_factors.append(max(1.0, peak_memory_mb / predicted_peak_mb))
        self.successes.append(MemoryObservation(batch_cost, peak_memory_mb, batch_size))
        self._cache_dirty = True
        self._maybe_clear_danger_cost(batch_cost)

    def observe_oom(self, batch_cost: float) -> None:
        if batch_cost > 0.0:
            if self._danger_cost is None or batch_cost < self._danger_cost:
                self._danger_cost = batch_cost
                self._recovery_cost = self._max_success_cost()
            self._recovery_successes = 0

    def known_oom_cost(self, batch_cost: float) -> float | None:
        if self._danger_cost is None:
            return None
        return self._danger_cost if batch_cost >= self._danger_cost else None

    def estimate_peak_memory_mb(
        self,
        batch_cost: float,
        *,
        safety_factor: float,
    ) -> float | None:
        peak_memory_mb = self._estimate_peak_memory_base(
            batch_cost, use_observed_floor=True
        )
        if peak_memory_mb is None:
            return None
        return peak_memory_mb * max(safety_factor, self._residual_safety_factor())

    def _estimate_peak_memory_base(
        self, batch_cost: float, *, use_observed_floor: bool = False
    ) -> float | None:
        self._rebuild_cache()
        points = self._cached_monotone_points
        if batch_cost <= 0.0 or not points:
            return None

        max_observed_cost = points[-1][0]
        if batch_cost <= max_observed_cost:
            return _interpolate_peak(points, self._cached_monotone_costs, batch_cost)
        extrapolated_peak = _extrapolate_peak(
            points, batch_cost, self._cached_regression
        )
        if extrapolated_peak is not None:
            return extrapolated_peak
        if use_observed_floor:
            return points[-1][1]
        return None

    def _maybe_clear_danger_cost(self, batch_cost: float) -> None:
        if self._danger_cost is None:
            return

        if batch_cost >= self._danger_cost:
            self._clear_danger_cost()
            return

        recovery_cost = self._recovery_cost or 0.0
        if batch_cost < recovery_cost:
            return

        self._recovery_successes += 1
        if self._recovery_successes >= self._recovery_successes_required():
            self._clear_danger_cost()

    def _clear_danger_cost(self) -> None:
        self._danger_cost = None
        self._recovery_cost = None
        self._recovery_successes = 0

    def _max_success_cost(self) -> float | None:
        if not self.successes:
            return None
        return max(obs.batch_cost for obs in self.successes)

    def _residual_safety_factor(self) -> float:
        self._rebuild_cache()
        return self._cached_residual_max

    def _recovery_successes_required(self) -> int:
        return self.successes.maxlen or _MEMORY_PROFILE_HISTORY_SIZE

    def distinct_success_cost_count(self) -> int:
        self._rebuild_cache()
        return self._cached_distinct_cost_count

    def max_success_batch_size(self) -> int:
        self._rebuild_cache()
        return self._cached_max_batch_size

    def _rebuild_cache(self) -> None:
        if not self._cache_dirty:
            return

        by_cost: dict[float, float] = {}
        max_batch_size = 0
        for obs in self.successes:
            by_cost[obs.batch_cost] = max(
                by_cost.get(obs.batch_cost, 0.0), obs.peak_memory_mb
            )
            max_batch_size = max(max_batch_size, obs.batch_size)

        points = _monotone_upper_points(sorted(by_cost.items()))
        self._cached_distinct_cost_count = len(by_cost)
        self._cached_max_batch_size = max_batch_size
        self._cached_residual_max = max(self.residual_factors, default=1.0)
        self._cached_monotone_points = points
        self._cached_monotone_costs = [cost for cost, _peak in points]
        self._cached_regression = _fit_peak_regression(points)
        self._cache_dirty = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": [
                {
                    "batch_cost": obs.batch_cost,
                    "peak_memory_mb": obs.peak_memory_mb,
                    "batch_size": obs.batch_size,
                }
                for obs in self.successes
            ],
            "residual_factors": list(self.residual_factors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryProfile":
        profile = cls()
        for item in data.get("successes", []):
            profile.successes.append(
                MemoryObservation(
                    batch_cost=float(item["batch_cost"]),
                    peak_memory_mb=float(item["peak_memory_mb"]),
                    batch_size=max(1, int(item.get("batch_size", 1))),
                )
            )
        for value in data.get("residual_factors", []):
            profile.residual_factors.append(max(1.0, float(value)))
        profile._cache_dirty = True
        return profile


class BatchAdmissionController:
    """Applies configured caps before adding requests to a batch."""

    def __init__(self, server_args: "ServerArgs", gpu_id: int):
        self._mode = getattr(server_args, "batching_mode", "dynamic")
        self._user_max_batch_size = max(1, int(server_args.batching_max_size))
        self._model_path = server_args.model_path
        self._offload = bool(server_args.dit_layerwise_offload)
        self._device_memory_gb = self._get_device_memory_gb(gpu_id)
        self._device_name = self._get_device_name(gpu_id)
        self._rules = load_batching_config(server_args.batching_config)
        self._pipeline_config = server_args.pipeline_config
        self._limit_cache: dict[str | None, AdmissionLimit] = {}
        self._runtime_memory_key = _build_runtime_memory_key(
            server_args,
            model_path=self._model_path,
            device_name=self._device_name,
            device_memory_gb=self._device_memory_gb,
        )
        self._lora_revision = 0
        self._memory_aware = bool(server_args.batching_memory_aware)
        self._memory_safety_factor = max(
            1.0, float(server_args.batching_memory_safety_factor)
        )
        self._memory_reserve_mb = max(
            0.0, float(server_args.batching_memory_reserve_gb) * 1024.0
        )
        self._gpu_id = gpu_id
        self._device_module = None
        self._memory_budget_missing_adjustment_logged = False
        self._memory_profiles: dict[tuple[Any, ...], MemoryProfile] = {}
        self._memory_profiles_dirty = False
        self._memory_profile_cache_path = self._resolve_memory_profile_cache_path(
            server_args
        )
        if self._memory_aware:
            self._load_memory_profiles()
        self._memory_budget_mb: float | None = None
        self.refresh_memory_budget()

        if self.enabled:
            logger.info(
                "Batch admission enabled: user_max=%d, device_memory=%.1fGiB, rules=%d, memory_aware=%s",
                self._user_max_batch_size,
                self._device_memory_gb or 0.0,
                len(self._rules),
                self._memory_aware,
            )

    @property
    def enabled(self) -> bool:
        return self._mode == "dynamic" and self._user_max_batch_size > 1

    def reject_reason_for_candidate(
        self, current_reqs: list[Req], candidate_req: Req
    ) -> str | None:
        if not self.enabled:
            return None
        head_req = current_reqs[0] if current_reqs else candidate_req
        batch_size = len(current_reqs) + 1
        batch_cost = self.estimate_batch_cost(current_reqs) + self._request_cost(
            candidate_req
        )
        limit = self.limit_for(head_req)
        reject_reason = limit.reject_reason(
            batch_size=batch_size,
            batch_cost=batch_cost,
        )
        if reject_reason is not None:
            return reject_reason
        return self._memory_reject_reason(
            batch_size=batch_size,
            batch_cost=batch_cost,
            memory_key=self._memory_key(head_req),
        )

    def batch_is_full(self, reqs: list[Req]) -> bool:
        """Return whether another roughly similar request would exceed the cap."""
        if not self.enabled or not reqs:
            return len(reqs) >= self._user_max_batch_size

        limit = self.limit_for(reqs[0])
        if len(reqs) >= limit.max_batch_size:
            return True

        next_cost = self.estimate_batch_cost(reqs) + self._request_cost(reqs[0])
        return (
            limit.max_cost is not None and next_cost > limit.max_cost
        ) or self._memory_reject_reason(
            batch_size=len(reqs) + 1,
            batch_cost=next_cost,
            memory_key=self._memory_key(reqs[0]),
        ) is not None

    def limit_reason_for_batch(self, reqs: list[Req]) -> str | None:
        if not self.enabled or not reqs:
            return None

        limit = self.limit_for(reqs[0])
        if len(reqs) >= limit.max_batch_size:
            return limit.cap_reason or f"config_cap:{limit.max_batch_size}"

        next_cost = self.estimate_batch_cost(reqs) + self._request_cost(reqs[0])
        return limit.stop_reason_for_next_cost(next_cost) or self._memory_reject_reason(
            batch_size=len(reqs) + 1,
            batch_cost=next_cost,
            memory_key=self._memory_key(reqs[0]),
            next_batch=True,
        )

    def max_admissible_batch_size(self, req: Req) -> int:
        limit = self.limit_for(req)
        max_batch_size = limit.max_batch_size
        if not self._memory_aware:
            return max_batch_size

        cost_per_req = self._request_cost(req)
        memory_key = self._memory_key(req)
        low = 1
        high = max_batch_size
        while low < high:
            mid = (low + high + 1) // 2
            batch_cost = cost_per_req * mid
            cost_reject = limit.max_cost is not None and batch_cost > limit.max_cost
            memory_reject = (
                self._memory_reject_reason(
                    batch_size=mid,
                    batch_cost=batch_cost,
                    memory_key=memory_key,
                )
                is not None
            )
            if cost_reject or memory_reject:
                high = mid - 1
            else:
                low = mid
        return low

    def limit_for(self, req: Req) -> AdmissionLimit:
        """Return the effective admission limit for the request's model and shape."""
        resolution = req.resolution_key
        cached_limit = self._limit_cache.get(resolution)
        if cached_limit is not None:
            return cached_limit

        rules = self._matching_rules(req)
        if not rules:
            limit = AdmissionLimit(max_batch_size=self._user_max_batch_size)
            self._limit_cache[resolution] = limit
            return limit

        config_cap = min(rule.max_batch_size for rule in rules)
        max_batch_size = min(self._user_max_batch_size, config_cap)
        cap_reason = (
            f"config_cap:{max_batch_size}"
            if max_batch_size < self._user_max_batch_size
            else None
        )
        costs = [rule.max_cost for rule in rules if rule.max_cost is not None]
        limit = AdmissionLimit(
            max_batch_size=max(1, max_batch_size),
            max_cost=min(costs) if costs else None,
            cap_reason=cap_reason,
        )
        self._limit_cache[resolution] = limit
        return limit

    def estimate_batch_cost(self, reqs: list[Req]) -> float:
        return sum(self._request_cost(req) for req in reqs)

    def _request_cost(self, req: Req) -> float:
        cached_cost = getattr(req, "_batch_admission_cost", None)
        if cached_cost is not None:
            return float(cached_cost)

        cost = float(self._pipeline_config.estimate_request_cost(req))
        setattr(req, "_batch_admission_cost", cost)
        return cost

    def refresh_memory_budget(self) -> None:
        if not self._memory_aware:
            return

        budget_mb = self._measure_memory_budget_mb()
        if budget_mb is None:
            return
        self._memory_budget_mb = budget_mb

    def observe_batch_result(
        self,
        reqs: list[Req],
        *,
        peak_memory_mb: float,
        error: str | None,
        is_oom: bool,
    ) -> None:
        if not self._memory_aware or not reqs:
            return

        profile = self._memory_profile_for(reqs[0])
        batch_cost = self.estimate_batch_cost(reqs)
        if error is None:
            profile.observe_success(batch_cost, peak_memory_mb, batch_size=len(reqs))
            if self._memory_profile_cache_path is not None:
                self._memory_profiles_dirty = True
        elif is_oom:
            profile.observe_oom(batch_cost)

    def save_memory_profiles_if_dirty(self) -> None:
        if not self._memory_aware or not self._memory_profiles_dirty:
            return
        self._save_memory_profiles_now()

    def invalidate_runtime_memory_profiles(self) -> None:
        """Drop profiles after mutable runtime state changes, such as LoRA updates."""
        self.save_memory_profiles_if_dirty()
        self._lora_revision += 1
        self._memory_profiles.clear()

    def _matching_rules(self, req: Req) -> list[BatchingRule]:
        return [
            rule
            for rule in self._rules
            if rule.matches(
                model_path=self._model_path,
                resolution=req.resolution_key,
                device_memory_gb=self._device_memory_gb,
                offload=self._offload,
            )
        ]

    def _memory_profile_for(self, req: Req) -> MemoryProfile:
        key = self._memory_key(req)
        profile = self._memory_profiles.get(key)
        if profile is None:
            profile = MemoryProfile()
            self._memory_profiles[key] = profile
        return profile

    def _memory_reject_reason(
        self,
        *,
        batch_size: int,
        batch_cost: float,
        memory_key: tuple[Any, ...],
        next_batch: bool = False,
    ) -> str | None:
        if not self._memory_aware:
            return None

        profile = self._memory_profiles.get(memory_key)
        if profile is not None:
            failed_cost = profile.known_oom_cost(batch_cost)
            if failed_cost is not None:
                prefix = "memory_oom_next" if next_batch else "memory_oom"
                return f"{prefix}:{batch_cost:.0f}>={failed_cost:.0f}"

        calibration_reject_reason = self._memory_calibration_reject_reason(
            profile, batch_size=batch_size, next_batch=next_batch
        )
        if calibration_reject_reason is not None:
            return calibration_reject_reason
        if profile is None:
            return None

        memory_budget_mb = self._memory_budget_mb
        if memory_budget_mb is None:
            return None

        peak_memory_mb = profile.estimate_peak_memory_mb(
            batch_cost,
            safety_factor=self._memory_safety_factor,
        )
        if peak_memory_mb is None or peak_memory_mb <= memory_budget_mb:
            return None

        prefix = "memory_budget_next" if next_batch else "memory_budget"
        return f"{prefix}:{peak_memory_mb:.0f}>{memory_budget_mb:.0f}MiB"

    def _memory_calibration_reject_reason(
        self,
        profile: MemoryProfile | None,
        *,
        batch_size: int,
        next_batch: bool,
    ) -> str | None:
        observed_costs = 0 if profile is None else profile.distinct_success_cost_count()
        max_observed_batch = 0 if profile is None else profile.max_success_batch_size()
        calibration_cap = max(observed_costs + 1, 2 * max_observed_batch)
        if batch_size <= calibration_cap:
            return None

        prefix = "memory_uncalibrated_next" if next_batch else "memory_uncalibrated"
        return f"{prefix}:{batch_size}>{calibration_cap}"

    def _measure_memory_budget_mb(self) -> float | None:
        if self._device_memory_gb is None:
            return None

        total_mb = self._device_memory_gb * 1024.0
        budget_mb = total_mb - self._memory_reserve_mb
        available_mb = self._get_available_memory_mb(self._gpu_id)
        reserved_mb = self._get_process_reserved_memory_mb(self._gpu_id)
        if available_mb is not None and reserved_mb is not None:
            external_used_mb = max(0.0, total_mb - available_mb - reserved_mb)
            budget_mb -= external_used_mb
        elif not self._memory_budget_missing_adjustment_logged:
            logger.debug(
                "Skipping external GPU memory adjustment for memory-aware batching; available=%s reserved=%s",
                available_mb,
                reserved_mb,
            )
            self._memory_budget_missing_adjustment_logged = True
        return max(0.0, budget_mb)

    def _memory_key(self, req: Req) -> tuple[Any, ...]:
        return (
            self._runtime_memory_key,
            ("lora_revision", self._lora_revision),
            self._request_memory_key(req),
        )

    def _request_memory_key(self, req: Req) -> tuple[Any, ...]:
        cached_key = getattr(req, "_batch_request_memory_key", None)
        if cached_key is not None:
            return cached_key

        key = self._build_request_memory_key(req)
        setattr(req, "_batch_request_memory_key", key)
        return key

    def _load_memory_profiles(self) -> None:
        path = self._memory_profile_cache_path
        if path is None:
            return
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("schema_version") != 1:
                return
            loaded = 0
            for item in payload.get("profiles", []):
                key = _tuple_key_value(item["key"])
                if not key or key[0] != self._runtime_memory_key:
                    continue
                profile = MemoryProfile.from_dict(item["profile"])
                if profile.successes:
                    self._memory_profiles[key] = profile
                    loaded += 1
            if loaded:
                logger.info("Loaded %d memory-aware batching profile(s)", loaded)
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning("Failed to load memory-aware batching profiles: %s", e)
            self._memory_profile_cache_path = None

    def _save_memory_profiles_now(self) -> None:
        path = self._memory_profile_cache_path
        if path is None:
            return
        profiles = [
            {"key": _json_key_value(key), "profile": profile.to_dict()}
            for key, profile in self._memory_profiles.items()
            if profile.successes
        ]
        if not profiles:
            self._memory_profiles_dirty = False
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Cache schema v1 stores successful observations and residual factors.
            payload = {"schema_version": 1, "profiles": profiles}
            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp-", suffix=".json", dir=os.path.dirname(path)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, sort_keys=True)
                os.replace(tmp_path, path)
                self._memory_profiles_dirty = False
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("Failed to save memory-aware batching profiles: %s", e)
            self._memory_profile_cache_path = None
            self._memory_profiles_dirty = False

    def _build_request_memory_key(self, req: Req) -> tuple[Any, ...]:
        return (
            ("resolution", req.resolution_key),
            ("num_frames", int(req.num_frames or 1)),
            ("num_outputs_per_prompt", int(req.num_outputs_per_prompt or 1)),
            ("max_sequence_length", req.max_sequence_length),
            ("prompt_template", _freeze_key_value(req.prompt_template)),
            ("enable_sequence_shard", req.enable_sequence_shard),
            ("sampling", _sampling_memory_key(req)),
            (
                "diffusers_kwargs",
                _freeze_key_value((req.extra or {}).get("diffusers_kwargs")),
            ),
        )

    @staticmethod
    def _get_device_memory_gb(gpu_id: int) -> float | None:
        try:
            return current_platform.get_device_total_memory(gpu_id) / BYTES_PER_GB
        except Exception:
            return None

    @staticmethod
    def _get_device_name(gpu_id: int) -> str | None:
        try:
            return str(current_platform.get_device_name(gpu_id))
        except Exception:
            return None

    def _resolve_memory_profile_cache_path(
        self, server_args: "ServerArgs"
    ) -> str | None:
        cache_setting = getattr(server_args, "batching_memory_profile_cache", "auto")
        if cache_setting in (None, ""):
            return None
        cache_root = (
            os.environ.get("SGLANG_DIFFUSION_BATCH_MEMORY_CACHE")
            if cache_setting == "auto"
            else str(cache_setting)
        )
        if not cache_root:
            cache_root = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "sglang",
                "diffusion_batch_memory",
            )
        key_hash = _stable_hash(_json_key_value(self._runtime_memory_key))
        return os.path.join(cache_root, f"{key_hash}.json")

    @staticmethod
    def _get_available_memory_mb(gpu_id: int) -> float | None:
        try:
            # Platform helper returns GiB here, while total memory returns bytes.
            return (
                current_platform.get_available_gpu_memory(gpu_id, empty_cache=False)
                * 1024.0
            )
        except Exception:
            return None

    def _get_process_reserved_memory_mb(self, gpu_id: int) -> float | None:
        try:
            if self._device_module is None:
                import torch

                self._device_module = torch.get_device_module()
            memory_reserved = getattr(self._device_module, "memory_reserved", None)
            if memory_reserved is None:
                return None
            return float(memory_reserved(gpu_id)) / (1024**2)
        except Exception:
            return None


def load_batching_config(path: str | None) -> list[BatchingRule]:
    if path is None:
        return []

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    source = os.path.abspath(path)
    entries = _config_entries(payload)
    rules = [BatchingRule.from_dict(entry, source=source) for entry in entries]
    if not rules:
        raise ValueError(f"batching config {source} does not contain any rules")
    return rules


def _config_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("schema_version") not in (None, 1):
        raise ValueError("batching config schema_version must be 1")
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entries: list[dict[str, Any]] = []
        for key, value in payload.items():
            if key == "schema_version" or not isinstance(value, dict):
                continue
            model, _sep, resolution = key.partition("|")
            entry = dict(value)
            if model:
                entry.setdefault("model", model)
            if resolution:
                entry.setdefault("resolution", resolution)
            entries.append(entry)
        return entries
    raise ValueError(
        "batching config must be a {'schema_version': 1, 'rules': [...]} object, "
        "a list of rules, or a mapping keyed by model|resolution"
    )


def _validate_rule_keys(data: dict[str, Any], *, source: str) -> None:
    unknown = sorted(set(data) - _BATCHING_RULE_KEYS)
    if not unknown:
        return

    hints = []
    for key in unknown:
        matches = get_close_matches(key, _BATCHING_RULE_KEYS, n=1)
        if matches:
            hints.append(f"{key!r} (did you mean {matches[0]!r}?)")
        else:
            hints.append(repr(key))
    raise ValueError(
        f"batching config rule from {source} contains unknown key(s): "
        f"{', '.join(hints)}"
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"cannot parse boolean batching config value: {value!r}")


def _build_runtime_memory_key(
    server_args: "ServerArgs",
    *,
    model_path: str,
    device_name: str | None,
    device_memory_gb: float | None,
) -> tuple[Any, ...]:
    config = server_args.pipeline_config
    return (
        ("model_path", model_path),
        ("model_id", getattr(server_args, "model_id", None)),
        ("backend", _freeze_key_value(getattr(server_args, "backend", None))),
        ("pipeline", _class_key(config)),
        ("device", device_name),
        ("device_memory_gb", device_memory_gb),
        ("attention", _attention_memory_key(server_args)),
        ("parallel", _parallel_memory_key(server_args)),
        ("precision", _precision_memory_key(config, server_args)),
        ("quantization", _quantization_memory_key(config, server_args)),
        ("offload", _offload_memory_key(server_args)),
        ("lora", _initial_lora_memory_key(server_args)),
        (
            "components",
            _freeze_key_value(getattr(server_args, "component_paths", None)),
        ),
        ("model_paths", _freeze_key_value(getattr(server_args, "model_paths", None))),
        (
            "cache_dit",
            _freeze_key_value(getattr(server_args, "cache_dit_config", None)),
        ),
        ("torch_compile", bool(getattr(server_args, "enable_torch_compile", False))),
    )


def _attention_memory_key(server_args: "ServerArgs") -> tuple[Any, ...]:
    return (
        ("backend", getattr(server_args, "attention_backend", None)),
        (
            "config",
            _freeze_key_value(getattr(server_args, "attention_backend_config", None)),
        ),
    )


def _parallel_memory_key(server_args: "ServerArgs") -> tuple[Any, ...]:
    return (
        ("num_gpus", getattr(server_args, "num_gpus", None)),
        ("tp_size", getattr(server_args, "tp_size", None)),
        ("sp_degree", getattr(server_args, "sp_degree", None)),
        ("ulysses_degree", getattr(server_args, "ulysses_degree", None)),
        ("ring_degree", getattr(server_args, "ring_degree", None)),
        ("dp_size", getattr(server_args, "dp_size", None)),
        ("dp_degree", getattr(server_args, "dp_degree", None)),
        ("enable_cfg_parallel", getattr(server_args, "enable_cfg_parallel", None)),
        ("hsdp_replicate_dim", getattr(server_args, "hsdp_replicate_dim", None)),
        ("hsdp_shard_dim", getattr(server_args, "hsdp_shard_dim", None)),
        ("disagg_role", _freeze_key_value(getattr(server_args, "disagg_role", None))),
        ("encoder_tp", getattr(server_args, "encoder_tp", None)),
        ("denoiser_tp", getattr(server_args, "denoiser_tp", None)),
        ("denoiser_sp", getattr(server_args, "denoiser_sp", None)),
        ("decoder_tp", getattr(server_args, "decoder_tp", None)),
    )


def _precision_memory_key(config: Any, server_args: "ServerArgs") -> tuple[Any, ...]:
    return (
        ("disable_autocast", getattr(server_args, "disable_autocast", None)),
        ("dit_precision", getattr(config, "dit_precision", None)),
        ("vae_precision", getattr(config, "vae_precision", None)),
        ("image_encoder_precision", getattr(config, "image_encoder_precision", None)),
        (
            "text_encoder_precisions",
            _freeze_key_value(getattr(config, "text_encoder_precisions", None)),
        ),
        ("audio_vae_precision", getattr(config, "audio_vae_precision", None)),
    )


def _quantization_memory_key(config: Any, server_args: "ServerArgs") -> tuple[Any, ...]:
    dit_config = getattr(config, "dit_config", None)
    return (
        (
            "transformer_weights_path",
            getattr(server_args, "transformer_weights_path", None),
        ),
        (
            "nunchaku_config",
            _freeze_key_value(getattr(server_args, "nunchaku_config", None)),
        ),
        ("dit_config", _class_key(dit_config)),
        (
            "dit_quant_config",
            _freeze_key_value(getattr(dit_config, "quant_config", None)),
        ),
    )


def _offload_memory_key(server_args: "ServerArgs") -> tuple[Any, ...]:
    return (
        ("dit_cpu_offload", getattr(server_args, "dit_cpu_offload", None)),
        ("dit_layerwise_offload", getattr(server_args, "dit_layerwise_offload", None)),
        (
            "dit_offload_prefetch_size",
            getattr(server_args, "dit_offload_prefetch_size", None),
        ),
        (
            "text_encoder_cpu_offload",
            getattr(server_args, "text_encoder_cpu_offload", None),
        ),
        (
            "image_encoder_cpu_offload",
            getattr(server_args, "image_encoder_cpu_offload", None),
        ),
        ("vae_cpu_offload", getattr(server_args, "vae_cpu_offload", None)),
        ("use_fsdp_inference", getattr(server_args, "use_fsdp_inference", None)),
        ("pin_cpu_memory", getattr(server_args, "pin_cpu_memory", None)),
        (
            "ltx2_two_stage_device_mode",
            getattr(server_args, "ltx2_two_stage_device_mode", None),
        ),
    )


def _initial_lora_memory_key(server_args: "ServerArgs") -> tuple[Any, ...]:
    return (
        ("lora_path", getattr(server_args, "lora_path", None)),
        ("lora_nickname", getattr(server_args, "lora_nickname", None)),
        ("lora_scale", getattr(server_args, "lora_scale", None)),
        ("lora_weight_name", getattr(server_args, "lora_weight_name", None)),
        (
            "lora_target_modules",
            _freeze_key_value(getattr(server_args, "lora_target_modules", None)),
        ),
    )


def _class_key(value: Any) -> str | None:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _freeze_key_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (f.name, _freeze_key_value(getattr(value, f.name, None)))
            for f in fields(value)
        )
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_key_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_key_value(v) for v in value), key=repr))
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
        return f"{module}.{qualname}" if module and qualname else repr(value)
    return repr(value)


def _json_key_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_key_value(item) for item in value]
    if isinstance(value, list):
        return [_json_key_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_key_value(v) for k, v in sorted(value.items())}
    return value


def _tuple_key_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (str(k), _tuple_key_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    return value


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:32]


def _sampling_memory_key(req: Req) -> tuple[Any, ...] | None:
    sp = req.sampling_params
    if sp is None:
        return None

    try:
        sp_fields = fields(sp)
    except TypeError:
        return repr(sp)

    return tuple(
        (f.name, _freeze_key_value(getattr(sp, f.name, None)))
        for f in sp_fields
        if not f.metadata.get("batch_sig_exclude", False)
    )


def _monotone_upper_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    adjusted: list[tuple[float, float]] = []
    running_peak = 0.0
    for cost, peak in points:
        running_peak = max(running_peak, peak)
        adjusted.append((cost, running_peak))
    return adjusted


def _interpolate_peak(
    points: list[tuple[float, float]], costs: list[float], batch_cost: float
) -> float:
    idx = bisect.bisect_left(costs, batch_cost)
    if idx == 0:
        return points[0][1]
    if idx == len(points):
        return points[-1][1]

    low_cost, low_peak = points[idx - 1]
    high_cost, high_peak = points[idx]
    fraction = (batch_cost - low_cost) / (high_cost - low_cost)
    return low_peak + fraction * (high_peak - low_peak)


def _extrapolate_peak(
    points: list[tuple[float, float]],
    batch_cost: float,
    regression: tuple[float, float] | None,
) -> float | None:
    if regression is None:
        return None

    slope, intercept = regression
    return max(points[-1][1], intercept + slope * batch_cost)


def _fit_peak_regression(
    points: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None

    mean_cost = sum(cost for cost, _peak in points) / len(points)
    mean_peak = sum(peak for _cost, peak in points) / len(points)
    denominator = sum((cost - mean_cost) ** 2 for cost, _peak in points)
    if denominator == 0.0:
        return None

    numerator = sum((cost - mean_cost) * (peak - mean_peak) for cost, peak in points)
    slope = max(0.0, numerator / denominator)
    intercept = mean_peak - slope * mean_cost
    return slope, intercept
