from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.mem_cache.marconi_cost_model import MarconiCostProfile

DEFAULT_MARCONI_EFF_WEIGHT = 0.0
DEFAULT_MARCONI_BOOTSTRAP_MULTIPLIER = 5
DEFAULT_MARCONI_TUNING_INTERVAL = 500
DEFAULT_MARCONI_WEIGHT_GRID = tuple(round(0.1 * idx, 1) for idx in range(21))


@dataclass(frozen=True, kw_only=True)
class MarconiTuningConfig:
    enabled: bool = True
    bootstrap_multiplier: int = DEFAULT_MARCONI_BOOTSTRAP_MULTIPLIER
    tuning_interval: int = DEFAULT_MARCONI_TUNING_INTERVAL
    weight_grid: tuple[float, ...] = DEFAULT_MARCONI_WEIGHT_GRID


@dataclass(frozen=True, kw_only=True)
class MarconiConfig:
    enable: bool
    eviction_enabled: bool = False
    eff_weight: float = DEFAULT_MARCONI_EFF_WEIGHT
    cost_profile: Optional[MarconiCostProfile] = None
    tuning: MarconiTuningConfig = MarconiTuningConfig()

    @classmethod
    def enabled(
        cls,
        *,
        cost_profile: Optional[MarconiCostProfile] = None,
        eff_weight: Optional[float] = None,
        eviction_enabled: bool = False,
        tuning: Optional[MarconiTuningConfig] = None,
    ) -> "MarconiConfig":
        return cls(
            enable=True,
            eviction_enabled=eviction_enabled,
            eff_weight=(
                eff_weight if eff_weight is not None else DEFAULT_MARCONI_EFF_WEIGHT
            ),
            cost_profile=cost_profile,
            tuning=tuning if tuning is not None else MarconiTuningConfig(),
        )


def get_marconi_branch_align_interval(
    page_size: Optional[int] = None, *, align_interval: Optional[int] = None
) -> int:
    if align_interval is None or align_interval <= 0:
        raise ValueError("Marconi branch alignment must be a positive integer.")
    if page_size is not None and page_size > 0:
        if align_interval % page_size != 0:
            raise ValueError("Marconi branch alignment must be divisible by page_size.")
    return align_interval
