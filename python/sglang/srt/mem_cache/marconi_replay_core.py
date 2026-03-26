from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sglang.srt.mem_cache.marconi_cost_model import MarconiCostProfile
from sglang.srt.mem_cache.marconi_utils import normalize

MarconiReplayAction = Literal["full_leaf", "mamba_leaf", "mamba_internal"]


@dataclass(frozen=True)
class MarconiReplayCandidateMetrics:
    action: MarconiReplayAction
    replay_tokens: int
    recurrent_flops: float
    attention_flops: float
    ffn_flops: float
    memory_bytes: int
    last_access_time: float

    @property
    def total_flops(self) -> float:
        return self.recurrent_flops + self.attention_flops + self.ffn_flops


def build_marconi_candidate_metrics(
    *,
    profile: MarconiCostProfile,
    action: MarconiReplayAction,
    prefix_len: int,
    parent_prefix_len: int,
    local_kv_tokens: int,
    replay_tokens: int,
    last_access_time: float,
) -> MarconiReplayCandidateMetrics:
    recurrent_flops = 0.0
    attention_flops = 0.0
    ffn_flops = 0.0
    memory_bytes = 0

    recurrent_flops = profile.recurrent_flops(replay_tokens)
    if action in ("full_leaf", "mamba_leaf"):
        attention_flops = profile.attn_flops_delta(prefix_len, parent_prefix_len)
        ffn_flops = profile.ffn_flops_delta(prefix_len, parent_prefix_len)
        memory_bytes = profile.kv_bytes(local_kv_tokens)
        memory_bytes += profile.recurrent_bytes(1)
    elif action == "mamba_internal":
        memory_bytes = profile.recurrent_bytes(1)
    else:
        raise ValueError(f"Unknown Marconi replay action: {action}")

    return MarconiReplayCandidateMetrics(
        action=action,
        replay_tokens=replay_tokens,
        recurrent_flops=recurrent_flops,
        attention_flops=attention_flops,
        ffn_flops=ffn_flops,
        memory_bytes=memory_bytes,
        last_access_time=last_access_time,
    )


def select_marconi_candidate_index(
    *,
    candidates: list[MarconiReplayCandidateMetrics],
    current_time: float,
    eff_weight: float,
    use_efficiency: bool,
) -> Optional[int]:
    if not candidates:
        return None
    recency_scores = []
    efficiency_scores = []
    for candidate in candidates:
        delta = current_time - candidate.last_access_time
        recency_scores.append(1.0 / delta if delta > 0 else 0.0)
        if use_efficiency and candidate.memory_bytes > 0:
            efficiency_scores.append(candidate.total_flops / candidate.memory_bytes)
        else:
            efficiency_scores.append(0.0)
    recency_scores = normalize(recency_scores)
    efficiency_scores = normalize(efficiency_scores)
    utilities = [
        recency + eff_weight * efficiency
        for recency, efficiency in zip(recency_scores, efficiency_scores)
    ]
    return utilities.index(min(utilities))
