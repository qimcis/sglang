from __future__ import annotations

from math import isqrt
from typing import Any, Dict, Optional, Tuple

from sglang.srt.environ import envs


def _config_dict(config: Any) -> Dict[str, Any]:
    raw = getattr(config, "ndlinear_config", None)
    if raw is None:
        raw = getattr(config, "ndlinear", None)
    return raw if isinstance(raw, dict) else {}


def _targets_from_config(config: Any) -> set[str]:
    raw = _config_dict(config).get("targets", [])
    if isinstance(raw, str):
        return {raw}
    return {str(item) for item in raw}


def deepseek_v4_ndlinear_enabled(config: Any, target: str) -> bool:
    cfg = _config_dict(config)
    enabled = (
        bool(cfg.get("enabled", False)) or envs.SGLANG_ENABLE_DEEPSEEK_V4_NDLINEAR.get()
    )
    if not enabled:
        return False
    targets = _targets_from_config(config) | set(
        envs.SGLANG_DEEPSEEK_V4_NDLINEAR_TARGETS.get()
    )
    return not targets or "all" in targets or target in targets


def _shape_from_config(
    config: Any, target: str
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    cfg = _config_dict(config)
    modules = cfg.get("modules", {})
    target_cfg = modules.get(target, cfg.get(target, {}))
    if not isinstance(target_cfg, dict):
        return None
    input_shape = target_cfg.get("input_shape")
    output_shape = target_cfg.get("output_shape")
    if input_shape is None or output_shape is None:
        return None
    return tuple(int(x) for x in input_shape), tuple(int(x) for x in output_shape)


def _two_axis_shape(size: int, preferred_inner: int = 64) -> Tuple[int, int]:
    if size % preferred_inner == 0:
        return size // preferred_inner, preferred_inner
    root = isqrt(size)
    for inner in range(root, 0, -1):
        if size % inner == 0:
            return size // inner, inner
    return size, 1


def deepseek_v4_ndlinear_shapes(
    config: Any,
    target: str,
    input_size: int,
    output_size: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    configured = _shape_from_config(config, target)
    if configured is not None:
        return configured

    hidden_shape = _two_axis_shape(int(config.hidden_size))
    inter_shape = _two_axis_shape(
        int(
            getattr(config, "moe_intermediate_size", 0)
            or getattr(config, "intermediate_size")
        )
    )
    q_rank_shape = _two_axis_shape(int(config.q_lora_rank))
    head_dim = int(config.qk_nope_head_dim + config.qk_rope_head_dim)

    if target == "wq_a":
        return hidden_shape, q_rank_shape
    if target == "wkv":
        return hidden_shape, _two_axis_shape(output_size)
    if target == "wqkv_a":
        return hidden_shape, _two_axis_shape(output_size)
    if target == "wq_b":
        return q_rank_shape, (int(config.num_attention_heads), head_dim)
    if target == "wo_b":
        return (int(config.o_groups), int(config.o_lora_rank)), hidden_shape
    if target == "shared_experts.gate_up_proj":
        return (1, *hidden_shape), (2, *inter_shape)
    if target == "shared_experts.down_proj":
        return inter_shape, hidden_shape
    if target in {"experts.gate_proj", "experts.up_proj"}:
        return hidden_shape, inter_shape
    if target == "experts.down_proj":
        return inter_shape, hidden_shape

    return _two_axis_shape(input_size), _two_axis_shape(output_size)


def deepseek_v4_wo_a_shapes(config: Any) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    per_group_input = (
        int(config.num_attention_heads)
        * int(config.qk_nope_head_dim + config.qk_rope_head_dim)
        // int(config.o_groups)
    )
    return (int(config.o_groups), per_group_input), (
        int(config.o_groups),
        int(config.o_lora_rank),
    )
