from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.mem_cache.marconi_utils import (
    get_attn_flops,
    get_dense_mlp_flops,
    get_gdn_flops,
    get_kv_cache_size_bytes,
    get_mamba2_flops,
    get_moe_mlp_flops,
)
from sglang.srt.utils.common import is_float4_e2m1fn_x2

@dataclass(frozen=True, kw_only=True)
class _MarconiProfileBuildContext:
    recurrent_family: str
    cache_params: object | None
    num_attn_layers: int
    model_dim: int
    kv_cache_dtype_size: int
    state_size_bytes: int | None
    ffn_cost: MarconiFFNCost | None


@dataclass(frozen=True, kw_only=True)
class MarconiRecurrentCost:
    family: Literal["mamba2", "gdn"]
    num_layers: int
    model_dim: int
    state_size: int
    state_size_bytes: int
    key_dim: int | None = None
    value_dim: int | None = None
    num_value_heads: int | None = None

    def flops(self, replay_tokens: int) -> float:
        if replay_tokens <= 0 or self.num_layers <= 0:
            return 0.0
        if self.family == "mamba2":
            return self.num_layers * get_mamba2_flops(
                replay_tokens, self.model_dim, self.state_size
            )
        if self.family == "gdn":
            if (
                self.key_dim is None
                or self.value_dim is None
                or self.num_value_heads is None
            ):
                raise ValueError("GDN cost profile is missing key/value dimensions.")
            return self.num_layers * get_gdn_flops(
                replay_tokens,
                self.model_dim,
                self.key_dim,
                self.value_dim,
                self.state_size,
                self.num_value_heads,
            )
        raise ValueError(f"Unsupported Marconi recurrent family: {self.family}")

    def bytes(self, num_states: int = 1) -> int:
        if num_states <= 0 or self.num_layers <= 0:
            return 0
        return num_states * self.num_layers * self.state_size_bytes


@dataclass(frozen=True, kw_only=True)
class MarconiFFNCost:
    family: Literal["dense_mlp", "moe_topk"]
    num_layers: int
    model_dim: int
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    moe_top_k: int | None = None
    shared_expert_intermediate_size: int = 0

    def flops_delta(self, total_tokens: int, parent_tokens: int) -> float:
        if self.num_layers <= 0 or total_tokens <= parent_tokens:
            return 0.0
        if self.family == "dense_mlp":
            if self.intermediate_size is None:
                raise ValueError("Dense MLP cost profile is missing intermediate_size.")
            return self.num_layers * (
                get_dense_mlp_flops(total_tokens, self.model_dim, self.intermediate_size)
                - get_dense_mlp_flops(
                    parent_tokens, self.model_dim, self.intermediate_size
                )
            )
        if self.family == "moe_topk":
            if self.moe_intermediate_size is None or self.moe_top_k is None:
                raise ValueError("MoE cost profile is missing expert dimensions.")
            return self.num_layers * (
                get_moe_mlp_flops(
                    total_tokens,
                    self.model_dim,
                    self.moe_intermediate_size,
                    self.moe_top_k,
                    self.shared_expert_intermediate_size,
                )
                - get_moe_mlp_flops(
                    parent_tokens,
                    self.model_dim,
                    self.moe_intermediate_size,
                    self.moe_top_k,
                    self.shared_expert_intermediate_size,
                )
            )
        raise ValueError(f"Unsupported Marconi FFN family: {self.family}")


@dataclass(frozen=True, kw_only=True)
class MarconiCostProfile:
    recurrent_family: str
    recurrent: Optional[MarconiRecurrentCost]
    ffn: Optional[MarconiFFNCost]
    num_attn_layers: int
    model_dim: int
    kv_cache_dtype_size: int

    def recurrent_flops(self, replay_tokens: int) -> float:
        if self.recurrent is None:
            return 0.0
        return self.recurrent.flops(replay_tokens)

    def attn_flops_delta(self, total_tokens: int, parent_tokens: int) -> float:
        if self.num_attn_layers <= 0 or total_tokens <= parent_tokens:
            return 0.0
        return self.num_attn_layers * (
            get_attn_flops(total_tokens, self.model_dim)
            - get_attn_flops(parent_tokens, self.model_dim)
        )

    def ffn_flops_delta(self, total_tokens: int, parent_tokens: int) -> float:
        if self.ffn is None:
            return 0.0
        return self.ffn.flops_delta(total_tokens, parent_tokens)

    def recurrent_bytes(self, num_states: int = 1) -> int:
        if self.recurrent is None:
            return 0
        return self.recurrent.bytes(num_states)

    def kv_bytes(self, total_tokens: int) -> int:
        if self.num_attn_layers <= 0 or total_tokens <= 0:
            return 0
        return self.num_attn_layers * get_kv_cache_size_bytes(
            total_tokens, self.model_dim, self.kv_cache_dtype_size
        )


def _get_marconi_kv_dtype_size(model_runner) -> int:
    kv_cache_dtype = getattr(model_runner, "kv_cache_dtype", None)
    if kv_cache_dtype is None:
        return 2
    kv_cache_dtype_size = torch._utils._element_size(kv_cache_dtype)
    if is_float4_e2m1fn_x2(kv_cache_dtype):
        kv_cache_dtype_size = max(1, kv_cache_dtype_size // 2)
    return kv_cache_dtype_size


def _get_marconi_state_size_bytes(req_to_token_pool) -> int | None:
    mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
    if mamba_pool is None or not getattr(mamba_pool, "num_mamba_layers", None):
        return None
    try:
        total_bytes = mamba_pool.mamba_cache.mem_usage_bytes()
        slots = max(1, getattr(mamba_pool, "size", 0) + 1)
        return int(total_bytes / (slots * max(mamba_pool.num_mamba_layers, 1)))
    except Exception:
        return None


def _get_num_attn_layers(hf_config, num_recurrent_layers: int) -> int | None:
    if hasattr(hf_config, "full_attention_layer_ids"):
        return len(list(hf_config.full_attention_layer_ids))
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type:
        return sum(1 for layer_type in layers_block_type if layer_type == "attention")
    hybrid_pattern = getattr(hf_config, "hybrid_override_pattern", None)
    if hybrid_pattern:
        return hybrid_pattern.count("*")
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is not None:
        return max(0, num_hidden_layers - num_recurrent_layers)
    return None


def _build_marconi_ffn_cost(hf_config, model_dim: int) -> MarconiFFNCost | None:
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return None
    moe_top_k = (
        getattr(hf_config, "num_experts_per_tok", None)
        or getattr(hf_config, "moe_top_k", None)
        or getattr(hf_config, "num_experts_per_token", None)
    )
    moe_intermediate_size = (
        getattr(hf_config, "moe_intermediate_size", None)
        or getattr(hf_config, "expert_ffn_hidden_size", None)
        or getattr(hf_config, "expert_intermediate_size", None)
    )
    shared_expert_intermediate_size = (
        getattr(hf_config, "shared_expert_intermediate_size", None)
        or getattr(hf_config, "moe_shared_expert_intermediate_size", None)
        or 0
    )
    if moe_top_k is not None and moe_intermediate_size is not None:
        return MarconiFFNCost(
            family="moe_topk",
            num_layers=num_hidden_layers,
            model_dim=model_dim,
            moe_intermediate_size=moe_intermediate_size,
            moe_top_k=moe_top_k,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
    intermediate_size = getattr(hf_config, "intermediate_size", None)
    if intermediate_size is None:
        return None
    return MarconiFFNCost(
        family="dense_mlp",
        num_layers=num_hidden_layers,
        model_dim=model_dim,
        intermediate_size=intermediate_size,
    )


def _infer_marconi_recurrent_family(hf_config, model_runner, cache_params) -> str:
    if getattr(model_runner, "hybrid_gdn_config", None) is not None:
        return "gdn"
    cache_params_type = type(cache_params).__name__
    if cache_params_type == "KimiLinearCacheParams":
        return "kda"
    return "mamba2"


def _build_exact_mamba2_recurrent_cost(
    *,
    hf_config,
    cache_params,
    state_size_bytes: int,
    model_dim: int,
) -> Optional[MarconiRecurrentCost]:
    shape = getattr(cache_params, "shape", None)
    state_size = getattr(shape, "state_size", None) or getattr(shape, "head_dim", None)
    if state_size is None:
        return None
    return MarconiRecurrentCost(
        family="mamba2",
        num_layers=len(cache_params.layers),
        model_dim=model_dim,
        state_size=state_size,
        state_size_bytes=state_size_bytes,
    )


def _build_exact_gdn_recurrent_cost(
    *,
    hf_config,
    cache_params,
    state_size_bytes: int,
    model_dim: int,
) -> Optional[MarconiRecurrentCost]:
    shape = getattr(cache_params, "shape", None)
    state_size = getattr(shape, "state_size", None) or getattr(shape, "head_dim", None)
    key_heads = getattr(hf_config, "linear_num_key_heads", None)
    value_heads = getattr(hf_config, "linear_num_value_heads", None)
    key_head_dim = getattr(hf_config, "linear_key_head_dim", None)
    value_head_dim = getattr(hf_config, "linear_value_head_dim", None)
    if (
        state_size is None
        or key_heads is None
        or value_heads is None
        or key_head_dim is None
        or value_head_dim is None
    ):
        return None
    return MarconiRecurrentCost(
        family="gdn",
        num_layers=len(cache_params.layers),
        model_dim=model_dim,
        state_size=state_size,
        state_size_bytes=state_size_bytes,
        key_dim=key_heads * key_head_dim,
        value_dim=value_heads * value_head_dim,
        num_value_heads=value_heads,
    )


_EXACT_RECURRENT_BUILDERS = {
    "mamba2": _build_exact_mamba2_recurrent_cost,
    "gdn": _build_exact_gdn_recurrent_cost,
}


def _get_marconi_model_dim(hf_config) -> int:
    return (
        getattr(hf_config, "hidden_size", None)
        or getattr(hf_config, "d_model", None)
        or getattr(hf_config, "n_embd", None)
        or 0
    )


def _build_marconi_profile_context(
    *,
    hf_config,
    model_runner,
    req_to_token_pool,
) -> _MarconiProfileBuildContext:
    cache_params = getattr(hf_config, "mamba2_cache_params", None)
    recurrent_family = _infer_marconi_recurrent_family(
        hf_config, model_runner, cache_params
    )
    num_recurrent_layers = len(cache_params.layers) if cache_params is not None else 0
    model_dim = _get_marconi_model_dim(hf_config)
    return _MarconiProfileBuildContext(
        recurrent_family=recurrent_family,
        cache_params=cache_params,
        num_attn_layers=_get_num_attn_layers(hf_config, num_recurrent_layers) or 0,
        model_dim=model_dim,
        kv_cache_dtype_size=_get_marconi_kv_dtype_size(model_runner),
        state_size_bytes=_get_marconi_state_size_bytes(req_to_token_pool),
        ffn_cost=_build_marconi_ffn_cost(hf_config, model_dim) if model_dim > 0 else None,
    )


def _build_exact_recurrent_cost(
    *,
    hf_config,
    context: _MarconiProfileBuildContext,
) -> MarconiRecurrentCost | None:
    builder = _EXACT_RECURRENT_BUILDERS.get(context.recurrent_family)
    if (
        builder is None
        or context.cache_params is None
        or context.state_size_bytes is None
        or context.model_dim <= 0
    ):
        return None
    return builder(
        hf_config=hf_config,
        cache_params=context.cache_params,
        state_size_bytes=context.state_size_bytes,
        model_dim=context.model_dim,
    )


def _build_exact_profile(
    *,
    hf_config,
    context: _MarconiProfileBuildContext,
) -> MarconiCostProfile | None:
    recurrent_cost = _build_exact_recurrent_cost(hf_config=hf_config, context=context)
    if (
        recurrent_cost is None
        or context.ffn_cost is None
        or context.num_attn_layers <= 0
        or context.model_dim <= 0
    ):
        return None
    return MarconiCostProfile(
        recurrent_family=context.recurrent_family,
        recurrent=recurrent_cost,
        ffn=context.ffn_cost,
        num_attn_layers=context.num_attn_layers,
        model_dim=context.model_dim,
        kv_cache_dtype_size=context.kv_cache_dtype_size,
    )


def build_marconi_cost_profile(
    *,
    hf_config,
    model_runner,
    req_to_token_pool,
) -> MarconiCostProfile | None:
    context = _build_marconi_profile_context(
        hf_config=hf_config,
        model_runner=model_runner,
        req_to_token_pool=req_to_token_pool,
    )
    return _build_exact_profile(hf_config=hf_config, context=context)
