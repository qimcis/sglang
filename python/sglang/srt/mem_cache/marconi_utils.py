from __future__ import annotations

from typing import List


def normalize(values: List[float]) -> List[float]:
    if len(values) > 1:
        min_val = min(values)
        max_val = max(values)
        if min_val != max_val:
            return [(val - min_val) / (max_val - min_val) for val in values]
    return [1.0] * len(values)


def get_attn_flops(seq_len: int, model_dim: int) -> int:
    return 8 * seq_len * model_dim**2 + 4 * seq_len**2 * model_dim


def get_mlp_flops(seq_len: int, model_dim: int) -> int:
    return 16 * seq_len * model_dim**2


def get_dense_mlp_flops(seq_len: int, model_dim: int, intermediate_size: int) -> int:
    return 6 * seq_len * model_dim * intermediate_size


def get_moe_mlp_flops(
    seq_len: int,
    model_dim: int,
    moe_intermediate_size: int,
    moe_top_k: int,
    shared_expert_intermediate_size: int = 0,
) -> int:
    total_intermediate = shared_expert_intermediate_size + (
        moe_top_k * moe_intermediate_size
    )
    return 6 * seq_len * model_dim * total_intermediate


def get_mamba2_flops(seq_len: int, model_dim: int, state_size: int) -> int:
    return (
        12 * seq_len * model_dim**2
        + 16 * seq_len * model_dim * state_size
        + 10 * seq_len * model_dim
    )


def get_mamba1_flops(seq_len: int, model_dim: int, state_size: int) -> int:
    return get_mamba2_flops(seq_len, model_dim, state_size)


def get_gdn_flops(
    seq_len: int,
    model_dim: int,
    key_dim: int,
    value_dim: int,
    state_size: int,
    num_value_heads: int,
) -> int:
    proj_flops = 2 * seq_len * model_dim * (
        2 * key_dim + 2 * value_dim + 2 * num_value_heads
    )
    recurrent_flops = 8 * seq_len * value_dim * state_size
    gating_flops = 4 * seq_len * value_dim
    return proj_flops + recurrent_flops + gating_flops


def get_kv_cache_size_bytes(seq_len: int, model_dim: int, dtype_size: int) -> int:
    return 2 * seq_len * model_dim * dtype_size
