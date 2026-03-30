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


def get_mamba2_flops(
    seq_len: int,
    model_dim: int,
    state_size: int,
    intermediate_size: int | None = None,
    conv_dim: int | None = None,
    num_heads: int | None = None,
    conv_kernel: int | None = None,
) -> int:
    if (
        intermediate_size is None
        or conv_dim is None
        or num_heads is None
        or conv_kernel is None
    ):
        return (
            12 * seq_len * model_dim**2
            + 16 * seq_len * model_dim * state_size
            + 10 * seq_len * model_dim
        )

    input_proj_flops = 2 * seq_len * model_dim * (
        intermediate_size + conv_dim + num_heads
    )
    conv_flops = 2 * seq_len * conv_dim * conv_kernel
    recurrent_flops = 8 * seq_len * intermediate_size * state_size
    gating_flops = 8 * seq_len * intermediate_size
    output_proj_flops = 2 * seq_len * intermediate_size * model_dim
    return (
        input_proj_flops
        + conv_flops
        + recurrent_flops
        + gating_flops
        + output_proj_flops
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
    conv_kernel: int | None = None,
) -> int:
    proj_flops = 2 * seq_len * model_dim * (
        2 * key_dim + 2 * value_dim + 2 * num_value_heads
    )
    conv_flops = (
        2 * seq_len * (2 * key_dim + value_dim) * conv_kernel
        if conv_kernel is not None
        else 0
    )
    recurrent_flops = 8 * seq_len * value_dim * state_size
    gating_flops = 4 * seq_len * value_dim
    output_proj_flops = 2 * seq_len * value_dim * model_dim
    return (
        proj_flops
        + conv_flops
        + recurrent_flops
        + gating_flops
        + output_proj_flops
    )


def get_kda_flops(
    seq_len: int,
    model_dim: int,
    projection_dim: int,
    state_size: int,
    num_heads: int,
    conv_kernel: int,
) -> int:
    qkv_proj_flops = 6 * seq_len * model_dim * projection_dim
    beta_proj_flops = 2 * seq_len * model_dim * num_heads
    gate_a_proj_flops = 4 * seq_len * model_dim * state_size
    gate_b_proj_flops = 8 * seq_len * state_size * projection_dim
    qkv_conv_flops = 6 * seq_len * projection_dim * conv_kernel
    recurrent_flops = seq_len * num_heads * (8 * state_size**2 + 8 * state_size)
    output_proj_flops = 2 * seq_len * projection_dim * model_dim
    output_gate_flops = 4 * seq_len * projection_dim
    return (
        qkv_proj_flops
        + beta_proj_flops
        + gate_a_proj_flops
        + gate_b_proj_flops
        + qkv_conv_flops
        + recurrent_flops
        + output_proj_flops
        + output_gate_flops
    )


def get_kv_cache_size_bytes(seq_len: int, model_dim: int, dtype_size: int) -> int:
    return 2 * seq_len * model_dim * dtype_size
