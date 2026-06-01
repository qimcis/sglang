from __future__ import annotations

from math import prod
from types import SimpleNamespace
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKOutput,
)
from sglang.srt.utils import set_weight_attrs


def _replace_dim(shape: Tuple[int, ...], axis: int, value: int) -> Tuple[int, ...]:
    shape = list(shape)
    shape[axis] = value
    return tuple(shape)


class NdLinearFusedMoE(nn.Module):
    """NdLinear expert stack for routed MoE layers.

    The parameter layout is factorized per expert:
    ``w1_factors.<mode>``, ``w3_factors.<mode>``, and ``w2_factors.<mode>``.
    Checkpoints should store factors as
    ``experts.<id>.gate_proj.nd_factors.<mode>.weight``,
    ``experts.<id>.up_proj.nd_factors.<mode>.weight``, and
    ``experts.<id>.down_proj.nd_factors.<mode>.weight``.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_shape: Iterable[int],
        intermediate_shape: Iterable[int],
        params_dtype: torch.dtype,
        swiglu_limit: float | None = None,
        layer_id: int = 0,
        **_: object,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_local_experts = self.num_experts
        self.layer_id = layer_id
        self.hidden_shape = tuple(int(x) for x in hidden_shape)
        self.intermediate_shape_global = tuple(int(x) for x in intermediate_shape)
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_tp_rank = get_moe_tensor_parallel_rank()
        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        if self.moe_ep_size != 1:
            raise NotImplementedError(
                "NdLinear routed experts currently require expert parallel size 1. "
                "Use shared-expert/attention NdLinear targets or disable EP for "
                "NdLinear expert checkpoints."
            )
        if len(self.hidden_shape) != len(self.intermediate_shape_global):
            raise ValueError(
                "NdLinear MoE hidden/intermediate shapes must have equal rank"
            )
        if self.intermediate_shape_global[0] % self.moe_tp_size != 0:
            raise ValueError(
                f"intermediate_shape[0]={self.intermediate_shape_global[0]} must be "
                f"divisible by moe_tp_size={self.moe_tp_size}"
            )
        self.intermediate_shape = _replace_dim(
            self.intermediate_shape_global,
            0,
            divide(self.intermediate_shape_global[0], self.moe_tp_size),
        )
        self.hidden_size = prod(self.hidden_shape)
        self.intermediate_size = prod(self.intermediate_shape)
        self.swiglu_limit = swiglu_limit
        self.should_fuse_routed_scaling_factor_in_topk = False
        self.moe_runner_config = SimpleNamespace(inplace=False, is_gated=True)
        self.quant_method = None

        self.w1_factors = self._make_factors(
            self.hidden_shape, self.intermediate_shape, params_dtype, "w1"
        )
        self.w3_factors = self._make_factors(
            self.hidden_shape, self.intermediate_shape, params_dtype, "w3"
        )
        self.w2_factors = self._make_factors(
            self.intermediate_shape, self.hidden_shape, params_dtype, "w2"
        )

    def _make_factors(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        dtype: torch.dtype,
        shard_id: str,
    ) -> nn.ParameterList:
        factors = nn.ParameterList()
        for mode, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            param = Parameter(
                torch.empty(self.num_experts, out_dim, in_dim, dtype=dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                param,
                {
                    "weight_loader": self.weight_loader,
                    "ndlinear_moe_shard_id": shard_id,
                    "ndlinear_mode": mode,
                },
            )
            factors.append(param)
        return factors

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        rank: int,
    ):
        return [
            (
                f"experts.{param_prefix}.{mode}",
                f"experts.{expert_id}.{weight_name}.nd_factors.{mode}.weight",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name, param_prefix in [
                ("w1", ckpt_gate_proj_name, "w1_factors"),
                ("w2", ckpt_down_proj_name, "w2_factors"),
                ("w3", ckpt_up_proj_name, "w3_factors"),
            ]
            for mode in range(rank)
        ]

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str | None = None,
        *,
        shard_id: str | None = None,
        expert_id: int | None = None,
    ):
        if expert_id is None:
            raise ValueError("NdLinear MoE factor loading requires expert_id")
        mode = getattr(param, "ndlinear_mode")
        shard_id = shard_id or getattr(param, "ndlinear_moe_shard_id")
        loaded_weight = self._shard_factor(shard_id, mode, loaded_weight)
        param_data = param.data[expert_id]
        assert (
            param_data.shape == loaded_weight.shape
        ), f"NdLinear MoE shape mismatch: {param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)

    def _shard_factor(
        self, shard_id: str, mode: int, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        if self.moe_tp_size == 1 or mode != 0:
            return loaded_weight
        if shard_id in {"w1", "w3"}:
            shard_size = self.intermediate_shape[0]
            return loaded_weight.narrow(0, self.moe_tp_rank * shard_size, shard_size)
        if shard_id == "w2":
            shard_size = self.intermediate_shape[0]
            return loaded_weight.narrow(1, self.moe_tp_rank * shard_size, shard_size)
        return loaded_weight

    def _apply_factors(
        self,
        x: torch.Tensor,
        factors: nn.ParameterList,
        expert_id: int,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        y = x.reshape(x.shape[0], *input_shape)
        feature_start = 1
        for mode, factor in enumerate(factors):
            y = y.movedim(feature_start + mode, -1)
            y = F.linear(y, factor[expert_id])
            y = y.movedim(-1, feature_start + mode)
        return y.reshape(x.shape[0], prod(output_shape))

    def _expert_forward(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        gate = self._apply_factors(
            x, self.w1_factors, expert_id, self.hidden_shape, self.intermediate_shape
        )
        up = self._apply_factors(
            x, self.w3_factors, expert_id, self.hidden_shape, self.intermediate_shape
        )
        if self.swiglu_limit is not None:
            gate = F.silu(gate).clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        else:
            gate = F.silu(gate)
        x = gate * up
        return self._apply_factors(
            x, self.w2_factors, expert_id, self.intermediate_shape, self.hidden_shape
        )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if isinstance(topk_output, BypassedTopKOutput):
            topk_output = topk_output.to_standard(layer_id=self.layer_id)
        if not isinstance(topk_output, StandardTopKOutput):
            raise NotImplementedError(
                "NdLinear MoE only supports standard top-k output, "
                f"got {topk_output.format}"
            )
        final = hidden_states.new_zeros(hidden_states.shape[0], self.hidden_size)
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights.to(hidden_states.dtype)
        for expert_id in range(self.num_experts):
            for route in range(topk_ids.shape[1]):
                mask = topk_ids[:, route] == expert_id
                if not torch.any(mask):
                    continue
                out = self._expert_forward(hidden_states[mask], expert_id)
                out = out * topk_weights[mask, route].unsqueeze(-1)
                final[mask] += out
        return final
