from __future__ import annotations

from math import prod
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.utils import set_weight_attrs


def _as_shape(shape: Iterable[int], name: str) -> Tuple[int, ...]:
    value = tuple(int(dim) for dim in shape)
    if not value or any(dim <= 0 for dim in value):
        raise ValueError(f"{name} must be a non-empty sequence of positive ints")
    return value


def _replace_dim(shape: Sequence[int], dim: int, value: int) -> Tuple[int, ...]:
    out = list(shape)
    out[dim] = value
    return tuple(out)


class _NdLinearFactor(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        self.weight = Parameter(
            torch.empty(out_features, in_features, dtype=dtype), requires_grad=False
        )


class NdLinearProjection(nn.Module):
    """Mode-wise factorized replacement for a flattened linear projection.

    The module interprets the last input dimension as ``input_shape``, applies one
    learned matrix per tensor mode, and flattens ``output_shape`` back into the
    last output dimension. It intentionally mirrors SGLang linear layers by
    returning ``(output, output_bias)``.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        *,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        factor_input_shard_axis: Optional[int] = None,
        factor_output_shard_axis: Optional[int] = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.global_input_shape = _as_shape(input_shape, "input_shape")
        self.global_output_shape = _as_shape(output_shape, "output_shape")
        if len(self.global_input_shape) != len(self.global_output_shape):
            raise ValueError(
                "NdLinear input_shape and output_shape must have the same rank"
            )

        self.input_shape = self.global_input_shape
        self.output_shape = self.global_output_shape
        self.factor_input_shard_axis = factor_input_shard_axis
        self.factor_output_shard_axis = factor_output_shard_axis
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype

        if factor_input_shard_axis is not None:
            axis = factor_input_shard_axis
            if self.input_shape[axis] % tp_size != 0:
                raise ValueError(
                    f"input_shape[{axis}]={self.input_shape[axis]} must be "
                    f"divisible by tp_size={tp_size}"
                )
            self.input_shape = _replace_dim(
                self.input_shape, axis, divide(self.input_shape[axis], tp_size)
            )
        if factor_output_shard_axis is not None:
            axis = factor_output_shard_axis
            if self.output_shape[axis] % tp_size != 0:
                raise ValueError(
                    f"output_shape[{axis}]={self.output_shape[axis]} must be "
                    f"divisible by tp_size={tp_size}"
                )
            self.output_shape = _replace_dim(
                self.output_shape, axis, divide(self.output_shape[axis], tp_size)
            )

        self.input_size = prod(self.global_input_shape)
        self.output_size = prod(self.global_output_shape)
        self.input_size_per_partition = prod(self.input_shape)
        self.output_size_per_partition = prod(self.output_shape)

        self.nd_factors = nn.ModuleList(
            [
                _NdLinearFactor(in_dim, out_dim, params_dtype)
                for in_dim, out_dim in zip(self.input_shape, self.output_shape)
            ]
        )
        for mode, factor in enumerate(self.nd_factors):
            set_weight_attrs(
                factor.weight,
                {
                    "weight_loader": self.weight_loader,
                    "ndlinear_mode": mode,
                },
            )

        if bias:
            self.nd_biases = nn.ParameterList(
                [
                    Parameter(torch.empty(dim, dtype=params_dtype), requires_grad=False)
                    for dim in self.output_shape
                ]
            )
            for mode, bias_param in enumerate(self.nd_biases):
                set_weight_attrs(
                    bias_param,
                    {
                        "weight_loader": self.weight_loader,
                        "ndlinear_mode": mode,
                        "ndlinear_is_bias": True,
                    },
                )
        else:
            self.nd_biases = None

    @property
    def rank(self) -> int:
        return len(self.input_shape)

    def _shard_loaded_weight(
        self, param: Parameter, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        mode = getattr(param, "ndlinear_mode", None)
        is_bias = getattr(param, "ndlinear_is_bias", False)
        if mode is None:
            return loaded_weight

        if (
            self.factor_output_shard_axis is not None
            and mode == self.factor_output_shard_axis
        ):
            shard_size = param.shape[0]
            start = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start, shard_size)

        if (
            not is_bias
            and self.factor_input_shard_axis is not None
            and mode == self.factor_input_shard_axis
        ):
            shard_size = param.shape[1]
            start = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(1, start, shard_size)

        return loaded_weight

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        loaded_weight = self._shard_loaded_weight(param, loaded_weight)
        assert (
            param.shape == loaded_weight.shape
        ), f"NdLinear shape mismatch: {param.shape=} {loaded_weight.shape=}"
        param.data.copy_(loaded_weight)

    def apply_ndlinear(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if x_shape[-1] != self.input_size_per_partition:
            raise ValueError(
                f"NdLinear expected last dim {self.input_size_per_partition}, "
                f"got {x_shape[-1]}"
            )
        y = x.reshape(*x_shape[:-1], *self.input_shape)
        feature_start = y.ndim - self.rank
        for mode, factor in enumerate(self.nd_factors):
            bias = None if self.nd_biases is None else self.nd_biases[mode]
            y = y.movedim(feature_start + mode, -1)
            y = F.linear(y, factor.weight, bias)
            y = y.movedim(-1, feature_start + mode)
        return y.reshape(*x_shape[:-1], self.output_size_per_partition)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.apply_ndlinear(x)
        return output, None

    def extra_repr(self) -> str:
        return (
            f"input_shape={self.input_shape}, output_shape={self.output_shape}, "
            f"tp_size={self.tp_size}"
        )


class NdReplicatedLinear(NdLinearProjection):
    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        *,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        **_: object,
    ):
        super().__init__(
            input_shape,
            output_shape,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
        )


class NdColumnParallelLinear(NdLinearProjection):
    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        *,
        bias: bool = False,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        output_tp_axis: int = 0,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        **_: object,
    ):
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.gather_output = gather_output
        super().__init__(
            input_shape,
            output_shape,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            factor_output_shard_axis=output_tp_axis,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_parallel = self.apply_ndlinear(x)
        if self.gather_output:
            from sglang.srt.distributed import tensor_model_parallel_all_gather

            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        return output, None


class NdRowParallelLinear(NdLinearProjection):
    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        *,
        bias: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        input_tp_axis: int = 0,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        **_: object,
    ):
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        super().__init__(
            input_shape,
            output_shape,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            factor_input_shard_axis=input_tp_axis,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def forward(
        self, x: torch.Tensor, skip_all_reduce: bool = False, forward_batch=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.input_is_parallel:
            input_parallel = x
        else:
            input_parallel = split_tensor_along_last_dim(x, self.tp_size)[
                self.tp_rank
            ].contiguous()

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output_parallel = self.apply_ndlinear(input_parallel)

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        return output, None
