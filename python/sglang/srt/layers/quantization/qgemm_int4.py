from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch.nn import Parameter

try:
    import qgemm  # noqa: F401
    # Best-effort: try to load the compiled extension so torch.ops.qgemm.* exist
    try:
        if hasattr(qgemm, "_try_import_extension"):
            qgemm._try_import_extension()
    except Exception:
        pass
except Exception:
    # Allow import without the extension; runtime will error if ops are called
    pass

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import set_weight_attrs


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class QgemmInt4Config(QuantizationConfig):
    """Config for qgemm INT4 weight-only linear.

    Expects weights saved as:
      - packed_w: [N, ceil(K/2)] uint8 (two nibbles per byte along K)
      - scales:   [N, ceil(K/group_size)] float16 (row-major groups over K)

    Min capability: sm80 (A100/H100). Activations supported: float16.
    """

    def __init__(self, group_size: int = 64, fuse_activation: Optional[str] = None) -> None:
        super().__init__()
        self.group_size = int(group_size)
        # Optional fused activation variant (e.g. "silu") for certain layers
        self.fuse_activation = fuse_activation  # None | "silu"

    def get_name(self) -> str:
        return "qgemm_int4"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        # Kernels expect FP16 inputs/outputs
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        # No required side-car config files; defaults suffice.
        return []

    @classmethod
    def from_config(cls, config: dict) -> "QgemmInt4Config":
        # Try common keys; fallback to defaults
        group_size = config.get("group_size", 64)
        # Optionally allow a fused activation hint
        fuse_activation = config.get("fuse_activation", None)
        if fuse_activation is not None:
            fuse_activation = str(fuse_activation).lower()
        return cls(group_size=group_size, fuse_activation=fuse_activation)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            return QgemmInt4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class QgemmInt4LinearMethod(LinearMethodBase):
    """Linear method using qgemm INT4 CUDA ops.

    Registers two parameters on the layer:
      - packed_w: [Nshard, ceil(K/2)] uint8
      - scales:   [Nshard, ceil(K/group_size)] float16
    and applies one of:
      - torch.ops.qgemm.int4_bias
      - torch.ops.qgemm.int4_bias_silu
    """

    def __init__(self, quant_config: QgemmInt4Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del output_size, input_size
        # Column-parallel: outputs are partitioned across TP; inputs are not.
        Nshard = sum(output_partition_sizes)
        K = int(input_size_per_partition)
        group_size = int(self.quant_config.group_size)

        # Validate shapes
        if group_size <= 0:
            raise ValueError(f"group_size must be positive. Got {group_size}")

        # Register packed weights and scales
        weight_loader = extra_weight_attrs.get("weight_loader")

        packed_w = ModelWeightParameter(
            data=torch.empty(Nshard, _ceil_div(K, 2), dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("packed_w", packed_w)
        set_weight_attrs(packed_w, extra_weight_attrs)

        scales = GroupQuantScaleParameter(
            data=torch.empty(Nshard, _ceil_div(K, group_size), dtype=torch.float16),
            input_dim=1,  # groups along K
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        # Nothing special for bias; ColumnParallelLinear already registers it

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure qgemm ops are available
        if not hasattr(torch.ops, "qgemm") or not hasattr(torch.ops.qgemm, "int4_bias"):
            raise RuntimeError(
                "qgemm ops not available. Ensure qgemm is installed and built with CUDA."
            )

        # Shapes
        assert hasattr(layer, "packed_w") and hasattr(layer, "scales"), (
            "Layer missing qgemm parameters (packed_w, scales)."
        )
        packed_w: torch.Tensor = layer.packed_w
        scales: torch.Tensor = layer.scales

        # Flatten batch to [M, K]
        x_dtype = x.dtype
        if x_dtype != torch.float16:
            x_mat = x.to(torch.float16)
        else:
            x_mat = x
        x_2d = x_mat.reshape(-1, x_mat.shape[-1]).contiguous()

        # Select kernel
        fuse_act = getattr(layer, "fused_activation", None) or self.quant_config.fuse_activation
        use_silu = isinstance(fuse_act, str) and fuse_act.lower() == "silu"

        # Call op
        if use_silu:
            y_2d = torch.ops.qgemm.int4_bias_silu(
                x_2d, packed_w, scales, int(self.quant_config.group_size), bias
            )
        else:
            y_2d = torch.ops.qgemm.int4_bias(
                x_2d, packed_w, scales, int(self.quant_config.group_size), bias
            )

        # Reshape back to original batch dims with output dim
        out_shape = x.shape[:-1] + (packed_w.shape[0],)
        y = y_2d.reshape(out_shape)

        # Keep FP16 (kernel output); caller may cast if needed.
        return y

