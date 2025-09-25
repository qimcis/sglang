from __future__ import annotations

import json
import math
import os
from pathlib import Path
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

        # Check for PTQ mode
        self.ptq_path = os.environ.get("QGEMM_PTQ_PATH")
        self.is_ptq_mode = bool(self.ptq_path)

        if self.is_ptq_mode:
            print(f"[QGEMM] Enabling PTQ mode with quantized model at: {self.ptq_path}")
            # Load the quantized model index (similar to our PTQ config)
            index_path = os.path.join(self.ptq_path, "index.json")
            if not os.path.exists(index_path):
                raise ValueError(f"Pre-quantized model index not found at {index_path}")

            with open(index_path, "r") as f:
                self.ptq_index = json.load(f)
            print(f"[QGEMM PTQ] Loaded index with {len(self.ptq_index['layers'])} quantized layers")
        else:
            self.ptq_index = None

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
            if self.is_ptq_mode:
                # Import here to avoid circular imports
                from .qgemm_int4_ptq import QgemmInt4PTQLinearMethod
                # Create a compatible config object for the PTQ method
                ptq_config = type('Config', (), {
                    'quantized_model_path': self.ptq_path,
                    'index': self.ptq_index,
                    'group_size': self.group_size,
                    'fuse_activation': self.fuse_activation
                })()
                return QgemmInt4PTQLinearMethod(ptq_config, prefix)
            else:
                return QgemmInt4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


FAIL_DUMP_PATH = Path(os.environ.get("QGEMM_FAIL_DUMP", "/workspace/qgemm_fail.pt"))


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
        # Avoid overwriting existing attributes such as weight_loader
        extra_attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        if extra_attrs:
            set_weight_attrs(packed_w, extra_attrs)

        scales = GroupQuantScaleParameter(
            data=torch.empty(Nshard, _ceil_div(K, group_size), dtype=torch.float16),
            input_dim=1,  # groups along K
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("scales", scales)
        if extra_attrs:
            set_weight_attrs(scales, extra_attrs)

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

        # Debug stats before invoking the kernel
        with torch.no_grad():
            x_stats = {
                "shape": tuple(x_2d.shape),
                "dtype": str(x_2d.dtype),
                "nan": bool(torch.isnan(x_2d).any().item()) if x_2d.numel() else False,
                "inf": bool(torch.isinf(x_2d).any().item()) if x_2d.numel() else False,
                "max": float(x_2d.abs().max().item()) if x_2d.numel() else 0.0,
                "mean": float(x_2d.float().mean().item()) if x_2d.numel() else 0.0,
            }
            packed_stats = {
                "shape": tuple(packed_w.shape),
                "dtype": str(packed_w.dtype),
                "max": int(packed_w.max().item()) if packed_w.numel() else 0,
                "min": int(packed_w.min().item()) if packed_w.numel() else 0,
            }
            scale_stats = {
                "shape": tuple(scales.shape),
                "dtype": str(scales.dtype),
                "min": float(scales.min().item()) if scales.numel() else 0.0,
                "max": float(scales.max().item()) if scales.numel() else 0.0,
                "nan": bool(torch.isnan(scales).any().item()) if scales.numel() else False,
            }
        print(
            f"[qgemm] apply start layer={layer.__class__.__name__} "
            f"bias={'yes' if bias is not None else 'no'} "
            f"x_stats={x_stats} packed_stats={packed_stats} scale_stats={scale_stats}"
        )

        m_size = x_2d.shape[0]
        if m_size < 64:
            padded_m = 64
        else:
            padded_m = int(math.ceil(m_size / 16) * 16) if m_size % 16 else m_size

        if padded_m != m_size:
            x_input = torch.zeros((padded_m, x_2d.shape[1]), dtype=x_2d.dtype, device=x_2d.device)
            x_input[:m_size].copy_(x_2d)
        else:
            x_input = x_2d

        # Call op
        dump_payload = None
        if not FAIL_DUMP_PATH.exists():
            dump_payload = {
                "x": x_input.detach().cpu(),
                "packed": packed_w.detach().cpu(),
                "scales": scales.detach().cpu(),
                "bias": None if bias is None else bias.detach().cpu(),
                "group_size": int(self.quant_config.group_size),
            }

        dump_once = dump_payload is not None and not FAIL_DUMP_PATH.exists()

        if dump_once:
            torch.save(dump_payload, FAIL_DUMP_PATH)
            print(f"[qgemm] captured operands to {FAIL_DUMP_PATH}")

        try:
            if use_silu:
                y_2d = torch.ops.qgemm.int4_bias_silu(
                    x_input, packed_w, scales, int(self.quant_config.group_size), bias
                )
            else:
                y_2d = torch.ops.qgemm.int4_bias(
                    x_input, packed_w, scales, int(self.quant_config.group_size), bias
                )
        except Exception as err:
            print(f"[qgemm] kernel raised {err.__class__.__name__}: {err}")
            raise

        print("[qgemm] apply succeeded")

        if padded_m != m_size:
            y_2d = y_2d[:m_size]

        # Reshape back to original batch dims with output dim
        out_shape = x.shape[:-1] + (packed_w.shape[0],)
        y = y_2d.reshape(out_shape)

        # Keep FP16 (kernel output); caller may cast if needed.
        return y
