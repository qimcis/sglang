from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
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
from sglang.srt.utils import set_weight_attrs

# Import our PTQ loader
import sys
import os
_QGEMM_PATH = "/workspace/qgemm/python"
if _QGEMM_PATH not in sys.path:
    sys.path.insert(0, _QGEMM_PATH)

from convert_llama import load_int4_safetensors


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class QgemmInt4PTQConfig(QuantizationConfig):
    """Config for qgemm INT4 weight-only linear with pre-quantized weights (PTQ).

    Expects a directory with pre-quantized safetensors files and index.json
    from our convert_llama.py script.

    Min capability: sm80 (A100/H100). Activations supported: float16.
    """

    def __init__(self, quantized_model_path: str, fuse_activation: Optional[str] = None) -> None:
        super().__init__()
        self.quantized_model_path = quantized_model_path
        self.fuse_activation = fuse_activation

        # Load the quantized model index
        index_path = os.path.join(quantized_model_path, "index.json")
        if not os.path.exists(index_path):
            raise ValueError(f"Pre-quantized model index not found at {index_path}")

        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.group_size = self.index.get("default_group_size", 64)
        print(f"[PTQ] Loaded quantized model index from {quantized_model_path}")
        print(f"[PTQ] Found {len(self.index['layers'])} quantized layers")

    def get_name(self) -> str:
        return "qgemm_int4_ptq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QgemmInt4PTQConfig:
        quantized_model_path = config.get("quantized_model_path")

        # If not provided in config, try environment variable
        if not quantized_model_path:
            quantized_model_path = os.getenv("QGEMM_PTQ_PATH")

        if not quantized_model_path:
            raise ValueError(
                "quantized_model_path is required for qgemm_int4_ptq. "
                "Provide it in config or set QGEMM_PTQ_PATH environment variable."
            )
        return cls(quantized_model_path=quantized_model_path)

    @classmethod
    def from_param_path(cls, param_path: str) -> QgemmInt4PTQConfig:
        """Create config from quantization param path (used by sglang)."""
        return cls(quantized_model_path=param_path)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            return QgemmInt4PTQLinearMethod(self, prefix)
        return None

    def get_linear_method(self) -> LinearMethodBase:
        return QgemmInt4PTQLinearMethod(self, prefix=None)

    def get_scaled_act_names(self) -> List[str]:
        return []


class QgemmInt4PTQLinearMethod(LinearMethodBase):
    """Linear method using qgemm INT4 CUDA ops with pre-quantized weights."""

    def __init__(self, quant_config: QgemmInt4PTQConfig, prefix: str = None) -> None:
        self.quant_config = quant_config
        self.prefix = prefix

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
        # Get the layer name from the prefix passed by sglang
        layer_name = self.prefix or "unknown"

        print(f"[PTQ] Creating weights for layer: {layer_name}")

        # Check if this layer was quantized using the original layer name with dots
        if layer_name not in self.quant_config.index["layers"]:
            print(f"[PTQ] Layer {layer_name} not found in quantized model, skipping")
            # Fall back to regular weights - this layer wasn't quantized
            return

        layer_info = self.quant_config.index["layers"][layer_name]
        layer_file = os.path.join(self.quant_config.quantized_model_path, layer_info["file"])

        print(f"[PTQ] Loading pre-quantized weights from {layer_file}")

        try:
            # Load the pre-quantized weights
            packed_w, scales, bias, metadata, (out_idx, outlier_w) = load_int4_safetensors(layer_file)

            # Move to appropriate device (assume cuda for now)
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            packed_w = packed_w.to(device)
            scales = scales.to(device)
            if bias is not None:
                bias = bias.to(device)

            print(f"[PTQ] Loaded weights: packed_w={packed_w.shape}, scales={scales.shape}, bias={bias.shape if bias is not None else None}")

            # Create parameters with the loaded data
            packed_w_param = Parameter(packed_w, requires_grad=False)
            scales_param = Parameter(scales, requires_grad=False)

            layer.register_parameter("packed_w", packed_w_param)
            layer.register_parameter("scales", scales_param)

            # Store metadata for later use
            layer.qgemm_metadata = metadata

            # Register a dummy weight parameter to satisfy sglang's weight loader
            # This won't be used during forward pass, but prevents loading errors
            Nshard = sum(output_partition_sizes)
            K = int(input_size_per_partition)
            weight_loader = extra_weight_attrs.get("weight_loader")
            dummy_weight = ModelWeightParameter(
                data=torch.empty(Nshard, K, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", dummy_weight)
            # Avoid overwriting existing attributes such as weight_loader
            extra_attrs = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
            if extra_attrs:
                set_weight_attrs(dummy_weight, extra_attrs)

            # Handle bias if present
            if bias is not None:
                bias_param = Parameter(bias, requires_grad=False)
                layer.register_parameter("bias", bias_param)

            print(f"[PTQ] Successfully registered parameters for {layer_name}")

        except Exception as e:
            print(f"[PTQ] Failed to load quantized weights for {layer_name}: {e}")
            # Fall back to regular weight loading
            raise

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check if this layer has quantized weights
        if not (hasattr(layer, "packed_w") and hasattr(layer, "scales")):
            # This layer wasn't quantized, should not happen if create_weights worked
            raise RuntimeError(f"Layer {layer.__class__.__name__} missing qgemm PTQ parameters")

        # Ensure qgemm ops are available
        if not hasattr(torch.ops, "qgemm") or not hasattr(torch.ops.qgemm, "int4_bias"):
            raise RuntimeError(
                "qgemm ops not available. Ensure qgemm is installed and built with CUDA."
            )

        packed_w: torch.Tensor = layer.packed_w
        scales: torch.Tensor = layer.scales

        # Get group_size from metadata
        group_size = getattr(layer, "qgemm_metadata", {}).get("group_size", self.quant_config.group_size)

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

        print(f"[PTQ] Applying kernel to {x_2d.shape} with group_size={group_size}, use_silu={use_silu}")

        # No padding needed with PTQ - we should have proper batch sizes
        m_size = x_2d.shape[0]

        try:
            if use_silu:
                y_2d = torch.ops.qgemm.int4_bias_silu(
                    x_2d, packed_w, scales, int(group_size), bias
                )
            else:
                y_2d = torch.ops.qgemm.int4_bias(
                    x_2d, packed_w, scales, int(group_size), bias
                )
        except Exception as err:
            err_str = str(err).lower()
            # Handle various kernel errors with fallback
            # Match the exact error pattern from C++ code: "qgemm.int4_bias: batch size M=X is too small (< 16)"
            is_small_batch = (
                ("batch size" in err_str and "too small" in err_str) or
                ("is too small" in err_str) or
                ("qgemm.int4_bias" in err_str and "too small" in err_str) or
                ("< 16" in err_str)
            )
            is_cuda_error = (
                "cuda error" in err_str or
                "acceleratorerror" in err_str or
                "unspecified launch failure" in err_str or
                "launch failure" in err_str
            )

            if is_small_batch or is_cuda_error:
                if is_small_batch:
                    print(f"[PTQ] Small batch fallback: {err}")
                else:
                    print(f"[PTQ] CUDA kernel error, using fallback: {err}")

                # Fall back to PyTorch native operations
                # Dequantize the weights for the fallback computation
                w_dequant = self._dequantize_weights(packed_w, scales, group_size)
                y_2d = torch.nn.functional.linear(x_2d, w_dequant, bias)
                if use_silu:
                    y_2d = torch.nn.functional.silu(y_2d)
                print(f"[PTQ] Fallback computation successful, output shape: {y_2d.shape}")
            else:
                print(f"[PTQ] kernel raised {err.__class__.__name__}: {err}")
                raise

        print(f"[PTQ] kernel succeeded, output shape: {y_2d.shape}")

        # Reshape back to original batch dims with output dim
        out_shape = x.shape[:-1] + (packed_w.shape[0],)
        y = y_2d.reshape(out_shape)

        return y

    def _dequantize_weights(self, packed_w: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
        """Dequantize INT4 packed weights back to FP16 for fallback computation."""
        # packed_w: [N, ceil(K/2)] uint8 (two nibbles per byte)
        # scales: [N, ceil(K/group_size)] float16
        # Returns: [N, K] float16

        N, packed_K = packed_w.shape
        K = packed_K * 2  # Two nibbles per byte

        device = packed_w.device

        # Unpack the nibbles
        # Each byte contains two 4-bit values: [high_nibble, low_nibble]
        w_unpacked = torch.zeros(N, K, dtype=torch.uint8, device=device)

        # Extract low nibbles (bits 0-3)
        w_unpacked[:, 0::2] = packed_w & 0xF
        # Extract high nibbles (bits 4-7)
        w_unpacked[:, 1::2] = (packed_w >> 4) & 0xF

        # Convert to signed values (INT4 range: -8 to +7)
        # Values > 7 (0x8-0xF) represent negative numbers (-8 to -1)
        w_int4 = w_unpacked.to(torch.int8)
        w_int4 = torch.where(w_int4 > 7, w_int4 - 16, w_int4)

        # Convert to float for dequantization
        w_float = w_int4.to(torch.float16)

        # Apply per-group scaling
        # scales: [N, ceil(K/group_size)]
        num_groups = scales.shape[1]
        w_dequant = torch.zeros_like(w_float)

        for g in range(num_groups):
            start_k = g * group_size
            end_k = min(start_k + group_size, K)
            if start_k < K:
                # Broadcast scale across the group
                group_scale = scales[:, g:g+1]  # [N, 1]
                w_dequant[:, start_k:end_k] = w_float[:, start_k:end_k] * group_scale

        return w_dequant