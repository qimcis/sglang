from __future__ import annotations

"""
Quantization controller for KV cache (int8/per-head) with error-aware fallback.

This is a skeleton implementation providing the interfaces and light logic.
Integration is feature-flagged via ServerArgs and can be extended to fuse
dequantization with attention kernels or a scratch-buffer path.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class QuantConfig:
    mode: Optional[str] = None  # None, "int8_head", "nf4", "fp8"
    calib_steps: int = 2
    error_threshold: float = 0.02
    fallback: str = "head"  # "head", "layer", "off"
    per_block_scale: bool = False


class QuantController:
    """Tracks per-layer/head calibration, scales and fallback state."""

    def __init__(self, num_layers: int, num_heads: int, cfg: QuantConfig) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg = cfg

        # State: calibration counts and scales
        self._calib_left: Dict[int, int] = {i: cfg.calib_steps for i in range(num_layers)}
        self._k_scales: Dict[int, torch.Tensor] = {}
        self._v_scales: Dict[int, torch.Tensor] = {}
        self._head_fallback: Dict[Tuple[int, int], bool] = {}
        self._layer_fallback: Dict[int, bool] = {}

    def begin_calibration(self, layer_id: int) -> None:
        # no-op placeholder (useful if we later need to lazily init buffers per layer)
        if layer_id not in self._k_scales:
            self._k_scales[layer_id] = torch.ones(self.num_heads, dtype=torch.float32, device="cuda")
            self._v_scales[layer_id] = torch.ones(self.num_heads, dtype=torch.float32, device="cuda")

    def update_observation(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update statistics from FP activations during the calibration window.

        k, v: [tokens, heads, head_dim]
        """
        if self._calib_left.get(layer_id, 0) <= 0:
            return
        # symmetric per-head scale: max-abs / 127
        with torch.no_grad():
            k_max = k.abs().amax(dim=(0, 2), keepdim=False).clamp(min=1e-6)
            v_max = v.abs().amax(dim=(0, 2), keepdim=False).clamp(min=1e-6)
            self._k_scales[layer_id] = torch.maximum(self._k_scales[layer_id], k_max / 127.0)
            self._v_scales[layer_id] = torch.maximum(self._v_scales[layer_id], v_max / 127.0)
            self._calib_left[layer_id] -= 1

    def finalize(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-head scales for the layer after calibration."""
        if layer_id not in self._k_scales:
            self.begin_calibration(layer_id)
        return self._k_scales[layer_id], self._v_scales[layer_id]

    def mark_error_and_maybe_fallback(
        self, layer_id: int, head_id: Optional[int], error_value: float
    ) -> None:
        if self.cfg.fallback == "off":
            return
        if error_value <= self.cfg.error_threshold:
            return
        if self.cfg.fallback == "layer" or head_id is None:
            self._layer_fallback[layer_id] = True
        else:
            self._head_fallback[(layer_id, head_id)] = True

    def should_bypass(self, layer_id: int, head_id: Optional[int]) -> bool:
        if self._layer_fallback.get(layer_id, False):
            return True
        if head_id is None:
            return False
        return self._head_fallback.get((layer_id, head_id), False)

    def get_scales(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_scales[layer_id], self._v_scales[layer_id]

