import logging
from typing import Optional

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

_ORIGINAL_APPLY = UnquantizedLinearMethod.apply
_PATCHED = False
_FGEMM_READY = False
_FGEMM_IMPORT_FAILED = False
_FGEMM_WARNED_FALLBACK = False
_ensure_ops_loaded: Optional[callable] = None


def configure_fgemm_linear(enable: bool) -> bool:
    if enable:
        return _enable_fgemm_linear()
    _disable_fgemm_linear()
    return False


def _enable_fgemm_linear() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not _lazy_load_ops():
        return False

    def _fgemm_apply(self, layer, x, bias=None):
        if not _fgemm_applicable(layer, x):
            return _ORIGINAL_APPLY(self, layer, x, bias)
        try:
            return _apply_with_fgemm(layer, x, bias)
        except Exception:
            return _fallback_with_warning(self, layer, x, bias)

    UnquantizedLinearMethod.apply = _fgemm_apply
    _PATCHED = True
    logger.info("FGemm linear kernels enabled")
    return True


def _disable_fgemm_linear() -> None:
    global _PATCHED
    if _PATCHED:
        UnquantizedLinearMethod.apply = _ORIGINAL_APPLY
        _PATCHED = False


def _lazy_load_ops() -> bool:
    global _FGEMM_READY, _FGEMM_IMPORT_FAILED, _ensure_ops_loaded
    if _FGEMM_READY:
        return True
    if _FGEMM_IMPORT_FAILED:
        return False
    try:
        from fgemm.python.fused_linear_fp16 import _ensure_ops_loaded as ensure_ops
    except ImportError as exc:
        logger.warning("FGemm integration unavailable (missing module): %s", exc)
        _FGEMM_IMPORT_FAILED = True
        return False

    try:
        ensure_ops()
    except Exception as exc:
        logger.warning("FGemm integration unavailable (extension not loaded): %s", exc)
        _FGEMM_IMPORT_FAILED = True
        return False

    _ensure_ops_loaded = ensure_ops
    _FGEMM_READY = True
    return True


def _fgemm_applicable(layer, x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if not x.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if not weight.is_cuda:
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.size(-1) != x.shape[-1]:
        return False
    return True


def _apply_with_fgemm(layer, x: torch.Tensor, bias: Optional[torch.Tensor]):
    assert _ensure_ops_loaded is not None
    _ensure_ops_loaded()

    original_dtype = x.dtype
    in_shape = x.shape
    x_mat = x.reshape(-1, in_shape[-1])
    if not x_mat.is_contiguous():
        x_mat = x_mat.contiguous()
    if x_mat.dtype != layer.weight.dtype:
        x_mat = x_mat.to(layer.weight.dtype)

    weight = layer.weight
    if not weight.is_contiguous():
        weight = weight.contiguous()

    bias_tensor = bias
    if bias_tensor is not None:
        if bias_tensor.dtype != weight.dtype:
            bias_tensor = bias_tensor.to(weight.dtype)
        if not bias_tensor.is_contiguous():
            bias_tensor = bias_tensor.contiguous()

    output_2d = _run_fgemm(layer, x_mat, weight, bias_tensor)
    if output_2d.dtype != original_dtype:
        output_2d = output_2d.to(original_dtype)
    return output_2d.reshape(*in_shape[:-1], weight.size(0))


def _run_fgemm(layer, x_mat: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    if _should_fuse_gate(layer, weight):
        return _run_gate_split(layer, x_mat, weight, bias)
    return torch.ops.fgemm.fp16_bias(x_mat, weight, bias)


def _run_gate_split(layer, x_mat, weight, bias):
    output_sizes = getattr(layer, "output_sizes", None)
    tp_size = getattr(layer, "tp_size", 1)
    gate_rows = output_sizes[0] // tp_size
    up_rows = weight.size(0) - gate_rows

    gate_weight = weight.narrow(0, 0, gate_rows).contiguous()
    gate_bias = (
        bias.narrow(0, 0, gate_rows).contiguous() if bias is not None else None
    )
    gate_out = torch.ops.fgemm.fp16_bias_silu(x_mat, gate_weight, gate_bias)

    if up_rows == 0:
        return gate_out

    up_weight = weight.narrow(0, gate_rows, up_rows).contiguous()
    up_bias = (
        bias.narrow(0, gate_rows, up_rows).contiguous()
        if bias is not None
        else None
    )
    up_out = torch.ops.fgemm.fp16_bias(x_mat, up_weight, up_bias)
    return torch.cat((gate_out, up_out), dim=1)


def _should_fuse_gate(layer, weight: torch.Tensor) -> bool:
    if not isinstance(layer, MergedColumnParallelLinear):
        return False
    output_sizes = getattr(layer, "output_sizes", None)
    if not output_sizes or len(output_sizes) != 2:
        return False
    tp_size = getattr(layer, "tp_size", 1)
    if output_sizes[0] % tp_size != 0:
        return False
    if output_sizes[0] // tp_size <= 0:
        return False
    if output_sizes[0] // tp_size >= weight.size(0):
        return False
    prefix = getattr(layer, "prefix", "") or ""
    name = prefix.lower()
    return "gate" in name


def _fallback_with_warning(self, layer, x, bias):
    global _FGEMM_WARNED_FALLBACK
    if not _FGEMM_WARNED_FALLBACK:
        logger.warning("FGemm kernel failure detected, falling back to PyTorch linear", exc_info=True)
        _FGEMM_WARNED_FALLBACK = True
    return _ORIGINAL_APPLY(self, layer, x, bias)

