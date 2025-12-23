import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    _DenoiseOverlapContext,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def test_double_buffer_flag_default_false():
    args = ServerArgs.from_dict({"model_path": "dummy"})
    assert args.enable_double_buffered_denoising is False


def test_double_buffer_flag_can_be_enabled():
    args = ServerArgs.from_dict(
        {"model_path": "dummy", "enable_double_buffered_denoising": True}
    )
    assert args.enable_double_buffered_denoising is True


def test_overlap_ctx_respects_platform():
    ctx = _DenoiseOverlapContext(enabled=True)
    if current_platform.is_cuda_alike():
        assert ctx.use_overlap() is True
    else:
        # No CUDA/ROCm in environment, should gracefully disable overlap
        assert ctx.use_overlap() is False
