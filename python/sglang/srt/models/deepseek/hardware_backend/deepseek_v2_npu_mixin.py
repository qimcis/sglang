from __future__ import annotations

from ..utils import (
    _is_npu,
    forward_dsa_core_npu,
    forward_dsa_prepare_npu,
    forward_mha_core_npu,
    forward_mha_prepare_npu,
    forward_mla_core_npu,
    forward_mla_prepare_npu,
)


class DeepseekV2NPUMixin:
    def forward_mha_prepare_npu(self, *args, **kwargs):
        if not _is_npu or forward_mha_prepare_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_mha_prepare_npu(self, *args, **kwargs)

    def forward_mla_prepare_npu(self, *args, **kwargs):
        if not _is_npu or forward_mla_prepare_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_mla_prepare_npu(self, *args, **kwargs)

    def forward_dsa_prepare_npu(self, *args, **kwargs):
        if not _is_npu or forward_dsa_prepare_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_dsa_prepare_npu(self, *args, **kwargs)

    def forward_mha_core_npu(self, *args, **kwargs):
        if not _is_npu or forward_mha_core_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_mha_core_npu(self, *args, **kwargs)

    def forward_mla_core_npu(self, *args, **kwargs):
        if not _is_npu or forward_mla_core_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_mla_core_npu(self, *args, **kwargs)

    def forward_dsa_core_npu(self, *args, **kwargs):
        if not _is_npu or forward_dsa_core_npu is None:
            raise RuntimeError("NPU attention backend is not available.")
        return forward_dsa_core_npu(self, *args, **kwargs)
