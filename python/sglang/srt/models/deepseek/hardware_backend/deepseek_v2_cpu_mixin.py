from __future__ import annotations

from ..utils import _is_cpu, _is_cpu_amx_available


class DeepseekV2CpuMixin:
    @staticmethod
    def cpu_supports_amx() -> bool:
        return _is_cpu and _is_cpu_amx_available
