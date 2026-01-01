from __future__ import annotations

from ..utils import (
    _is_gfx95_supported,
    _use_aiter,
    get_dsv3_gemm_output_zero_allocator_size,
)


class DeepseekV2AmdMixin:
    @staticmethod
    def use_aiter_gfx95():
        return _use_aiter and _is_gfx95_supported

    @staticmethod
    def rocm_zero_allocator_size(
        n_routed_experts: int, num_moe_layers: int, allocate_size: int, embed_dim: int
    ) -> int:
        if get_dsv3_gemm_output_zero_allocator_size is None:
            return 0
        return get_dsv3_gemm_output_zero_allocator_size(
            n_routed_experts, num_moe_layers, allocate_size, embed_dim
        )
