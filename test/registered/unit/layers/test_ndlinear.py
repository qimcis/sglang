import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.ndlinear_moe import NdLinearFusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.ndlinear import NdColumnParallelLinear, NdReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestNdLinear(CustomTestCase):
    def test_replicated_forward_matches_mode_products(self):
        layer = NdReplicatedLinear((2, 3), (4, 5), params_dtype=torch.float32)
        with torch.no_grad():
            layer.nd_factors[0].weight.copy_(torch.randn(4, 2))
            layer.nd_factors[1].weight.copy_(torch.randn(5, 3))

        x = torch.randn(7, 6)
        out, bias = layer(x)
        expected = torch.einsum(
            "bxy,ox,py->bop",
            x.view(7, 2, 3),
            layer.nd_factors[0].weight,
            layer.nd_factors[1].weight,
        ).reshape(7, 20)
        self.assertIsNone(bias)
        torch.testing.assert_close(out, expected)

    def test_column_loader_shards_output_axis(self):
        layer = NdColumnParallelLinear(
            (2, 2),
            (4, 2),
            params_dtype=torch.float32,
            output_tp_axis=0,
            tp_rank=1,
            tp_size=2,
        )
        full_factor = torch.arange(8, dtype=torch.float32).view(4, 2)
        layer.nd_factors[0].weight.weight_loader(
            layer.nd_factors[0].weight, full_factor
        )
        torch.testing.assert_close(layer.nd_factors[0].weight, full_factor[2:])

    def test_moe_factor_loader_and_forward(self):
        with (
            patch(
                "sglang.srt.layers.moe.ndlinear_moe."
                "get_moe_expert_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.moe.ndlinear_moe." "get_moe_tensor_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.ndlinear_moe."
                "get_moe_tensor_parallel_world_size",
                return_value=1,
            ),
        ):
            moe = NdLinearFusedMoE(
                num_experts=1,
                hidden_shape=(2, 2),
                intermediate_shape=(2, 2),
                params_dtype=torch.float32,
            )

        eye = torch.eye(2)
        for factors in (moe.w1_factors, moe.w2_factors, moe.w3_factors):
            for factor in factors:
                factor.weight_loader(factor, eye, expert_id=0)

        x = torch.randn(3, 4)
        topk = StandardTopKOutput(
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int32),
            router_logits=torch.zeros(3, 1),
        )
        out = moe(x, topk)
        expected = torch.nn.functional.silu(x) * x
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
