import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.ramp import (
    _load_ramp_profile,
    build_ramp_routing_stats,
    classify_ramp_bucket,
    maybe_dump_ramp_histogram,
    select_ramp_provider,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRampRoutingStats(CustomTestCase):
    def test_bucket_decode_tiny_extreme_skew(self):
        bucket = classify_ramp_bucket(
            total_assignments=64,
            active_experts=16,
            max_tokens_per_expert=49,
            mean_tokens_per_active_expert=4.0,
            singleton_frac=15 / 16,
        )
        self.assertEqual(bucket, "decode_tiny_extreme_skew")

    def test_bucket_many_tiny_experts(self):
        bucket = classify_ramp_bucket(
            total_assignments=96,
            active_experts=64,
            max_tokens_per_expert=4,
            mean_tokens_per_active_expert=1.5,
            singleton_frac=0.75,
        )
        self.assertEqual(bucket, "many_tiny_experts")

    def test_build_stats_from_dispatch_topk_ids(self):
        topk_ids = torch.tensor([[0, 0], [0, 1], [2, -1], [2, 2]], dtype=torch.int32)
        dispatch_output = SimpleNamespace(topk_ids=topk_ids)
        stats = build_ramp_routing_stats(
            dispatch_output, MoeRunnerConfig(num_experts=4, top_k=2)
        )

        self.assertIsNotNone(stats)
        fields = stats.to_log_dict(include_tokens_per_expert=True)
        self.assertEqual(fields["total_assignments"], 7)
        self.assertEqual(fields["active_experts"], 3)
        self.assertEqual(fields["tokens_per_expert"], [3, 1, 3, 0])
        self.assertEqual(fields["max_tokens_per_expert"], 3)
        self.assertAlmostEqual(fields["mean_tokens_per_active_expert"], 7 / 3)

    def test_dump_histogram_jsonl(self):
        topk_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int32)
        dispatch_output = SimpleNamespace(topk_ids=topk_ids)
        stats = build_ramp_routing_stats(
            dispatch_output, MoeRunnerConfig(num_experts=3, top_k=2, layer_id=7)
        )

        with tempfile.NamedTemporaryFile() as f:
            with (
                envs.SGLANG_MOE_RAMP_HISTOGRAM_PATH.override(f.name),
                envs.SGLANG_MOE_RAMP_HISTOGRAM_INTERVAL.override(1),
            ):
                maybe_dump_ramp_histogram(
                    stats,
                    call_count=1,
                    layer_id=7,
                    top_k=2,
                    runner_backend="triton",
                    dispatch_format="standard",
                )
            f.seek(0)
            line = f.read().decode("utf-8")

        self.assertIn('"layer_id":7', line)
        self.assertIn('"top_k":2', line)
        self.assertIn('"num_experts":3', line)
        self.assertIn('"tokens_per_expert":[1,3,0]', line)


class TestRampProviderSelection(CustomTestCase):
    def tearDown(self):
        _load_ramp_profile.cache_clear()

    def _build_many_tiny_stats(self):
        topk_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int32)
        dispatch_output = SimpleNamespace(topk_ids=topk_ids)
        return build_ramp_routing_stats(
            dispatch_output, MoeRunnerConfig(num_experts=3, top_k=2, layer_id=7)
        )

    def test_select_provider_wildcard_profile(self):
        stats = self._build_many_tiny_stats()

        with envs.SGLANG_MOE_RAMP_PROFILE_JSON.override(
            '{"entries":[{"winner":"triton"}]}'
        ):
            _load_ramp_profile.cache_clear()
            selection = select_ramp_provider(stats, layer_id=7, top_k=2)

        self.assertEqual(selection.provider, "triton")
        self.assertEqual(selection.source, "wildcard")
        self.assertEqual(selection.bucket, "many_tiny_experts")
        self.assertEqual(selection.num_tokens, 2)

    def test_select_provider_bucket_fallback(self):
        stats = self._build_many_tiny_stats()
        profile = (
            '{"entries":['
            '{"bucket":"many_tiny_experts","top_k":4,"winner":"deep_gemm"},'
            '{"bucket":"many_tiny_experts","top_k":6,"winner":"triton"},'
            '{"bucket":"many_tiny_experts","top_k":8,"winner":"triton"}'
            "]}"
        )

        with envs.SGLANG_MOE_RAMP_PROFILE_JSON.override(profile):
            _load_ramp_profile.cache_clear()
            selection = select_ramp_provider(stats, layer_id=7, top_k=2)

        self.assertEqual(selection.provider, "triton")
        self.assertEqual(selection.source, "bucket_fallback")
        self.assertEqual(selection.bucket, "many_tiny_experts")
        self.assertEqual(selection.num_tokens, 2)

    def test_select_provider_maps_unsafe_provider(self):
        stats = self._build_many_tiny_stats()
        profile = (
            '{"provider_fallbacks":{"deep_gemm":"triton"},"entries":['
            '{"layer_id":7,"bucket":"many_tiny_experts","top_k":2,'
            '"num_tokens":2,"winner":"deep_gemm"}'
            "]}"
        )

        with envs.SGLANG_MOE_RAMP_PROFILE_JSON.override(profile):
            _load_ramp_profile.cache_clear()
            selection = select_ramp_provider(stats, layer_id=7, top_k=2)

        self.assertEqual(selection.provider, "triton")
        self.assertEqual(selection.source, "exact_provider_fallback")

    def test_select_provider_uses_fallback_provider_on_miss(self):
        stats = self._build_many_tiny_stats()
        profile = (
            '{"fallback_provider":"triton","entries":['
            '{"bucket":"decode_medium","winner":"deep_gemm"}'
            "]}"
        )

        with envs.SGLANG_MOE_RAMP_PROFILE_JSON.override(profile):
            _load_ramp_profile.cache_clear()
            selection = select_ramp_provider(stats, layer_id=7, top_k=2)

        self.assertEqual(selection.provider, "triton")
        self.assertEqual(selection.source, "fallback_provider")
        self.assertEqual(selection.bucket, "many_tiny_experts")
        self.assertEqual(selection.num_tokens, 2)


if __name__ == "__main__":
    unittest.main()
