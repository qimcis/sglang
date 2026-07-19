"""Unit tests for protected DSA top-k dispatch."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSATopKProtection(CustomTestCase):
    def setUp(self):
        self.protection = {"request_indices": object()}
        self.metadata = SimpleNamespace(
            page_table_1=torch.zeros((2, 8), dtype=torch.int32),
            kv_page_protection=self.protection,
        )
        self.logits = torch.zeros((2, 8), dtype=torch.float32)
        self.lengths = torch.full((2,), 8, dtype=torch.int32)
        self.cu_seqlens_q = torch.arange(3, dtype=torch.int32)

    def test_sgl_fused_paged_path_forwards_protection(self):
        expected = torch.ones((2, 2048), dtype=torch.int32)
        fused = MagicMock(return_value=expected)
        fake_sgl_kernel = SimpleNamespace(
            fast_topk_transform_fused=fused,
            fast_topk_transform_ragged_fused=MagicMock(),
        )

        with envs.SGLANG_DSA_FUSE_TOPK.override(True), patch.dict(
            sys.modules, {"sgl_kernel": fake_sgl_kernel}
        ):
            result = DSATopKBackend.SGL_KERNEL.topk_transform(
                logits=self.logits,
                lengths=self.lengths,
                topk=2048,
                topk_transform_method=TopkTransformMethod.PAGED,
                attn_metadata=self.metadata,
                cu_seqlens_q_topk=self.cu_seqlens_q,
            )

        self.assertIs(result, expected)
        self.assertIs(fused.call_args.kwargs["kv_page_protection"], self.protection)

    def test_unsupported_protected_paths_fail_before_dispatch(self):
        cases = [
            (DSATopKBackend.FLASHINFER, {}),
            (DSATopKBackend.SGL_KERNEL, {"force_unfused_topk": True}),
            (
                DSATopKBackend.SGL_KERNEL,
                {"topk_transform_method": TopkTransformMethod.RAGGED},
            ),
            (
                DSATopKBackend.SGL_KERNEL,
                {"row_starts": torch.zeros(2, dtype=torch.int32)},
            ),
            (DSATopKBackend.SGL_KERNEL, {"batch_idx_list": [0]}),
        ]
        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            for backend, kwargs in cases:
                with self.subTest(backend=backend, kwargs=kwargs):
                    with self.assertRaisesRegex(
                        RuntimeError, "requires the SGL fused paged top-k decode path"
                    ):
                        call_kwargs = {
                            "topk_transform_method": TopkTransformMethod.PAGED,
                            **kwargs,
                        }
                        backend.topk_transform(
                            logits=self.logits,
                            lengths=self.lengths,
                            topk=2048,
                            attn_metadata=self.metadata,
                            cu_seqlens_q_topk=self.cu_seqlens_q,
                            **call_kwargs,
                        )


if __name__ == "__main__":
    unittest.main()
