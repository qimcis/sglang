"""Correctness tests for fused FA3 KV page protection."""

import unittest

import pytest
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")

_SKIP = not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9
_SENTINEL = 123.0


def _make_case(*, seqlen=4, page_size=1):
    batch_size = 2
    num_heads = 8
    qk_head_dim = 64
    v_head_dim = 512
    pages_per_request = (seqlen + page_size - 1) // page_size
    num_pages = batch_size * pages_per_request
    device = "cuda"

    q = torch.randn(
        batch_size,
        1,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    qv = torch.randn(
        batch_size,
        1,
        num_heads,
        v_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages + 1,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        num_pages + 1,
        page_size,
        1,
        v_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(1, num_pages + 1, device=device, dtype=torch.int32).view(
        batch_size, pages_per_request
    )
    seqlens = torch.full((batch_size,), seqlen, device=device, dtype=torch.int32)
    request_indices = torch.tensor([1, 2], device=device, dtype=torch.int64)

    actual_tags = torch.arange(num_pages + 1, device=device, dtype=torch.int64)
    actual_generations = torch.ones_like(actual_tags)
    actual_transfer_tags = torch.zeros(num_pages + 1, device=device, dtype=torch.int32)
    owner_request_indices = torch.full(
        (num_pages + 1,), -1, device=device, dtype=torch.int32
    )
    owner_request_indices[page_table[0].long()] = 1
    owner_request_indices[page_table[1].long()] = 2
    owner_page_positions = torch.full_like(owner_request_indices, -1)
    owner_page_positions[page_table[0].long()] = torch.arange(
        pages_per_request, device=device, dtype=torch.int32
    )
    owner_page_positions[page_table[1].long()] = torch.arange(
        pages_per_request, device=device, dtype=torch.int32
    )

    protection = {
        "request_indices": request_indices,
        "seqlens": seqlens,
        "page_table": page_table,
        "page_size": page_size,
        "actual_tags": actual_tags,
        "actual_generations": actual_generations,
        "actual_transfer_tags": actual_transfer_tags,
        "owner_request_indices": owner_request_indices,
        "owner_page_positions": owner_page_positions,
        "expected_tags": actual_tags.clone(),
        "expected_generations": actual_generations.clone(),
        "expected_transfer_tags": actual_transfer_tags.clone(),
        "request_epochs": torch.tensor([0, 1, 1], device=device, dtype=torch.int32),
        "validated_epochs": torch.full((3,), -1, device=device, dtype=torch.int32),
        "status": torch.zeros(3, device=device, dtype=torch.int32),
    }
    return q, qv, k_cache, v_cache, page_table, seqlens, protection


def _run_fa3_kv_page_protection_fail_closed(num_splits):
    seqlen = 4096 if num_splits > 1 else 4
    page_size = 64 if num_splits > 1 else 1
    q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case(
        seqlen=seqlen, page_size=page_size
    )
    baseline = flash_attn_with_kvcache(
        q=q,
        qv=qv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        num_splits=num_splits,
    )

    bad_page = int(page_table[1, 0].item())
    protection["expected_tags"][bad_page] += 1
    out = torch.full_like(qv, _SENTINEL)
    result = flash_attn_with_kvcache(
        q=q,
        qv=qv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        num_splits=num_splits,
        out=out,
        kv_page_protection=protection,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result[0], baseline[0])
    assert torch.all(result[1] == _SENTINEL)
    assert protection["status"][1].item() == 0
    assert protection["status"][2].item() & (1 << 3)
    assert protection["validated_epochs"][1:3].tolist() == [1, 1]


class TestFA3KVPageProtection(CustomTestCase):
    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_absorbed_mla_valid_page_size_64(self):
        q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case(
            seqlen=129, page_size=64
        )
        baseline = flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
        )
        protected = flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
            kv_page_protection=protection,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(protected, baseline)
        self.assertEqual(protection["status"].tolist(), [0, 0, 0])
        self.assertEqual(protection["validated_epochs"].tolist(), [-1, 1, 1])

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_fail_closed_non_split(self):
        _run_fa3_kv_page_protection_fail_closed(num_splits=1)

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_fail_closed_split(self):
        _run_fa3_kv_page_protection_fail_closed(num_splits=2)

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_request_slot_zero_bypasses_validation(self):
        q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case()
        baseline = flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
        )
        protection["request_indices"][1] = 0
        protection["owner_request_indices"][page_table[1].long()] = -1
        protected = flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
            kv_page_protection=protection,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(protected, baseline)
        self.assertEqual(protection["status"].tolist(), [0, 0, 0])
        self.assertEqual(protection["validated_epochs"].tolist(), [-1, 1, -1])

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_absorbed_mla_cuda_graph_replay(self):
        q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case(
            seqlen=129, page_size=64
        )
        baseline = flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
        )
        out = torch.empty_like(qv)
        flash_attn_with_kvcache(
            q=q,
            qv=qv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            num_splits=1,
            out=out,
            kv_page_protection=protection,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flash_attn_with_kvcache(
                q=q,
                qv=qv,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=seqlens,
                num_splits=1,
                out=out,
                kv_page_protection=protection,
            )

        protection["request_epochs"][1:].add_(1)
        protection["status"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, baseline)
        self.assertEqual(protection["validated_epochs"].tolist(), [-1, 2, 2])

        bad_page = int(page_table[1, 0].item())
        protection["expected_tags"][bad_page] += 1
        protection["request_epochs"][1:].add_(1)
        protection["status"].zero_()
        out.fill_(_SENTINEL)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(out[0], baseline[0])
        self.assertTrue(torch.all(out[1] == _SENTINEL))
        self.assertEqual(protection["status"][1].item(), 0)
        self.assertTrue(protection["status"][2].item() & (1 << 3))
        self.assertEqual(protection["validated_epochs"].tolist(), [-1, 3, 3])

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_rejects_partial_protection_metadata(self):
        q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case()
        with self.assertRaisesRegex(RuntimeError, "metadata is incomplete"):
            flash_attn_with_kvcache(
                q=q,
                qv=qv,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=seqlens,
                num_splits=1,
                kv_page_protection={"actual_tags": protection["actual_tags"]},
            )

    @unittest.skipIf(_SKIP, "fused FA3 protection requires Hopper")
    def test_rejects_cache_batch_remapping(self):
        q, qv, k_cache, v_cache, page_table, seqlens, protection = _make_case()
        with self.assertRaisesRegex(RuntimeError, "cache_batch_idx remapping"):
            flash_attn_with_kvcache(
                q=q,
                qv=qv,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=seqlens,
                cache_batch_idx=torch.arange(2, device="cuda", dtype=torch.int32),
                num_splits=1,
                kv_page_protection=protection,
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
