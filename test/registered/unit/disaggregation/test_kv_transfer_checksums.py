"""Unit tests for KV transfer checksums.

Key property under test: checksums are computed in *logical* token order and
NEVER include node-local physical page/slot ids, so differing prefill/decode
physical layouts cannot cause false failures.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_page_tags import (
    ChecksumPlan,
    KVChecksumError,
    KVPageProtectionManager,
    KVProtectionConfig,
    compare_checksums,
    compute_transfer_checksum,
    direct_kv_checksum_from_loc,
    gather_logical_kv_rows,
    select_checksum_token_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakePool:
    """Minimal KV cache exposing the per-layer accessors the gather needs."""

    def __init__(self, k_buffers):
        self.layer_num = len(k_buffers)
        self._k = k_buffers

    def get_key_buffer(self, layer_id):
        return self._k[layer_id]

    def get_value_buffer(self, layer_id):
        raise NotImplementedError


class _FakeMetrics:
    def __init__(self):
        self.checked = 0
        self.mismatches = 0

    def increment_kv_transfer_checksum_checked_pages(self, n):
        self.checked += n

    def increment_kv_transfer_checksum_mismatches(self, n=1):
        self.mismatches += n


class TestChecksumRowLevel(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.N = 16
        self.rows = torch.randint(0, 1000, (self.N, 8), dtype=torch.int32)
        self.cfg = KVProtectionConfig(enable_transfer_checksum=True)

    def _cksum(self, rows):
        return compute_transfer_checksum(
            rows, bootstrap_room=42, num_tokens=self.N, config=self.cfg
        ).checksum

    def test_identical_bytes_match(self):
        self.assertEqual(self._cksum(self.rows), self._cksum(self.rows.clone()))

    def test_corruption_detected(self):
        bad = self.rows.clone()
        bad[5, 2] += 1
        self.assertNotEqual(self._cksum(self.rows), self._cksum(bad))

    def test_reorder_detected(self):
        re = self.rows.clone()
        re[[3, 4]] = re[[4, 3]]
        self.assertNotEqual(self._cksum(self.rows), self._cksum(re))


class TestChecksumExcludesPhysicalPageIds(CustomTestCase):
    """The core acceptance criterion from the prior rejected patch."""

    def _manager(self, metrics=None):
        cfg = KVProtectionConfig(enable_transfer_checksum=True)
        return KVPageProtectionManager(
            cfg,
            allocator=None,
            num_pages=0,
            page_size=4,
            device="cpu",
            metrics_collector=metrics,
            transfer_backend="mooncake",
        )

    def test_same_logical_bytes_different_physical_slots_match(self):
        torch.manual_seed(1)
        size, h, d, L, N = 64, 2, 4, 2, 10
        src = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        loc_src = torch.tensor([5, 6, 7, 20, 21, 22, 40, 41, 42, 50])
        mgr = self._manager()
        rows_src = gather_logical_kv_rows(_FakePool(src), loc_src, torch.arange(N))
        plan = mgr.compute_source_checksum(rows_src, bootstrap_room=99, num_tokens=N)

        # Decode places the SAME logical content at DIFFERENT physical slots.
        loc_dst = torch.tensor([1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
        dst = {l: torch.zeros(size, h, d, dtype=torch.int32) for l in range(L)}
        for l in range(L):
            dst[l][loc_dst] = src[l][loc_src]
        rows_dst = gather_logical_kv_rows(_FakePool(dst), loc_dst, torch.arange(N))
        err = mgr.verify_destination_checksum(
            rows_dst,
            bootstrap_room=99,
            num_tokens=N,
            expected=plan,
            rid="r1",
        )
        self.assertIsNone(err)

    def test_byte_corruption_after_transfer_detected(self):
        torch.manual_seed(2)
        size, h, d, L, N = 64, 2, 4, 2, 8
        src = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        loc_src = torch.arange(N)
        mgr = self._manager(_FakeMetrics())
        rows_src = gather_logical_kv_rows(_FakePool(src), loc_src, torch.arange(N))
        plan = mgr.compute_source_checksum(rows_src, bootstrap_room=7, num_tokens=N)
        dst = {l: src[l].clone() for l in range(L)}
        dst[0][3, 0, 0] += 1  # flip a destination byte
        rows_dst = gather_logical_kv_rows(_FakePool(dst), loc_src, torch.arange(N))
        err = mgr.verify_destination_checksum(
            rows_dst,
            bootstrap_room=7,
            num_tokens=N,
            expected=plan,
            rid="r1",
        )
        self.assertIsInstance(err, KVChecksumError)

    def test_unsupported_pool_fails_fast(self):
        class BadPool:
            pass

        with self.assertRaises(RuntimeError):
            gather_logical_kv_rows(BadPool(), torch.arange(4), torch.arange(4))


class TestChecksumSelectionAndPayload(CustomTestCase):
    def test_selects_all_tokens(self):
        idx = select_checksum_token_indices(50, 1, 1.0)
        self.assertEqual(idx.numel(), 50)

    def test_plan_payload_roundtrip(self):
        plan = ChecksumPlan(
            bootstrap_room=5,
            num_tokens=12,
            checksum=(1 << 63) + 123,  # exercises uint64 bit pattern
        )
        restored = ChecksumPlan.from_payload(plan.to_payload())
        self.assertEqual(restored.bootstrap_room, plan.bootstrap_room)
        self.assertEqual(restored.num_tokens, plan.num_tokens)
        # checksum compares equal as int64 bit patterns
        self.assertTrue(compare_checksums(plan, restored.checksum))


class _FakePoolKV:
    """Pool exposing both K and V per-layer buffers."""

    def __init__(self, k, v):
        self.layer_num = len(k)
        self._k = k
        self._v = v

    def get_key_buffer(self, layer_id):
        return self._k[layer_id]

    def get_value_buffer(self, layer_id):
        return self._v[layer_id]


class TestDirectKernelRequired(CustomTestCase):
    """The direct-KV checksum kernel is required for loc-based checksums."""

    def _manager(self):
        cfg = KVProtectionConfig(enable_transfer_checksum=True)
        return KVPageProtectionManager(
            cfg,
            allocator=None,
            num_pages=0,
            page_size=4,
            device="cpu",
            transfer_backend="mooncake",
        )

    def test_direct_kernel_raises_on_cpu(self):
        k = [torch.randn(32, 2, 8) for _ in range(2)]
        v = [torch.randn(32, 2, 8) for _ in range(2)]
        cfg = KVProtectionConfig(enable_transfer_checksum=True)
        with self.assertRaises(RuntimeError):
            direct_kv_checksum_from_loc(
                _FakePoolKV(k, v),
                torch.arange(32),
                torch.arange(8),
                config=cfg,
            )

    def test_manager_loc_checksum_raises_on_cpu(self):
        torch.manual_seed(7)
        size, h, d, L, N = 64, 2, 4, 2, 10
        k = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        v = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        loc = torch.arange(N)
        manager = self._manager()
        with self.assertRaises(RuntimeError):
            manager.compute_source_checksum_from_loc(
                _FakePoolKV(k, v), loc, bootstrap_room=3, num_tokens=N
            )


if __name__ == "__main__":
    unittest.main()
