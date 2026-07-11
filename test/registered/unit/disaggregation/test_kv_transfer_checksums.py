"""Unit tests for KV transfer checksum semantics.

These CPU tests cover the Python reference hash properties. CUDA parity for the
production table-batched op lives in ``sgl-kernel/tests/test_kv_checksum.py``.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_page_tags import (
    ChecksumPlan,
    KVPageProtectionManager,
    KVProtectionConfig,
    compare_checksums,
    hash_rows_with_positions,
    select_checksum_token_indices,
    swa_checksum_evicted_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeMetrics:
    def __init__(self):
        self.checked = 0
        self.mismatches = 0

    def increment_kv_transfer_checksum_checked_pages(self, n):
        self.checked += n

    def increment_kv_transfer_checksum_mismatches(self, n=1):
        self.mismatches += n


def _checksum(rows: torch.Tensor) -> int:
    positions = torch.arange(rows.shape[0], dtype=torch.long)
    return hash_rows_with_positions(rows, positions=positions)


def _gather_logical_rows(buffers, locs):
    rows = [buf.index_select(0, locs).reshape(locs.numel(), -1) for buf in buffers]
    return torch.cat(rows, dim=1)


class TestChecksumRowLevel(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rows = torch.randint(0, 1000, (16, 8), dtype=torch.int32)

    def test_identical_bytes_match(self):
        self.assertEqual(_checksum(self.rows), _checksum(self.rows.clone()))

    def test_corruption_detected(self):
        bad = self.rows.clone()
        bad[5, 2] += 1
        self.assertNotEqual(_checksum(self.rows), _checksum(bad))

    def test_reorder_detected(self):
        reordered = self.rows.clone()
        reordered[[3, 4]] = reordered[[4, 3]]
        self.assertNotEqual(_checksum(self.rows), _checksum(reordered))


class TestChecksumExcludesPhysicalPageIds(CustomTestCase):
    """Checksums are over logical bytes/positions, never physical page ids."""

    def test_same_logical_bytes_different_physical_slots_match(self):
        torch.manual_seed(1)
        size, h, d, layers, tokens = 64, 2, 4, 2, 10
        src = [
            torch.randint(0, 100, (size, h, d), dtype=torch.int32)
            for _ in range(layers)
        ]
        loc_src = torch.tensor([5, 6, 7, 20, 21, 22, 40, 41, 42, 50])
        rows_src = _gather_logical_rows(src, loc_src)
        source_checksum = _checksum(rows_src)

        loc_dst = torch.tensor([1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
        dst = [torch.zeros(size, h, d, dtype=torch.int32) for _ in range(layers)]
        for layer_id in range(layers):
            dst[layer_id][loc_dst] = src[layer_id][loc_src]

        rows_dst = _gather_logical_rows(dst, loc_dst)
        self.assertEqual(source_checksum, _checksum(rows_dst))

    def test_byte_corruption_after_transfer_detected(self):
        torch.manual_seed(2)
        size, h, d, layers, tokens = 64, 2, 4, 2, 8
        src = [
            torch.randint(0, 100, (size, h, d), dtype=torch.int32)
            for _ in range(layers)
        ]
        loc = torch.arange(tokens)
        source_checksum = _checksum(_gather_logical_rows(src, loc))

        dst = [buf.clone() for buf in src]
        dst[0][3, 0, 0] += 1
        self.assertNotEqual(source_checksum, _checksum(_gather_logical_rows(dst, loc)))


class TestChecksumSelectionAndPayload(CustomTestCase):
    def test_selects_all_tokens(self):
        idx = select_checksum_token_indices(50, 1, 1.0)
        self.assertEqual(idx.numel(), 50)

    def test_plan_payload_roundtrip(self):
        plan = ChecksumPlan(
            bootstrap_room=5,
            num_tokens=12,
            checksum=(1 << 31) + 123,
        )
        restored = ChecksumPlan.from_payload(plan.to_payload())
        self.assertEqual(restored.bootstrap_room, plan.bootstrap_room)
        self.assertEqual(restored.num_tokens, plan.num_tokens)
        self.assertTrue(compare_checksums(plan, restored.checksum))

    def test_compare_uses_uint32_bits(self):
        plan = ChecksumPlan(bootstrap_room=1, num_tokens=2, checksum=0xFFFF_FFFE)
        self.assertTrue(compare_checksums(plan, -2))
        self.assertFalse(compare_checksums(plan, 0xFFFF_FFFD))

    def test_swa_evicted_len_is_page_aligned(self):
        self.assertEqual(swa_checksum_evicted_len(100, 32, 16), 64)
        self.assertEqual(swa_checksum_evicted_len(100, 32, 1), 68)
        self.assertEqual(swa_checksum_evicted_len(31, 32, 16), 0)
        self.assertEqual(swa_checksum_evicted_len(100, None, 16), 0)


class TestManagerCompare(CustomTestCase):
    def test_compare_destination_checksum_updates_metrics(self):
        metrics = _FakeMetrics()
        manager = KVPageProtectionManager(
            KVProtectionConfig(enable_transfer_checksum=True),
            allocator=None,
            num_pages=0,
            page_size=4,
            device="cpu",
            metrics_collector=metrics,
            transfer_backend="mooncake",
        )
        expected = ChecksumPlan(bootstrap_room=3, num_tokens=8, checksum=123)
        self.assertTrue(manager.compare_destination_checksum(expected, 123, rid="r"))
        self.assertFalse(manager.compare_destination_checksum(expected, 124, rid="r"))
        self.assertEqual(metrics.checked, 16)
        self.assertEqual(metrics.mismatches, 1)

    def test_batched_table_requires_cuda_req_to_token(self):
        manager = KVPageProtectionManager(
            KVProtectionConfig(enable_transfer_checksum=True),
            allocator=None,
            num_pages=0,
            page_size=4,
            device="cpu",
            transfer_backend="mooncake",
        )
        with self.assertRaises(RuntimeError):
            manager.begin_transfer_checksums_from_table(
                object(),
                torch.zeros((1, 4), dtype=torch.int32),
                req_pool_indices=[0],
                bootstrap_rooms=[1],
                num_tokens=[4],
            )


if __name__ == "__main__":
    unittest.main()
