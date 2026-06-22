"""Tests for failure isolation: a corrupted page/transfer aborts ONLY the
affected request, leaving sibling requests verifiable, and the allocator sidecar
is gated (no overhead when disabled).
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_page_tags import (
    KVPageProtectionManager,
    KVPageTagMismatch,
    KVPageTagTable,
    KVProtectionConfig,
    compute_page_tag_scalar,
    tags_to_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MiniAllocator(BaseTokenToKVPoolAllocator):
    """Concrete minimal allocator to exercise the gated sidecar hook."""

    def __init__(self):
        # Bypass the abstract __init__ chain; only need the sidecar fields.
        self.page_tag_table = None

    def clear(self):  # pragma: no cover - abstract requirement
        pass

    def alloc(self, need_size):  # pragma: no cover - abstract requirement
        return None

    def free(self, free_index):  # pragma: no cover - abstract requirement
        pass


class TestAllocatorGating(CustomTestCase):
    def test_bump_is_noop_without_table(self):
        alloc = _MiniAllocator()
        # Must not raise and must not create any table implicitly.
        alloc._bump_page_generations(torch.tensor([1, 2, 3]))
        self.assertIsNone(alloc.page_tag_table)

    def test_attach_then_bump(self):
        alloc = _MiniAllocator()
        table = KVPageTagTable(num_pages=16, device="cpu")
        alloc.attach_page_tag_table(table)
        alloc._bump_page_generations(torch.tensor([4, 4, 5]))
        self.assertEqual(table.generation_of(torch.tensor([4]))[0].item(), 2)
        self.assertEqual(table.generation_of(torch.tensor([5]))[0].item(), 1)


class TestAbortIsolation(CustomTestCase):
    def _manager(self):
        return KVPageProtectionManager(
            KVProtectionConfig(enable_page_tags=True),
            allocator=None,
            num_pages=128,
            page_size=4,
            device="cpu",
            transfer_backend="mooncake",
        )

    def test_only_affected_request_aborts(self):
        mgr = self._manager()
        mgr.table.bump_generations(torch.tensor([10, 11, 20, 30, 31]))
        a = mgr.register_pages(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            page_physical_ids=[10, 11],
            bootstrap_room=100,
        )
        b = mgr.register_pages(
            token_ids=[9, 10, 11, 12], page_physical_ids=[20], bootstrap_room=200
        )
        c = mgr.register_pages(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            page_physical_ids=[30, 31],
            bootstrap_room=300,
        )

        # Corrupt only request B's physical page 20.
        mgr.table.bump_generations(torch.tensor([20]))
        g = int(mgr.table.generation_of(torch.tensor([20]))[0].item())
        mgr.table.write_tags(
            torch.tensor([20]),
            tags_to_tensor([compute_page_tag_scalar([0, 0, 0, 0], 0, 0, g, 4)]),
        )

        mismatches = mgr.verify_batch([("A", a), ("B", b), ("C", c)])
        bad_rids = {m.rid for m in mismatches}
        self.assertEqual(bad_rids, {"B"})

        # A and C still verify cleanly on their own.
        self.assertIsNone(mgr.verify_request(a, rid="A"))
        self.assertIsNone(mgr.verify_request(c, rid="C"))

    def test_mismatch_diagnostics(self):
        mgr = self._manager()
        mgr.table.bump_generations(torch.tensor([7]))
        man = mgr.register_pages(
            token_ids=[1, 2, 3, 4], page_physical_ids=[7], bootstrap_room=555
        )
        # Overwrite the page tag to force a mismatch.
        mgr.table.write_tags(torch.tensor([7]), tags_to_tensor([12345]))
        err = mgr.verify_request(man, rid="rid-x")
        self.assertIsInstance(err, KVPageTagMismatch)
        self.assertEqual(err.rid, "rid-x")
        self.assertEqual(err.bootstrap_room, 555)
        self.assertEqual(err.page_id, 7)
        self.assertEqual(err.page_position, 0)
        self.assertIsNotNone(err.expected_tag)
        self.assertIsNotNone(err.actual_tag)


if __name__ == "__main__":
    unittest.main()
