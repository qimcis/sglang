"""Tests for failure isolation and allocator sidecar gating."""

import unittest

import torch

from sglang.srt.mem_cache.allocator.base import (
    BaseTokenToKVPoolAllocator,
    KVTransferPagePinManager,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_page_tags import (
    KVAttentionTagMismatch,
    KVAttentionTagTable,
    KVPageProtectionManager,
    KVProtectionConfig,
    KVTransferPageTagMismatch,
    compute_attention_tag_scalar,
    compute_transfer_page_tag_scalar,
    tags_to_tensor,
    transfer_page_tags_to_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MiniAllocator(BaseTokenToKVPoolAllocator):
    """Concrete minimal allocator to exercise the gated sidecar hook."""

    def __init__(self):
        self.attention_tag_table = None

    def clear(self):  # pragma: no cover - abstract requirement
        pass

    def alloc(self, need_size):  # pragma: no cover - abstract requirement
        return None

    def free(self, free_index):  # pragma: no cover - abstract requirement
        pass


class TestAllocatorGating(CustomTestCase):
    def test_bump_is_noop_without_table(self):
        alloc = _MiniAllocator()
        alloc._bump_page_generations(torch.tensor([1, 2, 3]))
        self.assertIsNone(alloc.attention_tag_table)

    def test_attach_then_bump(self):
        alloc = _MiniAllocator()
        table = KVAttentionTagTable(num_pages=16, device="cpu")
        alloc.attach_attention_tag_table(table)
        alloc._bump_page_generations(torch.tensor([4, 4, 5]))
        self.assertEqual(table.generation_of(torch.tensor([4]))[0].item(), 2)
        self.assertEqual(table.generation_of(torch.tensor([5]))[0].item(), 1)


class TestTransferPagePinning(CustomTestCase):
    def _allocator(self):
        alloc = TokenToKVPoolAllocator(
            size=8,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        manager = KVTransferPagePinManager(alloc)
        alloc.attach_transfer_page_pin_manager(manager)
        return alloc, manager

    def test_free_is_deferred_until_pin_release(self):
        alloc, manager = self._allocator()
        allocated = alloc.alloc(4)
        self.assertEqual(allocated.tolist(), [1, 2, 3, 4])

        manager.pin_pages([2, 3])
        alloc.free(torch.tensor([2, 3, 4]))
        self.assertNotIn(2, alloc.free_pages.tolist())
        self.assertNotIn(3, alloc.free_pages.tolist())
        self.assertIn(4, alloc.free_pages.tolist())

        manager.release_pages([2])
        self.assertIn(2, alloc.free_pages.tolist())
        self.assertNotIn(3, alloc.free_pages.tolist())

        manager.release_pages([3])
        self.assertIn(3, alloc.free_pages.tolist())

    def test_duplicate_pins_require_duplicate_releases(self):
        alloc, manager = self._allocator()
        alloc.alloc(3)

        manager.pin_pages([2])
        manager.pin_pages([2])
        alloc.free(torch.tensor([2]))
        manager.release_pages([2])
        self.assertNotIn(2, alloc.free_pages.tolist())

        manager.release_pages([2])
        self.assertIn(2, alloc.free_pages.tolist())

    def test_paged_allocator_defers_physical_page_free(self):
        alloc = PagedTokenToKVPoolAllocator(
            size=16,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        manager = KVTransferPagePinManager(alloc)
        alloc.attach_transfer_page_pin_manager(manager)

        kv_indices = alloc.alloc(8)
        self.assertEqual(kv_indices.tolist(), [4, 5, 6, 7, 8, 9, 10, 11])

        manager.pin_pages([1])
        alloc.free(kv_indices)
        self.assertNotIn(1, alloc.free_pages.tolist())
        self.assertIn(2, alloc.free_pages.tolist())

        manager.release_pages([1])
        self.assertIn(1, alloc.free_pages.tolist())


class TestAbortIsolation(CustomTestCase):
    def _manager(self):
        return KVPageProtectionManager(
            KVProtectionConfig(enable_attention_tags=True),
            allocator=None,
            num_pages=128,
            page_size=4,
            device="cpu",
            transfer_backend="mooncake",
        )

    def test_only_affected_request_aborts(self):
        mgr = self._manager()
        mgr.table.bump_generations(torch.tensor([10, 11, 20, 30, 31]))
        a = mgr.register_attention_tags(page_physical_ids=[10, 11], bootstrap_room=100)
        b = mgr.register_attention_tags(page_physical_ids=[20], bootstrap_room=200)
        c = mgr.register_attention_tags(page_physical_ids=[30, 31], bootstrap_room=300)

        mgr.table.bump_generations(torch.tensor([20]))
        g = int(mgr.table.generation_of(torch.tensor([20]))[0].item())
        mgr.table.write_tags(
            torch.tensor([20]),
            tags_to_tensor([compute_attention_tag_scalar(20, 0, 999, g)]),
        )

        mismatches = mgr.verify_batch([("A", a), ("B", b), ("C", c)])
        bad_rids = {m.rid for m in mismatches}
        self.assertEqual(bad_rids, {"B"})

        self.assertIsNone(mgr.verify_request(a, rid="A"))
        self.assertIsNone(mgr.verify_request(c, rid="C"))

    def test_mismatch_diagnostics(self):
        mgr = self._manager()
        mgr.table.bump_generations(torch.tensor([7]))
        man = mgr.register_attention_tags(page_physical_ids=[7], bootstrap_room=555)
        mgr.table.write_tags(torch.tensor([7]), tags_to_tensor([12345]))
        err = mgr.verify_request(man, rid="rid-x")
        self.assertIsInstance(err, KVAttentionTagMismatch)
        self.assertEqual(err.rid, "rid-x")
        self.assertEqual(err.bootstrap_room, 555)
        self.assertEqual(err.page_id, 7)
        self.assertEqual(err.page_position, 0)
        self.assertIsNotNone(err.expected_tag)
        self.assertIsNotNone(err.actual_tag)

    def test_transfer_page_tag_only_affected_request_aborts(self):
        mgr = self._manager()
        mgr.table.bump_generations(torch.tensor([10, 11, 20, 30, 31]))
        a = mgr.register_transfer_page_tags(
            page_physical_ids=[10, 11], bootstrap_room=100, write_actual=True
        )
        b = mgr.register_transfer_page_tags(
            page_physical_ids=[20], bootstrap_room=200, write_actual=True
        )
        c = mgr.register_transfer_page_tags(
            page_physical_ids=[30, 31], bootstrap_room=300, write_actual=True
        )

        g = int(mgr.table.generation_of(torch.tensor([20]))[0].item())
        stale_tag = compute_transfer_page_tag_scalar(20, 0, 999, g)
        mgr.table.write_transfer_page_tags(
            torch.tensor([20]),
            transfer_page_tags_to_tensor([stale_tag]),
        )

        mismatches = mgr.verify_transfer_page_tag_batch([("A", a), ("B", b), ("C", c)])
        bad_rids = {m.rid for m in mismatches}
        self.assertEqual(bad_rids, {"B"})
        self.assertIsInstance(mismatches[0], KVTransferPageTagMismatch)


if __name__ == "__main__":
    unittest.main()
