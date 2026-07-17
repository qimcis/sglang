"""Unit tests for KV attention ownership tags.

Attention tags validate page ownership/generation before decode attention reads
KV. They intentionally do not hash token ids or KV bytes; transfer checksums own
byte-level correctness.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import _OffsetAttentionTagTable
from sglang.srt.mem_cache.kv_page_tags import (
    KV_PAGE_VALIDATION_INCOMPLETE,
    AttentionTagManifest,
    AttentionTagManifestGroup,
    KVAttentionTagMismatch,
    KVAttentionTagTable,
    KVPageHistory,
    KVPageProtectionManager,
    KVProtectionConfig,
    KVTransferPageTagMismatch,
    TransferPageTagManifestGroup,
    compute_attention_tag_scalar,
    compute_attention_tags_tensor,
    compute_transfer_page_tag_scalar,
    compute_transfer_page_tags_tensor,
    tags_to_tensor,
    verify_attention_tags,
    verify_transfer_page_tags,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_U64 = (1 << 64) - 1


class _FakeMetrics:
    def __init__(self):
        self.checked = 0
        self.mismatches = 0

    def increment_kv_attention_tag_checked_pages(self, n):
        self.checked += n

    def increment_kv_attention_tag_mismatches(self, n=1):
        self.mismatches += n


class TestAttentionTagHashing(CustomTestCase):
    def test_scalar_tensor_agreement(self):
        physical = [3, 9]
        pos = [0, 1]
        rooms = [7, 7]
        gens = [3, 5]
        t = compute_attention_tags_tensor(
            torch.tensor(physical),
            torch.tensor(pos),
            torch.tensor(rooms),
            torch.tensor(gens),
        )
        for i in range(2):
            s = compute_attention_tag_scalar(physical[i], pos[i], rooms[i], gens[i])
            self.assertEqual(s & _U64, int(t[i].item()) & _U64)

    def test_fields_change_tag(self):
        base = compute_attention_tag_scalar(3, 0, 7, 1)
        self.assertNotEqual(base, compute_attention_tag_scalar(4, 0, 7, 1))
        self.assertNotEqual(base, compute_attention_tag_scalar(3, 1, 7, 1))
        self.assertNotEqual(base, compute_attention_tag_scalar(3, 0, 8, 1))
        self.assertNotEqual(base, compute_attention_tag_scalar(3, 0, 7, 2))

    def test_transfer_page_tag_scalar_tensor_agreement(self):
        physical = [3, 9]
        pos = [0, 1]
        rooms = [7, 7]
        gens = [3, 5]
        t = compute_transfer_page_tags_tensor(
            torch.tensor(physical),
            torch.tensor(pos),
            torch.tensor(rooms),
            torch.tensor(gens),
        )
        for i in range(2):
            s = compute_transfer_page_tag_scalar(physical[i], pos[i], rooms[i], gens[i])
            self.assertEqual(s & 0xFFFF_FFFF, int(t[i].item()) & 0xFFFF_FFFF)


class TestAttentionTagTable(CustomTestCase):
    def test_bump_generations(self):
        table = KVAttentionTagTable(num_pages=16, device="cpu")
        pages = torch.tensor([3, 4, 5])
        table.bump_generations(pages)
        table.bump_generations(torch.tensor([4]))
        self.assertEqual(table.generation_of(torch.tensor([3]))[0].item(), 1)
        self.assertEqual(table.generation_of(torch.tensor([4]))[0].item(), 2)

    def test_token_allocator_bumps_generations(self):
        allocator = TokenToKVPoolAllocator(
            size=8,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        table = KVAttentionTagTable(num_pages=8, device="cpu")
        allocator.attach_attention_tag_table(table)

        first = allocator.alloc(2)
        self.assertEqual(first.tolist(), [1, 2])
        self.assertEqual(table.generation_of(first).tolist(), [1, 1])

        allocator.free(first[:1])
        second = allocator.alloc(7)
        self.assertEqual(second.tolist(), [3, 4, 5, 6, 7, 8, 1])
        self.assertEqual(table.generation_of(second[-1:]).tolist(), [2])

    def test_vectorized_verify_and_mismatch(self):
        table = KVAttentionTagTable(num_pages=32, device="cpu")
        pages = torch.tensor([3, 4, 5])
        table.bump_generations(pages)
        gens = table.generation_of(pages)
        tags = compute_attention_tags_tensor(
            pages,
            torch.tensor([0, 1, 2]),
            torch.tensor([7, 7, 7]),
            gens,
        )
        table.write_tags(pages, tags)
        ok, mask = verify_attention_tags(table, pages, tags, gens)
        self.assertTrue(ok)
        self.assertFalse(bool(mask.any().item()))

        table.bump_generations(torch.tensor([4]))
        new_gen = int(table.generation_of(torch.tensor([4]))[0].item())
        table.write_tags(
            torch.tensor([4]),
            tags_to_tensor([compute_attention_tag_scalar(4, 1, 999, new_gen)]),
        )
        ok2, mask2 = verify_attention_tags(table, pages, tags, gens)
        self.assertFalse(ok2)
        self.assertEqual(mask2.tolist(), [False, True, False])

    def test_transfer_page_tags_are_separate_actuals(self):
        table = KVAttentionTagTable(num_pages=32, device="cpu")
        pages = torch.tensor([3, 4, 5])
        table.bump_generations(pages)
        gens = table.generation_of(pages)
        tags = compute_transfer_page_tags_tensor(
            pages,
            torch.tensor([0, 1, 2]),
            torch.tensor([7, 7, 7]),
            gens,
        )

        ok_before, _ = verify_transfer_page_tags(table, pages, tags, gens)
        self.assertFalse(ok_before)

        table.write_transfer_page_tags(pages, tags)
        ok_after, mask_after = verify_transfer_page_tags(table, pages, tags, gens)
        self.assertTrue(ok_after)
        self.assertFalse(bool(mask_after.any().item()))

    def test_history_retains_last_eight_operations(self):
        history = KVPageHistory(size=4, device="cpu")
        for value in range(10):
            history.record([2], KVPageHistory.TAG_REFRESH, values=[value])

        entries = history.materialize(2)
        self.assertEqual([entry["sequence"] for entry in entries], list(range(2, 10)))
        self.assertEqual(entries[-1]["operation"], "tag_refresh")
        self.assertEqual(entries[-1]["value"], "0x0000000000000009")

    def test_swa_offset_table_forwards_free_history(self):
        table = KVAttentionTagTable(num_pages=16, device="cpu", enable_history=True)
        offset_table = _OffsetAttentionTagTable(table, offset=8)
        local_pages = torch.tensor([2])
        offset_table.bump_generations(local_pages)
        offset_table.record_free(local_pages)
        offset_table.record_free_released(local_pages)

        self.assertEqual(table.generation_of(torch.tensor([10])).tolist(), [1])
        operations = [entry["operation"] for entry in table.history.materialize(10)]
        self.assertEqual(operations, ["alloc", "free", "free_released"])

    def test_fused_forward_requires_current_epoch_completion(self):
        table = KVAttentionTagTable(num_pages=16, num_request_slots=4, device="cpu")
        request_indices = torch.tensor([1, 2])

        table.begin_fused_forward(request_indices)
        self.assertEqual(
            table.fused_failure_status(request_indices).tolist(),
            [KV_PAGE_VALIDATION_INCOMPLETE, KV_PAGE_VALIDATION_INCOMPLETE],
        )

        table.validated_epochs[1] = table.request_epochs[1]
        table.validated_epochs[2] = table.request_epochs[2]
        table.validation_status[2] = 8
        self.assertEqual(table.fused_failure_status(request_indices).tolist(), [0, 8])

    def test_begin_fused_forward_clears_stale_status(self):
        table = KVAttentionTagTable(num_pages=16, num_request_slots=3, device="cpu")
        table.validation_status[1] = 16
        table.begin_fused_forward(torch.tensor([1]))

        self.assertEqual(table.validation_status[1].item(), 0)
        self.assertEqual(table.request_epochs[1].item(), 1)

    def test_graph_padding_slot_zero_is_ignored(self):
        table = KVAttentionTagTable(num_pages=16, num_request_slots=3, device="cpu")
        table.begin_fused_forward(torch.tensor([0, 1]))

        self.assertEqual(table.request_epochs.tolist(), [0, 1, 0])
        self.assertEqual(
            table.fused_failure_status(torch.tensor([0, 1])).tolist(),
            [0, KV_PAGE_VALIDATION_INCOMPLETE],
        )

    def test_invalid_request_slot_fails_closed_without_indexing_sidecars(self):
        table = KVAttentionTagTable(num_pages=16, num_request_slots=3, device="cpu")
        request_indices = torch.tensor([-1, 3, 1])

        table.begin_fused_forward(request_indices)

        self.assertEqual(table.request_epochs.tolist(), [0, 1, 0])
        self.assertEqual(
            table.fused_failure_status(request_indices).tolist(),
            [1, 1, KV_PAGE_VALIDATION_INCOMPLETE],
        )

    def test_free_invalidates_stale_request_slot_owner(self):
        table = KVAttentionTagTable(num_pages=16, num_request_slots=3, device="cpu")
        page = torch.tensor([3])
        table.write_expected_owner(
            page,
            request_pool_idx=1,
            page_positions=torch.tensor([0]),
            generations=torch.tensor([1], dtype=torch.int64),
            attention_tags=torch.tensor([11], dtype=torch.int64),
            transfer_page_tags=torch.tensor([13], dtype=torch.int32),
        )

        table.record_free(page)

        self.assertEqual(table.owner_request_indices[3].item(), -1)


class TestAttentionTagManifest(CustomTestCase):
    def test_manifest_pages_and_refresh(self):
        man = AttentionTagManifest.from_pages(
            page_size=4,
            bootstrap_room=7,
            physical_page_ids=[3, 4],
            generations=[1, 1],
        )
        self.assertEqual(man.num_pages, 2)
        old_first_tag = man.expected_tags_t[0].clone()

        man.refresh_page_tensor(
            logical_pos=8,
            physical_page_id=torch.tensor([8]),
            generation=torch.tensor([1]),
        )
        self.assertEqual(man.num_pages, 3)
        self.assertEqual(man.physical_page_ids_t.tolist(), [3, 4, 8])
        self.assertTrue(torch.equal(man.expected_tags_t[0], old_first_tag))

    def test_sparse_page_positions_for_swa_tail(self):
        man = AttentionTagManifest.from_pages(
            page_size=4,
            bootstrap_room=7,
            physical_page_ids=[103, 104],
            generations=[1, 1],
            page_positions=[10, 11],
        )
        self.assertEqual(man.page_positions_t.tolist(), [10, 11])
        expected = compute_attention_tag_scalar(103, 10, 7, 1)
        self.assertEqual(int(man.expected_tags_t[0].item()) & _U64, expected & _U64)

        man.refresh_page_tensor(
            logical_pos=48,
            physical_page_id=torch.tensor([105]),
            generation=torch.tensor([1]),
        )
        self.assertEqual(man.page_positions_t.tolist(), [10, 11, 12])
        self.assertEqual(man.physical_page_ids_t.tolist(), [103, 104, 105])


class TestProtectionManager(CustomTestCase):
    def _make_manager(self, metrics=None):
        cfg = KVProtectionConfig(
            enable_attention_tags=True,
            enable_page_history=True,
        )
        return KVPageProtectionManager(
            cfg,
            allocator=None,
            num_pages=64,
            page_size=4,
            device="cpu",
            metrics_collector=metrics,
            transfer_backend="mooncake",
        )

    def test_register_then_verify_passes(self):
        metrics = _FakeMetrics()
        mgr = self._make_manager(metrics)
        mgr.table.bump_generations(torch.tensor([2, 3]))
        manifest = mgr.register_attention_tags(
            page_physical_ids=[2, 3],
            bootstrap_room=42,
        )
        self.assertIsNone(mgr.verify_request(manifest, rid="r1"))
        self.assertEqual(metrics.mismatches, 0)
        self.assertGreater(metrics.checked, 0)

    def test_verify_batch_isolates_bad_request(self):
        metrics = _FakeMetrics()
        mgr = self._make_manager(metrics)
        mgr.table.bump_generations(torch.tensor([2, 3, 8, 9]))
        m_good = mgr.register_attention_tags(page_physical_ids=[2, 3], bootstrap_room=1)
        m_bad = mgr.register_attention_tags(page_physical_ids=[8], bootstrap_room=2)

        mgr.table.bump_generations(torch.tensor([8]))
        g = int(mgr.table.generation_of(torch.tensor([8]))[0].item())
        mgr.table.write_tags(
            torch.tensor([8]),
            tags_to_tensor([compute_attention_tag_scalar(8, 0, 999, g)]),
        )
        mismatches = mgr.verify_batch([("r-good", m_good), ("r-bad", m_bad)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r-bad")
        self.assertIsInstance(mismatches[0], KVAttentionTagMismatch)
        self.assertEqual(mismatches[0].bootstrap_room, 2)
        self.assertIsNotNone(mismatches[0].incident)
        self.assertEqual(mismatches[0].incident.kind, "attention_tag")
        self.assertTrue(mismatches[0].incident.history)
        self.assertEqual(metrics.mismatches, 1)

    def test_combined_verification_reports_both_tag_families(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([2, 3]))
        attention = mgr.register_attention_tags(page_physical_ids=[2], bootstrap_room=1)
        transfer = mgr.register_transfer_page_tags(
            page_physical_ids=[3], bootstrap_room=2, write_actual=True
        )
        self.assertEqual(
            mgr.verify_protection_batch(
                [("attention", attention)], [("transfer", transfer)]
            ),
            [],
        )

        mgr.table.bump_generations(torch.tensor([2, 3]))
        mismatches = mgr.verify_protection_batch(
            [("attention", attention)], [("transfer", transfer)]
        )
        self.assertEqual(
            {type(mismatch) for mismatch in mismatches},
            {KVAttentionTagMismatch, KVTransferPageTagMismatch},
        )

    def test_flattened_verification_cache_reuses_and_invalidates(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3]))
        attention = mgr.register_attention_tags(page_physical_ids=[3], bootstrap_room=7)
        transfer = mgr.register_transfer_page_tags(
            page_physical_ids=[3], bootstrap_room=7, write_actual=True
        )
        items = [("r", attention)]
        transfer_items = [("r", transfer)]

        self.assertEqual(mgr.verify_protection_batch(items, transfer_items), [])
        attention_cache = mgr._attention_verification_cache
        transfer_cache = mgr._transfer_verification_cache
        with patch("torch.cat", side_effect=AssertionError("unexpected rebuild")):
            self.assertEqual(mgr.verify_protection_batch(items, transfer_items), [])
        self.assertIs(mgr._attention_verification_cache, attention_cache)
        self.assertIs(mgr._transfer_verification_cache, transfer_cache)

        mgr.table.bump_generations(torch.tensor([8]))
        mgr.refresh_tail_page(
            attention, logical_pos=4, physical_page_id=torch.tensor([8])
        )
        mgr.refresh_transfer_page_tag_tail_page(
            transfer, logical_pos=4, physical_page_id=torch.tensor([8])
        )
        self.assertEqual(mgr.verify_protection_batch(items, transfer_items), [])
        self.assertIsNot(mgr._attention_verification_cache, attention_cache)
        self.assertIsNot(mgr._transfer_verification_cache, transfer_cache)

        mgr.table.bump_generations(torch.tensor([8]))
        mismatches = mgr.verify_protection_batch(items, transfer_items)
        self.assertEqual({m.page_position for m in mismatches}, {1})

        mgr.clear_verification_cache()
        self.assertIsNone(mgr._attention_verification_cache)
        self.assertIsNone(mgr._transfer_verification_cache)

    def test_refresh_tail_page_appends_new_page(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3]))
        manifest = mgr.register_attention_tags(page_physical_ids=[3], bootstrap_room=7)
        old_first_tag = manifest.expected_tags_t[0].clone()

        mgr.table.bump_generations(torch.tensor([8]))
        mgr.refresh_tail_page(
            manifest,
            logical_pos=4,
            physical_page_id=torch.tensor([8]),
        )

        self.assertEqual(manifest.num_pages, 2)
        self.assertEqual(manifest.physical_page_ids_t.tolist(), [3, 8])
        self.assertTrue(torch.equal(manifest.expected_tags_t[0], old_first_tag))
        self.assertIsNone(mgr.verify_request(manifest, rid="r"))

    def test_group_verifies_full_and_swa_manifests(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 4, 53, 54]))
        full = mgr.register_attention_tags(
            page_physical_ids=[3, 4],
            page_positions=[0, 1],
            bootstrap_room=7,
        )
        swa = mgr.register_attention_tags(
            page_physical_ids=[53, 54],
            page_positions=[10, 11],
            bootstrap_room=7,
        )
        group = AttentionTagManifestGroup((full, swa))
        self.assertIsNone(mgr.verify_request(group, rid="r"))

        mgr.table.bump_generations(torch.tensor([54]))
        mismatches = mgr.verify_batch([("r", group)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r")
        self.assertEqual(mismatches[0].page_position, 11)

    def test_transfer_page_tag_detects_stale_transfer_write(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([4]))
        new_owner = mgr.register_transfer_page_tags(
            page_physical_ids=[4],
            page_positions=[0],
            bootstrap_room=100,
        )
        old_owner = mgr.register_transfer_page_tags(
            page_physical_ids=[4],
            page_positions=[0],
            bootstrap_room=200,
        )

        mgr.write_transfer_page_tags(
            page_physical_ids=old_owner.physical_page_ids_t,
            transfer_page_tags=old_owner.expected_tags_t,
        )
        mismatches = mgr.verify_transfer_page_tag_batch([("new", new_owner)])
        self.assertEqual(len(mismatches), 1)
        self.assertIsInstance(mismatches[0], KVTransferPageTagMismatch)
        self.assertEqual(mismatches[0].rid, "new")
        self.assertEqual(mismatches[0].bootstrap_room, 100)
        self.assertEqual(mismatches[0].page_id, 4)

    def test_transfer_page_tag_group_isolates_bad_request(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 4, 53, 54]))
        full = mgr.register_transfer_page_tags(
            page_physical_ids=[3, 4],
            page_positions=[0, 1],
            bootstrap_room=7,
            write_actual=True,
        )
        swa = mgr.register_transfer_page_tags(
            page_physical_ids=[53, 54],
            page_positions=[10, 11],
            bootstrap_room=7,
            write_actual=True,
        )
        group = TransferPageTagManifestGroup((full, swa))
        self.assertEqual(mgr.verify_transfer_page_tag_batch([("r", group)]), [])

        mgr.table.bump_generations(torch.tensor([54]))
        mismatches = mgr.verify_transfer_page_tag_batch([("r", group)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r")
        self.assertEqual(mismatches[0].page_position, 11)

    def test_commit_transfer_page_tags_writes_retained_group(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 53]))
        full = mgr.register_transfer_page_tags(
            page_physical_ids=[3], page_positions=[0], bootstrap_room=7
        )
        swa = mgr.register_transfer_page_tags(
            page_physical_ids=[53], page_positions=[10], bootstrap_room=7
        )
        group = TransferPageTagManifestGroup((full, swa))

        self.assertEqual(len(mgr.verify_transfer_page_tag_batch([("r", group)])), 1)
        mgr.commit_transfer_page_tags(group)
        self.assertEqual(mgr.verify_transfer_page_tag_batch([("r", group)]), [])

    def test_refresh_transfer_page_tag_tail_page_writes_actual(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3]))
        manifest = mgr.register_transfer_page_tags(
            page_physical_ids=[3], bootstrap_room=7, write_actual=True
        )

        mgr.table.bump_generations(torch.tensor([8]))
        mgr.refresh_transfer_page_tag_tail_page(
            manifest,
            logical_pos=4,
            physical_page_id=torch.tensor([8]),
        )

        self.assertEqual(manifest.num_pages, 2)
        self.assertEqual(manifest.physical_page_ids_t.tolist(), [3, 8])
        self.assertEqual(mgr.verify_transfer_page_tag_batch([("r", manifest)]), [])

    def test_refresh_tail_page_does_not_mask_generation_bump(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 4]))
        manifest = mgr.register_attention_tags(
            page_physical_ids=[3, 4],
            bootstrap_room=7,
        )

        mgr.table.bump_generations(torch.tensor([4]))
        mgr.refresh_tail_page(
            manifest,
            logical_pos=6,
            physical_page_id=torch.tensor([4]),
        )

        mismatches = mgr.verify_batch([("r", manifest)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r")

    def test_disabled_manager_is_noop(self):
        mgr = KVPageProtectionManager(
            KVProtectionConfig(),
            allocator=None,
            num_pages=0,
            page_size=4,
            transfer_backend="mooncake",
        )
        self.assertIsNone(mgr.table)
        self.assertIsNone(
            mgr.register_attention_tags(page_physical_ids=[1], bootstrap_room=1)
        )
        self.assertEqual(mgr.verify_batch([("r", None)]), [])


if __name__ == "__main__":
    unittest.main()
