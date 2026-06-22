"""Unit tests for KV page token tags (hashing, generations, vectorized verify).

These run on CPU; the tag/verify logic is dtype/device-agnostic so the same code
paths exercise the GPU sidecar buffer in production.  See the module docstring
in ``sglang/srt/mem_cache/kv_page_tags.py`` for the CUDA verification command.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_page_tags import (
    KVPageProtectionManager,
    KVPageTagMismatch,
    KVPageTagTable,
    KVProtectionConfig,
    PageManifest,
    compute_page_tag_scalar,
    compute_page_tags_tensor,
    tags_to_tensor,
    verify_page_tags,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_U64 = (1 << 64) - 1


class _FakeMetrics:
    def __init__(self):
        self.checked = 0
        self.mismatches = 0

    def increment_kv_page_tag_checked_pages(self, n):
        self.checked += n

    def increment_kv_page_tag_mismatches(self, n=1):
        self.mismatches += n


class TestPageTagHashing(CustomTestCase):
    def test_scalar_tensor_agreement(self):
        page_size = 4
        padded = [[10, 11, 12, 13], [20, 21, -1, -1]]
        logical = [[10, 11, 12, 13], [20, 21]]
        counts = [4, 2]
        pos = [0, 1]
        rooms = [7, 7]
        gens = [3, 5]
        t = compute_page_tags_tensor(
            torch.tensor(padded),
            torch.tensor(pos),
            torch.tensor(rooms),
            torch.tensor(gens),
            torch.tensor(counts),
        )
        for i in range(2):
            s = compute_page_tag_scalar(
                logical[i], pos[i], rooms[i], gens[i], page_size
            )
            self.assertEqual(s & _U64, int(t[i].item()) & _U64)

    def test_partial_page_differs_from_full(self):
        page_size = 4
        partial = compute_page_tag_scalar([20, 21], 1, 7, 5, page_size)
        full = compute_page_tag_scalar([20, 21, 22, 23], 1, 7, 5, page_size)
        self.assertNotEqual(partial, full)

    def test_fields_change_tag(self):
        ps = 4
        base = compute_page_tag_scalar([1, 2, 3, 4], 0, 7, 1, ps)
        # different generation
        self.assertNotEqual(base, compute_page_tag_scalar([1, 2, 3, 4], 0, 7, 2, ps))
        # different position
        self.assertNotEqual(base, compute_page_tag_scalar([1, 2, 3, 4], 1, 7, 1, ps))
        # different bootstrap room
        self.assertNotEqual(base, compute_page_tag_scalar([1, 2, 3, 4], 0, 8, 1, ps))
        # different token content
        self.assertNotEqual(base, compute_page_tag_scalar([1, 2, 3, 5], 0, 7, 1, ps))


class TestPageTagTable(CustomTestCase):
    def test_bump_generations(self):
        table = KVPageTagTable(num_pages=16, device="cpu")
        pages = torch.tensor([3, 4, 5])
        table.bump_generations(pages)
        table.bump_generations(torch.tensor([4]))
        self.assertEqual(table.generation_of(torch.tensor([3]))[0].item(), 1)
        self.assertEqual(table.generation_of(torch.tensor([4]))[0].item(), 2)

    def test_vectorized_verify_and_mismatch(self):
        table = KVPageTagTable(num_pages=32, device="cpu")
        pages = torch.tensor([3, 4, 5])
        table.bump_generations(pages)
        gens = table.generation_of(pages)
        tags = compute_page_tags_tensor(
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            torch.tensor([0, 1, 2]),
            torch.tensor([7, 7, 7]),
            gens,
            torch.tensor([4, 4, 4]),
        )
        table.write_tags(pages, tags)
        ok, mask = verify_page_tags(table, pages, tags)
        self.assertTrue(ok)
        self.assertFalse(bool(mask.any().item()))

        # Simulate another request stealing physical page 4: bump generation +
        # overwrite its tag. Page 4 must now mismatch, pages 3/5 must still pass.
        table.bump_generations(torch.tensor([4]))
        new_gen = int(table.generation_of(torch.tensor([4]))[0].item())
        table.write_tags(
            torch.tensor([4]),
            tags_to_tensor(
                [compute_page_tag_scalar([99, 99, 99, 99], 1, 999, new_gen, 4)]
            ),
        )
        ok2, mask2 = verify_page_tags(table, pages, tags)
        self.assertFalse(ok2)
        self.assertEqual(mask2.tolist(), [False, True, False])


class TestPageManifest(CustomTestCase):
    def test_manifest_pages_and_refresh(self):
        man = PageManifest.from_tokens(
            [10, 11, 12, 13, 20, 21],
            page_size=4,
            bootstrap_room=7,
            physical_page_ids=[3, 4],
            generations=[1, 1],
        )
        self.assertEqual(man.num_pages, 2)
        self.assertEqual(man.pages, [[10, 11, 12, 13], [20, 21]])
        old_tail = man.expected_tags[1]
        # Append a token to the partial tail page -> tag must change.
        man.refresh_page(1, [20, 21, 22], physical_page_id=4, generation=1)
        self.assertNotEqual(man.expected_tags[1], old_tail)
        # First page's tag must be untouched (refresh only touches the tail).
        self.assertEqual(
            man.expected_tags[0],
            compute_page_tag_scalar([10, 11, 12, 13], 0, 7, 1, 4),
        )
        self.assertTrue(torch.equal(man.physical_page_ids_t, torch.tensor([3, 4])))
        self.assertTrue(torch.equal(man.generations_t, torch.tensor([1, 1])))
        self.assertTrue(
            torch.equal(man.expected_tags_t, tags_to_tensor(man.expected_tags))
        )


class TestProtectionManager(CustomTestCase):
    def _make_manager(self, metrics=None):
        cfg = KVProtectionConfig(enable_page_tags=True)
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
        # pages physically backed at ids 2 and 3.
        mgr.table.bump_generations(torch.tensor([2, 3]))
        manifest = mgr.register_pages(
            token_ids=[1, 2, 3, 4, 5, 6],
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
        m_good = mgr.register_pages(
            token_ids=[1, 2, 3, 4, 5, 6], page_physical_ids=[2, 3], bootstrap_room=1
        )
        m_bad = mgr.register_pages(
            token_ids=[7, 8, 9, 10], page_physical_ids=[8], bootstrap_room=2
        )
        # Corrupt page 8 (the bad request's page) by reallocation + overwrite.
        mgr.table.bump_generations(torch.tensor([8]))
        g = int(mgr.table.generation_of(torch.tensor([8]))[0].item())
        mgr.table.write_tags(
            torch.tensor([8]),
            tags_to_tensor([compute_page_tag_scalar([0, 0, 0, 0], 0, 0, g, 4)]),
        )
        mismatches = mgr.verify_batch([("r-good", m_good), ("r-bad", m_bad)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r-bad")
        self.assertIsInstance(mismatches[0], KVPageTagMismatch)
        self.assertEqual(mismatches[0].bootstrap_room, 2)
        self.assertEqual(metrics.mismatches, 1)

    def test_verify_batch_uses_cached_manifest_tensors(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([2, 3]))
        manifest = mgr.register_pages(
            token_ids=[1, 2, 3, 4, 5, 6], page_physical_ids=[2, 3], bootstrap_room=9
        )

        # Hot-path verification must use the manifest's cached tensors, not
        # rebuild page/tag tensors from these Python diagnostic lists.
        manifest.physical_page_ids[:] = [99, 100]
        manifest.expected_tags[:] = [0, 0]
        self.assertEqual(mgr.verify_batch([("r", manifest)]), [])

    def test_refresh_tail_token_updates_only_changed_page(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 4]))
        manifest = mgr.register_pages(
            token_ids=[10, 11, 12, 13, 20, 21],
            page_physical_ids=[3, 4],
            bootstrap_room=7,
        )
        old_first_tag = manifest.expected_tags_t[0].clone()
        old_tail_tag = manifest.expected_tags_t[1].clone()

        mgr.refresh_tail_token(
            manifest,
            logical_pos=6,
            token_id=22,
            physical_page_id=torch.tensor([4]),
        )

        self.assertTrue(torch.equal(manifest.expected_tags_t[0], old_first_tag))
        self.assertFalse(torch.equal(manifest.expected_tags_t[1], old_tail_tag))
        self.assertEqual(manifest.pages[1], [20, 21, 22])
        self.assertIsNone(mgr.verify_request(manifest, rid="r"))

    def test_refresh_tail_token_appends_new_page(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3]))
        manifest = mgr.register_pages(
            token_ids=[10, 11, 12, 13], page_physical_ids=[3], bootstrap_room=7
        )
        old_first_tag = manifest.expected_tags_t[0].clone()

        mgr.table.bump_generations(torch.tensor([8]))
        mgr.refresh_tail_token(
            manifest,
            logical_pos=4,
            token_id=20,
            physical_page_id=torch.tensor([8]),
        )

        self.assertEqual(manifest.num_pages, 2)
        self.assertEqual(manifest.physical_page_ids_t.tolist(), [3, 8])
        self.assertTrue(torch.equal(manifest.expected_tags_t[0], old_first_tag))
        self.assertIsNone(mgr.verify_request(manifest, rid="r"))

    def test_refresh_tail_token_does_not_mask_generation_bump(self):
        mgr = self._make_manager()
        mgr.table.bump_generations(torch.tensor([3, 4]))
        manifest = mgr.register_pages(
            token_ids=[10, 11, 12, 13, 20, 21],
            page_physical_ids=[3, 4],
            bootstrap_room=7,
        )

        # Simulate the existing partial tail page being reallocated underneath
        # this request. Refreshing the tail must not accept the new generation.
        mgr.table.bump_generations(torch.tensor([4]))
        mgr.refresh_tail_token(
            manifest,
            logical_pos=6,
            token_id=22,
            physical_page_id=torch.tensor([4]),
        )

        mismatches = mgr.verify_batch([("r", manifest)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].rid, "r")

    def test_disabled_manager_is_noop(self):
        cfg = KVProtectionConfig()  # disabled
        mgr = KVPageProtectionManager(
            cfg, allocator=None, num_pages=0, page_size=4, transfer_backend="mooncake"
        )
        self.assertIsNone(mgr.table)
        self.assertIsNone(
            mgr.register_pages(
                token_ids=[1, 2], page_physical_ids=[1], bootstrap_room=1
            )
        )
        self.assertEqual(mgr.verify_batch([("r", None)]), [])


if __name__ == "__main__":
    unittest.main()
