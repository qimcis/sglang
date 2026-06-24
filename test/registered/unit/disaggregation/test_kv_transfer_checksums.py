"""Unit tests for KV transfer checksums.

Key property under test: checksums are computed in *logical* token order and
NEVER include node-local physical page/slot ids, so differing prefill/decode
physical layouts cannot cause false failures.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_page_tags import (
    ChecksumMode,
    ChecksumPlan,
    KVChecksumError,
    KVPageProtectionManager,
    KVProtectionConfig,
    checksum_code_to_mode,
    checksum_mode_to_code,
    compute_transfer_checksum,
    direct_kv_checksum_from_loc,
    gather_logical_kv_rows,
    parse_checksum_mode,
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
        self.cfg = KVProtectionConfig(checksum_mode=ChecksumMode.ALWAYS_FULL)

    def _cksum(self, rows, mode=ChecksumMode.ALWAYS_FULL):
        return compute_transfer_checksum(
            rows, bootstrap_room=42, num_tokens=self.N, mode=mode, config=self.cfg
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
        cfg = KVProtectionConfig(checksum_mode=ChecksumMode.ALWAYS_FULL)
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
        plan = mgr.compute_source_checksum_from_loc(
            _FakePool(src), loc_src, bootstrap_room=99, num_tokens=N
        )

        # Decode places the SAME logical content at DIFFERENT physical slots.
        loc_dst = torch.tensor([1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
        dst = {l: torch.zeros(size, h, d, dtype=torch.int32) for l in range(L)}
        for l in range(L):
            dst[l][loc_dst] = src[l][loc_src]
        err = mgr.verify_destination_checksum_from_loc(
            _FakePool(dst),
            loc_dst,
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
        plan = mgr.compute_source_checksum_from_loc(
            _FakePool(src), loc_src, bootstrap_room=7, num_tokens=N
        )
        dst = {l: src[l].clone() for l in range(L)}
        dst[0][3, 0, 0] += 1  # flip a destination byte
        err = mgr.verify_destination_checksum_from_loc(
            _FakePool(dst),
            loc_src,
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


class TestChecksumModesAndSampling(CustomTestCase):
    def test_parse_modes(self):
        self.assertEqual(parse_checksum_mode("none"), ChecksumMode.NONE)
        self.assertEqual(
            parse_checksum_mode("sampled_partial"), ChecksumMode.SAMPLED_PARTIAL
        )
        self.assertEqual(
            parse_checksum_mode("sample_partial"), ChecksumMode.SAMPLED_PARTIAL
        )
        self.assertEqual(parse_checksum_mode("sampled_full"), ChecksumMode.SAMPLED_FULL)
        self.assertEqual(parse_checksum_mode("always_full"), ChecksumMode.ALWAYS_FULL)

    def test_parse_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_checksum_mode("bogus")

    def test_sampling_is_deterministic(self):
        a = select_checksum_token_indices(100, 42, ChecksumMode.SAMPLED_PARTIAL, 0.1)
        b = select_checksum_token_indices(100, 42, ChecksumMode.SAMPLED_PARTIAL, 0.1)
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(a.numel(), 0)
        self.assertLessEqual(a.numel(), 100)

    def test_always_full_selects_all(self):
        idx = select_checksum_token_indices(50, 1, ChecksumMode.ALWAYS_FULL, 0.05)
        self.assertEqual(idx.numel(), 50)

    def test_mode_code_roundtrip(self):
        for mode in ChecksumMode:
            self.assertEqual(checksum_code_to_mode(checksum_mode_to_code(mode)), mode)

    def test_plan_payload_roundtrip(self):
        plan = ChecksumPlan(
            bootstrap_room=5,
            num_tokens=12,
            mode=ChecksumMode.SAMPLED_FULL,
            num_lanes=3,
            checksum=(1 << 63) + 123,  # exercises uint64 bit pattern
        )
        restored = ChecksumPlan.from_payload(plan.to_payload())
        self.assertEqual(restored.bootstrap_room, plan.bootstrap_room)
        self.assertEqual(restored.num_tokens, plan.num_tokens)
        self.assertEqual(restored.mode, plan.mode)
        self.assertEqual(restored.num_lanes, plan.num_lanes)
        # checksum compares equal as int64 bit patterns
        from sglang.srt.mem_cache.kv_page_tags import compare_checksums

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


class TestDirectKernelGating(CustomTestCase):
    """The direct-KV checksum kernel is CUDA-only; on CPU it must fall back
    cleanly (auto) or fail explicitly (strict), preserving all semantics."""

    def _manager(self, direct_kernel):
        cfg = KVProtectionConfig(
            checksum_mode=ChecksumMode.ALWAYS_FULL,
            checksum_direct_kernel=direct_kernel,
        )
        return KVPageProtectionManager(
            cfg,
            allocator=None,
            num_pages=0,
            page_size=4,
            device="cpu",
            transfer_backend="mooncake",
        )

    def test_auto_falls_back_on_cpu(self):
        # On a CPU pool the kernel is unavailable -> direct path returns None.
        k = [torch.randn(32, 2, 8) for _ in range(2)]
        v = [torch.randn(32, 2, 8) for _ in range(2)]
        cfg = KVProtectionConfig(
            checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="auto"
        )
        idx = torch.arange(8)
        got = direct_kv_checksum_from_loc(
            _FakePoolKV(k, v), torch.arange(32), idx, mode=cfg.checksum_mode, config=cfg
        )
        self.assertIsNone(got)

    def test_strict_raises_on_cpu(self):
        k = [torch.randn(32, 2, 8) for _ in range(2)]
        v = [torch.randn(32, 2, 8) for _ in range(2)]
        cfg = KVProtectionConfig(
            checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="strict"
        )
        with self.assertRaises(RuntimeError):
            direct_kv_checksum_from_loc(
                _FakePoolKV(k, v),
                torch.arange(32),
                torch.arange(8),
                mode=cfg.checksum_mode,
                config=cfg,
            )

    def test_off_returns_none(self):
        k = [torch.randn(32, 2, 8)]
        v = [torch.randn(32, 2, 8)]
        cfg = KVProtectionConfig(
            checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="off"
        )
        got = direct_kv_checksum_from_loc(
            _FakePoolKV(k, v),
            torch.arange(32),
            torch.arange(8),
            mode=cfg.checksum_mode,
            config=cfg,
        )
        self.assertIsNone(got)

    def test_manager_checksum_matches_across_gates_on_cpu(self):
        # auto (falls back) and off must produce the identical CPU checksum.
        torch.manual_seed(7)
        size, h, d, L, N = 64, 2, 4, 2, 10
        k = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        v = {
            l: torch.randint(0, 100, (size, h, d), dtype=torch.int32) for l in range(L)
        }
        loc = torch.arange(N)
        auto = self._manager("auto")
        off = self._manager("off")
        ca = auto.compute_source_checksum_from_loc(
            _FakePoolKV(k, v), loc, bootstrap_room=3, num_tokens=N
        )
        co = off.compute_source_checksum_from_loc(
            _FakePoolKV(k, v), loc, bootstrap_room=3, num_tokens=N
        )
        self.assertEqual(ca.checksum, co.checksum)


if __name__ == "__main__":
    unittest.main()
