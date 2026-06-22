"""Tests for KV protection gating, env parsing, and unsupported-layout fail-fast.

Acceptance: the feature must be OFF for non-PD serving (no allocator overhead),
configurable via env vars, and must fail-fast (not silently disable) on
unsupported allocators/backends/speculative decoding.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_page_tags import (
    ChecksumMode,
    KVProtectionConfig,
    assert_protection_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePagedAllocator:
    pass


class _FakeSWAAllocator:
    pass


# Name the fakes to match the supported allowlist for the positive case.
_FakePagedAllocator.__name__ = "PagedTokenToKVPoolAllocator"
_FakeSWAAllocator.__name__ = "SWATokenToKVPoolAllocator"


class TestGating(CustomTestCase):
    def test_non_pd_is_always_disabled(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            True
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.override("always_full"):
            cfg = KVProtectionConfig.from_env(is_pd_decode=False)
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.enable_page_tags)
        self.assertEqual(cfg.checksum_mode, ChecksumMode.NONE)

    def test_pd_disabled_by_default(self):
        # Neither env var set -> disabled even in PD.
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            False
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.override("none"):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertFalse(cfg.enabled)

    def test_pd_page_tags_enabled(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            True
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.override("none"):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.enable_page_tags)
        self.assertFalse(cfg.checksum_enabled)

    def test_pd_checksum_mode_parsed(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            False
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.override(
            "sampled_partial"
        ), envs.SGLANG_KV_CHECKSUM_SAMPLE_RATE.override(
            0.2
        ):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertTrue(cfg.checksum_enabled)
        self.assertEqual(cfg.checksum_mode, ChecksumMode.SAMPLED_PARTIAL)
        self.assertAlmostEqual(cfg.checksum_sample_rate, 0.2)

    def test_invalid_checksum_mode_falls_back_to_none(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            False
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM_MODE.override("garbage"):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertFalse(cfg.checksum_enabled)


class TestFailFast(CustomTestCase):
    def test_disabled_config_never_raises(self):
        # No-op even for an unsupported allocator when the feature is off.
        assert_protection_supported(
            KVProtectionConfig(),
            allocator=_FakeSWAAllocator(),
            transfer_backend="nixl",
            is_spec_decode=True,
        )

    def test_supported_allocator_ok(self):
        assert_protection_supported(
            KVProtectionConfig(enable_page_tags=True),
            allocator=_FakePagedAllocator(),
            transfer_backend="mooncake",
        )

    def test_unsupported_allocator_fails_fast(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(enable_page_tags=True),
                allocator=_FakeSWAAllocator(),
            )

    def test_unsupported_backend_fails_fast_for_checksum(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(checksum_mode=ChecksumMode.ALWAYS_FULL),
                transfer_backend="nixl",
            )

    def test_spec_decode_fails_fast_for_page_tags(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(enable_page_tags=True),
                allocator=_FakePagedAllocator(),
                is_spec_decode=True,
            )


if __name__ == "__main__":
    unittest.main()
