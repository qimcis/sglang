"""Tests for KV protection gating, env parsing, and unsupported-layout fail-fast.

Acceptance: the feature must be OFF for non-PD serving (no allocator overhead),
configurable via env vars, and must fail-fast (not silently disable) on
unsupported allocators/backends/speculative decoding.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_page_tags import (
    KVProtectionConfig,
    assert_protection_supported,
    should_use_fused_kv_page_protection,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePagedAllocator:
    pass


class _FakeSWAAllocator:
    pass


class _FakeUnsupportedAllocator:
    pass


# Name the fakes to match the supported allowlist for the positive case.
_FakePagedAllocator.__name__ = "PagedTokenToKVPoolAllocator"
_FakeSWAAllocator.__name__ = "SWATokenToKVPoolAllocator"


class TestGating(CustomTestCase):
    def test_non_pd_is_always_disabled(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            True
        ), envs.SGLANG_KV_PAGE_HISTORY.override(
            True
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM.override(
            True
        ), envs.SGLANG_ENABLE_LEGACY_KV_PROTECTION_COMPLETION.override(
            True
        ):
            cfg = KVProtectionConfig.from_env(is_pd_decode=False)
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.enable_attention_tags)
        self.assertFalse(cfg.checksum_enabled)
        self.assertFalse(cfg.enable_page_history)
        self.assertFalse(cfg.allow_legacy_completion)

    def test_pd_disabled_by_default(self):
        # Neither env var set -> disabled even in PD.
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            False
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM.override(False):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertFalse(cfg.enabled)

    def test_pd_attention_tags_enabled(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            True
        ), envs.SGLANG_KV_PAGE_HISTORY.override(
            True
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM.override(
            False
        ):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.enable_attention_tags)
        self.assertTrue(cfg.enable_page_history)
        self.assertFalse(cfg.checksum_enabled)

    def test_pd_checksum_enabled(self):
        with envs.SGLANG_KV_PAGE_PROTECTION.override(
            False
        ), envs.SGLANG_KV_TRANSFER_CHECKSUM.override(
            True
        ), envs.SGLANG_ENABLE_LEGACY_KV_PROTECTION_COMPLETION.override(
            True
        ):
            cfg = KVProtectionConfig.from_env(is_pd_decode=True)
        self.assertTrue(cfg.checksum_enabled)
        self.assertTrue(cfg.allow_legacy_completion)

    def test_fused_validation_kill_switch(self):
        table = object()
        with envs.SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION.override(False):
            self.assertTrue(should_use_fused_kv_page_protection(table, supported=True))
            self.assertFalse(should_use_fused_kv_page_protection(None, supported=True))
            self.assertFalse(
                should_use_fused_kv_page_protection(table, supported=False)
            )

        with envs.SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION.override(True):
            self.assertFalse(should_use_fused_kv_page_protection(table, supported=True))


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
            KVProtectionConfig(enable_attention_tags=True),
            allocator=_FakePagedAllocator(),
            transfer_backend="mooncake",
        )
        assert_protection_supported(
            KVProtectionConfig(
                enable_attention_tags=True, enable_transfer_checksum=True
            ),
            allocator=_FakeSWAAllocator(),
            transfer_backend="mooncake",
        )

    def test_unsupported_allocator_fails_fast(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakeUnsupportedAllocator(),
            )

    def test_unsupported_backend_fails_fast_for_checksum(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(enable_transfer_checksum=True),
                transfer_backend="nixl",
            )

    def test_spec_decode_fails_fast_for_attention_tags(self):
        with self.assertRaises(RuntimeError):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakePagedAllocator(),
                is_spec_decode=True,
            )

    def test_pipeline_parallel_fails_fast_for_attention_tags(self):
        with self.assertRaisesRegex(RuntimeError, "pipeline parallelism"):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakePagedAllocator(),
                pp_size=2,
            )

    def test_dp_attention_fails_fast_for_attention_tags(self):
        with self.assertRaisesRegex(RuntimeError, "DP attention"):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakePagedAllocator(),
                enable_dp_attention=True,
            )

    def test_radix_cache_fails_fast_for_attention_tags(self):
        with self.assertRaisesRegex(RuntimeError, "shared radix-prefix"):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakePagedAllocator(),
                radix_cache_enabled=True,
            )

    def test_non_hopper_device_fails_fast_for_attention_tags(self):
        with envs.SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION.override(False):
            with self.assertRaisesRegex(RuntimeError, "Hopper SM90"):
                assert_protection_supported(
                    KVProtectionConfig(enable_attention_tags=True),
                    allocator=_FakePagedAllocator(),
                    device_capability_major=8,
                )
            with self.assertRaisesRegex(RuntimeError, "NVIDIA Hopper SM90"):
                assert_protection_supported(
                    KVProtectionConfig(enable_attention_tags=True),
                    allocator=_FakePagedAllocator(),
                    is_cuda_device=False,
                    device_capability_major=9,
                )

        with envs.SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION.override(True):
            assert_protection_supported(
                KVProtectionConfig(enable_attention_tags=True),
                allocator=_FakePagedAllocator(),
                is_cuda_device=False,
                device_capability_major=8,
            )

        assert_protection_supported(
            KVProtectionConfig(enable_attention_tags=True),
            allocator=_FakePagedAllocator(),
            device_capability_major=9,
        )


if __name__ == "__main__":
    unittest.main()
