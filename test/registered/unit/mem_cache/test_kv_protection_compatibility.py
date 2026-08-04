import unittest

from sglang.srt.mem_cache.kv_protection import (
    KVProtectionConfig,
    assert_protection_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVProtectionCompatibility(unittest.TestCase):
    def test_attention_tags_reject_speculative_decode(self):
        config = KVProtectionConfig(enable_attention_tags=True)

        with self.assertRaisesRegex(RuntimeError, "does not support speculative"):
            assert_protection_supported(config, is_spec_decode=True)

    def test_checksum_only_allows_speculative_decode(self):
        config = KVProtectionConfig(enable_transfer_checksum=True)

        assert_protection_supported(config, is_spec_decode=True)

    def test_disabled_protection_allows_speculative_decode(self):
        assert_protection_supported(KVProtectionConfig.disabled(), is_spec_decode=True)


if __name__ == "__main__":
    unittest.main()
