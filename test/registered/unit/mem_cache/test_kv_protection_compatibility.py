import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    PrefillServerInfo,
    kv_row_layout_fingerprint,
)
from sglang.srt.disaggregation.mooncake.conn import (
    synchronize_fan_in_transfer_nonce,
)
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

    def test_attention_tags_allow_dp_attention(self):
        config = KVProtectionConfig(enable_attention_tags=True)

        assert_protection_supported(config, enable_dp_attention=True)

    def test_attention_tags_reject_dp_lm_head(self):
        config = KVProtectionConfig(enable_attention_tags=True)

        with self.assertRaisesRegex(RuntimeError, "enable-dp-lm-head"):
            assert_protection_supported(
                config,
                enable_dp_attention=True,
                enable_dp_lm_head=True,
            )

    def test_checksum_only_allows_dp_lm_head(self):
        config = KVProtectionConfig(enable_transfer_checksum=True)

        assert_protection_supported(
            config,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
        )


class TestProtectedDisaggregationRankMapping(unittest.TestCase):
    @staticmethod
    def _manager(*, is_mla_backend: bool, engine_rank: int = 1, attn_tp_size: int = 4):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.attn_tp_size = attn_tp_size
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = is_mla_backend
        manager.enable_all_cp_ranks_for_transfer = False
        manager.enable_staging = False
        manager.kv_args = SimpleNamespace(
            engine_rank=engine_rank,
            kv_item_lens=[36864] * 78,
            state_types=[StateType.DSA],
            state_item_lens=[[4096] * 78],
            state_dim_per_tensor=[[]],
            transfer_page_tag_manager=SimpleNamespace(
                config=SimpleNamespace(checksum_enabled=True)
            ),
        )
        return manager

    @staticmethod
    def _prefill_info(manager, **overrides):
        values = dict(
            attn_tp_size=8,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=64,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            kv_protection_enabled=True,
            kv_row_layout_fingerprint=kv_row_layout_fingerprint(manager.kv_args),
        )
        values.update(overrides)
        return PrefillServerInfo(**values)

    def test_protected_mla_maps_prefill_tp8_to_dpa_attention_tp4(self):
        manager = self._manager(is_mla_backend=True)
        info = self._prefill_info(manager)

        manager._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 2)
        self.assertEqual(info.target_tp_ranks, [2, 3])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_protected_mla_maps_dpa_attention_tp1_to_decode_tp8(self):
        manager = self._manager(is_mla_backend=True, engine_rank=7, attn_tp_size=8)
        info = self._prefill_info(manager, attn_tp_size=1, dp_size=4)

        manager._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 0)
        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.required_dst_info_num, 8)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_protected_mla_rejects_different_row_layout(self):
        manager = self._manager(is_mla_backend=True)
        info = self._prefill_info(manager, kv_row_layout_fingerprint="0" * 64)

        with self.assertRaisesRegex(RuntimeError, "identical transferred KV row"):
            manager._resolve_rank_mapping(info)

    def test_protected_non_mla_rejects_heterogeneous_attention_tp(self):
        manager = self._manager(is_mla_backend=False)
        info = self._prefill_info(manager)

        with self.assertRaisesRegex(RuntimeError, "integral MLA TP mapping"):
            manager._resolve_rank_mapping(info)


class TestProtectedMooncakeFanInNonce(unittest.TestCase):
    class _Group:
        world_size = 8

        def __init__(self, rank_in_group, root_nonce):
            self.rank_in_group = rank_in_group
            self.root_nonce = root_nonce
            self.broadcasts = []

        def broadcast_object(self, value, src):
            self.broadcasts.append((value, src))
            return self.root_nonce

    def test_root_publishes_nonce_for_fan_in(self):
        group = self._Group(rank_in_group=0, root_nonce=101)

        nonce = synchronize_fan_in_transfer_nonce(101, 8, group)

        self.assertEqual(nonce, 101)
        self.assertEqual(group.broadcasts, [(101, 0)])

    def test_follower_receives_root_nonce_for_fan_in(self):
        group = self._Group(rank_in_group=7, root_nonce=101)

        nonce = synchronize_fan_in_transfer_nonce(202, 8, group)

        self.assertEqual(nonce, 101)
        self.assertEqual(group.broadcasts, [(None, 0)])

    def test_one_to_one_mapping_keeps_local_nonce_without_collective(self):
        group = self._Group(rank_in_group=7, root_nonce=101)

        nonce = synchronize_fan_in_transfer_nonce(202, 1, group)

        self.assertEqual(nonce, 202)
        self.assertEqual(group.broadcasts, [])

    def test_rejects_fan_in_larger_than_attention_tp_group(self):
        group = self._Group(rank_in_group=0, root_nonce=101)

        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            synchronize_fan_in_transfer_nonce(101, 9, group)


if __name__ == "__main__":
    unittest.main()
