import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.marconi_config import MarconiConfig, MarconiTuningConfig
from sglang.srt.mem_cache.marconi_cost_model import (
    MarconiCostProfile,
    MarconiFFNCost,
    MarconiRecurrentCost,
    build_marconi_cost_profile,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.marconi_tuner import MarconiReplayInsertEvent
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=9, suite="stage-b-test-small-1-gpu-amd")


class _FakeMambaCache:
    def __init__(self, total_bytes: int):
        self._total_bytes = total_bytes

    def mem_usage_bytes(self) -> int:
        return self._total_bytes


class _FakeMambaPool:
    def __init__(self, *, total_bytes: int, size: int, num_layers: int):
        self.mamba_cache = _FakeMambaCache(total_bytes)
        self.size = size
        self.num_mamba_layers = num_layers


class _KimiLinearCacheParams:
    def __init__(self, *, shape, layers):
        self.shape = shape
        self.layers = layers


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _make_dummy_req(self, req_to_token_pool, rid: int):
        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )
        req_to_token_pool.alloc([req])
        return req

    def _make_marconi_tree(self, eff_weight: float = 4.0):
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path="dummy",
                page_size=1,
                enable_marconi=True,
                radix_eviction_policy="marconi",
            )
        )
        size = 64
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 64
        num_layers = 8
        global_interval = 4
        max_num_reqs = 32
        mamba_cache_size = 32
        max_context_len = 64
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=512,
                n_groups=4,
                num_heads=8,
                head_dim=64,
                state_size=16,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        marconi_config = MarconiConfig.enabled(
            eviction_enabled=True,
            eff_weight=eff_weight,
            cost_profile=MarconiCostProfile(
                recurrent_family="mamba2",
                recurrent=MarconiRecurrentCost(
                    family="mamba2",
                    num_layers=len(mamba_layers),
                    model_dim=64,
                    state_size=16,
                    state_size_bytes=16,
                ),
                ffn=MarconiFFNCost(
                    family="dense_mlp",
                    num_layers=num_layers,
                    model_dim=64,
                    intermediate_size=128,
                ),
                num_attn_layers=len(full_attention_layer_ids),
                model_dim=64,
                kv_cache_dtype_size=2,
            ),
            tuning=MarconiTuningConfig(enabled=False),
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            marconi_config=marconi_config,
        )
        tree = MambaRadixCache(params=params)
        return tree, req_to_token_pool, allocator

    def _insert_cached_path(
        self,
        tree: MambaRadixCache,
        req_to_token_pool: HybridReqToTokenPool,
        allocator: TokenToKVPoolAllocator,
        rid: int,
        token_ids: list[int],
        *,
        branch_token_ids: list[int] | None = None,
    ):
        req = self._make_dummy_req(req_to_token_pool, rid)
        kv_indices = allocator.alloc(len(token_ids))
        branch_mamba_value = None
        if branch_token_ids is not None:
            branch_req = self._make_dummy_req(req_to_token_pool, rid + 10_000)
            branch_mamba_value = branch_req.mamba_pool_idx.unsqueeze(0)
        result = tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                mamba_value=req.mamba_pool_idx.unsqueeze(0),
                branchoff_mamba_value=branch_mamba_value,
                branch_checkpoint_len=(
                    len(branch_token_ids) if branch_token_ids is not None else None
                ),
            )
        )
        return req, result

    def test_build_marconi_cost_profile_exact_mamba2_dense(self):
        hf_config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(state_size=16),
                layers=[0, 1, 2],
            ),
            num_hidden_layers=6,
            hidden_size=64,
            intermediate_size=128,
            full_attention_layer_ids=[1, 3, 5],
        )
        model_runner = SimpleNamespace(
            kv_cache_dtype=torch.bfloat16,
            hybrid_gdn_config=None,
        )
        req_to_token_pool = SimpleNamespace(
            mamba_pool=_FakeMambaPool(total_bytes=33 * 3 * 16, size=32, num_layers=3)
        )

        profile = build_marconi_cost_profile(
            hf_config=hf_config,
            model_runner=model_runner,
            req_to_token_pool=req_to_token_pool,
        )

        self.assertEqual(profile.recurrent_family, "mamba2")
        self.assertEqual(profile.ffn.family, "dense_mlp")

    def test_build_marconi_cost_profile_exact_gdn_moe(self):
        hf_config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(state_size=16),
                layers=[0, 1],
            ),
            num_hidden_layers=4,
            hidden_size=64,
            moe_intermediate_size=128,
            num_experts_per_tok=2,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layers_block_type=["linear", "attention", "linear", "attention"],
        )
        model_runner = SimpleNamespace(
            kv_cache_dtype=torch.bfloat16,
            hybrid_gdn_config=object(),
        )
        req_to_token_pool = SimpleNamespace(
            mamba_pool=_FakeMambaPool(total_bytes=17 * 2 * 16, size=16, num_layers=2)
        )

        profile = build_marconi_cost_profile(
            hf_config=hf_config,
            model_runner=model_runner,
            req_to_token_pool=req_to_token_pool,
        )

        self.assertEqual(profile.recurrent_family, "gdn")
        self.assertEqual(profile.ffn.family, "moe_topk")

    def test_build_marconi_cost_profile_rejects_unsupported_kda(self):
        hf_config = SimpleNamespace(
            mamba2_cache_params=_KimiLinearCacheParams(
                shape=SimpleNamespace(state_size=16),
                layers=[0, 1],
            ),
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            full_attention_layer_ids=[1, 3],
        )
        model_runner = SimpleNamespace(
            kv_cache_dtype=torch.bfloat16,
            hybrid_gdn_config=None,
        )
        req_to_token_pool = SimpleNamespace(
            mamba_pool=_FakeMambaPool(total_bytes=17 * 2 * 16, size=16, num_layers=2)
        )

        profile = build_marconi_cost_profile(
            hf_config=hf_config,
            model_runner=model_runner,
            req_to_token_pool=req_to_token_pool,
        )

        self.assertIsNone(profile)

    def test_marconi_records_exact_insert_events_for_tuner(self):
        tree, req_to_token_pool, _ = self._make_marconi_tree()
        req = self._make_dummy_req(req_to_token_pool, 50)
        req.origin_input_ids = [1, 2]
        req.output_ids = [3, 4]

        tree._marconi_record_insert_event(req, [1, 2, 3], 2)
        tree._marconi_record_insert_event(req, [1, 2, 3, 4], None)
        tree._marconi_record_finished_request(req)

        replay_request = tree.marconi_request_history_window[-1]
        self.assertEqual(
            replay_request.insert_events,
            (
                MarconiReplayInsertEvent(token_ids=(1, 2, 3), branch_checkpoint_len=2),
                MarconiReplayInsertEvent(
                    token_ids=(1, 2, 3, 4), branch_checkpoint_len=None
                ),
            ),
        )
        self.assertIsNone(tree.marconi_pending_insert_events.get(req.rid))

    def test_hybrid_linear_kv_pool(self):
        size = 16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=None,
        )
        assert pool._transfer_full_attention_id(global_interval - 1) == 0
        assert pool._transfer_full_attention_id(2 * global_interval - 1) == 1
        with self.assertRaises(ValueError) as context:
            pool._transfer_full_attention_id(1)
        self.assertIn(
            "layer_id=1 not in full attention layers:", str(context.exception)
        )

    def test_mamba_pool(self):
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        global_interval = 4
        num_layers = 48
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )

        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        req_to_token_pool.free_mamba_cache(req)
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        req.mamba_pool_idx = None
        req_to_token_pool.alloc([req])
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

    def test_mamba_radix_cache_1(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]

        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )

        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
        )
        tree = MambaRadixCache(params=params)

        def make_dummy_req():
            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sampling_params,
            )
            req_to_token_pool.alloc([req])
            return req

        mamba_pool = req_to_token_pool.mamba_pool
        print(
            f"[Start] allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req1 = make_dummy_req()
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        assert len(req1_token_ids) == len(req1_kv_indices)
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req1_token_ids),
                value=req1_kv_indices,
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req2 = make_dummy_req()
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        assert len(req2_token_ids) == len(req2_kv_indices)
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req2_token_ids),
                value=req2_kv_indices,
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        req3 = make_dummy_req()
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        assert len(req3_token_ids) == len(req3_kv_indices)
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req3_token_ids),
                value=req3_kv_indices,
                mamba_value=req3.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req4 = make_dummy_req()
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        assert len(req4_token_ids) == len(req4_kv_indices)
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req4_token_ids),
                value=req4_kv_indices,
                mamba_value=req4.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req4: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        tree.pretty_print()
        full_num_tokens = 1
        print(f"evicting {full_num_tokens} full token")
        result = tree.evict(EvictParams(num_tokens=full_num_tokens))
        assert (
            result.num_tokens_evicted >= full_num_tokens
        ), f"evicted {result.num_tokens_evicted} full tokens, expected {full_num_tokens}"
        tree.pretty_print()

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
        assert (
            result.mamba_num_evicted >= mamba_num
        ), f"evicted {result.mamba_num_evicted} mamba states, expected {mamba_num}"
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        req7_token_ids = [1, 2, 3, 4, 5, 6, 7]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req7_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req7: token_ids: {req7_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
        assert (
            result.mamba_num_evicted >= mamba_num
        ), f"evicted {result.mamba_num_evicted} mamba states, expected {mamba_num}"
        tree.pretty_print()

        req8_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req8_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req8: token_ids: {req8_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0
        assert len(last_node.key) == 0

        req9_token_ids = [1, 2, 3, 4, 5, 6, 7]
        req9 = make_dummy_req()
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(req9_token_ids), req=req9, cow_mamba=True)
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        assert req9.mamba_pool_idx is not None
        assert torch.all(
            mamba_pool.mamba_cache.conv[0][:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.conv[0][:, last_node.mamba_value]
        )
        assert torch.all(
            mamba_pool.mamba_cache.temporal[:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.temporal[:, last_node.mamba_value]
        )

        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))
        tree.sanity_check()

    def test_marconi_replay_tokens_use_nearest_live_ancestor(self):
        tree, req_to_token_pool, allocator = self._make_marconi_tree()
        self._insert_cached_path(
            tree, req_to_token_pool, allocator, 1, [1, 2, 5, 7]
        )
        self._insert_cached_path(
            tree,
            req_to_token_pool,
            allocator,
            2,
            [1, 2, 3, 4],
            branch_token_ids=[1, 2],
        )
        self._insert_cached_path(
            tree,
            req_to_token_pool,
            allocator,
            3,
            [1, 2, 5, 6],
            branch_token_ids=[1, 2, 5],
        )

        result = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 5, 7])))
        leaf_node = result.last_device_node
        internal_node = leaf_node.parent
        self.assertEqual(internal_node.prefix_len, 3)
        self.assertEqual(len(leaf_node.key), 1)

        tree.req_to_token_pool.mamba_pool.free(internal_node.mamba_value)
        tree.mamba_lru_list.remove_node(internal_node)
        tree._tombstone_internal_node(internal_node)

        self.assertEqual(tree._marconi_replay_tokens(leaf_node), 2)

    def test_marconi_mamba_eviction_can_tombstone_internal_node(self):
        tree, req_to_token_pool, allocator = self._make_marconi_tree(eff_weight=8.0)
        self._insert_cached_path(
            tree, req_to_token_pool, allocator, 10, [1, 2, 5, 6]
        )
        self._insert_cached_path(
            tree,
            req_to_token_pool,
            allocator,
            11,
            [1, 2, 3, 4],
            branch_token_ids=[1, 2],
        )

        branch_node = tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 5, 6]))
        ).last_device_node.parent
        branch_node.last_access_time = 100.0
        leaf_node = tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 5, 6]))
        ).last_device_node
        leaf_node.last_access_time = 100.0

        result = tree.evict(EvictParams(mamba_num=1))
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertIsNone(branch_node.mamba_value)
        self.assertIn(branch_node, tree.root_node.children.values())
        self.assertIsNotNone(leaf_node.mamba_value)


if __name__ == "__main__":
    unittest.main()
