from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
import math
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.marconi_admission_cache import MarconiAdmissionTree
from sglang.srt.mem_cache.marconi_config import (
    get_marconi_branch_align_interval,
)
from sglang.srt.mem_cache.marconi_cost_model import MarconiCostProfile
from sglang.srt.mem_cache.marconi_replay_core import (
    MarconiReplayCandidateMetrics,
    build_marconi_candidate_metrics,
    select_marconi_candidate_index,
)
from sglang.srt.mem_cache.marconi_tuner import (
    MarconiReplayInsertEvent,
    MarconiReplayNodeSnapshot,
    MarconiReplayRequest,
    MarconiReplaySnapshot,
    MarconiTuneResult,
    tune_marconi_eff_weight,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

logger = logging.getLogger(__name__)

MARCONI_TUNER_MAX_WORKERS = max(
    1, min(get_int_env_var("SGLANG_MARCONI_TUNER_MAX_WORKERS", 1), os.cpu_count() or 1)
)
MARCONI_TUNER_MP_CONTEXT = mp.get_context("spawn")


class TreeNode:
    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.prefix_len = 0
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: "TreeNode") -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


@dataclass
class MarconiEvictionCandidate:
    node: TreeNode
    metrics: MarconiReplayCandidateMetrics

    @property
    def action(self) -> str:
        return self.metrics.action

    @property
    def replay_tokens(self) -> int:
        return self.metrics.replay_tokens

    @property
    def recurrent_flops(self) -> float:
        return self.metrics.recurrent_flops

    @property
    def attention_flops(self) -> float:
        return self.metrics.attention_flops

    @property
    def ffn_flops(self) -> float:
        return self.metrics.ffn_flops

    @property
    def memory_bytes(self) -> int:
        return self.metrics.memory_bytes

    @property
    def total_flops(self) -> float:
        return self.metrics.total_flops


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional["MambaRadixCache"] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {x_lru.last_access_time:f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by last_access_time): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        heapq.heapify(nodes)
        while len(nodes):
            x = heapq.heappop(nodes)
            msg += f"[{x.id}] {x.last_access_time:f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "MambaRadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_tensor_model_parallel_rank() == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.marconi_config = params.marconi_config
        self.marconi_cost_profile: Optional[MarconiCostProfile] = None
        self.marconi_enabled = bool(
            self.marconi_config is not None and self.marconi_config.enable
        )
        self.marconi_eviction_enabled = bool(
            self.marconi_enabled and self.marconi_config.eviction_enabled
        )
        if self.marconi_eviction_enabled:
            self.marconi_cost_profile = self.marconi_config.cost_profile
            if self.marconi_cost_profile is None:
                raise ValueError(
                    "Marconi eviction is enabled but cost profile is missing; "
                    "cannot compute eviction utilities."
                )
        if self.marconi_eviction_enabled:
            self.marconi_use_efficiency = True
            self.marconi_eff_weight = self.marconi_config.eff_weight
            self.marconi_tuning_config = self.marconi_config.tuning
            self.marconi_tuner_pool = (
                ProcessPoolExecutor(
                    max_workers=MARCONI_TUNER_MAX_WORKERS,
                    mp_context=MARCONI_TUNER_MP_CONTEXT,
                )
                if self.marconi_tuning_config.enabled
                else None
            )
            logger.info(
                "Marconi eviction enabled for recurrent family '%s'.",
                self.marconi_cost_profile.recurrent_family,
            )
        else:
            self.marconi_use_efficiency = False
            self.marconi_tuning_config = None
            self.marconi_tuner_pool = None
        self.marconi_admission_tree = (
            self._make_marconi_admission_tree() if self.marconi_enabled else None
        )

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        server_args = get_global_server_args()
        self.marconi_branch_align_interval = None
        if self.marconi_enabled and server_args is not None:
            self.marconi_branch_align_interval = get_marconi_branch_align_interval(
                self.page_size, align_interval=server_args.mamba_cache_chunk_size
            )

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def _node_has_full_mamba_state(self, node: TreeNode) -> bool:
        return node.mamba_value is not None

    def _make_marconi_admission_tree(self) -> MarconiAdmissionTree:
        max_tokens = getattr(self.token_to_kv_pool_allocator, "size", None)
        return MarconiAdmissionTree(max_tokens=max_tokens)

    def _cancel_marconi_tuning_future(self) -> None:
        future = getattr(self, "marconi_tuning_future", None)
        if future is None:
            return
        future.cancel()
        self.marconi_tuning_future = None

    def _shutdown_marconi_tuner_pool(self) -> None:
        tuner_pool = getattr(self, "marconi_tuner_pool", None)
        if tuner_pool is None:
            return
        tuner_pool.shutdown(wait=False, cancel_futures=True)
        self.marconi_tuner_pool = None

    def _fork_mamba_value_for_cache(
        self,
        mamba_value: torch.Tensor,
        *,
        context: str,
        required: bool = False,
    ) -> Optional[torch.Tensor]:
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)
        if mamba_value_forked is None:
            self.evict_mamba(1)
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
        if mamba_value_forked is not None:
            return mamba_value_forked
        if required:
            raise AssertionError("Can not alloc mamba cache")
        logger.debug("Marconi skipped %s due to mamba alloc pressure", context)
        return None

    def _marconi_serialize_node(self, node: TreeNode) -> MarconiReplayNodeSnapshot:
        children = tuple(
            self._marconi_serialize_node(child) for child in node.children.values()
        )
        return MarconiReplayNodeSnapshot(
            key=tuple(node.key.token_ids),
            extra_key=node.key.extra_key,
            prefix_len=node.prefix_len,
            has_mamba=node.mamba_value is not None,
            last_access_time=float(node.last_access_time),
            children=children,
        )

    def _marconi_snapshot_tree(self) -> Optional[MarconiReplaySnapshot]:
        if not self.marconi_eviction_enabled or self.marconi_cost_profile is None:
            return None
        full_capacity = getattr(self.token_to_kv_pool_allocator, "size", None)
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        mamba_capacity = getattr(mamba_pool, "size", None)
        if full_capacity is None or mamba_capacity is None:
            return None
        return MarconiReplaySnapshot(
            root=self._marconi_serialize_node(self.root_node),
            cost_profile=self.marconi_cost_profile,
            full_capacity_tokens=int(full_capacity),
            mamba_capacity_states=int(mamba_capacity),
            current_time=float(TreeNode.last_access_time_counter_float),
        )

    def _marconi_get_autotune_target(self) -> Optional[int]:
        if not self.marconi_eviction_enabled or self.marconi_tuning_config is None:
            return None
        if self.marconi_bootstrap_window_size is None:
            return None
        if self.marconi_live_autotune_rounds == 0:
            return self.marconi_bootstrap_window_size
        return self.marconi_tuning_config.tuning_interval

    def _marconi_get_tuning_weight_grid(self) -> tuple[float, ...]:
        assert self.marconi_tuning_config is not None
        weight_grid = self.marconi_tuning_config.weight_grid
        if self.marconi_live_autotune_rounds > 0 or len(weight_grid) <= 5:
            return weight_grid

        # Keep the first round cheap by probing a small number of evenly spaced
        # weights over the configured range. Later rounds use the full grid.
        last_index = len(weight_grid) - 1
        coarse_indices = {round(last_index * fraction / 4) for fraction in range(5)}
        return tuple(weight_grid[idx] for idx in sorted(coarse_indices))

    def _marconi_note_autotune_skip(self, reason: str) -> None:
        self.marconi_live_autotune_skips += 1
        self.marconi_last_autotune_status = "skipped"
        self.marconi_last_autotune_skip_reason = reason
        logger.info(
            "Marconi autotune skipped: reason=%s completed_requests=%d "
            "window_requests=%d target=%s",
            reason,
            self.marconi_completed_request_count,
            self.marconi_window_request_count,
            self._marconi_get_autotune_target(),
        )

    def update_scheduler_request_counts(
        self, running_req_count: int, queue_req_count: int
    ) -> None:
        self.marconi_last_running_req_count = int(running_req_count)
        self.marconi_last_queue_req_count = int(queue_req_count)

    def _marconi_poll_tuning_future(self, source: str) -> None:
        if self.marconi_tuning_future is None or not self.marconi_tuning_future.done():
            return
        future = self.marconi_tuning_future
        self.marconi_tuning_future = None
        self.marconi_last_autotune_poll_source = source
        try:
            tuning_result = future.result()
        except Exception:
            self.marconi_live_autotune_failures += 1
            self.marconi_last_autotune_status = "failed"
            self.marconi_last_autotune_apply_source = None
            self.marconi_last_autotune_finished_request_count = (
                self.marconi_completed_request_count
            )
            logger.exception(
                "Marconi autotuning failed: source=%s completed_requests=%d",
                source,
                self.marconi_completed_request_count,
            )
            return
        if tuning_result is None:
            self._marconi_note_autotune_skip("empty_result")
            return
        assert isinstance(tuning_result, MarconiTuneResult)
        tuned_weight = tuning_result.best_weight
        self.marconi_live_autotune_finishes += 1
        self.marconi_last_autotune_status = "finished"
        self.marconi_last_autotune_finished_request_count = (
            self.marconi_completed_request_count
        )
        score_summary = ", ".join(
            f"{weight:.1f}:{score:.3e}" for weight, score in tuning_result.weight_scores
        )
        old_weight = self.marconi_eff_weight
        active_weight_score = next(
            (
                score
                for weight, score in tuning_result.weight_scores
                if math.isclose(weight, old_weight, rel_tol=0.0, abs_tol=1e-9)
            ),
            None,
        )
        stabilized_to_current = (
            tuned_weight != old_weight
            and active_weight_score is not None
            and math.isclose(
                active_weight_score,
                tuning_result.best_flops_saved,
                rel_tol=1e-9,
                abs_tol=0.0,
            )
        )
        if stabilized_to_current:
            tuned_weight = old_weight
        logger.info(
            "Marconi autotune finished: source=%s tuned_weight=%s "
            "completed_requests=%d window_requests=%d input_tokens=%d "
            "output_tokens=%d insert_events=%d current_weight=%.4f "
            "current_score=%s stabilized_to_current=%s scores=[%s]",
            source,
            tuned_weight,
            self.marconi_completed_request_count,
            tuning_result.request_count,
            tuning_result.input_token_count,
            tuning_result.output_token_count,
            tuning_result.insert_event_count,
            old_weight,
            (f"{active_weight_score:.3e}" if active_weight_score is not None else None),
            stabilized_to_current,
            score_summary,
        )
        self.marconi_eff_weight = tuned_weight
        self.marconi_last_tuned_eff_weight = tuned_weight
        self.marconi_live_autotune_rounds += 1
        self.marconi_live_autotune_applies += 1
        self.marconi_last_autotune_status = "applied"
        self.marconi_last_autotune_apply_source = source
        self.marconi_last_autotune_applied_request_count = (
            self.marconi_completed_request_count
        )
        self.marconi_log_next_eviction_after_apply = True
        discarded_window_requests = 0
        if tuned_weight != old_weight:
            discarded_window_requests = self.marconi_window_request_count
            # Start the next tuning round from the post-apply cache state so
            # later windows reflect the active weight instead of mixed history.
            self.marconi_tuning_snapshot = self._marconi_snapshot_tree()
            self.marconi_request_history_window = []
            self.marconi_window_request_count = 0
        target = self._marconi_get_autotune_target()
        next_round_ready = (
            target is not None
            and self.marconi_window_request_count >= max(1, target)
            and self.marconi_tuning_future is None
        )
        logger.info(
            "Marconi autotune applied: old_weight=%.4f new_weight=%.4f "
            "round=%d completed_requests=%d source=%s running_req=%s "
            "queue_req=%s window_requests=%d target=%s next_round_ready=%s "
            "discarded_window_requests=%d "
            "evict_full=%d evict_mamba=%d evict_mamba_leaf=%d "
            "evict_mamba_internal=%d evict_mamba_no_candidate=%d",
            old_weight,
            tuned_weight,
            self.marconi_live_autotune_rounds,
            self.marconi_completed_request_count,
            source,
            self.marconi_last_running_req_count,
            self.marconi_last_queue_req_count,
            self.marconi_window_request_count,
            target,
            next_round_ready,
            discarded_window_requests,
            self.marconi_live_evict_full,
            self.marconi_live_evict_mamba,
            self.marconi_live_evict_mamba_leaf_count,
            self.marconi_live_evict_mamba_internal_count,
            self.marconi_live_evict_mamba_no_candidate_count,
        )

    def _marconi_record_insert_event(
        self,
        req: Req,
        token_ids,
        branch_checkpoint_len: Optional[int],
    ) -> None:
        if not self.marconi_eviction_enabled:
            return
        token_tuple = tuple(int(token_id) for token_id in token_ids)
        if not token_tuple:
            return
        if branch_checkpoint_len is not None:
            branch_checkpoint_len = int(branch_checkpoint_len)
            if branch_checkpoint_len <= 0 or branch_checkpoint_len > len(token_tuple):
                branch_checkpoint_len = None
        self.marconi_pending_insert_events.setdefault(req.rid, []).append(
            (token_tuple, branch_checkpoint_len)
        )

    def _marconi_record_finished_request(self, req: Req) -> None:
        if not self.marconi_eviction_enabled:
            return
        self._marconi_poll_tuning_future("request_finished")
        pending_events = self.marconi_pending_insert_events.pop(req.rid, ())
        insert_events = tuple(
            MarconiReplayInsertEvent(
                token_ids=token_ids,
                branch_checkpoint_len=branch_checkpoint_len,
            )
            for token_ids, branch_checkpoint_len in pending_events
        )
        self.marconi_request_history_window.append(
            MarconiReplayRequest(
                input_ids=tuple(req.origin_input_ids),
                output_ids=tuple(req.output_ids),
                extra_key=req.extra_key,
                insert_events=insert_events,
            )
        )
        self.marconi_completed_request_count += 1
        self.marconi_window_request_count += 1
        self._marconi_maybe_start_tuning()

    def _marconi_maybe_start_tuning(self) -> None:
        if not self.marconi_eviction_enabled or self.marconi_tuning_config is None:
            return
        if not self.marconi_tuning_config.enabled or self.marconi_tuner_pool is None:
            return
        self._marconi_poll_tuning_future("maybe_start")
        if self.marconi_tuning_future is not None:
            return
        if self.marconi_bootstrap_window_size is None:
            return
        target = self._marconi_get_autotune_target()
        assert target is not None
        if self.marconi_window_request_count < max(1, target):
            return
        if not self.marconi_request_history_window:
            self._marconi_note_autotune_skip("empty_request_window")
            return
        if self.marconi_tuning_snapshot is None:
            self._marconi_note_autotune_skip("missing_snapshot")
            return
        snapshot = self.marconi_tuning_snapshot
        request_history = list(self.marconi_request_history_window)
        weight_grid = self._marconi_get_tuning_weight_grid()
        self.marconi_tuning_future = self.marconi_tuner_pool.submit(
            tune_marconi_eff_weight,
            snapshot=snapshot,
            request_history_window=request_history,
            weight_grid=weight_grid,
        )
        self.marconi_live_autotune_starts += 1
        self.marconi_last_autotune_status = "running"
        self.marconi_last_autotune_skip_reason = None
        self.marconi_last_autotune_started_request_count = (
            self.marconi_completed_request_count
        )
        self.marconi_last_autotune_window_size = len(request_history)
        self.marconi_last_autotune_weight_grid_size = len(weight_grid)
        logger.info(
            "Marconi autotune started: round=%d completed_requests=%d "
            "window_requests=%d target=%d grid_size=%d current_eff_weight=%.4f",
            self.marconi_live_autotune_rounds + 1,
            self.marconi_completed_request_count,
            len(request_history),
            target,
            len(weight_grid),
            self.marconi_eff_weight,
        )
        self.marconi_tuning_snapshot = self._marconi_snapshot_tree()
        self.marconi_request_history_window = []
        self.marconi_window_request_count = 0

    def _marconi_note_eviction(self) -> None:
        if (
            not self.marconi_eviction_enabled
            or self.marconi_tuning_config is None
            or self.marconi_first_eviction_request_count is not None
        ):
            return
        self.marconi_first_eviction_request_count = self.marconi_completed_request_count
        bootstrap_window_uncapped_size = max(
            1,
            self.marconi_first_eviction_request_count
            * self.marconi_tuning_config.bootstrap_multiplier,
        )
        self.marconi_bootstrap_window_uncapped_size = bootstrap_window_uncapped_size
        self.marconi_bootstrap_window_size = min(
            bootstrap_window_uncapped_size,
            max(
                self.marconi_first_eviction_request_count,
                self.marconi_tuning_config.tuning_interval,
            ),
        )
        # Start the first replay window from the first real eviction. Pre-eviction
        # traffic does not exercise eviction and only makes the bootstrap replay
        # later and more expensive.
        self.marconi_request_history_window = []
        self.marconi_window_request_count = 0
        self.marconi_tuning_snapshot = self._marconi_snapshot_tree()
        logger.info(
            "Marconi autotune bootstrap armed: first_eviction_request=%d "
            "bootstrap_window=%d raw_bootstrap_window=%d tuning_interval=%d",
            self.marconi_first_eviction_request_count,
            self.marconi_bootstrap_window_size,
            self.marconi_bootstrap_window_uncapped_size,
            self.marconi_tuning_config.tuning_interval,
        )

    def _get_marconi_live_stats(self) -> dict:
        self._marconi_poll_tuning_future("runtime_stats")
        hit_rate = (
            self.marconi_live_hit_count / self.marconi_live_match_count
            if self.marconi_live_match_count > 0
            else 0.0
        )
        token_hit_rate = (
            self.marconi_live_token_hit / self.marconi_live_token_total
            if self.marconi_live_token_total > 0
            else 0.0
        )
        admission_branch_rate = (
            self.marconi_admission_branch_count / self.marconi_admission_match_count
            if getattr(self, "marconi_admission_match_count", 0) > 0
            else 0.0
        )
        admission_tree = self.marconi_admission_tree
        admission_nodes = admission_tree.num_nodes if admission_tree is not None else 0
        admission_tokens = (
            admission_tree.num_tokens if admission_tree is not None else 0
        )
        full_tokens, mamba_states = self.total_size()
        avg_replay_tokens = (
            self.marconi_live_replay_tokens_total / self.marconi_live_replay_samples
            if self.marconi_live_replay_samples > 0
            else 0.0
        )
        return {
            "enabled": self.marconi_enabled,
            "eviction_enabled": self.marconi_eviction_enabled,
            "profile_family": (
                self.marconi_cost_profile.recurrent_family
                if self.marconi_cost_profile is not None
                else None
            ),
            "matches": self.marconi_live_match_count,
            "hits": self.marconi_live_hit_count,
            "hit_rate": hit_rate,
            "token_total": self.marconi_live_token_total,
            "token_hit": self.marconi_live_token_hit,
            "token_hit_rate": token_hit_rate,
            "branch_checkpoint_inserts": self.marconi_live_branch_checkpoint_count,
            "track_entry_inserts": self.marconi_live_track_entry_insert_count,
            "admission_matches": getattr(self, "marconi_admission_match_count", 0),
            "admission_branches": getattr(self, "marconi_admission_branch_count", 0),
            "admission_branch_rate": admission_branch_rate,
            "admission_tree_nodes": admission_nodes,
            "admission_tree_tokens": admission_tokens,
            "evict_full_tokens": self.marconi_live_evict_full,
            "evict_mamba_states": self.marconi_live_evict_mamba,
            "evict_mamba_leaf_count": self.marconi_live_evict_mamba_leaf_count,
            "evict_mamba_internal_count": self.marconi_live_evict_mamba_internal_count,
            "evict_mamba_attempts": self.marconi_live_evict_mamba_attempts,
            "evict_mamba_no_candidate_count": self.marconi_live_evict_mamba_no_candidate_count,
            "recurrent_replay_tokens_total": self.marconi_live_replay_tokens_total,
            "recurrent_replay_tokens_avg": avg_replay_tokens,
            "current_eff_weight": (
                self.marconi_eff_weight if self.marconi_eviction_enabled else None
            ),
            "autotune_enabled": (
                self.marconi_tuning_config.enabled
                if self.marconi_tuning_config is not None
                else False
            ),
            "autotune_status": self.marconi_last_autotune_status,
            "autotune_inflight": self.marconi_tuning_future is not None,
            "autotune_started": self.marconi_live_autotune_starts,
            "autotune_finished": self.marconi_live_autotune_finishes,
            "autotune_applied": self.marconi_live_autotune_applies,
            "autotune_skipped": self.marconi_live_autotune_skips,
            "autotune_rounds": self.marconi_live_autotune_rounds,
            "autotune_failures": self.marconi_live_autotune_failures,
            "autotune_last_skip_reason": self.marconi_last_autotune_skip_reason,
            "autotune_last_poll_source": self.marconi_last_autotune_poll_source,
            "autotune_last_apply_source": self.marconi_last_autotune_apply_source,
            "autotune_last_started_request_count": self.marconi_last_autotune_started_request_count,
            "autotune_last_finished_request_count": self.marconi_last_autotune_finished_request_count,
            "autotune_last_applied_request_count": self.marconi_last_autotune_applied_request_count,
            "autotune_last_window_size": self.marconi_last_autotune_window_size,
            "autotune_last_weight_grid_size": self.marconi_last_autotune_weight_grid_size,
            "autotune_window_request_count": self.marconi_window_request_count,
            "autotune_target_window_size": self._marconi_get_autotune_target(),
            "requests_before_first_eviction": self.marconi_first_eviction_request_count,
            "bootstrap_window_size": self.marconi_bootstrap_window_size,
            "bootstrap_window_uncapped_size": self.marconi_bootstrap_window_uncapped_size,
            "tuning_interval": (
                self.marconi_tuning_config.tuning_interval
                if self.marconi_tuning_config is not None
                else None
            ),
            "last_tuned_eff_weight": self.marconi_last_tuned_eff_weight,
            "scheduler_running_req_count": self.marconi_last_running_req_count,
            "scheduler_queue_req_count": self.marconi_last_queue_req_count,
            "last_recurrent_flops_saved": self.marconi_last_recurrent_flops_saved,
            "last_attention_flops_saved": self.marconi_last_attention_flops_saved,
            "last_ffn_flops_saved": self.marconi_last_ffn_flops_saved,
            "cached_full_tokens": full_tokens,
            "cached_mamba_states": mamba_states,
        }

    def _log_marconi_live_stats(self, *, force: bool = False) -> None:
        if not self.marconi_enabled:
            return
        if not force and not logger.isEnabledFor(logging.DEBUG):
            return
        if not force and (
            self.marconi_live_match_count == 0
            or self.marconi_live_match_count % self.marconi_live_log_interval != 0
        ):
            return
        stats = self._get_marconi_live_stats()
        logger.debug(
            "Marconi live stats: matches=%d hit_rate=%.4f token_hit_rate=%.4f "
            "admission_branch_rate=%.4f branch_checkpoint_inserts=%d "
            "track_entry_inserts=%d evict_full=%d evict_mamba=%d "
            "cached_full=%d cached_mamba=%d current_eff_weight=%.4f "
            "autotune_status=%s autotune_rounds=%d autotune_inflight=%s",
            stats["matches"],
            stats["hit_rate"],
            stats["token_hit_rate"],
            stats["admission_branch_rate"],
            stats["branch_checkpoint_inserts"],
            stats["track_entry_inserts"],
            stats["evict_full_tokens"],
            stats["evict_mamba_states"],
            stats["cached_full_tokens"],
            stats["cached_mamba_states"],
            stats["current_eff_weight"] or 0.0,
            stats["autotune_status"],
            stats["autotune_rounds"],
            stats["autotune_inflight"],
        )

    def get_runtime_stats(self) -> Optional[dict[str, object]]:
        if not self.marconi_enabled:
            return None
        return {"marconi": self._get_marconi_live_stats()}

    def reset(self) -> None:
        self._cancel_marconi_tuning_future()
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([], None)
        self.root_node.value = []
        self.root_node.prefix_len = 0
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)
        self.marconi_cached_kv_mask = None
        if self.marconi_enabled:
            kv_capacity = getattr(self.token_to_kv_pool_allocator, "size", None)
            if kv_capacity is not None and kv_capacity > 0:
                self.marconi_cached_kv_mask = torch.zeros(
                    kv_capacity + 1, dtype=torch.bool, device="cpu"
                )
        self.marconi_live_match_count = 0
        self.marconi_live_hit_count = 0
        self.marconi_live_token_total = 0
        self.marconi_live_token_hit = 0
        self.marconi_live_evict_full = 0
        self.marconi_live_evict_mamba = 0
        self.marconi_live_evict_mamba_leaf_count = 0
        self.marconi_live_evict_mamba_internal_count = 0
        self.marconi_live_evict_mamba_attempts = 0
        self.marconi_live_evict_mamba_no_candidate_count = 0
        self.marconi_live_branch_checkpoint_count = 0
        self.marconi_live_track_entry_insert_count = 0
        self.marconi_live_replay_tokens_total = 0
        self.marconi_live_replay_samples = 0
        self.marconi_live_autotune_starts = 0
        self.marconi_live_autotune_finishes = 0
        self.marconi_live_autotune_applies = 0
        self.marconi_live_autotune_skips = 0
        self.marconi_live_autotune_rounds = 0
        self.marconi_live_autotune_failures = 0
        self.marconi_last_autotune_status = "idle"
        self.marconi_last_autotune_skip_reason = None
        self.marconi_last_autotune_poll_source = None
        self.marconi_last_autotune_apply_source = None
        self.marconi_last_autotune_started_request_count = None
        self.marconi_last_autotune_finished_request_count = None
        self.marconi_last_autotune_applied_request_count = None
        self.marconi_last_autotune_window_size = None
        self.marconi_last_autotune_weight_grid_size = None
        self.marconi_last_tuned_eff_weight = None
        self.marconi_last_recurrent_flops_saved = 0.0
        self.marconi_last_attention_flops_saved = 0.0
        self.marconi_last_ffn_flops_saved = 0.0
        self.marconi_request_history_window = []
        self.marconi_pending_insert_events = {}
        self.marconi_completed_request_count = 0
        self.marconi_window_request_count = 0
        self.marconi_first_eviction_request_count = None
        self.marconi_bootstrap_window_size = None
        self.marconi_bootstrap_window_uncapped_size = None
        self.marconi_last_running_req_count = None
        self.marconi_last_queue_req_count = None
        self.marconi_log_next_eviction_after_apply = False
        self.marconi_tuning_future: Optional[Future] = None
        self.marconi_tuning_snapshot = self._marconi_snapshot_tree()
        self.marconi_live_log_interval = 50
        if self.marconi_enabled:
            self.marconi_admission_match_count = 0
            self.marconi_admission_branch_count = 0
            self.marconi_admission_tree = self._make_marconi_admission_tree()

    def __del__(self):
        self._cancel_marconi_tuning_future()
        self._shutdown_marconi_tuner_pool()

    def _marconi_filter_free_indices(
        self, indices: torch.Tensor, where: str
    ) -> Optional[torch.Tensor]:
        if not self.marconi_enabled:
            return indices
        if indices is None or indices.numel() == 0:
            return None
        if self.marconi_cached_kv_mask is None:
            return indices
        idx_cpu = indices.detach().to(device="cpu", dtype=torch.int64, copy=False)
        if (
            idx_cpu.numel() > 0
            and idx_cpu.max().item() >= self.marconi_cached_kv_mask.numel()
        ):
            return indices
        mask = self.marconi_cached_kv_mask[idx_cpu]
        if mask.numel() == 0:
            return None
        keep = ~mask
        if not keep.any():
            return None
        free_idx = idx_cpu[keep]
        return free_idx.to(dtype=indices.dtype, device=indices.device)

    def _marconi_kv_mask_add(self, indices: torch.Tensor) -> None:
        if (
            self.marconi_cached_kv_mask is None
            or indices is None
            or indices.numel() == 0
        ):
            return
        idx_cpu = indices.detach().to(device="cpu", dtype=torch.int64, copy=False)
        self.marconi_cached_kv_mask[idx_cpu] = True

    def _marconi_kv_mask_remove(self, indices: torch.Tensor) -> None:
        if (
            self.marconi_cached_kv_mask is None
            or indices is None
            or indices.numel() == 0
        ):
            return
        idx_cpu = indices.detach().to(device="cpu", dtype=torch.int64, copy=False)
        self.marconi_cached_kv_mask[idx_cpu] = False

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, best_last_node, last_node, mamba_branching_seqlen, best_value_len = (
            self._match_prefix_helper(key)
        )
        return self._match_post_processor(
            params,
            value,
            best_last_node,
            last_node,
            mamba_branching_seqlen,
            best_value_len,
        )

    def _marconi_align_cache_len(self, cache_len: int) -> int:
        if cache_len <= 0:
            return 0
        if self.page_size > 1:
            return cache_len // self.page_size * self.page_size
        return cache_len

    def _marconi_align_branch_len(self, cache_len: int) -> Optional[int]:
        if cache_len <= 0:
            return None
        align_interval = self.marconi_branch_align_interval
        if align_interval is None:
            return cache_len
        if align_interval <= 1:
            return cache_len
        aligned = cache_len // align_interval * align_interval
        return aligned if aligned > 0 else None

    def _marconi_branch_seqlen(
        self,
        cache_len: int,
        cached_prefix_len: int,
        existing_branch: Optional[int],
    ) -> Optional[int]:
        branch_len = None
        if cache_len > 0 and cache_len >= cached_prefix_len:
            branch_len = self._marconi_align_branch_len(cache_len)
        if existing_branch is not None:
            if branch_len is None or existing_branch < branch_len:
                branch_len = existing_branch
        return branch_len

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node,
            key,
            value,
            mamba_value,
            branchoff_mamba_value=params.branchoff_mamba_value,
            branch_checkpoint_len=params.branch_checkpoint_len,
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        kv_cache_protected_len = getattr(
            req, "kv_cache_protected_len", req.cache_protected_len
        )

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        if (
            self.marconi_enabled
            and self.marconi_admission_tree is not None
            and kv_committed_len > 0
            and not getattr(req, "marconi_admission_seeded", False)
        ):
            input_len = min(len(req.origin_input_ids), kv_committed_len)
            if input_len > 0:
                self.marconi_admission_tree.insert(
                    req.origin_input_ids[:input_len], req.extra_key
                )
                req.marconi_admission_seeded = True

        track_entries = None
        primary_entry = None
        secondary_entries = None
        if self.marconi_enabled:
            track_entries = getattr(req, "mamba_track_entries", None)
            if track_entries:
                entries = sorted(track_entries, key=lambda x: x[0])
                primary_entry = entries[-1]
                secondary_entries = entries[:-1]
        primary_mamba_forked = None

        if is_insert:
            if primary_entry is not None:
                cache_len = primary_entry[0]
            else:
                cache_len = (
                    req.mamba_last_track_seqlen
                    if self.enable_mamba_extra_buffer
                    else len(token_ids)
                )
            if cache_len is None:
                cache_len = 0
            if cache_len > len(token_ids):
                cache_len = len(token_ids)
            if cache_len <= 0:
                is_insert = False

        if is_insert:
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, kv_cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                if primary_entry is not None:
                    primary_len, primary_idx = primary_entry
                    mamba_value_src = torch.tensor(
                        [primary_idx],
                        dtype=torch.int64,
                        device=page_aligned_kv_indices.device,
                    )
                    mamba_ping_pong_track_buffer_to_keep = None
                    if req.mamba_ping_pong_track_buffer is not None:
                        buffer_list = req.mamba_ping_pong_track_buffer.tolist()
                        if primary_idx in buffer_list:
                            mamba_ping_pong_track_buffer_to_keep = buffer_list.index(
                                primary_idx
                            )
                else:
                    mamba_ping_pong_track_buffer_to_keep = (
                        self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            req.mamba_next_track_idx
                        )
                    )
                    mamba_value_src = (
                        req.mamba_ping_pong_track_buffer[
                            mamba_ping_pong_track_buffer_to_keep
                        ]
                        .unsqueeze(-1)
                        .clone()
                    )
            else:
                mamba_value_src = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            mamba_value = mamba_value_src
            if self.marconi_enabled:
                primary_mamba_forked = self._fork_mamba_value_for_cache(
                    mamba_value_src,
                    context="cache_finished_req:primary",
                )
                if primary_mamba_forked is None:
                    is_insert = False
                else:
                    mamba_value = primary_mamba_forked
                    mamba_ping_pong_track_buffer_to_keep = None

            if is_insert:
                result = self.insert(
                    InsertParams(
                        key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                        value=page_aligned_kv_indices,
                        mamba_value=mamba_value,
                    )
                )
                new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
                self._marconi_record_insert_event(
                    req, token_ids[:page_aligned_len], None
                )
                if mamba_exist and primary_mamba_forked is not None:
                    self.req_to_token_pool.mamba_pool.free(primary_mamba_forked)

                inserted_start = getattr(req, "kv_cache_inserted_start", None)
                inserted_end = getattr(req, "kv_cache_inserted_end", None)
                dup_ranges = []
                if (
                    inserted_start is not None
                    and inserted_end is not None
                    and inserted_start < inserted_end
                ):
                    pre_end = min(new_prefix_len, inserted_start)
                    if kv_cache_protected_len < pre_end:
                        dup_ranges.append(
                            (
                                kv_cache_protected_len,
                                pre_end,
                                "cache_finished_req:dup_prefix_pre",
                            )
                        )
                    post_start = max(kv_cache_protected_len, inserted_end)
                    if post_start < new_prefix_len:
                        dup_ranges.append(
                            (
                                post_start,
                                new_prefix_len,
                                "cache_finished_req:dup_prefix_post",
                            )
                        )
                elif kv_cache_protected_len < new_prefix_len:
                    dup_ranges.append(
                        (
                            kv_cache_protected_len,
                            new_prefix_len,
                            "cache_finished_req:dup_prefix",
                        )
                    )
                for start, end, where in dup_ranges:
                    dup_slice = kv_indices[start:end]
                    if self.marconi_enabled:
                        safe = self._marconi_filter_free_indices(dup_slice, where)
                        if safe is not None:
                            self.token_to_kv_pool_allocator.free(safe)
                    else:
                        self.token_to_kv_pool_allocator.free(dup_slice)

                if secondary_entries:
                    for cache_len, mamba_idx in secondary_entries:
                        cache_len = min(cache_len, len(token_ids))
                        if cache_len <= 0:
                            continue
                        kv_indices_sub = kv_indices[:cache_len]
                        if self.page_size != 1:
                            page_aligned_len = (
                                len(kv_indices_sub) // self.page_size * self.page_size
                            )
                            if page_aligned_len <= 0:
                                continue
                            kv_indices_sub = kv_indices_sub[:page_aligned_len].to(
                                dtype=torch.int64, copy=True
                            )
                            token_ids_sub = token_ids[:page_aligned_len]
                            cache_len = page_aligned_len
                        else:
                            kv_indices_sub = kv_indices_sub.to(
                                dtype=torch.int64, copy=True
                            )
                            token_ids_sub = token_ids[:cache_len]

                        mamba_value = torch.tensor(
                            [mamba_idx], dtype=torch.int64, device=kv_indices_sub.device
                        )
                        secondary_forked = self._fork_mamba_value_for_cache(
                            mamba_value,
                            context="cache_finished_req:secondary",
                        )
                        if secondary_forked is None:
                            continue
                        result = self.insert(
                            InsertParams(
                                key=RadixKey(token_ids_sub, req.extra_key),
                                value=kv_indices_sub,
                                mamba_value=secondary_forked,
                                branch_checkpoint_len=cache_len,
                            )
                        )
                        mamba_exist = result.mamba_exist
                        self._marconi_record_insert_event(req, token_ids_sub, cache_len)
                        if mamba_exist:
                            self.req_to_token_pool.mamba_pool.free(secondary_forked)

                if self.marconi_enabled:
                    req.kv_cache_protected_len = max(
                        kv_cache_protected_len, new_prefix_len
                    )
            else:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[kv_cache_protected_len:]
                )
                mamba_exist = True
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[kv_cache_protected_len:])
            mamba_exist = True

        if track_entries:
            req.mamba_track_entries = None

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_keep = None

        free_mamba_cache = (
            True
            if self.enable_mamba_extra_buffer
            else (mamba_exist or primary_mamba_forked is not None)
        )

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)
        self._marconi_record_finished_request(req)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        kv_cache_protected_len = getattr(
            req, "kv_cache_protected_len", req.cache_protected_len
        )

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None or cache_len <= 0:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        track_entries = getattr(req, "mamba_track_entries", None)
        if self.marconi_enabled and track_entries:
            inserted = self._marconi_cache_track_entries(
                req, track_entries, kv_cache_protected_len, kv_indices_orig, token_ids
            )
            req.mamba_track_entries = None
            if not inserted:
                _skip_cache_unfinished_req(req)
            return
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1)
                .clone()
            )
        else:
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        branch_checkpoint_len = None
        if req.mamba_branching_seqlen is not None:
            if page_aligned_len == req.mamba_branching_seqlen:
                branch_checkpoint_len = req.mamba_branching_seqlen
            else:
                branch_align_interval = (
                    self.marconi_branch_align_interval or self.page_size
                )
                if (
                    page_aligned_len < req.mamba_branching_seqlen
                    and req.mamba_branching_seqlen
                    <= page_aligned_len + branch_align_interval
                ):
                    branch_checkpoint_len = page_aligned_len
        branch_checkpoint = branch_checkpoint_len is not None
        needs_branch_checkpoint = req.mamba_branching_seqlen is not None
        if self.marconi_enabled and needs_branch_checkpoint and not branch_checkpoint:
            return _skip_cache_unfinished_req(req)
        if self.marconi_enabled and branch_checkpoint_len is not None:
            self.marconi_live_branch_checkpoint_count += 1
            if branch_checkpoint_len < page_aligned_len:
                page_aligned_token_ids = page_aligned_token_ids[:branch_checkpoint_len]
                page_aligned_kv_indices = page_aligned_kv_indices[
                    :branch_checkpoint_len
                ]
                page_aligned_len = branch_checkpoint_len
        mamba_value_forked = self._fork_mamba_value_for_cache(
            mamba_value,
            context="cache_unfinished_req",
        )
        if mamba_value_forked is None:
            return _skip_cache_unfinished_req(req)
        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
                branchoff_mamba_value=(
                    mamba_value_forked if branch_checkpoint else None
                ),
                branch_checkpoint_len=branch_checkpoint_len,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        self._marconi_record_insert_event(
            req, page_aligned_token_ids, branch_checkpoint_len
        )
        if self.marconi_enabled:
            if getattr(req, "kv_cache_inserted_start", None) is None:
                req.kv_cache_inserted_start = new_prefix_len
                req.kv_cache_inserted_end = page_aligned_len
            else:
                if req.kv_cache_inserted_end is None:
                    req.kv_cache_inserted_end = page_aligned_len
                else:
                    req.kv_cache_inserted_end = max(
                        req.kv_cache_inserted_end, page_aligned_len
                    )
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

        if not self.marconi_enabled:
            assert (
                req.cache_protected_len <= len(new_indices) + self.page_size - 1
            ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
            assert new_prefix_len <= len(
                new_indices
            ), f"{new_prefix_len=}, {len(new_indices)=}"
        elif new_prefix_len > len(new_indices):
            logger.warning(
                "Marconi KV prefix exceeds mamba prefix: rid=%s "
                "new_prefix_len=%d mamba_prefix_len=%d",
                req.rid,
                new_prefix_len,
                len(new_indices),
            )

        new_cache_protected_len = len(new_indices)
        free_end = min(new_prefix_len, new_cache_protected_len)
        dup_slice = None
        if kv_cache_protected_len < free_end:
            dup_slice = kv_indices_orig[kv_cache_protected_len:free_end].clone()

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(kv_cache_protected_len, len(new_indices))),
            new_indices[kv_cache_protected_len:],
        )

        if dup_slice is not None:
            if self.marconi_enabled:
                safe = self._marconi_filter_free_indices(
                    dup_slice, "cache_unfinished_req:dup_prefix"
                )
                if safe is not None:
                    self.token_to_kv_pool_allocator.free(safe)
            else:
                self.token_to_kv_pool_allocator.free(dup_slice)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        if self.marconi_enabled:
            req.kv_cache_protected_len = max(kv_cache_protected_len, len(new_indices))
        else:
            req.kv_cache_protected_len = req.cache_protected_len
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def _marconi_cache_track_entries(
        self,
        req: Req,
        track_entries: List[Tuple[int, int]],
        kv_cache_protected_len: int,
        kv_indices_orig: torch.Tensor,
        token_ids: List[int],
    ) -> bool:
        entries = sorted(track_entries, key=lambda x: x[0])
        self.marconi_live_track_entry_insert_count += len(entries)
        primary_len, primary_idx = entries[-1]
        cache_len = min(primary_len, len(token_ids))
        if cache_len <= 0:
            return False

        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        if page_aligned_len <= 0:
            return False

        page_aligned_token_ids = token_ids[:page_aligned_len]

        mamba_value = torch.tensor(
            [primary_idx], dtype=torch.int64, device=page_aligned_kv_indices.device
        )
        mamba_value_forked = self._fork_mamba_value_for_cache(
            mamba_value,
            context="cache_track_entries:primary",
        )
        if mamba_value_forked is None:
            return False

        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
                branch_checkpoint_len=page_aligned_len,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        self._marconi_record_insert_event(req, page_aligned_token_ids, page_aligned_len)
        if self.marconi_enabled:
            if getattr(req, "kv_cache_inserted_start", None) is None:
                req.kv_cache_inserted_start = new_prefix_len
                req.kv_cache_inserted_end = page_aligned_len
            else:
                if req.kv_cache_inserted_end is None:
                    req.kv_cache_inserted_end = page_aligned_len
                else:
                    req.kv_cache_inserted_end = max(
                        req.kv_cache_inserted_end, page_aligned_len
                    )
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        new_cache_protected_len = len(new_indices)
        free_end = min(new_prefix_len, new_cache_protected_len)
        dup_slice = None
        if kv_cache_protected_len < free_end:
            dup_slice = kv_indices_orig[kv_cache_protected_len:free_end].clone()

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(kv_cache_protected_len, len(new_indices))),
            new_indices[kv_cache_protected_len:],
        )

        if dup_slice is not None:
            safe = self._marconi_filter_free_indices(
                dup_slice, "cache_unfinished_req:dup_prefix"
            )
            if safe is not None:
                self.token_to_kv_pool_allocator.free(safe)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.kv_cache_protected_len = max(kv_cache_protected_len, len(new_indices))
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

        for cache_len, mamba_idx in entries[:-1]:
            cache_len = min(cache_len, len(token_ids))
            if cache_len <= 0:
                continue
            kv_indices = kv_indices_orig[:cache_len]
            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                if page_aligned_len <= 0:
                    continue
                kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
                token_ids_sub = token_ids[:page_aligned_len]
                cache_len = page_aligned_len
            else:
                kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
                token_ids_sub = token_ids[:cache_len]

            mamba_value = torch.tensor(
                [mamba_idx], dtype=torch.int64, device=kv_indices.device
            )
            mamba_value_forked = self._fork_mamba_value_for_cache(
                mamba_value,
                context="cache_track_entries:secondary",
            )
            if mamba_value_forked is None:
                continue
            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids_sub, req.extra_key),
                    value=kv_indices,
                    mamba_value=mamba_value_forked,
                    branch_checkpoint_len=cache_len,
                )
            )
            mamba_exist = result.mamba_exist
            self._marconi_record_insert_event(req, token_ids_sub, cache_len)
            if mamba_exist:
                self.req_to_token_pool.mamba_pool.free(mamba_value_forked)
        return True

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _marconi_find_nearest_live_mamba_ancestor(
        self, node: TreeNode
    ) -> Optional[TreeNode]:
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.mamba_value is not None:
                return ancestor
            ancestor = ancestor.parent
        return None

    def _marconi_replay_tokens(self, node: TreeNode) -> int:
        ancestor = self._marconi_find_nearest_live_mamba_ancestor(node)
        ancestor_prefix_len = ancestor.prefix_len if ancestor is not None else 0
        return max(node.prefix_len - ancestor_prefix_len, 0)

    def _marconi_make_candidate(
        self, node: TreeNode, action: str
    ) -> Optional[MarconiEvictionCandidate]:
        profile = self.marconi_cost_profile
        if profile is None or node.mamba_value is None:
            return None
        replay_tokens = self._marconi_replay_tokens(node)
        metrics = build_marconi_candidate_metrics(
            profile=profile,
            action=action,
            prefix_len=node.prefix_len,
            parent_prefix_len=node.parent.prefix_len if node.parent is not None else 0,
            local_kv_tokens=len(node.key),
            replay_tokens=replay_tokens,
            last_access_time=float(node.last_access_time),
        )

        return MarconiEvictionCandidate(
            node=node,
            metrics=metrics,
        )

    def _marconi_collect_leaf_nodes_full_scan(self) -> List[TreeNode]:
        nodes = []
        for node in self._collect_leaves():
            if node == self.root_node:
                continue
            if node.full_lock_ref == 0 and node.mamba_lock_ref == 0:
                nodes.append(node)
        return nodes

    def _marconi_collect_leaf_nodes(self) -> List[TreeNode]:
        return self._marconi_collect_leaf_nodes_full_scan()

    def _marconi_collect_leaf_and_single_child_nodes_full_scan(self) -> List[TreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node != self.root_node and len(node.children) <= 1:
                if node.mamba_value is not None and node.mamba_lock_ref == 0:
                    if len(node.children) == 0 and node.full_lock_ref > 0:
                        continue
                    nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def _marconi_collect_leaf_and_single_child_nodes(self) -> List[TreeNode]:
        return self._marconi_collect_leaf_and_single_child_nodes_full_scan()

    def _marconi_select_candidate(
        self, candidates: List[MarconiEvictionCandidate]
    ) -> Optional[MarconiEvictionCandidate]:
        idx = select_marconi_candidate_index(
            candidates=[candidate.metrics for candidate in candidates],
            current_time=float(get_last_access_time()),
            eff_weight=self.marconi_eff_weight,
            use_efficiency=self.marconi_use_efficiency,
        )
        return None if idx is None else candidates[idx]

    def _marconi_update_last_flops(self, candidate: MarconiEvictionCandidate) -> None:
        self.marconi_live_replay_tokens_total += candidate.replay_tokens
        self.marconi_live_replay_samples += 1
        self.marconi_last_recurrent_flops_saved = candidate.recurrent_flops
        self.marconi_last_attention_flops_saved = candidate.attention_flops
        self.marconi_last_ffn_flops_saved = candidate.ffn_flops

    def _marconi_evict_full(self, full_num_tokens: int) -> None:
        full_num_evicted = 0
        while full_num_evicted < full_num_tokens:
            candidates = [
                candidate
                for candidate in (
                    self._marconi_make_candidate(node, "full_leaf")
                    for node in self._marconi_collect_leaf_nodes()
                )
                if candidate is not None
            ]
            if not candidates:
                break
            candidate = self._marconi_select_candidate(candidates)
            if candidate is None:
                break
            self._marconi_note_eviction()
            self._marconi_update_last_flops(candidate)
            full_evicted_delta, _, _, _ = self._evict_leaf_node(candidate.node, False)
            full_num_evicted += full_evicted_delta
        if full_num_evicted > 0:
            self.marconi_live_evict_full += full_num_evicted
            logger.debug(
                "Marconi eviction(full): evicted=%d total_full=%d total_mamba=%d",
                full_num_evicted,
                self.marconi_live_evict_full,
                self.marconi_live_evict_mamba,
            )

    def _marconi_evict_mamba(self, mamba_num: int) -> None:
        mamba_num_evicted = 0
        self.marconi_live_evict_mamba_attempts += 1
        while mamba_num_evicted < mamba_num:
            candidates = []
            for node in self._marconi_collect_leaf_and_single_child_nodes():
                action = "mamba_leaf" if len(node.children) == 0 else "mamba_internal"
                candidate = self._marconi_make_candidate(node, action)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                self.marconi_live_evict_mamba_no_candidate_count += 1
                break
            candidate = self._marconi_select_candidate(candidates)
            if candidate is None:
                self.marconi_live_evict_mamba_no_candidate_count += 1
                break
            self._marconi_note_eviction()
            self._marconi_update_last_flops(candidate)
            if candidate.action == "mamba_leaf":
                _, mamba_evicted_delta, _, _ = self._evict_leaf_node(
                    candidate.node, True
                )
                mamba_num_evicted += mamba_evicted_delta
                self.marconi_live_evict_mamba_leaf_count += 1
            else:
                self.req_to_token_pool.mamba_pool.free(candidate.node.mamba_value)
                mamba_num_evicted += len(candidate.node.mamba_value)
                self.mamba_lru_list.remove_node(candidate.node)
                self._tombstone_internal_node(candidate.node)
                self.marconi_live_evict_mamba_internal_count += 1
        if mamba_num_evicted > 0:
            self.marconi_live_evict_mamba += mamba_num_evicted
            logger.debug(
                "Marconi eviction(mamba): evicted=%d total_full=%d total_mamba=%d",
                mamba_num_evicted,
                self.marconi_live_evict_full,
                self.marconi_live_evict_mamba,
            )

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert (
            x.mamba_value is not None
        ), f"leaf node mamba value must not be None, {x.id=}"
        if self.marconi_enabled:
            self._marconi_kv_mask_remove(x.value)
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        self.req_to_token_pool.mamba_pool.free(x.mamba_value)
        mamba_num_evicted = len(x.mamba_value)

        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        self._delete_leaf(x)

        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        if self.marconi_eviction_enabled:
            before = self.marconi_live_evict_mamba
            self._marconi_evict_mamba(mamba_num)
            if self.marconi_log_next_eviction_after_apply:
                self.marconi_log_next_eviction_after_apply = False
                logger.info(
                    "Marconi autotune post-apply eviction: kind=mamba "
                    "completed_requests=%d current_eff_weight=%.4f "
                    "running_req=%s queue_req=%s evict_full=%d evict_mamba=%d "
                    "evict_mamba_leaf=%d evict_mamba_internal=%d "
                    "evict_mamba_no_candidate=%d",
                    self.marconi_completed_request_count,
                    self.marconi_eff_weight,
                    self.marconi_last_running_req_count,
                    self.marconi_last_queue_req_count,
                    self.marconi_live_evict_full,
                    self.marconi_live_evict_mamba,
                    self.marconi_live_evict_mamba_leaf_count,
                    self.marconi_live_evict_mamba_internal_count,
                    self.marconi_live_evict_mamba_no_candidate_count,
                )
            return self.marconi_live_evict_mamba - before
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0
        if self.marconi_eviction_enabled:
            before = self.marconi_live_evict_full
            self._marconi_evict_full(full_num_tokens)
            if self.marconi_log_next_eviction_after_apply:
                self.marconi_log_next_eviction_after_apply = False
                logger.info(
                    "Marconi autotune post-apply eviction: kind=full "
                    "completed_requests=%d current_eff_weight=%.4f "
                    "running_req=%s queue_req=%s evict_full=%d evict_mamba=%d "
                    "evict_mamba_leaf=%d evict_mamba_internal=%d "
                    "evict_mamba_no_candidate=%d",
                    self.marconi_completed_request_count,
                    self.marconi_eff_weight,
                    self.marconi_last_running_req_count,
                    self.marconi_last_queue_req_count,
                    self.marconi_live_evict_full,
                    self.marconi_live_evict_mamba,
                    self.marconi_live_evict_mamba_leaf_count,
                    self.marconi_live_evict_mamba_internal_count,
                    self.marconi_live_evict_mamba_no_candidate_count,
                )
            return self.marconi_live_evict_full - before

        full_num_evicted = 0
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return None

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return None

    def dec_lock_ref(self, node: TreeNode):
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return None

        if node.mamba_value is not None and node.mamba_lock_ref > 0:
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> int:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        return self.full_evictable_size_

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, TreeNode, Optional[int], int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        mamba_branching_seqlen: Optional[int] = None
        matched_len = 0
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if self._node_has_full_mamba_state(node):
                best_value_len = len(value)
                best_last_node = node

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                if len(key) > prefix_len:
                    mamba_branching_seqlen = matched_len + prefix_len
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                matched_len += prefix_len
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]
                matched_len += prefix_len

                if len(key):
                    child_key = self.get_child_key_fn(key)

        if (
            mamba_branching_seqlen is None
            and matched_len > 0
            and not self._node_has_full_mamba_state(node)
            and len(node.children) > 0
        ):
            # Reuse of a purely-input prefix would require a checkpoint here.
            mamba_branching_seqlen = matched_len
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if self._node_has_full_mamba_state(node):
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, node, mamba_branching_seqlen, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        best_last_node: TreeNode,
        last_node: TreeNode,
        mamba_branching_seqlen: Optional[int],
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req
        key = params.key

        if value:
            value_full = torch.cat(value)
        else:
            value_full = torch.empty((0,), dtype=torch.int64, device=self.device)

        mamba_len = 0
        if best_value_len > 0:
            mamba_len = int(sum(len(v) for v in value[:best_value_len]))

        if len(value) > best_value_len and mamba_branching_seqlen is None:
            mamba_branching_seqlen = int(value_full.numel())
        if mamba_branching_seqlen is not None:
            mamba_branching_seqlen = self._marconi_align_branch_len(
                mamba_branching_seqlen
            )

        last_node_for_return = best_last_node

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = last_node_for_return
        if self.marconi_eviction_enabled:
            if node_update != self.root_node:
                self.full_lru_list.reset_node_mru(node_update)
            if best_last_node != self.root_node and self._node_has_full_mamba_state(
                best_last_node
            ):
                self.mamba_lru_list.reset_node_mru(best_last_node)
            cur_time = get_last_access_time()
            node_update.last_access_time = cur_time
            if best_last_node is not node_update:
                best_last_node.last_access_time = cur_time
            node_update.hit_count += 1
        else:
            self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
            self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

            # This last_access_time is for sanity check, can be deleted after validation in production
            cur_time = get_last_access_time()
            while node_update:
                node_update.last_access_time = cur_time
                cur_time -= (
                    0.00001  # assuming less than 100000 nodes in a branch of the tree
                )
                node_update = node_update.parent

        # Copy mamba state to req local space if cow is true and state is full.
        if (
            cow_mamba
            and req is not None
            and self._node_has_full_mamba_state(best_last_node)
        ):
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                # Try to alloc again after evicting an unpinned mamba state.
                if dst_index is None:
                    self.inc_lock_ref(best_last_node)
                    self.evict_mamba(1)
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(best_last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                src_index = best_last_node.mamba_value
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                req.mamba_pool_idx = dst_index[0]
            else:
                src_index = best_last_node.mamba_value
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)

        value = value_full[:mamba_len]

        if self.marconi_enabled:
            matched_len = int(value.numel())
            total_len = len(key.token_ids)
            self.marconi_live_match_count += 1
            self.marconi_live_token_total += total_len
            self.marconi_live_token_hit += matched_len
            if matched_len > 0:
                self.marconi_live_hit_count += 1
            self._log_marconi_live_stats()

            if (
                self.marconi_admission_tree is not None
                and matched_len > 0
                and key is not None
            ):
                self.marconi_admission_tree.record_cache_hit(
                    key.token_ids, matched_len, key.extra_key
                )

        if req is not None and self.marconi_admission_tree is not None:
            if not getattr(req, "marconi_admission_seeded", False):
                input_len = min(len(req.origin_input_ids), len(key.token_ids))
                if input_len > 0:
                    self.marconi_admission_tree.insert(
                        list(req.origin_input_ids[:input_len]), key.extra_key
                    )
                    req.marconi_admission_seeded = True
            admission_len, branchoff_required = (
                self.marconi_admission_tree.match_prefix(key.token_ids, key.extra_key)
            )
            self.marconi_admission_match_count += 1
            if branchoff_required:
                self.marconi_admission_branch_count += 1
            cached_prefix_len = len(value)
            cache_len = admission_len if branchoff_required else 0
            req.marconi_cache_len = cache_len if cache_len > 0 else None
            mamba_branching_seqlen = self._marconi_branch_seqlen(
                cache_len, cached_prefix_len, mamba_branching_seqlen
            )

        return MatchResult(
            device_indices=value,
            last_device_node=last_node_for_return,
            last_host_node=last_node_for_return,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(
        self,
        key: RadixKey,
        child: TreeNode,
        split_len: int,
        branchoff_mamba_value: Optional[torch.Tensor] = None,
    ) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        new_node.prefix_len = new_node.parent.prefix_len + len(new_node.value)

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        child.prefix_len = new_node.prefix_len + len(child.value)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if branchoff_mamba_value is not None:
            new_node.mamba_value = branchoff_mamba_value
            self.mamba_lru_list.insert_mru(new_node)
            self.mamba_evictable_size_ += len(branchoff_mamba_value)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        branchoff_mamba_value=None,
        branch_checkpoint_len: Optional[int] = None,
    ) -> Tuple[int, bool]:
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            mamba_value_exist = node.mamba_value is not None
            if not mamba_value_exist:
                node.mamba_value = mamba_value
                # Existing node is already in full_lru_list; add to mamba LRU.
                self.mamba_lru_list.insert_mru(node)
                self.mamba_evictable_size_ += len(mamba_value)
                node.last_access_time = get_last_access_time()
            else:
                self.mamba_lru_list.reset_node_mru(node)
                node.last_access_time = get_last_access_time()
            return 0, mamba_value_exist

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        branch_state_attached = False
        mamba_value_attached = False
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                if branchoff_mamba_value is not None:
                    mamba_value_attached = True
                new_node = self._split_node(
                    node.key,
                    node,
                    prefix_len,
                    branchoff_mamba_value,
                )
                branchoff_mamba_value = None
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

            if (
                branch_checkpoint_len is not None
                and not branch_state_attached
                and total_prefix_length == branch_checkpoint_len
                and node.mamba_value is None
            ):
                node.mamba_value = (
                    branchoff_mamba_value
                    if branchoff_mamba_value is not None
                    else mamba_value
                )
                self.mamba_lru_list.insert_mru(node)
                self.mamba_evictable_size_ += len(node.mamba_value)
                node.last_access_time = get_last_access_time()
                branch_state_attached = True
                mamba_value_attached = True

        if (
            branch_checkpoint_len is not None
            and not branch_state_attached
            and total_prefix_length == branch_checkpoint_len
            and node.mamba_value is None
        ):
            node.mamba_value = (
                branchoff_mamba_value
                if branchoff_mamba_value is not None
                else mamba_value
            )
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(node.mamba_value)
            node.last_access_time = get_last_access_time()
            branch_state_attached = True
            mamba_value_attached = True

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.prefix_len = node.prefix_len + len(value)
            self.full_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            if self.marconi_enabled:
                self._marconi_kv_mask_add(new_node.value)
            # If we already attached a mamba state at the branch checkpoint, do not
            # attach (and double-count) the same state at the new leaf.
            if not mamba_value_attached:
                new_node.mamba_value = mamba_value
                self.mamba_lru_list.insert_mru(new_node)
                self.mamba_evictable_size_ += len(mamba_value)
            else:
                new_node.mamba_value = None
            return total_prefix_length, False

        # len(key) == 0
        mamba_value_exist = node.mamba_value is not None and not mamba_value_attached
        if not mamba_value_exist:
            if not mamba_value_attached:
                node.mamba_value = mamba_value
                self.mamba_lru_list.insert_mru(node)
                self.mamba_evictable_size_ += len(mamba_value)
            else:
                self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()
        else:
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        self.full_lru_list.reset_node_mru(node)
        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            if self.marconi_enabled:
                self._marconi_kv_mask_remove(node.parent.value)
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_leaves(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())
        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
