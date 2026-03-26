from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.mem_cache.marconi_cost_model import MarconiCostProfile
from sglang.srt.mem_cache.marconi_replay_core import (
    MarconiReplayCandidateMetrics,
    build_marconi_candidate_metrics,
    select_marconi_candidate_index,
)


def _key_match(key0: tuple[int, ...], key1: tuple[int, ...]) -> int:
    idx = 0
    for token0, token1 in zip(key0, key1):
        if token0 != token1:
            break
        idx += 1
    return idx


def _child_key(extra_key: Optional[str], key: tuple[int, ...]):
    token = key[0]
    return token if extra_key is None else (extra_key, token)


@dataclass(frozen=True)
class MarconiReplayNodeSnapshot:
    key: tuple[int, ...]
    extra_key: Optional[str]
    prefix_len: int
    has_mamba: bool
    last_access_time: float
    children: tuple["MarconiReplayNodeSnapshot", ...] = ()


@dataclass(frozen=True)
class MarconiReplaySnapshot:
    root: MarconiReplayNodeSnapshot
    cost_profile: MarconiCostProfile
    full_capacity_tokens: int
    mamba_capacity_states: int
    current_time: float


@dataclass(frozen=True)
class MarconiReplayInsertEvent:
    token_ids: tuple[int, ...]
    branch_checkpoint_len: Optional[int] = None


@dataclass(frozen=True)
class MarconiReplayRequest:
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    extra_key: Optional[str]
    insert_events: tuple[MarconiReplayInsertEvent, ...] = ()

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.input_ids + self.output_ids


@dataclass
class _ReplayNode:
    key: tuple[int, ...]
    extra_key: Optional[str]
    prefix_len: int
    has_mamba: bool
    last_access_time: float
    parent: Optional["_ReplayNode"] = None
    children: dict[object, "_ReplayNode"] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}


@dataclass
class _Candidate:
    node: _ReplayNode
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


class _ReplayCache:
    def __init__(self, snapshot: MarconiReplaySnapshot, eff_weight: float):
        self.cost_profile = snapshot.cost_profile
        self.eff_weight = eff_weight
        self.full_capacity_tokens = max(0, snapshot.full_capacity_tokens)
        self.mamba_capacity_states = max(0, snapshot.mamba_capacity_states)
        self.root = self._restore_node(snapshot.root, None)
        self.current_time = max(snapshot.current_time, self._max_time(self.root))
        self.full_used_tokens = self._count_full_tokens(self.root)
        self.mamba_used_states = self._count_mamba_states(self.root)

    def _restore_node(
        self,
        snapshot_node: MarconiReplayNodeSnapshot,
        parent: Optional[_ReplayNode],
    ) -> _ReplayNode:
        node = _ReplayNode(
            key=snapshot_node.key,
            extra_key=snapshot_node.extra_key,
            prefix_len=snapshot_node.prefix_len,
            has_mamba=snapshot_node.has_mamba,
            last_access_time=snapshot_node.last_access_time,
            parent=parent,
        )
        for child_snapshot in snapshot_node.children:
            child = self._restore_node(child_snapshot, node)
            node.children[_child_key(child.extra_key, child.key)] = child
        return node

    def _max_time(self, node: _ReplayNode) -> float:
        latest = node.last_access_time
        for child in node.children.values():
            latest = max(latest, self._max_time(child))
        return latest

    def _count_full_tokens(self, node: _ReplayNode) -> int:
        total = len(node.key)
        for child in node.children.values():
            total += self._count_full_tokens(child)
        return total

    def _count_mamba_states(self, node: _ReplayNode) -> int:
        total = 1 if node.has_mamba else 0
        for child in node.children.values():
            total += self._count_mamba_states(child)
        return total

    def _tick(self) -> float:
        self.current_time += 1.0
        return self.current_time

    def _nearest_live_mamba_ancestor(self, node: _ReplayNode) -> Optional[_ReplayNode]:
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.has_mamba:
                return ancestor
            ancestor = ancestor.parent
        return None

    def _replay_tokens(self, node: _ReplayNode) -> int:
        ancestor = self._nearest_live_mamba_ancestor(node)
        ancestor_prefix_len = ancestor.prefix_len if ancestor is not None else 0
        return max(node.prefix_len - ancestor_prefix_len, 0)

    def match_prefix(self, input_ids: tuple[int, ...], extra_key: Optional[str]) -> int:
        key = input_ids
        node = self.root
        best_node = self.root
        last_node = self.root
        while key:
            child = node.children.get(_child_key(extra_key, key))
            if child is None:
                break
            prefix_len = _key_match(child.key, key)
            last_node = child
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
            if child.has_mamba:
                best_node = child

        current_ts = self._tick()
        if last_node is not self.root:
            last_node.last_access_time = current_ts
        if best_node is not self.root:
            best_node.last_access_time = current_ts
        return best_node.prefix_len if best_node is not self.root else 0

    def _ensure_full_capacity(self, new_tokens: int) -> None:
        required = self.full_used_tokens + new_tokens - self.full_capacity_tokens
        if required <= 0:
            return
        self._evict_full(required)

    def _ensure_mamba_capacity(self, new_states: int) -> None:
        required = self.mamba_used_states + new_states - self.mamba_capacity_states
        if required <= 0:
            return
        self._evict_mamba(required)

    def _attach_mamba(self, node: _ReplayNode) -> None:
        if node.has_mamba:
            return
        self._ensure_mamba_capacity(1)
        node.has_mamba = True
        node.last_access_time = self._tick()
        self.mamba_used_states += 1

    def insert(
        self,
        token_ids: tuple[int, ...],
        extra_key: Optional[str],
        branch_checkpoint_len: Optional[int],
    ) -> None:
        key = token_ids
        node = self.root
        total_prefix_len = 0
        branch_attached = False
        leaf_attached = False

        while key:
            child = node.children.get(_child_key(extra_key, key))
            if child is None:
                break
            child.last_access_time = self._tick()
            prefix_len = _key_match(child.key, key)
            total_prefix_len += prefix_len

            if prefix_len < len(child.key):
                node = self._split_node(child, prefix_len)
                if (
                    branch_checkpoint_len is not None
                    and not branch_attached
                    and total_prefix_len == branch_checkpoint_len
                    and not node.has_mamba
                ):
                    self._attach_mamba(node)
                    branch_attached = True
                    leaf_attached = True
                key = key[prefix_len:]
                break

            node = child
            key = key[prefix_len:]
            if (
                branch_checkpoint_len is not None
                and not branch_attached
                and total_prefix_len == branch_checkpoint_len
                and not node.has_mamba
            ):
                self._attach_mamba(node)
                branch_attached = True
                leaf_attached = True

        if (
            branch_checkpoint_len is not None
            and not branch_attached
            and total_prefix_len == branch_checkpoint_len
            and not node.has_mamba
        ):
            self._attach_mamba(node)
            branch_attached = True
            leaf_attached = True

        if key:
            self._ensure_full_capacity(len(key))
            new_node = _ReplayNode(
                key=key,
                extra_key=extra_key,
                prefix_len=node.prefix_len + len(key),
                has_mamba=False,
                last_access_time=self._tick(),
                parent=node,
            )
            node.children[_child_key(extra_key, key)] = new_node
            self.full_used_tokens += len(key)
            if not leaf_attached:
                self._attach_mamba(new_node)
            return

        if not leaf_attached:
            self._attach_mamba(node)
        else:
            node.last_access_time = self._tick()

    def _split_node(self, child: _ReplayNode, split_len: int) -> _ReplayNode:
        new_node = _ReplayNode(
            key=child.key[:split_len],
            extra_key=child.extra_key,
            prefix_len=child.parent.prefix_len + split_len,
            has_mamba=False,
            last_access_time=self._tick(),
            parent=child.parent,
        )
        parent = child.parent
        parent.children[_child_key(new_node.extra_key, new_node.key)] = new_node
        child.parent = new_node
        child.key = child.key[split_len:]
        child.prefix_len = new_node.prefix_len + len(child.key)
        child.last_access_time = self._tick()
        new_node.children[_child_key(child.extra_key, child.key)] = child
        return new_node

    def _collect_leaf_nodes(self) -> list[_ReplayNode]:
        leaves: list[_ReplayNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and not node.children and node.has_mamba:
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def _collect_mamba_candidates(self) -> list[_ReplayNode]:
        candidates: list[_ReplayNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and node.has_mamba and len(node.children) <= 1:
                candidates.append(node)
            stack.extend(node.children.values())
        return candidates

    def _make_candidate(self, node: _ReplayNode, action: str) -> _Candidate:
        metrics = build_marconi_candidate_metrics(
            profile=self.cost_profile,
            action=action,
            prefix_len=node.prefix_len,
            parent_prefix_len=node.parent.prefix_len if node.parent is not None else 0,
            local_kv_tokens=len(node.key),
            replay_tokens=self._replay_tokens(node),
            last_access_time=node.last_access_time,
        )
        return _Candidate(node=node, metrics=metrics)

    def _select_candidate(self, candidates: list[_Candidate]) -> Optional[_Candidate]:
        idx = select_marconi_candidate_index(
            candidates=[candidate.metrics for candidate in candidates],
            current_time=self.current_time + 1.0,
            eff_weight=self.eff_weight,
            use_efficiency=True,
        )
        return None if idx is None else candidates[idx]

    def _evict_full(self, full_num_tokens: int) -> None:
        remaining = full_num_tokens
        while remaining > 0:
            candidates = [
                self._make_candidate(node, "full_leaf")
                for node in self._collect_leaf_nodes()
            ]
            candidate = self._select_candidate(candidates)
            if candidate is None:
                break
            evicted_tokens, _ = self._evict_leaf(candidate.node)
            remaining -= evicted_tokens

    def _evict_mamba(self, mamba_num: int) -> None:
        remaining = mamba_num
        while remaining > 0:
            candidates = []
            for node in self._collect_mamba_candidates():
                action = "mamba_leaf" if not node.children else "mamba_internal"
                candidates.append(self._make_candidate(node, action))
            candidate = self._select_candidate(candidates)
            if candidate is None:
                break
            if candidate.action == "mamba_leaf":
                _, evicted_states = self._evict_leaf(candidate.node)
            else:
                self._tombstone_internal(candidate.node)
                evicted_states = 1
            remaining -= evicted_states

    def _evict_leaf(self, node: _ReplayNode) -> tuple[int, int]:
        parent = node.parent
        parent.children.pop(_child_key(node.extra_key, node.key), None)
        evicted_tokens = len(node.key)
        self.full_used_tokens -= evicted_tokens
        evicted_states = 1 if node.has_mamba else 0
        if node.has_mamba:
            self.mamba_used_states -= 1
        current = parent
        while (
            current is not None
            and current is not self.root
            and not current.children
            and not current.has_mamba
        ):
            parent = current.parent
            parent.children.pop(_child_key(current.extra_key, current.key), None)
            self.full_used_tokens -= len(current.key)
            current = parent
        return evicted_tokens, evicted_states

    def _tombstone_internal(self, node: _ReplayNode) -> None:
        if not node.has_mamba:
            return
        node.has_mamba = False
        self.mamba_used_states -= 1


def tune_marconi_eff_weight(
    *,
    snapshot: Optional[MarconiReplaySnapshot],
    request_history_window: list[MarconiReplayRequest],
    weight_grid: tuple[float, ...],
) -> Optional[float]:
    if snapshot is None or not request_history_window or not weight_grid:
        return None

    best_weight = None
    best_flops_saved = float("-inf")
    for eff_weight in weight_grid:
        replay_cache = _ReplayCache(snapshot=snapshot, eff_weight=eff_weight)
        total_flops_saved = 0.0
        for request in request_history_window:
            matched_len = replay_cache.match_prefix(request.input_ids, request.extra_key)
            total_flops_saved += replay_cache.cost_profile.recurrent_flops(matched_len)
            total_flops_saved += replay_cache.cost_profile.attn_flops_delta(
                matched_len, 0
            )
            total_flops_saved += replay_cache.cost_profile.ffn_flops_delta(
                matched_len, 0
            )
            for event in request.insert_events:
                replay_cache.insert(
                    event.token_ids,
                    request.extra_key,
                    event.branch_checkpoint_len,
                )

        if total_flops_saved > best_flops_saved:
            best_flops_saved = total_flops_saved
            best_weight = eff_weight

    return best_weight
