"""
Copyright 2025 SGLang Team
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

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict, Set, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class KVTransferPagePinManager:
    """Defers allocator frees for pages that can still receive direct PD writes."""

    def __init__(self, allocator: BaseTokenToKVPoolAllocator):
        self.allocator = allocator
        self._pin_counts: Dict[int, int] = {}
        self._deferred_free_pages: Set[int] = set()

    @staticmethod
    def _to_page_tuple(page_ids) -> Tuple[int, ...]:
        if page_ids is None:
            return ()
        if isinstance(page_ids, torch.Tensor):
            raw = page_ids.detach().cpu().reshape(-1).tolist()
        elif hasattr(page_ids, "reshape") and hasattr(page_ids, "tolist"):
            raw = page_ids.reshape(-1).tolist()
        else:
            raw = list(page_ids)
        return tuple(int(page_id) for page_id in raw if int(page_id) > 0)

    def pin_pages(self, page_ids) -> Tuple[int, ...]:
        pages = self._to_page_tuple(page_ids)
        for page_id in pages:
            self._pin_counts[page_id] = self._pin_counts.get(page_id, 0) + 1
        return pages

    def defer_free_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        if not self._pin_counts or page_indices.numel() == 0:
            return page_indices
        page_ids = page_indices.detach().cpu().reshape(-1).tolist()
        unpinned = []
        for page_id_raw in page_ids:
            page_id = int(page_id_raw)
            if self._pin_counts.get(page_id, 0) > 0:
                self._deferred_free_pages.add(page_id)
                table = self.allocator.attention_tag_table
                if table is not None:
                    table.record_free([page_id], deferred=True)
            else:
                unpinned.append(page_id)
        if len(unpinned) == len(page_ids):
            return page_indices
        if not unpinned:
            return page_indices.new_empty((0,))
        return torch.tensor(
            unpinned, dtype=page_indices.dtype, device=page_indices.device
        )

    def release_pages(self, page_ids) -> None:
        pages = self._to_page_tuple(page_ids)
        releasable = []
        for page_id in pages:
            count = self._pin_counts.get(page_id, 0)
            if count <= 1:
                self._pin_counts.pop(page_id, None)
                if page_id in self._deferred_free_pages:
                    self._deferred_free_pages.remove(page_id)
                    releasable.append(page_id)
            else:
                self._pin_counts[page_id] = count - 1
        self.allocator._release_transfer_pinned_pages(releasable)

    def clear(self) -> None:
        self._pin_counts.clear()
        self._deferred_free_pages.clear()


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        # Optional sidecar KV attention-tag table (PD disaggregation protection).
        # Stays ``None`` unless explicitly attached, so the default allocator
        # hot path is unchanged for non-PD / protection-disabled serving.
        self.attention_tag_table = None
        self.transfer_page_pin_manager = None

    def attach_attention_tag_table(self, table) -> None:
        """Attach a KV attention-tag table so allocations bump page generations.

        Only used when KV page protection is enabled for PD decode.
        """
        self.attention_tag_table = table

    def attach_transfer_page_pin_manager(self, manager) -> None:
        """Attach decode-side transfer pinning for direct PD KV writes."""
        self.transfer_page_pin_manager = manager

    def _filter_transfer_pinned_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        manager = self.transfer_page_pin_manager
        if manager is None:
            return page_indices
        return manager.defer_free_pages(page_indices)

    def _release_transfer_pinned_pages(self, page_ids) -> None:
        if not page_ids:
            return
        pages = torch.tensor(page_ids, dtype=torch.int64, device=self.device)
        if self.attention_tag_table is not None:
            self.attention_tag_table.record_free_released(pages)
        if self.need_sort:
            self.release_pages = torch.cat((pages, self.release_pages))
        else:
            self.free_pages = torch.cat((pages, self.free_pages))

    def _clear_transfer_page_pins(self) -> None:
        manager = self.transfer_page_pin_manager
        if manager is not None:
            manager.clear()

    def _bump_page_generations(self, page_ids) -> None:
        """Bump allocation generations for newly allocated physical pages.

        No-op unless an attention-tag table is attached (one attribute check on the
        disabled path).
        """
        if self.attention_tag_table is None:
            return
        self.attention_tag_table.bump_generations(page_ids)

    def _record_page_free(self, page_ids) -> None:
        if self.attention_tag_table is not None:
            self.attention_tag_table.record_free(page_ids)

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
