import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_num_new_pages

_is_npu = is_npu()

if _is_npu:
    import torch_npu

    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )


class _OffsetAttentionTagTable:
    """Namespace a shared attention-tag table for an independent page pool."""

    def __init__(self, table, offset: int):
        self.table = table
        self.offset = int(offset)

    @property
    def device(self):
        return self.table.device

    def _global_page_ids(self, page_ids: torch.Tensor) -> torch.Tensor:
        if self.offset == 0:
            return page_ids
        return page_ids + self.offset

    def bump_generations(self, page_ids: torch.Tensor) -> None:
        self.table.bump_generations(self._global_page_ids(page_ids))

    def generation_of(self, page_ids: torch.Tensor) -> torch.Tensor:
        return self.table.generation_of(self._global_page_ids(page_ids))

    def write_tags(self, page_ids: torch.Tensor, tags: torch.Tensor) -> None:
        self.table.write_tags(self._global_page_ids(page_ids), tags)

    def read_tags(self, page_ids: torch.Tensor) -> torch.Tensor:
        return self.table.read_tags(self._global_page_ids(page_ids))

    def record_free(self, page_ids, *, deferred: bool = False) -> None:
        page_ids = torch.as_tensor(page_ids, dtype=torch.long, device=self.device)
        self.table.record_free(self._global_page_ids(page_ids), deferred=deferred)

    def record_free_released(self, page_ids) -> None:
        page_ids = torch.as_tensor(page_ids, dtype=torch.long, device=self.device)
        self.table.record_free_released(self._global_page_ids(page_ids))


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: BaseSWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, BaseSWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        else:
            if _is_npu:
                PagedTokenToKVPoolAllocatorClass = NPUPagedTokenToKVPoolAllocator
            else:
                PagedTokenToKVPoolAllocatorClass = PagedTokenToKVPoolAllocator
            self.full_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size,
                page_size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size_swa,
                page_size,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self._free_group_ops = []
        self.attention_tag_table = None
        self.transfer_page_pin_manager = None

        self._kvcache = kvcache
        self.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    # Slot-conservation views for the leak invariant. On the non-shared allocator
    # the static budget IS physical (conserve == physical); the shared composite
    # overrides these with the static-cap view.
    def _conserve_full_available_size(self):
        return self.full_available_size()

    def _conserve_swa_available_size(self):
        return self.swa_available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def _num_full_pages(self) -> int:
        return self._size_full // self.page_size

    def _num_swa_pages(self) -> int:
        return self._size_swa // self.page_size

    def attention_tag_num_pages(self) -> int:
        return self._num_full_pages() + self._num_swa_pages()

    def attention_tag_swa_page_offset(self) -> int:
        return self._num_full_pages()

    def attention_tag_full_page_ids(self, page_ids):
        return page_ids

    def attention_tag_swa_page_ids(self, page_ids):
        offset = self.attention_tag_swa_page_offset()
        if isinstance(page_ids, torch.Tensor):
            return page_ids + offset
        return [int(page_id) + offset for page_id in page_ids]

    def attach_attention_tag_table(self, table) -> None:
        # Full and SWA sub-pools have independent page-id namespaces, both
        # starting at 1.  Share one table by offsetting all SWA page ids after the
        # full-pool range.
        self.attention_tag_table = table
        self.full_attn_allocator.attach_attention_tag_table(table)
        self.swa_attn_allocator.attach_attention_tag_table(
            _OffsetAttentionTagTable(table, self.attention_tag_swa_page_offset())
        )

    def attach_transfer_page_pin_manager(self, manager) -> None:
        self.transfer_page_pin_manager = manager

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages
            <= self.full_attn_allocator.available_size() // self.page_size
            and num_swa_pages
            <= self.swa_attn_allocator.available_size() // self.page_size
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)

        return alloc_full_indices

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        """Allocate full KV for the whole extend and SWA KV only for the tail.

        This is used by disaggregated decode preallocation: decode receives full
        prompt KV for full-attention layers, but only the sliding-window state is
        transferred for SWA layers.
        """
        assert self.page_size > 1
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= extend_num_tokens

        num_full_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size
        if not self.new_pages_available(num_full_pages, num_swa_pages):
            return None

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_full_pages,
        )
        assert alloc_full_indices is not None

        if swa_tail_len == 0:
            return alloc_full_indices

        device = self.device
        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
        swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            swa_prefix_lens,
            swa_prefix_lens_cpu,
            swa_seq_lens,
            swa_seq_lens_cpu,
            swa_last_loc,
            swa_tail_len,
            num_new_pages=num_swa_pages,
        )
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(
            alloc_full_indices[-swa_tail_len:], alloc_swa_indices
        )
        if swa_tail_len < extend_num_tokens:
            self.full_to_swa_index_mapping[
                alloc_full_indices[:-swa_tail_len].to(torch.int64)
            ] = 0
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        if _is_npu:
            indices_2d = alloc_full_indices.to(torch.int64).unsqueeze(-1)
            torch_npu.npu_scatter_nd_update_(
                self.full_to_swa_index_mapping,
                indices_2d,
                alloc_swa_indices.to(torch.int64),
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            free_index = self._filter_transfer_pinned_full_indices(free_index)
            if free_index.numel() == 0:
                return
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
            self._free_group_ops.append(("full", free_index))
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """Write full_to_swa_index_mapping[full_indices[i]] = swa_indices[i].

        Used by HiCache load-back path to rebuild the mapping after FULL and SWA device alloc.
        """
        if full_indices.numel() == 0:
            return
        assert full_indices.numel() == swa_indices.numel()
        full_indices = full_indices.to(torch.int64)
        swa_indices = swa_indices.to(self.full_to_swa_index_mapping.dtype)
        self.full_to_swa_index_mapping[full_indices] = swa_indices

    def free_swa(self, free_index: torch.Tensor, *, released: bool = False):
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self._free_group_ops.append(("swa", free_index, released))
            return

        if self.page_size == 1:
            mapping_indices = free_index
        else:
            mapping_indices = self._expand_to_full_pages(free_index)

        swa_indices = self.full_to_swa_index_mapping[mapping_indices]
        swa_indices = swa_indices[swa_indices > 0]
        if released:
            swa_page_ids = torch.unique(swa_indices // self.page_size)
            self.swa_attn_allocator._release_transfer_pinned_pages(
                swa_page_ids.detach().cpu().tolist()
            )
        else:
            self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[mapping_indices] = 0

    def _filter_transfer_pinned_full_indices(
        self, free_index: torch.Tensor
    ) -> torch.Tensor:
        manager = self.transfer_page_pin_manager
        if manager is None or free_index.numel() == 0:
            return free_index
        if self.page_size == 1:
            return manager.defer_free_pages(free_index)

        free_pages = torch.unique(free_index // self.page_size)
        unpinned_pages = manager.defer_free_pages(free_pages)
        if unpinned_pages.numel() == free_pages.numel():
            return free_index
        unpinned_set = set(unpinned_pages.detach().cpu().tolist())
        deferred_pages = [
            page
            for page in free_pages.detach().cpu().tolist()
            if page not in unpinned_set
        ]
        if deferred_pages and self.swa_attn_allocator.attention_tag_table is not None:
            deferred = torch.tensor(
                deferred_pages, dtype=torch.int64, device=free_index.device
            )
            deferred_indices = self._expand_to_full_pages(deferred * self.page_size)
            swa_indices = self.full_to_swa_index_mapping[deferred_indices]
            swa_page_ids = torch.unique(swa_indices[swa_indices > 0] // self.page_size)
            self.swa_attn_allocator.attention_tag_table.record_free(
                swa_page_ids, deferred=True
            )
        if unpinned_pages.numel() == 0:
            return free_index.new_empty((0,))
        page_offsets = torch.arange(
            self.page_size, dtype=free_index.dtype, device=free_index.device
        )
        return (
            unpinned_pages[:, None] * self.page_size + page_offsets[None, :]
        ).reshape(-1)

    def _release_transfer_pinned_pages(self, page_ids) -> None:
        if not page_ids:
            return
        pages = torch.tensor(page_ids, dtype=torch.int64, device=self.device)
        self.full_attn_allocator._release_transfer_pinned_pages(page_ids)
        if self.page_size == 1:
            full_indices = pages
        else:
            page_offsets = torch.arange(
                self.page_size, dtype=torch.int64, device=self.device
            )
            full_indices = (
                pages[:, None] * self.page_size + page_offsets[None, :]
            ).reshape(-1)
        self.free_swa(full_indices, released=True)

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        pages = torch.unique(indices // self.page_size)
        page_offsets = torch.arange(
            self.page_size, dtype=indices.dtype, device=indices.device
        )
        return (pages[:, None] * self.page_size + page_offsets[None, :]).reshape(-1)

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self._clear_transfer_page_pins()
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []
        self._free_group_ops = []

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []
        self._free_group_ops = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        operations = self._free_group_ops
        self.free_group = []
        self._free_group_ops = []
        for operation in operations:
            if operation[0] == "full":
                self.free(operation[1])
            else:
                self.free_swa(operation[1], released=operation[2])

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )


class PureSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Single-pool allocator for models whose every layer is sliding-window attention."""

    def __init__(
        self,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: BaseSWAKVPool,
        need_sort: bool,
    ):
        assert page_size == 1
        assert isinstance(kvcache, BaseSWAKVPool)

        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.need_sort = need_sort
        self._size_full = self._size_swa = size_swa

        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )
        self.full_attn_allocator = self.swa_attn_allocator

        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.arange(size_swa + page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self._kvcache = kvcache
        self.swa_attn_allocator.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def available_size(self):
        return self.swa_attn_allocator.available_size()

    def full_available_size(self):
        return self.swa_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        avail = self.swa_attn_allocator.available_size() // self.page_size
        return num_full_pages <= avail and num_swa_pages <= avail

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return kv_indices

    def alloc(self, need_size: int):
        assert self.page_size == 1
        return self.swa_attn_allocator.alloc(need_size)

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def alloc_extend_swa_tail(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.swa_attn_allocator.free(free_index[free_index > 0])
        else:
            self.free_group.append(free_index)
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.swa_attn_allocator.free(free_index[free_index > 0])

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
        self.free_group = []

    def backup_state(self):
        return self.swa_attn_allocator.backup_state()

    def restore_state(self, state):
        self.swa_attn_allocator.restore_state(state)

    def clear(self):
        self.swa_attn_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []
