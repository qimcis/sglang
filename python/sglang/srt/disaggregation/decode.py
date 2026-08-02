"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    send_metadata_with_staging_registration,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.base import KVTransferPagePinManager
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    page_align_floor,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils import get_num_new_pages, is_cuda
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
        speculative_eagle_topk: Optional[int] = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_allocator.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    transfer_pinned_page_ids: Tuple[int, ...] = ()
    transfer_pinned_page_generations: Tuple[int, ...] = ()

    # HiCache Status
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    hicache_restored_node: Any = None
    hicache_load_consumer_index: int = -1
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        # KV page protection / transfer checksums (PD disaggregation, gated).
        # Constructed PD-decode-side only; inert for non-PD serving.  Fails fast
        # on unsupported allocators/backends rather than silently disabling.
        self._init_kv_protection()
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

    def _init_kv_protection(self) -> None:
        from sglang.srt.mem_cache.kv_page_tags import (
            KVPageProtectionManager,
            KVProtectionConfig,
        )

        config = KVProtectionConfig.from_env(is_pd_decode=True)
        if not config.enabled:
            self.scheduler.kv_protection_manager = None
            return

        allocator = self.token_to_kv_pool_allocator
        page_size = allocator.page_size
        if hasattr(allocator, "attention_tag_num_pages"):
            num_pages = max(1, int(allocator.attention_tag_num_pages()))
        else:
            num_pages = max(1, allocator.size // max(page_size, 1))
        metrics_collector = (
            self.scheduler.metrics_collector
            if getattr(self.scheduler, "metrics_reporter", None) is not None
            and self.scheduler.metrics_reporter.enable_metrics
            else None
        )
        manager = KVPageProtectionManager(
            config,
            allocator=allocator,
            num_pages=num_pages,
            page_size=page_size,
            num_request_slots=self.scheduler.req_to_token_pool.req_to_token.shape[0],
            num_logical_pages=(
                self.scheduler.req_to_token_pool.req_to_token.shape[1] + page_size - 1
            )
            // page_size,
            device=allocator.device,
            metrics_collector=metrics_collector,
            transfer_backend=(
                str(self.transfer_backend.value)
                if hasattr(self.transfer_backend, "value")
                else str(self.transfer_backend)
            ),
            is_spec_decode=not self.scheduler.spec_algorithm.is_none(),
            supports_spec_target_verify=getattr(
                self.scheduler.tp_worker.model_runner,
                "kv_requires_pre_indexer_page_validation",
                False,
            )
            and getattr(
                self.scheduler.tp_worker.model_runner,
                "kv_fused_page_protection_enabled",
                False,
            ),
            is_cuda_device=is_cuda(),
        )
        self.scheduler.kv_protection_manager = manager
        logger.info(
            "KV page protection enabled (attention_tags=%s, transfer_checksum=%s)",
            config.enable_attention_tags,
            config.checksum_enabled,
        )

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _prepare_transfer_page_tags(
        self,
        *,
        req: Req,
        page_indices,
        page_position_start: int,
        state_indices: Optional[List],
        state_types: Sequence[StateType],
        seq_len: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create expected transfer tags retained locally and sent to prefill."""
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        if manager is not None and manager.config.enable_attention_tags:
            manager.clear_request_slot(req.req_pool_idx)
        if _is_fake_transfer(req, self.scheduler.server_args):
            req.kv_transfer_page_tag_manifest = None
            return None, None
        if (
            manager is None
            or not manager.config.enable_attention_tags
            or self.scheduler.enable_hisparse
        ):
            req.kv_transfer_page_tag_manifest = None
            return None, None

        try:
            from sglang.srt.mem_cache.kv_page_tags import (
                KV_EXPECTED_MAPPING_SWA,
                TransferPageTagManifestGroup,
            )

            manifests = []

            def append_manifest(manifest) -> None:
                if manifest is None or manifest.num_pages == 0:
                    return
                manifests.append(manifest)

            full_pages = np.asarray(page_indices, dtype=np.int64).reshape(-1)
            if full_pages.size > 0:
                positions = np.arange(
                    page_position_start,
                    page_position_start + full_pages.size,
                    dtype=np.int64,
                )
                append_manifest(
                    manager.register_transfer_page_tags(
                        page_physical_ids=full_pages.tolist(),
                        request_pool_idx=req.req_pool_idx,
                        page_positions=positions.tolist(),
                        bootstrap_room=req.bootstrap_room or 0,
                    )
                )

            if state_indices:
                page_size = self.token_to_kv_pool_allocator.page_size
                allocator = self.token_to_kv_pool_allocator
                for st, component_indices in zip(
                    state_types, state_indices, strict=False
                ):
                    if component_indices is None:
                        continue
                    pages = np.asarray(component_indices, dtype=np.int64).reshape(-1)
                    if pages.size == 0:
                        continue
                    if st == StateType.SWA and hasattr(
                        allocator, "attention_tag_swa_page_ids"
                    ):
                        window_size = self.scheduler.sliding_window_size or seq_len
                        window_start = max(0, seq_len - window_size)
                        window_start = page_align_floor(window_start, page_size)
                        positions = np.arange(
                            window_start // page_size,
                            window_start // page_size + pages.size,
                            dtype=np.int64,
                        )
                        page_ids = allocator.attention_tag_swa_page_ids(
                            [int(x) for x in pages.tolist()]
                        )
                        max_num_pages = (window_size + page_size - 1) // page_size + 1
                    elif st == StateType.DSA:
                        positions = np.arange(pages.size, dtype=np.int64)
                        page_ids = [int(x) for x in pages.tolist()]
                        max_num_pages = None
                    else:
                        continue
                    append_manifest(
                        manager.register_transfer_page_tags(
                            page_physical_ids=page_ids,
                            request_pool_idx=req.req_pool_idx,
                            page_positions=positions.tolist(),
                            bootstrap_room=req.bootstrap_room or 0,
                            max_num_pages=max_num_pages,
                            mapping_namespace=(
                                KV_EXPECTED_MAPPING_SWA if st == StateType.SWA else 0
                            ),
                        )
                    )

            if len(manifests) > 1:
                req.kv_transfer_page_tag_manifest = TransferPageTagManifestGroup(
                    tuple(manifests)
                )
            elif manifests:
                req.kv_transfer_page_tag_manifest = manifests[0]
            else:
                req.kv_transfer_page_tag_manifest = None

            page_tensors = [
                manifest.physical_page_ids_t.to(dtype=torch.int32)
                for manifest in manifests
            ]
            tag_tensors = [manifest.expected_tags_t for manifest in manifests]
            if not page_tensors:
                return None, None
            pages_t = (
                page_tensors[0] if len(page_tensors) == 1 else torch.cat(page_tensors)
            )
            tags_t = tag_tensors[0] if len(tag_tensors) == 1 else torch.cat(tag_tensors)
            packed = torch.stack((pages_t, tags_t)).detach().cpu().numpy()
            return packed[0], packed[1]
        except Exception as e:
            logger.error(
                "KV transfer page tag preparation failed for rid=%s: %s", req.rid, e
            )
            req.kv_transfer_page_tag_manifest = None
            from sglang.srt.mem_cache.kv_page_tags import (
                KVProtectionBookkeepingError,
                attach_kv_protection_incident,
                emit_kv_protection_incident,
            )

            error = KVProtectionBookkeepingError(
                rid=req.rid,
                bootstrap_room=req.bootstrap_room,
                cause="transfer_page_tag_preparation",
                detail=str(e),
            )
            attach_kv_protection_incident(
                error, kind="transfer_page_tag", phase="tag_preparation"
            )
            emit_kv_protection_incident(error, logger)
            raise error from e

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        swa_reserved = self.num_reserved_decode_tokens
        if self.scheduler.server_args.disable_radix_cache:
            swa_reserved = 0
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + swa_reserved,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            transfer_kv_pool.get_contiguous_buf_infos()
        )
        kv_data_mem_kinds = (
            ["DRAM"] * len(kv_data_ptrs)
            if self.scheduler.enable_hisparse
            else ["VRAM"] * len(kv_data_ptrs)
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
            kv_data_mem_kinds += ["VRAM"] * len(device_kv_data_ptrs[c4_layer_num:])
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if self.transfer_backend == TransferBackend.NIXL:
            kv_args.kv_data_mem_kinds = kv_data_mem_kinds
        kv_args.page_size = self.token_to_kv_pool.page_size
        kv_args.transfer_page_tag_manager = getattr(
            self.scheduler, "kv_protection_manager", None
        )
        manager = kv_args.transfer_page_tag_manager
        kv_args.kv_protection_config = getattr(manager, "config", None)

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:
        """
        Match a request against the decode-side radix cache, lock the matched
        node to prevent eviction, and return the matched prefix information.
        """
        result = match_prefix_for_req(
            self.tree_cache,
            req,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            include_req=True,
        )
        # Always lock to match aggregated scheduling behavior
        self.tree_cache.inc_lock_ref(result.last_device_node)
        return self._build_decode_prefix_match(req, result)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        if self._uses_swa_tail_prealloc():
            _, swa_required = self._prealloc_required_tokens(req)
            swa_capacity = self.token_to_kv_pool_allocator.size_swa
            if swa_required > swa_capacity:
                message = (
                    f"Request {req.rid} requires too many SWA KV tokens for "
                    f"decode preallocation: {swa_required} > {swa_capacity}"
                )
                logger.error(message)
                prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self.queue.clear()
        self.retracted_queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        if uses_swa_tail_prealloc:
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if full_required > full_allocatable_tokens:
                break
            if uses_swa_tail_prealloc and swa_required > swa_allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens -= full_required
            if uses_swa_tail_prealloc:
                swa_allocatable_tokens -= swa_required

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            return

        # Still poll if any receiver was aborted, otherwise it stays stuck.
        if all(decode_req.waiting_for_input for decode_req in self.queue) and not any(
            decode_req.kv_receiver.conclude_state == KVPoll.Failed
            for decode_req in self.queue
        ):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        swa_allocatable_tokens = 0
        if uses_swa_tail_prealloc:
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
        reserved_restore_tokens = self._hicache_pending_restore_tokens()
        full_allocatable_tokens -= reserved_restore_tokens
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.queue),
            )

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = len(decode_req.req.origin_input_ids)
            prefix_match: Optional[DecodePrefixMatch] = None
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                # Match prefix against decode's radix cache.
                prefix_match = self._match_prefix_and_lock(decode_req.req)
                prefix_indices = prefix_match.prefix_indices
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len

                fill_len = origin_input_len + max(len(decode_req.req.output_ids) - 1, 0)
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                    hicache_reserved_tokens=reserved_restore_tokens,
                )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                required_alloc_tokens = origin_input_len

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    - prefix_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > full_allocatable_tokens
            ):
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break
            if required_tokens_for_request > full_allocatable_tokens:
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break

            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                if (
                    max(
                        swa_required,
                        swa_len + max_new_tokens - retractable_swa_tokens,
                    )
                    > swa_allocatable_tokens
                ):
                    if prefix_len > 0:
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    break

            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
            )
            decode_req.prefix_match = prefix_match
            if self.scheduler.enable_decode_hicache:
                self._start_hicache_prefetch(decode_req.req, prefix_match)
            hisparse_req_budget -= 1
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            if prefix_match is not None:
                reserved_restore_tokens += prefix_match.restore_token_count
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=len(preallocated_reqs) + 1,
                hicache_reserved_tokens=reserved_restore_tokens,
            )
            if uses_swa_tail_prealloc:
                # SWA budget uses simple decrement (no radix cache eviction in
                # the SWA pool, so page-rounding drift is negligible).
                swa_allocatable_tokens -= swa_required
            decode_req.req.cache_protected_len = total_prefix_len

            page_size = self.token_to_kv_pool_allocator.page_size
            kv_transfer_page_size = page_size
            if self.scheduler.enable_hisparse:
                # Direct-to-host sends host/C4 rows; keep allocator.page_size
                # logical and use the compressed page size only for these indices.
                kv_transfer_page_size = getattr(
                    self.token_to_kv_pool_allocator,
                    "hisparse_page_size",
                    page_size,
                )
                # Must cast to int32 for ZMQ serialization -- from_zmq reads np.int32.
                kv_indices = (
                    dst_kv_indices[: origin_input_len - prefix_len]
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
            else:
                # Only send delta indices (beyond prefix) to prefill.
                kv_indices = (
                    self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                        total_prefix_len:origin_input_len
                    ]
                    .cpu()
                    .numpy()
                )

            seq_len = len(decode_req.req.origin_input_ids)

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.scheduler.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = page_align_floor(window_start, page_size)
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(
                    window_kv_indices_swa.cpu().numpy(), page_size
                )

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                return kv_to_page_indices(
                    kv_indices_full.cpu().numpy(), device_page_size
                )

            def _swa_ring_payload():
                # Mirror of prefill _swa_ring_payload using this side's req_pool_idx.
                # Same window positions and order -> positional match with prefill.
                ring_stride = self.token_to_kv_pool.unified_swa_ring_size
                window_size = self.token_to_kv_pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(decode_req.req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = 1 if online else self.token_to_kv_pool.get_ring_size(128)
                return get_dsv4_c128_state_indices(
                    int(decode_req.req.req_pool_idx),
                    seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            state_types = self.kv_manager.kv_args.state_types
            state_indices: Optional[List] = []
            if StateType.C128_STATE in state_types:
                clear_c128_state = getattr(
                    self.token_to_kv_pool, "clear_c128_req_state", None
                )
                if clear_c128_state is not None:
                    clear_c128_state(int(decode_req.req.req_pool_idx))
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.MINIMAX_INDEX_K:
                    # Index rows live at the same loc as main KV on the same
                    # page_size, so reuse the full-seq page-ids.
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                elif st == StateType.C128_STATE:
                    state_indices.append(_c128_state_payload())
                else:
                    state_indices.append(None)

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size)
            try:
                transfer_page_tag_ids, transfer_page_tags = (
                    self._prepare_transfer_page_tags(
                        req=decode_req.req,
                        page_indices=page_indices,
                        page_position_start=total_prefix_len // page_size,
                        state_indices=state_indices,
                        state_types=state_types,
                        seq_len=seq_len,
                    )
                )
            except Exception as error:
                prepare_abort(
                    decode_req.req,
                    str(error),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                self._clean_hicache_prefetch_resources(decode_req)
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                self.req_to_metadata_buffer_idx_allocator.free(
                    decode_req.metadata_buffer_index
                )
                decode_req.metadata_buffer_index = -1
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)
                if prefix_match is not None:
                    reserved_restore_tokens -= prefix_match.restore_token_count
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                    hicache_reserved_tokens=reserved_restore_tokens,
                )
                if uses_swa_tail_prealloc:
                    swa_allocatable_tokens += swa_required
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            self.transfer_queue.pin_transfer_pages(decode_req, page_indices)
            metadata_kwargs = {"decode_prefix_len": total_prefix_len}
            if transfer_page_tag_ids is not None and transfer_page_tags is not None:
                metadata_kwargs["transfer_page_tag_ids"] = transfer_page_tag_ids
                metadata_kwargs["transfer_page_tags"] = transfer_page_tags
            register_staging = (
                self.transfer_queue.enable_staging
                and hasattr(decode_req.kv_receiver, "require_staging")
                and decode_req.kv_receiver.require_staging
            )
            send_metadata_with_staging_registration(
                decode_req.kv_receiver,
                (
                    page_indices,
                    decode_req.metadata_buffer_index,
                    state_indices,
                ),
                metadata_kwargs,
                staging_handler=(
                    self.transfer_queue.staging_handler if register_staging else None
                ),
                room=decode_req.req.bootstrap_room,
                decode_req=decode_req,
            )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            decode_req.req.extend_range.end for decode_req in self.transfer_queue.queue
        )

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + len(self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_reserved_tokens: int = 0,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )

        if self.scheduler.enable_hisparse:
            # HiSparse pre-alloc only allocates logical indices (alloc_logical_only),
            # so the logical pool is the binding constraint for admission control.
            available_size = (
                self.token_to_kv_pool_allocator.logical_attn_allocator.available_size()
            )
        elif self._uses_swa_tail_prealloc():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        allocatable_tokens -= hicache_reserved_tokens
        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count()
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        swa_total = self.token_to_kv_pool_allocator.size_swa
        swa_used = swa_total - self.token_to_kv_pool_allocator.swa_available_size()
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - max(swa_reserved_tokens, need_swa_space_for_single_req)
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if (
            self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            and self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens
        ):
            num_to_evict = (
                required_alloc_tokens - self.token_to_kv_pool_allocator.available_size()
            )
            result = self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))
            if self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self.token_to_kv_pool_allocator.available_size()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self.tree_cache.evictable_size()}, "
                    f"protected_size={self.tree_cache.protected_size()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        elif self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(delta_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            last_loc = (
                prefix_indices[-1:].to(dtype=torch.int64, device=device)
                if prefix_len > 0
                else torch.tensor([-1], dtype=torch.int64, device=device)
            )
            if self._uses_swa_tail_prealloc() and prefix_len == 0:
                # Tail-only SWA allocation: only valid when prefix_len == 0.
                # When prefix_len > 0 (radix cache hit), we fall back to
                # alloc_extend which allocates SWA at full page count; the
                # SWA budget in that case may slightly under-estimate.
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                    prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=fill_len,
                    swa_tail_len=self._swa_tail_len(fill_len),
                )
                req.swa_evicted_seqlen = fill_len - self._swa_tail_len(fill_len)
            else:
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                    prefix_lens=torch.tensor(
                        [total_prefix_len], dtype=torch.int64, device=device
                    ),
                    prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=delta_len,
                )

        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self.tree_cache.evictable_size()}, "
            f"protected={self.tree_cache.protected_size()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"page_size={self.token_to_kv_pool_allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so cache_unfinished_req only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        # Set prefix_indices so downstream consumers (init_next_round_input,
        # prepare_for_extend) see the correct prefix length. In the agg path
        # this is done inside init_next_round_input, but decode-disagg needs
        # allocation info before batch assembly so we set it here.
        req.prefix_indices = (
            prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
        )
        req.set_extend_range(total_prefix_len, req.kv_committed_len)

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None
        self.transfer_page_pin_manager = KVTransferPagePinManager(
            scheduler.token_to_kv_pool_allocator
        )
        self.quarantined_transfer_reqs: Dict[int, DecodeRequest] = {}
        scheduler.token_to_kv_pool_allocator.attach_transfer_page_pin_manager(
            self.transfer_page_pin_manager
        )

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if (
                    hasattr(dr.kv_receiver, "require_staging")
                    and dr.kv_receiver.require_staging
                ):
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def pin_transfer_pages(self, decode_req: DecodeRequest, page_indices) -> None:
        decode_req.transfer_pinned_page_ids = self.transfer_page_pin_manager.pin_pages(
            page_indices
        )
        decode_req.transfer_pinned_page_generations = ()
        if not decode_req.transfer_pinned_page_ids:
            return
        if not self.enable_staging or not getattr(
            decode_req.kv_receiver, "require_staging", False
        ):
            return
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        table = getattr(manager, "table", None)
        if table is None:
            return
        page_ids = torch.tensor(decode_req.transfer_pinned_page_ids, dtype=torch.long)
        decode_req.transfer_pinned_page_generations = tuple(
            int(x) for x in table.generation_of(page_ids).detach().cpu().tolist()
        )

    def _release_transfer_pins(self, decode_req: DecodeRequest) -> None:
        if not decode_req.transfer_pinned_page_ids:
            return
        self.transfer_page_pin_manager.release_pages(
            decode_req.transfer_pinned_page_ids
        )
        decode_req.transfer_pinned_page_ids = ()
        decode_req.transfer_pinned_page_generations = ()

    def _clear_receiver(self, decode_req: DecodeRequest) -> None:
        if decode_req.kv_receiver is not None:
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
        self._release_transfer_pins(decode_req)

    def _release_metadata_buffer(self, decode_req: DecodeRequest) -> None:
        if self.enable_staging and self.staging_handler.is_staging_room(
            decode_req.req.bootstrap_room
        ):
            self.staging_handler.unregister_decode_req(decode_req.req.bootstrap_room)
        idx = decode_req.metadata_buffer_index
        assert idx != -1
        self.metadata_buffers.bootstrap_room[idx] = 0
        self.req_to_metadata_buffer_idx_allocator.free(idx)
        decode_req.metadata_buffer_index = -1

    def _reap_quarantined_transfers(self) -> None:
        quarantined = getattr(self, "quarantined_transfer_reqs", {})
        for room, decode_req in list(quarantined.items()):
            if getattr(decode_req, "_staging_scatter_tracking_failed", False):
                continue
            receiver = decode_req.kv_receiver
            if not getattr(decode_req, "quarantine_receiver_cleared", False):
                if (
                    receiver is None
                    or not getattr(
                        receiver, "try_release_page_quarantine", lambda: False
                    )()
                ):
                    continue
                receiver.clear()
                decode_req.kv_receiver = None
                decode_req.quarantine_receiver_cleared = True
            if not getattr(decode_req, "quarantine_kv_released", False):
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                decode_req.quarantine_kv_released = True
            self._release_transfer_pins(decode_req)
            if decode_req.metadata_buffer_index != -1:
                self._release_metadata_buffer(decode_req)
            quarantined.pop(room, None)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest):
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Keep transfer-produced output metadata private until all integrity and
        # bookkeeping gates accept the request. Failed requests are streamed by
        # pop_transferred(), so mutating Req any earlier can publish corrupt data.
        staged_output_id = output_id[0].item()
        staged_cached_tokens = tuple(cached_tokens[i].item() for i in range(7))
        staged_logprobs = None
        if decode_req.req.return_logprob:
            top_logprobs_num = decode_req.req.logprob.top_logprobs_num
            staged_logprobs = (
                output_token_logprobs_val[0].item(),
                output_token_logprobs_idx[0].item(),
                output_top_logprobs_val[:top_logprobs_num].tolist(),
                output_top_logprobs_idx[:top_logprobs_num].tolist(),
            )

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            self._clear_receiver(decode_req)
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            self._clear_receiver(decode_req)
            return

        if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
            self._clear_receiver(decode_req)
            return

        self._commit_hicache_local_restore_to_req(decode_req)

        # Verify the KV transfer checksum (gated). Compares decode-side
        # destination KV bytes against the prefill-side source checksum that was
        # transmitted in the spare metadata slots. Aborts only this request on
        # mismatch.
        if self._verify_transfer_checksum(decode_req.req, output_bootstrap_room):
            self._clear_receiver(decode_req)
            return

        if self.enable_staging and getattr(
            decode_req.kv_receiver, "require_staging", False
        ):
            if self._write_staged_transfer_page_tags(decode_req.req):
                self._clear_receiver(decode_req)
                return

        # Register baseline KV attention tags for the transferred prompt pages
        # so a later cross-request page reuse is detectable before decode
        # attention reads them (gated; no-op when disabled).
        if self._register_kv_attention_tags(decode_req.req):
            self._clear_receiver(decode_req)
            return

        # Case 3: Success - publish the staged transfer metadata atomically with
        # respect to all expected validation failures.
        req = decode_req.req
        req.output_ids.append(staged_output_id)
        req.cached_tokens = staged_cached_tokens[0]
        # The prefill node already reported its prefix-cache hit. Seed
        # already_computed so prepare_for_prebuilt only adds decode-side reuse.
        req.already_computed = req.cached_tokens
        req.cached_tokens_device = staged_cached_tokens[1]
        req.cached_tokens_host = staged_cached_tokens[2]
        req.cached_tokens_storage = staged_cached_tokens[3]
        req.mm_image_tokens = staged_cached_tokens[4]
        req.mm_audio_tokens = staged_cached_tokens[5]
        req.mm_video_tokens = staged_cached_tokens[6]
        if not self.spec_algorithm.is_none():
            req.output_topk_p = output_topk_p
            req.output_topk_index = output_topk_index
            req.hidden_states_tensor = output_hidden_states

        if staged_logprobs is not None:
            req.logprob.output_token_logprobs_val.append(staged_logprobs[0])
            req.logprob.output_token_logprobs_idx.append(staged_logprobs[1])
            req.logprob.output_top_logprobs_val.append(staged_logprobs[2])
            req.logprob.output_top_logprobs_idx.append(staged_logprobs[3])

        self._clear_receiver(decode_req)
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _verify_transfer_checksum(self, req: Req, meta_bootstrap_room) -> bool:
        """Return True if the request was aborted due to a checksum mismatch."""
        if _is_fake_transfer(req, self.scheduler.server_args):
            return False
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        if manager is None or not manager.config.checksum_enabled:
            return False
        try:
            manifest_error = getattr(req, "_kv_transfer_manifest_error", None)
            if manifest_error is not None:
                raise ValueError(manifest_error)
            expected = getattr(req, "_kv_transfer_expected_checksum", None)
            if expected is None:
                raise ValueError("missing checksum page manifest")
            num_tokens = expected.num_tokens
            if num_tokens <= 0 or req.req_pool_idx is None:
                raise ValueError("checksum page manifest has invalid token count")
            actual = getattr(req, "_kv_transfer_actual_checksum_plan", None)
            if actual is not None:
                err = None
                if not manager.compare_destination_checksum(
                    expected, actual, rid=req.rid
                ):
                    from sglang.srt.mem_cache.kv_page_tags import (
                        KVChecksumError,
                        first_checksum_page_mismatch,
                    )

                    mismatch_page = first_checksum_page_mismatch(expected, actual)
                    digest_index = (
                        None
                        if mismatch_page is None
                        else mismatch_page
                        - expected.logical_start // max(expected.page_size, 1)
                    )

                    err = KVChecksumError(
                        rid=req.rid,
                        bootstrap_room=req.bootstrap_room or 0,
                        expected_checksum=expected.checksum,
                        actual_checksum=actual.checksum,
                        num_checked_tokens=num_tokens,
                        page_position=mismatch_page,
                        expected_page_digest=(
                            expected.page_digests[digest_index]
                            if digest_index is not None
                            and 0 <= digest_index < len(expected.page_digests)
                            else None
                        ),
                        actual_page_digest=(
                            actual.page_digests[digest_index]
                            if digest_index is not None
                            and 0 <= digest_index < len(actual.page_digests)
                            else None
                        ),
                    )
            else:
                from sglang.srt.mem_cache.kv_page_tags import KVChecksumError

                logger.error(
                    "KV transfer checksum missing batched destination result for rid=%s",
                    req.rid,
                )
                if manager.metrics is not None:
                    manager.metrics.increment_kv_transfer_checksum_mismatches()
                err = KVChecksumError(
                    rid=req.rid,
                    bootstrap_room=req.bootstrap_room or 0,
                    expected_checksum=expected.checksum,
                    actual_checksum=0,
                    num_checked_tokens=num_tokens,
                )
        except Exception as e:
            logger.error(
                "KV transfer checksum verification error for rid=%s: %s", req.rid, e
            )
            from sglang.srt.mem_cache.kv_page_tags import KVChecksumError

            if manager.metrics is not None:
                manager.metrics.increment_kv_transfer_checksum_mismatches()
            err = KVChecksumError(
                rid=req.rid,
                bootstrap_room=req.bootstrap_room or 0,
                num_checked_tokens=getattr(
                    getattr(req, "_kv_transfer_expected_checksum", None),
                    "num_tokens",
                    None,
                ),
                cause="manifest_error",
                detail=str(e),
            )
        if err is None:
            return False
        from sglang.srt.mem_cache.kv_page_tags import (
            attach_kv_protection_incident,
            emit_kv_protection_incident,
        )

        attach_kv_protection_incident(
            err,
            kind="transfer_checksum",
            phase="checksum_verify",
        )
        emit_kv_protection_incident(err, logger)
        logger.error(str(err))
        prepare_abort(
            req,
            f"KV transfer checksum mismatch: {err}",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        return True

    def _begin_destination_checksum_batch(
        self, decode_reqs: List[DecodeRequest]
    ) -> Tuple[Optional[object], List[Req]]:
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        if manager is None or not manager.config.checksum_enabled:
            return None, []
        from sglang.srt.mem_cache.kv_page_tags import (
            TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE,
            swa_checksum_evicted_len,
        )

        sliding_window = getattr(self.scheduler, "sliding_window_size", None)
        page_size = getattr(self.scheduler.token_to_kv_pool_allocator, "page_size", 1)
        eligible_reqs = [
            decode_req
            for decode_req in decode_reqs
            if decode_req.req.req_pool_idx is not None
            and not _is_fake_transfer(decode_req.req, self.scheduler.server_args)
        ]
        if not eligible_reqs:
            return None, []
        req_pool_indices = []
        bootstrap_rooms = []
        num_tokens = []
        swa_evicted = []
        reqs = []
        for decode_req in eligible_reqs:
            req = decode_req.req
            try:
                get_checksum_plan = getattr(
                    decode_req.kv_receiver, "get_checksum_plan", None
                )
                expected = (
                    get_checksum_plan() if get_checksum_plan is not None else None
                )
            except Exception as e:
                req._kv_transfer_manifest_error = str(e)
                continue
            if expected is None:
                continue
            req._kv_transfer_expected_checksum = expected
            try:
                n = expected.num_tokens
                expected_room = int(req.bootstrap_room or 0)
                actual_tokens = len(req.origin_input_ids)
                table = self.scheduler.req_to_token_pool.req_to_token
                req_pool_idx = int(req.req_pool_idx)
                if expected.bootstrap_room != expected_room:
                    raise ValueError("checksum page manifest bootstrap room mismatch")
                if expected.page_size != TRANSFER_CHECKSUM_DIGEST_PAGE_SIZE:
                    raise ValueError("checksum page manifest page size mismatch")
                if expected.logical_start != 0:
                    raise ValueError("checksum page manifest logical start mismatch")
                if n != actual_tokens:
                    raise ValueError(
                        "checksum page manifest token count mismatch: "
                        f"expected={actual_tokens}, actual={n}"
                    )
                if (
                    req_pool_idx < 0
                    or req_pool_idx >= table.shape[0]
                    or n > table.shape[1]
                ):
                    raise ValueError(
                        "checksum page manifest token range is out of bounds"
                    )
            except Exception as e:
                req._kv_transfer_manifest_error = str(e)
                continue
            reqs.append(req)
            req_pool_indices.append(req_pool_idx)
            bootstrap_rooms.append(int(req.bootstrap_room or 0))
            num_tokens.append(n)
            swa_evicted.append(swa_checksum_evicted_len(n, sliding_window, page_size))
        if not reqs:
            return None, []
        kv_pool = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
        batch = manager.begin_transfer_checksums_from_table(
            kv_pool,
            self.scheduler.req_to_token_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            bootstrap_rooms=bootstrap_rooms,
            num_tokens=num_tokens,
            swa_evicted_lens=swa_evicted,
        )
        return batch, reqs

    def _register_kv_attention_tags(self, req: Req) -> bool:
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        if manager is None or not manager.config.enable_attention_tags:
            return False
        try:
            from sglang.srt.mem_cache.kv_page_tags import (
                KV_EXPECTED_MAPPING_SWA,
                AttentionTagManifestGroup,
            )

            page_size = manager.page_size
            seq_len = len(req.origin_input_ids)
            if seq_len == 0 or req.req_pool_idx is None:
                return False
            kv_loc = self.scheduler.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ]
            # First token's loc within each logical page -> physical page id.
            first_locs = kv_loc[::page_size]
            page_physical_ids = first_locs // page_size
            full_manifest = manager.register_attention_tags(
                page_physical_ids=page_physical_ids,
                request_pool_idx=req.req_pool_idx,
                bootstrap_room=req.bootstrap_room or 0,
            )
            manifests = [full_manifest] if full_manifest is not None else []

            allocator = self.scheduler.token_to_kv_pool_allocator
            if hasattr(allocator, "translate_loc_from_full_to_swa") and hasattr(
                allocator, "attention_tag_swa_page_ids"
            ):
                swa_locs = allocator.translate_loc_from_full_to_swa(kv_loc)
                swa_first_locs = swa_locs[::page_size]
                valid_swa_pages = swa_first_locs > 0
                swa_page_ids = swa_first_locs[valid_swa_pages] // page_size
                swa_page_ids = allocator.attention_tag_swa_page_ids(swa_page_ids)
                window_size = self.scheduler.sliding_window_size or seq_len
                window_start = page_align_floor(
                    max(0, seq_len - window_size), page_size
                )
                swa_page_position_start = window_start // page_size
                swa_page_positions = range(
                    swa_page_position_start,
                    swa_page_position_start + int(swa_page_ids.numel()),
                )
                swa_manifest = manager.register_attention_tags(
                    page_physical_ids=swa_page_ids,
                    request_pool_idx=req.req_pool_idx,
                    page_positions=swa_page_positions,
                    bootstrap_room=req.bootstrap_room or 0,
                    max_num_pages=((window_size + page_size - 1) // page_size + 1),
                    mapping_namespace=KV_EXPECTED_MAPPING_SWA,
                )
                if swa_manifest is not None and swa_manifest.num_pages > 0:
                    manifests.append(swa_manifest)

            if len(manifests) > 1:
                req.kv_attention_tag_manifest = AttentionTagManifestGroup(
                    tuple(manifests)
                )
            elif manifests:
                req.kv_attention_tag_manifest = manifests[0]
            else:
                req.kv_attention_tag_manifest = None
            return False
        except Exception as e:
            logger.error(
                "KV attention tag registration failed for rid=%s: %s", req.rid, e
            )
            req.kv_attention_tag_manifest = None
            from sglang.srt.mem_cache.kv_page_tags import (
                KVProtectionBookkeepingError,
                attach_kv_protection_incident,
                emit_kv_protection_incident,
            )

            error = KVProtectionBookkeepingError(
                rid=req.rid,
                bootstrap_room=req.bootstrap_room,
                cause="attention_tag_registration",
                detail=str(e),
            )
            attach_kv_protection_incident(
                error, kind="attention_tag", phase="tag_registration"
            )
            emit_kv_protection_incident(error, logger)
            prepare_abort(
                req,
                str(error),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return True

    def _write_staged_transfer_page_tags(self, req: Req) -> bool:
        """Return True if the request was aborted due to tag write failure."""
        manager = getattr(self.scheduler, "kv_protection_manager", None)
        manifest = getattr(req, "kv_transfer_page_tag_manifest", None)
        if (
            manager is None
            or not manager.config.enable_attention_tags
            or manifest is None
        ):
            return False
        try:
            manager.commit_transfer_page_tags(manifest)
            return False
        except Exception as e:
            logger.error(
                "KV transfer page tag write failed for staged rid=%s: %s", req.rid, e
            )
            from sglang.srt.mem_cache.kv_page_tags import (
                KVProtectionBookkeepingError,
                attach_kv_protection_incident,
                emit_kv_protection_incident,
            )

            error = KVProtectionBookkeepingError(
                rid=req.rid,
                bootstrap_room=req.bootstrap_room,
                cause="transfer_page_tag_commit",
                detail=str(e),
            )
            attach_kv_protection_incident(
                error, kind="transfer_page_tag", phase="tag_commit"
            )
            emit_kv_protection_incident(error, logger)
            prepare_abort(
                req,
                str(error),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return True

    def _poll_with_metadata_gate(self) -> List[int]:
        pollers = (
            [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
            if self.scheduler.enable_decode_hicache
            else [dr.kv_receiver for dr in self.queue]
        )
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        self._reap_quarantined_transfers()
        if not self.queue:
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        checksum_batch = None
        checksum_reqs: List[Req] = []
        checksum_finalized = False
        checksum_success_reqs = [
            decode_req
            for decode_req, poll in zip(self.queue, polls)
            if poll == KVPoll.Success
            and (
                not self.scheduler.enable_decode_hicache
                or decode_req.hicache_restore_status != HiCacheRestoreResult.PENDING
            )
        ]
        if checksum_success_reqs:
            try:
                checksum_batch, checksum_reqs = self._begin_destination_checksum_batch(
                    checksum_success_reqs
                )
            except Exception as e:
                logger.error(
                    "KV transfer checksum destination batch launch failed: %s", e
                )
                checksum_batch = None
                checksum_reqs = []

        def finalize_checksum_batch_once() -> None:
            nonlocal checksum_finalized
            if checksum_finalized:
                return
            checksum_finalized = True
            if checksum_batch is None:
                return
            try:
                for req, plan in zip(
                    checksum_reqs, checksum_batch.finalize(), strict=True
                ):
                    req._kv_transfer_actual_checksum_plan = plan
            except Exception as e:
                logger.error(
                    "KV transfer checksum destination batch finalize failed: %s", e
                )

        transferred_reqs = []
        indices_to_remove = set()
        quarantined_indices = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            hicache_restore_status = decode_req.hicache_restore_status
            if (
                poll == KVPoll.Failed
                or hicache_restore_status == HiCacheRestoreResult.FAILED
            ):
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                receiver = decode_req.kv_receiver
                if poll == KVPoll.Failed:
                    try:
                        receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                        is_propagated = getattr(e, "is_from_another_rank", False)
                quarantine = bool(
                    receiver is not None
                    and getattr(receiver, "requires_page_quarantine", lambda: False)()
                ) or getattr(decode_req, "_staging_scatter_tracking_failed", False)
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                if quarantine:
                    room = decode_req.req.bootstrap_room
                    self.quarantined_transfer_reqs[room] = decode_req
                    quarantined_indices.add(i)
                else:
                    # Release failed pre-allocation only after producer quiescence.
                    release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                    self._clear_receiver(decode_req)
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                finalize_checksum_batch_once()
                self._commit_transfer_to_req(decode_req)
                if hasattr(decode_req.req, "_kv_transfer_actual_checksum_plan"):
                    delattr(decode_req.req, "_kv_transfer_actual_checksum_plan")
                if hasattr(decode_req.req, "_kv_transfer_expected_checksum"):
                    delattr(decode_req.req, "_kv_transfer_expected_checksum")
                if hasattr(decode_req.req, "_kv_transfer_manifest_error"):
                    delattr(decode_req.req, "_kv_transfer_manifest_error")
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    if self.scheduler.enable_hisparse:
                        self.scheduler.hisparse_coordinator.request_finished(
                            decode_req.req
                        )
                    self._clean_hicache_prefetch_resources(decode_req)
                    release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                else:
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if i in quarantined_indices:
                continue
            self._release_metadata_buffer(self.queue[i])

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory."""
        if getattr(self, "quarantined_transfer_reqs", {}):
            raise RuntimeError(
                "Cannot release GPU memory while protected transfer pages are quarantined"
            )
        for decode_req in self.queue:
            self._release_transfer_pins(decode_req)
        self.queue.clear()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted."""
        pass


class SchedulerDisaggregationDecodeMixin:
    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self._run_batch_with_war_barrier(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, self.running_batch)

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode-radix path: new requests already matched in
                # `pop_preallocated`. Retracted requests reset `last_node`,
                # so re-match only when that state is missing.
                if self.server_args.disaggregation_decode_enable_radix_cache:
                    tree_cache = self.tree_cache if req.last_node is None else None
                else:
                    tree_cache = self.tree_cache
                req.init_next_round_input(tree_cache)
                # Truncate fill_len to kv_committed_len so cache_unfinished_req
                # only sees committed KV (full array includes one uncommitted
                # token because init_next_round_input rebuilt it as full).
                if req.kv_committed_len is not None:
                    req.set_extend_range(len(req.prefix_indices), req.kv_committed_len)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()

        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        self.disagg_decode_transfer_queue._reap_quarantined_transfers()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(transferred_reqs)
