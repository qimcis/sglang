"""External MTP first, resident native MTP second, target AR last.

This worker deliberately retains the native NextN model and draft KV on the
target GPU.  It is therefore a separate benchmark policy from REMOTE_MTP,
whose fallback is memory-maximizing AR and which never allocates this worker.
Both remote and local proposals enter the same native target verifier.
"""

from __future__ import annotations

import logging
import time
from copy import copy

import torch
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
)
from sglang.srt.speculative.remote_mtp_io import (
    RemoteMTPBatchClaim,
    RemoteMTPRequestView,
    remote_mtp_linear_chain_layout,
)
from sglang.srt.speculative.remote_mtp_worker_v2 import RemoteMTPWorkerV2
from sglang.srt.speculative.spec_utils import spec_stage_span

logger = logging.getLogger(__name__)


class RemoteMTPLocalWorkerV2(RemoteMTPWorkerV2):
    """Use an exact remote claim when ready, else run resident local MTP."""

    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        if not self.speculative_algorithm.has_local_mtp_fallback():
            raise ValueError("RemoteMTPLocalWorkerV2 requires REMOTE_MTP_LOCAL")
        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )
        self.adaptive_controller = None
        self.remote_local_fallback_batches = 0
        self.remote_local_fallback_requests = 0
        # Compatibility alias: this counts proposal rows only. State extension
        # is separate and remains mandatory for remote-ready rows so resident
        # local-MTP fallback is current at the next service window.
        self.remote_local_draft_rows = 0
        self.local_proposal_draft_rows = 0
        self.local_state_extend_rows = 0
        self.remote_mtp_max_depth = self.speculative_num_steps
        self._remote_mtp_depth_buffers = {}
        original_steps = self._draft_worker.speculative_num_steps
        original_tokens = self._draft_worker.speculative_num_draft_tokens
        try:
            for depth in range(1, self.remote_mtp_max_depth + 1):
                self._draft_worker.speculative_num_steps = depth
                self._draft_worker.speculative_num_draft_tokens = depth + 1
                self._draft_worker._rebuild_topk1_chain_buffers()
                self._remote_mtp_depth_buffers[depth] = (
                    self._draft_worker._topk1_parents_prealloc,
                    self._draft_worker._topk1_score_indices_prealloc,
                )
        finally:
            self._draft_worker.speculative_num_steps = original_steps
            self._draft_worker.speculative_num_draft_tokens = original_tokens
            (
                self._draft_worker._topk1_parents_prealloc,
                self._draft_worker._topk1_score_indices_prealloc,
            ) = self._remote_mtp_depth_buffers[original_steps]

    @property
    def war_fastpath_runner(self):
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def verify(self, batch):
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=1,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
        )

    @staticmethod
    def _set_backend_depth(backend, *, depth: int, width: int, seen=None) -> None:
        if backend is None:
            return
        if seen is None:
            seen = set()
        identity = id(backend)
        if identity in seen:
            return
        seen.add(identity)
        if hasattr(backend, "speculative_num_steps"):
            backend.speculative_num_steps = depth
        if hasattr(backend, "speculative_num_draft_tokens"):
            backend.speculative_num_draft_tokens = width
        for child in tuple(getattr(backend, "attn_backends", ()) or ()):
            RemoteMTPLocalWorkerV2._set_backend_depth(
                child,
                depth=depth,
                width=width,
                seen=seen,
            )

    def _has_shape_bound_cuda_graphs(self) -> bool:
        return any(
            runner is not None
            for runner in (
                getattr(self._draft_worker, "cuda_graph_runner", None),
                getattr(
                    self._draft_worker,
                    "cuda_graph_runner_for_draft_extend",
                    None,
                ),
                getattr(
                    self.target_worker.model_runner,
                    "decode_cuda_graph_runner",
                    None,
                ),
            )
        )

    def _apply_exact_batch_depth(self, depth: int) -> None:
        if depth not in self._remote_mtp_depth_buffers:
            raise ValueError("REMOTE_MTP selected an unqualified target depth")
        width = depth + 1
        self.speculative_num_steps = depth
        self.speculative_num_draft_tokens = width
        self._draft_worker.speculative_num_steps = depth
        self._draft_worker.speculative_num_draft_tokens = width
        (
            self._draft_worker._topk1_parents_prealloc,
            self._draft_worker._topk1_score_indices_prealloc,
        ) = self._remote_mtp_depth_buffers[depth]
        seen = set()
        for backend in (
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend,
            self._draft_worker.draft_runner.attn_backend,
            self.target_worker.model_runner.attn_backend,
        ):
            self._set_backend_depth(
                backend,
                depth=depth,
                width=width,
                seen=seen,
            )

    def _activate_batch_depth(self, batch) -> str | None:
        selected_depth = getattr(batch, "remote_mtp_selected_depth", None)
        if selected_depth is None:
            selected_depth = self.remote_mtp_max_depth
        if (
            isinstance(selected_depth, bool)
            or not isinstance(selected_depth, int)
            or not 1 <= selected_depth <= self.remote_mtp_max_depth
        ):
            self._apply_exact_batch_depth(self.remote_mtp_max_depth)
            return "selected_depth_invalid"
        if (
            selected_depth != self.remote_mtp_max_depth
            and self._has_shape_bound_cuda_graphs()
        ):
            # Never replay a graph captured for another width.  This campaign
            # qualifies adaptive exact-depth execution in eager mode only.
            self._apply_exact_batch_depth(self.remote_mtp_max_depth)
            return "selected_depth_requires_eager_execution"
        self._apply_exact_batch_depth(selected_depth)
        return None

    def _publish_local_prefill(
        self,
        batch,
        batch_output,
        *,
        prefill_input_ids,
        request_views,
        started_monotonic_ns,
    ):
        batch_output.remote_mtp_claim = None
        batch_output.remote_mtp_executed_source = "autoregressive"
        batch_output.remote_mtp_fallback_reason = "prefill"
        batch_output.remote_mtp_request_views = request_views
        batch_output.remote_mtp_selected_depth = None
        batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
            batch=batch,
            batch_output=batch_output,
            phase="prefill",
            claim=None,
            fallback_reason="prefill",
            row_stride=1,
            prefill_input_ids=prefill_input_ids,
            request_views=request_views,
        )
        self._record_target_execution(
            batch=batch,
            phase="prefill",
            executed_source="autoregressive",
            verification_tokens=max(1, sum(batch.extend_lens or ())),
            started_monotonic_ns=started_monotonic_ns,
        )
        return batch_output

    def _publish_local_verify(
        self,
        batch,
        batch_output,
        *,
        fallback_reason: str,
        request_views,
        started_monotonic_ns: int,
    ):
        batch_output.remote_mtp_claim = None
        batch_output.remote_mtp_executed_source = "local_mtp"
        batch_output.remote_mtp_fallback_reason = fallback_reason
        batch_output.remote_mtp_request_views = request_views
        batch_output.remote_mtp_selected_depth = self.speculative_num_steps
        batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
            batch=batch,
            batch_output=batch_output,
            phase="verify",
            claim=None,
            fallback_reason=fallback_reason,
            row_stride=self.speculative_num_draft_tokens,
            request_views=request_views,
        )
        self.remote_local_fallback_batches += 1
        self.remote_local_fallback_requests += len(batch.reqs)
        self.remote_local_draft_rows += len(batch.reqs)
        self.local_proposal_draft_rows += len(batch.reqs)
        self.local_state_extend_rows += len(batch.reqs)
        self._record_target_execution(
            batch=batch,
            phase="verify",
            executed_source="local_mtp",
            verification_tokens=(len(batch.reqs) * self.speculative_num_draft_tokens),
            started_monotonic_ns=started_monotonic_ns,
            selected_depth=self.speculative_num_steps,
            local_proposal_draft_rows=len(batch.reqs),
            local_state_extend_rows=len(batch.reqs),
        )
        return batch_output

    def _try_remote_row_claim(
        self,
        batch,
        request_views: tuple[RemoteMTPRequestView, ...],
    ) -> tuple[RemoteMTPBatchClaim | None, tuple[int, ...], str | None]:
        """Atomically claim the planned remote subset without blocking.

        A missing per-row plan preserves the legacy all-remote claim contract.
        The scheduler only installs a mixed plan for REMOTE_MTP_LOCAL.
        """

        candidate_ids_by_row = batch.remote_mtp_candidate_ids_by_row
        if candidate_ids_by_row is None:
            verify_input, claim, fallback_reason = self._try_remote_verify_input(batch)
            # The all-remote fast path rebuilds the verify input below.  Avoid
            # retaining a second GPU tree allocation beyond this boundary.
            del verify_input
            return (
                claim,
                tuple(range(len(request_views))) if claim is not None else (),
                fallback_reason,
            )
        if len(candidate_ids_by_row) != len(request_views):
            self.remote_invalid_claims += 1
            logger.error("REMOTE_MTP mixed plan lost its native row mapping")
            return None, (), "candidate_row_mapping_invalid"

        remote_indices = tuple(
            index
            for index, candidate_id in enumerate(candidate_ids_by_row)
            if candidate_id is not None
        )
        if not remote_indices:
            return None, (), "live_policy_selected_local_frontier"
        remote_requests = tuple(request_views[index] for index in remote_indices)
        candidate_ids = tuple(
            candidate_id
            for candidate_id in candidate_ids_by_row
            if candidate_id is not None
        )
        try:
            claim = self.candidate_source.try_claim_batch(
                remote_requests,
                depth=self.speculative_num_steps,
                candidate_ids=candidate_ids,
                plan_id=batch.remote_mtp_target_plan_id,
                snapshot_generation=batch.remote_mtp_target_plan_generation,
                window_id=batch.remote_mtp_target_window_id,
            )
        except Exception:
            self.remote_invalid_claims += 1
            logger.exception(
                "REMOTE_MTP candidate source failed; using local MTP fallback"
            )
            return None, (), "candidate_source_error"
        if claim is None:
            take_reason = getattr(
                self.candidate_source,
                "take_last_claim_failure_reason",
                None,
            )
            reason = take_reason() if callable(take_reason) else None
            return None, (), reason or "candidate_batch_absent"
        try:
            claim.validate()
            if claim.requests != remote_requests:
                raise ValueError("remote MTP claim changed the planned row identities")
            if claim.depth != self.speculative_num_steps:
                raise ValueError("remote MTP claim depth differs from local MTP")
            vocab_size = int(self.target_worker.model_config.vocab_size)
            if any(
                token_id >= vocab_size
                for candidate in claim.token_ids
                for token_id in candidate
            ):
                raise ValueError("remote MTP claim contains an out-of-vocabulary token")
        except Exception:
            self.remote_invalid_claims += 1
            logger.exception(
                "REMOTE_MTP candidate claim was invalid; using local MTP fallback"
            )
            return None, (), "candidate_batch_invalid"
        return claim, remote_indices, None

    @staticmethod
    def _slice_optional_rows(value, row_index, *, batch_rows: int):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_rows:
            return value[row_index]
        return value

    def _build_local_draft_batch(
        self,
        batch,
        local_indices: tuple[int, ...],
    ):
        """Own a local-only view without mutating target or peer draft state."""

        batch_rows = len(batch.reqs)
        row_index = torch.tensor(
            local_indices,
            dtype=torch.long,
            device=batch.seq_lens.device,
        )
        local_batch = copy(batch)
        local_batch.reqs = [batch.reqs[index] for index in local_indices]
        local_batch.has_grammar = any(req.grammar for req in local_batch.reqs)
        local_batch.return_logprob = any(req.return_logprob for req in local_batch.reqs)
        for field_name in (
            "req_pool_indices",
            "seq_lens",
            "orig_seq_lens",
            "input_ids",
            "input_embeds",
            "mamba_track_indices",
            "mamba_track_mask",
            "mamba_track_seqlens",
            "encoder_lens",
        ):
            setattr(
                local_batch,
                field_name,
                self._slice_optional_rows(
                    getattr(batch, field_name),
                    row_index,
                    batch_rows=batch_rows,
                ),
            )
        local_batch.req_pool_indices_cpu = (
            batch.req_pool_indices_cpu[list(local_indices)]
            if batch.req_pool_indices_cpu is not None
            else None
        )
        local_batch.seq_lens_cpu = (
            batch.seq_lens_cpu[list(local_indices)]
            if batch.seq_lens_cpu is not None
            else None
        )
        local_batch.seq_lens_sum = (
            int(local_batch.seq_lens_cpu.sum())
            if local_batch.seq_lens_cpu is not None
            else None
        )
        local_batch.out_cache_loc = None
        local_batch.out_cache_loc_dsv4 = None
        local_batch.multimodal_inputs = (
            [batch.multimodal_inputs[index] for index in local_indices]
            if batch.multimodal_inputs is not None
            else None
        )
        local_batch.top_logprobs_nums = (
            [batch.top_logprobs_nums[index] for index in local_indices]
            if batch.top_logprobs_nums is not None
            else None
        )
        local_batch.token_ids_logprobs = (
            [batch.token_ids_logprobs[index] for index in local_indices]
            if batch.token_ids_logprobs is not None
            else None
        )

        local_batch.spec_info = copy(batch.spec_info)
        local_batch.spec_info.filter_batch(
            new_indices=row_index,
            has_been_filtered=False,
            new_indices_cpu=list(local_indices),
        )
        local_batch.sampling_info = copy(batch.sampling_info)
        for field_name in (
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
            "logit_bias",
            "rids_int",
            "bootstrap_room_ids_int",
        ):
            value = getattr(batch.sampling_info, field_name, None)
            setattr(
                local_batch.sampling_info,
                field_name,
                self._slice_optional_rows(
                    value,
                    row_index,
                    batch_rows=batch_rows,
                ),
            )
        local_batch.sampling_info.grammars = (
            [batch.sampling_info.grammars[index] for index in local_indices]
            if batch.sampling_info.grammars is not None
            else None
        )
        local_batch.sampling_info.return_sampling_masks = (
            [
                batch.sampling_info.return_sampling_masks[index]
                for index in local_indices
            ]
            if batch.sampling_info.return_sampling_masks is not None
            else None
        )
        # REMOTE_MTP_LOCAL forbids rejection sampling, top-k branching, and
        # custom draft sampling.  The copied sampling object is therefore
        # read-only during draft and its target-only penalizer may remain shared.
        return local_batch

    def _build_mixed_verify_input(
        self,
        batch,
        draft_input,
        local_verify_input,
        claim: RemoteMTPBatchClaim,
        remote_indices: tuple[int, ...],
    ):
        """Scatter local/remote candidate rows into one native verify tree."""

        batch_rows = len(batch.reqs)
        width = self.speculative_num_draft_tokens
        remote_set = set(remote_indices)
        local_indices = tuple(
            index for index in range(batch_rows) if index not in remote_set
        )
        if local_verify_input.draft_token.numel() != len(local_indices) * width:
            raise ValueError("local MTP verify input has an unexpected row width")
        remote_rows = torch.tensor(
            claim.token_ids,
            dtype=local_verify_input.draft_token.dtype,
            device=local_verify_input.draft_token.device,
        )
        expected_shape = (len(remote_indices), self.speculative_num_steps)
        if tuple(remote_rows.shape) != expected_shape:
            raise ValueError("remote MTP candidate matrix has an unexpected shape")
        local_rows = local_verify_input.draft_token.reshape(len(local_indices), width)[
            :, 1:
        ]
        candidate_rows = torch.empty(
            (batch_rows, self.speculative_num_steps),
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        remote_index = torch.tensor(
            remote_indices,
            dtype=torch.long,
            device=candidate_rows.device,
        )
        local_index = torch.tensor(
            local_indices,
            dtype=torch.long,
            device=candidate_rows.device,
        )
        candidate_rows[remote_index] = remote_rows
        candidate_rows[local_index] = local_rows

        parents, selected_indices = remote_mtp_linear_chain_layout(
            self.speculative_num_steps
        )
        parent_list = torch.tensor(
            parents,
            dtype=torch.long,
            device=candidate_rows.device,
        ).repeat(batch_rows, 1)
        top_scores_index = torch.tensor(
            selected_indices,
            dtype=torch.long,
            device=candidate_rows.device,
        ).repeat(batch_rows, 1)
        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            candidate_rows,
            None,
            target_worker=self.target_worker,
            topk=1,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def _publish_mixed_verify(
        self,
        batch,
        batch_output,
        *,
        claim: RemoteMTPBatchClaim,
        remote_indices: tuple[int, ...],
        request_views: tuple[RemoteMTPRequestView, ...],
        started_monotonic_ns: int,
    ):
        remote_set = set(remote_indices)
        row_sources = tuple(
            "remote_mtp" if index in remote_set else "local_mtp"
            for index in range(len(request_views))
        )
        row_fallback_reasons = tuple(
            None if index in remote_set else "live_policy_selected_local_frontier"
            for index in range(len(request_views))
        )
        batch_output.remote_mtp_claim = claim
        batch_output.remote_mtp_claim_row_indices = remote_indices
        batch_output.remote_mtp_executed_source = "mixed_mtp"
        batch_output.remote_mtp_fallback_reason = "live_policy_selected_local_frontier"
        batch_output.remote_mtp_row_sources = row_sources
        batch_output.remote_mtp_row_fallback_reasons = row_fallback_reasons
        batch_output.remote_mtp_request_views = request_views
        batch_output.remote_mtp_selected_depth = self.speculative_num_steps
        # A partial claim cannot be attached to the full-row feature batch:
        # the feature ABI deliberately requires exact request coverage.
        batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
            batch=batch,
            batch_output=batch_output,
            phase="verify",
            claim=None,
            fallback_reason="mixed_remote_local_mtp",
            row_stride=self.speculative_num_draft_tokens,
            request_views=request_views,
        )
        local_rows = len(request_views) - len(remote_indices)
        self.remote_claimed_batches += 1
        self.remote_claimed_requests += len(remote_indices)
        self.remote_local_fallback_batches += 1
        self.remote_local_fallback_requests += local_rows
        self._record_target_execution(
            batch=batch,
            phase="verify",
            executed_source="mixed_mtp",
            verification_tokens=(
                len(request_views) * self.speculative_num_draft_tokens
            ),
            started_monotonic_ns=started_monotonic_ns,
            row_sources=row_sources,
            selected_depth=self.speculative_num_steps,
            local_proposal_draft_rows=local_rows,
            local_state_extend_rows=len(request_views),
        )
        return batch_output

    def forward_batch_generation(self, batch, on_publish=None):
        execution_started_ns = time.monotonic_ns()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._apply_exact_batch_depth(self.remote_mtp_max_depth)
            # EAGLE's local prefill rotates ``batch.input_ids`` in preparation
            # for the resident NextN extend (prompt[1:] + target root).  The
            # remote state transaction still needs the authoritative, unrotated
            # prompt to derive the exact target-prefix digest.  Keep the original
            # tensor reference; EAGLE rebinds rather than mutating it in place.
            prefill_input_ids = batch.input_ids
            request_views = self._request_views(
                batch,
                use_decode_kv_boundary=False,
            )
            batch_output = EAGLEWorkerV2.forward_batch_generation(
                self,
                batch,
                on_publish=on_publish,
            )
            return self._publish_local_prefill(
                batch,
                batch_output,
                prefill_input_ids=prefill_input_ids,
                request_views=request_views,
                started_monotonic_ns=execution_started_ns,
            )

        request_views = self._request_views(batch)
        depth_fallback_reason = self._activate_batch_depth(batch)
        if depth_fallback_reason is None:
            claim, remote_indices, fallback_reason = self._try_remote_row_claim(
                batch,
                request_views,
            )
        else:
            claim, remote_indices, fallback_reason = (
                None,
                (),
                depth_fallback_reason,
            )
        if claim is None:
            batch_output = EAGLEWorkerV2.forward_batch_generation(
                self,
                batch,
                on_publish=on_publish,
            )
            return self._publish_local_verify(
                batch,
                batch_output,
                fallback_reason=fallback_reason or "candidate_batch_absent",
                request_views=request_views,
                started_monotonic_ns=execution_started_ns,
            )

        if len(remote_indices) == len(request_views):
            verify_input = self._build_remote_verify_input(
                batch,
                batch.spec_info,
                claim,
            )
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])
            remote_set = set(remote_indices)
            local_indices = tuple(
                index for index in range(len(request_views)) if index not in remote_set
            )
            local_batch = self._build_local_draft_batch(batch, local_indices)
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft"),
            ):
                local_verify_input = self._draft_worker.draft(local_batch)
            local_rows = len(local_indices)
            self.remote_local_draft_rows += local_rows
            self.local_proposal_draft_rows += local_rows
            verify_input = self._build_mixed_verify_input(
                batch,
                batch.spec_info,
                local_verify_input,
                claim,
                remote_indices,
            )
        batch.spec_info = verify_input
        with spec_stage_span(
            "remote_mtp_mixed_target_verify"
            if len(remote_indices) != len(request_views)
            else "remote_mtp_target_verify"
        ):
            batch_output = self.verify(batch)
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        with (
            self._draft_worker.draft_tp_context(
                self._draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            self._draft_worker._draft_extend_for_decode(batch, batch_output)
        self.local_state_extend_rows += len(request_views)

        if len(remote_indices) != len(request_views):
            return self._publish_mixed_verify(
                batch,
                batch_output,
                claim=claim,
                remote_indices=remote_indices,
                request_views=request_views,
                started_monotonic_ns=execution_started_ns,
            )

        self.remote_claimed_batches += 1
        self.remote_claimed_requests += len(batch.reqs)
        batch_output.remote_mtp_claim = claim
        batch_output.remote_mtp_executed_source = "remote_mtp"
        batch_output.remote_mtp_fallback_reason = None
        batch_output.remote_mtp_request_views = request_views
        batch_output.remote_mtp_selected_depth = self.speculative_num_steps
        batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
            batch=batch,
            batch_output=batch_output,
            phase="verify",
            claim=claim,
            fallback_reason=None,
            row_stride=self.speculative_num_draft_tokens,
            request_views=request_views,
        )
        self._record_target_execution(
            batch=batch,
            phase="verify",
            executed_source="remote_mtp",
            verification_tokens=(len(batch.reqs) * self.speculative_num_draft_tokens),
            started_monotonic_ns=execution_started_ns,
            selected_depth=self.speculative_num_steps,
            local_proposal_draft_rows=0,
            local_state_extend_rows=len(request_views),
        )
        return batch_output


__all__ = ["RemoteMTPLocalWorkerV2"]
