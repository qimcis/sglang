"""Target-only speculative worker for externally produced MTP chains.

The worker never owns a local draft model and never waits for remote work.  At
each decode seal it atomically claims a complete fixed-depth candidate batch;
when that exact batch is unavailable or invalid, it immediately executes the
native one-root EAGLE verify path, which is functionally autoregressive decode.
"""

from __future__ import annotations

import logging
import time

import torch
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.remote_mtp_io import (
    RemoteMTPBatchClaim,
    RemoteMTPRequestView,
    RemoteMTPTargetFeatureBatch,
    RemoteMTPTargetSettlement,
    create_remote_mtp_candidate_source,
    remote_mtp_linear_chain_layout,
    remote_mtp_output_token_count,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import get_plan_stream, spec_stage_span

logger = logging.getLogger(__name__)


class RemoteMTPWorkerV2(BaseSpecWorker):
    """Native target verifier with a failure-independent external MTP source."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del nccl_port  # No local draft process/group is created.
        self.server_args = server_args
        self.ps = ps
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self._draft_worker = None
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        if not self.speculative_algorithm.is_remote_mtp():
            raise ValueError(
                "RemoteMTPWorkerV2 requires speculative_algorithm=REMOTE_MTP"
            )
        if self.speculative_num_draft_tokens != self.speculative_num_steps + 1:
            raise ValueError(
                "REMOTE_MTP linear verification requires num_draft_tokens == "
                "num_steps + 1"
            )

        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
        self.candidate_source = create_remote_mtp_candidate_source(server_args, gpu_id)
        self._remote_request_incarnations: dict[str, str] = {}

        self.remote_claimed_batches = 0
        self.remote_claimed_requests = 0
        self.remote_fallback_batches = 0
        self.remote_invalid_claims = 0
        self.remote_outcome_batches_published = 0
        self.remote_outcome_batches_dropped = 0
        self.remote_feature_batches_published = 0
        self.remote_feature_batches_dropped = 0
        self.remote_request_finishes_published = 0
        self.remote_request_finishes_dropped = 0

    @property
    def war_fastpath_runner(self):
        return self._target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self._target_worker.model_runner.attn_backend,)

    def _request_views(
        self,
        batch: ScheduleBatch,
        *,
        use_decode_kv_boundary: bool = True,
    ) -> tuple[RemoteMTPRequestView, ...]:
        views = []
        for req in batch.reqs:
            prompt_token_count = len(req.origin_input_ids)
            output_token_count = remote_mtp_output_token_count(
                prompt_token_count=prompt_token_count,
                req_output_token_count=len(req.output_ids),
                kv_boundary=(
                    int(req.kv_committed_len)
                    if (
                        use_decode_kv_boundary
                        and not self.server_args.disable_overlap_schedule
                    )
                    else None
                ),
            )
            view = RemoteMTPRequestView(
                request_id=req.rid,
                request_incarnation=req.remote_mtp_incarnation,
                prompt_token_count=prompt_token_count,
                output_token_count=output_token_count,
                last_target_service_monotonic_ns=(
                    req.remote_mtp_last_target_service_ns
                ),
                max_service_gap_ns=req.remote_mtp_max_service_gap_ns,
                service_completion_guard_ns=(
                    req.remote_mtp_service_completion_guard_ns
                ),
                service_period_ns=req.remote_mtp_service_period_ns,
            ).validate()
            self._remote_request_incarnations[req.rid] = req.remote_mtp_incarnation
            views.append(view)
        return tuple(views)

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        """Release external draft state after SGLang commits request completion."""

        request_incarnation = self._remote_request_incarnations.pop(rid, None)
        if request_incarnation is None:
            return
        try:
            offer = getattr(self.candidate_source, "offer_request_finished", None)
            if offer is None or not bool(
                offer(
                    request_id=rid,
                    request_incarnation=request_incarnation,
                    natural_stop=natural_stop,
                )
            ):
                self.remote_request_finishes_dropped += 1
                return
        except Exception:
            self.remote_request_finishes_dropped += 1
            logger.exception("REMOTE_MTP request-finish publication failed")
            return
        self.remote_request_finishes_published += 1

    def build_relay_input_from_bonus_tokens(
        self,
        bonus_tokens: torch.Tensor,
    ) -> EagleDraftInput:
        """Create shape-valid FutureMap fields without a local draft state."""

        bs = bonus_tokens.shape[0]
        return EagleDraftInput(
            # The target sampler may return int32 tokens on DSV4, while the
            # speculative verifier ABI requires int64 candidates.
            bonus_tokens=bonus_tokens.to(torch.int64),
            topk_p=torch.zeros(
                (bs, 1), dtype=torch.float32, device=bonus_tokens.device
            ),
            topk_index=torch.zeros(
                (bs, 1), dtype=torch.int64, device=bonus_tokens.device
            ),
            hidden_states=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    # Kept as a private alias for the prefill path; the scheduler uses the
    # public method only when rebuilding a non-overlap detached decode batch.
    _relay_input = build_relay_input_from_bonus_tokens

    def _build_remote_verify_input(
        self,
        batch: ScheduleBatch,
        draft_input: EagleDraftInput,
        claim: RemoteMTPBatchClaim,
    ) -> EagleVerifyInput:
        claim.validate()
        expected_requests = self._request_views(batch)
        if claim.requests != expected_requests:
            raise ValueError("remote MTP claim does not cover the current batch")
        if claim.depth != self.speculative_num_steps:
            raise ValueError(
                "remote MTP claim depth does not match the configured verify depth"
            )

        vocab_size = int(self.target_worker.model_config.vocab_size)
        if any(
            token_id >= vocab_size
            for candidate in claim.token_ids
            for token_id in candidate
        ):
            raise ValueError("remote MTP claim contains an out-of-vocabulary token")

        bs = len(claim.requests)
        depth = claim.depth
        draft_tokens = torch.tensor(
            claim.token_ids,
            dtype=torch.long,
            device=self.device,
        )
        # EAGLE's topk=1 tree representation for K remote candidates.  A
        # single-token chain has no parent table; K>1 uses [-1, 0, ..., K-2].
        parents, selected_indices = remote_mtp_linear_chain_layout(depth)
        parent_list = torch.tensor(
            parents,
            dtype=torch.long,
            device=self.device,
        ).repeat(bs, 1)
        top_scores_index = torch.tensor(
            selected_indices,
            dtype=torch.long,
            device=self.device,
        ).repeat(bs, 1)

        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            None,
            target_worker=self.target_worker,
            topk=1,
            num_steps=depth,
            num_draft_tokens=depth + 1,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build the one-root native verifier input used for AR fallback."""

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=1,
                spec_steps=0,
                num_verify_tokens=1,
                device=self.device,
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        mask_buf, position_buf = attn_backend.get_verify_buffers_to_fill_after_draft()
        if mask_buf is not None:
            custom_mask = mask_buf
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(
                seq_lens_sum + bs,
                dtype=torch.bool,
                device=device,
            )

        if position_buf is not None:
            positions = position_buf
            positions[:bs].copy_(batch.seq_lens)
        else:
            positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens.to(torch.int64),
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=1,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _try_remote_verify_input(
        self,
        batch: ScheduleBatch,
    ) -> tuple[EagleVerifyInput | None, RemoteMTPBatchClaim | None, str | None]:
        if batch.forward_mode.is_idle():
            return None, None, "idle"
        requests = self._request_views(batch)
        try:
            claim = self.candidate_source.try_claim_batch(
                requests,
                depth=self.speculative_num_steps,
                candidate_ids=batch.remote_mtp_candidate_ids,
                plan_id=batch.remote_mtp_target_plan_id,
                snapshot_generation=batch.remote_mtp_target_plan_generation,
                window_id=batch.remote_mtp_target_window_id,
            )
        except Exception:
            # Candidate ingress is advisory.  A broken adapter cannot make the
            # target unavailable; report and take local progress immediately.
            self.remote_invalid_claims += 1
            logger.exception("REMOTE_MTP candidate source failed; using AR fallback")
            return None, None, "candidate_source_error"
        if claim is None:
            take_reason = getattr(
                self.candidate_source,
                "take_last_claim_failure_reason",
                None,
            )
            reason = take_reason() if callable(take_reason) else None
            return None, None, reason or "candidate_batch_absent"
        try:
            return (
                self._build_remote_verify_input(batch, batch.spec_info, claim),
                claim,
                None,
            )
        except Exception:
            self.remote_invalid_claims += 1
            logger.exception(
                "REMOTE_MTP candidate claim was invalid; using AR fallback"
            )
            return None, None, "candidate_batch_invalid"

    def _offer_target_feature_batch(
        self,
        *,
        batch: ScheduleBatch,
        batch_output,
        phase: str,
        claim: RemoteMTPBatchClaim | None,
        fallback_reason: str | None,
        row_stride: int,
        prefill_input_ids: torch.Tensor | None = None,
        request_views: tuple[RemoteMTPRequestView, ...] | None = None,
    ) -> str | None:
        offer = getattr(self.candidate_source, "offer_target_feature_batch", None)
        hidden_states = getattr(batch_output.logits_output, "hidden_states", None)
        if offer is None or hidden_states is None:
            self.remote_feature_batches_dropped += 1
            return None
        try:
            feature_batch_id = self._target_feature_batch_id(batch)

            # SGLang reuses graph/output buffers.  Own stream-ordered device
            # snapshots before returning from the forward; the background
            # exporter waits/copies them without synchronizing this thread.
            def snapshot(value):
                return value.detach().clone() if torch.is_tensor(value) else value

            hidden_snapshot = snapshot(hidden_states)
            token_snapshot = snapshot(batch_output.next_token_ids)
            accept_snapshot = (
                snapshot(batch_output.accept_lens) if phase == "verify" else None
            )
            seq_snapshot = snapshot(batch_output.new_seq_lens)
            input_snapshot = (
                snapshot(
                    batch.input_ids if prefill_input_ids is None else prefill_input_ids
                )
                if phase == "prefill"
                else None
            )
            completion_event = None
            if torch.cuda.is_available() and str(self.device) != "cpu":
                completion_event = torch.cuda.Event()
                completion_event.record(torch.cuda.current_stream(self.device))

            feature_batch = RemoteMTPTargetFeatureBatch(
                feature_batch_id=feature_batch_id,
                phase=phase,
                requests=(
                    request_views
                    if request_views is not None
                    else self._request_views(
                        batch,
                        use_decode_kv_boundary=phase == "verify",
                    )
                ),
                target_hidden_states=hidden_snapshot,
                next_token_ids=token_snapshot,
                accept_lens=accept_snapshot,
                new_seq_lens=seq_snapshot,
                claim=claim,
                fallback_reason=fallback_reason,
                input_token_ids=input_snapshot,
                extend_lens=(
                    tuple(int(value) for value in batch.extend_lens)
                    if phase == "prefill"
                    else None
                ),
                prefix_lens=(
                    tuple(int(value) for value in batch.prefix_lens)
                    if phase == "prefill"
                    else None
                ),
                row_stride=row_stride,
                completion_event=completion_event,
            ).validate()
            if not bool(offer(feature_batch)):
                self.remote_feature_batches_dropped += 1
                return None
            self.remote_feature_batches_published += 1
            return feature_batch_id
        except Exception:
            self.remote_feature_batches_dropped += 1
            logger.exception("REMOTE_MTP target feature publication failed")
            return None

    def _target_feature_batch_id(self, batch: ScheduleBatch) -> str:
        return f"tp{self.ps.tp_rank}:gpu{self.gpu_id}:forward{batch.forward_iter}"

    def _record_target_execution(
        self,
        *,
        batch: ScheduleBatch,
        phase: str,
        executed_source: str,
        verification_tokens: int,
        started_monotonic_ns: int,
    ) -> None:
        record = getattr(self.candidate_source, "record_target_execution", None)
        if record is None:
            return
        try:
            record(
                feature_batch_id=self._target_feature_batch_id(batch),
                phase=phase,
                executed_source=executed_source,
                batch_rows=len(batch.reqs),
                verification_tokens=verification_tokens,
                started_monotonic_ns=started_monotonic_ns,
                finished_monotonic_ns=time.monotonic_ns(),
            )
        except Exception:
            logger.exception("REMOTE_MTP target execution telemetry failed")

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        execution_started_ns = time.monotonic_ns()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            request_views = self._request_views(
                batch,
                use_decode_kv_boundary=False,
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            batch_output.next_draft_input = self._relay_input(
                _eagle_prefill_tail_tokens(batch, batch_output.next_token_ids)
            )
            batch_output.remote_mtp_fallback_reason = "prefill"
            batch_output.remote_mtp_request_views = request_views
            batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
                batch=batch,
                batch_output=batch_output,
                phase="prefill",
                claim=None,
                fallback_reason="prefill",
                row_stride=1,
                request_views=request_views,
            )
            self._record_target_execution(
                batch=batch,
                phase="prefill",
                executed_source="autoregressive",
                verification_tokens=max(1, sum(batch.extend_lens or ())),
                started_monotonic_ns=execution_started_ns,
            )
            return batch_output

        request_views = self._request_views(batch)
        verify_input, claim, fallback_reason = self._try_remote_verify_input(batch)
        if verify_input is None:
            verify_input = self._build_trivial_verify_input(batch)
            active_num_steps = 0
            active_num_draft_tokens = 1
            self.remote_fallback_batches += 1
        else:
            active_num_steps = self.speculative_num_steps
            active_num_draft_tokens = self.speculative_num_draft_tokens
            self.remote_claimed_batches += 1
            self.remote_claimed_requests += len(batch.reqs)

        batch.spec_info = verify_input
        with spec_stage_span("remote_mtp_target_verify"):
            batch_output = run_eagle_verify(
                batch,
                target_worker=self.target_worker,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                plan_stream=self.plan_stream,
                plan_stream_ctx=self.plan_stream_ctx,
                topk=1,
                num_steps=active_num_steps,
                num_draft_tokens=active_num_draft_tokens,
                device=self.device,
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
            )
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        batch_output.next_draft_input.topk_p = torch.zeros(
            (len(batch.reqs), 1), dtype=torch.float32, device=self.device
        )
        batch_output.next_draft_input.topk_index = torch.zeros(
            (len(batch.reqs), 1), dtype=torch.int64, device=self.device
        )
        batch_output.next_draft_input.hidden_states = None
        batch_output.remote_mtp_claim = claim
        batch_output.remote_mtp_executed_source = (
            "remote_mtp" if claim is not None else "autoregressive"
        )
        batch_output.remote_mtp_fallback_reason = fallback_reason
        batch_output.remote_mtp_request_views = request_views
        batch_output.remote_mtp_feature_batch_id = self._offer_target_feature_batch(
            batch=batch,
            batch_output=batch_output,
            phase="verify",
            claim=claim,
            fallback_reason=fallback_reason,
            row_stride=active_num_draft_tokens,
            request_views=request_views,
        )
        self._record_target_execution(
            batch=batch,
            phase="verify",
            executed_source=batch_output.remote_mtp_executed_source,
            verification_tokens=len(batch.reqs) * active_num_draft_tokens,
            started_monotonic_ns=execution_started_ns,
        )
        return batch_output

    def on_remote_mtp_result_cpu(self, batch, result, committed_tokens) -> None:
        commit_time = time.monotonic_ns()
        for index, tokens in enumerate(committed_tokens):
            if tokens and (
                batch.reqs[index].remote_mtp_max_service_gap_ns is not None
                or batch.reqs[index].remote_mtp_service_period_ns is not None
            ):
                batch.reqs[index].remote_mtp_last_target_service_ns = commit_time
        feature_batch_id = result.remote_mtp_feature_batch_id
        if feature_batch_id is None:
            return
        claim = result.remote_mtp_claim
        executed_source = getattr(
            result,
            "remote_mtp_executed_source",
            "remote_mtp" if claim is not None else "autoregressive",
        )
        if (
            executed_source != "autoregressive"
            and result.num_correct_drafts_per_req_cpu is None
        ):
            self.remote_outcome_batches_dropped += 1
            logger.error("REMOTE_MTP result is missing per-request accept counts")
            return

        try:
            requests = result.remote_mtp_request_views
            if requests is None or len(requests) != len(batch.reqs):
                raise ValueError(
                    "REMOTE_MTP result lost its pre-forward request boundaries"
                )
            outcomes = tuple(
                RemoteMTPTargetSettlement(
                    feature_batch_id=feature_batch_id,
                    phase="verify",
                    request=(
                        claim.requests[index] if claim is not None else requests[index]
                    ),
                    executed_source=executed_source,
                    committed_token_ids=tuple(
                        int(token) for token in committed_tokens[index]
                    ),
                    candidate_id=(
                        claim.candidate_ids[index] if claim is not None else None
                    ),
                    prefix_index_digest=(
                        claim.prefix_index_digests[index] if claim is not None else None
                    ),
                    candidate_token_ids=(
                        claim.token_ids[index] if claim is not None else ()
                    ),
                    verified_depth=(
                        claim.depth
                        if claim is not None
                        else (
                            self.speculative_num_steps
                            if executed_source == "local_mtp"
                            else 0
                        )
                    ),
                    accepted_draft_count=(
                        result.num_correct_drafts_per_req_cpu[index]
                        if executed_source != "autoregressive"
                        else 0
                    ),
                    fallback_reason=(
                        None if claim is not None else result.remote_mtp_fallback_reason
                    ),
                    plan_id=(
                        batch.remote_mtp_target_plan_id if claim is not None else None
                    ),
                    snapshot_generation=(
                        batch.remote_mtp_target_plan_generation
                        if claim is not None
                        else None
                    ),
                    window_id=(
                        batch.remote_mtp_target_window_id if claim is not None else None
                    ),
                ).validate()
                for index in range(len(requests))
                if committed_tokens[index]
            )
            if not outcomes:
                return
            offer = getattr(
                self.candidate_source,
                "offer_verification_outcomes",
                None,
            )
            if offer is None or not bool(offer(outcomes)):
                self.remote_outcome_batches_dropped += 1
                return
            self.remote_outcome_batches_published += 1
        except Exception:
            # Outcome delivery is advisory to future remote work.  Losing it
            # may force resynchronization but cannot invalidate target output.
            self.remote_outcome_batches_dropped += 1
            logger.exception("REMOTE_MTP outcome publication failed")

    def on_remote_mtp_prefill_result_cpu(self, batch, result, committed_tokens) -> None:
        commit_time = time.monotonic_ns()
        for index, tokens in enumerate(committed_tokens):
            if tokens and (
                batch.reqs[index].remote_mtp_max_service_gap_ns is not None
                or batch.reqs[index].remote_mtp_service_period_ns is not None
            ):
                batch.reqs[index].remote_mtp_last_target_service_ns = commit_time
        feature_batch_id = result.remote_mtp_feature_batch_id
        if feature_batch_id is None:
            return
        try:
            requests = result.remote_mtp_request_views
            if requests is None or len(requests) != len(batch.reqs):
                raise ValueError(
                    "REMOTE_MTP prefill result lost its pre-forward request boundaries"
                )
            outcomes = tuple(
                RemoteMTPTargetSettlement(
                    feature_batch_id=feature_batch_id,
                    phase="prefill",
                    request=requests[index],
                    executed_source="autoregressive",
                    committed_token_ids=tuple(
                        int(token) for token in committed_tokens[index]
                    ),
                    verified_depth=0,
                    fallback_reason="prefill",
                ).validate()
                for index in range(len(requests))
                if committed_tokens[index]
            )
            if not outcomes:
                return
            offer = getattr(
                self.candidate_source,
                "offer_verification_outcomes",
                None,
            )
            if offer is None or not bool(offer(outcomes)):
                self.remote_outcome_batches_dropped += 1
                return
            self.remote_outcome_batches_published += 1
        except Exception:
            self.remote_outcome_batches_dropped += 1
            logger.exception("REMOTE_MTP prefill settlement publication failed")
