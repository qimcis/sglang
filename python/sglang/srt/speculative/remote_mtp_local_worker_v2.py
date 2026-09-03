"""External MTP first, resident native MTP second, target AR last.

This worker deliberately retains the native NextN model and draft KV on the
target GPU.  It is therefore a separate benchmark policy from REMOTE_MTP,
whose fallback is memory-maximizing AR and which never allocates this worker.
Both remote and local proposals enter the same native target verifier.
"""

from __future__ import annotations

import logging
import time

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
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
        self._record_target_execution(
            batch=batch,
            phase="verify",
            executed_source="local_mtp",
            verification_tokens=(len(batch.reqs) * self.speculative_num_draft_tokens),
            started_monotonic_ns=started_monotonic_ns,
        )
        return batch_output

    def forward_batch_generation(self, batch, on_publish=None):
        execution_started_ns = time.monotonic_ns()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
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
        verify_input, claim, fallback_reason = self._try_remote_verify_input(batch)
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

        assert verify_input is not None
        batch.spec_info = verify_input
        with spec_stage_span("remote_mtp_target_verify"):
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

        self.remote_claimed_batches += 1
        self.remote_claimed_requests += len(batch.reqs)
        batch_output.remote_mtp_claim = claim
        batch_output.remote_mtp_executed_source = "remote_mtp"
        batch_output.remote_mtp_fallback_reason = None
        batch_output.remote_mtp_request_views = request_views
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
        )
        return batch_output


__all__ = ["RemoteMTPLocalWorkerV2"]
