"""Unit tests for fused KV protection survivor filtering."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_page_tags import (
    KV_PAGE_VALIDATION_REMOTE_FAILURE,
    KVFusedProtectionError,
)
from sglang.srt.model_executor.model_runner import (
    FusedKVPageProtectionCheck,
    ModelRunner,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.out_cache_loc = torch.tensor([10 + i for i in range(len(reqs))])
        self.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, *, keep_indices):
        self.reqs = [self.reqs[index] for index in keep_indices]
        self.out_cache_loc = None


def _request(rid, pool_idx):
    req = SimpleNamespace(
        rid=rid,
        req_pool_idx=pool_idx,
        bootstrap_room=pool_idx,
        kv_attention_tag_manifest=None,
        kv_transfer_page_tag_manifest=None,
        kv_committed_freed=False,
        kv_fused_protection_deferred_release=False,
        finished_reason=None,
        to_finish=None,
        is_retracted=False,
    )
    req.finished = lambda: req.finished_reason is not None
    return req


class TestFusedKVProtectionRetry(CustomTestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.kv_protection_manager = SimpleNamespace(
            verify_protection_batch=MagicMock(return_value=[])
        )
        scheduler.tree_cache = SimpleNamespace(supports_mamba=lambda: False)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        scheduler.enable_overlap = False
        scheduler.cur_batch = None
        return scheduler

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_failed_request_is_released_without_publishing_result(
        self, release_kv_cache
    ):
        scheduler = self._scheduler()
        good = _request("good", 1)
        bad = _request("bad", 2)
        batch = _FakeBatch([good, bad])
        error = KVFusedProtectionError(
            batch_indices=[1], request_pool_indices=[2], statuses=[8]
        )

        scheduler._handle_fused_kv_protection_failure(batch, error)

        self.assertEqual([req.rid for req in batch.reqs], ["good", "bad"])
        self.assertEqual(batch.out_cache_loc.tolist(), [10, 11])
        self.assertIsNotNone(bad.finished_reason)
        release_kv_cache.assert_called_once_with(
            bad, scheduler.tree_cache, is_insert=False
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_all_failed_requests_are_aborted(self, release_kv_cache):
        scheduler = self._scheduler()
        batch = _FakeBatch([_request("a", 1), _request("b", 2)])
        error = KVFusedProtectionError(
            batch_indices=[0, 1],
            request_pool_indices=[1, 2],
            statuses=[8, 16],
        )

        scheduler._handle_fused_kv_protection_failure(batch, error)

        self.assertTrue(all(req.finished_reason is not None for req in batch.reqs))
        self.assertEqual(release_kv_cache.call_count, 2)

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_request_pool_identity_mismatch_fails_closed(self, release_kv_cache):
        scheduler = self._scheduler()
        batch = _FakeBatch([_request("a", 1), _request("b", 2)])
        error = KVFusedProtectionError(
            batch_indices=[1], request_pool_indices=[99], statuses=[8]
        )

        with self.assertRaisesRegex(RuntimeError, "request-pool identity mismatch"):
            scheduler._handle_fused_kv_protection_failure(batch, error)

        self.assertTrue(all(req.finished_reason is None for req in batch.reqs))
        release_kv_cache.assert_not_called()

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_overlap_failure_defers_release_until_drain_finishes(
        self, release_kv_cache
    ):
        scheduler = self._scheduler()
        scheduler.enable_overlap = True
        good = _request("good", 1)
        bad = _request("bad", 2)
        batch = _FakeBatch([good, bad])
        check = FusedKVPageProtectionCheck(
            request_pool_indices=torch.tensor([1, 2]),
            statuses=torch.tensor([0, 8], dtype=torch.int32),
            failed=torch.tensor([0, 1], dtype=torch.int32),
        )
        copy_done = MagicMock()
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([5, 6]),
            copy_done=copy_done,
            fused_kv_page_protection_check=check,
        )
        scheduler.cur_batch = _FakeBatch([good, bad])

        scheduler._finalize_fused_kv_protection_result(batch, result)

        copy_done.synchronize.assert_called_once_with()
        self.assertIsNone(result.fused_kv_page_protection_check)
        self.assertEqual(result.fused_kv_page_protection_deferred_release_rids, {"bad"})
        self.assertEqual(result.fused_kv_page_protection_failed_rids, {"bad"})
        self.assertTrue(bad.kv_fused_protection_deferred_release)
        release_kv_cache.assert_not_called()

        scheduler.cur_batch = _FakeBatch([good])
        error = KVFusedProtectionError(
            batch_indices=[1], request_pool_indices=[2], statuses=[8]
        )
        scheduler._handle_fused_kv_protection_failure(batch, error)

        self.assertFalse(bad.kv_fused_protection_deferred_release)
        release_kv_cache.assert_called_once_with(
            bad, scheduler.tree_cache, is_insert=False
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_finished_drain_failure_does_not_overwrite_completion(
        self, release_kv_cache
    ):
        scheduler = self._scheduler()
        req = _request("done", 1)
        completed_reason = object()
        req.finished_reason = completed_reason
        batch = _FakeBatch([req])
        error = KVFusedProtectionError(
            batch_indices=[0], request_pool_indices=[1], statuses=[8]
        )

        scheduler._handle_fused_kv_protection_failure(batch, error)

        self.assertIs(req.finished_reason, completed_reason)
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_tp_remote_failure_is_propagated_after_async_reduce(self):
        runner = SimpleNamespace(
            kv_fused_page_protection_enabled=True,
            tp_size=2,
            tp_group=SimpleNamespace(device_group=object()),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            spec_info=None,
            batch_size=2,
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        )
        table = SimpleNamespace(
            fused_failure_status=MagicMock(
                return_value=torch.zeros(2, dtype=torch.int32)
            )
        )

        work = MagicMock()
        reduced = {}

        def propagate_remote_failure(failed, **kwargs):
            self.assertTrue(kwargs["async_op"])
            reduced["failed"] = failed
            return work

        work.wait.side_effect = lambda: reduced["failed"].__setitem__(1, 1)

        with patch(
            "sglang.srt.model_executor.model_runner.dist.all_reduce",
            side_effect=propagate_remote_failure,
        ):
            check = ModelRunner._start_fused_kv_page_protection_check(
                runner, forward_batch, table
            )
            error = check.materialize_error()

        work.wait.assert_called_once_with()
        self.assertIsInstance(error, KVFusedProtectionError)
        self.assertEqual(error.batch_indices, (1,))
        self.assertEqual(error.request_pool_indices, (2,))
        self.assertEqual(error.statuses, (KV_PAGE_VALIDATION_REMOTE_FAILURE,))

    def test_sampling_defers_protection_materialization(self):
        events = []
        runner = SimpleNamespace(
            _preprocess_logits=lambda *_args: events.append("preprocess"),
            sampler=lambda *_args: events.append("sample") or torch.tensor([7]),
            maybe_update_ngram_token_table=lambda *_args, **_kwargs: events.append(
                "ngram"
            ),
        )
        forward_batch = SimpleNamespace(
            sampling_info=object(),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            positions=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            ngram_embedding_info=None,
        )
        check = FusedKVPageProtectionCheck(
            request_pool_indices=torch.tensor([1]),
            statuses=torch.tensor([0], dtype=torch.int32),
            failed=torch.tensor([0], dtype=torch.int32),
        )

        next_token_ids = ModelRunner.sample(
            runner, object(), forward_batch, fused_kv_page_protection_check=check
        )

        self.assertEqual(next_token_ids.tolist(), [7])
        self.assertEqual(events, ["preprocess", "sample", "ngram"])

    @patch("sglang.srt.model_executor.model_runner.update_token_table_decode")
    def test_ngram_update_masks_globally_failed_rows(self, update_token_table):
        work = MagicMock()
        check = FusedKVPageProtectionCheck(
            request_pool_indices=torch.tensor([3, 4]),
            statuses=torch.tensor([0, 8], dtype=torch.int32),
            failed=torch.tensor([0, 1], dtype=torch.int32),
            work=work,
        )
        runner = SimpleNamespace(
            _preprocess_logits=MagicMock(),
            sampler=MagicMock(return_value=torch.tensor([7, 8])),
        )
        runner.maybe_update_ngram_token_table = lambda *args, **kwargs: (
            ModelRunner.maybe_update_ngram_token_table(runner, *args, **kwargs)
        )
        ngram_embedding_info = SimpleNamespace(
            token_table=object(),
            out_column_starts=torch.zeros(2, dtype=torch.int64),
            out_req_lens=torch.zeros(2, dtype=torch.int64),
        )
        forward_batch = SimpleNamespace(
            sampling_info=object(),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            positions=torch.tensor([0, 0]),
            seq_lens=torch.tensor([4, 5]),
            req_pool_indices=torch.tensor([3, 4]),
            batch_size=2,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            ngram_embedding_info=ngram_embedding_info,
        )

        ModelRunner.sample(
            runner,
            object(),
            forward_batch,
            fused_kv_page_protection_check=check,
        )

        work.wait.assert_called_once_with()
        self.assertIsNone(check.work)
        self.assertEqual(
            update_token_table.call_args.kwargs["row_indices"].tolist(), [3, 0]
        )

    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
    )
    def test_failed_drain_result_does_not_append_token(self, release_kv_cache):
        req = _request("bad", 2)
        req.finished_reason = object()
        req.kv_fused_protection_deferred_release = True
        req.output_ids = []
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            batch_size=lambda: 1,
        )
        triggering_result = GenerationBatchResult(
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=[98],
            fused_kv_page_protection_failed_rids={"bad"},
            fused_kv_page_protection_deferred_release_rids={"bad"},
        )
        metrics_reporter = SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            report_decode_stats=MagicMock(),
        )
        output_streamer = SimpleNamespace(stream_output=MagicMock())
        allocator = SimpleNamespace(
            free_group_begin=MagicMock(), free_group_end=MagicMock()
        )
        processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=MagicMock(),
            enable_overlap=True,
            enable_overlap_mlx=False,
            server_args=SimpleNamespace(enable_metrics=False),
            model_config=SimpleNamespace(think_end_id=None),
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(supports_mamba=lambda: False),
            hisparse_coordinator=None,
            req_to_token_pool=MagicMock(),
            decode_offload_manager=None,
            metrics_collector=MagicMock(),
            metrics_reporter=metrics_reporter,
            draft_worker=MagicMock(),
            model_worker=MagicMock(),
            logprob_result_processor=MagicMock(),
            output_streamer=output_streamer,
            abort_request=MagicMock(),
        )

        processor.process_batch_result_decode(batch, triggering_result)

        self.assertEqual(req.output_ids, [])
        self.assertTrue(req.kv_fused_protection_deferred_release)
        release_kv_cache.assert_not_called()

        drain_result = GenerationBatchResult(
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=[99],
        )
        processor.process_batch_result_decode(batch, drain_result)

        self.assertEqual(req.output_ids, [])
        self.assertFalse(req.kv_fused_protection_deferred_release)
        release_kv_cache.assert_called_once_with(
            req, processor.tree_cache, is_insert=False
        )
        self.assertEqual(output_streamer.stream_output.call_count, 2)

    def test_non_overlap_syncs_tokens_before_materializing_status(self):
        scheduler = self._scheduler()
        batch = _FakeBatch([_request("good", 1)])
        result = GenerationBatchResult(next_token_ids=torch.tensor([7]))
        check = MagicMock()

        def materialize_error():
            self.assertEqual(result.next_token_ids, [7])
            return None

        check.materialize_error.side_effect = materialize_error
        result.fused_kv_page_protection_check = check

        scheduler._finalize_fused_kv_protection_result(batch, result)

        check.materialize_error.assert_called_once_with()
        self.assertIsNone(result.fused_kv_page_protection_check)

    def test_preprocess_exception_drains_pending_collective(self):
        work = MagicMock()
        runner = SimpleNamespace(
            _preprocess_logits=MagicMock(side_effect=ValueError("bad logits")),
            sampler=MagicMock(),
            maybe_update_ngram_token_table=MagicMock(),
        )
        forward_batch = SimpleNamespace(sampling_info=object())
        check = FusedKVPageProtectionCheck(
            request_pool_indices=torch.tensor([1]),
            statuses=torch.tensor([0], dtype=torch.int32),
            failed=torch.tensor([0], dtype=torch.int32),
            work=work,
        )

        with self.assertRaisesRegex(ValueError, "bad logits"):
            ModelRunner.sample(
                runner,
                object(),
                forward_batch,
                fused_kv_page_protection_check=check,
            )

        work.wait.assert_called_once_with()
        self.assertIsNone(check.work)
        runner.sampler.assert_not_called()
        runner.maybe_update_ngram_token_table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
