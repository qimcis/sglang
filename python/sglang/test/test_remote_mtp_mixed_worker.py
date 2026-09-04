from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal contract environment
    torch = None


def _module(name: str, **values):
    module = types.ModuleType(name)
    for key, value in values.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_worker_module():
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.speculative",
    )
    dependency_names = (
        "sglang.srt.layers.moe.utils",
        "sglang.srt.speculative.eagle_worker_common",
        "sglang.srt.speculative.eagle_worker_v2",
        "sglang.srt.speculative.remote_mtp_io",
        "sglang.srt.speculative.remote_mtp_worker_v2",
        "sglang.srt.speculative.spec_utils",
    )
    module_name = "remote_mtp_local_worker_under_test"
    touched_names = package_names + dependency_names + (module_name,)
    saved = {name: sys.modules.get(name) for name in touched_names}
    try:
        for package in package_names:
            sys.modules[package] = types.ModuleType(package)

        def null_context(*args, **kwargs):
            return contextlib.nullcontext()
        _module(
            "sglang.srt.layers.moe.utils",
            speculative_moe_a2a_backend_context=null_context,
            speculative_moe_backend_context=null_context,
        )
        common = _module(
            "sglang.srt.speculative.eagle_worker_common",
            build_eagle_verify_input=lambda *args, **kwargs: None,
            run_eagle_verify=lambda *args, **kwargs: None,
        )

        class EagleDraftWorker:
            pass

        class EAGLEWorkerV2:
            pass

        _module(
            "sglang.srt.speculative.eagle_worker_v2",
            EagleDraftWorker=EagleDraftWorker,
            EAGLEWorkerV2=EAGLEWorkerV2,
        )
        _module(
            "sglang.srt.speculative.remote_mtp_io",
            RemoteMTPBatchClaim=object,
            RemoteMTPRequestView=object,
            remote_mtp_linear_chain_layout=lambda depth: (
                tuple(range(-1, depth - 1)),
                tuple(range(depth)),
            ),
        )

        class RemoteMTPWorkerV2:
            pass

        _module(
            "sglang.srt.speculative.remote_mtp_worker_v2",
            RemoteMTPWorkerV2=RemoteMTPWorkerV2,
        )
        _module(
            "sglang.srt.speculative.spec_utils",
            spec_stage_span=null_context,
        )

        path = (
            pathlib.Path(__file__).parents[1]
            / "srt"
            / "speculative"
            / "remote_mtp_local_worker_v2.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, common
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _SpecInfo:
    def __init__(self, values):
        self.values = values

    def __copy__(self):
        return _SpecInfo(self.values)

    def filter_batch(self, *, new_indices, **_kwargs):
        self.values = self.values[new_indices]


@unittest.skipIf(torch is None, "torch is required for the executable CPU worker test")
class TestRemoteMTPMixedWorker(unittest.TestCase):
    def test_local_drafts_only_missing_rows_then_verifies_one_full_batch(self):
        module, common = _load_worker_module()
        owner = module.RemoteMTPLocalWorkerV2
        worker = owner.__new__(owner)
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.device = "cpu"
        worker.tree_mask_mode = "linear"
        worker.target_worker = object()
        worker.remote_claimed_batches = 0
        worker.remote_claimed_requests = 0
        worker.remote_local_fallback_batches = 0
        worker.remote_local_fallback_requests = 0
        worker.remote_local_draft_rows = 0
        worker.local_proposal_draft_rows = 0
        worker.local_state_extend_rows = 0
        worker.activate_step_by_batch = lambda _rows: None
        worker._activate_batch_depth = lambda _batch: None

        request_views = tuple(f"view-{name}" for name in "abcd")
        worker._request_views = lambda _batch: request_views
        claim = SimpleNamespace(token_ids=((101, 102), (301, 302)))
        worker._try_remote_row_claim = lambda _batch, _views: (
            claim,
            (0, 2),
            None,
        )

        draft_calls = []
        extend_calls = []

        class DraftWorker:
            draft_runner = SimpleNamespace(tp_group=None)

            @staticmethod
            def draft_tp_context(_group):
                return contextlib.nullcontext()

            @staticmethod
            def draft(local_batch):
                draft_calls.append(
                    (
                        tuple(req.rid for req in local_batch.reqs),
                        tuple(local_batch.spec_info.values.tolist()),
                        tuple(
                            local_batch.sampling_info.temperatures.flatten().tolist()
                        ),
                    )
                )
                # root + K local candidates for rows b and d only
                return SimpleNamespace(
                    draft_token=torch.tensor(
                        [[11, 111, 112], [33, 331, 332]],
                        dtype=torch.long,
                    ).flatten()
                )

            @staticmethod
            def _draft_extend_for_decode(batch, _output):
                extend_calls.append(tuple(req.rid for req in batch.reqs))

        worker._draft_worker = DraftWorker()

        assembled = []

        def build_verify(
            batch, _draft_input, _parents, _selected, candidates, *_args, **_kwargs
        ):
            assembled.append(
                (
                    tuple(req.rid for req in batch.reqs),
                    candidates.clone(),
                )
            )
            return SimpleNamespace(candidates=candidates)

        common.build_eagle_verify_input = build_verify
        module.build_eagle_verify_input = build_verify
        verify_calls = []

        def verify(batch):
            verify_calls.append(tuple(req.rid for req in batch.reqs))
            return SimpleNamespace(new_seq_lens=batch.seq_lens)

        worker.verify = verify
        worker._offer_target_feature_batch = lambda **_kwargs: "feature"
        execution_records = []
        worker._record_target_execution = lambda **values: execution_records.append(
            values
        )

        rows = [
            SimpleNamespace(rid=name, grammar=None, return_logprob=False)
            for name in "abcd"
        ]
        sampling_info = SimpleNamespace(
            temperatures=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            top_ps=torch.ones(4),
            top_ks=torch.ones(4, dtype=torch.int32),
            min_ps=torch.zeros(4),
            sampling_seed=None,
            logit_bias=None,
            rids_int=None,
            bootstrap_room_ids_int=None,
            grammars=None,
            return_sampling_masks=None,
        )
        batch = SimpleNamespace(
            reqs=rows,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            seq_lens=torch.tensor([20, 21, 22, 23]),
            seq_lens_cpu=torch.tensor([20, 21, 22, 23]),
            orig_seq_lens=torch.tensor([20, 21, 22, 23]),
            req_pool_indices=torch.tensor([0, 1, 2, 3]),
            req_pool_indices_cpu=torch.tensor([0, 1, 2, 3]),
            input_ids=torch.tensor([1, 2, 3, 4]),
            input_embeds=None,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            encoder_lens=None,
            out_cache_loc=None,
            out_cache_loc_dsv4=None,
            multimodal_inputs=None,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            spec_info=_SpecInfo(torch.tensor([10, 11, 12, 13])),
            sampling_info=sampling_info,
        )

        output = worker.forward_batch_generation(batch)

        self.assertEqual(draft_calls, [(("b", "d"), (11, 13), (2.0, 4.0))])
        self.assertEqual(verify_calls, [("a", "b", "c", "d")])
        self.assertEqual(extend_calls, [("a", "b", "c", "d")])
        self.assertEqual(len(assembled), 1)
        self.assertTrue(
            torch.equal(
                assembled[0][1],
                torch.tensor([[101, 102], [111, 112], [301, 302], [331, 332]]),
            )
        )
        self.assertEqual(worker.remote_local_draft_rows, 2)
        self.assertEqual(worker.local_proposal_draft_rows, 2)
        self.assertEqual(worker.local_state_extend_rows, 4)
        self.assertEqual(
            output.remote_mtp_row_sources,
            (
                "remote_mtp",
                "local_mtp",
                "remote_mtp",
                "local_mtp",
            ),
        )
        self.assertEqual(execution_records[0]["executed_source"], "mixed_mtp")
        self.assertEqual(execution_records[0]["local_proposal_draft_rows"], 2)
        self.assertEqual(execution_records[0]["local_state_extend_rows"], 4)

    def test_all_remote_rows_skip_local_proposals_but_extend_local_state(self):
        module, _common = _load_worker_module()
        owner = module.RemoteMTPLocalWorkerV2
        worker = owner.__new__(owner)
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.remote_claimed_batches = 0
        worker.remote_claimed_requests = 0
        worker.remote_local_fallback_batches = 0
        worker.remote_local_fallback_requests = 0
        worker.remote_local_draft_rows = 0
        worker.local_proposal_draft_rows = 0
        worker.local_state_extend_rows = 0
        worker._activate_batch_depth = lambda _batch: None
        request_views = ("view-a", "view-b")
        claim = SimpleNamespace(token_ids=((101, 102), (201, 202)))
        worker._request_views = lambda _batch: request_views
        worker._try_remote_row_claim = lambda _batch, _views: (
            claim,
            (0, 1),
            None,
        )
        worker._build_remote_verify_input = lambda *_args: SimpleNamespace()
        verify_calls = []
        worker.verify = lambda batch: (
            verify_calls.append(tuple(req.rid for req in batch.reqs))
            or SimpleNamespace(new_seq_lens=batch.seq_lens)
        )

        class DraftWorker:
            draft_runner = SimpleNamespace(tp_group=None)

            @staticmethod
            def draft_tp_context(_group):
                return contextlib.nullcontext()

            @staticmethod
            def draft(_batch):
                raise AssertionError("all-remote batch must not run resident draft")

            @staticmethod
            def _draft_extend_for_decode(_batch, _output):
                return None

        worker._draft_worker = DraftWorker()
        worker._offer_target_feature_batch = lambda **_kwargs: "feature"
        execution_records = []
        worker._record_target_execution = lambda **values: execution_records.append(
            values
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")],
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            seq_lens=torch.tensor([20, 21]),
            spec_info=object(),
        )

        worker.forward_batch_generation(batch)

        self.assertEqual(verify_calls, [("a", "b")])
        self.assertEqual(worker.remote_local_draft_rows, 0)
        self.assertEqual(worker.local_proposal_draft_rows, 0)
        self.assertEqual(worker.local_state_extend_rows, 2)
        self.assertEqual(execution_records[0]["local_proposal_draft_rows"], 0)
        self.assertEqual(execution_records[0]["local_state_extend_rows"], 2)

    def test_stale_remote_subset_immediately_falls_back_to_full_local_mtp(self):
        module, _common = _load_worker_module()
        owner = module.RemoteMTPLocalWorkerV2
        worker = owner.__new__(owner)
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.remote_invalid_claims = 0
        worker.remote_local_fallback_batches = 0
        worker.remote_local_fallback_requests = 0
        worker.remote_local_draft_rows = 0
        worker.local_proposal_draft_rows = 0
        worker.local_state_extend_rows = 0
        worker._activate_batch_depth = lambda _batch: None
        worker._request_views = lambda _batch: ("view-a", "view-b", "view-c")

        claim_calls = []

        class CandidateSource:
            @staticmethod
            def try_claim_batch(requests, **_kwargs):
                claim_calls.append(requests)

            @staticmethod
            def take_last_claim_failure_reason():
                return "candidate_stale"

        worker.candidate_source = CandidateSource()
        worker._offer_target_feature_batch = lambda **_kwargs: "feature"
        worker._record_target_execution = lambda **_kwargs: None

        local_calls = []

        def local_forward(_worker, batch, on_publish=None):
            local_calls.append(tuple(req.rid for req in batch.reqs))
            if on_publish is not None:
                on_publish(batch.seq_lens)
            return SimpleNamespace(new_seq_lens=batch.seq_lens)

        module.EAGLEWorkerV2.forward_batch_generation = local_forward
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=name) for name in "abc"],
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            seq_lens=torch.tensor([20, 21, 22]),
            remote_mtp_candidate_ids_by_row=("ca", None, "cc"),
            remote_mtp_target_plan_id="plan",
            remote_mtp_target_plan_generation=4,
            remote_mtp_target_window_id="window",
        )

        output = worker.forward_batch_generation(batch)

        self.assertEqual(claim_calls, [("view-a", "view-c")])
        self.assertEqual(local_calls, [("a", "b", "c")])
        self.assertEqual(worker.remote_local_draft_rows, 3)
        self.assertEqual(worker.local_proposal_draft_rows, 3)
        self.assertEqual(worker.local_state_extend_rows, 3)
        self.assertEqual(output.remote_mtp_executed_source, "local_mtp")
        self.assertEqual(output.remote_mtp_fallback_reason, "candidate_stale")

    def test_adaptive_k2_k3_k4_runs_one_exact_width_fused_verification(self):
        module, _common = _load_worker_module()
        owner = module.RemoteMTPLocalWorkerV2

        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                worker = owner.__new__(owner)
                worker.speculative_num_steps = 4
                worker.speculative_num_draft_tokens = 5
                worker.remote_mtp_max_depth = 4
                worker._remote_mtp_depth_buffers = {
                    value: (f"parents-k{value}", f"indices-k{value}")
                    for value in range(1, 5)
                }
                worker.remote_claimed_batches = 0
                worker.remote_claimed_requests = 0
                worker.remote_local_fallback_batches = 0
                worker.remote_local_fallback_requests = 0
                worker.remote_local_draft_rows = 0
                worker.local_proposal_draft_rows = 0
                worker.local_state_extend_rows = 0
                worker.activate_step_by_batch = lambda _rows: None

                inner_backend = SimpleNamespace(
                    speculative_num_steps=4,
                    speculative_num_draft_tokens=5,
                )
                draft_backend = SimpleNamespace(
                    speculative_num_steps=4,
                    speculative_num_draft_tokens=5,
                    attn_backends=(inner_backend,),
                )
                extend_backend = SimpleNamespace(
                    speculative_num_steps=4,
                    speculative_num_draft_tokens=5,
                )
                target_backend = SimpleNamespace(
                    speculative_num_steps=4,
                    speculative_num_draft_tokens=5,
                )
                target_runner = SimpleNamespace(
                    attn_backend=target_backend,
                    decode_cuda_graph_runner=None,
                )
                worker.target_worker = SimpleNamespace(model_runner=target_runner)

                draft_calls = []
                extend_calls = []

                class DraftWorker:
                    speculative_num_steps = 4
                    speculative_num_draft_tokens = 5
                    _topk1_parents_prealloc = "parents-k4"
                    _topk1_score_indices_prealloc = "indices-k4"
                    draft_attn_backend = draft_backend
                    draft_extend_attn_backend = extend_backend
                    cuda_graph_runner = None
                    cuda_graph_runner_for_draft_extend = None
                    draft_runner = SimpleNamespace(
                        tp_group=None,
                        attn_backend=draft_backend,
                    )

                    @staticmethod
                    def draft_tp_context(_group):
                        return contextlib.nullcontext()

                    def draft(self, local_batch, *, calls=draft_calls):
                        calls.append(
                            (
                                tuple(req.rid for req in local_batch.reqs),
                                self.speculative_num_steps,
                            )
                        )
                        width = self.speculative_num_draft_tokens
                        return SimpleNamespace(
                            draft_token=torch.arange(width, dtype=torch.long)
                        )

                    @staticmethod
                    def _draft_extend_for_decode(
                        batch,
                        _output,
                        calls=extend_calls,
                    ):
                        calls.append(tuple(req.rid for req in batch.reqs))

                worker._draft_worker = DraftWorker()
                request_views = ("view-remote", "view-local")
                worker._request_views = (
                    lambda _batch, views=request_views: views
                )
                claim = SimpleNamespace(
                    depth=depth,
                    token_ids=(tuple(range(100, 100 + depth)),),
                )

                def claim_rows(
                    _batch,
                    _views,
                    active_worker=worker,
                    selected_depth=depth,
                    selected_claim=claim,
                ):
                    self.assertEqual(
                        active_worker.speculative_num_steps,
                        selected_depth,
                    )
                    return selected_claim, (0,), None

                worker._try_remote_row_claim = claim_rows
                worker._build_local_draft_batch = lambda batch, indices: SimpleNamespace(
                    reqs=[batch.reqs[index] for index in indices]
                )
                assembled_widths = []

                def build_mixed(
                    _batch,
                    _draft_input,
                    local_verify,
                    remote,
                    indices,
                    selected_claim=claim,
                    widths=assembled_widths,
                ):
                    self.assertIs(remote, selected_claim)
                    self.assertEqual(indices, (0,))
                    widths.append(local_verify.draft_token.numel() - 1)
                    return SimpleNamespace()

                worker._build_mixed_verify_input = build_mixed
                verify_calls = []

                def verify(batch, calls=verify_calls, active_worker=worker):
                    calls.append(
                        (
                            tuple(req.rid for req in batch.reqs),
                            active_worker.speculative_num_steps,
                        )
                    )
                    return SimpleNamespace(new_seq_lens=batch.seq_lens)

                worker.verify = verify
                worker._offer_target_feature_batch = lambda **_kwargs: "feature"
                execution_records = []
                worker._record_target_execution = (
                    lambda records=execution_records, **values: records.append(values)
                )
                batch = SimpleNamespace(
                    reqs=[
                        SimpleNamespace(rid="remote"),
                        SimpleNamespace(rid="local"),
                    ],
                    forward_mode=SimpleNamespace(is_extend=lambda: False),
                    is_extend_in_batch=False,
                    seq_lens=torch.tensor([20, 21]),
                    spec_info=object(),
                    remote_mtp_selected_depth=depth,
                )

                output = worker.forward_batch_generation(batch)

                self.assertEqual(draft_calls, [(('local',), depth)])
                self.assertEqual(assembled_widths, [depth])
                self.assertEqual(verify_calls, [(('remote', 'local'), depth)])
                self.assertEqual(extend_calls, [('remote', 'local')])
                self.assertEqual(worker.remote_local_draft_rows, 1)
                self.assertEqual(worker.local_proposal_draft_rows, 1)
                self.assertEqual(worker.local_state_extend_rows, 2)
                self.assertEqual(output.remote_mtp_selected_depth, depth)
                self.assertEqual(execution_records[0]["selected_depth"], depth)
                self.assertEqual(
                    worker._draft_worker._topk1_parents_prealloc,
                    f"parents-k{depth}",
                )
                for backend in (
                    inner_backend,
                    draft_backend,
                    extend_backend,
                    target_backend,
                ):
                    self.assertEqual(backend.speculative_num_steps, depth)
                    self.assertEqual(backend.speculative_num_draft_tokens, depth + 1)

    def test_shallow_adaptive_depth_falls_back_when_cuda_graph_is_bound(self):
        module, _common = _load_worker_module()
        owner = module.RemoteMTPLocalWorkerV2
        worker = owner.__new__(owner)
        worker.remote_mtp_max_depth = 4
        worker._remote_mtp_depth_buffers = {
            depth: (object(), object()) for depth in range(1, 5)
        }
        backend = SimpleNamespace(
            speculative_num_steps=4,
            speculative_num_draft_tokens=5,
        )
        worker._draft_worker = SimpleNamespace(
            speculative_num_steps=4,
            speculative_num_draft_tokens=5,
            _topk1_parents_prealloc=object(),
            _topk1_score_indices_prealloc=object(),
            draft_attn_backend=backend,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(attn_backend=backend),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=None,
        )
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=backend,
                decode_cuda_graph_runner=None,
            )
        )

        reason = worker._activate_batch_depth(
            SimpleNamespace(remote_mtp_selected_depth=2)
        )

        self.assertEqual(reason, "selected_depth_requires_eager_execution")
        self.assertEqual(worker.speculative_num_steps, 4)


if __name__ == "__main__":
    unittest.main()
