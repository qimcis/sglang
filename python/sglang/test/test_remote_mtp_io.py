import hashlib
import importlib.util
import pathlib
import itertools
import sys
import unittest

try:
    from sglang.srt.speculative.remote_mtp_io import (
        BoundedRemoteMTPMailbox,
        RemoteMTPCandidate,
        RemoteMTPContractError,
        RemoteMTPRequestView,
        RemoteMTPSchedulerAdvice,
        RemoteMTPTargetFeatureBatch,
        RemoteMTPTargetSettlement,
        RemoteMTPVerificationOutcome,
        _reset_remote_mtp_candidate_source_factory_for_test,
        build_remote_mtp_decode_slice,
        build_remote_mtp_mixed_batch_plan,
        create_remote_mtp_scheduler_advisor,
        install_remote_mtp_scheduler_advisor_factory,
        remote_mtp_linear_chain_layout,
        remote_mtp_output_token_count,
        remote_mtp_rejoin_order,
        remote_mtp_service_window_partition,
    )
except ModuleNotFoundError:
    # CPU contract tests are intentionally runnable in a minimal environment
    # without importing SGLang's torch/transformers frontend dependencies.
    module_path = (
        pathlib.Path(__file__).parents[1] / "srt" / "speculative" / "remote_mtp_io.py"
    )
    spec = importlib.util.spec_from_file_location(
        "remote_mtp_io_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    BoundedRemoteMTPMailbox = module.BoundedRemoteMTPMailbox
    RemoteMTPCandidate = module.RemoteMTPCandidate
    RemoteMTPContractError = module.RemoteMTPContractError
    RemoteMTPRequestView = module.RemoteMTPRequestView
    RemoteMTPSchedulerAdvice = module.RemoteMTPSchedulerAdvice
    RemoteMTPTargetFeatureBatch = module.RemoteMTPTargetFeatureBatch
    RemoteMTPTargetSettlement = module.RemoteMTPTargetSettlement
    RemoteMTPVerificationOutcome = module.RemoteMTPVerificationOutcome
    _reset_remote_mtp_candidate_source_factory_for_test = (
        module._reset_remote_mtp_candidate_source_factory_for_test
    )
    build_remote_mtp_decode_slice = module.build_remote_mtp_decode_slice
    build_remote_mtp_mixed_batch_plan = module.build_remote_mtp_mixed_batch_plan
    create_remote_mtp_scheduler_advisor = module.create_remote_mtp_scheduler_advisor
    install_remote_mtp_scheduler_advisor_factory = (
        module.install_remote_mtp_scheduler_advisor_factory
    )
    remote_mtp_linear_chain_layout = module.remote_mtp_linear_chain_layout
    remote_mtp_output_token_count = module.remote_mtp_output_token_count
    remote_mtp_rejoin_order = module.remote_mtp_rejoin_order
    remote_mtp_service_window_partition = module.remote_mtp_service_window_partition


def request(rid: str, output_token_count: int = 0) -> RemoteMTPRequestView:
    return RemoteMTPRequestView(
        request_id=rid,
        request_incarnation=f"inc-{rid}",
        prompt_token_count=4096,
        output_token_count=output_token_count,
    )


def candidate(
    rid: str,
    tokens: tuple[int, ...],
    *,
    output_token_count: int = 0,
) -> RemoteMTPCandidate:
    return RemoteMTPCandidate(
        candidate_id=f"candidate-{rid}-{output_token_count}",
        request=request(rid, output_token_count),
        prefix_index_digest=hashlib.sha256(
            f"{rid}:{output_token_count}".encode()
        ).hexdigest(),
        token_ids=tokens,
    )


class TestBoundedRemoteMTPMailbox(unittest.TestCase):
    def tearDown(self):
        _reset_remote_mtp_candidate_source_factory_for_test()

    def test_scheduler_advice_is_observe_only_and_exactly_covers_the_seal(self):
        advice = RemoteMTPSchedulerAdvice(
            requests=(request("a"), request("b")),
            preferred_request_ids=("b",),
            compatible_depth=2,
            reason="candidate_ready_subset",
            seal_monotonic_ns=100,
        ).validate()
        self.assertEqual(advice.preferred_request_ids, ("b",))
        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPSchedulerAdvice(
                requests=(request("a"),),
                preferred_request_ids=("missing",),
                compatible_depth=2,
                reason="bad_subset",
                seal_monotonic_ns=100,
            ).validate()

        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPSchedulerAdvice(
                requests=(request("a"),),
                preferred_request_ids=("a",),
                compatible_depth=2,
                reason="premature_enforcement",
                seal_monotonic_ns=100,
                enforce=True,
            ).validate()
        enforced = RemoteMTPSchedulerAdvice(
            requests=(request("a"), request("b")),
            preferred_request_ids=("b",),
            compatible_depth=2,
            reason="committed_target_plan",
            seal_monotonic_ns=100,
            enforce=True,
            plan_id="plan",
            snapshot_generation=7,
            window_id="window",
            candidate_ids=("candidate-b",),
        ).validate()
        self.assertTrue(enforced.enforce)
        partition = build_remote_mtp_decode_slice(
            enforced.requests,
            enforced,
            configured_depth=2,
        )
        self.assertEqual(partition.selected_indices, (1,))
        self.assertEqual(partition.deferred_indices, (0,))
        self.assertEqual(partition.candidate_ids, ("candidate-b",))
        mixed = build_remote_mtp_mixed_batch_plan(
            enforced.requests,
            enforced,
            configured_depth=2,
        )
        self.assertEqual(mixed.remote_indices, (1,))
        self.assertEqual(mixed.local_indices, (0,))
        self.assertEqual(mixed.candidate_ids_by_row, (None, "candidate-b"))
        self.assertEqual(mixed.selected_depth, 2)
        self.assertEqual(mixed.plan_id, "plan")
        with self.assertRaises(RemoteMTPContractError):
            build_remote_mtp_decode_slice(
                enforced.requests,
                RemoteMTPSchedulerAdvice(
                    requests=enforced.requests,
                    preferred_request_ids=("b", "a"),
                    compatible_depth=2,
                    reason="reordered",
                    seal_monotonic_ns=100,
                    enforce=True,
                    plan_id="plan",
                    snapshot_generation=7,
                    window_id="window",
                    candidate_ids=("candidate-b", "candidate-a"),
                ),
                configured_depth=2,
            )
        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPSchedulerAdvice(
                requests=(request("a"),),
                preferred_request_ids=("a",),
                compatible_depth=2,
                reason="missing_candidate_ownership",
                seal_monotonic_ns=100,
                enforce=True,
                plan_id="plan",
                snapshot_generation=7,
                window_id="window",
            ).validate()

    def test_scheduler_advisor_factory_is_single_owner_and_optional(self):
        self.assertIsNone(create_remote_mtp_scheduler_advisor(object(), 0))

        class Advisor:
            def observe_decode_seal(self, requests, *, seal_monotonic_ns):
                return RemoteMTPSchedulerAdvice(
                    requests=tuple(requests),
                    preferred_request_ids=(),
                    compatible_depth=None,
                    reason=f"seal:{seal_monotonic_ns}",
                    seal_monotonic_ns=seal_monotonic_ns,
                )

        advisor = Advisor()

        def factory(server_args, gpu_id):
            del server_args, gpu_id
            return advisor

        install_remote_mtp_scheduler_advisor_factory(factory)
        self.assertIs(create_remote_mtp_scheduler_advisor(object(), 0), advisor)
        with self.assertRaises(RemoteMTPContractError):
            install_remote_mtp_scheduler_advisor_factory(lambda args, gpu: None)

    def test_engine_forces_request_before_maximum_gap_by_completion_guard(self):
        forced = RemoteMTPRequestView(
            "a",
            "inc-a",
            4,
            0,
            last_target_service_monotonic_ns=10,
            max_service_gap_ns=90,
            service_completion_guard_ns=20,
        ).validate()
        normal = request("b")
        advice = RemoteMTPSchedulerAdvice(
            requests=(forced, normal),
            preferred_request_ids=("b",),
            compatible_depth=2,
            reason="omitted_forced_row",
            seal_monotonic_ns=85,
            enforce=True,
            plan_id="plan",
            snapshot_generation=7,
            window_id="window",
            candidate_ids=("candidate-b",),
        ).validate()
        with self.assertRaisesRegex(
            RemoteMTPContractError, "engine-forced maximum-gap"
        ):
            build_remote_mtp_decode_slice(
                advice.requests,
                advice,
                configured_depth=2,
            )
        # The fused hybrid path does not defer the forced row: it runs local
        # MTP for that row in the same target execution as the preferred row.
        mixed = build_remote_mtp_mixed_batch_plan(
            advice.requests,
            advice,
            configured_depth=2,
        )
        self.assertEqual(mixed.remote_indices, (1,))
        self.assertEqual(mixed.local_indices, (0,))

    def test_mixed_batch_selects_one_exact_depth_up_to_configured_maximum(self):
        requests = (request("a"), request("b"))
        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                advice = RemoteMTPSchedulerAdvice(
                    requests=requests,
                    preferred_request_ids=("a",),
                    compatible_depth=depth,
                    reason="adaptive_exact_depth",
                    seal_monotonic_ns=100,
                    enforce=True,
                    plan_id=f"plan-k{depth}",
                    snapshot_generation=depth,
                    window_id=f"window-k{depth}",
                    candidate_ids=(f"candidate-k{depth}",),
                ).validate()
                mixed = build_remote_mtp_mixed_batch_plan(
                    requests,
                    advice,
                    configured_depth=4,
                )
                self.assertEqual(mixed.selected_depth, depth)
                self.assertEqual(mixed.candidate_ids_by_row, (f"candidate-k{depth}", None))

        with self.assertRaisesRegex(RemoteMTPContractError, "maximum"):
            build_remote_mtp_mixed_batch_plan(
                requests,
                RemoteMTPSchedulerAdvice(
                    requests=requests,
                    preferred_request_ids=("a",),
                    compatible_depth=4,
                    reason="too_deep",
                    seal_monotonic_ns=100,
                    enforce=True,
                    plan_id="plan",
                    snapshot_generation=1,
                    window_id="window",
                    candidate_ids=("candidate",),
                ).validate(),
                configured_depth=3,
            )

    def test_enforced_empty_subset_selects_all_local_depth_for_hybrid_only(self):
        requests = (request("a"), request("b"))
        advice = RemoteMTPSchedulerAdvice(
            requests=requests,
            preferred_request_ids=(),
            compatible_depth=3,
            reason="adaptive_all_local",
            seal_monotonic_ns=100,
            enforce=True,
            plan_id="plan-k3",
            snapshot_generation=3,
            window_id="window-k3",
            candidate_ids=(),
        ).validate()

        mixed = build_remote_mtp_mixed_batch_plan(
            requests,
            advice,
            configured_depth=4,
        )

        self.assertEqual(mixed.remote_indices, ())
        self.assertEqual(mixed.local_indices, (0, 1))
        self.assertEqual(mixed.candidate_ids_by_row, (None, None))
        self.assertEqual(mixed.selected_depth, 3)
        with self.assertRaisesRegex(RemoteMTPContractError, "non-empty"):
            build_remote_mtp_decode_slice(
                requests,
                advice,
                configured_depth=3,
            )

    def test_service_window_partition_keeps_unpaced_work_runnable(self):
        paced_early = RemoteMTPRequestView(
            "paced-early",
            "inc-paced-early",
            4,
            0,
            last_target_service_monotonic_ns=100,
            service_period_ns=500,
        ).validate()
        paced_due = RemoteMTPRequestView(
            "paced-due",
            "inc-paced-due",
            4,
            0,
            last_target_service_monotonic_ns=10,
            service_period_ns=500,
        ).validate()
        eligible, deferred, deadline = remote_mtp_service_window_partition(
            (paced_early, request("unpaced"), paced_due),
            now_monotonic_ns=550,
        )
        self.assertEqual(eligible, (1, 2))
        self.assertEqual(deferred, (0,))
        self.assertEqual(deadline, 600)

        eligible, deferred, deadline = remote_mtp_service_window_partition(
            (paced_early,),
            now_monotonic_ns=600,
        )
        self.assertEqual(eligible, (0,))
        self.assertEqual(deferred, ())
        self.assertIsNone(deadline)

    def test_linear_chain_layout_matches_native_eagle_topk1(self):
        self.assertEqual(remote_mtp_linear_chain_layout(1), ((), (0,)))
        self.assertEqual(remote_mtp_linear_chain_layout(2), ((-1, 0), (0, 1)))
        self.assertEqual(
            remote_mtp_linear_chain_layout(3),
            ((-1, 0, 1), (0, 1, 2)),
        )

    def test_rejoin_order_preserves_every_surviving_native_permutation(self):
        native = (("a", "1"), ("b", "1"), ("c", "1"))
        for live_size in range(4):
            for surviving in itertools.combinations(native, live_size):
                for permuted in itertools.permutations(surviving):
                    live = permuted + permuted[:1] + (("new", "1"),)
                    self.assertEqual(
                        remote_mtp_rejoin_order(native, live),
                        surviving + (("new", "1"),),
                    )

    def test_completion_guard_forces_before_expiry_but_not_before_guard_window(self):
        guarded = RemoteMTPRequestView(
            "a",
            "inc-a",
            4,
            0,
            last_target_service_monotonic_ns=10,
            max_service_gap_ns=90,
            service_completion_guard_ns=20,
        ).validate()
        for seal, must_force in ((79, False), (80, True), (99, True)):
            preferred = ("a",) if must_force else ("b",)
            candidates = ("candidate-a",) if must_force else ("candidate-b",)
            advice = RemoteMTPSchedulerAdvice(
                requests=(guarded, request("b")),
                preferred_request_ids=preferred,
                compatible_depth=2,
                reason="guard_property",
                seal_monotonic_ns=seal,
                enforce=True,
                plan_id="plan",
                snapshot_generation=7,
                window_id="window",
                candidate_ids=candidates,
            ).validate()
            if must_force:
                build_remote_mtp_decode_slice(
                    advice.requests,
                    advice,
                    configured_depth=2,
                )
            else:
                build_remote_mtp_decode_slice(
                    advice.requests,
                    advice,
                    configured_depth=2,
                )

    def test_target_feature_batch_requires_verify_accept_lengths(self):
        claim = BoundedRemoteMTPMailbox(capacity=1)
        claim.offer(candidate("a", (10, 11)))
        claimed = claim.try_claim_batch((request("a"),), depth=2)
        feature = RemoteMTPTargetFeatureBatch(
            feature_batch_id="tp0:forward1",
            phase="verify",
            requests=(request("a"),),
            target_hidden_states=object(),
            next_token_ids=object(),
            accept_lens=object(),
            new_seq_lens=object(),
            claim=claimed,
            fallback_reason=None,
            input_token_ids=None,
            extend_lens=None,
            prefix_lens=None,
            row_stride=3,
            completion_event=None,
        ).validate()
        self.assertEqual(feature.claim.candidate_ids, ("candidate-a-0",))
        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPTargetFeatureBatch(
                feature_batch_id="tp0:forward2",
                phase="verify",
                requests=(request("a"),),
                target_hidden_states=object(),
                next_token_ids=object(),
                accept_lens=None,
                new_seq_lens=object(),
                claim=None,
                fallback_reason="candidate_batch_absent",
                input_token_ids=None,
                extend_lens=None,
                prefix_lens=None,
                row_stride=1,
                completion_event=None,
            ).validate()

    def test_target_settlement_covers_remote_and_ar_progress(self):
        remote = RemoteMTPTargetSettlement(
            feature_batch_id="tp0:forward1",
            phase="verify",
            request=request("a"),
            executed_source="remote_mtp",
            committed_token_ids=(10, 99),
            candidate_id="candidate-a-0",
            prefix_index_digest="a" * 64,
            candidate_token_ids=(10, 11),
            verified_depth=2,
            accepted_draft_count=1,
        ).validate()
        self.assertEqual(remote.accepted_draft_count, 1)
        planned = RemoteMTPTargetSettlement(
            feature_batch_id="tp0:forward1",
            phase="verify",
            request=request("a"),
            executed_source="remote_mtp",
            committed_token_ids=(10, 99),
            candidate_id="candidate-a-0",
            prefix_index_digest="a" * 64,
            candidate_token_ids=(10, 11),
            verified_depth=2,
            accepted_draft_count=1,
            plan_id="plan",
            snapshot_generation=7,
            window_id="window",
        ).validate()
        self.assertEqual(planned.plan_id, "plan")
        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPTargetSettlement(
                feature_batch_id="tp0:forward1",
                phase="verify",
                request=request("a"),
                executed_source="remote_mtp",
                committed_token_ids=(10, 99),
                candidate_id="candidate-a-0",
                prefix_index_digest="a" * 64,
                candidate_token_ids=(10, 11),
                verified_depth=2,
                accepted_draft_count=1,
                plan_id="partial",
            ).validate()

        fallback = RemoteMTPTargetSettlement(
            feature_batch_id="tp0:forward2",
            phase="verify",
            request=request("a", 2),
            executed_source="autoregressive",
            committed_token_ids=(22,),
            fallback_reason="candidate_batch_absent",
        ).validate()
        self.assertEqual(fallback.executed_source, "autoregressive")

        local = RemoteMTPTargetSettlement(
            feature_batch_id="tp0:forward-local",
            phase="verify",
            request=request("a", 2),
            executed_source="local_mtp",
            committed_token_ids=(10, 99),
            verified_depth=3,
            accepted_draft_count=1,
            fallback_reason="candidate_batch_absent",
        ).validate()
        self.assertEqual(local.executed_source, "local_mtp")

        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPTargetSettlement(
                feature_batch_id="tp0:forward3",
                phase="verify",
                request=request("a", 3),
                executed_source="autoregressive",
                committed_token_ids=(23,),
                candidate_id="forbidden",
            ).validate()

    def test_verification_outcome_binds_claim_and_committed_run(self):
        outcome = RemoteMTPVerificationOutcome(
            candidate_id="candidate-a-0",
            request=request("a"),
            prefix_index_digest="a" * 64,
            candidate_token_ids=(10, 11, 12),
            accepted_draft_count=2,
            committed_token_ids=(10, 11, 99),
        ).validate()
        self.assertEqual(outcome.accepted_draft_count, 2)
        with self.assertRaises(RemoteMTPContractError):
            RemoteMTPVerificationOutcome(
                candidate_id="candidate-a-0",
                request=request("a"),
                prefix_index_digest="a" * 64,
                candidate_token_ids=(10, 11, 12),
                accepted_draft_count=4,
                committed_token_ids=(10, 11, 12, 99),
            ).validate()

    def test_overlap_boundary_is_newer_than_processed_request_outputs(self):
        self.assertEqual(
            remote_mtp_output_token_count(
                prompt_token_count=4096,
                req_output_token_count=3,
                kv_boundary=4103,
            ),
            8,
        )
        self.assertEqual(
            remote_mtp_output_token_count(
                prompt_token_count=4096,
                req_output_token_count=3,
                kv_boundary=None,
            ),
            3,
        )

    def test_committed_kv_boundary_includes_the_relayed_root_semantically(self):
        self.assertEqual(
            remote_mtp_output_token_count(
                prompt_token_count=90,
                req_output_token_count=0,
                kv_boundary=90,
            ),
            1,
        )

    def test_v0516_local_mtp_boundaries_advance_by_committed_runs(self):
        observed = tuple(
            remote_mtp_output_token_count(
                prompt_token_count=6,
                req_output_token_count=0,
                kv_boundary=boundary,
            )
            for boundary in (6, 9, 11)
        )
        self.assertEqual(observed, (1, 4, 6))

    def test_complete_batch_claim_is_ordered_and_consume_once(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=4)
        self.assertTrue(mailbox.offer(candidate("b", (20, 21))))
        self.assertTrue(mailbox.offer(candidate("a", (10, 11))))

        claim = mailbox.try_claim_batch((request("a"), request("b")), depth=2)

        self.assertIsNotNone(claim)
        self.assertEqual(claim.token_ids, ((10, 11), (20, 21)))
        self.assertEqual(claim.candidate_ids, ("candidate-a-0", "candidate-b-0"))
        self.assertIsNone(
            mailbox.try_claim_batch((request("a"), request("b")), depth=2)
        )
        self.assertEqual(mailbox.stats().claimed, 2)
        self.assertEqual(mailbox.stats().resident, 0)

    def test_planned_candidate_ids_are_exact_and_cannot_substitute(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=4)
        self.assertTrue(mailbox.offer(candidate("a", (10, 11))))
        self.assertTrue(mailbox.offer(candidate("b", (20, 21))))
        self.assertIsNone(
            mailbox.try_claim_batch(
                (request("a"),), depth=2, candidate_ids=("candidate-b-0",)
            )
        )
        self.assertEqual(mailbox.stats().resident, 2)
        claim = mailbox.try_claim_batch(
            (request("a"),),
            depth=2,
            candidate_ids=("candidate-a-0",),
            plan_id="plan",
            snapshot_generation=7,
            window_id="window",
        )
        self.assertEqual(claim.candidate_ids, ("candidate-a-0",))
        with self.assertRaises(RemoteMTPContractError):
            mailbox.try_claim_batch(
                (request("b"),),
                depth=2,
                candidate_ids=("candidate-b-0",),
                plan_id="incomplete-plan-identity",
            )

    def test_incomplete_batch_does_not_partially_consume(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=4)
        self.assertTrue(mailbox.offer(candidate("a", (10, 11))))

        self.assertIsNone(
            mailbox.try_claim_batch((request("a"), request("b")), depth=2)
        )
        self.assertEqual(mailbox.stats().resident, 1)
        claim = mailbox.try_claim_batch((request("a"),), depth=2)
        self.assertEqual(claim.candidate_ids, ("candidate-a-0",))

    def test_stale_output_boundary_cannot_match(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=2)
        self.assertTrue(mailbox.offer(candidate("a", (10, 11), output_token_count=4)))

        self.assertIsNone(mailbox.try_claim_batch((request("a", 5),), depth=2))
        self.assertEqual(mailbox.stats().resident, 1)
        self.assertIsNotNone(mailbox.try_claim_batch((request("a", 4),), depth=2))

    def test_wrong_depth_cannot_consume(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=2)
        self.assertTrue(mailbox.offer(candidate("a", (10, 11))))
        self.assertIsNone(mailbox.try_claim_batch((request("a"),), depth=3))
        self.assertEqual(mailbox.stats().resident, 1)

    def test_duplicate_and_overload_are_explicit(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=1)
        first = candidate("a", (10,))
        self.assertTrue(mailbox.offer(first))
        self.assertFalse(mailbox.offer(first))
        self.assertFalse(mailbox.offer(candidate("b", (11,))))

        stats = mailbox.stats()
        self.assertEqual(stats.duplicate_rejected, 1)
        self.assertEqual(stats.overload_rejected, 1)
        self.assertEqual(stats.resident, 1)

    def test_invalid_tokens_fail_before_admission(self):
        mailbox = BoundedRemoteMTPMailbox(capacity=1)
        invalid = RemoteMTPCandidate(
            candidate_id="bad",
            request=request("a"),
            prefix_index_digest="0" * 64,
            token_ids=(-1,),
        )
        with self.assertRaises(RemoteMTPContractError):
            mailbox.offer(invalid)
        self.assertEqual(mailbox.stats().resident, 0)


if __name__ == "__main__":
    unittest.main()
