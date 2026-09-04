from __future__ import annotations

import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1] / "srt"


def parse(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(), filename=relative_path)


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def method_node(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class TestRemoteMTPSchedulerSourceContract(unittest.TestCase):
    def test_remote_only_ar_fallback_uses_int64_verifier_tokens(self):
        owner = class_node(
            parse("speculative/remote_mtp_worker_v2.py"),
            "RemoteMTPWorkerV2",
        )
        method = method_node(owner, "_build_trivial_verify_input")
        assignment = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.keyword) and node.arg == "draft_token"
        )
        self.assertIsInstance(assignment.value, ast.Call)
        assert isinstance(assignment.value, ast.Call)
        self.assertIsInstance(assignment.value.func, ast.Attribute)
        assert isinstance(assignment.value.func, ast.Attribute)
        self.assertEqual(assignment.value.func.attr, "to")
        self.assertTrue(
            any(
                isinstance(argument, ast.Attribute)
                and isinstance(argument.value, ast.Name)
                and argument.value.id == "torch"
                and argument.attr == "int64"
                for argument in assignment.value.args
            )
        )

    def test_remote_only_worker_does_not_require_a_local_draft_kv_pool(self):
        tree = parse("mem_cache/kv_cache_builder.py")
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_draft_kv_pool"
        )
        calls = {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("has_draft_kv", calls)

    def test_schedule_batch_carries_exact_target_plan_ownership(self):
        owner = class_node(parse("managers/schedule_batch.py"), "ScheduleBatch")
        fields = {
            node.target.id
            for node in owner.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue(
            {
                "remote_mtp_target_plan_id",
                "remote_mtp_target_plan_generation",
                "remote_mtp_target_window_id",
                "remote_mtp_candidate_ids",
                "remote_mtp_detached_decode",
            }
            <= fields
        )

    def test_scheduler_owns_detach_rejoin_and_fail_open_slice_methods(self):
        owner = class_node(parse("managers/scheduler.py"), "Scheduler")
        methods = {
            node.name
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(
            {
                "_build_resident_decode_batch",
                "_slice_remote_mtp_relay_input",
                "_rejoin_remote_mtp_detached_decode",
                "apply_remote_mtp_service_windows",
                "_merge_remote_mtp_deferred_batches",
                "observe_remote_mtp_decode_seal",
                "apply_remote_mtp_decode_advice",
            }
            <= methods
        )
        apply = method_node(owner, "apply_remote_mtp_decode_advice")
        calls = {
            node.func.attr
            for node in ast.walk(apply)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        names = {
            node.func.id
            for node in ast.walk(apply)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("prepare_for_decode", calls)
        self.assertIn("build_remote_mtp_decode_slice", names)

    def test_resident_remote_batch_stashes_complete_speculative_relay(self):
        owner = class_node(parse("managers/scheduler.py"), "Scheduler")
        method = method_node(owner, "_build_resident_decode_batch")
        calls = {
            (node.func.value.id, node.func.attr)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        self.assertIn(("RelayPayload", "from_draft_input"), calls)
        slicer = method_node(owner, "_slice_remote_mtp_relay_input")
        slicer_calls = {
            node.func.attr
            for node in ast.walk(slicer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("filter_batch", slicer_calls)
        self.assertIn("merge_batch", slicer_calls)

        rejoin = method_node(owner, "_rejoin_remote_mtp_detached_decode")
        rebuild = next(
            node
            for node in ast.walk(rejoin)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_build_resident_decode_batch"
        )
        source_batches = next(
            keyword.value
            for keyword in rebuild.keywords
            if keyword.arg == "source_batches"
        )
        self.assertIsInstance(source_batches, ast.Tuple)
        assert isinstance(source_batches, ast.Tuple)
        self.assertEqual(
            [item.id for item in source_batches.elts if isinstance(item, ast.Name)],
            ["running_batch", "last_batch"],
        )

    def test_native_claim_forwards_complete_plan_identity(self):
        owner = class_node(
            parse("speculative/remote_mtp_worker_v2.py"),
            "RemoteMTPWorkerV2",
        )
        method = method_node(owner, "_try_remote_verify_input")
        claim = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "try_claim_batch"
        )
        keywords = {item.arg for item in claim.keywords}
        self.assertTrue(
            {
                "depth",
                "candidate_ids",
                "plan_id",
                "snapshot_generation",
                "window_id",
            }
            <= keywords
        )

    def test_cpu_settlement_preserves_complete_plan_identity(self):
        owner = class_node(
            parse("speculative/remote_mtp_worker_v2.py"),
            "RemoteMTPWorkerV2",
        )
        method = method_node(owner, "on_remote_mtp_result_cpu")
        settlement = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RemoteMTPTargetSettlement"
        )
        keywords = {item.arg for item in settlement.keywords}
        self.assertTrue(
            {
                "plan_id",
                "snapshot_generation",
                "window_id",
                "verified_depth",
            }
            <= keywords
        )

    def test_finished_request_is_offered_to_external_state_owner(self):
        owner = class_node(
            parse("speculative/remote_mtp_worker_v2.py"),
            "RemoteMTPWorkerV2",
        )
        method = method_node(owner, "note_request_finished")
        string_values = {
            node.value for node in ast.walk(method) if isinstance(node, ast.Constant)
        }
        self.assertIn("offer_request_finished", string_values)
        self.assertIn("request_incarnation", {
            keyword.arg
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        })
        self.assertTrue(any(isinstance(node, ast.Try) for node in ast.walk(method)))
        self.assertTrue(
            any(
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
                for node in ast.walk(method)
            )
        )

    def test_hybrid_worker_keeps_local_draft_and_remote_first_branch(self):
        owner = class_node(
            parse("speculative/remote_mtp_local_worker_v2.py"),
            "RemoteMTPLocalWorkerV2",
        )
        init = method_node(owner, "__init__")
        forward = method_node(owner, "forward_batch_generation")
        init_calls = {
            node.func.id
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        forward_calls = {
            node.func.attr
            for node in ast.walk(forward)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("EagleDraftWorker", init_calls)
        self.assertIn("_try_remote_verify_input", forward_calls)
        self.assertIn("_draft_extend_for_decode", forward_calls)

    def test_hybrid_prefill_exports_the_unrotated_target_prompt(self):
        owner = class_node(
            parse("speculative/remote_mtp_local_worker_v2.py"),
            "RemoteMTPLocalWorkerV2",
        )
        forward = method_node(owner, "forward_batch_generation")
        prefill_branch = next(node for node in forward.body if isinstance(node, ast.If))
        original_prompt = next(
            node
            for node in prefill_branch.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prefill_input_ids"
                for target in node.targets
            )
        )
        self.assertIsInstance(original_prompt.value, ast.Attribute)
        assert isinstance(original_prompt.value, ast.Attribute)
        self.assertEqual(original_prompt.value.attr, "input_ids")
        publish = next(
            node
            for node in ast.walk(prefill_branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_publish_local_prefill"
        )
        exported_prompt = next(
            item.value for item in publish.keywords if item.arg == "prefill_input_ids"
        )
        self.assertIsInstance(exported_prompt, ast.Name)
        assert isinstance(exported_prompt, ast.Name)
        self.assertEqual(exported_prompt.id, "prefill_input_ids")

    def test_feature_publication_uses_pre_forward_request_boundaries(self):
        for relative_path, owner_name in (
            ("speculative/remote_mtp_worker_v2.py", "RemoteMTPWorkerV2"),
            (
                "speculative/remote_mtp_local_worker_v2.py",
                "RemoteMTPLocalWorkerV2",
            ),
        ):
            owner = class_node(parse(relative_path), owner_name)
            offers = [
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_offer_target_feature_batch"
            ]
            self.assertTrue(offers)
            self.assertTrue(
                all(
                    "request_views" in {keyword.arg for keyword in offer.keywords}
                    for offer in offers
                )
            )

    def test_settlement_reuses_pre_forward_request_boundaries(self):
        result = class_node(parse("managers/utils.py"), "GenerationBatchResult")
        result_fields = {
            node.target.id
            for node in result.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("remote_mtp_request_views", result_fields)

        base = class_node(
            parse("speculative/remote_mtp_worker_v2.py"),
            "RemoteMTPWorkerV2",
        )
        for method_name in (
            "on_remote_mtp_result_cpu",
            "on_remote_mtp_prefill_result_cpu",
        ):
            method = method_node(base, method_name)
            result_attributes = {
                node.attr
                for node in ast.walk(method)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "result"
            }
            self.assertIn("remote_mtp_request_views", result_attributes)

        for relative_path, owner_name in (
            ("speculative/remote_mtp_worker_v2.py", "RemoteMTPWorkerV2"),
            (
                "speculative/remote_mtp_local_worker_v2.py",
                "RemoteMTPLocalWorkerV2",
            ),
        ):
            owner = class_node(parse(relative_path), owner_name)
            assignments = [
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "remote_mtp_request_views"
                    for target in node.targets
                )
            ]
            self.assertTrue(assignments)
            self.assertTrue(
                all(
                    isinstance(assignment.value, ast.Name)
                    and assignment.value.id == "request_views"
                    for assignment in assignments
                )
            )

    def test_scheduler_seal_uses_one_engine_clock_sample(self):
        owner = class_node(parse("managers/scheduler.py"), "Scheduler")
        method = method_node(owner, "observe_remote_mtp_decode_seal")
        observation = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "observe_decode_seal"
        )
        timestamp = next(
            item.value
            for item in observation.keywords
            if item.arg == "seal_monotonic_ns"
        )
        self.assertIsInstance(timestamp, ast.Name)
        assert isinstance(timestamp, ast.Name)
        self.assertEqual(timestamp.id, "seal_monotonic_ns")
        assignments = [
            node
            for node in method.body
            if isinstance(node, ast.Try)
            for node in node.body
            if isinstance(node, ast.Assign)
        ]
        seal_assignment = next(
            node
            for node in assignments
            if any(
                isinstance(target, ast.Name) and target.id == "seal_monotonic_ns"
                for target in node.targets
            )
        )
        self.assertIsInstance(seal_assignment.value, ast.Call)
        assert isinstance(seal_assignment.value, ast.Call)
        self.assertIsInstance(seal_assignment.value.func, ast.Attribute)
        assert isinstance(seal_assignment.value.func, ast.Attribute)
        self.assertEqual(seal_assignment.value.func.attr, "monotonic_ns")

    def test_detached_decode_carries_and_restores_native_request_order(self):
        batch = class_node(parse("managers/schedule_batch.py"), "ScheduleBatch")
        fields = {
            node.target.id
            for node in batch.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("remote_mtp_native_order", fields)
        scheduler = class_node(parse("managers/scheduler.py"), "Scheduler")
        apply = method_node(scheduler, "apply_remote_mtp_decode_advice")
        rejoin = method_node(scheduler, "_rejoin_remote_mtp_detached_decode")
        apply_attrs = {
            node.attr for node in ast.walk(apply) if isinstance(node, ast.Attribute)
        }
        rejoin_attrs = {
            node.attr for node in ast.walk(rejoin) if isinstance(node, ast.Attribute)
        }
        self.assertIn("remote_mtp_native_order", apply_attrs)
        self.assertIn("remote_mtp_native_order", rejoin_attrs)

    def test_ignore_eos_prefill_honors_the_dsv4_planner_token_budget(self):
        owner = class_node(parse("managers/schedule_policy.py"), "PrefillAdder")
        method = method_node(owner, "add_one_req_ignore_eos")
        attributes = {
            node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
        }
        self.assertIn("rem_input_tokens", attributes)
        self.assertIn("can_run_list", attributes)
        self.assertTrue(
            any(
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "AddReqResult"
                and node.value.attr == "OTHER"
                for node in ast.walk(method)
            )
        )


if __name__ == "__main__":
    unittest.main()
