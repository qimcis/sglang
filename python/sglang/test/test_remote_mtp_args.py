import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

module_path = (
    pathlib.Path(__file__).parents[1] / "srt" / "arg_groups" / "speculative_hook.py"
)
spec = importlib.util.spec_from_file_location(
    "speculative_hook_under_test", module_path
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
_handle_remote_mtp = module._handle_remote_mtp


def args(**overrides):
    values = {
        "pp_size": 1,
        "speculative_algorithm": "REMOTE_MTP",
        "enable_multi_layer_eagle": False,
        "speculative_adaptive": False,
        "speculative_use_rejection_sampling": False,
        "speculative_draft_model_path": None,
        "speculative_num_steps": None,
        "speculative_eagle_topk": None,
        "speculative_num_draft_tokens": None,
        "max_running_requests": None,
        "device": "cuda",
        "disable_overlap_schedule": False,
        "enable_mixed_chunk": True,
        "chunked_prefill_size": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestRemoteMTPArgs(unittest.TestCase):
    def test_default_is_fixed_k2_target_only_profile(self):
        server_args = args()

        _handle_remote_mtp(server_args)

        self.assertEqual(server_args.speculative_num_steps, 2)
        self.assertEqual(server_args.speculative_eagle_topk, 1)
        self.assertEqual(server_args.speculative_num_draft_tokens, 3)
        self.assertEqual(server_args.max_running_requests, 48)
        self.assertEqual(server_args.chunked_prefill_size, -1)
        self.assertFalse(server_args.enable_mixed_chunk)
        self.assertFalse(server_args.disable_overlap_schedule)

    def test_explicit_k1_and_k3_are_supported(self):
        for depth in (1, 3):
            with self.subTest(depth=depth):
                server_args = args(
                    speculative_num_steps=depth,
                    speculative_num_draft_tokens=depth + 1,
                )
                _handle_remote_mtp(server_args)
                self.assertEqual(server_args.speculative_num_steps, depth)

    def test_cpu_profile_disables_overlap(self):
        server_args = args(device="cpu")
        _handle_remote_mtp(server_args)
        self.assertTrue(server_args.disable_overlap_schedule)

    def test_local_draft_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not set"):
            _handle_remote_mtp(args(speculative_draft_model_path="target"))

    def test_hybrid_profile_retains_local_mtp_and_defaults_to_k3(self):
        original = module._handle_eagle_family

        def configure_local(server_args):
            server_args.speculative_draft_model_path = "target"

        module._handle_eagle_family = configure_local
        try:
            server_args = args(speculative_algorithm="REMOTE_MTP_LOCAL")
            _handle_remote_mtp(server_args)
        finally:
            module._handle_eagle_family = original

        self.assertEqual(server_args.speculative_num_steps, 3)
        self.assertEqual(server_args.speculative_num_draft_tokens, 4)
        self.assertEqual(server_args.speculative_draft_model_path, "target")

    def test_incompatible_modes_are_rejected(self):
        incompatible = (
            {"pp_size": 2},
            {"enable_multi_layer_eagle": True},
            {"speculative_adaptive": True},
            {"speculative_use_rejection_sampling": True},
            {"speculative_num_steps": 4},
            {"speculative_eagle_topk": 2},
            {
                "speculative_num_steps": 2,
                "speculative_num_draft_tokens": 2,
            },
        )
        for overrides in incompatible:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _handle_remote_mtp(args(**overrides))


if __name__ == "__main__":
    unittest.main()
