import importlib.util
import pathlib
import sys
import types
import unittest

try:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo
except ModuleNotFoundError:
    root = pathlib.Path(__file__).parents[1]
    for name in ("sglang", "sglang.srt", "sglang.srt.speculative"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault("torch", types.ModuleType("torch"))

    registry_name = "sglang.srt.speculative.spec_registry"
    registry_spec = importlib.util.spec_from_file_location(
        registry_name,
        root / "srt" / "speculative" / "spec_registry.py",
    )
    registry_module = importlib.util.module_from_spec(registry_spec)
    sys.modules[registry_name] = registry_module
    registry_spec.loader.exec_module(registry_module)

    info_name = "sglang.srt.speculative.spec_info"
    info_spec = importlib.util.spec_from_file_location(
        info_name,
        root / "srt" / "speculative" / "spec_info.py",
    )
    info_module = importlib.util.module_from_spec(info_spec)
    sys.modules[info_name] = info_module
    info_spec.loader.exec_module(info_module)
    SpeculativeAlgorithm = info_module.SpeculativeAlgorithm
    CustomSpecAlgo = registry_module.CustomSpecAlgo


class TestRemoteMTPSpecInfo(unittest.TestCase):
    def test_remote_mtp_reuses_eagle_verify_but_owns_no_draft_kv(self):
        algorithm = SpeculativeAlgorithm.from_string("remote_mtp")
        self.assertTrue(algorithm.is_remote_mtp())
        self.assertTrue(algorithm.is_eagle())
        self.assertFalse(algorithm.has_draft_kv())
        self.assertFalse(algorithm.carries_draft_hidden_states())
        self.assertTrue(algorithm.need_topk())

    def test_custom_algorithm_contract_has_remote_predicate(self):
        self.assertFalse(
            CustomSpecAlgo(
                name="CUSTOM",
                factory=lambda server_args: object,
            ).is_remote_mtp()
        )

    def test_hybrid_remote_mtp_retains_local_draft_state(self):
        algorithm = SpeculativeAlgorithm.from_string("remote_mtp_local")
        self.assertTrue(algorithm.is_remote_mtp())
        self.assertTrue(algorithm.has_local_mtp_fallback())
        self.assertTrue(algorithm.has_draft_kv())
        self.assertTrue(algorithm.carries_draft_hidden_states())


if __name__ == "__main__":
    unittest.main()
