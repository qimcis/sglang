import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _FakeLoadedPipeline:
    def __init__(self):
        self.transformer = None
        self.unet = None
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeDiffusionPipeline:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.calls.append((model_path, kwargs))
        return _FakeLoadedPipeline()


class _FakeTransformer:
    calls = []

    @classmethod
    def from_single_file(cls, file_path, **kwargs):
        cls.calls.append((file_path, kwargs))
        return {"file_path": file_path, "kwargs": kwargs}


class _FakeToTensor:
    def __call__(self, _image):
        return torch.zeros((3, 1, 1), dtype=torch.float32)


def _install_import_stubs():
    diffusers_stub = types.ModuleType("diffusers")
    diffusers_stub.DiffusionPipeline = _FakeDiffusionPipeline
    diffusers_stub.FakeTransformer = _FakeTransformer

    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.ToTensor = _FakeToTensor
    torchvision_stub.transforms = transforms_stub

    originals = {
        "diffusers": sys.modules.get("diffusers"),
        "torchvision": sys.modules.get("torchvision"),
        "torchvision.transforms": sys.modules.get("torchvision.transforms"),
    }
    sys.modules["diffusers"] = diffusers_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub
    return originals


def _restore_import_stubs(originals):
    for name, module in originals.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestDiffusersGGUFLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = _install_import_stubs()
        module_name = "sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline"
        sys.modules.pop(module_name, None)
        cls.diffusers_pipeline_module = importlib.import_module(module_name)

    @classmethod
    def tearDownClass(cls):
        _restore_import_stubs(cls._original_modules)

    def setUp(self):
        _FakeDiffusionPipeline.calls.clear()
        _FakeTransformer.calls.clear()

    @staticmethod
    def _make_server_args(**overrides):
        base = {
            "gguf_file": "weights.gguf",
            "gguf_base_model_path": "black-forest-labs/FLUX.1-dev",
            "trust_remote_code": False,
            "revision": None,
            "attention_backend": None,
            "cache_dit_config": None,
            "pipeline_config": SimpleNamespace(
                quantization_config=None,
                dit_precision="fp16",
                vae_slicing=False,
                vae_tiling=False,
            ),
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_loads_gguf_component_and_injects_into_from_pretrained(self):
        module = self.diffusers_pipeline_module
        pipeline = module.DiffusersPipeline.__new__(module.DiffusersPipeline)
        server_args = self._make_server_args()

        with (
            patch.object(
                module, "maybe_download_model", return_value="/resolved/base_model"
            ),
            patch.object(
                module,
                "maybe_download_model_index",
                return_value={"transformer": ["diffusers", "FakeTransformer"]},
            ),
            patch.object(
                module, "check_gguf_file", side_effect=lambda path: str(path).endswith(".gguf")
            ),
            patch.object(module, "get_local_torch_device", return_value="cpu"),
        ):
            loaded = pipeline._load_diffusers_pipeline(
                "black-forest-labs/FLUX.1-dev", server_args
            )

        self.assertIsNotNone(loaded)
        self.assertEqual(len(_FakeTransformer.calls), 1)
        gguf_file_path, _ = _FakeTransformer.calls[0]
        self.assertEqual(gguf_file_path, "/resolved/base_model/weights.gguf")

        self.assertEqual(len(_FakeDiffusionPipeline.calls), 1)
        model_path, kwargs = _FakeDiffusionPipeline.calls[0]
        self.assertEqual(model_path, "/resolved/base_model")
        self.assertIn("transformer", kwargs)
        self.assertEqual(kwargs["torch_dtype"], torch.float16)

    def test_requires_base_model_when_model_path_is_gguf(self):
        module = self.diffusers_pipeline_module
        pipeline = module.DiffusersPipeline.__new__(module.DiffusersPipeline)
        server_args = self._make_server_args(
            gguf_file=None,
            gguf_base_model_path=None,
        )

        with patch.object(
            module, "check_gguf_file", side_effect=lambda path: str(path).endswith(".gguf")
        ):
            with self.assertRaisesRegex(ValueError, "gguf-base-model-path"):
                pipeline._resolve_gguf_settings("flux-dev-q4.gguf", server_args)


if __name__ == "__main__":
    unittest.main()
