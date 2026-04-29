import json
import tempfile
import unittest
from pathlib import Path

from sglang.multimodal_gen.runtime.managers.dynamic_batch_admission import (
    load_batching_config,
)


class TestBatchingConfigValidation(unittest.TestCase):
    def _write_config(self, payload) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with tmp:
            json.dump(payload, tmp)
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_accepts_known_rule_keys_and_calibration_metadata(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "rules": [
                    {
                        "model_contains": "Qwen-Image-2512",
                        "resolution": "512x512",
                        "device_memory_gb_min": 120,
                        "device_memory_gb_max": 160,
                        "offload": False,
                        "max_batch_size": 8,
                        "max_cost": 1000,
                        "calibration": {
                            "hardware": "H200",
                            "source": "benchmark summary",
                        },
                    }
                ],
            }
        )

        rules = load_batching_config(path)

        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].max_batch_size, 8)

    def test_rejects_unknown_rule_key_with_hint(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "rules": [
                    {
                        "model_contains": "Qwen-Image-2512",
                        "resolution": "512x512",
                        "device_memory_min_gb": 120,
                        "max_batch_size": 8,
                    }
                ],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown key.*device_memory_min_gb.*device_memory_gb_min",
        ):
            load_batching_config(path)

    def test_rejects_unknown_rule_key_in_mapping_config(self):
        path = self._write_config(
            {
                "Qwen/Qwen-Image-2512|512x512": {
                    "max_batch_siz": 8,
                }
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown key.*max_batch_siz.*max_batch_size",
        ):
            load_batching_config(path)


if __name__ == "__main__":
    unittest.main()
