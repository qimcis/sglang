import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKVProtectionBackendTraversal(CustomTestCase):
    @staticmethod
    def _make_logits_processor(gather_call):
        processor = object.__new__(LogitsProcessor)
        processor.kv_protection_enabled = True
        processor.logit_scale = None
        processor.use_attn_tp_group = False
        processor.do_tensor_parallel_all_gather = True
        processor.do_tensor_parallel_all_gather_dp_attn = True
        processor.final_logit_softcapping = None
        processor._gather_dp_attn_hidden_states = lambda hidden, metadata: (
            torch.zeros(3, 4),
            torch.empty(0, 4),
        )
        processor._compute_lm_head = lambda hidden, lm_head, bias: torch.zeros(3, 8)
        processor._logits_gatherer = gather_call
        processor._scatter_dp_attn_logits = (
            lambda logits, local_hidden, metadata, **kwargs: logits
        )
        processor._copy_logits_to_buffer = lambda logits, metadata: logits
        return processor

    def test_result_discovery_reaches_hybrid_decode_backend(self):
        protection = {"validated_in_metadata": True}
        dsa_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(kv_page_protection=protection)
        )
        hybrid_backend = SimpleNamespace(decode_backend=dsa_backend)
        tbo_backend = SimpleNamespace(primary=hybrid_backend, children=[])
        runner = SimpleNamespace(
            attn_backend=tbo_backend,
            decode_attn_backend_group=[],
            decode_attn_backend=None,
        )

        self.assertIs(ModelRunner._get_active_kv_page_protection(runner), protection)

    def test_result_discovery_prefers_active_pdmux_backend(self):
        active_protection = {"source": "active"}
        stale_protection = {"source": "stale"}
        active = SimpleNamespace(
            forward_metadata=SimpleNamespace(kv_page_protection=active_protection)
        )
        stale = SimpleNamespace(
            forward_metadata=SimpleNamespace(kv_page_protection=stale_protection)
        )
        runner = SimpleNamespace(
            attn_backend=SimpleNamespace(),
            decode_attn_backend_group=[stale, active],
            decode_attn_backend=active,
        )

        self.assertIs(
            ModelRunner._get_active_kv_page_protection(runner),
            active_protection,
        )

    def test_graph_bank_reaches_tbo_and_hybrid_children(self):
        class FakeBackend:
            def __init__(self):
                self.banks = []

            def set_kv_protection_graph_bank(self, bank):
                self.banks.append(bank)

        primary = FakeBackend()
        tbo_children = [FakeBackend(), FakeBackend()]
        hybrid_decode = FakeBackend()
        wrapper = SimpleNamespace(
            primary=primary,
            children=tbo_children,
            decode_backend=hybrid_decode,
            attn_backends=(),
        )
        runner = object.__new__(DecodeCudaGraphRunner)

        runner._set_kv_protection_graph_bank(wrapper, 1)

        self.assertEqual(primary.banks, [1])
        self.assertEqual([child.banks for child in tbo_children], [[1], [1]])
        self.assertEqual(hybrid_decode.banks, [1])

    @mock.patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=SimpleNamespace(forward_metadata=None),
    )
    @mock.patch(
        "sglang.srt.layers.logits_processor.get_parallel",
        return_value=SimpleNamespace(attn_dp_size=2, tp_size=8),
    )
    def test_idle_dpa_lane_participates_in_failure_consensus(
        self, _get_parallel, _get_attn_backend
    ):
        gather_kwargs = {}

        def gather(logits, **kwargs):
            gather_kwargs.update(kwargs)
            return logits

        processor = self._make_logits_processor(gather)
        metadata = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            dp_local_start_pos=torch.tensor(3),
            dp_local_num_tokens=torch.tensor(0),
        )

        processor._get_logits(torch.empty(0, 4), None, metadata)

        self.assertEqual(gather_kwargs["local_failure"].tolist(), [0])
        self.assertEqual(gather_kwargs["global_failure"].numel(), 3)
        self.assertIs(gather_kwargs["failure_local_start"], metadata.dp_local_start_pos)
        self.assertIs(
            gather_kwargs["failure_local_num_rows"], metadata.dp_local_num_tokens
        )

    @mock.patch(
        "sglang.srt.layers.logits_processor.get_parallel",
        return_value=SimpleNamespace(attn_dp_size=2, tp_size=8),
    )
    def test_idle_dpa_lane_uses_nested_validator_metadata(self, _get_parallel):
        protection = {
            "validated_in_metadata": True,
            "local_failed": torch.zeros(2, dtype=torch.int32),
            "failed": torch.zeros(2, dtype=torch.int32),
            "global_failed": torch.empty(3, dtype=torch.int32),
        }
        dsa_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(kv_page_protection=protection)
        )
        wrapper = SimpleNamespace(
            primary=SimpleNamespace(decode_backend=dsa_backend), children=[]
        )
        gather_kwargs = {}

        def gather(logits, **kwargs):
            gather_kwargs.update(kwargs)
            return logits

        processor = self._make_logits_processor(gather)
        metadata = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            dp_local_start_pos=torch.tensor(3),
            dp_local_num_tokens=torch.tensor(0),
        )

        with mock.patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=wrapper,
        ):
            processor._get_logits(torch.empty(0, 4), None, metadata)

        self.assertIs(gather_kwargs["local_failure"], protection["local_failed"])
        self.assertEqual(
            gather_kwargs["global_failure"].data_ptr(),
            protection["global_failed"].data_ptr(),
        )
        self.assertEqual(gather_kwargs["global_failure"].numel(), 3)

    @mock.patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=SimpleNamespace(forward_metadata=None),
    )
    @mock.patch(
        "sglang.srt.layers.logits_processor.get_parallel",
        return_value=SimpleNamespace(attn_dp_size=2, tp_size=8),
    )
    def test_active_dpa_lane_fails_closed_without_validator_metadata(
        self, _get_parallel, _get_attn_backend
    ):
        processor = self._make_logits_processor(lambda logits, **kwargs: logits)
        metadata = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            dp_local_start_pos=torch.tensor(0),
            dp_local_num_tokens=torch.tensor(3),
        )

        with self.assertRaisesRegex(RuntimeError, "missing fused validator metadata"):
            processor._get_logits(torch.empty(3, 4), None, metadata)


if __name__ == "__main__":
    unittest.main()
