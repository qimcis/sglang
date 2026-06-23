import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def _info(mask=None, eos=None, batch_size=2):
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.ones(batch_size, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=10,
        penalizer_orchestrator=Mock(is_required=False),
        device="cpu",
        thinking_phase_mask=mask,
        thinking_eos_token_ids=eos,
    )


class TestThinkingEosBanState(unittest.TestCase):
    def test_refresh_tracks_current_reasoning_state(self):
        info = _info(batch_size=2)
        reqs = [
            SimpleNamespace(
                require_reasoning=True,
                _is_reasoning_over=False,
                eos_token_ids={1},
                tokenizer=None,
            ),
            SimpleNamespace(
                require_reasoning=True,
                _is_reasoning_over=True,
                eos_token_ids={1},
                tokenizer=None,
            ),
        ]

        info.refresh_thinking_phase_state(reqs)

        self.assertEqual(info.thinking_phase_mask.tolist(), [True, False])
        self.assertEqual(info.thinking_eos_token_ids.tolist(), [1])

    def test_filter_batch_filters_thinking_phase_mask(self):
        info = _info(
            mask=torch.tensor([True, False, True]),
            eos=torch.tensor([1]),
            batch_size=3,
        )

        info.filter_batch([0, 2], torch.tensor([0, 2]))

        self.assertEqual(info.thinking_phase_mask.tolist(), [True, True])
        self.assertEqual(info.thinking_eos_token_ids.tolist(), [1])

    def test_merge_batch_merges_thinking_phase_state(self):
        lhs = _info(mask=torch.tensor([True, False]), eos=torch.tensor([1]), batch_size=2)
        rhs = _info(mask=torch.tensor([False, True]), eos=torch.tensor([2]), batch_size=2)

        lhs.merge_batch(rhs)

        self.assertEqual(lhs.thinking_phase_mask.tolist(), [True, False, False, True])
        self.assertEqual(lhs.thinking_eos_token_ids.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
