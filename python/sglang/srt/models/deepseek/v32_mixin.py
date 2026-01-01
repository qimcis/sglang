from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    prepare_input_dp_with_cp_dsa,
)


class DeepseekV32Mixin:
    def prepare_cp_metadata(self, input_ids: torch.Tensor, forward_batch):
        if not getattr(self, "nsa_enable_prefill_cp", False):
            return

        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = len(input_ids) // (self.cp_size * 2)
        if can_cp_split(cur_cp_seq_len, self.cp_size, self.use_nsa, forward_batch):
            forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                torch.tensor(len(input_ids)),
                self.cp_rank,
                self.cp_size,
                forward_batch.seq_lens_cpu.tolist(),
            )
