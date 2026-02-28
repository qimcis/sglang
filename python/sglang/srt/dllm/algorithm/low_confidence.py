from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.ops import (
    compute_confidence_batched,
    compute_start_list,
    compute_transfer_indices_batched,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        input_ids_2d = forward_batch.input_ids.view(batch_size, self.block_size)
        mask_index = input_ids_2d == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        start_list = compute_start_list(
            forward_batch.input_ids,
            self.mask_id,
            batch_size,
            self.block_size,
        )

        for _ in range(self.block_size):
            mask_index = input_ids_2d == self.mask_id
            if not mask_index.any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            x, confidence, mask_index = compute_confidence_batched(
                logits_output.full_logits,
                forward_batch.input_ids,
                self.mask_id,
                batch_size,
                self.block_size,
            )
            transfer_index = compute_transfer_indices_batched(
                confidence,
                self.threshold,
                mask_index,
            )
            input_ids_2d[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        start_list_cpu = start_list.tolist()
        next_token_ids_list = [
            input_ids_2d[i, start:] for i, start in enumerate(start_list_cpu)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
