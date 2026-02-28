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


class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        input_ids_2d = forward_batch.input_ids.view(batch_size, self.block_size)

        mask_index = input_ids_2d == self.mask_id
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list = compute_start_list(
            forward_batch.input_ids,
            self.mask_id,
            batch_size,
            self.block_size,
        )
        prompt_masks = ~mask_index

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Controls whether to perform an additional forward pass for KV cache persistence.
        # For certain decoding rounds where the terminal step yields no state change,
        # this can be set to False to bypass the overhead of an idle forward pass.
        any_changed_in_last_step = torch.zeros((), dtype=torch.bool, device=device)

        max_iterations = self.block_size + self.max_post_edit_steps
        for _ in range(max_iterations):
            if finished.all():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            logits_3d = logits_output.full_logits.view(batch_size, self.block_size, -1)
            if self.penalty_lambda > 0:
                prev_ids = input_ids_2d[:, :-1]
                logits_3d[:, 1:, :].scatter_(
                    2, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                )

            x, p, confidence, mask_index = compute_confidence_batched(
                logits_output.full_logits,
                forward_batch.input_ids,
                self.mask_id,
                batch_size,
                self.block_size,
                return_probs=True,
            )
            active = ~finished
            has_mask = mask_index.any(dim=1)

            # Mask to token (M2T)
            mask_transfer_index = compute_transfer_indices_batched(
                confidence,
                self.threshold,
                mask_index,
            )

            no_mask = active & ~has_mask
            post_edit_steps[no_mask] += 1
            exceeded_post_edit_steps = no_mask & (
                post_edit_steps > self.max_post_edit_steps
            )
            finished = finished | exceeded_post_edit_steps

            # Token to token (T2T)
            edit_mask = ~mask_index & ~prompt_masks
            edit_transfer_index = (
                (p > self.edit_threshold) & (input_ids_2d != x) & edit_mask
            )

            active_after_post_edit = active & ~exceeded_post_edit_steps
            transfer_index = (mask_transfer_index | edit_transfer_index) & (
                active_after_post_edit.unsqueeze(1)
            )
            no_transfer = active_after_post_edit & ~transfer_index.any(dim=1)
            finished = finished | no_transfer

            input_ids_2d[transfer_index] = x[transfer_index]
            any_changed_in_last_step = transfer_index.any()

        if any_changed_in_last_step.item():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        start_list_cpu = start_list.tolist()
        next_token_ids_list = [
            input_ids_2d[i, start:] for i, start in enumerate(start_list_cpu)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold
