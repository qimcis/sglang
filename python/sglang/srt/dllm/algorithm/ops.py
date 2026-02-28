from __future__ import annotations

import torch
import torch.nn.functional as F


def _view_blocks(
    tensor: torch.Tensor, batch_size: int, block_size: int
) -> torch.Tensor:
    return tensor.view(batch_size, block_size, *tensor.shape[1:])


def compute_confidence_batched(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_id: int,
    batch_size: int,
    block_size: int,
    *,
    return_probs: bool = False,
) -> tuple[torch.Tensor, ...]:
    input_ids_2d = _view_blocks(input_ids, batch_size, block_size)
    logits_3d = _view_blocks(logits, batch_size, block_size)

    mask_index = input_ids_2d == mask_id
    x = torch.argmax(logits_3d, dim=-1)
    probs = F.softmax(logits_3d, dim=-1)
    top_probs = probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    x = x.to(dtype=input_ids_2d.dtype)
    confidence = torch.where(
        mask_index,
        top_probs,
        torch.full_like(top_probs, float("-inf")),
    )

    if return_probs:
        return x, top_probs, confidence, mask_index

    return x, confidence, mask_index


def compute_transfer_indices_batched(
    confidence: torch.Tensor,
    threshold: float,
    mask_index: torch.Tensor,
) -> torch.Tensor:
    transfer_index = confidence > threshold
    needs_fallback = mask_index.any(dim=1) & ~transfer_index.any(dim=1)
    fallback_index = torch.argmax(confidence, dim=1, keepdim=True)
    fallback_transfer = torch.zeros_like(transfer_index)
    fallback_transfer.scatter_(1, fallback_index, True)
    return transfer_index | (fallback_transfer & needs_fallback.unsqueeze(1))


def compute_start_list(
    input_ids: torch.Tensor,
    mask_id: int,
    batch_size: int,
    block_size: int,
) -> torch.Tensor:
    input_ids_2d = _view_blocks(input_ids, batch_size, block_size)
    return block_size - (input_ids_2d == mask_id).sum(dim=1)
