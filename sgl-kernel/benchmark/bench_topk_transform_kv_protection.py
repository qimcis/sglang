"""Microbenchmark for producer-fused DSA KV page protection."""

import torch
import triton
import triton.testing
from sgl_kernel import fast_topk_transform_fused

TOPK = 2048
PAGE_SIZE = 64
SHAPES = [
    (batch_size, seq_len) for batch_size in (1, 8) for seq_len in (2048, 8192, 32768)
]


def _make_case(batch_size: int, seq_len: int):
    pages_per_request = triton.cdiv(seq_len, PAGE_SIZE)
    logical = torch.arange(seq_len, device="cuda", dtype=torch.int32)
    logical_pages = logical // PAGE_SIZE
    offsets = logical % PAGE_SIZE
    page_table = torch.stack(
        [
            (1 + row * pages_per_request + logical_pages) * PAGE_SIZE + offsets
            for row in range(batch_size)
        ]
    ).contiguous()
    scores = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)

    num_pages = 1 + batch_size * pages_per_request
    page_ids = torch.arange(num_pages, device="cuda", dtype=torch.int64)
    owners = torch.full((num_pages,), -1, device="cuda", dtype=torch.int32)
    positions = torch.full_like(owners, -1)
    for row in range(batch_size):
        begin = 1 + row * pages_per_request
        end = begin + pages_per_request
        owners[begin:end] = row + 1
        positions[begin:end] = torch.arange(
            pages_per_request, device="cuda", dtype=torch.int32
        )

    actual_tags = page_ids * 17 + 3
    actual_generations = page_ids * 5 + 1
    actual_transfer_tags = (page_ids * 11 + 7).to(torch.int32)
    protection = {
        "request_indices": torch.arange(
            1, batch_size + 1, device="cuda", dtype=torch.int64
        ),
        "page_size": PAGE_SIZE,
        "page_table_page_offset": 0,
        "actual_tags": actual_tags,
        "actual_generations": actual_generations,
        "actual_transfer_tags": actual_transfer_tags,
        "owner_request_indices": owners,
        "owner_page_positions": positions,
        "expected_tags": actual_tags.clone(),
        "expected_generations": actual_generations.clone(),
        "expected_transfer_tags": actual_transfer_tags.clone(),
        "request_epochs": torch.ones(batch_size + 1, device="cuda", dtype=torch.int32),
        "validated_epochs": torch.full(
            (batch_size + 1,), -1, device="cuda", dtype=torch.int32
        ),
        "status": torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32),
    }
    return scores, lengths, page_table, cu_seqlens_q, protection


def _bench(batch_size: int, seq_len: int):
    scores, lengths, page_table, cu_seqlens_q, protection = _make_case(
        batch_size, seq_len
    )

    def unprotected():
        fast_topk_transform_fused(scores, lengths, page_table, cu_seqlens_q, TOPK)

    def protected():
        fast_topk_transform_fused(
            scores,
            lengths,
            page_table,
            cu_seqlens_q,
            TOPK,
            kv_page_protection=protection,
        )

    unprotected_output = fast_topk_transform_fused(
        scores, lengths, page_table, cu_seqlens_q, TOPK
    )
    protected_output = fast_topk_transform_fused(
        scores,
        lengths,
        page_table,
        cu_seqlens_q,
        TOPK,
        kv_page_protection=protection,
    )
    torch.testing.assert_close(
        torch.sort(protected_output, dim=-1).values,
        torch.sort(unprotected_output, dim=-1).values,
    )
    base_ms, _, _ = triton.testing.do_bench_cudagraph(
        unprotected, quantiles=[0.5, 0.2, 0.8]
    )
    protected_ms, _, _ = triton.testing.do_bench_cudagraph(
        protected, quantiles=[0.5, 0.2, 0.8]
    )
    assert not protection["status"].any().item()
    assert torch.equal(
        protection["validated_epochs"][1:], protection["request_epochs"][1:]
    )
    base_us = base_ms * 1000
    protected_us = protected_ms * 1000
    return base_us, protected_us, 100 * (protected_us - base_us) / base_us


if __name__ == "__main__":
    print("batch seq_len unprotected_us protected_us overhead_pct")
    for batch_size, seq_len in SHAPES:
        base_us, protected_us, overhead = _bench(batch_size, seq_len)
        print(
            f"{batch_size:5d} {seq_len:7d} {base_us:14.2f} "
            f"{protected_us:12.2f} {overhead:12.2f}"
        )
