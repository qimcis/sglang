import sys
from typing import Any, Optional

import pytest
import torch
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)


def _ref_torch_impl(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))
        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)
    return torch.where(mask, indices + topk_indices_offset, indices)


MAX_SEQ_LEN = 131072


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_kernel(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
    indices_our = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    # Tests can pass with max_permit_error=3, set to 5 for safety
    assert_equal(score, indices_ref, indices_our, bs, k, seq_len, max_permit_error=5)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_topk_transform_kernel(bs: int, k: int, seq_len: int, mode: str) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # NOTE: for decode, cumulative seqlens_q is just 0..=bs
    # NOTE: since page table is arange, they equal topk indices
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    if mode == "extend":
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="cuda"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)

    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        max_permit_error=5,
    )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_transform_ragged_kernel(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    # Used in prefill only
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # bs: # of q tokens
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    # kv_len
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        seq_len=seq_len,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        topk_indices_offset,
        max_permit_error=5,
    )


def _make_kv_protection_case(seq_len: int, page_size: int = 64):
    batch_size = 2
    pages_per_request = (seq_len + page_size - 1) // page_size
    logical_positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)
    logical_pages = logical_positions // page_size
    logical_offsets = logical_positions % page_size
    src_page_table = torch.stack(
        [
            (logical_pages + 1 + row * pages_per_request) * page_size + logical_offsets
            for row in range(batch_size)
        ]
    ).contiguous()
    score = torch.arange(seq_len, device="cuda", dtype=torch.float32).repeat(
        batch_size, 1
    )
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)

    num_physical_pages = batch_size * pages_per_request + 1
    page_ids = torch.arange(num_physical_pages, device="cuda", dtype=torch.int64)
    owner_request_indices = torch.full(
        (num_physical_pages,), -1, device="cuda", dtype=torch.int32
    )
    owner_page_positions = torch.full_like(owner_request_indices, -1)
    for row in range(batch_size):
        begin = 1 + row * pages_per_request
        end = begin + pages_per_request
        owner_request_indices[begin:end] = row + 1
        owner_page_positions[begin:end] = torch.arange(
            pages_per_request, device="cuda", dtype=torch.int32
        )

    actual_tags = page_ids * 17 + 3
    actual_generations = page_ids * 5 + 1
    actual_transfer_tags = (page_ids * 11 + 7).to(torch.int32)
    protection = {
        "request_indices": torch.tensor([1, 2], device="cuda", dtype=torch.int64),
        "page_size": page_size,
        "page_table_page_offset": 0,
        "actual_tags": actual_tags,
        "actual_generations": actual_generations,
        "actual_transfer_tags": actual_transfer_tags,
        "owner_request_indices": owner_request_indices,
        "owner_page_positions": owner_page_positions,
        "expected_tags": actual_tags.clone(),
        "expected_generations": actual_generations.clone(),
        "expected_transfer_tags": actual_transfer_tags.clone(),
        "request_epochs": torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        "validated_epochs": torch.full((3,), -1, device="cuda", dtype=torch.int32),
        "status": torch.zeros(3, device="cuda", dtype=torch.int32),
    }
    return score, lengths, src_page_table, cu_seqlens_q, protection


@torch.inference_mode()
def test_kv_protection_status_orchestration() -> None:
    from sglang.srt.mem_cache.kv_page_tags import (
        KV_PAGE_INVALID_MAPPING,
        KV_PAGE_VALIDATION_INCOMPLETE,
        KVAttentionTagTable,
    )

    table = KVAttentionTagTable(1, device="cuda", num_request_slots=4)
    table.request_epochs.copy_(
        torch.tensor([0, 4, 9, 12], device="cuda", dtype=torch.int32)
    )
    table.validated_epochs.copy_(
        torch.tensor([-1, 5, 9, 12], device="cuda", dtype=torch.int32)
    )
    table.validation_status.copy_(
        torch.tensor([99, 8, 16, 32], device="cuda", dtype=torch.int32)
    )
    request_indices = torch.tensor([-1, 0, 1, 3, 4], device="cuda", dtype=torch.int64)

    table.begin_fused_forward(request_indices)
    assert table.request_epochs.tolist() == [0, 5, 9, 13]
    assert table.validation_status.tolist() == [0, 0, 16, 0]

    # Request 1 completed this epoch with a fault; request 3 did not publish.
    table.validation_status[1] = 8
    statuses, failed = table.fused_failure_status(request_indices, return_failed=True)
    assert statuses.tolist() == [
        KV_PAGE_INVALID_MAPPING,
        0,
        8,
        KV_PAGE_VALIDATION_INCOMPLETE,
        KV_PAGE_INVALID_MAPPING,
    ]
    assert failed.tolist() == [1, 0, 1, 1, 1]


@torch.inference_mode()
def test_kv_protection_status_orchestration_cuda_graph() -> None:
    request_indices = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
    request_epochs = torch.zeros(3, device="cuda", dtype=torch.int32)
    validated_epochs = torch.full((3,), -1, device="cuda", dtype=torch.int32)
    status = torch.full((3,), 7, device="cuda", dtype=torch.int32)
    failure_status = torch.empty(3, device="cuda", dtype=torch.int32)
    failed = torch.empty_like(failure_status)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        torch.ops.sgl_kernel.kv_page_protection_begin_forward(
            request_indices, request_epochs, status
        )
        torch.ops.sgl_kernel.kv_page_protection_failure_status(
            request_indices,
            request_epochs,
            validated_epochs,
            status,
            failure_status,
            failed,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    request_epochs.zero_()
    validated_epochs.fill_(-1)
    status.fill_(7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.sgl_kernel.kv_page_protection_begin_forward(
            request_indices, request_epochs, status
        )
        torch.ops.sgl_kernel.kv_page_protection_failure_status(
            request_indices,
            request_epochs,
            validated_epochs,
            status,
            failure_status,
            failed,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert request_epochs.tolist() == [0, 1, 1]
    assert status.tolist() == [0, 0, 0]
    assert failure_status.tolist() == [0, 1 << 30, 1 << 30]
    assert failed.tolist() == [0, 1, 1]


@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("page_size", [32, 64])
@torch.inference_mode()
def test_topk_transform_kv_protection_valid(seq_len: int, page_size: int) -> None:
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len, page_size=page_size
    )
    expected = _ref_torch_transform_decode_impl(score, seq_len, src_page_table, 2048)
    actual = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=2048,
        kv_page_protection=protection,
    )

    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )
    assert protection["status"].tolist() == [0, 0, 0]
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]


@torch.inference_mode()
def test_topk_transform_kv_protection_generic_page_size() -> None:
    seq_len = 2048
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len, page_size=32
    )
    expected = _ref_torch_transform_decode_impl(score, seq_len, src_page_table, 2048)
    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    torch.testing.assert_close(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )
    assert protection["status"].tolist() == [0, 0, 0]
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]


@torch.inference_mode()
def test_topk_transform_kv_protection_revalidates_after_epoch_publication() -> None:
    seq_len = 2048
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len
    )
    fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]

    physical_page = int(src_page_table[0, 0].item()) // protection["page_size"]
    protection["owner_request_indices"][physical_page] = 2
    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    assert protection["status"][1].item() == 0x02
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]
    assert torch.any(actual[0] == 0)


@pytest.mark.parametrize(
    ("fault", "expected_status"),
    [
        ("zero_slot", 0x01),
        ("negative_slot", 0x01),
        ("out_of_range_slot", 0x01),
        ("same_page_offset", 0x01),
        ("owner", 0x02),
        ("position", 0x04),
        ("attention_tag", 0x08),
        ("generation", 0x10),
        ("transfer_tag", 0x20),
    ],
)
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("page_size", [32, 64])
@torch.inference_mode()
def test_topk_transform_kv_protection_faults(
    fault: str, expected_status: int, seq_len: int, page_size: int
) -> None:
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len, page_size=page_size
    )
    logical_position = seq_len - 2
    physical_page = int(src_page_table[0, logical_position].item()) // page_size

    if fault == "zero_slot":
        src_page_table[0, logical_position] = 0
    elif fault == "negative_slot":
        src_page_table[0, logical_position] = -1
    elif fault == "out_of_range_slot":
        src_page_table[0, logical_position] = (
            protection["actual_tags"].numel() * page_size + logical_position % page_size
        )
    elif fault == "same_page_offset":
        src_page_table[0, logical_position] += 1
    elif fault == "owner":
        protection["owner_request_indices"][physical_page] = 2
    elif fault == "position":
        protection["owner_page_positions"][physical_page] += 1
    elif fault == "attention_tag":
        protection["expected_tags"][physical_page] += 1
    elif fault == "generation":
        protection["expected_generations"][physical_page] += 1
    elif fault == "transfer_tag":
        protection["expected_transfer_tags"][physical_page] += 1
    else:
        raise AssertionError(f"unknown fault: {fault}")

    actual = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=2048,
        kv_page_protection=protection,
    )

    assert protection["status"][1].item() == expected_status
    assert protection["status"][2].item() == 0
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]
    assert torch.any(actual[0] == 0)
    assert not torch.any(actual[1] == 0)


@torch.inference_mode()
def test_topk_transform_kv_protection_ignores_unselected_page() -> None:
    seq_len = 4096
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len
    )
    unselected_page = int(src_page_table[0, 0].item()) // protection["page_size"]
    protection["expected_tags"][unselected_page] += 1

    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    assert protection["status"].tolist() == [0, 0, 0]
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]
    assert not torch.any(actual == 0)


@pytest.mark.parametrize("bad_length", [-1, 4097])
@torch.inference_mode()
def test_topk_transform_kv_protection_rejects_bad_length(
    bad_length: int,
) -> None:
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        4096
    )
    lengths[0] = bad_length

    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    assert protection["status"][1].item() == 0x01
    assert protection["status"][2].item() == 0
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]
    assert torch.all(actual[0] == 0)


@torch.inference_mode()
def test_topk_transform_kv_protection_page_offset() -> None:
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        4096
    )
    page_offset = 7
    for name in (
        "actual_tags",
        "actual_generations",
        "actual_transfer_tags",
        "owner_request_indices",
        "owner_page_positions",
        "expected_tags",
        "expected_generations",
        "expected_transfer_tags",
    ):
        sidecar = protection[name]
        shifted = sidecar.new_zeros(sidecar.numel() + page_offset)
        shifted[page_offset:] = sidecar
        protection[name] = shifted
    protection["page_table_page_offset"] = page_offset

    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    assert protection["status"].tolist() == [0, 0, 0]
    assert protection["validated_epochs"].tolist() == [-1, 3, 5]
    assert not torch.any(actual == 0)


@torch.inference_mode()
def test_topk_transform_kv_protection_graph_padding_uses_safe_slot() -> None:
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        2048
    )
    protection["request_indices"][0] = 0

    actual = fast_topk_transform_fused(
        score,
        lengths,
        src_page_table,
        cu_seqlens_q,
        2048,
        kv_page_protection=protection,
    )

    assert torch.all(actual[0] == 0)
    assert not torch.any(actual[1] == 0)
    assert protection["status"].tolist() == [0, 0, 0]
    assert protection["validated_epochs"].tolist() == [-1, -1, 5]


@torch.inference_mode()
def test_topk_transform_kv_protection_cuda_graph_replay() -> None:
    seq_len = 4096
    score, lengths, src_page_table, cu_seqlens_q, protection = _make_kv_protection_case(
        seq_len
    )
    logical_position = seq_len - 2

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fast_topk_transform_fused(
            score,
            lengths,
            src_page_table,
            cu_seqlens_q,
            2048,
            kv_page_protection=protection,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    protection["validated_epochs"].fill_(-1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fast_topk_transform_fused(
            score,
            lengths,
            src_page_table,
            cu_seqlens_q,
            2048,
            kv_page_protection=protection,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert protection["status"][1].item() == 0
    assert protection["validated_epochs"][1].item() == 3

    src_page_table[0, logical_position] += 1
    graph.replay()
    torch.cuda.synchronize()
    assert protection["status"][1].item() == 0x01
    assert protection["validated_epochs"][1].item() == 3
    assert torch.any(output[0] == 0)

    src_page_table[0, logical_position] -= 1
    protection["status"].zero_()
    protection["request_epochs"][1] += 1
    graph.replay()
    torch.cuda.synchronize()
    assert protection["status"][1].item() == 0
    assert protection["validated_epochs"][1].item() == 4
    assert not torch.any(output[0] == 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
