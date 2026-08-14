import pytest
import torch
from sgl_kernel.kvcacheio import (
    dsv4_batched_page_digests,
    dsv4_bind_pages,
    dsv4_page_digests,
    dsv4_validate_pages,
)

from sglang.srt.mem_cache.dsv4_kv_integrity import (
    DSV4Component,
    DSV4ComponentDescriptor,
    DSV4IntegrityDomain,
    DSV4IntegrityError,
    DSV4KVIntegrityManager,
    DSV4TransferGroup,
    component_seed,
    mapping_seed,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_batched_page_digests_match_individual_hashes():
    buffers = [
        torch.arange(4 * width, dtype=torch.uint8, device="cuda").reshape(4, width)
        for width in (32, 48)
    ]
    pages = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 2, 3, 3], dtype=torch.int32, device="cuda")
    descriptors = torch.tensor(
        [
            [buffers[0].data_ptr(), 4, 32, 11, 0],
            [buffers[1].data_ptr(), 4, 48, 13, 1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    actual = dsv4_batched_page_digests(descriptors, pages, offsets)
    torch.testing.assert_close(
        actual[0, :2], dsv4_page_digests(buffers[0], pages[:2], 11)
    )
    torch.testing.assert_close(
        actual[1, :1], dsv4_page_digests(buffers[1], pages[2:], 13)
    )


def test_protected_nonpaged_gather_sanitizes_mapping_mismatch():
    from sglang.kernels.ops.attention.dsa.index_buf_accessor import (
        _get_k_and_s_triton,
    )

    page_size = 64
    index_head_dim = 128
    row_bytes = page_size * (index_head_dim + 4)
    buffer = torch.zeros((4, row_bytes), dtype=torch.uint8, device="cuda")
    buffer[1:].copy_(
        torch.arange(3 * row_bytes, dtype=torch.int64, device="cuda")
        .remainder(251)
        .to(torch.uint8)
        .reshape(3, row_bytes)
    )
    page_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    request = torch.tensor([1], dtype=torch.int64, device="cuda")
    generations = torch.zeros(4, dtype=torch.int64, device="cuda")
    generations[1:] = 1
    request_epochs = torch.zeros(4, dtype=torch.int64, device="cuda")
    request_epochs[1] = 1
    tags = torch.zeros((4, 4), dtype=torch.int64, device="cuda")
    status = torch.zeros(4, dtype=torch.int32, device="cuda")
    bound = torch.empty_like(page_table)
    dsv4_bind_pages(
        page_table,
        logical,
        request,
        generations,
        request_epochs,
        tags,
        bound,
        status,
        17,
        1,
        0,
        True,
    )
    integrity_args = {
        "request_indices": request,
        "generations": generations,
        "request_epochs": request_epochs,
        "expected_tags": tags,
        "failure_status": status,
        "mapping_seed": 17,
    }
    k, scale = _get_k_and_s_triton(
        buffer,
        page_table,
        torch.tensor([128], dtype=torch.int32, device="cuda"),
        128,
        128,
        page_size,
        index_head_dim,
        integrity_args,
    )
    torch.testing.assert_close(
        k[:64], buffer[1, : page_size * index_head_dim].view(64, 128)
    )
    torch.testing.assert_close(
        scale[:64], buffer[1, page_size * index_head_dim :].view(64, 4)
    )
    assert status[1].item() == 0

    status.zero_()
    corrupted = page_table.clone()
    corrupted[0, 1] = 3
    k, scale = _get_k_and_s_triton(
        buffer,
        corrupted,
        torch.tensor([128], dtype=torch.int32, device="cuda"),
        128,
        128,
        page_size,
        index_head_dim,
        integrity_args,
    )
    assert torch.count_nonzero(k[64:]).item() == 0
    assert torch.count_nonzero(scale[64:]).item() == 0
    assert status[1].item() & (1 << 4)

    # A corrupt live device length must not exceed the trusted CPU allocation
    # bounds, including the unused lanes in the final 256-token Triton tile.
    status.zero_()
    k, scale = _get_k_and_s_triton(
        buffer,
        page_table,
        torch.tensor([torch.iinfo(torch.int32).max], dtype=torch.int32, device="cuda"),
        128,
        128,
        page_size,
        index_head_dim,
        integrity_args,
    )
    torch.cuda.synchronize()
    assert k.shape == (128, index_head_dim)
    assert scale.shape == (128, 4)
    assert status[1].item() == 0


def _state():
    device = "cuda"
    buffer = torch.zeros((8, 584), dtype=torch.uint8, device=device)
    buffer[1].fill_(17)
    pages = torch.tensor([1], dtype=torch.int32, device=device)
    digests = torch.zeros(8, dtype=torch.int64, device=device)
    digests[1] = dsv4_page_digests(buffer, pages, 99)[0]
    component_valid = torch.zeros(8, dtype=torch.uint8, device=device)
    component_valid[1] = 1
    validation_state = torch.zeros(8, dtype=torch.int32, device=device)
    validation_failure = torch.zeros(8, dtype=torch.int32, device=device)
    validation_pages = torch.empty(8, dtype=torch.int32, device=device)
    validation_count = torch.zeros(1, dtype=torch.int32, device=device)
    generations = torch.zeros(8, dtype=torch.int64, device=device)
    generations[1:3] = 1
    request_epochs = torch.zeros(4, dtype=torch.int64, device=device)
    request_epochs[1] = 1
    expected_tags = torch.zeros((4, 4), dtype=torch.int64, device=device)
    failure_status = torch.zeros(4, dtype=torch.int32, device=device)
    return (
        buffer,
        digests,
        component_valid,
        validation_state,
        validation_failure,
        validation_pages,
        validation_count,
        generations,
        request_epochs,
        expected_tags,
        failure_status,
    )


def _bind(state, slot=1):
    _, _, _, _, _, _, _, generations, epochs, tags, status = state
    slots = torch.tensor([[slot]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        epochs,
        tags,
        output,
        status,
        17,
        1,
        0,
        True,
    )
    return slots, logical, reqs, output


def _validate(state, slots, logical, reqs, *, allow_missing_digest=False):
    (
        buffer,
        digests,
        component_valid,
        validation_state,
        validation_failure,
        validation_pages,
        validation_count,
        generations,
        epochs,
        tags,
        status,
    ) = state
    output = torch.empty_like(slots)
    dsv4_validate_pages(
        buffer,
        slots,
        logical,
        reqs,
        digests,
        component_valid,
        validation_state,
        validation_failure,
        validation_pages,
        validation_count,
        generations,
        epochs,
        tags,
        output,
        status,
        99,
        17,
        1,
        0,
        allow_missing_digest,
    )
    return output


def test_clean_mapping_and_digest_pass():
    state = _state()
    slots, logical, reqs, bound = _bind(state)
    assert bound.item() == 1
    assert _validate(state, slots, logical, reqs).item() == 1
    assert state[-1][1].item() == 0


def test_bit_flip_is_sanitized_and_recorded():
    state = _state()
    slots, logical, reqs, _ = _bind(state)
    state[0][1, 0] ^= 1
    assert _validate(state, slots, logical, reqs).item() == 0
    assert state[-1][1].item() & (1 << 2)


def test_first_write_may_initialize_missing_digest_but_reads_fail_closed():
    state = _state()
    slots, logical, reqs, _ = _bind(state)
    state[2][1] = 0
    assert _validate(state, slots, logical, reqs).item() == 0
    assert state[-1][1].item() & (1 << 1)

    state[-1].zero_()
    assert (
        _validate(
            state,
            slots,
            logical,
            reqs,
            allow_missing_digest=True,
        ).item()
        == 1
    )
    assert state[-1][1].item() == 0


def test_cross_page_mapping_and_generation_reuse_fail_closed():
    state = _state()
    slots, logical, reqs, _ = _bind(state)
    remapped = slots.new_tensor([[2]])
    assert _validate(state, remapped, logical, reqs).item() == 0
    assert state[-1][1].item() & (1 << 4)

    state[-1].zero_()
    state[7][1] += 1
    assert _validate(state, slots, logical, reqs).item() == 0
    assert state[-1][1].item() & (1 << 4)


def test_duplicate_bind_is_graph_safe_and_conflicts_fail():
    state = _state()
    _, _, _, _, _, _, _, generations, epochs, tags, status = state
    slots = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device="cuda")
    logical = torch.zeros_like(slots, dtype=torch.int64)
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        epochs,
        tags,
        output,
        status,
        17,
        1,
        0,
        True,
    )
    assert torch.equal(output, slots)

    conflicting = slots.clone()
    conflicting[0, -1] = 2
    dsv4_bind_pages(
        conflicting,
        logical,
        reqs,
        generations,
        epochs,
        tags,
        output,
        status,
        17,
        1,
        0,
        True,
    )
    assert output[0, -1].item() == 0
    assert status[1].item() & (1 << 4)


def test_verify_only_does_not_learn_first_observed_mapping():
    state = _state()
    _, _, _, _, _, _, _, generations, epochs, tags, status = state
    slots = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        epochs,
        tags,
        output,
        status,
        17,
        1,
        0,
        False,
    )
    assert output.item() == 0
    assert status[1].item() & (1 << 3)
    assert tags[1, 0].item() == 0


def test_live_negative_slot_fails_instead_of_being_treated_as_padding():
    state = _state()
    _, _, _, _, _, _, _, generations, epochs, tags, status = state
    slots = torch.tensor([[-1]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        epochs,
        tags,
        output,
        status,
        17,
        1,
        0,
        False,
    )
    assert output.item() == 0
    assert status[1].item() & (1 << 0)


def test_request_slot_reuse_clears_kv_mappings_and_rebinds_state_generation():
    buffers = [torch.zeros((4, 16), dtype=torch.uint8, device="cuda") for _ in range(3)]
    descriptors = [
        DSV4ComponentDescriptor(
            DSV4Component.C4_ATTENTION_KV,
            0,
            4,
            DSV4TransferGroup.KV,
            1,
            16,
            4,
            buffers[0],
        ),
        DSV4ComponentDescriptor(
            DSV4Component.SWA_KV,
            0,
            0,
            DSV4TransferGroup.SWA,
            1,
            16,
            4,
            buffers[1],
        ),
        DSV4ComponentDescriptor(
            DSV4Component.C128_ATTENTION_STATE,
            0,
            128,
            DSV4TransferGroup.C128_STATE,
            1,
            16,
            4,
            buffers[2],
        ),
    ]
    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=4,
        full_page_size=1,
        swa_page_size=1,
    )
    manager.failure_status[1] = 123
    for space in manager.address_spaces.values():
        space.expected_tags[1].fill_(1)
    manager.register_requests([1], [1])
    assert manager.failure_status[1].item() == 0
    assert (
        not manager.address_spaces[DSV4IntegrityDomain.FULL]
        .expected_tags[1]
        .any()
        .item()
    )
    assert (
        not manager.address_spaces[DSV4IntegrityDomain.SWA]
        .expected_tags[1]
        .any()
        .item()
    )
    state_space = manager.address_spaces[DSV4IntegrityDomain.C128_STATE]
    first_tag = state_space.expected_tags[1, 0].item()
    assert first_tag != 0
    state_generation = manager.address_spaces[DSV4IntegrityDomain.C128_STATE].generation
    assert state_generation[1].item() == 1
    manager.register_requests([1], [2])
    assert state_generation[1].item() == 2
    assert state_space.expected_tags[1, 0].item() not in (0, first_tag)


def test_transfer_group_supports_heterogeneous_component_capacities():
    specs = (
        (DSV4Component.C4_ATTENTION_KV, DSV4TransferGroup.KV, 4),
        (DSV4Component.SWA_KV, DSV4TransferGroup.SWA, 4),
        (DSV4Component.C4_ATTENTION_STATE, DSV4TransferGroup.SWA, 8),
        (
            DSV4Component.C128_ATTENTION_STATE,
            DSV4TransferGroup.C128_STATE,
            4,
        ),
    )
    descriptors = []
    for component, group, capacity in specs:
        buffer = torch.arange(capacity * 16, dtype=torch.uint8, device="cuda").reshape(
            capacity, 16
        )
        descriptors.append(
            DSV4ComponentDescriptor(
                component,
                0,
                4 if component == DSV4Component.C4_ATTENTION_STATE else 0,
                group,
                1,
                16,
                capacity,
                buffer,
            )
        )

    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=4,
        full_page_size=1,
        swa_page_size=1,
    )
    assert manager.address_spaces[DSV4IntegrityDomain.SWA].physical_capacity == 8
    manager.register_requests([1], [1])

    page = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    request = torch.tensor([1], dtype=torch.int64, device="cuda")
    manager.bump_allocations(DSV4IntegrityDomain.SWA, page)
    manager.bind_pages(
        DSV4IntegrityDomain.SWA,
        page,
        logical,
        request,
        slot_page_size=1,
    )
    for descriptor in descriptors:
        if descriptor.transfer_group != DSV4TransferGroup.SWA:
            continue
        manager.refresh_pages(descriptor, page)
        assert (
            manager.validate_pages(
                descriptor, page, logical, request, slot_page_size=1
            ).item()
            == 1
        )

    # A generation event in the larger state-only tail must not index the
    # smaller SWA-KV sidecar out of bounds.
    manager.bump_allocations(
        DSV4IntegrityDomain.SWA,
        torch.tensor([6], dtype=torch.int32, device="cuda"),
    )


def test_empty_compressor_mapping_is_a_noop():
    buffer = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    descriptor = DSV4ComponentDescriptor(
        DSV4Component.C4_ATTENTION_STATE,
        0,
        4,
        DSV4TransferGroup.SWA,
        8,
        16,
        4,
        buffer,
    )
    manager = DSV4KVIntegrityManager(
        (
            DSV4ComponentDescriptor(
                DSV4Component.C4_ATTENTION_KV,
                0,
                4,
                DSV4TransferGroup.KV,
                1,
                16,
                4,
                buffer.clone(),
            ),
            descriptor,
            DSV4ComponentDescriptor(
                DSV4Component.C128_ATTENTION_STATE,
                0,
                128,
                DSV4TransferGroup.C128_STATE,
                1,
                16,
                4,
                buffer.clone(),
            ),
        ),
        request_capacity=4,
        max_context_len=4,
        full_page_size=1,
        swa_page_size=1,
    )
    slots = torch.empty(0, dtype=torch.int32, device="cuda")
    logical = torch.empty(0, dtype=torch.int64, device="cuda")
    requests = torch.empty(0, dtype=torch.int64, device="cuda")

    assert (
        manager.bind_pages(
            DSV4IntegrityDomain.SWA,
            slots,
            logical,
            requests,
            slot_page_size=2,
        ).numel()
        == 0
    )
    assert (
        manager.verify_mapping(
            DSV4IntegrityDomain.SWA,
            slots,
            logical,
            requests,
            slot_page_size=2,
        ).numel()
        == 0
    )
    assert (
        manager.replace_pages(
            DSV4IntegrityDomain.SWA,
            slots,
            logical,
            requests,
            slot_page_size=2,
        ).numel()
        == 0
    )
    assert (
        manager.validate_pages(
            descriptor,
            slots,
            logical,
            requests,
            slot_page_size=2,
        ).numel()
        == 0
    )


def test_request_table_write_is_trusted_but_first_consumer_mapping_is_not():
    buffers = [torch.zeros((4, 16), dtype=torch.uint8, device="cuda") for _ in range(3)]
    descriptors = [
        DSV4ComponentDescriptor(
            DSV4Component.C4_ATTENTION_KV,
            0,
            4,
            DSV4TransferGroup.KV,
            1,
            16,
            4,
            buffers[0],
        ),
        DSV4ComponentDescriptor(
            DSV4Component.SWA_KV,
            0,
            0,
            DSV4TransferGroup.SWA,
            1,
            16,
            4,
            buffers[1],
        ),
        DSV4ComponentDescriptor(
            DSV4Component.C128_ATTENTION_STATE,
            0,
            128,
            DSV4TransferGroup.C128_STATE,
            1,
            16,
            4,
            buffers[2],
        ),
    ]
    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=4,
        full_page_size=1,
        swa_page_size=1,
    )
    manager.attach_full_to_swa_mapping(
        torch.arange(4, dtype=torch.int64, device="cuda")
    )
    manager.bump_allocations(
        DSV4IntegrityDomain.FULL,
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
    )
    manager.bump_allocations(
        DSV4IntegrityDomain.SWA,
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
    )
    manager.register_requests([1], [1])
    manager.bind_request_table_write(
        (1, slice(0, 1)), torch.tensor([1], dtype=torch.int32, device="cuda")
    )

    wrong = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    sanitized = manager.verify_mapping(
        DSV4IntegrityDomain.FULL,
        wrong,
        logical,
        reqs,
        slot_page_size=1,
    )
    assert sanitized.item() == 0
    assert manager.failure_status[1].item() & (1 << 4)


def test_fused_core_mapping_validation_sanitizes_all_consumers():
    buffers = [torch.zeros((8, 16), dtype=torch.uint8, device="cuda") for _ in range(3)]
    descriptors = [
        DSV4ComponentDescriptor(
            component,
            0,
            ratio,
            group,
            page_size,
            16,
            8,
            buffer,
        )
        for component, ratio, group, page_size, buffer in (
            (
                DSV4Component.C4_ATTENTION_KV,
                4,
                DSV4TransferGroup.KV,
                2,
                buffers[0],
            ),
            (
                DSV4Component.SWA_KV,
                0,
                DSV4TransferGroup.SWA,
                4,
                buffers[1],
            ),
            (
                DSV4Component.C128_ATTENTION_STATE,
                128,
                DSV4TransferGroup.C128_STATE,
                1,
                buffers[2],
            ),
        )
    ]
    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=8,
        full_page_size=2,
        swa_page_size=4,
    )
    requests = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    manager.register_requests([1, 2], [11, 12])
    pages = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    manager.bump_allocations(DSV4IntegrityDomain.FULL, pages)
    manager.bump_allocations(DSV4IntegrityDomain.SWA, pages)

    logical = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    full = torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
    out = torch.tensor([2, 4], dtype=torch.int64, device="cuda")
    swa = torch.tensor([[4], [8]], dtype=torch.int32, device="cuda")
    manager.bind_pages(
        DSV4IntegrityDomain.FULL,
        full,
        logical,
        requests,
        slot_page_size=1,
    )
    manager.bind_pages(
        DSV4IntegrityDomain.SWA,
        swa,
        logical,
        requests,
        slot_page_size=4,
    )

    manager.verify_core_mappings(
        full,
        logical,
        out,
        logical.flatten(),
        swa,
        logical,
        requests,
    )
    torch.testing.assert_close(
        full, torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        out, torch.tensor([2, 4], dtype=torch.int64, device="cuda")
    )
    torch.testing.assert_close(
        swa, torch.tensor([[4], [8]], dtype=torch.int32, device="cuda")
    )
    assert not manager.failure_status[requests].any().item()

    manager.failure_status.zero_()
    full[1, 0] = 1
    out[0] = 4
    swa[1, 0] = 4
    manager.verify_core_mappings(
        full,
        logical,
        out,
        logical.flatten(),
        swa,
        logical,
        requests,
    )
    assert full[1, 0].item() == 0
    assert out[0].item() == 0
    assert swa[1, 0].item() == 0
    assert manager.failure_status[1].item() & (1 << 4)
    assert manager.failure_status[2].item() & (1 << 4)

    manager.verify_core_mappings(
        torch.tensor([[1], [2]], dtype=torch.int32, device="cuda"),
        logical,
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int64, device="cuda"),
        torch.tensor([[4], [8]], dtype=torch.int32, device="cuda"),
        logical,
        requests,
    )


def _complete_manager():
    specs = (
        (DSV4Component.SWA_KV, 0, DSV4TransferGroup.SWA, 584),
        (DSV4Component.C4_ATTENTION_KV, 4, DSV4TransferGroup.KV, 584),
        (DSV4Component.C128_ATTENTION_KV, 128, DSV4TransferGroup.KV, 584),
        (DSV4Component.C4_INDEXER_KV, 4, DSV4TransferGroup.KV, 132),
        (DSV4Component.C4_ATTENTION_STATE, 4, DSV4TransferGroup.SWA, 256),
        (DSV4Component.C4_INDEXER_STATE, 4, DSV4TransferGroup.SWA, 256),
        (
            DSV4Component.C128_ATTENTION_STATE,
            128,
            DSV4TransferGroup.C128_STATE,
            256,
        ),
    )
    descriptors = []
    for component, ratio, group, item_nbytes in specs:
        buffer = torch.arange(
            8 * item_nbytes, dtype=torch.uint8, device="cuda"
        ).reshape(8, item_nbytes)
        descriptors.append(
            DSV4ComponentDescriptor(
                component,
                0,
                ratio,
                group,
                1,
                item_nbytes,
                8,
                buffer,
            )
        )

    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=4,
        full_page_size=1,
        swa_page_size=1,
    )
    manager.register_requests([1], [1])
    page = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    request = torch.tensor([1], dtype=torch.int64, device="cuda")
    for domain in (DSV4IntegrityDomain.FULL, DSV4IntegrityDomain.SWA):
        manager.bump_allocations(domain, page)
        assert (
            manager.bind_pages(domain, page, logical, request, slot_page_size=1).item()
            == 1
        )
    for descriptor in descriptors:
        manager.refresh_pages(descriptor, page)
    return manager, descriptors, page, logical, request


def test_batched_transfer_digest_keeps_cross_stream_inputs_alive():
    manager, descriptors, _, _, _ = _complete_manager()
    pages = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    pages_by_group = {group: pages for group in DSV4TransferGroup}

    # Delay the private digest stream so the caller can release and try to
    # recycle its temporary page/offset tensors before the kernel reads them.
    with torch.cuda.stream(manager._digest_stream):
        torch.cuda._sleep(50_000_000)
    actual, counts, digest_stream = manager._batched_transfer_digests(pages_by_group)
    torch.cuda.current_stream().synchronize()
    allocator_churn = [
        torch.full((9 if i % 2 else 4,), -1, dtype=torch.int32, device="cuda")
        for i in range(64)
    ]
    digest_stream.synchronize()
    assert allocator_churn

    for row, descriptor in enumerate(descriptors):
        count = counts[descriptor.transfer_group]
        expected = dsv4_page_digests(
            descriptor.buffer, pages[:count], component_seed(descriptor)
        )
        torch.testing.assert_close(actual[row, :count], expected)


def test_transfer_digest_install_keeps_cross_stream_values_alive():
    manager, descriptors, _, _, _ = _complete_manager()
    indices = {group: [1] for group in DSV4TransferGroup}
    logical_starts = {group: 0 for group in DSV4TransferGroup}
    manifest = manager.build_manifest(
        bootstrap_room=1,
        transfer_nonce=2,
        indices_by_group=indices,
        logical_starts=logical_starts,
    )

    for sidecar in manager.sidecars.values():
        sidecar.digest[1] = 0
        sidecar.valid[1] = 0

    # Hold the caller stream before its first sidecar write.  A digest-stream
    # allocation that is consumed on this stream without ownership transfer can
    # then be recycled and overwritten while the write is still pending.
    delayed_sidecar = next(iter(manager.sidecars.values()))
    original_install = delayed_sidecar._install_prevalidated

    def delayed_install(page_indices, digests):
        torch.cuda._sleep(50_000_000)
        original_install(page_indices, digests)

    delayed_sidecar._install_prevalidated = delayed_install
    try:
        manager.verify_and_install(
            manifest,
            bootstrap_room=1,
            transfer_nonce=2,
            indices_by_group=indices,
            logical_starts=logical_starts,
            request_index=1,
        )
    finally:
        delayed_sidecar._install_prevalidated = original_install

    output_shape = (len(descriptors), len(DSV4TransferGroup))
    with torch.cuda.stream(manager._digest_stream):
        allocator_churn = [
            torch.full(output_shape, -1, dtype=torch.int64, device="cuda")
            for _ in range(64)
        ]
    torch.cuda.synchronize()
    assert allocator_churn

    for descriptor in descriptors:
        expected = dsv4_page_digests(
            descriptor.buffer,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            component_seed(descriptor),
        )
        torch.testing.assert_close(
            manager.sidecars[descriptor.identity].digest[1:2], expected
        )
        assert manager.sidecars[descriptor.identity].valid[1].item() == 1


def test_fused_metadata_gathers_validate_before_topk_and_attention():
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        BuildCausalSwaPageIndices,
        BuildPageTablePositions,
    )

    buffers = [torch.zeros((8, 16), dtype=torch.uint8, device="cuda") for _ in range(3)]
    descriptors = [
        DSV4ComponentDescriptor(
            component,
            0,
            ratio,
            group,
            page_size,
            16,
            8,
            buffer,
        )
        for component, ratio, group, page_size, buffer in (
            (
                DSV4Component.C4_ATTENTION_KV,
                4,
                DSV4TransferGroup.KV,
                256,
                buffers[0],
            ),
            (
                DSV4Component.SWA_KV,
                0,
                DSV4TransferGroup.SWA,
                128,
                buffers[1],
            ),
            (
                DSV4Component.C128_ATTENTION_STATE,
                128,
                DSV4TransferGroup.C128_STATE,
                1,
                buffers[2],
            ),
        )
    ]
    manager = DSV4KVIntegrityManager(
        descriptors,
        request_capacity=4,
        max_context_len=1024,
        full_page_size=256,
        swa_page_size=128,
    )
    manager.register_requests([1], [7])
    pages = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    manager.bump_allocations(DSV4IntegrityDomain.FULL, pages)
    manager.bump_allocations(DSV4IntegrityDomain.SWA, pages)
    request = torch.tensor([1], dtype=torch.int64, device="cuda")
    manager.bind_pages(
        DSV4IntegrityDomain.FULL,
        torch.tensor([[1, 2]], dtype=torch.int32, device="cuda"),
        torch.tensor([[0, 1]], dtype=torch.int64, device="cuda"),
        request,
        slot_page_size=1,
    )
    manager.bind_pages(
        DSV4IntegrityDomain.SWA,
        torch.tensor([[128]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int64, device="cuda"),
        request,
        slot_page_size=128,
    )
    full = manager.address_spaces[DSV4IntegrityDomain.FULL]
    swa = manager.address_spaces[DSV4IntegrityDomain.SWA]
    integrity_args = {
        "full_generations": full.generation,
        "swa_generations": swa.generation,
        "request_epochs": manager.request_epochs,
        "full_tags": full.expected_tags,
        "swa_tags": swa.expected_tags,
        "failure_status": manager.failure_status,
        "full_seed": mapping_seed(DSV4IntegrityDomain.FULL),
        "swa_seed": mapping_seed(DSV4IntegrityDomain.SWA),
        "full_page_size": 256,
        "swa_page_size": 128,
    }

    req_to_token = torch.zeros((4, 1024), dtype=torch.int32, device="cuda")
    offsets = torch.arange(512, dtype=torch.int32, device="cuda")
    req_to_token[1, :256] = 256 + offsets[:256]
    req_to_token[1, 256:512] = 512 + offsets[256:512] - 256
    page_metadata = BuildPageTablePositions.execute(
        req_to_token=req_to_token,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=torch.tensor([512], dtype=torch.int32, device="cuda"),
        max_seq_len=512,
        page_size=256,
        swa_window=128,
        integrity_args=integrity_args,
    )
    torch.testing.assert_close(
        page_metadata.page_table,
        torch.tensor([[1, 2]], dtype=torch.int32, device="cuda"),
    )
    assert manager.failure_status[1].item() == 0

    req_to_token[1, 256] = 3 * 256
    page_metadata = BuildPageTablePositions.execute(
        req_to_token=req_to_token,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=torch.tensor([512], dtype=torch.int32, device="cuda"),
        max_seq_len=512,
        page_size=256,
        swa_window=128,
        integrity_args=integrity_args,
    )
    assert page_metadata.page_table[0, 1].item() == 0
    assert manager.failure_status[1].item() & (1 << 4)

    manager.failure_status.zero_()
    req_to_token[1, 256] = 512
    full_to_swa = torch.zeros(2048, dtype=torch.int32, device="cuda")
    positions = torch.arange(128, 256, dtype=torch.int32, device="cuda")
    full_locations = 256 + positions
    full_to_swa[full_locations.to(torch.long)] = 128 + (positions - 128)
    swa_indices = BuildCausalSwaPageIndices.execute(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=torch.tensor([256], dtype=torch.int32, device="cuda"),
        swa_window=128,
        page_index_aligned_size=64,
        integrity_args=integrity_args,
    )
    assert swa_indices[0, 0].item() == 255
    assert swa_indices[0, 127].item() == 128
    assert manager.failure_status[1].item() == 0

    bad_full_location = 2 * 256 + 200
    req_to_token[1, 200] = bad_full_location
    full_to_swa[bad_full_location] = 2 * 128 + (200 % 128)
    swa_indices = BuildCausalSwaPageIndices.execute(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=torch.tensor([256], dtype=torch.int32, device="cuda"),
        swa_window=128,
        page_index_aligned_size=64,
        integrity_args=integrity_args,
    )
    assert swa_indices[0, 55].item() == 0
    assert manager.failure_status[1].item() & (1 << 4)

    manager.failure_status.zero_()
    extreme = torch.tensor(
        [torch.iinfo(torch.int32).max], dtype=torch.int32, device="cuda"
    )
    bounded_metadata = BuildPageTablePositions.execute(
        req_to_token=req_to_token,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=extreme,
        max_seq_len=512,
        page_size=256,
        swa_window=128,
        integrity_args=integrity_args,
    )
    assert bounded_metadata.seq_lens_casual.item() == 512
    assert bounded_metadata.positions_casual.item() == 511
    assert bounded_metadata.swa_topk_lengths.item() == 128

    bounded_swa = BuildCausalSwaPageIndices.execute(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=extreme,
        swa_window=128,
        page_index_aligned_size=64,
        integrity_args=integrity_args,
    )
    torch.cuda.synchronize()
    assert bounded_swa.shape == (1, 128)

    manager.failure_status.zero_()
    inactive_swa = BuildCausalSwaPageIndices.execute(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        req_pool_indices_repeated=request.to(torch.int32),
        seq_lens_casual=torch.tensor([-1], dtype=torch.int32, device="cuda"),
        swa_window=128,
        page_index_aligned_size=64,
        integrity_args=integrity_args,
    )
    torch.cuda.synchronize()
    assert torch.all(inactive_swa == -1)
    assert manager.failure_status[1].item() == 0


@pytest.mark.parametrize("component", list(DSV4Component))
def test_every_dsv4_component_bit_flip_fails_closed(component):
    manager, descriptors, page, logical, request = _complete_manager()
    descriptor = next(d for d in descriptors if d.component == component)

    assert (
        manager.validate_pages(
            descriptor, page, logical, request, slot_page_size=1
        ).item()
        == 1
    )
    descriptor.buffer[1, 0] ^= 1
    assert (
        manager.validate_pages(
            descriptor, page, logical, request, slot_page_size=1
        ).item()
        == 0
    )
    assert manager.failure_status[1].item() & (1 << 2)
    with pytest.raises(DSV4IntegrityError, match="rejected KV state"):
        manager.assert_clean()


def test_bit_flip_during_cuda_graph_replay_fails_closed():
    state = _state()
    slots, logical, reqs, _ = _bind(state)
    assert _validate(state, slots, logical, reqs).item() == 1
    state[-1].zero_()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        protected_slots = _validate(state, slots, logical, reqs)

    graph.replay()
    torch.cuda.synchronize()
    assert protected_slots.item() == 1

    state[0][1, 0] ^= 1
    state[-1].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert protected_slots.item() == 0
    assert state[-1][1].item() & (1 << 2)


def test_padded_slot_refresh_is_cuda_graph_safe():
    manager, descriptors, _, _, _ = _complete_manager()
    descriptor = descriptors[0]
    slots = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([True, False, True, False], device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_slots = torch.where(valid, slots, 0)
        manager.refresh_written_slots(descriptor, graph_slots, slot_page_size=1)

    graph.replay()
    torch.cuda.synchronize()
    assert manager.sidecars[descriptor.identity].valid[1].item() == 1
