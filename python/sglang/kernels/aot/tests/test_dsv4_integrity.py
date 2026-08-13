import pytest
import torch
from sgl_kernel.kvcacheio import (
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
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
    expected_pages = torch.zeros((4, 4), dtype=torch.int32, device=device)
    expected_generations = torch.zeros((4, 4), dtype=torch.int64, device=device)
    expected_valid = torch.zeros((4, 4), dtype=torch.int32, device=device)
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
        expected_pages,
        expected_generations,
        expected_valid,
        failure_status,
    )


def _bind(state, slot=1):
    _, _, _, _, _, _, _, generations, pages, page_generations, valid, status = state
    slots = torch.tensor([[slot]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        pages,
        page_generations,
        valid,
        output,
        status,
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
        pages,
        page_generations,
        valid,
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
        pages,
        page_generations,
        valid,
        output,
        status,
        99,
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
    assert state[-1][1].item() & (1 << 5)


def test_duplicate_bind_is_graph_safe_and_conflicts_fail():
    state = _state()
    _, _, _, _, _, _, _, generations, pages, page_generations, valid, status = state
    slots = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device="cuda")
    logical = torch.zeros_like(slots, dtype=torch.int64)
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        pages,
        page_generations,
        valid,
        output,
        status,
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
        pages,
        page_generations,
        valid,
        output,
        status,
        1,
        0,
        True,
    )
    assert output[0, -1].item() == 0
    assert status[1].item() & (1 << 4)


def test_verify_only_does_not_learn_first_observed_mapping():
    state = _state()
    _, _, _, _, _, _, _, generations, pages, page_generations, valid, status = state
    slots = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        pages,
        page_generations,
        valid,
        output,
        status,
        1,
        0,
        False,
    )
    assert output.item() == 0
    assert status[1].item() & (1 << 3)
    assert valid[1, 0].item() == 0


def test_live_negative_slot_fails_instead_of_being_treated_as_padding():
    state = _state()
    _, _, _, _, _, _, _, generations, pages, page_generations, valid, status = state
    slots = torch.tensor([[-1]], dtype=torch.int32, device="cuda")
    logical = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    reqs = torch.tensor([1], dtype=torch.int64, device="cuda")
    output = torch.empty_like(slots)
    dsv4_bind_pages(
        slots,
        logical,
        reqs,
        generations,
        pages,
        page_generations,
        valid,
        output,
        status,
        1,
        0,
        False,
    )
    assert output.item() == 0
    assert status[1].item() & (1 << 0)


def test_request_slot_reuse_clears_mappings_status_and_bumps_state_generation():
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
        space.expected_valid[1].fill_(1)
    manager.register_requests([1], [1])
    assert manager.failure_status[1].item() == 0
    assert all(
        not space.expected_valid[1].any().item()
        for space in manager.address_spaces.values()
    )
    state_generation = manager.address_spaces[DSV4IntegrityDomain.C128_STATE].generation
    assert state_generation[1].item() == 1
    manager.register_requests([1], [2])
    assert state_generation[1].item() == 2


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
