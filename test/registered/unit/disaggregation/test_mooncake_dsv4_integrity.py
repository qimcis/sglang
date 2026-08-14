import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.mem_cache.dsv4_kv_integrity import DSV4TransferGroup
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_protected_pd_warmup_fake_receiver_accepts_integrity_metadata():
    receiver = FakeKVReceiver(None, "fake")
    receiver.send_metadata(
        np.asarray([7], dtype=np.int32),
        decode_prefix_len=0,
        device_kv_indices=np.asarray([7], dtype=np.int32),
        request_index=1,
        request_seq_len=256,
    )
    assert receiver.has_sent_metadata


def _manager():
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.dsv4_integrity = SimpleNamespace(
        full_page_size=256,
        swa_page_size=256,
        max_context_len=1024,
        request_capacity=4,
        request_epochs=torch.tensor([0, 3, 0, 0], dtype=torch.int64),
    )
    manager._integrity_lock = threading.Lock()
    manager._integrity_condition = threading.Condition(manager._integrity_lock)
    manager._integrity_source_indices = {}
    manager._integrity_destination_indices = {}
    manager._integrity_nonces = {}
    manager._integrity_verified_ranks = {}
    manager._integrity_seen_manifests = {}
    manager._integrity_pending_success_ranks = {}
    manager._integrity_inflight_rooms = set()
    manager._integrity_cancelled_rooms = set()
    manager.kv_args = SimpleNamespace(state_types=[StateType.SWA, StateType.C128_STATE])
    return manager


def _record(
    manager,
    pages,
    *,
    index_slice=slice(0, 1),
    is_last_chunk=False,
    state_indices=None,
    request_index=1,
    request_epoch=3,
):
    manager.record_integrity_source_chunk(
        99,
        np.asarray(pages, dtype=np.int32),
        state_indices,
        index_slice=index_slice,
        is_last_chunk=is_last_chunk,
        num_kv_tokens=256 * len(pages),
        request_index=request_index,
        request_epoch=request_epoch,
    )


def test_protected_mooncake_retry_requires_identical_pages_and_identity():
    manager = _manager()
    _record(manager, [7])
    _record(manager, [7])

    with pytest.raises(RuntimeError, match="changed source pages"):
        _record(manager, [8])
    with pytest.raises(RuntimeError, match="identity changed"):
        _record(manager, [7], request_epoch=4)


def test_protected_mooncake_rejects_overlapping_chunks():
    manager = _manager()
    _record(manager, [7], index_slice=slice(0, 1))
    with pytest.raises(RuntimeError, match="overlapping"):
        _record(manager, [8, 9], index_slice=slice(0, 2))


def test_protected_mooncake_invalid_first_chunk_leaves_no_retry_state():
    manager = _manager()
    with pytest.raises(RuntimeError, match="slice length"):
        _record(manager, [7], index_slice=slice(0, 2))
    assert manager._integrity_source_indices == {}


def test_protected_mooncake_retry_requires_identical_final_state():
    manager = _manager()
    state = [np.asarray([3], dtype=np.int32), np.asarray([1], dtype=np.int32)]
    _record(manager, [7], is_last_chunk=True, state_indices=state)
    _record(manager, [7], is_last_chunk=True, state_indices=state)

    changed = [np.asarray([4], dtype=np.int32), np.asarray([1], dtype=np.int32)]
    with pytest.raises(RuntimeError, match="changed source state pages"):
        _record(manager, [7], is_last_chunk=True, state_indices=changed)


def test_protected_mooncake_destination_registration_is_atomic_and_epoch_bound():
    manager = _manager()
    with pytest.raises(RuntimeError, match="page count"):
        manager.register_integrity_destination(
            99,
            123,
            np.asarray([7, 8], dtype=np.int32),
            None,
            decode_prefix_len=0,
            request_seq_len=256,
            request_index=1,
        )
    assert manager._integrity_nonces == {}
    assert manager._integrity_destination_indices == {}

    manager.register_integrity_destination(
        99,
        123,
        np.asarray([7], dtype=np.int32),
        None,
        decode_prefix_len=0,
        request_seq_len=256,
        request_index=1,
    )
    assert manager._integrity_nonces == {99: 123}
    assert manager._integrity_destination_indices[99][2:] == (1, 3)


def test_protected_mooncake_validates_source_pages_in_logical_order():
    manager = _manager()
    calls = []
    clean = []
    manager.dsv4_integrity.failure_status = torch.zeros(4, dtype=torch.int32)
    manager.dsv4_integrity.domain_for_group = lambda group: group

    def verify(domain, pages, logical, requests, *, slot_page_size):
        calls.append(
            (
                domain,
                pages.tolist(),
                logical.tolist(),
                requests.tolist(),
                slot_page_size,
            )
        )

    manager.dsv4_integrity.verify_mapping = verify
    manager.dsv4_integrity.assert_request_clean = clean.append
    groups = {
        DSV4TransferGroup.KV: np.asarray([7, 8], dtype=np.int32),
        DSV4TransferGroup.SWA: np.asarray([3], dtype=np.int32),
        DSV4TransferGroup.C128_STATE: np.asarray([1], dtype=np.int32),
    }
    logical_starts = {
        DSV4TransferGroup.KV: 4,
        DSV4TransferGroup.SWA: 9,
        DSV4TransferGroup.C128_STATE: 0,
    }

    manager._validate_integrity_source_mappings(groups, logical_starts, 1)

    assert calls == [
        (DSV4TransferGroup.KV, [[7, 8]], [[4, 5]], [1], 1),
        (DSV4TransferGroup.SWA, [[3]], [[9]], [1], 1),
        (DSV4TransferGroup.C128_STATE, [[1]], [[0]], [1], 1),
    ]
    assert clean == [1]


def test_protected_mooncake_room_cleanup_clears_all_lifecycle_state():
    manager = _manager()
    manager._integrity_source_indices[99] = object()
    manager._integrity_destination_indices[99] = object()
    manager._integrity_nonces[99] = 123
    manager._integrity_verified_ranks[99] = {0}
    manager._integrity_seen_manifests[99] = {(0, 123)}
    manager._integrity_pending_success_ranks[99] = {0}
    manager.clear_integrity_room(99)
    assert all(
        99 not in mapping
        for mapping in (
            manager._integrity_source_indices,
            manager._integrity_destination_indices,
            manager._integrity_nonces,
            manager._integrity_verified_ranks,
            manager._integrity_seen_manifests,
            manager._integrity_pending_success_ranks,
        )
    )
