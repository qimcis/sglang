import dataclasses
from typing import NamedTuple

import pytest
import torch

from sglang.srt.mem_cache.dsv4_kv_integrity import (
    DSV4Component,
    DSV4ComponentDescriptor,
    DSV4IntegrityDomain,
    DSV4IntegrityError,
    DSV4KVIntegrityManager,
    DSV4ManifestEntry,
    DSV4ManifestError,
    DSV4TransferGroup,
    DSV4TransferManifest,
    causal_swa_logical_pages,
    layout_fingerprint,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _entry(*, logical_start=3, digests=(0, 1, 2)):
    return DSV4ManifestEntry(
        component=DSV4Component.C4_ATTENTION_KV,
        layer_id=7,
        compress_ratio=4,
        transfer_group=DSV4TransferGroup.KV,
        page_size=64,
        item_nbytes=64 * 584,
        logical_start=logical_start,
        logical_count=len(digests),
        digests=digests,
    )


def _manifest(entries=None):
    return DSV4TransferManifest(
        bootstrap_room=123,
        transfer_nonce=456,
        layout_digest=bytes(range(32)),
        entries=tuple((_entry(),) if entries is None else entries),
    )


def test_manifest_round_trip_preserves_unsigned_digests():
    manifest = _manifest([_entry(digests=(0, (1 << 63) + 9, (1 << 64) - 1))])
    assert DSV4TransferManifest.from_bytes(manifest.to_bytes()) == manifest


def test_empty_manifest_round_trip_for_zero_byte_transfer():
    manifest = _manifest([])
    assert DSV4TransferManifest.from_bytes(manifest.to_bytes()) == manifest


@pytest.mark.parametrize("cut", [0, 1, 7, 31, 63])
def test_manifest_rejects_truncation(cut):
    payload = _manifest().to_bytes()
    with pytest.raises(DSV4ManifestError):
        DSV4TransferManifest.from_bytes(payload[:cut])


def test_manifest_rejects_trailing_bytes_and_unknown_magic():
    payload = _manifest().to_bytes()
    with pytest.raises(DSV4ManifestError, match="trailing"):
        DSV4TransferManifest.from_bytes(payload + b"x")
    with pytest.raises(DSV4ManifestError, match="magic"):
        DSV4TransferManifest.from_bytes(b"BADMAGIC" + payload[8:])


def test_manifest_rejects_duplicate_component_layer_across_ranges():
    duplicate = dataclasses.replace(_entry(), logical_start=99)
    with pytest.raises(DSV4ManifestError, match="duplicate"):
        _manifest([_entry(), duplicate]).to_bytes()


def test_manifest_rejects_geometry_and_digest_count_drift():
    with pytest.raises(DSV4ManifestError, match="digest count"):
        _manifest([dataclasses.replace(_entry(), logical_count=4)]).to_bytes()
    with pytest.raises(DSV4ManifestError, match="geometry"):
        _manifest([dataclasses.replace(_entry(), page_size=0)]).to_bytes()


def test_layout_fingerprint_binds_component_geometry():
    buffer = torch.zeros((2, 16), dtype=torch.uint8)
    descriptor = DSV4ComponentDescriptor(
        component=DSV4Component.SWA_KV,
        layer_id=0,
        compress_ratio=0,
        transfer_group=DSV4TransferGroup.SWA,
        page_size=256,
        item_nbytes=16,
        capacity=2,
        buffer=buffer,
    )
    changed = dataclasses.replace(descriptor, page_size=128)
    assert layout_fingerprint([descriptor]) != layout_fingerprint([changed])


def test_causal_swa_logical_pages_follow_newest_to_oldest_index_order():
    logical = causal_swa_logical_pages(
        torch.tensor([200, 2]),
        torch.tensor([128, 2]),
        width=4,
        page_size=128,
    )
    assert logical.tolist() == [[1, 1, 1, 1], [0, 0, -1, -1]]


class _RecordingManager(DSV4KVIntegrityManager):
    def __init__(self):
        self.failure_status = torch.zeros(4, dtype=torch.int32)
        self.max_context_len = 16
        self.full_page_size = 4
        self.swa_page_size = 2
        self._full_to_swa_mapping = torch.arange(32, dtype=torch.int64)
        self.mapping_calls = []
        self.validation_calls = []

    def replace_pages(
        self,
        domain,
        slots,
        logical_pages,
        request_indices,
        *,
        slot_page_size,
        invalid_value=0,
    ):
        self.mapping_calls.append(
            (
                domain,
                slots.clone(),
                logical_pages.clone(),
                request_indices.clone(),
                slot_page_size,
                invalid_value,
            )
        )
        return slots

    def validate_pages(
        self,
        descriptor,
        slots,
        logical_pages,
        request_indices,
        *,
        slot_page_size,
        invalid_value=0,
    ):
        self.validation_calls.append(
            ("validate", logical_pages.clone(), request_indices.clone())
        )
        return slots

    def verify_mapping(
        self,
        domain,
        slots,
        logical_pages,
        request_indices,
        *,
        slot_page_size,
        invalid_value=0,
    ):
        self.validation_calls.append(
            ("write", logical_pages.clone(), request_indices.clone())
        )
        return slots


def test_trusted_request_table_sliced_write_binds_full_and_swa_pages():
    manager = _RecordingManager()
    manager.bind_request_table_write(
        (2, slice(4, 8)), torch.arange(4, 8, dtype=torch.int32)
    )

    full, swa = manager.mapping_calls
    assert full[0] == DSV4IntegrityDomain.FULL
    assert full[1].shape == (1, 4)
    assert full[2].tolist() == [[1, 1, 1, 1]]
    assert full[3].tolist() == [2]
    assert swa[0] == DSV4IntegrityDomain.SWA
    assert swa[2].tolist() == [[2, 2, 3, 3]]


def test_trusted_request_table_paired_advanced_write_preserves_pairs():
    manager = _RecordingManager()
    manager.bind_request_table_write(
        (
            torch.tensor([1, 2]),
            torch.tensor([3, 7]),
        ),
        torch.tensor([3, 7], dtype=torch.int32),
    )

    full = manager.mapping_calls[0]
    assert full[1].tolist() == [[3], [7]]
    assert full[2].tolist() == [[0], [1]]
    assert full[3].tolist() == [1, 2]


def test_trusted_request_table_rejects_strided_slice():
    manager = _RecordingManager()
    with pytest.raises(DSV4IntegrityError, match="strided"):
        manager.bind_request_table_write(
            (1, slice(0, 4, 2)), torch.tensor([0, 2], dtype=torch.int32)
        )


class _PrefillPlan(NamedTuple):
    is_decode: bool
    plan_c: torch.Tensor
    plan_w: torch.Tensor


class _DecodePlan(NamedTuple):
    is_decode: bool
    plan_d: torch.Tensor


def _state_descriptor(ratio):
    component = (
        DSV4Component.C4_ATTENTION_STATE
        if ratio == 4
        else DSV4Component.C128_ATTENTION_STATE
    )
    group = DSV4TransferGroup.SWA if ratio == 4 else DSV4TransferGroup.C128_STATE
    page_size = 8 if ratio == 4 else 128
    return DSV4ComponentDescriptor(
        component=component,
        layer_id=0,
        compress_ratio=ratio,
        transfer_group=group,
        page_size=page_size,
        item_nbytes=16,
        capacity=8,
        buffer=torch.zeros((8, 16), dtype=torch.uint8),
    )


def _prefill_plan(seq_len, buffer_len):
    packed = (buffer_len << 16) | 0
    plan_c = torch.tensor([[seq_len, packed, 8, 9]], dtype=torch.int32)
    plan_w = torch.tensor([[0, 10]], dtype=torch.int32)
    return _PrefillPlan(False, plan_c, plan_w)


def test_cold_prefill_does_not_validate_uninitialized_c4_state():
    manager = _RecordingManager()
    manager.protect_compressor_plan(
        _state_descriptor(4),
        _prefill_plan(seq_len=4, buffer_len=4),
        torch.tensor([1]),
        torch.tensor([3]),
    )
    reads = [call[1].tolist() for call in manager.validation_calls[:2]]
    assert reads == [[-1], [-1]]


def test_warm_prefill_validates_both_c4_state_pages():
    manager = _RecordingManager()
    manager.protect_compressor_plan(
        _state_descriptor(4),
        _prefill_plan(seq_len=8, buffer_len=5),
        torch.tensor([1]),
        torch.tensor([7]),
    )
    reads = [call[1].tolist() for call in manager.validation_calls[:2]]
    assert reads == [[1], [3]]


def test_decode_validates_state_only_on_compression_boundary():
    manager = _RecordingManager()
    descriptor = _state_descriptor(4)
    manager.protect_compressor_plan(
        descriptor,
        _DecodePlan(True, torch.tensor([[5, 10, 8, 9]], dtype=torch.int32)),
        torch.tensor([1]),
        torch.tensor([4]),
    )
    assert [call[1].tolist() for call in manager.validation_calls[:2]] == [
        [-1],
        [-1],
    ]

    manager.validation_calls.clear()
    manager.protect_compressor_plan(
        descriptor,
        _DecodePlan(True, torch.tensor([[8, 10, 8, 9]], dtype=torch.int32)),
        torch.tensor([1]),
        torch.tensor([7]),
    )
    assert [call[1].tolist() for call in manager.validation_calls[:2]] == [
        [1],
        [3],
    ]


def test_c128_prefill_never_validates_unused_read_page_zero():
    manager = _RecordingManager()
    manager.protect_compressor_plan(
        _state_descriptor(128),
        _prefill_plan(seq_len=128, buffer_len=1),
        torch.tensor([1]),
        torch.tensor([127]),
    )
    reads = [call[1].tolist() for call in manager.validation_calls[:2]]
    assert reads == [[-1], [0]]
