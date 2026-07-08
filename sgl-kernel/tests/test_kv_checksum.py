"""Parity tests for the production KV transfer checksum CUDA op.

The public ABI is intentionally limited to ``kv_checksum_direct_table_batched``.
These tests compare that batched op against the Python reference hash for plain
MHA/MLA buffers and SWA buffers translated through the full-to-SWA mapping.
"""

import pytest
import torch

from sglang.srt.mem_cache.kv_page_tags import (
    _CKSUM32_SEED,
    KVPageProtectionManager,
    KVProtectionConfig,
    _fmix32_scalar,
    hash_rows_with_positions,
    select_checksum_byte_count,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KV checksum CUDA op requires CUDA"
)


def _have_batched_op() -> bool:
    try:
        from sgl_kernel.kvcacheio import kv_checksum_direct_table_batched  # noqa: F401

        return True
    except Exception:
        return False


class _Pool:
    """Minimal KV pool exposing per-layer K and optional V buffers."""

    def __init__(self, k_buffers, v_buffers=None):
        self.layer_num = len(k_buffers)
        self._k = k_buffers
        self._v = v_buffers

    def get_key_buffer(self, layer_id):
        return self._k[layer_id]

    def get_value_buffer(self, layer_id):
        if self._v is None:
            raise NotImplementedError
        return self._v[layer_id]


class _SWAPool:
    """Minimal hybrid pool with global layer order and SWA loc translation."""

    def __init__(self, full_k, full_v, swa_k, swa_v, full_to_swa_index_mapping):
        self.layer_num = len(full_k) + len(swa_k)
        self.full_k = full_k
        self.full_v = full_v
        self.swa_k = swa_k
        self.swa_v = swa_v
        self.full_to_swa_index_mapping = full_to_swa_index_mapping
        self.layers_mapping = {}
        full_idx = 0
        swa_idx = 0
        for layer_id in range(self.layer_num):
            if layer_id % 2 == 0:
                self.layers_mapping[layer_id] = (full_idx, False)
                full_idx += 1
            else:
                self.layers_mapping[layer_id] = (swa_idx, True)
                swa_idx += 1

    def get_key_buffer(self, layer_id):
        local_id, is_swa = self.layers_mapping[layer_id]
        return self.swa_k[local_id] if is_swa else self.full_k[local_id]

    def get_value_buffer(self, layer_id):
        local_id, is_swa = self.layers_mapping[layer_id]
        return self.swa_v[local_id] if is_swa else self.full_v[local_id]


def _rows_for_buffer(buf, locs):
    return buf.index_select(0, locs).reshape(locs.numel(), -1)


def _reference(pool, req_to_token, req_idx, start, length):
    if length == 0:
        return _fmix32_scalar(_CKSUM32_SEED)

    positions = torch.arange(start, start + length, dtype=torch.long, device="cuda")
    full_locs = req_to_token[req_idx, start : start + length].to(torch.long)
    rows = []
    layers_mapping = getattr(pool, "layers_mapping", None)
    for layer_id in range(pool.layer_num):
        locs = full_locs
        if layers_mapping is not None and bool(layers_mapping[layer_id][1]):
            mapping = pool.full_to_swa_index_mapping
            locs = mapping.index_select(0, full_locs).clamp_min(0)
        rows.append(_rows_for_buffer(pool.get_key_buffer(layer_id), locs))
        try:
            rows.append(_rows_for_buffer(pool.get_value_buffer(layer_id), locs))
        except (NotImplementedError, AttributeError):
            pass

    selected = torch.cat(rows, dim=1)
    row_nbytes = selected.contiguous().view(torch.uint8).reshape(length, -1).shape[1]
    num_lanes = select_checksum_byte_count(row_nbytes)
    return hash_rows_with_positions(selected, positions=positions, num_lanes=num_lanes)


def _batched(pool, req_to_token, req_indices, starts, lengths):
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    manager = KVPageProtectionManager(
        config=cfg,
        allocator=None,
        num_pages=1,
        page_size=1,
        device="cuda",
        transfer_backend="mooncake",
    )
    batch = manager.begin_transfer_checksums_from_table(
        pool,
        req_to_token,
        req_pool_indices=req_indices,
        bootstrap_rooms=[100 + i for i in range(len(req_indices))],
        num_tokens=lengths,
        starts=starts,
    )
    assert batch is not None
    return [plan.checksum for plan in batch.finalize()]


def _one(pool, req_to_token, req_idx, start, length):
    return _batched(pool, req_to_token, [req_idx], [start], [length])[0]


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_removed_direct_abi_is_not_exported():
    import sgl_kernel.kvcacheio as kvcacheio

    assert not hasattr(kvcacheio, "kv_checksum_direct")
    assert not hasattr(kvcacheio, "kv_checksum_direct_range")
    assert hasattr(kvcacheio, "kv_checksum_direct_table_batched")


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
def test_mha_batched_parity(dtype):
    torch.manual_seed(0)
    size, h, d, layers, tokens = 256, 4, 16, 6, 64
    if dtype == torch.int8:
        k = [
            torch.randint(-120, 120, (size, h, d), dtype=dtype, device="cuda")
            for _ in range(layers)
        ]
        v = [
            torch.randint(-120, 120, (size, h, d), dtype=dtype, device="cuda")
            for _ in range(layers)
        ]
    else:
        k = [torch.randn(size, h, d, dtype=dtype, device="cuda") for _ in range(layers)]
        v = [torch.randn(size, h, d, dtype=dtype, device="cuda") for _ in range(layers)]
    pool = _Pool(k, v)
    req_to_token = torch.empty((3, tokens), dtype=torch.int32, device="cuda")
    for req_idx in range(req_to_token.shape[0]):
        req_to_token[req_idx].copy_(torch.randperm(size, device="cuda")[:tokens])

    starts = [0, 3, 11]
    lengths = [32, 40, 21]
    got = _batched(pool, req_to_token, [0, 1, 2], starts, lengths)
    expected = [
        _reference(pool, req_to_token, req_idx, start, length)
        for req_idx, start, length in zip([0, 1, 2], starts, lengths, strict=True)
    ]
    assert got == expected


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_mla_k_only_batched_parity():
    torch.manual_seed(1)
    size, lora, layers, tokens = 256, 64, 4, 48
    k = [
        torch.randn(size, 1, lora, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    pool = _Pool(k, v_buffers=None)
    req_to_token = (
        torch.randperm(size, device="cuda")[:tokens].reshape(1, -1).to(torch.int32)
    )

    got = _one(pool, req_to_token, 0, 0, tokens)
    expected = _reference(pool, req_to_token, 0, 0, tokens)
    assert got == expected


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_same_logical_bytes_different_slots_match():
    torch.manual_seed(2)
    size, h, d, layers, tokens = 128, 2, 16, 3, 16
    src_k = [
        torch.randn(size, h, d, dtype=torch.float16, device="cuda")
        for _ in range(layers)
    ]
    src_v = [
        torch.randn(size, h, d, dtype=torch.float16, device="cuda")
        for _ in range(layers)
    ]
    dst_k = [torch.zeros_like(b) for b in src_k]
    dst_v = [torch.zeros_like(b) for b in src_v]
    src_locs = torch.arange(tokens, device="cuda")
    dst_locs = torch.arange(size - tokens, size, device="cuda")
    for layer_id in range(layers):
        dst_k[layer_id][dst_locs] = src_k[layer_id][src_locs]
        dst_v[layer_id][dst_locs] = src_v[layer_id][src_locs]

    src_table = src_locs.reshape(1, -1).to(torch.int32)
    dst_table = dst_locs.reshape(1, -1).to(torch.int32)
    a = _one(_Pool(src_k, src_v), src_table, 0, 0, tokens)
    b = _one(_Pool(dst_k, dst_v), dst_table, 0, 0, tokens)
    assert a == b


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_batched_detects_corruption_and_reorder():
    torch.manual_seed(3)
    size, h, d, layers, tokens = 128, 2, 16, 2, 32
    k = [
        torch.randn(size, h, d, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    v = [
        torch.randn(size, h, d, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    pool = _Pool(k, v)
    base_locs = torch.randperm(size, device="cuda")[:tokens]
    req_to_token = torch.empty((2, tokens), dtype=torch.int32, device="cuda")
    req_to_token[0].copy_(base_locs)
    req_to_token[1].copy_(base_locs.flip(0))

    ordered, reordered = _batched(pool, req_to_token, [0, 1], [0, 0], [tokens, tokens])
    assert ordered != reordered

    k[0][base_locs[5], 0, 0] += 1.0
    corrupted = _one(pool, req_to_token, 0, 0, tokens)
    assert corrupted != ordered


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_empty_selection():
    k = [torch.randn(16, 2, 16, dtype=torch.float16, device="cuda")]
    pool = _Pool(k, None)
    req_to_token = torch.arange(16, device="cuda", dtype=torch.int32).reshape(1, -1)
    got = _one(pool, req_to_token, 0, 0, 0)
    assert got == _fmix32_scalar(_CKSUM32_SEED)


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_swa_batched_parity_uses_full_to_swa_mapping():
    torch.manual_seed(4)
    full_size, swa_size, h, d, tokens = 256, 128, 2, 16, 48
    full_layers = 2
    swa_layers = 2
    full_k = [
        torch.randn(full_size, h, d, dtype=torch.float16, device="cuda")
        for _ in range(full_layers)
    ]
    full_v = [torch.randn_like(buf) for buf in full_k]
    swa_k = [
        torch.randn(swa_size, h, d, dtype=torch.float16, device="cuda")
        for _ in range(swa_layers)
    ]
    swa_v = [torch.randn_like(buf) for buf in swa_k]

    full_locs = torch.randperm(full_size, device="cuda")[:tokens]
    swa_locs = torch.randperm(swa_size - 1, device="cuda")[:tokens] + 1
    mapping = torch.zeros(full_size + 1, dtype=torch.long, device="cuda")
    mapping[full_locs] = swa_locs
    pool = _SWAPool(full_k, full_v, swa_k, swa_v, mapping)
    req_to_token = full_locs.reshape(1, -1).to(torch.int32)

    got = _one(pool, req_to_token, 0, 0, tokens)
    expected = _reference(pool, req_to_token, 0, 0, tokens)
    assert got == expected

    swa_k[0][swa_locs[7], 0, 0] += 1.0
    corrupted = _one(pool, req_to_token, 0, 0, tokens)
    assert corrupted != got


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
def test_mixed_alignment_and_odd_lane_cap():
    """Exercise 16B vector loads, scalar fallback, and odd cap at buffer boundary."""
    torch.manual_seed(5)
    size, h, tokens = 128, 2, 32
    # K row: 64 bytes (8 x 8B lanes) with 16B-aligned stride -> vector path.
    k = [torch.randint(-120, 120, (size, h, 32), dtype=torch.int8, device="cuda")]
    # V row: 24 bytes (3 x 8B lanes) with 24-byte stride -> scalar path.
    v = [torch.randint(-120, 120, (size, h, 12), dtype=torch.int8, device="cuda")]
    pool = _Pool(k, v)
    req_to_token = (
        torch.randperm(size, device="cuda")[:tokens].reshape(1, -1).to(torch.int32)
    )

    got = _one(pool, req_to_token, 0, 0, tokens)
    expected = _reference(pool, req_to_token, 0, 0, tokens)
    assert got == expected

    # Corrupt the scalar-fallback V buffer and ensure checksum changes.
    v[0][req_to_token[0, 5].item(), 0, 0] += 1
    corrupted = _one(pool, req_to_token, 0, 0, tokens)
    assert corrupted != got


@pytest.mark.skipif(
    not _have_batched_op(),
    reason="sgl_kernel.kv_checksum_direct_table_batched not built",
)
@pytest.mark.parametrize("num_lanes", [6, 5])
def test_direct_odd_lane_vector_tail_and_scalar_fallback(num_lanes):
    """Exercise the vector path's odd-lane tail and the scalar fallback directly.

    The production metadata builder rejects non-contiguous rows, which makes it
    impossible to get a 16B-aligned buffer with an odd lane count through the
    public API.  We call the op directly with raw metadata so we can stage a
    16B-aligned 24-byte row (3 lanes, odd -> vector tail) alongside a 24-byte
    row whose 24-byte stride is not 16B aligned (scalar fallback).

    ``num_lanes=6`` is the full 48-byte row (uncapped); ``num_lanes=5`` clips
    the second buffer to 2 lanes and exercises the ``kCapped=true`` template.
    """
    torch.manual_seed(6 + num_lanes)
    from sgl_kernel.kvcacheio import kv_checksum_direct_table_batched

    size, tokens = 64, 31
    # Buffer 0: 24 useful bytes per row, 32-byte aligned stride -> 16B vector
    # path with limit=3 (odd), forcing the scalar tail inside the vector loop.
    vec = torch.randint(-120, 120, (size, 4, 8), dtype=torch.int8, device="cuda")
    vec[:, 3:, :].zero_()  # padding after the first 24 bytes per row
    # Buffer 1: 24 bytes per row, 24-byte stride -> scalar fallback.
    scalar = torch.randint(-120, 120, (size, 3, 8), dtype=torch.int8, device="cuda")

    buffer_ptrs = torch.tensor(
        [vec.data_ptr(), scalar.data_ptr()],
        dtype=torch.int64,
        device="cuda",
    )
    row_strides = torch.tensor(
        [vec.stride(0), scalar.stride(0)], dtype=torch.int64, device="cuda"
    )
    row_nbytes = torch.tensor([24, 24], dtype=torch.int64, device="cuda")
    swa_buffer_flags = torch.zeros(2, dtype=torch.int64, device="cuda")
    full_to_swa_index_mapping = torch.empty(0, dtype=torch.int64, device="cuda")
    req_to_token = torch.arange(tokens, dtype=torch.int32, device="cuda").reshape(1, -1)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cuda")
    starts = torch.tensor([0], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([tokens], dtype=torch.int64, device="cuda")
    accum = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    is_capped = (num_lanes * 8) < 48

    kv_checksum_direct_table_batched(
        buffer_ptrs,
        row_strides,
        row_nbytes,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        req_to_token,
        req_pool_indices,
        starts,
        lengths,
        tokens,
        num_lanes,
        False,  # has_swa
        is_capped,
        accum,
        out,
    )

    # Reference: first 24 bytes of each vector row followed by the scalar row.
    positions = torch.arange(tokens, dtype=torch.long, device="cuda")
    selected = torch.cat(
        [
            vec[:tokens, :3, :].reshape(tokens, 24),
            scalar[:tokens].reshape(tokens, 24),
        ],
        dim=1,
    )
    expected = hash_rows_with_positions(
        selected, positions=positions, num_lanes=num_lanes
    )
    assert int(out[0].item()) & 0xFFFFFFFF == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
