"""Parity tests for the direct-KV transfer checksum CUDA kernel.

The direct kernel (`kv_checksum_direct`) hashes K/V cache bytes straight from the
per-layer buffers in logical token order, WITHOUT materializing the
`[selected_tokens, row_bytes]` tensor that `gather_logical_kv_rows` builds in the
Torch reference path.  These tests assert bit-for-bit parity against that
reference for the supported (contiguous, 8-byte-aligned MHA/MLA) layouts.

Requires a CUDA device and a built `sgl_kernel` with `kv_checksum_direct`.
"""

import pytest
import torch

from sglang.srt.mem_cache.kv_page_tags import (
    KVProtectionConfig,
    direct_kv_checksum_from_loc,
    gather_logical_kv_rows,
    hash_rows_with_positions,
    select_checksum_byte_count,
    select_checksum_token_indices,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="direct KV checksum kernel requires CUDA"
)


def _have_op() -> bool:
    try:
        from sgl_kernel.kvcacheio import kv_checksum_direct  # noqa: F401

        return True
    except Exception:
        return False


class _Pool:
    """Minimal KV pool exposing per-layer K (and optional V) buffers."""

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


def _reference(pool, kv_loc, indices):
    rows = gather_logical_kv_rows(pool, kv_loc, indices)
    row_nbytes = (
        rows.contiguous().view(torch.uint8).reshape(rows.shape[0], -1).shape[1]
        if rows.numel()
        else 0
    )
    num_lanes = select_checksum_byte_count(row_nbytes)
    return hash_rows_with_positions(rows, positions=indices, num_lanes=num_lanes)


def _run(pool, kv_loc, num_tokens, cfg, room=99):
    indices = select_checksum_token_indices(num_tokens, room, 1.0)
    ref = _reference(pool, kv_loc, indices)
    got = direct_kv_checksum_from_loc(pool, kv_loc, indices, config=cfg)
    return ref, got


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
def test_mha_parity(dtype):
    torch.manual_seed(0)
    size, h, d, L, N = 256, 4, 16, 6, 48
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    if dtype == torch.int8:
        k = [
            torch.randint(-120, 120, (size, h, d), dtype=dtype, device="cuda")
            for _ in range(L)
        ]
        v = [
            torch.randint(-120, 120, (size, h, d), dtype=dtype, device="cuda")
            for _ in range(L)
        ]
    else:
        k = [torch.randn(size, h, d, dtype=dtype, device="cuda") for _ in range(L)]
        v = [torch.randn(size, h, d, dtype=dtype, device="cuda") for _ in range(L)]
    pool = _Pool(k, v)
    kv_loc = torch.randperm(size, device="cuda")[:N].contiguous()
    ref, got = _run(pool, kv_loc, N, cfg)
    assert got is not None, "direct path unexpectedly fell back"
    assert ref == got, f"mismatch: ref={ref} got={got}"


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_mla_k_only_parity():
    """MLA-style pool: get_value_buffer raises -> only K is hashed."""
    torch.manual_seed(1)
    size, lora, L, N = 256, 64, 4, 40
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    k = [
        torch.randn(size, 1, lora, dtype=torch.bfloat16, device="cuda")
        for _ in range(L)
    ]
    pool = _Pool(k, v_buffers=None)
    kv_loc = torch.randperm(size, device="cuda")[:N].contiguous()
    ref, got = _run(pool, kv_loc, N, cfg)
    assert got is not None
    assert ref == got


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_same_logical_bytes_different_slots_match():
    torch.manual_seed(2)
    size, h, d, L, N = 128, 2, 16, 3, 12
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    src_k = [
        torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)
    ]
    src_v = [
        torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)
    ]
    loc_src = torch.arange(0, N, device="cuda")
    loc_dst = torch.arange(size - N, size, device="cuda")
    dst_k = [torch.zeros_like(b) for b in src_k]
    dst_v = [torch.zeros_like(b) for b in src_v]
    for l in range(L):
        dst_k[l][loc_dst] = src_k[l][loc_src]
        dst_v[l][loc_dst] = src_v[l][loc_src]
    idx = torch.arange(N)
    a = direct_kv_checksum_from_loc(_Pool(src_k, src_v), loc_src, idx, config=cfg)
    b = direct_kv_checksum_from_loc(_Pool(dst_k, dst_v), loc_dst, idx, config=cfg)
    assert a is not None and a == b


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_corruption_detected():
    torch.manual_seed(3)
    size, h, d, L, N = 128, 2, 16, 2, 16
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    k = [torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)]
    v = [torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)]
    kv_loc = torch.arange(N, device="cuda")
    idx = torch.arange(N)
    base = direct_kv_checksum_from_loc(_Pool(k, v), kv_loc, idx, config=cfg)
    k[0][5, 1, 3] += 1.0
    bad = direct_kv_checksum_from_loc(_Pool(k, v), kv_loc, idx, config=cfg)
    assert base != bad


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_empty_selection():
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    k = [torch.randn(16, 2, 16, dtype=torch.float16, device="cuda")]
    pool = _Pool(k, None)
    kv_loc = torch.arange(16, device="cuda")
    idx = torch.arange(0)  # empty
    got = direct_kv_checksum_from_loc(pool, kv_loc, idx, config=cfg)
    from sglang.srt.mem_cache.kv_page_tags import _CKSUM_SEED, _splitmix64_scalar

    assert got == _splitmix64_scalar(_CKSUM_SEED)


def test_cpu_requires_direct_kernel():
    """No CUDA op needed: CPU pool cannot run the required direct kernel."""
    k = [torch.randn(16, 2, 8, dtype=torch.float32)]
    pool = _Pool(k, None)
    kv_loc = torch.arange(16)
    idx = torch.arange(4)
    cfg = KVProtectionConfig(enable_transfer_checksum=True)
    with pytest.raises(RuntimeError):
        direct_kv_checksum_from_loc(pool, kv_loc, idx, config=cfg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
