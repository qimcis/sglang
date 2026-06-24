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
    ChecksumMode,
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


def _reference(pool, kv_loc, indices, mode, cfg):
    rows = gather_logical_kv_rows(pool, kv_loc, indices)
    row_nbytes = (
        rows.contiguous().view(torch.uint8).reshape(rows.shape[0], -1).shape[1]
        if rows.numel()
        else 0
    )
    num_lanes = select_checksum_byte_count(
        row_nbytes, mode, cfg.checksum_partial_byte_rate
    )
    return hash_rows_with_positions(rows, positions=indices, num_lanes=num_lanes)


def _run(pool, kv_loc, num_tokens, mode, cfg, room=99):
    indices = select_checksum_token_indices(
        num_tokens, room, mode, cfg.checksum_sample_rate
    )
    ref = _reference(pool, kv_loc, indices, mode, cfg)
    got = direct_kv_checksum_from_loc(pool, kv_loc, indices, mode=mode, config=cfg)
    return ref, got


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
@pytest.mark.parametrize(
    "mode",
    [ChecksumMode.ALWAYS_FULL, ChecksumMode.SAMPLED_FULL, ChecksumMode.SAMPLED_PARTIAL],
)
def test_mha_parity(dtype, mode):
    torch.manual_seed(0)
    size, h, d, L, N = 256, 4, 16, 6, 48
    cfg = KVProtectionConfig(checksum_mode=mode, checksum_direct_kernel="auto")
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
    ref, got = _run(pool, kv_loc, N, mode, cfg)
    assert got is not None, "direct path unexpectedly fell back"
    assert ref == got, f"mismatch: ref={ref} got={got}"


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
@pytest.mark.parametrize(
    "mode", [ChecksumMode.ALWAYS_FULL, ChecksumMode.SAMPLED_PARTIAL]
)
def test_mla_k_only_parity(mode):
    """MLA-style pool: get_value_buffer raises -> only K is hashed."""
    torch.manual_seed(1)
    size, lora, L, N = 256, 64, 4, 40
    cfg = KVProtectionConfig(checksum_mode=mode, checksum_direct_kernel="auto")
    k = [
        torch.randn(size, 1, lora, dtype=torch.bfloat16, device="cuda")
        for _ in range(L)
    ]
    pool = _Pool(k, v_buffers=None)
    kv_loc = torch.randperm(size, device="cuda")[:N].contiguous()
    ref, got = _run(pool, kv_loc, N, mode, cfg)
    assert got is not None
    assert ref == got


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_same_logical_bytes_different_slots_match():
    torch.manual_seed(2)
    size, h, d, L, N = 128, 2, 16, 3, 12
    cfg = KVProtectionConfig(
        checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="auto"
    )
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
    a = direct_kv_checksum_from_loc(
        _Pool(src_k, src_v), loc_src, idx, mode=cfg.checksum_mode, config=cfg
    )
    b = direct_kv_checksum_from_loc(
        _Pool(dst_k, dst_v), loc_dst, idx, mode=cfg.checksum_mode, config=cfg
    )
    assert a is not None and a == b


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_corruption_detected():
    torch.manual_seed(3)
    size, h, d, L, N = 128, 2, 16, 2, 16
    cfg = KVProtectionConfig(
        checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="auto"
    )
    k = [torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)]
    v = [torch.randn(size, h, d, dtype=torch.float16, device="cuda") for _ in range(L)]
    kv_loc = torch.arange(N, device="cuda")
    idx = torch.arange(N)
    base = direct_kv_checksum_from_loc(
        _Pool(k, v), kv_loc, idx, mode=cfg.checksum_mode, config=cfg
    )
    k[0][5, 1, 3] += 1.0
    bad = direct_kv_checksum_from_loc(
        _Pool(k, v), kv_loc, idx, mode=cfg.checksum_mode, config=cfg
    )
    assert base != bad


@pytest.mark.skipif(not _have_op(), reason="sgl_kernel.kv_checksum_direct not built")
def test_empty_selection():
    cfg = KVProtectionConfig(
        checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="auto"
    )
    k = [torch.randn(16, 2, 16, dtype=torch.float16, device="cuda")]
    pool = _Pool(k, None)
    kv_loc = torch.arange(16, device="cuda")
    idx = torch.arange(0)  # empty
    got = direct_kv_checksum_from_loc(
        pool, kv_loc, idx, mode=cfg.checksum_mode, config=cfg
    )
    from sglang.srt.mem_cache.kv_page_tags import _CKSUM_SEED, _splitmix64_scalar

    assert got == _splitmix64_scalar(_CKSUM_SEED)


def test_cpu_falls_back_and_strict_raises():
    """No CUDA op needed: auto -> None on CPU pool, strict -> RuntimeError."""
    k = [torch.randn(16, 2, 8, dtype=torch.float32)]
    pool = _Pool(k, None)
    kv_loc = torch.arange(16)
    idx = torch.arange(4)
    auto = KVProtectionConfig(
        checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="auto"
    )
    assert (
        direct_kv_checksum_from_loc(
            pool, kv_loc, idx, mode=auto.checksum_mode, config=auto
        )
        is None
    )
    strict = KVProtectionConfig(
        checksum_mode=ChecksumMode.ALWAYS_FULL, checksum_direct_kernel="strict"
    )
    with pytest.raises(RuntimeError):
        direct_kv_checksum_from_loc(
            pool, kv_loc, idx, mode=strict.checksum_mode, config=strict
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
