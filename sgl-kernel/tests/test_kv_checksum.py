"""Tests for the fused CUDA KV transfer checksum op and its Torch fallback.

Two layers of testing:

1. ``TestKernelContract`` runs on CPU and proves, bit-for-bit, that the
   *algorithm* the CUDA kernel implements (a Python mirror of
   ``checksum.cu``: per-row splitmix64 fold over little-endian int64 lanes,
   XOR-reduced, plus the two scalar finishing mixes applied by the Python
   wrapper) equals the Torch reference ``hash_rows_with_positions`` /
   ``hash_kv_rows``. This validates the parity contract without a GPU.

2. ``TestCudaOp`` / ``TestCudaIntegration`` run only when CUDA + the built
   ``kv_checksum`` op are available, exercising the real kernel and the
   ``kv_page_tags`` integration path.

3. ``TestFallback`` proves the runtime falls back to the Torch path on CPU
   tensors, a missing op, and a runtime failure.
"""

import importlib

import pytest
import torch

from sglang.srt.mem_cache import kv_page_tags as kpt
from sglang.srt.mem_cache.kv_page_tags import (
    _as_int64_lanes,
    _CKSUM_SEED,
    _mix_scalar,
    _splitmix64_scalar,
    _U64_MASK,
    hash_kv_rows,
    hash_rows_with_positions,
)


def _cuda_kernel_reference(rows, row_indices, positions, num_lanes):
    """Pure-python mirror of checksum.cu + the wrapper finishing mixes.

    Reads the exact little-endian int64 lanes that ``_as_int64_lanes`` produces,
    folds per selected row with the scalar splitmix64, XOR-reduces across rows,
    then applies the two scalar finishing mixes. Returns the full uint64.
    """
    lanes = _as_int64_lanes(rows)
    total_lanes = lanes.shape[1]
    if num_lanes is not None and num_lanes >= 0:
        total_lanes = min(total_lanes, num_lanes)

    combined = 0
    idx_list = row_indices.tolist()
    pos_list = None if positions is None else positions.tolist()
    for t, ridx in enumerate(idx_list):
        acc = _CKSUM_SEED
        if pos_list is not None:
            acc = _mix_scalar(acc, int(pos_list[t]) & _U64_MASK)
        for j in range(total_lanes):
            lane = int(lanes[ridx, j].item()) & _U64_MASK
            acc = _mix_scalar(acc, lane)
        combined ^= acc
    # ``combined`` here is an unsigned XOR fold; the kernel returns the signed
    # int64 bit pattern, but ``_mix_scalar`` masks to uint64 so either is fine.
    total = _mix_scalar(_CKSUM_SEED, combined)
    total = _mix_scalar(total, len(idx_list))
    return total


def _make_rows(shape, dtype=torch.int32, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min, info.max, shape, generator=g, dtype=torch.int64
    ).to(dtype)


HAS_CUDA = torch.cuda.is_available()


def _cuda_op_available():
    if not HAS_CUDA:
        return False
    try:
        from sgl_kernel.kvcacheio import kv_checksum  # noqa: F401
    except Exception:
        return False
    return hasattr(torch.ops.sgl_kernel, "kv_checksum")


HAS_CUDA_OP = _cuda_op_available()


class TestKernelContract:
    """CPU parity: the kernel algorithm == the Torch reference, bit-for-bit."""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((1, 8), torch.int32),
            ((4, 8), torch.int32),
            ((16, 4096 // 4), torch.int32),  # 4096 bytes/row
            ((33, 257), torch.int16),  # odd row width -> trailing partial lane
            ((7, 5), torch.uint8),  # 5-byte rows -> 1 partial lane
            ((8, 128), torch.float32),
        ],
    )
    def test_full_hash_matches_reference(self, shape, dtype):
        rows = _make_rows(shape, dtype, seed=shape[0])
        indices = torch.arange(shape[0], dtype=torch.long)
        ref = hash_kv_rows(rows, indices)
        kern = _cuda_kernel_reference(rows, indices, indices, None)
        assert ref == kern

    def test_partial_lanes_match_reference(self):
        rows = _make_rows((32, 64), torch.int32, seed=3)  # 256 bytes -> 32 lanes
        indices = torch.arange(32, dtype=torch.long)
        for num_lanes in (1, 4, 8, 31, 32, 100):
            ref = hash_kv_rows(rows, indices, num_lanes=num_lanes)
            kern = _cuda_kernel_reference(rows, indices, indices, num_lanes)
            assert ref == kern, f"num_lanes={num_lanes}"

    def test_sampled_indices_match_reference(self):
        rows = _make_rows((100, 64), torch.int32, seed=5)
        indices = torch.arange(0, 100, 3, dtype=torch.long)  # non-contiguous stride
        ref = hash_kv_rows(rows, indices)
        kern = _cuda_kernel_reference(rows, indices, indices, None)
        assert ref == kern

    def test_no_position_match_reference(self):
        rows = _make_rows((20, 64), torch.int32, seed=7)
        indices = torch.arange(20, dtype=torch.long)
        ref = hash_kv_rows(rows, indices, include_positions=False)
        kern = _cuda_kernel_reference(rows, indices, None, None)
        assert ref == kern

    def test_one_d_rows_match_reference(self):
        rows = _make_rows((48,), torch.int32, seed=9)  # 1D treated as [48, 1]
        indices = torch.arange(48, dtype=torch.long)
        ref = hash_kv_rows(rows, indices)
        kern = _cuda_kernel_reference(rows, indices, indices, None)
        assert ref == kern

    def test_empty_selection(self):
        rows = _make_rows((10, 64), torch.int32, seed=11)
        indices = torch.empty(0, dtype=torch.long)
        assert hash_kv_rows(rows, indices) == _splitmix64_scalar(_CKSUM_SEED)

    def test_reorder_changes_checksum(self):
        # Order sensitivity comes from position-binding: the same bytes at a
        # different logical position must hash differently. (Permuting the
        # *selection order* with positions == indices is XOR-commutative and is
        # intentionally NOT detected -- logical order is layout-independent.)
        rows = _make_rows((16, 64), torch.int32, seed=13)
        perm = torch.randperm(16, generator=torch.Generator().manual_seed(1))
        rows2 = rows[perm].contiguous()
        idx = torch.arange(16, dtype=torch.long)
        assert hash_kv_rows(rows, idx) != hash_kv_rows(rows2, idx)

    def test_returns_uint64_range(self):
        rows = _make_rows((16, 64), torch.int32, seed=15)
        indices = torch.arange(16, dtype=torch.long)
        v = hash_kv_rows(rows, indices)
        assert 0 <= v <= _U64_MASK


@pytest.mark.skipif(not HAS_CUDA_OP, reason="CUDA kv_checksum op not built")
class TestCudaOp:
    """Real kernel parity (requires a CUDA build of sgl-kernel)."""

    @pytest.mark.parametrize(
        "shape", [(1, 1024), (1024, 1024), (4096, 1024)]
    )
    def test_op_matches_torch_reference(self, shape):
        from sgl_kernel.kvcacheio import kv_checksum

        rows = _make_rows(shape, torch.int32, seed=shape[0]).cuda()
        indices = torch.arange(shape[0], dtype=torch.long, device="cuda")
        combined = kv_checksum(rows, indices, indices, -1)
        total = _mix_scalar(_CKSUM_SEED, combined)
        total = _mix_scalar(total, shape[0])
        # Torch reference on CPU copy (CUDA path disabled by .cpu()).
        ref = hash_kv_rows(rows.cpu(), indices.cpu())
        assert total == ref


@pytest.mark.skipif(not HAS_CUDA_OP, reason="CUDA kv_checksum op not built")
class TestCudaIntegration:
    """kv_page_tags hash_* functions on CUDA must match the CPU Torch path."""

    @pytest.mark.parametrize("num_lanes", [None, 1, 8, 64])
    def test_hash_kv_rows_cuda_matches_cpu(self, num_lanes):
        rows = _make_rows((512, 256), torch.int32, seed=21)
        indices = torch.arange(0, 512, 2, dtype=torch.long)
        cpu = hash_kv_rows(rows, indices, num_lanes=num_lanes)
        gpu = hash_kv_rows(
            rows.cuda(), indices.cuda(), num_lanes=num_lanes
        )
        assert cpu == gpu

    def test_hash_rows_with_positions_cuda_matches_cpu(self):
        rows = _make_rows((300, 128), torch.int32, seed=23)
        positions = torch.arange(300, dtype=torch.long)
        cpu = hash_rows_with_positions(rows, positions=positions)
        gpu = hash_rows_with_positions(rows.cuda(), positions=positions.cuda())
        assert cpu == gpu


class TestFallback:
    """The runtime must fall back to the Torch path safely."""

    def test_cpu_tensor_returns_none(self):
        rows = _make_rows((8, 64), torch.int32, seed=31)
        indices = torch.arange(8, dtype=torch.long)
        assert (
            kpt._try_cuda_checksum(rows, indices, indices, None, num_rows=8)
            is None
        )

    def test_cpu_hash_matches_reference_without_op(self):
        # On CPU the op is never used; hash_kv_rows must equal the contract.
        rows = _make_rows((40, 64), torch.int32, seed=33)
        indices = torch.arange(40, dtype=torch.long)
        assert hash_kv_rows(rows, indices) == _cuda_kernel_reference(
            rows, indices, indices, None
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA tensor")
    def test_runtime_failure_falls_back(self, monkeypatch):
        rows = _make_rows((8, 64), torch.int32, seed=35).cuda()
        indices = torch.arange(8, dtype=torch.long, device="cuda")

        import sgl_kernel.kvcacheio as kvio

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated kernel failure")

        monkeypatch.setattr(kvio, "kv_checksum", _boom)
        # _try_cuda_checksum imports kv_checksum from the module, so patching
        # the module attribute makes the call raise -> fallback (None).
        result = kpt._try_cuda_checksum(rows, indices, indices, None, num_rows=8)
        assert result is None

    @pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA tensor")
    def test_more_than_2d_falls_back(self):
        rows = torch.zeros((4, 4, 4), dtype=torch.int32, device="cuda")
        indices = torch.arange(4, dtype=torch.long, device="cuda")
        assert (
            kpt._try_cuda_checksum(rows, indices, indices, None, num_rows=4)
            is None
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
