import pytest
import torch

from sgl_kernel.kvcacheio import kv_checksum


U64_MASK = (1 << 64) - 1
I64_SIGN = 1 << 63
SPLITMIX_ADD = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
CKSUM_SEED = 0x53474C414E474353


def to_i64(x: int) -> int:
    x &= U64_MASK
    return x - (1 << 64) if x & I64_SIGN else x


def splitmix64(x: int) -> int:
    x = (x + SPLITMIX_ADD) & U64_MASK
    z = x
    z = ((z ^ (z >> 30)) * SPLITMIX_M1) & U64_MASK
    z = ((z ^ (z >> 27)) * SPLITMIX_M2) & U64_MASK
    z ^= z >> 31
    return z & U64_MASK


def mix(acc: int, field: int) -> int:
    return splitmix64((acc ^ (field & U64_MASK)) & U64_MASK)


def reference_checksum(
    rows: torch.Tensor,
    row_indices: torch.Tensor,
    positions: torch.Tensor,
    num_lanes: int = -1,
    include_positions: bool = True,
) -> int:
    rows_cpu = rows.detach().cpu().contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
    row_indices_cpu = row_indices.detach().cpu().tolist()
    positions_cpu = positions.detach().cpu().tolist()
    if not row_indices_cpu:
        return to_i64(splitmix64(CKSUM_SEED))
    row_bytes = rows_cpu.shape[1]
    max_lanes = (row_bytes + 7) // 8
    if num_lanes < 0 or num_lanes > max_lanes:
        num_lanes = max_lanes

    combined = 0
    for out_i, row_i in enumerate(row_indices_cpu):
        acc = CKSUM_SEED
        if include_positions:
            acc = mix(acc, positions_cpu[out_i])
        row = rows_cpu[row_i]
        for lane in range(num_lanes):
            offset = lane * 8
            value = 0
            for byte_i in range(8):
                pos = offset + byte_i
                if pos < row_bytes:
                    value |= int(row[pos]) << (8 * byte_i)
            acc = mix(acc, value)
        combined ^= acc
    total = mix(CKSUM_SEED, combined)
    total = mix(total, len(row_indices_cpu))
    return to_i64(total)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("row_bytes", [7, 64, 4096])
@pytest.mark.parametrize("num_rows", [1, 33, 1024])
@pytest.mark.parametrize("include_positions", [False, True])
def test_kv_checksum_matches_reference(row_bytes, num_rows, include_positions):
    torch.manual_seed(0)
    rows = torch.randint(0, 256, (num_rows, row_bytes), dtype=torch.uint8, device="cuda")
    row_indices = torch.arange(num_rows, dtype=torch.long, device="cuda")
    if num_rows > 8:
        row_indices = row_indices[::3]
    positions = row_indices * 7 + 11

    actual = kv_checksum(rows, row_indices, positions, -1, include_positions)
    expected = reference_checksum(rows, row_indices, positions, -1, include_positions)
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kv_checksum_partial_lanes_and_row_order():
    rows = torch.arange(16 * 80, dtype=torch.uint8, device="cuda").reshape(16, 80)
    row_indices = torch.tensor([9, 1, 7, 3], dtype=torch.long, device="cuda")
    positions = torch.tensor([100, 200, 300, 400], dtype=torch.long, device="cuda")
    actual = kv_checksum(rows, row_indices, positions, 3, True)
    expected = reference_checksum(rows, row_indices, positions, 3, True)
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kv_checksum_detects_content_change():
    rows = torch.arange(32 * 128, dtype=torch.uint8, device="cuda").reshape(32, 128)
    row_indices = torch.arange(32, dtype=torch.long, device="cuda")
    positions = row_indices.clone()
    base = kv_checksum(rows, row_indices, positions, -1, True)
    rows[5, 17] ^= 0x7F
    changed = kv_checksum(rows, row_indices, positions, -1, True)
    assert base != changed
