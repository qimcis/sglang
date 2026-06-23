"""Microbenchmark: Torch KV transfer checksum vs the fused CUDA kv_checksum op.

Compares the existing Python/Torch checksum path (one splitmix64 pass per int64
lane + XOR reduce) against the fused CUDA op for representative ``always_full``
byte-row shapes on H100.

Run on a GPU pod with a built sgl-kernel:

    python3 sgl-kernel/benchmark/bench_kv_checksum.py

Appends rows to ``benchmark.csv`` at the repo root (kept as KDA evidence).
"""

import csv
import os
import time

import torch

from sglang.srt.mem_cache.kv_page_tags import (
    _as_int64_lanes,
    _CKSUM_SEED,
    _hash_rows_with_positions_torch,
    _mix_scalar,
)

# (num_tokens, row_bytes) byte-row shapes from the task contract.
SHAPES = [
    (1024, 4096),
    (4096, 4096),
    (4096, 16384),
    (8192, 16384),
]

ITERS = 50
WARMUP = 10


def _torch_full_checksum(rows, indices):
    lanes = _as_int64_lanes(rows)
    sel = lanes.index_select(0, indices)
    return _hash_rows_with_positions_torch(sel, positions=indices, num_lanes=None)


def _cuda_full_checksum(kv_checksum, rows, indices):
    combined = kv_checksum(rows, indices, indices, -1)
    total = _mix_scalar(_CKSUM_SEED, combined)
    total = _mix_scalar(total, int(indices.numel()))
    return total


def _time(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; run this on a GPU pod.")
    try:
        from sgl_kernel.kvcacheio import kv_checksum
    except Exception as e:
        raise SystemExit(f"kv_checksum op unavailable (build sgl-kernel): {e}")

    dev = torch.device("cuda")
    rows_out = []
    print(f"{'shape':>16} {'torch_ms':>10} {'cuda_ms':>10} {'speedup':>8} {'match':>6}")
    for n, row_bytes in SHAPES:
        rows = torch.randint(0, 256, (n, row_bytes), dtype=torch.uint8, device=dev)
        indices = torch.arange(n, dtype=torch.long, device=dev)

        torch_ref = _torch_full_checksum(rows, indices)
        cuda_ref = _cuda_full_checksum(kv_checksum, rows, indices)
        match = torch_ref == cuda_ref

        torch_ms = _time(lambda: _torch_full_checksum(rows, indices))
        cuda_ms = _time(lambda: _cuda_full_checksum(kv_checksum, rows, indices))
        speedup = torch_ms / cuda_ms if cuda_ms > 0 else float("inf")
        print(
            f"{f'({n},{row_bytes})':>16} {torch_ms:>10.4f} {cuda_ms:>10.4f} "
            f"{speedup:>8.2f} {str(match):>6}"
        )
        rows_out.append(
            {
                "task": "kv_checksum",
                "shape": f"{n}x{row_bytes}",
                "num_tokens": n,
                "row_bytes": row_bytes,
                "torch_ms": round(torch_ms, 6),
                "cuda_ms": round(cuda_ms, 6),
                "speedup": round(speedup, 4),
                "match": bool(match),
                "device": torch.cuda.get_device_name(0),
            }
        )

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmark.csv",
    )
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nAppended {len(rows_out)} rows to {csv_path}")


if __name__ == "__main__":
    main()
