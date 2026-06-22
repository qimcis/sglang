import argparse
import json
import statistics
import time

import torch

from sgl_kernel.kvcacheio import kv_checksum


def torch_reference(rows, row_indices, positions, num_lanes):
    # Import the runtime helper only for benchmark comparison.  The CUDA kernel
    # test has an independent scalar reference.
    from sglang.srt.mem_cache.kv_page_tags import hash_rows_with_positions

    selected = rows.index_select(0, row_indices)
    return hash_rows_with_positions(selected, positions=positions, num_lanes=num_lanes)


def measure(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    vals = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000)
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--shapes", nargs="*", default=["1024x4096", "4096x4096", "4096x16384", "8192x16384"])
    args = parser.parse_args()

    torch.manual_seed(0)
    out = []
    for shape in args.shapes:
        n_str, b_str = shape.lower().split("x")
        n, row_bytes = int(n_str), int(b_str)
        rows = torch.randint(0, 256, (n, row_bytes), dtype=torch.uint8, device="cuda")
        row_indices = torch.arange(n, dtype=torch.long, device="cuda")
        positions = row_indices
        num_lanes = row_bytes // 8

        ref_vals = measure(lambda: torch_reference(rows, row_indices, positions, num_lanes), args.repeats, args.warmup)
        cuda_vals = measure(lambda: kv_checksum(rows, row_indices, positions, num_lanes, True), args.repeats, args.warmup)
        out.append(
            {
                "tokens": n,
                "row_bytes": row_bytes,
                "mb": n * row_bytes / 1e6,
                "torch_ms_avg": statistics.mean(ref_vals),
                "torch_ms_p50": statistics.median(ref_vals),
                "cuda_ms_avg": statistics.mean(cuda_vals),
                "cuda_ms_p50": statistics.median(cuda_vals),
                "speedup_avg": statistics.mean(ref_vals) / statistics.mean(cuda_vals),
            }
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
