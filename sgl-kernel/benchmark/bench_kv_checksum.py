"""Benchmark legacy root checksums against root plus logical-page digests.

Run on a CUDA host with a freshly built ``sgl_kernel``:

    python3 sgl-kernel/benchmark/bench_kv_checksum.py
"""

import argparse
import csv
import os
import time

import torch
from sgl_kernel.kvcacheio import (
    kv_checksum_direct_table_batched,
    kv_checksum_direct_table_batched_with_pages,
)

from sglang.srt.mem_cache.kv_page_tags import (
    _direct_metadata_from_pool,
    _DirectKVChecksumCache,
    select_checksum_byte_count,
)


class _Pool:
    def __init__(self, k, v=None):
        self.layer_num = len(k)
        self._k = k
        self._v = v

    def get_key_buffer(self, layer_id):
        return self._k[layer_id]

    def get_value_buffer(self, layer_id):
        if self._v is None:
            raise NotImplementedError
        return self._v[layer_id]


def _time(fn, *, iterations=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _make_pool(layers, heads, head_dim, dtype, slots, mla):
    if mla:
        k = [
            torch.randn(slots, 1, head_dim, dtype=dtype, device="cuda")
            for _ in range(layers)
        ]
        return _Pool(k), "mla"
    k = [
        torch.randn(slots, heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(layers)
    ]
    v = [torch.randn_like(buffer) for buffer in k]
    return _Pool(k, v), "mha"


def _benchmark_config(layers, heads, head_dim, dtype, slots, tokens, batch, mla):
    pool, layout = _make_pool(layers, heads, head_dim, dtype, slots, mla)
    req_to_token = torch.stack(
        [torch.randperm(slots, device="cuda")[:tokens] for _ in range(batch)]
    ).to(torch.int32)
    req_pool_indices = torch.arange(batch, dtype=torch.int64, device="cuda")
    starts = torch.zeros(batch, dtype=torch.int64, device="cuda")
    lengths = torch.full((batch,), tokens, dtype=torch.int64, device="cuda")
    logical_starts = starts.clone()

    cache = _DirectKVChecksumCache()

    def unsupported(reason):
        raise RuntimeError(reason)

    (
        buffer_ptrs,
        row_strides,
        row_nbytes,
        swa_buffer_flags,
        full_to_swa_index_mapping,
        total_bytes,
    ) = _direct_metadata_from_pool(pool, req_to_token.device, cache, unsupported)
    num_lanes = select_checksum_byte_count(total_bytes)
    accum = torch.zeros(batch, dtype=torch.int32, device="cuda")
    root_out = torch.empty(batch, dtype=torch.int64, device="cuda")
    page_size = 64
    page_count = (tokens + page_size - 1) // page_size
    page_accum = torch.zeros((batch, page_count), dtype=torch.int64, device="cuda")
    page_out = torch.empty_like(page_accum)

    def root_only():
        accum.zero_()
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
            False,
            False,
            accum,
            root_out,
        )

    def root_with_pages():
        accum.zero_()
        page_accum.zero_()
        kv_checksum_direct_table_batched_with_pages(
            buffer_ptrs,
            row_strides,
            row_nbytes,
            swa_buffer_flags,
            full_to_swa_index_mapping,
            req_to_token,
            req_pool_indices,
            starts,
            lengths,
            logical_starts,
            tokens,
            num_lanes,
            page_size,
            page_count,
            False,
            False,
            accum,
            root_out,
            page_accum,
            page_out,
        )

    root_only()
    expected_root = root_out.clone()
    root_with_pages()
    torch.testing.assert_close(root_out, expected_root, rtol=0, atol=0)

    root_us = _time(root_only)
    pages_us = _time(root_with_pages)
    overhead = (pages_us / root_us - 1) * 100
    print(
        f"{layout} L={layers} H={heads} D={head_dim} N={tokens} B={batch}: "
        f"root={root_us:.1f}us pages={pages_us:.1f}us overhead={overhead:+.2f}%"
    )
    common = {
        "device": torch.cuda.get_device_name(0),
        "layout": layout,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "tokens": tokens,
        "batch": batch,
    }
    return [
        {**common, "path": "legacy_root", "latency_us": root_us, "overhead_percent": 0},
        {
            **common,
            "path": "root_with_pages",
            "latency_us": pages_us,
            "overhead_percent": overhead,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configs = [
        (32, 8, 128, torch.bfloat16, 8192, 512, 8, False),
        (32, 8, 128, torch.bfloat16, 8192, 2048, 4, False),
        (61, 1, 576, torch.bfloat16, 8192, 4096, 4, True),
    ]
    rows = []
    for config in configs:
        rows.extend(_benchmark_config(*config))

    fieldnames = list(rows[0])
    exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
