#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        if path.is_dir():
            yield from iter_jsonl(sorted(path.glob("*.jsonl")))
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def num_tokens(row: dict) -> int:
    top_k = int(row.get("top_k") or 1)
    return int(row.get("total_assignments") or 0) // max(top_k, 1)


def summarize(paths: list[Path], top_n: int, min_num_tokens: int):
    groups = defaultdict(list)
    bucket_counts = Counter()
    rows = list(iter_jsonl(paths))
    for row in rows:
        n_tokens = num_tokens(row)
        if n_tokens < min_num_tokens:
            continue
        key = (
            row.get("layer_id"),
            row.get("bucket", "unknown"),
            row.get("top_k"),
            n_tokens,
        )
        groups[key].append(row)
        bucket_counts[key] += 1

    print(f"records: {len(rows)}")
    print(f"groups: {len(groups)}")
    print()
    print("top regimes by count:")
    print(
        "count,layer_id,bucket,top_k,num_tokens,"
        "imbalance_p50,imbalance_p95,imbalance_p99,"
        "active_p50,active_p95,singleton_frac_p50,singleton_frac_p95"
    )
    for key, count in bucket_counts.most_common(top_n):
        members = groups[key]
        imbalances = [float(r.get("imbalance_ratio", 0.0)) for r in members]
        active = [float(r.get("active_experts", 0.0)) for r in members]
        singleton = [float(r.get("singleton_frac", 0.0)) for r in members]
        layer_id, bucket, top_k, n_tokens = key
        print(
            f"{count},{layer_id},{bucket},{top_k},{n_tokens},"
            f"{median(imbalances):.4f},{percentile(imbalances, 0.95):.4f},"
            f"{percentile(imbalances, 0.99):.4f},"
            f"{median(active):.1f},{percentile(active, 0.95):.1f},"
            f"{median(singleton):.4f},{percentile(singleton, 0.95):.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-num-tokens", type=int, default=1)
    args = parser.parse_args()
    summarize(args.paths, args.top_n, args.min_num_tokens)


if __name__ == "__main__":
    main()
