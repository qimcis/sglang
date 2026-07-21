#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def key_for(row: dict) -> tuple[str, str, str, str]:
    return (
        row["layer_id"],
        row["bucket"],
        row["top_k"],
        row["num_tokens"],
    )


def build_winner_table(
    csv_paths: list[Path],
    min_win_pct: float,
    fallback_provider: str | None = None,
    provider_fallbacks: dict[str, str] | None = None,
) -> dict:
    groups = defaultdict(list)
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["median_ms"] = float(row["median_ms"])
                row["p20_ms"] = float(row["p20_ms"])
                row["p80_ms"] = float(row["p80_ms"])
                groups[key_for(row)].append(row)

    entries = []
    for key, rows in sorted(groups.items()):
        best = min(rows, key=lambda r: r["median_ms"])
        by_provider = {}
        for row in rows:
            provider = row["provider"]
            current = by_provider.get(provider)
            if current is None or row["median_ms"] < current["median_ms"]:
                by_provider[provider] = row

        providers = {
            provider: {
                "median_ms": row["median_ms"],
                "p20_ms": row["p20_ms"],
                "p80_ms": row["p80_ms"],
            }
            for provider, row in sorted(by_provider.items())
        }
        sorted_rows = sorted(by_provider.values(), key=lambda r: r["median_ms"])
        second = sorted_rows[1] if len(sorted_rows) > 1 else None
        win_pct = 0.0
        if second is not None and second["median_ms"] > 0:
            win_pct = (second["median_ms"] - best["median_ms"]) / second["median_ms"]

        layer_id, bucket, top_k, num_tokens = key
        entries.append(
            {
                "layer_id": None if layer_id == "None" else int(layer_id),
                "bucket": bucket,
                "top_k": int(top_k),
                "num_tokens": int(num_tokens),
                "winner": best["provider"] if win_pct >= min_win_pct else None,
                "best_provider": best["provider"],
                "best_median_ms": best["median_ms"],
                "second_best_provider": second["provider"] if second else None,
                "second_best_median_ms": second["median_ms"] if second else None,
                "win_pct": win_pct,
                "providers": providers,
            }
        )

    table = {
        "min_win_pct": min_win_pct,
        "entries": entries,
    }
    if fallback_provider:
        table["fallback_provider"] = fallback_provider
    if provider_fallbacks:
        table["provider_fallbacks"] = dict(sorted(provider_fallbacks.items()))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--min-win-pct", type=float, default=0.08)
    parser.add_argument("--fallback-provider")
    parser.add_argument(
        "--provider-fallback",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Map an unsafe selected provider to a safe provider in the output profile.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provider_fallbacks = {}
    for item in args.provider_fallback:
        if "=" not in item:
            parser.error(f"--provider-fallback must be FROM=TO, got {item!r}")
        source, target = item.split("=", 1)
        if not source or not target:
            parser.error(f"--provider-fallback must be FROM=TO, got {item!r}")
        provider_fallbacks[source] = target

    table = build_winner_table(
        args.csv,
        args.min_win_pct,
        fallback_provider=args.fallback_provider,
        provider_fallbacks=provider_fallbacks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(table, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output} entries={len(table['entries'])}")


if __name__ == "__main__":
    main()
