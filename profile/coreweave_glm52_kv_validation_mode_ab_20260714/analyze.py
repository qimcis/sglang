#!/usr/bin/env python3

import json
import math
import statistics
import sys
from pathlib import Path

MODES = ("off", "scheduler", "fused")
CONCURRENCIES = (1, 2, 4, 8)
REPEATS = (1, 2)


def load(root: Path):
    results = {mode: {} for mode in MODES}
    totals = {"completed": 0, "failed": 0, "drain_timeouts": 0}
    configs = set()
    for mode in MODES:
        for repeat in REPEATS:
            with (root / mode / f"repeat{repeat}.json").open() as f:
                doc = json.load(f)
            configs.add(json.dumps(doc["config"], sort_keys=True))
            assert all(item["failed"] == 0 for item in doc["iterations"])
            assert all(not item["drain_timed_out"] for item in doc["iterations"])
            totals["completed"] += sum(item["completed"] for item in doc["iterations"])
            totals["failed"] += sum(item["failed"] for item in doc["iterations"])
            totals["drain_timeouts"] += sum(
                bool(item["drain_timed_out"]) for item in doc["iterations"]
            )
            results[mode][repeat] = {
                item["concurrency"]: item for item in doc["iterations"]
            }
    assert len(configs) == 1, "benchmark configurations differ"
    return results, json.loads(configs.pop()), totals


def geometric_delta(numerators, denominators):
    ratios = [a / b for a, b in zip(numerators, denominators)]
    return 100 * (math.exp(statistics.mean(map(math.log, ratios))) - 1)


def metric(results, mode, concurrency, getter):
    return [getter(results[mode][repeat][concurrency]) for repeat in REPEATS]


def main():
    root = Path(sys.argv[1])
    results, config, totals = load(root)
    rows = []
    for concurrency in CONCURRENCIES:
        throughput = {
            mode: metric(
                results, mode, concurrency, lambda item: item["total_throughput_tps"]
            )
            for mode in MODES
        }
        tpot = {
            mode: metric(
                results,
                mode,
                concurrency,
                lambda item: 1000 / item["per_request_tps"]["p50"],
            )
            for mode in MODES
        }
        ttft = {
            mode: metric(
                results, mode, concurrency, lambda item: item["ttft_ms"]["p50"]
            )
            for mode in MODES
        }
        rows.append(
            {
                "concurrency": concurrency,
                "off_tps": statistics.mean(throughput["off"]),
                "scheduler_tps": statistics.mean(throughput["scheduler"]),
                "fused_tps": statistics.mean(throughput["fused"]),
                "scheduler_vs_off_tps_pct": geometric_delta(
                    throughput["scheduler"], throughput["off"]
                ),
                "fused_vs_off_tps_pct": geometric_delta(
                    throughput["fused"], throughput["off"]
                ),
                "fused_vs_scheduler_tps_pct": geometric_delta(
                    throughput["fused"], throughput["scheduler"]
                ),
                "off_tpot_ms": statistics.mean(tpot["off"]),
                "scheduler_tpot_ms": statistics.mean(tpot["scheduler"]),
                "fused_tpot_ms": statistics.mean(tpot["fused"]),
                "scheduler_vs_off_tpot_pct": geometric_delta(
                    tpot["scheduler"], tpot["off"]
                ),
                "fused_vs_off_tpot_pct": geometric_delta(tpot["fused"], tpot["off"]),
                "off_ttft_ms": statistics.mean(ttft["off"]),
                "scheduler_ttft_ms": statistics.mean(ttft["scheduler"]),
                "fused_ttft_ms": statistics.mean(ttft["fused"]),
                "scheduler_vs_off_ttft_pct": geometric_delta(
                    ttft["scheduler"], ttft["off"]
                ),
                "fused_vs_off_ttft_pct": geometric_delta(ttft["fused"], ttft["off"]),
            }
        )

    summary = {
        "artifact_root": str(root),
        "config": config,
        "repeats": len(REPEATS),
        "totals": totals,
        "rows": rows,
    }

    report = [
        "# GLM-5.2 KV Validation Mode A/B",
        "",
        f"Artifact set: `{root.name}`.",
        f"All {totals['completed']} measured requests completed with "
        f"{totals['failed']} failures and {totals['drain_timeouts']} drain timeouts.",
        "Transfer checksum is enabled identically in all three arms.",
        "",
        "| C | OFF tok/s | Scheduler tok/s | Fused tok/s | Scheduler/OFF | Fused/OFF | Fused/Scheduler |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {concurrency} | {off_tps:.2f} | {scheduler_tps:.2f} | "
            "{fused_tps:.2f} | {scheduler_vs_off_tps_pct:+.3f}% | "
            "{fused_vs_off_tps_pct:+.3f}% | "
            "{fused_vs_scheduler_tps_pct:+.3f}% |".format(**row)
        )
    report.extend(
        [
            "",
            "| C | OFF TPOT ms | Scheduler TPOT ms | Fused TPOT ms | Scheduler/OFF | Fused/OFF |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        report.append(
            "| {concurrency} | {off_tpot_ms:.3f} | {scheduler_tpot_ms:.3f} | "
            "{fused_tpot_ms:.3f} | {scheduler_vs_off_tpot_pct:+.3f}% | "
            "{fused_vs_off_tpot_pct:+.3f}% |".format(**row)
        )
    report.extend(
        [
            "",
            "| C | OFF TTFT ms | Scheduler TTFT ms | Fused TTFT ms | Scheduler/OFF | Fused/OFF |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        report.append(
            "| {concurrency} | {off_ttft_ms:.3f} | {scheduler_ttft_ms:.3f} | "
            "{fused_ttft_ms:.3f} | {scheduler_vs_off_ttft_pct:+.3f}% | "
            "{fused_vs_off_ttft_pct:+.3f}% |".format(**row)
        )
    text = "\n".join(report) + "\n"
    for output_dir in (root, root.parent):
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output_dir / "SUMMARY.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
