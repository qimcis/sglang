#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROMPTS = [
    "Write a Python function to compute Fibonacci numbers.",
    "Explain why mixture-of-experts inference can have GPU stragglers.",
    "Return JSON with keys name and value for a test object.",
    "List three practical ways to reduce long-context LLM serving latency.",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def send_one(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - start
    row = {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "e2e_ms": elapsed * 1000,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "output_chars": 0,
        "error": "",
    }
    if response.status_code != 200:
        row["error"] = response.text[:500]
        return row

    payload = response.json()
    usage = payload.get("usage") or {}
    row["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
    row["completion_tokens"] = int(usage.get("completion_tokens") or 0)
    row["total_tokens"] = int(usage.get("total_tokens") or 0)
    choices = payload.get("choices") or []
    if choices:
        row["output_chars"] = len(choices[0].get("text") or "")
    if row["completion_tokens"]:
        row["tpot_ms"] = row["e2e_ms"] / row["completion_tokens"]
        row["output_tok_s"] = row["completion_tokens"] / elapsed
    else:
        row["tpot_ms"] = 0.0
        row["output_tok_s"] = 0.0
    return row


def summarize(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r["ok"]]
    e2e = [r["e2e_ms"] for r in ok_rows]
    tpot = [r["tpot_ms"] for r in ok_rows if r.get("tpot_ms")]
    output_tok_s = [r["output_tok_s"] for r in ok_rows if r.get("output_tok_s")]
    completion_tokens = sum(r["completion_tokens"] for r in ok_rows)
    return {
        "requests": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "completion_tokens": completion_tokens,
        "e2e_ms_p50": statistics.median(e2e) if e2e else 0.0,
        "e2e_ms_p95": percentile(e2e, 0.95),
        "e2e_ms_p99": percentile(e2e, 0.99),
        "tpot_ms_p50": statistics.median(tpot) if tpot else 0.0,
        "tpot_ms_p95": percentile(tpot, 0.95),
        "output_tok_s_mean": statistics.mean(output_tok_s) if output_tok_s else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    url = args.url.rstrip("/") + "/v1/completions"
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send_one, url, args.model, p, args.max_tokens) for p in prompts
        ]
        for future in as_completed(futures):
            row = future.result()
            row["backend"] = args.backend
            rows.append(row)

    summary = summarize(rows)
    summary["backend"] = args.backend
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted(rows[0].keys()) if rows else ["backend"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
