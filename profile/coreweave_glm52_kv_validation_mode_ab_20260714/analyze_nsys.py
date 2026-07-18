#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path

MODES = ("off", "scheduler", "fused")
KERNEL_PATTERNS = {
    "unprotected_topk": "topk_transform_decode_kernel(",
    "protected_topk_64": "topk_transform_decode_protected_kernel<(bool)1>",
    "begin_status": "kv_page_protection_begin_forward_kernel",
    "failure_status": "kv_page_protection_failure_status_kernel",
    "status_allreduce": "ncclDevKernel_AllReduce_Sum_u32_RING_LL",
    "checksum_scan": "kv_checksum_direct_table_batched_kernel",
    "checksum_finalizer": "kv_checksum_finalize_requests_with_pages_kernel",
}
API_NAMES = ("cudaGraphLaunch", "cudaLaunchKernel", "cudaMemcpyAsync")


def read_csv(path: Path):
    lines = path.read_text().splitlines()
    header = next(
        index for index, line in enumerate(lines) if line.startswith("Time (%)")
    )
    return list(csv.DictReader(lines[header:]))


def aggregate(rows, pattern):
    selected = [row for row in rows if pattern in row["Name"]]
    instances = sum(
        int(row["Instances"] if "Instances" in row else row["Num Calls"])
        for row in selected
    )
    total_ns = sum(int(row["Total Time (ns)"]) for row in selected)
    return {
        "instances": instances,
        "total_us": total_ns / 1000,
        "avg_us": total_ns / instances / 1000 if instances else 0,
    }


def aggregate_exact(rows, name):
    return aggregate([row for row in rows if row["Name"] == name], name)


def main():
    root = Path(sys.argv[1])
    kernel_rows = {mode: read_csv(root / mode / "cuda-kernels.csv") for mode in MODES}
    api_rows = {mode: read_csv(root / mode / "cuda-api.csv") for mode in MODES}
    kernels = {
        mode: {
            name: aggregate(kernel_rows[mode], pattern)
            for name, pattern in KERNEL_PATTERNS.items()
        }
        for mode in MODES
    }
    apis = {
        mode: {name: aggregate_exact(api_rows[mode], name) for name in API_NAMES}
        for mode in MODES
    }
    total_kernel_instances = {
        mode: sum(int(row["Instances"]) for row in kernel_rows[mode]) for mode in MODES
    }

    for mode in MODES:
        with (root / mode / "workload.json").open() as file:
            workload = json.load(file)
        assert all(item["completed"] > 0 for item in workload["iterations"])
        assert all(item["failed"] == 0 for item in workload["iterations"])
        assert all(not item["drain_timed_out"] for item in workload["iterations"])

    assert kernels["off"]["protected_topk_64"]["instances"] == 0
    assert kernels["scheduler"]["protected_topk_64"]["instances"] == 0
    assert kernels["fused"]["unprotected_topk"]["instances"] == 0
    assert kernels["off"]["begin_status"]["instances"] == 0
    assert kernels["scheduler"]["begin_status"]["instances"] == 0

    begin_count = kernels["fused"]["begin_status"]["instances"]
    status_count = kernels["fused"]["failure_status"]["instances"]
    allreduce_count = kernels["fused"]["status_allreduce"]["instances"]
    topk_count = kernels["fused"]["protected_topk_64"]["instances"]
    assert begin_count == apis["fused"]["cudaGraphLaunch"]["instances"]
    assert status_count == allreduce_count
    assert topk_count % status_count == 0
    assert {kernels[mode]["checksum_scan"]["instances"] for mode in MODES} == {8}
    assert {kernels[mode]["checksum_finalizer"]["instances"] for mode in MODES} == {8}

    topk_per_step = topk_count / status_count
    protected_topk_delta_us = topk_per_step * (
        kernels["fused"]["protected_topk_64"]["avg_us"]
        - kernels["scheduler"]["unprotected_topk"]["avg_us"]
    )
    local_status_us = (
        kernels["fused"]["begin_status"]["avg_us"]
        + kernels["fused"]["failure_status"]["avg_us"]
    )
    status_collective_us = kernels["fused"]["status_allreduce"]["avg_us"]
    scheduler_extra_kernels_per_step = (
        total_kernel_instances["scheduler"] - total_kernel_instances["fused"]
    ) / status_count

    summary = {
        "kernel_rows": kernels,
        "cuda_api_rows": apis,
        "total_kernel_instances": total_kernel_instances,
        "derived": {
            "graph_launches": apis["fused"]["cudaGraphLaunch"]["instances"],
            "begin_status_instances": begin_count,
            "completed_status_instances": status_count,
            "protected_topk_instances_per_status_step": topk_per_step,
            "protected_topk_increment_us_per_status_step": protected_topk_delta_us,
            "local_status_us_per_status_step": local_status_us,
            "status_collective_us_per_status_step": status_collective_us,
            "attributed_fused_increment_us_per_status_step": (
                protected_topk_delta_us + local_status_us + status_collective_us
            ),
            "scheduler_extra_gpu_kernels_per_status_step": (
                scheduler_extra_kernels_per_step
            ),
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# Nsight Systems Attribution",
        "",
        "All modes use the optimized wheels and the same TP8 GLM-5.2 C1 "
        "8172-by-1000 workload. Nsight captured CUDA graph nodes and NCCL "
        "for 40 requested decode steps; profiler throughput is not used as "
        "an end-to-end performance result.",
        "",
        "| Kernel | OFF count / avg us | Scheduler count / avg us | Fused count / avg us |",
        "|---|---:|---:|---:|",
    ]
    for name in KERNEL_PATTERNS:
        values = [kernels[mode][name] for mode in MODES]
        report.append(
            f"| {name} | {values[0]['instances']} / {values[0]['avg_us']:.3f} | "
            f"{values[1]['instances']} / {values[1]['avg_us']:.3f} | "
            f"{values[2]['instances']} / {values[2]['avg_us']:.3f} |"
        )
    report.extend(
        [
            "",
            "| Mode | GPU kernel instances | cudaLaunchKernel calls | cudaGraphLaunch calls | cudaMemcpyAsync calls |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        report.append(
            f"| {mode} | {total_kernel_instances[mode]} | "
            f"{apis[mode]['cudaLaunchKernel']['instances']} | "
            f"{apis[mode]['cudaGraphLaunch']['instances']} | "
            f"{apis[mode]['cudaMemcpyAsync']['instances']} |"
        )
    report.extend(
        [
            "",
            f"- Begin-status kernels equal CUDA graph launches: {begin_count}/{begin_count}.",
            f"- Final-status kernels equal status all-reduces: {status_count}/{status_count}.",
            f"- Protected top-k runs {topk_per_step:.0f} producers per completed status step.",
            f"- Protected top-k adds {protected_topk_delta_us:.3f} us per status step versus scheduler top-k.",
            f"- Local begin/final status costs {local_status_us:.3f} us per status step.",
            f"- The TP status all-reduce costs {status_collective_us:.3f} us per status step.",
            f"- Total directly attributed fused increment is {protected_topk_delta_us + local_status_us + status_collective_us:.3f} us per status step.",
            f"- Scheduler executes {scheduler_extra_kernels_per_step:.3f} more GPU kernels per status step than fused.",
            "- Checksum scan and request-finalizer counts are identical in all modes (8 each).",
        ]
    )
    text = "\n".join(report) + "\n"
    (root / "SUMMARY.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
