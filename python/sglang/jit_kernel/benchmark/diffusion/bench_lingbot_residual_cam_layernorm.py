"""Benchmark LingBot residual + camera affine + layernorm fusion.

This benchmark compares the old production chain against the new native helper
and CUDA fused path for LingBot causal block shapes.
"""

import argparse
import csv
import statistics
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    ScaleResidualLayerNormScaleShift,
)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def _make_inputs(
    *,
    batch: int,
    seq_len: int,
    frames: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
):
    if seq_len % frames != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by frames={frames}")
    torch.manual_seed(0)
    return {
        "hidden": torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype),
        "attn": torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype),
        "gate": torch.randn(batch, frames, 1, hidden_dim, device=device, dtype=dtype),
        "cam_scale": torch.randn(
            batch, seq_len, hidden_dim, device=device, dtype=dtype
        )
        * 0.01,
        "cam_shift": torch.randn(
            batch, seq_len, hidden_dim, device=device, dtype=dtype
        )
        * 0.01,
    }


def _make_layer(hidden_dim: int, dtype: torch.dtype, device: str):
    layer = ScaleResidualLayerNormScaleShift(
        hidden_dim,
        eps=1e-6,
        elementwise_affine=True,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    layer.requires_grad_(False)
    return layer


def _old_split(layer, hidden, attn, gate, cam_scale, cam_shift):
    zero = torch.zeros((1,), device=hidden.device, dtype=hidden.dtype)
    _, residual = layer.forward_cuda(hidden, attn, gate, zero, zero)
    cam_hidden = residual * (1 + cam_scale) + cam_shift
    norm_hidden = layer.norm(cam_hidden).to(hidden.dtype)
    return norm_hidden, cam_hidden


def _native_new(layer, hidden, attn, gate, cam_scale, cam_shift):
    return layer.forward_lingbot_residual_cam_norm_native(
        hidden,
        attn,
        gate,
        cam_scale,
        cam_shift,
    )


def _fused_cuda(layer, hidden, attn, gate, cam_scale, cam_shift):
    return layer.forward_lingbot_residual_cam_norm(
        hidden,
        attn,
        gate,
        cam_scale,
        cam_shift,
    )


def _validate(provider, ref, out, dtype: torch.dtype):
    atol = rtol = 1e-5 if dtype == torch.float32 else 5e-2
    ref_norm, ref_hidden = ref
    out_norm, out_hidden = out
    norm_ok = torch.allclose(out_norm, ref_norm, atol=atol, rtol=rtol)
    hidden_ok = torch.allclose(out_hidden, ref_hidden, atol=atol, rtol=rtol)
    norm_diff = (out_norm - ref_norm).abs().max().item()
    hidden_diff = (out_hidden - ref_hidden).abs().max().item()
    if not (norm_ok and hidden_ok):
        raise AssertionError(
            f"{provider} failed parity: norm_ok={norm_ok} hidden_ok={hidden_ok} "
            f"norm_diff={norm_diff} hidden_diff={hidden_diff}"
        )
    return norm_diff, hidden_diff


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    elapsed_us = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed_us.append(start.elapsed_time(end) * 1000.0)
    return (
        statistics.median(elapsed_us),
        _percentile(elapsed_us, 0.2),
        _percentile(elapsed_us, 0.8),
    )


def _run_case(args, seq_len: int, provider: str):
    dtype = DTYPES[args.dtype]
    inputs = _make_inputs(
        batch=args.batch,
        seq_len=seq_len,
        frames=args.frames,
        hidden_dim=args.hidden_dim,
        dtype=dtype,
        device=args.device,
    )
    layer = _make_layer(args.hidden_dim, dtype, args.device)

    fns = {
        "old_split": _old_split,
        "native_new": _native_new,
        "fused_cuda": _fused_cuda,
    }
    ref = _native_new(layer, **inputs)
    out = fns[provider](layer, **inputs)
    torch.cuda.synchronize()
    norm_diff, hidden_diff = _validate(provider, ref, out, dtype)

    fn = lambda: fns[provider](layer, **inputs)
    # First call compiles the CuTe DSL kernel; exclude it from timing.
    fn()
    torch.cuda.synchronize()
    median_us, p20_us, p80_us = _bench(fn, warmup=args.warmup, iters=args.iters)
    return {
        "gpu": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "B": args.batch,
        "S": seq_len,
        "F": args.frames,
        "D": args.hidden_dim,
        "provider": provider,
        "median_us": median_us,
        "p20_us": p20_us,
        "p80_us": p80_us,
        "norm_max_diff": norm_diff,
        "hidden_max_diff": hidden_diff,
        "correct": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark.csv")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[4680, 2340, 1170, 585],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    rows = []
    for seq_len in args.seq_lens:
        baseline_median = None
        for provider in ("old_split", "native_new", "fused_cuda"):
            row = _run_case(args, seq_len, provider)
            if provider == "old_split":
                baseline_median = row["median_us"]
            row["speedup_vs_old"] = (
                baseline_median / row["median_us"] if baseline_median else 1.0
            )
            rows.append(row)
            print(
                f"S={seq_len} {provider}: {row['median_us']:.2f} us "
                f"speedup={row['speedup_vs_old']:.3f}x"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
