import argparse
import json
import os
import time

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm,
    apply_qk_norm_rope,
)


def reference_qwen3_rope(q, k, cos, sin):
    half = q.shape[-1] // 2
    q1 = q[..., :half]
    q2 = q[..., half:]
    q_out = torch.empty_like(q)
    q_out[..., :half] = q1 * cos[..., :half] - q2 * sin[..., :half]
    q_out[..., half:] = q2 * cos[..., half:] + q1 * sin[..., half:]

    half = k.shape[-1] // 2
    k1 = k[..., :half]
    k2 = k[..., half:]
    k_out = torch.empty_like(k)
    k_out[..., :half] = k1 * cos[..., :half] - k2 * sin[..., :half]
    k_out[..., half:] = k2 * cos[..., half:] + k1 * sin[..., half:]
    return q_out, k_out


def make_inputs(args):
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = torch.randn(
        args.batch,
        args.seq_len,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        args.batch,
        args.seq_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    q_norm = RMSNorm(args.head_dim, eps=1e-6).to(device=device, dtype=dtype)
    k_norm = RMSNorm(args.head_dim, eps=1e-6).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(generator=generator)
    k_norm.weight.data.normal_(generator=generator)

    half = args.head_dim // 2
    freqs = torch.randn(
        args.batch,
        args.seq_len,
        half,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos_sin_cache = torch.cat((cos_half, sin_half), dim=-1).reshape(-1, args.head_dim)
    cos_full = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(2).to(dtype)
    sin_full = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(2).to(dtype)
    positions = torch.arange(
        args.batch * args.seq_len, device=device, dtype=torch.long
    )
    return q, k, q_norm, k_norm, cos_sin_cache, positions, cos_full, sin_full


def run_once(args, state):
    q, k, q_norm, k_norm, cos_sin_cache, positions, cos_full, sin_full = state
    if args.mode == "split":
        q, k = apply_qk_norm(q.contiguous(), k.contiguous(), q_norm, k_norm, args.head_dim)
        q, k = reference_qwen3_rope(q, k, cos_full, sin_full)
        state[0] = q
        state[1] = k
        return q, k
    if args.mode == "fused":
        q, k = apply_qk_norm_rope(
            q=q.contiguous(),
            k=k.contiguous(),
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=args.head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
            positions=positions,
        )
        state[0] = q
        state[1] = k
        return q, k
    raise ValueError(f"unknown mode: {args.mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split", "fused"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=28520)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("SGLANG_ENABLE_FUSED_QKNORM_ROPE", "1")
    torch.cuda.set_device(0)
    state = list(make_inputs(args))
    torch.cuda.synchronize()

    if args.single:
        run_once(args, state)
        torch.cuda.synchronize()
        print(json.dumps({"mode": args.mode, "single": True}))
        return

    for _ in range(args.warmup):
        run_once(args, state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_once(args, state)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    result = {
        "mode": args.mode,
        "dtype": args.dtype,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "iters": args.iters,
        "total_ms": total_ms,
        "avg_us": total_ms * 1000.0 / args.iters,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
