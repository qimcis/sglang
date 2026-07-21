#!/usr/bin/env python3
"""Replay RaMP MoE routing histograms against SGLang's Triton fused-MoE path.

Capture histograms from a serving run with:

  SGLANG_MOE_RAMP_ENABLE=1 \
  SGLANG_MOE_RAMP_HISTOGRAM_PATH=/tmp/ramp-{pid}.jsonl \
  SGLANG_MOE_RAMP_HISTOGRAM_INTERVAL=1 \
  sglang serve ...

Then replay a sample of those routing distributions:

  PYTHONPATH=python python3 benchmark/kernels/fused_moe_triton/replay_ramp_histograms.py \
    --histograms /tmp/ramp-123.jsonl \
    --hidden-size 7168 \
    --intermediate-size 2048 \
    --dtype bfloat16

The replay reconstructs synthetic ``topk_ids`` with the same
``tokens_per_expert`` distribution. It is intended to answer whether current
kernel/backend choices are sensitive to real routing histograms before adding
behavior-changing RaMP dispatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.benchmark.bench_utils import run_bench
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.server_args import set_global_server_args_for_scheduler


@dataclass
class HistogramRecord:
    layer_id: Optional[int]
    bucket: str
    top_k: int
    tokens_per_expert: list[int]

    @property
    def num_experts(self) -> int:
        return len(self.tokens_per_expert)

    @property
    def total_assignments(self) -> int:
        return sum(self.tokens_per_expert)

    @property
    def num_tokens(self) -> int:
        return self.total_assignments // self.top_k


class DefaultFalseArgs(SimpleNamespace):
    def __getattr__(self, name):
        return False


@triton.jit
def _tiny_topk6_align_kernel(
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    BLOCK_SIZE_M: tl.constexpr,
    TOP_K: tl.constexpr,
):
    expert_slot = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE_M)
    block_start = expert_slot * BLOCK_SIZE_M
    padding_token_id = TOP_K
    token_offsets = tl.where(offsets == 0, expert_slot, padding_token_id)
    tl.store(sorted_token_ids + block_start + offsets, token_offsets)
    expert_id = tl.load(topk_ids + expert_slot)
    tl.store(expert_ids + expert_slot, expert_id)
    if expert_slot == 0:
        tl.store(num_tokens_post_padded, TOP_K * BLOCK_SIZE_M)


def tiny_topk6_align(
    topk_ids: torch.Tensor,
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert topk_ids.shape == (1, 6)
    sorted_token_ids = torch.empty(
        (topk_ids.numel() * block_size_m,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (topk_ids.numel(),), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_padded = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    _tiny_topk6_align_kernel[(topk_ids.numel(),)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        BLOCK_SIZE_M=block_size_m,
        TOP_K=topk_ids.numel(),
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


_CACHED_TINY_TOPK6_ALIGN: dict[
    tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def cached_tiny_topk6_align(
    topk_ids: torch.Tensor,
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert topk_ids.shape == (1, 6)
    key = (str(topk_ids.device), block_size_m, topk_ids.numel())
    cached = _CACHED_TINY_TOPK6_ALIGN.get(key)
    if cached is None:
        padding_token_id = topk_ids.numel()
        sorted_ids_cpu = torch.full(
            (topk_ids.numel() * block_size_m,), padding_token_id, dtype=torch.int32
        )
        sorted_ids_cpu[torch.arange(topk_ids.numel()) * block_size_m] = torch.arange(
            topk_ids.numel(), dtype=torch.int32
        )
        num_tokens_post_padded_cpu = torch.tensor(
            [topk_ids.numel() * block_size_m], dtype=torch.int32
        )
        cached = (
            sorted_ids_cpu.to(device=topk_ids.device),
            num_tokens_post_padded_cpu.to(device=topk_ids.device),
        )
        _CACHED_TINY_TOPK6_ALIGN[key] = cached
    sorted_token_ids, num_tokens_post_padded = cached
    return sorted_token_ids, topk_ids.view(-1), num_tokens_post_padded


def iter_histogram_records(paths: Iterable[Path]) -> Iterator[HistogramRecord]:
    for path in paths:
        if path.is_dir():
            yield from iter_histogram_records(sorted(path.glob("*.jsonl")))
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                top_k = int(row.get("top_k") or 0)
                tokens_per_expert = row.get("tokens_per_expert")
                if not top_k or not isinstance(tokens_per_expert, list):
                    continue
                yield HistogramRecord(
                    layer_id=row.get("layer_id"),
                    bucket=row.get("bucket", "unknown"),
                    top_k=top_k,
                    tokens_per_expert=[int(x) for x in tokens_per_expert],
                )


def reconstruct_topk_ids(record: HistogramRecord, device: torch.device) -> torch.Tensor:
    ids: list[int] = []
    for expert_id, count in enumerate(record.tokens_per_expert):
        ids.extend([expert_id] * count)

    usable = (len(ids) // record.top_k) * record.top_k
    if usable == 0:
        raise ValueError("histogram has no complete top-k rows")
    ids = ids[:usable]
    return torch.tensor(ids, dtype=torch.int32, device=device).view(-1, record.top_k)


def dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def init_single_process_distributed(port: int):
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def benchmark_record(
    record: HistogramRecord,
    *,
    provider: str,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    block_shape: Optional[list[int]],
    warmup_ms: int,
    rep_ms: int,
    profile_iterations: int = 0,
    triton_config: Optional[dict] = None,
    precompute_triton_routing: bool = False,
    tiny_align_topk6: bool = False,
    cached_tiny_align_topk6: bool = False,
    check_triton_reference: bool = False,
) -> tuple[float, float, float]:
    device = torch.device("cuda")
    topk_ids = reconstruct_topk_ids(record, device)
    num_tokens = topk_ids.shape[0]
    topk_weights = torch.full(
        topk_ids.shape,
        fill_value=1.0 / record.top_k,
        dtype=torch.float32,
        device=device,
    )
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty(0, device=device),
    )

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w1 = torch.randn(
        record.num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    w2 = torch.randn(
        record.num_experts,
        hidden_size,
        intermediate_size,
        dtype=dtype,
        device=device,
    )
    w1_scale = w2_scale = a1_scale = a2_scale = None
    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        if block_shape is None:
            w1_scale = torch.rand(
                record.num_experts, dtype=torch.float32, device=device
            )
            w2_scale = torch.rand(
                record.num_experts, dtype=torch.float32, device=device
            )
            a1_scale = torch.rand(1, dtype=torch.float32, device=device)
            a2_scale = torch.rand(1, dtype=torch.float32, device=device)
        else:
            block_n, block_k = block_shape
            w1_n_tiles = (2 * intermediate_size + block_n - 1) // block_n
            w1_k_tiles = (hidden_size + block_k - 1) // block_k
            w2_n_tiles = (hidden_size + block_n - 1) // block_n
            w2_k_tiles = (intermediate_size + block_k - 1) // block_k
            w1_scale = torch.rand(
                (record.num_experts, w1_n_tiles, w1_k_tiles),
                dtype=torch.float32,
                device=device,
            )
            w2_scale = torch.rand(
                (record.num_experts, w2_n_tiles, w2_k_tiles),
                dtype=torch.float32,
                device=device,
            )
    config = MoeRunnerConfig(
        num_experts=record.num_experts,
        num_local_experts=record.num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        layer_id=record.layer_id,
        top_k=record.top_k,
        inplace=False,
    )

    deep_gemm_runner = None
    trtllm_quant_info = None
    if provider == "deep_gemm":
        deep_gemm_runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config)
        deep_gemm_quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w1,
            w2_weight=w2,
            use_fp8=use_fp8_w8a8,
            w13_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
    elif provider == "flashinfer_trtllm_routed":
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmFp8MoeQuantInfo,
            get_activation_type,
        )

        if not use_fp8_w8a8:
            raise NotImplementedError(
                "flashinfer_trtllm_routed replay currently requires --use-fp8-w8a8"
            )
        trtllm_quant_info = FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=w1,
            w2_weight=w2,
            global_num_experts=record.num_experts,
            local_expert_offset=0,
            local_num_experts=record.num_experts,
            intermediate_size=intermediate_size,
            routing_method_type=RoutingMethodType.TopK,
            block_quant=True,
            use_mxfp8=False,
            weight_block_k=block_shape[1] if block_shape else 128,
            w13_weight_scale_inv=w1_scale,
            w2_weight_scale_inv=w2_scale,
            activation_type=get_activation_type("silu"),
        )

    cutlass_fused_moe = None
    cutlass_fp8_scales = None
    if provider in ("flashinfer_cutlass", "flashinfer_cutlass_fp8"):
        if use_fp8_w8a8:
            if provider != "flashinfer_cutlass_fp8":
                raise NotImplementedError(
                    "Use --provider flashinfer_cutlass_fp8 for FlashInfer FP8 replay."
                )
            input_scale = torch.rand((), dtype=torch.float32, device=device) + 1.0
            activation_scale = torch.rand((), dtype=torch.float32, device=device) + 1.0
            w1_scale = (
                torch.rand(record.num_experts, dtype=torch.float32, device=device) + 1.0
            )
            w2_scale = (
                torch.rand(record.num_experts, dtype=torch.float32, device=device) + 1.0
            )
            cutlass_fp8_scales = [
                w1_scale * input_scale,
                activation_scale.reciprocal(),
                activation_scale * w2_scale,
                input_scale,
            ]
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

        cutlass_fused_moe = (cutlass_fused_moe, ActivationType.Swiglu, scaled_fp8_quant)

    triton_precomputed = None
    triton_tiny_align = None
    fused_moe_kernel_sequence = None
    if provider == "triton" and (
        precompute_triton_routing or tiny_align_topk6 or cached_tiny_align_topk6
    ):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
            _prepare_fused_moe_run,
        )

        fused_moe_kernel_sequence = _fused_moe_kernel_sequence
        context = override_config(triton_config) if triton_config else nullcontext()
        with context:
            prepared = _prepare_fused_moe_run(
                hidden_states,
                w1,
                w2,
                topk_ids,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=block_shape,
            )
            if precompute_triton_routing:
                triton_precomputed = prepared
            if tiny_align_topk6:
                kernel_config, down_config, down_moe_use_tma, *_ = prepared
                if down_moe_use_tma:
                    raise NotImplementedError(
                        "tiny-align replay does not support TMA down MoE"
                    )
                triton_tiny_align = (kernel_config, down_config, down_moe_use_tma)
            if cached_tiny_align_topk6:
                kernel_config, down_config, down_moe_use_tma, *_ = prepared
                if down_moe_use_tma:
                    raise NotImplementedError(
                        "cached tiny-align replay does not support TMA down MoE"
                    )
                triton_tiny_align = (kernel_config, down_config, down_moe_use_tma)

    def run_once():
        if provider == "triton":
            if triton_precomputed is not None:
                (
                    kernel_config,
                    down_config,
                    down_moe_use_tma,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                ) = triton_precomputed
                return fused_moe_kernel_sequence(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    kernel_config,
                    down_config,
                    down_moe_use_tma,
                    b1=None,
                    b2=None,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=None,
                    w2_zp=None,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                    activation=config.activation,
                    is_gated=config.is_gated,
                    no_combine=config.no_combine,
                    inplace=config.inplace,
                    apply_router_weight_on_input=config.apply_router_weight_on_input,
                    routed_scaling_factor=config.routed_scaling_factor,
                    gemm1_alpha=config.gemm1_alpha,
                    gemm1_limit=config.gemm1_clamp_limit,
                    filter_expert=False,
                    swiglu_limit=config.swiglu_limit,
                )
            if triton_tiny_align is not None:
                kernel_config, down_config, down_moe_use_tma = triton_tiny_align
                if cached_tiny_align_topk6:
                    sorted_token_ids, expert_ids, num_tokens_post_padded = (
                        cached_tiny_topk6_align(topk_ids, kernel_config["BLOCK_SIZE_M"])
                    )
                else:
                    sorted_token_ids, expert_ids, num_tokens_post_padded = (
                        tiny_topk6_align(topk_ids, kernel_config["BLOCK_SIZE_M"])
                    )
                return fused_moe_kernel_sequence(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    kernel_config,
                    down_config,
                    down_moe_use_tma,
                    b1=None,
                    b2=None,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=None,
                    w2_zp=None,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                    activation=config.activation,
                    is_gated=config.is_gated,
                    no_combine=config.no_combine,
                    inplace=config.inplace,
                    apply_router_weight_on_input=config.apply_router_weight_on_input,
                    routed_scaling_factor=config.routed_scaling_factor,
                    gemm1_alpha=config.gemm1_alpha,
                    gemm1_limit=config.gemm1_clamp_limit,
                    filter_expert=False,
                    swiglu_limit=config.swiglu_limit,
                )
            context = override_config(triton_config) if triton_config else nullcontext()
            with context:
                return fused_moe(
                    hidden_states,
                    w1,
                    w2,
                    topk_output,
                    moe_runner_config=config,
                    use_fp8_w8a8=use_fp8_w8a8,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                )
        if provider == "deep_gemm":
            dispatch_output = StandardDispatchOutput(
                hidden_states=hidden_states.clone(),
                hidden_states_scale=None,
                topk_output=topk_output,
            )
            return deep_gemm_runner.run(
                dispatch_output, deep_gemm_quant_info
            ).hidden_states
        if provider == "flashinfer_cutlass":
            kernel, activation_type, _ = cutlass_fused_moe
            return kernel(
                input=hidden_states,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=w1,
                fc2_expert_weights=w2,
                output_dtype=hidden_states.dtype,
                quant_scales=None,
                ep_size=1,
                ep_rank=0,
                tp_size=1,
                tp_rank=0,
                tune_max_num_tokens=1
                << (max(1, hidden_states.shape[0] - 1).bit_length()),
                activation_type=activation_type,
            )[0]
        if provider == "flashinfer_cutlass_fp8":
            kernel, activation_type, quantize = cutlass_fused_moe
            x_fp8, _ = quantize(hidden_states, cutlass_fp8_scales[3])
            return kernel(
                input=x_fp8,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=w1,
                fc2_expert_weights=w2,
                output_dtype=hidden_states.dtype,
                quant_scales=cutlass_fp8_scales,
                ep_size=1,
                ep_rank=0,
                tp_size=1,
                tp_rank=0,
                tune_max_num_tokens=1
                << (max(1, hidden_states.shape[0] - 1).bit_length()),
                activation_type=activation_type,
            )[0]
        if provider == "flashinfer_trtllm_routed":
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                fused_experts_none_to_flashinfer_trtllm_fp8,
            )

            dispatch_output = StandardDispatchOutput(
                hidden_states=hidden_states,
                hidden_states_scale=None,
                topk_output=topk_output,
            )
            return fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                trtllm_quant_info,
                config,
                use_routed_topk=True,
            ).hidden_states
        raise ValueError(f"Unsupported provider: {provider}")

    if (
        check_triton_reference
        and provider == "triton"
        and (tiny_align_topk6 or cached_tiny_align_topk6)
    ):
        torch.cuda.synchronize()
        context = override_config(triton_config) if triton_config else nullcontext()
        with context:
            reference = fused_moe(
                hidden_states,
                w1,
                w2,
                topk_output,
                moe_runner_config=config,
                use_fp8_w8a8=use_fp8_w8a8,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )
        candidate = run_once()
        torch.cuda.synchronize()
        diff = (candidate.float() - reference.float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        ref_abs = reference.float().abs().clamp_min(1e-6)
        max_rel = float((diff / ref_abs).max().item()) if diff.numel() else 0.0
        # FP8 block-scale inputs with BF16 outputs can differ at BF16 ULP scale
        # when equivalent paths change atomic accumulation order. At DSV4 output
        # magnitudes this is commonly O(1e2), so use a serving-oriented tolerance.
        if not torch.allclose(candidate, reference, atol=512.0, rtol=5e-2):
            raise AssertionError(
                f"tiny-align output mismatch vs Triton reference: max_abs={max_abs} max_rel={max_rel}"
            )
        print(
            json.dumps(
                {
                    "triton_reference_check": "ok",
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                },
                sort_keys=True,
            )
        )

    for _ in range(5):
        run_once()
    torch.cuda.synchronize()

    if profile_iterations > 0:
        start = time.perf_counter()
        for _ in range(profile_iterations):
            run_once()
        torch.cuda.synchronize()
        per_iter_ms = (time.perf_counter() - start) * 1000.0 / profile_iterations
        return per_iter_ms, per_iter_ms, per_iter_ms

    return run_bench(run_once, use_cuda_graph=False, warmup_ms=warmup_ms, rep_ms=rep_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--histograms", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=(
            "triton",
            "deep_gemm",
            "flashinfer_cutlass",
            "flashinfer_cutlass_fp8",
            "flashinfer_trtllm_routed",
        ),
        default="triton",
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument("--block-shape", nargs=2, type=int, default=None)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--layer-id", type=int, default=None)
    parser.add_argument("--min-num-tokens", type=int, default=1)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--exact-num-tokens", type=int, default=None)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="Run a fixed number of measured iterations instead of run_bench; useful for NCU.",
    )
    parser.add_argument(
        "--triton-config-json",
        default=None,
        help="Override Triton fused-MoE config with a JSON object for replay experiments.",
    )
    parser.add_argument(
        "--precompute-triton-routing",
        action="store_true",
        help="Replay-only upper bound: precompute Triton routing/alignment outside the timed loop.",
    )
    parser.add_argument(
        "--tiny-align-topk6",
        action="store_true",
        help="Replay-only candidate: use a DSV4 num_tokens=1/top_k=6 tiny GPU alignment kernel.",
    )
    parser.add_argument(
        "--cached-tiny-align-topk6",
        action="store_true",
        help="Replay-only candidate: use cached sorted ids and topk_ids.view(-1) as expert ids for num_tokens=1/top_k=6.",
    )
    parser.add_argument(
        "--check-triton-reference",
        action="store_true",
        help="For replay candidates, compare output against standard Triton on the same synthetic inputs before benchmarking.",
    )
    parser.add_argument("--dist-init-port", type=int, default=29577)
    parser.add_argument(
        "--enable-fused-moe-sum-all-reduce",
        action="store_true",
        help="Enable SGLang's fused MoE sum path in global server args for replay experiments.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This replay benchmark requires CUDA")

    set_global_server_args_for_scheduler(
        DefaultFalseArgs(
            enable_fused_moe_sum_all_reduce=args.enable_fused_moe_sum_all_reduce
        )
    )
    distributed_initialized = False
    if args.provider in ("flashinfer_cutlass", "flashinfer_trtllm_routed"):
        init_single_process_distributed(args.dist_init_port)
        distributed_initialized = True

    dtype = dtype_from_name(args.dtype)
    triton_config = (
        json.loads(args.triton_config_json) if args.triton_config_json else None
    )
    records = []
    for record in iter_histogram_records(args.histograms):
        if args.bucket is not None and record.bucket != args.bucket:
            continue
        if args.layer_id is not None and record.layer_id != args.layer_id:
            continue
        if record.num_tokens < args.min_num_tokens:
            continue
        if args.max_num_tokens is not None and record.num_tokens > args.max_num_tokens:
            continue
        if (
            args.exact_num_tokens is not None
            and record.num_tokens != args.exact_num_tokens
        ):
            continue
        records.append(record)
        if len(records) >= args.max_records:
            break

    if not records:
        raise RuntimeError("No matching histogram records found")

    fieldnames = [
        "idx",
        "provider",
        "layer_id",
        "bucket",
        "num_tokens",
        "top_k",
        "num_experts",
        "total_assignments",
        "median_ms",
        "p20_ms",
        "p80_ms",
    ]
    rows = []
    print(",".join(fieldnames))
    try:
        for idx, record in enumerate(records):
            median_ms, p20_ms, p80_ms = benchmark_record(
                record,
                provider=args.provider,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                dtype=dtype,
                use_fp8_w8a8=args.use_fp8_w8a8,
                block_shape=args.block_shape,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                profile_iterations=args.profile_iterations,
                triton_config=triton_config,
                precompute_triton_routing=args.precompute_triton_routing,
                tiny_align_topk6=args.tiny_align_topk6,
                cached_tiny_align_topk6=args.cached_tiny_align_topk6,
                check_triton_reference=args.check_triton_reference,
            )
            row = {
                "idx": idx,
                "provider": args.provider,
                "layer_id": record.layer_id,
                "bucket": record.bucket,
                "num_tokens": record.num_tokens,
                "top_k": record.top_k,
                "num_experts": record.num_experts,
                "total_assignments": record.total_assignments,
                "median_ms": f"{median_ms:.6f}",
                "p20_ms": f"{p20_ms:.6f}",
                "p80_ms": f"{p80_ms:.6f}",
            }
            rows.append(row)
            print(",".join(str(row[name]) for name in fieldnames))
    finally:
        if distributed_initialized:
            destroy_model_parallel()
            destroy_distributed_environment()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists() or not args.append
        mode = "a" if args.append else "w"
        with output_path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
