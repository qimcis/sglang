# Nsight Compute Report

## Profile Target

- Kernel / dispatch path: Cosmos3 QK RMSNorm + Qwen3 half-split RoPE, GQA layout `q_heads=32`, `kv_heads=8`, `head_dim=128`.
- Main baseline: split path using `apply_qk_norm` plus reference Qwen3 RoPE math.
- Patched path: `apply_qk_norm_rope(..., is_neox=True)` dispatching `fused_qknorm_rope_warp<128, 128, true, ..., bf16, int64>`.
- Inputs / shapes: `batch=1`, `seq_len=28520`, BF16, representative single-GPU Cosmos3 GEN token shape for 720p/121-frame style workload.
- GPU: CoreWeave staging `cw-us-east-04a`, NVIDIA H200, compute capability 9.0, driver `595.45.04`.
- Nsight Compute: `2025.3.1.0`.
- Main revision in pod: `83bc7766122b439d6e1564ec999e61eac6ab46a7`.
- Profiling pod: `cosmos3-qknorm-rope-ncu` in namespace `staging`.

## Summary

- Primary result: patched fused QK-norm+RoPE segment is `480.88 us` vs main split segment `1923.93 us`, a `4.00x` speedup for the representative GQA shape.
- End-to-end Cosmos3 generation result: patched warmup-excluded request is `858.76 ms` vs main `900.60 ms` on `480x832`, `17` frames, `5` denoise steps, a `4.65%` request speedup. Denoising stage improves `526.79 ms -> 483.27 ms` (`8.26%`).
- PR-quality 5-second generation result: patched warmup-excluded request is `99.29 s` vs main `102.53 s` on `1280x720`, `121` frames, `35` denoise steps, a `3.27%` request speedup. Denoising stage improves `96.56 s -> 93.33 s` (`3.46%`).
- Primary bottleneck in the fused kernel: L1TEX/global-load dependency and access efficiency, not launch occupancy. NCU reports long-scoreboard stalls at `9.78 cycles/issue-active` and a memory-access-pattern optimization estimate of `44.02%`.
- Best next validation: run full Cosmos3 denoising-stage timing to confirm the microbenchmark speedup translates through all 36 GEN layers and scheduler overhead.

## Key Metrics

| Metric | Main split segment | Patched fused segment | Source |
|---|---:|---:|---|
| End-to-end segment avg | `1923.93 us` | `480.88 us` | `artifacts/timing_*.jsonl` |
| Segment speedup | baseline | `4.00x` | CUDA events |
| E2E warmed request | `900.60 ms` | `858.76 ms` | `artifacts/e2e-*/perf.json` |
| E2E request speedup | baseline | `1.049x` | perf dump `total_duration_ms` |
| Denoising stage | `526.79 ms` | `483.27 ms` | perf dump `Cosmos3DenoisingStage` |
| Denoising speedup | baseline | `1.090x` | perf dump stage timings |
| 5s warmed request | `102.53 s` | `99.29 s` | `artifacts/quality-5s-*/perf.json` |
| 5s request speedup | baseline | `1.033x` | perf dump `total_duration_ms` |
| 5s denoising stage | `96.56 s` | `93.33 s` | perf dump `Cosmos3DenoisingStage` |
| 5s denoising speedup | baseline | `1.035x` | perf dump stage timings |
| Decode stage | `213.32 ms` | `215.36 ms` | perf dump `Cosmos3DecodingStage` |
| Peak reserved memory | `35976 MB` | `35972 MB` | perf dump `after_forward.peak_reserved_mb` |
| NCU profiled kernel | `fused_qknorm_warp` | `fused_qknorm_rope_warp` | `metrics_key_*.txt` |
| NCU kernel duration | `179.39 us` | `282.62 us` | `gpu__time_duration.sum` |
| SM throughput | `66.80%` | `68.40%` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput | `64.57%` | `73.15%` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| L1/TEX throughput | `41.46%` | `76.29%` | `l1tex__throughput.avg.pct_of_peak_sustained_active` |
| L1 sector hit rate | `65.02%` | `87.95%` | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | `50.08%` | `50.16%` | `lts__t_sector_hit_rate.pct` |
| DRAM read bytes | `292.08 MB` | `307.02 MB` | `dram__bytes_read.sum` |
| DRAM write bytes | `265.10 MB` | `265.76 MB` | `dram__bytes_write.sum` |
| Achieved occupancy | `87.12%` | `88.62%` | details pages |
| Long scoreboard | `11.95 cycles` | `9.78 cycles` | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` |

Important interpretation: the NCU kernel-duration row is not an apples-to-apples replacement-time comparison. Main's NCU report captures only the QK-norm kernel; main's RoPE work is additional PyTorch elementwise traffic measured in the CUDA-event segment timing.

## Source Hotspots

| Target | Source line / PC | Stall or metric | Evidence | Interpretation |
|---|---:|---|---|---|
| patched fused | `?:0` | long scoreboard | `13551` samples; source line unavailable in JIT report | Dominant stall class is global/L1TEX dependency. |
| patched fused | NCU rule | uncoalesced/excess sectors | `36505600` excessive sectors, `36%` of `101531200` sectors | Memory access pattern inside fused kernel is the next kernel-level target. |
| patched fused | NCU rule | global load sector utilization | `12.7 / 32` bytes used per sector | Loads are not ideally coalesced. |

The source-level report did not resolve JIT source lines (`?:0`). The `.ncu-rep` files are present, but source correlation was unavailable for this JIT-built kernel in the container.

## Diagnosis

- The patch is a clear performance win for the relevant Cosmos3 segment: replacing split QK norm plus Qwen3 RoPE with one fused kernel reduced measured segment time by about `1.44 ms` per call at `seq_len=28520`.
- The end-to-end run confirms the microbenchmark translates into full generation: denoising drops by `43.52 ms` over a 5-step request, while decode remains unchanged. The total warmed request improves by `41.84 ms`.
- The longer PR-quality run confirms the gain on a 5-second 720p video: denoising drops by `3.23 s` over 35 steps, and total warmup-excluded request time drops by `3.24 s`.
- The fused kernel itself is heavier than main's QK-norm-only kernel (`282.62 us` vs `179.39 us`) because it performs both norm and RoPE, but the full segment is much faster because it removes main's separate RoPE read/modify/write passes.
- The fused kernel is already well occupied (`88.62%` achieved occupancy, one wave/SM), so underfill is not the main issue.
- NCU's rule engine flags memory access efficiency as the strongest remaining kernel-level issue: `44.02%` estimated speedup from global-load access pattern, plus `49.17%` local speedup attributed to long-scoreboard stalls.
- Tensor cores are irrelevant here (`0%` tensor op HMMA), as expected for elementwise/norm/rope work.

## Ranked Recommendations

1. Keep the fused Cosmos3 integration and validate end-to-end denoising-stage latency.
   Evidence: microbenchmark shows `4.00x` improvement for the QK-norm+RoPE segment; e2e 5-step T2V run improves denoising by `8.26%` and warmed request by `4.65%`.
   Validation command: repeat on the 35-step 720p target used for release benchmarking.

2. Investigate fused kernel memory coalescing if further kernel work is needed.
   Evidence: patched fused NCU details report `12.7/32` bytes per sector and `36%` excessive sectors with `44.02%` estimated speedup in Memory Workload Analysis Tables.
   Risk: medium; this means touching `python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh` and validating both NeoX and non-NeoX variants.

3. Improve JIT source correlation for future NCU runs.
   Evidence: source-level profile resolves stalls only to `?:0` despite collecting `--set source --section SourceCounters`.
   Expected value: necessary before making source-line-specific kernel changes.

## Artifacts

- Full fused report: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/reports/full_patched_fused.ncu-rep`
- Source fused report: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/reports/source_patched_fused.ncu-rep`
- Full main QK-norm report: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/reports/full_main_qknorm.ncu-rep`
- Source main QK-norm report: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/reports/source_main_qknorm.ncu-rep`
- Parsed metrics: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/analysis/metrics_key_*.txt`
- Side-by-side NCU compare: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/analysis/compare_main_qknorm_vs_patched_fused.txt`
- NCU details pages: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/analysis/details_*.txt`
- Timing results: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/artifacts/timing_*.jsonl`
- End-to-end perf dumps and logs: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/artifacts/e2e-*`
- 5-second quality outputs, perf dumps, and logs: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/artifacts/quality-5s-*`
- Harness: `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/harness/cosmos3_qknorm_rope_profile.py`

## Reproduction

```bash
# In the profiling pod, after cloning main and applying the runtime patch to sglang-patched:
cd /workspace/sglang-main
PYTHONPATH=/workspace/sglang-main/python \
  python3 cosmos3_qknorm_rope_profile.py --mode split --seq-len 28520 --warmup 10 --iters 50

cd /workspace/sglang-patched
PYTHONPATH=/workspace/sglang-patched/python \
  python3 cosmos3_qknorm_rope_profile.py --mode fused --seq-len 28520 --warmup 10 --iters 50

# E2E comparison command shape, run once from sglang-main and once from sglang-patched:
PYTHONPATH=/workspace/sglang-patched/python \
HF_HOME=/workspace/hf-cache \
SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
CUDA_VISIBLE_DEVICES=0 \
sglang generate --model-path nvidia/Cosmos3-Nano \
  --prompt "A curious raccoon walking through a quiet forest" \
  --height 480 --width 832 --num-frames 17 --num-inference-steps 5 \
  --seed 42 --num-gpus 1 --performance-mode manual \
  --dit-cpu-offload false --vae-cpu-offload false \
  --no-save-output --warmup --perf-dump-path /workspace/e2e/patched/perf.json

HOME=/tmp ncu -f --set full --section PmSampling --section PmSampling_WarpStates \
  --target-processes all -k "regex:qknorm_rope" -c 1 \
  -o /workspace/ncu-reports/full_patched_fused \
  python3 cosmos3_qknorm_rope_profile.py --mode fused --seq-len 28520 --single

HOME=/tmp ncu -f --set source --section SourceCounters \
  --target-processes all -k "regex:qknorm_rope" -c 1 \
  -o /workspace/ncu-reports/source_patched_fused \
  python3 cosmos3_qknorm_rope_profile.py --mode fused --seq-len 28520 --single
```

## Caveats

- This is a focused microbenchmark of the QK-norm+RoPE segment, not full Cosmos3 e2e generation.
- The report now includes one full Cosmos3 e2e generation comparison, but at a short `480x832`, `17` frame, `5` step workload rather than the full 35-step 720p target.
- The report also includes a 5-second `1280x720`, `121` frame, `35` step quality-output comparison suitable for PR results. Visual quality still needs human inspection of the saved videos.
- The baseline NCU report captures main's QK-norm kernel only; main's RoPE component is represented in the CUDA-event timing, not as one single NCU kernel.
- The CoreWeave staging GPU available for this run was H200 (`sm_90`), not B200.
