## Motivation

Cosmos3 applies Q/K RMSNorm and Qwen3-style mRoPE as separate operations in both the UND causal-attention path and GEN cross-attention path. In the GEN denoising loop this sequence runs for every layer and every denoising step, so the extra read/modify/write passes are visible in per-step latency.

This PR reuses the existing fused QK-norm + RoPE kernel for Cosmos3's Qwen3 half-split RoPE convention. The goal is to reduce bandwidth-bound Q/K normalization and RoPE overhead without changing the attention math or adding a new CUDA kernel.

## Modifications

- Extend `apply_qk_norm_rope` to accept GQA layouts where Q and K have different head counts but share batch, sequence length, and head dimension.
- Extend the RoPE fallback wrapper to handle GQA and partial RoPE layouts consistently.
- Add a Cosmos3 rotary-cache input path that builds the fused kernel's `[cos, sin]` cache directly from interleaved mRoPE frequencies.
- Route `Cosmos3CausalAttention` and `Cosmos3CrossAttention` through `apply_qk_norm_rope(..., is_neox=True)`.
- Preserve the existing Qwen3 half-split rotation convention by using the NeoX-style fused kernel path.

## Accuracy Tests

Generated main and patched Cosmos3 outputs with the same model, prompt, seed, resolution, frame count, and denoising steps:

- Model: `nvidia/Cosmos3-Nano`
- Prompt: `A curious raccoon walking through a quiet forest`
- Seed: `42`
- Resolution: `1280x720`
- Frames / FPS: `121` frames at `24 FPS` (`5.0s`)
- Steps: `35`
- GPU: CoreWeave H200, 1 GPU
- Guardrails: disabled with `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`

Both main and patched runs completed successfully and produced videos for manual quality inspection.

Local copies for review:

- `~/Downloads/cosmos3-qknorm-rope-results/main_5s.mp4`
- `~/Downloads/cosmos3-qknorm-rope-results/patched_5s.mp4`

I did not run an automated visual metric such as SSIM/PSNR because `ffmpeg` was not available on the local Mac environment. Human visual comparison is still recommended.

## Speed Tests and Profiling

Environment:

- CoreWeave staging, context `cw-us-east-04a`, namespace `staging`
- GPU: NVIDIA H200, compute capability 9.0
- Nsight Compute: `2025.3.1.0`
- Main revision: `83bc7766122b439d6e1564ec999e61eac6ab46a7`

### Kernel Segment Microbenchmark

Representative Cosmos3 GEN shape:

- `batch=1`
- `seq_len=28520`
- `q_heads=32`
- `kv_heads=8`
- `head_dim=128`
- `dtype=bf16`

| Path | Avg Time |
|---|---:|
| Main split QK-norm + Qwen3 RoPE | `1923.93 us` |
| Patched fused QK-norm + RoPE | `480.88 us` |

Speedup: `4.00x` for the QK-norm + RoPE segment.

NCU reports collected:

- Main QK-norm kernel: `fused_qknorm_warp`
- Patched fused kernel: `fused_qknorm_rope_warp`

Key fused-kernel NCU metrics:

- Kernel duration: `282.62 us`
- SM throughput: `68.40%`
- Memory throughput: `73.15%`
- L1 sector hit rate: `87.95%`
- Long scoreboard: `9.78 cycles/issue-active`

NCU rule engine flagged global-load access pattern as the next kernel-level bottleneck, but this PR focuses only on reusing the existing fused kernel.

### End-to-End Cosmos3 Generation: 5s 720p Run

Command shape:

- Model: `nvidia/Cosmos3-Nano`
- Resolution: `1280x720`
- Frames / FPS: `121` frames at `24 FPS`
- Steps: `35`
- Seed: `42`
- `--performance-mode manual`
- `--dit-cpu-offload false`
- `--vae-cpu-offload false`
- `--warmup`

| Metric | Main | Patched | Delta |
|---|---:|---:|---:|
| Warmup-excluded total | `102.53s` | `99.29s` | `-3.27%` |
| Denoising stage | `96.56s` | `93.33s` | `-3.46%` |
| Decode stage | `5.59s` | `5.58s` | flat |

### Short E2E Sanity Run

For a faster sanity run at `480x832`, `17` frames, `5` steps:

| Metric | Main | Patched | Delta |
|---|---:|---:|---:|
| Warmup-excluded total | `900.60 ms` | `858.76 ms` | `-4.65%` |
| Denoising stage | `526.79 ms` | `483.27 ms` | `-8.26%` |

Full profiling artifacts are kept locally under:

- `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/REPORT.md`
- `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/reports/`
- `profile/cosmos3-qknorm-rope-coreweave-main-vs-patched/analysis/`

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`.
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
