# Continuous Batching Benchmark Results

Generated: 2026-06-17 23:03:35 UTC

## Setup

- Host: RunPod 1x NVIDIA H100 80GB HBM3
- Branch: `continous-batch`
- Commit used for code checkout before local RunPod hotfix: `232e03e1b`
- Local fix committed after finding the continuous startup bug: `c7ad2fa6a fix`
- Model: `Qwen/Qwen-Image-2512`
- Backend: native `sglang`
- Benchmark client: `python -m sglang.multimodal_gen.benchmarks.bench_serving`
- Dataset: `vbench`
- Traffic: `request-rate=inf`, `max-concurrency=4`, `warmup-requests=4`
- Repeats: 3 per mode/resolution

## Server Modes

| Mode | Flags |
|---|---|
| No Batching | `--batching-mode dynamic --batching-max-size 1 --batching-delay-ms 0` |
| Dynamic Batching | `--batching-mode dynamic --batching-max-size 4 --batching-delay-ms 5` |
| Continuous Batching | `--batching-mode continuous --batching-max-size 4 --batching-delay-ms 5` |

Common server flags: `--backend sglang --num-gpus 1 --performance-mode speed --dit-cpu-offload false --warmup-mode off --enable-batching-metrics`.


## Commands run

The benchmark ran from `/workspace/sglang` on the RunPod H100 host with these environment variables:

```bash
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export SGLANG_CACHE_DIR=/workspace/sglang-cache
```

Before each server mode, stale server and benchmark processes were stopped so only one model server used the GPU:

```bash
pkill -f 'sglang.multimodal_gen.benchmarks.bench_serving' || true
pkill -f 'sglang.multimodal_gen.runtime.entrypoints.cli.main serve' || true
pkill -f 'sgl_diffusion::scheduler' || true
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill "$pid" 2>/dev/null || true
done
sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2
```

### Server commands

No batching:

```bash
python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
  --model-path Qwen/Qwen-Image-2512 \
  --backend sglang \
  --num-gpus 1 \
  --host 0.0.0.0 \
  --port 30200 \
  --log-level info \
  --output-path "" \
  --warmup-mode off \
  --performance-mode speed \
  --dit-cpu-offload false \
  --batching-mode dynamic \
  --batching-max-size 1 \
  --batching-delay-ms 0 \
  --enable-batching-metrics
```

Dynamic batching:

```bash
python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
  --model-path Qwen/Qwen-Image-2512 \
  --backend sglang \
  --num-gpus 1 \
  --host 0.0.0.0 \
  --port 30200 \
  --log-level info \
  --output-path "" \
  --warmup-mode off \
  --performance-mode speed \
  --dit-cpu-offload false \
  --batching-mode dynamic \
  --batching-max-size 4 \
  --batching-delay-ms 5 \
  --enable-batching-metrics
```

Continuous batching:

```bash
python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
  --model-path Qwen/Qwen-Image-2512 \
  --backend sglang \
  --num-gpus 1 \
  --host 0.0.0.0 \
  --port 30200 \
  --log-level info \
  --output-path "" \
  --warmup-mode off \
  --performance-mode speed \
  --dit-cpu-offload false \
  --batching-mode continuous \
  --batching-max-size 4 \
  --batching-delay-ms 5 \
  --enable-batching-metrics
```

### Benchmark commands

Each mode used the same benchmark command shape. The only changes were `<WIDTH>`, `<HEIGHT>`, `<NUM_PROMPTS>`, and output file name.

```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --base-url http://127.0.0.1:30200 \
  --model Qwen/Qwen-Image-2512 \
  --dataset vbench \
  --task text-to-image \
  --width <WIDTH> \
  --height <HEIGHT> \
  --num-prompts <NUM_PROMPTS> \
  --max-concurrency 4 \
  --request-rate inf \
  --warmup-requests 4 \
  --output-file /workspace/continuous-bench/results/qwen_core/<MODE>_<WIDTH>x<HEIGHT>_inf_run<RUN>.json \
  --disable-tqdm
```

The concrete resolution matrix was:

```bash
# 512x512
--width 512 --height 512 --num-prompts 64

# 768x768
--width 768 --height 768 --num-prompts 64

# 1024x1024
--width 1024 --height 1024 --num-prompts 32
```

### Recreate a single request with `sglang generate`

Use `sglang generate` for a single offline request. This does not benchmark concurrent serving, but it recreates the same native model path and GPU-resident runtime settings used by the server benchmarks.

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-2512 \
  --backend sglang \
  --num-gpus 1 \
  --performance-mode speed \
  --dit-cpu-offload false \
  --prompt "A cinematic photo of a red fox sitting in a snowy forest at sunrise" \
  --width 512 \
  --height 512 \
  --output-file-path /workspace/continuous-bench/results/qwen_generate_512.png
```

If the installed `sglang` entrypoint does not expose the multimodal subcommands in your environment, use the module entrypoint directly:

```bash
python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path Qwen/Qwen-Image-2512 \
  --backend sglang \
  --num-gpus 1 \
  --performance-mode speed \
  --dit-cpu-offload false \
  --prompt "A cinematic photo of a red fox sitting in a snowy forest at sunrise" \
  --width 512 \
  --height 512 \
  --output-file-path /workspace/continuous-bench/results/qwen_generate_512.png
```

## Main Results

| Resolution | Mode | n | Throughput (req/s) | Δ vs No Batch | Mean Lat (s) | Δ | P99 Lat (s) | Δ | Duration (s) | Δ | Peak Mem (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512x512 | No Batching | 3 | 0.125 | baseline | 31.17 | baseline | 32.28 | baseline | 510.71 | baseline | 55071 |
| 512x512 | Dynamic Batching | 3 | 0.218 | +73.8% | 18.14 | -41.8% | 18.98 | -41.2% | 293.91 | -42.5% | 57699 |
| 512x512 | Continuous Batching | 3 | 0.294 | +134.5% | 13.60 | -56.4% | 13.73 | -57.5% | 217.80 | -57.4% | 57539 |
| 768x768 | No Batching | 3 | 0.116 | baseline | 33.64 | baseline | 34.66 | baseline | 551.12 | baseline | 56341 |
| 768x768 | Dynamic Batching | 3 | 0.135 | +16.4% | 29.26 | -13.0% | 31.85 | -8.1% | 473.34 | -14.1% | 61809 |
| 768x768 | Continuous Batching | 3 | 0.142 | +22.2% | 28.18 | -16.2% | 28.28 | -18.4% | 451.02 | -18.2% | 58028 |
| 1024x1024 | No Batching | 3 | 0.073 | baseline | 52.40 | baseline | 55.10 | baseline | 439.79 | baseline | 58602 |
| 1024x1024 | Dynamic Batching | 3 | 0.078 | +6.9% | 50.20 | -4.2% | 55.57 | +0.8% | 411.31 | -6.5% | 68127 |
| 1024x1024 | Continuous Batching | 3 | 0.078 | +7.3% | 51.20 | -2.3% | 51.47 | -6.6% | 409.94 | -6.8% | 58768 |

## Continuous vs Dynamic

| Resolution | Throughput Δ | Mean Lat Δ | P99 Lat Δ | Duration Δ | Peak Mem Δ |
|---|---:|---:|---:|---:|---:|
| 512x512 | +34.9% | -25.0% | -27.7% | -25.9% | -0.3% |
| 768x768 | +4.9% | -3.7% | -11.2% | -4.7% | -6.1% |
| 1024x1024 | +0.3% | +2.0% | -7.4% | -0.3% | -13.7% |

## Per-Run Results

| Resolution | Mode | Run | Throughput | Mean Lat | P99 Lat | Duration | Peak Mem | Failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512x512 | No Batching | 1 | 0.125 | 31.18 | 32.18 | 510.83 | 55074 | 0 |
| 512x512 | No Batching | 2 | 0.125 | 31.31 | 32.57 | 512.82 | 55070 | 0 |
| 512x512 | No Batching | 3 | 0.126 | 31.02 | 32.09 | 508.49 | 55070 | 0 |
| 512x512 | Dynamic Batching | 1 | 0.218 | 18.16 | 18.97 | 293.14 | 57326 | 0 |
| 512x512 | Dynamic Batching | 2 | 0.223 | 17.53 | 18.71 | 286.47 | 58444 | 0 |
| 512x512 | Dynamic Batching | 3 | 0.212 | 18.72 | 19.26 | 302.12 | 57326 | 0 |
| 512x512 | Continuous Batching | 1 | 0.294 | 13.60 | 13.69 | 217.75 | 55164 | 0 |
| 512x512 | Continuous Batching | 2 | 0.295 | 13.54 | 13.65 | 216.78 | 58604 | 0 |
| 512x512 | Continuous Batching | 3 | 0.292 | 13.67 | 13.84 | 218.88 | 58850 | 0 |
| 768x768 | No Batching | 1 | 0.116 | 33.64 | 34.58 | 551.14 | 56346 | 0 |
| 768x768 | No Batching | 2 | 0.116 | 33.65 | 34.87 | 551.30 | 56338 | 0 |
| 768x768 | No Batching | 3 | 0.116 | 33.63 | 34.54 | 550.93 | 56338 | 0 |
| 768x768 | Dynamic Batching | 1 | 0.135 | 29.35 | 29.72 | 474.89 | 60964 | 0 |
| 768x768 | Dynamic Batching | 2 | 0.135 | 29.27 | 29.74 | 473.56 | 60964 | 0 |
| 768x768 | Dynamic Batching | 3 | 0.136 | 29.15 | 36.08 | 471.58 | 63500 | 0 |
| 768x768 | Continuous Batching | 1 | 0.142 | 28.18 | 28.26 | 451.02 | 56630 | 0 |
| 768x768 | Continuous Batching | 2 | 0.142 | 28.17 | 28.28 | 450.94 | 58604 | 0 |
| 768x768 | Continuous Batching | 3 | 0.142 | 28.18 | 28.29 | 451.09 | 58850 | 0 |
| 1024x1024 | No Batching | 1 | 0.073 | 52.40 | 54.99 | 439.85 | 58602 | 0 |
| 1024x1024 | No Batching | 2 | 0.073 | 52.39 | 55.07 | 439.73 | 58602 | 0 |
| 1024x1024 | No Batching | 3 | 0.073 | 52.40 | 55.24 | 439.77 | 58602 | 0 |
| 1024x1024 | Dynamic Batching | 1 | 0.078 | 50.17 | 51.48 | 411.64 | 66714 | 0 |
| 1024x1024 | Dynamic Batching | 2 | 0.078 | 50.29 | 51.49 | 411.77 | 66714 | 0 |
| 1024x1024 | Dynamic Batching | 3 | 0.078 | 50.14 | 63.73 | 410.53 | 70954 | 0 |
| 1024x1024 | Continuous Batching | 1 | 0.078 | 51.22 | 51.63 | 410.09 | 58604 | 0 |
| 1024x1024 | Continuous Batching | 2 | 0.078 | 51.21 | 51.52 | 410.05 | 58850 | 0 |
| 1024x1024 | Continuous Batching | 3 | 0.078 | 51.17 | 51.26 | 409.69 | 58850 | 0 |


## Additional T2I smoke coverage

After the core Qwen benchmark, additional prompt-only T2I smoke tests were run on the same RunPod 1xH100 setup. This provides representative T2I smoke coverage across Qwen, FLUX2 Klein, Z-Image, SANA, ERNIE, GLM, and Ideogram. These smoke tests used 512x512, `max-concurrency=4`, and small prompt counts to validate model compatibility.

| Model | No batching | Continuous batching | Notes |
|---|---|---|---|
| `Qwen/Qwen-Image-2512` | ✅ | ✅ | Full 27-run core matrix completed. |
| `black-forest-labs/FLUX.2-klein-4B` | ✅ | ✅ | 512/768 small matrix, 2 repeats per mode. |
| `black-forest-labs/FLUX.2-klein-base-4B` | ✅ | ✅ | 512/768 small matrix, 2 repeats per mode. |
| `Tongyi-MAI/Z-Image` | ✅ | ✅ | Continuous passed after fixing packed per-sample caption list merge. |
| `Efficient-Large-Model/Sana_600M_512px_diffusers` | ✅ | ✅ | Smoke passed. |
| `baidu/ERNIE-Image-Turbo` | ✅ | ✅ | Smoke passed; continuous was slower in the smoke. |
| `zai-org/GLM-Image` | ✅ | ✅ | Continuous passed after cloning the scheduler runtime per request. |
| `ideogram-ai/ideogram-4-fp8` | ✅ | ✅ | Passed after adding CUDA 13 NVRTC builtins to `LD_LIBRARY_PATH` in the RunPod environment and disabling generic packed denoising for custom denoising subclasses. |

Models probed but not benchmarked in this run:

- `black-forest-labs/FLUX.1-schnell`: blocked by gated Hugging Face access during the first probe.
- `black-forest-labs/FLUX.2-dev`: gated Hugging Face model; not part of the completed benchmark tables.
- `black-forest-labs/FLUX.2-klein-base-9B` and `black-forest-labs/FLUX.2-klein-9B`: not run due to 1xH100 scope.
- Image-edit and image/3D-conditioned models were not included in continuous batching results because current continuous request validation rejects image-conditioned requests.

## Notes

- Continuous batching initially failed for `QwenImagePipeline` because it uses `ProgressiveDenoisingStageRouter`; the continuous split helper only looked for direct `DenoisingStage` instances.
- The fix unwraps `ProgressiveDenoisingStageRouter.standard_stage` for full-resolution continuous batching.
- Continuous batching completed all 9 core runs with zero failed requests.
- Continuous logs showed full step batches such as `Continuous denoising step batch: size=4/4`.
- `Qwen/Qwen-Image-2512` is slow on 1xH100 in this configuration, but continuous batching still improved throughput and latency versus both no batching and dynamic batching in the core burst workload.
- Flux2 Klein was not part of this core matrix, but continuous batching should support prompt-only Flux2 requests after the request-level validation fix that mirrors dynamic batching behavior.

## PR Output Images

Two 1024x1024 Qwen output images were generated on the same RunPod H100 host through the OpenAI-compatible `/v1/images/generations` endpoint using this payload:

```json
{
  "model": "Qwen/Qwen-Image-2512",
  "prompt": "A cinematic photo of a red fox sitting in a snowy forest at sunrise, highly detailed, natural lighting",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "response_format": "b64_json"
}
```

The API returned JPEG image bytes; local artifacts use `.jpg` extensions to match the payload format.

| Mode | Server batching flags | Status | Elapsed | Local artifact |
|---|---|---:|---:|---|
| No Batching | `--batching-mode dynamic --batching-max-size 1 --batching-delay-ms 0` | 200 | 22.726 s | `pr_outputs/qwen_1024/qwen_1024_no_batch.jpg` |
| Continuous Batching | `--batching-mode continuous --batching-max-size 4 --batching-delay-ms 5` | 200 | 15.857 s | `pr_outputs/qwen_1024/qwen_1024_continuous.jpg` |

## Artifact Layout

- JSON results: `results/qwen_core/*.json`
- Server logs: `logs/qwen_core/server_*.log`
- Driver logs: `logs/qwen_core/*driver.log`
- PR output images and responses: `pr_outputs/qwen_1024/*`
- Runner scripts: `run_qwen_core.sh`, `run_qwen_continuous_core.sh`
