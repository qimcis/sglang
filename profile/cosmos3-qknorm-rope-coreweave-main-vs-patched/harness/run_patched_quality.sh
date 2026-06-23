#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang-patched
export PYTHONPATH=/workspace/sglang-patched/python
export HF_HOME=/workspace/hf-cache
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1
export CUDA_VISIBLE_DEVICES=0
sglang generate \
  --model-path nvidia/Cosmos3-Nano \
  --prompt "A curious raccoon walking through a quiet forest" \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 35 \
  --seed 42 \
  --num-gpus 1 \
  --performance-mode manual \
  --dit-cpu-offload false \
  --vae-cpu-offload false \
  --save-output \
  --output-file-path /workspace/e2e/patched-quality/patched.mp4 \
  --warmup \
  --perf-dump-path /workspace/e2e/patched-quality/perf.json
