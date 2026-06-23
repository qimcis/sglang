#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
output_dir="$2"
output_file="$3"

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}/python"
export HF_HOME=/workspace/hf-cache
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${output_dir}"

sglang generate \
  --model-path nvidia/Cosmos3-Nano \
  --prompt "A curious raccoon walking through a quiet forest" \
  --height 720 \
  --width 1280 \
  --num-frames 121 \
  --fps 24 \
  --num-inference-steps 35 \
  --seed 42 \
  --num-gpus 1 \
  --performance-mode manual \
  --dit-cpu-offload false \
  --vae-cpu-offload false \
  --save-output \
  --output-file-path "${output_dir}/${output_file}" \
  --warmup \
  --perf-dump-path "${output_dir}/perf.json"
