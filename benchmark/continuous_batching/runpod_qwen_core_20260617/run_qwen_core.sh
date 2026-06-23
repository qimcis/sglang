#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export SGLANG_CACHE_DIR=/workspace/sglang-cache
BASE_URL=http://127.0.0.1:30200
MODEL=Qwen/Qwen-Image-2512
RESULT_DIR=/workspace/continuous-bench/results/qwen_core
LOG_DIR=/workspace/continuous-bench/logs/qwen_core
mkdir -p "$RESULT_DIR" "$LOG_DIR"

cleanup_server() {
  pkill -f 'sglang.multimodal_gen.benchmarks.bench_serving' || true
  pkill -f 'sglang.multimodal_gen.runtime.entrypoints.cli.main serve' || true
  pkill -f 'sgl_diffusion::scheduler' || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill "$pid" 2>/dev/null || true; done
  sleep 5
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 2
}

start_server() {
  local mode_name="$1"
  local batching_mode="$2"
  local max_size="$3"
  local delay_ms="$4"
  echo "=== starting Qwen server mode=${mode_name} batching_mode=${batching_mode} max=${max_size} delay=${delay_ms} ==="
  cleanup_server
  nohup python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
    --model-path "$MODEL" \
    --backend sglang \
    --num-gpus 1 \
    --host 0.0.0.0 \
    --port 30200 \
    --log-level info \
    --output-path "" \
    --warmup-mode off \
    --performance-mode speed \
    --dit-cpu-offload false \
    --batching-mode "$batching_mode" \
    --batching-max-size "$max_size" \
    --batching-delay-ms "$delay_ms" \
    --enable-batching-metrics \
    > "$LOG_DIR/server_${mode_name}.log" 2>&1 &
  echo $! > /workspace/continuous-bench/server.pid
  for i in $(seq 1 180); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      echo "server ${mode_name} ready"
      return 0
    fi
    if ! kill -0 "$(cat /workspace/continuous-bench/server.pid)" 2>/dev/null; then
      echo "server ${mode_name} exited"
      tail -200 "$LOG_DIR/server_${mode_name}.log"
      return 1
    fi
    if (( i % 12 == 0 )); then
      echo "waiting for ${mode_name}: ${i}/180"
      tail -20 "$LOG_DIR/server_${mode_name}.log" || true
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
    fi
    sleep 10
  done
  echo "server ${mode_name} health timeout"
  tail -200 "$LOG_DIR/server_${mode_name}.log"
  return 1
}

run_case() {
  local mode_name="$1"
  local width="$2"
  local height="$3"
  local prompts="$4"
  local run="$5"
  local out="$RESULT_DIR/${mode_name}_${width}x${height}_inf_run${run}.json"
  echo "=== bench Qwen ${mode_name} ${width}x${height} run${run} ==="
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset vbench \
    --task text-to-image \
    --width "$width" \
    --height "$height" \
    --num-prompts "$prompts" \
    --max-concurrency 4 \
    --request-rate inf \
    --warmup-requests 4 \
    --output-file "$out" \
    --disable-tqdm
}

run_mode() {
  local mode_name="$1"
  local batching_mode="$2"
  local max_size="$3"
  local delay_ms="$4"
  start_server "$mode_name" "$batching_mode" "$max_size" "$delay_ms"
  for run in 1 2 3; do
    run_case "$mode_name" 512 512 64 "$run"
    run_case "$mode_name" 768 768 64 "$run"
    run_case "$mode_name" 1024 1024 32 "$run"
  done
}

run_mode no_batch dynamic 1 0
run_mode dynamic dynamic 4 5
run_mode continuous continuous 4 5

echo "=== qwen core matrix complete ==="
