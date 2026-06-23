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
    --batching-mode continuous \
    --batching-max-size 4 \
    --batching-delay-ms 5 \
    --enable-batching-metrics \
    > "$LOG_DIR/server_continuous_retry.log" 2>&1 &
  echo $! > /workspace/continuous-bench/server.pid
  for i in $(seq 1 180); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      echo "server continuous ready"
      return 0
    fi
    if ! kill -0 "$(cat /workspace/continuous-bench/server.pid)" 2>/dev/null; then
      echo "server continuous exited"
      tail -200 "$LOG_DIR/server_continuous_retry.log"
      return 1
    fi
    if (( i % 12 == 0 )); then
      echo "waiting for continuous: ${i}/180"
      tail -30 "$LOG_DIR/server_continuous_retry.log" || true
    fi
    sleep 10
  done
  echo "server continuous health timeout"
  tail -200 "$LOG_DIR/server_continuous_retry.log"
  return 1
}
run_case() {
  local width="$1"; local height="$2"; local prompts="$3"; local run="$4"
  local out="$RESULT_DIR/continuous_${width}x${height}_inf_run${run}.json"
  echo "=== bench Qwen continuous ${width}x${height} run${run} ==="
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
start_server
for run in 1 2 3; do
  run_case 512 512 64 "$run"
  run_case 768 768 64 "$run"
  run_case 1024 1024 32 "$run"
done
echo "=== qwen continuous core complete ==="
