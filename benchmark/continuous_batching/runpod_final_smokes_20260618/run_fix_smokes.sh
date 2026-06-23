#!/usr/bin/env bash
set -u -o pipefail
cd /workspace/sglang
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export SGLANG_CACHE_DIR=/workspace/sglang-cache
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
ROOT=/workspace/continuous-bench
LOG_DIR=$ROOT/logs/fix_smokes
RESULT_DIR=$ROOT/results/fix_smokes
mkdir -p "$LOG_DIR" "$RESULT_DIR"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
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
  local model="$1" tag="$2" mode="$3" batching_mode="$4" max_size="$5" delay_ms="$6" port="$7"
  echo "[$(ts)] start_server tag=$tag mode=$mode model=$model" | tee -a "$LOG_DIR/queue.log"
  cleanup_server
  nohup python3 -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
    --model-path "$model" \
    --backend sglang \
    --num-gpus 1 \
    --host 0.0.0.0 \
    --port "$port" \
    --log-level info \
    --output-path "" \
    --warmup-mode off \
    --performance-mode speed \
    --dit-cpu-offload false \
    --batching-mode "$batching_mode" \
    --batching-max-size "$max_size" \
    --batching-delay-ms "$delay_ms" \
    --enable-batching-metrics \
    > "$LOG_DIR/${tag}_${mode}_server.log" 2>&1 &
  echo $! > "$ROOT/fix_smoke_server.pid"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$(ts)] server_ready tag=$tag mode=$mode" | tee -a "$LOG_DIR/queue.log"
      return 0
    fi
    if ! kill -0 "$(cat "$ROOT/fix_smoke_server.pid")" 2>/dev/null; then
      echo "[$(ts)] server_exit tag=$tag mode=$mode" | tee -a "$LOG_DIR/queue.log"
      tail -160 "$LOG_DIR/${tag}_${mode}_server.log" | tee -a "$LOG_DIR/queue.log"
      return 1
    fi
    if (( i % 12 == 0 )); then
      echo "[$(ts)] waiting tag=$tag mode=$mode i=$i/180" | tee -a "$LOG_DIR/queue.log"
      tail -25 "$LOG_DIR/${tag}_${mode}_server.log" | tee -a "$LOG_DIR/queue.log" || true
    fi
    sleep 10
  done
  echo "[$(ts)] server_timeout tag=$tag mode=$mode" | tee -a "$LOG_DIR/queue.log"
  tail -160 "$LOG_DIR/${tag}_${mode}_server.log" | tee -a "$LOG_DIR/queue.log"
  return 1
}
run_bench() {
  local model="$1" tag="$2" mode="$3" prompts="$4" port="$5"
  local out="$RESULT_DIR/${tag}_${mode}_512x512_smoke.json"
  echo "[$(ts)] bench tag=$tag mode=$mode" | tee -a "$LOG_DIR/queue.log"
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --base-url "http://127.0.0.1:${port}" \
    --model "$model" \
    --dataset vbench \
    --task text-to-image \
    --width 512 \
    --height 512 \
    --num-prompts "$prompts" \
    --max-concurrency 4 \
    --request-rate inf \
    --warmup-requests 1 \
    --output-file "$out" \
    --disable-tqdm \
    > "$RESULT_DIR/${tag}_${mode}_512x512_smoke.stdout" 2>&1
  local rc=$?
  tail -45 "$RESULT_DIR/${tag}_${mode}_512x512_smoke.stdout" | tee -a "$LOG_DIR/queue.log" || true
  return $rc
}
run_pair() {
  local model="$1" tag="$2" prompts="$3" port="$4"
  start_server "$model" "$tag" no_batch dynamic 1 0 "$port" && run_bench "$model" "$tag" no_batch "$prompts" "$port" || true
  start_server "$model" "$tag" continuous continuous 4 5 "$port" && run_bench "$model" "$tag" continuous "$prompts" "$port" || true
}
: > "$LOG_DIR/queue.log"
echo "[$(ts)] fix_smokes_start" | tee -a "$LOG_DIR/queue.log"
run_pair 'Tongyi-MAI/Z-Image' zimage 4 30310
run_pair 'zai-org/GLM-Image' glm_image 4 30311
run_pair 'ideogram-ai/ideogram-4-fp8' ideogram4_fp8 4 30312
echo "[$(ts)] fix_smokes_done" | tee -a "$LOG_DIR/queue.log"
cleanup_server
