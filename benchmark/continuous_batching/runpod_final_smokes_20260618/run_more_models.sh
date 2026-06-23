#!/usr/bin/env bash
set -u -o pipefail
cd /workspace/sglang
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export SGLANG_CACHE_DIR=/workspace/sglang-cache
if [ -f /workspace/continuous-bench/.hf_token ]; then
  export HF_TOKEN="$(cat /workspace/continuous-bench/.hf_token)"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
ROOT=/workspace/continuous-bench
LOG_ROOT=$ROOT/logs/more_models
RESULT_ROOT=$ROOT/results/more_models
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
slugify() { echo "$1" | tr '/:.' '---' | tr -cd 'A-Za-z0-9_-' | tr '[:upper:]' '[:lower:]'; }
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
  local log_dir="$LOG_ROOT/$tag"; mkdir -p "$log_dir"
  echo "[$(ts)] start_server tag=$tag mode=$mode model=$model batching=$batching_mode max=$max_size delay=$delay_ms" | tee -a "$LOG_ROOT/queue.log"
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
    > "$log_dir/server_${mode}.log" 2>&1 &
  echo $! > "$ROOT/server.pid"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$(ts)] server_ready tag=$tag mode=$mode" | tee -a "$LOG_ROOT/queue.log"
      return 0
    fi
    if ! kill -0 "$(cat "$ROOT/server.pid")" 2>/dev/null; then
      echo "[$(ts)] server_exit tag=$tag mode=$mode" | tee -a "$LOG_ROOT/queue.log"
      tail -160 "$log_dir/server_${mode}.log" | tee -a "$LOG_ROOT/queue.log"
      return 1
    fi
    if (( i % 12 == 0 )); then
      echo "[$(ts)] waiting tag=$tag mode=$mode i=$i/180" | tee -a "$LOG_ROOT/queue.log"
      tail -20 "$log_dir/server_${mode}.log" | tee -a "$LOG_ROOT/queue.log" || true
    fi
    sleep 10
  done
  echo "[$(ts)] server_timeout tag=$tag mode=$mode" | tee -a "$LOG_ROOT/queue.log"
  tail -160 "$log_dir/server_${mode}.log" | tee -a "$LOG_ROOT/queue.log"
  return 1
}
run_bench() {
  local model="$1" tag="$2" mode="$3" width="$4" height="$5" prompts="$6" run="$7" port="$8"
  local res_dir="$RESULT_ROOT/$tag"; mkdir -p "$res_dir"
  local out="$res_dir/${mode}_${width}x${height}_inf_run${run}.json"
  echo "[$(ts)] bench tag=$tag mode=$mode ${width}x${height} run=$run prompts=$prompts" | tee -a "$LOG_ROOT/queue.log"
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --base-url "http://127.0.0.1:${port}" \
    --model "$model" \
    --dataset vbench \
    --task text-to-image \
    --width "$width" \
    --height "$height" \
    --num-prompts "$prompts" \
    --max-concurrency 4 \
    --request-rate inf \
    --warmup-requests 1 \
    --output-file "$out" \
    --disable-tqdm \
    > "$res_dir/${mode}_${width}x${height}_run${run}.stdout" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(ts)] bench_fail tag=$tag mode=$mode ${width}x${height} run=$run rc=$rc" | tee -a "$LOG_ROOT/queue.log"
    tail -80 "$res_dir/${mode}_${width}x${height}_run${run}.stdout" | tee -a "$LOG_ROOT/queue.log" || true
    return $rc
  fi
  tail -35 "$res_dir/${mode}_${width}x${height}_run${run}.stdout" | tee -a "$LOG_ROOT/queue.log" || true
}
run_mode_cases() {
  local model="$1" tag="$2" mode="$3" batching_mode="$4" max_size="$5" delay_ms="$6" port="$7" matrix="$8"
  start_server "$model" "$tag" "$mode" "$batching_mode" "$max_size" "$delay_ms" "$port" || return 1
  if [ "$matrix" = "small" ]; then
    for run in 1 2; do
      run_bench "$model" "$tag" "$mode" 512 512 16 "$run" "$port" || return 1
      run_bench "$model" "$tag" "$mode" 768 768 16 "$run" "$port" || return 1
    done
  else
    run_bench "$model" "$tag" "$mode" 512 512 4 1 "$port" || return 1
  fi
}
run_matrix() {
  local model="$1" tag="$2" port="$3" matrix="$4"
  echo "[$(ts)] matrix_start tag=$tag model=$model matrix=$matrix" | tee -a "$LOG_ROOT/queue.log"
  run_mode_cases "$model" "$tag" no_batch dynamic 1 0 "$port" "$matrix" || { echo "[$(ts)] matrix_fail tag=$tag mode=no_batch" | tee -a "$LOG_ROOT/queue.log"; return 1; }
  run_mode_cases "$model" "$tag" continuous continuous 4 5 "$port" "$matrix" || { echo "[$(ts)] matrix_fail tag=$tag mode=continuous" | tee -a "$LOG_ROOT/queue.log"; return 1; }
  echo "[$(ts)] matrix_done tag=$tag" | tee -a "$LOG_ROOT/queue.log"
}
probe_config() {
  local model="$1" tag="$2"
  echo "[$(ts)] probe_config tag=$tag model=$model" | tee -a "$LOG_ROOT/queue.log"
  python3 - <<PY >> "$LOG_ROOT/queue.log" 2>&1
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
model = ${model@Q}
pc = PipelineConfig.from_kwargs(dict(model_path=model, backend='sglang'))
print(model, type(pc).__name__, pc.task_type, pc.supports_continuous_batching())
PY
}

: > "$LOG_ROOT/queue.log"
echo "[$(ts)] queue_start" | tee -a "$LOG_ROOT/queue.log"

declare -a SMOKES=(
  'Tongyi-MAI/Z-Image|zimage|30230'
  'Efficient-Large-Model/Sana_600M_512px_diffusers|sana600m512|30231'
  'baidu/ERNIE-Image-Turbo|ernie_turbo|30232'
  'zai-org/GLM-Image|glm_image|30233'
  'stabilityai/stable-diffusion-3-medium-diffusers|sd3_medium|30234'
)
for item in "${SMOKES[@]}"; do
  IFS='|' read -r model tag port <<< "$item"
  probe_config "$model" "$tag" || { echo "[$(ts)] probe_fail tag=$tag" | tee -a "$LOG_ROOT/queue.log"; continue; }
  run_matrix "$model" "$tag" "$port" smoke || true
done

# Ideogram gets the same small 512/768 matrix used for Flux2 Klein.
probe_config 'ideogram-ai/ideogram-4-fp8' 'ideogram4_fp8' || true
run_matrix 'ideogram-ai/ideogram-4-fp8' 'ideogram4_fp8' 30240 small || true

echo "[$(ts)] queue_done" | tee -a "$LOG_ROOT/queue.log"
