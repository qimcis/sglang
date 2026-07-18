#!/usr/bin/env bash

set -Eeuo pipefail

: "${SGLANG_WHEEL:?Set SGLANG_WHEEL to the optimized wheel on /cache}"
: "${KERNEL_WHEEL:?Set KERNEL_WHEEL to the optimized kernel wheel on /cache}"
: "${PREFILL_NODE:?Set PREFILL_NODE to the campaign prefill node}"
: "${DECODE_NODE:?Set DECODE_NODE to the campaign decode node}"
: "${RELEASE:?Set RELEASE to the isolated Helm release}"

readonly KUBECONFIG_PATH="${HOME}/.kube/cw/config"
readonly CONTEXT=cw-us-east-04a
readonly NAMESPACE=staging
readonly CHART_DIR="${HOME}/mooncake-helm/disaggregated-mooncake-connector"
readonly BENCH_POD=chi-misc-testing-ai-load-test-h200
readonly BENCH_BIN=/work/cli/target/release/ai-load-test
readonly REMOTE_PROFILE_RUNNER=/tmp/glm52-kv-run-profile-detached.sh
readonly MODEL="${MODEL:-@cf/staging/glm-5.2-kv-exact-h200}"
readonly BASE_URL="${BASE_URL:-http://${RELEASE}-proxy:8080}"
readonly PROFILE_STEPS="${PROFILE_STEPS:-40}"
readonly NSYS_REMOTE_DIR="${NSYS_REMOTE_DIR:-/cache/sgl-kv-opt-20260717-1af209-v1/nsys-v2}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_VALUES="${SCRIPT_DIR}/../coreweave_glm52_fused_kv_protection_short_ab_20260713/values-common.yaml"
readonly COMMON_VALUES="${SCRIPT_DIR}/values-common.yaml"
readonly LOCAL_RESULTS="${LOCAL_RESULTS:-${SCRIPT_DIR}/artifacts-optimized-v4/nsys}"

ORIGINAL_REVISION=""
STABLE_PREFILL_UID=""
RESTORED=0

export KUBECONFIG="${KUBECONFIG_PATH}"

kube() {
  kubectl --context "${CONTEXT}" -n "${NAMESPACE}" "$@"
}

mode_flags() {
  case "$1" in
    off) printf '0 0\n' ;;
    scheduler) printf '1 1\n' ;;
    fused) printf '1 0\n' ;;
    *) return 1 ;;
  esac
}

restore_original() {
  if [[ "${RESTORED}" == 1 || -z "${ORIGINAL_REVISION}" ]]; then
    return
  fi
  RESTORED=1
  helm rollback "${RELEASE}" "${ORIGINAL_REVISION}" \
    --kube-context "${CONTEXT}" \
    -n "${NAMESPACE}" \
    --wait \
    --timeout 45m
}

apply_profile_mode() {
  local mode="$1"
  local protection
  local disable_fused
  read -r protection disable_fused <<<"$(mode_flags "${mode}")"
  local decode_env
  decode_env="$(jq -nc \
    --arg protection "${protection}" \
    --arg disable_fused "${disable_fused}" \
    --arg output_dir "${NSYS_REMOTE_DIR}" \
    --arg prefix "kv-opt-${mode}" \
    '[
      {name:"SGLANG_KV_PAGE_PROTECTION",value:$protection},
      {name:"SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION",value:$disable_fused},
      {name:"SGLANG_ENABLE_LAYERWISE_NVTX_MARKER",value:"1"},
      {name:"SGLANG_NSYS_PROFILE",value:"1"},
      {name:"SGLANG_NSYS_OUTPUT_DIR",value:$output_dir},
      {name:"SGLANG_NSYS_OUTPUT_PREFIX",value:$prefix},
      {name:"SGLANG_NSYS_TRACE",value:"cuda,nvtx,nccl"},
      {name:"SGLANG_NSYS_CAPTURE_RANGE",value:"cudaProfilerApi"},
      {name:"SGLANG_NSYS_CAPTURE_RANGE_END",value:"stop"},
      {name:"SGLANG_NSYS_EXTRA_ARGS",value:"--stats=true --sample=none --cpuctxsw=none"},
      {name:"NVIDIA_DRIVER_CAPABILITIES",value:"all"}
    ]')"
  local wheel_json
  wheel_json="[\"mooncake-transfer-engine-cuda13==0.3.11.post1\",\"${KERNEL_WHEEL}\",\"${SGLANG_WHEEL}\"]"

  helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
    --kube-context "${CONTEXT}" \
    -n "${NAMESPACE}" \
    -f "${CHART_DIR}/engines/sglang.yaml" \
    -f "${CHART_DIR}/engines/sglang-v0.5.12.post1.yaml" \
    -f "${CHART_DIR}/providers/coreweave.yaml" \
    -f "${CHART_DIR}/models/glm-52-fp8.yaml" \
    -f "${CHART_DIR}/models/glm-52-fp8-cw-staging-natural.yaml" \
    -f "${BASE_VALUES}" \
    -f "${COMMON_VALUES}" \
    --set-json "inferenceServer.pipPackages=${wheel_json}" \
    --set-json "decode.env=${decode_env}" \
    --set-json 'containerSecurityContext.capabilities.add=["IPC_LOCK","SYS_ADMIN"]' \
    --set "prefill.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=${PREFILL_NODE}" \
    --set "decode.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=${DECODE_NODE}" \
    --set zone=cf-us-east-04a-cw.andromeda.cfdata.org \
    --history-max 10

  kube rollout status "statefulset/${RELEASE}-prefill" --timeout=2700s
  kube rollout status "statefulset/${RELEASE}-decode" --timeout=2700s
  kube wait --for=condition=Available "deployment/${RELEASE}-proxy" --timeout=300s

  local actual
  actual="$(kube exec "${RELEASE}-decode-0" -- sh -lc \
    'printf "%s %s\n" "$SGLANG_KV_PAGE_PROTECTION" "$SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION"')"
  [[ "${actual}" == "${protection} ${disable_fused}" ]]
  kube exec "${RELEASE}-decode-0" -- pgrep -f 'nsys profile' >/dev/null

  local prefill_uid
  prefill_uid="$(kube get pod "${RELEASE}-prefill-0" -o jsonpath='{.metadata.uid}')"
  if [[ -z "${STABLE_PREFILL_UID}" ]]; then
    STABLE_PREFILL_UID="${prefill_uid}"
  else
    [[ "${prefill_uid}" == "${STABLE_PREFILL_UID}" ]]
  fi
}

wait_for_report() {
  local report="$1"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if kube exec "${RELEASE}-decode-0" -- test -s "${report}"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

write_stats() {
  local report="$1"
  local stats_report="$2"
  local output="$3"
  local attempt
  for attempt in 1 2 3; do
    if kube exec "${RELEASE}-decode-0" -- nsys stats \
      --report "${stats_report}" \
      --format csv \
      "${report}" >"${output}"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

copy_report() {
  local remote_path="$1"
  local local_path="$2"
  local attempt
  for attempt in 1 2 3; do
    if kube cp "${RELEASE}-decode-0:${remote_path}" "${local_path}"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

run_detached_profile() {
  local duration="$1"
  local remote_result="$2"
  local local_result="$3"
  local remote_status="${remote_result}.status"
  local remote_log="${remote_result}.log"

  kube exec "${BENCH_POD}" -- env \
    "BENCH_BIN=${BENCH_BIN}" \
    "RESULT_PATH=${remote_result}" \
    "STATUS_PATH=${remote_status}" \
    "LOG_PATH=${remote_log}" \
    INPUT_TOKENS=8172 \
    OUTPUT_TOKENS=1000 \
    MAX_CONCURRENCY=1 \
    "DURATION_SECONDS=${duration}" \
    DRAIN_TIMEOUT_SECONDS=900 \
    "MODEL=${MODEL}" \
    "BASE_URL=${BASE_URL}" \
    sh -c "nohup bash '${REMOTE_PROFILE_RUNNER}' >/dev/null 2>&1 </dev/null &"

  local deadline=$((SECONDS + duration + 1500))
  local status=""
  while (( SECONDS < deadline )); do
    set +e
    status="$(kube exec "${BENCH_POD}" -- sh -c \
      'if [ -s "$1" ]; then read -r value <"$1"; printf "%s" "$value"; else exit 3; fi' \
      sh "${remote_status}" 2>/dev/null)"
    local poll_status=$?
    set -e
    if (( poll_status == 0 )); then
      break
    fi
    sleep 10
  done
  if [[ -z "${status}" || "${status}" != "0" ]]; then
    kube exec "${BENCH_POD}" -- sh -c 'cat "$1"' sh "${remote_log}" || true
    return 1
  fi

  local attempt
  for attempt in 1 2 3; do
    if kube cp "${BENCH_POD}:${remote_result}" "${local_result}"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

profile_mode() {
  local mode="$1"
  local output_dir="${LOCAL_RESULTS}/${mode}"
  local report="${NSYS_REMOTE_DIR}/kv-opt-${mode}.nsys-rep"
  local remote_result="/work/results/glm52-kv-opt-nsys-${mode}.json"
  local local_result="${output_dir}/workload.json"

  apply_profile_mode "${mode}"
  local decode_ip
  decode_ip="$(kube get pod "${RELEASE}-decode-0" -o jsonpath='{.status.podIP}')"
  kube exec "${BENCH_POD}" -- mkdir -p /work/results
  kube exec "${RELEASE}-decode-0" -- curl -fsS \
    -H 'Content-Type: application/json' \
    -d "{\"activities\":[\"CUDA_PROFILER\"],\"num_steps\":${PROFILE_STEPS}}" \
    "http://${decode_ip}:8200/start_profile"

  run_detached_profile 30 "${remote_result}" "${local_result}"
  wait_for_report "${report}"
  write_stats "${report}" cuda_gpu_kern_sum "${output_dir}/cuda-kernels.csv"
  write_stats "${report}" cuda_api_sum "${output_dir}/cuda-api.csv"
  if ! copy_report "${report}" "${output_dir}/kv-opt-${mode}.nsys-rep"; then
    printf '%s\n' "${report}" >"${output_dir}/report-remote-path.txt"
  fi
  kube logs "${RELEASE}-decode-0" --timestamps >"${output_dir}/decode.log"

  python3 - "${local_result}" "${output_dir}/decode.log" <<'PY'
import json
import sys

result_path, log_path = sys.argv[1:]
with open(result_path) as file:
    result = json.load(file)
assert all(item["completed"] > 0 for item in result["iterations"])
assert all(item["failed"] == 0 for item in result["iterations"])
assert all(not item["drain_timed_out"] for item in result["iterations"])
with open(log_path) as file:
    assert "KVProtectionIncident" not in file.read()
PY
}

trap restore_original EXIT
trap 'exit 130' INT TERM

ORIGINAL_REVISION="$(helm --kube-context "${CONTEXT}" -n "${NAMESPACE}" list \
  --filter "^${RELEASE}$" -o json | jq -r '.[0].revision')"
[[ "${ORIGINAL_REVISION}" =~ ^[0-9]+$ ]]
kube cp "${SCRIPT_DIR}/run_profile_detached.sh" "${BENCH_POD}:${REMOTE_PROFILE_RUNNER}"

read -r -a profile_modes <<<"${PROFILE_MODES:-off scheduler fused}"
for mode in "${profile_modes[@]}"; do
  [[ "${mode}" =~ ^(off|scheduler|fused)$ ]]
  mkdir -p "${LOCAL_RESULTS}/${mode}"
  profile_mode "${mode}"
done

restore_original
trap - EXIT
