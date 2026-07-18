#!/usr/bin/env bash

set -Eeuo pipefail

: "${SGLANG_WHEEL:?Set SGLANG_WHEEL to the new wheel path on /cache}"
: "${PREFILL_NODE:?Set PREFILL_NODE to a verified schedulable H200 node}"
: "${DECODE_NODE:?Set DECODE_NODE to a verified schedulable H200 node}"
: "${RELEASE:?Set RELEASE to the isolated Helm release name}"
: "${KERNEL_WHEEL:?Set KERNEL_WHEEL to the new kernel wheel path on /cache}"
[[ "${PREFILL_NODE}" != "${DECODE_NODE}" ]]
[[ "${RELEASE}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]
[[ "${RELEASE}" != "glm52-fused-kv-h200" ]]

readonly KUBECONFIG_PATH="${HOME}/.kube/cw/config"
readonly CONTEXT=cw-us-east-04a
readonly NAMESPACE=staging
readonly RELEASE
readonly CHART_DIR="${HOME}/mooncake-helm/disaggregated-mooncake-connector"
readonly BENCH_POD=chi-misc-testing-ai-load-test-h200
readonly BENCH_BIN=/work/cli/target/release/ai-load-test
readonly REMOTE_PROFILE_RUNNER=/tmp/glm52-kv-run-profile-detached.sh
readonly MODEL="${MODEL:-@cf/staging/glm-5.2-kv-exact-h200}"
readonly BASE_URL="${BASE_URL:-http://${RELEASE}-proxy:8080}"
readonly KERNEL_WHEEL
readonly CAMPAIGN_ID="${CAMPAIGN_ID:-glm52-kv-validation-mode-20260714}"
readonly CAMPAIGN_SEQUENCE="${CAMPAIGN_SEQUENCE:-off:1 scheduler:1 fused:1 fused:2 scheduler:2 off:2}"
readonly REMOTE_ROOT="/work/results/${CAMPAIGN_ID}"
readonly INPUT_TOKENS=8172
readonly OUTPUT_TOKENS=1000
readonly WARMUP_SECONDS=30
readonly MEASUREMENT_SECONDS=120
readonly DRAIN_TIMEOUT_SECONDS=900
readonly PROXY_STABILITY_SECONDS=30
readonly PREFILL_NODE
readonly DECODE_NODE

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_VALUES="${SCRIPT_DIR}/../coreweave_glm52_fused_kv_protection_short_ab_20260713/values-common.yaml"
readonly COMMON_VALUES="${SCRIPT_DIR}/values-common.yaml"
readonly LOCAL_RESULTS="${LOCAL_RESULTS:-${SCRIPT_DIR}/artifacts}"

ORIGINAL_REVISION=""
STABLE_PREFILL_UID=""
RESTORED=0

export KUBECONFIG="${KUBECONFIG_PATH}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

kube() {
  kubectl --context "${CONTEXT}" -n "${NAMESPACE}" "$@"
}

verify_target_node() {
  local node="$1"
  local model
  local gpu_count
  local unschedulable
  model="$(kubectl --context "${CONTEXT}" get node "${node}" -o jsonpath='{.metadata.labels.gpu\.nvidia\.com/model}')"
  gpu_count="$(kubectl --context "${CONTEXT}" get node "${node}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
  unschedulable="$(kubectl --context "${CONTEXT}" get node "${node}" -o jsonpath='{.spec.unschedulable}')"
  [[ "${model}" == "H200" ]]
  [[ "${gpu_count}" -ge 8 ]]
  [[ "${unschedulable}" != "true" ]]
}

overlay_for_mode() {
  case "$1" in
    off|scheduler) printf '%s/values-%s.yaml\n' "${SCRIPT_DIR}" "$1" ;;
    fused) printf '%s/values-fused.yaml\n' "${SCRIPT_DIR}" ;;
    *) log "Unknown validation mode: $1"; return 1 ;;
  esac
}

expected_decode_flags() {
  case "$1" in
    off) printf '0 1 0\n' ;;
    scheduler) printf '1 1 1\n' ;;
    fused) printf '1 1 0\n' ;;
    *) return 1 ;;
  esac
}

wait_for_proxy_stable() {
  local selector="app.kubernetes.io/instance=${RELEASE},app.kubernetes.io/component=proxy"
  local deadline=$((SECONDS + 600))

  while (( SECONDS < deadline )); do
    kube wait --for=condition=Ready pod -l "${selector}" --timeout=300s

    local pod_uid_before
    local restart_count_before
    pod_uid_before="$(kube get pod -l "${selector}" -o jsonpath='{.items[0].metadata.uid}')"
    restart_count_before="$(kube get pod -l "${selector}" -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}')"
    sleep "${PROXY_STABILITY_SECONDS}"

    local pod_uid_after
    local restart_count_after
    local ready_after
    pod_uid_after="$(kube get pod -l "${selector}" -o jsonpath='{.items[0].metadata.uid}')"
    restart_count_after="$(kube get pod -l "${selector}" -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}')"
    ready_after="$(kube get pod -l "${selector}" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')"

    if [[ "${pod_uid_before}" == "${pod_uid_after}" &&
          "${restart_count_before}" == "${restart_count_after}" &&
          "${ready_after}" == "True" ]]; then
      log "Proxy remained ready without restarting for ${PROXY_STABILITY_SECONDS}s"
      return 0
    fi

    log "Proxy changed or restarted during readiness guard; waiting again"
  done

  log "Timed out waiting for a stable proxy"
  return 1
}

helm_apply() {
  local mode="$1"
  local overlay
  overlay="$(overlay_for_mode "${mode}")"
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
    -f "${overlay}" \
    --set-json "inferenceServer.pipPackages=${wheel_json}" \
    --set "prefill.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=${PREFILL_NODE}" \
    --set "decode.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=${DECODE_NODE}" \
    --set zone=cf-us-east-04a-cw.andromeda.cfdata.org \
    --history-max 10
}

verify_mode() {
  local mode="$1"
  local expected
  expected="$(expected_decode_flags "${mode}")"

  kube rollout status "statefulset/${RELEASE}-prefill" --timeout=2700s
  kube rollout status "statefulset/${RELEASE}-decode" --timeout=2700s
  kube wait --for=condition=Available "deployment/${RELEASE}-proxy" --timeout=300s
  wait_for_proxy_stable

  local actual
  actual="$(kube exec "${RELEASE}-decode-0" -- python -c \
    "import os; print(os.environ['SGLANG_KV_PAGE_PROTECTION'], os.environ['SGLANG_KV_TRANSFER_CHECKSUM'], os.environ['SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION'])")"
  [[ "${actual}" == "${expected}" ]]

  kube exec "${RELEASE}-decode-0" -- python -c '
import inspect
from sgl_kernel.kvcacheio import kv_checksum_direct_table_batched_with_pages_compact
from sglang.srt.mem_cache.kv_page_tags import KVPageProtectionManager
from sglang.srt.mem_cache.kv_page_tags import should_use_fused_kv_page_protection
assert "SGLANG_DISABLE_FUSED_KV_PAGE_PROTECTION" in inspect.getsource(should_use_fused_kv_page_protection)
assert "page_output_offsets" in inspect.signature(kv_checksum_direct_table_batched_with_pages_compact).parameters
checksum_source = inspect.getsource(KVPageProtectionManager.begin_transfer_checksums_from_table)
assert "batch_size + total_num_pages" in checksum_source
'

  if [[ "${mode}" == "fused" ]]; then
    kube exec "${RELEASE}-decode-0" -- python -c '
import inspect
import torch
from sglang.srt.model_executor.model_runner import FusedKVPageProtectionCheck
from sglang.srt.mem_cache.kv_page_tags import KVAttentionTagTable

begin_source = inspect.getsource(KVAttentionTagTable.begin_fused_forward)
status_source = inspect.getsource(KVAttentionTagTable.fused_failure_status)
assert "kv_page_protection_begin_forward" in begin_source
assert "kv_page_protection_failure_status" in status_source
assert "failed" in FusedKVPageProtectionCheck.__dataclass_fields__
assert torch.ops.sgl_kernel.kv_page_protection_begin_forward
assert torch.ops.sgl_kernel.kv_page_protection_failure_status
'
  fi

  local prefill_uid
  prefill_uid="$(kube get pod "${RELEASE}-prefill-0" -o jsonpath='{.metadata.uid}')"
  if [[ -z "${STABLE_PREFILL_UID}" ]]; then
    STABLE_PREFILL_UID="${prefill_uid}"
  else
    [[ "${prefill_uid}" == "${STABLE_PREFILL_UID}" ]]
  fi
}

ensure_mode() {
  local mode="$1"
  log "Rolling decode to validation mode ${mode}"
  helm_apply "${mode}"
  verify_mode "${mode}"
}

validate_result() {
  local path="$1"
  local duration="$2"
  python3 - "${path}" "${duration}" "${MODEL}" <<'PY'
import json
import sys

path, duration, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path) as f:
    doc = json.load(f)
assert doc["config"] == {
    "input_tokens": 8172,
    "max_output_tokens": 1000,
    "max_concurrency": 8,
    "duration_secs": duration,
    "drain_timeout_secs": 900,
    "model": model,
}
assert [item["concurrency"] for item in doc["iterations"]] == [1, 2, 4, 8]
assert all(item["failed"] == 0 for item in doc["iterations"])
assert all(not item["drain_timed_out"] for item in doc["iterations"])
assert all(item["completed"] > 0 for item in doc["iterations"])
PY
}

wait_for_profile_idle() {
  local deadline=$((SECONDS + DRAIN_TIMEOUT_SECONDS + 60))
  local status

  while (( SECONDS < deadline )); do
    set +e
    kube exec "${BENCH_POD}" -- pgrep -f '[a]i-load-test profile-decode' \
      >/dev/null 2>&1
    status=$?
    set -e
    if (( status == 1 )); then
      sleep 30
      return 0
    fi
    sleep 5
  done

  log "Timed out waiting for the previous profile process to exit"
  return 1
}

run_profile() {
  local duration="$1"
  local remote_path="$2"
  local local_path="$3"
  local remote_status="${remote_path}.status"
  local remote_log="${remote_path}.log"

  local result_attempt=1
  while true; do
    kube exec "${BENCH_POD}" -- env \
      "BENCH_BIN=${BENCH_BIN}" \
      "RESULT_PATH=${remote_path}" \
      "STATUS_PATH=${remote_status}" \
      "LOG_PATH=${remote_log}" \
      "INPUT_TOKENS=${INPUT_TOKENS}" \
      "OUTPUT_TOKENS=${OUTPUT_TOKENS}" \
      "DURATION_SECONDS=${duration}" \
      "DRAIN_TIMEOUT_SECONDS=${DRAIN_TIMEOUT_SECONDS}" \
      "MODEL=${MODEL}" \
      "BASE_URL=${BASE_URL}" \
      sh -c "nohup bash '${REMOTE_PROFILE_RUNNER}' >/dev/null 2>&1 </dev/null &"

    local deadline=$((SECONDS + 4 * duration + DRAIN_TIMEOUT_SECONDS + 600))
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
    if [[ -z "${status}" ]]; then
      log "Timed out waiting for detached profile status"
      return 1
    fi
    if [[ "${status}" != "0" ]]; then
      kube exec "${BENCH_POD}" -- sh -c 'cat "$1"' sh "${remote_log}" || true
      return "${status}"
    fi

    local copy_attempt=1
    until kube cp "${BENCH_POD}:${remote_path}" "${local_path}"; do
      if (( copy_attempt >= 3 )); then
        return 1
      fi
      copy_attempt=$((copy_attempt + 1))
      sleep 10
    done
    if validate_result "${local_path}" "${duration}"; then
      return 0
    fi
    if (( result_attempt >= 3 )); then
      return 1
    fi

    result_attempt=$((result_attempt + 1))
    wait_for_profile_idle
    wait_for_proxy_stable
    log "Retrying rejected profile sample (attempt ${result_attempt}/3)"
  done
}

capture_arm_evidence() {
  local mode="$1"
  local repeat="$2"
  local output_dir="${LOCAL_RESULTS}/${mode}"
  local log_path="${output_dir}/decode-repeat${repeat}.log"
  local pod_path="${output_dir}/decode-repeat${repeat}-pod.json"

  kube logs "${RELEASE}-decode-0" --timestamps >"${log_path}"
  kube get pod "${RELEASE}-decode-0" -o json >"${pod_path}"
  python3 - "${log_path}" "${pod_path}" <<'PY'
import json
import sys

log_path, pod_path = sys.argv[1:]
with open(log_path) as file:
    log_text = file.read()
assert "KVProtectionIncident" not in log_text, "decode log contains a KV protection incident"

with open(pod_path) as file:
    pod = json.load(file)
container = pod["status"]["containerStatuses"][0]
assert container["ready"], "decode container was not ready after measurement"
assert container["restartCount"] == 0, "decode container restarted during the arm"
PY
}

run_arm() {
  local mode="$1"
  local repeat="$2"
  local output_dir="${LOCAL_RESULTS}/${mode}"
  local remote_dir="${REMOTE_ROOT}/${mode}"

  ensure_mode "${mode}"
  log "Warming ${mode} repeat ${repeat}"
  run_profile \
    "${WARMUP_SECONDS}" \
    "${remote_dir}/warmup-repeat${repeat}.json" \
    "${output_dir}/warmup-repeat${repeat}.json"
  log "Measuring ${mode} repeat ${repeat}"
  run_profile \
    "${MEASUREMENT_SECONDS}" \
    "${remote_dir}/repeat${repeat}.json" \
    "${output_dir}/repeat${repeat}.json"
  capture_arm_evidence "${mode}" "${repeat}"
}

restore_original() {
  if [[ "${RESTORED}" == 1 || -z "${ORIGINAL_REVISION}" ]]; then
    return
  fi
  RESTORED=1
  log "Restoring Helm revision ${ORIGINAL_REVISION}"
  helm rollback "${RELEASE}" "${ORIGINAL_REVISION}" \
    --kube-context "${CONTEXT}" \
    -n "${NAMESPACE}" \
    --wait \
    --timeout 45m
}

trap restore_original EXIT
trap 'exit 130' INT TERM

ORIGINAL_REVISION="$(helm --kube-context "${CONTEXT}" -n "${NAMESPACE}" list \
  --filter "^${RELEASE}$" -o json | jq -r '.[0].revision')"
[[ "${ORIGINAL_REVISION}" =~ ^[0-9]+$ ]]
verify_target_node "${PREFILL_NODE}"
verify_target_node "${DECODE_NODE}"

for mode in off scheduler fused; do
  mkdir -p "${LOCAL_RESULTS}/${mode}"
  kube exec "${BENCH_POD}" -- mkdir -p "${REMOTE_ROOT}/${mode}"
done
kube cp "${SCRIPT_DIR}/run_profile_detached.sh" "${BENCH_POD}:${REMOTE_PROFILE_RUNNER}"
wait_for_profile_idle

log "Original Helm revision: ${ORIGINAL_REVISION}"
log "Wheel: ${SGLANG_WHEEL}"
log "Sequence: ${CAMPAIGN_SEQUENCE}"

read -r -a campaign_arms <<< "${CAMPAIGN_SEQUENCE}"
for arm in "${campaign_arms[@]}"; do
  mode="${arm%%:*}"
  repeat="${arm##*:}"
  [[ "${mode}" =~ ^(off|scheduler|fused)$ ]]
  [[ "${repeat}" =~ ^[12]$ ]]
  run_arm "${mode}" "${repeat}"
done

restore_original
trap - EXIT
log "Campaign complete and original Helm revision restored"
