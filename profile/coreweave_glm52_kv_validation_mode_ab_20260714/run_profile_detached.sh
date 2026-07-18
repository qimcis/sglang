set -Eeuo pipefail

: "${BENCH_BIN:?}"
: "${RESULT_PATH:?}"
: "${STATUS_PATH:?}"
: "${LOG_PATH:?}"
: "${INPUT_TOKENS:?}"
: "${OUTPUT_TOKENS:?}"
: "${DURATION_SECONDS:?}"
: "${DRAIN_TIMEOUT_SECONDS:?}"
: "${MODEL:?}"
: "${BASE_URL:?}"
: "${MAX_CONCURRENCY:=8}"

rm -f "${RESULT_PATH}" "${STATUS_PATH}" "${STATUS_PATH}.tmp" "${LOG_PATH}"
set +e
"${BENCH_BIN}" profile-decode \
  --input-tokens "${INPUT_TOKENS}" \
  --max-output-tokens "${OUTPUT_TOKENS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --duration "${DURATION_SECONDS}" \
  --drain-timeout "${DRAIN_TIMEOUT_SECONDS}" \
  --model "${MODEL}" \
  --json-output "${RESULT_PATH}" \
  --no-upload \
  partner-bouncer \
  --base-url "${BASE_URL}" \
  >"${LOG_PATH}" 2>&1
status=$?
set -e
printf '%s\n' "${status}" >"${STATUS_PATH}.tmp"
mv "${STATUS_PATH}.tmp" "${STATUS_PATH}"
exit "${status}"
