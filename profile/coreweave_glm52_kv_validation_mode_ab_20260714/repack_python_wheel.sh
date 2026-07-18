#!/usr/bin/env bash

set -Eeuo pipefail

: "${BASE_WHEEL:?Set BASE_WHEEL to the validated SGLang wheel}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the source tree containing python/sglang}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique /cache directory}"

readonly WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${OUTPUT_DIR}"
python -m wheel unpack "${BASE_WHEEL}" --dest "${WORK_DIR}"

mapfile -t unpacked_wheels < <(compgen -G "${WORK_DIR}/sglang-*")
if [[ ${#unpacked_wheels[@]} -ne 1 ]]; then
  printf 'Expected one unpacked wheel, found %s\n' "${#unpacked_wheels[@]}" >&2
  exit 1
fi
readonly UNPACKED_WHEEL="${unpacked_wheels[0]}"

for path in \
  srt/model_executor/model_runner.py \
  srt/managers/tp_worker.py \
  srt/managers/utils.py \
  srt/managers/schedule_batch.py \
  srt/managers/scheduler_components/batch_result_processor.py \
  srt/managers/scheduler.py \
  srt/mem_cache/kv_page_tags.py; do
  install -m 644 \
    "${SOURCE_ROOT}/python/sglang/${path}" \
    "${UNPACKED_WHEEL}/sglang/${path}"
done

python - \
  "${UNPACKED_WHEEL}/sglang/srt/model_executor/model_runner.py" \
  "${UNPACKED_WHEEL}/sglang/srt/managers/tp_worker.py" \
  "${UNPACKED_WHEEL}/sglang/srt/managers/utils.py" \
  "${UNPACKED_WHEEL}/sglang/srt/managers/schedule_batch.py" \
  "${UNPACKED_WHEEL}/sglang/srt/managers/scheduler_components/batch_result_processor.py" \
  "${UNPACKED_WHEEL}/sglang/srt/managers/scheduler.py" \
  "${UNPACKED_WHEEL}/sglang/srt/mem_cache/kv_page_tags.py" <<'PY'
import ast
import sys

for path in sys.argv[1:]:
    with open(path) as file:
        compile(file.read(), path, "exec")

table_path = sys.argv[-1]
with open(table_path) as file:
    tree = ast.parse(file.read(), table_path)
table = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVAttentionTagTable")
method = next(node for node in table.body if isinstance(node, ast.FunctionDef) and node.name == "fused_forward_args")
accepted = {arg.arg for arg in method.args.kwonlyargs}
required = {
    "request_indices",
    "seqlens",
    "page_table",
    "page_size",
    "page_table_2",
    "page_table_page_offset",
    "page_table_2_page_offset",
    "page_table_2_window_size",
    "validate_full_mapping",
}
assert required <= accepted, f"fused_forward_args is missing {sorted(required - accepted)}"
PY

python -m wheel pack "${UNPACKED_WHEEL}" --dest-dir "${OUTPUT_DIR}"
sha256sum \
  "${BASE_WHEEL}" \
  "${SOURCE_ROOT}/python/sglang/srt/model_executor/model_runner.py" \
  "${SOURCE_ROOT}/python/sglang/srt/managers/tp_worker.py" \
  "${SOURCE_ROOT}/python/sglang/srt/managers/utils.py" \
  "${SOURCE_ROOT}/python/sglang/srt/managers/schedule_batch.py" \
  "${SOURCE_ROOT}/python/sglang/srt/managers/scheduler_components/batch_result_processor.py" \
  "${SOURCE_ROOT}/python/sglang/srt/managers/scheduler.py" \
  "${SOURCE_ROOT}/python/sglang/srt/mem_cache/kv_page_tags.py" \
  "${OUTPUT_DIR}"/*.whl \
  >"${OUTPUT_DIR}/PROVENANCE_SHA256SUMS"
