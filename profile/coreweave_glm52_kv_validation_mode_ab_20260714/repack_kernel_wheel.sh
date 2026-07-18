#!/usr/bin/env bash

set -Eeuo pipefail

: "${BASE_WHEEL:?Set BASE_WHEEL to the prior fused sglang-kernel wheel}"
: "${COMMON_OPS:?Set COMMON_OPS to the newly linked SM90 common_ops library}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the isolated SGLang source tree}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a unique /cache directory}"

readonly WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${OUTPUT_DIR}"
python -m wheel unpack "${BASE_WHEEL}" --dest "${WORK_DIR}"

mapfile -t unpacked_wheels < <(compgen -G "${WORK_DIR}/sglang_kernel-*")
if [[ ${#unpacked_wheels[@]} -ne 1 ]]; then
  printf 'Expected one unpacked wheel, found %s\n' "${#unpacked_wheels[@]}" >&2
  exit 1
fi
readonly UNPACKED_WHEEL="${unpacked_wheels[0]}"

install -m 755 \
  "${COMMON_OPS}" \
  "${UNPACKED_WHEEL}/sgl_kernel/sm90/common_ops.abi3.so"
install -m 644 \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/top_k.py" \
  "${UNPACKED_WHEEL}/sgl_kernel/top_k.py"
install -m 644 \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/kvcacheio.py" \
  "${UNPACKED_WHEEL}/sgl_kernel/kvcacheio.py"
install -m 644 \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/__init__.py" \
  "${UNPACKED_WHEEL}/sgl_kernel/__init__.py"

python -m wheel pack "${UNPACKED_WHEEL}" --dest-dir "${OUTPUT_DIR}"
sha256sum \
  "${BASE_WHEEL}" \
  "${COMMON_OPS}" \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/top_k.py" \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/kvcacheio.py" \
  "${SOURCE_ROOT}/sgl-kernel/python/sgl_kernel/__init__.py" \
  "${OUTPUT_DIR}"/*.whl \
  >"${OUTPUT_DIR}/PROVENANCE_SHA256SUMS"
