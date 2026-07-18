#!/usr/bin/env bash

set -Eeuo pipefail

: "${SOURCE_ROOT:?Set SOURCE_ROOT to the isolated SGLang source tree}"
: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to a unique /cache artifact directory}"

readonly WHEEL_DIR="${ARTIFACT_ROOT}/wheels"
readonly STATUS_FILE="${ARTIFACT_ROOT}/build.status"
readonly KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:-${SOURCE_ROOT}/sgl-kernel/build}"

mkdir -p "${WHEEL_DIR}"
rm -f "${STATUS_FILE}"

write_status() {
  local rc=$?
  printf '%s\n' "${rc}" >"${STATUS_FILE}"
}
trap write_status EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${SOURCE_ROOT}/sgl-kernel"
rm -rf dist
if [[ "${CLEAN_KERNEL_BUILD:-1}" == 1 ]]; then
  rm -rf "${KERNEL_BUILD_DIR}"
fi
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}" \
  CMAKE_ARGS="${CMAKE_ARGS:--DSGL_KERNEL_COMPILE_THREADS=2 -DENABLE_BELOW_SM90=OFF}" \
  python -m build \
    --wheel \
    --no-isolation \
    --config-setting "build-dir=${KERNEL_BUILD_DIR}" \
    --outdir "${WHEEL_DIR}"

cd "${SOURCE_ROOT}/python"
python -m pip install 'setuptools-rust>=1.10' 'setuptools-scm>=8.0'
rm -rf build dist sglang.egg-info
SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_BUILD_VERSION:-0.0.0.dev1+g0e8b11e45.d20260715}" \
  python -m build \
    --wheel \
    --no-isolation \
    --outdir "${WHEEL_DIR}"

sha256sum "${WHEEL_DIR}"/*.whl >"${ARTIFACT_ROOT}/SHA256SUMS"
