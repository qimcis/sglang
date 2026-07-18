set -Eeuo pipefail

: "${COMMON_OPS:?Set COMMON_OPS to the shipped SM90 common_ops library}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for disassembly evidence}"

mkdir -p "${OUTPUT_DIR}"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
(
  cd "${work_dir}"
  cuobjdump --extract-elf all "${COMMON_OPS}" >/dev/null
)
mapfile -t topk_cubins < <(
  for cubin in "${work_dir}"/*sm_90a.cubin; do
    if cuobjdump --dump-elf-symbols "${cubin}" | grep -q 'topk_transform_decode_protected_kernelILb1EE'; then
      printf '%s\n' "${cubin}"
    fi
  done
)
[[ "${#topk_cubins[@]}" -eq 1 ]]
topk_cubin="${topk_cubins[0]}"
cp "${topk_cubin}" "${OUTPUT_DIR}/topk.sm_90a.cubin"

symbols="$(cuobjdump --dump-elf-symbols "${topk_cubin}")"
generic="$(printf '%s\n' "${symbols}" | grep 'topk_transform_decode_protected_kernelILb0EE' | awk '{print $NF}' | sort -u)"
specialized="$(printf '%s\n' "${symbols}" | grep 'topk_transform_decode_protected_kernelILb1EE' | awk '{print $NF}' | sort -u)"
[[ "$(printf '%s\n' "${generic}" | wc -l)" -eq 1 ]]
[[ "$(printf '%s\n' "${specialized}" | wc -l)" -eq 1 ]]

cuobjdump --dump-sass --function "${generic}" "${topk_cubin}" >"${OUTPUT_DIR}/generic.sass"
cuobjdump --dump-sass --function "${specialized}" "${topk_cubin}" >"${OUTPUT_DIR}/page64.sass"

generic_mufu_rcp="$(grep -c 'MUFU.RCP' "${OUTPUT_DIR}/generic.sass")"
page64_mufu_rcp="$(grep -c 'MUFU.RCP' "${OUTPUT_DIR}/page64.sass" || true)"
page64_shift6="$(grep -E '(SHF|SHR).*0x6' "${OUTPUT_DIR}/page64.sass" | wc -l)"

printf 'generic_mufu_rcp=%s\npage64_mufu_rcp=%s\npage64_shift6=%s\n' \
  "${generic_mufu_rcp}" "${page64_mufu_rcp}" "${page64_shift6}" \
  >"${OUTPUT_DIR}/SUMMARY.txt"
printf '%s\n' \
  "generic_mufu_rcp=${generic_mufu_rcp}" \
  "page64_mufu_rcp=${page64_mufu_rcp}" \
  "page64_shift6=${page64_shift6}"

[[ "${generic_mufu_rcp}" -gt 0 ]]
[[ "${page64_mufu_rcp}" -eq 0 ]]
[[ "${page64_shift6}" -gt 0 ]]
