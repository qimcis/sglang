#!/usr/bin/env python3
"""Validate KV-protection packaging definitions without compiling CUDA."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def named_block(text: str, kind: str, name: str) -> str:
    match = re.search(rf'\b{re.escape(kind)}\s+"{re.escape(name)}"\s*\{{', text)
    require(match is not None, f'missing {kind} "{name}"')
    start = match.end() - 1
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f'unclosed {kind} "{name}"')


def docker_stage(text: str, name: str) -> tuple[str, str]:
    stages = list(
        re.finditer(r"(?im)^FROM\s+([^\s]+)\s+AS\s+([a-zA-Z0-9_-]+)\s*$", text)
    )
    for index, match in enumerate(stages):
        if match.group(2) == name:
            end = stages[index + 1].start() if index + 1 < len(stages) else len(text)
            return match.group(1), text[match.end() : end]
    raise ValueError(f'missing Docker stage "{name}"')


def validate_cmake() -> None:
    cmake = (ROOT / "sgl-kernel/CMakeLists.txt").read_text()
    flashmla = (ROOT / "sgl-kernel/cmake/flashmla.cmake").read_text()
    for profile in ("auto", "hopper-sm90", "blackwell-sm100", "blackwell-sm103"):
        require(profile in cmake, f"CMake profile is missing: {profile}")
    require(
        "blackwell-sm103 requires CUDA 13 or newer" in cmake,
        "SM103 profile does not enforce CUDA 13",
    )
    require(
        'SGL_KERNEL_ARCH_PROFILE STREQUAL "hopper-sm90"' in cmake
        and "set(SGL_KERNEL_ENABLE_FA3 ON)" in cmake,
        "Hopper profile does not force-enable FA3",
    )
    require(
        "set(SGL_KERNEL_ENABLE_FA3 OFF)" in cmake
        and "set(ENABLE_BELOW_SM90 OFF)" in cmake,
        "Blackwell profiles do not disable FA3 and below-SM90 images",
    )
    for arch in ("compute_90", "compute_90a", "compute_100a", "compute_103a"):
        require(arch in cmake, f"CMake architecture flag is missing: {arch}")
    require(
        "SGL_KERNEL_ARCH_PROFILE STREQUAL" in flashmla,
        "FlashMLA does not honor architecture profiles",
    )


def validate_dockerfile() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    builder_base, builder = docker_stage(dockerfile, "local_kernel_builder")
    final_base, final = docker_stage(dockerfile, "kv_protection_final")
    require(builder_base == "torch_deps", "local_kernel_builder must follow torch_deps")
    require(
        final_base == "framework_final", "kv_protection_final must use framework_final"
    )
    require("COPY sgl-kernel" in builder, "local builder does not copy sgl-kernel")
    require(
        "pip install scikit-build-core" in builder,
        "local builder does not install its wheel build backend",
    )
    require(
        "-DSGL_KERNEL_ARCH_PROFILE=${SGL_KERNEL_ARCH_PROFILE}" in builder,
        "local wheel build does not pass the architecture profile",
    )
    require(
        "pip wheel --no-build-isolation --no-deps" in builder,
        "local builder does not build a wheel",
    )
    require(
        "COPY --from=local_kernel_builder /wheels" in final,
        "final stage does not copy the local wheel",
    )
    require(
        "pip uninstall -y sglang-kernel" in final,
        "final stage does not remove the public kernel package",
    )
    require(
        "pip install --force-reinstall --no-deps "
        "/tmp/local-sgl-kernel/sglang_kernel-*.whl" in final,
        "final stage does not force-install the local wheel",
    )
    require(
        'm.version("sglang-kernel")' in final and 'u.find_spec("sgl_kernel")' in final,
        "final stage does not verify installed version and module path",
    )
    require(
        "SGL_KERNEL_ARCH_PROFILE=${SGL_KERNEL_ARCH_PROFILE}" in final,
        "final stage does not record the architecture profile",
    )
    require(
        "ai.sglang.kernel.source-revision" in final,
        "final stage does not label the kernel source revision",
    )


def validate_bake() -> None:
    bake = (ROOT / "docker/kv-protection-bake.hcl").read_text()
    common = named_block(bake, "target", "kv-protection")
    require('context    = "."' in common, "bake context must be the repository root")
    require(
        'target     = "kv_protection_final"' in common,
        "bake target must select kv_protection_final",
    )
    require(
        'platforms  = ["linux/amd64"]' in common,
        "bake target must select linux/amd64",
    )
    expected = {
        "hopper": ("12.9.1", "hopper-sm90"),
        "b200": ("12.9.1", "blackwell-sm100"),
        "b300": ("13.0.1", "blackwell-sm103"),
    }
    for target, (cuda, profile) in expected.items():
        body = named_block(bake, "target", target)
        require(
            'inherits = ["kv-protection"]' in body,
            f"{target} does not inherit the common target",
        )
        require(
            f'CUDA_VERSION            = "{cuda}"' in body,
            f"{target} CUDA version does not match {cuda}",
        )
        require(
            f'SGL_KERNEL_ARCH_PROFILE = "{profile}"' in body,
            f"{target} profile does not match {profile}",
        )
        require(
            "${REGISTRY}" in body and "${SOURCE_REVISION}" in body,
            f"{target} tag is not configurable",
        )


def main() -> int:
    try:
        validate_cmake()
        validate_dockerfile()
        validate_bake()
    except ValueError as error:
        print(f"KV packaging validation failed: {error}", file=sys.stderr)
        return 1
    print("KV packaging definitions are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
