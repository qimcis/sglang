# SPDX-License-Identifier: Apache-2.0
"""Memory metadata shared between disaggregated diffusion roles."""

from __future__ import annotations

from typing import Any

DISAGG_MEMORY_ROLE_FIELD = "_disagg_memory_role"
DISAGG_PEAK_MEMORY_MB_FIELD = "_disagg_peak_memory_mb"
DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD = "_disagg_peak_allocated_memory_mb"
DISAGG_IS_OOM_FIELD = "_disagg_is_oom"


def attach_object_memory(
    obj: Any,
    *,
    peak_memory_mb: float,
    peak_allocated_memory_mb: float,
    is_oom: bool = False,
) -> None:
    setattr(obj, DISAGG_PEAK_MEMORY_MB_FIELD, float(peak_memory_mb or 0.0))
    setattr(
        obj,
        DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD,
        float(peak_allocated_memory_mb or 0.0),
    )
    setattr(obj, DISAGG_IS_OOM_FIELD, bool(is_oom))


def role_memory_from_object(
    obj: Any, *, role: str, error: str | None = None
) -> dict[str, Any] | None:
    peak_memory_mb = _memory_attr(obj, "peak_memory_mb", DISAGG_PEAK_MEMORY_MB_FIELD)
    peak_allocated_memory_mb = _memory_attr(
        obj,
        "peak_allocated_memory_mb",
        DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD,
    )
    is_oom = bool(
        getattr(obj, "is_oom", False) or getattr(obj, DISAGG_IS_OOM_FIELD, False)
    )
    if not (peak_memory_mb or peak_allocated_memory_mb or is_oom or error):
        return None
    return {
        "role": role,
        "peak_memory_mb": float(peak_memory_mb or 0.0),
        "peak_allocated_memory_mb": float(peak_allocated_memory_mb or 0.0),
        "is_oom": is_oom,
        "error": error,
    }


def attach_role_memory_fields(
    scalar_fields: dict[str, Any], obj: Any, *, role: str
) -> None:
    memory = role_memory_from_object(obj, role=role)
    if memory is None:
        return
    scalar_fields[DISAGG_MEMORY_ROLE_FIELD] = memory["role"]
    scalar_fields[DISAGG_PEAK_MEMORY_MB_FIELD] = memory["peak_memory_mb"]
    scalar_fields[DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD] = memory[
        "peak_allocated_memory_mb"
    ]
    scalar_fields[DISAGG_IS_OOM_FIELD] = memory["is_oom"]


def role_memory_from_scalar_fields(
    scalar_fields: dict[str, Any] | None, *, default_role: str | None = None
) -> dict[str, Any] | None:
    if not scalar_fields:
        return None
    has_memory_fields = any(
        key in scalar_fields
        for key in (
            DISAGG_PEAK_MEMORY_MB_FIELD,
            DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD,
            DISAGG_IS_OOM_FIELD,
        )
    )
    if not has_memory_fields:
        return None
    role = scalar_fields.get(DISAGG_MEMORY_ROLE_FIELD, default_role)
    peak_memory_mb = float(scalar_fields.get(DISAGG_PEAK_MEMORY_MB_FIELD) or 0.0)
    peak_allocated_memory_mb = float(
        scalar_fields.get(DISAGG_PEAK_ALLOCATED_MEMORY_MB_FIELD) or 0.0
    )
    is_oom = bool(scalar_fields.get(DISAGG_IS_OOM_FIELD, False))
    if not (role or peak_memory_mb or peak_allocated_memory_mb or is_oom):
        return None
    return {
        "role": role,
        "peak_memory_mb": peak_memory_mb,
        "peak_allocated_memory_mb": peak_allocated_memory_mb,
        "is_oom": is_oom,
        "error": scalar_fields.get("error") or scalar_fields.get("_disagg_error"),
    }


def aggregate_role_memory(role_memory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    peak_memory_mb = 0.0
    peak_allocated_memory_mb = 0.0
    is_oom = False
    for memory in role_memory.values():
        peak_memory_mb = max(peak_memory_mb, float(memory.get("peak_memory_mb") or 0.0))
        peak_allocated_memory_mb = max(
            peak_allocated_memory_mb,
            float(memory.get("peak_allocated_memory_mb") or 0.0),
        )
        is_oom = is_oom or bool(memory.get("is_oom", False))
    return {
        "peak_memory_mb": peak_memory_mb,
        "peak_allocated_memory_mb": peak_allocated_memory_mb,
        "is_oom": is_oom,
    }


def _memory_attr(obj: Any, public_name: str, disagg_name: str) -> float:
    return float(
        getattr(obj, public_name, 0.0) or getattr(obj, disagg_name, 0.0) or 0.0
    )
