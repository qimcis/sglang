from typing import Any, Dict, Literal, Optional

import torch


def kv_page_protection_preflight_supported() -> bool:
    """Whether this package has a loadable preflight image for the active GPU."""
    return bool(torch.ops.sgl_kernel.kv_page_protection_preflight_supported())


def kv_page_protection_preflight(
    protection: Dict[str, Any],
    *,
    phase: Literal["attention", "pre_indexer"] = "attention",
    seqlens: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
) -> None:
    """Validate and fail-close mutable per-forward paged-attention metadata."""
    if phase == "attention":
        validated_epochs = protection["validated_epochs"]
        cache_validated_epochs = True
    elif phase == "pre_indexer":
        validated_epochs = protection["pre_indexer_validated_epochs"]
        cache_validated_epochs = protection.get("pre_indexer_cache_by_request", True)
    else:
        raise ValueError(f"unknown KV page protection phase: {phase}")
    torch.ops.sgl_kernel.kv_page_protection_preflight.default(
        protection["request_indices"],
        protection["seqlens"] if seqlens is None else seqlens,
        protection["page_table"] if page_table is None else page_table,
        protection.get("page_table_2"),
        protection.get("page_table_page_offset", 0),
        protection.get("page_table_2_page_offset", 0),
        protection.get("page_table_expected_mapping_offset", 0),
        protection.get("page_table_2_expected_mapping_offset", 0),
        protection.get("page_table_2_window_size", 0),
        protection["page_size"] if page_size is None else page_size,
        cache_validated_epochs,
        protection["actual_tags"],
        protection["actual_generations"],
        protection["actual_transfer_tags"],
        protection["expected_physical_pages"],
        protection["expected_mapping_stride"],
        protection["expected_mapping_namespace_stride"],
        protection["expected_tags"],
        protection["expected_generations"],
        protection["expected_transfer_tags"],
        protection["request_epochs"],
        validated_epochs,
        protection["status"],
    )
