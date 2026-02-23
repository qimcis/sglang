# SPDX-License-Identifier: Apache-2.0
"""
Generic pipeline configuration for diffusers backend.

This module provides a minimal pipeline configuration that works with the diffusers backend.
Since diffusers handles its own model loading and configuration, this config is intentionally minimal.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

_TASK_HINT_PATTERNS: tuple[tuple[ModelTaskType, tuple[str, ...]], ...] = (
    (
        ModelTaskType.TI2V,
        (
            r"(^|[^a-z0-9])ti2v([^a-z0-9]|$)",
            r"text[-_ ]image[-_ ]to[-_ ]video",
            r"text[-_ ]and[-_ ]image[-_ ]to[-_ ]video",
        ),
    ),
    (
        ModelTaskType.I2V,
        (
            r"(^|[^a-z0-9])i2v([^a-z0-9]|$)",
            r"img2vid(?:eo)?",
            r"image2video",
            r"image[-_ ]to[-_ ]video",
        ),
    ),
    (
        ModelTaskType.T2V,
        (
            r"(^|[^a-z0-9])t2v([^a-z0-9]|$)",
            r"text2video",
            r"text[-_ ]to[-_ ]video",
        ),
    ),
    (
        ModelTaskType.TI2I,
        (
            r"(^|[^a-z0-9])ti2i([^a-z0-9]|$)",
            r"text[-_ ]image[-_ ]to[-_ ]image",
        ),
    ),
    (
        ModelTaskType.I2I,
        (
            r"(^|[^a-z0-9])i2i([^a-z0-9]|$)",
            r"img2img",
            r"image2image",
            r"image[-_ ]to[-_ ]image",
            r"inpaint(?:ing)?",
            r"pix2pix",
            r"(^|[^a-z0-9])edit([^a-z0-9]|$)",
        ),
    ),
)


@dataclass
class DiffusersGenericPipelineConfig(PipelineConfig):
    """
    Generic pipeline configuration for diffusers backend.

    This is a minimal configuration since the diffusers backend handles most
    configuration internally. It provides sensible defaults for the required fields.
    """

    # default to T2I since it's the most common
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"

    should_use_guidance: bool = True
    embedded_cfg_scale: float = 1.0
    flow_shift: float | None = None
    disable_autocast: bool = True  # let diffusers handle dtype

    # diffusers handles its own loading
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

    # VAE settings
    vae_tiling: bool = False  # diffusers handles this
    vae_slicing: bool = False  # slice VAE decode for lower memory usage
    vae_sp: bool = False

    # Attention backend for diffusers models (e.g., "flash", "_flash_3_hub", "sage", "xformers")
    # See: https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
    diffusers_attention_backend: str | None = None

    # Quantization config for pipeline-level quantization
    # See: https://huggingface.co/docs/diffusers/main/en/quantization/overview
    # Use PipelineQuantizationConfig for component-level control:
    #   from diffusers.quantizers import PipelineQuantizationConfig
    #   quantization_config = PipelineQuantizationConfig(
    #       quant_backend="bitsandbytes_4bit",
    #       quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
    #       components_to_quantize=["transformer", "text_encoder_2"],
    #   )
    quantization_config: Any = None

    @staticmethod
    def _infer_task_type_from_hint_text(hint_text: str) -> ModelTaskType | None:
        normalized = hint_text.lower()
        for task_type, patterns in _TASK_HINT_PATTERNS:
            if any(re.search(pattern, normalized) for pattern in patterns):
                return task_type
        return None

    @staticmethod
    def _infer_task_type_from_local_model_index(
        model_path: str,
    ) -> ModelTaskType | None:
        if not os.path.isdir(model_path):
            return None

        model_index_path = os.path.join(model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            return None

        try:
            with open(model_index_path, encoding="utf-8") as f:
                model_index = json.load(f)
        except Exception:
            return None

        class_name = model_index.get("_class_name")
        if not isinstance(class_name, str):
            return None

        return DiffusersGenericPipelineConfig._infer_task_type_from_hint_text(
            class_name
        )

    @staticmethod
    def infer_task_type_from_model_path(model_path: str | None) -> ModelTaskType | None:
        """Best-effort task inference for diffusers models.

        Resolution order:
        1. Local `model_index.json` class name (most reliable without loading model)
        2. Conservative regex hints from model id/path (fallback only)
        """
        if not model_path:
            return None

        inferred_from_local_index = (
            DiffusersGenericPipelineConfig._infer_task_type_from_local_model_index(
                model_path
            )
        )
        if inferred_from_local_index is not None:
            return inferred_from_local_index

        return DiffusersGenericPipelineConfig._infer_task_type_from_hint_text(
            model_path
        )

    def check_pipeline_config(self) -> None:
        """
        Override to skip most validation since diffusers handles its own config.
        """
        pass

    def adjust_size(self, width, height, image):
        """
        Pass through - diffusers handles size adjustments.
        """
        return width, height

    def adjust_num_frames(self, num_frames):
        """
        Pass through - diffusers handles frame count.
        """
        return num_frames
