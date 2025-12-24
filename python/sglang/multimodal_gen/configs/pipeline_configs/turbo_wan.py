# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class TurboWanPipelineConfig(PipelineConfig):
    """TurboDiffusion Wan pipeline configuration."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # Turbo-specific runtime parameters
    num_steps: int = 4
    sigma_max: float = 80.0
    attention_type: str = "sagesla"
    sla_topk: float = 0.1
    quant_linear: bool = False
    default_norm: bool = False

    # Wan-specific defaults
    flow_shift: float | None = None

    def __post_init__(self):
        # TurboDiffusion handles autocast internally; keep autocast enabled here.
        self.enable_autocast = True
        # These modules are handled by the TurboDiffusion loader, not the diffusers component loader.
        self.text_encoder_configs = tuple()
        self.text_encoder_precisions = tuple()
        self.image_encoder_config = None
        self.image_encoder_precision = "fp32"
        self.text_encoder_extra_args = []
        self.preprocess_text_funcs = tuple()
        self.postprocess_text_funcs = tuple()


@dataclass
class TurboWanI2VPipelineConfig(TurboWanPipelineConfig):
    """Configuration for the Wan 2.2 I2V Turbo path."""

    task_type: ModelTaskType = ModelTaskType.I2V
    boundary: float = 0.9
    sigma_max: float = 200.0
    # Optional ODE sampling toggle used by the TurboDiffusion script
    ode_sampling: bool = False    """Thin wrapper that drives the TurboDiffusion inference path."""
