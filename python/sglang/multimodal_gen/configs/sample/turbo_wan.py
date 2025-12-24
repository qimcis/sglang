# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class TurboWanSamplingParams(SamplingParams):
    """Sampling params tailored for TurboDiffusion Wan checkpoints."""

    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising/scheduler
    guidance_scale: float = 3.0
    num_inference_steps: int = 4
    sigma_max: float | None = None  # overridden by pipeline_config
    attention_type: str | None = None
    sla_topk: float | None = None
    quant_linear: bool | None = None
    default_norm: bool | None = None

    # Wan defaults
    negative_prompt: str | None = None

    # Supported resolutions to validate requests
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),  # 16:9
            (480, 832),  # 9:16
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
        ]
    )
