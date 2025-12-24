# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.turbo_assets import ensure_turbo_assets

logger = init_logger(__name__)


class TurboWanRunner:
    def __init__(self, model_path: str, server_args: ServerArgs) -> None:
        self.model_path = model_path
        self.server_args = server_args

        self._imports: dict[str, Any] = {}
        self._dit_model = None
        self._tokenizer = None

        lower = model_path.lower()
        if "2.2" in lower or "i2v" in lower:
            self.model_name = "Wan2.2-A14B"
            self.is_i2v = True
        elif "14b" in lower:
            self.model_name = "Wan2.1-14B"
            self.is_i2v = False
        else:
            self.model_name = "Wan2.1-1.3B"
            self.is_i2v = False

    def _lazy_imports(self) -> None:
        if self._imports:
            return
        try:
            from turbodiffusion.inference.modify_model import (
                create_model,
                tensor_kwargs,
            )
            from turbodiffusion.rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
            from turbodiffusion.rcm.utils.umt5 import (
                clear_umt5_memory,
                get_umt5_embedding,
            )
        except Exception as e:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "TurboDiffusion is not installed. Install the optional "
                "extra `sglang[diffusion,turbodiffusion]` or add TurboDiffusion "
                "to PYTHONPATH."
            ) from e

        self._imports = {
            "create_model": create_model,
            "tensor_kwargs": dict(tensor_kwargs),
            "Wan2pt1VAEInterface": Wan2pt1VAEInterface,
            "clear_umt5_memory": clear_umt5_memory,
            "get_umt5_embedding": get_umt5_embedding,
        }

    def _resolve_paths(self) -> tuple[str, str, str]:
        """Resolve the checkpoint paths we need."""
        dit_path = (
            self.server_args.turbo_dit_path
            or (self.server_args.turbo_low_noise_dit_path if self.is_i2v else None)
            or self.model_path
        )
        assets = ensure_turbo_assets(
            vae_path=self.server_args.vae_path,
            text_path=self.server_args.turbo_text_encoder_path,
            cache_dir=self.server_args.turbo_cache_dir,
            allow_download=self.server_args.turbo_auto_download,
        )

        if not dit_path:
            raise ValueError(
                "Missing Turbo DiT checkpoint (--turbo-dit-path or --model-path)"
            )

        return dit_path, assets.vae_path, assets.text_encoder_path

    def _build_model(self) -> None:
        if self._dit_model is not None:
            return

        self._lazy_imports()
        dit_path, _, _ = self._resolve_paths()

        pipeline_cfg = self.server_args.pipeline_config
        args = SimpleNamespace(
            model=self.model_name,
            attention_type=pipeline_cfg.attention_type,
            sla_topk=pipeline_cfg.sla_topk,
            quant_linear=pipeline_cfg.quant_linear,
            default_norm=pipeline_cfg.default_norm,
        )
        logger.info(
            "Loading TurboDiffusion DiT from %s (model=%s, attn=%s, topk=%s, quant_linear=%s, default_norm=%s)",
            dit_path,
            self.model_name,
            args.attention_type,
            args.sla_topk,
            args.quant_linear,
            args.default_norm,
        )
        self._dit_model = self._imports["create_model"](dit_path=dit_path, args=args)

    def _build_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        self._lazy_imports()
        _, vae_path, _ = self._resolve_paths()
        logger.info("Loading TurboDiffusion VAE from %s", vae_path)
        self._tokenizer = self._imports["Wan2pt1VAEInterface"](vae_pth=vae_path)

    def _sample_trigflow(self, batch: Req) -> torch.Tensor:
        """Run the TurboDiffusion sampling loop for T2V."""
        if self.is_i2v:
            raise RuntimeError(
                "TurboDiffusion I2V path is not implemented in this integration yet."
            )

        self._build_model()
        self._build_tokenizer()
        self._lazy_imports()

        tensor_kwargs = self._imports["tensor_kwargs"]
        get_umt5_embedding = self._imports["get_umt5_embedding"]
        clear_umt5_memory = self._imports["clear_umt5_memory"]

        prompt = batch.prompt[0] if isinstance(batch.prompt, list) else batch.prompt
        if prompt is None:
            raise ValueError("TurboDiffusion requires a prompt.")

        pipeline_cfg = self.server_args.pipeline_config
        num_steps = max(1, int(getattr(pipeline_cfg, "num_steps", 4)))
        sigma_max = pipeline_cfg.sigma_max or 80.0

        width = batch.width[0] if isinstance(batch.width, list) else batch.width or 832
        height = (
            batch.height[0] if isinstance(batch.height, list) else batch.height or 480
        )
        num_frames = (
            batch.num_frames[0]
            if isinstance(batch.num_frames, list)
            else batch.num_frames
        )
        num_frames = num_frames or 81

        supported = getattr(batch, "supported_resolutions", None)
        if supported and (width, height) not in supported:
            supported_str = ", ".join([f"{w}x{h}" for w, h in supported])
            raise ValueError(
                f"Unsupported resolution {width}x{height} for TurboDiffusion. "
                f"Supported: {supported_str}"
            )

        seed = batch.seed if batch.seed is not None else 0
        num_samples = batch.num_outputs_per_prompt or 1

        logger.info(
            "TurboDiffusion T2V | prompt='%s' | size=%sx%s | frames=%s | steps=%s | sigma_max=%.2f | seed=%s",
            prompt,
            width,
            height,
            num_frames,
            num_steps,
            sigma_max,
            seed,
        )

        with torch.no_grad():
            text_emb = get_umt5_embedding(
                checkpoint_path=self._resolve_paths()[2], prompts=prompt
            ).to(**tensor_kwargs)
        clear_umt5_memory()

        net = self._dit_model
        net = net.to(**tensor_kwargs).eval()
        tokenizer = self._tokenizer

        w = int(width)
        h = int(height)
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(seed)

        init_noise = torch.randn(
            num_samples,
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
        t_steps = torch.tensor(
            [math.atan(sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        total_steps = t_steps.shape[0] - 1

        condition = {
            "crossattn_emb": text_emb.to(**tensor_kwargs).repeat(num_samples, 1, 1),
        }

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

        samples = x.float()
        with torch.no_grad():
            video = tokenizer.decode(samples)

        # Normalize to [0, 1]
        video = (1.0 + video.clamp(-1, 1)) / 2.0
        return video.cpu().float()

    def __call__(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output = self._sample_trigflow(batch)
        return OutputBatch(
            output=output,
            trajectory_timesteps=None,
            trajectory_latents=None,
            trajectory_decoded=None,
            timings=batch.timings,
        )


class TurboWanPipeline(ComposedPipelineBase):
    """Pipeline wrapper that delegates execution to TurboDiffusion."""

    pipeline_name = "TurboWanPipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        runner = TurboWanRunner(model_path=self.model_path, server_args=server_args)
        return {"runner": runner}

    def create_pipeline_stages(self, server_args: ServerArgs):
        # TurboDiffusion runs end-to-end inside the runner; no staged execution needed.
        return

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:  # type: ignore[override]
        runner: TurboWanRunner = self.get_module("runner")
        return runner(batch, server_args)


EntryClass = TurboWanPipeline
