import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import is_ltx23_native_variant
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2AVDenoisingStage(LTX2DenoisingStage):
    """
    Thin AV layer that adds audio trajectory gathering and final unpacking on top of
    the LTX-2 denoising semantics.
    """

    def __init__(self, transformer, scheduler, vae=None, audio_vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.audio_vae = audio_vae

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        trajectory_audio_latents: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Finalize AV requests by gathering audio latents and unpacking both streams."""
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )
        latents = self._truncate_sp_padded_token_latents(batch, latents)

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        if trajectory_audio_latents:
            trajectory_audio_tensor = torch.stack(trajectory_audio_latents, dim=1)
            batch.trajectory_audio_latents = trajectory_audio_tensor.cpu()

        audio_latents = batch.audio_latents
        if batch.did_sp_shard_audio_latents and isinstance(audio_latents, torch.Tensor):
            audio_latents = server_args.pipeline_config.gather_audio_latents_for_sp(
                audio_latents, batch
            )
            batch.audio_latents = audio_latents

        if self.vae is None or self.audio_vae is None:
            logger.warning(
                "VAE or Audio VAE not found in DenoisingStage. Skipping unpack and denormalize."
            )
            batch.latents = latents
            batch.audio_latents = audio_latents
        else:
            latents, audio_latents = (
                server_args.pipeline_config._unpad_and_unpack_latents(
                    latents, audio_latents, batch, self.vae, self.audio_vae
                )
            )
            batch.latents = latents
            batch.audio_latents = audio_latents

        if isinstance(self.transformer, OffloadableDiTMixin):
            for manager in self.transformer.layerwise_offload_managers:
                manager.release_all()


class LTX2RefinementStage(LTX2AVDenoisingStage):
    """Stage-2 refinement wrapper that runs on pre-noised stage-2 latents."""

    def __init__(
        self, transformer, scheduler, distilled_sigmas, vae=None, audio_vae=None
    ):
        super().__init__(transformer, scheduler, vae, audio_vae)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "timesteps",
            batch.timesteps,
            lambda x: x is None or (V.is_tensor(x) and V.min_dims(1)(x)),
        )
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            lambda x: V.is_tensor(x) or V.list_not_empty(x),
        )
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: (
                (not batch.do_classifier_free_guidance)
                or V.is_tensor(x)
                or V.list_not_empty(x)
            ),
        )
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage2"
        original_clean_latent_background = getattr(
            batch, "ltx2_ti2v_clean_latent_background", None
        )
        is_native_ti2v = (
            is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config)
            and batch.image_path is not None
            and isinstance(batch.latents, torch.Tensor)
        )
        if is_native_ti2v:
            # Official two-stage TI2V keeps the upsampled stage-2 latent as the
            # clean background and only overwrites the conditioned frame tokens.
            batch.ltx2_ti2v_clean_latent_background = batch.latents.detach().clone()
        else:
            batch.ltx2_ti2v_clean_latent_background = None
        if self._should_reset_stage2_generators(server_args):
            self._reset_stage2_generators(batch)
        noise_scale = float(self.distilled_sigmas[0].item())
        if is_native_ti2v:
            prepared_latents, denoise_mask, _ = self._prepare_ltx2_ti2v_clean_state(
                latents=batch.latents,
                image_latent=batch.image_latent,
                num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                zero_clean_latent=True,
                clean_latent_background=batch.ltx2_ti2v_clean_latent_background,
            )
            video_noise = self._randn_like_with_batch_generators(
                prepared_latents, batch
            )
            scaled_mask = (
                denoise_mask.to(device=prepared_latents.device, dtype=torch.float32)
                * noise_scale
            )
            batch.latents = (
                video_noise * scaled_mask + prepared_latents * (1 - scaled_mask)
            ).to(prepared_latents.dtype)
        else:
            video_noise = self._randn_like_with_batch_generators(batch.latents, batch)
            batch.latents = (
                video_noise * noise_scale + batch.latents * (1 - noise_scale)
            ).to(batch.latents.dtype)

        if isinstance(batch.audio_latents, torch.Tensor):
            audio_noise = self._randn_like_with_batch_generators(
                batch.audio_latents, batch
            )
            audio_scaled_mask = (
                torch.ones_like(batch.audio_latents[..., :1], dtype=torch.float32)
                * noise_scale
            )
            batch.audio_latents = (
                audio_noise * audio_scaled_mask
                + batch.audio_latents * (1 - audio_scaled_mask)
            ).to(batch.audio_latents.dtype)
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            batch.latents = batch.latents.to(
                device=batch.latents.device, dtype=torch.float32
            )
            if isinstance(batch.audio_latents, torch.Tensor):
                batch.audio_latents = batch.audio_latents.to(
                    device=batch.audio_latents.device, dtype=torch.float32
                )

        batch.image_latent = None
        batch.ltx2_num_image_tokens = 0

        original_scheduler = self.scheduler
        original_batch_timesteps = batch.timesteps
        original_batch_sigmas = batch.sigmas
        original_batch_num_inference_steps = batch.num_inference_steps
        try:
            num_refinement_steps = len(distilled_sigmas) - 1
            self.scheduler.sigmas = distilled_sigmas
            self.scheduler.timesteps = (
                distilled_sigmas[:num_refinement_steps] * 1000
            ).to(device=batch.latents.device)
            self.scheduler._step_index = None
            self.scheduler._begin_index = None
            self.scheduler.num_inference_steps = num_refinement_steps

            batch.timesteps = self.scheduler.timesteps
            batch.sigmas = distilled_sigmas.detach().cpu().tolist()
            batch.num_inference_steps = num_refinement_steps
            batch = super().forward(batch, server_args)
        finally:
            self.scheduler.sigmas = original_sigmas
            self.scheduler.timesteps = original_timesteps
            if had_num_inference_steps or original_num_inference_steps is not None:
                self.scheduler.num_inference_steps = original_num_inference_steps
            elif hasattr(self.scheduler, "num_inference_steps"):
                delattr(self.scheduler, "num_inference_steps")
            batch.timesteps = original_batch_timesteps
            batch.sigmas = original_batch_sigmas
            batch.num_inference_steps = original_batch_num_inference_steps
            batch.do_classifier_free_guidance = original_do_cfg
            batch.ltx2_ti2v_clean_latent_background = original_clean_latent_background

        return batch
