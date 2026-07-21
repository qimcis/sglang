from dataclasses import dataclass, field

import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
    shard_rotary_emb_for_sp,
)


@dataclass
class GlmImagePipelineConfig(SpatialImagePipelineConfig):
    """Configuration for the GlmImage pipeline."""

    vae_precision: str = "bf16"

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.TI2I

    vae_tiling: bool = False

    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )

    dit_config: DiTConfig = field(default_factory=GlmImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)

    # GLM-Image uses T5 text encoder; base default is EncoderConfig() which lacks
    # parallel_folding and causes AttributeError + fallback to native T5 with missing weights.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )

    enable_autocast: bool = False

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def supports_dynamic_batching(self):
        """Allow batching for text-only GLM-Image requests."""
        return True

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        height = batch.height // self.vae_scale_factor
        width = batch.width // self.vae_scale_factor
        hidden_states = torch.empty(1, 1, height, width, device=device, dtype=dtype)
        cos, sin = rotary_emb(hidden_states)
        cos = shard_rotary_emb_for_sp(cos)
        sin = shard_rotary_emb_for_sp(sin)
        return cos, sin

    @staticmethod
    def _get_kv_caches(batch):
        if batch.kv_caches is not None or not batch.glm_image_kv_cache_keys:
            return batch.kv_caches

        from sglang.multimodal_gen.runtime.models.dits.glm_image import (
            GlmImageKVCache,
        )

        if len(batch.glm_image_kv_cache_keys) != len(batch.glm_image_kv_cache_values):
            raise ValueError("GLM-Image transferred KV cache is incomplete")
        kv_caches = GlmImageKVCache(len(batch.glm_image_kv_cache_keys))
        for layer_cache, key, value in zip(
            kv_caches.caches,
            batch.glm_image_kv_cache_keys,
            batch.glm_image_kv_cache_values,
        ):
            layer_cache.k_cache = key
            layer_cache.v_cache = value
        batch.kv_caches = kv_caches
        return kv_caches

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_cond,
            "attention_mask": (
                batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None
            ),
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
            "kv_caches": self._get_kv_caches(batch),
            "kv_caches_mode": "read",
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_uncond,
            "attention_mask": (
                batch.negative_attention_mask[0]
                if batch.negative_attention_mask
                else None
            ),
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
            "kv_caches": self._get_kv_caches(batch),
            "kv_caches_mode": "skip",
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        latents_mean = (
            torch.tensor(self.vae_config.latents_mean)
            .view(1, self.vae_config.latent_channels, 1, 1)
            .to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae_config.latents_std)
            .view(1, self.vae_config.latent_channels, 1, 1)
            .to(device, dtype)
        )
        return 1.0 / latents_std, latents_mean

    def post_denoising_loop(self, latents, batch):
        if getattr(batch, "kv_caches", None) is not None:
            batch.kv_caches.clear()
        batch.glm_image_kv_cache_keys = []
        batch.glm_image_kv_cache_values = []
        return latents.bfloat16()

    def post_decoding(self, frames, server_args):
        return self.image_processor.postprocess(frames, output_type="latent")
