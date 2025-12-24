# SPDX-License-Identifier: Apache-2.0
"""Utilities for resolving TurboDiffusion assets (VAE, text encoder, DiT)."""

import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Known public TurboDiffusion assets (paths are HF repo ids).
WAN_VAE_URL = "Wan-AI/Wan2.1-T2V-1.3B"  # contains Wan2.1_VAE.pth
UMT5_URL = "Wan-AI/Wan2.1-T2V-1.3B"  # contains models_t5_umt5-xxl-enc-bf16.pth


@dataclass
class TurboAssetPaths:
    vae_path: str
    text_encoder_path: str
    dit_path: Optional[str] = None


def _maybe_download(repo_id: str, filename: str, cache_dir: Optional[str]) -> str:
    logger.info("Downloading %s from %s...", filename, repo_id)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir)
    logger.info("Downloaded to %s", path)
    return path


def ensure_turbo_assets(
    vae_path: Optional[str],
    text_path: Optional[str],
    cache_dir: Optional[str],
    allow_download: bool,
) -> TurboAssetPaths:
    """
    Resolve VAE and umT5 paths. If missing and allow_download is True, fetch from HF.
    """
    resolved_vae = vae_path
    resolved_text = text_path

    if resolved_vae is None or not os.path.exists(resolved_vae):
        if not allow_download:
            raise FileNotFoundError(
                "VAE path missing. Provide --vae-path or enable --turbo-auto-download "
                f"to fetch Wan2.1_VAE.pth from {WAN_VAE_URL}."
            )
        resolved_vae = _maybe_download(
            repo_id=WAN_VAE_URL, filename="Wan2.1_VAE.pth", cache_dir=cache_dir
        )

    if resolved_text is None or not os.path.exists(resolved_text):
        if not allow_download:
            raise FileNotFoundError(
                "umT5 text encoder path missing. Provide --turbo-text-encoder-path or enable "
                f"--turbo-auto-download to fetch models_t5_umt5-xxl-enc-bf16.pth from {UMT5_URL}."
            )
        resolved_text = _maybe_download(
            repo_id=UMT5_URL,
            filename="models_t5_umt5-xxl-enc-bf16.pth",
            cache_dir=cache_dir,
        )

    return TurboAssetPaths(vae_path=resolved_vae, text_encoder_path=resolved_text)


# Supported TurboDiT identifiers
TURBO_DIT_MODELS = {
    "turbowan2.1-t2v-1.3b-480p": (
        "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
        "TurboWan2.1-T2V-1.3B-480P.pth",
    ),
    "turbowan2.1-t2v-14b-480p": (
        "TurboDiffusion/TurboWan2.1-T2V-14B-480P",
        "TurboWan2.1-T2V-14B-480P.pth",
    ),
    "turbowan2.1-t2v-14b-720p": (
        "TurboDiffusion/TurboWan2.1-T2V-14B-720P",
        "TurboWan2.1-T2V-14B-720P.pth",
    ),
    # I2V requires high/low; not wired yet
}


def download_turbo_dit(model_key: str, cache_dir: Optional[str]) -> str:
    """
    Explicitly download a TurboDiT checkpoint by key.
    """
    key = model_key.lower()
    if key not in TURBO_DIT_MODELS:
        raise ValueError(
            f"Unknown TurboDiffusion model key '{model_key}'. "
            f"Supported: {list(TURBO_DIT_MODELS.keys())}"
        )
    repo_id, filename = TURBO_DIT_MODELS[key]
    logger.info(
        "Downloading TurboDiT %s (%s) from %s. This is a large file; ensure bandwidth and space.",
        model_key,
        filename,
        repo_id,
    )
    return _maybe_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
