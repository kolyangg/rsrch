from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


def prepare_reference_latents(
    pipeline,
    reference_image: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Encode reference image to latents.
    """
    device = pipeline.device
    vae = pipeline.vae

    # Convert PIL to tensor if needed
    if isinstance(reference_image, Image.Image):
        reference_image = pipeline.feature_extractor(
            reference_image, return_tensors="pt"
        ).pixel_values[0]

    # Ensure correct shape
    if reference_image.dim() == 3:
        reference_image = reference_image.unsqueeze(0)

    # Move to VAE device/dtype for encoding
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    reference_image = reference_image.to(device=vae_device, dtype=vae_dtype)

    # Encode
    with torch.no_grad():
        latents = vae.encode(reference_image).latent_dist.sample(generator)
        latents = latents * vae.config.scaling_factor

    # Resize if needed
    target_h = height // pipeline.vae_scale_factor
    target_w = width // pipeline.vae_scale_factor

    if latents.shape[2] != target_h or latents.shape[3] != target_w:
        latents = F.interpolate(
            latents.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )

    # Normalize for stable attention
    latents = (latents - latents.mean()) / latents.std().clamp(min=1e-4)

    return latents.to(device=device, dtype=dtype)
