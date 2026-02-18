from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "prepare_mask4",
]


# ────────────────────────────────────────────────────────────────
# helper: face-mask tensor @ current latent resolution
# ────────────────────────────────────────────────────────────────

def prepare_mask4(pipeline, latents: torch.Tensor, suffix) -> torch.Tensor:
    """Return `(1,1,H,W)` tensor mask matching *latents* spatial size."""

    # Use high-res mask for reference if available
    if suffix == "_ref" and hasattr(pipeline, f"_face_mask_highres{suffix}"):
        mask_np_highres = getattr(pipeline, f"_face_mask_highres{suffix}")
        m = torch.from_numpy(mask_np_highres).to(device=latents.device, dtype=torch.float32)[None, None]
        # Use bicubic for smoother downsampling
        m = F.interpolate(m, size=latents.shape[-2:], mode="bicubic", align_corners=False)
        m = (m > 0.5).to(dtype=latents.dtype)  # Re-binarize
        return m

    # pick which numpy mask to use
    mask_attr = f"_face_mask{suffix}"
    mask_np = getattr(pipeline, mask_attr)

    ### V2 ###
    is_np = isinstance(mask_np, np.ndarray)
    m = (
        torch.from_numpy(mask_np).to(device=latents.device, dtype=latents.dtype)[None, None]
        if is_np else getattr(pipeline, mask_attr)[:, None].to(dtype=latents.dtype)
    )
    ### V2 ###

    if m.shape[-2:] != latents.shape[-2:]:
        m = F.interpolate(m, size=latents.shape[-2:], mode="nearest")
    return m
