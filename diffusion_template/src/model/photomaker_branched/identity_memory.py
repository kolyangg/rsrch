from __future__ import annotations

from typing import Sequence

from PIL import Image
import torch
from torch import nn


def bbox_normalized_reference(
    image: Image.Image,
    bbox: Sequence[float],
    *,
    padding: float = 0.10,
) -> Image.Image:
    """Return a square, bbox-centered reference crop for BA identity memory."""
    if bbox is None or len(bbox) < 4:
        raise ValueError("bbox_normalized identity memory requires a reference bbox")
    image = image.convert("RGB")
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid reference bbox: {bbox}")

    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * max(0.0, float(padding)))
    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    right = int(round(cx + side * 0.5))
    bottom = int(round(cy + side * 0.5))
    if right <= left or bottom <= top:
        raise ValueError(f"Degenerate normalized reference crop: {bbox}")

    # PIL pads out-of-image crop coordinates with black, keeping the face centered.
    return image.crop((left, top, right, bottom))


def reference_bbox_to_clip_patch_mask(
    image,
    bbox: Sequence[float],
    *,
    processed_height: int,
    processed_width: int,
    patch_size: int,
    padding: float = 0.0,
) -> torch.Tensor:
    """Map an original-image bbox through CLIP's resize/center-crop to patch tokens."""
    if bbox is None or len(bbox) < 4:
        raise ValueError("face-patch identity memory requires a reference bbox")
    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    elif torch.is_tensor(image) and image.ndim >= 2:
        image_height, image_width = image.shape[-2:]
    else:
        raise TypeError(f"Unsupported reference image type: {type(image)!r}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid reference image size: {(image_width, image_height)}")

    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid reference bbox: {bbox}")
    pad = max(0.0, float(padding))
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy

    # CLIPImageProcessor resizes the shorter edge, then takes an integer center crop.
    scale = max(processed_width / image_width, processed_height / image_height)
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    crop_left = max(0, (resized_width - processed_width) // 2)
    crop_top = max(0, (resized_height - processed_height) // 2)
    x0, x1 = x0 * scale - crop_left, x1 * scale - crop_left
    y0, y1 = y0 * scale - crop_top, y1 * scale - crop_top

    grid_height = processed_height // int(patch_size)
    grid_width = processed_width // int(patch_size)
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(
            f"Patch size {patch_size} is incompatible with processed size "
            f"{(processed_height, processed_width)}"
        )
    ys = (torch.arange(grid_height, dtype=torch.float32) + 0.5) * patch_size
    xs = (torch.arange(grid_width, dtype=torch.float32) + 0.5) * patch_size
    mask = (
        (ys[:, None] >= y0)
        & (ys[:, None] < y1)
        & (xs[None, :] >= x0)
        & (xs[None, :] < x1)
    )
    if not bool(mask.any()):
        # A heavily center-cropped or tiny bbox still gets its nearest patch.
        cx = min(max((x0 + x1) * 0.5, 0.0), float(processed_width))
        cy = min(max((y0 + y1) * 0.5, 0.0), float(processed_height))
        col = min(grid_width - 1, max(0, int(cx // patch_size)))
        row = min(grid_height - 1, max(0, int(cy // patch_size)))
        mask[row, col] = True
    return mask.reshape(-1)


class FacePatchIdentityResampler(nn.Module):
    """Produce identity-specific memory tokens from hard-bbox CLIP patch features."""

    def __init__(
        self,
        *,
        num_tokens: int = 8,
        patch_dim: int = 1024,
        identity_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 2048,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_tokens = int(num_tokens)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.identity_norm = nn.LayerNorm(identity_dim)
        self.query_proj = nn.Linear(identity_dim, self.num_tokens * hidden_dim)
        self.query_bias = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        patch_features: torch.Tensor,
        identity_embeds: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        if patch_features.ndim != 3:
            raise ValueError(f"Expected patch features [B,P,D], got {tuple(patch_features.shape)}")
        if identity_embeds.ndim != 2:
            raise ValueError(f"Expected identity embeddings [B,D], got {tuple(identity_embeds.shape)}")
        if patch_mask.shape != patch_features.shape[:2]:
            raise ValueError(
                f"Patch mask {tuple(patch_mask.shape)} does not match "
                f"features {tuple(patch_features.shape[:2])}"
            )
        if not bool(patch_mask.any(dim=1).all()):
            raise ValueError("Each identity sample must retain at least one reference patch")

        patches = self.patch_proj(self.patch_norm(patch_features))
        queries = self.query_proj(self.identity_norm(identity_embeds)).view(
            identity_embeds.shape[0], self.num_tokens, -1
        )
        queries = queries + self.query_bias
        attended, _ = self.cross_attn(
            queries,
            patches,
            patches,
            key_padding_mask=~patch_mask.bool(),
            need_weights=False,
        )
        hidden = queries + attended
        hidden = hidden + self.ff(hidden)
        return self.output_norm(self.output_proj(hidden))
