from __future__ import annotations

from typing import Sequence

from PIL import Image


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
