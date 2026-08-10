# 10 Aug 2026 - E13C-DATA-03: Ported sealed reference crop/canvas hygiene as a
# dataset-only helper; model and pipeline remain dataset agnostic.
"""Deterministic reference-image policies shared by Cosmic experiments."""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image


BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)


def valid_bbox(
    bbox: Sequence[float] | None,
    image_size: tuple[int, int],
) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    width, height = image_size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def _square_face_crop(
    image: Image.Image,
    bbox: Sequence[float],
    margin: float,
) -> tuple[Image.Image, list[float]]:
    if margin < 0:
        raise ValueError(f"reference crop margin must be non-negative, got {margin}")
    if not valid_bbox(bbox, image.size):
        raise ValueError(f"invalid reference bbox {bbox!r} for image size {image.size}")

    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    face_side = max(x1 - x0, y1 - y0)
    crop_side = min(
        width,
        height,
        max(1, int(math.ceil(face_side * (1.0 + 2.0 * margin)))),
    )
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    crop_x0 = min(max(int(round(center_x - crop_side / 2.0)), 0), width - crop_side)
    crop_y0 = min(max(int(round(center_y - crop_side / 2.0)), 0), height - crop_side)
    cropped = image.crop(
        (crop_x0, crop_y0, crop_x0 + crop_side, crop_y0 + crop_side)
    )
    transformed_bbox = [
        x0 - crop_x0,
        y0 - crop_y0,
        x1 - crop_x0,
        y1 - crop_y0,
    ]
    if not valid_bbox(transformed_bbox, cropped.size):
        raise ValueError(
            f"invalid bbox after square reference crop: {transformed_bbox!r}"
        )
    return cropped, transformed_bbox


def _resize_square(
    image: Image.Image,
    bbox: Sequence[float],
    output_size: int,
) -> tuple[Image.Image, list[float]]:
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError(f"reference content size must be positive, got {output_size}")
    width, height = image.size
    if width != height:
        raise ValueError(
            "reference content resizing expects a square source; configure a "
            "crop margin before content_size"
        )
    scale = output_size / float(width)
    resized_bbox = [float(value) * scale for value in bbox]
    resized = image.resize((output_size, output_size), BICUBIC)
    if not valid_bbox(resized_bbox, resized.size):
        raise ValueError(f"invalid bbox after reference resize: {resized_bbox!r}")
    return resized, resized_bbox


def _center_on_canvas(
    image: Image.Image,
    bbox: Sequence[float],
    canvas_size: int,
    fill: int,
) -> tuple[Image.Image, list[float]]:
    canvas_size = int(canvas_size)
    fill = int(fill)
    if canvas_size <= 0:
        raise ValueError(f"reference canvas size must be positive, got {canvas_size}")
    if not 0 <= fill <= 255:
        raise ValueError(f"reference canvas fill must be in [0, 255], got {fill}")
    if image.width > canvas_size or image.height > canvas_size:
        raise ValueError(
            f"reference content {image.size} does not fit canvas {canvas_size}"
        )

    left = (canvas_size - image.width) // 2
    top = (canvas_size - image.height) // 2
    canvas = Image.new("RGB", (canvas_size, canvas_size), (fill, fill, fill))
    canvas.paste(image, (left, top))
    shifted_bbox = [
        float(bbox[0]) + left,
        float(bbox[1]) + top,
        float(bbox[2]) + left,
        float(bbox[3]) + top,
    ]
    if not valid_bbox(shifted_bbox, canvas.size):
        raise ValueError(f"invalid bbox on reference canvas: {shifted_bbox!r}")
    return canvas, shifted_bbox


def apply_reference_policy(
    image: Image.Image,
    bbox: Sequence[float],
    *,
    crop_margin: float | None = None,
    content_size: int | None = None,
    canvas_size: int | None = None,
    canvas_fill: int = 127,
) -> tuple[Image.Image, list[float], str]:
    """Apply an explicit crop/resize/canvas policy and describe it for caching."""
    result = image.convert("RGB")
    result_bbox = [float(value) for value in bbox]
    if not valid_bbox(result_bbox, result.size):
        raise ValueError(
            f"invalid input reference bbox {result_bbox!r} for {result.size}"
        )

    policy_parts: list[str] = []
    if crop_margin is not None:
        margin = float(crop_margin)
        result, result_bbox = _square_face_crop(result, result_bbox, margin)
        policy_parts.append(f"crop_margin={margin:.6g}")
    if content_size is not None:
        result, result_bbox = _resize_square(result, result_bbox, int(content_size))
        policy_parts.append(f"content_size={int(content_size)}")
    if canvas_size is not None:
        result, result_bbox = _center_on_canvas(
            result,
            result_bbox,
            int(canvas_size),
            int(canvas_fill),
        )
        policy_parts.append(f"canvas_size={int(canvas_size)}")
        policy_parts.append(f"canvas_fill={int(canvas_fill)}")

    # AICODE-NOTE: This exact descriptor belongs in reference_cache_key.
    # Otherwise crop/canvas variants can incorrectly reuse legacy latents.
    descriptor = ";".join(policy_parts) if policy_parts else "identity"
    return result, result_bbox, descriptor
