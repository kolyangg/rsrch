"""Target-frame reference compositing for crop-only reference datasets.

# 06 Aug 2026 - The branched spatial lane consumes the reference as a full
# 1024px image that is VAE-encoded and denoised through the frozen U-Net; the
# face bbox mask only selects tokens, it cannot rescale them. Cosmic's 256px
# crops therefore reach the branch with a face roughly 2.1x larger (linear) than
# the target face, and with a 4x bilinear upscale behind it. This module renders
# such a crop into the target's face frame *before* the VAE so the frozen
# encoder observes the reference face at the target's own scale and position.
#
# This is deliberately separate from `reference_policy.apply_reference_policy`,
# which stays byte-identical for every historical run.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)

#: `edge` replicates border pixels (smooth, no fabricated structure), `symmetric`
#: mirrors real content, `gray` reproduces the historical blank-canvas surround
#: and exists only as an ablation.
FILL_MODES = ("edge", "symmetric", "gray")

_NUMPY_PAD_MODE = {"edge": "edge", "symmetric": "symmetric"}


def _short_side(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return min(x1 - x0, y1 - y0)


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def compose_target_frame_reference(
    reference: Image.Image,
    reference_bbox: Sequence[float],
    target_bbox: Sequence[float],
    *,
    canvas_size: int = 1024,
    fill: str = "edge",
    gray_level: int = 127,
    target_face_fraction: float | None = None,
    position_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[Image.Image, list[float], str, dict]:
    """Render `reference` so its face matches `target_bbox` in scale and centre.

    Returns the composited canvas, the propagated reference bbox in canvas
    coordinates, a cache descriptor, and telemetry describing how faithfully the
    requested framing was realised.
    """
    if fill not in FILL_MODES:
        raise ValueError(f"reference frame fill must be one of {FILL_MODES}, got {fill!r}")
    canvas_size = int(canvas_size)
    if canvas_size <= 0:
        raise ValueError(f"canvas_size must be positive, got {canvas_size}")

    reference_short = _short_side(reference_bbox)
    target_short = _short_side(target_bbox)
    if reference_short <= 0 or target_short <= 0:
        raise ValueError(
            f"degenerate face box: reference={list(reference_bbox)}, target={list(target_bbox)}"
        )

    # 1. Scale so the reference face short side equals the target face short side.
    #    CL9: when a face fraction is requested, size the reference face to that
    #    fraction of the canvas instead. Training references then span the range
    #    inference actually supplies (~6-30%) rather than locking to one scale,
    #    which is what left CL2 calibrated to a single point.
    if target_face_fraction is not None:
        fraction = float(target_face_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"target_face_fraction must be in (0, 1), got {fraction}")
        # Scale by AREA, not short side: face boxes are not square, so sizing the
        # short side to sqrt(fraction)*canvas overshoots the requested area by
        # exactly the box aspect ratio.
        rx0, ry0, rx1, ry1 = [float(v) for v in reference_bbox]
        reference_area = (rx1 - rx0) * (ry1 - ry0)
        if reference_area <= 0:
            raise ValueError(f"degenerate reference face box: {list(reference_bbox)}")
        scale = ((fraction * canvas_size * canvas_size) / reference_area) ** 0.5
    else:
        scale = target_short / reference_short
    scaled_w = max(1, int(round(reference.width * scale)))
    scaled_h = max(1, int(round(reference.height * scale)))
    scaled = reference.convert("RGB").resize((scaled_w, scaled_h), BICUBIC)
    scaled_bbox = [float(v) * scale for v in reference_bbox]

    # 2. Choose the canvas window in scaled-reference coordinates so the two face
    #    centres coincide. Cropping the periphery is preferred over shrinking the
    #    face, so the realised scale ratio stays exactly 1.0.
    ref_cx, ref_cy = _center(scaled_bbox)
    tgt_cx, tgt_cy = _center(target_bbox)
    # CL9: displacing the paste centre breaks the positional copy shortcut. With
    # the reference face landing exactly on the target face every sample, the
    # branch can satisfy training by copying in place, which does not transfer to
    # validation where the composition differs.
    tgt_cx += float(position_offset[0]) * canvas_size
    tgt_cy += float(position_offset[1]) * canvas_size
    offset_x = ref_cx - tgt_cx
    offset_y = ref_cy - tgt_cy

    # 3. Keep the whole reference face inside the canvas; this can nudge the
    #    centre when the face aspect ratios differ near a border.
    offset_x = min(max(offset_x, scaled_bbox[2] - canvas_size), scaled_bbox[0])
    offset_y = min(max(offset_y, scaled_bbox[3] - canvas_size), scaled_bbox[1])
    center_offset = float(np.hypot(offset_x - (ref_cx - tgt_cx), offset_y - (ref_cy - tgt_cy)))

    window_x0 = int(round(offset_x))
    window_y0 = int(round(offset_y))
    window_x1 = window_x0 + canvas_size
    window_y1 = window_y0 + canvas_size

    crop_x0 = max(0, window_x0)
    crop_y0 = max(0, window_y0)
    crop_x1 = min(scaled_w, window_x1)
    crop_y1 = min(scaled_h, window_y1)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError("target-frame reference window does not intersect the reference")

    patch = np.asarray(scaled)[crop_y0:crop_y1, crop_x0:crop_x1]
    pad_top = crop_y0 - window_y0
    pad_left = crop_x0 - window_x0
    pad_bottom = window_y1 - crop_y1
    pad_right = window_x1 - crop_x1

    if fill == "gray":
        canvas_array = np.full((canvas_size, canvas_size, 3), int(gray_level), dtype=np.uint8)
        canvas_array[pad_top : pad_top + patch.shape[0], pad_left : pad_left + patch.shape[1]] = patch
    else:
        # `symmetric` and `edge` both tolerate pad widths larger than the patch.
        canvas_array = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=_NUMPY_PAD_MODE[fill],
        )
    if canvas_array.shape[:2] != (canvas_size, canvas_size):
        raise RuntimeError(
            f"target-frame compositing produced {canvas_array.shape[:2]}, expected "
            f"{(canvas_size, canvas_size)}"
        )

    canvas_bbox = [
        scaled_bbox[0] - window_x0,
        scaled_bbox[1] - window_y0,
        scaled_bbox[2] - window_x0,
        scaled_bbox[3] - window_y0,
    ]
    canvas_bbox = [float(min(max(v, 0.0), canvas_size)) for v in canvas_bbox]
    if canvas_bbox[2] <= canvas_bbox[0] or canvas_bbox[3] <= canvas_bbox[1]:
        raise ValueError(f"invalid bbox after target-frame compositing: {canvas_bbox}")

    real_pixels = int(patch.shape[0]) * int(patch.shape[1])
    telemetry = {
        # 1.0 by construction in the default path; under a requested face
        # fraction it is intentionally not 1.0, so the preflight gates on
        # `face_fraction` instead.
        "scale_ratio": _short_side(canvas_bbox) / target_short,
        "face_fraction": ((canvas_bbox[2] - canvas_bbox[0])
                          * (canvas_bbox[3] - canvas_bbox[1])) / float(canvas_size ** 2),
        "requested_face_fraction": target_face_fraction,
        "resize_factor": scale,
        "center_offset_px": center_offset,
        "real_fraction": real_pixels / float(canvas_size * canvas_size),
        "cropped": bool(crop_x0 > 0 or crop_y0 > 0 or crop_x1 < scaled_w or crop_y1 < scaled_h),
    }
    # AICODE-NOTE: This descriptor must reach `reference_cache_key`. The composed
    # reference depends on the *target* box, so two samples sharing a reference
    # path do not share conditioning.
    descriptor = (
        f"target_face_frame;canvas={canvas_size};fill={fill};"
        f"scale={scale:.6g};win={window_x0},{window_y0}"
    )
    if target_face_fraction is not None:
        descriptor += f";frac={float(target_face_fraction):.4f}"
    if position_offset != (0.0, 0.0):
        descriptor += f";pos={position_offset[0]:+.4f},{position_offset[1]:+.4f}"
    return Image.fromarray(canvas_array), canvas_bbox, descriptor, telemetry
