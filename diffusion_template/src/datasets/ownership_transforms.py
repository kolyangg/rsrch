"""Apply CL39 reference-frame geometry to cached ownership probabilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from src.datasets.reference_frame import compose_target_frame_reference


def compose_target_frame_ownership(
    probabilities: torch.Tensor,
    reference_bbox,
    target_bbox,
    *,
    canvas_size: int,
    target_face_fraction,
    position_offset,
) -> tuple[torch.Tensor, list[float]]:
    if probabilities.ndim != 3 or probabilities.shape[0] != 6:
        raise ValueError("Ownership probabilities must be [6,H,W]")
    channels, propagated_bbox = [], None
    for index, probability in enumerate(probabilities):
        image = Image.fromarray(
            (probability.float().clamp(0, 1) * 255.0).round().byte().numpy(),
            mode="L",
        )
        canvas, bbox, _descriptor, _telemetry = compose_target_frame_reference(
            image,
            reference_bbox,
            target_bbox,
            canvas_size=canvas_size,
            fill="gray",
            gray_level=255 if index == 5 else 0,
            target_face_fraction=target_face_fraction,
            position_offset=position_offset,
        )
        channels.append(torch.from_numpy(np.asarray(canvas)[..., 0].copy()).float() / 255.0)
        if propagated_bbox is None:
            propagated_bbox = bbox
        elif max(abs(a-b) for a, b in zip(propagated_bbox, bbox)) > 1.0e-4:
            raise RuntimeError("Ownership channels received inconsistent reference geometry")
    result = torch.stack(channels).clamp_min(0)
    result = result / result.sum(0, keepdim=True).clamp_min(1.0e-8)
    return result, propagated_bbox
