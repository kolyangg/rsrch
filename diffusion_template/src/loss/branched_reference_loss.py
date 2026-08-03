"""Defaults-off objective for residual reference-conditioned BA experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .diffusion_loss import _masked_face_mse


def _face_boundary_ring_mse(model_pred, target, face_bbox, ring_width: int = 2):
    """MSE on a latent-space ring immediately outside the target face box."""
    total = model_pred.float().new_tensor(0.0)
    valid = 0
    height, width = model_pred.shape[-2:]
    ring_width = max(1, int(ring_width))
    for index, box in enumerate(face_bbox):
        scaled = (np.array(box) / 8).astype(np.int32)
        x0, y0, x1, y1 = [int(value) for value in scaled]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        ex0 = max(0, x0 - ring_width)
        ex1 = min(width, x1 + ring_width)
        ey0 = max(0, y0 - ring_width)
        ey1 = min(height, y1 + ring_width)
        ring = torch.ones(
            ey1 - ey0,
            ex1 - ex0,
            dtype=torch.bool,
            device=model_pred.device,
        )
        ring[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0] = False
        if not torch.any(ring):
            continue
        pred_region = model_pred[index, :, ey0:ey1, ex0:ex1]
        target_region = target[index, :, ey0:ey1, ex0:ex1]
        total = total + F.mse_loss(
            pred_region[:, ring].float(), target_region[:, ring].float()
        )
        valid += 1
    if valid == 0:
        return model_pred.float().new_tensor(0.0)
    return total / valid


class BranchedReferenceLoss(nn.Module):
    """Full/face/boundary objective with an opt-in spatial-reference margin."""

    def __init__(
        self,
        full_weight: float = 1.0,
        face_weight: float = 1.0,
        boundary_weight: float = 0.1,
        boundary_ring_width: int = 2,
        reference_weight: float = 0.0,
        reference_margin: float = 0.0,
        reference_mode: str = "detached_diagnostic",
        reference_relative_margin: float = 0.0,
    ) -> None:
        super().__init__()
        weights = {
            "full_weight": full_weight,
            "face_weight": face_weight,
            "boundary_weight": boundary_weight,
            "reference_weight": reference_weight,
        }
        for name, value in weights.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if float(full_weight) + float(face_weight) <= 0.0:
            raise ValueError("At least one of full_weight/face_weight must be positive")
        self.full_weight = float(full_weight)
        self.face_weight = float(face_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_ring_width = max(1, int(boundary_ring_width))
        self.reference_weight = float(reference_weight)
        self.reference_margin = float(reference_margin)
        self.reference_mode = (reference_mode or "detached_diagnostic").lower()
        if self.reference_mode not in {
            "detached_diagnostic",
            "differentiable_rank",
        }:
            raise ValueError(
                "reference_mode must be 'detached_diagnostic' or "
                f"'differentiable_rank', got {self.reference_mode!r}"
            )
        self.reference_relative_margin = float(reference_relative_margin)
        if self.reference_relative_margin < 0.0:
            raise ValueError("reference_relative_margin must be non-negative")

    def forward(
        self,
        model_pred,
        target,
        face_bbox,
        pred_wrong_spatial_ref=None,
        reference_shuffle_applied=None,
        reference_prediction_delta_ratio=None,
        ba_telemetry=None,
        **batch,
    ):
        del batch
        full = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        face = _masked_face_mse(model_pred, target, face_bbox)
        boundary = _face_boundary_ring_mse(
            model_pred,
            target,
            face_bbox,
            ring_width=self.boundary_ring_width,
        )
        loss = (
            self.full_weight * full
            + self.face_weight * face
            + self.boundary_weight * boundary
        )

        if reference_shuffle_applied is None:
            shuffle_applied = model_pred.float().new_tensor(0.0)
        elif torch.is_tensor(reference_shuffle_applied):
            shuffle_applied = reference_shuffle_applied.detach().to(
                device=model_pred.device, dtype=torch.float32
            )
        else:
            shuffle_applied = model_pred.float().new_tensor(
                float(reference_shuffle_applied)
            )
        reference_gap = model_pred.float().new_tensor(0.0)
        reference_relative_gap = model_pred.float().new_tensor(0.0)
        reference_causal = model_pred.float().new_tensor(0.0)
        wrong_face = model_pred.float().new_tensor(0.0)
        if pred_wrong_spatial_ref is not None:
            if self.reference_mode == "differentiable_rank":
                # 2 Aug 2026 - Both counterfactual predictions remain in the
                # graph only for the explicit ranking mode. The denominator is
                # detached so the relative scale cannot be gamed.
                wrong_face = _masked_face_mse(
                    pred_wrong_spatial_ref, target, face_bbox
                )
                reference_gap = (wrong_face.detach() - face.detach())
                reference_relative_gap = (
                    (wrong_face - face) / face.detach().clamp_min(1.0e-6)
                )
                reference_causal = F.relu(
                    self.reference_relative_margin - reference_relative_gap
                )
            else:
                # Exact residual-v2 detached diagnostic and absolute-margin
                # behavior remains available when the new mode is disabled.
                wrong_face = _masked_face_mse(
                    pred_wrong_spatial_ref.detach(), target, face_bbox
                )
                gap_for_loss = wrong_face.detach() - face
                reference_gap = gap_for_loss.detach()
                reference_relative_gap = (
                    reference_gap / face.detach().clamp_min(1.0e-6)
                )
                reference_causal = F.relu(self.reference_margin - gap_for_loss)
            loss = loss + self.reference_weight * reference_causal

        if reference_prediction_delta_ratio is None:
            reference_prediction_delta_ratio = model_pred.float().new_tensor(0.0)
        elif torch.is_tensor(reference_prediction_delta_ratio):
            reference_prediction_delta_ratio = (
                reference_prediction_delta_ratio.detach()
                .to(device=model_pred.device, dtype=torch.float32)
            )
        else:
            reference_prediction_delta_ratio = model_pred.float().new_tensor(
                float(reference_prediction_delta_ratio)
            )

        result = {
            "loss": loss,
            "loss_full": full.detach(),
            "loss_face": face.detach(),
            "loss_boundary": boundary.detach(),
            "loss_reference_causal": reference_causal.detach(),
            "reference_error_gap": reference_gap,
            "reference_error_relative_gap": reference_relative_gap.detach(),
            "loss_wrong_reference_face": wrong_face.detach(),
            "reference_prediction_delta_ratio": reference_prediction_delta_ratio,
            "reference_shuffle_applied": shuffle_applied,
        }
        if ba_telemetry:
            for name, value in ba_telemetry.items():
                if not torch.is_tensor(value):
                    value = model_pred.float().new_tensor(float(value))
                result[str(name)] = value.detach().to(
                    device=model_pred.device, dtype=torch.float32
                )
        return result
