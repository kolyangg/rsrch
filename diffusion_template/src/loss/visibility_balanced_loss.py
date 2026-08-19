"""Region-normalized reconstruction loss for synthetic face occlusions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _mean_on_region(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = (mask.sum() * error.shape[1]).clamp_min(1.0)
    return (error * mask).sum() / denom


class VisibilityBalancedBranchedLoss(nn.Module):
    """Balance visible-face and occluder pixels without changing clean batches."""

    def __init__(
        self,
        visible_face_weight: float = 0.75,
        top_object_weight: float = 0.25,
        contact_weight: float = 0.05,
        contact_ring_width: int = 1,
        full_weight: float = 0.05,
        apply_partition_only_when_occluded: bool = True,
        include_ba_aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.visible_face_weight = float(visible_face_weight)
        self.top_object_weight = float(top_object_weight)
        self.contact_weight = float(contact_weight)
        self.contact_ring_width = int(contact_ring_width)
        self.full_weight = float(full_weight)
        self.apply_partition_only_when_occluded = bool(
            apply_partition_only_when_occluded
        )
        self.include_ba_aux_loss = bool(include_ba_aux_loss)
        if min(
            self.visible_face_weight,
            self.top_object_weight,
            self.contact_weight,
            self.full_weight,
        ) < 0.0 or self.contact_ring_width < 1:
            raise ValueError("Visibility-balanced weights must be non-negative")

    @staticmethod
    def _face_mask(model_pred, face_bbox) -> torch.Tensor:
        batch, _, height, width = model_pred.shape
        mask = model_pred.float().new_zeros(batch, 1, height, width)
        scale_x, scale_y = width / 1024.0, height / 1024.0
        for index, box in enumerate(face_bbox):
            x0, y0, x1, y1 = [float(value) for value in box]
            x0, x1 = int(x0 * scale_x), int(x1 * scale_x)
            y0, y1 = int(y0 * scale_y), int(y1 * scale_y)
            mask[index, :, max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1.0
        return mask

    def forward(
        self,
        model_pred,
        target,
        face_bbox,
        ba_occluder_mask=None,
        ba_aux_loss=None,
        **batch,
    ):
        del batch
        error = (model_pred.float() - target.float()).square()
        face = self._face_mask(model_pred, face_bbox)
        face_loss = _mean_on_region(error, face)
        zero = face_loss.new_tensor(0.0)
        top_loss = contact_loss = full_loss = zero
        applied = zero

        top = None
        if ba_occluder_mask is not None:
            items = []
            for value in ba_occluder_mask:
                value = torch.as_tensor(value, device=error.device, dtype=torch.float32)
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                items.append(value)
            top = torch.stack(items)
            top = F.interpolate(top, size=error.shape[-2:], mode="nearest") * face

        if top is not None:
            applied = (top.sum(dim=(1, 2, 3)) > 0).float().mean()
            visible = (face - top).clamp(0.0, 1.0)
            kernel = 2 * self.contact_ring_width + 1
            dilated = F.max_pool2d(top, kernel, stride=1, padding=self.contact_ring_width)
            eroded = 1.0 - F.max_pool2d(
                1.0 - top, kernel, stride=1, padding=self.contact_ring_width
            )
            contact = (dilated - eroded).clamp(0.0, 1.0) * face
            face_loss = _mean_on_region(error, visible)
            top_loss = _mean_on_region(error, top)
            contact_loss = _mean_on_region(error, contact)
            full_loss = error.mean()
            partition = (
                self.visible_face_weight * face_loss
                + self.top_object_weight * top_loss
                + self.contact_weight * contact_loss
                + self.full_weight * full_loss
            )
            reconstruction = applied * partition + (1.0 - applied) * face_loss
        else:
            reconstruction = face_loss

        auxiliary = zero if ba_aux_loss is None else ba_aux_loss
        objective = (
            reconstruction + auxiliary
            if self.include_ba_aux_loss and ba_aux_loss is not None
            else reconstruction
        )
        return {
            "loss": objective,
            "loss_visible_face": face_loss.detach(),
            "loss_top_object": top_loss.detach(),
            "loss_contact": contact_loss.detach(),
            "loss_full": full_loss.detach(),
            "loss_ba_aux": auxiliary.detach(),
            "visibility_partition_applied_fraction": applied,
        }
