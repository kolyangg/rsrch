import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def _masked_face_mse(model_pred, target, face_bbox):
    loss = model_pred.float().new_tensor(0.0)
    model_pred_split = list(torch.split(model_pred, 1, dim=0))
    target_split = list(torch.split(target, 1, dim=0))
    valid = 0

    for i, box in enumerate(face_bbox):
        scaled_box = np.array(box) / 8
        scaled_box = scaled_box.astype(np.int32)

        x0, y0, x1, y1 = [int(v) for v in scaled_box]
        if x1 <= x0 or y1 <= y0:
            continue

        model_pred_i = model_pred_split[i][0, :, y0:y1, x0:x1]
        target_i = target_split[i][0, :, y0:y1, x0:x1]
        if model_pred_i.numel() == 0 or target_i.numel() == 0:
            continue

        loss = loss + F.mse_loss(model_pred_i.float(), target_i.float())
        valid += 1

    if valid == 0:
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss / valid


class DiffusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, **batch):
        if isinstance(model_pred, list):
            loss = 0
            for i in range(len(model_pred)):
                loss = loss + F.mse_loss(model_pred[i].float(), target[i].float())
            loss = loss / len(model_pred)
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return {'loss': loss}


class MaskedDiffusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, is_masked_loss, face_bbox, **batch):
        if is_masked_loss:
            loss = _masked_face_mse(model_pred, target, face_bbox)
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return {'loss': loss}


class BlendedMaskedDiffusionLoss(nn.Module):
    def __init__(self, lambda_face: float = 0.1) -> None:
        super().__init__()
        lambda_face = float(lambda_face)
        if not (0.0 <= lambda_face <= 1.0):
            raise ValueError(f"lambda_face must be in [0, 1], got {lambda_face}")
        self.lambda_face = lambda_face

    def forward(self, model_pred, target, is_masked_loss, face_bbox, **batch):
        full_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if face_bbox is None:
            return {"loss": full_loss}

        masked_loss = _masked_face_mse(model_pred, target, face_bbox)
        loss = (1.0 - self.lambda_face) * full_loss + self.lambda_face * masked_loss

        return {'loss': loss}


class CoreNormalizedDiffusionLoss(nn.Module):
    """Normalize diffusion MSE by each sample's feathered face-core area."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, ba_core_mask, **batch):
        del batch
        if ba_core_mask is None:
            raise ValueError("CoreNormalizedDiffusionLoss requires ba_core_mask")
        mask = ba_core_mask.float()
        if mask.shape[-2:] != model_pred.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=model_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if mask.shape[0] != model_pred.shape[0]:
            raise ValueError(
                "Core mask/model batch mismatch: "
                f"{mask.shape[0]} vs {model_pred.shape[0]}"
            )

        per_pixel = (
            model_pred.float() - target.float()
        ).square().mean(dim=1, keepdim=True)
        numerator = (per_pixel * mask).flatten(1).sum(dim=1)
        denominator = mask.flatten(1).sum(dim=1)
        valid = denominator > 0
        if not bool(valid.any()):
            return {
                "loss": F.mse_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="mean",
                )
            }
        per_sample = numerator[valid] / denominator[valid].clamp_min(1e-6)
        return {"loss": per_sample.mean()}
