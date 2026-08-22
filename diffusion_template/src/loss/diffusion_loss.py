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

        if "ba_aux_loss" in batch or "ba_crossview_loss" in batch:
            auxiliary = batch.get("ba_aux_loss")
            if auxiliary is None:
                auxiliary = loss.new_tensor(0.0)
            crossview = batch.get("ba_crossview_loss")
            if crossview is None:
                crossview = loss.new_tensor(0.0)
            return {
                'loss': loss + auxiliary,
                'loss_ba_aux': auxiliary.detach(),
                'loss_ba_crossview': crossview.detach(),
            }

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
