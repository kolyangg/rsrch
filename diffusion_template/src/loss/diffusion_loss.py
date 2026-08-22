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


class MaskedDiffusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model_pred,
        target,
        is_masked_loss,
        face_bbox,
        ba_aux_loss=None,
        ba_crossview_loss=None,
        **batch,
    ):
        if is_masked_loss:
            loss = _masked_face_mse(model_pred, target, face_bbox)
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        auxiliary = loss.new_tensor(0.0) if ba_aux_loss is None else ba_aux_loss
        return {
            'loss': loss + auxiliary,
            'loss_ba_aux': auxiliary.detach(),
            'loss_ba_crossview': (
                loss.new_tensor(0.0)
                if ba_crossview_loss is None
                else ba_crossview_loss.detach()
            ),
        }
