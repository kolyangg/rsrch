import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


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
            loss = 0
            model_pred = list(torch.split(model_pred, 1, dim=0))
            target = list(torch.split(target, 1, dim=0))
            for i, box in enumerate(face_bbox):
                scaled_box = np.array(box) / 8
                scaled_box = scaled_box.astype(np.int32)
                
                model_pred_i = model_pred[i][0, :, scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]]
                target_i = target[i][0, :, scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]]
                
                loss = loss + F.mse_loss(model_pred_i.float(), target_i.float())
            
            loss = loss / len(face_bbox)
            
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
        return {'loss': loss}
