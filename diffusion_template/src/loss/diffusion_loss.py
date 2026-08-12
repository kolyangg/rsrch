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

    def forward(
        self,
        model_pred,
        target,
        is_masked_loss,
        face_bbox,
        ba_aux_loss=None,
        ba_ownership_loss=None,
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
            'loss_ba_ownership': (
                loss.new_tensor(0.0)
                if ba_ownership_loss is None
                else ba_ownership_loss.detach()
            ),
            'loss_ba_crossview': (
                loss.new_tensor(0.0)
                if ba_crossview_loss is None
                else ba_crossview_loss.detach()
            ),
        }


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


class MetricAlignedMaskedDiffusionLoss(nn.Module):
    """Face diffusion loss plus a defaults-off differentiable ID objective."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model_pred,
        target,
        is_masked_loss,
        face_bbox,
        identity_aux_loss=None,
        identity_aux_weight=None,
        identity_aux_applied=None,
        identity_aux_cosine=None,
        identity_aux_timestep=None,
        identity_aux_pred_norm=None,
        identity_aux_target_norm=None,
        **batch,
    ):
        del batch
        if not is_masked_loss:
            raise RuntimeError(
                "MetricAlignedMaskedDiffusionLoss requires face masking every step"
            )
        diffusion = _masked_face_mse(model_pred, target, face_bbox)
        zero = diffusion.new_tensor(0.0)
        identity_aux_loss = zero if identity_aux_loss is None else identity_aux_loss
        identity_aux_weight = zero if identity_aux_weight is None else identity_aux_weight
        identity_aux_applied = zero if identity_aux_applied is None else identity_aux_applied
        identity_weighted = identity_aux_weight * identity_aux_loss

        def detached(value):
            return zero if value is None else value.detach()

        # 6 Aug 2026 - AICODE-NOTE: The two private graph values let the
        # trainer calibrate the frozen ArcFace gradient without changing any
        # historical loss class or optimizer ownership.
        return {
            "loss": diffusion + identity_weighted,
            "loss_face": diffusion.detach(),
            "loss_identity_aux": identity_aux_loss.detach(),
            "identity_aux_weight": identity_aux_weight.detach(),
            "identity_aux_weighted": identity_weighted.detach(),
            "identity_aux_applied": identity_aux_applied.detach(),
            "identity_aux_cosine": detached(identity_aux_cosine),
            "identity_aux_timestep": detached(identity_aux_timestep),
            "identity_aux_pred_norm": detached(identity_aux_pred_norm),
            "identity_aux_target_norm": detached(identity_aux_target_norm),
            "_loss_diffusion_graph": diffusion,
            "_loss_identity_raw_graph": identity_aux_loss,
        }


class AuditedAlternatingDiffusionLoss(nn.Module):
    """Alternate face-only and full-latent MSE while logging both components."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, is_masked_loss, face_bbox, **batch):
        del batch
        face = _masked_face_mse(model_pred, target, face_bbox)
        full = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = face if is_masked_loss else full
        return {
            "loss": loss,
            "loss_face": face.detach(),
            "loss_full": full.detach(),
            "loss_mode_face": loss.new_tensor(float(bool(is_masked_loss))),
        }
