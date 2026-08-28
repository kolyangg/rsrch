"""Counterfactual spatial-reference supervision for CL39-X06."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F


@dataclass
class CounterfactualResult:
    loss: torch.Tensor
    outside_loss: torch.Tensor
    rank_loss: torch.Tensor
    correct_face_error: torch.Tensor
    counterfactual_face_error: torch.Tensor
    outside_delta: torch.Tensor
    applied: torch.Tensor
    mode_wrong_fraction: torch.Tensor
    mode_null_fraction: torch.Tensor


def deterministic_mode(*, global_step: int, rank: int, batch_size: int,
                       probability: float, wrong_fraction: float) -> str | None:
    rng = random.Random(39_000_061 + 1_000_003 * int(rank) + int(global_step))
    if rng.random() >= float(probability):
        return None
    mode = "wrong" if rng.random() < float(wrong_fraction) else "null"
    return None if mode == "wrong" and int(batch_size) <= 1 else mode


def derangement(batch_size: int, device=None) -> torch.Tensor:
    if batch_size <= 1:
        raise ValueError("A wrong-identity batch requires at least two samples")
    return torch.roll(torch.arange(batch_size, device=device), 1)


def compute_counterfactual_reference_loss(
    *, pred_correct: torch.Tensor, pred_counterfactual: torch.Tensor,
    target_noise: torch.Tensor, target_mask: torch.Tensor,
    outside_weight: float, rank_weight: float, rank_margin: float,
    mode: str,
) -> CounterfactualResult:
    mask = F.interpolate(target_mask.float(), pred_correct.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    outside = 1.0 - mask
    outside_loss = F.smooth_l1_loss(
        pred_counterfactual.float(), pred_correct.detach().float(), reduction="none"
    )
    outside_loss = (outside_loss * outside).sum() / (outside.sum() * pred_correct.shape[1]).clamp_min(1.0)
    correct_error = ((pred_correct.float() - target_noise.float()).square() * mask).sum()
    correct_error = correct_error / (mask.sum() * pred_correct.shape[1]).clamp_min(1.0)
    cf_error = ((pred_counterfactual.float() - target_noise.float()).square() * mask).sum()
    cf_error = cf_error / (mask.sum() * pred_correct.shape[1]).clamp_min(1.0)
    rank_loss = F.relu(float(rank_margin) + correct_error - cf_error.detach()).square()
    loss = float(outside_weight) * outside_loss + float(rank_weight) * rank_loss
    zero, one = loss.new_zeros(()), loss.new_ones(())
    return CounterfactualResult(
        loss=loss, outside_loss=outside_loss, rank_loss=rank_loss,
        correct_face_error=correct_error.detach(), counterfactual_face_error=cf_error.detach(),
        outside_delta=((pred_counterfactual.float() - pred_correct.detach().float()) * outside).square().mean().sqrt(),
        applied=one, mode_wrong_fraction=one if mode == "wrong" else zero,
        mode_null_fraction=one if mode == "null" else zero,
    )


def empty_result(reference: torch.Tensor) -> CounterfactualResult:
    zero = reference.float().new_zeros(())
    return CounterfactualResult(zero, zero, zero, zero, zero, zero, zero, zero, zero)
