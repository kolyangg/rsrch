"""Small defaults-off identity/geometry helpers for CL40-CL43."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityMotionProjector(nn.Module):
    """Remove a motion-aligned component before mapping back to BA width."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.target = nn.Linear(hidden_size, rank, bias=False)
        self.reference = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, target: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 19 Aug 2026 - Normalize stably in fp32, then match the learned
        # projector dtype used by bf16 validation and training.
        target_normalized = F.layer_norm(target.float(), (target.shape[-1],)).to(
            self.target.weight.dtype
        )
        reference_normalized = F.layer_norm(
            reference.float(), (reference.shape[-1],)
        ).to(self.reference.weight.dtype)
        target_low = self.target(target_normalized)
        reference_low = self.reference(reference_normalized)
        common = 0.5 * (target_low + reference_low)
        motion = target_low - reference_low
        coefficient = (common * motion).sum(-1, keepdim=True) / motion.square().sum(
            -1, keepdim=True
        ).clamp_min(1.0e-6)
        identity = common - coefficient * motion
        before = F.cosine_similarity(common, motion, dim=-1).abs().mean()
        after = F.cosine_similarity(identity, motion, dim=-1).abs().mean()
        return self.output(identity).to(target.dtype), before, after


class IDAdaptiveModulation(nn.Module):
    """Zero-start AdaLN-style correction driven by a 512-D face vector."""

    def __init__(self, hidden_size: int, bottleneck: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.input = nn.Linear(embedding_dim, bottleneck)
        self.output = nn.Linear(bottleneck, 2 * hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, delta: torch.Tensor, identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 19 Aug 2026 - Keep stable fp32 normalization, then honor the
        # projector dtype used by mixed-precision validation/training.
        identity = F.normalize(identity.float(), dim=-1).to(self.input.weight.dtype)
        gamma, beta = self.output(F.silu(self.input(identity))).chunk(2, dim=-1)
        normalized = F.layer_norm(delta.float(), (delta.shape[-1],))
        correction = gamma[:, None] * normalized + beta[:, None]
        return correction.to(delta.dtype), gamma, beta


ARCFACE_112_TEMPLATE = torch.tensor(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=torch.float32,
) / 112.0


def similarity_grid_from_landmarks(
    landmarks: torch.Tensor,
    *,
    side: int,
    template: torch.Tensor = ARCFACE_112_TEMPLATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a canonical output grid to observed normalized reference points."""

    landmarks = landmarks.float()
    template = template.to(device=landmarks.device, dtype=torch.float32)
    grids, valid_rows = [], []
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, side, device=landmarks.device),
        torch.linspace(0.0, 1.0, side, device=landmarks.device),
        indexing="ij",
    )
    canonical = torch.stack((x, y), dim=-1)
    for points in landmarks:
        finite = torch.isfinite(points).all()
        spread = torch.linalg.vector_norm(points.max(0).values - points.min(0).values)
        valid = bool(finite.item() and spread.item() > 0.05)
        if valid:
            source_center = points.mean(0)
            target_center = template.mean(0)
            source_zero = points - source_center
            target_zero = template - target_center
            covariance = target_zero.transpose(0, 1) @ source_zero
            u, _, vh = torch.linalg.svd(covariance)
            rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
            if torch.linalg.det(rotation) < 0:
                vh = vh.clone()
                vh[-1] *= -1
                rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
            scale = (source_zero * (target_zero @ rotation.transpose(0, 1))).sum() / target_zero.square().sum().clamp_min(1.0e-6)
            mapped = (canonical - target_center) @ rotation.transpose(0, 1) * scale + source_center
            valid = bool(torch.isfinite(mapped).all().item())
        if not valid:
            mapped = canonical
        grids.append(mapped.mul(2.0).sub(1.0))
        valid_rows.append(valid)
    return torch.stack(grids), torch.tensor(valid_rows, device=landmarks.device)
