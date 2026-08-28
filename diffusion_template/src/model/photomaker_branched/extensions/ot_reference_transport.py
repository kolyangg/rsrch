"""Fixed-grid, parameter-free stage-split reference transport for CL39-X03."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class OTTransportResult:
    message: torch.Tensor
    plan_entropy: torch.Tensor
    displacement: torch.Tensor
    row_error: torch.Tensor
    col_error: torch.Tensor
    valid_fraction: torch.Tensor
    late_fraction: torch.Tensor
    eligible: torch.Tensor


def _resize_heads(value: torch.Tensor, source_side: int, target_side: int) -> torch.Tensor:
    batch, heads, _, width = value.shape
    image = value.permute(0, 1, 3, 2).reshape(batch * heads, width, source_side, source_side)
    image = F.interpolate(image.float(), (target_side, target_side), mode="bilinear", align_corners=False)
    return image.reshape(batch, heads, width, target_side * target_side).permute(0, 1, 3, 2).to(value.dtype)


class StageSplitReferenceTransport:
    def __init__(
        self, *, grid_size: int, epsilon: float, iterations: int,
        coordinate_weight: float, transition_start: float,
        transition_end: float, late_top_k: int, min_valid_tokens: int,
        detach_plan: bool = True,
    ):
        self.grid_size = int(grid_size)
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.coordinate_weight = float(coordinate_weight)
        self.transition_start = float(transition_start)
        self.transition_end = float(transition_end)
        self.late_top_k = int(late_top_k)
        self.min_valid_tokens = int(min_valid_tokens)
        self.detach_plan = bool(detach_plan)

    def __call__(
        self, *, query_heads: torch.Tensor, key_heads: torch.Tensor,
        value_heads: torch.Tensor, target_mask: torch.Tensor,
        reference_mask: torch.Tensor, progress: torch.Tensor,
        fallback: torch.Tensor,
    ) -> OTTransportResult:
        batch, heads, length, width = query_heads.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise ValueError("OT transport requires square token grids")
        grid = min(self.grid_size, side)
        q = _resize_heads(query_heads, side, grid)
        k = _resize_heads(key_heads, side, grid)
        v = _resize_heads(value_heads, side, grid)
        target_valid = F.interpolate(
            target_mask.transpose(1, 2).reshape(batch, 1, side, side).float(),
            (grid, grid), mode="bilinear", align_corners=False,
        ).flatten(1).gt(0.5)
        ref_valid = F.interpolate(
            reference_mask.transpose(1, 2).reshape(batch, 1, side, side).float(),
            (grid, grid), mode="bilinear", align_corners=False,
        ).flatten(1).gt(0.5)
        eligible = (
            target_valid.sum(-1).ge(self.min_valid_tokens)
            & ref_valid.sum(-1).ge(self.min_valid_tokens)
        )
        q_match = F.normalize(q.float(), dim=-1).mean(1)
        k_match = F.normalize(k.float(), dim=-1).mean(1)
        semantic = 1.0 - torch.matmul(q_match, k_match.transpose(1, 2))
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid, device=q.device),
            torch.linspace(-1, 1, grid, device=q.device), indexing="ij",
        ), -1).reshape(grid * grid, 2)
        coordinate = torch.cdist(coords, coords).square()[None]
        cost = semantic + self.coordinate_weight * coordinate
        valid_pair = target_valid[:, :, None] & ref_valid[:, None, :]
        log_kernel = (-cost / self.epsilon).masked_fill(~valid_pair, -1.0e4)
        log_a = -target_valid.sum(-1).clamp_min(1).float().log()[:, None]
        log_b = -ref_valid.sum(-1).clamp_min(1).float().log()[:, None]
        log_u = torch.where(target_valid, log_a, torch.full_like(log_a.expand_as(target_valid), -1.0e4))
        log_v = torch.where(ref_valid, log_b, torch.full_like(log_b.expand_as(ref_valid), -1.0e4))
        context = torch.no_grad() if self.detach_plan else torch.enable_grad()
        with context:
            for _ in range(self.iterations):
                log_u = torch.where(target_valid, log_a - torch.logsumexp(log_kernel + log_v[:, None], dim=2), log_u)
                log_v = torch.where(ref_valid, log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1), log_v)
            plan = torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None]) * valid_pair
            row = plan.sum(-1)
            col = plan.sum(-2)
            row_plan = plan / row[:, :, None].clamp_min(1.0e-8)
        early = torch.einsum("bij,bhjd->bhid", row_plan.to(v.dtype), v)
        top = row_plan.topk(min(self.late_top_k, row_plan.shape[-1]), dim=-1).indices
        gather_index = top[:, None, :, :, None].expand(-1, heads, -1, -1, width)
        expanded_k = k[:, :, None].expand(-1, -1, grid * grid, -1, -1)
        expanded_v = v[:, :, None].expand_as(expanded_k)
        top_k = torch.gather(expanded_k, 3, gather_index)
        top_v = torch.gather(expanded_v, 3, gather_index)
        logits = (q[:, :, :, None].float() * top_k.float()).sum(-1) / math.sqrt(width)
        late = (torch.softmax(logits, -1).to(v.dtype)[..., None] * top_v).sum(-2)
        p = progress.reshape(batch, 1, 1, 1).to(q.dtype)
        alpha = ((p - self.transition_start) / (self.transition_end - self.transition_start)).clamp(0, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        message = early * (1.0 - alpha) + late * alpha
        message = _resize_heads(message, grid, side)
        message = torch.where(eligible[:, None, None, None], message, fallback)
        entropy = -(row_plan * row_plan.clamp_min(1e-12).log()).sum(-1)
        displacement = (row_plan * coordinate).sum(-1).sqrt()
        row_target = torch.where(target_valid, torch.exp(log_a), torch.zeros_like(row))
        col_target = torch.where(ref_valid, torch.exp(log_b), torch.zeros_like(col))
        target_count = target_valid.float().sum().clamp_min(1.0)
        return OTTransportResult(
            message=message,
            plan_entropy=(entropy * target_valid).sum() / target_count,
            displacement=(displacement * target_valid).sum() / target_count,
            row_error=(row - row_target).abs().mean(),
            col_error=(col - col_target).abs().mean(),
            valid_fraction=(target_valid.float().mean() + ref_valid.float().mean()) * 0.5,
            late_fraction=alpha.float().mean(),
            eligible=eligible.float().mean(),
        )
