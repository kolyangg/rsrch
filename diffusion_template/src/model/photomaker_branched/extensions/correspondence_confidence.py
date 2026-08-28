"""Detached entropy-margin-cycle confidence for CL39-X02."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class CorrespondenceConfidence:
    confidence: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    cycle_score: torch.Tensor
    cycle_distance: torch.Tensor
    eligible: torch.Tensor


@torch.no_grad()
def compute_cycle_confidence(
    query: torch.Tensor,
    ref_key: torch.Tensor,
    target_router: torch.Tensor,
    ref_valid: torch.Tensor,
    *,
    floor: float,
    margin_center: float,
    margin_temperature: float,
    cycle_sigma_cells: float,
    chunk_size: int,
    entropy_weight: float = 0.25,
    margin_weight: float = 0.25,
    cycle_weight: float = 0.50,
) -> CorrespondenceConfidence:
    q = F.normalize(query.float(), dim=-1).mean(dim=1)
    k = F.normalize(ref_key.float(), dim=-1).mean(dim=1)
    batch, length, _ = q.shape
    side = int(math.isqrt(length))
    if side * side != length:
        raise ValueError("Cycle confidence requires a square target grid")
    ref_valid = ref_valid.to(q.device, torch.bool)
    target_valid = target_router.squeeze(-1).gt(0)
    eligible_sample = ref_valid.any(-1)
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(side, device=q.device),
        torch.arange(side, device=q.device), indexing="ij"
    ), dim=-1).reshape(length, 2).float()
    entropy_parts, margin_parts, cycle_parts, distance_parts = [], [], [], []
    normalizer = ref_valid.sum(-1).clamp_min(2).float().log()[:, None]
    for start in range(0, length, int(chunk_size)):
        sim = torch.matmul(q[:, start : start + chunk_size], k.transpose(1, 2))
        sim = sim.masked_fill(~ref_valid[:, None], -torch.inf)
        safe = sim.clone()
        safe[~eligible_sample, :, 0] = 0.0
        probability = torch.softmax(safe, dim=-1)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(-1)
        entropy = (entropy / normalizer).clamp(0, 1)
        top = safe.topk(k=min(2, safe.shape[-1]), dim=-1)
        best = top.indices[..., 0]
        if top.values.shape[-1] == 1:
            margin = torch.ones_like(top.values[..., 0])
        else:
            margin = top.values[..., 0] - top.values[..., 1]
        gathered_ref = k.gather(1, best[..., None].expand(-1, -1, k.shape[-1]))
        backward = torch.matmul(gathered_ref, q.transpose(1, 2))
        backward = backward.masked_fill(~target_valid[:, None], -torch.inf)
        safe_back = backward.clone()
        no_target = ~target_valid.any(-1)
        safe_back[no_target, :, 0] = 0.0
        back_index = safe_back.argmax(-1)
        expected = coordinates[start : start + sim.shape[1]][None]
        returned = coordinates[back_index]
        distance = torch.linalg.vector_norm(returned - expected, dim=-1)
        cycle = torch.exp(-0.5 * (distance / float(cycle_sigma_cells)).square())
        entropy_parts.append(entropy)
        margin_parts.append(margin)
        cycle_parts.append(cycle)
        distance_parts.append(distance)
    entropy = torch.cat(entropy_parts, dim=1).unsqueeze(-1)
    margin = torch.cat(margin_parts, dim=1).unsqueeze(-1)
    cycle = torch.cat(cycle_parts, dim=1).unsqueeze(-1)
    distance = torch.cat(distance_parts, dim=1).unsqueeze(-1)
    scores = torch.stack((
        (1.0 - entropy).clamp_min(1.0e-6),
        torch.sigmoid((margin - margin_center) / margin_temperature).clamp_min(1.0e-6),
        cycle.clamp_min(1.0e-6),
    ))
    weights = scores.new_tensor([entropy_weight, margin_weight, cycle_weight]).view(3, 1, 1, 1)
    raw = (weights * scores.log()).sum(0).exp()
    eligible = target_valid.unsqueeze(-1) & eligible_sample[:, None, None]
    confidence = float(floor) + (1.0 - float(floor)) * raw
    confidence = torch.where(
        eligible,
        confidence,
        torch.where(
            eligible_sample[:, None, None],
            torch.ones_like(confidence),
            torch.zeros_like(confidence),
        ),
    )
    return CorrespondenceConfidence(
        confidence=confidence.to(query.dtype), entropy=entropy, margin=margin,
        cycle_score=cycle, cycle_distance=distance, eligible=eligible.float(),
    )
