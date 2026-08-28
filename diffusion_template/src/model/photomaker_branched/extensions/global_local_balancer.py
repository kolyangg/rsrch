"""Deterministic low-frequency global/head correction for CL39-X08."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .valid_kv_attention import valid_key_sdpa


@dataclass
class GlobalAppearanceResult:
    delta: torch.Tensor
    global_mask: torch.Tensor
    scale: torch.Tensor
    delta_rms: torch.Tensor
    delta_native_ratio: torch.Tensor
    local_overlap_rms: torch.Tensor
    leakage_rms: torch.Tensor


def _lowpass(sequence: torch.Tensor) -> torch.Tensor:
    batch, length, channels = sequence.shape
    side = int(math.isqrt(length))
    image = sequence.float().transpose(1, 2).reshape(batch, channels, side, side)
    kernel_1d = image.new_tensor([1., 4., 6., 4., 1.]) / 16.
    kernel = (kernel_1d[:, None] * kernel_1d[None]).view(1, 1, 5, 5).expand(channels, 1, 5, 5)
    return F.conv2d(image, kernel, padding=2, groups=channels).flatten(2).transpose(1, 2).to(sequence.dtype)


def compute_global_appearance_delta(
    *, query_heads: torch.Tensor, reference_key_heads: torch.Tensor,
    reference_value_heads: torch.Tensor, target_face_mask: torch.Tensor,
    reference_face_mask: torch.Tensor, native_message: torch.Tensor,
    native_out: torch.Tensor, project_output, progress: torch.Tensor,
    dilation_cells: int, early_scale: float, late_scale: float,
    native_cap: float, local_exclusion: float,
) -> GlobalAppearanceResult:
    batch, _, length, _ = query_heads.shape
    side = int(math.isqrt(length))
    face = target_face_mask.transpose(1, 2).reshape(batch, 1, side, side).float()
    ref_face = reference_face_mask.transpose(1, 2).reshape(batch, 1, side, side).float()
    kernel = 2 * int(dilation_cells) + 1
    global_target = F.max_pool2d(face, kernel, 1, dilation_cells)
    global_ref = F.max_pool2d(ref_face, kernel, 1, dilation_cells)
    # A second average pass produces a soft boundary without changing the core.
    global_target = F.avg_pool2d(global_target, 3, 1, 1).clamp(0, 1)
    global_ref = F.avg_pool2d(global_ref, 3, 1, 1).clamp(0, 1)
    valid = global_ref.flatten(2).squeeze(1).gt(0.05)
    result = valid_key_sdpa(
        query_heads, reference_key_heads, reference_value_heads, valid,
        fallback=torch.zeros_like(query_heads), return_entropy=False,
    )
    global_message = result.message.transpose(1, 2).reshape_as(native_message)
    delta = _lowpass(project_output(global_message) - native_out)
    support = (global_target - float(local_exclusion) * face).clamp(0, 1)
    support = support.flatten(2).transpose(1, 2).to(delta.dtype)
    native_rms = native_out.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    delta_rms = delta.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    delta = delta * (float(native_cap) * native_rms / delta_rms).clamp(max=1).to(delta.dtype)
    scale = float(early_scale) + progress * (float(late_scale) - float(early_scale))
    confidence = result.valid_fraction.view(batch, 1, 1).to(delta.dtype)
    delta = delta * support * scale * confidence
    leakage = (delta.float() * (1.0 - support.float())).square().mean().sqrt()
    local_overlap = (delta.float() * face.flatten(2).transpose(1, 2)).square().mean().sqrt()
    final_rms = delta.float().square().mean().sqrt()
    return GlobalAppearanceResult(
        delta=delta, global_mask=support, scale=scale,
        delta_rms=final_rms,
        delta_native_ratio=final_rms / native_rms.mean().clamp_min(1.0e-6),
        local_overlap_rms=local_overlap, leakage_rms=leakage,
    )
