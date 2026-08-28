"""Fixed-resolution small-face correction for CL39-X04."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .valid_kv_attention import valid_key_sdpa


@dataclass
class RoiRouteResult:
    delta: torch.Tensor
    eligible_fraction: torch.Tensor
    face_area: torch.Tensor
    gate: torch.Tensor
    delta_rms: torch.Tensor
    boundary_energy: torch.Tensor


def _boxes(mask: torch.Tensor, expansion: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = mask.shape
    side = int(math.isqrt(length))
    image = mask.reshape(batch, side, side) > 0.5
    rows, cols = image.any(2), image.any(1)
    valid = rows.any(1) & cols.any(1)
    x0 = cols.float().argmax(1).float()
    y0 = rows.float().argmax(1).float()
    x1 = (side - cols.flip(1).float().argmax(1)).float()
    y1 = (side - rows.flip(1).float().argmax(1)).float()
    width, height = x1 - x0, y1 - y0
    boxes = torch.stack((
        (x0 - float(expansion) * width).floor().clamp(0, side-1),
        (y0 - float(expansion) * height).floor().clamp(0, side-1),
        (x1 + float(expansion) * width).ceil().clamp(1, side),
        (y1 + float(expansion) * height).ceil().clamp(1, side),
    ), dim=1)
    boxes = torch.where(valid[:, None], boxes, torch.tensor(
        [0, 0, 1, 1], device=boxes.device, dtype=boxes.dtype,
    )[None])
    return boxes, valid


def _sample(sequence: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor, size: int) -> torch.Tensor:
    batch, length, channels = sequence.shape
    side = int(math.isqrt(length))
    image = sequence.transpose(1, 2).reshape(batch, channels, side, side)
    phase = (torch.arange(size, device=sequence.device).float() + 0.5) / float(size)
    x0, y0, x1, y1 = boxes.float().unbind(1)
    xs = x0[:, None] + phase[None] * (x1-x0)[:, None]
    ys = y0[:, None] + phase[None] * (y1-y0)[:, None]
    grid_x = (2.0 * xs / float(side) - 1.0)[:, None].expand(-1, size, -1)
    grid_y = (2.0 * ys / float(side) - 1.0)[:, :, None].expand(-1, -1, size)
    grid = torch.stack((grid_x, grid_y), -1)
    output = F.grid_sample(image.float(), grid, mode="bilinear", align_corners=False)
    output = output * valid[:, None, None, None]
    return output.to(sequence.dtype).flatten(2).transpose(1, 2)


def _scatter(sequence: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor, length: int) -> torch.Tensor:
    batch, roi_length, channels = sequence.shape
    side, roi_side = int(math.isqrt(length)), int(math.isqrt(roi_length))
    source = sequence.transpose(1, 2).reshape(batch, channels, roi_side, roi_side)
    coordinate = torch.arange(side, device=sequence.device).float() + 0.5
    x0, y0, x1, y1 = boxes.float().unbind(1)
    grid_x = (2.0 * (coordinate[None] - x0[:, None]) / (x1-x0)[:, None] - 1.0)
    grid_y = (2.0 * (coordinate[None] - y0[:, None]) / (y1-y0)[:, None] - 1.0)
    grid = torch.stack((
        grid_x[:, None].expand(-1, side, -1),
        grid_y[:, :, None].expand(-1, -1, side),
    ), -1)
    output = F.grid_sample(source.float(), grid, mode="bilinear", align_corners=False)
    support = (
        (coordinate[None] >= x0[:, None]) & (coordinate[None] < x1[:, None])
    )[:, None] & (
        (coordinate[None] >= y0[:, None]) & (coordinate[None] < y1[:, None])
    )[:, :, None]
    output = output * support[:, None] * valid[:, None, None, None]
    return output.to(sequence.dtype).flatten(2).transpose(1, 2)


class FaceRoiReferenceRoute(nn.Module):
    def __init__(self, *, roi_size: int, face_area_threshold: float,
                 box_expansion: float, gate_max: float,
                 delta_native_cap: float, boundary_ring_cells: int):
        super().__init__()
        self.roi_size = int(roi_size)
        self.face_area_threshold = float(face_area_threshold)
        self.box_expansion = float(box_expansion)
        self.gate_max = float(gate_max)
        self.delta_native_cap = float(delta_native_cap)
        self.boundary_ring_cells = int(boundary_ring_cells)
        self.gate_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, *, target_hidden, reference_hidden, target_mask,
                reference_mask, native_out, heads: int, project_q, project_native_k,
                project_native_v, project_reference_k, project_reference_v,
                project_output) -> RoiRouteResult:
        batch, length, _ = target_hidden.shape
        target_boxes, target_valid = _boxes(target_mask, self.box_expansion)
        reference_boxes, reference_valid = _boxes(reference_mask, self.box_expansion)
        target_roi = _sample(target_hidden, target_boxes, target_valid, self.roi_size)
        reference_roi = _sample(reference_hidden, reference_boxes, reference_valid, self.roi_size)
        target_roi_mask = _sample(target_mask, target_boxes, target_valid, self.roi_size)
        reference_roi_mask = _sample(reference_mask, reference_boxes, reference_valid, self.roi_size)
        reshape = lambda value: value.view(batch, -1, heads, value.shape[-1] // heads).transpose(1, 2)
        merge = lambda value: value.transpose(1, 2).reshape(batch, -1, value.shape[1] * value.shape[-1])
        q = reshape(project_q(target_roi))
        native_msg = F.scaled_dot_product_attention(
            q, reshape(project_native_k(target_roi)), reshape(project_native_v(target_roi)),
            dropout_p=0.0, is_causal=False,
        )
        ref_result = valid_key_sdpa(
            q, reshape(project_reference_k(reference_roi)), reshape(project_reference_v(reference_roi)),
            reference_roi_mask.squeeze(-1).gt(0.05), fallback=native_msg,
        )
        delta_roi = project_output(merge(ref_result.message)) - project_output(merge(native_msg))
        window_1d = torch.hann_window(self.roi_size + 2, periodic=False, device=delta_roi.device,
                                      dtype=torch.float32)[1:-1]
        window = (window_1d[:, None] * window_1d[None]).flatten().view(1, -1, 1)
        if self.boundary_ring_cells:
            ring = self.boundary_ring_cells
            window_image = window.reshape(1, self.roi_size, self.roi_size, 1)
            window_image[:, :ring] = 0
            window_image[:, -ring:] = 0
            window_image[:, :, :ring] = 0
            window_image[:, :, -ring:] = 0
            window = window_image.reshape(1, -1, 1)
        delta_roi = delta_roi * window.to(delta_roi.dtype) * target_roi_mask.clamp(0, 1)
        native_roi = _sample(native_out, target_boxes, target_valid, self.roi_size)
        native_rms = native_roi.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        delta_rms = delta_roi.float().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        delta_roi = delta_roi * (self.delta_native_cap * native_rms / delta_rms).clamp(max=1).to(delta_roi.dtype)
        face_area = target_mask.float().mean((1, 2))
        eligible = (
            face_area.lt(self.face_area_threshold)
            & target_valid
            & reference_valid
            & ref_result.eligible.flatten()
        )
        gate = self.gate_max * torch.tanh(self.gate_raw)
        delta = _scatter(delta_roi, target_boxes, target_valid, length) * eligible[:, None, None].to(delta_roi.dtype) * gate.to(delta_roi.dtype)
        return RoiRouteResult(
            delta=delta, eligible_fraction=eligible.float().mean(), face_area=face_area.mean(),
            gate=gate.detach(), delta_rms=delta.float().square().mean().sqrt(),
            boundary_energy=(delta_roi[:, : self.roi_size].float().square().mean().sqrt()),
        )
