"""Experiment-local landmark-aware correspondence for NN7 clean patch memory.

This module monkey-patches only instantiated experiment processors. It does not
modify or replace production source. Target queries retain target coordinates;
only the center of their local reference K/V window is smoothly displaced by
five-point landmark correspondence.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F


def _landmark_local_spatial_candidate(
    self,
    attn,
    target_query: torch.Tensor,
    target_base: torch.Tensor,
    target_core: torch.Tensor,
    target_face: torch.Tensor,
    target_hidden: torch.Tensor | None = None,
):
    if self.ref_to_k is None or self.ref_to_v is None:
        raise RuntimeError("Clean spatial K/V projections were not initialized")
    if self.spatial_patch_tokens is None:
        raise RuntimeError("Clean spatial memory has no patch tokens")
    if not hasattr(self, "_experimental_target_landmarks_xy"):
        raise RuntimeError("Target/reference landmark correspondence was not set")

    patches = self.spatial_patch_tokens.to(
        device=target_base.device,
        dtype=target_base.dtype,
    )
    batch_size, patch_count, patch_dim = patches.shape
    if batch_size != target_base.shape[0] or patch_dim != self.spatial_patch_dim:
        raise ValueError(
            f"Clean patches must be [B,P,{self.spatial_patch_dim}] with "
            f"B={target_base.shape[0]}; got {tuple(patches.shape)}"
        )
    patch_side = int(math.isqrt(patch_count))
    query_side = int(math.isqrt(target_base.shape[1]))
    if patch_side * patch_side != patch_count:
        raise ValueError(f"Clean patch count {patch_count} is not square")
    if query_side * query_side != target_base.shape[1]:
        raise ValueError(f"Target token count {target_base.shape[1]} is not square")

    if self.spatial_attention_space == "sibling_attn2_full":
        if target_hidden is None:
            raise RuntimeError("Full sibling attention requires target hidden states")
        reference_query = self._to_heads(self.spatial_to_q(target_hidden), attn.heads)
        if self.spatial_q_norm is not None:
            reference_query = self.spatial_q_norm(reference_query)
    else:
        reference_query = target_query

    key = self._to_heads(self.ref_to_k(patches), attn.heads)
    value = self._to_heads(self.ref_to_v(patches), attn.heads)
    if self.spatial_attention_space == "sibling_attn2_full":
        if self.spatial_k_norm is not None:
            key = self.spatial_k_norm(key)
    else:
        key = self._apply_k_norm(attn, key)

    candidate = target_base.clone()
    lengths = torch.zeros(batch_size, device=target_base.device, dtype=torch.long)
    support = torch.zeros(batch_size, device=target_base.device, dtype=torch.bool)
    radius = self.spatial_local_window // 2
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=target_base.device),
        torch.arange(-radius, radius + 1, device=target_base.device),
        indexing="ij",
    )
    offsets_y = offsets_y.flatten()
    offsets_x = offsets_x.flatten()
    target_landmarks = self._experimental_target_landmarks_xy.to(
        device=target_base.device, dtype=torch.float32
    )
    reference_landmarks = self._experimental_reference_landmarks_xy.to(
        device=target_base.device, dtype=torch.float32
    )
    sigma = float(getattr(self, "_experimental_landmark_sigma", 0.22))

    for sample_index in range(batch_size):
        active = target_core[sample_index, :, 0] > 0
        face = target_face[sample_index, :, 0] > 0
        face_indices = face.nonzero(as_tuple=False).flatten()
        query_indices = active.nonzero(as_tuple=False).flatten()
        if face_indices.numel() == 0 or query_indices.numel() == 0:
            continue
        face_y = torch.div(face_indices, query_side, rounding_mode="floor")
        face_x = face_indices.remainder(query_side)
        y0, y1 = face_y.min(), face_y.max()
        x0, x1 = face_x.min(), face_x.max()
        query_y = torch.div(query_indices, query_side, rounding_mode="floor")
        query_x = query_indices.remainder(query_side)
        query_xy = torch.stack(
            [
                (query_x - x0).float() / float(max(int((x1 - x0).item()), 1)),
                (query_y - y0).float() / float(max(int((y1 - y0).item()), 1)),
            ],
            dim=-1,
        )

        landmark_row = min(sample_index, target_landmarks.shape[0] - 1)
        target_points = target_landmarks[landmark_row]
        reference_points = reference_landmarks[landmark_row]
        semantic_radius = getattr(self, "_experimental_semantic_radius", None)
        if semantic_radius is not None:
            semantic_distance = torch.cdist(
                query_xy.float(), target_points.float()
            ).amin(dim=1)
            eligible = semantic_distance <= float(semantic_radius)
            query_indices = query_indices[eligible]
            query_xy = query_xy[eligible]
            if query_indices.numel() == 0:
                continue
        displacement = reference_points - target_points
        distance_sq = (query_xy[:, None, :] - target_points[None, :, :]).square().sum(-1)
        weights = torch.softmax(-distance_sq / (2.0 * sigma * sigma), dim=-1)
        mapped_xy = (query_xy + weights @ displacement).clamp(0.0, 1.0)
        ref_x = torch.round(mapped_xy[:, 0] * float(patch_side - 1)).long()
        ref_y = torch.round(mapped_xy[:, 1] * float(patch_side - 1)).long()
        local_y = (ref_y[:, None] + offsets_y[None]).clamp(0, patch_side - 1)
        local_x = (ref_x[:, None] + offsets_x[None]).clamp(0, patch_side - 1)
        local_indices = local_y * patch_side + local_x
        local_key = key[sample_index][:, local_indices, :]
        local_value = value[sample_index][:, local_indices, :]
        query = reference_query[sample_index, :, query_indices, :]
        scores = torch.einsum("hqd,hqwd->hqw", query, local_key)
        weights_attn = (scores / math.sqrt(float(query.shape[-1]))).softmax(dim=-1)
        local_output = torch.einsum("hqw,hqwd->hqd", weights_attn, local_value)
        local_output = local_output.transpose(0, 1).reshape(1, query_indices.numel(), -1)
        if self.spatial_attention_space == "sibling_attn2_full":
            local_output = self.spatial_to_out(local_output)
        candidate[sample_index, query_indices] = local_output.squeeze(0)
        lengths[sample_index] = self.spatial_local_window**2
        support[sample_index] = True
    return candidate, lengths, support


def install(processors) -> int:
    count = 0
    for processor in processors:
        if getattr(processor, "spatial_memory_mode", None) != "clean_clip_patches":
            continue
        if not hasattr(processor, "_experimental_original_local_candidate"):
            processor._experimental_original_local_candidate = (
                processor._clean_local_spatial_candidate
            )
        processor._clean_local_spatial_candidate = MethodType(
            _landmark_local_spatial_candidate, processor
        )
        processor._experimental_landmark_correspondence = True
        count += 1
    return count


def set_landmarks(
    processors,
    target_landmarks_xy,
    reference_landmarks_xy,
    *,
    sigma: float = 0.22,
    semantic_radius: float | None = None,
) -> int:
    target = torch.as_tensor(target_landmarks_xy, dtype=torch.float32)
    reference = torch.as_tensor(reference_landmarks_xy, dtype=torch.float32)
    if target.shape != (5, 2) or reference.shape != (5, 2):
        raise ValueError("Expected target and reference landmarks shaped [5,2]")
    count = 0
    for processor in processors:
        if not getattr(processor, "_experimental_landmark_correspondence", False):
            continue
        processor._experimental_target_landmarks_xy = target.unsqueeze(0)
        processor._experimental_reference_landmarks_xy = reference.unsqueeze(0)
        processor._experimental_landmark_sigma = float(sigma)
        processor._experimental_semantic_radius = (
            None if semantic_radius is None else float(semantic_radius)
        )
        count += 1
    return count
