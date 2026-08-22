"""Opt-in CL39--44 attention extensions for the unified clean model.

The June attention processor had no landmark canonicalization, component-token
memory, null-key abstention, identity motion projection, adaptive modulation,
or semantic-window gate.  This mixin keeps their helpers out of the shared
processor body while preserving the established tensor operations.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .clean_full_identity_conditioning import similarity_grid_from_landmarks


class RecentAttentionExtensionsMixin:
    """Helpers used only by the recent CL39--44 configuration arms."""

    def _step_ramp(self, start: int, end: int) -> float:
        if end <= start:
            return float(self.ba_training_step >= end)
        return max(0.0, min(1.0, (self.ba_training_step - start) / (end - start)))

    def _null_key_confidence(
        self, attn, q: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached query confidence without retaining an LxL graph."""
        batch, heads, length, width = q.shape
        ref_mask = self._binary_mask(
            self.mask_ref, length, batch, reference.dtype
        )
        keys = self._reshape_heads(
            self._k_ref(attn, reference * ref_mask), heads
        ).detach().float()
        chunks = []
        with torch.no_grad():
            for q_chunk in q.detach().float().split(256, dim=2):
                logits = torch.matmul(q_chunk, keys.transpose(-1, -2)) / math.sqrt(width)
                probability = logits.softmax(dim=-1)
                entropy = -(
                    probability * probability.clamp_min(1.0e-8).log()
                ).sum(dim=-1) / math.log(max(length, 2))
                chunks.append(entropy.mean(dim=1, keepdim=False)[..., None])
            entropy = torch.cat(chunks, dim=1)
            null_mass = torch.sigmoid(
                (entropy - self.null_key_entropy_threshold)
                / self.null_key_temperature
            )
            confidence = (
                1.0 - self.null_key_max_abstention * null_mass
            ).clamp(min=self.null_key_min_reference_fraction, max=1.0)
        return confidence.to(q.dtype), null_mass

    def _landmark_rows(
        self, batch: int, device: torch.device
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        points = self.reference_landmarks_5
        confidence = self.reference_landmark_confidence
        if points is None or confidence is None:
            return None, None
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
        confidence = torch.as_tensor(confidence, device=device, dtype=torch.float32).flatten()
        if points.ndim == 2:
            points = points.unsqueeze(0)
        if points.shape[0] == 1:
            points = points.expand(batch, -1, -1)
        elif batch % points.shape[0] == 0:
            points = points.repeat(batch // points.shape[0], 1, 1)
        if confidence.shape[0] == 1:
            confidence = confidence.expand(batch)
        elif batch % confidence.shape[0] == 0:
            confidence = confidence.repeat(batch // confidence.shape[0])
        if points.shape != (batch, 5, 2) or confidence.shape[0] != batch:
            return None, None
        return points, confidence

    def _canonical_reference_out(
        self,
        attn,
        q: torch.Tensor,
        reference: torch.Tensor,
        original: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels = reference.shape
        side = int(math.isqrt(length))
        points, confidence = self._landmark_rows(batch, reference.device)
        if points is None or side * side != length:
            zero = reference.new_tensor(0.0)
            return original, {"applied": zero, "confidence": zero, "cosine": zero, "rms": zero}
        grid, geometrically_valid = similarity_grid_from_landmarks(points, side=side)
        valid = geometrically_valid & (
            confidence >= self.landmark_canonical_kv_min_confidence
        )
        canonical = F.grid_sample(
            reference.transpose(1, 2).reshape(batch, channels, side, side).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).to(reference.dtype).flatten(2).transpose(1, 2)
        candidate = self._reference_target_out(attn, q, canonical)
        mixed = (1.0 - self.landmark_canonical_kv_mix) * original
        mixed = mixed + self.landmark_canonical_kv_mix * candidate
        ratio = self._masked_rms(original, torch.ones_like(original[..., :1]))
        ratio = ratio / self._masked_rms(mixed, torch.ones_like(mixed[..., :1]))
        mixed = mixed * ratio.to(mixed.dtype)
        valid_mask = valid[:, None, None]
        output = torch.where(valid_mask, mixed, original)
        correction = output.float() - original.float()
        cosine = F.cosine_similarity(
            original.float().flatten(1), candidate.float().flatten(1)
        ).mean()
        return output, {
            "applied": valid.float().mean(),
            "confidence": confidence.mean(),
            "cosine": cosine.detach(),
            "rms": correction.square().mean().sqrt().detach(),
        }

    def _component_memory_correction(
        self,
        attn,
        q: torch.Tensor,
        reference: torch.Tensor,
        routed_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels = reference.shape
        side = int(math.isqrt(length))
        points, confidence = self._landmark_rows(batch, reference.device)
        zero = reference.new_tensor(0.0)
        empty_metrics = {"applied": zero, "rms": zero, "masses": reference.new_zeros(5)}
        if points is None or side * side != length:
            return torch.zeros_like(routed_delta), empty_metrics
        valid = confidence >= self.component_token_memory_min_confidence
        y, x = torch.meshgrid(
            torch.arange(side, device=reference.device, dtype=torch.float32),
            torch.arange(side, device=reference.device, dtype=torch.float32),
            indexing="ij",
        )
        xy = torch.stack((x, y), dim=-1).reshape(1, length, 1, 2)
        centers = torch.stack(
            (points[:, 0], points[:, 1], points[:, 2], points[:, 3:5].mean(1)),
            dim=1,
        ) * float(side - 1)
        distances = (xy - centers[:, None]).square().sum(-1)
        weights = torch.exp(
            -0.5 * distances / (self.component_token_memory_sigma_cells ** 2)
        )
        face = self._binary_mask(self.mask_ref, length, batch, torch.float32)
        weights = weights * face
        global_weight = face
        weights = torch.cat((weights, global_weight), dim=-1)
        mass = weights.sum(dim=1).clamp_min(1.0e-6)
        tokens = torch.einsum("blc,bld->bcd", weights, reference.float()) / mass[..., None]
        component = torch.arange(5, device=reference.device, dtype=torch.float32)
        channel = torch.arange(channels, device=reference.device, dtype=torch.float32)
        type_code = torch.sin((component[:, None] + 1.0) * (channel[None] + 1.0) / channels)
        tokens = (tokens + 0.01 * type_code[None]).to(reference.dtype)
        heads = int(attn.heads)
        message = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_ref(attn, tokens), heads),
            self._reshape_heads(self._v_ref(attn, tokens), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        part = attn.to_out[0](self._merge_heads(message))
        ratio = self._masked_rms(routed_delta, torch.ones_like(routed_delta[..., :1]))
        ratio = ratio / self._masked_rms(part, torch.ones_like(part[..., :1]))
        correction = self.component_token_memory_scale * part * ratio.to(part.dtype)
        correction = correction * valid[:, None, None].to(correction.dtype)
        attention_mass = mass / mass.sum(dim=-1, keepdim=True)
        return correction, {
            "applied": valid.float().mean(),
            "rms": correction.float().square().mean().sqrt().detach(),
            "masses": attention_mass.mean(dim=0).detach(),
        }

    def _semantic_window_scale(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        progress: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        face = self._binary_mask(
            self.mask, target.shape[1], target.shape[0], torch.float32
        )
        ref_face = self._binary_mask(
            self.mask_ref, reference.shape[1], reference.shape[0], torch.float32
        )
        agreement = F.cosine_similarity(
            self._masked_pool(target, face),
            self._masked_pool(reference, ref_face),
            dim=-1,
        ).detach().view(-1, 1, 1)
        rising = torch.sigmoid(
            (progress - self.semantic_window_progress_start)
            / self.semantic_window_progress_temperature
        )
        falling = torch.sigmoid(
            (self.semantic_window_progress_end - progress)
            / self.semantic_window_progress_temperature
        )
        time_weight = rising * falling
        agreement_weight = torch.sigmoid(
            (agreement - self.semantic_window_agreement_threshold)
            / self.semantic_window_agreement_temperature
        )
        window_scale = self.semantic_window_min_scale + (
            self.semantic_window_max_scale - self.semantic_window_min_scale
        ) * time_weight * agreement_weight
        # 19 Aug 2026 - A per-sample fp32 gate would promote the BA residual
        # and break the following bf16 SDXL LayerNorm.
        window_scale = window_scale.to(dtype=target.dtype)
        return window_scale, {
            "semantic_window/agreement": agreement.mean(),
            "semantic_window/time_weight": time_weight.mean(),
            "semantic_window/high_scale": window_scale.mean(),
            "semantic_window/object_minus_visible_scale": window_scale.new_tensor(0.0),
        }
