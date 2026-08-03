"""Anchored native/reference interpolation for branched self-attention.

The target lane keeps frozen native self-attention outside the target face.
Inside the face, target queries interpolate between native target K/V and
explicit, true-key-masked reference K/V.  A frozen native output projection
makes the reference route nonzero at initialization; only its low-rank delta,
reference K/V deltas, and bounded mix are trainable.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_processor_cleanest import _clone_effective_linear
from .residual_sa_processor_v2 import (
    ResidualBranchedSelfAttnProcessorV2,
    ResidualLoRALinear,
)


def _bounded_logit(value: float, lower: float, upper: float) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "mix bounds must satisfy 0 <= floor < max <= 1, got "
            f"floor={lower}, max={upper}"
        )
    if not lower < value < upper:
        raise ValueError(
            f"mix_init must be strictly inside ({lower}, {upper}), got {value}"
        )
    probability = (value - lower) / (upper - lower)
    return math.log(probability / (1.0 - probability))


class AnchoredMixBranchedSelfAttnProcessorV3(
    ResidualBranchedSelfAttnProcessorV2
):
    """Target-Q/reference-KV anchored interpolation for doubled batches."""

    architecture_version = "anchored_mix_sa_v3"
    has_cross_attention_kwargs = True

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        ref_kv_rank: int = 32,
        output_rank: int = 32,
        mix_init: float = 0.50,
        mix_floor: float = 0.25,
        mix_max: float = 0.90,
        mix_timestep: bool = True,
        mix_face_area: bool = True,
        reference_rms_match: bool = True,
        reference_rms_clip_min: float = 0.50,
        reference_rms_clip_max: float = 2.00,
        trainable_dtype: torch.dtype = torch.float32,
        require_denoise_progress: bool = True,
        telemetry_enabled: bool = False,
        telemetry_interval: int = 50,
        mix_override: Optional[float] = None,
    ) -> None:
        nn.Module.__init__(self)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Anchored BA-v3 requires PyTorch 2.0+")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if int(ref_kv_rank) <= 0 or int(output_rank) <= 0:
            raise ValueError("Anchored BA-v3 ranks must be positive")
        if not 0.0 < float(reference_rms_clip_min):
            raise ValueError("reference_rms_clip_min must be positive")
        if float(reference_rms_clip_max) < float(reference_rms_clip_min):
            raise ValueError(
                "reference_rms_clip_max must be >= reference_rms_clip_min"
            )
        if int(telemetry_interval) <= 0:
            raise ValueError("telemetry_interval must be positive")

        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim or hidden_size)
        self.scale = float(scale)
        self.ref_kv_rank = int(ref_kv_rank)
        self.output_rank = int(output_rank)
        self.mix_init = float(mix_init)
        self.mix_floor = float(mix_floor)
        self.mix_max = float(mix_max)
        self.mix_timestep = bool(mix_timestep)
        self.mix_face_area = bool(mix_face_area)
        self.reference_rms_match = bool(reference_rms_match)
        self.reference_rms_clip_min = float(reference_rms_clip_min)
        self.reference_rms_clip_max = float(reference_rms_clip_max)
        self.trainable_dtype = trainable_dtype
        self.require_denoise_progress = bool(require_denoise_progress)

        self.ref_to_k = None
        self.ref_to_v = None
        self.ref_out = ResidualLoRALinear(
            self.hidden_size,
            rank=self.output_rank,
            device=None,
            dtype=trainable_dtype,
            zero_init_output=True,
        )
        self.mix_logit = nn.Parameter(
            torch.tensor(
                _bounded_logit(self.mix_init, self.mix_floor, self.mix_max),
                dtype=trainable_dtype,
            )
        )
        self.mix_t = nn.Parameter(
            torch.zeros((), dtype=trainable_dtype),
            requires_grad=self.mix_timestep,
        )
        self.mix_area = nn.Parameter(
            torch.zeros((), dtype=trainable_dtype),
            requires_grad=self.mix_face_area,
        )

        self.mask = None
        self.mask_ref = None
        self.ba_denoise_progress = None
        self.force_binary_masks = True
        self.cache_prepared_masks = False
        self.telemetry_enabled = bool(telemetry_enabled)
        self.telemetry_interval = int(telemetry_interval)
        self._telemetry_forward_count = 0
        self._latest_ba_telemetry = None
        self.mix_override = None
        self.set_mix_override(mix_override)

    def init_from_attention(self, attn) -> None:
        # 2 Aug 2026 - AICODE-NOTE: target Q and the native path stay frozen;
        # explicit reference K/V start from the effective PhotoMaker projection.
        self.ref_to_k = _clone_effective_linear(
            attn.to_k,
            kind="lora",
            rank=self.ref_kv_rank,
            trainable_dtype=self.trainable_dtype,
        )
        self.ref_to_v = _clone_effective_linear(
            attn.to_v,
            kind="lora",
            rank=self.ref_kv_rank,
            trainable_dtype=self.trainable_dtype,
        )

    def named_ba_trainables(
        self,
    ) -> Iterator[tuple[str, nn.Parameter, str]]:
        if self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Anchored BA-v3 processor was not initialized")
        for prefix, module in (
            ("ref_to_k", self.ref_to_k),
            ("ref_to_v", self.ref_to_v),
        ):
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter, "ref_kv"
        for name, parameter in self.ref_out.named_parameters():
            yield f"ref_out.{name}", parameter, "ref_output"
        yield "mix_logit", self.mix_logit, "mix"
        if self.mix_t.requires_grad:
            yield "mix_t", self.mix_t, "mix"
        if self.mix_area.requires_grad:
            yield "mix_area", self.mix_area, "mix"

    def set_mix_override(self, value: Optional[float]) -> None:
        if value is None:
            self.mix_override = None
            return
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ba_mix_override must be in [0, 1], got {value}")
        self.mix_override = value

    def set_telemetry_enabled(self, enabled: bool) -> None:
        self.telemetry_enabled = bool(enabled)

    def _bounded_mix(
        self,
        *,
        batch_size: int,
        target_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.mix_override is not None:
            return torch.full(
                (batch_size, 1, 1),
                self.mix_override,
                device=device,
                dtype=self.mix_logit.dtype,
            )

        train_dtype = self.mix_logit.dtype
        logits = self.mix_logit.expand(batch_size).to(device=device)
        progress = self.ba_denoise_progress
        if progress is None:
            if self.require_denoise_progress:
                raise RuntimeError("Anchored BA-v3 requires denoise progress")
            progress = torch.full((batch_size,), 0.5, device=device)
        elif not torch.is_tensor(progress):
            progress = torch.tensor(progress, device=device)
        progress = progress.to(device=device, dtype=train_dtype).reshape(-1)
        if progress.numel() == 1:
            progress = progress.expand(batch_size)
        elif progress.numel() != batch_size:
            if batch_size % progress.numel() != 0:
                raise RuntimeError(
                    "BA-v3 denoise-progress batch mismatch: "
                    f"progress={progress.numel()}, target={batch_size}"
                )
            progress = progress.repeat(batch_size // progress.numel())
        if torch.any((progress < 0.0) | (progress > 1.0)):
            raise RuntimeError("BA-v3 denoise progress must be in [0, 1]")
        if self.mix_timestep:
            logits = logits + self.mix_t * (2.0 * progress - 1.0)

        if self.mix_face_area:
            area = target_mask.float().mean(dim=(1, 2)).to(dtype=train_dtype)
            logits = logits + self.mix_area * torch.log(area.clamp_min(1.0e-4))

        unit = torch.sigmoid(logits)
        mix = self.mix_floor + (self.mix_max - self.mix_floor) * unit
        return mix.view(batch_size, 1, 1)

    def _rms_match_reference(
        self,
        reference: torch.Tensor,
        native: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.float()
        denom = (
            mask.sum(dim=(1, 2), keepdim=True) * native.shape[-1]
        ).clamp_min(1.0)
        native_rms = (
            (native.float().square() * mask).sum((1, 2), keepdim=True) / denom
        ).sqrt()
        reference_rms = (
            (reference.float().square() * mask).sum((1, 2), keepdim=True)
            / denom
        ).sqrt()
        ratio = (native_rms / reference_rms.clamp_min(1.0e-6)).clamp(
            self.reference_rms_clip_min,
            self.reference_rms_clip_max,
        ).detach()
        return reference * ratio.to(dtype=reference.dtype)

    @staticmethod
    def _masked_rms(
        tensor: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.float()
        denom = (
            mask.sum(dim=(1, 2)) * tensor.shape[-1]
        ).clamp_min(1.0)
        energy = (tensor.float().square() * mask).sum(dim=(1, 2)) / denom
        return energy.clamp_min(0.0).sqrt()

    def _record_telemetry(
        self,
        *,
        native: torch.Tensor,
        reference: torch.Tensor,
        contribution: torch.Tensor,
        mix: torch.Tensor,
        target_mask: torch.Tensor,
        reference_key_bias: torch.Tensor,
    ) -> None:
        if not self.telemetry_enabled:
            return
        self._telemetry_forward_count += 1
        if (self._telemetry_forward_count - 1) % self.telemetry_interval:
            return
        # 2 Aug 2026 - Store detached scalars only. The diagnostic must never
        # retain a U-Net graph or replace the matched-forward sample.
        with torch.no_grad():
            native_rms = self._masked_rms(native, target_mask).clamp_min(1.0e-8)
            reference_rms = self._masked_rms(reference, target_mask).clamp_min(
                1.0e-8
            )
            contribution_rms = self._masked_rms(contribution, target_mask)
            mask = target_mask.float()
            dot_denom = (
                mask.sum(dim=(1, 2)) * native.shape[-1]
            ).clamp_min(1.0)
            reference_native_dot = (
                reference.float() * native.float() * mask
            ).sum(dim=(1, 2)) / dot_denom
            reference_native_cosine = reference_native_dot / (
                reference_rms * native_rms
            ).clamp_min(1.0e-8)
            merged_rms = self._masked_rms(
                native + contribution,
                target_mask,
            )
            progress = self.ba_denoise_progress
            if progress is None:
                progress_mean = native.new_tensor(float("nan"), dtype=torch.float32)
            else:
                progress_mean = torch.as_tensor(
                    progress, device=native.device, dtype=torch.float32
                ).mean()
            self._latest_ba_telemetry = {
                "mix_mean": mix.detach().float().mean(),
                "mix_min": mix.detach().float().min(),
                "mix_max": mix.detach().float().max(),
                "reference_native_rms_ratio": (
                    reference_rms / native_rms
                ).mean().detach(),
                "contribution_native_rms_ratio": (
                    contribution_rms / native_rms
                ).mean().detach(),
                # 2 Aug 2026 - AICODE-NOTE: RMS matching does not constrain
                # direction; cosine exposes output-adapter rotation that can
                # increase the interpolation delta while alpha decreases.
                "reference_native_cosine": (
                    reference_native_cosine.mean().detach()
                ),
                "merged_native_rms_ratio": (
                    merged_rms / native_rms
                ).mean().detach(),
                "reference_valid_key_fraction": (
                    reference_key_bias == 0
                ).float().mean().detach(),
                "denoise_progress_mean": progress_mean.detach(),
            }

    def latest_ba_telemetry(self) -> Optional[dict[str, torch.Tensor]]:
        return self._latest_ba_telemetry

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_hidden_states, attention_mask, scale, kwargs
        if self.ref_to_k is None or self.ref_to_v is None:
            raise RuntimeError("Anchored BA-v3 processor was not initialized")
        if hidden_states.shape[0] % 2:
            raise RuntimeError(
                "Anchored BA-v3 expects [target, reference] doubled batches"
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch, channels, height * width
            ).transpose(1, 2)
        elif input_ndim != 3:
            raise RuntimeError(f"Unsupported attention input rank: {input_ndim}")

        batch_size = hidden_states.shape[0] // 2
        target_hidden = hidden_states[:batch_size]
        reference_hidden = hidden_states[batch_size:]
        target_residual = residual[:batch_size]
        reference_residual = residual[batch_size:]

        if attn.group_norm is not None:
            target_hidden = attn.group_norm(
                target_hidden.transpose(1, 2)
            ).transpose(1, 2)
            reference_hidden = attn.group_norm(
                reference_hidden.transpose(1, 2)
            ).transpose(1, 2)

        heads = int(attn.heads)
        q_target = self._reshape_heads(attn.to_q(target_hidden), heads)
        k_target = self._reshape_heads(attn.to_k(target_hidden), heads)
        v_target = self._reshape_heads(attn.to_v(target_hidden), heads)
        native_message = F.scaled_dot_product_attention(
            q_target, k_target, v_target, dropout_p=0.0, is_causal=False
        )
        native_out = self._apply_output_projection(
            attn, self._merge_heads(native_message)
        )

        # The target mask must exist before optional RMS matching: both native
        # and reference scales are measured over the target face query region.
        target_mask = self._target_output_mask(
            seq_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=native_out.device,
            dtype=native_out.dtype,
        )
        k_reference = self._reshape_heads(self.ref_to_k(reference_hidden), heads)
        v_reference = self._reshape_heads(self.ref_to_v(reference_hidden), heads)
        key_bias = self._reference_key_bias(
            seq_len=reference_hidden.shape[1],
            batch_size=batch_size,
            device=q_target.device,
            dtype=q_target.dtype,
        )
        reference_message = F.scaled_dot_product_attention(
            q_target,
            k_reference,
            v_reference,
            attn_mask=key_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_message = self._merge_heads(reference_message)
        # Frozen native output supplies a live reference route at update zero;
        # the zero-initialized low-rank projection is an additive adapter only.
        reference_out = self._apply_output_projection(attn, reference_message)
        reference_out = reference_out + self.ref_out(reference_message)
        if self.reference_rms_match:
            reference_out = self._rms_match_reference(
                reference_out, native_out, target_mask
            )

        mix = self._bounded_mix(
            batch_size=batch_size,
            target_mask=target_mask,
            device=native_out.device,
        ).to(dtype=native_out.dtype)
        contribution = (
            target_mask * mix * (reference_out - native_out) * self.scale
        )
        target_out = native_out + contribution
        self._record_telemetry(
            native=native_out,
            reference=reference_out,
            contribution=contribution,
            mix=mix,
            target_mask=target_mask,
            reference_key_bias=key_bias,
        )

        q_reference = self._reshape_heads(attn.to_q(reference_hidden), heads)
        k_reference_base = self._reshape_heads(attn.to_k(reference_hidden), heads)
        v_reference_base = self._reshape_heads(attn.to_v(reference_hidden), heads)
        reference_lane_out = F.scaled_dot_product_attention(
            q_reference,
            k_reference_base,
            v_reference_base,
            dropout_p=0.0,
            is_causal=False,
        )
        reference_lane_out = self._apply_output_projection(
            attn, self._merge_heads(reference_lane_out)
        )

        if input_ndim == 4:
            target_out = target_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )
            reference_lane_out = reference_lane_out.transpose(-1, -2).reshape(
                batch_size, channels, height, width
            )

        if attn.residual_connection:
            target_out = target_out + target_residual
            reference_lane_out = reference_lane_out + reference_residual
        target_out = target_out / attn.rescale_output_factor
        reference_lane_out = reference_lane_out / attn.rescale_output_factor
        return torch.cat([target_out, reference_lane_out], dim=0)
