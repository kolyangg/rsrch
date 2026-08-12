"""Bounded residual PhotoMaker-ID cross-attention for E17."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .identity_ca_processor_v2 import HardIdentityCrossAttnProcessorV2
from .residual_sa_processor_v2 import ResidualLoRALinear, _logit


class ResidualIdentityCrossAttnProcessorV3(HardIdentityCrossAttnProcessorV2):
    """Keep native CA intact and add a face-local, zero-init ID residual."""

    is_identity_ca_v2 = False
    is_residual_identity_ca_v3 = True

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int,
        gate_init: float = 0.02,
        gate_max: float = 0.20,
        rms_epsilon: float = 1.0e-6,
        trainable_dtype=torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim)
        self.rank = int(rank)
        self.gate_init = float(gate_init)
        self.gate_max = float(gate_max)
        self.rms_epsilon = float(rms_epsilon)
        if self.rank <= 0:
            raise ValueError("Residual identity-CA rank must be positive")
        if not 0.0 < self.gate_init < self.gate_max <= 1.0:
            raise ValueError("Residual identity-CA requires 0 < gate_init < gate_max <= 1")
        if self.rms_epsilon <= 0.0:
            raise ValueError("rms_epsilon must be positive")

        self.id_delta_out = ResidualLoRALinear(
            self.hidden_size,
            rank=self.rank,
            dtype=trainable_dtype,
            zero_init_output=True,
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(
                _logit(self.gate_init / self.gate_max),
                dtype=trainable_dtype,
            )
        )
        self.mask: Optional[torch.Tensor] = None
        self.class_tokens_mask: Optional[torch.Tensor] = None
        self.has_cross_attention_kwargs = True
        self._latest_telemetry: dict[str, torch.Tensor] = {}

    def init_from_attention(self, attn) -> None:
        del attn

    def named_ba_trainables(self):
        for name, parameter in self.id_delta_out.named_parameters():
            yield f"id_delta_out.{name}", parameter, "identity_ca_output"
        yield "gate_logit", self.gate_logit, "identity_ca_gate"

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        del attention_mask, scale, cross_attention_kwargs
        if encoder_hidden_states is None:
            raise RuntimeError("Residual identity CA requires encoder hidden states")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch, channel, height * width
            ).transpose(1, 2)
        else:
            total_batch = hidden_states.shape[0]
            channel = height = width = None
        if total_batch % 2 or encoder_hidden_states.shape[0] % 2:
            raise RuntimeError("Residual identity CA requires doubled even batches")
        batch_size = total_batch // 2
        if encoder_hidden_states.shape[0] // 2 != batch_size:
            raise RuntimeError("Residual identity CA latent/prompt batch mismatch")

        target_hidden = hidden_states[:batch_size]
        reference_hidden = hidden_states[batch_size:]
        generation_prompt = encoder_hidden_states[:batch_size]
        identity_prompt = encoder_hidden_states[batch_size:]
        if attn.group_norm is not None:
            target_hidden = attn.group_norm(
                target_hidden.transpose(1, 2)
            ).transpose(1, 2)
            reference_hidden = attn.group_norm(
                reference_hidden.transpose(1, 2)
            ).transpose(1, 2)
        if getattr(attn, "norm_cross", False):
            generation_prompt = attn.norm_encoder_hidden_states(generation_prompt)
            identity_prompt = attn.norm_encoder_hidden_states(identity_prompt)

        # 12 Aug 2026 - Training optimization: target and reference native CA
        # are independent batch rows, so one doubled-batch projection/SDPA is
        # mathematically identical and avoids a second set of kernel launches.
        native_hidden = torch.cat([target_hidden, reference_hidden], dim=0)
        native_prompt = torch.cat([generation_prompt, identity_prompt], dim=0)
        native_output = self._project_attention(
            native_hidden,
            native_prompt,
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=int(attn.heads),
        )
        native_output = attn.to_out[1](attn.to_out[0](native_output))
        native_target, native_reference = native_output.split(batch_size, dim=0)

        gathered_identity, token_counts = self._gather_identity_tokens(
            identity_prompt,
            batch_size,
        )
        identity_hidden = self._project_attention(
            target_hidden,
            gathered_identity,
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=int(attn.heads),
        )
        identity_delta = self.id_delta_out(identity_hidden)
        # 5 Aug 2026 - AICODE-NOTE: Clamp the mean square before sqrt. The
        # zero-init output otherwise has a finite forward value but NaN
        # gradients from sqrt'(0), corrupting E17 on its first optimizer step.
        delta_rms = (
            identity_delta.float()
            .square()
            .mean(dim=-1, keepdim=True)
            .clamp_min(self.rms_epsilon**2)
            .sqrt()
        )
        normalized_delta = identity_delta / delta_rms.to(identity_delta.dtype)
        gate = torch.sigmoid(self.gate_logit) * self.gate_max
        target_mask = self._prepare_spatial_mask(
            target_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=native_target.device,
            dtype=native_target.dtype,
        )

        # 5 Aug 2026 - AICODE-NOTE: Native PhotoMaker/text CA remains the
        # complete base path. E17 may only add a bounded face-local ID delta.
        residual_message = (
            target_mask
            * gate.to(native_target.dtype)
            * normalized_delta.to(native_target.dtype)
        )
        target_output = native_target + residual_message
        hidden_states = torch.cat([target_output, native_reference], dim=0)
        with torch.no_grad():
            face_elements = (
                target_mask.float().sum() * native_target.shape[-1]
            ).clamp_min(1.0)
            native_face_rms = (
                (native_target.float().square() * target_mask.float()).sum()
                / face_elements
            ).sqrt()
            residual_face_rms = (
                residual_message.float().square().sum() / face_elements
            ).sqrt()
            self._latest_telemetry = {
                "identity_ca_token_count": token_counts.float().mean().detach(),
                "identity_ca_delta_rms": delta_rms.float().mean().detach(),
                "identity_ca_gate": gate.float().detach(),
                "identity_ca_native_face_rms": native_face_rms.detach(),
                "identity_ca_residual_face_rms": residual_face_rms.detach(),
                "identity_ca_residual_native_ratio": (
                    residual_face_rms / native_face_rms.clamp_min(1.0e-6)
                ).detach(),
            }

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
