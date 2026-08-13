"""CL14_CA's bounded residual PhotoMaker-ID cross-attention."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"gate probability must be in (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


class _ResidualLoRALinear(nn.Module):
    """Low-rank delta with no base projection and a zero-init output."""

    def __init__(self, features: int, rank: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(features, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.linear(
            F.linear(hidden_states.to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return hidden_states.to(dtype)


class ResidualIdentityCrossAttnProcessorV3(nn.Module):
    """Keep native CA intact and add a face-local, zero-init ID residual."""

    is_residual_identity_ca_v3 = True
    has_cross_attention_kwargs = True

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int = 64,
        gate_init: float = 0.02,
        gate_max: float = 0.20,
        rms_epsilon: float = 1.0e-6,
        trainable_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim)
        self.rank = int(rank)
        self.gate_max = float(gate_max)
        self.rms_epsilon = float(rms_epsilon)
        if self.rank <= 0:
            raise ValueError("Residual identity-CA rank must be positive")
        if not 0.0 < float(gate_init) < self.gate_max <= 1.0:
            raise ValueError("Residual identity-CA requires 0 < gate_init < gate_max <= 1")
        if self.rms_epsilon <= 0.0:
            raise ValueError("rms_epsilon must be positive")

        self.id_delta_out = _ResidualLoRALinear(
            self.hidden_size, self.rank, trainable_dtype
        )
        self.gate_logit = nn.Parameter(torch.tensor(
            _logit(float(gate_init) / self.gate_max), dtype=trainable_dtype
        ))
        self.mask: Optional[torch.Tensor] = None
        self.class_tokens_mask: Optional[torch.Tensor] = None
        self.identity_token_indices: Optional[torch.Tensor] = None
        self._latest_telemetry: dict[str, torch.Tensor] = {}

    def init_from_attention(self, attn) -> None:
        del attn  # Native frozen Q/K/V/out remain the complete base CA path.

    def named_ba_trainables(self):
        for name, parameter in self.id_delta_out.named_parameters():
            yield f"id_delta_out.{name}", parameter, "residual_identity_ca_r64"
        yield "gate_logit", self.gate_logit, "residual_identity_ca_r64"

    def set_masks(
        self,
        mask: Optional[torch.Tensor],
        mask_ref: Optional[torch.Tensor] = None,
    ) -> None:
        del mask_ref
        self.mask = mask

    def set_class_tokens_mask(
        self,
        class_tokens_mask: Optional[torch.Tensor],
        identity_token_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.class_tokens_mask = class_tokens_mask
        self.identity_token_indices = identity_token_indices

    def latest_ba_telemetry(self) -> dict[str, torch.Tensor]:
        return dict(self._latest_telemetry)

    @staticmethod
    def _project_attention(
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        *,
        query_projection,
        key_projection,
        value_projection,
        heads: int,
    ) -> torch.Tensor:
        query = query_projection(query_states)
        key = key_projection(key_value_states)
        value = value_projection(key_value_states)
        inner_dim = int(query.shape[-1])
        if inner_dim % heads:
            raise RuntimeError(
                f"Identity-CA inner dim {inner_dim} is not divisible by {heads} heads"
            )
        head_dim = inner_dim // heads
        query = query.view(query.shape[0], -1, heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, heads, head_dim).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return hidden.transpose(1, 2).reshape(query_states.shape[0], -1, inner_dim)

    def _prepare_spatial_mask(
        self,
        *,
        target_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mask is None:
            raise RuntimeError("Residual identity CA requires a target face mask")
        side = int(math.sqrt(target_len))
        if side * side != target_len:
            raise RuntimeError(f"Identity CA requires square tokens, got {target_len}")
        mask = self.mask
        if mask.ndim == 4:
            mask_2d = mask[:, :1].float()
        else:
            flat_len = int(mask.reshape(mask.shape[0], -1).shape[1])
            source_side = int(math.sqrt(flat_len))
            if source_side * source_side != flat_len:
                raise RuntimeError(f"Identity CA mask length is not square: {flat_len}")
            mask_2d = mask.reshape(
                mask.shape[0], 1, source_side, source_side
            ).float()
        flat = F.interpolate(mask_2d, (side, side), mode="nearest")
        flat = flat.flatten(2).transpose(1, 2)
        if flat.shape[0] != batch_size:
            if batch_size % flat.shape[0]:
                raise RuntimeError("Identity CA target-mask batch mismatch")
            flat = flat.repeat(batch_size // flat.shape[0], 1, 1)
        return (flat > 0.5).to(device=device, dtype=dtype)

    def _gather_identity_tokens(
        self, identity_prompt: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.identity_token_indices
        if indices is not None:
            if indices.ndim != 2 or indices.shape[1] == 0:
                raise RuntimeError("Identity-token indices must be a nonempty 2D tensor")
            if indices.shape[0] != batch_size:
                if batch_size % indices.shape[0]:
                    raise RuntimeError("Identity-token index batch mismatch")
                indices = indices.repeat(batch_size // indices.shape[0], 1)
            indices = indices.to(identity_prompt.device, dtype=torch.long)
            gathered = torch.gather(
                identity_prompt,
                1,
                indices.unsqueeze(-1).expand(-1, -1, identity_prompt.shape[-1]),
            )
            counts = torch.full(
                (batch_size,), indices.shape[1], device=identity_prompt.device,
                dtype=torch.long,
            )
            return gathered, counts

        if self.class_tokens_mask is None:
            raise RuntimeError("Residual identity CA requires class_tokens_mask")
        token_mask = self.class_tokens_mask.to(identity_prompt.device, dtype=torch.bool)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        if token_mask.ndim != 2 or token_mask.shape[1] != identity_prompt.shape[1]:
            raise RuntimeError("Identity-token mask shape mismatch")
        if token_mask.shape[0] != batch_size:
            if batch_size % token_mask.shape[0]:
                raise RuntimeError("Identity-token mask batch mismatch")
            token_mask = token_mask.repeat(batch_size // token_mask.shape[0], 1)
        counts = token_mask.sum(dim=1)
        if bool((counts <= 0).any()) or int(torch.unique(counts).numel()) != 1:
            raise RuntimeError("Identity CA requires equal, nonzero ID-token counts")
        active_count = int(counts[0].item())
        gathered = identity_prompt[token_mask].reshape(
            batch_size, active_count, identity_prompt.shape[-1]
        )
        return gathered, counts

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

        target_hidden, reference_hidden = hidden_states.split(batch_size, dim=0)
        generation_prompt, identity_prompt = encoder_hidden_states.split(
            batch_size, dim=0
        )
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

        # 13 Aug 2026 - CL14_CA-PERF-01: target/reference native CA rows are
        # independent, so one fused projection/SDPA call preserves the equation.
        native_output = self._project_attention(
            torch.cat([target_hidden, reference_hidden]),
            torch.cat([generation_prompt, identity_prompt]),
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=int(attn.heads),
        )
        native_output = attn.to_out[1](attn.to_out[0](native_output))
        native_target, native_reference = native_output.split(batch_size, dim=0)

        identity_tokens, token_counts = self._gather_identity_tokens(
            identity_prompt, batch_size
        )
        identity_hidden = self._project_attention(
            target_hidden,
            identity_tokens,
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=int(attn.heads),
        )
        identity_delta = self.id_delta_out(identity_hidden)
        # The clamp prevents sqrt'(0) from producing NaN on the zero-init step.
        delta_rms = identity_delta.float().square().mean(
            dim=-1, keepdim=True
        ).clamp_min(self.rms_epsilon**2).sqrt()
        normalized_delta = identity_delta / delta_rms.to(identity_delta.dtype)
        gate = torch.sigmoid(self.gate_logit) * self.gate_max
        target_mask = self._prepare_spatial_mask(
            target_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=native_target.device,
            dtype=native_target.dtype,
        )
        # 13 Aug 2026 - AICODE-NOTE: CL14_CA-CORE-01 keeps native
        # PhotoMaker/text CA complete; this bounded face-local ID term is the
        # only scientific delta.
        residual_message = (
            target_mask * gate.to(native_target.dtype)
            * normalized_delta.to(native_target.dtype)
        )
        hidden_states = torch.cat(
            [native_target + residual_message, native_reference], dim=0
        )

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
