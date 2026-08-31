"""Raw InsightFace identity-token residual CA expert for CL39-X07."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..identity_ca_processor_v2 import HardIdentityCrossAttnProcessorV2
from ..residual_sa_processor_v2 import ResidualLoRALinear, _logit


class LowRankLinear(nn.Module):
    """Small rectangular low-rank projection with no frozen base term."""

    def __init__(self, input_dim: int, output_dim: int, rank: int, *, dtype):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, input_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(output_dim, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = F.linear(
            F.linear(hidden_states.to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return value.to(hidden_states.dtype)


class IntrinsicIDTokenProjector(nn.Module):
    def __init__(self, input_dim=512, token_dim=2048, num_tokens=4, hidden_dim=2048):
        super().__init__()
        self.input_dim, self.token_dim, self.num_tokens = int(input_dim), int(token_dim), int(num_tokens)
        self.in_proj = nn.Linear(self.input_dim, int(hidden_dim))
        self.out_proj = nn.Linear(int(hidden_dim), self.token_dim * self.num_tokens, bias=False)
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 3:
            embedding = embedding[:, 0]
        if embedding.ndim != 2 or embedding.shape[-1] != self.input_dim:
            raise ValueError(f"Intrinsic ID embedding must be [B,{self.input_dim}]")
        valid = embedding.float().square().sum(-1, keepdim=True).gt(0)
        normalized = F.normalize(embedding.float(), dim=-1).to(self.in_proj.weight.dtype)
        tokens = self.out_proj(F.gelu(self.in_proj(normalized)))
        tokens = self.norm(tokens.view(-1, self.num_tokens, self.token_dim))
        return tokens * valid[:, None].to(tokens.dtype)


class IntrinsicIdentityCrossAttnProcessor(HardIdentityCrossAttnProcessorV2):
    is_intrinsic_identity_ca = True

    def __init__(self, *, hidden_size: int, cross_attention_dim: int, rank: int,
                 gate_init: float, gate_max: float,
                 confidence_source: str = "none",
                 trainable_dtype=torch.float32):
        nn.Module.__init__(self)
        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim)
        self.id_delta_out = ResidualLoRALinear(
            self.hidden_size, rank=int(rank), dtype=trainable_dtype, zero_init_output=True
        )
        self.confidence_source = str(confidence_source)
        self.id_to_k = None
        self.id_to_v = None
        if self.confidence_source == "cl39_complement_detached":
            self.id_to_k = LowRankLinear(
                self.cross_attention_dim, self.hidden_size, int(rank),
                dtype=trainable_dtype,
            )
            self.id_to_v = LowRankLinear(
                self.cross_attention_dim, self.hidden_size, int(rank),
                dtype=trainable_dtype,
            )
        elif self.confidence_source != "none":
            raise ValueError("Unknown intrinsic-ID confidence source")
        self.gate_max = float(gate_max)
        self.gate_logit = nn.Parameter(torch.tensor(_logit(float(gate_init) / self.gate_max), dtype=trainable_dtype))
        self.mask: Optional[torch.Tensor] = None
        self.mask_ref: Optional[torch.Tensor] = None
        self.intrinsic_id_tokens: Optional[torch.Tensor] = None
        self.has_cross_attention_kwargs = True
        self._latest_telemetry = {}
        self._route_source = None
        self.extension_telemetry_enabled = True

    def init_from_attention(self, attn) -> None:
        del attn

    def set_intrinsic_id_tokens(self, tokens) -> None:
        self.intrinsic_id_tokens = tokens

    def named_ba_trainables(self):
        role = (
            "intrinsic_id_residual_ca_r32"
            if self.confidence_source != "none"
            else "intrinsic_id_residual_ca_r64"
        )
        for prefix, module in (("id_to_k", self.id_to_k), ("id_to_v", self.id_to_v)):
            if module is not None:
                for name, parameter in module.named_parameters():
                    yield f"{prefix}.{name}", parameter, "intrinsic_id_residual_ca_r32"
        for name, parameter in self.id_delta_out.named_parameters():
            yield f"id_delta_out.{name}", parameter, role
        yield "gate_logit", self.gate_logit, role

    def latest_ba_telemetry(self):
        return self._latest_telemetry

    def set_route_source(self, source) -> None:
        self._route_source = source

    def set_extension_telemetry_enabled(self, enabled: bool) -> None:
        self.extension_telemetry_enabled = bool(enabled)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale=1.0,
                 cross_attention_kwargs=None):
        del attention_mask, scale, cross_attention_kwargs
        if encoder_hidden_states is None:
            raise RuntimeError("Intrinsic-ID sidecar requires native CA context")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total, channel, height * width).transpose(1, 2)
        else:
            total = hidden_states.shape[0]
            channel = height = width = None
        if total % 2:
            raise RuntimeError("Intrinsic-ID sidecar requires doubled batches")
        batch = total // 2
        normalized = hidden_states
        if attn.group_norm is not None:
            normalized = attn.group_norm(normalized.transpose(1, 2)).transpose(1, 2)
        context = encoder_hidden_states
        if getattr(attn, "norm_cross", False):
            context = attn.norm_encoder_hidden_states(context)
        native = self._project_attention(
            normalized, context, query_projection=attn.to_q, key_projection=attn.to_k,
            value_projection=attn.to_v, heads=int(attn.heads),
        )
        native = attn.to_out[1](attn.to_out[0](native))
        native_target, native_reference = native.split(batch)
        tokens = self.intrinsic_id_tokens
        if tokens is None:
            delta = torch.zeros_like(native_target)
            active = native_target.new_zeros(())
            self._latest_telemetry = {
                "intrinsic_id/gate": active,
                "intrinsic_id/delta_rms": active,
                "intrinsic_id/token_norm": active,
                "intrinsic_id/active_fraction": active,
            }
        else:
            tokens = tokens.to(device=normalized.device, dtype=context.dtype)
            if tokens.shape[0] != batch and batch % tokens.shape[0] == 0:
                tokens = tokens.repeat(batch // tokens.shape[0], 1, 1)
            if tokens.shape[0] != batch:
                raise RuntimeError("Intrinsic-ID token batch mismatch")
            key_projection = self.id_to_k or attn.to_k
            value_projection = self.id_to_v or attn.to_v
            message = self._project_attention(
                normalized[:batch], tokens, query_projection=attn.to_q,
                key_projection=key_projection, value_projection=value_projection,
                heads=int(attn.heads),
            )
            raw_delta = self.id_delta_out(message)
            rms = (raw_delta.float().square().mean(-1, keepdim=True) + 1.0e-6).sqrt()
            normalized_delta = raw_delta / rms.to(raw_delta.dtype)
            mask = self._prepare_spatial_mask(
                target_len=normalized.shape[1], batch_size=batch,
                device=normalized.device, dtype=normalized.dtype,
            )
            low_confidence_usage = mask.new_zeros(())
            if self.confidence_source == "cl39_complement_detached":
                if self._route_source is None:
                    raise RuntimeError("CL39N9 has no paired self-attention route source")
                confidence, face_router = self._route_source.latest_route_context()
                if confidence is None or face_router is None:
                    raise RuntimeError("CL39N9 route context was not populated by attn1")
                confidence = confidence.to(mask.device, mask.dtype)
                face_router = face_router.to(mask.device, mask.dtype)
                if confidence.shape != mask.shape or face_router.shape != mask.shape:
                    raise RuntimeError("CL39N9 route-context shape mismatch")
                mask = face_router * (1.0 - confidence.detach())
                low_confidence_usage = mask.float().mean()
            gate = torch.sigmoid(self.gate_logit) * self.gate_max
            delta = mask * gate.to(normalized.dtype) * normalized_delta.to(normalized.dtype)
            active = tokens.float().square().sum(-1).gt(0).float().mean()
            if self.extension_telemetry_enabled:
                self._latest_telemetry = {}
                native_rms = native_target.detach().float().square().mean().sqrt()
                delta_rms = delta.detach().float().square().mean().sqrt()
                self._latest_telemetry = {
                    "intrinsic_id/gate": gate.detach().float(),
                    "intrinsic_id/delta_rms": delta_rms,
                    "intrinsic_id/residual_native_ratio": delta_rms / native_rms.clamp_min(1.0e-6),
                    "intrinsic_id/low_confidence_usage": low_confidence_usage.detach(),
                    "intrinsic_id/token_norm": tokens.detach().float().norm(dim=-1).mean(),
                    "intrinsic_id/active_fraction": active.detach(),
                }
        output = torch.cat((native_target + delta, native_reference))
        if input_ndim == 4:
            output = output.transpose(-1, -2).reshape(total, channel, height, width)
        if attn.residual_connection:
            output = output + residual
        return output / attn.rescale_output_factor
