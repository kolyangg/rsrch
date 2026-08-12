"""Corrected hard face-local identity-token cross-attention for E12."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_processor_cleanest import _clone_effective_linear


class HardIdentityCrossAttnProcessorV2(nn.Module):
    """Use target queries and gathered PhotoMaker ID tokens inside the face.

    Native frozen text/PhotoMaker cross-attention owns the target exterior and
    the complete reference lane. The branch-local ID path owns the target face;
    there is no native/ID interpolation, residual gate, or legacy ref-query CA.
    """

    is_identity_ca_v2 = True

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int,
        trainable_dtype=None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cross_attention_dim = int(cross_attention_dim)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("Hard identity-CA rank must be positive")
        self.trainable_dtype = trainable_dtype

        self.id_to_q = None
        self.id_to_k = None
        self.id_to_v = None
        self.id_to_out = None
        self.mask: Optional[torch.Tensor] = None
        self.class_tokens_mask: Optional[torch.Tensor] = None
        self.identity_token_indices: Optional[torch.Tensor] = None
        self.has_cross_attention_kwargs = True
        self._latest_telemetry: dict[str, torch.Tensor] = {}

    def init_from_attention(self, attn) -> None:
        self.id_to_q = _clone_effective_linear(
            attn.to_q,
            kind="lora",
            rank=self.rank,
            trainable_dtype=self.trainable_dtype,
        )
        self.id_to_k = _clone_effective_linear(
            attn.to_k,
            kind="lora",
            rank=self.rank,
            trainable_dtype=self.trainable_dtype,
        )
        self.id_to_v = _clone_effective_linear(
            attn.to_v,
            kind="lora",
            rank=self.rank,
            trainable_dtype=self.trainable_dtype,
        )
        self.id_to_out = _clone_effective_linear(
            attn.to_out[0],
            kind="lora",
            rank=self.rank,
            trainable_dtype=self.trainable_dtype,
        )

    def named_ba_trainables(self):
        roles = (
            ("id_to_q", self.id_to_q, "identity_ca_query"),
            ("id_to_k", self.id_to_k, "identity_ca_kv"),
            ("id_to_v", self.id_to_v, "identity_ca_kv"),
            ("id_to_out", self.id_to_out, "identity_ca_output"),
        )
        for prefix, module, role in roles:
            if module is None:
                raise RuntimeError("Identity-CA processor was not initialized")
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter, role

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
        if inner_dim % heads != 0:
            raise RuntimeError(
                f"Identity-CA inner dim {inner_dim} is not divisible by {heads} heads"
            )
        head_dim = inner_dim // heads
        query = query.view(query.shape[0], -1, heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, heads, head_dim).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
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
            raise RuntimeError("Hard identity CA requires a target face mask")
        side = int(math.sqrt(target_len))
        if side * side != target_len:
            raise RuntimeError(
                f"Hard identity CA requires square spatial tokens, got {target_len}"
            )
        mask = self.mask
        if mask.ndim == 4:
            mask_2d = mask[:, :1].float()
        else:
            flat_len = int(mask.reshape(mask.shape[0], -1).shape[1])
            source_side = int(math.sqrt(flat_len))
            if source_side * source_side != flat_len:
                raise RuntimeError(
                    f"Hard identity CA mask length is not square: {flat_len}"
                )
            mask_2d = mask.reshape(mask.shape[0], 1, source_side, source_side).float()
        mask_2d = F.interpolate(mask_2d, size=(side, side), mode="nearest")
        flat = mask_2d.flatten(2).transpose(1, 2)
        if flat.shape[0] != batch_size:
            if batch_size % flat.shape[0] != 0:
                raise RuntimeError(
                    "Hard identity CA mask batch mismatch: "
                    f"mask={flat.shape[0]}, target={batch_size}"
                )
            flat = flat.repeat(batch_size // flat.shape[0], 1, 1)
        return (flat > 0.5).to(device=device, dtype=dtype)

    def _expanded_token_mask(
        self,
        *,
        batch_size: int,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.class_tokens_mask is None:
            raise RuntimeError(
                "Hard identity CA requires class_tokens_mask; refusing face-text fallback"
            )
        mask = self.class_tokens_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or mask.shape[1] != token_count:
            raise RuntimeError(
                "Hard identity CA token-mask shape mismatch: "
                f"mask={tuple(mask.shape)}, expected=(*, {token_count})"
            )
        if mask.shape[0] != batch_size:
            if batch_size % mask.shape[0] != 0:
                raise RuntimeError(
                    "Hard identity CA token-mask batch mismatch: "
                    f"mask={mask.shape[0]}, target={batch_size}"
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1)
        return mask

    def _gather_identity_tokens(
        self,
        identity_prompt: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.identity_token_indices
        if indices is not None:
            if indices.ndim != 2 or indices.shape[1] == 0:
                raise RuntimeError(
                    "Identity-token indices must be a nonempty 2D tensor"
                )
            if indices.shape[0] != batch_size:
                if batch_size % indices.shape[0] != 0:
                    raise RuntimeError("Identity-token index batch mismatch")
                indices = indices.repeat(batch_size // indices.shape[0], 1)
            # 12 Aug 2026 - Training optimization: fixed-shape gather avoids
            # boolean-index shape discovery and repeated CUDA synchronizations.
            indices = indices.to(device=identity_prompt.device, dtype=torch.long)
            gathered = torch.gather(
                identity_prompt,
                dim=1,
                index=indices.unsqueeze(-1).expand(
                    -1, -1, identity_prompt.shape[-1]
                ),
            )
            counts = torch.full(
                (batch_size,),
                indices.shape[1],
                device=identity_prompt.device,
                dtype=torch.long,
            )
            return gathered, counts

        token_mask = self._expanded_token_mask(
            batch_size=batch_size,
            token_count=identity_prompt.shape[1],
            device=identity_prompt.device,
        )
        token_counts = token_mask.sum(dim=1)
        if bool((token_counts <= 0).any()) or int(
            torch.unique(token_counts).numel()
        ) != 1:
            raise RuntimeError(
                "Identity CA requires equal, nonzero active ID-token counts"
            )
        active_count = int(token_counts[0].item())
        return (
            identity_prompt[token_mask].reshape(
                batch_size, active_count, identity_prompt.shape[-1]
            ),
            token_counts,
        )

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
            raise RuntimeError("Hard identity CA requires encoder hidden states")
        if any(
            module is None
            for module in (self.id_to_q, self.id_to_k, self.id_to_v, self.id_to_out)
        ):
            raise RuntimeError("Hard identity-CA processor was not initialized")

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

        if total_batch % 2 != 0 or encoder_hidden_states.shape[0] % 2 != 0:
            raise RuntimeError("Hard identity CA requires doubled even batches")
        batch_size = total_batch // 2
        if encoder_hidden_states.shape[0] // 2 != batch_size:
            raise RuntimeError(
                "Hard identity CA latent/prompt batch mismatch: "
                f"latent={total_batch}, prompt={encoder_hidden_states.shape[0]}"
            )

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

        heads = int(attn.heads)
        native_target = self._project_attention(
            target_hidden,
            generation_prompt,
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=heads,
        )
        native_reference = self._project_attention(
            reference_hidden,
            identity_prompt,
            query_projection=attn.to_q,
            key_projection=attn.to_k,
            value_projection=attn.to_v,
            heads=heads,
        )
        native_target = attn.to_out[1](attn.to_out[0](native_target))
        native_reference = attn.to_out[1](attn.to_out[0](native_reference))

        gathered_identity, token_counts = self._gather_identity_tokens(
            identity_prompt,
            batch_size,
        )
        identity_message = self._project_attention(
            target_hidden,
            gathered_identity,
            query_projection=self.id_to_q,
            key_projection=self.id_to_k,
            value_projection=self.id_to_v,
            heads=heads,
        )
        identity_message = self.id_to_out(identity_message)
        identity_message = attn.to_out[1](identity_message)

        target_mask = self._prepare_spatial_mask(
            target_len=target_hidden.shape[1],
            batch_size=batch_size,
            device=native_target.device,
            dtype=native_target.dtype,
        )
        target_output = (
            native_target * (1.0 - target_mask)
            + identity_message.to(dtype=native_target.dtype) * target_mask
        )
        hidden_states = torch.cat([target_output, native_reference], dim=0)

        # 4 Aug 2026 - AICODE-NOTE: E12 is a hard spatial CA branch. The ID
        # message exclusively owns the target face; no PhotoMaker/native face
        # output, alpha, gate, or residual blend is permitted here.
        with torch.no_grad():
            face_denom = target_mask.float().sum().clamp_min(1.0)
            self._latest_telemetry = {
                "identity_ca_token_count": token_counts.float().mean().detach(),
                "identity_ca_message_rms": identity_message.float().square().mean().sqrt().detach(),
                "identity_ca_native_face_rms": (
                    (native_target.float().square() * target_mask.float()).sum()
                    / (face_denom * native_target.shape[-1])
                ).sqrt().detach(),
            }

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
