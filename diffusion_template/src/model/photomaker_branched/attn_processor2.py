"""
### Modified to make attn_processor trainable in branched version ###
attn_processor2.py - Branched attention processors with consistent batch handling
and explicitly registered trainable parameters created in __init__.
### Modified to make attn_processor trainable in branched version ###
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class BranchedAttnProcessor(nn.Module):
    """
    Self-attention processor with face/background branching.
    Expects doubled batch: [noise_batch, reference_batch]
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        equalize_face_kv: bool = True,
        equalize_clip = (1/3, 8.0),
        #
        # ### Modified to make attn_processor trainable in branched version ###
        # Explicitly register a trainable projection for ID features so that
        # the processor owns parameters visible to the optimizer.
        id_embeds_dim: int = 2048,
        # ### Modified to make attn_processor trainable in branched version ###
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale

        self.mask = None
        self.mask_ref = None

        self.equalize_face_kv = equalize_face_kv
        self.equalize_clip = equalize_clip
        # Runtime-tunable identity/pose controls
        self.pose_adapt_ratio: float = 0.25
        self.ca_mixing_for_face: bool = True
        self.use_id_embeds: bool = True

        # Optional: ID feature cache (runtime data)
        self.id_embeds = None

        # ### Modified to make attn_processor trainable in branched version ###
        # Trainable projection from 2048-D ID features -> hidden_size used in mixing.
        self.id_to_hidden = nn.Linear(id_embeds_dim, hidden_size, bias=False)
        with torch.no_grad():
            self.id_to_hidden.weight.mul_(0.1)
        # ### Modified to make attn_processor trainable in branched version ###

        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        """Set masks for current denoising step"""
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
        id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Process self-attention with face/background branching.

        Input: doubled batch [noise_hidden, ref_hidden]
        Output: doubled batch [merged_hidden, face_hidden]
        """

        full_debug = False

        residual = hidden_states

        # Handle spatial norm
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # Handle 4D input
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Split doubled batch
        total_batch = hidden_states.shape[0]
        half_batch = total_batch // 2
        noise_hidden = hidden_states[:half_batch]
        ref_hidden = hidden_states[half_batch:]

        batch_size = half_batch
        seq_len = noise_hidden.shape[1]

        # Handle group norm
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)

        # Compute queries from noise
        query = attn.to_q(noise_hidden)

        # Reshape for multi-head attention
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim
        q = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # Prepare mask (gate for queries) and a token-aligned version for hidden mixing
        mask_gate = None
        mask_tokens = None
        if self.mask is not None:
            mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)
            mask_gate = mask_gate.to(dtype=q.dtype, device=q.device)            # [B,1,L,1]
            # token-aligned mask [B,L,1] for mixing hidden states
            mask_tokens = mask_gate.squeeze(1).squeeze(-1)
            if mask_tokens.dim() == 2:
                mask_tokens = mask_tokens.unsqueeze(-1)

        # === BACKGROUND BRANCH ===
        # Q: background from noise, K/V: full noise
        key_bg = attn.to_k(noise_hidden)
        value_bg = attn.to_v(noise_hidden)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Gate queries for background (inverse of face gate)
        if mask_gate is not None:
            q_bg = q * (1 - mask_gate)
        else:
            q_bg = q

        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])

        # === FACE BRANCH ===
        if self.mask_ref is not None:
            ref_mask = self._prepare_mask(self.mask_ref, seq_len, batch_size)
            ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
            ref_mask_flat = ref_mask.squeeze(1).squeeze(-1)
            if ref_mask_flat.dim() == 2:
                ref_mask_flat = ref_mask_flat.unsqueeze(-1)

            noise_face_hidden = noise_hidden * (mask_tokens if mask_tokens is not None else 1)
            ref_face_hidden = ref_hidden * ref_mask_flat

            # Blend for pose/identity handoff
            POSE_ADAPT_RATIO = getattr(self, "pose_adapt_ratio", 0.25)
            face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden

            # ### Modified to make attn_processor trainable in branched version ###
            # Inject ID features through registered projection to participate in training.
            if self.id_embeds is not None:
                id_features = self.id_to_hidden(self.id_embeds)
                if id_features.dim() == 2:
                    id_features = id_features.unsqueeze(1).expand(-1, face_hidden_mixed.shape[1], -1)
                id_alpha = 0.3
                face_hidden_mixed = face_hidden_mixed * (1 - id_alpha) + id_features * id_alpha
            # ### Modified to make attn_processor trainable in branched version ###

            CA_MIXING_FOR_FACE = getattr(self, "ca_mixing_for_face", True)
            if CA_MIXING_FOR_FACE:
                combined_face_hidden = torch.cat([face_hidden_mixed, noise_face_hidden], dim=1)
            else:
                combined_face_hidden = face_hidden_mixed
        else:
            raise ValueError("Branched attention requires a reference mask for face branch")

        key_face = attn.to_k(combined_face_hidden)
        value_face = attn.to_v(combined_face_hidden)
        key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        if mask_gate is not None:
            q_face = q * mask_gate
        else:
            q_face = q

        hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])

        # Concat for downstream split: [merged(bg+face), face_only]
        hidden_states = torch.cat([hidden_bg, hidden_face], dim=0)

        # Output projection + residual
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops."""
        H = int(math.sqrt(target_len))
        W = H
        assert H * W == target_len, f"seq_len {target_len} is not square"

        if mask.ndim == 4:  # [B, C, H0, W0]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode="bilinear", align_corners=False)
        else:
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = int(math.sqrt(L0))
            w0 = h0
            assert h0 * w0 == L0, f"mask length {L0} not square"
            m2d = mask.view(mask.shape[0], -1).float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode="bilinear", align_corners=False)

        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]

        if m.shape[0] != batch_size:
            m = m.expand(batch_size, -1, -1)

        return m.view(batch_size, 1, target_len, 1)


class BranchedCrossAttnProcessor(nn.Module):
    """
    Simplified cross-attention processor with branching.
    Only processes the first half (noise batch) with branching.
    Second half (reference batch) gets standard processing.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
        num_tokens: int = 77,
        # ### Modified to make attn_processor trainable in branched version ###
        id_embeds_dim: int = 2048,
        # ### Modified to make attn_processor trainable in branched version ###
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.mask = None
        self.mask_ref = None

        # Optional ID path (kept for parity; not strictly required for training
        # but defined here to keep module parameterization symmetric if extended)
        # ### Modified to make attn_processor trainable in branched version ###
        self.id_to_hidden = nn.Linear(id_embeds_dim, hidden_size, bias=False)
        with torch.no_grad():
            self.id_to_hidden.weight.mul_(0.1)
        # ### Modified to make attn_processor trainable in branched version ###

        self.id_embeds = None

        self.has_cross_attention_kwargs = True

    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
        id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        total_batch = hidden_states.shape[0]
        half_batch = total_batch // 2

        noise_hidden = hidden_states[:half_batch]
        ref_hidden = hidden_states[half_batch:]

        if encoder_hidden_states is not None:
            gen_prompt = encoder_hidden_states[:half_batch]
            face_prompt = encoder_hidden_states[half_batch:]
        else:
            raise ValueError("Branched cross-attention requires encoder_hidden_states")

        batch_size = half_batch
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim

        # Q from noise / ref
        q_bg = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        q_ref = attn.to_q(ref_hidden).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # K/V from text prompts
        key_bg = attn.to_k(gen_prompt).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = attn.to_v(gen_prompt).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])

        key_ref = attn.to_k(face_prompt).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = attn.to_v(face_prompt).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        hidden_ref = F.scaled_dot_product_attention(q_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])

        hidden_states = torch.cat([hidden_bg, hidden_ref], dim=0)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        H = int(math.sqrt(target_len))
        W = H
        assert H * W == target_len, f"seq_len {target_len} is not square"

        if mask.ndim == 4:  # [B, C, H0, W0]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode="bilinear", align_corners=False)
        else:
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = int(math.sqrt(L0))
            w0 = h0
            assert h0 * w0 == L0, f"mask length {L0} not square"
            m2d = mask.view(mask.shape[0], -1).float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode="bilinear", align_corners=False)

        m = m2d.flatten(2).transpose(1, 2)

        if m.shape[0] != batch_size:
            m = m.expand(batch_size, -1, -1)

        return m.view(batch_size, 1, target_len, 1)
