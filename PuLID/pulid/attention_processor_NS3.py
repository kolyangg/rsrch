"""
attention_processor_NS2.py - Branched attention processors for PuLID
Adapted from PhotoMaker's branched attention implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Import original PuLID settings
from .attention_processor import NUM_ZERO, ORTHO, ORTHO_v2

# Check for PyTorch 2.0+ for scaled_dot_product_attention
if hasattr(F, "scaled_dot_product_attention"):
    HAS_TORCH2 = True
else:
    HAS_TORCH2 = False
    import warnings
    warnings.warn("PyTorch 2.0+ recommended for optimal performance")


class BranchedAttnProcessor_NS2(nn.Module):
    """
    Self-attention processor with face/background branching for PuLID.
    Adapted from PhotoMaker's BranchedAttnProcessor.
    Expects doubled batch: [noise_batch, reference_batch]
    """
    
    def __init__(
        self,
        hidden_size: int = None,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        equalize_face_kv: bool = True,
        equalize_clip: tuple = (1/3, 8.0)
    ):
        super().__init__()
        
        if not HAS_TORCH2:
            raise ImportError("Requires PyTorch 2.0+ for scaled_dot_product_attention")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        
        # Mask storage
        self.mask = None
        self.mask_ref = None
        
        self.equalize_face_kv = equalize_face_kv
        self.equalize_clip = equalize_clip
        
        # Runtime-tunable parameters (matching PhotoMaker)
        self.pose_adapt_ratio: float = 0.25
        self.ca_mixing_for_face: bool = True
        self.use_id_embeds: bool = True
        
        # Optional: ID feature cache for PuLID
        self.id_embeds = None
        self.id_to_hidden = None
        
        # Track if we need cross-attention kwargs
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
        # PuLID specific parameters
        id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Process self-attention with face/background branching.
        Input: doubled batch [noise_hidden, ref_hidden]
        Output: doubled batch [merged_hidden, ref_processed]
        """
        
        if self.mask is not None:
            print(f"[DEBUG] BranchedAttn using mask: shape={self.mask.shape}, mean={self.mask.mean().item():.3f}")
        
        residual = hidden_states
        
        # Handle spatial norm
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        # Handle 4D input
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # Split doubled batch if present; otherwise fallback to standard self-attention
        total_batch = hidden_states.shape[0]
        if total_batch < 2:
            # Fallback: standard self-attention (no branching)
            # Prepare attention mask
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            # Group norm
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            # QKV
            query = attn.to_q(hidden_states)
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            
            # Ensure dtype consistency with output projection
            hidden_states = hidden_states.to(dtype=attn.to_out[0].weight.dtype)

            # Output projection and reshape
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        # Doubled batch path
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
        
        # Prepare mask
        mask_gate = None
        if self.mask is not None:
            mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)
            mask_gate = mask_gate.to(dtype=q.dtype, device=q.device)
        
        # === BACKGROUND BRANCH ===
        # Q: background from noise, K/V: full noise
        key_bg = attn.to_k(noise_hidden)
        value_bg = attn.to_v(noise_hidden)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is not None:
            q_bg = q * (1.0 - mask_gate)  # non-face area
        else:
            q_bg = q
            
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # === FACE BRANCH ===
        # Prepare reference features for face branch
        if self.mask_ref is not None:
            mask_ref_flat = self._prepare_mask(self.mask_ref, seq_len, batch_size)
            mask_ref_flat = mask_ref_flat.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
            ref_face_hidden = ref_hidden * mask_ref_flat.squeeze(-1).unsqueeze(-1)
        else:
            ref_face_hidden = ref_hidden
        
        # Pose adaptation: blend reference and noise for face features
        if self.pose_adapt_ratio > 0:
            if mask_gate is not None:
                noise_face_hidden = noise_hidden * mask_gate.squeeze(-1).unsqueeze(-1)
            else:
                noise_face_hidden = noise_hidden
            
            face_kv_hidden = (
                ref_face_hidden * (1 - self.pose_adapt_ratio) +
                noise_face_hidden * self.pose_adapt_ratio
            )
        else:
            face_kv_hidden = ref_face_hidden
        
        # Optional: mix with full features for K/V
        if self.ca_mixing_for_face:
            face_kv_hidden = torch.cat([face_kv_hidden, noise_hidden * 0.5], dim=1)
        
        key_face = attn.to_k(face_kv_hidden)
        value_face = attn.to_v(face_kv_hidden)
        
        # Optionally equalize K/V magnitudes
        if self.equalize_face_kv and key_face.shape[1] > seq_len:
            k_face_part = key_face[:, :seq_len]
            k_noise_part = key_face[:, seq_len:]
            k_std_ratio = k_face_part.std() / (k_noise_part.std() + 1e-6)
            k_std_ratio = k_std_ratio.clamp(*self.equalize_clip)
            key_face = torch.cat([k_face_part, k_noise_part * k_std_ratio], dim=1)
            
            v_face_part = value_face[:, :seq_len]
            v_noise_part = value_face[:, seq_len:]
            v_std_ratio = v_face_part.std() / (v_noise_part.std() + 1e-6)
            v_std_ratio = v_std_ratio.clamp(*self.equalize_clip)
            value_face = torch.cat([v_face_part, v_noise_part * v_std_ratio], dim=1)
        
        key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is not None:
            q_face = q * mask_gate  # face area
        else:
            q_face = q
            
        hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # === REFERENCE SELF-ATTENTION ===
        query_ref = attn.to_q(ref_hidden)
        key_ref = attn.to_k(ref_hidden)
        value_ref = attn.to_v(ref_hidden)
        
        query_ref = query_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        key_ref = key_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        hidden_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # === MERGE ===
        if mask_gate is not None:
            mask_flat = mask_gate.squeeze(1).squeeze(-1).to(dtype=hidden_bg.dtype)
            if mask_flat.dim() == 1:
                mask_flat = mask_flat.unsqueeze(0)
            if mask_flat.dim() == 2:
                mask_flat = mask_flat.unsqueeze(-1)
            
            merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
        else:
            merged = hidden_bg + hidden_face * self.scale
        
        # Combine: [merged_result, reference_result]
        hidden_states = torch.cat([merged, hidden_ref], dim=0)
        
        # Ensure dtype consistency
        hidden_states = hidden_states.to(dtype=attn.to_out[0].weight.dtype)
        
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )
        
        # Add residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops"""
        if mask.ndim == 2:  # H x W
            H = int(math.sqrt(target_len))
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(H, H),
                mode='bilinear',
                align_corners=False
            )
            mask = mask.flatten(2).transpose(1, 2)  # [1, H*W, 1]
            mask = mask.expand(batch_size, -1, 1)
        elif mask.ndim == 3:  # [1, H, W] or [B, H, W]
            H = int(math.sqrt(target_len))
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1)
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(H, H),
                mode='bilinear',
                align_corners=False
            )
            mask = mask.flatten(2).transpose(1, 2).unsqueeze(-1)  # [B, H*W, 1]
        elif mask.ndim == 4:  # [B, 1, H, W]
            H = int(math.sqrt(target_len))
            mask = F.interpolate(
                mask.float(),
                size=(H, H),
                mode='bilinear',
                align_corners=False
            )
            mask = mask.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        
        return mask


class BranchedCrossAttnProcessor_NS2(nn.Module):
    """
    Cross-attention processor with face/background branching for PuLID.
    Adapted from PhotoMaker's BranchedCrossAttnProcessor.
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        
        if not HAS_TORCH2:
            raise ImportError("Requires PyTorch 2.0+ for scaled_dot_product_attention")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        
        # Mask storage
        self.mask = None
        self.mask_ref = None
        
        # PuLID ID adapter parameters
        self.id_to_k = None
        self.id_to_v = None
        
        # Track if we need cross-attention kwargs
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
        # PuLID specific parameters
        id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Process cross-attention with branching.
        For PuLID, this handles text conditioning with optional ID injection.
        """
        
        residual = hidden_states
        
        # Handle spatial norm
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        # Handle 4D input
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # For cross-attention, we need encoder_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # Split doubled batch if present
        total_batch = hidden_states.shape[0]
        if total_batch % 2 == 0 and self.mask is not None:
            # Branched mode: split batch
            half_batch = total_batch // 2
            noise_hidden = hidden_states[:half_batch]
            ref_hidden = hidden_states[half_batch:]
            
            # Also split encoder states if they're doubled
            if encoder_hidden_states.shape[0] == total_batch:
                gen_prompt = encoder_hidden_states[:half_batch]
                face_prompt = encoder_hidden_states[half_batch:]
            else:
                gen_prompt = encoder_hidden_states
                face_prompt = encoder_hidden_states
            
            # Process noise batch with standard cross-attention
            query_noise = attn.to_q(noise_hidden)
            key_noise = attn.to_k(gen_prompt)
            value_noise = attn.to_v(gen_prompt)
            
            # Add PuLID ID features if available
            if id_embedding is not None and self.id_to_k is not None:
                id_key = self.id_to_k(id_embedding) * id_scale
                id_value = self.id_to_v(id_embedding) * id_scale
                key_noise = torch.cat([key_noise, id_key], dim=1)
                value_noise = torch.cat([value_noise, id_value], dim=1)
            
            query_noise = attn.head_to_batch_dim(query_noise)
            key_noise = attn.head_to_batch_dim(key_noise)
            value_noise = attn.head_to_batch_dim(value_noise)
            
            
            # Ensure dtype consistency
            query_noise = query_noise.to(dtype=key_noise.dtype)
            query_ref = query_ref.to(dtype=key_ref.dtype) if 'query_ref' in locals() else None
            
            attention_probs_noise = attn.get_attention_scores(query_noise, key_noise, attention_mask)
            hidden_noise = torch.bmm(attention_probs_noise, value_noise)
            hidden_noise = attn.batch_to_head_dim(hidden_noise)
            
            # Process reference batch
            query_ref = attn.to_q(ref_hidden)
            key_ref = attn.to_k(face_prompt)
            value_ref = attn.to_v(face_prompt)
            
            # Add PuLID ID features for reference too
            if id_embedding is not None and self.id_to_k is not None:
                key_ref = torch.cat([key_ref, id_key], dim=1)
                value_ref = torch.cat([value_ref, id_value], dim=1)
            
            query_ref = attn.head_to_batch_dim(query_ref)
            key_ref = attn.head_to_batch_dim(key_ref)
            value_ref = attn.head_to_batch_dim(value_ref)
            
            attention_probs_ref = attn.get_attention_scores(query_ref, key_ref, attention_mask)
            hidden_ref = torch.bmm(attention_probs_ref, value_ref)
            hidden_ref = attn.batch_to_head_dim(hidden_ref)
            
            # Combine
            hidden_states = torch.cat([hidden_noise, hidden_ref], dim=0)
            
        else:
            # Standard cross-attention (no branching)
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # Add PuLID ID features if available
            if id_embedding is not None and hasattr(self, 'id_to_k') and self.id_to_k is not None:
                id_key = self.id_to_k(id_embedding) * id_scale
                id_value = self.id_to_v(id_embedding) * id_scale
                key = torch.cat([key, id_key], dim=1)
                value = torch.cat([value, id_value], dim=1)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            # Ensure dtype consistency
            query = query.to(dtype=key.dtype)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            
            
        # Ensure dtype matches the projection layer
        hidden_states = hidden_states.to(dtype=attn.to_out[0].weight.dtype)
        
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )
        
        # Add residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


# Keep compatibility with original PuLID processor names
class IDAttnProcessor2_0_NS2(BranchedCrossAttnProcessor_NS2):
    """
    PuLID-specific ID attention processor with branching support.
    This maintains compatibility with PuLID's ID adapter architecture.
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
    ):
        super().__init__(hidden_size, cross_attention_dim, scale)
        
        # Initialize PuLID ID adapter layers
        self.id_to_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        
        # Initialize with small values for stability
        nn.init.zeros_(self.id_to_k.weight)
        nn.init.zeros_(self.id_to_v.weight)


class AttnProcessor2_0_NS2(nn.Module):
    """
    Standard attention processor for non-branched layers.
    Maintains compatibility with PuLID's architecture.
    """
    
    def __init__(self):
        super().__init__()
        self.has_cross_attention_kwargs = True
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
        cross_attention_kwargs=None,
    ):
        # Handle id_embedding if passed as tuple (PuLID compatibility)
        if isinstance(id_embedding, tuple):
            id_embedding = id_embedding[0] if len(id_embedding) > 0 else None
            
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Ensure dtype consistency
        query = query.to(dtype=key.dtype)
                
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


# Compatibility aliases
if HAS_TORCH2:
    AttnProcessor_NS2 = AttnProcessor2_0_NS2
    IDAttnProcessor_NS2 = IDAttnProcessor2_0_NS2
else:
    AttnProcessor_NS2 = AttnProcessor2_0_NS2
    IDAttnProcessor_NS2 = IDAttnProcessor2_0_NS2
