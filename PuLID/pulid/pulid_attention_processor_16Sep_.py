"""
pulid_attention_processor_16Sep.py - Branched attention processors for PuLID
Adapted from PhotoMaker's branched attention implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from contextlib import nullcontext
import math

# Check for PyTorch 2.0+ for scaled_dot_product_attention
if hasattr(F, "scaled_dot_product_attention"):
    HAS_TORCH2 = True
else:
    HAS_TORCH2 = False
    import warnings
    warnings.warn("PyTorch 2.0+ recommended for optimal performance")


def _sdp_context():
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    return nullcontext()


class BranchedAttnProcessorPuLID(nn.Module):
    """
    Self-attention processor with face/background branching for PuLID.
    Processes doubled batch: [noise_batch, reference_batch]
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
        
        # Runtime-tunable parameters
        self.pose_adapt_ratio: float = 0.25
        self.ca_mixing_for_face: bool = True
        self.use_id_embeds: bool = True

        # Optional: ID feature cache for PuLID
        self.id_embeds = None
        self.id_to_hidden = None

        # Optional compute dtype override for memory tuning
        self.attention_dtype: Optional[torch.dtype] = None

        # Sequential mode (VRAM-saving): 'record_ref' → cache ref hidden; 'use_cached' → use it
        self.seq_mode: Optional[str] = None
        self._cached_ref_hidden: Optional[torch.Tensor] = None

    @staticmethod
    def _call_linear(layer, x, scale: float):
        try:
            return layer(x, scale=scale)
        except TypeError:
            return layer(x)

    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        """Set masks for current denoising step"""
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def set_attention_dtype(self, dtype: Optional[torch.dtype]):
        """Configure compute dtype for branched attention (e.g., torch.float16)."""
        self.attention_dtype = dtype
        

    def set_seq_mode(self, mode: Optional[str]):
        # mode in {None, 'record_ref', 'use_cached'}
        self.seq_mode = mode
        if mode is None:
            self._cached_ref_hidden = None
    
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
        uncond_id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Process cross-attention with branching.
        For PuLID, this handles text conditioning with optional face-specific prompts.
        """
        if encoder_hidden_states is None:
            raise ValueError("Cross-attention requires encoder_hidden_states")
        
        # Handle doubled batch
        full_batch = hidden_states.shape[0]
        half_batch = full_batch // 2
        output_dtype = hidden_states.dtype
        device = hidden_states.device
        
        # If not doubled or no masks AND not in sequential mode, use standard path
        if (self.mask is None or full_batch % 2 != 0) and self.seq_mode is None:
            return self._standard_cross_attention(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, scale
            )
        
        # Split hidden states
        noise_hidden = hidden_states[:half_batch]
        ref_hidden = hidden_states[half_batch:]
        
        # Compute dtype override if requested
        target_dtype = self.attention_dtype or noise_hidden.dtype

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(device=device, dtype=target_dtype)

        if noise_hidden.dtype != target_dtype:
            noise_hidden = noise_hidden.to(device=device, dtype=target_dtype)
            ref_hidden = ref_hidden.to(device=device, dtype=target_dtype)

        # Split encoder hidden states
        gen_prompt = encoder_hidden_states[:half_batch]
        face_prompt = encoder_hidden_states[half_batch:] if self.face_prompt_embeds is None else self.face_prompt_embeds

        gen_prompt = gen_prompt.to(device=device, dtype=target_dtype)
        face_prompt = face_prompt.to(device=device, dtype=target_dtype)

        residual = noise_hidden
        input_ndim = noise_hidden.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = noise_hidden.shape
            noise_hidden = noise_hidden.view(batch_size, channel, height * width).transpose(1, 2)
            ref_hidden = ref_hidden.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size = noise_hidden.shape[0]
        seq_len = noise_hidden.shape[1]
        
        # Handle group norm
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
        
        # Get masks for this resolution
        mask = self._prepare_mask(self.mask, seq_len, batch_size)
        mask = mask.to(device=noise_hidden.device, dtype=target_dtype)

        if attention_mask is not None and attention_mask.dtype != target_dtype:
            attention_mask = attention_mask.to(dtype=target_dtype)
        
        # === BACKGROUND BRANCH ===
        # Q: from noise, K/V: from generation prompt
        query_bg = attn.to_q(noise_hidden)
        key_bg = attn.to_k(gen_prompt)
        value_bg = attn.to_v(gen_prompt)
        
        # === FACE BRANCH ===
        # Q: from noise, K/V: from face prompt
        query_face = attn.to_q(noise_hidden)
        key_face = attn.to_k(face_prompt) if face_prompt.shape[0] == batch_size else attn.to_k(face_prompt[:batch_size])
        value_face = attn.to_v(face_prompt) if face_prompt.shape[0] == batch_size else attn.to_v(face_prompt[:batch_size])
        
        # Reshape for attention
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim
        
        def reshape_for_attention(tensor):
            return tensor.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        query_bg = reshape_for_attention(query_bg)
        key_bg = reshape_for_attention(key_bg)
        value_bg = reshape_for_attention(value_bg)
        
        query_face = reshape_for_attention(query_face)
        key_face = reshape_for_attention(key_face)
        value_face = reshape_for_attention(value_face)
        
        # Compute attention
        hidden_bg = F.scaled_dot_product_attention(
            query_bg, key_bg, value_bg,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        hidden_face = F.scaled_dot_product_attention(
            query_face, key_face, value_face,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # Combine branches using mask
        hidden_states = (1 - mask) * hidden_bg + mask * hidden_face
        
        # Ensure correct dtype before linear projection
        hidden_states = hidden_states.to(dtype=noise_hidden.dtype)
        
        # Linear output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Add residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor

        # Duplicate for doubled batch
        hidden_states = torch.cat([hidden_states, hidden_states], dim=0)

        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)

        return hidden_states
    
    def _standard_cross_attention(self, attn, hidden_states, encoder_hidden_states,
                                 attention_mask, temb, scale):
        """Standard cross-attention fallback"""
        output_dtype = hidden_states.dtype
        target_dtype = self.attention_dtype or hidden_states.dtype

        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(dtype=target_dtype)

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(dtype=target_dtype)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        if attention_mask is not None and attention_mask.dtype != target_dtype:
            attention_mask = attention_mask.to(dtype=target_dtype)

        query = attn.to_q(hidden_states, scale=scale)
        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor

        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)

        return hidden_states
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention computation"""
        if mask is None:
            return torch.zeros(batch_size, target_len, 1, device='cuda', dtype=torch.float32)
        
        # Handle different mask formats
        H = W = int(math.sqrt(target_len))
        
        if mask.ndim == 4:  # [B, C, H, W]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode='bilinear', align_corners=False)
        else:  # [B, L, 1] or [B, 1, L]
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = w0 = int(math.sqrt(L0))
            m2d = mask.view(mask.shape[0], -1)[:, :L0].float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode='bilinear', align_corners=False)
        
        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        
        # Expand for batch if needed
        if m.shape[0] != batch_size:
            if m.shape[0] > batch_size:
                m = m[:batch_size]
            else:
                m = m.expand(batch_size, -1, -1)
        
        return m.view(batch_size, target_len, 1).to(dtype=torch.float32)

        
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
        uncond_id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Process self-attention with face/background branching.
        
        For doubled batch: first half is noise, second half is reference.
        """
        # Handle doubled batch
        full_batch = hidden_states.shape[0]
        half_batch = full_batch // 2
        output_dtype = hidden_states.dtype
        target_dtype = self.attention_dtype or hidden_states.dtype

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(dtype=target_dtype)

        # If not doubled (fallback), use standard attention
        if self.mask is None or self.mask_ref is None or full_batch % 2 != 0:
            return self._standard_attention(
                attn, hidden_states, encoder_hidden_states, 
                attention_mask, temb, scale
            )

        # Split into noise and reference

        # === Parallel (doubled-batch) branch ===
        # (kept as before when not using sequential mode)
        if self.seq_mode is None:
            noise_hidden = hidden_states[:half_batch]
            ref_hidden = hidden_states[half_batch:]
        else:
            # === Sequential mode ===
            # In 'record_ref' pass, inputs are *reference* latents only (B)
            # In 'use_cached' pass, inputs are *noise* latents only (B)
            noise_hidden = hidden_states
            ref_hidden = None  # filled from cache when use_cached


        if noise_hidden.dtype != target_dtype:
            noise_hidden = noise_hidden.to(dtype=target_dtype)
            ref_hidden = ref_hidden.to(dtype=target_dtype)

        residual = noise_hidden
        input_ndim = noise_hidden.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = noise_hidden.shape
            noise_hidden = noise_hidden.view(batch_size, channel, height * width).transpose(1, 2)

            if ref_hidden is not None:
                ref_hidden = ref_hidden.view(batch_size, channel, height * width).transpose(1, 2)

        
        batch_size = noise_hidden.shape[0]
        
        # Handle group norm
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)

        # Get masks for this resolution
        mask = self._prepare_mask(self.mask, noise_hidden.shape[1], batch_size)
        # mask_ref = self._prepare_mask(self.mask_ref, ref_hidden.shape[1], batch_size)
        mask = mask.to(device=noise_hidden.device, dtype=target_dtype)
        # mask_ref = mask_ref.to(device=ref_hidden.device, dtype=target_dtype)
        
        # ======= Sequential short-circuits =======
        if self.seq_mode == 'record_ref':
            # # Cache ref hidden (pre-attention, normalized & reshaped), return passthrough
            # self._cached_ref_hidden = noise_hidden.detach()
            
            # Cache ref hidden on **CPU fp16** to avoid holding VRAM between passes
            self._cached_ref_hidden = noise_hidden.detach().to('cpu', dtype=torch.float16)

            
            # Return residual passthrough to keep UNet shape flow (no heavy compute)
            if input_ndim == 4:
                out = noise_hidden.transpose(-1, -2).reshape(batch_size, channel, height, width)
            else:
                out = noise_hidden
            if attn.residual_connection:
                out = out + (hidden_states if input_ndim != 4 else hidden_states)
            return out / attn.rescale_output_factor

        if self.seq_mode == 'use_cached':
            if self._cached_ref_hidden is None:
                # Fallback if someone forgot the record stage
                return self._standard_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
            # ref_hidden = self._cached_ref_hidden        

            # Move cached ref back to device with target dtype
            ref_hidden = self._cached_ref_hidden.to(device=noise_hidden.device, dtype=target_dtype, non_blocking=True)



        # === BRANCH 1: BACKGROUND ===
        # Q: background from noise, K/V: all from noise
        # query_bg = attn.to_q(noise_hidden)
        # key_bg = attn.to_k(noise_hidden)
        # value_bg = attn.to_v(noise_hidden)        

        query_bg = self._call_linear(attn.to_q, noise_hidden, scale)
        key_bg   = self._call_linear(attn.to_k, noise_hidden, scale)
        value_bg = self._call_linear(attn.to_v, noise_hidden, scale)
        
        # PuLID ID adapter integration
        if id_embedding is not None and hasattr(attn, 'to_k_pulid'):
            if id_embedding.dtype != target_dtype:
                id_embedding = id_embedding.to(dtype=target_dtype)
            # Add PuLID ID features to keys/values
            key_id = attn.to_k_pulid(id_embedding)
            value_id = attn.to_v_pulid(id_embedding)
            
            # Combine with original K/V
            key_bg = key_bg + id_scale * key_id
            value_bg = value_bg + id_scale * value_id
        
        # === BRANCH 2: FACE ===
        # Q: face from noise, K/V: face from reference
        # query_face = attn.to_q(noise_hidden)
        # key_face = attn.to_k(ref_hidden)
        # value_face = attn.to_v(ref_hidden)
        
        query_face = self._call_linear(attn.to_q, noise_hidden, scale)
        key_face   = self._call_linear(attn.to_k, ref_hidden, scale)
        value_face = self._call_linear(attn.to_v, ref_hidden, scale)

        
        # Apply equalization if enabled
        if self.equalize_face_kv:
            key_face = self._equalize_tensor(key_face)
            value_face = self._equalize_tensor(value_face)
        
        # Reshape for attention
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim
        
        def reshape_for_attention(tensor):
            return tensor.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        query_bg = reshape_for_attention(query_bg)
        key_bg = reshape_for_attention(key_bg)
        value_bg = reshape_for_attention(value_bg)
        
        query_face = reshape_for_attention(query_face)
        key_face = reshape_for_attention(key_face)
        value_face = reshape_for_attention(value_face)
        
        # Compute attention for both branches
        if attention_mask is not None and attention_mask.dtype != target_dtype:
            attention_mask = attention_mask.to(dtype=target_dtype)

        hidden_bg = F.scaled_dot_product_attention(
            query_bg, key_bg, value_bg, 
            attn_mask=attention_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        hidden_face = F.scaled_dot_product_attention(
            query_face, key_face, value_face,
            attn_mask=None,  # No mask for face branch
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # Apply pose adaptation
        if self.pose_adapt_ratio < 1.0:
            # Mix in some background features for pose
            hidden_face = (1 - self.pose_adapt_ratio) * hidden_face + self.pose_adapt_ratio * hidden_bg
        
        # Combine branches using mask
        hidden_states = (1 - mask) * hidden_bg + mask * hidden_face
        
        # Ensure correct dtype before linear projection
        hidden_states = hidden_states.to(dtype=attn.to_out[0].weight.dtype)
        
        # Linear output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Add residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor

        # For doubled batch, duplicate the result
        # hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
        
        # For sequential mode we DO NOT duplicate; for parallel we do
        if self.seq_mode is None:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)    


        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)

        return hidden_states
    
    def _standard_attention(self, attn, hidden_states, encoder_hidden_states, 
                          attention_mask, temb, scale):
        """Fallback to standard attention when not using branching"""
        output_dtype = hidden_states.dtype
        target_dtype = self.attention_dtype or hidden_states.dtype

        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(dtype=target_dtype)

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(dtype=target_dtype)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        if attention_mask is not None and attention_mask.dtype != target_dtype:
            attention_mask = attention_mask.to(dtype=target_dtype)

        query = self._call_linear(attn.to_q, hidden_states, scale)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self._call_linear(attn.to_k, encoder_hidden_states, scale)
        value = self._call_linear(attn.to_v, encoder_hidden_states, scale)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = self._call_linear(attn.to_out[0], hidden_states, scale)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor

        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)
        
        return hidden_states
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention computation"""
        if mask is None:
            return torch.zeros(batch_size, target_len, 1, device='cuda')
        
        # Handle different mask formats
        H = W = int(math.sqrt(target_len))
        
        if mask.ndim == 4:  # [B, C, H, W]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode='bilinear', align_corners=False)
        else:  # [B, L, 1] or [B, 1, L]
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = w0 = int(math.sqrt(L0))
            m2d = mask.view(mask.shape[0], -1)[:, :L0].float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode='bilinear', align_corners=False)
        
        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        
        # Expand for batch if needed
        if m.shape[0] != batch_size:
            if m.shape[0] > batch_size:
                m = m[:batch_size]
            else:
                m = m.expand(batch_size, -1, -1)
        
        return m.view(batch_size, target_len, 1)
    
    def _equalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply equalization to tensor values"""
        if not self.equalize_face_kv:
            return tensor
        
        min_clip, max_clip = self.equalize_clip
        std = tensor.std()
        mean = tensor.mean()
        
        # Clip to reasonable range
        tensor = torch.clamp(tensor, mean - max_clip * std, mean + max_clip * std)
        
        # Normalize
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        
        return tensor


# class BranchedCrossAttnProcessorPuLID(nn.Module):
#     """
#     Cross-attention processor with branching for PuLID.
#     Handles text conditioning with optional face-specific prompts.
#     """
    
#     def __init__(
#         self,
#         hidden_size: int,
#         cross_attention_dim: Optional[int] = None,
#         scale: float = 1.0,
#         num_tokens: int = 77,
#     ):
#         super().__init__()
        
#         if not HAS_TORCH2:
#             raise ImportError("Requires PyTorch 2.0+ for scaled_dot_product_attention")
        
#         self.hidden_size = hidden_size
#         self.cross_attention_dim = cross_attention_dim
#         self.scale = scale
#         self.num_tokens = num_tokens
        
#         # Mask storage
#         self.mask = None
#         self.mask_ref = None
        
#         # Face prompt embeddings
#         self.face_prompt_embeds = None
        
#         # PuLID ID features
#         self.id_embeds = None



class BranchedCrossAttnProcessorPuLID(nn.Module):
    """
    Cross-attention processor with face/background branching for PuLID.
    Q comes from the (noise) hidden_states; background K/V come from the
    normal text encoder prompts; face K/V come from `face_prompt_embeds`.
    During branched runs we pass a *doubled* batch to the UNet:
      - without CFG: [noise, reference]
      - with CFG:    [uncond_noise, cond_noise, reference, reference]
    We compute cross-attention only for the "noise" half and then
    duplicate the result so shapes match the doubled batch.
    """
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 77,
    ):
        super().__init__()
        if not HAS_TORCH2:
            raise ImportError("Requires PyTorch 2.0+ for scaled_dot_product_attention")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        # Masks (latent-space resolution)
        self.mask: Optional[torch.Tensor] = None
        self.mask_ref: Optional[torch.Tensor] = None

        # Optional face prompt embeddings (shape usually [2*B, T, C] under CFG)
        self.face_prompt_embeds: Optional[torch.Tensor] = None

        # Runtime knobs (parity with PhotoMaker impl.)
        self.pose_adapt_ratio: float = 1.0         # not used directly in CA
        self.ca_mixing_for_face: bool = True       # allow light mixing of bg tokens into face K/V
        self.use_id_embeds: bool = True            # reserved for future CA integration

        # Optional compute dtype override
        self.attention_dtype: Optional[torch.dtype] = None

    # ---- utilities ----
    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask

    def set_attention_dtype(self, dtype: Optional[torch.dtype]):
        self.attention_dtype = dtype

    @staticmethod
    def _call_linear(layer, x, scale: float):
        # Diffusers layers sometimes accept `scale` and sometimes not
        try:
            return layer(x, scale=scale)
        except TypeError:
            return layer(x)

    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        if mask is None:
            device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            return torch.zeros(batch_size, target_len, 1, device=device, dtype=torch.float32)

        H = W = int(math.sqrt(target_len))
        if mask.ndim == 4:  # [B,1,H,W] or [B,C,H,W]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode='bilinear', align_corners=False)
        else:
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = w0 = int(math.sqrt(L0))
            m2d = mask.view(mask.shape[0], -1)[:, :L0].float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode='bilinear', align_corners=False)

        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        if m.shape[0] != batch_size:
            if m.shape[0] > batch_size:
                m = m[:batch_size]
            else:
                m = m.expand(batch_size, -1, -1)
        return m.to(dtype=torch.float32)

    def _standard_cross_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale):
        """Fallback: vanilla cross-attention (no branching)."""
        output_dtype = hidden_states.dtype
        target_dtype = self.attention_dtype or hidden_states.dtype

        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(dtype=target_dtype)

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(dtype=target_dtype)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        b, _, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        query = self._call_linear(attn.to_q, hidden_states, scale)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self._call_linear(attn.to_k, encoder_hidden_states, scale)
        value = self._call_linear(attn.to_v, encoder_hidden_states, scale)

        if HAS_TORCH2:
            head_dim = attn.heads
            dim_per_head = query.shape[-1] // head_dim

            def reshape_for_attention(tensor):
                return tensor.reshape(b, -1, head_dim, dim_per_head).transpose(1, 2)

            q = reshape_for_attention(query)
            k = reshape_for_attention(key)
            v = reshape_for_attention(value)

            attn_mask_sdp = None
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=target_dtype)
                attn_mask_sdp = attn.prepare_attention_mask(attention_mask, k.shape[-2], b)
                attn_mask_sdp = attn_mask_sdp.view(b, attn.heads, -1, attn_mask_sdp.shape[-1])

            with _sdp_context():
                hidden_states = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask_sdp, dropout_p=0.0, is_causal=False
                )

            hidden_states = hidden_states.to(q.dtype)
            hidden_states = hidden_states.transpose(1, 2).reshape(b, -1, attn.heads * dim_per_head)
        else:
            if attention_mask is not None and attention_mask.dtype != target_dtype:
                attention_mask = attention_mask.to(dtype=target_dtype)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = self._call_linear(attn.to_out[0], hidden_states, scale)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)

        return hidden_states

    # ---- main ----
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        # passthrough from UNet (ignored here but kept for signature-compat)
        id_embedding: Optional[torch.Tensor] = None,
        uncond_id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        # If not in branched mode (no masks or batch not doubled), just do vanilla CA.
        full_batch = hidden_states.shape[0]
        if self.mask is None or full_batch % 2 != 0:
            return self._standard_cross_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        # Only operate on the "noise" half; later we'll duplicate to match input batch.
        half_batch = full_batch // 2
        output_dtype = hidden_states.dtype
        target_dtype = self.attention_dtype or hidden_states.dtype

        if temb is not None and temb.dtype != target_dtype:
            temb = temb.to(dtype=target_dtype)

        noise_hidden = hidden_states[:half_batch]
        if noise_hidden.dtype != target_dtype:
            noise_hidden = noise_hidden.to(dtype=target_dtype)

        residual = noise_hidden
        input_ndim = noise_hidden.ndim

        if attn.spatial_norm is not None:
            noise_hidden = attn.spatial_norm(noise_hidden, temb)

        if input_ndim == 4:
            b, c, h, w = noise_hidden.shape
            noise_hidden = noise_hidden.view(b, c, h * w).transpose(1, 2)
        else:
            b = noise_hidden.shape[0]

        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)

        # Slice matching text prompts for the noise half
        if encoder_hidden_states is None:
            raise ValueError("BranchedCrossAttnProcessorPuLID requires encoder_hidden_states for cross-attention.")
        gen_prompt = encoder_hidden_states[:half_batch].to(device=noise_hidden.device, dtype=target_dtype)

        # Prepare face prompts
        if self.face_prompt_embeds is not None:
            face_prompt = self.face_prompt_embeds.to(device=gen_prompt.device, dtype=target_dtype)
            if face_prompt.shape[0] != half_batch:
                if face_prompt.shape[0] == 1:
                    face_prompt = face_prompt.expand(half_batch, -1, -1)
                else:
                    face_prompt = face_prompt[:half_batch]
        else:
            face_prompt = gen_prompt  # fallback

        # Optional light mixing for stability
        if self.ca_mixing_for_face and face_prompt is not gen_prompt:
            mix_alpha = 0.1
            face_prompt = (1.0 - mix_alpha) * face_prompt + mix_alpha * gen_prompt

        # Projections
        query = self._call_linear(attn.to_q, noise_hidden, scale)
        key_bg = self._call_linear(attn.to_k, gen_prompt, scale)
        value_bg = self._call_linear(attn.to_v, gen_prompt, scale)

        key_face = self._call_linear(attn.to_k, face_prompt, scale)
        value_face = self._call_linear(attn.to_v, face_prompt, scale)

        head_dim = attn.heads
        dim_per_head = query.shape[-1] // head_dim

        def reshape_for_attention(tensor):
            return tensor.reshape(b, -1, head_dim, dim_per_head).transpose(1, 2)

        q = reshape_for_attention(query)
        k_bg = reshape_for_attention(key_bg)
        v_bg = reshape_for_attention(value_bg)

        k_face = reshape_for_attention(key_face)
        v_face = reshape_for_attention(value_face)

        if HAS_TORCH2:
            attn_mask_sdp = None
            if attention_mask is not None:
                attn_mask_sdp = attn.prepare_attention_mask(attention_mask, k_bg.shape[-2], b)
                attn_mask_sdp = attn_mask_sdp.view(b, attn.heads, -1, attn_mask_sdp.shape[-1])

            with _sdp_context():
                out_bg = F.scaled_dot_product_attention(
                    q, k_bg, v_bg, attn_mask=attn_mask_sdp, dropout_p=0.0, is_causal=False
                )

            with _sdp_context():
                out_face = F.scaled_dot_product_attention(
                    q, k_face, v_face, attn_mask=None, dropout_p=0.0, is_causal=False
                )

            out_bg = out_bg.to(q.dtype)
            out_face = out_face.to(q.dtype)
            out_bg = out_bg.transpose(1, 2).reshape(b, -1, noise_hidden.shape[-1])
            out_face = out_face.transpose(1, 2).reshape(b, -1, noise_hidden.shape[-1])
        else:
            if attention_mask is not None and attention_mask.dtype != target_dtype:
                attention_mask = attention_mask.to(dtype=target_dtype)

            q_flat = attn.head_to_batch_dim(query)
            k_bg_flat = attn.head_to_batch_dim(key_bg)
            v_bg_flat = attn.head_to_batch_dim(value_bg)
            attn_bg = attn.get_attention_scores(q_flat, k_bg_flat, attention_mask)
            out_bg = torch.bmm(attn_bg, v_bg_flat)
            out_bg = attn.batch_to_head_dim(out_bg)

            k_face_flat = attn.head_to_batch_dim(key_face)
            v_face_flat = attn.head_to_batch_dim(value_face)
            attn_face = attn.get_attention_scores(q_flat, k_face_flat, None)
            out_face = torch.bmm(attn_face, v_face_flat)
            out_face = attn.batch_to_head_dim(out_face)

        # Blend by spatial mask
        seq_len = noise_hidden.shape[1]
        mask = self._prepare_mask(self.mask, seq_len, b)  # [b, L, 1]
        mask = mask.to(device=out_bg.device, dtype=out_bg.dtype)
        hidden = (1.0 - mask) * out_bg + mask * out_face

        # Output projection
        hidden = self._call_linear(attn.to_out[0], hidden, scale)
        hidden = attn.to_out[1](hidden)

        if input_ndim == 4:
            hidden = hidden.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden = hidden + residual
        hidden = hidden / attn.rescale_output_factor

        # Duplicate to match the doubled input batch (noise + reference)
        hidden = torch.cat([hidden, hidden], dim=0)

        if hidden.dtype != output_dtype:
            hidden = hidden.to(dtype=output_dtype)

        return hidden

        
