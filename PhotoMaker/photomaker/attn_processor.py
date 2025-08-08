"""
attn_processor.py - Optimized branched attention processors following IP-Adapter pattern
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class BranchedAttnProcessor(nn.Module):
    """
    Self-attention processor with face/background branching.
    Follows the IP-Adapter pattern with learnable projections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale
        
        # Learnable projections for face branch (uses reference)
        self.to_k_face = nn.Linear(self.hidden_size, hidden_size, bias=False)
        self.to_v_face = nn.Linear(self.hidden_size, hidden_size, bias=False)
        
        
    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        """Set masks for current denoising step"""
        self.mask = mask
        self.mask_ref = mask_ref
        
    def set_reference(self, reference_latents: Optional[torch.Tensor]):
        """Set reference latents for face branch"""
        self.reference_latents = reference_latents
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Process self-attention with face/background branching.
        
        For self-attention (encoder_hidden_states=None):
        - Background branch: Standard self-attention
        - Face branch: Cross-attention with reference
        - Merge based on mask
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
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Handle group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # For cross-attention, use standard processor
        if encoder_hidden_states is not None:
            return self._standard_cross_attention(
                attn, hidden_states, encoder_hidden_states, 
                attention_mask, residual, input_ndim
            )
        
        # Self-attention with branching
        query = attn.to_q(hidden_states)
        
        # Get attention parameters
        head_dim = attn.heads
        dim_per_head = hidden_states.shape[-1] // head_dim
        
        # === BACKGROUND BRANCH (standard self-attention) ===
        key_bg = attn.to_k(hidden_states)
        value_bg = attn.to_v(hidden_states)
        
        query = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Compute background attention
        hidden_bg = F.scaled_dot_product_attention(
            query, key_bg, value_bg, dropout_p=0.0, is_causal=False
        )
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, hidden_states.shape[-1])
        
        # === FACE BRANCH (cross-attention with reference) ===
        if hasattr(self, 'reference_latents') and self.reference_latents is not None:
            # Prepare reference for attention
            ref_latents = self.reference_latents
            if ref_latents.ndim == 4:
                # ref_latents = ref_latents.view(batch_size, -1, seq_len).transpose(1, 2)
                # # Flatten: [B, C, H, W] -> [B, H*W, C]
                # ref_latents = ref_latents.flatten(2).transpose(1, 2)
                
                # Reference latents are VAE encoded [B, 4, H, W]
                # Need to project to hidden states space
                # Use a simple repeat to match hidden dimension
                b, c, h, w = ref_latents.shape
                # Flatten and repeat channels to match hidden_dim
                ref_latents = ref_latents.flatten(2).transpose(1, 2)  # [B, H*W, 4]
                # Expand to hidden dimension by repeating and projecting
                ref_latents = ref_latents.repeat(1, 1, hidden_states.shape[-1] // c)[:, :, :hidden_states.shape[-1]]
                
                # Ensure dtype matches
                ref_latents = ref_latents.to(dtype=hidden_states.dtype)
   
   
            
            # Use learnable projections for reference
            key_face = self.to_k_face(ref_latents)
            value_face = self.to_v_face(ref_latents)
            
            # # Use standard projections (they handle dimension conversion)
            # key_face = attn.to_k(ref_latents)
            # value_face = attn.to_v(ref_latents)
                        
            key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            
            # Compute face attention
            hidden_face = F.scaled_dot_product_attention(
                query, key_face, value_face, dropout_p=0.0, is_causal=False
            )
            hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, hidden_states.shape[-1])
            
            # Merge branches based on mask
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self._prepare_mask(self.mask, seq_len)
                mask = mask.to(dtype=hidden_states.dtype)
                hidden_states = hidden_bg * (1 - mask) + hidden_face * mask * self.scale
            else:
                # No mask: weighted blend
                hidden_states = hidden_bg + hidden_face * self.scale
        else:
            # No reference: use background only
            hidden_states = hidden_bg
        
        # Ensure dtype consistency before output projection
        hidden_states = hidden_states.to(dtype=query.dtype)
        
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Add residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    def _standard_cross_attention(
        self, attn, hidden_states, encoder_hidden_states, 
        attention_mask, residual, input_ndim
    ):
        """Standard cross-attention implementation"""
        batch_size = hidden_states.shape[0]
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        head_dim = attn.heads
        dim_per_head = hidden_states.shape[-1] // head_dim
        
        query = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, hidden_states.shape[-1] * head_dim
        )
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # Reshape if needed
        if input_ndim == 4:
            channel = residual.shape[1]
            height = width = int(math.sqrt(hidden_states.shape[1]))
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Add residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Prepare mask for attention operations"""
        if mask.ndim == 4:
            # Spatial mask [B, C, H, W] -> [B, target_len, 1]
            batch_size = mask.shape[0]
            mask = mask[:, :1]  # Use first channel
            mask = mask.flatten(2).float()
            mask = F.interpolate(mask, size=target_len, mode='linear', align_corners=False)
            mask = mask.transpose(1, 2)
        elif mask.ndim == 3:
            # Already [B, target_len, 1]
            pass
        else:
            # Ensure proper shape
            mask = mask.view(mask.shape[0], -1, 1)
            if mask.shape[1] != target_len:
                mask = F.interpolate(
                    mask.transpose(1, 2), 
                    size=target_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
        
        return mask


class BranchedCrossAttnProcessor(nn.Module):
    """
    Cross-attention processor with ID embeddings support.
    Handles both text and face prompts following IP-Adapter pattern.
    """
    
    def __init__(
        self, 
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
        num_tokens: int = 77,
    ):
        super().__init__()
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        # Projections for ID embeddings (face prompts)
        self.to_k_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Process cross-attention with text and ID embeddings.
        
        Expected encoder_hidden_states format:
        - First num_tokens: text embeddings
        - Remaining: ID embeddings (if present)
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
        
        batch_size = hidden_states.shape[0]
        
        # Handle group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Query from hidden states
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Split encoder states into text and ID embeddings
        text_embeds = encoder_hidden_states[:, :self.num_tokens, :]
        id_embeds = None
        if encoder_hidden_states.shape[1] > self.num_tokens:
            id_embeds = encoder_hidden_states[:, self.num_tokens:, :]
        
        # Process text embeddings
        key_text = attn.to_k(text_embeds)
        value_text = attn.to_v(text_embeds)
        
        # Reshape for attention
        head_dim = attn.heads
        dim_per_head = hidden_states.shape[-1] // head_dim
        
        query = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        key_text = key_text.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_text = value_text.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Text attention
        hidden_states = F.scaled_dot_product_attention(
            query, key_text, value_text, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, hidden_states.shape[-1] * head_dim
        )
        
        # Process ID embeddings if present
        if id_embeds is not None:
            key_id = self.to_k_id(id_embeds)
            value_id = self.to_v_id(id_embeds)
            
            key_id = key_id.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_id = value_id.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            
            # ID attention
            hidden_id = F.scaled_dot_product_attention(
                query, key_id, value_id, dropout_p=0.0, is_causal=False
            )
            hidden_id = hidden_id.transpose(1, 2).reshape(
                batch_size, -1, hidden_id.shape[-1] * head_dim
            )
            
            # Combine text and ID branches
            hidden_states = hidden_states + hidden_id * self.scale
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape if needed
        if input_ndim == 4:
            channel = residual.shape[1]
            height = width = int(math.sqrt(hidden_states.shape[1]))
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        # Add residual
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states