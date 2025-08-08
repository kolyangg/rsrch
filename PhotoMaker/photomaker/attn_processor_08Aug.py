"""
attn_processor.py - Custom attention processors for branched attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

def prepare_attention_mask(mask_tensor, threshold=0.5):
    """
    Convert mask tensor to boolean for attention masking.
    
    Args:
        mask_tensor: Float or bool tensor with mask values
        threshold: Threshold for converting float to bool
    
    Returns:
        Boolean tensor suitable for masked_fill operations
    """
    if mask_tensor.dtype == torch.bool:
        return mask_tensor
    else:
        return mask_tensor > threshold


class BranchedAttnProcessor:
    """
    Custom attention processor for self-attention layers with branching.
    
    Implements:
    - Background branch: Q=background, K/V=background+face (from noise)
    - Face branch: Q=face, K/V=face (from reference)
    - Merges results based on mask
    """
    
    def __init__(
        self,
        mask: torch.Tensor,
        mask_ref: torch.Tensor,
        reference_latents: torch.Tensor,
        step_idx: int = 0,
    ):
        """
        Args:
            mask: Face mask [B, 4, H, W] 
            reference_latents: Reference image latents [B, 4, H, W]
            step_idx: Current denoising step
        """
        self.mask = mask
        self.mask_ref = mask_ref
        self.reference_latents = reference_latents
        self.step_idx = step_idx
        
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
        Process attention with branching for face/background.
        
        Args:
            attn: Attention module
            hidden_states: Input hidden states [2B, seq_len, hidden_dim]
                         First B is noise, second B is reference
            encoder_hidden_states: Cross-attention states (None for self-attention)
            attention_mask: Attention mask
            temb: Time embeddings
            scale: Attention scale
            
        Returns:
            Processed hidden states with branching applied
        """
        # Split batch
        batch_size = hidden_states.shape[0] // 2
        noise_hidden = hidden_states[:batch_size]
        ref_hidden = hidden_states[batch_size:]
        
        # Process noise through branched attention
        noise_output = self._branched_attention_for_noise(
            attn, noise_hidden, ref_hidden, attention_mask, temb, scale
        )
        
        # Process reference through standard self-attention
        ref_output = self._standard_attention(
            attn, ref_hidden, None, temb, scale
        )
        
       # Combine and return
        return torch.cat([noise_output, ref_output], dim=0)
    
    def _branched_attention_for_noise(self, attn, noise_hidden, ref_hidden, attention_mask=None, temb=None, scale=1.0):
        """Process noise hiddens through branched attention."""
        import math
        
        residual = noise_hidden
       
        # Normalize        
        if attn.spatial_norm is not None:
            noise_hidden = attn.spatial_norm(noise_hidden)
            ref_hidden = attn.spatial_norm(ref_hidden)
            
       
        # Get dimensions
        batch_size, seq_len, hidden_dim = noise_hidden.shape
        num_heads = attn.heads
        head_dim = hidden_dim // num_heads
        
        has_ref = ref_hidden is not None and ref_hidden.shape[0] == batch_size
        # build a local residual that matches what we will return
        residual = torch.cat([noise_hidden, ref_hidden], dim=0) if has_ref else noise_hidden
       
        # === BACKGROUND BRANCH (self-attention) ===
        # Optional hard gating: keep BG queries/keys/values off the face region        
        HARD_GATE_QKV = True
        if self.mask is not None and self.mask.numel() > 0 and HARD_GATE_QKV:
            N = noise_hidden.shape[1]
            # use 1-channel mask; if 4ch passed in, take :1
            mask_img = self.mask[:, :1] if self.mask.dim() == 4 and self.mask.size(1) > 1 else self.mask
            # flatten to [B,1,HW] → interpolate in 1D to length N → [B,N,1]
            mask_1d = mask_img.flatten(2).float()                         # [B,1,HW]
            mask_1d = F.interpolate(mask_1d, size=N, mode='linear', align_corners=False)
            # mask_seq = mask_1d.transpose(1, 2)                             # [B,N,1]
            mask_seq = mask_1d.transpose(1, 2).to(dtype=noise_hidden.dtype)
            # bg_seq   = 1.0 - mask_seq
            bg_seq   = (1.0 - mask_seq).to(dtype=noise_hidden.dtype)
            noise_bg_tokens = noise_hidden * bg_seq
            ref_face_tokens = ref_hidden   # face/ref gating below for K/V
               
        else:
            noise_bg_tokens = noise_hidden
            ref_face_tokens = ref_hidden

        q_bg = attn.to_q(noise_bg_tokens)
        k_bg = attn.to_k(noise_bg_tokens)
        v_bg = attn.to_v(noise_bg_tokens)
        
        
        # Reshape for multi-head attention
        q_bg = q_bg.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k_bg = k_bg.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v_bg = v_bg.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
       
        # Compute attention
        attn_bg = (q_bg @ k_bg.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
        attn_bg = attn_bg.softmax(dim=-1)
        hidden_bg = (attn_bg @ v_bg).transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
       
        
        # === FACE BRANCH (cross-attention with reference) ===
        if self.mask is not None and self.mask.numel() > 0 and HARD_GATE_QKV:
            # mask_seq built above; if not, fall through and use full tokens
            q_face = attn.to_q(noise_hidden * mask_seq)
        
        else:
            q_face = attn.to_q(noise_hidden)
        # gate K/V to reference face if mask_ref is available
        if getattr(self, "mask_ref", None) is not None and self.mask_ref.numel() > 0 and HARD_GATE_QKV:
            N = ref_hidden.shape[1]
            mask_ref_img = self.mask_ref[:, :1] if self.mask_ref.dim() == 4 and self.mask_ref.size(1) > 1 else self.mask_ref
            mask_ref_1d = mask_ref_img.flatten(2).float()
            mask_ref_1d = F.interpolate(mask_ref_1d, size=N, mode='linear', align_corners=False)
            mask_ref_seq = mask_ref_1d.transpose(1, 2)                     # [B,N,1]
            # k_face = attn.to_k(ref_face_tokens * mask_ref_seq)
            # v_face = attn.to_v(ref_face_tokens * mask_ref_seq)
            k_face = attn.to_k(ref_face_tokens * mask_ref_seq.to(dtype=ref_face_tokens.dtype))
            v_face = attn.to_v(ref_face_tokens * mask_ref_seq.to(dtype=ref_face_tokens.dtype))
        else:
            k_face = attn.to_k(ref_face_tokens)
            v_face = attn.to_v(ref_face_tokens)
       
        # Reshape for multi-head attention
        q_face = q_face.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k_face = k_face.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v_face = v_face.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
       
        # Compute attention
        attn_face = (q_face @ k_face.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
        attn_face = attn_face.softmax(dim=-1)
        hidden_face = (attn_face @ v_face).transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
       
        # === MERGE BRANCHES ===
        if self.mask is not None and self.mask.numel() > 0:
           
           # Prepare mask
           
            N = hidden_bg.shape[1]
            mask_img = self.mask[:, :1] if self.mask.dim() == 4 and self.mask.size(1) > 1 else self.mask
            mask_1d = mask_img.flatten(2).float()
            mask_1d = F.interpolate(mask_1d, size=N, mode='linear', align_corners=False)
            mask_seq = mask_1d.transpose(1, 2)
            mask_expanded = mask_seq
        

           
            # Merge using mask (0=bg → hidden_bg, 1=face → hidden_face)
            hidden_merged = hidden_bg * (1 - mask_expanded) + hidden_face * mask_expanded
           
            if self.step_idx % 10 == 0:
                print(f"[Branched] Merging with mask (coverage: {mask_expanded.mean().item():.3f})")
        else:
           # No mask: blend 50/50
            hidden_merged = 0.5 * hidden_bg + 0.5 * hidden_face
            if self.step_idx % 10 == 0:
                print(f"[Branched] Merging 50/50 (no mask)")
       

        # Reference half: keep a standard attention path to preserve stats
        ref_output = self._standard_attention(attn, ref_hidden, attention_mask, temb, scale)

        # Output projections
        
        target_dtype = residual.dtype
        hidden_merged = hidden_merged.to(dtype=target_dtype)
        ref_output    = ref_output.to(dtype=target_dtype)

       
        
        out_noise = attn.to_out[0](hidden_merged)
        out_noise = attn.to_out[1](out_noise)
        if has_ref:
            out_ref = attn.to_out[0](ref_output)
            out_ref = attn.to_out[1](out_ref)
            
        if self.step_idx % 10 == 0 or self.step_idx < 2:
            print(f"[Branched] step={self.step_idx} | N={noise_hidden.shape[1]} | maskN={mask_seq.shape[1] if 'mask_seq' in locals() else -1} | ||bg-face||={(hidden_bg-hidden_face).abs().mean().item():.4f}")
      

        # Add residual(s) and return
        if has_ref:
            noise_out = out_noise + residual[:batch_size]
            ref_out   = out_ref   + residual[batch_size:]
            merged    = torch.cat([noise_out, ref_out], dim=0)
            if self.step_idx % 10 == 0 or self.step_idx < 2:
                print(f"[Branched] step={self.step_idx} stacked=1 | N={noise_hidden.shape[1]}")
            return merged
        else:
            noise_out = out_noise + residual
            if self.step_idx % 10 == 0 or self.step_idx < 2:
                print(f"[Branched] step={self.step_idx} stacked=0 | N={noise_hidden.shape[1]}")
            return noise_out


       
        # # TEST: Bypass branched attention entirely
        # USE_STANDARD = False  # Toggle this for testing
        # if USE_STANDARD:
        #     print(f"[TEST] Using standard attention instead of branched")
        #     return self._standard_attention(attn, hidden_states, None, temb, scale)
    
    
    def _standard_attention(self, attn, hidden_states, attention_mask=None, temb=None, scale=1.0):
        """Standard self-attention without branching."""
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        # Standard Q, K, V
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        
        # Reshape for multi-head attention
        head_dim = hidden_dim // attn.heads
        q = q.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        
        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale if scale != 1.0 else None,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(b, c, h, w)
            
        # Apply output projection
        if hasattr(attn, 'to_out'):
            if isinstance(attn.to_out, nn.ModuleList):
                for module in attn.to_out:
                    hidden_states = module(hidden_states)
            else:
                hidden_states = attn.to_out(hidden_states)
                
        return hidden_states


class BranchedCrossAttnProcessor:
    """
    Custom attention processor for cross-attention layers with branching.
    
    For cross-attention:
    - Noise branch uses generation prompt
    - Reference branch uses "face" prompt
    """
    
    def __init__(
        self,
        mask: torch.Tensor,
        mask_ref: torch.Tensor,
        face_prompt_embeds: torch.Tensor,
        step_idx: int = 0,
    ):
        """
        Args:
            mask: Face mask [B, 4, H, W]
            face_prompt_embeds: Text embeddings for "face" [B, seq_len, hidden_dim]
            step_idx: Current denoising step
        """
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask
        self.face_prompt_embeds = face_prompt_embeds
        self.step_idx = step_idx

        # Check if these are ID embeds (shorter sequence) or text embeds
        self.is_id_embeds = face_prompt_embeds.shape[1] < 77  # ID embeds are typically shorter

        # Debug: Check mask stats
        if step_idx == 0:
            face_ratio = mask.mean().item() if mask is not None else 0
            face_ratio_ref = mask_ref.mean().item() if mask_ref is not None else 0
            print(f"[Mask Debug] Noise mask face ratio: {face_ratio:.3f}")
            print(f"[Mask Debug] Ref mask face ratio: {face_ratio_ref:.3f}")

        # Check if these are ID embeds (shorter sequence) or text embeds
        self.is_id_embeds = face_prompt_embeds.shape[1] < 77  # ID embeds are typically shorter
        
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
        Process cross-attention with appropriate prompts for each branch.
        
        Args:
            attn: Attention module
            hidden_states: Input hidden states [2B, seq_len, hidden_dim]
            encoder_hidden_states: Text embeddings [2B, text_seq_len, text_dim]
                                 First B is generation prompt, second B is face prompt
            attention_mask: Attention mask
            temb: Time embeddings
            scale: Attention scale
            
        Returns:
            Processed hidden states
        """
        if encoder_hidden_states is None:
            # If no encoder states, fall back to self-attention processor
            processor = BranchedAttnProcessor(self.mask, None, self.step_idx)
            return processor(attn, hidden_states, None, attention_mask, temb, scale)
            
        residual = hidden_states
        
        # Critical debug: check if we're getting double batch
        expected_batch_size = hidden_states.shape[0] // 2
        if hidden_states.shape[0] % 2 != 0:
            print(f"[ERROR] Hidden states batch size {hidden_states.shape[0]} is not even!")
            # Fallback to standard attention
            return self._standard_attention(attn, hidden_states, None, temb, scale)
        
        
        # Split batch
        batch_size = hidden_states.shape[0] // 2
        noise_hidden = hidden_states[:batch_size]
        ref_hidden = hidden_states[batch_size:]
        
        
        # Verify the split worked correctly
        print(f"[BranchedAttn] Split check:")
        print(f"  Total batch: {hidden_states.shape[0]}")
        print(f"  Noise batch: {noise_hidden.shape[0]}, norm: {noise_hidden.std().item():.4f}")
        print(f"  Ref batch: {ref_hidden.shape[0]}, norm: {ref_hidden.std().item():.4f}")
        
        # Check if reference is actually different from noise
        diff = (noise_hidden - ref_hidden).abs().mean().item()
        print(f"  Noise-Ref difference: {diff:.6f}")
        if diff < 0.001:
            print(f"  WARNING: Noise and reference are identical!")
        
        
        # Split encoder states (prompts)
        noise_prompt = encoder_hidden_states[:batch_size]  # Generation prompt
        face_prompt = encoder_hidden_states[batch_size:]   # Face prompt

        # If face_prompt has different sequence length, adjust attention accordingly
        if self.is_id_embeds and face_prompt.shape[1] != noise_prompt.shape[1]:
           # ID embeddings have different sequence length
           # Use them as-is without trying to match dimensions
           pass  # The attention mechanism will handle different K/V sequence lengths

        # Convert masks to boolean if needed
        if hasattr(self, 'mask') and self.mask is not None:
            if self.mask.dtype != torch.bool:
                mask_bool = self.mask > 0.5
            else:
                mask_bool = self.mask
        
        # If face_prompt has different sequence length, adjust attention accordingly
        if self.is_id_embeds and face_prompt.shape[1] != noise_prompt.shape[1]:
            # ID embeddings have different sequence length
            # Use them as-is without trying to match dimensions
            pass  # The attention mechanism will handle different K/V sequence lengths
        
        # Apply spatial norm if needed
        if attn.spatial_norm is not None:
            noise_hidden = attn.spatial_norm(noise_hidden, temb)
            ref_hidden = attn.spatial_norm(ref_hidden, temb)
            
        input_ndim = noise_hidden.ndim
        
        if input_ndim == 4:
            b, c, h, w = noise_hidden.shape
            noise_hidden = noise_hidden.view(b, c, h * w).transpose(1, 2)
            ref_hidden = ref_hidden.view(b, c, h * w).transpose(1, 2)
            
        batch_size, seq_len, hidden_dim = noise_hidden.shape
        
        # Apply norm if needed
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
            
        # Noise branch: standard cross-attention with generation prompt
        q_noise = attn.to_q(noise_hidden)
        k_noise = attn.to_k(noise_prompt)
        v_noise = attn.to_v(noise_prompt)
        
        head_dim = hidden_dim // attn.heads
        q_noise = q_noise.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        k_noise = k_noise.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_noise = v_noise.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_noise = F.scaled_dot_product_attention(
            q_noise, k_noise, v_noise,
            attn_mask=attention_mask[:batch_size] if attention_mask is not None else None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale if scale != 1.0 else None,
        )
        hidden_noise = hidden_noise.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        
        # Reference branch: cross-attention with face prompt
        q_ref = attn.to_q(ref_hidden)
        k_ref = attn.to_k(face_prompt)
        v_ref = attn.to_v(face_prompt)
        
        q_ref = q_ref.view(batch_size, seq_len, attn.heads, head_dim).transpose(1, 2)
        k_ref = k_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_ref = v_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref,
            attn_mask=attention_mask[batch_size:] if attention_mask is not None else None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale if scale != 1.0 else None,
        )
        hidden_ref = hidden_ref.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        
        # Convert back to original format if needed
        if input_ndim == 4:
            hidden_noise = hidden_noise.transpose(1, 2).view(b, c, h, w)
            hidden_ref = hidden_ref.transpose(1, 2).view(b, c, h, w)
            
        # Apply output projection
        if hasattr(attn, 'to_out'):
            if isinstance(attn.to_out, nn.ModuleList):
                for module in attn.to_out:
                    hidden_noise = module(hidden_noise)
                    hidden_ref = module(hidden_ref)
            else:
                hidden_noise = attn.to_out(hidden_noise)
                hidden_ref = attn.to_out(hidden_ref)
                
        # Add residuals
        noise_output = hidden_noise + residual[:batch_size]
        ref_output = hidden_ref + residual[batch_size:]
        
        # Stack outputs
        output = torch.cat([noise_output, ref_output], dim=0)
        
        return output


class StandardAttnProcessor:
    """
    Standard attention processor without any modifications.
    Used as a fallback when branching is not needed.
    """
    
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
        Standard attention processing.
        """
            
        
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
            
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        batch_size, seq_len, _ = hidden_states.shape
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        if hasattr(attn, 'norm_cross') and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale if scale != 1.0 else None,
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(batch_size, channel, height, width)
            
        # Linear projection
        if hasattr(attn, 'to_out'):
            if isinstance(attn.to_out, nn.ModuleList):
                for module in attn.to_out:
                    hidden_states = module(hidden_states)
            else:
                hidden_states = attn.to_out(hidden_states)
                
        # Add residual
        hidden_states = hidden_states + residual
        
        return hidden_states