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
        
        # # Learnable projections for face branch (uses reference)
        # self.to_k_face = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)
        # self.to_v_face = nn.Linear(self.cross_attention_dim, hidden_size, bias=False)

        # # Input dim matches text embedding size (4096 for SDXL) or hidden_size
        # input_dim = 4096 if cross_attention_dim and cross_attention_dim > 2048 else hidden_size
        # self.to_k_face = nn.Linear(input_dim, hidden_size, bias=False)
        # self.to_v_face = nn.Linear(input_dim, hidden_size, bias=False)

        # Don't initialize here - will create dynamically based on actual input size
        self.to_k_face = None
        self.to_v_face = None
        self.hidden_size = hidden_size


    def set_face_embeds(self, face_embeds: Optional[torch.Tensor]):
        """Set face embeddings for cross-attention"""
        self.face_embeds = face_embeds    
    
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
        
        # Split batched inputs if we have both noise and reference
        # Expect [noise_batch, reference_batch]
        if batch_size > 1 and hasattr(self, "reference_latents"):
            # First half is noise, second half is reference
            half_batch = batch_size // 2
            noise_hidden = hidden_states[:half_batch]
            ref_hidden = hidden_states[half_batch:]
            # Process with split
            batch_size = half_batch
        else:
            noise_hidden = hidden_states
            ref_hidden = None
        
        # Handle group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # For cross-attention, use standard processor
        if encoder_hidden_states is not None:
            return self._standard_cross_attention(
                attn, hidden_states, encoder_hidden_states, 
                attention_mask, residual, input_ndim
            )
        
        # # Self-attention with branching
        # query = attn.to_q(hidden_states)
        
        # Self-attention with branching - use noise latents for Q
        query = attn.to_q(noise_hidden if ref_hidden is not None else hidden_states)
        
        # # Get attention parameters
        # head_dim = attn.heads
        # dim_per_head = hidden_states.shape[-1] // head_dim
        
        
        # if hasattr(self, 'mask') and self.mask is not None:
        #         # Ensure shape (B, L[,1]) → (B,1,L,1), safely expanding for CFG batches
        #         # mask_gate = self._prepare_mask(self.mask, seq_len)
        #         mask_gate = self._prepare_mask(self.mask, seq_len)
        #         # if step := getattr(self, "_dbg_step", None) in (0, 1, 2):
        #         #     # quick sanity: check grid & means
        #         #     _h = int(math.isqrt(seq_len)); _w = _h
        #         #     print(f"[MaskDBG] L={seq_len} grid={_h}x{_w}  mean(mask)={mask_gate.mean().item():.3f}")
        #         if mask_gate.dim() == 3 and mask_gate.shape[-1] == 1:
        #             mask_gate = mask_gate.squeeze(-1)          # (B_or_1, L)
        #         if mask_gate.dim() == 1:
        #             mask_gate = mask_gate.unsqueeze(0)         # (1, L)
        #         if mask_gate.shape[0] != batch_size:           # expand 1→B when CFG is on
        #             mask_gate = mask_gate.expand(batch_size, -1)
        #         mask_gate = mask_gate.to(dtype=query.dtype, device=query.device)
                
        #         query_face = query * mask_gate.view(batch_size, 1, seq_len, 1)
        #         query_bg = query * (1 - mask_gate.view(batch_size, 1, seq_len, 1))
        
        
        query = attn.to_q(hidden_states)
        # attention params
        head_dim = attn.heads
        dim_per_head = hidden_states.shape[-1] // head_dim
        # reshape Q to (B, heads, L, d)
        q = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        # default: no split
        q_bg = q
        q_face = q
        # gate AFTER reshape
        if getattr(self, "mask", None) is not None:
            m = self._prepare_mask(self.mask, seq_len)  # (B_or_1, L[,1])
            if m.dim() == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)                       # (B_or_1, L)
            if m.dim() == 1:
                m = m.unsqueeze(0)                      # (1, L)
            if m.shape[0] != batch_size:
                m = m.expand(batch_size, -1)            # CFG-safe
            m = m.to(dtype=q.dtype, device=q.device).view(batch_size, 1, seq_len, 1)
            q_face = q * m
            # print('q_face masked')
            q_bg   = q * (1.0 - m)
            # print('q_bg masked')


        # === BACKGROUND BRANCH (standard self-attention) ===
        
        # Use noise latents for K/V in background branch
        key_bg = attn.to_k(noise_hidden if ref_hidden is not None else hidden_states)
        value_bg = attn.to_v(noise_hidden if ref_hidden is not None else hidden_states)

        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Compute background attention
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, hidden_states.shape[-1])
        

        # === FACE BRANCH (self-attn but with K/V from reference latents only) ===
        key_face = value_face = None
        
        if ref_hidden is not None:
            # Use reference hidden states directly (already in correct format)
            ref_tok = ref_hidden
             
            # Already in correct shape from UNet processing
            
            # Pad/trim to hidden size
            target_dim = hidden_states.shape[-1]
            if ref_tok.shape[-1] != target_dim:
                diff = target_dim - ref_tok.shape[-1]
                ref_tok = F.pad(ref_tok, (0, diff)) if diff > 0 else ref_tok[..., :target_dim]
            
            # Use standard projections
            key_face = attn.to_k(ref_tok)
            value_face = attn.to_v(ref_tok)
            
            # # Apply reference mask to K/V
            # if hasattr(self, "mask_ref") and self.mask_ref is not None:
            #     kv_len = key_face.shape[1]
            #     mask_ref_flat = self._prepare_mask(self.mask_ref, kv_len)
            #     if mask_ref_flat.dim() == 2:
            #         mask_ref_flat = mask_ref_flat.unsqueeze(-1)
            #     mask_ref_flat = mask_ref_flat.to(dtype=key_face.dtype, device=key_face.device)
            #     key_face = key_face * mask_ref_flat
            #     value_face = value_face * mask_ref_flat
            #     print('key_face and value_face masked')


        if key_face is not None:
            # reshape for MH attention
            key_face   = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)


            # Queries for face branch are already gated in q_face

            
            
            # --- gate K/V by reference mask (same style as Q gate) ---
            if hasattr(self, "mask_ref") and self.mask_ref is not None:
                kv_len = key_face.shape[2]                     # L_kv tokens
                mask_gate_ref = self._prepare_mask(self.mask_ref, kv_len)
                if mask_gate_ref.dim() == 3 and mask_gate_ref.shape[-1] == 1:
                    mask_gate_ref = mask_gate_ref.squeeze(-1)  # (B_or_1, L_kv)
                if mask_gate_ref.dim() == 1:
                    mask_gate_ref = mask_gate_ref.unsqueeze(0) # (1, L_kv)
                if mask_gate_ref.shape[0] != batch_size:       # expand for CFG batches
                    mask_gate_ref = mask_gate_ref.expand(batch_size, -1)
                kv_gate = mask_gate_ref.to(dtype=key_face.dtype,
                                           device=key_face.device).view(batch_size, 1, kv_len, 1)
                key_face   = key_face   * kv_gate
                value_face = value_face * kv_gate
                #  print('key_face and value_face masked')

           
            # Compute face attention
            hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
            hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, hidden_states.shape[-1])
            

            if hasattr(self, 'mask') and self.mask is not None:
                mask = self._prepare_mask(self.mask, seq_len)  # (B_or_1, L[,1])
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)                   # (B_or_1, L, 1)
                if mask.shape[0] != batch_size:
                    mask = mask.expand(batch_size, -1, -1)     # match CFG batch
                mask = mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
                
                hidden_states = hidden_bg * (1 - mask) + hidden_face * mask * self.scale
                # hidden_states = hidden_face * mask * self.scale # TEMP DEBUG WTF          
                # hidden_states = hidden_face  # TEMP DEBUG WTF
                # hidden_states = hidden_bg # TEMP DEBUG WTF
                # hidden_states = hidden_bg * (1 - mask) # TEMP DEBUG WTF    
            else:
                # No mask: weighted blend
                hidden_states = hidden_bg + hidden_face * self.scale
        else:
            # No reference: use background only
            hidden_states = hidden_bg

            # No face embeddings: duplicate background
            hidden_face = hidden_bg.clone()
        
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
            print('residual added')
        
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
        """Prepare mask for attention ops, using **2-D** resize to avoid scanline artifacts."""
        # infer current grid
        h = int(math.isqrt(target_len))
        w = h
        if h * w != target_len:
            # fallback – shouldn't happen in UNet 2D attn
            return mask.view(mask.shape[0], -1, 1)[:, :target_len]

        if mask.ndim == 4:
            # [B, C, H, W] → (2-D resize) → [B, h*w, 1]
            m = F.interpolate(mask[:, :1].float(), size=(h, w), mode="bilinear", align_corners=False)
            return m.flatten(2).transpose(1, 2)
        elif mask.ndim == 3:
            # [B, L, 1] – if L matches, keep; if not, try 2-D path
            if mask.shape[1] == target_len:
                return mask
            L0 = mask.shape[1]
            s = int(math.isqrt(L0))
            if s * s == L0:
                m = mask.transpose(1, 2).reshape(mask.shape[0], 1, s, s)
                m = F.interpolate(m.float(), size=(h, w), mode="bilinear", align_corners=False)
                return m.flatten(2).transpose(1, 2)
            # last resort 1-D (rare)
            return F.interpolate(mask.transpose(1, 2).float(), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        else:
            # [B, *] → [B, L, 1]
            m = mask.view(mask.shape[0], -1, 1).float()
            if m.shape[1] == target_len:
                return m
            return F.interpolate(m.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)


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
        self._face_prompt = None
        self.mask = None
        self.mask_ref = None
        
        # Projections for ID embeddings (face prompts)
        self.to_k_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_id = nn.Linear(cross_attention_dim, hidden_size, bias=False)
    
    # --- NEW: setters used by pipeline patching ---
    def set_masks(self, mask: torch.Tensor, mask_ref: torch.Tensor | None = None):
        self.mask = mask
        self.mask_ref = mask_ref

    def set_face_prompt(self, face_prompt_embeds: torch.Tensor):
        # (B, Lf, C_cross) from text encoder for the "face" prompt
        self._face_prompt = face_prompt_embeds
        
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
        
        # Split batched hidden states if we have both noise and reference
        if batch_size > 1 and encoder_hidden_states is not None:
            # Expect encoder_hidden_states to be [generation_prompt, face_prompt]
            half_batch = batch_size // 2
            noise_hidden = hidden_states[:half_batch]
            ref_hidden = hidden_states[half_batch:]
            gen_prompt = encoder_hidden_states[:half_batch]
            face_prompt = encoder_hidden_states[half_batch:]
            # batch_size = half_batch
            bg_bs = noise_hidden.shape[0]
            fc_bs = ref_hidden.shape[0]
        else:
            noise_hidden = hidden_states
            gen_prompt = encoder_hidden_states
            face_prompt = None
            bg_bs = noise_hidden.shape[0]
            fc_bs = 0
        
        # Handle group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # # Query from hidden states
        # query = attn.to_q(noise_hidden)
        
                
        # Project queries
        q_bg = attn.to_q(noise_hidden)
        q_face_raw = attn.to_q(ref_hidden) if face_prompt is not None else None
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Process text embeddings (generation prompt for background)
        text_embeds = gen_prompt[:, :self.num_tokens, :]
        
        # Optional ID embeds (keep None if absent to avoid NameError later)
        id_embeds = None
        if gen_prompt is not None and gen_prompt.shape[1] > self.num_tokens:
            id_embeds = gen_prompt[:, self.num_tokens:, :]
        
        key_text = attn.to_k(text_embeds)
        value_text = attn.to_v(text_embeds)
        

        
            
        heads = getattr(attn, "heads", 1)
        # reshape via helpers (B,L,C) -> (B*H, L, C/H)
        q_bg     = attn.head_to_batch_dim(q_bg)
        key_text = attn.head_to_batch_dim(key_text)
        value_text = attn.head_to_batch_dim(value_text)

        # # === BRANCHED CROSS-ATTN ===
        # if self.mask is not None and face_prompt is not None:
        #     # Use face_prompt directly (already batched correctly)
        #     # face_ctx = face_prompt[:, :self.num_tokens, :]
        #     face_ctx = face_prompt[:, :self.num_tokens, :].to(dtype=hidden_states.dtype,
        #                                                      device=hidden_states.device)
        
        # === BRANCHED CROSS-ATTN ===
        branched = False
        if self.mask is not None and face_prompt is not None:
            branched = True
            # Face prompt (FC) K/V with FC batch
            face_ctx = face_prompt[:, :self.num_tokens, :].to(dtype=hidden_states.dtype, device=hidden_states.device)
            
            if face_ctx.shape[0] != fc_bs:
                face_ctx = face_ctx[:1].repeat(fc_bs, 1, 1)

            # Project face context
            k_face = attn.head_to_batch_dim(attn.to_k(face_ctx))
            v_face = attn.head_to_batch_dim(attn.to_v(face_ctx))
            q_face = attn.head_to_batch_dim(q_face_raw)
            

            # Gate queries into BG / FACE
            
            
            # token lengths after head_to_batch_dim: (B*H, L, C/H) → L at dim=1
            seq_len_bg = q_bg.shape[1]
            seq_len_fc = q_face.shape[1]
            
            m = self.mask
            # # Normalize mask to (B_or_1, L)
            # if m.ndim == 4:  # (B,1,H,W) → resize to current token grid and flatten
            #     side = int(math.sqrt(seq_len))
            
            # Normalize mask to (B_or_1, L) for BG
            if m.ndim == 4:  # (B,1,H,W) → resize to current token grid and flatten
                side = int(math.sqrt(seq_len_bg))
            
                m = F.interpolate(m.to(dtype=q_bg.dtype, device=q_bg.device),
                                  size=(side, side), mode="nearest")
                m = m.flatten(1)  # (B, L)
            elif m.ndim == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)  # (B_or_1, L)
            elif m.ndim == 1:
                m = m.unsqueeze(0) # (1, L)
            # # Expand batch for CFG etc.
            # if m.shape[0] != batch_size:
            #     m = m.expand(batch_size, -1)
            # # to (B,1,L,1)
            # m = m.to(dtype=query.dtype, device=query.device).view(batch_size, 1, seq_len, 1)
            
            
            
            # q_bg   = query * (1.0 - m)
            # q_face = query * m
            
            # Build masks in (B*H, L, 1)
            def _mask_bh(mv, B, L, H, dev, dt):
                mv = mv.expand(B, -1) if mv.shape[0] != B else mv
                mv = mv.to(dtype=dt, device=dev)              # (B, L)
                mv = mv.unsqueeze(1).expand(B, H, L).reshape(B * H, L, 1)
                return mv
            m_bg = _mask_bh(m, bg_bs, seq_len_bg, heads, q_bg.device, q_bg.dtype)
            m_fc = _mask_bh(m, fc_bs, seq_len_fc, heads, q_face.device, q_face.dtype)
            # Gate queries
            q_bg   = q_bg   * (1.0 - m_bg)
            q_face = q_face * m_fc
            
            # Attend
            h_bg = F.scaled_dot_product_attention(q_bg,   key_text, value_text, dropout_p=0.0, is_causal=False)
            h_fc = F.scaled_dot_product_attention(q_face, k_face,    v_face,    dropout_p=0.0, is_causal=False)
            # Back to (B, L, C) and rebuild full batch [BG, FACE]
            h_bg = attn.batch_to_head_dim(h_bg)
            h_fc = attn.batch_to_head_dim(h_fc)
            hidden_states = torch.cat([h_bg, h_fc * self.scale], dim=0)

            
        else:
            print('Fallback to original text path')
            # Fallback: original text (and optional ID) path
            
            # Non-branched path
            h = F.scaled_dot_product_attention(q_bg, key_text, value_text, dropout_p=0.0, is_causal=False)
            hidden_states = attn.batch_to_head_dim(h)  # (bg_bs, L, C)

       
        # # Process ID embeddings if present
        # if id_embeds is not None:
        # Process ID embeddings if present (skip if branched path used to avoid batch mismatch)
        if id_embeds is not None and not branched:
            key_id = attn.head_to_batch_dim(self.to_k_id(id_embeds))
            value_id = attn.head_to_batch_dim(self.to_v_id(id_embeds))
            hidden_id = F.scaled_dot_product_attention(q_bg, key_id, value_id, dropout_p=0.0, is_causal=False)
            hidden_id = attn.batch_to_head_dim(hidden_id)
            
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
            print('Added residual to cross attn')
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states