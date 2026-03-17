"""
attn_processor.py - Branched attention processors with consistent batch handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def _clone_effective_linear(attn_linear, adapter_name: str = "default") -> nn.Linear:
    base = attn_linear.get_base_layer() if hasattr(attn_linear, "get_base_layer") else attn_linear
    cloned = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        weight = base.weight.detach().clone()
        if hasattr(attn_linear, "lora_A") and adapter_name in attn_linear.lora_A:
            weight = weight + attn_linear.get_delta_weight(adapter_name).detach().to(weight.device, weight.dtype)
        cloned.weight.copy_(weight)
        if base.bias is not None:
            cloned.bias.copy_(base.bias.detach())
    return cloned


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
        branched_attn_weight_mode: str = "shared",
    ):
        super().__init__()

        # print("[DEBUG] Using attn_processor_clean.py")
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale
        self.branched_attn_weight_mode = (branched_attn_weight_mode or "shared").lower()
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        
        # If True: keep masks strictly binary after resize (avoids soft boundary blending)
        self.force_binary_masks: bool = True # False
        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

    def init_from_attention(self, attn) -> None:
        mode = self.branched_attn_weight_mode
        if mode in {"ref_only", "noise_and_ref"}:
            self.ref_to_q = _clone_effective_linear(attn.to_q)
            self.ref_to_k = _clone_effective_linear(attn.to_k)
            self.ref_to_v = _clone_effective_linear(attn.to_v)
        if mode == "noise_and_ref":
            self.noise_to_q = _clone_effective_linear(attn.to_q)
            self.noise_to_k = _clone_effective_linear(attn.to_k)
            self.noise_to_v = _clone_effective_linear(attn.to_v)

    def _q_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_q if self.noise_to_q is not None else attn.to_q
        return layer(hidden_states)

    def _k_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_k if self.noise_to_k is not None else attn.to_k
        return layer(hidden_states)

    def _v_noise(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.noise_to_v if self.noise_to_v is not None else attn.to_v
        return layer(hidden_states)

    def _q_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_q if self.ref_to_q is not None else attn.to_q
        return layer(hidden_states)

    def _k_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_k if self.ref_to_k is not None else attn.to_k
        return layer(hidden_states)

    def _v_ref(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.ref_to_v if self.ref_to_v is not None else attn.to_v
        return layer(hidden_states)

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
        
    ) -> torch.Tensor:
        """
        Process self-attention with face/background branching.
        
        Input: doubled batch [noise_hidden, ref_hidden]
        Output: doubled batch [merged_hidden, face_hidden]
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
        query = self._q_noise(attn, noise_hidden)
        
        # Reshape for multi-head attention
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim
        q = query.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Prepare mask
        mask_gate = None
        if self.mask is  None:
            raise ValueError("Branched attention requires a mask for the background branch")
        
        # mask is injected by branched_runtime.patch_unet_attention_processors(...)
        mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)
        mask_gate = mask_gate.to(dtype=q.dtype, device=q.device)
        

        # ======================================== BACKGROUND BRANCH ==========================================================
        # Q: background from noise, K/V: full noise
        key_bg = self._k_noise(attn, noise_hidden)
        value_bg = self._v_noise(attn, noise_hidden)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is None:
            raise ValueError("Branched attention requires a mask for the background branch")
        
        q_bg = q * (1.0 - mask_gate) # non-face area of noise_hidden
            
            
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        # ======================================== BACKGROUND BRANCH ==========================================================
        




        # ======================================== FACE BRANCH ================================================================
        # Q: face from noise, K/V: face from reference
        # key_face = attn.to_k(ref_hidden)
        # value_face = attn.to_v(ref_hidden)
        
        if mask_gate is None:
            raise ValueError("mask_gate is required for face branch")

        mask_flat = mask_gate.squeeze(1).to(dtype=hidden_bg.dtype)  # [B, L, 1]

        # --- use runtime-tunable values instead of hard-coded locals ---
        # POSE_ADAPT_RATIO   = getattr(self, "pose_adapt_ratio", 0.25)
        # CA_MIXING_FOR_FACE = getattr(self, "ca_mixing_for_face", True)
        
        # Runtime values are passed via UNet cross_attention_kwargs
        runtime = cross_attention_kwargs if isinstance(cross_attention_kwargs, dict) else {}
        POSE_ADAPT_RATIO = 0.0 # hardcoded to 0.0 for simplicity
        CA_MIXING_FOR_FACE = False # hardcoded to False for simplicity


        # #### Check if we're in pre-PhotoMaker state (and override POSE_ADAPT_RATIO) ####
        # if hasattr(self, "_disable_reference") and self._disable_reference:
        #     original_ratio = POSE_ADAPT_RATIO
        #     POSE_ADAPT_RATIO = 1.0  # Use only current noise, no reference
        #     if not hasattr(self, "_printed_force"):
        #         print(f"[BranchedAttn] Forcing POSE_ADAPT_RATIO=1.0 (was {original_ratio:.2f}) - pre-PhotoMaker state")
        #         self._printed_force = True
        # elif hasattr(self, "_printed_force") and self._printed_force:
        #     print(f"[BranchedAttn] Relaxing POSE_ADAPT_RATIO back to {POSE_ADAPT_RATIO:.2f}")
        #     self._printed_force = F
        # #### Check if we're in pre-PhotoMaker state (and override POSE_ADAPT_RATIO) ####
        


        
        if self.mask_ref is None:
            raise ValueError("Branched attention requires a mask for the reference branch")

        ref_mask = self._prepare_mask(self.mask_ref, seq_len, batch_size)
        ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
        ref_mask_flat = ref_mask.squeeze(1)  # [B, L, 1]


        # Extract face regions from both noise and reference
        noise_face_hidden = noise_hidden * mask_flat  # Face from current noise
        ref_face_hidden = ref_hidden * ref_mask_flat   # Face from reference

        # Blend them to allow pose adaptation while preserving identity
        # Higher POSE_ADAPT_RATIO = more pose flexibility, less identity preservation
        face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden
        
        # Just use the blended face directly (previously had option for CA_MIXING_FOR_FACE but removed for simplicity)
        key_face = self._k_ref(attn, face_hidden_mixed)
        value_face = self._v_ref(attn, face_hidden_mixed)


        key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is  None:
            raise ValueError("Branched attention requires a mask for the face branch")
        
        q_face = q * mask_gate # face area of noise_hidden
            
        hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])



        # ======================================== FACE BRANCH ================================================================
        

        # === NEW BRANCH - SELF-ATTN FOR REFERENCE ===
        # Q: face from reference, K/V: face from as well
        key_ref = self._k_ref(attn, ref_hidden)
        value_ref = self._v_ref(attn, ref_hidden)
        query_ref = self._q_ref(attn, ref_hidden)
        
        # Reshape for multi-head attention
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim
        query_ref = query_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        key_ref = key_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # hidden_ref needs to be without any masks
        hidden_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        # === NEW BRANCH - SELF-ATTN FOR REFERENCE ===


        # === MERGE ===
        if mask_gate is  None:
            raise ValueError("Branched attention requires a mask for the background branch")

        mask_flat = mask_gate.squeeze(1).to(dtype=hidden_bg.dtype)  # [B, L, 1]
        
        merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
    
        
        # Combine:
        hidden_states = torch.cat([merged, hidden_ref], dim=0) # merged = updated noise and face branch output

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        
        # Reshape if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch, channel, height, width
            )

        # Add residual # TODO check if neeeded / do separately for each branch
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor # TODO check if neeeded / do separately for each branch
        
        return hidden_states
    
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops — always resize in 2-D (no 1-D raster)."""
        H = int(math.sqrt(target_len))
        W = H
        assert H * W == target_len, f"seq_len {target_len} is not square"
        
        B = mask.shape[0]
        if mask.ndim == 4:  # [B, C, H0, W0]
            m4 = mask[:, :1].float()              # [B,1,H0,W0]
        else:               # [B, L, 1] or [B, 1, L] → [B,1,H0,W0] first
            flat = mask.reshape(B, -1).float()    # [B,L0]
            h0 = int(math.isqrt(flat.shape[1]))
            assert h0 * h0 == flat.shape[1], f"mask length {flat.shape[1]} not square"
            m4 = flat.reshape(B, 1, h0, h0)       # [B,1,h0,w0]

        m2d = F.interpolate(m4, size=(H, W), mode="bilinear", align_corners=False)
                    
                    
        if getattr(self, "force_binary_masks", False):
            m2d = (m2d > 0.5).to(dtype=m2d.dtype)
        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        
        # Expand for batch if needed
        if m.shape[0] != batch_size:
            # --- ADDED For training integration ---
            reps = (batch_size + m.shape[0] - 1) // m.shape[0]
            # --- ADDED For training integration ---
            m = m.repeat(reps, 1, 1)[:batch_size]
            
        # Reshape for multi-head attention [B, 1, L, 1]
        return m.view(batch_size, 1, target_len, 1)
    
    
    def _standard_cross_attention(self, attn, hidden_states, encoder_hidden_states, 
                                  attention_mask, residual, input_ndim):
        """Standard cross-attention (delegates to cross-attention processor if available)"""
        # This is just a fallback - the actual branched cross-attention 
        # is handled by BranchedCrossAttnProcessor
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
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            channel = residual.shape[1]
            height = width = int(math.sqrt(hidden_states.shape[1]))
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

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

        self.has_cross_attention_kwargs = True # Accept cross_attention_kwargs to avoid noisy warnings
    
    def set_masks(self, mask: torch.Tensor, mask_ref: Optional[torch.Tensor] = None):
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
    ) -> torch.Tensor:
        """
        Process cross-attention with branching ONLY for the first half.
        
        Inputs:
        - hidden_states: doubled batch [noise_hidden, ref_hidden]
        - encoder_hidden_states: doubled batch [generation_prompt, face_prompt]
        
        Output: doubled batch [merged_result, ref_standard_result]
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
        
        # Split doubled batches
        total_batch = hidden_states.shape[0]
        half_batch = total_batch // 2
        
        noise_hidden = hidden_states[:half_batch]
        ref_hidden = hidden_states[half_batch:]
        
        if encoder_hidden_states is None:
            raise ValueError ("Branched cross-attention requires encoder_hidden_states")
        
        gen_prompt = encoder_hidden_states[:half_batch]
        face_prompt = encoder_hidden_states[half_batch:]
            
    


        # Ensure encoder prompts match the **latent half-batch** (handles num_images_per_prompt > 1)
        batch_size = half_batch
        if gen_prompt.shape[0] != batch_size:
            # tile or repeat to match, then trim
            rep = (batch_size + gen_prompt.shape[0] - 1) // gen_prompt.shape[0]
            gen_prompt = gen_prompt.repeat(rep, 1, 1)[:batch_size].contiguous()
        if face_prompt.shape[0] != batch_size:
            rep = (batch_size + face_prompt.shape[0] - 1) // face_prompt.shape[0]
            face_prompt = face_prompt.repeat(rep, 1, 1)[:batch_size].contiguous()

        # Defensive: recompute from tensors actually used below
        batch_size = noise_hidden.shape[0]

        
        # Handle group norm
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
        
        # ========== PROCESS FIRST HALF (NOISE BATCH) WITH BRANCHING ==========
        
        # Compute query from noise
        query_bg = attn.to_q(noise_hidden)
        
        # Get attention parameters
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim

        q_bg = query_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Compute query from ref
        query_ref = attn.to_q(ref_hidden)

        # Get attention parameters
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim

        q_ref = query_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # === BACKGROUND BRANCH ===
        # Q: background from noise, K/V: generation prompt
        key_bg = attn.to_k(gen_prompt)
        value_bg = attn.to_v(gen_prompt)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # === FACE BRANCH ===
        # Q: face from noise, K/V: face prompt (should be different from gen_prompt!)
        key_ref = attn.to_k(face_prompt)
        value_ref = attn.to_v(face_prompt)
        key_ref = key_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        hidden_ref = F.scaled_dot_product_attention(q_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        
        
        # ========== COMBINE RESULTS ==========
        hidden_states = torch.cat([hidden_bg, hidden_ref], dim=0)

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
        
        # Expand for batch if needed
        if m.shape[0] != batch_size:
            # m = m.expand(batch_size, -1, -1)
            m = m.repeat((batch_size + m.shape[0] - 1) // m.shape[0], 1, 1)[:batch_size]
            
        # Reshape for multi-head attention [B, 1, L, 1]
        return m.view(batch_size, 1, target_len, 1)
