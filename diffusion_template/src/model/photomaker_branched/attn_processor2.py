"""
attn_processor.py - Branched attention processors with consistent batch handling
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
        equalize_face_kv = True,
        equalize_clip = (1/3, 8.0)
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
        # Runtime-tunable identity/pose controls (defaults match current behavior)
        self.pose_adapt_ratio: float = 0.25    # POSE_ADAPT_RATIO
        # self.pose_adapt_ratio: float = 0.0    # POSE_ADAPT_RATIO
        self.ca_mixing_for_face: bool = True   # CA_MIXING_FOR_FACE
        self.use_id_embeds: bool = True        # USE_ID_EMBEDS

        # Optional: ID feature cache
        self.id_embeds = None

        ### Modified to make attention processors train ###
        # self.id_to_hidden = None

        # Pre-register so optimizer sees this trainable projection
        self.id_to_hidden = nn.Linear(2048, self.hidden_size, bias=False)
        with torch.no_grad():
            self.id_to_hidden.weight.mul_(0.1)
        ### Modified to make attention processors train ###

        ### 29 Nov - Clean separataion of BA-specific parameters ###
        # Optional branch-specific adapters for face and reference paths (disabled by default).
        self.ba_weights_split: bool = getattr(self, "ba_weights_split", False)
        if self.ba_weights_split:
            # Small residual adapters; initialized near zero so they start as no-op.
            self.face_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.ref_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            with torch.no_grad():
                self.face_adapter.weight.mul_(0.01)
                self.ref_adapter.weight.mul_(0.01)
        else:
            self.face_adapter = None
            self.ref_adapter = None
        ### 29 Nov - Clean separataion of BA-specific parameters ###


        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

        # Training control: which branch to train in self-attn
        # 'both' (default) or 'ref_only'
        self.train_branch_mode: str = getattr(self, "train_branch_mode", "both")

    def set_masks(self, mask: Optional[torch.Tensor], mask_ref: Optional[torch.Tensor] = None):
        """Set masks for current denoising step"""
        # print(f'[TEMP DEBUG] mask in set_masks: {mask}')
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
        
        # Prepare mask
        mask_gate = None
        if self.mask is not None:
            mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)
            mask_gate = mask_gate.to(dtype=q.dtype, device=q.device)
        else:
            raise ValueError("Branched attention requires a mask for the background branch")
        
        # --- quick check 
        
        # print a few times per run only
        if not hasattr(self, "_dbg_sa"): self._dbg_sa = 0
        if self._dbg_sa < 3 and mask_gate is not None:
            mg = mask_gate.float()
            if full_debug:
                print(f"[BrSA] L={seq_len}  mask_mean={mg.mean().item():.4f}  mask_>0.5={(mg>0.5).float().mean().item():.4f}")

        
        # === BACKGROUND BRANCH ===
        # Q: background from noise, K/V: full noise
        key_bg = attn.to_k(noise_hidden)
        value_bg = attn.to_v(noise_hidden)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is not None:
            q_bg = q * (1.0 - mask_gate) # non-face area of noise_hidden
        else:
            q_bg = q
            raise ValueError("Branched attention requires a mask for the background branch")
            
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        # === BACKGROUND BRANCH ===

        
        # === FACE BRANCH ===
        # Q: face from noise, K/V: face from reference
        # key_face = attn.to_k(ref_hidden)
        # value_face = attn.to_v(ref_hidden)
        
        
        # CRITICAL FIX: Apply face mask to reference K/V to isolate face features

        if mask_gate is not None:
            # mask_flat = mask_gate.squeeze(1).squeeze(-1)
            mask_flat = mask_gate.squeeze(1).squeeze(-1).to(dtype=hidden_bg.dtype)
            if mask_flat.dim() == 1:
                mask_flat = mask_flat.unsqueeze(0)
            if mask_flat.dim() == 2:
                mask_flat = mask_flat.unsqueeze(-1)

        # Mix reference face with noise face to allow pose adaptation
        # # POSE_ADAPT_RATIO = 0.0  # 0 = full reference (current), 1 = full noise (loses identity)
        # # POSE_ADAPT_RATIO = 0.15
        # POSE_ADAPT_RATIO = 0.25
        # # POSE_ADAPT_RATIO = 0.5
        # # POSE_ADAPT_RATIO = 1.0

        # # CA_MIXING_FOR_FACE = False
        # CA_MIXING_FOR_FACE = True

        # --- use runtime-tunable values instead of hard-coded locals ---
        POSE_ADAPT_RATIO   = getattr(self, "pose_adapt_ratio", 0.25)
        CA_MIXING_FOR_FACE = getattr(self, "ca_mixing_for_face", True)

        # Check if we're in pre-PhotoMaker state and override POSE_ADAPT_RATIO
        if hasattr(self, "_disable_reference") and self._disable_reference:
            original_ratio = POSE_ADAPT_RATIO
            POSE_ADAPT_RATIO = 1.0  # Use only current noise, no reference
            if not hasattr(self, "_printed_force"):
                print(f"[BranchedAttn] Forcing POSE_ADAPT_RATIO=1.0 (was {original_ratio:.2f}) - pre-PhotoMaker state")
                self._printed_force = True
        elif hasattr(self, "_printed_force") and self._printed_force:
            print(f"[BranchedAttn] Relaxing POSE_ADAPT_RATIO back to {POSE_ADAPT_RATIO:.2f}")
            self._printed_force = F

        # # # USE_ID_EMBEDS = False
        # # USE_ID_EMBEDS = True

        # USE_ID_EMBEDS = getattr(self, "use_id_embeds", True)

        # # print(f'[BrSA] L={seq_len}  POSE_ADAPT_RATIO={POSE_ADAPT_RATIO}  CA_MIXING_FOR_FACE={CA_MIXING_FOR_FACE}  USE_ID_EMBEDS={USE_ID_EMBEDS}')

        # if not USE_ID_EMBEDS:
        #     self.id_embeds = None

        # Always use ID features whenever they're provided on the processor.
        # Ignore any pipeline/runtime flag to "disable" them.
        USE_ID_EMBEDS = self.id_embeds is not None

        
        if self.mask_ref is not None:
            ref_mask = self._prepare_mask(self.mask_ref, seq_len, batch_size)
            ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
            ref_mask_flat = ref_mask.squeeze(1).squeeze(-1)
            if ref_mask_flat.dim() == 2:
                ref_mask_flat = ref_mask_flat.unsqueeze(-1)
            # Isolate face region in reference
            # ref_face_hidden = ref_hidden * ref_mask_flat

            # Extract face regions from both noise and reference
            noise_face_hidden = noise_hidden * mask_flat  # Face from current noise
            
            ref_face_hidden = ref_hidden * ref_mask_flat   # Face from reference

            # Blend them to allow pose adaptation while preserving identity
            # Higher POSE_ADAPT_RATIO = more pose flexibility, less identity preservation
            face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden
            

            # # NEW: Inject ID embeddings into the mixed face hidden states
            # if hasattr(self, 'id_embeds') and self.id_embeds is not None:

            # # Inject ID embeddings into the mixed face hidden states (2048-D → hidden)
            # if USE_ID_EMBEDS:
            #     if not hasattr(self, 'id_to_hidden') or self.id_to_hidden is None:
            #         self.id_to_hidden = nn.Linear(
            #             self.id_embeds.shape[-1], 
            #             face_hidden_mixed.shape[-1],
            #             bias=False
            #         ).to(face_hidden_mixed.device, face_hidden_mixed.dtype)
            #         # Initialize with small weights
            #         with torch.no_grad():
            #             self.id_to_hidden.weight.mul_(0.1)
                
            ### Modified to make attention processors train ###
            if USE_ID_EMBEDS:
                # Ensure dtype/device match before use
                self.id_to_hidden = self.id_to_hidden.to(
                    device=face_hidden_mixed.device, dtype=face_hidden_mixed.dtype
                )
                id_features = self.id_to_hidden(self.id_embeds)

            ### Modified to make attention processors train ###

                id_features = self.id_to_hidden(self.id_embeds)
                if id_features.dim() == 2:
                    id_features = id_features.unsqueeze(1).expand(-1, face_hidden_mixed.shape[1], -1)
                
                # Blend ID features with the mixed face
                # id_alpha = 0.3  # Control ID influence strength
                # id_alpha = 1.0
                # id_alpha = 0.7
                id_alpha = 0.4
                face_hidden_mixed = face_hidden_mixed * (1 - id_alpha) + id_features * id_alpha

            ### 29 Nov - Clean separataion of BA-specific parameters ###
            # Apply face-branch adapter if enabled (separate from background branch).
            if getattr(self, "ba_weights_split", False) and self.face_adapter is not None:
                face_hidden_mixed = face_hidden_mixed + self.face_adapter(face_hidden_mixed)
            ### 29 Nov - Clean separataion of BA-specific parameters ###
            
            
            if CA_MIXING_FOR_FACE:
            #     # Combine face from reference with face from noise for K/V
            #     # This allows the model to learn pose from noise while preserving identity from reference
            #     noise_face = noise_hidden * mask_flat if mask_gate is not None else noise_hidden
            #     ref_face = ref_hidden * ref_mask_flat

            #     # Stack both sources for keys and values
            #     combined_face_hidden = torch.cat([ref_face, noise_face], dim=1)  # Double sequence length

            
            # # Blend them to allow pose adaptation while preserving identity
            # # Higher POSE_ADAPT_RATIO = more pose flexibility, less identity preservation
            # face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden

               # Use blended face and pure noise face for K/V
                # This allows attending to both pose-adapted reference and current noise
                combined_face_hidden = torch.cat([face_hidden_mixed, noise_face_hidden], dim=1)  # Double sequence length
            else:
                # Just use the blended face directly
                ref_face_hidden = face_hidden_mixed


            # ref_face_hidden = face_hidden_mixed

        else:
            if not CA_MIXING_FOR_FACE:
                ref_face_hidden = ref_hidden
            else:
                combined_face_hidden = ref_hidden
            raise ValueError("Branched attention requires a mask for the reference branch")

        if not CA_MIXING_FOR_FACE:
            key_face = attn.to_k(ref_face_hidden)
            value_face = attn.to_v(ref_face_hidden)
        else:
            key_face = attn.to_k(combined_face_hidden)
            value_face = attn.to_v(combined_face_hidden)

        # REF_MASK_TO_KV = True
        REF_MASK_TO_KV = False

        if REF_MASK_TO_KV:
            # Apply reference mask to K/V if available
            if self.mask_ref is not None:
                ref_mask = self._prepare_mask(self.mask_ref, seq_len, batch_size)
                ref_mask = ref_mask.to(dtype=key_face.dtype, device=key_face.device)
                ref_mask_flat = ref_mask.squeeze(1).squeeze(-1)
                if ref_mask_flat.dim() == 2:
                    ref_mask_flat = ref_mask_flat.unsqueeze(-1)
                key_face = key_face * ref_mask_flat # face part of ref noise
                value_face = value_face * ref_mask_flat # face part of ref noise
            else:
                raise ValueError("Branched attention requires a mask for the face branch")
        else:
            pass
            # ⚠️ Do NOT mask reference K/V here: collapsing keys with a tight
            # face mask causes scrambled features. Gate only the queries (q_face)
            # and the final merge with the face mask.

        key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        if mask_gate is not None:
            q_face = q * mask_gate # face area of noise_hidden
        else:
            q_face = q
            raise ValueError("Branched attention requires a mask for the face branch")
            
        hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, dropout_p=0.0, is_causal=False)
        hidden_face = hidden_face.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        # === FACE BRANCH ===


        #### DEBUG ####
        if full_debug:
            # --- quick check 1
            if self._dbg_sa < 3:
                bg = hidden_bg.float(); fc = hidden_face.float()
                cos = torch.nn.functional.cosine_similarity(bg.flatten(1), fc.flatten(1), dim=1).mean().item()
                print(f"[BrSA]  σ(bg)={bg.std().item():.4f}  σ(face)={fc.std().item():.4f}  cos(bg,face)={cos:.3f}")


        # --- quick check 2 (broadcast-safe) ---
        if self._dbg_sa < 3 and mask_gate is not None:
            mf = mask_gate
            # normalize shapes -> [B, L]
            if mf.dim() == 4:            # [B, 1, H, W]
                mf = mf.flatten(1)
            elif mf.dim() == 3:          # [B, 1, L] or [B, L, 1]
                if mf.shape[1] == 1:
                    mf = mf.squeeze(1)   # -> [B, L]
                else:
                    mf = mf.flatten(1)   # be safe
            elif mf.dim() == 1:
                mf = mf.unsqueeze(0)     # [1, L]

            BHL, L, D = hidden_bg.shape  # e.g. [B*heads, tokens, dim]
            # repeat mask across heads if needed
            if mf.shape[0] != BHL:
                rep = BHL // mf.shape[0]
                mf = mf.repeat_interleave(rep, dim=0)

            # align token length and add feature axis -> [B*H, L, 1]
            mf = mf[:, :L].unsqueeze(-1).to(dtype=hidden_bg.dtype, device=hidden_bg.device)

            contrib_bg   = (hidden_bg * (1 - mf)).float().std().item()
            contrib_face = (hidden_face * mf).float().std().item()
            if full_debug:
                print(f"[BrSA]  merge parts σ: bg_part={contrib_bg:.4f}  face_part={contrib_face:.4f}")
        self._dbg_sa += 1
        #### DEBUG ####

        ### NEW DEBUG 12 AUG
        # print(f"BG std: {hidden_bg.std()}, Face std: {hidden_face.std()}")
        ### NEW DEBUG 12 AUG


        # === NEW BRANCH - SELF-ATTN FOR REFERENCE ===
        # Q: face from reference, K/V: face from as well
        key_ref = attn.to_k(ref_hidden)
        value_ref = attn.to_v(ref_hidden)
        query_ref = attn.to_q(ref_hidden)
        
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

        ### 29 Nov - Clean separataion of BA-specific parameters ###
        # Apply reference-branch adapter if enabled (separate from background branch).
        if getattr(self, "ba_weights_split", False) and self.ref_adapter is not None:
            hidden_ref = hidden_ref + self.ref_adapter(hidden_ref)
        ### 29 Nov - Clean separataion of BA-specific parameters ###


        # Optionally restrict training to the reference branch only
        if getattr(self, "train_branch_mode", "both") == "ref_only":
            hidden_bg = hidden_bg.detach()
            hidden_face = hidden_face.detach()

        # === MERGE ===
        if mask_gate is not None:
            # mask_flat = mask_gate.squeeze(1).squeeze(-1)
            mask_flat = mask_gate.squeeze(1).squeeze(-1).to(dtype=hidden_bg.dtype)
            if mask_flat.dim() == 1:
                mask_flat = mask_flat.unsqueeze(0)
            if mask_flat.dim() == 2:
                mask_flat = mask_flat.unsqueeze(-1)
            
            # NORMALIZE = True
            NORMALIZE = False
            
            if not NORMALIZE:
                merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
                # merged = hidden_bg # DEBUG ONLY
                # merged = hidden_face # DEBUG ONLY
            else:
                # CRITICAL FIX: Normalize branches before merging
                bg_norm = hidden_bg / (hidden_bg.std() + 1e-6)
                face_norm = hidden_face / (hidden_face.std() + 1e-6)
                
                # Merge with normalization
                merged_norm = bg_norm * (1 - mask_flat) + face_norm * mask_flat * self.scale
                # Rescale to match expected magnitude
                merged = merged_norm * hidden_bg.std()
        else:
            merged = hidden_bg + hidden_face * self.scale
            print('warning - no mask on merge')
            raise ValueError("Branched attention requires a mask for the background branch")
        
        # Combine: [merged_result, face_branch_output]
        # hidden_states = torch.cat([merged, hidden_face], dim=0) # merged = updated noise and face branch output

        # NEW - now using hidden_ref for referenc part of output
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
        
        # --- DDP safety: ensure this processor's trainable param participates every step ---
        # Touch a scalar that depends on id_to_hidden so DDP never marks it "unused".
        try:
            w = self.id_to_hidden.weight
            if w.device != hidden_states.device or w.dtype != hidden_states.dtype:
                w = w.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states + (w.reshape(-1)[:1].sum() * 0.0)
        except Exception:
            pass
        # --- DDP safety: ensure this processor's trainable param participates every step ---

            
        return hidden_states
    
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops — always resize in 2-D (no 1-D raster)."""
        H = int(math.sqrt(target_len))
        W = H
        assert H * W == target_len, f"seq_len {target_len} is not square"
        if mask.ndim == 4:  # [B, C, H0, W0]
            m2d = F.interpolate(mask[:, :1].float(), size=(H, W), mode="bilinear", align_corners=False)
        else:               # [B, L, 1] or [B, 1, L] → [B,1,H0,W0] first
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = int(math.sqrt(L0)); w0 = h0
            assert h0 * w0 == L0, f"mask length {L0} not square"
            m2d = mask.view(mask.shape[0], -1)[:, :L0].float().view(mask.shape[0], 1, h0, w0)
            m2d = F.interpolate(m2d, size=(H, W), mode="bilinear", align_corners=False)
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

        self.id_embeds = None
        self.id_to_hidden = None
        # Accept cross_attention_kwargs to avoid noisy warnings
        self.has_cross_attention_kwargs = True
    
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
        id_embedding: Optional[torch.Tensor] = None,
        id_scale: float = 1.0,
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
        
        if encoder_hidden_states is not None:
            gen_prompt = encoder_hidden_states[:half_batch]
            face_prompt = encoder_hidden_states[half_batch:]
        else:
            # Self-attention fallback
            gen_prompt = noise_hidden
            face_prompt = ref_hidden
            raise ValueError ("Branched cross-attention requires encoder_hidden_states")
        
        # batch_size = half_batch


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

        seq_len = noise_hidden.shape[1]
        
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


        # hidden_states = torch.cat([merged, hidden_face], dim=0)


        
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
            m = m.expand(batch_size, -1, -1)
            
        # Reshape for multi-head attention [B, 1, L, 1]
        return m.view(batch_size, 1, target_len, 1)
