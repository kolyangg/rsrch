"""
attn_processor.py - Branched attention processors with consistent batch handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class BranchLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: Optional[int] = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.scaling = float(alpha if alpha is not None else rank) / float(rank)
        self.register_buffer("base_weight", torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.register_buffer("base_bias", torch.empty(out_features, device=device, dtype=dtype) if bias else None)
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base_weight, self.base_bias) + F.linear(
            F.linear(x, self.lora_A),
            self.lora_B,
        ) * self.scaling


def _clone_effective_linear(
    attn_linear,
    *,
    kind: str,
    rank: int,
    alpha: Optional[int] = None,
    adapter_name: str = "default",
):
    base = attn_linear.get_base_layer() if hasattr(attn_linear, "get_base_layer") else attn_linear
    if kind == "full":
        cloned = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
    elif kind == "lora":
        cloned = BranchLoRALinear(
            base.in_features,
            base.out_features,
            rank=rank,
            alpha=alpha,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
    else:
        raise ValueError(f"Unknown branched_attn_new_weight_kind: {kind}")
    with torch.no_grad():
        weight = base.weight.detach().clone()
        if hasattr(attn_linear, "lora_A") and adapter_name in attn_linear.lora_A:
            weight = weight + attn_linear.get_delta_weight(adapter_name).detach().to(weight.device, weight.dtype)
        if kind == "full":
            cloned.weight.copy_(weight)
            if base.bias is not None:
                cloned.bias.copy_(base.bias.detach())
        else:
            cloned.base_weight.copy_(weight)
            if base.bias is not None:
                cloned.base_bias.copy_(base.bias.detach())
    return cloned


def _branch_batch_sizes(mask, total_batch):
    if mask is None:
        if total_batch % 2 != 0:
            raise RuntimeError(f"Cannot infer branch sizes from total_batch={total_batch}")
        gen_batch = total_batch // 2
    else:
        gen_batch = int(mask.shape[0])
    ref_batch = total_batch - gen_batch
    if ref_batch != gen_batch:
        raise RuntimeError(
            f"Invalid branched batch: total={total_batch}, generation={gen_batch}, "
            f"reference={ref_batch}; expected one reference per sample"
        )
    return gen_batch, ref_batch


def _infer_spatial_hw(target_len: int, mask: Optional[torch.Tensor] = None) -> tuple[int, int]:
    """Infer the 2-D attention grid, preserving mask aspect ratio when possible."""
    if mask is not None and mask.ndim == 4 and mask.shape[-2] > 0 and mask.shape[-1] > 0:
        src_h, src_w = int(mask.shape[-2]), int(mask.shape[-1])
        ratio = src_h / max(float(src_w), 1.0)
        h0 = max(1, int(round(math.sqrt(target_len * ratio))))
        candidates = []
        for h in range(max(1, h0 - 4), h0 + 5):
            if target_len % h == 0:
                w = target_len // h
                candidates.append((abs((h / max(float(w), 1.0)) - ratio), h, w))
        if candidates:
            _, h, w = min(candidates, key=lambda item: item[0])
            return h, w

    h = int(math.isqrt(target_len))
    if h * h == target_len:
        return h, h
    raise AssertionError(f"seq_len {target_len} is not square and no 2-D mask aspect was available")


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
        branched_attn_new_weight_kind: str = "full",
        branched_attn_lora_rank: int = 16,
    ):
        super().__init__()

        # print("[DEBUG] Using attn_processor_clean.py")
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.scale = scale
        self.branched_attn_weight_mode = (branched_attn_weight_mode or "shared").lower()
        self.branched_attn_new_weight_kind = (branched_attn_new_weight_kind or "full").lower()
        self.branched_attn_lora_rank = int(branched_attn_lora_rank)
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        self.ba_enable_runtime_sa_knobs: bool = False
        self.pose_adapt_ratio: float = 0.0
        self.ca_mixing_for_face: bool = False
        self.use_id_embeds: bool = False
        self.id_alpha: float = 0.3
        self.id_embeds = None
        self.id_to_hidden = None
        self.ba_face_fusion_mode: str = "legacy"
        self.ba_face_fusion_gate_init: float = 0.25
        self.ba_face_fusion_gate_max: float = 1.0
        self.face_fusion_logit = None
        self.num_heads = None
        
        # If True: keep masks strictly binary after resize (avoids soft boundary blending)
        self.force_binary_masks: bool = True # False
        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

    def init_from_attention(self, attn) -> None:
        self.num_heads = int(attn.heads)
        mode = self.branched_attn_weight_mode
        if mode in {"ref_only", "noise_and_ref"}:
            self.ref_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.ref_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.ref_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
        if mode == "noise_and_ref":
            self.noise_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.noise_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.noise_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )

    def configure_face_fusion(self) -> None:
        mode = str(getattr(self, "ba_face_fusion_mode", "legacy") or "legacy").lower()
        if mode not in {"legacy", "dual_attention_gate"}:
            raise ValueError(f"Unknown ba_face_fusion_mode: {mode}")
        if mode != "dual_attention_gate" or self.face_fusion_logit is not None:
            return
        if not self.num_heads:
            raise RuntimeError("init_from_attention must run before configuring face fusion")

        gate_max = float(getattr(self, "ba_face_fusion_gate_max", 1.0))
        if not 0.0 < gate_max <= 1.0:
            raise ValueError(f"ba_face_fusion_gate_max must be in (0, 1], got {gate_max}")
        gate_init = float(getattr(self, "ba_face_fusion_gate_init", 0.25))
        if not 0.0 <= gate_init <= gate_max:
            raise ValueError(
                f"ba_face_fusion_gate_init must be in [0, {gate_max}], got {gate_init}"
            )
        relative_init = min(max(gate_init / gate_max, 1e-4), 1.0 - 1e-4)
        logit = math.log(relative_init / (1.0 - relative_init))
        template = next(self.parameters())
        self.face_fusion_logit = nn.Parameter(
            torch.full(
                (self.num_heads,),
                logit,
                device=template.device,
                dtype=template.dtype,
            )
        )

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
        
        
        total_batch = hidden_states.shape[0]
        batch_size, ref_batch_size = _branch_batch_sizes(self.mask, total_batch)
        noise_hidden = hidden_states[:batch_size]
        ref_hidden = hidden_states[batch_size:]
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
        # Q: background from noise, K/V: full noise (or face-suppressed noise in strict mode)
        strict_face_routing = bool(getattr(self, "strict_face_routing", False))
        bg_source = noise_hidden
        if strict_face_routing:
            bg_source = noise_hidden * (1.0 - mask_gate.squeeze(1).to(dtype=noise_hidden.dtype, device=noise_hidden.device))
        key_bg = self._k_noise(attn, bg_source)
        value_bg = self._v_noise(attn, bg_source)
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
        
        # Preserve current behavior by default. New runtime SA controls are opt-in
        # through ba_enable_runtime_sa_knobs to keep old checkpoints/configs stable.
        use_runtime_sa_knobs = bool(getattr(self, "ba_enable_runtime_sa_knobs", False))
        POSE_ADAPT_RATIO = float(getattr(self, "pose_adapt_ratio", 0.0)) if use_runtime_sa_knobs else 0.0
        CA_MIXING_FOR_FACE = bool(getattr(self, "ca_mixing_for_face", False)) if use_runtime_sa_knobs else False
        USE_ID_EMBEDS = bool(
            use_runtime_sa_knobs
            and getattr(self, "use_id_embeds", False)
            and self.id_embeds is not None
        )


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

        ref_mask = self._prepare_mask(self.mask_ref, seq_len, ref_batch_size)
        ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
        ref_mask_flat = ref_mask.squeeze(1)  # [B, L, 1]


        # Extract face regions from both noise and reference
        noise_face_hidden = noise_hidden * mask_flat  # Face from current noise
        ref_face_hidden = ref_hidden * ref_mask_flat

        fusion_mode = str(getattr(self, "ba_face_fusion_mode", "legacy") or "legacy").lower()

        # Dual attention keeps the two spatial grids separate; legacy directly mixes same-index tokens.
        if fusion_mode == "dual_attention_gate":
            face_hidden_mixed = ref_face_hidden
        else:
            face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden

        if USE_ID_EMBEDS:
            if self.id_to_hidden is None:
                self.id_to_hidden = nn.Linear(
                    int(self.id_embeds.shape[-1]),
                    int(face_hidden_mixed.shape[-1]),
                    bias=False,
                ).to(face_hidden_mixed.device, face_hidden_mixed.dtype)
                with torch.no_grad():
                    self.id_to_hidden.weight.mul_(0.1)
            id_embeds = self.id_embeds.to(device=face_hidden_mixed.device, dtype=face_hidden_mixed.dtype)
            if id_embeds.shape[0] != batch_size:
                reps = (batch_size + id_embeds.shape[0] - 1) // id_embeds.shape[0]
                id_embeds = id_embeds.repeat((reps,) + (1,) * (id_embeds.ndim - 1))[:batch_size]
            id_features = self.id_to_hidden(id_embeds)
            if id_features.dim() == 2:
                id_features = id_features.unsqueeze(1).expand(-1, face_hidden_mixed.shape[1], -1)
            id_alpha = float(getattr(self, "id_alpha", 0.3))
            face_hidden_mixed = face_hidden_mixed * (1.0 - id_alpha) + id_features * id_alpha

        if mask_gate is  None:
            raise ValueError("Branched attention requires a mask for the face branch")
        
        q_face = q * mask_gate # face area of noise_hidden

        if fusion_mode == "dual_attention_gate":
            if self.face_fusion_logit is None:
                raise RuntimeError("dual_attention_gate requires configure_face_fusion() before forward")
            key_face_ref = self._k_ref(attn, face_hidden_mixed)
            value_face_ref = self._v_ref(attn, face_hidden_mixed)
            key_face_noise = self._k_ref(attn, noise_face_hidden)
            value_face_noise = self._v_ref(attn, noise_face_hidden)
            key_face_ref = key_face_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_face_ref = value_face_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            key_face_noise = key_face_noise.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_face_noise = value_face_noise.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            hidden_face_ref = F.scaled_dot_product_attention(
                q_face, key_face_ref, value_face_ref, dropout_p=0.0, is_causal=False
            )
            hidden_face_noise = F.scaled_dot_product_attention(
                q_face, key_face_noise, value_face_noise, dropout_p=0.0, is_causal=False
            )
            gate = torch.sigmoid(self.face_fusion_logit.float()).to(q_face.dtype)
            gate = gate.view(1, -1, 1, 1) * float(self.ba_face_fusion_gate_max)
            hidden_face = hidden_face_ref * (1.0 - gate) + hidden_face_noise * gate
        else:
            if CA_MIXING_FOR_FACE:
                face_kv_hidden = torch.cat([face_hidden_mixed, noise_face_hidden], dim=1)
            else:
                face_kv_hidden = face_hidden_mixed
            key_face = self._k_ref(attn, face_kv_hidden)
            value_face = self._v_ref(attn, face_kv_hidden)
            key_face = key_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            value_face = value_face.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
            hidden_face = F.scaled_dot_product_attention(
                q_face, key_face, value_face, dropout_p=0.0, is_causal=False
            )
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
        query_ref = query_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        key_ref = key_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # hidden_ref needs to be without any masks
        hidden_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(ref_batch_size, -1, noise_hidden.shape[-1])
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
            if strict_face_routing:
                res_noise = residual[:batch_size]
                res_ref = residual[batch_size:]
                if input_ndim == 4:
                    res_mask = mask_flat.transpose(1, 2).reshape(batch_size, 1, height, width)
                else:
                    res_mask = mask_flat
                res_noise = res_noise * (1.0 - res_mask.to(dtype=res_noise.dtype, device=res_noise.device))
                residual_to_add = torch.cat([res_noise, res_ref], dim=0)
            else:
                residual_to_add = residual
            hidden_states = hidden_states + residual_to_add
        
        hidden_states = hidden_states / attn.rescale_output_factor # TODO check if neeeded / do separately for each branch
        
        return hidden_states
    
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops — always resize in 2-D (no 1-D raster)."""
        H, W = _infer_spatial_hw(target_len, mask)
        
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
        branched_attn_weight_mode: str = "shared",
        branched_attn_new_weight_kind: str = "full",
        branched_attn_lora_rank: int = 16,
    ):
        super().__init__()
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.branched_attn_weight_mode = (branched_attn_weight_mode or "shared").lower()
        self.branched_attn_new_weight_kind = (branched_attn_new_weight_kind or "full").lower()
        self.branched_attn_lora_rank = int(branched_attn_lora_rank)
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None

        self.has_cross_attention_kwargs = True # Accept cross_attention_kwargs to avoid noisy warnings

    def init_from_attention(self, attn) -> None:
        mode = self.branched_attn_weight_mode
        if mode in {"ref_only", "noise_and_ref"}:
            self.ref_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.ref_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.ref_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
        if mode == "noise_and_ref":
            self.noise_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.noise_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.noise_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )

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
        
        total_batch = hidden_states.shape[0]
        batch_size, ref_batch_size = _branch_batch_sizes(self.mask, total_batch)
        noise_hidden = hidden_states[:batch_size]
        ref_hidden = hidden_states[batch_size:]
        
        if encoder_hidden_states is None:
            raise ValueError ("Branched cross-attention requires encoder_hidden_states")
        
        gen_prompt = encoder_hidden_states[:batch_size]
        face_prompt = encoder_hidden_states[batch_size:]
            
    


        # Ensure encoder prompts match the **latent half-batch** (handles num_images_per_prompt > 1)
        if gen_prompt.shape[0] != batch_size:
            # tile or repeat to match, then trim
            rep = (batch_size + gen_prompt.shape[0] - 1) // gen_prompt.shape[0]
            gen_prompt = gen_prompt.repeat(rep, 1, 1)[:batch_size].contiguous()
        if face_prompt.shape[0] != ref_batch_size:
            rep = (ref_batch_size + face_prompt.shape[0] - 1) // face_prompt.shape[0]
            face_prompt = face_prompt.repeat(rep, 1, 1)[:ref_batch_size].contiguous()

        # Defensive: recompute from tensors actually used below
        batch_size = noise_hidden.shape[0]

        
        # Handle group norm
        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
        
        # ========== PROCESS FIRST HALF (NOISE BATCH) WITH BRANCHING ==========
        
        # Compute query from noise
        query_bg = self._q_noise(attn, noise_hidden)
        
        # Get attention parameters
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim

        q_bg = query_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        # Compute query from ref
        query_ref = self._q_ref(attn, ref_hidden)

        # Get attention parameters
        head_dim = attn.heads
        dim_per_head = noise_hidden.shape[-1] // head_dim

        q_ref = query_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        # === BACKGROUND BRANCH ===
        # Q: background from noise, K/V: generation prompt
        key_bg = self._k_noise(attn, gen_prompt)
        value_bg = self._v_noise(attn, gen_prompt)
        key_bg = key_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_bg = value_bg.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        
        hidden_bg = F.scaled_dot_product_attention(q_bg, key_bg, value_bg, dropout_p=0.0, is_causal=False)
        hidden_bg = hidden_bg.transpose(1, 2).reshape(batch_size, -1, noise_hidden.shape[-1])
        
        # === FACE BRANCH ===
        # Q: reference hidden, K/V: face prompt (should be different from gen_prompt!)
        key_ref = self._k_ref(attn, face_prompt)
        value_ref = self._v_ref(attn, face_prompt)
        key_ref = key_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        hidden_ref = F.scaled_dot_product_attention(q_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False)
        hidden_ref = hidden_ref.transpose(1, 2).reshape(ref_batch_size, -1, noise_hidden.shape[-1])
        
        
        
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
        H, W = _infer_spatial_hw(target_len, mask)
        
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
