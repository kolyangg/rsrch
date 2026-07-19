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


class BranchedAttnProcessor(nn.Module):
    """
    Self-attention processor with face/background branching.
    Expects doubled batch: [noise_batch, reference_batch]
    """

    _is_branched_processor = True
    _branched_kind = "self"
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        branched_attn_weight_mode: str = "shared",
        branched_attn_new_weight_kind: str = "full",
        branched_attn_lora_rank: int = 16,
        processor_name: str = "",
        ba_sa_ref_token_mode: str = "full_grid",
        ba_sa_face_mode: str = "reference",
        ba_sa_ref_layer_scope: str = "all",
        ba_sa_roi_grid_size: int = 8,
        ba_sa_core_ratio: float = 0.7,
        ba_sa_mix_init: float = 0.25,
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
        self.processor_name = str(processor_name or "")
        self.ba_sa_ref_token_mode = str(ba_sa_ref_token_mode or "full_grid").lower()
        self.ba_sa_face_mode = str(ba_sa_face_mode or "reference").lower()
        self.ba_sa_ref_layer_scope = str(ba_sa_ref_layer_scope or "all").lower()
        self.ba_sa_roi_grid_size = int(ba_sa_roi_grid_size)
        self.ba_sa_core_ratio = float(ba_sa_core_ratio)
        self.ba_sa_mix_init = float(ba_sa_mix_init)
        if self.ba_sa_ref_token_mode not in {"full_grid", "roi"}:
            raise ValueError(f"Unknown ba_sa_ref_token_mode: {self.ba_sa_ref_token_mode}")
        if self.ba_sa_face_mode not in {
            "reference",
            "dual",
            "core_ring",
            "confidence_residual",
        }:
            raise ValueError(f"Unknown ba_sa_face_mode: {self.ba_sa_face_mode}")
        if self.ba_sa_ref_layer_scope not in {"all", "up"}:
            raise ValueError(f"Unknown ba_sa_ref_layer_scope: {self.ba_sa_ref_layer_scope}")
        if self.ba_sa_roi_grid_size <= 0:
            raise ValueError("ba_sa_roi_grid_size must be positive")
        if not 0.0 < self.ba_sa_core_ratio <= 1.0:
            raise ValueError("ba_sa_core_ratio must be in (0, 1]")
        if not 0.0 < self.ba_sa_mix_init < 1.0:
            raise ValueError("ba_sa_mix_init must be in (0, 1)")
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        self.face_mix_logits = None
        self.face_residual_gain = None
        
        # If True: keep masks strictly binary after resize (avoids soft boundary blending)
        self.force_binary_masks: bool = True # False
        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

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
        base_q = attn.to_q.get_base_layer() if hasattr(attn.to_q, "get_base_layer") else attn.to_q
        if self.ba_sa_face_mode == "dual":
            initial_logit = math.log(self.ba_sa_mix_init / (1.0 - self.ba_sa_mix_init))
            self.face_mix_logits = nn.Parameter(
                torch.full(
                    (int(attn.heads),),
                    initial_logit,
                    device=base_q.weight.device,
                    dtype=base_q.weight.dtype,
                )
            )
        elif self.ba_sa_face_mode == "confidence_residual":
            # tanh(0) gives exact target-attention parity while retaining a
            # non-zero gradient for the residual gain itself.
            self.face_residual_gain = nn.Parameter(
                torch.zeros(
                    int(attn.heads),
                    device=base_q.weight.device,
                    dtype=base_q.weight.dtype,
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

    @staticmethod
    def _to_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, channels = tensor.shape
        return tensor.view(batch, length, heads, channels // heads).transpose(1, 2)

    @staticmethod
    def _from_heads(tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, length, channels = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, length, heads * channels)

    def _reference_enabled_here(self) -> bool:
        return (
            self.ba_sa_ref_layer_scope == "all"
            or self.processor_name.startswith("up_blocks.")
        )

    def _normalized_roi_tokens(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Crop each hard ROI and normalize it to a dense fixed token grid."""
        batch, length, channels = hidden_states.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise ValueError(f"ROI packing requires square spatial tokens, got {length}")
        spatial = hidden_states.transpose(1, 2).reshape(batch, channels, side, side)
        mask_2d = mask[:, 0, :, 0].reshape(batch, side, side)
        grid = self.ba_sa_roi_grid_size
        packed = []
        for sample_idx in range(batch):
            valid = mask_2d[sample_idx] > 0.01
            coords = valid.nonzero(as_tuple=False)
            if coords.numel() == 0:
                raise ValueError(
                    f"Reference ROI vanished at {side}x{side} in {self.processor_name}"
                )
            y0, x0 = coords.amin(dim=0)
            y1, x1 = coords.amax(dim=0) + 1
            crop = spatial[
                sample_idx : sample_idx + 1,
                :,
                int(y0.item()) : int(y1.item()),
                int(x0.item()) : int(x1.item()),
            ]
            normalized = F.interpolate(
                crop.float(),
                size=(grid, grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=hidden_states.dtype)
            packed.append(normalized.flatten(2).transpose(1, 2))
        return torch.cat(packed, dim=0)

    def _inner_core_mask(self, mask_gate: torch.Tensor) -> torch.Tensor:
        """Return an elliptical identity core inside each target-face mask."""
        batch, _, length, _ = mask_gate.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise ValueError(f"Core mask requires square spatial tokens, got {length}")
        outer = mask_gate[:, 0, :, 0].reshape(batch, side, side) > 0.5
        core = torch.zeros_like(outer)
        for sample_idx in range(batch):
            coords = outer[sample_idx].nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue
            low = coords.amin(dim=0).float()
            high = coords.amax(dim=0).float()
            center = (low + high) * 0.5
            radius = ((high - low + 1.0) * 0.5 * self.ba_sa_core_ratio).clamp_min(0.5)
            yy, xx = torch.meshgrid(
                torch.arange(side, device=outer.device, dtype=torch.float32),
                torch.arange(side, device=outer.device, dtype=torch.float32),
                indexing="ij",
            )
            ellipse = (
                ((yy - center[0]) / radius[0]).square()
                + ((xx - center[1]) / radius[1]).square()
            ) <= 1.0
            sample_core = ellipse & outer[sample_idx]
            if not bool(sample_core.any()):
                nearest = coords[
                    ((coords.float() - center).square().sum(dim=1)).argmin()
                ]
                sample_core[nearest[0], nearest[1]] = True
            core[sample_idx] = sample_core
        return core.reshape(batch, 1, length, 1).to(mask_gate.dtype)

    @staticmethod
    def _attention_with_confidence(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention output plus normalized inverse-entropy confidence."""
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * (query.shape[-1] ** -0.5)
        probs = scores.softmax(dim=-1)
        output = torch.matmul(probs.to(value.dtype), value)
        token_count = int(key.shape[-2])
        if token_count <= 1:
            confidence = torch.ones_like(probs[..., :1])
        else:
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            confidence = (1.0 - entropy / math.log(token_count)).clamp(0.0, 1.0)
        return output, confidence.to(output.dtype)

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
        
        # Runtime values are passed via UNet cross_attention_kwargs
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

        ref_mask = self._prepare_mask(self.mask_ref, seq_len, ref_batch_size)
        ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
        ref_mask_flat = ref_mask.squeeze(1)  # [B, L, 1]

        q_face = q * mask_gate  # face area of noise_hidden
        reference_enabled = self._reference_enabled_here()
        legacy_reference_path = (
            reference_enabled
            and self.ba_sa_face_mode == "reference"
            and self.ba_sa_ref_token_mode == "full_grid"
        )

        if legacy_reference_path:
            # Exact N3a/NN1 face route. Keep this explicit so all new toggles
            # can be disabled without changing the previous architecture.
            noise_face_hidden = noise_hidden * mask_flat
            ref_face_hidden = ref_hidden * ref_mask_flat
            face_hidden_mixed = (
                (1 - POSE_ADAPT_RATIO) * ref_face_hidden
                + POSE_ADAPT_RATIO * noise_face_hidden
            )
            key_face = self._to_heads(self._k_ref(attn, face_hidden_mixed), head_dim)
            value_face = self._to_heads(self._v_ref(attn, face_hidden_mixed), head_dim)
            hidden_face_heads = F.scaled_dot_product_attention(
                q_face, key_face, value_face, dropout_p=0.0, is_causal=False
            )
        else:
            target_key = self._to_heads(self._k_noise(attn, noise_hidden), head_dim)
            target_value = self._to_heads(self._v_noise(attn, noise_hidden), head_dim)
            target_face_heads = F.scaled_dot_product_attention(
                q_face, target_key, target_value, dropout_p=0.0, is_causal=False
            )

            if not reference_enabled:
                hidden_face_heads = target_face_heads
            else:
                if self.ba_sa_ref_token_mode == "roi":
                    soft_ref_mask = self._prepare_mask(
                        self.mask_ref,
                        seq_len,
                        ref_batch_size,
                        force_binary=False,
                    ).to(dtype=ref_hidden.dtype, device=ref_hidden.device)
                    reference_source = self._normalized_roi_tokens(ref_hidden, soft_ref_mask)
                else:
                    reference_source = ref_hidden * ref_mask_flat
                reference_key = self._to_heads(self._k_ref(attn, reference_source), head_dim)
                reference_value = self._to_heads(self._v_ref(attn, reference_source), head_dim)

                if self.ba_sa_face_mode == "confidence_residual":
                    reference_face_heads, confidence = self._attention_with_confidence(
                        q_face,
                        reference_key,
                        reference_value,
                    )
                    if self.face_residual_gain is None:
                        raise RuntimeError("confidence_residual mode is missing face_residual_gain")
                    gain = self.face_residual_gain.tanh().view(1, head_dim, 1, 1)
                    hidden_face_heads = target_face_heads + (
                        gain * confidence * (reference_face_heads - target_face_heads)
                    )
                else:
                    reference_face_heads = F.scaled_dot_product_attention(
                        q_face,
                        reference_key,
                        reference_value,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    if self.ba_sa_face_mode == "dual":
                        if self.face_mix_logits is None:
                            raise RuntimeError("dual mode is missing face_mix_logits")
                        ref_weight = self.face_mix_logits.sigmoid().view(1, head_dim, 1, 1)
                        hidden_face_heads = (
                            target_face_heads * (1.0 - ref_weight)
                            + reference_face_heads * ref_weight
                        )
                    elif self.ba_sa_face_mode == "core_ring":
                        core_gate = self._inner_core_mask(mask_gate)
                        hidden_face_heads = (
                            target_face_heads * (1.0 - core_gate)
                            + reference_face_heads * core_gate
                        )
                    elif self.ba_sa_face_mode == "reference":
                        hidden_face_heads = reference_face_heads
                    else:
                        raise RuntimeError(f"Unhandled face mode: {self.ba_sa_face_mode}")

        hidden_face = self._from_heads(hidden_face_heads)



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
    
    
    def _prepare_mask(
        self,
        mask: torch.Tensor,
        target_len: int,
        batch_size: int,
        force_binary: Optional[bool] = None,
    ) -> torch.Tensor:
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
                    
                    
        if force_binary is None:
            force_binary = bool(getattr(self, "force_binary_masks", False))
        if force_binary:
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

    _is_branched_processor = True
    _branched_kind = "cross"
    
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
        self.face_prompt_attention_mask = None

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
        # Q: face from noise, K/V: face prompt (should be different from gen_prompt!)
        key_ref = self._k_ref(attn, face_prompt)
        value_ref = self._v_ref(attn, face_prompt)
        key_ref = key_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
        value_ref = value_ref.view(ref_batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

        face_attention_mask = None
        allowed_tokens = self.face_prompt_attention_mask
        if allowed_tokens is not None:
            allowed_tokens = allowed_tokens.to(device=q_ref.device, dtype=torch.bool)
            if allowed_tokens.ndim == 1:
                allowed_tokens = allowed_tokens.unsqueeze(0)
            if allowed_tokens.shape[-1] != key_ref.shape[-2]:
                raise RuntimeError(
                    "Face-prompt attention-mask sequence mismatch: "
                    f"mask={tuple(allowed_tokens.shape)}, key={tuple(key_ref.shape)}"
                )
            if allowed_tokens.shape[0] != ref_batch_size:
                if ref_batch_size % allowed_tokens.shape[0] != 0:
                    raise RuntimeError(
                        "Face-prompt attention-mask batch mismatch: "
                        f"mask={tuple(allowed_tokens.shape)}, reference_batch={ref_batch_size}"
                    )
                allowed_tokens = allowed_tokens.repeat(
                    ref_batch_size // allowed_tokens.shape[0],
                    1,
                )
            if not bool(allowed_tokens.any(dim=1).all()):
                raise RuntimeError("Face-prompt attention mask contains an empty row")
            face_attention_mask = torch.zeros(
                ref_batch_size,
                1,
                1,
                key_ref.shape[-2],
                device=q_ref.device,
                dtype=q_ref.dtype,
            )
            face_attention_mask.masked_fill_(
                ~allowed_tokens[:, None, None, :],
                float("-inf"),
            )

        hidden_ref = F.scaled_dot_product_attention(
            q_ref,
            key_ref,
            value_ref,
            attn_mask=face_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
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
