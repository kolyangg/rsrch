"""
attn_processor.py - Branched attention processors with consistent batch handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from diffusers.models.attention_processor import AttnProcessor2_0


_STANDARD_ATTN_PROCESSOR = AttnProcessor2_0()


class ZeroInitResidualProjection(nn.Module):
    """Low-rank output adapter whose initial contribution is exactly zero."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        rank = max(1, min(int(rank), int(hidden_size)))
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = self.down.weight.dtype
        return self.up(self.down(hidden_states.to(dtype=dtype)))


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
        base = F.linear(
            x.to(dtype=self.base_weight.dtype),
            self.base_weight,
            self.base_bias,
        )
        delta = F.linear(
            F.linear(x.to(dtype=self.lora_A.dtype), self.lora_A),
            self.lora_B,
        ) * self.scaling
        return base.to(dtype=delta.dtype) + delta


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


def _linear_forward(layer, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run mixed-precision trainable clones without changing frozen UNet dtype."""
    if isinstance(layer, BranchLoRALinear):
        return layer(hidden_states)
    weight = getattr(layer, "weight", None)
    if weight is None and hasattr(layer, "get_base_layer"):
        weight = layer.get_base_layer().weight
    if weight is None:
        return layer(hidden_states)
    return layer(hidden_states.to(dtype=weight.dtype))


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
        ba_sa_mode: str = "legacy",
        ba_face_kv_mode: str = "zero_masked_full",
        ba_face_roi_size: int = 4,
        ba_hard_mask_resize: str = "legacy_threshold",
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
        self.ba_sa_mode = str(ba_sa_mode or "legacy").lower()
        self.ba_face_kv_mode = str(ba_face_kv_mode or "zero_masked_full").lower()
        self.ba_face_roi_size = max(1, int(ba_face_roi_size))
        self.ba_hard_mask_resize = str(ba_hard_mask_resize or "legacy_threshold").lower()
        if self.ba_sa_mode not in {"legacy", "pm_face_residual"}:
            raise ValueError(f"Unknown ba_sa_mode: {self.ba_sa_mode}")
        if self.ba_face_kv_mode not in {"zero_masked_full", "compact_hard_bbox"}:
            raise ValueError(f"Unknown ba_face_kv_mode: {self.ba_face_kv_mode}")
        
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
        self.face_delta_out = (
            ZeroInitResidualProjection(hidden_size, self.branched_attn_lora_rank)
            if self.ba_sa_mode == "pm_face_residual"
            else None
        )
        self.face_residual_gate = (
            nn.Parameter(torch.ones(1)) if self.ba_sa_mode == "pm_face_residual" else None
        )
        
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

    @staticmethod
    def _split_batch_arg(value, batch_size: int, total_batch: int, *, second: bool = False):
        if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != total_batch:
            return value
        return value[batch_size:] if second else value[:batch_size]

    @staticmethod
    def _normalize_sequence(attn, hidden_states: torch.Tensor, temb: Optional[torch.Tensor]):
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if hidden_states.ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        return hidden_states

    def _roi_pool_tokens(self, hidden_states: torch.Tensor, mask_gate: torch.Tensor) -> torch.Tensor:
        """Pool each hard rectangular ROI to a fixed compact token grid."""
        batch_size, seq_len, channels = hidden_states.shape
        height, width = _infer_spatial_hw(seq_len, self.mask_ref)
        feature_map = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
        mask_map = mask_gate.squeeze(1).transpose(1, 2).reshape(batch_size, 1, height, width) > 0
        pooled = []
        for idx in range(batch_size):
            coords = mask_map[idx, 0].nonzero(as_tuple=False)
            if coords.numel() == 0:
                raise RuntimeError(f"Empty reference face bbox at attention grid {height}x{width}")
            y0, x0 = coords.amin(dim=0)
            y1, x1 = coords.amax(dim=0) + 1
            crop = feature_map[idx:idx + 1, :, y0:y1, x0:x1]
            crop = F.adaptive_avg_pool2d(crop, (self.ba_face_roi_size, self.ba_face_roi_size))
            pooled.append(crop.flatten(2).transpose(1, 2))
        return torch.cat(pooled, dim=0)

    def _pm_face_residual_forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.mask is None or self.mask_ref is None:
            raise ValueError("pm_face_residual requires target and reference hard bboxes")
        total_batch = int(hidden_states.shape[0])
        batch_size, _ = _branch_batch_sizes(self.mask, total_batch)
        target_input = hidden_states[:batch_size]
        ref_input = hidden_states[batch_size:]
        target_temb = self._split_batch_arg(temb, batch_size, total_batch)
        ref_temb = self._split_batch_arg(temb, batch_size, total_batch, second=True)
        target_attn_mask = self._split_batch_arg(attention_mask, batch_size, total_batch)
        ref_attn_mask = self._split_batch_arg(attention_mask, batch_size, total_batch, second=True)

        target_pm = _STANDARD_ATTN_PROCESSOR(
            attn, target_input, attention_mask=target_attn_mask, temb=target_temb
        )
        ref_pm = _STANDARD_ATTN_PROCESSOR(
            attn, ref_input, attention_mask=ref_attn_mask, temb=ref_temb
        )

        normalized = self._normalize_sequence(attn, hidden_states, temb)
        target_hidden = normalized[:batch_size]
        ref_hidden = normalized[batch_size:]
        seq_len = int(target_hidden.shape[1])
        target_mask = self._prepare_mask(self.mask, seq_len, batch_size).to(
            device=target_hidden.device, dtype=target_hidden.dtype
        )
        ref_mask = self._prepare_mask(self.mask_ref, seq_len, batch_size).to(
            device=ref_hidden.device, dtype=ref_hidden.dtype
        )
        if self.ba_face_kv_mode != "compact_hard_bbox":
            raise ValueError("pm_face_residual requires ba_face_kv_mode=compact_hard_bbox")
        ref_tokens = self._roi_pool_tokens(ref_hidden, ref_mask)

        num_heads = int(attn.heads)
        query = attn.to_q(target_hidden)
        key = self._k_ref(attn, ref_tokens)
        value = self._v_ref(attn, ref_tokens)
        head_dim = int(key.shape[-1]) // num_heads
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        face_hidden = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        face_hidden = face_hidden.transpose(1, 2).reshape(batch_size, seq_len, -1)
        face_delta = self.face_delta_out(face_hidden) * self.face_residual_gate.to(face_hidden.dtype)
        face_delta = face_delta * target_mask.squeeze(1)
        if target_pm.ndim == 4:
            _, channels, height, width = target_pm.shape
            face_delta = face_delta.transpose(1, 2).reshape(batch_size, channels, height, width)
        target_out = target_pm + face_delta.to(dtype=target_pm.dtype)
        return torch.cat([target_out, ref_pm], dim=0)
        
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

        if self.ba_sa_mode == "pm_face_residual":
            return self._pm_face_residual_forward(attn, hidden_states, attention_mask, temb)


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

        if self.ba_hard_mask_resize == "area_preserving":
            if H <= m4.shape[-2] and W <= m4.shape[-1]:
                m2d = F.adaptive_max_pool2d(m4, output_size=(H, W))
            else:
                m2d = F.interpolate(m4, size=(H, W), mode="nearest")
            m2d = (m2d > 0).to(dtype=m2d.dtype)
        elif self.ba_hard_mask_resize == "legacy_threshold":
            m2d = F.interpolate(m4, size=(H, W), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unknown ba_hard_mask_resize: {self.ba_hard_mask_resize}")
                    
                    
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
        ba_ca_mode: str = "legacy_ref_branch",
        ba_identity_token_count: int = 4,
        ba_identity_memory_mode: str = "mean_plus_basis",
        ba_hard_mask_resize: str = "legacy_threshold",
        ba_face_gate_mode: str = "legacy_scalar",
        ba_face_gate_init: float = 1.0,
        ba_face_gate_max: float = 1.0,
        ba_pm_identity_context_scale: float = 1.0,
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
        self.ba_ca_mode = str(ba_ca_mode or "legacy_ref_branch").lower()
        self.ba_identity_token_count = max(1, int(ba_identity_token_count))
        self.ba_identity_memory_mode = str(ba_identity_memory_mode or "mean_plus_basis").lower()
        self.ba_hard_mask_resize = str(ba_hard_mask_resize or "legacy_threshold").lower()
        self.ba_face_gate_mode = str(ba_face_gate_mode or "legacy_scalar").lower()
        self.ba_face_gate_init = float(ba_face_gate_init)
        self.ba_face_gate_max = float(ba_face_gate_max)
        self.ba_pm_identity_context_scale = float(ba_pm_identity_context_scale)
        if self.ba_ca_mode not in {"legacy_ref_branch", "target_face_residual"}:
            raise ValueError(f"Unknown ba_ca_mode: {self.ba_ca_mode}")
        if self.ba_identity_memory_mode not in {
            "mean_plus_basis",
            "qformer_tokens",
            "face_patch_resampler",
            "canonical_face_parts",
            "qformer_plus_canonical_parts",
        }:
            raise ValueError(f"Unknown ba_identity_memory_mode: {self.ba_identity_memory_mode}")
        if self.ba_face_gate_mode not in {"legacy_scalar", "bounded_sigmoid"}:
            raise ValueError(f"Unknown ba_face_gate_mode: {self.ba_face_gate_mode}")
        if self.ba_face_gate_max <= 0:
            raise ValueError("ba_face_gate_max must be positive")
        if not 0.0 <= self.ba_pm_identity_context_scale <= 1.0:
            raise ValueError("ba_pm_identity_context_scale must be in [0, 1]")
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        self.target_id_to_k = None
        self.target_id_to_v = None
        self.id_token_basis = None
        self.face_delta_out = None
        self.face_residual_gate = None
        self.id_embeds = None
        self.class_tokens_mask = None
        self.pm_text_only_embeds = None

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
        if self.ba_ca_mode == "target_face_residual":
            self.target_id_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            self.target_id_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
            )
            template = next(self.target_id_to_k.parameters(), None)
            device = template.device if template is not None else attn.to_k.weight.device
            dtype = template.dtype if template is not None else attn.to_k.weight.dtype
            if self.ba_identity_memory_mode == "mean_plus_basis":
                self.id_token_basis = nn.Parameter(
                    torch.empty(
                        self.ba_identity_token_count,
                        self.cross_attention_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                nn.init.normal_(self.id_token_basis, mean=0.0, std=0.01)
            self.face_delta_out = ZeroInitResidualProjection(
                self.hidden_size, self.branched_attn_lora_rank
            ).to(device=device, dtype=dtype)
            if self.ba_face_gate_mode == "bounded_sigmoid":
                ratio = min(
                    max(self.ba_face_gate_init / self.ba_face_gate_max, 1e-4),
                    1.0 - 1e-4,
                )
                raw_init = math.log(ratio / (1.0 - ratio))
                self.face_residual_gate = nn.Parameter(
                    torch.full((1,), raw_init, device=device, dtype=dtype)
                )
            else:
                self.face_residual_gate = nn.Parameter(
                    torch.full((1,), self.ba_face_gate_init, device=device, dtype=dtype)
                )

    def effective_face_residual_gate(self) -> torch.Tensor:
        if self.ba_face_gate_mode == "bounded_sigmoid":
            return self.ba_face_gate_max * torch.sigmoid(self.face_residual_gate)
        return self.face_residual_gate

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

    def _target_face_residual_forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.mask is None:
            raise ValueError("target_face_residual requires a target hard bbox")
        if self.id_embeds is None:
            raise ValueError("target_face_residual requires 2048-D reference identity features")

        pm_context = encoder_hidden_states
        if self.ba_pm_identity_context_scale < 1.0:
            if self.pm_text_only_embeds is None:
                raise ValueError(
                    "PhotoMaker identity-context attenuation requires text-only prompt embeddings"
                )
            text_context = self.pm_text_only_embeds.to(
                device=encoder_hidden_states.device,
                dtype=encoder_hidden_states.dtype,
            )
            if text_context.shape[0] != encoder_hidden_states.shape[0]:
                if (
                    text_context.shape[0] <= 0
                    or encoder_hidden_states.shape[0] % text_context.shape[0] != 0
                ):
                    raise RuntimeError(
                        f"Text-only prompt batch {text_context.shape[0]} does not match "
                        f"PhotoMaker prompt batch {encoder_hidden_states.shape[0]}"
                    )
                text_context = text_context.repeat(
                    (encoder_hidden_states.shape[0] // text_context.shape[0], 1, 1)
                )
            if text_context.shape != encoder_hidden_states.shape:
                raise RuntimeError(
                    f"Text-only prompt shape {tuple(text_context.shape)} does not match "
                    f"PhotoMaker prompt shape {tuple(encoder_hidden_states.shape)}"
                )
            pm_context = text_context + self.ba_pm_identity_context_scale * (
                encoder_hidden_states - text_context
            )

        pm_out = _STANDARD_ATTN_PROCESSOR(
            attn,
            hidden_states,
            encoder_hidden_states=pm_context,
            attention_mask=attention_mask,
            temb=temb,
        )
        normalized = BranchedAttnProcessor._normalize_sequence(attn, hidden_states, temb)
        batch_size, seq_len, _ = normalized.shape
        mask_gate = self._prepare_mask(self.mask, seq_len, batch_size).to(
            device=normalized.device, dtype=normalized.dtype
        )

        target_param = next(self.target_id_to_k.parameters(), None)
        identity_dtype = target_param.dtype if target_param is not None else normalized.dtype
        id_embeds = self.id_embeds.to(device=normalized.device, dtype=identity_dtype)
        if id_embeds.ndim not in {2, 3} or id_embeds.shape[-1] != self.cross_attention_dim:
            raise ValueError(f"Unsupported identity memory shape: {tuple(id_embeds.shape)}")
        if id_embeds.shape[0] != batch_size:
            if id_embeds.shape[0] <= 0 or batch_size % id_embeds.shape[0] != 0:
                raise RuntimeError(
                    f"Identity batch {id_embeds.shape[0]} does not match target batch {batch_size}"
                )
            id_embeds = id_embeds.repeat(
                (batch_size // id_embeds.shape[0],) + (1,) * (id_embeds.ndim - 1)
            )
        if self.ba_identity_memory_mode in {
            "qformer_tokens",
            "face_patch_resampler",
            "canonical_face_parts",
            "qformer_plus_canonical_parts",
        }:
            if id_embeds.ndim != 3:
                raise ValueError(f"Token identity memory expects [B,T,D], got {tuple(id_embeds.shape)}")
            if id_embeds.shape[1] != self.ba_identity_token_count:
                raise ValueError(
                    f"Expected {self.ba_identity_token_count} identity tokens, got {id_embeds.shape[1]}"
                )
            has_identity = (id_embeds.float().abs().sum(dim=(1, 2), keepdim=True) > 0).to(id_embeds.dtype)
            id_tokens = id_embeds * has_identity
        else:
            if id_embeds.ndim == 3 and id_embeds.shape[1] == 1:
                id_embeds = id_embeds[:, 0]
            if id_embeds.ndim != 2:
                raise ValueError(f"mean_plus_basis expects [B,D], got {tuple(id_embeds.shape)}")
            has_identity = (id_embeds.float().abs().sum(dim=-1, keepdim=True) > 0).to(id_embeds.dtype)
            id_tokens = id_embeds.unsqueeze(1) + self.id_token_basis.unsqueeze(0) * has_identity.unsqueeze(1)
            id_tokens = id_tokens * has_identity.unsqueeze(1)

        num_heads = int(attn.heads)
        query = attn.to_q(normalized)
        key = _linear_forward(self.target_id_to_k, id_tokens)
        value = _linear_forward(self.target_id_to_v, id_tokens)
        query = query.to(dtype=key.dtype)
        value = value.to(dtype=key.dtype)
        head_dim = int(key.shape[-1]) // num_heads
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        face_hidden = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        face_hidden = face_hidden.transpose(1, 2).reshape(batch_size, seq_len, -1)
        gate = self.effective_face_residual_gate().to(
            device=face_hidden.device, dtype=face_hidden.dtype
        )
        face_delta = self.face_delta_out(face_hidden) * gate
        face_delta = face_delta * has_identity.reshape(batch_size, 1, 1).to(face_delta.dtype)
        face_delta = face_delta * mask_gate.squeeze(1)
        if pm_out.ndim == 4:
            _, channels, height, width = pm_out.shape
            face_delta = face_delta.transpose(1, 2).reshape(batch_size, channels, height, width)
        return pm_out + face_delta.to(dtype=pm_out.dtype)
        
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
        if self.ba_ca_mode == "target_face_residual":
            if encoder_hidden_states is None:
                raise ValueError("target_face_residual requires encoder_hidden_states")
            return self._target_face_residual_forward(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )

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
            m4 = mask[:, :1].float()
        else:
            L0 = mask.view(mask.shape[0], -1).shape[1]
            h0 = int(math.sqrt(L0))
            w0 = h0
            assert h0 * w0 == L0, f"mask length {L0} not square"
            m4 = mask.view(mask.shape[0], -1).float().view(mask.shape[0], 1, h0, w0)

        if self.ba_hard_mask_resize == "area_preserving":
            if H <= m4.shape[-2] and W <= m4.shape[-1]:
                m2d = F.adaptive_max_pool2d(m4, output_size=(H, W))
            else:
                m2d = F.interpolate(m4, size=(H, W), mode="nearest")
            m2d = (m2d > 0).to(dtype=m2d.dtype)
        elif self.ba_hard_mask_resize == "legacy_threshold":
            m2d = F.interpolate(m4, size=(H, W), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unknown ba_hard_mask_resize: {self.ba_hard_mask_resize}")
        
        m = m2d.flatten(2).transpose(1, 2)  # [B, H*W, 1]
        
        # Expand for batch if needed
        if m.shape[0] != batch_size:
            # m = m.expand(batch_size, -1, -1)
            m = m.repeat((batch_size + m.shape[0] - 1) // m.shape[0], 1, 1)[:batch_size]
            
        # Reshape for multi-head attention [B, 1, L, 1]
        return m.view(batch_size, 1, target_len, 1)
