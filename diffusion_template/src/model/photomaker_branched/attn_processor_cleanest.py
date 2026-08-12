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
        trainable_dtype=None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.scaling = float(alpha if alpha is not None else rank) / float(rank)
        self.register_buffer("base_weight", torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.register_buffer("base_bias", torch.empty(out_features, device=device, dtype=dtype) if bias else None)
        parameter_dtype = trainable_dtype if trainable_dtype is not None else dtype
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=parameter_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=parameter_dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight, self.base_bias)
        parameter_dtype = self.lora_A.dtype
        delta = F.linear(
            F.linear(x.to(dtype=parameter_dtype), self.lora_A),
            self.lora_B,
        )
        return base + (delta * self.scaling).to(dtype=base.dtype)


def _clone_effective_linear(
    attn_linear,
    *,
    kind: str,
    rank: int,
    alpha: Optional[int] = None,
    adapter_name: str = "default",
    trainable_dtype=None,
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
            trainable_dtype=trainable_dtype,
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
        trainable_dtype=None,
        true_reference_key_mask: bool = False,
        branch_output_rank: Optional[int] = None,
        reference_roi_warp: bool = False,
        hardcase_mode: str = "off",
        hardcase_rank: int = 64,
        hardcase_gate_max: float = 0.20,
        hardcase_roi_size: int = 32,
        hardcase_face_threshold_px: int = 256,
        hardcase_transition_cells: int = 2,
        hardcase_ownership_hidden_dim: int = 128,
        hardcase_visible_face_floor: float = 0.20,
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
        self.trainable_dtype = trainable_dtype
        self.true_reference_key_mask = bool(true_reference_key_mask)
        self.branch_output_rank = (
            None if branch_output_rank is None else int(branch_output_rank)
        )
        if self.branch_output_rank is not None and self.branch_output_rank <= 0:
            raise ValueError("branch_output_rank must be positive when enabled")
        self.reference_roi_warp = bool(reference_roi_warp)
        self.hardcase_mode = str(hardcase_mode or "off").lower()
        if self.hardcase_mode not in {
            "off",
            "highres_roi",
            "clean_memory",
            "semantic_ownership",
            "soft_router",
        }:
            raise ValueError(f"Unknown hardcase_mode={hardcase_mode!r}")
        self.hardcase_rank = int(hardcase_rank)
        self.hardcase_gate_max = float(hardcase_gate_max)
        self.hardcase_roi_size = int(hardcase_roi_size)
        self.hardcase_face_threshold_px = int(hardcase_face_threshold_px)
        self.hardcase_transition_cells = int(hardcase_transition_cells)
        self.hardcase_visible_face_floor = float(hardcase_visible_face_floor)
        if self.hardcase_rank <= 0 or self.hardcase_roi_size <= 1:
            raise ValueError("Hard-case rank and ROI size must be positive")
        if not 0.0 < self.hardcase_gate_max <= 1.0:
            raise ValueError("hardcase_gate_max must be in (0, 1]")
        if self.hardcase_transition_cells < 1:
            raise ValueError("hardcase_transition_cells must be positive")
        if not 0.0 <= self.hardcase_visible_face_floor <= 1.0:
            raise ValueError("hardcase_visible_face_floor must be in [0, 1]")

        self.roi_gate_raw = None
        self.memory_to_k = None
        self.memory_to_v = None
        self.memory_to_out = None
        self.memory_gate_raw = None
        self.ownership_norm = None
        self.ownership_mlp = None
        self.ownership_scale_raw = None
        if self.hardcase_mode == "highres_roi":
            self.roi_gate_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        elif self.hardcase_mode == "clean_memory":
            self.memory_gate_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        elif self.hardcase_mode == "semantic_ownership":
            ownership_hidden = int(hardcase_ownership_hidden_dim)
            if ownership_hidden <= 0:
                raise ValueError("hardcase_ownership_hidden_dim must be positive")
            self.ownership_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.ownership_mlp = nn.Sequential(
                nn.Linear(hidden_size + 2, ownership_hidden),
                nn.SiLU(),
                nn.Linear(ownership_hidden, 1),
            )
            nn.init.zeros_(self.ownership_mlp[-1].weight)
            nn.init.zeros_(self.ownership_mlp[-1].bias)
            self.ownership_scale_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        self.clean_reference_memory = None
        self.capture_clean_memory = False
        self.ownership_target_mask = None
        self._ownership_aux_loss = None
        self.ba_denoise_progress = None
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        self.face_to_out = None
        
        # If True: keep masks strictly binary after resize (avoids soft boundary blending)
        self.force_binary_masks: bool = True # False
        # Opt-in per-forward memoization. The cache lives on the injected mask
        # tensor, so it cannot leak across samples or training steps.
        self.cache_prepared_masks: bool = False
        # Historical behavior uses reference-only face K/V. The runtime patcher
        # may opt in to target-native face features without changing this default.
        self.pose_adapt_ratio: float = 0.0
        # Let diffusers know we accept cross_attention_kwargs to silence warnings
        self.has_cross_attention_kwargs = True

    def init_from_attention(self, attn) -> None:
        mode = self.branched_attn_weight_mode
        if mode in {"ref_only", "noise_and_ref"}:
            self.ref_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.ref_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.ref_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
        if mode == "noise_and_ref":
            self.noise_to_q = _clone_effective_linear(
                attn.to_q,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.noise_to_k = _clone_effective_linear(
                attn.to_k,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.noise_to_v = _clone_effective_linear(
                attn.to_v,
                kind=self.branched_attn_new_weight_kind,
                rank=self.branched_attn_lora_rank,
                trainable_dtype=self.trainable_dtype,
            )
        if self.branch_output_rank is not None:
            self.face_to_out = _clone_effective_linear(
                attn.to_out[0],
                kind="lora",
                rank=self.branch_output_rank,
                trainable_dtype=self.trainable_dtype,
            )
        if self.hardcase_mode == "clean_memory":
            # 11 Aug 2026 - AICODE-NOTE: the clean-memory lane owns separate
            # low-rank K/V/output deltas, while its zero gate preserves CL14 at
            # initialization. Target Q always remains in target coordinates.
            self.memory_to_k = _clone_effective_linear(
                attn.to_k,
                kind="lora",
                rank=self.hardcase_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.memory_to_v = _clone_effective_linear(
                attn.to_v,
                kind="lora",
                rank=self.hardcase_rank,
                trainable_dtype=self.trainable_dtype,
            )
            self.memory_to_out = _clone_effective_linear(
                attn.to_out[0],
                kind="lora",
                rank=self.hardcase_rank,
                trainable_dtype=self.trainable_dtype,
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

    def set_clean_memory_capture(self, enabled: bool) -> None:
        self.capture_clean_memory = bool(enabled)

    def clear_clean_memory(self) -> None:
        self.clean_reference_memory = None

    def set_ownership_target_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.ownership_target_mask = mask

    def set_denoise_progress(self, progress: Optional[torch.Tensor]) -> None:
        self.ba_denoise_progress = progress

    def ownership_aux_loss(self) -> Optional[torch.Tensor]:
        return self._ownership_aux_loss

    @staticmethod
    def _reshape_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, channels = tensor.shape
        if channels % heads:
            raise RuntimeError(f"Attention width {channels} is not divisible by {heads}")
        return tensor.view(batch, length, heads, channels // heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, length, width = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, length, heads * width)

    def _normalized_halves(self, attn, hidden_states, temb):
        normalized = hidden_states
        if attn.spatial_norm is not None:
            normalized = attn.spatial_norm(normalized, temb)
        input_ndim = normalized.ndim
        spatial = None
        if input_ndim == 4:
            total_batch, channels, height, width = normalized.shape
            spatial = (channels, height, width)
            normalized = normalized.view(total_batch, channels, height * width).transpose(1, 2)
        elif input_ndim != 3:
            raise RuntimeError(f"Unsupported attention input rank: {input_ndim}")
        if normalized.shape[0] % 2:
            raise RuntimeError("Hard-case BA requires [target, reference] doubled batches")
        batch = normalized.shape[0] // 2
        target = normalized[:batch]
        reference = normalized[batch:]
        if attn.group_norm is not None:
            target = attn.group_norm(target.transpose(1, 2)).transpose(1, 2)
            reference = attn.group_norm(reference.transpose(1, 2)).transpose(1, 2)
        return target, reference, input_ndim, spatial

    def _binary_mask(self, mask: torch.Tensor, length: int, batch: int, dtype) -> torch.Tensor:
        previous = self.force_binary_masks
        self.force_binary_masks = True
        try:
            prepared = self._prepare_mask(mask, length, batch).squeeze(1)
        finally:
            self.force_binary_masks = previous
        return prepared.to(dtype=dtype)

    def _soft_router_mask(self, mask: torch.Tensor, length: int, batch: int, dtype) -> torch.Tensor:
        binary = self._binary_mask(mask, length, batch, torch.float32)
        side = int(math.isqrt(length))
        image = binary.transpose(1, 2).reshape(batch, 1, side, side)
        remaining = image
        result = torch.ones_like(image)
        cells = self.hardcase_transition_cells
        for index in range(cells):
            eroded = 1.0 - F.max_pool2d(1.0 - remaining, 3, stride=1, padding=1)
            ring = (remaining - eroded).clamp(0.0, 1.0)
            phase = float(index + 1) / float(cells + 1)
            weight = 0.5 - 0.5 * math.cos(math.pi * phase)
            result = result * (1.0 - ring) + ring * weight
            remaining = eroded
        result = result * image
        return result.flatten(2).transpose(1, 2).to(dtype=dtype)

    @staticmethod
    def _masked_rms(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = (mask.float().sum(dim=(1, 2)) * tensor.shape[-1]).clamp_min(1.0)
        energy = (tensor.float().square() * mask.float()).sum(dim=(1, 2)) / denom
        return energy.clamp_min(1.0e-12).sqrt().view(-1, 1, 1)

    def _full_target_lanes(self, attn, target, reference):
        batch, length, _ = target.shape
        heads = int(attn.heads)
        q = self._reshape_heads(self._q_noise(attn, target), heads)
        native = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_noise(attn, target), heads),
            self._reshape_heads(self._v_noise(attn, target), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        ref_mask = self._binary_mask(self.mask_ref, length, batch, reference.dtype)
        reference_face = reference * ref_mask
        reference_message = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_ref(attn, reference_face), heads),
            self._reshape_heads(self._v_ref(attn, reference_face), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        native_message = self._merge_heads(native)
        reference_message = self._merge_heads(reference_message)
        native_out = attn.to_out[0](native_message)
        reference_out = (
            self.face_to_out(reference_message)
            if self.face_to_out is not None
            else attn.to_out[0](reference_message)
        )
        return native_out, reference_out, q

    def _finish_full_router(
        self, attn, residual, target_out, reference, input_ndim, spatial
    ) -> torch.Tensor:
        heads = int(attn.heads)
        reference_message = F.scaled_dot_product_attention(
            self._reshape_heads(self._q_ref(attn, reference), heads),
            self._reshape_heads(self._k_ref(attn, reference), heads),
            self._reshape_heads(self._v_ref(attn, reference), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        reference_out = attn.to_out[0](self._merge_heads(reference_message))
        joined = torch.cat([target_out, reference_out], dim=0)
        joined = attn.to_out[1](joined)
        if input_ndim == 4:
            channels, height, width = spatial
            joined = joined.transpose(-1, -2).reshape(
                joined.shape[0], channels, height, width
            )
        if attn.residual_connection:
            joined = joined + residual
        return joined / attn.rescale_output_factor

    def _ownership_probability(
        self,
        target: torch.Tensor,
        native_out: torch.Tensor,
        reference_out: torch.Tensor,
    ) -> torch.Tensor:
        disagreement = (reference_out.float() - native_out.float()).square().mean(
            dim=-1, keepdim=True
        ).clamp_min(0.0).sqrt().to(dtype=target.dtype)
        progress = getattr(self, "ba_denoise_progress", None)
        if progress is None:
            progress_feature = target.new_zeros(target.shape[0], 1, 1)
        else:
            progress_feature = torch.as_tensor(
                progress, device=target.device, dtype=target.dtype
            ).reshape(-1, 1, 1)
            if progress_feature.shape[0] == 1:
                progress_feature = progress_feature.expand(target.shape[0], -1, -1)
        progress_feature = progress_feature.expand(-1, target.shape[1], -1)
        features = torch.cat(
            [self.ownership_norm(target), disagreement, progress_feature], dim=-1
        )
        logits = self.ownership_mlp(features)
        semantic_probability = torch.sigmoid(logits)
        # Starts at exactly zero, but has a live derivative at the boundary.
        scale = self.hardcase_gate_max * 2.0 * torch.clamp(
            torch.sigmoid(self.ownership_scale_raw) - 0.5,
            min=0.0,
            max=0.5,
        )
        routed_probability = semantic_probability * scale
        supervision = self.ownership_target_mask
        self._ownership_aux_loss = None
        if supervision is not None:
            target_mask = self._binary_mask(
                supervision, target.shape[1], target.shape[0], torch.float32
            )
            face = self._binary_mask(
                self.mask, target.shape[1], target.shape[0], torch.float32
            )
            denom = face.sum().clamp_min(1.0)
            self._ownership_aux_loss = (
                F.binary_cross_entropy(
                    semantic_probability.float(), target_mask, reduction="none"
                )
                * face
            ).sum() / denom
        return routed_probability

    @staticmethod
    def _roi_bounds(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = mask.squeeze(-1) > 0.5
        side = int(math.isqrt(image.shape[1]))
        image = image.reshape(image.shape[0], side, side)
        rows, cols = image.any(dim=2), image.any(dim=1)
        if not bool(rows.any(dim=1).all() and cols.any(dim=1).all()):
            raise RuntimeError("High-resolution ROI received an empty mask")
        y0 = rows.float().argmax(dim=1)
        x0 = cols.float().argmax(dim=1)
        y1 = side - rows.flip(1).float().argmax(dim=1)
        x1 = side - cols.flip(1).float().argmax(dim=1)
        return x0, y0, x1, y1

    def _sample_roi(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, channels = hidden.shape
        side = int(math.isqrt(length))
        x0, y0, x1, y1 = self._roi_bounds(mask)
        samples = []
        source = hidden.transpose(1, 2).reshape(batch, channels, side, side)
        for index in range(batch):
            crop = source[index:index + 1, :, y0[index]:y1[index], x0[index]:x1[index]]
            samples.append(F.interpolate(
                crop.float(),
                size=(self.hardcase_roi_size, self.hardcase_roi_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=hidden.dtype))
        return torch.cat(samples).flatten(2).transpose(1, 2)

    def _scatter_roi(self, roi: torch.Tensor, mask: torch.Tensor, length: int) -> torch.Tensor:
        batch, _, channels = roi.shape
        side = int(math.isqrt(length))
        x0, y0, x1, y1 = self._roi_bounds(mask)
        source = roi.transpose(1, 2).reshape(
            batch, channels, self.hardcase_roi_size, self.hardcase_roi_size
        )
        canvases = []
        for index in range(batch):
            canvas = roi.new_zeros(1, channels, side, side)
            resized = F.interpolate(
                source[index:index + 1].float(),
                size=(int(y1[index] - y0[index]), int(x1[index] - x0[index])),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=roi.dtype)
            canvas[:, :, y0[index]:y1[index], x0[index]:x1[index]] = resized
            canvases.append(canvas)
        return torch.cat(canvases).flatten(2).transpose(1, 2)

    def _highres_roi_residual(self, attn, target, reference) -> torch.Tensor:
        batch, length, _ = target.shape
        target_mask = self._binary_mask(self.mask, length, batch, target.dtype)
        reference_mask = self._binary_mask(self.mask_ref, length, batch, reference.dtype)
        source_px = 1024.0 * target_mask.sum(dim=1).sqrt().squeeze(-1) / math.sqrt(length)
        active = (source_px <= float(self.hardcase_face_threshold_px)).to(target.dtype)
        target_roi = self._sample_roi(target, target_mask)
        reference_roi = self._sample_roi(reference, reference_mask)
        heads = int(attn.heads)
        roi_message = F.scaled_dot_product_attention(
            self._reshape_heads(self._q_noise(attn, target_roi), heads),
            self._reshape_heads(self._k_ref(attn, reference_roi), heads),
            self._reshape_heads(self._v_ref(attn, reference_roi), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        roi_out = attn.to_out[0](self._merge_heads(roi_message))
        scattered = self._scatter_roi(roi_out, target_mask, length) * target_mask
        gate = self.hardcase_gate_max * torch.tanh(self.roi_gate_raw)
        return scattered * gate * active.view(batch, 1, 1)

    def _call_hardcase(self, attn, hidden_states, temb) -> torch.Tensor:
        residual = hidden_states
        target, reference, input_ndim, spatial = self._normalized_halves(
            attn, hidden_states, temb
        )
        if self.capture_clean_memory:
            self.clean_reference_memory = reference.detach()
            return self._call_legacy(attn, hidden_states, temb=temb)

        mode = self.hardcase_mode
        if mode in {"highres_roi", "clean_memory"}:
            baseline = self._call_legacy(attn, hidden_states, temb=temb)
            batch = target.shape[0]
            if mode == "highres_roi":
                addition = self._highres_roi_residual(attn, target, reference)
            else:
                memory = self.clean_reference_memory
                if memory is None:
                    raise RuntimeError("Clean reference memory was not captured")
                if memory.shape != reference.shape:
                    raise RuntimeError(
                        f"Clean-memory shape mismatch: {tuple(memory.shape)} vs {tuple(reference.shape)}"
                    )
                heads = int(attn.heads)
                q = self._reshape_heads(self._q_noise(attn, target), heads)
                ref_mask = self._binary_mask(
                    self.mask_ref, memory.shape[1], batch, memory.dtype
                )
                message = F.scaled_dot_product_attention(
                    q,
                    self._reshape_heads(self.memory_to_k(memory * ref_mask), heads),
                    self._reshape_heads(self.memory_to_v(memory * ref_mask), heads),
                    dropout_p=0.0,
                    is_causal=False,
                )
                memory_out = self.memory_to_out(self._merge_heads(message))
                face = self._binary_mask(self.mask, target.shape[1], batch, memory_out.dtype)
                target_base = baseline[:batch]
                if input_ndim == 4:
                    channels, height, width = spatial
                    target_base = target_base.view(batch, channels, height * width).transpose(1, 2)
                ratio = self._masked_rms(target_base, face) / self._masked_rms(memory_out, face)
                gate = self.hardcase_gate_max * torch.tanh(self.memory_gate_raw)
                addition = memory_out * ratio.to(memory_out.dtype) * face * gate
            if input_ndim == 4:
                channels, height, width = spatial
                addition = addition.transpose(-1, -2).reshape(
                    batch, channels, height, width
                )
            target_out = baseline[:batch] + addition / attn.rescale_output_factor
            return torch.cat([target_out, baseline[batch:]], dim=0)

        native_out, reference_out, _ = self._full_target_lanes(attn, target, reference)
        if mode == "semantic_ownership":
            p_occluder = self._ownership_probability(
                target, native_out, reference_out
            ).to(dtype=native_out.dtype)
            face = self._binary_mask(
                self.mask, target.shape[1], target.shape[0], native_out.dtype
            )
            native_weight = face * p_occluder * (
                1.0 - self.hardcase_visible_face_floor
            )
            native_full = self._finish_full_router(
                attn, residual, native_out, reference, input_ndim, spatial
            )[: target.shape[0]]
            baseline = self._call_legacy(attn, hidden_states, temb=temb)
            baseline_target = baseline[: target.shape[0]]
            if input_ndim == 4:
                _, height, width = spatial
                native_weight = native_weight.transpose(-1, -2).reshape(
                    target.shape[0], 1, height, width
                )
            target_out = baseline_target * (1.0 - native_weight)
            target_out = target_out + native_full * native_weight
            return torch.cat([target_out, baseline[target.shape[0]:]], dim=0)

        if mode == "soft_router":
            router = self._soft_router_mask(
                self.mask, target.shape[1], target.shape[0], native_out.dtype
            )
        else:
            raise RuntimeError(f"Unhandled hard-case mode {mode!r}")
        target_out = native_out * (1.0 - router) + reference_out * router
        return self._finish_full_router(
            attn, residual, target_out, reference, input_ndim, spatial
        )
        
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


        if self.hardcase_mode != "off" or self.capture_clean_memory:
            # 11 Aug 2026 - All CL15+ routes are explicit opt-ins. The legacy
            # function below remains untouched and is the sole path when off.
            return self._call_hardcase(attn, hidden_states, temb)
        return self._call_legacy(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            scale=scale,
            cross_attention_kwargs=cross_attention_kwargs,
        )

    def _call_legacy(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        cross_attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
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
        
        # 26 Jul 2026 - AICODE-NOTE: This value is refreshed from the pipeline
        # before every branched forward. A ratio of 0.0 preserves the historical
        # reference-only K/V path; higher values add target-native face geometry.
        POSE_ADAPT_RATIO = float(getattr(self, "pose_adapt_ratio", 0.0))
        if not 0.0 <= POSE_ADAPT_RATIO <= 1.0:
            raise ValueError(
                f"pose_adapt_ratio must be in [0, 1], got {POSE_ADAPT_RATIO}"
            )
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
        face_key_mask_flat = ref_mask_flat

        if self.reference_roi_warp:
            if POSE_ADAPT_RATIO != 0.0:
                raise RuntimeError(
                    "reference_roi_warp requires pose_adapt_ratio=0"
                )
            # 3 Aug 2026 - Map reference-face features into the target bbox
            # coordinate frame without introducing target K/V or a native-face
            # output mixer. This isolates spatial alignment as one BA element.
            ref_face_hidden = self._warp_reference_roi_to_target(
                ref_face_hidden,
                reference_mask=ref_mask_flat,
                target_mask=mask_flat,
            )
            face_key_mask_flat = mask_flat.to(
                dtype=ref_mask_flat.dtype,
                device=ref_mask_flat.device,
            )

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
            
        reference_attention_mask = None
        if self.true_reference_key_mask:
            valid_reference_keys = face_key_mask_flat.squeeze(-1) > 0.5
            if not bool(valid_reference_keys.any(dim=1).all()):
                raise RuntimeError(
                    "true reference-key masking requires at least one valid key per sample"
                )
            # 3 Aug 2026 - Zeroing reference features is not a key mask: those
            # zero keys still consume softmax probability. True means allowed
            # for PyTorch SDPA and broadcasts across heads and target queries.
            reference_attention_mask = valid_reference_keys[:, None, None, :]
        hidden_face = F.scaled_dot_product_attention(
            q_face,
            key_face,
            value_face,
            attn_mask=reference_attention_mask,
            dropout_p=0.0,
            is_causal=False,
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
        
        if self.face_to_out is None:
            merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
            hidden_states = torch.cat([merged, hidden_ref], dim=0)
            hidden_states = attn.to_out[0](hidden_states)
        else:
            # 3 Aug 2026 - The optional output LoRA is reference-branch-local.
            # Its frozen base is cloned from native to_out, so zero LoRA-B gives
            # exact baseline parity while generic U-Net output weights stay frozen.
            hidden_bg_out = attn.to_out[0](hidden_bg)
            hidden_face_out = self.face_to_out(hidden_face * self.scale)
            hidden_ref_out = attn.to_out[0](hidden_ref)
            merged = (
                hidden_bg_out * (1 - mask_flat)
                + hidden_face_out * mask_flat
            )
            hidden_states = torch.cat([merged, hidden_ref_out], dim=0)

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

    @staticmethod
    def _warp_reference_roi_to_target(
        reference_hidden: torch.Tensor,
        *,
        reference_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly express a masked reference ROI in the target bbox frame."""
        batch_size, seq_len, channels = reference_hidden.shape
        side = int(math.isqrt(seq_len))
        if side * side != seq_len:
            raise RuntimeError(f"reference sequence length {seq_len} is not square")

        reference_mask_2d = reference_mask.reshape(batch_size, side, side) > 0.5
        target_mask_2d = target_mask.reshape(batch_size, side, side) > 0.5
        if not bool(reference_mask_2d.flatten(1).any(dim=1).all()):
            raise RuntimeError("reference ROI warp received an empty reference mask")
        if not bool(target_mask_2d.flatten(1).any(dim=1).all()):
            raise RuntimeError("reference ROI warp received an empty target mask")

        def bounds(mask_2d: torch.Tensor):
            rows = mask_2d.any(dim=2)
            cols = mask_2d.any(dim=1)
            y0 = rows.float().argmax(dim=1)
            x0 = cols.float().argmax(dim=1)
            y1 = (side - 1) - rows.flip(1).float().argmax(dim=1)
            x1 = (side - 1) - cols.flip(1).float().argmax(dim=1)
            return x0.float(), y0.float(), x1.float(), y1.float()

        ref_x0, ref_y0, ref_x1, ref_y1 = bounds(reference_mask_2d)
        tgt_x0, tgt_y0, tgt_x1, tgt_y1 = bounds(target_mask_2d)
        device = reference_hidden.device
        coord_dtype = torch.float32
        ys = torch.arange(side, device=device, dtype=coord_dtype)[None, :, None]
        xs = torch.arange(side, device=device, dtype=coord_dtype)[None, None, :]

        target_width = (tgt_x1 - tgt_x0).clamp_min(1.0)[:, None, None]
        target_height = (tgt_y1 - tgt_y0).clamp_min(1.0)[:, None, None]
        relative_x = (xs - tgt_x0[:, None, None]) / target_width
        relative_y = (ys - tgt_y0[:, None, None]) / target_height
        source_x = ref_x0[:, None, None] + relative_x * (
            ref_x1 - ref_x0
        )[:, None, None]
        source_y = ref_y0[:, None, None] + relative_y * (
            ref_y1 - ref_y0
        )[:, None, None]

        if side > 1:
            grid_x = source_x.mul(2.0 / float(side - 1)).sub(1.0)
            grid_y = source_y.mul(2.0 / float(side - 1)).sub(1.0)
        else:
            grid_x = torch.zeros_like(source_x)
            grid_y = torch.zeros_like(source_y)
        grid = torch.stack(
            [grid_x.expand(-1, side, -1), grid_y.expand(-1, -1, side)],
            dim=-1,
        )

        source = reference_hidden.transpose(1, 2).reshape(
            batch_size, channels, side, side
        )
        warped = F.grid_sample(
            source.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(dtype=reference_hidden.dtype)
        warped = warped * target_mask_2d[:, None].to(dtype=warped.dtype)
        return warped.flatten(2).transpose(1, 2)
    
    
    def _prepare_mask(self, mask: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        """Prepare mask for attention ops — always resize in 2-D (no 1-D raster)."""
        cache_key = (
            int(target_len),
            int(batch_size),
            bool(getattr(self, "force_binary_masks", False)),
            str(mask.device),
            str(mask.dtype),
        )
        if getattr(self, "cache_prepared_masks", False):
            prepared_cache = getattr(mask, "_ba_prepared_mask_cache", None)
            if prepared_cache is not None and cache_key in prepared_cache:
                return prepared_cache[cache_key]

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
        result = m.view(batch_size, 1, target_len, 1)
        if getattr(self, "cache_prepared_masks", False):
            prepared_cache = getattr(mask, "_ba_prepared_mask_cache", None)
            if prepared_cache is None:
                prepared_cache = {}
                mask._ba_prepared_mask_cache = prepared_cache
            prepared_cache[cache_key] = result
        return result
    
    
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

        q_ref = query_ref.view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)

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
