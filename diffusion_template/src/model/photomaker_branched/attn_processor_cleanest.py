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
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        branched_attn_weight_mode: str = "shared",
        branched_attn_new_weight_kind: str = "full",
        branched_attn_lora_rank: int = 16,
        hardcase_mode: str = "off",
        hardcase_transition_cells: int = 2,
        hardcase_frequency_low_early: float = 0.50,
        hardcase_frequency_low_late: float = 0.85,
        hardcase_frequency_high_early: float = 0.75,
        hardcase_frequency_high_late: float = 1.25,
        hardcase_telemetry_enabled: bool = False,
        frequency_surface_loss_enabled: bool = False,
        frequency_surface_top_low_band_factor: float = 0.25,
        frequency_surface_visible_floor_ratio: float = 0.35,
        null_key_router_enabled: bool = False,
        null_key_entropy_threshold: float = 0.75,
        null_key_temperature: float = 0.08,
        null_key_max_abstention: float = 0.75,
        null_key_min_reference_fraction: float = 0.25,
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
        self.hardcase_mode = str(hardcase_mode or "off").lower()
        if self.hardcase_mode not in {"off", "soft_router", "temporal_frequency"}:
            raise ValueError(f"Unknown hardcase_mode={hardcase_mode!r}")
        self.hardcase_transition_cells = int(hardcase_transition_cells)
        if self.hardcase_transition_cells < 1:
            raise ValueError("hardcase_transition_cells must be positive")
        self.hardcase_frequency_low_early = float(hardcase_frequency_low_early)
        self.hardcase_frequency_low_late = float(hardcase_frequency_low_late)
        self.hardcase_frequency_high_early = float(hardcase_frequency_high_early)
        self.hardcase_frequency_high_late = float(hardcase_frequency_high_late)
        if min(
            self.hardcase_frequency_low_early,
            self.hardcase_frequency_low_late,
            self.hardcase_frequency_high_early,
            self.hardcase_frequency_high_late,
        ) < 0.50:
            raise ValueError("Temporal-frequency reference scales require a 0.50 floor")
        self.hardcase_telemetry_enabled = bool(hardcase_telemetry_enabled)
        self.frequency_surface_loss_enabled = bool(frequency_surface_loss_enabled)
        self.frequency_surface_top_low_band_factor = float(
            frequency_surface_top_low_band_factor
        )
        self.frequency_surface_visible_floor_ratio = float(
            frequency_surface_visible_floor_ratio
        )
        self.null_key_router_enabled = bool(null_key_router_enabled)
        self.null_key_entropy_threshold = float(null_key_entropy_threshold)
        self.null_key_temperature = float(null_key_temperature)
        self.null_key_max_abstention = float(null_key_max_abstention)
        self.null_key_min_reference_fraction = float(null_key_min_reference_fraction)
        if self.frequency_surface_loss_enabled and self.hardcase_mode != "temporal_frequency":
            raise ValueError("CL27 frequency-surface loss requires temporal_frequency")
        if not 0.0 <= self.frequency_surface_top_low_band_factor <= 1.0:
            raise ValueError("frequency_surface_top_low_band_factor must be in [0, 1]")
        if not 0.0 < self.frequency_surface_visible_floor_ratio < 1.0:
            raise ValueError("frequency_surface_visible_floor_ratio must be in (0, 1)")
        if not (
            0.0 <= self.null_key_max_abstention <= 1.0
            and 0.0 < self.null_key_min_reference_fraction <= 1.0
        ):
            raise ValueError("CL39 null-key bounds are invalid")
        if self.null_key_temperature <= 0.0:
            raise ValueError("CL39 null-key temperature must be positive")
        
        self.mask = None
        self.mask_ref = None
        self.ref_to_q = None
        self.ref_to_k = None
        self.ref_to_v = None
        self.noise_to_q = None
        self.noise_to_k = None
        self.noise_to_v = None
        self.ba_denoise_progress = None
        self.ownership_target_mask = None
        self._frequency_surface_aux_loss = None
        self._latest_ba_telemetry = None
        
        # If True: keep masks strictly binary after resize (avoids soft boundary blending)
        self.force_binary_masks: bool = True # False
        # 10 Aug 2026 - E13C-PERF-02: Reuse resized masks within one forward;
        # the cache is attached to the current mask tensor and cannot cross
        # samples or steps, so attention values remain unchanged.
        self.cache_prepared_masks: bool = False
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

    def set_denoise_progress(self, progress: Optional[torch.Tensor]) -> None:
        self.ba_denoise_progress = progress

    def set_ownership_target_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.ownership_target_mask = mask

    def set_hardcase_telemetry_enabled(self, enabled: bool) -> None:
        self.hardcase_telemetry_enabled = bool(enabled)

    def frequency_surface_aux_loss(self):
        return self._frequency_surface_aux_loss

    def latest_ba_telemetry(self):
        return self._latest_ba_telemetry

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
            normalized = normalized.view(
                total_batch, channels, height * width
            ).transpose(1, 2)
        elif input_ndim != 3:
            raise RuntimeError(f"Unsupported attention input rank: {input_ndim}")
        if normalized.shape[0] % 2:
            raise RuntimeError("CL19 requires [target, reference] doubled batches")
        batch = normalized.shape[0] // 2
        target = normalized[:batch]
        reference = normalized[batch:]
        if attn.group_norm is not None:
            target = attn.group_norm(target.transpose(1, 2)).transpose(1, 2)
            reference = attn.group_norm(reference.transpose(1, 2)).transpose(1, 2)
        return target, reference, input_ndim, spatial

    def _binary_mask(self, mask: torch.Tensor, length: int, batch: int, dtype):
        previous = self.force_binary_masks
        self.force_binary_masks = True
        try:
            prepared = self._prepare_mask(mask, length, batch).squeeze(1)
        finally:
            self.force_binary_masks = previous
        return prepared.to(dtype=dtype)

    def _soft_router_mask(self, mask: torch.Tensor, length: int, batch: int, dtype):
        binary = self._binary_mask(mask, length, batch, torch.float32)
        side = int(math.isqrt(length))
        image = binary.transpose(1, 2).reshape(batch, 1, side, side)
        remaining = image
        result = torch.ones_like(image)
        for index in range(self.hardcase_transition_cells):
            eroded = 1.0 - F.max_pool2d(
                1.0 - remaining, 3, stride=1, padding=1
            )
            ring = (remaining - eroded).clamp(0.0, 1.0)
            phase = float(index + 1) / float(self.hardcase_transition_cells + 1)
            weight = 0.5 - 0.5 * math.cos(math.pi * phase)
            result = result * (1.0 - ring) + ring * weight
            remaining = eroded
        return (result * image).flatten(2).transpose(1, 2).to(dtype=dtype)

    @staticmethod
    def _gaussian_split(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = delta.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise RuntimeError("Temporal-frequency BA requires square token grids")
        image = delta.float().transpose(1, 2).reshape(batch, channels, side, side)
        kernel_1d = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 5, 5)
        low = F.conv2d(
            image,
            kernel.expand(channels, 1, -1, -1),
            padding=2,
            groups=channels,
        ).flatten(2).transpose(1, 2)
        return low.to(delta.dtype), (delta.float() - low).to(delta.dtype)

    def _progress(self, target: torch.Tensor) -> torch.Tensor:
        if self.ba_denoise_progress is None:
            return target.new_zeros(target.shape[0], 1, 1)
        value = torch.as_tensor(
            self.ba_denoise_progress, device=target.device, dtype=target.dtype
        ).reshape(-1, 1, 1)
        if value.shape[0] == 1:
            value = value.expand(target.shape[0], -1, -1)
        if value.shape[0] != target.shape[0]:
            raise RuntimeError("Temporal-frequency progress batch mismatch")
        return value.clamp(0.0, 1.0)

    @staticmethod
    def _masked_mean_square(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = (mask.float().sum(dim=(1, 2)) * tensor.shape[-1]).clamp_min(1.0)
        return (tensor.float().square() * mask.float()).sum(dim=(1, 2)) / denom

    def _frequency_surface_loss(
        self,
        native_out: torch.Tensor,
        low_component: torch.Tensor,
        high_component: torch.Tensor,
        routed_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = native_out.float().new_tensor(0.0)
        metrics = {
            "top_high_rms": zero,
            "top_low_rms": zero,
            "visible_ratio": zero,
            "applied_fraction": zero,
        }
        self._frequency_surface_aux_loss = None
        if (
            not self.frequency_surface_loss_enabled
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return metrics
        if self.ownership_target_mask is None:
            raise RuntimeError("CL27 frequency-surface loss requires an ownership mask")
        batch, length, _ = native_out.shape
        face = self._binary_mask(self.mask, length, batch, torch.float32)
        top = self._binary_mask(
            self.ownership_target_mask, length, batch, torch.float32
        ) * face
        visible = (face - top).clamp(0.0, 1.0)
        eligible = (top.sum(dim=(1, 2)) > 0.0) & (visible.sum(dim=(1, 2)) > 0.0)
        eligible_float = eligible.float()
        eligible_count = eligible_float.sum().clamp_min(1.0)
        top_high = self._masked_mean_square(high_component, top)
        top_low = self._masked_mean_square(low_component, top)
        routed_rms = self._masked_mean_square(routed_delta, visible).clamp_min(1e-12).sqrt()
        native_rms = self._masked_mean_square(native_out, visible).clamp_min(1e-12).sqrt()
        ratio = routed_rms / native_rms.detach().clamp_min(1e-6)
        # 18 Aug 2026 - AICODE-NOTE: CL27 eligibility remains on-device; a
        # Python bool here synchronizes CUDA once per selected processor.
        top_loss = (
            (top_high + self.frequency_surface_top_low_band_factor * top_low)
            * eligible_float
        ).sum() / eligible_count
        floor_loss = (
            F.relu(ratio.new_tensor(self.frequency_surface_visible_floor_ratio) - ratio).square()
            * eligible_float
        ).sum() / eligible_count
        self._frequency_surface_aux_loss = (top_loss, floor_loss)
        metrics.update(
            top_high_rms=(top_high * eligible_float).sum().div(eligible_count).sqrt().detach(),
            top_low_rms=(top_low * eligible_float).sum().div(eligible_count).sqrt().detach(),
            visible_ratio=(ratio * eligible_float).sum().div(eligible_count).detach(),
            applied_fraction=eligible_float.mean().detach(),
        )
        return metrics

    def _full_target_lanes(self, attn, target, reference):
        batch, length, _ = target.shape
        heads = int(attn.heads)
        query = self._reshape_heads(self._q_noise(attn, target), heads)
        native = F.scaled_dot_product_attention(
            query,
            self._reshape_heads(self._k_noise(attn, target), heads),
            self._reshape_heads(self._v_noise(attn, target), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        ref_mask = self._binary_mask(
            self.mask_ref, length, batch, reference.dtype
        )
        reference_face = reference * ref_mask
        reference_message = F.scaled_dot_product_attention(
            query,
            self._reshape_heads(self._k_ref(attn, reference_face), heads),
            self._reshape_heads(self._v_ref(attn, reference_face), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        return (
            attn.to_out[0](self._merge_heads(native)),
            attn.to_out[0](self._merge_heads(reference_message)),
            query,
        )

    def _null_key_confidence(
        self, attn, query: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CL39's detached per-query reference confidence."""
        batch, heads, length, width = query.shape
        ref_mask = self._binary_mask(
            self.mask_ref, length, batch, reference.dtype
        )
        keys = self._reshape_heads(
            self._k_ref(attn, reference * ref_mask), heads
        ).detach().float()
        entropy_chunks = []
        with torch.no_grad():
            for query_chunk in query.detach().float().split(256, dim=2):
                logits = torch.matmul(
                    query_chunk, keys.transpose(-1, -2)
                ) / math.sqrt(width)
                probability = logits.softmax(dim=-1)
                entropy = -(
                    probability * probability.clamp_min(1.0e-8).log()
                ).sum(dim=-1) / math.log(max(length, 2))
                entropy_chunks.append(entropy.mean(dim=1)[..., None])
            entropy = torch.cat(entropy_chunks, dim=1)
            null_mass = torch.sigmoid(
                (entropy - self.null_key_entropy_threshold)
                / self.null_key_temperature
            )
            confidence = (
                1.0 - self.null_key_max_abstention * null_mass
            ).clamp(min=self.null_key_min_reference_fraction, max=1.0)
        return confidence.to(query.dtype), null_mass

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
        joined = attn.to_out[1](torch.cat([target_out, reference_out], dim=0))
        if input_ndim == 4:
            channels, height, width = spatial
            joined = joined.transpose(-1, -2).reshape(
                joined.shape[0], channels, height, width
            )
        if attn.residual_connection:
            joined = joined + residual
        return joined / attn.rescale_output_factor

    def _call_soft_router(self, attn, hidden_states, temb) -> torch.Tensor:
        # 12 Aug 2026 - AICODE-NOTE: CL19 computes full native and full
        # target-Q/reference-KV messages, then applies one cosine blend. The
        # reference key mask remains binary, preserving the historical sinks.
        residual = hidden_states
        target, reference, input_ndim, spatial = self._normalized_halves(
            attn, hidden_states, temb
        )
        native_out, reference_out, _query = self._full_target_lanes(
            attn, target, reference
        )
        router = self._soft_router_mask(
            self.mask, target.shape[1], target.shape[0], native_out.dtype
        )
        target_out = native_out * (1.0 - router) + reference_out * router
        return self._finish_full_router(
            attn, residual, target_out, reference, input_ndim, spatial
        )

    def _call_temporal_frequency(self, attn, hidden_states, temb) -> torch.Tensor:
        # 18 Aug 2026 - CL23 keeps CL19's full native/reference messages and
        # cosine router; only the routed reference-minus-native frequency gains
        # vary with the real scheduler timestep.
        residual = hidden_states
        target, reference, input_ndim, spatial = self._normalized_halves(
            attn, hidden_states, temb
        )
        native_out, reference_out, query = self._full_target_lanes(
            attn, target, reference
        )
        router = self._soft_router_mask(
            self.mask, target.shape[1], target.shape[0], native_out.dtype
        )
        low, high = self._gaussian_split(reference_out - native_out)
        progress = self._progress(target)
        low_scale = self.hardcase_frequency_low_early + progress * (
            self.hardcase_frequency_low_late - self.hardcase_frequency_low_early
        )
        high_scale = self.hardcase_frequency_high_early + progress * (
            self.hardcase_frequency_high_late - self.hardcase_frequency_high_early
        )
        low_component = router * low_scale * low
        high_component = router * high_scale * high
        null_telemetry = {}
        if self.null_key_router_enabled:
            # 21 Aug 2026 - AICODE-NOTE: CL39 scales only CL27's reference
            # delta with detached confidence; native target SA stays intact.
            confidence, null_mass = self._null_key_confidence(
                attn, query, reference
            )
            low_component = low_component * confidence
            high_component = high_component * confidence
            object_minus_visible = null_mass.new_tensor(0.0)
            if self.ownership_target_mask is not None:
                face = self._binary_mask(
                    self.mask, target.shape[1], target.shape[0], torch.float32
                )
                top = self._binary_mask(
                    self.ownership_target_mask,
                    target.shape[1],
                    target.shape[0],
                    torch.float32,
                ) * face
                visible = (face - top).clamp(0.0, 1.0)
                object_minus_visible = (
                    (null_mass * top).sum() / top.sum().clamp_min(1.0)
                    - (null_mass * visible).sum()
                    / visible.sum().clamp_min(1.0)
                )
            null_telemetry = {
                "null_key/null_mass": null_mass.mean(),
                "null_key/reference_fraction": confidence.float().mean(),
                "null_key/object_minus_visible_mass": object_minus_visible,
            }
        routed_delta = low_component + high_component
        target_out = native_out + routed_delta
        self._latest_ba_telemetry = null_telemetry or None
        if self.hardcase_telemetry_enabled:
            if self._latest_ba_telemetry is None:
                self._latest_ba_telemetry = {}
            self._latest_ba_telemetry.update({
                "frequency_low_scale": low_scale.detach().float().mean(),
                "frequency_high_scale": high_scale.detach().float().mean(),
            })
        if self.frequency_surface_loss_enabled:
            surface = self._frequency_surface_loss(
                native_out, low_component, high_component, routed_delta
            )
            if self._latest_ba_telemetry is None:
                self._latest_ba_telemetry = {}
            self._latest_ba_telemetry.update(
                frequency_surface_top_high_rms=surface["top_high_rms"],
                frequency_surface_top_low_rms=surface["top_low_rms"],
                frequency_surface_visible_ratio=surface["visible_ratio"],
                frequency_surface_applied_fraction=surface["applied_fraction"],
            )
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


        if self.hardcase_mode == "soft_router":
            return self._call_soft_router(attn, hidden_states, temb)
        if self.hardcase_mode == "temporal_frequency":
            return self._call_temporal_frequency(attn, hidden_states, temb)

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

        ref_mask = self._prepare_mask(self.mask_ref, seq_len, ref_batch_size)
        ref_mask = ref_mask.to(dtype=ref_hidden.dtype, device=ref_hidden.device)
        ref_mask_flat = ref_mask.squeeze(1)  # [B, L, 1]


        # Extract face regions from both noise and reference
        noise_face_hidden = noise_hidden * mask_flat  # Face from current noise
        ref_face_hidden = ref_hidden * ref_mask_flat

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
        cache_key = (
            int(target_len),
            int(batch_size),
            bool(getattr(self, "force_binary_masks", False)),
            str(mask.device),
            str(mask.dtype),
        )
        if self.cache_prepared_masks:
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
        if self.cache_prepared_masks:
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
