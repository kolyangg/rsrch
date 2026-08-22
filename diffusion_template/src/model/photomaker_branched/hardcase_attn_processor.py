"""CL19/CL23/CL27/CL39 self-attention extensions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .attn_processor_cleanest import BranchedAttnProcessor


# 22 Aug 2026 - CL19/23/27/39 equations are isolated here so the shared E13
# processor remains the concise target-Q/reference-KV hard-replacement path.
class HardcaseBranchedAttnProcessor(BranchedAttnProcessor):
    """Full-query router selected only for the hard-case recipe family."""

    def __init__(
        self,
        *,
        hidden_size: int,
        cross_attention_dim: int,
        scale: float,
        hardcase_mode: str,
        hardcase_transition_cells: int = 2,
        hardcase_frequency_low_early: float = 0.50,
        hardcase_frequency_low_late: float = 0.85,
        hardcase_frequency_high_early: float = 0.75,
        hardcase_frequency_high_late: float = 1.25,
        frequency_surface_loss_enabled: bool = False,
        frequency_surface_top_low_band_factor: float = 0.25,
        frequency_surface_visible_floor_ratio: float = 0.35,
        null_key_router_enabled: bool = False,
        null_key_entropy_threshold: float = 0.75,
        null_key_temperature: float = 0.08,
        null_key_max_abstention: float = 0.75,
        null_key_min_reference_fraction: float = 0.25,
    ):
        super().__init__(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            scale=scale,
        )
        self.hardcase_mode = str(hardcase_mode).lower()
        if self.hardcase_mode not in {"soft_router", "temporal_frequency"}:
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
        self.ba_denoise_progress = None
        self.ownership_target_mask = None
        self._frequency_surface_aux_loss = None
        self._latest_ba_telemetry = None

    def set_denoise_progress(self, progress: Optional[torch.Tensor]) -> None:
        self.ba_denoise_progress = progress

    def set_ownership_target_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.ownership_target_mask = mask

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
        prepared = self._prepare_mask(mask, length, batch).squeeze(1)
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
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        if self.hardcase_mode == "soft_router":
            return self._call_soft_router(attn, hidden_states, temb)
        return self._call_temporal_frequency(attn, hidden_states, temb)


def create_hardcase_processor(pipeline, name: str, hidden_size: int, scale: float):
    """Build a hard-case processor only for a leaf's declared U-Net groups."""
    mode = str(pipeline.ba_hardcase_mode)
    if mode == "off" or not any(
        name.startswith(f"{group}.") for group in pipeline.ba_hardcase_groups
    ):
        return None
    in_surface_group = any(
        name.startswith(f"{group}.")
        for group in pipeline.ba_frequency_surface_loss_groups
    )
    in_null_key_group = any(
        name.startswith(f"{group}.") for group in pipeline.ba_null_key_router_groups
    )
    return HardcaseBranchedAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=hidden_size,
        scale=scale,
        hardcase_mode=mode,
        hardcase_transition_cells=pipeline.ba_hardcase_transition_cells,
        hardcase_frequency_low_early=pipeline.ba_hardcase_frequency_low_early,
        hardcase_frequency_low_late=pipeline.ba_hardcase_frequency_low_late,
        hardcase_frequency_high_early=pipeline.ba_hardcase_frequency_high_early,
        hardcase_frequency_high_late=pipeline.ba_hardcase_frequency_high_late,
        frequency_surface_loss_enabled=(
            pipeline.ba_frequency_surface_loss_enabled and in_surface_group
        ),
        frequency_surface_top_low_band_factor=(
            pipeline.ba_frequency_surface_top_low_band_factor
        ),
        frequency_surface_visible_floor_ratio=(
            pipeline.ba_frequency_surface_visible_floor_ratio
        ),
        null_key_router_enabled=(
            pipeline.ba_null_key_router_enabled and in_null_key_group
        ),
        null_key_entropy_threshold=pipeline.ba_null_key_entropy_threshold,
        null_key_temperature=pipeline.ba_null_key_temperature,
        null_key_max_abstention=pipeline.ba_null_key_max_abstention,
        null_key_min_reference_fraction=(
            pipeline.ba_null_key_min_reference_fraction
        ),
    )
