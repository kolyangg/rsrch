"""
attn_processor.py - Branched attention processors with consistent batch handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .identity_conditioning import (
    IDAdaptiveModulation,
    IdentityMotionProjector,
    similarity_grid_from_landmarks,
)


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
        hardcase_top_native_floor: float = 0.95,
        hardcase_frequency_low_early: float = 0.50,
        hardcase_frequency_low_late: float = 0.85,
        hardcase_frequency_high_early: float = 0.75,
        hardcase_frequency_high_late: float = 1.25,
        hardcase_telemetry_enabled: bool = True,
        frequency_surface_experiment_enabled: bool = False,
        frequency_surface_loss_enabled: bool = False,
        frequency_surface_top_low_band_factor: float = 0.25,
        frequency_surface_visible_floor_ratio: float = 0.35,
        frequency_learnable_schedule_enabled: bool = False,
        frequency_low_late_center: float = 0.85,
        frequency_low_late_half_range: float = 0.15,
        frequency_high_early_center: float = 0.75,
        frequency_high_early_half_range: float = 0.15,
        frequency_high_late_center: float = 1.25,
        frequency_high_late_half_range: float = 0.15,
        frequency_lowband_contrastive_enabled: bool = False,
        attention_ownership_enabled: bool = False,
        attention_ownership_visible_floor: float = 0.55,
        attention_ownership_top_ceiling: float = 0.10,
        attention_ownership_contact_width: int = 1,
        frequency_surface_region_mode: str = "full_top",
        frequency_surface_contact_width: int = 1,
        frequency_surface_top_interior_factor: float = 1.0,
        frequency_surface_contact_factor: float = 1.0,
        frequency_shared_schedule_enabled: bool = False,
        frequency_shared_low_late_center: float = 0.85,
        frequency_shared_low_late_half_range: float = 0.05,
        frequency_shared_high_early_center: float = 0.75,
        frequency_shared_high_early_half_range: float = 0.05,
        frequency_shared_high_late_center: float = 1.25,
        frequency_shared_high_late_half_range: float = 0.05,
        roi_teacher_enabled: bool = False,
        roi_teacher_size: int = 32,
        roi_teacher_face_threshold_px: int = 256,
        roi_teacher_progress_min: float = 0.60,
        hardcase_roi_gate_init: float = 0.10,
        hardcase_roi_gate_min: float = 0.05,
        hardcase_roi_progress_min: float = 0.60,
        hardcase_roi_rms_cap: float = 0.25,
        visibility_ownership_v2_enabled: bool = False,
        visibility_ownership_v2_dilate_cells: int = 1,
        visibility_ownership_v2_min_top_area: float = 0.002,
        visibility_ownership_v2_delta_only: bool = False,
        null_key_router_enabled: bool = False,
        null_key_entropy_threshold: float = 0.75,
        null_key_temperature: float = 0.08,
        null_key_max_abstention: float = 0.75,
        null_key_min_reference_fraction: float = 0.25,
        landmark_canonical_kv_enabled: bool = False,
        landmark_canonical_kv_mix: float = 0.50,
        landmark_canonical_kv_min_confidence: float = 0.80,
        component_token_memory_enabled: bool = False,
        component_token_memory_scale: float = 0.15,
        component_token_memory_sigma_cells: float = 1.75,
        component_token_memory_min_confidence: float = 0.80,
        identity_motion_projector_enabled: bool = False,
        identity_motion_projector_rank: int = 32,
        identity_motion_projector_gate_max: float = 0.35,
        identity_motion_projector_ramp_start_step: int = 1000,
        identity_motion_projector_ramp_end_step: int = 6000,
        id_adaptive_modulation_enabled: bool = False,
        id_adaptive_modulation_embedding_dim: int = 512,
        id_adaptive_modulation_bottleneck: int = 32,
        id_adaptive_modulation_scale_max: float = 0.20,
        id_adaptive_modulation_ramp_start_step: int = 1000,
        id_adaptive_modulation_ramp_end_step: int = 6000,
        semantic_window_gate_enabled: bool = False,
        semantic_window_progress_start: float = 0.20,
        semantic_window_progress_end: float = 0.85,
        semantic_window_progress_temperature: float = 0.08,
        semantic_window_agreement_threshold: float = 0.15,
        semantic_window_agreement_temperature: float = 0.08,
        semantic_window_min_scale: float = 0.60,
        semantic_window_max_scale: float = 1.15,
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
            "visibility_order",
            "temporal_frequency",
            "anchored_roi",
        }:
            raise ValueError(f"Unknown hardcase_mode={hardcase_mode!r}")
        self.hardcase_rank = int(hardcase_rank)
        self.hardcase_gate_max = float(hardcase_gate_max)
        self.hardcase_roi_size = int(hardcase_roi_size)
        self.hardcase_face_threshold_px = int(hardcase_face_threshold_px)
        self.hardcase_transition_cells = int(hardcase_transition_cells)
        self.hardcase_visible_face_floor = float(hardcase_visible_face_floor)
        self.hardcase_top_native_floor = float(hardcase_top_native_floor)
        self.hardcase_frequency_low_early = float(hardcase_frequency_low_early)
        self.hardcase_frequency_low_late = float(hardcase_frequency_low_late)
        self.hardcase_frequency_high_early = float(hardcase_frequency_high_early)
        self.hardcase_frequency_high_late = float(hardcase_frequency_high_late)
        self.hardcase_telemetry_enabled = bool(hardcase_telemetry_enabled)
        self.frequency_surface_experiment_enabled = bool(
            frequency_surface_experiment_enabled
        )
        self.frequency_surface_loss_enabled = bool(frequency_surface_loss_enabled)
        self.frequency_surface_top_low_band_factor = float(
            frequency_surface_top_low_band_factor
        )
        self.frequency_surface_visible_floor_ratio = float(
            frequency_surface_visible_floor_ratio
        )
        self.frequency_learnable_schedule_enabled = bool(
            frequency_learnable_schedule_enabled
        )
        self.frequency_low_late_center = float(frequency_low_late_center)
        self.frequency_low_late_half_range = float(frequency_low_late_half_range)
        self.frequency_high_early_center = float(frequency_high_early_center)
        self.frequency_high_early_half_range = float(
            frequency_high_early_half_range
        )
        self.frequency_high_late_center = float(frequency_high_late_center)
        self.frequency_high_late_half_range = float(frequency_high_late_half_range)
        self.frequency_lowband_contrastive_enabled = bool(
            frequency_lowband_contrastive_enabled
        )
        self.attention_ownership_enabled = bool(attention_ownership_enabled)
        self.attention_ownership_visible_floor = float(
            attention_ownership_visible_floor
        )
        self.attention_ownership_top_ceiling = float(
            attention_ownership_top_ceiling
        )
        self.attention_ownership_contact_width = int(
            attention_ownership_contact_width
        )
        self.frequency_surface_region_mode = str(frequency_surface_region_mode)
        self.frequency_surface_contact_width = int(frequency_surface_contact_width)
        self.frequency_surface_top_interior_factor = float(
            frequency_surface_top_interior_factor
        )
        self.frequency_surface_contact_factor = float(frequency_surface_contact_factor)
        self.frequency_shared_schedule_enabled = bool(
            frequency_shared_schedule_enabled
        )
        self.frequency_shared_low_late_center = float(
            frequency_shared_low_late_center
        )
        self.frequency_shared_low_late_half_range = float(
            frequency_shared_low_late_half_range
        )
        self.frequency_shared_high_early_center = float(
            frequency_shared_high_early_center
        )
        self.frequency_shared_high_early_half_range = float(
            frequency_shared_high_early_half_range
        )
        self.frequency_shared_high_late_center = float(
            frequency_shared_high_late_center
        )
        self.frequency_shared_high_late_half_range = float(
            frequency_shared_high_late_half_range
        )
        self._frequency_shared_schedule_raw = None
        self.roi_teacher_enabled = bool(roi_teacher_enabled)
        self.roi_teacher_size = int(roi_teacher_size)
        self.roi_teacher_face_threshold_px = int(roi_teacher_face_threshold_px)
        self.roi_teacher_progress_min = float(roi_teacher_progress_min)
        self.hardcase_roi_gate_init = float(hardcase_roi_gate_init)
        self.hardcase_roi_gate_min = float(hardcase_roi_gate_min)
        self.hardcase_roi_progress_min = float(hardcase_roi_progress_min)
        self.hardcase_roi_rms_cap = float(hardcase_roi_rms_cap)
        self.visibility_ownership_v2_enabled = bool(
            visibility_ownership_v2_enabled
        )
        self.visibility_ownership_v2_dilate_cells = int(visibility_ownership_v2_dilate_cells)
        self.visibility_ownership_v2_min_top_area = float(
            visibility_ownership_v2_min_top_area
        )
        self.visibility_ownership_v2_delta_only = bool(
            visibility_ownership_v2_delta_only
        )
        self.null_key_router_enabled = bool(null_key_router_enabled)
        self.null_key_entropy_threshold = float(null_key_entropy_threshold)
        self.null_key_temperature = float(null_key_temperature)
        self.null_key_max_abstention = float(null_key_max_abstention)
        self.null_key_min_reference_fraction = float(null_key_min_reference_fraction)
        self.landmark_canonical_kv_enabled = bool(landmark_canonical_kv_enabled)
        self.landmark_canonical_kv_mix = float(landmark_canonical_kv_mix)
        self.landmark_canonical_kv_min_confidence = float(
            landmark_canonical_kv_min_confidence
        )
        self.component_token_memory_enabled = bool(component_token_memory_enabled)
        self.component_token_memory_scale = float(component_token_memory_scale)
        self.component_token_memory_sigma_cells = float(
            component_token_memory_sigma_cells
        )
        self.component_token_memory_min_confidence = float(
            component_token_memory_min_confidence
        )
        self.identity_motion_projector_enabled = bool(identity_motion_projector_enabled)
        self.identity_motion_projector_gate_max = float(identity_motion_projector_gate_max)
        self.identity_motion_projector_ramp_start_step = int(
            identity_motion_projector_ramp_start_step
        )
        self.identity_motion_projector_ramp_end_step = int(
            identity_motion_projector_ramp_end_step
        )
        self.id_adaptive_modulation_enabled = bool(id_adaptive_modulation_enabled)
        self.id_adaptive_modulation_scale_max = float(id_adaptive_modulation_scale_max)
        self.id_adaptive_modulation_ramp_start_step = int(
            id_adaptive_modulation_ramp_start_step
        )
        self.id_adaptive_modulation_ramp_end_step = int(
            id_adaptive_modulation_ramp_end_step
        )
        self.semantic_window_gate_enabled = bool(semantic_window_gate_enabled)
        self.semantic_window_progress_start = float(semantic_window_progress_start)
        self.semantic_window_progress_end = float(semantic_window_progress_end)
        self.semantic_window_progress_temperature = float(semantic_window_progress_temperature)
        self.semantic_window_agreement_threshold = float(semantic_window_agreement_threshold)
        self.semantic_window_agreement_temperature = float(semantic_window_agreement_temperature)
        self.semantic_window_min_scale = float(semantic_window_min_scale)
        self.semantic_window_max_scale = float(semantic_window_max_scale)
        if self.hardcase_rank <= 0 or self.hardcase_roi_size <= 1:
            raise ValueError("Hard-case rank and ROI size must be positive")
        if not 0.0 < self.hardcase_gate_max <= 1.0:
            raise ValueError("hardcase_gate_max must be in (0, 1]")
        if self.hardcase_transition_cells < 1:
            raise ValueError("hardcase_transition_cells must be positive")
        if not 0.0 <= self.hardcase_visible_face_floor <= 1.0:
            raise ValueError("hardcase_visible_face_floor must be in [0, 1]")
        if not 0.0 < self.hardcase_top_native_floor <= 1.0:
            raise ValueError("hardcase_top_native_floor must be in (0, 1]")
        if min(
            self.hardcase_frequency_low_early,
            self.hardcase_frequency_low_late,
            self.hardcase_frequency_high_early,
            self.hardcase_frequency_high_late,
        ) < 0.50:
            raise ValueError("Temporal-frequency reference scales require a 0.50 floor")
        if not 0.0 <= self.frequency_surface_top_low_band_factor <= 1.0:
            raise ValueError("frequency_surface_top_low_band_factor must be in [0, 1]")
        if not 0.0 < self.frequency_surface_visible_floor_ratio < 1.0:
            raise ValueError("frequency_surface_visible_floor_ratio must be in (0, 1)")
        if min(
            self.frequency_low_late_half_range,
            self.frequency_high_early_half_range,
            self.frequency_high_late_half_range,
        ) <= 0.0:
            raise ValueError("Learnable frequency half-ranges must be positive")
        if not (
            0.0 < self.hardcase_roi_gate_min
            < self.hardcase_roi_gate_init
            < self.hardcase_gate_max <= 1.0
        ):
            raise ValueError("Anchored ROI gate bounds must satisfy 0 < min < init < max")
        if not 0.0 <= self.hardcase_roi_progress_min < 1.0:
            raise ValueError("hardcase_roi_progress_min must be in [0, 1)")
        if self.hardcase_roi_rms_cap <= 0.0:
            raise ValueError("hardcase_roi_rms_cap must be positive")
        if (
            self.visibility_ownership_v2_dilate_cells < 1
            or self.visibility_ownership_v2_min_top_area < 0.0
        ):
            raise ValueError("CL38 ownership geometry is invalid")
        if not (
            0.0 <= self.null_key_max_abstention <= 1.0
            and 0.0 < self.null_key_min_reference_fraction <= 1.0
        ):
            raise ValueError("CL39 null-key bounds are invalid")
        if self.null_key_temperature <= 0.0:
            raise ValueError("CL39 null-key temperature must be positive")
        if not 0.0 <= self.landmark_canonical_kv_mix <= 1.0:
            raise ValueError("CL41 canonical mix must be in [0, 1]")
        if min(self.component_token_memory_scale, self.component_token_memory_sigma_cells) <= 0.0:
            raise ValueError("CL42 component-memory controls must be positive")
        if not 0.0 < self.identity_motion_projector_gate_max <= 1.0:
            raise ValueError("CL40 projector gate must be in (0, 1]")
        if not 0.0 < self.id_adaptive_modulation_scale_max <= 1.0:
            raise ValueError("CL43 modulation scale must be in (0, 1]")
        if not (
            0.0 <= self.semantic_window_progress_start < self.semantic_window_progress_end <= 1.0
            and self.semantic_window_progress_temperature > 0.0
            and self.semantic_window_agreement_temperature > 0.0
            and 0.0 < self.semantic_window_min_scale <= self.semantic_window_max_scale
        ):
            raise ValueError("CL44 semantic-window controls are invalid")

        self.roi_gate_raw = None
        self.memory_to_k = None
        self.memory_to_v = None
        self.memory_to_out = None
        self.memory_gate_raw = None
        self.ownership_norm = None
        self.ownership_mlp = None
        self.ownership_scale_raw = None
        self.frequency_schedule_raw = None
        self.identity_motion_projector = None
        self.id_adaptive_modulation = None
        if self.hardcase_mode == "highres_roi":
            self.roi_gate_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        elif self.hardcase_mode == "anchored_roi":
            unit = (
                self.hardcase_roi_gate_init - self.hardcase_roi_gate_min
            ) / (self.hardcase_gate_max - self.hardcase_roi_gate_min)
            self.roi_gate_raw = nn.Parameter(
                torch.tensor(math.log(unit / (1.0 - unit)), dtype=trainable_dtype)
            )
        elif self.hardcase_mode == "clean_memory":
            self.memory_gate_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
        elif self.hardcase_mode in {"semantic_ownership", "visibility_order"}:
            ownership_hidden = int(hardcase_ownership_hidden_dim)
            if ownership_hidden <= 0:
                raise ValueError("hardcase_ownership_hidden_dim must be positive")
            self.ownership_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            output_dim = 3 if self.hardcase_mode == "visibility_order" else 1
            self.ownership_mlp = nn.Sequential(
                nn.Linear(hidden_size + 2, ownership_hidden),
                nn.SiLU(),
                nn.Linear(ownership_hidden, output_dim),
            )
            nn.init.zeros_(self.ownership_mlp[-1].weight)
            nn.init.zeros_(self.ownership_mlp[-1].bias)
            if self.hardcase_mode == "semantic_ownership":
                self.ownership_scale_raw = nn.Parameter(torch.zeros((), dtype=trainable_dtype))
            else:
                # 13 Aug 2026 - AICODE-NOTE: visibility routing starts as the
                # CL19 visible-face route, but the three probabilities directly
                # own generation and cannot hide behind a collapsible output gate.
                with torch.no_grad():
                    self.ownership_mlp[-1].bias.copy_(
                        torch.tensor([-4.0, 4.0, -4.0], dtype=self.ownership_mlp[-1].bias.dtype)
                    )
        if self.frequency_learnable_schedule_enabled:
            if self.hardcase_mode != "temporal_frequency":
                raise ValueError("Learnable frequency endpoints require temporal_frequency")
            self.frequency_schedule_raw = nn.Parameter(
                torch.zeros(3, dtype=trainable_dtype or torch.float32)
            )
        if self.identity_motion_projector_enabled:
            self.identity_motion_projector = IdentityMotionProjector(
                hidden_size, int(identity_motion_projector_rank)
            )
        if self.id_adaptive_modulation_enabled:
            self.id_adaptive_modulation = IDAdaptiveModulation(
                hidden_size,
                int(id_adaptive_modulation_bottleneck),
                int(id_adaptive_modulation_embedding_dim),
            )
        if (
            self.frequency_surface_experiment_enabled
            or self.frequency_lowband_contrastive_enabled
        ) and self.hardcase_mode != "temporal_frequency":
            raise ValueError("Frequency auxiliaries require temporal_frequency")
        self.clean_reference_memory = None
        self.capture_clean_memory = False
        self.ownership_target_mask = None
        self._ownership_aux_loss = None
        self._frequency_surface_aux_loss = None
        self._attention_ownership_capture = False
        self._attention_ownership_aux = None
        self._roi_teacher_capture = False
        self._roi_teacher_aux = None
        self._lowband_contrastive_mode = "off"
        self._lowband_negative_permutation = None
        self._lowband_anchor_q = None
        self._lowband_anchor_native_out = None
        self._lowband_anchor_embedding = None
        self._lowband_positive_embedding = None
        self._lowband_negative_embedding = None
        self.ba_denoise_progress = None
        self._latest_ba_telemetry = None
        self._visibility_ownership_v2_aux = None
        self.identity_embedding_512 = None
        self.reference_landmarks_5 = None
        self.reference_landmark_confidence = None
        self.ba_training_step = 0
        
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

    def set_training_step(self, step: int) -> None:
        self.ba_training_step = int(step)

    def set_identity_context(
        self,
        identity_embedding_512: Optional[torch.Tensor],
        reference_landmarks_5: Optional[torch.Tensor],
        reference_landmark_confidence: Optional[torch.Tensor],
    ) -> None:
        self.identity_embedding_512 = identity_embedding_512
        self.reference_landmarks_5 = reference_landmarks_5
        self.reference_landmark_confidence = reference_landmark_confidence

    def visibility_ownership_v2_aux(self):
        return self._visibility_ownership_v2_aux

    def ownership_aux_loss(self) -> Optional[torch.Tensor]:
        return self._ownership_aux_loss

    def frequency_surface_aux_loss(self):
        return self._frequency_surface_aux_loss

    def frequency_schedule_anchor_loss(self) -> Optional[torch.Tensor]:
        if self.frequency_schedule_raw is None:
            return None
        return self.frequency_schedule_raw.float().square().mean()

    def set_lowband_contrastive(
        self,
        mode: str,
        negative_permutation: Optional[torch.Tensor] = None,
    ) -> None:
        mode = str(mode or "off").lower()
        if mode not in {"off", "anchor", "contrast", "positive"}:
            raise ValueError(f"Unknown low-band contrastive mode {mode!r}")
        if mode != "off" and not self.frequency_lowband_contrastive_enabled:
            raise RuntimeError("Low-band capture enabled on an unselected processor")
        if mode == "off":
            self._lowband_anchor_q = None
            self._lowband_anchor_native_out = None
            self._lowband_anchor_embedding = None
            self._lowband_positive_embedding = None
            self._lowband_negative_embedding = None
        elif mode == "anchor":
            self._lowband_anchor_q = None
            self._lowband_anchor_native_out = None
            self._lowband_anchor_embedding = None
            self._lowband_positive_embedding = None
            self._lowband_negative_embedding = None
        elif self._lowband_anchor_q is None or self._lowband_anchor_native_out is None:
            raise RuntimeError("Low-band contrast pass has no matched anchor query")
        self._lowband_contrastive_mode = mode
        self._lowband_negative_permutation = negative_permutation

    def lowband_contrastive_embeddings(self):
        values = (
            self._lowband_anchor_embedding,
            self._lowband_positive_embedding,
            self._lowband_negative_embedding,
        )
        return values if all(value is not None for value in values) else None

    def lowband_positive_embeddings(self):
        values = (self._lowband_anchor_embedding, self._lowband_positive_embedding)
        return values if all(value is not None for value in values) else None

    def set_attention_ownership_capture(self, enabled: bool) -> None:
        self._attention_ownership_capture = bool(enabled)
        self._attention_ownership_aux = None

    def attention_ownership_aux(self):
        return self._attention_ownership_aux

    def set_roi_teacher_capture(self, enabled: bool) -> None:
        self._roi_teacher_capture = bool(enabled)
        self._roi_teacher_aux = None

    def roi_teacher_aux(self):
        return self._roi_teacher_aux

    def set_frequency_shared_schedule(self, parameter) -> None:
        object.__setattr__(self, "_frequency_shared_schedule_raw", parameter)

    def latest_ba_telemetry(self) -> Optional[dict[str, torch.Tensor]]:
        return self._latest_ba_telemetry

    def set_hardcase_telemetry_enabled(self, enabled: bool) -> None:
        self.hardcase_telemetry_enabled = bool(enabled)

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

    def _reference_target_out(self, attn, q, reference, mask_ref=None):
        batch, length, _ = reference.shape
        heads = int(attn.heads)
        ref_mask = self._binary_mask(
            self.mask_ref if mask_ref is None else mask_ref,
            length,
            batch,
            reference.dtype,
        )
        reference_face = reference * ref_mask
        message = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_ref(attn, reference_face), heads),
            self._reshape_heads(self._v_ref(attn, reference_face), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        message = self._merge_heads(message)
        return (
            self.face_to_out(message)
            if self.face_to_out is not None
            else attn.to_out[0](message)
        )

    def _full_target_lanes(self, attn, target, reference):
        batch, length, _ = target.shape
        heads = int(attn.heads)
        q = self._reshape_heads(self._q_noise(attn, target), heads)
        if self._lowband_contrastive_mode in {"contrast", "positive"}:
            cached_q = self._lowband_anchor_q
            if cached_q is None or cached_q.shape != q.shape:
                raise RuntimeError("Low-band contrastive query cache mismatch")
            q = cached_q
        native = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_noise(attn, target), heads),
            self._reshape_heads(self._v_noise(attn, target), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        native_message = self._merge_heads(native)
        native_out = attn.to_out[0](native_message)
        reference_out = self._reference_target_out(attn, q, reference)
        if self._attention_ownership_capture:
            self._capture_attention_ownership(attn, q, reference)
        if self._lowband_contrastive_mode == "anchor":
            self._lowband_anchor_q = q.detach()
            self._lowband_anchor_native_out = native_out.detach()
        return native_out, reference_out, q

    def _capture_attention_ownership(self, attn, q, reference) -> None:
        """Chunk Q/K probabilities so CL31 never retains a full all-layer map."""
        if not self.attention_ownership_enabled or self.ownership_target_mask is None:
            raise RuntimeError("Attention ownership capture lacks supervision")
        batch, _, length, width = q.shape
        face = self._binary_mask(self.mask, length, batch, torch.float32)
        top = self._binary_mask(
            self.ownership_target_mask, length, batch, torch.float32
        ) * face
        side = int(math.isqrt(length))
        top_image = top.transpose(1, 2).reshape(batch, 1, side, side)
        kernel = 2 * self.attention_ownership_contact_width + 1
        contact = F.max_pool2d(
            top_image, kernel, stride=1, padding=self.attention_ownership_contact_width
        ).flatten(2).transpose(1, 2) * face
        visible = (face - contact).clamp(0.0, 1.0)
        ref_mask = self._binary_mask(self.mask_ref, length, batch, torch.float32)
        reference_face = reference * ref_mask.to(reference.dtype)
        keys = self._reshape_heads(self._k_ref(attn, reference_face), int(attn.heads))
        key_face = ref_mask.squeeze(-1)[:, None, None, :]
        visible_denom = visible.sum().clamp_min(1.0)
        top_denom = contact.sum().clamp_min(1.0)
        # Pool the two supervised query regions before QK. This preserves
        # attention ownership while avoiding an LxL activation per layer.
        visible_q = (
            q.float() * visible[:, None]
        ).sum(dim=2) / visible.sum(dim=1).clamp_min(1.0)[:, None]
        top_q = (
            q.float() * contact[:, None]
        ).sum(dim=2) / contact.sum(dim=1).clamp_min(1.0)[:, None]
        pooled_q = torch.stack((visible_q, top_q), dim=2)
        logits = torch.matmul(pooled_q, keys.float().transpose(-1, -2)) / math.sqrt(width)
        mass = (logits.softmax(dim=-1) * key_face).sum(dim=-1).mean(dim=1)
        visible_mean, top_mean = mass[:, 0].mean(), mass[:, 1].mean()
        visible_loss = F.relu(
            mass[:, 0].new_tensor(self.attention_ownership_visible_floor) - mass[:, 0]
        ).square().mean()
        top_loss = F.relu(
            mass[:, 1] - mass[:, 1].new_tensor(self.attention_ownership_top_ceiling)
        ).square().mean()
        eligible = (visible.sum() > 0) & (contact.sum() > 0)
        scale = eligible.to(mass.dtype)
        self._attention_ownership_aux = (
            scale * (visible_loss + top_loss),
            visible_mean,
            scale * top_mean,
        )

    @staticmethod
    def _masked_mean_square(
        tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        denom = (mask.float().sum(dim=(1, 2)) * tensor.shape[-1]).clamp_min(1.0)
        return (tensor.float().square() * mask.float()).sum(dim=(1, 2)) / denom

    @staticmethod
    def _masked_pool(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum(dim=1).clamp_min(1.0)
        return (tensor.float() * mask.float()).sum(dim=1) / denom

    def _capture_lowband_contrastive(
        self,
        attn,
        reference: torch.Tensor,
        q: torch.Tensor,
        router: torch.Tensor,
        reference_out: torch.Tensor,
    ) -> None:
        mode = self._lowband_contrastive_mode
        if not self.frequency_lowband_contrastive_enabled or mode == "off":
            return
        native_anchor = self._lowband_anchor_native_out
        if native_anchor is None:
            raise RuntimeError("Low-band contrastive native anchor is missing")
        if mode == "anchor":
            # 14 Aug 2026 - AICODE-NOTE: CL29's auxiliary representation must
            # not encode target pixels through Q. Recompute only the reference
            # message under detached target queries; production routing is unchanged.
            reference_aux = self._reference_target_out(attn, q.detach(), reference)
            low, _ = self._gaussian_split(reference_aux - native_anchor)
            self._lowband_anchor_embedding = self._masked_pool(low, router)
            return

        if mode == "positive":
            low_positive, _ = self._gaussian_split(reference_out - native_anchor)
            self._lowband_positive_embedding = self._masked_pool(low_positive, router)
            return
        permutation = self._lowband_negative_permutation
        if permutation is None or permutation.numel() != reference.shape[0]:
            raise RuntimeError("Low-band negative permutation is missing")
        low_positive, _ = self._gaussian_split(reference_out - native_anchor)
        wrong_reference = reference.index_select(0, permutation)
        wrong_mask = self.mask_ref.index_select(0, permutation)
        wrong_out = self._reference_target_out(
            attn, q, wrong_reference, mask_ref=wrong_mask
        )
        low_negative, _ = self._gaussian_split(wrong_out - native_anchor)
        self._lowband_positive_embedding = self._masked_pool(low_positive, router)
        self._lowband_negative_embedding = self._masked_pool(low_negative, router)

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
            "contact_rms": zero,
            "interior_rms": zero,
            "visible_ratio": zero,
            "applied_fraction": zero,
        }
        self._frequency_surface_aux_loss = None
        # 14 Aug 2026 - Alternate-base validation does not call Module.eval();
        # gradient mode is the reliable boundary for this training-only loss.
        if (
            not self.frequency_surface_loss_enabled
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return metrics
        supervision = self.ownership_target_mask
        if supervision is None:
            raise RuntimeError("Frequency-surface loss requires an ownership mask")
        batch, length, _ = native_out.shape
        face = self._binary_mask(self.mask, length, batch, torch.float32)
        top = self._binary_mask(supervision, length, batch, torch.float32) * face
        visible = (face - top).clamp(0.0, 1.0)
        eligible = (top.sum(dim=(1, 2)) > 0.0) & (
            visible.sum(dim=(1, 2)) > 0.0
        )
        eligible_float = eligible.float()
        metrics["applied_fraction"] = eligible_float.mean()
        eligible_count = eligible_float.sum().clamp_min(1.0)
        top_high = self._masked_mean_square(high_component, top)
        top_low = self._masked_mean_square(low_component, top)
        routed_rms = self._masked_mean_square(routed_delta, visible).clamp_min(
            1.0e-12
        ).sqrt()
        native_rms = self._masked_mean_square(native_out, visible).clamp_min(
            1.0e-12
        ).sqrt()
        ratio = routed_rms / native_rms.detach().clamp_min(1.0e-6)
        # 16 Aug 2026 - Keep eligibility on-device; the previous Python bool
        # forced a CUDA-to-host synchronization at every CL27 processor.
        top_energy = top_high + self.frequency_surface_top_low_band_factor * top_low
        if self.frequency_surface_region_mode == "contact_partition":
            side = int(math.isqrt(length))
            top_image = top.transpose(1, 2).reshape(batch, 1, side, side)
            width = self.frequency_surface_contact_width
            eroded = 1.0 - F.max_pool2d(
                1.0 - top_image, 2 * width + 1, stride=1, padding=width
            )
            contact = (top_image - eroded).clamp(0.0, 1.0).flatten(2).transpose(1, 2)
            interior = eroded.flatten(2).transpose(1, 2)
            contact_energy = self._masked_mean_square(
                high_component, contact
            ) + self.frequency_surface_top_low_band_factor * self._masked_mean_square(
                low_component, contact
            )
            interior_energy = self._masked_mean_square(
                high_component, interior
            ) + self.frequency_surface_top_low_band_factor * self._masked_mean_square(
                low_component, interior
            )
            factor_sum = (
                self.frequency_surface_contact_factor
                + self.frequency_surface_top_interior_factor
            )
            top_energy = (
                self.frequency_surface_contact_factor * contact_energy
                + self.frequency_surface_top_interior_factor * interior_energy
            ) / max(factor_sum, 1.0e-6)
            metrics["contact_rms"] = (
                contact_energy * eligible_float
            ).sum().div(eligible_count).clamp_min(0.0).sqrt().detach()
            metrics["interior_rms"] = (
                interior_energy * eligible_float
            ).sum().div(eligible_count).clamp_min(0.0).sqrt().detach()
        top_loss = (top_energy * eligible_float).sum() / eligible_count
        floor_loss = (
            F.relu(
                ratio.new_tensor(self.frequency_surface_visible_floor_ratio) - ratio
            ).square()
            * eligible_float
        ).sum() / eligible_count
        self._frequency_surface_aux_loss = (top_loss, floor_loss)
        metrics.update(
            top_high_rms=(top_high * eligible_float).sum().div(eligible_count).sqrt().detach(),
            top_low_rms=(top_low * eligible_float).sum().div(eligible_count).sqrt().detach(),
            visible_ratio=(ratio * eligible_float).sum().div(eligible_count).detach(),
        )
        return metrics

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

    def _visibility_weights(
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
        probabilities = torch.softmax(logits.float(), dim=-1).to(target.dtype)

        face = self._binary_mask(
            self.mask, target.shape[1], target.shape[0], torch.float32
        )
        top = self.ownership_target_mask
        self._ownership_aux_loss = None
        if top is not None:
            top = self._binary_mask(
                top, target.shape[1], target.shape[0], torch.float32
            ) * face
            visible = (face - top).clamp(0.0, 1.0)
            background = (1.0 - face).clamp(0.0, 1.0)
            labels = torch.cat([top, visible, background], dim=-1).argmax(dim=-1)
            counts = torch.bincount(labels.flatten(), minlength=3).float().clamp_min(1.0)
            class_weights = counts.sum() / (3.0 * counts)
            self._ownership_aux_loss = F.cross_entropy(
                logits.float().reshape(-1, 3),
                labels.reshape(-1),
                weight=class_weights.to(logits.device),
            )
        return probabilities

    @staticmethod
    def _gaussian_split(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = delta.shape
        side = int(math.isqrt(length))
        if side * side != length:
            raise RuntimeError("Temporal-frequency BA requires square token grids")
        image = delta.float().transpose(1, 2).reshape(batch, channels, side, side)
        kernel_1d = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 5, 5)
        kernel = kernel.expand(channels, 1, -1, -1)
        low = F.conv2d(image, kernel, padding=2, groups=channels)
        low = low.flatten(2).transpose(1, 2)
        return low.to(delta.dtype), (delta.float() - low).to(delta.dtype)

    def _progress(self, target: torch.Tensor) -> torch.Tensor:
        progress = getattr(self, "ba_denoise_progress", None)
        if progress is None:
            return target.new_zeros(target.shape[0], 1, 1)
        value = torch.as_tensor(progress, device=target.device, dtype=target.dtype)
        value = value.reshape(-1, 1, 1)
        if value.shape[0] == 1:
            value = value.expand(target.shape[0], -1, -1)
        return value.clamp(0.0, 1.0)

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
        if self.hardcase_mode == "anchored_roi":
            gate = self.hardcase_roi_gate_min + (
                self.hardcase_gate_max - self.hardcase_roi_gate_min
            ) * torch.sigmoid(self.roi_gate_raw)
            progress_active = (
                self._progress(target) >= self.hardcase_roi_progress_min
            ).to(target.dtype)
            native_roi = self._sample_roi(target, target_mask)
            ratio = self._masked_rms(native_roi, torch.ones_like(native_roi[..., :1]))
            delta_rms = self._masked_rms(roi_out, torch.ones_like(roi_out[..., :1]))
            cap = (self.hardcase_roi_rms_cap * ratio / delta_rms).clamp(max=1.0)
            scattered = scattered * cap
            active = active.view(batch, 1, 1) * progress_active
            self._latest_ba_telemetry = {
                "anchored_roi_gate": gate.detach().float(),
                "anchored_roi_eligible_fraction": active.detach().float().mean(),
                "anchored_roi_delta_rms": delta_rms.detach().float().mean(),
                "anchored_roi_residual_native_ratio": (
                    gate.detach().float() * delta_rms.detach().float()
                    / ratio.detach().float().clamp_min(1.0e-6)
                ).mean(),
            }
            return scattered * gate * active
        gate = self.hardcase_gate_max * torch.tanh(self.roi_gate_raw)
        return scattered * gate * active.view(batch, 1, 1)

    def _capture_roi_teacher(self, attn, target, reference, reference_out) -> None:
        if not self.roi_teacher_enabled:
            raise RuntimeError("ROI teacher capture enabled on an unselected layer")
        batch, length, _ = target.shape
        target_mask = self._binary_mask(self.mask, length, batch, target.dtype)
        reference_mask = self._binary_mask(
            self.mask_ref, length, batch, reference.dtype
        )
        source_px = (
            1024.0 * target_mask.sum(dim=1).sqrt().squeeze(-1) / math.sqrt(length)
        )
        eligible = (
            (source_px <= float(self.roi_teacher_face_threshold_px))
            & (self._progress(target).flatten() >= self.roi_teacher_progress_min)
        ).float()
        with torch.no_grad():
            target_roi = self._sample_roi(target, target_mask)
            reference_roi = self._sample_roi(reference, reference_mask)
            heads = int(attn.heads)
            teacher_message = F.scaled_dot_product_attention(
                self._reshape_heads(self._q_noise(attn, target_roi), heads),
                self._reshape_heads(self._k_ref(attn, reference_roi), heads),
                self._reshape_heads(self._v_ref(attn, reference_roi), heads),
                dropout_p=0.0,
                is_causal=False,
            )
            teacher_roi = attn.to_out[0](self._merge_heads(teacher_message))
            teacher = self._scatter_roi(teacher_roi, target_mask, length) * target_mask
        student = reference_out * target_mask
        per_sample = F.smooth_l1_loss(student.float(), teacher.float(), reduction="none")
        denom = (target_mask.sum((1, 2)) * student.shape[-1]).clamp_min(1.0)
        smooth = (per_sample * target_mask).sum((1, 2)) / denom
        cosine = F.cosine_similarity(student.float().flatten(1), teacher.float().flatten(1))
        count = eligible.sum().clamp_min(1.0)
        loss = ((smooth + 0.10 * (1.0 - cosine)) * eligible).sum() / count
        self._roi_teacher_aux = (
            loss,
            (cosine * eligible).sum() / count,
            eligible.mean(),
        )

    def _step_ramp(self, start: int, end: int) -> float:
        if end <= start:
            return float(self.ba_training_step >= end)
        return max(0.0, min(1.0, (self.ba_training_step - start) / (end - start)))

    def _null_key_confidence(
        self, attn, q: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached query confidence without retaining an LxL graph."""
        batch, heads, length, width = q.shape
        ref_mask = self._binary_mask(
            self.mask_ref, length, batch, reference.dtype
        )
        keys = self._reshape_heads(
            self._k_ref(attn, reference * ref_mask), heads
        ).detach().float()
        chunks = []
        with torch.no_grad():
            for q_chunk in q.detach().float().split(256, dim=2):
                logits = torch.matmul(q_chunk, keys.transpose(-1, -2)) / math.sqrt(width)
                probability = logits.softmax(dim=-1)
                entropy = -(
                    probability * probability.clamp_min(1.0e-8).log()
                ).sum(dim=-1) / math.log(max(length, 2))
                chunks.append(entropy.mean(dim=1, keepdim=False)[..., None])
            entropy = torch.cat(chunks, dim=1)
            null_mass = torch.sigmoid(
                (entropy - self.null_key_entropy_threshold)
                / self.null_key_temperature
            )
            confidence = (
                1.0 - self.null_key_max_abstention * null_mass
            ).clamp(min=self.null_key_min_reference_fraction, max=1.0)
        return confidence.to(q.dtype), null_mass

    def _landmark_rows(
        self, batch: int, device: torch.device
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        points = self.reference_landmarks_5
        confidence = self.reference_landmark_confidence
        if points is None or confidence is None:
            return None, None
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
        confidence = torch.as_tensor(confidence, device=device, dtype=torch.float32).flatten()
        if points.ndim == 2:
            points = points.unsqueeze(0)
        if points.shape[0] == 1:
            points = points.expand(batch, -1, -1)
        elif batch % points.shape[0] == 0:
            points = points.repeat(batch // points.shape[0], 1, 1)
        if confidence.shape[0] == 1:
            confidence = confidence.expand(batch)
        elif batch % confidence.shape[0] == 0:
            confidence = confidence.repeat(batch // confidence.shape[0])
        if points.shape != (batch, 5, 2) or confidence.shape[0] != batch:
            return None, None
        return points, confidence

    def _canonical_reference_out(
        self,
        attn,
        q: torch.Tensor,
        reference: torch.Tensor,
        original: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels = reference.shape
        side = int(math.isqrt(length))
        points, confidence = self._landmark_rows(batch, reference.device)
        if points is None or side * side != length:
            zero = reference.new_tensor(0.0)
            return original, {"applied": zero, "confidence": zero, "cosine": zero, "rms": zero}
        grid, geometrically_valid = similarity_grid_from_landmarks(points, side=side)
        valid = geometrically_valid & (
            confidence >= self.landmark_canonical_kv_min_confidence
        )
        canonical = F.grid_sample(
            reference.transpose(1, 2).reshape(batch, channels, side, side).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).to(reference.dtype).flatten(2).transpose(1, 2)
        candidate = self._reference_target_out(attn, q, canonical)
        mixed = (1.0 - self.landmark_canonical_kv_mix) * original
        mixed = mixed + self.landmark_canonical_kv_mix * candidate
        ratio = self._masked_rms(original, torch.ones_like(original[..., :1]))
        ratio = ratio / self._masked_rms(mixed, torch.ones_like(mixed[..., :1]))
        mixed = mixed * ratio.to(mixed.dtype)
        valid_mask = valid[:, None, None]
        output = torch.where(valid_mask, mixed, original)
        correction = output.float() - original.float()
        cosine = F.cosine_similarity(
            original.float().flatten(1), candidate.float().flatten(1)
        ).mean()
        return output, {
            "applied": valid.float().mean(),
            "confidence": confidence.mean(),
            "cosine": cosine.detach(),
            "rms": correction.square().mean().sqrt().detach(),
        }

    def _component_memory_correction(
        self,
        attn,
        q: torch.Tensor,
        reference: torch.Tensor,
        routed_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels = reference.shape
        side = int(math.isqrt(length))
        points, confidence = self._landmark_rows(batch, reference.device)
        zero = reference.new_tensor(0.0)
        empty_metrics = {"applied": zero, "rms": zero, "masses": reference.new_zeros(5)}
        if points is None or side * side != length:
            return torch.zeros_like(routed_delta), empty_metrics
        valid = confidence >= self.component_token_memory_min_confidence
        y, x = torch.meshgrid(
            torch.arange(side, device=reference.device, dtype=torch.float32),
            torch.arange(side, device=reference.device, dtype=torch.float32),
            indexing="ij",
        )
        xy = torch.stack((x, y), dim=-1).reshape(1, length, 1, 2)
        centers = torch.stack(
            (points[:, 0], points[:, 1], points[:, 2], points[:, 3:5].mean(1)),
            dim=1,
        ) * float(side - 1)
        distances = (xy - centers[:, None]).square().sum(-1)
        weights = torch.exp(
            -0.5 * distances / (self.component_token_memory_sigma_cells ** 2)
        )
        face = self._binary_mask(self.mask_ref, length, batch, torch.float32)
        weights = weights * face
        global_weight = face
        weights = torch.cat((weights, global_weight), dim=-1)
        mass = weights.sum(dim=1).clamp_min(1.0e-6)
        tokens = torch.einsum("blc,bld->bcd", weights, reference.float()) / mass[..., None]
        component = torch.arange(5, device=reference.device, dtype=torch.float32)
        channel = torch.arange(channels, device=reference.device, dtype=torch.float32)
        type_code = torch.sin((component[:, None] + 1.0) * (channel[None] + 1.0) / channels)
        tokens = (tokens + 0.01 * type_code[None]).to(reference.dtype)
        heads = int(attn.heads)
        message = F.scaled_dot_product_attention(
            q,
            self._reshape_heads(self._k_ref(attn, tokens), heads),
            self._reshape_heads(self._v_ref(attn, tokens), heads),
            dropout_p=0.0,
            is_causal=False,
        )
        part = attn.to_out[0](self._merge_heads(message))
        ratio = self._masked_rms(routed_delta, torch.ones_like(routed_delta[..., :1]))
        ratio = ratio / self._masked_rms(part, torch.ones_like(part[..., :1]))
        correction = self.component_token_memory_scale * part * ratio.to(part.dtype)
        correction = correction * valid[:, None, None].to(correction.dtype)
        attention_mass = mass / mass.sum(dim=-1, keepdim=True)
        return correction, {
            "applied": valid.float().mean(),
            "rms": correction.float().square().mean().sqrt().detach(),
            "masses": attention_mass.mean(dim=0).detach(),
        }

    def _visibility_ownership_loss(
        self, native: torch.Tensor, candidate: torch.Tensor
    ) -> None:
        self._visibility_ownership_v2_aux = None
        if (
            not self.visibility_ownership_v2_enabled
            or not self.training
            or not torch.is_grad_enabled()
            or self.ownership_target_mask is None
        ):
            return
        batch, length, _ = native.shape
        face = self._binary_mask(self.mask, length, batch, torch.float32)
        top = self._binary_mask(
            self.ownership_target_mask, length, batch, torch.float32
        ) * face
        side = int(math.isqrt(length))
        top_image = top.transpose(1, 2).reshape(batch, 1, side, side)
        width = self.visibility_ownership_v2_dilate_cells
        dilated = F.max_pool2d(
            top_image, 2 * width + 1, stride=1, padding=width
        )
        contact = (dilated - top_image).clamp(0.0, 1.0).flatten(2).transpose(1, 2) * face
        top_area = top.mean(dim=(1, 2))
        eligible = top_area >= self.visibility_ownership_v2_min_top_area
        # 20 Aug 2026 - AICODE-NOTE: CL38's candidate shares its native path.
        # Keeping native live in this subtraction cancels native-path gradients
        # algebraically, so this auxiliary can update only the explicit BA delta.
        native_target = (
            native if self.visibility_ownership_v2_delta_only else native.detach()
        )
        difference = (candidate.float() - native_target.float()).abs()

        def region_mean(mask: torch.Tensor) -> torch.Tensor:
            per_sample = (difference * mask).sum((1, 2)) / (
                mask.sum((1, 2)) * difference.shape[-1]
            ).clamp_min(1.0)
            return (per_sample * eligible.float()).sum() / eligible.float().sum().clamp_min(1.0)

        self._visibility_ownership_v2_aux = (
            region_mean(top),
            region_mean(contact),
            top_area.mean().detach(),
            eligible.float().mean().detach(),
        )

    def _call_hardcase(self, attn, hidden_states, temb) -> torch.Tensor:
        residual = hidden_states
        target, reference, input_ndim, spatial = self._normalized_halves(
            attn, hidden_states, temb
        )
        if self.capture_clean_memory:
            self.clean_reference_memory = reference.detach()
            return self._call_legacy(attn, hidden_states, temb=temb)

        mode = self.hardcase_mode
        if mode in {"highres_roi", "anchored_roi", "clean_memory"}:
            batch = target.shape[0]
            if mode == "anchored_roi":
                # 16 Aug 2026 - AICODE-NOTE: CL26 reconstructs the CL19
                # soft-router baseline directly; do not compute and discard a
                # full legacy attention pass first.
                native_out, reference_out, _ = self._full_target_lanes(
                    attn, target, reference
                )
                router = self._soft_router_mask(
                    self.mask, target.shape[1], batch, native_out.dtype
                )
                soft_target = native_out + router * (reference_out - native_out)
                baseline = self._finish_full_router(
                    attn,
                    residual,
                    soft_target,
                    reference,
                    input_ndim,
                    spatial,
                )
            else:
                baseline = self._call_legacy(attn, hidden_states, temb=temb)
            if mode in {"highres_roi", "anchored_roi"}:
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
            # 13 Aug 2026 - AICODE-NOTE: bounded gates are kept in fp32 for
            # stable optimization, but every residual branch must rejoin the
            # frozen UNet in its activation dtype (normally bf16 on Serv).
            addition = addition.to(device=baseline.device, dtype=baseline.dtype)
            if input_ndim == 4:
                channels, height, width = spatial
                addition = addition.transpose(-1, -2).reshape(
                    batch, channels, height, width
                )
            target_out = baseline[:batch] + addition / attn.rescale_output_factor
            return torch.cat([target_out, baseline[batch:]], dim=0)

        native_out, reference_out, q = self._full_target_lanes(attn, target, reference)
        if mode == "visibility_order":
            probabilities = self._visibility_weights(
                target, native_out, reference_out
            ).to(dtype=native_out.dtype)
            base_router = self._soft_router_mask(
                self.mask, target.shape[1], target.shape[0], native_out.dtype
            )
            top_probability = probabilities[..., 0:1]
            visible_probability = probabilities[..., 1:2]
            reference_weight = visible_probability * torch.maximum(
                base_router,
                base_router.new_tensor(self.hardcase_visible_face_floor),
            )
            face = self._binary_mask(
                self.mask, target.shape[1], target.shape[0], native_out.dtype
            )
            reference_weight = (
                reference_weight
                * face
                * (1.0 - self.hardcase_top_native_floor * top_probability)
            )
            reference_weight = reference_weight.clamp(0.0, 1.0)
            target_out = native_out + reference_weight * (reference_out - native_out)
            delta = target_out.float() - (
                native_out.float() * (1.0 - base_router.float())
                + reference_out.float() * base_router.float()
            )
            self._latest_ba_telemetry = {
                "visibility_order_router_delta_rms": delta.square().mean().clamp_min(1e-12).sqrt().detach(),
                "visibility_order_routed_native_ratio": (
                    target_out.float().square().mean().clamp_min(1e-12).sqrt()
                    / native_out.float().square().mean().clamp_min(1e-12).sqrt()
                ).detach(),
                "visibility_order_top_probability": top_probability.detach().float().mean(),
            }
            return self._finish_full_router(
                attn, residual, target_out, reference, input_ndim, spatial
            )

        if mode == "temporal_frequency":
            router = self._soft_router_mask(
                self.mask, target.shape[1], target.shape[0], native_out.dtype
            )
            extension_metrics = {}
            if self.landmark_canonical_kv_enabled:
                reference_out, canonical_metrics = self._canonical_reference_out(
                    attn, q, reference, reference_out
                )
                extension_metrics.update(
                    **{
                        "canonical_kv/applied_fraction": canonical_metrics["applied"],
                        "canonical_kv/landmark_confidence": canonical_metrics["confidence"],
                        "canonical_kv/native_vs_canonical_cosine": canonical_metrics["cosine"],
                        "canonical_kv/correction_rms": canonical_metrics["rms"],
                    }
                )
            low, high = self._gaussian_split(reference_out - native_out)
            progress = self._progress(target)
            if self.frequency_shared_schedule_enabled:
                raw_parameter = self._frequency_shared_schedule_raw
                if raw_parameter is None:
                    raise RuntimeError("Shared frequency schedule was not attached")
                raw = torch.tanh(raw_parameter).to(progress.dtype)
                low_late = (
                    self.frequency_shared_low_late_center
                    + self.frequency_shared_low_late_half_range * raw[0]
                )
                high_early = (
                    self.frequency_shared_high_early_center
                    + self.frequency_shared_high_early_half_range * raw[1]
                )
                high_late = (
                    self.frequency_shared_high_late_center
                    + self.frequency_shared_high_late_half_range * raw[2]
                )
                low_scale = self.hardcase_frequency_low_early + progress * (
                    low_late - self.hardcase_frequency_low_early
                )
                high_scale = high_early + progress * (high_late - high_early)
            elif self.frequency_learnable_schedule_enabled:
                raw = torch.tanh(self.frequency_schedule_raw).to(progress.dtype)
                low_scale = self.hardcase_frequency_low_early + progress * (
                    self.hardcase_frequency_low_late
                    - self.hardcase_frequency_low_early
                )
                high_scale = self.hardcase_frequency_high_early + progress * (
                    self.hardcase_frequency_high_late
                    - self.hardcase_frequency_high_early
                )
                # Zero raw vectors reproduce CL23 exactly; only bounded endpoint
                # corrections are new, and low-early remains fixed at 0.50.
                low_scale = low_scale + progress * (
                    progress.new_tensor(
                        self.frequency_low_late_center
                        - self.hardcase_frequency_low_late
                    )
                    + self.frequency_low_late_half_range * raw[0]
                )
                high_early_correction = (
                    progress.new_tensor(
                        self.frequency_high_early_center
                        - self.hardcase_frequency_high_early
                    )
                    + self.frequency_high_early_half_range * raw[1]
                )
                high_late_correction = (
                    progress.new_tensor(
                        self.frequency_high_late_center
                        - self.hardcase_frequency_high_late
                    )
                    + self.frequency_high_late_half_range * raw[2]
                )
                high_scale = high_scale + high_early_correction + progress * (
                    high_late_correction - high_early_correction
                )
            else:
                low_scale = self.hardcase_frequency_low_early + progress * (
                    self.hardcase_frequency_low_late
                    - self.hardcase_frequency_low_early
                )
                high_scale = self.hardcase_frequency_high_early + progress * (
                    self.hardcase_frequency_high_late
                    - self.hardcase_frequency_high_early
                )
            if self.semantic_window_gate_enabled:
                face = self._binary_mask(
                    self.mask, target.shape[1], target.shape[0], torch.float32
                )
                ref_face = self._binary_mask(
                    self.mask_ref, reference.shape[1], reference.shape[0], torch.float32
                )
                agreement = F.cosine_similarity(
                    self._masked_pool(target, face),
                    self._masked_pool(reference, ref_face),
                    dim=-1,
                ).detach().view(-1, 1, 1)
                rising = torch.sigmoid(
                    (progress - self.semantic_window_progress_start)
                    / self.semantic_window_progress_temperature
                )
                falling = torch.sigmoid(
                    (self.semantic_window_progress_end - progress)
                    / self.semantic_window_progress_temperature
                )
                time_weight = rising * falling
                agreement_weight = torch.sigmoid(
                    (agreement - self.semantic_window_agreement_threshold)
                    / self.semantic_window_agreement_temperature
                )
                window_scale = self.semantic_window_min_scale + (
                    self.semantic_window_max_scale - self.semantic_window_min_scale
                ) * time_weight * agreement_weight
                # 19 Aug 2026 - A per-sample fp32 gate would promote the BA
                # residual and break the following bf16 SDXL LayerNorm.
                window_scale = window_scale.to(dtype=target.dtype)
                high_scale = high_scale * window_scale
                extension_metrics.update(
                    **{
                        "semantic_window/agreement": agreement.mean(),
                        "semantic_window/time_weight": time_weight.mean(),
                        "semantic_window/high_scale": window_scale.mean(),
                        "semantic_window/object_minus_visible_scale": window_scale.new_tensor(0.0),
                    }
                )
            low_component = router * low_scale * low
            high_component = router * high_scale * high
            if self.null_key_router_enabled:
                confidence, null_mass = self._null_key_confidence(
                    attn, q, reference
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
                        - (null_mass * visible).sum() / visible.sum().clamp_min(1.0)
                    )
                extension_metrics.update(
                    **{
                        "null_key/null_mass": null_mass.mean(),
                        "null_key/reference_fraction": confidence.float().mean(),
                        "null_key/object_minus_visible_mass": object_minus_visible,
                    }
                )
            routed_delta = low_component + high_component
            if self.component_token_memory_enabled:
                correction, component_metrics = self._component_memory_correction(
                    attn, q, reference, routed_delta
                )
                routed_delta = routed_delta + correction
                extension_metrics.update(
                    **{
                        "component_memory/applied_fraction": component_metrics["applied"],
                        "component_memory/correction_rms": component_metrics["rms"],
                        "component_memory/left_eye_mass": component_metrics["masses"][0],
                        "component_memory/right_eye_mass": component_metrics["masses"][1],
                        "component_memory/nose_mass": component_metrics["masses"][2],
                        "component_memory/mouth_mass": component_metrics["masses"][3],
                        "component_memory/global_mass": component_metrics["masses"][4],
                    }
                )
            if self.identity_motion_projector is not None:
                face = self._binary_mask(
                    self.mask, target.shape[1], target.shape[0], target.dtype
                )
                ref_face = self._binary_mask(
                    self.mask_ref, reference.shape[1], reference.shape[0], reference.dtype
                )
                correction, cosine_before, cosine_after = self.identity_motion_projector(
                    target * face, reference * ref_face
                )
                correction_rms = self._masked_rms(correction, face)
                routed_rms = self._masked_rms(routed_delta, face)
                correction = correction * (routed_rms / correction_rms).clamp(max=1.0).to(correction.dtype)
                gate = self.identity_motion_projector_gate_max * self._step_ramp(
                    self.identity_motion_projector_ramp_start_step,
                    self.identity_motion_projector_ramp_end_step,
                )
                routed_delta = routed_delta + gate * correction * face
                extension_metrics.update(
                    **{
                        "id_motion/cosine_before": cosine_before.detach(),
                        "id_motion/cosine_after": cosine_after.detach(),
                        "id_motion/correction_rms": correction.float().square().mean().sqrt().detach(),
                        "id_motion/gate": target.new_tensor(gate),
                    }
                )
            if self.id_adaptive_modulation is not None:
                identity = self.identity_embedding_512
                if identity is None:
                    raise RuntimeError("CL43 requires the raw 512-D identity embedding")
                identity = torch.as_tensor(identity, device=target.device)
                if identity.ndim == 3:
                    identity = identity.mean(dim=1)
                if identity.shape[0] == 1:
                    identity = identity.expand(target.shape[0], -1)
                elif target.shape[0] % identity.shape[0] == 0:
                    identity = identity.repeat(target.shape[0] // identity.shape[0], 1)
                if identity.shape != (target.shape[0], 512):
                    raise RuntimeError(
                        f"CL43 identity shape mismatch: {tuple(identity.shape)}"
                    )
                correction, gamma, beta = self.id_adaptive_modulation(
                    routed_delta, identity
                )
                scale = self.id_adaptive_modulation_scale_max * self._step_ramp(
                    self.id_adaptive_modulation_ramp_start_step,
                    self.id_adaptive_modulation_ramp_end_step,
                )
                routed_delta = routed_delta + scale * correction
                extension_metrics.update(
                    **{
                        "id_modulation/gamma_rms": gamma.float().square().mean().sqrt().detach(),
                        "id_modulation/beta_rms": beta.float().square().mean().sqrt().detach(),
                        "id_modulation/output_rms": correction.float().square().mean().sqrt().detach(),
                        "id_modulation/active_fraction": target.new_tensor(float(scale > 0.0)),
                    }
                )
            target_out = native_out + routed_delta
            # 19 Aug 2026 - AICODE-NOTE: all CL38-CL44 arms modify only the
            # explicit CL27 reference delta; target Q and reference K/V ownership stay intact.
            self._visibility_ownership_loss(native_out, target_out)
            self._latest_ba_telemetry = (
                dict(extension_metrics) if extension_metrics else None
            )
            if self.hardcase_telemetry_enabled:
                # 16 Aug 2026 - Full-activation fp32 reductions are optional;
                # objectives and routed activations remain unchanged.
                base_telemetry = {
                    "frequency_low_scale": low_scale.detach().float().mean(),
                    "frequency_high_scale": high_scale.detach().float().mean(),
                    "frequency_low_delta_rms": low.detach().float().square().mean().clamp_min(1e-12).sqrt(),
                    "frequency_high_delta_rms": high.detach().float().square().mean().clamp_min(1e-12).sqrt(),
                    "frequency_merged_native_ratio": (
                        target_out.detach().float().square().mean().clamp_min(1e-12).sqrt()
                        / native_out.detach().float().square().mean().clamp_min(1e-12).sqrt()
                    ),
                }
                if self._latest_ba_telemetry is None:
                    self._latest_ba_telemetry = {}
                self._latest_ba_telemetry.update(base_telemetry)
            if self.frequency_learnable_schedule_enabled and self._latest_ba_telemetry is not None:
                raw_abs = self.frequency_schedule_raw.detach().float().abs()
                self._latest_ba_telemetry.update(
                    frequency_schedule_raw_abs_mean=raw_abs.mean(),
                    frequency_schedule_raw_abs_max=raw_abs.max(),
                )
            if self.frequency_surface_experiment_enabled:
                surface = self._frequency_surface_loss(
                    native_out,
                    low_component,
                    high_component,
                    routed_delta,
                )
                if self._latest_ba_telemetry is None:
                    self._latest_ba_telemetry = {}
                self._latest_ba_telemetry.update(
                    frequency_surface_top_high_rms=surface["top_high_rms"],
                    frequency_surface_top_low_rms=surface["top_low_rms"],
                    frequency_surface_contact_rms=surface["contact_rms"],
                    frequency_surface_interior_rms=surface["interior_rms"],
                    frequency_surface_visible_ratio=surface["visible_ratio"],
                    frequency_surface_applied_fraction=surface["applied_fraction"],
                )
            self._capture_lowband_contrastive(
                attn, reference, q, router, reference_out
            )
            if self._roi_teacher_capture:
                self._capture_roi_teacher(attn, target, reference, reference_out)
            return self._finish_full_router(
                attn, residual, target_out, reference, input_ndim, spatial
            )

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
