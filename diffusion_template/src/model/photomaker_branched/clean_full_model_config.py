"""Closed-schema options for the allowlisted clean_full model family.

The public model constructor accepts one mapping instead of hundreds of flat
historical experiment arguments.  This module keeps two concerns explicit:

* nested fields that supported configs may select; and
* fixed compatibility defaults referenced by retained inactive code paths.

Unknown sections or fields fail closed before model weights are loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Defaults formerly declared after num_inference_steps in
# PhotomakerBranchedLora.__init__. Most belong to experiments outside the
# clean_full allowlist; keeping them here prevents inactive compatibility
# branches from growing the supported constructor surface again.
_INTERNAL_DEFAULTS: dict[str, Any] = {
    'skip_unused_text_conditioning': False,
    'conditioning_cache_enabled': False,
    'conditioning_cache_max_entries': 512,
    'batched_conditioning_preparation': False,
    'face_subject_selection_policy': 'legacy_first',
    'cache_prepared_masks': False,
    'compute_branch_debug_outputs': True,
    'strict_branched_install': False,
    'strict_trainable_contract': False,
    'branched_state_dict_mode': 'legacy',
    'ba_architecture_version': 'hard_replace_v1',
    'branched_trainable_dtype': 'inherit',
    'ba_ref_kv_rank': None,
    'ba_output_rank': None,
    'ba_branch_q_rank': 16,
    'ba_face_fusion_mode': 'hard_reference_replace',
    'ba_face_branch_scale': 1.0,
    'ba_gate_init': 0.1,
    'ba_gate_max': 1.0,
    'ba_gate_timestep': True,
    'ba_gate_face_area': True,
    'ba_mix_init': 0.5,
    'ba_mix_floor': 0.0,
    'ba_mix_max': 1.0,
    'ba_mix_timestep': True,
    'ba_mix_face_area': True,
    'ba_reference_rms_match': False,
    'ba_reference_rms_clip_min': 0.5,
    'ba_reference_rms_clip_max': 2.0,
    'ba_mix_override': None,
    'ba_telemetry_enabled': False,
    'ba_telemetry_interval': 50,
    'ba_reference_loss_mode': 'detached_diagnostic',
    'ba_require_denoise_progress': True,
    'ba_self_attention_groups': None,
    'ba_training_timestep_policy': 'uniform_all',
    'ba_spatial_reference_shuffle_probability': 0.0,
    'ba_install_on_device': False,
    'ba_enforce_reference_only_hard_route': False,
    'ba_hard_v1_true_reference_key_mask': False,
    'ba_hard_v1_branch_output_rank': None,
    'ba_hard_v1_reference_roi_warp': False,
    'ba_reference_dropout_probability': 0.0,
    'ba_training_mask_feather': 0,
    'ba_hardcase_mode': 'off',
    'ba_hardcase_groups': None,
    'ba_hardcase_fallback_mode': 'off',
    'ba_hardcase_rank': 64,
    'ba_hardcase_gate_max': 0.2,
    'ba_hardcase_roi_size': 32,
    'ba_hardcase_face_threshold_px': 256,
    'ba_hardcase_transition_cells': 2,
    'ba_hardcase_ownership_hidden_dim': 128,
    'ba_hardcase_visible_face_floor': 0.2,
    'ba_hardcase_top_native_floor': 0.95,
    'ba_hardcase_frequency_low_early': 0.5,
    'ba_hardcase_frequency_low_late': 0.85,
    'ba_hardcase_frequency_high_early': 0.75,
    'ba_hardcase_frequency_high_late': 1.25,
    'ba_hardcase_telemetry_enabled': True,
    'ba_frequency_surface_loss_enabled': False,
    'ba_frequency_surface_loss_groups': None,
    'ba_frequency_surface_top_weight': 0.02,
    'ba_frequency_surface_top_low_band_factor': 0.25,
    'ba_frequency_surface_visible_floor_weight': 0.005,
    'ba_frequency_surface_visible_floor_ratio': 0.35,
    'ba_frequency_learnable_schedule_enabled': False,
    'ba_frequency_learnable_low_early': False,
    'ba_frequency_low_late_center': 0.85,
    'ba_frequency_low_late_half_range': 0.15,
    'ba_frequency_high_early_center': 0.75,
    'ba_frequency_high_early_half_range': 0.15,
    'ba_frequency_high_late_center': 1.25,
    'ba_frequency_high_late_half_range': 0.15,
    'ba_frequency_schedule_anchor_weight': 0.0001,
    'ba_frequency_lowband_contrastive_enabled': False,
    'ba_frequency_lowband_contrastive_groups': None,
    'ba_frequency_lowband_contrastive_probability': 0.125,
    'ba_frequency_lowband_contrastive_weight': 0.02,
    'ba_frequency_lowband_contrastive_temperature': 0.1,
    'ba_frequency_lowband_contrastive_ramp_start_step': 2000,
    'ba_frequency_lowband_contrastive_ramp_end_step': 6000,
    'ba_frequency_lowband_contrastive_detach_target_query': True,
    'ba_frequency_lowband_sample_on_cpu': False,
    'ba_frequency_lowband_contrastive_negative_mode': 'in_batch_different_identity',
    'ba_frequency_positive_sameid_enabled': False,
    'ba_frequency_positive_sameid_groups': None,
    'ba_frequency_positive_sameid_probability': 0.125,
    'ba_frequency_positive_sameid_weight': 0.01,
    'ba_frequency_positive_sameid_ramp_start_step': 2000,
    'ba_frequency_positive_sameid_ramp_end_step': 6000,
    'ba_frequency_positive_sameid_detach_target_query': True,
    'ba_frequency_positive_sameid_stopgrad_anchor': True,
    'ba_frequency_positive_sameid_sample_on_cpu': True,
    'ba_attention_ownership_loss_enabled': False,
    'ba_attention_ownership_groups': None,
    'ba_attention_ownership_probability': 0.25,
    'ba_attention_ownership_weight': 0.02,
    'ba_attention_ownership_visible_ref_mass_floor': 0.55,
    'ba_attention_ownership_top_ref_mass_ceiling': 0.1,
    'ba_attention_ownership_contact_width': 1,
    'ba_attention_ownership_sample_on_cpu': True,
    'ba_frequency_surface_region_mode': 'full_top',
    'ba_frequency_surface_contact_width': 1,
    'ba_frequency_surface_top_interior_factor': 1.0,
    'ba_frequency_surface_contact_factor': 1.0,
    'ba_frequency_surface_normalize_partition_weights': False,
    'ba_frequency_shared_schedule_enabled': False,
    'ba_frequency_shared_low_early_fixed': 0.5,
    'ba_frequency_shared_low_late_center': 0.85,
    'ba_frequency_shared_low_late_half_range': 0.05,
    'ba_frequency_shared_high_early_center': 0.75,
    'ba_frequency_shared_high_early_half_range': 0.05,
    'ba_frequency_shared_high_late_center': 1.25,
    'ba_frequency_shared_high_late_half_range': 0.05,
    'ba_frequency_shared_enforce_monotonic': True,
    'ba_frequency_shared_anchor_weight': 0.001,
    'ba_patch_identity_enabled': False,
    'ba_patch_identity_backend': 'dinov2_vits14',
    'ba_patch_identity_cadence': 16,
    'ba_patch_identity_max_timestep': 200,
    'ba_patch_identity_weight': 0.01,
    'ba_patch_identity_ramp_start_step': 2000,
    'ba_patch_identity_ramp_end_step': 6000,
    'ba_patch_identity_min_gate_mass': 0.55,
    'ba_patch_identity_max_samples_per_step': 1,
    'ba_roi_teacher_distill_enabled': False,
    'ba_roi_teacher_distill_groups': None,
    'ba_roi_teacher_size': 32,
    'ba_roi_teacher_face_threshold_px': 256,
    'ba_roi_teacher_progress_min': 0.6,
    'ba_roi_teacher_probability': 0.125,
    'ba_roi_teacher_weight': 0.02,
    'ba_roi_teacher_stopgrad': True,
    'ba_roi_teacher_sample_on_cpu': True,
    'ba_hardcase_roi_gate_init': 0.1,
    'ba_hardcase_roi_gate_min': 0.05,
    'ba_hardcase_roi_progress_min': 0.6,
    'ba_hardcase_roi_rms_cap': 0.25,
    'ba_visibility_ownership_v2_enabled': False,
    'ba_visibility_ownership_v2_groups': None,
    'ba_visibility_ownership_v2_top_native_weight': 0.02,
    'ba_visibility_ownership_v2_contact_native_weight': 0.01,
    'ba_visibility_ownership_v2_dilate_cells': 1,
    'ba_visibility_ownership_v2_min_top_area': 0.002,
    'ba_visibility_ownership_v2_stopgrad_native': True,
    'ba_visibility_ownership_v2_delta_only': False,
    'ba_visibility_ownership_v2_ramp_start_step': 1000,
    'ba_visibility_ownership_v2_ramp_end_step': 4000,
    'ba_null_key_router_enabled': False,
    'ba_null_key_router_groups': None,
    'ba_null_key_entropy_threshold': 0.75,
    'ba_null_key_temperature': 0.08,
    'ba_null_key_max_abstention': 0.75,
    'ba_null_key_min_reference_fraction': 0.25,
    'ba_landmark_canonical_kv_enabled': False,
    'ba_landmark_canonical_kv_groups': None,
    'ba_landmark_canonical_kv_mix': 0.5,
    'ba_landmark_canonical_kv_min_confidence': 0.8,
    'ba_component_token_memory_enabled': False,
    'ba_component_token_memory_groups': None,
    'ba_component_token_memory_scale': 0.15,
    'ba_component_token_memory_sigma_cells': 1.75,
    'ba_component_token_memory_min_confidence': 0.8,
    'ba_identity_motion_projector_enabled': False,
    'ba_identity_motion_projector_groups': None,
    'ba_identity_motion_projector_rank': 32,
    'ba_identity_motion_projector_gate_max': 0.35,
    'ba_identity_motion_projector_ramp_start_step': 1000,
    'ba_identity_motion_projector_ramp_end_step': 6000,
    'ba_id_adaptive_modulation_enabled': False,
    'ba_id_adaptive_modulation_groups': None,
    'ba_id_adaptive_modulation_embedding_dim': 512,
    'ba_id_adaptive_modulation_bottleneck': 32,
    'ba_id_adaptive_modulation_scale_max': 0.2,
    'ba_id_adaptive_modulation_ramp_start_step': 1000,
    'ba_id_adaptive_modulation_ramp_end_step': 6000,
    'ba_semantic_window_gate_enabled': False,
    'ba_semantic_window_gate_groups': None,
    'ba_semantic_window_gate_progress_start': 0.2,
    'ba_semantic_window_gate_progress_end': 0.85,
    'ba_semantic_window_gate_progress_temperature': 0.08,
    'ba_semantic_window_gate_agreement_threshold': 0.15,
    'ba_semantic_window_gate_agreement_temperature': 0.08,
    'ba_semantic_window_gate_min_scale': 0.6,
    'ba_semantic_window_gate_max_scale': 1.15,
    'ba_semantic_ownership_loss_weight': 0.05,
    'ba_crossview_consistency_enabled': False,
    'ba_crossview_consistency_probability': 0.25,
    'ba_crossview_consistency_weight': 0.05,
    'generic_adapter_train_scope': 'none',
    'photomaker_default_train_scope': 'none',
    'ba_hard_v1_lora_rank': None,
    'ba_identity_ca_v2_enabled': False,
    'ba_identity_ca_v2_groups': None,
    'ba_identity_ca_v2_rank': 16,
    'ba_residual_identity_ca_v3_enabled': False,
    'ba_residual_identity_ca_v3_groups': None,
    'ba_residual_identity_ca_v3_rank': 64,
    'ba_residual_identity_ca_v3_gate_init': 0.02,
    'ba_residual_identity_ca_v3_gate_max': 0.2,
    'identity_aux_enabled': False,
    'identity_aux_cadence': 4,
    'identity_aux_max_timestep': 400,
    'identity_aux_ramp_start_step': 2000,
    'identity_aux_ramp_end_step': 6000,
    'identity_aux_max_weight': 0.05,
    'identity_aux_crop_padding': 0.25,
    'identity_aux_backend': 'photomaker_clip_v1',
    'identity_aux_model_path': None,
    'identity_aux_model_sha256': None,
    'identity_aux_dynamic_weight': False,
    'identity_aux_grad_target_ratio': 0.075,
    'identity_aux_grad_norm_interval': 200,
    'identity_aux_mode': 'cosine',
    'identity_aux_hinge_margin': 0.55,
    'identity_aux_gradient_scope': 'all_trainable',
    'ba_pm_boundary_distill_enabled': False,
    'ba_pm_boundary_distill_probability': 0.25,
    'ba_pm_boundary_distill_weight': 0.05,
    'ba_pm_boundary_distill_top_weight': 0.02,
    'ba_pm_boundary_distill_width': 2,
    'ba_low_noise_id_reward_enabled': False,
    'ba_low_noise_id_reward_last_steps': 4,
    'ba_low_noise_id_reward_kl_weight': 1.0,
    'ba_allow_objective_only_checkpoint_init': False,
}


_RUNTIME_FIELDS = {
    "skip_unused_text_conditioning": "skip_unused_text_conditioning",
    "conditioning_cache_enabled": "conditioning_cache_enabled",
    "conditioning_cache_max_entries": "conditioning_cache_max_entries",
    "batched_conditioning_preparation": "batched_conditioning_preparation",
    "face_subject_selection_policy": "face_subject_selection_policy",
    "cache_prepared_masks": "cache_prepared_masks",
    "compute_branch_debug_outputs": "compute_branch_debug_outputs",
}
_CONTRACT_FIELDS = {
    "strict_branched_install": "strict_branched_install",
    "strict_trainable_contract": "strict_trainable_contract",
    "state_dict_mode": "branched_state_dict_mode",
    "generic_adapter_train_scope": "generic_adapter_train_scope",
    "photomaker_default_train_scope": "photomaker_default_train_scope",
    "hard_v1_lora_rank": "ba_hard_v1_lora_rank",
}
_ARCHITECTURE_FIELDS = {
    "version": "ba_architecture_version",
    "trainable_dtype": "branched_trainable_dtype",
    "training_timestep_policy": "ba_training_timestep_policy",
    "self_attention_groups": "ba_self_attention_groups",
    "enforce_reference_only_hard_route": "ba_enforce_reference_only_hard_route",
    "true_reference_key_mask": "ba_hard_v1_true_reference_key_mask",
    "branch_output_rank": "ba_hard_v1_branch_output_rank",
    "reference_roi_warp": "ba_hard_v1_reference_roi_warp",
}
_HARDCASE_FIELDS = {
    "mode": "ba_hardcase_mode",
    "groups": "ba_hardcase_groups",
    "fallback_mode": "ba_hardcase_fallback_mode",
    "rank": "ba_hardcase_rank",
    "gate_max": "ba_hardcase_gate_max",
    "roi_size": "ba_hardcase_roi_size",
    "face_threshold_px": "ba_hardcase_face_threshold_px",
    "transition_cells": "ba_hardcase_transition_cells",
    "ownership_hidden_dim": "ba_hardcase_ownership_hidden_dim",
    "visible_face_floor": "ba_hardcase_visible_face_floor",
    "top_native_floor": "ba_hardcase_top_native_floor",
    "frequency_low_early": "ba_hardcase_frequency_low_early",
    "frequency_low_late": "ba_hardcase_frequency_low_late",
    "frequency_high_early": "ba_hardcase_frequency_high_early",
    "frequency_high_late": "ba_hardcase_frequency_high_late",
    "telemetry_enabled": "ba_hardcase_telemetry_enabled",
}
_FREQUENCY_SURFACE_FIELDS = {
    "enabled": "ba_frequency_surface_loss_enabled",
    "groups": "ba_frequency_surface_loss_groups",
    "top_weight": "ba_frequency_surface_top_weight",
    "top_low_band_factor": "ba_frequency_surface_top_low_band_factor",
    "visible_floor_weight": "ba_frequency_surface_visible_floor_weight",
    "visible_floor_ratio": "ba_frequency_surface_visible_floor_ratio",
}
_RECENT_EXTENSION_KINDS = {
    "null_key_router",
    "identity_motion_projector",
    "landmark_canonical_kv",
    "component_token_memory",
    "id_adaptive_modulation",
    "semantic_window_gate",
}
_OPTIONAL_GROUP_FIELDS = {
    "ba_self_attention_groups",
    "ba_hardcase_groups",
    "ba_identity_ca_v2_groups",
    "ba_residual_identity_ca_v3_groups",
}
_LOWERCASE_FIELDS = {
    "branched_state_dict_mode",
    "generic_adapter_train_scope",
    "photomaker_default_train_scope",
    "ba_architecture_version",
    "branched_trainable_dtype",
    "ba_training_timestep_policy",
    "face_subject_selection_policy",
    "ba_hardcase_mode",
    "ba_hardcase_fallback_mode",
    "identity_aux_backend",
    "identity_aux_mode",
    "identity_aux_gradient_scope",
    "ba_frequency_surface_region_mode",
    "ba_frequency_lowband_contrastive_negative_mode",
    "ba_patch_identity_backend",
}


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _overlay_section(
    values: dict[str, Any],
    raw: dict[str, Any],
    *,
    section: str,
    fields: dict[str, str],
) -> None:
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"Unknown clean_full_config.{section} fields: {sorted(unknown)}")
    for public_name, value in raw.items():
        values[fields[public_name]] = value


def resolve_clean_full_model_config(raw_config: Any) -> dict[str, Any]:
    """Resolve the small supported schema into the historical flat attributes."""
    raw = _plain_mapping(raw_config, name="clean_full_config")
    allowed_top = {
        "runtime",
        "contract",
        "architecture",
        "training_mask_feather",
        "hardcase",
        "frequency_surface",
        "recent_extension",
    }
    unknown = set(raw) - allowed_top
    if unknown:
        raise ValueError(f"Unknown clean_full_config sections: {sorted(unknown)}")

    values = dict(_INTERNAL_DEFAULTS)
    for section, fields in (
        ("runtime", _RUNTIME_FIELDS),
        ("contract", _CONTRACT_FIELDS),
        ("architecture", _ARCHITECTURE_FIELDS),
        ("hardcase", _HARDCASE_FIELDS),
        ("frequency_surface", _FREQUENCY_SURFACE_FIELDS),
    ):
        _overlay_section(
            values,
            _plain_mapping(raw.get(section), name=f"clean_full_config.{section}"),
            section=section,
            fields=fields,
        )

    if "training_mask_feather" in raw:
        values["ba_training_mask_feather"] = raw["training_mask_feather"]

    recent = _plain_mapping(
        raw.get("recent_extension"), name="clean_full_config.recent_extension"
    )
    kind = str(recent.pop("kind", "none")).lower()
    groups = recent.pop("groups", None)
    if kind == "none":
        if groups not in (None, [], ()) or recent:
            raise ValueError("recent_extension kind=none cannot define groups/settings")
    elif kind in _RECENT_EXTENSION_KINDS:
        setting_prefix = "ba_null_key_" if kind == "null_key_router" else f"ba_{kind}_"
        values[f"ba_{kind}_enabled"] = True
        values[f"ba_{kind}_groups"] = groups
        for setting, value in recent.items():
            flat_name = f"{setting_prefix}{setting}"
            if flat_name not in _INTERNAL_DEFAULTS:
                raise ValueError(
                    f"Unknown {kind} setting {setting!r} in clean_full_config"
                )
            values[flat_name] = value
    else:
        raise ValueError(f"Unknown recent_extension kind: {kind!r}")

    return values


def apply_clean_full_model_config(model, raw_config: Any) -> None:
    """Install and validate options used by the allowlisted hard-v1 family."""
    values = resolve_clean_full_model_config(raw_config)
    values["conditioning_cache_max_entries"] = max(
        0, int(values["conditioning_cache_max_entries"])
    )
    values["ba_ref_kv_rank"] = int(
        values["ba_ref_kv_rank"]
        if values["ba_ref_kv_rank"] is not None
        else model.lora_rank
    )
    values["ba_output_rank"] = int(
        values["ba_output_rank"]
        if values["ba_output_rank"] is not None
        else model.lora_rank
    )
    values["ba_branch_q_rank"] = int(values["ba_branch_q_rank"])
    for name, value in tuple(values.items()):
        if name in _LOWERCASE_FIELDS and value is not None:
            value = str(value).lower()
        elif name in _OPTIONAL_GROUP_FIELDS:
            value = None if value is None else tuple(str(item) for item in value)
        elif name.endswith("_groups"):
            value = tuple(str(item) for item in (value or ()))
        values[name] = value
        setattr(model, name, value)

    model._ba_lowband_capture_mode = "off"
    model._ba_lowband_negative_permutation = None
    model._ba_attention_ownership_capture = False
    model.ba_patch_identity_encoder = None
    model._ba_roi_teacher_capture = False
    model.identity_aux_recognizer = None
    model._ba_frozen_teacher_unet = None
    model._ba_frozen_teacher_original_processors = None

    if model.ba_architecture_version != "hard_replace_v1":
        raise ValueError("clean_full supports hard_replace_v1 only")
    if not all(
        (
            model.train_ba_only,
            model.strict_branched_install,
            model.strict_trainable_contract,
            model.branched_state_dict_mode == "trainable_v2",
            model.branched_attn_weight_mode == "noise_and_ref",
            model.branched_attn_new_weight_kind == "lora",
            not model.train_branched_ca_lora,
            model.pose_adapt_ratio == 0.0,
            not model.ca_mixing_for_face,
            model.ba_enforce_reference_only_hard_route,
            model.ba_hard_v1_lora_rank is not None,
            model.ba_hard_v1_lora_rank > 0,
        )
    ):
        raise ValueError("clean_full hard-v1 ownership/runtime contract is incomplete")
    if model.generic_adapter_train_scope != "effective_all":
        raise ValueError("clean_full requires generic_adapter_train_scope=effective_all")
    if model.photomaker_default_train_scope != "effective_all":
        raise ValueError("clean_full requires photomaker_default_train_scope=effective_all")
    if not 0 <= int(model.ba_training_mask_feather) <= 8:
        raise ValueError("training_mask_feather must be in [0, 8]")
    if model.ba_training_timestep_policy != "uniform_all":
        raise ValueError("clean_full requires uniform_all training timesteps")
    if model.ba_spatial_reference_shuffle_probability != 0.0:
        raise ValueError("clean_full forbids spatial-reference shuffling")
    if model.ba_hardcase_mode not in {"off", "soft_router", "temporal_frequency"}:
        raise ValueError(f"Unsupported clean_full hardcase mode {model.ba_hardcase_mode!r}")
    hardcase_groups = set(model.ba_hardcase_groups or ())
    if model.ba_hardcase_mode != "off" and not hardcase_groups:
        raise ValueError("Selected hardcase route requires processor groups")
    if model.ba_frequency_surface_loss_enabled:
        if not (
            model.ba_hardcase_mode == "temporal_frequency"
            and model.ba_frequency_surface_loss_groups
            and set(model.ba_frequency_surface_loss_groups) <= hardcase_groups
            and model.ba_frequency_surface_top_weight > 0.0
            and 0.0 <= model.ba_frequency_surface_top_low_band_factor <= 1.0
            and model.ba_frequency_surface_visible_floor_weight > 0.0
            and 0.0 < model.ba_frequency_surface_visible_floor_ratio < 1.0
        ):
            raise ValueError("Invalid clean_full frequency-surface configuration")

    enabled_recent = [
        kind
        for kind in sorted(_RECENT_EXTENSION_KINDS)
        if bool(getattr(model, f"ba_{kind}_enabled"))
    ]
    if len(enabled_recent) > 1:
        raise ValueError(f"Recent clean_full extensions are exclusive: {enabled_recent}")
    if enabled_recent:
        kind = enabled_recent[0]
        groups = set(getattr(model, f"ba_{kind}_groups"))
        if not (
            model.ba_hardcase_mode == "temporal_frequency"
            and model.ba_frequency_surface_loss_enabled
            and groups
            and groups <= hardcase_groups
        ):
            raise ValueError(
                f"{kind} requires the CL27 temporal-frequency/surface substrate"
            )

    if model.ba_identity_motion_projector_enabled and not (
        model.ba_identity_motion_projector_rank > 0
        and 0.0 < model.ba_identity_motion_projector_gate_max <= 1.0
        and 0 <= model.ba_identity_motion_projector_ramp_start_step
        < model.ba_identity_motion_projector_ramp_end_step
    ):
        raise ValueError("Invalid identity-motion projector configuration")
    if model.ba_id_adaptive_modulation_enabled and not (
        model.ba_id_adaptive_modulation_embedding_dim > 0
        and model.ba_id_adaptive_modulation_bottleneck > 0
        and 0.0 < model.ba_id_adaptive_modulation_scale_max <= 1.0
        and 0 <= model.ba_id_adaptive_modulation_ramp_start_step
        < model.ba_id_adaptive_modulation_ramp_end_step
    ):
        raise ValueError("Invalid ID-adaptive modulation configuration")
    if model.ba_null_key_router_enabled and not (
        0.0 <= model.ba_null_key_entropy_threshold <= 1.0
        and model.ba_null_key_temperature > 0.0
        and 0.0 <= model.ba_null_key_max_abstention <= 1.0
        and 0.0 <= model.ba_null_key_min_reference_fraction <= 1.0
    ):
        raise ValueError("Invalid null-key router configuration")
    if model.ba_landmark_canonical_kv_enabled and not (
        0.0 <= model.ba_landmark_canonical_kv_mix <= 1.0
        and 0.0 <= model.ba_landmark_canonical_kv_min_confidence <= 1.0
    ):
        raise ValueError("Invalid landmark canonical-K/V configuration")
    if model.ba_component_token_memory_enabled and not (
        model.ba_component_token_memory_scale >= 0.0
        and model.ba_component_token_memory_sigma_cells > 0.0
        and 0.0 <= model.ba_component_token_memory_min_confidence <= 1.0
    ):
        raise ValueError("Invalid component-token memory configuration")
    if model.ba_semantic_window_gate_enabled and not (
        0.0 <= model.ba_semantic_window_gate_progress_start
        < model.ba_semantic_window_gate_progress_end
        <= 1.0
        and model.ba_semantic_window_gate_progress_temperature > 0.0
        and model.ba_semantic_window_gate_agreement_temperature > 0.0
        and 0.0 <= model.ba_semantic_window_gate_min_scale
        <= model.ba_semantic_window_gate_max_scale
    ):
        raise ValueError("Invalid semantic-window gate configuration")


PUBLIC_SECTIONS = (
    "runtime",
    "contract",
    "architecture",
    "training_mask_feather",
    "hardcase",
    "frequency_surface",
    "recent_extension",
)
INTERNAL_DEFAULT_COUNT = len(_INTERNAL_DEFAULTS)
