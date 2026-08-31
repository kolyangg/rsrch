"""Fail-closed configuration contract for the CL39-X01..X08 wave."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


GROUPS = ("up_blocks.0", "up_blocks.1")

DEFAULTS = {
    "ba_valid_kv_enabled": False,
    "ba_valid_kv_groups": (),
    "ba_valid_kv_threshold": 0.5,
    "ba_valid_kv_entropy_chunk_size": 256,
    "ba_cycle_confidence_enabled": False,
    "ba_cycle_confidence_groups": (),
    "ba_cycle_confidence_floor": 0.25,
    "ba_cycle_confidence_margin_center": 0.04,
    "ba_cycle_confidence_margin_temperature": 0.02,
    "ba_cycle_confidence_cycle_sigma_cells": 1.5,
    "ba_cycle_confidence_chunk_size": 256,
    "ba_cycle_confidence_entropy_weight": 0.25,
    "ba_cycle_confidence_margin_weight": 0.25,
    "ba_cycle_confidence_cycle_weight": 0.50,
    "ba_ot_transport_enabled": False,
    "ba_ot_transport_groups": (),
    "ba_ot_grid_size": 16,
    "ba_ot_epsilon": 0.05,
    "ba_ot_iterations": 20,
    "ba_ot_coordinate_weight": 0.15,
    "ba_ot_transition_start": 0.50,
    "ba_ot_transition_end": 0.70,
    "ba_ot_late_top_k": 4,
    "ba_ot_min_valid_tokens": 8,
    "ba_ot_detach_plan": True,
    "ba_roi_route_enabled": False,
    "ba_roi_route_groups": (),
    "ba_roi_size": 16,
    "ba_roi_face_area_threshold": 0.035,
    "ba_roi_box_expansion": 0.20,
    "ba_roi_gate_max": 0.20,
    "ba_roi_delta_native_cap": 0.35,
    "ba_roi_boundary_ring_cells": 1,
    "ba_automask_os_enabled": False,
    "ba_automask_os_policy_version": "automask_os_v1",
    "ba_automask_reference_hair_weight": 0.35,
    "ba_automask_subject_score_threshold": 0.35,
    "ba_automask_subject_margin_threshold": 0.05,
    "ba_automask_uncertain_native": True,
    "ba_automask_two_pass_validation": True,
    "ba_counterfactual_enabled": False,
    "ba_counterfactual_probability": 0.25,
    "ba_counterfactual_wrong_fraction": 0.50,
    "ba_counterfactual_null_fraction": 0.50,
    "ba_counterfactual_outside_weight": 0.02,
    "ba_counterfactual_rank_weight": 0.02,
    "ba_counterfactual_rank_margin": 0.01,
    "ba_intrinsic_id_sidecar_enabled": False,
    "ba_intrinsic_id_sidecar_groups": (),
    "ba_intrinsic_id_num_tokens": 4,
    "ba_intrinsic_id_token_dim": 2048,
    "ba_intrinsic_id_projector_hidden": 2048,
    "ba_intrinsic_id_residual_rank": 64,
    "ba_intrinsic_id_gate_init": 0.01,
    "ba_intrinsic_id_gate_max": 0.15,
    "ba_intrinsic_id_confidence_source": "none",
    "ba_intrinsic_id_face_router": "hard_face",
    "ba_intrinsic_id_missing_policy": "exact_zero",
    "ba_native_orthogonal_band_enabled": False,
    "ba_native_orthogonal_band_groups": (),
    "ba_native_orthogonal_band": "high",
    "ba_native_orthogonal_mode": "remove_positive_parallel",
    "ba_native_orthogonal_strength": 1.0,
    "ba_native_orthogonal_epsilon": 1.0e-6,
    "ba_native_orthogonal_detach_native": True,
    "ba_global_local_enabled": False,
    "ba_global_local_groups": (),
    "ba_global_dilation_cells": 4,
    "ba_global_early_scale": 0.30,
    "ba_global_late_scale": 0.10,
    "ba_global_native_cap": 0.20,
    "ba_global_local_exclusion": 0.50,
}

ENABLED_KEYS = tuple(key for key in DEFAULTS if key.endswith("_enabled"))


def _plain_mapping(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return {key: value[key] for key in value.keys()}
    except Exception as exc:  # OmegaConf and ordinary mappings are both accepted.
        raise TypeError("cl39x_settings must be a mapping") from exc


def _require_groups(settings: dict, prefix: str) -> None:
    groups = tuple(str(value) for value in settings[f"{prefix}_groups"])
    if groups != GROUPS:
        raise ValueError(f"{prefix}_groups must equal {GROUPS!r}, got {groups!r}")
    settings[f"{prefix}_groups"] = groups


def configure_cl39x(model, supplied) -> dict:
    """Validate settings, expose runtime attributes, and return manifest data."""
    values = _plain_mapping(supplied)
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown CL39-X settings: {unknown}")
    settings = deepcopy(DEFAULTS)
    settings.update(values)
    enabled = [key for key in ENABLED_KEYS if bool(settings[key])]
    if len(enabled) > 1:
        raise ValueError(f"Exactly one CL39-X arm may be enabled, got {enabled}")
    if enabled:
        # 24 Aug 2026 - AICODE-NOTE: every arm is a one-change child of CL39;
        # rejecting a weakened parent prevents an apparently valid but biased run.
        if not (
            getattr(model, "ba_hardcase_mode", "off") == "temporal_frequency"
            and bool(getattr(model, "ba_frequency_surface_loss_enabled", False))
            and bool(getattr(model, "ba_null_key_router_enabled", False))
            and tuple(getattr(model, "ba_null_key_router_groups", ())) == GROUPS
            and float(getattr(model, "pose_adapt_ratio", -1.0)) == 0.0
            and not bool(getattr(model, "ca_mixing_for_face", True))
        ):
            raise ValueError("CL39-X requires the exact CL39 parent contract")

    group_prefix = {
        "ba_valid_kv_enabled": "ba_valid_kv",
        "ba_cycle_confidence_enabled": "ba_cycle_confidence",
        "ba_ot_transport_enabled": "ba_ot_transport",
        "ba_roi_route_enabled": "ba_roi_route",
        "ba_intrinsic_id_sidecar_enabled": "ba_intrinsic_id_sidecar",
        "ba_native_orthogonal_band_enabled": "ba_native_orthogonal_band",
        "ba_global_local_enabled": "ba_global_local",
    }
    for key in enabled:
        prefix = group_prefix.get(key)
        if prefix:
            _require_groups(settings, prefix)

    if not 0.0 <= float(settings["ba_valid_kv_threshold"]) <= 1.0:
        raise ValueError("ba_valid_kv_threshold must be in [0,1]")
    for key in (
        "ba_valid_kv_entropy_chunk_size", "ba_cycle_confidence_chunk_size",
        "ba_ot_grid_size", "ba_ot_iterations", "ba_ot_late_top_k",
        "ba_ot_min_valid_tokens", "ba_roi_size", "ba_intrinsic_id_num_tokens",
        "ba_intrinsic_id_token_dim", "ba_intrinsic_id_projector_hidden",
        "ba_intrinsic_id_residual_rank", "ba_global_dilation_cells",
    ):
        if int(settings[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    weights = [float(settings[key]) for key in (
        "ba_cycle_confidence_entropy_weight",
        "ba_cycle_confidence_margin_weight",
        "ba_cycle_confidence_cycle_weight",
    )]
    if min(weights) < 0.0 or abs(sum(weights) - 1.0) > 1.0e-6:
        raise ValueError("cycle-confidence weights must be nonnegative and sum to one")
    if min(
        float(settings["ba_cycle_confidence_margin_temperature"]),
        float(settings["ba_cycle_confidence_cycle_sigma_cells"]),
        float(settings["ba_ot_epsilon"]),
    ) <= 0.0:
        raise ValueError("confidence/OT temperatures must be positive")
    if not 0.0 <= float(settings["ba_ot_transition_start"]) < float(settings["ba_ot_transition_end"]) <= 1.0:
        raise ValueError("OT transition must satisfy 0 <= start < end <= 1")
    if int(settings["ba_ot_late_top_k"]) > int(settings["ba_ot_grid_size"]) ** 2:
        raise ValueError("OT late_top_k exceeds the transport grid")
    if not 0.0 < float(settings["ba_roi_face_area_threshold"]) < 1.0:
        raise ValueError("ROI face-area threshold must be in (0,1)")
    if not 0.0 < float(settings["ba_roi_gate_max"]) <= 1.0:
        raise ValueError("ROI gate cap must be in (0,1]")
    if abs(
        float(settings["ba_counterfactual_wrong_fraction"])
        + float(settings["ba_counterfactual_null_fraction"]) - 1.0
    ) > 1.0e-6:
        raise ValueError("counterfactual mode fractions must sum to one")
    if not 0.0 < float(settings["ba_counterfactual_probability"]) <= 1.0:
        raise ValueError("counterfactual probability must be in (0,1]")
    if not 0.0 <= float(settings["ba_global_local_exclusion"]) <= 1.0:
        raise ValueError("global local exclusion must be in [0,1]")
    if settings["ba_native_orthogonal_band_enabled"] and not (
        settings["ba_native_orthogonal_band"] == "high"
        and settings["ba_native_orthogonal_mode"] == "remove_positive_parallel"
        and float(settings["ba_native_orthogonal_strength"]) == 1.0
        and float(settings["ba_native_orthogonal_epsilon"]) > 0.0
        and bool(settings["ba_native_orthogonal_detach_native"])
    ):
        raise ValueError("CL39N8 requires the sealed detached high-band projection")
    if settings["ba_intrinsic_id_sidecar_enabled"]:
        historical_x07 = (
            int(settings["ba_intrinsic_id_projector_hidden"]) == 2048
            and int(settings["ba_intrinsic_id_residual_rank"]) == 64
            and float(settings["ba_intrinsic_id_gate_max"]) == 0.15
            and settings["ba_intrinsic_id_confidence_source"] == "none"
            and settings["ba_intrinsic_id_face_router"] == "hard_face"
        )
        cl39n9 = (
            int(settings["ba_intrinsic_id_projector_hidden"]) == 1024
            and int(settings["ba_intrinsic_id_residual_rank"]) == 32
            and float(settings["ba_intrinsic_id_gate_max"]) == 0.10
            and settings["ba_intrinsic_id_confidence_source"]
            == "cl39_complement_detached"
            and settings["ba_intrinsic_id_face_router"] == "cl39_soft_face"
            and settings["ba_intrinsic_id_missing_policy"] == "exact_zero"
        )
        if not (historical_x07 or cl39n9):
            raise ValueError("Intrinsic-ID sidecar must match X07 or sealed CL39N9")

    for key, value in settings.items():
        setattr(model, key, value)
    model.cl39x_settings = settings
    active = enabled[0].removeprefix("ba_").removesuffix("_enabled") if enabled else None
    manifest = {"active_arm": active, "settings": settings} if active else {}
    model._cl39x_manifest = manifest
    return manifest


def cl39x_runtime_attributes() -> tuple[str, ...]:
    return tuple(DEFAULTS)
