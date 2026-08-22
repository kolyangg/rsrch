"""Defaults-off runtime wiring for the CL38--44 attention extension arms."""

from __future__ import annotations


EXTENSION_SPECS = {
    "visibility_ownership_v2": "ba_visibility_ownership_v2",
    "null_key_router": "ba_null_key_router",
    "landmark_canonical_kv": "ba_landmark_canonical_kv",
    "component_token_memory": "ba_component_token_memory",
    "identity_motion_projector": "ba_identity_motion_projector",
    "id_adaptive_modulation": "ba_id_adaptive_modulation",
    "semantic_window_gate": "ba_semantic_window_gate",
}


def resolve_attention_extensions(pipeline, *, hardcase_mode: str):
    """Resolve and validate the independent extension selected by config."""
    enabled = {
        name: bool(getattr(pipeline, f"{prefix}_enabled", False))
        for name, prefix in EXTENSION_SPECS.items()
    }
    groups = {
        name: tuple(
            str(group)
            for group in (getattr(pipeline, f"{prefix}_groups", None) or ())
        )
        for name, prefix in EXTENSION_SPECS.items()
    }
    selected = [name for name, is_enabled in enabled.items() if is_enabled]
    if len(selected) > 1:
        raise RuntimeError(f"CL38-CL44 arms are independent; got {selected}")
    if selected:
        extension = selected[0]
        if not groups[extension]:
            raise RuntimeError(f"{extension} requires non-empty processor groups")
        if hardcase_mode != "temporal_frequency":
            raise RuntimeError(
                f"{extension} requires the CL27 temporal-frequency route"
            )
    return enabled, groups


def attention_extension_kwargs(
    pipeline,
    *,
    processor_name: str,
    enabled: dict[str, bool],
    groups: dict[str, tuple[str, ...]],
) -> dict:
    """Build constructor arguments for the one selected processor extension."""

    def active(name: str) -> bool:
        return enabled[name] and any(
            processor_name.startswith(f"{group}.") for group in groups[name]
        )

    return {
        "visibility_ownership_v2_enabled": active("visibility_ownership_v2"),
        "visibility_ownership_v2_dilate_cells": int(
            getattr(pipeline, "ba_visibility_ownership_v2_dilate_cells", 1)
        ),
        "visibility_ownership_v2_min_top_area": float(
            getattr(pipeline, "ba_visibility_ownership_v2_min_top_area", 0.002)
        ),
        "visibility_ownership_v2_delta_only": bool(
            getattr(pipeline, "ba_visibility_ownership_v2_delta_only", False)
        ),
        "null_key_router_enabled": active("null_key_router"),
        "null_key_entropy_threshold": float(
            getattr(pipeline, "ba_null_key_entropy_threshold", 0.75)
        ),
        "null_key_temperature": float(
            getattr(pipeline, "ba_null_key_temperature", 0.08)
        ),
        "null_key_max_abstention": float(
            getattr(pipeline, "ba_null_key_max_abstention", 0.75)
        ),
        "null_key_min_reference_fraction": float(
            getattr(pipeline, "ba_null_key_min_reference_fraction", 0.25)
        ),
        "landmark_canonical_kv_enabled": active("landmark_canonical_kv"),
        "landmark_canonical_kv_mix": float(
            getattr(pipeline, "ba_landmark_canonical_kv_mix", 0.50)
        ),
        "landmark_canonical_kv_min_confidence": float(
            getattr(pipeline, "ba_landmark_canonical_kv_min_confidence", 0.80)
        ),
        "component_token_memory_enabled": active("component_token_memory"),
        "component_token_memory_scale": float(
            getattr(pipeline, "ba_component_token_memory_scale", 0.15)
        ),
        "component_token_memory_sigma_cells": float(
            getattr(pipeline, "ba_component_token_memory_sigma_cells", 1.75)
        ),
        "component_token_memory_min_confidence": float(
            getattr(pipeline, "ba_component_token_memory_min_confidence", 0.80)
        ),
        "identity_motion_projector_enabled": active("identity_motion_projector"),
        "identity_motion_projector_rank": int(
            getattr(pipeline, "ba_identity_motion_projector_rank", 32)
        ),
        "identity_motion_projector_gate_max": float(
            getattr(pipeline, "ba_identity_motion_projector_gate_max", 0.35)
        ),
        "identity_motion_projector_ramp_start_step": int(
            getattr(pipeline, "ba_identity_motion_projector_ramp_start_step", 1000)
        ),
        "identity_motion_projector_ramp_end_step": int(
            getattr(pipeline, "ba_identity_motion_projector_ramp_end_step", 6000)
        ),
        "id_adaptive_modulation_enabled": active("id_adaptive_modulation"),
        "id_adaptive_modulation_embedding_dim": int(
            getattr(pipeline, "ba_id_adaptive_modulation_embedding_dim", 512)
        ),
        "id_adaptive_modulation_bottleneck": int(
            getattr(pipeline, "ba_id_adaptive_modulation_bottleneck", 32)
        ),
        "id_adaptive_modulation_scale_max": float(
            getattr(pipeline, "ba_id_adaptive_modulation_scale_max", 0.20)
        ),
        "id_adaptive_modulation_ramp_start_step": int(
            getattr(pipeline, "ba_id_adaptive_modulation_ramp_start_step", 1000)
        ),
        "id_adaptive_modulation_ramp_end_step": int(
            getattr(pipeline, "ba_id_adaptive_modulation_ramp_end_step", 6000)
        ),
        "semantic_window_gate_enabled": active("semantic_window_gate"),
        "semantic_window_progress_start": float(
            getattr(pipeline, "ba_semantic_window_gate_progress_start", 0.20)
        ),
        "semantic_window_progress_end": float(
            getattr(pipeline, "ba_semantic_window_gate_progress_end", 0.85)
        ),
        "semantic_window_progress_temperature": float(
            getattr(pipeline, "ba_semantic_window_gate_progress_temperature", 0.08)
        ),
        "semantic_window_agreement_threshold": float(
            getattr(pipeline, "ba_semantic_window_gate_agreement_threshold", 0.15)
        ),
        "semantic_window_agreement_temperature": float(
            getattr(pipeline, "ba_semantic_window_gate_agreement_temperature", 0.08)
        ),
        "semantic_window_min_scale": float(
            getattr(pipeline, "ba_semantic_window_gate_min_scale", 0.60)
        ),
        "semantic_window_max_scale": float(
            getattr(pipeline, "ba_semantic_window_gate_max_scale", 1.15)
        ),
    }
