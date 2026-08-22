"""Ownership, settings, and checkpoint contract for the selected E13 family."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import torch


SCHEMA_VERSION = 2
STATE_FORMAT = "trainable_unet_v2"
ARCHITECTURE = "hard_replace_v1"

EXPECTED_ROLE_SUMMARIES = {
    "branched_sa_r128": {"tensors": 840, "parameters": 127_795_200},
    "generic_effective_adapter_r32": {
        "tensors": 700,
        "parameters": 30_474_240,
    },
    "photomaker_default_effective_adapter_r64": {
        "tensors": 700,
        "parameters": 60_948_480,
    },
}
IDENTITY_CA_SUMMARY = {"tensors": 108, "parameters": 5_406_756}

DEFAULT_E13_SETTINGS = {
    "ba_training_mask_feather": 0,
    "ba_hardcase_mode": "off",
    "ba_hardcase_groups": (),
    "ba_hardcase_transition_cells": 2,
    "ba_hardcase_frequency_low_early": 0.50,
    "ba_hardcase_frequency_low_late": 0.85,
    "ba_hardcase_frequency_high_early": 0.75,
    "ba_hardcase_frequency_high_late": 1.25,
    "ba_frequency_surface_loss_enabled": False,
    "ba_frequency_surface_loss_groups": (),
    "ba_frequency_surface_top_weight": 0.02,
    "ba_frequency_surface_top_low_band_factor": 0.25,
    "ba_frequency_surface_visible_floor_weight": 0.005,
    "ba_frequency_surface_visible_floor_ratio": 0.35,
    "ba_null_key_router_enabled": False,
    "ba_null_key_router_groups": (),
    "ba_null_key_entropy_threshold": 0.75,
    "ba_null_key_temperature": 0.08,
    "ba_null_key_max_abstention": 0.75,
    "ba_null_key_min_reference_fraction": 0.25,
    "ba_crossview_consistency_enabled": False,
    "ba_crossview_consistency_probability": 0.25,
    "ba_crossview_consistency_weight": 0.05,
    "ba_residual_identity_ca_v3_enabled": False,
    "ba_residual_identity_ca_v3_groups": (),
    "ba_residual_identity_ca_v3_rank": 64,
    "ba_residual_identity_ca_v3_gate_init": 0.02,
    "ba_residual_identity_ca_v3_gate_max": 0.20,
}

PIPELINE_RUNTIME_SETTINGS = (
    "ba_hardcase_mode",
    "ba_hardcase_groups",
    "ba_hardcase_transition_cells",
    "ba_hardcase_frequency_low_early",
    "ba_hardcase_frequency_low_late",
    "ba_hardcase_frequency_high_early",
    "ba_hardcase_frequency_high_late",
    "ba_frequency_surface_loss_enabled",
    "ba_frequency_surface_loss_groups",
    "ba_frequency_surface_top_low_band_factor",
    "ba_frequency_surface_visible_floor_ratio",
    "ba_null_key_router_enabled",
    "ba_null_key_router_groups",
    "ba_null_key_entropy_threshold",
    "ba_null_key_temperature",
    "ba_null_key_max_abstention",
    "ba_null_key_min_reference_fraction",
    "ba_residual_identity_ca_v3_enabled",
    "ba_residual_identity_ca_v3_groups",
    "ba_residual_identity_ca_v3_rank",
    "ba_residual_identity_ca_v3_gate_init",
    "ba_residual_identity_ca_v3_gate_max",
)

_BOOL_SETTINGS = {
    "ba_frequency_surface_loss_enabled",
    "ba_null_key_router_enabled",
    "ba_crossview_consistency_enabled",
    "ba_residual_identity_ca_v3_enabled",
}
_INT_SETTINGS = {
    "ba_training_mask_feather",
    "ba_hardcase_transition_cells",
    "ba_residual_identity_ca_v3_rank",
}
_GROUP_SETTINGS = {
    "ba_hardcase_groups",
    "ba_frequency_surface_loss_groups",
    "ba_null_key_router_groups",
    "ba_residual_identity_ca_v3_groups",
}


def normalise_e13_settings(settings: Mapping | None) -> dict:
    supplied = dict(settings or {})
    unknown = sorted(set(supplied) - set(DEFAULT_E13_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown E13 settings: {unknown}")
    values = {**DEFAULT_E13_SETTINGS, **supplied}
    for name in _BOOL_SETTINGS:
        values[name] = bool(values[name])
    for name in _INT_SETTINGS:
        values[name] = int(values[name])
    for name in _GROUP_SETTINGS:
        values[name] = tuple(str(group) for group in (values[name] or ()))
    for name in set(values) - _BOOL_SETTINGS - _INT_SETTINGS - _GROUP_SETTINGS - {
        "ba_hardcase_mode"
    }:
        values[name] = float(values[name])
    values["ba_hardcase_mode"] = str(values["ba_hardcase_mode"] or "off").lower()
    return values


def initialise_e13_contract(model, settings: Mapping | None = None) -> None:
    """Validate and persist the selected leaf deltas over fixed E13."""
    values = normalise_e13_settings(settings)
    if not 0 <= values["ba_training_mask_feather"] <= 8:
        raise ValueError("ba_training_mask_feather must be within [0, 8]")
    hardcase_mode = values["ba_hardcase_mode"]
    hardcase_groups = values["ba_hardcase_groups"]
    if hardcase_mode not in {"off", "soft_router", "temporal_frequency"}:
        raise ValueError("The clean extension supports CL19 or CL23 routing only")
    if hardcase_mode != "off" and not hardcase_groups:
        raise ValueError("CL19 soft_router requires explicit U-Net groups")
    if values["ba_hardcase_transition_cells"] < 1:
        raise ValueError("ba_hardcase_transition_cells must be positive")
    frequency_values = (
        values["ba_hardcase_frequency_low_early"],
        values["ba_hardcase_frequency_low_late"],
        values["ba_hardcase_frequency_high_early"],
        values["ba_hardcase_frequency_high_late"],
    )
    # 18 Aug 2026 - CL23/CL27 are exact clean leaves, not a general frequency
    # experiment framework; reject schedule or objective drift at construction.
    if hardcase_mode == "temporal_frequency" and frequency_values != (
        0.50, 0.85, 0.75, 1.25
    ):
        raise ValueError("CL23 fixed temporal-frequency schedule drifted")
    surface_groups = values["ba_frequency_surface_loss_groups"]
    if values["ba_frequency_surface_loss_enabled"] and (
        hardcase_mode != "temporal_frequency"
        or surface_groups != ("up_blocks.0", "up_blocks.1")
        or values["ba_frequency_surface_top_weight"] != 0.02
        or values["ba_frequency_surface_top_low_band_factor"] != 0.25
        or values["ba_frequency_surface_visible_floor_weight"] != 0.005
        or values["ba_frequency_surface_visible_floor_ratio"] != 0.35
    ):
        raise ValueError("CL27 frequency-surface objective contract drifted")
    null_key_groups = values["ba_null_key_router_groups"]
    if values["ba_null_key_router_enabled"] and (
        hardcase_mode != "temporal_frequency"
        or not values["ba_frequency_surface_loss_enabled"]
        or null_key_groups != ("up_blocks.0", "up_blocks.1")
        or values["ba_null_key_entropy_threshold"] != 0.75
        or values["ba_null_key_temperature"] != 0.08
        or values["ba_null_key_max_abstention"] != 0.75
        or values["ba_null_key_min_reference_fraction"] != 0.25
    ):
        raise ValueError("CL39 null-key router contract drifted")
    crossview_probability = values["ba_crossview_consistency_probability"]
    crossview_weight = values["ba_crossview_consistency_weight"]
    if values["ba_crossview_consistency_enabled"] and not (
        0.0 < crossview_probability <= 1.0 and crossview_weight > 0.0
    ):
        raise ValueError("CL18 cross-view probability/weight must be positive")
    identity_groups = values["ba_residual_identity_ca_v3_groups"]
    if values["ba_residual_identity_ca_v3_enabled"] and (
        identity_groups != ("up_blocks.0", "up_blocks.1")
        or values["ba_residual_identity_ca_v3_rank"] != 64
        or values["ba_residual_identity_ca_v3_gate_init"] != 0.02
        or values["ba_residual_identity_ca_v3_gate_max"] != 0.20
        or hardcase_mode != "off"
        or values["ba_crossview_consistency_enabled"]
    ):
        raise ValueError("CL14_CA residual identity-CA contract drifted")

    model.ba_architecture_version = ARCHITECTURE
    model.branched_attn_lora_rank = int(model.lora_rank)
    model._e13_settings = values
    for name, value in values.items():
        setattr(model, name, value)


def copy_pipeline_runtime_settings(model, pipeline) -> None:
    """Copy the output-affecting leaf settings to validation exactly once."""
    for name in PIPELINE_RUNTIME_SETTINGS:
        setattr(pipeline, name, getattr(model, name))


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _processor_prefixes(model) -> tuple[str, ...]:
    names = tuple(getattr(model, "_ba_patched_processor_names", ()))
    selected = tuple(name for name in names if name.endswith("attn1.processor"))
    if not selected:
        raise RuntimeError("E13 processor installation selected zero self-attention blocks")
    return tuple(f"{name}." for name in selected)


def trainable_role(
    name: str,
    processor_prefixes: tuple[str, ...],
    identity_prefixes: tuple[str, ...],
) -> str | None:
    """Map a U-Net tensor to one and only one E13 optimizer role."""
    if name.startswith(identity_prefixes) and (
        ".attn2.processor.id_delta_out." in name
        or name.endswith(".attn2.processor.gate_logit")
    ):
        return "residual_identity_ca_r64"
    if not _is_lora_parameter(name):
        return None
    in_processor = name.startswith(processor_prefixes)
    if in_processor and (
        ".attn1.processor.ref_to_" in name
        or ".attn1.processor.noise_to_" in name
    ):
        return "branched_sa_r128"
    # 10 Aug 2026 - E13C-CORE-03: hard-v1 bypasses generic SA Q/K/V. The only
    # effective outer sites are ordinary cross-attention and post-merge SA
    # output; this exact allowlist reproduces the historical 840/700/700 split.
    effective_outer = ".attn2." in name or (
        ".attn1." in name and ".to_out.0." in name
    )
    if effective_outer and ".lora_adapter." in name:
        return "generic_effective_adapter_r32"
    if effective_outer and ".default." in name:
        return "photomaker_default_effective_adapter_r64"
    return None


def expected_trainable_names(model) -> tuple[str, ...]:
    prefixes = _processor_prefixes(model)
    identity_prefixes = tuple(
        f"{name}." for name in getattr(model, "_ba_identity_ca_processor_names", ())
    )
    return tuple(sorted(
        name for name, _ in model.unet.named_parameters()
        if trainable_role(name, prefixes, identity_prefixes) is not None
    ))


def configure_trainables(model) -> None:
    expected = set(expected_trainable_names(model))
    if not expected:
        raise RuntimeError("E13 trainable allowlist is empty")
    for name, parameter in model.unet.named_parameters():
        parameter.requires_grad_(name in expected)
    model._ba_expected_trainable_names = tuple(sorted(expected))
    assert_trainable_contract(model)


def _summary(parameters: Iterable[torch.nn.Parameter]) -> dict[str, int]:
    values = tuple(parameters)
    return {
        "tensors": len(values),
        "parameters": sum(int(parameter.numel()) for parameter in values),
    }


def assert_trainable_contract(model, optimizer=None) -> dict:
    expected = set(expected_trainable_names(model))
    actual = {
        name for name, parameter in model.unet.named_parameters()
        if parameter.requires_grad
    }
    non_unet = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("unet.")
    }
    if actual != expected or non_unet:
        raise RuntimeError(
            "E13 trainable ownership mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, "
            f"non_unet={sorted(non_unet)}"
        )

    named = dict(model.unet.named_parameters())
    prefixes = _processor_prefixes(model)
    identity_prefixes = tuple(
        f"{name}." for name in getattr(model, "_ba_identity_ca_processor_names", ())
    )
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name in sorted(expected):
        grouped.setdefault(
            trainable_role(name, prefixes, identity_prefixes), []
        ).append(named[name])
    summary = {role: _summary(params) for role, params in grouped.items()}
    summary["total"] = _summary(named[name] for name in sorted(expected))

    expected_summary = dict(EXPECTED_ROLE_SUMMARIES)
    if model.ba_residual_identity_ca_v3_enabled:
        expected_summary["residual_identity_ca_r64"] = IDENTITY_CA_SUMMARY
    if set(summary) - {"total"} != set(expected_summary):
        raise RuntimeError(
            "E13 trainable roles differ from the sealed profile: "
            f"actual={sorted(set(summary) - {'total'})}, "
            f"expected={sorted(expected_summary)}"
        )
    for role, expected_role in expected_summary.items():
        if summary[role] != expected_role:
            raise RuntimeError(
                f"E13 trainable count mismatch for {role}: "
                f"actual={summary[role]}, expected={expected_role}"
            )
    expected_total = {
        "tensors": sum(value["tensors"] for value in expected_summary.values()),
        "parameters": sum(
            value["parameters"] for value in expected_summary.values()
        ),
    }
    if summary["total"] != expected_total:
        raise RuntimeError(
            "E13 total trainable count mismatch: "
            f"actual={summary['total']}, expected={expected_total}"
        )

    if optimizer is not None:
        expected_ids = {id(named[name]) for name in expected}
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        if optimizer_ids != expected_ids:
            raise RuntimeError("Optimizer membership differs from the E13 allowlist")
    return summary


def optimizer_groups(model, config) -> list[dict]:
    named = dict(model.unet.named_parameters())
    prefixes = _processor_prefixes(model)
    identity_prefixes = tuple(
        f"{name}." for name in getattr(model, "_ba_identity_ca_processor_names", ())
    )
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name in expected_trainable_names(model):
        grouped.setdefault(
            trainable_role(name, prefixes, identity_prefixes), []
        ).append(named[name])
    lr_by_role = {
        "branched_sa_r128": float(getattr(config, "ba_lr", config.lr_for_lora)),
        "residual_identity_ca_r64": float(
            getattr(config, "ba_lr", config.lr_for_lora)
        ),
        "generic_effective_adapter_r32": float(
            getattr(config, "generic_adapter_lr", config.lr_for_lora)
        ),
        "photomaker_default_effective_adapter_r64": float(
            getattr(config, "photomaker_default_lr", config.lr_for_lora)
        ),
    }
    return [
        {"params": grouped[role], "lr": lr_by_role[role], "name": role}
        for role in lr_by_role
        if grouped.get(role)
    ]


def architecture_manifest(model) -> dict:
    named = dict(model.unet.named_parameters())
    names = expected_trainable_names(model)
    semantic_names = tuple(getattr(model, "_ba_patched_processor_names", ()))
    hard_v1_extensions = {
        "true_reference_key_mask": False,
        "branch_output_rank": None,
        "reference_roi_warp": False,
        "face_fusion_mode": "hard_reference_replace",
        "lora_rank": 128,
    }
    if str(getattr(model, "ba_hardcase_mode", "off")) != "off":
        hard_v1_extensions["hardcase_route"] = {
            "mode": str(model.ba_hardcase_mode),
            "groups": list(model.ba_hardcase_groups),
            "transition_cells": int(model.ba_hardcase_transition_cells),
        }
        if str(model.ba_hardcase_mode) == "temporal_frequency":
            hard_v1_extensions["hardcase_route"]["frequency_schedule"] = {
                "low_early": float(model.ba_hardcase_frequency_low_early),
                "low_late": float(model.ba_hardcase_frequency_low_late),
                "high_early": float(model.ba_hardcase_frequency_high_early),
                "high_late": float(model.ba_hardcase_frequency_high_late),
            }
    if bool(getattr(model, "ba_frequency_surface_loss_enabled", False)):
        hard_v1_extensions["frequency_surface_loss"] = {
            "groups": list(model.ba_frequency_surface_loss_groups),
            "top_weight": float(model.ba_frequency_surface_top_weight),
            "top_low_band_factor": float(
                model.ba_frequency_surface_top_low_band_factor
            ),
            "visible_floor_weight": float(
                model.ba_frequency_surface_visible_floor_weight
            ),
            "visible_floor_ratio": float(
                model.ba_frequency_surface_visible_floor_ratio
            ),
        }
    if bool(getattr(model, "ba_null_key_router_enabled", False)):
        hard_v1_extensions["null_key_router"] = {
            "groups": list(model.ba_null_key_router_groups),
            "entropy_threshold": float(model.ba_null_key_entropy_threshold),
            "temperature": float(model.ba_null_key_temperature),
            "max_abstention": float(model.ba_null_key_max_abstention),
            "min_reference_fraction": float(
                model.ba_null_key_min_reference_fraction
            ),
            "native_fallback": True,
            "trainable_parameters": 0,
        }
    if bool(getattr(model, "ba_crossview_consistency_enabled", False)):
        hard_v1_extensions["crossview_consistency"] = {
            "probability": float(model.ba_crossview_consistency_probability),
            "weight": float(model.ba_crossview_consistency_weight),
            "teacher_stop_gradient": True,
        }
    if bool(getattr(model, "ba_residual_identity_ca_v3_enabled", False)):
        hard_v1_extensions["residual_identity_ca_v3"] = {
            "groups": list(model.ba_residual_identity_ca_v3_groups),
            "rank": int(model.ba_residual_identity_ca_v3_rank),
            "gate_init": float(model.ba_residual_identity_ca_v3_gate_init),
            "gate_max": float(model.ba_residual_identity_ca_v3_gate_max),
            "routing": "target_q_active_photomaker_id_kv",
            "merge": "native_plus_face_mask_times_bounded_rms_delta",
            "zero_init_output": True,
        }
    return {
        "format": "photomaker_branched_trainable_unet_v2",
        "ba_architecture_version": ARCHITECTURE,
        "processor_code_version": (
            4 if bool(getattr(model, "ba_null_key_router_enabled", False))
            else 3 if bool(getattr(model, "ba_residual_identity_ca_v3_enabled", False))
            else 2
        ),
        "branched_attn_lora_rank": int(model.branched_attn_lora_rank),
        "branched_attn_weight_mode": "noise_and_ref",
        "branched_attn_new_weight_kind": "lora",
        "train_ba_only": True,
        "train_branched_ca_lora": False,
        "ba_patch_top_k": 1.0,
        "ba_train_top_k": 1.0,
        "non_ba_train": False,
        "pose_adapt_ratio": 0.0,
        "ca_mixing_for_face": False,
        "photomaker_start_step": 10,
        "branched_attn_start_step": 15,
        "num_inference_steps": 50,
        "generic_adapter_train_scope": "effective_all",
        "photomaker_default_train_scope": "effective_all",
        "hard_v1_extensions": hard_v1_extensions,
        "semantic_processor_names_sha256": hashlib.sha256(
            "\n".join(semantic_names).encode("utf-8")
        ).hexdigest(),
        "trainable_names": list(names),
        "trainable_shapes": {name: list(named[name].shape) for name in names},
        "trainable_dtypes": {
            name: str(named[name].dtype).replace("torch.", "") for name in names
        },
    }


def get_state_dict(model) -> dict:
    # 10 Aug 2026 - E13C-CORE-04: Save the complete requires-grad allowlist,
    # including both outer adapters; the June subset saver lost trained paths.
    assert_trainable_contract(model)
    named = dict(model.unet.named_parameters())
    names = expected_trainable_names(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "state_format": STATE_FORMAT,
        "architecture": architecture_manifest(model),
        "trainable_unet": {
            name: named[name].detach().cpu().clone() for name in names
        },
    }


def _validate_compatible_manifest(saved: dict, current: dict) -> None:
    # The clean launch surface starts fresh schema-v2 runs only. Exact
    # manifests keep checkpoint semantics explicit without old E14-E24
    # projection branches.
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(f"E13 checkpoint architecture mismatch: {mismatches}")


def load_state_dict(model, state: dict) -> None:
    if int(state.get("schema_version", 1)) != SCHEMA_VERSION:
        raise RuntimeError("The clean E13 family accepts schema-v2 checkpoints only")
    if state.get("state_format") != STATE_FORMAT:
        raise RuntimeError(f"Unknown E13 state format: {state.get('state_format')!r}")
    current = architecture_manifest(model)
    _validate_compatible_manifest(state.get("architecture") or {}, current)
    received = state.get("trainable_unet")
    if not isinstance(received, dict):
        raise RuntimeError("Schema-v2 checkpoint is missing trainable_unet")
    expected = set(current["trainable_names"])
    if set(received) != expected:
        raise RuntimeError("Schema-v2 checkpoint trainable names do not match E13")
    named = dict(model.unet.named_parameters())
    with torch.no_grad():
        for name in sorted(expected):
            value = received[name]
            parameter = named[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"Checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    assert_trainable_contract(model)
