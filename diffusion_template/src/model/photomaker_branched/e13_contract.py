"""Audited ownership and checkpoint contract for E13, BC_E13 and CL14."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch


SCHEMA_VERSION = 2
STATE_FORMAT = "trainable_unet_v2"
ARCHITECTURE = "hard_replace_v1"


def initialise_e13_contract(model, *, ba_hard_v1_lora_rank: int = 128,
                            generic_adapter_train_scope: str = "effective_all",
                            photomaker_default_train_scope: str = "effective_all",
                            strict_branched_install: bool = True,
                            strict_trainable_contract: bool = True,
                            branched_state_dict_mode: str = STATE_FORMAT,
                            ba_training_mask_feather: int = 0,
                            conditioning_cache_enabled: bool = False,
                            skip_unused_text_conditioning: bool = True,
                            batched_conditioning_preparation: bool = True,
                            cache_prepared_masks: bool = True,
                            compute_branch_debug_outputs: bool = False) -> None:
    """Persist the deliberately small set of E13-family runtime controls."""
    # 10 Aug 2026 - E13C-CORE-01/02: E13 is one fail-closed hard-v1 route with
    # rank-128 branch LoRA. Rejecting other values prevents a failed installer
    # or a stale config from silently training the June rank-32/base route.
    if int(ba_hard_v1_lora_rank) != 128:
        raise ValueError("The clean E13-family contract requires BA rank 128")
    if generic_adapter_train_scope != "effective_all":
        raise ValueError("E13 requires generic_adapter_train_scope=effective_all")
    if photomaker_default_train_scope != "effective_all":
        raise ValueError("E13 requires photomaker_default_train_scope=effective_all")
    if not strict_branched_install or not strict_trainable_contract:
        raise ValueError("The clean E13-family contract is always fail closed")
    if branched_state_dict_mode not in {STATE_FORMAT, "trainable_v2"}:
        raise ValueError("E13 requires schema-v2 trainable checkpoints")
    if not 0 <= int(ba_training_mask_feather) <= 8:
        raise ValueError("ba_training_mask_feather must be within [0, 8]")
    if conditioning_cache_enabled:
        raise ValueError("E13 diverse-pair training requires conditioning cache off")

    model.ba_architecture_version = ARCHITECTURE
    model.ba_hard_v1_lora_rank = 128
    # The sealed schema retains the generic adapter rank (32) in
    # branched_attn_lora_rank and records the hard-v1 override separately.
    # Runtime installation always reads ba_hard_v1_lora_rank first.
    model.branched_attn_lora_rank = int(model.lora_rank)
    model.generic_adapter_train_scope = generic_adapter_train_scope
    model.photomaker_default_train_scope = photomaker_default_train_scope
    model.strict_branched_install = True
    model.strict_trainable_contract = True
    model.branched_state_dict_mode = STATE_FORMAT
    model.ba_training_mask_feather = int(ba_training_mask_feather)
    # 10 Aug 2026 - E13C-PERF-01: Large/BigCelebs/Cosmic pairs are effectively
    # unique; record the sealed cache-off policy instead of paying bookkeeping
    # for a cache that cannot warm.
    model.conditioning_cache_enabled = False
    model.skip_unused_text_conditioning = bool(skip_unused_text_conditioning)
    model.batched_conditioning_preparation = bool(batched_conditioning_preparation)
    model.cache_prepared_masks = bool(cache_prepared_masks)
    model.compute_branch_debug_outputs = bool(compute_branch_debug_outputs)


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _processor_prefixes(model) -> tuple[str, ...]:
    names = tuple(getattr(model, "_ba_patched_processor_names", ()))
    selected = tuple(name for name in names if name.endswith("attn1.processor"))
    if not selected:
        raise RuntimeError("E13 processor installation selected zero self-attention blocks")
    return tuple(f"{name}." for name in selected)


def trainable_role(name: str, processor_prefixes: tuple[str, ...]) -> str | None:
    """Map a U-Net tensor to one and only one E13 optimizer role."""
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
    return tuple(sorted(
        name for name, _ in model.unet.named_parameters()
        if trainable_role(name, prefixes) is not None
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
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name in sorted(expected):
        grouped.setdefault(trainable_role(name, prefixes), []).append(named[name])
    summary = {role: _summary(params) for role, params in grouped.items()}
    summary["total"] = _summary(named[name] for name in sorted(expected))

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
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name in expected_trainable_names(model):
        grouped.setdefault(trainable_role(name, prefixes), []).append(named[name])
    lr_by_role = {
        "branched_sa_r128": float(getattr(config, "ba_lr", config.lr_for_lora)),
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
    return {
        "format": "photomaker_branched_trainable_unet_v2",
        "ba_architecture_version": ARCHITECTURE,
        "processor_code_version": 2,
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
        "photomaker_start_step": int(model.photomaker_start_step),
        "branched_attn_start_step": int(model.branched_attn_start_step),
        "num_inference_steps": int(model.num_inference_steps),
        "generic_adapter_train_scope": "effective_all",
        "photomaker_default_train_scope": "effective_all",
        "hard_v1_extensions": {
            "true_reference_key_mask": False,
            "branch_output_rank": None,
            "reference_roi_warp": False,
            "face_fusion_mode": "hard_reference_replace",
            "lora_rank": 128,
        },
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
    # Older E13/CL14 manifests contain inert E14-E24 fields. Compare only the
    # output-affecting E13 projection plus exact trainable names/shapes so those
    # sealed checkpoints remain loadable without carrying later mechanisms.
    required = (
        "format", "ba_architecture_version", "processor_code_version",
        "branched_attn_lora_rank",
        "branched_attn_weight_mode", "branched_attn_new_weight_kind",
        "train_ba_only", "train_branched_ca_lora", "ba_patch_top_k",
        "ba_train_top_k", "non_ba_train", "pose_adapt_ratio",
        "ca_mixing_for_face", "photomaker_start_step",
        "branched_attn_start_step", "num_inference_steps",
        "generic_adapter_train_scope", "photomaker_default_train_scope",
        "semantic_processor_names_sha256", "trainable_names",
        "trainable_shapes", "trainable_dtypes",
    )
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in required if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(f"E13 checkpoint architecture mismatch: {mismatches}")
    # 10 Aug 2026 - E13C-CORE-04: Compare the hard-v1 extension projection
    # explicitly. Later manifests may carry unrelated extensions, but the four
    # routing invariants and rank used by E13 must still match exactly.
    extension_keys = (
        "true_reference_key_mask", "branch_output_rank",
        "reference_roi_warp", "face_fusion_mode", "lora_rank",
    )
    saved_extensions = saved.get("hard_v1_extensions") or {}
    current_extensions = current["hard_v1_extensions"]
    extension_mismatches = {
        key: (saved_extensions.get(key), current_extensions.get(key))
        for key in extension_keys
        if saved_extensions.get(key) != current_extensions.get(key)
    }
    if extension_mismatches:
        raise RuntimeError(
            "E13 checkpoint hard-v1 extension mismatch: "
            f"{extension_mismatches}"
        )


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
