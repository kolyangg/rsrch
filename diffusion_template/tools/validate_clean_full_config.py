#!/usr/bin/env python3
"""Compose and fail closed on every config supported by clean_full."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "src" / "configs"
RUNS_PATH = CONFIG_DIR / "clean_full_runs.json"
MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "src/model/photomaker_branched/clean_full_model_config.py"
)
_MODEL_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "clean_full_model_config", MODEL_CONFIG_PATH
)
if _MODEL_CONFIG_SPEC is None or _MODEL_CONFIG_SPEC.loader is None:
    raise RuntimeError(f"Cannot load clean_full model config resolver: {MODEL_CONFIG_PATH}")
_MODEL_CONFIG_MODULE = importlib.util.module_from_spec(_MODEL_CONFIG_SPEC)
_MODEL_CONFIG_SPEC.loader.exec_module(_MODEL_CONFIG_MODULE)
resolve_clean_full_model_config = _MODEL_CONFIG_MODULE.resolve_clean_full_model_config
COMMON_TARGETS = {
    "trainer._target_": "src.trainer.clean_full_trainers.PhotomakerLoraTrainer",
    "model._target_": (
        "src.model.photomaker_branched.clean_full_model.PhotomakerBranchedLora"
    ),
    "pipeline._target_": (
        "src.pipelines.photomaker_branched_clean."
        "PhotomakerBranchedPipeline.from_pretrained"
    ),
    "loss_function._target_": "src.loss.diffusion_loss.MaskedDiffusionLoss",
    "optimizer._target_": "torch.optim.AdamW",
    "lr_scheduler._target_": "src.lr_schedulers.lr_schedulers.WarmupHoldCosineLR",
    "writer._target_": "src.logger.cometml.CometMLWriter",
}
COMMON_VALUES = {
    "model.clean_full_config.runtime.skip_unused_text_conditioning": True,
    "model.clean_full_config.runtime.conditioning_cache_enabled": False,
    "model.clean_full_config.runtime.batched_conditioning_preparation": True,
    "model.clean_full_config.runtime.cache_prepared_masks": True,
    "model.clean_full_config.runtime.compute_branch_debug_outputs": False,
    "model.clean_full_config.contract.strict_branched_install": True,
    "model.clean_full_config.contract.strict_trainable_contract": True,
    "model.clean_full_config.contract.state_dict_mode": "trainable_v2",
    "model.clean_full_config.contract.generic_adapter_train_scope": "effective_all",
    "model.clean_full_config.contract.photomaker_default_train_scope": "effective_all",
    "model.clean_full_config.contract.hard_v1_lora_rank": 128,
    "model.clean_full_config.architecture.version": "hard_replace_v1",
    "model.clean_full_config.architecture.enforce_reference_only_hard_route": True,
    "model.clean_full_config.architecture.training_timestep_policy": "uniform_all",
    "model.pose_adapt_ratio": 0.0,
    "model.ca_mixing_for_face": False,
    "pipeline.pose_adapt_ratio": 0.0,
    "pipeline.ca_mixing_for_face": False,
    "pipeline.auto_mask_ref": False,
    "disable_branched_sa": False,
    "disable_branched_ca": True,
    "loss_kind": "masked_alternating",
    "trainer.epoch_len": 2000,
    "trainer.n_epochs": 12,
    "trainer.validation_interval_steps": 2000,
    "trainer.skip_initial_validation": False,
    "trainer.from_pretrained": None,
    "trainer.face_quality.enabled": True,
    "trainer.face_quality.expected_images": 96,
    "trainer.face_quality.execution_mode": "deferred",
    "datasets.val.manual_val.limit": 96,
    "dataloaders.manual_val.batch_size": 12,
    "validation_args.num_images_per_prompt": 1,
    "validation_args.num_inference_steps": 50,
    "validation_args.guidance_scale": 5,
    "lr_scheduler.total_steps": 24000,
    "writer.mode": "online",
    "writer.require_online_registration": True,
}
RECENT_EXTENSION_KINDS = (
    "null_key_router",
    "identity_motion_projector",
    "landmark_canonical_kv",
    "component_token_memory",
    "id_adaptive_modulation",
    "semantic_window_gate",
)
FORBIDDEN_INTERNAL_FLAGS = (
    "ba_visibility_ownership_v2_enabled",
    "ba_frequency_learnable_schedule_enabled",
    "ba_frequency_lowband_contrastive_enabled",
    "ba_frequency_positive_sameid_enabled",
    "ba_attention_ownership_loss_enabled",
    "ba_roi_teacher_distill_enabled",
    "ba_frequency_shared_schedule_enabled",
    "ba_pm_boundary_distill_enabled",
    "ba_patch_identity_enabled",
    "identity_aux_enabled",
    "ba_identity_ca_v2_enabled",
    "ba_residual_identity_ca_v3_enabled",
)


def selected(config, path: str) -> Any:
    value = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("runs"), dict):
        raise RuntimeError(f"Invalid clean_full run manifest: {RUNS_PATH}")
    return manifest


def compose_and_validate(config_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest()
    spec = manifest["runs"].get(config_name)
    if spec is None:
        raise RuntimeError(
            f"Unsupported clean_full config {config_name!r}; use --list to inspect the allowlist"
        )
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Allowlisted config is absent: {config_path}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=["writer=cometml"])
    flat_model_options = resolve_clean_full_model_config(
        selected(config, "model.clean_full_config")
    )

    checks = dict(COMMON_TARGETS)
    checks.update(COMMON_VALUES)
    checks.update(
        {
            "train_dataset_name": spec["dataset"],
            f"datasets.train.{spec['dataset']}._target_": spec["dataset_target"],
            "model.clean_full_config.hardcase.mode": spec["hardcase_mode"],
            "expected_trainable_contract.total_tensors": spec["trainable_tensors"],
            "expected_trainable_contract.total_parameters": spec["trainable_parameters"],
            "expected_trainable_contract.optimizer_tensors": spec["trainable_tensors"],
            "expected_trainable_contract.optimizer_parameters": spec["trainable_parameters"],
            spec["feature_path"]: spec["feature_value"],
        }
    )
    drift = {
        path: {"expected": expected, "actual": selected(config, path)}
        for path, expected in checks.items()
        if selected(config, path) != expected
    }
    if drift:
        raise RuntimeError(f"clean_full config drift for {config_name}: {drift}")

    expected_validation_only = bool(spec["validation_only"])
    actual_validation_only = bool(selected(config, "validation_only") is True)
    if actual_validation_only != expected_validation_only:
        raise RuntimeError(
            f"validation_only drift: expected={expected_validation_only}, "
            f"actual={actual_validation_only}"
        )
    expected_ba_validation = not expected_validation_only
    if selected(config, "validation_args.use_branched_attention") is not expected_ba_validation:
        raise RuntimeError("PM0/training branched-validation boundary drifted")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("clean_full permits only the sealed manual_val96 panel")
    if selected(config, "inference_metrics") != [
        "clip_ts",
        "id_sim_best_legacy",
        "id_sim_subject_v2",
    ]:
        raise RuntimeError("clean_full metric contract drifted")

    recent_kind = selected(config, "model.clean_full_config.recent_extension.kind")
    expected_recent = (
        spec["feature_value"]
        if spec["feature_path"]
        == "model.clean_full_config.recent_extension.kind"
        else "none"
    )
    if recent_kind != expected_recent:
        raise RuntimeError(
            f"Recent independent-arm violation: expected={expected_recent}, actual={recent_kind}"
        )
    enabled_recent = [
        kind
        for kind in RECENT_EXTENSION_KINDS
        if flat_model_options[f"ba_{kind}_enabled"] is True
    ]
    if enabled_recent != ([] if expected_recent == "none" else [expected_recent]):
        raise RuntimeError(
            f"Resolved extension mismatch: expected={expected_recent}, actual={enabled_recent}"
        )
    forbidden_active = [
        name for name in FORBIDDEN_INTERNAL_FLAGS if flat_model_options[name] is True
    ]
    if forbidden_active:
        raise RuntimeError(
            f"clean_full config enables excluded experiment paths: {forbidden_active}"
        )
    legacy_flat_keys = sorted(set(flat_model_options) & set(config.model.keys()))
    if legacy_flat_keys:
        raise RuntimeError(
            f"Flat historical model options escaped clean_full_config: {legacy_flat_keys}"
        )
    if flat_model_options["ba_spatial_reference_shuffle_probability"] != 0.0:
        raise RuntimeError("clean_full forbids spatial-reference shuffling")
    pcgrad = selected(config, "trainer.ba_pcgrad_enabled") is True
    if pcgrad != (config_name == "CL45_cosmic_ba_pcgrad_24k"):
        raise RuntimeError("PCGrad must be active only for CL45")

    summary = {
        "status": "ok",
        "config_name": config_name,
        "family": spec["family"],
        "dataset": spec["dataset"],
        "dataset_target": spec["dataset_target"],
        "validation_only": expected_validation_only,
        "steps": 0 if expected_validation_only else 24000,
        "validation_steps": [0]
        if expected_validation_only
        else list(range(0, 24001, 2000)),
        "feature": {"path": spec["feature_path"], "value": spec["feature_value"]},
        "trainable_tensors": spec["trainable_tensors"],
        "trainable_parameters": spec["trainable_parameters"],
        "canonical_run": spec["canonical_run"],
        "canonical_comet_key": spec["canonical_comet_key"],
    }
    return summary, manifest


def write_run_record(
    path: Path,
    *,
    run_name: str,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise RuntimeError(f"Unsafe run name: {run_name!r}")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing run record: {path}")
    if path.parent.exists() and any(path.parent.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at_utc": now,
        "source": "tools/validate_clean_full_config.py",
        "plan": {
            "branch": manifest["branch"],
            "config": f"src/configs/{summary['config_name']}.yaml",
            "launcher": "launchers/active/run_clean_full_config_1gpu.sh",
            "machine": "serv",
            "gpus": 1,
            "comet_project": manifest["comet_project"],
            "validation_contract": manifest["validation_contract"],
            "dataset": summary["dataset"],
            "feature": summary["feature"],
            "historical_reference": {
                "run_name": summary["canonical_run"],
                "comet_experiment_key": summary["canonical_comet_key"],
            },
        },
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name")
    parser.add_argument("--run-name")
    parser.add_argument("--write-run-record", type=Path)
    parser.add_argument("--field", choices=("dataset", "validation_steps", "validation_only"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list:
        print("\n".join(manifest["runs"]))
        return
    if not args.config_name:
        parser.error("--config-name is required unless --list is used")
    summary, manifest = compose_and_validate(args.config_name)
    if args.write_run_record:
        if not args.run_name:
            parser.error("--run-name is required with --write-run-record")
        write_run_record(
            args.write_run_record,
            run_name=args.run_name,
            summary=summary,
            manifest=manifest,
        )
    if args.field:
        value = summary[args.field]
        print(",".join(str(item) for item in value) if isinstance(value, list) else value)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
