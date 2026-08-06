#!/usr/bin/env python3
"""Fail-closed composition/spec gate for the E19-E24 parallel suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


RUNS = {
    "E19_large_ds_joint_shadow_sa128_multiref_24k": (
        "E19_large_ds_joint_shadow_sa128_multiref_24k_full96_r2"
    ),
    "E20_large_ds_joint_shadow_sa128_branchout_r32_24k": (
        "E20_large_ds_joint_shadow_sa128_branchout_r32_24k_full96_r2"
    ),
    "E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k": (
        "E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k_full96_r2"
    ),
    "E22_large_ds_joint_shadow_sa128_arcfaceaux_24k": (
        "E22_large_ds_joint_shadow_sa128_arcfaceaux_24k_full96_r2"
    ),
    "E23_large_ds_joint_shadow_sa128_earlydecay_24k": (
        "E23_large_ds_joint_shadow_sa128_earlydecay_24k_full96_r2"
    ),
    "E24_large_ds_joint_shadow_sa128_alternating_24k": (
        "E24_large_ds_joint_shadow_sa128_alternating_24k_full96_r2"
    ),
}
ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"


def value(config, path: str):
    selected = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


def require(config, path: str, expected) -> None:
    actual = value(config, path)
    if actual != expected:
        raise RuntimeError(
            f"E19-E24 invariant failed for {path}: "
            f"expected={expected!r}, actual={actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=sorted(RUNS), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    args = parser.parse_args()
    if args.run_name != RUNS[args.config_name]:
        raise RuntimeError(f"Run/config mismatch: expected {RUNS[args.config_name]!r}")

    config_dir = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=args.config_name, overrides=["writer=cometml"])
    experiment_number = int(args.config_name[1:3])
    multiref = experiment_number in {19, 21}
    branch_output = experiment_number in {20, 21}

    common = {
        "val_datasets_names": ["manual_val"],
        "train_ba_only": True,
        "disable_branched_ca": True,
        "model.ba_architecture_version": "hard_replace_v1",
        "model.ba_hard_v1_lora_rank": 128,
        "model.generic_adapter_train_scope": "effective_all",
        "model.photomaker_default_train_scope": "effective_all",
        "model.strict_trainable_contract": True,
        "model.branched_state_dict_mode": "trainable_v2",
        "model.ba_hard_v1_branch_output_rank": 32 if branch_output else None,
        "validation_shadow_photomaker_default": True,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.save_period": 1,
        "trainer.face_quality.expected_images": 96,
        "trainer.face_quality.enabled": True,
        "trainer.face_quality.device": "cuda",
        "trainer.face_quality.execution_mode": "deferred",
        "weights_only_save_period": 1,
        "lr_scheduler.warmup_steps": 20,
        "lr_scheduler.hold_steps": 8000 if experiment_number == 23 else 14000,
        "lr_scheduler.total_steps": 24000,
        "lr_scheduler.min_factor": 0.1,
        "writer.project_name": "aug-large-ds",
        "expected_trainable_contract.enabled": True,
        "ba_lr": 0.0001,
        "generic_adapter_lr": 0.0001,
        "photomaker_default_lr": 0.0001,
    }
    for path, expected in common.items():
        require(config, path, expected)

    require(
        config,
        "train_dataset_name",
        "large_dataset_balanced_multiref" if multiref else "large_dataset",
    )
    require(config, "trainer.masked_loss_step", 2 if experiment_number == 24 else 1)
    expected_loss = {
        22: "masked_identity_aux",
        24: "masked_alternating_audited",
    }.get(experiment_number, "masked_alternating")
    require(config, "loss_kind", expected_loss)
    require(config, "model.identity_aux_enabled", experiment_number == 22)
    require(
        config,
        "model.identity_aux_backend",
        "arcface_torch_v2" if experiment_number == 22 else "photomaker_clip_v1",
    )

    expected_tensors = 2380 if branch_output else 2240
    expected_parameters = 224542720 if branch_output else 219217920
    require(config, "expected_trainable_contract.total_tensors", expected_tensors)
    require(config, "expected_trainable_contract.total_parameters", expected_parameters)
    require(config, "expected_trainable_contract.optimizer_tensors", expected_tensors)
    require(config, "expected_trainable_contract.optimizer_parameters", expected_parameters)
    if branch_output:
        require(config, "expected_trainable_contract.categories.branched_sa_r128.tensors", 980)
        require(
            config,
            "expected_trainable_contract.categories.branched_sa_r128.parameters",
            133120000,
        )

    if multiref:
        require(config, "train_dataloader_shuffle", False)
        prefix = "datasets.train.large_dataset_balanced_multiref"
        require(config, f"{prefix}.schedule_rows", 48000)
        require(config, f"{prefix}.schedule_start_row", 0)
        require(config, f"{prefix}.schedule_seed", 130018)
        require(config, f"{prefix}.num_identity_refs", 3)
        require(config, "model.batched_conditioning_preparation", False)

    if experiment_number == 22:
        checks = {
            "model.identity_aux_model_sha256": ARCFACE_SHA256,
            "model.identity_aux_cadence": 2,
            "model.identity_aux_max_timestep": 300,
            "model.identity_aux_ramp_start_step": 4000,
            "model.identity_aux_ramp_end_step": 6000,
            "model.identity_aux_max_weight": 0.05,
            "model.identity_aux_dynamic_weight": True,
            "model.identity_aux_grad_target_ratio": 0.075,
            "model.identity_aux_grad_norm_interval": 200,
        }
        for path, expected in checks.items():
            require(config, path, expected)
        if not str(value(config, "model.identity_aux_model_path")).endswith(
            "/buffalo_l/w600k_r50.onnx"
        ):
            raise RuntimeError("E22 must use the buffalo_l w600k_r50 ONNX model")

    comment = str(value(config, "writer.experiment_comment")).strip()
    if not comment:
        raise RuntimeError("writer.experiment_comment must be non-empty")
    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run_name mismatch")
    plan = spec.get("plan") or {}
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_E19_E24_large_ds_24k_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("comet_project") != "aug-large-ds":
        raise RuntimeError("Experiment spec Comet project mismatch")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "optimizer_steps": 24000,
                "shadow_default_validation": True,
                "trainable_parameters": expected_parameters,
                "loss_kind": expected_loss,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
