#!/usr/bin/env python3
"""Fail-closed composition gate for the E13-E18 parallel suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


RUNS = {
    "E13_large_ds_joint_shadow_sa128_24k": "E13_large_ds_joint_shadow_sa128_24k_full96_r1",
    "E14_large_ds_joint_shadow_sa128_protected_24k": "E14_large_ds_joint_shadow_sa128_protected_24k_full96_r2",
    "E15_large_ds_joint_persist_sa128_protected_24k": "E15_large_ds_joint_persist_sa128_protected_24k_full96_r2",
    "E16_large_ds_joint_persist_sa128_idloss_24k": "E16_large_ds_joint_persist_sa128_idloss_24k_full96_r2",
    "E17_large_ds_joint_persist_sa128_resididca_24k": "E17_large_ds_joint_persist_sa128_resididca_24k_full96_r2",
    "E18_large_ds_joint_persist_sa128_multiref_24k": "E18_large_ds_joint_persist_sa128_multiref_24k_full96_r2",
}


def value(config, path: str):
    selected = OmegaConf.select(config, path, default="<missing>")
    return OmegaConf.to_container(selected, resolve=True) if OmegaConf.is_config(selected) else selected


def require(config, path: str, expected) -> None:
    actual = value(config, path)
    if actual != expected:
        raise RuntimeError(
            f"E13-E18 invariant failed for {path}: expected={expected!r}, actual={actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=sorted(RUNS), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    args = parser.parse_args()
    if args.run_name != RUNS[args.config_name]:
        raise RuntimeError(
            f"Run/config mismatch: expected {RUNS[args.config_name]!r}"
        )

    config_dir = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=args.config_name, overrides=["writer=cometml"])

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
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.save_period": 1,
        "trainer.face_quality.expected_images": 96,
        "weights_only_save_period": 1,
        "lr_scheduler.warmup_steps": 20,
        "lr_scheduler.hold_steps": 14000,
        "lr_scheduler.total_steps": 24000,
        "lr_scheduler.min_factor": 0.1,
        "writer.project_name": "aug-large-ds",
        "expected_trainable_contract.enabled": True,
    }
    for path, expected in common.items():
        require(config, path, expected)

    experiment_number = int(args.config_name[1:3])
    require(
        config,
        "train_dataset_name",
        "large_dataset_balanced_multiref" if experiment_number == 18 else "large_dataset",
    )
    require(
        config,
        "validation_shadow_photomaker_default",
        experiment_number in {13, 14},
    )
    require(config, "ba_lr", 0.0001)
    require(
        config,
        "generic_adapter_lr",
        0.0001 if experiment_number in {13, 14} else 0.00005,
    )
    require(
        config,
        "photomaker_default_lr",
        0.0001 if experiment_number in {13, 14} else 0.00001,
    )
    require(config, "model.identity_aux_enabled", experiment_number == 16)
    require(
        config,
        "loss_kind",
        "masked_alternating" if experiment_number == 13 else "branched_reference",
    )
    if experiment_number >= 14:
        require(
            config,
            "loss_function._target_",
            "src.loss.branched_reference_loss.BranchedReferenceLoss",
        )
    require(
        config,
        "model.ba_residual_identity_ca_v3_enabled",
        experiment_number == 17,
    )
    if experiment_number == 17:
        require(config, "model.ba_residual_identity_ca_v3_rank", 64)
        require(config, "model.ba_residual_identity_ca_v3_gate_init", 0.02)
        require(config, "model.ba_residual_identity_ca_v3_gate_max", 0.2)
    if experiment_number == 18:
        require(config, "train_dataloader_shuffle", False)
        require(
            config,
            "datasets.train.large_dataset_balanced_multiref.schedule_rows",
            48000,
        )
        require(
            config,
            "datasets.train.large_dataset_balanced_multiref.num_identity_refs",
            3,
        )
        require(
            config,
            "datasets.train.large_dataset_balanced_multiref.schedule_start_row",
            0,
        )

    comment = str(value(config, "writer.experiment_comment")).strip()
    if not comment:
        raise RuntimeError("writer.experiment_comment must be non-empty")
    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run_name mismatch")
    plan = spec.get("plan") or {}
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_E13_E18_large_ds_24k_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("comet_project") != "aug-large-ds":
        raise RuntimeError("Experiment spec Comet project mismatch")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "optimizer_steps": 24000,
                "shadow_default_validation": experiment_number in {13, 14},
                "trainable_parameters": int(
                    config.expected_trainable_contract.total_parameters
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
