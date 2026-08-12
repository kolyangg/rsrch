#!/usr/bin/env python3
"""Fail-closed composition gate for the BC_E13 BigCelebs dataset transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_NAME = "BC_E13_big_celebs_joint_shadow_sa128_24k"
RUN_NAME = "BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1"
LAUNCHER = "launchers/active/run_BC_E13_big_celebs_24k_1gpu.sh"
SERV_YAML = (
    "serv_run_packages/BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1/"
    "run_BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1_1gpu.yaml"
)


def value(config, path: str):
    selected = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


def require(config, path: str, expected) -> None:
    actual = value(config, path)
    if actual != expected:
        raise RuntimeError(
            f"BC_E13 BigCelebs invariant failed for {path}: "
            f"expected={expected!r}, actual={actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=[CONFIG_NAME], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    args = parser.parse_args()

    if args.run_name != RUN_NAME:
        raise RuntimeError(f"Run/config mismatch: expected {RUN_NAME!r}")

    config_dir = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=args.config_name, overrides=["writer=cometml"])

    invariants = {
        "train_dataset_name": "big_celebs",
        "val_datasets_names": ["manual_val"],
        "datasets.train.big_celebs._target_": (
            "src.datasets.big_celebs.BigCelebsTrain"
        ),
        "datasets.train.big_celebs.num_refs": 1,
        "datasets.train.big_celebs.trigger_word": "img",
        "datasets.train.big_celebs.strict_manifest_fields": True,
        "datasets.train.big_celebs.random_horizontal_flip": True,
        "train_ba_only": True,
        "disable_branched_ca": True,
        "model.rank": 32,
        "model.ba_architecture_version": "hard_replace_v1",
        "model.ba_hard_v1_lora_rank": 128,
        "model.generic_adapter_train_scope": "effective_all",
        "model.photomaker_default_train_scope": "effective_all",
        "model.strict_trainable_contract": True,
        "model.branched_state_dict_mode": "trainable_v2",
        "validation_shadow_photomaker_default": True,
        "loss_kind": "masked_alternating",
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "dataloaders.train.batch_size": 2,
        "dataloaders.train.num_workers": 2,
        "dataloaders.manual_val.batch_size": 12,
        "datasets.val.manual_val.limit": 96,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.save_period": 1,
        "trainer.masked_loss_step": 1,
        "trainer.face_quality.enabled": True,
        "trainer.face_quality.expected_images": 96,
        "trainer.face_quality.device": "cuda",
        "trainer.face_quality.execution_mode": "deferred",
        "weights_only_save_period": 1,
        "lr_for_lora": 0.0001,
        "ba_lr": 0.0001,
        "generic_adapter_lr": 0.0001,
        "photomaker_default_lr": 0.0001,
        "lr_scheduler.warmup_steps": 20,
        "lr_scheduler.hold_steps": 14000,
        "lr_scheduler.total_steps": 24000,
        "lr_scheduler.min_factor": 0.1,
        "writer.project_name": "aug-large-ds",
        "expected_trainable_contract.enabled": True,
        "expected_trainable_contract.total_tensors": 2240,
        "expected_trainable_contract.total_parameters": 219217920,
        "expected_trainable_contract.optimizer_tensors": 2240,
        "expected_trainable_contract.optimizer_parameters": 219217920,
    }
    for path, expected in invariants.items():
        require(config, path, expected)

    comment = str(value(config, "writer.experiment_comment")).strip()
    if "BigCelebs v2" not in comment or "only the training dataset changes" not in comment:
        raise RuntimeError("writer.experiment_comment must describe the dataset-only delta")

    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run_name mismatch")
    plan = spec.get("plan") or {}
    expected_plan = {
        "machine": "serv",
        "gpus": 1,
        "config": f"src/configs/{args.config_name}.yaml",
        "launcher": LAUNCHER,
        "serv_yaml": SERV_YAML,
        "comet_project": "aug-large-ds",
    }
    for field, expected in expected_plan.items():
        actual = plan.get(field)
        if actual != expected:
            raise RuntimeError(
                f"Experiment spec mismatch for plan.{field}: "
                f"expected={expected!r}, actual={actual!r}"
            )
    lifecycle_statuses = {
        "prepared_local",
        "running",
        "completed",
        "failed",
        "stopped",
    }
    if plan.get("status") not in lifecycle_statuses:
        raise RuntimeError(f"Unexpected experiment lifecycle status: {plan.get('status')!r}")
    if plan.get("controlled_delta") != "training dataset: large_dataset -> big_celebs":
        raise RuntimeError("Experiment spec must record the dataset-only delta")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "train_dataset_name": "big_celebs",
                "optimizer_steps": 24000,
                "shadow_default_validation": True,
                "trainable_tensors": 2240,
                "trainable_parameters": 219217920,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
