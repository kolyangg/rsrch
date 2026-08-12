#!/usr/bin/env python3
"""Fail-closed config/spec gate for BC_E13_ds1, ds2, and ds3."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


RUNS = {
    "BC_E13_ds1_repeatdepth_balanced_24k_full96_r1": {
        "config": "BC_E13_ds1_repeatdepth_balanced_24k",
        "selector": "bc_e13_ds1",
        "mode": "ds1",
    },
    "BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1": {
        "config": "BC_E13_ds2_scene_target_canonical_ref_24k",
        "selector": "bc_e13_ds2",
        "mode": "ds2",
    },
    "BC_E13_ds3_large_anchor_2to1_24k_full96_r1": {
        "config": "BC_E13_ds3_large_anchor_2to1_24k",
        "selector": "bc_e13_ds3",
        "mode": "ds3",
    },
}
BASE_CONFIG = "BC_E13_big_celebs_joint_shadow_sa128_24k"
LAUNCHER = "launchers/active/run_BC_E13_dataset_experiments_24k_1gpu.sh"


def selected(config, path: str):
    result = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(result):
        return OmegaConf.to_container(result, resolve=True)
    return result


def require(config, path: str, expected) -> None:
    actual = selected(config, path)
    if actual != expected:
        raise RuntimeError(
            f"Invariant failed for {path}: expected={expected!r}, actual={actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", choices=sorted(RUNS), required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    args = parser.parse_args()
    plan = RUNS[args.run_name]
    if args.config_name != plan["config"]:
        raise RuntimeError("Run/config mismatch")

    config_dir = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name=BASE_CONFIG, overrides=["writer=cometml"])
        config = compose(config_name=args.config_name, overrides=["writer=cometml"])

    require(config, "train_dataset_name", plan["selector"])
    require(
        config,
        f"datasets.train.{plan['selector']}._target_",
        "src.datasets.big_celebs_e13_scheduled.BigCelebsE13ScheduledTrain",
    )
    require(config, f"datasets.train.{plan['selector']}.expected_mode", plan["mode"])
    require(config, f"datasets.train.{plan['selector']}.expected_schedule_rows", 48000)
    require(config, f"datasets.train.{plan['selector']}.random_horizontal_flip", False)
    invariants = {
        "val_datasets_names": ["manual_val"],
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
        "trainer.face_quality.execution_mode": "deferred",
        "lr_for_lora": 0.0001,
        "ba_lr": 0.0001,
        "generic_adapter_lr": 0.0001,
        "photomaker_default_lr": 0.0001,
        "lr_scheduler.total_steps": 24000,
        "expected_trainable_contract.total_tensors": 2240,
        "expected_trainable_contract.total_parameters": 219217920,
        "expected_trainable_contract.optimizer_tensors": 2240,
        "expected_trainable_contract.optimizer_parameters": 219217920,
    }
    for path, expected in invariants.items():
        require(config, path, expected)

    comment = str(selected(config, "writer.experiment_comment")).strip()
    if "E13 base" not in comment or f"BC_E13_{plan['mode']}" not in comment:
        raise RuntimeError("Comet comment must identify E13 base and dataset arm")

    # The child config is allowed to differ from BC_E13 only in the selected
    # training dataset and its auditable Comet description.
    base_dict = OmegaConf.to_container(base, resolve=True)
    child_dict = OmegaConf.to_container(config, resolve=True)
    normalized_child = deepcopy(child_dict)
    normalized_child["train_dataset_name"] = base_dict["train_dataset_name"]
    normalized_child["writer"]["experiment_comment"] = base_dict["writer"][
        "experiment_comment"
    ]
    if normalized_child != base_dict:
        differing = sorted(
            key
            for key in set(base_dict) | set(normalized_child)
            if base_dict.get(key) != normalized_child.get(key)
        )
        raise RuntimeError(
            "Dataset experiment drifted beyond selector/comment; "
            f"top-level differences={differing}"
        )

    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run_name mismatch")
    spec_plan = spec.get("plan") or {}
    expected_spec = {
        "machine": "serv",
        "gpus": 1,
        "config": f"src/configs/{args.config_name}.yaml",
        "launcher": LAUNCHER,
        "serv_yaml": (
            f"serv_run_packages/{args.run_name}/"
            f"run_{args.run_name}_1gpu.yaml"
        ),
        "comet_project": "aug-large-ds",
        "dataset_mode": plan["mode"],
    }
    for key, expected in expected_spec.items():
        if spec_plan.get(key) != expected:
            raise RuntimeError(
                f"Spec mismatch for plan.{key}: expected={expected!r}, "
                f"actual={spec_plan.get(key)!r}"
            )
    if spec_plan.get("status") not in {
        "prepared_local",
        "submitted",
        "running",
        "completed",
        "failed",
        "stopped",
    }:
        raise RuntimeError(f"Unexpected lifecycle status: {spec_plan.get('status')!r}")
    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "config": args.config_name,
                "dataset_selector": plan["selector"],
                "dataset_mode": plan["mode"],
                "controlled_config_delta": [
                    "train_dataset_name",
                    "writer.experiment_comment",
                ],
                "trainable_tensors": 2240,
                "trainable_parameters": 219217920,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
