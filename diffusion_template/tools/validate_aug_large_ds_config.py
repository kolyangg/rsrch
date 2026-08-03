#!/usr/bin/env python3
"""Fail-closed composition gate for the August Large Dataset hard-BA suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DELTAS = {
    "E1_large_ds_truekey_20k": {
        "model.ba_hard_v1_true_reference_key_mask": True,
    },
    "E2_large_ds_branchout_20k": {
        "model.ba_hard_v1_branch_output_rank": 32,
    },
    "E3_large_ds_roiwarp_20k": {
        "model.ba_hard_v1_reference_roi_warp": True,
    },
    "E4_large_ds_midup_20k": {
        "model.ba_self_attention_groups": [
            "mid_block",
            "up_blocks.0",
            "up_blocks.1",
        ],
    },
    "E5_large_ds_infersteps_20k": {
        "model.ba_training_timestep_policy": "inference_active",
    },
    "E6_large_ds_fp32_20k": {
        "model.branched_trainable_dtype": "fp32",
    },
}

CONFIG_RUN_NAMES = {
    "E1_large_ds_truekey_20k": (
        "E1_large_ds_truekey_r32_20k_full96_r1"
    ),
    "E2_large_ds_branchout_20k": (
        "E2_large_ds_branchout_r32_20k_full96_r1"
    ),
    "E3_large_ds_roiwarp_20k": (
        "E3_large_ds_roiwarp_r32_20k_full96_r1"
    ),
    "E4_large_ds_midup_20k": (
        "E4_large_ds_midup_r32_20k_full96_r1"
    ),
    "E5_large_ds_infersteps_20k": (
        "E5_large_ds_infersteps_r32_20k_full96_r1"
    ),
    "E6_large_ds_fp32_20k": (
        "E6_large_ds_fp32_r32_20k_full96_r1"
    ),
}

SCIENTIFIC_BASE = {
    "model.ba_hard_v1_true_reference_key_mask": False,
    "model.ba_hard_v1_branch_output_rank": None,
    "model.ba_hard_v1_reference_roi_warp": False,
    "model.ba_self_attention_groups": None,
    "model.ba_training_timestep_policy": "uniform_all",
    "model.branched_trainable_dtype": "inherit",
}

FIXED_VALUES = {
    "train_dataset_name": "large_dataset",
    "val_datasets_names": ["manual_val"],
    "train_ba_only": True,
    "branched_attn_weight_mode": "noise_and_ref",
    "train_branched_ca_lora": False,
    "disable_branched_sa": False,
    "disable_branched_ca": True,
    "ba_patch_top_k": 1.0,
    "ba_train_top_k": 1.0,
    "non_ba_train": False,
    "train_ba_all_steps": True,
    "strict_face_routing": False,
    "model.rank": 32,
    "model.ba_architecture_version": "hard_replace_v1",
    "model.ba_face_fusion_mode": "hard_reference_replace",
    "model.ba_face_branch_scale": 1.0,
    "model.ba_enforce_reference_only_hard_route": True,
    "model.strict_branched_install": True,
    "model.strict_trainable_contract": True,
    "model.branched_state_dict_mode": "trainable_v2",
    "model.branched_attn_new_weight_kind": "lora",
    "model.train_branched_ca_lora": False,
    "model.branched_attn_weight_mode": "noise_and_ref",
    "model.pose_adapt_ratio": 0.0,
    "model.ca_mixing_for_face": False,
    "model.photomaker_start_step": 10,
    "model.branched_attn_start_step": 15,
    "model.num_inference_steps": 50,
    "pipeline.pose_adapt_ratio": 0.0,
    "pipeline.ca_mixing_for_face": False,
    "pipeline.photomaker_start_step": 10,
    "pipeline.branched_attn_start_step": 15,
    "dataloaders.train.batch_size": 2,
    "datasets.val.manual_val.limit": 96,
    "validation_args.num_images_per_prompt": 1,
    "validation_args.num_inference_steps": 50,
    "validation_args.guidance_scale": 5,
    "validation_args.use_branched_attention": True,
    "validation_args.photomaker_use_lora_adapter": False,
    "validation_args.branched_attn_start_step": 15,
    "datasets.val.manual_val.seeds": [0],
    "trainer.epoch_len": 2000,
    "trainer.n_epochs": 10,
    "trainer.validation_interval_steps": 2000,
    "trainer.save_period": 1,
    "trainer.skip_initial_validation": False,
    "trainer.log_per_image_id_sim_table": True,
    "trainer.face_quality.enabled": True,
    "trainer.face_quality.expected_images": 96,
    "trainer.seed": 0,
    "weights_only_save_period": 1,
    "validation_processor_base_mode": "legacy_full_copy",
    "strict_validation_processor_copy": True,
    "update_proc_weights_val": True,
    "lr_for_lora": 0.0001,
    "trainer.masked_loss_step": 1,
    "writer.project_name": "aug-large-ds",
    "writer.require_online_registration": True,
}


def normalized(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def require_equal(config, path: str, expected) -> None:
    actual = normalized(OmegaConf.select(config, path, default="<missing>"))
    if actual != expected:
        raise RuntimeError(
            f"Config invariant failed for {path}: expected={expected!r}, actual={actual!r}"
        )


def flatten_leaves(value, prefix="") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_leaves(child, path))
        return result
    return {prefix: value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, choices=sorted(CONFIG_DELTAS))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    args = parser.parse_args()

    expected_run_name = CONFIG_RUN_NAMES[args.config_name]
    if args.run_name != expected_run_name:
        raise RuntimeError(
            "Run name must contain the experiment number and key change: "
            f"expected={expected_run_name!r}, actual={args.run_name!r}"
        )

    config_dir = Path(__file__).resolve().parents[1] / "src" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=args.config_name,
            overrides=["writer=cometml"],
        )
        baseline = compose(
            config_name="large_dataset_rhca_hard_v1_audited_20k",
            overrides=["writer=cometml"],
        )

    # 3 Aug 2026 - Compare the complete resolved jobs, not just a shortlist of
    # invariants: every child may change one scientific leaf plus its comment.
    child_leaves = flatten_leaves(
        OmegaConf.to_container(config, resolve=True)
    )
    baseline_leaves = flatten_leaves(
        OmegaConf.to_container(baseline, resolve=True)
    )
    changed_paths = {
        path
        for path in set(child_leaves) | set(baseline_leaves)
        if child_leaves.get(path, "<missing>")
        != baseline_leaves.get(path, "<missing>")
    }
    expected_changed_paths = {
        *CONFIG_DELTAS[args.config_name],
        "writer.experiment_comment",
    }
    if changed_paths != expected_changed_paths:
        raise RuntimeError(
            "Child config does not have exactly one scientific delta: "
            f"expected={sorted(expected_changed_paths)}, "
            f"actual={sorted(changed_paths)}"
        )

    for path, expected in FIXED_VALUES.items():
        require_equal(config, path, expected)

    expected_scientific = dict(SCIENTIFIC_BASE)
    expected_scientific.update(CONFIG_DELTAS[args.config_name])
    for path, expected in expected_scientific.items():
        require_equal(config, path, expected)

    comment = str(OmegaConf.select(config, "writer.experiment_comment", default="")).strip()
    if not comment:
        raise RuntimeError("writer.experiment_comment must be non-empty")

    with args.experiment_spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("experiment spec run_name does not match launcher")
    plan = spec.get("plan") or {}
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("experiment spec config does not match launcher")
    if plan.get("comet_project") != "aug-large-ds":
        raise RuntimeError("experiment spec must pin Comet project aug-large-ds")
    if plan.get("experiment_comment") != comment:
        raise RuntimeError("experiment spec and composed Comet comments differ")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "config_name": args.config_name,
                "optimizer_steps": int(config.trainer.epoch_len)
                * int(config.trainer.n_epochs),
                "single_scientific_delta": CONFIG_DELTAS[args.config_name],
                "experiment_comment": comment,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
