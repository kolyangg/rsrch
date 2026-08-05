#!/usr/bin/env python3
"""Fail-closed composition gate for the August Large Dataset hard-BA suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DELTAS = {
    "E0_large_ds_base_historical_20k": {},
    "E0_large_ds_base_fixed_20k": {},
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
    "E7_large_ds_generic_effective_20k": {
        "model.generic_adapter_train_scope": "effective_all",
    },
    "E8_large_ds_generic_ca_20k": {
        "model.generic_adapter_train_scope": "cross_attention",
    },
    "E9_large_ds_shared_saout_20k": {
        "model.generic_adapter_train_scope": "self_attention_output",
    },
    "E10_large_ds_pmdefault_effective_20k": {
        "model.photomaker_default_train_scope": "effective_all",
    },
    "E11_large_ds_ba_sa_r128_20k": {
        "model.ba_hard_v1_lora_rank": 128,
    },
    "E12_large_ds_ba_idca_up_r256_20k": {
        "model.ba_identity_ca_v2_enabled": True,
        "model.ba_identity_ca_v2_groups": ["up_blocks.0", "up_blocks.1"],
        "model.ba_identity_ca_v2_rank": 256,
    },
}

CONFIG_RUN_NAMES = {
    "E0_large_ds_base_historical_20k": (
        "E0_large_ds_base_historical_r4_20k_full96_r1"
    ),
    "E0_large_ds_base_fixed_20k": (
        "E0_large_ds_base_fixed_baonly_r32_20k_full96_r1"
    ),
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
    "E7_large_ds_generic_effective_20k": (
        "E7_large_ds_generic_effective_r32_20k_full96_r1"
    ),
    "E8_large_ds_generic_ca_20k": (
        "E8_large_ds_generic_ca_r32_20k_full96_r1"
    ),
    "E9_large_ds_shared_saout_20k": (
        "E9_large_ds_shared_saout_r32_20k_full96_r1"
    ),
    "E10_large_ds_pmdefault_effective_20k": (
        "E10_large_ds_pmdefault_effective_r64_20k_full96_r1"
    ),
    "E11_large_ds_ba_sa_r128_20k": (
        "E11_large_ds_ba_sa_r128_20k_full96_r1"
    ),
    "E12_large_ds_ba_idca_up_r256_20k": (
        "E12_large_ds_ba_idca_up_r256_20k_full96_r1"
    ),
}

CONFIG_BASELINES = {
    "E0_large_ds_base_historical_20k": (
        "large_dataset_rhca_historical_observed_20k"
    ),
}

SCIENTIFIC_BASE = {
    "model.ba_hard_v1_true_reference_key_mask": False,
    "model.ba_hard_v1_branch_output_rank": None,
    "model.ba_hard_v1_reference_roi_warp": False,
    "model.ba_self_attention_groups": None,
    "model.ba_training_timestep_policy": "uniform_all",
    "model.branched_trainable_dtype": "inherit",
    "model.generic_adapter_train_scope": "none",
    "model.photomaker_default_train_scope": "none",
    "model.ba_hard_v1_lora_rank": None,
    "model.ba_identity_ca_v2_enabled": False,
    "model.ba_identity_ca_v2_groups": None,
    "model.ba_identity_ca_v2_rank": 16,
}

COMMON_BA_CONTRACT = {
    "expected_trainable_contract.enabled": True,
    "expected_trainable_contract.categories.branched_processors.name_substring": (
        ".processor."
    ),
    "expected_trainable_contract.categories.branched_processors.tensors": 840,
    "expected_trainable_contract.categories.branched_processors.parameters": 31948800,
}

CONFIG_AUDIT_DELTAS = {
    "E7_large_ds_generic_effective_20k": {
        **COMMON_BA_CONTRACT,
        "expected_trainable_contract.total_tensors": 1540,
        "expected_trainable_contract.total_parameters": 62423040,
        "expected_trainable_contract.optimizer_tensors": 1540,
        "expected_trainable_contract.optimizer_parameters": 62423040,
        "expected_trainable_contract.categories.generic_effective_adapter.name_substring": (
            ".lora_adapter."
        ),
        "expected_trainable_contract.categories.generic_effective_adapter.tensors": 700,
        "expected_trainable_contract.categories.generic_effective_adapter.parameters": 30474240,
    },
    "E8_large_ds_generic_ca_20k": {
        **COMMON_BA_CONTRACT,
        "expected_trainable_contract.total_tensors": 1400,
        "expected_trainable_contract.total_parameters": 57098240,
        "expected_trainable_contract.optimizer_tensors": 1400,
        "expected_trainable_contract.optimizer_parameters": 57098240,
        "expected_trainable_contract.categories.generic_cross_attention_adapter.name_substring": (
            ".lora_adapter."
        ),
        "expected_trainable_contract.categories.generic_cross_attention_adapter.tensors": 560,
        "expected_trainable_contract.categories.generic_cross_attention_adapter.parameters": 25149440,
    },
    "E9_large_ds_shared_saout_20k": {
        **COMMON_BA_CONTRACT,
        "expected_trainable_contract.total_tensors": 980,
        "expected_trainable_contract.total_parameters": 37273600,
        "expected_trainable_contract.optimizer_tensors": 980,
        "expected_trainable_contract.optimizer_parameters": 37273600,
        "expected_trainable_contract.categories.generic_shared_sa_output_adapter.name_substring": (
            ".lora_adapter."
        ),
        "expected_trainable_contract.categories.generic_shared_sa_output_adapter.tensors": 140,
        "expected_trainable_contract.categories.generic_shared_sa_output_adapter.parameters": 5324800,
    },
    "E10_large_ds_pmdefault_effective_20k": {
        **COMMON_BA_CONTRACT,
        "expected_trainable_contract.total_tensors": 1540,
        "expected_trainable_contract.total_parameters": 92897280,
        "expected_trainable_contract.optimizer_tensors": 1540,
        "expected_trainable_contract.optimizer_parameters": 92897280,
        "expected_trainable_contract.categories.photomaker_default_effective_adapter.name_substring": (
            ".default."
        ),
        "expected_trainable_contract.categories.photomaker_default_effective_adapter.tensors": 700,
        "expected_trainable_contract.categories.photomaker_default_effective_adapter.parameters": 60948480,
    },
    "E11_large_ds_ba_sa_r128_20k": {
        "expected_trainable_contract.enabled": True,
        "expected_trainable_contract.total_tensors": 840,
        "expected_trainable_contract.total_parameters": 127795200,
        "expected_trainable_contract.optimizer_tensors": 840,
        "expected_trainable_contract.optimizer_parameters": 127795200,
        "expected_trainable_contract.categories.branched_sa_r128.name_substring": (
            ".attn1.processor."
        ),
        "expected_trainable_contract.categories.branched_sa_r128.tensors": 840,
        "expected_trainable_contract.categories.branched_sa_r128.parameters": 127795200,
    },
    "E12_large_ds_ba_idca_up_r256_20k": {
        "expected_trainable_contract.enabled": True,
        "expected_trainable_contract.total_tensors": 1128,
        "expected_trainable_contract.total_parameters": 134578176,
        "expected_trainable_contract.optimizer_tensors": 1128,
        "expected_trainable_contract.optimizer_parameters": 134578176,
        "expected_trainable_contract.categories.branched_sa_r32.name_substring": (
            ".attn1.processor."
        ),
        "expected_trainable_contract.categories.branched_sa_r32.tensors": 840,
        "expected_trainable_contract.categories.branched_sa_r32.parameters": 31948800,
        "expected_trainable_contract.categories.corrected_identity_ca_r256.name_substring": (
            ".attn2.processor."
        ),
        "expected_trainable_contract.categories.corrected_identity_ca_r256.tensors": 288,
        "expected_trainable_contract.categories.corrected_identity_ca_r256.parameters": 102629376,
    },
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

HISTORICAL_FIXED_OVERRIDES = {
    "model.ba_enforce_reference_only_hard_route": False,
    "model.strict_branched_install": False,
    "model.strict_trainable_contract": False,
    "model.branched_state_dict_mode": "legacy",
    "strict_validation_processor_copy": False,
    "expected_trainable_contract.enabled": True,
    "expected_trainable_contract.total_tensors": 3080,
    "expected_trainable_contract.total_parameters": 171294720,
    "expected_trainable_contract.optimizer_tensors": 3080,
    "expected_trainable_contract.optimizer_parameters": 171294720,
    "expected_trainable_contract.categories.branched_processors.name_substring": (
        ".processor."
    ),
    "expected_trainable_contract.categories.branched_processors.tensors": 840,
    "expected_trainable_contract.categories.branched_processors.parameters": 31948800,
    "expected_trainable_contract.categories.generic_lora_adapter.name_substring": (
        ".lora_adapter."
    ),
    "expected_trainable_contract.categories.generic_lora_adapter.tensors": 1120,
    "expected_trainable_contract.categories.generic_lora_adapter.parameters": 46448640,
    "expected_trainable_contract.categories.photomaker_default_adapter.name_substring": (
        ".default."
    ),
    "expected_trainable_contract.categories.photomaker_default_adapter.tensors": 1120,
    "expected_trainable_contract.categories.photomaker_default_adapter.parameters": 92897280,
}

E0_PAIR_ALLOWED_DIFFERENCES = {
    "expected_trainable_contract.enabled",
    "expected_trainable_contract.total_tensors",
    "expected_trainable_contract.total_parameters",
    "expected_trainable_contract.optimizer_tensors",
    "expected_trainable_contract.optimizer_parameters",
    "expected_trainable_contract.categories.branched_processors.name_substring",
    "expected_trainable_contract.categories.branched_processors.tensors",
    "expected_trainable_contract.categories.branched_processors.parameters",
    "expected_trainable_contract.categories.generic_lora_adapter.name_substring",
    "expected_trainable_contract.categories.generic_lora_adapter.tensors",
    "expected_trainable_contract.categories.generic_lora_adapter.parameters",
    "expected_trainable_contract.categories.photomaker_default_adapter.name_substring",
    "expected_trainable_contract.categories.photomaker_default_adapter.tensors",
    "expected_trainable_contract.categories.photomaker_default_adapter.parameters",
    "model.ba_enforce_reference_only_hard_route",
    "model.branched_state_dict_mode",
    "model.strict_branched_install",
    "model.strict_trainable_contract",
    "strict_validation_processor_copy",
    "writer.experiment_comment",
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
            config_name=CONFIG_BASELINES.get(
                args.config_name,
                "large_dataset_rhca_hard_v1_audited_20k",
            ),
            overrides=["writer=cometml"],
        )
        if args.config_name.startswith("E0_large_ds_base_"):
            historical_e0 = compose(
                config_name="E0_large_ds_base_historical_20k",
                overrides=["writer=cometml"],
            )
            fixed_e0 = compose(
                config_name="E0_large_ds_base_fixed_20k",
                overrides=["writer=cometml"],
            )
        else:
            historical_e0 = None
            fixed_e0 = None

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
        *CONFIG_AUDIT_DELTAS.get(args.config_name, {}),
        "writer.experiment_comment",
    }
    if changed_paths != expected_changed_paths:
        raise RuntimeError(
            "Child config does not have exactly its approved scientific delta set: "
            f"expected={sorted(expected_changed_paths)}, "
            f"actual={sorted(changed_paths)}"
        )

    if historical_e0 is not None and fixed_e0 is not None:
        historical_leaves = flatten_leaves(
            OmegaConf.to_container(historical_e0, resolve=True)
        )
        fixed_leaves = flatten_leaves(
            OmegaConf.to_container(fixed_e0, resolve=True)
        )
        pair_changed_paths = {
            path
            for path in set(historical_leaves) | set(fixed_leaves)
            if historical_leaves.get(path, "<missing>")
            != fixed_leaves.get(path, "<missing>")
        }
        if pair_changed_paths != E0_PAIR_ALLOWED_DIFFERENCES:
            raise RuntimeError(
                "E0 historical/fixed pair drifted beyond ownership/checkpoint "
                "corrections: "
                f"expected={sorted(E0_PAIR_ALLOWED_DIFFERENCES)}, "
                f"actual={sorted(pair_changed_paths)}"
            )

    fixed_values = dict(FIXED_VALUES)
    if args.config_name == "E0_large_ds_base_historical_20k":
        fixed_values.update(HISTORICAL_FIXED_OVERRIDES)
    for path, expected in fixed_values.items():
        require_equal(config, path, expected)

    expected_scientific = dict(SCIENTIFIC_BASE)
    expected_scientific.update(CONFIG_DELTAS[args.config_name])
    for path, expected in expected_scientific.items():
        require_equal(config, path, expected)
    for path, expected in CONFIG_AUDIT_DELTAS.get(args.config_name, {}).items():
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
