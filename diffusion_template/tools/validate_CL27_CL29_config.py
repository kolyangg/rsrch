#!/usr/bin/env python3
"""Fail-closed composition/spec gate for the three CL23 successors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL27_cosmic_frequency_surface_energy_24k": (2240, 219217920),
    "CL28_cosmic_learnable_frequency_schedule_24k": (2310, 219218130),
    "CL29_cosmic_lowband_causal_contrastive_24k": (2240, 219217920),
    "CL29_cosmic_lowband_causal_contrastive_24k_fixed_pipeline": (2240, 219217920),
}
ALL_GROUPS = [
    "down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block",
    "up_blocks.0", "up_blocks.1", "up_blocks.2",
]


def selected(config, path):
    value = OmegaConf.select(config, path, default="<missing>")
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True)
    args = parser.parse_args()
    if args.config_name not in ARMS:
        raise RuntimeError(f"Unapproved CL27-CL29 config: {args.config_name}")
    arm = args.config_name.split("_", 1)[0]
    if not args.run_name.startswith(f"{arm}_"):
        raise RuntimeError("Run/config arm mismatch")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
        cl23 = compose(config_name="CL23_cosmic_temporal_frequency_router_24k")

    for path in (
        "datasets.val", "dataloaders", "validation_args", "pipeline",
        "optimizer", "loss_function", "lr_scheduler", "inference_metrics",
    ):
        if selected(config, path) != selected(cl23, path):
            raise RuntimeError(f"{path} drifted from the sealed CL23 contract")

    tensors, parameters = ARMS[args.config_name]
    checks = {
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
        "validation_args.num_images_per_prompt": 1,
        "validation_args.num_inference_steps": 50,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "model.ba_architecture_version": "hard_replace_v1",
        "model.branched_attn_weight_mode": "noise_and_ref",
        "model.ba_hardcase_mode": "temporal_frequency",
        "model.ba_hardcase_groups": ALL_GROUPS,
        "model.ba_hardcase_frequency_low_early": 0.50,
        "model.ba_hardcase_frequency_low_late": 0.85,
        "model.ba_hardcase_frequency_high_early": 0.75,
        "model.ba_hardcase_frequency_high_late": 1.25,
        "expected_trainable_contract.total_tensors": tensors,
        "expected_trainable_contract.total_parameters": parameters,
        "expected_trainable_contract.optimizer_tensors": tensors,
        "expected_trainable_contract.optimizer_parameters": parameters,
        "train_dataset_name": "cosmic_large_adapted",
    }
    drift = {
        key: (want, selected(config, key))
        for key, want in checks.items()
        if selected(config, key) != want
    }
    if drift:
        raise RuntimeError(f"CL27-CL29 fixed-contract drift: {drift}")
    expected_branched = (910, 127795410) if arm == "CL28" else (840, 127795200)
    expected_categories = {
        "branched_sa_r128": (".attn1.processor.", *expected_branched),
        "generic_effective_adapter_r32": (".lora_adapter.", 700, 30474240),
        "photomaker_default_effective_adapter_r64": (".default.", 700, 60948480),
    }
    categories = selected(config, "expected_trainable_contract.categories")
    actual_categories = {
        name: (value["name_substring"], value["tensors"], value["parameters"])
        for name, value in categories.items()
    }
    if actual_categories != expected_categories:
        raise RuntimeError(
            f"CL27-CL29 trainable category drift: {actual_categories}"
        )
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("Only the fixed manual_val panel is allowed")
    if any(
        selected(config, key) not in (None, "<missing>")
        for key in ("trainer.from_pretrained", "trainer.resume_from", "saved_checkpoint")
    ):
        raise RuntimeError("CL27-CL29 must be cold starts")

    enabled = {
        "CL27": selected(config, "model.ba_frequency_surface_loss_enabled") is True,
        "CL28": selected(config, "model.ba_frequency_learnable_schedule_enabled") is True,
        "CL29": selected(config, "model.ba_frequency_lowband_contrastive_enabled") is True,
    }
    if enabled != {"CL27": arm == "CL27", "CL28": arm == "CL28", "CL29": arm == "CL29"}:
        raise RuntimeError(f"Successor toggle isolation failed: {enabled}")
    if arm == "CL27":
        if selected(config, "model.ba_frequency_surface_loss_groups") != ["up_blocks.0", "up_blocks.1"]:
            raise RuntimeError("CL27 loss groups drifted")
        if float(selected(config, "datasets.train.cosmic_large_adapted.semantic_occlusion_probability")) != 0.25:
            raise RuntimeError("CL27 synthetic-occlusion probability drifted")
    elif arm == "CL28":
        if bool(selected(config, "model.ba_frequency_learnable_low_early")):
            raise RuntimeError("CL28 must keep low-early fixed")
        endpoints = [
            selected(config, "model.ba_frequency_low_late_center"),
            selected(config, "model.ba_frequency_high_early_center"),
            selected(config, "model.ba_frequency_high_late_center"),
        ]
        if endpoints != [0.85, 0.75, 1.25]:
            raise RuntimeError("CL28 schedule centers drifted")
        telemetry_groups = ("down1", "down2", "mid", "up0", "up1")
        expected_frequency_metrics = {
            f"ba/frequency_{band}_scale/{group}"
            for band in ("low", "high")
            for group in telemetry_groups
        }
        actual_frequency_metrics = {
            name
            for name in selected(config, "writer.loss_names")
            if str(name).startswith("ba/frequency_") and "_scale/" in str(name)
        }
        if actual_frequency_metrics != expected_frequency_metrics:
            raise RuntimeError(
                "CL28 telemetry must name only installed SDXL attention groups: "
                f"{sorted(actual_frequency_metrics)}"
            )
    else:
        if selected(config, "model.ba_frequency_lowband_contrastive_groups") != ["mid_block", "up_blocks.0", "up_blocks.1"]:
            raise RuntimeError("CL29 contrastive groups drifted")
        if not bool(selected(config, "datasets.train.cosmic_large_adapted.same_identity_dual_reference")):
            raise RuntimeError("CL29 dual-reference sampling is disabled")
        if int(selected(config, "datasets.train.cosmic_large_adapted.min_reference_candidates_for_target")) != 3:
            raise RuntimeError("CL29 requires three distinct reference candidates")
        if bool(selected(config, "model.ba_crossview_consistency_enabled")):
            raise RuntimeError("CL29 must not reuse CL18 prediction consistency")
        if args.config_name.endswith("_fixed_pipeline"):
            optimized = {
                "model.ba_hardcase_telemetry_enabled": False,
                "model.ba_frequency_lowband_sample_on_cpu": True,
                "trainer.active_grad_norm_mode": "requested_only",
                "trainer.skip_initial_validation": False,
            }
            drift = {
                key: (want, selected(config, key))
                for key, want in optimized.items()
                if selected(config, key) != want
            }
            if drift:
                raise RuntimeError(f"CL29 optimized-pipeline drift: {drift}")

    baseline = flatten(OmegaConf.to_container(cl23, resolve=True))
    candidate = flatten(OmegaConf.to_container(config, resolve=True))
    changed = {
        key for key in set(baseline) | set(candidate)
        if baseline.get(key, "<missing>") != candidate.get(key, "<missing>")
    }
    allowed_prefixes = {
        "CL27": (
            "model.ba_frequency_surface_",
            "datasets.train.cosmic_large_adapted.semantic_occlusion_",
        ),
        "CL28": ("model.ba_frequency_",),
        "CL29": (
            "model.ba_frequency_lowband_",
            "datasets.train.cosmic_large_adapted.same_identity_dual_reference",
            "datasets.train.cosmic_large_adapted.min_reference_candidates_for_target",
        ),
    }[arm] + ("expected_trainable_contract.", "writer.")
    allowed_exact = set()
    if args.config_name.endswith("_fixed_pipeline"):
        allowed_exact.update({
            "model.ba_hardcase_telemetry_enabled",
            "trainer.active_grad_norm_mode",
        })
    unexpected = sorted(
        key for key in changed
        if key not in allowed_exact and not key.startswith(allowed_prefixes)
    )
    if unexpected:
        raise RuntimeError(f"Unexpected CL23 diff paths: {unexpected}")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL27_CL29_cl23_followups_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if int(plan.get("gpus", -1)) != 1 or plan.get("machine") != "serv":
        raise RuntimeError("Experiment spec must request one Serv GPU")
    print(json.dumps({
        "status": "ok", "run_name": args.run_name, "config": args.config_name,
        "optimizer_steps": 24000, "validation_images": 96,
        "trainable_tensors": tensors, "trainable_parameters": parameters,
        "changed_paths": len(changed),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
