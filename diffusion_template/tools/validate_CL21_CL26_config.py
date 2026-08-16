#!/usr/bin/env python3
"""Fail-closed config/spec gate for the six CL19-controlled follow-ups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL21_cosmic_true_soft_router_resididca_v3_24k": ("soft_router", 12, 2348, 224624676),
    "CL22_cosmic_visibility_order_router_24k": ("visibility_order", 12, 2384, 224652396),
    "CL23_cosmic_temporal_frequency_router_24k": ("temporal_frequency", 12, 2240, 219217920),
    "CL24_cosmic_pm_boundary_distill_24k": ("soft_router", 12, 2240, 219217920),
    "CL25_cosmic_low_noise_id_reward_4k": ("soft_router", 2, 2240, 219217920),
    "CL26_cosmic_anchored_highres_roi_ba_24k": ("anchored_roi", 12, 2276, 219217956),
}
ALL_GROUPS = [
    "down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block",
    "up_blocks.0", "up_blocks.1", "up_blocks.2",
]


def selected(config, path):
    value = OmegaConf.select(config, path, default="<missing>")
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True)
    args = parser.parse_args()
    if args.config_name not in ARMS:
        raise RuntimeError(f"Unapproved CL21-CL26 config: {args.config_name}")
    arm = args.config_name.split("_", 1)[0]
    if not args.run_name.startswith(f"{arm}_"):
        raise RuntimeError("Run/config arm mismatch")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
        cl19 = compose(config_name="CL19_cosmic_true_soft_fullquery_router_24k")

    for path in (
        "datasets.val", "dataloaders.manual_val", "validation_args", "pipeline",
        "optimizer", "loss_function",
    ):
        if selected(config, path) != selected(cl19, path):
            raise RuntimeError(f"{path} drifted from the sealed CL19 contract")
    mode, epochs, tensors, parameters = ARMS[args.config_name]
    checks = {
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": epochs,
        "trainer.validation_interval_steps": 2000,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
        "validation_args.num_images_per_prompt": 1,
        "validation_args.num_inference_steps": 50,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "model.ba_architecture_version": "hard_replace_v1",
        "model.branched_attn_weight_mode": "noise_and_ref",
        "model.ba_hardcase_mode": mode,
        "expected_trainable_contract.total_tensors": tensors,
        "expected_trainable_contract.total_parameters": parameters,
        "expected_trainable_contract.optimizer_tensors": tensors,
        "expected_trainable_contract.optimizer_parameters": parameters,
        "train_dataset_name": "cosmic_large_adapted",
    }
    drift = {key: (want, selected(config, key)) for key, want in checks.items() if selected(config, key) != want}
    if drift:
        raise RuntimeError(f"CL21-CL26 fixed-contract drift: {drift}")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("Only the fixed manual_val panel is allowed")

    if arm == "CL21" and not bool(selected(config, "model.ba_residual_identity_ca_v3_enabled")):
        raise RuntimeError("CL21 residual identity CA is disabled")
    if arm in {"CL22", "CL26"}:
        if selected(config, "model.ba_hardcase_groups") != ["up_blocks.0", "up_blocks.1"]:
            raise RuntimeError(f"{arm} must specialize only up0/up1")
        if selected(config, "model.ba_hardcase_fallback_mode") != "soft_router":
            raise RuntimeError(f"{arm} must retain CL19 routing elsewhere")
    if arm == "CL22" and float(selected(config, "datasets.train.cosmic_large_adapted.semantic_occlusion_probability")) != 0.25:
        raise RuntimeError("CL22 synthetic visibility supervision drifted")
    if arm == "CL23" and selected(config, "model.ba_hardcase_groups") != ALL_GROUPS:
        raise RuntimeError("CL23 must schedule every CL19 router")
    if arm == "CL24":
        if not bool(selected(config, "model.ba_pm_boundary_distill_enabled")):
            raise RuntimeError("CL24 boundary teacher is disabled")
        if float(selected(config, "datasets.train.cosmic_large_adapted.semantic_occlusion_probability")) != 0.25:
            raise RuntimeError("CL24 synthetic boundary subset drifted")
    if arm == "CL25":
        if not bool(selected(config, "model.ba_low_noise_id_reward_enabled")):
            raise RuntimeError("CL25 low-noise reward is disabled")
        if selected(config, "loss_kind") != "masked_identity_aux":
            raise RuntimeError("CL25 requires the metric-aligned loss")
        if int(selected(config, "lr_scheduler.total_steps")) != 4000:
            raise RuntimeError("CL25 is a 4k local continuation")
        if int(selected(config, "datasets.train.cosmic_large_adapted.num_identity_refs")) != 3:
            raise RuntimeError("CL25 requires a three-reference identity centroid")
    if arm == "CL26" and not (
        float(selected(config, "model.ba_hardcase_roi_gate_min")) == 0.05
        and float(selected(config, "model.ba_hardcase_roi_gate_init")) == 0.10
        and float(selected(config, "model.ba_hardcase_gate_max")) == 0.25
    ):
        raise RuntimeError("CL26 anchored ROI bounds drifted")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL21_CL26_cl19_followups_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if int(plan.get("gpus", -1)) != 1 or plan.get("machine") != "serv":
        raise RuntimeError("Experiment spec must request one Serv GPU")
    print(json.dumps({
        "status": "ok", "run_name": args.run_name, "config": args.config_name,
        "optimizer_steps": epochs * 2000, "validation_images": 96,
        "trainable_tensors": tensors, "trainable_parameters": parameters,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
