#!/usr/bin/env python3
"""Fail-closed config/spec gate for the CL15-CL20 hard-case suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL15_cosmic_shared_highres_roi_ba_24k": ("highres_roi", "cosmic_large_adapted"),
    "CL16_cosmic_clean_multiscale_ref_memory_24k": ("clean_memory", "cosmic_large_adapted"),
    "CL17_cosmic_semantic_visibility_ownership_24k": ("semantic_ownership", "cosmic_large_adapted"),
    "CL18_cosmic_crossview_spatial_consistency_24k": ("off", "cosmic_large_adapted"),
    "CL19_cosmic_true_soft_fullquery_router_24k": ("soft_router", "cosmic_large_adapted"),
    "CL20_cosmic_bigcelebs_hardcase_curriculum_24k": ("off", "cl20_hardcase_curriculum"),
}


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
        raise RuntimeError(f"Unapproved CL15-CL20 config: {args.config_name}")
    arm = args.config_name.split("_", 1)[0]
    if not args.run_name.startswith(f"{arm}_"):
        raise RuntimeError("Run/config arm mismatch")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
        cl14 = compose(config_name="CL14_cosmic_joint_shadow_sa128_softmask_24k")

    # These sections define comparability and must remain identical to CL14.
    for path in (
        "datasets.val",
        "dataloaders.manual_val",
        "validation_args",
        "pipeline",
        "optimizer",
        "lr_scheduler",
        "loss_function",
    ):
        if selected(config, path) != selected(cl14, path):
            raise RuntimeError(f"{path} drifted from the sealed CL14 contract")

    expected_mode, expected_dataset = ARMS[args.config_name]
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
        "model.ba_hard_v1_true_reference_key_mask": False,
        "model.ba_training_mask_feather": 2,
        "model.ba_hardcase_mode": expected_mode,
        "train_dataset_name": expected_dataset,
    }
    drift = {
        key: (expected, selected(config, key))
        for key, expected in checks.items()
        if selected(config, key) != expected
    }
    if drift:
        raise RuntimeError(f"CL15-CL20 fixed-contract drift: {drift}")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("CL15-CL20 must use only fixed manual_val")
    if expected_mode != "off" and not selected(config, "model.ba_hardcase_groups"):
        raise RuntimeError("Hard-case route selected no U-Net groups")

    if arm == "CL15":
        if selected(config, "model.ba_hardcase_groups") != [
            "up_blocks.0", "up_blocks.1"
        ]:
            raise RuntimeError("CL15 must use only up0/up1 ROI processors")
        if (
            int(selected(config, "model.ba_hardcase_roi_size")) != 32
            or int(selected(config, "model.ba_hardcase_face_threshold_px")) != 256
        ):
            raise RuntimeError("CL15 must use the sealed ROI32/256px route")
    if arm == "CL16":
        if selected(config, "model.ba_hardcase_groups") != [
            "mid_block", "up_blocks.0", "up_blocks.1"
        ] or int(selected(config, "model.ba_hardcase_rank")) != 64:
            raise RuntimeError("CL16 clean memory must use mid/up0/up1 at rank 64")
    if arm == "CL17":
        if selected(config, "model.ba_hardcase_groups") != [
            "up_blocks.0", "up_blocks.1"
        ]:
            raise RuntimeError("CL17 must use only up0/up1 ownership processors")
        probability = float(selected(
            config,
            "datasets.train.cosmic_large_adapted.semantic_occlusion_probability",
        ))
        if probability != 0.25:
            raise RuntimeError("CL17 requires the sealed 25% synthetic supervision")
    if arm == "CL18":
        if not bool(selected(config, "model.ba_crossview_consistency_enabled")):
            raise RuntimeError("CL18 consistency is disabled")
        if (
            float(selected(config, "model.ba_crossview_consistency_probability"))
            != 0.25
            or float(selected(config, "model.ba_crossview_consistency_weight"))
            != 0.05
        ):
            raise RuntimeError("CL18 consistency cadence/weight drifted")
        if not bool(selected(
            config,
            "datasets.train.cosmic_large_adapted.same_identity_dual_reference",
        )):
            raise RuntimeError("CL18 dataset does not return two spatial refs")
    elif bool(selected(config, "model.ba_crossview_consistency_enabled")):
        raise RuntimeError(f"{arm} must not enable CL18 consistency")
    if arm == "CL19":
        expected_groups = [
            "down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block",
            "up_blocks.0", "up_blocks.1", "up_blocks.2",
        ]
        if (
            selected(config, "model.ba_hardcase_groups") != expected_groups
            or int(selected(config, "model.ba_hardcase_transition_cells")) != 2
        ):
            raise RuntimeError("CL19 full-U-Net two-cell router drifted")
    if arm == "CL20":
        if bool(selected(config, "train_dataloader_shuffle")):
            raise RuntimeError("CL20 sealed row order cannot be shuffled")
        if expected_mode != "off":
            raise RuntimeError("CL20 must keep the CL14 model")

    spec_path = Path(args.experiment_spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL15_CL20_hardcases_24k_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if int(plan.get("gpus", -1)) != 1 or plan.get("machine") != "serv":
        raise RuntimeError("Experiment spec must request one Serv GPU")
    print(json.dumps({
        "status": "ok",
        "run_name": args.run_name,
        "config": args.config_name,
        "mode": expected_mode,
        "dataset": expected_dataset,
        "optimizer_steps": 24000,
        "validation_images": 96,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
