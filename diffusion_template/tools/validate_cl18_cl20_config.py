#!/usr/bin/env python3
"""Fail-closed composition gate for the clean CL18, CL19 and CL20 recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.model.photomaker_branched.e13_contract import normalise_e13_settings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL18_cosmic_crossview_spatial_consistency_24k": (
        "off", "cosmic_large_adapted"
    ),
    "CL19_cosmic_true_soft_fullquery_router_24k": (
        "soft_router", "cosmic_large_adapted"
    ),
    "CL20_cosmic_bigcelebs_hardcase_curriculum_24k": (
        "off", "cl20_hardcase_curriculum"
    ),
}
CL19_GROUPS = [
    "down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block",
    "up_blocks.0", "up_blocks.1", "up_blocks.2",
]


def value(config, path):
    selected = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


def require(config, path, expected):
    actual = value(config, path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, choices=sorted(ARMS))
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
        cl14 = compose(config_name="CL14_cosmic_joint_shadow_sa128_softmask_24k")

    expected_mode, expected_dataset = ARMS[args.config_name]
    fixed = {
        "train_dataset_name": expected_dataset,
        "pipeline._target_": (
            "src.pipelines.photomaker_branched_subject_v2."
            "PhotomakerBranchedSubjectV2Pipeline.from_pretrained"
        ),
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "validation_args.face_subject_selection_policy": "bbox_overlap_v2",
        "validation_args.num_images_per_prompt": 1,
        "validation_args.num_inference_steps": 50,
        "validation_args.guidance_scale": 5,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "dataloaders.train.batch_size": 2,
        "dataloaders.manual_val.batch_size": 12,
        "datasets.val.manual_val.limit": 96,
        "datasets.val.manual_val.bbox_mask_gen": (
            "../dataset_full/val_dataset/protocols/cl14/pm96_bboxes_new_auto.json"
        ),
    }
    for path, expected in fixed.items():
        require(config, path, expected)

    if value(config, "inference_metrics") != [
        "clip_ts", "id_sim_best_legacy", "id_sim_subject_v2"
    ]:
        raise RuntimeError("Subject-v2 metric set drifted")
    if value(config, "writer.loss_names") != [
        "loss", "loss_ba_aux", "loss_ba_crossview"
    ]:
        raise RuntimeError("CL18-CL20 loss telemetry drifted")
    if value(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("CL18-CL20 must use only fixed manual_val")
    # All unchanged validation/generation values remain inherited from CL14.
    for path in (
        "dataloaders.manual_val",
        "optimizer",
        "lr_scheduler",
        "loss_function",
        "pretrained_model_for_validation_name_or_path",
    ):
        if value(config, path) != value(cl14, path):
            raise RuntimeError(f"{path} drifted from CL14")

    arm = args.config_name.split("_", 1)[0]
    settings = normalise_e13_settings(config.model.e13_settings)
    if settings["ba_training_mask_feather"] != 2:
        raise RuntimeError("CL18-CL20 training-mask feather drifted")
    if settings["ba_hardcase_mode"] != expected_mode:
        raise RuntimeError("CL18-CL20 hard-case route drifted")
    crossview = settings["ba_crossview_consistency_enabled"]
    if arm == "CL18":
        if settings["ba_crossview_consistency_probability"] != 0.25:
            raise RuntimeError("CL18 cross-view probability drifted")
        if settings["ba_crossview_consistency_weight"] != 0.05:
            raise RuntimeError("CL18 cross-view weight drifted")
        require(
            config,
            "datasets.train.cosmic_large_adapted.same_identity_dual_reference",
            True,
        )
        if not crossview:
            raise RuntimeError("CL18 consistency is disabled")
    elif crossview:
        raise RuntimeError(f"{arm} must not enable CL18 consistency")

    if arm == "CL19":
        if list(settings["ba_hardcase_groups"]) != CL19_GROUPS:
            raise RuntimeError("CL19 processor groups drifted")
        if settings["ba_hardcase_transition_cells"] != 2:
            raise RuntimeError("CL19 transition width drifted")
    elif settings["ba_hardcase_groups"]:
        raise RuntimeError(f"{arm} unexpectedly selected hard-case groups")

    if arm == "CL20":
        require(config, "train_dataloader_shuffle", False)
        require(config, "datasets.train.cl20_hardcase_curriculum.expected_rows", 48000)

    print(json.dumps({
        "status": "ok",
        "config": args.config_name,
        "mode": expected_mode,
        "dataset": expected_dataset,
        "optimizer_steps": 24000,
        "validation_images": 96,
        "trainable_tensors": 2240,
        "trainable_parameters": 219217920,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
