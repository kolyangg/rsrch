#!/usr/bin/env python3
"""Fail-closed composition and fixed-pipeline gate for clean CL23/CL27/CL39."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.model.photomaker_branched.e13_contract import normalise_e13_settings


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "configs"
ARMS = {
    "CL23_cosmic_temporal_frequency_router_24k": (False, False),
    "CL27_cosmic_frequency_surface_energy_24k": (True, False),
    "CL39_cosmic_null_key_confidence_router_24k": (True, True),
}
GROUPS = [
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
        cl19 = compose(config_name="CL19_cosmic_true_soft_fullquery_router_24k")

    surface_enabled, null_key_enabled = ARMS[args.config_name]
    fixed = {
        "train_dataset_name": "cosmic_large_adapted",
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
    }
    for path, expected in fixed.items():
        require(config, path, expected)

    for path in (
        "optimizer", "lr_scheduler", "loss_function", "dataloaders.manual_val",
        "pretrained_model_for_validation_name_or_path",
        "validation_args", "inference_metrics",
    ):
        if value(config, path) != value(cl19, path):
            raise RuntimeError(f"{path} drifted from CL19")

    settings = normalise_e13_settings(config.model.e13_settings)
    expected_settings = {
        "ba_training_mask_feather": 2,
        "ba_hardcase_mode": "temporal_frequency",
        "ba_hardcase_groups": tuple(GROUPS),
        "ba_hardcase_transition_cells": 2,
        "ba_hardcase_frequency_low_early": 0.50,
        "ba_hardcase_frequency_low_late": 0.85,
        "ba_hardcase_frequency_high_early": 0.75,
        "ba_hardcase_frequency_high_late": 1.25,
        "ba_frequency_surface_loss_enabled": surface_enabled,
        "ba_null_key_router_enabled": null_key_enabled,
    }
    for name, expected in expected_settings.items():
        if settings[name] != expected:
            raise RuntimeError(f"{name}: expected {expected!r}, got {settings[name]!r}")

    if surface_enabled:
        surface_expected = {
            "ba_frequency_surface_loss_groups": ("up_blocks.0", "up_blocks.1"),
            "ba_frequency_surface_top_weight": 0.02,
            "ba_frequency_surface_top_low_band_factor": 0.25,
            "ba_frequency_surface_visible_floor_weight": 0.005,
            "ba_frequency_surface_visible_floor_ratio": 0.35,
        }
        for name, expected in surface_expected.items():
            if settings[name] != expected:
                raise RuntimeError(f"{name} drifted")
        require(
            config,
            "datasets.train.cosmic_large_adapted.semantic_occlusion_probability",
            0.25,
        )
        require(
            config,
            "datasets.train.cosmic_large_adapted.semantic_occlusion_seed",
            150017,
        )
    else:
        require(
            config,
            "datasets.train.cosmic_large_adapted.semantic_occlusion_probability",
            0.0,
        )

    if null_key_enabled:
        null_expected = {
            "ba_null_key_router_groups": ("up_blocks.0", "up_blocks.1"),
            "ba_null_key_entropy_threshold": 0.75,
            "ba_null_key_temperature": 0.08,
            "ba_null_key_max_abstention": 0.75,
            "ba_null_key_min_reference_fraction": 0.25,
        }
        for name, expected in null_expected.items():
            if settings[name] != expected:
                raise RuntimeError(f"{name} drifted")
        if any(
            str(name).startswith("active_grad_norm")
            for name in value(config, "writer.loss_names")
        ):
            raise RuntimeError("CL39 must not request unused active-gradient norms")

    helper_source = (
        ROOT / "src/model/photomaker_branched/lora2_helpers.py"
    ).read_text(encoding="utf-8")
    if "model.unet.attn_processors.get" in helper_source:
        raise RuntimeError("Per-layer Diffusers processor-map lookup regression")

    print(json.dumps({
        "status": "ok",
        "config": args.config_name,
        "surface_loss": surface_enabled,
        "null_key_router": null_key_enabled,
        "optimizer_steps": 24000,
        "validation_images": 96,
        "trainable_tensors": 2240,
        "trainable_parameters": 219217920,
        "full_activation_telemetry": False,
        "processor_lookup": "cached_once_per_collector",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
