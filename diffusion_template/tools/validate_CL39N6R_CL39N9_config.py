#!/usr/bin/env python3
"""Fail-closed contract gate for the four post-CL39 architecture leaves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "configs"
ARMS = {
    "CL39N6R_cosmic_up1_low_pruned_24k": (2240, 219217920),
    "CL39N7_cosmic_posterior_null_router_24k": (2240, 219217920),
    "CL39N8_cosmic_native_orthogonal_highband_24k": (2240, 219217920),
    "CL39N9_cosmic_intrinsic_id_sidecar_24k": (2497, 238261284),
}


def value(config, path, default="<missing>"):
    result = OmegaConf.select(config, path, default=default)
    return OmegaConf.to_container(result, resolve=True) if OmegaConf.is_config(result) else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True, type=Path)
    args = parser.parse_args()
    if args.config_name not in ARMS:
        raise RuntimeError(f"Unapproved config: {args.config_name}")
    arm = args.config_name.split("_", 1)[0]
    if not args.run_name.startswith(f"{arm}_"):
        raise RuntimeError("Run/config arm mismatch")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
    tensors, parameters = ARMS[args.config_name]
    checks = {
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.skip_initial_validation": False,
        "trainer.preserve_training_model_during_validation": True,
        "trainer.active_grad_norm_mode": "requested_only",
        "datasets.val.manual_val.limit": 96,
        "trainer.face_quality.expected_images": 96,
        "validation_args.num_inference_steps": 50,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "model.pose_adapt_ratio": 0.0,
        "model.ca_mixing_for_face": False,
        "model.ba_hardcase_mode": "temporal_frequency",
        "model.ba_frequency_surface_loss_enabled": True,
        "model.ba_null_key_router_enabled": True,
        "model.ba_null_key_router_groups": ["up_blocks.0", "up_blocks.1"],
        "expected_trainable_contract.total_tensors": tensors,
        "expected_trainable_contract.total_parameters": parameters,
        "expected_trainable_contract.optimizer_tensors": tensors,
        "expected_trainable_contract.optimizer_parameters": parameters,
        "datasets.train.cosmic_large_adapted.semantic_occlusion_probability": 0.25,
        "dataloaders.train.batch_size": 2,
        "trainer.seed": 0,
    }
    drift = {key: (want, value(config, key)) for key, want in checks.items()
             if value(config, key) != want}
    if drift:
        raise RuntimeError(f"Fixed CL39 contract drift: {drift}")

    settings = value(config, "model.cl39x_settings", {})
    active = sorted(key for key, enabled in settings.items()
                    if key.endswith("_enabled") and enabled)
    if arm in {"CL39N6R", "CL39N7"} and active:
        raise RuntimeError(f"{arm} must not enable a CL39-X arm: {active}")
    if arm == "CL39N6R":
        map_path = ROOT / value(config, "model.ba_group_band_map_path")
        digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
        if digest != value(config, "model.ba_group_band_map_sha256"):
            raise RuntimeError("N6R map hash mismatch")
        route = json.loads(map_path.read_text(encoding="utf-8"))["groups"]
        disabled = [(group, band) for group, bands in route.items()
                    for band, enabled in bands.items() if enabled == 0]
        if disabled != [("up_blocks.1", "low")]:
            raise RuntimeError(f"Invalid N6R map: {disabled}")
    elif arm == "CL39N7":
        if value(config, "model.ba_null_key_confidence_mode") != "posterior_invalid_mass_v1":
            raise RuntimeError("N7 posterior mode is absent")
    elif arm == "CL39N8":
        if active != ["ba_native_orthogonal_band_enabled"]:
            raise RuntimeError(f"Invalid N8 leaf: {active}")
    elif arm == "CL39N9":
        if active != ["ba_intrinsic_id_sidecar_enabled"]:
            raise RuntimeError(f"Invalid N9 leaf: {active}")
        if not (
            settings["ba_intrinsic_id_projector_hidden"] == 1024
            and settings["ba_intrinsic_id_residual_rank"] == 32
            and settings["ba_intrinsic_id_gate_max"] == 0.10
            and settings["ba_intrinsic_id_confidence_source"]
            == "cl39_complement_detached"
        ):
            raise RuntimeError("N9 sidecar contract drift")

    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment run-name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL39N_qualified_production_1gpu.sh":
        raise RuntimeError("Experiment launcher mismatch")
    if plan.get("production_launcher") != "launchers/active/run_CL39N6R_CL39N9_1gpu.sh":
        raise RuntimeError("Experiment production launcher mismatch")
    if plan.get("machine") != "serv" or plan.get("gpus") != 1:
        raise RuntimeError("Each experiment must request one Serv GPU")
    print(json.dumps({"status": "ok", "arm": arm, "steps": 24000,
                      "trainable_tensors": tensors,
                      "trainable_parameters": parameters}, indent=2))


if __name__ == "__main__":
    main()
