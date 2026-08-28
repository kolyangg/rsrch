#!/usr/bin/env python3
"""Fail-closed Hydra/spec gate for the eight one-change CL39 children."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "configs"
ARMS = {
    "CL39X01_cosmic_valid_key_attention_24k": ("ba_valid_kv_enabled", 2240, 219217920),
    "CL39X02_cosmic_cycle_confidence_24k": ("ba_cycle_confidence_enabled", 2240, 219217920),
    "CL39X03_cosmic_stage_split_ot_transport_24k": ("ba_ot_transport_enabled", 2240, 219217920),
    "CL39X04_cosmic_small_face_roi_route_24k": ("ba_roi_route_enabled", 2276, 219217956),
    "CL39X05_cosmic_automask_os_24k": ("ba_automask_os_enabled", 2240, 219217920),
    "CL39X06_cosmic_counterfactual_reference_24k": ("ba_counterfactual_enabled", 2240, 219217920),
    "CL39X07_cosmic_intrinsic_id_sidecar_24k": ("ba_intrinsic_id_sidecar_enabled", 2353, 242456612),
    "CL39X08_cosmic_global_local_balance_24k": ("ba_global_local_enabled", 2240, 219217920),
}


def selected(config, path):
    value = OmegaConf.select(config, path, default="<missing>")
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True, type=Path)
    args = parser.parse_args()
    if args.config_name not in ARMS:
        raise RuntimeError(f"Unapproved CL39-X config: {args.config_name}")
    arm = args.config_name.split("_", 1)[0]
    if not args.run_name.startswith(f"{arm}_"):
        raise RuntimeError("Run/config arm mismatch")
    raw = yaml.safe_load((CONFIG_DIR / f"{args.config_name}.yaml").read_text())
    defaults = raw.get("defaults") or []
    if defaults[0] != "CL39_cosmic_null_key_confidence_router_24k":
        raise RuntimeError("Every leaf must default directly from CL39")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
    enabled_key, tensors, parameters = ARMS[args.config_name]
    settings = selected(config, "model.cl39x_settings")
    active = sorted(key for key, value in settings.items() if key.endswith("_enabled") and value)
    if active != [enabled_key]:
        raise RuntimeError(f"Expected only {enabled_key}, got {active}")
    checks = {
        "trainer.epoch_len": 2000, "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.skip_initial_validation": False,
        "trainer.active_grad_norm_mode": "requested_only",
        "datasets.val.manual_val.limit": 96,
        "trainer.face_quality.expected_images": 96,
        "validation_args.num_inference_steps": 50,
        "pipeline.pose_adapt_ratio": 0.0, "pipeline.ca_mixing_for_face": False,
        "model.pose_adapt_ratio": 0.0, "model.ca_mixing_for_face": False,
        "model.ba_hardcase_mode": "temporal_frequency",
        "model.ba_frequency_surface_loss_enabled": True,
        "model.ba_null_key_router_enabled": True,
        "model.ba_null_key_router_groups": ["up_blocks.0", "up_blocks.1"],
        "expected_trainable_contract.total_tensors": tensors,
        "expected_trainable_contract.total_parameters": parameters,
        "expected_trainable_contract.optimizer_tensors": tensors,
        "expected_trainable_contract.optimizer_parameters": parameters,
        "train_dataset_name": "cosmic_large_adapted",
        "datasets.train.cosmic_large_adapted.min_face_res": 192,
        "datasets.train.cosmic_large_adapted.random_horizontal_flip": True,
        "datasets.train.cosmic_large_adapted.random_reference_flip": False,
        "datasets.train.cosmic_large_adapted.reference_frame_mode": "target_face_frame",
        "datasets.train.cosmic_large_adapted.reference_scale_jitter": [0.06, 0.30],
        "datasets.train.cosmic_large_adapted.reference_position_jitter": 0.15,
        "datasets.train.cosmic_large_adapted.semantic_occlusion_probability": 0.25,
        "dataloaders.train.batch_size": 2,
        "trainer.seed": 0,
    }
    drift = {key: (want, selected(config, key)) for key, want in checks.items()
             if selected(config, key) != want}
    if drift:
        raise RuntimeError(f"Fixed CL39 contract drift: {drift}")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("Only fixed manual_val96 is allowed")
    if arm == "CL39X05":
        if not selected(config, "automatic_bboxes_every_val"):
            raise RuntimeError("X05 must rebuild preview ownership at every validation event")
        if not selected(config, "datasets.train.cosmic_large_adapted.ownership_cache_required"):
            raise RuntimeError("X05 training cache must fail closed")
        if not selected(config, "datasets.val.manual_val.ownership_cache_required"):
            raise RuntimeError("X05 reference-ownership cache must fail closed")
    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run-name mismatch")
    if plan.get("baseline") != "CL39_cosmic_null_key_confidence_router_24k":
        raise RuntimeError("Experiment record baseline mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment record config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL39X01_CL39X08_cl39_followups_1gpu.sh":
        raise RuntimeError("Experiment record launcher mismatch")
    if plan.get("machine") != "serv" or int(plan.get("gpus", -1)) != 1:
        raise RuntimeError("Each run must request one Serv GPU")
    print(json.dumps({"status": "ok", "arm": arm, "steps": 24000,
                      "trainable_tensors": tensors, "trainable_parameters": parameters}, indent=2))


if __name__ == "__main__":
    main()
