#!/usr/bin/env python3
"""Fail-closed composition/spec gate for the independent CL38-CL45 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL38_cosmic_visibility_ownership_v2_24k": (2240, 219217920),
    "CL39_cosmic_null_key_confidence_router_24k": (2240, 219217920),
    "CL40_cosmic_identity_motion_projector_24k": (2348, 223272960),
    "CL41_cosmic_landmark_canonical_kv_24k": (2240, 219217920),
    "CL42_cosmic_component_token_memory_24k": (2240, 219217920),
    "CL43_cosmic_id_adaptive_modulation_24k": (2384, 222596736),
    "CL44_cosmic_semantic_window_gate_24k": (2240, 219217920),
    "CL45_cosmic_ba_pcgrad_24k": (2240, 219217920),
}
TOGGLES = {
    "CL38": "model.ba_visibility_ownership_v2_enabled",
    "CL39": "model.ba_null_key_router_enabled",
    "CL40": "model.ba_identity_motion_projector_enabled",
    "CL41": "model.ba_landmark_canonical_kv_enabled",
    "CL42": "model.ba_component_token_memory_enabled",
    "CL43": "model.ba_id_adaptive_modulation_enabled",
    "CL44": "model.ba_semantic_window_gate_enabled",
    "CL45": "trainer.ba_pcgrad_enabled",
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
        "trainer.active_grad_norm_mode": "requested_only",
        "dataloaders.train.batch_size": 2,
        "dataloaders.manual_val.batch_size": 12,
        "trainer.face_quality.expected_images": 96,
        "datasets.val.manual_val.limit": 96,
        "validation_args.num_images_per_prompt": 1,
        "validation_args.num_inference_steps": 50,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "model.pose_adapt_ratio": 0.0,
        "model.ca_mixing_for_face": False,
        "model.ba_hardcase_mode": "temporal_frequency",
        "model.ba_hardcase_telemetry_enabled": False,
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
        raise RuntimeError(f"Fixed-contract drift: {drift}")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("Only manual_val96 is allowed")
    if selected(config, TOGGLES[arm]) is not True:
        raise RuntimeError(f"{arm} critical toggle is not active")
    if arm == "CL38" and selected(config, "model.ba_visibility_ownership_v2_delta_only") is not True:
        raise RuntimeError("CL38 corrected replay must isolate the ownership gradient to the BA delta")
    if selected(config, "trainer.from_pretrained") not in (None, "<missing>"):
        raise RuntimeError("Every CL38-CL45 arm must cold-start")
    extension_paths = list(TOGGLES.values())[:-1]
    active_extensions = [path for path in extension_paths if selected(config, path) is True]
    expected_count = 0 if arm == "CL45" else 1
    if len(active_extensions) != expected_count:
        raise RuntimeError(f"Independent-arm violation: {active_extensions}")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run-name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL38_CL45_cl27_architecture_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("machine") != "serv" or int(plan.get("gpus", -1)) != 1:
        raise RuntimeError("Every arm must request one Serv GPU")
    print(json.dumps({"status": "ok", "arm": arm, "steps": 24000,
                      "trainable_tensors": tensors,
                      "trainable_parameters": parameters}, indent=2))


if __name__ == "__main__":
    main()
