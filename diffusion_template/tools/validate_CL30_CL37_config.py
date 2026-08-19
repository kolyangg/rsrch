#!/usr/bin/env python3
"""Fail-closed composition/spec gate for CL30-CL37."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
ARMS = {
    "CL30_cosmic_positive_lowband_sameid_24k": (2240, 219217920, 24000),
    "CL31_cosmic_attention_ownership_alignment_24k": (2240, 219217920, 24000),
    "CL32_cosmic_contact_frequency_surface_24k": (2240, 219217920, 24000),
    "CL33_cosmic_visibility_balanced_reconstruction_24k": (2240, 219217920, 24000),
    "CL34_cosmic_shared_frequency_calibration_24k": (2241, 219217923, 24000),
    "CL35_cosmic_attention_gated_patch_identity_24k": (2240, 219217920, 24000),
    "CL36_cosmic_ba_arcface_hinge_4k": (2240, 219217920, 4000),
    "CL37_cosmic_smallface_roi_teacher_distill_24k": (2240, 219217920, 24000),
    "CL31_cosmic_attention_ownership_alignment_oneval_smoke": (2240, 219217920, 24000),
    "CL35_cosmic_attention_gated_patch_identity_oneval_smoke": (2240, 219217920, 24000),
    "CL36_cosmic_ba_arcface_hinge_oneval_smoke": (2240, 219217920, 4000),
}
TOGGLES = {
    "CL30": "model.ba_frequency_positive_sameid_enabled",
    "CL31": "model.ba_attention_ownership_loss_enabled",
    "CL32": "model.ba_frequency_surface_region_mode",
    "CL33": "loss_kind",
    "CL34": "model.ba_frequency_shared_schedule_enabled",
    "CL35": "model.ba_patch_identity_enabled",
    "CL36": "model.identity_aux_enabled",
    "CL37": "model.ba_roi_teacher_distill_enabled",
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

    tensors, parameters, steps = ARMS[args.config_name]
    checks = {
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": steps // 2000,
        "trainer.validation_interval_steps": 2000,
        "trainer.skip_initial_validation": False,
        "trainer.active_grad_norm_mode": "requested_only",
        "dataloaders.train.batch_size": 2,
        "dataloaders.manual_val.batch_size": (
            1 if args.config_name.endswith("_oneval_smoke") else 12
        ),
        "trainer.face_quality.expected_images": (
            1 if args.config_name.endswith("_oneval_smoke") else 96
        ),
        "datasets.val.manual_val.limit": (
            1 if args.config_name.endswith("_oneval_smoke") else 96
        ),
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
    drift = {key: (want, selected(config, key)) for key, want in checks.items() if selected(config, key) != want}
    if drift:
        raise RuntimeError(f"Fixed-contract drift: {drift}")
    if selected(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("Only manual_val96 is allowed")
    value = selected(config, TOGGLES[arm])
    expected = {"CL32": "contact_partition", "CL33": "visibility_balanced_ba"}.get(arm, True)
    if value != expected:
        raise RuntimeError(f"{arm} critical toggle is not active: {value!r}")
    if arm == "CL34" and selected(config, "model.ba_frequency_learnable_schedule_enabled"):
        raise RuntimeError("CL34 must not install CL28 per-layer schedules")
    if arm == "CL36":
        if selected(config, "model.identity_aux_gradient_scope") != "branched_sa_only":
            raise RuntimeError("CL36 identity gradients must be BA-only")
        if selected(config, "model.identity_aux_mode") != "quadratic_hinge":
            raise RuntimeError("CL36 must use the quadratic hinge")
        if not str(selected(config, "trainer.from_pretrained")):
            raise RuntimeError("CL36 source checkpoint is missing")
    elif selected(config, "trainer.from_pretrained") not in (None, "<missing>"):
        raise RuntimeError("Cold-start arm unexpectedly loads a checkpoint")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run-name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL30_CL37_cl27_followups_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("machine") != "serv" or int(plan.get("gpus", -1)) != 1:
        raise RuntimeError("Every arm must request one Serv GPU")
    print(json.dumps({"status": "ok", "arm": arm, "steps": steps, "trainable_tensors": tensors, "trainable_parameters": parameters}, indent=2))


if __name__ == "__main__":
    main()
