#!/usr/bin/env python3
"""Fail closed unless CL14_CA is CL14 plus its one declared model delta."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.model.photomaker_branched.e13_contract import normalise_e13_settings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src/configs"
CONFIG_NAME = "CL14_CA_cosmic_residual_identity_ca_24k"
BASE_NAME = "CL14_cosmic_joint_shadow_sa128_softmask_24k"
ALLOWED_ROOTS = (
    "model.e13_settings.ba_residual_identity_ca_v3_",
    "writer.loss_names",
    "writer.experiment_comment",
    # Latest CL14_CA production used CL20's validation-only Eddie repair.
    "pipeline._target_",
    "validation_args.face_subject_selection_policy",
    "metrics",
    "inference_metrics",
)


def _flat(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flat(child, path))
        return output
    return {prefix: value}


def _value(config, path):
    selected = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


def _require(config, path, expected):
    actual = _value(config, path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        base = compose(config_name=BASE_NAME)
        candidate = compose(config_name=CONFIG_NAME)

    base_flat = _flat(OmegaConf.to_container(base, resolve=False))
    candidate_flat = _flat(OmegaConf.to_container(candidate, resolve=False))
    drift = {
        path: (base_flat.get(path, "<missing>"), candidate_flat.get(path, "<missing>"))
        for path in sorted(set(base_flat) | set(candidate_flat))
        if base_flat.get(path, object()) != candidate_flat.get(path, object())
        and not any(path.startswith(root) for root in ALLOWED_ROOTS)
    }
    if drift:
        raise RuntimeError(f"CL14_CA contains undeclared CL14 drift: {drift}")

    fixed = {
        "train_dataset_name": "cosmic_large_adapted",
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "validation_args.face_subject_selection_policy": "bbox_overlap_v2",
        "validation_args.num_inference_steps": 50,
        "validation_args.guidance_scale": 5,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "dataloaders.train.batch_size": 2,
        "datasets.val.manual_val.limit": 96,
    }
    for path, expected in fixed.items():
        _require(candidate, path, expected)
    settings = normalise_e13_settings(candidate.model.e13_settings)
    expected_settings = {
        "ba_training_mask_feather": 2,
        "ba_hardcase_mode": "off",
        "ba_crossview_consistency_enabled": False,
        "ba_residual_identity_ca_v3_enabled": True,
        "ba_residual_identity_ca_v3_groups": ("up_blocks.0", "up_blocks.1"),
        "ba_residual_identity_ca_v3_rank": 64,
        "ba_residual_identity_ca_v3_gate_init": 0.02,
        "ba_residual_identity_ca_v3_gate_max": 0.20,
    }
    for name, expected in expected_settings.items():
        if settings[name] != expected:
            raise RuntimeError(
                f"model.e13_settings.{name}: expected {expected!r}, "
                f"got {settings[name]!r}"
            )
    _require(
        candidate,
        "pipeline._target_",
        "src.pipelines.photomaker_branched_subject_v2."
        "PhotomakerBranchedSubjectV2Pipeline.from_pretrained",
    )
    _require(
        candidate,
        "inference_metrics",
        ["clip_ts", "id_sim_best_legacy", "id_sim_subject_v2"],
    )
    expected_loss_names = [
        "loss",
        *[
            f"ba/identity_ca_{metric}/{group}"
            for metric in (
                "token_count", "delta_rms", "gate", "native_face_rms",
                "residual_face_rms", "residual_native_ratio",
            )
            for group in ("up0", "up1", "all")
        ],
    ]
    if _value(candidate, "writer.loss_names") != expected_loss_names:
        raise RuntimeError("CL14_CA residual-CA telemetry schema drifted")
    print(
        f"{CONFIG_NAME}: OK (2348 tensors / 224624676 parameters; "
        "one model delta from CL14)"
    )


if __name__ == "__main__":
    main()
