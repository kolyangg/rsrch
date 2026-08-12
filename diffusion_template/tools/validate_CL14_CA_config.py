#!/usr/bin/env python3
"""Fail-closed config/spec gate for the CL14_CA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
BASE_CONFIG = "CL14_cosmic_joint_shadow_sa128_softmask_24k"
RUN_CONFIGS = {
    "CL14_CA": "CL14_CA",
    "CL14_CA_r3": "CL14_CA",
    "CL14_CA_r4": "CL14_CA",
    "CL14_CA_r5": "CL14_CA",
    "CL14_CA_r6": "CL14_CA",
    "CL14_CA_r7": "CL14_CA",
    # 12 Aug 2026 - Training optimization launch pair: identical science,
    # optimized scalar synchronization, ID-token gather, and data transfer.
    "CL14_CA_optimized_r1": "CL14_CA",
    "CL14_CA_optimized_speed_smoke_r1": "CL14_CA_skipval_smoke",
    "CL14_CA_optimized_r2": "CL14_CA",
    "CL14_CA_optimized_speed_smoke_r2": "CL14_CA_skipval_smoke",
    "CL14_CA_skipval_smoke_r1": "CL14_CA_skipval_smoke",
    "CL14_CA_skipval_smoke_r2": "CL14_CA_skipval_smoke",
    "CL14_CA_skipval_smoke_r3": "CL14_CA_skipval_smoke",
    "CL14_CA_skipval_smoke_r4": "CL14_CA_skipval_smoke",
    "CL14_CA_skipval_smoke_r5": "CL14_CA_skipval_smoke",
    "CL14_CA_oneval_smoke_r1": "CL14_CA_oneval_smoke",
    "CL14_CA_onebatch_smoke_r1": "CL14_CA_onebatch_smoke",
    "CL14_CA_onebatch_smoke_r2": "CL14_CA_onebatch_smoke",
}
BASELINE_KEY = "6fe0028be92242c38056b3d36665fdd6"
ALLOWED_DIFFS = {
    "model.ba_identity_ca_v2_enabled",
    "model.ba_residual_identity_ca_v3_enabled",
    "model.ba_residual_identity_ca_v3_groups",
    "model.ba_residual_identity_ca_v3_rank",
    "model.ba_residual_identity_ca_v3_gate_init",
    "model.ba_residual_identity_ca_v3_gate_max",
    "expected_trainable_contract",
    "writer.loss_names",
    "writer.experiment_comment",
    "trainer.skip_initial_validation",
    "datasets.val.manual_val.limit",
    "trainer.face_quality.expected_images",
    "non_blocking_dataloader",
    "dataloaders.train.pin_memory",
    "dataloaders.train.persistent_workers",
}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def selected(config: Any, path: str) -> Any:
    value = OmegaConf.select(config, path, default="<missing>")
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def require(config: Any, path: str, expected: Any) -> None:
    actual = selected(config, path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected {expected!r}, found {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True)
    args = parser.parse_args()
    expected_config = RUN_CONFIGS.get(args.run_name)
    if expected_config != args.config_name:
        raise RuntimeError(
            f"Unexpected CL14_CA run/config pair: {args.run_name}/{args.config_name}"
        )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        base = compose(config_name=BASE_CONFIG, overrides=["writer=cometml"])
        candidate = compose(config_name=args.config_name, overrides=["writer=cometml"])

    base_flat = flatten(OmegaConf.to_container(base, resolve=False))
    candidate_flat = flatten(OmegaConf.to_container(candidate, resolve=False))
    missing = object()

    def allowed(path: str) -> bool:
        return any(path == root or path.startswith(root + ".") for root in ALLOWED_DIFFS)

    drift = {
        path: (base_flat.get(path, "<missing>"), candidate_flat.get(path, "<missing>"))
        for path in sorted(set(base_flat) | set(candidate_flat))
        if base_flat.get(path, missing) != candidate_flat.get(path, missing)
        and not allowed(path)
    }
    if drift:
        raise RuntimeError(f"CL14_CA has non-CA drift from CL14: {drift}")

    checks = {
        "disable_branched_ca": True,
        "train_branched_ca_lora": False,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "model.ba_architecture_version": "hard_replace_v1",
        "model.ba_training_mask_feather": 2,
        "model.ba_identity_ca_v2_enabled": False,
        "model.ba_residual_identity_ca_v3_enabled": True,
        "model.ba_residual_identity_ca_v3_groups": ["up_blocks.0", "up_blocks.1"],
        "model.ba_residual_identity_ca_v3_rank": 64,
        "model.ba_residual_identity_ca_v3_gate_init": 0.02,
        "model.ba_residual_identity_ca_v3_gate_max": 0.20,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "trainer.validation_interval_steps": 2000,
        "trainer.skip_initial_validation": args.config_name.endswith("_skipval_smoke"),
        "dataloaders.train.batch_size": 2,
        "dataloaders.train.num_workers": 2,
        "dataloaders.train.pin_memory": True,
        "dataloaders.train.persistent_workers": True,
        "non_blocking_dataloader": True,
        "dataloaders.manual_val.batch_size": 12,
        "datasets.val.manual_val.limit": (
            12 if args.config_name.endswith("_onebatch_smoke") else
            1 if args.config_name.endswith("_oneval_smoke") else 96
        ),
        "trainer.face_quality.expected_images": (
            12 if args.config_name.endswith("_onebatch_smoke") else 96
        ),
        "validation_args.num_images_per_prompt": 1,
        "validation_args.num_inference_steps": 50,
        "expected_trainable_contract.total_tensors": 2348,
        "expected_trainable_contract.total_parameters": 224624676,
        "expected_trainable_contract.optimizer_tensors": 2348,
        "expected_trainable_contract.optimizer_parameters": 224624676,
    }
    for path, expected in checks.items():
        require(candidate, path, expected)
    require(candidate, "val_datasets_names", ["manual_val"])

    expected_metrics = [
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
    require(candidate, "writer.loss_names", expected_metrics)

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    spec_checks = {
        "run_name": spec.get("run_name"),
        "config": plan.get("config"),
        "launcher": plan.get("launcher"),
        "machine": plan.get("machine"),
        "gpus": plan.get("gpus"),
        "comet_project": plan.get("comet_project"),
        "baseline_key": plan.get("baseline_comet_experiment_key"),
    }
    expected_spec = {
        "run_name": args.run_name,
        "config": f"src/configs/{args.config_name}.yaml",
        "launcher": "launchers/active/run_CL14_CA_24k_1gpu.sh",
        "machine": "serv",
        "gpus": 1,
        "comet_project": "aug-large-ds",
        "baseline_key": BASELINE_KEY,
    }
    if spec_checks != expected_spec:
        raise RuntimeError(f"CL14_CA experiment spec drift: {spec_checks}")

    print(json.dumps({
        "status": "ok",
        "run_name": args.run_name,
        "base_config": BASE_CONFIG,
        "optimizer_steps": 24000,
        "validation_images": (
            12 if args.config_name.endswith("_onebatch_smoke") else
            1 if args.config_name.endswith("_oneval_smoke") else 96
        ),
        "trainable_tensors": 2348,
        "trainable_parameters": 224624676,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
