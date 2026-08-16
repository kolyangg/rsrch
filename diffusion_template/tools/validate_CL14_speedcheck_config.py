#!/usr/bin/env python3
"""Fail-closed gate for the current-pipeline CL14 throughput smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
CONFIG_NAME = "CL14_cosmic_joint_shadow_sa128_softmask_24k_speedcheck"
BASE_NAME = "CL14_cosmic_joint_shadow_sa128_softmask_24k"
LAUNCHER = "launchers/active/run_CL14_speedcheck_1gpu.sh"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True)
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        candidate = compose(config_name=CONFIG_NAME)
        baseline = compose(config_name=BASE_NAME)
    candidate_flat = flatten(OmegaConf.to_container(candidate, resolve=True))
    baseline_flat = flatten(OmegaConf.to_container(baseline, resolve=True))
    changed = {
        key
        for key in set(candidate_flat) | set(baseline_flat)
        if candidate_flat.get(key, "<missing>")
        != baseline_flat.get(key, "<missing>")
    }
    allowed = {
        "datasets.val.manual_val.limit",
        "trainer.active_grad_norm_mode",
        "trainer.face_quality.expected_images",
        "writer.experiment_comment",
    }
    if changed != allowed:
        raise RuntimeError(f"Unexpected CL14 speedcheck drift: {sorted(changed ^ allowed)}")

    expected = {
        "datasets.val.manual_val.limit": 12,
        "trainer.active_grad_norm_mode": "requested_only",
        "trainer.face_quality.expected_images": 12,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "expected_trainable_contract.total_tensors": 2240,
        "expected_trainable_contract.total_parameters": 219217920,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
    }
    drift = {
        key: (value, candidate_flat.get(key))
        for key, value in expected.items()
        if candidate_flat.get(key) != value
    }
    if drift:
        raise RuntimeError(f"CL14 speedcheck contract drift: {drift}")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    plan = spec.get("plan", {})
    if plan.get("config") != f"src/configs/{CONFIG_NAME}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != LAUNCHER:
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("comet_project") != "aug-large-ds":
        raise RuntimeError("Experiment spec Comet project mismatch")
    print(json.dumps({"status": "ok", "changed_paths": sorted(changed)}, indent=2))


if __name__ == "__main__":
    main()
