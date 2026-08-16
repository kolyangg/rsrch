#!/usr/bin/env python3
"""Fail-closed gate for the bounded CL29 throughput qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
CONFIG_NAME = "CL29_cosmic_lowband_causal_contrastive_24k_speedcheck"
BASE_NAME = "CL29_cosmic_lowband_causal_contrastive_24k_speedopt"


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
        if candidate_flat.get(key, "<missing>") != baseline_flat.get(key, "<missing>")
    }
    allowed = {
        "trainer.epoch_len",
        "trainer.n_epochs",
        "trainer.save_period",
        "trainer.validation_interval_steps",
        "weights_only_save_period",
        "writer.experiment_comment",
    }
    if changed != allowed:
        raise RuntimeError(f"Unexpected CL29 speedcheck drift: {sorted(changed ^ allowed)}")
    expected = {
        "trainer.epoch_len": 100,
        "trainer.n_epochs": 1,
        "trainer.save_period": 999,
        "trainer.skip_initial_validation": True,
        "trainer.validation_interval_steps": 0,
        "weights_only_save_period": 0,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
        "expected_trainable_contract.total_tensors": 2240,
        "expected_trainable_contract.total_parameters": 219217920,
    }
    drift = {
        key: (value, candidate_flat.get(key))
        for key, value in expected.items()
        if candidate_flat.get(key) != value
    }
    if drift:
        raise RuntimeError(f"CL29 speedcheck contract drift: {drift}")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    plan = spec.get("plan", {})
    if plan.get("config") != f"src/configs/{CONFIG_NAME}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL29_speedcheck_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    print(json.dumps({"status": "ok", "changed_paths": sorted(changed)}, indent=2))


if __name__ == "__main__":
    main()
