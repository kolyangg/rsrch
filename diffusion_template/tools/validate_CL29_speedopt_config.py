#!/usr/bin/env python3
"""Fail-closed gate for the CL29 throughput-corrected continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
CONFIG_NAME = "CL29_cosmic_lowband_causal_contrastive_24k_speedopt"
BASE_NAME = "CL29_cosmic_lowband_causal_contrastive_24k"


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
        "model.ba_hardcase_telemetry_enabled",
        "model.ba_frequency_lowband_sample_on_cpu",
        "trainer.active_grad_norm_mode",
        "trainer.skip_initial_validation",
        "writer.experiment_comment",
    }
    if changed != allowed:
        raise RuntimeError(f"Unexpected CL29 speedopt drift: {sorted(changed ^ allowed)}")
    expected = {
        "model.ba_hardcase_telemetry_enabled": False,
        "model.ba_frequency_lowband_sample_on_cpu": True,
        "trainer.active_grad_norm_mode": "requested_only",
        "trainer.skip_initial_validation": True,
        "trainer.epoch_len": 2000,
        "trainer.n_epochs": 12,
        "pipeline.pose_adapt_ratio": 0.0,
        "pipeline.ca_mixing_for_face": False,
    }
    drift = {
        key: (value, candidate_flat.get(key))
        for key, value in expected.items()
        if candidate_flat.get(key) != value
    }
    if drift:
        raise RuntimeError(f"CL29 speedopt contract drift: {drift}")

    spec = json.loads(Path(args.experiment_spec).read_text(encoding="utf-8"))
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    plan = spec.get("plan", {})
    if plan.get("config") != f"src/configs/{CONFIG_NAME}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != "launchers/active/run_CL29_speedopt_1gpu.sh":
        raise RuntimeError("Experiment spec launcher mismatch")
    print(json.dumps({"status": "ok", "changed_paths": sorted(changed)}, indent=2))


if __name__ == "__main__":
    main()
