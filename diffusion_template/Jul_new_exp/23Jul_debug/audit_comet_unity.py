#!/usr/bin/env python3
"""Require one Comet experiment and one key across a completed 4k run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


WORKSPACE = "nikolay-2104"
PROJECT = "rsrch-30oct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def image_count(run_dir: Path, mode: str, step: int) -> int:
    root = run_dir / "validation" / mode / f"step_{step:04d}" / "outputs"
    return len(
        [
            path
            for path in root.rglob("*.png")
            if not path.stem.endswith("_mask")
        ]
    )


def main() -> int:
    if not os.environ.get("COMET_API_KEY"):
        raise RuntimeError("COMET_API_KEY must be exported by the caller")
    run_dir = parse_args().run_dir.resolve()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    run_name = manifest["run_name"]
    training_key = manifest.get("comet_experiment_key")
    if not training_key:
        raise RuntimeError("Training manifest has no resolved Comet key")

    validation_manifests = []
    validation_keys = set()
    for path in sorted((run_dir / "validation").rglob("validation_manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        validation_manifests.append(str(path))
        if record.get("status") != "completed":
            raise RuntimeError(f"Incomplete validation manifest: {path}")
        if record.get("comet_upload_status") != "completed":
            raise RuntimeError(f"Validation images were not uploaded: {path}")
        validation_keys.add(record.get("comet_experiment_key"))
    if validation_keys != {training_key}:
        raise RuntimeError(
            f"Validation keys {validation_keys} != training key {training_key}"
        )

    steps = [
        int(step)
        for step in manifest["protocol"]["validation_steps"]
    ]
    stage_counts = {
        str(step): image_count(run_dir, "canonical50", step) for step in steps
    }
    if any(count != 4 for count in stage_counts.values()):
        raise RuntimeError(f"Incomplete canonical image counts: {stage_counts}")
    pm_count = image_count(run_dir, "pmControl50", 0)
    if pm_count != 4:
        raise RuntimeError(f"Expected four PM controls, found {pm_count}")
    incremental_metric_steps = []
    for step in steps:
        metric_path = (
            run_dir
            / "report"
            / "incremental_metrics"
            / f"step_{step:04d}.json"
        )
        if not metric_path.exists():
            raise RuntimeError(
                f"Missing same-run Comet metric payload for step {step}: "
                f"{metric_path}"
            )
        metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
        summary = metric_payload.get("summary", {})
        if int(summary.get("step", -1)) != step:
            raise RuntimeError(f"Wrong metric step in {metric_path}: {summary}")
        if "median_prompt_image_clip_cosine" not in summary:
            raise RuntimeError(f"Missing text CLIP metric in {metric_path}")
        receipt_path = metric_path.with_name(
            f"step_{step:04d}.comet_uploaded.json"
        )
        if not receipt_path.exists():
            raise RuntimeError(
                f"Missing Comet metric upload receipt for step {step}: "
                f"{receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "completed"
            or receipt.get("comet_experiment_key") != training_key
            or int(receipt.get("step", -1)) != step
        ):
            raise RuntimeError(f"Invalid Comet metric receipt: {receipt}")
        incremental_metric_steps.append(step)

    from comet_ml import API

    api = API()
    # Comet's paginated API can return the boundary experiment on two
    # adjacent pages.  Treat repeated observations of the same immutable
    # experiment key as one match; distinct keys with the same name must
    # still fail the one-training-experiment invariant.
    exact_matches_by_key = {}
    raw_exact_match_count = 0
    for page in range(10):
        experiments = api.get_experiments(
            WORKSPACE,
            PROJECT,
            page=page,
            page_size=100,
            sort_by="startTime",
            sort_order="desc",
        )
        for experiment in experiments:
            if experiment.name != run_name:
                continue
            raw_exact_match_count += 1
            exact_matches_by_key[str(experiment.key)] = {
                "key": str(experiment.key),
                "name": experiment.name,
            }
        if len(experiments) < 100:
            break
    exact_matches = sorted(
        exact_matches_by_key.values(),
        key=lambda match: match["key"],
    )
    if exact_matches != [{"key": training_key, "name": run_name}]:
        raise RuntimeError(
            f"Expected exactly one Comet run named {run_name}: {exact_matches}"
        )

    payload = {
        "status": "PASS",
        "run_name": run_name,
        "training_comet_key": training_key,
        "api_exact_name_matches": exact_matches,
        "api_raw_exact_name_match_count": raw_exact_match_count,
        "validation_keys": sorted(validation_keys),
        "validation_manifest_count": len(validation_manifests),
        "canonical_image_counts": stage_counts,
        "pm_control_image_count": pm_count,
        "incremental_metric_steps": incremental_metric_steps,
    }
    output = run_dir / "report" / "comet_unity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
