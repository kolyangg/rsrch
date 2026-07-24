#!/usr/bin/env python3
"""Consolidate already-rendered validation images into their training Comet run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comet_ml import API


COMET_WORKSPACE = "nikolay-2104"
COMET_PROJECT = "rsrch-30oct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def resolve_comet_key(run_name: str) -> str:
    api = API()
    for page in range(10):
        experiments = api.get_experiments(
            COMET_WORKSPACE,
            COMET_PROJECT,
            page=page,
            page_size=100,
            sort_by="startTime",
            sort_order="desc",
        )
        for experiment in experiments:
            if experiment.name == run_name:
                return str(experiment.key)
        if len(experiments) < 100:
            break
    raise RuntimeError(f"Could not resolve Comet experiment key for {run_name!r}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_name = manifest["run_name"]
    run_id = manifest.get("comet_experiment_key") or resolve_comet_key(run_name)
    manifest["comet_experiment_key"] = run_id
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(previous_experiment=run_id)
    if experiment.get_key() != run_id:
        raise RuntimeError(
            f"Comet resume verification failed: {experiment.get_key()} != {run_id}"
        )
    experiment.set_name(run_name)
    experiment.add_tag("23Jul_validation_consolidated")
    logged = []
    validation_root = run_dir / "validation"
    for mode_dir in sorted(validation_root.iterdir()):
        if not mode_dir.is_dir() or mode_dir.name not in {
            "canonical50",
            "earlyBA50",
            "pmControl50",
        }:
            continue
        for step_dir in sorted(mode_dir.glob("step_*")):
            step = int(step_dir.name.removeprefix("step_"))
            images = sorted((step_dir / "outputs").rglob("*.png"))
            for prompt_index, path in enumerate(images):
                name = (
                    f"{mode_dir.name}__step{step:04d}__p{prompt_index:02d}"
                    f"__{path.name}"
                )
                experiment.log_image(str(path), name=name, step=step)
                logged.append(
                    {"mode": mode_dir.name, "step": step, "name": name, "path": str(path)}
                )
    experiment.log_other("23Jul_validation_image_count", len(logged))
    experiment.end()
    output = run_dir / "validation" / "comet_consolidation_manifest.json"
    output.write_text(
        json.dumps(
            {"training_run_name": run_name, "training_run_id": run_id, "images": logged},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Logged {len(logged)} images to {run_name}; manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
