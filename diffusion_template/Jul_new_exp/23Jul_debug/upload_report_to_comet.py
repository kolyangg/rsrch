#!/usr/bin/env python3
"""Upload local checkpoint metrics/PDF to the verified training Comet run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from migrate_validation_to_comet import resolve_comet_key
from launch_validation import ensure_comet_api_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--extra-asset",
        action="append",
        type=Path,
        default=[],
        help="Additional comparison artifact to attach to the same training run.",
    )
    return parser.parse_args()


def main() -> int:
    ensure_comet_api_key()
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_name = manifest["run_name"]
    comet_key = manifest.get("comet_experiment_key") or resolve_comet_key(run_name)
    manifest["comet_experiment_key"] = comet_key
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    report_dir = run_dir / "report"
    summary_path = report_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(previous_experiment=comet_key)
    if experiment.get_key() != comet_key:
        raise RuntimeError(
            f"Comet resume verification failed: {experiment.get_key()} != {comet_key}"
        )
    experiment.set_name(run_name)
    logged_metrics = []
    for record in summary:
        mode = record["mode"]
        step = int(record["step"])
        for key, value in record.items():
            if key in {"mode", "step"} or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            metric_name = f"validation/{mode}/{key}"
            experiment.log_metric(metric_name, value, step=step)
            logged_metrics.append(
                {"name": metric_name, "step": step, "value": value}
            )

    uploaded_assets = []
    architecture_id = manifest["architecture_id"]
    for path in (
        run_dir / "run_manifest.json",
        report_dir / "checkpoint_visual_summary.pdf",
        report_dir / "metrics_summary.json",
        report_dir / "metrics_per_image.csv",
        report_dir / "SUMMARY.md",
        report_dir / "comet_unity_audit.json",
        report_dir / "pm_bbox_debug" / "pm_generated_mask_contact_sheet.pdf",
        report_dir / "pm_bbox_debug" / "pm_generated_mask_contact_sheet.png",
        report_dir / "pm_bbox_debug" / "manifest.json",
    ):
        if path.exists():
            section = "run" if path.parent == run_dir else path.parent.name
            asset_name = f"23Jul_{architecture_id}__{section}__{path.name}"
            if len(asset_name) > 100:
                raise RuntimeError(f"Comet asset name exceeds 100 characters: {asset_name}")
            experiment.log_asset(str(path), file_name=asset_name)
            uploaded_assets.append({"name": asset_name, "path": str(path)})
    for path in args.extra_asset:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        asset_name = f"23Jul_extra__{path.name}"
        if len(asset_name) > 100:
            raise RuntimeError(f"Comet asset name exceeds 100 characters: {asset_name}")
        experiment.log_asset(str(path), file_name=asset_name)
        uploaded_assets.append({"name": asset_name, "path": str(path)})
    experiment.end()

    output = report_dir / "comet_report_upload_manifest.json"
    output.write_text(
        json.dumps(
            {
                "training_run_name": run_name,
                "training_comet_key": comet_key,
                "metrics": logged_metrics,
                "assets": uploaded_assets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Uploaded {len(logged_metrics)} metrics and {len(uploaded_assets)} assets "
        f"to {run_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
