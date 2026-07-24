#!/usr/bin/env python3
"""Export incremental 4k metrics from ignored runs into compact Git files."""

from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "experiments_4k"


def fmt(value):
    return "—" if value is None else f"{float(value):.4f}"


def export_rows() -> int:
    rows = []
    for manifest_path in sorted(RUNS.glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for metric_path in sorted(
            (run_dir / "report" / "incremental_metrics").glob("step_*.json")
        ):
            if metric_path.name.endswith(".comet_uploaded.json"):
                continue
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            step = int(summary["step"])
            receipt_path = metric_path.with_name(
                f"step_{step:04d}.comet_uploaded.json"
            )
            receipt_status = None
            if receipt_path.exists():
                receipt_status = json.loads(
                    receipt_path.read_text(encoding="utf-8")
                ).get("status")
            rows.append(
                {
                    "architecture_id": manifest["architecture_id"],
                    "base_architecture_id": manifest.get(
                        "base_architecture_id", manifest["architecture_id"]
                    ),
                    "dataset_profile": manifest["dataset_profile"],
                    "run_name": manifest["run_name"],
                    "run_status": manifest.get("status"),
                    "comet_experiment_key": manifest.get(
                        "comet_experiment_key"
                    ),
                    "comet_metric_upload": receipt_status,
                    **summary,
                }
            )

    csv_path = HERE / "results_4k_live.csv"
    json_path = HERE / "results_4k_live.json"
    md_path = HERE / "results_4k_live.md"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Live NN3a 4k checkpoint metrics",
        "",
        "Generated from ignored run folders after each same-experiment Comet "
        "metric upload. Visual anatomy remains the promotion gate.",
        "",
        "| architecture | dataset | step | ref sim | gain vs PM | sim to PM | "
        "landmark | text CLIP | selection | Comet |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {architecture_id} | {dataset_profile} | {step} | {ref} | "
            "{gain} | {pm} | {landmark} | {clip} | {selection} | "
            "{comet} |".format(
                architecture_id=row["architecture_id"],
                dataset_profile=row["dataset_profile"],
                step=row["step"],
                ref=fmt(row.get("median_reference_similarity")),
                gain=fmt(row.get("median_reference_gain_vs_pm")),
                pm=fmt(row.get("median_face_similarity_to_pm_output")),
                landmark=fmt(
                    row.get("median_landmark_displacement_vs_pm")
                ),
                clip=fmt(row.get("median_prompt_image_clip_cosine")),
                selection=fmt(row.get("selection_score")),
                comet=row.get("comet_metric_upload") or "missing",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported {len(rows)} live metric rows")
    return 0


def main() -> int:
    lock_path = HERE / "scheduler_4k" / "export_live.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return export_rows()


if __name__ == "__main__":
    raise SystemExit(main())
