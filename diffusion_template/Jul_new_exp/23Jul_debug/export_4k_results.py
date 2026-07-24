#!/usr/bin/env python3
"""Export ignored 4k run artifacts into compact Git-friendly result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "experiments_4k"


def finite(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> int:
    rows = []
    runs = []
    for manifest_path in sorted(RUNS.glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "report" / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        comet_audit_path = run_dir / "report" / "comet_unity_audit.json"
        comet_audit = (
            json.loads(comet_audit_path.read_text(encoding="utf-8"))
            if comet_audit_path.exists()
            else {}
        )
        run_record = {
            "architecture_id": manifest["architecture_id"],
            "dataset_profile": manifest["dataset_profile"],
            "run_name": manifest["run_name"],
            "status": manifest.get("status"),
            "comet_experiment_key": manifest.get("comet_experiment_key"),
            "pairing_audit_status": (
                manifest.get("pairing_audit") or {}
            ).get("status"),
            "comet_unity_status": comet_audit.get("status"),
            "architecture": manifest["architecture"],
        }
        runs.append(run_record)
        for metric in metrics:
            if metric.get("mode") != "canonical50":
                continue
            rows.append(
                {
                    **{
                        key: run_record[key]
                        for key in (
                            "architecture_id",
                            "dataset_profile",
                            "run_name",
                            "status",
                            "comet_experiment_key",
                            "pairing_audit_status",
                            "comet_unity_status",
                        )
                    },
                    **metric,
                }
            )

    csv_path = HERE / "results_4k_latest.csv"
    json_path = HERE / "results_4k_latest.json"
    md_path = HERE / "results_4k_latest.md"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(
        json.dumps({"runs": runs, "metrics": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# NN3a 4k compact results",
        "",
        "Generated from ignored local run folders. Visual promotion decisions remain "
        "in `EXPERIMENT_LOG_4K.md`.",
        "",
        "| architecture | dataset | step | ref sim | gain vs PM | sim to PM | text CLIP | text gain vs PM | landmark | bbox IoU | pairing | Comet unity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {architecture_id} | {dataset_profile} | {step} | {ref} | {gain} | "
            "{pm} | {text} | {text_gain} | {landmark} | {iou} | {pairing} | {unity} |".format(
                architecture_id=row["architecture_id"],
                dataset_profile=row["dataset_profile"],
                step=row["step"],
                ref=finite(row.get("median_reference_similarity")),
                gain=finite(row.get("median_reference_gain_vs_pm")),
                pm=finite(row.get("median_face_similarity_to_pm_output")),
                text=finite(row.get("median_prompt_image_clip_cosine")),
                text_gain=finite(
                    row.get("median_prompt_image_clip_gain_vs_pm")
                ),
                landmark=finite(
                    row.get("median_landmark_displacement_vs_pm")
                ),
                iou=finite(row.get("median_bbox_iou_vs_pm")),
                pairing=row.get("pairing_audit_status") or "—",
                unity=row.get("comet_unity_status") or "—",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_count": len(runs),
                "metric_row_count": len(rows),
                "outputs": [str(csv_path), str(json_path), str(md_path)],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
