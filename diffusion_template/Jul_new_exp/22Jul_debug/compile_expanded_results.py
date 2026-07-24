#!/usr/bin/env python3
"""Compile latest immutable experiment bundles into study-level CSV/Markdown logs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
SUMMARY_CSV = HERE / "expanded_results_latest.csv"
IDENTITY_CSV = HERE / "expanded_results_by_identity.csv"
MARKDOWN = HERE / "expanded_results_latest.md"


def latest_summaries() -> dict[str, tuple[Path, dict]]:
    latest: dict[str, tuple[float, Path, dict]] = {}
    for path in EXPERIMENTS.glob("*/metrics_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiment_id = payload.get("experiment_id")
        if not experiment_id or payload.get("decision") == "error":
            continue
        stamp = path.stat().st_mtime
        if experiment_id not in latest or stamp > latest[experiment_id][0]:
            latest[experiment_id] = (stamp, path, payload)
    return {key: (value[1], value[2]) for key, value in latest.items()}


def relevant(payload: dict) -> bool:
    experiment_id = str(payload.get("experiment_id", ""))
    role = str(payload.get("role", ""))
    return (
        int(payload.get("sample_count", 0)) >= 8
        or experiment_id.startswith("matrix24_")
        or experiment_id.startswith("matrix96_")
        or experiment_id.startswith("nn7v2_")
        or experiment_id.startswith("n3a_alignrepair_")
        or role in {"24_case_existing_finalist", "24_case_modern_promotion"}
    )


def finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def median(values: list[float]) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def fmt(value, digits: int = 5) -> str:
    value = finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    completed = latest_summaries()
    selected = sorted(
        ((path, payload) for path, payload in completed.values() if relevant(payload)),
        key=lambda pair: (int(pair[1].get("sample_count", 0)), pair[1]["experiment_id"]),
    )

    summary_fields = [
        "experiment_id", "family", "role", "sample_count", "decision",
        "causal_baseline", "median_face_mae", "median_reference_gain",
        "positive_reference_fraction", "median_landmark_displacement",
        "median_bbox_iou", "median_outside_mae", "experiment_dir",
    ]
    summary_rows = []
    identity_values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for summary_path, payload in selected:
        causal = payload.get("median_face_mae_vs_ba0") is not None
        suffix = "_vs_ba0" if causal else "_vs_pm"
        summary_rows.append({
            "experiment_id": payload["experiment_id"],
            "family": payload.get("family"),
            "role": payload.get("role"),
            "sample_count": payload.get("sample_count"),
            "decision": payload.get("decision"),
            "causal_baseline": "BA0" if causal else "PM",
            "median_face_mae": payload.get(f"median_face_mae{suffix}"),
            "median_reference_gain": payload.get(f"median_reference_gain{suffix}"),
            "positive_reference_fraction": payload.get(
                "positive_reference_gain_vs_ba0_fraction" if causal
                else "positive_reference_gain_fraction"
            ),
            "median_landmark_displacement": payload.get(f"median_landmark_displacement{suffix}"),
            "median_bbox_iou": payload.get(f"median_bbox_iou{suffix}"),
            "median_outside_mae": payload.get(f"median_outside_mae{suffix}"),
            "experiment_dir": payload.get("experiment_dir"),
        })
        metrics_path = summary_path.parent / "metrics_per_sample.csv"
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (payload["experiment_id"], row["id"])
                    identity_values[key].append({
                        "gain": float(row["reference_gain_vs_ba0"] if causal else row["reference_gain_vs_pm"]),
                        "face": float(row["face_mae_vs_ba0"] if causal else row["face_mae_vs_pm"]),
                    })

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    identity_rows = []
    for (experiment_id, identity), values in sorted(identity_values.items()):
        gains = [value["gain"] for value in values]
        faces = [value["face"] for value in values]
        identity_rows.append({
            "experiment_id": experiment_id,
            "id": identity,
            "n": len(values),
            "median_reference_gain": median(gains),
            "positive_fraction": sum(value > 0 for value in gains) / len(gains),
            "median_face_mae": median(faces),
        })
    with IDENTITY_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment_id", "id", "n", "median_reference_gain", "positive_fraction", "median_face_mae"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(identity_rows)

    lines = [
        "# Expanded step-zero results — latest completed bundles", "",
        "Causal packed/NN7 experiments use BA0 as baseline; legacy N3a experiments use ordinary PhotoMaker.", "",
        "| experiment | n | baseline | face MAE | ref gain | positive | landmark | bbox IoU | outside |", 
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['experiment_id']}` | {row['sample_count']} | {row['causal_baseline']} | "
            f"{fmt(row['median_face_mae'])} | {fmt(row['median_reference_gain'])} | "
            f"{fmt(row['positive_reference_fraction'], 2)} | {fmt(row['median_landmark_displacement'])} | "
            f"{fmt(row['median_bbox_iou'])} | {fmt(row['median_outside_mae'])} |"
        )
    lines.extend(["", f"Identity-stratified values: `{IDENTITY_CSV.name}`.", ""])
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"experiments": len(summary_rows), "summary_csv": str(SUMMARY_CSV), "identity_csv": str(IDENTITY_CSV), "markdown": str(MARKDOWN)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
