#!/usr/bin/env python3
"""Rebuild deterministic leaderboards from immutable experiment summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiments"
DECISION_ORDER = {
    "promising_step0_candidate": 0,
    "active_but_not_reference_improving": 1,
    "too_close_to_photomaker": 2,
    "n3a_like_unsafe": 3,
    "invalid_face_detection": 4,
    "error": 5,
}
PREFERRED = [
    "run_id",
    "experiment_id",
    "family",
    "role",
    "decision",
    "screen_score",
    "sample_count",
    "median_face_mae_vs_pm",
    "median_reference_gain_vs_pm",
    "positive_reference_gain_fraction",
    "median_landmark_displacement_vs_pm",
    "median_bbox_iou_vs_pm",
    "median_outside_mae_vs_pm",
    "median_boundary_ring_mae_vs_pm",
    "face_detection_rate",
    "experiment_dir",
    "error_type",
    "error",
]


def sort_key(row: dict) -> tuple:
    score = row.get("screen_score")
    return (
        DECISION_ORDER.get(row.get("decision"), 99),
        -(float(score) if isinstance(score, (int, float)) else -1e30),
        row.get("experiment_id", ""),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    extra = sorted({key for row in rows for key in row} - set(PREFERRED))
    fields = [key for key in PREFERRED if any(key in row for row in rows)] + extra
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    for path in sorted(ROOT.glob("*/metrics_summary.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"Skipping unreadable summary {path}: {error}")
            continue
        row["summary_path"] = str(path)
        rows.append(row)
    rows.sort(key=sort_key)
    write_csv(ROOT / "leaderboard_all.csv", rows)

    latest: dict[str, dict] = {}
    for row in rows:
        if row.get("sample_count") != 4:
            continue
        experiment_id = row.get("experiment_id")
        previous = latest.get(experiment_id)
        if previous is None or row.get("run_id", "") > previous.get("run_id", ""):
            latest[experiment_id] = row
    latest_rows = sorted(latest.values(), key=sort_key)
    write_csv(ROOT / "leaderboard_latest_four_case.csv", latest_rows)
    print(f"Summaries: {len(rows)}")
    print(f"Latest four-case experiments: {len(latest_rows)}")
    print(ROOT / "leaderboard_latest_four_case.csv")


if __name__ == "__main__":
    main()
