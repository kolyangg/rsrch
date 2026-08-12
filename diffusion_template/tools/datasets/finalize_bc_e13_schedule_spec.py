#!/usr/bin/env python3
"""Atomically pin a generated BC_E13 schedule into its pre-Comet run spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.bc_e13_schedule_policy import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-spec", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=["ds1", "ds2", "ds3"], required=True)
    args = parser.parse_args()

    schedule_sha = sha256_file(args.schedule)
    summary_sha = sha256_file(args.summary)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("kind") != "bc_e13_dataset_schedule":
        raise RuntimeError("Unexpected schedule-summary kind")
    if summary.get("mode") != args.expected_mode:
        raise RuntimeError("Schedule mode does not match experiment mode")
    if summary.get("schedule", {}).get("sha256") != schedule_sha:
        raise RuntimeError("Schedule bytes do not match summary")
    if int(summary.get("schedule", {}).get("rows", -1)) != 48000:
        raise RuntimeError("Experiment schedule must contain exactly 48,000 rows")

    spec = json.loads(args.experiment_spec.read_text(encoding="utf-8"))
    plan = spec.get("plan") or {}
    if plan.get("dataset_mode") != args.expected_mode:
        raise RuntimeError("Experiment spec dataset mode mismatch")
    schedule = plan.get("schedule") or {}
    previous = schedule.get("schedule_sha256")
    if previous not in (None, schedule_sha):
        raise RuntimeError(
            f"Refusing to replace a different pinned schedule hash: {previous}"
        )
    schedule.update(
        {
            "schedule_sha256": schedule_sha,
            "schedule_summary_sha256": summary_sha,
            "schedule_path": str(args.schedule),
            "schedule_summary_path": str(args.summary),
            "sealed_counts": summary.get("counts"),
            "sealed_cohort": summary.get("cohort"),
        }
    )
    plan["schedule"] = schedule
    spec["plan"] = plan
    temporary = args.experiment_spec.with_name(f".{args.experiment_spec.name}.tmp")
    temporary.write_text(
        json.dumps(spec, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.experiment_spec)
    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_spec": str(args.experiment_spec),
                "schedule_sha256": schedule_sha,
                "summary_sha256": summary_sha,
                "rows": 48000,
                "mode": args.expected_mode,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
