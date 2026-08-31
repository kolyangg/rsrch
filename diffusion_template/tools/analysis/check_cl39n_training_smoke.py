#!/usr/bin/env python3
"""Gate a completed 100-step CL39N qualification and report hot-path speed."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


MARKERS = {
    "CL39N6R": "ba/group_band/accepted_high_rms/all",
    "CL39N7": "ba/posterior_null/null_mass/all",
    "CL39N8": "ba/native_orthogonal/affected_query_fraction/all",
    "CL39N9": "ba/intrinsic_id/residual_native_ratio/all",
}


def last_scalar(text: str, name: str) -> float | None:
    values = re.findall(
        rf"train/{re.escape(name)}\s*=\s*(?:tensor\()?([-+0-9.eE]+)", text
    )
    return None if not values else float(values[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(MARKERS), required=True)
    parser.add_argument("--mode", choices=("no_validation", "validated"), required=True)
    parser.add_argument("--max-median-seconds", type=float, required=True)
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    if "Traceback (most recent call last)" in text or "AssertionError" in text:
        raise RuntimeError("Qualification log contains a Python failure")
    speeds = {}
    for step, value in re.findall(r"\|\s*(\d+)/100\s*\[[^\n]*?([0-9]+(?:\.[0-9]+)?)s/it", text):
        speeds[int(step)] = float(value)
    window = [value for step, value in speeds.items() if 21 <= step <= 99]
    median = statistics.median(window) if len(window) >= 70 else float("inf")
    images = 0
    if args.expected_images:
        if args.images_root is None:
            raise RuntimeError("Validated qualification requires --images-root")
        images = len(list(args.images_root.glob("step_0_batch_*/*.png")))
    activity_value = last_scalar(text, MARKERS[args.arm])
    activity_ok = activity_value is not None and activity_value > 0.0
    if args.arm == "CL39N7":
        activity_ok = activity_ok and activity_value < 1.0
    if args.arm == "CL39N6R":
        disabled_low = last_scalar(text, "ba/group_band/accepted_low_rms/up1")
        activity_ok = activity_ok and disabled_low == 0.0
    else:
        disabled_low = None
    checks = {
        "completed_100_steps": max(speeds, default=0) >= 99,
        "timing_window": len(window) >= 70,
        "median_within_ceiling": median <= args.max_median_seconds,
        "mechanism_telemetry": activity_ok,
        "finite": "nan" not in " ".join(
            line.lower() for line in text.splitlines() if "Step " in line and "train/" in line
        ),
        "validation_images": images == args.expected_images,
        "resident_validation_fix": (
            args.mode == "no_validation"
            or "Skipping training-model offload (resident-through-validation" in text
        ),
    }
    payload = {
        "schema_version": 1, "status": "pass" if all(checks.values()) else "fail",
        "arm": args.arm, "mode": args.mode, "checks": checks,
        "timing_steps": len(window), "median_seconds_per_iteration": median,
        "ceiling_seconds_per_iteration": args.max_median_seconds,
        "validation_images": images,
        "activity_value": activity_value,
        "n6r_disabled_low_rms": disabled_low,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit("CL39N 100-step qualification failed")


if __name__ == "__main__":
    main()
