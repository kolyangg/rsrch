#!/usr/bin/env python3
"""Verify one 96-image checkpoint appended to an existing Comet trajectory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from finalize_full96_eval_record import (
    expected_output_names,
    load_bbox_protocol,
    load_object,
    local_pixel_manifest,
    manifest_sha256,
    sha256,
    validate_static_inputs,
    verify_comet_export,
)


EXPECTED_CONTINUATION_STEPS = tuple(range(6000, 20001, 2000))
BASE_STEPS = [0, 1000, 2000, 3000, 4000]


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".full96-continuation-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--continuation-record", type=Path, required=True)
    parser.add_argument("--bbox-manual", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comet-export", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.step not in EXPECTED_CONTINUATION_STEPS:
        raise ValueError(
            f"Continuation step must be one of {EXPECTED_CONTINUATION_STEPS}, "
            f"found {args.step}"
        )

    base_record_path = args.base_record.resolve()
    continuation_record_path = args.continuation_record.resolve()
    bbox_path = args.bbox_manual.resolve()
    images_root = args.images_root.resolve()
    checkpoint = args.checkpoint.resolve()
    comet_export = args.comet_export.resolve()

    base_record = load_object(base_record_path)
    experiment_key = str((base_record.get("comet") or {}).get("experiment_key", ""))
    run_name = str(base_record.get("run_name", ""))
    base_result = base_record.get("validation_result") or {}
    if len(experiment_key) != 32:
        raise ValueError("Base validation record has no immutable Comet key")
    if base_result.get("optimizer_steps") != BASE_STEPS:
        raise ValueError("Base validation record is not the completed 0/1k/2k/3k/4k run")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Continuation checkpoint is missing: {checkpoint}")
    if not comet_export.is_file() or comet_export.stat().st_size == 0:
        raise ValueError(f"Comet export is missing: {comet_export}")

    manual, _, auto_path, routing = load_bbox_protocol(
        bbox_path,
        auto_min=95,
        require_complete=True,
    )
    validation_data_dir = bbox_path.parent.parent.parent
    static_inputs = validate_static_inputs(validation_data_dir)
    expected_names = expected_output_names(set(manual))

    batch_dirs = sorted(images_root.glob(f"step_{args.step}_batch_*"))
    image_count = sum(
        1
        for directory in batch_dirs
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".png"
    )
    if len(batch_dirs) != 8 or image_count != 96:
        raise ValueError(
            f"Step {args.step}: expected eight batches and 96 PNGs, found "
            f"{len(batch_dirs)} and {image_count}"
        )

    local_pixels = local_pixel_manifest(batch_dirs, expected_names=expected_names)
    comet_verification = verify_comet_export(
        comet_export,
        experiment_key=experiment_key,
        run_name=run_name,
        expected_pixel_manifest=local_pixels,
        optimizer_step=args.step,
    )
    step_result = {
        "batch_count": 8,
        "image_count": 96,
        "pixel_manifest_sha256": manifest_sha256(local_pixels),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "comet_verification": comet_verification,
    }

    if args.verify_only:
        print(
            "FULL96_CONTINUATION_STEP_VERIFIED "
            f"run={run_name} comet_key={experiment_key} step={args.step}"
        )
        return 0

    record: dict = {}
    if continuation_record_path.is_file():
        record = load_object(continuation_record_path)
        if record.get("run_name") != run_name:
            raise ValueError("Continuation record run name changed")
        if record.get("comet_experiment_key") != experiment_key:
            raise ValueError("Continuation record Comet key changed")

    step_results = dict(record.get("step_results") or {})
    existing = step_results.get(str(args.step))
    if existing is not None and existing != step_result:
        raise ValueError(f"Continuation step {args.step} was already recorded differently")
    step_results[str(args.step)] = step_result

    now = datetime.now(timezone.utc).isoformat()
    record.update(
        {
            "schema_version": 1,
            "kind": "appended_full96_continuation",
            "run_name": run_name,
            "comet_experiment_key": experiment_key,
            "created_at_utc": record.get("created_at_utc", now),
            "updated_at_utc": now,
            "base_optimizer_steps": BASE_STEPS,
            "scheduled_optimizer_steps": list(EXPECTED_CONTINUATION_STEPS),
            "completed_optimizer_steps": sorted(int(step) for step in step_results),
            "images_per_step": 96,
            "batch_size": 12,
            "manual_bbox_sha256": sha256(bbox_path),
            "automatic_bbox_sha256": sha256(auto_path),
            "automatic_bbox_entries": routing["automatic_entries"],
            "force_manual_entries": routing["force_manual_entries"],
            "routing_entries": routing["routing_entries"],
            "static_inputs": static_inputs,
            "step_results": step_results,
        }
    )
    write_atomic(continuation_record_path, record)
    print(
        "FULL96_CONTINUATION_STEP_RECORDED "
        f"run={run_name} comet_key={experiment_key} step={args.step} "
        f"record={continuation_record_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
