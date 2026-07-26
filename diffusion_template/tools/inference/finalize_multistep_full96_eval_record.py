#!/usr/bin/env python3
"""Verify and finalize a five-checkpoint full-96 Comet evaluation."""

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
    png_hashes,
    sha256,
    validate_static_inputs,
    verify_comet_export,
)

EXPECTED_STEPS = (0, 1000, 2000, 3000, 4000)


def write_atomic(path: Path, value: dict) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".full96-multistep-result-",
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


def parse_step_paths(values: list[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        raw_step, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"{label} entry must use STEP=PATH: {value!r}")
        step = int(raw_step)
        if step in result:
            raise ValueError(f"Duplicate {label} step: {step}")
        path = Path(raw_path).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing {label} file for step {step}: {path}")
        result[step] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--bbox-manual", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--source-step4000-images", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="Required for steps 1000, 2000, 3000, and 4000",
    )
    parser.add_argument(
        "--comet-export",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="Required for all five validation steps",
    )
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="Defaults to the val_dataset directory containing protocols/",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    record_path = args.record.resolve()
    manual_path = args.bbox_manual.resolve()
    images_root = args.images_root.resolve()
    record = load_object(record_path)
    experiment_key = str((record.get("comet") or {}).get("experiment_key", ""))
    run_name = str(record.get("run_name", ""))
    if len(experiment_key) != 32:
        raise ValueError("Canonical record has no immutable Comet experiment key")

    checkpoints = parse_step_paths(args.checkpoint, label="checkpoint")
    expected_checkpoint_steps = set(EXPECTED_STEPS) - {0}
    if set(checkpoints) != expected_checkpoint_steps:
        raise ValueError(
            "Checkpoint steps must be exactly "
            f"{sorted(expected_checkpoint_steps)}, found {sorted(checkpoints)}"
        )
    comet_exports = parse_step_paths(args.comet_export, label="Comet export")
    if set(comet_exports) != set(EXPECTED_STEPS):
        raise ValueError(
            f"Comet export steps must be {list(EXPECTED_STEPS)}, "
            f"found {sorted(comet_exports)}"
        )

    manual, _, auto_path, routing = load_bbox_protocol(
        manual_path,
        auto_min=95,
        require_complete=True,
    )
    validation_data_dir = (
        args.validation_data_dir.resolve()
        if args.validation_data_dir is not None
        else manual_path.parent.parent.parent
    )
    static_inputs = validate_static_inputs(validation_data_dir)
    expected_names = expected_output_names(set(manual))

    step_results: dict[str, dict] = {}
    for step in EXPECTED_STEPS:
        batch_dirs = sorted(images_root.glob(f"step_{step}_batch_*"))
        image_count = sum(
            1
            for directory in batch_dirs
            for path in directory.iterdir()
            if path.is_file() and path.suffix == ".png"
        )
        if len(batch_dirs) != 8 or image_count != 96:
            raise ValueError(
                f"Step {step}: expected eight batches and 96 PNGs, found "
                f"{len(batch_dirs)} and {image_count}"
            )
        local_pixels = local_pixel_manifest(
            batch_dirs,
            expected_names=expected_names,
        )
        comet_verification = verify_comet_export(
            comet_exports[step],
            experiment_key=experiment_key,
            run_name=run_name,
            expected_pixel_manifest=local_pixels,
            optimizer_step=step,
        )
        step_results[str(step)] = {
            "batch_count": 8,
            "image_count": image_count,
            "pixel_manifest_sha256": manifest_sha256(local_pixels),
            "checkpoint": (
                {
                    "path": str(checkpoints[step]),
                    "sha256": sha256(checkpoints[step]),
                }
                if step > 0
                else {
                    "path": None,
                    "kind": "seeded_initial_state",
                }
            ),
            "comet_verification": comet_verification,
        }

    source_hashes = png_hashes(args.source_step4000_images.resolve())
    endpoint_hashes = png_hashes(images_root / "step_4000_batch_0")
    endpoint_reproduced_source = source_hashes == endpoint_hashes
    if not endpoint_reproduced_source:
        raise ValueError(
            "The step-4000 first batch does not reproduce the source endpoint"
        )

    if args.verify_only:
        print(
            "FULL96_MULTISTEP_RESULT_VERIFIED "
            f"record={record_path} comet_key={experiment_key} "
            f"steps={','.join(str(step) for step in EXPECTED_STEPS)}"
        )
        return 0

    record["validation_result"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "multi_checkpoint_full96",
        "optimizer_steps": list(EXPECTED_STEPS),
        "images_per_step": 96,
        "batch_size": 12,
        "step_results": step_results,
        "step4000_first_batch_reproduced_source": True,
        "source_step4000_png_sha256": source_hashes,
        "manual_bbox_sha256": sha256(manual_path),
        "automatic_bbox_sha256": sha256(auto_path),
        "automatic_bbox_entries": routing["automatic_entries"],
        "force_manual_entries": routing["force_manual_entries"],
        "force_manual_keys": routing["force_manual_keys"],
        "routing_entries": routing["routing_entries"],
        "static_inputs": static_inputs,
    }
    write_atomic(record_path, record)
    print(
        "FULL96_MULTISTEP_RECORD_FINALIZED "
        f"record={record_path} comet_key={experiment_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
