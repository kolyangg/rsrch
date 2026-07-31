#!/usr/bin/env python3
"""Read-only readiness checks for a trainer-native full-96 evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from full96_protocol import (
    EXPECTED_AUTOMATIC_ENTRIES,
    load_bbox_protocol,
    load_object,
    sha256,
    validate_static_inputs,
)

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COMET_KEY = re.compile(r"^[A-Za-z0-9]{32}$")


def require_run_name(value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"Unsafe run name: {value!r}")
    return value


def require_comet_record(path: Path, expected_key: str | None = None) -> str:
    record = load_object(path)
    key = str((record.get("comet") or {}).get("experiment_key", ""))
    if not COMET_KEY.fullmatch(key):
        raise ValueError(f"Missing immutable Comet key: {path}")
    if expected_key is not None and key != expected_key:
        raise ValueError(
            f"Comet key mismatch in {path}: expected {expected_key}, found {key}"
        )
    return key


def count_pngs(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix == ".png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-comet-key", required=True)
    parser.add_argument("--bbox-manual", type=Path, required=True)
    parser.add_argument("--auto-min", type=int, default=12)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="Defaults to <project-root>/../dataset_full/val_dataset",
    )
    parser.add_argument(
        "--require-completed-eval",
        help="Also require this full-96 evaluation run to be complete",
    )
    parser.add_argument(
        "--required-eval-kind",
        choices=("reproduction", "intervention"),
        default="reproduction",
        help=(
            "Expected relationship between the required evaluation and its "
            "source first batch. Defaults to exact reproduction."
        ),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_run = require_run_name(args.source_run)
    if not COMET_KEY.fullmatch(args.source_comet_key):
        raise ValueError("source Comet key must contain 32 letters/digits")

    source_dir = project_root / "saved" / source_run
    checkpoint_path = source_dir / "checkpoint-epoch8.pth"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise ValueError(f"Missing endpoint checkpoint: {checkpoint_path}")
    if not (source_dir / "config.yaml").is_file():
        raise ValueError(f"Missing source config: {source_dir / 'config.yaml'}")
    require_comet_record(
        source_dir / "comet_experiment.json",
        expected_key=args.source_comet_key,
    )

    source_images = (
        source_dir / "val_images" / "manual_val" / "step_4000_batch_0"
    )
    if not source_images.is_dir() or count_pngs(source_images) != 12:
        raise ValueError("Source endpoint does not contain exactly 12 validation PNGs")

    manual_path = args.bbox_manual.resolve()
    require_complete = bool(args.require_completed_eval)
    _, automatic, auto_path, routing = load_bbox_protocol(
        manual_path,
        auto_min=args.auto_min,
        require_complete=require_complete,
    )
    validation_data_dir = (
        args.validation_data_dir.resolve()
        if args.validation_data_dir is not None
        else (project_root.parent / "dataset_full" / "val_dataset").resolve()
    )
    static_inputs = validate_static_inputs(validation_data_dir)

    if args.require_completed_eval:
        completed_name = require_run_name(args.require_completed_eval)
        completed_dir = project_root / "saved" / completed_name
        completed_record_path = completed_dir / "comet_experiment.json"
        require_comet_record(completed_record_path)
        completed_record = load_object(completed_record_path)
        result = completed_record.get("validation_result")
        if not isinstance(result, dict):
            raise ValueError(
                f"Required evaluation is not finalized: {args.require_completed_eval}"
            )
        # 26 Jul 2026 - AICODE-NOTE: A documented fixed-checkpoint intervention
        # must change the source pixels. Keep exact reproduction as the default,
        # and accept changed pixels only through the explicit intervention mode.
        reproduced_source = result.get("first_batch_reproduced_source")
        # 27 Jul 2026 - Multi-checkpoint full-96 records name this relationship
        # for the compared endpoint explicitly; it is the same reproduction
        # invariant as the older single-checkpoint field.
        if (
            reproduced_source is None
            and result.get("kind") == "multi_checkpoint_full96"
        ):
            reproduced_source = result.get(
                "step4000_first_batch_reproduced_source"
            )
        if args.required_eval_kind == "reproduction":
            relationship_is_valid = reproduced_source is True
        else:
            relationship_is_valid = (
                reproduced_source is False
                and result.get("first_batch_source_kind")
                == "fixed_checkpoint_intervention"
                and bool(result.get("intervention_label"))
            )
        if not relationship_is_valid:
            raise ValueError(
                "Required evaluation has the wrong source-pixel relationship: "
                f"expected {args.required_eval_kind}, "
                f"found reproduced_source={reproduced_source!r}"
            )
        completed_images = completed_dir / "val_images" / "manual_val"
        batch_dirs = sorted(completed_images.glob("step_4000_batch_*"))
        if len(batch_dirs) != 8 or sum(count_pngs(path) for path in batch_dirs) != 96:
            raise ValueError(
                f"Required evaluation is incomplete: {args.require_completed_eval}"
            )
        if routing["automatic_entries"] != EXPECTED_AUTOMATIC_ENTRIES:
            raise ValueError("Canonical automatic bbox cache is not complete")
        if int(result.get("automatic_bbox_entries", -1)) != routing[
            "automatic_entries"
        ]:
            raise ValueError("Required evaluation recorded a different auto count")
        if int(result.get("force_manual_entries", -1)) != routing[
            "force_manual_entries"
        ]:
            raise ValueError("Required evaluation recorded a different manual count")
        if int(result.get("routing_entries", -1)) != routing["routing_entries"]:
            raise ValueError("Required evaluation recorded incomplete bbox routing")
        if str(result.get("manual_bbox_sha256", "")) != sha256(manual_path):
            raise ValueError("Required evaluation recorded a different manual map")
        if result.get("static_inputs") != static_inputs:
            raise ValueError("Required evaluation recorded different static inputs")
        comet_verification = result.get("comet_verification")
        if (
            not isinstance(comet_verification, dict)
            and result.get("kind") == "multi_checkpoint_full96"
        ):
            step_result = (result.get("step_results") or {}).get("4000") or {}
            comet_verification = step_result.get("comet_verification")
        if (
            not isinstance(comet_verification, dict)
            or not bool(comet_verification.get("verified"))
            or int(comet_verification.get("resolved_step", -1)) != 4000
            or int(comet_verification.get("downloaded_images", -1)) != 96
        ):
            raise ValueError("Required evaluation has no complete Comet verification")
        completed_auto_sha256 = str(result.get("automatic_bbox_sha256", ""))
        current_auto_sha256 = sha256(auto_path)
        if completed_auto_sha256 != current_auto_sha256:
            raise ValueError(
                "Canonical automatic bbox cache changed after the required "
                f"evaluation: expected {completed_auto_sha256}, found "
                f"{current_auto_sha256}"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 8:
        raise ValueError("Endpoint checkpoint is not epoch 8")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Endpoint checkpoint has no state_dict")

    print(
        "FULL96_PREREQUISITES_OK "
        f"source={source_run} auto_bbox_entries={len(automatic)} "
        f"routing_entries={routing['routing_entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
