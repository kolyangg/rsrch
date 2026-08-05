#!/usr/bin/env python3
"""Score staged validation images after training and backfill Comet metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ("topiq_nr-face", "topiq_nr", "musiq", "maniqa-pipal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine deferred training-validation manifests, score them once, "
            "and optionally backfill the canonical compact Comet metrics."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--expected-steps", required=True)
    parser.add_argument("--images-per-step", type=int, default=96)
    parser.add_argument("--partition", default="manual_val")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument(
        "--scorer-script",
        type=Path,
        default=PROJECT_ROOT / "tools/inference/calculate_face_quality_metrics.py",
    )
    parser.add_argument(
        "--backfill-script",
        type=Path,
        default=PROJECT_ROOT / "tools/comet/backfill_face_quality_metrics.py",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--upload-per-image-asset", action="store_true")
    parser.add_argument(
        "--nonfatal",
        action="store_true",
        help="Record failure and return success so completed training stays successful.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_unique_csv(value: str, cast) -> list[Any]:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Expected a non-empty unique CSV list, got {value!r}")
    return values


def _experiment_key(run_dir: Path) -> str:
    record_path = run_dir / "comet_experiment.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    key = str((record.get("comet") or {}).get("experiment_key") or "").strip()
    if len(key) != 32:
        raise ValueError(f"Invalid Comet experiment key in {record_path}")
    return key


def _combined_manifest(
    *,
    run_dir: Path,
    experiment_key: str,
    project_name: str,
    partition: str,
    expected_steps: list[int],
    images_per_step: int,
) -> dict[str, Any]:
    manifests = sorted(
        (run_dir / "face_quality" / partition).glob("step_*/input_manifest.json")
    )
    by_step: dict[int, list[dict[str, Any]]] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported staged manifest: {manifest_path}")
        if str(manifest.get("experiment_key") or "") != experiment_key:
            raise ValueError(f"Experiment-key mismatch in {manifest_path}")
        manifest_steps = manifest.get("steps") or {}
        if len(manifest_steps) != 1:
            raise ValueError(f"Expected one step in {manifest_path}")
        step_text, records = next(iter(manifest_steps.items()))
        step = int(step_text)
        if step in by_step:
            raise ValueError(f"Duplicate staged face-quality step: {step}")
        if len(records) != images_per_step:
            raise ValueError(
                f"Step {step} has {len(records)} staged images; "
                f"expected {images_per_step}"
            )
        for record in records:
            image_path = Path(record["local_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if int(record.get("file_size") or -1) != image_path.stat().st_size:
                raise ValueError(f"Staged image size drifted: {image_path}")
            if str(record.get("sha256") or "") != _sha256(image_path):
                raise ValueError(f"Staged image checksum drifted: {image_path}")
        by_step[step] = records

    if set(by_step) != set(expected_steps):
        raise ValueError(
            "Deferred face-quality step mismatch: "
            f"expected={expected_steps}, actual={sorted(by_step)}"
        )
    return {
        "schema_version": 1,
        "kind": "deferred_training_validation_images",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": experiment_key,
        "project_name": project_name,
        "steps": {str(step): by_step[step] for step in sorted(by_step)},
    }


def _run(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    expected_steps = _parse_unique_csv(args.expected_steps, int)
    metrics = _parse_unique_csv(args.metrics, str)
    if args.images_per_step < 1:
        raise ValueError("--images-per-step must be positive")
    for path in (args.scorer_python, args.scorer_script, args.backfill_script):
        if not path.is_file():
            raise FileNotFoundError(path)

    experiment_key = _experiment_key(run_dir)
    manifest = _combined_manifest(
        run_dir=run_dir,
        experiment_key=experiment_key,
        project_name=args.expected_project,
        partition=args.partition,
        expected_steps=expected_steps,
        images_per_step=args.images_per_step,
    )
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "download_manifest.json"
    results_json = results_dir / "face_quality_metrics.json"
    results_csv = results_dir / "face_quality_per_image.csv"
    _atomic_json(manifest_path, manifest)

    # 5 Aug 2026 - AICODE-NOTE: This subprocess is intentionally created only
    # after Accelerate has exited, so PyIQA cannot hold memory, mutate RNG, or
    # abort the optimizer/checkpoint trajectory.
    subprocess.run(
        [
            str(args.scorer_python.absolute()),
            str(args.scorer_script.resolve()),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(results_json),
            "--output-csv",
            str(results_csv),
            "--metrics",
            ",".join(metrics),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--crop-padding",
            str(args.crop_padding),
            "--crop-size",
            str(args.crop_size),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if args.write:
        command = [
            str(args.scorer_python.absolute()),
            str(args.backfill_script.resolve()),
            "--experiment-key",
            experiment_key,
            "--expected-project",
            args.expected_project,
            "--steps",
            ",".join(str(step) for step in expected_steps),
            "--images-per-step",
            str(args.images_per_step),
            "--metrics",
            ",".join(metrics),
            "--work-dir",
            str(work_dir),
            "--scorer-python",
            str(args.scorer_python.absolute()),
            "--scorer-script",
            str(args.scorer_script.resolve()),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--crop-padding",
            str(args.crop_padding),
            "--crop-size",
            str(args.crop_size),
            "--reuse-results",
            "--write",
        ]
        if args.upload_per_image_asset:
            command.append("--upload-per-image-asset")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    return {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": experiment_key,
        "partition": args.partition,
        "steps": expected_steps,
        "images_per_step": args.images_per_step,
        "metrics": metrics,
        "comet_written": bool(args.write),
        "per_image_asset_uploaded": bool(args.write and args.upload_per_image_asset),
        "results_json": str(results_json),
        "results_csv": str(results_csv),
    }


def main() -> int:
    args = parse_args()
    work_dir = args.run_dir.resolve() / "post_training_face_quality"
    status_path = work_dir / "status.json"
    try:
        status = _run(args, work_dir)
    except Exception as error:
        status = {
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "training_affected": False,
        }
        _atomic_json(status_path, status)
        traceback.print_exc()
        print(f"DEFERRED_FACE_QUALITY_FAILED status={status_path}", flush=True)
        return 0 if args.nonfatal else 1
    _atomic_json(status_path, status)
    print(f"DEFERRED_FACE_QUALITY_COMPLETE status={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
