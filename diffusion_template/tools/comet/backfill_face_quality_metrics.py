# 10 Aug 2026 - E13C-PERF-04: Retained post-training face-quality tooling so PyIQA work cannot perturb or delay optimizer steps.
#!/usr/bin/env python3
"""Download exact Comet image steps, score face quality, and backfill scalars.

Writes are opt-in via ``--write``. Existing equal metric values are treated as
idempotent; conflicting or duplicate values abort the backfill.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from export_comet_runs import (
    CometRestClient,
    normalize_export_image_stem,
)


DEFAULT_STEPS = (0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)
DEFAULT_METRICS = ("topiq_nr-face", "topiq_nr", "musiq", "maniqa-pipal")
METRIC_NAMESPACE = "face_quality"
LEGACY_METRIC_NAMESPACE = "manual_val/face_quality"
STEP_ASSET_RE = re.compile(r"^face_quality_step_(\d{6})\.(json|csv)$")
PER_IMAGE_ASSET_FILE_NAME = "face_quality_details__per_image_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill no-reference face-quality metrics into one immutable Comet experiment."
    )
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--steps", default=",".join(str(step) for step in DEFAULT_STEPS))
    parser.add_argument("--images-per-step", type=int, default=96)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scorer-python", type=Path, required=True)
    parser.add_argument(
        "--scorer-script",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "inference"
        / "calculate_face_quality_metrics.py",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--base-url", default="https://www.comet.com")
    parser.add_argument("--download-retries", type=int, default=4)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Reuse an existing complete result JSON/CSV in work-dir without downloading or scoring.",
    )
    parser.add_argument(
        "--cleanup-legacy-layout",
        action="store_true",
        help=(
            "After compact metrics are verified, delete this tool's old "
            "manual_val/face_quality metrics and per-step table assets."
        ),
    )
    parser.add_argument(
        "--upload-per-image-asset",
        action="store_true",
        help=(
            "Upload one API-accessible CSV asset containing every per-image "
            "metric row. This does not create Comet tables or report curves."
        ),
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--keep-images", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Steps must be a non-empty unique list: {value!r}")
    return result


def parse_str_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Metrics must be a non-empty unique list: {value!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path, expected_size: int | None) -> None:
    if expected_size and path.stat().st_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    with Image.open(path) as image:
        image.verify()


def download_with_retry(
    client: CometRestClient,
    experiment_key: str,
    asset_id: str,
    destination: Path,
    expected_size: int | None,
    retries: int,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if destination.is_file():
                validate_image(destination, expected_size)
                return
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.unlink(missing_ok=True)
            client.download_asset(experiment_key, asset_id, partial)
            validate_image(partial, expected_size)
            partial.replace(destination)
            return
        except Exception as error:
            last_error = error
            destination.unlink(missing_ok=True)
            destination.with_suffix(destination.suffix + ".partial").unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download and validate asset {asset_id}") from last_error


def build_download_manifest(
    client: CometRestClient,
    experiment_key: str,
    project_name: str,
    steps: list[int],
    images_per_step: int,
    images_root: Path,
    retries: int,
) -> dict[str, Any]:
    payload = client.get_json(
        "/experiment/asset/list",
        experimentKey=experiment_key,
        type="image",
    )
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for asset in payload.get("assets", []):
        if asset.get("step") is not None:
            by_step[int(asset["step"])].append(asset)

    manifest_steps: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        assets = sorted(
            by_step.get(step, []),
            key=lambda item: (str(item.get("fileName") or ""), str(item.get("assetId") or "")),
        )
        if len(assets) != images_per_step:
            raise ValueError(
                f"Step {step} has {len(assets)} image assets; expected {images_per_step}"
            )
        step_dir = images_root / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index, asset in enumerate(assets):
            asset_id = str(asset["assetId"])
            original_name = str(asset.get("fileName") or asset_id)
            destination = step_dir / (
                f"{index:03d}__{normalize_export_image_stem(original_name)}.png"
            )
            expected_size = int(asset.get("fileSize") or 0) or None
            download_with_retry(
                client,
                experiment_key,
                asset_id,
                destination,
                expected_size,
                retries,
            )
            records.append(
                {
                    "asset_id": asset_id,
                    "file_name": original_name,
                    "local_path": str(destination.resolve()),
                    "file_size": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )
        manifest_steps[str(step)] = records
        print(f"FACE_QUALITY_DOWNLOAD_COMPLETE step={step} images={len(records)}")

    return {
        "schema_version": 1,
        "kind": "exact_comet_image_steps",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": experiment_key,
        "project_name": project_name,
        "steps": manifest_steps,
    }


def run_scorer(
    scorer_python: Path,
    scorer_script: Path,
    manifest_path: Path,
    output_json: Path,
    output_csv: Path,
    metrics: list[str],
    device: str,
    batch_size: int,
    crop_padding: float,
    crop_size: int,
) -> None:
    command = [
        str(scorer_python),
        str(scorer_script),
        "--manifest",
        str(manifest_path),
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--metrics",
        ",".join(metrics),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--crop-padding",
        str(crop_padding),
        "--crop-size",
        str(crop_size),
    ]
    subprocess.run(command, check=True)


def planned_metrics(results: dict[str, Any]) -> dict[int, dict[str, float]]:
    planned: dict[int, dict[str, float]] = {}
    for step_text, step_result in results["steps"].items():
        image_count = float(step_result["image_count"])
        metric_results = step_result["metrics"]
        step_metrics = {
            f"{METRIC_NAMESPACE}/face_detection_rate": float(
                step_result["face_detection_rate"]
            ),
            f"{METRIC_NAMESPACE}/topiq_face_mean": float(
                metric_results["topiq_nr_face"]["mean"]
            ),
            f"{METRIC_NAMESPACE}/topiq_face_p10": float(
                metric_results["topiq_nr_face"]["p10"]
            ),
            f"{METRIC_NAMESPACE}/topiq_face_coverage": (
                float(metric_results["topiq_nr_face"]["count"]) / image_count
            ),
            f"{METRIC_NAMESPACE}/topiq_mean": float(
                metric_results["topiq_nr"]["mean"]
            ),
            f"{METRIC_NAMESPACE}/musiq_mean": float(metric_results["musiq"]["mean"]),
            f"{METRIC_NAMESPACE}/maniqa_mean": float(
                metric_results["maniqa_pipal"]["mean"]
            ),
        }
        planned[int(step_text)] = step_metrics
    return planned


def legacy_metric_names(results: dict[str, Any]) -> set[str]:
    names = {
        f"{LEGACY_METRIC_NAMESPACE}/face_detection_rate",
        f"{LEGACY_METRIC_NAMESPACE}/multi_face_rate",
    }
    names.update(
        f"{LEGACY_METRIC_NAMESPACE}/det_score_{summary}"
        for summary in ("mean", "median", "p10")
    )
    for slug in results["steps"][next(iter(results["steps"]))]["metrics"]:
        names.add(f"{LEGACY_METRIC_NAMESPACE}/{slug}_score_coverage_rate")
        names.update(
            f"{LEGACY_METRIC_NAMESPACE}/{slug}_{summary}"
            for summary in ("mean", "median", "p10")
        )
    return names


def metric_history(
    client: CometRestClient,
    experiment_key: str,
    metric_name: str,
) -> dict[int, list[float]]:
    payload = client.get_json(
        "/experiment/metrics/get-metric",
        experimentKey=experiment_key,
        metricName=metric_name,
    )
    values: dict[int, list[float]] = defaultdict(list)
    for entry in payload.get("metrics", []):
        if entry.get("step") is not None:
            values[int(entry["step"])].append(float(entry["metricValue"]))
    return values


def values_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def log_to_comet(
    client: CometRestClient,
    api_key: str,
    experiment_key: str,
    planned: dict[int, dict[str, float]],
) -> None:
    histories = {
        name: metric_history(client, experiment_key, name)
        for name in sorted({name for metrics in planned.values() for name in metrics})
    }
    writes: dict[int, dict[str, float]] = {}
    for step, metrics in planned.items():
        writes[step] = {}
        for name, value in metrics.items():
            existing = histories[name].get(step, [])
            if len(existing) > 1:
                raise ValueError(f"Metric {name} already has duplicate values at step {step}")
            if existing:
                if not values_equal(existing[0], value):
                    raise ValueError(
                        f"Metric {name} conflicts at step {step}: {existing[0]} != {value}"
                    )
                continue
            writes[step][name] = value

    if not any(writes.values()):
        print("FACE_QUALITY_COMPACT_METRICS_ALREADY_PRESENT")
        return

    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(
        api_key=api_key,
        previous_experiment=experiment_key,
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_output_logging=None,
        log_code=False,
        log_graph=False,
    )
    try:
        for step in sorted(planned):
            if writes[step]:
                experiment.log_metrics(writes[step], step=step)
            print(
                f"FACE_QUALITY_COMET_STEP_LOGGED step={step} metrics={len(writes[step])}"
            )
    finally:
        experiment.end()


def verify_logged_metrics(
    client: CometRestClient,
    experiment_key: str,
    planned: dict[int, dict[str, float]],
) -> None:
    pending = {
        (step, name): value
        for step, metrics in planned.items()
        for name, value in metrics.items()
    }
    for attempt in range(12):
        missing: list[str] = []
        by_name = sorted({name for _, name in pending})
        histories = {
            name: metric_history(client, experiment_key, name) for name in by_name
        }
        for (step, name), expected in pending.items():
            values = histories[name].get(step, [])
            if len(values) != 1 or not values_equal(values[0], expected):
                missing.append(f"{name}@{step}")
        if not missing:
            print(
                "FACE_QUALITY_COMET_VERIFIED "
                f"steps={','.join(str(step) for step in sorted(planned))}"
            )
            return
        if attempt < 11:
            time.sleep(10)
    raise RuntimeError(f"Comet verification failed for: {missing[:20]}")


def validate_per_image_csv(
    path: Path,
    steps: list[int],
    images_per_step: int,
) -> int:
    by_step: dict[int, int] = defaultdict(int)
    unique_rows: set[tuple[int, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "step",
            "asset_id",
            "face_detected",
            "topiq_nr_face",
            "topiq_nr",
            "musiq",
            "maniqa_pipal",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Per-image CSV is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            step = int(row["step"])
            key = (step, row["asset_id"])
            if key in unique_rows:
                raise ValueError(f"Duplicate per-image CSV row: {key}")
            unique_rows.add(key)
            by_step[step] += 1

    expected_steps = set(steps)
    if set(by_step) != expected_steps:
        raise ValueError(
            f"Per-image CSV step set mismatch: {sorted(by_step)} != {sorted(steps)}"
        )
    wrong_counts = {
        step: by_step[step]
        for step in steps
        if by_step[step] != images_per_step
    }
    if wrong_counts:
        raise ValueError(f"Per-image CSV count mismatch: {wrong_counts}")
    return len(unique_rows)


def normalize_asset_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def matching_per_image_assets(
    client: CometRestClient,
    experiment_key: str,
) -> list[dict[str, Any]]:
    payload = client.get_json(
        "/experiment/asset/list",
        experimentKey=experiment_key,
        type="all",
    )
    return [
        asset
        for asset in payload.get("assets", [])
        if str(asset.get("fileName") or "") == PER_IMAGE_ASSET_FILE_NAME
    ]


def upload_per_image_asset(
    client: CometRestClient,
    api_key: str,
    experiment_key: str,
    csv_path: Path,
    steps: list[int],
    images_per_step: int,
) -> None:
    row_count = validate_per_image_csv(csv_path, steps, images_per_step)
    csv_sha256 = sha256(csv_path)
    metadata = {
        "schema_version": 1,
        "kind": "face_quality_per_image_metrics",
        "namespace": "face_quality_details",
        "logical_path": "face_quality_details/per_image_metrics.csv",
        "hidden_in_report_by_default": True,
        "experiment_key": experiment_key,
        "row_count": row_count,
        "steps": steps,
        "images_per_step": images_per_step,
        "sha256": csv_sha256,
    }

    existing = matching_per_image_assets(client, experiment_key)
    if len(existing) > 1:
        raise ValueError(
            f"Duplicate {PER_IMAGE_ASSET_FILE_NAME} assets already exist"
        )
    if existing:
        existing_metadata = normalize_asset_metadata(existing[0].get("metadata"))
        existing_size = int(existing[0].get("fileSize") or 0)
        if (
            existing_metadata.get("sha256") != csv_sha256
            or existing_metadata.get("row_count") != row_count
            or existing_size != csv_path.stat().st_size
        ):
            raise ValueError(
                f"Existing {PER_IMAGE_ASSET_FILE_NAME} conflicts with local CSV"
            )
        print(
            "FACE_QUALITY_PER_IMAGE_ASSET_ALREADY_PRESENT "
            f"rows={row_count} sha256={csv_sha256}"
        )
        return

    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(
        api_key=api_key,
        previous_experiment=experiment_key,
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_output_logging=None,
        log_code=False,
        log_graph=False,
    )
    try:
        result = experiment.log_asset(
            str(csv_path),
            file_name=PER_IMAGE_ASSET_FILE_NAME,
            overwrite=False,
            metadata=metadata,
        )
        print(f"FACE_QUALITY_PER_IMAGE_ASSET_QUEUED result={result}")
    finally:
        experiment.end()

    for attempt in range(12):
        uploaded = matching_per_image_assets(client, experiment_key)
        if len(uploaded) == 1:
            uploaded_metadata = normalize_asset_metadata(uploaded[0].get("metadata"))
            if (
                uploaded_metadata.get("sha256") == csv_sha256
                and uploaded_metadata.get("row_count") == row_count
                and int(uploaded[0].get("fileSize") or 0)
                == csv_path.stat().st_size
            ):
                print(
                    "FACE_QUALITY_PER_IMAGE_ASSET_VERIFIED "
                    f"rows={row_count} sha256={csv_sha256}"
                )
                return
        if attempt < 11:
            time.sleep(10)
    raise RuntimeError("Per-image CSV asset verification failed")


def metric_summary_names(
    client: CometRestClient,
    experiment_key: str,
) -> set[str]:
    payload = client.get_json(
        "/experiment/metrics/summary",
        experimentKey=experiment_key,
    )
    return {
        str(entry["name"])
        for entry in payload.get("values", [])
        if entry.get("name")
    }


def delete_metric(
    client: CometRestClient,
    experiment_key: str,
    metric_name: str,
) -> None:
    url = f"{client.base_url}/api/rest/v2/write/experiment/metric/delete"
    response = client.session.post(
        url,
        json={"experimentKey": experiment_key, "metricName": metric_name},
        timeout=client.timeout,
    )
    response.raise_for_status()


def delete_asset(
    client: CometRestClient,
    experiment_key: str,
    asset_id: str,
) -> None:
    url = f"{client.base_url}/api/rest/v2/write/experiment/asset/delete"
    response = client.session.get(
        url,
        params={"experimentKey": experiment_key, "assetId": asset_id},
        timeout=client.timeout,
    )
    response.raise_for_status()


def cleanup_legacy_layout(
    client: CometRestClient,
    experiment_key: str,
    results: dict[str, Any],
    steps: list[int],
) -> None:
    # 27 Jul 2026 - AICODE-NOTE: Cleanup is fail-closed. Only the exact legacy
    # names derived from this result schema and exact per-step filenames may be deleted.
    allowed_legacy_names = legacy_metric_names(results)
    present_legacy_names = {
        name
        for name in metric_summary_names(client, experiment_key)
        if name.startswith(f"{LEGACY_METRIC_NAMESPACE}/")
    }
    unexpected_names = present_legacy_names - allowed_legacy_names
    if unexpected_names:
        raise ValueError(
            f"Refusing to delete unexpected legacy metrics: {sorted(unexpected_names)}"
        )
    for name in sorted(present_legacy_names):
        delete_metric(client, experiment_key, name)
        print(f"FACE_QUALITY_LEGACY_METRIC_DELETED name={name}")

    asset_payload = client.get_json(
        "/experiment/asset/list",
        experimentKey=experiment_key,
        type="all",
    )
    expected_steps = set(steps)
    legacy_assets: list[dict[str, Any]] = []
    for asset in asset_payload.get("assets", []):
        file_name = str(asset.get("fileName") or "")
        match = STEP_ASSET_RE.fullmatch(file_name)
        if not match:
            continue
        if int(match.group(1)) not in expected_steps:
            raise ValueError(f"Refusing to delete unexpected step asset: {file_name}")
        legacy_assets.append(asset)
    for asset in legacy_assets:
        delete_asset(
            client,
            experiment_key,
            str(asset["assetId"]),
        )
        print(
            "FACE_QUALITY_LEGACY_ASSET_DELETED "
            f"name={asset.get('fileName')}"
        )

    for attempt in range(12):
        remaining_metrics = {
            name
            for name in metric_summary_names(client, experiment_key)
            if name.startswith(f"{LEGACY_METRIC_NAMESPACE}/")
        }
        remaining_assets = [
            asset
            for asset in client.get_json(
                "/experiment/asset/list",
                experimentKey=experiment_key,
                type="all",
            ).get("assets", [])
            if STEP_ASSET_RE.fullmatch(str(asset.get("fileName") or ""))
        ]
        if not remaining_metrics and not remaining_assets:
            print(
                "FACE_QUALITY_LEGACY_LAYOUT_REMOVED "
                f"metrics={len(present_legacy_names)} assets={len(legacy_assets)}"
            )
            return
        if attempt < 11:
            time.sleep(5)
    raise RuntimeError(
        "Legacy layout cleanup did not converge: "
        f"metrics={sorted(remaining_metrics)} assets={len(remaining_assets)}"
    )


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("COMET_API_KEY is required")
    steps = parse_int_list(args.steps)
    metrics = parse_str_list(args.metrics)
    if args.images_per_step < 1:
        raise ValueError("--images-per-step must be positive")
    if args.download_retries < 1:
        raise ValueError("--download-retries must be positive")
    if args.upload_per_image_asset and not args.write:
        raise ValueError("--upload-per-image-asset requires --write")
    if not args.scorer_python.is_file():
        raise FileNotFoundError(args.scorer_python)
    if not args.scorer_script.is_file():
        raise FileNotFoundError(args.scorer_script)

    work_dir = args.work_dir.resolve()
    images_root = work_dir / "images"
    results_dir = work_dir / "results"
    manifest_path = work_dir / "download_manifest.json"
    results_json = results_dir / "face_quality_metrics.json"
    results_csv = results_dir / "face_quality_per_image.csv"
    client = CometRestClient(args.api_key, args.base_url, timeout=120)

    metadata = client.get_json(
        "/experiment/metadata",
        experimentKey=args.experiment_key,
    )
    project_name = str(metadata.get("projectName") or "")
    if project_name != args.expected_project:
        raise ValueError(
            f"Experiment is in project {project_name!r}, expected {args.expected_project!r}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_results:
        if not results_json.is_file() or not results_csv.is_file():
            raise FileNotFoundError(
                f"--reuse-results requires {results_json} and {results_csv}"
            )
        print(f"FACE_QUALITY_REUSING_RESULTS json={results_json} csv={results_csv}")
    else:
        manifest = build_download_manifest(
            client,
            args.experiment_key,
            project_name,
            steps,
            args.images_per_step,
            images_root,
            args.download_retries,
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # 27 Jul 2026 - Preserve a venv interpreter symlink here. Path.resolve()
        # follows it into the base Conda interpreter and silently loses venv packages.
        run_scorer(
            args.scorer_python.absolute(),
            args.scorer_script.resolve(),
            manifest_path,
            results_json,
            results_csv,
            metrics,
            args.device,
            args.batch_size,
            args.crop_padding,
            args.crop_size,
        )
    results = json.loads(results_json.read_text(encoding="utf-8"))
    if results.get("experiment_key") != args.experiment_key:
        raise ValueError("Scorer result experiment key does not match")
    if results.get("metric_backend", {}).get("metrics") != metrics:
        raise ValueError("Scorer result metric list does not match --metrics")
    if set(results.get("steps", {})) != {str(step) for step in steps}:
        raise ValueError("Scorer result step set does not match --steps")
    for step in steps:
        summary = results["steps"].get(str(step))
        if not summary or summary["image_count"] != args.images_per_step:
            raise ValueError(f"Scorer result is incomplete at step {step}")

    planned = planned_metrics(results)
    if args.write:
        log_to_comet(
            client,
            args.api_key,
            args.experiment_key,
            planned,
        )
        verify_logged_metrics(client, args.experiment_key, planned)
        if args.upload_per_image_asset:
            upload_per_image_asset(
                client,
                args.api_key,
                args.experiment_key,
                results_csv,
                steps,
                args.images_per_step,
            )
        if args.cleanup_legacy_layout:
            cleanup_legacy_layout(
                client,
                args.experiment_key,
                results,
                steps,
            )
    else:
        print("FACE_QUALITY_DRY_RUN_COMPLETE use --write to append compact metrics")

    if not args.keep_images and images_root.exists():
        shutil.rmtree(images_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
