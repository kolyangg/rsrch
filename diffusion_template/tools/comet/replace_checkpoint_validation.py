#!/usr/bin/env python3
"""Replace selected checkpoint-validation steps on one immutable Comet run.

The command is deliberately two-phase: it validates every staged image, table,
bbox record, and scalar before performing any Comet mutation. Metric deletion
is series-wide in Comet, so the untouched step-zero points are backed up and
restored together with the newly calculated checkpoint points.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from comet_ml import API
import pandas as pd
from PIL import Image
import requests


METRIC_NAMES = (
    "manual_val/id_sim",
    "manual_val/text_sim",
    "face_quality/face_detection_rate",
    "face_quality/topiq_face_mean",
    "face_quality/topiq_face_p10",
    "face_quality/topiq_face_coverage",
    "face_quality/topiq_mean",
    "face_quality/musiq_mean",
    "face_quality/maniqa_mean",
)
ID_TABLE_COLUMNS = (
    "validation_step",
    "partition",
    "image_index",
    "output_key",
    "identity",
    "prompt",
    "seed",
    "generated_image_count",
    "id_sim",
)
FACE_ALIASES = {
    "topiq_nr_face": "topiq_face",
    "topiq_nr": "topiq",
    "musiq": "musiq",
    "maniqa_pipal": "maniqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed replacement of selected validation steps on Comet."
    )
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument("--sidecar-name", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--images-per-step", type=int, default=96)
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--base-url", default="https://www.comet.com")
    parser.add_argument("--verify-attempts", type=int, default=30)
    parser.add_argument("--verify-delay", type=float, default=10.0)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_steps(raw: str) -> list[int]:
    result = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not result or result != sorted(set(result)) or any(step <= 0 for step in result):
        raise ValueError("--steps must be a strictly increasing list of positive steps")
    return result


def normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def face_quality_scalars(path: Path, step: int) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("steps", {}).get(str(step))
    if not isinstance(result, dict) or int(result.get("image_count", -1)) != 96:
        raise ValueError(f"Incomplete face-quality result at step {step}: {path}")
    metrics = result.get("metrics", {})
    values = {
        "face_quality/face_detection_rate": float(result["face_detection_rate"]),
    }
    for source, alias in FACE_ALIASES.items():
        metric = metrics.get(source)
        if not isinstance(metric, dict):
            raise ValueError(f"Face-quality metric {source!r} is missing at step {step}")
        if source == "topiq_nr_face":
            values[f"face_quality/{alias}_mean"] = float(metric["mean"])
            values[f"face_quality/{alias}_p10"] = float(metric["p10"])
            values[f"face_quality/{alias}_coverage"] = (
                float(metric["count"]) / float(result["image_count"])
            )
        elif source in {"topiq_nr", "musiq", "maniqa_pipal"}:
            values[f"face_quality/{alias}_mean"] = float(metric["mean"])
    return values


def parse_logged_scalar(log_path: Path, metric_name: str) -> float:
    payload = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        re.compile(
            rf"^\s+{re.escape(metric_name)}:\s+([-+0-9.eE]+)\s*$",
            re.MULTILINE,
        ),
        re.compile(
            rf"^Step\s+\d+:\s+{re.escape(metric_name)}\s+=\s+"
            rf"([-+0-9.eE]+)\s*$",
            re.MULTILINE,
        ),
    )
    matches = [float(value) for pattern in patterns for value in pattern.findall(payload)]
    if not matches:
        raise ValueError(
            f"Expected {metric_name} in {log_path}, found no values"
        )
    if any(
        not math.isclose(matches[0], value, rel_tol=1e-9, abs_tol=1e-12)
        for value in matches[1:]
    ):
        raise ValueError(
            f"Conflicting {metric_name} values in {log_path}: {matches}"
        )
    return matches[0]


def validate_bbox_payload(path: Path, expected_keys: set[str], step: int) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"Dynamic bbox key mismatch at step {step}: "
            f"actual={len(payload) if isinstance(payload, dict) else 'invalid'} "
            f"expected={len(expected_keys)}"
        )
    for key, record in payload.items():
        bbox = record.get("face_crop_new") if isinstance(record, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Missing dynamic face bbox for {key} at step {step}")
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if not (0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024):
            raise ValueError(f"Invalid dynamic face bbox for {key} at step {step}: {bbox}")
    return sha256(path)


def validate_stage(args: argparse.Namespace, steps: list[int]) -> dict[str, Any]:
    staged: dict[str, Any] = {}
    canonical_names: set[str] | None = None
    for step in steps:
        tag = f"{step:06d}"
        step_root = args.staging_root / f"step_{tag}"
        run_name = f"{args.sidecar_name}_step_{tag}"
        run_dir = step_root / run_name
        image_root = run_dir / "val_images" / "manual_val"
        images = sorted(image_root.glob(f"step_{step}_batch_*/*.png"))
        if len(images) != args.images_per_step:
            raise ValueError(
                f"Step {step} has {len(images)} staged images; "
                f"expected {args.images_per_step}"
            )
        names = [path.name for path in images]
        if len(names) != len(set(names)):
            raise ValueError(f"Step {step} has duplicate staged image names")
        with Image.open(images[0]) as first:
            expected_size = first.size
        for path in images:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != expected_size:
                    raise ValueError(f"Image-size drift at step {step}: {path}")
        name_set = set(names)
        if canonical_names is None:
            canonical_names = name_set
        elif name_set != canonical_names:
            raise ValueError(f"Validation output-key set drifted at step {step}")

        table_path = run_dir / "validation_tables" / (
            f"id_sim__manual_val__step_{step:06d}.csv"
        )
        table = pd.read_csv(table_path)
        if list(table.columns) != list(ID_TABLE_COLUMNS) or len(table) != args.images_per_step:
            raise ValueError(f"Invalid per-image ID table at step {step}: {table_path}")
        if list(table["image_index"].astype(int)) != list(range(args.images_per_step)):
            raise ValueError(f"Per-image ID indices drifted at step {step}")
        if set(table["output_key"].astype(str)) != name_set:
            raise ValueError(f"ID table/image output keys disagree at step {step}")

        quality_root = run_dir / "face_quality" / "manual_val" / f"step_{step:08d}"
        quality_json = quality_root / "face_quality_metrics.json"
        quality_csv = quality_root / "face_quality_per_image.csv"
        quality_rows = pd.read_csv(quality_csv)
        if len(quality_rows) != args.images_per_step:
            raise ValueError(f"Face-quality table row mismatch at step {step}")

        log_path = step_root / "validation.log"
        id_mean = float(table["id_sim"].mean())
        logged_id_mean = parse_logged_scalar(log_path, "manual_val/id_sim")
        if not math.isclose(id_mean, logged_id_mean, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"Per-image and aggregate ID_sim disagree at step {step}: "
                f"{id_mean} != {logged_id_mean}"
            )
        metrics = {
            "manual_val/id_sim": id_mean,
            "manual_val/text_sim": parse_logged_scalar(
                log_path, "manual_val/text_sim"
            ),
        }
        metrics.update(face_quality_scalars(quality_json, step))
        if set(metrics) != set(METRIC_NAMES):
            raise ValueError(f"Unexpected metric set at step {step}: {sorted(metrics)}")

        bbox_path = step_root / "bbox_manual_auto.json"
        bbox_sha = validate_bbox_payload(bbox_path, name_set, step)
        staged[str(step)] = {
            "run_dir": str(run_dir.resolve()),
            "image_size": list(expected_size),
            "images": [
                {
                    "path": str(path.resolve()),
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in images
            ],
            "id_table": {
                "path": str(table_path.resolve()),
                "file_name": table_path.name,
                "file_size": table_path.stat().st_size,
                "sha256": sha256(table_path),
            },
            "face_quality_table": {
                "path": str(quality_csv.resolve()),
                "file_name": (
                    f"face_quality_details__manual_val__step_{step:08d}.csv"
                ),
                "file_size": quality_csv.stat().st_size,
                "sha256": sha256(quality_csv),
            },
            "bbox": {
                "path": str(bbox_path.resolve()),
                "sha256": bbox_sha,
                "entries": args.images_per_step,
                "protocol": "checkpoint-current PhotoMaker-only pass, then CPU face detection",
            },
            "metrics": metrics,
        }
    return staged


class RestClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = api_key

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/rest/v2{path}", params=params, timeout=120
        )
        response.raise_for_status()
        return response.json()

    def delete_metric(self, experiment_key: str, metric_name: str) -> None:
        response = self.session.post(
            f"{self.base_url}/api/rest/v2/write/experiment/metric/delete",
            json={"experimentKey": experiment_key, "metricName": metric_name},
            timeout=120,
        )
        response.raise_for_status()


def metric_history(client: RestClient, key: str, name: str) -> list[dict[str, Any]]:
    payload = client.get(
        "/experiment/metrics/get-metric", experimentKey=key, metricName=name
    )
    return [entry for entry in payload.get("metrics", []) if entry.get("step") is not None]


def asset_list(client: RestClient, key: str) -> list[dict[str, Any]]:
    return client.get(
        "/experiment/asset/list", experimentKey=key, type="all"
    ).get("assets", [])


def target_assets(
    assets: list[dict[str, Any]], staged: dict[str, Any], steps: list[int]
) -> list[dict[str, Any]]:
    expected_images = {
        step: {item["file_name"] for item in staged[str(step)]["images"]}
        for step in steps
    }
    expected_other = {
        staged[str(step)]["id_table"]["file_name"] for step in steps
    } | {
        staged[str(step)]["face_quality_table"]["file_name"] for step in steps
    }
    selected = []
    for asset in assets:
        raw_step = asset.get("step")
        step = int(raw_step) if raw_step is not None else None
        file_name = str(asset.get("fileName") or "")
        if str(asset.get("type")) == "image" and step in expected_images:
            if file_name not in expected_images[step]:
                raise ValueError(
                    f"Refusing to delete unexpected image at step {step}: {file_name}"
                )
            selected.append(asset)
        elif file_name in expected_other:
            selected.append(asset)
    return selected


def value_map(history: list[dict[str, Any]]) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for entry in history:
        result[int(entry["step"])].append(float(entry["metricValue"]))
    return result


def wait_for_assets(
    client: RestClient,
    key: str,
    staged: dict[str, Any],
    steps: list[int],
    attempts: int,
    delay: float,
) -> None:
    for attempt in range(attempts):
        assets = asset_list(client, key)
        images_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for asset in assets:
            by_name[str(asset.get("fileName") or "")].append(asset)
            if str(asset.get("type")) == "image" and asset.get("step") is not None:
                images_by_step[int(asset["step"])].append(asset)
        failures = []
        for step in steps:
            expected_images = {
                item["file_name"]: item for item in staged[str(step)]["images"]
            }
            actual = images_by_step.get(step, [])
            if len(actual) != len(expected_images) or {
                str(asset.get("fileName") or "") for asset in actual
            } != set(expected_images):
                failures.append(f"images@{step}")
            for kind in ("id_table", "face_quality_table"):
                record = staged[str(step)][kind]
                matches = by_name.get(record["file_name"], [])
                if len(matches) != 1:
                    failures.append(record["file_name"])
                    continue
                metadata = normalize_metadata(matches[0].get("metadata"))
                if metadata.get("sha256") != record["sha256"]:
                    failures.append(f"sha256:{record['file_name']}")
        if not failures:
            return
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise RuntimeError(f"Comet asset verification failed: {failures[:20]}")


def wait_for_metrics(
    client: RestClient,
    key: str,
    expected: dict[str, dict[int, float]],
    attempts: int,
    delay: float,
) -> None:
    for attempt in range(attempts):
        failures = []
        for name, by_step in expected.items():
            actual = value_map(metric_history(client, key, name))
            if set(actual) != set(by_step):
                failures.append(name)
                continue
            for step, value in by_step.items():
                if len(actual[step]) != 1 or not math.isclose(
                    actual[step][0], value, rel_tol=1e-9, abs_tol=1e-12
                ):
                    failures.append(f"{name}@{step}")
        if not failures:
            return
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise RuntimeError(f"Comet metric verification failed: {failures[:20]}")


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("COMET_API_KEY is required")
    if args.images_per_step != 96:
        raise ValueError("This replacement is locked to the fixed 96-image panel")
    steps = parse_steps(args.steps)
    staging_root = args.staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    staged = validate_stage(args, steps)
    stage_manifest = {
        "schema_version": 1,
        "kind": "dynamic_mask_checkpoint_validation_replacement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_key": args.experiment_key,
        "expected_project": args.expected_project,
        "expected_run_name": args.expected_run_name,
        "sidecar_name": args.sidecar_name,
        "untouched_steps": [0],
        "replacement_steps": steps,
        "images_per_step": args.images_per_step,
        "staged": staged,
    }
    stage_manifest_path = staging_root / "replacement_manifest.json"
    write_json(stage_manifest_path, stage_manifest)

    client = RestClient(args.api_key, args.base_url)
    metadata = client.get("/experiment/metadata", experimentKey=args.experiment_key)
    if str(metadata.get("projectName") or "") != args.expected_project:
        raise ValueError("Live Comet project does not match the approved target")
    if str(metadata.get("experimentName") or "") != args.expected_run_name:
        raise ValueError("Live Comet run name does not match the approved target")

    backup_path = staging_root / "comet_before_replacement.json"
    current_assets = asset_list(client, args.experiment_key)
    selected_assets = target_assets(current_assets, staged, steps)
    if backup_path.is_file():
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        if backup.get("experiment_key") != args.experiment_key:
            raise ValueError("Existing replacement backup belongs to another experiment")
    else:
        histories = {
            name: metric_history(client, args.experiment_key, name)
            for name in METRIC_NAMES
        }
        expected_original_steps = {0, *steps}
        for name, history in histories.items():
            by_step = value_map(history)
            if set(by_step) != expected_original_steps or any(
                len(values) != 1 for values in by_step.values()
            ):
                raise ValueError(
                    f"Original metric history is not the expected 11-point series: {name}"
                )
        for step in steps:
            images = [
                asset
                for asset in selected_assets
                if str(asset.get("type")) == "image"
                and int(asset.get("step", -1)) == step
            ]
            if len(images) != args.images_per_step:
                raise ValueError(
                    f"Original Comet step {step} has {len(images)} images; expected 96"
                )
        backup = {
            "schema_version": 1,
            "kind": "pre_replacement_comet_manifest",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_key": args.experiment_key,
            "metadata": metadata,
            "metric_histories": histories,
            "target_assets": selected_assets,
        }
        write_json(backup_path, backup)

    step_zero: dict[str, float] = {}
    for name in METRIC_NAMES:
        by_step = value_map(backup["metric_histories"][name])
        if len(by_step.get(0, [])) != 1:
            raise ValueError(f"Backup has no unique step-zero point for {name}")
        step_zero[name] = by_step[0][0]
    expected_metrics = {
        name: {0: step_zero[name], **{step: staged[str(step)]["metrics"][name] for step in steps}}
        for name in METRIC_NAMES
    }

    print(
        "COMET_REPLACEMENT_PREFLIGHT_OK "
        f"key={args.experiment_key} steps={','.join(map(str, steps))} "
        f"images={len(steps) * args.images_per_step}"
    )
    if not args.write:
        print("COMET_REPLACEMENT_DRY_RUN_COMPLETE")
        return 0

    api = API(api_key=args.api_key, cache=False)
    experiment = api.get_experiment_by_key(args.experiment_key)
    if experiment is None:
        raise RuntimeError("Comet APIExperiment lookup failed")

    # 5 Aug 2026 - AICODE-NOTE: delete only the exact staged output-key set and
    # exact validation-table filenames. Step zero and every training asset stay intact.
    current_assets = asset_list(client, args.experiment_key)
    for asset in target_assets(current_assets, staged, steps):
        experiment.delete_asset(str(asset["assetId"]))
    for name in METRIC_NAMES:
        client.delete_metric(args.experiment_key, name)

    for attempt in range(args.verify_attempts):
        remaining = target_assets(asset_list(client, args.experiment_key), staged, steps)
        metric_points = sum(
            len(metric_history(client, args.experiment_key, name))
            for name in METRIC_NAMES
        )
        if not remaining and metric_points == 0:
            break
        if attempt + 1 < args.verify_attempts:
            time.sleep(args.verify_delay)
    else:
        raise RuntimeError(
            "Comet deletion did not converge: "
            f"assets={len(remaining)} metric_points={metric_points}"
        )

    for step in [0, *steps]:
        experiment.log_metrics(
            {name: expected_metrics[name][step] for name in METRIC_NAMES}, step=step
        )
    for step in steps:
        for image_record in staged[str(step)]["images"]:
            result = experiment.log_image(
                image_record["path"],
                image_name=image_record["file_name"],
                step=step,
                overwrite=False,
                metadata={
                    "schema_version": 1,
                    "kind": "dynamic_mask_revalidated_image",
                    "validation_step": step,
                    "sha256": image_record["sha256"],
                    "source_sidecar": args.sidecar_name,
                },
            )
            if result is None:
                raise RuntimeError(f"Comet rejected image upload at step {step}")
        id_record = staged[str(step)]["id_table"]
        if experiment.log_asset(
            id_record["path"],
            step=step,
            name=id_record["file_name"],
            overwrite=False,
            ftype="dataframe",
            metadata={
                "sha256": id_record["sha256"],
                "row_count": args.images_per_step,
                "validation_step": step,
                "partition": "manual_val",
                "source": args.sidecar_name,
            },
        ) is None:
            raise RuntimeError(f"Comet rejected ID table upload at step {step}")
        quality_record = staged[str(step)]["face_quality_table"]
        if experiment.log_asset(
            quality_record["path"],
            step=step,
            name=quality_record["file_name"],
            overwrite=False,
            metadata={
                "schema_version": 1,
                "kind": "face_quality_per_image_metrics",
                "sha256": quality_record["sha256"],
                "row_count": args.images_per_step,
                "validation_step": step,
                "partition": "manual_val",
                "source": args.sidecar_name,
            },
        ) is None:
            raise RuntimeError(f"Comet rejected face-quality table at step {step}")

    wait_for_metrics(
        client,
        args.experiment_key,
        expected_metrics,
        args.verify_attempts,
        args.verify_delay,
    )
    wait_for_assets(
        client,
        args.experiment_key,
        staged,
        steps,
        args.verify_attempts,
        args.verify_delay,
    )

    completed = dict(stage_manifest)
    completed["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    completed["status"] = "verified_on_comet"
    completed_path = staging_root / "replacement_verified.json"
    write_json(completed_path, completed)
    audit_name = f"{args.sidecar_name}__replacement_verified.json"
    for asset in asset_list(client, args.experiment_key):
        if str(asset.get("fileName") or "") == audit_name:
            experiment.delete_asset(str(asset["assetId"]))
    experiment.log_asset(
        str(completed_path),
        name=audit_name,
        overwrite=False,
        metadata={
            "kind": "dynamic_mask_validation_replacement_audit",
            "experiment_key": args.experiment_key,
            "replacement_steps": steps,
            "sha256": sha256(completed_path),
        },
    )
    experiment.log_other(
        "validation_replacement_comment",
        (
            "Steps 2000-20000 were regenerated from the saved E10 checkpoints "
            "with a checkpoint-current PhotoMaker-only locator pass and fresh "
            "automatic face masks; the corresponding images, ID/text metrics, "
            "face-quality metrics, and per-image CSV assets were replaced. "
            "Step 0 was preserved."
        ),
    )
    print(
        "COMET_REPLACEMENT_VERIFIED "
        f"key={args.experiment_key} steps={','.join(map(str, steps))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
