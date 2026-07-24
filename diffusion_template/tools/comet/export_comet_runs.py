#!/usr/bin/env python3
"""
Export Comet experiment data listed in a JSON manifest.

Usage:
    COMET_API_KEY=... python diffusion_template/tools/comet/export_comet_runs.py
    COMET_API_KEY=... python diffusion_template/tools/comet/export_comet_runs.py --step-number 500

Manifest format:
{
  "runs": [
    {
      "run_id": "COMET_EXPERIMENT_KEY",
      "run_name": "optional local override",
      "step_number": 500
    }
  ]
}

If the requested step is missing for a run, the exporter falls back to the
nearest lower available image step for that run and records the fallback in the
output JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://www.comet.com"
API_ROOT = "/api/rest/v2"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "comet_runs.json"
DEFAULT_OUTPUT_DIR = TEMPLATE_ROOT / "comet_data"
DEFAULT_OUTPUT_JSON_NAME = "comet_runs_export.json"
CHUNK_SIZE = 1024 * 1024
PLACEHOLDER_RUN_ID = "REPLACE_WITH_COMET_EXPERIMENT_KEY"

LOGGER = logging.getLogger("comet_export")

INT_RE = re.compile(r"^[+-]?\d+$")


class ManifestError(ValueError):
    """Raised when the run manifest is invalid."""


class CometAPIError(RuntimeError):
    """Raised when a Comet REST call fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Comet run metadata, parameters, metric history, and step-filtered "
            "image assets for the runs defined in a JSON manifest. If an exact "
            "image step is missing, the nearest lower available step is used."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to the JSON manifest. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for exported JSON and image folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path for the combined export JSON. "
            f"Default: <output-dir>/{DEFAULT_OUTPUT_JSON_NAME}"
        ),
    )
    parser.add_argument(
        "--step-number",
        type=int,
        default=None,
        help="Optional global step_number override applied to every run.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("COMET_API_KEY"),
        help="Comet API key. Defaults to the COMET_API_KEY environment variable.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Comet base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for Comet requests. Default: 60",
    )
    parser.add_argument(
        "--clean-run-dir",
        action="store_true",
        help="Deprecated. Run folders are cleaned before export by default.",
    )
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        help="Keep existing files in each run folder instead of replacing them.",
    )
    return parser.parse_args()


class CometRestClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 60) -> None:
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})

    def get_json(self, path: str, **params: Any) -> dict[str, Any]:
        url = f"{self.base_url}{API_ROOT}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise CometAPIError(
                f"GET {response.url} failed with status {response.status_code}: {response.text}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise CometAPIError(f"GET {response.url} did not return JSON") from exc

        if isinstance(payload, dict):
            message = payload.get("msg") or payload.get("message")
            if message and "error" in str(message).lower():
                raise CometAPIError(f"GET {response.url} returned an error payload: {payload}")

        return payload

    def download_asset(self, experiment_key: str, asset_id: str, destination: Path) -> dict[str, Any]:
        url = f"{self.base_url}{API_ROOT}/experiment/asset/get-asset"
        with self.session.get(
            url,
            params={"experimentKey": experiment_key, "assetId": asset_id},
            timeout=self.timeout,
            stream=True,
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise CometAPIError(
                    f"Asset download failed for {asset_id}: "
                    f"status {response.status_code}, body={response.text}"
                ) from exc

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type.lower():
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"body": response.text}
                raise CometAPIError(
                    f"Asset download for {asset_id} returned JSON instead of binary: {payload}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            sniff_bytes = bytearray()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        remaining = 64 - len(sniff_bytes)
                        if remaining > 0:
                            sniff_bytes.extend(chunk[:remaining])
                        handle.write(chunk)
        return {
            "content_type": content_type,
            "sniff_bytes": bytes(sniff_bytes),
        }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ManifestError(f"Manifest file does not exist: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {path}") from exc

    if isinstance(raw_data, list):
        runs = raw_data
    elif isinstance(raw_data, dict) and isinstance(raw_data.get("runs"), list):
        runs = raw_data["runs"]
    else:
        raise ManifestError("Manifest must be a JSON list or an object with a 'runs' list")

    if not runs:
        raise ManifestError("Manifest contains no runs")

    normalized_runs: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ManifestError(f"Run #{index} must be a JSON object")

        run_id = str(run.get("run_id", "")).strip()
        if not run_id:
            raise ManifestError(f"Run #{index} is missing 'run_id'")
        if run_id == PLACEHOLDER_RUN_ID:
            raise ManifestError(
                f"Run #{index} still uses the template run_id placeholder: {PLACEHOLDER_RUN_ID}"
            )

        run_name_raw = run.get("run_name")
        run_name = None if run_name_raw in (None, "") else str(run_name_raw).strip()

        step_raw = run.get("step_number")
        step_number = None
        if step_raw not in (None, ""):
            step_number = parse_required_int(step_raw, field_name=f"runs[{index}].step_number")

        normalized_runs.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "step_number": step_number,
            }
        )

    return normalized_runs


def parse_required_int(value: Any, field_name: str) -> int:
    parsed = parse_optional_int(value)
    if parsed is None:
        raise ManifestError(f"{field_name} must be an integer, got {value!r}")
    return parsed


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text or not INT_RE.match(text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return value

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if INT_RE.match(text):
        try:
            return int(text)
        except ValueError:
            return value

    try:
        return float(text)
    except ValueError:
        return value


def millis_to_iso(value: Any) -> str | None:
    millis = parse_optional_int(value)
    if millis is None:
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat()


def format_duration_millis(value: Any) -> str | None:
    millis = parse_optional_int(value)
    if millis is None:
        return None

    remaining = abs(millis)
    seconds, milliseconds = divmod(remaining, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    if milliseconds:
        parts.append(f"{milliseconds}ms")

    sign = "-" if millis < 0 else ""
    return sign + " ".join(parts)


def sanitize_folder_name(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]+', "_", name).strip()
    sanitized = sanitized.rstrip(".")
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized or "unnamed_run"


def sanitize_file_name(name: str) -> str:
    basename = Path(name).name
    sanitized = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]+', "_", basename).strip()
    sanitized = sanitized.rstrip(".")
    return sanitized or "asset.bin"


def normalize_export_image_stem(name: str) -> str:
    stem = Path(name).stem if Path(name).suffix else Path(name).name
    stem = stem.strip()

    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"\s*\(\d+\)$", "", stem).rstrip()
        stem = re.sub(r"__\d+$", "", stem).rstrip()

    stem = stem.strip(" ._-")
    sanitized = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]+', "_", stem).strip()
    return sanitized or "asset"


def normalized_export_image_file_name(name: str) -> str:
    return f"{normalize_export_image_stem(name)}.png"


def deterministic_output_path(directory: Path, file_name: str) -> Path:
    return directory / sanitize_file_name(file_name)


def make_unique_path(directory: Path, file_name: str) -> Path:
    candidate = directory / sanitize_file_name(file_name)
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        next_candidate = directory / f"{stem}_{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def reserve_run_folder(base_name: str, run_id: str, reserved: dict[str, str]) -> str:
    candidate = sanitize_folder_name(base_name)
    if candidate not in reserved or reserved[candidate] == run_id:
        reserved[candidate] = run_id
        return candidate

    suffix = run_id[:8]
    counter = 1
    while True:
        if counter == 1:
            next_candidate = f"{candidate}__{suffix}"
        else:
            next_candidate = f"{candidate}__{suffix}_{counter}"
        if next_candidate not in reserved or reserved[next_candidate] == run_id:
            reserved[next_candidate] = run_id
            return next_candidate
        counter += 1


def normalize_summary_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "value_current": parse_scalar(entry.get("valueCurrent")),
        "value_min": parse_scalar(entry.get("valueMin")),
        "value_max": parse_scalar(entry.get("valueMax")),
        "timestamp_current_millis": parse_optional_int(entry.get("timestampCurrent")),
        "timestamp_current_utc": millis_to_iso(entry.get("timestampCurrent")),
        "timestamp_min_millis": parse_optional_int(entry.get("timestampMin")),
        "timestamp_min_utc": millis_to_iso(entry.get("timestampMin")),
        "timestamp_max_millis": parse_optional_int(entry.get("timestampMax")),
        "timestamp_max_utc": millis_to_iso(entry.get("timestampMax")),
        "step_current": parse_optional_int(entry.get("stepCurrent")),
        "step_min": parse_optional_int(entry.get("stepMin")),
        "step_max": parse_optional_int(entry.get("stepMax")),
        "run_context_current": entry.get("runContextCurrent"),
        "run_context_min": entry.get("runContextMin"),
        "run_context_max": entry.get("runContextMax"),
    }


def extract_hyperparameters(parameter_entries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    hyperparameters: dict[str, Any] = {}
    hyperparameter_summary: dict[str, Any] = {}

    for entry in parameter_entries:
        name = entry.get("name")
        if not name:
            continue
        hyperparameters[name] = parse_scalar(entry.get("valueCurrent"))
        hyperparameter_summary[name] = normalize_summary_entry(entry)

    return hyperparameters, hyperparameter_summary


def normalize_metric_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": parse_scalar(entry.get("metricValue")),
        "timestamp_millis": parse_optional_int(entry.get("timestamp")),
        "timestamp_utc": millis_to_iso(entry.get("timestamp")),
        "step": parse_optional_int(entry.get("step")),
        "epoch": parse_optional_int(entry.get("epoch")),
        "run_context": entry.get("runContext"),
        "offset": parse_optional_int(entry.get("offset")),
    }


def maybe_clean_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def force_download_to_png(downloaded_path: Path) -> Path:
    normalized_file_name = normalized_export_image_file_name(downloaded_path.name)
    if downloaded_path.name == normalized_file_name:
        return downloaded_path
    target_path = deterministic_output_path(downloaded_path.parent, normalized_file_name)
    if target_path.exists():
        target_path.unlink()
    downloaded_path.rename(target_path)
    return target_path


def select_assets_for_step(
    assets: list[dict[str, Any]],
    requested_step: int,
) -> tuple[list[dict[str, Any]], int | None, list[int], bool]:
    assets_by_step: dict[int, list[dict[str, Any]]] = {}
    for asset in assets:
        step = parse_optional_int(asset.get("step"))
        if step is None:
            continue
        assets_by_step.setdefault(step, []).append(asset)

    available_steps = sorted(assets_by_step)
    if requested_step in assets_by_step:
        return assets_by_step[requested_step], requested_step, available_steps, False

    lower_steps = [step for step in available_steps if step < requested_step]
    if not lower_steps:
        return [], None, available_steps, False

    resolved_step = max(lower_steps)
    return assets_by_step[resolved_step], resolved_step, available_steps, True


def export_run(
    client: CometRestClient,
    run_config: dict[str, Any],
    output_dir: Path,
    step_override: int | None,
    reserved_folders: dict[str, str],
    clean_run_dir: bool,
) -> tuple[dict[str, Any], bool]:
    run_id = run_config["run_id"]
    manifest_step_number = run_config.get("step_number")
    requested_step = step_override if step_override is not None else manifest_step_number
    step_number_source = "global_override" if step_override is not None else "manifest"
    errors: list[str] = []
    warnings: list[str] = []

    if requested_step is None:
        return (
            {
                "id": run_id,
                "name": run_config.get("run_name") or run_id,
                "running_time": None,
                "hyperparameters": {},
                "hyperparameters_summary": {},
                "metrics": {},
                "metrics_summary": {},
                "manifest_step_number": manifest_step_number,
                "global_step_number_override": step_override,
                "step_number_source": step_number_source,
                "requested_step_number": None,
                "resolved_step_number": None,
                "step_selection": {
                    "manifest_step_number": manifest_step_number,
                    "global_step_number_override": step_override,
                    "step_number_source": step_number_source,
                    "requested_step_number": None,
                    "resolved_step_number": None,
                    "fallback_used": False,
                    "exact_match_found": False,
                    "available_image_steps": [],
                },
                "downloaded_images": [],
                "warnings": [],
                "errors": ["No step_number was provided for this run and no global override was set."],
            },
            True,
        )

    try:
        metadata = client.get_json("/experiment/metadata", experimentKey=run_id)
    except Exception as exc:
        return (
            {
                "id": run_id,
                "name": run_config.get("run_name") or run_id,
                "running_time": None,
                "hyperparameters": {},
                "hyperparameters_summary": {},
                "metrics": {},
                "metrics_summary": {},
                "manifest_step_number": manifest_step_number,
                "global_step_number_override": step_override,
                "step_number_source": step_number_source,
                "requested_step_number": requested_step,
                "resolved_step_number": None,
                "step_selection": {
                    "manifest_step_number": manifest_step_number,
                    "global_step_number_override": step_override,
                    "step_number_source": step_number_source,
                    "requested_step_number": requested_step,
                    "resolved_step_number": None,
                    "fallback_used": False,
                    "exact_match_found": False,
                    "available_image_steps": [],
                },
                "downloaded_images": [],
                "warnings": [],
                "errors": [f"Failed to fetch experiment metadata: {exc}"],
            },
            True,
        )

    original_name = metadata.get("experimentName")
    final_name = run_config.get("run_name") or original_name or run_id
    folder_name = reserve_run_folder(final_name, run_id, reserved_folders)
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if clean_run_dir:
        maybe_clean_directory(run_dir)

    LOGGER.info(
        "Exporting run %s into %s using step %s from %s",
        run_id,
        run_dir,
        requested_step,
        step_number_source,
    )

    try:
        parameter_payload = client.get_json("/experiment/parameters", experimentKey=run_id)
        parameter_entries = parameter_payload.get("values", [])
        hyperparameters, hyperparameter_summary = extract_hyperparameters(parameter_entries)
    except Exception as exc:
        hyperparameters = {}
        hyperparameter_summary = {}
        errors.append(f"Failed to fetch hyperparameters: {exc}")

    try:
        metric_summary_payload = client.get_json("/experiment/metrics/summary", experimentKey=run_id)
        metric_summary_entries = metric_summary_payload.get("values", [])
        metrics_summary = {
            entry["name"]: normalize_summary_entry(entry)
            for entry in metric_summary_entries
            if entry.get("name")
        }
    except Exception as exc:
        metrics_summary = {}
        metric_summary_entries = []
        errors.append(f"Failed to fetch metric summary: {exc}")

    metrics: dict[str, Any] = {}
    for entry in metric_summary_entries:
        metric_name = entry.get("name")
        if not metric_name:
            continue
        try:
            metric_payload = client.get_json(
                "/experiment/metrics/get-metric",
                experimentKey=run_id,
                metricName=metric_name,
            )
        except Exception as exc:
            errors.append(f"Failed to fetch metric history for {metric_name!r}: {exc}")
            continue
        metrics[metric_name] = [
            normalize_metric_entry(metric_entry)
            for metric_entry in metric_payload.get("metrics", [])
        ]

    downloaded_images_by_name: dict[str, dict[str, Any]] = {}
    try:
        asset_payload = client.get_json(
            "/experiment/asset/list",
            experimentKey=run_id,
            type="image",
        )
        assets = asset_payload.get("assets", [])
    except Exception as exc:
        assets = []
        errors.append(f"Failed to fetch image assets: {exc}")

    matching_assets, resolved_step, available_image_steps, fallback_used = select_assets_for_step(
        assets,
        requested_step,
    )

    if fallback_used and resolved_step is not None:
        warning_message = (
            f"Requested step {requested_step} was not found for run {run_id}; "
            f"using nearest lower available step {resolved_step}."
        )
        LOGGER.warning(warning_message)
        warnings.append(warning_message)
    elif resolved_step is None:
        if available_image_steps:
            warning_message = (
                f"Requested step {requested_step} was not found for run {run_id}, and there are "
                f"no image steps below it. Available image steps: {available_image_steps}."
            )
        else:
            warning_message = (
                f"No image assets with step information were found for run {run_id}; "
                f"nothing to download for requested step {requested_step}."
            )
        LOGGER.warning(warning_message)
        warnings.append(warning_message)

    for asset in matching_assets:
        asset_id = asset.get("assetId")
        if not asset_id:
            errors.append(f"Encountered image asset without assetId: {asset}")
            continue

        raw_file_name = asset.get("fileName") or asset_id
        file_name = normalized_export_image_file_name(str(raw_file_name))
        destination = deterministic_output_path(run_dir, file_name)

        try:
            download_info = client.download_asset(run_id, asset_id, destination)
        except Exception as exc:
            errors.append(f"Failed to download asset {asset_id}: {exc}")
            continue

        final_destination = force_download_to_png(destination)
        if final_destination.name in downloaded_images_by_name:
            warning_message = (
                f"Run {run_id}: replacing duplicate normalized image name {final_destination.name}."
            )
            LOGGER.warning(warning_message)
            warnings.append(warning_message)

        downloaded_images_by_name[final_destination.name] = {
            "asset_id": asset_id,
            "file_name": final_destination.name,
            "saved_path": str(final_destination.relative_to(output_dir)),
            "step": parse_optional_int(asset.get("step")),
            "run_context": asset.get("runContext"),
            "created_at_millis": parse_optional_int(asset.get("createdAt")),
            "created_at_utc": millis_to_iso(asset.get("createdAt")),
            "file_size": parse_optional_int(asset.get("fileSize")),
            "remote": bool(asset.get("remote", False)),
            "content_type": download_info.get("content_type"),
            "extension_source": "forced_png",
            "metadata": asset.get("metadata"),
            "link": asset.get("link"),
        }

    downloaded_images = list(downloaded_images_by_name.values())

    duration_millis = parse_optional_int(metadata.get("durationMillis"))
    running_time = {
        "duration_millis": duration_millis,
        "duration_seconds": None if duration_millis is None else duration_millis / 1000,
        "human_readable": format_duration_millis(duration_millis),
        "start_time_millis": parse_optional_int(metadata.get("startTimeMillis")),
        "start_time_utc": millis_to_iso(metadata.get("startTimeMillis")),
        "end_time_millis": parse_optional_int(metadata.get("endTimeMillis")),
        "end_time_utc": millis_to_iso(metadata.get("endTimeMillis")),
    }

    result = {
        "id": run_id,
        "name": final_name,
        "original_name": original_name,
        "workspace_name": metadata.get("workspaceName"),
        "project_name": metadata.get("projectName"),
        "running_time": running_time,
        "manifest_step_number": manifest_step_number,
        "global_step_number_override": step_override,
        "step_number_source": step_number_source,
        "requested_step_number": requested_step,
        "resolved_step_number": resolved_step,
        "step_selection": {
            "manifest_step_number": manifest_step_number,
            "global_step_number_override": step_override,
            "step_number_source": step_number_source,
            "requested_step_number": requested_step,
            "resolved_step_number": resolved_step,
            "fallback_used": fallback_used,
            "exact_match_found": resolved_step == requested_step if resolved_step is not None else False,
            "available_image_steps": available_image_steps,
        },
        "output_folder": folder_name,
        "hyperparameters": hyperparameters,
        "hyperparameters_summary": hyperparameter_summary,
        "metrics_summary": metrics_summary,
        "metrics": metrics,
        "downloaded_images": downloaded_images,
        "warnings": warnings,
        "errors": errors,
    }
    return result, bool(errors)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.api_key:
        print(
            "Missing Comet API key. Pass --api-key or run the script as "
            "'COMET_API_KEY=... python diffusion_template/tools/comet/export_comet_runs.py'.",
            file=sys.stderr,
        )
        return 2

    try:
        runs = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = (args.output_json or (output_dir / DEFAULT_OUTPUT_JSON_NAME)).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    client = CometRestClient(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    reserved_folders: dict[str, str] = {}
    exported_runs: list[dict[str, Any]] = []
    had_errors = False

    for run in runs:
        exported_run, run_failed = export_run(
            client=client,
            run_config=run,
            output_dir=output_dir,
            step_override=args.step_number,
            reserved_folders=reserved_folders,
            clean_run_dir=not args.keep_run_dir,
        )
        exported_runs.append(exported_run)
        had_errors = had_errors or run_failed

    export_payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "manifest_path": str(args.manifest.resolve()),
        "output_dir": str(output_dir),
        "output_json": str(output_json),
        "base_url": args.base_url,
        "global_step_number_override": args.step_number,
        "runs": exported_runs,
    }
    output_json.write_text(
        json.dumps(export_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Wrote combined export JSON to %s", output_json)
    if had_errors:
        LOGGER.warning("Completed with errors. Inspect the 'errors' field in %s", output_json)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
