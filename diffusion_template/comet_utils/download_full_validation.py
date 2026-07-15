#!/usr/bin/env python3
"""Download multi-step full-validation images and metrics from Comet.

The output mirrors the project's local validation layout:

    full_validation_results/<run_name>/<run_name>_step2000/*.png
    full_validation_results/<run_name>/metrics_<run_name>_steps.json
    full_validation_results/<run_name>/comet_export.json

Comet replaces spaces in logged image names with underscores. The downloader
uses the canonical 96-image bbox JSON to restore the original filenames.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from comet_utils.export_comet_runs import (  # noqa: E402
    CometRestClient,
    extract_hyperparameters,
    millis_to_iso,
    normalize_export_image_stem,
    normalize_metric_entry,
    normalize_summary_entry,
    parse_optional_int,
    sanitize_folder_name,
)


DEFAULT_CONFIG = SCRIPT_DIR / "comet_full_validation_download_template.json"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "full_validation_results"
DEFAULT_REFS_DIR = PROJECT_DIR.parent / "dataset_full" / "val_dataset" / "references"
DEFAULT_NAMES_JSON = PROJECT_DIR.parent / "dataset_full" / "val_dataset" / "pm96_bboxes_new.json"
DEFAULT_STEPS = [2000, 6000, 10000]
DEFAULT_BASE_URL = "https://www.comet.com"
PLACEHOLDER_RUN_ID = "REPLACE_WITH_COMET_EXPERIMENT_KEY"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

LOGGER = logging.getLogger("comet_full_validation")


class ConfigError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download exact Comet validation steps and compute missing full-validation metrics."
    )
    parser.add_argument("--config", type=Path, required=True, help=f"JSON config; template: {DEFAULT_CONFIG}")
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--skip-local-metrics", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (useful for machine-readable logs).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional step override for every run; otherwise use each run/config default.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON: {path}: {exc}") from exc


def resolve_path(config_dir: Path, value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    path = Path(str(value)).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def parse_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be true or false")
    return value


def parse_steps(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field_name} must be a non-empty list of positive integers")
    steps: list[int] = []
    for raw in value:
        if isinstance(raw, bool):
            raise ConfigError(f"{field_name} contains invalid step {raw!r}")
        try:
            step = int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{field_name} contains invalid step {raw!r}") from exc
        if step <= 0:
            raise ConfigError(f"{field_name} contains non-positive step {step}")
        if step not in steps:
            steps.append(step)
    return steps


def load_config(path: Path, cli_steps: list[int] | None) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a JSON object")
    config_dir = path.resolve().parent
    default_steps = parse_steps(raw.get("default_steps", DEFAULT_STEPS), "default_steps")
    if cli_steps is not None:
        default_steps = parse_steps(cli_steps, "--steps")

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ConfigError("Config must contain a non-empty 'runs' list")

    runs = []
    seen_ids = set()
    for index, run in enumerate(runs_raw):
        if not isinstance(run, dict):
            raise ConfigError(f"runs[{index}] must be an object")
        run_id = str(run.get("run_id", "")).strip()
        if not run_id or run_id == PLACEHOLDER_RUN_ID:
            raise ConfigError(f"runs[{index}].run_id is missing or still a placeholder")
        if run_id in seen_ids:
            raise ConfigError(f"Duplicate run_id: {run_id}")
        seen_ids.add(run_id)
        run_name = run.get("run_name")
        run_steps = default_steps if cli_steps is not None else run.get("steps", default_steps)
        runs.append(
            {
                "run_id": run_id,
                "run_name": None if run_name in (None, "") else str(run_name).strip(),
                "steps": parse_steps(run_steps, f"runs[{index}].steps"),
                "epoch_len": parse_optional_int(run.get("epoch_len")),
                "compute_metrics": parse_bool(
                    run.get("compute_metrics"),
                    f"runs[{index}].compute_metrics",
                    parse_bool(raw.get("compute_metrics"), "compute_metrics", True),
                ),
            }
        )

    expected_images = raw.get("expected_images", 96)
    if isinstance(expected_images, bool):
        raise ConfigError("expected_images must be a positive integer")
    try:
        expected_images = int(expected_images)
    except (TypeError, ValueError) as exc:
        raise ConfigError("expected_images must be a positive integer") from exc
    if expected_images <= 0:
        raise ConfigError("expected_images must be a positive integer")

    return {
        "path": path.resolve(),
        "output_root": resolve_path(config_dir, raw.get("output_root"), DEFAULT_OUTPUT_ROOT),
        "refs_dir": resolve_path(config_dir, raw.get("refs_dir"), DEFAULT_REFS_DIR),
        "expected_names_json": resolve_path(
            config_dir, raw.get("expected_names_json"), DEFAULT_NAMES_JSON
        ),
        "expected_images": expected_images,
        "strict_steps": parse_bool(raw.get("strict_steps"), "strict_steps", True),
        "clean_step_dirs": parse_bool(raw.get("clean_step_dirs"), "clean_step_dirs", False),
        "runs": runs,
    }


def load_expected_names(path: Path, expected_images: int) -> list[str]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"Expected-name JSON must be an object keyed by filename: {path}")
    names = [str(name) for name in raw.keys() if Path(str(name)).suffix.lower() in IMAGE_SUFFIXES]
    if len(names) != expected_images:
        raise ConfigError(
            f"Expected-name JSON has {len(names)} image names, expected {expected_images}: {path}"
        )
    return names


def logged_name_key(file_name: str) -> str:
    """Normalize a canonical or Comet-mutated validation image name for matching."""
    stem = Path(file_name).stem.replace(" ", "_")[:80]
    return normalize_export_image_stem(stem)


def expected_name_lookup(names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in names:
        key = logged_name_key(name)
        if key in lookup and lookup[key] != name:
            raise ConfigError(f"Canonical image names collide after Comet normalization: {name!r}")
        lookup[key] = name
    return lookup


def fetch_run_data(
    client: CometRestClient, run_id: str, *, show_progress: bool = False
) -> dict[str, Any]:
    metadata = client.get_json("/experiment/metadata", experimentKey=run_id)
    fetch_warnings = []

    try:
        parameter_payload = client.get_json("/experiment/parameters", experimentKey=run_id)
        parameter_entries = parameter_payload.get("values", [])
        hyperparameters, hyperparameters_summary = extract_hyperparameters(parameter_entries)
    except Exception as exc:
        hyperparameters = {}
        hyperparameters_summary = {}
        fetch_warnings.append(f"Failed to fetch parameters: {exc}")

    try:
        metric_summary_payload = client.get_json("/experiment/metrics/summary", experimentKey=run_id)
        metric_summary_entries = metric_summary_payload.get("values", [])
        metrics_summary = {
            entry["name"]: normalize_summary_entry(entry)
            for entry in metric_summary_entries
            if entry.get("name")
        }
    except Exception as exc:
        metric_summary_entries = []
        metrics_summary = {}
        fetch_warnings.append(f"Failed to fetch metric summary: {exc}")
    metrics: dict[str, list[dict[str, Any]]] = {}
    metric_errors = []
    metric_progress = tqdm(
        metric_summary_entries,
        desc=f"{run_id[:8]}: Comet metric histories",
        unit="metric",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for entry in metric_progress:
        name = entry.get("name")
        if not name:
            continue
        try:
            payload = client.get_json(
                "/experiment/metrics/get-metric",
                experimentKey=run_id,
                metricName=name,
            )
            metrics[name] = [normalize_metric_entry(item) for item in payload.get("metrics", [])]
        except Exception as exc:
            metric_errors.append(f"Failed to fetch metric {name!r}: {exc}")

    asset_payload = client.get_json("/experiment/asset/list", experimentKey=run_id, type="image")
    assets = asset_payload.get("assets", [])
    available_steps = sorted(
        {step for asset in assets if (step := parse_optional_int(asset.get("step"))) is not None}
    )
    return {
        "metadata": metadata,
        "hyperparameters": hyperparameters,
        "hyperparameters_summary": hyperparameters_summary,
        "metrics_summary": metrics_summary,
        "metrics": metrics,
        "metric_errors": metric_errors,
        "fetch_warnings": fetch_warnings,
        "assets": assets,
        "available_image_steps": available_steps,
    }


def metrics_at_steps(
    metrics: dict[str, list[dict[str, Any]]], steps: list[int]
) -> dict[str, dict[str, Any]]:
    result = {str(step): {} for step in steps}
    for metric_name, history in metrics.items():
        by_step: dict[int, dict[str, Any]] = {}
        for point in history:
            point_step = parse_optional_int(point.get("step"))
            if point_step in steps:
                previous = by_step.get(point_step)
                if previous is None or (
                    parse_optional_int(point.get("offset")) or -1
                ) >= (parse_optional_int(previous.get("offset")) or -1):
                    by_step[point_step] = point
        for step, point in by_step.items():
            result[str(step)][metric_name] = point
    return result


def select_step_assets(
    assets: list[dict[str, Any]],
    requested_step: int,
    available_steps: list[int],
    strict_steps: bool,
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    exact = [asset for asset in assets if parse_optional_int(asset.get("step")) == requested_step]
    if exact:
        return exact, requested_step, None
    if strict_steps:
        return [], None, f"Step {requested_step} is absent; available image steps: {available_steps}"
    lower = [step for step in available_steps if step < requested_step]
    if not lower:
        return [], None, f"No image step at or below {requested_step}; available: {available_steps}"
    resolved = max(lower)
    fallback = [asset for asset in assets if parse_optional_int(asset.get("step")) == resolved]
    return fallback, resolved, f"Step {requested_step} absent; using lower step {resolved}"


def canonicalize_assets(
    assets: list[dict[str, Any]], lookup: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Map Comet assets to canonical names, retaining the newest duplicate."""
    selected: dict[str, dict[str, Any]] = {}
    warnings = []
    for asset in assets:
        raw_name = str(asset.get("fileName") or asset.get("assetId") or "")
        key = logged_name_key(raw_name)
        if key.endswith("_mask"):
            continue
        canonical = lookup.get(key)
        if canonical is None:
            warnings.append(f"Ignoring unrecognized image asset: {raw_name}")
            continue
        previous = selected.get(canonical)
        if previous is None:
            selected[canonical] = asset
            continue
        previous_created = parse_optional_int(previous.get("createdAt")) or -1
        current_created = parse_optional_int(asset.get("createdAt")) or -1
        if current_created >= previous_created:
            selected[canonical] = asset
        warnings.append(f"Deduplicated repeated asset for {canonical}")
    return selected, warnings


def valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_image_asset(
    client: CometRestClient,
    run_id: str,
    asset: dict[str, Any],
    destination: Path,
    force: bool,
) -> str:
    if not force and valid_image(destination):
        return "existing"
    asset_id = asset.get("assetId")
    if not asset_id:
        raise RuntimeError(f"Image asset is missing assetId: {asset}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{asset_id}.part")
    temporary.unlink(missing_ok=True)
    try:
        client.download_asset(run_id, str(asset_id), temporary)
        with Image.open(temporary) as image:
            image.load()
            if image.format == "PNG":
                temporary.replace(destination)
            else:
                image.save(destination, format="PNG")
                temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def list_step_images(step_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in step_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith("_")
        and not path.stem.endswith("_mask")
    )


def read_metrics_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def epoch_for_step(step: int, run_config: dict[str, Any], hyperparameters: dict[str, Any]) -> int | None:
    epoch_len = run_config.get("epoch_len") or parse_optional_int(hyperparameters.get("trainer.epoch_len"))
    if not epoch_len or step % epoch_len != 0:
        return None
    return step // epoch_len


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if not args.api_key:
        print("Missing COMET_API_KEY; export it or pass --api-key.", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config, args.steps)
        expected_names = load_expected_names(
            config["expected_names_json"], config["expected_images"]
        )
        name_lookup = expected_name_lookup(expected_names)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    output_root: Path = config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    client = CometRestClient(args.api_key, args.base_url, timeout=args.timeout)
    generated_at = datetime.now(timezone.utc).isoformat()
    combined = {
        "generated_at_utc": generated_at,
        "config_path": str(config["path"]),
        "output_root": str(output_root),
        "runs": [],
    }
    combined_path = output_root / "comet_full_validation_export.json"
    had_errors = False
    reserved_names: dict[str, str] = {}
    metric_analyzer = None
    reference_embeddings = None
    show_progress = not args.no_progress

    run_progress = tqdm(
        config["runs"],
        desc="Runs",
        unit="run",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for run_config in run_progress:
        run_id = run_config["run_id"]
        configured_name = run_config.get("run_name") or run_id[:8]
        run_progress.set_postfix_str(configured_name)
        tqdm.write(f"[Comet] Fetching run metadata, metrics, and image index: {configured_name}")
        run_errors: list[str] = []
        run_warnings: list[str] = []
        try:
            fetched = fetch_run_data(client, run_id, show_progress=show_progress)
        except Exception as exc:
            combined["runs"].append({"run_id": run_id, "errors": [str(exc)]})
            had_errors = True
            write_json(combined_path, combined)
            continue

        metadata = fetched["metadata"]
        comet_name = str(metadata.get("experimentName") or run_id)
        logical_name = run_config.get("run_name") or comet_name
        folder_name = sanitize_folder_name(logical_name)
        if folder_name in reserved_names and reserved_names[folder_name] != run_id:
            folder_name = f"{folder_name}__{run_id[:8]}"
        reserved_names[folder_name] = run_id
        run_dir = output_root / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / f"metrics_{folder_name}_steps.json"
        run_progress.set_postfix_str(folder_name)
        tqdm.write(
            f"[Comet] {folder_name}: found {len(fetched['assets'])} image assets "
            f"across steps {fetched['available_image_steps']}"
        )

        run_result: dict[str, Any] = {
            "run_id": run_id,
            "run_name": logical_name,
            "comet_run_name": comet_name,
            "output_folder": str(run_dir.relative_to(output_root)),
            "workspace_name": metadata.get("workspaceName"),
            "project_name": metadata.get("projectName"),
            "start_time_utc": millis_to_iso(metadata.get("startTimeMillis")),
            "end_time_utc": millis_to_iso(metadata.get("endTimeMillis")),
            "requested_steps": run_config["steps"],
            "available_image_steps": fetched["available_image_steps"],
            "hyperparameters": fetched["hyperparameters"],
            "hyperparameters_summary": fetched["hyperparameters_summary"],
            "comet_metrics_summary": fetched["metrics_summary"],
            "comet_metrics": fetched["metrics"],
            "comet_metrics_at_steps": metrics_at_steps(
                fetched["metrics"], run_config["steps"]
            ),
            "steps": [],
            "warnings": run_warnings,
            "errors": run_errors,
        }
        run_warnings.extend(fetched["metric_errors"])
        run_warnings.extend(fetched["fetch_warnings"])

        step_progress = tqdm(
            run_config["steps"],
            desc=f"{folder_name}: steps",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for requested_step in step_progress:
            step_progress.set_postfix_str(str(requested_step))
            assets, resolved_step, selection_message = select_step_assets(
                fetched["assets"],
                requested_step,
                fetched["available_image_steps"],
                config["strict_steps"],
            )
            step_result: dict[str, Any] = {
                "requested_step": requested_step,
                "resolved_step": resolved_step,
                "output_folder": None,
                "n_assets_at_step": len(assets),
                "n_images": 0,
                "downloaded": 0,
                "reused": 0,
                "metric_record": None,
                "warnings": [],
                "errors": [],
            }
            run_result["steps"].append(step_result)
            if selection_message:
                target = step_result["errors"] if resolved_step is None else step_result["warnings"]
                target.append(selection_message)
            if resolved_step is None:
                had_errors = True
                continue

            output_step = int(resolved_step)
            run_key = f"{folder_name}_step{output_step}"
            step_dir = run_dir / run_key
            if config["clean_step_dirs"] and step_dir.exists():
                shutil.rmtree(step_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            step_result["output_folder"] = str(step_dir.relative_to(output_root))

            canonical_assets, asset_warnings = canonicalize_assets(assets, name_lookup)
            step_result["warnings"].extend(asset_warnings)
            missing_remote = sorted(set(expected_names) - set(canonical_assets))
            if missing_remote:
                step_result["errors"].append(
                    f"Comet step has {len(canonical_assets)}/{config['expected_images']} recognized images; "
                    f"missing examples: {missing_remote[:8]}"
                )

            asset_records = []
            asset_progress = tqdm(
                sorted(canonical_assets.items()),
                desc=f"{folder_name} step {requested_step}: images",
                unit="image",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            for canonical_name, asset in asset_progress:
                destination = step_dir / canonical_name
                try:
                    status = download_image_asset(
                        client, run_id, asset, destination, args.force_download
                    )
                    step_result["downloaded" if status == "downloaded" else "reused"] += 1
                    asset_records.append(
                        {
                            "asset_id": asset.get("assetId"),
                            "comet_file_name": asset.get("fileName"),
                            "file_name": canonical_name,
                            "created_at_utc": millis_to_iso(asset.get("createdAt")),
                            "status": status,
                        }
                    )
                except Exception as exc:
                    step_result["errors"].append(f"Failed {canonical_name}: {exc}")
                asset_progress.set_postfix(
                    downloaded=step_result["downloaded"],
                    reused=step_result["reused"],
                    errors=len(step_result["errors"]),
                    refresh=False,
                )

            local_images = list_step_images(step_dir)
            step_result["n_images"] = len(local_images)
            step_result["assets"] = asset_records
            actual_names = {path.name for path in local_images}
            expected_set = set(expected_names)
            if actual_names != expected_set:
                missing = sorted(expected_set - actual_names)
                extra = sorted(actual_names - expected_set)
                step_result["errors"].append(
                    f"Local folder is not the exact {config['expected_images']}-image set; "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )

            should_compute = run_config["compute_metrics"] and not args.skip_local_metrics
            if should_compute and actual_names == expected_set:
                existing_metrics = read_metrics_json(metrics_path).get(run_key)
                complete_existing = bool(
                    isinstance(existing_metrics, dict)
                    and existing_metrics.get("n_images") == config["expected_images"]
                    and len(existing_metrics.get("per_image_id_sim", {})) == config["expected_images"]
                )
                if complete_existing and not args.force_metrics:
                    step_result["metric_record"] = existing_metrics
                    step_result["metric_status"] = "existing"
                else:
                    try:
                        from scripts.full_val_metrics import (  # noqa: WPS433
                            compute_full_val_metrics,
                            create_face_analyzer,
                            load_reference_embeddings,
                            update_metrics_json,
                        )

                        if metric_analyzer is None:
                            tqdm.write("[Metrics] Initializing InsightFace on CPU (one time)")
                            metric_analyzer = create_face_analyzer(
                                providers=["CPUExecutionProvider"],
                                allowed_modules=["detection", "recognition"],
                                ctx_id=-1,
                                det_size=(640, 640),
                                fallback_ctx_id=-1,
                                quiet=True,
                            )
                            reference_embeddings = load_reference_embeddings(
                                metric_analyzer,
                                config["refs_dir"],
                                show_progress=show_progress,
                            )
                        tqdm.write(
                            f"[Metrics] Computing identity scores: {folder_name} "
                            f"step {output_step}"
                        )
                        record = compute_full_val_metrics(
                            metric_analyzer,
                            step_dir,
                            config["refs_dir"],
                            epoch=epoch_for_step(
                                output_step, run_config, fetched["hyperparameters"]
                            ),
                            step=output_step,
                            checkpoint=f"comet:{run_id}@step{resolved_step}",
                            reference_embeddings=reference_embeddings,
                            show_progress=show_progress,
                            progress_desc=f"{folder_name} step {output_step}: metrics",
                        )
                        update_metrics_json(metrics_path, run_key, record)
                        step_result["metric_record"] = record
                        step_result["metric_status"] = "computed"
                    except Exception as exc:
                        step_result["errors"].append(f"Metric computation failed: {exc}")

            if step_result["errors"]:
                had_errors = True
            LOGGER.info(
                "%s step %s: images=%s downloaded=%s reused=%s metric=%s",
                folder_name,
                requested_step,
                step_result["n_images"],
                step_result["downloaded"],
                step_result["reused"],
                step_result.get("metric_status", "skipped"),
            )

        write_json(run_dir / "comet_export.json", run_result)
        combined["runs"].append(
            {
                "run_id": run_id,
                "run_name": logical_name,
                "output_folder": run_result["output_folder"],
                "steps": run_result["steps"],
                "warnings": run_warnings,
                "errors": run_errors,
            }
        )
        write_json(combined_path, combined)

    LOGGER.info("Combined export index: %s", combined_path)
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
