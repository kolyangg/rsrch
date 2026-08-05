#!/usr/bin/env python3
"""Validate and backfill saved per-image ID-sim tables to exact Comet runs.

The tool is read-only unless ``--write`` is supplied. It uses Comet's REST-backed
``APIExperiment`` rather than resuming an experiment session, so it does not end,
pause, rename, or otherwise change the lifecycle of an ongoing training run.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from comet_ml import API

from comet_experiment import load_env_file, load_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
TABLE_RE = re.compile(r"^id_sim__manual_val__step_(\d{6})\.csv$")
EXPECTED_COLUMNS = [
    "validation_step",
    "partition",
    "image_index",
    "output_key",
    "identity",
    "prompt",
    "seed",
    "generated_image_count",
    "id_sim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach locally saved per-image ID-sim CSVs to their immutable "
            "Comet experiment keys."
        )
    )
    parser.add_argument(
        "--tables-root",
        type=Path,
        required=True,
        help=(
            "Directory containing <run>/comet_experiment.json and "
            "<run>/validation_tables/*.csv."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory to process; repeat as needed. Default: every run under root.",
    )
    parser.add_argument("--expected-project", default="aug-large-ds")
    parser.add_argument("--expected-rows", type=int, default=96)
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--verify-attempts", type=int, default=12)
    parser.add_argument("--verify-delay", type=float, default=5.0)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Perform uploads. Without this flag the command is a read-only dry run.",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_table(path: Path, expected_rows: int) -> tuple[int, str]:
    match = TABLE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected table filename: {path}")
    step = int(match.group(1))
    table = pd.read_csv(path)
    if list(table.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column mismatch in {path}: expected={EXPECTED_COLUMNS}, "
            f"actual={list(table.columns)}"
        )
    if len(table) != expected_rows:
        raise ValueError(
            f"Row mismatch in {path}: expected={expected_rows}, actual={len(table)}"
        )
    if table["image_index"].tolist() != list(range(expected_rows)):
        raise ValueError(f"Image indices are not exactly 0..{expected_rows - 1}: {path}")
    if not table["validation_step"].eq(step).all():
        raise ValueError(f"validation_step does not match filename: {path}")
    if not table["partition"].eq("manual_val").all():
        raise ValueError(f"partition is not uniformly manual_val: {path}")
    if not table["generated_image_count"].eq(1).all():
        raise ValueError(f"generated_image_count is not uniformly one: {path}")
    if table["id_sim"].isna().any() or not table["id_sim"].map(math.isfinite).all():
        raise ValueError(f"id_sim contains a missing or non-finite value: {path}")
    return step, sha256_file(path)


def assets_by_filename(experiment: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for asset in experiment.get_asset_list():
        filename = str(asset.get("fileName") or "")
        result.setdefault(filename, []).append(asset)
    return result


def verify_exact_asset(
    experiment: Any,
    filename: str,
    expected_sha256: str,
    assets: list[dict[str, Any]],
) -> bool:
    if len(assets) > 1:
        raise ValueError(f"Comet contains duplicate assets named {filename}")
    if not assets:
        return False
    asset_type = str(assets[0].get("type") or "")
    if asset_type != "dataframe":
        raise ValueError(
            f"Existing Comet asset is not a dataframe table {filename}: "
            f"type={asset_type!r}"
        )
    payload = experiment.get_asset(str(assets[0]["assetId"]))
    if not isinstance(payload, bytes):
        raise TypeError(f"Comet returned non-binary content for {filename}")
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Existing Comet asset conflicts with local CSV {filename}: "
            f"expected_sha256={expected_sha256}, actual_sha256={actual_sha256}"
        )
    return True


def selected_runs(root: Path, requested: list[str]) -> list[Path]:
    if requested:
        if len(requested) != len(set(requested)):
            raise ValueError("--run values must be unique")
        paths = [root / run_name for run_name in requested]
    else:
        paths = sorted(path.parent for path in root.glob("*/comet_experiment.json"))
    if not paths:
        raise ValueError(f"No run records found under {root}")
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(path)
    return paths


def main() -> int:
    args = parse_args()
    if args.expected_rows < 1:
        raise ValueError("--expected-rows must be positive")
    if args.verify_attempts < 1:
        raise ValueError("--verify-attempts must be positive")
    if args.verify_delay < 0:
        raise ValueError("--verify-delay cannot be negative")

    environment = dict(os.environ)
    if not args.api_key:
        load_env_file(args.env_file.resolve(), environment)
    api_key = args.api_key or environment.get("COMET_API_KEY")
    if not api_key:
        raise ValueError("COMET_API_KEY is required")

    root = args.tables_root.resolve()
    runs = selected_runs(root, args.run)
    api = API(api_key=api_key, cache=False)
    uploaded = 0
    already_present = 0
    planned = 0

    for run_dir in runs:
        run_name = run_dir.name
        record = load_record(
            run_dir / "comet_experiment.json",
            expected_run_name=run_name,
        )
        comet = record["comet"]
        if comet["project_name"] != args.expected_project:
            raise ValueError(
                f"Record project mismatch for {run_name}: "
                f"{comet['project_name']!r} != {args.expected_project!r}"
            )
        experiment_key = str(comet["experiment_key"])
        experiment = api.get_experiment_by_key(experiment_key)
        if experiment is None:
            raise ValueError(f"Comet experiment does not exist: {experiment_key}")
        metadata = experiment.get_metadata()
        actual_project = str(metadata.get("projectName") or "")
        if actual_project != args.expected_project:
            raise ValueError(
                f"Live Comet project mismatch for {run_name}: "
                f"{actual_project!r} != {args.expected_project!r}"
            )
        actual_name = str(experiment.get_name() or "")
        if actual_name != run_name:
            raise ValueError(
                f"Live Comet name mismatch for key {experiment_key}: "
                f"{actual_name!r} != {run_name!r}"
            )

        paths = sorted((run_dir / "validation_tables").glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No validation tables found for {run_name}")
        validated = [
            (path, *validate_table(path, args.expected_rows)) for path in paths
        ]
        asset_index = assets_by_filename(experiment)
        run_uploaded = 0
        run_present = 0
        run_planned = 0
        queued: list[tuple[Path, int, str]] = []
        for path, step, csv_sha256 in validated:
            if verify_exact_asset(
                experiment,
                path.name,
                csv_sha256,
                asset_index.get(path.name, []),
            ):
                run_present += 1
                already_present += 1
                continue
            if not args.write:
                run_planned += 1
                planned += 1
                continue

            # AICODE-NOTE: APIExperiment writes directly through Comet's API;
            # it does not resume/end the active training experiment lifecycle.
            result = experiment.log_asset(
                str(path),
                step=step,
                name=path.name,
                overwrite=False,
                ftype="dataframe",
                metadata={
                    "sha256": csv_sha256,
                    "row_count": args.expected_rows,
                    "validation_step": step,
                    "partition": "manual_val",
                    "source": "saved_validation_table_local_backfill",
                },
            )
            if result is None:
                raise RuntimeError(f"Comet returned no upload result for {path.name}")
            queued.append((path, step, csv_sha256))

        if queued:
            unverified = {path.name: csv_sha256 for path, _, csv_sha256 in queued}
            for attempt in range(args.verify_attempts):
                asset_index = assets_by_filename(experiment)
                unverified = {
                    filename: csv_sha256
                    for filename, csv_sha256 in unverified.items()
                    if not verify_exact_asset(
                        experiment,
                        filename,
                        csv_sha256,
                        asset_index.get(filename, []),
                    )
                }
                if not unverified:
                    break
                if attempt + 1 < args.verify_attempts:
                    time.sleep(args.verify_delay)
            if unverified:
                raise RuntimeError(
                    "Comet uploads were not visible after retries: "
                    f"{sorted(unverified)}"
                )
            run_uploaded += len(queued)
            uploaded += len(queued)

        print(
            "ID_SIM_TABLE_RUN "
            f"run={run_name} key={experiment_key} tables={len(validated)} "
            f"uploaded={run_uploaded} existing={run_present} planned={run_planned}"
        )

    print(
        "ID_SIM_TABLE_SUMMARY "
        f"runs={len(runs)} uploaded={uploaded} existing={already_present} "
        f"planned={planned} write={args.write}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
