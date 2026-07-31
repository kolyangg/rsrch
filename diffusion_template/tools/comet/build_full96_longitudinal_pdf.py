#!/usr/bin/env python3
"""Build a fail-closed longitudinal PDF from verified Comet image manifests.

Each report page compares the same validation samples across runs at one
optimizer step. The builder consumes:

* ``download_manifest.json`` from ``download_face_quality_images.py``;
* the per-image JSON from ``calc_metrics.py --manifest --id-only``; and
* the per-image CSV from ``calculate_face_quality_metrics.py``.

No Comet access occurs here. Downloading, image-count verification, and metric
calculation remain explicit upstream stages so the final PDF is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image

LANDSCAPE_A4 = (11.69, 8.27)
DEFAULT_METRICS = [
    {"key": "id_sim", "label": "ID", "format": ".3f"},
    {"key": "face_detected", "label": "FD", "format": ".0f"},
    {"key": "topiq_nr_face", "label": "TQ-F", "format": ".3f"},
    {"key": "topiq_nr", "label": "TQ", "format": ".3f"},
    {"key": "musiq", "label": "MUSIQ", "format": ".1f"},
    {"key": "maniqa_pipal", "label": "MANIQA", "format": ".3f"},
]


class ReportInputError(ValueError):
    """Raised when report inputs are incomplete or cannot be joined exactly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PDF comparing exact validation images and per-image metrics "
            "across runs and steps."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: Any, field: str) -> Path:
    if raw_path in (None, ""):
        raise ReportInputError(f"Missing required path: {field}")
    path = Path(str(raw_path)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def canonical_sample_key(file_name: str) -> str:
    stem = Path(file_name).stem if Path(file_name).suffix else Path(file_name).name
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"\s*\(\d+\)$", "", stem).rstrip()
        stem = re.sub(r"__\d+$", "", stem).rstrip()
    return stem.strip(" ._-")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportInputError(f"Missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise ReportInputError(f"Invalid JSON file: {path}") from error


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    raise ReportInputError(f"Metric file must contain a row list: {path}")


def parse_step(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"Invalid step {value!r} in {context}") from error


def metric_index(
    rows: list[dict[str, Any]],
    source_path: Path,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    by_asset: dict[tuple[int, str], dict[str, Any]] = {}
    by_sample: dict[tuple[int, str], dict[str, Any]] = {}
    for row_index, row in enumerate(rows, start=2):
        step = parse_step(row.get("step"), f"{source_path}:{row_index}")
        asset_id = str(row.get("asset_id") or "").strip()
        sample_key = str(row.get("sample_key") or "").strip()
        if not sample_key:
            sample_key = canonical_sample_key(
                str(row.get("file_name") or row.get("file") or "")
            )
        if not sample_key:
            raise ReportInputError(
                f"Metric row has no asset ID or sample name: {source_path}:{row_index}"
            )
        if asset_id:
            key = (step, asset_id)
            if key in by_asset:
                raise ReportInputError(f"Duplicate metric asset key {key} in {source_path}")
            by_asset[key] = row
        sample_lookup = (step, sample_key)
        if sample_lookup in by_sample:
            raise ReportInputError(
                f"Duplicate metric sample key {sample_lookup} in {source_path}"
            )
        by_sample[sample_lookup] = row
    return by_asset, by_sample


def merge_metric_sources(
    assets: list[dict[str, Any]],
    step: int,
    source_paths: list[Path],
) -> None:
    for source_path in source_paths:
        by_asset, by_sample = metric_index(load_metric_rows(source_path), source_path)
        for asset in assets:
            row = by_asset.get((step, asset["asset_id"]))
            if row is None:
                row = by_sample.get((step, asset["sample_key"]))
            if row is None:
                raise ReportInputError(
                    f"No per-image metric row for step={step}, "
                    f"asset={asset['asset_id']}, sample={asset['sample_key']!r} "
                    f"in {source_path}"
                )
            for key, value in row.items():
                if key not in {"step", "asset_id", "file", "file_name", "local_path"}:
                    asset["metrics"][key] = value


def load_run(
    run_cfg: dict[str, Any],
    base_dir: Path,
    requested_steps: list[int],
    expected_images: int,
) -> dict[str, Any]:
    name = str(run_cfg.get("name") or "").strip()
    if not name:
        raise ReportInputError("Each run requires a non-empty name")
    manifest_path = resolve_path(base_dir, run_cfg.get("download_manifest"), f"{name}.download_manifest")
    manifest = load_json(manifest_path)
    if str(manifest.get("experiment_key") or "") != str(run_cfg.get("validation_key") or ""):
        raise ReportInputError(
            f"{name}: manifest experiment key does not match validation_key"
        )
    manifest_steps = manifest.get("steps")
    if not isinstance(manifest_steps, dict):
        raise ReportInputError(f"{name}: manifest has no steps mapping")

    metric_paths = [
        resolve_path(base_dir, raw_path, f"{name}.per_image_metrics")
        for raw_path in run_cfg.get("per_image_metrics", [])
    ]
    if not metric_paths:
        raise ReportInputError(f"{name}: at least one per_image_metrics file is required")

    steps: dict[int, dict[str, dict[str, Any]]] = {}
    for step in requested_steps:
        raw_assets = manifest_steps.get(str(step))
        if not isinstance(raw_assets, list) or len(raw_assets) != expected_images:
            count = len(raw_assets) if isinstance(raw_assets, list) else 0
            raise ReportInputError(
                f"{name}: step {step} has {count} images, expected {expected_images}"
            )
        assets: list[dict[str, Any]] = []
        for raw_asset in raw_assets:
            local_path = Path(str(raw_asset.get("local_path") or "")).expanduser().resolve()
            if not local_path.is_file():
                raise ReportInputError(f"{name}: missing image {local_path}")
            sample_key = canonical_sample_key(str(raw_asset.get("file_name") or ""))
            assets.append(
                {
                    "asset_id": str(raw_asset.get("asset_id") or ""),
                    "file_name": str(raw_asset.get("file_name") or ""),
                    "sample_key": sample_key,
                    "path": local_path,
                    "metrics": {},
                }
            )
        sample_counts = Counter(asset["sample_key"] for asset in assets)
        duplicates = sorted(key for key, count in sample_counts.items() if count != 1)
        if duplicates:
            raise ReportInputError(f"{name}: duplicate sample keys at step {step}: {duplicates}")
        merge_metric_sources(assets, step, metric_paths)
        steps[step] = {asset["sample_key"]: asset for asset in assets}

    return {
        "name": name,
        "validation_key": str(run_cfg.get("validation_key") or ""),
        "training_key": str(run_cfg.get("training_key") or ""),
        "steps": steps,
    }


def metric_text(asset: dict[str, Any], metric_specs: list[dict[str, str]]) -> str:
    labels: list[str] = []
    for spec in metric_specs:
        key = spec["key"]
        raw_value = asset["metrics"].get(key)
        try:
            value = float(raw_value)
            rendered = format(value, spec["format"])
        except (TypeError, ValueError):
            rendered = "NA"
        labels.append(f"{spec['label']} {rendered}")
    midpoint = math.ceil(len(labels) / 2)
    return " | ".join(labels[:midpoint]) + "\n" + " | ".join(labels[midpoint:])


def render_cover(
    pdf: PdfPages,
    title: str,
    runs: list[dict[str, Any]],
    steps: list[int],
    metric_specs: list[dict[str, str]],
    dpi: int,
) -> None:
    fig = plt.figure(figsize=LANDSCAPE_A4)
    fig.text(0.05, 0.92, title, fontsize=22, weight="bold", va="top")
    fig.text(
        0.05,
        0.84,
        "Exact full-panel comparison",
        fontsize=14,
        color="#444444",
        va="top",
    )
    lines = [f"Steps: {', '.join(f'{step:,}' for step in steps)}", "", "Runs:"]
    for run in runs:
        lines.append(
            f"• {run['name']}\n"
            f"  validation {run['validation_key']} | training {run['training_key'] or 'n/a'}"
        )
    lines.extend(
        [
            "",
            "Per-image labels:",
            " | ".join(f"{spec['label']} = {spec['key']}" for spec in metric_specs),
        ]
    )
    fig.text(
        0.06,
        0.77,
        "\n".join(lines),
        fontsize=10,
        va="top",
        family="monospace",
        linespacing=1.3,
    )
    fig.text(
        0.05,
        0.035,
        "Images are matched by normalized Comet filename; metrics join by "
        "step + immutable asset ID with a step + sample-key fallback.",
        fontsize=8,
        color="#555555",
    )
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def render_pages(
    pdf: PdfPages,
    title: str,
    runs: list[dict[str, Any]],
    steps: list[int],
    sample_keys: list[str],
    samples_per_page: int,
    metric_specs: list[dict[str, str]],
    dpi: int,
    thumbnail_side: int,
    highlight_dir: Path | None = None,
    highlight_steps: set[int] | None = None,
    highlight_samples: set[str] | None = None,
) -> None:
    page_number = 1
    for step in steps:
        for start in range(0, len(sample_keys), samples_per_page):
            page_samples = sample_keys[start : start + samples_per_page]
            fig = plt.figure(figsize=LANDSCAPE_A4)
            grid = GridSpec(
                len(page_samples) + 1,
                len(runs),
                figure=fig,
                left=0.055,
                right=0.99,
                top=0.91,
                bottom=0.05,
                hspace=0.34,
                wspace=0.08,
                height_ratios=[0.13] + [1.0] * len(page_samples),
            )
            fig.text(
                0.02,
                0.975,
                f"{title} — step {step:,}",
                fontsize=13,
                weight="bold",
                va="top",
            )
            for column, run in enumerate(runs):
                header = fig.add_subplot(grid[0, column])
                header.axis("off")
                header.text(
                    0.5,
                    0.45,
                    "\n".join(textwrap.wrap(run["name"], 22)[:2]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                )
            for row, sample_key in enumerate(page_samples, start=1):
                for column, run in enumerate(runs):
                    asset = run["steps"][step][sample_key]
                    axis = fig.add_subplot(grid[row, column])
                    with Image.open(asset["path"]) as source:
                        image = source.convert("RGB")
                        image.thumbnail((thumbnail_side, thumbnail_side), Image.LANCZOS)
                    axis.imshow(image)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    axis.set_title(
                        metric_text(asset, metric_specs),
                        fontsize=5.0,
                        pad=1.5,
                        linespacing=1.05,
                    )
                    if column == 0:
                        axis.set_ylabel(
                            "\n".join(textwrap.wrap(sample_key, 20)[:3]),
                            fontsize=6,
                            rotation=0,
                            labelpad=37,
                            va="center",
                        )
            fig.text(
                0.985,
                0.015,
                f"Page {page_number}",
                ha="right",
                fontsize=7,
                color="#555555",
            )
            if (
                highlight_dir is not None
                and step in (highlight_steps or set())
                and set(page_samples) & (highlight_samples or set())
            ):
                highlight_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    highlight_dir / f"step_{step:06d}_page_{page_number:03d}.png",
                    dpi=dpi,
                )
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            page_number += 1


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ReportInputError("Config must be a YAML object")
    base_dir = config_path.parent
    steps = [int(step) for step in cfg.get("steps", [])]
    if not steps or len(steps) != len(set(steps)):
        raise ReportInputError("steps must be a non-empty unique list")
    expected_images = int(cfg.get("expected_images_per_step", 96))
    samples_per_page = int(cfg.get("samples_per_page", 6))
    if expected_images < 1 or samples_per_page < 1:
        raise ReportInputError("Image counts must be positive")
    run_configs = cfg.get("runs")
    if not isinstance(run_configs, list) or len(run_configs) < 2:
        raise ReportInputError("runs must contain at least two run objects")

    runs = [
        load_run(run_cfg, base_dir, steps, expected_images)
        for run_cfg in run_configs
    ]
    base_samples = set(runs[0]["steps"][steps[0]])
    if len(base_samples) != expected_images:
        raise ReportInputError("First run does not contain the expected sample count")
    for run in runs:
        for step in steps:
            samples = set(run["steps"][step])
            if samples != base_samples:
                raise ReportInputError(
                    f"{run['name']}: sample set differs at step {step}; "
                    f"missing={sorted(base_samples - samples)}, "
                    f"extra={sorted(samples - base_samples)}"
                )

    metric_specs = cfg.get("metrics", DEFAULT_METRICS)
    if not isinstance(metric_specs, list) or not metric_specs:
        raise ReportInputError("metrics must be a non-empty list")
    normalized_specs: list[dict[str, str]] = []
    for index, spec in enumerate(metric_specs):
        if not isinstance(spec, dict) or not spec.get("key"):
            raise ReportInputError(f"metrics[{index}] requires a key")
        normalized_specs.append(
            {
                "key": str(spec["key"]),
                "label": str(spec.get("label") or spec["key"]),
                "format": str(spec.get("format") or ".3f"),
            }
        )

    # 27 Jul 2026 - AICODE-NOTE: A report must never silently compare
    # different validation samples or substitute a nearby Comet image step.
    for run in runs:
        for step in steps:
            for asset in run["steps"][step].values():
                missing = [
                    spec["key"]
                    for spec in normalized_specs
                    if spec["key"] not in asset["metrics"]
                ]
                if missing:
                    raise ReportInputError(
                        f"{run['name']} step {step} sample {asset['sample_key']!r} "
                        f"is missing metrics: {missing}"
                    )

    output_path = resolve_path(base_dir, cfg.get("out_pdf"), "out_pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = str(cfg.get("title") or "Longitudinal validation comparison")
    dpi = int(cfg.get("dpi", 180))
    thumbnail_side = int(cfg.get("thumbnail_side", 512))
    highlight_dir = (
        resolve_path(base_dir, cfg["highlight_dir"], "highlight_dir")
        if cfg.get("highlight_dir")
        else None
    )
    highlight_steps = {int(step) for step in cfg.get("highlight_steps", [])}
    highlight_samples = {str(sample) for sample in cfg.get("highlight_samples", [])}
    if highlight_steps - set(steps):
        raise ReportInputError(
            f"highlight_steps are not report steps: {sorted(highlight_steps - set(steps))}"
        )
    if highlight_samples - base_samples:
        raise ReportInputError(
            "highlight_samples are not in the sealed sample set: "
            f"{sorted(highlight_samples - base_samples)}"
        )
    with PdfPages(output_path) as pdf:
        render_cover(pdf, title, runs, steps, normalized_specs, dpi)
        render_pages(
            pdf,
            title,
            runs,
            steps,
            sorted(base_samples),
            samples_per_page,
            normalized_specs,
            dpi,
            thumbnail_side,
            highlight_dir,
            highlight_steps,
            highlight_samples,
        )
    print(
        f"Wrote {output_path} with {len(runs)} runs, {len(steps)} steps, "
        f"and {len(base_samples)} samples per step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
