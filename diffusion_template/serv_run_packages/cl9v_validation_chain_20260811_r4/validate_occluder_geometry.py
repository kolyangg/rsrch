#!/usr/bin/env python3
"""Fail-closed validation and visual overlays for CL9 occluder polygons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPECTED = {
    2: "Skiing",
    7: "Crying",
    14: "Skiing",
    19: "Crying",
    26: "Skiing",
    31: "Crying",
    38: "Skiing",
    43: "Crying",
    50: "Skiing",
    55: "Crying",
    62: "Skiing",
    67: "Crying",
    74: "Skiing",
    79: "Crying",
    86: "Skiing",
    91: "Crying",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_point(point) -> tuple[float, float]:
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"Invalid normalized point: {point!r}")
    x, y = [float(value) for value in point]
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"Normalized point outside [0,1]: {point!r}")
    return x, y


def polygon_mask(size: tuple[int, int], polygons) -> Image.Image:
    width, height = size
    blocked = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(blocked)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("Each exclusion polygon must contain at least 3 points")
        points = []
        for raw_point in polygon:
            x, y = checked_point(raw_point)
            points.append((int(round(x * (width - 1))), int(round(y * (height - 1)))))
        draw.polygon(points, fill=255)
    return blocked


def compose_row(crop: Image.Image, blocked: Image.Image, title: str) -> Image.Image:
    tile = 256
    crop = crop.resize((tile, tile), Image.Resampling.LANCZOS)
    blocked = blocked.resize((tile, tile), Image.Resampling.NEAREST)
    overlay = crop.copy()
    red = Image.new("RGB", overlay.size, color=(255, 20, 20))
    overlay = Image.composite(red, overlay, blocked.point(lambda value: value // 2))
    visibility = Image.eval(blocked, lambda value: 255 - value).convert("RGB")
    row = Image.new("RGB", (tile * 3, tile + 28), color="white")
    row.paste(crop, (0, 28))
    row.paste(overlay, (tile, 28))
    row.paste(visibility, (tile * 2, 28))
    draw = ImageDraw.Draw(row)
    draw.text((4, 7), title, fill="black", font=ImageFont.load_default())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Expected visibility plan schema_version 1")
    if payload.get("coordinate_space") != "generation_bbox_fraction":
        raise ValueError("Visibility plan must use generation_bbox_fraction")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Visibility plan entries must be an object")
    parsed_indices = {int(key) for key in entries}
    if parsed_indices != set(EXPECTED):
        raise ValueError(
            f"Visibility indices drifted: {sorted(parsed_indices)} != {sorted(EXPECTED)}"
        )

    baseline_rows = json.loads(
        (baseline_dir / "per_image.json").read_text(encoding="utf-8")
    )
    rows_by_index = {int(row["dataset_index"]): row for row in baseline_rows}
    if sorted(rows_by_index) != list(range(96)):
        raise ValueError("Baseline must contain the full fixed 96-row panel")

    result_rows = []
    contact_rows: dict[str, list[Image.Image]] = {"Skiing": [], "Crying": []}
    for dataset_index, expected_family in EXPECTED.items():
        entry = entries[str(dataset_index)]
        if entry.get("family") != expected_family:
            raise ValueError(f"Family mismatch at dataset index {dataset_index}")
        polygons = entry.get("exclude_polygons")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Index {dataset_index} has no exclusion polygons")
        row = rows_by_index[dataset_index]
        image_path = baseline_dir / "images" / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        bbox = [int(round(float(value))) for value in row["face_bbox_gen"]]
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid generation bbox for index {dataset_index}: {bbox}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            crop = image.crop((x0, y0, x1, y1))
        blocked = polygon_mask(crop.size, polygons)
        blocked_fraction = float(np.asarray(blocked, dtype=np.float32).mean() / 255.0)
        # 11 Aug 2026 - AICODE-NOTE: The oracle geometry must stay local. A
        # near-empty mask is not a mechanism test; a majority-face mask merely
        # recreates the rejected family-wide ownership intervention.
        if not 0.02 <= blocked_fraction <= 0.50:
            raise ValueError(
                f"Index {dataset_index} blocked fraction {blocked_fraction:.4f} "
                "is outside the precise-geometry contract [0.02, 0.50]"
            )
        mask_path = output_dir / f"{dataset_index:03d}_blocked.png"
        overlay_path = output_dir / f"{dataset_index:03d}_overlay.png"
        blocked.save(mask_path)
        visual = compose_row(
            crop,
            blocked,
            f"{dataset_index:03d} {expected_family} | original / blocked overlay / visibility",
        )
        visual.save(overlay_path)
        contact_rows[expected_family].append(visual)
        result_rows.append(
            {
                "dataset_index": dataset_index,
                "family": expected_family,
                "filename": row["filename"],
                "face_bbox_gen": bbox,
                "polygon_count": len(polygons),
                "blocked_fraction_within_bbox": blocked_fraction,
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "overlay_path": str(overlay_path),
                "review": entry.get("review"),
            }
        )

    for family, family_rows in contact_rows.items():
        width = max(row.width for row in family_rows)
        height = sum(row.height for row in family_rows)
        sheet = Image.new("RGB", (width, height), color="white")
        y = 0
        for row in family_rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(output_dir / f"{family.lower()}_geometry_review.png")

    summary = {
        "schema_version": 1,
        "kind": "cl9_precise_occluder_geometry_preflight",
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "baseline_dir": str(baseline_dir),
        "baseline_run_manifest_sha256": sha256_file(
            baseline_dir / "run_manifest.json"
        ),
        "row_count": len(result_rows),
        "families": {family: len(rows) for family, rows in contact_rows.items()},
        "rows": result_rows,
    }
    summary_path = output_dir / "geometry_preflight.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
