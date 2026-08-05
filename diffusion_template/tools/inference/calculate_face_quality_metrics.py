#!/usr/bin/env python3
"""Score generated-face crops with no-reference IQA models.

The input manifest is produced by ``tools/comet/backfill_face_quality_metrics.py``.
This script deliberately does not use a reference identity image: it measures
visual face quality/coherence, while identity similarity remains a separate
metric.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.id_utils import analyze_faces, create_face_analyzer


DEFAULT_METRICS = (
    "topiq_nr-face",
    "topiq_nr",
    "musiq",
    "maniqa-pipal",
)
SCORE_DIRECTION = {name: "higher_is_better" for name in DEFAULT_METRICS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate face-crop IQA metrics for an exact Comet image manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.25,
        help="Padding on each side as a fraction of the largest detected-face side.",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args(argv)


def metric_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise ValueError("At least one IQA metric is required")
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"Duplicate metric names are not allowed: {metrics}")
    return metrics


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    steps = payload.get("steps")
    if not isinstance(steps, dict) or not steps:
        raise ValueError(f"Manifest has no steps: {path}")
    for step, assets in steps.items():
        int(step)
        if not isinstance(assets, list) or not assets:
            raise ValueError(f"Manifest step {step} has no assets")
        for asset in assets:
            image_path = Path(asset["local_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
    return payload


def build_face_detector():
    # 27 Jul 2026 - AICODE-NOTE: Keep detection on CPU so IQA GPU execution is
    # independent of machine-specific ONNX Runtime/CuDNN provider libraries.
    return create_face_analyzer(
        providers=["CPUExecutionProvider"],
        provider_options=None,
        allowed_modules=["detection"],
        ctx_id=-1,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )


def choose_largest_face(faces) -> Any:
    return max(
        faces,
        key=lambda face: max(0.0, float(face["bbox"][2] - face["bbox"][0]))
        * max(0.0, float(face["bbox"][3] - face["bbox"][1])),
    )


def square_face_crop(
    image: Image.Image,
    bbox: list[float],
    padding: float,
    output_size: int,
) -> tuple[Image.Image, list[int], int]:
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    face_side = max(x1 - x0, y1 - y0)
    side = max(2, int(round(face_side * (1.0 + 2.0 * padding))))
    side = min(side, width, height)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0 = int(round(center_x - side / 2.0))
    crop_y0 = int(round(center_y - side / 2.0))
    crop_x0 = min(max(crop_x0, 0), width - side)
    crop_y0 = min(max(crop_y0, 0), height - side)
    crop_box = [crop_x0, crop_y0, crop_x0 + side, crop_y0 + side]
    crop = image.crop(tuple(crop_box))
    if crop.size != (output_size, output_size):
        crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return crop, crop_box, side


def to_float_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).float().div_(255.0)


def finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def aggregate(values: list[float | None]) -> dict[str, float | int | None]:
    clean = finite_values(values)
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "std": None,
        }
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "p10": percentile(clean, 0.10),
        "std": float(pstdev(clean)),
    }


def score_batches(
    model,
    tensors: list[torch.Tensor],
    batch_size: int,
    device: str,
    tolerate_single_image_failures: bool = False,
) -> list[float | None]:
    results: list[float | None] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
            try:
                prediction = model(batch)
            except Exception as error:
                if not tolerate_single_image_failures or batch.shape[0] != 1:
                    raise
                print(
                    "FACE_QUALITY_MODEL_SKIPPED "
                    f"crop_index={start} error={type(error).__name__}: {error}"
                )
                results.append(None)
                continue
            values = prediction.detach().float().cpu().reshape(-1).tolist()
            expected = batch.shape[0]
            if len(values) != expected:
                raise RuntimeError(
                    f"IQA model returned {len(values)} values for a batch of {expected}"
                )
            results.extend(float(value) for value in values)
    return results


def prepare_step_rows(
    assets: list[dict[str, Any]],
    face_detector,
    crop_padding: float,
    crop_size: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    crops: list[torch.Tensor] = []
    for asset in assets:
        image_path = Path(asset["local_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        bgr = np.asarray(image)[:, :, ::-1]
        faces = analyze_faces(face_detector, bgr)
        row: dict[str, Any] = {
            "asset_id": asset["asset_id"],
            "file_name": asset["file_name"],
            "local_path": str(image_path),
            "image_width": image.width,
            "image_height": image.height,
            "face_detected": int(bool(faces)),
            "face_count": len(faces),
            "det_score": None,
            "bbox_x0": None,
            "bbox_y0": None,
            "bbox_x1": None,
            "bbox_y1": None,
            "crop_x0": None,
            "crop_y0": None,
            "crop_x1": None,
            "crop_y1": None,
            "crop_source_side": None,
            "face_area_ratio": None,
        }
        if not faces:
            rows.append(row)
            continue

        face = choose_largest_face(faces)
        bbox = [float(value) for value in face["bbox"]]
        bbox[0] = min(max(bbox[0], 0.0), float(image.width))
        bbox[2] = min(max(bbox[2], 0.0), float(image.width))
        bbox[1] = min(max(bbox[1], 0.0), float(image.height))
        bbox[3] = min(max(bbox[3], 0.0), float(image.height))
        crop, crop_box, crop_source_side = square_face_crop(
            image, bbox, crop_padding, crop_size
        )
        face_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        row.update(
            {
                "det_score": float(face["det_score"]),
                "bbox_x0": bbox[0],
                "bbox_y0": bbox[1],
                "bbox_x1": bbox[2],
                "bbox_y1": bbox[3],
                "crop_x0": crop_box[0],
                "crop_y0": crop_box[1],
                "crop_x1": crop_box[2],
                "crop_y1": crop_box[3],
                "crop_source_side": crop_source_side,
                "face_area_ratio": face_area / float(image.width * image.height),
                "_crop_index": len(crops),
            }
        )
        crops.append(to_float_tensor(crop))
        rows.append(row)
    return rows, crops


def write_csv(path: Path, rows: list[dict[str, Any]], metric_slugs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "asset_id",
        "file_name",
        "local_path",
        "image_width",
        "image_height",
        "face_detected",
        "face_count",
        "det_score",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_source_side",
        "face_area_ratio",
        *metric_slugs,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = parse_metrics(args.metrics)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.crop_size < 64:
        raise ValueError("--crop-size must be at least 64")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is not available: {args.device}")

    import pyiqa

    manifest = load_manifest(args.manifest.resolve())
    torch.manual_seed(0)
    np.random.seed(0)
    face_detector = build_face_detector()

    models: dict[str, Any] = {}
    for name in metrics:
        if name not in pyiqa.list_models():
            raise ValueError(f"Unknown PyIQA model: {name}")
        models[name] = pyiqa.create_metric(name, device=args.device)
        models[name].eval()

    all_rows: list[dict[str, Any]] = []
    step_results: dict[str, Any] = {}
    metric_slugs = [metric_slug(name) for name in metrics]
    for step_text in sorted(manifest["steps"], key=int):
        assets = manifest["steps"][step_text]
        rows, crops = prepare_step_rows(
            assets,
            face_detector,
            args.crop_padding,
            args.crop_size,
        )
        for name, slug in zip(metrics, metric_slugs):
            # PyIQA's face-specific TOPIQ runs its own face alignment and only
            # accepts one input image at a time.
            model_batch_size = 1 if name == "topiq_nr-face" else args.batch_size
            scores = score_batches(
                models[name],
                crops,
                model_batch_size,
                args.device,
                tolerate_single_image_failures=name == "topiq_nr-face",
            )
            for row in rows:
                crop_index = row.get("_crop_index")
                row[slug] = scores[crop_index] if crop_index is not None else None

        step = int(step_text)
        for row in rows:
            row["step"] = step
            row.pop("_crop_index", None)
        detected_rows = [row for row in rows if row["face_detected"]]
        step_results[step_text] = {
            "image_count": len(rows),
            "detected_face_count": len(detected_rows),
            "face_detection_rate": len(detected_rows) / len(rows),
            "multi_face_count": sum(row["face_count"] > 1 for row in rows),
            "multi_face_rate": sum(row["face_count"] > 1 for row in rows) / len(rows),
            "det_score": aggregate([row["det_score"] for row in rows]),
            "face_area_ratio": aggregate([row["face_area_ratio"] for row in rows]),
            "metrics": {
                slug: {
                    "source_model": name,
                    "score_direction": SCORE_DIRECTION.get(name, "higher_is_better"),
                    **aggregate([row.get(slug) for row in rows]),
                }
                for name, slug in zip(metrics, metric_slugs)
            },
        }
        all_rows.extend(rows)
        print(
            "FACE_QUALITY_STEP_COMPLETE "
            f"step={step} images={len(rows)} detected={len(detected_rows)}"
        )

    result = {
        "schema_version": 1,
        "kind": "no_reference_face_quality_metrics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "experiment_key": manifest.get("experiment_key"),
        "project_name": manifest.get("project_name"),
        "metric_backend": {
            "pyiqa_version": importlib.metadata.version("pyiqa"),
            "torch_version": torch.__version__,
            "insightface_version": importlib.metadata.version("insightface"),
            "metrics": metrics,
            "score_direction": {
                metric_slug(name): SCORE_DIRECTION.get(name, "higher_is_better")
                for name in metrics
            },
            "device": args.device,
            "batch_size": args.batch_size,
            "crop_policy": {
                "detector": "InsightFace detection-only; largest face",
                "padding_each_side": args.crop_padding,
                "square_crop": True,
                "resize": [args.crop_size, args.crop_size],
                "resampling": "PIL Lanczos",
                "undetected_faces": "excluded from IQA aggregates and reported by face_detection_rate",
            },
        },
        "steps": step_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, all_rows, metric_slugs)
    print(f"FACE_QUALITY_RESULTS json={args.output_json} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
