#!/usr/bin/env python3
"""Fail-closed integrity and face-alignment gate for seed-specific CL39 boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def normalize_key(value: str) -> str:
    return value.replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mean-iou", type=float, default=0.50)
    parser.add_argument("--maximum-unowned", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boxes = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    images = sorted(args.images_root.glob("step_24000_batch_*/*.png"))
    image_by_name = {path.name: path for path in images}
    if len(boxes) != 96 or len(image_by_name) != 96:
        raise RuntimeError(
            f"Expected 96 boxes and images, found boxes={len(boxes)} images={len(image_by_name)}"
        )

    from src.face_subject_selector import bbox_iou
    from src.metrics.aligner import Aligner

    aligner = Aligner()
    rows = []
    for key, record in boxes.items():
        meta = record.get("_meta") or {}
        if int(meta.get("seed", -1)) != args.seed:
            raise RuntimeError(f"Seed metadata mismatch for {key}: {meta}")
        bbox = record.get("face_crop_new") or record.get("face_crop_old")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(f"Invalid bbox for {key}: {bbox}")
        image_path = image_by_name.get(normalize_key(key))
        if image_path is None:
            raise RuntimeError(f"No generated A-arm image for bbox key {key!r}")
        detected, _embeds = aligner([Image.open(image_path).convert("RGB")])
        faces = detected[0] or []
        best_iou = max((float(bbox_iou(face, bbox)) for face in faces), default=0.0)
        rows.append(
            {
                "output_key": key,
                "normalized_output_key": normalize_key(key),
                "face_count": len(faces),
                "best_iou": best_iou,
                "unowned": int(best_iou < 0.05),
            }
        )

    mean_iou = sum(row["best_iou"] for row in rows) / len(rows)
    no_face = sum(row["face_count"] == 0 for row in rows)
    unowned = sum(row["unowned"] for row in rows)
    payload = {
        "schema_version": 1,
        "validation_seed": args.seed,
        "bbox_count": len(boxes),
        "image_count": len(image_by_name),
        "no_face": no_face,
        "unowned": unowned,
        "mean_best_iou": mean_iou,
        "minimum_mean_iou": args.minimum_mean_iou,
        "maximum_unowned": args.maximum_unowned,
        "accepted": bool(
            no_face == 0
            and unowned <= args.maximum_unowned
            and mean_iou >= args.minimum_mean_iou
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if not payload["accepted"]:
        raise SystemExit("Dynamic bbox alignment gate failed")


if __name__ == "__main__":
    main()
