#!/usr/bin/env python3
"""Fail-closed ownership/alignment gate for the seed-1 PM-derived boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def normalized(value: str) -> str:
    return Path(value).name.replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boxes = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    images = {path.name: path for path in args.images_root.glob("step_16000_batch_*/*.png")}
    if len(boxes) != 96 or len(images) != 96:
        raise RuntimeError(f"Expected 96 boxes/images, got {len(boxes)}/{len(images)}")
    if {int((value.get("_meta") or {}).get("seed", -1)) for value in boxes.values()} != {1}:
        raise RuntimeError("Dynamic bbox cache is not sealed to inference seed 1")
    from src.face_subject_selector import bbox_iou
    from src.metrics.aligner import Aligner
    aligner = Aligner()
    rows = []
    for key, record in boxes.items():
        box = record.get("face_crop_new") or record.get("face_crop_old")
        path = images.get(normalized(key))
        if path is None or not box:
            raise RuntimeError(f"Seed-1 bbox/image join failed for {key}")
        detected, _ = aligner([Image.open(path).convert("RGB")])
        faces = detected[0] or []
        best = max((float(bbox_iou(face, box)) for face in faces), default=0.0)
        rows.append({"output_key": key, "face_count": len(faces), "best_iou": best,
                     "unowned": int(best < 0.05)})
    no_face = sum(row["face_count"] == 0 for row in rows)
    unowned = sum(row["unowned"] for row in rows)
    mean_iou = sum(row["best_iou"] for row in rows) / 96
    payload = {
        "schema_version": 1, "validation_seed": 1, "bbox_count": 96,
        "image_count": 96, "no_face": no_face, "unowned": unowned,
        "mean_best_iou": mean_iou,
        "accepted": no_face == 0 and unowned <= 2 and mean_iou >= 0.50,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if not payload["accepted"]:
        raise SystemExit("Seed-1 dynamic bbox alignment gate failed")


if __name__ == "__main__":
    main()
