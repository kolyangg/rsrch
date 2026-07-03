#!/usr/bin/env python
"""Crop validation reference images to the training-style face crop.

CosmicLargeTrain feeds the model tight square face crops (+20% margin) as
references, while validation feeds full photos. This script pre-crops a
references directory with the exact training logic
(CosmicLargeTrain._get_bigger_crop_with_bbox) so the ref-domain gap can be
A/B-tested without touching pipeline code.

Usage:
    python scripts/crop_refs_to_face.py \
        --images-dir ../dataset_full/val_dataset/references_two \
        --bbox-json ../dataset_full/val_dataset/ref_bboxes.json \
        --out-dir ../dataset_full/val_dataset/references_two_cropped \
        --out-json ../dataset_full/val_dataset/ref_bboxes_two_cropped.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.cosmic import CosmicLargeTrain  # noqa: E402

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--margin", type=float, default=0.2, help="crop margin (training default 0.2)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.bbox_json, encoding="utf-8") as fh:
        bbox_map = json.load(fh)
    bbox_by_stem = {Path(k).stem: v for k, v in bbox_map.items()}

    out_bboxes = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        record = bbox_by_stem.get(path.stem)
        if not isinstance(record, dict):
            print(f"[skip] no bbox record for {path.name}")
            continue
        bbox = record.get("face_crop_new") or record.get("face_crop_old")
        if bbox is None:
            print(f"[skip] bbox record without face_crop for {path.name}")
            continue

        img = Image.open(path).convert("RGB")
        cropped_img, cropped_bbox = CosmicLargeTrain._get_bigger_crop_with_bbox(
            img, [float(v) for v in bbox], scale=args.margin
        )
        if cropped_bbox is None:
            print(f"[skip] invalid bbox after crop for {path.name}")
            continue

        out_path = out_dir / path.name
        cropped_img.save(out_path)
        out_bboxes[path.name] = {"face_crop_new": [round(float(v), 2) for v in cropped_bbox]}
        print(
            f"[ok] {path.name}: {img.size} -> {cropped_img.size}, "
            f"bbox {list(map(int, bbox))} -> {list(map(int, cropped_bbox))}"
        )

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(out_bboxes, fh, indent=2)
    print(f"\nwrote {len(out_bboxes)} refs to {out_dir} and bboxes to {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
