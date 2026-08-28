#!/usr/bin/env python3
"""Build the pinned canonical-reference recipe for CL39-X05 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from src.model.photomaker_branched.masking.automask_os import POLICY_VERSION, image_sha256
from src.pipelines.validation_automask import validation_reference_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--bbox-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-version", default=POLICY_VERSION)
    args = parser.parse_args()
    bboxes = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    records = []
    for path in sorted(args.images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image = Image.open(path).convert("RGB")
        record = bboxes.get(path.name) or bboxes.get(path.stem)
        bbox = None
        if isinstance(record, dict):
            bbox = record.get("face_crop_new") or record.get("face_crop_old")
        if bbox is None:
            raise ValueError(f"Missing canonical reference bbox for {path.name}")
        absolute = str(path.absolute())
        reference_hash = image_sha256(image)
        records.append({
            "image_path": absolute,
            "reference_image_path": absolute,
            "expected_location": [float(value) for value in bbox],
            "cache_identity": validation_reference_identity(
                path.stem, args.policy_version, reference_hash,
            ),
        })
    if not records:
        raise RuntimeError("No manual-validation references found")
    payload = {
        "kind": "cl39x05_manual_val_references_v1",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
