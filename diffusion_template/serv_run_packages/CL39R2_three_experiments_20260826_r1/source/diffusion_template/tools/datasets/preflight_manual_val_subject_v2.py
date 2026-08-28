#!/usr/bin/env python3
"""Fail-closed audit of the fixed-96 reference-subject contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_subject_selector import BBOX_OVERLAP_V2, select_subject_face
from src.metrics.aligner import Aligner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/references"),
    )
    parser.add_argument(
        "--bbox-json",
        type=Path,
        default=Path("../dataset_full/val_dataset/ref_bboxes.json"),
    )
    parser.add_argument("--expected-identities", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bbox_payload = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    bboxes = {}
    for key, value in bbox_payload.items():
        bbox = value.get("face_crop_new") or value.get("face_crop_old")
        if bbox is not None:
            bboxes[Path(key).stem] = bbox

    image_paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(image_paths) != args.expected_identities:
        raise RuntimeError(
            f"Expected {args.expected_identities} references, found {len(image_paths)}"
        )
    if {path.stem for path in image_paths} != set(bboxes):
        raise RuntimeError("Reference image and declared-bbox identity sets differ")

    aligner = Aligner()
    rows = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        detected_boxes, detected_embeddings = aligner([image])
        faces = [
            {"bbox": bbox, "embedding": embedding}
            for bbox, embedding in zip(detected_boxes[0] or [], detected_embeddings[0] or [])
        ]
        _selected, audit = select_subject_face(
            faces,
            declared_bbox=bboxes[path.stem],
            policy=BBOX_OVERLAP_V2,
        )
        rows.append({"identity": path.stem, **audit.to_dict()})

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "manual_val_subject_v2_preflight",
                    "identity_count": len(rows),
                    "ambiguous_count": sum(bool(row["ambiguous"]) for row in rows),
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        "MANUAL_VAL_SUBJECT_V2_PREFLIGHT_OK "
        f"identities={len(rows)} multi_face={sum(row['face_count'] > 1 for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
