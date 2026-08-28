#!/usr/bin/env python3
"""Build the finite raw-target/raw-reference cache recipe for CL39-X05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def valid_bbox(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    x0, y0, x1, y1 = (float(item) for item in value)
    return 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]


def lookup_bbox(values, path):
    for key in (str(path), str(path).lstrip("/")):
        if key in values:
            return values[key]
    return None


def absolute_path(root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-version", default="automask_os_v1")
    parser.add_argument("--min-face-res", type=int, default=192)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root.expanduser().absolute()
    records, references, accepted = [], set(), 0
    for target_path, row in manifest.items():
        target_bbox = row.get("face_crop_new") if isinstance(row, dict) else None
        if not valid_bbox(target_bbox, (1024, 1024)):
            continue
        x0, y0, x1, y1 = (float(item) for item in target_bbox)
        if min(x1-x0, y1-y0) < args.min_face_res:
            continue
        candidates = []
        bboxes = row.get("face_bboxes") or {}
        for path in row.get("face_paths") or []:
            bbox = lookup_bbox(bboxes, path)
            if valid_bbox(bbox, (256, 256)):
                candidates.append((str(path), bbox))
        if not candidates:
            continue
        absolute_target = absolute_path(dataset_root, target_path)
        identity = {
            "kind": "target", "path": absolute_target,
            "policy": args.policy_version,
        }
        first_reference = absolute_path(dataset_root, candidates[0][0])
        records.append({
            "image_path": absolute_target,
            "reference_image_path": first_reference,
            "body_crop": row.get("body_crop"),
            "expected_location": target_bbox,
            "cache_identity": identity,
        })
        for relative_reference, bbox in candidates:
            absolute_reference = absolute_path(dataset_root, relative_reference)
            if absolute_reference in references:
                continue
            references.add(absolute_reference)
            records.append({
                "image_path": absolute_reference,
                "reference_image_path": absolute_reference,
                "expected_location": bbox,
                "cache_identity": {
                    "kind": "reference_source", "path": absolute_reference,
                    "policy": args.policy_version,
                },
            })
        accepted += 1
    if not records:
        raise RuntimeError("No Cosmic rows passed the CL39-X05 recipe filters")
    payload = {
        "kind": "cl39x05_cosmic_raw_sources_v1",
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(dataset_root),
        "accepted_targets": accepted,
        "unique_references": len(references),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "targets": accepted, "references": len(references)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
