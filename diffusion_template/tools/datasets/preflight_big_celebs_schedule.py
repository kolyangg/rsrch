#!/usr/bin/env python3
"""Fail-closed audit of a pinned BigCelebs target/reference schedule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image


DIRECTION_RE = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def face_side(metadata: dict) -> float:
    x0, y0, x1, y1 = metadata["new_face_crop"]
    return min(float(x1) - float(x0), float(y1) - float(y0))


def decode(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG" or image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError(
                f"Invalid scheduled image {path}: format={image.format}, "
                f"mode={image.mode}, size={image.size}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--schedule-start-step", type=int, default=0)
    parser.add_argument("--schedule-start-row", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_digest = sha256(args.manifest)
    if manifest_digest != args.expected_manifest_sha256.strip().lower():
        raise RuntimeError("BigCelebs source-manifest SHA-256 mismatch")
    plan_digest = sha256(args.plan)
    if plan_digest != args.expected_plan_sha256.strip().lower():
        raise RuntimeError("BigCelebs sampling-plan SHA-256 mismatch")

    plan_manifest = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    if plan_manifest.get("kind") != "big_celebs_sampling_plan":
        raise RuntimeError("Unexpected sampling-plan manifest kind")
    if plan_manifest.get("source_manifest_sha256") != manifest_digest:
        raise RuntimeError("Sampling plan belongs to a different source manifest")
    if plan_manifest.get("plan_file_sha256") != plan_digest:
        raise RuntimeError("Sampling plan does not match its manifest")
    batch_size = int(plan_manifest["batch_size"])
    expected_start_row = args.schedule_start_step * batch_size
    if args.schedule_start_row != expected_start_row:
        raise RuntimeError(
            "Schedule recovery offset mismatch: "
            f"step={args.schedule_start_step}, batch_size={batch_size}, "
            f"expected_row={expected_start_row}, configured_row={args.schedule_start_row}"
        )

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    identities: Counter[str] = Counter()
    target_bins: Counter[str] = Counter()
    with args.plan.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_index = len(rows)
            if int(row["row"]) != row_index:
                raise RuntimeError(f"Non-contiguous plan row at line {line_number}")
            if int(row["optimizer_step"]) != row_index // batch_size:
                raise RuntimeError(f"Invalid optimizer step at plan row {row_index}")
            identity = str(row["identity_id"])
            target_id = str(row["target_image_id"])
            reference_id = str(row["reference_image_id"])
            if identity not in records:
                raise RuntimeError(f"Unknown identity at plan row {row_index}")
            if target_id == reference_id:
                raise RuntimeError(f"Self-reference at plan row {row_index}")
            if target_id not in records[identity] or reference_id not in records[identity]:
                raise RuntimeError(f"Unknown target/reference at plan row {row_index}")
            target_metadata = records[identity][target_id]
            reference_metadata = records[identity][reference_id]
            target_side = face_side(target_metadata)
            reference_side = face_side(reference_metadata)
            expected_bin = "ge256" if target_side >= 256 else "192_255"
            if row["target_face_bin"] != expected_bin or target_side < 192:
                raise RuntimeError(f"Target face policy violation at row {row_index}")
            if reference_side < 256:
                raise RuntimeError(f"Reference face policy violation at row {row_index}")
            if bool(row["flip_target"]) and DIRECTION_RE.search(target_metadata["text"]):
                raise RuntimeError(f"Directional caption flipped at row {row_index}")
            if int(row["reference_rank"]) not in {1, 2, 3}:
                raise RuntimeError(f"Invalid reference rank at row {row_index}")
            rows.append(row)
            identities[identity] += 1
            target_bins[str(row["target_face_bin"])] += 1

    if len(rows) != int(plan_manifest["rows"]):
        raise RuntimeError("Sampling-plan row count does not match its manifest")
    if not 0 <= args.schedule_start_row < len(rows):
        raise RuntimeError("Configured schedule start is outside the plan")

    available = rows[args.schedule_start_row :]
    rng = random.Random(20260801 + args.schedule_start_row)
    sampled_rows = rng.sample(available, min(args.sample_count, len(available)))
    decoded_pairs = []
    for row in sampled_rows:
        identity = str(row["identity_id"])
        target = f"{identity}/{row['target_image_id']}.jpg"
        reference = f"{identity}/{row['reference_image_id']}.jpg"
        decode(args.images_root / target)
        decode(args.images_root / reference)
        decoded_pairs.append(
            {"row": row["row"], "target": target, "reference": reference}
        )

    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "plan": str(args.plan),
        "plan_sha256": plan_digest,
        "rows": len(rows),
        "schedule_start_step": args.schedule_start_step,
        "schedule_start_row": args.schedule_start_row,
        "remaining_rows": len(available),
        "sampled_identities": len(identities),
        "target_bins": dict(sorted(target_bins.items())),
        "decoded_pairs": decoded_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
