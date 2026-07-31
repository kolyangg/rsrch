#!/usr/bin/env python3
"""Fail-closed structural and decode preflight for LargeDatasetTrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re

from PIL import Image


EXPECTED_MANIFEST_SHA256 = (
    "0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=(
            Path(os.environ["LARGE_DATASET_SEAL"])
            if os.environ.get("LARGE_DATASET_SEAL")
            else None
        ),
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    expected_images = 47500
    expected_identities = 2561
    minimum_face_side = None
    release_name = "legacy_47k"
    if args.dataset_manifest is not None:
        # AICODE-NOTE: 31 Jul 2026 - New large-dataset releases are selected
        # by their sealed manifest; the old 47.5k hash remains a legacy path.
        release = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        ready_path = args.dataset_manifest.parent / "READY"
        if not ready_path.is_file():
            raise RuntimeError(f"Missing release READY seal: {ready_path}")
        release_digest = hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest()
        if ready_path.read_text(encoding="utf-8").strip() != release_digest:
            raise RuntimeError("dataset_manifest.json does not match READY")
        selected = None
        for variant in release.get("variants", {}).values():
            if variant.get("path") == args.manifest.name:
                selected = variant
                break
        if selected is None:
            raise RuntimeError(
                f"Selected manifest is absent from release seal: {args.manifest.name}"
            )
        if digest != selected.get("sha256"):
            raise RuntimeError(
                f"Manifest SHA-256 mismatch: expected {selected.get('sha256')}, "
                f"found {digest}"
            )
        expected_images = int(selected["images"])
        expected_identities = int(selected["identities"])
        minimum_face_side = selected.get("minimum_face_side")
        release_name = str(release.get("release") or args.dataset_manifest.parent.name)
        if release.get("state") != "ready_for_training":
            raise RuntimeError(f"Release is not ready for training: {release.get('state')}")
    elif digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, "
            f"found {digest}"
        )

    with args.manifest.open(encoding="utf-8") as handle:
        records = json.load(handle)
    counts = {identity: len(images) for identity, images in records.items()}
    if len(records) != expected_identities or sum(counts.values()) != expected_images:
        raise RuntimeError(
            f"Unexpected manifest population: {len(records)} IDs, "
            f"{sum(counts.values())} images"
        )
    if min(counts.values()) < 2:
        raise RuntimeError("Every identity must provide a distinct reference")

    trigger = re.compile(r"(?<![A-Za-z0-9_])img(?![A-Za-z0-9_])")
    for identity, image_records in records.items():
        for image_id, metadata in image_records.items():
            bbox = metadata.get("new_face_crop")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise RuntimeError(f"Invalid face bbox for {identity}/{image_id}: {bbox}")
            face_side = min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
            if minimum_face_side is not None and face_side < float(minimum_face_side):
                raise RuntimeError(
                    f"Face below sealed threshold for {identity}/{image_id}: {face_side}"
                )
            text = str(metadata.get("text") or "")
            if len(trigger.findall(text)) != 1:
                raise RuntimeError(
                    f"Caption trigger count is not one for {identity}/{image_id}"
                )

    flattened = [
        (identity, image_id, metadata)
        for identity, images in records.items()
        for image_id, metadata in images.items()
    ]
    rng = random.Random(20260727)
    sample = rng.sample(flattened, min(args.sample_count, len(flattened)))
    decoded = []
    for identity, image_id, metadata in sample:
        path = args.images_root / identity / f"{image_id}.jpg"
        with Image.open(path) as image:
            image.load()
            if image.size != (1024, 1024):
                raise RuntimeError(f"Unexpected image size for {path}: {image.size}")
        bbox = metadata.get("new_face_crop")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or min(bbox) < 0
            or max(bbox) > 1024
        ):
            raise RuntimeError(f"Invalid face bbox for {path}: {bbox}")
        decoded.append(str(path.relative_to(args.images_root)))

    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "dataset_manifest": str(args.dataset_manifest) if args.dataset_manifest else None,
        "release": release_name,
        "images_root": str(args.images_root),
        "identity_count": len(records),
        "image_count": len(flattened),
        "min_images_per_identity": min(counts.values()),
        "max_images_per_identity": max(counts.values()),
        "decoded_samples": decoded,
        "same_id_distinct_reference_required": True,
        "minimum_face_side": minimum_face_side,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
