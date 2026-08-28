#!/usr/bin/env python3
"""Fail-closed release preflight for the Big Celebs training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--min-face-res", type=int, default=192)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_bbox(bbox, min_face_res: int) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in bbox
    ):
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (
        0.0 <= x0 < x1 <= 1024.0
        and 0.0 <= y0 < y1 <= 1024.0
        and min(x1 - x0, y1 - y0) >= min_face_res
    )


def safe_component(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def decode_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG" or image.mode != "RGB" or image.size != (1024, 1024):
            raise RuntimeError(
                f"Expected 1024x1024 RGB JPEG at {path}, found "
                f"format={image.format}, mode={image.mode}, size={image.size}"
            )


def main():
    args = parse_args()
    if args.min_face_res < 1 or args.min_face_res > 1024:
        raise ValueError("--min-face-res must be within [1, 1024]")
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")

    expected_digest = args.expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("--expected-sha256 must be a 64-character SHA-256")
    actual_digest = sha256(args.manifest)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch: expected {expected_digest}, "
            f"found {actual_digest}"
        )

    release_name = None
    selected = None
    sealed_paths = None
    if args.dataset_manifest is not None:
        release_root = args.dataset_manifest.parent.resolve()
        if args.manifest.parent.resolve() != release_root:
            raise RuntimeError("Selected manifest is outside the sealed release")
        if args.images_root.resolve() != release_root / "large_dataset":
            raise RuntimeError("Image root does not belong to the sealed release")
        release_digest = sha256(args.dataset_manifest)
        ready_path = args.dataset_manifest.parent / "READY"
        if not ready_path.is_file():
            raise RuntimeError(f"Missing release READY seal: {ready_path}")
        if ready_path.read_text(encoding="utf-8").strip() != release_digest:
            raise RuntimeError("dataset_manifest.json does not match READY")

        release = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        if release.get("state") != "ready_for_training":
            raise RuntimeError(
                f"Release is not ready for training: {release.get('state')}"
            )
        variants = release.get("variants", {})
        default_variant = variants.get(release.get("default_variant"), {})
        if default_variant.get("sha256") != actual_digest:
            raise RuntimeError("Selected manifest is not the sealed default variant")
        audit = release.get("audit", {})
        required_audit_flags = (
            "all_selected_paths_exist",
            "all_selected_images_1024_rgb_jpeg",
            "all_selected_images_fully_decoded",
        )
        if any(audit.get(flag) is not True for flag in required_audit_flags):
            raise RuntimeError("Release image audit is incomplete")
        if audit.get("caption_errors") != 0:
            raise RuntimeError("Release caption audit reports errors")
        validation = release.get("validation", {})
        overlap_fields = (
            "identity_name_overlap",
            "exact_image_overlap",
            "perceptual_image_overlap",
        )
        if any(validation.get(field) != 0 for field in overlap_fields):
            raise RuntimeError("Release overlaps the fixed validation panel")
        selected = next(
            (
                variant
                for variant in variants.values()
                if variant.get("path") == args.manifest.name
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"Selected manifest is absent from release seal: "
                f"{args.manifest.name}"
            )
        if selected.get("sha256") != actual_digest:
            raise RuntimeError("Selected variant hash does not match release seal")
        if int(selected.get("minimum_face_side", -1)) != args.min_face_res:
            raise RuntimeError(
                "Configured min-face policy does not match sealed variant: "
                f"configured={args.min_face_res}, "
                f"sealed={selected.get('minimum_face_side')}"
            )
        sealed_files = release.get("image_files")
        if not isinstance(sealed_files, list) or not sealed_files:
            raise RuntimeError("Release seal has no image_files records")
        sealed_paths = {str(record["path"]) for record in sealed_files}
        if len(sealed_paths) != len(sealed_files):
            raise RuntimeError("Release seal contains duplicate image paths")
        if audit.get("content_seal_files") != len(sealed_paths):
            raise RuntimeError("Release image seal count does not match audit")
        release_name = str(
            release.get("release") or args.dataset_manifest.parent.name
        )

    with args.manifest.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Invalid or empty manifest: {args.manifest}")

    flattened = []
    expected_paths = set()
    trigger_pattern = re.compile(r"(?<!\w)img(?!\w)")
    caption_word_counts = []
    for identity, images in records.items():
        if not isinstance(identity, str) or not safe_component(identity):
            raise ValueError(f"Unsafe identity ID: {identity!r}")
        if not isinstance(images, dict) or len(images) < 2:
            raise ValueError(
                f"Identity {identity!r} does not provide two distinct references"
            )
        for image_id, metadata in images.items():
            if not isinstance(image_id, str) or not safe_component(image_id):
                raise ValueError(f"Unsafe image ID: {image_id!r}")
            if not isinstance(metadata, dict) or set(metadata) != {
                "new_face_crop",
                "text",
            }:
                raise ValueError(
                    f"Unexpected metadata fields for {identity}/{image_id}: "
                    f"{sorted(metadata) if isinstance(metadata, dict) else metadata!r}"
                )
            bbox = metadata["new_face_crop"]
            if not valid_bbox(bbox, args.min_face_res):
                raise ValueError(
                    f"Invalid or undersized bbox for {identity}/{image_id}: {bbox!r}"
                )
            prompt = metadata["text"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing caption for {identity}/{image_id}")
            trigger_count = len(trigger_pattern.findall(prompt))
            if trigger_count != 1:
                raise ValueError(
                    f"Expected one 'img' trigger for {identity}/{image_id}, "
                    f"found {trigger_count}"
                )
            relative_path = f"{identity}/{image_id}.jpg"
            expected_paths.add(relative_path)
            flattened.append((identity, image_id, metadata))
            caption_word_counts.append(len(prompt.split()))

    if selected is not None:
        if len(records) != int(selected["identities"]):
            raise RuntimeError(
                "Selected identity count does not match release seal: "
                f"found={len(records)}, sealed={selected['identities']}"
            )
        if len(flattened) != int(selected["images"]):
            raise RuntimeError(
                "Selected image count does not match release seal: "
                f"found={len(flattened)}, sealed={selected['images']}"
            )

    actual_paths = set()
    for directory, _, filenames in os.walk(args.images_root):
        directory_path = Path(directory)
        for filename in filenames:
            actual_paths.add(
                str((directory_path / filename).relative_to(args.images_root))
            )
    required_tree_paths = sealed_paths if sealed_paths is not None else expected_paths
    missing = required_tree_paths - actual_paths
    extra = actual_paths - required_tree_paths
    if missing or extra:
        raise RuntimeError(
            f"Image-tree mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    unsealed_selected = expected_paths - required_tree_paths
    if unsealed_selected:
        raise RuntimeError(
            f"Selected manifest contains {len(unsealed_selected)} unsealed images"
        )

    rng = random.Random(20260731)
    sample = rng.sample(flattened, min(args.sample_count, len(flattened)))
    decoded_pairs = []
    for identity, image_id, _ in sample:
        reference_id = next(
            candidate for candidate in records[identity] if candidate != image_id
        )
        target_path = args.images_root / identity / f"{image_id}.jpg"
        reference_path = args.images_root / identity / f"{reference_id}.jpg"
        if target_path == reference_path:
            raise RuntimeError(f"Target/reference collision for {target_path}")
        decode_image(target_path)
        decode_image(reference_path)
        decoded_pairs.append(
            {
                "identity_id": identity,
                "target": str(target_path.relative_to(args.images_root)),
                "reference": str(reference_path.relative_to(args.images_root)),
            }
        )

    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": actual_digest,
        "dataset_manifest": (
            str(args.dataset_manifest) if args.dataset_manifest else None
        ),
        "release": release_name,
        "images_root": str(args.images_root),
        "identity_count": len(records),
        "image_count": len(flattened),
        "min_images_per_identity": min(len(images) for images in records.values()),
        "max_images_per_identity": max(len(images) for images in records.values()),
        "min_face_res": args.min_face_res,
        "caption_word_count_min": min(caption_word_counts),
        "caption_word_count_max": max(caption_word_counts),
        "decoded_pairs": decoded_pairs,
        "same_id_distinct_reference_required": True,
        "exactly_one_img_trigger_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
