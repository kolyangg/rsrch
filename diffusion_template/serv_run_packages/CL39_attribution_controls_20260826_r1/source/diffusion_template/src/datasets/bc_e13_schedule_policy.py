"""Torch-free policy and manifest helpers for sealed BC_E13 schedules."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Iterable


SCHEMA_VERSION = 1
POLICY_VERSION = "bc_e13_dataset_schedule_v1"
SCHEDULE_FIELDS = {
    "schema_version", "schedule_index", "optimizer_step", "source", "phase",
    "identity_id", "target_path", "reference_path", "target_role",
    "reference_tier", "flip_target", "source_manifest_sha256", "target_bbox",
    "reference_bbox", "prompt",
}
DIRECTION_RE = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)
PORTRAIT_RE = re.compile(
    r"\b(?:close[- ]?up|headshot|shoulders up|chest up|portrait)\b",
    re.IGNORECASE,
)
REFERENCE_EXCLUSION_RE = re.compile(
    r"\b(?:other people|other person|another person|two people|group of people|crowd|"
    r"hand|hands|hold|holds|holding|glasses|sunglasses|goggles|eyewear|"
    r"hat|cap|beanie|helmet|run|runs|running|jump|jumps|jumping|dance|dances|"
    r"dancing|ski|skis|skiing|fight|fights|fighting|box|boxing|kickboxing|"
    r"drum|drums|drumming|ride|rides|riding|rush|rushes|rushing)\b",
    re.IGNORECASE,
)
TRIGGER_RE = re.compile(r"(?<!\w)img(?!\w)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in bbox
    ):
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0.0 <= x0 < x1 <= 1024.0 and 0.0 <= y0 < y1 <= 1024.0


def face_side(metadata: dict) -> float:
    x0, y0, x1, y1 = [float(value) for value in metadata["new_face_crop"]]
    return min(x1 - x0, y1 - y0)


def face_area(metadata: dict) -> float:
    x0, y0, x1, y1 = [float(value) for value in metadata["new_face_crop"]]
    return (x1 - x0) * (y1 - y0) / (1024.0 * 1024.0)


def is_directional(prompt: str) -> bool:
    return bool(DIRECTION_RE.search(prompt))


def is_canonical_reference(metadata: dict, min_side: float = 384.0) -> bool:
    prompt = str(metadata.get("text") or "")
    return (
        face_side(metadata) >= float(min_side)
        and bool(PORTRAIT_RE.search(prompt))
        and not REFERENCE_EXCLUSION_RE.search(prompt)
    )


def load_identity_manifest(path: Path) -> dict[str, dict[str, dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Invalid or empty identity manifest: {path}")
    normalized: dict[str, dict[str, dict]] = {}
    for raw_identity, raw_images in records.items():
        identity = str(raw_identity)
        if not isinstance(raw_images, dict) or not raw_images:
            raise ValueError(f"Invalid images for identity {identity!r}")
        images: dict[str, dict] = {}
        for raw_image_id, metadata in raw_images.items():
            image_id = str(raw_image_id)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata for {identity}/{image_id}.jpg")
            bbox = metadata.get("new_face_crop")
            prompt = metadata.get("text")
            if not valid_bbox(bbox):
                raise ValueError(f"Invalid bbox for {identity}/{image_id}.jpg")
            if not isinstance(prompt, str) or len(TRIGGER_RE.findall(prompt)) != 1:
                raise ValueError(
                    f"Expected exactly one lowercase img trigger for "
                    f"{identity}/{image_id}.jpg"
                )
            images[image_id] = metadata
        normalized[identity] = images
    return normalized


def iter_manifest_paths(
    records: dict[str, dict[str, dict]],
) -> Iterable[tuple[str, str, dict]]:
    for identity in sorted(records):
        for image_id in sorted(records[identity]):
            yield identity, f"{identity}/{image_id}.jpg", records[identity][image_id]
