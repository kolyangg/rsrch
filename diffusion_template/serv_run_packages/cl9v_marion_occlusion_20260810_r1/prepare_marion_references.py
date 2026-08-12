#!/usr/bin/env python3
"""Create deterministic same-file Marion conditioning transforms on Serv."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


LANDMARKS = np.asarray(
    [
        [327.3774108886719, 240.83578491210938],
        [415.2748107910156, 229.0334014892578],
        [403.93792724609375, 285.9154357910156],
        [340.67352294921875, 341.7558898925781],
        [420.2079772949219, 330.5019226074219],
    ],
    dtype=np.float64,
)
REFERENCE_BBOX = np.asarray([234.0, 109.0, 441.0, 417.0], dtype=np.float64)
ROLL_DEGREES = -7.647632947081888
ARCFACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def transform_bbox(matrix: np.ndarray, bbox: np.ndarray, width: int, height: int):
    x0, y0, x1, y1 = bbox.tolist()
    corners = np.asarray(
        [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]],
        dtype=np.float64,
    )
    mapped = corners @ matrix.T
    return [
        float(np.clip(mapped[:, 0].min(), 0, width)),
        float(np.clip(mapped[:, 1].min(), 0, height)),
        float(np.clip(mapped[:, 0].max(), 0, width)),
        float(np.clip(mapped[:, 1].max(), 0, height)),
    ]


def similarity_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (x, y), (tx, ty) in zip(source, target):
        rows.append([x, -y, 1.0, 0.0])
        values.append(tx)
        rows.append([y, x, 0.0, 1.0])
        values.append(ty)
    a, b, shift_x, shift_y = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )[0]
    return np.asarray(
        [[a, -b, shift_x], [b, a, shift_y]], dtype=np.float64
    )


def redetect(project_root: Path, image: Image.Image, expected_bbox: list[float]):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.metrics.aligner import Aligner

    def iou(first, second):
        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        union += max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
        return inter / union if union > 0 else 0.0

    boxes, _embeddings = Aligner()([image])
    detected = [[float(value) for value in box] for box in (boxes[0] or [])]
    if not detected:
        raise RuntimeError("No face detected after Marion conditioning transform")
    selected = max(detected, key=lambda box: iou(box, expected_bbox))
    return selected, detected, iou(selected, expected_bbox)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(input_path).convert("RGB")
    width, height = original.size
    bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)

    eye_center = tuple(((LANDMARKS[0] + LANDMARKS[1]) / 2.0).tolist())
    roll_matrix = cv2.getRotationMatrix2D(eye_center, ROLL_DEGREES, 1.0)

    canonical_size = 384.0
    offset = np.asarray(
        [(width - canonical_size) / 2.0, (height - canonical_size) / 2.0],
        dtype=np.float64,
    )
    target_landmarks = ARCFACE_TEMPLATE_112 * (canonical_size / 112.0) + offset
    canonical_matrix = similarity_matrix(LANDMARKS, target_landmarks)

    variants = {}
    for name, matrix in (("roll", roll_matrix), ("similarity", canonical_matrix)):
        transformed_bgr = cv2.warpAffine(
            bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        transformed = Image.fromarray(
            cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB)
        )
        image_path = output_dir / f"marion_{name}.png"
        transformed.save(image_path)
        propagated_bbox = transform_bbox(matrix, REFERENCE_BBOX, width, height)
        selected, all_boxes, overlap = redetect(
            args.project_root.resolve(), transformed, propagated_bbox
        )
        variants[name] = {
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "rgb_sha256": rgb_sha256(transformed),
            "affine_matrix": matrix.tolist(),
            "propagated_bbox": propagated_bbox,
            "redetected_bbox": selected,
            "redetected_all_bboxes": all_boxes,
            "redetected_propagated_iou": overlap,
            "output_size": [width, height],
            "edge_fill": "cv2.BORDER_REPLICATE",
        }
    variants["roll"]["rotation_degrees"] = ROLL_DEGREES
    variants["roll"]["rotation_center"] = list(eye_center)
    variants["similarity"]["canonical_square_size"] = canonical_size
    variants["similarity"]["target_landmarks"] = target_landmarks.tolist()

    manifest = {
        "schema_version": 1,
        "kind": "marion_same_file_conditioning_transforms",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_rgb_sha256": rgb_sha256(original),
        "input_size": [width, height],
        "source_bbox": REFERENCE_BBOX.tolist(),
        "source_landmarks": LANDMARKS.tolist(),
        "scoring_reference_policy": "unchanged_original_subject_v2_vector",
        "variants": variants,
    }
    manifest_path = output_dir / "marion_transform_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
