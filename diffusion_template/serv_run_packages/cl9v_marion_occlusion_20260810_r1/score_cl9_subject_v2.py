#!/usr/bin/env python3
"""Score sidecar images with the current mask-owned subject-v2 contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


PROBLEM_FAMILIES = {"Skiing", "Crying", "Dancing", "Jumping"}


def bbox_iou(first, second) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in first]
    bx0, by0, bx1, by1 = [float(value) for value in second]
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def cosine(first, second) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom else 0.0


def stats(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(clean),
        "mean": float(np.mean(clean)) if clean else None,
        "median": float(np.median(clean)) if clean else None,
        "min": float(np.min(clean)) if clean else None,
        "max": float(np.max(clean)) if clean else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subject-embeddings", type=Path, required=True)
    parser.add_argument("--legacy-embeddings", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.metrics.aligner import Aligner

    subject = torch.load(args.subject_embeddings, map_location="cpu")
    legacy = torch.load(args.legacy_embeddings, map_location="cpu")
    rows = json.loads((args.run_dir / "per_image.json").read_text(encoding="utf-8"))
    aligner = Aligner()
    scored = []
    for row in rows:
        path = args.run_dir / "images" / row["filename"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        boxes, embeddings = aligner([image])
        boxes = boxes[0] or []
        embeddings = embeddings[0] or []
        target = row["face_bbox_gen"]
        ranked = sorted(
            (
                (bbox_iou(box, target), index, box, embedding)
                for index, (box, embedding) in enumerate(zip(boxes, embeddings))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        identity = row["identity_id"]
        if not ranked or ranked[0][0] < 0.05:
            id_sim = 0.0
            owned_iou = 0.0 if not ranked else ranked[0][0]
            chosen_bbox = None
            unowned = 1
        else:
            owned_iou, _index, chosen_bbox, chosen_embedding = ranked[0]
            id_sim = cosine(chosen_embedding, subject[identity])
            unowned = 0
        legacy_sim = (
            max(cosine(embedding, legacy[identity]) for embedding in embeddings)
            if embeddings
            else 0.0
        )
        family = str(row["prompt"]).split()[0]
        face_short_side = (
            min(chosen_bbox[2] - chosen_bbox[0], chosen_bbox[3] - chosen_bbox[1])
            if chosen_bbox is not None
            else 0.0
        )
        scored.append(
            {
                "dataset_index": int(row["dataset_index"]),
                "identity": identity,
                "family": family,
                "prompt": row["prompt"],
                "filename": row["filename"],
                "id_sim": id_sim,
                "id_sim_legacy_best": legacy_sim,
                "id_sim_mask_iou": owned_iou,
                "id_sim_face_count": len(boxes),
                "id_sim_no_face": int(not boxes),
                "id_sim_unowned": unowned,
                "face_short_side": float(face_short_side),
            }
        )

    family_names = sorted({row["family"] for row in scored})
    identity_names = sorted({row["identity"] for row in scored})
    summary = {
        "all": stats([row["id_sim"] for row in scored]),
        "families": {
            family: stats([row["id_sim"] for row in scored if row["family"] == family])
            for family in family_names
        },
        "identities": {
            identity: stats([row["id_sim"] for row in scored if row["identity"] == identity])
            for identity in identity_names
        },
        "clean": stats(
            [row["id_sim"] for row in scored if row["family"] not in PROBLEM_FAMILIES]
        ),
        "problem": stats(
            [row["id_sim"] for row in scored if row["family"] in PROBLEM_FAMILIES]
        ),
        "face_detection_rate": float(np.mean([not row["id_sim_no_face"] for row in scored])),
        "mask_iou": stats([row["id_sim_mask_iou"] for row in scored]),
        "face_short_side": stats([row["face_short_side"] for row in scored]),
    }
    output = {
        "schema_version": 1,
        "kind": "manual_val_subject_v2_sidecar_score",
        "subject_embeddings": str(args.subject_embeddings.resolve()),
        "legacy_embeddings": str(args.legacy_embeddings.resolve()),
        "image_count": len(scored),
        "summary": summary,
        "rows": scored,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
