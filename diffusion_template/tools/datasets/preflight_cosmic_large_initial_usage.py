#!/usr/bin/env python3
"""Preflight the initial Cosmic Large usage baseline and its policy arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.cosmic_large_initial_usage import (
    CosmicLargeInitialUsageTrain,
)
from src.datasets.reference_policy import valid_bbox


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument(
        "--reference-mode",
        choices=sorted(CosmicLargeInitialUsageTrain.REFERENCE_MODES),
        required=True,
    )
    parser.add_argument("--min-face-res", type=int, required=True)
    parser.add_argument("--topk-temperature", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.reference_mode != "self" and args.candidate_manifest is None:
        raise SystemExit("distinct-reference modes require --candidate-manifest")

    random.seed(args.seed)
    dataset = CosmicLargeInitialUsageTrain(
        cosmic_large_json_pth=str(args.metadata),
        cosmic_large_texts_json_pth=str(args.captions),
        images_path=str(args.images_root),
        candidate_manifest_path=(
            None
            if args.candidate_manifest is None
            else str(args.candidate_manifest)
        ),
        reference_mode=args.reference_mode,
        min_face_res=args.min_face_res,
        reference_crop_margin=None,
        reference_content_size=256,
        reference_canvas_size=None,
        random_horizontal_flip=False,
        random_reference_flip=False,
        topk_temperature=args.topk_temperature,
    )
    count = min(max(1, args.sample_count), len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    samples = []
    errors = []
    for index in indices:
        try:
            sample = dataset[index]
            target = sample["pixel_values"]
            reference = sample["ref_images"][0]
            if target.size != (1024, 1024):
                raise ValueError(f"target size is {target.size}, expected 1024x1024")
            if not valid_bbox(sample["face_bbox"], target.size):
                raise ValueError(f"invalid target bbox {sample['face_bbox']!r}")
            if not valid_bbox(sample["face_bbox_ref"], reference.size):
                raise ValueError(
                    f"invalid reference bbox {sample['face_bbox_ref']!r}"
                )
            samples.append(
                {
                    "index": index,
                    "target_path": sample["target_path"],
                    "reference_path": sample["reference_path"],
                    "self_reference": (
                        sample["target_path"] == sample["reference_path"]
                    ),
                    "target_size": list(target.size),
                    "reference_size": list(reference.size),
                    "prompt_words": len(sample["prompt"].split()),
                    "reference_cache_key": sample["reference_cache_key"],
                }
            )
        except Exception as error:
            errors.append(
                {
                    "index": index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    payload = {
        "schema_version": 1,
        "initial_test_branch_commit": (
            "6782e9d62345fe910633cc8ceec0e2fda6ec2fd1"
        ),
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "captions": str(args.captions.resolve()),
        "captions_sha256": sha256_file(args.captions),
        "candidate_manifest": (
            None
            if args.candidate_manifest is None
            else str(args.candidate_manifest.resolve())
        ),
        "candidate_manifest_sha256": (
            None
            if args.candidate_manifest is None
            else sha256_file(args.candidate_manifest)
        ),
        "images_root": str(args.images_root.resolve()),
        "policy": {
            "reference_mode": args.reference_mode,
            "min_face_res": args.min_face_res,
            "topk_temperature": args.topk_temperature,
            "reference_crop_margin": None,
            "reference_content_size": 256,
            "reference_canvas_size": None,
        },
        "loader_audit": dataset.audit,
        "sample_count": count,
        "sample_errors": errors,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        print(
            f"Cosmic initial-usage preflight failed: "
            f"{len(errors)}/{count} sampled records",
            file=sys.stderr,
        )
        return 1
    print(
        f"Cosmic initial-usage preflight passed: {count}/{count}; "
        f"accepted={len(dataset)}; audit={dataset.audit}"
    )
    print(f"Preflight record: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
