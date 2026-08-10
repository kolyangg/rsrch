#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DATA-03/04: Retained the Cosmic decode/policy preflight so
# reference geometry, prompts, boxes, and cache keys are audited pre-launch.
"""Decode-sample and audit the full-Cosmic experiment loader before training."""

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

from src.datasets.cosmic_large_adapted import (
    CosmicLargeAdaptedTrain,
    TRIGGER_WORD_RE,
)
from src.datasets.reference_policy import valid_bbox


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_optional_float(value: str):
    return None if value.lower() == "null" else float(value)


def parse_optional_int(value: str):
    return None if value.lower() == "null" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-margin", default="0.2")
    parser.add_argument("--content-size", default="256")
    parser.add_argument("--canvas-size", default="null")
    parser.add_argument("--prompt-mode", choices=("legacy", "pose_first"), default="legacy")
    parser.add_argument("--prompt-max-words", default="null")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    dataset = CosmicLargeAdaptedTrain(
        manifest_path=str(args.manifest),
        dataset_root=str(args.dataset_root),
        min_face_res=192,
        reference_crop_margin=parse_optional_float(args.crop_margin),
        reference_content_size=parse_optional_int(args.content_size),
        reference_canvas_size=parse_optional_int(args.canvas_size),
        random_horizontal_flip=False,
        random_reference_flip=False,
        prompt_mode=args.prompt_mode,
        prompt_max_words=parse_optional_int(args.prompt_max_words),
    )
    count = min(max(1, int(args.sample_count)), len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    sampled = []
    errors = []
    for index in indices:
        try:
            sample = dataset[index]
            target = sample["pixel_values"]
            reference = sample["ref_images"][0]
            if sample["target_path"] == sample["reference_path"]:
                raise ValueError("target and reference paths are equal")
            if target.size != (1024, 1024):
                raise ValueError(f"target size is {target.size}, expected 1024x1024")
            if not valid_bbox(sample["face_bbox"], target.size):
                raise ValueError(f"invalid target bbox {sample['face_bbox']!r}")
            if not valid_bbox(sample["face_bbox_ref"], reference.size):
                raise ValueError(f"invalid reference bbox {sample['face_bbox_ref']!r}")
            trigger_count = len(TRIGGER_WORD_RE.findall(sample["prompt"]))
            if trigger_count != 1:
                raise ValueError(
                    "prompt must contain exactly one lowercase PhotoMaker "
                    f"trigger word, found {trigger_count}: {sample['prompt']!r}"
                )
            target_bbox = sample["face_bbox"]
            ref_bbox = sample["face_bbox_ref"]
            sampled.append(
                {
                    "index": index,
                    "identity_id": sample["identity_id"],
                    "target_path": sample["target_path"],
                    "reference_path": sample["reference_path"],
                    "target_size": list(target.size),
                    "reference_size": list(reference.size),
                    "target_face_area_fraction": (
                        (target_bbox[2] - target_bbox[0])
                        * (target_bbox[3] - target_bbox[1])
                        / float(target.width * target.height)
                    ),
                    "reference_face_area_fraction": (
                        (ref_bbox[2] - ref_bbox[0])
                        * (ref_bbox[3] - ref_bbox[1])
                        / float(reference.width * reference.height)
                    ),
                    "prompt_words": len(sample["prompt"].split()),
                    "trigger_count": trigger_count,
                    "reference_cache_key": sample["reference_cache_key"],
                }
            )
        except Exception as error:
            errors.append({"index": index, "error": f"{type(error).__name__}: {error}"})

    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_root": str(args.dataset_root.resolve()),
        "loader_audit": dataset.audit,
        "policy": {
            "crop_margin": parse_optional_float(args.crop_margin),
            "content_size": parse_optional_int(args.content_size),
            "canvas_size": parse_optional_int(args.canvas_size),
            "prompt_mode": args.prompt_mode,
            "prompt_max_words": parse_optional_int(args.prompt_max_words),
        },
        "sample_count": count,
        "sample_errors": errors,
        "samples": sampled,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Cosmic preflight: {args.output}")
    else:
        print(text, end="")
    if errors:
        print(f"Cosmic preflight failed: {len(errors)}/{count} sampled records", file=sys.stderr)
        return 1
    print(
        f"Cosmic preflight passed: {count}/{count} decoded samples; "
        f"{len(dataset)} accepted records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
