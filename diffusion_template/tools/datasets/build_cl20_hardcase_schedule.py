#!/usr/bin/env python3
"""Build the sealed 48k-row CL20 Cosmic/BigCelebs curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from src.datasets.bc_e13_schedule_policy import load_identity_manifest
from src.datasets.cosmic_large_adapted import build_cosmic_prompt


OCCLUSION_WORDS = re.compile(
    r"\b(glass|goggle|helmet|hair|hand|cry|tear|ski|mask|scarf)\w*\b", re.I
)
ACTION_WORDS = re.compile(
    r"\b(jump|dance|kick|run|sport|fight|ride|climb|swim|action)\w*\b", re.I
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox(mapping, path: str):
    if isinstance(mapping, list):
        return None
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(path)
    if value is not None:
        return value
    candidate = Path(path)
    for key in (candidate.name, candidate.stem):
        if key in mapping:
            return mapping[key]
    return None


def valid_box(value) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    x0, y0, x1, y1 = [float(item) for item in value]
    return 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024


def cosmic_pool(manifest: dict) -> list[dict]:
    pool = []
    for target_path, record in manifest.items():
        if not isinstance(record, dict):
            continue
        target_box = record.get("face_crop_new")
        if not valid_box(target_box):
            continue
        if min(target_box[2] - target_box[0], target_box[3] - target_box[1]) < 192:
            continue
        candidates = []
        paths = record.get("face_paths") or []
        boxes = record.get("face_bboxes") or {}
        for index, reference_path in enumerate(paths):
            reference_box = (
                boxes[index]
                if isinstance(boxes, list) and index < len(boxes)
                else bbox(boxes, reference_path)
            )
            if (
                reference_path != target_path
                and valid_box(reference_box)
            ):
                candidates.append((str(reference_path), reference_box))
        if not candidates:
            continue
        pool.append(
            {
                "source": "cosmic",
                "identity_id": str(
                    record.get("identity_id")
                    or record.get("person_id")
                    or Path(candidates[0][0]).parent
                ),
                "target_path": str(target_path),
                "target_bbox": target_box,
                "target_body_crop": record.get("body_crop"),
                "prompt": build_cosmic_prompt(record, "pose_first", 50),
                "references": candidates,
            }
        )
    if not pool:
        raise RuntimeError("No eligible Cosmic records")
    return pool


def big_pool(manifest: Path) -> tuple[list[dict], dict[str, int]]:
    records = load_identity_manifest(manifest)
    eligible = []
    depth_histogram = {}
    for identity, images in records.items():
        if not isinstance(images, dict) or len(images) < 6:
            continue
        valid = [
            (f"{identity}/{image_id}.jpg", metadata)
            for image_id, metadata in images.items()
            if isinstance(metadata, dict)
            and valid_box(metadata.get("new_face_crop"))
            and isinstance(metadata.get("text"), str)
            and metadata["text"].strip()
        ]
        if len(valid) < 6:
            continue
        depth_histogram[str(min(20, len(valid)))] = (
            depth_histogram.get(str(min(20, len(valid))), 0) + 1
        )
        for target_path, metadata in valid:
            eligible.append(
                {
                    "source": "big_celebs",
                    "identity_id": str(identity),
                    "target_path": target_path,
                    "target_bbox": metadata["new_face_crop"],
                    "target_body_crop": None,
                    "prompt": metadata["text"],
                    "references": [
                        (path, other["new_face_crop"])
                        for path, other in valid
                        if path != target_path
                    ],
                }
            )
    if not eligible:
        raise RuntimeError("No BigCelebs identities have depth >= 6")
    return eligible, depth_histogram


def choose_big_pools(records: list[dict]) -> list[list[dict]]:
    occlusion = [record for record in records if OCCLUSION_WORDS.search(record["prompt"])]
    action = [record for record in records if ACTION_WORDS.search(record["prompt"])]
    return [records, occlusion or records, action or records]


def build_row(record: dict, rng: random.Random, index: int, *, small: bool) -> dict:
    reference_path, reference_bbox = rng.choice(record["references"])
    target_scale = 1.0
    if small:
        target_short = min(
            float(record["target_bbox"][2]) - float(record["target_bbox"][0]),
            float(record["target_bbox"][3]) - float(record["target_bbox"][1]),
        )
        desired = rng.uniform(96.0, 180.0)
        target_scale = max(0.35, min(0.90, desired / target_short))
    return {
        "index": index,
        "optimizer_step": index // 2,
        "source": record["source"],
        "identity_id": record["identity_id"],
        "target_path": record["target_path"],
        "reference_path": reference_path,
        "target_bbox": record["target_bbox"],
        "target_body_crop": record["target_body_crop"],
        "reference_bbox": reference_bbox,
        "prompt": record["prompt"],
        "target_scale": round(target_scale, 8),
        "reference_face_fraction": round(rng.uniform(0.06, 0.30), 8),
        "reference_offset": [
            round(rng.uniform(-0.15, 0.15), 8),
            round(rng.uniform(-0.15, 0.15), 8),
        ],
        "flip_target": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmic-manifest", required=True)
    parser.add_argument("--cosmic-root", required=True)
    parser.add_argument("--big-manifest", required=True)
    parser.add_argument("--big-images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--seed", type=int, default=200020)
    args = parser.parse_args()

    cosmic_manifest = Path(args.cosmic_manifest)
    big_manifest = Path(args.big_manifest)
    cosmic_root = Path(args.cosmic_root)
    big_root = Path(args.big_images_root)
    for path in (cosmic_manifest, big_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (cosmic_root, big_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    cosmic = cosmic_pool(json.loads(cosmic_manifest.read_text(encoding="utf-8")))
    big, depth_histogram = big_pool(big_manifest)
    big_strata = choose_big_pools(big)
    rng = random.Random(args.seed)
    rng.shuffle(cosmic)
    for stratum in big_strata:
        rng.shuffle(stratum)

    rows = []
    cosmic_index = 0
    big_counts = [0, 0, 0]
    for index in range(48000):
        optimizer_step = index // 2
        use_big = optimizer_step < 20000 and index % 5 == 0
        if use_big:
            stratum_index = sum(big_counts) % 3
            stratum = big_strata[stratum_index]
            record = stratum[big_counts[stratum_index] % len(stratum)]
            big_counts[stratum_index] += 1
            row = build_row(
                record,
                rng,
                index,
                small=stratum_index == 0,
            )
        else:
            record = cosmic[cosmic_index % len(cosmic)]
            cosmic_index += 1
            row = build_row(record, rng, index, small=False)
        rows.append(row)

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    schedule_hash = sha256(output)
    counts = {
        "cosmic": sum(row["source"] == "cosmic" for row in rows),
        "big_celebs": sum(row["source"] == "big_celebs" for row in rows),
        "big_small_synthetic": sum(
            row["source"] == "big_celebs" and row["target_scale"] < 0.999
            for row in rows
        ),
    }
    summary = {
        "kind": "cl20_hardcase_curriculum_v1",
        "seed": args.seed,
        "rows": len(rows),
        "optimizer_steps": 24000,
        "schedule_sha256": schedule_hash,
        "source_sha256": {
            "cosmic": sha256(cosmic_manifest),
            "big_celebs": sha256(big_manifest),
        },
        "source_roots": {
            "cosmic": str(cosmic_root),
            "big_celebs": str(big_root),
        },
        "counts": counts,
        "big_strata_rows": {
            "small_synthetic": big_counts[0],
            "occlusion_caption": big_counts[1],
            "action_caption": big_counts[2],
        },
        "big_identity_depth_histogram_capped20": depth_histogram,
        "contract": "80/20 rows through step 19999; Cosmic-only steps 20000-23999",
    }
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
