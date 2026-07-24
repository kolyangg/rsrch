#!/usr/bin/env python3
"""Build and seal a small immutable artifact for the identity/data factorial."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 1
CROP_MARGIN = 0.20
OUTPUT_SIZE = 256
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
INTERPOLATION = "PIL.Image.BICUBIC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_ids(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != 8 or len(set(values)) != 8:
        raise argparse.ArgumentTypeError(
            "--training-image-ids must contain exactly eight distinct comma-separated IDs"
        )
    return values


def valid_bbox(bbox, size: tuple[int, int]) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    width, height = size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(
                pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            )
    return value


def hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def deterministic_face_crop(
    image: Image.Image,
    face_bbox,
) -> tuple[Image.Image, list[float], list[int]]:
    """Apply a fixed 20%-per-side square margin and return a 256px crop."""
    image_width, image_height = image.size
    x0, y0, x1, y1 = [float(value) for value in face_bbox]
    face_side = max(x1 - x0, y1 - y0)
    crop_side = min(
        image_width,
        image_height,
        max(1, int(math.ceil(face_side * (1.0 + 2.0 * CROP_MARGIN)))),
    )
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    crop_x0 = int(round(center_x - crop_side / 2.0))
    crop_y0 = int(round(center_y - crop_side / 2.0))
    crop_x0 = min(max(crop_x0, 0), image_width - crop_side)
    crop_y0 = min(max(crop_y0, 0), image_height - crop_side)
    crop_xyxy = [
        crop_x0,
        crop_y0,
        crop_x0 + crop_side,
        crop_y0 + crop_side,
    ]

    cropped = image.crop(tuple(crop_xyxy))
    scale = OUTPUT_SIZE / float(crop_side)
    transformed_bbox = [
        (x0 - crop_x0) * scale,
        (y0 - crop_y0) * scale,
        (x1 - crop_x0) * scale,
        (y1 - crop_y0) * scale,
    ]
    transformed_bbox = [
        max(0.0, min(float(OUTPUT_SIZE), value)) for value in transformed_bbox
    ]
    if not valid_bbox(transformed_bbox, (OUTPUT_SIZE, OUTPUT_SIZE)):
        raise ValueError(f"Invalid derived bbox {transformed_bbox}")
    return (
        cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC),
        transformed_bbox,
        crop_xyxy,
    )


def bbox_iou(left, right) -> float:
    lx0, ly0, lx1, ly1 = [float(value) for value in left]
    rx0, ry0, rx1, ry1 = [float(value) for value in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def run_face_embedding_audit(
    candidates: dict[str, dict],
    *,
    minimum_similarity: float,
) -> dict:
    """Select detections by annotated-bbox IoU and audit ArcFace consistency."""
    import numpy as np

    from src.model.photomaker_branched.insightface_package import (
        analyze_faces,
        create_face_analyzer,
    )

    analyzer = create_face_analyzer(
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
        ctx_id=-1,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )

    embeddings = {}
    per_image = {}
    for image_id, candidate in candidates.items():
        with Image.open(candidate["path"]) as image:
            image_rgb = image.convert("RGB")
            faces = analyze_faces(
                analyzer,
                np.array(image_rgb)[:, :, ::-1],
            )
        if not faces:
            per_image[image_id] = {"status": "rejected", "reason": "no_face"}
            continue
        ranked = sorted(
            (
                (
                    bbox_iou(candidate["face_bbox"], face["bbox"]),
                    face,
                )
                for face in faces
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        overlap, face = ranked[0]
        if overlap <= 0:
            per_image[image_id] = {
                "status": "rejected",
                "reason": "detected_face_does_not_overlap_annotation",
            }
            continue
        embedding = np.asarray(face["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if not math.isfinite(norm) or norm <= 0:
            per_image[image_id] = {
                "status": "rejected",
                "reason": "invalid_embedding",
            }
            continue
        embeddings[image_id] = embedding / norm
        per_image[image_id] = {
            "status": "candidate",
            "annotation_detection_iou": overlap,
            "detected_face_count": len(faces),
        }

    if len(embeddings) < 10:
        raise RuntimeError(
            f"Face audit retained {len(embeddings)} images; at least 10 are required"
        )

    matrix = np.stack(list(embeddings.values()), axis=0)
    medoid_index = int(np.argmax((matrix @ matrix.T).mean(axis=1)))
    medoid_id = list(embeddings)[medoid_index]
    medoid = embeddings[medoid_id]
    accepted = []
    rejected = []
    similarities = []
    for image_id, embedding in embeddings.items():
        similarity = float(embedding @ medoid)
        similarities.append(similarity)
        per_image[image_id]["similarity_to_medoid"] = similarity
        if similarity < minimum_similarity:
            per_image[image_id].update(
                {"status": "rejected", "reason": "low_identity_similarity"}
            )
            rejected.append(image_id)
        else:
            per_image[image_id]["status"] = "accepted"
            accepted.append(image_id)

    return {
        "status": "verified",
        "model": "InsightFace recognition embedding via project face analyzer",
        "face_selection": "maximum IoU with annotated bbox",
        "minimum_similarity": minimum_similarity,
        "medoid_image_id": medoid_id,
        "minimum_observed_similarity": min(similarities),
        "median_observed_similarity": float(np.median(similarities)),
        "accepted_image_ids": accepted,
        "rejected_image_ids": rejected,
        "per_image": per_image,
    }


def audit_candidates(
    identity_records: dict,
    source_image_root: Path,
    identity: str,
    *,
    perceptual_duplicate_distance: int,
) -> tuple[dict[str, dict], dict]:
    candidates = {}
    rejected = {}
    seen_sha = {}
    seen_dhash = {}
    for image_id in sorted(identity_records, key=lambda value: (len(str(value)), str(value))):
        record = identity_records[image_id]
        path = source_image_root / identity / f"{image_id}.jpg"
        if not path.is_file():
            rejected[str(image_id)] = "missing_image"
            continue
        with Image.open(path) as image:
            image_rgb = image.convert("RGB")
            size = image_rgb.size
            bbox = record.get("new_face_crop")
            if size != (1024, 1024):
                rejected[str(image_id)] = f"unexpected_size:{size[0]}x{size[1]}"
                continue
            if not valid_bbox(bbox, size):
                rejected[str(image_id)] = "invalid_face_bbox"
                continue
            image_dhash = difference_hash(image_rgb)
        image_sha = sha256_file(path)
        if image_sha in seen_sha:
            rejected[str(image_id)] = f"exact_duplicate_of:{seen_sha[image_sha]}"
            continue
        near_duplicate = next(
            (
                prior_id
                for prior_id, prior_hash in seen_dhash.items()
                if hamming_distance(image_dhash, prior_hash)
                <= perceptual_duplicate_distance
            ),
            None,
        )
        if near_duplicate is not None:
            rejected[str(image_id)] = f"perceptual_duplicate_of:{near_duplicate}"
            continue
        seen_sha[image_sha] = str(image_id)
        seen_dhash[str(image_id)] = image_dhash
        candidates[str(image_id)] = {
            "path": path,
            "face_bbox": [float(value) for value in bbox],
            "prompt": str(record.get("text") or "A person img"),
            "sha256": image_sha,
            "difference_hash": f"{image_dhash:016x}",
        }

    return candidates, {
        "raw_count": len(identity_records),
        "accepted_before_face_check": len(candidates),
        "rejected_count": len(rejected),
        "reject_reasons": dict(Counter(reason.split(":", 1)[0] for reason in rejected.values())),
        "rejected_images": rejected,
        "perceptual_duplicate_hamming_threshold": perceptual_duplicate_distance,
    }


def build(args) -> Path:
    source_metadata = args.source_metadata.resolve()
    source_image_root = args.source_image_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact directory: {output_root}"
        )

    with source_metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    identity_records = metadata.get(args.identity)
    if not isinstance(identity_records, dict):
        raise KeyError(f"Identity {args.identity!r} not found in {source_metadata}")

    candidates, candidate_audit = audit_candidates(
        identity_records,
        source_image_root,
        args.identity,
        perceptual_duplicate_distance=args.perceptual_duplicate_distance,
    )

    if args.skip_face_embedding_check:
        face_audit = {
            "status": "skipped",
            "reason": "explicit --skip-face-embedding-check",
            "accepted_image_ids": sorted(candidates),
            "rejected_image_ids": [],
        }
    else:
        face_audit = run_face_embedding_audit(
            candidates,
            minimum_similarity=args.minimum_face_similarity,
        )

    selected = (
        list(args.training_image_ids)
        + [args.recurring_validation_id, args.final_holdout_id]
    )
    if len(set(selected)) != 10:
        raise ValueError("Training, recurring validation, and final holdout IDs must differ")
    accepted_after_face = set(face_audit["accepted_image_ids"])
    unavailable = [
        image_id
        for image_id in selected
        if image_id not in candidates or image_id not in accepted_after_face
    ]
    if unavailable:
        raise ValueError(
            "Selected IDs failed duplicate/bbox/embedding checks: "
            + ", ".join(unavailable)
        )

    prompts = [
        line.strip()
        for line in args.validation_prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 12:
        raise ValueError(
            f"Expected exactly 12 recurring validation prompts, got {len(prompts)}"
        )
    generation_bboxes = {}
    named_generation_bboxes = {}
    if args.generation_bboxes is not None:
        with args.generation_bboxes.open("r", encoding="utf-8") as handle:
            generation_bboxes = json.load(handle)
        if not isinstance(generation_bboxes, dict) or len(generation_bboxes) != 12:
            raise ValueError(
                "--generation-bboxes must contain exactly 12 fixed validation entries"
            )
        for key, record in generation_bboxes.items():
            bbox = record.get("face_crop_new") if isinstance(record, dict) else None
            if bbox is None and isinstance(record, dict):
                bbox = record.get("face_crop_old")
            if not valid_bbox(bbox, (1024, 1024)):
                raise ValueError(f"Invalid generation bbox for {key!r}: {bbox!r}")
        ordered_bbox_records = [
            generation_bboxes[key]
            for key in sorted(
                generation_bboxes,
                key=lambda key: int(Path(str(key)).stem),
            )
        ]
        resolved_validation_prompts = [
            prompt.replace("<class>", f"{args.class_name} img")
            for prompt in prompts
        ]
        named_generation_bboxes = {
            f"{prompt[:10]}_{args.recurring_validation_id}.png": record
            for prompt, record in zip(
                resolved_validation_prompts,
                ordered_bbox_records,
            )
        }
        if len(named_generation_bboxes) != 12:
            raise ValueError(
                "Resolved validation prompts do not produce 12 unique bbox cache keys"
            )

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(parent))
    )
    try:
        image_manifest = {}
        derived_manifest = {}
        for image_id in args.training_image_ids:
            candidate = candidates[image_id]
            destination = temporary / "images" / args.identity / f"{image_id}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate["path"], destination)
            artifact_sha = sha256_file(destination)
            image_manifest[image_id] = {
                "artifact_path": destination.relative_to(temporary).as_posix(),
                "artifact_sha256": artifact_sha,
                "source_relative_path": f"{args.identity}/{image_id}.jpg",
                "source_sha256": candidate["sha256"],
                "face_bbox": candidate["face_bbox"],
                "prompt": candidate["prompt"],
                "difference_hash": candidate["difference_hash"],
            }

            with Image.open(candidate["path"]) as image:
                derived_image, derived_bbox, crop_xyxy = deterministic_face_crop(
                    image.convert("RGB"),
                    candidate["face_bbox"],
                )
            derived_destination = (
                temporary
                / "derived_references"
                / "cosmic_256"
                / args.identity
                / f"{image_id}_margin20_bicubic_q95.jpg"
            )
            derived_destination.parent.mkdir(parents=True, exist_ok=True)
            derived_image.save(
                derived_destination,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=JPEG_SUBSAMPLING,
                optimize=False,
            )
            derived_sha = sha256_file(derived_destination)
            cache_payload = {
                "source_sha256": candidate["sha256"],
                "crop_xyxy": crop_xyxy,
                "interpolation": INTERPOLATION,
                "output_sha256": derived_sha,
            }
            cache_digest = hashlib.sha256(
                json.dumps(cache_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            derived_manifest[image_id] = {
                "source_image_id": image_id,
                "path": derived_destination.relative_to(temporary).as_posix(),
                "sha256": derived_sha,
                "face_bbox": derived_bbox,
                "crop_xyxy": crop_xyxy,
                "crop_margin_per_side": CROP_MARGIN,
                "interpolation": INTERPOLATION,
                "jpeg_quality": JPEG_QUALITY,
                "jpeg_subsampling": JPEG_SUBSAMPLING,
                "cache_key": f"controlled-cosmic256:{cache_digest}",
            }

        recurring = candidates[args.recurring_validation_id]
        recurring_destination = (
            temporary
            / "validation_refs"
            / f"{args.recurring_validation_id}.jpg"
        )
        recurring_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recurring["path"], recurring_destination)

        final_holdout = candidates[args.final_holdout_id]
        final_destination = (
            temporary / "final_holdout" / f"{args.final_holdout_id}.jpg"
        )
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_holdout["path"], final_destination)

        (temporary / "validation_prompts.txt").write_text(
            "\n".join(prompts) + "\n",
            encoding="utf-8",
        )
        write_json(
            temporary / "classes_ref.json",
            {args.recurring_validation_id: args.class_name},
        )
        write_json(
            temporary / "reference_bboxes.json",
            {
                f"{args.recurring_validation_id}.jpg": {
                    "face_crop_old": recurring["face_bbox"],
                    "face_crop_new": recurring["face_bbox"],
                    "body_crop": [0, 0, 1024, 1024],
                }
            },
        )
        # An empty map deliberately marks a preflight artifact. A promoted
        # factorial artifact must be rebuilt with --generation-bboxes so every
        # arm consumes the same inspected and hashed 12-image package.
        write_json(
            temporary / "photomaker_generated_bboxes.json",
            generation_bboxes,
        )
        write_json(
            temporary / "photomaker_generated_bboxes_auto.json",
            named_generation_bboxes,
        )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_version": args.artifact_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": args.identity,
            "source": {
                "metadata_path": str(source_metadata),
                "metadata_sha256": sha256_file(source_metadata),
                "image_root": str(source_image_root),
            },
            "selection": {
                "selection_method": (
                    "explicit_ids_after_visual_duplicate_bbox_and_embedding_review"
                ),
                "selection_seed": args.selection_seed,
                "training_image_ids": list(args.training_image_ids),
                "single_target_image_id": args.training_image_ids[0],
                "recurring_validation_image_id": args.recurring_validation_id,
                "final_holdout_image_id": args.final_holdout_id,
                "target_modes": ["multi", "single"],
                "reference_modes": ["full_scene", "cosmic_256"],
            },
            "candidate_audit": candidate_audit,
            "face_embedding_audit": face_audit,
            "images": image_manifest,
            "derived_references": {"cosmic_256": derived_manifest},
            "validation": {
                "reference_path": recurring_destination.relative_to(temporary).as_posix(),
                "reference_sha256": sha256_file(recurring_destination),
                "reference_face_bbox": recurring["face_bbox"],
                "prompts_path": "validation_prompts.txt",
                "prompts_sha256": sha256_file(
                    temporary / "validation_prompts.txt"
                ),
                "classes_path": "classes_ref.json",
                "classes_sha256": sha256_file(
                    temporary / "classes_ref.json"
                ),
                "reference_bboxes_path": "reference_bboxes.json",
                "reference_bboxes_sha256": sha256_file(
                    temporary / "reference_bboxes.json"
                ),
                "generation_bboxes_path": "photomaker_generated_bboxes.json",
                "generation_bboxes_sha256": sha256_file(
                    temporary / "photomaker_generated_bboxes.json"
                ),
                "generation_bbox_cache_path": (
                    "photomaker_generated_bboxes_auto.json"
                ),
                "generation_bbox_cache_sha256": sha256_file(
                    temporary / "photomaker_generated_bboxes_auto.json"
                ),
                "generation_bboxes_status": (
                    "sealed"
                    if generation_bboxes and named_generation_bboxes
                    else "unsealed_requires_photomaker_preflight"
                ),
            },
            "final_holdout": {
                "path": final_destination.relative_to(temporary).as_posix(),
                "sha256": sha256_file(final_destination),
                "face_bbox": final_holdout["face_bbox"],
            },
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest_path, manifest)
        (temporary / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n",
            encoding="utf-8",
        )

        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return output_root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the controlled identity factorial artifact"
    )
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-image-ids", type=parse_ids, required=True)
    parser.add_argument("--recurring-validation-id", required=True)
    parser.add_argument("--final-holdout-id", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--validation-prompts", type=Path, required=True)
    parser.add_argument(
        "--generation-bboxes",
        type=Path,
        help="Inspected 12-entry PhotoMaker bbox JSON used to seal a final artifact.",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--minimum-face-similarity", type=float, default=0.25)
    parser.add_argument("--perceptual-duplicate-distance", type=int, default=2)
    parser.add_argument(
        "--skip-face-embedding-check",
        action="store_true",
        help="For code-path smoke checks only; final artifacts must run the face audit.",
    )
    args = parser.parse_args()
    try:
        manifest_path = build(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(manifest_path)


if __name__ == "__main__":
    main()
