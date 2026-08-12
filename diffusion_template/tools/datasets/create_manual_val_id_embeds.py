import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from src.metrics.aligner import Aligner
from src.face_subject_selector import (
    BBOX_OVERLAP_V2,
    LEGACY_FIRST,
    SUPPORTED_POLICIES,
    select_subject_face,
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths(images_dir: Path):
    return sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_declared_bboxes(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        bbox = value.get("face_crop_new") or value.get("face_crop_old")
        if bbox is not None:
            result[Path(key).stem] = [float(item) for item in bbox]
    return result


def main():
    parser = argparse.ArgumentParser(description="Create ID embeddings for manual validation set.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/references"),
        help="Directory with reference images.",
    )
    parser.add_argument(
        "--bbox-json",
        type=Path,
        help="Declared reference-face boxes used by bbox_overlap_v2.",
    )
    parser.add_argument(
        "--selector-policy",
        choices=sorted(SUPPORTED_POLICIES),
        default=LEGACY_FIRST,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional JSON provenance manifest for the selected faces.",
    )
    parser.add_argument(
        "--legacy-embeddings",
        type=Path,
        help="Optional legacy .pth used to assert unchanged identities.",
    )
    parser.add_argument(
        "--allowed-changed-ids",
        default="",
        help="Comma-separated identities allowed to differ from --legacy-embeddings.",
    )
    parser.add_argument("--unchanged-cosine-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../dataset_full/val_dataset/id_embeds_manual_val.pth"),
        help="Path to save the generated embeddings (.pth).",
    )
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    image_paths = collect_image_paths(args.images_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {args.images_dir}")

    aligner = Aligner()
    declared_bboxes = load_declared_bboxes(args.bbox_json)
    if args.selector_policy == BBOX_OVERLAP_V2:
        missing_boxes = sorted(path.stem for path in image_paths if path.stem not in declared_bboxes)
        if missing_boxes:
            raise ValueError(
                f"bbox_overlap_v2 requires a declared bbox for every identity: {missing_boxes}"
            )
    id_embeds = {}
    missing_ids = []
    selections = {}

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        face_bboxes, face_embeds = aligner([img])
        stem = img_path.stem

        if not face_embeds or not face_embeds[0]:
            missing_ids.append(stem)
            continue

        faces = [
            {"bbox": bbox, "embedding": embedding}
            for bbox, embedding in zip(face_bboxes[0], face_embeds[0])
        ]
        selected, audit = select_subject_face(
            faces,
            declared_bbox=declared_bboxes.get(stem),
            policy=args.selector_policy,
        )
        embedding = torch.as_tensor(selected["embedding"]).float()
        id_embeds[stem] = embedding
        selections[stem] = {
            **audit.to_dict(),
            "source_path": str(img_path.resolve()),
            "source_sha256": sha256(img_path),
            "declared_bbox": declared_bboxes.get(stem),
            "embedding_sha256": hashlib.sha256(
                embedding.numpy().astype(np.float32).tobytes()
            ).hexdigest(),
        }

    if missing_ids:
        if args.selector_policy != LEGACY_FIRST:
            raise RuntimeError(
                f"Subject-v2 preflight found no faces for {len(missing_ids)} images: {missing_ids}"
            )
        print(f"Warning: no faces detected for {len(missing_ids)} images: {missing_ids}")

    comparison = {}
    if args.legacy_embeddings is not None:
        legacy = torch.load(args.legacy_embeddings, map_location="cpu")
        allowed = {
            value.strip()
            for value in args.allowed_changed_ids.split(",")
            if value.strip()
        }
        if set(legacy) != set(id_embeds):
            raise ValueError("Legacy and new embedding identity sets differ")
        for identity, embedding in id_embeds.items():
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    embedding.float().reshape(1, -1),
                    torch.as_tensor(legacy[identity]).float().reshape(1, -1),
                ).item()
            )
            comparison[identity] = {
                "legacy_cosine": cosine,
                "allowed_to_change": identity in allowed,
            }
            if identity not in allowed and 1.0 - cosine > args.unchanged_cosine_tolerance:
                raise RuntimeError(
                    f"Unexpected embedding change for {identity}: cosine={cosine:.9f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(id_embeds, args.output)
    print(f"Saved {len(id_embeds)} embeddings to {args.output}")
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "kind": "manual_val_subject_embeddings",
            "selector_policy": args.selector_policy,
            "images_dir": str(args.images_dir.resolve()),
            "bbox_json": None if args.bbox_json is None else str(args.bbox_json.resolve()),
            "bbox_json_sha256": None if args.bbox_json is None else sha256(args.bbox_json),
            "embedding_path": str(args.output.resolve()),
            "embedding_sha256": sha256(args.output),
            "legacy_embedding_path": (
                None
                if args.legacy_embeddings is None
                else str(args.legacy_embeddings.resolve())
            ),
            "legacy_embedding_sha256": (
                None
                if args.legacy_embeddings is None
                else sha256(args.legacy_embeddings)
            ),
            "identities": selections,
            "legacy_comparison": comparison,
        }
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved selector manifest to {args.manifest_output}")


if __name__ == "__main__":
    main()
