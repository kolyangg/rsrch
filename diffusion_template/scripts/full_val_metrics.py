#!/usr/bin/env python
"""Compute identity-similarity metrics for one full-validation run and append to a shared JSON.

For every generated image in --out-dir, matches it to its reference identity (the last
`_`-separated token of the filename, e.g. `Reading pa_jensen.png` -> `jensen`), computes the
InsightFace cosine similarity of the generated face vs the reference image, and records the mean
(overall and per-identity) plus the face-detection rate. Same method used throughout the project's
analysis, so numbers are directly comparable.

Usage:
  python scripts/full_val_metrics.py --out-dir full_validation_results/ba_saonly_N11 \
      --refs-dir ../dataset_full/val_dataset/references --run ba_saonly_N11 \
      --epoch 3 --step 3000 --json full_validation_results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.photomaker_branched.insightface_package import (  # noqa: E402
    analyze_faces,
    create_face_analyzer,
)

IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def embed(analyzer, path: Path):
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    faces = analyze_faces(analyzer, img)
    if not faces:
        return None
    e = faces[0]["embedding"].astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-8)


def identity_of(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--checkpoint", default="")
    args = ap.parse_args()

    analyzer = create_face_analyzer(
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
        ctx_id=-1, det_size=(640, 640), fallback_ctx_id=-1, quiet=True,
    )

    # Reference embedding per identity (by file stem).
    ref_emb = {}
    for p in sorted(Path(args.refs_dir).iterdir()):
        if p.suffix.lower() in IMG_SUFFIXES:
            ref_emb[p.stem] = embed(analyzer, p)

    out_dir = Path(args.out_dir)
    imgs = sorted(p for p in out_dir.glob("*.png") if not p.name.startswith("_"))

    per_image = {}
    per_id_vals: dict[str, list] = {}
    detected = 0
    for img in imgs:
        ident = identity_of(img.name)
        ref = ref_emb.get(ident)
        e = embed(analyzer, img)
        if e is None:
            per_image[img.name] = None  # face not detected (corrupted)
            per_id_vals.setdefault(ident, [])
            continue
        detected += 1
        if ref is None:
            per_image[img.name] = None
            continue
        sim = float(np.dot(e, ref))
        per_image[img.name] = round(sim, 4)
        per_id_vals.setdefault(ident, []).append(sim)

    valid = [v for v in per_image.values() if v is not None]
    per_identity = {
        k: round(float(np.mean(v)), 4) if v else None for k, v in sorted(per_id_vals.items())
    }
    record = {
        "epoch": args.epoch,
        "step": args.step,
        "checkpoint": args.checkpoint,
        "n_images": len(imgs),
        "n_faces_detected": detected,
        "detection_rate": round(detected / max(1, len(imgs)), 4),
        "mean_id_sim": round(float(np.mean(valid)), 4) if valid else None,
        "per_identity_id_sim": per_identity,
        "per_image_id_sim": per_image,
    }

    json_path = Path(args.json)
    data = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            data = {}
    data[args.run] = record
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2))

    print(f"[metrics] {args.run}: epoch={args.epoch} step={args.step} "
          f"mean_id_sim={record['mean_id_sim']} det={detected}/{len(imgs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
