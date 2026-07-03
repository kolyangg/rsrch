#!/usr/bin/env python
"""Post-hoc identity-similarity report for BA debug outputs.

For each output directory, computes cosine similarity between the InsightFace
embedding of every generated image and its reference image (matched by the
`{prompt[:10]}_{ref_stem}.png` naming that infer.py uses). Also reports the
face-detection rate — heavily corrupted faces often fail detection, which is a
signal by itself.

Usage:
    python scripts/idsim_report.py --refs-dir ../dataset_full/val_dataset/references_two \
        outputs/ba_debug/T0 outputs/ba_debug/T1_gs1 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.photomaker_branched.insightface_package import (  # noqa: E402
    analyze_faces,
    create_face_analyzer,
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def embed(analyzer, path: Path):
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    faces = analyze_faces(analyzer, img)
    if not faces:
        return None
    emb = faces[0]["embedding"].astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dirs", nargs="+", help="output directories to score")
    parser.add_argument("--refs-dir", required=True, help="reference images directory")
    args = parser.parse_args()

    analyzer = create_face_analyzer(
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
        ctx_id=-1,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )

    ref_embs = {}
    for ref_path in sorted(Path(args.refs_dir).iterdir()):
        if ref_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        emb = embed(analyzer, ref_path)
        if emb is None:
            print(f"[warn] no face detected in reference {ref_path.name}")
            continue
        ref_embs[ref_path.stem] = emb
    if not ref_embs:
        raise RuntimeError("No usable reference embeddings")

    summary = []
    for out_dir in args.out_dirs:
        out_path = Path(out_dir)
        pngs = sorted(p for p in out_path.glob("*.png"))
        sims, misses = [], 0
        rows = []
        for png in pngs:
            stem_match = next(
                (s for s in ref_embs if png.stem == s or png.stem.endswith(f"_{s}") or f"_{s}_" in png.stem),
                None,
            )
            if stem_match is None:
                continue
            gen_emb = embed(analyzer, png)
            if gen_emb is None:
                misses += 1
                rows.append((png.name, None))
                continue
            sim = float(np.dot(gen_emb, ref_embs[stem_match]))
            sims.append(sim)
            rows.append((png.name, sim))

        print(f"\n=== {out_dir} ===")
        for name, sim in rows:
            print(f"  {name:45s} {'NO FACE' if sim is None else f'{sim:.4f}'}")
        n = len(rows)
        mean_sim = float(np.mean(sims)) if sims else float("nan")
        print(f"  -> images={n}  detected={n - misses}  no_face={misses}  mean_id_sim={mean_sim:.4f}")
        summary.append((out_dir, n, misses, mean_sim))

    print("\n===== SUMMARY =====")
    print(f"{'dir':40s} {'imgs':>5s} {'no_face':>8s} {'mean_id_sim':>12s}")
    for out_dir, n, misses, mean_sim in summary:
        print(f"{out_dir:40s} {n:5d} {misses:8d} {mean_sim:12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
