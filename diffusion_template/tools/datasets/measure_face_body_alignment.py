#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DOC-01: Retained fixed-panel face/body alignment
# measurement as offline analysis tooling, separate from the training path.
"""Measure face/body misalignment in full-96 validation output.

The branched face route writes reference-derived content into a fixed mask box.
All 96 generation boxes are pinned in the protocol's sealed ``*_auto.json`` and
reused at every validation step. If a run's learned composition places the head
somewhere else, the branch paints a face at the stale location and the head
detaches from the shoulders.

This detects the actual face in each generated image and compares it with the
cached box the mask used, giving three per-image numbers:

  center_offset_norm  distance between detected and mask centres, in units of
                      mask width - the direct "is the face where the mask is"
                      measure
  size_ratio          detected face short side / mask short side - catches the
                      oversized-face signature of the 2.1x Cosmic reference
  iou                 overlap of detected face and mask box

A high center_offset with a healthy id_sim is exactly the reported failure:
a good-looking face in the wrong place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.photomaker_branched.insightface_package import (  # noqa: E402
    analyze_faces,
    create_face_analyzer,
)


def normalize_key(name: str) -> str:
    """Match bbox keys to exported filenames.

    # 07 Aug 2026 - The trainer builds bbox keys from the raw prompt
    # (`prompt[:10]_id.png`), so they contain spaces, while exported PNGs use
    # filesystem-safe underscores. Comparing them literally silently matches
    # only the minority of prompts whose first ten characters have no space.
    """
    return Path(name).name.replace(" ", "_")


def load_mask_boxes(path: str) -> dict[str, list[float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, value in raw.items():
        box = None
        if isinstance(value, dict):
            box = value.get("face_crop_new") or value.get("face_crop_old")
        elif isinstance(value, list):
            box = value
        if box:
            out[normalize_key(key)] = [float(v) for v in box]
    return out


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--mask-boxes", required=True)
    parser.add_argument("--id-sim-csv", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    boxes = load_mask_boxes(args.mask_boxes)
    analyzer = create_face_analyzer(
        providers=["CPUExecutionProvider"], allowed_modules=["detection"],
        ctx_id=-1, det_size=(640, 640), fallback_ctx_id=-1, quiet=True,
    )

    id_sim = {}
    if args.id_sim_csv and Path(args.id_sim_csv).exists():
        import csv
        for row in csv.DictReader(Path(args.id_sim_csv).open()):
            id_sim[row["output_key"]] = float(row["id_sim"])

    rows = []
    for path in sorted(Path(args.images_dir).glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} and path.is_file():
            key = path.name + ".png"
        elif path.is_file():
            key = path.name
        else:
            continue
        mask = boxes.get(normalize_key(key)) or boxes.get(normalize_key(path.name))
        if mask is None:
            continue
        image = Image.open(path).convert("RGB")
        faces = analyze_faces(analyzer, np.array(image)[:, :, ::-1])
        mw, mh = mask[2] - mask[0], mask[3] - mask[1]
        mcx, mcy = (mask[0] + mask[2]) / 2, (mask[1] + mask[3]) / 2
        if not faces:
            rows.append({"key": key, "detected": False, "center_offset_norm": None,
                         "size_ratio": None, "iou": 0.0, "id_sim": id_sim.get(key)})
            continue
        # the detection closest to the mask is the one the branch was meant to write
        best = min(faces, key=lambda f: (
            ((f["bbox"][0] + f["bbox"][2]) / 2 - mcx) ** 2
            + ((f["bbox"][1] + f["bbox"][3]) / 2 - mcy) ** 2))
        b = [float(v) for v in best["bbox"]]
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        rows.append({
            "key": key, "detected": True,
            "center_offset_norm": float(np.hypot(bcx - mcx, bcy - mcy) / max(mw, 1.0)),
            "dx_norm": float((bcx - mcx) / max(mw, 1.0)),
            "dy_norm": float((bcy - mcy) / max(mh, 1.0)),
            "size_ratio": float(min(b[2] - b[0], b[3] - b[1]) / max(min(mw, mh), 1.0)),
            "iou": float(iou(b, mask)),
            "faces_detected": len(faces),
            "id_sim": id_sim.get(key),
        })

    det = [r for r in rows if r["detected"]]
    offs = [r["center_offset_norm"] for r in det]
    report = {
        "label": args.label, "images": len(rows), "detected": len(det),
        "no_face": len(rows) - len(det),
        "center_offset_norm": {
            "median": float(np.median(offs)) if offs else None,
            "p90": float(np.percentile(offs, 90)) if offs else None,
            "over_0.25": sum(1 for o in offs if o > 0.25),
            "over_0.50": sum(1 for o in offs if o > 0.50),
        },
        "size_ratio_median": float(np.median([r["size_ratio"] for r in det])) if det else None,
        "iou_median": float(np.median([r["iou"] for r in det])) if det else None,
        "iou_below_0.3": sum(1 for r in det if r["iou"] < 0.3),
        "misaligned_but_good_id": [
            r["key"] for r in det
            if r["center_offset_norm"] > 0.25 and (r["id_sim"] or 0) > 0.3
        ],
        "worst": sorted(det, key=lambda r: -r["center_offset_norm"])[:10],
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
