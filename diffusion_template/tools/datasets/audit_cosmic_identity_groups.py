#!/usr/bin/env python3
"""Visual + numeric audit of the Cosmic identity grouping used by CL1.

The grouping decides which 1024px target becomes another target's reference, so a
wrong-identity component would poison CL1 exactly where it is meant to be clean.
This samples within-group pairs stratified by cosine band, recomputes each pair's
similarity through the same frozen ArcFace graph, and renders a labelled contact
sheet so the pairs can actually be looked at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.datasets.build_cosmic_identity_assets import (  # noqa: E402
    ARCFACE_SHA256,
    DEFAULT_ARCFACE,
    load_manifest,
    load_target,
)

BANDS = ((0.00, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))
TILE = 128


def face_crop(root: Path, target_path: str, record: dict, size: int) -> Image.Image:
    image = load_target(root, target_path, record)
    x0, y0, x1, y1 = [int(round(float(v))) for v in record["face_crop_new"]]
    return image.crop((x0, y0, x1, y1)).resize((size, size), Image.BILINEAR)


def embed(model, crops, device, torch):
    arrays = [(np.asarray(c.resize((112, 112), Image.BILINEAR), dtype=np.float32) - 127.5) / 127.5
              for c in crops]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        out = model(tensor).float()
        return (out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-band", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-identities", type=int, default=0,
                        help="0 = audit every grouped identity exhaustively")
    parser.add_argument("--arcface-onnx", default=DEFAULT_ARCFACE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from src.model.photomaker_branched.arcface_identity_aux import FrozenOnnxArcFace

    payload = json.loads(Path(args.groups).read_text(encoding="utf-8"))
    groups = payload["groups"]
    records = load_manifest(args.manifest)
    root = Path(args.dataset_root)
    model = FrozenOnnxArcFace(
        model_path=str(Path(args.arcface_onnx).expanduser()),
        expected_sha256=ARCFACE_SHA256,
    ).to(args.device).eval()

    rng = random.Random(args.seed)
    sizes = [len(m) for m in groups.values()]
    # Sample one within-group pair per selected identity.
    identities = [k for k, m in groups.items() if len(m) >= 2]
    rng.shuffle(identities)
    wanted = args.pairs_per_band * len(BANDS)
    candidates = (
        identities
        if args.max_identities <= 0
        else identities[: max(wanted * 6, args.max_identities)]
    )

    pairs = []
    for identity in candidates:
        members = groups[identity]
        a, b = rng.sample(members, 2)
        try:
            crop_a = face_crop(root, a, records[a], TILE)
            crop_b = face_crop(root, b, records[b], TILE)
        except Exception:
            continue
        vectors = embed(model, [crop_a, crop_b], args.device, torch)
        pairs.append({
            "identity": identity, "a": a, "b": b,
            "cosine": float(vectors[0] @ vectors[1]),
            "crops": (crop_a, crop_b),
        })

    by_band = {f"{lo:.2f}-{hi:.2f}": [] for lo, hi in BANDS}
    for pair in pairs:
        for lo, hi in BANDS:
            if lo <= pair["cosine"] < hi:
                by_band[f"{lo:.2f}-{hi:.2f}"].append(pair)
                break

    selected = []
    for key, items in by_band.items():
        items.sort(key=lambda p: p["cosine"])
        selected.extend(items[: args.pairs_per_band])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if selected:
        cols, rows = 6, (len(selected) + 5) // 6
        sheet = Image.new("RGB", (cols * (TILE * 2 + 8), rows * (TILE + 16)), (26, 26, 30))
        draw = ImageDraw.Draw(sheet)
        for index, pair in enumerate(selected):
            cx = (index % cols) * (TILE * 2 + 8)
            cy = (index // cols) * (TILE + 16)
            sheet.paste(pair["crops"][0], (cx, cy + 16))
            sheet.paste(pair["crops"][1], (cx + TILE, cy + 16))
            draw.text((cx + 3, cy + 3), f"cos {pair['cosine']:.3f}", fill=(235, 235, 240))
        sheet_path = out_dir / "identity_pair_contact_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = None

    report = {
        "groups_file": str(args.groups),
        "identities": len(groups),
        "targets_in_groups": int(sum(sizes)),
        "group_size_histogram": {str(s): sizes.count(s) for s in sorted(set(sizes))},
        "pairs_sampled": len(pairs),
        "cosine_by_band": {k: len(v) for k, v in by_band.items()},
        "cosine_percentiles": {
            "p10": float(np.percentile([p["cosine"] for p in pairs], 10)) if pairs else None,
            "median": float(np.median([p["cosine"] for p in pairs])) if pairs else None,
            "p90": float(np.percentile([p["cosine"] for p in pairs], 90)) if pairs else None,
        },
        "near_duplicate_pairs_ge_0.95": sum(1 for p in pairs if p["cosine"] >= 0.95),
        "near_duplicate_pairs_ge_0.98": sum(1 for p in pairs if p["cosine"] >= 0.98),
        "contact_sheet": str(sheet_path) if sheet_path else None,
        "note": ("Cosine is recomputed on the unaligned supplied face box through the same "
                 "frozen w600k_r50 graph used for grouping, so it is directly comparable to "
                 "the grouping threshold. Low-band pairs are the ones to inspect visually."),
    }
    (out_dir / "identity_group_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
