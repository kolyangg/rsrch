#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

import sys
from pathlib import Path as _P
# Ensure we can import diffusion_template/src as 'src'
sys.path.append(str(_P(__file__).resolve().parents[1]))

# Reuse in-repo metric utilities for exact behavior
# (id similarity via insightface embeddings; text similarity via CLIP logits)
from src.metrics.text_sim import TextSimMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_prompts(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_classes(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def detect_ref_id_from_name(name: str, ref_ids: List[str]) -> Optional[str]:
    """Pick ref id that is a suffix in filename stem.

    Accept separators before id: space / underscore / hyphen or none.
    Prefer the longest matching id (if there are collisions).
    """
    base = Path(name).stem
    low = base.lower()
    match = None
    best_len = -1
    for rid in ref_ids:
        rid_l = rid.lower()
        if low.endswith(rid_l) or low.endswith("_" + rid_l) or low.endswith("-" + rid_l) or low.endswith(" " + rid_l):
            if len(rid_l) > best_len:
                match = rid
                best_len = len(rid_l)
    return match


def split_prefix_and_ref(name: str, ref_id: str) -> str:
    """Return filename prefix before the ref suffix.

    Example: "Angry man _eddie.png" -> "Angry man " (for ref_id="eddie").
    """
    stem = Path(name).stem
    low = stem.lower()
    rid = ref_id.lower()
    # find last occurrence of id (with or without a common separator)
    for sep in ("_", "-", " ", ""):
        suffix = sep + rid
        if low.endswith(suffix):
            return stem[: -(len(suffix))]
    # fallback
    return stem


def find_full_prompt(prefix: str, prompts: List[str], person_class: str) -> Tuple[Optional[str], Optional[int]]:
    """Return the matching full prompt (after <class> substitution) and its index.

    Match on prefix being the beginning of the full prompt (case/underscore-insensitive).
    Includes a fallback that drops trailing 1–2 letter fragments like the stray "i"
    in filenames such as "Chef man i_eddie.png".
    """
    prefix_n = normalize_text(prefix)
    best = None
    best_idx = None
    # Prefer the longest matching prompt (to disambiguate similar starts)
    best_len = -1
    for i, p in enumerate(prompts):
        full = p.replace("<class>", person_class)
        cand_n = normalize_text(full)
        if cand_n.startswith(prefix_n):
            if len(cand_n) > best_len:
                best = full
                best_idx = i
                best_len = len(cand_n)
    if best is not None:
        return best, best_idx

    # Fallback: drop trailing short token(s) (<=2 chars) from prefix and try again
    tokens = prefix_n.split()
    changed = False
    while tokens and len(tokens[-1]) <= 2:
        tokens.pop()
        changed = True
    if changed and tokens:
        prefix2 = " ".join(tokens)
        best = None
        best_idx = None
        best_len = -1
        for i, p in enumerate(prompts):
            full = p.replace("<class>", person_class)
            cand_n = normalize_text(full)
            if cand_n.startswith(prefix2):
                if len(cand_n) > best_len:
                    best = full
                    best_idx = i
                    best_len = len(cand_n)
        if best is not None:
            return best, best_idx

    return None, None


def build_ref_id_embeddings(reference_dir: Path) -> Dict[str, List[float]]:
    """Compute reference embeddings using the same Aligner/insightface pipeline.

    For each file in reference_dir, use the largest detected face embedding.
    Returns a mapping ref_id -> embedding (list of floats).
    """
    aligner = Aligner()
    id_to_embed: Dict[str, List[float]] = {}
    for p in sorted(reference_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        ref_id = p.stem
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        bboxes, embeds = aligner([img])
        face_bboxes = bboxes[0]
        face_embeds = embeds[0]
        if not face_embeds:
            continue
        # choose largest face
        pairs = list(zip(face_embeds, face_bboxes))
        pairs.sort(key=lambda x: -((x[1][2]-x[1][0]) * (x[1][3]-x[1][1])))
        best_embed = pairs[0][0]
        id_to_embed[ref_id] = list(map(float, best_embed))
    return id_to_embed


def compute_metrics_for_image(
    img_path: Path,
    prompt: str,
    ref_id: str,
    id_embeds: Dict[str, List[float]],
    text_metric: TextSimMetric,
    aligner: Aligner,
) -> Tuple[Optional[float], Optional[float]]:
    # text similarity (CLIP) — TextSimMetric expects list of images
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None, None

    # TextSimMetric assumes a batch of images per call
    text_metric_in = {"prompt": prompt, "generated": [img]}
    try:
        ts = text_metric(**text_metric_in)["text_sim"]
    except Exception:
        ts = None

    # ID sim: detect faces on generated, compare to reference id embed
    try:
        bboxes, embeds = aligner([img])
        face_bboxes = bboxes[0]
        face_embeds = embeds[0]
        # Match training behavior: if no face found → contribute 0
        if ref_id not in id_embeds:
            ids = None
        elif not face_embeds:
            ids = 0.0
        else:
            best = -1e9
            ref_vec = id_embeds[ref_id]
            for e in face_embeds:
                best = max(best, cos_sim(e, ref_vec))
            ids = float(best)
    except Exception:
        ids = None

    return ids, ts


def main():
    ap = argparse.ArgumentParser(description="Calculate id_sim and text_sim for generated images.")
    ap.add_argument("--gen_dir", required=True, help="Folder with generated images")
    ap.add_argument("--ref_dir", required=True, help="Folder with reference images")
    ap.add_argument("--prompts", required=True, help="Txt file with prompts (with <class>)")
    ap.add_argument("--classes", required=True, help="JSON mapping ref id -> class")
    ap.add_argument("--out_json", required=True, help="Path to save JSON results")
    ap.add_argument("--device", default=None, help="Compute device for CLIP (cuda or cpu). Default: cuda if available")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    ref_dir = Path(args.ref_dir)
    prompts_path = Path(args.prompts)
    classes_path = Path(args.classes)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)
    classes = load_classes(classes_path)
    ref_ids = list(classes.keys())

    # CLIP device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # text metric (matches training inference metric)
    text_metric = TextSimMetric(model_name="ViT-L/14@336px", device=device)
    aligner = Aligner()

    # Build id embeddings from references (same encoder via insightface)
    id_embeds = build_ref_id_embeddings(ref_dir)
    if not id_embeds:
        raise RuntimeError(f"No reference embeddings could be computed from {ref_dir}")
    missing_ids = sorted(set(classes.keys()) - set(id_embeds.keys()))
    if missing_ids:
        print("ERROR: Missing reference embeddings for:", ", ".join(missing_ids))
        print(f"Checked reference dir: {ref_dir}")
        raise SystemExit(1)

    results = []
    errors = []

    image_files = [p for p in sorted(gen_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    for p in image_files:
        # detect reference id from filename
        ref_id = detect_ref_id_from_name(p.name, ref_ids)
        if ref_id is None:
            errors.append(f"[ref-id] Could not parse reference id from filename: {p.name}")
            continue
        prefix = split_prefix_and_ref(p.name, ref_id)

        # find full prompt for this ref_id
        person_class = classes[ref_id]
        full_prompt, prompt_idx = find_full_prompt(prefix, prompts, person_class)
        if full_prompt is None:
            errors.append(f"[prompt-match] Could not match prompt for file: {p.name} | prefix='{prefix}' | class='{person_class}'")
            continue

        id_sim, text_sim = compute_metrics_for_image(
            p, full_prompt, ref_id, id_embeds, text_metric, aligner
        )
        if id_sim is None:
            errors.append(f"[id-sim] Missing ID similarity (no ref embedding or face error): {p.name} (ref={ref_id})")
            continue
        if text_sim is None:
            errors.append(f"[text-sim] Text similarity failed for: {p.name}")
            continue

        results.append(
            {
                "file": str(p.relative_to(gen_dir)),
                "prompt": full_prompt,
                "prompt_idx": prompt_idx,
                "ref": ref_id,
                "class": person_class,
                "id_sim": float(id_sim),
                "text_sim": float(text_sim),
            }
        )

    if errors:
        print(f"ERROR: {len(errors)} images could not be processed out of {len(image_files)}.")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics for {len(results)} images to {out_path}")


if __name__ == "__main__":
    main()
    
# example:
# python3 infer_tools/calc_metrics.py --gen_dir outputs/infer_branched_11new_noca_par25 --ref_dir ../dataset_full/val_dataset/references --prompts ../dataset_full/val_dataset/prompts_10.txt --classes ../dataset_full/val_dataset/classes_ref.json --out_json outputs/metrics_infer_branched_11new_noca_par25.json
