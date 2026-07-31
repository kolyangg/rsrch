#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

import sys
from pathlib import Path as _P
# Ensure we can import diffusion_template/src as 'src'
sys.path.append(str(_P(__file__).resolve().parents[2]))

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


def canonical_comet_image_name(name: str) -> str:
    """Remove Comet duplicate suffixes while preserving the semantic filename."""
    stem = Path(name).stem if Path(name).suffix else Path(name).name
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"\s*\(\d+\)$", "", stem).rstrip()
        stem = re.sub(r"__\d+$", "", stem).rstrip()
    return f"{stem.strip()}.png"


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


def load_reference_embeddings(path: Path) -> Dict[str, List[float]]:
    """Load the exact embedding mapping configured by validation."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Reference embedding file is not a non-empty mapping: {path}")
    embeddings: Dict[str, List[float]] = {}
    for ref_id, embedding in payload.items():
        value = embedding.detach().cpu().numpy() if hasattr(embedding, "detach") else embedding
        embeddings[str(ref_id)] = list(map(float, value))
    return embeddings


def compute_metrics_for_image(
    img_path: Path,
    prompt: str,
    ref_id: str,
    id_embeds: Dict[str, List[float]],
    text_metric: TextSimMetric | None,
    aligner: Aligner,
) -> Tuple[Optional[float], Optional[float]]:
    # text similarity (CLIP) — TextSimMetric expects list of images
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None, None

    # TextSimMetric assumes a batch of images per call
    ts = None
    if text_metric is not None:
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


def load_manifest_images(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps")
    if not isinstance(steps, dict) or not steps:
        raise ValueError(f"Manifest has no non-empty 'steps' mapping: {path}")

    records: list[dict[str, Any]] = []
    for raw_step, assets in steps.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid step key {raw_step!r} in {path}") from error
        if not isinstance(assets, list):
            raise ValueError(f"Manifest step {step} is not a list")
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError(f"Manifest step {step} contains a non-object asset")
            local_path = Path(str(asset.get("local_path") or "")).expanduser()
            file_name = str(asset.get("file_name") or local_path.name)
            records.append(
                {
                    "step": step,
                    "asset_id": str(asset.get("asset_id") or ""),
                    "file_name": file_name,
                    "local_path": local_path,
                }
            )
    return records


def main():
    ap = argparse.ArgumentParser(description="Calculate id_sim and text_sim for generated images.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--gen_dir", help="Folder with generated images")
    source.add_argument(
        "--manifest",
        help=(
            "Verified download_manifest.json from download_face_quality_images.py. "
            "Outputs step and Comet asset IDs for exact per-image joins."
        ),
    )
    ap.add_argument("--ref_dir", required=True, help="Folder with reference images")
    ap.add_argument("--prompts", required=True, help="Txt file with prompts (with <class>)")
    ap.add_argument("--classes", required=True, help="JSON mapping ref id -> class")
    ap.add_argument(
        "--id-embeds-pth",
        default=None,
        help=(
            "Optional precomputed reference-embedding mapping. Omit this when the "
            "validation metric derived embeddings from --ref_dir at runtime."
        ),
    )
    ap.add_argument("--out_json", required=True, help="Path to save JSON results")
    ap.add_argument("--device", default=None, help="Compute device for CLIP (cuda or cpu). Default: cuda if available")
    ap.add_argument(
        "--id-only",
        action="store_true",
        help="Skip CLIP text similarity when only per-image identity similarity is needed.",
    )
    ap.add_argument(
        "--expected-images-per-step",
        type=int,
        default=None,
        help="Fail unless every manifest step produces exactly this many result rows.",
    )
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir) if args.gen_dir else None
    manifest_path = Path(args.manifest) if args.manifest else None
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
    text_metric = (
        None
        if args.id_only
        else TextSimMetric(model_name="ViT-L/14@336px", device=device)
    )
    aligner = Aligner()

    # Reuse a precomputed embedding artifact only when it matches the validation
    # contract. The full-96 one-ID config derives these held-out identities from
    # ref_images at runtime, so its exact offline replay leaves this option unset.
    id_embeds = (
        load_reference_embeddings(Path(args.id_embeds_pth))
        if args.id_embeds_pth
        else build_ref_id_embeddings(ref_dir)
    )
    if not id_embeds:
        raise RuntimeError(f"No reference embeddings could be computed from {ref_dir}")
    missing_ids = sorted(set(classes.keys()) - set(id_embeds.keys()))
    if missing_ids:
        print("ERROR: Missing reference embeddings for:", ", ".join(missing_ids))
        print(f"Checked reference dir: {ref_dir}")
        raise SystemExit(1)

    results = []
    errors = []

    if manifest_path is not None:
        image_records = load_manifest_images(manifest_path)
    else:
        assert gen_dir is not None
        image_records = [
            {
                "step": None,
                "asset_id": "",
                "file_name": p.name,
                "local_path": p,
            }
            for p in sorted(gen_dir.rglob("*"))
            if p.suffix.lower() in IMG_EXTS
        ]

    for image_record in image_records:
        p = Path(image_record["local_path"])
        semantic_name = canonical_comet_image_name(str(image_record["file_name"]))
        # detect reference id from filename
        ref_id = detect_ref_id_from_name(semantic_name, ref_ids)
        if ref_id is None:
            errors.append(
                f"[ref-id] Could not parse reference id from filename: "
                f"{image_record['file_name']}"
            )
            continue
        prefix = split_prefix_and_ref(semantic_name, ref_id)

        # find full prompt for this ref_id
        person_class = classes[ref_id]
        full_prompt, prompt_idx = find_full_prompt(prefix, prompts, person_class)
        if full_prompt is None:
            errors.append(
                f"[prompt-match] Could not match prompt for file: "
                f"{image_record['file_name']} | prefix='{prefix}' | "
                f"class='{person_class}'"
            )
            continue

        id_sim, text_sim = compute_metrics_for_image(
            p, full_prompt, ref_id, id_embeds, text_metric, aligner
        )
        if id_sim is None:
            errors.append(
                f"[id-sim] Missing ID similarity (no ref embedding or face error): "
                f"{image_record['file_name']} (ref={ref_id})"
            )
            continue
        if not args.id_only and text_sim is None:
            errors.append(
                f"[text-sim] Text similarity failed for: {image_record['file_name']}"
            )
            continue

        row = {
            "file": (
                str(p.relative_to(gen_dir))
                if gen_dir is not None
                else str(p.resolve())
            ),
            "file_name": str(image_record["file_name"]),
            "sample_key": Path(semantic_name).stem,
            "prompt": full_prompt,
            "prompt_idx": prompt_idx,
            "ref": ref_id,
            "class": person_class,
            "id_sim": float(id_sim),
        }
        if image_record["step"] is not None:
            row["step"] = int(image_record["step"])
        if image_record["asset_id"]:
            row["asset_id"] = image_record["asset_id"]
        if text_sim is not None:
            row["text_sim"] = float(text_sim)
        results.append(row)

    if errors:
        print(
            f"ERROR: {len(errors)} images could not be processed out of "
            f"{len(image_records)}."
        )
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    if args.expected_images_per_step is not None:
        if manifest_path is None:
            raise ValueError("--expected-images-per-step requires --manifest")
        if args.expected_images_per_step < 1:
            raise ValueError("--expected-images-per-step must be positive")
        counts = Counter(int(row["step"]) for row in results)
        bad = {
            step: count
            for step, count in sorted(counts.items())
            if count != args.expected_images_per_step
        }
        expected_steps = {
            int(step)
            for step in json.loads(manifest_path.read_text(encoding="utf-8"))["steps"]
        }
        missing_steps = sorted(expected_steps - set(counts))
        if bad or missing_steps:
            raise RuntimeError(
                "Per-step result count mismatch: "
                f"bad={bad}, missing={missing_steps}, "
                f"expected={args.expected_images_per_step}"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics for {len(results)} images to {out_path}")


if __name__ == "__main__":
    main()
    
# example:
# python3 tools/inference/calc_metrics.py --gen_dir outputs/infer_branched_11new_noca_par25 --ref_dir ../dataset_full/val_dataset/references --prompts ../dataset_full/val_dataset/prompts_10.txt --classes ../dataset_full/val_dataset/classes_ref.json --out_json outputs/metrics_infer_branched_11new_noca_par25.json
