#!/usr/bin/env python3
"""Compute face/geometry metrics and build a checkpoint inspection PDF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.model.photomaker_branched.insightface_package import (
    analyze_faces,
    create_face_analyzer,
)


COSMIC_DATA = HERE / "data" / "id_00081_1017318003459"
ONE_ID_ROOT = Path("/home/niko/rsrch/dataset_full/one_id")
ONE_ID_PROMPTS = Path("/home/niko/rsrch/dataset_full/val_dataset/prompts_10.txt")
DEFAULT_STEPS = (0, 200, 400, 600)
CLIP_MODEL_NAME = "ViT-L/14@336px"
CLIP_CACHE = Path("/home/niko/.cache/clip")


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def resolve_dataset(run_dir: Path) -> tuple[Path, list[str], str]:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    profile = manifest.get("dataset_profile", "cosmic_large_id00081")
    if profile == "cosmic_large_id00081":
        prompts_path = COSMIC_DATA / "validation_prompts_4.txt"
        reference = COSMIC_DATA / "validation_refs" / "holdout_A.jpg"
        class_name = "woman"
    elif profile in {
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    }:
        prompts_path = ONE_ID_PROMPTS
        reference = ONE_ID_ROOT / "ref" / "51.jpg"
        class_name = "man img"
    else:
        raise ValueError(f"Unsupported dataset profile: {profile}")
    prompts = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:4]
    if class_name:
        prompts = [prompt.replace("<class>", class_name) for prompt in prompts]
    return reference, prompts, profile


def clip_prompts(prompts):
    """Remove the PhotoMaker trigger token before semantic scoring."""
    normalized = []
    for prompt in prompts:
        prompt = prompt.replace("<class>", "person")
        prompt = re.sub(r"\bimg\b", "", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        prompt = re.sub(r"\s+,", ",", prompt)
        normalized.append(prompt.strip())
    return normalized


def load_clip():
    import clip

    model, preprocess = clip.load(
        CLIP_MODEL_NAME,
        device="cpu",
        jit=False,
        download_root=str(CLIP_CACHE),
    )
    model.eval()
    return model, preprocess


def prompt_image_scores(model, preprocess, images, prompts):
    import clip
    import torch

    image_batch = torch.stack([preprocess(image) for image in images])
    text_batch = clip.tokenize(clip_prompts(prompts), truncate=True)
    with torch.inference_mode():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text_batch)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        scores = (image_features * text_features).sum(dim=-1)
    return [float(value) for value in scores.cpu()]


def face_value(face, key):
    return face.get(key) if isinstance(face, dict) else getattr(face, key, None)


def detect(analyzer, image):
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
    faces = analyze_faces(analyzer, bgr)
    if not faces:
        return {"detected": False, "bbox": None, "embedding": None, "kps": None}

    def area(face):
        box = face_value(face, "bbox")
        if box is None:
            return -1
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    selected = max(faces, key=area)
    embedding = face_value(selected, "embedding")
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
    bbox = face_value(selected, "bbox")
    kps = face_value(selected, "kps")
    return {
        "detected": bbox is not None,
        "bbox": None if bbox is None else np.asarray(bbox, dtype=np.float32),
        "embedding": embedding,
        "kps": None if kps is None else np.asarray(kps, dtype=np.float32),
    }


def cosine(left, right):
    if left is None or right is None:
        return float("nan")
    return float(np.dot(left, right))


def bbox_iou(left, right):
    if left is None or right is None:
        return float("nan")
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union else float("nan")


def landmark_displacement(current, baseline):
    if current["kps"] is None or baseline["kps"] is None or baseline["bbox"] is None:
        return float("nan")
    x0, y0, x1, y1 = baseline["bbox"]
    diagonal = math.hypot(float(x1 - x0), float(y1 - y0))
    if diagonal <= 0:
        return float("nan")
    return float(
        np.sqrt(np.mean((current["kps"] - baseline["kps"]) ** 2)) / diagonal
    )


def mae_regions(current, baseline, bbox):
    current_a = np.asarray(current.convert("RGB"), dtype=np.float32) / 255
    baseline_a = np.asarray(baseline.convert("RGB"), dtype=np.float32) / 255
    difference = np.abs(current_a - baseline_a)
    if bbox is None:
        return float(difference.mean()), float("nan"), float("nan")
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0, x1 = max(0, x0), min(current.width, x1)
    y0, y1 = max(0, y0), min(current.height, y1)
    mask = np.zeros(current_a.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = True
    face = float(difference[mask].mean()) if mask.any() else float("nan")
    outside = float(difference[~mask].mean()) if (~mask).any() else float("nan")
    return float(difference.mean()), face, outside


def validation_images(run_dir, mode, step):
    root = run_dir / "validation" / mode / f"step_{step:04d}" / "outputs"
    candidates = sorted(
        path
        for path in root.glob("*/val_images/manual_val/step_*_batch_*/*.png")
        if not path.stem.endswith("_mask")
    )
    if len(candidates) != 4:
        raise RuntimeError(
            f"Expected four {mode} step {step} images, found {len(candidates)} under {root}"
        )
    return [Image.open(path).convert("RGB") for path in candidates]


def finite_median(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmedian(array)) if np.isfinite(array).any() else float("nan")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    run_dir = args().run_dir.resolve()
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    steps = tuple(
        int(step)
        for step in run_manifest.get("protocol", {}).get(
            "validation_steps", DEFAULT_STEPS
        )
    )
    reference_path, prompts, dataset_profile = resolve_dataset(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    analyzer = create_face_analyzer(
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
        ctx_id=-1,
        det_size=(640, 640),
        fallback_ctx_id=-1,
        quiet=True,
    )

    reference = Image.open(reference_path).convert("RGB")
    reference_face = detect(analyzer, reference)
    pm_images = validation_images(run_dir, "pmControl50", 0)
    pm_faces = [detect(analyzer, image) for image in pm_images]
    pm_ref_sims = [
        cosine(face["embedding"], reference_face["embedding"]) for face in pm_faces
    ]
    clip_model, clip_preprocess = load_clip()
    pm_text_scores = prompt_image_scores(
        clip_model, clip_preprocess, pm_images, prompts
    )

    modes = []
    for candidate_mode in ("canonical50", "earlyBA50"):
        roots = [
            run_dir
            / "validation"
            / candidate_mode
            / f"step_{step:04d}"
            / "outputs"
            for step in steps
        ]
        if all(len(list(root.rglob("*.png"))) >= 4 for root in roots):
            modes.append(candidate_mode)
    if "canonical50" not in modes:
        raise RuntimeError(
            "A complete canonical50 sweep is required at steps "
            + ",".join(str(step) for step in steps)
        )

    rows = []
    loaded = {}
    for mode in modes:
        for step in steps:
            images = validation_images(run_dir, mode, step)
            loaded[(mode, step)] = images
            text_scores = prompt_image_scores(
                clip_model, clip_preprocess, images, prompts
            )
            for prompt_index, (image, pm_image, pm_face) in enumerate(
                zip(images, pm_images, pm_faces)
            ):
                face = detect(analyzer, image)
                full_mae, face_mae, outside_mae = mae_regions(
                    image, pm_image, pm_face["bbox"]
                )
                ref_sim = cosine(face["embedding"], reference_face["embedding"])
                pm_sim = cosine(face["embedding"], pm_face["embedding"])
                rows.append(
                    {
                        "mode": mode,
                        "step": step,
                        "prompt_index": prompt_index,
                        "prompt": prompts[prompt_index],
                        "face_detected": face["detected"],
                        "reference_similarity": ref_sim,
                        "pm_reference_similarity": pm_ref_sims[prompt_index],
                        "reference_gain_vs_pm": ref_sim - pm_ref_sims[prompt_index],
                        "face_similarity_to_pm_output": pm_sim,
                        "face_distance_from_pm_output": 1.0 - pm_sim,
                        "bbox_iou_vs_pm": bbox_iou(face["bbox"], pm_face["bbox"]),
                        "landmark_displacement_vs_pm": landmark_displacement(
                            face, pm_face
                        ),
                        "full_mae_vs_pm": full_mae,
                        "face_mae_vs_pm": face_mae,
                        "outside_mae_vs_pm": outside_mae,
                        "prompt_image_clip_cosine": text_scores[prompt_index],
                        "prompt_image_clip_score": (
                            100.0 * max(0.0, text_scores[prompt_index])
                        ),
                        "pm_prompt_image_clip_cosine": (
                            pm_text_scores[prompt_index]
                        ),
                        "prompt_image_clip_gain_vs_pm": (
                            text_scores[prompt_index]
                            - pm_text_scores[prompt_index]
                        ),
                    }
                )

    summary = []
    for mode in modes:
        for step in steps:
            subset = [
                row for row in rows if row["mode"] == mode and row["step"] == step
            ]
            record = {"mode": mode, "step": step, "sample_count": len(subset)}
            for key in (
                "reference_similarity",
                "reference_gain_vs_pm",
                "face_similarity_to_pm_output",
                "face_distance_from_pm_output",
                "bbox_iou_vs_pm",
                "landmark_displacement_vs_pm",
                "full_mae_vs_pm",
                "face_mae_vs_pm",
                "outside_mae_vs_pm",
                "prompt_image_clip_cosine",
                "prompt_image_clip_score",
                "pm_prompt_image_clip_cosine",
                "prompt_image_clip_gain_vs_pm",
            ):
                record[f"median_{key}"] = finite_median([row[key] for row in subset])
            record["face_detection_rate"] = float(
                np.mean([row["face_detected"] for row in subset])
            )
            record["selection_score"] = float(
                record["median_reference_similarity"]
                + 0.20 * record["median_face_distance_from_pm_output"]
                - 1.5 * record["median_landmark_displacement_vs_pm"]
                - 3.0 * record["median_outside_mae_vs_pm"]
            )
            summary.append(record)

    fieldnames = list(rows[0])
    with (report_dir / "metrics_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "metrics_per_image.json").write_text(
        json.dumps(json_safe(rows), indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "metrics_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )

    pdf_path = report_dir / "checkpoint_visual_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        for mode in modes:
            for prompt_index, prompt in enumerate(prompts):
                panels = [
                    ("Held-out ref", reference),
                    ("PhotoMaker", pm_images[prompt_index]),
                    *[
                        (f"step {step}", loaded[(mode, step)][prompt_index])
                        for step in steps
                    ],
                ]
                columns = min(6, len(panels))
                rows_count = math.ceil(len(panels) / columns)
                fig, axes = plt.subplots(
                    rows_count,
                    columns,
                    figsize=(3.7 * columns, 4.1 * rows_count),
                    squeeze=False,
                )
                for axis, (title, image) in zip(axes.flat, panels):
                    axis.imshow(image)
                    axis.set_title(title)
                    axis.axis("off")
                for axis in axes.flat[len(panels):]:
                    axis.axis("off")
                fig.suptitle(f"{mode} · prompt {prompt_index}: {prompt}", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        keys = (
            ("median_reference_similarity", "Reference identity similarity"),
            ("median_face_similarity_to_pm_output", "Similarity to PhotoMaker face"),
            ("median_landmark_displacement_vs_pm", "Landmark displacement vs PM"),
            ("median_outside_mae_vs_pm", "Outside-face MAE vs PM"),
            ("median_prompt_image_clip_cosine", "CLIP prompt–image cosine"),
            ("selection_score", "Screening score"),
        )
        for axis, (key, title) in zip(axes.flat, keys):
            for mode in modes:
                subset = [row for row in summary if row["mode"] == mode]
                axis.plot(
                    [row["step"] for row in subset],
                    [row[key] for row in subset],
                    marker="o",
                    label=mode,
                )
            axis.set_title(title)
            axis.set_xlabel("optimizer step")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(run_dir.name)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    best = max(
        (row for row in summary if row["mode"] == "canonical50"),
        key=lambda row: row["selection_score"],
    )
    (report_dir / "SUMMARY.md").write_text(
        f"# {run_dir.name} checkpoint summary\n\n"
        f"Dataset profile: `{dataset_profile}`.\n\n"
        f"Best canonical checkpoint by screening score: **step {best['step']}**.\n\n"
        "The score rewards held-out-reference similarity and remaining distinct "
        "from PhotoMaker, while penalizing landmark displacement and outside-face drift. "
        "The PDF remains the promotion gate.\n\n"
        f"Visual report: `{pdf_path}`\n",
        encoding="utf-8",
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
