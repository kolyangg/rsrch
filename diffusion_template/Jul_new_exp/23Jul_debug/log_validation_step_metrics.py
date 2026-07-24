#!/usr/bin/env python3
"""Compute and log one checkpoint's validation metrics into its training run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from launch_validation import ensure_comet_api_key
from summarize_run import (
    bbox_iou,
    cosine,
    detect,
    finite_median,
    landmark_displacement,
    mae_regions,
    resolve_dataset,
    validation_images,
)
from src.model.photomaker_branched.insightface_package import create_face_analyzer


CLIP_MODEL_NAME = "ViT-L/14@336px"
CLIP_CACHE = Path("/home/niko/.cache/clip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    return parser.parse_args()


def clip_prompts(prompts: list[str]) -> list[str]:
    """Remove PhotoMaker-only trigger tokens before CLIP text encoding."""
    normalized = []
    for prompt in prompts:
        prompt = prompt.replace("<class>", "person")
        prompt = re.sub(r"\bimg\b", "", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        prompt = re.sub(r"\s+,", ",", prompt)
        normalized.append(prompt.strip())
    return normalized


def prompt_image_scores(
    images: list[Image.Image],
    pm_images: list[Image.Image],
    prompts: list[str],
) -> tuple[list[float], list[float]]:
    """Return aligned CLIP cosine scores for BA and PhotoMaker images."""
    import clip
    import torch

    model, preprocess = clip.load(
        CLIP_MODEL_NAME,
        device="cpu",
        jit=False,
        download_root=str(CLIP_CACHE),
    )
    model.eval()
    all_images = images + pm_images
    image_batch = torch.stack([preprocess(image) for image in all_images])
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
        current = (image_features[: len(images)] * text_features).sum(dim=-1)
        baseline = (image_features[len(images) :] * text_features).sum(dim=-1)
    return (
        [float(value) for value in current.cpu()],
        [float(value) for value in baseline.cpu()],
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    step = int(args.step)
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    valid_steps = {
        int(value) for value in manifest["protocol"]["validation_steps"]
    }
    if step not in valid_steps:
        raise ValueError(f"Step {step} is not in {sorted(valid_steps)}")
    comet_key = manifest.get("comet_experiment_key")
    if not comet_key:
        raise RuntimeError("Validation must resolve the training Comet key first")

    reference_path, prompts, dataset_profile = resolve_dataset(run_dir)
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
    images = validation_images(run_dir, "canonical50", step)
    text_scores, pm_text_scores = prompt_image_scores(
        images, pm_images, prompts
    )

    rows = []
    for prompt_index, (image, pm_image) in enumerate(zip(images, pm_images)):
        face = detect(analyzer, image)
        pm_face = detect(analyzer, pm_image)
        full_mae, face_mae, outside_mae = mae_regions(
            image, pm_image, pm_face["bbox"]
        )
        reference_similarity = cosine(
            face["embedding"], reference_face["embedding"]
        )
        pm_reference_similarity = cosine(
            pm_face["embedding"], reference_face["embedding"]
        )
        face_similarity_to_pm = cosine(
            face["embedding"], pm_face["embedding"]
        )
        rows.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompts[prompt_index],
                "face_detected": face["detected"],
                "reference_similarity": reference_similarity,
                "pm_reference_similarity": pm_reference_similarity,
                "reference_gain_vs_pm": (
                    reference_similarity - pm_reference_similarity
                ),
                "face_similarity_to_pm_output": face_similarity_to_pm,
                "face_distance_from_pm_output": 1.0 - face_similarity_to_pm,
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
                "pm_prompt_image_clip_cosine": pm_text_scores[prompt_index],
                "prompt_image_clip_gain_vs_pm": (
                    text_scores[prompt_index] - pm_text_scores[prompt_index]
                ),
            }
        )

    metric_keys = (
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
    )
    summary = {
        "mode": "canonical50",
        "step": step,
        "dataset_profile": dataset_profile,
        "sample_count": len(rows),
        "clip_model": CLIP_MODEL_NAME,
    }
    for key in metric_keys:
        summary[f"median_{key}"] = finite_median([row[key] for row in rows])
    summary["face_detection_rate"] = float(
        np.mean([row["face_detected"] for row in rows])
    )
    summary["selection_score"] = float(
        summary["median_reference_similarity"]
        + 0.20 * summary["median_face_distance_from_pm_output"]
        - 1.5 * summary["median_landmark_displacement_vs_pm"]
        - 3.0 * summary["median_outside_mae_vs_pm"]
    )

    output_dir = run_dir / "report" / "incremental_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"step_{step:04d}.json"
    output.write_text(
        json.dumps({"summary": summary, "per_prompt": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    ensure_comet_api_key()
    from comet_ml import ExistingExperiment

    experiment = ExistingExperiment(previous_experiment=comet_key)
    if experiment.get_key() != comet_key:
        raise RuntimeError(
            f"Comet resume verification failed: {experiment.get_key()} != {comet_key}"
        )
    experiment.set_name(manifest["run_name"])
    for key, value in summary.items():
        if key in {"mode", "step", "dataset_profile", "clip_model"}:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            experiment.log_metric(
                f"validation/canonical50/{key}", value, step=step
            )
    for row in rows:
        prompt_index = int(row["prompt_index"])
        for key in metric_keys:
            value = row[key]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                experiment.log_metric(
                    f"validation_per_prompt/canonical50/p{prompt_index:02d}/{key}",
                    value,
                    step=step,
                )
    experiment.log_other(
        f"4k_validation_step_{step:04d}_summary",
        json.dumps(summary, sort_keys=True),
    )
    experiment.log_asset(
        str(output),
        file_name=f"4k_incremental_metrics_step_{step:04d}.json",
    )
    experiment.end()
    receipt = output_dir / f"step_{step:04d}.comet_uploaded.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "completed",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "step": step,
                "comet_experiment_key": comet_key,
                "run_name": manifest["run_name"],
                "metric_payload": str(output),
                "clip_model": CLIP_MODEL_NAME,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
