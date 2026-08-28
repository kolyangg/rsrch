#!/usr/bin/env python3
"""Score the full-96 CL39 all-70 and PM/spatial attribution controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity


ARMS = {
    "actual16": ("CL39_16k_all70_actual_attribution_r1", 16000),
    "ba_off16": ("CL39_16k_all70_ba_off_attribution_r1", 16000),
    "A": ("CL39_24k_cross_A_correct_pm_correct_spatial_r1", 24000),
    "B": ("CL39_24k_cross_B_correct_pm_wrong_spatial_r1", 24000),
    "C": ("CL39_24k_cross_C_wrong_pm_correct_spatial_r1", 24000),
    "D": ("CL39_24k_cross_D_wrong_pm_wrong_spatial_r1", 24000),
}
ARM_LABELS = {
    "actual16": "16k actual",
    "ba_off16": "16k all-70 BA-off",
    "A": "A correct PM / correct spatial",
    "B": "B correct PM / wrong spatial",
    "C": "C wrong PM / correct spatial",
    "D": "D wrong PM / wrong spatial",
}
CONTRASTS = {
    "actual16_minus_ba_off16": ("actual16", "ba_off16"),
    "A_minus_B_spatial_given_correct_PM": ("A", "B"),
    "C_minus_D_spatial_given_wrong_PM": ("C", "D"),
    "A_minus_C_PM_given_correct_spatial": ("A", "C"),
    "B_minus_D_PM_given_wrong_spatial": ("B", "D"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--subject-v2-embeds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_key(value: str) -> str:
    # AICODE-NOTE: validation metadata uses spaces while exported PNGs use
    # underscores. Literal joins silently lose most fixed-panel cells.
    return value.replace(" ", "_")


def image_path(task_root: Path, arm: str, output_key: str) -> Path:
    run_name, step = ARMS[arm]
    matches = list(
        (task_root / "saved" / run_name / "val_images" / "manual_val").glob(
            f"step_{step}_batch_*/{normalize_key(output_key)}"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one image for arm={arm} key={output_key!r}, found {matches}"
        )
    return matches[0]


def crop(array: np.ndarray, bbox, padding: float = 0.25) -> np.ndarray:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width, height = x1 - x0, y1 - y0
    x0 = max(0, int(round(x0 - padding * width)))
    y0 = max(0, int(round(y0 - padding * height)))
    x1 = min(array.shape[1], int(round(x1 + padding * width)))
    y1 = min(array.shape[0], int(round(y1 + padding * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox {bbox} for image shape {array.shape}")
    return array[y0:y1, x0:x1]


def distance(candidate: Image.Image, baseline: Image.Image, bbox) -> dict[str, float]:
    first = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(baseline.convert("RGB"), dtype=np.float32) / 255.0
    delta = np.abs(first - second)
    face_delta = np.abs(crop(first, bbox) - crop(second, bbox))
    return {
        "rgb_mae_all": float(delta.mean()),
        "rgb_mae_face": float(face_delta.mean()),
        "changed_all": float((delta.max(axis=2) > 1.0 / 255.0).mean()),
        "changed_face": float(
            (face_delta.max(axis=2) > 1.0 / 255.0).mean()
        ),
        "ssim_all": float(
            structural_similarity(first, second, data_range=1.0, channel_axis=2)
        ),
    }


def paired_summary(frame: pd.DataFrame, first: str, second: str) -> dict:
    pivot = frame.pivot(index="output_key", columns="arm", values="id_intended")
    delta = (pivot[first] - pivot[second]).dropna().to_numpy()
    rng = np.random.default_rng(390027)
    samples = delta[
        rng.integers(0, len(delta), size=(100_000, len(delta)))
    ].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "n": int(len(delta)),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "wins": int(np.sum(delta > 1.0e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1.0e-12)),
        "losses": int(np.sum(delta < -1.0e-12)),
        "bootstrap_95_low": float(np.quantile(samples, 0.025)),
        "bootstrap_95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_seed": 390027,
    }


def score_face(metric, image: Image.Image, bbox, intended: str, wrong: str) -> dict:
    from src.utils.model_utils import cos_sim

    batch_bboxes, batch_embeds = metric.aligner([image])
    boxes = batch_bboxes[0]
    embeds = batch_embeds[0]
    if not boxes or not embeds:
        return {
            "id_intended": 0.0,
            "id_wrong": 0.0,
            "id_margin": 0.0,
            "mask_iou": 0.0,
            "face_count": 0,
            "no_face": 1,
            "unowned": 1,
            "ambiguous": 0,
        }
    ranked = sorted(
        (
            (metric._bbox_iou(box, bbox), index, embed)
            for index, (box, embed) in enumerate(zip(boxes, embeds))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_iou, _index, best_embed = ranked[0]
    ambiguous = bool(
        len(ranked) > 1
        and ranked[1][0] >= metric.minimum_mask_iou
        and abs(best_iou - ranked[1][0]) <= metric.ambiguity_iou_margin
    )
    unowned = best_iou < metric.minimum_mask_iou
    if unowned:
        intended_score = wrong_score = 0.0
    else:
        intended_score = float(cos_sim(best_embed, metric.id_embeds[intended]))
        wrong_score = float(cos_sim(best_embed, metric.id_embeds[wrong]))
    return {
        "id_intended": intended_score,
        "id_wrong": wrong_score,
        "id_margin": intended_score - wrong_score,
        "mask_iou": float(best_iou),
        "face_count": int(len(boxes)),
        "no_face": 0,
        "unowned": int(unowned),
        "ambiguous": int(ambiguous),
    }


def render_identity_summary(aggregate: pd.DataFrame, output: Path) -> None:
    order = list(ARMS)
    selected = aggregate.set_index("arm").loc[order]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, selected.id_intended, width, label="intended identity")
    ax.bar(x + width / 2, selected.id_wrong, width, label="next/wrong identity")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in order], rotation=20, ha="right")
    ax.set_ylabel("Subject-v2 mask-matched cosine")
    ax.set_title("CL39 attribution controls: intended versus wrong identity")
    ax.legend()
    for xpos, value in zip(x - width / 2, selected.id_intended):
        ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", fontsize=8)
    for xpos, value in zip(x + width / 2, selected.id_wrong):
        ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_effects(effects: pd.DataFrame, output: Path) -> None:
    labels = [
        "16k actual − all-70 BA-off",
        "24k spatial: A − B | correct PM",
        "24k spatial: C − D | wrong PM",
        "24k PM: A − C | correct spatial",
        "24k PM: B − D | wrong spatial",
    ]
    y = np.arange(len(effects))[::-1]
    means = effects["mean"].to_numpy()
    low = effects["bootstrap_95_low"].to_numpy()
    high = effects["bootstrap_95_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - low, high - means)),
        fmt="o",
        color="#0b6e99",
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired mean Δ intended-ID (fixed-cell bootstrap 95% interval)")
    ax.set_title("Causal identity effects on the fixed 96-cell panel")
    for xpos, ypos in zip(means, y):
        ax.text(xpos, ypos + 0.18, f"{xpos:+.4f}", ha="center", fontsize=9)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_all70_scatter(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.query("arm in ['actual16', 'ba_off16']").pivot(
        index="output_key", columns="arm", values="id_intended"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].scatter(pivot.ba_off16, pivot.actual16, alpha=0.75, s=28)
    lo = float(min(pivot.min()))
    hi = float(max(pivot.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    axes[0].set_xlabel("All-70 BA-off intended-ID")
    axes[0].set_ylabel("Actual intended-ID")
    axes[0].set_title("16k paired cells")
    delta = pivot.actual16 - pivot.ba_off16
    axes[1].hist(delta, bins=18, color="#0b6e99", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(delta.mean(), color="#c44e52", linewidth=1.5)
    axes[1].set_xlabel("Actual − BA-off intended-ID")
    axes[1].set_ylabel("Cells")
    axes[1].set_title(f"Mean Δ = {delta.mean():+.4f}")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_identity_strata(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot_table(
        index="identity", columns="arm", values="id_intended", aggfunc="mean"
    )
    values = pd.DataFrame(
        {
            "16k actual−BA-off": pivot.actual16 - pivot.ba_off16,
            "24k A−B spatial|PM✓": pivot.A - pivot.B,
            "24k C−D spatial|PM✗": pivot.C - pivot.D,
            "24k A−C PM|spatial✓": pivot.A - pivot.C,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    vmax = float(np.abs(values.to_numpy()).max())
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(values.shape[1]), values.columns, rotation=22, ha="right")
    ax.set_yticks(range(values.shape[0]), values.index)
    ax.set_title("Identity-stratified paired effects (12 prompts per identity)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Δ intended-ID")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_representatives(
    frame: pd.DataFrame,
    task_root: Path,
    reference_root: Path,
    output: Path,
) -> list[dict]:
    ids = sorted(frame.identity.unique())
    wrong_by_id = {identity: ids[(index + 1) % len(ids)] for index, identity in enumerate(ids)}
    pivot = frame.pivot(index="output_key", columns="arm", values="id_intended")
    choices = [
        ("largest A−B", (pivot.A - pivot.B).idxmax()),
        ("largest C−D", (pivot.C - pivot.D).idxmax()),
        ("largest A−C", (pivot.A - pivot.C).idxmax()),
        ("lowest A−B", (pivot.A - pivot.B).idxmin()),
    ]
    unique = []
    for label, key in choices:
        if key not in {entry[1] for entry in unique}:
            unique.append((label, key))
    columns = ["target ref", "wrong ref", "A", "B", "C", "D"]
    fig, axes = plt.subplots(len(unique), len(columns), figsize=(15, 3.4 * len(unique)))
    if len(unique) == 1:
        axes = axes[None, :]
    records = []
    for row_index, (label, output_key) in enumerate(unique):
        subset = frame[frame.output_key == output_key].set_index("arm")
        identity = str(subset.identity.iloc[0])
        wrong = wrong_by_id[identity]
        bbox = json.loads(subset.bbox.iloc[0])
        images = [
            Image.open(next(reference_root.glob(f"{identity}.*"))).convert("RGB"),
            Image.open(next(reference_root.glob(f"{wrong}.*"))).convert("RGB"),
        ] + [Image.open(image_path(task_root, arm, output_key)).convert("RGB") for arm in "ABCD"]
        for column_index, (column, image) in enumerate(zip(columns, images)):
            ax = axes[row_index, column_index]
            ax.imshow(image)
            ax.axis("off")
            if column_index >= 2:
                arm = column
                score = subset.loc[arm]
                ax.add_patch(
                    Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        fill=False,
                        edgecolor="#ff3b30",
                        linewidth=2.0,
                    )
                )
                ax.set_title(
                    f"{column}\nID✓ {score.id_intended:.3f} / ID✗ {score.id_wrong:.3f}",
                    fontsize=9,
                )
            else:
                ax.set_title(column, fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"{label}\n{identity}: {output_key.rsplit('_', 1)[0]}", fontsize=9
        )
        records.append({"selection": label, "output_key": output_key, "identity": identity})
    fig.suptitle("Representative 24k identity-source crossings (red = fixed target face box)")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    bbox_data = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    normalized = {normalize_key(key): key for key in bbox_data}
    if len(normalized) != 96:
        raise RuntimeError("Normalized bbox keys are not one-to-one")
    identities = sorted(
        {
            key.removesuffix(".png").rsplit("_", 1)[-1]
            for key in bbox_data
        }
    )
    if len(identities) != 8:
        raise RuntimeError(f"Expected eight fixed-panel identities, found {identities}")
    wrong_by_id = {
        identity: identities[(index + 1) % len(identities)]
        for index, identity in enumerate(identities)
    }

    from src.metrics.id_sim_metric import IDSimMaskMatched

    metric = IDSimMaskMatched(
        id_embeds_pth=str(args.subject_v2_embeds),
        device=args.device,
        metric_name="id_sim_subject_v2",
    )
    rows = []
    for cell_index, output_key in enumerate(bbox_data):
        identity = output_key.removesuffix(".png").rsplit("_", 1)[-1]
        wrong = wrong_by_id[identity]
        bbox = bbox_data[output_key].get("face_crop_new") or bbox_data[output_key].get("face_crop_old")
        if bbox is None:
            raise RuntimeError(f"Missing generated face box for {output_key}")
        prompt_key = output_key.removesuffix(".png").rsplit("_", 1)[0].strip()
        area_fraction = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (1024.0 * 1024.0)
        images = {
            arm: Image.open(image_path(args.task_root, arm, output_key)).convert("RGB")
            for arm in ARMS
        }
        for arm, image in images.items():
            if arm in {"actual16", "ba_off16"}:
                baseline = images["actual16"]
            else:
                baseline = images["A"]
            rows.append(
                {
                    "cell_index": cell_index,
                    "output_key": output_key,
                    "normalized_output_key": normalize_key(output_key),
                    "identity": identity,
                    "wrong_identity": wrong,
                    "prompt_key": prompt_key,
                    "bbox": json.dumps(bbox),
                    "face_area_fraction": area_fraction,
                    "arm": arm,
                    "run_name": ARMS[arm][0],
                    "step": ARMS[arm][1],
                    **score_face(metric, image, bbox, identity, wrong),
                    **{
                        f"{key}_vs_actual": value
                        for key, value in distance(image, baseline, bbox).items()
                    },
                }
            )
        if (cell_index + 1) % 12 == 0:
            print(f"scored {cell_index + 1}/96 cells", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_image.csv", index=False)
    numeric = [
        column
        for column in frame.columns
        if column
        not in {
            "output_key",
            "normalized_output_key",
            "identity",
            "wrong_identity",
            "prompt_key",
            "bbox",
            "arm",
            "run_name",
        }
    ]
    aggregate = frame.groupby("arm", sort=False)[numeric].mean().reset_index()
    aggregate.to_csv(args.output_root / "aggregate.csv", index=False)

    effects = []
    for name, (first, second) in CONTRASTS.items():
        effects.append({"contrast": name, **paired_summary(frame, first, second)})
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(args.output_root / "paired_effects.csv", index=False)

    identity_means = frame.pivot_table(
        index="identity", columns="arm", values="id_intended", aggfunc="mean"
    )
    identity_means.to_csv(args.output_root / "identity_means.csv")
    prompt_means = frame.pivot_table(
        index="prompt_key", columns="arm", values="id_intended", aggfunc="mean"
    )
    prompt_means.to_csv(args.output_root / "prompt_means.csv")
    size_frame = frame.copy()
    size_frame["face_size_tercile"] = pd.qcut(
        size_frame.face_area_fraction, 3, labels=["small", "medium", "large"]
    )
    size_means = size_frame.pivot_table(
        index="face_size_tercile",
        columns="arm",
        values="id_intended",
        aggfunc="mean",
        observed=True,
    )
    size_means.to_csv(args.output_root / "face_size_means.csv")

    render_identity_summary(aggregate, args.output_root / "identity_summary.png")
    render_effects(effects_frame, args.output_root / "paired_effects.png")
    render_all70_scatter(frame, args.output_root / "all70_scatter.png")
    render_identity_strata(frame, args.output_root / "identity_strata.png")
    representatives = render_representatives(
        frame,
        args.task_root,
        args.reference_root,
        args.output_root / "representative_crossing_grid.png",
    )

    pivot = frame.pivot(index="output_key", columns="arm", values="id_intended")
    wrong_pivot = frame.pivot(index="output_key", columns="arm", values="id_wrong")
    margin_pivot = frame.pivot(index="output_key", columns="arm", values="id_margin")
    summary = {
        "protocol": "manual_val fixed-96, seed 0, one image per cell",
        "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
        "primary_metric": "subject-v2 identity similarity matched to the fixed generated-face box",
        "arms": {arm: {"label": ARM_LABELS[arm], "run_name": ARMS[arm][0], "step": ARMS[arm][1]} for arm in ARMS},
        "aggregate": aggregate.to_dict("records"),
        "paired_effects": effects,
        "factorial_interaction_intended_id": float(
            ((pivot.A - pivot.B) - (pivot.C - pivot.D)).mean()
        ),
        "factorial_interaction_wrong_id": float(
            ((wrong_pivot.A - wrong_pivot.B) - (wrong_pivot.C - wrong_pivot.D)).mean()
        ),
        "factorial_interaction_identity_margin": float(
            ((margin_pivot.A - margin_pivot.B) - (margin_pivot.C - margin_pivot.D)).mean()
        ),
        "representative_cells": representatives,
        "join_contract": "normalize spaces to underscores on both metadata and PNG keys; 96/96 cells joined in every arm",
        "limitations": [
            "The 16k all-70 comparison and 24k crossing are matched only within checkpoint, never across steps.",
            "This is one fixed seed and cannot establish training-seed or population uncertainty.",
            "RGB distance and SSIM measure intervention size, not image quality.",
            "Prompt/text and seven-curve face-quality metrics were disabled for these validation-only jobs.",
        ],
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
