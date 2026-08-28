#!/usr/bin/env python3
"""Paired fixed-box analysis for CL39 BA start 10 versus accepted start 15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from tools.analysis.analyze_cl39_attribution_controls import distance, score_face


BOOTSTRAP_SEED = 391010
BOOTSTRAP_DRAWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--subject-v2-embeds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument("--grid-limit", type=int, default=12)
    return parser.parse_args()


def normalize_key(value: str) -> str:
    # AICODE-NOTE: bbox metadata contains spaces that validation PNG names
    # replace with underscores; literal joining silently drops valid cells.
    return value.replace(" ", "_")


def bootstrap(delta: np.ndarray) -> dict[str, float | int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = delta[rng.integers(0, len(delta), (BOOTSTRAP_DRAWS, len(delta)))].mean(axis=1)
    return {
        "n": int(len(delta)),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "wins": int((delta > 1.0e-12).sum()),
        "ties": int((np.abs(delta) <= 1.0e-12).sum()),
        "losses": int((delta < -1.0e-12).sum()),
        "bootstrap_95_low": float(np.quantile(means, 0.025)),
        "bootstrap_95_high": float(np.quantile(means, 0.975)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def image_by_name(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename!r} below {root}, found {matches}")
    return matches[0]


def select_grid_filenames(frame: pd.DataFrame, limit: int) -> list[str]:
    filenames = sorted(frame.filename.unique())
    if len(filenames) <= limit:
        return filenames
    pivot = frame.pivot(index="filename", columns="arm", values="id_intended")
    delta = pivot.start10 - pivot.start15
    lower_count = limit // 2
    upper_count = limit - lower_count
    return list(delta.nsmallest(lower_count).index) + list(delta.nlargest(upper_count).index)


def render_grid(
    frame: pd.DataFrame,
    baseline_root: Path,
    candidate_root: Path,
    output_root: Path,
    filenames: list[str],
) -> None:
    for page, start in enumerate(range(0, len(filenames), 6), 1):
        selected = filenames[start : start + 6]
        fig, axes = plt.subplots(len(selected), 2, figsize=(8.4, 3.7 * len(selected)), squeeze=False)
        for row_index, filename in enumerate(selected):
            bbox = json.loads(frame.loc[frame.filename == filename, "bbox"].iloc[0])
            for column_index, (arm, root, title) in enumerate(
                (
                    ("start15", baseline_root, "Original: PM@10, BA@15"),
                    ("start10", candidate_root, "Candidate: PM+BA@10"),
                )
            ):
                image = Image.open(image_by_name(root, filename)).convert("RGB")
                row = frame[(frame.filename == filename) & (frame.arm == arm)].iloc[0]
                ax = axes[row_index, column_index]
                ax.imshow(image)
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
                    f"{title}\nID {row.id_intended:.3f}; wrong {row.id_wrong:.3f}; margin {row.id_margin:+.3f}",
                    fontsize=9,
                )
                ax.axis("off")
            axes[row_index, 0].set_ylabel(filename.rsplit("_", 1)[0], fontsize=8)
        fig.suptitle(f"CL39 24k BA-start paired comparison — page {page}", fontsize=14)
        fig.savefig(output_root / f"paired_grid_{page}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def render_effects(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot(index="filename", columns="arm", values=["id_intended", "id_wrong", "id_margin"])
    deltas = pd.DataFrame(
        {
            "intended ID": pivot.id_intended.start10 - pivot.id_intended.start15,
            "wrong ID": pivot.id_wrong.start10 - pivot.id_wrong.start15,
            "ID margin": pivot.id_margin.start10 - pivot.id_margin.start15,
        }
    )
    means = deltas.mean()
    colors = ["#0b6e99" if value >= 0 else "#c44e52" for value in means]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    bars = ax.bar(means.index, means.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Candidate − original cosine")
    ax.set_title(f"CL39 BA start 10: paired identity effects (n={len(deltas)})")
    for bar, value in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.4f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_identity_effects(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot_table(
        index="identity", columns="arm", values=["id_intended", "id_margin"], aggfunc="mean"
    )
    values = pd.DataFrame(
        {
            "intended ID": pivot.id_intended.start10 - pivot.id_intended.start15,
            "ID margin": pivot.id_margin.start10 - pivot.id_margin.start15,
        }
    )
    x = np.arange(len(values))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, values["intended ID"], width, label="intended ID")
    ax.bar(x + width / 2, values["ID margin"], width, label="ID margin")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, values.index, rotation=20, ha="right")
    ax.set_ylabel("Candidate − original cosine")
    ax.set_title("CL39 BA start 10: identity-stratified paired effects")
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline_paths = sorted(args.baseline_root.rglob("*.png"), key=lambda path: path.name)
    candidate_paths = sorted(args.candidate_root.rglob("*.png"), key=lambda path: path.name)
    if len(baseline_paths) != args.expected_count or len(candidate_paths) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count}+{args.expected_count} images, "
            f"found {len(baseline_paths)}+{len(candidate_paths)}"
        )
    if [path.name for path in baseline_paths] != [path.name for path in candidate_paths]:
        raise SystemExit("Paired filename gate failed")

    bbox_data = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    by_normalized = {normalize_key(key): (key, value) for key, value in bbox_data.items()}
    if len(by_normalized) != len(bbox_data):
        raise SystemExit("Space/underscore normalization is not one-to-one")
    identities = sorted({key.removesuffix(".png").rsplit("_", 1)[-1] for key in bbox_data})
    wrong_by_id = {identity: identities[(index + 1) % len(identities)] for index, identity in enumerate(identities)}

    from src.metrics.id_sim_metric import IDSimMaskMatched

    metric = IDSimMaskMatched(
        id_embeds_pth=str(args.subject_v2_embeds),
        device=args.device,
        metric_name="id_sim_subject_v2",
    )
    rows = []
    for baseline_path, candidate_path in zip(baseline_paths, candidate_paths):
        metadata_key, metadata = by_normalized[baseline_path.name]
        bbox = metadata.get("face_crop_new") or metadata.get("face_crop_old")
        identity = metadata_key.removesuffix(".png").rsplit("_", 1)[-1]
        wrong = wrong_by_id[identity]
        images = {
            "start15": Image.open(baseline_path).convert("RGB"),
            "start10": Image.open(candidate_path).convert("RGB"),
        }
        pixel = distance(images["start10"], images["start15"], bbox)
        for arm, image in images.items():
            rows.append(
                {
                    "filename": baseline_path.name,
                    "metadata_key": metadata_key,
                    "identity": identity,
                    "wrong_identity": wrong,
                    "prompt_key": metadata_key.removesuffix(".png").rsplit("_", 1)[0].strip(),
                    "arm": arm,
                    "bbox": json.dumps(bbox),
                    **score_face(metric, image, bbox, identity, wrong),
                    **({f"candidate_vs_original_{key}": value for key, value in pixel.items()} if arm == "start10" else {}),
                }
            )
        print(f"scored {baseline_path.name}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_image.csv", index=False)
    aggregate = frame.groupby("arm", sort=False).agg(
        id_intended=("id_intended", "mean"),
        id_wrong=("id_wrong", "mean"),
        id_margin=("id_margin", "mean"),
        mask_iou=("mask_iou", "mean"),
        face_count=("face_count", "mean"),
        no_face=("no_face", "sum"),
        unowned=("unowned", "sum"),
        ambiguous=("ambiguous", "sum"),
    ).reset_index()
    aggregate.to_csv(args.output_root / "aggregate.csv", index=False)
    pivot = frame.pivot(index="filename", columns="arm", values=["id_intended", "id_wrong", "id_margin"])
    effects = {
        metric_name: bootstrap((pivot[metric_name].start10 - pivot[metric_name].start15).to_numpy())
        for metric_name in ("id_intended", "id_wrong", "id_margin")
    }
    candidate_rows = frame[frame.arm == "start10"]
    pixel_metrics = {
        column.removeprefix("candidate_vs_original_"): float(candidate_rows[column].mean())
        for column in candidate_rows.columns
        if column.startswith("candidate_vs_original_")
    }
    identity_effects = frame.pivot_table(
        index="identity", columns="arm", values=["id_intended", "id_wrong", "id_margin"], aggfunc="mean"
    )
    identity_effects.to_csv(args.output_root / "identity_means.csv")
    prompt_effects = frame.pivot_table(
        index="prompt_key", columns="arm", values=["id_intended", "id_wrong", "id_margin"], aggfunc="mean"
    )
    prompt_effects.to_csv(args.output_root / "prompt_means.csv")
    grid_filenames = select_grid_filenames(frame, args.grid_limit)
    identity_count = int(frame.identity.nunique())
    summary = {
        "protocol": (
            f"CL39 24k, manual_val {args.expected_count} cells, seed 0, "
            "fixed generated-face boxes"
        ),
        "primary_metric": "subject-v2 intended-identity cosine matched to the fixed generated-face box",
        "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
        "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
        "arms": {
            "start15": {"photomaker_start_step": 10, "branched_attn_start_step": 15, "branched_active_steps": 35},
            "start10": {"photomaker_start_step": 10, "branched_attn_start_step": 10, "branched_active_steps": 40},
        },
        "aggregate": aggregate.to_dict("records"),
        "paired_effects_candidate_minus_original": effects,
        "pixel_change_candidate_vs_original": pixel_metrics,
        "identity_count": identity_count,
        "selected_visual_filenames": grid_filenames,
        "join_contract": "exact PNG filename pairing; bbox metadata normalized with space-to-underscore replacement",
        "limitations": [
            (
                f"The {args.expected_count} cells cover {identity_count} identities but only one generation "
                "seed, so intervals quantify fixed-cell resampling only."
            ),
            "This validation-only comparison does not estimate retraining effects.",
            "Prompt-adherence and no-reference perceptual-quality metrics were not computed in this quick run.",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_effects(frame, args.output_root / "paired_identity_effects.png")
    render_identity_effects(frame, args.output_root / "identity_effects.png")
    render_grid(
        frame,
        args.baseline_root,
        args.candidate_root,
        args.output_root,
        grid_filenames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
