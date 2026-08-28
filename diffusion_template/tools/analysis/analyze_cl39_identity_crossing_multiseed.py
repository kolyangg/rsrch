#!/usr/bin/env python3
"""Score the CL39 fixed-96 A/B/C/D identity crossing across seeds 0-3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from tools.analysis.analyze_cl39_attribution_controls import normalize_key, score_face


ARMS = {
    "A": "correct PM / correct spatial",
    "B": "correct PM / wrong spatial",
    "C": "wrong PM / correct spatial",
    "D": "wrong PM / wrong spatial",
}
CONTRASTS = {
    "A_minus_B_spatial_given_correct_PM": ("A", "B"),
    "C_minus_D_spatial_given_wrong_PM": ("C", "D"),
    "A_minus_C_PM_given_correct_spatial": ("A", "C"),
    "B_minus_D_PM_given_wrong_spatial": ("B", "D"),
}
METRICS = ("id_intended", "id_wrong", "id_margin")
CELL_BOOTSTRAP_SEED = 390426
TWO_WAY_BOOTSTRAP_SEED = 390427
PER_SEED_BOOTSTRAP_BASE = 390100
BOOTSTRAP_DRAWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-csv", type=Path, required=True)
    parser.add_argument("--seed0-task-root", type=Path, required=True)
    parser.add_argument("--new-task-root", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--subject-v2-embeds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_name(seed: int, arm: str) -> str:
    if seed == 0:
        return {
            "A": "CL39_24k_cross_A_correct_pm_correct_spatial_r1",
            "B": "CL39_24k_cross_B_correct_pm_wrong_spatial_r1",
            "C": "CL39_24k_cross_C_wrong_pm_correct_spatial_r1",
            "D": "CL39_24k_cross_D_wrong_pm_wrong_spatial_r1",
        }[arm]
    slug = {
        "A": "correct_pm_correct_spatial",
        "B": "correct_pm_wrong_spatial",
        "C": "wrong_pm_correct_spatial",
        "D": "wrong_pm_wrong_spatial",
    }[arm]
    return f"CL39_24k_seed{seed}_cross_{arm}_{slug}_r1"


def image_path(task_root: Path, seed: int, arm: str, output_key: str) -> Path:
    matches = list(
        (task_root / "saved" / run_name(seed, arm) / "val_images" / "manual_val").glob(
            f"step_24000_batch_*/{normalize_key(output_key)}"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one image for seed={seed} arm={arm} key={output_key!r}; found {matches}"
        )
    return matches[0]


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def two_way_bootstrap(matrix: np.ndarray) -> tuple[float, float]:
    """Resample inference seeds and fixed-panel cells independently."""
    rng = np.random.default_rng(TWO_WAY_BOOTSTRAP_SEED)
    outputs = []
    for start in range(0, BOOTSTRAP_DRAWS, 5_000):
        count = min(5_000, BOOTSTRAP_DRAWS - start)
        seed_indices = rng.integers(0, matrix.shape[0], size=(count, matrix.shape[0]))
        cell_indices = rng.integers(0, matrix.shape[1], size=(count, matrix.shape[1]))
        seed_sample = matrix[seed_indices]
        sample = np.take_along_axis(seed_sample, cell_indices[:, None, :], axis=2)
        outputs.append(sample.mean(axis=(1, 2)))
    draws = np.concatenate(outputs)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def effect_rows(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for metric in METRICS:
        pivot = frame.pivot(index=["seed", "output_key"], columns="arm", values=metric)
        for contrast, (first, second) in CONTRASTS.items():
            delta = (pivot[first] - pivot[second]).unstack("output_key").sort_index()
            if delta.shape != (4, 96) or delta.isna().any().any():
                raise RuntimeError(f"Incomplete matrix for {metric} {contrast}: {delta.shape}")
            for seed, values in delta.iterrows():
                array = values.to_numpy()
                low, high = bootstrap_mean(array, PER_SEED_BOOTSTRAP_BASE + int(seed))
                records.append(
                    {
                        "metric": metric,
                        "contrast": contrast,
                        "scope": f"seed{seed}",
                        "n_seeds": 1,
                        "n_cells": 96,
                        "mean": float(array.mean()),
                        "median": float(np.median(array)),
                        "wins": int((array > 1.0e-12).sum()),
                        "ties": int((np.abs(array) <= 1.0e-12).sum()),
                        "losses": int((array < -1.0e-12).sum()),
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                        "bootstrap_method": "fixed-cell",
                        "bootstrap_seed": PER_SEED_BOOTSTRAP_BASE + int(seed),
                    }
                )
            cell_means = delta.mean(axis=0).to_numpy()
            low, high = bootstrap_mean(cell_means, CELL_BOOTSTRAP_SEED)
            records.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "scope": "pooled_cell_mean",
                    "n_seeds": 4,
                    "n_cells": 96,
                    "mean": float(cell_means.mean()),
                    "median": float(np.median(cell_means)),
                    "wins": int((cell_means > 1.0e-12).sum()),
                    "ties": int((np.abs(cell_means) <= 1.0e-12).sum()),
                    "losses": int((cell_means < -1.0e-12).sum()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_method": "fixed-cell on four-seed cell means",
                    "bootstrap_seed": CELL_BOOTSTRAP_SEED,
                }
            )
            low, high = two_way_bootstrap(delta.to_numpy())
            records.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "scope": "pooled_two_way",
                    "n_seeds": 4,
                    "n_cells": 96,
                    "mean": float(delta.to_numpy().mean()),
                    "median": float(np.median(delta.to_numpy())),
                    "wins": int((cell_means > 1.0e-12).sum()),
                    "ties": int((np.abs(cell_means) <= 1.0e-12).sum()),
                    "losses": int((cell_means < -1.0e-12).sum()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_method": "two-way seed-and-cell",
                    "bootstrap_seed": TWO_WAY_BOOTSTRAP_SEED,
                }
            )
    return pd.DataFrame(records)


def render_effects(effects: pd.DataFrame, output: Path) -> None:
    contrasts = list(CONTRASTS)
    labels = ["A−B spatial | PM correct", "C−D spatial | PM wrong", "A−C PM | spatial correct", "B−D PM | spatial wrong"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS, ("Intended identity", "Wrong-identity attraction", "Identity margin")):
        selected = effects.query("metric == @metric")
        for index, (contrast, label) in enumerate(zip(contrasts, labels)):
            pooled = selected.query("contrast == @contrast and scope == 'pooled_two_way'").iloc[0]
            y = len(contrasts) - 1 - index
            ax.errorbar(
                pooled["mean"], y,
                xerr=[[pooled["mean"] - pooled["bootstrap_95_low"]], [pooled["bootstrap_95_high"] - pooled["mean"]]],
                fmt="D", color="#0b6e99", capsize=4, zorder=3,
            )
            seeds = selected.query("contrast == @contrast and scope.str.startswith('seed')", engine="python")
            ax.scatter(seeds["mean"], np.full(len(seeds), y) + np.linspace(-0.12, 0.12, len(seeds)), color="#c44e52", s=24, zorder=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(contrasts)), labels[::-1] if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlabel("first − second cosine")
    fig.suptitle("CL39 24k identity-source effects: seed means (red) and pooled two-way 95% intervals")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def representative_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pivot = frame.pivot(index=["seed", "output_key"], columns="arm", values="id_margin")
    ab = (pivot.A - pivot.B).unstack("output_key")
    cd = (pivot.C - pivot.D).unstack("output_key")
    robust = pd.concat([ab.min(axis=0), cd.min(axis=0)], axis=1).min(axis=1).idxmax()
    counter = ab.mean(axis=0).idxmin()
    box_miss = (
        frame.query("seed > 0").groupby("output_key")["unowned"].sum().idxmax()
    )
    return [
        ("robust spatial-margin gain", robust),
        ("lowest mean A−B margin", counter),
        ("fixed-box ownership stress case", box_miss),
    ]


def owned_pair_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Post-selection sensitivity only; primary analysis retains all 96 cells."""
    records = []
    for seed in range(4):
        pivot = frame.query("seed == @seed").pivot(
            index="output_key", columns="arm", values=[*METRICS, "unowned"]
        )
        for contrast, (first, second) in CONTRASTS.items():
            owned = (pivot["unowned"][first] == 0) & (pivot["unowned"][second] == 0)
            for metric in METRICS:
                delta = (pivot[metric][first] - pivot[metric][second])[owned]
                records.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "contrast": contrast,
                        "both_owned_cells": int(owned.sum()),
                        "mean": float(delta.mean()),
                        "wins": int((delta > 1.0e-12).sum()),
                        "ties": int((np.abs(delta) <= 1.0e-12).sum()),
                        "losses": int((delta < -1.0e-12).sum()),
                    }
                )
    return pd.DataFrame(records)


def render_grid(
    frame: pd.DataFrame,
    seed0_root: Path,
    new_root: Path,
    reference_root: Path,
    label: str,
    output_key: str,
    output: Path,
) -> None:
    columns = ["target ref", "wrong ref", "A", "B", "C", "D"]
    subset = frame.query("output_key == @output_key").set_index(["seed", "arm"])
    identity = str(subset.identity.iloc[0])
    wrong = str(subset.wrong_identity.iloc[0])
    bbox = json.loads(subset.bbox.iloc[0])
    target_ref = Image.open(next(reference_root.glob(f"{identity}.*"))).convert("RGB")
    wrong_ref = Image.open(next(reference_root.glob(f"{wrong}.*"))).convert("RGB")
    fig, axes = plt.subplots(4, 6, figsize=(15, 12.5), constrained_layout=True)
    for seed in range(4):
        root = seed0_root if seed == 0 else new_root
        images = [target_ref, wrong_ref] + [Image.open(image_path(root, seed, arm, output_key)).convert("RGB") for arm in ARMS]
        for column_index, (column, image) in enumerate(zip(columns, images)):
            ax = axes[seed, column_index]
            ax.imshow(image)
            ax.axis("off")
            if column_index >= 2:
                score = subset.loc[(seed, column)]
                ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor="#ff3b30", linewidth=2))
                ax.set_title(f"{column}: ID✓ {score.id_intended:.3f}\nID✗ {score.id_wrong:.3f}; margin {score.id_margin:.3f}", fontsize=8)
            elif seed == 0:
                ax.set_title(column, fontsize=9)
        axes[seed, 0].set_ylabel(f"seed {seed}", fontsize=10)
    fig.suptitle(f"{label}: {identity}, {output_key.rsplit('_', 1)[0]}\nred = immutable generated-face box")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    bbox_data = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    if len({normalize_key(key) for key in bbox_data}) != 96:
        raise RuntimeError("Expected 96 one-to-one normalized bbox keys")
    identities = sorted(key.removesuffix(".png").rsplit("_", 1)[-1] for key in bbox_data)
    identities = sorted(set(identities))
    wrong_by_id = {identity: identities[(index + 1) % len(identities)] for index, identity in enumerate(identities)}

    seed0 = pd.read_csv(args.seed0_csv).query("arm in ['A', 'B', 'C', 'D']").copy()
    seed0["seed"] = 0
    keep = ["cell_index", "output_key", "normalized_output_key", "identity", "wrong_identity", "prompt_key", "bbox", "face_area_fraction", "arm", "run_name", "step", *METRICS, "mask_iou", "face_count", "no_face", "unowned", "ambiguous", "seed"]
    seed0 = seed0[keep]

    final_path = args.output_root / "per_image.csv"
    partial_path = args.output_root / "new_seed_rows.partial.csv"
    if final_path.exists():
        existing = pd.read_csv(final_path)
        new_rows = existing.query("seed > 0").to_dict("records")
    elif partial_path.exists():
        new_rows = pd.read_csv(partial_path).to_dict("records")
    else:
        new_rows = []
    completed = {(int(row["seed"]), str(row["arm"]), str(row["output_key"])) for row in new_rows}

    from src.metrics.id_sim_metric import IDSimMaskMatched

    metric = IDSimMaskMatched(id_embeds_pth=str(args.subject_v2_embeds), device=args.device, metric_name="id_sim_subject_v2")
    for seed in (1, 2, 3):
        for arm in ARMS:
            for cell_index, (output_key, bbox_record) in enumerate(bbox_data.items()):
                if (seed, arm, output_key) in completed:
                    continue
                identity = output_key.removesuffix(".png").rsplit("_", 1)[-1]
                wrong = wrong_by_id[identity]
                bbox = bbox_record.get("face_crop_new") or bbox_record.get("face_crop_old")
                image = Image.open(image_path(args.new_task_root, seed, arm, output_key)).convert("RGB")
                score = score_face(metric, image, bbox, identity, wrong)
                new_rows.append(
                    {
                        "cell_index": cell_index,
                        "output_key": output_key,
                        "normalized_output_key": normalize_key(output_key),
                        "identity": identity,
                        "wrong_identity": wrong,
                        "prompt_key": output_key.removesuffix(".png").rsplit("_", 1)[0].strip(),
                        "bbox": json.dumps(bbox),
                        "face_area_fraction": ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (1024.0 * 1024.0),
                        "arm": arm,
                        "run_name": run_name(seed, arm),
                        "step": 24000,
                        **score,
                        "seed": seed,
                    }
                )
                if len(new_rows) % 12 == 0:
                    pd.DataFrame(new_rows).to_csv(partial_path, index=False)
                    print(f"scored {len(new_rows)}/1152 new images", flush=True)

    new_frame = pd.DataFrame(new_rows)[keep]
    frame = pd.concat([seed0, new_frame], ignore_index=True)
    if len(frame) != 1536 or frame.groupby(["seed", "arm"]).size().nunique() != 1:
        raise RuntimeError(f"Expected 1536 rows and balanced cells; got {len(frame)}")
    frame.to_csv(args.output_root / "per_image.csv", index=False)
    aggregate = frame.groupby(["seed", "arm"], sort=True)[[*METRICS, "mask_iou", "face_count", "no_face", "unowned", "ambiguous"]].mean().reset_index()
    aggregate.to_csv(args.output_root / "aggregate_by_seed.csv", index=False)
    effects = effect_rows(frame)
    effects.to_csv(args.output_root / "paired_effects.csv", index=False)
    sensitivity = owned_pair_sensitivity(frame)
    sensitivity.to_csv(args.output_root / "owned_pair_sensitivity.csv", index=False)
    render_effects(effects, args.output_root / "paired_effects_multiseed.png")

    representatives = []
    for index, (label, output_key) in enumerate(representative_keys(frame), 1):
        slug = ("robust", "counterexample", "ownership_stress")[index - 1]
        filename = f"representative_{index}_{slug}.png"
        render_grid(frame, args.seed0_task_root, args.new_task_root, args.reference_root, label, output_key, args.output_root / filename)
        representatives.append({"selection": label, "output_key": output_key, "figure": filename})

    summary = {
        "protocol": "manual_val fixed-96, inference seeds 0-3, one image per cell per seed",
        "immutable_parent_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a",
        "checkpoint_sha256": "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07",
        "source_manifest_sha256": "9566862387800eded64c8972461b873ddd9ac9c86fd1cd27ae23425a27a2d10f",
        "primary_metric": "subject-v2 identity cosine matched to the fixed generated-face box",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "per_seed_base": PER_SEED_BOOTSTRAP_BASE, "pooled_cell_seed": CELL_BOOTSTRAP_SEED, "two_way_seed": TWO_WAY_BOOTSTRAP_SEED},
        "aggregate_by_seed": aggregate.to_dict("records"),
        "effects": effects.to_dict("records"),
        "owned_pair_sensitivity": sensitivity.to_dict("records"),
        "representatives": representatives,
        "join_contract": "replace spaces with underscores; exactly 96/96 cells per seed and arm",
        "limitations": [
            "Seeds 0-3 are inference seeds on one training checkpoint, not independent training seeds.",
            "Only four predeclared inference seeds are available; two-way intervals describe finite-seed robustness, not population training uncertainty.",
            "Identity cosine and face ownership do not measure prompt adherence or overall visual quality.",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(json.dumps({"rows": len(frame), "aggregate": aggregate.to_dict("records"), "representatives": representatives}, indent=2))


if __name__ == "__main__":
    main()
