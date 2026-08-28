#!/usr/bin/env python3
"""Score the corrected CL39 crossing with one automatic face box per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from tools.analysis.analyze_cl39_attribution_controls import normalize_key, score_face
from tools.analysis.analyze_cl39_identity_crossing_multiseed import (
    ARMS,
    CONTRASTS,
    METRICS,
    BOOTSTRAP_DRAWS,
    CELL_BOOTSTRAP_SEED,
    PER_SEED_BOOTSTRAP_BASE,
    TWO_WAY_BOOTSTRAP_SEED,
    effect_rows,
    owned_pair_sensitivity,
    render_effects,
)


SOURCE_MANIFEST_SHA256 = "e1022d515296892ab6c46a36a51db37bcf9dead4798c30502660138b5b0d7643"
CHECKPOINT_SHA256 = "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
PARENT_COMET_KEY = "b1ca0b3da679401c85b991f1bbdf0b2a"
JOBS = {
    1: "lm-mpi-job-ee43b350-de5c-44e3-9cab-d694e9f5806e",
    2: "lm-mpi-job-9c599a15-1d97-49a9-8609-81f38d03ca85",
    3: "lm-mpi-job-f04b6ebf-aded-4da2-ad5d-206a65534f15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-csv", type=Path, required=True)
    parser.add_argument("--seed0-task-root", type=Path, required=True)
    parser.add_argument("--new-task-root", type=Path, required=True)
    parser.add_argument("--flawed-csv", type=Path, required=True)
    parser.add_argument("--flawed-task-root", type=Path, required=True)
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
    return f"CL39_24k_seed{seed}_cross_{arm}_{slug}_dynamic_bbox_r1"


def flawed_run_name(seed: int, arm: str) -> str:
    slug = {
        "A": "correct_pm_correct_spatial",
        "B": "correct_pm_wrong_spatial",
        "C": "wrong_pm_correct_spatial",
        "D": "wrong_pm_wrong_spatial",
    }[arm]
    return f"CL39_24k_seed{seed}_cross_{arm}_{slug}_r1"


def image_path(task_root: Path, run: str, output_key: str) -> Path:
    matches = list(
        (task_root / "saved" / run / "val_images" / "manual_val").glob(
            f"step_24000_batch_*/{normalize_key(output_key)}"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one image for {run} {output_key!r}; found {matches}")
    return matches[0]


def bbox_contract(task_root: Path) -> tuple[dict[int, dict], dict[int, str]]:
    boxes: dict[int, dict] = {}
    hashes: dict[int, str] = {}
    normalized_sets = []
    for seed in (1, 2, 3):
        path = task_root / "dynamic_bboxes" / f"seed{seed}" / f"pm96_bboxes_seed{seed}_auto.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data) != 96 or {int((v.get("_meta") or {}).get("seed", -1)) for v in data.values()} != {seed}:
            raise RuntimeError(f"Invalid automatic-box contract for seed {seed}")
        boxes[seed] = data
        hashes[seed] = hashlib.sha256(path.read_bytes()).hexdigest()
        normalized_sets.append({normalize_key(key) for key in data})
    if not all(keys == normalized_sets[0] for keys in normalized_sets[1:]):
        raise RuntimeError("Seed-specific automatic-box files do not cover the same 96 cells")
    if len(set(hashes.values())) != 3:
        raise RuntimeError("Expected three distinct seed-specific automatic-box hashes")
    return boxes, hashes


def render_corrected_vs_flawed_effects(corrected: pd.DataFrame, flawed: pd.DataFrame, output: Path) -> None:
    contrasts = list(CONTRASTS)
    labels = ["A−B\nspatial | PM✓", "C−D\nspatial | PM✗", "A−C\nPM | spatial✓", "B−D\nPM | spatial✗"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ("id_intended", "id_margin"), ("Intended identity", "Identity margin")):
        old = flawed.query("metric == @metric and scope == 'pooled_two_way'").set_index("contrast")
        new = corrected.query("metric == @metric and scope == 'pooled_two_way'").set_index("contrast")
        x = np.arange(len(contrasts))
        width = 0.34
        ax.bar(x - width / 2, [old.loc[c, "mean"] for c in contrasts], width, label="invalid static seed-0 mask", color="#a6a6a6")
        ax.bar(x + width / 2, [new.loc[c, "mean"] for c in contrasts], width, label="corrected seed-specific mask", color="#0b6e99")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("first − second cosine")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Corrected versus invalid CL39 multiseed crossing (seeds 0–3 pooled)")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_corrected_grid(
    frame: pd.DataFrame,
    seed0_root: Path,
    new_root: Path,
    reference_root: Path,
    output_key: str,
    label: str,
    output: Path,
) -> None:
    subset = frame.query("output_key == @output_key").set_index(["seed", "arm"])
    identity = str(subset.identity.iloc[0])
    wrong = str(subset.wrong_identity.iloc[0])
    target_ref = Image.open(next(reference_root.glob(f"{identity}.*"))).convert("RGB")
    wrong_ref = Image.open(next(reference_root.glob(f"{wrong}.*"))).convert("RGB")
    columns = ["target ref", "next ref", "A", "B", "C", "D"]
    fig, axes = plt.subplots(4, 6, figsize=(15, 12.5), constrained_layout=True)
    for seed in range(4):
        root = seed0_root if seed == 0 else new_root
        images = [target_ref, wrong_ref] + [Image.open(image_path(root, run_name(seed, arm), output_key)).convert("RGB") for arm in ARMS]
        for column_index, (column, image) in enumerate(zip(columns, images)):
            ax = axes[seed, column_index]
            ax.imshow(image)
            ax.axis("off")
            if column_index >= 2:
                score = subset.loc[(seed, column)]
                bbox = json.loads(score.bbox)
                ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor="#ff3b30", linewidth=2))
                ax.set_title(f"{column}: ID✓ {score.id_intended:.3f}\nID✗ {score.id_wrong:.3f}; margin {score.id_margin:.3f}", fontsize=8)
            elif seed == 0:
                ax.set_title(column, fontsize=9)
        axes[seed, 0].set_ylabel(f"seed {seed}", fontsize=10)
    fig.suptitle(f"{label}: {identity}, {output_key.rsplit('_', 1)[0]}\nred = accepted generated-face box for that inference seed")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_old_new_grid(
    corrected: pd.DataFrame,
    flawed: pd.DataFrame,
    new_root: Path,
    flawed_root: Path,
    output_key: str,
    output: Path,
) -> None:
    new_subset = corrected.query("seed > 0 and output_key == @output_key").set_index(["seed", "arm"])
    old_subset = flawed.query("seed > 0 and output_key == @output_key").set_index(["seed", "arm"])
    fig, axes = plt.subplots(3, 8, figsize=(20, 8.2), constrained_layout=True)
    for row, seed in enumerate((1, 2, 3)):
        for pair_index, arm in enumerate(ARMS):
            for version_index, (version, subset, root, run) in enumerate((
                ("invalid", old_subset, flawed_root, flawed_run_name(seed, arm)),
                ("corrected", new_subset, new_root, run_name(seed, arm)),
            )):
                ax = axes[row, pair_index * 2 + version_index]
                image = Image.open(image_path(root, run, output_key)).convert("RGB")
                ax.imshow(image)
                ax.axis("off")
                score = subset.loc[(seed, arm)]
                bbox = json.loads(score.bbox)
                ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor="#ff3b30", linewidth=2))
                ax.set_title(f"{arm} {version}\nmargin {score.id_margin:.3f}", fontsize=8)
        axes[row, 0].set_ylabel(f"seed {seed}", fontsize=10)
    fig.suptitle(f"Largest corrected-versus-invalid score change: {output_key}\nred = scoring/generation mask used by that version")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    boxes_by_seed, bbox_hashes = bbox_contract(args.new_task_root)
    identities = sorted({key.removesuffix(".png").rsplit("_", 1)[-1] for key in boxes_by_seed[1]})
    wrong_by_id = {identity: identities[(index + 1) % len(identities)] for index, identity in enumerate(identities)}

    seed0 = pd.read_csv(args.seed0_csv).query("arm in ['A', 'B', 'C', 'D']").copy()
    seed0["seed"] = 0
    seed0["bbox_seed"] = 0
    seed0["bbox_sha256"] = "seed0_accepted_box_embedded_in_seed0_csv"
    base_keep = [
        "cell_index", "output_key", "normalized_output_key", "identity", "wrong_identity", "prompt_key",
        "bbox", "face_area_fraction", "arm", "run_name", "step", *METRICS, "mask_iou", "face_count",
        "no_face", "unowned", "ambiguous", "seed", "bbox_seed", "bbox_sha256",
    ]
    seed0 = seed0[base_keep]

    final_path = args.output_root / "per_image.csv"
    partial_path = args.output_root / "new_seed_rows.partial.csv"
    new_rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {(int(row["seed"]), str(row["arm"]), str(row["output_key"])) for row in new_rows}

    from src.metrics.id_sim_metric import IDSimMaskMatched

    metric = IDSimMaskMatched(id_embeds_pth=str(args.subject_v2_embeds), device=args.device, metric_name="id_sim_subject_v2")
    for seed in (1, 2, 3):
        bbox_data = boxes_by_seed[seed]
        for arm in ARMS:
            for cell_index, (output_key, bbox_record) in enumerate(bbox_data.items()):
                if (seed, arm, output_key) in completed:
                    continue
                identity = output_key.removesuffix(".png").rsplit("_", 1)[-1]
                wrong = wrong_by_id[identity]
                bbox = bbox_record.get("face_crop_new") or bbox_record.get("face_crop_old")
                image = Image.open(image_path(args.new_task_root, run_name(seed, arm), output_key)).convert("RGB")
                score = score_face(metric, image, bbox, identity, wrong)
                new_rows.append({
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
                    "bbox_seed": seed,
                    "bbox_sha256": bbox_hashes[seed],
                })
                if len(new_rows) % 12 == 0:
                    pd.DataFrame(new_rows).to_csv(partial_path, index=False)
                    print(f"scored {len(new_rows)}/1152 corrected images", flush=True)

    new_frame = pd.DataFrame(new_rows)[base_keep]
    frame = pd.concat([seed0, new_frame], ignore_index=True)
    if len(frame) != 1536 or not (frame.groupby(["seed", "arm"]).size() == 96).all():
        raise RuntimeError(f"Expected 1536 balanced rows, got {len(frame)}")
    frame.to_csv(final_path, index=False)
    aggregate = frame.groupby(["seed", "arm"], sort=True)[[*METRICS, "mask_iou", "face_count", "no_face", "unowned", "ambiguous"]].mean().reset_index()
    aggregate.to_csv(args.output_root / "aggregate_by_seed.csv", index=False)
    effects = effect_rows(frame)
    effects.to_csv(args.output_root / "paired_effects.csv", index=False)
    sensitivity = owned_pair_sensitivity(frame)
    sensitivity.to_csv(args.output_root / "owned_pair_sensitivity.csv", index=False)
    render_effects(effects, args.output_root / "paired_effects_multiseed_dynamic.png")

    flawed = pd.read_csv(args.flawed_csv)
    flawed_effects = effect_rows(flawed)
    effect_comparison = effects.merge(
        flawed_effects,
        on=["metric", "contrast", "scope"],
        suffixes=("_corrected", "_flawed"),
    )
    effect_comparison["corrected_minus_flawed_mean"] = effect_comparison["mean_corrected"] - effect_comparison["mean_flawed"]
    effect_comparison.to_csv(args.output_root / "corrected_vs_flawed_effects.csv", index=False)
    render_corrected_vs_flawed_effects(effects, flawed_effects, args.output_root / "corrected_vs_flawed_effects.png")

    merged = frame.merge(
        flawed[["seed", "arm", "output_key", *METRICS, "mask_iou", "unowned"]],
        on=["seed", "arm", "output_key"], suffixes=("_corrected", "_flawed"),
    )
    for metric_name in (*METRICS, "mask_iou", "unowned"):
        merged[f"delta_{metric_name}"] = merged[f"{metric_name}_corrected"] - merged[f"{metric_name}_flawed"]
    by_seed_arm = merged.groupby(["seed", "arm"], sort=True)[[c for c in merged if c.startswith("delta_")]].mean().reset_index()
    by_seed_arm.to_csv(args.output_root / "corrected_vs_flawed_by_seed_arm.csv", index=False)

    margin_pivot = frame.pivot(index=["seed", "output_key"], columns="arm", values="id_margin")
    spatial_floor = pd.concat([(margin_pivot.A - margin_pivot.B), (margin_pivot.C - margin_pivot.D)], axis=1).groupby("output_key").min().min(axis=1)
    robust_key = str(spatial_floor.idxmax())
    changed = merged.query("seed > 0").groupby("output_key")["delta_id_margin"].apply(lambda x: float(np.mean(np.abs(x))))
    changed_key = str(changed.idxmax())
    corrected_new = frame.query("seed > 0")
    ownership = corrected_new.groupby("output_key").agg(unowned=("unowned", "sum"), iou=("mask_iou", "mean"))
    worst_key = str(ownership.sort_values(["unowned", "iou"], ascending=[False, True]).index[0])

    render_corrected_grid(frame, args.seed0_task_root, args.new_task_root, args.reference_root, robust_key, "robust corrected spatial-margin gain", args.output_root / "representative_robust_corrected.png")
    render_old_new_grid(frame, flawed, args.new_task_root, args.flawed_task_root, changed_key, args.output_root / "representative_largest_correction.png")
    render_corrected_grid(frame, args.seed0_task_root, args.new_task_root, args.reference_root, worst_key, "lowest corrected ownership/alignment case", args.output_root / "representative_ownership_stress_corrected.png")

    alignment = []
    for seed in (1, 2, 3):
        gate = json.loads((args.new_task_root / "gates" / f"seed{seed}_dynamic_bbox_alignment.json").read_text(encoding="utf-8"))
        alignment.append({key: gate[key] for key in ("validation_seed", "bbox_count", "image_count", "no_face", "unowned", "mean_best_iou", "accepted")})
    summary = {
        "protocol": "manual_val fixed-96, inference seeds 0-3, one image per cell per seed; seed 1-3 use matched-seed PhotoMaker-only automatic boxes",
        "immutable_parent_comet_key": PARENT_COMET_KEY,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "jobs": JOBS,
        "bbox_sha256_by_seed": bbox_hashes,
        "alignment_gates": alignment,
        "primary_metric": "subject-v2 identity cosine matched to each inference seed's accepted generated-face box",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "per_seed_base": PER_SEED_BOOTSTRAP_BASE, "pooled_cell_seed": CELL_BOOTSTRAP_SEED, "two_way_seed": TWO_WAY_BOOTSTRAP_SEED},
        "aggregate_by_seed": aggregate.to_dict("records"),
        "effects": effects.to_dict("records"),
        "corrected_vs_flawed_effects": effect_comparison.to_dict("records"),
        "owned_pair_sensitivity": sensitivity.to_dict("records"),
        "representatives": {"robust": robust_key, "largest_correction": changed_key, "ownership_stress": worst_key},
        "join_contract": "normalize spaces to underscores; exactly 96/96 cells per seed and arm",
        "limitations": [
            "Seeds 0-3 are inference seeds on one trained checkpoint, not independent training seeds.",
            "The seed-specific boxes are generated from matched PhotoMaker-only images and then reused across A/B/C/D within that seed.",
            "Identity cosine and ownership do not measure prompt adherence or overall visual quality.",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(json.dumps({"rows": len(frame), "aggregate": aggregate.to_dict("records"), "representatives": summary["representatives"]}, indent=2))


if __name__ == "__main__":
    main()
