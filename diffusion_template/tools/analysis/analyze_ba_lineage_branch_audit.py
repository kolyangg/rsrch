#!/usr/bin/env python3
"""Score the sealed 16-cell CL19/23/27/39 branch-lineage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity


LINEAGES = {
    "CL19": {
        "actual": "BA_lineage_CL19_actual",
        "native": "BA_lineage_CL19_native",
    },
    "CL23": {
        "actual": "BA_lineage_CL23_actual",
        "native": "BA_lineage_CL23_native",
        "reference_face": "BA_lineage_CL23_reference_face",
        "low_only": "BA_lineage_CL23_low_only",
        "high_only": "BA_lineage_CL23_high_only",
    },
    "CL27": {
        "actual": "BA_lineage_CL27_actual",
        "native": "BA_lineage_CL27_native",
        "reference_face": "BA_lineage_CL27_reference_face",
        "low_only": "BA_lineage_CL27_low_only",
        "high_only": "BA_lineage_CL27_high_only",
    },
    "CL39": {
        "actual": "BA_lineage_CL39_actual",
        "native": "BA_lineage_CL39_native",
        "reference_face": "BA_lineage_CL39_reference_face",
        "low_only": "BA_lineage_CL39_low_only",
        "high_only": "BA_lineage_CL39_high_only",
        "confidence_one": "BA_lineage_CL39_confidence_one",
    },
}
ARM_ORDER = (
    "actual",
    "native",
    "reference_face",
    "low_only",
    "high_only",
    "confidence_one",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--subject-v2-embeds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _image_path(task_root: Path, run_name: str, record: dict) -> Path:
    index = int(record["index"])
    filename = str(record["face_bbox_gen_key"]).replace(" ", "_")
    path = (
        task_root
        / "saved"
        / run_name
        / "val_images"
        / "manual_val"
        / f"step_24000_batch_{index // 12}"
        / filename
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _crop(array: np.ndarray, bbox, padding: float = 0.25) -> np.ndarray:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width, height = x1 - x0, y1 - y0
    x0 = max(0, int(round(x0 - padding * width)))
    y0 = max(0, int(round(y0 - padding * height)))
    x1 = min(array.shape[1], int(round(x1 + padding * width)))
    y1 = min(array.shape[0], int(round(y1 + padding * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox {bbox} for shape {array.shape}")
    return array[y0:y1, x0:x1]


def _distance(candidate: np.ndarray, baseline: np.ndarray, bbox) -> dict[str, float]:
    difference = np.abs(candidate - baseline)
    candidate_face = _crop(candidate, bbox)
    baseline_face = _crop(baseline, bbox)
    face_difference = np.abs(candidate_face - baseline_face)
    return {
        "rgb_mae_all": float(difference.mean()),
        "rgb_mae_face": float(face_difference.mean()),
        "changed_all": float((difference.max(axis=2) > 1.0 / 255.0).mean()),
        "changed_face": float(
            (face_difference.max(axis=2) > 1.0 / 255.0).mean()
        ),
        "ssim_all": float(
            structural_similarity(
                candidate,
                baseline,
                data_range=1.0,
                channel_axis=2,
            )
        ),
    }


def _paired_summary(frame: pd.DataFrame, baseline: str, candidate: str) -> dict:
    pivot = frame.pivot(index="index", columns="arm", values="id_sim_subject_v2")
    if baseline not in pivot or candidate not in pivot:
        return {}
    delta = (pivot[baseline] - pivot[candidate]).dropna().to_numpy()
    rng = np.random.default_rng(390026)
    bootstrap = delta[
        rng.integers(0, len(delta), size=(100_000, len(delta)))
    ].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "wins": int(np.sum(delta > 1.0e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1.0e-12)),
        "losses": int(np.sum(delta < -1.0e-12)),
        "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_seed": 390026,
    }


def _render_heatmaps(aggregate: pd.DataFrame, output_path: Path) -> None:
    lineages = list(LINEAGES)
    arms = list(ARM_ORDER)
    id_values = np.full((len(lineages), len(arms)), np.nan)
    mae_values = np.full_like(id_values, np.nan)
    for row_index, lineage in enumerate(lineages):
        for column_index, arm in enumerate(arms):
            selected = aggregate[
                (aggregate.lineage == lineage) & (aggregate.arm == arm)
            ]
            if selected.empty:
                continue
            id_values[row_index, column_index] = selected.id_sim_subject_v2.iloc[0]
            mae_values[row_index, column_index] = selected.rgb_mae_face_vs_native.iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), constrained_layout=True)
    panels = (
        (axes[0], id_values, "Subject-v2 identity similarity (higher is better)", "viridis"),
        (axes[1], mae_values, "Face RGB MAE versus N-only (intervention strength)", "magma"),
    )
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(arms)), arms, rotation=25, ha="right")
        ax.set_yticks(range(len(lineages)), lineages)
        ax.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value < np.nanmedian(values) else "black",
                        fontsize=9,
                    )
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["samples"]
    if len(records) != 16:
        raise RuntimeError(f"Expected 16 sealed cells, found {len(records)}")

    from src.metrics.id_sim_metric import IDSimMaskMatched

    metric = IDSimMaskMatched(
        id_embeds_pth=str(args.subject_v2_embeds),
        device=args.device,
        metric_name="id_sim_subject_v2",
    )
    rows = []
    for record in records:
        reference = Image.open(
            args.reference_root / Path(record["reference_path"]).name
        ).convert("RGB")
        for lineage, arms in LINEAGES.items():
            images = {
                arm: Image.open(_image_path(args.task_root, run_name, record)).convert("RGB")
                for arm, run_name in arms.items()
            }
            arrays = {arm: _rgb(image) for arm, image in images.items()}
            native = arrays["native"]
            actual = arrays["actual"]
            reference_face = arrays.get("reference_face", actual)
            for arm, image in images.items():
                identity = metric(
                    generated=[image],
                    ref_images=[reference],
                    id=record["identity"],
                    face_bbox_gen=record["face_bbox_gen"],
                )
                vs_native = _distance(arrays[arm], native, record["face_bbox_gen"])
                vs_actual = _distance(arrays[arm], actual, record["face_bbox_gen"])
                vs_reference = _distance(
                    arrays[arm], reference_face, record["face_bbox_gen"]
                )
                rows.append(
                    {
                        "index": int(record["index"]),
                        "identity": record["identity"],
                        "prompt": record["prompt"],
                        "lineage": lineage,
                        "arm": arm,
                        **{key: float(value) for key, value in identity.items()},
                        **{f"{key}_vs_native": value for key, value in vs_native.items()},
                        **{f"{key}_vs_actual": value for key, value in vs_actual.items()},
                        **{
                            f"{key}_vs_reference_face": value
                            for key, value in vs_reference.items()
                        },
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "identity_and_distance_per_sample.csv", index=False)
    numeric = [
        column
        for column in frame.columns
        if column not in {"index", "identity", "prompt", "lineage", "arm"}
    ]
    aggregate = frame.groupby(["lineage", "arm"], sort=False)[numeric].mean().reset_index()
    aggregate.to_csv(args.output_root / "identity_and_distance_aggregate.csv", index=False)

    paired = {}
    for lineage, subset in frame.groupby("lineage", sort=False):
        paired[lineage] = {
            f"actual_minus_{arm}": _paired_summary(subset, "actual", arm)
            for arm in LINEAGES[lineage]
            if arm != "actual"
        }
    raw_reference = {
        "CL19/actual_trained_reference_route": float(
            aggregate.query("lineage == 'CL19' and arm == 'actual'")
            .id_sim_subject_v2.iloc[0]
        )
    }
    for lineage in ("CL23", "CL27", "CL39"):
        raw_reference[f"{lineage}/reference_face"] = float(
            aggregate.query(
                "lineage == @lineage and arm == 'reference_face'"
            ).id_sim_subject_v2.iloc[0]
        )
    summary = {
        "protocol": manifest["protocol"],
        "selection_seed": manifest["selection_seed"],
        "indices": manifest["indices"],
        "sample_count": len(records),
        "primary_metric": "id_sim_subject_v2, mask-matched to fixed face box",
        "aggregate": aggregate.to_dict("records"),
        "paired_actual_minus_counterfactual": paired,
        "raw_reference_route_identity_means": raw_reference,
        "important_limit": (
            "Every arm is a whole-denoising intervention. Low-only and high-only "
            "RGB images are nonlinear stress views and must not be added together."
        ),
    }
    (args.output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _render_heatmaps(aggregate, args.output_root / "lineage_metric_heatmaps.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
