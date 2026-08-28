#!/usr/bin/env python3
"""Render matched CL19/23/27/39 N, R, and frequency-route comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

DISPLAY_ARMS = (
    ("actual", "Actual"),
    ("native", "N-only"),
    ("reference_face", "Raw R-on-face"),
    ("low_only", "Low correction only"),
    ("high_only", "High correction only"),
    ("confidence_one", "Actual, C=1"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _crop(image: np.ndarray, bbox, *, padding: float = 0.25) -> np.ndarray:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width, height = x1 - x0, y1 - y0
    x0 = max(0, int(round(x0 - padding * width)))
    y0 = max(0, int(round(y0 - padding * height)))
    x1 = min(image.shape[1], int(round(x1 + padding * width)))
    y1 = min(image.shape[0], int(round(y1 + padding * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop bbox {bbox} for image shape {image.shape}")
    return image[y0:y1, x0:x1]


def _difference_metrics(
    candidate: np.ndarray, native: np.ndarray, bbox
) -> dict[str, float]:
    difference = np.abs(candidate - native)
    candidate_face = _crop(candidate, bbox)
    native_face = _crop(native, bbox)
    face_difference = np.abs(candidate_face - native_face)
    return {
        "rgb_mae_all_vs_native": float(difference.mean()),
        "rgb_mae_face_vs_native": float(face_difference.mean()),
        "pixel_changed_gt_1_255_all_vs_native": float(
            (difference.max(axis=2) > 1.0 / 255.0).mean()
        ),
        "pixel_changed_gt_1_255_face_vs_native": float(
            (face_difference.max(axis=2) > 1.0 / 255.0).mean()
        ),
    }


def _arm_images(task_root: Path, lineage: str, record: dict) -> dict[str, np.ndarray]:
    images = {
        arm: _rgb(_image_path(task_root, run_name, record))
        for arm, run_name in LINEAGES[lineage].items()
    }
    if lineage == "CL19":
        # CL19's trained operating point is already N + S(R-N).
        images["reference_face"] = images["actual"]
    return images


def _render_sample(
    task_root: Path,
    reference_root: Path,
    record: dict,
    output_path: Path,
) -> list[dict]:
    reference = _rgb(reference_root / Path(record["reference_path"]).name)
    reference_crop = _crop(reference, record["face_bbox_ref"])
    rows = []
    fig, axes = plt.subplots(
        len(LINEAGES),
        1 + len(DISPLAY_ARMS),
        figsize=(20, 11.5),
        constrained_layout=True,
    )
    for row_index, lineage in enumerate(LINEAGES):
        images = _arm_images(task_root, lineage, record)
        native = images["native"]
        axes[row_index, 0].imshow(reference_crop)
        axes[row_index, 0].set_ylabel(lineage, fontsize=12, fontweight="bold")
        axes[row_index, 0].axis("off")
        for column_index, (arm, _) in enumerate(DISPLAY_ARMS, start=1):
            ax = axes[row_index, column_index]
            image = images.get(arm)
            if image is None:
                ax.text(0.5, 0.5, "not applicable", ha="center", va="center")
                ax.set_facecolor("#eeeeee")
            else:
                ax.imshow(_crop(image, record["face_bbox_gen"]))
                metrics = _difference_metrics(
                    image, native, record["face_bbox_gen"]
                )
                rows.append(
                    {
                        "index": int(record["index"]),
                        "identity": record["identity"],
                        "prompt": record["prompt"],
                        "lineage": lineage,
                        "arm": arm,
                        **metrics,
                    }
                )
            ax.axis("off")
    titles = ["Identity reference"] + [title for _, title in DISPLAY_ARMS]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        f"BA lineage branch audit — cell {int(record['index']):02d}, "
        f"{record['identity']}\n{record['prompt']}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return rows


def _render_overviews(
    task_root: Path, records: list[dict], output_root: Path
) -> None:
    for lineage in LINEAGES:
        for page, start in enumerate(range(0, len(records), 4), start=1):
            subset = records[start : start + 4]
            fig, axes = plt.subplots(
                len(subset),
                len(DISPLAY_ARMS),
                figsize=(17, 3.1 * len(subset)),
                constrained_layout=True,
            )
            if len(subset) == 1:
                axes = axes[None, :]
            for row_index, record in enumerate(subset):
                images = _arm_images(task_root, lineage, record)
                for column_index, (arm, _) in enumerate(DISPLAY_ARMS):
                    ax = axes[row_index, column_index]
                    image = images.get(arm)
                    if image is None:
                        ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                        ax.set_facecolor("#eeeeee")
                    else:
                        ax.imshow(_crop(image, record["face_bbox_gen"]))
                    ax.axis("off")
                axes[row_index, 0].set_ylabel(
                    f"{int(record['index']):02d} {record['identity']}", fontsize=10
                )
            for ax, (_, title) in zip(axes[0], DISPLAY_ARMS):
                ax.set_title(title, fontsize=11)
            fig.suptitle(f"{lineage} branch faces — page {page}", fontsize=15)
            fig.savefig(
                output_root / "overviews" / f"{lineage.lower()}_{page}.png",
                dpi=180,
            )
            plt.close(fig)


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["samples"]
    if len(records) != 16:
        raise RuntimeError(f"Expected the sealed 16-cell view, found {len(records)}")
    (args.output_root / "samples").mkdir(parents=True, exist_ok=True)
    (args.output_root / "overviews").mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        rows.extend(
            _render_sample(
                args.task_root,
                args.reference_root,
                record,
                args.output_root
                / "samples"
                / f"{int(record['index']):02d}_{record['identity']}.png",
            )
        )
    _render_overviews(args.task_root, records, args.output_root)

    columns = list(rows[0])
    with (args.output_root / "branch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    numeric = [column for column in columns if column.endswith("_vs_native")]
    means = {}
    for lineage in LINEAGES:
        for arm in LINEAGES[lineage]:
            selected = [
                row for row in rows
                if row["lineage"] == lineage and row["arm"] == arm
            ]
            means[f"{lineage}/{arm}"] = {
                column: float(np.mean([row[column] for row in selected]))
                for column in numeric
            }
    summary = {
        "sample_count": len(records),
        "lineages": list(LINEAGES),
        "definition": {
            "native": "N at every analysed BA processor",
            "reference_face": "N + S(R-N) at every analysed BA processor",
            "low_only": "N + S*C*g_low*D_low (C only where configured)",
            "high_only": "N + S*C*g_high*D_high (C only where configured)",
            "confidence_one": "ordinary CL39 frequency route with C forced to one",
        },
        "important_limit": (
            "Every arm is a whole-denoising intervention. Low-only and high-only "
            "images are nonlinear stress views and cannot be added in RGB space."
        ),
        "means": means,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
