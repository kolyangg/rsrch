#!/usr/bin/env python3
"""Render direct CL39 N, routed-R, and R-N face comparisons.

The R arm is an explicit evaluation-only intervention: at every shipped CL39
processor, target attention becomes N + router * (R - N). Thus R replaces N
inside the existing soft face router while N remains untouched outside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference-face-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser.parse_args()


def _trainer_image(root: Path, record: dict) -> Path:
    index = int(record["index"])
    filename = str(record["face_bbox_gen_key"]).replace(" ", "_")
    path = root / f"step_24000_batch_{index // 12}" / filename
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
        raise ValueError(f"Invalid crop bbox {bbox} for shape {image.shape}")
    return image[y0:y1, x0:x1]


def _signed_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    # Fixed gain makes panels comparable; mid-gray is exactly zero difference.
    return np.clip(0.5 + 4.0 * (first - second), 0.0, 1.0)


def _difference_overlay(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    magnitude = np.abs(other - base).mean(axis=2)
    scale = max(float(np.quantile(magnitude, 0.99)), 1.0 / 255.0)
    heat = plt.get_cmap("magma")(np.clip(magnitude / scale, 0.0, 1.0))[..., :3]
    alpha = (0.15 + 0.75 * np.clip(magnitude / scale, 0.0, 1.0))[..., None]
    return np.clip((1.0 - alpha) * base + alpha * heat, 0.0, 1.0)


def _slug(record: dict) -> str:
    action = record["prompt"].split()[0].lower().replace("/", "-")
    return f"{int(record['index']):02d}_{record['identity']}_{action}"


def _panel(record: dict, paths: dict[str, Path], figure_path: Path) -> dict:
    actual = _rgb(paths["actual"])
    native = _rgb(paths["native"])
    reference_face = _rgb(paths["reference_face"])
    reference = _rgb(paths["reference"])
    bbox = record["face_bbox_gen"]
    bbox_ref = record["face_bbox_ref"]

    actual_face = _crop(actual, bbox)
    native_face = _crop(native, bbox)
    reference_face_crop = _crop(reference_face, bbox)
    reference_crop = _crop(reference, bbox_ref)
    signed_face = _signed_difference(reference_face_crop, native_face)
    overlay = _difference_overlay(native, reference_face)

    difference = np.abs(reference_face - native)
    face_difference = np.abs(reference_face_crop - native_face)
    metrics = {
        "index": int(record["index"]),
        "identity": record["identity"],
        "prompt": record["prompt"],
        "rgb_mae_all": float(difference.mean()),
        "rgb_mae_face": float(face_difference.mean()),
        "pixel_changed_gt_1_255_all": float(
            (difference.max(axis=2) > 1.0 / 255.0).mean()
        ),
        "pixel_changed_gt_1_255_face": float(
            (face_difference.max(axis=2) > 1.0 / 255.0).mean()
        ),
    }

    fig, axes = plt.subplots(2, 5, figsize=(18, 8.2), constrained_layout=True)
    top = [reference, actual, native, reference_face, overlay]
    top_titles = [
        "Identity reference",
        "Actual CL39",
        "N-only final image\n(existing BA-off arm)",
        "R-on-face final image\nN + router·(R−N)",
        "|R-on-face − N| overlay",
    ]
    bottom = [
        reference_crop,
        actual_face,
        native_face,
        reference_face_crop,
        signed_face,
    ]
    bottom_titles = [
        "Reference face crop",
        "Actual CL39 face crop",
        "N-only face crop",
        "R-on-face face crop",
        "Signed R-on-face − N\n4× gain; gray = zero",
    ]
    for ax, image, title in zip(axes[0], top, top_titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax, image, title in zip(axes[1], bottom, bottom_titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"CL39 branch-face intervention — cell {metrics['index']:02d}, "
        f"{metrics['identity']}\nface RGB MAE={metrics['rgb_mae_face']:.4f}; "
        f"changed>{1}/255={100 * metrics['pixel_changed_gt_1_255_face']:.1f}%",
        fontsize=15,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    return metrics


def _overview(records: list[dict], output_root: Path, figure_dir: Path) -> None:
    for group_index, start in enumerate(range(0, len(records), 4), start=1):
        subset = records[start : start + 4]
        fig, axes = plt.subplots(
            len(subset), 5, figsize=(14.5, 3.0 * len(subset)), constrained_layout=True
        )
        if len(subset) == 1:
            axes = axes[None, :]
        for row, record in enumerate(subset):
            root = output_root / "samples" / _slug(record)
            reference = _rgb(root / "reference.png")
            actual = _rgb(root / "actual.png")
            native = _rgb(root / "native.png")
            routed_reference = _rgb(root / "reference_face.png")
            bbox = record["face_bbox_gen"]
            views = [
                _crop(reference, record["face_bbox_ref"]),
                _crop(actual, bbox),
                _crop(native, bbox),
                _crop(routed_reference, bbox),
                _signed_difference(_crop(routed_reference, bbox), _crop(native, bbox)),
            ]
            for col, view in enumerate(views):
                axes[row, col].imshow(view)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(
                f"{int(record['index']):02d} {record['identity']}", fontsize=11
            )
        titles = [
            "Reference",
            "Actual CL39",
            "N-only",
            "R-on-face",
            "Signed R−N (4×)",
        ]
        for ax, title in zip(axes[0], titles):
            ax.set_title(title, fontsize=12)
        fig.savefig(
            figure_dir / f"cl39_branch_faces_overview_{group_index}.png", dpi=180
        )
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["samples"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_figure_dir = args.figure_dir / "branch_samples"
    sample_figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for record in records:
        sample_root = args.output_root / "samples" / _slug(record)
        sample_root.mkdir(parents=True, exist_ok=True)
        sources = {
            "actual": _trainer_image(args.actual_dir, record),
            "native": _trainer_image(args.native_dir, record),
            "reference_face": _trainer_image(args.reference_face_dir, record),
            "reference": args.reference_root / Path(record["reference_path"]).name,
        }
        for name, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, sample_root / f"{name}.png")
        local_paths = {name: sample_root / f"{name}.png" for name in sources}
        rows.append(
            _panel(
                record,
                local_paths,
                sample_figure_dir / f"{int(record['index']):02d}_{record['identity']}.png",
            )
        )

    _overview(records, args.output_root, args.figure_dir)
    columns = list(rows[0])
    with (args.output_root / "branch_face_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in columns if key.startswith("rgb_") or key.startswith("pixel_")]
    summary = {
        "definition": {
            "N": "target-Q / target-KV attention output",
            "R": "target-Q / reference-KV attention output",
            "reference_face_arm": "N + soft_face_router * (R - N)",
            "difference_panel": "reference_face final RGB - N-only final RGB",
        },
        "sample_count": len(rows),
        "means": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "important_limit": (
            "R and N are intermediate attention features. The R-on-face image is a "
            "controlled whole-denoising intervention, not a direct VAE decode of R."
        ),
    }
    (args.output_root / "branch_face_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
