#!/usr/bin/env python3
"""Build a paginated, same-sample visual comparison across experiment bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--samples", required=True, help="Comma-separated source indices")
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--title", default="Step-zero architecture comparison")
    return parser.parse_args()


def candidate_bundles(experiment_id: str) -> list[tuple[int, float, Path, dict]]:
    found = []
    for summary_path in EXPERIMENTS.glob(f"*__{experiment_id}/metrics_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        bundle = summary_path.parent
        found.append(
            (
                int(payload.get("sample_count", 0)),
                summary_path.stat().st_mtime,
                bundle,
                payload,
            )
        )
    return sorted(found, reverse=True)


def select_bundle(experiment_id: str, samples: list[int]) -> tuple[Path, dict]:
    for _, _, bundle, summary in candidate_bundles(experiment_id):
        if all((bundle / "images" / f"sample_{idx:02d}_BA.png").exists() for idx in samples):
            return bundle, summary
    raise FileNotFoundError(
        f"No completed {experiment_id!r} bundle contains every requested sample: {samples}"
    )


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def short_label(experiment_id: str) -> str:
    replacements = {
        "matrix96_n3a_fullgrid_up_core_ring_anchor": "canonical N3a core68",
        "n3a_fullgrid_up_dual25_div8": "fullgrid dual25",
        "n3a_fullgrid_up_dual35_div8": "fullgrid dual35",
        "n3a_core68_plus_zero_refpooled_div8": "core68 + zero pooled text",
        "nn7v2_lmkidw3_staged_up003_up1075_ba0": "landmark staged 0.03/0.075",
    }
    return replacements.get(experiment_id, experiment_id)


def metric_line(summary: dict) -> str:
    def val(name: str) -> str:
        value = summary.get(name)
        return "n/a" if value is None else f"{float(value):.5f}"

    return (
        f"n={summary.get('sample_count', '?')} | face MAE={val('median_face_mae_vs_pm')} | "
        f"lmk={val('median_landmark_displacement_vs_pm')} | "
        f"bbox={val('median_bbox_iou_vs_pm')} | outside={val('median_outside_mae_vs_pm')}"
    )


def main() -> None:
    args = parse_args()
    samples = [int(item) for item in args.samples.split(",") if item.strip()]
    selected = [select_bundle(item, samples) for item in args.experiments]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.output) as pdf:
        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle(args.title, fontsize=18, y=0.95)
        lines = []
        for experiment_id, (_, summary) in zip(args.experiments, selected):
            lines.append(f"{short_label(experiment_id)}\n  {metric_line(summary)}")
        fig.text(0.05, 0.86, "\n\n".join(lines), va="top", family="monospace", fontsize=10)
        fig.text(
            0.05,
            0.08,
            "Identity similarity is diagnostic only at step zero. Review visible branch activity, "
            "face coherence, pose/head alignment, expression preservation, and containment.",
            fontsize=10,
        )
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        base_images = selected[0][0] / "images"
        columns = ["Reference", "PhotoMaker"] + [short_label(item) for item in args.experiments]
        for offset in range(0, len(samples), args.rows_per_page):
            page_samples = samples[offset : offset + args.rows_per_page]
            fig, axes = plt.subplots(
                len(page_samples), len(columns), figsize=(3.1 * len(columns), 3.2 * len(page_samples))
            )
            if len(page_samples) == 1:
                axes = axes[None, :]
            for row, sample in enumerate(page_samples):
                paths = [
                    base_images / f"sample_{sample:02d}_reference.png",
                    base_images / f"sample_{sample:02d}_PM0.png",
                ] + [bundle / "images" / f"sample_{sample:02d}_BA.png" for bundle, _ in selected]
                for col, (axis, path) in enumerate(zip(axes[row], paths)):
                    axis.imshow(load(path))
                    axis.axis("off")
                    if row == 0:
                        axis.set_title(columns[col], fontsize=10)
                    if col == 0:
                        axis.text(
                            -0.04, 0.5, f"sample {sample}", transform=axis.transAxes,
                            rotation=90, va="center", ha="right", fontsize=10,
                        )
            fig.suptitle(args.title, fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
