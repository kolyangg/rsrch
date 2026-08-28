#!/usr/bin/env python3
"""Render compact figures for the CL39 reference-branch architecture report."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    return numerator / math.sqrt(left_energy * right_energy)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        rank = 0.5 * (cursor + end + 2)
        for offset in range(cursor, end + 1):
            result[ordered[offset][1]] = rank
        cursor = end + 1
    return result


def _load_joined(audit_csv: Path, branch_csv: Path) -> list[dict[str, float | str]]:
    audit = {
        int(row["index"]): row for row in csv.DictReader(audit_csv.open())
    }
    branch = {
        int(row["index"]): row for row in csv.DictReader(branch_csv.open())
    }
    if set(audit) != set(branch):
        raise RuntimeError("Audit and branch CSV sample indices differ")
    joined = []
    for index in sorted(audit):
        joined.append(
            {
                "index": str(index),
                "label": f"{index:02d} {audit[index]['identity']}",
                "face_mae": float(branch[index]["rgb_mae_face"]),
                "confidence": float(audit[index]["confidence_face"]),
                "raw_delta": float(audit[index]["raw_delta_face"]),
            }
        )
    return joined


def render_diagnostic(rows: list[dict[str, float | str]], output: Path) -> None:
    face_mae = [float(row["face_mae"]) for row in rows]
    confidence = [float(row["confidence"]) for row in rows]
    raw_delta = [float(row["raw_delta"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1), constrained_layout=True)
    panels = [
        (
            axes[0],
            confidence,
            "Mean CL39 confidence on face queries",
            "Entropy confidence is not an artifact-size detector",
        ),
        (
            axes[1],
            raw_delta,
            "Raw |R-N| feature magnitude on face queries",
            "Intervention size follows residual magnitude more closely",
        ),
    ]
    for axis, x_values, x_label, title in panels:
        axis.scatter(x_values, face_mae, s=46, color="#2563eb", alpha=0.86)
        for row, x_value, y_value in zip(rows, x_values, face_mae):
            axis.annotate(
                str(row["index"]),
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color="#334155",
            )
        pearson = _pearson(x_values, face_mae)
        spearman = _pearson(_ranks(x_values), _ranks(face_mae))
        axis.text(
            0.03,
            0.96,
            f"Pearson r = {pearson:+.2f}\nSpearman rho = {spearman:+.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1"},
        )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel("R-on-face vs N-only face RGB MAE")
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Selected 16-sample diagnostic: intervention magnitude is not visual quality",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Face MAE measures how much the final image changed, not whether it looks good; labels are validation indices.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _box(axis, xy, width, height, text, color, *, fontsize=9, linestyle="-"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
        linestyle=linestyle,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(axis, start, end, *, color="#475569", linestyle="-", width=1.4):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def render_architecture(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(15.2, 7.1), constrained_layout=True)
    axis.set_xlim(0, 15.2)
    axis.set_ylim(0, 7.1)
    axis.axis("off")

    blue = "#2563eb"
    orange = "#ea580c"
    green = "#15803d"
    purple = "#7e22ce"
    slate = "#475569"

    axis.text(0.2, 6.72, "CL39-R2 target architecture", fontsize=17, weight="bold")
    axis.text(
        0.2,
        6.36,
        "Keep N as the denoising anchor; make R trainable as a usable face route and calibrate only its residual.",
        fontsize=10.5,
        color=slate,
    )

    _box(axis, (0.35, 4.45), 1.45, 0.85, "Target state\nT", blue, fontsize=11)
    _box(axis, (0.35, 2.15), 1.45, 0.85, "Reference state\nHᵣ + face mask", orange, fontsize=10)
    _box(axis, (2.35, 4.45), 1.55, 0.85, "Target Q/K/V\nattention", blue)
    _box(axis, (2.35, 2.15), 1.55, 0.85, "Target Q +\nreference K/V", orange)
    _box(axis, (4.45, 4.45), 1.30, 0.85, "Native N", blue, fontsize=11)
    _box(axis, (4.45, 2.15), 1.30, 0.85, "Reference R", orange, fontsize=11)
    _arrow(axis, (1.80, 4.88), (2.35, 4.88), color=blue)
    _arrow(axis, (1.80, 2.58), (2.35, 2.58), color=orange)
    _arrow(axis, (1.80, 4.62), (2.35, 2.83), color=orange)
    _arrow(axis, (3.90, 4.88), (4.45, 4.88), color=blue)
    _arrow(axis, (3.90, 2.58), (4.45, 2.58), color=orange)

    _box(axis, (6.30, 3.28), 1.35, 1.02, "D = R - N\nGaussian split\nL + H", purple, fontsize=10)
    _arrow(axis, (5.75, 4.65), (6.30, 4.02), color=purple)
    _arrow(axis, (5.75, 2.78), (6.30, 3.56), color=purple)

    _box(
        axis,
        (6.15, 5.00),
        2.90,
        0.95,
        "Detached reliability features\nvalid-key mass · conditional entropy\nN/R agreement · band/native RMS · progress",
        green,
        fontsize=9,
    )
    _arrow(axis, (5.45, 3.00), (6.75, 5.00), color=green, linestyle="--")
    _arrow(axis, (5.45, 4.45), (7.35, 5.00), color=green, linestyle="--")

    _box(
        axis,
        (8.20, 3.28),
        2.05,
        1.02,
        "Tail-safe bands\nRMS caps at audited\nlayer/face percentiles",
        purple,
        fontsize=9.5,
    )
    _arrow(axis, (7.65, 3.79), (8.20, 3.79), color=purple)
    _box(
        axis,
        (9.55, 5.00),
        2.30,
        0.95,
        "Bounded band gates\nCᴸ, Cᴴ = current CL39 C\n+ zero-init small correction",
        green,
        fontsize=9,
    )
    _arrow(axis, (9.05, 5.47), (9.55, 5.47), color=green)

    _box(
        axis,
        (10.85, 3.28),
        2.05,
        1.02,
        "Actual route\nN + S(CᴸgᴸL + CᴴgᴴH)",
        blue,
        fontsize=10,
    )
    _arrow(axis, (10.25, 3.79), (10.85, 3.79), color=purple)
    _arrow(axis, (10.70, 5.00), (11.55, 4.30), color=green)
    _arrow(axis, (5.75, 4.88), (10.85, 4.02), color=blue)
    _box(axis, (13.45, 3.28), 1.45, 1.02, "U-Net\ncontinuation", blue, fontsize=10)
    _arrow(axis, (12.90, 3.79), (13.45, 3.79), color=blue)

    _box(
        axis,
        (7.45, 0.58),
        3.35,
        1.18,
        "Training-only coherent R-route dropout\nOn a small globally selected fraction of batches:\nN + S(R - N), with a warm ramp",
        orange,
        fontsize=9.5,
        linestyle="--",
    )
    _arrow(axis, (5.35, 2.15), (7.95, 1.76), color=orange, linestyle="--")
    _arrow(axis, (5.10, 4.45), (8.55, 1.76), color=orange, linestyle="--")
    _box(
        axis,
        (11.50, 0.73),
        2.30,
        0.88,
        "Same diffusion target\nforces R self-sufficiency",
        orange,
        fontsize=9.5,
        linestyle="--",
    )
    _arrow(axis, (10.80, 1.17), (11.50, 1.17), color=orange, linestyle="--")

    axis.text(
        0.35,
        0.72,
        "Build as a ladder:\nA. route dropout only\nB. bounded gate/caps only\nC. combine only if both pass",
        fontsize=10,
        color=slate,
        bbox={"boxstyle": "round,pad=0.5", "fc": "#f8fafc", "ec": "#cbd5e1"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--branch-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_joined(args.audit_csv, args.branch_csv)
    render_diagnostic(rows, args.output_dir / "fig_r_intervention_diagnostic.png")
    render_architecture(args.output_dir / "fig_cl39_r2_architecture.png")


if __name__ == "__main__":
    main()
