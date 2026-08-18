#!/usr/bin/env python3
"""Build report figures for the two critical Cosmic Large geometry repairs.

The figures use real target/reference pairs from the checked Cosmic sample
manifest and call the production ``compose_target_frame_reference`` function.
No model output or synthetic image content is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


RECORD_NAMES = (
    "339886005404.jpg",
    "1223294003468.jpg",
    "1087567013854.jpg",
)
CL14_DRAWS = (
    (0.06, (-0.13, 0.11)),
    (0.17, (0.12, -0.10)),
    (0.29, (-0.10, -0.13)),
)
GREEN = (34, 197, 94)
MAGENTA = (236, 72, 153)
CYAN = (6, 182, 212)
WHITE = (255, 255, 255)
BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/kolyangg/rsrch/dataset_full"),
        help="Parent of LAION-5B-Filtered-Large and cosmic_large.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "analysis" / "assets",
    )
    return parser.parse_args()


def load_compositor(repo: Path):
    source = repo / "src" / "datasets" / "reference_frame.py"
    spec = importlib.util.spec_from_file_location("clean_reference_frame", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import compositor from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compose_target_frame_reference


def fraction(box: Sequence[float], side: int = 1024) -> float:
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0) / float(side * side)


def short_side(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box
    return min(x1 - x0, y1 - y0)


def center(box: Sequence[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    *,
    fill: tuple[int, int, int, int],
    width: int = 7,
    dash: int = 18,
) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    for start in range(x0, x1, 2 * dash):
        draw.line((start, y0, min(start + dash, x1), y0), fill=fill, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=fill, width=width)
    for start in range(y0, y1, 2 * dash):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=fill, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=fill, width=width)


def overlay_boxes(
    image: Image.Image,
    boxes: Iterable[tuple[Sequence[float], tuple[int, int, int], bool]],
    *,
    connect: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for box, color, dashed in boxes:
        xy = tuple(int(round(v)) for v in box)
        draw.rectangle(xy, fill=(*color, 42))
        if dashed:
            dashed_rectangle(draw, box, fill=(*color, 255))
        else:
            draw.rectangle(xy, outline=(*color, 255), width=8)
    if connect is not None:
        a, b = center(connect[0]), center(connect[1])
        draw.line((*a, *b), fill=(*WHITE, 245), width=8)
        for x, y, color in ((*a, GREEN), (*b, CYAN)):
            radius = 12
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def add_panel(ax, image: Image.Image, title: str, note: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=10.5,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "black", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("#334155")


def load_records(dataset_root: Path):
    cosmic = dataset_root / "cosmic_large"
    manifest_path = cosmic / "gathered_data_cosmic_large_filtered_sample_two.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    selected = []
    for wanted in RECORD_NAMES:
        matches = [(path, record) for path, record in manifest.items() if Path(path).name == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one record named {wanted}, found {len(matches)}")
        target_rel, record = matches[0]
        ref_rel = record["face_paths"][0]
        target_path = dataset_root / target_rel
        reference_path = cosmic / ref_rel
        if not target_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(f"Missing Cosmic sample: {target_path} or {reference_path}")
        selected.append(
            {
                "target_path": target_path,
                "target_bbox": record["face_crop_new"],
                "reference_path": reference_path,
                "reference_bbox": record["face_bboxes"][ref_rel],
            }
        )
    return selected


def build_scale_figure(records, compose, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 13.5), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "Critical issue 1 — the 256px reference crop became an oversized 1024px spatial memory",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        "Real Cosmic Large pairs. Green = target face mask; magenta/cyan = reference face mask.",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )

    for row, record in enumerate(records):
        target = Image.open(record["target_path"]).convert("RGB")
        reference = Image.open(record["reference_path"]).convert("RGB")
        target_box = record["target_bbox"]
        ref_box = record["reference_bbox"]
        target_frac = fraction(target_box)

        target_vis = overlay_boxes(target, [(target_box, GREEN, False)])
        add_panel(
            axes[row, 0],
            target_vis,
            f"Sample {row + 1}: 1024px target scene",
            f"target face area = {100 * target_frac:.2f}%\nshort side = {short_side(target_box):.0f}px",
        )

        old = reference.resize((1024, 1024), BILINEAR)
        old_box = [4.0 * value for value in ref_box]
        old_frac = fraction(old_box)
        linear_ratio = short_side(old_box) / short_side(target_box)
        old_vis = overlay_boxes(old, [(old_box, MAGENTA, False)])
        add_panel(
            axes[row, 1],
            old_vis,
            "BEFORE: historical model input",
            f"256→1024 bilinear (4× per axis)\nface area = {100 * old_frac:.2f}%\nref/target short side = {linear_ratio:.2f}×",
        )

        exact, exact_box, _, telemetry = compose(
            reference,
            ref_box,
            target_box,
            canvas_size=1024,
            fill="edge",
            target_face_fraction=None,
            position_offset=(0.0, 0.0),
        )
        exact_vis = overlay_boxes(
            exact,
            [(target_box, GREEN, True), (exact_box, CYAN, False)],
        )
        add_panel(
            axes[row, 2],
            exact_vis,
            "SCALE REPAIR: compose before VAE",
            f"edge-filled 1024px frame\nshort-side ratio = {telemetry['scale_ratio']:.2f}×\ncyan is now target-scale",
        )

    fig.text(
        0.5,
        0.012,
        "The right column isolates the scale repair. Its exact centre alignment is intentionally shown here—and is the shortcut repaired in Figure 2.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.925, bottom=0.04, hspace=0.18, wspace=0.06)
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_position_figure(records, compose, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 13.5), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "Critical issue 2 — exact scale + exact position enabled an in-place copy shortcut",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        "Green dashed = target mask. Magenta/cyan = reference mask. White line joins mask centres.",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )

    for row, (record, (requested_fraction, offset)) in enumerate(zip(records, CL14_DRAWS)):
        target = Image.open(record["target_path"]).convert("RGB")
        reference = Image.open(record["reference_path"]).convert("RGB")
        target_box = record["target_bbox"]
        ref_box = record["reference_bbox"]

        target_vis = overlay_boxes(target, [(target_box, GREEN, False)])
        add_panel(
            axes[row, 0],
            target_vis,
            f"Sample {row + 1}: target query geometry",
            f"target centre = ({center(target_box)[0]:.0f}, {center(target_box)[1]:.0f})\ntarget area = {100 * fraction(target_box):.2f}%",
        )

        exact, exact_box, _, _ = compose(
            reference,
            ref_box,
            target_box,
            canvas_size=1024,
            fill="edge",
            target_face_fraction=None,
            position_offset=(0.0, 0.0),
        )
        exact_delta = math.dist(center(target_box), center(exact_box))
        exact_vis = overlay_boxes(
            exact,
            [(target_box, GREEN, True), (exact_box, MAGENTA, False)],
            connect=(target_box, exact_box),
        )
        add_panel(
            axes[row, 1],
            exact_vis,
            "BEFORE: CL2 exact registration",
            f"same short-side scale\ncentre separation = {exact_delta:.1f}px\ncell-for-cell shortcut",
        )

        jittered, jittered_box, _, telemetry = compose(
            reference,
            ref_box,
            target_box,
            canvas_size=1024,
            fill="edge",
            target_face_fraction=requested_fraction,
            position_offset=offset,
        )
        target_cx, target_cy = center(target_box)
        ref_cx, ref_cy = center(jittered_box)
        jittered_vis = overlay_boxes(
            jittered,
            [(target_box, GREEN, True), (jittered_box, CYAN, False)],
            connect=(target_box, jittered_box),
        )
        add_panel(
            axes[row, 2],
            jittered_vis,
            "AFTER: CL14 independent geometry",
            f"draw u = {requested_fraction:.2f}; realised = {telemetry['face_fraction']:.3f}\nrequested offset = ({offset[0]:+.2f}, {offset[1]:+.2f})\nactual Δcentre = ({ref_cx-target_cx:+.0f}, {ref_cy-target_cy:+.0f})px",
        )

    fig.text(
        0.5,
        0.012,
        "Displayed CL14 draws are fixed only for reproducibility; training samples u∼U(0.06,0.30) and x/y offsets∼U(−0.15,0.15) independently for every pair.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.925, bottom=0.04, hspace=0.18, wspace=0.06)
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.dataset_root)
    compose = load_compositor(repo)
    scale_path = args.output_dir / "cosmic_scale_mismatch_before_after.png"
    position_path = args.output_dir / "cosmic_positional_shortcut_before_after.png"
    build_scale_figure(records, compose, scale_path)
    build_position_figure(records, compose, position_path)
    print(scale_path)
    print(position_path)


if __name__ == "__main__":
    main()
