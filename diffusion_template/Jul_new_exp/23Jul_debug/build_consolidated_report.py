#!/usr/bin/env python3
"""Build one cross-run PDF from every fully validated 23Jul training arm.

Each checkpoint page is a true comparison grid: one architecture per row and
the same four prompts as columns.  This makes visual failure modes harder to
hide behind aggregate face metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
REPORTS = HERE / "visual_reports"
COSMIC_DATA = HERE / "data" / "id_00081_1017318003459"
ONE_ID_ROOT = Path("/home/niko/rsrch/dataset_full/one_id")
ONE_ID_PROMPTS = Path("/home/niko/rsrch/dataset_full/val_dataset/prompts_10.txt")
STEPS = (0, 200, 400, 600)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="*",
        help="Optional explicit run directories; otherwise discover complete reports.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows-per-page", type=int, default=5)
    parser.add_argument(
        "--dataset-profile",
        choices=(
            "cosmic_large_id00081",
            "one_id_nm0005092_subset8",
            "one_id_nm0005092_subset8_distinct",
            "one_id_nm0005092_full18_heldout_distinct",
            "all",
        ),
        default="cosmic_large_id00081",
        help=(
            "Select one dataset profile, or 'all' to place profile-separated "
            "sections in one PDF. Cross-profile metrics are not compared."
        ),
    )
    return parser.parse_args()


def images_for(run_dir: Path, mode: str, step: int) -> list[Path]:
    root = run_dir / "validation" / mode / f"step_{step:04d}" / "outputs"
    return sorted(
        path
        for path in root.glob("*/val_images/manual_val/step_*_batch_*/*.png")
        if not path.stem.endswith("_mask")
    )


def finite(value, digits=4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def load_image(path: Path, max_side=700):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image.copy()


def discover(
    explicit: list[Path], dataset_profile: str
) -> tuple[list[dict], list[dict]]:
    candidates = explicit or sorted(EXPERIMENTS.glob("*"))
    included = []
    skipped = []
    for candidate in candidates:
        run_dir = candidate if candidate.is_absolute() else (HERE / candidate)
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "report" / "metrics_summary.json"
        if not manifest_path.exists() or not metrics_path.exists():
            skipped.append({"run_dir": str(run_dir), "reason": "missing manifest or metrics"})
            continue
        missing = [
            step for step in STEPS if len(images_for(run_dir, "canonical50", step)) != 4
        ]
        if missing:
            skipped.append(
                {"run_dir": str(run_dir), "reason": f"incomplete canonical steps {missing}"}
            )
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_profile = manifest.get(
            "dataset_profile", "cosmic_large_id00081"
        )
        if dataset_profile != "all" and manifest_profile != dataset_profile:
            skipped.append(
                {
                    "run_dir": str(run_dir),
                    "reason": f"dataset profile {manifest_profile}",
                }
            )
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        by_step = {
            int(record["step"]): record
            for record in metrics
            if record.get("mode") == "canonical50"
        }
        if set(by_step) != set(STEPS):
            skipped.append({"run_dir": str(run_dir), "reason": "incomplete metrics"})
            continue
        included.append(
            {
                "run_dir": run_dir,
                "architecture_id": manifest["architecture_id"],
                "display_architecture_id": (
                    f"{manifest['architecture_id']} [INVALID SAME-IMAGE]"
                    if manifest_profile == "one_id_nm0005092_subset8"
                    else manifest["architecture_id"]
                ),
                "run_name": manifest["run_name"],
                "dataset_profile": manifest_profile,
                "metrics": by_step,
            }
        )
    return included, skipped


def reference_page(pdf: PdfPages, runs: list[dict]):
    profile = runs[0]["dataset_profile"]
    if profile == "cosmic_large_id00081":
        prompts_path = COSMIC_DATA / "validation_prompts_4.txt"
        reference_path = COSMIC_DATA / "validation_refs" / "holdout_A.jpg"
        subject = "id_00081 / CosmicLarge-style loader"
        replacement = ""
    elif profile in {
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    }:
        prompts_path = ONE_ID_PROMPTS
        reference_path = ONE_ID_ROOT / "ref" / "51.jpg"
        if profile == "one_id_nm0005092_full18_heldout_distinct":
            subject = (
                "nm0005092 / OneIDTrain full set minus held-out 51.jpg / "
                "distinct target/reference"
            )
        else:
            pairing = (
                "distinct target/reference"
                if profile.endswith("_distinct")
                else "same-image leakage audit"
            )
            subject = f"nm0005092 / OneIDTrain subset8 / {pairing}"
        replacement = "man img"
    else:
        raise ValueError(profile)
    prompts = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:4]
    if replacement:
        prompts = [prompt.replace("<class>", replacement) for prompt in prompts]
    reference = load_image(reference_path)
    pm_paths = images_for(runs[0]["run_dir"], "pmControl50", 0)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    axes[0].imshow(reference)
    axes[0].set_title("Held-out reference")
    axes[0].axis("off")
    for index, (axis, path) in enumerate(zip(axes[1:], pm_paths)):
        axis.imshow(load_image(path))
        axis.set_title(f"PhotoMaker p{index}\n{prompts[index][:38]}", fontsize=8)
        axis.axis("off")
    fig.suptitle(
        f"23Jul NN3a_new1 training study — {subject}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "600 optimizer steps; checkpoints every 200. Identity metrics are secondary "
        "at step zero; visual alignment and geometry are promotion gates.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.83, bottom=0.08, wspace=0.03)
    pdf.savefig(fig, bbox_inches="tight", dpi=130)
    plt.close(fig)


def metric_pages(
    pdf: PdfPages, runs: list[dict], rows_per_page=20, section_label=""
) -> int:
    rows = []
    for run in runs:
        for step in STEPS:
            metric = run["metrics"][step]
            rows.append(
                [
                    run["display_architecture_id"],
                    str(step),
                    finite(metric.get("median_reference_similarity")),
                    finite(metric.get("median_reference_gain_vs_pm")),
                    finite(metric.get("median_face_distance_from_pm_output")),
                    finite(metric.get("median_bbox_iou_vs_pm")),
                    finite(metric.get("median_landmark_displacement_vs_pm")),
                    finite(metric.get("median_outside_mae_vs_pm")),
                    finite(metric.get("selection_score")),
                ]
            )
    headers = [
        "architecture",
        "step",
        "ref sim",
        "gain vs PM",
        "face Δ PM",
        "bbox IoU",
        "landmark Δ",
        "outside MAE",
        "score",
    ]
    page_count = 0
    for start in range(0, len(rows), rows_per_page):
        chunk = rows[start : start + rows_per_page]
        fig, axis = plt.subplots(figsize=(15, 10))
        axis.axis("off")
        table = axis.table(
            cellText=chunk,
            colLabels=headers,
            cellLoc="center",
            colLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.55)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#26384d")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")
            if column == 0:
                cell.get_text().set_ha("left")
        fig.suptitle(
            f"Canonical validation metrics — {section_label} — "
            f"rows {start + 1}–{start + len(chunk)}",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.035,
            "Face embedding scores can reward malformed faces; inspect the grids before "
            "promoting any arm.",
            ha="center",
            fontsize=9,
        )
        pdf.savefig(fig, bbox_inches="tight", dpi=130)
        plt.close(fig)
        page_count += 1
    return page_count


def grid_pages(
    pdf: PdfPages, runs: list[dict], rows_per_page: int, section_label=""
) -> int:
    page_count = 0
    for step in STEPS:
        for start in range(0, len(runs), rows_per_page):
            chunk = runs[start : start + rows_per_page]
            fig, axes = plt.subplots(
                len(chunk),
                4,
                figsize=(15, 3.55 * len(chunk) + 1),
                squeeze=False,
            )
            for row_index, run in enumerate(chunk):
                metric = run["metrics"][step]
                paths = images_for(run["run_dir"], "canonical50", step)
                for prompt_index, (axis, path) in enumerate(
                    zip(axes[row_index], paths)
                ):
                    axis.imshow(load_image(path))
                    axis.axis("off")
                    if row_index == 0:
                        axis.set_title(f"prompt {prompt_index}", fontsize=10)
                label = (
                    f"{run['display_architecture_id']}\n"
                    f"ref={finite(metric.get('median_reference_similarity'), 3)}  "
                    f"IoU={finite(metric.get('median_bbox_iou_vs_pm'), 3)}  "
                    f"lmk={finite(metric.get('median_landmark_displacement_vs_pm'), 3)}\n"
                    f"out={finite(metric.get('median_outside_mae_vs_pm'), 3)}  "
                    f"score={finite(metric.get('selection_score'), 3)}"
                )
                axes[row_index, 0].text(
                    -0.04,
                    0.5,
                    label,
                    transform=axes[row_index, 0].transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            fig.suptitle(
                f"Canonical BA checkpoint step {step} — {section_label}\n"
                "same four cases for every arm in this dataset section",
                fontsize=16,
                fontweight="bold",
            )
            fig.subplots_adjust(left=0.18, top=0.92, bottom=0.02, hspace=0.08, wspace=0.02)
            pdf.savefig(fig, bbox_inches="tight", dpi=130)
            plt.close(fig)
            page_count += 1
    return page_count


def audit_page(pdf: PdfPages, skipped: list[dict]) -> int:
    """List run directories that could not enter a visual comparison grid."""
    if not skipped:
        return 0
    rows = [
        [Path(record["run_dir"]).name, record["reason"]]
        for record in skipped
    ]
    fig, axis = plt.subplots(figsize=(16, 10))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["run directory", "why it is not in a checkpoint grid"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.70, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.45)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#26384d")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
    fig.suptitle(
        "Run audit — incomplete/failed runs excluded from visual grids",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.03,
        "A run requires four canonical images and metrics at steps 0/200/400/600 "
        "to enter the comparison grids.",
        ha="center",
        fontsize=9,
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return 1


def main() -> int:
    args = parse_args()
    if args.rows_per_page < 1:
        raise ValueError("--rows-per-page must be positive")
    runs, skipped = discover(args.run_dirs, args.dataset_profile)
    if not runs:
        raise RuntimeError("No fully validated run with metrics was found")
    REPORTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or REPORTS / f"{stamp}_training_architecture_comparison.pdf"
    if not output.is_absolute():
        output = HERE / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")

    profile_order = (
        "cosmic_large_id00081",
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    )
    grouped_runs = [
        (profile, [run for run in runs if run["dataset_profile"] == profile])
        for profile in profile_order
    ]
    grouped_runs = [(profile, group) for profile, group in grouped_runs if group]
    page_count = 0
    section_payload = []
    with PdfPages(output) as pdf:
        pdf.infodict()["Title"] = "23Jul NN3a_new1 training architecture comparison"
        for profile, group in grouped_runs:
            reference_page(pdf, group)
            metric_page_count = metric_pages(
                pdf, group, section_label=profile
            )
            grid_page_count = grid_pages(
                pdf, group, args.rows_per_page, section_label=profile
            )
            section_pages = 1 + metric_page_count + grid_page_count
            page_count += section_pages
            section_payload.append(
                {
                    "dataset_profile": profile,
                    "run_count": len(group),
                    "page_count": section_pages,
                }
            )
        if args.dataset_profile == "all":
            page_count += audit_page(pdf, skipped)

    payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_profile": args.dataset_profile,
        "output_pdf": str(output),
        "included": [
            {
                "architecture_id": run["architecture_id"],
                "run_name": run["run_name"],
                "run_dir": str(run["run_dir"]),
            }
            for run in runs
        ],
        "sections": section_payload,
        "skipped": skipped,
        "page_count": page_count,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
