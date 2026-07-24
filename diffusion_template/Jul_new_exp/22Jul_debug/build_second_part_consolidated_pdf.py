#!/usr/bin/env python3
"""Build a four-case-per-run PDF for every bundle completed after wave0."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments"
REPORTS = HERE / "visual_reports"
DEFAULT_CUTOFF = REPORTS / "20260722_wave0_step0_visual_summary.pdf"


def number(value: object, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def read_image(path: Path, max_side: int = 480) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image.copy()


def bundle_note(bundle_name: str, sample_count: int) -> str:
    if sample_count < 4:
        return "smoke test: fewer than four cases exist"
    if "__n3aConfidence__" in bundle_name:
        return "superseded: confidence branch gain was misrouted"
    if "__stagedRetry__" in bundle_name:
        return "retry bundle"
    return ""


def discover_bundles(cutoff: Path) -> tuple[list[dict], list[dict]]:
    cutoff_mtime = cutoff.stat().st_mtime
    included: list[dict] = []
    skipped: list[dict] = []
    for summary_path in EXPERIMENTS.glob("*/metrics_summary.json"):
        if summary_path.stat().st_mtime <= cutoff_mtime:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as error:
            skipped.append({"path": str(summary_path), "reason": f"bad JSON: {error}"})
            continue
        experiment_dir = summary_path.parent
        metrics_path = experiment_dir / "metrics_per_sample.csv"
        ba_images = list((experiment_dir / "images").glob("sample_*_BA.png"))
        if not metrics_path.exists() or not ba_images:
            skipped.append(
                {
                    "path": str(experiment_dir),
                    "experiment_id": summary.get("experiment_id"),
                    "reason": "no completed sample rows or BA images",
                }
            )
            continue
        try:
            metrics = pd.read_csv(metrics_path).sort_values("source_index")
        except Exception as error:
            skipped.append(
                {
                    "path": str(experiment_dir),
                    "experiment_id": summary.get("experiment_id"),
                    "reason": f"bad metrics CSV: {error}",
                }
            )
            continue
        if metrics.empty:
            skipped.append(
                {
                    "path": str(experiment_dir),
                    "experiment_id": summary.get("experiment_id"),
                    "reason": "empty metrics CSV",
                }
            )
            continue
        summary["_summary_path"] = str(summary_path)
        summary["_bundle_name"] = experiment_dir.name
        summary["_mtime"] = summary_path.stat().st_mtime
        summary["_metrics"] = metrics
        summary["_note"] = bundle_note(experiment_dir.name, len(metrics))
        included.append(summary)
    included.sort(key=lambda item: (item["_mtime"], item["_bundle_name"]))
    return included, skipped


def select_representative_rows(metrics: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(metrics) <= count:
        return metrics
    positions = np.rint(np.linspace(0, len(metrics) - 1, count)).astype(int)
    positions = np.unique(positions)
    if len(positions) < count:
        for position in range(len(metrics)):
            if position not in positions:
                positions = np.append(positions, position)
            if len(positions) == count:
                break
    return metrics.iloc[np.sort(positions[:count])]


def metric_suffix(summary: dict) -> tuple[str, str]:
    if summary.get("median_face_mae_vs_ba0") is not None:
        return "_vs_ba0", "BA0"
    return "_vs_pm", "PM"


def overview_pages(pdf: PdfPages, bundles: list[dict], rows_per_page: int = 18) -> int:
    columns = [
        "#",
        "experiment",
        "n",
        "base",
        "face",
        "gain",
        "landmark",
        "IoU",
        "outside",
        "run / note",
    ]
    page_count = 0
    for start in range(0, len(bundles), rows_per_page):
        chunk = bundles[start : start + rows_per_page]
        fig = plt.figure(figsize=(17, 10.5))
        fig.suptitle(
            "Post-wave0 experiment index",
            fontsize=19,
            fontweight="bold",
            y=0.978,
        )
        fig.text(
            0.04,
            0.935,
            f"Distinct immutable run bundles {start + 1}–{start + len(chunk)} of "
            f"{len(bundles)}. Metrics use each bundle's causal baseline where available.",
            fontsize=10,
        )
        rows = []
        for offset, summary in enumerate(chunk):
            suffix, baseline = metric_suffix(summary)
            bundle = summary["_bundle_name"]
            run_short = bundle.split("__", 2)[1] if "__" in bundle else bundle
            note = summary["_note"]
            rows.append(
                [
                    str(start + offset + 1),
                    summary.get("experiment_id", ""),
                    str(len(summary["_metrics"])),
                    baseline,
                    number(summary.get(f"median_face_mae{suffix}")),
                    number(summary.get(f"median_reference_gain{suffix}")),
                    number(summary.get(f"median_landmark_displacement{suffix}")),
                    number(summary.get(f"median_bbox_iou{suffix}")),
                    number(summary.get(f"median_outside_mae{suffix}")),
                    f"{run_short}" + (f" — {note}" if note else ""),
                ]
            )
        ax = fig.add_axes([0.025, 0.055, 0.95, 0.84])
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            colWidths=[0.028, 0.255, 0.033, 0.043, 0.056, 0.056, 0.065, 0.054, 0.06, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.75)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#23364d")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")
            if row > 0 and rows[row - 1][-1].find("superseded") >= 0:
                cell.set_facecolor("#f6dfc4")
            if column in (1, 9):
                cell.get_text().set_ha("left")
        pdf.savefig(fig, bbox_inches="tight", dpi=120)
        plt.close(fig)
        page_count += 1
    return page_count


def metric_text(row: pd.Series) -> str:
    result = (
        f"vs PM: face={number(row.get('face_mae_vs_pm'))}  "
        f"gain={number(row.get('reference_gain_vs_pm'))}\n"
        f"lmk={number(row.get('landmark_displacement_vs_pm'))}  "
        f"IoU={number(row.get('bbox_iou_vs_pm'))}  "
        f"out={number(row.get('outside_mae_vs_pm'))}"
    )
    if pd.notna(row.get("face_mae_vs_ba0")):
        result += (
            f"\nvs BA0: face={number(row.get('face_mae_vs_ba0'))}  "
            f"gain={number(row.get('reference_gain_vs_ba0'))}  "
            f"out={number(row.get('outside_mae_vs_ba0'))}"
        )
    return result


def run_page(
    pdf: PdfPages,
    summary: dict,
    run_number: int,
    total_runs: int,
    samples_per_run: int,
) -> dict:
    metrics = summary["_metrics"]
    selected = select_representative_rows(metrics, samples_per_run)
    experiment_dir = Path(summary["experiment_dir"])
    if not experiment_dir.exists():
        experiment_dir = Path(summary["_summary_path"]).parent
    images_dir = experiment_dir / "images"
    row_count = len(selected)
    fig, axes = plt.subplots(
        row_count,
        6,
        figsize=(17, 3.2 * row_count + 1.45),
        squeeze=False,
    )
    fig.suptitle(
        f"{run_number}/{total_runs}  |  {summary.get('experiment_id', 'unknown')}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    suffix, baseline = metric_suffix(summary)
    aggregate = (
        f"bundle: {summary['_bundle_name']}  |  n={len(metrics)}  |  baseline={baseline}  |  "
        f"face={number(summary.get(f'median_face_mae{suffix}'))}  "
        f"gain={number(summary.get(f'median_reference_gain{suffix}'))}  "
        f"lmk={number(summary.get(f'median_landmark_displacement{suffix}'))}  "
        f"IoU={number(summary.get(f'median_bbox_iou{suffix}'))}  "
        f"outside={number(summary.get(f'median_outside_mae{suffix}'))}"
    )
    if summary["_note"]:
        aggregate += f"  |  {summary['_note']}"
    fig.text(0.5, 0.965, aggregate, ha="center", va="top", fontsize=8.5)

    chosen_indices: list[int] = []
    for row_number, (_, row) in enumerate(selected.iterrows()):
        source_index = int(row["source_index"])
        chosen_indices.append(source_index)
        prefix = f"sample_{source_index:02d}"
        target_ref = read_image(images_dir / f"{prefix}_reference.png")
        branch_ref = read_image(images_dir / f"{prefix}_ppr_reference.png") or target_ref
        pm = read_image(images_dir / f"{prefix}_PM0.png")
        ba0 = read_image(images_dir / f"{prefix}_BA0.png")
        ba = read_image(images_dir / f"{prefix}_BA.png")
        pm_face = read_image(images_dir / f"{prefix}_PM0_face.png")
        ba_face = read_image(images_dir / f"{prefix}_BA_face.png")
        if ba0 is not None:
            panels = [
                ("Reference", target_ref),
                ("PhotoMaker", pm),
                ("BA0", ba0),
                ("BA", ba),
                ("PM face", pm_face),
                ("BA face", ba_face),
            ]
        else:
            panels = [
                ("Target ref", target_ref),
                ("Branch ref", branch_ref),
                ("PhotoMaker", pm),
                ("BA", ba),
                ("PM face", pm_face),
                ("BA face", ba_face),
            ]
        for column, (title, image) in enumerate(panels):
            ax = axes[row_number, column]
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#777777")
            ax.axis("off")
            ax.set_title(
                f"{title}\nsample {source_index}" if row_number == 0 else title,
                fontsize=9,
            )
        axes[row_number, 5].text(
            0.5,
            -0.075,
            metric_text(row),
            ha="center",
            va="top",
            transform=axes[row_number, 5].transAxes,
            fontsize=7.2,
        )
    fig.subplots_adjust(top=0.925, bottom=0.035, hspace=0.31, wspace=0.035)
    pdf.savefig(fig, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return {
        "run_number": run_number,
        "bundle": summary["_bundle_name"],
        "experiment_id": summary.get("experiment_id"),
        "sample_count": len(metrics),
        "selected_source_indices": chosen_indices,
        "note": summary["_note"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=Path, default=DEFAULT_CUTOFF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples-per-run", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_run < 1 or args.samples_per_run > 8:
        raise ValueError("--samples-per-run must be between 1 and 8")

    cutoff = args.cutoff if args.cutoff.is_absolute() else HERE / args.cutoff
    if not cutoff.exists():
        raise FileNotFoundError(cutoff)
    bundles, skipped = discover_bundles(cutoff)
    if not bundles:
        raise RuntimeError("No completed post-cutoff bundles found")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    output = args.output or REPORTS / f"{stamp}_second_part_all_runs_4sample_summary.pdf"
    manifest = args.manifest or REPORTS / f"{stamp}_second_part_all_runs_4sample_summary_manifest.json"
    if not output.is_absolute():
        output = HERE / output
    if not manifest.is_absolute():
        manifest = HERE / manifest
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite: {output} or {manifest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "PhotoMaker branched-attention post-wave0 four-case summary"
        metadata["Author"] = "Automated architecture search"
        metadata["Subject"] = "One page per distinct run bundle, with four representative cases"
        overview_count = overview_pages(pdf, bundles)
        for run_number, summary in enumerate(bundles, start=1):
            entries.append(
                run_page(pdf, summary, run_number, len(bundles), args.samples_per_run)
            )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_pdf": str(cutoff),
        "output_pdf": str(output),
        "selection": (
            f"Every completed metrics bundle newer than the cutoff; up to "
            f"{args.samples_per_run} rows evenly spaced over source ordering."
        ),
        "included_run_count": len(bundles),
        "overview_page_count": overview_count,
        "run_page_count": len(entries),
        "total_page_count": overview_count + len(entries),
        "runs": entries,
        "skipped": skipped,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
