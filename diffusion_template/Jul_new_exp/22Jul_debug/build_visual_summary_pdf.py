#!/usr/bin/env python3
"""Build a compact multi-experiment visual/metric PDF from immutable bundles."""

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

DEFAULT_IDS = [
    "n3a_exact",
    "n3a_fullgrid_up_core_ring_anchor",
    "n3a_roi_up_confidence50_anchor",
    "n3a_roi_up_confidence50_early25_anchor",
    "n3a_roi_up_dual25_anchor",
    "n3a_roi_up_dual30_anchor",
    "nn7a_init_v2_default",
    "promote50_confidence50_standard",
    "promote50_dual25_standard",
    "wrongref50_confidence50_elon12",
    "rescue50_confidence50_late20",
    "matrix24_confidence50_step5",
    "matrix24_dual25_step6",
    "nn4_step0_default",
    "nn5a_step0_default",
    "nn5b_step0_default",
    "nn6a_step0_default",
    "nn4_step0_vs_ba0",
    "nn5a_step0_vs_ba0",
    "nn5b_step0_vs_ba0",
    "nn6a_step0_vs_ba0",
    "align_confidence50_step5",
    "nn7v2_local5_gate10_ba0",
    "nn7v2_lmkidw1_gate10_ba0",
    "nn7v2_lmkidw3_gate10_ba0",
    "nn7v2_lmkidw5_gate10_ba0",
    "nn7v2_lmkidw3_gate05_ba0",
    "nn7v2_lmkidw3_gate20_ba0",
    "nn7v2_local3_gate05_ba0",
    "nn7v2_lmkidw3_gate065_ba0",
    "nn7v2_lmkidw3_gate075_ba0",
    "nn7v2_lmkidw3_gate085_ba0",
    "nn7v2_lmkidw3_gate05_sigma12_ba0",
    "nn7v2_lmkidw3_gate05_sigma35_ba0",
    "matrix24_nn7v2_lmkidw3_gate05_ba0",
    "matrix24_nn7v2_lmkidw3_gate065_ba0",
    "n3a_alignrepair_core50_div8",
    "n3a_alignrepair_core35_div8",
    "n3a_alignrepair_core50_late8_div8",
    "n3a_alignrepair_core50_erode20_div8",
    "nn7v2_lmkidw3_gate065_semantic18_ba0",
    "nn7v2_lmkidw3_gate065_semantic25_ba0",
    "nn7v2_lmkidw3_staged_up002_up1065_ba0",
    "nn7v2_lmkidw3_staged_up003_up1075_ba0",
    "nn7v2_lmkidw3_gate075_erode22_ba0",
    "nn7v2_lmkidw3_gate065_late8_ba0",
    "nn7v2_lmkidw3_gate065_wrongref_cycle_ba0",
    "n3a_core60_standard_div8",
    "n3a_core68_late7_div8",
    "n3a_core68_late8_div8",
    "n3a_core68_erode15_div8",
    "n3a_core68_erode20_div8",
    "n3a_fullgrid_up_dual25_div8",
    "n3a_fullgrid_up_dual35_div8",
    "n3a_fullgrid_up_dual50_div8",
]


def load_completed() -> dict[str, dict]:
    latest: dict[str, tuple[float, dict]] = {}
    for path in EXPERIMENTS.glob("*/metrics_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("decision") == "error" or not payload.get("experiment_id"):
            continue
        stamp = path.stat().st_mtime
        experiment_id = str(payload["experiment_id"])
        if experiment_id not in latest or stamp > latest[experiment_id][0]:
            payload["_summary_path"] = str(path)
            latest[experiment_id] = (stamp, payload)
    return {key: value[1] for key, value in latest.items()}


def number(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def read_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def overview_page(pdf: PdfPages, summaries: list[dict]) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "PhotoMaker branched attention — step-zero visual summary",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.05,
        0.925,
        "Safety screen: face MAE ≥ 0.012; outside MAE ≤ 0.015; landmarks ≤ 0.08; "
        "bbox IoU ≥ 0.60; median reference gain ≥ 0.003 and ≥75% positive.",
        fontsize=10,
    )
    columns = [
        "experiment",
        "n",
        "baseline",
        "decision",
        "face MAE",
        "ref gain",
        "positive",
        "landmark",
        "bbox IoU",
        "outside",
    ]
    rows = []
    for item in summaries:
        causal = item.get("median_face_mae_vs_ba0") is not None
        suffix = "_vs_ba0" if causal else "_vs_pm"
        rows.append(
            [
                item.get("experiment_id", ""),
                str(item.get("sample_count", "—")),
                "BA0" if causal else "PM",
                item.get("decision", ""),
                number(item.get(f"median_face_mae{suffix}")),
                number(item.get(f"median_reference_gain{suffix}")),
                number(item.get(
                    "positive_reference_gain_vs_ba0_fraction"
                    if causal else "positive_reference_gain_fraction"
                ), 2),
                number(item.get(f"median_landmark_displacement{suffix}")),
                number(item.get(f"median_bbox_iou{suffix}")),
                number(item.get(f"median_outside_mae{suffix}")),
            ]
        )
    ax = fig.add_axes([0.035, 0.07, 0.93, 0.82])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=[0.225, 0.032, 0.047, 0.16, 0.072, 0.072, 0.057, 0.072, 0.067, 0.072],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1, 1.42)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#23364d")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf2f7")
        if row > 0 and col == 3:
            decision = rows[row - 1][3]
            if decision == "promising_step0_candidate":
                cell.set_facecolor("#c9ead3")
            elif "unsafe" in decision:
                cell.set_facecolor("#f3c6c3")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def metric_text(row: pd.Series) -> str:
    text = (
        f"face={number(row.get('face_mae_vs_pm'))}  "
        f"gain={number(row.get('reference_gain_vs_pm'))}\n"
        f"lmk={number(row.get('landmark_displacement_vs_pm'))}  "
        f"IoU={number(row.get('bbox_iou_vs_pm'))}  "
        f"out={number(row.get('outside_mae_vs_pm'))}"
    )
    if pd.notna(row.get("face_mae_vs_ba0")):
        text += (
            f"\nvs BA0: face={number(row.get('face_mae_vs_ba0'))}  "
            f"gain={number(row.get('reference_gain_vs_ba0'))}  "
            f"out={number(row.get('outside_mae_vs_ba0'))}"
        )
        if (
            pd.notna(row.get("ppr_reference_source_index"))
            and int(row.get("ppr_reference_source_index")) != int(row.get("source_index"))
        ):
            text += (
                f"\nbranch ref={int(row.get('ppr_reference_source_index'))}  "
                f"branch gain={number(row.get('ppr_reference_gain_vs_ba0'))}"
            )
    return text


def experiment_pages(pdf: PdfPages, summary: dict, rows_per_page: int) -> int:
    experiment_dir = Path(summary["experiment_dir"])
    metrics_path = experiment_dir / "metrics_per_sample.csv"
    if not metrics_path.exists():
        return 0
    metrics = pd.read_csv(metrics_path).sort_values("source_index")
    pages = 0
    for start in range(0, len(metrics), rows_per_page):
        chunk = metrics.iloc[start : start + rows_per_page]
        fig, axes = plt.subplots(
            len(chunk),
            6,
            figsize=(17, 3.25 * len(chunk) + 1.15),
            squeeze=False,
        )
        fig.suptitle(
            f"{summary['experiment_id']}  |  {summary.get('decision', '')}  |  "
            f"n={summary.get('sample_count', len(metrics))}",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        images_dir = experiment_dir / "images"
        for row_index, (_, row) in enumerate(chunk.iterrows()):
            source_index = int(row["source_index"])
            prefix = f"sample_{source_index:02d}"
            target_ref = read_image(images_dir / f"{prefix}_reference.png")
            ppr_ref = read_image(images_dir / f"{prefix}_ppr_reference.png")
            pm = read_image(images_dir / f"{prefix}_PM0.png")
            ba = read_image(images_dir / f"{prefix}_BA.png")
            ba0 = read_image(images_dir / f"{prefix}_BA0.png")
            pm_face = read_image(images_dir / f"{prefix}_PM0_face.png")
            ba_face = read_image(images_dir / f"{prefix}_BA_face.png")
            branch_ref = ppr_ref or target_ref
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
            for column_index, (title, image) in enumerate(panels):
                ax = axes[row_index, column_index]
                if image is not None:
                    ax.imshow(image)
                ax.axis("off")
                ax.set_title(
                    f"{title}\nsample {source_index}" if row_index == 0 else title,
                    fontsize=9,
                )
            axes[row_index, 5].text(
                0.5,
                -0.08,
                metric_text(row),
                ha="center",
                va="top",
                transform=axes[row_index, 5].transAxes,
                fontsize=7.5,
            )
        fig.subplots_adjust(top=0.94, bottom=0.04, hspace=0.30, wspace=0.04)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1
    return pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--experiment-id", action="append", default=[])
    parser.add_argument(
        "--only-requested",
        action="store_true",
        help="Do not automatically append every completed 24-case experiment.",
    )
    args = parser.parse_args()
    if args.rows_per_page < 1 or args.rows_per_page > 6:
        raise ValueError("--rows-per-page must be between 1 and 6")

    completed = load_completed()
    requested = args.experiment_id or DEFAULT_IDS
    selected = [completed[item] for item in requested if item in completed]
    # Always append completed 24-case experiments, including future repair waves.
    known = {item["experiment_id"] for item in selected}
    if not args.only_requested:
        for item in completed.values():
            if int(item.get("sample_count", 0)) == 24 and item["experiment_id"] not in known:
                selected.append(item)
                known.add(item["experiment_id"])
    if not selected:
        raise RuntimeError("No completed experiment summaries matched")

    REPORTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or REPORTS / f"{timestamp}_step0_visual_summary.pdf"
    if not output.is_absolute():
        output = HERE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "PhotoMaker branched-attention step-zero visual summary"
        metadata["Author"] = "Automated architecture search"
        metadata["Subject"] = "Visual grids and per-sample metrics"
        overview_page(pdf, selected)
        page_count = 1
        for item in selected:
            page_count += experiment_pages(pdf, item, args.rows_per_page)
    print(json.dumps({"output": str(output), "experiments": len(selected), "pages": page_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
