#!/usr/bin/env python3
"""Build compact, reproducible CL30-CL37 result tables and visual sheets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
TEMP_ROOT = Path("/mnt/c/Users/ogure/AppData/Local/Temp/cl30_cl37_20260819")
PEAK_ROOT = TEMP_ROOT / "peak"
FINAL_ROOT = TEMP_ROOT / "final"
PRIOR_ASSETS = PROJECT / "analysis/assets/cl27_cl29_vs_cl23_20260817"

RUNS = {
    "PM0": {
        "key": "74efd227d3f8488a98e83d815c77c07c",
        "name": "PM0_original_photomaker_CL19_full96_r1",
        "step": 0,
        "root": PEAK_ROOT,
    },
    "CL27": {
        "key": "dbfbf40c3bdd4f70bedc58bda3dfb9cd",
        "name": "CL27_cosmic_frequency_surface_energy_24k_full96_r3",
        "step": 16000,
        "root": PEAK_ROOT,
    },
    "CL30": {
        "key": "db38cfb250d241cf89bf57705ff86b18",
        "name": "CL30_cosmic_positive_lowband_sameid_24k_full96_r4",
        "step": 16000,
        "root": PEAK_ROOT,
    },
    "CL31": {
        "key": "ed5077fd3cfc41bd898c1234b8c3ba24",
        "name": "CL31_cosmic_attention_ownership_alignment_24k_full96_r4",
        "step": 24000,
        "root": FINAL_ROOT,
    },
    "CL32": {
        "key": "078cf231674f4fa499e160a435300511",
        "name": "CL32_cosmic_contact_frequency_surface_24k_full96_r1",
        "step": 18000,
        "root": PEAK_ROOT,
    },
    "CL33": {
        "key": "3173f3086fa344f7ad3eb6ce7b07ac1f",
        "name": "CL33_cosmic_visibility_balanced_reconstruction_24k_full96_r1",
        "step": 16000,
        "root": PEAK_ROOT,
    },
    "CL34": {
        "key": "577cc412ffa04e5686e5c10760186c65",
        "name": "CL34_cosmic_shared_frequency_calibration_24k_full96_r4",
        "step": 18000,
        "root": PEAK_ROOT,
    },
    "CL35": {
        "key": "f3417ee9a86342cb9bc13e5eb37bb3e2",
        "name": "CL35_cosmic_attention_gated_patch_identity_24k_full96_r7",
        "step": 24000,
        "root": FINAL_ROOT,
    },
    "CL36": {
        "key": "41dcb0987d5d439bb14329052953ff6d",
        "name": "CL36_cosmic_ba_arcface_hinge_4k_full96_r4",
        "step": 4000,
        "root": FINAL_ROOT,
    },
    "CL37": {
        "key": "f3c535315da242d78494d7df6dd1eaa3",
        "name": "CL37_cosmic_smallface_roi_teacher_distill_24k_full96_r4",
        "step": 18000,
        "root": PEAK_ROOT,
    },
}

FINAL_STEPS = {
    "CL30": 24000,
    "CL31": 24000,
    "CL32": 24000,
    "CL33": 24000,
    "CL34": 24000,
    "CL35": 24000,
    "CL36": 4000,
    "CL37": 24000,
}

PANEL_A = ("PM0", "CL27", "CL30", "CL31", "CL32")
PANEL_B = ("CL27", "CL33", "CL34", "CL35", "CL36", "CL37")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def normalize_key(value: str) -> str:
    return Path(value).name.replace(" ", "_")


def run_dir(label: str) -> Path:
    item = RUNS[label]
    return Path(item["root"]) / str(item["name"])


def table_path(label: str, step: int | None = None, final: bool = False) -> Path:
    item = RUNS[label]
    actual_step = int(step if step is not None else item["step"])
    root = FINAL_ROOT if final else Path(item["root"])
    return root / str(item["name"]) / "_tables" / f"id_sim__manual_val__step_{actual_step:06d}.csv"


def load_rows(label: str, step: int | None = None, final: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with table_path(label, step, final).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in (
                "id_sim",
                "id_sim_legacy_best",
                "id_sim_mask_iou",
                "id_sim_face_count",
                "id_sim_no_face",
                "id_sim_unowned",
                "id_sim_ambiguous",
            ):
                row[key] = float(raw[key])
            row["image_index"] = int(raw["image_index"])
            row["normalized_key"] = normalize_key(raw["output_key"])
            rows.append(row)
    if len(rows) != 96:
        raise RuntimeError(f"{label}: expected 96 table rows, found {len(rows)}")
    return rows


def load_manifests() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in (PEAK_ROOT / "comet_runs_export.json", FINAL_ROOT / "comet_runs_export.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            result[str(run["id"])] = run
    return result


MANIFESTS = load_manifests()


def metric_history(label: str, metric: str) -> dict[int, float]:
    run = MANIFESTS[str(RUNS[label]["key"])]
    return {
        int(item["step"]): float(item["value"])
        for item in run["metrics"].get(metric, [])
        if item.get("step") is not None
    }


def metric_at(label: str, metric: str, step: int) -> float | None:
    return metric_history(label, metric).get(step)


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 50000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        end = min(start + 1000, draws)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def control_rows(label: str) -> tuple[list[dict[str, object]], str, int]:
    if label == "CL36":
        return load_rows("CL27", 16000), "CL27-source", 16000
    step = int(RUNS[label]["step"])
    prior = PRIOR_ASSETS / "tables" / f"CL27_step_{step:06d}.csv"
    if not prior.exists():
        raise RuntimeError(f"Missing matched CL27 table for {label} at {step}")
    rows = []
    with prior.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("id_sim", "id_sim_mask_iou", "id_sim_face_count"):
                row[key] = float(raw[key])
            row["image_index"] = int(raw["image_index"])
            rows.append(row)
    return rows, "CL27-matched", step


def build_tables() -> None:
    selected_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []

    metrics = (
        "manual_val/id_sim",
        "manual_val/text_sim",
        "manual_val/id_sim_mask_iou",
        "manual_val/id_sim_face_count",
        "manual_val/id_sim_no_face",
        "manual_val/id_sim_unowned",
        "manual_val/id_sim_ambiguous",
        "face_quality/topiq_face_mean",
        "face_quality/topiq_face_p10",
        "face_quality/face_detection_rate",
    )
    for label in RUNS:
        histories = {metric: metric_history(label, metric) for metric in metrics}
        for step in sorted(histories["manual_val/id_sim"]):
            history_rows.append(
                {"run": label, "step": step, **{metric: histories[metric].get(step) for metric in metrics}}
            )

    pm_rows = load_rows("PM0")
    pm_by_index = {int(row["image_index"]): row for row in pm_rows}
    for label, item in RUNS.items():
        step = int(item["step"])
        rows = load_rows(label)
        values = np.array([float(row["id_sim"]) for row in rows])
        selected_rows.append(
            {
                "run": label,
                "comet_key": item["key"],
                "selected_step": step,
                "id_sim": values.mean(),
                "text_sim": metric_at(label, "manual_val/text_sim", step),
                "mask_iou": np.mean([float(row["id_sim_mask_iou"]) for row in rows]),
                "face_count": np.mean([float(row["id_sim_face_count"]) for row in rows]),
                "topiq_face_mean": metric_at(label, "face_quality/topiq_face_mean", step),
                "topiq_face_p10": metric_at(label, "face_quality/topiq_face_p10", step),
                "delta_vs_PM0": values.mean() - np.mean([float(row["id_sim"]) for row in pm_rows]),
            }
        )
        for row in rows:
            per_image_rows.append(
                {
                    "run": label,
                    "step": step,
                    "image_index": row["image_index"],
                    "output_key": row["output_key"],
                    "identity": row["identity"],
                    "prompt": row["prompt"],
                    "id_sim": row["id_sim"],
                    "mask_iou": row["id_sim_mask_iou"],
                    "face_count": row["id_sim_face_count"],
                }
            )

        if label not in {"PM0", "CL27"}:
            control, control_label, control_step = control_rows(label)
            control_by_index = {int(row["image_index"]): row for row in control}
            deltas = np.array(
                [float(row["id_sim"]) - float(control_by_index[int(row["image_index"])]["id_sim"]) for row in rows]
            )
            low, high = bootstrap_interval(deltas, 20260819 + int(label[2:]))
            paired_rows.append(
                {
                    "run": label,
                    "step": step,
                    "control": control_label,
                    "control_step": control_step,
                    "mean_delta": deltas.mean(),
                    "wins": int((deltas > 0).sum()),
                    "ties": int((deltas == 0).sum()),
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                }
            )
        if label != "PM0":
            deltas_pm = np.array(
                [float(row["id_sim"]) - float(pm_by_index[int(row["image_index"])]["id_sim"]) for row in rows]
            )
            low, high = bootstrap_interval(deltas_pm, 20260919 + (27 if label == "CL27" else int(label[2:])))
            paired_rows.append(
                {
                    "run": label,
                    "step": step,
                    "control": "PM0",
                    "control_step": 0,
                    "mean_delta": deltas_pm.mean(),
                    "wins": int((deltas_pm > 0).sum()),
                    "ties": int((deltas_pm == 0).sum()),
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                }
            )

        for field in ("identity", "prompt"):
            groups: dict[str, list[float]] = {}
            for row in rows:
                value = str(row[field])
                if field == "prompt":
                    value = value.split()[0]
                groups.setdefault(value, []).append(float(row["id_sim"]))
            for group, group_values in sorted(groups.items()):
                slice_rows.append(
                    {
                        "run": label,
                        "step": step,
                        "slice_type": field,
                        "slice": group,
                        "count": len(group_values),
                        "mean_id_sim": float(np.mean(group_values)),
                    }
                )

    write_csv(ASSET_DIR / "aggregate_history.csv", history_rows)
    write_csv(ASSET_DIR / "selected_summary.csv", selected_rows)
    write_csv(ASSET_DIR / "paired_comparisons.csv", paired_rows)
    write_csv(ASSET_DIR / "slice_means.csv", slice_rows)
    write_csv(ASSET_DIR / "per_image_selected.csv", per_image_rows)


def build_trajectory_plot() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {
        "CL27": "#d62728",
        "CL30": "#4c78a8",
        "CL31": "#72b7b2",
        "CL32": "#f58518",
        "CL33": "#54a24b",
        "CL34": "#b279a2",
        "CL35": "#ff9da6",
        "CL37": "#9d755d",
    }
    for label in ("CL27", "CL30", "CL31", "CL32", "CL33", "CL34", "CL35", "CL37"):
        history = metric_history(label, "manual_val/id_sim")
        steps = sorted(step for step in history if step <= 24000)
        lw = 2.8 if label in {"CL27", "CL33"} else 1.45
        alpha = 1.0 if label in {"CL27", "CL33"} else 0.78
        ax.plot(steps, [history[step] for step in steps], marker="o", ms=3, lw=lw, alpha=alpha, label=label, color=colors[label])
    ax.axhline(metric_history("PM0", "manual_val/id_sim")[0], color="#333333", ls="--", lw=1.2, label="PhotoMaker")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("manual_val/id_sim (subject-v2)")
    ax.set_xticks(range(0, 24001, 4000), [f"{step // 1000}k" for step in range(0, 24001, 4000)])
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=8)

    selected = ["CL30", "CL31", "CL32", "CL33", "CL34", "CL35", "CL36", "CL37"]
    paired = list(csv.DictReader((ASSET_DIR / "paired_comparisons.csv").open(encoding="utf-8")))
    deltas = {row["run"]: float(row["mean_delta"]) for row in paired if row["control"].startswith("CL27")}
    lows = {row["run"]: float(row["bootstrap_low"]) for row in paired if row["control"].startswith("CL27")}
    highs = {row["run"]: float(row["bootstrap_high"]) for row in paired if row["control"].startswith("CL27")}
    values = [deltas[label] for label in selected]
    errors = [[values[i] - lows[label] for i, label in enumerate(selected)], [highs[label] - values[i] for i, label in enumerate(selected)]]
    bar_colors = ["#54a24b" if label == "CL33" else "#9aa0a6" for label in selected]
    ax2.bar(selected, values, color=bar_colors, yerr=errors, capsize=3)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Paired ID delta vs CL27 control")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "id_sim_results.png", dpi=190)
    plt.close(fig)


def load_boxes() -> dict[str, list[float]]:
    raw = json.loads((PRIOR_ASSETS / "pm96_bboxes_new_auto.json").read_text(encoding="utf-8"))
    return {
        normalize_key(key): [float(value) for value in (entry.get("face_crop_new") or entry["face_crop_old"])]
        for key, entry in raw.items()
    }


BOXES = load_boxes()


def crop_face(image: Image.Image, box: list[float], scale: float = 1.9) -> Image.Image:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * scale
    left = max(0.0, min(float(image.width) - side, cx - side / 2))
    top = max(0.0, min(float(image.height) - side, cy - side / 2))
    return image.crop((int(left), int(top), int(min(image.width, left + side)), int(min(image.height, top + side))))


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copied = image.copy()
    copied.thumbnail(size, LANCZOS)
    canvas.paste(copied, ((size[0] - copied.width) // 2, (size[1] - copied.height) // 2))
    return canvas


def image_path(label: str, row: dict[str, object]) -> Path:
    return run_dir(label) / str(row["normalized_key"])


def prompt_sheet(prompt_name: str, labels: tuple[str, ...], crop: bool) -> Image.Image:
    rows_by_run = {label: load_rows(label) for label in labels}
    selected = [row for row in rows_by_run[labels[0]] if str(row["prompt"]).split()[0] == prompt_name]
    selected.sort(key=lambda row: int(row["image_index"]))
    cell_w, cell_h, left_w, header_h, text_h = 215, 215, 125, 45, 42
    sheet = Image.new("RGB", (left_w + len(labels) * cell_w, header_h + len(selected) * (cell_h + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        step = int(RUNS[label]["step"])
        draw.text((left_w + col * cell_w + 7, 12), f"{label} · {step // 1000}k", font=FONT_BOLD, fill="black")
    for row_no, base in enumerate(selected):
        y = header_h + row_no * (cell_h + text_h)
        draw.text((7, y + 8), str(base["identity"]), font=FONT_BOLD, fill="black")
        draw.text((7, y + 31), f"idx {base['image_index']}", font=FONT_SMALL, fill="#555555")
        for col, label in enumerate(labels):
            row = next(item for item in rows_by_run[label] if int(item["image_index"]) == int(base["image_index"]))
            image = Image.open(image_path(label, row)).convert("RGB")
            if crop:
                image = crop_face(image, BOXES[str(row["normalized_key"])])
            x = left_w + col * cell_w
            sheet.paste(fit_image(image, (cell_w, cell_h)), (x, y))
            draw.text(
                (x + 6, y + cell_h + 7),
                f"ID {float(row['id_sim']):.3f} · IoU {float(row['id_sim_mask_iou']):.3f}",
                font=FONT_SMALL,
                fill="black",
            )
    return sheet


def selected_sheet(indices: list[int], labels: tuple[str, ...], title: str) -> Image.Image:
    rows_by_run = {label: load_rows(label) for label in labels}
    left_w, header_h, cell_w, image_h, text_h = 155, 46, 300, 150, 40
    sheet = Image.new("RGB", (left_w + len(labels) * cell_w, header_h + len(indices) * (image_h + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((7, 13), title, font=FONT_BOLD, fill="black")
    for col, label in enumerate(labels):
        draw.text((left_w + col * cell_w + 7, 13), label, font=FONT_BOLD, fill="black")
    for row_no, index in enumerate(indices):
        base = next(row for row in rows_by_run[labels[0]] if int(row["image_index"]) == index)
        y = header_h + row_no * (image_h + text_h)
        draw.text((7, y + 7), str(base["identity"]), font=FONT_BOLD, fill="black")
        draw.text((7, y + 29), f"{str(base['prompt']).split()[0]} · {index}", font=FONT_SMALL, fill="#555555")
        for col, label in enumerate(labels):
            row = next(item for item in rows_by_run[label] if int(item["image_index"]) == index)
            original = Image.open(image_path(label, row)).convert("RGB")
            x = left_w + col * cell_w
            sheet.paste(fit_image(original, (image_h, image_h)), (x, y))
            sheet.paste(fit_image(crop_face(original, BOXES[str(row["normalized_key"])]), (image_h, image_h)), (x + image_h, y))
            draw.text((x + 5, y + image_h + 7), f"ID {float(row['id_sim']):.3f} · IoU {float(row['id_sim_mask_iou']):.3f}", font=FONT_SMALL, fill="black")
    return sheet


def build_sheets() -> None:
    for prompt in ("Skiing", "Crying", "Jumping", "Dancing"):
        for suffix, labels in (("a", PANEL_A), ("b", PANEL_B)):
            prompt_sheet(prompt, labels, crop=True).save(ASSET_DIR / f"{prompt.lower()}_peak_face_{suffix}.jpg", quality=92, subsampling=0)
    selected_sheet([50, 62, 31, 43, 5, 94], ("PM0", "CL27", "CL32", "CL33", "CL36"), "Base decision").save(
        ASSET_DIR / "base_decision_critical.jpg", quality=92, subsampling=0
    )
    selected_sheet([69, 70, 27, 58, 14, 2], ("CL27", "CL30", "CL31", "CL34", "CL35", "CL37"), "Other differentiators").save(
        ASSET_DIR / "other_differentiators.jpg", quality=92, subsampling=0
    )


def copy_tables_and_registry() -> None:
    output = ASSET_DIR / "tables"
    output.mkdir(exist_ok=True)
    for label, item in RUNS.items():
        source = table_path(label)
        shutil.copyfile(source, output / f"{label}_selected_step_{int(item['step']):06d}.csv")
        if label in FINAL_STEPS and FINAL_STEPS[label] != int(item["step"]):
            source = table_path(label, FINAL_STEPS[label], final=True)
            shutil.copyfile(source, output / f"{label}_final_step_{FINAL_STEPS[label]:06d}.csv")
    shutil.copy2(PRIOR_ASSETS / "pm96_bboxes_new_auto.json", ASSET_DIR / "pm96_bboxes_new_auto.json")
    rows = [
        {
            "label": label,
            "run_name": item["name"],
            "comet_key": item["key"],
            "selected_step": item["step"],
            "source": "peak_export" if item["root"] == PEAK_ROOT else "final_export",
        }
        for label, item in RUNS.items()
    ]
    write_csv(ASSET_DIR / "run_registry.csv", rows)


def build_manifest() -> None:
    entries = []
    for path in sorted(ASSET_DIR.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt" and "__pycache__" not in path.parts:
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ASSET_DIR)}")
    (ASSET_DIR / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    copy_tables_and_registry()
    build_tables()
    build_trajectory_plot()
    build_sheets()
    build_manifest()


if __name__ == "__main__":
    main()
