#!/usr/bin/env python3
"""Freeze and summarize the live CL21--CL26 Comet validation snapshot.

Run from ``diffusion_template/`` with ``COMET_API_KEY`` in the environment.
The immutable experiment key is the only lookup key.  Tables are joined by the
sealed ``image_index``; filename normalization is used only to retrieve images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import API
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/assets/cl21_cl26_20260814_current"
BBOX_JSON = ROOT.parent / "dataset_full/val_dataset/pm96_bboxes_new_auto.json"
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_SAMPLES = 100_000

RUNS = {
    "PM0": {
        "key": "74efd227d3f8488a98e83d815c77c07c",
        "name": "PM0_original_photomaker_CL19_full96_r1",
        "selected_step": 0,
    },
    "CL19": {
        "key": "cfeda7b55c174b3c83e8d40537ebb6dd",
        "name": "CL19_cosmic_true_soft_fullquery_router_24k_full96_r2",
        "selected_step": 24000,
    },
    "CL21": {
        "key": "6670db89c44a489388b8f09b91423b0d",
        "name": "CL21_cosmic_true_soft_router_resididca_v3_24k_full96_r2",
        "selected_step": 10000,
    },
    "CL22": {
        "key": "b181feb6c54644e69fb7e8709a59f32e",
        "name": "CL22_cosmic_visibility_order_router_24k_full96_r2",
        "selected_step": 10000,
    },
    "CL23": {
        "key": "a9ec9c59d1624c68acb98737dcd65298",
        "name": "CL23_cosmic_temporal_frequency_router_24k_full96_r1",
        "selected_step": 12000,
    },
    "CL24": {
        "key": "a18e22ae9f0e4a24b6252f6b392fab62",
        "name": "CL24_cosmic_pm_boundary_distill_24k_full96_r1",
        "selected_step": 14000,
    },
    "CL25": {
        "key": "120b72df8134474ca094e6162d085eb0",
        "name": "CL25_cosmic_low_noise_id_reward_4k_full96_r2",
        "selected_step": 4000,
    },
    "CL26": {
        "key": "e9c0a9b505f041a68a183ca3cb4ca0af",
        "name": "CL26_cosmic_anchored_highres_roi_ba_24k_full96_r3",
        "selected_step": 10000,
    },
}

METRICS = (
    "manual_val/id_sim",
    "manual_val/id_sim_legacy_best",
    "manual_val/id_sim_mask_iou",
    "manual_val/id_sim_face_count",
    "manual_val/text_sim",
    "face_quality/face_detection_rate",
    "face_quality/topiq_face_mean",
    "face_quality/topiq_face_p10",
    "face_quality/topiq_mean",
    "face_quality/musiq_mean",
    "face_quality/maniqa_mean",
)

ACTION_BY_OFFSET = (
    "Reading", "Rushing", "Skiing", "Drumming", "Kickboxing", "Dancing",
    "Angry", "Crying", "Laughing", "Jumping", "Night ride", "Chef",
)
IDENTITIES = ("eddie", "elon", "jennie", "jensen", "jisoo", "keanu", "lex", "marion")
TABLE_RE = re.compile(r"^id_sim__manual_val__step_(\d{6})\.csv$")
COMET_SUFFIX_RE = re.compile(r" \(\d+\)$")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_points(experiment, metric: str) -> list[dict]:
    points = []
    for raw in experiment.get_metrics(metric):
        if raw.get("step") is None or raw.get("metricValue") is None:
            continue
        points.append({"step": int(raw["step"]), "value": float(raw["metricValue"])})
    return sorted(points, key=lambda item: item["step"])


def parse_table(data: bytes, expected_step: int, label: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    indices = sorted(int(row["image_index"]) for row in rows)
    assert indices == list(range(96)), f"{label} step {expected_step}: unsealed table"
    assert {int(row["validation_step"]) for row in rows} == {expected_step}
    parsed = []
    for row in rows:
        item = dict(row)
        item["image_index"] = int(item["image_index"])
        item["action"] = ACTION_BY_OFFSET[item["image_index"] % 12]
        for field in (
            "id_sim", "id_sim_legacy_best", "id_sim_mask_iou",
            "id_sim_face_count", "id_sim_no_face", "id_sim_unowned",
            "id_sim_ambiguous",
        ):
            item[field] = float(item[field])
        parsed.append(item)
    return sorted(parsed, key=lambda item: item["image_index"])


def canonical_stem(name: str) -> str:
    stem = Path(name).stem.replace(" ", "_")
    return COMET_SUFFIX_RE.sub("", stem.replace("_(", " (")).replace(" ", "_")


def valid_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def paired_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(lo), float(hi)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_bboxes() -> dict[int, list[int]]:
    raw = json.loads(BBOX_JSON.read_text(encoding="utf-8"))
    return {int(value["_meta"]["debug_idx"]): value["face_crop_new"] for value in raw.values()}


def full_tile(path: Path, score: float, size: int = 220) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 24), fill=(0, 0, 0))
    draw.text((5, 4), f"ID {score:.3f}", fill="white", font=font(14))
    return image


def face_tile(path: Path, bbox: list[int], score: float, size: int = 220) -> Image.Image:
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.6
    crop = (
        max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
        min(image.width, int(cx + side / 2)), min(image.height, int(cy + side / 2)),
    )
    tile = image.crop(crop).resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 24), fill=(0, 0, 0))
    draw.text((5, 4), f"ID {score:.3f}", fill="white", font=font(14))
    return tile


def build_grid(
    action: str,
    rows: dict[str, dict[int, dict]],
    image_paths: dict[tuple[str, int], Path],
    bboxes: dict[int, list[int]],
    crop_face: bool,
) -> None:
    labels = tuple(RUNS)
    tile, header, side = 220, 38, 90
    canvas = Image.new("RGB", (side + len(labels) * tile, header + len(IDENTITIES) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((side + column * tile + 5, 8), label, fill="black", font=font(17))
    for row_index, identity in enumerate(IDENTITIES):
        draw.text((5, header + row_index * tile + 8), identity, fill="black", font=font(16))
        for column, label in enumerate(labels):
            match = next(item for item in rows[label].values() if item["identity"] == identity and item["action"] == action)
            path = image_paths[(label, match["image_index"])]
            if crop_face:
                tile_image = face_tile(path, bboxes[match["image_index"]], match["id_sim"], tile)
            else:
                tile_image = full_tile(path, match["id_sim"], tile)
            canvas.paste(tile_image, (side + column * tile, header + row_index * tile))
    suffix = "face_crops" if crop_face else "full_images"
    canvas.save(OUT / f"{action.lower()}_current_{suffix}.jpg", quality=94)


def build_single_run_sheets(
    label: str,
    rows: dict[int, dict],
    image_paths: dict[tuple[str, int], Path],
    bboxes: dict[int, list[int]],
) -> None:
    tile, header, side = 340, 42, 90
    for crop_face in (False, True):
        canvas = Image.new(
            "RGB", (side + 2 * tile, header + len(IDENTITIES) * tile), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for column, action in enumerate(("Skiing", "Crying")):
            draw.text(
                (side + column * tile + 8, 9), f"{label} — {action}",
                fill="black", font=font(20),
            )
        for row_index, identity in enumerate(IDENTITIES):
            draw.text((5, header + row_index * tile + 8), identity, fill="black", font=font(16))
            for column, action in enumerate(("Skiing", "Crying")):
                match = next(
                    item for item in rows.values()
                    if item["identity"] == identity and item["action"] == action
                )
                path = image_paths[(label, match["image_index"])]
                tile_image = (
                    face_tile(path, bboxes[match["image_index"]], match["id_sim"], tile)
                    if crop_face else full_tile(path, match["id_sim"], tile)
                )
                canvas.paste(tile_image, (side + column * tile, header + row_index * tile))
        suffix = "face_crops" if crop_face else "full_images"
        canvas.save(OUT / f"hardcases_{label}_{suffix}.jpg", quality=90, optimize=True)


def build_curve_figure(histories: dict[str, dict[str, list[dict]]]) -> None:
    colors = {
        "CL19": "#6b7280", "CL21": "#8b5cf6", "CL22": "#f59e0b",
        "CL23": "#005bbb", "CL24": "#dc2626", "CL25": "#059669", "CL26": "#7c3aed",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for label in ("CL19", "CL21", "CL22", "CL23", "CL24", "CL25", "CL26"):
        points = histories[label]["manual_val/id_sim"]
        ax.plot(
            [point["step"] / 1000 for point in points],
            [point["value"] for point in points],
            marker="o", ms=3.5, lw=3 if label == "CL23" else 1.5,
            color=colors[label], alpha=1 if label == "CL23" else 0.82, label=label,
        )
    pm = histories["PM0"]["manual_val/id_sim"][0]["value"]
    ax.axhline(pm, color="#111827", ls="--", lw=2, label=f"PhotoMaker {pm:.3f}")
    ax.set(
        xlabel="Optimizer step (thousands)",
        ylabel="manual_val subject-v2 ID similarity",
        title="Current CL21--CL26 snapshot: CL23 leads, PhotoMaker remains higher",
    )
    ax.set_ylim(0.39, 0.57)
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "id_trajectories_current.png", dpi=200)
    plt.close(fig)


def hash_outputs() -> None:
    paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-run", choices=tuple(RUNS),
        help="Cache one run's sealed tables and hard-case images, then exit.",
    )
    parser.add_argument(
        "--sheet-run", choices=tuple(RUNS),
        help="Build compact hard-case sheets from an already cached run, then exit.",
    )
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Build numerical outputs without re-downloading hard-case images.",
    )
    args = parser.parse_args()
    if args.cache_run and args.sheet_run:
        parser.error("--cache-run and --sheet-run are mutually exclusive")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "hardcase_images").mkdir(exist_ok=True)
    if args.sheet_run:
        label = args.sheet_run
        step = int(RUNS[label]["selected_step"])
        table_path = OUT / "tables" / f"{label}_step_{step:06d}.csv"
        rows = parse_table(table_path.read_bytes(), step, label)
        row_map = {row["image_index"]: row for row in rows}
        paths = {}
        for row in rows:
            if row["action"] not in {"Skiing", "Crying"}:
                continue
            candidates = list((OUT / "hardcase_images").glob(f"{label}_{row['image_index']:02d}_*.png"))
            assert len(candidates) == 1 and valid_image(candidates[0]), (label, row["image_index"], candidates)
            paths[(label, row["image_index"])] = candidates[0]
        build_single_run_sheets(label, row_map, paths, load_bboxes())
        print(f"SHEETS {label}")
        return
    api = API()
    captured_at = datetime.now(timezone.utc).isoformat()
    histories: dict[str, dict[str, list[dict]]] = {}
    tables: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    image_paths: dict[tuple[str, int], Path] = {}
    comments: dict[str, str | None] = {}
    metric_rows: list[dict] = []

    selected_specs = (
        {args.cache_run: RUNS[args.cache_run]} if args.cache_run else RUNS
    )
    for label, spec in selected_specs.items():
        experiment = api.get_experiment_by_key(spec["key"])
        assert experiment.name == spec["name"], (label, experiment.name)
        assets = experiment.get_asset_list()
        histories[label] = {}
        for metric in METRICS:
            points = metric_points(experiment, metric)
            histories[label][metric] = points
            metric_rows.extend({"run": label, "metric": metric, **point} for point in points)
        other = experiment.get_others_summary("experiment_comment")
        comments[label] = other[-1] if other else None

        table_assets = []
        for asset in assets:
            match = TABLE_RE.match(asset.get("fileName", ""))
            if match:
                table_assets.append((int(match.group(1)), asset))
        assert table_assets, f"{label}: no validation tables"
        for step, asset in sorted(table_assets):
            path = OUT / "tables" / f"{label}_step_{step:06d}.csv"
            data = path.read_bytes() if path.exists() else experiment.get_asset(asset["assetId"])
            rows = parse_table(data, step, label)
            path.write_bytes(data)
            tables[label][step] = rows

        selected = int(spec["selected_step"])
        assert selected in tables[label], f"{label}: selected table {selected} missing"
        if args.metrics_only:
            continue
        expected = {
            Path(row["output_key"]).stem.replace(" ", "_"): row
            for row in tables[label][selected]
            if row["action"] in {"Skiing", "Crying"}
        }
        selected_assets = [
            asset for asset in assets
            if asset.get("type") == "image" and int(asset.get("step", -1)) == selected
            and asset.get("fileName", "").startswith(("Skiing", "Crying"))
        ]
        for asset in selected_assets:
            canonical = canonical_stem(asset["fileName"])
            if canonical not in expected:
                continue
            row = expected.pop(canonical)
            path = OUT / "hardcase_images" / f"{label}_{row['image_index']:02d}_{canonical}.png"
            if not valid_image(path):
                path.write_bytes(experiment.get_asset(asset["assetId"]))
            assert valid_image(path), f"{label}: invalid image asset {path.name}"
            image_paths[(label, row["image_index"])] = path
        assert not expected, f"{label}: missing hard-case assets {sorted(expected)}"

    if args.cache_run:
        step = int(RUNS[args.cache_run]["selected_step"])
        row_map = {row["image_index"]: row for row in tables[args.cache_run][step]}
        build_single_run_sheets(args.cache_run, row_map, image_paths, load_bboxes())
        print(f"CACHED {args.cache_run}")
        return

    write_csv(OUT / "metric_history.csv", metric_rows)

    selected_rows: dict[str, dict[int, dict]] = {
        label: {row["image_index"]: row for row in tables[label][int(spec["selected_step"])]}
        for label, spec in RUNS.items()
    }
    per_image: list[dict] = []
    for index in range(96):
        base = selected_rows["CL23"][index]
        record = {
            "image_index": index, "output_key": base["output_key"],
            "identity": base["identity"], "action": base["action"],
        }
        for label in RUNS:
            record[f"id_{label}"] = selected_rows[label][index]["id_sim"]
        per_image.append(record)
    write_csv(OUT / "per_image_selected.csv", per_image)

    endpoint_rows = []
    for label, spec in RUNS.items():
        step = int(spec["selected_step"])
        record = {
            "run": label, "comet_key": spec["key"], "selected_step": step,
            "latest_complete_step_at_capture": max(tables[label]),
        }
        for metric in METRICS:
            values = [point["value"] for point in histories[label][metric] if point["step"] == step]
            record[metric] = values[-1] if values else ""
        endpoint_rows.append(record)
    write_csv(OUT / "selected_endpoints.csv", endpoint_rows)

    comparison_rows = []
    for label in ("CL21", "CL22", "CL23", "CL24", "CL25", "CL26"):
        for step in (0, 2000, 4000):
            left = np.array([row["id_sim"] for row in tables[label][step]])
            right = np.array([row["id_sim"] for row in tables["CL19"][step]])
            delta = left - right
            lo, hi = paired_interval(delta)
            comparison_rows.append({
                "comparison": f"{label}-CL19", "step": step,
                "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                "wins": int((delta > 0).sum()), "n": len(delta), "ci_low": lo, "ci_high": hi,
            })
    for step in (6000, 8000, 10000, 12000):
        left = np.array([row["id_sim"] for row in tables["CL23"][step]])
        right = np.array([row["id_sim"] for row in tables["CL19"][step]])
        delta = left - right
        lo, hi = paired_interval(delta)
        comparison_rows.append({
            "comparison": "CL23-CL19", "step": step,
            "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "wins": int((delta > 0).sum()), "n": len(delta), "ci_low": lo, "ci_high": hi,
        })
    for baseline in ("CL19", "PM0"):
        selected_step = int(RUNS["CL23"]["selected_step"])
        delta = np.array([
            selected_rows["CL23"][index]["id_sim"] - selected_rows[baseline][index]["id_sim"]
            for index in range(96)
        ])
        lo, hi = paired_interval(delta)
        comparison_rows.append({
            "comparison": f"CL23@{selected_step // 1000}k-{baseline}@{RUNS[baseline]['selected_step']}",
            "step": selected_step,
            "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "wins": int((delta > 0).sum()), "n": len(delta), "ci_low": lo, "ci_high": hi,
        })
    write_csv(OUT / "paired_comparisons.csv", comparison_rows)

    slice_rows = []
    for label, rows in selected_rows.items():
        for dimension, names in (("action", ACTION_BY_OFFSET), ("identity", IDENTITIES)):
            for name in names:
                subset = [row["id_sim"] for row in rows.values() if row[dimension] == name]
                slice_rows.append({
                    "run": label, "selected_step": RUNS[label]["selected_step"],
                    "dimension": dimension, "slice": name, "n": len(subset),
                    "mean_id_sim": float(np.mean(subset)),
                })
    write_csv(OUT / "slice_means_selected.csv", slice_rows)

    hard_rows = []
    for label, rows in selected_rows.items():
        for action in ("Skiing", "Crying"):
            subset = [row for row in rows.values() if row["action"] == action]
            for row in subset:
                hard_rows.append({
                    "run": label, "selected_step": RUNS[label]["selected_step"],
                    "action": action, "identity": row["identity"],
                    "image_index": row["image_index"], "id_sim": row["id_sim"],
                    "mask_iou": row["id_sim_mask_iou"], "face_count": row["id_sim_face_count"],
                })
    write_csv(OUT / "hardcase_rows_selected.csv", hard_rows)

    build_curve_figure(histories)
    if not args.metrics_only:
        bboxes = load_bboxes()
        for action in ("Skiing", "Crying"):
            build_grid(action, selected_rows, image_paths, bboxes, crop_face=False)
            build_grid(action, selected_rows, image_paths, bboxes, crop_face=True)

    manifest = {
        "captured_at_utc": captured_at,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "runs": RUNS,
        "comments": comments,
        "table_steps": {label: sorted(step_rows) for label, step_rows in tables.items()},
    }
    (OUT / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hash_outputs()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
