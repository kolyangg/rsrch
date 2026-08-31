#!/usr/bin/env python3
"""Score the preregistered seed-1 all-on versus up1/low-off confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


MAP_SHA = "858c4663083ccffbd461e94215d4e9951f2765b59b4f49ce454de92c5910904f"


def normalized(value: str) -> str:
    return Path(value).name.replace(" ", "_")


def image_map(root: Path) -> dict[str, Path]:
    paths = sorted(root.glob("step_16000_batch_*/*.png"))
    result = {normalized(path.name): path for path in paths}
    if len(paths) != 96 or len(result) != 96:
        raise RuntimeError(f"Expected 96 unique images under {root}, got {len(paths)}/{len(result)}")
    return result


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop(image: Image.Image, box: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = map(float, box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(x1 - x0, y1 - y0) * 1.9, image.width, image.height)
    left = min(max(0.0, cx - side / 2), image.width - side)
    top = min(max(0.0, cy - side / 2), image.height - side)
    face = image.crop((int(left), int(top), int(left + side), int(top + side)))
    return np.asarray(face.resize((192, 192), Image.Resampling.LANCZOS), dtype=np.uint8)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_pages(output: Path, keys: list[str], arms: dict[str, dict[str, Path]], boxes: dict) -> list[str]:
    pages = []
    for page, start in enumerate(range(0, 96, 12), 1):
        canvas = Image.new("RGB", (1536, 12 * 300), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, key in enumerate(keys[start:start + 12]):
            for arm_index, arm in enumerate(("pm", "all_on", "up1_low_off")):
                image = Image.open(arms[arm][key]).convert("RGB").resize((256, 256))
                x, y = arm_index * 512, slot * 300
                box = boxes[key].get("face_crop_new") or boxes[key]["face_crop_old"]
                scale = 256 / 1024
                ImageDraw.Draw(image).rectangle(tuple(int(v * scale) for v in box), outline="red", width=4)
                canvas.paste(image, (x + 128, y + 8))
                draw.text((x + 16, y + 270), f"{arm} | {key}", fill="black")
        path = output / f"visual_review_page_{page:02d}.jpg"
        canvas.save(path, quality=90)
        pages.append(path.name)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-on-root", type=Path, required=True)
    parser.add_argument("--pruned-root", type=Path, required=True)
    parser.add_argument("--pm-root", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--bbox-gate", type=Path, required=True)
    parser.add_argument("--route-log", type=Path, required=True)
    parser.add_argument("--subject-v2-embeds", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    arms = {
        "all_on": image_map(args.all_on_root),
        "up1_low_off": image_map(args.pruned_root),
        "pm": {normalized(path.name): path for path in sorted(args.pm_root.glob("*.png"))},
    }
    keys = sorted(arms["all_on"])
    if any(set(values) != set(keys) for values in arms.values()):
        raise RuntimeError("CL39N6R exact 96-cell image join failed")
    boxes_raw = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    boxes = {normalized(key): value for key, value in boxes_raw.items()}
    if set(boxes) != set(keys):
        raise RuntimeError("CL39N6R seed-1 bbox join failed")
    bbox_gate = json.loads(args.bbox_gate.read_text(encoding="utf-8"))
    if not bbox_gate.get("accepted"):
        raise RuntimeError("CL39N6R seed-1 dynamic-box gate did not pass")
    route_marker = f"CL39N6R_CONFIRMATION_ROUTE_ACTIVE map_sha256={MAP_SHA}"
    if route_marker not in args.route_log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("CL39N6R route activity marker is absent")

    manifest = {"schema_version": 1, "kind": "cl39n6r_seed1", "steps": {}}
    for step, arm in enumerate(("all_on", "up1_low_off")):
        manifest["steps"][str(step)] = [
            {"asset_id": file_sha(arms[arm][key]), "file_name": f"{arm}__{key}",
             "local_path": str(arms[arm][key]), "sample_index": index}
            for index, key in enumerate(keys)
        ]
    manifest_path = args.output / "quality_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from tools.inference.calculate_face_quality_metrics import main as quality_main
    quality_main([
        "--manifest", str(manifest_path), "--output-json", str(args.output / "topiq.json"),
        "--output-csv", str(args.output / "topiq.csv"), "--metrics", "topiq_nr-face",
        "--device", args.device, "--batch-size", "1",
    ])
    quality = {}
    with (args.output / "topiq.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm, key = row["file_name"].split("__", 1)
            quality[(arm, key)] = float(row["topiq_nr_face"])

    from src.datasets.manual_val import ManualPhotoMakerValDataset
    from src.metrics.id_sim_metric import IDSimMaskMatched
    from src.metrics.text_sim import TextSimMetric
    dataset = ManualPhotoMakerValDataset(
        images_dir=str(args.references), prompts_path=str(args.prompts),
        classes_json_path=str(args.classes), bbox_mask_gen=str(args.bbox_json),
        seeds=[1], limit=96,
    )
    sample_by_key = {
        normalized(f"{sample['prompt'][:10]}_{sample['id']}.png"): sample
        for sample in dataset.samples
    }
    if set(sample_by_key) != set(keys):
        raise RuntimeError("CL39N6R prompt/identity join failed")
    identity = IDSimMaskMatched(
        id_embeds_pth=str(args.subject_v2_embeds), device="cpu",
        metric_name="id_sim_subject_v2",
    )
    text = TextSimMetric(model_name="ViT-L/14@336px", device=args.device)
    rows = []
    for index, key in enumerate(keys):
        box = boxes[key].get("face_crop_new") or boxes[key]["face_crop_old"]
        pm = Image.open(arms["pm"][key]).convert("RGB")
        pm_face = crop(pm, box)
        sample = sample_by_key[key]
        for arm in ("all_on", "up1_low_off"):
            image = Image.open(arms[arm][key]).convert("RGB")
            id_values = identity(generated=[image], face_bbox_gen=box, id=sample["id"])
            rows.append({
                "output_key": key, "arm": arm,
                "id_sim_subject_v2": float(id_values["id_sim_subject_v2"]),
                "mask_iou": float(id_values["id_sim_mask_iou"]),
                "no_face": float(id_values["id_sim_no_face"]),
                "unowned": float(id_values["id_sim_unowned"]),
                "ambiguous": float(id_values["id_sim_ambiguous"]),
                "face_ssim_to_pm": float(structural_similarity(crop(image, box), pm_face, channel_axis=2, data_range=255)),
                "topiq_face": quality[(arm, key)],
                "text_sim": float(text(prompt=sample["prompt"], generated=[image])["text_sim"]),
            })
        if (index + 1) % 12 == 0:
            print(f"CL39N6R_CONFIRM_SCORE {index + 1}/96", flush=True)
    write_csv(args.output / "per_image.csv", rows)
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ("all_on", "up1_low_off")}
    paired = {
        metric: np.asarray([candidate[metric] - baseline[metric] for baseline, candidate in zip(by_arm["all_on"], by_arm["up1_low_off"])])
        for metric in ("id_sim_subject_v2", "mask_iou", "face_ssim_to_pm", "topiq_face", "text_sim")
    }
    summary = {
        metric: {"delta_mean": float(values.mean()), "delta_p10": float(np.quantile(values, .10))}
        for metric, values in paired.items()
    }
    topiq_p10_delta = float(
        np.quantile([row["topiq_face"] for row in by_arm["up1_low_off"]], .10)
        - np.quantile([row["topiq_face"] for row in by_arm["all_on"]], .10)
    )
    summary["topiq_face"]["arm_p10_delta"] = topiq_p10_delta
    candidate = by_arm["up1_low_off"]
    checks = {
        "identity_delta": summary["id_sim_subject_v2"]["delta_mean"] >= 0.003,
        "pm_face_ssim_delta": summary["face_ssim_to_pm"]["delta_mean"] <= 0.0,
        "topiq_mean_delta": summary["topiq_face"]["delta_mean"] >= -0.003,
        "topiq_p10_delta": topiq_p10_delta >= -0.010,
        "mask_iou_delta": summary["mask_iou"]["delta_mean"] > -0.010,
        "candidate_ownership": all(sum(row[key] for row in candidate) == 0 for key in ("no_face", "unowned", "ambiguous")),
        "text_delta": summary["text_sim"]["delta_mean"] > -0.15,
        "dynamic_bbox": bool(bbox_gate["accepted"]),
        "route_activity": True,
    }
    pages = render_pages(args.output, keys, arms, boxes)
    payload = {
        "schema_version": 1,
        "status": "metrics_pass_pending_visual_review" if all(checks.values()) else "fail",
        "map_sha256": MAP_SHA,
        "validation_seed": 1,
        "exact_join_count": 96,
        "checks": checks,
        "paired_summary": summary,
        "bbox_gate": {key: bbox_gate[key] for key in ("no_face", "unowned", "mean_best_iou", "accepted")},
        "visual_review_pages": pages,
    }
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] == "fail":
        raise SystemExit("CL39N6R seed-1 metrics gate failed")


if __name__ == "__main__":
    main()
