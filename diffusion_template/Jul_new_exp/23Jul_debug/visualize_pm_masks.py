#!/usr/bin/env python3
"""Overlay the canonical PhotoMaker-derived validation boxes for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (
    HERE
    / "experiments"
    / "20260723T192610Z__23Jul_E02_up1_detail_id00081_s0_600__20260723T192610Z"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve_profile(run_dir: Path) -> dict:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    profile = manifest.get("dataset_profile", "cosmic_large_id00081")
    if profile == "cosmic_large_id00081":
        return {
            "name": profile,
            "mask": (
                HERE
                / "data"
                / "id_00081_1017318003459"
                / "pm_generated_bboxes_holdout_A_seed0.json"
            ),
            "reference": "validation_refs/holdout_A.jpg",
            "title": "holdout_A, seed 0",
        }
    if profile in {
        "one_id_nm0005092_subset8",
        "one_id_nm0005092_subset8_distinct",
        "one_id_nm0005092_full18_heldout_distinct",
    }:
        return {
            "name": profile,
            "mask": (
                HERE
                / "data"
                / "one_id_nm0005092"
                / "pm_generated_bboxes_ref51_seed0.json"
            ),
            "reference": "/home/niko/rsrch/dataset_full/one_id/ref/51.jpg",
            "title": "nm0005092 ref 51, seed 0",
        }
    raise ValueError(f"Unsupported dataset profile: {profile}")


def find_pm_images(run_dir: Path) -> list[Path]:
    root = run_dir / "validation" / "pmControl50" / "step_0000" / "outputs"
    paths = sorted(
        path
        for path in root.glob("*/val_images/manual_val/step_*_batch_*/*.png")
        if not path.stem.endswith("_mask")
    )
    if len(paths) != 4:
        raise RuntimeError(f"Expected four PhotoMaker images under {root}, found {len(paths)}")
    return paths


def font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(image: Image.Image, record: dict, title: str) -> Image.Image:
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    old_box = tuple(record["face_crop_old"])
    new_box = tuple(record["face_crop_new"])
    draw.rectangle(new_box, fill=(52, 211, 153, 42))
    draw.rectangle(old_box, outline=(255, 70, 70, 255), width=6)
    draw.rectangle(new_box, outline=(52, 255, 153, 255), width=6)
    result = Image.alpha_composite(result, overlay)

    banner = Image.new("RGBA", (result.width, 88), (12, 18, 28, 235))
    result.alpha_composite(banner, (0, 0))
    draw = ImageDraw.Draw(result)
    title_font = font(25)
    small_font = font(18)
    draw.text((18, 10), title, fill="white", font=title_font)
    draw.text(
        (18, 49),
        f"RED padded detector {list(old_box)}",
        fill=(255, 105, 105),
        font=small_font,
    )
    draw.text(
        (495, 49),
        f"GREEN BA mask/raw {list(new_box)}",
        fill=(80, 255, 170),
        font=small_font,
    )
    return result.convert("RGB")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    profile = resolve_profile(run_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            HERE
            / "visual_reports"
            / f"pm_mask_debug_{profile['name']}_seed0"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(profile["mask"].read_text(encoding="utf-8"))
    images = find_pm_images(run_dir)
    annotated = []
    manifest_images = []

    for index, path in enumerate(images):
        key = next(
            key
            for key, value in records.items()
            if int(value["_meta"]["debug_idx"]) == index
        )
        record = records[key]
        source = Image.open(path).convert("RGB")
        title = f"p{index}: {record['_meta']['prompt']}"
        result = annotate(source, record, title)
        output = output_dir / f"p{index:02d}_{key.removesuffix('.png')}_PM_bbox_overlay.png"
        result.save(output)
        annotated.append(result)
        manifest_images.append(
            {
                "prompt_index": index,
                "source_pm_image": str(path),
                "output_overlay": str(output),
                "bbox_key": key,
                "face_crop_old": record["face_crop_old"],
                "face_crop_new_used_by_ba": record["face_crop_new"],
            }
        )

    thumb_size = 720
    margin = 18
    header = 86
    sheet = Image.new(
        "RGB",
        (thumb_size * 2 + margin * 3, thumb_size * 2 + margin * 3 + header),
        (20, 25, 34),
    )
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text(
        (margin, 16),
        f"PhotoMaker-derived generated-face masks — {profile['title']}",
        fill="white",
        font=font(28),
    )
    sheet_draw.text(
        (margin, 51),
        "Red = padded detector context; green = raw face_crop_new used by branched attention",
        fill=(210, 220, 235),
        font=font(18),
    )
    for index, image in enumerate(annotated):
        thumb = image.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = margin + (index % 2) * (thumb_size + margin)
        y = header + margin + (index // 2) * (thumb_size + margin)
        sheet.paste(thumb, (x, y))

    sheet_png = output_dir / "pm_generated_mask_contact_sheet.png"
    sheet_pdf = output_dir / "pm_generated_mask_contact_sheet.pdf"
    sheet.save(sheet_png)
    sheet.save(sheet_pdf, "PDF", resolution=150.0)
    manifest = {
        "dataset_profile": profile["name"],
        "mask_file": str(profile["mask"]),
        "reference": profile["reference"],
        "seed": 0,
        "run_dir": str(run_dir),
        "contact_sheet_png": str(sheet_png),
        "contact_sheet_pdf": str(sheet_pdf),
        "images": manifest_images,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(sheet_png)
    print(sheet_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
