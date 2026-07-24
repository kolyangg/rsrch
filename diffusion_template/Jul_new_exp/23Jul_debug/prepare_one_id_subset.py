#!/usr/bin/env python3
"""Prepare the fixed eight-image OneIDTrain ablation subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/home/niko/rsrch/dataset_full/one_id")
SOURCE_JSON = SOURCE_ROOT / "nm0005092_adj_train.json"
SOURCE_IMAGES = SOURCE_ROOT / "nm0005092_adj"
OUTPUT = HERE / "data" / "one_id_nm0005092"
SELECTED = (
    "83.jpg",
    "109.jpg",
    "38.jpg",
    "57.jpg",
    "104.jpg",
    "36.jpg",
    "1.jpg",
    "116.jpg",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> int:
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    missing = [name for name in SELECTED if name not in source]
    if missing:
        raise RuntimeError(f"One-ID source JSON is missing {missing}")
    subset = {name: source[name] for name in SELECTED}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset_path = OUTPUT / "subset8_train.json"
    subset_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")

    cell = 420
    header = 70
    margin = 12
    sheet = Image.new(
        "RGB",
        (cell * 4 + margin * 5, cell * 2 + margin * 3 + header),
        (17, 24, 33),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 14),
        "one_id / nm0005092 — fixed 8-image training subset",
        fill="white",
        font=font(28),
    )
    draw.text(
        (margin, 46),
        "Green = training face_crop; 51.jpg held out as the recurring validation reference",
        fill=(205, 218, 234),
        font=font(16),
    )
    entries = []
    for index, name in enumerate(SELECTED):
        path = SOURCE_IMAGES / name
        image = Image.open(path).convert("RGB")
        bbox = subset[name]["face_crop"]
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(tuple(bbox), fill=(35, 255, 150, 28))
        overlay_draw.rectangle(tuple(bbox), outline=(45, 255, 155, 255), width=6)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        image_draw = ImageDraw.Draw(image)
        image_draw.rectangle((0, 0, 140, 38), fill=(10, 16, 24))
        image_draw.text((10, 7), name, fill="white", font=font(20))
        x = margin + (index % 4) * (cell + margin)
        y = header + margin + (index // 4) * (cell + margin)
        sheet.paste(image, (x, y))
        entries.append(
            {
                "filename": name,
                "image_path": str(path),
                "image_sha256": sha256(path),
                "face_crop": bbox,
                "caption": subset[name]["text"],
            }
        )

    sheet_path = OUTPUT / "subset8_training_images_with_bboxes.png"
    sheet.save(sheet_path)
    manifest = {
        "dataset_class": "src.datasets.cosmic.OneIDTrain",
        "identity": "nm0005092",
        "source_json": str(SOURCE_JSON),
        "source_json_sha256": sha256(SOURCE_JSON),
        "source_images": str(SOURCE_IMAGES),
        "subset_json": str(subset_path),
        "subset_count": len(subset),
        "selection": (
            "Eight high-resolution, moderate-face-scale records; excludes 51.jpg, "
            "which is the native one_id validation reference."
        ),
        "pairing_profiles": {
            "one_id_nm0005092_subset8": {
                "train_on_separate_image": False,
                "status": "invalid_leakage_audit_only",
                "pairing": "target image is copied as its own reference",
            },
            "one_id_nm0005092_subset8_distinct": {
                "train_on_separate_image": True,
                "status": "valid",
                "pairing": (
                    "reference index is sampled from all subset indices except "
                    "the target index"
                ),
            },
        },
        "entries": entries,
        "validation_reference": str(SOURCE_ROOT / "ref" / "51.jpg"),
        "validation_reference_is_training_image": False,
        "contact_sheet": str(sheet_path),
    }
    (OUTPUT / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(subset_path)
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
