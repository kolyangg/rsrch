#!/usr/bin/env python3
"""Render A4 contact sheets with the matched Seed-2 automatic face box."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ARM_LABELS = {
    "A": "A — correct PM identity / correct spatial identity",
    "B": "B — correct PM identity / next spatial identity",
    "C": "C — next PM identity / correct spatial identity (partial snapshot)",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def normalized(name: str) -> str:
    return name.replace(" ", "_")


def overlay_box(image: Image.Image, bbox: list[int]) -> Image.Image:
    canvas = image.convert("RGBA")
    tint = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tint)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 38), outline=(255, 22, 22, 255), width=12)
    label_font = font(28, bold=True)
    label = "SEED 2 PM FACE MASK"
    label_box = draw.textbbox((0, 0), label, font=label_font)
    label_w = label_box[2] - label_box[0]
    label_h = label_box[3] - label_box[1]
    label_y = max(0, y1 - label_h - 16)
    draw.rectangle((x1, label_y, min(x2, x1 + label_w + 20), label_y + label_h + 12), fill=(210, 0, 0, 230))
    draw.text((x1 + 10, label_y + 4), label, fill="white", font=label_font)
    return Image.alpha_composite(canvas, tint).convert("RGB")


def render(args: argparse.Namespace) -> None:
    snapshot = args.snapshot.resolve()
    bbox_path = args.bbox_json.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    boxes = json.loads(bbox_path.read_text(encoding="utf-8"))
    by_filename = {normalized(key): (key, value) for key, value in boxes.items()}
    if len(boxes) != 96:
        raise SystemExit(f"Expected 96 boxes, found {len(boxes)}")
    seeds = {int((value.get("_meta") or {}).get("seed", -1)) for value in boxes.values()}
    if seeds != {2}:
        raise SystemExit(f"Expected only seed 2 metadata, found {sorted(seeds)}")

    page_paths: list[str] = []
    arm_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    page_w, page_h = 1800, 2546
    header_h = 146
    columns, rows_per_page = 3, 4
    cell_w = page_w // columns
    cell_h = (page_h - header_h) // rows_per_page
    image_side = 530
    title_font = font(38, bold=True)
    subtitle_font = font(22)
    cell_font = font(18, bold=True)
    coord_font = font(16)

    for arm in ("A", "B", "C"):
        files = sorted((snapshot / arm).rglob("*.png"))
        matched: list[tuple[int, Path, str, dict]] = []
        for path in files:
            if path.name not in by_filename:
                raise SystemExit(f"No automatic box for {path.name}")
            source_key, record = by_filename[path.name]
            debug_idx = int((record.get("_meta") or {}).get("debug_idx", -1))
            matched.append((debug_idx, path, source_key, record))
        matched.sort(key=lambda item: item[0])
        arm_counts[arm] = len(matched)

        for page_index, start in enumerate(range(0, len(matched), 12), start=1):
            page_items = matched[start : start + 12]
            sheet = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((36, 22), ARM_LABELS[arm], fill=(18, 24, 33), font=title_font)
            draw.text(
                (36, 76),
                f"Seed 2 matched PhotoMaker-only mask • page {page_index} • images {start + 1}–{start + len(page_items)} of {len(matched)}",
                fill=(74, 85, 104),
                font=subtitle_font,
            )
            draw.line((36, 128, page_w - 36, 128), fill=(206, 212, 222), width=2)

            for slot, (debug_idx, path, source_key, record) in enumerate(page_items):
                row, col = divmod(slot, columns)
                x0 = col * cell_w
                y0 = header_h + row * cell_h
                bbox = record.get("face_crop_new") or record.get("face_crop_old")
                if not bbox:
                    raise SystemExit(f"Missing bbox for {source_key}")
                with Image.open(path) as image:
                    marked = overlay_box(image, bbox)
                resampling = getattr(Image, "Resampling", Image)
                marked.thumbnail((image_side, image_side), resampling.LANCZOS)
                image_x = x0 + (cell_w - marked.width) // 2
                image_y = y0 + 8
                sheet.paste(marked, (image_x, image_y))
                draw.rectangle((image_x, image_y, image_x + marked.width - 1, image_y + marked.height - 1), outline=(95, 105, 120), width=1)
                meta = record.get("_meta") or {}
                identity = str(meta.get("id", "?"))
                label = f"#{debug_idx:02d}  {path.stem}"
                draw.text((x0 + 34, y0 + 546), label[:53], fill=(20, 26, 36), font=cell_font)
                draw.text((x0 + 34, y0 + 572), f"ID: {identity}   box: {list(map(int, bbox))}", fill=(78, 87, 103), font=coord_font)
                rows.append({
                    "arm": arm,
                    "debug_idx": debug_idx,
                    "filename": path.name,
                    "source_key": source_key,
                    "identity": identity,
                    "bbox": list(map(int, bbox)),
                    "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })

            page_path = output / f"seed2_{arm}_mask_overlay_page_{page_index:02d}.png"
            sheet.save(page_path, optimize=True)
            page_paths.append(page_path.name)

    manifest = {
        "schema_version": 1,
        "validation_seed": 2,
        "bbox_json": str(bbox_path),
        "bbox_sha256": hashlib.sha256(bbox_path.read_bytes()).hexdigest(),
        "bbox_count": len(boxes),
        "arm_counts": arm_counts,
        "total_images": sum(arm_counts.values()),
        "page_count": len(page_paths),
        "pages": page_paths,
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("bbox_sha256", "arm_counts", "total_images", "page_count")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
