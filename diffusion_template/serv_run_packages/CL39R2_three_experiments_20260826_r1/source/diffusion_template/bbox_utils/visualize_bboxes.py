"""
Visualize bounding boxes for a folder of images.

The script overlays bounding boxes stored in a JSON file on top of each image
and writes the annotated image into the output directory, preserving any
sub-folder structure.

Example:
    python bbox_utils/visualize_bboxes.py \
        --input-dir ../dataset_full/val_dataset/references \
        --bbox-json ../dataset_full/val_dataset/ref_bboxes.json \
        --output-dir ../dataset_full/val_dataset/ref_bbox_debug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/references"),
        help="Folder containing the source images.",
    )
    parser.add_argument(
        "--bbox-json",
        type=Path,
        default=Path("../dataset_full/val_dataset/ref_bboxes.json"),
        help="JSON file with bounding boxes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../dataset_full/val_dataset/ref_bboxes_check"),
        help="Destination folder for annotated images.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=4,
        help="Line width for the drawn boxes.",
    )
    return parser.parse_args()


def load_bboxes(json_path: Path) -> Dict[str, Dict[str, Iterable[int]]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Bbox JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON data to be a dictionary.")
    return data


def draw_boxes(
    image_path: Path,
    boxes: Dict[str, Iterable[int]],
    output_path: Path,
    line_width: int,
) -> None:
    with Image.open(image_path).convert("RGB") as img:
        img = annotate_pil(img, boxes, line_width=line_width)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


def annotate_pil(img: Image.Image, boxes: Dict[str, Iterable[int]], line_width: int = 4) -> Image.Image:
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - PIL always ships a default font
        font = None

    color_cycle = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 128, 255),
        (255, 165, 0),
        (255, 255, 0),
    ]

    for idx, (label, coords) in enumerate(boxes.items()):
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        color = color_cycle[idx % len(color_cycle)]
        x1, y1, x2, y2 = [int(round(c)) for c in coords]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if font is not None:
            text = str(label)
            try:
                text_w, text_h = draw.textsize(text, font=font)
            except Exception:  # pragma: no cover - fallback in case Pillow removes textsize
                bbox = font.getbbox(text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            text_pos = (x1 + 4, max(0, y1 - text_h - 4))
            bg_rect = [
                text_pos[0] - 2,
                text_pos[1] - 2,
                text_pos[0] + text_w + 2,
                text_pos[1] + text_h + 2,
            ]
            draw.rectangle(bg_rect, fill=color)
            draw.text(text_pos, text, fill=(0, 0, 0), font=font)

    return img


def save_annotated_pil(img: Image.Image, boxes: Dict[str, Iterable[int]], output_path: Path, line_width: int = 4) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_pil(img, boxes, line_width=line_width).save(output_path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    bbox_data = load_bboxes(args.bbox_json.resolve())

    for rel_path_str, boxes in bbox_data.items():
        rel_path = Path(rel_path_str)
        candidate_paths = [
            input_dir / rel_path,
            Path(rel_path_str),
        ]
        image_path = next((p for p in candidate_paths if p.exists()), None)
        if image_path is None:
            print(f"[WARN] Skipping missing image: {rel_path_str}")
            continue
        target_path = output_dir / rel_path
        draw_boxes(image_path, boxes, target_path, args.line_width)
        print(f"Saved annotated image to {target_path}")


if __name__ == "__main__":
    main()
