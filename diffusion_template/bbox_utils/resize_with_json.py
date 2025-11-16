#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Dict, Any, Tuple

from PIL import Image


def infer_person_id(image_dir: str, data: Dict[str, Any], explicit_id: str | None) -> str:
    if explicit_id is not None:
        if explicit_id not in data:
            raise KeyError(f"ID '{explicit_id}' not found in JSON (available keys include: "
                           f"{', '.join(list(data.keys())[:10])} ...)")
        return explicit_id

    dirname = os.path.basename(os.path.normpath(image_dir))
    candidates = [dirname]
    if "_" in dirname:
        candidates.append(dirname.split("_", 1)[0])

    for cand in candidates:
        if cand in data:
            return cand

    raise KeyError(
        f"Could not infer person ID from directory '{dirname}'. "
        f"Explicitly pass --id (available keys include: {', '.join(list(data.keys())[:10])} ...)."
    )


def compute_scale_and_offset(
    width: int, height: int, target: int
) -> Tuple[float, int, int, int, int]:
    scale = max(target / float(width), target / float(height))
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))

    left_f = (new_w - target) / 2.0
    top_f = (new_h - target) / 2.0
    left = int(round(left_f))
    top = int(round(top_f))

    right = left + target
    bottom = top + target

    if right > new_w:
        right = new_w
        left = right - target
    if bottom > new_h:
        bottom = new_h
        top = bottom - target

    return scale, left, top, new_w, new_h


def transform_box(
    box: list[int], scale: float, offset_x: int, offset_y: int, target: int
) -> list[int]:
    # Expect boxes in standard [x1, y1, x2, y2] format
    x1, y1, x2, y2 = box

    x1 = x1 * scale - offset_x
    x2 = x2 * scale - offset_x
    y1 = y1 * scale - offset_y
    y2 = y2 * scale - offset_y

    def clamp(v: float) -> int:
        v = max(0.0, min(float(target), v))
        return int(round(v))

    x1_c = clamp(x1)
    x2_c = clamp(x2)
    y1_c = clamp(y1)
    y2_c = clamp(y2)

    if x1_c > x2_c:
        x1_c, x2_c = x2_c, x1_c
    if y1_c > y2_c:
        y1_c, y2_c = y2_c, y1_c

    return [x1_c, y1_c, x2_c, y2_c]


def process(
    image_dir: str,
    json_path: str,
    output_dir: str,
    person_id: str | None,
    target_size: int,
    json_output_path: str | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    pid = infer_person_id(image_dir, data, person_id)
    entries: Dict[str, Any] = data[pid]

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files: Dict[str, str] = {}
    for fname in os.listdir(image_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in exts:
            files[base] = fname

    new_entries: Dict[str, Any] = {}
    processed = 0

    for base, fname in files.items():
        if base not in entries:
            # Image not present in JSON: skip it.
            continue

        src_path = os.path.join(image_dir, fname)
        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        scale, off_x, off_y, new_w, new_h = compute_scale_and_offset(w, h, target_size)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        cropped = resized.crop((off_x, off_y, off_x + target_size, off_y + target_size))

        dst_path = os.path.join(output_dir, fname)
        cropped.save(dst_path, quality=95)

        entry = dict(entries[base])
        entry["orig_image_size"] = [target_size, target_size]

        # Normalize face crop naming to a single key: 'face_crop'
        face_box = None
        for face_key in (
            "face_crop",
            "face_crop_new",
            "face_crop_old",
            "new_face_crop",
            "orig_face_crop",
        ):
            box = entry.get(face_key)
            if isinstance(box, (list, tuple)) and len(box) == 4:
                face_box = box
                break

        if face_box is not None:
            entry["face_crop"] = transform_box(
                face_box, scale, off_x, off_y, target_size
            )

        # Body crop (if present) keeps its name.
        if "body_crop" in entry and entry["body_crop"] is not None:
            entry["body_crop"] = transform_box(
                entry["body_crop"], scale, off_x, off_y, target_size
            )

        # Remove any alternative face crop keys from the output JSON.
        for k in ("orig_face_crop", "new_face_crop", "face_crop_new", "face_crop_old"):
            entry.pop(k, None)

        json_key = fname
        new_entries[json_key] = entry
        processed += 1

    if not new_entries:
        raise RuntimeError("No images from the folder were found in the JSON entries.")

    new_data = new_entries

    if json_output_path is not None:
        out_json_path = json_output_path
        json_dir = os.path.dirname(out_json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
    else:
        out_json_path = os.path.join(output_dir, os.path.basename(json_path))

    with open(out_json_path, "w") as f:
        json.dump(new_data, f)

    print(f"Processed {processed} images for ID '{pid}'.")
    print(f"Resized images saved to: {output_dir}")
    print(f"Adjusted JSON saved to: {out_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize images to a fixed square size and adjust JSON annotation coordinates."
    )
    parser.add_argument("image_dir", help="Input image folder.")
    parser.add_argument("json_path", help="JSON file with area annotations.")
    parser.add_argument("output_dir", help="Destination folder for resized images and new JSON.")
    parser.add_argument(
        "--out-json",
        dest="json_output",
        default=None,
        help="Full path for the output JSON file. "
             "Defaults to <output_dir>/<basename(json_path)>.",
    )
    parser.add_argument(
        "--id",
        dest="person_id",
        default=None,
        help="Person ID key inside the JSON. "
             "If omitted, inferred from image folder name.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Target square size (default: 1024).",
    )

    args = parser.parse_args()

    process(
        image_dir=args.image_dir,
        json_path=args.json_path,
        output_dir=args.output_dir,
        person_id=args.person_id,
        target_size=args.size,
        json_output_path=args.json_output,
    )


if __name__ == "__main__":
    main()
