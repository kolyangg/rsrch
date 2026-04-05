#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from multiprocessing import cpu_count, get_context
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter dataset_full/filtered_ids3.json to images listed in a selection JSON "
            "and update crop coordinates to match the resized 1024x1024 outputs. "
            "Optionally resize the images and write the adjusted JSON in the same pass."
        )
    )
    parser.add_argument(
        "selection_json",
        type=Path,
        help="JSON file with selected images, e.g. large_dataset_filtered_body1024_face200_min5_top30.json",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path for the updated filtered_ids-style JSON to write.",
    )
    parser.add_argument(
        "--output-images-root",
        type=Path,
        default=None,
        help=(
            "Optional output directory for resized images. If provided, images are "
            "resized to 1024x1024 and saved under this root while preserving "
            "large_dataset/<class>/<image> paths."
        ),
    )
    parser.add_argument(
        "--filtered-json",
        type=Path,
        default=SCRIPT_DIR / "filtered_ids3.json",
        help="Source filtered_ids JSON. Default: dataset_full/filtered_ids3.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Root directory that image paths are relative to. Default: dataset_full/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count(), 16)),
        help="Parallel workers for image-size reads. Default: min(cpu_count(), 16)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected image is missing from filtered_ids JSON or on disk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite resized image files if --output-images-root is set.",
    )
    return parser.parse_args()


def load_selection(selection_json: Path) -> dict[str, dict[str, str]]:
    with selection_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Selection JSON must contain a top-level 'classes' object.")

    selected: dict[str, dict[str, str]] = {}
    for folder_name, class_info in classes.items():
        if not isinstance(class_info, dict):
            raise ValueError(f"Class entry for '{folder_name}' must be an object.")
        images = class_info.get("images", [])
        if not isinstance(images, list):
            raise ValueError(f"'images' for class '{folder_name}' must be a list.")

        folder_map: dict[str, str] = {}
        for image_info in images:
            if not isinstance(image_info, dict):
                raise ValueError(f"Image entry in class '{folder_name}' must be an object.")

            image_path = image_info.get("image_path")
            if image_path is None:
                raise ValueError(f"Image entry in class '{folder_name}' is missing 'image_path'.")

            image_id = str(image_info.get("image_id", Path(str(image_path)).stem))
            folder_map[image_id] = str(PurePosixPath(image_path))

        if folder_map:
            selected[folder_name] = folder_map

    return selected


def load_filtered_json(filtered_json: Path) -> dict:
    with filtered_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_resize_and_crop_params(width: int, height: int, target_size: int = TARGET_SIZE):
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    scale = target_size / width
    resized_width = target_size
    resized_height = max(1, round(height * scale))

    if resized_height < target_size:
        scale = target_size / height
        resized_height = target_size
        resized_width = max(1, round(width * scale))

    left = max(0, (resized_width - target_size) // 2)
    top = max(0, (resized_height - target_size) // 2)

    return {
        "scale": scale,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_left": left,
        "crop_top": top,
        "target_size": target_size,
    }


def clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def transform_face_box(box, params):
    x0, y0, x1, y1 = box
    scale = params["scale"]
    left = params["crop_left"]
    top = params["crop_top"]
    max_coord = params["target_size"]

    new_box = [
        clip_int(int(round(x0 * scale - left)), 0, max_coord),
        clip_int(int(round(y0 * scale - top)), 0, max_coord),
        clip_int(int(round(x1 * scale - left)), 0, max_coord),
        clip_int(int(round(y1 * scale - top)), 0, max_coord),
    ]
    return new_box


def transform_body_box(box, params):
    x0, x1, y0, y1 = box
    scale = params["scale"]
    left = params["crop_left"]
    top = params["crop_top"]
    max_coord = params["target_size"]

    new_box = [
        clip_int(int(round(x0 * scale - left)), 0, max_coord),
        clip_int(int(round(x1 * scale - left)), 0, max_coord),
        clip_int(int(round(y0 * scale - top)), 0, max_coord),
        clip_int(int(round(y1 * scale - top)), 0, max_coord),
    ]
    return new_box


def resize_and_center_crop(image: Image.Image, params) -> Image.Image:
    image = image.resize(
        (params["resized_width"], params["resized_height"]),
        Image.Resampling.LANCZOS,
    )

    left = params["crop_left"]
    top = params["crop_top"]
    right = left + params["target_size"]
    bottom = top + params["target_size"]
    image = image.crop((left, top, right, bottom))

    if image.size != (params["target_size"], params["target_size"]):
        raise RuntimeError(f"Unexpected output size after crop: {image.size}")
    return image


def save_image(image: Image.Image, output_path: Path):
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {".jpg", ".jpeg"}:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(output_path, quality=95, subsampling=0)
    elif suffix == ".png":
        image.save(output_path)
    elif suffix == ".webp":
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(output_path, quality=95, method=6)
    else:
        image.save(output_path)


def _process_one(task: tuple[str, str, str, str, dict, str | None, bool]) -> tuple[bool, str, str, dict | None, str | None]:
    dataset_root_str, folder_name, image_id, rel_path_str, record, output_images_root_str, overwrite = task
    src_path = Path(dataset_root_str) / rel_path_str

    try:
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        with Image.open(src_path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            params = compute_resize_and_crop_params(width, height, target_size=TARGET_SIZE)

            if output_images_root_str is not None:
                dst_path = Path(output_images_root_str) / rel_path_str
                if overwrite or not dst_path.exists():
                    resized = resize_and_center_crop(img, params)
                    save_image(resized, dst_path)

        updated = dict(record)

        if "orig_face_crop" in updated:
            updated["orig_face_crop"] = transform_face_box(updated["orig_face_crop"], params)
        if "new_face_crop" in updated:
            updated["new_face_crop"] = transform_face_box(updated["new_face_crop"], params)
        if "body_crop" in updated:
            body_crop = updated["body_crop"]
            body_width = int(body_crop[1] - body_crop[0])
            body_height = int(body_crop[3] - body_crop[2])
            if width == body_width and height == body_height:
                updated["body_crop"] = [0, TARGET_SIZE, 0, TARGET_SIZE]
            else:
                updated["body_crop"] = transform_body_box(body_crop, params)
        if "orig_image_size" in updated:
            updated["orig_image_size"] = [TARGET_SIZE, TARGET_SIZE]

        return True, folder_name, image_id, updated, None
    except Exception as exc:
        return False, folder_name, image_id, None, repr(exc)


def process_tasks(tasks, workers: int):
    if workers <= 1 or len(tasks) < 2:
        iterator = map(_process_one, tasks)
        return list(tqdm(iterator, total=len(tasks), desc="Updating JSON", unit="img"))

    try:
        ctx = get_context("fork")
    except ValueError:
        ctx = get_context()

    chunksize = max(1, len(tasks) // (workers * 4))
    with ctx.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_process_one, tasks, chunksize=chunksize)
        return list(tqdm(iterator, total=len(tasks), desc="Updating JSON", unit="img"))


def main():
    args = parse_args()

    selection_json = args.selection_json.resolve()
    output_json = args.output_json.resolve()
    filtered_json = args.filtered_json.resolve()
    dataset_root = args.dataset_root.resolve()
    output_images_root = args.output_images_root.resolve() if args.output_images_root else None

    selected = load_selection(selection_json)
    filtered = load_filtered_json(filtered_json)

    tasks = []
    missing_in_filtered = []

    for folder_name, image_map in selected.items():
        folder_records = filtered.get(folder_name)
        if not isinstance(folder_records, dict):
            for image_id in image_map:
                missing_in_filtered.append((folder_name, image_id, "missing_folder"))
            continue

        for image_id, rel_path in image_map.items():
            record = folder_records.get(str(image_id))
            if not isinstance(record, dict):
                missing_in_filtered.append((folder_name, image_id, "missing_image"))
                continue
            tasks.append(
                (
                    str(dataset_root),
                    folder_name,
                    str(image_id),
                    rel_path,
                    record,
                    str(output_images_root) if output_images_root else None,
                    args.overwrite,
                )
            )

    print(f"Selection JSON: {selection_json}")
    print(f"Filtered JSON: {filtered_json}")
    print(f"Dataset root: {dataset_root}")
    if output_images_root is not None:
        output_images_root.mkdir(parents=True, exist_ok=True)
        print(f"Output images root: {output_images_root}")
    print(f"Selected classes: {len(selected):,}")
    print(f"Selected images requested: {sum(len(v) for v in selected.values()):,}")
    print(f"Selected images found in filtered_ids JSON: {len(tasks):,}")
    print(f"Missing in filtered_ids JSON: {len(missing_in_filtered):,}")

    if missing_in_filtered:
        print("Missing metadata preview:")
        for folder_name, image_id, reason in missing_in_filtered[:20]:
            print(f"  - {folder_name}/{image_id}: {reason}")
        if args.strict:
            raise SystemExit(1)

    results = process_tasks(tasks, workers=args.workers)

    success = [item for item in results if item[0]]
    failed = [item for item in results if not item[0]]

    print(f"Successfully updated: {len(success):,}")
    print(f"Failed while transforming: {len(failed):,}")

    if failed:
        print("Failure preview:")
        for _, folder_name, image_id, _, error in failed[:20]:
            print(f"  - {folder_name}/{image_id}: {error}")
        if args.strict:
            raise SystemExit(1)

    output = {}
    for _, folder_name, image_id, updated_record, _ in success:
        output.setdefault(folder_name, {})[image_id] = updated_record

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote updated JSON: {output_json}")
    if output_images_root is not None:
        print(f"Resized images root: {output_images_root}")
    print(f"Output classes: {len(output):,}")
    print(f"Output images: {sum(len(v) for v in output.values()):,}")


if __name__ == "__main__":
    main()
