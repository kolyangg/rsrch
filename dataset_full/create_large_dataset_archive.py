#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from multiprocessing import cpu_count, get_context
from pathlib import Path, PurePosixPath


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an archive from image paths listed in a filtered large-dataset JSON. "
            "Paths inside the archive preserve the JSON subfolder layout."
        )
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the JSON file produced from large_dataset_analysis.ipynb.",
    )
    parser.add_argument(
        "archive_file",
        type=Path,
        help="Output archive path. Supported suffixes: .zip, .tar, .tar.gz, .tgz",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Root directory that JSON image paths are relative to. Default: dataset_full/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count(), 16)),
        help="Parallel workers used for path validation. Default: min(cpu_count(), 16)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image path from the JSON is missing.",
    )
    return parser.parse_args()


def load_json_paths(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("JSON must contain a top-level 'classes' object.")

    rel_paths: list[str] = []
    for class_name, class_info in classes.items():
        if not isinstance(class_info, dict):
            raise ValueError(f"Class entry for '{class_name}' must be a JSON object.")
        images = class_info.get("images", [])
        if not isinstance(images, list):
            raise ValueError(f"'images' for class '{class_name}' must be a list.")
        for image_info in images:
            if not isinstance(image_info, dict) or "image_path" not in image_info:
                raise ValueError(f"Image entry in class '{class_name}' is missing 'image_path'.")
            rel_paths.append(str(PurePosixPath(image_info["image_path"])))

    # Preserve order while removing duplicates.
    unique_rel_paths = list(dict.fromkeys(rel_paths))
    return unique_rel_paths


def _validate_one(task: tuple[str, str]) -> tuple[str, bool, int]:
    dataset_root_str, rel_path_str = task
    abs_path = Path(dataset_root_str) / Path(rel_path_str)
    try:
        stat = abs_path.stat()
    except FileNotFoundError:
        return rel_path_str, False, 0
    return rel_path_str, abs_path.is_file(), stat.st_size


def validate_paths_parallel(dataset_root: Path, rel_paths: list[str], workers: int):
    tasks = [(str(dataset_root), rel_path) for rel_path in rel_paths]

    if workers <= 1 or len(tasks) < 2:
        results = [_validate_one(task) for task in tasks]
    else:
        try:
            ctx = get_context("fork")
        except ValueError:
            ctx = get_context()
        chunksize = max(1, len(tasks) // (workers * 4))
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_validate_one, tasks, chunksize=chunksize)

    existing = []
    missing = []
    total_bytes = 0
    for rel_path, ok, size in results:
        if ok:
            existing.append((rel_path, size))
            total_bytes += size
        else:
            missing.append(rel_path)

    return existing, missing, total_bytes


def ensure_supported_archive_path(path: Path):
    suffixes = tuple(path.suffixes)
    if suffixes[-1:] == (".zip",):
        return "zip"
    if suffixes[-1:] == (".tar",):
        return "tar"
    if suffixes[-2:] == (".tar", ".gz") or suffixes[-1:] == (".tgz",):
        return "tar.gz"
    raise ValueError("Unsupported archive extension. Use .zip, .tar, .tar.gz, or .tgz")


def write_zip_archive(archive_path: Path, dataset_root: Path, rel_paths: list[str]):
    compression = zipfile.ZIP_STORED
    with zipfile.ZipFile(archive_path, mode="w", compression=compression, allowZip64=True) as zf:
        for idx, rel_path in enumerate(rel_paths, start=1):
            abs_path = dataset_root / rel_path
            zf.write(abs_path, arcname=rel_path)
            if idx % 1000 == 0 or idx == len(rel_paths):
                print(f"[zip] added {idx:,}/{len(rel_paths):,}")


def write_tar_archive(archive_path: Path, dataset_root: Path, rel_paths: list[str], gz: bool):
    mode = "w:gz" if gz else "w"
    label = "tar.gz" if gz else "tar"
    with tarfile.open(archive_path, mode=mode) as tf:
        for idx, rel_path in enumerate(rel_paths, start=1):
            abs_path = dataset_root / rel_path
            tf.add(abs_path, arcname=rel_path, recursive=False)
            if idx % 1000 == 0 or idx == len(rel_paths):
                print(f"[{label}] added {idx:,}/{len(rel_paths):,}")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def main():
    args = parse_args()

    json_path = args.json_file.resolve()
    archive_path = args.archive_file.resolve()
    dataset_root = args.dataset_root.resolve()

    archive_kind = ensure_supported_archive_path(archive_path)
    rel_paths = load_json_paths(json_path)

    print(f"JSON file: {json_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Archive output: {archive_path}")
    print(f"Requested files from JSON: {len(rel_paths):,}")

    existing, missing, total_bytes = validate_paths_parallel(
        dataset_root=dataset_root,
        rel_paths=rel_paths,
        workers=args.workers,
    )

    existing_rel_paths = [rel_path for rel_path, _ in existing]

    print(f"Existing files to archive: {len(existing_rel_paths):,}")
    print(f"Missing files: {len(missing):,}")
    print(f"Estimated input size: {format_bytes(total_bytes)}")

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        print("Missing path preview:")
        print(preview)
        if args.strict:
            raise FileNotFoundError("Missing files detected and --strict was set.")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")

    # Group by subfolder for more sequential disk access during archiving.
    existing_rel_paths.sort(key=lambda p: (Path(p).parent.as_posix(), Path(p).name))

    if archive_kind == "zip":
        write_zip_archive(archive_path, dataset_root, existing_rel_paths)
    elif archive_kind == "tar":
        write_tar_archive(archive_path, dataset_root, existing_rel_paths, gz=False)
    else:
        write_tar_archive(archive_path, dataset_root, existing_rel_paths, gz=True)

    print("Archive creation complete.")
    print(f"Archive saved to: {archive_path}")


if __name__ == "__main__":
    main()
