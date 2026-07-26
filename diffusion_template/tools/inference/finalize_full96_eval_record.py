#!/usr/bin/env python3
"""Append verified full-96 result provenance to its canonical Comet record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from full96_protocol import (
    load_bbox_protocol,
    load_object,
    sha256,
    validate_static_inputs,
)

EXPECTED_METRICS = ("manual_val/text_sim", "manual_val/id_sim")


def write_atomic(path: Path, value: dict) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".full96-result-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def png_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise ValueError(f"Image directory is missing: {path}")
    images = sorted(item for item in path.iterdir() if item.suffix == ".png")
    if len(images) != 12:
        raise ValueError(f"Expected 12 PNGs in {path}, found {len(images)}")
    return {item.name: sha256(item) for item in images}


def pixel_fingerprint(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(f"RGB:{width}x{height}\0".encode("ascii"))
        digest.update(rgb.tobytes())
    return {
        "width": width,
        "height": height,
        "rgb_sha256": digest.hexdigest(),
    }


def manifest_sha256(manifest: dict[str, object]) -> str:
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_output_names(manual_keys: set[str]) -> set[str]:
    names = {
        f"{key[:-4].replace(' ', '_')[:80]}.png"
        for key in manual_keys
        if key.endswith(".png")
    }
    if len(names) != len(manual_keys):
        raise ValueError("Sealed bbox keys collide under trainer filename sanitization")
    return names


def local_pixel_manifest(
    batch_dirs: list[Path],
    *,
    expected_names: set[str],
) -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    for directory in batch_dirs:
        for path in sorted(directory.glob("*.png")):
            if path.name in manifest:
                raise ValueError(f"Duplicate local validation filename: {path.name}")
            manifest[path.name] = pixel_fingerprint(path)
    if set(manifest) != expected_names:
        raise ValueError(
            "Local validation filenames do not match the sealed protocol: "
            f"missing={sorted(expected_names - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - expected_names)}"
        )
    return manifest


def verify_comet_export(
    export_path: Path,
    *,
    experiment_key: str,
    run_name: str,
    expected_pixel_manifest: dict[str, dict],
    optimizer_step: int = 4000,
) -> dict:
    export_path = export_path.resolve()
    export = load_object(export_path)
    runs = export.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError("Comet export must contain exactly one run")
    run = runs[0]
    if str(run.get("id", "")) != experiment_key:
        raise ValueError("Comet export experiment key does not match the record")
    if str(run.get("name", "")) != run_name:
        raise ValueError("Comet export run name does not match the record")
    selection = run.get("step_selection")
    if (
        not isinstance(selection, dict)
        or int(run.get("resolved_step_number", -1)) != optimizer_step
        or int(selection.get("requested_step_number", -1)) != optimizer_step
        or int(selection.get("resolved_step_number", -1)) != optimizer_step
        or bool(selection.get("fallback_used"))
        or not bool(selection.get("exact_match_found"))
    ):
        raise ValueError(
            "Comet export did not resolve the exact requested step "
            f"{optimizer_step}"
        )
    if run.get("warnings") not in (None, []):
        raise ValueError(f"Comet export contains warnings: {run.get('warnings')}")
    if run.get("errors") not in (None, []):
        raise ValueError(f"Comet export contains errors: {run.get('errors')}")

    downloaded = run.get("downloaded_images")
    if not isinstance(downloaded, list) or len(downloaded) != 96:
        raise ValueError(
            "Comet export does not contain exactly 96 downloaded images"
        )
    downloaded_pixel_manifest: dict[str, dict] = {}
    export_root = export_path.parent
    for image in downloaded:
        if (
            not isinstance(image, dict)
            or int(image.get("step", -1)) != optimizer_step
        ):
            raise ValueError(
                "Comet export contains an image outside step "
                f"{optimizer_step}"
            )
        name = str(image.get("file_name", ""))
        if not name.endswith(".png") or name in downloaded_pixel_manifest:
            raise ValueError(f"Comet export contains an invalid image name: {name!r}")
        saved_path = export_root / str(image.get("saved_path", ""))
        resolved = saved_path.resolve()
        if export_root != resolved and export_root not in resolved.parents:
            raise ValueError("Comet export image path escapes its export directory")
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError(f"Downloaded Comet image is missing: {resolved}")
        downloaded_pixel_manifest[name] = pixel_fingerprint(resolved)
    if downloaded_pixel_manifest != expected_pixel_manifest:
        raise ValueError(
            "Downloaded Comet images do not pixel-match the local validation outputs"
        )

    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Comet export contains no metric histories")
    metric_values: dict[str, float] = {}
    for name in EXPECTED_METRICS:
        history = metrics.get(name)
        if not isinstance(history, list):
            raise ValueError(f"Comet export is missing metric history {name}")
        points = [
            point
            for point in history
            if (
                isinstance(point, dict)
                and int(point.get("step", -1)) == optimizer_step
            )
        ]
        if len(points) != 1:
            raise ValueError(
                f"Comet metric {name} has {len(points)} values at step "
                f"{optimizer_step}"
            )
        value = float(points[0].get("value"))
        if not math.isfinite(value):
            raise ValueError(f"Comet metric {name} is not finite at step 4000")
        metric_values[name] = value

    return {
        "verified": True,
        "experiment_key": experiment_key,
        "resolved_step": optimizer_step,
        "downloaded_images": len(downloaded),
        "pixel_manifest_sha256": manifest_sha256(
            downloaded_pixel_manifest
        ),
        "metric_values": metric_values,
        "export_sha256": sha256(export_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bbox-manual", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--trainer-source-images", type=Path)
    parser.add_argument("--trainer-reproduction-images", type=Path)
    parser.add_argument(
        "--first-batch-source-kind",
        choices=("trainer_endpoint", "canonical_protocol_preflight"),
        default="trainer_endpoint",
    )
    parser.add_argument(
        "--intervention-label",
        help=(
            "Allow the first batch to differ from the source and record the "
            "named fixed-checkpoint intervention. The intervention must change "
            "at least one first-batch image."
        ),
    )
    parser.add_argument("--comet-export", type=Path, required=True)
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="Defaults to the val_dataset directory containing protocols/",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    record_path = args.record.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manual_path = args.bbox_manual.resolve()
    images_root = args.images_root.resolve()
    trainer_pair = (
        args.trainer_source_images is not None,
        args.trainer_reproduction_images is not None,
    )
    if trainer_pair[0] != trainer_pair[1]:
        raise ValueError(
            "--trainer-source-images and --trainer-reproduction-images "
            "must be supplied together"
        )
    if (
        args.first_batch_source_kind == "canonical_protocol_preflight"
        and not all(trainer_pair)
    ):
        raise ValueError(
            "Canonical-protocol preflight mode requires the trainer "
            "source-reproduction pair"
        )

    record = load_object(record_path)
    experiment_key = str((record.get("comet") or {}).get("experiment_key", ""))
    if len(experiment_key) != 32:
        raise ValueError("Canonical record has no immutable Comet experiment key")

    manual, _, auto_path, routing = load_bbox_protocol(
        manual_path,
        auto_min=95,
        require_complete=True,
    )
    validation_data_dir = (
        args.validation_data_dir.resolve()
        if args.validation_data_dir is not None
        else manual_path.parent.parent.parent
    )
    static_inputs = validate_static_inputs(validation_data_dir)
    batch_dirs = sorted(images_root.glob("step_4000_batch_*"))
    image_count = sum(
        1
        for directory in batch_dirs
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".png"
    )
    if len(batch_dirs) != 8 or image_count != 96:
        raise ValueError(
            f"Expected eight batches and 96 PNGs, found {len(batch_dirs)} and "
            f"{image_count}"
        )
    local_pixels = local_pixel_manifest(
        batch_dirs,
        expected_names=expected_output_names(set(manual)),
    )
    source_hashes = png_hashes(args.source_images.resolve())
    first_batch_hashes = png_hashes(images_root / "step_4000_batch_0")
    first_batch_reproduced_source = source_hashes == first_batch_hashes
    if args.intervention_label:
        if first_batch_reproduced_source:
            raise ValueError(
                "The fixed-checkpoint intervention did not change the first batch"
            )
    elif not first_batch_reproduced_source:
        raise ValueError("First full-96 batch does not reproduce the source panel")

    trainer_source_reproduction = None
    if all(trainer_pair):
        trainer_source_hashes = png_hashes(args.trainer_source_images.resolve())
        trainer_reproduction_hashes = png_hashes(
            args.trainer_reproduction_images.resolve()
        )
        if trainer_source_hashes != trainer_reproduction_hashes:
            raise ValueError(
                "Source-protocol preflight does not reproduce the trainer endpoint"
            )
        trainer_source_reproduction = {
            "verified": True,
            "trainer_source_images": str(args.trainer_source_images.resolve()),
            "reproduction_images": str(
                args.trainer_reproduction_images.resolve()
            ),
            "trainer_source_png_sha256": trainer_source_hashes,
            "reproduction_png_sha256": trainer_reproduction_hashes,
            "manifest_sha256": manifest_sha256(trainer_source_hashes),
        }

    comet_verification = verify_comet_export(
        args.comet_export,
        experiment_key=experiment_key,
        run_name=str(record.get("run_name", "")),
        expected_pixel_manifest=local_pixels,
    )
    if args.verify_only:
        print(
            "FULL96_RESULT_VERIFIED "
            f"record={record_path} comet_key={experiment_key} images=96"
        )
        return 0

    record["validation_result"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_step": 4000,
        "batch_count": 8,
        "image_count": 96,
        "pixel_manifest_sha256": manifest_sha256(local_pixels),
        "first_batch_reproduced_source": first_batch_reproduced_source,
        "first_batch_source_kind": (
            "fixed_checkpoint_intervention"
            if args.intervention_label
            else args.first_batch_source_kind
        ),
        "intervention_label": args.intervention_label,
        "source_first_batch_sha256": source_hashes,
        "first_batch_sha256": first_batch_hashes,
        "checkpoint_sha256": sha256(checkpoint_path),
        "manual_bbox_sha256": sha256(manual_path),
        "automatic_bbox_sha256": sha256(auto_path),
        "automatic_bbox_entries": routing["automatic_entries"],
        "force_manual_entries": routing["force_manual_entries"],
        "force_manual_keys": routing["force_manual_keys"],
        "routing_entries": routing["routing_entries"],
        "static_inputs": static_inputs,
        "comet_verification": comet_verification,
    }
    if trainer_source_reproduction is not None:
        record["validation_result"][
            "trainer_source_reproduction"
        ] = trainer_source_reproduction
    write_atomic(record_path, record)
    print(
        "FULL96_RECORD_FINALIZED "
        f"record={record_path} comet_key={experiment_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
