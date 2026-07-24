#!/usr/bin/env python3
"""Reuse deterministic PM/NN3a step-zero controls without another GPU render."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from launch_validation import resolve_comet_key, upload_images_to_comet


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE
    / "experiments"
    / "20260723T192610Z__23Jul_E02_up1_detail_id00081_s0_600__20260723T192610Z"
)
CONTROL_SPECS = (("pmControl50", 0), ("canonical50", 0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_run_dirs", type=Path, nargs="+")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def images_for(run_dir: Path, mode: str, step: int) -> list[Path]:
    root = run_dir / "validation" / mode / f"step_{step:04d}" / "outputs"
    return sorted(
        path
        for path in root.glob("*/val_images/manual_val/step_*_batch_*/*.png")
        if not path.stem.endswith("_mask")
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    args = parse_args()
    source = args.source_run_dir.resolve()
    source_manifest = json.loads(
        (source / "run_manifest.json").read_text(encoding="utf-8")
    )
    source_profile = source_manifest.get(
        "dataset_profile", "cosmic_large_id00081"
    )
    records = []
    for target_arg in args.target_run_dirs:
        target = target_arg.resolve()
        target_manifest_path = target / "run_manifest.json"
        target_manifest = json.loads(
            target_manifest_path.read_text(encoding="utf-8")
        )
        target_profile = target_manifest.get(
            "dataset_profile", "cosmic_large_id00081"
        )
        if target_profile != source_profile:
            raise RuntimeError(
                "Shared controls require the same dataset/identity profile: "
                f"{source_profile} != {target_profile}"
            )
        if target_manifest.get("status") != "completed":
            raise RuntimeError(f"Target training run is not complete: {target}")
        comet_key = target_manifest.get("comet_experiment_key") or resolve_comet_key(
            target_manifest["run_name"]
        )
        target_manifest["comet_experiment_key"] = comet_key
        target_manifest_path.write_text(
            json.dumps(target_manifest, indent=2) + "\n", encoding="utf-8"
        )

        for mode, step in CONTROL_SPECS:
            source_images = images_for(source, mode, step)
            if len(source_images) != 4:
                raise RuntimeError(
                    f"Expected four shared source images for {mode} step {step}"
                )
            existing = images_for(target, mode, step)
            if existing:
                raise RuntimeError(
                    f"Refusing to overwrite existing {mode} step {step} images in {target}"
                )
            destination_root = (
                target
                / "validation"
                / mode
                / f"step_{step:04d}"
                / "outputs"
                / f"{mode}__step{step:04d}"
                / "val_images"
                / "manual_val"
            )
            copied = []
            for prompt_index, source_path in enumerate(source_images):
                batch_dir = destination_root / f"step_{step}_batch_{prompt_index}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                destination = batch_dir / source_path.name
                shutil.copy2(source_path, destination)
                if digest(destination) != digest(source_path):
                    raise RuntimeError(f"Checksum mismatch copying {source_path}")
                copied.append(destination)
            uploaded = upload_images_to_comet(
                comet_key,
                target_manifest["run_name"],
                mode,
                step,
                copied,
            )
            provenance = {
                "kind": "deterministic_shared_control",
                "mode": mode,
                "step": step,
                "source_run": str(source),
                "source_run_name": source_manifest["run_name"],
                "dataset_profile": source_profile,
                "reason": (
                    "These arms share the exact dataset profile, reference, "
                    "prompts, seed, and NN3a_new1 inference graph; optimizer/loss "
                    "toggles have zero effect before training."
                ),
                "files": [
                    {
                        "source": str(source_path),
                        "destination": str(destination),
                        "sha256": digest(destination),
                    }
                    for source_path, destination in zip(source_images, copied)
                ],
                "comet_uploaded_names": uploaded,
            }
            provenance_path = (
                target
                / "validation"
                / mode
                / f"step_{step:04d}"
                / "shared_control_provenance.json"
            )
            provenance_path.write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            records.append(provenance)
    print(json.dumps({"materialized": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
