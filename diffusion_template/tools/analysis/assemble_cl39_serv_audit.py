#!/usr/bin/env python3
"""Assemble selected Serv trainer outputs for the CL39 report renderer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--ba-off-dir", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional remote references directory when manifest paths are local",
    )
    return parser.parse_args()


def _output_dir(output_root: Path, record: dict) -> Path:
    action = record["prompt"].split()[0].lower().replace("/", "-")
    return output_root / "outputs" / (
        f"{int(record['index']):02d}_{record['identity']}_{action}"
    )


def _trainer_image(root: Path, record: dict) -> Path:
    index = int(record["index"])
    batch_dir = root / f"step_24000_batch_{index // 12}"
    filename = str(record["face_bbox_gen_key"]).replace(" ", "_")
    path = batch_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["samples"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    assembled = []
    arm_roots = {
        "actual": args.actual_dir,
        "c1": args.c1_dir,
        "ba_off": args.ba_off_dir,
    }
    telemetry_out = args.output_root / "telemetry"
    telemetry_out.mkdir(parents=True, exist_ok=True)

    for record in records:
        index = int(record["index"])
        out_dir = _output_dir(args.output_root, record)
        out_dir.mkdir(parents=True, exist_ok=True)
        for arm, root in arm_roots.items():
            source = _trainer_image(root, record)
            shutil.copy2(source, out_dir / f"{arm}.png")
        reference = Path(record["reference_path"])
        if not reference.is_file() and args.reference_root is not None:
            reference = args.reference_root / reference.name
        if not reference.is_file():
            raise FileNotFoundError(reference)
        shutil.copy2(reference, out_dir / "reference.png")
        record["reference_path"] = str(reference.resolve())
        (out_dir / "sample.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

        npz_source = args.telemetry_dir / f"{index:02d}.npz"
        csv_source = args.telemetry_dir / f"{index:02d}.csv"
        if not npz_source.is_file() or not csv_source.is_file():
            raise FileNotFoundError(
                f"Missing telemetry for index {index}: {npz_source}, {csv_source}"
            )
        shutil.copy2(npz_source, telemetry_out / npz_source.name)
        shutil.copy2(csv_source, telemetry_out / f"{index:02d}_layers.csv")
        assembled.append(index)

    manifest["samples"] = records
    (args.output_root / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "source": "Serv trainer/YAML validation-only audit",
        "assembled_indices": assembled,
        "sample_count": len(assembled),
        "arms": list(arm_roots),
    }
    (args.output_root / "generation_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
