#!/usr/bin/env python3
"""Gate an Eddie counterfactual on reproduction of historical validation pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_name(replay_name: str) -> str:
    return re.sub(r"^\d{3}_", "", replay_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mean-abs-rgb", type=float, default=0.0)
    parser.add_argument("--max-changed-pixel-fraction", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = args.replay_dir / "run_manifest.json"
    per_image_path = args.replay_dir / "per_image.json"
    replay_images = args.replay_dir / "images"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_image = json.loads(per_image_path.read_text(encoding="utf-8"))

    contract = manifest.get("validation_contract") or {}
    required_contract = {
        "validation_base": "SG161222/RealVisXL_V4.0",
        "processor_base_mode": "legacy_full_copy",
        "batch_size": 12,
        "disable_branched_ca": True,
        "validation_shadow_photomaker_default": True,
        "strict_validation_processor_copy": True,
        "num_inference_steps": 50,
        "pose_adapt_ratio": 0.0,
        "ca_mixing_for_face": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": contract.get(key)}
        for key, expected in required_contract.items()
        if contract.get(key) != expected
    }

    rows = []
    for item in per_image:
        replay_path = replay_images / item["filename"]
        historical_path = args.historical_dir / historical_name(item["filename"])
        if not replay_path.is_file() or not historical_path.is_file():
            raise FileNotFoundError(
                f"Missing replay pair: {replay_path} / {historical_path}"
            )
        replay = np.asarray(Image.open(replay_path).convert("RGB"), dtype=np.int16)
        historical = np.asarray(
            Image.open(historical_path).convert("RGB"), dtype=np.int16
        )
        if replay.shape != historical.shape:
            raise ValueError(
                f"Shape mismatch for {item['filename']}: "
                f"{replay.shape} != {historical.shape}"
            )
        absolute = np.abs(replay - historical)
        rows.append(
            {
                "replay": str(replay_path.resolve()),
                "historical": str(historical_path.resolve()),
                "replay_sha256": sha256_file(replay_path),
                "historical_sha256": sha256_file(historical_path),
                "pixel_exact": bool(np.array_equal(replay, historical)),
                "mean_abs_rgb": float(absolute.mean()),
                "max_abs_rgb": int(absolute.max()),
                "changed_pixel_fraction": float(np.any(absolute != 0, axis=2).mean()),
            }
        )

    failures = [
        row
        for row in rows
        if row["mean_abs_rgb"] > args.max_mean_abs_rgb
        or row["changed_pixel_fraction"] > args.max_changed_pixel_fraction
    ]
    result = {
        "schema_version": 1,
        "kind": "historical_training_validation_reproduction_gate",
        "replay_dir": str(args.replay_dir.resolve()),
        "historical_dir": str(args.historical_dir.resolve()),
        "contract_mismatches": mismatches,
        "thresholds": {
            "max_mean_abs_rgb": args.max_mean_abs_rgb,
            "max_changed_pixel_fraction": args.max_changed_pixel_fraction,
        },
        "pair_count": len(rows),
        "failed_pair_count": len(failures),
        "passed": not mismatches and not failures and len(rows) == 12,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(
            "Historical replay gate failed; corrected Eddie generation is blocked"
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
