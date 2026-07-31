#!/usr/bin/env python3
"""Rebase transferred face-quality manifests and verify every image exactly."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backfill_face_quality_metrics import sha256, validate_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_root = args.old_root.as_posix().rstrip("/")
    new_root = args.new_root.resolve()
    seen_paths: set[Path] = set()
    image_count = 0

    for step, assets in payload["steps"].items():
        for asset in assets:
            old_path = str(asset["local_path"])
            prefix = f"{old_root}/"
            if not old_path.startswith(prefix):
                raise ValueError(
                    f"Manifest path does not start with old root: {old_path}"
                )
            relative = Path(old_path[len(prefix) :])
            new_path = (new_root / relative).resolve()
            if new_path in seen_paths:
                raise ValueError(f"Duplicate manifest path: {new_path}")
            seen_paths.add(new_path)
            validate_image(new_path, int(asset["file_size"]))
            actual_sha256 = sha256(new_path)
            if actual_sha256 != asset["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {new_path}: "
                    f"{actual_sha256} != {asset['sha256']}"
                )
            asset["local_path"] = str(new_path)
            image_count += 1
        print(
            "FACE_QUALITY_TRANSFER_STEP_VERIFIED "
            f"step={step} images={len(assets)}"
        )

    payload["transfer_verification"] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_machine": "serv",
        "destination_machine": "neb",
        "old_root": old_root,
        "new_root": str(new_root),
        "image_count": image_count,
        "file_size_verified": True,
        "sha256_verified": True,
        "pil_decode_verified": True,
    }
    if args.write:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    print(
        "FACE_QUALITY_TRANSFER_VERIFIED "
        f"images={image_count} manifest={manifest_path} write={args.write}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
