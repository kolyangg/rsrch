#!/usr/bin/env python3
"""Fail closed unless selected sidecar outputs equal historical CL9 RGB pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def rgb_hash(path: Path) -> tuple[str, list[int]]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    return hashlib.sha256(image.tobytes()).hexdigest(), list(image.size)


def parse_indices(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--step", type=int, default=24000)
    parser.add_argument("--skip-indices")
    parser.add_argument("--expect-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay_dir = args.replay_dir.resolve()
    per_image = json.loads((replay_dir / "per_image.json").read_text(encoding="utf-8"))
    historical = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    assets = historical["steps"].get(str(args.step))
    if not isinstance(assets, list):
        raise KeyError(f"Historical manifest has no step {args.step}")
    by_index = {int(asset["sample_index"]): asset for asset in assets}
    if len(by_index) != 96:
        raise ValueError(f"Expected 96 unique historical rows, found {len(by_index)}")

    skipped = parse_indices(args.skip_indices)
    rows = []
    for row in per_image:
        index = int(row["dataset_index"])
        if index in skipped:
            continue
        asset = by_index[index]
        replay_path = replay_dir / "images" / row["filename"]
        historical_path = Path(asset["local_path"])
        replay_hash, replay_size = rgb_hash(replay_path)
        historical_hash, historical_size = rgb_hash(historical_path)
        rows.append(
            {
                "dataset_index": index,
                "replay_path": str(replay_path),
                "historical_path": str(historical_path),
                "replay_rgb_sha256": replay_hash,
                "historical_rgb_sha256": historical_hash,
                "replay_size": replay_size,
                "historical_size": historical_size,
                "exact": replay_hash == historical_hash and replay_size == historical_size,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "exact_rgb_replay_gate",
        "step": args.step,
        "expected_count": args.expect_count,
        "checked_count": len(rows),
        "skipped_indices": sorted(skipped),
        "exact_count": sum(int(row["exact"]) for row in rows),
        "mismatch_count": sum(int(not row["exact"]) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if len(rows) != args.expect_count or result["mismatch_count"]:
        raise SystemExit(
            "RGB replay gate failed: "
            f"checked={len(rows)} expected={args.expect_count} "
            f"mismatches={result['mismatch_count']}"
        )
    print(f"RGB_REPLAY_EXACT {len(rows)}/{args.expect_count}")


if __name__ == "__main__":
    main()
