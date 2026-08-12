#!/usr/bin/env python3
"""Complete schedule audit plus deterministic target/reference decode smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from src.datasets.big_celebs_e13_scheduled import BigCelebsE13ScheduledTrain


def decode(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError(f"Scheduled image is not RGB: {path}")
        return {"path": str(path), "size": list(image.size), "format": image.format}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ds1", "ds2", "ds3"], required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--big-manifest", required=True)
    parser.add_argument("--big-images", required=True)
    parser.add_argument("--big-manifest-sha256", required=True)
    parser.add_argument("--large-manifest")
    parser.add_argument("--large-images")
    parser.add_argument("--large-manifest-sha256")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = BigCelebsE13ScheduledTrain(
        schedule_path=args.schedule,
        schedule_summary_path=args.summary,
        expected_schedule_sha256=args.expected_schedule_sha256,
        expected_mode=args.mode,
        big_manifest_path=args.big_manifest,
        big_images_path=args.big_images,
        expected_big_manifest_sha256=args.big_manifest_sha256,
        large_manifest_path=args.large_manifest,
        large_images_path=args.large_images,
        expected_large_manifest_sha256=args.large_manifest_sha256,
        schedule_start_row=0,
        instance_transforms=None,
    )
    rows = dataset.schedule
    sample_count = min(max(args.sample_count, 1), len(rows))
    indices = sorted(
        {
            round(index * (len(rows) - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        }
    )
    decoded = []
    for index in indices:
        row = rows[index]
        root = dataset.source_roots[row["source"]]
        target = root / row["target_path"]
        reference = root / row["reference_path"]
        decoded.append(
            {
                "schedule_index": index,
                "source": row["source"],
                "target": decode(target),
                "reference": decode(reference),
            }
        )
    sources = Counter(row["source"] for row in rows)
    roles = Counter(
        f"{row['source']}:{row['target_role']}" for row in rows
    )
    result = {
        "status": "ok",
        "mode": args.mode,
        "schedule_sha256": args.expected_schedule_sha256,
        "rows": len(rows),
        "optimizer_steps": len(rows) // 2,
        "sources": dict(sorted(sources.items())),
        "target_roles": dict(sorted(roles.items())),
        "complete_schedule_loader_audit": True,
        "decoded_pairs": decoded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
