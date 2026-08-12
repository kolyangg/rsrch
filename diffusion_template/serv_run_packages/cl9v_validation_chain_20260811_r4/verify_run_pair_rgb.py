#!/usr/bin/env python3
"""Fail closed unless two fixed-panel sidecar directories have identical RGB."""

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


def indexed_rows(run_dir: Path) -> dict[int, dict]:
    rows = json.loads((run_dir / "per_image.json").read_text(encoding="utf-8"))
    result = {int(row["dataset_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate dataset indices in {run_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = args.first.resolve()
    second = args.second.resolve()
    first_rows = indexed_rows(first)
    second_rows = indexed_rows(second)
    indices = sorted(set(first_rows) & set(second_rows))
    rows = []
    for index in indices:
        first_path = first / "images" / first_rows[index]["filename"]
        second_path = second / "images" / second_rows[index]["filename"]
        first_hash, first_size = rgb_hash(first_path)
        second_hash, second_size = rgb_hash(second_path)
        rows.append(
            {
                "dataset_index": index,
                "first_path": str(first_path),
                "second_path": str(second_path),
                "first_rgb_sha256": first_hash,
                "second_rgb_sha256": second_hash,
                "first_size": first_size,
                "second_size": second_size,
                "exact": first_hash == second_hash and first_size == second_size,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "sidecar_pair_exact_rgb_gate",
        "expected_count": args.expect_count,
        "checked_count": len(rows),
        "exact_count": sum(int(row["exact"]) for row in rows),
        "mismatch_count": sum(int(not row["exact"]) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if len(first_rows) != args.expect_count or len(second_rows) != args.expect_count:
        raise SystemExit(
            f"Run sizes differ from expected {args.expect_count}: "
            f"{len(first_rows)} and {len(second_rows)}"
        )
    if len(rows) != args.expect_count or result["mismatch_count"]:
        raise SystemExit(
            "Sidecar RGB pair gate failed: "
            f"checked={len(rows)} expected={args.expect_count} "
            f"mismatches={result['mismatch_count']}"
        )
    print(f"RUN_PAIR_RGB_EXACT {len(rows)}/{args.expect_count}")


if __name__ == "__main__":
    main()
