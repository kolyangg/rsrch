#!/usr/bin/env python3
"""Convert explicit CL39-X05 preview outputs into a sealed cache recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.photomaker_branched.masking.automask_os import POLICY_VERSION
from src.model.photomaker_branched.masking.automask_os import image_sha256
from PIL import Image
from src.pipelines.validation_automask import (
    PREVIEW_PROTOCOL,
    PREVIEW_PROTOCOL_SHA256,
    recipe_record,
    validation_reference_identity,
    validation_target_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-version", default=POLICY_VERSION)
    args = parser.parse_args()
    rows = json.loads(args.preview_index.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError("X05 preview index must contain the fixed 96 validation rows")
    records, seen_references = [], set()
    for row in rows:
        required = {"id", "prompt", "seed", "preview_image_path", "reference_image_path"}
        if required - set(row):
            raise ValueError(f"Incomplete X05 preview row: {sorted(required-set(row))}")
        reference_hash = image_sha256(
            Image.open(row["reference_image_path"]).convert("RGB")
        )
        reference_identity = validation_reference_identity(
            row["id"], args.policy_version, reference_hash
        )
        reference_key = json.dumps(reference_identity, sort_keys=True)
        if reference_key not in seen_references:
            records.append(recipe_record(
                image_path=row["reference_image_path"],
                reference_image_path=row["reference_image_path"],
                cache_identity=reference_identity,
            ))
            seen_references.add(reference_key)
        records.append(recipe_record(
            image_path=row["preview_image_path"],
            reference_image_path=row["reference_image_path"],
            cache_identity=validation_target_identity(
                row["id"], row["prompt"], row["seed"], args.policy_version,
                reference_hash,
            ),
            expected_location=row.get("expected_location"),
        ))
    payload = {
        "preview_protocol": PREVIEW_PROTOCOL,
        "preview_protocol_sha256": PREVIEW_PROTOCOL_SHA256,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "protocol": PREVIEW_PROTOCOL_SHA256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
