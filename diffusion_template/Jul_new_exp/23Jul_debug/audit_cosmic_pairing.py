#!/usr/bin/env python3
"""Fail unless the single-ID Cosmic target and all references are distinct."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_JSON = HERE / "data" / "id_00081_1017318003459" / "train_8refs.json"
SPLIT = HERE / "data" / "id_00081_1017318003459" / "split_manifest.json"
TARGET_ROOT = Path("/home/niko/datasets")
REFERENCE_ROOT = Path(
    "/home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if len(data) != 1:
        raise RuntimeError(f"Expected one Cosmic target record, found {len(data)}")
    target_key, record = next(iter(data.items()))
    target = TARGET_ROOT / target_key
    if not target.exists():
        raise FileNotFoundError(target)
    target_hash = sha256(target)
    refs = []
    violations = []
    for ref_key in record["face_paths"]:
        relative = ref_key.removeprefix(
            "LAION-5B-Filtered-Large-Faces/laion1B-nolang/"
        )
        ref = REFERENCE_ROOT / relative
        if not ref.exists():
            raise FileNotFoundError(ref)
        ref_hash = sha256(ref)
        same_path = ref.resolve() == target.resolve()
        same_file_hash = ref_hash == target_hash
        if same_path or same_file_hash:
            violations.append(
                {
                    "reference": str(ref),
                    "same_path": same_path,
                    "same_file_hash": same_file_hash,
                }
            )
        refs.append(
            {
                "path": str(ref),
                "sha256": ref_hash,
                "different_from_target": not same_path and not same_file_hash,
            }
        )
    training_ref_keys = set(record["face_paths"])
    holdout_overlap = [
        holdout["source_path"]
        for holdout in split["holdouts"]
        if holdout["source_path"] in training_ref_keys
    ]
    if holdout_overlap:
        violations.append({"validation_holdout_overlap": holdout_overlap})
    payload = {
        "dataset_class": "src.datasets.cosmic.CosmicLargeTrain",
        "target": str(target),
        "target_sha256": target_hash,
        "reference_count": len(refs),
        "references": refs,
        "validation_holdout_overlap": holdout_overlap,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit("Cosmic target/reference audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
