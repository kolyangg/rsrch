#!/usr/bin/env python3
"""Snapshot the native one_id validation inputs used by the dataset ablation.

The generated-face boxes come from the repository's PhotoMaker-only automatic
bbox pass for reference 51.jpg, seed 0.  Only the first four protocol prompts
are retained so the local mask bundle exactly matches this experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path("/home/niko/rsrch/dataset_full/one_id")
OUTPUT = HERE / "data" / "one_id_nm0005092"
PROMPTS = Path("/home/niko/rsrch/dataset_full/val_dataset/prompts_10.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_bbox = SOURCE / "one_id_ref_bboxes_auto.json"
    source_records = json.loads(source_bbox.read_text(encoding="utf-8"))
    prompts = [
        line.strip()
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:4]
    resolved_prompts = [prompt.replace("<class>", "man img") for prompt in prompts]

    selected = {}
    for index, prompt in enumerate(resolved_prompts):
        matches = [
            (key, record)
            for key, record in source_records.items()
            if record.get("_meta", {}).get("prompt") == prompt
            and record.get("_meta", {}).get("id") == "51"
            and int(record.get("_meta", {}).get("seed", -1)) == 0
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one PhotoMaker bbox for prompt {index}: {prompt!r}; "
                f"found {len(matches)}"
            )
        key, record = matches[0]
        selected[key] = record

    destination = OUTPUT / "pm_generated_bboxes_ref51_seed0.json"
    destination.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "kind": "photomaker_only_automatic_generation_bbox",
        "identity": "nm0005092",
        "reference": str(SOURCE / "ref" / "51.jpg"),
        "reference_sha256": sha256(SOURCE / "ref" / "51.jpg"),
        "seed": 0,
        "prompts": resolved_prompts,
        "source": str(source_bbox),
        "source_sha256": sha256(source_bbox),
        "selected_record_count": len(selected),
        "destination": str(destination),
        "destination_sha256": sha256(destination),
        "note": (
            "The source is the repository's automatic generated-face bbox file. "
            "Its per-record _meta binds each box to the PhotoMaker validation "
            "prompt, reference id 51, and seed 0. BA validation reuses these "
            "PhotoMaker-derived boxes and never derives masks from BA outputs."
        ),
    }
    (OUTPUT / "PM_GENERATED_MASK_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
