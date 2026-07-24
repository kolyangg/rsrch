#!/usr/bin/env python3
"""Fail fast unless OneIDTrain uses a different same-ID reference image."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CONFIGS = PROJECT / "src" / "configs"
SUBSET = HERE / "data" / "one_id_nm0005092" / "subset8_train.json"
IMAGES = Path("/home/niko/rsrch/dataset_full/one_id/nm0005092_adj")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-json",
        type=Path,
        default=SUBSET,
        help="OneID annotation JSON to instantiate and audit.",
    )
    parser.add_argument(
        "--allow-same-image",
        action="store_true",
        help="Audit the historical leakage branch without failing.",
    )
    parser.add_argument("--seeds-per-target", type=int, default=8)
    return parser.parse_args()


def pixel_digest(image) -> str:
    array = np.asarray(image.convert("RGB"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.seeds_per_target < 1:
        raise ValueError("--seeds-per-target must be positive")
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    separate = not args.allow_same_image
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="one_id_ba_N3a_new1",
            overrides=[
                "datasets=all_datasets",
                "train_dataset_name=one_id",
                f"datasets.train.one_id.cosmic_json_pth={args.data_json.resolve()}",
                f"datasets.train.one_id.images_path={IMAGES}",
                "datasets.train.one_id.num_refs=1",
                f"train_on_separate_image={'true' if separate else 'false'}",
            ],
        )
    dataset = instantiate(cfg.datasets.train.one_id)
    source_by_digest = {
        pixel_digest(Image.open(IMAGES / name)): name
        for name in dataset.ids
    }

    records = []
    violations = []
    for target_index, target_name in enumerate(dataset.ids):
        observed_refs = set()
        for seed in range(args.seeds_per_target):
            random.seed(10_000 * target_index + seed)
            np.random.seed(10_000 * target_index + seed)
            sample = dataset[target_index]
            ref_digest = pixel_digest(sample["ref_images"][0])
            ref_name = source_by_digest.get(ref_digest, "<unresolved>")
            observed_refs.add(ref_name)
            different = ref_name != target_name
            if separate and not different:
                violations.append(
                    {
                        "target": target_name,
                        "reference": ref_name,
                        "seed": seed,
                    }
                )
        records.append(
            {
                "target": target_name,
                "observed_references": sorted(observed_refs),
                "target_observed_as_reference": target_name in observed_refs,
            }
        )

    payload = {
        "dataset_class": f"{type(dataset).__module__}.{type(dataset).__name__}",
        "data_json": str(args.data_json.resolve()),
        "subset_count": len(dataset),
        "train_on_separate_image": dataset.train_on_separate_image,
        "seeds_per_target": args.seeds_per_target,
        "records": records,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit("Reference/target pairing audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
