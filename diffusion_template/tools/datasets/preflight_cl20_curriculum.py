#!/usr/bin/env python3
"""Loader-level boundary preflight for the sealed CL20 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "src" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name)
    dataset_config = OmegaConf.to_container(
        config.datasets.train[config.train_dataset_name], resolve=True
    )
    dataset_config.pop("instance_transforms", None)
    dataset = instantiate(dataset_config, _convert_="all")
    rows = dataset.rows
    first_phase = rows[:40000]
    final_phase = rows[40000:]
    counts = {
        "first_cosmic": sum(row["source"] == "cosmic" for row in first_phase),
        "first_big": sum(row["source"] == "big_celebs" for row in first_phase),
        "final_cosmic": sum(row["source"] == "cosmic" for row in final_phase),
        "final_big": sum(row["source"] == "big_celebs" for row in final_phase),
    }
    if counts != {
        "first_cosmic": 32000,
        "first_big": 8000,
        "final_cosmic": 8000,
        "final_big": 0,
    }:
        raise RuntimeError(f"CL20 schedule distribution drift: {counts}")
    probes = [0, 1, 39998, 39999, 40000, 40001, 47998, 47999]
    decoded = []
    for index in probes:
        sample = dataset[index]
        if sample["target_path"] == sample["reference_path"]:
            raise RuntimeError(f"CL20 target/reference leakage at row {index}")
        decoded.append({
            "row": index,
            "source": rows[index]["source"],
            "target": sample["target_path"],
            "reference": sample["reference_path"],
        })
    report = {"status": "ok", "rows": len(rows), "counts": counts, "decoded": decoded}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
