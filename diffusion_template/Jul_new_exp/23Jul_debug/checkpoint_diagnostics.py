#!/usr/bin/env python3
"""Summarize learned NN3a LoRA-B drift at each saved checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def group_for(processor_name: str, tensor_name: str) -> tuple[str, str, str]:
    if processor_name.startswith("up_blocks.0."):
        block = "up0"
    elif processor_name.startswith("up_blocks.1."):
        block = "up1"
    elif processor_name.startswith("down_blocks."):
        block = "down"
    elif processor_name.startswith("mid_block."):
        block = "mid"
    else:
        block = "other"

    if ".ref_to_" in f".{tensor_name}":
        branch = "ref"
    elif ".noise_to_" in f".{tensor_name}":
        branch = "noise"
    else:
        branch = "other"

    projection = "other"
    for candidate in ("q", "k", "v"):
        if f"_to_{candidate}." in tensor_name:
            projection = candidate
            break
    return block, branch, projection


def main():
    run_dir = parse_args().run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint_every = int(
        manifest.get("protocol", {}).get("checkpoint_every", 200)
    )
    checkpoint_dir = run_dir / "checkpoints" / manifest["run_name"]
    results = []
    for checkpoint_path in sorted(checkpoint_dir.glob("checkpoint-epoch*.pth")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        accumulators = defaultdict(
            lambda: {"tensor_count": 0, "numel": 0, "l2_sq": 0.0, "max_abs": 0.0}
        )
        processor_state = checkpoint["state_dict"].get("attn_processors", {})
        for processor_name, tensors in processor_state.items():
            for tensor_name, tensor in tensors.items():
                if "lora_B" not in tensor_name:
                    continue
                value = tensor.detach().float()
                block, branch, projection = group_for(processor_name, tensor_name)
                keys = (
                    f"branch/{branch}",
                    f"block/{block}",
                    f"projection/{branch}_{projection}",
                    f"block_branch/{block}_{branch}",
                    "all",
                )
                for key in keys:
                    record = accumulators[key]
                    record["tensor_count"] += 1
                    record["numel"] += int(value.numel())
                    record["l2_sq"] += float(value.square().sum())
                    record["max_abs"] = max(
                        record["max_abs"], float(value.abs().max())
                    )
        groups = {}
        for key, record in sorted(accumulators.items()):
            record["l2"] = math.sqrt(record.pop("l2_sq"))
            record["rms"] = record["l2"] / math.sqrt(max(record["numel"], 1))
            groups[key] = record
        results.append(
            {
                "epoch": epoch,
                "step": epoch * checkpoint_every,
                "checkpoint": str(checkpoint_path),
                "processor_state_count": len(processor_state),
                "groups": groups,
            }
        )

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "checkpoint_weight_diagnostics.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        ref = result["groups"].get("branch/ref", {}).get("l2", 0.0)
        noise = result["groups"].get("branch/noise", {}).get("l2", 0.0)
        print(
            f"step={result['step']} processors={result['processor_state_count']} "
            f"ref_B_l2={ref:.6f} noise_B_l2={noise:.6f}"
        )
    print(output)


if __name__ == "__main__":
    main()
