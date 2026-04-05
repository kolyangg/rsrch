#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _auto_name_end(pm: int, ba_start: int, ba_end) -> str:
    if ba_end is None:
        return f"{pm}{ba_start}none"
    return f"{pm}{ba_start}{ba_end}"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_infer_combinations.py CONFIG_JSON", file=sys.stderr)
        return 1

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    repo_root = Path(__file__).resolve().parents[1]
    accelerate_config = cfg.get("accelerate_config_file", "src/configs/ddp/accelerate.yaml")
    infer_script = cfg.get("infer_script", "infer.py")
    num_processes = int(cfg.get("num_processes", 1))
    config_name = str(cfg["config_name"])
    batch_size = int(cfg["batch_size"])
    project_name = str(cfg["writer.project_name"])
    run_name_base = str(cfg["writer.run_name_base"])
    extra_overrides = [str(x) for x in cfg.get("extra_overrides", [])]
    combinations = list(cfg["combinations"])

    for combo in combinations:
        pm = int(combo["photomaker_start_step"])
        ba_start = int(combo["branched_attn_start_step"])
        ba_end = combo.get("branched_attn_end_step", None)
        ba_end_override = "null" if ba_end is None else str(int(ba_end))
        name_end = str(combo.get("name_end") or _auto_name_end(pm, ba_start, ba_end))
        run_name = f"{run_name_base}_{name_end}"

        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(num_processes),
            infer_script,
            "--config-name",
            config_name,
            "batch_size=" + str(batch_size),
            "photomaker_start_step=" + str(pm),
            "branched_attn_start_step=" + str(ba_start),
            "branched_attn_end_step=" + ba_end_override,
            "writer.project_name=" + project_name,
            "writer.run_name=" + run_name,
            *extra_overrides,
        ]

        print("\n[run_infer_combinations] Running:")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
