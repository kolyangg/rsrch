#!/usr/bin/env python3
"""Initialize or publish one fixed-checkpoint validation arm to Comet."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger.cometml import CometMLWriter


class _ConsoleLogger:
    @staticmethod
    def info(message, *args):
        print(message % args if args else message)

    @staticmethod
    def warning(message, *args):
        print(message % args if args else message, file=sys.stderr)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_leaves(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from numeric_leaves(child, child_prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def aggregate_per_image_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in numeric_leaves(row.get("metrics", {})):
            values[name].append(value)
    return {
        name: sum(items) / len(items)
        for name, items in sorted(values.items())
        if items
    }


def face_quality_scalars(payload: dict[str, Any], step: int) -> dict[str, float]:
    result = payload.get("steps", {}).get(str(step))
    if not isinstance(result, dict):
        raise KeyError(f"Face-quality result has no step {step}")
    scalars = {
        "face_detection_rate": float(result["face_detection_rate"]),
    }
    aliases = {
        "topiq_nr_face": "topiq_face",
        "topiq_nr": "topiq",
        "musiq": "musiq",
        "maniqa_pipal": "maniqa",
    }
    image_count = int(result["image_count"])
    for source_name, metric_result in result.get("metrics", {}).items():
        alias = aliases.get(source_name, source_name)
        for statistic in ("mean", "p10"):
            value = metric_result.get(statistic)
            if value is not None:
                scalars[f"{alias}_{statistic}"] = float(value)
        scalars[f"{alias}_coverage"] = (
            float(metric_result.get("count", 0)) / float(image_count)
        )
    return scalars


def build_writer(args, spec: dict[str, Any]) -> CometMLWriter:
    project_config = dict(spec)
    project_config.setdefault("trainer", {})["save_dir"] = "saved"
    writer = CometMLWriter(
        logger=_ConsoleLogger(),
        project_config=project_config,
        project_name=args.project_name,
        workspace=args.workspace,
        run_id=args.run_id,
        run_name=args.run_name,
        mode="online",
        tags=["fixed-checkpoint", "full96", "D0", "branched-attention"],
    )
    if writer._experiment is None or not writer.run_id:
        raise RuntimeError("Comet experiment initialization failed")
    return writer


def finish(writer: CometMLWriter) -> None:
    writer._experiment.end()


def initialize(args, spec: dict[str, Any]) -> None:
    writer = build_writer(args, spec)
    try:
        writer.step = int(args.step)
        writer.mode = "validation"
        writer.add_scalar("general/checkpoint_step", args.step)
        writer.add_asset(
            "experiment_spec.json",
            args.spec,
            metadata={"kind": "fixed_checkpoint_validation_spec"},
        )
        print(f"COMET_EXPERIMENT_KEY={writer.run_id}")
    finally:
        finish(writer)


def publish(args, spec: dict[str, Any]) -> None:
    if not args.run_id:
        raise ValueError("--run-id is required when publishing an arm")
    if args.arm_dir is None:
        raise ValueError("--arm-dir is required when publishing an arm")
    arm_dir = args.arm_dir.resolve()
    required = [
        arm_dir / "run_manifest.json",
        arm_dir / "command_manifest.json",
        arm_dir / "resolved_config.yaml",
        arm_dir / "per_image.json",
        arm_dir / "face_quality_metrics.json",
        arm_dir / "face_quality_per_image.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = load_json(arm_dir / "per_image.json")
    if not isinstance(rows, list) or len(rows) != 96:
        raise RuntimeError(f"Expected 96 per-image rows in {arm_dir}")
    image_paths = [arm_dir / "images" / str(row["filename"]) for row in rows]
    if any(not path.is_file() for path in image_paths):
        raise RuntimeError(f"One or more generated images are missing in {arm_dir}")

    writer = build_writer(args, spec)
    try:
        writer.step = int(args.step)
        writer.mode = "validation"
        for name, value in aggregate_per_image_metrics(rows).items():
            writer.add_scalar(f"manual_val/{name}", value)
        quality = load_json(arm_dir / "face_quality_metrics.json")
        for name, value in face_quality_scalars(quality, args.step).items():
            writer.add_scalar(f"manual_val/face_quality/{name}", value)

        for path in image_paths:
            with Image.open(path) as opened:
                writer.add_image(path.name, opened.convert("RGB"))

        for path in required:
            writer.add_asset(
                path.name,
                path,
                metadata={
                    "kind": "fixed_checkpoint_validation_output",
                    "step": int(args.step),
                },
            )
        print(f"COMET_ARM_PUBLISHED key={writer.run_id} images={len(image_paths)}")
    finally:
        finish(writer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-name", default="jul-comet-large-testing-tr")
    parser.add_argument("--workspace")
    parser.add_argument("--run-id")
    parser.add_argument("--step", type=int, default=32000)
    parser.add_argument("--arm-dir", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()

    spec = load_json(args.spec.resolve())
    if not isinstance(spec, dict):
        raise TypeError(f"Expected a JSON object in {args.spec}")
    if spec.get("run_name") != args.run_name:
        raise ValueError(
            f"Spec run_name={spec.get('run_name')!r} does not match "
            f"--run-name={args.run_name!r}"
        )
    if args.initialize_only:
        initialize(args, spec)
    else:
        publish(args, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
