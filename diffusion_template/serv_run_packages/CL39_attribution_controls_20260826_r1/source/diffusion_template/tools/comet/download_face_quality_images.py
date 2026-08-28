#!/usr/bin/env python3
"""Download and verify exact Comet image steps without running the IQA scorer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backfill_face_quality_metrics import (
    DEFAULT_STEPS,
    CometRestClient,
    build_download_manifest,
    parse_int_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage exact Comet validation images and a verified manifest."
    )
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--steps", default=",".join(str(step) for step in DEFAULT_STEPS))
    parser.add_argument("--images-per-step", type=int, default=96)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.getenv("COMET_API_KEY"))
    parser.add_argument("--base-url", default="https://www.comet.com")
    parser.add_argument("--download-retries", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("COMET_API_KEY is required")
    steps = parse_int_list(args.steps)
    if args.images_per_step < 1:
        raise ValueError("--images-per-step must be positive")
    if args.download_retries < 1:
        raise ValueError("--download-retries must be positive")

    client = CometRestClient(args.api_key, args.base_url, timeout=120)
    metadata = client.get_json(
        "/experiment/metadata",
        experimentKey=args.experiment_key,
    )
    project_name = str(metadata.get("projectName") or "")
    if project_name != args.expected_project:
        raise ValueError(
            f"Experiment is in project {project_name!r}, expected "
            f"{args.expected_project!r}"
        )

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_download_manifest(
        client,
        args.experiment_key,
        project_name,
        steps,
        args.images_per_step,
        work_dir / "images",
        args.download_retries,
    )
    manifest_path = work_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    image_count = sum(len(assets) for assets in manifest["steps"].values())
    print(
        "FACE_QUALITY_STAGING_COMPLETE "
        f"key={args.experiment_key} steps={len(steps)} images={image_count} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
