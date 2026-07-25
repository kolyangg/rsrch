#!/usr/bin/env python3
"""Seed and verify the machine-local Cosmic full-96 bbox protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


INFERENCE_TOOLS = Path(__file__).resolve().parents[1] / "inference"
sys.path.insert(0, str(INFERENCE_TOOLS))
from full96_protocol import (  # noqa: E402
    AUTO_SEED_SHA256,
    MANUAL_SHA256,
    PROTOCOL_ID,
    load_object,
    sha256,
    validate_bbox_routing,
    validate_static_inputs,
)


def copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: Path, value: dict) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-manual", type=Path, required=True)
    parser.add_argument("--current-auto-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless all 95 automatic plus one forced-manual routes exist",
    )
    args = parser.parse_args()

    manual_source = args.historical_manual.resolve()
    auto_source = args.current_auto_seed.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    if sha256(manual_source) != MANUAL_SHA256:
        raise ValueError("Historical manual bbox source has an unexpected SHA-256")
    if sha256(auto_source) != AUTO_SEED_SHA256:
        raise ValueError("Current 12-entry automatic bbox seed has an unexpected SHA-256")

    manual_source_data = load_object(manual_source)
    auto_source_data = load_object(auto_source)
    if len(manual_source_data) != 96:
        raise ValueError("Historical manual bbox source must have 96 entries")
    if len(auto_source_data) != 12:
        raise ValueError("Automatic bbox seed must contain exactly 12 Eddie entries")
    validate_bbox_routing(manual_source_data, auto_source_data)

    manual_target = output_dir / "pm96_bboxes_new.json"
    auto_target = output_dir / "pm96_bboxes_new_auto.json"
    copy_once(manual_source, manual_target)
    copy_once(auto_source, auto_target)

    if sha256(manual_target) != MANUAL_SHA256:
        raise ValueError("Existing protocol manual bbox file differs from the seal")
    manual_target_data = load_object(manual_target)
    auto_target_data = load_object(auto_target)
    for key, value in auto_source_data.items():
        if auto_target_data.get(key) != value:
            raise ValueError(f"Protocol automatic bbox seed changed at {key}")
    routing = validate_bbox_routing(
        manual_target_data,
        auto_target_data,
        require_complete=args.require_complete,
    )
    validation_data_dir = output_dir.parent.parent
    static_inputs = validate_static_inputs(validation_data_dir)

    now = datetime.now(timezone.utc).isoformat()
    manifest_path = output_dir / "protocol_manifest.json"
    existing = load_object(manifest_path) if manifest_path.exists() else {}
    manifest = dict(existing)
    manifest.update(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": existing.get("created_at_utc", now),
            "updated_at_utc": now,
            "manual_bbox": {
                "path": str(manual_target),
                "entries": 96,
                "sha256": sha256(manual_target),
            },
            "automatic_bbox": {
                "path": str(auto_target),
                "entries": routing["automatic_entries"],
                "sha256": sha256(auto_target),
                "seed_entries": 12,
                "seed_sha256": AUTO_SEED_SHA256,
                "status": "complete" if routing["complete"] else "seeded",
            },
            "force_manual": {
                "entries": routing["force_manual_entries"],
                "keys": routing["force_manual_keys"],
            },
            "routing_entries": routing["routing_entries"],
            "static_inputs": static_inputs,
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(
        "FULL96_PROTOCOL_READY "
        f"manual_entries=96 auto_entries={routing['automatic_entries']} "
        f"force_manual_entries={routing['force_manual_entries']} "
        f"routing_complete={routing['complete']} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
