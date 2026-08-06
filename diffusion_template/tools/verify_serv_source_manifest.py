#!/usr/bin/env python3
"""Build or verify an exact file manifest for a sealed Serv source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
IGNORED_TOP_LEVEL = {".env", "logs", "saved"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in IGNORED_TOP_LEVEL:
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Unexpected symlink inside sealed source: {relative}")
        if path.is_file():
            files[relative.as_posix()] = sha256(path)
    return files


def build(root: Path, output: Path, source_revision: str) -> None:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest must be stored outside the source root")
    record = {
        "schema_version": 1,
        "source_revision": source_revision,
        "files": inventory(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def verify(root: Path, manifest: Path) -> None:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    expected = record.get("files")
    if record.get("schema_version") != 1 or not isinstance(expected, dict):
        raise RuntimeError(f"Invalid source manifest: {manifest}")
    actual = inventory(root)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        path for path in set(expected) & set(actual) if expected[path] != actual[path]
    )
    if missing or extra or changed:
        raise RuntimeError(
            "Sealed source verification failed: "
            f"missing={missing[:8]}, extra={extra[:8]}, changed={changed[:8]}"
        )
    print(
        "Sealed Serv source verified: "
        f"revision={record.get('source_revision')}, files={len(actual)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--source-revision", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"source root is not a directory: {root}")
    if args.command == "build":
        build(root, args.output.expanduser().resolve(), args.source_revision)
    else:
        verify(root, args.manifest.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
