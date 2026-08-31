#!/usr/bin/env python3
"""Record the required human/model visual review without weakening metric gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    payload = json.loads(args.confirmation.read_text(encoding="utf-8"))
    if payload.get("status") != "metrics_pass_pending_visual_review":
        raise RuntimeError(f"Confirmation is not awaiting review: {payload.get('status')}")
    root = args.confirmation.parent
    pages = payload.get("visual_review_pages") or []
    if len(pages) != 8 or any(not (root / page).is_file() for page in pages):
        raise RuntimeError("The complete eight-page visual review set is absent")
    payload["visual_review"] = {
        "decision": args.decision, "reviewer": args.reviewer, "note": args.note,
        "page_sha256": {
            page: hashlib.sha256((root / page).read_bytes()).hexdigest()
            for page in pages
        },
    }
    payload["status"] = args.decision
    args.confirmation.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "reviewer": args.reviewer}, indent=2))


if __name__ == "__main__":
    main()
