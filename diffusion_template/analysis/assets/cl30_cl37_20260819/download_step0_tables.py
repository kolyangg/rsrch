#!/usr/bin/env python3
"""Download exact step-zero per-image tables without downloading image panels."""

from __future__ import annotations

import os
from pathlib import Path

from tools.comet.export_comet_runs import CometRestClient


OUTPUT = Path(__file__).resolve().parent / "tables"
RUNS = {
    "CL30": "db38cfb250d241cf89bf57705ff86b18",
    "CL31": "ed5077fd3cfc41bd898c1234b8c3ba24",
    "CL32": "078cf231674f4fa499e160a435300511",
    "CL33": "3173f3086fa344f7ad3eb6ce7b07ac1f",
    "CL34": "577cc412ffa04e5686e5c10760186c65",
    "CL35": "f3417ee9a86342cb9bc13e5eb37bb3e2",
    "CL36": "41dcb0987d5d439bb14329052953ff6d",
    "CL37": "f3c535315da242d78494d7df6dd1eaa3",
}


def main() -> None:
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        raise RuntimeError("COMET_API_KEY is required")
    client = CometRestClient(api_key, "https://www.comet.com", timeout=120)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for label, key in RUNS.items():
        payload = client.get_json("/experiment/asset/list", experimentKey=key, type="all")
        expected = "id_sim__manual_val__step_000000.csv"
        matches = [asset for asset in payload.get("assets", []) if asset.get("fileName") == expected]
        if len(matches) != 1:
            raise RuntimeError(f"{label}: expected one {expected}, found {len(matches)}")
        destination = OUTPUT / f"{label}_step_000000.csv"
        client.download_asset(key, str(matches[0]["assetId"]), destination)
        if len(destination.read_text(encoding="utf-8").splitlines()) != 97:
            raise RuntimeError(f"{label}: invalid table row count")
        print(label, key, matches[0]["assetId"], destination)


if __name__ == "__main__":
    main()
