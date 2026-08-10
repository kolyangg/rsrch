#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DOC-01: Retained the hash-verifying project uploader for
# report artifacts; credentials remain in the ignored machine-local .env.
"""Upload local files to Dropbox under /rsrch/YYYY-MM-DD/.

Credentials are loaded from diffusion_template/.env using Dropbox's
refresh-token flow. Uploaded bytes are verified against Dropbox's content hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
DROPBOX_ROOT = "/rsrch"
CONTENT_BLOCK_SIZE = 4 * 1024 * 1024
UPLOAD_SESSION_THRESHOLD = 100 * 1024 * 1024


class DropboxError(RuntimeError):
    """A Dropbox request failed."""


def _load_env(path: Path) -> None:
    if not path.is_file():
        raise DropboxError(f"credential file not found: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            parsed = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise DropboxError(f"invalid .env value on line {line_number}") from exc
        os.environ.setdefault(key, parsed[0] if parsed else "")


def _credentials() -> tuple[str, str, str]:
    _load_env(ENV_PATH)
    names = ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN")
    values = tuple(os.environ.get(name, "").strip() for name in names)
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise DropboxError(f"missing Dropbox credentials in {ENV_PATH}: {', '.join(missing)}")
    return values  # type: ignore[return-value]


def _request(
    url: str,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    data: bytes = b"",
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DropboxError(f"Dropbox API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DropboxError(f"Dropbox request failed: {exc.reason}") from exc
    return json.loads(body) if body else {}


def _access_token() -> str:
    app_key, app_secret, refresh_token = _credentials()
    payload = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        }
    ).encode("ascii")
    response = _request(
        "https://api.dropboxapi.com/oauth2/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=payload,
    )
    token = response.get("access_token")
    if not isinstance(token, str) or not token:
        raise DropboxError("Dropbox token response did not contain an access token")
    return token


def _rpc(token: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request(
        f"https://api.dropboxapi.com/2/{method}",
        token=token,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )


def _content_request(token: str, method: str, args: dict[str, Any], data: bytes) -> dict[str, Any]:
    return _request(
        f"https://content.dropboxapi.com/2/{method}",
        token=token,
        headers={
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps(args, separators=(",", ":")),
        },
        data=data,
    )


def _ensure_folder(token: str, path: str) -> None:
    try:
        _rpc(token, "files/create_folder_v2", {"path": path, "autorename": False})
    except DropboxError as exc:
        # Dropbox reports an existing folder as a 409 path/conflict/folder error.
        if "conflict" not in str(exc) or "folder" not in str(exc):
            raise


def _content_hash(path: Path) -> str:
    overall = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CONTENT_BLOCK_SIZE):
            overall.update(hashlib.sha256(block).digest())
    return overall.hexdigest()


def _upload(token: str, source: Path, destination: str) -> dict[str, Any]:
    commit = {"path": destination, "mode": "overwrite", "autorename": False, "mute": False}
    if source.stat().st_size <= UPLOAD_SESSION_THRESHOLD:
        return _content_request(token, "files/upload", commit, source.read_bytes())

    with source.open("rb") as handle:
        first_chunk = handle.read(CONTENT_BLOCK_SIZE)
        started = _content_request(token, "files/upload_session/start", {"close": False}, first_chunk)
        session_id = started["session_id"]
        offset = len(first_chunk)
        while True:
            chunk = handle.read(CONTENT_BLOCK_SIZE)
            cursor = {"session_id": session_id, "offset": offset}
            if len(chunk) < CONTENT_BLOCK_SIZE:
                return _content_request(
                    token,
                    "files/upload_session/finish",
                    {"cursor": cursor, "commit": commit},
                    chunk,
                )
            _content_request(token, "files/upload_session/append_v2", {"cursor": cursor, "close": False}, chunk)
            offset += len(chunk)


def _temporary_download_link(token: str, destination: str) -> str:
    # AICODE-NOTE: An upload is incomplete for agent use until Dropbox returns
    # a direct temporary link that can be included in the user-facing reply.
    result = _rpc(token, "files/get_temporary_link", {"path": destination})
    link = result.get("link")
    if not isinstance(link, str) or not link:
        raise DropboxError(f"Dropbox did not return a temporary download link for {destination}")
    return link


def upload_files(paths: list[Path]) -> list[dict[str, Any]]:
    sources = [path.expanduser().resolve() for path in paths]
    invalid = [str(path) for path in sources if not path.is_file()]
    if invalid:
        raise DropboxError(f"not a regular file: {', '.join(invalid)}")

    token = _access_token()
    date_folder = datetime.now().astimezone().date().isoformat()
    remote_folder = f"{DROPBOX_ROOT}/{date_folder}"
    _ensure_folder(token, DROPBOX_ROOT)
    _ensure_folder(token, remote_folder)

    results = []
    for source in sources:
        destination = f"{remote_folder}/{source.name}"
        metadata = _upload(token, source, destination)
        verified = metadata.get("content_hash") == _content_hash(source)
        if not verified:
            raise DropboxError(f"content-hash mismatch after uploading {source}")
        temporary_download_link = _temporary_download_link(token, destination)
        result = {
            "source": str(source),
            "path": destination,
            "size": metadata.get("size"),
            "verified": True,
            "temporary_download_link": temporary_download_link,
        }
        results.append(result)
        print(f"Uploaded {source} -> {destination} ({metadata.get('size')} bytes, integrity OK)")
        print(f"Temporary direct-download link (~4h): {temporary_download_link}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload files to Dropbox at /rsrch/YYYY-MM-DD/<filename>.",
    )
    parser.add_argument("files", nargs="*", type=Path, help="one or more local files")
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="validate credentials without uploading anything",
    )
    args = parser.parse_args()
    if not args.files and not args.check_auth:
        parser.error("provide at least one file or use --check-auth")

    try:
        if args.check_auth:
            _rpc(_access_token(), "users/get_current_account", {})
            print("Dropbox credentials are valid.")
        if args.files:
            upload_files(args.files)
    except DropboxError as exc:
        print(f"Dropbox upload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
