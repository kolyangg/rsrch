from __future__ import annotations

import glob
import os
from pathlib import Path


_EMPTY_VALUES = {"", "none", "null"}


def _normalize_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _EMPTY_VALUES:
        return None
    return Path(os.path.expandvars(os.path.expanduser(text)))


def resolve_photomaker_path(photomaker_path, *, version: str) -> str:
    version = (version or "").lower()
    if version == "v1":
        env_names = ("PM_V1_PATH", "PM_PATH", "PHOTOMAKER_PATH")
        cache_patterns = (
            "~/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/*/photomaker-v1.bin",
        )
    elif version == "v2":
        env_names = ("PM_V2_PATH", "PM_PATH", "PHOTOMAKER_PATH")
        cache_patterns = (
            "~/.cache/huggingface/hub/models--TencentARC--PhotoMaker-V2/snapshots/*/photomaker-v2.bin",
        )
    else:
        raise ValueError(f"Unknown PhotoMaker version: {version}")

    tried: list[str] = []

    direct_path = _normalize_path(photomaker_path)
    if direct_path is not None:
        if direct_path.is_file():
            return str(direct_path)
        tried.append(str(direct_path))

    for env_name in env_names:
        env_path = _normalize_path(os.environ.get(env_name))
        if env_path is None:
            continue
        if env_path.is_file():
            return str(env_path)
        tried.append(f"{env_name}={env_path}")

    for pattern in cache_patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        for match in matches:
            if Path(match).is_file():
                return match
        tried.append(os.path.expanduser(pattern))

    path_hint = (
        "model.photomaker_path was empty or invalid"
        if _normalize_path(photomaker_path) is None
        else f"model.photomaker_path={direct_path}"
    )
    tried_hint = "; ".join(tried) if tried else "no candidate paths"
    raise FileNotFoundError(
        f"PhotoMaker {version} checkpoint not found: {path_hint}. "
        f"Tried: {tried_hint}. "
        "Set model.photomaker_path to a valid .bin file or export PM_PATH."
    )
