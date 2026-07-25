"""Shared integrity checks for the historical 96-image validation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "cosmic_full96_auto_v1"
MANUAL_SHA256 = "a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614"
AUTO_SEED_SHA256 = "f93e04ecdc0283a54837a18efe0f4a99c913594ece68e15be0a47d2321189dfe"
EXPECTED_MANUAL_ENTRIES = 96
EXPECTED_AUTOMATIC_ENTRIES = 95
EXPECTED_FORCE_MANUAL_KEYS = {"Reading pa_jensen.png"}
SUPPORTED_REFERENCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

STATIC_FILE_SHA256 = {
    "prompts_10.txt": "e8fb3290e6da6eacc70c6cc67f2affa0c923c1ca605efc35ddca95ee48f1ebaf",
    "classes_ref.json": "d1f53322d6964c2d30d28ef2cc765366a42776117e3982909d6fdfc1ae99872b",
    "ref_bboxes.json": "eadb9411b9d0b98238714bb263db708e56a30abee91c67c4df0c7e1e5c4a268f",
    "id_embeds_manual_val.pth": "23ae97075e967f2bcb790c5094ef350b316249c7023df67a68f735bfebb747c6",
}
REFERENCE_SHA256 = {
    "eddie.webp": "488c1ba267c3bada5aed1d72bf5b569b5be6ce7fb9050554559f307155cdcb8e",
    "elon.jpg": "6e68491ee0f393df834ff9570dd15eaa01fb5f8805f6fce3f075818a7ea02381",
    "jennie.webp": "ce286f8242cb1f702b0289ceaa20d67cd4ac1ffd8b8a909658ff6648a0129c81",
    "jensen.png": "2f540b82ece53e4f3f4862a72fb2fbefd67854dbb9aa2d33b8183d322a50831a",
    "jisoo.webp": "62c380c9b5ec08ec8b1fe613a390ff18b0f16497a23ebbfd1459ff887988e806",
    "keanu.jpg": "750d34d29d14fc8875bbebecff56c1fbd32fa642e3a1a6454fd6f79c489531c3",
    "lex.jpeg": "cb0fc3ea4ffad8973b5e5eef8ffac84b84f19b467d786f9cadc0b0aeb7254d15",
    "marion.jpg": "3884de5c8ca4c97840512c4976daa3cc79bb9e33eef4369c9b6ec93aed3f5a22",
    "michael.jpg": "aebeb74d7df036204ad077fea58b01e89d488d5005bcc3afc8dd673568b7d0e3",
    "robert.webp": "1496154a4d55749521b9e09b4b14c9294af0a555218a017a97264b574976ca5d",
    "sydney.jpg": "114f74d1728558b2488cb30c3e2a7b13ded13885285f95efe9b902706f145402",
    "tom.jpg": "dff3797d55eccaf1e9b72289a4f0d126ff3aee2cc79442dcb4f15124000bd5a6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_bbox_entries(entries: dict[str, Any], label: str) -> None:
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError(f"{label} contains an invalid record at {key!r}")
        bbox = value.get("face_crop_new") or value.get("face_crop_old")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bbox)
        ):
            raise ValueError(f"{label} record {key!r} has no valid face bbox")


def validate_bbox_routing(
    manual: dict[str, Any],
    automatic: dict[str, Any],
    *,
    auto_min: int = 12,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate 95 automatic routes plus the sealed Jensen manual override."""
    if len(manual) != EXPECTED_MANUAL_ENTRIES:
        raise ValueError(
            f"Manual bbox protocol has {len(manual)} entries, "
            f"expected {EXPECTED_MANUAL_ENTRIES}"
        )
    _validate_bbox_entries(manual, "Manual bbox protocol")
    _validate_bbox_entries(automatic, "Automatic bbox cache")

    force_manual_keys = {
        key
        for key, value in manual.items()
        if isinstance(value, dict) and bool(value.get("force_manual"))
    }
    if force_manual_keys != EXPECTED_FORCE_MANUAL_KEYS:
        raise ValueError(
            "Unexpected force-manual key set: "
            f"{sorted(force_manual_keys)}"
        )

    manual_keys = set(manual)
    expected_auto_keys = manual_keys - force_manual_keys
    automatic_keys = set(automatic)
    unexpected_auto = automatic_keys - expected_auto_keys
    if unexpected_auto:
        raise ValueError(
            "Automatic bbox cache contains unknown or force-manual keys: "
            f"{sorted(unexpected_auto)}"
        )
    if not auto_min <= len(automatic_keys) <= EXPECTED_AUTOMATIC_ENTRIES:
        raise ValueError(
            f"Automatic bbox cache has {len(automatic_keys)} entries; expected "
            f"{auto_min}..{EXPECTED_AUTOMATIC_ENTRIES}"
        )

    missing_auto = expected_auto_keys - automatic_keys
    complete = not missing_auto
    if require_complete and not complete:
        raise ValueError(
            "Full-96 routing is incomplete; missing automatic keys: "
            f"{sorted(missing_auto)}"
        )

    return {
        "manual_entries": len(manual_keys),
        "automatic_entries": len(automatic_keys),
        "force_manual_entries": len(force_manual_keys),
        "routing_entries": len(automatic_keys | force_manual_keys),
        "force_manual_keys": sorted(force_manual_keys),
        "missing_automatic_keys": sorted(missing_auto),
        "complete": complete,
    }


def load_bbox_protocol(
    manual_path: Path,
    *,
    auto_min: int = 12,
    require_complete: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    manual_path = manual_path.resolve()
    if sha256(manual_path) != MANUAL_SHA256:
        raise ValueError("Manual bbox protocol failed its SHA-256 seal")
    manual = load_object(manual_path)
    auto_path = manual_path.with_name(f"{manual_path.stem}_auto.json")
    automatic = load_object(auto_path)
    status = validate_bbox_routing(
        manual,
        automatic,
        auto_min=auto_min,
        require_complete=require_complete,
    )
    return manual, automatic, auto_path, status


def validate_static_inputs(validation_data_dir: Path) -> dict[str, Any]:
    """Seal prompts, identity metadata, metrics input, and reference ordering."""
    validation_data_dir = validation_data_dir.resolve()
    static_hashes: dict[str, str] = {}
    for name, expected in STATIC_FILE_SHA256.items():
        path = validation_data_dir / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Static validation input changed at {path}: "
                f"expected {expected}, found {actual}"
            )
        static_hashes[name] = actual

    references_dir = validation_data_dir / "references"
    actual_names = sorted(
        path.name
        for path in references_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_REFERENCE_SUFFIXES
    )
    expected_names = sorted(REFERENCE_SHA256)
    if actual_names != expected_names:
        raise ValueError(
            "Validation reference filenames changed: "
            f"expected {expected_names}, found {actual_names}"
        )

    reference_hashes: dict[str, str] = {}
    for name in expected_names:
        path = references_dir / name
        actual = sha256(path)
        expected = REFERENCE_SHA256[name]
        if actual != expected:
            raise ValueError(
                f"Validation reference changed at {path}: "
                f"expected {expected}, found {actual}"
            )
        reference_hashes[name] = actual

    return {
        "validation_data_dir": str(validation_data_dir),
        "static_file_sha256": static_hashes,
        "reference_sha256": reference_hashes,
        "first_eight_references": actual_names[:8],
    }
