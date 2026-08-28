"""Sealed recipe identities for CL39-X05's automatic two-pass validation."""

from __future__ import annotations

import hashlib
import json


PREVIEW_PROTOCOL = {
    "version": "cl39x05_preview_v1",
    "parent": "CL39_cosmic_null_key_confidence_router_24k",
    "branched_attention": False,
    "resolution": [1024, 1024],
    "inference_steps": 50,
    "scheduler": "unchanged_manual_val96",
    "prompt_reference_seed": "identical_to_final_pass",
}
PREVIEW_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PREVIEW_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def validation_reference_identity(
    sample_id: str, policy: str, reference_sha256: str,
) -> dict:
    return {
        "kind": "validation_reference", "id": str(sample_id),
        "policy": str(policy), "reference_sha256": str(reference_sha256),
    }


def validation_target_identity(
    sample_id: str, prompt: str, seed: int, policy: str, reference_sha256: str,
) -> dict:
    return {
        "kind": "validation_target",
        "id": str(sample_id),
        "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
        "reference_sha256": str(reference_sha256),
        "seed": int(seed),
        "policy": str(policy),
        "preview_protocol_sha256": PREVIEW_PROTOCOL_SHA256,
    }


def recipe_record(*, image_path, reference_image_path, cache_identity, expected_location=None):
    """Return one explicit precompute record; no detector-order fallback is encoded."""
    value = {
        "image_path": str(image_path),
        "reference_image_path": str(reference_image_path),
        "cache_identity": dict(cache_identity),
    }
    if expected_location is not None:
        value["expected_location"] = [float(item) for item in expected_location]
    return value
