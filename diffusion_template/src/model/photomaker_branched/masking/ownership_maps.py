"""Serialization and routing policy for AutoMask-OS probability maps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ("visible_face", "hair_head", "accessory", "occluder", "uncertain", "background")


@dataclass(frozen=True)
class OwnershipMaps:
    probabilities: torch.Tensor  # [6,H,W]
    confidence: torch.Tensor
    selected_bbox: tuple[float, float, float, float]
    subject_score: float
    subject_margin: float
    policy_version: str
    source_hash: str

    def validate(self, tolerance: float = 1.0e-4) -> "OwnershipMaps":
        if self.probabilities.ndim != 3 or self.probabilities.shape[0] != len(CLASS_NAMES):
            raise ValueError("Ownership probabilities must be [6,H,W]")
        if not torch.isfinite(self.probabilities).all() or self.probabilities.min() < 0:
            raise ValueError("Ownership probabilities must be finite and nonnegative")
        error = (self.probabilities.sum(0) - 1.0).abs().max()
        if float(error) > tolerance:
            raise ValueError(f"Ownership probabilities are not normalized (max error {error})")
        return self


def resize_probabilities(probabilities: torch.Tensor, size, *, dtype, device) -> torch.Tensor:
    values = F.interpolate(probabilities[None].float(), size=size, mode="bilinear", align_corners=False)[0]
    values = values.clamp_min(0)
    values = values / values.sum(0, keepdim=True).clamp_min(1.0e-8)
    return values.to(device=device, dtype=dtype)


def routing_masks(probabilities: torch.Tensor, hair_weight: float = 0.35):
    visible, hair, _accessory, occluder, uncertain, _background = probabilities.unbind(1)
    target = (visible * (1.0 - occluder)).unsqueeze(1)
    reference = (visible + float(hair_weight) * hair).clamp(0, 1).unsqueeze(1)
    top = (occluder + uncertain * occluder).clamp(0, 1).unsqueeze(1)
    return target, reference, top


def cache_key(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cache_index(root: Path, expected_policy: str) -> dict:
    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("policy_version") != expected_policy
        or not manifest.get("complete")
        or manifest.get("failures")
        or not isinstance(manifest.get("cache_index"), dict)
    ):
        raise ValueError(f"Incomplete or incompatible ownership manifest: {manifest_path}")
    return manifest["cache_index"]


def load_indexed_ownership_maps(
    root: Path, base_identity: dict, expected_policy: str, cache_index: dict
) -> OwnershipMaps:
    entry = cache_index.get(cache_key(base_identity))
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"Missing ownership cache identity: {base_identity}")
    effective_identity = entry.get("cache_identity")
    filename = entry.get("filename")
    if not isinstance(effective_identity, dict) or filename != f"{cache_key(effective_identity)}.npz":
        raise ValueError(f"Invalid ownership cache index entry: {entry}")
    return load_ownership_maps(
        Path(root) / filename, expected_policy, effective_identity
    )


def save_ownership_maps(path: Path, maps: OwnershipMaps, metadata: dict) -> None:
    maps.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=maps.probabilities.cpu().numpy().astype(np.float16),
        confidence=maps.confidence.cpu().numpy().astype(np.float16),
        metadata=np.asarray(json.dumps({**metadata, "bbox": maps.selected_bbox,
                                        "subject_score": maps.subject_score,
                                        "subject_margin": maps.subject_margin,
                                        "policy_version": maps.policy_version,
                                        "source_hash": maps.source_hash}, sort_keys=True)),
    )


def load_ownership_maps(
    path: Path, expected_policy: str, expected_identity: dict | None = None
) -> OwnershipMaps:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("policy_version") != expected_policy:
            raise ValueError(f"Ownership cache policy mismatch in {path}")
        if expected_identity is not None and metadata.get("cache_identity") != expected_identity:
            raise ValueError(f"Ownership cache identity mismatch in {path}")
        probabilities = torch.from_numpy(
            archive["probabilities"].astype(np.float32)
        ).clamp_min(0)
        probabilities = probabilities / probabilities.sum(
            0, keepdim=True
        ).clamp_min(1.0e-8)
        maps = OwnershipMaps(
            probabilities=probabilities,
            confidence=torch.from_numpy(archive["confidence"].astype(np.float32)),
            selected_bbox=tuple(float(v) for v in metadata["bbox"]),
            subject_score=float(metadata["subject_score"]),
            subject_margin=float(metadata["subject_margin"]),
            policy_version=str(metadata["policy_version"]),
            source_hash=str(metadata["source_hash"]),
        )
    return maps.validate()
