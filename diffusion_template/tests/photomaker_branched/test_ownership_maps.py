import json
from pathlib import Path

import torch
from PIL import Image

from src.model.photomaker_branched.masking.automask_os import AutoMaskOS
from src.model.photomaker_branched.masking.ownership_maps import (
    OwnershipMaps,
    cache_key,
    load_cache_index,
    load_indexed_ownership_maps,
    load_ownership_maps,
    routing_masks,
    save_ownership_maps,
)


def test_ownership_round_trip_and_soft_routing(tmp_path: Path):
    probabilities = torch.rand(6, 8, 8)
    probabilities /= probabilities.sum(0, keepdim=True)
    maps = OwnershipMaps(
        probabilities=probabilities, confidence=1.0-probabilities[4],
        selected_bbox=(1, 1, 7, 7), subject_score=0.8, subject_margin=0.2,
        policy_version="automask_os_v1", source_hash="a" * 64,
    )
    path = tmp_path / "map.npz"
    save_ownership_maps(path, maps, {})
    loaded = load_ownership_maps(path, "automask_os_v1")
    torch.testing.assert_close(loaded.probabilities.sum(0), torch.ones(8, 8), atol=1e-4, rtol=0)
    target, reference, top = routing_masks(loaded.probabilities.unsqueeze(0))
    assert target.shape == reference.shape == top.shape == (1, 1, 8, 8)
    assert min(float(target.min()), float(reference.min()), float(top.min())) >= 0.0
    assert max(float(target.max()), float(reference.max()), float(top.max())) <= 1.0

    base = {"kind": "target", "path": "/fixed/image.png", "policy": "automask_os_v1"}
    effective = {**base, "image_sha256": "b" * 64, "model_fingerprint": "c" * 64}
    indexed_path = tmp_path / f"{cache_key(effective)}.npz"
    save_ownership_maps(indexed_path, maps, {"cache_identity": effective})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "policy_version": "automask_os_v1", "complete": True, "failures": [],
        "cache_index": {cache_key(base): {
            "filename": indexed_path.name, "cache_identity": effective,
        }},
    }))
    index = load_cache_index(tmp_path, "automask_os_v1")
    indexed = load_indexed_ownership_maps(tmp_path, base, "automask_os_v1", index)
    torch.testing.assert_close(indexed.probabilities.sum(0), torch.ones(8, 8), atol=1e-4, rtol=0)


def test_automask_policy_builds_normalized_subject_owned_maps():
    def detector(_image):
        return [{"embedding": torch.tensor([1.0, 0.0]), "bbox": [1, 1, 7, 7], "det_score": 1.0}]

    def parser(_image, _bbox):
        probabilities = torch.zeros(19, 8, 8)
        probabilities[1] = 1.0
        return probabilities, torch.ones(8, 8)

    maps = AutoMaskOS(
        detector, parser,
        {"visible_face": (1,), "hair_head": (), "accessory": ()},
    ).build(Image.new("RGB", (8, 8)), reference_embedding=torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(maps.probabilities.sum(0), torch.ones(8, 8))
    assert maps.subject_score > 0.99
