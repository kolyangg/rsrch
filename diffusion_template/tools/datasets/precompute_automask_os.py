#!/usr/bin/env python3
"""Build fail-closed AutoMask-OS caches from an explicit transformed-image recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
import torch

from src.model.photomaker_branched.masking.automask_os import (
    BISEnet_CLASSES, POLICY_VERSION, PinnedAutoMaskBuilder, image_sha256,
)
from src.model.photomaker_branched.masking.ownership_maps import (
    cache_key, save_ownership_maps,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_hash(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path,
                        help="JSON list of transformed image/reference/cache_identity records")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-version", default=POLICY_VERSION)
    parser.add_argument("--fail-on-missing-face", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-normalization", action="store_true")
    args = parser.parse_args()
    recipe_payload = json.loads(args.recipe.read_text(encoding="utf-8"))
    records = (
        recipe_payload.get("records")
        if isinstance(recipe_payload, dict)
        else recipe_payload
    )
    if not isinstance(records, list) or not records:
        raise ValueError("AutoMask-OS recipe must be a non-empty JSON list")
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    builder = PinnedAutoMaskBuilder(
        device=device, policy_version=args.policy_version,
    )
    parsing_model, analyzer = builder.parser_model, builder.analyzer
    parser_hash = _module_hash(parsing_model)
    insightface_hashes = {}
    for name, model in sorted(getattr(analyzer, "models", {}).items()):
        path = getattr(model, "model_file", None)
        if path and Path(path).is_file():
            insightface_hashes[name] = _sha256_file(path)
    model_fingerprint = hashlib.sha256(json.dumps({
        "parser": parser_hash, "insightface": insightface_hashes,
    }, sort_keys=True).encode()).hexdigest()
    failures, written, cache_index, image_hashes = [], [], {}, {}
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        body_crop = record.get("body_crop")
        if image.size != (1024, 1024) and body_crop is not None:
            left, top, right, bottom = (int(value) for value in body_crop)
            image = image.crop((left, top, right, bottom))
            if image.size != (1024, 1024):
                raise ValueError(f"body_crop did not produce 1024x1024: {record['image_path']}")
        reference = Image.open(record["reference_image_path"]).convert("RGB")
        try:
            maps = builder.build(
                image, reference,
                expected_location=record.get("expected_location"),
            )
        except RuntimeError as error:
            failures.append({"record": record, "reason": str(error)})
            if args.fail_on_missing_face:
                raise
            continue
        if args.verify_normalization:
            error = float((maps.probabilities.sum(0) - 1.0).abs().max())
            if error > 1.0e-4:
                raise RuntimeError(f"Ownership normalization failed: {error}")
        base_identity = dict(record["cache_identity"])
        base_identity["policy"] = args.policy_version
        reference_path = str(Path(record["reference_image_path"]).absolute())
        if reference_path not in image_hashes:
            image_hashes[reference_path] = image_sha256(reference)
        identity = {
            **base_identity,
            "image_sha256": maps.source_hash,
            "reference_sha256": image_hashes[reference_path],
            "output_size": list(image.size),
            "model_fingerprint": model_fingerprint,
        }
        filename = f"{cache_key(identity)}.npz"
        output = args.output_root / filename
        save_ownership_maps(output, maps, {"cache_identity": identity})
        written.append(output.name)
        cache_index[cache_key(base_identity)] = {
            "filename": filename, "cache_identity": identity,
        }
    manifest = {
        "policy_version": args.policy_version,
        "recipe_kind": recipe_payload.get("kind") if isinstance(recipe_payload, dict) else None,
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "preview_protocol_sha256": (
            recipe_payload.get("preview_protocol_sha256")
            if isinstance(recipe_payload, dict)
            else None
        ),
        "parser": "facexlib_bisenet_19class",
        "class_ids": {key: list(value) for key, value in BISEnet_CLASSES.items()},
        "subject_selection": {
            "identity_weight": 0.70,
            "location_weight": 0.20,
            "detector_weight": 0.10,
            "score_threshold": 0.35,
            "margin_threshold": 0.05,
        },
        "parser_state_sha256": parser_hash,
        "insightface_model_sha256": insightface_hashes,
        "model_fingerprint": model_fingerprint,
        "complete": not failures and len(cache_index) == len(records),
        "written": written,
        "cache_index": cache_index,
        "failures": failures,
    }
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    (args.output_root / "manifest.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({"written": len(written), "failures": len(failures),
                      "manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
