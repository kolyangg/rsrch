#!/usr/bin/env python3
"""Run one audited CL9 fixed-checkpoint validation intervention."""

from __future__ import annotations

import argparse
from argparse import Namespace
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def load_evaluator(path: Path, project_root: Path):
    os.environ["PM_EVAL_PROJECT_ROOT"] = str(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("cl9_edge_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluator overlay: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_visibility_compatibility(evaluator) -> None:
    """Bridge the sidecar mask into the immutable CL9 pipeline at runtime."""
    original_build = evaluator.build_pipeline

    def wrapped_build(*args, **kwargs):
        pipeline = original_build(*args, **kwargs)
        pipeline_module = importlib.import_module(pipeline.__class__.__module__)
        original_call = pipeline.__class__.__call__
        original_setup = pipeline_module.run_branched_setup_helper

        def wrapped_call(
            instance,
            *call_args,
            ba_target_visibility_mask=None,
            **call_kwargs,
        ):
            instance._cl9_sidecar_target_visibility = ba_target_visibility_mask
            try:
                return original_call(instance, *call_args, **call_kwargs)
            finally:
                instance._cl9_sidecar_target_visibility = None

        def wrapped_setup(instance, *setup_args, **setup_kwargs):
            original_setup(instance, *setup_args, **setup_kwargs)
            visibility = getattr(
                instance, "_cl9_sidecar_target_visibility", None
            )
            if visibility is None:
                return
            base = np.asarray(instance._face_mask, dtype=np.float32)
            value = np.asarray(visibility, dtype=np.float32)
            if value.ndim == 4 and value.shape[1] == 1:
                value = value[:, 0]
            if value.ndim == 2:
                value = value[None]
            if base.ndim == 2:
                base = base[None]
            if base.shape != value.shape:
                raise RuntimeError(
                    "Sidecar visibility/base-mask shape mismatch: "
                    f"{value.shape} vs {base.shape}"
                )
            if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError("Target visibility values must be finite in [0, 1]")
            masked = base * value
            # 10 Aug 2026 - AICODE-NOTE: This compatibility hook only removes
            # target queries from the face lane in the immutable CL9 runtime.
            # The original reference mask, K/V tensors, and weights are not
            # replaced, and a no-mask replay never enters this branch.
            instance._face_mask = masked if np.asarray(instance._face_mask).ndim == 3 else masked[0]
            instance._face_mask_t = torch.from_numpy(masked.astype(np.float32))[:, None]

        pipeline.__class__.__call__ = wrapped_call
        pipeline_module.run_branched_setup_helper = wrapped_setup
        return pipeline

    evaluator.build_pipeline = wrapped_build


def install_reference_transform(evaluator, manifest_path: Path, variant: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported Marion transform manifest: {manifest_path}")
    entry = payload.get("variants", {}).get(variant)
    if not isinstance(entry, dict):
        raise KeyError(f"Transform variant {variant!r} is absent from {manifest_path}")
    image_path = Path(entry["image_path"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    bbox = [float(value) for value in entry["propagated_bbox"]]
    transformed = Image.open(image_path).convert("RGB")
    original_apply = evaluator.apply_reference_condition

    def transformed_apply(eval_args, samples):
        indices = [int(sample.get("_dataset_index", -1)) for sample in samples]
        identities = {str(sample.get("id")).lower() for sample in samples}
        if indices != list(range(84, 96)) or identities != {"marion"}:
            raise ValueError(
                "Marion transform requires exact manual_val indices 84-95 "
                "as one batch"
            )
        conditioned = []
        for sample in samples:
            copy = dict(sample)
            copy["ref_images"] = [transformed.copy()]
            copy["face_bbox_ref"] = list(bbox)
            conditioned.append(copy)
        refs, bboxes, embeds, intervention = original_apply(eval_args, conditioned)
        intervention = dict(intervention)
        intervention.update(
            {
                "kind": f"marion_{variant}",
                "same_source_photograph": True,
                "conditioning_pixels_changed": True,
                "scoring_reference_changed": False,
                "transform_manifest": str(manifest_path.resolve()),
                "transform_manifest_sha256": evaluator.sha256_file(manifest_path),
                "transformed_image": str(image_path),
                "transformed_image_sha256": evaluator.sha256_file(image_path),
                "propagated_bbox": bbox,
                "redetected_bbox": entry.get("redetected_bbox"),
            }
        )
        return refs, bboxes, embeds, intervention

    evaluator.apply_reference_condition = transformed_apply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["baseline", "marion_roll", "marion_similarity", "occlusion_ownership"],
        required=True,
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=24000)
    parser.add_argument("--generation-bbox-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-indices")
    parser.add_argument("--reference-transform-manifest", type=Path)
    parser.add_argument("--target-visibility-plan", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    evaluator = load_evaluator(args.evaluator.resolve(), project_root)
    install_visibility_compatibility(evaluator)

    if args.variant in {"marion_roll", "marion_similarity"}:
        if args.reference_transform_manifest is None:
            raise ValueError("Marion variants require --reference-transform-manifest")
        install_reference_transform(
            evaluator,
            args.reference_transform_manifest.resolve(),
            args.variant.removeprefix("marion_"),
        )
    if args.variant == "occlusion_ownership" and args.target_visibility_plan is None:
        raise ValueError("Occlusion ownership requires --target-visibility-plan")
    if args.variant != "occlusion_ownership" and args.target_visibility_plan is not None:
        raise ValueError("Only occlusion_ownership may set a visibility plan")

    run_args = Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        validation_dataset="manual_val",
        guidance_scale=None,
        disable_branched_ca=None,
        validation_base=None,
        photomaker_path=None,
        processor_base_mode=None,
        reference_condition="matched",
        spatial_reference_condition="matched",
        ba_mix_override=None,
        limit=96 if args.sample_indices is None else 12,
        sample_indices=args.sample_indices,
        batch_size=None,
        checkpoint_step=args.checkpoint_step,
        reference_id_embedding_policy="legacy_first",
        generation_bbox_map=args.generation_bbox_map,
        target_visibility_plan=args.target_visibility_plan,
        wrong_reference=None,
        wrong_reference_bbox=None,
        device="cuda",
        skip_metrics=True,
        allow_untrained_ca=False,
        allow_validation_contract_override=False,
    )
    evaluator.run(run_args)


if __name__ == "__main__":
    main()
