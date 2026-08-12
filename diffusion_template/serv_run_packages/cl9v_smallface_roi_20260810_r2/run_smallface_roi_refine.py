#!/usr/bin/env python3
"""Deterministic late-denoise face-ROI refinement for CL9 small-face rows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch


SMALL_FACE_INDICES = (5, 9, 17, 21, 29, 33, 41, 45, 53, 57, 65, 69, 77, 81, 89, 93)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_evaluator(path: Path, project_root: Path):
    os.environ["PM_EVAL_PROJECT_ROOT"] = str(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("cl9_roi_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expanded_crop(bbox, width: int, height: int, ratio: float):
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    crop_w = (x1 - x0) * ratio
    crop_h = (y1 - y0) * ratio
    ix0 = max(0, int(math.floor(cx - crop_w / 2.0)))
    iy0 = max(0, int(math.floor(cy - crop_h / 2.0)))
    ix1 = min(width, int(math.ceil(cx + crop_w / 2.0)))
    iy1 = min(height, int(math.ceil(cy + crop_h / 2.0)))
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError(f"Invalid expanded crop for bbox {bbox}")
    return [ix0, iy0, ix1, iy1]


def round_up_8(value: float) -> int:
    return max(8, int(math.ceil(value / 8.0)) * 8)


def remap_bbox(bbox, crop_box, work_width: int, work_height: int):
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx0, cy0, cx1, cy1 = crop_box
    sx = work_width / float(cx1 - cx0)
    sy = work_height / float(cy1 - cy0)
    return [
        (x0 - cx0) * sx,
        (y0 - cy0) * sy,
        (x1 - cx0) * sx,
        (y1 - cy0) * sy,
    ]


def cosine_alpha(width: int, height: int, feather_fraction: float) -> np.ndarray:
    feather = max(1.0, min(width, height) * feather_fraction)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    distance = np.minimum(y[:, None], x[None, :])
    phase = np.clip(distance / feather, 0.0, 1.0) * (math.pi / 2.0)
    return np.sin(phase) ** 2


@contextmanager
def prepared_latents(pipeline, value: torch.Tensor):
    original = pipeline.prepare_latents

    def fixed(*args, **kwargs):
        del args, kwargs
        return value

    pipeline.prepare_latents = fixed
    try:
        yield
    finally:
        pipeline.prepare_latents = original


@contextmanager
def standard_ddim_suffix(pipeline, total_steps: int, expected_timesteps: list[int]):
    """Use a suffix of the standard DDIM grid without requesting a custom grid."""
    pipeline_module = importlib.import_module(pipeline.__class__.__module__)
    original = pipeline_module.retrieve_timesteps
    expected = [int(value) for value in expected_timesteps]

    def retrieve(
        scheduler,
        num_inference_steps=None,
        device=None,
        timesteps=None,
        sigmas=None,
        **kwargs,
    ):
        if timesteps is None:
            return original(
                scheduler,
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                **kwargs,
            )
        if sigmas is not None:
            raise ValueError("ROI suffix cannot combine timesteps and sigmas")
        requested = [int(value) for value in timesteps]
        if requested != expected:
            raise ValueError(
                f"Unexpected ROI timestep suffix: {requested} != {expected}"
            )
        # 10 Aug 2026 - AICODE-NOTE: The immutable DDIMScheduler cannot accept
        # a custom list. Initialize its unchanged 50-step contract, then expose
        # only the verified late suffix while retaining num_inference_steps=50
        # inside scheduler.step for the correct previous-timestep calculation.
        scheduler.set_timesteps(total_steps, device=device, **kwargs)
        suffix = scheduler.timesteps[-len(expected) :]
        actual = [int(value) for value in suffix.tolist()]
        if actual != expected:
            raise RuntimeError(f"DDIM suffix drift: {actual} != {expected}")
        return suffix, len(suffix)

    pipeline_module.retrieve_timesteps = retrieve
    try:
        yield
    finally:
        pipeline_module.retrieve_timesteps = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-scale", type=float, default=2.0)
    parser.add_argument("--bbox-expansion", type=float, default=1.5)
    parser.add_argument("--late-steps", type=int, required=True)
    parser.add_argument("--feather-fraction", type=float, default=0.12)
    args = parser.parse_args()

    if args.roi_scale < 1.0 or args.bbox_expansion < 1.0:
        raise ValueError("ROI scale and bbox expansion must be >=1")
    if not 1 <= args.late_steps < 50:
        raise ValueError("--late-steps must be in [1, 49]")
    if not 0.0 < args.feather_fraction <= 0.5:
        raise ValueError("--feather-fraction must be in (0, 0.5]")

    project_root = args.project_root.resolve()
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    images_dir = output_dir / "images"
    roi_dir = output_dir / "roi_debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(exist_ok=True)

    evaluator = load_evaluator(args.evaluator.resolve(), project_root)
    config, config_source = evaluator.load_config(args.config)
    config.val_datasets_names = ["manual_val"]
    validation_base = str(
        getattr(
            config,
            "pretrained_model_for_validation_name_or_path",
            config.model.pretrained_model_name_or_path,
        )
    )
    processor_mode = evaluator.configured_processor_base_mode(config)
    disable_branched_ca = bool(getattr(config, "disable_branched_ca", False))
    state, checkpoint_metadata = evaluator.checkpoint_state(args.checkpoint.resolve())
    device = torch.device("cuda")
    model, processor_audit = evaluator.load_evaluation_model(
        config,
        state,
        validation_base=validation_base,
        processor_base_mode=processor_mode,
        device=device,
        disable_branched_ca=disable_branched_ca,
    )
    pipeline = evaluator.build_pipeline(
        config,
        model,
        validation_base=validation_base,
        device=device,
        disable_branched_ca=disable_branched_ca,
    )
    setattr(pipeline, "ba_mix_override", None)

    dataset = instantiate(config.datasets.val.manual_val)
    if len(dataset) != 96:
        raise ValueError(f"Expected full96 manual_val, found {len(dataset)}")
    baseline_rows = json.loads((baseline_dir / "per_image.json").read_text(encoding="utf-8"))
    by_index = {int(row["dataset_index"]): row for row in baseline_rows}
    if sorted(by_index) != list(range(96)):
        raise ValueError("Baseline sidecar must contain every manual_val index")

    output_rows = []
    for index in range(96):
        row = dict(by_index[index])
        source = baseline_dir / "images" / row["filename"]
        destination = images_dir / row["filename"]
        shutil.copy2(source, destination)
        row["roi_refinement"] = {"modified": False}
        output_rows.append(row)

    validation_kwargs = OmegaConf.to_container(config.validation_args, resolve=True)
    validation_kwargs["debug_dir"] = None
    validation_kwargs["debug_idx"] = 0
    validation_kwargs["debug_total"] = 96
    validation_kwargs["val_debug"] = False
    validation_kwargs.pop("num_inference_steps", None)

    refinement_rows = []
    for ordinal, index in enumerate(SMALL_FACE_INDICES):
        sample = dataset[index]
        row = output_rows[index]
        source_path = baseline_dir / "images" / row["filename"]
        with Image.open(source_path) as opened:
            baseline = opened.convert("RGB")
        baseline_np = np.asarray(baseline).copy()
        crop_box = expanded_crop(
            row["face_bbox_gen"], baseline.width, baseline.height, args.bbox_expansion
        )
        cx0, cy0, cx1, cy1 = crop_box
        crop = baseline.crop(tuple(crop_box))
        work_width = round_up_8(max(256.0, crop.width * args.roi_scale))
        work_height = round_up_8(max(256.0, crop.height * args.roi_scale))
        work_image = crop.resize((work_width, work_height), Image.Resampling.LANCZOS)
        work_bbox = remap_bbox(
            row["face_bbox_gen"], crop_box, work_width, work_height
        )

        pixels = pipeline.image_processor.preprocess(
            work_image, height=work_height, width=work_width
        ).to(device=device, dtype=pipeline.unet.dtype)
        with torch.no_grad():
            clean_latent = (
                pipeline.vae.encode(pixels).latent_dist.mode()
                * pipeline.vae.config.scaling_factor
            )
        pipeline.scheduler.set_timesteps(50, device=device)
        all_timesteps = pipeline.scheduler.timesteps.detach().clone()
        start_index = 50 - args.late_steps
        start_timestep = all_timesteps[start_index]
        custom_timesteps = [int(value) for value in all_timesteps[start_index:].tolist()]
        seed = int(sample.get("seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=device,
            dtype=clean_latent.dtype,
        )
        noised_latent = pipeline.scheduler.add_noise(
            clean_latent,
            noise,
            start_timestep.reshape(1),
        )

        call_kwargs = dict(validation_kwargs)
        call_kwargs.update(
            {
                "height": work_height,
                "width": work_width,
                "target_size": (work_height, work_width),
                "original_size": (work_height, work_width),
                "crops_coords_top_left": (0, 0),
                "timesteps": custom_timesteps,
            }
        )
        with prepared_latents(pipeline, noised_latent), standard_ddim_suffix(
            pipeline, total_steps=50, expected_timesteps=custom_timesteps
        ):
            with torch.no_grad():
                refined = pipeline(
                    prompt=str(sample["prompt"]),
                    input_id_images=sample["ref_images"],
                    face_bbox_ref=sample.get("face_bbox_ref"),
                    face_bbox_gen=work_bbox,
                    generator=generator,
                    id_embeds=None,
                    **call_kwargs,
                ).images[0]

        refined_small = refined.resize(crop.size, Image.Resampling.LANCZOS)
        alpha = cosine_alpha(crop.width, crop.height, args.feather_fraction)
        refined_np = np.asarray(refined_small, dtype=np.float32)
        original_crop_np = baseline_np[cy0:cy1, cx0:cx1].astype(np.float32)
        composite_crop = np.rint(
            original_crop_np * (1.0 - alpha[:, :, None])
            + refined_np * alpha[:, :, None]
        ).clip(0, 255).astype(np.uint8)
        composite = baseline_np.copy()
        composite[cy0:cy1, cx0:cx1] = composite_crop
        outside = np.ones((baseline.height, baseline.width), dtype=bool)
        outside[cy0:cy1, cx0:cx1] = False
        outside_exact = bool(np.array_equal(composite[outside], baseline_np[outside]))
        if not outside_exact:
            raise RuntimeError(f"Outside-ROI pixels changed for dataset index {index}")

        destination = images_dir / row["filename"]
        Image.fromarray(composite).save(destination)
        raw_path = roi_dir / f"{index:03d}_refined.png"
        alpha_path = roi_dir / f"{index:03d}_alpha.png"
        refined.save(raw_path)
        Image.fromarray(np.rint(alpha * 255).astype(np.uint8), mode="L").save(alpha_path)
        boundary = alpha < 0.25
        boundary_delta = float(
            np.abs(composite_crop.astype(np.float32) - original_crop_np)[boundary].mean()
        )
        details = {
            "modified": True,
            "ordinal": ordinal,
            "crop_box": crop_box,
            "working_size": [work_width, work_height],
            "working_face_bbox": work_bbox,
            "roi_scale_requested": args.roi_scale,
            "bbox_expansion": args.bbox_expansion,
            "late_steps": args.late_steps,
            "timestep_contract": "late suffix of the unchanged standard DDIM50 grid",
            "start_timestep": int(start_timestep),
            "seed": seed,
            "feather_fraction": args.feather_fraction,
            "outside_roi_pixel_exact": outside_exact,
            "boundary_mean_abs_delta": boundary_delta,
            "baseline_rgb_sha256": hashlib.sha256(baseline_np.tobytes()).hexdigest(),
            "composite_rgb_sha256": hashlib.sha256(composite.tobytes()).hexdigest(),
            "raw_roi_path": str(raw_path),
            "alpha_path": str(alpha_path),
        }
        row["roi_refinement"] = details
        row["image_sha256"] = sha256_file(destination)
        refinement_rows.append({"dataset_index": index, **details})

    write_json(output_dir / "per_image.json", output_rows)
    face_quality_manifest = {
        "schema_version": 1,
        "kind": "cl9_smallface_roi_refinement",
        "experiment_key": None,
        "project_name": "rhca_fixed_checkpoint_diagnostics",
        "steps": {
            "24000": [
                {
                    "asset_id": f"local-{row['dataset_index']:03d}",
                    "file_name": row["filename"],
                    "local_path": str((images_dir / row["filename"]).resolve()),
                }
                for row in output_rows
            ]
        },
    }
    write_json(output_dir / "face_quality_input_manifest.json", face_quality_manifest)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "kind": "cl9_smallface_roi_refine_fixed_checkpoint",
            "config_source": config_source,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "checkpoint_metadata": checkpoint_metadata,
            "baseline_dir": str(baseline_dir),
            "baseline_manifest_sha256": sha256_file(baseline_dir / "run_manifest.json"),
            "validation_base": validation_base,
            "processor_base_mode": processor_mode,
            "disable_branched_ca": disable_branched_ca,
            "pose_adapt_ratio": float(config.pipeline.pose_adapt_ratio),
            "ca_mixing_for_face": bool(config.pipeline.ca_mixing_for_face),
            "full_panel_image_count": 96,
            "modified_indices": list(SMALL_FACE_INDICES),
            "sentinel_count": 80,
            "outside_roi_exact_count": sum(
                int(row["outside_roi_pixel_exact"]) for row in refinement_rows
            ),
            "roi_scale": args.roi_scale,
            "bbox_expansion": args.bbox_expansion,
            "late_steps": args.late_steps,
            "feather_fraction": args.feather_fraction,
            "processor_load_audit": processor_audit,
            "rows": refinement_rows,
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
