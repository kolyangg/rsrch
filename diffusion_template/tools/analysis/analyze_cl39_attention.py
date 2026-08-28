#!/usr/bin/env python3
"""Generate and render a reproducible CL39 attention/confidence audit.

Run from ``diffusion_template``. The generation path reconstructs the sealed
CL39 validation model: SDXL training base, RealVisXL validation base, schema-v2
trainables, and the historical strict full processor-state copy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity

try:
    from tools.analysis.cl39_attention_capture import (
        CL39AttentionCollector,
        attach_cl39_analysis,
    )
except ModuleNotFoundError:
    from cl39_attention_capture import CL39AttentionCollector, attach_cl39_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RUN_NAME = "CL39_cosmic_null_key_confidence_router_24k_full96_r4"
COMET_KEY = "b1ca0b3da679401c85b991f1bbdf0b2a"
CHECKPOINT_SHA256 = "74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints" / RUN_NAME
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cl39_attention_24k"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "analysis" / "assets" / "cl39_attention_24k"
DEFAULT_PM_PATH = Path(
    os.environ.get(
        "PM_PATH",
        Path.home()
        / ".cache/huggingface/hub/models--TencentARC--PhotoMaker-V2/snapshots"
        / "f5a1e5155dc02166253fa7e29d13519f5ba22eac/photomaker-v2.bin",
    )
)
SUBJECT_V2_EMBEDS = Path(
    os.environ.get(
        "SUBJECT_V2_ID_EMBEDS",
        PROJECT_ROOT.parent
        / "dataset_full/val_dataset/id_embeds_manual_val_subject_v2.pth",
    )
)
SEALED_CL39_IMAGE_DIR = (
    PROJECT_ROOT
    / "comet_data/23Aug_PM0_CL14_CL19_CL23_CL27_CL39_faces"
    / RUN_NAME
)
SEALED_CL39_ID_TABLE = (
    SEALED_CL39_IMAGE_DIR / "_tables/id_sim__manual_val__step_024000.csv"
)
ARMS = {
    "actual": {"confidence_override": None, "delta_scale": 1.0},
    "c1": {"confidence_override": 1.0, "delta_scale": 1.0},
    "ba_off": {"confidence_override": None, "delta_scale": 0.0},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed_config(checkpoint_dir: Path):
    config_path = checkpoint_dir / "config.yaml"
    record_path = checkpoint_dir / "comet_experiment.json"
    if not config_path.is_file() or not record_path.is_file():
        raise FileNotFoundError("The copied sealed config and Comet record are required")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    observed_key = (record.get("comet") or {}).get("experiment_key")
    if observed_key != COMET_KEY:
        raise RuntimeError(f"Unexpected Comet key: {observed_key!r}")
    config = OmegaConf.load(config_path)
    config.model.photomaker_path = str(DEFAULT_PM_PATH)
    config.metrics.id_sim_subject_v2.id_embeds_pth = str(SUBJECT_V2_EMBEDS)
    assert str(config.model.ba_hardcase_mode) == "temporal_frequency"
    assert bool(config.model.ba_null_key_router_enabled)
    assert list(config.model.ba_null_key_router_groups) == ["up_blocks.0", "up_blocks.1"]
    assert float(config.pipeline.pose_adapt_ratio) == 0.0
    assert not bool(config.pipeline.ca_mixing_for_face)
    assert bool(config.disable_branched_ca)
    assert int(config.validation_args.num_inference_steps) == 50
    assert float(config.validation_args.guidance_scale) == 5.0
    assert int(config.trainer.seed) == 0
    assert str(config.pretrained_model_for_validation_name_or_path) == "SG161222/RealVisXL_V4.0"
    assert str(config.validation_processor_base_mode) == "legacy_full_copy"
    return config, record


def verify_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint = checkpoint_dir / "checkpoint-epoch12.pth"
    if checkpoint.stat().st_size != 1_318_771_270:
        raise RuntimeError(f"Unexpected checkpoint size: {checkpoint.stat().st_size}")
    observed_sha = sha256_file(checkpoint)
    if observed_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {observed_sha}")
    config, record = load_sealed_config(checkpoint_dir)
    return {
        "run_name": RUN_NAME,
        "comet_key": (record["comet"] or {})["experiment_key"],
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": observed_sha,
        "checkpoint_epoch": 12,
        "optimizer_step": 24_000,
        "validation_base": str(config.pretrained_model_for_validation_name_or_path),
        "scheduler": "DDIM",
        "inference_steps": int(config.validation_args.num_inference_steps),
        "guidance_scale": float(config.validation_args.guidance_scale),
    }


def _constructor_kwargs(config) -> dict[str, Any]:
    return {
        name: getattr(config, name, default)
        for name, default in (
            ("train_ba_only", False),
            ("ba_train_top_k", 1.0),
            ("ba_patch_top_k", 1.0),
            ("non_ba_train", False),
            ("train_ba_all_steps", False),
            ("ba_weights_split", False),
            ("use_attn_v2", False),
        )
    }


def _build_model(config, base_model: str):
    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    model_config.pretrained_model_name_or_path = base_model
    model = instantiate(
        model_config,
        device=torch.device("cpu"),
        **_constructor_kwargs(config),
    )
    model.disable_branched_sa = bool(config.disable_branched_sa)
    model.disable_branched_ca = bool(config.disable_branched_ca)
    model.strict_face_routing = bool(getattr(config, "strict_face_routing", False))
    model.prepare_for_training()
    model.eval()
    return model


def _snapshot_processors(unet) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for name, processor in unet.attn_processors.items():
        if not hasattr(processor, "state_dict"):
            continue
        state = processor.state_dict()
        if state:
            result[name] = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }
    if not result:
        raise RuntimeError("No stateful training processors were found")
    return result


def _load_processor_snapshot(unet, snapshot: dict[str, dict[str, torch.Tensor]]) -> int:
    processors = unet.attn_processors
    copied = 0
    for name, state in snapshot.items():
        processor = processors.get(name)
        if processor is None:
            raise RuntimeError(f"Validation U-Net is missing processor {name}")
        processor.load_state_dict(state, strict=True)
        copied += 1
    return copied


def _sync_pipeline_flags(config, model, pipeline) -> None:
    pipeline.disable_branched_sa = bool(config.disable_branched_sa)
    pipeline.disable_branched_ca = bool(config.disable_branched_ca)
    for name in config.model.keys():
        if name.startswith("ba_"):
            value = getattr(model, name, config.model[name])
            setattr(pipeline, name, value)
    for name in (
        "strict_face_routing",
        "branched_trainable_dtype",
        "branched_attn_weight_mode",
        "branched_attn_new_weight_kind",
        "ba_weights_split",
        "ba_patch_top_k",
        "cache_prepared_masks",
    ):
        if hasattr(model, name):
            setattr(pipeline, name, getattr(model, name))


def build_validation_pipeline(config, checkpoint_path: Path, *, offload: str = "model"):
    """Recreate CL39's historical alternate-base validation path exactly."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(checkpoint.get("epoch", -1)) != 12:
        raise RuntimeError(f"Expected epoch 12, got {checkpoint.get('epoch')}")
    trainable_state = checkpoint.get("state_dict")
    if not isinstance(trainable_state, dict) or int(trainable_state.get("schema_version", 0)) != 2:
        raise RuntimeError("CL39 endpoint must contain a schema-v2 trainable state")

    training_base = str(config.model.pretrained_model_name_or_path)
    validation_base = str(config.pretrained_model_for_validation_name_or_path)
    started = time.time()
    print(f"[CL39 analysis] constructing training base {training_base}")
    training_model = _build_model(config, training_base)
    training_model.load_state_dict_(trainable_state)
    processor_snapshot = _snapshot_processors(training_model.unet)
    del training_model
    gc.collect()

    print(f"[CL39 analysis] constructing validation base {validation_base}")
    validation_model = _build_model(config, validation_base)
    validation_model.load_state_dict_(trainable_state)
    copied = _load_processor_snapshot(validation_model.unet, processor_snapshot)
    del processor_snapshot, checkpoint, trainable_state
    gc.collect()

    pipeline_config = OmegaConf.create(
        OmegaConf.to_container(config.pipeline, resolve=True)
    )
    pipeline_config.pretrained_model_name_or_path = validation_base
    accelerator = Accelerator(cpu=True)
    pipeline = instantiate(
        pipeline_config,
        model=validation_model,
        accelerator=accelerator,
    )
    _sync_pipeline_flags(config, validation_model, pipeline)
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    if offload == "model":
        pipeline.enable_model_cpu_offload(gpu_id=0)
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload(gpu_id=0)
    elif offload == "none":
        pipeline.to("cuda")
    else:
        raise ValueError(f"Unknown offload mode: {offload}")
    # PhotoMaker's ID encoder is an analysis-time attribute rather than a
    # registered Diffusers component, so keep only this comparatively small
    # module resident while the SDXL modules follow their normal offload hooks.
    pipeline.id_encoder.to(device="cuda", dtype=pipeline.unet.dtype)
    pipeline.set_progress_bar_config(disable=True)

    processors = getattr(pipeline, "_branched_attn_processors", {})
    null_processors = [
        name
        for name, processor in processors.items()
        if bool(getattr(processor, "null_key_router_enabled", False))
    ]
    expected_groups = {name.split(".")[1] for name in null_processors}
    if expected_groups != {"0", "1"}:
        raise RuntimeError(f"Unexpected CL39 processor groups: {expected_groups}")
    metadata = {
        "training_base": training_base,
        "validation_base": validation_base,
        "validation_processor_base_mode": "legacy_full_copy",
        "strict_processor_states_copied": copied,
        "null_key_processor_count": len(null_processors),
        "offload": offload,
        "construction_seconds": time.time() - started,
    }
    print(f"[CL39 analysis] pipeline ready: {metadata}")
    return pipeline, validation_model, metadata


def _dataset(config):
    return instantiate(config.datasets.val.manual_val)


def select_samples(dataset, *, seed: int, per_identity: int = 2) -> list[int]:
    by_identity: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        by_identity.setdefault(str(sample["id"]), []).append(index)
    rng = random.Random(seed)
    selected = []
    for identity in sorted(by_identity):
        candidates = list(by_identity[identity])
        selected.extend(rng.sample(candidates, per_identity))
    return sorted(selected)


def _sample_record(dataset, index: int) -> dict[str, Any]:
    item = dataset[index]
    source = dataset.samples[index]
    return {
        "index": index,
        "identity": str(item["id"]),
        "prompt": str(item["prompt"]),
        "seed": int(item["seed"]),
        "reference_path": str(Path(source["image_path"]).resolve()),
        "face_bbox_ref": item.get("face_bbox_ref"),
        "face_bbox_gen": item.get("face_bbox_gen"),
        "face_subject_selection_policy": item.get(
            "face_subject_selection_policy", "bbox_overlap_v2"
        ),
    }


def write_sample_manifest(config, output_root: Path, *, selection_seed: int) -> list[dict[str, Any]]:
    dataset = _dataset(config)
    indices = select_samples(dataset, seed=selection_seed)
    records = [_sample_record(dataset, index) for index in indices]
    bbox_path = Path(str(config.datasets.val.manual_val.bbox_mask_gen))
    bbox_map = json.loads(bbox_path.read_text(encoding="utf-8"))
    for record in records:
        # Validation joins with the literal space-bearing prompt[:10] key.
        # Underscore normalization belongs only to exported PNG/table joins.
        bbox_key = f"{record['prompt'][:10]}_{record['identity']}.png"
        entry = bbox_map.get(bbox_key)
        if not isinstance(entry, dict):
            raise KeyError(f"Missing sealed generated-face bbox: {bbox_key!r}")
        bbox = entry.get("face_crop_new") or entry.get("face_crop_old")
        if not bbox:
            raise KeyError(f"Sealed generated-face bbox has no face crop: {bbox_key!r}")
        record["face_bbox_gen"] = bbox
        record["face_bbox_gen_key"] = bbox_key
    payload = {
        "protocol": "manual_val fixed-96; deterministic stratified random 2/identity",
        "selection_seed": selection_seed,
        "indices": indices,
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sample_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return records


def _output_dir(output_root: Path, record: dict[str, Any]) -> Path:
    action = record["prompt"].split()[0].lower().replace("/", "-")
    return output_root / "outputs" / f"{record['index']:02d}_{record['identity']}_{action}"


def _generation_kwargs(config, item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    kwargs = OmegaConf.to_container(config.validation_args, resolve=True)
    kwargs.update(
        debug_dir=None,
        val_debug=False,
        debug_idx=int(record["index"]),
        debug_total=96,
        face_subject_selection_policy=str(record["face_subject_selection_policy"]),
    )
    return {
        "prompt": record["prompt"],
        "input_id_images": item["ref_images"],
        "face_bbox_ref": record["face_bbox_ref"],
        "face_bbox_gen": record["face_bbox_gen"],
        **kwargs,
    }


def generate(
    *,
    config,
    pipeline,
    records: list[dict[str, Any]],
    output_root: Path,
    force: bool,
    max_samples: int | None,
) -> None:
    dataset = _dataset(config)
    telemetry_dir = output_root / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "generation_status.json"
    selected = records[:max_samples] if max_samples is not None else records
    status: dict[str, Any] = {"completed": [], "failed": None}

    for ordinal, record in enumerate(selected, start=1):
        out_dir = _output_dir(output_root, record)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = telemetry_dir / f"{record['index']:02d}.npz"
        rows_path = telemetry_dir / f"{record['index']:02d}_layers.csv"
        required = [out_dir / f"{arm}.png" for arm in ARMS]
        if not force and npz_path.is_file() and all(path.is_file() for path in required):
            print(f"[CL39 analysis] skip complete sample {record['index']:02d}")
            status["completed"].append(record["index"])
            continue

        item = dataset[int(record["index"])]
        item["ref_images"][0].save(out_dir / "reference.png")
        (out_dir / "sample.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        kwargs = _generation_kwargs(config, item, record)
        print(
            f"[CL39 analysis] sample {ordinal}/{len(selected)} "
            f"index={record['index']:02d} {record['identity']} {record['prompt'].split()[0]}"
        )
        sample_started = time.time()
        try:
            for arm, arm_config in ARMS.items():
                collector = CL39AttentionCollector(map_size=64) if arm == "actual" else None
                attached = attach_cl39_analysis(
                    pipeline,
                    collector=collector,
                    confidence_override=arm_config["confidence_override"],
                    delta_scale=arm_config["delta_scale"],
                )
                if arm == "actual" and not attached:
                    raise RuntimeError("Capture attached to zero processors")
                # 25 Aug 2026 - The real trainer creates validation generators
                # on its CUDA device. Pixel replay additionally requires the
                # sealed 12-item batching, so publication runs use the YAML
                # trainer path; this one-item path is for interactive probing.
                generator = torch.Generator(device="cuda").manual_seed(
                    int(record["seed"])
                )
                arm_started = time.time()
                with torch.inference_mode():
                    image = pipeline(generator=generator, **kwargs).images[0]
                image.save(out_dir / f"{arm}.png")
                print(
                    f"[CL39 analysis]   arm={arm} seconds={time.time() - arm_started:.1f}"
                )
                if collector is not None:
                    collector.save(npz_path, rows_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            status["completed"].append(record["index"])
            (out_dir / "timing.json").write_text(
                json.dumps({"seconds": time.time() - sample_started}, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            status["failed"] = {
                "index": record["index"],
                "type": type(exc).__name__,
                "message": str(exc),
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            raise
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def _weighted_map(npz, field: str, selection: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(npz[field], dtype=np.float64)
    counts = np.asarray(npz["layer_count"], dtype=np.float64)
    if selection is not None:
        values = values[selection]
        counts = counts[selection]
    return np.average(values, axis=0, weights=counts)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _letterbox_reference(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Match the reference-latent resize/pad geometry used by the pipeline."""
    height, width = image.shape[:2]
    scale = min(size / float(width), size / float(height))
    resized_width = max(8, int(round(width * scale)) // 8 * 8)
    resized_height = max(8, int(round(height * scale)) // 8 * 8)
    resized = Image.fromarray((image * 255).round().astype(np.uint8)).resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(
        resized,
        ((size - resized_width) // 2, (size - resized_height) // 2),
    )
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _letterboxed_bbox(image: np.ndarray, bbox, size: int = 1024) -> list[float]:
    height, width = image.shape[:2]
    scale = min(size / float(width), size / float(height))
    resized_width = max(8, int(round(width * scale)) // 8 * 8)
    resized_height = max(8, int(round(height * scale)) // 8 * 8)
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [
        x0 * scale + pad_left,
        y0 * scale + pad_top,
        x1 * scale + pad_left,
        y1 * scale + pad_top,
    ]


def _draw_bbox(ax, bbox, *, color: str = "white") -> None:
    if not bbox:
        return
    x0, y0, x1, y1 = [float(value) for value in bbox]
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=1.5)
    )


def _show_heat(ax, image: np.ndarray, heat: np.ndarray, title: str, *, vmin=None, vmax=None, cmap="magma"):
    ax.imshow(image)
    shown = ax.imshow(heat, alpha=0.52, cmap=cmap, vmin=vmin, vmax=vmax, extent=(0, image.shape[1], image.shape[0], 0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return shown


def _difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first - second).mean(axis=2)


def render_sample_figure(record: dict[str, Any], output_root: Path, figure_dir: Path) -> dict[str, Any]:
    out_dir = _output_dir(output_root, record)
    npz_path = output_root / "telemetry" / f"{record['index']:02d}.npz"
    with np.load(npz_path) as npz:
        maps = {
            name: _weighted_map(npz, name)
            for name in (
                "entropy",
                "confidence",
                "router",
                "reference_key_mass_face",
                "raw_delta_magnitude",
                "low_magnitude",
                "high_magnitude",
                "effective_low_weight",
                "effective_high_weight",
                "low_applied_magnitude",
                "high_applied_magnitude",
                "routed_delta_magnitude",
                "routed_to_native_ratio",
            )
        }
    reference = _load_rgb(out_dir / "reference.png")
    reference_letterboxed = _letterbox_reference(reference)
    actual = _load_rgb(out_dir / "actual.png")
    c1 = _load_rgb(out_dir / "c1.png")
    ba_off = _load_rgb(out_dir / "ba_off.png")

    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    axes[0, 0].imshow(reference)
    axes[0, 0].set_title("Reference")
    _draw_bbox(axes[0, 0], record["face_bbox_ref"])
    axes[0, 0].axis("off")
    axes[0, 1].imshow(actual)
    axes[0, 1].set_title("Actual CL39")
    _draw_bbox(axes[0, 1], record["face_bbox_gen"])
    axes[0, 1].axis("off")
    _show_heat(axes[0, 2], actual, maps["entropy"], "Normalized entropy", vmin=0, vmax=1, cmap="viridis")
    _show_heat(axes[0, 3], actual, maps["confidence"], "Confidence C", vmin=0.25, vmax=1, cmap="viridis")
    ref_mass = maps["reference_key_mass_face"]
    ref_mass = ref_mass / max(float(ref_mass.max()), 1.0e-12)
    _show_heat(
        axes[0, 4],
        reference_letterboxed,
        ref_mass,
        "R key mass from face Q (normalized)",
        vmin=0,
        vmax=1,
    )
    _draw_bbox(
        axes[0, 4],
        _letterboxed_bbox(reference, record["face_bbox_ref"]),
        color="cyan",
    )

    for ax, field, title, cmap in (
        (axes[1, 0], "router", "Target router mask", "viridis"),
        (axes[1, 1], "raw_delta_magnitude", "|R−N|", "magma"),
        (axes[1, 2], "low_magnitude", "|D_low|", "magma"),
        (axes[1, 3], "high_magnitude", "|D_high|", "magma"),
        (axes[1, 4], "routed_to_native_ratio", "|correction| / |N|", "magma"),
    ):
        vmax = 1.0 if field == "router" else float(np.quantile(maps[field], 0.99))
        _show_heat(ax, actual, maps[field], title, vmin=0, vmax=max(vmax, 1.0e-8), cmap=cmap)

    for ax, field, title, vmax, cmap in (
        (axes[2, 0], "effective_low_weight", "C·router·s_low", 1.0, "viridis"),
        (axes[2, 1], "effective_high_weight", "C·router·s_high", 1.0, "viridis"),
        (axes[2, 2], "low_applied_magnitude", "Applied |D_low|", None, "magma"),
        (axes[2, 3], "high_applied_magnitude", "Applied |D_high|", None, "magma"),
        (axes[2, 4], "routed_delta_magnitude", "Applied |correction|", None, "magma"),
    ):
        if vmax is None:
            vmax = float(np.quantile(maps[field], 0.99))
        _show_heat(ax, actual, maps[field], title, vmin=0, vmax=max(vmax, 1.0e-8), cmap=cmap)

    axes[3, 0].imshow(actual)
    axes[3, 0].set_title("Actual C")
    axes[3, 1].imshow(c1)
    axes[3, 1].set_title("Counterfactual C=1")
    axes[3, 2].imshow(ba_off)
    axes[3, 2].set_title("Counterfactual correction=0")
    diff_c1 = _difference(actual, c1)
    diff_off = _difference(actual, ba_off)
    axes[3, 3].imshow(diff_c1, cmap="inferno", vmin=0, vmax=max(float(np.quantile(diff_c1, 0.99)), 1e-6))
    axes[3, 3].set_title("|actual − C=1| RGB")
    axes[3, 4].imshow(diff_off, cmap="inferno", vmin=0, vmax=max(float(np.quantile(diff_off, 0.99)), 1e-6))
    axes[3, 4].set_title("|actual − correction=0| RGB")
    for ax in axes[3]:
        ax.axis("off")
    fig.suptitle(
        f"manual_val {record['index']:02d} · {record['identity']} · {record['prompt']}",
        fontsize=12,
    )
    sample_figure_dir = figure_dir / "samples"
    sample_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = sample_figure_dir / f"{record['index']:02d}_{record['identity']}.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return {
        "index": record["index"],
        "figure": str(figure_path),
        "confidence_map_mean": float(maps["confidence"].mean()),
        "routed_to_native_map_mean": float(maps["routed_to_native_ratio"].mean()),
    }


def _face_crop(array: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(array.shape[1], x1), min(array.shape[0], y1)
    return array[y0:y1, x0:x1]


def score_counterfactuals(records, output_root: Path, *, score_identity: bool) -> pd.DataFrame:
    metric = None
    if score_identity:
        from src.metrics.id_sim_metric import IDSimMaskMatched

        metric = IDSimMaskMatched(
            id_embeds_pth=str(SUBJECT_V2_EMBEDS),
            device="cpu",
            metric_name="id_sim",
        )
    sealed_id_by_index = {}
    if SEALED_CL39_ID_TABLE.is_file():
        sealed_id = pd.read_csv(SEALED_CL39_ID_TABLE)
        sealed_id_by_index = sealed_id.set_index("image_index").to_dict("index")
    rows = []
    for record in records:
        out_dir = _output_dir(output_root, record)
        images = {arm: _load_rgb(out_dir / f"{arm}.png") for arm in ARMS}
        pil_images = {
            arm: Image.open(out_dir / f"{arm}.png").convert("RGB") for arm in ARMS
        }
        actual = images["actual"]
        for arm in ("c1", "ba_off"):
            other = images[arm]
            diff = np.abs(actual - other)
            face_actual = _face_crop(actual, record["face_bbox_gen"])
            face_other = _face_crop(other, record["face_bbox_gen"])
            rows.append(
                {
                    "index": record["index"],
                    "identity": record["identity"],
                    "action": record["prompt"].split()[0],
                    "comparison": f"actual_vs_{arm}",
                    "pixel_mae": float(diff.mean()),
                    "pixel_changed_gt_1_255": float(np.mean(diff.max(axis=2) > 1 / 255)),
                    "ssim": float(
                        structural_similarity(actual, other, data_range=1.0, channel_axis=2)
                    ),
                    "face_pixel_mae": float(np.abs(face_actual - face_other).mean()),
                }
            )
        sealed_key = f"{record['prompt'][:10].replace(' ', '_')}_{record['identity']}.png"
        sealed_path = SEALED_CL39_IMAGE_DIR / sealed_key
        if sealed_path.is_file():
            sealed = _load_rgb(sealed_path)
            diff = np.abs(actual - sealed)
            face_actual = _face_crop(actual, record["face_bbox_gen"])
            face_sealed = _face_crop(sealed, record["face_bbox_gen"])
            rows.append(
                {
                    "index": record["index"],
                    "identity": record["identity"],
                    "action": record["prompt"].split()[0],
                    "comparison": "actual_vs_sealed_serv",
                    "pixel_mae": float(diff.mean()),
                    "pixel_changed_gt_1_255": float(np.mean(diff.max(axis=2) > 1 / 255)),
                    "ssim": float(
                        structural_similarity(actual, sealed, data_range=1.0, channel_axis=2)
                    ),
                    "face_pixel_mae": float(np.abs(face_actual - face_sealed).mean()),
                }
            )
        if metric is not None:
            for arm, image in pil_images.items():
                result = metric(
                    generated=[image],
                    ref_images=[Image.open(record["reference_path"]).convert("RGB")],
                    id=record["identity"],
                    face_bbox_gen=record["face_bbox_gen"],
                )
                rows.append(
                    {
                        "index": record["index"],
                        "identity": record["identity"],
                        "action": record["prompt"].split()[0],
                        "comparison": f"id_sim_{arm}",
                        "id_sim": float(result["id_sim"]),
                        "id_sim_mask_iou": float(result["id_sim_mask_iou"]),
                        "id_sim_unowned": float(result["id_sim_unowned"]),
                    }
                )
        sealed_id_row = sealed_id_by_index.get(int(record["index"]))
        if sealed_id_row is not None:
            rows.append(
                {
                    "index": record["index"],
                    "identity": record["identity"],
                    "action": record["prompt"].split()[0],
                    "comparison": "id_sim_sealed_serv",
                    "id_sim": float(sealed_id_row["id_sim"]),
                    "id_sim_mask_iou": float(sealed_id_row["id_sim_mask_iou"]),
                    "id_sim_unowned": float(sealed_id_row["id_sim_unowned"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "counterfactual_metrics.csv", index=False)
    return frame


def render_overview(records, output_root: Path, figure_dir: Path) -> None:
    def _render(subset, filename: str) -> None:
        fig, axes = plt.subplots(
            len(subset),
            5,
            figsize=(12, 2.25 * len(subset)),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(subset), 5)
        for row_index, record in enumerate(subset):
            out_dir = _output_dir(output_root, record)
            reference = _load_rgb(out_dir / "reference.png")
            actual = _load_rgb(out_dir / "actual.png")
            ba_off = _load_rgb(out_dir / "ba_off.png")
            with np.load(output_root / "telemetry" / f"{record['index']:02d}.npz") as npz:
                confidence = _weighted_map(npz, "confidence")
                correction = _weighted_map(npz, "routed_delta_magnitude")
            axes[row_index, 0].imshow(reference)
            axes[row_index, 1].imshow(actual)
            _show_heat(
                axes[row_index, 2],
                actual,
                confidence,
                "",
                vmin=0.25,
                vmax=1,
                cmap="viridis",
            )
            vmax = max(float(np.quantile(correction, 0.99)), 1e-8)
            _show_heat(
                axes[row_index, 3], actual, correction, "", vmin=0, vmax=vmax
            )
            diff = _difference(actual, ba_off)
            axes[row_index, 4].imshow(
                diff,
                cmap="inferno",
                vmin=0,
                vmax=max(float(np.quantile(diff, 0.99)), 1e-6),
            )
            for column in range(5):
                axes[row_index, column].axis("off")
            axes[row_index, 0].set_ylabel(
                f"{record['index']:02d} {record['identity']}\n{record['prompt'].split()[0]}",
                fontsize=8,
            )
        for ax, title in zip(
            axes[0],
            (
                "Reference",
                "Actual CL39",
                "Confidence C",
                "Applied |correction|",
                "|actual−BA-off|",
            ),
        ):
            ax.set_title(title, fontsize=10)
        fig.savefig(figure_dir / filename, dpi=150)
        plt.close(fig)

    _render(records, "cl39_16_sample_overview.png")
    for start in range(0, len(records), 4):
        subset = records[start : start + 4]
        first, last = subset[0]["index"], subset[-1]["index"]
        _render(subset, f"cl39_overview_{first:02d}_{last:02d}.png")


def render_temporal(rows: pd.DataFrame, figure_dir: Path) -> None:
    mean = rows.groupby("progress", as_index=False).mean(numeric_only=True).sort_values("progress")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(mean.progress, mean.confidence_mean_face, label="C")
    axes[0, 0].plot(mean.progress, mean.confidence_face_p10, label="C p10", linestyle="--")
    axes[0, 0].plot(mean.progress, mean.confidence_face_p90, label="C p90", linestyle="--")
    axes[0, 0].set_title("Entropy confidence within routed face")
    axes[0, 0].set_ylim(0.2, 1.02)
    axes[0, 0].legend()

    axes[0, 1].plot(mean.progress, mean.effective_low_weight_mean_face, label="effective low weight")
    axes[0, 1].plot(mean.progress, mean.effective_high_weight_mean_face, label="effective high weight")
    axes[0, 1].plot(mean.progress, mean.low_scale, label="scheduled low scale", linestyle=":")
    axes[0, 1].plot(mean.progress, mean.high_scale, label="scheduled high scale", linestyle=":")
    axes[0, 1].set_title("Scheduled versus actually applied band weights")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(mean.progress, mean.low_applied_magnitude_mean_face, label="applied D_low")
    axes[1, 0].plot(mean.progress, mean.high_applied_magnitude_mean_face, label="applied D_high")
    axes[1, 0].plot(mean.progress, mean.routed_delta_magnitude_mean_face, label="merged correction")
    axes[1, 0].set_title("Correction-band magnitudes")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(mean.progress, mean.routed_to_native_ratio_mean_face, label="|correction| / |N|")
    axes[1, 1].plot(mean.progress, mean.confidence_floor_fraction, label="fraction at C floor")
    axes[1, 1].plot(mean.progress, mean.confidence_full_fraction, label="fraction at C≈1")
    axes[1, 1].set_title("Reference-lane use versus fallback")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("denoising progress")
        ax.grid(alpha=0.2)
    fig.savefig(figure_dir / "cl39_temporal_mechanism.png", dpi=170)
    plt.close(fig)


def render_entropy_confidence(records, output_root: Path, figure_dir: Path) -> None:
    entropies = []
    confidences = []
    for record in records:
        with np.load(output_root / "telemetry" / f"{record['index']:02d}.npz") as npz:
            entropy = np.asarray(npz["entropy"], dtype=np.float32)
            confidence = np.asarray(npz["confidence"], dtype=np.float32)
            router = np.asarray(npz["router"], dtype=np.float32)
            active = router > 1.0e-4
            entropies.append(entropy[active])
            confidences.append(confidence[active])
    entropy = np.concatenate(entropies)
    confidence = np.concatenate(confidences)
    if entropy.size > 300_000:
        indices = np.linspace(0, entropy.size - 1, 300_000).round().astype(int)
        entropy = entropy[indices]
        confidence = confidence[indices]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    shown = axes[0].hexbin(
        entropy,
        confidence,
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    entropy_curve = np.linspace(0, 1, 500)
    null_mass = 1.0 / (1.0 + np.exp(-(entropy_curve - 0.75) / 0.08))
    confidence_curve = np.clip(1.0 - 0.75 * null_mass, 0.25, 1.0)
    axes[0].plot(entropy_curve, confidence_curve, color="cyan", linewidth=2, label="fixed mapping")
    axes[0].set_xlabel("normalized reference-attention entropy")
    axes[0].set_ylabel("confidence C")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0.24, 1.01)
    axes[0].set_title("Entropy–confidence calibration (aggregated cells)")
    axes[0].legend()
    fig.colorbar(shown, ax=axes[0], label="log count")

    axes[1].hist(confidence, bins=np.linspace(0.25, 1.0, 61), color="#3366aa", alpha=0.85)
    axes[1].axvline(float(confidence.mean()), color="black", linestyle="--", label=f"mean {confidence.mean():.3f}")
    axes[1].set_xlabel("confidence C in routed target positions")
    axes[1].set_ylabel("aggregated map cells")
    axes[1].set_xlim(0.24, 1.01)
    axes[1].set_title("Routed-position confidence distribution")
    axes[1].legend()
    fig.savefig(figure_dir / "cl39_entropy_confidence_calibration.png", dpi=170)
    plt.close(fig)


def render_layer_heatmaps(rows: pd.DataFrame, figure_dir: Path) -> None:
    metrics = (
        ("confidence_mean_face", "Mean C"),
        ("routed_to_native_ratio_mean_face", "|correction| / |N|"),
        ("low_applied_magnitude_mean_face", "Applied D_low"),
        ("high_applied_magnitude_mean_face", "Applied D_high"),
    )
    progress_values = sorted(rows.progress.unique())
    group_values = ["up0", "up1"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        matrix = np.full((len(group_values), len(progress_values)), np.nan)
        for yi, group in enumerate(group_values):
            subset = rows[rows.group == group].groupby("progress")[metric].mean()
            for xi, progress in enumerate(progress_values):
                if progress in subset:
                    matrix[yi, xi] = subset[progress]
        shown = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_yticks(range(len(group_values)), group_values)
        ticks = np.linspace(0, len(progress_values) - 1, 6).round().astype(int)
        ax.set_xticks(ticks, [f"{progress_values[index]:.2f}" for index in ticks])
        ax.set_title(title)
        fig.colorbar(shown, ax=ax, fraction=0.018, pad=0.01)
    axes[-1].set_xlabel("denoising progress")
    fig.savefig(figure_dir / "cl39_layer_step_heatmaps.png", dpi=170)
    plt.close(fig)


def render_counterfactual_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    pixel = metrics[metrics.comparison.str.startswith("actual_vs_")].copy()
    labels = ["actual_vs_c1", "actual_vs_ba_off"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for comparison in labels:
        subset = pixel[pixel.comparison == comparison]
        axes[0].scatter([comparison] * len(subset), subset.pixel_mae, alpha=0.75)
        axes[1].scatter([comparison] * len(subset), subset.face_pixel_mae, alpha=0.75)
        axes[2].scatter([comparison] * len(subset), 1.0 - subset.ssim, alpha=0.75)
    axes[0].set_title("Full-image RGB MAE")
    axes[1].set_title("Sealed-face-box RGB MAE")
    axes[2].set_title("1 − SSIM")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelrotation=15)
    fig.savefig(figure_dir / "cl39_counterfactual_distributions.png", dpi=170)
    plt.close(fig)


def render_replay_boundary(records, output_root: Path, figure_dir: Path, metrics: pd.DataFrame) -> None:
    """Verify the trainer/YAML replay against the sealed Serv A100 outputs."""
    count = min(4, len(records))
    chosen = [
        records[index]
        for index in np.linspace(0, len(records) - 1, count).round().astype(int)
    ]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(chosen), 3)
    for row, record in enumerate(chosen):
        local = _load_rgb(_output_dir(output_root, record) / "actual.png")
        sealed_key = f"{record['prompt'][:10].replace(' ', '_')}_{record['identity']}.png"
        sealed = _load_rgb(SEALED_CL39_IMAGE_DIR / sealed_key)
        comparison = metrics[
            (metrics["index"] == record["index"])
            & (metrics.comparison == "actual_vs_sealed_serv")
        ].iloc[0]
        axes[row, 0].imshow(local)
        axes[row, 1].imshow(sealed)
        difference = _difference(local, sealed)
        axes[row, 2].imshow(
            difference,
            cmap="inferno",
            vmin=0,
            vmax=max(float(np.quantile(difference, 0.99)), 1.0e-6),
        )
        axes[row, 0].set_ylabel(
            f"{record['index']:02d} {record['identity']}\n{record['prompt'].split()[0]}",
            fontsize=8,
        )
        axes[row, 2].set_title(
            f"MAE {comparison.pixel_mae:.3f} · SSIM {comparison.ssim:.3f}",
            fontsize=9,
        )
        for ax in axes[row]:
            ax.axis("off")
    axes[0, 0].set_title("Serv trainer/YAML replay")
    axes[0, 1].set_title("Sealed Serv output (A100)")
    fig.savefig(figure_dir / "cl39_serv_trainer_vs_sealed_replay.png", dpi=160)
    plt.close(fig)


def _band_spatial_variation(records, output_root: Path) -> dict[str, float]:
    """Measure spatial roughness of the captured low/high magnitude maps."""
    aggregate: dict[str, list[float]] = {
        "low_absolute": [],
        "high_absolute": [],
        "low_normalized": [],
        "high_normalized": [],
    }
    for record in records:
        with np.load(output_root / "telemetry" / f"{record['index']:02d}.npz") as npz:
            counts = np.asarray(npz["layer_count"], dtype=np.float64)
            for band in ("low", "high"):
                maps = np.asarray(npz[f"{band}_magnitude"], dtype=np.float64)
                variation = (
                    np.abs(np.diff(maps, axis=1)).mean(axis=(1, 2))
                    + np.abs(np.diff(maps, axis=2)).mean(axis=(1, 2))
                )
                magnitude = np.abs(maps).mean(axis=(1, 2))
                aggregate[f"{band}_absolute"].append(
                    float(np.average(variation, weights=counts))
                )
                aggregate[f"{band}_normalized"].append(
                    float(np.average(variation / np.maximum(magnitude, 1.0e-12), weights=counts))
                )
    result = {name: float(np.mean(values)) for name, values in aggregate.items()}
    result["high_to_low_absolute_ratio"] = result["high_absolute"] / max(
        result["low_absolute"], 1.0e-12
    )
    result["high_to_low_normalized_ratio"] = result["high_normalized"] / max(
        result["low_normalized"], 1.0e-12
    )
    return result


def _reference_face_attention(records, output_root: Path) -> dict[str, float]:
    """Estimate how much face-query reference-key mass lands inside the sealed bbox."""
    face_mass_values = []
    face_area_values = []
    enrichment_values = []
    for record in records:
        reference = Image.open(record["reference_path"]).convert("RGB")
        width, height = reference.size
        size = 1024
        scale = min(size / float(width), size / float(height))
        resized_width = max(8, int(round(width * scale)) // 8 * 8)
        resized_height = max(8, int(round(height * scale)) // 8 * 8)
        pad_left = (size - resized_width) // 2
        pad_top = (size - resized_height) // 2
        x0, y0, x1, y1 = [float(value) for value in record["face_bbox_ref"]]
        x0 = int(np.floor((x0 * scale + pad_left) * 64 / size))
        x1 = int(np.ceil((x1 * scale + pad_left) * 64 / size))
        y0 = int(np.floor((y0 * scale + pad_top) * 64 / size))
        y1 = int(np.ceil((y1 * scale + pad_top) * 64 / size))
        x0, x1 = max(0, x0), min(64, x1)
        y0, y1 = max(0, y0), min(64, y1)
        mask = np.zeros((64, 64), dtype=bool)
        mask[y0:y1, x0:x1] = True
        with np.load(output_root / "telemetry" / f"{record['index']:02d}.npz") as npz:
            mass = _weighted_map(npz, "reference_key_mass_face")
        face_fraction = float(mass[mask].sum() / max(float(mass.sum()), 1.0e-12))
        area_fraction = float(mask.mean())
        face_mass_values.append(face_fraction)
        face_area_values.append(area_fraction)
        enrichment_values.append(face_fraction / max(area_fraction, 1.0e-12))
    return {
        "mass_inside_reference_face_bbox_mean": float(np.mean(face_mass_values)),
        "mass_outside_reference_face_bbox_mean": float(1.0 - np.mean(face_mass_values)),
        "reference_face_bbox_area_fraction_mean": float(np.mean(face_area_values)),
        "face_mass_enrichment_over_uniform_mean": float(np.mean(enrichment_values)),
    }


def build_summary(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    records,
    output_root: Path,
) -> dict[str, Any]:
    pixel = metrics[metrics.comparison.str.startswith("actual_vs_")]
    comparisons = {}
    for comparison, subset in pixel.groupby("comparison"):
        comparisons[comparison] = {
            "pixel_mae_mean": float(subset.pixel_mae.mean()),
            "pixel_mae_median": float(subset.pixel_mae.median()),
            "face_pixel_mae_mean": float(subset.face_pixel_mae.mean()),
            "ssim_mean": float(subset.ssim.mean()),
            "pixel_changed_gt_1_255_mean": float(subset.pixel_changed_gt_1_255.mean()),
        }
    id_rows = metrics[metrics.comparison.str.startswith("id_sim_")]
    id_summary = {
        comparison: float(subset.id_sim.mean())
        for comparison, subset in id_rows.groupby("comparison")
    } if not id_rows.empty else {}
    id_diagnostics = {
        comparison: {
            "mask_iou_mean": float(subset.id_sim_mask_iou.mean()),
            "unowned_fraction": float(subset.id_sim_unowned.mean()),
        }
        for comparison, subset in id_rows.groupby("comparison")
    } if not id_rows.empty else {}
    paired_id = {}
    if not id_rows.empty:
        pivot = id_rows.pivot(index="index", columns="comparison", values="id_sim")
        rng = np.random.default_rng(390025)
        for comparison in ("id_sim_ba_off", "id_sim_c1"):
            delta = (pivot["id_sim_actual"] - pivot[comparison]).to_numpy()
            bootstrap = delta[
                rng.integers(0, len(delta), size=(100_000, len(delta)))
            ].mean(axis=1)
            paired_id[f"actual_minus_{comparison.removeprefix('id_sim_')}"] = {
                "mean": float(delta.mean()),
                "median": float(np.median(delta)),
                "wins": int(np.sum(delta > 1.0e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1.0e-12)),
                "losses": int(np.sum(delta < -1.0e-12)),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_seed": 390025,
            }
    face = rows
    low = face.low_applied_magnitude_mean_face
    high = face.high_applied_magnitude_mean_face
    summary = {
        "run_name": RUN_NAME,
        "comet_key": COMET_KEY,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_count": len(records),
        "selection_indices": [record["index"] for record in records],
        "layer_call_count": len(rows),
        "entropy_face_mean": float(face.entropy_mean_face.mean()),
        "null_mass_face_mean": float(face.null_mass_mean_face.mean()),
        "target_router_grid_mean": float(face.router_mean_all.mean()),
        "confidence_all_queries_mean": float(face.confidence_mean_all.mean()),
        "confidence_face_mean": float(face.confidence_mean_face.mean()),
        "confidence_face_p10_mean": float(face.confidence_face_p10.mean()),
        "confidence_face_p50_mean": float(face.confidence_face_p50.mean()),
        "confidence_face_p90_mean": float(face.confidence_face_p90.mean()),
        "confidence_floor_fraction_mean": float(face.confidence_floor_fraction.mean()),
        "confidence_full_fraction_mean": float(face.confidence_full_fraction.mean()),
        "effective_low_weight_face_mean": float(face.effective_low_weight_mean_face.mean()),
        "effective_high_weight_face_mean": float(face.effective_high_weight_mean_face.mean()),
        "raw_delta_face_mean": float(face.raw_delta_magnitude_mean_face.mean()),
        "routed_delta_face_mean": float(face.routed_delta_magnitude_mean_face.mean()),
        "routed_to_native_face_mean": float(face.routed_to_native_ratio_mean_face.mean()),
        "low_applied_face_mean": float(low.mean()),
        "high_applied_face_mean": float(high.mean()),
        "high_share_of_band_magnitude": float((high / (low + high).clip(lower=1e-12)).mean()),
        "band_spatial_total_variation": _band_spatial_variation(records, output_root),
        "reference_attention": _reference_face_attention(records, output_root),
        "reconstruction_error_relative": float(
            face.reconstruction_error_magnitude_mean_face.mean()
            / max(face.raw_delta_magnitude_mean_face.mean(), 1e-12)
        ),
        "counterfactuals": comparisons,
        "id_sim_subject_v2": id_summary,
        "id_sim_subject_v2_diagnostics": id_diagnostics,
        "id_sim_subject_v2_paired": paired_id,
    }
    return summary


def write_per_sample_summary(
    records,
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "index": [record["index"] for record in records],
            "identity": [record["identity"] for record in records],
            "action": [record["prompt"].split()[0] for record in records],
            "prompt": [record["prompt"] for record in records],
        }
    )
    telemetry_columns = {
        "entropy_mean_face": "entropy_face",
        "confidence_mean_face": "confidence_face",
        "effective_low_weight_mean_face": "effective_low_weight_face",
        "effective_high_weight_mean_face": "effective_high_weight_face",
        "raw_delta_magnitude_mean_face": "raw_delta_face",
        "routed_delta_magnitude_mean_face": "applied_correction_face",
        "routed_to_native_ratio_mean_face": "correction_to_native_face",
        "low_applied_magnitude_mean_face": "low_applied_face",
        "high_applied_magnitude_mean_face": "high_applied_face",
    }
    telemetry = (
        rows.groupby("sample_index")[list(telemetry_columns)]
        .mean()
        .rename(columns=telemetry_columns)
        .rename_axis("index")
        .reset_index()
    )
    result = metadata.merge(telemetry, on="index", how="left", validate="one_to_one")
    pixel = metrics[metrics.comparison.isin(("actual_vs_c1", "actual_vs_ba_off"))]
    for comparison, suffix in (("actual_vs_c1", "c1"), ("actual_vs_ba_off", "ba_off")):
        subset = pixel[pixel.comparison == comparison][
            ["index", "pixel_mae", "face_pixel_mae", "ssim", "pixel_changed_gt_1_255"]
        ].rename(columns={
            "pixel_mae": f"actual_vs_{suffix}_pixel_mae",
            "face_pixel_mae": f"actual_vs_{suffix}_face_pixel_mae",
            "ssim": f"actual_vs_{suffix}_ssim",
            "pixel_changed_gt_1_255": f"actual_vs_{suffix}_changed_fraction",
        })
        result = result.merge(subset, on="index", how="left", validate="one_to_one")
    identity = metrics[metrics.comparison.isin(
        ("id_sim_actual", "id_sim_c1", "id_sim_ba_off", "id_sim_sealed_serv")
    )].pivot(index="index", columns="comparison", values="id_sim")
    identity = identity.rename(columns=lambda name: name.removeprefix("id_sim_")).reset_index()
    result = result.merge(identity, on="index", how="left", validate="one_to_one")
    result["actual_minus_ba_off_id"] = result["actual"] - result["ba_off"]
    result["actual_minus_c1_id"] = result["actual"] - result["c1"]
    result.to_csv(output_root / "per_sample_summary.csv", index=False)
    return result


def render(records, output_root: Path, figure_dir: Path, *, score_identity: bool) -> dict[str, Any]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sample_summaries = [
        render_sample_figure(record, output_root, figure_dir) for record in records
    ]
    row_frames = [
        pd.read_csv(output_root / "telemetry" / f"{record['index']:02d}_layers.csv")
        .assign(sample_index=record["index"], identity=record["identity"])
        for record in records
    ]
    rows = pd.concat(row_frames, ignore_index=True)
    rows.to_csv(output_root / "all_layer_calls.csv", index=False)
    metrics = score_counterfactuals(records, output_root, score_identity=score_identity)
    render_overview(records, output_root, figure_dir)
    render_temporal(rows, figure_dir)
    render_entropy_confidence(records, output_root, figure_dir)
    render_layer_heatmaps(rows, figure_dir)
    render_counterfactual_summary(metrics, figure_dir)
    render_replay_boundary(records, output_root, figure_dir, metrics)
    write_per_sample_summary(records, rows, metrics, output_root)
    summary = build_summary(rows, metrics, records, output_root)
    summary["sample_figures"] = sample_summaries
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_manifest(output_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((output_root / "sample_manifest.json").read_text(encoding="utf-8"))
    return payload["samples"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "generate", "render", "all"))
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--selection-seed", type=int, default=390024)
    parser.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-id-score", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    verification = verify_checkpoint(args.checkpoint_dir)
    print(json.dumps(verification, indent=2))
    if args.stage == "verify":
        return
    config, _record = load_sealed_config(args.checkpoint_dir)
    if args.stage in {"generate", "all"}:
        records = write_sample_manifest(
            config, args.output_root, selection_seed=args.selection_seed
        )
        pipeline, validation_model, build_metadata = build_validation_pipeline(
            config,
            args.checkpoint_dir / "checkpoint-epoch12.pth",
            offload=args.offload,
        )
        (args.output_root / "pipeline_metadata.json").write_text(
            json.dumps({**verification, **build_metadata}, indent=2) + "\n",
            encoding="utf-8",
        )
        generate(
            config=config,
            pipeline=pipeline,
            records=records,
            output_root=args.output_root,
            force=args.force,
            max_samples=args.max_samples,
        )
        del pipeline, validation_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage in {"render", "all"}:
        records = load_manifest(args.output_root)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        summary = render(
            records,
            args.output_root,
            args.figure_dir,
            score_identity=not args.skip_id_score,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
