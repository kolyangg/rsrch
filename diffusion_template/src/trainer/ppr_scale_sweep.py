"""Checkpoint-only residual-strength sweep for the NN2-PPR branch."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from src.trainer.ppr_diagnostic import (
    _generate,
    _metric_values,
    _normalize_refs,
    _per_sample,
    _pixel_mae,
    _select_spatial_swap_indices,
    _swap_source,
)


METADATA_FIELDS = [
    "sample_index",
    "filename",
    "checkpoint",
    "runtime_scale",
    "seed",
    "prompt_id",
    "prompt",
    "reference_id",
    "spatial_swap_reference_id",
    "ppr_mode",
    "active_processor_count",
    "mean_gate",
    "applied_delta_rms_ratio",
    "cap_fraction",
    "sha256",
    "whole_image_mae_vs_scale_0",
    "face_core_mae_vs_scale_0",
    "face_core_lpips_vs_scale_0",
    "whole_image_mae_vs_same_scale",
    "face_core_mae_vs_same_scale",
    "id_similarity",
    "text_similarity",
]


def _scale_label(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _parse_scales(config) -> tuple[float, ...]:
    raw = getattr(config, "ppr_scale_sweep_scales", (0, 1, 2, 3, 4))
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    scales = tuple(float(value) for value in raw)
    if (
        not scales
        or len(set(scales)) != len(scales)
        or any(not math.isfinite(scale) or scale < 0 for scale in scales)
    ):
        raise ValueError(f"Invalid ppr_scale_sweep_scales: {scales}")
    if 0.0 not in scales:
        raise ValueError("PPR scale sweep requires scale 0 as the exact baseline")
    return scales


def _collapse_cfg(values: list[float], batch_size: int, *, capped: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.size % batch_size != 0:
        raise RuntimeError(
            f"Cannot map {array.size} processor values to batch size {batch_size}"
        )
    array = array.reshape(-1, batch_size)
    if capped:
        return (array < 1.0 - 1e-7).mean(axis=0)
    return array.mean(axis=0)


def _processor_stats(
    diagnostics: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, float | int]]:
    records = [
        record
        for record in diagnostics
        if record.get("record_type") == "processor_applied_ratio"
    ]
    processors = {str(record["processor"]) for record in records}
    if not records:
        raise RuntimeError("Scale sweep recorded no active PPR processor diagnostics")
    ratios = np.stack(
        [
            _collapse_cfg(record.get("applied_ratios", []), batch_size, capped=False)
            for record in records
        ]
    )
    caps = np.stack(
        [
            _collapse_cfg(record.get("cap_scales", []), batch_size, capped=True)
            for record in records
        ]
    )
    mean_gate = float(np.mean([float(record["gate"]) for record in records]))
    return [
        {
            "active_processor_count": len(processors),
            "mean_gate": mean_gate,
            "applied_delta_rms_ratio": float(ratios[:, index].mean()),
            "cap_fraction": float(caps[:, index].mean()),
        }
        for index in range(batch_size)
    ]


def _face_crop(image: Image.Image, bbox) -> Image.Image | None:
    if bbox is None or len(bbox) != 4:
        return None
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    inset_x = max((x1 - x0) * 0.10, 0.0)
    inset_y = max((y1 - y0) * 0.10, 0.0)
    box = (
        max(0, min(width, int(round(x0 + inset_x)))),
        max(0, min(height, int(round(y0 + inset_y)))),
        max(0, min(width, int(round(x1 - inset_x)))),
        max(0, min(height, int(round(y1 - inset_y)))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.convert("RGB").crop(box).resize((256, 256), Image.Resampling.BICUBIC)


def _face_lpips(state, image: Image.Image, baseline: Image.Image, bbox) -> float:
    if image is baseline:
        return 0.0
    if "_lpips_model" not in state:
        try:
            import lpips

            state["_lpips_model"] = lpips.LPIPS(net="alex").to(state["device"]).eval()
            state["lpips_status"] = "available"
        except Exception as error:
            state["_lpips_model"] = None
            state["lpips_status"] = f"unavailable: {type(error).__name__}: {error}"
    model = state["_lpips_model"]
    if model is None:
        return float("nan")
    crop = _face_crop(image, bbox)
    baseline_crop = _face_crop(baseline, bbox)
    if crop is None or baseline_crop is None:
        return float("nan")

    def _tensor(value: Image.Image) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(state["device"])

    with torch.no_grad():
        return float(model(_tensor(crop), _tensor(baseline_crop)).item())


def _initialize_state(trainer) -> dict[str, Any]:
    root = Path(str(trainer.config.ppr_scale_sweep_output_dir)).expanduser().resolve()
    overwrite = bool(getattr(trainer.config, "ppr_scale_sweep_overwrite", False))
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"PPR scale-sweep output already exists: {root}. "
                "Set ppr_scale_sweep_overwrite=true to replace it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "contact_sheets").mkdir()

    scales = _parse_scales(trainer.config)
    for scale in scales:
        (root / f"scale_{_scale_label(scale)}").mkdir()

    if len(trainer.evaluation_dataloaders) != 1:
        raise RuntimeError("PPR scale sweep requires exactly one validation dataset")
    dataset = next(iter(trainer.evaluation_dataloaders.values())).dataset
    swap_scale_raw = getattr(trainer.config, "ppr_scale_sweep_swap_scale", None)
    swap_scale = None if swap_scale_raw is None else float(swap_scale_raw)
    if swap_scale is not None and swap_scale not in scales:
        raise ValueError(
            f"ppr_scale_sweep_swap_scale={swap_scale} is not in scales={scales}"
        )
    swap_indices: set[int] = set()
    if swap_scale is not None:
        swap_count = int(getattr(trainer.config, "ppr_scale_sweep_swap_count", 12))
        swap_indices = _select_spatial_swap_indices(dataset, swap_count)
        (root / f"reference_swap_scale_{_scale_label(swap_scale)}").mkdir()

    identity_sources = []
    for image_path in getattr(dataset, "images", ()):
        identity = image_path.stem
        bbox = getattr(dataset, "_bbox_map_ref", {}).get(identity)
        identity_sources.append((identity, Path(image_path), bbox))
    if swap_scale is not None and len({item[0] for item in identity_sources}) < 2:
        raise RuntimeError("Reference-swap sweep requires at least two identities")

    state = {
        "root": root,
        "scales": scales,
        "swap_scale": swap_scale,
        "swap_indices": swap_indices,
        "identity_sources": identity_sources,
        "rows": [],
        "diagnostics": [],
        "filenames": [],
        "next_index": 0,
        "device": trainer.device,
        "lpips_status": "not attempted",
    }
    trainer._ppr_scale_sweep_state = state
    print(
        "[PPR scale sweep] "
        f"output={root} scales={list(scales)} swap_scale={swap_scale} "
        f"samples={len(dataset)}"
    )
    return state


def _save_image(path: Path, image: Image.Image) -> str:
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _append_row(
    trainer,
    state,
    eval_metrics,
    *,
    image: Image.Image,
    baseline: Image.Image,
    same_scale_baseline: Image.Image | None,
    bbox,
    filename: str,
    output_path: Path,
    sample_index: int,
    scale: float,
    seed: int,
    prompt: str,
    reference_id: str,
    swap_reference_id: str,
    mode: str,
    processor_stats: dict[str, float | int],
) -> None:
    sha256 = _save_image(output_path, image)
    whole_mae, face_mae = _pixel_mae(image, baseline, bbox)
    same_scale_whole_mae, same_scale_face_mae = _pixel_mae(
        image,
        same_scale_baseline if same_scale_baseline is not None else image,
        bbox,
    )
    lpips_value = _face_lpips(state, image, baseline, bbox)
    metric_values = _metric_values(
        trainer,
        image=image,
        prompt=prompt,
        identity=reference_id,
    )
    for metric_name, metric_value in metric_values.items():
        if np.isfinite(metric_value):
            eval_metrics.update(f"scale_{_scale_label(scale)}/{metric_name}", metric_value)
    state["rows"].append(
        {
            "sample_index": sample_index,
            "filename": filename,
            "checkpoint": str(getattr(trainer.config, "saved_checkpoint", "")),
            "runtime_scale": scale,
            "seed": seed,
            "prompt_id": _prompt_id(prompt),
            "prompt": prompt,
            "reference_id": reference_id,
            "spatial_swap_reference_id": swap_reference_id,
            "ppr_mode": mode,
            **processor_stats,
            "sha256": sha256,
            "whole_image_mae_vs_scale_0": whole_mae,
            "face_core_mae_vs_scale_0": face_mae,
            "face_core_lpips_vs_scale_0": lpips_value,
            "whole_image_mae_vs_same_scale": same_scale_whole_mae,
            "face_core_mae_vs_same_scale": same_scale_face_mae,
            **metric_values,
        }
    )


@torch.no_grad()
def run_ppr_scale_sweep_batch(trainer, batch, eval_metrics):
    state = getattr(trainer, "_ppr_scale_sweep_state", None)
    if state is None:
        state = _initialize_state(trainer)

    prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
    batch_size = len(prompts)
    identities = [str(value) for value in _per_sample(batch.get("id"), batch_size)]
    seeds = [
        int(value)
        for value in _per_sample(
            batch.get("seed", trainer.config.validation_args.get("seed", 0)),
            batch_size,
        )
    ]
    references = _normalize_refs(batch.get("ref_images"), batch_size)
    reference_bboxes = _per_sample(batch.get("face_bbox_ref"), batch_size)
    generation_bboxes = _per_sample(batch.get("face_bbox_gen"), batch_size)
    if any(bbox is None for bbox in generation_bboxes):
        raise RuntimeError("PPR scale sweep requires fixed generation bboxes")

    start = int(state["next_index"])
    indices = list(range(start, start + batch_size))
    state["next_index"] += batch_size
    filenames = [
        f"{index:03d}_{identity}_seed{seed}.png"
        for index, identity, seed in zip(indices, identities, seeds)
    ]
    state["filenames"].extend(filenames)

    generated: dict[float, list[Image.Image]] = {}
    scale_stats = {}
    for scale in state["scales"]:
        exact_base = scale == 0.0
        variant = f"scale_{_scale_label(scale)}"
        images, diagnostics = _generate(
            trainer,
            option="A" if exact_base else "B",
            prompts=prompts,
            seeds=seeds,
            identities=identities,
            references=references,
            reference_bboxes=reference_bboxes,
            generation_bboxes=generation_bboxes,
            sample_keys=filenames,
            runtime_settings=(
                exact_base,
                "base_outside_core",
                float(scale),
            ),
            diagnostic_variant=variant,
        )
        generated[scale] = images
        scale_stats[scale] = _processor_stats(diagnostics, batch_size)
        if exact_base:
            for sample_stats in scale_stats[scale]:
                sample_stats["active_processor_count"] = 0
        state["diagnostics"].extend(diagnostics)

    baseline_images = generated[0.0]
    for local_index, filename in enumerate(filenames):
        baseline = baseline_images[local_index]
        for scale in state["scales"]:
            _append_row(
                trainer,
                state,
                eval_metrics,
                image=generated[scale][local_index],
                baseline=baseline,
                same_scale_baseline=None,
                bbox=generation_bboxes[local_index],
                filename=filename,
                output_path=(
                    state["root"]
                    / f"scale_{_scale_label(scale)}"
                    / filename
                ),
                sample_index=indices[local_index],
                scale=scale,
                seed=seeds[local_index],
                prompt=prompts[local_index],
                reference_id=identities[local_index],
                swap_reference_id="",
                mode="exact_photomaker" if scale == 0.0 else "ppr_base_outside_core",
                processor_stats=scale_stats[scale][local_index],
            )

    swap_scale = state["swap_scale"]
    if swap_scale is not None:
        for local_index, global_index in enumerate(indices):
            if global_index not in state["swap_indices"]:
                continue
            swap_identity, swap_image, swap_bbox = _swap_source(
                state,
                identities[local_index],
            )
            variant = f"reference_swap_scale_{_scale_label(swap_scale)}"
            images, diagnostics = _generate(
                trainer,
                option="B",
                prompts=[prompts[local_index]],
                seeds=[seeds[local_index]],
                identities=[identities[local_index]],
                references=[references[local_index]],
                reference_bboxes=[reference_bboxes[local_index]],
                generation_bboxes=[generation_bboxes[local_index]],
                sample_keys=[filenames[local_index]],
                ppr_reference_image=[swap_image],
                ppr_face_bbox_ref=swap_bbox,
                runtime_settings=(False, "base_outside_core", swap_scale),
                diagnostic_variant=variant,
            )
            state["diagnostics"].extend(diagnostics)
            stats = _processor_stats(diagnostics, 1)[0]
            _append_row(
                trainer,
                state,
                eval_metrics,
                image=images[0],
                baseline=baseline_images[local_index],
                same_scale_baseline=generated[swap_scale][local_index],
                bbox=generation_bboxes[local_index],
                filename=filenames[local_index],
                output_path=state["root"] / variant / filenames[local_index],
                sample_index=global_index,
                scale=swap_scale,
                seed=seeds[local_index],
                prompt=prompts[local_index],
                reference_id=identities[local_index],
                swap_reference_id=swap_identity,
                mode="ppr_spatial_reference_swap",
                processor_stats=stats,
            )

    batch["generated"] = generated[1.0] if 1.0 in generated else baseline_images
    batch["generated_masks"] = [None] * batch_size
    return batch


def _create_contact_sheets(state, rows_per_page: int = 6) -> None:
    columns = [
        (f"{_scale_label(scale)}x", state["root"] / f"scale_{_scale_label(scale)}")
        for scale in state["scales"]
    ]
    if state["swap_scale"] is not None:
        label = _scale_label(state["swap_scale"])
        columns.append((f"{label}x swap", state["root"] / f"reference_swap_scale_{label}"))
    cell_width, cell_height, label_height = 256, 256, 24
    for page_start in range(0, len(state["filenames"]), rows_per_page):
        filenames = state["filenames"][page_start : page_start + rows_per_page]
        sheet = Image.new(
            "RGB",
            (
                cell_width * len(columns),
                label_height + (cell_height + label_height) * len(filenames),
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for column_index, (label, _) in enumerate(columns):
            draw.text((column_index * cell_width + 4, 4), label, fill="black")
        for row_index, filename in enumerate(filenames):
            y = label_height + row_index * (cell_height + label_height)
            for column_index, (_, directory) in enumerate(columns):
                path = directory / filename
                if not path.exists():
                    continue
                with Image.open(path) as image_file:
                    image = image_file.convert("RGB")
                image.thumbnail((cell_width, cell_height))
                x = column_index * cell_width + (cell_width - image.width) // 2
                sheet.paste(image, (x, y))
            draw.text((4, y + cell_height + 3), filename, fill="black")
        page_end = page_start + len(filenames) - 1
        sheet.save(
            state["root"]
            / "contact_sheets"
            / f"samples_{page_start:03d}_{page_end:03d}.jpg",
            quality=92,
        )


def _mean_std(values) -> tuple[float, float]:
    finite = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def finalize_ppr_scale_sweep(trainer) -> None:
    state = getattr(trainer, "_ppr_scale_sweep_state", None)
    if state is None:
        raise RuntimeError("PPR scale sweep produced no batches")
    root = state["root"]
    rows = sorted(
        state["rows"],
        key=lambda row: (
            int(row["sample_index"]),
            float(row["runtime_scale"]),
            str(row["ppr_mode"]),
        ),
    )
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["runtime_scale", "sample_count"]
    summary_metrics = [
        "id_similarity",
        "text_similarity",
        "whole_image_mae_vs_scale_0",
        "face_core_mae_vs_scale_0",
        "face_core_lpips_vs_scale_0",
        "applied_delta_rms_ratio",
        "cap_fraction",
        "mean_gate",
    ]
    for metric in summary_metrics:
        summary_fields.extend((f"{metric}_mean", f"{metric}_std"))
    summary_rows = []
    for scale in state["scales"]:
        scale_rows = [
            row
            for row in rows
            if float(row["runtime_scale"]) == scale
            and row["ppr_mode"] != "ppr_spatial_reference_swap"
        ]
        summary = {"runtime_scale": scale, "sample_count": len(scale_rows)}
        for metric in summary_metrics:
            mean, std = _mean_std(row[metric] for row in scale_rows)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)
    with (root / "metrics_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (root / "processor_diagnostics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in state["diagnostics"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    randomness = {}
    for record in state["diagnostics"]:
        if record.get("record_type") != "generation_randomness":
            continue
        randomness.setdefault(record["sample"], {})[record["variant"]] = (
            record["initial_latents_sha256"],
            record["reference_noise_sha256"],
        )
    expected_variants = {
        f"scale_{_scale_label(scale)}" for scale in state["scales"]
    }
    for sample, variants in randomness.items():
        missing = expected_variants - set(variants)
        if missing:
            raise RuntimeError(
                f"Missing scale-sweep randomness for {sample}: {sorted(missing)}"
            )
        baseline = variants["scale_0"]
        for variant, fingerprint in variants.items():
            if fingerprint != baseline:
                raise RuntimeError(
                    f"Scale-sweep randomness mismatch for sample={sample}, variant={variant}"
                )

    id_means = {
        float(row["runtime_scale"]): float(row["id_similarity_mean"])
        for row in summary_rows
        if np.isfinite(float(row["id_similarity_mean"]))
    }
    provisional_best = (
        max(id_means, key=id_means.get) if id_means else float("nan")
    )
    conclusion = f"""# PPR 8k residual-scale sweep conclusion

## Automatic measurements

- Provisional identity-similarity leader: `{provisional_best:g}x`
- LPIPS status: {state["lpips_status"]}
- Samples per main scale: {state["next_index"]}

## Required visual review

Review `contact_sheets/` for face/head alignment, realism, landmarks, prompt
and pose adherence, seams, duplicated features, texture corruption, and body
drift. Metrics alone must not select the deployment scale.

## Final decision

Pending visual review. Replace this paragraph with exactly one action:
retain the checkpoint with inference scaling; tune spatial application; audit
reference routing; or retrain the objective.
"""
    (root / "conclusion.md").write_text(conclusion, encoding="utf-8")
    manifest = {
        "checkpoint": str(getattr(trainer.config, "saved_checkpoint", "")),
        "validation_base": str(
            getattr(
                trainer.config,
                "pretrained_model_for_validation_name_or_path",
                "",
            )
        ),
        "scales": list(state["scales"]),
        "swap_scale": state["swap_scale"],
        "swap_indices": sorted(state["swap_indices"]),
        "sample_count": state["next_index"],
        "lpips_status": state["lpips_status"],
        "randomness_fingerprints_verified": True,
        "validation_args": OmegaConf.to_container(
            trainer.config.validation_args,
            resolve=True,
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _create_contact_sheets(state)
    print(
        "[PPR scale sweep complete] "
        f"output={root} rows={len(rows)} lpips={state['lpips_status']}"
    )
