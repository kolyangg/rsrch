"""Reference-content versus reference-noise diagnostic for NN2-PPR."""

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
    _normalize_refs,
    _per_sample,
    _pixel_mae,
)
from src.trainer.ppr_scale_sweep import _face_crop, _face_lpips
from src.utils.id_utils import analyze_faces
from src.utils.model_utils import cos_sim


VARIANTS = {
    "PM0": (0.0, "R1", "N1"),
    "R1N1": (4.0, "R1", "N1"),
    "R2N1": (4.0, "R2", "N1"),
    "R1N2": (4.0, "R1", "N2"),
    "R2N2": (4.0, "R2", "N2"),
}
HASH_FIELDS = (
    "initial_latents_sha256",
    "target_prompt_embeds_sha256",
    "target_photomaker_id_embeds_sha256",
    "reference_latents_sha256",
    "reference_mask_sha256",
    "reference_noise_sha256",
    "ref_noised_step_15_sha256",
    "ref_noised_step_25_sha256",
    "ref_noised_step_35_sha256",
)
METRIC_FIELDS = (
    "sample_index",
    "filename",
    "variant",
    "target_identity",
    "spatial_reference_identity",
    "target_seed",
    "reference_noise_seed",
    "prompt",
    "sha256",
    "pixel_mae_full_vs_PM0",
    "pixel_mae_core_vs_PM0",
    "lpips_core_vs_PM0",
    "id_similarity_original",
    "id_similarity_swapped",
    "text_similarity",
    "face_detected",
    "face_confidence",
    "landmark_displacement_vs_PM0",
    "seam_gradient_proxy_vs_PM0",
    "applied_delta_rms_ratio",
    "cap_fraction",
)


def _image_hash(image: Image.Image) -> str:
    image = image.convert("RGB")
    payload = (
        f"{image.width}x{image.height}:RGB:".encode()
        + np.asarray(image, dtype=np.uint8).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _noise_seeds(config) -> dict[str, int]:
    raw = getattr(
        config,
        "ppr_reference_noise_seeds",
        (918273, 271828),
    )
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    values = tuple(int(value) for value in raw)
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError(
            "ppr_reference_noise_seeds must contain exactly two distinct integers"
        )
    return {"N1": values[0], "N2": values[1]}


def _initialize_state(trainer) -> dict[str, Any]:
    root = Path(
        str(trainer.config.ppr_reference_noise_output_dir)
    ).expanduser().resolve()
    overwrite = bool(
        getattr(trainer.config, "ppr_reference_noise_overwrite", False)
    )
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Reference/noise diagnostic output exists: {root}. "
                "Set ppr_reference_noise_overwrite=true to replace it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name in VARIANTS:
        (root / name).mkdir()
    for name in ("contact_sheets", "difference_heatmaps", "face_crops"):
        (root / name).mkdir()

    if len(trainer.evaluation_dataloaders) != 1:
        raise RuntimeError("Reference/noise test requires one validation dataset")
    dataloader = next(iter(trainer.evaluation_dataloaders.values()))
    # Accelerate's DataLoaderShard does not reliably preserve the source
    # DataLoader.batch_size metadata. The effective batch is checked from the
    # actual prompts in run_ppr_reference_noise_batch instead.
    dataset = dataloader.dataset
    sources = []
    for image_path in getattr(dataset, "images", ()):
        identity = image_path.stem
        bbox = getattr(dataset, "_bbox_map_ref", {}).get(identity)
        sources.append((identity, Path(image_path), bbox))
    identities = [item[0] for item in sources]
    if len(set(identities)) < 2:
        raise RuntimeError("Reference/noise test requires at least two identities")
    # Deterministic cyclic permutation, independent of dataloader iteration.
    swap_map = {
        identity: sources[(index + 1) % len(sources)]
        for index, identity in enumerate(identities)
    }
    state = {
        "root": root,
        "noise_seeds": _noise_seeds(trainer.config),
        "swap_map": swap_map,
        "rows": [],
        "pair_rows": [],
        "tensor_rows": [],
        "integrity": {},
        "filenames": [],
        "next_index": 0,
        "device": trainer.device,
        "lpips_status": "not attempted",
    }
    trainer._ppr_reference_noise_state = state
    print(
        "[PPR reference/noise] "
        f"output={root} noise_seeds={state['noise_seeds']} samples={len(dataset)}"
    )
    return state


def _processor_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    selected = [
        record
        for record in records
        if record.get("record_type") == "processor_applied_ratio"
    ]
    if not selected:
        raise RuntimeError("No PPR applied-residual diagnostics were recorded")
    ratios = [
        float(value)
        for record in selected
        for value in record.get("applied_ratios", ())
    ]
    cap_scales = [
        float(value)
        for record in selected
        for value in record.get("cap_scales", ())
    ]
    return {
        "applied_delta_rms_ratio": float(np.mean(ratios)),
        "cap_fraction": float(
            np.mean(np.asarray(cap_scales) < (1.0 - 1e-7))
        ),
    }


def _randomness_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("record_type") == "generation_randomness"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one randomness record for batch_size=1, got {len(matches)}"
        )
    return dict(matches[0])


def _face_observation(trainer, image: Image.Image) -> dict[str, Any]:
    metric = next(
        (
            candidate
            for candidate in trainer.metrics
            if candidate.__class__.__name__ == "IDSimMax"
        ),
        None,
    )
    if metric is None:
        metric = next(
            (
                candidate
                for candidate in trainer.metrics
                if hasattr(candidate, "aligner")
            ),
            None,
        )
    if metric is None:
        return {
            "face_detected": 0,
            "face_confidence": float("nan"),
            "landmarks": None,
            "embedding": None,
        }
    array = np.asarray(image.convert("RGB"))[:, :, ::-1]
    faces = analyze_faces(metric.aligner.face_detector, array)
    if not faces:
        return {
            "face_detected": 0,
            "face_confidence": float("nan"),
            "landmarks": None,
            "embedding": None,
        }
    face = max(
        faces,
        key=lambda item: float(
            (item["bbox"][2] - item["bbox"][0])
            * (item["bbox"][3] - item["bbox"][1])
        ),
    )
    score = face.get("det_score", float("nan"))
    landmarks = face.get("kps")
    return {
        "face_detected": 1,
        "face_confidence": float(score),
        "landmarks": (
            None
            if landmarks is None
            else np.asarray(landmarks, dtype=np.float32)
        ),
        "embedding": np.asarray(face["embedding"], dtype=np.float32),
    }


def _identity_text_scores(
    trainer,
    *,
    image: Image.Image,
    prompt: str,
    original_identity: str,
    swapped_identity: str,
    observation: dict[str, Any],
) -> dict[str, float]:
    values = {
        "id_similarity_original": float("nan"),
        "id_similarity_swapped": float("nan"),
        "text_similarity": float("nan"),
    }
    id_metric = next(
        (
            metric
            for metric in trainer.metrics
            if metric.__class__.__name__ == "IDSimMax"
        ),
        None,
    )
    if id_metric is None:
        id_metric = next(
            (
                metric
                for metric in trainer.metrics
                if hasattr(metric, "id_embeds")
            ),
            None,
        )
    embedding = observation.get("embedding")
    if id_metric is not None and embedding is not None:
        for field, identity in (
            ("id_similarity_original", original_identity),
            ("id_similarity_swapped", swapped_identity),
        ):
            if identity in id_metric.id_embeds:
                values[field] = float(
                    cos_sim(embedding, id_metric.id_embeds[identity])
                )
    text_metric = next(
        (
            metric
            for metric in trainer.metrics
            if metric.__class__.__name__ == "TextSimMetric"
        ),
        None,
    )
    if text_metric is not None:
        result = text_metric(
            generated=[image],
            prompt=prompt,
            id=original_identity,
        )
        if "text_sim" in result:
            values["text_similarity"] = float(result["text_sim"])
    return values


def _landmark_displacement(current, baseline, image: Image.Image) -> float:
    if current is None or baseline is None or current.shape != baseline.shape:
        return float("nan")
    diagonal = math.hypot(image.width, image.height)
    return float(np.linalg.norm(current - baseline, axis=1).mean() / diagonal)


def _seam_proxy(image: Image.Image, baseline: Image.Image, bbox) -> float:
    """Boundary-gradient discrepancy; a transparent seam/artifact proxy."""
    if bbox is None or len(bbox) != 4:
        return float("nan")
    current = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    base = np.asarray(baseline.convert("RGB"), dtype=np.float32) / 255.0
    height, width = current.shape[:2]
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0, x1 = np.clip((x0, x1), 1, width - 1)
    y0, y1 = np.clip((y0, y1), 1, height - 1)
    ring = np.zeros((height, width), dtype=bool)
    ring[max(0, y0 - 2):min(height, y0 + 3), x0:x1] = True
    ring[max(0, y1 - 2):min(height, y1 + 3), x0:x1] = True
    ring[y0:y1, max(0, x0 - 2):min(width, x0 + 3)] = True
    ring[y0:y1, max(0, x1 - 2):min(width, x1 + 3)] = True
    if not ring.any():
        return float("nan")

    def gradient(value):
        gx = np.diff(value, axis=1, append=value[:, -1:])
        gy = np.diff(value, axis=0, append=value[-1:, :])
        return np.sqrt(gx * gx + gy * gy).mean(axis=2)

    return float(np.abs(gradient(current) - gradient(base))[ring].mean())


def _save_heatmap(path: Path, image: Image.Image, baseline: Image.Image) -> None:
    current = np.asarray(image.convert("RGB"), dtype=np.float32)
    base = np.asarray(baseline.convert("RGB"), dtype=np.float32)
    difference = np.abs(current - base).mean(axis=2)
    scale = max(float(np.percentile(difference, 99)), 1.0)
    normalized = np.clip(difference / scale, 0.0, 1.0)
    heatmap = np.stack(
        (
            normalized,
            np.sqrt(normalized) * 0.35,
            np.zeros_like(normalized),
        ),
        axis=2,
    )
    Image.fromarray(np.uint8(255.0 * heatmap)).save(path)


def _relative_signature(left: dict, right: dict) -> float:
    left_values = np.asarray(left["sketch"], dtype=np.float64)
    right_values = np.asarray(right["sketch"], dtype=np.float64)
    length = min(left_values.size, right_values.size)
    if length == 0:
        return float("nan")
    left_values, right_values = left_values[:length], right_values[:length]
    return float(
        np.sqrt(np.mean((left_values - right_values) ** 2))
        / (np.sqrt(np.mean(left_values ** 2)) + 1e-12)
    )


def _tensor_comparisons(
    sample: str,
    diagnostics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    comparisons = (
        ("reference_content", "R1N1", "R2N1"),
        ("reference_noise", "R1N1", "R1N2"),
    )
    for effect, left_name, right_name in comparisons:
        left_records = {
            (
                record["record_type"],
                int(record["step"]),
                str(record.get("processor", "")),
            ): record
            for record in diagnostics[left_name]
            if record.get("record_type") in {
                "processor_tensor_signature",
                "epsilon_tensor_signature",
            }
        }
        right_records = {
            (
                record["record_type"],
                int(record["step"]),
                str(record.get("processor", "")),
            ): record
            for record in diagnostics[right_name]
            if record.get("record_type") in {
                "processor_tensor_signature",
                "epsilon_tensor_signature",
            }
        }
        if set(left_records) != set(right_records):
            raise RuntimeError(
                f"Tensor-stage mismatch for {sample} {effect}"
            )
        for key in sorted(left_records):
            left, right = left_records[key], right_records[key]
            stages = (
                (
                    "reference_hidden",
                    "reference_candidate",
                    "connector_down",
                    "raw_delta",
                    "bounded_delta",
                    "applied_delta",
                )
                if key[0] == "processor_tensor_signature"
                else (
                    "target_epsilon_pre_anchor",
                    "target_epsilon_post_anchor",
                )
            )
            for stage in stages:
                output.append(
                    {
                        "sample": sample,
                        "effect": effect,
                        "left_variant": left_name,
                        "right_variant": right_name,
                        "step": key[1],
                        "processor": key[2],
                        "stage": stage,
                        "relative_difference": _relative_signature(
                            left[stage],
                            right[stage],
                        ),
                        "left_sha256": left[stage]["sha256"],
                        "right_sha256": right[stage]["sha256"],
                        "capture_kind": "exact_hash_rms_plus_deterministic_512_sketch",
                    }
                )
    return output


def _assert_integrity(
    sample: str,
    fingerprints: dict[str, dict[str, Any]],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> None:
    target_fields = (
        "initial_latents_sha256",
        "target_prompt_embeds_sha256",
        "target_photomaker_id_embeds_sha256",
    )
    for field in target_fields:
        values = {fingerprints[name].get(field) for name in VARIANTS}
        if None in values or len(values) != 1:
            raise RuntimeError(
                f"{sample}: target invariant failed for {field}: {values}"
            )
    for name, fingerprint in fingerprints.items():
        if not bool(fingerprint.get("reference_mask_nonempty", False)):
            raise RuntimeError(f"{sample}: empty reference mask in {name}")
        missing = [field for field in HASH_FIELDS if field not in fingerprint]
        if missing:
            raise RuntimeError(
                f"{sample}: missing fingerprints in {name}: {missing}"
            )

    def equal(left, right, fields):
        return all(
            fingerprints[left][field] == fingerprints[right][field]
            for field in fields
        )

    if not equal("R1N1", "R2N1", ("reference_noise_sha256",)):
        raise RuntimeError(f"{sample}: R1N1/R2N1 reference noise changed")
    for field in (
        "spatial_reference_image_sha256",
        "reference_latents_sha256",
    ):
        if equal("R1N1", "R2N1", (field,)):
            raise RuntimeError(
                f"{sample}: R1/R2 spatial-reference swap did not change {field}"
            )
    if not equal(
        "R1N1",
        "R1N2",
        (
            "spatial_reference_image_sha256",
            "reference_latents_sha256",
            "reference_mask_sha256",
        ),
    ):
        raise RuntimeError(f"{sample}: R1N1/R1N2 reference content changed")
    if equal("R1N1", "R1N2", ("reference_noise_sha256",)):
        raise RuntimeError(f"{sample}: N1/N2 reference noise did not change")
    for field in (
        "ref_noised_step_15_sha256",
        "ref_noised_step_25_sha256",
        "ref_noised_step_35_sha256",
    ):
        if equal("R1N1", "R2N1", (field,)):
            raise RuntimeError(
                f"{sample}: reference swap did not change {field}"
            )
        if equal("R1N1", "R1N2", (field,)):
            raise RuntimeError(
                f"{sample}: reference-noise swap did not change {field}"
            )

    pm_epsilon = [
        record
        for record in diagnostics["PM0"]
        if record.get("record_type") == "epsilon_ratio"
    ]
    if not pm_epsilon or any(
        record.get("output_control") != "diagnostic-force-base"
        for record in pm_epsilon
    ):
        raise RuntimeError(
            f"{sample}: PM0 was not routed through exact ordinary PhotoMaker output"
        )

    for name in ("R1N1", "R2N1", "R1N2", "R2N2"):
        tensor_records = [
            record
            for record in diagnostics[name]
            if record.get("record_type") == "processor_tensor_signature"
        ]
        if not tensor_records or any(
            int(record.get("roi_tokens", 0)) <= 0 for record in tensor_records
        ):
            raise RuntimeError(f"{sample}: invalid packed ROI in {name}")
        if _processor_stats(diagnostics[name])["applied_delta_rms_ratio"] <= 0:
            raise RuntimeError(f"{sample}: zero applied PPR residual in {name}")


@torch.no_grad()
def run_ppr_reference_noise_batch(trainer, batch, eval_metrics):
    state = getattr(trainer, "_ppr_reference_noise_state", None)
    if state is None:
        state = _initialize_state(trainer)
    prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
    if len(prompts) != 1:
        raise RuntimeError("Reference/noise test expects batch_size=1")
    identity = str(_per_sample(batch.get("id"), 1)[0])
    target_seed = int(
        _per_sample(
            batch.get("seed", trainer.config.validation_args.get("seed", 0)),
            1,
        )[0]
    )
    references = _normalize_refs(batch.get("ref_images"), 1)
    ref_bbox = _per_sample(batch.get("face_bbox_ref"), 1)[0]
    gen_bbox = _per_sample(batch.get("face_bbox_gen"), 1)[0]
    if ref_bbox is None or gen_bbox is None:
        raise RuntimeError("Reference/noise test requires fixed ref/gen bboxes")
    swap_identity, swap_path, swap_bbox = state["swap_map"][identity]
    swap_image = Image.open(swap_path).convert("RGB")
    sample_index = int(state["next_index"])
    state["next_index"] += 1
    filename = f"{sample_index:03d}_{identity}_seed{target_seed}.png"
    state["filenames"].append(filename)

    images: dict[str, Image.Image] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, (scale, reference_kind, noise_kind) in VARIANTS.items():
        use_swap = reference_kind == "R2"
        variant_images, records = _generate(
            trainer,
            option="A" if name == "PM0" else "B",
            prompts=prompts,
            seeds=[target_seed],
            identities=[identity],
            references=references,
            reference_bboxes=[ref_bbox],
            generation_bboxes=[gen_bbox],
            sample_keys=[filename],
            ppr_reference_image=[swap_image] if use_swap else None,
            ppr_face_bbox_ref=swap_bbox if use_swap else None,
            ppr_reference_noise_seed=state["noise_seeds"][noise_kind],
            runtime_settings=(
                name == "PM0",
                "base_outside_core",
                scale,
            ),
            diagnostic_variant=name,
            capture_tensor_signatures=name != "PM0",
        )
        images[name] = variant_images[0]
        diagnostics[name] = records
        fingerprint = _randomness_record(records)
        reference_image = swap_image if use_swap else references[0][0]
        fingerprint["spatial_reference_image_sha256"] = _image_hash(
            reference_image
        )
        fingerprints[name] = fingerprint

    _assert_integrity(filename, fingerprints, diagnostics)
    state["tensor_rows"].extend(
        _tensor_comparisons(filename, diagnostics)
    )
    state["integrity"][filename] = fingerprints

    baseline = images["PM0"]
    baseline_face = _face_observation(trainer, baseline)
    observations = {
        name: _face_observation(trainer, image)
        for name, image in images.items()
    }
    variant_rows = {}
    for name, image in images.items():
        output_path = state["root"] / name / filename
        image.save(output_path)
        image_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        full_mae, core_mae = _pixel_mae(image, baseline, gen_bbox)
        observation = observations[name]
        score_values = _identity_text_scores(
            trainer,
            image=image,
            prompt=prompts[0],
            original_identity=identity,
            swapped_identity=swap_identity,
            observation=observation,
        )
        stats = (
            {"applied_delta_rms_ratio": 0.0, "cap_fraction": 0.0}
            if name == "PM0"
            else _processor_stats(diagnostics[name])
        )
        state["rows"].append(
            {
                "sample_index": sample_index,
                "filename": filename,
                "variant": name,
                "target_identity": identity,
                "spatial_reference_identity": (
                    swap_identity if "R2" in name else identity
                ),
                "target_seed": target_seed,
                "reference_noise_seed": state["noise_seeds"][
                    VARIANTS[name][2]
                ],
                "prompt": prompts[0],
                "sha256": image_sha,
                "pixel_mae_full_vs_PM0": full_mae,
                "pixel_mae_core_vs_PM0": core_mae,
                "lpips_core_vs_PM0": _face_lpips(
                    state, image, baseline, gen_bbox
                ),
                **score_values,
                "face_detected": observation["face_detected"],
                "face_confidence": observation["face_confidence"],
                "landmark_displacement_vs_PM0": _landmark_displacement(
                    observation["landmarks"],
                    baseline_face["landmarks"],
                    image,
                ),
                "seam_gradient_proxy_vs_PM0": _seam_proxy(
                    image, baseline, gen_bbox
                ),
                **stats,
            }
        )
        variant_rows[name] = state["rows"][-1]
        if name != "PM0":
            _save_heatmap(
                state["root"]
                / "difference_heatmaps"
                / f"{name}_{filename}",
                image,
                baseline,
            )
        crop = _face_crop(image, gen_bbox)
        if crop is not None:
            crop.save(
                state["root"] / "face_crops" / f"{name}_{filename}"
            )
        for metric_name in (
            "id_similarity_original",
            "id_similarity_swapped",
            "text_similarity",
        ):
            value = state["rows"][-1][metric_name]
            if np.isfinite(value):
                eval_metrics.update(f"{name}/{metric_name}", value)

    pair_definitions = (
        ("reference_content_N1", "R1N1", "R2N1", "reference_image_effect"),
        ("reference_content_N2", "R1N2", "R2N2", "reference_image_effect"),
        ("reference_noise_R1", "R1N1", "R1N2", "reference_noise_effect"),
        ("reference_noise_R2", "R2N1", "R2N2", "reference_noise_effect"),
    )
    for pair, left, right, effect in pair_definitions:
        full, core = _pixel_mae(images[left], images[right], gen_bbox)
        left_row, right_row = variant_rows[left], variant_rows[right]
        state["pair_rows"].append(
            {
                "sample_index": sample_index,
                "filename": filename,
                "pair": pair,
                "effect": effect,
                "left_variant": left,
                "right_variant": right,
                "pixel_mae_full": full,
                "pixel_mae_core": core,
                "lpips_core": _face_lpips(
                    state, images[left], images[right], gen_bbox
                ),
                "id_original_abs_difference": abs(
                    float(left_row["id_similarity_original"])
                    - float(right_row["id_similarity_original"])
                ),
                "id_swapped_abs_difference": abs(
                    float(left_row["id_similarity_swapped"])
                    - float(right_row["id_similarity_swapped"])
                ),
                "text_abs_difference": abs(
                    float(left_row["text_similarity"])
                    - float(right_row["text_similarity"])
                ),
                "face_confidence_abs_difference": abs(
                    float(left_row["face_confidence"])
                    - float(right_row["face_confidence"])
                ),
                "landmark_pair_displacement": _landmark_displacement(
                    observations[left]["landmarks"],
                    observations[right]["landmarks"],
                    images[left],
                ),
                "seam_gradient_pair_proxy": _seam_proxy(
                    images[left], images[right], gen_bbox
                ),
            }
        )

    batch["generated"] = [images["R1N1"]]
    batch["generated_masks"] = [None]
    return batch


def _bootstrap_ci(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.asarray(
        [
            rng.choice(finite, size=finite.size, replace=True).mean()
            for _ in range(2000)
        ]
    )
    return (
        float(finite.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _create_contact_sheets(state, rows_per_page: int = 6) -> None:
    cell, label = 256, 24
    names = list(VARIANTS)
    for start in range(0, len(state["filenames"]), rows_per_page):
        filenames = state["filenames"][start:start + rows_per_page]
        sheet = Image.new(
            "RGB",
            (cell * len(names), label + (cell + label) * len(filenames)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for column, name in enumerate(names):
            draw.text((column * cell + 4, 4), name, fill="black")
        for row, filename in enumerate(filenames):
            y = label + row * (cell + label)
            for column, name in enumerate(names):
                with Image.open(state["root"] / name / filename) as source:
                    image = source.convert("RGB")
                image.thumbnail((cell, cell))
                sheet.paste(
                    image,
                    (
                        column * cell + (cell - image.width) // 2,
                        y + (cell - image.height) // 2,
                    ),
                )
            draw.text((4, y + cell + 3), filename, fill="black")
        sheet.save(
            state["root"]
            / "contact_sheets"
            / f"samples_{start:03d}_{start + len(filenames) - 1:03d}.jpg",
            quality=92,
        )


def finalize_ppr_reference_noise(trainer) -> None:
    state = getattr(trainer, "_ppr_reference_noise_state", None)
    if state is None:
        raise RuntimeError("Reference/noise diagnostic produced no batches")
    root = state["root"]
    with (root / "metrics_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(state["rows"])
    pair_fields = (
        "sample_index", "filename", "pair", "effect", "left_variant",
        "right_variant", "pixel_mae_full", "pixel_mae_core", "lpips_core",
        "id_original_abs_difference", "id_swapped_abs_difference",
        "text_abs_difference", "face_confidence_abs_difference",
        "landmark_pair_displacement", "seam_gradient_pair_proxy",
    )
    with (root / "paired_effects.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(state["pair_rows"])

    summary = []
    for effect in ("reference_image_effect", "reference_noise_effect"):
        effect_rows = [
            row for row in state["pair_rows"] if row["effect"] == effect
        ]
        for metric in (
            "pixel_mae_full",
            "pixel_mae_core",
            "lpips_core",
            "id_original_abs_difference",
            "id_swapped_abs_difference",
            "text_abs_difference",
            "face_confidence_abs_difference",
            "landmark_pair_displacement",
            "seam_gradient_pair_proxy",
        ):
            mean, low, high = _bootstrap_ci(
                [row[metric] for row in effect_rows]
            )
            summary.append(
                {
                    "effect": effect,
                    "metric": metric,
                    "mean": mean,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "pair_count": len(effect_rows),
                }
            )
    with (root / "metrics_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with (root / "tensor_diagnostics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in state["tensor_rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metric_lookup = {
        (row["effect"], row["metric"]): float(row["mean"])
        for row in summary
    }
    reference_effect = metric_lookup[
        ("reference_image_effect", "pixel_mae_core")
    ]
    noise_effect = metric_lookup[
        ("reference_noise_effect", "pixel_mae_core")
    ]
    if reference_effect > 1.25 * noise_effect:
        automatic = "primarily reference-content-driven"
    elif noise_effect > 1.25 * reference_effect:
        automatic = "primarily reference-noise-driven"
    else:
        automatic = "mixed or generic/target-conditioned"
    stage_rows = [
        row
        for row in state["tensor_rows"]
        if row["effect"] == "reference_content"
    ]
    stage_means = {}
    for stage in dict.fromkeys(row["stage"] for row in stage_rows):
        values = [
            row["relative_difference"]
            for row in stage_rows
            if row["stage"] == stage
        ]
        stage_means[stage] = float(np.nanmean(values))
    first_low = next(
        (
            stage
            for stage in (
                "reference_hidden",
                "reference_candidate",
                "connector_down",
                "raw_delta",
                "bounded_delta",
                "applied_delta",
                "target_epsilon_pre_anchor",
                "target_epsilon_post_anchor",
            )
            if stage_means.get(stage, float("inf")) < 1e-3
        ),
        "none detected automatically",
    )
    conclusion = f"""# PPR 8k reference-content versus reference-noise conclusion

## Automatic result

- Classification from face-core pixel effects: **{automatic}**
- Mean reference-image effect: `{reference_effect:.8f}`
- Mean reference-noise effect: `{noise_effect:.8f}`
- First tensor stage with mean swap sensitivity below `1e-3`: `{first_low}`
- LPIPS status: {state["lpips_status"]}

This classification is provisional. Review `contact_sheets/`, `face_crops/`,
and `difference_heatmaps/`, then compare identity-to-original and
identity-to-swapped columns in `metrics_per_image.csv`.

The reported seam score is a bbox-boundary gradient-discrepancy proxy, not a
learned perceptual artifact detector. Tensor differences use exact tensor
SHA-256/RMS plus the same deterministic 512-value sketch at each paired stage.
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
        "sample_count": int(state["next_index"]),
        "variants": VARIANTS,
        "reference_noise_seeds": state["noise_seeds"],
        "integrity_assertions_passed": True,
        "tensor_capture": "exact SHA-256/RMS plus deterministic 512-value sketch",
        "lpips_status": state["lpips_status"],
        "validation_args": OmegaConf.to_container(
            trainer.config.validation_args,
            resolve=True,
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "integrity_hashes.json").write_text(
        json.dumps(state["integrity"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _create_contact_sheets(state)
    print(
        "[PPR reference/noise complete] "
        f"output={root} samples={state['next_index']} result={automatic}"
    )
