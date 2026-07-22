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


VARIANT_NAMES = ("PM0", "R1N1", "R2N1", "R1N2", "R2N2")


def _variants(scale: float):
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("ppr_reference_noise_scale must be finite and non-negative")
    return {
        "PM0": (0.0, "R1", "N1"),
        "R1N1": (scale, "R1", "N1"),
        "R2N1": (scale, "R2", "N1"),
        "R1N2": (scale, "R1", "N2"),
        "R2N2": (scale, "R2", "N2"),
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
    "reference_ca_prompt_sha256",
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


def _reference_ca_mode(config) -> str:
    mode = str(
        getattr(config, "ppr_reference_ca_mode", "original") or "original"
    ).lower()
    if mode not in {"original", "zero"}:
        raise ValueError(
            "ppr_reference_ca_mode must be one of: original, zero"
        )
    return mode


def _effective_reference_ca_mode(config) -> str:
    """Describe the CA tensor after architecture and diagnostic controls."""
    diagnostic_mode = _reference_ca_mode(config)
    model_config = getattr(config, "model", None)
    token_mode = str(
        getattr(model_config, "ba_reference_token_text_mode", "original")
        or "original"
    ).lower()
    if token_mode not in {"original", "zero"}:
        raise ValueError(
            "model.ba_reference_token_text_mode must be original or zero"
        )
    if token_mode == "original":
        return diagnostic_mode
    return (
        token_mode
        if diagnostic_mode == "original"
        else f"{token_mode}+ppr_{diagnostic_mode}"
    )


def _initialize_state(trainer) -> dict[str, Any]:
    # Missing key preserves NN2-NN4's historical 4x diagnostic. NN5 configs
    # opt into the approval-scale value explicitly.
    scale = float(getattr(trainer.config, "ppr_reference_noise_scale", 4.0))
    variants = _variants(scale)
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
    for name in variants:
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
        "scale": scale,
        "variants": variants,
        "noise_seeds": _noise_seeds(trainer.config),
        "reference_ca_override": _reference_ca_mode(trainer.config),
        "reference_ca_mode": _effective_reference_ca_mode(trainer.config),
        "swap_map": swap_map,
        "rows": [],
        "pair_rows": [],
        "tensor_rows": [],
        "integrity": {},
        "filenames": [],
        "next_index": 0,
        "device": trainer.device,
        "lpips_status": "not attempted",
        "observed_batch_sizes": [],
        "identity_token_lane": bool(
            getattr(
                getattr(trainer.config, "model", None),
                "ba_identity_token_lane",
                False,
            )
        ),
        "identity_fusion_mode": str(
            getattr(
                getattr(trainer.config, "model", None),
                "ba_identity_fusion_mode",
                "blend",
            )
            or "blend"
        ).lower(),
        "identity_noise_tolerance": float(
            getattr(trainer.config, "ppr_identity_noise_tolerance", 0.0)
        ),
        "identity_noise_invariant": {},
    }
    trainer._ppr_reference_noise_state = state
    print(
        "[PPR reference/noise] "
        f"output={root} noise_seeds={state['noise_seeds']} "
        f"reference_ca={state['reference_ca_mode']} "
        f"diagnostic_override={state['reference_ca_override']} "
        f"samples={len(dataset)}"
    )
    return state


def _processor_stats(
    records: list[dict[str, Any]],
    sample: str | None = None,
) -> dict[str, float]:
    selected = [
        record
        for record in records
        if record.get("record_type") == "processor_applied_ratio"
    ]
    if not selected:
        raise RuntimeError("No PPR applied-residual diagnostics were recorded")
    ratios, cap_scales = [], []
    for record in selected:
        record_ratios = list(record.get("applied_ratios", ()))
        record_caps = list(record.get("cap_scales", ()))
        samples = list(record.get("samples", ()))
        if sample is not None and samples:
            if sample not in samples:
                continue
            sample_index = samples.index(sample)
            # Processor target rows are CFG ordered [uncond B, cond B].
            if len(record_ratios) == 2 * len(samples):
                sample_index += len(samples)
            record_ratios = [record_ratios[sample_index]]
            record_caps = [record_caps[sample_index]]
        ratios.extend(float(value) for value in record_ratios)
        cap_scales.extend(float(value) for value in record_caps)
    if not ratios or not cap_scales:
        raise RuntimeError(
            f"No PPR processor statistics found for sample={sample!r}"
        )
    return {
        "applied_delta_rms_ratio": float(np.mean(ratios)),
        "cap_fraction": float(
            np.mean(np.asarray(cap_scales) < (1.0 - 1e-7))
        ),
    }


def _randomness_record(
    records: list[dict[str, Any]],
    sample: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("record_type") == "generation_randomness"
        and record.get("sample") == sample
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one randomness record for {sample}, got {len(matches)}"
        )
    return dict(matches[0])


def _sample_records(
    records: list[dict[str, Any]],
    sample: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if (
            record.get("sample") == sample
            or (
                "sample" not in record
                and sample in record.get("samples", ())
            )
        )
    ]


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
            if key[0] == "processor_tensor_signature":
                stages = []
                for optional_stage in (
                    "reference_hidden",
                    "clean_spatial_patch_tokens",
                    "reference_candidate",
                    "connector_input",
                    "connector_down",
                    "raw_delta",
                    "bounded_delta",
                    "applied_delta",
                    "identity_candidate",
                    "identity_null_candidate",
                    "identity_connector_input",
                    "identity_raw_delta",
                    "identity_bounded_delta",
                    "identity_applied_delta",
                    "spatial_reference_candidate",
                    "spatial_candidate",
                    "spatial_null_candidate",
                    "spatial_connector_input",
                    "spatial_raw_delta",
                    "spatial_bounded_delta",
                    "spatial_applied_delta",
                    "combined_applied_delta",
                ):
                    if optional_stage in left and optional_stage in right:
                        stages.append(optional_stage)
            else:
                stages = [
                    "target_epsilon_pre_anchor",
                    "target_epsilon_post_anchor",
                ]
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
    reference_ca_mode: str,
    identity_token_lane: bool = False,
    identity_fusion_mode: str = "blend",
    identity_noise_tolerance: float = 0.0,
    spatial_memory_mode: str = "reference_unet",
) -> None:
    target_fields = (
        "initial_latents_sha256",
        "target_prompt_embeds_sha256",
        "target_photomaker_id_embeds_sha256",
    )
    for field in target_fields:
        values = {fingerprints[name].get(field) for name in VARIANT_NAMES}
        if None in values or len(values) != 1:
            raise RuntimeError(
                f"{sample}: target invariant failed for {field}: {values}"
            )
    for name, fingerprint in fingerprints.items():
        if not bool(fingerprint.get("reference_mask_nonempty", False)):
            raise RuntimeError(f"{sample}: empty reference mask in {name}")
        required_fields = list(HASH_FIELDS)
        if identity_token_lane:
            required_fields.append("spatial_identity_tokens_sha256")
        if spatial_memory_mode == "clean_clip_patches":
            required_fields.append("clean_spatial_patch_tokens_sha256")
        missing = [
            field
            for field in required_fields
            if field not in fingerprint
        ]
        if missing:
            raise RuntimeError(
                f"{sample}: missing fingerprints in {name}: {missing}"
            )
        if fingerprint.get("reference_ca_mode") != reference_ca_mode:
            raise RuntimeError(
                f"{sample}: reference CA mode mismatch in {name}: "
                f"{fingerprint.get('reference_ca_mode')!r}"
            )
        if reference_ca_mode != "original":
            if int(fingerprint.get("reference_ca_prompt_nonzero_count", -1)) != 0:
                raise RuntimeError(
                    f"{sample}: neutral reference CA is nonzero in {name}"
                )
            if float(fingerprint.get("reference_ca_prompt_rms", -1.0)) != 0.0:
                raise RuntimeError(
                    f"{sample}: neutral reference CA has nonzero RMS in {name}"
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
    if identity_token_lane:
        token_field = ("spatial_identity_tokens_sha256",)
        if equal("R1N1", "R2N1", token_field):
            raise RuntimeError(
                f"{sample}: R1/R2 swap did not change identity tokens"
            )
        if not equal("R1N1", "R1N2", token_field):
            raise RuntimeError(
                f"{sample}: reference-noise swap changed identity tokens"
            )
        if not equal("R2N1", "R2N2", token_field):
            raise RuntimeError(
                f"{sample}: R2 reference-noise swap changed identity tokens"
            )
        if equal("R1N2", "R2N2", token_field):
            raise RuntimeError(
                f"{sample}: N2 R1/R2 swap did not change identity tokens"
            )
    if spatial_memory_mode == "clean_clip_patches":
        patch_field = ("clean_spatial_patch_tokens_sha256",)
        if equal("R1N1", "R2N1", patch_field):
            raise RuntimeError(
                f"{sample}: R1/R2 swap did not change clean spatial patches"
            )
        for left_name, right_name in (("R1N1", "R1N2"), ("R2N1", "R2N2")):
            if not equal(left_name, right_name, patch_field):
                raise RuntimeError(
                    f"{sample}: reference noise changed clean spatial patches in "
                    f"{left_name}/{right_name}"
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
        if (
            _processor_stats(diagnostics[name], sample)[
                "applied_delta_rms_ratio"
            ]
            <= 0
        ):
            raise RuntimeError(f"{sample}: zero applied PPR residual in {name}")

    if identity_fusion_mode == "identity_only":
        stage_names = (
            "identity_candidate",
            "identity_null_candidate",
            "identity_connector_input",
            "identity_raw_delta",
            "identity_bounded_delta",
            "identity_applied_delta",
            "combined_applied_delta",
        )
        for left_name, right_name in (("R1N1", "R1N2"), ("R2N1", "R2N2")):
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
                    f"{sample}: identity-only noise tensor-stage mismatch for "
                    f"{left_name}/{right_name}"
                )
            for key in sorted(left_records):
                left, right = left_records[key], right_records[key]
                fields = (
                    stage_names
                    if key[0] == "processor_tensor_signature"
                    else (
                        "target_epsilon_pre_anchor",
                        "target_epsilon_post_anchor",
                    )
                )
                for field in fields:
                    if field not in left or field not in right:
                        if key[0] == "processor_tensor_signature":
                            raise RuntimeError(
                                f"{sample}: identity-only diagnostic lacks {field}"
                            )
                        continue
                    exact = left[field]["sha256"] == right[field]["sha256"]
                    relative = _relative_signature(left[field], right[field])
                    tolerance = float(identity_noise_tolerance)
                    violates_invariant = (
                        (not exact)
                        if tolerance <= 0.0
                        else ((not exact) and relative > tolerance)
                    )
                    if violates_invariant:
                        raise RuntimeError(
                            f"{sample}: identity-only reference-noise leak at "
                            f"{field} {left_name}/{right_name}: "
                            f"exact={exact}, relative={relative}, "
                            f"tolerance={tolerance}"
                        )


@torch.no_grad()
def run_ppr_reference_noise_batch(trainer, batch, eval_metrics):
    state = getattr(trainer, "_ppr_reference_noise_state", None)
    if state is None:
        state = _initialize_state(trainer)
    if "variants" not in state:
        state["scale"] = float(state.get("scale", 4.0))
        state["variants"] = _variants(state["scale"])
    state.setdefault(
        "identity_token_lane",
        bool(
            getattr(
                getattr(trainer.config, "model", None),
                "ba_identity_token_lane",
                False,
            )
        ),
    )
    state.setdefault(
        "identity_fusion_mode",
        str(
            getattr(
                getattr(trainer.config, "model", None),
                "ba_identity_fusion_mode",
                "blend",
            )
            or "blend"
        ).lower(),
    )
    state.setdefault(
        "identity_noise_tolerance",
        float(getattr(trainer.config, "ppr_identity_noise_tolerance", 0.0)),
    )
    state.setdefault("identity_noise_invariant", {})
    state.setdefault(
        "spatial_memory_mode",
        str(
            getattr(
                getattr(trainer.config, "model", None),
                "ba_spatial_memory_mode",
                "reference_unet",
            )
            or "reference_unet"
        ).lower(),
    )
    prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
    batch_size = len(prompts)
    state["observed_batch_sizes"].append(batch_size)
    identities = [
        str(value)
        for value in _per_sample(batch.get("id"), batch_size)
    ]
    target_seeds = [
        int(value)
        for value in _per_sample(
            batch.get("seed", trainer.config.validation_args.get("seed", 0)),
            batch_size,
        )
    ]
    references = _normalize_refs(batch.get("ref_images"), batch_size)
    reference_bboxes = _per_sample(
        batch.get("face_bbox_ref"), batch_size
    )
    generation_bboxes = _per_sample(
        batch.get("face_bbox_gen"), batch_size
    )
    if any(
        bbox is None
        for bbox in (*reference_bboxes, *generation_bboxes)
    ):
        raise RuntimeError("Reference/noise test requires fixed ref/gen bboxes")

    swap_identities, swap_images, swap_bboxes = [], [], []
    for identity in identities:
        swap_identity, swap_path, swap_bbox = state["swap_map"][identity]
        with Image.open(swap_path) as source:
            swap_image = source.convert("RGB")
        swap_identities.append(swap_identity)
        swap_images.append(swap_image)
        swap_bboxes.append(swap_bbox)

    start_index = int(state["next_index"])
    validation_indices = batch.get("validation_index")
    sample_indices = (
        [int(value) for value in _per_sample(validation_indices, batch_size)]
        if validation_indices is not None
        else list(range(start_index, start_index + batch_size))
    )
    state["next_index"] += batch_size
    filenames = [
        f"{sample_index:03d}_{identity}_seed{target_seed}.png"
        for sample_index, identity, target_seed in zip(
            sample_indices,
            identities,
            target_seeds,
        )
    ]
    state["filenames"].extend(filenames)

    images: dict[str, list[Image.Image]] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    fingerprints = {filename: {} for filename in filenames}
    for name, (scale, reference_kind, noise_kind) in state["variants"].items():
        use_swap = reference_kind == "R2"
        variant_images, records = _generate(
            trainer,
            option="A" if name == "PM0" else "B",
            prompts=prompts,
            seeds=target_seeds,
            identities=identities,
            references=references,
            reference_bboxes=reference_bboxes,
            generation_bboxes=generation_bboxes,
            sample_keys=filenames,
            ppr_reference_image=swap_images if use_swap else None,
            ppr_face_bbox_ref=(
                swap_bboxes
                if use_swap and batch_size > 1
                else (swap_bboxes[0] if use_swap else None)
            ),
            ppr_reference_noise_seed=state["noise_seeds"][noise_kind],
            runtime_settings=(
                name == "PM0",
                "base_outside_core",
                scale,
            ),
            diagnostic_variant=name,
            capture_tensor_signatures=name != "PM0",
            ppr_reference_ca_mode=state["reference_ca_override"],
        )
        images[name] = variant_images
        diagnostics[name] = records
        for local_index, filename in enumerate(filenames):
            fingerprint = _randomness_record(records, filename)
            reference_image = (
                swap_images[local_index]
                if use_swap
                else references[local_index][0]
            )
            fingerprint["spatial_reference_image_sha256"] = _image_hash(
                reference_image
            )
            fingerprints[filename][name] = fingerprint

    for local_index, filename in enumerate(filenames):
        sample_diagnostics = {
            name: _sample_records(records, filename)
            for name, records in diagnostics.items()
        }
        _assert_integrity(
            filename,
            fingerprints[filename],
            sample_diagnostics,
            state["reference_ca_mode"],
            identity_token_lane=state["identity_token_lane"],
            identity_fusion_mode=state["identity_fusion_mode"],
            identity_noise_tolerance=state["identity_noise_tolerance"],
            spatial_memory_mode=state["spatial_memory_mode"],
        )
        if state["identity_fusion_mode"] == "identity_only":
            sample_invariants = {}
            for left_name, right_name in (("R1N1", "R1N2"), ("R2N1", "R2N2")):
                left_pixels = np.asarray(
                    images[left_name][local_index].convert("RGB"),
                    dtype=np.int16,
                )
                right_pixels = np.asarray(
                    images[right_name][local_index].convert("RGB"),
                    dtype=np.int16,
                )
                max_pixel_difference = int(
                    np.abs(left_pixels - right_pixels).max()
                )
                allowed = int(math.ceil(255.0 * state["identity_noise_tolerance"]))
                if max_pixel_difference > allowed:
                    raise RuntimeError(
                        f"{filename}: identity-only final-image reference-noise "
                        f"leak in {left_name}/{right_name}: "
                        f"max_pixel_difference={max_pixel_difference}, allowed={allowed}"
                    )
                sample_invariants[f"{left_name}_vs_{right_name}"] = {
                    "exact": max_pixel_difference == 0,
                    "max_pixel_difference": max_pixel_difference,
                }
            state["identity_noise_invariant"][filename] = sample_invariants
        state["tensor_rows"].extend(
            _tensor_comparisons(filename, sample_diagnostics)
        )
        state["integrity"][filename] = fingerprints[filename]

        sample_index = sample_indices[local_index]
        identity = identities[local_index]
        target_seed = target_seeds[local_index]
        swap_identity = swap_identities[local_index]
        gen_bbox = generation_bboxes[local_index]
        baseline = images["PM0"][local_index]
        baseline_face = _face_observation(trainer, baseline)
        observations = {
            name: _face_observation(
                trainer, variant_images[local_index]
            )
            for name, variant_images in images.items()
        }
        variant_rows = {}
        for name, variant_images in images.items():
            image = variant_images[local_index]
            output_path = state["root"] / name / filename
            image.save(output_path)
            image_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
            full_mae, core_mae = _pixel_mae(image, baseline, gen_bbox)
            observation = observations[name]
            score_values = _identity_text_scores(
                trainer,
                image=image,
                prompt=prompts[local_index],
                original_identity=identity,
                swapped_identity=swap_identity,
                observation=observation,
            )
            stats = (
                {"applied_delta_rms_ratio": 0.0, "cap_fraction": 0.0}
                if name == "PM0"
                else _processor_stats(diagnostics[name], filename)
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
                        state["variants"][name][2]
                    ],
                    "prompt": prompts[local_index],
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
                    state["root"]
                    / "face_crops"
                    / f"{name}_{filename}"
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
            left_image = images[left][local_index]
            right_image = images[right][local_index]
            full, core = _pixel_mae(
                left_image, right_image, gen_bbox
            )
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
                        state, left_image, right_image, gen_bbox
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
                        left_image,
                    ),
                    "seam_gradient_pair_proxy": _seam_proxy(
                        left_image, right_image, gen_bbox
                    ),
                }
            )

    batch["generated"] = images["R1N1"]
    batch["generated_masks"] = [None] * batch_size
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


def _write_effect_and_identity_summaries(state) -> None:
    root = state["root"]

    def ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            return float("nan")
        return numerator / denominator

    metric_map = {
        "pixel_mae_full": (
            "pixel_mae_full_vs_PM0",
            "pixel_mae_full",
        ),
        "pixel_mae_core": (
            "pixel_mae_core_vs_PM0",
            "pixel_mae_core",
        ),
        "lpips_core": (
            "lpips_core_vs_PM0",
            "lpips_core",
        ),
    }
    effect_rows = []
    for metric, (variant_field, pair_field) in metric_map.items():
        ppr_values = [
            float(row[variant_field])
            for row in state["rows"]
            if row["variant"] != "PM0"
        ]
        reference_values = [
            float(row[pair_field])
            for row in state["pair_rows"]
            if row["effect"] == "reference_image_effect"
        ]
        noise_values = [
            float(row[pair_field])
            for row in state["pair_rows"]
            if row["effect"] == "reference_noise_effect"
        ]
        ppr_mean = _bootstrap_ci(ppr_values)[0]
        reference_mean = _bootstrap_ci(reference_values)[0]
        noise_mean = _bootstrap_ci(noise_values)[0]
        effect_rows.append(
            {
                "metric": metric,
                "ppr_effect_S": ppr_mean,
                "reference_effect_I": reference_mean,
                "noise_effect_N": noise_mean,
                "I_over_S": ratio(reference_mean, ppr_mean),
                "N_over_S": ratio(noise_mean, ppr_mean),
                "I_over_N": ratio(reference_mean, noise_mean),
            }
        )
    with (root / "effect_decomposition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=effect_rows[0].keys())
        writer.writeheader()
        writer.writerows(effect_rows)

    by_variant = {
        (row["filename"], row["variant"]): row
        for row in state["rows"]
    }
    direction_rows = []
    for noise_kind in ("N1", "N2"):
        left_name, right_name = f"R1{noise_kind}", f"R2{noise_kind}"
        for filename in state["filenames"]:
            left = by_variant[(filename, left_name)]
            right = by_variant[(filename, right_name)]
            original_change = (
                float(right["id_similarity_original"])
                - float(left["id_similarity_original"])
            )
            swapped_change = (
                float(right["id_similarity_swapped"])
                - float(left["id_similarity_swapped"])
            )
            direction_rows.append(
                {
                    "filename": filename,
                    "target_identity": left["target_identity"],
                    "noise": noise_kind,
                    "original_similarity_change_R2_minus_R1": original_change,
                    "swapped_similarity_change_R2_minus_R1": swapped_change,
                    "directional_gain_toward_R2": (
                        swapped_change - original_change
                    ),
                }
            )
    with (root / "identity_direction_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=direction_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(direction_rows)

    direction_summary = []
    by_filename = {}
    for row in direction_rows:
        by_filename.setdefault(row["filename"], {})[row["noise"]] = row
    target_rows = []
    for filename, noise_rows in by_filename.items():
        if set(noise_rows) != {"N1", "N2"}:
            raise RuntimeError(f"Missing paired noise direction rows for {filename}")
        n1, n2 = noise_rows["N1"], noise_rows["N2"]
        target_rows.append(
            {
                "filename": filename,
                "target_identity": n1["target_identity"],
                "directional_gain_toward_R2": 0.5 * (
                    n1["directional_gain_toward_R2"]
                    + n2["directional_gain_toward_R2"]
                ),
                "original_similarity_change_R2_minus_R1": 0.5 * (
                    n1["original_similarity_change_R2_minus_R1"]
                    + n2["original_similarity_change_R2_minus_R1"]
                ),
                "swapped_similarity_change_R2_minus_R1": 0.5 * (
                    n1["swapped_similarity_change_R2_minus_R1"]
                    + n2["swapped_similarity_change_R2_minus_R1"]
                ),
                "both_noise_positive": (
                    n1["directional_gain_toward_R2"] > 0
                    and n2["directional_gain_toward_R2"] > 0
                ),
                "noise_sign_flip": (
                    (n1["directional_gain_toward_R2"] > 0)
                    != (n2["directional_gain_toward_R2"] > 0)
                ),
            }
        )

    for noise_kind in ("N1", "N2", "all"):
        selected = [
            row
            for row in (target_rows if noise_kind == "all" else direction_rows)
            if noise_kind == "all" or row["noise"] == noise_kind
        ]
        gains = np.asarray(
            [row["directional_gain_toward_R2"] for row in selected],
            dtype=np.float64,
        )
        gains = gains[np.isfinite(gains)]
        mean, low, high = _bootstrap_ci(gains.tolist())
        direction_summary.append(
            {
                "noise": noise_kind,
                "sample_count": int(gains.size),
                "mean_directional_gain_toward_R2": mean,
                "median_directional_gain_toward_R2": float(np.median(gains)),
                "positive_fraction": float(np.mean(gains > 0)),
                "mean_swapped_similarity_change_R2_minus_R1": float(
                    np.nanmean(
                        [row["swapped_similarity_change_R2_minus_R1"] for row in selected]
                    )
                ),
                "mean_original_similarity_change_R2_minus_R1": float(
                    np.nanmean(
                        [row["original_similarity_change_R2_minus_R1"] for row in selected]
                    )
                ),
                "both_noise_positive_fraction": (
                    float(np.mean([row["both_noise_positive"] for row in selected]))
                    if noise_kind == "all"
                    else float("nan")
                ),
                "noise_sign_flip_fraction": (
                    float(np.mean([row["noise_sign_flip"] for row in selected]))
                    if noise_kind == "all"
                    else float("nan")
                ),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    with (root / "identity_direction_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=direction_summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(direction_summary)

    identity_rows = []
    for identity in sorted({row["target_identity"] for row in target_rows}):
        selected = [row for row in target_rows if row["target_identity"] == identity]
        gains = [row["directional_gain_toward_R2"] for row in selected]
        mean, low, high = _bootstrap_ci(gains)
        identity_rows.append(
            {
                "target_identity": identity,
                "target_count": len(selected),
                "mean_directional_gain_toward_R2": mean,
                "positive_fraction": float(np.mean(np.asarray(gains) > 0)),
                "mean_swapped_similarity_change_R2_minus_R1": float(
                    np.nanmean(
                        [row["swapped_similarity_change_R2_minus_R1"] for row in selected]
                    )
                ),
                "mean_original_similarity_change_R2_minus_R1": float(
                    np.nanmean(
                        [row["original_similarity_change_R2_minus_R1"] for row in selected]
                    )
                ),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    with (root / "identity_direction_by_identity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=identity_rows[0].keys())
        writer.writeheader()
        writer.writerows(identity_rows)

    pm_original = np.asarray(
        [
            float(row["id_similarity_original"])
            for row in state["rows"]
            if row["variant"] == "PM0"
        ],
        dtype=np.float64,
    )
    ppr_original = np.asarray(
        [
            float(row["id_similarity_original"])
            for row in state["rows"]
            if row["variant"] != "PM0"
        ],
        dtype=np.float64,
    )
    report = f"""# Neutral reference-CA diagnostic summary

- Reference-half CA mode: `{state["reference_ca_mode"]}`
- Mean original-ID similarity, PM0: `{np.nanmean(pm_original):.6f}`
- Mean original-ID similarity, scale-{state.get("scale", 4.0):g} PPR: `{np.nanmean(ppr_original):.6f}`
- PPR minus PM0 original-ID similarity: `{np.nanmean(ppr_original) - np.nanmean(pm_original):.6f}`
- Mean directional gain toward R2: `{direction_summary[-1]["mean_directional_gain_toward_R2"]:.6f}`
- Fraction with positive directional gain: `{direction_summary[-1]["positive_fraction"]:.3f}`

`effect_decomposition.csv` reports the configured-scale PPR effect `S`, matched-noise
reference-image effect `I`, matched-reference noise effect `N`, and their
ratios. `identity_direction_summary.csv` is the decisive identity-transfer
test. Compare both files with the original-CA run.
"""
    (root / "neutral_reference_ca_summary.md").write_text(
        report, encoding="utf-8"
    )


def _create_contact_sheets(state, rows_per_page: int = 6) -> None:
    cell, label = 256, 24
    names = list(VARIANT_NAMES)
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
    _write_effect_and_identity_summaries(state)

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
                "identity_candidate",
                "identity_connector_input",
                "identity_raw_delta",
                "identity_bounded_delta",
                "identity_applied_delta",
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
- Reference-half CA mode: `{state["reference_ca_mode"]}`

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
        "variants": state.get("variants", _variants(state.get("scale", 4.0))),
        "ppr_reference_noise_scale": state.get("scale", 4.0),
        "reference_noise_seeds": state["noise_seeds"],
        "reference_ca_mode": state["reference_ca_mode"],
        "reference_ca_diagnostic_override": state[
            "reference_ca_override"
        ],
        "observed_batch_sizes": sorted(
            set(state["observed_batch_sizes"])
        ),
        "integrity_assertions_passed": True,
        "identity_token_lane": state["identity_token_lane"],
        "identity_fusion_mode": state["identity_fusion_mode"],
        "identity_token_swap_integrity_checked": state[
            "identity_token_lane"
        ],
        "identity_noise_tolerance": state["identity_noise_tolerance"],
        "identity_noise_invariant_checked": (
            state["identity_fusion_mode"] == "identity_only"
        ),
        "identity_noise_invariant": state["identity_noise_invariant"],
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
