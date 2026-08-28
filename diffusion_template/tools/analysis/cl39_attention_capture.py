#!/usr/bin/env python3
"""Compact, analysis-only aggregation for CL39 processor telemetry."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


MAP_FIELDS = (
    "entropy",
    "null_mass",
    "confidence",
    "applied_confidence",
    "router",
    "effective_low_weight",
    "effective_high_weight",
    "reference_key_mass_all",
    "reference_key_mass_face",
    "native_magnitude",
    "reference_magnitude",
    "raw_delta_magnitude",
    "low_magnitude",
    "high_magnitude",
    "low_before_confidence_magnitude",
    "high_before_confidence_magnitude",
    "low_applied_magnitude",
    "high_applied_magnitude",
    "routed_delta_magnitude",
    "reconstruction_error_magnitude",
)


def _group_name(layer: str) -> str:
    if layer.startswith("up_blocks.0"):
        return "up0"
    if layer.startswith("up_blocks.1"):
        return "up1"
    return layer.split(".", 1)[0]


def _conditional_map(value: torch.Tensor, map_size: int) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 3:
        value = value[-1, :, 0]
    elif value.ndim == 2:
        value = value[-1]
    else:
        raise ValueError(f"Expected a batched query map, got {tuple(value.shape)}")
    side = int(math.isqrt(value.numel()))
    if side * side != value.numel():
        raise ValueError(f"Query map is not square: {value.numel()} tokens")
    image = value.reshape(1, 1, side, side)
    if side != map_size:
        image = F.interpolate(
            image,
            size=(map_size, map_size),
            mode="bilinear",
            align_corners=False,
        )
    return image[0, 0]


class CL39AttentionCollector:
    """Reduce every layer call to 64x64 maps and scalar rows on CPU."""

    def __init__(self, *, map_size: int = 64) -> None:
        self.map_size = int(map_size)
        self._sums: dict[tuple[float, str, str], np.ndarray] = {}
        self._counts: defaultdict[tuple[float, str], int] = defaultdict(int)
        self.rows: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        layer = str(payload["layer"])
        group = _group_name(layer)
        progress_tensor = torch.as_tensor(payload["progress"]).detach().float()
        progress = round(float(progress_tensor[-1].mean().item()), 6)

        tensors = []
        present_fields = []
        for field in MAP_FIELDS:
            value = payload.get(field)
            if value is None:
                continue
            tensors.append(_conditional_map(value, self.map_size))
            present_fields.append(field)
        if not tensors:
            return
        stacked = torch.stack(tensors)
        arrays = stacked.cpu().numpy().astype(np.float32, copy=False)
        maps = dict(zip(present_fields, arrays))

        native = maps["native_magnitude"]
        routed = maps["routed_delta_magnitude"]
        maps["routed_to_native_ratio"] = routed / np.maximum(native, 1.0e-8)

        key = (progress, group)
        self._counts[key] += 1
        for field, array in maps.items():
            sum_key = (progress, group, field)
            if sum_key not in self._sums:
                self._sums[sum_key] = array.copy()
            else:
                self._sums[sum_key] += array

        router = maps["router"]
        face_denom = max(float(router.sum()), 1.0e-8)
        row: dict[str, Any] = {
            "progress": progress,
            "group": group,
            "layer": layer,
            "low_scale": float(torch.as_tensor(payload["low_scale"])[-1].mean()),
            "high_scale": float(torch.as_tensor(payload["high_scale"])[-1].mean()),
            "analysis_delta_scale": float(payload["analysis_delta_scale"]),
        }
        for field, array in maps.items():
            row[f"{field}_mean_all"] = float(array.mean())
            row[f"{field}_mean_face"] = float((array * router).sum() / face_denom)
        confidence = maps["confidence"]
        active = router > 1.0e-4
        face_confidence = confidence[active]
        if face_confidence.size:
            row.update(
                confidence_face_p10=float(np.quantile(face_confidence, 0.10)),
                confidence_face_p50=float(np.quantile(face_confidence, 0.50)),
                confidence_face_p90=float(np.quantile(face_confidence, 0.90)),
                confidence_floor_fraction=float(np.mean(face_confidence <= 0.251)),
                confidence_full_fraction=float(np.mean(face_confidence >= 0.999)),
            )
        self.rows.append(row)

    def save(self, npz_path: Path, rows_path: Path) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._counts, key=lambda item: (item[0], item[1]))
        payload: dict[str, Any] = {
            "progress": np.asarray([item[0] for item in keys], dtype=np.float32),
            "group": np.asarray([item[1] for item in keys]),
            "layer_count": np.asarray([self._counts[item] for item in keys], dtype=np.int32),
            "map_size": np.asarray(self.map_size, dtype=np.int32),
        }
        fields = sorted({sum_key[2] for sum_key in self._sums})
        for field in fields:
            payload[field] = np.stack(
                [
                    self._sums[(progress, group, field)]
                    / float(self._counts[(progress, group)])
                    for progress, group in keys
                ]
            ).astype(np.float32)
        np.savez_compressed(npz_path, **payload)

        rows_path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in self.rows for column in row})
        with rows_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


class CL39BatchedAttentionCollector:
    """Split one trainer validation batch into independent sample collectors."""

    def __init__(
        self,
        sample_indices: list[int],
        *,
        keep_indices: list[int] | None = None,
        map_size: int = 64,
    ) -> None:
        if not sample_indices:
            raise ValueError("sample_indices must not be empty")
        self.batch_sample_indices = [int(index) for index in sample_indices]
        keep = None if keep_indices is None else {int(index) for index in keep_indices}
        self._selected = [
            (local_index, sample_index)
            for local_index, sample_index in enumerate(self.batch_sample_indices)
            if keep is None or sample_index in keep
        ]
        self.sample_indices = [sample_index for _, sample_index in self._selected]
        self.collectors = [
            CL39AttentionCollector(map_size=map_size) for _ in self.sample_indices
        ]

    def __call__(self, payload: dict[str, Any]) -> None:
        confidence = torch.as_tensor(payload["confidence"])
        if confidence.ndim == 0:
            raise ValueError("CL39 confidence payload is missing a batch dimension")
        payload_batch = int(confidence.shape[0])
        sample_count = len(self.batch_sample_indices)
        if payload_batch < sample_count:
            raise ValueError(
                "CL39 payload batch is smaller than the validation batch: "
                f"{payload_batch} < {sample_count}"
            )

        # Classifier-free guidance places the conditional samples last. When
        # guidance is disabled, payload_batch == sample_count and this is zero.
        conditional_start = payload_batch - sample_count
        for (local_index, _), collector in zip(self._selected, self.collectors):
            payload_index = conditional_start + local_index
            sample_payload: dict[str, Any] = {}
            for key, value in payload.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and int(value.shape[0]) == payload_batch
                ):
                    sample_payload[key] = value[payload_index : payload_index + 1]
                else:
                    sample_payload[key] = value
            collector(sample_payload)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        for sample_index, collector in zip(self.sample_indices, self.collectors):
            collector.save(
                output_dir / f"{sample_index:02d}.npz",
                output_dir / f"{sample_index:02d}.csv",
            )


def attach_cl39_analysis(
    pipeline,
    *,
    collector: CL39AttentionCollector | None,
    confidence_override: float | None,
    delta_scale: float,
    branch_mode: str = "actual",
    processor_scope: str = "null_key",
) -> list[str]:
    """Attach analysis state to the already-installed CL39 processor objects."""
    supported_modes = {
        "actual",
        "reference_face",
        "native",
        "low_only",
        "high_only",
    }
    if branch_mode not in supported_modes:
        raise ValueError(
            "BA analysis branch_mode must be one of "
            f"{sorted(supported_modes)}, got {branch_mode!r}"
        )
    if processor_scope not in {"null_key", "all_hardcase"}:
        raise ValueError(
            "BA analysis processor_scope must be 'null_key' or "
            f"'all_hardcase', got {processor_scope!r}"
        )
    processor_map = getattr(pipeline, "_branched_attn_processors", None)
    if processor_map is None:
        processor_map = pipeline.unet.attn_processors
    attached = []
    for name, processor in processor_map.items():
        if processor_scope == "null_key" and not bool(
            getattr(processor, "null_key_router_enabled", False)
        ):
            continue
        if processor_scope == "all_hardcase" and getattr(
            processor, "hardcase_mode", "off"
        ) not in {"soft_router", "temporal_frequency"}:
            continue
        processor._cl39_analysis_layer = name
        processor._cl39_analysis_sink = collector
        processor._cl39_analysis_confidence_override = confidence_override
        processor._cl39_analysis_delta_scale = float(delta_scale)
        processor._cl39_analysis_branch_mode = branch_mode
        attached.append(name)
    if not attached:
        raise RuntimeError(
            f"No BA processors were found for analysis scope {processor_scope!r}"
        )
    return attached
