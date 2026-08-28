"""Opt-in sequential dataset for sealed BC_E13 dataset schedules."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.bc_e13_schedule_policy import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    SCHEDULE_FIELDS,
    is_directional,
    iter_manifest_paths,
    load_identity_manifest,
    sha256_file,
)


class BigCelebsE13ScheduledTrain(BaseDataset):
    """Consume an exact target/reference/flip schedule in row order."""

    requires_sequential_sampling = True

    def __init__(
        self,
        schedule_path: str,
        schedule_summary_path: str,
        expected_schedule_sha256: str,
        expected_mode: str,
        big_manifest_path: str,
        big_images_path: str,
        expected_big_manifest_sha256: str,
        large_manifest_path: str | None = None,
        large_images_path: str | None = None,
        expected_large_manifest_sha256: str | None = None,
        expected_schedule_rows: int = 48000,
        schedule_start_row: int = 0,
        num_refs: int = 1,
        random_horizontal_flip: bool = False,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("BC_E13 schedules support exactly one reference")
        if bool(random_horizontal_flip):
            raise ValueError("Flip decisions are sealed in the BC_E13 schedule")
        if expected_mode not in {"ds1", "ds2", "ds3"}:
            raise ValueError(f"Unsupported BC_E13 schedule mode: {expected_mode!r}")
        self.expected_schedule_rows = int(expected_schedule_rows)
        if self.expected_schedule_rows < 2 or self.expected_schedule_rows % 2:
            raise ValueError("expected_schedule_rows must be a positive multiple of 2")

        self.schedule_path = Path(schedule_path)
        self.schedule_summary_path = Path(schedule_summary_path)
        self.expected_mode = expected_mode
        self.source_manifests = {"big_celebs": Path(big_manifest_path)}
        self.source_roots = {"big_celebs": Path(big_images_path)}
        self.expected_manifest_hashes = {
            "big_celebs": str(expected_big_manifest_sha256).strip().lower()
        }
        if expected_mode == "ds3":
            if not large_manifest_path or not large_images_path:
                raise ValueError("ds3 requires the Large Dataset manifest and image root")
            if not expected_large_manifest_sha256:
                raise ValueError("ds3 requires the Large Dataset manifest SHA-256")
            self.source_manifests["large_dataset"] = Path(large_manifest_path)
            self.source_roots["large_dataset"] = Path(large_images_path)
            self.expected_manifest_hashes["large_dataset"] = str(
                expected_large_manifest_sha256
            ).strip().lower()

        self.records: dict[str, dict[str, dict[str, dict]]] = {}
        self.meta_by_path: dict[str, dict[str, dict]] = {}
        self.identity_by_path: dict[str, dict[str, str]] = {}
        for source, manifest_path in self.source_manifests.items():
            actual = sha256_file(manifest_path)
            if actual != self.expected_manifest_hashes[source]:
                raise RuntimeError(
                    f"{source} manifest SHA-256 mismatch: expected="
                    f"{self.expected_manifest_hashes[source]}, found={actual}"
                )
            records = load_identity_manifest(manifest_path)
            self.records[source] = records
            self.meta_by_path[source] = {}
            self.identity_by_path[source] = {}
            for identity, relative_path, metadata in iter_manifest_paths(records):
                self.meta_by_path[source][relative_path] = metadata
                self.identity_by_path[source][relative_path] = identity

        expected_schedule_sha256 = str(expected_schedule_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_schedule_sha256):
            raise ValueError("expected_schedule_sha256 must be 64 lowercase hex digits")
        actual_schedule_sha256 = sha256_file(self.schedule_path)
        if actual_schedule_sha256 != expected_schedule_sha256:
            raise RuntimeError(
                "BC_E13 schedule SHA-256 mismatch: "
                f"expected={expected_schedule_sha256}, found={actual_schedule_sha256}"
            )

        summary = json.loads(self.schedule_summary_path.read_text(encoding="utf-8"))
        if summary.get("kind") != "bc_e13_dataset_schedule":
            raise ValueError("Invalid BC_E13 schedule summary kind")
        if summary.get("policy_version") != POLICY_VERSION:
            raise ValueError("Unsupported BC_E13 schedule policy version")
        if summary.get("mode") != expected_mode:
            raise ValueError("BC_E13 config/schedule mode mismatch")
        if summary.get("schedule", {}).get("sha256") != actual_schedule_sha256:
            raise RuntimeError("Schedule bytes do not match the schedule summary")
        for source, expected_hash in self.expected_manifest_hashes.items():
            source_summary = summary.get("sources", {}).get(source, {})
            sealed = source_summary.get("sha256")
            if sealed != expected_hash:
                raise RuntimeError(f"Summary source hash mismatch for {source}")
            if Path(str(source_summary.get("images_root"))) != self.source_roots[source]:
                raise RuntimeError(f"Summary image-root mismatch for {source}")

        schedule: list[dict] = []
        with self.schedule_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Blank schedule row at line {line_number}")
                row = json.loads(line)
                self._validate_row(row, len(schedule))
                schedule.append(row)
        if len(schedule) != int(summary.get("schedule", {}).get("rows", -1)):
            raise RuntimeError("Schedule row count does not match its summary")
        if len(schedule) != self.expected_schedule_rows:
            raise RuntimeError(
                f"Expected {self.expected_schedule_rows:,} sealed rows, got {len(schedule)}"
            )

        self.schedule_start_row = int(schedule_start_row)
        if not 0 <= self.schedule_start_row < len(schedule):
            raise ValueError("schedule_start_row is outside the sealed schedule")
        if self.schedule_start_row % 2:
            raise ValueError("schedule_start_row must align to global batch size 2")
        self.schedule = schedule
        self.schedule_summary = summary
        # AICODE-NOTE: 09 Aug 2026 - Row order, pair selection, and target flips
        # are scientific inputs. The file hash and source hashes must validate
        # before BaseDataset transforms or a Comet experiment can be created.
        super().__init__(schedule, *args, **kwargs)

    def _validate_row(self, row: dict, expected_index: int) -> None:
        if set(row) != SCHEDULE_FIELDS:
            raise ValueError(
                f"Unexpected BC_E13 schedule fields at row {expected_index}: "
                f"{sorted(row)}"
            )
        if int(row["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("Unsupported BC_E13 row schema")
        if int(row["schedule_index"]) != expected_index:
            raise ValueError(f"Non-contiguous schedule at row {expected_index}")
        if int(row["optimizer_step"]) != expected_index // 2:
            raise ValueError(f"Incorrect optimizer_step at row {expected_index}")
        source = str(row["source"])
        if source not in self.source_manifests:
            raise ValueError(f"Unavailable source {source!r} at row {expected_index}")
        if row["source_manifest_sha256"] != self.expected_manifest_hashes[source]:
            raise ValueError(f"Source hash mismatch in row {expected_index}")
        target_path = str(row["target_path"])
        reference_path = str(row["reference_path"])
        if target_path == reference_path:
            raise ValueError(f"Self-reference at row {expected_index}")
        if target_path not in self.meta_by_path[source]:
            raise ValueError(f"Unknown target at row {expected_index}: {target_path}")
        if reference_path not in self.meta_by_path[source]:
            raise ValueError(f"Unknown reference at row {expected_index}: {reference_path}")
        if not (self.source_roots[source] / target_path).is_file():
            raise FileNotFoundError(
                f"Missing scheduled target at row {expected_index}: "
                f"{self.source_roots[source] / target_path}"
            )
        if not (self.source_roots[source] / reference_path).is_file():
            raise FileNotFoundError(
                f"Missing scheduled reference at row {expected_index}: "
                f"{self.source_roots[source] / reference_path}"
            )
        identity = str(row["identity_id"])
        if self.identity_by_path[source][target_path] != identity:
            raise ValueError(f"Cross-identity target at row {expected_index}")
        if self.identity_by_path[source][reference_path] != identity:
            raise ValueError(f"Cross-identity reference at row {expected_index}")
        target_metadata = self.meta_by_path[source][target_path]
        reference_metadata = self.meta_by_path[source][reference_path]
        if row["target_bbox"] != target_metadata["new_face_crop"]:
            raise ValueError(f"Target bbox drift at row {expected_index}")
        if row["reference_bbox"] != reference_metadata["new_face_crop"]:
            raise ValueError(f"Reference bbox drift at row {expected_index}")
        if row["prompt"] != target_metadata["text"]:
            raise ValueError(f"Prompt drift at row {expected_index}")
        if bool(row["flip_target"]) and is_directional(str(row["prompt"])):
            raise ValueError(f"Directional caption flipped at row {expected_index}")

    def __len__(self) -> int:
        return len(self.schedule) - self.schedule_start_row

    def validate_resume_position(self, completed_optimizer_steps: int) -> None:
        expected_row = int(completed_optimizer_steps) * 2
        if self.schedule_start_row != expected_row:
            raise RuntimeError(
                "BC_E13 schedule/checkpoint mismatch: "
                f"completed_steps={completed_optimizer_steps}, "
                f"expected_start_row={expected_row}, "
                f"configured_start_row={self.schedule_start_row}"
            )

    @staticmethod
    def _load_image(root: Path, relative_path: str, metadata: dict) -> Image.Image:
        image = Image.open(root / relative_path).convert("RGB")
        if image.size != (1024, 1024):
            body_crop = metadata.get("body_crop")
            if not isinstance(body_crop, list) or len(body_crop) != 4:
                raise ValueError(
                    f"Non-1024 image {relative_path} lacks a valid body_crop"
                )
            left, right, top, bottom = [int(value) for value in body_crop]
            image_array = np.asarray(image)[top:bottom, left:right]
            if image_array.shape[:2] != (1024, 1024):
                raise ValueError(
                    f"body_crop for {relative_path} produced "
                    f"{image_array.shape[:2]}, expected (1024, 1024)"
                )
            image = Image.fromarray(image_array)
        return image

    def __getitem__(self, index: int):
        row = self.schedule[self.schedule_start_row + int(index)]
        source = str(row["source"])
        target_relative = str(row["target_path"])
        reference_relative = str(row["reference_path"])
        target_metadata = self.meta_by_path[source][target_relative]
        reference_metadata = self.meta_by_path[source][reference_relative]
        root = self.source_roots[source]

        target = self._load_image(root, target_relative, target_metadata)
        target_bbox = deepcopy(target_metadata["new_face_crop"])
        if bool(row["flip_target"]):
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]
        reference = self._load_image(root, reference_relative, reference_metadata)
        reference_bbox = deepcopy(reference_metadata["new_face_crop"])

        target_path = str(root / target_relative)
        reference_path = str(root / reference_relative)
        prompt = str(row["prompt"])
        sample = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_path": target_path,
            "reference_path": reference_path,
            "reference_cache_key": f"{reference_path}::raw",
            "identity_id": str(row["identity_id"]),
        }
        sample = self.preprocess_data(sample)
        if min(sample["face_bbox"]) < 0 or max(sample["face_bbox"]) > 1024:
            raise ValueError(f"Invalid transformed target bbox for {target_relative}")
        if min(sample["face_bbox_ref"]) < 0 or max(sample["face_bbox_ref"]) > 1024:
            raise ValueError(
                f"Invalid transformed reference bbox for {reference_relative}"
            )
        return sample


big_celebs_e13_scheduled = BigCelebsE13ScheduledTrain
