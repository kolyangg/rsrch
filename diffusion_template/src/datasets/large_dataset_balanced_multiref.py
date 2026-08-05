"""Deterministic identity-balanced multi-reference Large Dataset schedule."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Iterable

from PIL import ImageOps

from src.datasets.large_dataset import LargeDatasetTrain


def _stable_int(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class LargeDatasetBalancedMultiRefTrain(LargeDatasetTrain):
    """Emit a sealed-by-algorithm 48k schedule for a 24k, batch-two run.

    The first reference is the sole spatial BA reference. Remaining references
    contribute PhotoMaker identity tokens only because the model encodes the
    spatial latent/mask from ``ref_images[0]``.
    """

    requires_sequential_sampling = True

    def __init__(
        self,
        schedule_rows: int = 48000,
        schedule_start_row: int = 0,
        schedule_seed: int = 130018,
        num_identity_refs: int = 3,
        random_horizontal_flip: bool | None = None,
        *args,
        **kwargs,
    ) -> None:
        if random_horizontal_flip not in (None, False):
            raise ValueError(
                "Balanced multi-reference flips are schedule-derived; "
                "random_horizontal_flip must be false"
            )
        self.schedule_rows = int(schedule_rows)
        self.schedule_start_row = int(schedule_start_row)
        self.schedule_seed = int(schedule_seed)
        self.num_identity_refs = int(num_identity_refs)
        if self.schedule_rows <= 0:
            raise ValueError("schedule_rows must be positive")
        if not 0 <= self.schedule_start_row < self.schedule_rows:
            raise ValueError("schedule_start_row must be within the schedule")
        if self.schedule_start_row % 2:
            raise ValueError("schedule_start_row must align to global batch size 2")
        if not 1 <= self.num_identity_refs <= 3:
            raise ValueError("num_identity_refs must be in [1, 3]")
        super().__init__(random_horizontal_flip=False, *args, **kwargs)
        eligible = [
            identity
            for identity, paths in self.same_id_ref_map.items()
            if len(paths) >= 2
        ]
        if not eligible:
            raise ValueError("Balanced multi-reference schedule found no eligible IDs")
        self.scheduled_identities = tuple(
            sorted(
                eligible,
                key=lambda identity: (_stable_int(self.schedule_seed, identity), identity),
            )
        )
        self.schedule_audit = {
            "kind": "large_dataset_identity_balanced_multiref_v1",
            "rows": self.schedule_rows,
            "seed": self.schedule_seed,
            "identity_count": len(self.scheduled_identities),
            "num_identity_refs": self.num_identity_refs,
            "spatial_reference": "quality_then_pose_diversity",
            "identity_references": "deterministic_quality_rotated_topk",
        }

    def __len__(self) -> int:
        return self.schedule_rows - self.schedule_start_row

    def validate_resume_position(self, completed_optimizer_steps: int) -> None:
        expected_row = int(completed_optimizer_steps) * 2
        if self.schedule_start_row != expected_row:
            raise RuntimeError(
                "Large Dataset multi-reference schedule/checkpoint mismatch: "
                f"completed_steps={completed_optimizer_steps}, "
                f"expected_start_row={expected_row}, "
                f"configured_start_row={self.schedule_start_row}"
            )

    @staticmethod
    def _face_side(metadata: dict) -> float:
        x0, y0, x1, y1 = metadata["new_face_crop"]
        return min(float(x1) - float(x0), float(y1) - float(y0))

    @staticmethod
    def _face_geometry(metadata: dict) -> tuple[float, float, float]:
        x0, y0, x1, y1 = [float(value) for value in metadata["new_face_crop"]]
        return ((x0 + x1) / 2048.0, (y0 + y1) / 2048.0, (x1 - x0) / 1024.0)

    def _quality(self, path: str) -> float:
        metadata = self.meta_by_path[path]
        for field in ("reference_score", "face_quality", "quality_score"):
            value = metadata.get(field)
            if value is not None:
                return float(value)
        return self._face_side(metadata) / 1024.0

    def _spatial_rank(self, target_path: str, candidate: str) -> tuple[float, ...]:
        target_geometry = self._face_geometry(self.meta_by_path[target_path])
        candidate_geometry = self._face_geometry(self.meta_by_path[candidate])
        pose_distance = sum(
            abs(left - right)
            for left, right in zip(target_geometry, candidate_geometry)
        )
        return (
            self._quality(candidate),
            pose_distance,
            self._face_side(self.meta_by_path[candidate]),
            -float(_stable_int(self.schedule_seed, target_path, candidate)),
        )

    def _rotated_quality_refs(
        self,
        candidates: Iterable[str],
        *,
        row: int,
    ) -> list[str]:
        ordered = sorted(
            candidates,
            key=lambda path: (
                -self._quality(path),
                _stable_int(self.schedule_seed, "id-ref", path),
                path,
            ),
        )
        if not ordered:
            return []
        offset = _stable_int(self.schedule_seed, "offset", row) % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        return rotated[: self.num_identity_refs]

    def __getitem__(self, index: int):
        row = self.schedule_start_row + int(index)
        if not self.schedule_start_row <= row < self.schedule_rows:
            raise IndexError(index)
        identity_count = len(self.scheduled_identities)
        identity = self.scheduled_identities[row % identity_count]
        cycle = row // identity_count
        paths = sorted(self.same_id_ref_map[identity])
        target_offset = _stable_int(self.schedule_seed, "target", identity)
        target_path = paths[(cycle + target_offset) % len(paths)]
        target_metadata = self.meta_by_path[target_path]
        target = self._load_image(target_path, target_metadata)
        target_bbox = deepcopy(target_metadata["new_face_crop"])
        flip_target = bool(_stable_int(self.schedule_seed, "flip", row) & 1)
        if flip_target:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        distinct = [path for path in paths if path != target_path]
        spatial_path = max(
            distinct,
            key=lambda path: self._spatial_rank(target_path, path),
        )
        identity_paths = self._rotated_quality_refs(
            [path for path in distinct if path != spatial_path],
            row=row,
        )
        reference_paths = [spatial_path, *identity_paths]
        references = [
            self._load_image(path, self.meta_by_path[path])
            for path in reference_paths
        ]
        reference_bbox = deepcopy(
            self.meta_by_path[spatial_path]["new_face_crop"]
        )
        prompt = str(target_metadata.get("text") or "person img")
        absolute_reference_paths = [
            str(self.images_path / path) for path in reference_paths
        ]
        sample = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "ref_images": references,
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_path": str(self.images_path / target_path),
            "reference_path": absolute_reference_paths[0],
            "reference_paths": absolute_reference_paths,
            "reference_cache_key": [
                f"{path}::balanced-multiref-v1" for path in absolute_reference_paths
            ],
            "identity_id": identity,
            "schedule_row": row,
        }
        sample = self.preprocess_data(sample)
        if min(sample["face_bbox"]) < 0 or max(sample["face_bbox"]) > 1024:
            raise ValueError(f"Invalid scheduled target bbox for {target_path}")
        if min(sample["face_bbox_ref"]) < 0 or max(sample["face_bbox_ref"]) > 1024:
            raise ValueError(f"Invalid scheduled spatial bbox for {spatial_path}")
        return sample
