"""Sealed 80/20 Cosmic/BigCelebs hard-case curriculum for CL20."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.cosmic_large_adapted import load_cosmic_target, open_cosmic_image
from src.datasets.reference_frame import compose_target_frame_reference


FIELDS = {
    "index",
    "optimizer_step",
    "source",
    "identity_id",
    "target_path",
    "reference_path",
    "target_bbox",
    "target_body_crop",
    "reference_bbox",
    "prompt",
    "target_scale",
    "reference_face_fraction",
    "reference_offset",
    "flip_target",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CL20HardcaseCurriculumTrain(BaseDataset):
    """Consume a deterministic 48k-row schedule; two rows equal one step."""

    requires_sequential_sampling = True

    def __init__(
        self,
        schedule_path: str,
        summary_path: str,
        expected_schedule_sha256: str,
        cosmic_manifest_path: str,
        cosmic_root: str,
        big_manifest_path: str,
        big_images_root: str,
        expected_rows: int = 48000,
        schedule_start_row: int = 0,
        num_refs: int = 1,
        random_horizontal_flip: bool = False,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1 or bool(random_horizontal_flip):
            raise ValueError("CL20 schedule owns its single reference and flip decisions")
        self.schedule_path = Path(schedule_path)
        self.summary_path = Path(summary_path)
        self.cosmic_manifest_path = Path(cosmic_manifest_path)
        self.cosmic_root = Path(cosmic_root)
        self.big_manifest_path = Path(big_manifest_path)
        self.big_images_root = Path(big_images_root)
        actual_hash = _sha256(self.schedule_path)
        if actual_hash != str(expected_schedule_sha256).lower():
            raise RuntimeError(
                f"CL20 schedule hash mismatch: expected={expected_schedule_sha256}, "
                f"actual={actual_hash}"
            )
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        if summary.get("kind") != "cl20_hardcase_curriculum_v1":
            raise ValueError("Invalid CL20 summary kind")
        if summary.get("schedule_sha256") != actual_hash:
            raise RuntimeError("CL20 summary does not seal the schedule bytes")
        if summary.get("source_sha256", {}).get("cosmic") != _sha256(
            self.cosmic_manifest_path
        ):
            raise RuntimeError("CL20 Cosmic manifest drift")
        if summary.get("source_sha256", {}).get("big_celebs") != _sha256(
            self.big_manifest_path
        ):
            raise RuntimeError("CL20 BigCelebs manifest drift")

        rows = []
        with self.schedule_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                if set(row) != FIELDS or int(row["index"]) != index:
                    raise ValueError(f"Malformed CL20 schedule row {index}")
                if int(row["optimizer_step"]) != index // 2:
                    raise ValueError(f"CL20 row {index} has the wrong optimizer step")
                if row["source"] not in {"cosmic", "big_celebs"}:
                    raise ValueError(f"Unknown CL20 source at row {index}")
                if row["target_path"] == row["reference_path"]:
                    raise ValueError(f"CL20 self-reference at row {index}")
                rows.append(row)
        if len(rows) != int(expected_rows):
            raise RuntimeError(f"Expected {expected_rows} CL20 rows, got {len(rows)}")
        unique_paths = {
            (row["source"], str(row[key]))
            for row in rows
            for key in ("target_path", "reference_path")
        }
        for source, relative in unique_paths:
            root = self.cosmic_root if source == "cosmic" else self.big_images_root
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
        self.schedule_start_row = int(schedule_start_row)
        if self.schedule_start_row % 2 or not 0 <= self.schedule_start_row < len(rows):
            raise ValueError("CL20 schedule_start_row must be an in-range batch boundary")
        self.rows = rows
        self.summary = summary
        super().__init__(rows, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.rows) - self.schedule_start_row

    def validate_resume_position(self, completed_optimizer_steps: int) -> None:
        expected = int(completed_optimizer_steps) * 2
        if self.schedule_start_row != expected:
            raise RuntimeError(
                f"CL20 resume mismatch: configured row={self.schedule_start_row}, "
                f"checkpoint row={expected}"
            )

    def _load(self, row: dict, which: str) -> Image.Image:
        source = row["source"]
        relative = str(row[f"{which}_path"])
        if source == "cosmic":
            if which == "target":
                return load_cosmic_target(
                    self.cosmic_root,
                    relative,
                    row.get("target_body_crop"),
                )
            return open_cosmic_image(self.cosmic_root, relative)
        path = self.big_images_root / relative
        image = Image.open(path).convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError(f"Scheduled BigCelebs image is not 1024: {relative}")
        return image

    @staticmethod
    def _scale_target(
        image: Image.Image,
        bbox: list[float],
        scale: float,
    ) -> tuple[Image.Image, list[float]]:
        if scale >= 0.999:
            return image, bbox
        size = max(64, int(round(1024 * scale)))
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        left = (1024 - size) // 2
        top = 1024 - size
        canvas = Image.new("RGB", (1024, 1024), (127, 127, 127))
        canvas.paste(resized, (left, top))
        x0, y0, x1, y1 = [float(value) for value in bbox]
        return canvas, [
            left + x0 * scale,
            top + y0 * scale,
            left + x1 * scale,
            top + y1 * scale,
        ]

    def __getitem__(self, index: int):
        row = self.rows[self.schedule_start_row + int(index)]
        target = self._load(row, "target")
        reference = self._load(row, "reference")
        target_bbox = deepcopy(row["target_bbox"])
        reference_bbox = deepcopy(row["reference_bbox"])
        target, target_bbox = self._scale_target(
            target, target_bbox, float(row["target_scale"])
        )
        if bool(row["flip_target"]):
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]
        reference, reference_bbox, descriptor, _ = compose_target_frame_reference(
            reference,
            reference_bbox,
            target_bbox,
            canvas_size=1024,
            fill="edge",
            gray_level=127,
            target_face_fraction=float(row["reference_face_fraction"]),
            position_offset=tuple(float(value) for value in row["reference_offset"]),
        )
        source_root = self.cosmic_root if row["source"] == "cosmic" else self.big_images_root
        target_path = str(source_root / str(row["target_path"]))
        reference_path = str(source_root / str(row["reference_path"]))
        sample = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "bbox": deepcopy(target_bbox),
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": str(row["prompt"]),
            "prompt": str(row["prompt"]),
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_sizes": (1024, 1024),
            "identity_id": str(row["identity_id"]),
            "target_path": target_path,
            "reference_path": reference_path,
            "reference_cache_key": f"{reference_path}::{descriptor}",
        }
        return self.preprocess_data(sample)


cl20_hardcase_curriculum = CL20HardcaseCurriculumTrain
