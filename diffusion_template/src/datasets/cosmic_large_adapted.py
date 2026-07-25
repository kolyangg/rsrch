"""Isolated full-Cosmic loader with explicit reference-format policies."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.reference_policy import apply_reference_policy, valid_bbox


logger = logging.getLogger(__name__)
CLASS_RE = re.compile(r"\b(woman|man|girl|boy|child|person)\b", re.IGNORECASE)
TRIGGER_WORD_RE = re.compile(r"\bimg\b")
LEADING_CLASS_RE = re.compile(
    r"^\s*the\s+(woman|man|girl|boy|child|person)\s+img\s*",
    re.IGNORECASE,
)


class CosmicLargeAdaptedTrain(BaseDataset):
    """Read the 59k-record Cosmic package without changing legacy loaders."""

    PROMPT_MODES = {"legacy", "pose_first"}

    def __init__(
        self,
        manifest_path: str,
        dataset_root: str,
        num_refs: int = 1,
        min_face_res: int = 192,
        min_reference_score: float | None = None,
        reference_crop_margin: float | None = 0.2,
        reference_content_size: int | None = 256,
        reference_canvas_size: int | None = None,
        reference_canvas_fill: int = 127,
        random_horizontal_flip: bool = True,
        random_reference_flip: bool = True,
        prompt_mode: str = "legacy",
        prompt_max_words: int | None = None,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("CosmicLargeAdaptedTrain currently supports num_refs=1")
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.min_face_res = int(min_face_res)
        self.min_reference_score = (
            None
            if min_reference_score is None
            else float(min_reference_score)
        )
        self.reference_crop_margin = (
            None
            if reference_crop_margin is None
            else float(reference_crop_margin)
        )
        self.reference_content_size = (
            None
            if reference_content_size is None
            else int(reference_content_size)
        )
        self.reference_canvas_size = (
            None
            if reference_canvas_size is None
            else int(reference_canvas_size)
        )
        self.reference_canvas_fill = int(reference_canvas_fill)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.random_reference_flip = bool(random_reference_flip)
        self.prompt_mode = str(prompt_mode).lower()
        self.prompt_max_words = (
            None if prompt_max_words is None else int(prompt_max_words)
        )
        if self.prompt_mode not in self.PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {sorted(self.PROMPT_MODES)}, "
                f"got {prompt_mode!r}"
            )
        if self.prompt_max_words is not None and self.prompt_max_words < 8:
            raise ValueError("prompt_max_words must be at least 8 when configured")

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Invalid or empty Cosmic manifest: {manifest_path}")

        index = []
        audit = {
            "input_records": len(records),
            "filtered_target_face": 0,
            "filtered_target_bbox": 0,
            "filtered_no_reference": 0,
            "filtered_reference_bbox": 0,
            "filtered_reference_score": 0,
        }
        for target_path, raw_record in records.items():
            if not isinstance(raw_record, dict):
                continue
            target_bbox = raw_record.get("face_crop_new")
            if not valid_bbox(target_bbox, (1024, 1024)):
                audit["filtered_target_bbox"] += 1
                continue
            x0, y0, x1, y1 = [float(value) for value in target_bbox]
            if min(x1 - x0, y1 - y0) < self.min_face_res:
                audit["filtered_target_face"] += 1
                continue

            face_paths = list(raw_record.get("face_paths") or [])
            face_bboxes = raw_record.get("face_bboxes") or {}
            face_scores = list(raw_record.get("face_scores") or [])
            scores_aligned = len(face_scores) == len(face_paths)
            candidates = []
            for ref_index, reference_path in enumerate(face_paths):
                reference_bbox = self._lookup_bbox(face_bboxes, reference_path)
                if not valid_bbox(reference_bbox, (256, 256)):
                    audit["filtered_reference_bbox"] += 1
                    continue
                score = (
                    float(face_scores[ref_index])
                    if scores_aligned
                    else None
                )
                if (
                    self.min_reference_score is not None
                    and score is not None
                    and score < self.min_reference_score
                ):
                    audit["filtered_reference_score"] += 1
                    continue
                candidates.append(
                    {
                        "path": str(reference_path),
                        "bbox": [float(value) for value in reference_bbox],
                        "score": score,
                    }
                )
            if not candidates:
                audit["filtered_no_reference"] += 1
                continue

            record = dict(raw_record)
            record["_target_path"] = str(target_path)
            record["_reference_candidates"] = candidates
            record["_identity_id"] = self._identity_id(record, target_path)
            index.append(record)

        audit["accepted_records"] = len(index)
        self.audit = audit
        if not index:
            raise ValueError("No Cosmic records passed the configured filters")
        logger.info("CosmicLargeAdaptedTrain audit: %s", audit)
        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _lookup_bbox(face_bboxes: dict, path: str):
        candidates = (str(path), str(path).lstrip("/"))
        for candidate in candidates:
            if candidate in face_bboxes:
                return face_bboxes[candidate]
        return None

    @staticmethod
    def _identity_id(record: dict, target_path: str) -> str:
        explicit = (
            record.get("identity_id")
            or record.get("person_id")
            or record.get("id")
        )
        if explicit is not None:
            return str(explicit)
        face_paths = record.get("face_paths") or []
        return str(Path(face_paths[0]).parent if face_paths else target_path)

    def _open(self, relative_path: str) -> Image.Image:
        path = Path(relative_path)
        resolved = path if path.is_absolute() else self.dataset_root / path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return Image.open(resolved).convert("RGB")

    def _load_target(self, record: dict) -> Image.Image:
        target = self._open(record["_target_path"])
        if target.size != (1024, 1024):
            body_crop = record.get("body_crop")
            if body_crop is None or len(body_crop) != 4:
                raise ValueError(
                    f"{record['_target_path']} is {target.size}, but has no body_crop"
                )
            left, top, right, bottom = [int(value) for value in body_crop]
            target_array = np.asarray(target)[top:bottom, left:right]
            if target_array.shape[:2] != (1024, 1024):
                raise ValueError(
                    f"body_crop for {record['_target_path']} produced "
                    f"{target_array.shape[:2]}, expected (1024, 1024)"
                )
            target = Image.fromarray(target_array)
        return target

    def _build_prompt(self, record: dict) -> str:
        facial = str(record.get("facial_caption") or "").strip()
        pose = str(record.get("pose_caption") or "").strip()
        background = str(record.get("background_caption") or "").strip()
        if self.prompt_mode == "legacy":
            prompt = ", ".join(value for value in (facial, pose, background) if value)
        else:
            match = CLASS_RE.search(facial)
            class_name = match.group(1).lower() if match else "person"
            appearance = LEADING_CLASS_RE.sub("", facial).strip(" ,")
            # AICODE-NOTE: 25 Jul 2026 - Full-Cosmic facial captions already
            # contain the lowercase PhotoMaker trigger. Pose-first adds its
            # own leading trigger, so remove inherited copies while preserving
            # legacy prompts and uppercase prose such as "IMG Academy".
            appearance = " ".join(
                TRIGGER_WORD_RE.sub("", appearance).split()
            ).strip(" ,")
            prompt = ", ".join(
                value
                for value in (
                    f"{class_name} img",
                    pose,
                    background,
                    appearance,
                )
                if value
            )
        if not prompt:
            prompt = "person img"
        if self.prompt_max_words is not None:
            prompt = " ".join(prompt.split()[: self.prompt_max_words])
        return prompt

    @staticmethod
    def _crop_top_left(record: dict) -> tuple[int, int]:
        body_crop = record["body_crop"]
        crop_size = float(body_crop[2] - body_crop[0])
        if crop_size <= 0:
            raise ValueError(f"Invalid body_crop: {body_crop!r}")
        coefficient = 1024.0 / crop_size
        x0 = int(float(body_crop[0]) * coefficient)
        y0 = int(float(body_crop[1]) * coefficient)
        return y0, x0

    def __getitem__(self, ind):
        record = self._index[ind]
        target = self._load_target(record)
        target_bbox = deepcopy(record["face_crop_new"])
        target_flipped = self.random_horizontal_flip and random.random() < 0.5
        if target_flipped:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        reference_record = random.choice(record["_reference_candidates"])
        reference_path = reference_record["path"]
        if reference_path == record["_target_path"]:
            raise RuntimeError("Cosmic target/reference path leakage")
        reference = self._open(reference_path)
        reference_bbox = deepcopy(reference_record["bbox"])
        if not valid_bbox(reference_bbox, reference.size):
            raise ValueError(
                f"Reference bbox {reference_bbox!r} is invalid for "
                f"{reference_path} with size {reference.size}"
            )
        reference, reference_bbox, policy_descriptor = apply_reference_policy(
            reference,
            reference_bbox,
            crop_margin=self.reference_crop_margin,
            content_size=self.reference_content_size,
            canvas_size=self.reference_canvas_size,
            canvas_fill=self.reference_canvas_fill,
        )
        reference_flipped = self.random_reference_flip and random.random() < 0.5
        if reference_flipped:
            reference = ImageOps.mirror(reference)
            width = reference.width
            x0, y0, x1, y1 = reference_bbox
            reference_bbox = [width - x1, y0, width - x0, y1]

        if "orig_size" in record:
            orig_size = record["orig_size"]
            original_sizes = (orig_size[1], orig_size[0])
            crop_top_lefts = self._crop_top_left(record)
        else:
            original_sizes = (1024, 1024)
            crop_top_lefts = (0, 0)

        prompt = self._build_prompt(record)
        target_path = str(self.dataset_root / record["_target_path"])
        resolved_reference_path = str(self.dataset_root / reference_path)
        cache_key = (
            f"{resolved_reference_path}::{policy_descriptor}::"
            f"hflip={int(reference_flipped)}"
        )
        instance_data = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "bbox": deepcopy(target_bbox),
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "target_sizes": (1024, 1024),
            "identity_id": record["_identity_id"],
            "target_path": target_path,
            "reference_path": resolved_reference_path,
            "reference_cache_key": cache_key,
        }
        instance_data = self.preprocess_data(instance_data)
        if not valid_bbox(instance_data["face_bbox"], (1024, 1024)):
            raise ValueError(
                f"Invalid transformed target bbox: {instance_data['face_bbox']}"
            )
        if not valid_bbox(reference_bbox, reference.size):
            raise ValueError(f"Invalid transformed reference bbox: {reference_bbox}")
        return instance_data


cosmic_large_adapted = CosmicLargeAdaptedTrain
