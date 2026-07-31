"""Controlled Cosmic Large dataset policies around the initial test-branch path."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import get_crop_values
from src.datasets.reference_policy import apply_reference_policy, valid_bbox


logger = logging.getLogger(__name__)


class CosmicLargeInitialUsageTrain(BaseDataset):
    """Replay initial Cosmic Large semantics with isolated dataset-policy arms."""

    REFERENCE_MODES = {
        "self",
        "uniform",
        "highest_score",
        "top3_softmax",
    }

    def __init__(
        self,
        cosmic_large_json_pth: str,
        cosmic_large_texts_json_pth: str,
        images_path: str,
        candidate_manifest_path: str | None = None,
        reference_mode: str = "self",
        min_face_res: int = 0,
        reference_crop_margin: float | None = None,
        reference_content_size: int | None = 256,
        reference_canvas_size: int | None = None,
        reference_canvas_fill: int = 127,
        random_horizontal_flip: bool = True,
        random_reference_flip: bool = True,
        topk_temperature: float = 0.05,
        num_refs: int = 1,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("CosmicLargeInitialUsageTrain supports num_refs=1")
        self.images_path = Path(images_path)
        self.reference_mode = str(reference_mode).lower()
        self.min_face_res = int(min_face_res)
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
        self.topk_temperature = float(topk_temperature)
        if self.reference_mode not in self.REFERENCE_MODES:
            raise ValueError(
                "reference_mode must be one of "
                f"{sorted(self.REFERENCE_MODES)}, got {reference_mode!r}"
            )
        if self.min_face_res < 0:
            raise ValueError("min_face_res must be non-negative")
        if self.topk_temperature <= 0:
            raise ValueError("topk_temperature must be positive")

        with open(cosmic_large_json_pth, encoding="utf-8") as handle:
            cosmic_large = json.load(handle)
        with open(cosmic_large_texts_json_pth, encoding="utf-8") as handle:
            cosmic_large_texts = json.load(handle)
        if not isinstance(cosmic_large, dict) or not cosmic_large:
            raise ValueError(
                f"Invalid Cosmic Large metadata: {cosmic_large_json_pth}"
            )
        if not isinstance(cosmic_large_texts, dict):
            raise ValueError(
                f"Invalid Cosmic Large captions: {cosmic_large_texts_json_pth}"
            )

        candidates_by_target: dict[str, list[dict]] = {}
        candidate_manifest_records = 0
        if self.reference_mode != "self":
            if not candidate_manifest_path:
                raise ValueError(
                    f"reference_mode={self.reference_mode!r} requires "
                    "candidate_manifest_path"
                )
            with open(candidate_manifest_path, encoding="utf-8") as handle:
                candidate_manifest = json.load(handle)
            if not isinstance(candidate_manifest, dict) or not candidate_manifest:
                raise ValueError(
                    f"Invalid candidate manifest: {candidate_manifest_path}"
                )
            candidate_manifest_records = len(candidate_manifest)
            for target_path, record in candidate_manifest.items():
                if not isinstance(record, dict):
                    continue
                candidates = self._valid_candidates(record, str(target_path))
                if candidates:
                    candidates_by_target[str(target_path)] = candidates

        index = []
        audit = {
            "input_records": len(cosmic_large),
            "missing_caption": 0,
            "filtered_target_bbox": 0,
            "filtered_target_face": 0,
            "accepted_records": 0,
            "candidate_manifest_records": candidate_manifest_records,
            "accepted_with_candidates": 0,
            "accepted_self_fallback": 0,
            "valid_reference_candidates": 0,
        }
        for raw_path, raw_record in cosmic_large.items():
            if raw_path not in cosmic_large_texts:
                audit["missing_caption"] += 1
                continue
            record = dict(raw_record)
            bbox = record.get("face_crop_new")
            # 26 Jul 2026 - The zero-threshold baseline deliberately mirrors
            # the initial test-branch bbox gate; positive thresholds are the
            # only target-filter intervention in this matrix.
            if not self._initial_target_bbox_is_in_bounds(bbox):
                audit["filtered_target_bbox"] += 1
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox]
            if min(x1 - x0, y1 - y0) < self.min_face_res:
                audit["filtered_target_face"] += 1
                continue

            record.update(cosmic_large_texts[raw_path])
            target_path = str(raw_path).replace(
                "LAION-5B", "LAION-5B-Filtered-Large", 1
            )
            record["_target_path"] = target_path
            record["_reference_candidates"] = candidates_by_target.get(
                target_path, []
            )
            if record["_reference_candidates"]:
                audit["accepted_with_candidates"] += 1
                audit["valid_reference_candidates"] += len(
                    record["_reference_candidates"]
                )
            elif self.reference_mode != "self":
                # AICODE-NOTE: Preserve the exact 76,045-row legacy target
                # population. Candidate-policy arms fall back to the historical
                # self-reference only for old rows absent from the 59k package.
                audit["accepted_self_fallback"] += 1
            index.append(record)

        audit["accepted_records"] = len(index)
        self.audit = audit
        if not index:
            raise ValueError("No Cosmic Large records passed the configured policy")
        logger.info("CosmicLargeInitialUsageTrain audit: %s", audit)
        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _lookup_bbox(face_bboxes: dict, path: str):
        for candidate in (str(path), str(path).lstrip("/")):
            if candidate in face_bboxes:
                return face_bboxes[candidate]
        return None

    @staticmethod
    def _initial_target_bbox_is_in_bounds(bbox) -> bool:
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return False
        return (
            len(values) == 4
            and min(values) >= 0
            and max(values) <= 1024
        )

    @classmethod
    def _valid_candidates(
        cls, record: dict, target_path: str
    ) -> list[dict]:
        face_paths = list(record.get("face_paths") or [])
        face_bboxes = record.get("face_bboxes") or {}
        face_scores = list(record.get("face_scores") or [])
        scores_aligned = len(face_scores) == len(face_paths)
        candidates = []
        for index, reference_path in enumerate(face_paths):
            reference_path = str(reference_path)
            if reference_path == target_path:
                continue
            reference_bbox = cls._lookup_bbox(face_bboxes, reference_path)
            if not valid_bbox(reference_bbox, (256, 256)):
                continue
            score = float(face_scores[index]) if scores_aligned else None
            candidates.append(
                {
                    "path": reference_path,
                    "bbox": [float(value) for value in reference_bbox],
                    "score": score,
                }
            )
        return candidates

    def _open(self, relative_path: str) -> Image.Image:
        path = Path(relative_path)
        resolved = path if path.is_absolute() else self.images_path / path
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

    def _select_reference(self, candidates: list[dict]) -> dict | None:
        if self.reference_mode == "self" or not candidates:
            return None
        if self.reference_mode == "uniform":
            return random.choice(candidates)
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                float("-inf")
                if candidate["score"] is None
                else float(candidate["score"])
            ),
            reverse=True,
        )
        if self.reference_mode == "highest_score":
            return ranked[0]
        top = ranked[:3]
        finite_scores = [
            0.0 if candidate["score"] is None else float(candidate["score"])
            for candidate in top
        ]
        maximum = max(finite_scores)
        weights = [
            math.exp((score - maximum) / self.topk_temperature)
            for score in finite_scores
        ]
        return random.choices(top, weights=weights, k=1)[0]

    @staticmethod
    def _build_legacy_prompt(record: dict) -> str:
        prompt = ", ".join(
            str(record.get(key) or "").strip()
            for key in (
                "facial_caption",
                "pose_caption",
                "background_caption",
            )
            if str(record.get(key) or "").strip()
        )
        return prompt or "person img"

    def __getitem__(self, ind):
        record = self._index[ind]
        target = self._load_target(record)
        target_bbox = deepcopy(record["face_crop_new"])
        target_flipped = (
            self.random_horizontal_flip and random.random() < 0.5
        )
        if target_flipped:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        reference_record = self._select_reference(
            record["_reference_candidates"]
        )
        if reference_record is None:
            reference = deepcopy(target)
            reference_bbox = deepcopy(target_bbox)
            reference_path = record["_target_path"]
            reference_descriptor = (
                f"initial_self::hflip={int(target_flipped)}"
            )
        else:
            reference_path = reference_record["path"]
            if reference_path == record["_target_path"]:
                raise RuntimeError("Cosmic target/reference path leakage")
            reference = self._open(reference_path)
            reference_bbox = deepcopy(reference_record["bbox"])
            reference, reference_bbox, policy_descriptor = (
                apply_reference_policy(
                    reference,
                    reference_bbox,
                    crop_margin=self.reference_crop_margin,
                    content_size=self.reference_content_size,
                    canvas_size=self.reference_canvas_size,
                    canvas_fill=self.reference_canvas_fill,
                )
            )
            reference_flipped = (
                self.random_reference_flip and random.random() < 0.5
            )
            if reference_flipped:
                reference = ImageOps.mirror(reference)
                width = reference.width
                x0, y0, x1, y1 = reference_bbox
                reference_bbox = [width - x1, y0, width - x0, y1]
            reference_descriptor = (
                f"{self.reference_mode}::{policy_descriptor}::"
                f"hflip={int(reference_flipped)}"
            )

        if "orig_size" in record:
            orig_size = record["orig_size"]
            original_sizes = (orig_size[1], orig_size[0])
            crop_top_lefts = get_crop_values(record)
        else:
            original_sizes = (1024, 1024)
            crop_top_lefts = (0, 0)

        resolved_target = str(self.images_path / record["_target_path"])
        resolved_reference = str(self.images_path / reference_path)
        prompt = self._build_legacy_prompt(record)
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
            "identity_id": str(Path(reference_path).parent),
            "target_path": resolved_target,
            "reference_path": resolved_reference,
            "reference_cache_key": (
                f"{resolved_reference}::{reference_descriptor}"
            ),
        }
        instance_data = self.preprocess_data(instance_data)
        if not valid_bbox(instance_data["face_bbox"], (1024, 1024)):
            raise ValueError(
                f"Invalid transformed target bbox: "
                f"{instance_data['face_bbox']}"
            )
        if not valid_bbox(reference_bbox, reference.size):
            raise ValueError(
                f"Invalid transformed reference bbox: {reference_bbox}"
            )
        return instance_data


cosmic_large_initial_usage = CosmicLargeInitialUsageTrain
