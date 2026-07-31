"""Strict training dataset for the versioned Big Celebs release."""

from __future__ import annotations

import math
import re
from numbers import Real

from src.datasets.large_dataset import LargeDatasetTrain


class BigCelebsTrain(LargeDatasetTrain):
    """Load only release records eligible for distinct same-ID references."""

    def __init__(
        self,
        manifest_path: str,
        images_path: str,
        num_refs: int = 1,
        min_face_res: int = 192,
        trigger_word: str = "img",
        strict_manifest_fields: bool = True,
        random_horizontal_flip: bool = True,
        *args,
        **kwargs,
    ):
        self.min_face_res = int(min_face_res)
        if self.min_face_res < 1 or self.min_face_res > 1024:
            raise ValueError("min_face_res must be within [1, 1024]")
        self.trigger_word = str(trigger_word)
        if not self.trigger_word:
            raise ValueError("trigger_word must not be empty")
        self.strict_manifest_fields = bool(strict_manifest_fields)

        super().__init__(
            data_json_pth=manifest_path,
            images_path=images_path,
            num_refs=num_refs,
            train_on_separate_image=True,
            singleton_reference_policy="error",
            same_id_ref_map_json_pth=None,
            random_horizontal_flip=random_horizontal_flip,
            *args,
            **kwargs,
        )

        # 31 Jul 2026 - Fail before model startup if a release can violate the
        # distinct-reference, spatial-mask, or one-trigger training contract.
        # AICODE-NOTE: Big Celebs manifests are prefiltered release artifacts;
        # the loader validates their policy but never silently drops records.
        trigger_pattern = re.compile(
            rf"(?<!\w){re.escape(self.trigger_word)}(?!\w)"
        )
        expected_fields = {"new_face_crop", "text"}
        for relative_path, metadata in zip(self.paths, self._index):
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata for {relative_path!r}")
            if self.strict_manifest_fields and set(metadata) != expected_fields:
                raise ValueError(
                    f"Unexpected manifest fields for {relative_path!r}: "
                    f"{sorted(metadata)}"
                )

            bbox = metadata.get("new_face_crop")
            if not self._valid_bbox(bbox):
                raise ValueError(
                    f"Invalid new_face_crop for {relative_path!r}: {bbox!r}"
                )
            face_width = float(bbox[2]) - float(bbox[0])
            face_height = float(bbox[3]) - float(bbox[1])
            if min(face_width, face_height) < self.min_face_res:
                raise ValueError(
                    f"Face bbox for {relative_path!r} is below min_face_res="
                    f"{self.min_face_res}: {bbox!r}"
                )

            prompt = metadata.get("text")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing caption for {relative_path!r}")
            trigger_count = len(trigger_pattern.findall(prompt))
            if trigger_count != 1:
                raise ValueError(
                    f"Expected exactly one {self.trigger_word!r} trigger for "
                    f"{relative_path!r}, found {trigger_count}"
                )

    @staticmethod
    def _valid_bbox(bbox) -> bool:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in bbox
        ):
            return False
        x0, y0, x1, y1 = [float(value) for value in bbox]
        return 0.0 <= x0 < x1 <= 1024.0 and 0.0 <= y0 < y1 <= 1024.0
