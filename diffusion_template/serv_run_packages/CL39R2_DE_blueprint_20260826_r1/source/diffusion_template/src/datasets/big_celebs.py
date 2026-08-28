"""Strict training dataset for the versioned Big Celebs release."""

from __future__ import annotations

import math
import random
import re
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw

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
        semantic_occlusion_probability: float = 0.0,
        semantic_occlusion_seed: int = 150017,
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
        self.semantic_occlusion_probability = float(
            semantic_occlusion_probability
        )
        self.semantic_occlusion_seed = int(semantic_occlusion_seed)
        if not 0.0 <= self.semantic_occlusion_probability <= 0.5:
            raise ValueError("semantic_occlusion_probability must be in [0, 0.5]")

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

    def _augment_target(
        self,
        index: int,
        target: Image.Image,
        target_bbox: list,
    ) -> tuple[Image.Image, np.ndarray | None]:
        if self.semantic_occlusion_probability <= 0.0:
            return target, None

        occluder_mask = np.zeros((1024, 1024), dtype=np.float32)
        rng = random.Random(self.semantic_occlusion_seed + int(index))
        if rng.random() >= self.semantic_occlusion_probability:
            return target, occluder_mask

        # 25 Aug 2026 - Port CL27's deterministic synthetic ownership labels
        # to BC39. A zero mask is still emitted for unsampled examples because
        # the frequency-surface objective requires an ownership tensor.
        # AICODE-NOTE: This augmentation changes target pixels only; reference
        # images and branched-attention K/V routing remain untouched.
        overlay = Image.new("RGBA", target.size, (0, 0, 0, 0))
        alpha = Image.new("L", target.size, 0)
        draw = ImageDraw.Draw(overlay)
        alpha_draw = ImageDraw.Draw(alpha)
        x0, y0, x1, y1 = [int(value) for value in target_bbox]
        width = max(4, x1 - x0)
        height = max(4, y1 - y0)
        family = rng.choice(("eyewear", "goggles", "hair", "hand", "tears"))
        if family in {"eyewear", "goggles"}:
            band_y0 = y0 + int(0.28 * height)
            band_y1 = y0 + int(
                (0.52 if family == "goggles" else 0.45) * height
            )
            shapes = [(x0, band_y0, x1, band_y1)]
        elif family == "hair":
            strand = max(3, width // 12)
            shapes = [
                (
                    x0 + offset,
                    y0,
                    x0 + offset + strand,
                    y0 + int(0.72 * height),
                )
                for offset in (width // 5, width // 2, 4 * width // 5)
            ]
        elif family == "hand":
            shapes = [(x0 + width // 2, y0 + height // 2, x1, y1)]
        else:
            tear_w = max(2, width // 18)
            shapes = [
                (
                    x0 + width // 3,
                    y0 + height // 2,
                    x0 + width // 3 + tear_w,
                    y1,
                ),
                (
                    x0 + 2 * width // 3,
                    y0 + height // 2,
                    x0 + 2 * width // 3 + tear_w,
                    y1,
                ),
            ]
        color = {
            "eyewear": (28, 28, 32, 210),
            "goggles": (35, 90, 130, 225),
            "hair": (45, 28, 20, 200),
            "hand": (184, 130, 105, 220),
            "tears": (120, 190, 235, 180),
        }[family]
        for shape in shapes:
            radius = max(1, width // 30)
            draw.rounded_rectangle(shape, radius=radius, fill=color)
            alpha_draw.rounded_rectangle(shape, radius=radius, fill=255)
        target = Image.alpha_composite(
            target.convert("RGBA"), overlay
        ).convert("RGB")
        return target, np.asarray(alpha, dtype=np.float32) / 255.0

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
