"""Identity-aware loader for the adjusted 47.5k-image celebrity dataset."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset


class LargeDatasetTrain(BaseDataset):
    """Sample a target and a distinct reference from the same identity."""

    def __init__(
        self,
        data_json_pth: str,
        images_path: str,
        num_refs: int = 1,
        train_on_separate_image: bool = True,
        singleton_reference_policy: str = "error",
        same_id_ref_map_json_pth: str | None = None,
        random_horizontal_flip: bool = True,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("LargeDatasetTrain currently supports num_refs=1")

        self.images_path = Path(images_path)
        self.train_on_separate_image = bool(train_on_separate_image)
        self.singleton_reference_policy = str(singleton_reference_policy)
        if self.singleton_reference_policy not in {"error", "self"}:
            raise ValueError(
                "singleton_reference_policy must be 'error' or 'self'"
            )
        self.random_horizontal_flip = bool(random_horizontal_flip)

        with open(data_json_pth, encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Invalid or empty identity manifest: {data_json_pth}")

        index = []
        self.paths = []
        self.meta_by_path = {}
        self.identity_by_path = {}
        self.same_id_ref_map = {}
        for identity, image_records in records.items():
            if not isinstance(image_records, dict) or not image_records:
                continue
            identity_paths = []
            for image_id, image_data in image_records.items():
                relative_path = f"{identity}/{image_id}.jpg"
                index.append(image_data)
                self.paths.append(relative_path)
                self.meta_by_path[relative_path] = image_data
                self.identity_by_path[relative_path] = str(identity)
                identity_paths.append(relative_path)
            self.same_id_ref_map[str(identity)] = identity_paths

        if same_id_ref_map_json_pth is not None:
            with open(same_id_ref_map_json_pth, encoding="utf-8") as handle:
                loaded_map = json.load(handle)
            self.same_id_ref_map = {
                str(identity): [str(path) for path in paths]
                for identity, paths in loaded_map.items()
                if isinstance(paths, list)
            }

        if (
            self.train_on_separate_image
            and self.singleton_reference_policy == "error"
        ):
            undersized = [
                identity
                for identity, paths in self.same_id_ref_map.items()
                if len(paths) < 2
            ]
            if undersized:
                raise ValueError(
                    "Distinct same-ID references require at least two images per "
                    f"identity; invalid identities include {undersized[:5]}"
                )

        super().__init__(index, *args, **kwargs)

    def _load_image(self, relative_path: str, metadata: dict) -> Image.Image:
        image = Image.open(self.images_path / relative_path).convert("RGB")
        if image.size != (1024, 1024):
            left, right, top, bottom = [
                int(value) for value in metadata["body_crop"]
            ]
            image_array = np.asarray(image)[top:bottom, left:right]
            if image_array.shape[:2] != (1024, 1024):
                raise ValueError(
                    f"body_crop for {relative_path} produced "
                    f"{image_array.shape[:2]}, expected (1024, 1024)"
                )
            image = Image.fromarray(image_array)
        return image

    def __getitem__(self, index: int):
        target_metadata = self._index[index]
        target_path = self.paths[index]
        identity = self.identity_by_path[target_path]

        target = self._load_image(target_path, target_metadata)
        target_bbox = deepcopy(target_metadata["new_face_crop"])
        if self.random_horizontal_flip and random.random() < 0.5:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        if self.train_on_separate_image:
            candidates = [
                path
                for path in self.same_id_ref_map[identity]
                if path != target_path
            ]
            if not candidates:
                if self.singleton_reference_policy != "self":
                    raise ValueError(
                        f"No distinct same-ID reference for {target_path!r}"
                    )
                # AICODE-NOTE: 31 Jul 2026 - The opt-in singleton manifest uses
                # self-reference only for identities with no distinct image.
                reference_path = target_path
                reference = deepcopy(target)
                reference_bbox = deepcopy(target_bbox)
            else:
                reference_path = random.choice(candidates)
                reference_metadata = self.meta_by_path[reference_path]
                reference = self._load_image(reference_path, reference_metadata)
                reference_bbox = deepcopy(reference_metadata["new_face_crop"])
        else:
            reference_path = target_path
            reference = deepcopy(target)
            reference_bbox = deepcopy(target_bbox)

        prompt = str(target_metadata.get("text") or "person img")
        sample = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_path": str(self.images_path / target_path),
            "reference_path": str(self.images_path / reference_path),
            "reference_cache_key": f"{self.images_path / reference_path}::raw",
            "identity_id": identity,
        }
        sample = self.preprocess_data(sample)

        if min(sample["face_bbox"]) < 0 or max(sample["face_bbox"]) > 1024:
            raise ValueError(f"Invalid target face bbox for {target_path}")
        if min(sample["face_bbox_ref"]) < 0 or max(sample["face_bbox_ref"]) > 1024:
            raise ValueError(f"Invalid reference face bbox for {reference_path}")
        return sample
