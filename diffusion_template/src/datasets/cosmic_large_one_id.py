"""Self-contained one-identity subset of Cosmic Large.

The diffusion target and every identity reference are stored as different
files.  Validation references are not present in ``face_paths`` and therefore
cannot be sampled during training.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset


class CosmicLargeOneID(BaseDataset):
    """Train on one target while sampling one of several distinct ID refs."""

    def __init__(
        self,
        data_json_path: str,
        images_path: str,
        num_refs: int = 1,
        random_horizontal_flip: bool = True,
        virtual_length: int | None = None,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("CosmicLargeOneID currently supports num_refs=1")

        self.images_path = Path(images_path)
        self.num_refs = 1
        self.random_horizontal_flip = bool(random_horizontal_flip)

        with open(data_json_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not records:
            raise ValueError(f"No records found in {data_json_path}")

        index = []
        for target_path, record in records.items():
            item = deepcopy(record)
            item["image_path"] = target_path
            refs = item.get("face_paths") or []
            if not refs:
                raise ValueError(f"{target_path} has no training references")
            if target_path in refs:
                raise ValueError(
                    f"Target/reference leakage in {target_path}: target is in face_paths"
                )
            index.append(item)

        if virtual_length is not None:
            virtual_length = int(virtual_length)
            if virtual_length < len(index):
                raise ValueError(
                    f"virtual_length={virtual_length} is smaller than "
                    f"the {len(index)} real records"
                )
            index = (index * math.ceil(virtual_length / len(index)))[:virtual_length]

        super().__init__(index, *args, **kwargs)

    def _open(self, relative_path: str) -> Image.Image:
        path = self.images_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return Image.open(path).convert("RGB")

    def __getitem__(self, ind):
        record = self._index[ind]
        target_path = record["image_path"]
        target = self._open(target_path)
        bbox = deepcopy(record["face_crop_new"])

        if target.size != (1024, 1024):
            body_crop = record.get("body_crop")
            if body_crop is None:
                raise ValueError(
                    f"{target_path} is {target.size}, but no body_crop was supplied"
                )
            left, top, right, bottom = body_crop
            target = target.crop((left, top, right, bottom))
            bbox = [
                bbox[0] - left,
                bbox[1] - top,
                bbox[2] - left,
                bbox[3] - top,
            ]
        if target.size != (1024, 1024):
            raise ValueError(f"Expected a 1024x1024 target, got {target.size}")

        if self.random_horizontal_flip and random.random() < 0.5:
            target = ImageOps.mirror(target)
            width = target.size[0]
            x0, y0, x1, y1 = bbox
            bbox = [width - x1, y0, width - x0, y1]

        ref_path = random.choice(record["face_paths"])
        if ref_path == target_path:
            raise RuntimeError("Target/reference leakage detected at sampling time")
        ref_image = self._open(ref_path)
        ref_bbox = deepcopy(record["face_bboxes"][ref_path])

        prompt = ", ".join(
            value
            for value in (
                record.get("facial_caption"),
                record.get("pose_caption"),
                record.get("background_caption"),
            )
            if value
        )
        if not prompt:
            prompt = f"{record.get('class', 'person')} img"

        instance_data = {
            "pixel_values": target,
            "face_bbox": bbox,
            "bbox": deepcopy(bbox),
            "ref_images": [ref_image],
            "face_bbox_ref": ref_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_sizes": (1024, 1024),
            "identity_id": record.get("identity_id"),
            "target_path": str(self.images_path / target_path),
            "reference_path": str(self.images_path / ref_path),
            "reference_cache_key": f"{self.images_path / ref_path}::raw",
        }
        instance_data = self.preprocess_data(instance_data)

        if min(instance_data["face_bbox"]) < 0 or max(instance_data["face_bbox"]) > 1024:
            raise ValueError(f"Invalid transformed face bbox: {instance_data['face_bbox']}")
        return instance_data


# Config-friendly alias matching the snake_case dataset name.
cosmic_large_one_id = CosmicLargeOneID
