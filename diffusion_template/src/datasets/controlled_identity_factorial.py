"""Manifest-backed dataset for the controlled identity/data factorial."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageOps

from src.datasets.base_dataset import BaseDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_bbox(bbox, width: int, height: int) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


class ControlledIdentityFactorial(BaseDataset):
    """Hold identity/split fixed while varying target and reference formats."""

    TARGET_MODES = {"multi", "single"}
    REFERENCE_MODES = {"full_scene", "cosmic_256"}

    def __init__(
        self,
        manifest_path: str,
        artifact_root: str | None = None,
        target_mode: str = "multi",
        reference_mode: str = "full_scene",
        virtual_length: int = 1000,
        random_horizontal_flip: bool = True,
        verify_hashes: bool = True,
        require_sealed_validation: bool = True,
        *args,
        **kwargs,
    ):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        if int(self.manifest.get("schema_version", -1)) != 1:
            raise ValueError(
                f"Unsupported controlled-factorial manifest schema in {manifest_path}"
            )
        validation_status = (
            (self.manifest.get("validation") or {}).get(
                "generation_bboxes_status"
            )
        )
        if require_sealed_validation and validation_status != "sealed":
            raise ValueError(
                "Controlled-factorial validation package is not sealed; run "
                "the PhotoMaker bbox preflight, inspect all 12 outputs, and "
                "rebuild the artifact with --generation-bboxes."
            )

        self.artifact_root = (
            Path(artifact_root)
            if artifact_root is not None
            else self.manifest_path.parent
        )
        self.target_mode = str(target_mode).lower()
        self.reference_mode = str(reference_mode).lower()
        self.random_horizontal_flip = bool(random_horizontal_flip)
        if self.target_mode not in self.TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {sorted(self.TARGET_MODES)}, "
                f"got {target_mode!r}"
            )
        if self.reference_mode not in self.REFERENCE_MODES:
            raise ValueError(
                f"reference_mode must be one of {sorted(self.REFERENCE_MODES)}, "
                f"got {reference_mode!r}"
            )

        selection = self.manifest.get("selection") or {}
        training_ids = [str(value) for value in selection.get("training_image_ids", [])]
        if len(training_ids) != 8 or len(set(training_ids)) != 8:
            raise ValueError("Manifest must contain exactly eight distinct training image IDs")
        holdouts = {
            str(selection.get("recurring_validation_image_id")),
            str(selection.get("final_holdout_image_id")),
        }
        if set(training_ids) & holdouts:
            raise ValueError("Recurring/final holdouts must not be training images")

        single_target_id = str(selection.get("single_target_image_id"))
        if single_target_id not in training_ids:
            raise ValueError("single_target_image_id must be one of the training IDs")

        self.images = self.manifest.get("images") or {}
        self.derived_refs = (
            (self.manifest.get("derived_references") or {}).get("cosmic_256") or {}
        )
        self.training_ids = training_ids
        self.single_target_id = single_target_id
        self.identity = str(self.manifest.get("identity"))

        real_index = training_ids if self.target_mode == "multi" else [single_target_id]
        virtual_length = int(virtual_length)
        if virtual_length < len(real_index):
            raise ValueError(
                f"virtual_length={virtual_length} is smaller than {len(real_index)}"
            )
        index = (real_index * math.ceil(virtual_length / len(real_index)))[:virtual_length]

        self._validate_artifact(verify_hashes=bool(verify_hashes))
        super().__init__(index, *args, **kwargs)

    def _path(self, relative_path: str) -> Path:
        path = self.artifact_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _validate_artifact(self, *, verify_hashes: bool) -> None:
        for image_id in self.training_ids:
            image_record = self.images.get(image_id)
            if not isinstance(image_record, dict):
                raise ValueError(f"Missing image record for training ID {image_id}")
            image_path = self._path(str(image_record.get("artifact_path")))
            with Image.open(image_path) as image:
                if image.size != (1024, 1024):
                    raise ValueError(
                        f"Expected 1024x1024 artifact for {image_id}, got {image.size}"
                    )
            if not _valid_bbox(image_record.get("face_bbox"), 1024, 1024):
                raise ValueError(f"Invalid face bbox for training ID {image_id}")
            if verify_hashes and _sha256(image_path) != image_record.get("artifact_sha256"):
                raise ValueError(f"Artifact hash mismatch for {image_path}")

            derived = self.derived_refs.get(image_id)
            if not isinstance(derived, dict):
                raise ValueError(f"Missing cosmic_256 reference for training ID {image_id}")
            derived_path = self._path(str(derived.get("path")))
            with Image.open(derived_path) as image:
                if image.size != (256, 256):
                    raise ValueError(
                        f"Expected 256x256 derived ref for {image_id}, got {image.size}"
                    )
            if not _valid_bbox(derived.get("face_bbox"), 256, 256):
                raise ValueError(f"Invalid derived face bbox for training ID {image_id}")
            if verify_hashes and _sha256(derived_path) != derived.get("sha256"):
                raise ValueError(f"Derived reference hash mismatch for {derived_path}")

        if not verify_hashes:
            return
        validation = self.manifest.get("validation") or {}
        validation_hashes = (
            ("reference_path", "reference_sha256"),
            ("prompts_path", "prompts_sha256"),
            ("classes_path", "classes_sha256"),
            ("reference_bboxes_path", "reference_bboxes_sha256"),
            ("generation_bboxes_path", "generation_bboxes_sha256"),
            ("generation_bbox_cache_path", "generation_bbox_cache_sha256"),
        )
        for path_key, hash_key in validation_hashes:
            artifact_path = self._path(str(validation.get(path_key)))
            if _sha256(artifact_path) != validation.get(hash_key):
                raise ValueError(f"Validation artifact hash mismatch for {artifact_path}")

        final_holdout = self.manifest.get("final_holdout") or {}
        final_path = self._path(str(final_holdout.get("path")))
        if _sha256(final_path) != final_holdout.get("sha256"):
            raise ValueError(f"Final holdout hash mismatch for {final_path}")

    def _open(self, relative_path: str) -> Image.Image:
        return Image.open(self._path(relative_path)).convert("RGB")

    def _reference_record(self, image_id: str) -> tuple[dict, str]:
        if self.reference_mode == "full_scene":
            record = self.images[image_id]
            return record, str(record["artifact_path"])
        record = self.derived_refs[image_id]
        return record, str(record["path"])

    def __getitem__(self, ind):
        target_id = str(self._index[ind])
        target_record = self.images[target_id]
        target_path = str(target_record["artifact_path"])
        target = self._open(target_path)
        target_bbox = deepcopy(target_record["face_bbox"])

        if self.random_horizontal_flip and random.random() < 0.5:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        # AICODE-NOTE: The same eight source IDs back every factorial arm.
        # Reference mode may change the file representation, but never the
        # source identity/image split, and self-reference is rejected here.
        reference_candidates = [
            image_id for image_id in self.training_ids if image_id != target_id
        ]
        reference_id = random.choice(reference_candidates)
        if reference_id == target_id:
            raise RuntimeError("Target/reference leakage detected at sampling time")

        reference_record, reference_relative_path = self._reference_record(reference_id)
        reference = self._open(reference_relative_path)
        reference_bbox = deepcopy(reference_record["face_bbox"])
        reference_path = str(self._path(reference_relative_path))

        prompt = str(target_record.get("prompt") or "A person img")
        reference_cache_key = str(
            reference_record.get(
                "cache_key",
                f"{reference_path}::{reference_record.get('artifact_sha256')}",
            )
        )

        instance_data = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "bbox": deepcopy(target_bbox),
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_sizes": (1024, 1024),
            "identity_id": self.identity,
            "target_path": str(self._path(target_path)),
            "reference_path": reference_path,
            "reference_cache_key": reference_cache_key,
        }
        instance_data = self.preprocess_data(instance_data)

        if not _valid_bbox(instance_data["face_bbox"], 1024, 1024):
            raise ValueError(
                f"Invalid transformed target bbox: {instance_data['face_bbox']}"
            )
        ref_width, ref_height = reference.size
        if not _valid_bbox(reference_bbox, ref_width, ref_height):
            raise ValueError(f"Invalid reference bbox: {reference_bbox}")
        if target_id == reference_id:
            raise RuntimeError("Target/reference source IDs unexpectedly match")
        return instance_data


controlled_identity_factorial = ControlledIdentityFactorial
