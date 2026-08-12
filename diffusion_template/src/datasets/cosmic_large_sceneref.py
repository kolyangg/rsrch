"""Cosmic Large with native 1024px same-identity scene references (CL1).

# 06 Aug 2026 - The adapted loader supplies a 256x256 tight face crop, which the
# branched spatial lane receives as a 4x bilinear upscale whose face is ~2.1x
# larger (linear) than the target face. This loader instead samples the
# reference from ANOTHER accepted 1024px Cosmic target of the same identity,
# reproducing `LargeDatasetTrain` conventions exactly, so the reference-asset
# format is removed as a variable. It requires an offline identity grouping
# because the Cosmic manifest joins no targets into identities on its own.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import random

from PIL import ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.cosmic_large_adapted import (
    PROMPT_MODES,
    build_cosmic_prompt,
    load_cosmic_target,
)
from src.datasets.cosmic_large_adapted import CosmicLargeAdaptedTrain
from src.datasets.reference_policy import valid_bbox


logger = logging.getLogger(__name__)


class CosmicLargeSceneRefTrain(BaseDataset):
    """Sample a Cosmic target and a distinct same-identity Cosmic scene reference."""

    def __init__(
        self,
        manifest_path: str,
        dataset_root: str,
        identity_groups_path: str,
        expected_identity_groups_sha256: str | None = None,
        num_refs: int = 1,
        min_face_res: int = 192,
        random_horizontal_flip: bool = True,
        random_reference_flip: bool = False,
        prompt_mode: str = "pose_first",
        prompt_max_words: int | None = 50,
        *args,
        **kwargs,
    ):
        if int(num_refs) != 1:
            raise ValueError("CosmicLargeSceneRefTrain currently supports num_refs=1")
        if bool(random_reference_flip):
            # Mirroring destroys the registration this loader exists to provide.
            raise ValueError(
                "CosmicLargeSceneRefTrain requires random_reference_flip=false"
            )
        self.dataset_root = Path(dataset_root)
        self.min_face_res = int(min_face_res)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.prompt_mode = str(prompt_mode).lower()
        self.prompt_max_words = (
            None if prompt_max_words is None else int(prompt_max_words)
        )
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {sorted(PROMPT_MODES)}, got {prompt_mode!r}"
            )

        with open(manifest_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Invalid or empty Cosmic manifest: {manifest_path}")

        groups = self._load_identity_groups(
            identity_groups_path,
            expected_identity_groups_sha256,
        )

        audit = {
            "input_records": len(records),
            "input_identities": len(groups),
            "filtered_target_bbox": 0,
            "filtered_target_face": 0,
            "filtered_not_grouped": 0,
            "filtered_singleton_identity": 0,
        }

        # 1. Keep the targets that pass the same geometry gate as the adapted loader.
        accepted: dict[str, dict] = {}
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
            accepted[str(target_path)] = raw_record

        # 2. Keep only identities that still have two distinct accepted targets.
        index = []
        self.group_by_identity: dict[str, list[str]] = {}
        grouped_paths = set()
        for identity, member_paths in groups.items():
            members = [str(path) for path in member_paths if str(path) in accepted]
            grouped_paths.update(str(path) for path in member_paths)
            if len(members) < 2:
                audit["filtered_singleton_identity"] += 1
                continue
            self.group_by_identity[str(identity)] = members
            for target_path in members:
                record = dict(accepted[target_path])
                record["_target_path"] = target_path
                record["_identity_id"] = str(identity)
                index.append(record)
        audit["filtered_not_grouped"] = sum(
            1 for path in accepted if path not in grouped_paths
        )
        audit["accepted_records"] = len(index)
        audit["accepted_identities"] = len(self.group_by_identity)
        self.audit = audit
        if not index:
            raise ValueError(
                "No Cosmic records survived identity grouping; rebuild "
                "identity_groups.json or lower the grouping threshold"
            )
        logger.info("CosmicLargeSceneRefTrain audit: %s", audit)
        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _load_identity_groups(
        identity_groups_path: str,
        expected_sha256: str | None,
    ) -> dict[str, list[str]]:
        path = Path(identity_groups_path)
        if not path.is_file():
            raise FileNotFoundError(f"Identity groups file not found: {path}")
        payload_bytes = path.read_bytes()
        if expected_sha256:
            actual = hashlib.sha256(payload_bytes).hexdigest()
            if actual != str(expected_sha256).lower():
                raise ValueError(
                    f"Identity groups SHA-256 mismatch for {path}: "
                    f"expected {expected_sha256}, got {actual}"
                )
        payload = json.loads(payload_bytes.decode("utf-8"))
        groups = payload.get("groups") if isinstance(payload, dict) else payload
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"Invalid or empty identity groups file: {path}")
        return groups

    def _load(self, record: dict):
        target = load_cosmic_target(
            self.dataset_root,
            record["_target_path"],
            record.get("body_crop"),
        )
        return target, deepcopy(record["face_crop_new"])

    def __getitem__(self, ind):
        record = self._index[ind]
        target_path = record["_target_path"]
        identity = record["_identity_id"]

        target, target_bbox = self._load(record)
        if self.random_horizontal_flip and random.random() < 0.5:
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        candidates = [
            path for path in self.group_by_identity[identity] if path != target_path
        ]
        if not candidates:
            raise ValueError(f"No distinct same-identity reference for {target_path!r}")
        reference_path = random.choice(candidates)
        reference_record = self._record_by_path(reference_path)
        # The reference keeps its native 1024px scene and its raw box: no crop,
        # no resize, no canvas, and no mirroring.
        reference, reference_bbox = self._load(reference_record)

        prompt = build_cosmic_prompt(record, self.prompt_mode, self.prompt_max_words)
        if "orig_size" in record:
            orig_size = record["orig_size"]
            original_sizes = (orig_size[1], orig_size[0])
            crop_top_lefts = CosmicLargeAdaptedTrain._crop_top_left(record)
        else:
            original_sizes = (1024, 1024)
            crop_top_lefts = (0, 0)

        resolved_target = str(self.dataset_root / target_path)
        resolved_reference = str(self.dataset_root / reference_path)
        instance_data = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "target_path": resolved_target,
            "reference_path": resolved_reference,
            "reference_cache_key": f"{resolved_reference}::raw",
            "identity_id": identity,
        }
        instance_data = self.preprocess_data(instance_data)

        if not valid_bbox(instance_data["face_bbox"], (1024, 1024)):
            raise ValueError(f"Invalid target face bbox for {target_path}")
        if not valid_bbox(reference_bbox, (1024, 1024)):
            raise ValueError(f"Invalid reference face bbox for {reference_path}")
        return instance_data

    def _record_by_path(self, target_path: str) -> dict:
        cache = getattr(self, "_record_cache", None)
        if cache is None:
            cache = {record["_target_path"]: record for record in self._index}
            self._record_cache = cache
        return cache[target_path]


cosmic_large_sceneref = CosmicLargeSceneRefTrain
