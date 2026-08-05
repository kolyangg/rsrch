"""Schedule-driven, opt-in BigCelebs training dataset."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from PIL import ImageOps

from src.datasets.big_celebs import BigCelebsTrain
from src.datasets.reference_policy import apply_reference_policy, valid_bbox


DIRECTION_RE = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)
PLAN_FIELDS = {
    "schema_version",
    "row",
    "optimizer_step",
    "identity_id",
    "target_image_id",
    "reference_image_id",
    "target_face_bin",
    "reference_rank",
    "reference_centroid_similarity",
    "flip_target",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BigCelebsScheduledTrain(BigCelebsTrain):
    """Consume a pinned target/reference plan without runtime pair sampling."""

    requires_sequential_sampling = True

    def __init__(
        self,
        manifest_path: str,
        images_path: str,
        sampling_plan_path: str,
        sampling_plan_manifest_path: str,
        expected_sampling_plan_sha256: str,
        schedule_start_row: int = 0,
        reference_crop_margin: float | None = None,
        reference_content_size: int | None = None,
        reference_canvas_size: int | None = None,
        reference_canvas_fill: int = 127,
        random_horizontal_flip: bool | None = None,
        *args,
        **kwargs,
    ):
        if random_horizontal_flip not in (None, False):
            raise ValueError(
                "BigCelebsScheduledTrain receives flip decisions from its plan; "
                "random_horizontal_flip must be false"
            )
        self.source_manifest_path = Path(manifest_path)
        self.sampling_plan_path = Path(sampling_plan_path)
        self.sampling_plan_manifest_path = Path(sampling_plan_manifest_path)
        self.reference_crop_margin = (
            None if reference_crop_margin is None else float(reference_crop_margin)
        )
        self.reference_content_size = (
            None if reference_content_size is None else int(reference_content_size)
        )
        self.reference_canvas_size = (
            None if reference_canvas_size is None else int(reference_canvas_size)
        )
        self.reference_canvas_fill = int(reference_canvas_fill)

        super().__init__(
            manifest_path=manifest_path,
            images_path=images_path,
            random_horizontal_flip=False,
            *args,
            **kwargs,
        )

        expected_digest = str(expected_sampling_plan_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError(
                "expected_sampling_plan_sha256 must contain 64 lowercase hex digits"
            )
        plan_digest = _sha256(self.sampling_plan_path)
        if plan_digest != expected_digest:
            raise RuntimeError(
                "Sampling-plan SHA-256 mismatch: "
                f"expected={expected_digest}, found={plan_digest}"
            )

        plan_manifest = json.loads(
            self.sampling_plan_manifest_path.read_text(encoding="utf-8")
        )
        if plan_manifest.get("kind") != "big_celebs_sampling_plan":
            raise ValueError("Invalid BigCelebs sampling-plan manifest kind")
        if plan_manifest.get("plan_file_sha256") != plan_digest:
            raise RuntimeError("Sampling plan does not match its plan manifest")
        source_digest = _sha256(self.source_manifest_path)
        if plan_manifest.get("source_manifest_sha256") != source_digest:
            raise RuntimeError("Sampling plan belongs to a different source manifest")

        schedule = []
        with self.sampling_plan_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != PLAN_FIELDS:
                    raise ValueError(
                        f"Unexpected fields in plan line {line_number}: "
                        f"{sorted(row)}"
                    )
                self._validate_row(row, len(schedule), int(plan_manifest["batch_size"]))
                schedule.append(row)
        if len(schedule) != int(plan_manifest.get("rows", -1)):
            raise RuntimeError(
                "Sampling-plan row count does not match its manifest: "
                f"found={len(schedule)}, sealed={plan_manifest.get('rows')}"
            )

        self.schedule_start_row = int(schedule_start_row)
        batch_size = int(plan_manifest["batch_size"])
        if not 0 <= self.schedule_start_row < len(schedule):
            raise ValueError(
                f"schedule_start_row must be within [0, {len(schedule) - 1}]"
            )
        if self.schedule_start_row % batch_size != 0:
            raise ValueError(
                "schedule_start_row must align to the sealed global batch size "
                f"of {batch_size}"
            )
        self.schedule = schedule
        self.sampling_plan_manifest = plan_manifest
        self.sampling_plan_audit = plan_manifest.get("audit", {})

    def _validate_row(self, row: dict, expected_row: int, batch_size: int) -> None:
        if row["schema_version"] != 1:
            raise ValueError(f"Unsupported sampling row schema: {row['schema_version']}")
        if int(row["row"]) != expected_row:
            raise ValueError(
                f"Non-contiguous sampling row: expected={expected_row}, "
                f"found={row['row']}"
            )
        if int(row["optimizer_step"]) != expected_row // batch_size:
            raise ValueError(f"Incorrect optimizer_step in sampling row {expected_row}")
        identity = str(row["identity_id"])
        target_path = f"{identity}/{row['target_image_id']}.jpg"
        reference_path = f"{identity}/{row['reference_image_id']}.jpg"
        if target_path == reference_path:
            raise ValueError(f"Self-reference in sampling row {expected_row}")
        if target_path not in self.meta_by_path:
            raise ValueError(f"Unknown target in sampling row {expected_row}: {target_path}")
        if reference_path not in self.meta_by_path:
            raise ValueError(
                f"Unknown reference in sampling row {expected_row}: {reference_path}"
            )
        if self.identity_by_path[target_path] != identity:
            raise ValueError(f"Cross-identity target in sampling row {expected_row}")
        if self.identity_by_path[reference_path] != identity:
            raise ValueError(f"Cross-identity reference in sampling row {expected_row}")

        target_side = self._face_side(self.meta_by_path[target_path])
        reference_side = self._face_side(self.meta_by_path[reference_path])
        expected_bin = "ge256" if target_side >= 256 else "192_255"
        if row["target_face_bin"] != expected_bin:
            raise ValueError(
                f"Target scale-bin mismatch in sampling row {expected_row}"
            )
        if target_side < 192 or reference_side < 256:
            raise ValueError(
                f"Sampling row {expected_row} violates target/reference face gates"
            )
        prompt = str(self.meta_by_path[target_path]["text"])
        if bool(row["flip_target"]) and DIRECTION_RE.search(prompt):
            raise ValueError(
                f"Directional caption is flipped in sampling row {expected_row}"
            )
        if int(row["reference_rank"]) not in {1, 2, 3}:
            raise ValueError(f"Invalid reference rank in sampling row {expected_row}")

    @staticmethod
    def _face_side(metadata: dict) -> float:
        x0, y0, x1, y1 = metadata["new_face_crop"]
        return min(float(x1) - float(x0), float(y1) - float(y0))

    def __len__(self):
        return len(self.schedule) - self.schedule_start_row

    def validate_resume_position(self, completed_optimizer_steps: int) -> None:
        """Fail if checkpoint progress and the scheduled row offset diverge."""
        batch_size = int(self.sampling_plan_manifest["batch_size"])
        expected_row = int(completed_optimizer_steps) * batch_size
        if self.schedule_start_row != expected_row:
            raise RuntimeError(
                "BigCelebs schedule/checkpoint mismatch: "
                f"completed_steps={completed_optimizer_steps}, "
                f"batch_size={batch_size}, expected_start_row={expected_row}, "
                f"configured_start_row={self.schedule_start_row}"
            )

    def __getitem__(self, index: int):
        row = self.schedule[self.schedule_start_row + int(index)]
        identity = str(row["identity_id"])
        target_relative = f"{identity}/{row['target_image_id']}.jpg"
        reference_relative = f"{identity}/{row['reference_image_id']}.jpg"
        target_metadata = self.meta_by_path[target_relative]
        reference_metadata = self.meta_by_path[reference_relative]

        target = self._load_image(target_relative, target_metadata)
        target_bbox = deepcopy(target_metadata["new_face_crop"])
        if bool(row["flip_target"]):
            target = ImageOps.mirror(target)
            x0, y0, x1, y1 = target_bbox
            target_bbox = [1024 - x1, y0, 1024 - x0, y1]

        reference = self._load_image(reference_relative, reference_metadata)
        reference_bbox = deepcopy(reference_metadata["new_face_crop"])
        reference, reference_bbox, policy_descriptor = apply_reference_policy(
            reference,
            reference_bbox,
            crop_margin=self.reference_crop_margin,
            content_size=self.reference_content_size,
            canvas_size=self.reference_canvas_size,
            canvas_fill=self.reference_canvas_fill,
        )

        prompt = str(target_metadata["text"])
        target_path = str(self.images_path / target_relative)
        reference_path = str(self.images_path / reference_relative)
        sample = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "ref_images": [reference],
            "face_bbox_ref": reference_bbox,
            "prompts": prompt,
            "prompt": prompt,
            "original_sizes": (1024, 1024),
            "crop_top_lefts": (0, 0),
            "target_path": target_path,
            "reference_path": reference_path,
            "reference_cache_key": (
                f"{reference_path}::scheduled::{policy_descriptor}"
            ),
            "identity_id": identity,
        }
        sample = self.preprocess_data(sample)
        if not valid_bbox(sample["face_bbox"], (1024, 1024)):
            raise ValueError(f"Invalid transformed target bbox for {target_relative}")
        if not valid_bbox(sample["face_bbox_ref"], reference.size):
            raise ValueError(
                f"Invalid transformed reference bbox for {reference_relative}"
            )
        return sample


big_celebs_scheduled = BigCelebsScheduledTrain
