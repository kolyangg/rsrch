"""Isolated full-Cosmic loader with explicit reference-format policies."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.datasets.base_dataset import BaseDataset
from src.datasets.reference_frame import compose_target_frame_reference
from src.datasets.reference_policy import apply_reference_policy, valid_bbox


logger = logging.getLogger(__name__)
CLASS_RE = re.compile(r"\b(woman|man|girl|boy|child|person)\b", re.IGNORECASE)
TRIGGER_WORD_RE = re.compile(r"\bimg\b")
LEADING_CLASS_RE = re.compile(
    r"^\s*the\s+(woman|man|girl|boy|child|person)\s+img\s*",
    re.IGNORECASE,
)
PROMPT_MODES = {"legacy", "pose_first"}
REFERENCE_FRAME_MODES = {"native", "target_face_frame"}
#: Guard so a pathological scale band cannot explode the index.
MAX_OVERSAMPLE_FACTOR = 4.0


def build_cosmic_prompt(
    record: dict,
    prompt_mode: str,
    prompt_max_words: int | None,
) -> str:
    """Compose a Cosmic caption. Shared by every Cosmic loader."""
    facial = str(record.get("facial_caption") or "").strip()
    pose = str(record.get("pose_caption") or "").strip()
    background = str(record.get("background_caption") or "").strip()
    if prompt_mode == "legacy":
        prompt = ", ".join(value for value in (facial, pose, background) if value)
    else:
        match = CLASS_RE.search(facial)
        class_name = match.group(1).lower() if match else "person"
        appearance = LEADING_CLASS_RE.sub("", facial).strip(" ,")
        # AICODE-NOTE: 25 Jul 2026 - Full-Cosmic facial captions already
        # contain the lowercase PhotoMaker trigger. Pose-first adds its
        # own leading trigger, so remove inherited copies while preserving
        # legacy prompts and uppercase prose such as "IMG Academy".
        appearance = " ".join(TRIGGER_WORD_RE.sub("", appearance).split()).strip(" ,")
        prompt = ", ".join(
            value
            for value in (f"{class_name} img", pose, background, appearance)
            if value
        )
    if not prompt:
        prompt = "person img"
    if prompt_max_words is not None:
        prompt = " ".join(prompt.split()[: int(prompt_max_words)])
    return prompt


def open_cosmic_image(dataset_root: Path, relative_path: str) -> Image.Image:
    path = Path(relative_path)
    resolved = path if path.is_absolute() else Path(dataset_root) / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return Image.open(resolved).convert("RGB")


def load_cosmic_target(
    dataset_root: Path,
    relative_path: str,
    body_crop: Sequence[float] | None,
) -> Image.Image:
    """Open a Cosmic target and apply `body_crop` when it is not already 1024."""
    target = open_cosmic_image(dataset_root, relative_path)
    if target.size != (1024, 1024):
        if body_crop is None or len(body_crop) != 4:
            raise ValueError(f"{relative_path} is {target.size}, but has no body_crop")
        left, top, right, bottom = [int(value) for value in body_crop]
        target_array = np.asarray(target)[top:bottom, left:right]
        if target_array.shape[:2] != (1024, 1024):
            raise ValueError(
                f"body_crop for {relative_path} produced "
                f"{target_array.shape[:2]}, expected (1024, 1024)"
            )
        target = Image.fromarray(target_array)
    return target


def load_reference_accept_list(path: str | None) -> set[str] | None:
    """Load an offline reference accept-list produced by the Cosmic audit tool."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("accepted") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Reference accept-list is empty or malformed: {path}")
    return {str(entry) for entry in entries}


class CosmicLargeAdaptedTrain(BaseDataset):
    """Read the 59k-record Cosmic package without changing legacy loaders."""

    #: Preserved as a class attribute for backward compatibility; the module-level
    #: constant is the single source of truth and is shared with the CL1 loader.
    PROMPT_MODES = frozenset(PROMPT_MODES)

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
        # 06 Aug 2026 - CL2 controls. Defaults reproduce the historical
        # behaviour exactly, so every existing Cosmic config is unaffected.
        reference_frame_mode: str = "native",
        reference_frame_fill: str = "edge",
        reference_accept_list_path: str | None = None,
        # 07 Aug 2026 - CL5 control. 1 reproduces the historical single-reference
        # behaviour exactly. Above 1, extra distinct same-target crops are passed
        # to the PhotoMaker ID encoder only; ref_images[0] remains the sole
        # spatial latent/KV lane, matching the E18/E19 invariant.
        num_identity_refs: int = 1,
        # 08 Aug 2026 - CL8/CL9 controls. Defaults reproduce the historical
        # behaviour exactly, so every existing config is unaffected.
        target_scale_balance: bool = False,
        target_scale_bins: Sequence[float] | None = None,
        # 09 Aug 2026 - CL8 shipped "reorder", which only permutes the index and
        # is therefore INERT: the DataLoader shuffles every epoch and destroys the
        # ordering. "oversample" duplicates entries from under-represented bands so
        # the index itself is balanced, which survives shuffling. "reorder" is kept
        # only so CL8 reproduces exactly.
        target_scale_balance_mode: str = "reorder",
        reference_scale_jitter: Sequence[float] | None = None,
        reference_position_jitter: float = 0.0,
        semantic_occlusion_probability: float = 0.0,
        semantic_occlusion_seed: int = 150017,
        same_identity_dual_reference: bool = False,
        min_reference_candidates_for_target: int = 3,
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

        self.reference_frame_mode = str(reference_frame_mode).lower()
        self.reference_frame_fill = str(reference_frame_fill).lower()
        if self.reference_frame_mode not in REFERENCE_FRAME_MODES:
            raise ValueError(
                f"reference_frame_mode must be one of {sorted(REFERENCE_FRAME_MODES)}, "
                f"got {reference_frame_mode!r}"
            )
        if self.reference_frame_mode == "target_face_frame" and any(
            value is not None
            for value in (
                self.reference_crop_margin,
                self.reference_content_size,
                self.reference_canvas_size,
            )
        ):
            # Fail closed: the frame policy alone owns reference geometry.
            raise ValueError(
                "reference_frame_mode='target_face_frame' requires "
                "reference_crop_margin, reference_content_size, and "
                "reference_canvas_size to be null"
            )
        self.reference_accept_list_path = reference_accept_list_path
        accept_list = load_reference_accept_list(reference_accept_list_path)

        self.num_identity_refs = int(num_identity_refs)
        if not 1 <= self.num_identity_refs <= 4:
            raise ValueError("num_identity_refs must be in [1, 4]")

        self.target_scale_balance = bool(target_scale_balance)
        self.target_scale_bins = (
            None if target_scale_bins is None else [float(v) for v in target_scale_bins]
        )
        if self.target_scale_balance and not self.target_scale_bins:
            raise ValueError("target_scale_balance requires target_scale_bins")
        self.target_scale_balance_mode = str(target_scale_balance_mode).lower()
        if self.target_scale_balance_mode not in {"reorder", "oversample"}:
            raise ValueError(
                "target_scale_balance_mode must be 'reorder' or 'oversample', "
                f"got {target_scale_balance_mode!r}"
            )
        if (
            self.target_scale_balance_mode == "oversample"
            and not self.target_scale_balance
        ):
            raise ValueError(
                "target_scale_balance_mode='oversample' requires "
                "target_scale_balance=true"
            )
        self.reference_scale_jitter = (
            None
            if reference_scale_jitter is None
            else tuple(float(v) for v in reference_scale_jitter)
        )
        if self.reference_scale_jitter is not None:
            low, high = self.reference_scale_jitter
            if not 0.0 < low < high < 1.0:
                raise ValueError(
                    "reference_scale_jitter must be (min, max) face fractions in (0, 1)"
                )
            if self.reference_frame_mode != "target_face_frame":
                raise ValueError(
                    "reference_scale_jitter requires reference_frame_mode="
                    "'target_face_frame'"
                )
        self.reference_position_jitter = float(reference_position_jitter)
        if not 0.0 <= self.reference_position_jitter <= 0.5:
            raise ValueError("reference_position_jitter must be in [0, 0.5]")
        self.semantic_occlusion_probability = float(
            semantic_occlusion_probability
        )
        self.semantic_occlusion_seed = int(semantic_occlusion_seed)
        if not 0.0 <= self.semantic_occlusion_probability <= 0.5:
            raise ValueError("semantic_occlusion_probability must be in [0, 0.5]")
        self.same_identity_dual_reference = bool(same_identity_dual_reference)
        self.min_reference_candidates_for_target = int(
            min_reference_candidates_for_target
        )
        if self.min_reference_candidates_for_target < 2:
            raise ValueError("min_reference_candidates_for_target must be >= 2")

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
            "filtered_reference_accept_list": 0,
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
                if accept_list is not None and str(reference_path) not in accept_list:
                    # 06 Aug 2026 - Offline identity gate: the reference must have
                    # a detected face overlapping its supplied box. Resolving this
                    # offline keeps InsightFace out of the DataLoader workers.
                    audit["filtered_reference_accept_list"] += 1
                    continue
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
            if (
                self.same_identity_dual_reference
                and len(candidates) < self.min_reference_candidates_for_target
            ):
                audit["filtered_no_reference"] += 1
                continue

            record = dict(raw_record)
            record["_target_path"] = str(target_path)
            record["_reference_candidates"] = candidates
            record["_identity_id"] = self._identity_id(record, target_path)
            index.append(record)

        if self.target_scale_balance:
            # 08 Aug 2026 - CL8: min_face_res=192 discarded 96% of cosmic's
            # full-body targets, so the model only ever saw portrait framing and
            # could not place a small face on a full body. Lowering the filter
            # restores them but leaves the small-face majority dominant, so
            # round-robin across face-area bands to keep every scale represented.
            # Deterministic: bucket order follows the existing manifest order and
            # no RNG is used here.
            edges = sorted(self.target_scale_bins)
            buckets: dict[int, list] = {}
            for record in index:
                x0, y0, x1, y1 = [float(v) for v in record["face_crop_new"]]
                area_pct = (x1 - x0) * (y1 - y0) / (1024.0 * 1024.0) * 100.0
                slot = sum(1 for e in edges if area_pct >= e)
                buckets.setdefault(slot, []).append(record)
            order = [buckets[k] for k in sorted(buckets) if buckets[k]]
            audit["scale_bucket_counts"] = {
                f"bin_{k}": len(buckets[k]) for k in sorted(buckets)
            }
            audit["target_scale_balance"] = True
            audit["target_scale_balance_mode"] = self.target_scale_balance_mode
            if self.target_scale_balance_mode == "oversample":
                # Equalise the bands in the index itself by cycling each smaller
                # band's own entries. This changes the sampled distribution and so
                # survives DataLoader shuffling, unlike "reorder".
                largest = max(len(b) for b in order)
                factors = {}
                for slot, bucket in zip(sorted(buckets), order):
                    factor = largest / float(len(bucket))
                    if factor > MAX_OVERSAMPLE_FACTOR:
                        raise ValueError(
                            f"scale band bin_{slot} would need {factor:.1f}x "
                            f"oversampling (cap {MAX_OVERSAMPLE_FACTOR}); widen "
                            "target_scale_bins or raise min_face_res"
                        )
                    factors[f"bin_{slot}"] = round(factor, 3)
                balanced = []
                for bucket in order:
                    balanced.extend(
                        bucket[position % len(bucket)] for position in range(largest)
                    )
                audit["scale_bucket_oversample_factors"] = factors
                audit["balanced_records"] = len(balanced)
            else:
                balanced = []
                for position in range(max(len(b) for b in order)):
                    for bucket in order:
                        if position < len(bucket):
                            balanced.append(bucket[position])
                audit["balance_warning"] = (
                    "reorder only permutes the index; DataLoader shuffling makes it "
                    "inert. Use target_scale_balance_mode='oversample'."
                )
            index = balanced

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
        return open_cosmic_image(self.dataset_root, relative_path)

    def _load_target(self, record: dict) -> Image.Image:
        return load_cosmic_target(
            self.dataset_root,
            record["_target_path"],
            record.get("body_crop"),
        )

    def _build_prompt(self, record: dict) -> str:
        return build_cosmic_prompt(record, self.prompt_mode, self.prompt_max_words)

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

        occluder_mask = None
        if self.semantic_occlusion_probability > 0.0:
            occluder_mask = np.zeros((1024, 1024), dtype=np.float32)
            rng = random.Random(self.semantic_occlusion_seed + int(ind))
            if rng.random() < self.semantic_occlusion_probability:
                # 11 Aug 2026 - Deterministic synthetic ownership labels teach
                # the native lane to retain target-scene objects inside the face
                # box. The mask is returned explicitly; it never changes BA K/V.
                overlay = Image.new("RGBA", target.size, (0, 0, 0, 0))
                alpha = Image.new("L", target.size, 0)
                draw = ImageDraw.Draw(overlay)
                alpha_draw = ImageDraw.Draw(alpha)
                x0, y0, x1, y1 = [int(value) for value in target_bbox]
                width = max(4, x1 - x0)
                height = max(4, y1 - y0)
                family = rng.choice(("eyewear", "goggles", "hair", "hand", "tears"))
                shapes = []
                if family in {"eyewear", "goggles"}:
                    band_y0 = y0 + int(0.28 * height)
                    band_y1 = y0 + int((0.52 if family == "goggles" else 0.45) * height)
                    shapes = [(x0, band_y0, x1, band_y1)]
                elif family == "hair":
                    strand = max(3, width // 12)
                    shapes = [
                        (x0 + offset, y0, x0 + offset + strand, y0 + int(0.72 * height))
                        for offset in (width // 5, width // 2, 4 * width // 5)
                    ]
                elif family == "hand":
                    shapes = [
                        (x0 + width // 2, y0 + height // 2, x1, y1)
                    ]
                else:
                    tear_w = max(2, width // 18)
                    shapes = [
                        (x0 + width // 3, y0 + height // 2, x0 + width // 3 + tear_w, y1),
                        (x0 + 2 * width // 3, y0 + height // 2, x0 + 2 * width // 3 + tear_w, y1),
                    ]
                color = {
                    "eyewear": (28, 28, 32, 210),
                    "goggles": (35, 90, 130, 225),
                    "hair": (45, 28, 20, 200),
                    "hand": (184, 130, 105, 220),
                    "tears": (120, 190, 235, 180),
                }[family]
                for shape in shapes:
                    draw.rounded_rectangle(shape, radius=max(1, width // 30), fill=color)
                    alpha_draw.rounded_rectangle(
                        shape, radius=max(1, width // 30), fill=255
                    )
                target = Image.alpha_composite(target.convert("RGBA"), overlay).convert("RGB")
                occluder_mask = np.asarray(alpha, dtype=np.float32) / 255.0

        candidates = record["_reference_candidates"]
        reference_record = random.choice(candidates)
        # CL5: extra distinct crops feed the PhotoMaker ID encoder only. Sampling
        # them here keeps ref_images[0] — the spatial lane — exactly as it was.
        identity_extra = []
        if self.num_identity_refs > 1:
            pool = [c for c in candidates if c["path"] != reference_record["path"]]
            random.shuffle(pool)
            identity_extra = pool[: self.num_identity_refs - 1]
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
        requested_fraction = None
        position_offset = (0.0, 0.0)
        if self.reference_scale_jitter is not None:
            low, high = self.reference_scale_jitter
            requested_fraction = random.uniform(low, high)
        if self.reference_position_jitter > 0.0:
            jitter = self.reference_position_jitter
            position_offset = (
                random.uniform(-jitter, jitter),
                random.uniform(-jitter, jitter),
            )

        frame_telemetry = None
        if self.reference_frame_mode == "target_face_frame":
            # 06 Aug 2026 - Present the reference face to the frozen VAE/U-Net at
            # the target's own scale and centre, so the branched spatial lane
            # receives matched-granularity features instead of a 2.1x oversized,
            # 4x upscaled crop. See analysis/2026-08-06_cosmic_large_vs_...
            (
                reference,
                reference_bbox,
                policy_descriptor,
                frame_telemetry,
            ) = compose_target_frame_reference(
                reference,
                reference_bbox,
                target_bbox,
                canvas_size=1024,
                fill=self.reference_frame_fill,
                gray_level=self.reference_canvas_fill,
                target_face_fraction=requested_fraction,
                position_offset=position_offset,
            )
            if requested_fraction is None:
                ratio = float(frame_telemetry["scale_ratio"])
                if not 0.95 <= ratio <= 1.05:
                    raise ValueError(
                        f"target-frame reference scale ratio {ratio:.4f} is outside "
                        f"[0.95, 1.05] for {reference_path}"
                    )
            else:
                # Under jitter the target is the requested fraction, not parity
                # with the target face, so assert against what was asked for.
                realised = float(frame_telemetry["face_fraction"])
                if not 0.9 <= realised / requested_fraction <= 1.1:
                    raise ValueError(
                        f"requested reference face fraction {requested_fraction:.4f} "
                        f"but realised {realised:.4f} for {reference_path}"
                    )
        else:
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

        alternate_reference = None
        alternate_reference_bbox = None
        if self.same_identity_dual_reference:
            alternate_pool = [
                candidate
                for candidate in candidates
                if candidate["path"] != reference_record["path"]
            ]
            if not alternate_pool:
                raise RuntimeError("Dual-reference record has no alternate candidate")
            alternate_record = random.choice(alternate_pool)
            alternate_reference = self._open(alternate_record["path"])
            alternate_reference_bbox = deepcopy(alternate_record["bbox"])
            if self.reference_frame_mode == "target_face_frame":
                alt_fraction = (
                    None
                    if self.reference_scale_jitter is None
                    else random.uniform(*self.reference_scale_jitter)
                )
                alt_jitter = self.reference_position_jitter
                alt_offset = (
                    random.uniform(-alt_jitter, alt_jitter),
                    random.uniform(-alt_jitter, alt_jitter),
                )
                (
                    alternate_reference,
                    alternate_reference_bbox,
                    _,
                    _,
                ) = compose_target_frame_reference(
                    alternate_reference,
                    alternate_reference_bbox,
                    target_bbox,
                    canvas_size=1024,
                    fill=self.reference_frame_fill,
                    gray_level=self.reference_canvas_fill,
                    target_face_fraction=alt_fraction,
                    position_offset=alt_offset,
                )
            else:
                (
                    alternate_reference,
                    alternate_reference_bbox,
                    _,
                ) = apply_reference_policy(
                    alternate_reference,
                    alternate_reference_bbox,
                    crop_margin=self.reference_crop_margin,
                    content_size=self.reference_content_size,
                    canvas_size=self.reference_canvas_size,
                    canvas_fill=self.reference_canvas_fill,
                )

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
        identity_extra_images = [self._open(extra["path"]) for extra in identity_extra]
        identity_reference_bboxes = [deepcopy(reference_bbox)] + [
            deepcopy(extra["bbox"]) for extra in identity_extra
        ]
        instance_data = {
            "pixel_values": target,
            "face_bbox": target_bbox,
            "bbox": deepcopy(target_bbox),
            "ref_images": [reference] + identity_extra_images,
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
        if self.num_identity_refs > 1:
            instance_data["identity_face_bboxes_ref"] = identity_reference_bboxes
        if self.semantic_occlusion_probability > 0.0:
            if occluder_mask is None:
                raise RuntimeError("Semantic occlusion mask was not initialized")
            instance_data["ba_occluder_mask"] = occluder_mask[None]
        if self.same_identity_dual_reference:
            instance_data["spatial_ref_images_alt"] = [alternate_reference]
            instance_data["face_bbox_ref_alt"] = alternate_reference_bbox
            instance_data["spatial_reference_alt_path"] = str(
                self.dataset_root / alternate_record["path"]
            )
        instance_data = self.preprocess_data(instance_data)
        if not valid_bbox(instance_data["face_bbox"], (1024, 1024)):
            raise ValueError(
                f"Invalid transformed target bbox: {instance_data['face_bbox']}"
            )
        if not valid_bbox(reference_bbox, reference.size):
            raise ValueError(f"Invalid transformed reference bbox: {reference_bbox}")
        return instance_data


cosmic_large_adapted = CosmicLargeAdaptedTrain
