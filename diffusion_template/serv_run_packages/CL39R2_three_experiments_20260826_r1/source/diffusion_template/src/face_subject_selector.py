"""Deterministic face ownership for reference conditioning and validation.

Historical PhotoMaker code used the first InsightFace detection.  Detection
order is not a subject contract: a small bystander can precede the intended
person.  The v2 policy below binds a detection to an explicit face box when
one is available and otherwise falls back to the largest confident face.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


LEGACY_FIRST = "legacy_first"
BBOX_OVERLAP_V2 = "bbox_overlap_v2"
SUPPORTED_POLICIES = {LEGACY_FIRST, BBOX_OVERLAP_V2}


@dataclass(frozen=True)
class SubjectSelection:
    index: int
    bbox: tuple[float, float, float, float]
    detection_score: float | None
    selection_reason: str
    declared_bbox_iou: float | None
    ambiguous: bool
    face_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _face_value(face: Any, key: str, default: Any = None) -> Any:
    if isinstance(face, dict):
        return face.get(key, default)
    try:
        return face[key]
    except Exception:
        return getattr(face, key, default)


def face_bbox(face: Any) -> tuple[float, float, float, float]:
    value = _face_value(face, "bbox")
    if value is None or len(value) != 4:
        raise ValueError("Face detection has no four-value bbox")
    x0, y0, x1, y1 = (float(item) for item in value)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Face detection has invalid bbox: {value}")
    return x0, y0, x1, y1


def bbox_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    ax0, ay0, ax1, ay1 = (float(item) for item in first)
    bx0, by0, bx1, by1 = (float(item) for item in second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _detection_score(face: Any) -> float | None:
    value = _face_value(face, "det_score")
    return None if value is None else float(value)


def select_subject_face(
    faces: Sequence[Any],
    *,
    declared_bbox: Sequence[float] | None = None,
    policy: str = BBOX_OVERLAP_V2,
    minimum_declared_iou: float = 0.05,
    ambiguity_iou_margin: float = 0.02,
    ambiguity_area_ratio: float = 0.95,
    fail_on_ambiguous: bool = True,
) -> tuple[Any, SubjectSelection]:
    """Select the face owned by ``declared_bbox`` and return an audit record.

    ``legacy_first`` is intentionally preserved for exact historical replay.
    Under v2 an explicit box is authoritative.  Without one, the largest face
    is used; a near-equal largest-face tie fails closed by default.
    """

    # 09 Aug 2026 - AICODE-NOTE: detector order is not identity ownership.
    # Subject-v2 must be deterministic, bbox-owned, and fail closed on an
    # ambiguous reference; legacy_first exists only for exact historical replay.
    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unknown face subject policy {policy!r}; expected {sorted(SUPPORTED_POLICIES)}"
        )
    if not faces:
        raise RuntimeError("No face detection is available for subject selection")

    candidates = []
    for index, face in enumerate(faces):
        box = face_bbox(face)
        area = (box[2] - box[0]) * (box[3] - box[1])
        candidates.append(
            {
                "index": index,
                "face": face,
                "bbox": box,
                "area": area,
                "score": _detection_score(face),
                "iou": None if declared_bbox is None else bbox_iou(box, declared_bbox),
            }
        )

    if normalized_policy == LEGACY_FIRST:
        best = candidates[0]
        audit = SubjectSelection(
            index=0,
            bbox=best["bbox"],
            detection_score=best["score"],
            selection_reason=LEGACY_FIRST,
            declared_bbox_iou=best["iou"],
            ambiguous=False,
            face_count=len(candidates),
        )
        return best["face"], audit

    if declared_bbox is not None:
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item["iou"]),
                -(item["score"] if item["score"] is not None else -1.0),
                -item["area"],
                item["index"],
            ),
        )
        best = ranked[0]
        if float(best["iou"]) < float(minimum_declared_iou):
            raise RuntimeError(
                "No detected face overlaps the declared subject box: "
                f"best_iou={best['iou']:.6f}, minimum={minimum_declared_iou:.6f}"
            )
        ambiguous = bool(
            len(ranked) > 1
            and float(ranked[1]["iou"]) >= float(minimum_declared_iou)
            and abs(float(best["iou"]) - float(ranked[1]["iou"]))
            <= float(ambiguity_iou_margin)
        )
        reason = "declared_bbox_max_iou"
    else:
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item["area"],
                -(item["score"] if item["score"] is not None else -1.0),
                item["index"],
            ),
        )
        best = ranked[0]
        ambiguous = bool(
            len(ranked) > 1
            and ranked[1]["area"] / max(best["area"], 1e-12)
            >= float(ambiguity_area_ratio)
        )
        reason = "largest_face_fallback"

    if ambiguous and fail_on_ambiguous:
        raise RuntimeError(
            "Ambiguous subject selection: the top two detections are within "
            "the configured ownership margin"
        )
    audit = SubjectSelection(
        index=int(best["index"]),
        bbox=best["bbox"],
        detection_score=best["score"],
        selection_reason=reason,
        declared_bbox_iou=(
            None if best["iou"] is None else float(best["iou"])
        ),
        ambiguous=ambiguous,
        face_count=len(candidates),
    )
    return best["face"], audit
