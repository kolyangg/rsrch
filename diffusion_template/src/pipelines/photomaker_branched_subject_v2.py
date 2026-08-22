"""Declared-face subject-v2 validation wrapper for corrected E13 leaves."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from src.face_subject_selector import BBOX_OVERLAP_V2, select_subject_face
from src.model.photomaker_branched.insightface_package import analyze_faces
from src.pipelines.br_pipeline_helpers import (
    build_pipeline_from_pretrained,
    ensure_face_analyzer,
)
from src.pipelines.photomaker_branched_clean import (
    PhotoMakerStableDiffusionXLPipeline,
)


def _subject_v2_id_embeds(
    pipeline,
    *,
    input_id_images: Sequence[Any],
    face_bbox_ref,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve the declared validation subject without editing CL14 code."""
    ensure_face_analyzer(pipeline)
    per_prompt = bool(
        isinstance(input_id_images, (list, tuple))
        and input_id_images
        and isinstance(input_id_images[0], (list, tuple))
    )
    if per_prompt:
        refs = [items[0] for items in input_id_images]
        if (
            isinstance(face_bbox_ref, (list, tuple))
            and len(face_bbox_ref) == len(refs)
            and all(
                bbox is None
                or (isinstance(bbox, (list, tuple)) and len(bbox) == 4)
                for bbox in face_bbox_ref
            )
        ):
            declared_bboxes = list(face_bbox_ref)
        else:
            declared_bboxes = [face_bbox_ref] * len(refs)
    else:
        refs = list(input_id_images)
        declared_bboxes = [None] * len(refs)
        if declared_bboxes:
            declared_bboxes[0] = face_bbox_ref

    embeddings = []
    selections = []
    for ref, declared_bbox in zip(refs, declared_bboxes):
        if isinstance(ref, torch.Tensor):
            image = ref.detach().cpu()
            image = image.unsqueeze(0) if image.dim() == 3 else image
            image = (image[0] * 0.5 + 0.5).clamp(0, 1)
            rgb = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            rgb = np.array(ref.convert("RGB"))
        faces = analyze_faces(pipeline._face_analyzer, rgb[:, :, ::-1])
        if not faces:
            raise RuntimeError(
                "Subject-v2 identity conditioning found no face in a reference image"
            )
        selected, audit = select_subject_face(
            faces,
            declared_bbox=declared_bbox,
            policy=BBOX_OVERLAP_V2,
        )
        embeddings.append(torch.from_numpy(selected["embedding"]).float())
        selections.append(audit.to_dict())

    pipeline._face_subject_selections = selections
    stacked = torch.stack(embeddings, dim=0)
    stacked = stacked.unsqueeze(1) if per_prompt else stacked.unsqueeze(0)
    return stacked.to(device=device, dtype=dtype)


class PhotoMakerStableDiffusionXLSubjectV2Pipeline(
    PhotoMakerStableDiffusionXLPipeline
):
    def __call__(
        self,
        *args,
        input_id_images=None,
        face_bbox_ref=None,
        id_embeds: Optional[torch.Tensor] = None,
        face_subject_selection_policy: str = BBOX_OVERLAP_V2,
        **kwargs,
    ):
        if str(face_subject_selection_policy).lower() != BBOX_OVERLAP_V2:
            raise ValueError("Subject-v2 validation requires bbox_overlap_v2")
        if id_embeds is None:
            # 12 Aug 2026 - Corrected-r2 validation binds PhotoMaker identity to
            # the declared face box while leaving sealed CL14 unchanged.
            id_embeds = _subject_v2_id_embeds(
                self,
                input_id_images=input_id_images,
                face_bbox_ref=face_bbox_ref,
                device=self._execution_device,
                dtype=next(self.id_encoder.parameters()).dtype,
            )
        return super().__call__(
            *args,
            input_id_images=input_id_images,
            face_bbox_ref=face_bbox_ref,
            id_embeds=id_embeds,
            **kwargs,
        )


class PhotomakerBranchedSubjectV2Pipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        return build_pipeline_from_pretrained(
            PhotoMakerStableDiffusionXLSubjectV2Pipeline,
            model=model,
            accelerator=accelerator,
            args=args,
            kwargs=kwargs,
        )
