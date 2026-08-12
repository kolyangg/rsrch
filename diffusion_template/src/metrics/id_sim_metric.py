import torch
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim


class IDSimBest(BaseMetric):
    def __init__(
        self,
        id_embeds_pth,
        device,
        metric_name="id_sim",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.metric_name = str(metric_name)
        self.aligner = Aligner()
        self.id_embeds = torch.load(id_embeds_pth)
        
    def to_cpu(self):
        pass

    def to_cuda(self):
        pass

    def __call__(self, **batch):
        self._ensure_identity_embed(batch)
        batch_bboxes, batch_embeds = self.aligner(batch['generated'])

        result = 0
        for face_bboxes, facce_embeds in zip(batch_bboxes, batch_embeds):
            if facce_embeds is None:
                continue
            score = self.choose_face(facce_embeds, face_bboxes, batch["id"])
            result +=  score
        result = result / len(batch_embeds)
        return {self.metric_name: result}

    def _ensure_identity_embed(self, batch):
        person_id = batch["id"]
        if person_id in self.id_embeds:
            return

        ref_images = batch.get("ref_images")
        if not ref_images:
            raise KeyError(
                f"Identity '{person_id}' is absent from the configured ID embeddings "
                "and no reference image was supplied."
            )
        if not isinstance(ref_images, (list, tuple)):
            ref_images = [ref_images]

        _, ref_embeds = self.aligner(ref_images)
        ref_embed = next((embeds[0] for embeds in ref_embeds if embeds), None)
        if ref_embed is None:
            raise RuntimeError(
                f"No face was detected in the reference image for identity '{person_id}'."
            )

        # 24 Jul 2026 - New validation identities use their reference
        # face when no precomputed embedding exists; historical known-ID
        # metrics remain byte-for-byte on the original lookup path.
        # AICODE-NOTE: Cache this per metric instance to avoid re-running
        # InsightFace for every prompt using the same held-out reference.
        self.id_embeds[person_id] = torch.as_tensor(ref_embed)

    def choose_face(self, embeds, bboxes, person_id):
        best_score = -np.inf
        # get available ids in self.embeds
        # available_ids = self.id_embeds.keys()
        # print(f"Available IDs: {available_ids}")
        for embed in embeds:
            best_score = max(cos_sim(embed, self.id_embeds[person_id]), best_score)
        return best_score


class IDSimMax(IDSimBest):
    def choose_face(self, embeds, bboxes, person_id):
        best_score = -np.inf
        pairs = list(zip(embeds, bboxes))
        pairs = sorted(pairs, key=lambda x: -(x[1][3] - x[1][1]) * (x[1][2] - x[1][0]))
        best_embed = pairs[0][0]
        best_score = cos_sim(best_embed, self.id_embeds[person_id])
        return best_score


class IDSimMaskMatched(IDSimBest):
    """Identity similarity for the generated face owned by the target mask."""

    def __init__(
        self,
        *args,
        minimum_mask_iou=0.05,
        ambiguity_iou_margin=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.minimum_mask_iou = float(minimum_mask_iou)
        self.ambiguity_iou_margin = float(ambiguity_iou_margin)

    @staticmethod
    def _bbox_iou(first, second):
        from src.face_subject_selector import bbox_iou

        return bbox_iou(first, second)

    def __call__(self, **batch):
        self._ensure_identity_embed(batch)
        target_bbox = batch.get("face_bbox_gen")
        if (
            not isinstance(target_bbox, (list, tuple))
            or len(target_bbox) != 4
        ):
            raise ValueError(
                "IDSimMaskMatched requires one four-value face_bbox_gen per sample"
            )

        batch_bboxes, batch_embeds = self.aligner(batch["generated"])
        scores = []
        overlaps = []
        face_counts = []
        no_face = []
        unowned = []
        ambiguous = []
        for face_bboxes, face_embeds in zip(batch_bboxes, batch_embeds):
            count = 0 if face_bboxes is None else len(face_bboxes)
            face_counts.append(float(count))
            if not face_bboxes or not face_embeds:
                scores.append(0.0)
                overlaps.append(0.0)
                no_face.append(1.0)
                unowned.append(1.0)
                ambiguous.append(0.0)
                continue

            ranked = sorted(
                (
                    (self._bbox_iou(box, target_bbox), index, embed)
                    for index, (box, embed) in enumerate(zip(face_bboxes, face_embeds))
                ),
                key=lambda item: (-item[0], item[1]),
            )
            best_iou, _index, best_embed = ranked[0]
            is_ambiguous = bool(
                len(ranked) > 1
                and ranked[1][0] >= self.minimum_mask_iou
                and abs(best_iou - ranked[1][0]) <= self.ambiguity_iou_margin
            )
            is_unowned = best_iou < self.minimum_mask_iou
            overlaps.append(float(best_iou))
            no_face.append(0.0)
            unowned.append(float(is_unowned))
            ambiguous.append(float(is_ambiguous))
            # 09 Aug 2026 - AICODE-NOTE: an off-mask identity fragment must not
            # become a validation win. Preserve the zero together with the
            # ownership diagnostics instead of scoring an unrelated body.
            scores.append(
                0.0
                if is_unowned
                else float(cos_sim(best_embed, self.id_embeds[batch["id"]]))
            )

        divisor = float(max(len(batch_embeds), 1))
        return {
            self.metric_name: sum(scores) / divisor,
            "id_sim_mask_iou": sum(overlaps) / divisor,
            "id_sim_face_count": sum(face_counts) / divisor,
            "id_sim_no_face": sum(no_face) / divisor,
            "id_sim_unowned": sum(unowned) / divisor,
            "id_sim_ambiguous": sum(ambiguous) / divisor,
        }
