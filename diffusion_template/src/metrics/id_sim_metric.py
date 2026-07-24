import torch
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.metrics.aligner import Aligner
from src.utils.model_utils import cos_sim


class IDSimBest(BaseMetric):
    def __init__(self, id_embeds_pth, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
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
        return {"id_sim": result}

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
