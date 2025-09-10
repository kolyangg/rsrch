# src/metrics/id_sim_metric_NS.py

import numpy as np
from src.metrics.base_metric import BaseMetric
from src.id_utils.aligner import Aligner      # produces InsightFace embeddings

def cos_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

class IDSimOnDemand(BaseMetric):
    """
    Computes cosine similarity between the *largest* face in the reference
    image and every face in the generated image, then averages.
    Expects keys:
        batch["reference"] :  list[PIL.Image]  (len==1 is fine)
        batch["generated"] :  list[PIL.Image]
    """
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = Aligner()            # runs on CPU by default
        self.last_no_face = 0               # 0 = OK, 1 = missing face

    def __call__(self, **batch):
        ref_crops, _, ref_embeds = self.aligner(batch["reference"])
        
        if len(ref_embeds) == 0:
            print(f"NO FACE IN REFERENCE: {batch.get('reference_name','?')}")
            self.last_no_face = 1
            return 0.0
        ref_vec = ref_embeds[0]

        gen_crops, _, gen_embeds = self.aligner(batch["generated"])
        if len(gen_embeds) == 0:
            # print("NO FACE IN GENERATED")
            print(f"NO FACE IN GENERATED: {batch.get('generated_name','?')}")
            self.last_no_face = 1
            return 0.0

        sims = [cos_sim(e, ref_vec) for e in gen_embeds]
        return float(np.mean(sims))
