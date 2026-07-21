from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from PIL import Image

from src.datasets.cosmic import CosmicLargeTrain
from src.datasets.collate import collate_fn
from src.loss.id_loss import CounterfactualIdentityLoss, IdentityLoss
from src.model.photomaker_branched.packed_residual_attn_processor import (
    PackedResidualBranchedAttnProcessor,
    make_inner_core_mask,
)
from src.trainer.ppr_reference_noise import _variants


class _MeanEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, images):
        means = images.mean(dim=(2, 3))
        return torch.cat([means, means.square()], dim=1)


class NN5ComponentTests(unittest.TestCase):
    def test_identity_key_precedence_and_prompt_class(self):
        record = {
            "identity_id": "stable-id",
            "person_id": "ignored",
            "face_paths": ["root/person/image.jpg"],
            "facial_caption": "portrait of a Woman outdoors",
        }
        self.assertEqual(CosmicLargeTrain._identity_key(record, "x.jpg"), "stable-id")
        self.assertEqual(CosmicLargeTrain._prompt_class(record), "woman")

    def test_counterfactual_dataset_collates_distinct_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for index, person in enumerate(("person_a", "person_b")):
                person_dir = root / person
                person_dir.mkdir()
                target_path = root / f"target_{index}.jpg"
                ref_path = person_dir / "ref.jpg"
                Image.new("RGB", (1024, 1024), (80 + index * 80, 100, 120)).save(target_path)
                Image.new("RGB", (512, 512), (90 + index * 80, 110, 130)).save(ref_path)
                records[str(target_path)] = {
                    "face_crop_new": [200, 200, 500, 500],
                    "face_paths": [str(ref_path)],
                    "face_bboxes": {str(ref_path): [100, 100, 400, 400]},
                    "facial_caption": "portrait of a woman",
                    "pose_caption": "standing",
                    "background_caption": "studio",
                }
            json_path = root / "data.json"
            json_path.write_text(json.dumps(records), encoding="utf-8")

            def to_tensor(image):
                return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)

            dataset = CosmicLargeTrain(
                data_json_path=json_path,
                images_path=root,
                min_face_res=10,
                ref_crop_margin_min=0.0,
                return_counterfactual_ref=True,
                counterfactual_same_class_probability=1.0,
                instance_transforms={"pixel_values": to_tensor},
            )
            item = dataset[0]
            self.assertNotEqual(
                item["matched_identity_key"],
                item["counterfactual_identity_key"],
            )
            batch = collate_fn([item])
            self.assertEqual(len(batch["counterfactual_ref_images"]), 1)
            self.assertEqual(len(batch["counterfactual_ref_images"][0]), 1)

    def test_directional_loss_prefers_wrong_reference(self):
        identity = IdentityLoss.__new__(IdentityLoss)
        nn.Module.__init__(identity)
        identity.face_size = 4
        identity.net = _MeanEmbedding()
        loss = CounterfactualIdentityLoss(identity)

        matched = -torch.ones(3, 4, 4)
        wrong = torch.ones(3, 4, 4)
        generated_toward_wrong = torch.ones(1, 3, 4, 4, requires_grad=True)
        generated_toward_matched = -torch.ones(1, 3, 4, 4, requires_grad=True)
        kwargs = dict(
            target_bboxes=[[0, 0, 4, 4]],
            matched_reference_images=[[matched]],
            matched_reference_bboxes=[[0, 0, 4, 4]],
            wrong_reference_images=[[wrong]],
            wrong_reference_bboxes=[[0, 0, 4, 4]],
            margin=0.03,
        )
        toward_wrong = loss(generated_toward_wrong, **kwargs)
        toward_matched = loss(generated_toward_matched, **kwargs)
        self.assertLess(
            float(toward_wrong["directional_loss"]),
            float(toward_matched["directional_loss"]),
        )
        toward_wrong["absolute_loss"].backward()
        self.assertIsNotNone(generated_toward_wrong.grad)

    def test_identity_lane_is_zero_connector_safe_and_token_sensitive(self):
        torch.manual_seed(5)
        channels, side = 16, 4
        attention = Attention(
            query_dim=channels,
            heads=4,
            dim_head=4,
            residual_connection=True,
        )
        processor = PackedResidualBranchedAttnProcessor(
            channels,
            ref_kv_rank=4,
            connector_rank=4,
            connector_input_mode="reference_minus_learned_null",
            identity_token_lane=True,
            identity_token_rank=4,
            identity_token_weight=0.5,
        )
        processor.init_from_attention(attention)
        mask = torch.ones(1, 1, side, side)
        processor.set_masks(mask, mask, make_inner_core_mask(mask))
        hidden = torch.randn(2, side * side, channels)

        processor.identity_tokens = torch.zeros(1, 2, 2048)
        zero_tokens = processor(attention, hidden.clone())
        processor.identity_tokens = torch.ones(1, 2, 2048)
        one_tokens = processor(attention, hidden.clone())
        self.assertTrue(torch.equal(zero_tokens, one_tokens))

        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.1)
        zero_tokens = processor(attention, hidden.clone())
        processor.identity_tokens = torch.zeros(1, 2, 2048)
        zero_tokens = processor(attention, hidden.clone())
        processor.identity_tokens = torch.ones(1, 2, 2048)
        one_tokens = processor(attention, hidden.clone())
        self.assertFalse(torch.equal(zero_tokens, one_tokens))

    def test_reference_noise_variants_use_requested_scale(self):
        variants = _variants(1.0)
        self.assertEqual(variants["PM0"][0], 0.0)
        self.assertTrue(all(variants[name][0] == 1.0 for name in variants if name != "PM0"))


if __name__ == "__main__":
    unittest.main()
