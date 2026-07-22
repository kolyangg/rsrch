from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from src.model.photomaker_branched.model_v2_NS import (
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken,
)
from src.metrics.tracker import MetricTracker
from src.pipelines.br_pipeline_helpers import prepare_spatial_identity_tokens
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ppr_reference_noise import _assert_integrity, _variants


class _MeanEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, images):
        means = images.mean(dim=(2, 3))
        return torch.cat([means, means.square()], dim=1)


class _CountingOptimizer:
    def __init__(self):
        self.steps = 0
        self.zero_grad_calls = 0

    def zero_grad(self, *args, **kwargs):
        del args, kwargs
        self.zero_grad_calls += 1

    def step(self):
        self.steps += 1


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _AccumulationHarness(BaseTrainer):
    def __init__(self):
        self.is_train = True
        self.grad_accum_steps = 2
        self.epoch_len = 2
        self.train_dataloader = [{}, {}, {}, {}]
        self.evaluation_dataloaders = {}
        self.train_metrics = MetricTracker()
        self.optimizer = _CountingOptimizer()
        self.lr_scheduler = _CountingScheduler()
        self.processed_indices = []
        self.process_calls = 0
        self.model = nn.Identity()
        self.accelerator = SimpleNamespace(
            is_main_process=False,
            unwrap_model=lambda model: model,
            wait_for_everyone=lambda: None,
        )
        self.config = SimpleNamespace(
            pretrained_model_for_validation_name_or_path=None,
        )
        self.log_step = 1000

    def process_batch(self, batch, train_metrics):
        del train_metrics
        self.process_calls += 1
        batch_idx = int(batch["batch_idx"])
        self.processed_indices.append(batch_idx)
        if self.process_calls == 2:
            return {"skip_batch": True}
        if batch_idx % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
        return {"loss": torch.ones(())}

    def _get_grad_norms(self):
        return {}


class NN5ComponentTests(unittest.TestCase):
    def test_direct_spatial_checkpoint_preflight_uses_lora_b(self):
        trainer = SimpleNamespace(
            config=SimpleNamespace(ppr_checkpoint_require_nonzero=True),
            model=SimpleNamespace(
                _last_ppr_checkpoint_diagnostics={
                    "connector_up_tensors": 0,
                    "connector_up_nonzero": 0,
                    "connector_up_l2": 0.0,
                    "direct_spatial_tensors": 2,
                    "direct_spatial_nonzero": 8,
                    "direct_spatial_l2": 0.125,
                    "gate_min": 0.09,
                    "gate_max": 0.11,
                }
            ),
            accelerator=SimpleNamespace(
                unwrap_model=lambda model: model,
                is_main_process=False,
            ),
            logger=None,
        )
        BaseTrainer._check_ppr_checkpoint_preflight(trainer)
        trainer.model._last_ppr_checkpoint_diagnostics[
            "direct_spatial_nonzero"
        ] = 0
        with self.assertRaisesRegex(RuntimeError, "zero K/V LoRA-B"):
            BaseTrainer._check_ppr_checkpoint_preflight(trainer)

    def test_nn7_spatial_patch_projection_modes(self):
        class Vision(nn.Module):
            def forward(self, pixels):
                batch = pixels.shape[0]
                values = torch.arange(
                    batch * 5 * 1024,
                    dtype=pixels.dtype,
                    device=pixels.device,
                ).reshape(batch, 5, 1024)
                return (values / 1000.0,)

        proj_in = nn.Linear(1024, 2048, bias=False)
        norm1 = nn.LayerNorm(2048)
        for parameter in (*proj_in.parameters(), *norm1.parameters()):
            parameter.requires_grad_(False)
        encoder = SimpleNamespace(
            vision_model=Vision(),
            qformer_perceiver=SimpleNamespace(
                perceiver_resampler=SimpleNamespace(
                    proj_in=proj_in,
                    layers=[[SimpleNamespace(norm1=norm1)]],
                )
            ),
        )
        pixels = torch.zeros(2, 1, 3, 4, 4)
        raw = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.extract_spatial_patch_tokens(
            encoder,
            pixels,
            projection="raw_clip",
        )
        context = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.extract_spatial_patch_tokens(
            encoder,
            pixels,
            projection="pmv2_perceiver_context",
        )
        self.assertEqual(tuple(raw.shape), (2, 4, 1024))
        self.assertEqual(tuple(context.shape), (2, 4, 2048))
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(raw.shape[1], context.shape[1])
        self.assertFalse(any(parameter.requires_grad for parameter in proj_in.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in norm1.parameters()))

    def test_accumulation_window_rewinds_after_second_microbatch_skip(self):
        trainer = _AccumulationHarness()
        trainer._train_epoch(epoch=2)
        self.assertEqual(trainer.processed_indices, [0, 1, 0, 1])
        self.assertEqual(trainer.optimizer.steps, 1)
        self.assertEqual(trainer.lr_scheduler.steps, 1)
        self.assertEqual(trainer._optimizer_step_from_microbatches(4), 2)

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

    def test_identity_lane_projection_weights_receive_gradients(self):
        torch.manual_seed(17)
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
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.1)
        mask = torch.ones(1, 1, side, side)
        processor.set_masks(mask, mask, make_inner_core_mask(mask))
        processor.identity_tokens = torch.randn(1, 2, 2048)
        output = processor(
            attention,
            torch.randn(2, side * side, channels),
        )
        output[:1].square().mean().backward()
        for projection in (processor.identity_to_k, processor.identity_to_v):
            for layer_index in (0, 2):
                gradient = projection[layer_index].weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_nn6_identity_only_is_spatially_independent_and_uses_shared_kv(self):
        torch.manual_seed(23)
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
            connector_input_mode="reference_minus_learned_null",
            identity_token_lane=True,
            identity_token_rank=4,
            identity_fusion_mode="identity_only",
            enable_identity=True,
            enable_spatial=False,
            identity_connector_rank=4,
        )
        processor.init_from_attention(attention)
        self.assertIsNone(processor.ref_to_k)
        self.assertIsNone(processor.ref_to_v)
        self.assertIsNone(processor.connector_down)
        self.assertIsNone(processor.null_memory)
        mask = torch.ones(1, 1, side, side)
        processor.set_masks(mask, mask, make_inner_core_mask(mask))
        processor.identity_tokens = torch.randn(1, 2, 2048)
        target = torch.randn(1, side * side, channels)
        reference_a = torch.randn_like(target)
        reference_b = torch.randn_like(target) * 9.0

        call_counts = {"k": 0, "v": 0}
        hooks = [
            processor.identity_to_k.register_forward_hook(
                lambda *unused: call_counts.__setitem__("k", call_counts["k"] + 1)
            ),
            processor.identity_to_v.register_forward_hook(
                lambda *unused: call_counts.__setitem__("v", call_counts["v"] + 1)
            ),
        ]
        zero_a = processor(attention, torch.cat([target, reference_a], dim=0))
        zero_b = processor(attention, torch.cat([target, reference_b], dim=0))
        for hook in hooks:
            hook.remove()
        self.assertEqual(call_counts, {"k": 4, "v": 4})
        torch.testing.assert_close(zero_a[0], zero_b[0], atol=0, rtol=0)

        with torch.no_grad():
            processor.identity_connector_up.weight.normal_(std=0.1)
        output_a = processor(attention, torch.cat([target, reference_a], dim=0))
        output_b = processor(attention, torch.cat([target, reference_b], dim=0))
        torch.testing.assert_close(output_a[0], output_b[0], atol=0, rtol=0)
        first_identity = output_a[0].clone()
        processor.identity_tokens = torch.randn(1, 2, 2048)
        second_identity = processor(
            attention,
            torch.cat([target, reference_a], dim=0),
        )[0]
        self.assertFalse(torch.equal(first_identity, second_identity))

        second_identity.square().mean().backward()
        self.assertGreater(
            float(processor.identity_connector_down.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(processor.identity_to_k[0].weight.grad.abs().sum()),
            0.0,
        )

    def test_invalid_spatial_identity_embedding_is_rejected(self):
        class _ImageProcessor:
            def __call__(self, images, return_tensors):
                del return_tensors
                return SimpleNamespace(
                    pixel_values=torch.ones(len(images), 3, 4, 4)
                )

        pipeline = SimpleNamespace(
            ba_identity_token_lane=True,
            id_encoder=SimpleNamespace(
                dtype=torch.float32,
                extract_id_tokens=lambda pixels, embeds: torch.ones(
                    pixels.shape[0], 2, 2048
                ),
            ),
            id_image_processor=_ImageProcessor(),
            unet=SimpleNamespace(dtype=torch.float32),
        )
        refs = [Image.new("RGB", (4, 4)) for _ in range(2)]
        with patch(
            "src.pipelines.br_pipeline_helpers.ensure_id_embeds",
            return_value=torch.zeros(2, 1, 512),
        ):
            with self.assertRaisesRegex(RuntimeError, "valid spatial-reference"):
                prepare_spatial_identity_tokens(
                    pipeline,
                    input_id_images=refs,
                    device=torch.device("cpu"),
                )

    def test_identity_token_swap_integrity(self):
        sample = "sample.png"
        content = {
            "PM0": ("r1", "l1", "m1", "n1", "r1n1", "token-r1"),
            "R1N1": ("r1", "l1", "m1", "n1", "r1n1", "token-r1"),
            "R2N1": ("r2", "l2", "m2", "n1", "r2n1", "token-r2"),
            "R1N2": ("r1", "l1", "m1", "n2", "r1n2", "token-r1"),
            "R2N2": ("r2", "l2", "m2", "n2", "r2n2", "token-r2"),
        }
        fingerprints = {}
        for name, (image, latent, mask, noise, noised, token) in content.items():
            fingerprints[name] = {
                "initial_latents_sha256": "target-latent",
                "target_prompt_embeds_sha256": "target-prompt",
                "target_photomaker_id_embeds_sha256": "target-pm-id",
                "spatial_reference_image_sha256": image,
                "reference_latents_sha256": latent,
                "reference_mask_sha256": mask,
                "reference_mask_nonempty": True,
                "reference_noise_sha256": noise,
                "ref_noised_step_15_sha256": noised + "-15",
                "ref_noised_step_25_sha256": noised + "-25",
                "ref_noised_step_35_sha256": noised + "-35",
                "reference_ca_prompt_sha256": "reference-ca",
                "reference_ca_mode": "original",
                "spatial_identity_tokens_sha256": token,
            }
        diagnostics = {
            "PM0": [
                {
                    "record_type": "epsilon_ratio",
                    "output_control": "diagnostic-force-base",
                }
            ]
        }
        for name in ("R1N1", "R2N1", "R1N2", "R2N2"):
            diagnostics[name] = [
                {
                    "record_type": "processor_tensor_signature",
                    "roi_tokens": 1,
                },
                {
                    "record_type": "processor_applied_ratio",
                    "samples": [sample],
                    "applied_ratios": [0.1],
                    "cap_scales": [1.0],
                },
            ]

        _assert_integrity(
            sample,
            fingerprints,
            diagnostics,
            "original",
            identity_token_lane=True,
        )
        fingerprints["R2N1"]["spatial_identity_tokens_sha256"] = "token-r1"
        with self.assertRaisesRegex(RuntimeError, "did not change identity tokens"):
            _assert_integrity(
                sample,
                fingerprints,
                diagnostics,
                "original",
                identity_token_lane=True,
            )

    def test_identity_only_zero_tolerance_requires_exact_tensor_hashes(self):
        sample = "sample.png"
        content = {
            "PM0": ("r1", "l1", "m1", "n1", "r1n1", "token-r1"),
            "R1N1": ("r1", "l1", "m1", "n1", "r1n1", "token-r1"),
            "R2N1": ("r2", "l2", "m2", "n1", "r2n1", "token-r2"),
            "R1N2": ("r1", "l1", "m1", "n2", "r1n2", "token-r1"),
            "R2N2": ("r2", "l2", "m2", "n2", "r2n2", "token-r2"),
        }
        fingerprints = {
            name: {
                "initial_latents_sha256": "target-latent",
                "target_prompt_embeds_sha256": "target-prompt",
                "target_photomaker_id_embeds_sha256": "target-pm-id",
                "spatial_reference_image_sha256": image,
                "reference_latents_sha256": latent,
                "reference_mask_sha256": mask,
                "reference_mask_nonempty": True,
                "reference_noise_sha256": noise,
                "ref_noised_step_15_sha256": noised + "-15",
                "ref_noised_step_25_sha256": noised + "-25",
                "ref_noised_step_35_sha256": noised + "-35",
                "reference_ca_prompt_sha256": "reference-ca",
                "reference_ca_mode": "original",
                "spatial_identity_tokens_sha256": token,
            }
            for name, (image, latent, mask, noise, noised, token) in content.items()
        }
        stages = (
            "identity_candidate",
            "identity_null_candidate",
            "identity_connector_input",
            "identity_raw_delta",
            "identity_bounded_delta",
            "identity_applied_delta",
            "combined_applied_delta",
        )

        def signature(digest):
            return {"sha256": digest, "sketch": [1.0, 2.0]}

        diagnostics = {
            "PM0": [
                {
                    "record_type": "epsilon_ratio",
                    "output_control": "diagnostic-force-base",
                }
            ]
        }
        for name in ("R1N1", "R2N1", "R1N2", "R2N2"):
            processor_record = {
                "record_type": "processor_tensor_signature",
                "step": 15,
                "processor": "up_blocks.0.attn1",
                "roi_tokens": 2,
            }
            processor_record.update(
                {stage: signature(f"{name[:2]}-{stage}") for stage in stages}
            )
            # Same identity and identical 512-value sketch, but a different
            # full-tensor hash under N2: exact tolerance must reject this.
            if name == "R1N2":
                processor_record["identity_candidate"] = signature(
                    "different-full-tensor-hash"
                )
            epsilon_record = {
                "record_type": "epsilon_tensor_signature",
                "step": 15,
                "processor": "",
                "target_epsilon_pre_anchor": signature(f"{name[:2]}-pre"),
                "target_epsilon_post_anchor": signature(f"{name[:2]}-post"),
            }
            diagnostics[name] = [
                processor_record,
                epsilon_record,
                {
                    "record_type": "processor_applied_ratio",
                    "samples": [sample],
                    "applied_ratios": [0.1],
                    "cap_scales": [1.0],
                },
            ]

        with self.assertRaisesRegex(
            RuntimeError,
            "identity-only reference-noise leak.*exact=False",
        ):
            _assert_integrity(
                sample,
                fingerprints,
                diagnostics,
                "original",
                identity_token_lane=True,
                identity_fusion_mode="identity_only",
                identity_noise_tolerance=0.0,
            )

    def test_reference_noise_variants_use_requested_scale(self):
        variants = _variants(1.0)
        self.assertEqual(variants["PM0"][0], 0.0)
        self.assertTrue(all(variants[name][0] == 1.0 for name in variants if name != "PM0"))


if __name__ == "__main__":
    unittest.main()
