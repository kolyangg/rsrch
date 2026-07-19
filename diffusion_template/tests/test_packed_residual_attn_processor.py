from __future__ import annotations

import os
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from PIL import Image

from src.model.photomaker_branched.branched_runtime import (
    patch_unet_attention_processors,
    select_branched_self_attention_names,
    two_branch_predict,
)
from src.model.photomaker_branched.lora2_helpers import (
    _assert_branched_installation,
    configure_branched_trainables,
    install_branched_processors_for_training,
)
from src.model.photomaker_branched.lora2 import PhotomakerBranchedLora
from src.model.photomaker_branched.packed_residual_attn_processor import (
    PackedResidualBranchedAttnProcessor,
    make_inner_core_mask,
    pack_valid_tokens,
)
from src.pipelines.br_pipeline_helpers import reset_branched_generation_caches
from src.trainer.ppr_diagnostic import (
    METRIC_FIELDS,
    _diagnostic_options,
    _initialize_state,
    _pixel_mae,
    _select_spatial_swap_indices,
)
from src.trainer.ppr_scale_sweep import (
    _parse_scales,
    _processor_stats,
    _scale_label,
)


def _attention(channels: int = 16) -> Attention:
    return Attention(
        query_dim=channels,
        heads=4,
        dim_head=channels // 4,
        residual_connection=True,
    )


def _masks(batch: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(batch, 1, side, side)
    margin = max(1, side // 8)
    mask[:, :, margin : side - margin, margin : side - margin] = 1
    return mask, make_inner_core_mask(mask, erode_frac=0.10)


def _processor(attn: Attention, channels: int = 16) -> PackedResidualBranchedAttnProcessor:
    processor = PackedResidualBranchedAttnProcessor(
        channels,
        ref_kv_rank=4,
        connector_rank=4,
        gate_max=0.5,
        delta_rms_cap=0.25,
    )
    processor.init_from_attention(attn)
    return processor


class PackedResidualProcessorTests(unittest.TestCase):
    def test_exact_branch_off_parity_fp32(self) -> None:
        for input_ndim in (3, 4):
            for batch in (1, 2):
                with self.subTest(input_ndim=input_ndim, batch=batch):
                    torch.manual_seed(0)
                    side = 8
                    channels = 16
                    attn = _attention(channels)
                    if input_ndim == 3:
                        hidden = torch.randn(2 * batch, side * side, channels)
                    else:
                        hidden = torch.randn(2 * batch, channels, side, side)
                    mask, core = _masks(batch, side)

                    expected = AttnProcessor2_0()(attn, hidden)
                    processor = _processor(attn, channels)
                    processor.set_masks(mask, mask, core)
                    actual = processor(attn, hidden)

                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    self.assertTrue(torch.equal(actual, expected))

    def test_branch_off_parity_bfloat16(self) -> None:
        torch.manual_seed(7)
        side = 8
        channels = 16
        attn = _attention(channels).to(dtype=torch.bfloat16)
        hidden = torch.randn(2, channels, side, side).to(torch.bfloat16)
        mask, core = _masks(1, side)
        expected = AttnProcessor2_0()(attn, hidden)
        processor = _processor(attn, channels).to(dtype=torch.bfloat16)
        processor.set_masks(mask, mask, core)
        actual = processor(attn, hidden)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(
        os.environ.get("PPR_RUN_FULL_PARITY") == "1",
        "Set PPR_RUN_FULL_PARITY=1 for the 8/16/32/64 FP32/BF16 matrix",
    )
    def test_full_parity_matrix(self) -> None:
        for side in (8, 16, 32, 64):
            for batch in (1, 2):
                for input_ndim in (3, 4):
                    for dtype in (torch.float32, torch.bfloat16):
                        with self.subTest(
                            side=side,
                            batch=batch,
                            input_ndim=input_ndim,
                            dtype=dtype,
                        ):
                            torch.manual_seed(1)
                            channels = 16
                            attn = _attention(channels).to(dtype=dtype)
                            if input_ndim == 3:
                                hidden = torch.randn(
                                    2 * batch,
                                    side * side,
                                    channels,
                                ).to(dtype)
                            else:
                                hidden = torch.randn(
                                    2 * batch,
                                    channels,
                                    side,
                                    side,
                                ).to(dtype)
                            mask, core = _masks(batch, side)
                            expected = AttnProcessor2_0()(attn, hidden)
                            processor = _processor(attn, channels).to(dtype=dtype)
                            processor.set_masks(mask, mask, core)
                            actual = processor(attn, hidden)
                            tolerance = 1e-5 if dtype == torch.float32 else 2e-2
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=tolerance,
                                rtol=tolerance,
                            )

    def test_packed_padding_is_excluded_and_empty_roi_fails_closed(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]],
                [[9.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]],
            ]
        )
        valid = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, False, False],
            ]
        )
        packed, lengths, additive_mask, has_roi = pack_valid_tokens(hidden, valid)
        self.assertEqual(lengths.tolist(), [2, 1, 0])
        self.assertEqual(has_roi.tolist(), [True, True, False])
        self.assertTrue(torch.isneginf(additive_mask[1, 0, 0, 1:]).all())
        self.assertEqual(additive_mask[2, 0, 0, 0], 0)

        query = torch.ones(3, 1, 1, 2)
        key = packed[:, None].clone()
        value = packed[:, None].clone()
        baseline = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
        )
        key[1:, :, 1:] = 1e6
        value[1:, :, 1:] = 1e6
        changed = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
        )
        torch.testing.assert_close(changed, baseline)
        self.assertTrue(torch.isfinite(baseline).all())

    def test_empty_roi_has_exactly_zero_target_residual(self) -> None:
        torch.manual_seed(2)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_()
        target_mask, core = _masks(2, side)
        reference_mask = target_mask.clone()
        reference_mask[0].zero_()
        hidden = torch.randn(4, side * side, 16)
        base = AttnProcessor2_0()(attn, hidden)
        processor.set_masks(target_mask, reference_mask, core)
        actual = processor(attn, hidden)
        torch.testing.assert_close(actual[0], base[0], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_zero_up_stages_gradients_then_opens_reference_lane(self) -> None:
        torch.manual_seed(3)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(4, side * side, 16)

        first = processor(attn, hidden)
        first[:2].square().mean().backward()
        self.assertIsNotNone(processor.connector_up.weight.grad)
        self.assertGreater(processor.connector_up.weight.grad.abs().sum(), 0)
        self.assertEqual(processor.connector_down.weight.grad.abs().sum(), 0)
        self.assertEqual(processor.gate_logit.grad.abs().sum(), 0)
        self.assertEqual(processor.ref_to_k.lora_B.grad.abs().sum(), 0)
        self.assertEqual(processor.ref_to_v.lora_B.grad.abs().sum(), 0)

        with torch.no_grad():
            processor.connector_up.weight.add_(
                -0.1 * processor.connector_up.weight.grad
            )
        processor.zero_grad(set_to_none=True)
        second = processor(attn, hidden)
        second[:2].square().mean().backward()
        self.assertGreater(processor.connector_down.weight.grad.abs().sum(), 0)
        self.assertGreater(processor.gate_logit.grad.abs().sum(), 0)
        self.assertGreater(processor.ref_to_k.lora_B.grad.abs().sum(), 0)
        self.assertGreater(processor.ref_to_v.lora_B.grad.abs().sum(), 0)

    def test_masked_rms_cap_is_per_sample_and_bounded(self) -> None:
        torch.manual_seed(4)
        base = torch.randn(2, 16, 8)
        delta = 100 * torch.randn_like(base)
        mask = torch.ones(2, 16, 1)
        bounded, scale, pre_ratio, post_ratio = (
            PackedResidualBranchedAttnProcessor._masked_rms_cap(
                delta,
                base=base,
                mask=mask,
                max_ratio=0.25,
            )
        )
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue((scale <= 1).all())
        self.assertTrue((pre_ratio > post_ratio).all())
        self.assertTrue((post_ratio <= 0.25001).all())

    def test_retrieval_weights_do_not_change_reference_continuation(self) -> None:
        torch.manual_seed(5)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.1)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(4, side * side, 16)
        before = processor(attn, hidden)
        with torch.no_grad():
            processor.ref_to_k.lora_B.normal_(std=0.5)
            processor.ref_to_v.lora_B.normal_(std=0.5)
        after = processor(attn, hidden)
        torch.testing.assert_close(after[2:], before[2:], atol=0, rtol=0)
        self.assertFalse(torch.equal(after[:2], before[:2]))

    def test_runtime_scale_multiplies_only_applied_target_delta(self) -> None:
        torch.manual_seed(51)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.05)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(4, side * side, 16)

        processor.runtime_scale = 0.0
        base = processor(attn, hidden)
        processor.runtime_scale = 1.0
        scaled_one = processor(attn, hidden)
        processor.runtime_scale = 4.0
        scaled_four = processor(attn, hidden)

        torch.testing.assert_close(
            scaled_four[:2] - base[:2],
            4.0 * (scaled_one[:2] - base[:2]),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(scaled_four[2:], base[2:], atol=0, rtol=0)

    def test_inner_core_is_soft_and_zero_at_bbox_edges(self) -> None:
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 2:14, 3:13] = 1
        core = make_inner_core_mask(mask, erode_frac=0.2)
        self.assertEqual(core.shape, mask.shape)
        self.assertEqual(core[:, :, 2, 3:13].max(), 0)
        self.assertEqual(core[:, :, 13, 3:13].max(), 0)
        self.assertEqual(core[:, :, 2:14, 3].max(), 0)
        self.assertEqual(core[:, :, 2:14, 12].max(), 0)
        self.assertEqual(core.max(), 1)
        self.assertTrue(bool(((core > 0) & (core < 1)).any()))

    def test_up_block_site_policy_is_explicit(self) -> None:
        names = [
            "down_blocks.0.attn1.processor",
            "mid_block.attn1.processor",
            "up_blocks.0.attn1.processor",
            "up_blocks.0.attn2.processor",
        ]
        self.assertEqual(
            select_branched_self_attention_names(names, "up_blocks_attn1"),
            ["up_blocks.0.attn1.processor"],
        )


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _attention()
        self.attn2 = Attention(
            query_dim=16,
            cross_attention_dim=16,
            heads=4,
            dim_head=4,
            residual_connection=True,
        )


class _TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([_TinyBlock()])
        self.mid_block = _TinyBlock()
        self.up_blocks = nn.ModuleList([_TinyBlock()])
        self.config = SimpleNamespace(
            block_out_channels=[16],
            cross_attention_dim=16,
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def attn_processors(self):
        return {
            "down_blocks.0.attn1.processor": self.down_blocks[0].attn1.processor,
            "down_blocks.0.attn2.processor": self.down_blocks[0].attn2.processor,
            "mid_block.attn1.processor": self.mid_block.attn1.processor,
            "mid_block.attn2.processor": self.mid_block.attn2.processor,
            "up_blocks.0.attn1.processor": self.up_blocks[0].attn1.processor,
            "up_blocks.0.attn2.processor": self.up_blocks[0].attn2.processor,
        }

    def set_attn_processor(self, processors):
        for name, processor in processors.items():
            module = self
            for part in name.rsplit(".processor", 1)[0].split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            module.set_processor(processor)


def _tiny_model() -> SimpleNamespace:
    model = SimpleNamespace(
        unet=_TinyUNet(),
        device=torch.device("cpu"),
        target_size=64,
        vae_scale_factor=8,
        face_embed_strategy="id_image",
        use_attn_v2=False,
        ba_correctness_guards=True,
        disable_branched_sa=False,
        disable_branched_ca=False,
        ba_processor_variant="packed_residual_v1",
        ba_site_policy="up_blocks_attn1",
        ba_patch_top_k=1.0,
        ba_train_top_k=1.0,
        branched_attn_weight_mode="ref_only",
        branched_attn_new_weight_kind="lora",
        branched_attn_lora_rank=4,
        ba_connector_rank=4,
        ba_gate_max=0.5,
        ba_gate_init_logit=0.0,
        ba_delta_rms_cap=0.25,
        ba_target_core_erode_frac=0.1,
        ba_diagnostics=False,
        train_ba_only=True,
        train_branched_ca_lora=False,
        non_ba_train=False,
        ba_sa_train_mode="packed_residual",
        ba_sa_face_mode="reference",
        ba_sa_ref_token_mode="full_grid",
        ba_sa_ref_layer_scope="all",
        ba_sa_roi_grid_size=8,
        ba_sa_core_ratio=0.7,
    )
    return model


class _OptimizerConfig(dict):
    def __getattr__(self, name):
        return self[name]


class PackedResidualRuntimeTests(unittest.TestCase):
    def test_training_installer_skips_plain_diffusers_processors(self) -> None:
        model = _tiny_model()
        install_branched_processors_for_training(model)
        plain_processors = [
            processor
            for processor in model.unet.attn_processors.values()
            if isinstance(processor, AttnProcessor2_0)
        ]
        self.assertEqual(len(plain_processors), 2)
        self.assertEqual(
            set(model._ba_patched_sa_names),
            {"up_blocks.0.attn1.processor"},
        )
        _assert_branched_installation(model)

    def test_processors_persist_and_trainability_manifest_is_exact(self) -> None:
        model = _tiny_model()
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(model, mask, mask)
        first_ids = {
            name: id(processor)
            for name, processor in model.unet.attn_processors.items()
            if getattr(processor, "_is_branched_processor", False)
        }
        patch_unet_attention_processors(model, mask, mask)
        second_ids = {
            name: id(processor)
            for name, processor in model.unet.attn_processors.items()
            if getattr(processor, "_is_branched_processor", False)
        }
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            set(model._ba_patched_sa_names),
            {"up_blocks.0.attn1.processor"},
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        groups = PhotomakerBranchedLora.get_trainable_params(
            SimpleNamespace(
                unet=model.unet,
                ba_processor_variant="packed_residual_v1",
            ),
            _OptimizerConfig(lr_for_lora=5e-5),
        )
        self.assertEqual(
            [group["name"] for group in groups],
            [
                "ba_ppr_ref_k",
                "ba_ppr_ref_v",
                "ba_ppr_connector_down",
                "ba_ppr_connector_up",
                "ba_ppr_gate",
            ],
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for name, parameter in model.unet.named_parameters()
                if ".attn2.processor." in name
            )
        )

    def test_legacy_variant_remains_available(self) -> None:
        model = _tiny_model()
        model.ba_processor_variant = "legacy"
        model.ba_site_policy = "all"
        model.ba_sa_train_mode = "ref_kv_only"
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(model, mask, mask)
        self.assertEqual(len(model._ba_patched_sa_names), 3)
        self.assertTrue(
            all(
                model.unet.attn_processors[name].__class__.__name__
                == "BranchedAttnProcessor"
                for name in model._ba_patched_sa_names
            )
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)

    def test_output_anchor_is_exact_at_zero_and_face_local_after_update(self) -> None:
        class PackedResidualBranchedAttnProcessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.connector_up = nn.Linear(1, 1, bias=False)
                nn.init.zeros_(self.connector_up.weight)

        class FakeScheduler:
            @staticmethod
            def add_noise(latents, noise, timesteps):
                del noise, timesteps
                return latents

            @staticmethod
            def scale_model_input(latents, timesteps):
                del timesteps
                return latents

        class FakeUNet:
            def __init__(self, packed):
                self._branched = {"packed": packed}
                self._original = {"plain": object()}
                self.attn_processors = dict(self._branched)

            def set_attn_processor(self, processors):
                # Match diffusers.UNet2DConditionModel, which consumes the
                # caller's mapping while installing processors.
                self.attn_processors = {
                    name: processors.pop(name)
                    for name in list(processors)
                }

            def __call__(self, sample, *args, **kwargs):
                del args, kwargs
                is_base = "plain" in self.attn_processors
                return (sample + (1.0 if is_base else 10.0),)

        packed = PackedResidualBranchedAttnProcessor()
        unet = FakeUNet(packed)
        pipeline = SimpleNamespace(
            unet=unet,
            scheduler=FakeScheduler(),
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
            do_classifier_free_guidance=False,
            face_embed_strategy="face",
            ba_output_anchor_mode="base_outside_core",
            ba_target_core_erode_frac=0.10,
            _original_attn_processors=dict(unet._original),
            _cross_attention_kwargs=None,
        )
        target = torch.zeros(1, 4, 8, 8)
        reference = torch.zeros_like(target)
        prompt = torch.zeros(1, 4, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 1:7, 1:7] = 1
        kwargs = {
            "pipeline": pipeline,
            "latent_model_input": target,
            "t": torch.tensor([1]),
            "prompt_embeds": prompt,
            "added_cond_kwargs": {},
            "mask4": mask,
            "mask4_ref": mask,
            "reference_latents": reference,
            "face_prompt_embeds": prompt,
        }

        with patch(
            "src.model.photomaker_branched.branched_runtime."
            "patch_unet_attention_processors"
        ), torch.no_grad():
            exact, _, _ = two_branch_predict(**kwargs)
        torch.testing.assert_close(exact, target + 1.0, atol=0, rtol=0)
        self.assertTrue(pipeline._ba_packed_branch_exactly_off)
        self.assertTrue(pipeline._ba_output_anchor_logged)
        self.assertIn("packed", unet.attn_processors)

        with torch.no_grad():
            packed.connector_up.weight.fill_(1.0)
        reset_branched_generation_caches(pipeline)
        self.assertFalse(hasattr(pipeline, "_ba_packed_branch_exactly_off"))
        self.assertFalse(hasattr(pipeline, "_ba_output_anchor_logged"))

        pipeline.ba_ppr_force_base_output = True
        pipeline.ba_output_anchor_mode = "none"
        pipeline.ba_ppr_collect_diagnostics = True
        pipeline.ba_ppr_diagnostic_steps = (0,)
        pipeline.ba_ppr_diagnostic_variant = "A"
        pipeline.ba_ppr_diagnostic_sample_keys = ("sample",)
        pipeline._ba_ppr_epsilon_diagnostics = []
        with patch(
            "src.model.photomaker_branched.branched_runtime."
            "patch_unet_attention_processors"
        ), torch.no_grad():
            forced_base, _, _ = two_branch_predict(**kwargs)
        torch.testing.assert_close(forced_base, target + 1.0, atol=0, rtol=0)
        self.assertEqual(
            pipeline._ba_ppr_epsilon_diagnostics[0]["output_control"],
            "diagnostic-force-base",
        )
        self.assertEqual(
            pipeline._ba_ppr_epsilon_diagnostics[0]["inside_core_post_anchor"],
            0.0,
        )

        pipeline.ba_ppr_force_base_output = False
        pipeline.ba_ppr_collect_diagnostics = False
        pipeline.ba_output_anchor_mode = "base_outside_core"
        reset_branched_generation_caches(pipeline)
        with patch(
            "src.model.photomaker_branched.branched_runtime."
            "patch_unet_attention_processors"
        ), torch.no_grad():
            localized, _, _ = two_branch_predict(**kwargs)
        core = make_inner_core_mask(mask, erode_frac=0.10)
        expected = (target + 1.0) + core * 9.0
        torch.testing.assert_close(localized, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(localized[:, :, 0, :], target[:, :, 0, :] + 1.0))
        self.assertIn("packed", unet.attn_processors)

    def test_diagnostic_swap_selection_spans_face_area_tertiles(self) -> None:
        class Dataset:
            def __len__(self):
                return 12

            def __getitem__(self, index):
                side = index + 1
                return {"face_bbox_gen": [0, 0, side, side]}

        selected = _select_spatial_swap_indices(Dataset(), 6)
        self.assertEqual(len(selected), 6)
        self.assertTrue(any(index < 4 for index in selected))
        self.assertTrue(any(4 <= index < 8 for index in selected))
        self.assertTrue(any(index >= 8 for index in selected))

    def test_diagnostic_pixel_mae_uses_normalized_face_core(self) -> None:
        baseline = Image.new("RGB", (16, 16), "black")
        changed = Image.new("RGB", (16, 16), "white")
        whole, face = _pixel_mae(changed, baseline, [2, 2, 14, 14])
        self.assertEqual(whole, 1.0)
        self.assertEqual(face, 1.0)
        exact_whole, exact_face = _pixel_mae(
            baseline,
            baseline,
            [2, 2, 14, 14],
        )
        self.assertEqual(exact_whole, 0.0)
        self.assertEqual(exact_face, 0.0)

    def test_e_only_reuses_a_to_d_without_eagerly_deleting_old_e(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint-epoch4.pth"
            checkpoint.touch()
            output = root / "ppr_8k_diagnostic"
            baseline_dir = output / "A_exact_pm"
            old_e_dir = output / "E_reference_swap"
            contact_dir = output / "contact_sheets"
            baseline_dir.mkdir(parents=True)
            old_e_dir.mkdir()
            contact_dir.mkdir()
            for index in range(2):
                Image.new("RGB", (8, 8), "black").save(
                    baseline_dir / f"{index}.png"
                )
            Image.new("RGB", (8, 8), "white").save(old_e_dir / "old.png")
            (contact_dir / "old.jpg").touch()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "validation_base": "SG161222/RealVisXL_V4.0",
                        "sample_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            with (output / "metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                for option in ("A", "E"):
                    writer.writerow(
                        {
                            field: option if field == "option" else ""
                            for field in METRIC_FIELDS
                        }
                    )
            with (output / "epsilon_diagnostics.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for option in ("A", "E"):
                    handle.write(json.dumps({"variant": option}) + "\n")

            images = []
            for identity in ("id0", "id1"):
                path = root / f"{identity}.png"
                Image.new("RGB", (8, 8), "gray").save(path)
                images.append(path)

            class Dataset:
                _bbox_map_ref = {"id0": [0, 0, 8, 8], "id1": [0, 0, 8, 8]}

                def __init__(self):
                    self.images = images

                def __len__(self):
                    return 2

                def __getitem__(self, index):
                    return {"face_bbox_gen": [0, 0, 4 + index, 4 + index]}

            config = SimpleNamespace(
                ppr_diagnostic_output_dir=str(output),
                ppr_diagnostic_overwrite=False,
                ppr_diagnostic_options=["E"],
                ppr_diagnostic_reuse_output=True,
                ppr_diagnostic_swap_count=2,
                saved_checkpoint=str(checkpoint),
                pretrained_model_for_validation_name_or_path=(
                    "SG161222/RealVisXL_V4.0"
                ),
            )
            trainer = SimpleNamespace(
                config=config,
                evaluation_dataloaders={
                    "manual_val": SimpleNamespace(dataset=Dataset())
                },
            )
            state = _initialize_state(trainer)
            self.assertEqual(_diagnostic_options(config), ("E",))
            self.assertEqual([row["option"] for row in state["rows"]], ["A"])
            self.assertEqual(
                [record["variant"] for record in state["epsilon"]],
                ["A"],
            )
            self.assertTrue((old_e_dir / "old.png").exists())
            self.assertTrue(old_e_dir.is_dir())
            self.assertTrue((contact_dir / "old.jpg").exists())
            self.assertEqual(state["swap_indices"], {0, 1})

    def test_scale_sweep_parses_scales_and_maps_cfg_processor_stats(self) -> None:
        config = SimpleNamespace(ppr_scale_sweep_scales=[0, 1, 2, 3, 4, 6])
        self.assertEqual(_parse_scales(config), (0.0, 1.0, 2.0, 3.0, 4.0, 6.0))
        self.assertEqual(_scale_label(2.5), "2p5")
        diagnostics = [
            {
                "record_type": "processor_applied_ratio",
                "processor": "up.0",
                "gate": 0.25,
                # CFG order: two unconditional samples, then two conditional.
                "applied_ratios": [0.1, 0.2, 0.3, 0.4],
                "cap_scales": [1.0, 0.5, 0.8, 1.0],
            },
            {
                "record_type": "processor_applied_ratio",
                "processor": "up.1",
                "gate": 0.35,
                "applied_ratios": [0.2, 0.4, 0.6, 0.8],
                "cap_scales": [1.0, 1.0, 0.9, 0.7],
            },
        ]
        stats = _processor_stats(diagnostics, batch_size=2)
        self.assertEqual(stats[0]["active_processor_count"], 2)
        self.assertAlmostEqual(stats[0]["mean_gate"], 0.30)
        self.assertAlmostEqual(stats[0]["applied_delta_rms_ratio"], 0.30)
        self.assertAlmostEqual(stats[0]["cap_fraction"], 0.50)
        self.assertAlmostEqual(stats[1]["applied_delta_rms_ratio"], 0.45)
        self.assertAlmostEqual(stats[1]["cap_fraction"], 0.50)


if __name__ == "__main__":
    unittest.main()
