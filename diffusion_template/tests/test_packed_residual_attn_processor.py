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
    apply_ppr_reference_ca_mode,
    patch_unet_attention_processors,
    select_branched_self_attention_names,
    two_branch_predict,
)
from src.model.photomaker_branched.lora2_helpers import (
    _assert_branched_installation,
    configure_branched_trainables,
    install_branched_processors_for_training,
)
from src.model.photomaker_branched.lora2 import (
    PhotomakerBranchedLora,
    attenuate_photomaker_identity,
)
from src.model.photomaker_branched.attn_processor_cleanest import (
    BranchLoRALinear,
    BranchedAttnProcessor,
)
from src.model.photomaker_branched.packed_residual_attn_processor import (
    PackedResidualBranchedAttnProcessor,
    make_inner_core_mask,
    pack_valid_tokens,
)
from src.loss.diffusion_loss import CoreNormalizedDiffusionLoss
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
from src.trainer.ppr_reference_noise import (
    _effective_reference_ca_mode,
    _noise_seeds,
    _reference_ca_mode,
    _relative_signature,
    run_ppr_reference_noise_batch,
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


def _processor(
    attn: Attention,
    channels: int = 16,
    connector_input_mode: str = "reference_minus_target",
) -> PackedResidualBranchedAttnProcessor:
    processor = PackedResidualBranchedAttnProcessor(
        channels,
        ref_kv_rank=4,
        connector_rank=4,
        connector_input_mode=connector_input_mode,
        gate_max=0.5,
        delta_rms_cap=0.25,
    )
    processor.init_from_attention(attn)
    return processor


class PackedResidualProcessorTests(unittest.TestCase):
    def test_nn7a_init_v2_uses_complete_sibling_attention_space(self) -> None:
        torch.manual_seed(37)
        channels = 16
        attn1 = _attention(channels)
        attn2 = Attention(
            query_dim=channels,
            cross_attention_dim=channels,
            heads=4,
            dim_head=4,
            residual_connection=True,
        )
        with torch.no_grad():
            attn1.to_q.weight.mul_(0.1)
            attn1.to_out[0].weight.mul_(0.2)
            attn2.to_q.weight.mul_(1.7)
            attn2.to_out[0].weight.mul_(1.9)
        processor = PackedResidualBranchedAttnProcessor(
            channels,
            ref_kv_rank=4,
            identity_fusion_mode="factorized_dual",
            enable_identity=False,
            enable_spatial=True,
            spatial_memory_mode="clean_clip_patches",
            spatial_patch_dim=channels,
            spatial_patch_projection="pmv2_perceiver_context",
            spatial_kv_init="sibling_attn2",
            spatial_kv_kind="lora",
            spatial_local_window=3,
            spatial_mix_mode="direct_candidate_takeover",
            spatial_attention_space="sibling_attn2_full",
            spatial_gate_position="pre_cap",
            spatial_gate_max=0.8,
            gate_init_logit=-1.9459101490553132,
            spatial_delta_rms_cap=1.0,
            total_delta_rms_cap=1.0,
        )
        processor.init_from_attention(attn1, sibling_attn2=attn2)
        self.assertFalse(any(p.requires_grad for p in processor.spatial_to_q.parameters()))
        self.assertFalse(any(p.requires_grad for p in processor.spatial_to_out.parameters()))

        target_hidden = torch.randn(1, 1, channels)
        reference_hidden = torch.randn_like(target_hidden)
        patches = torch.randn(1, 4, channels)
        processor.spatial_patch_tokens = patches
        mask = torch.ones(1, 1, 1, 1)
        processor.set_masks(mask, mask, mask)

        target_pre, attn1_query = processor._base_self_attention_pre_out(
            attn1,
            target_hidden,
            None,
        )
        target_post = attn1.to_out[1](attn1.to_out[0](target_pre))
        candidate, _, _ = processor._clean_local_spatial_candidate(
            attn1,
            attn1_query,
            target_post,
            mask.flatten(2).transpose(1, 2),
            mask.flatten(2).transpose(1, 2).bool(),
            target_hidden,
        )

        query = processor._to_heads(attn2.to_q(target_hidden), attn2.heads)
        key = processor._to_heads(attn2.to_k(patches), attn2.heads)
        value = processor._to_heads(attn2.to_v(patches), attn2.heads)
        local_indices = torch.tensor([[0, 0, 1, 0, 0, 1, 2, 2, 3]])
        local_key = key[0][:, local_indices, :]
        local_value = value[0][:, local_indices, :]
        scores = torch.einsum("hqd,hqwd->hqw", query[0], local_key)
        weights = (scores / (channels // attn2.heads) ** 0.5).softmax(dim=-1)
        expected_pre = torch.einsum("hqw,hqwd->hqd", weights, local_value)
        expected_pre = processor._from_heads(expected_pre.unsqueeze(0))
        expected_candidate = attn2.to_out[0](expected_pre)
        torch.testing.assert_close(candidate, expected_candidate)

        output = processor(
            attn1,
            torch.cat([target_hidden, reference_hidden], dim=0),
        )[:1]
        alpha = 0.8 * torch.sigmoid(torch.tensor(-1.9459101490553132))
        scaled = alpha * (expected_candidate - target_post)
        bounded, _, _, _ = processor._masked_rms_cap(
            scaled,
            base=target_post,
            mask=torch.ones_like(mask.flatten(2).transpose(1, 2)),
            max_ratio=1.0,
        )
        expected = target_post + bounded + target_hidden
        torch.testing.assert_close(output, expected)

        output.square().mean().backward()
        for parameter in (
            processor.ref_to_k.lora_B,
            processor.ref_to_v.lora_B,
            processor.gate_logit,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_nn7a_init_v2_gate_before_cap_has_interpretable_authority(self) -> None:
        base = torch.ones(1, 4, 8)
        mask = torch.ones(1, 4, 1)
        alpha = 0.10
        small_raw = torch.full_like(base, 0.5)
        small, scale, _, ratio = PackedResidualBranchedAttnProcessor._masked_rms_cap(
            alpha * small_raw,
            base=base,
            mask=mask,
            max_ratio=0.20,
        )
        torch.testing.assert_close(small, alpha * small_raw)
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(ratio, torch.full_like(ratio, 0.05))

        large_raw = torch.full_like(base, 10.0)
        large, scale, _, ratio = PackedResidualBranchedAttnProcessor._masked_rms_cap(
            alpha * large_raw,
            base=base,
            mask=mask,
            max_ratio=0.20,
        )
        torch.testing.assert_close(large, torch.full_like(large, 0.20))
        torch.testing.assert_close(scale, torch.full_like(scale, 0.20))
        torch.testing.assert_close(ratio, torch.full_like(ratio, 0.20))

    def test_nn7a_init_v2_reference_sensitivity_is_core_local(self) -> None:
        torch.manual_seed(41)
        channels, side = 16, 4
        attn1 = _attention(channels)
        attn2 = Attention(
            query_dim=channels,
            cross_attention_dim=channels,
            heads=4,
            dim_head=4,
            residual_connection=True,
        )
        processor = PackedResidualBranchedAttnProcessor(
            channels,
            ref_kv_rank=4,
            identity_fusion_mode="factorized_dual",
            enable_identity=False,
            enable_spatial=True,
            spatial_memory_mode="clean_clip_patches",
            spatial_patch_dim=channels,
            spatial_patch_projection="pmv2_perceiver_context",
            spatial_kv_init="sibling_attn2",
            spatial_kv_kind="lora",
            spatial_local_window=3,
            spatial_mix_mode="direct_candidate_takeover",
            spatial_attention_space="sibling_attn2_full",
            spatial_gate_position="pre_cap",
            spatial_gate_max=0.8,
            gate_init_logit=-1.9459101490553132,
            spatial_delta_rms_cap=0.20,
            total_delta_rms_cap=0.20,
        )
        processor.init_from_attention(attn1, sibling_attn2=attn2)
        face = torch.ones(1, 1, side, side)
        core = torch.zeros_like(face)
        core[:, :, 1:3, 1:3] = 1
        processor.set_masks(face, face, core)
        hidden = torch.randn(2, side * side, channels)

        processor.spatial_patch_tokens = torch.randn(1, 16, channels)
        processor.runtime_scale = 0.0
        processor(attn1, hidden)
        self.assertFalse(processor._warm_runtime_checked)

        processor.runtime_scale = 1.0
        first = processor(attn1, hidden)[:1]
        self.assertTrue(processor._warm_runtime_checked)
        processor.spatial_patch_tokens = torch.randn(1, 16, channels)
        second = processor(attn1, hidden)[:1]
        outside = (core.flatten(2).transpose(1, 2) == 0).expand_as(first)
        inside = ~outside
        self.assertTrue(torch.equal(first[outside], second[outside]))
        difference = first[inside] - second[inside]
        self.assertGreater(float(difference.square().mean().sqrt()), 1e-4)

    def test_nn7a_init_warm_kv_is_active_local_and_trainable(self) -> None:
        torch.manual_seed(31)
        side = 8
        attn1 = _attention()
        attn2 = Attention(
            query_dim=16,
            cross_attention_dim=16,
            heads=4,
            dim_head=4,
            residual_connection=True,
        )
        processor = PackedResidualBranchedAttnProcessor(
            16,
            ref_kv_rank=4,
            identity_fusion_mode="factorized_dual",
            enable_identity=False,
            enable_spatial=True,
            spatial_memory_mode="clean_clip_patches",
            spatial_patch_dim=16,
            spatial_patch_projection="pmv2_perceiver_context",
            spatial_kv_init="sibling_attn2",
            spatial_kv_kind="lora",
            spatial_local_window=3,
            spatial_mix_mode="direct_candidate_takeover",
            spatial_gate_max=0.8,
            gate_init_logit=-2.70805020110221,
            spatial_delta_rms_cap=0.45,
            total_delta_rms_cap=0.45,
        )
        processor.init_from_attention(attn1, sibling_attn2=attn2)
        self.assertIsInstance(processor.ref_to_k, BranchLoRALinear)
        self.assertIsInstance(processor.ref_to_v, BranchLoRALinear)
        self.assertEqual(int(torch.count_nonzero(processor.ref_to_k.lora_B)), 0)
        self.assertEqual(int(torch.count_nonzero(processor.ref_to_v.lora_B)), 0)
        self.assertNotIn("ref_to_k.base_weight", processor.state_dict())
        self.assertAlmostEqual(
            0.8 * torch.sigmoid(torch.tensor(-2.70805020110221)).item(),
            0.05,
            places=6,
        )

        tokens = torch.randn(1, 16, 16)
        torch.testing.assert_close(processor.ref_to_k(tokens), attn2.to_k(tokens))
        torch.testing.assert_close(processor.ref_to_v(tokens), attn2.to_v(tokens))

        mask, core = _masks(1, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(2, side * side, 16)
        processor.spatial_patch_tokens = torch.randn(1, 16, 16)
        first = processor(attn1, hidden)[:1]
        processor.spatial_patch_tokens = torch.randn(1, 16, 16)
        second = processor(attn1, hidden)[:1]
        outside = (core.flatten(2).transpose(1, 2) == 0).expand_as(first)
        inside = ~outside
        self.assertTrue(torch.equal(first[outside], second[outside]))
        self.assertGreater(float((first[inside] - second[inside]).abs().max()), 0.0)

        second.square().mean().backward()
        for parameter in (
            processor.ref_to_k.lora_B,
            processor.ref_to_v.lora_B,
            processor.gate_logit,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        processor.zero_grad(set_to_none=True)
        with torch.no_grad():
            processor.ref_to_k.lora_B.add_(0.01)
            processor.ref_to_v.lora_B.add_(0.01)
        processor(attn1, hidden)[:1].square().mean().backward()
        self.assertGreater(float(processor.ref_to_k.lora_A.grad.abs().sum()), 0.0)
        self.assertGreater(float(processor.ref_to_v.lora_A.grad.abs().sum()), 0.0)

    def test_nn7_clean_local_takeover_is_face_local_and_trainable(self) -> None:
        torch.manual_seed(27)
        side = 8
        attn = _attention()
        processor = PackedResidualBranchedAttnProcessor(
            16,
            identity_fusion_mode="factorized_dual",
            enable_identity=False,
            enable_spatial=True,
            spatial_memory_mode="clean_clip_patches",
            spatial_patch_dim=8,
            spatial_local_window=3,
            spatial_mix_mode="direct_candidate_takeover",
            spatial_gate_max=0.8,
            spatial_delta_rms_cap=0.45,
            total_delta_rms_cap=0.45,
        )
        processor.init_from_attention(attn)
        mask, core = _masks(1, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(2, side * side, 16)

        processor.spatial_patch_tokens = torch.randn(1, 16, 8)
        first = processor(attn, hidden)[:1]
        processor.spatial_patch_tokens = torch.randn(1, 16, 8)
        second = processor(attn, hidden)[:1]
        outside = (core.flatten(2).transpose(1, 2) == 0).expand_as(first)
        inside = ~outside
        self.assertTrue(torch.equal(first[outside], second[outside]))
        self.assertGreater(float((first[inside] - second[inside]).abs().max()), 0.0)

        second.square().mean().backward()
        self.assertIsNotNone(processor.ref_to_k.weight.grad)
        self.assertGreater(float(processor.ref_to_k.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(processor.gate_logit.grad)
        self.assertIsNone(processor.connector_down)

    def test_match_null_margin_uses_main_difference_pre_ratio(self) -> None:
        torch.manual_seed(13)
        side = 8
        attn = _attention()
        processor = PackedResidualBranchedAttnProcessor(
            16,
            ref_kv_rank=4,
            connector_rank=4,
            connector_input_mode="reference_minus_learned_null",
            collect_aux_losses=True,
            match_null_margin=0.02,
        )
        processor.init_from_attention(attn)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)

        # The first cap call is D(C_ref-C_null); the second is D(C_null).
        # The margin must use the first pre-ratio without a third cap call.
        ratios = iter((0.01, 0.50))

        def fake_cap(delta, *, base, mask, max_ratio):
            del base, mask, max_ratio
            ratio = delta.new_full((delta.shape[0],), next(ratios))
            return delta, torch.ones_like(ratio), ratio, ratio

        processor._masked_rms_cap = fake_cap
        processor(attn, torch.randn(4, side * side, 16))

        self.assertAlmostEqual(
            float(processor.last_aux_losses["match_null_margin"]),
            (0.02 - 0.01) ** 2,
            places=7,
        )

    def test_auxiliary_losses_exclude_rows_without_target_core(self) -> None:
        torch.manual_seed(14)
        side = 8
        attn = _attention()
        processor = PackedResidualBranchedAttnProcessor(
            16,
            ref_kv_rank=4,
            connector_rank=4,
            connector_input_mode="reference_minus_learned_null",
            collect_aux_losses=True,
            match_null_margin=0.02,
        )
        processor.init_from_attention(attn)
        mask, core = _masks(2, side)
        core[0].zero_()
        processor.set_masks(mask, mask, core)

        ratios = iter(
            (
                torch.tensor([0.0, 0.01]),
                torch.tensor([0.5, 0.5]),
            )
        )

        def fake_cap(delta, *, base, mask, max_ratio):
            del base, mask, max_ratio
            ratio = next(ratios).to(delta)
            return delta, torch.ones_like(ratio), ratio, ratio

        processor._masked_rms_cap = fake_cap
        processor(attn, torch.randn(4, side * side, 16))
        self.assertAlmostEqual(
            float(processor.last_aux_losses["match_null_margin"]),
            (0.02 - 0.01) ** 2,
            places=7,
        )

    def test_learned_null_uses_same_kv_route_and_receives_gradients(self) -> None:
        torch.manual_seed(11)
        side = 8
        attn = _attention()
        processor = _processor(
            attn,
            connector_input_mode="reference_minus_learned_null",
        )
        with torch.no_grad():
            processor.connector_up.weight.normal_()
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(4, side * side, 16)
        processor(attn, hidden)[:2].square().mean().backward()
        self.assertIsNotNone(processor.null_memory)
        self.assertIsNotNone(processor.null_memory.grad)
        self.assertGreater(processor.null_memory.grad.abs().sum(), 0)

    def test_learned_null_auxiliary_losses_are_finite_and_differentiable(self) -> None:
        torch.manual_seed(12)
        side = 8
        attn = _attention()
        processor = PackedResidualBranchedAttnProcessor(
            16,
            ref_kv_rank=4,
            connector_rank=4,
            connector_input_mode="reference_minus_learned_null",
            collect_aux_losses=True,
            match_null_margin=0.02,
            cap_loss_target=1e-8,
        )
        processor.init_from_attention(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.05)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        hidden = torch.randn(4, side * side, 16)
        processor(attn, hidden)

        self.assertEqual(
            set(processor.last_aux_losses),
            {"null_residual", "match_null_margin", "cap"},
        )
        loss = torch.stack(
            list(processor.last_aux_losses.values())
        ).sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(processor.connector_up.weight.grad)
        self.assertGreater(processor.connector_up.weight.grad.abs().sum(), 0)

    def test_photomaker_attenuation_preserves_half_the_batch(self) -> None:
        base = torch.randn(2, 5, 4)
        fused = base + 3.0
        output, mask = attenuate_photomaker_identity(
            fused,
            base,
            probability=0.5,
            scale=0.0,
        )
        self.assertEqual(int(mask.sum().item()), 1)
        torch.testing.assert_close(output[mask], base[mask])
        torch.testing.assert_close(output[~mask], fused[~mask])

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

    def test_reference_minus_null_removes_target_base_shortcut(self) -> None:
        torch.manual_seed(31)
        side = 8
        channels = 16
        attn = _attention(channels)
        legacy = _processor(attn, channels, "reference_minus_target")
        contrastive = _processor(attn, channels, "reference_minus_null")
        contrastive.load_state_dict(legacy.state_dict())
        with torch.no_grad():
            legacy.connector_up.weight.normal_(std=0.1)
            contrastive.connector_up.weight.copy_(legacy.connector_up.weight)
            # Remove all reference values. The NN3 connector must then receive
            # an exact zero input; NN2 can still act through -target_base.
            for processor in (legacy, contrastive):
                processor.ref_to_v.base_weight.zero_()
                if processor.ref_to_v.base_bias is not None:
                    processor.ref_to_v.base_bias.zero_()
                for parameter in processor.ref_to_v.parameters():
                    parameter.zero_()

        mask, core = _masks(2, side)
        hidden = torch.randn(4, side * side, channels)
        base = AttnProcessor2_0()(attn, hidden)
        legacy.set_masks(mask, mask, core)
        contrastive.set_masks(mask, mask, core)
        legacy_output = legacy(attn, hidden)
        contrastive_output = contrastive(attn, hidden)

        torch.testing.assert_close(
            contrastive_output[:2],
            base[:2],
            atol=0,
            rtol=0,
        )
        self.assertFalse(torch.equal(legacy_output[:2], base[:2]))
        torch.testing.assert_close(
            contrastive_output[2:],
            base[2:],
            atol=0,
            rtol=0,
        )

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

    def test_tensor_diagnostic_signature_records_all_processor_stages(self) -> None:
        torch.manual_seed(52)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.05)
        mask, core = _masks(2, side)
        processor.set_masks(mask, mask, core)
        processor.tensor_diagnostics = True
        processor.diagnostic_step = 15
        processor.diagnostic_steps = (15,)
        processor.diagnostic_variant = "R1N1"
        processor.diagnostic_sink = []
        processor(attn, torch.randn(4, side * side, 16))
        records = [
            value
            for value in processor.diagnostic_sink
            if value["record_type"] == "processor_tensor_signature"
        ]
        self.assertEqual(len(records), 1)
        self.assertGreater(records[0]["roi_tokens"], 0)
        for stage in (
            "reference_hidden",
            "reference_candidate",
            "connector_input",
            "connector_down",
            "raw_delta",
            "bounded_delta",
            "applied_delta",
        ):
            self.assertEqual(len(records[0][stage]["sha256"]), 64)
            self.assertGreater(len(records[0][stage]["sketch"]), 0)

    def test_tensor_diagnostics_map_batched_cfg_rows_to_samples(self) -> None:
        torch.manual_seed(53)
        side = 8
        attn = _attention()
        processor = _processor(attn)
        with torch.no_grad():
            processor.connector_up.weight.normal_(std=0.05)
        mask, core = _masks(4, side)
        processor.set_masks(mask, mask, core)
        processor.tensor_diagnostics = True
        processor.diagnostic_step = 15
        processor.diagnostic_steps = (15,)
        processor.diagnostic_variant = "R1N1"
        processor.diagnostic_sample_keys = ("sample0", "sample1")
        processor.diagnostic_do_cfg = True
        processor.diagnostic_sink = []
        processor(attn, torch.randn(8, side * side, 16))
        records = [
            value
            for value in processor.diagnostic_sink
            if value["record_type"] == "processor_tensor_signature"
        ]
        self.assertEqual(
            [record["sample"] for record in records],
            ["sample0", "sample1"],
        )
        self.assertNotEqual(
            records[0]["reference_hidden"]["sha256"],
            records[1]["reference_hidden"]["sha256"],
        )

    def test_reference_noise_helpers_require_two_distinct_seeds(self) -> None:
        config = SimpleNamespace(ppr_reference_noise_seeds=[11, 22])
        self.assertEqual(_noise_seeds(config), {"N1": 11, "N2": 22})
        with self.assertRaises(ValueError):
            _noise_seeds(
                SimpleNamespace(ppr_reference_noise_seeds=[11, 11])
            )
        left = {"sketch": [1.0, 2.0]}
        same = {"sketch": [1.0, 2.0]}
        doubled = {"sketch": [2.0, 4.0]}
        self.assertEqual(_relative_signature(left, same), 0.0)
        self.assertAlmostEqual(_relative_signature(left, doubled), 1.0)

    def test_neutral_reference_ca_is_explicit_and_diagnostic_only(self) -> None:
        prompt = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        pipeline = SimpleNamespace(
            ba_ppr_reference_ca_mode="original",
            ba_ppr_collect_diagnostics=False,
        )
        unchanged, unchanged_mask, mode = apply_ppr_reference_ca_mode(
            pipeline, prompt, mask
        )
        self.assertIs(unchanged, prompt)
        self.assertIs(unchanged_mask, mask)
        self.assertEqual(mode, "original")

        pipeline.ba_ppr_reference_ca_mode = "zero"
        with self.assertRaises(RuntimeError):
            apply_ppr_reference_ca_mode(pipeline, prompt, mask)
        pipeline.ba_ppr_collect_diagnostics = True
        pipeline.ppr_reference_ca_mode = "zero"
        neutral, neutral_mask, mode = apply_ppr_reference_ca_mode(
            pipeline, prompt, mask
        )
        self.assertEqual(torch.count_nonzero(neutral), 0)
        self.assertIsNone(neutral_mask)
        self.assertEqual(mode, "zero")
        self.assertEqual(_reference_ca_mode(pipeline), "zero")
        nn4_config = SimpleNamespace(
            ppr_reference_ca_mode="original",
            model=SimpleNamespace(ba_reference_token_text_mode="zero"),
        )
        self.assertEqual(_effective_reference_ca_mode(nn4_config), "zero")

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
            "up_blocks.1.attn1.processor",
            "up_blocks.0.attn2.processor",
        ]
        self.assertEqual(
            select_branched_self_attention_names(names, "up_blocks_attn1"),
            ["up_blocks.0.attn1.processor", "up_blocks.1.attn1.processor"],
        )
        self.assertEqual(
            select_branched_self_attention_names(
                names,
                "up_blocks0_attn1",
            ),
            ["up_blocks.0.attn1.processor"],
        )
        self.assertEqual(
            select_branched_self_attention_names(
                names,
                "up_blocks1_attn1",
            ),
            ["up_blocks.1.attn1.processor"],
        )

    def test_core_normalized_diffusion_loss_ignores_outside_core(self) -> None:
        target = torch.zeros(2, 4, 4, 4)
        prediction = target.clone()
        prediction[:, :, 0, :] = 100
        prediction[0, :, 1:3, 1:3] = 2
        prediction[1, :, 1:3, 1:3] = 4
        core = torch.zeros(2, 1, 4, 4)
        core[:, :, 1:3, 1:3] = 1
        loss = CoreNormalizedDiffusionLoss()(
            prediction,
            target,
            core,
        )["loss"]
        self.assertEqual(float(loss), 10.0)

    def test_core_normalized_diffusion_loss_rejects_empty_rows(self) -> None:
        target = torch.zeros(2, 4, 4, 4)
        core = torch.ones(2, 1, 4, 4)
        core[1].zero_()
        with self.assertRaisesRegex(ValueError, r"rows \[1\]"):
            CoreNormalizedDiffusionLoss()(target, target, core)


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
    def test_n3a_new2_dual_mix_and_up_scope_are_initialized_exactly(self) -> None:
        attn = _attention()
        up_processor = BranchedAttnProcessor(
            hidden_size=16,
            branched_attn_weight_mode="noise_and_ref",
            branched_attn_new_weight_kind="lora",
            branched_attn_lora_rank=4,
            processor_name="up_blocks.0.attn1.processor",
            ba_sa_face_mode="dual",
            ba_sa_mix_init=0.35,
            ba_sa_ref_layer_scope="up",
        )
        up_processor.init_from_attention(attn)
        torch.testing.assert_close(
            up_processor.face_mix_logits.sigmoid(),
            torch.full((attn.heads,), 0.35),
        )
        self.assertTrue(up_processor._reference_enabled_here())

        down_processor = BranchedAttnProcessor(
            hidden_size=16,
            processor_name="down_blocks.0.attn1.processor",
            ba_sa_face_mode="core_ring",
            ba_sa_core_ratio=0.68,
            ba_sa_ref_layer_scope="up",
        )
        self.assertFalse(down_processor._reference_enabled_here())
        self.assertEqual(down_processor.ba_sa_core_ratio, 0.68)

    def test_legacy_anchor_is_recorded_only_when_enabled(self) -> None:
        model = SimpleNamespace(
            ba_sa_ref_token_mode="full_grid",
            ba_sa_face_mode="core_ring",
            ba_sa_ref_layer_scope="up",
            ba_sa_roi_grid_size=8,
            ba_sa_core_ratio=0.68,
            ba_sa_mix_init=0.35,
            ba_processor_variant="legacy",
            ba_target_core_erode_frac=0.10,
            ba_output_anchor_mode="base_outside_core",
        )
        architecture = PhotomakerBranchedLora._ba_architecture_state(model)
        self.assertEqual(
            architecture["ba_output_anchor_mode"],
            "base_outside_core",
        )
        self.assertEqual(architecture["ba_target_core_erode_frac"], 0.10)

        model.ba_output_anchor_mode = "none"
        architecture = PhotomakerBranchedLora._ba_architecture_state(model)
        self.assertNotIn("ba_output_anchor_mode", architecture)
        self.assertNotIn("ba_target_core_erode_frac", architecture)

    def test_direct_spatial_checkpoint_diagnostics_track_lora_b(self) -> None:
        attn1 = _attention()
        attn2 = Attention(
            query_dim=16,
            cross_attention_dim=16,
            heads=4,
            dim_head=4,
        )
        processor = PackedResidualBranchedAttnProcessor(
            16,
            ref_kv_rank=4,
            identity_fusion_mode="factorized_dual",
            enable_identity=False,
            enable_spatial=True,
            spatial_memory_mode="clean_clip_patches",
            spatial_patch_dim=16,
            spatial_kv_init="sibling_attn2",
            spatial_kv_kind="lora",
            spatial_mix_mode="direct_candidate_takeover",
            spatial_attention_space="sibling_attn2_full",
            spatial_gate_position="pre_cap",
            spatial_gate_max=0.8,
        )
        processor.init_from_attention(attn1, sibling_attn2=attn2)
        k_delta = torch.full_like(processor.ref_to_k.lora_B, 0.01)
        v_delta = torch.full_like(processor.ref_to_v.lora_B, -0.02)
        model = SimpleNamespace(
            unet=SimpleNamespace(attn_processors={"site": processor}),
            ba_strict_processor_restore=False,
            ba_identity_fusion_mode="factorized_dual",
            ba_gate_max=0.5,
            ba_spatial_gate_max=0.8,
            ba_identity_gate_max=0.5,
        )
        checkpoint = {
            "lora_weights": {},
            "attn_processors": {
                "site": {
                    "ref_to_k.lora_B": k_delta,
                    "ref_to_v.lora_B": v_delta,
                    "gate_logit": torch.tensor(-1.9459101490553132),
                }
            },
        }
        with patch(
            "src.model.photomaker_branched.lora2.convert_unet_state_dict_to_peft",
            return_value={},
        ), patch(
            "src.model.photomaker_branched.lora2.set_peft_model_state_dict",
            return_value=None,
        ):
            PhotomakerBranchedLora.load_state_dict_(model, checkpoint)
        diagnostics = model._last_ppr_checkpoint_diagnostics
        self.assertEqual(diagnostics["connector_up_tensors"], 0)
        self.assertEqual(diagnostics["direct_spatial_tensors"], 2)
        self.assertGreater(diagnostics["direct_spatial_nonzero"], 0)
        self.assertGreater(diagnostics["direct_spatial_l2"], 0.0)
        self.assertAlmostEqual(diagnostics["gate_min"], 0.1, places=6)

    def test_nn7a_init_checkpoint_architecture_is_strictly_separated(self) -> None:
        nn7a_architecture = {
            "ba_connector_input_mode": "reference_minus_target",
            "ba_spatial_memory_mode": "clean_clip_patches",
            "ba_spatial_patch_dim": 1024,
            "ba_spatial_patch_projection": "raw_clip",
            "ba_spatial_kv_init": "xavier",
            "ba_spatial_kv_kind": "full",
            "ba_spatial_attention_space": "attn1_hybrid",
            "ba_spatial_gate_position": "post_cap",
        }
        nn7a_init_architecture = {
            **nn7a_architecture,
            "ba_spatial_patch_dim": 2048,
            "ba_spatial_patch_projection": "pmv2_perceiver_context",
            "ba_spatial_kv_init": "sibling_attn2",
            "ba_spatial_kv_kind": "lora",
        }
        nn7a_init_v2_architecture = {
            **nn7a_init_architecture,
            "ba_spatial_attention_space": "sibling_attn2_full",
            "ba_spatial_gate_position": "pre_cap",
        }

        def restore(current_architecture, saved_architecture) -> None:
            model = SimpleNamespace(
                unet=SimpleNamespace(attn_processors={}),
                ba_strict_processor_restore=True,
                _ba_patched_processor_names=(),
                _ba_architecture_state=lambda: dict(current_architecture),
                ba_gate_max=0.8,
                ba_identity_gate_max=0.5,
            )
            checkpoint = {
                "lora_weights": {},
                "ba_processor_manifest": {
                    "installed_processor_names": [],
                    "state_processor_names": [],
                    "trainable_keys_by_processor": {},
                    "processor_classes": {},
                    "architecture": dict(saved_architecture),
                },
            }
            with patch(
                "src.model.photomaker_branched.lora2."
                "convert_unet_state_dict_to_peft",
                return_value={},
            ), patch(
                "src.model.photomaker_branched.lora2."
                "set_peft_model_state_dict",
                return_value=None,
            ):
                PhotomakerBranchedLora.load_state_dict_(model, checkpoint)

        restore(nn7a_architecture, nn7a_architecture)
        restore(nn7a_init_architecture, nn7a_init_architecture)
        restore(nn7a_init_v2_architecture, nn7a_init_v2_architecture)
        legacy_nn7a_architecture = dict(nn7a_architecture)
        legacy_nn7a_architecture.pop("ba_spatial_patch_projection")
        legacy_nn7a_architecture.pop("ba_spatial_kv_init")
        legacy_nn7a_architecture.pop("ba_spatial_kv_kind")
        legacy_nn7a_architecture.pop("ba_spatial_attention_space")
        legacy_nn7a_architecture.pop("ba_spatial_gate_position")
        restore(nn7a_architecture, legacy_nn7a_architecture)
        with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
            restore(nn7a_init_architecture, nn7a_architecture)
        with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
            restore(nn7a_architecture, nn7a_init_architecture)
        with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
            restore(nn7a_init_v2_architecture, nn7a_init_architecture)
        with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
            restore(nn7a_init_architecture, nn7a_init_v2_architecture)

    def test_nn7a_init_warm_lora_trainability_is_exact(self) -> None:
        model = _tiny_model()
        model.ba_site_policy = "up_blocks0_attn1"
        model.disable_branched_ca = True
        model.ba_identity_fusion_mode = "factorized_dual"
        model.ba_identity_token_lane = False
        model.ba_spatial_lane_enabled = True
        model.ba_identity_site_policy = "up_blocks0_attn1"
        model.ba_spatial_site_policy = "up_blocks0_attn1"
        model.ba_spatial_memory_mode = "clean_clip_patches"
        model.ba_spatial_patch_dim = 16
        model.ba_spatial_patch_projection = "pmv2_perceiver_context"
        model.ba_spatial_kv_init = "sibling_attn2"
        model.ba_spatial_kv_kind = "lora"
        model.ba_spatial_local_window = 3
        model.ba_spatial_mix_mode = "direct_candidate_takeover"
        model.ba_spatial_gate_max = 0.8
        model.ba_gate_init_logit = -2.70805020110221
        model.ba_spatial_delta_rms_cap = 0.45
        model.ba_total_delta_rms_cap = 0.45
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(
            model,
            mask,
            mask,
            spatial_patch_tokens=torch.ones(1, 16, 16),
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        processor = model.unet.attn_processors["up_blocks.0.attn1.processor"]
        self.assertEqual(
            {name for name, parameter in processor.named_parameters() if parameter.requires_grad},
            {
                "ref_to_k.lora_A",
                "ref_to_k.lora_B",
                "ref_to_v.lora_A",
                "ref_to_v.lora_B",
                "gate_logit",
            },
        )
        self.assertEqual(processor.spatial_attention_space, "attn1_hybrid")
        self.assertEqual(processor.spatial_gate_position, "post_cap")

    def test_nn7a_init_v2_trainability_excludes_warm_q_and_out(self) -> None:
        model = _tiny_model()
        model.ba_site_policy = "up_blocks0_attn1"
        model.disable_branched_ca = True
        model.ba_identity_fusion_mode = "factorized_dual"
        model.ba_identity_token_lane = False
        model.ba_spatial_lane_enabled = True
        model.ba_identity_site_policy = "up_blocks0_attn1"
        model.ba_spatial_site_policy = "up_blocks0_attn1"
        model.ba_spatial_memory_mode = "clean_clip_patches"
        model.ba_spatial_patch_dim = 16
        model.ba_spatial_patch_projection = "pmv2_perceiver_context"
        model.ba_spatial_kv_init = "sibling_attn2"
        model.ba_spatial_kv_kind = "lora"
        model.ba_spatial_local_window = 3
        model.ba_spatial_mix_mode = "direct_candidate_takeover"
        model.ba_spatial_attention_space = "sibling_attn2_full"
        model.ba_spatial_gate_position = "pre_cap"
        model.ba_spatial_gate_max = 0.8
        model.ba_gate_init_logit = -1.9459101490553132
        model.ba_spatial_delta_rms_cap = 0.20
        model.ba_total_delta_rms_cap = 0.20
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(
            model,
            mask,
            mask,
            spatial_patch_tokens=torch.ones(1, 16, 16),
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        processor = model.unet.attn_processors["up_blocks.0.attn1.processor"]
        self.assertEqual(
            {name for name, parameter in processor.named_parameters() if parameter.requires_grad},
            {
                "ref_to_k.lora_A",
                "ref_to_k.lora_B",
                "ref_to_v.lora_A",
                "ref_to_v.lora_B",
                "gate_logit",
            },
        )
        self.assertFalse(any(p.requires_grad for p in processor.spatial_to_q.parameters()))
        self.assertFalse(any(p.requires_grad for p in processor.spatial_to_out.parameters()))

    def test_nn7_direct_clean_spatial_trainability_is_exact(self) -> None:
        model = _tiny_model()
        model.ba_site_policy = "up_blocks0_attn1"
        model.disable_branched_ca = True
        model.ba_identity_fusion_mode = "factorized_dual"
        model.ba_identity_token_lane = False
        model.ba_spatial_lane_enabled = True
        model.ba_identity_site_policy = "up_blocks0_attn1"
        model.ba_spatial_site_policy = "up_blocks0_attn1"
        model.ba_spatial_memory_mode = "clean_clip_patches"
        model.ba_spatial_patch_dim = 8
        model.ba_spatial_local_window = 3
        model.ba_spatial_mix_mode = "direct_candidate_takeover"
        model.ba_spatial_gate_max = 0.8
        model.ba_spatial_delta_rms_cap = 0.45
        model.ba_total_delta_rms_cap = 0.45
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(
            model,
            mask,
            mask,
            spatial_patch_tokens=torch.ones(1, 16, 8),
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        processor = model.unet.attn_processors["up_blocks.0.attn1.processor"]
        self.assertEqual(
            {name for name, parameter in processor.named_parameters() if parameter.requires_grad},
            {"ref_to_k.weight", "ref_to_v.weight", "gate_logit"},
        )
        groups = PhotomakerBranchedLora.get_trainable_params(
            SimpleNamespace(
                unet=model.unet,
                ba_processor_variant="packed_residual_v1",
                ba_identity_fusion_mode="factorized_dual",
                ba_spatial_lane_enabled=True,
                ba_spatial_mix_mode="direct_candidate_takeover",
                ba_connector_input_mode="reference_minus_target",
                ba_identity_token_lane=False,
            ),
            _OptimizerConfig(lr_for_lora=5e-5),
        )
        self.assertEqual(
            [group["name"] for group in groups],
            ["ba_ppr_ref_k", "ba_ppr_ref_v", "ba_ppr_gate"],
        )

    def test_nn4_site_policy_and_disabled_ca_install_only_up0_sa(self) -> None:
        model = _tiny_model()
        model.ba_site_policy = "up_blocks0_attn1"
        model.disable_branched_ca = True
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(model, mask, mask)
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        self.assertEqual(
            tuple(model._ba_patched_processor_names),
            ("up_blocks.0.attn1.processor",),
        )
        self.assertTrue(
            all(
                not bool(getattr(processor, "_is_branched_processor", False))
                for name, processor in model.unet.attn_processors.items()
                if name.endswith("attn2.processor")
            )
        )

    def test_cfg_reference_noise_and_reference_text_are_isolated(self) -> None:
        class FakeScheduler:
            @staticmethod
            def add_noise(latents, noise, timesteps):
                del timesteps
                return latents + noise

            @staticmethod
            def scale_model_input(latents, timesteps):
                del timesteps
                return latents

        class CapturingUNet:
            def __init__(self):
                self.attn_processors = {}
                self.last_sample = None
                self.samples = []
                self.last_encoder = None
                self.encoders = []
                self.last_added = None
                self.addeds = []

            def set_attn_processor(self, processors):
                self.attn_processors = processors

            def __call__(
                self,
                sample,
                timestep,
                *,
                encoder_hidden_states,
                added_cond_kwargs,
                **kwargs,
            ):
                del timestep, kwargs
                self.last_sample = sample
                self.samples.append(sample)
                self.last_encoder = encoder_hidden_states
                self.encoders.append(encoder_hidden_states)
                self.last_added = added_cond_kwargs
                self.addeds.append(added_cond_kwargs)
                return (sample,)

        unet = CapturingUNet()
        pipeline = SimpleNamespace(
            unet=unet,
            scheduler=FakeScheduler(),
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(123),
            do_classifier_free_guidance=True,
            face_embed_strategy="face",
            ba_cfg_reference_noise_pairing=True,
            ba_reference_token_text_mode="zero",
            ba_reference_pooled_text_mode="zero",
            ba_output_anchor_mode="none",
            ba_ppr_collect_diagnostics=True,
            ba_ppr_diagnostic_steps=(0,),
            ba_ppr_diagnostic_sample_keys=("sample0",),
            _ba_ppr_randomness_fingerprints={},
            _ba_ppr_epsilon_diagnostics=[],
            _ba_ppr_tensor_diagnostics=[],
            _original_attn_processors={"plain.processor": object()},
            guidance_scale=5.0,
            _cross_attention_kwargs=None,
        )
        target = torch.zeros(2, 4, 8, 8)
        reference = torch.ones(1, 4, 8, 8)
        prompt = torch.randn(2, 4, 8)
        face_prompt = torch.randn_like(prompt)
        pooled = torch.randn(2, 6)
        time_ids = torch.randn(2, 6)
        mask = torch.ones(1, 1, 8, 8)

        with patch(
            "src.model.photomaker_branched.branched_runtime."
            "patch_unet_attention_processors"
        ):
            two_branch_predict(
                pipeline=pipeline,
                latent_model_input=target,
                t=torch.tensor([1, 1]),
                prompt_embeds=prompt,
                added_cond_kwargs={
                    "text_embeds": pooled,
                    "time_ids": time_ids,
                },
                mask4=mask,
                mask4_ref=mask,
                reference_latents=reference,
                face_prompt_embeds=face_prompt,
            )

        self.assertEqual(tuple(pipeline._ref_noise_base.shape), (1, 4, 8, 8))
        self.assertTrue(
            pipeline._ba_ppr_randomness_fingerprints[
                "cfg_reference_noise_equal"
            ]
        )
        self.assertEqual(
            len(
                pipeline._ba_ppr_randomness_fingerprints[
                    "reference_noise_used_sha256"
                ]
            ),
            1,
        )
        self.assertEqual(
            pipeline._ba_ppr_randomness_fingerprints["reference_ca_mode"],
            ["zero"],
        )
        torch.testing.assert_close(
            unet.samples[0][2],
            unet.samples[0][3],
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            int(torch.count_nonzero(unet.encoders[0][2:])),
            0,
        )
        torch.testing.assert_close(
            unet.addeds[0]["text_embeds"][:2],
            pooled,
        )
        self.assertEqual(
            int(torch.count_nonzero(unet.addeds[0]["text_embeds"][2:])),
            0,
        )
        torch.testing.assert_close(
            unet.addeds[0]["time_ids"][2:],
            time_ids,
        )

    def test_reference_noise_runner_processes_batched_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("PM0", "R1N1", "R2N1", "R1N2", "R2N2"):
                (root / name).mkdir()
            for name in (
                "contact_sheets",
                "difference_heatmaps",
                "face_crops",
            ):
                (root / name).mkdir()
            source_paths = []
            for index in range(2):
                path = root / f"id{index}.png"
                Image.new("RGB", (16, 16), (index * 80, 0, 0)).save(path)
                source_paths.append(path)

            state = {
                "root": root,
                "noise_seeds": {"N1": 11, "N2": 22},
                "reference_ca_mode": "zero",
                "reference_ca_override": "zero",
                "swap_map": {
                    "id0": ("id1", source_paths[1], [1, 1, 15, 15]),
                    "id1": ("id0", source_paths[0], [1, 1, 15, 15]),
                },
                "rows": [],
                "pair_rows": [],
                "tensor_rows": [],
                "integrity": {},
                "filenames": [],
                "next_index": 0,
                "device": torch.device("cpu"),
                "lpips_status": "not attempted",
                "observed_batch_sizes": [],
            }
            trainer = SimpleNamespace(
                _ppr_reference_noise_state=state,
                config=SimpleNamespace(validation_args={"seed": 0}),
                metrics=[],
            )
            batch = {
                "prompt": ["prompt0", "prompt1"],
                "id": ["id0", "id1"],
                "seed": [3, 4],
                "ref_images": [
                    [Image.open(source_paths[0]).convert("RGB")],
                    [Image.open(source_paths[1]).convert("RGB")],
                ],
                "face_bbox_ref": [[1, 1, 15, 15], [1, 1, 15, 15]],
                "face_bbox_gen": [[1, 1, 15, 15], [1, 1, 15, 15]],
            }
            calls = []

            def fake_generate(*args, **kwargs):
                del args
                calls.append(kwargs)
                sample_keys = kwargs["sample_keys"]
                variant = kwargs["diagnostic_variant"]
                images = [
                    Image.new(
                        "RGB",
                        (16, 16),
                        (20 * list(("PM0", "R1N1", "R2N1", "R1N2", "R2N2")).index(variant), index, 0),
                    )
                    for index in range(len(sample_keys))
                ]
                records = [
                    {
                        "record_type": "generation_randomness",
                        "sample": sample,
                    }
                    for sample in sample_keys
                ]
                if variant != "PM0":
                    records.append(
                        {
                            "record_type": "processor_applied_ratio",
                            "samples": list(sample_keys),
                            "applied_ratios": [0.1]
                            * (2 * len(sample_keys)),
                            "cap_scales": [1.0]
                            * (2 * len(sample_keys)),
                        }
                    )
                return images, records

            metrics = SimpleNamespace(update=lambda *args, **kwargs: None)
            with (
                patch(
                    "src.trainer.ppr_reference_noise._generate",
                    side_effect=fake_generate,
                ),
                patch(
                    "src.trainer.ppr_reference_noise._assert_integrity"
                ),
                patch(
                    "src.trainer.ppr_reference_noise._tensor_comparisons",
                    return_value=[],
                ),
                patch(
                    "src.trainer.ppr_reference_noise._face_lpips",
                    return_value=0.0,
                ),
            ):
                output = run_ppr_reference_noise_batch(
                    trainer, batch, metrics
                )

            self.assertEqual(len(output["generated"]), 2)
            self.assertEqual(len(state["rows"]), 10)
            self.assertEqual(len(state["pair_rows"]), 8)
            self.assertEqual(state["observed_batch_sizes"], [2])
            swapped_calls = [
                call for call in calls
                if call["diagnostic_variant"].startswith("R2")
            ]
            self.assertTrue(swapped_calls)
            self.assertTrue(
                all(len(call["ppr_reference_image"]) == 2 for call in swapped_calls)
            )
            self.assertTrue(
                all(len(call["ppr_face_bbox_ref"]) == 2 for call in swapped_calls)
            )

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

    def test_learned_null_is_registered_in_manifest_and_optimizer(self) -> None:
        model = _tiny_model()
        model.ba_connector_input_mode = "reference_minus_learned_null"
        model.ba_null_memory_tokens = 3
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(model, mask, mask)
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        groups = PhotomakerBranchedLora.get_trainable_params(
            SimpleNamespace(
                unet=model.unet,
                ba_processor_variant="packed_residual_v1",
                ba_connector_input_mode="reference_minus_learned_null",
            ),
            _OptimizerConfig(lr_for_lora=5e-5),
        )
        self.assertEqual(groups[-1]["name"], "ba_ppr_null_memory")
        self.assertEqual(len(groups[-1]["params"]), 1)

    def test_nn5b_identity_kv_are_registered_in_manifest_and_optimizer(self) -> None:
        model = _tiny_model()
        model.ba_connector_input_mode = "reference_minus_learned_null"
        model.ba_null_memory_tokens = 3
        model.ba_identity_token_lane = True
        model.ba_identity_token_dim = 2048
        model.ba_identity_token_rank = 4
        model.ba_identity_token_weight = 0.5
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(model, mask, mask)
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        groups = PhotomakerBranchedLora.get_trainable_params(
            SimpleNamespace(
                unet=model.unet,
                ba_processor_variant="packed_residual_v1",
                ba_connector_input_mode="reference_minus_learned_null",
                ba_identity_token_lane=True,
            ),
            _OptimizerConfig(lr_for_lora=5e-5),
        )
        self.assertEqual(groups[-1]["name"], "ba_ppr_identity_tokens")
        self.assertEqual(len(groups[-1]["params"]), 4)

    def test_nn6_identity_only_trainability_and_optimizer_are_exact(self) -> None:
        model = _tiny_model()
        model.ba_site_policy = "up_blocks0_attn1"
        model.disable_branched_ca = True
        model.ba_connector_input_mode = "reference_minus_learned_null"
        model.ba_identity_token_lane = True
        model.ba_identity_token_dim = 2048
        model.ba_identity_token_rank = 4
        model.ba_identity_token_weight = 0.5
        model.ba_identity_fusion_mode = "identity_only"
        model.ba_identity_site_policy = "up_blocks0_attn1"
        model.ba_spatial_site_policy = "up_blocks0_attn1"
        model.ba_spatial_lane_enabled = False
        model.ba_identity_null_tokens = 2
        model.ba_identity_connector_rank = 4
        model.ba_identity_gate_max = 0.5
        model.ba_identity_gate_init_logit = 0.0
        model.ba_identity_delta_rms_cap = 0.15
        model.ba_spatial_gate_max = 0.15
        model.ba_spatial_delta_rms_cap = 0.03
        model.ba_total_delta_rms_cap = 0.15
        mask = torch.ones(1, 1, 8, 8)
        patch_unet_attention_processors(
            model,
            mask,
            mask,
            identity_tokens=torch.ones(1, 2, 2048),
        )
        configure_branched_trainables(model)
        _assert_branched_installation(model)
        processor = model.unet.attn_processors["up_blocks.0.attn1.processor"]
        self.assertIsNone(processor.ref_to_k)
        self.assertIsNone(processor.connector_down)
        self.assertEqual(
            {name for name, parameter in processor.named_parameters() if parameter.requires_grad},
            {
                "identity_to_k.0.weight",
                "identity_to_k.2.weight",
                "identity_to_v.0.weight",
                "identity_to_v.2.weight",
                "identity_null_memory",
                "identity_connector_down.weight",
                "identity_connector_up.weight",
                "identity_gate_logit",
            },
        )
        groups = PhotomakerBranchedLora.get_trainable_params(
            SimpleNamespace(
                unet=model.unet,
                ba_processor_variant="packed_residual_v1",
                ba_identity_fusion_mode="identity_only",
            ),
            _OptimizerConfig(lr_for_lora=5e-5),
        )
        self.assertEqual(
            [group["name"] for group in groups],
            [
                "ba_ppr_identity_k",
                "ba_ppr_identity_v",
                "ba_ppr_identity_null",
                "ba_ppr_identity_connector_down",
                "ba_ppr_identity_connector_up",
                "ba_ppr_identity_gate",
            ],
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

    def test_n3a_new2_install_has_all_legacy_sa_and_no_ca(self) -> None:
        model = _tiny_model()
        model.ba_processor_variant = "legacy"
        model.ba_site_policy = "all"
        model.ba_sa_train_mode = "all"
        model.ba_sa_face_mode = "dual"
        model.ba_sa_mix_init = 0.35
        model.ba_sa_ref_layer_scope = "up"
        model.branched_attn_weight_mode = "noise_and_ref"
        model.branched_attn_new_weight_kind = "lora"
        model.disable_branched_ca = True
        model.train_branched_ca_lora = False
        mask = torch.ones(1, 1, 8, 8)

        patch_unet_attention_processors(model, mask, mask)
        configure_branched_trainables(model)
        _assert_branched_installation(model)

        self.assertEqual(len(model._ba_patched_sa_names), 3)
        self.assertEqual(len(model._ba_patched_ca_names), 0)
        for name in model._ba_patched_sa_names:
            processor = model.unet.attn_processors[name]
            self.assertEqual(processor.ba_sa_ref_layer_scope, "up")
            torch.testing.assert_close(
                processor.face_mix_logits.sigmoid(),
                torch.full((4,), 0.35),
            )
        self.assertFalse(
            any(
                parameter.requires_grad
                for name, parameter in model.unet.named_parameters()
                if ".attn2.processor." in name
            )
        )

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

        # The final epsilon anchor is processor-agnostic; legacy N3a uses the
        # same ordinary PhotoMaker base pass and face-core localization.
        unet.attn_processors = {"legacy": nn.Identity()}
        reset_branched_generation_caches(pipeline)
        with patch(
            "src.model.photomaker_branched.branched_runtime."
            "patch_unet_attention_processors"
        ), torch.no_grad():
            legacy_localized, _, _ = two_branch_predict(**kwargs)
        torch.testing.assert_close(legacy_localized, expected, atol=0, rtol=0)
        self.assertIn("legacy", unet.attn_processors)

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
