from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.model.photomaker_branched.branched_runtime import (
    compose_post_cfg_identity_delta,
)
from src.model.photomaker_branched.config_utils import (
    branched_model_runtime_kwargs,
)
from src.model.photomaker_branched.lora2_helpers import (
    InvalidIdentityConditioningSample,
    _detect_face_with_bbox_fallback,
    _raise_if_any_rank_has_invalid_identity,
)
from src.pipelines.br_pipeline_helpers import select_mode_and_prompts


def _mode(
    step: int,
    *,
    photomaker_start_step: int = 10,
    branched_attn_start_step: int = 10,
    branched_start_mode: str = "both",
) -> str:
    pipeline = SimpleNamespace(
        branched_start_mode=branched_start_mode,
        _pose_user_ratio=0.0,
        pose_adapt_ratio=0.0,
    )
    embeds = torch.zeros(1, 2, 3)
    pooled = torch.zeros(1, 3)
    mode, *_ = select_mode_and_prompts(
        pipeline,
        i=step,
        photomaker_start_step=photomaker_start_step,
        branched_attn_start_step=branched_attn_start_step,
        branched_attn_end_step=None,
        prompt_embeds_text_only=embeds,
        pooled_prompt_embeds_text_only=pooled,
        prompt_embeds=embeds,
        pooled_prompt_embeds=pooled,
        force_par_before_pm=False,
        pose_forced_logged=False,
        pose_relaxed_logged=False,
    )
    return mode


class BranchedIdentityScheduleTest(unittest.TestCase):
    def test_equal_identity_start_steps_activate_both_paths(self):
        self.assertEqual(_mode(9), "NO_ID")
        self.assertEqual(_mode(10), "BOTH")
        self.assertEqual(_mode(49), "BOTH")

    def test_equal_identity_start_steps_support_branched_only_mode(self):
        self.assertEqual(_mode(10, branched_start_mode="branched"), "BRANCHED")

    def test_staggered_photomaker_then_branched_schedule_is_unchanged(self):
        self.assertEqual(_mode(10, branched_attn_start_step=15), "PHOTOMAKER")
        self.assertEqual(_mode(15, branched_attn_start_step=15), "BOTH")

    def test_n36_n37_and_n38_resolve_to_staged_schedule(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "src" / "configs")
        for config_name in (
            "one_id_ba_identity_owner_qformer_N36",
            "one_id_ba_identity_owner_hybrid_N37",
            "one_id_ba_identity_owner_cropped_qformer_N38",
        ):
            with self.subTest(config_name=config_name):
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                self.assertEqual(cfg.model.photomaker_start_step, 10)
                self.assertEqual(cfg.model.branched_attn_start_step, 15)
                self.assertEqual(cfg.pipeline.photomaker_start_step, 10)
                self.assertEqual(cfg.pipeline.branched_attn_start_step, 15)
                self.assertEqual(cfg.pipeline.branched_start_mode, "both")
                self.assertEqual(cfg.validation_args.photomaker_start_step, 10)
                self.assertEqual(cfg.validation_args.branched_attn_start_step, 15)

    def test_new_identity_owner_family_guidance_scales_post_cfg_delta(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "src" / "configs")
        for config_name in (
            "one_id_ba_causal_highres_qformer_N34",
            "one_id_ba_causal_canonical_parts_N35",
            "one_id_ba_identity_owner_qformer_N36",
            "one_id_ba_identity_owner_hybrid_N37",
            "one_id_ba_identity_owner_cropped_qformer_N38",
        ):
            with self.subTest(config_name=config_name):
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                self.assertTrue(cfg.model.ba_post_cfg_guidance_scale)
                self.assertTrue(cfg.model.ba_reference_face_bbox_fallback)
                self.assertTrue(cfg.model.ba_skip_invalid_identity_samples)

    def test_reference_detection_retries_known_bbox_and_restores_landmark_coordinates(self):
        face = SimpleNamespace(
            bbox=np.asarray([4.0, 4.0, 30.0, 30.0], dtype=np.float32),
            embedding=np.ones(512, dtype=np.float32),
            kps=np.asarray(
                [[8.0, 9.0], [20.0, 9.0], [14.0, 15.0], [9.0, 23.0], [19.0, 23.0]],
                dtype=np.float32,
            ),
        )
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        with patch(
            "src.model.photomaker_branched.lora2_helpers.analyze_faces",
            side_effect=[[], [face]],
        ):
            detected, landmarks, used_fallback = _detect_face_with_bbox_fallback(
                object(),
                image,
                [20.0, 30.0, 60.0, 70.0],
                require_embedding=True,
                require_landmarks=True,
            )

        self.assertIs(detected, face)
        self.assertTrue(used_fallback)
        np.testing.assert_allclose(landmarks[0], [14.0, 25.0])

    def test_invalid_identity_raises_typed_error_without_distributed_runtime(self):
        with self.assertRaises(InvalidIdentityConditioningSample):
            _raise_if_any_rank_has_invalid_identity(
                SimpleNamespace(device=torch.device("cpu")),
                ["reference_face_missing sample=0"],
            )

    def test_post_cfg_identity_delta_can_restore_cfg_strength(self):
        pm = torch.tensor([1.0, 3.0]).reshape(2, 1, 1, 1)
        ba = torch.tensor([100.0, 5.0]).reshape(2, 1, 1, 1)
        mask = torch.ones(1, 1, 1, 1)

        legacy_strength = compose_post_cfg_identity_delta(
            pm,
            ba,
            mask,
            guidance_scale=5.0,
            residual_scale=1.0,
            do_classifier_free_guidance=True,
            scale_identity_delta_by_guidance=False,
        )
        restored_strength = compose_post_cfg_identity_delta(
            pm,
            ba,
            mask,
            guidance_scale=5.0,
            residual_scale=1.0,
            do_classifier_free_guidance=True,
            scale_identity_delta_by_guidance=True,
        )

        self.assertTrue(torch.equal(legacy_strength, torch.full_like(pm, 13.0)))
        self.assertTrue(torch.equal(restored_strength, torch.full_like(pm, 21.0)))

    def test_alternate_validation_uses_training_runtime_constructor_kwargs(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                _target_="src.model.photomaker_branched.lora2.PhotomakerBranchedLora"
            ),
            train_ba_only=True,
            ba_train_top_k=0.5,
            ba_patch_top_k=0.75,
            non_ba_train=True,
            train_ba_all_steps=True,
            ba_weights_split=True,
            use_attn_v2=True,
        )
        self.assertEqual(
            branched_model_runtime_kwargs(cfg),
            {
                "train_ba_only": True,
                "ba_train_top_k": 0.5,
                "ba_patch_top_k": 0.75,
                "non_ba_train": True,
                "train_ba_all_steps": True,
                "ba_weights_split": True,
                "use_attn_v2": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
