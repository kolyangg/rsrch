#!/usr/bin/env python3
"""Compose and validate the three clean E13-family recipes."""

from __future__ import annotations

import hashlib
import json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src/configs"
DIFFUSION_ROOT = CONFIG_DIR.parents[1]
RECIPES = {
    "E13_large_ds_joint_shadow_sa128_24k": ("large_dataset", 0),
    "BC_E13_big_celebs_joint_shadow_sa128_24k": ("big_celebs", 0),
    "CL14_cosmic_joint_shadow_sa128_softmask_24k": ("cosmic_large_adapted", 2),
}
BBOX_PROTOCOLS = {
    "E13_large_ds_joint_shadow_sa128_24k": (
        "../dataset_full/val_dataset/protocols/e13_bc_e13/pm96_bboxes_new.json",
        "4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099",
    ),
    "BC_E13_big_celebs_joint_shadow_sa128_24k": (
        "../dataset_full/val_dataset/protocols/e13_bc_e13/pm96_bboxes_new.json",
        "4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099",
    ),
    "CL14_cosmic_joint_shadow_sa128_softmask_24k": (
        "../dataset_full/val_dataset/protocols/cl14/pm96_bboxes_new.json",
        "b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d",
    ),
}
MANUAL_BBOX_SHA256 = "a39645e22b68027175946a028e185b7c5393a7514f5d68c94cd74e7cc9f5e614"


def _require(config, path: str, expected) -> None:
    actual = OmegaConf.select(config, path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected {expected!r}, got {actual!r}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bbox_protocol(relative_path: str, expected_auto_sha256: str) -> None:
    manual_path = (DIFFUSION_ROOT / relative_path).resolve()
    auto_path = manual_path.with_name(f"{manual_path.stem}_auto.json")
    for path, expected_hash in (
        (manual_path, MANUAL_BBOX_SHA256),
        (auto_path, expected_auto_sha256),
    ):
        if not path.is_file():
            raise RuntimeError(f"Pinned validation bbox file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Pinned validation bbox hash drifted at {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or len(payload) != 96:
            raise RuntimeError(f"Expected 96 validation bbox records in {path}")
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    force_manual = {key for key, value in manual.items() if value.get("force_manual")}
    if force_manual != {"Reading pa_jensen.png"}:
        raise RuntimeError(f"Pinned force-manual bbox routing drifted: {force_manual}")


def main() -> None:
    # 10 Aug 2026 - E13C-CFG-01/02: Configuration validation is intentionally
    # independent of model loading and catches recipe drift before GPU startup.
    composed = {}
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        for name, (dataset, feather) in RECIPES.items():
            config = compose(config_name=name)
            composed[name] = config
            for path, expected in {
                "train_dataset_name": dataset,
                "train_on_separate_image": True,
                "train_ba_all_steps": True,
                "train_ba_only": True,
                "branched_attn_weight_mode": "noise_and_ref",
                "branched_attn_new_weight_kind": "lora",
                "train_branched_ca_lora": False,
                "disable_branched_sa": False,
                "disable_branched_ca": True,
                "model.e13_family_contract": True,
                "model.rank": 32,
                "model.ba_hard_v1_lora_rank": 128,
                "model.generic_adapter_train_scope": "effective_all",
                "model.photomaker_default_train_scope": "effective_all",
                "model.strict_branched_install": True,
                "model.strict_trainable_contract": True,
                "model.branched_state_dict_mode": "trainable_unet_v2",
                "model.ba_training_mask_feather": feather,
                "model.conditioning_cache_enabled": False,
                "model.skip_unused_text_conditioning": True,
                "model.batched_conditioning_preparation": True,
                "model.cache_prepared_masks": True,
                "model.compute_branch_debug_outputs": False,
                "pipeline.pose_adapt_ratio": 0.0,
                "pipeline.ca_mixing_for_face": False,
                "pipeline.photomaker_start_step": 10,
                "pipeline.branched_attn_start_step": 15,
                "trainer.epoch_len": 2000,
                "trainer.n_epochs": 12,
                "trainer.save_period": 1,
                "trainer.post_backward_parameter_touch": False,
                "trainer.grad_norm_log_only": True,
                "trainer.face_quality.execution_mode": "deferred",
                "weights_only_save_period": 1,
                "lr_scheduler._target_": (
                    "src.lr_schedulers.lr_schedulers.WarmupHoldCosineLR"
                ),
                "lr_scheduler.warmup_steps": 20,
                "lr_scheduler.hold_steps": 14000,
                "lr_scheduler.total_steps": 24000,
                "lr_scheduler.min_factor": 0.1,
                "dataloaders.train.batch_size": 2,
                "dataloaders.train.num_workers": 2,
                "dataloaders.manual_val.batch_size": 12,
                "dataloaders.manual_val.num_workers": 1,
                "validation_shadow_photomaker_default": True,
                "validation_processor_base_mode": "legacy_full_copy",
                "strict_validation_processor_copy": True,
                "update_proc_weights_val": True,
                "pretrained_model_for_validation_name_or_path": (
                    "SG161222/RealVisXL_V4.0"
                ),
                "validation_args.num_images_per_prompt": 1,
                "validation_args.num_inference_steps": 50,
                "validation_args.guidance_scale": 5,
                "validation_args.photomaker_start_step": 10,
                "validation_args.branched_attn_start_step": 15,
                "validation_args.use_bbox_mask_gen": True,
                "automatic_bboxes": True,
                "automatic_bboxes_every_val": False,
                "expected_trainable_contract.total_tensors": 2240,
                "expected_trainable_contract.total_parameters": 219217920,
            }.items():
                _require(config, path, expected)
            bbox_path, expected_auto_hash = BBOX_PROTOCOLS[name]
            _require(config, "datasets.val.manual_val.bbox_mask_gen", bbox_path)
            # 10 Aug 2026 - E13C-PIPE-03: Validation generations depend on the
            # exact historical automatic box cache, so reject a fresh detector
            # pass or a similarly named protocol before GPU startup.
            _validate_bbox_protocol(bbox_path, expected_auto_hash)
            print(f"{name}: OK")

    # 10 Aug 2026 - E13C-CFG-02: Compare the shared output-affecting projection
    # across leaves. Only CL14's training mask feather is excluded here; dataset
    # policy is validated independently below.
    projection_paths = (
        "train_on_separate_image", "train_ba_all_steps", "train_ba_only",
        "branched_attn_weight_mode", "branched_attn_new_weight_kind",
        "train_branched_ca_lora", "disable_branched_sa",
        "disable_branched_ca", "ba_patch_top_k", "ba_train_top_k",
        "non_ba_train", "strict_face_routing", "loss_kind", "lambda_face",
        "model.rank", "model.ba_hard_v1_lora_rank",
        "model.generic_adapter_train_scope",
        "model.photomaker_default_train_scope",
        "pipeline.pose_adapt_ratio", "pipeline.ca_mixing_for_face",
        "pipeline.photomaker_start_step", "pipeline.branched_attn_start_step",
        "trainer.epoch_len", "trainer.n_epochs", "trainer.save_period",
        "lr_scheduler._target_", "lr_scheduler.warmup_steps",
        "lr_scheduler.hold_steps", "lr_scheduler.total_steps",
        "lr_scheduler.min_factor", "validation_args.num_inference_steps",
        "validation_args.guidance_scale", "validation_args.photomaker_start_step",
        "validation_args.branched_attn_start_step",
        "validation_shadow_photomaker_default", "validation_processor_base_mode",
        "pretrained_model_for_validation_name_or_path",
    )
    projections = {
        name: tuple(OmegaConf.select(config, path) for path in projection_paths)
        for name, config in composed.items()
    }
    baseline_name = "E13_large_ds_joint_shadow_sa128_24k"
    for name, projection in projections.items():
        if projection != projections[baseline_name]:
            raise RuntimeError(f"Shared E13 projection drifted in {name}")

    e13 = composed[baseline_name]
    bc = composed["BC_E13_big_celebs_joint_shadow_sa128_24k"]
    cl14 = composed["CL14_cosmic_joint_shadow_sa128_softmask_24k"]
    for config, path, expected in (
        (e13, "datasets.train.large_dataset._target_", "src.datasets.large_dataset.LargeDatasetTrain"),
        (e13, "datasets.train.large_dataset.train_on_separate_image", True),
        (bc, "datasets.train.big_celebs._target_", "src.datasets.big_celebs.BigCelebsTrain"),
        # oc.env values compose as strings; BigCelebsTrain performs the
        # fail-closed integer conversion and range check at instantiation.
        (bc, "datasets.train.big_celebs.min_face_res", "192"),
        (bc, "datasets.train.big_celebs.random_horizontal_flip", True),
        (cl14, "datasets.train.cosmic_large_adapted._target_", "src.datasets.cosmic_large_adapted.CosmicLargeAdaptedTrain"),
        (cl14, "datasets.train.cosmic_large_adapted.reference_frame_mode", "target_face_frame"),
        (cl14, "datasets.train.cosmic_large_adapted.random_reference_flip", False),
        (cl14, "datasets.train.cosmic_large_adapted.prompt_mode", "pose_first"),
        (cl14, "datasets.train.cosmic_large_adapted.prompt_max_words", 50),
        (cl14, "datasets.train.cosmic_large_adapted.reference_position_jitter", 0.15),
    ):
        _require(config, path, expected)
    if list(OmegaConf.select(
        cl14, "datasets.train.cosmic_large_adapted.reference_scale_jitter"
    )) != [0.06, 0.30]:
        raise RuntimeError("CL14 reference_scale_jitter drifted")
    print("E13-family shared projection, dataset leaves and bbox protocols: OK")


if __name__ == "__main__":
    main()
