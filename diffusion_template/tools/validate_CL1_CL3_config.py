#!/usr/bin/env python3
"""Fail-closed composition/spec gate for the CL1-CL3 Cosmic Large suite.

Asserts that every arm inherits the exact E13 contract and differs from it only
by its declared reference-lane delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"
BASELINE = "E13_large_ds_joint_shadow_sa128_24k"
LAUNCHER = "launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh"

# config name -> (train dataset, reference_roi_warp, reference_frame_mode)
ARMS = {
    "CL0_cosmic_joint_shadow_sa128_asis_24k": (
        "cosmic_large_adapted", False, "native",
    ),
    "CL1_cosmic_joint_shadow_sa128_sceneref_24k": (
        "cosmic_large_sceneref", False, None,
    ),
    "CL2_cosmic_joint_shadow_sa128_facecanon_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL3_cosmic_joint_shadow_sa128_fmtfix_24k": (
        "cosmic_large_adapted", True, "native",
    ),
    "CL4_cosmic_joint_shadow_sa128_hygiene_24k": (
        "cosmic_large_adapted", False, "native",
    ),
    "CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k": (
        "cosmic_large_adapted", True, "native",
    ),
    "CL6_cosmic_joint_shadow_sa128_boundary_24k": (
        "cosmic_large_adapted", True, "native",
    ),
    "CL7_cosmic_joint_shadow_sa128_altloss_24k": (
        "cosmic_large_adapted", True, "native",
    ),
    "CL8_cosmic_joint_shadow_sa128_fullbody_24k": (
        "cosmic_large_adapted", False, "native",
    ),
    "CL9_cosmic_joint_shadow_sa128_refscale_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL13_cosmic_joint_shadow_sa128_refdropout_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
    "CL14_cosmic_joint_shadow_sa128_softmask_24k": (
        "cosmic_large_adapted", False, "target_face_frame",
    ),
}
# CL8 deliberately lowers min_face_res to restore the full-body targets, so it is
# the only arm exempt from the shared min_face_res=192 control.
FULLBODY_ARMS = ("CL8_cosmic_joint_shadow_sa128_fullbody_24k",
                 "CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k",
                 "CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k")
# CL7 deliberately changes trainer.masked_loss_step (face-only -> alternating),
# so it is the only arm exempt from that parity field.
ALT_LOSS_ARM = "CL7_cosmic_joint_shadow_sa128_altloss_24k"
# CL0 is the deliberately unimproved baseline, so it is exempt from the shared
# hygiene controls that every other arm must satisfy.
BASELINE_ARM = "CL0_cosmic_joint_shadow_sa128_asis_24k"

# Fields that must match E13 exactly in every arm.
INHERITED = (
    "trainer.epoch_len",
    "trainer.n_epochs",
    "trainer.validation_interval_steps",
    "trainer.masked_loss_step",
    "trainer.face_quality.expected_images",
    "validation_shadow_photomaker_default",
    "model.rank",
    "model.ba_hard_v1_lora_rank",
    "model.ba_architecture_version",
    "model.generic_adapter_train_scope",
    "model.photomaker_default_train_scope",
    "model.ba_hard_v1_true_reference_key_mask",
    "model.ba_hard_v1_branch_output_rank",
    "lr_for_lora",
    "ba_lr",
    "generic_adapter_lr",
    "photomaker_default_lr",
    "lr_scheduler.warmup_steps",
    "lr_scheduler.hold_steps",
    "lr_scheduler.total_steps",
    "lr_scheduler.min_factor",
    "expected_trainable_contract.total_tensors",
    "expected_trainable_contract.total_parameters",
    "expected_trainable_contract.optimizer_tensors",
    "expected_trainable_contract.optimizer_parameters",
    "pipeline.pose_adapt_ratio",
    "pipeline.ca_mixing_for_face",
    "disable_branched_ca",
    "train_branched_ca_lora",
    "validation_processor_base_mode",
    # NOTE: model.batched_conditioning_preparation is intentionally NOT inherited-
    # checked; multi-reference arms must disable it (see E19).
    "dataloaders.train.batch_size",
    "validation_args.guidance_scale",
    "validation_args.num_images_per_prompt",
    "datasets.val.manual_val.limit",
)


def value(config, path: str):
    selected = OmegaConf.select(config, path, default="<missing>")
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


def load(name: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-spec", required=True)
    args = parser.parse_args()

    if args.config_name not in ARMS:
        raise RuntimeError(f"Unapproved CL config: {args.config_name}")
    dataset_name, expect_warp, expect_frame = ARMS[args.config_name]
    if not args.run_name.startswith(args.config_name.split("_")[0] + "_"):
        raise RuntimeError(f"Run name {args.run_name} does not match its arm")

    baseline = load(BASELINE)
    config = load(args.config_name)

    inherited = tuple(
        f for f in INHERITED
        if not (args.config_name == ALT_LOSS_ARM and f == "trainer.masked_loss_step")
    )
    drift = {
        path: (value(baseline, path), value(config, path))
        for path in inherited
        if value(baseline, path) != value(config, path)
    }
    if drift:
        raise RuntimeError(f"Arm drifted from the E13 contract: {drift}")

    if value(config, "train_dataset_name") != dataset_name:
        raise RuntimeError(f"Expected train_dataset_name={dataset_name}")
    if value(config, "val_datasets_names") != ["manual_val"]:
        raise RuntimeError("CL arms must validate on the fixed manual_val panel")
    if bool(value(config, "model.ba_hard_v1_reference_roi_warp")) is not expect_warp:
        raise RuntimeError(f"Expected ba_hard_v1_reference_roi_warp={expect_warp}")

    dataset = value(config, f"datasets.train.{dataset_name}")
    if args.config_name == BASELINE_ARM:
        if dataset.get("random_reference_flip") is not True or dataset.get("prompt_mode") != "legacy":
            raise RuntimeError("CL0 must preserve the pre-CL loader behaviour exactly")
        print(json.dumps({"status": "ok", "run_name": args.run_name, "arm": "baseline",
                          "train_dataset": dataset_name,
                          "optimizer_steps": int(value(config, "trainer.epoch_len"))
                          * int(value(config, "trainer.n_epochs"))}, indent=2, sort_keys=True))
        return
    if dataset.get("prompt_mode") != "pose_first" or dataset.get("prompt_max_words") != 50:
        raise RuntimeError("CL arms share pose-first captions capped at 50 words")
    if dataset.get("random_reference_flip") is not False:
        raise RuntimeError("CL arms must not mirror the reference")
    if args.config_name in FULLBODY_ARMS:
        if int(dataset.get("min_face_res", 192)) >= 192:
            raise RuntimeError("CL8 must lower min_face_res to restore full-body targets")
        if not dataset.get("target_scale_balance"):
            raise RuntimeError("CL8 requires target_scale_balance to offset the small-face majority")
        if not dataset.get("target_scale_bins"):
            raise RuntimeError("CL8 requires target_scale_bins")
        if (args.config_name.startswith("CL10")
                and dataset.get("target_scale_balance_mode") != "oversample"):
            # `reorder` is destroyed by DataLoader shuffling, i.e. it is inert.
            raise RuntimeError("CL10 requires target_scale_balance_mode=oversample")
    elif dataset.get("min_face_res") != 192:
        raise RuntimeError("CL arms share min_face_res=192")

    if args.config_name in ("CL9_cosmic_joint_shadow_sa128_refscale_24k",
                            "CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k",
                            "CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k",
                            "CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k",
                            "CL13_cosmic_joint_shadow_sa128_refdropout_24k",
                            "CL14_cosmic_joint_shadow_sa128_softmask_24k"):
        jit = dataset.get("reference_scale_jitter")
        if not jit or not (0.03 <= float(jit[0]) < float(jit[1]) <= 0.60):
            raise RuntimeError("CL9 requires reference_scale_jitter inside [0.03, 0.60]")
        if float(dataset.get("reference_position_jitter", 0)) <= 0:
            raise RuntimeError("CL9 requires a positive reference_position_jitter")

    if expect_frame is not None:
        if dataset.get("reference_frame_mode") != expect_frame:
            raise RuntimeError(f"Expected reference_frame_mode={expect_frame}")
        if expect_frame == "target_face_frame" and any(
            dataset.get(key) is not None
            for key in (
                "reference_crop_margin",
                "reference_content_size",
                "reference_canvas_size",
            )
        ):
            raise RuntimeError(
                "target_face_frame requires null crop_margin/content_size/canvas_size"
            )

    loss_kind = value(config, "loss_kind")
    loss_names = value(config, "writer.loss_names") or []
    if args.config_name == "CL6_cosmic_joint_shadow_sa128_boundary_24k":
        if loss_kind != "branched_reference":
            raise RuntimeError("CL6 requires loss_kind=branched_reference")
        lf = value(config, "loss_function") or {}
        if float(lf.get("boundary_weight", 0)) <= 0 or int(lf.get("boundary_ring_width", 0)) < 1:
            raise RuntimeError("CL6 requires a positive boundary ring term")
        if "loss_boundary" not in loss_names:
            raise RuntimeError("CL6 must log loss_boundary or the term is invisible")
        if int(value(config, "trainer.masked_loss_step")) != 1:
            raise RuntimeError("CL6 keeps masked_loss_step=1; only CL7 alternates")
    elif args.config_name == ALT_LOSS_ARM:
        if loss_kind != "masked_alternating_audited":
            raise RuntimeError("CL7 requires loss_kind=masked_alternating_audited")
        if int(value(config, "trainer.masked_loss_step")) != 2:
            raise RuntimeError("CL7 requires masked_loss_step=2")
        if "loss_full" not in loss_names:
            raise RuntimeError("CL7 must log loss_full or the term is invisible")

    # `value` returns the literal "<missing>" sentinel for an absent key. Both
    # flags default to off in `lora2.py`, so absent means off.
    def training_flag(path, cast):
        raw = value(config, path)
        return cast(0) if raw in ("<missing>", None, "") else cast(raw)

    drop = training_flag("model.ba_reference_dropout_probability", float)
    feather = training_flag("model.ba_training_mask_feather", int)
    if args.config_name.startswith("CL13"):
        if not 0.0 < drop <= 0.5:
            raise RuntimeError("CL13 requires a positive ba_reference_dropout_probability")
        if feather:
            raise RuntimeError("CL13 must not also feather the mask")
    elif args.config_name.startswith("CL14"):
        if feather < 1:
            raise RuntimeError("CL14 requires ba_training_mask_feather >= 1")
        if drop:
            raise RuntimeError("CL14 must not also drop the reference")
    else:
        if drop or feather:
            raise RuntimeError(
                f"{args.config_name} must leave both training-only flags off "
                f"(drop={drop}, feather={feather})"
            )

    refs = int(dataset.get("num_identity_refs", 1) or 1)
    if refs > 1 and bool(value(config, "model.batched_conditioning_preparation")):
        raise RuntimeError(
            "num_identity_refs>1 requires model.batched_conditioning_preparation=false"
        )

    spec_path = Path(args.experiment_spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    plan = spec.get("plan", {})
    if spec.get("run_name") != args.run_name:
        raise RuntimeError("Experiment spec run name mismatch")
    if plan.get("config") != f"src/configs/{args.config_name}.yaml":
        raise RuntimeError("Experiment spec config mismatch")
    if plan.get("launcher") != LAUNCHER:
        raise RuntimeError("Experiment spec launcher mismatch")
    if plan.get("comet_project") != "aug-large-ds":
        raise RuntimeError("Experiment spec Comet project mismatch")

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": args.run_name,
                "train_dataset": dataset_name,
                "optimizer_steps": int(value(config, "trainer.epoch_len"))
                * int(value(config, "trainer.n_epochs")),
                "reference_roi_warp": expect_warp,
                "reference_frame_mode": expect_frame,
                "trainable_parameters": value(
                    config, "expected_trainable_contract.total_parameters"
                ),
                "inherited_fields_verified": len(INHERITED),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
