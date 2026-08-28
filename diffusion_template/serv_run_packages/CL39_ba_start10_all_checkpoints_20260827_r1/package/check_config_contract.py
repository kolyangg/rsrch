#!/usr/bin/env python3
"""Verify that the multi-checkpoint config changes only validation orchestration."""

import argparse
from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ALLOWED_PATHS = (
    "validation_only",
    "validation_epochs",
    "validation_checkpoint_paths",
    "validation_args.branched_attn_start_step",
    "trainer.from_pretrained",
    "trainer.save_dir",
    "trainer.face_quality.expected_images",
    "writer",
    "validation_debug_timing",
)


def drop_path(value, dotted: str) -> None:
    parts = dotted.split(".")
    node = value
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        parent = compose(config_name="CL39_cosmic_null_key_confidence_router_24k")
        candidate = compose(config_name="CL39_ba_start10_all_checkpoints_full96")

    parent_value = OmegaConf.to_container(parent, resolve=True)
    candidate_value = OmegaConf.to_container(candidate, resolve=True)
    parent_cmp, candidate_cmp = deepcopy(parent_value), deepcopy(candidate_value)
    for path in ALLOWED_PATHS:
        drop_path(parent_cmp, path)
        drop_path(candidate_cmp, path)
    if parent_cmp != candidate_cmp:
        raise SystemExit("Candidate differs from CL39 outside the allowed validation paths")

    epochs = list(candidate.validation_epochs)
    paths = list(candidate.validation_checkpoint_paths)
    expected_epochs = list(range(13))
    if epochs != expected_epochs or len(paths) != 13 or paths[0] is not None:
        raise SystemExit(f"Invalid checkpoint schedule: epochs={epochs}, paths={paths}")
    for epoch, path in zip(epochs[1:], paths[1:]):
        checkpoint = Path(str(path))
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise SystemExit(f"Missing checkpoint for epoch {epoch}: {checkpoint}")

    assertions = {
        "validation_only": bool(candidate.validation_only) is True,
        "epoch_len": int(candidate.trainer.epoch_len) == 2000,
        "model_pm_start": int(candidate.model.photomaker_start_step) == 10,
        "model_ba_start": int(candidate.model.branched_attn_start_step) == 15,
        "validation_pm_start": int(candidate.validation_args.photomaker_start_step) == 10,
        "validation_ba_start": int(candidate.validation_args.branched_attn_start_step) == 10,
        "pose_adapt_ratio": float(candidate.pipeline.pose_adapt_ratio) == 0.0,
        "ca_mixing_for_face": bool(candidate.pipeline.ca_mixing_for_face) is False,
        "fixed_manual_val": list(candidate.val_datasets_names) == ["manual_val"],
        "comet_writer": str(candidate.writer._target_) == "src.logger.cometml.CometMLWriter",
        "comet_project": str(candidate.writer.project_name) == "aug-large-ds",
        "fresh_comet": OmegaConf.select(candidate, "cometml_id") in (None, ""),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise SystemExit(f"Config contract failed: {failed}")
    print(
        "CL39_BA10_ALL_CONFIG_OK checkpoints=13 steps=0:2000:24000 "
        "images_per_step=96 single_comet=true"
    )


if __name__ == "__main__":
    main()
