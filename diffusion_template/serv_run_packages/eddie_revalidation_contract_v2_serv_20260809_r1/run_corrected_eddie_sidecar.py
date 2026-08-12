#!/usr/bin/env python3
"""Replay the 12 Eddie rows under the exact in-training validation contract.

The corrected arm replaces the ArcFace vector fused into PhotoMaker's prompt
tokens. That is a *global identity-conditioning intervention*, not a face-local
BA-mask edit, so it may alter composition outside the target face. Use the
historical arm first as a reproduction gate before interpreting a corrected
arm. This script prepares the runs; it does not compare their images.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import importlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch


ASSET_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSET_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to the experiment's validation dataloader batch size.",
    )
    parser.add_argument(
        "--identity-condition",
        choices=["historical", "corrected"],
        default="corrected",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evaluator-path",
        type=Path,
        help=(
            "Optional patched evaluator overlay. This keeps immutable training "
            "runtime trees unmodified while importing their exact model code."
        ),
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=DATA_DIR / "eddie_foreground_arcface_embedding.npy",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if args.evaluator_path is None:
        evaluator = importlib.import_module("tools.inference.evaluate_rhca_checkpoint")
    else:
        evaluator_path = args.evaluator_path.resolve()
        os.environ["PM_EVAL_PROJECT_ROOT"] = str(project_root)
        spec = importlib.util.spec_from_file_location(
            "eddie_contract_v2_evaluator", evaluator_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load evaluator overlay: {evaluator_path}")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)

    embedding_path = args.embedding.resolve()
    embedding = torch.from_numpy(np.load(embedding_path)).float()
    if embedding.shape != (512,):
        raise ValueError(f"Unexpected Eddie embedding shape: {tuple(embedding.shape)}")

    original_apply = evaluator.apply_reference_condition

    if args.identity_condition == "corrected":
        def corrected_apply(eval_args, samples):
            refs, bboxes, _legacy_embeds, intervention = original_apply(
                eval_args, samples
            )
            identities = {str(sample["id"]).lower() for sample in samples}
            if identities != {"eddie"} or len(samples) != 12:
                raise ValueError(
                    "Corrected Eddie sidecar requires exactly the first 12 "
                    "manual_val rows"
                )
            corrected = embedding.unsqueeze(0).repeat(len(samples), 1)
            intervention = dict(intervention)
            intervention.update(
                {
                    "id_embedding_selection": "largest/intended foreground face",
                    "historical_selection_replaced": (
                        "detector result [0] background face"
                    ),
                    "embedding_path": str(embedding_path.resolve()),
                    "changed_input": (
                        "ArcFace vector fused into global PhotoMaker prompt tokens"
                    ),
                    "face_local_ba_only": False,
                    "composition_may_change": True,
                    "unchanged_inputs": (
                        "reference pixels, reference bbox, BA reference crop, prompts, "
                        "seeds, generation bboxes, scheduler, steps, CFG, and checkpoint"
                    ),
                }
            )
            return refs, bboxes, corrected, intervention

        evaluator.apply_reference_condition = corrected_apply
    run_args = Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        validation_dataset="manual_val",
        guidance_scale=None,
        disable_branched_ca=None,
        validation_base=None,
        photomaker_path=None,
        processor_base_mode=None,
        reference_condition="matched",
        spatial_reference_condition="matched",
        ba_mix_override=None,
        limit=12,
        batch_size=args.batch_size,
        checkpoint_step=args.checkpoint_step,
        wrong_reference=None,
        wrong_reference_bbox=None,
        device="cuda",
        skip_metrics=True,
        allow_untrained_ca=False,
        allow_validation_contract_override=False,
    )
    evaluator.run(run_args)


if __name__ == "__main__":
    main()
