#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DATA-03/04: Retained the sealed CL-arm preflight; CL14
# must realize its 1024 reference canvas, 6%-30% scale band, and prompt policy.
"""Fail-closed dataset preflight for the CL1-CL3 Cosmic Large arms.

Decodes a deterministic sample through the real configured loader and asserts
the property each arm exists to establish: the reference/target face-scale
relationship on the shared 1024px latent frame, plus the shared caption and
leakage controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "configs"
# Reference-face area band observed on large_dataset targets (7.32% median).
SCENE_REF_AREA_BAND = (2.0, 22.0)
MAX_TRUNCATED_PROMPT_FRACTION = 0.05


def short_side(bbox) -> float:
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def letterbox_scale(size, target_size: int = 1024) -> float:
    """The factor `_encode_reference_latent` applies before the VAE.

    # 06 Aug 2026 - The reference is letterboxed into a `target_size` square
    # before encoding, so a 256px reference face is enlarged 4x on the shared
    # latent frame. Comparing raw bbox short sides across differently sized
    # references understates the mismatch by exactly that factor.
    """
    width, height = size
    return min(target_size / float(width), target_size / float(height))


def area_fraction(bbox, size) -> float:
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return width * height / (size[0] * size[1]) * 100.0


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prompt-sample-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=args.config_name)
    dataset_name = str(config.train_dataset_name)
    dataset_config = OmegaConf.to_container(
        config.datasets.train[dataset_name], resolve=True
    )
    dataset_config.pop("instance_transforms", None)
    dataset = instantiate(dataset_config, _convert_="all")

    rng = random.Random(args.seed)
    total = len(dataset)
    picks = sorted(rng.sample(range(total), min(args.sample_count, total)))

    ratios, ref_areas, target_areas, center_offsets = [], [], [], []
    ref_sizes, failures = set(), []
    for index in picks:
        sample = dataset[index]
        reference = sample["ref_images"][0]
        ref_bbox, target_bbox = sample["face_bbox_ref"], sample["face_bbox"]
        ref_sizes.add(reference.size)
        # Measure in the shared 1024 latent frame, not in each image's own frame.
        ratios.append(
            short_side(ref_bbox)
            * letterbox_scale(reference.size)
            / short_side(target_bbox)
        )
        ref_areas.append(area_fraction(ref_bbox, reference.size))
        target_areas.append(area_fraction(target_bbox, (1024, 1024)))
        rc = ((ref_bbox[0] + ref_bbox[2]) / 2, (ref_bbox[1] + ref_bbox[3]) / 2)
        tc = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)
        center_offsets.append(max(abs(rc[0] - tc[0]), abs(rc[1] - tc[1])))
        if sample["target_path"] == sample["reference_path"]:
            failures.append(f"target/reference leakage at index {index}")

    # Caption budget: the CLIP tokenizer truncates at 77 before PhotoMaker
    # expands the class token, so long Cosmic captions silently lose pose and
    # background supervision.
    truncated = None
    try:
        from transformers import CLIPTokenizer

        from src.datasets.cosmic_large_adapted import build_cosmic_prompt

        base_model = OmegaConf.select(
            config, "model.pretrained_model_name_or_path"
        ) or "stabilityai/stable-diffusion-xl-base-1.0"
        tokenizer = CLIPTokenizer.from_pretrained(str(base_model), subfolder="tokenizer")
        prompt_picks = sorted(rng.sample(range(total), min(args.prompt_sample_count, total)))
        # Build prompts directly from the index so this stays image-decode free.
        lengths = [
            len(
                tokenizer(
                    build_cosmic_prompt(
                        dataset._index[i],
                        dataset.prompt_mode,
                        dataset.prompt_max_words,
                    ),
                    truncation=False,
                ).input_ids
            )
            for i in prompt_picks
        ]
        truncated = sum(1 for n in lengths if n > 77) / len(lengths)
    except Exception as error:  # tokenizer or network unavailable offline
        truncated = None
        print(f"[warn] caption token check skipped: {error}", file=sys.stderr)

    report = {
        "config_name": args.config_name,
        "train_dataset": dataset_name,
        "records": total,
        "sampled": len(picks),
        "reference_sizes": sorted(str(size) for size in ref_sizes),
        "face_scale_ratio_note": "reference/target face short side, both in the shared 1024 latent frame",
        "face_scale_ratio": {
            "p10": percentile(ratios, 0.10),
            "median": percentile(ratios, 0.50),
            "p90": percentile(ratios, 0.90),
        },
        "reference_face_area_pct": {
            "p10": percentile(ref_areas, 0.10),
            "median": percentile(ref_areas, 0.50),
            "p90": percentile(ref_areas, 0.90),
        },
        "target_face_area_pct_median": percentile(target_areas, 0.50),
        "center_offset_px_max": max(center_offsets) if center_offsets else 0.0,
        "prompt_over_77_tokens_fraction": truncated,
        "dataset_audit": getattr(dataset, "audit", None),
    }

    median_ratio = report["face_scale_ratio"]["median"]
    # 09 Aug 2026 - Match the exact arm token. `startswith("CL1")` is also true
    # for CL10 and CL11, which routed CL10 into CL1's reference-band gate (it
    # failed on a rule that does not apply) and let CL11 pass CL1's gate by luck.
    arm = args.config_name.split("_", 1)[0].upper()
    report["arm"] = arm
    if arm == "CL1":
        if ref_sizes != {(1024, 1024)}:
            failures.append(f"CL1 references must be native 1024 scenes, saw {ref_sizes}")
        median_ref_area = report["reference_face_area_pct"]["median"]
        if not SCENE_REF_AREA_BAND[0] <= median_ref_area <= SCENE_REF_AREA_BAND[1]:
            failures.append(
                f"CL1 reference face area {median_ref_area:.2f}% outside the "
                f"large_dataset band {SCENE_REF_AREA_BAND}"
            )
    elif arm in ("CL9", "CL10", "CL11", "CL12", "CL13", "CL14"):
        if ref_sizes != {(1024, 1024)}:
            failures.append(f"{arm} canvases must be 1024x1024, saw {ref_sizes}")
        med = report["reference_face_area_pct"]["median"] / 100.0
        upper = 0.40 if arm in ("CL10", "CL12") else 0.30
        if not 0.06 <= med <= upper:
            failures.append(
                f"{arm} reference face fraction median {med:.3f} outside the "
                f"configured jitter range [0.06, {upper:.2f}]"
            )
    elif arm == "CL2":
        if ref_sizes != {(1024, 1024)}:
            failures.append(f"CL2 canvases must be 1024x1024, saw {ref_sizes}")
        out_of_band = sum(1 for r in ratios if not 0.95 <= r <= 1.05)
        if out_of_band > 0.05 * len(ratios):
            failures.append(
                f"CL2 face-scale ratio out of [0.95, 1.05] for {out_of_band}/{len(ratios)}"
            )
        if report["center_offset_px_max"] > 4.0:
            failures.append(
                f"CL2 face centres differ by up to {report['center_offset_px_max']:.1f}px"
            )
    elif arm in ("CL0", "CL3", "CL4", "CL5", "CL6", "CL7", "CL8"):
        # These arms all keep the native 256px asset and its uncorrected
        # pixel-space mismatch. CL3/CL5 then correct it in feature space via
        # ba_hard_v1_reference_roi_warp; CL0/CL4 deliberately do not correct it.
        if ref_sizes != {(256, 256)}:
            failures.append(
                f"{arm} must keep the native 256px asset, saw {ref_sizes}"
            )
        if median_ratio <= 1.5:
            failures.append(
                f"{arm} expects the uncorrected mismatch on the shared "
                f"1024 frame, median ratio {median_ratio:.2f}"
            )
    else:
        failures.append(f"Unknown CL arm: {args.config_name}")

    # CL8 must actually have restored the full-body target distribution.
    if arm in ("CL8", "CL10", "CL12"):
        small = sum(1 for a in target_areas if a < 5.0) / max(len(target_areas), 1)
        report["targets_below_5pct_face"] = round(small, 4)
        report["accepted_records"] = total
        if small < 0.20:
            failures.append(
                f"CL8 expects >=20% of sampled targets below 5% face area, saw {small:.1%}"
            )

    # CL5 must emit the extra identity references while leaving the spatial lane alone.
    if arm in ("CL5", "CL11", "CL12"):
        expected_refs = int(dataset_config.get("num_identity_refs", 1) or 1)
        counts = {len(dataset[i]["ref_images"]) for i in picks[:8]}
        report["reference_images_per_sample"] = sorted(counts)
        if counts != {expected_refs}:
            failures.append(
                f"CL5 expects {expected_refs} reference images per sample, saw {sorted(counts)}"
            )

    # CL0 is the deliberately unimproved baseline: uncapped legacy captions are
    # its defining property, so the truncation gate must not apply to it.
    if arm == "CL0":
        report["caption_gate"] = "waived (CL0 baseline preserves uncapped legacy captions)"
    elif truncated is not None and truncated > MAX_TRUNCATED_PROMPT_FRACTION:
        failures.append(
            f"{truncated:.1%} of prompts exceed 77 CLIP tokens (limit "
            f"{MAX_TRUNCATED_PROMPT_FRACTION:.0%}); check prompt_max_words"
        )

    report["status"] = "ok" if not failures else "failed"
    report["failures"] = failures
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
