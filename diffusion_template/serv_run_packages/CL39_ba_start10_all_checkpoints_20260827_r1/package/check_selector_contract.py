#!/usr/bin/env python3
"""Fail-closed schedule check for the accepted and BA-at-10 inference modes."""

from types import SimpleNamespace

import torch

from src.pipelines.br_pipeline_helpers import select_mode_and_prompts


def mode_at(index: int, pm_start: int, ba_start: int) -> str:
    pipeline = SimpleNamespace(
        branched_start_mode="both",
        _pose_user_ratio=0.0,
        pose_adapt_ratio=0.0,
    )
    text = torch.zeros(1)
    identity = torch.ones(1)
    mode, *_ = select_mode_and_prompts(
        pipeline,
        i=index,
        photomaker_start_step=pm_start,
        branched_attn_start_step=ba_start,
        prompt_embeds_text_only=text,
        pooled_prompt_embeds_text_only=text,
        prompt_embeds=identity,
        pooled_prompt_embeds=identity,
        force_par_before_pm=False,
        pose_forced_logged=False,
        pose_relaxed_logged=False,
    )
    return mode


expected = {
    (9, 10, 15): "NO_ID",
    (10, 10, 15): "PHOTOMAKER",
    (14, 10, 15): "PHOTOMAKER",
    (15, 10, 15): "BOTH",
    (9, 10, 10): "NO_ID",
    (10, 10, 10): "BOTH",
    (49, 10, 10): "BOTH",
}
observed = {contract: mode_at(*contract) for contract in expected}
if observed != expected:
    raise SystemExit(f"selector contract failed: expected={expected}, observed={observed}")
print(f"selector contract passed: {observed}")
