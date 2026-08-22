"""Fixed E13 specialization of the unchanged 2 June attention processor."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .attn_processor_cleanest import BranchedAttnProcessor


class E13BranchedAttnProcessor(BranchedAttnProcessor):
    """Pin E13's rank-128 branch weights and cache resized masks."""

    def __init__(self, hidden_size: int, cross_attention_dim: int, scale: float):
        # 22 Aug 2026 - E13 keeps the 2 June equations and changes only the
        # branch ownership: independent rank-128 LoRA Q/K/V for both lanes.
        super().__init__(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            scale=scale,
            branched_attn_weight_mode="noise_and_ref",
            branched_attn_new_weight_kind="lora",
            branched_attn_lora_rank=128,
        )

    def _prepare_mask(
        self, mask: torch.Tensor, target_len: int, batch_size: int
    ) -> torch.Tensor:
        # AICODE-NOTE: The cache is attached to the exact input tensor, so it
        # cannot leak a resized mask across samples or validation calls.
        cache_key = (
            int(target_len),
            int(batch_size),
            str(mask.device),
            str(mask.dtype),
            bool(getattr(self, "force_binary_masks", False)),
        )
        cache = getattr(mask, "_ba_prepared_mask_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        side = math.isqrt(target_len)
        if side * side != target_len:
            raise RuntimeError(f"Sequence length {target_len} is not square")
        source_batch = mask.shape[0]
        if mask.ndim == 4:
            mask_4d = mask[:, :1].float()
        else:
            flat = mask.reshape(source_batch, -1).float()
            source_side = math.isqrt(flat.shape[1])
            if source_side * source_side != flat.shape[1]:
                raise RuntimeError(f"Mask length {flat.shape[1]} is not square")
            mask_4d = flat.reshape(source_batch, 1, source_side, source_side)

        resized = F.interpolate(
            mask_4d, size=(side, side), mode="bilinear", align_corners=False
        )
        if getattr(self, "force_binary_masks", False):
            resized = (resized > 0.5).to(resized.dtype)
        flattened = resized.flatten(2).transpose(1, 2)
        if flattened.shape[0] != batch_size:
            repeats = (batch_size + flattened.shape[0] - 1) // flattened.shape[0]
            flattened = flattened.repeat(repeats, 1, 1)[:batch_size]
        result = flattened.view(batch_size, 1, target_len, 1)

        if cache is None:
            cache = {}
            mask._ba_prepared_mask_cache = cache
        cache[cache_key] = result
        return result
