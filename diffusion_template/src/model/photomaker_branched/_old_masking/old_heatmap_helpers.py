from __future__ import annotations

from typing import Dict, List, Callable

import numpy as np
import torch
from PIL import Image

from .mask_utils import MASK_LAYERS_CONFIG, compute_binary_face_mask, simple_threshold_mask
from .heatmap_utils import build_hook_identity, build_hook_focus_token

__all__ = [
    "aggregate_heatmaps_to_mask",
    "collect_attention_hooks",
]


# ────────────────────────────────────────────────────────────────
#  helper: collapse heat-maps ➜ binary face mask
# ────────────────────────────────────────────────────────────────

def aggregate_heatmaps_to_mask(
    pipeline,
    mask_mode: str,
    import_mask: str | None,
    suffix: str = ""
) -> None:
    """
    Creates `pipeline._face_mask   # numpy  (H,W, bool)`
            `pipeline._face_mask_t # tensor (1,1,H,W)`
    Safe to call multiple times – it early-outs if the mask already exists.
    """

    # choose which attr to test/set
    mask_attr = f"_face_mask{suffix}"
    mask_t_attr = f"_face_mask_t{suffix}"
    if getattr(pipeline, mask_attr, None) is not None:
        return

    if import_mask is not None:
        from PIL import ImageOps
        _mask = Image.open(import_mask).convert("L")

        if suffix == "_ref" and hasattr(pipeline, "_ref_scaled_size") and hasattr(pipeline, "_ref_pad"):
            rh, rw = pipeline._ref_scaled_size          # (H', W') after AR-preserving scale
            pl, pr, pt, pb = pipeline._ref_pad          # stored in TORCH order: (L, R, T, B)

            # Store original high-res mask before scaling
            mask_np_highres = (np.array(_mask) > 127).astype(np.uint8)
            setattr(pipeline, f"_face_mask_highres{suffix}", mask_np_highres.astype(bool))

            # ALSO store the scaled size and padding for high-res mask usage later
            setattr(pipeline, f"_face_mask_scaled_size{suffix}", (rh, rw))
            setattr(pipeline, f"_face_mask_pad{suffix}", (pl, pr, pt, pb))

            # Use better interpolation for scaling
            _mask = _mask.resize((rw, rh), resample=Image.LANCZOS)

            # PIL expects (L, T, R, B) → reorder!
            _mask = ImageOps.expand(_mask, border=(pl, pt, pr, pb), fill=0)

        mask_np = (np.array(_mask) > 127).astype(np.uint8)

    else:
        # collapse per-step heat-maps → mean map / layer
        from .mask_utils import _resize_map
        snapshot = {}
        for ln, lst in pipeline._heatmaps.items():
            maps2d = [m for m in lst if m.ndim == 2]
            if not maps2d:
                continue
            max_H = max(m.shape[0] for m in maps2d)
            aligned = [m if m.shape[0] == max_H else _resize_map(m, max_H) for m in maps2d]
            snapshot[ln] = np.stack(aligned, 0).mean(0)

        mask_np = (
            simple_threshold_mask(snapshot)
            if mask_mode == "simple"
            else compute_binary_face_mask(snapshot, MASK_LAYERS_CONFIG)
        )

    # store under _face_mask vs. _face_mask_ref
    setattr(pipeline, mask_attr, mask_np.astype(bool))
    setattr(pipeline, mask_t_attr, torch.from_numpy(mask_np.astype(np.uint8)).unsqueeze(0).unsqueeze(0))

    # (Optional) quick sanity for ref: mask grid should match ref-latents grid×8
    if suffix == "_ref" and hasattr(pipeline, "_ref_latents_all"):
        hrl, wrl = pipeline._ref_latents_all.shape[-2:]
        H, W = mask_np.shape
        if (H, W) != (hrl * 8, wrl * 8):
            print(f"[WARN] ref mask {H}×{W} ≠ ref grid {hrl*8}×{wrl*8} (will be resized downstream)")

    # keep a latent-resolution copy for debug overlays
    pipeline._mask_lat_np = getattr(pipeline, mask_t_attr)[0, 0].float().cpu().numpy()
    pipeline._heatmaps.clear()


# ───────────────────────────────────────────────────────────────────
# helper: Attention-map forward-hooks (heat-map harvesting)
# ───────────────────────────────────────────────────────────────────

def collect_attention_hooks(
    pipeline,
    heatmap_mode: str,
    focus_token: str,
    class_tokens_mask: torch.Tensor | None,
    do_cfg: bool,
    attn_maps_current: Dict[str, List],
    orig_attn_forwards: Dict[str, Callable],
) -> int:
    """Register layer-specific forward hooks so external code can harvest
    raw attention logits and later build binary face masks.
    """
    from diffusers.models.attention_processor import Attention as CrossAttention

    wanted_layers = {spec["name"] for spec in MASK_LAYERS_CONFIG}
    if hasattr(pipeline.unet, "attn_processors"):
        # ensure raw PyTorch attention – disable xformers/Flash, etc.
        pipeline.unet.set_attn_processor(dict(pipeline.unet.attn_processors))

    # select hook builder --------------------------------------------------
    if heatmap_mode.lower() == "identity":
        def _builder(ln, mod):
            return build_hook_identity(
                ln, mod, wanted_layers, class_tokens_mask,
                pipeline.num_tokens, attn_maps_current, orig_attn_forwards,
                do_cfg,
            )
    else:  # "token"
        # one-shot build of aux prompt → focus_latents & token indices
        aux_prompt = f"a {focus_token}"
        focus_lat, *_ = pipeline.encode_prompt(
            prompt=aux_prompt, device=pipeline.device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        tok = pipeline.tokenizer or pipeline.tokenizer_2
        idsA = tok(aux_prompt, add_special_tokens=False).input_ids
        idsW = tok(" " + focus_token, add_special_tokens=False).input_ids

        def _find_sub(seq, sub):
            for i in range(len(seq) - len(sub) + 1):
                if seq[i: i + len(sub)] == sub:
                    return list(range(i, i + len(sub)))
            return []

        token_idx_global = _find_sub(idsA, idsW)
        if not token_idx_global:
            raise RuntimeError(f"focus token '{focus_token}' not found in '{aux_prompt}'")

        def _builder(ln, mod):
            return build_hook_focus_token(
                ln, mod, wanted_layers, focus_lat,
                token_idx_global, attn_maps_current,
                orig_attn_forwards, do_cfg,
            )

    # iterate & attach ------------------------------------------------------
    hooks = 0
    for ln, mod in pipeline.unet.named_modules():
        if isinstance(mod, CrossAttention) and ln in wanted_layers:
            mod.forward = _builder(ln, mod)
            hooks += 1
    return hooks
