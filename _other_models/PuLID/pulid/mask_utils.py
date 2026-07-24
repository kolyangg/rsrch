# mask_utils.py
"""
Mask helpers shared between the stand-alone heat-map script
(`attn_hm_NS_nosm7.py`) and the integrated SD-XL PhotoMaker pipeline.

🚩  This file **re-implements verbatim** the mask algorithm that the
     reference script uses – including invert-logic, weighted union,
     convex-hull closure and hull-area limiter.  Keeping it here avoids
     accidental divergences when we update the pipeline.
"""

from __future__ import annotations

import math, json, warnings, os
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

# -------------------------------------------------------------------- #
#  ❶  Default layer spec  (= the JSON you attached)
# -------------------------------------------------------------------- #
# MASK_LAYERS_CONFIG: List[Dict] = [
#     {   # 1st entry
#         "name"      : "up_blocks.0.attentions.1.transformer_blocks.1.attn2",
#         "weight"    : 0.25, # 0.0, # 0.25,
#         "top_ratio" : 0.10,
#         "invert"    : False,
#     },
#     {   # 2nd entry
#         "name"      : "up_blocks.0.attentions.1.transformer_blocks.7.attn2",
#         "weight"    : 0.75, # 1.0, # 0.75
#         "top_ratio" : 0.05,
#         "invert"    : True,
#     },
# ]


MASK_LAYERS_CONFIG: List[Dict] = [
    {   # one entry
        "name"      : "up_blocks.0.attentions.1.transformer_blocks.7.attn2",
        "weight"    : 1.0, # 1.0, # 0.75
        "top_ratio" : 0.10,
        "invert"    : False,
    },
]


# -------------------------------------------------------------------- #
#  ❷  Small helpers
# -------------------------------------------------------------------- #
def _resize_map(arr: np.ndarray, tgt_H: int) -> np.ndarray:
    """Bilinear resize *square* 2-D NumPy array to tgt_H×tgt_H."""
    if arr.shape[0] == tgt_H:
        return arr
    return np.array(
        Image.fromarray(arr).resize((tgt_H, tgt_H), Image.BILINEAR)
    )


# -------------------------------------------------------------------- #
#  ❸  Public API – *identical* to attn_hm_NS_nosm7.py behaviour
# -------------------------------------------------------------------- #
def compute_binary_face_mask(
    snapshot: Dict[str, np.ndarray],
    layers_spec: List[Dict] | None = None,
) -> np.ndarray:
    """
    Build the binary face-mask from a dict of attention maps.

    Parameters
    ----------
    snapshot
        ``dict[layer_name] → 2-D np.ndarray``; *each* map is the
        **mean-pooled, un-normalised logits** (not soft-maxed).
    layers_spec
        List of dicts as in the JSON; if *None* – fall back to
        ``MASK_LAYERS_CONFIG`` above.

    Returns
    -------
    mask : np.ndarray[bool]   shape **H×H** (largest resolution among layers)
    """

    if layers_spec is None:
        layers_spec = MASK_LAYERS_CONFIG

    # ❶ make sure every entry has the legacy defaults
    for spec in layers_spec:
        spec.setdefault("weight", 1.0)
        spec.setdefault("top_ratio", 0.10)
        spec.setdefault("invert", False)

    # ❷ choose target resolution: largest among selected layers present
    base_H = max(
        snapshot[sp["name"]].shape[0]
        for sp in layers_spec
        if sp["name"] in snapshot
    )

    combined = np.zeros((base_H, base_H), np.float32)
    total_w  = 0.0

    for spec in layers_spec:
        lname = spec["name"]
        if lname not in snapshot:
            warnings.warn(f"[Mask] layer '{lname}' not in snapshot – skipped")
            continue

        amap = _resize_map(snapshot[lname], base_H)
        amap_n = amap / amap.max() if amap.max() > 0 else amap

        if spec["invert"]:
            thr = np.quantile(amap_n, spec["top_ratio"])
            sel = amap_n < thr
        else:
            thr = np.quantile(amap_n, 1.0 - spec["top_ratio"])
            sel = amap_n > thr

        combined += spec["weight"] * sel.astype(np.float32)
        total_w  += spec["weight"]

    mask = combined / max(total_w, 1e-6) > 0.0

    # # ❸ keep **largest** connected blob
    # n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
    #     mask.astype(np.uint8), connectivity=8)
    # if n_lbl > 1:
    #     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #     mask = lbl == largest

    # # ❹ convex-hull & area limiter (1 × original by default – can tune)
    # main = mask.astype(np.uint8)
    # cnt, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if cnt:
    #     hull = cv2.convexHull(cnt[0])
    #     cv2.drawContours(main, [hull], -1, 1, -1)

    #     HULL_MAX_AREA_FACTOR = 1.0
    #     area_orig = mask.sum()
    #     k_erode   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     while main.sum() > HULL_MAX_AREA_FACTOR * area_orig:
    #         main = cv2.erode(main, k_erode, 1)
    #         if main.sum() == 0:
    #             main = mask.astype(np.uint8)
    #             break

    #     mask = main.astype(bool)


    # ❸ keep **largest** connected blob
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n_lbl > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == largest)

    # # ❹ convex-hull & area limiter — cloned verbatim from
    # #    `attn_hm_NS_nosm7.py` :contentReference[oaicite:3]{index=3}
    # main_blob = mask.astype(np.uint8)
    # cnt, _    = cv2.findContours(main_blob, cv2.RETR_EXTERNAL,
    #                              cv2.CHAIN_APPROX_SIMPLE)
    # if cnt:
    #     hull = cv2.convexHull(cnt[0])

    #     # draw hull on a *fresh* canvas (prevents hole-filling side-effect)
    #     main = np.zeros_like(main_blob, dtype=np.uint8)
    #     cv2.drawContours(main, [hull], -1, 1, -1)

    #     HULL_MAX_AREA_FACTOR = 1.0            # no area growth allowed
    #     area_orig = main_blob.sum()           # area before hull expansion
    #     k_erode   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     while main.sum() > HULL_MAX_AREA_FACTOR * area_orig:
    #         main = cv2.erode(main, k_erode, 1)
    #         if main.sum() == 0:               # safety fall-back
    #             main = main_blob
    #             break

    #     mask = main.astype(bool)


    # ❹ convex-hull *exactly* like `attn_hm_NS_nosm7.py`
    main_blob = mask.astype(np.uint8)
    cnt, _    = cv2.findContours(main_blob, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        hull  = cv2.convexHull(cnt[0])
        mask  = np.zeros_like(main_blob)            # fresh canvas
        cv2.drawContours(mask, [hull], -1, 1, -1)   # filled hull

        # — limit over-fill (same constants as the script) —
        HULL_MAX_AREA_FACTOR = 1.0                  # no growth allowed
        area_orig = main_blob.sum()
        k_erode   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        while mask.sum() > HULL_MAX_AREA_FACTOR * area_orig:
            mask = cv2.erode(mask, k_erode, 1)
            if not mask.any():                      # safety fall-back
                mask = main_blob.copy()
                break

        mask = mask.astype(bool)


    return mask


def simple_threshold_mask(snapshot: Dict[str, np.ndarray],
                          layer_name: str | None = None,
                          top_ratio: float = 0.10) -> np.ndarray:
    """
    Very small mask maker: take **one** layer (default = first in snapshot)
    and keep the top `top_ratio` fraction of its *normalised* logits.
    """
    if not snapshot:
        raise ValueError("empty snapshot – cannot build mask")

    if layer_name is None or layer_name not in snapshot:
        layer_name = next(iter(snapshot))     # first layer

    amap = snapshot[layer_name]
    amap = amap / amap.max() if amap.max() > 0 else amap
    thr  = np.quantile(amap, 1.0 - top_ratio)
    return (amap > thr)