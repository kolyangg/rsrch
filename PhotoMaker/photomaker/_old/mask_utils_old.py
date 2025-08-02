import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
#  Default mask‑layer specification copied from `attn_hm_NS_nosm7.py`
# -----------------------------------------------------------------------------
MASK_LAYERS_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "up_blocks.0.attentions.1.transformer_blocks.1.attn2",
        "weight": 0.25,
        "top_ratio": 0.10,
        "invert": False,
    },
    {
        "name": "up_blocks.0.attentions.1.transformer_blocks.7.attn2",
        "weight": 0.75,
        "top_ratio": 0.05,
        "invert": True,
    },
]


# -----------------------------------------------------------------------------
#  Helper to build **binary** face mask from raw attention maps
# -----------------------------------------------------------------------------

def _resize_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Bilinear‑resize *square* numpy array to `size`×`size`."""
    if arr.shape[0] == size:
        return arr
    return np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def compute_binary_face_mask(
    attn_maps: Dict[str, np.ndarray],
    mask_layers: List[Dict[str, Any]] | None = None,
) -> np.ndarray:
    """Return **bool H×H** mask (face=True) following the logic of
    `attn_hm_NS_nosm7.py`.

    Parameters
    ----------
    attn_maps : dict[str -> ndarray]
        Each value is a **square** attention map (already mean‑pooled over heads)
        in **float32**.
    mask_layers : list[dict] | None
        Layer spec in the same format as the JSON snippet used by the original
        script.  If *None* the default `MASK_LAYERS_CONFIG` is used.
    """
    if mask_layers is None:
        mask_layers = MASK_LAYERS_CONFIG

    # -------- choose the *largest* resolution among selected layers ----------
    base_H = max(
        attn_maps[sp["name"]].shape[0]
        for sp in mask_layers if sp["name"] in attn_maps
    )

    combined = np.zeros((base_H, base_H), dtype=np.float32)
    total_w  = 0.0

    for spec in mask_layers:
        lname = spec["name"]
        if lname not in attn_maps:
            continue

        amap = attn_maps[lname]
        amap = _resize_map(amap, base_H)

        # normalise 0…1 per‑layer
        amap_n = amap / amap.max() if amap.max() > 0 else amap

        top_ratio = spec.get("top_ratio", 0.10)
        invert    = spec.get("invert", False)
        weight    = spec.get("weight", 1.0)

        if invert:
            thr = np.quantile(amap_n, top_ratio)          # keep the *lowest* xx %
            sel = amap_n < thr
        else:
            thr = np.quantile(amap_n, 1.0 - top_ratio)    # keep the *highest* xx %
            sel = amap_n > thr

        combined += weight * sel.astype(np.float32)
        total_w  += weight

    if total_w == 0.0:
        return np.zeros_like(combined, dtype=bool)

    # average & binarise -------------------------------------------------------
    mask = (combined / total_w) > 0.0

    # ---- post‑process: keep largest blob & convex‑hull clean‑up --------------
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_lbl > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl == largest)

    area_orig = mask.sum()
    if area_orig == 0:
        return mask.astype(bool)

    # build convex hull around the blob ---------------------------------------
    cnt,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull  = cv2.convexHull(cnt[0]) if cnt else None
    if hull is not None:
        mask_hull = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(mask_hull, [hull], -1, 1, -1)
        mask = mask_hull.astype(bool)

    # optional shrink if hull grew too much (keep ≤100 % of original area) -----
    HULL_MAX_AREA_FACTOR = 1.0
    if mask.sum() > HULL_MAX_AREA_FACTOR * area_orig:
        k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        while mask.sum() > HULL_MAX_AREA_FACTOR * area_orig and mask.any():
            mask = cv2.erode(mask.astype(np.uint8), k_er, 1).astype(bool)

    return mask.astype(bool)
