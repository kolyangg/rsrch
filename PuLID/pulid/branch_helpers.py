

from __future__ import annotations

import math, types, torch, numpy as np, os
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
from typing import Tuple, Dict, List, Callable

# Re‑export commonly used mask utilities so caller does *one* import.
from .mask_utils import (
    MASK_LAYERS_CONFIG,
    compute_binary_face_mask,
    _resize_map,
    simple_threshold_mask,
)

# Forward‑hook builders already live in heatmap_utils – just re‑export.
from .heatmap_utils import build_hook_identity, build_hook_focus_token

__all__ = [
    "aggregate_heatmaps_to_mask",
    "prepare_mask4",
    "save_branch_previews",
    "debug_reference_latents_once",
    "save_debug_ref_latents",
    "collect_attention_hooks",
]



# ────────────────────────────────────────────────────────────────
#  NEW  helper 0:  collapse heat-maps ➜ binary face mask
# ────────────────────────────────────────────────────────────────

def aggregate_heatmaps_to_mask(
    pipeline,
    mask_mode: str,
    import_mask: str | None,
    suffix: str = "" 
) -> None:
    """
    Creates `pipeline._face_mask   # numpy  (H,W, bool)`
            pipeline._face_mask_t # tensor (1,1,H,W)`
    The heavy logic was copied verbatim from the old pipeline block.
    Safe to call multiple times – it early-outs if the mask already exists.
    """

    # choose which attr to test/set
    mask_attr   = f"_face_mask{suffix}"
    mask_t_attr = f"_face_mask_t{suffix}"
    if getattr(pipeline, mask_attr, None) is not None:
        return

    # if import_mask is not None:
    #     _mask = Image.open(import_mask).convert("L")
    #     mask_np = (np.array(_mask) > 127).astype(np.uint8)
    if import_mask is not None:
        from PIL import ImageOps
        _mask = Image.open(import_mask).convert("L")
        # For the *reference* mask, apply the SAME resize + letterbox as the ref image
        # if suffix == "_ref" and hasattr(pipeline, "_ref_scaled_size") and hasattr(pipeline, "_ref_pad"):
        #     rh, rw = pipeline._ref_scaled_size   # scaled face size (H', W')
        #     pl, pr, pt, pb = pipeline._ref_pad   # letterbox paddings to (H, W)
        #     _mask = _mask.resize((rw, rh), resample=Image.NEAREST)
        #     _mask = ImageOps.expand(_mask, border=(pl, pr, pt, pb), fill=0)
        
        if suffix == "_ref" and hasattr(pipeline, "_ref_scaled_size") and hasattr(pipeline, "_ref_pad"):
            rh, rw = pipeline._ref_scaled_size          # (H', W') after AR-preserving scale
            pl, pr, pt, pb = pipeline._ref_pad          # stored in TORCH order: (L, R, T, B)
            # _mask = _mask.resize((rw, rh), resample=Image.NEAREST)
            
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
    setattr(pipeline, mask_attr,   mask_np.astype(bool))
    # setattr(pipeline, mask_t_attr, torch.from_numpy(mask_np.astype(np.uint8))
    #                                           .unsqueeze(0).unsqueeze(0))
    
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




# ────────────────────────────────────────────────────────────────
#  NEW  helper 1:  face-mask tensor @ current latent resolution
# ────────────────────────────────────────────────────────────────

def prepare_mask4(pipeline, latents: torch.Tensor, suffix) -> torch.Tensor:
    """Return `(1,1,H,W)` tensor mask matching *latents* spatial size."""
    
    
    # Use high-res mask for reference if available
    if suffix == "_ref" and hasattr(pipeline, f"_face_mask_highres{suffix}"):
        mask_np_highres = getattr(pipeline, f"_face_mask_highres{suffix}")
        m = torch.from_numpy(mask_np_highres).to(device=latents.device, dtype=torch.float32)[None, None]
        # Use bicubic for smoother downsampling
        m = F.interpolate(m, size=latents.shape[-2:], mode="bicubic", align_corners=False)
        m = (m > 0.5).to(dtype=latents.dtype)  # Re-binarize
        return m
    
    # pick which numpy mask to use
    mask_attr = f"_face_mask{suffix}"
    mask_np   = getattr(pipeline, mask_attr)

    # ### V1 ###
    # m = (
    #     torch.from_numpy(mask_np)
    #     .to(device=latents.device, dtype=latents.dtype)
    #     .unsqueeze(0)
    #     .unsqueeze(0)
    #     if isinstance(pipeline._face_mask, np.ndarray)
    #     else pipeline._face_mask[:, None].to(dtype=latents.dtype)
    # )
    # ### V1 ###

    ### V2 ###
    is_np = isinstance(mask_np, np.ndarray)
    m = (
        torch.from_numpy(mask_np).to(device=latents.device, dtype=latents.dtype)[None, None]
        if is_np else getattr(pipeline, mask_attr)[:, None].to(dtype=latents.dtype)
    )
    ### V2 ###

    if m.shape[-2:] != latents.shape[-2:]:
        m = F.interpolate(m, size=latents.shape[-2:], mode="nearest")
    return m


# def prepare_mask4(pipeline, latents: torch.Tensor, suffix: str = "") -> torch.Tensor:
#     """
#     Return a 1×1×H×W float mask aligned to `latents`.
#     Prefer the already-tensorized mask `_face_mask_t{suffix}` if it exists to
#     avoid any re-quantization; otherwise fall back to `_face_mask{suffix}`.
#     """
#     import numpy as np, torch
#     import torch.nn.functional as F

#     t_name = f"_face_mask_t{suffix}"
#     n_name = f"_face_mask{suffix}"

#     m = getattr(pipeline, t_name, None)
#     if m is not None:
#         if m.dim() == 3:          # (1,H,W) -> (1,1,H,W)
#            m = m[:, None]
#         m = m.float()
#     else:
#         mask_np = getattr(pipeline, n_name, None)
#         if mask_np is None:
#             return None
#         if mask_np.dtype != np.float32:
#             mask_np = (mask_np > 0).astype(np.float32)
#         m = torch.from_numpy(mask_np)[None, None]

#     m = m.to(device=latents.device, dtype=latents.dtype)
#     if m.shape[-2:] != latents.shape[-2:]:
#        m = F.interpolate(m, size=latents.shape[-2:], mode="nearest")
#     return m

# ────────────────────────────────────────────────────────────────
#  NEW  helper 3:  per-step preview PNGs
# ────────────────────────────────────────────────────────────────


### NEW VERSION ###
def save_branch_previews(
    pipeline,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,  # Changed from noise_face, noise_bg
    mask4: torch.Tensor,
    t: torch.Tensor,
    step_idx: int,
    debug_dir: str,
    extra_step_kwargs: dict,
) -> None:
    """Save preview of merged prediction with mask overlay."""
    
    if mask4 is None or noise_pred is None:
        return
        
    os.makedirs(debug_dir, exist_ok=True)
    
    # # Step the prediction
    # saved_idx = getattr(pipeline.scheduler, "_step_index", None)
    # lat_next = pipeline.scheduler.step(
    #     noise_pred, t, latents.detach().clone(),
    #     **extra_step_kwargs, return_dict=False
    # )[0]
    # if saved_idx is not None:
    #     pipeline.scheduler._step_index = saved_idx

    # Step the prediction (align CFG × NIMG batch sizes)
    saved_idx = getattr(pipeline.scheduler, "_step_index", None)
    B_lat = latents.shape[0]
    B_pred = noise_pred.shape[0]
    # If UNet output is [uncond, cond] concatenation, keep the conditional half
    if B_pred == 2 * B_lat:
        noise_for_step = noise_pred.chunk(2)[1]
    elif B_pred == B_lat:
        noise_for_step = noise_pred
    else:
        # Fallback: tile/trim to match latents’ batch B_lat
        rep = (B_lat + B_pred - 1) // B_pred
        noise_for_step = noise_pred.repeat(rep, 1, 1, 1)[:B_lat]
    noise_for_step = noise_for_step.to(latents.dtype)
    lat_next = pipeline.scheduler.step(
        noise_for_step, t, latents.detach().clone(),
        **extra_step_kwargs, return_dict=False
    )[0]
    if saved_idx is not None:
        pipeline.scheduler._step_index = saved_idx
    
    # Decode
    import torch
    with torch.no_grad():
        img = pipeline.vae.decode(
            (lat_next / pipeline.vae.config.scaling_factor)
            .to(device=next(pipeline.vae.parameters()).device,
                dtype=pipeline.vae.dtype)
        ).sample[0].detach()
    img_np = (((img.float() / 2 + 0.5).clamp_(0, 1)).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    
    # Save with mask overlay
    H, W = img_np.shape[:2]
    mask_np = mask4[0, 0].float().cpu().numpy()
    from PIL import Image
    mask_resized = np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize((W, H)))
    
    # Create red overlay for mask
    overlay = img_np.copy()
    mask_area = mask_resized > 128
    overlay[mask_area, 0] = np.clip(overlay[mask_area, 0] + 50, 0, 255)  # Add red tint
    
    Image.fromarray(overlay).save(os.path.join(debug_dir, f"prediction_step{step_idx:03d}.png"))

        
# ────────────────────────────────────────────────────────────────
#  NEW  helper 4:  once-per-run debug dumps
# ────────────────────────────────────────────────────────────────

def debug_reference_latents_once(
    pipeline,
    mask4: torch.Tensor,
    debug_dir: str,
) -> None:
    """
    Replicates the old pipeline.py debug section.
    Executes **only once** per pipeline instance.
    """
    if getattr(pipeline, "_dbg_mask_once", False):
        return

    # Check for reference latents under both possible names
    if hasattr(pipeline, "_ref_latents_all"):
        ref_lat = pipeline._ref_latents_all.detach()
    elif hasattr(pipeline, "_reference_latents"):
        ref_lat = pipeline._reference_latents.detach()
    else:
        print("[DBG] Warning: No reference latents found, skipping debug")
        return

    os.makedirs(debug_dir, exist_ok=True)

    # Rest of the function remains the same...
    mask_bool = mask4.repeat(1, 4, 1, 1).bool()

    fσ = ref_lat[mask_bool].std().item()
    bσ = ref_lat[~mask_bool].std().item()
    print(f"[DBG mask] σ_face={fσ:.3f}  σ_bg={bσ:.3f}")

    # # ① masked-latents preview
    # vis = ref_lat[0].abs().mean(0, keepdim=True)
    # save_image(vis / vis.max().clamp(min=1e-8),
    #            os.path.join(debug_dir, "ref_latents_masked.png"))
    
    # ① masked-latents preview (on the VAE grid)
    # make sure mask is on the same H×W as ref_lat
    m = mask4
    if m.shape[-2:] != ref_lat.shape[-2:]:
        m = F.interpolate(m.float(), size=ref_lat.shape[-2:], mode="nearest")
    m = m[0, 0]  # [H,W] in {0,1}
    # channelwise magnitude, then apply mask; set BG to mid-gray for contrast
    mag = ref_lat[0].pow(2).sum(0).sqrt()           # [H,W]
    bg  = mag.mean()
    vis = mag * m + bg * (1.0 - m)                  # face-only shows structure
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    save_image(vis.unsqueeze(0), os.path.join(debug_dir, "ref_latents_faceonly.png"))

    # ② RGB crop of reference face (once)
    if not hasattr(pipeline, "_saved_ref_face") and hasattr(pipeline, "_ref_img"):
        ref_img = pipeline._ref_img
        if ref_img is not None:
            ref_img = ref_img.float() * 0.5 + 0.5  # de-norm to [0,1]
            while ref_img.dim() > 3:
                ref_img = ref_img[0]
            
            up_mask = F.interpolate(mask4.float(),
                                    size=ref_img.shape[-2:],
                                    mode="nearest")[0, 0] > 0.5
            ys, xs = up_mask.nonzero(as_tuple=True)
            y0, y1 = 0, ref_img.shape[-2]
            x0, x1 = 0, ref_img.shape[-1]
            if ys.numel():
                y0, y1 = ys.min().item(), ys.max().item() + 1
                x0, x1 = xs.min().item(), xs.max().item() + 1
            ref_np = (ref_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(ref_np[y0:y1, x0:x1]).save(
                os.path.join(debug_dir, "reference_face_crop.png"))
            pipeline._saved_ref_face = True
            print(f"[DBG] saved reference face crop → {debug_dir}/reference_face_crop.png")
    
    # ③ store reusable step noise
    if not hasattr(pipeline, "_step_noise"):
        pipeline._step_noise = torch.randn_like(ref_lat)

    # ④ tiny sanity report
    face_mean = ref_lat[mask_bool].mean()
    face_std = ref_lat[mask_bool].std()
    print("[DBG norm] face_mean={:.4f}  face_std={:.4f}".format(
          face_mean.item(), face_std.item()))

    pipeline._dbg_mask_once = True


# ────────────────────────────────────────────────────────────────
#  NEW  helper 5:  save decoded reference latents once
# ────────────────────────────────────────────────────────────────


def save_debug_ref_latents(pipeline, debug_dir: str) -> None:
    """
    Decode reference latents back to RGB once and write
    `<debug_dir>/debug_ref_latents.png`.
    """
    if getattr(pipeline, "_saved_ref_latents_img", False):
        return

    # Check for reference latents under both possible names
    if hasattr(pipeline, "_ref_latents_all"):
        ref_lat = pipeline._ref_latents_all
    elif hasattr(pipeline, "_reference_latents"):
        ref_lat = pipeline._reference_latents
    else:
        print("[Debug] Warning: No reference latents found, skipping debug image")
        return

    os.makedirs(debug_dir, exist_ok=True)

    vae_device = next(pipeline.vae.parameters()).device
    vae_dtype = next(pipeline.vae.parameters()).dtype

    ref_lat = ref_lat.to(device=vae_device, dtype=vae_dtype)
    ref_lat = ref_lat / pipeline.vae.config.scaling_factor

    # img = pipeline.vae.decode(ref_lat).sample[0]
    # img_np = (
    #     (img.float() / 2 + 0.5)
    #     .clamp(0, 1)
    #     .permute(1, 2, 0)
    #     .cpu()
    #     .numpy() * 255
    # ).astype("uint8")
    
    import torch
    with torch.no_grad():
        img = pipeline.vae.decode(ref_lat).sample[0].detach()  # [3,H,W], in [-1,1]
    # remove letterbox padding if present
    pad = getattr(pipeline, "_ref_pad", None)
    if pad is not None:
        pl, pr, pt, pb = pad
        _, H, W = img.shape
        img = img[:, pt: H - pb, pl: W - pr]
    # optional: resize back to original pixel size for visualization
    orig = getattr(pipeline, "_ref_orig_size", None)
    if orig is not None:
        oh, ow = orig
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(oh, ow), mode="bilinear", align_corners=False)[0]
    img_np = (((img.float() / 2 + 0.5).clamp(0, 1)).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    

    Image.fromarray(img_np).save(os.path.join(debug_dir, "debug_ref_latents.png"))
    print(f"[Debug] saved reference latents image → {debug_dir}/debug_ref_latents.png")

    pipeline._saved_ref_latents_img = True


# ────────────────────────────────────────────────────────────────



def save_debug_ref_mask_overlay(pipeline, mask4_ref, debug_dir: str) -> None:
    """Decode ref latents and overlay the ref mask (imported or mask4_ref) for alignment check."""
    if getattr(pipeline, "_saved_ref_mask_overlay", False):
        return
    # get ref latents
    # Do NOT use `or` with tensors — explicit None-check instead
    ref_lat = getattr(pipeline, "_ref_latents_all", None)
    if ref_lat is None:
        ref_lat = getattr(pipeline, "_reference_latents", None)
    if ref_lat is None:
        print("[Debug] No reference latents; skip mask overlay")
        return
        
    vae_device = next(pipeline.vae.parameters()).device
    vae_dtype  = next(pipeline.vae.parameters()).dtype
    ref_lat = ref_lat.to(device=vae_device, dtype=vae_dtype)
    ref_lat = ref_lat / pipeline.vae.config.scaling_factor
    import torch
    with torch.no_grad():
        img = pipeline.vae.decode(ref_lat).sample[0].detach()  # [3,H,W] in [-1,1]

    # build/get mask tensor
    import torch, torch.nn.functional as F
    m = None

    # Try high-res mask first
    if hasattr(pipeline, "_face_mask_highres_ref"):
        m = torch.from_numpy(pipeline._face_mask_highres_ref).float()
        m = m.to(device=img.device, dtype=img.dtype)
        # # Resize to match decoded image if needed
        # H, W = img.shape[-2:]
        # if m.shape != (H, W):
        #     m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    
        # Apply the same scaling and padding as the reference image
        if hasattr(pipeline, "_face_mask_scaled_size_ref") and hasattr(pipeline, "_face_mask_pad_ref"):
            rh, rw = pipeline._face_mask_scaled_size_ref
            pl, pr, pt, pb = pipeline._face_mask_pad_ref
            
            # First resize to scaled size (matching aspect ratio)
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False)[0, 0]
            
            # Then apply padding to match the decoded image size
            H, W = img.shape[-2:]
            m_padded = torch.zeros((H, W), device=m.device, dtype=m.dtype)
            m_padded[pt:pt+rh, pl:pl+rw] = m
            m = m_padded
        else:
            # Fallback: direct resize if no padding info
            H, W = img.shape[-2:]
            if m.shape != (H, W):
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]    
    
    # if mask4_ref is not None:
    elif mask4_ref is not None:
        m = mask4_ref
        if m.dim() == 4 and m.shape[1] == 4:
           m = m[:, :1]                      # [1,1,h,w]
    elif hasattr(pipeline, "_face_mask_t_ref"):
        m = pipeline._face_mask_t_ref.float() # [1,1,H,W]
    if m is None:
        print("[Debug] No ref mask available; skip overlay")
        return
    m = m.to(device=img.device, dtype=img.dtype)

   # upsample mask to decoded image grid
    H, W = img.shape[-2:]
    mh, mw = m.shape[-2:]
    if (mh, mw) == (H // 8, W // 8):
       m = F.interpolate(m, scale_factor=8, mode="nearest")
    elif (mh, mw) != (H, W):
       m = F.interpolate(m, size=(H, W), mode="nearest")
    # m = m[0, 0]  # [H,W], 0..1
    
    # Only squeeze if m is 4D, keep as 2D
    if m.dim() == 4:
        m = m[0, 0]  # [H,W], 0..1
    elif m.dim() == 3:
        m = m[0]
    
    # remove letterbox padding if present
    pad = getattr(pipeline, "_ref_pad", None)
    if pad is not None:
        pl, pr, pt, pb = pad
        img = img[:, pt:H - pb, pl:W - pr]
        # m   = m[pt:H - pb, pl:W - pr]

        # Only slice if m is 2D
        if m.dim() == 2:
            m = m[pt:H - pb, pl:W - pr]
        elif m.dim() == 0:
            # m got squeezed too much, this shouldn't happen
            print(f"[Debug] Warning: mask became scalar, skipping padding removal")        
        H, W = img.shape[-2:]

    # restore original size for visualization if known
    orig = getattr(pipeline, "_ref_orig_size", None)
    if orig is not None:
        oh, ow = orig
        img = F.interpolate(img.unsqueeze(0), size=(oh, ow), mode="bilinear", align_corners=False)[0]
        # m   = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(oh, ow), mode="nearest")[0,0]

        # Ensure m is 2D before interpolation
        if m.dim() == 2:
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(oh, ow), mode="nearest")[0, 0]
        elif m.dim() < 2:
            print(f"[Debug] Warning: mask has {m.dim()} dims, expected 2")
        

    # compose red overlay
    vis = (img.float() / 2 + 0.5).clamp(0, 1)
    red = torch.zeros_like(vis); red[0].fill_(1.0)
    alpha = 0.35
    # vis = vis * (1 - alpha * m) + red * (alpha * m)
    
    # Ensure m broadcasts correctly with vis (C, H, W)
    if m.dim() == 2:
        m_broadcast = m.unsqueeze(0)  # Add channel dimension for broadcasting
    else:
        m_broadcast = m
    vis = vis * (1 - alpha * m_broadcast) + red * (alpha * m_broadcast)

    import numpy as np
    from PIL import Image
    img_np = (vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(img_np).save(os.path.join(debug_dir, "debug_ref_latents_mask_overlay.png"))
    print(f"[Debug] saved → {debug_dir}/debug_ref_latents_mask_overlay.png")
    pipeline._saved_ref_mask_overlay = True


# ───────────────────────────────────────────────────────────────────
# 3)  Attention‑map forward‑hooks (heat‑map harvesting)
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
    """Register layer‑specific forward hooks so external code can harvest
    raw attention logits and later build binary face masks.

    Parameters mirror the original implementation; *attn_maps_current* and
    *orig_attn_forwards* are caller‑owned dicts where we will push maps /
    keep backups.
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
        # one‑shot build of aux prompt → focus_latents & token indices
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
                if seq[i : i + len(sub)] == sub:
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
