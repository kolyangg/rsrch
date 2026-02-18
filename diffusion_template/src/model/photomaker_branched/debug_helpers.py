from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

DEBUG_LOG_DEBUG_IMAGES = os.environ.get("PM_DEBUG_IMAGES", "1") not in {"0", "false", "False", ""}


def log_debug_image(message: str) -> None:
    if DEBUG_LOG_DEBUG_IMAGES:
        print(message)


__all__ = [
    "DEBUG_LOG_DEBUG_IMAGES",
    "log_debug_image",
    "save_branch_previews",
    "debug_reference_latents_once",
    "save_debug_ref_latents",
    "save_debug_ref_mask_overlay",
    "save_debug_images",
]


# ────────────────────────────────────────────────────────────────
# helper: per-step preview PNGs
# ────────────────────────────────────────────────────────────────

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

    debug_path = Path(debug_dir)  # --- MODIFIED For training integration ---
    debug_path.mkdir(parents=True, exist_ok=True)  # --- MODIFIED For training integration ---

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
    mask_resized = np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize((W, H)))

    # Create red overlay for mask
    overlay = img_np.copy()
    mask_area = mask_resized > 128
    overlay[mask_area, 0] = np.clip(overlay[mask_area, 0] + 50, 0, 255)  # Add red tint

    out_path = debug_path / f"prediction_step{step_idx:03d}.png"
    Image.fromarray(overlay).save(out_path)  # --- MODIFIED For training integration ---
    log_debug_image(f"[DebugImage] prediction step={step_idx} → {out_path}")  # --- MODIFIED For training integration ---


# ────────────────────────────────────────────────────────────────
# helper: once-per-run debug dumps
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

    debug_path = Path(debug_dir)  # --- MODIFIED For training integration ---
    debug_dir_key = str(debug_path)  # --- MODIFIED For training integration ---

    saved_dirs = getattr(pipeline, "_dbg_mask_dirs", set())
    if not isinstance(saved_dirs, set):
        saved_dirs = set()
    if debug_dir_key in saved_dirs:
        return
    saved_dirs.add(debug_dir_key)
    pipeline._dbg_mask_dirs = saved_dirs

    # Check for reference latents under both possible names
    if hasattr(pipeline, "_ref_latents_all"):
        ref_lat = pipeline._ref_latents_all.detach()
    elif hasattr(pipeline, "_reference_latents"):
        ref_lat = pipeline._reference_latents.detach()
    else:
        print("[DBG] Warning: No reference latents found, skipping debug")
        return

    debug_path.mkdir(parents=True, exist_ok=True)

    mask_bool = mask4.repeat(1, 4, 1, 1).bool()

    fσ = ref_lat[mask_bool].std().item()
    bσ = ref_lat[~mask_bool].std().item()
    print(f"[DBG mask] σ_face={fσ:.3f}  σ_bg={bσ:.3f}")

    # ① masked-latents preview (on the VAE grid)
    m = mask4
    if m.shape[-2:] != ref_lat.shape[-2:]:
        m = F.interpolate(m.float(), size=ref_lat.shape[-2:], mode="nearest")
    m = m[0, 0]  # [H,W] in {0,1}
    mag = ref_lat[0].pow(2).sum(0).sqrt()           # [H,W]
    bg = mag.mean()
    vis = mag * m + bg * (1.0 - m)                  # face-only shows structure
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    out_face_path = debug_path / "ref_latents_faceonly.png"
    save_image(vis.unsqueeze(0), out_face_path)  # --- MODIFIED For training integration ---
    log_debug_image(f"[DebugImage] ref_latents_faceonly → {out_face_path}")

    # ② RGB crop of reference face (once)
    saved_face_dirs = getattr(pipeline, "_saved_ref_face_dirs", set())
    if not isinstance(saved_face_dirs, set):
        saved_face_dirs = set()
    if hasattr(pipeline, "_ref_img"):
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

            if debug_dir_key not in saved_face_dirs:
                face_path = debug_path / "reference_face_crop.png"
                Image.fromarray(ref_np[y0:y1, x0:x1]).save(face_path)
                saved_face_dirs.add(debug_dir_key)
                pipeline._saved_ref_face_dirs = saved_face_dirs
                pipeline._saved_ref_face = True
                log_debug_image(f"[DebugImage] reference_face_crop → {face_path}")
                print(f"[DBG] saved reference face crop → {face_path}")

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
# helper: save decoded reference latents once
# ────────────────────────────────────────────────────────────────

def save_debug_ref_latents(pipeline, debug_dir: str) -> None:
    """
    Decode reference latents back to RGB once and write
    `<debug_dir>/debug_ref_latents.png`.
    """
    debug_path = Path(debug_dir)
    debug_dir_key = str(debug_path)

    saved_dirs = getattr(pipeline, "_saved_ref_latents_dirs", set())
    if not isinstance(saved_dirs, set):
        saved_dirs = set()
    if debug_dir_key in saved_dirs:
        return

    # Check for reference latents under both possible names
    if hasattr(pipeline, "_ref_latents_all"):
        ref_lat = pipeline._ref_latents_all
    elif hasattr(pipeline, "_reference_latents"):
        ref_lat = pipeline._reference_latents
    else:
        print("[Debug] Warning: No reference latents found, skipping debug image")
        return

    debug_path.mkdir(parents=True, exist_ok=True)

    vae_device = next(pipeline.vae.parameters()).device
    vae_dtype = next(pipeline.vae.parameters()).dtype

    ref_lat = ref_lat.to(device=vae_device, dtype=vae_dtype)
    ref_lat = ref_lat / pipeline.vae.config.scaling_factor

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

    lat_path = debug_path / "debug_ref_latents.png"
    Image.fromarray(img_np).save(lat_path)
    log_debug_image(f"[DebugImage] debug_ref_latents → {lat_path}")
    print(f"[Debug] saved reference latents image → {lat_path}")

    saved_dirs.add(debug_dir_key)
    pipeline._saved_ref_latents_dirs = saved_dirs
    pipeline._saved_ref_latents_img = True


def save_debug_ref_mask_overlay(pipeline, mask4_ref, debug_dir: str) -> None:
    """Decode ref latents and overlay the ref mask (imported or mask4_ref) for alignment check."""
    debug_path = Path(debug_dir)
    debug_dir_key = str(debug_path)

    saved_dirs = getattr(pipeline, "_saved_ref_mask_overlay_dirs", set())
    if not isinstance(saved_dirs, set):
        saved_dirs = set()
    if debug_dir_key in saved_dirs:
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
    vae_dtype = next(pipeline.vae.parameters()).dtype
    ref_lat = ref_lat.to(device=vae_device, dtype=vae_dtype)
    ref_lat = ref_lat / pipeline.vae.config.scaling_factor
    with torch.no_grad():
        img = pipeline.vae.decode(ref_lat).sample[0].detach()  # [3,H,W] in [-1,1]

    # build/get mask tensor
    m = None

    # Try high-res mask first
    if hasattr(pipeline, "_face_mask_highres_ref"):
        m = torch.from_numpy(pipeline._face_mask_highres_ref).float()
        m = m.to(device=img.device, dtype=img.dtype)

        # Apply the same scaling and padding as the reference image
        if hasattr(pipeline, "_face_mask_scaled_size_ref") and hasattr(pipeline, "_face_mask_pad_ref"):
            rh, rw = pipeline._face_mask_scaled_size_ref
            pl, pr, pt, pb = pipeline._face_mask_pad_ref

            # First resize to scaled size (matching aspect ratio)
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False)[0, 0]

            # Then apply padding to match the decoded image size
            H, W = img.shape[-2:]
            m_padded = torch.zeros((H, W), device=m.device, dtype=m.dtype)
            m_padded[pt:pt + rh, pl:pl + rw] = m
            m = m_padded
        else:
            # Fallback: direct resize if no padding info
            H, W = img.shape[-2:]
            if m.shape != (H, W):
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    elif mask4_ref is not None:
        m = mask4_ref
        if m.dim() == 4 and m.shape[1] == 4:
            m = m[:, :1]  # [1,1,h,w]
    elif hasattr(pipeline, "_face_mask_t_ref"):
        m = pipeline._face_mask_t_ref.float()  # [1,1,H,W]
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

        # Only slice if m is 2D
        if m.dim() == 2:
            m = m[pt:H - pb, pl:W - pr]
        elif m.dim() == 0:
            # m got squeezed too much, this shouldn't happen
            print("[Debug] Warning: mask became scalar, skipping padding removal")
        H, W = img.shape[-2:]

    # restore original size for visualization if known
    orig = getattr(pipeline, "_ref_orig_size", None)
    if orig is not None:
        oh, ow = orig
        img = F.interpolate(img.unsqueeze(0), size=(oh, ow), mode="bilinear", align_corners=False)[0]

        # Ensure m is 2D before interpolation
        if m.dim() == 2:
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(oh, ow), mode="nearest")[0, 0]
        elif m.dim() < 2:
            print(f"[Debug] Warning: mask has {m.dim()} dims, expected 2")

    # compose red overlay
    vis = (img.float() / 2 + 0.5).clamp(0, 1)
    red = torch.zeros_like(vis); red[0].fill_(1.0)
    alpha = 0.35

    # Ensure m broadcasts correctly with vis (C, H, W)
    if m.dim() == 2:
        m_broadcast = m.unsqueeze(0)  # Add channel dimension for broadcasting
    else:
        m_broadcast = m
    vis = vis * (1 - alpha * m_broadcast) + red * (alpha * m_broadcast)

    img_np = (vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    debug_path.mkdir(parents=True, exist_ok=True)
    overlay_path = debug_path / "debug_ref_latents_mask_overlay.png"
    Image.fromarray(img_np).save(overlay_path)
    log_debug_image(f"[DebugImage] debug_ref_latents_mask_overlay → {overlay_path}")
    print(f"[Debug] saved → {overlay_path}")
    saved_dirs.add(debug_dir_key)
    pipeline._saved_ref_mask_overlay_dirs = saved_dirs
    pipeline._saved_ref_mask_overlay = True


def save_debug_images(
   pipeline,
   noise_pred: torch.Tensor,
   mask: torch.Tensor,
   step_idx: int,
   output_dir: str = "debug",
):
   """Save debug visualizations."""
   os.makedirs(output_dir, exist_ok=True)
   
   # Save mask visualization
   if mask is not None and step_idx % 10 == 0:
       if mask.dim() == 4:
           mask_vis = mask[0, 0].cpu().numpy()
       else:
           mask_vis = mask[0].cpu().numpy()
       mask_vis = (mask_vis * 255).astype("uint8")
       Image.fromarray(mask_vis).save(f"{output_dir}/mask_step_{step_idx:03d}.png")
   
   # Save noise prediction stats
   if step_idx < 3:
       stats = {
           "step": step_idx,
           "mean": noise_pred.mean().item(),
           "std": noise_pred.std().item(),
           "min": noise_pred.min().item(),
           "max": noise_pred.max().item(),
       }
       print(f"[Debug] Step {step_idx} stats: {stats}")
