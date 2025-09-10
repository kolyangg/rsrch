# pm_debug.py ---------------------------------------------------------------
import os, math, cv2, torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _decode(pipe, lat):
    """
    Robust latent → RGB helper that never hits the
    *CPUBFloat16Type vs CUDABFloat16Type* mismatch.

    It queries the VAE **at call-time** (after accelerate’s off-loading
    decisions) and moves the latent tensor to that exact device/dtype before
    decoding.
    """

    # Figure out where the VAE will *actually* run:
    #   • Newer Accelerate puts `_execution_device` on the sub-module itself.
    #   • Older builds attach that attribute to the parent pipeline.
    #   • If neither exists we fall back to the param device.
    vae_dev = (
        getattr(pipe.vae, "_execution_device", None)     # Accelerate ≥ 0.25
        or getattr(pipe,     "_execution_device", None)  # Accelerate < 0.25
        or next(pipe.vae.parameters()).device            # no off-loading
    )

    # Off-loading sometimes leaves everything on the CPU until the very last
    # moment; pre-emptively switch to GPU-0 if we can—this matches where the
    # weights are about to be moved.
    if vae_dev.type == "cpu" and torch.cuda.is_available():
        vae_dev = torch.device("cuda", 0)

    vae_dtype = pipe.vae.dtype                           # bf16 / fp16 / fp32


    # match device & dtype first, then apply the standard scaling factor
    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    lat   = (lat / scale).to(device=vae_dev, dtype=vae_dtype, copy=False)

    with torch.no_grad():
        img = pipe.vae.decode(lat.contiguous()).sample[0]      # (3,H,W)

    img = (img.float() / 2 + 0.5).clamp_(0, 1)                 # (3,H,W)
    return img.permute(1, 2, 0).cpu().numpy()                  # (H,W,3)


def _overlay_mask(img_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay binary mask (H×W, bool/0-1) in solid red on RGB uint8 image."""
    if mask is None:
        return img_uint8
    m = (mask.astype(np.uint8) * 255)
    m = cv2.resize(m, (img_uint8.shape[1], img_uint8.shape[0]),
                   interpolation=cv2.INTER_NEAREST)
    red   = np.zeros_like(img_uint8); red[..., 0] = 255
    alpha = m[..., None] / 255.0
    return (img_uint8 * (1 - alpha) + red * alpha).astype(np.uint8)


def make_mask_callback(pipe,
                       mask_interval: int = 5,
                       container: list | None = None):
    """
    Returns a callback compatible with `callback_on_step_end`.

    It grabs a frame every `mask_interval` steps (always step-0 and last step),
    copies latents → CPU (fp32), and stores them in `container`.
    """
    frames = [] if container is None else container
    labels  = []                     # keep matching text labels
    steps  = {"total": None}          # mutable closure

    def cb(pipeline, step_index, t, tensors):
        if steps["total"] is None:
            steps["total"] = pipeline._num_timesteps - 1


        wanted = (
            step_index % mask_interval == 0
            or step_index == 0
            or step_index == steps["total"]
        )
        # must ALWAYS return a mapping so `pipeline_NS.py` can do `.pop()`
        if not wanted:
            return {}


        # Keep original device & dtype; `_decode()` will align them.
        lat = tensors["latents"].detach()


        img = _decode(pipeline, lat)                    # (H,W,3) 0-1
        img_u8 = (img * 255).round().clip(0, 255).astype(np.uint8)

        # ── 1) masked overlay (kept for every captured step) ─────────────
        mask = getattr(pipeline, "_face_mask", None)
        img_mask = _overlay_mask(img_u8, mask)
        frames.append(Image.fromarray(img_mask))
        labels.append(f"S{step_index}" if step_index != steps["total"] else "Mask")

        # ── 2) clean final image (only once, at very last step) ──────────
        if step_index == steps["total"]:
            frames.append(Image.fromarray(img_u8))
            labels.append("Final")

        return {}  # nothing to override, but keep contract

    cb.frames = frames
    cb.labels  = labels
    return cb


def save_strip(frames: list[Image.Image],
               out_path: str,
               labels: list[str] | None = None,
               head_h: int = 30):
    """Save the montage strip identical to attn_hm_NS_nosm7.py."""
    if not frames:
        return
    w, h = frames[0].width, frames[0].height
    strip = Image.new("RGB", (w * len(frames), h + head_h), "black")
    draw  = ImageDraw.Draw(strip)
    font  = ImageFont.load_default()

    for idx, frm in enumerate(frames):
        x = idx * w
        strip.paste(frm, (x, head_h))
        label = labels[idx] if labels else f"S{idx*5}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.text((x + (w - tw)//2, (head_h - th)//2),
                  label, font=font, fill="white")

    strip.save(out_path, quality=95)
    print(f"[DEBUG] mask strip saved → {out_path}")
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# NEW: identical to `make_mask_callback`, but NEVER overlays the mask
#      → collects the clean image evolution frames.
# ---------------------------------------------------------------------------
def make_image_callback(pipe,
                        mask_interval: int = 5,
                        container: list | None = None):
    """
    Grab decoder snapshots every `mask_interval` steps, *without* the red mask.
    Returned callback exposes `cb.frames` and `cb.labels`, just like the mask
    variant, so `save_strip()` works unmodified.
    """
    frames = [] if container is None else container
    labels = []
    steps  = {"total": None}

    def cb(pipeline, step_index, t, tensors):
        if steps["total"] is None:
            steps["total"] = pipeline._num_timesteps - 1

        want = (step_index % mask_interval == 0 or
                step_index == 0 or
                step_index == steps["total"])
        if not want:
            return

        lat = tensors["latents"].detach().to("cpu", torch.float32)
        img = _decode(pipeline, lat)              # (H,W,3) float 0-1
        img = (img * 255).round().clip(0, 255).astype(np.uint8)

        frames.append(Image.fromarray(img))
        lbl = "Final" if step_index == steps["total"] else f"S{step_index}"
        labels.append(lbl)
        return tensors

    cb.frames  = frames
    cb.labels  = labels
    return cb
