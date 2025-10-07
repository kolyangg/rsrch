import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .branched_NS import two_branch_predict_pulid as _two_branch_predict


class PipeProxy:
    """Light proxy exposing config flags that branched processors read.

    It forwards attribute access to the wrapped diffusers pipeline but provides
    concrete, mutable attributes such as `do_classifier_free_guidance` that may
    be read-only properties on the underlying object.
    """

    def __init__(self, base, do_cfg: bool, pose_adapt_ratio: float, ca_mixing_for_face: bool):
        self._base = base
        self.do_classifier_free_guidance = bool(do_cfg)
        self.pose_adapt_ratio = float(pose_adapt_ratio)
        self.ca_mixing_for_face = bool(ca_mixing_for_face)
        self._cross_attention_kwargs = None

    def __getattr__(self, name):
        return getattr(self._base, name)


def letterbox_and_encode_reference(pipeline, reference_pil, size_hw: Tuple[int, int], mask_ref_img: Optional[np.ndarray]):
    """Letterbox the reference PIL to target size and encode to latents.

    Also stores padding/size metadata on both the wrapper (PuLIDPipeline) and
    inner diffusers pipeline for debug helpers to use later.
    """
    from PIL import Image as _PILImage

    tgt_h, tgt_w = int(size_hw[0]), int(size_hw[1])
    orig_w, orig_h = reference_pil.size
    scale = min(tgt_w / float(orig_w), tgt_h / float(orig_h))
    rw = max(1, int(round(orig_w * scale)))
    rh = max(1, int(round(orig_h * scale)))
    pl = (tgt_w - rw) // 2
    pr = tgt_w - rw - pl
    pt = (tgt_h - rh) // 2
    pb = tgt_h - rh - pt
    resized = reference_pil.resize((rw, rh), resample=_PILImage.LANCZOS)
    canvas = _PILImage.new('RGB', (tgt_w, tgt_h), color=(0, 0, 0))
    canvas.paste(resized, (pl, pt))

    # Store metadata for debug
    setattr(pipeline, "_ref_pad", (pl, pr, pt, pb))
    setattr(pipeline, "_ref_scaled_size", (rh, rw))
    setattr(pipeline, "_ref_orig_size", (orig_h, orig_w))
    setattr(pipeline.pipe, "_ref_pad", (pl, pr, pt, pb))
    setattr(pipeline.pipe, "_ref_scaled_size", (rh, rw))
    setattr(pipeline.pipe, "_ref_orig_size", (orig_h, orig_w))

    # Provide high-res mask 1:1 so helpers can map it deterministically
    if mask_ref_img is not None:
        hi = (mask_ref_img > 0.5).astype(np.uint8)
        setattr(pipeline.pipe, "_face_mask_highres_ref", hi)
        setattr(pipeline.pipe, "_face_mask_scaled_size_ref", (rh, rw))
        setattr(pipeline.pipe, "_face_mask_pad_ref", (pl, pr, pt, pb))

    # Encode
    pixel_values = pipeline.pipe.image_processor.preprocess(canvas, height=tgt_h, width=tgt_w)
    pixel_values = pixel_values.to(device=pipeline.device, dtype=pipeline.pipe.vae.dtype)
    with torch.no_grad():
        dist = pipeline.pipe.vae.encode(pixel_values).latent_dist
        latents_ref = dist.mean
    latents_ref = latents_ref * pipeline.pipe.vae.config.scaling_factor
    setattr(pipeline, "_reference_latents", latents_ref)
    setattr(pipeline.pipe, "_reference_latents", latents_ref)
    return latents_ref


def _as_tensor_mask(arr: np.ndarray, B: int, H: int, W: int, device, dtype):
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = t.to(device=device, dtype=dtype)
    if (t.shape[-2], t.shape[-1]) != (H, W):
        t = F.interpolate(t, size=(H, W), mode='nearest')
    return t.expand(B, 1, H, W)


def build_face_masks(pipeline, B: int, H: int, W: int, device, dtype):
    """Prepare generator/ref masks on the latent grid; return (m_gen, m_ref)."""
    m_gen = None
    m_ref = None
    if getattr(pipeline, "_face_mask_img", None) is not None:
        m_gen = _as_tensor_mask(pipeline._face_mask_img, B, H, W, device, dtype)
    if getattr(pipeline, "_face_mask_ref_img", None) is not None:
        m_ref = _as_tensor_mask(pipeline._face_mask_ref_img, B, H, W, device, dtype)
    if m_gen is None and m_ref is None:
        # Fallback to a soft center mask
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cy, cx = H / 2.0, W / 2.0
        ry, rx = H * 0.35, W * 0.35
        m = (((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2)) <= 1.0
        m = m.to(dtype=dtype)[None, None].expand(B, 1, H, W)
        return m, m
    if m_gen is None:
        m_gen = m_ref
    if m_ref is None:
        m_ref = m_gen
    return (m_gen > 0.5).to(dtype=dtype), (m_ref > 0.5).to(dtype=dtype)


def two_branch_cfg_step(pipeline_proxy, x_ddim_space, t, pos_args, neg_args, m_gen, m_ref, reference_latents, cfg_scale: float, step_idx: int, debug_dir: Optional[str]):
    """Run two-branch predict for positive/negative passes and return CFG noise."""
    # Positive
    pipeline_proxy._cross_attention_kwargs = pos_args.get('cross_attention_kwargs', None)
    eps_pos = _two_branch_predict(
        pipeline_proxy,
        latent_model_input=x_ddim_space,
        t=t,
        prompt_embeds=pos_args['encoder_hidden_states'],
        added_cond_kwargs=pos_args.get('added_cond_kwargs', {}),
        mask4=m_gen,
        mask4_ref=m_ref,
        reference_latents=reference_latents,
        cross_attention_kwargs=pos_args.get('cross_attention_kwargs', None),
        id_embeds=None,
        step_idx=step_idx,
        debug_dir=debug_dir,
    )
    # Negative
    pipeline_proxy._cross_attention_kwargs = neg_args.get('cross_attention_kwargs', None)
    eps_neg = _two_branch_predict(
        pipeline_proxy,
        latent_model_input=x_ddim_space,
        t=t,
        prompt_embeds=neg_args['encoder_hidden_states'],
        added_cond_kwargs=neg_args.get('added_cond_kwargs', {}),
        mask4=m_gen,
        mask4_ref=m_ref,
        reference_latents=reference_latents,
        cross_attention_kwargs=neg_args.get('cross_attention_kwargs', None),
        id_embeds=None,
        step_idx=step_idx,
        debug_dir=debug_dir,
    )
    return eps_neg + cfg_scale * (eps_pos - eps_neg)

