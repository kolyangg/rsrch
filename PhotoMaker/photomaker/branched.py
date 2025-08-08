# photomaker/branched.py

"""
Branched attention and heatmap functionality for PhotoMaker pipeline.
Extracted from pipeline_NS2.py for cleaner code organization.
"""

import os
import math
import types
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import cv2

from diffusers.models.attention_processor import Attention as CrossAttention

# Import mask utilities
from .mask_utils import (
    MASK_LAYERS_CONFIG,
    compute_binary_face_mask,
    _resize_map,
    simple_threshold_mask,
)

from .heatmap_utils import build_hook_focus_token, build_hook_identity


# Debug configuration
DEBUG_DIR = os.getenv("PM_DEBUG_DIR", "./branched_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
FULL_DEBUG = False


def _save_gray(
    arr: np.ndarray,
    path: str,
    size: tuple[int, int] | None = None,  # (W,H)
) -> None:
    """
    Save a H×W uint8 array as an 8-bit grayscale PNG.
    If *size* is given the array is **nearest-neighbour**-resized first
    so the binary mask keeps hard edges.
    """
    img = Image.fromarray(arr, mode="L")
    if size and img.size != size:
        img = img.resize(size, Image.NEAREST)
    img.save(path)


def encode_face_latents(
    vae,
    device,
    id_pixel_values: torch.Tensor,
    target_hw: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return normalised VAE latents for the **first** ID image
    and resize them so that `latents.shape[-2:] == target_hw`.
    Always encodes in fp32 for numerical stability, then casts
    to the requested *dtype* on the caller's device.
    """
    vae_weight = next(vae.parameters())
    vae_dtype = vae_weight.dtype
    vae_device = vae_weight.device

    ref_img = (
        id_pixel_values[0, 0]  # (3,H,W)
        .unsqueeze(0)  # (1,3,H,W)
        .to(device=vae_device, dtype=vae_dtype)
    )

    with torch.no_grad():
        z = vae.encode(ref_img).latent_dist.mode() * vae.config.scaling_factor

    z = F.interpolate(z.float(), size=target_hw, mode="bilinear")
    z = z.clamp_(-5.0, 5.0)
    z = (z - z.mean()) / z.std().clamp(min=1e-4)
    return z.to(device=device, dtype=dtype).detach()


def patch_self_attention_for_branched_mode(pipeline):
    """
    Patches the self-attention modules in the UNet to support branched attention.
    """
    
    def custom_attn_forward(mod, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Keep default route for cross-attention or if no override
        if encoder_hidden_states is not None or getattr(pipeline, "_kv_override", None) is None:
            return mod._orig_forward(hidden_states, encoder_hidden_states, attention_mask)

        # Reference latents (preferred) or id-vector
        C_tgt = hidden_states.shape[-1]

        if pipeline._kv_override is not None:  # preferred
            ref = pipeline._kv_override  # (B,C,H,W)
            if ref.dim() == 4:
                ref = ref.permute(0, 2, 3, 1)  # (B,H,W,C)
                ref = ref.reshape(1, -1, ref.shape[3])  # (1,L,Csrc)
            C_src = ref.shape[-1]
            if C_src < C_tgt:
                ref = F.pad(ref, (0, C_tgt - C_src))
            elif C_src > C_tgt:
                ref = ref[..., :C_tgt]
            if not getattr(pipeline, "_dbg_attn_once", False):
                print(f"[DBG] {mod.__class__.__name__} uses REF latents  L={ref.shape[1]}  C={C_tgt}")
                pipeline._dbg_attn_once = True
        elif pipeline._id_embed_vec is not None:
            ref_vec = pipeline._id_embed_vec.to(hidden_states.dtype)
            if ref_vec.numel() < C_tgt:
                repeat = (C_tgt + ref_vec.numel() - 1) // ref_vec.numel()
                ref_vec = ref_vec.repeat(repeat)[:C_tgt]
            else:
                ref_vec = ref_vec[:C_tgt]
            ref = ref_vec.unsqueeze(0).unsqueeze(0)  # (1,1,C_tgt)
            if not hasattr(mod, "_dbg_id"):
                print(f"[DBG] {mod.__class__.__name__} uses ID vector  C={C_tgt}")
                mod._dbg_id = True
        else:
            raise RuntimeError("No reference source for branched attention")

        # Project current latent → original K/V
        q = (mod.to_q if hasattr(mod, "to_q") else mod.q_proj)(hidden_states)
        k_orig = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj)(hidden_states)
        v_orig = (mod.to_v if hasattr(mod, "to_v") else mod.v_proj)(hidden_states)

        # CFG-aware K/V replacement
        K_IN = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj).weight.shape[1]

        if pipeline.do_classifier_free_guidance and k_orig.shape[0] % 2 == 0:
            B_half = k_orig.shape[0] // 2
            _, L_cond, C_embed = k_orig[B_half:].shape

            ref_flat = (
                pipeline._kv_override.permute(0, 2, 3, 1)
                .reshape(B_half, L_cond, -1)
            )
            # pad / clip so last-dim == K_IN
            if ref_flat.shape[-1] < K_IN:
                ref_flat = F.pad(ref_flat, (0, K_IN - ref_flat.shape[-1]))
            elif ref_flat.shape[-1] > K_IN:
                ref_flat = ref_flat[..., :K_IN]

            k_cond = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj)(ref_flat)
            v_cond = (mod.to_v if hasattr(mod, "to_v") else mod.v_proj)(ref_flat)

            k = torch.cat([k_orig[:B_half], k_cond], 0)
            v = torch.cat([v_orig[:B_half], v_cond], 0)
        else:
            # no CFG ⇒ project reference for the whole batch
            B_lat = k_orig.shape[0]
            L_lat = k_orig.shape[1]

            ref_flat = (
                pipeline._kv_override.permute(0, 2, 3, 1)
                .reshape(B_lat, L_lat, -1)
            )
            if ref_flat.shape[-1] < K_IN:
                ref_flat = F.pad(ref_flat, (0, K_IN - ref_flat.shape[-1]))
            elif ref_flat.shape[-1] > K_IN:
                ref_flat = ref_flat[..., :K_IN]

            k = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj)(ref_flat)
            v = (mod.to_v if hasattr(mod, "to_v") else mod.v_proj)(ref_flat)

        B, L, C = hidden_states.shape
        h = mod.heads
        d = C // h
        scl = 1 / math.sqrt(d)

        q = q.view(B, L, h, d).permute(0, 2, 1, 3)  # B h L d

        Bk, Lk, _ = k.shape
        k = k.view(Bk, Lk, h, d).permute(0, 2, 1, 3)  # Bk h Lk d
        v = v.view(Bk, Lk, h, d).permute(0, 2, 1, 3)  # Bk h Lk d

        attn = (q @ k.transpose(-1, -2)) * scl
        attn = attn.softmax(dim=-1)

        if FULL_DEBUG and not hasattr(mod, "_once"):
            layer_name = getattr(mod, "_orig_name", "self_attn")
            print(f"[{layer_name}] mean-P = {attn.mean().item():.4f}")
            mod._once = True

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)
        out = mod.to_out[0](out)
        out = mod.to_out[1](out)

        return out

    patched = 0
    for n, m in pipeline.unet.named_modules():
        if isinstance(m, CrossAttention) and \
           not getattr(m, "is_cross_attention", False) and \
           not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward
            m.forward = types.MethodType(custom_attn_forward, m)
            patched += 1
    print(f"[DEBUG] self-attention patched count = {patched}")


def setup_attention_hooks(pipeline, heatmap_mode, focus_token, class_tokens_mask, 
                         attn_maps_current, orig_attn_forwards, device):
    """
    Sets up hooks for collecting attention maps.
    """
    wanted_layers = {spec["name"] for spec in MASK_LAYERS_CONFIG}

    if heatmap_mode == "identity":
        hook_builder = lambda ln, mod: build_hook_identity(
            ln,
            mod,
            wanted_layers,
            class_tokens_mask,
            pipeline.num_tokens,
            attn_maps_current,
            orig_attn_forwards,
            pipeline.do_classifier_free_guidance,
        )
    else:  # "token"
        # build focus_latents & token indices ONCE
        aux_prompt = f"a {focus_token}"
        with torch.no_grad():
            focus_lat, *_ = pipeline.encode_prompt(
                prompt=aux_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        tok = pipeline.tokenizer or pipeline.tokenizer_2
        idsA = tok(aux_prompt, add_special_tokens=False).input_ids
        idsW = tok(" " + focus_token, add_special_tokens=False).input_ids
        
        def _find_sub(seq, sub):
            for i in range(len(seq) - len(sub) + 1):
                if seq[i:i + len(sub)] == sub:
                    return list(range(i, i + len(sub)))
            return []
        
        token_idx_global = _find_sub(idsA, idsW)
        if not token_idx_global:
            raise RuntimeError(f"focus token '{focus_token}' not found")

        hook_builder = lambda ln, mod: build_hook_focus_token(
            ln, mod, wanted_layers, focus_lat, token_idx_global,
            attn_maps_current, orig_attn_forwards,
            pipeline.do_classifier_free_guidance
        )

    hook_count = 0
    for n, m in pipeline.unet.named_modules():
        if isinstance(m, CrossAttention) and n in wanted_layers:
            m.forward = hook_builder(n, m)
            hook_count += 1

    print(f"[DEBUG] wanted_layers={len(wanted_layers)}  hooks_loaded={hook_count}")


def save_heatmap_strip(heatmaps, hm_layers, step_tags):
    """
    Saves heatmap visualization strips.
    """
    header_h = 30
    final_clean = heatmaps[hm_layers[0]][-1] if heatmaps[hm_layers[0]] else None

    for ln, frames in heatmaps.items():
        if not frames:
            continue
        cols = frames + ([final_clean] if final_clean else [])

        img_w, img_h = cols[0].width, cols[0].height
        strip = Image.new("RGB", (img_w * len(cols), img_h + header_h), "black")
        draw = ImageDraw.Draw(strip)
        font = ImageFont.load_default()

        for idx, frm in enumerate(cols):
            x = idx * img_w
            strip.paste(frm, (x, header_h))
            label = ("Final" if idx >= len(step_tags) else f"S{step_tags[idx]}")
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.text((x + (img_w - tw) // 2, (header_h - th) // 2),
                     label, font=font, fill="white")

        safe_ln = ln.replace("/", "_").replace(".", "_")
        out_jpg = Path(DEBUG_DIR) / f"{safe_ln}_attn_hm.jpg"
        strip.save(out_jpg, quality=95)
        print(f"[DEBUG] heat-map strip saved → {out_jpg}")


def process_heatmap_visualization(pipeline, snapshot, latents, i, step_tags):
    """
    Processes heatmap visualization for debugging.
    """
    if not hasattr(pipeline, "_hm_layers"):
        pipeline._hm_layers = list(snapshot)
        pipeline._heatmaps = {ln: [] for ln in pipeline._hm_layers}
        pipeline._step_tags = []

    # Decode current latent → base RGB
    vae_dev = next(pipeline.vae.parameters()).device
    img = pipeline.vae.decode(
        (latents / pipeline.vae.config.scaling_factor)
        .to(device=vae_dev, dtype=pipeline.vae.dtype)
    ).sample[0]
    img_np = ((img.float() / 2 + 0.5)
              .clamp_(0, 1)
              .permute(1, 2, 0)
              .cpu()
              .numpy() * 255).astype(np.uint8)

    cmap = cm.get_cmap("jet")
    font = ImageFont.load_default()

    for ln in pipeline._hm_layers:
        if ln not in snapshot:
            continue
        amap = snapshot[ln]
        amap_n = amap / amap.max() if amap.max() > 0 else amap

        hmap = (cmap(amap_n)[..., :3] * 255).astype(np.uint8)
        hmap = np.array(Image.fromarray(hmap).resize((1024, 1024), Image.BILINEAR))
        heat_np = (0.5 * img_np + 0.5 * hmap).astype(np.uint8)
        heat_im = Image.fromarray(heat_np)

        # Numeric 5×5 grid
        draw = ImageDraw.Draw(heat_im)
        H_blk = amap.shape[0] // 5
        W_blk = amap.shape[1] // 5
        vis_h = 1024 // 5
        vis_w = 1024 // 5
        for bi in range(5):
            for bj in range(5):
                blk = amap[bi * H_blk:(bi + 1) * H_blk,
                          bj * W_blk:(bj + 1) * W_blk]
                mv = blk.mean()
                cx = bj * vis_w + vis_w // 2
                cy = bi * vis_h + vis_h // 2
                txt = f"{mv:.2f}"
                tw, th = draw.textbbox((0, 0), txt, font=font)[2:]
                draw.text((cx - tw // 2, cy - th // 2), txt,
                         font=font, fill="white",
                         stroke_width=1, stroke_fill="black")

        pipeline._heatmaps[ln].append(heat_im)

    pipeline._step_tags.append(i)


def prepare_reference_latents(pipeline, ref_latents_all, latents, face_mask_static, mask_4ch, device):
    """
    Prepares reference latents for branched attention.
    """
    B_total = 2 if pipeline.do_classifier_free_guidance else 1
    
    # Get the mask from face_mask_static
    mask_1 = torch.from_numpy(face_mask_static)[None, None].float().to(device)
    mask_bool = F.interpolate(mask_1, size=latents.shape[-2:], mode="nearest").bool()
    
    # Cache mask if needed
    if face_mask_static is not None:
        mask_bool = torch.from_numpy(face_mask_static).to(mask_bool.device).bool().unsqueeze(0).unsqueeze(0)
    
    # Build 4-channel mask
    if mask_4ch is None:
        mask_up = F.interpolate(mask_bool.float(), size=latents.shape[-2:], mode="nearest").bool()
        mask_4ch = mask_up.repeat(1, 4, 1, 1).clone()
    
    # Resize if needed
    if mask_4ch.shape[-2:] != latents.shape[-2:]:
        mask_up = F.interpolate(mask_4ch[:, :1].float(), size=latents.shape[-2:], mode="nearest").bool()
        mask_4ch = mask_up.repeat(1, 4, 1, 1)
    
    mask_4ch_expanded = mask_4ch.expand(B_total, -1, -1, -1).clone()
    
    # Compute face region statistics
    face_vals = ref_latents_all[0][mask_4ch[0]].float()
    face_mean = face_vals.mean()
    face_std = face_vals.std().clamp(min=1e-4)
    
    # Build normalized reference latents
    ref_latents_clean = torch.zeros_like(ref_latents_all).expand(B_total, -1, -1, -1).clone()
    norm_lat = ((ref_latents_all.float() - face_mean) / face_std).expand(B_total, -1, -1, -1).to(ref_latents_all.dtype)
    ref_latents_clean[mask_4ch_expanded] = norm_lat[mask_4ch_expanded]
    
    return ref_latents_clean, mask_4ch, mask_4ch_expanded


def save_branch_previews(pipeline, noise_face, noise_bg, latents, t, merge_mask_t, 
                        extra_step_kwargs, i, debug_save_face_branch, debug_save_bg_branch):
    """
    Saves preview images for face and background branches.
    """
    _saved_idx = getattr(pipeline.scheduler, "_step_index", None)
    
    def _step_and_decode(noise):
        lat = pipeline.scheduler.step(
            noise, t, latents.detach().clone(),
            **extra_step_kwargs, return_dict=False
        )[0]
        if _saved_idx is not None:
            pipeline.scheduler._step_index = _saved_idx
        vae_dev = next(pipeline.vae.parameters()).device
        img = pipeline.vae.decode(
            (lat / pipeline.vae.config.scaling_factor)
            .to(device=vae_dev, dtype=pipeline.vae.dtype)
        ).sample[0]
        img_np = (
            (img.float() / 2 + 0.5)
            .clamp_(0, 1)
            .permute(1, 2, 0)
            .cpu()
            .numpy() * 255
        ).astype("uint8")
        return img_np
    
    m_np_lat = None
    if hasattr(pipeline, "_merge_mask_t"):
        m_np_lat = merge_mask_t[0, 0].cpu().numpy().astype(bool)
    
    # Face branch
    if debug_save_face_branch:
        img_np = _step_and_decode(noise_face)
        if m_np_lat is not None:
            H, W = img_np.shape[:2]
            m_big = np.array(
                Image.fromarray(m_np_lat.astype(np.uint8) * 255)
                .resize((W, H), Image.NEAREST)
            ).astype(bool)
            img_np = img_np.copy()
        out_fb = os.path.join(DEBUG_DIR, f"face_branch_{i:03d}.png")
        Image.fromarray(img_np).save(out_fb)
        print(f"[DBG] face-branch preview saved → {out_fb}")
    
    # Background branch
    if debug_save_bg_branch:
        img_np = _step_and_decode(noise_bg)
        if m_np_lat is not None:
            H, W = img_np.shape[:2]
            m_big = np.array(
                Image.fromarray(m_np_lat.astype(np.uint8) * 255)
                .resize((W, H), Image.NEAREST)
            ).astype(bool)
            img_np = img_np.copy()
        out_bg = os.path.join(DEBUG_DIR, f"background_branch_{i:03d}.png")
        Image.fromarray(img_np).save(out_bg)
        print(f"[DBG] background-branch preview saved → {out_bg}")
        

def call_branched_attention(pipeline, **kwargs):
    """
    Main entry point for branched attention pipeline execution.
    This is called from PhotoMakerStableDiffusionXLPipeline when use_branched_attention=True.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from pathlib import Path
    from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
    from diffusers.utils import is_torch_xla_available
    
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
        XLA_AVAILABLE = True
    else:
        XLA_AVAILABLE = False

    # Extract all parameters
    prompt = kwargs.get('prompt')
    prompt_2 = kwargs.get('prompt_2')
    height = kwargs.get('height')
    width = kwargs.get('width')
    num_inference_steps = kwargs.get('num_inference_steps', 50)
    timesteps = kwargs.get('timesteps')
    sigmas = kwargs.get('sigmas')
    denoising_end = kwargs.get('denoising_end')
    guidance_scale = kwargs.get('guidance_scale', 5.0)
    negative_prompt = kwargs.get('negative_prompt')
    negative_prompt_2 = kwargs.get('negative_prompt_2')
    num_images_per_prompt = kwargs.get('num_images_per_prompt', 1)
    eta = kwargs.get('eta', 0.0)
    generator = kwargs.get('generator')
    latents = kwargs.get('latents')
    prompt_embeds = kwargs.get('prompt_embeds')
    negative_prompt_embeds = kwargs.get('negative_prompt_embeds')
    pooled_prompt_embeds = kwargs.get('pooled_prompt_embeds')
    negative_pooled_prompt_embeds = kwargs.get('negative_pooled_prompt_embeds')
    ip_adapter_image = kwargs.get('ip_adapter_image')
    ip_adapter_image_embeds = kwargs.get('ip_adapter_image_embeds')
    output_type = kwargs.get('output_type', 'pil')
    return_dict = kwargs.get('return_dict', True)
    cross_attention_kwargs = kwargs.get('cross_attention_kwargs')
    guidance_rescale = kwargs.get('guidance_rescale', 0.0)
    original_size = kwargs.get('original_size')
    crops_coords_top_left = kwargs.get('crops_coords_top_left', (0, 0))
    target_size = kwargs.get('target_size')
    negative_original_size = kwargs.get('negative_original_size')
    negative_crops_coords_top_left = kwargs.get('negative_crops_coords_top_left', (0, 0))
    negative_target_size = kwargs.get('negative_target_size')
    clip_skip = kwargs.get('clip_skip')
    callback_on_step_end = kwargs.get('callback_on_step_end')
    callback_on_step_end_tensor_inputs = kwargs.get('callback_on_step_end_tensor_inputs', ['latents'])
    
    # PhotoMaker specific
    input_id_images = kwargs.get('input_id_images')
    start_merge_step = kwargs.get('start_merge_step', 10)
    class_tokens_mask = kwargs.get('class_tokens_mask')
    id_embeds = kwargs.get('id_embeds')
    prompt_embeds_text_only = kwargs.get('prompt_embeds_text_only')
    pooled_prompt_embeds_text_only = kwargs.get('pooled_prompt_embeds_text_only')
    
    # Branched attention specific
    face_embed_strategy = kwargs.get('face_embed_strategy', 'heatmap')
    save_heatmaps = kwargs.get('save_heatmaps', False)
    heatmap_mode = kwargs.get('heatmap_mode', 'identity')
    focus_token = kwargs.get('focus_token', 'face')
    mask_mode = kwargs.get('mask_mode', 'spec')
    branched_attn_start_step = kwargs.get('branched_attn_start_step', 10)
    debug_save_masks = kwargs.get('debug_save_masks', False)
    mask_save_dir = kwargs.get('mask_save_dir', 'hm_debug')
    mask_interval = kwargs.get('mask_interval', 5)
    debug_save_face_branch = kwargs.get('debug_save_face_branch', True)
    debug_save_bg_branch = kwargs.get('debug_save_bg_branch', True)
    face_branch_interval = kwargs.get('face_branch_interval', 10)
    export_mask = kwargs.get('export_mask', False)
    import_mask = kwargs.get('import_mask')

    # Use pipeline's existing methods for the common setup
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs
    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._guidance_rescale = guidance_rescale
    pipeline._clip_skip = clip_skip
    pipeline._cross_attention_kwargs = cross_attention_kwargs
    pipeline._denoising_end = denoising_end
    pipeline._interrupt = False

    # Validate inputs
    if prompt_embeds is not None and class_tokens_mask is None:
        raise ValueError(
            "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed."
        )
    if input_id_images is None:
        raise ValueError(
            "Provide `input_id_images`. Cannot leave `input_id_images` undefined for PhotoMaker pipeline."
        )
    if not isinstance(input_id_images, list):
        input_id_images = [input_id_images]

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    # 3-7. Use pipeline's existing methods for prompt encoding and preparation
    lora_scale = (
       pipeline.cross_attention_kwargs.get("scale", None) if pipeline.cross_attention_kwargs is not None else None
    )
    
    num_id_images = len(input_id_images)
    (
        prompt_embeds, 
        _,
        pooled_prompt_embeds,
        _,
        class_tokens_mask,
    ) = pipeline.encode_prompt_with_trigger_word(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_id_images=num_id_images,
        class_tokens_mask=class_tokens_mask,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipeline.clip_skip,
    )

    # Encode prompt without trigger word
    tokens_text_only = pipeline.tokenizer.encode(prompt, add_special_tokens=False)
    trigger_word_token = pipeline.tokenizer.convert_tokens_to_ids(pipeline.trigger_word)
    tokens_text_only.remove(trigger_word_token)
    prompt_text_only = pipeline.tokenizer.decode(tokens_text_only, add_special_tokens=False)
    (
        prompt_embeds_text_only,
        negative_prompt_embeds,
        pooled_prompt_embeds_text_only,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt_text_only,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds_text_only,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds_text_only,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipeline.clip_skip,
    )

    # Prepare timesteps
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # Prepare ID images
    dtype = next(pipeline.id_encoder.parameters()).dtype
    if not isinstance(input_id_images[0], torch.Tensor):
        id_pixel_values = pipeline.id_image_processor(input_id_images, return_tensors="pt").pixel_values

    id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype)

    # Branched attention specific setup
    dtype_lat = next(pipeline.unet.parameters()).dtype
    pipeline._ref_latents_all = encode_face_latents(
        pipeline.vae,
        device,
        id_pixel_values,
        target_hw=(height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
        dtype=dtype_lat,
    )
    print(f"[DBG] VAE ref-latents ready  shape={pipeline._ref_latents_all.shape}  dtype={pipeline._ref_latents_all.dtype}")

    # Process ID embeddings based on strategy
    if face_embed_strategy.lower() == "heatmap":
        emb_dim = 512
        id_embeds_t = torch.zeros((1, num_id_images, emb_dim), device=device, dtype=dtype)
        print("[ID-DEBUG] Using zero-placeholder id_embeds for heat-map strategy")
    elif id_embeds is not None:
        id_embeds_t = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
        print(f"[ID-DEBUG pre-norm] ArcFace raw  shape={id_embeds_t.shape} dtype={id_embeds_t.dtype}")
    else:
        raise ValueError("`id_embeds` must be supplied when face_embed_strategy is not 'heatmap'")

    prompt_embeds = pipeline.id_encoder(
        id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds_t
    )

    pipeline._id_embed_vec = None
    if face_embed_strategy.lower() != "heatmap" and id_embeds is not None:
        pipeline._id_embed_vec = F.normalize(
            id_embeds.squeeze(0).float(), p=2, dim=-1
        ).detach().to(device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # Prepare latents
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Setup branched attention
    _branched_ready = False
    pipeline._face_mask_static = None
    pipeline._mask_4ch = None
    pipeline._merge_mask = None
    pipeline._freeze_mask = False

    # Handle mask import
    if import_mask:
        if not os.path.isfile(import_mask):
            raise FileNotFoundError(import_mask)
        m = Image.open(import_mask).convert("L")
        lat_hw = (height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)
        if m.size != (lat_hw[1], lat_hw[0]):
            m = m.resize((lat_hw[1], lat_hw[0]), Image.NEAREST)
        pipeline._merge_mask = (np.array(m) > 127).astype(np.uint8)
        pipeline._face_mask = pipeline._merge_mask
        pipeline._face_mask_static = pipeline._merge_mask
        pipeline._freeze_mask = True
        collect_hm = False
        print(f"[INFO] imported merge-mask  {import_mask}  {pipeline._merge_mask.shape}")
    else:
        collect_hm = True
        print("[INFO] no merge-mask imported; will build one from heat-maps")

    # Patch self-attention
    patch_self_attention_for_branched_mode(pipeline)

    # Setup attention hooks if needed
    attn_maps_current = {}
    orig_attn_forwards = {}
    if collect_hm:
        if hasattr(pipeline.unet, "attn_processors"):
            pipeline.unet.set_attn_processor(dict(pipeline.unet.attn_processors))
        setup_attention_hooks(pipeline, heatmap_mode, focus_token, class_tokens_mask,
                            attn_maps_current, orig_attn_forwards, device)

    # Prepare extra step kwargs
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

   # Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids
        
    if pipeline.do_classifier_free_guidance:
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            pipeline.do_classifier_free_guidance,
        )

    # Apply denoising_end
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    if (
        pipeline.denoising_end is not None
        and isinstance(pipeline.denoising_end, float)
        and pipeline.denoising_end > 0
        and pipeline.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                pipeline.scheduler.config.num_train_timesteps
                - (pipeline.denoising_end * pipeline.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # Get guidance scale embedding
    timestep_cond = None
    if pipeline.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipeline.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    pipeline._num_timesteps = len(timesteps)

    # Main denoising loop
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            # Standard denoising setup
            latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            if i <= start_merge_step:
                current_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                ) if pipeline.do_classifier_free_guidance else prompt_embeds_text_only
                add_text_embeds_current = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0
                ) if pipeline.do_classifier_free_guidance else pooled_prompt_embeds_text_only
            else:
                current_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                ) if pipeline.do_classifier_free_guidance else prompt_embeds
                add_text_embeds_current = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                ) if pipeline.do_classifier_free_guidance else pooled_prompt_embeds
                
            added_cond_kwargs = {"text_embeds": add_text_embeds_current, "time_ids": add_time_ids}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds

            # Process attention maps if collecting
            if collect_hm and i == start_merge_step - 1:
                attn_maps_current.clear()

            if (
                collect_hm
                and i >= start_merge_step
                and (i % mask_interval == 0 or i == len(timesteps) - 1)
                and attn_maps_current
            ):
                snapshot = {}
                for ln, lst in attn_maps_current.items():
                    lst2 = [m for m in lst if m.ndim == 2]
                    if not lst2:
                        continue
                    max_H = max(m.shape[0] for m in lst2)
                    aligned = [
                        m if m.shape[0] == max_H else _resize_map(m, max_H)
                        for m in lst2
                    ]
                    snapshot[ln] = np.stack(aligned, axis=0).mean(0)

                if snapshot and save_heatmaps:
                    process_heatmap_visualization(pipeline, snapshot, latents, i, [])

                if not pipeline._freeze_mask and pipeline._face_mask_static is None:
                    if mask_mode == "spec":
                        pipeline._face_mask = compute_binary_face_mask(snapshot, MASK_LAYERS_CONFIG)
                    else:
                        pipeline._face_mask = simple_threshold_mask(snapshot)
                    pipeline._face_mask_static = pipeline._face_mask

                attn_maps_current.clear()

            # Branched attention logic
            branched_now = (
                i >= branched_attn_start_step
                and hasattr(pipeline, "_face_mask")
                and pipeline._face_mask is not None
            )

            if branched_now:
                if not _branched_ready:
                    # Prepare reference latents
                    _ref_latents_clean, pipeline._mask_4ch, mask_4ch_expanded = prepare_reference_latents(
                        pipeline, pipeline._ref_latents_all, latents, 
                        pipeline._face_mask_static, pipeline._mask_4ch, device
                    )
                    pipeline._step_noise = torch.randn_like(_ref_latents_clean)
                    _branched_ready = True

                # Prepare merge mask
                if not hasattr(pipeline, "_merge_mask_t"):
                    if pipeline._merge_mask is None:
                        pipeline._merge_mask_t = torch.from_numpy(pipeline._face_mask)[None,None].float().to(device)
                    else:
                        _m = torch.from_numpy(pipeline._merge_mask)[None,None].float().to(latents.device)
                        pipeline._merge_mask_t = F.interpolate(_m, size=latents.shape[-2:], mode="nearest").bool()

                # Background branch
                dtype_lat = latents.dtype
                mask_bg = (~pipeline._mask_4ch).to(dtype_lat)
                mask_face = pipeline._mask_4ch.to(dtype_lat)
                
                bg_q = latent_model_input * mask_bg
                bg_kv = latent_model_input * mask_face
                pipeline._kv_override = bg_kv
                
                noise_bg = pipeline.unet(
                    bg_q,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Face branch
                kv_noised = pipeline._ref_latents_all.clone()
                face_q = latent_model_input * mask_face
                face_kv = kv_noised.to(dtype_lat) * mask_face
                pipeline._kv_override = face_kv
                
                noise_face = pipeline.unet(
                    face_q,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Save branch previews if needed
                do_dump = (
                    i % face_branch_interval == 0
                    or i == branched_attn_start_step
                    or i == len(timesteps) - 1
                )
                if do_dump and (debug_save_face_branch or debug_save_bg_branch):
                    save_branch_previews(pipeline, noise_face, noise_bg, latents, t, 
                                       pipeline._merge_mask_t, extra_step_kwargs, i,
                                       debug_save_face_branch, debug_save_bg_branch)

                # Merge predictions
                pipeline._kv_override = None
                noise_pred = torch.where(pipeline._merge_mask_t, noise_face, noise_bg)

            else:
                # Single-branch (original) call
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            # Perform guidance
            if pipeline.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if pipeline.do_classifier_free_guidance and pipeline.guidance_rescale > 0.0:
                from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipeline.guidance_rescale)

            # Compute the previous noisy sample
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

            # Call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()