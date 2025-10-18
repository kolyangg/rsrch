## photomaker/branched_v3.py
"""
Helpers for PhotoMaker‑XL branched attention upgrade
"""

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
    "two_branch_predict",
    "encode_face_latents",
    "patch_self_attention",
]


# ────────────────────────────────────────────────────────────────
#  Two-branch UNet prediction & merge
# ────────────────────────────────────────────────────────────────

def two_branch_predict(
    pipeline,
    latents: torch.Tensor,                 # current latents (before scheduler.step)
    latent_model_input: torch.Tensor,
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor | None,
    t: torch.Tensor,
    prompt_embeds_step: torch.Tensor,
    added_cond_kwargs: dict,
    timestep_cond: torch.Tensor | None,
    step_idx: int = 0,  # Add step index for debugging
    debug_dir: str | None = None,    # folder to drop visualisations
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements the spec:
      • BG-branch: Q=current background, K/V=current face
      • Face-branch: Q=current face, K/V=reference face
    Returns merged `noise_pred`.
    """
    
    DEBUG_NO_FACE = False  # Set to True to skip face branch (for debugging)
    
    B = latent_model_input.shape[0]
    dtype_lat = latent_model_input.dtype
    device = latent_model_input.device


    # Prepare 4-channel mask
    mask_4ch = mask4.repeat(1, 4, 1, 1).to(dtype_lat)
    mask_4ch_ref = mask4_ref.repeat(1, 4, 1, 1).to(dtype_lat)
    if mask_4ch.shape[0] < B:
        mask_4ch = mask_4ch.expand(B, -1, -1, -1)
    if mask_4ch_ref.shape[0] < B:
        mask_4ch_ref = mask_4ch_ref.expand(B, -1, -1, -1)
    
    # Debug save the mask if requested

    # every 10 steps
    if step_idx % 10 == 0:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            mask_np = mask_4ch_ref[0, 0].float().cpu().numpy().astype(bool)
            Image.fromarray(mask_np.astype(np.uint8) * 255).save(
                os.path.join(debug_dir, f"mask_ref_step{step_idx:03d}.png")
            )
            print(f"[Branch Debug] Saved mask ref for step {step_idx:03d} to {debug_dir}/mask_ref_step{step_idx:03d}.png")


    mask_noise_bg = 1.0 - mask_4ch  # 1 = background
    mask_noise_face = mask_4ch
    
    mask_ref_face = mask_4ch_ref      # 1 = face
    mask_ref_bg = 1.0 - mask_4ch_ref  # 1 = background


    # Get reference latents
    ref_latents = pipeline._ref_latents_all
   
    #  ###### ref_latents = pipeline._id_embeds  # Use ID embeds directly for K/V
   
    if ref_latents.shape[0] < B:
        ref_latents = ref_latents.expand(B, -1, -1, -1)
    
    # Add noise to reference to match current timestep
    if not hasattr(pipeline, "_ref_noise"):
        pipeline._ref_noise = torch.randn_like(ref_latents)
    
    ref_noised = pipeline.scheduler.add_noise(
        ref_latents, 
        pipeline._ref_noise[:ref_latents.shape[0]], 
        t.expand(ref_latents.shape[0])
    )
    

    # --- Branch 1: Background ---
    BG_EXPERIMENT = True  # Set to True to use experimental background branch
    
    # Store branch info and masks for attention layer
    if not BG_EXPERIMENT:
        pipeline._branch_mode = "background"
        pipeline._q_mask = mask_noise_bg  # Background mask for Q
        pipeline._kv_override = latent_model_input  # Full current as K/V

        noise_bg = pipeline.unet(
            latent_model_input,  # Pass FULL latents
            t,
            encoder_hidden_states=prompt_embeds_step,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    else:
        # Q-mask already set via pipeline._q_mask = mask_noise_bg
        pipeline._branch_mode = "background"
        pipeline._q_mask      = mask_noise_bg   # ensure Q attends only from bg

        # K/V = current face latents only (CFG-aware)
        if pipeline.do_classifier_free_guidance and B % 2 == 0:
            B_half = B // 2
            kv_bg = torch.cat([
                latent_model_input[:B_half] * mask_noise_face[:B_half],
                latent_model_input[B_half:] * mask_noise_face[B_half:]
            ], dim=0)
        else:
            kv_bg = latent_model_input * mask_noise_face
        pipeline._kv_override = kv_bg

        noise_bg = pipeline.unet(
            latent_model_input,  # Pass full latents; Q will be masked inside attention
            t,
            encoder_hidden_states=prompt_embeds_step,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    if not DEBUG_NO_FACE:
        # --- Branch 2: Face ---
        pipeline._branch_mode = "face"
        pipeline._q_mask = mask_noise_face  # Face mask for Q
        
        # For K/V: blend reference face with current background
        kv_face_blend = torch.where(
            mask_ref_face.bool(),
            ref_noised,  # Face from reference
            latent_model_input  # Background from current
        )
        pipeline._kv_override = kv_face_blend

        
        noise_fc = pipeline.unet(
            latent_model_input,  # Pass FULL latents
            t,
            encoder_hidden_states=prompt_embeds_step,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    
    # ---------------------------------------------------------------------
    # #  DEBUG Q/K/V VIS — every 10 steps (and first three for quick sanity)
    # # ---------------------------------------------------------------------
    # if debug_dir and (step_idx % 10 == 0 or step_idx < 3):
    #     os.makedirs(debug_dir, exist_ok=True)

    #     def _lat2rgb(z: torch.Tensor) -> np.ndarray:
    #         """Decode a *single-image* latent tensor → uint8 RGB."""
    #         z = z[:1]  # take first in batch
    #         z = z.to(device=next(pipeline.vae.parameters()).device,
    #                  dtype=pipeline.vae.dtype) / pipeline.vae.config.scaling_factor
    #         img = pipeline.vae.decode(z).sample[0]
    #         return (
    #             (img.float() / 2 + 0.5)
    #             .clamp(0, 1)
    #             .permute(1, 2, 0)
    #             .cpu()
    #             .numpy() * 255
    #         ).astype("uint8")

    #     # Q / K / V tensors
    #     # q_bg   = latent_model_input * mask_bg
    #     q_bg   = latent_model_input * mask_noise_bg
    #     k_bg   = current_face_kv                     # V same as K (without mask here: total noise)
        
    #     # q_face = latent_model_input * mask_face
    #     q_face = latent_model_input * mask_noise_face  # Q: face only of current noise
    #     k_face = ref_face_kv

    #     Image.fromarray(_lat2rgb(q_bg)).save  (os.path.join(debug_dir, f"step{step_idx:03d}_Q_BG.png"))
    #     Image.fromarray(_lat2rgb(k_bg)).save  (os.path.join(debug_dir, f"step{step_idx:03d}_K_BG.png"))
    #     Image.fromarray(_lat2rgb(q_face)).save(os.path.join(debug_dir, f"step{step_idx:03d}_Q_FC.png"))
    #     Image.fromarray(_lat2rgb(k_face)).save(os.path.join(debug_dir, f"step{step_idx:03d}_K_FC.png"))
    #     print(f"[Branch Debug] Saved Q/K visualisations for step {step_idx:03d} to {debug_dir}")

    # # ---------------------------------------------------------------------

    # Clear overrides
    pipeline._kv_override = None
    pipeline._branch_mode = None
    pipeline._q_mask = None
    
    # Merge results: use 4-channel mask for consistency
    
    if not DEBUG_NO_FACE:
        noise_pred = torch.where(mask_noise_face.bool(), noise_fc, noise_bg)
    else:
        noise_pred = noise_bg # TEMP: use only background branch for now
    
    # Debug output
    if step_idx < 3 or step_idx % 10 == 0:
        print(f"[Branch Debug] Step {step_idx}")
        print(f"  Input norm: {latent_model_input.std().item():.4f}")
        print(f"  Ref noised norm: {ref_noised.std().item():.4f}")
        print(f"  BG branch norm: {noise_bg.std().item():.4f}")
        if not DEBUG_NO_FACE:
            print(f"  Face branch norm: {noise_fc.std().item():.4f}")
        print(f"  Merged norm: {noise_pred.std().item():.4f}")
    

    if not DEBUG_NO_FACE:
        return noise_pred, noise_fc, noise_bg 
    else:
        return noise_pred, noise_bg, noise_bg # TEMP: use only background branch for now
    


# ────────────────────────────────────────────────────────────────
# 1)  Robust reference latents for branched attention
# ────────────────────────────────────────────────────────────────

def encode_face_latents(
    pipeline,
    id_pixel_values: torch.Tensor,  # (1, N, 3, H, W) or (1,3,H,W)
    target_hw: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return *normalised* VAE latents for **the first** ID image.

    The encode always runs in the VAE's own dtype/device for stability;
    the resulting tensor is then resized and cast to *dtype* on the
    caller's device.
    """
    # Ensure input is (1,3,H,W)
    if id_pixel_values.dim() == 5:   # (B,N,3,H,W)
        ref_img = id_pixel_values[0, 0]
    else:
        ref_img = id_pixel_values[0]

    vae_weight = next(pipeline.vae.parameters())
    vae_dtype  = vae_weight.dtype
    vae_dev    = vae_weight.device

    ref_img = ref_img.unsqueeze(0).to(device=vae_dev, dtype=vae_dtype)

    with torch.no_grad():
        z = pipeline.vae.encode(ref_img).latent_dist.mode() * pipeline.vae.config.scaling_factor

    # Resize to target resolution
    z = F.interpolate(z.float(), size=target_hw, mode="bilinear")
    
    # Normalize - this is critical for stable attention
    z = (z - z.mean()) / z.std().clamp(min=1e-4)
    return z.to(device=pipeline.device, dtype=dtype).detach()


# ────────────────────────────────────────────────────────────────
# 2)  Self‑attention monkey‑patch (idempotent)
# ────────────────────────────────────────────────────────────────

def patch_self_attention(pipeline) -> int:
    """Enhanced self-attention patch that supports masked Q for branches.
    
    The attention layer will check for:
    - pipeline._branch_mode: "background" or "face"
    - pipeline._q_mask: mask to apply to Q (spatial)
    - pipeline._kv_override: replacement K/V source
    """

    from diffusers.models.attention_processor import Attention as _CrossAttn

    if getattr(pipeline, "_self_attn_patched", False):
        return 0  # already done

    def custom_attn_forward(mod, hidden_states, *, encoder_hidden_states=None, attention_mask=None):
        # Cross-attention: use default
        if encoder_hidden_states is not None or getattr(pipeline, "_kv_override", None) is None:
            return mod._orig_forward(hidden_states, encoder_hidden_states, attention_mask)

        import math
        import torch.nn.functional as F
        
        B, L, C = hidden_states.shape
        h = mod.heads
        d = C // h
        scale = 1 / math.sqrt(d)
        
        # Get Q mask if available
        q_mask = getattr(pipeline, "_q_mask", None)
        
        # Project Q from hidden states
        q_full = (mod.to_q if hasattr(mod, "to_q") else mod.q_proj)(hidden_states)
        
    
        # Build the actual query tensor by interpolating 4× mask to each attention block's size
        if q_mask is not None:
            # attention has L=H1*W1 tokens
            H1 = W1 = int(math.sqrt(L))
            # take channel 0, interpolate to (H1,W1)
            q_ch = q_mask[:, 0:1, :, :]
            q_resized = F.interpolate(q_ch, size=(H1, W1), mode="nearest")
            # flatten to (B,L)
            q_mask_spatial = q_resized.view(B, L)
            # zero out Q vectors outside  mask
            q_masked = q_full * q_mask_spatial.unsqueeze(-1)
            # reshape for multi-head attention
            q = q_masked.view(B, L, h, d).permute(0, 2, 1, 3)
        else:
            # no mask → standard Q
            q = q_full.view(B, L, h, d).permute(0, 2, 1, 3)
        
        # Use override for K/V
        kv_source = pipeline._kv_override
        B_kv, C_kv, H_kv, W_kv = kv_source.shape
        L_kv = H_kv * W_kv

        # Reshape for attention
        kv_flat = kv_source.permute(0, 2, 3, 1).reshape(B_kv, L_kv, C_kv)
        
        # Adjust channels if needed
        K_IN = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj).weight.shape[1]
        
        if kv_flat.shape[-1] < K_IN:
            kv_flat = F.pad(kv_flat, (0, K_IN - kv_flat.shape[-1]))
        elif kv_flat.shape[-1] > K_IN:
            kv_flat = kv_flat[..., :K_IN]
        
        # Project K and V
        k = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj)(kv_flat)
        v = (mod.to_v if hasattr(mod, "to_v") else mod.v_proj)(kv_flat)

        k = k.view(B_kv, L_kv, h, d).permute(0, 2, 1, 3)
        v = v.view(B_kv, L_kv, h, d).permute(0, 2, 1, 3)
        
        # Compute attention
        attn = (q @ k.transpose(-1, -2)) * scale
            
        # # Apply attention mask if we have one
        # if attn_bias is not None:
        #     # Mask out attention FROM positions where Q is masked
        #     # This prevents masked positions from influencing the output
        #     attn = attn * attn_bias + (1 - attn_bias) * -1e9
                    
        attn = attn.softmax(dim=-1)
        
        # Apply dropout if needed (usually during training)
        if hasattr(mod, "dropout") and mod.dropout > 0:
            attn = F.dropout(attn, p=mod.dropout, training=mod.training)        

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)

        # Apply output projections
        out = mod.to_out[0](out)
        out = mod.to_out[1](out)
        return out

    patched = 0
    for name, m in pipeline.unet.named_modules():
        if isinstance(m, _CrossAttn) and not getattr(m, "is_cross_attention", False) and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward
            m.forward = types.MethodType(custom_attn_forward, m)
            patched += 1
    pipeline._self_attn_patched = True
    return patched


