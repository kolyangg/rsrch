"""
PuLID/pulid/branched_NS.py

PuLID-specific branched attention utilities:
- Patch ONLY self-attention (attn1.processor) with PhotoMaker's BranchedAttnProcessor
  so we don't disturb PuLID's cross-attention (ID adapters) or Lightning specifics.
- Provide a two-branch predict that doubles latents and reuses the same prompts
  (no face-prompt doubling) while applying self-attention gating via masks.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import os

import torch

from photomaker.attn_processor import BranchedAttnProcessor


def patch_unet_self_attention_processors(
    pipeline,
    mask: torch.Tensor,
    mask_ref: torch.Tensor,
    id_embeds: Optional[torch.Tensor] = None,
    scale: float = 1.0,
):
    """Replace only attn1 (self-attn) processors with BranchedAttnProcessor.
    Keep attn2 (cross-attn) processors as-is (PuLID uses ID adapters there).
    """
    if BranchedAttnProcessor is None:
        return

    # Save originals once
    if not hasattr(pipeline, '_pm_orig_attn1'):
        pipeline._pm_orig_attn1 = {}
    
    cross_attention_dim = pipeline.unet.config.cross_attention_dim
    if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = cross_attention_dim[0]

    new_procs = {}
    for name, proc in pipeline.unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            if name not in pipeline._pm_orig_attn1:
                pipeline._pm_orig_attn1[name] = proc
            # hidden size by block
            if "mid_block" in name:
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
            else:
                hidden_size = pipeline.unet.config.block_out_channels[0]
            br = BranchedAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=hidden_size,
                scale=scale,
            ).to(pipeline.device, dtype=pipeline.unet.dtype)
            br.set_masks(mask, mask_ref)
            # Avoid diffusers warnings about cross_attention_kwargs
            try:
                setattr(br, 'has_cross_attention_kwargs', True)
            except Exception:
                pass
            # propagate runtime flags if present
            for k in ("pose_adapt_ratio", "ca_mixing_for_face"):
                if hasattr(pipeline, k):
                    setattr(br, k, getattr(pipeline, k))
            if id_embeds is not None:
                br.id_embeds = id_embeds.to(pipeline.device, dtype=pipeline.unet.dtype)
            new_procs[name] = br
        else:
            new_procs[name] = proc

    pipeline.unet.set_attn_processor(new_procs)


def restore_self_attention_processors(pipeline):
    if hasattr(pipeline, '_pm_orig_attn1') and pipeline._pm_orig_attn1:
        procs = dict(pipeline.unet.attn_processors)
        for name, orig in pipeline._pm_orig_attn1.items():
            if name in procs:
                procs[name] = orig
        pipeline.unet.set_attn_processor(procs)
        pipeline._pm_orig_attn1.clear()
        return True
    return False


def two_branch_predict_pulid(
    pipeline,
    latent_model_input: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, torch.Tensor],
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor,
    reference_latents: torch.Tensor,
    cross_attention_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    id_embeds: Optional[torch.Tensor] = None,
    step_idx: int = 0,
    debug_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Double the latents [noise, ref-noise], keep the same prompts for both halves,
    patch only self-attn with masks, and run a single UNet forward. Return merged half.
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype

    # Prepare ref-noised latents via scheduler
    # Use native add_noise if available, else fallback to scale-free copy
    if hasattr(pipeline.scheduler, 'add_noise'):
        # Ensure timestep tensor has correct shape
        tt = t if torch.is_tensor(t) else torch.tensor([t], device=device)
        if tt.ndim == 0:
            tt = tt.unsqueeze(0)
        tt = tt.expand(reference_latents.shape[0])
        ref_noised = pipeline.scheduler.add_noise(
            reference_latents.to(device=device, dtype=dtype),
            torch.randn_like(reference_latents, dtype=dtype, device=device),
            tt,
        )
    else:
        ref_noised = reference_latents.to(device=device, dtype=dtype)

    # Double latents: ensure ref_noised matches CFG-doubled latent batch
    if ref_noised.shape[0] != latent_model_input.shape[0]:
        ref_noised = ref_noised.expand(latent_model_input.shape[0], -1, -1, -1)
    batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)

    # Double prompts and time ids to match doubled latents
    enc_states = torch.cat([prompt_embeds, prompt_embeds], dim=0)
    doubled_kwargs = {}
    for k, v in added_cond_kwargs.items():
        doubled_kwargs[k] = torch.cat([v, v], dim=0) if torch.is_tensor(v) else v

    # Patch self-attn only (temporary)
    patch_unet_self_attention_processors(pipeline, mask4, mask4_ref, id_embeds=id_embeds)

    # Prepare timestep for doubled batch
    t_b = t if torch.is_tensor(t) else torch.tensor([t], device=device)
    if t_b.ndim == 0:
        t_b = t_b.unsqueeze(0)
    t_b = t_b.expand(batched_latents.shape[0])

    # Prepare cross-attention kwargs; duplicate id embedding for doubled batch
    x_attn_kwargs = dict(cross_attention_kwargs or {})
    try:
        b_in = batched_latents.shape[0]
        if 'id_embedding' in x_attn_kwargs and torch.is_tensor(x_attn_kwargs['id_embedding']):
            id_emb = x_attn_kwargs['id_embedding']
            if id_emb.shape[0] != b_in:
                if id_emb.shape[0] == 1:
                    id_emb = id_emb.expand(b_in, *id_emb.shape[1:])
                elif b_in % id_emb.shape[0] == 0:
                    factor = b_in // id_emb.shape[0]
                    id_emb = id_emb.repeat_interleave(factor, dim=0)
                else:
                    # safe fallback: expand to b_in
                    id_emb = id_emb.expand(b_in, *id_emb.shape[1:])
                x_attn_kwargs['id_embedding'] = id_emb
        if 'id_scale' in x_attn_kwargs and torch.is_tensor(x_attn_kwargs['id_scale']):
            ids = x_attn_kwargs['id_scale']
            if ids.shape[0] != b_in:
                if ids.shape[0] == 1:
                    ids = ids.expand(b_in, *ids.shape[1:])
                elif b_in % ids.shape[0] == 0:
                    factor = b_in // ids.shape[0]
                    ids = ids.repeat_interleave(factor, dim=0)
                else:
                    ids = ids.expand(b_in, *ids.shape[1:])
                x_attn_kwargs['id_scale'] = ids
    except Exception:
        # Non-fatal; proceed with original kwargs if adjustment fails
        pass

    # Forward UNet
    try:
        noise_pred = pipeline.unet(
            batched_latents,
            t_b,
            encoder_hidden_states=enc_states,
            added_cond_kwargs=doubled_kwargs,
            cross_attention_kwargs=x_attn_kwargs,
            return_dict=False,
        )[0]
    finally:
        # Always restore original processors to avoid breaking non-doubled passes
        try:
            restore_self_attention_processors(pipeline)
        except Exception:
            pass

    # --- Debug export of reference latents + mask overlay (if requested) ---
    # Determine debug output directory
    _dbg_dir = debug_dir or getattr(pipeline, "_debug_dir", None)
    # Fallback to a generic folder if nothing provided
    if _dbg_dir is None:
        _dbg_dir = os.path.join(os.getcwd(), "pulid_debug")
    try:
        # Import PhotoMaker debug helpers lazily
        from photomaker.branch_helpers import (
            save_debug_ref_latents as _pm_save_ref_latents,
            save_debug_ref_mask_overlay as _pm_save_ref_mask_overlay,
        )
        os.makedirs(_dbg_dir, exist_ok=True)
        # Expose reference latents on pipeline as expected by helpers
        setattr(pipeline, "_reference_latents", reference_latents.detach().clone())
        # Save decoded reference latents image once
        _pm_save_ref_latents(pipeline, _dbg_dir)
        # Save overlay with provided ref mask
        _pm_save_ref_mask_overlay(pipeline, mask4_ref, _dbg_dir)
    except Exception as e:
        # Non-fatal: just print a short message
        print(f"[BrNS] Debug export skipped: {e}")

    # Return only the first half (merged branch)
    B = noise_pred.shape[0] // 2
    return noise_pred[:B]
