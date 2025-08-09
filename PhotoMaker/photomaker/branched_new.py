"""
branched_new.py - Optimized branched attention implementation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import os
from PIL import Image


def patch_unet_attention_processors(
    pipeline,
    mask: torch.Tensor,
    mask_ref: torch.Tensor, 
    reference_latents: torch.Tensor,
    face_prompt_embeds: torch.Tensor,
    timestep,
    step_idx: int = 0,
    scale: float = 1.0,
    latent_model_input: torch.Tensor | None = None,
):
    """
    Patch UNet with custom branched attention processors.
    Creates processors on first call, updates them on subsequent calls.
    
    Args:
        pipeline: Pipeline containing the UNet
        mask: Face mask for noise [B, C, H, W]
        mask_ref: Face mask for reference [B, C, H, W]  
        reference_latents: Reference image latents [B, 4, H, W]
        face_prompt_embeds: ID embeddings for face branch
        step_idx: Current denoising step
        scale: Attention scale factor
    """
    from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
    
    # Store original processors once
    if not hasattr(pipeline, '_original_attn_processors'):
        pipeline._original_attn_processors = {}
        for name, proc in pipeline.unet.attn_processors.items():
            pipeline._original_attn_processors[name] = proc
    
    # Check if already using branched processors
    current_procs = pipeline.unet.attn_processors
    has_branched = any(
        isinstance(p, (BranchedAttnProcessor, BranchedCrossAttnProcessor)) 
        for p in current_procs.values()
    )
    
    if not has_branched:
        # First time: create new processors
        new_procs = {}
        
        # Get cross-attention dimensions from UNet config
        cross_attention_dim = pipeline.unet.config.cross_attention_dim
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = cross_attention_dim[0]
        
        for name in pipeline.unet.attn_processors.keys():
            # Extract layer info from name
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
            
            if name.endswith("attn1.processor"):
                # Self-attention: use branched processor
                proc = BranchedAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=hidden_size,
                    scale=scale,
                ).to(reference_latents.device)
                proc = proc.to(dtype=pipeline.unet.dtype, device=pipeline.unet.device)  # Move to correct dtype/device
                # proc.set_masks(mask, mask_ref)
                # proc.set_reference(reference_latents)
                proc.set_masks(mask, mask_ref)
                # match timestep: add noise and keep only ref-face region
                # ensure noise & proper device/shape for timestep
                # if not hasattr(pipeline, "_ref_noise") or pipeline._ref_noise.shape != reference_latents.shape:
                #     pipeline._ref_noise = torch.randn_like(reference_latents)
                # t = timestep
                # if not torch.is_tensor(t):
                #     t = torch.tensor([t], device=reference_latents.device, dtype=torch.long)
                # if t.ndim == 0:
                #     t = t[None]
                # t = t.to(reference_latents.device)
                # if t.shape[0] != reference_latents.shape[0]:
                #     t = t.expand(reference_latents.shape[0])
                # ref_noised = pipeline.scheduler.add_noise(
                #     reference_latents,
                #     pipeline._ref_noise[: reference_latents.shape[0]],
                #     t,
                # )

                # # Match ref to current-noise stats to stabilize face K/V
                # if latent_model_input is not None:
                #     cur_m = latent_model_input.mean(dim=(1,2,3), keepdim=True)
                #     cur_s = latent_model_input.std (dim=(1,2,3), keepdim=True).clamp_min(1e-5)
                #     ref_m = ref_noised.mean(dim=(1,2,3), keepdim=True)
                #     ref_s = ref_noised.std (dim=(1,2,3), keepdim=True).clamp_min(1e-5)
                #     ref_noised = (ref_noised - ref_m) / ref_s * cur_s + cur_m
                #     ref_noised = ref_noised.clamp(cur_m - 3*cur_s, cur_m + 3*cur_s)

                # ## NEW!
                # # Normalize ref to current noise stats to avoid attention blow-ups
                # with torch.no_grad():
                #     cur_m = latent_model_input.mean(dim=(1,2,3), keepdim=True)
                #     cur_s = latent_model_input.std (dim=(1,2,3), keepdim=True).clamp_min(1e-5)
                #     ref_m = ref_noised.mean(dim=(1,2,3), keepdim=True)
                #     ref_s = ref_noised.std (dim=(1,2,3), keepdim=True).clamp_min(1e-5)
                #     ref_noised = (ref_noised - ref_m) / ref_s * cur_s + cur_m
                #     ref_noised = ref_noised.clamp(cur_m - 3*cur_s, cur_m + 3*cur_s)

   
                # if mask_ref is not None:
                #     ref_noised = ref_noised * mask_ref
                # proc.set_reference(ref_noised)


                # Add noise to reference to match current timestep
                if not hasattr(pipeline, "_ref_noise"):
                    pipeline._ref_noise = torch.randn_like(reference_latents)
                
                t = timestep
                if not torch.is_tensor(t):
                    t = torch.tensor([t], device=reference_latents.device, dtype=torch.long)
                if t.ndim == 0:
                    t = t[None]
                t = t.to(reference_latents.device)
                
                # Use same scheduler as main pipeline
                ref_noised = pipeline.scheduler.add_noise(
                    reference_latents,
                    pipeline._ref_noise[:reference_latents.shape[0]],
                    t.expand(reference_latents.shape[0])
                )
                
                # DON'T normalize too aggressively - just clamp extremes
                # ref_std = ref_noised.std()
                # if ref_std > 0:
                #     ref_noised = ref_noised.clamp(-3*ref_std, 3*ref_std)
                
                # Apply mask to reference AFTER noising
                if mask_ref is not None:
                    mask_ref_4ch = mask_ref.repeat(1, 4, 1, 1) if mask_ref.shape[1] == 1 else mask_ref
                    mask_ref_4ch = mask_ref_4ch.to(dtype=ref_noised.dtype, device=ref_noised.device)
                    # Keep only face region in reference
                    ref_noised = ref_noised * mask_ref_4ch
                 
                
                # proc.set_reference(ref_noised)
                # Don't set reference - it will be passed as batched input
                
                # Pass face embeddings correctly
                if face_prompt_embeds is not None:
                    proc.set_face_embeds(face_prompt_embeds)
                
                # Apply mask to keep only face region
                if mask_ref is not None:
                    mask_ref_4ch = mask_ref.repeat(1, 4, 1, 1) if mask_ref.shape[1] == 1 else mask_ref
                    ref_noised = ref_noised * mask_ref_4ch
                
                # proc.set_reference(ref_noised)



                # Set face embeddings based on strategy
                if hasattr(pipeline, 'face_embed_strategy'):
                    if pipeline.face_embed_strategy == "id_embeds" and hasattr(pipeline, '_face_prompt_embeds'):
                        proc.set_face_embeds(pipeline._face_prompt_embeds)
                    elif pipeline.face_embed_strategy == "face":
                        # For "face" strategy, use encoded face caption
                        proc.set_face_embeds(face_prompt_embeds)
                
                else:
                    # Fallback: no face embeddings
                    print("Warning - no face embed strategy, using None")
                    proc.set_face_embeds(None)

                new_procs[name] = proc
                
            # elif name.endswith("attn2.processor"):
            #     # Cross-attention: use branched cross processor
            #     num_tokens = 77  # Standard CLIP text token count
            #     if hasattr(pipeline, 'tokenizer') and hasattr(pipeline.tokenizer, 'model_max_length'):
            #         num_tokens = pipeline.tokenizer.model_max_length

            elif name.endswith("attn2.processor"):
                num_tokens = 77  # Standard CLIP text token count
                if hasattr(pipeline, 'tokenizer') and hasattr(pipeline.tokenizer, 'model_max_length'):
                    num_tokens = pipeline.tokenizer.model_max_length
                # Cross-attention: branched (BG→gen prompt, FACE→face prompt)
                proc = BranchedCrossAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale,
                    num_tokens=num_tokens,
                ).to(reference_latents.device)
                proc.set_masks(mask, mask_ref)
                if face_prompt_embeds is not None:
                    proc.set_face_prompt(face_prompt_embeds.to(reference_latents.device))
                new_procs[name] = proc
            else:
                # Keep original processor
                new_procs[name] = pipeline._original_attn_processors[name]
        
        pipeline.unet.set_attn_processor(new_procs)
        
    else:
        # Update existing branched processors
        for name, proc in pipeline.unet.attn_processors.items():
            if isinstance(proc, BranchedAttnProcessor):
                proc.set_masks(mask, mask_ref)
                # proc.set_reference(reference_latents)

                # # Update face embeddings
                # if hasattr(pipeline, 'face_embed_strategy'):
                #     if pipeline.face_embed_strategy == "id_embeds" and hasattr(pipeline, '_id_embeds'):
                #         proc.set_face_embeds(pipeline._id_embeds)
                #     elif pipeline.face_embed_strategy == "face" and face_prompt_embeds is not None:
                #         proc.set_face_embeds(face_prompt_embeds)
                
                # Update face embeddings
                if hasattr(pipeline, 'face_embed_strategy'):
                    if pipeline.face_embed_strategy == "id_embeds" and hasattr(pipeline, '_face_prompt_embeds'):
                        proc.set_face_embeds(pipeline._face_prompt_embeds)
                    elif pipeline.face_embed_strategy == "face":
                        proc.set_face_embeds(face_prompt_embeds)

            # Note: BranchedCrossAttnProcessor doesn't need per-step updates
            # as it uses the face_prompt_embeds passed through encoder_hidden_states


def restore_original_processors(pipeline):
    """Restore original attention processors."""
    if hasattr(pipeline, '_original_attn_processors'):
        pipeline.unet.set_attn_processor(pipeline._original_attn_processors)
        return True
    return False


def two_branch_predict(
    pipeline,
    latent_model_input: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor,
    reference_latents: torch.Tensor,
    face_prompt_embeds: Optional[torch.Tensor] = None,
    step_idx: int = 0,
    scale: float = 1.0,
    timestep_cond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute two-branch attention prediction using patched processors.
    
    Returns:
        noise_pred: Merged prediction
        noise_face: Face branch output (for debugging)
        noise_bg: Background branch output (for debugging)
    """
    batch_size = latent_model_input.shape[0]
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    
    # Batch noise and reference latents together
    if reference_latents is not None:
        # Ensure reference latents match batch size and add noise
        ref_noised = reference_latents  # Already noised in patch_unet_attention_processors
        if ref_noised.shape[0] < batch_size:
            ref_noised = ref_noised.expand(batch_size, -1, -1, -1)
        # Concatenate: [noise_latents, reference_latents]
        batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)
    else:
        batched_latents = latent_model_input
        print("no reference latents")

    # Patch processors with current step data
    patch_unet_attention_processors(
        pipeline, mask4, mask4_ref, reference_latents,
        face_prompt_embeds, t, step_idx, scale,
        latent_model_input=latent_model_input,
    )
    
    # Prepare encoder hidden states
    # if face_prompt_embeds is not None and prompt_embeds.shape[1] < face_prompt_embeds.shape[1]:
    #     # Concatenate text and face embeddings
    #     encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_embeds], dim=1)
    
    if face_prompt_embeds is not None:
        # For cross-attention: batch [generation_prompt, face_prompt]
        
        # # Ensure face_prompt_embeds has correct shape [B, seq_len, dim]
        # if face_prompt_embeds.dim() == 2:
        #     face_prompt_embeds = face_prompt_embeds.unsqueeze(0)
        
        # # Match sequence length with prompt_embeds
        # if face_prompt_embeds.shape[1] != prompt_embeds.shape[1]:
        #     # Pad or truncate to match
        #     target_len = prompt_embeds.shape[1]
        #     if face_prompt_embeds.shape[1] < target_len:
        #         padding = target_len - face_prompt_embeds.shape[1]
        #         face_prompt_embeds = F.pad(face_prompt_embeds, (0, 0, 0, padding))
        #     else:
        #         face_prompt_embeds = face_prompt_embeds[:, :target_len, :]
        
        # Fix shape if needed - face_prompt_embeds might be [B, dim] or [B, 1, dim]
        if face_prompt_embeds.dim() == 2:
            # Expand to match prompt_embeds shape [B, seq_len, dim]
            B, dim = face_prompt_embeds.shape
            seq_len = prompt_embeds.shape[1]  # Should be 77 for CLIP
            # Repeat the embedding across all token positions
            face_prompt_embeds = face_prompt_embeds.unsqueeze(1).expand(B, seq_len, dim)
        elif face_prompt_embeds.shape[1] == 1:
            # [B, 1, dim] -> [B, seq_len, dim]
            B, _, dim = face_prompt_embeds.shape
            seq_len = prompt_embeds.shape[1]
            face_prompt_embeds = face_prompt_embeds.expand(B, seq_len, dim)
            
        
        # Project face embeddings to match text embedding dimension
        if face_prompt_embeds.shape[-1] != prompt_embeds.shape[-1]:
            # Create a linear projection if dimensions don't match
            in_dim = face_prompt_embeds.shape[-1]
            out_dim = prompt_embeds.shape[-1]
            # Simple projection: pad with zeros or use a learned projection
            if in_dim < out_dim:
                # Pad with zeros to match dimension
                padding = out_dim - in_dim
                face_prompt_embeds = F.pad(face_prompt_embeds, (0, padding))
            else:
                # Truncate if larger (shouldn't happen)
                face_prompt_embeds = face_prompt_embeds[..., :out_dim]
        
        # Now face_prompt_embeds should be [B, 77, dim] matching prompt_embeds
        
          
        if pipeline.do_classifier_free_guidance:
            # Handle CFG: [uncond, cond] for both prompts
            B_half = prompt_embeds.shape[0] // 2
            face_prompt_batched = torch.cat([
               face_prompt_embeds[:B_half],  # uncond face
                face_prompt_embeds[B_half:]   # cond face
            ], dim=0)
        else:
            face_prompt_batched = face_prompt_embeds
        # Batch: [generation_prompt, face_prompt]
        encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_batched], dim=0)
    
    
    else:
        encoder_hidden_states = prompt_embeds
        print('no face embeds')
    
    # # Handle timestep
    # if t.dim() == 0:
    #     t = t.expand(batch_size)
        
    
    # # Expand timestep to match batched latents size
    
    # # Expand timestep to match batched latents size
    # if reference_latents is not None:
    #     # We have doubled the batch with [noise, reference]
    #     t = torch.cat([t, t], dim=0)  # Duplicate timesteps for reference batch
        
    # --- Ensure timesteps length matches the ACTUAL input batch (batched_latents) ---
    b_in = batched_latents.shape[0]
    t_in = t
    if not torch.is_tensor(t_in):
        t_in = torch.tensor(t_in, device=batched_latents.device)
    t_in = t_in.to(device=batched_latents.device)
    if t_in.ndim == 0:
        t_in = t_in.expand(b_in)
    elif t_in.shape[0] != b_in:
        # keep value, fix length
        t_in = t_in[:1].expand(b_in)
    
    # # Expand timestep to match batched latents size
    # if reference_latents is not None:
    #     # We have doubled the batch with [noise, reference]
    #     t = torch.cat([t, t], dim=0)  # Duplicate timesteps for reference batch
    
    
    # Build timestep to exactly match input batch (avoid cat on 0-D)
    b_in = batched_latents.shape[0]
    t_in = t
    if not torch.is_tensor(t_in):
        t_in = torch.tensor(t_in, device=batched_latents.device)
    t_in = t_in.to(device=batched_latents.device)
    if t_in.ndim == 0:
        t_in = t_in.expand(b_in)
    elif t_in.shape[0] != b_in:
        t_in = t_in[:1].expand(b_in)
    
    # Single forward pass - processors handle branching internally
    
    # --- make added_cond_kwargs match UNet batch ---
    ack = added_cond_kwargs
    if isinstance(ack, dict):
        b = batched_latents.shape[0]
        new_ack = {}
        for k, v in ack.items():
            if torch.is_tensor(v) and v.shape[:1] != (b,):
                if v.shape[0] == 1:
                    v = v.expand(b, *v.shape[1:])
                else:
                    repeat = b // v.shape[0]
                    v = v.repeat_interleave(repeat, dim=0) if repeat >= 1 else v[:1].expand(b, *v.shape[1:])
            new_ack[k] = v
        ack = new_ack
    
    noise_pred = pipeline.unet(
        # latent_model_input,
        batched_latents,
        t_in,
        encoder_hidden_states=encoder_hidden_states,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=getattr(pipeline, 'cross_attention_kwargs', None),
        added_cond_kwargs=ack,
        return_dict=False,
    )[0]
    
    # Extract only the noise prediction (first half of batch)
    if reference_latents is not None:
        noise_pred = noise_pred[:batch_size]
    
    # Extract branch outputs for debugging (approximate)
    mask_4ch = mask4.to(dtype=dtype)
    noise_bg = noise_pred * (1 - mask_4ch)
    noise_face = noise_pred * mask_4ch
    
    # Debug logging
    if step_idx < 3 or step_idx % 10 == 0:
        print(f"[Branched] Step {step_idx}: "
              f"norm={noise_pred.std().item():.4f}, "
              f"face={noise_face.std().item():.4f}, "
              f"bg={noise_bg.std().item():.4f}")
    
    return noise_pred, noise_face, noise_bg


def prepare_reference_latents(
    pipeline,
    reference_image: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Encode reference image to latents.
    
    Args:
        pipeline: Pipeline with VAE
        reference_image: Reference image tensor [B, C, H, W] or PIL Image
        height: Target height in pixels
        width: Target width in pixels
        dtype: Target dtype
        generator: Random generator for VAE sampling
        
    Returns:
        Encoded and normalized latents [B, 4, H//8, W//8]
    """
    device = pipeline.device
    vae = pipeline.vae
    
    # Convert PIL to tensor if needed
    if isinstance(reference_image, Image.Image):
        reference_image = pipeline.feature_extractor(
            reference_image, return_tensors="pt"
        ).pixel_values[0]
    
    # Ensure correct shape
    if reference_image.dim() == 3:
        reference_image = reference_image.unsqueeze(0)
    
    # Move to VAE device/dtype for encoding
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    reference_image = reference_image.to(device=vae_device, dtype=vae_dtype)
    
    # Encode
    with torch.no_grad():
        latents = vae.encode(reference_image).latent_dist.sample(generator)
        latents = latents * vae.config.scaling_factor
    
    # Resize if needed
    target_h = height // pipeline.vae_scale_factor
    target_w = width // pipeline.vae_scale_factor
    
    if latents.shape[2] != target_h or latents.shape[3] != target_w:
        latents = F.interpolate(
            latents.float(),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    
    # Normalize for stable attention
    latents = (latents - latents.mean()) / latents.std().clamp(min=1e-4)
    
    return latents.to(device=device, dtype=dtype)


def encode_face_prompt(
    pipeline,
    face_caption: str,
    device: torch.device,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Encode face-specific text prompt.
    
    Args:
        pipeline: Pipeline with text encoders
        face_caption: Text description for face
        device: Target device
        num_images_per_prompt: Number of images to generate
        do_classifier_free_guidance: Whether using CFG
        
    Returns:
        Encoded face prompt embeddings
    """
    # Use the pipeline's encode_prompt method if available
    if hasattr(pipeline, 'encode_prompt'):
        face_embeds, _ = pipeline.encode_prompt(
            face_caption,
            face_caption,  # prompt_2
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            negative_prompt_2=None,
        )[:2]  # Only need prompt_embeds, not pooled
        return face_embeds
    
    # Fallback: basic encoding
    text_encoder = pipeline.text_encoder or pipeline.text_encoder_1
    tokenizer = pipeline.tokenizer or pipeline.tokenizer_1
    
    text_inputs = tokenizer(
        face_caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]
    
    # Duplicate for classifier-free guidance
    if do_classifier_free_guidance:
        uncond_tokens = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_tokens.input_ids.to(device)
        
        with torch.no_grad():
            negative_prompt_embeds = text_encoder(uncond_input_ids)[0]
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
    return prompt_embeds


def create_face_mask_from_attention(
    attention_maps: Dict[str, torch.Tensor],
    height: int,
    width: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Create face mask from attention heatmaps.
    
    Args:
        attention_maps: Dictionary of attention maps by layer
        height: Target height
        width: Target width
        threshold: Binarization threshold
        
    Returns:
        Face mask tensor [1, 1, H, W]
    """
    if not attention_maps:
        # Return empty mask if no attention maps
        return torch.zeros(1, 1, height, width)
    
    # Average all attention maps
    all_maps = []
    for layer_name, attn_map in attention_maps.items():
        if attn_map.dim() == 4:  # [B, H, W, C]
            attn_map = attn_map.mean(dim=-1, keepdim=True)  # Average over channels
        elif attn_map.dim() == 3:  # [B, H, W]
            attn_map = attn_map.unsqueeze(-1)
        
        # Resize to target size
        attn_map = F.interpolate(
            attn_map.permute(0, 3, 1, 2),  # [B, C, H, W]
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        all_maps.append(attn_map)
    
    # Average across all layers
    combined_map = torch.stack(all_maps).mean(dim=0)
    
    # Normalize and threshold
    combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)
    mask = (combined_map > threshold).float()
    
    return mask


# Debug utilities
def save_debug_images(
    pipeline,
    reference_latents: torch.Tensor,
    mask: torch.Tensor,
    step_idx: int,
    output_dir: str = "debug",
):
    """Save debug visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    if step_idx == 0:  # Only save reference once
        # Decode reference latents
        with torch.no_grad():
            ref_latents = reference_latents / pipeline.vae.config.scaling_factor
            ref_image = pipeline.vae.decode(ref_latents).sample[0]
            ref_image = (ref_image / 2 + 0.5).clamp(0, 1)
            ref_image = ref_image.cpu().permute(1, 2, 0).numpy()
            ref_image = (ref_image * 255).astype("uint8")
            Image.fromarray(ref_image).save(f"{output_dir}/reference.png")
    
    # Save mask
    if mask.dim() == 4:
        mask_vis = mask[0, 0].cpu().numpy()
    else:
        mask_vis = mask[0].cpu().numpy()
    mask_vis = (mask_vis * 255).astype("uint8")
    Image.fromarray(mask_vis).save(f"{output_dir}/mask_step_{step_idx:03d}.png")


def debug_reference_latents_once(
    pipeline,
    debug_dir: str = "debug",
):
    """Save reference latents visualization once per run."""
    if not hasattr(pipeline, '_ref_latents_all'):
        return
        
    if getattr(pipeline, "_saved_ref_latents_img", False):
        return
    
    os.makedirs(debug_dir, exist_ok=True)
    
    vae_device = next(pipeline.vae.parameters()).device
    vae_dtype = next(pipeline.vae.parameters()).dtype
    
    ref_lat = pipeline._ref_latents_all.to(device=vae_device, dtype=vae_dtype)
    ref_lat = ref_lat / pipeline.vae.config.scaling_factor
    
    with torch.no_grad():
        img = pipeline.vae.decode(ref_lat).sample[0]
        img_np = ((img.float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    
    Image.fromarray(img_np).save(os.path.join(debug_dir, "debug_ref_latents.png"))
    print(f"[Debug] Saved reference latents image → {debug_dir}/debug_ref_latents.png")
    
    pipeline._saved_ref_latents_img = True