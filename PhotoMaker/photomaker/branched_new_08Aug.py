"""
branched_new.py - New implementation with custom attention processor approach
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, List
import copy


def patch_unet_attention_processors(
    pipeline,
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor,
    reference_latents: torch.Tensor,
    face_prompt_embeds: torch.Tensor,
    step_idx: int = 0,
):
    """
    Patch UNet attention processors with our custom BranchedAttnProcessor.
    Only stores original processors once per run.
    
    Args:
        pipeline: The pipeline object containing unet
        mask4: 4-channel face mask for current noise [B, 4, H, W]
        mask4_ref: 4-channel face mask for reference image [B, 4, H, W]
        reference_latents: Encoded reference image latents [B, 4, H, W]
        face_prompt_embeds: Text embeddings for "face" prompt
        step_idx: Current denoising step index
    """
    from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
    
  
    # Store original processors only once per run
    if not hasattr(pipeline, '_original_attn_processors'):
        pipeline._original_attn_processors = {}
        for name, proc in pipeline.unet.attn_processors.items():
            pipeline._original_attn_processors[name] = proc
        # pipeline._processors_patched = False
    
    # # # Only create new processors if not already patched
    # if not getattr(pipeline, '_processors_patched', False):
    
    # Check if we need to create new processors or just update existing ones
    current_procs = pipeline.unet.attn_processors
    has_branched = any(isinstance(p, (BranchedAttnProcessor, BranchedCrossAttnProcessor)) 
                    for p in current_procs.values())
   
    if not has_branched:
        # Create new branched processors

        new_attn_procs = {}
        
        # layer_names = list(pipeline.unet.attn_processors.keys())
        # print(f"[Branched New] Layer names in UNet: {layer_names}")
        
        for name in pipeline.unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                # Self-attention layers - apply branched processing
                new_attn_procs[name] = BranchedAttnProcessor(
                    mask=mask4,
                    mask_ref=mask4_ref,
                    reference_latents=reference_latents,
                    step_idx=step_idx,
                )
            elif name.endswith("attn2.processor"):
                # Cross-attention layers - also apply branched processing for face prompt
                new_attn_procs[name] = BranchedCrossAttnProcessor(
                    mask=mask4_ref, # REFERENCE MASK HERE
                    mask_ref=mask4_ref,  # REFERENCE MASK HERE (placeholder, not used)
                    face_prompt_embeds=face_prompt_embeds,
                    step_idx=step_idx,
                )
            else:
                # Keep original processor for other layers
                new_attn_procs[name] = pipeline._original_attn_processors[name]
        
        # Set the new processors
        pipeline.unet.set_attn_processor(new_attn_procs)
        # pipeline._processors_patched = True


    else:
        # Update existing branched processors with new data
        updated = 0
        for name, proc in pipeline.unet.attn_processors.items():

            if isinstance(proc, BranchedAttnProcessor):
                # ensure 1-channel masks inside processors
                proc.mask = mask4[:, :1] if mask4.dim() == 4 and mask4.size(1) > 1 else mask4
                proc.mask_ref = mask4_ref[:, :1] if mask4_ref.dim() == 4 and mask4_ref.size(1) > 1 else mask4_ref
                proc.reference_latents = reference_latents
                proc.step_idx = step_idx
                updated += 1

            elif isinstance(proc, BranchedCrossAttnProcessor):
                proc.mask = mask4_ref[:, :1] if mask4_ref.dim() == 4 and mask4_ref.size(1) > 1 else mask4_ref
                proc.mask_ref = proc.mask
                proc.face_prompt_embeds = face_prompt_embeds
                proc.step_idx = step_idx
                updated += 1
        if step_idx % 10 == 0:
             print(f"[Patch] Updated {updated} existing branched processors at step {step_idx}")

def restore_original_processors(pipeline):
    """Restore original attention processors after branched inference."""
    # if hasattr(pipeline, '_original_attn_processors'):
    #     pipeline.unet.set_attn_processor(pipeline._original_attn_processors)
    if hasattr(pipeline, '_original_attn_processors') and pipeline._original_attn_processors: # TO CHECK
        pipeline.unet.set_attn_processor(pipeline._original_attn_processors)


def prepare_double_batch_inputs(
    latent_model_input: torch.Tensor,
    reference_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    face_prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
) -> tuple:
    """
    Prepare inputs for double batch processing (noise + reference).
    Ensure everything is properly aligned.
    """
    batch_size = latent_model_input.shape[0]
    
    # Ensure matching batch sizes
    if reference_latents.shape[0] != batch_size:
        reference_latents = reference_latents.expand(batch_size, -1, -1, -1)
    
    
    ## STACKING for face_prompt_embeds ##
    
    # Ensure matching sequence lengths for stacking
    if prompt_embeds.shape[1] != face_prompt_embeds.shape[1]:
        # Pad or truncate to match sequence length
        target_seq_len = prompt_embeds.shape[1]
        if face_prompt_embeds.shape[1] < target_seq_len:
            # Pad face embeddings
            padding = torch.zeros(
                batch_size, 
                target_seq_len - face_prompt_embeds.shape[1], 
                face_prompt_embeds.shape[2],
                device=face_prompt_embeds.device,
                dtype=face_prompt_embeds.dtype
            )
            face_prompt_embeds = torch.cat([face_prompt_embeds, padding], dim=1)
        else:
            # Truncate face embeddings
            face_prompt_embeds = face_prompt_embeds[:, :target_seq_len, :]
    
    # Stack latents: [2B, 4, H, W]
    stacked_latents = torch.cat([latent_model_input, reference_latents], dim=0)
    
    # Stack prompt embeddings: [2B, seq_len, hidden_dim]
    stacked_embeds = torch.cat([prompt_embeds, face_prompt_embeds], dim=0)
    
    
    # Double the additional conditioning
    stacked_cond_kwargs = {}
    for key, value in added_cond_kwargs.items():
        if isinstance(value, torch.Tensor):
            if value.dim() > 0 and value.shape[0] == batch_size:
                # Double along batch dimension
                stacked_cond_kwargs[key] = torch.cat([value, value], dim=0)
            elif value.dim() > 0 and value.shape[0] == batch_size * 2:
                # Already doubled (e.g., for CFG)
                stacked_cond_kwargs[key] = torch.cat([value, value], dim=0)
            else:
                # Keep as is
                stacked_cond_kwargs[key] = value
        else:
            stacked_cond_kwargs[key] = value
    
    return stacked_latents, stacked_embeds, stacked_cond_kwargs


def split_double_batch_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the double batch output back into noise and reference predictions.
    
    Args:
        output: Combined output [2B, ...]
        
    Returns:
        Tuple of (noise_pred, reference_pred)
    """
    batch_size = output.shape[0] // 2
    return output[:batch_size], output[batch_size:]


def encode_face_prompt(pipeline, device, dtype, batch_size: int = 1) -> torch.Tensor:
    """
    Encode the "face" text prompt for reference branch.
    
    Args:
        pipeline: The pipeline object with text encoders
        device: Device to place tensors on
        dtype: Data type for tensors
        batch_size: Batch size for encoding
        
    Returns:
        Face prompt embeddings
    """
    face_prompt = "face"
    
    # Use the pipeline's encoding method if available
    if hasattr(pipeline, 'encode_prompt'):
        # print("[Branched New] Encoding face prompt using pipeline.encode_prompt")
        face_prompt_embeds, _, _, _ = pipeline.encode_prompt(
            prompt=face_prompt,
            prompt_2=face_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        # Expand to match batch size if needed
        if face_prompt_embeds.shape[0] != batch_size:
            face_prompt_embeds = face_prompt_embeds.repeat(batch_size, 1, 1)
    else:
        print("[Branched New] Encoding face prompt using fallback method")
        # Fallback: simple encoding
        from transformers import CLIPTokenizer
        tokenizer = pipeline.tokenizer if hasattr(pipeline, 'tokenizer') else CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = pipeline.text_encoder if hasattr(pipeline, 'text_encoder') else None
        
        if text_encoder is not None:
            text_inputs = tokenizer(
                face_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            face_prompt_embeds = text_encoder(text_input_ids)[0]
            face_prompt_embeds = face_prompt_embeds.repeat(batch_size, 1, 1)
        else:
            # If no text encoder available, create dummy embeddings
            face_prompt_embeds = torch.randn(batch_size, 77, 768, device=device, dtype=dtype)
    
    return face_prompt_embeds.to(dtype=dtype)



def two_branch_predict(
    pipeline,
    latents: torch.Tensor,
    latent_model_input: torch.Tensor,
    mask4: torch.Tensor,
    mask4_ref: torch.Tensor | None,
    t: torch.Tensor,
    prompt_embeds_step: torch.Tensor,
    added_cond_kwargs: dict,
    timestep_cond: torch.Tensor | None,
    step_idx: int = 0,
    debug_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    New two-branch prediction using custom attention processors.
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    batch_size = latent_model_input.shape[0]
    
    
    ### MASKING
    # Use reference mask if provided, otherwise use same as noise mask
    if mask4_ref is None:
        mask4_ref = mask4
    
    # Debug: Save masks at final step
    if debug_dir and (step_idx == pipeline._num_timesteps - 1 or step_idx == 0):
        import os
        from PIL import Image
        import numpy as np
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save standalone masks
        
        # # invert mask4 (DEBUG TEST!!! WTF)
        # mask4 = 1 - mask4  # Invert mask for noise branch
        
        mask_np = (mask4[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
        mask_ref_np = (mask4_ref[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_np).save(os.path.join(debug_dir, f"mask_noise_step{step_idx:03d}.png"))
        Image.fromarray(mask_ref_np).save(os.path.join(debug_dir, f"mask_ref_step{step_idx:03d}.png"))
        
        if hasattr(pipeline, '_reference_latents'):
            reference_latents = pipeline._reference_latents 
        else:
            print("[Branch] Warning: No reference latents found")
        
        # Save masks applied to images (decode and apply)
        if hasattr(pipeline, 'vae'):
            with torch.no_grad():
                # Decode noise latents
                noise_img = pipeline.vae.decode(latent_model_input / pipeline.vae.config.scaling_factor).sample[0]
                noise_img = ((noise_img + 1) * 127.5).clamp(0, 255).float().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                
                # Decode reference latents  
                ref_img = pipeline.vae.decode(reference_latents / pipeline.vae.config.scaling_factor).sample[0]
                ref_img = ((ref_img + 1) * 127.5).clamp(0, 255).float().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

                # Apply masks (resize to image size)
                from PIL import Image as PILImage
                mask_resized = np.array(PILImage.fromarray(mask_np).resize((noise_img.shape[1], noise_img.shape[0])))
                mask_ref_resized = np.array(PILImage.fromarray(mask_ref_np).resize((ref_img.shape[1], ref_img.shape[0])))
                
                # Create overlay
                noise_masked = noise_img.copy()
                noise_masked[mask_resized < 128] = noise_masked[mask_resized < 128] // 2  # Darken background
                
                ref_masked = ref_img.copy()
                ref_masked[mask_ref_resized < 128] = ref_masked[mask_ref_resized < 128] // 2  # Darken background
                
                PILImage.fromarray(noise_masked).save(os.path.join(debug_dir, f"noise_masked_step{step_idx:03d}.png"))
                PILImage.fromarray(ref_masked).save(os.path.join(debug_dir, f"ref_masked_step{step_idx:03d}.png"))
    ### MASKING
    
    # Get reference latents - check both possible attribute names
    if hasattr(pipeline, '_reference_latents'):
        reference_latents_clean = pipeline._reference_latents
    elif hasattr(pipeline, '_ref_latents_all'):
        reference_latents_clean = pipeline._ref_latents_all
    else:
        print("[Branch] Warning: No reference latents found, using current latents")
        reference_latents_clean = latents.clone()
        pipeline._reference_latents = reference_latents_clean
    
    # Ensure reference_latents has correct batch size
    if reference_latents_clean.shape[0] != batch_size:
        if reference_latents_clean.shape[0] == 1:
            reference_latents_clean = reference_latents_clean.expand(batch_size, -1, -1, -1)
        else:
            reference_latents_clean = reference_latents_clean[:batch_size]
    

    # NEW (fixed):
    if not hasattr(pipeline, '_ref_noise'):
        # Generate consistent noise for reference
        # Use torch.randn with manual shape and device instead of randn_like with generator
        if hasattr(pipeline, 'generator') and pipeline.generator is not None:
            generator = pipeline.generator
            if generator is not None:
                # Set the generator state for reproducibility
                if hasattr(generator, 'manual_seed'):
                    generator.manual_seed(42)  # Or use a seed from pipeline
        else:
            generator = None
        
        # Create noise with proper shape and device
        shape = reference_latents_clean.shape
        if generator is not None:
            pipeline._ref_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            pipeline._ref_noise = torch.randn(shape, device=device, dtype=dtype)



    # # Scale noise appropriately for current timestep

    if t.dim() == 0:
        # Scalar timestep - convert to list
        timesteps_for_noise = torch.tensor([t.item()], device=device, dtype=torch.long)
    elif t.shape[0] == 1:
        # Single timestep tensor
        timesteps_for_noise = t.long()
    else:
        # Multiple timesteps - use first one for all
        timesteps_for_noise = t[0:1].long()

    reference_latents_noised = pipeline.scheduler.add_noise(
        reference_latents_clean,
        pipeline._ref_noise[:reference_latents_clean.shape[0]],
        timesteps_for_noise
    )

    # Debug: Check if latents are properly noised
    if step_idx < 3:
        print(f"[Branch Debug] Step {step_idx}:")
        print(f"  Noise latents norm: {latent_model_input.std().item():.4f}")
        print(f"  Clean reference norm: {reference_latents_clean.std().item():.4f}")
        print(f"  Noised reference norm: {reference_latents_noised.std().item():.4f}")
        print(f"  Timestep: {t.item() if t.dim() == 0 else t[0].item()}")
        
    # Now reference_latents_noised is on the same diffusion trajectory
    reference_latents = reference_latents_noised
    
    # face_prompt_embeds = encode_face_prompt(pipeline, device, dtype, batch_size)
    
    use_id_embeds = getattr(pipeline, 'face_embed_strategy', 'face') == 'id_embeds'


    if use_id_embeds and hasattr(pipeline, '_id_embeds'):
        # Use ID embeddings from PhotoMaker
        face_prompt_embeds = pipeline._id_embeds
        # Ensure correct shape [batch_size, seq_len, hidden_dim]
        if face_prompt_embeds.dim() == 2:
            face_prompt_embeds = face_prompt_embeds.unsqueeze(0)
        if face_prompt_embeds.shape[0] != batch_size:
            face_prompt_embeds = face_prompt_embeds.expand(batch_size, -1, -1)


        # Project ID embeds to match prompt dimension if needed
        if face_prompt_embeds.shape[-1] != prompt_embeds_step.shape[-1]:
            # Create a linear projection if not exists
            if not hasattr(pipeline, '_id_to_prompt_projection'):
                pipeline._id_to_prompt_projection = torch.nn.Linear(
                    face_prompt_embeds.shape[-1], 
                    prompt_embeds_step.shape[-1],
                    bias=False
                ).to(device=device, dtype=dtype)
                # Initialize with small values for stability
                torch.nn.init.normal_(pipeline._id_to_prompt_projection.weight, std=0.01)
            
            # Project the embeddings
            face_prompt_embeds = pipeline._id_to_prompt_projection(face_prompt_embeds)
        
        print(f"[Branch Debug] Using ID embeds with shape {face_prompt_embeds.shape}")
    else:
        # Use "face" text prompt
        face_prompt_embeds = encode_face_prompt(pipeline, device, dtype, batch_size)
        print(f"[Branch Debug] Using face text prompt with shape {face_prompt_embeds.shape}")
    
    
    # Patch attention processors with our custom branched versions
    patch_unet_attention_processors(
        pipeline=pipeline,
        mask4=mask4,
        mask4_ref=mask4_ref,
        reference_latents=reference_latents,  # Use noised version
        face_prompt_embeds=face_prompt_embeds,
        step_idx=step_idx,
    )


    # Prepare double batch inputs
    stacked_latents, stacked_embeds, stacked_cond_kwargs = prepare_double_batch_inputs(
        latent_model_input=latent_model_input,
        reference_latents=reference_latents,
        prompt_embeds=prompt_embeds_step,
        face_prompt_embeds=face_prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
    )
    
        
        
    # Prepare timestep conditioning for double batch
    if timestep_cond is not None:
        if timestep_cond.shape[0] == batch_size:
            stacked_timestep_cond = torch.cat([timestep_cond, timestep_cond], dim=0)
        else:
            stacked_timestep_cond = timestep_cond.repeat(2)
    else:
        stacked_timestep_cond = None
    
    # Double the timestep tensor - handle scalar case
    if t.dim() == 0:
        # Scalar timestep - expand and repeat
        t_stacked = t.expand(batch_size * 2)
    else:
        # Tensor with batch dimension - concatenate
        t_stacked = torch.cat([t, t], dim=0)
    
    # Run single UNet forward pass with double batch
    # The custom attention processors will handle the branching internally
    stacked_noise_pred = pipeline.unet(
        stacked_latents,
        t_stacked,
        encoder_hidden_states=stacked_embeds,
        timestep_cond=stacked_timestep_cond,
        cross_attention_kwargs=pipeline.cross_attention_kwargs if hasattr(pipeline, 'cross_attention_kwargs') else None,
        added_cond_kwargs=stacked_cond_kwargs,
        return_dict=False,
    )[0]
    
    # Split the output back into noise and reference predictions
    noise_pred, reference_pred = split_double_batch_output(stacked_noise_pred)
    
    # The attention processors have already done the merging internally
    # noise_pred contains the merged result
    # reference_pred contains the reference branch output (for debugging)
    
    # Restore original processors
    # restore_original_processors(pipeline)
    # Don't restore processors here - they need to stay active for all steps
    
    # For compatibility, also return individual branch predictions
    # These are approximations since the actual branching happens inside attention
    mask_4ch = mask4.to(dtype=dtype)
    
    # Approximate branch outputs for debugging
    # In reality, these are mixed within the attention layers
    noise_bg = noise_pred * (1 - mask_4ch)
    noise_fc = noise_pred * mask_4ch
    
    # Debug output
    if step_idx < 3 or step_idx % 10 == 0:
        print(f"[Branched New] Step {step_idx}")
        print(f"  Input norm: {latent_model_input.std().item():.4f}")
        print(f"  Merged norm: {noise_pred.std().item():.4f}")
        print(f"  Reference norm: {reference_pred.std().item():.4f}")
    
    return noise_pred, noise_fc, noise_bg


# Additional helper functions

def prepare_reference_latents(pipeline, reference_image, device, dtype):
    """
    Encode reference image to latents and store in pipeline.
    
    Args:
        pipeline: Pipeline object with VAE
        reference_image: PIL Image, tensor, or already encoded latents
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    # Check if we already have reference latents computed
    if hasattr(pipeline, '_ref_latents_all'):
        pipeline._reference_latents = pipeline._ref_latents_all
        return
        
    if hasattr(pipeline, 'vae'):
        # If it's already a latent tensor
        if isinstance(reference_image, torch.Tensor) and reference_image.dim() == 4:
            if reference_image.shape[1] == 4:  # Already in latent space
                pipeline._reference_latents = reference_image.to(device, dtype)
                return
                
        # Convert image to tensor if needed
        if not isinstance(reference_image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            reference_tensor = transform(reference_image).unsqueeze(0).to(device, dtype)
        else:
            reference_tensor = reference_image
        
        # Ensure correct shape [B, C, H, W]
        if reference_tensor.dim() == 3:
            reference_tensor = reference_tensor.unsqueeze(0)
            
        # Encode to latents
        with torch.no_grad():
            reference_latents = pipeline.vae.encode(reference_tensor).latent_dist.sample()
            reference_latents = reference_latents * pipeline.vae.config.scaling_factor
        
        pipeline._reference_latents = reference_latents
        print(f"[Branch] Encoded reference image to latents with shape {reference_latents.shape}")
    else:
        # Fallback: use random latents
        print("[Branch] Warning: No VAE found, using random latents for reference")
        h = 64  # default latent height
        w = 64  # default latent width
        pipeline._reference_latents = torch.randn(1, 4, h, w, device=device, dtype=dtype)



def create_face_mask_from_attention(
    pipeline,
    attention_maps: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Create face mask from attention maps.
    
    Args:
        pipeline: Pipeline object
        attention_maps: Dictionary of attention maps from hooks
        threshold: Threshold for binarization
        
    Returns:
        Face mask tensor [B, 4, H, W]
    """
    # Aggregate attention maps
    aggregated = None
    for name, attn_map in attention_maps.items():
        if aggregated is None:
            aggregated = attn_map
        else:
            aggregated = aggregated + attn_map
    
    if aggregated is not None:
        # Average and threshold
        aggregated = aggregated / len(attention_maps)
        mask = (aggregated > threshold).float()
        
        # Resize to latent dimensions
        latent_h = pipeline.unet.config.sample_size // 8
        latent_w = pipeline.unet.config.sample_size // 8
        
        mask = F.interpolate(
            mask.unsqueeze(1),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Expand to 4 channels
        mask = mask.repeat(1, 4, 1, 1)
        print(f"[Branch] Created face mask from attention maps with shape {mask.shape}")
        
        return mask
    else:
        # Return default mask
        print("[Branch] Warning: No attention maps found, returning default mask")
        return torch.ones(1, 4, latent_h, latent_w, device=pipeline.device)