"""
pulid_branched_helpers_16Sep.py - Helper functions for branched attention in PuLID
Adapted from PhotoMaker's branched attention implementation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import gc
import numpy as np
from PIL import Image


def two_branch_predict_pulid(
    pipeline,
    latent_model_input: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: Dict[str, Any],
    mask4: Optional[torch.Tensor],
    mask4_ref: Optional[torch.Tensor],
    reference_latents: torch.Tensor,
    face_prompt_embeds: Optional[torch.Tensor] = None,
    id_embedding: Optional[torch.Tensor] = None,
    uncond_id_embedding: Optional[torch.Tensor] = None,
    step_idx: int = 0,
    scale: float = 1.0,
    timestep_cond: Optional[torch.Tensor] = None,
    sequential: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute two-branch prediction for PuLID with doubled batch for both latents and prompts.
    
    This function handles the special case of branched attention where we need to process
    both noise and reference latents through the UNet with different attention masks.
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    
    # Get the UNet from pipeline
    unet = pipeline.pipe.unet if hasattr(pipeline, 'pipe') else pipeline.unet
    
    # Ensure masks are provided
    if mask4 is None or mask4_ref is None:
        # Fallback to standard prediction if no masks
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred, None, None
    

    # If we choose sequential mode and CFG is OFF, avoid doubling the batch:
    if sequential and not pipeline.do_classifier_free_guidance:
        # 1) Tell self-attn processors to cache ref stream
        for _, proc in unet.attn_processors.items():
            if hasattr(proc, "set_seq_mode"):
                proc.set_seq_mode('record_ref')
            if hasattr(proc, 'set_masks'):
                proc.set_masks(mask4, mask4_ref)

        # run once on reference latents (no_grad + inference_mode to minimize buffers)
        with torch.inference_mode():
            _ = unet(
                reference_latents,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        # free any ephemeral CUDA buffers before the noise pass
        torch.cuda.empty_cache()
        gc.collect()
        # _ = unet(
        #     reference_latents,
        #     t,
        #     encoder_hidden_states=prompt_embeds,
        #     timestep_cond=timestep_cond,
        #     added_cond_kwargs=added_cond_kwargs,
        #     return_dict=False,
        # )
        # 2) Switch to using cached ref on noise stream
        for _, proc in unet.attn_processors.items():
            if hasattr(proc, "set_seq_mode"):
                proc.set_seq_mode('use_cached')
        with torch.inference_mode():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        # 3) Clear cache
        for _, proc in unet.attn_processors.items():
            if hasattr(proc, "set_seq_mode"):
                proc.set_seq_mode(None)
            if hasattr(proc, "_cached_ref_hidden"):
                proc._cached_ref_hidden = None
        torch.cuda.empty_cache()
        gc.collect()
        return noise_pred, None, None

    # === Parallel (original) path ===
    # Prepare doubled batch: [noise_batch, reference_batch]
    
    batch_size = latent_model_input.shape[0] // 2 if pipeline.do_classifier_free_guidance else latent_model_input.shape[0]
    
    # Ensure reference latents match batch size
    if reference_latents.shape[0] < batch_size:
        reference_latents = reference_latents.expand(batch_size, -1, -1, -1)
    
    # Create doubled latent input
    if pipeline.do_classifier_free_guidance:
        # For CFG, we have [uncond_noise, cond_noise]
        # We need to create [uncond_noise, cond_noise, uncond_ref, cond_ref]
        uncond_noise = latent_model_input[:batch_size]
        cond_noise = latent_model_input[batch_size:]
        
        # Double the reference latents for CFG
        doubled_latents = torch.cat([
            uncond_noise,
            cond_noise,
            reference_latents,
            reference_latents,
        ], dim=0)
        
        # Double the prompt embeddings
        doubled_prompts = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        
        # Double the masks
        doubled_mask = torch.cat([mask4, mask4_ref], dim=0)
        doubled_mask_ref = torch.cat([mask4_ref, mask4_ref], dim=0)
        
    else:
        # Without CFG
        doubled_latents = torch.cat([latent_model_input, reference_latents], dim=0)
        doubled_prompts = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        doubled_mask = torch.cat([mask4, mask4_ref], dim=0)
        doubled_mask_ref = mask4_ref
    
    # Update masks in processors
    for name, proc in unet.attn_processors.items():
        if hasattr(proc, 'set_masks'):
            proc.set_masks(doubled_mask, doubled_mask_ref)
    
    # Handle face prompt embeddings for cross-attention
    if face_prompt_embeds is not None:
        # Set face prompts in cross-attention processors
        for name, proc in unet.attn_processors.items():
            if hasattr(proc, 'face_prompt_embeds'):
                proc.face_prompt_embeds = face_prompt_embeds
    
    # Handle PuLID ID embeddings
    if id_embedding is not None:
        # For PuLID, we need to pass ID embeddings through cross_attention_kwargs
        cross_attention_kwargs = {
            'id_embedding': id_embedding,
            'uncond_id_embedding': uncond_id_embedding if uncond_id_embedding is not None else torch.zeros_like(id_embedding),
        }
    else:
        cross_attention_kwargs = None
    
    # Handle added_cond_kwargs - double them for the doubled batch
    doubled_added_cond = {}
    for key, value in added_cond_kwargs.items():
        if isinstance(value, torch.Tensor):
            # Double the tensor for doubled batch
            doubled_added_cond[key] = torch.cat([value, value], dim=0)
        else:
            doubled_added_cond[key] = value
    
    # Run the UNet with doubled batch
    noise_pred_doubled = unet(
        doubled_latents,
        t,
        encoder_hidden_states=doubled_prompts,
        timestep_cond=timestep_cond,
        added_cond_kwargs=doubled_added_cond,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False,
    )[0]
    
    # Split the predictions back
    if pipeline.do_classifier_free_guidance:
        # Extract noise predictions (first half of doubled batch)
        noise_pred = noise_pred_doubled[:2*batch_size]
        # Extract face and background components if needed
        noise_face = None  # Can be extracted from attention maps if needed
        noise_bg = None
    else:
        noise_pred = noise_pred_doubled[:batch_size]
        noise_face = None
        noise_bg = None
    
    return noise_pred, noise_face, noise_bg


def prepare_reference_latents_pulid(
    pipeline,
    reference_image,
    height: int,
    width: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Prepare reference latents from reference image for PuLID.
    """
    device = pipeline.device if hasattr(pipeline, 'device') else 'cuda'
    vae = pipeline.pipe.vae if hasattr(pipeline, 'pipe') else pipeline.vae
    
    # Convert to tensor if needed
    if isinstance(reference_image, Image.Image):
        reference_image = np.array(reference_image)
    
    if isinstance(reference_image, np.ndarray):
        reference_image = torch.from_numpy(reference_image).float() / 255.0
        if reference_image.ndim == 3:
            reference_image = reference_image.permute(2, 0, 1)
        reference_image = reference_image.unsqueeze(0)
    
    # Resize if needed
    if reference_image.shape[-2:] != (height, width):
        reference_image = F.interpolate(
            reference_image,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
    
    # Move to device and dtype
    reference_image = reference_image.to(device=device, dtype=dtype)
    
    # Normalize to [-1, 1]
    reference_image = 2.0 * reference_image - 1.0
    
    # Encode to latent space
    with torch.no_grad():
        ref_latents = vae.encode(reference_image).latent_dist.sample()
        ref_latents = ref_latents * vae.config.scaling_factor
    
    return ref_latents


def encode_face_prompt_pulid(
    pipeline,
    device: torch.device,
    batch_size: int,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Encode "face" text prompt for face branch cross-attention in PuLID.
    """
    # Simple face prompt for PuLID
    face_text = "a close-up human face"
    
    # Get the text encoder from pipeline
    pipe = pipeline.pipe if hasattr(pipeline, 'pipe') else pipeline
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokenizer_2 = getattr(pipe, 'tokenizer_2', None)
    text_encoder_2 = getattr(pipe, 'text_encoder_2', None)
    
    # Tokenize
    text_inputs = tokenizer(
        face_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    
    # Encode
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        
        # Handle SDXL dual text encoder
        if text_encoder_2 is not None and tokenizer_2 is not None:
            text_inputs_2 = tokenizer_2(
                face_text,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids_2 = text_inputs_2.input_ids.to(device)
            
            prompt_embeds_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
            pooled_prompt_embeds_2 = prompt_embeds_2[0]
            prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
            
            # Concatenate embeddings
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds_2
    
    # Expand for batch size
    prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
    
    # Handle classifier-free guidance
    if do_classifier_free_guidance:
        # Create negative embeddings (empty prompt)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    
    return prompt_embeds


def create_face_mask_from_image(
    image: Image.Image,
    height: int,
    width: int,
    face_detection_model=None,
) -> torch.Tensor:
    """
    Create a face mask from an image using face detection.
    Returns a binary mask tensor of shape [1, 1, H//8, W//8] for latent space.
    """
    import cv2
    
    # Convert PIL to numpy
    img_np = np.array(image)
    
    # Simple face detection using OpenCV if no model provided
    if face_detection_model is None:
        # Use OpenCV's cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Create mask
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
        for (x, y, w, h) in faces:
            # Add some padding
            padding = int(0.2 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_np.shape[1] - x, w + 2 * padding)
            h = min(img_np.shape[0] - y, h + 2 * padding)
            mask[y:y+h, x:x+w] = 1.0
    else:
        # Use provided face detection model
        # This would be implemented based on the specific model
        mask = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.float32) * 0.5
    
    # Convert to tensor and resize to latent dimensions
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    
    # Resize to latent space dimensions (1/8 of image size)
    latent_height = height // 8
    latent_width = width // 8
    mask_latent = F.interpolate(
        mask_tensor,
        size=(latent_height, latent_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Apply Gaussian blur for smoother boundaries
    kernel_size = 5
    sigma = 1.0
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    mask_latent = F.conv2d(mask_latent, kernel, padding=kernel_size//2)
    
    # Threshold to binary
    mask_latent = (mask_latent > 0.3).float()
    
    return mask_latent


def load_mask_from_file(mask_path: str, height: int, width: int) -> torch.Tensor:
    """
    Load a mask from an image file and prepare it for latent space.
    """
    mask_img = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_img) / 255.0
    
    # Convert to tensor
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    
    # Resize to latent dimensions
    latent_height = height // 8
    latent_width = width // 8
    mask_latent = F.interpolate(
        mask_tensor,
        size=(latent_height, latent_width),
        mode='bilinear',
        align_corners=False
    )
    
    return mask_latent