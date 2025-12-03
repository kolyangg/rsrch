"""
branched_new.py - Simplified branched attention implementation with cross-attention
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
    scale: float = 1.0,
    id_embeds: Optional[torch.Tensor] = None,
    class_tokens_mask: Optional[torch.Tensor] = None,
)-> None:
    """
    Patch UNet with branched attention processors for both self and cross attention.
    """
    ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor
    # Allow disabling branched self-attention and/or cross-attention via runtime flags.
    #  - disable_branched_sa=True  → keep original attn1 processors (no branched SA)
    #  - disable_branched_ca=True  → keep original attn2 processors (no branched CA)
    disable_sa = bool(getattr(pipeline, "disable_branched_sa", False))
    disable_ca = bool(getattr(pipeline, "disable_branched_ca", False))
    ### 25 Nov: AB testing to disable BranchedCrossAttnProcessor
    # Optional: switch between legacy (v1) and trainable (v2) branched attention processors.
    # Default to legacy (v1) when flag is not provided.
    use_attn_v2 = bool(getattr(pipeline, "use_attn_v2", False))
    if use_attn_v2:
        from ._old2.attn_processor2 import BranchedAttnProcessor, BranchedCrossAttnProcessor
    else:
        from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor

    # print(f'[TEMP DEBUG] mask in patch_unet_attention_processors: {mask}')
    
    # Store original processors once
    if not hasattr(pipeline, '_original_attn_processors'):
        pipeline._original_attn_processors = {}
        for name, proc in pipeline.unet.attn_processors.items():
            pipeline._original_attn_processors[name] = proc
    
    # Check if already patched
    current_procs = pipeline.unet.attn_processors
    has_branched = any(
        isinstance(p, (BranchedAttnProcessor, BranchedCrossAttnProcessor)) 
        for p in current_procs.values()
    )


    def _apply_runtime_flags(proc, pipe):
        # propagate key runtime knobs from model/pipeline onto processors
        for k in ("pose_adapt_ratio", "ca_mixing_for_face", "train_branch_mode", "id_alpha", "use_id_embeds"):
            if hasattr(pipe, k):
                setattr(proc, k, getattr(pipe, k))
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        # Optional toggle for per-branch BA-specific adapters.
        if hasattr(pipe, "ba_weights_split"):
            setattr(proc, "ba_weights_split", getattr(pipe, "ba_weights_split"))
        ### 29 Nov - Clean separataion of BA-specific parameters ###
   
    # Build safe, consistent context (batch, id_embeds)
    # Ensure masks are non-None to avoid runtime errors
    B = (mask.shape[0] if mask is not None else mask_ref.shape[0])
    dev, dt = pipeline.device, pipeline.unet.dtype
    _mask  = mask     if mask     is not None else torch.zeros(B, 1,  mask_ref.shape[-2], mask_ref.shape[-1], device=dev, dtype=dt)
    _mref  = mask_ref if mask_ref is not None else _mask
    # Always provide id_embeds so processor-local weights participate on every rank
    _idem = id_embeds.to(dev, dt) if id_embeds is not None else torch.zeros(B, 2048, device=dev, dtype=dt)   


    if not has_branched:
        # Create new processors
        new_procs = {}
        
        # Get cross-attention dimension
        cross_attention_dim = pipeline.unet.config.cross_attention_dim
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = cross_attention_dim[0]
        
        for name in pipeline.unet.attn_processors.keys():
            # Get hidden size
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
                if disable_sa:
                    # Keep original self-attn processor; no branching on attn1.
                    new_procs[name] = pipeline._original_attn_processors[name]
                else:
                    # Self-attention: use branched processor
                    proc = BranchedAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=hidden_size,
                        scale=scale,
                    ).to(pipeline.device, dtype=pipeline.unet.dtype)
                    proc.set_masks(_mask, _mref)
                    _apply_runtime_flags(proc, pipeline)

                    # Wire id_embeds (zeros if missing); whether they are used is controlled by use_id_embeds
                    proc.id_embeds = _idem

                    new_procs[name] = proc
                
            elif name.endswith("attn2.processor"):
                if disable_ca:
                    # Keep original cross-attn processor; no branched CA.
                    new_procs[name] = pipeline._original_attn_processors[name]
                else:
                    # Cross-attention: use branched cross-attention processor
                    num_tokens = 77  # Standard CLIP token count
                    if hasattr(pipeline, 'tokenizer_2'):
                        num_tokens = pipeline.tokenizer_2.model_max_length

                    proc = BranchedCrossAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=scale,
                        num_tokens=num_tokens,
                    ).to(pipeline.device, dtype=pipeline.unet.dtype)
                    # enable KV equalizer for face branch
                    setattr(proc, "equalize_face_kv", True)
                    setattr(proc, "equalize_clip", (1/3, 8.0))
                    proc.set_masks(_mask, _mref)
                    # Keep CA path consistent too (even if CA doesn’t always consume id_embeds)
                    proc.id_embeds = _idem
                    proc.class_tokens_mask = class_tokens_mask

                    new_procs[name] = proc
                
            else:
                # Keep original for other processors
                new_procs[name] = pipeline._original_attn_processors[name]
        
        pipeline.unet.set_attn_processor(new_procs)
    else:
                # Update masks on existing processors
        for name, proc in pipeline.unet.attn_processors.items():
            if isinstance(proc, (BranchedAttnProcessor, BranchedCrossAttnProcessor)):
                # proc.set_masks(mask, mask_ref)
                proc.set_masks(_mask, _mref)
                _apply_runtime_flags(proc, pipeline)

                # (Re)apply id_embeds (zeros if missing); actual usage is gated by use_id_embeds
                if hasattr(proc, "id_embeds"):
                    proc.id_embeds = _idem

def encode_face_prompt(
    pipeline,
    device: torch.device,
    batch_size: int,
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """
    Encode "face" text prompt for face branch cross-attention.
    """
    # Simple "face" prompt
    face_text = "a close-up human face laughing hard"
    
    # Use the pipeline's text encoder
    if hasattr(pipeline, 'encode_prompt'):
        # face_embeds, neg_embeds, _, _ = pipeline.encode_prompt(
        # Get the full prompt embeddings with correct sequence length
        face_embeds, neg_face_embeds, _, _ = pipeline.encode_prompt(
            face_text,
            face_text,  # prompt_2
            device,
            1,  # num_images_per_prompt
            do_classifier_free_guidance,
            negative_prompt="" if do_classifier_free_guidance else None,
            negative_prompt_2="" if do_classifier_free_guidance else None,
        )
        
        # Expand to batch size
        if do_classifier_free_guidance:
            # Already has [neg, pos] structure
            # Combine negative and positive
            return torch.cat([neg_face_embeds, face_embeds], dim=0)
        else:
            return face_embeds
    
    return None


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
    class_tokens_mask: Optional[torch.Tensor] = None,
    face_embed_strategy: str = "face",
    id_embeds: Optional[torch.Tensor] = None, 
    step_idx: int = 0,
    scale: float = 1.0,
    timestep_cond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute two-branch prediction with doubled batch for both latents and prompts.
    """

    full_debug = False

    # --- quick shape + CFG sanity ---
    if full_debug:
        if step_idx in (0, 1) or step_idx % 10 == 0:
            def stat(x): 
                x = x.float()
                return f"shape={tuple(x.shape)} μ={x.mean().item():.4f} σ={x.std().item():.4f}"
            print(f"[2BP] step={step_idx}  CFG={pipeline.do_classifier_free_guidance}")
        print(f"[2BP]   latent_in:   {stat(latent_model_input)}")
        print(f"[2BP]   ref_latents: {stat(reference_latents)}")


        # --- quick mask stats ---
        if step_idx in (0, 1) or step_idx % 10 == 0:
            m = mask4.detach().float()
            mr = mask4_ref.detach().float()
            def mstat(m):
                return f"{tuple(m.shape)}  mean={m.mean().item():.4f}  ones={(m>0.5).float().mean().item():.4f}"
            print(f"[2BP]   mask gen: {mstat(m)}   mask ref: {mstat(mr)}   |diff|={(m-mr).abs().mean().item():.4f}")



    device = latent_model_input.device
    dtype = latent_model_input.dtype
    batch_size = latent_model_input.shape[0]
    

    REF_NOISE_ONCE = True # CRITICAL FIX: Initialize reference noise ONCE at pipeline start
    
    if not hasattr(pipeline, '_ref_noise'):
        if not REF_NOISE_ONCE:
            pipeline._ref_noise = torch.randn_like(reference_latents)
        else:
            # Use a fixed seed for consistent reference noise
            ref_gen = torch.Generator(device=device)
            if hasattr(pipeline, 'generator') and pipeline.generator is not None:
                # Use pipeline's generator seed if available
                ref_gen.manual_seed(42)  # Or extract seed from pipeline.generator
            try:
                pipeline._ref_noise = torch.randn_like(reference_latents, generator=ref_gen)
            # --- ADDED For training integration ---
            except TypeError:
                pipeline._ref_noise = torch.randn(
                    reference_latents.shape,
                    generator=ref_gen,
                    device=reference_latents.device,
                    dtype=reference_latents.dtype,
                )
            # --- ADDED For training integration ---
            print(f"[2BP] Initialized reference noise (fixed for entire generation)")
            
    
    t_ref = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
    if t_ref.ndim == 0:
        t_ref = t_ref.unsqueeze(0)
    expected_ref = reference_latents.shape[0]
    current_ref = t_ref.shape[0]
    if current_ref != expected_ref:
        reps = (expected_ref + current_ref - 1) // current_ref
        t_ref = t_ref.repeat(reps)[:expected_ref]
    
    ref_noised = pipeline.scheduler.add_noise(
        reference_latents,
        pipeline._ref_noise[:reference_latents.shape[0]],
        t_ref
    )

    
    ref_noised = pipeline.scheduler.scale_model_input(ref_noised, t_ref).to(latent_model_input.dtype) # critical: match UNet’s expected scaling at this timestep

    if full_debug:
        if step_idx in (0, 1) or step_idx % 10 == 0:
            print(f"[2BP]   ref_noised:  {stat(ref_noised)}  Δ(noise,ref)σ={(latent_model_input.std()-ref_noised.std()).item():.4f}")

    
    # Ensure same batch size
    if ref_noised.shape[0] < batch_size:
        ref_noised = ref_noised.expand(batch_size, -1, -1, -1)
    
    # Create doubled batch: [noise, reference]
    batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)
    
    # Patch processors with masks
    patch_unet_attention_processors(
        pipeline, mask4, mask4_ref, scale,
        id_embeds=id_embeds if face_embed_strategy == "id_embeds" else None,
        class_tokens_mask=class_tokens_mask
    )

    # --- quick patch check
    if full_debug:
        if step_idx == 0:
            procs = pipeline.unet.attn_processors
            n_sa = sum("attn1.processor" in k for k in procs)  # self-attn slots
            n_ca = sum("attn2.processor" in k for k in procs)  # cross-attn slots
            any_branched = any(p.__class__.__name__.startswith("Branched") for p in procs.values())
            sample_k = next(iter(procs))
            print(f"[2BP]   processors patched? {any_branched}  (SA={n_sa}, CA={n_ca})  sample={procs[sample_k].__class__.__name__}")

        
    # Prepare timesteps for doubled batch
    t_batched = t if torch.is_tensor(t) else torch.tensor([t], device=device)
    if t_batched.ndim == 0:
        t_batched = t_batched.unsqueeze(0)
    expected = batched_latents.shape[0]
    current = t_batched.shape[0]
    if current != expected:
        reps = (expected + current - 1) // current
        t_batched = t_batched.repeat(reps)[:expected]
    
    # Prepare face prompt if not provided
    if face_prompt_embeds is None:
        face_prompt_embeds = encode_face_prompt(
            pipeline, 
            device, 
            batch_size,
            pipeline.do_classifier_free_guidance
        )

    
    # Only mirror the main text into the face branch for legacy "id".
    # For "id_embeds" we keep actual "face" text and use the 2048-D ID features.
    if (face_embed_strategy or "face") in {"id"}:    
        # keep dtype/device aligned with text encoder / UNet
        d, dev = prompt_embeds.dtype, prompt_embeds.device
        face_prompt_embeds = prompt_embeds.clone()
        if class_tokens_mask is not None:
            m = class_tokens_mask.to(dev)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            if m.shape[0] < face_prompt_embeds.shape[0]:
               m = m.expand(face_prompt_embeds.shape[0], -1)
            m = m.unsqueeze(-1).to(dtype=d)                # [B,L,1]
            one = torch.tensor(1.0, device=dev, dtype=d)
            id_scale = torch.tensor(getattr(pipeline, "id_token_scale", 2.5),
                                   device=dev, dtype=d)
            # face_prompt_embeds = face_prompt_embeds * (one - m) + face_prompt_embeds * m * id_scale
            
            # Use only ID tokens for the face branch (no leakage from other words)
            face_prompt_embeds = face_prompt_embeds * m * id_scale
           
        else:
         print(f"[2BP]   WARNING: class_tokens_mask is None, falling back to face text")
         # Fallback to face text encoding
         face_prompt_embeds = encode_face_prompt(
             pipeline, device, batch_size, pipeline.do_classifier_free_guidance
         ).to(prompt_embeds.device, prompt_embeds.dtype)
                  
        # per-token std match: bring face tokenwise std ~ gen tokenwise std
        eps = 1e-6
        std_gen  = prompt_embeds.float().std(dim=-1, keepdim=True).clamp_min(eps)
        std_face = face_prompt_embeds.float().std(dim=-1, keepdim=True).clamp_min(eps)
        # face_prompt_embeds = (face_prompt_embeds / std_face) * std_gen
        face_prompt_embeds = ((face_prompt_embeds.float() / std_face) * std_gen).to(d)
    
    if full_debug:
        # ---quick prompt stats---
        if step_idx in (0, 1) or step_idx % 10 == 0:
            pe = prompt_embeds.detach().float()
            fe = face_prompt_embeds.detach().float()
            same_shape = pe.shape == fe.shape
            # frac of zeros in face prompt (detect padding/truncation artefacts)
            frac_zero = (fe.abs() < 1e-8).float().mean().item()
            diff_mean = (pe - fe).abs().mean().item() if same_shape else float('nan')
            print(f"[2BP]   prompts: gen={tuple(pe.shape)}  face={tuple(fe.shape)}  zero_frac(face)={frac_zero:.3f}  Δμ={diff_mean:.4f}")


    
    # --- Build face-branch text properly and concat ------------------------
    # Ensure face_prompt_embeds exists and matches shape/dtype of prompt_embeds
    if face_prompt_embeds is None or face_prompt_embeds.shape != prompt_embeds.shape:
       # re-encode a clean face text that mirrors CFG/batch exactly
        face_prompt_embeds = encode_face_prompt(
            pipeline, device, batch_size, pipeline.do_classifier_free_guidance
        )
    face_prompt_embeds = face_prompt_embeds.to(prompt_embeds.device, prompt_embeds.dtype)

    # Double-stack encoder states for branched CA:
    #   first half → generation prompt
    #   second half → face prompt
    encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_embeds], dim=0)

    if full_debug:
        # quick sanity – these should *not* be identical
        if (step_idx in (0, 1)) or (step_idx % 10 == 0):
            diff_mu = (prompt_embeds.detach().float() - face_prompt_embeds.detach().float()).abs().mean().item()
            print(f"[2BP]   encoder_hidden_states Δ(gen,face)μ={diff_mu:.4f}")


    # Double added_cond_kwargs
    doubled_kwargs = {}
    for k, v in added_cond_kwargs.items():
        if torch.is_tensor(v):
            # Double the tensor
            doubled_kwargs[k] = torch.cat([v, v], dim=0)
        else:
            doubled_kwargs[k] = v
    
    # Double timestep_cond if present
    if timestep_cond is not None:
        timestep_cond_doubled = torch.cat([timestep_cond, timestep_cond], dim=0)
    else:
        timestep_cond_doubled = None
    
    # Single forward pass with doubled batch
    noise_pred = pipeline.unet(
        batched_latents,
        t_batched,
        encoder_hidden_states=encoder_hidden_states,
        timestep_cond=timestep_cond_doubled,
        cross_attention_kwargs=getattr(pipeline, '_cross_attention_kwargs', None),
        added_cond_kwargs=doubled_kwargs,
        return_dict=False,
    )[0]

    # --- quick check of cosine sim between halves
    # Split UNet output into halves (noise/merged vs face-pure)
    B2 = noise_pred.shape[0] // 2
    first, second = noise_pred[:B2].float(), noise_pred[B2:].float()

    if full_debug:
        # If CFG is on, each half is [uncond, cond]
        if pipeline.do_classifier_free_guidance and B2 % 2 == 0:
            fU, fC = first.chunk(2)
            sU, sC = second.chunk(2)
            def s2(x): return f"σ={x.std().item():.4f}"
            print(f"[2BP]   out halves: first({s2(first)})  second({s2(second)})  | first U/C {s2(fU)}/{s2(fC)}  second U/C {s2(sU)}/{s2(sC)}")
        else:
            print(f"[2BP]   out halves: first σ={first.std().item():.4f}  second σ={second.std().item():.4f}")

        # Mean cosine sim between halves → should NOT be ~1.0
        cos = torch.nn.functional.cosine_similarity(first.flatten(1), second.flatten(1), dim=1).mean().item()
        print(f"[2BP]   cos(first,second)={cos:.3f}")
    # --- end of quick check



    
    # Extract merged result (first half)
    noise_pred_merged = noise_pred[:batch_size]
    
    USE_SOFT_BLENDING = True
    
    if USE_SOFT_BLENDING:
        if mask4 is not None and mask4.shape[-2:] == noise_pred_merged.shape[-2:]:
            mask4 = gaussian_blur_mask(mask4, kernel_size=5) # Apply gaussian blur to mask for smoother transitions
    
    
    # For debugging: approximate branch outputs
    mask_4ch = mask4.repeat(1, 4, 1, 1).to(dtype=dtype)
    if mask_4ch.shape[0] < batch_size:
        mask_4ch = mask_4ch.expand(batch_size, -1, -1, -1)
    
    noise_bg = noise_pred_merged * (1 - mask_4ch)
    noise_face = noise_pred_merged * mask_4ch
    
    # Debug logging
    if full_debug:
        if step_idx < 3 or step_idx % 10 == 0:
            print(f"[Branch] Step {step_idx}: "
                f"merged_norm={noise_pred_merged.std().item():.4f}, "
                f"face={noise_face.std().item():.4f}, "
                f"bg={noise_bg.std().item():.4f}")
   
    return noise_pred_merged, noise_face, noise_bg


def restore_original_processors(pipeline):
   """Restore original attention processors."""
   if hasattr(pipeline, '_original_attn_processors'):
       pipeline.unet.set_attn_processor(pipeline._original_attn_processors)
       delattr(pipeline, '_original_attn_processors')
       return True
   return False


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
       
    
def gaussian_blur_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to mask for smoother transitions."""
    import torch.nn.functional as F
    
    # Create a simple Gaussian kernel
    sigma = kernel_size / 3.0
    kernel_1d = torch.exp(-torch.arange(kernel_size, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d[None, None, :, :].to(mask.device, mask.dtype)
    
    # Apply convolution
    mask_blurred = F.conv2d(mask, kernel_2d, padding=kernel_size // 2)
    
    return mask_blurred.clamp(0, 1)
