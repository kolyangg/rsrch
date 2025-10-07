"""
pipeline_branched_16Sep.py - PuLID pipeline with branched attention support
Integrates PhotoMaker's branched attention mechanism into PuLID
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union, Tuple

# Import base PuLID pipeline
from pulid.pipeline_v1_1 import PuLIDPipeline

# Import branched helpers
from .pulid_branched_helpers_16Sep import (
    two_branch_predict_pulid,
    prepare_reference_latents_pulid,
    encode_face_prompt_pulid,
    create_face_mask_from_image,
    load_mask_from_file,
)

# Import attention processors
from .pulid_attention_processor_16Sep import (
    BranchedAttnProcessorPuLID,
    BranchedCrossAttnProcessorPuLID,
)


class PuLIDPipelineBranched(PuLIDPipeline):
    """
    Extended PuLID pipeline with branched attention support.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Branched attention state
        self.use_branched_attention = False
        self.branched_processors_installed = False
        self._original_attn_processors = None
        
        # Mask storage
        self.current_mask = None
        self.current_mask_ref = None
        
        # Reference latents
        self._ref_latents_all = None
        
        # Face prompt embeddings
        self._face_prompt_embeds = None
        
        # Runtime parameters (matching PhotoMaker)
        self.pose_adapt_ratio = 0.25
        self.ca_mixing_for_face = True
        self.use_id_embeds = True
        self.do_classifier_free_guidance = True
        self.sequential_branched: bool = False
        self.sdpa_backend: str = "auto"

        # Debug
        self.debug_dir = None

        # Precision override for branched attention
        self.branched_attention_dtype: Optional[torch.dtype] = None
        
    def setup_branched_attention(
        self,
        use_branched: bool = True,
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
        use_id_embeds: bool = True,
        attention_dtype: Optional[torch.dtype] = None,
    ):
        """
        Setup branched attention processors for the UNet.
        """
        self.use_branched_attention = use_branched
        self.pose_adapt_ratio = pose_adapt_ratio
        self.ca_mixing_for_face = ca_mixing_for_face
        self.use_id_embeds = use_id_embeds
        self.branched_attention_dtype = attention_dtype
        
        if not use_branched:
            # Restore original processors if we have them
            if self._original_attn_processors is not None:
                self.pipe.unet.set_attn_processor(self._original_attn_processors)
                self.branched_processors_installed = False
            return
        
        # Store original processors if first time
        if self._original_attn_processors is None:
            self._original_attn_processors = self.pipe.unet.attn_processors.copy()
        
        # Install branched processors
        self._install_branched_processors()
        self.branched_processors_installed = True
        
    def _install_branched_processors(self):
        """Install branched attention processors on UNet"""
        new_procs = {}
        
        for name, proc in self.pipe.unet.attn_processors.items():
            # Determine hidden size from layer name
            if "mid_block" in name:
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            else:
                hidden_size = self.pipe.unet.config.block_out_channels[0]
            
            if name.endswith("attn1.processor"):
                # Self-attention: use branched processor
                new_proc = BranchedAttnProcessorPuLID(
                    hidden_size=hidden_size,
                    cross_attention_dim=hidden_size,
                    scale=1.0,
                )
                new_proc.pose_adapt_ratio = self.pose_adapt_ratio
                new_proc.ca_mixing_for_face = self.ca_mixing_for_face
                new_proc.use_id_embeds = self.use_id_embeds
                if hasattr(new_proc, "set_attention_dtype"):
                    new_proc.set_attention_dtype(self.branched_attention_dtype)
                new_procs[name] = new_proc

            elif name.endswith("attn2.processor"):
                # Cross-attention: use branched cross-attention processor
                cross_attention_dim = self.pipe.unet.config.cross_attention_dim
                num_tokens = 77  # Standard CLIP token count
                if hasattr(self.pipe, 'tokenizer_2'):
                    num_tokens = self.pipe.tokenizer_2.model_max_length
                    
                new_proc = BranchedCrossAttnProcessorPuLID(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_tokens,
                )
                if hasattr(new_proc, "set_attention_dtype"):
                    new_proc.set_attention_dtype(self.branched_attention_dtype)
                new_procs[name] = new_proc
            else:
                # Keep original for other processors
                new_procs[name] = self._original_attn_processors[name]
        
        self.pipe.unet.set_attn_processor(new_procs)

        # Optionally cast UNet weights to target dtype to lower VRAM
        if self.branched_attention_dtype is not None:
            self.pipe.unet.to(dtype=self.branched_attention_dtype)

        # Channels-last is friendlier on memory for large convs
        try:
            self.pipe.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass
        
    def set_masks(self, mask: torch.Tensor, mask_ref: Optional[torch.Tensor] = None):
        """Set masks for branched attention"""
        self.current_mask = mask
        self.current_mask_ref = mask_ref if mask_ref is not None else mask
        
        # Update masks in all processors
        if self.branched_processors_installed:
            for name, proc in self.pipe.unet.attn_processors.items():
                if hasattr(proc, 'set_masks'):
                    proc.set_masks(mask, mask_ref)
    
    def prepare_reference_latents(
        self,
        reference_image: Union[Image.Image, np.ndarray],
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare reference latents from reference image"""
        self._ref_latents_all = prepare_reference_latents_pulid(
            self, reference_image, height, width, dtype, generator
        )
        return self._ref_latents_all
    
    def prepare_face_prompts(self, batch_size: int = 1):
        """Prepare face prompt embeddings for cross-attention"""
        device = self.device if hasattr(self, 'device') else 'cuda'
        self._face_prompt_embeds = encode_face_prompt_pulid(
            self, device, batch_size, self.do_classifier_free_guidance
        )
        return self._face_prompt_embeds
    
    def inference_branched(
        self,
        prompt: str,
        size: tuple,
        prompt_n: str = '',
        id_embedding: torch.Tensor = None,
        uncond_id_embedding: torch.Tensor = None,
        id_scale: float = 1.0,
        guidance_scale: float = 1.2,
        steps: int = 4,
        seed: int = 42,
        # Branched attention parameters
        use_branched_attention: bool = True,
        branched_attn_start_step: int = 1,
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
        use_id_embeds: bool = True,
        branched_attention_dtype: Optional[torch.dtype] = None,
        # Mask parameters
        import_mask: Optional[str] = None,
        import_mask_ref: Optional[str] = None,
        reference_pil: Optional[Image.Image] = None,
        debug_dir: Optional[str] = None,
    ):
        """
        Run inference with branched attention support.
        """
        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Setup branched attention
        self.setup_branched_attention(
            use_branched=use_branched_attention,
            pose_adapt_ratio=pose_adapt_ratio,
            ca_mixing_for_face=ca_mixing_for_face,
            use_id_embeds=use_id_embeds,
            attention_dtype=branched_attention_dtype,
        )
        
        batch_size, height, width = size
        device = self.device if hasattr(self, 'device') else 'cuda'
        dtype = branched_attention_dtype or torch.float16

        # Enable extra memory savers (VAE is not the culprit, but this helps overall)
        # try:
        #     self.pipe.enable_vae_slicing()
        #     if hasattr(self.pipe.vae, "enable_tiling"):
        #         self.pipe.vae.enable_tiling()
        # except Exception:
        #     pass

        try:
            self.pipe.enable_vae_slicing()
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
        except Exception:
           pass
        # Keep heavy modules on CPU during denoise; only UNet on GPU
        try:
            self.pipe.text_encoder.to("cpu")
            self.pipe.vae.to("cpu")
        except Exception:
            pass
        # Optional CPU offload for UNet weights if Accelerate is present
        try:
            from accelerate import cpu_offload
           # prefer sequential offload to keep peak VRAM low
            self.pipe.enable_sequential_cpu_offload()
        except Exception:
            try:
               self.pipe.enable_model_cpu_offload()
            except Exception:
                pass

        
        # Prepare masks
        if use_branched_attention:
            if import_mask and os.path.exists(import_mask):
                mask = load_mask_from_file(import_mask, height, width)
                print(f"[Mask] Loaded mask from {import_mask}")
            elif reference_pil is not None:
                mask = create_face_mask_from_image(reference_pil, height, width)
                print("[Mask] Generated face mask from reference image")
            else:
                # Default mask (center region)
                mask = torch.ones(1, 1, height//8, width//8) * 0.5
                print("[Mask] Using default center mask")
            
            if import_mask_ref and os.path.exists(import_mask_ref):
                mask_ref = load_mask_from_file(import_mask_ref, height, width)
                print(f"[Mask] Loaded reference mask from {import_mask_ref}")
            else:
                mask_ref = mask
            
            mask = mask.to(device=device, dtype=dtype)
            mask_ref = mask_ref.to(device=device, dtype=dtype)
            self.set_masks(mask, mask_ref)
            
            # Prepare reference latents
            if reference_pil is not None:
                self.prepare_reference_latents(reference_pil, height, width, dtype)
                print("[Reference] Prepared reference latents")
            
            # Prepare face prompts
            self.prepare_face_prompts(batch_size)
            print("[Prompts] Prepared face prompt embeddings")
        
        # Set seed
        generator = torch.Generator(device).manual_seed(seed)
        
        # Run custom denoising loop with branched attention
        if use_branched_attention:
            return self._run_branched_denoising(
                prompt=prompt,
                negative_prompt=prompt_n,
                height=height,
                width=width,
                num_images=batch_size,
                id_embedding=id_embedding,
                uncond_id_embedding=uncond_id_embedding,
                id_scale=id_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                branched_attn_start_step=branched_attn_start_step,
            )
        else:
            # Fall back to standard inference
            return self.inference(
                prompt=prompt,
                size=size,
                prompt_n=prompt_n,
                id_embedding=id_embedding,
                uncond_id_embedding=uncond_id_embedding,
                id_scale=id_scale,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
            )
    
    def _run_branched_denoising(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_images: int,
        id_embedding: torch.Tensor,
        uncond_id_embedding: torch.Tensor,
        id_scale: float,
        guidance_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
        branched_attn_start_step: int,
    ):
        """
        Custom denoising loop with branched attention support.
        """
        device = self.device if hasattr(self, 'device') else 'cuda'
        dtype = self.branched_attention_dtype or torch.float16
        
        # Encode prompts (SDXL returns 4 values)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=num_images,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
        
        # Combine for CFG if needed
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        # Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare latents
        latent_channels = self.pipe.unet.config.in_channels
        latents_shape = (num_images, latent_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Prepare added conditions for SDXL
        # Get text encoder projection dim
        text_encoder_projection_dim = (
            self.pipe.text_encoder_2.config.projection_dim 
            if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None
            else None
        )
        
        add_time_ids = self.pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width),
            dtype=dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )
        add_time_ids = add_time_ids.repeat(num_images, 1)
        
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        
        # Move to device
        add_time_ids = add_time_ids.to(device=device, dtype=dtype)
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds.to(device=device, dtype=dtype),
            "time_ids": add_time_ids,
        }
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Use branched prediction after start step
            if i >= branched_attn_start_step and self.use_branched_attention:
                noise_pred, _, _ = two_branch_predict_pulid(
                    pipeline=self,
                    latent_model_input=latent_model_input,
                    t=t,
                    prompt_embeds=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    mask4=self.current_mask,
                    mask4_ref=self.current_mask_ref,
                    reference_latents=self._ref_latents_all,
                    face_prompt_embeds=self._face_prompt_embeds,
                    id_embedding=id_embedding,
                    uncond_id_embedding=uncond_id_embedding,
                    step_idx=i,
                    scale=1.0,
                    sequential=self.sequential_branched and (not self.do_classifier_free_guidance),
                )
                # Trim allocator between iterations to fight fragmentation
                torch.cuda.empty_cache()
            else:
                # Standard prediction
                cross_attention_kwargs = {
                    'id_embedding': id_embedding,
                    'uncond_id_embedding': uncond_id_embedding,
                } if id_embedding is not None else None
                
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            
            # Classifier-free guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
        
        # Decode latents
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        
        # Post-process
        # image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        return [image]

    # ---- SDPA backend control (VRAM tuning) ----
    def set_sdpa_backend(self, backend: str = "auto"):
        """
        backend: 'auto' | 'flash' | 'mem_efficient' | 'math'
        """
        self.sdpa_backend = backend
        # Prefer expandable segments to reduce fragmentation if user didn't set it
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            from torch.nn.attention import sdpa_kernel
            if backend == "flash":
                ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
            elif backend == "mem_efficient":
                ctx = sdpa_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
            elif backend == "math":
                ctx = sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            else:
                # auto: prefer mem_efficient on consumer GPUs
                ctx = sdpa_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
            self._sdpa_ctx = ctx
        except Exception:
            self._sdpa_ctx = None
