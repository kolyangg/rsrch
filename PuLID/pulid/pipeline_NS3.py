"""
pipeline_NS2.py - PuLID pipeline with integrated branched attention support
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union, Tuple
# from torch.cuda.amp import autocast

# ── AMP compat (torch>=2.0 uses torch.amp.autocast('cuda', ...)) ─────────────
try:
    from torch.amp import autocast as _autocast
    def autocast_cuda(dtype):
        return _autocast(device_type="cuda", dtype=dtype)
except Exception:
    from torch.cuda.amp import autocast as _autocast
    def autocast_cuda(dtype):
        return _autocast(dtype=dtype)


# Import base PuLID pipeline
from .pipeline_v1_1 import PuLIDPipeline

# Import branched attention processors
from .attention_processor_NS3 import (
    BranchedAttnProcessor_NS2,
    BranchedCrossAttnProcessor_NS2,
    IDAttnProcessor2_0_NS2,
    AttnProcessor2_0_NS2
)

# Import PhotoMaker's branched utilities (we'll copy these to PuLID folder)
import sys
import importlib.util

def import_from_photomaker(module_name: str):
    """Helper to import modules from PhotoMaker directory"""
    photomaker_path = os.path.join(os.path.dirname(__file__), '../../PhotoMaker/photomaker')
    module_path = os.path.join(photomaker_path, f'{module_name}.py')
    
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(f"photomaker_{module_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"photomaker_{module_name}"] = module
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not import {module_name} from PhotoMaker")

# Try to import PhotoMaker utilities
try:
    mask_utils = import_from_photomaker('mask_utils')
    add_masking = import_from_photomaker('add_masking')
    create_mask_ref = import_from_photomaker('create_mask_ref')
    
    # Import specific functions we need
    compute_binary_face_mask = mask_utils.compute_binary_face_mask
    DynamicMaskGenerator = add_masking.DynamicMaskGenerator
    compute_face_mask_from_pil = create_mask_ref.compute_face_mask_from_pil
except ImportError as e:
    print(f"Warning: Could not import PhotoMaker utilities: {e}")
    print("Some features may be limited. Consider copying mask_utils.py, add_masking.py, and create_mask_ref.py to PuLID/pulid/")
    
    # Define fallback functions
    compute_binary_face_mask = None
    DynamicMaskGenerator = None
    compute_face_mask_from_pil = None


class PuLIDPipeline_NS2(PuLIDPipeline):
    """
    Extended PuLID pipeline with branched attention support.
    Integrates PhotoMaker's branched attention mechanism into PuLID.
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
        
        # Runtime parameters (matching PhotoMaker)
        self.pose_adapt_ratio = 0.25
        self.ca_mixing_for_face = True
        self.use_id_embeds = True
        self.force_par_before_pm = False
        
        # Dynamic mask generator
        self.mask_generator = None
        
    def setup_branched_attention(
        self,
        use_branched: bool = True,
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
        use_id_embeds: bool = True,
    ):
        """
        Setup branched attention processors for the UNet.
        
        Args:
            use_branched: Whether to use branched attention
            pose_adapt_ratio: Blend ratio for pose adaptation (0=ref, 1=noise)
            ca_mixing_for_face: Whether to mix features in face branch
            use_id_embeds: Whether to use ID embeddings in face branch
        """
        self.use_branched_attention = use_branched
        print(f"[DEBUG] setup_branched_attention: use_branched={use_branched}, processors_installed={self.branched_processors_installed}")


        self.pose_adapt_ratio = pose_adapt_ratio
        self.ca_mixing_for_face = ca_mixing_for_face
        self.use_id_embeds = use_id_embeds
        
        if use_branched and not self.branched_processors_installed:
            self._install_branched_processors()
            
    def _install_branched_processors(self):
        """Install branched attention processors in the UNet"""
        
        # Store original processors
        if self._original_attn_processors is None:
            self._original_attn_processors = {}
            for name, proc in self.pipe.unet.attn_processors.items():
                self._original_attn_processors[name] = proc
        
        # Create new processors
        new_processors = {}
        
        # Get existing processors to determine structure
        existing_processors = self.pipe.unet.attn_processors
        
        # Get cross-attention dimension
        cross_attention_dim = self.pipe.unet.config.cross_attention_dim
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = cross_attention_dim[0]
        
        # for name in self.pipe.unet.attn_processors.keys():
        for name in existing_processors.keys():
            # Determine hidden size based on block location
            if "mid_block" in name:
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            else:
                hidden_size = None
            
            # For non-branched mode, still use our NS2 processors to handle tuple id_embedding
            if not self.use_branched_attention:
                # Use our standard processor that handles tuples
                if "attn2" in name and cross_attention_dim is not None:
                    processor = IDAttnProcessor2_0_NS2(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                else:
                    processor = AttnProcessor2_0_NS2()
            else:
                # Install branched processors
                if "attn1" in name:
                    # Self-attention - use branched processor
                    processor = BranchedAttnProcessor_NS2(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0
                    )
                    # Set runtime parameters
                    processor.pose_adapt_ratio = self.pose_adapt_ratio
                    processor.ca_mixing_for_face = self.ca_mixing_for_face
                    processor.use_id_embeds = self.use_id_embeds
                    
                elif "attn2" in name:
                    # Cross-attention - use ID processor for compatibility
                    if cross_attention_dim is not None:
                        processor = IDAttnProcessor2_0_NS2(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                        )
                    else:
                        processor = AttnProcessor2_0_NS2()
                else:
                    # Default processor
                    processor = AttnProcessor2_0_NS2()
            
            new_processors[name] = processor.to(self.device)
        
        # Set the new processors
        self.pipe.unet.set_attn_processor(new_processors)
        self.branched_processors_installed = True
        
        branched_count = sum(1 for p in new_processors.values() if isinstance(p, BranchedAttnProcessor_NS2))
        print(f"[DEBUG] Installed {branched_count} branched self-attention processors out of {len(new_processors)} total")
        
        # Store ID adapter layers for training compatibility
        self.id_adapter_attn_layers = nn.ModuleList(self.pipe.unet.attn_processors.values())
        
    def restore_original_processors(self):
        """Restore original attention processors"""
        if self._original_attn_processors is not None and len(self._original_attn_processors) > 0:
            self.pipe.unet.set_attn_processor(self._original_attn_processors)
            self.branched_processors_installed = False
        elif not self.branched_processors_installed:
            # Nothing to restore
            return
        else:
            # Fallback: reinstall default PuLID processors
            self.hack_unet_attn_layers(self.pipe.unet)
            self.branched_processors_installed = False
            
    def set_masks(self, mask: torch.Tensor, mask_ref: Optional[torch.Tensor] = None):
        """
        Set masks for branched attention processors.
        
        Args:
            mask: Face mask for noise latents (H, W) or (1, H, W) or (B, H, W)
            mask_ref: Face mask for reference latents (optional, defaults to mask)
        """
        self.current_mask = mask
        self.current_mask_ref = mask_ref if mask_ref is not None else mask
        
        print(f"[DEBUG] set_masks: mask shape={mask.shape if mask is not None else None}, "
            f"mask_ref shape={mask_ref.shape if mask_ref is not None else None}, "
            f"processors_installed={self.branched_processors_installed}")

        # Count how many processors got masks
        if self.branched_processors_installed:
            mask_count = sum(1 for p in self.pipe.unet.attn_processors.values() 
                            if hasattr(p, 'mask') and p.mask is not None)
            print(f"[DEBUG] Set masks on {mask_count} processors")
        
        # Update all branched processors with the new masks
        if self.branched_processors_installed:
            for name, processor in self.pipe.unet.attn_processors.items():
                if hasattr(processor, 'set_masks'):
                    processor.set_masks(self.current_mask, self.current_mask_ref)
                    
    def prepare_reference_latents(
        self,
        reference_image: Union[torch.Tensor, Image.Image, np.ndarray],
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Prepare reference image latents for branched attention.
        
        Args:
            reference_image: Reference image (PIL, numpy, or tensor)
            height: Output height
            width: Output width
            dtype: Data type for latents
            
        Returns:
            Reference latents tensor
        """
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
            reference_image = torch.nn.functional.interpolate(
                reference_image,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        # Move to device and dtype
        reference_image = reference_image.to(device=self.device, dtype=dtype)
        
        # Normalize to [-1, 1]
        reference_image = 2.0 * reference_image - 1.0
        
        # Encode to latent space
        with torch.no_grad():
            ref_latents = self.pipe.vae.encode(reference_image).latent_dist.sample()
            ref_latents = ref_latents * self.pipe.vae.config.scaling_factor
        
        return ref_latents
    
    def inference_branched(
        self,
        prompt: str,
        size: tuple,
        prompt_n: str = '',
        id_embedding: torch.Tensor = None,
        id_scale: float = 1.0,
        guidance_scale: float = 1.2,
        steps: int = 4,
        seed: int = 42,
        # Branched attention parameters
        use_branched_attention: bool = True,
        branched_attn_start_step: int = 10,
        face_embed_strategy: str = "id_embeds",
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
        use_id_embeds: bool = True,
        force_par_before_pm: bool = False,
        # Mask parameters
        auto_mask_ref: bool = True,
        import_mask: Optional[str] = None,
        import_mask_ref: Optional[str] = None,
        use_dynamic_mask: bool = False,
        # Reference image (for extracting from id_embedding)
        reference_image: Optional[Union[Image.Image, np.ndarray]] = None,
    ):
        """
        Run inference with branched attention support.
        
        This method extends the base PuLID inference with branched attention
        for better face/background separation.
        """
        
        # Setup branched attention if requested
        if use_branched_attention:
            self.setup_branched_attention(
                use_branched=True,
                pose_adapt_ratio=pose_adapt_ratio,
                ca_mixing_for_face=ca_mixing_for_face,
                use_id_embeds=use_id_embeds,
            )
        
        # Generate face masks if needed
        if auto_mask_ref and reference_image is not None and compute_face_mask_from_pil:
            # Auto-generate reference face mask
            if isinstance(reference_image, np.ndarray):
                reference_image = Image.fromarray(reference_image)
            mask_ref = compute_face_mask_from_pil(reference_image)
            mask_ref = torch.from_numpy(mask_ref).float() / 255.0
            mask_ref = mask_ref.unsqueeze(0).to(self.device)
        elif import_mask_ref and os.path.exists(import_mask_ref):
            # Load reference mask from file
            mask_ref = Image.open(import_mask_ref).convert('L')
            mask_ref = torch.from_numpy(np.array(mask_ref)).float() / 255.0
            mask_ref = mask_ref.unsqueeze(0).to(self.device)
        else:
            mask_ref = None
        
        # Load or generate noise mask
        if import_mask and os.path.exists(import_mask):
            mask = Image.open(import_mask).convert('L')
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = mask.unsqueeze(0).to(self.device)
        else:
            mask = mask_ref  # Use reference mask for noise if not provided
        
        # Set masks for branched processors
        if use_branched_attention and mask is not None:
            self.set_masks(mask, mask_ref)
            
            
        # After the mask loading section, before `if use_branched_attention and mask is not None:`
        print(f"[DEBUG] Loaded masks: import_mask={import_mask}, exists={os.path.exists(import_mask) if import_mask else False}")
        print(f"[DEBUG] mask tensor: shape={mask.shape if mask is not None else None}, "
            f"mean={mask.mean().item() if mask is not None else None}")
        
        # Prepare reference latents if using branched attention
        if use_branched_attention and reference_image is not None:
            ref_latents = self.prepare_reference_latents(
                reference_image,
                height=size[1],
                width=size[2],
                dtype=torch.float16
            )
        else:
            ref_latents = None
        
        # Run standard PuLID inference with potential branched modifications
        # Note: This would need modification of the base inference method to support
        # doubled batches and branched processing. For now, we use the standard method.
        # self.restore_original_processors()
        # self.setup_branched_attention(use_branched=False)
        
        # CRITICAL: PuLID's inference doesn't support doubled batches needed for branched attention
        # We need to implement a custom inference loop similar to PhotoMaker's
        print("[WARNING] Branched attention requires doubled batch - using simplified version")
        print(f"[DEBUG] Masks set: mask={self.current_mask is not None}, mask_ref={self.current_mask_ref is not None}")
        
        
        if use_branched_attention:
            # Use proper branched inference
            result = self.inference_branched_proper(
                prompt=prompt,
                size=size,
                prompt_n=prompt_n,
                id_embedding=id_embedding,
                id_scale=id_scale,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
                use_branched_attention=use_branched_attention,
                branched_attn_start_step=branched_attn_start_step,
                mask=mask,
                mask_ref=mask_ref,
            )
        
        else:
            # Fall back to standard inference
            result = self.inference(
                prompt=prompt,
                size=size,
                prompt_n=prompt_n,
                id_embedding=id_embedding,
                id_scale=id_scale,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed
            )
        
        # Restore original processors if we modified them
        if use_branched_attention:
            self.restore_original_processors()
            self.setup_branched_attention(use_branched=True)
        
        return result



    def inference_branched_proper(
        self,
        prompt: str,
        size: tuple,
        prompt_n: str = '',
        id_embedding: torch.Tensor = None,
        id_scale: float = 1.0,
        guidance_scale: float = 1.2,
        steps: int = 4,
        seed: int = 42,
        # Branched attention parameters
        use_branched_attention: bool = True,
        branched_attn_start_step: int = 10,
        # Mask parameters
        mask: Optional[torch.Tensor] = None,
        mask_ref: Optional[torch.Tensor] = None,
    ):  
        """
        Proper branched inference with doubled batch support for PuLID.
        """
        batch_size, height, width = size
        device = self.device
        unet_dtype = self.pipe.unet.dtype
        
        # Setup branched processors and masks
        if use_branched_attention and mask is not None:
            self.set_masks(mask, mask_ref)
        
        # # Encode prompt
        # prompt_embeds, pooled_prompt_embeds = self.pipe.encode_prompt(
        #     prompt=prompt,
        #     prompt_2=prompt,
        #     device=device,
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=True,
        #     negative_prompt=prompt_n,
        #     negative_prompt_2=prompt_n,
        # )

        # encode_prompt returns 4 values: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=prompt_n,
            negative_prompt_2=prompt_n,
        )
        
        # Match UNet dtype/device to avoid fp32 bloat
        prompt_embeds = prompt_embeds.to(device=device, dtype=unet_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=unet_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=unet_dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=device, dtype=unet_dtype)


        # Combine for CFG
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        # Prepare ID embeddings for cross-attention
        if id_embedding is not None:
            # PuLID format: (uncond_id_emb, cond_id_emb)
            if isinstance(id_embedding, tuple):
                uncond_id_emb, cond_id_emb = id_embedding
            else:
                cond_id_emb = id_embedding
                uncond_id_emb = torch.zeros_like(cond_id_emb)

            # Align to UNet dtype/device
            uncond_id_emb = uncond_id_emb.to(device=device, dtype=unet_dtype)
            cond_id_emb = cond_id_emb.to(device=device, dtype=unet_dtype)
            
            # Prepare for CFG
            id_embedding_cfg = torch.cat([uncond_id_emb, cond_id_emb], dim=0)
        else:
            id_embedding_cfg = None
        
        # Initialize latents
        latents_shape = (batch_size, 4, height // 8, width // 8)
        # latents = torch.randn(latents_shape, device=device, dtype=prompt_embeds.dtype)
        latents = torch.randn(latents_shape, device=device, dtype=unet_dtype)
        
        # Setup scheduler
        self.pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Create reference latents (clone of initial noise)
        ref_latents = latents.clone()
        
        # Denoising loop with branched attention
        for i, t in enumerate(timesteps):
            
            # Decide if branched attention is active
            branched_active = use_branched_attention and (i >= branched_attn_start_step)
            
            if branched_active:
                # === BRANCHED MODE: Double the batch ===
                # Prepare doubled latents [noise, reference]
                latent_model_input = torch.cat([latents, ref_latents], dim=0)
                print(f"[DEBUG] Step {i}: latent_model_input shape={latent_model_input.shape}, doubled batch")

                
                # # Double prompts for doubled batch
                # prompt_embeds_doubled = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                # pooled_embeds_doubled = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                
                # Double the batch for CFG if needed
                if guidance_scale > 1.0:
                    latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
                
                # For branched mode, we need [noise_uncond, noise_cond, ref_uncond, ref_cond] if using CFG
                # if guidance_scale > 1.0:
                #     # prompt_embeds already contains [uncond, cond] from encode_prompt
                #     prompt_embeds_doubled = prompt_embeds  # Already has CFG dimension
                #     pooled_embeds_doubled = pooled_prompt_embeds
                # else:
                #     # No CFG, just use conditional
                #     prompt_embeds_doubled = prompt_embeds[1:] if prompt_embeds.shape[0] > 1 else prompt_embeds
                #     pooled_embeds_doubled = pooled_prompt_embeds[1:] if pooled_prompt_embeds.shape[0] > 1 else pooled_prompt_embeds

                if guidance_scale > 1.0:
                    # Need [uncond_noise, uncond_ref, cond_noise, cond_ref]
                    prompt_embeds_doubled = torch.cat([
                        prompt_embeds[0:1], prompt_embeds[0:1],  # uncond for noise & ref
                        prompt_embeds[1:2], prompt_embeds[1:2]   # cond for noise & ref
                    ], dim=0)
                    pooled_embeds_doubled = torch.cat([
                        pooled_prompt_embeds[0:1], pooled_prompt_embeds[0:1],
                        pooled_prompt_embeds[1:2], pooled_prompt_embeds[1:2]
                    ], dim=0)
                else:
                    prompt_embeds_doubled = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                    pooled_embeds_doubled = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    

                # Ensure dtype/device
                prompt_embeds_doubled = prompt_embeds_doubled.to(device=device, dtype=unet_dtype)
                pooled_embeds_doubled = pooled_embeds_doubled.to(device=device, dtype=unet_dtype)


                
                # # Double ID embeddings if present
                # if id_embedding_cfg is not None:
                #     id_embedding_doubled = torch.cat([id_embedding_cfg, id_embedding_cfg], dim=0)
                
                # else:
                #     id_embedding_doubled = None

                # # REPLACE WITH:
                # id_embedding_doubled = id_embedding_cfg  # Already has CFG dimension
                
                # if id_embedding_cfg is not None and guidance_scale > 1.0:
                #     # Need to match the doubled batch structure
                #     id_embedding_doubled = torch.cat([
                #         id_embedding_cfg[0:1], id_embedding_cfg[0:1],  # uncond
                #         id_embedding_cfg[1:2], id_embedding_cfg[1:2]   # cond
                #     ], dim=0)
                # else:
                #     id_embedding_doubled = id_embedding_cfg


                if id_embedding_cfg is not None:
                    if guidance_scale > 1.0:
                        # For doubled batch with CFG: need 4 copies
                        id_embedding_doubled = torch.cat([
                            id_embedding_cfg[0:1], id_embedding_cfg[0:1],  # uncond for noise & ref
                            id_embedding_cfg[1:2], id_embedding_cfg[1:2]   # cond for noise & ref
                        ], dim=0)
                    else:
                        # No CFG, just double for [noise, ref]
                        id_embedding_doubled = torch.cat([id_embedding_cfg, id_embedding_cfg], dim=0) if id_embedding_cfg is not None else None
                else:
                    id_embedding_doubled = None
                    

                if id_embedding_doubled is not None:
                    id_embedding_doubled = id_embedding_doubled.to(device=device, dtype=unet_dtype)

                
                # # Add CFG dimension (unconditional + conditional for each)
                # if guidance_scale > 1.0:
                #     latent_model_input = torch.cat([latent_model_input] * 2)
                # CFG is already handled by doubled prompts, don't double latents again
                
                # Scale latents
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # # Predict noise with doubled batch
                # added_cond_kwargs = {"text_embeds": pooled_embeds_doubled, "time_ids": self.pipe._get_add_time_ids(
                #     (height, width), (0, 0), (height, width), dtype=prompt_embeds.dtype, device=device
                # ).repeat(latent_model_input.shape[0], 1)}
                
                # Create time_ids manually for SDXL
                original_size = (height, width)
                crops_coords_top_left = (0, 0)
                target_size = (height, width)
                # add_time_ids = list(original_size + crops_coords_top_left + target_size)
                # add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
                # add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)

                # added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                

                # Replace the similar _get_add_time_ids call with:
                # add_time_ids = torch.tensor([list(original_size + crops_coords_top_left + target_size)], 
                #                             dtype=prompt_embeds.dtype, device=device)
                
                add_time_ids = torch.tensor(
                    [list(original_size + crops_coords_top_left + target_size)],
                    dtype=unet_dtype, device=device
                )

                
                add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)

                # added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                added_cond_kwargs = {"text_embeds": pooled_embeds_doubled, "time_ids": add_time_ids}
                

                print(f"[DEBUG] UNet inputs - latent: {latent_model_input.shape}, prompt: {prompt_embeds_doubled.shape}, id: {id_embedding_doubled.shape if id_embedding_doubled is not None else None}")
                
                # Forward through UNet with branched processors
                # with autocast(device_type="cuda", dtype=unet_dtype):
                with autocast_cuda(unet_dtype):
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_doubled,
                        cross_attention_kwargs={"id_embedding": id_embedding_doubled, "id_scale": id_scale},
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                
                
                
                
                # # Split predictions: [noise_pred, ref_pred]
                # if guidance_scale > 1.0:
                #     # Split CFG predictions
                #     noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                #     # Get noise batch predictions (first half)
                #     noise_pred_uncond = noise_pred_uncond[:batch_size]
                #     noise_pred_cond = noise_pred_cond[:batch_size]
                #     # Apply CFG
                #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # else:
                #     # Just take noise batch prediction
                #     noise_pred = noise_pred[:batch_size]
                
                # # Also update reference latents with their prediction
                # if guidance_scale > 1.0:
                #     ref_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # else:
                #     ref_noise_pred = noise_pred[batch_size:batch_size*2]
                
                
                if guidance_scale > 1.0:
                    # Output order: [uncond_noise, uncond_ref, cond_noise, cond_ref]
                    noise_pred_uncond = noise_pred[0:1]
                    ref_pred_uncond = noise_pred[1:2]
                    noise_pred_cond = noise_pred[2:3]
                    ref_pred_cond = noise_pred[3:4]
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    ref_noise_pred = ref_pred_uncond + guidance_scale * (ref_pred_cond - ref_pred_uncond)
                else:
                    noise_pred = noise_pred[0:1]
                    ref_noise_pred = noise_pred[1:2]
                                
            else:
                # === STANDARD MODE: Single batch ===
                latent_model_input = latents
                
                if guidance_scale > 1.0:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": self.pipe._get_add_time_ids(
                #     (height, width), (0, 0), (height, width), dtype=prompt_embeds.dtype, device=device
                # ).repeat(latent_model_input.shape[0], 1)}
                
                
                # Create time_ids manually for SDXL
                original_size = (height, width)
                crops_coords_top_left = (0, 0)
                target_size = (height, width)
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                # add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
                add_time_ids = torch.tensor([add_time_ids], dtype=unet_dtype, device=device)
                add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)

                # added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(device=device, dtype=unet_dtype),
                    "time_ids": add_time_ids
                }
                
                

                # with autocast(device_type="cuda", dtype=unet_dtype):
                with autocast_cuda(unet_dtype):
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs={"id_embedding": id_embedding_cfg, "id_scale": id_scale},
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                
                # if guidance_scale > 1.0:
                #     noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                if guidance_scale > 1.0:
                    # For doubled batch with CFG: [noise_uncond, noise_cond, ref_uncond, ref_cond]
                    noise_pred_uncond = noise_pred[0:1]  # First is noise uncond
                    noise_pred_cond = noise_pred[1:2]    # Second is noise cond
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # No CFG, just take noise prediction
                    noise_pred = noise_pred[0:1]
                
                
                ref_noise_pred = noise_pred  # Use same for ref in standard mode
            
            # Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Add memory cleanup
            if i % 2 == 0:  # Every other step
                torch.cuda.empty_cache()
            
            # Update reference latents too (with slower update)
            if branched_active:
                ref_latents = self.pipe.scheduler.step(ref_noise_pred, t, ref_latents, return_dict=False)[0]
        
        # Decode latents
        latents = latents / self.pipe.vae.config.scaling_factor
        # with autocast(device_type="cuda", dtype=unet_dtype):
        with autocast_cuda(unet_dtype):
            image = self.pipe.vae.decode(latents, return_dict=False)[0]
            
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        return [image]