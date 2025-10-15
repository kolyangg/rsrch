"""
add_masking.py - Dynamic mask generation from attention maps during inference
"""

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
import torch
import math
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
try:
    import matplotlib.cm as cm
except Exception:  # pragma: no cover
    cm = None
from PIL import ImageDraw, ImageFont


def get_default_mask_config() -> List[Dict]:
    """Return the default mask layer configuration."""

    # return [
    #     {
    #         "name": "up_blocks.0.attentions.1.transformer_blocks.1.attn2",
    #         "weight": 0.25,
    #         "top_ratio": 0.10,
    #         "invert": False
    #     },
    #     {
    #         "name": "up_blocks.0.attentions.1.transformer_blocks.7.attn2",
    #         "weight": 0.75,
    #         "top_ratio": 0.05,
    #         "invert": True
    #     }
    # ]
    
    return [
        {
            "name": "up_blocks.0.attentions.1.transformer_blocks.2.attn2",
            "weight": 1.00,
            "top_ratio": 0.10,
            "invert": False
        },
        # {
        #     "name": "up_blocks.0.attentions.1.transformer_blocks.7.attn2",
        #     "weight": 0.75,
        #     "top_ratio": 0.05,
        #     "invert": True
        # }
    ]


def create_mask_from_attention_maps(
    attn_maps: Dict[str, np.ndarray],
    mask_layers_config: List[Dict],
    debug: bool = False, 
    target_size: tuple = (128, 128)  # Higher resolution for smoother mask
) -> np.ndarray:
    """
    Create a binary mask from attention maps using the same algorithm as attn_hm_NS_nosm7.py
    
    Args:
        attn_maps: Dictionary of layer_name -> attention map (2D numpy array)
        mask_layers_config: List of layer configurations with weights, thresholds, etc.
        debug: Whether to print debug information
        target_size: Target size for the final mask (H, W)
        
    Returns:
        Binary mask as numpy array (H x W) with bool dtype
    """
    if not attn_maps or not mask_layers_config:
        return None
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for dynamic mask generation in the branched PhotoMaker pipeline. "
            "Install it with `pip install opencv-python` or disable dynamic masking."
        )
    
    # Set defaults for mask layer configs
    for m in mask_layers_config:
        m.setdefault("top_ratio", 0.10)
        m.setdefault("invert", False)
        m.setdefault("weight", 1.0)
    
    # Find the largest resolution among selected layers
    base_H = 0
    for spec in mask_layers_config:
        if spec["name"] in attn_maps:
            H = attn_maps[spec["name"]].shape[0]
            base_H = max(base_H, H)
    
    if base_H == 0:
        return None
    
    # Initialize combined mask
    combined_bin = np.zeros((base_H, base_H), dtype=np.float32)
    total_weight = 0.0
    
    # Process each layer according to config
    for spec in mask_layers_config:
        layer_name = spec["name"]
        if layer_name not in attn_maps:
            if debug:
                print(f"[MaskGen] Warning: layer '{layer_name}' not in attention maps")
            continue
        
        amap = attn_maps[layer_name]
        
        # Resize to base resolution if needed
        if amap.shape[0] != base_H:
            amap = np.array(
                Image.fromarray(amap).resize((base_H, base_H), Image.BILINEAR)
            )
        
        # Normalize
        amap_n = amap / amap.max() if amap.max() > 0 else amap
        
        # Apply threshold based on invert flag
        if spec["invert"]:
            threshold = np.quantile(amap_n, spec["top_ratio"])
            selection = (amap_n < threshold)
        else:
            threshold = np.quantile(amap_n, 1.0 - spec["top_ratio"])
            selection = (amap_n > threshold)
        
        # Accumulate weighted selection
        combined_bin += spec["weight"] * selection.astype(np.float32)
        total_weight += spec["weight"]
    
    # Normalize and binarize
    mask = (combined_bin / max(total_weight, 1e-6)) > 0.0
    
    # Keep largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx)
    
    # Apply convex hull
    main_blob = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        main_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if contours:
        hull = cv2.convexHull(contours[0])
        mask = np.zeros_like(main_blob)
        cv2.drawContours(mask, [hull], -1, 1, -1)
        
        # Limit over-expansion (same as original)
        HULL_MAX_AREA_FACTOR = 1.0
        area_orig = main_blob.sum()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        while mask.sum() > HULL_MAX_AREA_FACTOR * area_orig:
            mask = cv2.erode(mask, kernel, 1)
            if not mask.any():
                mask = main_blob.copy()
                break
        
        mask = mask.astype(bool)

    # Upscale mask to target resolution with smoothing
    if mask is not None and target_size is not None:
        target_h, target_w = target_size
        if mask.shape != (target_h, target_w):
            # Convert to float for smooth interpolation
            mask_float = mask.astype(np.float32)
            mask_resized = cv2.resize(mask_float, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            # Apply gaussian blur for smoother edges
            mask_resized = cv2.GaussianBlur(mask_resized, (5, 5), 1.0)
            mask = mask_resized > 0.5  # Re-threshold to binary
    
    return mask


def make_attention_hook(
    layer_name: str,
    module: Any,
    class_tokens_mask: Optional[torch.Tensor],
    num_tokens: int,
    do_cfg: bool,
    heatmap_mode: str,  # "identity" or "focus_token"
    token_focus: str,
    focus_latents: Optional[torch.Tensor],
    token_idx_global: Optional[List[int]],
    attn_maps_buffer: Dict[str, List[np.ndarray]]
) -> callable:
    """
    Create a forward hook for collecting attention maps from a specific layer.
    """
    orig_forward = module.forward
    
    def hooked_forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Run original forward
        out = orig_forward(hidden_states, encoder_hidden_states, attention_mask)
        
        # Only process cross-attention with encoder states
        if encoder_hidden_states is None:
            return out
        
        # Handle CFG - use only conditional batch
        # B_all = hidden_states.shape[0]
        # if do_cfg and B_all % 2 == 0:
        #     hs_cond = hidden_states[B_all // 2:]
        #     enc_cond = encoder_hidden_states[B_all // 2:]
        # else:
        #     hs_cond = hidden_states
        #     enc_cond = encoder_hidden_states

        B_all = hidden_states.shape[0]
        E_all = encoder_hidden_states.shape[0]
        # Only slice encoder states if they truly match latent batch (diffusers already expands enc states in many paths)
        if do_cfg and (B_all % 2 == 0) and (E_all == B_all):
            hs_cond = hidden_states[B_all // 2:]
            enc_cond = encoder_hidden_states[B_all // 2:]
        else:
            hs_cond = hidden_states
            enc_cond = encoder_hidden_states

        # Align batch sizes: tile encoder states up to latent half-batch
        if enc_cond.shape[0] != hs_cond.shape[0]:
            rep = (hs_cond.shape[0] + enc_cond.shape[0] - 1) // enc_cond.shape[0]
            enc_cond = enc_cond.repeat(rep, 1, 1)[:hs_cond.shape[0]].contiguous()
        # dtype safety
        enc_cond = enc_cond.to(hs_cond.dtype)
        
        # Project Q and K
        q_proj = (module.to_q if hasattr(module, "to_q") else module.q_proj)(hs_cond)
        
        if heatmap_mode == "focus_token" and focus_latents is not None and token_idx_global:
            # Focus token mode: use specific token embeddings
            k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(
                # focus_latents.to(hs_cond.dtype).repeat(hs_cond.shape[0], 1, 1)
                focus_latents.to(hs_cond.dtype)
                .repeat(hs_cond.shape[0], 1, 1)
            )
        else:
            # Identity mode: use encoder hidden states directly
            k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(enc_cond)
        
        B, L, C = q_proj.shape
        h = module.heads
        d = C // h
        # k_proj now has batch B and inner-dim compatible with q_proj
        
        Q = q_proj.view(B, L, h, d).permute(0, 2, 1, 3)
        K = k_proj.view(B, -1, h, d).permute(0, 2, 1, 3)
        
        # Compute attention logits
        logits = (Q @ K.transpose(-2, -1)) * module.scale
        
        # Select relevant tokens
        if heatmap_mode == "focus_token" and token_idx_global:
            # Focus token mode: use specific token indices
            if len(token_idx_global) > 1:
                att = logits[..., token_idx_global].mean(-1).mean(1)[0]
            else:
                att = logits[..., token_idx_global[0]].mean(1)[0]
        else:
            # Identity mode: use PhotoMaker's ID tokens (where face embeddings are injected)
            if class_tokens_mask is not None:
                # Use the actual ID token positions from class_tokens_mask
                # These are the positions where PhotoMaker injects the face embeddings
                token_idx = class_tokens_mask[0].to(hidden_states.device).nonzero(as_tuple=True)[0]
                att = logits[..., token_idx].mean(-1).mean(1)[0]
            else:
                # Fallback: use last num_tokens if no class_tokens_mask
                # This maintains backward compatibility
                token_idx = torch.arange(K.shape[2] - num_tokens, K.shape[2], device=hidden_states.device)
                att = logits[..., token_idx].mean(-1).mean(1)[0]
       
        # Reshape to 2D and store
        H = int(math.sqrt(att.numel()))
        att2d = att.to(torch.float32).view(H, H).cpu().numpy()
        
        attn_maps_buffer.setdefault(layer_name, []).append(att2d)
        
        return out
    
    return hooked_forward


class DynamicMaskGenerator:
    """
    Manages dynamic mask generation during inference.
    Encapsulates all the logic for setting up hooks, collecting attention maps,
    and generating masks.
    """
    
    def __init__(
        self,
        pipeline,
        use_dynamic_mask: bool = False,
        mask_start: int = 10,
        mask_end: int = 15,
        save_heatmaps: bool = True,
        token_focus: str = "face",
        add_to_prompt: bool = False,
        mask_layers_config: Optional[List[Dict]] = None,
        debug_dir: str = "hm_debug",
        mask_resolution: int = 128,  # Resolution for mask generation
        save_hm_pdf: bool = False,
        heatmap_interval: int = 5,  # Save heatmap every N steps
        num_inference_steps: int = 50,
        heatmap_mode: str = "identity"  # "identity" or "focus_token"
    ):
        self.pipeline = pipeline
        self.use_dynamic_mask = use_dynamic_mask
        self.mask_start = mask_start
        self.mask_end = mask_end
        self.save_heatmaps = save_heatmaps
        self.token_focus = token_focus
        self.add_to_prompt = add_to_prompt
        self.mask_layers_config = mask_layers_config or get_default_mask_config()
        self.debug_dir = debug_dir
        self.mask_resolution = mask_resolution
        self.save_hm_pdf = save_hm_pdf
        self.heatmap_interval = heatmap_interval
        self.num_inference_steps = num_inference_steps
        self.heatmap_mode = heatmap_mode


        # State variables
        self.attn_maps = {}
        self.attn_maps_all_steps = {}  # Store attention maps for all steps
        self.current_mask = None
        self.mask_finalized = False
        self.hooks_installed = False
        self.original_forwards = {}
        
        # Focus latents for token-based masking
        self.focus_latents = None
        self.token_idx_global = None

        # Heatmap collection
        self.heatmap_frames = {}  # layer_name -> list of PIL images
        self.step_labels = []
    
    def setup_hooks(self, prompt: str, class_tokens_mask: Optional[torch.Tensor] = None, num_tokens: int = None):
        
        """
        Install forward hooks on the UNet attention layers to collect attention maps.
        """
        if not self.use_dynamic_mask or self.hooks_installed:
            return
        
        device = self.pipeline.device

        # Store num_tokens for identity mode
        if num_tokens is not None:
            self.pipeline.num_tokens = num_tokens
        
        # Prepare focus latents if using focus_token mode
        if self.heatmap_mode == "focus_token" and self.token_focus:
            focus_prompt = f"a {self.token_focus}"
            if self.add_to_prompt and self.token_focus not in prompt.split():
                focus_prompt = f"{prompt} {self.token_focus}"
            
            with torch.no_grad():
                self.focus_latents, *_ = self.pipeline.encode_prompt(
                    prompt=focus_prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )
            
            # Find token indices
            tokenizer = self.pipeline.tokenizer or self.pipeline.tokenizer_2
            prompt_ids = tokenizer(focus_prompt, add_special_tokens=False).input_ids
            word_ids = tokenizer(" " + self.token_focus, add_special_tokens=False).input_ids
            
            def find_sub(seq, sub):
                for i in range(len(seq) - len(sub) + 1):
                    if seq[i:i + len(sub)] == sub:
                        return list(range(i, i + len(sub)))
                return []
            
            self.token_idx_global = find_sub(prompt_ids, word_ids)
            if not self.token_idx_global:
                print(f"[DynamicMask] Warning: Could not locate '{self.token_focus}' in prompt")
                print(f"[DynamicMask] Falling back to identity mode")
                self.heatmap_mode = "identity"
                self.token_idx_global = None
        else:
            # Identity mode doesn't need focus latents
            self.focus_latents = None
            self.token_idx_global = None
        
        # Install hooks - if save_heatmap_pdf is True, track ALL cross-attention layers
        # Otherwise only track mask generation layers
        from diffusers.models.attention_processor import Attention as CrossAttention
        
        wanted_layers = {spec["name"] for spec in self.mask_layers_config}
        if self.save_heatmap_pdf:
            # Track ALL cross-attention layers for heatmap visualization
            wanted_layers = set()
            for name, module in self.pipeline.unet.named_modules():
                if isinstance(module, CrossAttention) and "attn2" in name:
                    wanted_layers.add(name)
        else:
            # Only track specified layers for mask generation
            wanted_layers = {spec["name"] for spec in self.mask_layers_config}


        hooks_count = 0
        
        for name, module in self.pipeline.unet.named_modules():
            if isinstance(module, CrossAttention) and name in wanted_layers:
                self.original_forwards[name] = module.forward
                module.forward = make_attention_hook(
                    name,
                    module,
                    class_tokens_mask,
                    self.pipeline.num_tokens,
                    self.pipeline.do_classifier_free_guidance,
                    self.heatmap_mode,
                    self.token_focus,
                    self.focus_latents,
                    self.token_idx_global,
                    self.attn_maps
                )
                hooks_count += 1
        
        self.hooks_installed = True
        print(f"[DynamicMask] Installed hooks on {hooks_count}/{len(wanted_layers)} layers (mode: {self.heatmap_mode})")
    
    # def update_mask(self, step: int):
    def update_mask(self, step: int, latents: Optional[torch.Tensor] = None):
        """
        Update the mask based on collected attention maps if within the update window.
        """
        if not self.use_dynamic_mask:
            return
                
        # Clear attention maps at step 0 to avoid accumulation from previous runs
        if step == 0:
            self.attn_maps.clear()
            self.attn_maps_all_steps.clear()

        # Always collect attention maps if save_hm_pdf is enabled
        if self.save_hm_pdf:
            # Store current attention maps for heatmap visualization
            if self.attn_maps:
                # Average the collected maps for this step
                step_snapshot = {}
                for layer_name, maps_list in self.attn_maps.items():
                    if maps_list:
                        step_snapshot[layer_name] = np.stack(maps_list).mean(0)
                
                # Store for later PDF generation
                if step_snapshot:
                    # Initialize if needed
                    for layer_name in step_snapshot:
                        if layer_name not in self.attn_maps_all_steps:
                            self.attn_maps_all_steps[layer_name] = []
                    
                   # Store snapshot for PDF generation
                    # Only save at exact intervals matching attn_hm_NS_nosm7.py
                    # For 50 steps with interval 10: save at 0, 10, 20, 30, 40, 50
                    # Include step 50 even if num_inference_steps is 49
                    should_save = ((step % self.heatmap_interval == 0 and step <= 50) or
                                   step == self.num_inference_steps - 1)  # Last step (49 becomes 50)

                    if should_save and latents is not None:
                        print(f"[DynamicMask] Saving heatmap for step {step}")
                        # Decode latents for this step
                        with torch.no_grad():
                            vae = self.pipeline.vae
                            vae_device = next(vae.parameters()).device
                            vae_dtype = next(vae.parameters()).dtype
                            
                            lat_scaled = (latents[0:1] / vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)
                            img = vae.decode(lat_scaled).sample[0]
                            img_np = ((img.float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        
                        # Store the snapshot with the decoded image
                        for layer_name, amap in step_snapshot.items():
                            self.attn_maps_all_steps[layer_name].append({
                                'step': step,
                                'amap': amap,
                                'image': img_np
                            })
        
        # if step < self.mask_start or step > self.mask_end or self.mask_finalized:
        # Only update mask if within the mask window
        if step < self.mask_start or step > self.mask_end or self.mask_finalized:
            # Clear buffers for next step
            self.attn_maps.clear()        
            return
        
        # Aggregate attention maps
        # Only process mask layers for mask generation
        mask_layer_names = {spec["name"] for spec in self.mask_layers_config}
        snapshot = {}
        for layer_name, maps_list in self.attn_maps.items():
            if maps_list and layer_name in mask_layer_names:
                snapshot[layer_name] = np.stack(maps_list).mean(0)
       
        if snapshot:
            # Get target size from pipeline latent resolution
            # Latents are typically 1/8 of image size, so mask should match latent grid
            # For 1024x1024 images, latents are 128x128
            target_size = (self.mask_resolution, self.mask_resolution)
            # Create mask from attention maps
            new_mask = create_mask_from_attention_maps(
                snapshot,
                self.mask_layers_config,
                debug=(step == self.mask_start),
                target_size=target_size
            )
            
            if new_mask is not None:
                self.current_mask = new_mask
                print(f"[DynamicMask] Updated at step {step}, shape={new_mask.shape}, face_pixels={new_mask.sum()}")
                
                # Save visualization if requested
                if self.save_heatmaps:
                    import os
                    os.makedirs(self.debug_dir, exist_ok=True)
                    mask_vis = (new_mask.astype(np.uint8) * 255)
                    Image.fromarray(mask_vis).save(f"{self.debug_dir}/dynamic_mask_step_{step:03d}.png")


        # # Clear buffers for next step
        # self.attn_maps.clear()

        # # Finalize if we've reached the end
        # if step == self.mask_end:
        #     self.mask_finalized = True
        #     print(f"[DynamicMask] Finalized at step {step}")
            
            # # Save PDF if we have heatmaps
            # if self.save_heatmap_pdf and self.heatmap_frames:
            #     self._save_heatmap_pdf()
    
    # def _save_heatmap_frame(self, snapshot: Dict[str, np.ndarray], latents: torch.Tensor, step: int):
    #     """Save heatmap overlays for each layer at current step."""
    #     # Decode latents to RGB
    #     with torch.no_grad():
    #         vae = self.pipeline.vae
    #         vae_device = next(vae.parameters()).device
    #         vae_dtype = next(vae.parameters()).dtype
            
    #         lat_scaled = (latents[0:1] / vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)
    #         img = vae.decode(lat_scaled).sample[0]
    #         img_np = ((img.float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
    #     # Create heatmap overlay for each layer
    #     colormap = cm.get_cmap("jet")
        
    #     for layer_name, amap in snapshot.items():
    #         # Normalize attention map
    #         amap_norm = (amap / amap.max()) if amap.max() > 0 else amap
            
    #         # Create colored heatmap
    #         hmap = (colormap(amap_norm)[..., :3] * 255).astype(np.uint8)
    #         hmap = np.array(Image.fromarray(hmap).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR))
            
    #         # Blend with original image
    #         heat_np = (0.5 * img_np + 0.5 * hmap).astype(np.uint8)
    #         heat_img = Image.fromarray(heat_np)
            
    #         # Store frame
    #         if layer_name not in self.heatmap_frames:
    #             self.heatmap_frames[layer_name] = []
    #         self.heatmap_frames[layer_name].append(heat_img)
        
    #     # Store step label
    #     self.step_labels.append(f"S{step}")


    def save_heatmap_pdf(self, final_image: Optional[np.ndarray] = None):
        """
        Save collected heatmaps as a multi-page PDF.
        Call this after inference completes.
        """
        if not self.save_hm_pdf or not self.attn_maps_all_steps:
            return
        
        print(f"[DynamicMask] Creating heatmap PDF with {len(self.attn_maps_all_steps)} layers")
        
        if cm is None:
            raise ImportError(
                "matplotlib is required for saving heatmap PDFs. "
                "Install it with `pip install matplotlib` or disable `save_hm_pdf`."
            )

        # Sort layer names for consistent ordering
        layer_names = sorted(self.attn_maps_all_steps.keys())
        colormap = cm.get_cmap("jet")

        # Constants from attn_hm_NS_nosm7.py
        header_h = 30
        pages = []

        # Process each layer and create strips
        strips = []
        for layer_name in layer_names:
            layer_data = self.attn_maps_all_steps[layer_name]
            if not layer_data:
                continue

            
            print(f"[DynamicMask] Processing layer {layer_name} with {len(layer_data)} frames")
            
            frames = []
            step_labels = []
            
            # Create heatmap overlay for each saved step
            for data in layer_data:
                step = data['step']
                amap = data['amap']
                img_np = data['image']
                
                # Normalize attention map
                amap_norm = (amap / amap.max()) if amap.max() > 0 else amap
                
                # Create colored heatmap
                hmap = (colormap(amap_norm)[..., :3] * 255).astype(np.uint8)
                hmap = np.array(Image.fromarray(hmap).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR))
                
                # Blend with original image
                heat_np = (0.5 * img_np + 0.5 * hmap).astype(np.uint8)
                heat_img = Image.fromarray(heat_np)
                
                frames.append(heat_img)
                step_labels.append(f"S{step}")
            
            # Add final clean image if provided
            if final_image is not None:
                final_img = Image.fromarray(final_image)
                frames.append(final_img)
                step_labels.append("Final")
            
            # Create strip for this layer
            if frames:
                img_w, img_h = frames[0].size
                strip_w = img_w * len(frames)
                strip_h = img_h + header_h

                strip = Image.new("RGB", (strip_w, strip_h), "black")
                draw = ImageDraw.Draw(strip)
                
                try:
                    font = ImageFont.load_default()
                except:
                    font = None


                
                # Paste frames and add labels
                for idx, (frame, label) in enumerate(zip(frames, step_labels)):
                    x_off = idx * img_w
                    strip.paste(frame, (x_off, header_h))
                    
                    if font:
                        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                        draw.text((x_off + (img_w - tw) // 2, (header_h - th) // 2),
                                 label, font=font, fill="white")
                
                strips.append((layer_name, strip))
        

        # Create pages with multiple strips per page (matching attn_hm_NS_nosm7.py layout)
        if strips:
            # Layout constants from attn_hm_NS_nosm7.py
            ROWS_PER_PAGE = 10
            LABEL_W_PX = 200
            ROW_H_PX = 150
            MARGIN = 20

            # Get dimensions from first strip
            first_strip = strips[0][1]
            strip_h = first_strip.height
            
            # Calculate image width to fit all 7 images (6 steps + final)
            # Total available width minus label and margins
            total_available = 1400  # Approximate page width for PDF
            imgs_per_row = 7  # S0, S10, S20, S30, S40, S50, Final
            target_img_w = int((total_available - LABEL_W_PX - 2*MARGIN) / imgs_per_row)
            
            # Calculate page dimensions based on strips
            # PAGE_W = LABEL_W_PX + target_img_w * 7 + MARGIN  # 7 images: S0,S10,S20,S30,S40,S50,Final
            PAGE_W = LABEL_W_PX + target_img_w * imgs_per_row + 2*MARGIN
            PAGE_H = MARGIN + ROW_H_PX * min(ROWS_PER_PAGE, len(strips))

           
            # Create pages
            for page_idx in range(0, len(strips), ROWS_PER_PAGE):
                page_strips = strips[page_idx:page_idx + ROWS_PER_PAGE]
                page_h = MARGIN + ROW_H_PX * len(page_strips)
                
                page = Image.new("RGB", (PAGE_W, page_h), "white")
                draw = ImageDraw.Draw(page)
               
                # Font for layer names
                try:
                    from PIL import ImageFont
                   # Try to get a slightly bigger font
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                    except:
                        font = ImageFont.load_default()
                except:
                    font = None
                
                # Draw each strip on the page
                y = MARGIN
                for layer_name, strip in page_strips:
                    # Scale strip to fit properly
                    orig_w = strip.width
                    orig_h = strip.height

                    # Calculate actual number of images in this strip
                    # Should be 7: 6 heatmaps (S0,S10,S20,S30,S40,S50) + 1 final
                    num_frames = orig_w // (orig_h - 30)  # Approximate based on aspect ratio
                    print(f"[DynamicMask] Layer {layer_name}: {num_frames} frames detected")
                    new_w = target_img_w * imgs_per_row  # Always scale to show 7 images


                    new_h = int(orig_h * (new_w / orig_w))
                    strip = strip.resize((new_w, new_h), Image.LANCZOS)
                    # Draw layer name in black on white background
                    if font:
                        # Wrap long layer names across multiple lines
                        max_width = LABEL_W_PX - 20  # Leave some padding
                        words = layer_name.replace("_", "_\n").replace(".", ".\n").split("\n")
                        lines = []
                        current_line = ""
                        

                        for word in words:
                            test_line = current_line + word if not current_line else current_line + word
                            if font.getlength(test_line) <= max_width:
                                current_line = test_line
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        if current_line:
                            lines.append(current_line)
                        
                        # Calculate vertical position to center the text block
                        line_height = 12  # Approximate line height
                        text_block_height = len(lines) * line_height
                        start_y = y + (ROW_H_PX - text_block_height) // 2
                        
                        # Draw each line
                        for i, line in enumerate(lines):
                            draw.text((10, start_y + i * line_height), 
                                     line, font=font, fill="black")
                    
                    # Paste the strip
                    page.paste(strip, (LABEL_W_PX, y))
                    y += ROW_H_PX
                
                pages.append(page)
        
        # Save multi-page PDF
        if pages:
            out_dir = Path(self.debug_dir) / "hm_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = out_dir / "attention_heatmaps.pdf"
            pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
            print(f"[DynamicMask] Saved heatmap PDF to {pdf_path} with {len(pages)} pages")
        else:
            print("[DynamicMask] No heatmap pages to save")
            return
    


        
        # out_dir = Path(self.debug_dir) / "hm_results"
        # out_dir.mkdir(parents=True, exist_ok=True)
        
        # # Create pages for each layer
        # pages = []
        # header_h = 30
        
        # for layer_name, frames in self.heatmap_frames.items():
        #     if not frames:
        #         continue
            
        #     # Create strip for this layer
        #     img_w, img_h = frames[0].size
        #     strip_w = img_w * len(frames)
        #     strip_h = img_h + header_h
            
        #     strip = Image.new("RGB", (strip_w, strip_h), "black")
        #     draw = ImageDraw.Draw(strip)
            
        #     # Try to load a font, fall back to default
        #     try:
        #         font = ImageFont.load_default()
        #     except:
        #         font = None
            
        #     # Paste frames and add labels
        #     for idx, (frame, label) in enumerate(zip(frames, self.step_labels)):
        #         x_off = idx * img_w
        #         strip.paste(frame, (x_off, header_h))
                
        #         if font:
        #             tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        #             draw.text((x_off + (img_w - tw) // 2, (header_h - th) // 2),
        #                      label, font=font, fill="white")
            
        #     pages.append(strip)
        
        # # Save multi-page PDF
        # if pages:
        #     pdf_path = out_dir / "attention_heatmaps.pdf"
        #     pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        #     print(f"[DynamicMask] Saved heatmap PDF to {pdf_path}")


        # # Clear buffers for next step
        # self.attn_maps.clear()
        
        # # Finalize if we've reached the end
        # if step == self.mask_end:
        #     self.mask_finalized = True
        #     print(f"[DynamicMask] Finalized at step {step}")
    
    def get_mask_for_pipeline(self):
        """
        Get the current mask in the format expected by the pipeline.
        Returns the mask as numpy array and tensor.
        """
        if self.current_mask is not None:
            mask_tensor = torch.from_numpy(self.current_mask.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
            return self.current_mask, mask_tensor
        return None, None
    
    def cleanup(self):
        """
        Remove hooks and clean up.
        """
        if not self.hooks_installed:
            return
        
        from diffusers.models.attention_processor import Attention as CrossAttention
        
        for name, module in self.pipeline.unet.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        
        self.original_forwards.clear()
        self.hooks_installed = False
        print("[DynamicMask] Hooks removed")
