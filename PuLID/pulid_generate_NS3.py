#!/usr/bin/env python3
"""
pulid_generate_NS2.py - PuLID inference with branched attention support
Compatible with id_grid.sh workflow
"""

import argparse
import os
import json
import random
import numpy as np
import torch
from PIL import Image

# Import PuLID components
from pulid import attention_processor as attention
from pulid.pipeline_NS3 import PuLIDPipeline_NS2
from pulid.utils import resize_numpy_image_long, seed_everything

# Try to import PhotoMaker mask utilities
try:
    import sys
    sys.path.append('../PhotoMaker')
    from photomaker.create_mask_ref import compute_face_mask_from_pil
    from photomaker.mask_utils import compute_binary_face_mask
    MASK_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: PhotoMaker mask utilities not found. Some features may be limited.")
    MASK_UTILS_AVAILABLE = False
    compute_face_mask_from_pil = None

# Default negative prompt (matching PhotoMaker)
DEFAULT_NEGATIVE = "(asymmetry, worst quality, low quality, illustration, 3d, cartoon, sketch)"


def load_numpy_image(path: str, long_side: int = 1024) -> np.ndarray:
    """Load an image file → resized numpy array that PuLID expects."""
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = resize_numpy_image_long(img, long_side)
    return img


def prepare_id_embeddings(pipeline, main_img_path: str) -> torch.Tensor:
    """Compute ID embeddings for a single image."""
    # PuLID's get_id_embedding expects a list of images
    return pipeline.get_id_embedding([load_numpy_image(main_img_path)])


def set_attention_mode(mode: str):
    """Set PuLID attention mode parameters."""
    if mode == "fidelity":
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif mode == "extremely style":
        attention.NUM_ZERO = 16
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'fidelity' or 'extremely style'.")


def main():
    parser = argparse.ArgumentParser(description="PuLID inference with branched attention")
    
    # Required arguments (matching PhotoMaker interface)
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Folder with reference images")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--class_file", type=str, required=True,
                        help="JSON mapping reference basenames → class names")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for generated images")
    
    # PuLID specific parameters
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEGATIVE,
                        help="Negative prompt")
    parser.add_argument("--scale", type=float, default=1.2,
                        help="CFG scale")
    parser.add_argument("--id_scale", type=float, default=0.8,
                        help="ID scale (0-5)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Sampling steps")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output image width")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mode", choices=["fidelity", "extremely style"],
                        default="fidelity", help="Attention mode")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="Number of images to generate per prompt")
    
    # Branched attention parameters (matching PhotoMaker)
    parser.add_argument("--use_branched_attention", action="store_true", default=False,
                        help="Enable branched attention mechanism")
    parser.add_argument("--no_branched_attention", dest="use_branched_attention", 
                        action="store_false",
                        help="Disable branched attention")
    parser.add_argument("--branched_attn_start_step", type=int, default=10,
                        help="Step to start branched attention")
    parser.add_argument("--branched_start_mode", choices=["both", "branched"], 
                        default="both",
                        help="What to enable at branched_attn_start_step")
    
    # Merge step parameters (for compatibility with PhotoMaker workflow)
    parser.add_argument("--start_merge_step", type=int, default=10,
                        help="Step to start ID merge (for compatibility)")
    parser.add_argument("--photomaker_start_step", type=int, default=None,
                        help="Step to switch to ID-enhanced prompts")
    parser.add_argument("--merge_start_step", type=int, default=None,
                        help="Step when face/bg merging starts")
    
    # Face embedding strategy
    parser.add_argument("--face_embed_strategy", choices=["face", "id_embeds"],
                        default="id_embeds",
                        help="Face branch conditioning strategy")
    
    # Runtime-tunable branched attention parameters
    parser.add_argument("--pose_adapt_ratio", type=float, default=0.25,
                        help="Pose adaptation ratio (0=ref, 1=noise)")
    parser.add_argument("--ca_mixing_for_face", type=int, choices=[0, 1], default=1,
                        help="Mix features in face branch K/V")
    parser.add_argument("--use_id_embeds", type=int, choices=[0, 1], default=1,
                        help="Use ID embeddings in face branch")
    parser.add_argument("--force_par_before_pm", type=int, choices=[0, 1], default=0,
                        help="Force pose adaptation before PhotoMaker")
    
    # Mask parameters
    parser.add_argument("--auto_mask_ref", action="store_true",
                        help="Auto-generate reference face mask")
    parser.add_argument("--import_mask_folder", type=str, 
                        default="../compare/testing/gen_masks",
                        help="Folder with pre-generated masks")
    parser.add_argument("--use_mask_folder", type=int, choices=[0, 1], default=1,
                        help="Use masks from folder")
    parser.add_argument("--use_dynamic_mask", type=int, choices=[0, 1], default=0,
                        help="Generate masks dynamically during inference")
    
    # Debug options
    parser.add_argument("--save_heatmaps", action="store_true", default=False,
                        help="Save attention heatmaps")
    
    args = parser.parse_args()
    
    # Set defaults for merge steps if not provided
    if args.photomaker_start_step is None:
        args.photomaker_start_step = args.start_merge_step
    if args.merge_start_step is None:
        args.merge_start_step = args.start_merge_step
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PuLID pipeline with branched attention support
    print("Loading PuLID pipeline with branched attention support...")
    pipeline = PuLIDPipeline_NS2()
    pipeline.debug_img_list = []
    
    # Set attention mode
    print(f"Setting attention mode: {args.mode}")
    set_attention_mode(args.mode)
    
    # Configure branched attention if enabled
    if args.use_branched_attention:
        print("Configuring branched attention...")
        pipeline.setup_branched_attention(
            use_branched=True,
            pose_adapt_ratio=args.pose_adapt_ratio,
            ca_mixing_for_face=bool(args.ca_mixing_for_face),
            use_id_embeds=bool(args.use_id_embeds),
        )
    
    # Load reference images and prompts
    id_image_paths = sorted([
        os.path.join(args.image_folder, f) 
        for f in os.listdir(args.image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [l.strip() for l in f if l.strip()]
    
    with open(args.class_file, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    
    # Set seed
    print(f"Using seed: {args.seed}")
    seed_everything(args.seed)
    BASE_SEED = args.seed
    
    # Output size
    out_size = (1, args.height, args.width)
    
    # Generate images
    total = len(id_image_paths) * len(prompt_list) * args.num_images_per_prompt
    print(f"Generating {total} images...")
    
    for id_path in id_image_paths:
        img_basename = os.path.splitext(os.path.basename(id_path))[0]
        
        # Load reference image
        ref_image = Image.open(id_path).convert("RGB")
        
        # Prepare ID embeddings
        id_embeddings = prepare_id_embeddings(pipeline, id_path)
        # get_id_embedding returns (uncond_id_embedding, id_embedding)
        # Unpack safely to avoid passing a tuple into attention processors
        if isinstance(id_embeddings, (tuple, list)) and len(id_embeddings) == 2:
            uncond_id_embedding, pos_id_embedding = id_embeddings
        else:
            pos_id_embedding = id_embeddings
            uncond_id_embedding = None
        
        # Generate mask if needed
        mask_path = None
        mask_ref_path = None
        
        if args.use_branched_attention:
            # Check for pre-generated mask
            if args.use_mask_folder and not args.use_dynamic_mask:
                mask_candidate = os.path.join(
                    args.import_mask_folder, 
                    f"{img_basename}_gen_mask.png"
                )
                if os.path.exists(mask_candidate):
                    mask_path = mask_candidate
                    print(f"Using imported mask for '{img_basename}'")
            
            # Auto-generate reference mask if requested
            if args.auto_mask_ref and MASK_UTILS_AVAILABLE:
                try:
                    mask_ref = compute_face_mask_from_pil(ref_image)
                    # Save for debugging
                    debug_mask_path = os.path.join(
                        args.output_dir, 
                        f"{img_basename}_auto_ref_mask.png"
                    )
                    Image.fromarray(mask_ref).save(debug_mask_path)
                    mask_ref_path = debug_mask_path
                    print(f"Generated reference mask for '{img_basename}'")
                except Exception as e:
                    print(f"Failed to generate reference mask: {e}")
        
        # Process each prompt
        class_name = class_map.get(img_basename)
        for p_idx, prompt in enumerate(prompt_list):
            cur_prompt = prompt
            
            # Replace <class> placeholder
            if "<class>" in prompt:
                if class_name:
                    cur_prompt = prompt.replace("<class>", class_name)
                else:
                    print(f"Warning: no class for '{img_basename}', keeping '<class>'")
            
            # Generate images
            for img_id in range(args.num_images_per_prompt):
                # Reset seed for reproducibility
                local_seed = BASE_SEED + p_idx * 1000 + img_id
                seed_everything(local_seed)
                
                # Run inference
                if args.use_branched_attention:
                    # Use branched inference method
                    img = pipeline.inference_branched(
                        prompt=cur_prompt,
                        size=out_size,
                        prompt_n=args.neg_prompt,
                        id_embedding=pos_id_embedding,
                        id_scale=args.id_scale,
                        guidance_scale=args.scale,
                        steps=args.steps,
                        seed=local_seed,
                        # Branched parameters
                        use_branched_attention=True,
                        branched_attn_start_step=args.branched_attn_start_step,
                        face_embed_strategy=args.face_embed_strategy,
                        pose_adapt_ratio=args.pose_adapt_ratio,
                        ca_mixing_for_face=bool(args.ca_mixing_for_face),
                        use_id_embeds=bool(args.use_id_embeds),
                        force_par_before_pm=bool(args.force_par_before_pm),
                        # Mask parameters
                        auto_mask_ref=args.auto_mask_ref,
                        import_mask=mask_path,
                        import_mask_ref=mask_ref_path,
                        use_dynamic_mask=bool(args.use_dynamic_mask),
                        reference_image=ref_image,
                    )[0]
                else:
                    # Standard PuLID inference - still setup processors to handle tuples
                    pipeline.setup_branched_attention(use_branched=False)
                    img = pipeline.inference(
                        prompt=cur_prompt,
                        size=out_size,
                        prompt_n=args.neg_prompt,
                        id_embedding=pos_id_embedding,
                        uncond_id_embedding=uncond_id_embedding,
                        id_scale=args.id_scale,
                        guidance_scale=args.scale,
                        steps=args.steps,
                        seed=local_seed,
                    )[0]
                
                # Save image
                filename = os.path.join(
                    args.output_dir,
                    f"{img_basename}_p{p_idx}_{img_id}.jpg"
                )
                img.save(filename)
                print(f"Saved: {filename}")
    
    print(f"\nAll done! Results in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
