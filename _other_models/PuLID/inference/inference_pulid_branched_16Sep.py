#!/usr/bin/env python3
"""
inference_pulid_branched_16Sep.py - Clean PuLID inference with branched attention support
Minimal modifications to standard PuLID inference, with branched attention imported from helpers
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from PIL import Image
import torch

# Add paths for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Standard PuLID imports
from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long

# Import branched pipeline
from pulid.pipeline_branched_16Sep import PuLIDPipelineBranched


def seed_everything(seed: int):
    """Set all random seeds for reproducibility"""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_attention_mode(mode: str):
    """Set PuLID attention mode parameters"""
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
    parser = argparse.ArgumentParser(description="PuLID inference with optional branched attention")
    
    # Dataset-style inputs (matching PhotoMaker interface)
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Folder with reference images")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--class_file", type=str, required=True,
                        help="JSON mapping reference basenames → class names")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for generated images")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="Number of images to generate per prompt")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of denoising steps")
    parser.add_argument("--scale", type=float, default=1.2,
                        help="Classifier-free guidance scale")
    parser.add_argument("--id_scale", type=float, default=0.8,
                        help="ID conditioning scale")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output image width")
    
    # PuLID specific
    parser.add_argument("--mode", choices=["fidelity", "extremely style"], 
                        default="fidelity",
                        help="Attention mode for PuLID")
    parser.add_argument("--neg_prompt", type=str,
                        default="flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, "
                                "worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, "
                                "ugly, disfigured, hands, low resolution, partially rendered objects",
                        help="Negative prompt")
    
    # Branched attention parameters
    parser.add_argument("--use_branched_attention", action="store_true", default=False,
                        help="Enable branched attention mechanism")
    parser.add_argument("--no_branched_attention", dest="use_branched_attention", 
                        action="store_false",
                        help="Disable branched attention")
    parser.add_argument("--branched_attn_start_step", type=int, default=1,
                        help="Step to start branched attention")
    parser.add_argument("--pose_adapt_ratio", type=float, default=0.25,
                        help="Pose adaptation ratio (0=strong identity, 1=more pose flexibility)")
    parser.add_argument("--ca_mixing_for_face", type=int, choices=[0,1], default=1,
                        help="Enable mixing in cross-attention for face branch")
    parser.add_argument("--branched_precision", choices=["auto", "fp16", "fp32", "bf16"], default="auto",
                        help="Precision override for branched attention (auto keeps default)")
    parser.add_argument("--attention_slicing", choices=["none", "auto", "max"], default="none",
                        help="Enable attention slicing in the underlying diffusion pipeline")
    parser.add_argument("--disable_cfg", action="store_true",
                        help="Disable classifier-free guidance to halve UNet batch usage")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set attention mode
    print(f"[Config] Setting attention mode: {args.mode}")
    set_attention_mode(args.mode)
    
    # Set seed
    print(f"[Seed] Using seed = {args.seed}")
    seed_everything(args.seed)
    
    # Load pipeline
    print("[PuLID] Loading pipeline...")
    pipeline = PuLIDPipelineBranched()

    if args.attention_slicing != "none":
        slice_arg = 1 if args.attention_slicing == "max" else "auto"
        pipeline.pipe.enable_attention_slicing(slice_arg)
        print(f"[Memory] Attention slicing enabled ({args.attention_slicing})")

    precision_map = {
        "auto": None,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    branched_dtype = precision_map[args.branched_precision]
    if branched_dtype is not None:
        print(f"[Branched] Using {args.branched_precision} precision for branched attention")

    pipeline.do_classifier_free_guidance = not args.disable_cfg
    if args.disable_cfg:
        print("[CFG] Classifier-free guidance disabled -> smaller UNet batch, less VRAM")
    
    # Load prompts and class mappings
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    
    with open(args.class_file, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    
    # Collect reference images
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ref_images = sorted([
        os.path.join(args.image_folder, f)
        for f in os.listdir(args.image_folder)
        if os.path.splitext(f)[1].lower() in image_exts
    ])
    
    print(f"[Init] Found {len(ref_images)} reference images")
    print(f"[Init] Found {len(prompts)} prompts")
    
    # Process each reference image
    for ref_path in ref_images:
        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
        ref_pil = Image.open(ref_path).convert("RGB")
        
        # Get class name for this reference
        class_name = class_map.get(ref_name, "person")
        
        # Prepare ID embeddings
        print(f"\n[Processing] Reference: {ref_name}")
        np_ref = resize_numpy_image_long(np.array(ref_pil), 1024)
        uncond_id_embedding, id_embedding = pipeline.get_id_embedding([np_ref])
        if args.disable_cfg:
            uncond_id_embedding = None
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            # Replace <class> placeholder if present
            current_prompt = prompt.replace("<class>", class_name) if "<class>" in prompt else prompt
            
            # Generate images
            for img_idx in range(args.num_images_per_prompt):
                print(f"  Generating image {img_idx+1}/{args.num_images_per_prompt} for prompt {prompt_idx+1}")
                
                # Use branched inference if enabled
                if args.use_branched_attention:
                    debug_dir = os.path.join(args.output_dir, "debug", ref_name) if args.use_branched_attention else None
                    
                    images = pipeline.inference_branched(
                        prompt=current_prompt,
                        size=(1, args.height, args.width),
                        prompt_n=args.neg_prompt,
                        id_embedding=id_embedding,
                        uncond_id_embedding=uncond_id_embedding,
                        id_scale=args.id_scale,
                        guidance_scale=args.scale,
                        steps=args.steps,
                        seed=args.seed + img_idx,
                        # Branched attention parameters
                        use_branched_attention=True,
                        branched_attn_start_step=args.branched_attn_start_step,
                        pose_adapt_ratio=args.pose_adapt_ratio,
                        ca_mixing_for_face=bool(args.ca_mixing_for_face),
                        use_id_embeds=True,
                        branched_attention_dtype=branched_dtype,
                        # Pass reference for mask generation
                        reference_pil=ref_pil,
                        debug_dir=debug_dir,
                    )
                else:
                    # Standard inference
                    images = pipeline.inference(
                        prompt=current_prompt,
                        size=(1, args.height, args.width),
                        prompt_n=args.neg_prompt,
                        id_embedding=id_embedding,
                        uncond_id_embedding=uncond_id_embedding,
                        id_scale=args.id_scale,
                        guidance_scale=args.scale,
                        steps=args.steps,
                        seed=args.seed + img_idx,
                    )
                
                # Save the generated image
                if images and len(images) > 0:
                    img = images[0]
                    
                    # Create filename
                    mode_suffix = "_branched" if args.use_branched_attention else ""
                    filename = f"{ref_name}_p{prompt_idx}_{img_idx}{mode_suffix}.png"
                    save_path = os.path.join(args.output_dir, filename)
                    
                    # Save image
                    img.save(save_path)
                    print(f"    Saved: {save_path}")
    
    print(f"\n[Complete] Generated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
