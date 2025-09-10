#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pulid.py – simple PuLID inference script (no Gradio UI)

Example:
    python run_pulid.py \
        --id_image example_inputs/lecun.jpg \
        --prompt "portrait, superman" \
        --mode fidelity \
        --n_samples 4
"""

import argparse
import os
import uuid
import numpy as np
import torch
import json
from PIL import Image

from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything

# ----------------------------------------------------------------------------- #
#  Defaults                                                                     #
# ----------------------------------------------------------------------------- #
DEFAULT_ID_IMAGE      = "example_inputs/lecun.jpg"
DEFAULT_NEGATIVE      = (
    "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, "
    "worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, "
    "ugly, disfigured, hands, low resolution, partially rendered objects, "
    "deformed or partially rendered eyes, deformed eyeballs, cross-eyed, blurry"
)
DEFAULT_PROMPT        = "portrait, superman"
# OUT_DIR               = "./outputs"


# ----------------------------------------------------------------------------- #
#  Helpers                                                                      #
# ----------------------------------------------------------------------------- #
def load_numpy_image(path: str, long_side: int = 1024) -> np.ndarray:
    """Load an image file → resized numpy array that PuLID expects."""
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = resize_numpy_image_long(img, long_side)
    return img


def prepare_id_embeddings(pipeline: PuLIDPipeline,
                          main_img: str,
                          supp_imgs: list[str] | None,
                          id_mix: bool) -> torch.Tensor | None:
    """Compute (and optionally concatenate) ID embeddings."""
    if main_img is None:
        return None

    emb = pipeline.get_id_embedding(load_numpy_image(main_img))

    if supp_imgs:
        for path in supp_imgs:
            if path:
                supp_emb = pipeline.get_id_embedding(load_numpy_image(path))
                # Mix full embedding or only first 5 tokens just like the Gradio demo
                emb = torch.cat((emb, supp_emb if id_mix else supp_emb[:, :5]), dim=1)

    return emb


def set_attention_mode(mode: str):
    """Mimic the same attention knobs as in the Gradio Blocks file."""
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


# ----------------------------------------------------------------------------- #
#  Main                                                                         #
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Pure Python PuLID inference script")
    # parser.add_argument("--id_image",   default=DEFAULT_ID_IMAGE, help="Path to main ID image")
    # parser.add_argument("--supp_image", action="append",
    #                     help="Path(s) to up to three auxiliary ID images (can repeat flag)")
    # parser.add_argument("--id_mix",     action="store_true",
    #                     help="Concatenate full embeddings of auxiliary IDs instead of first 5 tokens")
    # parser.add_argument("--prompt",     default=DEFAULT_PROMPT, help="Positive prompt")
    
    parser.add_argument("--image_folder", required=True,
                        help="Folder with reference images (one face per image)")
    parser.add_argument("--prompt_file",  required=True,
                        help="Text file with one prompt per line")    
    parser.add_argument("--class_file",   required=True,
                        help="JSON mapping reference basenames → class names")
    
    parser.add_argument("--neg_prompt", default=DEFAULT_NEGATIVE, help="Negative prompt")
    parser.add_argument("--scale",      type=float, default=1.2,  help="CFG scale (≃1-1.5)")
    parser.add_argument("--id_scale",   type=float, default=0.8,  help="ID scale (0-5)")
    parser.add_argument("--steps",      type=int,   default=4,    help="DDIM steps")
    parser.add_argument("--n_samples",  type=int,   default=3,    help="Images to generate")
    parser.add_argument("--height",     type=int,   default=1024, help="Output image height")
    # parser.add_argument("--width",      type=int,   default=768,  help="Output image width")
    parser.add_argument("--width",      type=int,   default=1024,  help="Output image width")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    parser.add_argument("--mode",       choices=["fidelity", "extremely style"],
                        default="fidelity", help="Attention mode")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Output directory for generated images")

    args = parser.parse_args()
    
    output_dir = args.output_dir

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PuLID pipeline …")
    pipeline = PuLIDPipeline()
    pipeline.debug_img_list = []  # still available if you need it

    print("Setting attention mode:", args.mode)
    set_attention_mode(args.mode)

    # print("Preparing ID embeddings …")
    # id_embeddings = prepare_id_embeddings(
    #     pipeline,
    #     args.id_image,
    #     (args.supp_image or [])[:3],  # respect the original 3-aux limit
    #     args.id_mix,
    # )
    
    # ------------------------------------------------------------------ #
    #  Load reference IDs & prompts                                      #
    # ------------------------------------------------------------------ #
    id_image_paths = sorted(
        [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)]
    )
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [l.strip() for l in f if l.strip()]
        
    class_map = json.load(open(args.class_file, "r", encoding="utf-8"))
    

    print(f"Seeding: {args.seed}")
    seed_everything(args.seed)

    H, W = args.height, args.width
    out_size = (1, H, W)

    
    total = len(id_image_paths) * len(prompt_list) * args.n_samples
    print(f"Generating {total} images …")

    for id_path in id_image_paths:
        img_basename = os.path.splitext(os.path.basename(id_path))[0]
        id_embeddings = prepare_id_embeddings(pipeline, id_path, None, False)

        # for p_idx, prompt in enumerate(prompt_list):
        class_name = class_map.get(img_basename)                         # ⭐
        for p_idx, prompt in enumerate(prompt_list):
            cur_prompt = prompt
            if "<class>" in prompt:                                      # ⭐
                if class_name:
                    cur_prompt = prompt.replace("<class>", class_name)
                else:
                    print(f"⚠️  no class for '{img_basename}', kept '<class>'")
            
            for img_id in range(args.n_samples):
                img = pipeline.inference(
                    # prompt,
                    cur_prompt,
                    out_size,
                    args.neg_prompt,
                    id_embeddings,
                    args.id_scale,
                    args.scale,
                    args.steps,
                )[0]

                filename = os.path.join(
                    output_dir, f"{img_basename}_p{p_idx}_{img_id}.jpg"
                )
                img.save(filename)
                print("✅ saved", filename)

    print("\nAll done!  Results in", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
