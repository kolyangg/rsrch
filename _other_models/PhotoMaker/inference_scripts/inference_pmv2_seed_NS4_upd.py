# inferece_scripts/inference_pmv2_seed_NS4_upd.py

# !pip install opencv-python transformers accelerate
import os
import sys
import argparse    
import json

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline2 as PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces

import argparse                              # ⭐ for CLI seed
import random                                # ⭐ optional; keep runs reproducible

# ── debug helpers (new)
from pm_debug import make_mask_callback, save_strip

face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (int); leave blank for a random run")
parser.add_argument("--image_folder", type=str, required=True,
                    help="Folder with reference images (one face per image)")
parser.add_argument("--prompt_file", type=str, required=True,
                    help="Text file with one prompt per line")
parser.add_argument("--class_file", type=str, required=True,
                    help="JSON mapping reference basenames → class names (e.g. {\"elon\":\"man\"})")
parser.add_argument("--num_images_per_prompt", type=int, default=3,
                    help="Images to generate for each prompt / ID")
parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Output directory for generated images")
# ───────────── NEW: branched-attention options ─────────────
parser.add_argument("--use_branched_attention", action="store_true", default=False,
                    help="Activate the new face/background branched attention")
parser.add_argument("--no_branched_attention", dest="use_branched_attention", action="store_false", help="Disable the branched attention mechanism")
parser.add_argument("--save_heatmaps", action="store_true", default=False,
                    help="Save heatmaps during inference")
parser.add_argument("--branched_start_step", type=int, default=10,
                    help="Denoising step at which branched attention kicks in")
parser.add_argument("--face_embed_strategy", choices=["faceanalysis", "heatmap"],
                    default="faceanalysis",
                    help="Reference-face embedding to use in the face branch")
args = parser.parse_args()

try:
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
if device == "mps":
    torch_dtype = torch.float16
    

if args.seed is None:
    args.seed = random.randint(0, 2**32 - 1)
print(f"[Seed] Using seed = {args.seed}")

generator = torch.Generator(device).manual_seed(args.seed)  # ⭐



# output_dir = "./outputs"
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")

# prompt = "instagram photo, portrait photo of a woman img, colorful, perfect face, natural skin, hard shadows, film grain, best quality"
with open(args.prompt_file, "r", encoding="utf-8") as f:
    prompt_list = [l.strip() for l in f if l.strip()]
    
# ⭐ load class-mapping once
with open(args.class_file, "r", encoding="utf-8") as f:
    class_map = json.load(f)

negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

# initialize the models and pipeline
### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch_dtype
).to("cuda")

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"  # define the trigger word
)     
### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="lcm-lora")
# pipe.set_adapters(["photomaker", "lcm-lora"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

# ── create one reusable callback & frame-store ───────────────────────────
frames_holder = []
mask_cb       = make_mask_callback(pipe, mask_interval=5, container=frames_holder)



### define the input ID images
image_basename_list = os.listdir(args.image_folder)
image_path_list    = sorted([os.path.join(args.image_folder, n) for n in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

id_embed_list = []

   
# load images, run face detector and keep lists aligned
input_id_images, id_embed_list, id_basename_list = [], [], []
for image_path in image_path_list:
    pil_img = load_image(image_path)
    img_np  = np.array(pil_img)[:, :, ::-1]
    faces   = analyze_faces(face_detector, img_np)
    if faces:                                     # only keep usable refs
        input_id_images.append(pil_img)
        id_embed_list.append(torch.from_numpy(faces[0]["embedding"]))
        id_basename_list.append(os.path.splitext(os.path.basename(image_path))[0])        
        

if len(id_embed_list) == 0:
    raise ValueError(f"No face detected in input image pool")

    
for ref_img, ref_embed, img_basename in zip(input_id_images, id_embed_list, id_basename_list):
    id_embeds = ref_embed.unsqueeze(0)

    class_name = class_map.get(img_basename)              # ⭐ lookup class
    for p_idx, prompt in enumerate(prompt_list):
        cur_prompt = prompt
        if "<class>" in prompt:
            if class_name:
                replace_str = f"{class_name}" + " img"
                cur_prompt = prompt.replace("<class>", replace_str)
            else:
                print(f"⚠️  No class found for '{img_basename}', keeping '<class>'")
        images = pipe(
            cur_prompt,
            negative_prompt=negative_prompt,
            input_id_images=[ref_img],
            id_embeds=id_embeds,
            num_images_per_prompt=args.num_images_per_prompt,
            start_merge_step=10,
            generator=generator,
            # ─── forward the new flags ────────────────────────────────
            use_branched_attention=args.use_branched_attention,
            save_heatmaps=args.save_heatmaps,  # ← NEW
            branched_attn_start_step=args.branched_start_step,
            face_embed_strategy=args.face_embed_strategy,  
            # debug_save_masks=True,          # ← NEW
            # mask_interval=5,                # ← NEW (optional)
            # debugging (handled by callback)
            callback_on_step_end            = mask_cb,
            callback_on_step_end_tensor_inputs=["latents"],          
        ).images

        # save generated images -----------
        for img_id, img in enumerate(images):
            img.save(os.path.join(
                output_dir,
                f"{img_basename}_p{p_idx}_{img_id}.jpg"
            ))

        
        # save the mask-evolution strip (then clear for next run) --------------
        strip_path = os.path.join(
            output_dir, f"{img_basename}_p{p_idx}_mask_evolution.jpg"
        )
        save_strip(frames_holder, strip_path)
        frames_holder.clear()
