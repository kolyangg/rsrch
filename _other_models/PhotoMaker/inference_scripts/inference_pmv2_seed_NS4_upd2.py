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
# from photomaker import PhotoMakerStableDiffusionXLPipeline2 as PhotoMakerStableDiffusionXLPipeline
from photomaker import PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces

import argparse                              # ⭐ for CLI seed
import random                                # ⭐ optional; keep runs reproducible

# ── debug helpers (new)
from pm_debug import make_mask_callback, save_strip, make_image_callback

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
parser.add_argument("--num_images_per_prompt", type=int, default=1, # changed default from 3 to 1
                    help="Images to generate for each prompt / ID")
parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Output directory for generated images")
# ───────────── NEW: branched-attention options ─────────────
parser.add_argument("--use_branched_attention", action="store_true", default=False,
                    help="Activate the new face/background branched attention")
parser.add_argument("--no_branched_attention", dest="use_branched_attention", action="store_false", help="Disable the branched attention mechanism")
parser.add_argument("--save_heatmaps", action="store_true", default=False,
                    help="Save heatmaps during inference")

parser.add_argument("--start_merge_step", type=int, default=10,
                    help="Denoising step to start PhotoMaker merge (ID begins here)")

parser.add_argument("--photomaker_start_step", type=int, default=None,
                    help="Step to switch to ID-enhanced prompts. Defaults to start_merge_step if not set.")
parser.add_argument("--merge_start_step", type=int, default=None,
                    help="Step when face/bg merging starts to affect output. Defaults to start_merge_step if not set.")

parser.add_argument("--branched_attn_start_step", "--branched_start_step",
                    dest="branched_attn_start_step", type=int, default=10,
                    help="Denoising step at which branched attention kicks in (alias: --branched_start_step)")
parser.add_argument("--branched_start_mode", choices=["both", "branched"], default="both",
    help="What to enable at branched_attn_start_step: 'both' (PM+Branched) or 'branched' (Branched only).")


parser.add_argument("--face_embed_strategy", choices=["face", "id_embeds"],
                    default="face", # default="faceanalysis",
                    help="Face branch conditioning: 'face' text or PhotoMaker ID–enhanced")

# ───────── mask source selection ─────────
parser.add_argument("--import_mask_folder", type=str, default="../compare/testing/gen_masks",
                    help="Folder with per-reference gen masks named '<ref_basename>_gen_mask.png'")
parser.add_argument("--use_mask_folder", type=int, choices=[0,1], default=1,
                    help="If 1, load masks from --import_mask_folder (ignored when --use_dynamic_mask=1)")
parser.add_argument("--use_dynamic_mask", type=int, choices=[0,1], default=0,
                    help="If 1, generate mask dynamically during denoising (overrides imported masks)")


parser.add_argument("--auto_mask_ref", action="store_true",
                    help="Auto-compute reference face mask via create_ref_mask.py")

# ───────── runtime-tunable branched-attn knobs from attn_processor.py ─────────
parser.add_argument("--pose_adapt_ratio", type=float, default=0.25,
                    help="Blend of reference-vs-noise for face pose adaptation (0=stick to ref, 1=follow noise)")
parser.add_argument("--ca_mixing_for_face", type=int, choices=[0,1], default=1,
                    help="Whether to concatenate blended face/noise features for K/V in face branch (1 on, 0 off)")
parser.add_argument("--use_id_embeds", type=int, choices=[0,1], default=1,
                    help="Whether to inject PhotoMaker ID features into the face branch (1 on, 0 off)")
parser.add_argument("--force_par_before_pm", type=int, choices=[0,1], default=0,
                    help="Whether to force pose adaptation ratio before PhotoMaker (1 on, 0 off)")


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

# generator = torch.Generator(device).manual_seed(args.seed)  # ⭐

# we will re-seed for *every* run so the first RNG draw is identical
BASE_SEED = args.seed    # keep a copy




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

# negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

# ------------------------------------------------------------------
# 2) Keep the negative prompt identical to attn_hm_NS_nosm7.py
# ------------------------------------------------------------------
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, cartoon, sketch)"

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

# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()


# ------------------------------------------------------------------
# 1) Use the *same* numerical path as the heat-map script
# ------------------------------------------------------------------
pipe.disable_xformers_memory_efficient_attention()          # <- NEW
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# (optional) comment out cpu-offload – avoids extra fp32/16 round-trips
# pipe.enable_model_cpu_offload()

# Make the knobs available to branched attention processors
pipe.pose_adapt_ratio   = float(args.pose_adapt_ratio)
pipe.ca_mixing_for_face = bool(args.ca_mixing_for_face)
pipe.use_id_embeds      = bool(args.use_id_embeds)

print(f'[InfScript] FACE_EMBED_STR={args.face_embed_strategy}  POSE_ADAPT_RATIO={pipe.pose_adapt_ratio}  CA_MIXING_FOR_FACE={pipe.ca_mixing_for_face}  USE_ID_EMBEDS={pipe.use_id_embeds}')


# # ── create one reusable callback & frame-store ───────────────────────────
# frames_holder = []
# mask_cb       = make_mask_callback(pipe, mask_interval=5, container=frames_holder)


# ── create two parallel callbacks & frame-stores ─────────────────────────
frames_mask = []   # with red overlay
frames_img  = []   # clean images

mask_cb = make_mask_callback (pipe, mask_interval=5, container=frames_mask)
img_cb  = make_image_callback(pipe, mask_interval=5, container=frames_img)

# tiny helper to drive both collectors with one handle
def dual_cb(pipeline, step_index, t, tensors):
    mask_cb(pipeline, step_index, t, tensors)
    img_cb (pipeline, step_index, t, tensors)
    
    return tensors



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
        
        # ------------------------------------------------------------------
        # 4) fresh generator per call -> identical latent noise
        # ------------------------------------------------------------------
        local_gen = torch.Generator(device).manual_seed(BASE_SEED)

        # configure pipeline-level mode selector for the scheduler
        pipe.branched_start_mode = args.branched_start_mode

        # ───────── resolve mask source per reference ─────────
        eff_use_dynamic = bool(args.use_dynamic_mask)
        eff_import_mask = None
        if not eff_use_dynamic and bool(args.use_mask_folder):
            cand = os.path.join(args.import_mask_folder, f"{img_basename}_gen_mask.png")
            if os.path.isfile(cand):
                eff_import_mask = cand
                print(f"[MASK] Using imported fiexed gen mask for '{img_basename}' at '{cand}'.")
            else:
                print(f"[WARNING] No gen mask for '{img_basename}' at '{cand}'. "
                      f"Falling back to dynamic mask for this reference.", flush=True)
                eff_use_dynamic = True

        images = pipe(
            cur_prompt,
            negative_prompt=negative_prompt,
            input_id_images=[ref_img],
            id_embeds=id_embeds,
            num_images_per_prompt=args.num_images_per_prompt,
            start_merge_step=args.start_merge_step,
            photomaker_start_step=args.photomaker_start_step,
            merge_start_step=args.merge_start_step,
            # generator=generator,
            generator=local_gen,
            # ─── forward the new flags ────────────────────────────────
            use_branched_attention=args.use_branched_attention,
            save_heatmaps=args.save_heatmaps,  # ← NEW
            branched_attn_start_step=args.branched_attn_start_step,
            face_embed_strategy=args.face_embed_strategy,  
            auto_mask_ref=args.auto_mask_ref,
            # debug_save_masks=True,          # ← NEW
            # mask_interval=5,                # ← NEW (optional)
            # debugging (handled by callback)
            # callback_on_step_end            = mask_cb,
            # mask selection
            import_mask=eff_import_mask,
            use_dynamic_mask=eff_use_dynamic,
            import_mask_folder=args.import_mask_folder,
            use_mask_folder=bool(args.use_mask_folder),
            # debugging (handled by *both* callbacks)
            callback_on_step_end            = dual_cb,
            callback_on_step_end_tensor_inputs=["latents"],  
            force_par_before_pm = args.force_par_before_pm        
        ).images

        # save generated images -----------
        for img_id, img in enumerate(images):
            img.save(os.path.join(
                output_dir,
                f"{img_basename}_p{p_idx}_{img_id}.jpg"
            ))

        
        # # save the mask-evolution strip (then clear for next run) --------------
        # strip_path = os.path.join(
        #     output_dir, f"{img_basename}_p{p_idx}_mask_evolution.jpg"
        # )
        # save_strip(frames_holder, strip_path)
        # frames_holder.clear()
        
        # save both evolution strips, then clear buffers -----------------------
        save_strip(
            frames_mask,
            os.path.join(output_dir, f"{img_basename}_p{p_idx}_mask_evolution.jpg"),
        )
        save_strip(
            frames_img,
            os.path.join(output_dir, f"{img_basename}_p{p_idx}_img_evolution.jpg"),
        )
        frames_mask.clear()
        frames_img .clear()
