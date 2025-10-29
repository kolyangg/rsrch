# inferece_scripts/inference_new.py

# Minimal-diff copy of inference_pmv2_seed_NS4_upd2.py with:
# - Optional YAML config loader (dataset/pipeline/validation_args/output_dir)
# - Fixed bbox-mask support via JSON files (ref/gen) with flags

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces  # kept import for compatibility; not used now

from pm_debug import make_mask_callback, save_strip, make_image_callback


def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyYAML is required for --config. Install with `pip install pyyaml`. ({e})")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _resolve_path(base_dir: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _bbox_from_map(entry: Any) -> Optional[list]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("face_crop_new") or entry.get("face_crop_old")
    return None


def _make_rect_mask(path: str, w: int, h: int, bbox: list):
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    img.save(path)


def _make_rect_mask_like_image(path: str, image_path: str, bbox: list):
    # Create an L mask sized exactly like the source image
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with Image.open(image_path) as im:
        w, h = im.size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    img.save(path)


def main():
    # Face detector no longer used for ordering or id_embeds; keep init cheap/no-op
    try:
        face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
        face_detector.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        face_detector = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--config-name", "--config_name", dest="config_name", type=str, default=None,
                        help="Hydra-like config name without extension, e.g. config/inference/photomaker_branched_infer")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_folder", type=str, required=False)
    parser.add_argument("--prompt_file", type=str, required=False)
    parser.add_argument("--class_file", type=str, required=False)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # Branched-attention options
    parser.add_argument("--use_branched_attention", action="store_true", default=False)
    parser.add_argument("--no_branched_attention", dest="use_branched_attention", action="store_false")
    parser.add_argument("--save_heatmaps", action="store_true", default=False)

    parser.add_argument("--start_merge_step", type=int, default=10)
    parser.add_argument("--photomaker_start_step", type=int, default=None)
    parser.add_argument("--merge_start_step", type=int, default=None)
    parser.add_argument("--branched_attn_start_step", "--branched_start_step", dest="branched_attn_start_step", type=int, default=10)
    parser.add_argument("--branched_start_mode", choices=["both", "branched"], default="both")
    parser.add_argument("--face_embed_strategy", choices=["face", "id_embeds"], default="face")

    # Existing mask options
    parser.add_argument("--import_mask_folder", type=str, default="../compare/testing/gen_masks")
    parser.add_argument("--use_mask_folder", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_dynamic_mask", type=int, choices=[0, 1], default=0)
    parser.add_argument("--auto_mask_ref", action="store_true")

    # Runtime-tunable switches
    parser.add_argument("--pose_adapt_ratio", type=float, default=0.25)
    parser.add_argument("--ca_mixing_for_face", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_id_embeds", type=int, choices=[0, 1], default=1)
    parser.add_argument("--force_par_before_pm", type=int, choices=[0, 1], default=0)

    # NEW: bbox-mask controls + resolution for gen mask
    parser.add_argument("--bbox_mask_ref", type=str, default=None, help="JSON with per-ref face bbox map")
    parser.add_argument("--bbox_mask_gen", type=str, default=None, help="JSON with per-sample face bbox list keyed by '00.png', ...")
    parser.add_argument("--use_bbox_mask_ref", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_bbox_mask_gen", type=int, choices=[0, 1], default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    args = parser.parse_args()

    # Optionally load YAML config, mapping values into args
    # Accept either --config or --config-name
    cfg_path = None
    val_args: Dict[str, Any] = {}
    seeds_list = None
    ds_limit = None
    if args.config_name and not args.config:
        cfg_path = args.config_name
        if not (cfg_path.endswith(".yaml") or cfg_path.endswith(".yml")):
            cfg_path += ".yaml"
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.abspath(cfg_path)
        args.config = cfg_path

    if args.config:
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        cfg = _load_yaml_cfg(args.config)

        # dataset
        ds = cfg.get("dataset", {})
        images_dir = _resolve_path(cfg_dir, ds.get("images_dir"))
        prompts_path = _resolve_path(cfg_dir, ds.get("prompts_path"))
        classes_json_path = _resolve_path(cfg_dir, ds.get("classes_json_path"))
        bbox_mask_ref = _resolve_path(cfg_dir, ds.get("bbox_mask_ref"))
        bbox_mask_gen = _resolve_path(cfg_dir, ds.get("bbox_mask_gen"))

        if images_dir:
            args.image_folder = images_dir
        if prompts_path:
            args.prompt_file = prompts_path
        if classes_json_path:
            args.class_file = classes_json_path
        if bbox_mask_ref:
            args.bbox_mask_ref = bbox_mask_ref
        if bbox_mask_gen:
            args.bbox_mask_gen = bbox_mask_gen

        # dataset extras
        if isinstance(ds.get("seeds"), (list, tuple)):
            try:
                seeds_list = [int(s) for s in ds.get("seeds")]
            except Exception:
                seeds_list = None
        if ds.get("limit") is not None:
            try:
                ds_limit = int(ds.get("limit"))
            except Exception:
                ds_limit = None

        # pipeline flags
        pl = cfg.get("pipeline", {})
        def _pick_bool(d: Dict[str, Any], k: str, cur: Optional[bool] = None) -> Optional[bool]:
            if k in d:
                v = d.get(k)
                if isinstance(v, bool):
                    return v
            return cur

        ubm_ref = _pick_bool(pl, "use_bbox_mask_ref", bool(args.use_bbox_mask_ref))
        ubm_gen = _pick_bool(pl, "use_bbox_mask_gen", bool(args.use_bbox_mask_gen))
        if ubm_ref is not None:
            args.use_bbox_mask_ref = int(ubm_ref)
        if ubm_gen is not None:
            args.use_bbox_mask_gen = int(ubm_gen)

        if "pretrained_model_name_or_path" in pl:
            args.base_model = str(pl.get("pretrained_model_name_or_path"))
        if "pose_adapt_ratio" in pl:
            args.pose_adapt_ratio = float(pl.get("pose_adapt_ratio"))
        if "ca_mixing_for_face" in pl:
            args.ca_mixing_for_face = 1 if pl.get("ca_mixing_for_face") else 0
        if "face_embed_strategy" in pl:
            args.face_embed_strategy = str(pl.get("face_embed_strategy"))
        if "photomaker_start_step" in pl:
            args.photomaker_start_step = int(pl.get("photomaker_start_step"))
        if "merge_start_step" in pl:
            args.merge_start_step = int(pl.get("merge_start_step"))
        if "branched_attn_start_step" in pl:
            args.branched_attn_start_step = int(pl.get("branched_attn_start_step"))
        if "branched_start_mode" in pl:
            args.branched_start_mode = str(pl.get("branched_start_mode"))
        if "auto_mask_ref" in pl:
            args.auto_mask_ref = bool(pl.get("auto_mask_ref"))
        if "use_dynamic_mask" in pl:
            args.use_dynamic_mask = 1 if pl.get("use_dynamic_mask") else 0

        # validation_args
        va = cfg.get("validation_args", {})
        if isinstance(va, dict):
            val_args = dict(va)
        if "use_branched_attention" in va:
            args.use_branched_attention = bool(va.get("use_branched_attention"))
        if "num_images_per_prompt" in va:
            args.num_images_per_prompt = int(va.get("num_images_per_prompt"))
        if "photomaker_start_step" in va:
            args.photomaker_start_step = int(va.get("photomaker_start_step"))
        if "merge_start_step" in va:
            args.merge_start_step = int(va.get("merge_start_step"))
        if "branched_attn_start_step" in va:
            args.branched_attn_start_step = int(va.get("branched_attn_start_step"))
        if "branched_start_mode" in va:
            args.branched_start_mode = str(va.get("branched_start_mode"))
        if "auto_mask_ref" in va:
            args.auto_mask_ref = bool(va.get("auto_mask_ref"))
        if "use_dynamic_mask" in va:
            args.use_dynamic_mask = 1 if va.get("use_dynamic_mask") else 0
        if "height" in va:
            args.height = int(va.get("height"))
        if "width" in va:
            args.width = int(va.get("width"))

        # top-level output_dir
        if "output_dir" in cfg:
            args.output_dir = _resolve_path(cfg_dir, cfg.get("output_dir"))

    # Device + dtype
    try:
        if torch.cuda.is_available():
            device = "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        torch_dtype = torch.float16

    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
    print(f"[Seed] Using seed = {args.seed}")

    BASE_SEED = args.seed

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    fixed_mask_dir = os.path.join(output_dir, "_fixed_masks")
    os.makedirs(fixed_mask_dir, exist_ok=True)

    photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")

    # Inputs
    if not args.prompt_file or not args.image_folder or not args.class_file:
        raise ValueError("image_folder, prompt_file and class_file must be provided either via CLI or YAML --config")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [l.strip() for l in f if l.strip()]

    with open(args.class_file, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # Load bbox maps if provided
    ref_bbox_map: Dict[str, Any] = {}
    if args.bbox_mask_ref:
        with open(args.bbox_mask_ref, "r", encoding="utf-8") as fh:
            ref_bbox_map = json.load(fh)

    gen_bbox_map: Dict[str, Any] = {}
    if args.bbox_mask_gen:
        with open(args.bbox_mask_gen, "r", encoding="utf-8") as fh:
            gen_bbox_map = json.load(fh)

    # Negative prompt (allow override via validation_args)
    negative_prompt = val_args.get(
        "negative_prompt",
        "(asymmetry, worst quality, low quality, illustration, 3d, cartoon, sketch)",
    )

    # Initialize pipeline – use same base model as diffusion_template unless overridden in config
    base_model = getattr(args, "base_model", None) or "SG161222/RealVisXL_V4.0"  # "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch_dtype
    ).to("cuda")

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img",
    )

    pipe.fuse_lora()

    pipe.pose_adapt_ratio = float(args.pose_adapt_ratio)
    pipe.ca_mixing_for_face = bool(args.ca_mixing_for_face)
    pipe.use_id_embeds = bool(args.use_id_embeds)

    print(f"[InfScript] FACE_EMBED_STR={args.face_embed_strategy}  POSE_ADAPT_RATIO={pipe.pose_adapt_ratio}  CA_MIXING_FOR_FACE={pipe.ca_mixing_for_face}  USE_ID_EMBEDS={pipe.use_id_embeds}")

    # Defaults for split steps
    if args.photomaker_start_step is None:
        args.photomaker_start_step = args.start_merge_step
    if args.merge_start_step is None:
        args.merge_start_step = args.start_merge_step

    frames_mask = []
    frames_img = []
    mask_cb = make_mask_callback(pipe, mask_interval=5, container=frames_mask)
    img_cb = make_image_callback(pipe, mask_interval=5, container=frames_img)

    def dual_cb(pipeline, step_index, t, tensors):
        mask_cb(pipeline, step_index, t, tensors)
        img_cb(pipeline, step_index, t, tensors)
        return tensors

    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
    image_basename_list = [n for n in os.listdir(args.image_folder) if os.path.splitext(n)[1].lower() in SUPPORTED_SUFFIXES]
    image_path_list = sorted([os.path.join(args.image_folder, n) for n in image_basename_list])

    # Read references in sorted order; compute 512-D face embeddings (zeros if no face)
    input_id_images = [load_image(p) for p in image_path_list]
    id_basename_list = [os.path.splitext(os.path.basename(p))[0] for p in image_path_list]
    kept_ref_paths = list(image_path_list)
    id_embed_list = []
    for pth, pil_img in zip(kept_ref_paths, input_id_images):
        try:
            img_np = np.array(pil_img.convert("RGB"))[:, :, ::-1]
            faces = analyze_faces(face_detector, img_np) if face_detector is not None else []
            if faces:
                emb = torch.from_numpy(faces[0]["embedding"]).float()
            else:
                emb = torch.zeros(512, dtype=torch.float32)
        except Exception:
            emb = torch.zeros(512, dtype=torch.float32)
        id_embed_list.append(emb)

    # Build seeds and total count
    if seeds_list is None:
        seeds_list = [BASE_SEED]

    total_planned = len(input_id_images) * len(prompt_list) * len(seeds_list)
    if ds_limit is not None:
        total_planned = min(total_planned, ds_limit)

    # Prepare debug base directory under output_dir
    debug_dir_base = val_args.get("debug_dir", "hm_debug") or "hm_debug"
    debug_root = os.path.join(output_dir, debug_dir_base)
    os.makedirs(debug_root, exist_ok=True)

    # Global sample index for bbox_mask_gen mapping (00.png, 01.png, ...)
    sample_idx = 0
    produced = 0
    idx_counter = 0

    # Iterate images → prompts → seeds to match diffusion_template
    for img_idx, (ref_img, img_basename, ref_img_path) in enumerate(zip(input_id_images, id_basename_list, kept_ref_paths)):
        class_name = class_map.get(img_basename)
        for p_idx, prompt in enumerate(prompt_list):
            for seed in seeds_list:
                if ds_limit is not None and produced >= ds_limit:
                    break

                cur_prompt = prompt
                if "<class>" in prompt:
                    if class_name:
                        cur_prompt = prompt.replace("<class>", f"{class_name} img")
                    else:
                        print(f"[Warn] No class found for '{img_basename}', keeping '<class>'")

                # Generator on pipeline device to satisfy pipeline expectations
                local_gen = torch.Generator(device=device).manual_seed(int(seed))
                pipe.branched_start_mode = args.branched_start_mode

                # Clear any previous masks to avoid carry-over (Alligned with diffusion_template)
                for _attr in ('_face_mask','_face_mask_t','_face_mask_ref','_face_mask_t_ref'):
                    if hasattr(pipe, _attr):
                        setattr(pipe, _attr, None)

                # BBox-ref
                face_bbox_ref = None
                import_mask_ref = None
                if int(args.use_bbox_mask_ref) and args.bbox_mask_ref:
                    entry_r = ref_bbox_map.get(os.path.basename(ref_img_path)) or \
                              ref_bbox_map.get(f"{img_basename}.png") or \
                              ref_bbox_map.get(f"{img_basename}.jpg") or \
                              ref_bbox_map.get(f"{img_basename}.jpeg") or \
                              ref_bbox_map.get(f"{img_basename}.webp")
                    face_bbox_ref = _bbox_from_map(entry_r)
                    if face_bbox_ref is None:
                        raise RuntimeError(f"use_bbox_mask_ref=True but no bbox entry for {img_basename} in {args.bbox_mask_ref}")
                    # Build ref mask directly in (H,W) as boolean (Alligned with diffusion_template)
                    W = int(val_args.get("width", args.width)); H = int(val_args.get("height", args.height))
                    ref_mask = np.zeros((H, W), dtype=bool)
                    ow, oh = ref_img.size
                    s_ar = min(W / float(ow), H / float(oh))
                    rw = max(8, int(round(ow * s_ar)) // 8 * 8)
                    rh = max(8, int(round(oh * s_ar)) // 8 * 8)
                    pl = (W - rw) // 2; pt = (H - rh) // 2
                    x0, y0, x1, y1 = [float(v) for v in face_bbox_ref]
                    x0s = int(round(x0 * s_ar + pl)); x1s = int(round(x1 * s_ar + pl))
                    y0s = int(round(y0 * s_ar + pt)); y1s = int(round(y1 * s_ar + pt))
                    x0s = max(0, min(W, x0s)); x1s = max(0, min(W, x1s))
                    y0s = max(0, min(H, y0s)); y1s = max(0, min(H, y1s))
                    if x1s > x0s and y1s > y0s:
                        ref_mask[y0s:y1s, x0s:x1s] = True
                    # Pre-inject into pipeline (Alligned with diffusion_template)
                    pipe._face_mask_ref = ref_mask
                    pipe._face_mask_t_ref = torch.from_numpy(ref_mask.astype(np.uint8))[None, None]
                    import_mask_ref = None


                # BBox-gen or fallback
                face_bbox_gen = None
                eff_use_dynamic = bool(args.use_dynamic_mask)
                eff_import_mask = None
                if int(args.use_bbox_mask_gen) and args.bbox_mask_gen:
                    # Match by exact final filename: "{prompt[:10]}_{ref_stem}.png"
                    base_name = f"{cur_prompt[:10]}_{img_basename}"
                    key = f"{base_name}.png"
                    entry_g = gen_bbox_map.get(key)
                    if entry_g is None:
                        raise RuntimeError(
                            f"No bbox entry in bbox_mask_gen for expected output name '{key}'"
                        )
                    face_bbox_gen = _bbox_from_map(entry_g)
                    if face_bbox_gen is None:
                        raise RuntimeError(f"BBox record for '{key}' missing face_crop_new/old")
                    eff_use_dynamic = False
                    # Build gen mask directly in (H,W) as boolean (Alligned with diffusion_template)
                    w_gen = int(val_args.get("width", args.width))
                    h_gen = int(val_args.get("height", args.height))
                    gen_mask = np.zeros((h_gen, w_gen), dtype=bool)
                    x0, y0, x1, y1 = [float(v) for v in face_bbox_gen]
                    x0i = max(0, min(w_gen, int(round(x0)))); x1i = max(0, min(w_gen, int(round(x1))))
                    y0i = max(0, min(h_gen, int(round(y0)))); y1i = max(0, min(h_gen, int(round(y1))))
                    if x1i > x0i and y1i > y0i:
                        gen_mask[y0i:y1i, x0i:x1i] = True
                    pipe._face_mask = gen_mask  # Alligned with diffusion_template
                    pipe._face_mask_t = torch.from_numpy(gen_mask.astype(np.uint8))[None, None]  # Alligned with diffusion_template
                    eff_import_mask = None
                else:
                    if not eff_use_dynamic and bool(args.use_mask_folder):
                        cand = os.path.join(args.import_mask_folder, f"{img_basename}_gen_mask.png")
                        if os.path.isfile(cand):
                            eff_import_mask = cand
                        else:
                            eff_use_dynamic = True

                # Per-sample debug dir under output_dir/hm_debug/idx
                sample_debug_dir = os.path.join(debug_root, f"{idx_counter:02d}")
                os.makedirs(sample_debug_dir, exist_ok=True)

                # Collect call kwargs from validation_args
                call_kwargs = {}
                for k in ("num_inference_steps", "guidance_scale", "height", "width", "target_size", "original_size", "crops_coords_top_left"):
                    if k in val_args:
                        call_kwargs[k] = val_args[k]

                # Pick per-image 512-D id embedding
                id_embeds_vec = id_embed_list[img_idx]

                images = pipe(
                    cur_prompt,
                    negative_prompt=negative_prompt,
                    input_id_images=[ref_img],
                    id_embeds=id_embeds_vec,
                    num_images_per_prompt=args.num_images_per_prompt,
                    start_merge_step=args.start_merge_step,
                    photomaker_start_step=args.photomaker_start_step,
                    merge_start_step=args.merge_start_step,
                    generator=local_gen,
                    # branched/dynamic
                    use_branched_attention=args.use_branched_attention,
                    save_heatmaps=args.save_heatmaps,
                    branched_attn_start_step=args.branched_attn_start_step,
                    face_embed_strategy=args.face_embed_strategy,
                    auto_mask_ref=args.auto_mask_ref if not int(args.use_bbox_mask_ref) else False,
                    use_dynamic_mask=eff_use_dynamic,
                    import_mask=eff_import_mask,
                    import_mask_folder=args.import_mask_folder,
                    use_mask_folder=bool(args.use_mask_folder),
                    # supply import_mask_ref so aggregate_heatmaps_to_mask can load it
                    import_mask_ref=import_mask_ref,
                    # bbox-driven flags/values (pipeline may ignore; kept for completeness)
                    use_bbox_mask_ref=bool(int(args.use_bbox_mask_ref)),
                    use_bbox_mask_gen=bool(int(args.use_bbox_mask_gen)),
                    face_bbox_ref=face_bbox_ref,
                    face_bbox_gen=face_bbox_gen,
                    # debug routing
                    debug_dir=sample_debug_dir,
                    debug_idx=idx_counter,
                    debug_total=total_planned,
                    # callbacks
                    callback_on_step_end=dual_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                    force_par_before_pm=args.force_par_before_pm,
                    **call_kwargs,
                ).images

                # Save only final image with matching naming
                from pathlib import Path as _P
                ref_stem = _P(ref_img_path).stem
                base = f"{cur_prompt[:10]}_{ref_stem}"
                out_path = os.path.join(output_dir, f"{base}.png") if len(images) == 1 else os.path.join(output_dir, f"{base}_00.png")
                images[0].save(out_path)

                # Save evolution strips to debug directory
                save_strip(frames_mask, os.path.join(sample_debug_dir, "mask_evolution.jpg"))
                save_strip(frames_img, os.path.join(sample_debug_dir, "img_evolution.jpg"))
                frames_mask.clear()
                frames_img.clear()

                produced += 1
                sample_idx += 1
                idx_counter += 1
            if ds_limit is not None and produced >= ds_limit:
                break
        if ds_limit is not None and produced >= ds_limit:
            break


if __name__ == "__main__":
    main()
