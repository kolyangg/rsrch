# +++ inference_scripts/inference_pulid_seed_NS2.py

import os, json, argparse, torch
import sys
from pathlib import Path

# --- TorchVision compat shim for basicsr ---
# basicsr imports torchvision.transforms.functional_tensor.rgb_to_grayscale
# which was removed in newer torchvision. Provide a tiny shim before imports.
try:
    import types
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except Exception:
        from torchvision.transforms import functional as _F_tv
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        _shim.rgb_to_grayscale = _F_tv.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _shim
except Exception:
    pass

# Allow running this file directly (python3 inference/inference_pulid_seed_NS2.py)
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from pathlib import Path
from PIL import Image
# from ..pulid.pipeline_v1_1_NS2 import PuLIDPipelineNS2

# Allow running this file directly (python3 inference/inference_pulid_seed_NS2.py)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pulid.pipeline_v1_1_NS2 import PuLIDPipelineNS2

def load_numpy(path): return np.array(Image.open(path).convert("RGB"))

ap = argparse.ArgumentParser()
# ── keep PM CLI parity so id_grid configs still work ─────────────────────────
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--image_folder", required=True)
ap.add_argument("--prompt_file",  required=True)
ap.add_argument("--class_file",   required=True)
ap.add_argument("--num_images_per_prompt", type=int, default=1)
ap.add_argument("--output_dir",   default="./outputs")
ap.add_argument("--use_branched_attention", action="store_true", default=False)
ap.add_argument("--save_heatmaps", action="store_true", default=False)
ap.add_argument("--photomaker_start_step", type=int, default=10)
ap.add_argument("--merge_start_step",      type=int, default=10)
ap.add_argument("--branched_attn_start_step", type=int, default=15)
ap.add_argument("--branched_start_mode", choices=["both","branched"], default="both")
ap.add_argument("--face_embed_strategy", choices=["face","id_embeds"], default="face")
# ap.add_argument("--auto_mask_ref", action="store_true", default=True)
ap.add_argument("--auto_mask_ref", action="store_true", default=False)
ap.add_argument("--use_dynamic_mask", type=int, choices=[0,1], default=0)
ap.add_argument("--pose_adapt_ratio", type=float, default=0.25)
ap.add_argument("--ca_mixing_for_face", type=int, choices=[0,1], default=1)
ap.add_argument("--force_par_before_pm", type=int, choices=[0,1], default=0)
# ap.add_argument("--import_mask", type=str, default=None)
# ap.add_argument("--import_mask_ref", type=str, default=None)
ap.add_argument("--import_mask", type=str, default="../compare/testing/ref3_masks/eddie_p0_2_pulid_gen_mask_new_easy.png")
ap.add_argument("--import_mask_ref", type=str, default="../compare/testing/ref3_masks/eddie_mask_new.png")
args = ap.parse_args()

torch.manual_seed(int(args.seed))
out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

pipe = PuLIDPipelineNS2()

prompts = [l.strip() for l in Path(args.prompt_file).read_text(encoding="utf-8").splitlines() if l.strip()]
ids = sorted([p for p in Path(args.image_folder).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])

for id_path in ids:
    id_np = load_numpy(id_path)
    sub = out_root / id_path.stem
    sub.mkdir(parents=True, exist_ok=True)

    # expose reference image to inner pipeline for ref-latents/masks
    pipe.pipe._ref_img = torch.from_numpy(id_np).permute(2,0,1)[None].float().to("cuda")/255*2-1

    pipe.enable_branched_attention(
        use_branched_attention=args.use_branched_attention,
        face_embed_strategy=args.face_embed_strategy,
        import_mask=args.import_mask,
        import_mask_ref=args.import_mask_ref,
        auto_mask_ref=args.auto_mask_ref,
        use_dynamic_mask=bool(args.use_dynamic_mask),
        pose_adapt_ratio=args.pose_adapt_ratio,
        ca_mixing_for_face=bool(args.ca_mixing_for_face),
        save_heatmaps_dynamic=args.save_heatmaps,
    )

    for pi, pr in enumerate(prompts):
        img = pipe.generate_branched(
            prompt=pr, id_image_np=id_np, steps=30, guidance_scale=5.0,
            seed=args.seed,
            photomaker_start_step=args.photomaker_start_step,
            merge_start_step=args.merge_start_step,
            branched_attn_start_step=args.branched_attn_start_step,
            branched_start_mode=args.branched_start_mode,
            face_embed_strategy=args.face_embed_strategy,
        )
        img.save(sub / f"{id_path.stem}_p0_{pi}.jpg")
