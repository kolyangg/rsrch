#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qk_corr_regions_fixed.py
------------------------
Fixed version that properly installs hooks and matches id_grid.sh generation.
"""

import os, json, math, argparse, random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from huggingface_hub import hf_hub_download
from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image

# Project imports
from photomaker.pipeline_br import PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces

# Global collector for hook data
GLOBAL_COLLECTOR = None
CURRENT_STEP = 0

class CorrCollector:
    """Collects correlation data from attention layers."""
    
    def __init__(self, q_masks_hw: Dict[str, np.ndarray], k_masks_hw: Dict[str, np.ndarray],
                 step_min: int = 30, latest_layers: int = 4):
        self.q_masks_hw = q_masks_hw
        self.k_masks_hw = k_masks_hw
        self.step_min = step_min
        self.latest_layers = latest_layers
        
        self.layer_order = []
        self._layer_seen = set()
        self.maps = {}  # {layer: {region: {'sum': np.array, 'cnt': int}}}
    
    def register_layer(self, layer_name: str):
        if layer_name not in self._layer_seen:
            self.layer_order.append(layer_name)
            self._layer_seen.add(layer_name)
    
    def resize_mask_to_tokens(self, mask_hw: np.ndarray, L: int) -> np.ndarray:
        """Resize HxW mask to L tokens."""
        H = int(math.sqrt(L))
        W = H
        mask_pil = Image.fromarray((mask_hw * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((W, H), Image.NEAREST)
        return (np.array(mask_resized) >= 128).astype(np.uint8).reshape(-1)
    
    def add_call(self, layer: str, q_face: torch.Tensor, k_face: torch.Tensor,
                 step_idx: int, q_region_name: str):
        """Add correlation data for a layer/region."""
        print(f"[DEBUG-COLLECT] Layer: {layer}, Region: {q_region_name}, Step: {step_idx}, Shape: {q_face.shape}")

        if step_idx < self.step_min:
            return
        
        B, H, Lq, D = q_face.shape
        _, _, Lk, _ = k_face.shape
        
        # Get Q region mask
        q_mask = self.resize_mask_to_tokens(self.q_masks_hw[q_region_name], Lq)
        if q_mask.sum() == 0:
            return
        
        # Normalize
        q_norm = F.normalize(q_face, dim=-1)
        k_norm = F.normalize(k_face, dim=-1)
        
        # Apply mask and average
        q_mask_t = torch.from_numpy(q_mask.astype(np.float32)).to(q_face.device)
        q_mask_exp = q_mask_t.view(1, 1, Lq, 1)
        q_region = (q_norm * q_mask_exp).sum(dim=2) / q_mask_exp.sum(dim=2).clamp(min=1e-6)
        
        # Compute similarity
        sim = torch.matmul(q_region.unsqueeze(2), k_norm.transpose(-2, -1)).squeeze(2)
        sim_vec = sim.mean(dim=(0, 1)).detach().cpu().numpy()
        
        # Accumulate
        bucket = self.maps.setdefault(layer, {}).setdefault(q_region_name, {"sum": None, "cnt": 0})
        if bucket["sum"] is None:
            bucket["sum"] = sim_vec.astype(np.float64)
        else:
            bucket["sum"] += sim_vec
        bucket["cnt"] += 1
    
    def build_report(self) -> Dict:
        """Build analysis report."""
        results = {
            'meta': {
                'step_min': self.step_min,
                'latest_layers': self.latest_layers,
                'layer_order': self.layer_order,
            },
            'per_layer': {},
            'aggregate_all_layers': {},
            'aggregate_lastN_layers': {}
        }
        
        # Per-layer analysis
        for layer in self.layer_order:
            if layer not in self.maps:
                continue
            layer_results = {}
            for region in ['mouth', 'eyes', 'nose']:
                if region not in self.maps[layer]:
                    continue
                
                bucket = self.maps[layer][region]
                avg_map = bucket["sum"] / max(1, bucket["cnt"])
                
                # Compute area means
                k_area_means = {}
                for area in ['mouth', 'eyes', 'nose']:
                    k_mask = self.resize_mask_to_tokens(self.k_masks_hw[area], len(avg_map))
                    if k_mask.sum() > 0:
                        k_area_means[area] = float(avg_map[k_mask.astype(bool)].mean())
                    else:
                        k_area_means[area] = float('nan')
                
                best_match = max(k_area_means.items(), 
                               key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf)[0]
                
                layer_results[region] = {
                    'k_area_means': k_area_means,
                    'verdict_same_area': (best_match == region),
                    'best_match': best_match
                }
            
            results['per_layer'][layer] = layer_results
        
        # Aggregate all layers
        results['aggregate_all_layers'] = self._aggregate_layers(self.layer_order)
        
        # Aggregate last N layers
        last_n = self.layer_order[-self.latest_layers:] if len(self.layer_order) > self.latest_layers else self.layer_order
        results['aggregate_lastN_layers'] = self._aggregate_layers(last_n)
        
        return results
    
    def _aggregate_layers(self, layers: List[str]) -> Dict:
        """Aggregate across layers."""
        aggregated = {}
        
        for region in ['mouth', 'eyes', 'nose']:
            # Collect maps
            all_maps = []
            for layer in layers:
                if layer in self.maps and region in self.maps[layer]:
                    bucket = self.maps[layer][region]
                    avg_map = bucket["sum"] / max(1, bucket["cnt"])
                    all_maps.append(avg_map)
            
            if all_maps:
                # Average
                combined = np.stack(all_maps).mean(axis=0)
                
                # Compute area means
                k_area_means = {}
                for area in ['mouth', 'eyes', 'nose']:
                    k_mask = self.resize_mask_to_tokens(self.k_masks_hw[area], len(combined))
                    if k_mask.sum() > 0:
                        k_area_means[area] = float(combined[k_mask.astype(bool)].mean())
                    else:
                        k_area_means[area] = float('nan')
                
                best_match = max(k_area_means.items(),
                               key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf)[0]
                
                aggregated[region] = {
                    'k_area_means': k_area_means,
                    'verdict_same_area': (best_match == region),
                    'best_match': best_match,
                    'correlation_map': combined
                }
            else:
                aggregated[region] = {
                    'k_area_means': {a: float('nan') for a in ['mouth', 'eyes', 'nose']},
                    'verdict_same_area': False,
                    'best_match': 'none'
                }
        
        return aggregated
    
    def save_heatmap_pdf(self, pdf_path: Path):
        """Save heatmap PDF."""
        report = self.build_report()
        
        with PdfPages(str(pdf_path)) as pdf:
            # Page 1: All layers
            fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig1.suptitle('All Layers Aggregated', fontsize=16)
            
            for idx, region in enumerate(['mouth', 'eyes', 'nose']):
                # Top: Q mask
                ax_q = axes[0, idx]
                ax_q.imshow(self.q_masks_hw[region], cmap='gray')
                ax_q.set_title(f'Q {region} mask')
                ax_q.axis('off')
                
                # Bottom: Heatmap
                ax_h = axes[1, idx]
                if region in report['aggregate_all_layers'] and 'correlation_map' in report['aggregate_all_layers'][region]:
                    data = report['aggregate_all_layers'][region]
                    corr_map = data['correlation_map']
                    H = int(math.sqrt(len(corr_map)))
                    corr_2d = corr_map.reshape(H, H)
                    
                    im = ax_h.imshow(corr_2d, cmap='hot', interpolation='nearest')
                    
                    verdict = "✓" if data['verdict_same_area'] else "✗"
                    ax_h.set_title(f'{region} → {data["best_match"]} {verdict}')
                    plt.colorbar(im, ax=ax_h, fraction=0.046)
                else:
                    ax_h.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax_h.set_title(f'{region} (no data)')
                ax_h.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close()
            
            # Page 2: Last N layers
            fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig2.suptitle(f'Last {self.latest_layers} Layers', fontsize=16)
            
            for idx, region in enumerate(['mouth', 'eyes', 'nose']):
                # Top: K mask
                ax_k = axes[0, idx]
                ax_k.imshow(self.k_masks_hw[region], cmap='gray')
                ax_k.set_title(f'K {region} mask')
                ax_k.axis('off')
                
                # Bottom: Heatmap
                ax_h = axes[1, idx]
                if region in report['aggregate_lastN_layers'] and 'correlation_map' in report['aggregate_lastN_layers'][region]:
                    data = report['aggregate_lastN_layers'][region]
                    corr_map = data['correlation_map']
                    H = int(math.sqrt(len(corr_map)))
                    corr_2d = corr_map.reshape(H, H)
                    
                    im = ax_h.imshow(corr_2d, cmap='hot', interpolation='nearest')
                    
                    verdict = "✓" if data['verdict_same_area'] else "✗"
                    ax_h.set_title(f'{region} → {data["best_match"]} {verdict}')
                    plt.colorbar(im, ax=ax_h, fraction=0.046)
                else:
                    ax_h.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax_h.set_title(f'{region} (no data)')
                ax_h.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close()

def make_capture_wrapper(layer_name: str, orig_call):
    """Wrap BranchedAttnProcessor to capture attention data."""
    
    def wrapped(self, attn, hidden_states, encoder_hidden_states=None,
                attention_mask=None, temb=None, scale=1.0):
        # Check if we still have our hook
        if not hasattr(self, '_orig_call_corr'):
            print(f"[DEBUG-HOOK] WARNING: Hook lost on {layer_name} at step {CURRENT_STEP}")
            return orig_call(self, attn, hidden_states, encoder_hidden_states,
                       attention_mask, temb, scale)
            
        # Run original
        output = orig_call(self, attn, hidden_states, encoder_hidden_states,
                          attention_mask, temb, scale)
        
        # Debug: Check if we're capturing
        print(f"[DEBUG-HOOK] Layer: {layer_name}, Step: {CURRENT_STEP}, Has mask: {hasattr(self, 'mask') and self.mask is not None}")
    
        
        # Capture if we have masks and collector
        global GLOBAL_COLLECTOR
        if GLOBAL_COLLECTOR is None or CURRENT_STEP < GLOBAL_COLLECTOR.step_min:
            if CURRENT_STEP == 30:  # First capture step
               print(f"[DEBUG-HOOK] Skipping - Collector: {GLOBAL_COLLECTOR is not None}, Step: {CURRENT_STEP}")
            return output
        
        if hasattr(self, 'mask') and self.mask is not None:
            try:
                # Handle 4D input
                if hidden_states.ndim == 4:
                    B, C, H, W = hidden_states.shape
                    hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
                
                # Split batch
                total_batch = hidden_states.shape[0]
                half_batch = total_batch // 2
                noise_hidden = hidden_states[:half_batch]
                ref_hidden = hidden_states[half_batch:]
                
                # Apply group norm if needed
                if attn.group_norm is not None:
                    noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
                    ref_hidden = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
                
                batch_size = noise_hidden.shape[0]
                seq_len = noise_hidden.shape[1]
                head_dim = attn.heads
                dim_per_head = noise_hidden.shape[-1] // head_dim
                
                # Get Q and K
                q = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
                k_face = attn.to_k(ref_hidden).view(batch_size, -1, head_dim, dim_per_head).transpose(1, 2)
                
                # Apply mask to Q
                mask_gate = None
                if hasattr(self, '_prepare_mask'):
                    mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)
                    mask_gate = mask_gate.to(dtype=q.dtype, device=q.device)
                
                q_face = q * mask_gate if mask_gate is not None else q
                
                # Capture for each region
                GLOBAL_COLLECTOR.register_layer(layer_name)
                for region in ['mouth', 'eyes', 'nose']:
                    GLOBAL_COLLECTOR.add_call(layer_name, q_face, k_face, CURRENT_STEP, region)
                    
            except Exception as e:
                pass  # Silent fail to not break generation
        
        return output
    
    return wrapped

def install_hooks_on_processors(pipeline):
    """Install hooks on existing BranchedAttnProcessor instances."""
    from photomaker.attn_processor import BranchedAttnProcessor
    
    count = 0
    total_procs = 0
    for name, proc in pipeline.unet.attn_processors.items():
        total_procs += 1
        print(f"[DEBUG-INSTALL] Processor {name}: {type(proc).__name__}")
        if isinstance(proc, BranchedAttnProcessor):
            if not hasattr(proc, '_orig_call_corr'):
                proc._orig_call_corr = proc.__call__
                proc.__call__ = make_capture_wrapper(name, proc._orig_call_corr).__get__(proc, proc.__class__)
                count += 1
                print(f"[DEBUG-INSTALL] ✓ Hooked: {name}")
    
    print(f"[Hooks] Installed on {count}/{total_procs} processors (BranchedAttnProcessor instances)")

    return count

def main():
    parser = argparse.ArgumentParser()
    
    # Masks
    # parser.add_argument("--q_mouth", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_mouth.png")
    # parser.add_argument("--q_eyes", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_eyes.png")
    # parser.add_argument("--q_nose", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_nose.png")
    # parser.add_argument("--k_mouth", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_mouth.png")
    # parser.add_argument("--k_eyes", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_eyes.png")
    # parser.add_argument("--k_nose", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_nose.png")

    parser.add_argument("--q_mouth", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_mouth.png")
    parser.add_argument("--q_eyes", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_eyes.png")
    parser.add_argument("--q_nose", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_nose.png")
    parser.add_argument("--k_mouth", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_mouth.png")
    parser.add_argument("--k_eyes", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_eyes.png")
    parser.add_argument("--k_nose", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_nose.png")
    
    # Standard inputs
    # parser.add_argument("--image_folder", default="../compare/testing/ref3") # eddie
    parser.add_argument("--image_folder", default="../compare/testing/ref2") # keanu
    parser.add_argument("--prompt_file", default="../compare/testing/prompt_one2.txt")
    parser.add_argument("--class_file", default="../compare/testing/classes_ref.json")
    # parser.add_argument("--output_dir", default="../compare/results/qk_corr_run1")
    parser.add_argument("--output_dir", default="../compare/results/qk_corr_keanu")
    
    # Analysis
    parser.add_argument("--step_min", type=int, default=30)
    parser.add_argument("--latest_layers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Branched params (matching id_grid.sh defaults)
    parser.add_argument("--photomaker_start_step", type=int, default=10)
    parser.add_argument("--merge_start_step", type=int, default=10)
    parser.add_argument("--branched_attn_start_step", type=int, default=15)  # Changed to 15 to match
    parser.add_argument("--branched_start_mode", default="both")
    parser.add_argument("--use_branched_attention", action="store_true", default=True)
    parser.add_argument("--auto_mask_ref", action="store_true", default=True)
    parser.add_argument("--pose_adapt_ratio", type=float, default=0.25)
    parser.add_argument("--ca_mixing_for_face", type=int, default=1)  # Changed to 1 to match
    parser.add_argument("--use_id_embeds", type=int, default=0)
    parser.add_argument("--face_embed_strategy", default="face")
    # parser.add_argument("--import_mask", default="../compare/testing/ref3_masks/eddie_pm_mask_new.jpg")
    parser.add_argument("--import_mask", default="../compare/testing/ref2_masks/keanu_gen_mask.png")
    parser.add_argument("--import_mask_ref", default=None)
    parser.add_argument("--use_dynamic_mask", type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load masks
    def load_mask(path):
        img = Image.open(path).convert('L')
        return (np.array(img) >= 128).astype(np.uint8)
    
    q_masks = {
        'mouth': load_mask(args.q_mouth),
        'eyes': load_mask(args.q_eyes),
        'nose': load_mask(args.q_nose)
    }
    print(f"[DEBUG-MASKS] Q masks loaded - mouth: {q_masks['mouth'].shape}, sum: {q_masks['mouth'].sum()}")
    print(f"[DEBUG-MASKS] Q masks loaded - eyes: {q_masks['eyes'].shape}, sum: {q_masks['eyes'].sum()}")
    print(f"[DEBUG-MASKS] Q masks loaded - nose: {q_masks['nose'].shape}, sum: {q_masks['nose'].sum()}")

    
    k_masks = {
        'mouth': load_mask(args.k_mouth),
        'eyes': load_mask(args.k_eyes),
        'nose': load_mask(args.k_nose)
    }
    
    # Initialize collector
    global GLOBAL_COLLECTOR
    GLOBAL_COLLECTOR = CorrCollector(q_masks, k_masks, args.step_min, args.latest_layers)
    
    # Setup pipeline
    print("Loading pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", torch_dtype=torch_dtype
    ).to(device)
    
    # Load PhotoMaker
    pm_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2",
                              filename="photomaker-v2.bin", repo_type="model")
    pipe.load_photomaker_adapter(
        os.path.dirname(pm_ckpt),
        subfolder="",
        weight_name=os.path.basename(pm_ckpt),
        trigger_word="img"
    )
    
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.disable_xformers_memory_efficient_attention()
    
    # Set runtime params
    pipe.pose_adapt_ratio = float(args.pose_adapt_ratio)
    pipe.ca_mixing_for_face = bool(args.ca_mixing_for_face)
    pipe.use_id_embeds = bool(args.use_id_embeds)
    
    # Load reference
    image_paths = sorted(Path(args.image_folder).glob("*.jpg")) + sorted(Path(args.image_folder).glob("*.png")) + sorted(Path(args.image_folder).glob("*.webp"))
    if not image_paths:
        raise ValueError(f"No images in {args.image_folder}")
    
    ref_image = load_image(str(image_paths[0]))
    
    # Get face embeddings
    face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'],
                                  allowed_modules=['detection', 'recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    
    img_np = np.array(ref_image)[:, :, ::-1]
    faces = analyze_faces(face_detector, img_np)
    
    if faces:
        id_embeds = torch.from_numpy(faces[0]["embedding"]).unsqueeze(0)
    else:
        id_embeds = torch.zeros(1, 512)
    
    # Load prompt
    with open(args.prompt_file) as f:
        prompt = f.read().strip()
    
    with open(args.class_file) as f:
        classes = json.load(f)
    
    ref_name = image_paths[0].stem
    class_name = classes.get(ref_name, "person")
    
    if "<class>" in prompt:
        prompt = prompt.replace("<class>", f"{class_name} img")
    elif "img" not in prompt:
        prompt = prompt + " img"
    
    print(f"Prompt: {prompt}")
    
    # Step callback
    def step_cb(pipe, step, t, tensors):
        global CURRENT_STEP
        CURRENT_STEP = step
        print(f"[DEBUG-STEP] Step {step}, Branched processors exist: {any('Branched' in str(type(p)) for p in pipe.unet.attn_processors.values())}")

            
        # Install hooks when BranchedAttnProcessor first appears
        if step == args.branched_attn_start_step and GLOBAL_COLLECTOR is not None:
             if not hasattr(pipe, '_hooks_installed'):
                 hook_count = install_hooks_on_processors(pipe)
                 print(f"[DEBUG-STEP] Installed {hook_count} hooks at step {step}")
                 pipe._hooks_installed = True
            
        if step == args.branched_attn_start_step:
            print(f"[DEBUG-STEP] Branched start step {step} - re-checking processors")
            for i, (name, proc) in enumerate(pipe.unet.attn_processors.items()):
                if i >= 3:  # Only check first 3
                   break
                print(f"  {name}: {type(proc).__name__}")
        
        # Double-check hooks are still there at capture start
        if step == args.step_min and GLOBAL_COLLECTOR is not None:
            branched_count = sum(1 for p in pipe.unet.attn_processors.values() 
                                if 'Branched' in type(p).__name__)
            if branched_count > 0 and not hasattr(pipe, '_hooks_installed'):
                hook_count = install_hooks_on_processors(pipe)
                print(f"[DEBUG-STEP] Re-installed {hook_count} hooks at capture step {step}")
                pipe._hooks_installed = True
        
        return tensors
    
    
    print(f"[DEBUG-PARAMS] branched_attn_start_step: {args.branched_attn_start_step}")
    print(f"[DEBUG-PARAMS] use_branched_attention: {args.use_branched_attention}")
    print(f"[DEBUG-PARAMS] step_min for capture: {args.step_min}")
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    images = pipe(
        prompt=prompt,
        input_id_images=[ref_image],
        id_embeds=id_embeds,
        num_images_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        use_branched_attention=args.use_branched_attention,
        photomaker_start_step=args.photomaker_start_step,
        merge_start_step=args.merge_start_step,
        branched_attn_start_step=args.branched_attn_start_step,
        branched_start_mode=args.branched_start_mode,
        face_embed_strategy=args.face_embed_strategy,
        auto_mask_ref=args.auto_mask_ref,
        import_mask=args.import_mask if not args.use_dynamic_mask else None,
        import_mask_ref=args.import_mask_ref,
        use_dynamic_mask=bool(args.use_dynamic_mask),
        callback_on_step_end=step_cb,
        callback_on_step_end_tensor_inputs=["latents"]
    ).images
    
    # Save image
    images[0].save(out_dir / "generated.png")
    print(f"Saved image to {out_dir / 'generated.png'}")
    
    print(f"[DEBUG-FINAL] Total layers registered: {len(GLOBAL_COLLECTOR.layer_order)}")
    print(f"[DEBUG-FINAL] Layers with data: {list(GLOBAL_COLLECTOR.maps.keys())}")
    for layer in list(GLOBAL_COLLECTOR.maps.keys())[:2]:  # First 2 layers
       for region in GLOBAL_COLLECTOR.maps[layer]:
           bucket = GLOBAL_COLLECTOR.maps[layer][region]
           print(f"  {layer}/{region}: {bucket['cnt']} captures")
    
    # Generate report
    report = GLOBAL_COLLECTOR.build_report()
    
    # Save JSON
    summary = {
        'configuration': {
            'step_min': args.step_min,
            'latest_layers': args.latest_layers,
            'num_layers_analyzed': len([l for l in GLOBAL_COLLECTOR.layer_order if l in GLOBAL_COLLECTOR.maps])
        },
        'verdicts': {
            'all_layers': {},
            'late_layers': {}
        },
        'detailed_results': report
    }
    
    for region in ['mouth', 'eyes', 'nose']:
        if region in report['aggregate_all_layers']:
            data = report['aggregate_all_layers'][region]
            summary['verdicts']['all_layers'][region] = {
                'correct': data['verdict_same_area'],
                'best_match': data['best_match'],
                'correlation_values': data['k_area_means']
            }
        
        if region in report['aggregate_lastN_layers']:
            data = report['aggregate_lastN_layers'][region]
            summary['verdicts']['late_layers'][region] = {
                'correct': data['verdict_same_area'],
                'best_match': data['best_match'],
                'correlation_values': data['k_area_means']
            }
    
    json_path = out_dir / "qk_corr_report.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved report to {json_path}")
    
    # Save PDF
    pdf_path = out_dir / "qk_corr_heatmaps.pdf"
    GLOBAL_COLLECTOR.save_heatmap_pdf(pdf_path)
    print(f"Saved heatmaps to {pdf_path}")

if __name__ == "__main__":
    main()