#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from huggingface_hub import hf_hub_download
from diffusers.schedulers import EulerDiscreteScheduler

# Project imports
from photomaker.pipeline_br import PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces
from diffusers.utils import load_image


# =========================
# CLI (kept the same names & defaults you were using)
# =========================
def build_parser():
    p = argparse.ArgumentParser("qk_corr_regions")
    
    ref = "keanu"
    # ref = "eddie"
    
    if ref == "eddie":

        # masks (q side)
        p.add_argument("--q_mouth", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_mouth.png")
        p.add_argument("--q_eyes",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_eyes.png")
        p.add_argument("--q_nose",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_noba_nose.png")

        # # masks (k side)
        # p.add_argument("--k_mouth", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_mouth.png")
        # p.add_argument("--k_eyes",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_eyes.png")
        # p.add_argument("--k_nose",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_p0_0_r05_cface_P10_B15_nose.png")

        p.add_argument("--k_mouth", default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_ref_mouth.png")
        p.add_argument("--k_eyes",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_ref_eyes.png")
        p.add_argument("--k_nose",  default="../compare/testing/ref3_masks/masks_for_qk_corr/eddie_ref_nose.png")
        
        p.add_argument("--image_folder", default="../compare/testing/ref3") # eddie
        p.add_argument("--output_dir", default="../compare/results/qk_corr_run1")
        p.add_argument("--import_mask", default="../compare/testing/ref3_masks/eddie_pm_mask_new.jpg")

    elif ref == "keanu":
        p.add_argument("--q_mouth", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_mouth.png")
        p.add_argument("--q_eyes", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_eyes.png")
        p.add_argument("--q_nose", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_noba_nose.png")
        p.add_argument("--k_mouth", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_mouth.png")
        p.add_argument("--k_eyes", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_eyes.png")
        p.add_argument("--k_nose", default="../compare/testing/ref2_masks/masks_for_qk_corr/keanu_ref_nose.png")
        
        p.add_argument("--image_folder", default="../compare/testing/ref2") # keanu
        p.add_argument("--output_dir", default="../compare/results/qk_corr_keanu") 
        p.add_argument("--import_mask", default="../compare/testing/ref2_masks/keanu_gen_mask.png")
    

    # standard inputs
    p.add_argument("--prompt_file", default="../compare/testing/prompt_one2.txt")
    p.add_argument("--class_file",  default="../compare/testing/classes_ref.json")
    


    # analysis knobs
    # p.add_argument("--step_min", type=int, default=30)
    p.add_argument("--step_min", type=int, default=40)
    p.add_argument("--latest_layers", type=int, default=4)

    # generation
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # branched / PM schedule
    p.add_argument("--photomaker_start_step", type=int, default=10)
    p.add_argument("--merge_start_step", type=int, default=10)
    p.add_argument("--branched_attn_start_step", type=int, default=15)
    p.add_argument("--branched_start_mode", type=str, default="both", choices=["branched", "both"])

    p.add_argument("--use_branched_attention", action="store_true", default=True)
    p.add_argument("--auto_mask_ref", action="store_true", default=True)
    p.add_argument("--pose_adapt_ratio", type=float, default=0.25)
    p.add_argument("--ca_mixing_for_face", type=int, default=1)
    p.add_argument("--use_id_embeds", type=int, default=0)
    p.add_argument("--face_embed_strategy", type=str, default="face", choices=["face", "id_embeds"])
        

    p.add_argument("--import_mask_ref", type=str, default=None)
    p.add_argument("--use_dynamic_mask", type=int, choices=[0, 1], default=0)  # 1=True, 0=False
    # debug
    p.add_argument("--debug", type=int, choices=[0,1], default=1)

    return p


# =========================
# Small utils
# =========================
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

def load_image_paths(folder: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])

def pil_to_binary_mask(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return (arr >= 128).astype(np.uint8)

def resize_mask_to_tokens(mask_hw: np.ndarray, L: int) -> np.ndarray:
    H = int(round(math.sqrt(L))); W = H
    m = Image.fromarray((mask_hw * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return (np.array(m, dtype=np.uint8) >= 128).astype(np.uint8).reshape(-1)


# =========================
# Correlation collector
# =========================
class CorrCollector:
    def __init__(self, q_masks_hw: Dict[str, np.ndarray], k_masks_hw: Dict[str, np.ndarray],
                 step_min: int, latest_layers: int, out_dir: Path, debug: bool = False):
        self.q_masks_hw = q_masks_hw
        self.k_masks_hw = k_masks_hw
        self.step_min = step_min
        self.latest_layers = latest_layers
        self.out_dir = out_dir
        self.debug = bool(debug)
        self.layer_order: List[str] = []
        self._seen = set()
        self.maps: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._mouth_debug_emitted = 0  # rate-limit mouth debug
        self._nsq_debug_emitted = 0    # non-square K debug (mouth-only)

    def register_layer(self, layer_name: str):
        if layer_name not in self._seen:
            self.layer_order.append(layer_name)
            self._seen.add(layer_name)

    def _accum(self, layer: str, region: str, vec: np.ndarray):
        bucket = self.maps.setdefault(layer, {}).setdefault(region, {"sum": None, "cnt": 0, "Lk": len(vec)})
        Lk_tgt = bucket["Lk"]
        v = vec
        if len(v) != Lk_tgt:
            # safety: resample to the target grid for this layer key
            Hs = int(round(math.sqrt(len(v)))); Ht = int(round(math.sqrt(Lk_tgt)))
            img = Image.fromarray(v.reshape(Hs, Hs))
            v = np.array(img.resize((Ht, Ht), Image.BILINEAR)).reshape(Lk_tgt)
        bucket["sum"] = (v.astype(np.float64) if bucket["sum"] is None else bucket["sum"] + v)
        bucket["cnt"] += 1

    def add_call(self, layer: str, q_face: torch.Tensor, k_face: torch.Tensor, step_idx: int,
                 q_region: str, device: torch.device):
        if step_idx < self.step_min:
            return
        # auto-register unseen layer keys (for class-level patching)
        if layer not in self._seen:
            self.register_layer(layer)

        B, Hh, Lq, D = q_face.shape
        _, _, Lk, _ = k_face.shape
        # Skip non-spatial K (e.g., cross-attn text: 77 tokens)
        Hk = int(round(math.sqrt(Lk)))
        if Hk * Hk != Lk:
           if self.debug and q_region == "mouth" and self._nsq_debug_emitted < 3:
                print(f"[DBG][{layer}][step {step_idx}] skip non-square K length: Lk={Lk}")
                self._nsq_debug_emitted += 1
           return

        q_mask = resize_mask_to_tokens(self.q_masks_hw[q_region], Lq)
        if q_mask.sum() == 0:
            if q_region == "mouth" and self._mouth_debug_emitted < 3:
                print(f"[DBG][{layer}][step {step_idx}] q_mouth mask empty after resize: Lq={Lq}")
                self._mouth_debug_emitted += 1
            return
        
        q_mask_t = torch.from_numpy(q_mask.astype(np.float32)).to(device)  # float32 weights
        # Upcast to fp32 for stable cosine + to avoid dtype mismatch with fp16 UNet
        qf = q_face.to(torch.float32)
        kf = k_face.to(torch.float32)
        qn = qf / (qf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        kn = kf / (kf.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        if q_region == "mouth" and self._mouth_debug_emitted < 3:
            qnan = torch.isnan(qf).any().item(); knan = torch.isnan(kf).any().item()
            print(f"[DBG][{layer}][step {step_idx}] q_face={tuple(q_face.shape)}/{q_face.dtype} "
                  f"k_face={tuple(k_face.shape)}/{k_face.dtype} → using fp32")
 
        w = q_mask_t.view(1, 1, Lq, 1)
        wsum = w.sum(dim=2).clamp_min(1e-6)
        q_region_vec = (qn * w).sum(dim=2) / wsum.squeeze(2)  # [B,H,D] (fp32)
        sim = torch.matmul(q_region_vec.unsqueeze(2), kn.transpose(-1, -2)).squeeze(2)  # [B,H,Lk] (fp32)
        sim_vec = sim.mean(dim=(0, 1)).detach().float().cpu().numpy()
        if q_region == "mouth" and self._mouth_debug_emitted < 3:
            kmask = resize_mask_to_tokens(self.k_masks_hw["mouth"], Lk)
            inside = float(np.nanmean(sim_vec[kmask.astype(bool)])) if kmask.sum() else float("nan")
            outside = float(np.nanmean(sim_vec[~kmask.astype(bool)])) if kmask.sum() and kmask.sum() < Lk else float("nan")
            any_nan = bool(np.isnan(sim_vec).any())
            max_idx = int(np.nanargmax(sim_vec)) if not any_nan else -1
            print(f"[DBG][{layer}][step {step_idx}] sim mouth→k: any_nan={any_nan} "
                  f"mean(k_mouth)={inside:.4f} mean(k_not_mouth)={outside:.4f} "
                  f"max@{max_idx}={np.nanmax(sim_vec) if not any_nan else float('nan'):.4f}")
            self._mouth_debug_emitted += 1
        self._accum(layer, q_region, sim_vec)

    def _area_mask_k(self, area: str, Lk: int) -> np.ndarray:
        return resize_mask_to_tokens(self.k_masks_hw[area], Lk)

    def _layer_region_triplet_means(self, layer: str, region: str) -> Dict[str, float]:
        b = self.maps[layer][region]
        Lk = int(b["Lk"])
        avg_map = b["sum"] / max(1, b["cnt"])
        out = {}
        for area in K_AREAS:
            km = self._area_mask_k(area, Lk)
            out[area] = float(avg_map[km.astype(bool)].mean()) if km.sum() else float("nan")
        return out

    def _aggregate_over_layers(self, layers: List[str]) -> Dict[str, Dict[str, float]]:
        out = {}
        for region in Q_REGIONS:
            maps, sizes = [], []
            for ly in layers:
                if ly in self.maps and region in self.maps[ly]:
                    b = self.maps[ly][region]
                    Lk = int(b["Lk"])
                    maps.append(b["sum"] / max(1, b["cnt"]))
                    sizes.append(int(round(math.sqrt(Lk))))
            if not maps:
                out[region] = {a: float("nan") for a in K_AREAS}
                continue
            tgtH = max(sizes); tgtL = tgtH * tgtH
            maps_rs = []
            for m, H in zip(maps, sizes):
                if H == tgtH:
                    maps_rs.append(m)
                else:
                    img = Image.fromarray(m.reshape(H, H))
                    img = img.resize((tgtH, tgtH), Image.BILINEAR)
                    maps_rs.append(np.array(img).reshape(tgtL))
            avg_all = np.stack(maps_rs, 0).mean(0)
            res = {}
            for area in K_AREAS:
                km = self._area_mask_k(area, tgtL)
                res[area] = float(avg_all[km.astype(bool)].mean()) if km.sum() else float("nan")
            out[region] = res
        return out

    def build_report(self) -> Dict:
        per_layer = {}
        for ly in self.layer_order:
            if ly not in self.maps:
                continue
            per_layer[ly] = {}
            for region in Q_REGIONS:
                if region not in self.maps[ly]:
                    continue
                tri = self._layer_region_triplet_means(ly, region)
                argmax_area = max(tri, key=lambda k: (tri[k] if not np.isnan(tri[k]) else -1e9))
                per_layer[ly][region] = {
                    "k_area_means": tri,
                    "verdict_same_area": (argmax_area == region),
                    "best_match": argmax_area,
                }

        layers_all = [ly for ly in self.layer_order if ly in self.maps]
        layers_lastN = layers_all[-self.latest_layers:] if self.latest_layers > 0 else layers_all

        def verdicts(agg):
            out = {}
            for region, tri in agg.items():
                argmax_area = max(tri, key=lambda k: (tri[k] if not np.isnan(tri[k]) else -1e9))
                out[region] = {
                    "k_area_means": tri,
                    "verdict_same_area": (argmax_area == region),
                    "best_match": argmax_area,
                }
            return out

        agg_all = self._aggregate_over_layers(layers_all)
        agg_lastN = self._aggregate_over_layers(layers_lastN)

        return {
            "meta": {
                "step_min": self.step_min,
                "latest_layers": self.latest_layers,
                "layer_order": layers_all,
                "layers_lastN": layers_lastN,
            },
            "per_layer": per_layer,
            "aggregate_all_layers": verdicts(agg_all),
            "aggregate_lastN_layers": verdicts(agg_lastN),
        }

    def save_heatmap_pdf(self, pdf_path: Path):
        layers_all = [ly for ly in self.layer_order if ly in self.maps]
        layers_lastN = layers_all[-self.latest_layers:] if self.latest_layers > 0 else layers_all


        # def k_means_str(heat: np.ndarray) -> str:
        #     vec = heat.reshape(-1)
        #     L = vec.size
        #     nums = []
        #     for area in K_AREAS:
        #         km = resize_mask_to_tokens(self.k_masks_hw[area], L)
        #         if km.sum():
        #             val = float(vec[km.astype(bool)].mean())
        #             nums.append(f"{val:.4f}" if not np.isnan(val) else "nan")
        #         else:
        #             nums.append("nan")
        #     return " ".join(nums)  # mouth eyes nose other
        

        def k_means_str(heat: np.ndarray) -> str:
            vec = heat.reshape(-1)
            L = vec.size
            lines = []
            for area in K_AREAS:  # mouth, eyes, nose, other
                km = resize_mask_to_tokens(self.k_masks_hw[area], L)
                if km.sum():
                    val = float(vec[km.astype(bool)].mean())
                    s = "nan" if np.isnan(val) else f"{val:.4f}"
                else:
                    s = "nan"
                lines.append(f"{area}: {s}")
            return "\n".join(lines)

        def make_avg(layers: List[str], region: str) -> np.ndarray:
            maps, sizes = [], []
            for ly in layers:
                if ly in self.maps and region in self.maps[ly]:
                    b = self.maps[ly][region]
                    Lk = int(b["Lk"])
                    maps.append(b["sum"] / max(1, b["cnt"]))
                    sizes.append(int(round(math.sqrt(Lk))))
            if not maps:
                return np.zeros((64, 64), dtype=np.float32)
            tgtH = max(sizes); tgtL = tgtH * tgtH
            maps_rs = []
            for m, H in zip(maps, sizes):
                if H == tgtH:
                    maps_rs.append(m)
                else:
                    img = Image.fromarray(m.reshape(H, H))
                    img = img.resize((tgtH, tgtH), Image.BILINEAR)
                    maps_rs.append(np.array(img).reshape(tgtL))
            return np.stack(maps_rs, 0).mean(0).reshape(tgtH, tgtH).astype(np.float32)

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path.as_posix()) as pdf:
            fig1 = plt.figure(figsize=(10, 3.3), dpi=150)
            for i, r in enumerate(Q_REGIONS, 1):
                ax = fig1.add_subplot(1, 3, i)
                # ax.imshow(make_avg(layers_all, r), interpolation="nearest")
                # ax.set_title(f"All layers: {r}"); ax.axis("off")

                heat = make_avg(layers_all, r)
                ax.imshow(heat, interpolation="nearest")
                ax.text(0.98, 0.98, k_means_str(heat), transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="white")
                ax.set_title(f"All layers: {r}"); ax.axis("off")

            fig1.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

            fig2 = plt.figure(figsize=(10, 3.3), dpi=150)
            for i, r in enumerate(Q_REGIONS, 1):
                ax = fig2.add_subplot(1, 3, i)

                heat = make_avg(layers_lastN, r)
                ax.imshow(heat, interpolation="nearest")
                ax.text(0.98, 0.98, k_means_str(heat), transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="white")
                ax.set_title(f"Last {self.latest_layers}: {r}"); ax.axis("off")

            fig2.tight_layout(); pdf.savefig(fig2); plt.close(fig2)
            
            # Per-layer pages: one page per analyzed layer with 3 heatmaps (mouth/eyes/nose)
            layers_captured = [ly for ly, regions in self.maps.items()
                               if any(reg in regions for reg in Q_REGIONS)]
            for ly in layers_captured:
                figL = plt.figure(figsize=(10, 3.3), dpi=150)
                for i, r in enumerate(Q_REGIONS, 1):
                    ax = figL.add_subplot(1, 3, i)
                    b = self.maps[ly].get(r)
                    if b and b["cnt"] > 0:
                        Lk = int(b["Lk"])
                        H = int(round(math.sqrt(Lk)))

                        heat = (b["sum"] / max(1, b["cnt"])).reshape(H, H)
                        ax.imshow(heat, interpolation="nearest")
                        ax.text(0.98, 0.98, k_means_str(heat), transform=ax.transAxes,
                                ha="right", va="top", fontsize=8, color="white")

                    ax.set_title(f"{ly}: {r}")
                    ax.axis("off")
                figL.tight_layout()
                pdf.savefig(figL)
                plt.close(figL)


# =========================
# Live capture (no math reimplementation)
# =========================
GLOBAL = {"STEP": 0, "COLLECTOR": None, "CALLS": 0}
# Regions (q-side fixed) and K-areas (k-side includes 'other' = complement)
Q_REGIONS = ("mouth", "eyes", "nose")
K_AREAS   = ("mouth", "eyes", "nose", "other")

# Try to import both processor classes; fall back to name checks if needed
try:
    from photomaker.attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
except Exception:
    BranchedAttnProcessor = None
    BranchedCrossAttnProcessor = None


def make_capture_wrapper(layer_name: str, orig_call):
    """
    Wrap BranchedAttnProcessor.__call__ and, **in the same inputs/tensors**,
    compute q (via attn.to_q(noise_hidden)), mask_gate (via _prepare_mask),
    q_face = q * mask_gate, and key_face = to_k(ref_hidden). Then call collector.
    """
    def _wrapped(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale: float = 1.0):
        # out = orig_call(self, attn, hidden_states, encoder_hidden_states,
        #                 attention_mask, temb, scale)

        # collector: CorrCollector = GLOBAL["COLLECTOR"]
        # if collector is None or GLOBAL["STEP"] < collector.step_min:
        #     return out
        out = orig_call(self, attn, hidden_states, encoder_hidden_states,
                        attention_mask, temb, scale)

        # Collector may be unset early; don't step-gate here.
        step_idx = GLOBAL["STEP"]
        collector: CorrCollector = GLOBAL["COLLECTOR"]
        if collector is None:
            return out

        # Recreate inputs exactly as the processor sees them
        x = hidden_states
        if x.ndim == 4:
            b0, c0, h0, w0 = x.shape
            x = x.view(b0, c0, h0 * w0).transpose(1, 2)

        total = x.shape[0]; half = total // 2
        noise_hidden = x[:half]
        ref_hidden   = x[half:]

        if attn.group_norm is not None:
            noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            ref_hidden   = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)

        batch_size = noise_hidden.shape[0]
        seq_len    = noise_hidden.shape[1]
        head_dim   = attn.heads
        dim_per    = noise_hidden.shape[-1] // head_dim

        q = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
        key_face = attn.to_k(ref_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)

        mask_gate = None
        if getattr(self, "mask", None) is not None:
            mask_gate = self._prepare_mask(self.mask, seq_len, batch_size)\
                          .to(dtype=q.dtype, device=q.device)  # [B,1,L,1]

        q_face = q * mask_gate if mask_gate is not None else q
        # One-time concise mouth debug at first capture on/after step_min
        col = collector
        if hasattr(col, "_mouth_debug_emitted") and col._mouth_debug_emitted < 1:
            mg = mask_gate
            mg_stats = (float(mg.mean().item()), float((mg > 0).float().mean().item())) if mg is not None else None
            print(f"[DBG][{layer_name}][step {GLOBAL['STEP']}] wrapper active: "
                  f"q={tuple(q.shape)} k={tuple(key_face.shape)} "
                  f"mask={'yes' if mg is not None else 'no'} "
                  f"{'(mean,>0frac)=' + str(mg_stats) if mg is not None else ''}")
            col._mouth_debug_emitted += 1

        # push three face regions
        for region in Q_REGIONS:
            collector.add_call(layer_name, q_face, key_face, step_idx, region, q.device)
        GLOBAL["CALLS"] += 1
        return out
    return _wrapped

# def attach_captures(pipeline, collector: CorrCollector):
#     from photomaker.attn_processor import BranchedAttnProcessor
#     new = 0
#     for name, proc in pipeline.unet.attn_processors.items():
#         if isinstance(proc, BranchedAttnProcessor):
#             collector.register_layer(name)
#             if not hasattr(proc, "_orig_call_for_corr"):
#                 proc._orig_call_for_corr = proc.__call__
#                 proc.__call__ = make_capture_wrapper(name, proc._orig_call_for_corr).__get__(proc, proc.__class__)
#                 new += 1
#     return new


def _is_target_proc(p):
    if p is None:
       return False
    clsname = p.__class__.__name__
    if BranchedAttnProcessor and isinstance(p, BranchedAttnProcessor):
        return True
    if BranchedCrossAttnProcessor and isinstance(p, BranchedCrossAttnProcessor):
        return True
    # Fallback by name (in case of import aliasing)
    return clsname in {"BranchedAttnProcessor", "BranchedCrossAttnProcessor"}

def attach_captures(pipeline, collector: CorrCollector):
    new = 0
    # 1) Dict-style processors (Diffusers mapping)
    attn_procs = getattr(pipeline.unet, "attn_processors", None)
    if isinstance(attn_procs, dict):
        for name, proc in attn_procs.items():
            if _is_target_proc(proc):
                collector.register_layer(name)
                if not hasattr(proc, "_orig_call_for_corr"):
                   proc._orig_call_for_corr = proc.__call__
                   proc.__call__ = make_capture_wrapper(name, proc._orig_call_for_corr).__get__(proc, proc.__class__)
                   new += 1
    # 2) Module-style processors (Attention modules with .processor)
    for name, mod in pipeline.unet.named_modules():
        proc = getattr(mod, "processor", None)
        if _is_target_proc(proc):
            layer_name = f"{name}.processor"
            collector.register_layer(layer_name)
            if not hasattr(proc, "_orig_call_for_corr"):
                proc._orig_call_for_corr = proc.__call__
                proc.__call__ = make_capture_wrapper(layer_name, proc._orig_call_for_corr).__get__(proc, proc.__class__)
                new += 1
    return new

# --- NEW: class-level patch so ALL instances are wrapped even after re-patching ---
_CLASSES_PATCHED = False
def patch_processor_classes():
    global _CLASSES_PATCHED
    if _CLASSES_PATCHED:
        return
    if BranchedAttnProcessor is not None and not hasattr(BranchedAttnProcessor, "_orig_call_for_corr_cls"):
        BranchedAttnProcessor._orig_call_for_corr_cls = BranchedAttnProcessor.__call__
        def _class_wrap_sa(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
            out = self._orig_call_for_corr_cls(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
            collector = GLOBAL["COLLECTOR"]
            if collector is None:
                return out
            # Recreate inputs exactly like instance wrapper (no step gating here)
            x = hidden_states
            if x.ndim == 4:
                b0, c0, h0, w0 = x.shape
                x = x.view(b0, c0, h0 * w0).transpose(1, 2)
            total = x.shape[0]; half = total // 2
            noise_hidden = x[:half]; ref_hidden = x[half:]
            if attn.group_norm is not None:
                noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
                ref_hidden   = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
            batch_size = noise_hidden.shape[0]; seq_len = noise_hidden.shape[1]
            head_dim = attn.heads; dim_per = noise_hidden.shape[-1] // head_dim
            q = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            k_face = attn.to_k(ref_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            mask_gate = None
            if getattr(self, "mask", None) is not None:
                mask_gate = self._prepare_mask(self.mask, seq_len, batch_size).to(dtype=q.dtype, device=q.device)
            q_face = q * mask_gate if mask_gate is not None else q
            step_idx = GLOBAL["STEP"]
               
            layer_key = f"BrSA@{id(self)}"
            for region in Q_REGIONS:
                collector.add_call(layer_key, q_face, k_face, step_idx, region, q.device)
                
            GLOBAL["CALLS"] += 1
            return out
        BranchedAttnProcessor.__call__ = _class_wrap_sa
    if BranchedCrossAttnProcessor is not None and not hasattr(BranchedCrossAttnProcessor, "_orig_call_for_corr_cls"):
        BranchedCrossAttnProcessor._orig_call_for_corr_cls = BranchedCrossAttnProcessor.__call__
        def _class_wrap_ca(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
            out = self._orig_call_for_corr_cls(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
            collector = GLOBAL["COLLECTOR"]
            if collector is None:
                return out
            # # Similar capture for CA (optional but harmless)
            # x = hidden_states
            # if x.ndim == 4:
            #     b0, c0, h0, w0 = x.shape
            #     x = x.view(b0, c0, h0 * w0).transpose(1, 2)
            # total = x.shape[0]; half = total // 2
            # noise_hidden = x[:half]; ref_hidden = x[half:]
            # if attn.group_norm is not None:
            #     noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)
            #     ref_hidden   = attn.group_norm(ref_hidden.transpose(1, 2)).transpose(1, 2)
            # batch_size = noise_hidden.shape[0]; seq_len = noise_hidden.shape[1]
            # head_dim = attn.heads; dim_per = noise_hidden.shape[-1] // head_dim
            # q = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            # k_face = attn.to_k(ref_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            
            # Cross-attn: Q from UNet hidden, K from encoder_hidden_states (context).
            x = hidden_states
            if x.ndim == 4:
                b0, c0, h0, w0 = x.shape
                x = x.view(b0, c0, h0 * w0).transpose(1, 2)
            total = x.shape[0]; half = total // 2
            noise_hidden = x[:half]                      # queries from noise half
            if attn.group_norm is not None:
                noise_hidden = attn.group_norm(noise_hidden.transpose(1, 2)).transpose(1, 2)

            # Context must be provided; split to take the face half for keys
            if encoder_hidden_states is None:
                return out
            ctx = encoder_hidden_states                 # [2B, L_ctx, C_ctx]
            if ctx.ndim == 4:                           # just in case, flatten spatial
                b1, c1, h1, w1 = ctx.shape
                ctx = ctx.view(b1, c1, h1 * w1).transpose(1, 2)
            Bq = noise_hidden.shape[0]
            if ctx.shape[0] >= 2 * Bq:
                ctx_face = ctx[Bq:]                     # face half
            else:
                ctx_face = ctx[ctx.shape[0] // 2:]      # fallback split
            if ctx_face.shape[0] != Bq:
                return out  # mismatch — don't collect

            batch_size = Bq
            seq_len = noise_hidden.shape[1]
            head_dim = attn.heads
            dim_per = noise_hidden.shape[-1] // head_dim

            q = attn.to_q(noise_hidden).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            # K from the *context* (correct in_features: cross_attention_dim, e.g., 2048)
            k_face = attn.to_k(ctx_face).view(batch_size, -1, head_dim, dim_per).transpose(1, 2)
            
            mask_gate = None
            if getattr(self, "mask", None) is not None:
               mask_gate = self._prepare_mask(self.mask, seq_len, batch_size).to(dtype=q.dtype, device=q.device)
            q_face = q * mask_gate if mask_gate is not None else q
            step_idx = GLOBAL["STEP"]
            # for region in ("mouth", "eyes", "nose"):
            #     collector.add_call("<BrCA>", q_face, k_face, step_idx, region, q.device)
            
            layer_key = f"BrCA@{id(self)}"
            for region in Q_REGIONS:
                collector.add_call(layer_key, q_face, k_face, step_idx, region, q.device)
            
            GLOBAL["CALLS"] += 1
            return out
        BranchedCrossAttnProcessor.__call__ = _class_wrap_ca
    _CLASSES_PATCHED = True

# =========================
# Main
# =========================
def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir); ensure_dir(out_dir)

    # --- load binary region masks
    q_masks_hw = {
        "mouth": pil_to_binary_mask(args.q_mouth),
        "eyes":  pil_to_binary_mask(args.q_eyes),
        "nose":  pil_to_binary_mask(args.q_nose),
    }
    k_masks_hw = {
        "mouth": pil_to_binary_mask(args.k_mouth),
        "eyes":  pil_to_binary_mask(args.k_eyes),
        "nose":  pil_to_binary_mask(args.k_nose),
    }

    # Derive 'other' on K-side: complement of union(mouth, eyes, nose)
    k_union = (k_masks_hw["mouth"] | k_masks_hw["eyes"] | k_masks_hw["nose"]).astype(np.uint8)
    k_masks_hw["other"] = (1 - k_union).astype(np.uint8)
    
    if args.debug:
        qm = q_masks_hw["mouth"]; km = k_masks_hw["mouth"]
        print(f"[DBG] q_mouth mask: shape={qm.shape} sum={qm.sum()} uniq={np.unique(qm, return_counts=True)}")
        print(f"[DBG] k_mouth mask: shape={km.shape} sum={km.sum()} uniq={np.unique(km, return_counts=True)}")

    # --- pipeline (mirror your runner)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", torch_dtype=torch_dtype
    ).to(device)

    # Load PM-v2 adapter (adds id encoder & trigger), then fuse
    pm_bin = hf_hub_download("TencentARC/PhotoMaker-V2", "photomaker-v2.bin", repo_type="model")
    pipe.load_photomaker_adapter(
        os.path.dirname(pm_bin), subfolder="", weight_name=os.path.basename(pm_bin), trigger_word="img"
    )
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.disable_xformers_memory_efficient_attention()

    # runtime knobs (same names as you tune)
    pipe.pose_adapt_ratio   = float(args.pose_adapt_ratio)
    pipe.ca_mixing_for_face = bool(args.ca_mixing_for_face)
    pipe.use_id_embeds      = bool(args.use_id_embeds)

    # --- inputs: use exactly ONE ref image (id_grid does per-ID pages)
    image_paths = load_image_paths(args.image_folder)
    if not image_paths:
        raise SystemExit("No images found in --image_folder")
    ref_path = image_paths[0]
    ref_pil  = Image.open(str(ref_path)).convert("RGB")

    # --- id_embeds (same detector path)
    face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection','recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640,640))
    id_embeds = None
    try:
        bgr = np.array(ref_pil)[:, :, ::-1]
        faces = analyze_faces(face_detector, bgr)
        if faces:
            id_embeds = torch.from_numpy(faces[0]["embedding"]).unsqueeze(0)  # [1,512]
    except Exception:
        pass
    if id_embeds is None:
        id_embeds = torch.zeros(1, 512)

    # --- prompt & negative prompt (same string used in your runner)
    prompt_txt = load_txt(args.prompt_file)
    class_map  = json.loads(load_txt(args.class_file))
    trigger = "img"
    cls = class_map.get(ref_path.stem)
    cur_prompt = prompt_txt
    if "<class>" in cur_prompt and cls:
        cur_prompt = cur_prompt.replace("<class>", f"{cls} {trigger}")
    if trigger not in cur_prompt.split():
        cur_prompt = f"{cur_prompt} {trigger}"
    negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, cartoon, sketch)"

    # --- seed (mirror runner)
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    print(f"[Seed] Using seed = {args.seed}")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- correlation collector & attach after processors are patched in-pipeline
    collector = CorrCollector(q_masks_hw, k_masks_hw, step_min=args.step_min,
                              latest_layers=args.latest_layers, out_dir=out_dir,
                              debug=bool(args.debug))
    
    GLOBAL["COLLECTOR"] = collector
    # attach_captures(pipe, collector)  # attach to current processors
    # if args.debug:
    #     print(f"[DBG] Attached wrappers to {len(collector.layer_order)} BrAttn layers "
    #           f"(first 5: {collector.layer_order[:5]})")
    
    
    # Monkey-patch the patcher so when your code swaps in branched processors, we auto-wrap them.
    import photomaker.branched_new as _bn
    _ORIG_PATCH = _bn.patch_unet_attention_processors
    def _patched_patch(*_a, **_kw):
        res = _ORIG_PATCH(*_a, **_kw)
        newly = attach_captures(_a[0], collector)
        if args.debug and newly:
            print(f"[DBG] patch_unet_attention_processors: attached {newly} new wrappers "
                  f"(total tracked: {len(collector.layer_order)})")
        return res
    _bn.patch_unet_attention_processors = _patched_patch

    
    # Patch classes (one-time) so future instances are automatically wrapped.
    patch_processor_classes()
    pre = attach_captures(pipe, collector)  # pre-run (may be 0 before patching)
    if args.debug:
        print(f"[DBG] Pre-run: attached {pre} BrAttn wrappers "
              f"(layers tracked: {len(collector.layer_order)})")

    # keep step index updated
    def step_cb(pipeline, step_index, t, tensors):
        # Advance to the *next* step so captures start exactly at step_min.
        GLOBAL["STEP"] = int(step_index) + 1

        # Re-attach each step to catch processors swapped in right at the threshold
        newly = attach_captures(pipeline, collector)
        if args.debug and newly > 0:
            print(f"[DBG] step {step_index}: attached {newly} new Br* wrappers "
                  f"(total layers tracked: {len(collector.layer_order)})")
        if args.debug and int(step_index) + 1 == collector.step_min:
            print(f"[DBG] step_min reached → capturing enabled at step {step_index}")
        return tensors

    # --- run (forward exactly the knobs you use)
    result = pipe(
        prompt=cur_prompt,
        negative_prompt=negative_prompt,
        input_id_images=[ref_pil],
        id_embeds=id_embeds,
        num_images_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        # legacy + new:
        start_merge_step=args.merge_start_step,
        photomaker_start_step=args.photomaker_start_step,
        merge_start_step=args.merge_start_step,
        # branched:
        use_branched_attention=args.use_branched_attention,
        branched_attn_start_step=args.branched_attn_start_step,
        branched_start_mode=args.branched_start_mode,
        # masks:
        auto_mask_ref=args.auto_mask_ref,
        import_mask=args.import_mask,
        import_mask_ref=args.import_mask_ref,
        use_dynamic_mask=bool(args.use_dynamic_mask),
        # strategy:
        face_embed_strategy=args.face_embed_strategy,
        # callback:
        callback_on_step_end=step_cb,
    )

    if args.debug:
        n_layers_wrapped = len(collector.layer_order)
        n_layers_captured = sum(1 for ly in collector.layer_order if ly in collector.maps)
        print(f"[DBG] Post-run: wrapped={n_layers_wrapped} layers, "
              f"captured_nonempty={n_layers_captured}")
        print(f"[DBG] Wrapper calls observed: {GLOBAL['CALLS']}")
        if n_layers_captured == 0:
            print("[DBG] No captures stored. Check step_min vs num_inference_steps, "
                  "and confirm branched layers execute after step_min.")

    # --- save generated image
    try:
        imgs = getattr(result, "images", result)
        if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
            (out_dir / "generated.png").parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(out_dir / "generated.png")
            print(f"[QK] Saved generated image → {out_dir / 'generated.png'}")
    except Exception as e:
        print(f"[QK] Warning: failed to save generated image: {e}")

    # --- save report & pdf
    report = collector.build_report()
    (out_dir / "qk_corr_report.json").write_text(json.dumps(report, indent=2))
    collector.save_heatmap_pdf(out_dir / "qk_corr_heatmaps.pdf")
    print(f"[QK] Saved report → {out_dir/'qk_corr_report.json'}")
    print(f"[QK] Saved heatmaps → {out_dir/'qk_corr_heatmaps.pdf'}")


if __name__ == "__main__":
    main()
