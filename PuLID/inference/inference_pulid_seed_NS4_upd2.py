#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_pulid_seed_NS4_upd2.py

CLI-compatible runner (mirrors PhotoMaker's inference script flags where
useful) that drives PuLID with SDXL-Lightning + DPM 4-step and optional
branched attention reusing PhotoMaker's helpers.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from typing import Optional

from PIL import Image, ImageDraw

import torch

# Ensure the repository root (one level up) is on sys.path when running
# this file directly (python3 inference/...). This makes `import pulid` work.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add sibling PhotoMaker package if running from the mono-repo
PHOTOMAKER_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir, "PhotoMaker"))
if os.path.isdir(PHOTOMAKER_ROOT) and PHOTOMAKER_ROOT not in sys.path:
    sys.path.insert(0, PHOTOMAKER_ROOT)

from pulid.pipeline import PuLIDPipeline
from pulid import attention_processor as attention


def parse_args():
    p = argparse.ArgumentParser("pulid_branched_runner")

    # dataset-style inputs (mirrors PhotoMaker runner)
    p.add_argument("--image_folder", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--class_file", required=True)
    p.add_argument("--output_dir", required=True)

    # generation controls
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--steps", type=int, default=4)  # Lightning default
    p.add_argument("--scale", type=float, default=1.2, help="CFG guidance scale")
    p.add_argument("--id_scale", type=float, default=0.8, help="PuLID ID scale (non-branched fallback)")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # branched controls (match names used by id_grid.sh)
    p.add_argument("--use_branched_attention", action="store_true", default=False)
    p.add_argument("--no_branched_attention", dest="use_branched_attention", action="store_false")
    p.add_argument("--branched_attn_start_step", type=int, default=1)
    # branched tuning (ported from PhotoMaker)
    p.add_argument("--pose_adapt_ratio", type=float, default=0.25,
                   help="0.0=preserve identity (strong face gating), 1.0=more pose (weaker gating)")
    p.add_argument("--ca_mixing_for_face", type=int, choices=[0,1], default=1,
                   help="If 1, also boost face in value path (stronger mixing)")

    # masking
    p.add_argument("--use_dynamic_mask", type=int, choices=[0,1], default=0)
    p.add_argument("--import_mask_folder", type=str, default=None)
    p.add_argument("--use_mask_folder", type=int, choices=[0,1], default=0)

    # prompts
    p.add_argument(
        "--neg_prompt",
        default=(
            "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, "
            "worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, "
            "ugly, disfigured, hands, low resolution, partially rendered objects, "
            "deformed or partially rendered eyes, deformed eyeballs, cross-eyed, blurry"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[PuLID] Loading pipeline …")
    base = PuLIDPipeline()
    print(f"[Init] Using pipeline class: {base.__class__.__module__}.{base.__class__.__name__}")

    # Match default attention mode used in pulid_generate3 (fidelity)
    attention.NUM_ZERO = 8
    attention.ORTHO = False
    attention.ORTHO_v2 = True

    # seed once (match pulid_generate3 seeding behavior)
    os.environ["PL_GLOBAL_SEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[Seed] Using seed = {args.seed}")

    # load prompts and classes
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    with open(args.class_file, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # collect reference images
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    refs = [os.path.join(args.image_folder, n) for n in sorted(os.listdir(args.image_folder))
            if os.path.splitext(n)[1].lower() in exts]
    print(f"[Init] Found {len(refs)} reference images in {args.image_folder}")

    # Always select the third variant seed offset (idx=2)
    TARGET_IMG_ID = 2

    for ref_path in refs:
        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
        ref_pil = Image.open(ref_path).convert("RGB")
        cls = class_map.get(ref_name)

        # Precompute ID embeddings (prefer NS v1.1 API when branched)
        from pulid.utils import resize_numpy_image_long
        np_ref = resize_numpy_image_long(np.array(ref_pil), 1024)
    
        
        # Always prefer NS v1.1 embeddings + pipeline for parity across modes
        use_v11_api = False
        image_embedding = None
        uncond_id_embedding = None
        id_embedding = None
        ns = None
        try:
            from pulid.pipeline_NS import PuLIDPipeline as PuLIDPipelineNS  # type: ignore
            ns = PuLIDPipelineNS()
            uncond_id_embedding, id_embedding = ns.get_id_embedding([np_ref])
            use_v11_api = True
        except Exception as _e_emb:
            print(f"[NS] Warning: NS pipeline unavailable ({_e_emb}); falling back to base encoder/pipeline.")
            try:
                image_embedding = base.get_id_embedding(np_ref)
            except Exception as _e_base_emb:
                print(f"[Emb] Error computing base id embedding: {_e_base_emb}")
                image_embedding = None

        for p_idx, prompt in enumerate(prompts):
            cur_prompt = prompt
            if "<class>" in prompt and cls:
                cur_prompt = prompt.replace("<class>", cls)

            # Single image only, using the seed corresponding to the 3rd variant
            for img_id in [TARGET_IMG_ID]:
                # Use manual binary masks (PNG paths specified here)
                # import_mask = "../compare/testing/ref3_masks/eddie_p0_2_pulid_gen_mask.png"
                # import_mask = "../compare/testing/ref3_masks/eddie_p0_0_pulid_gen_mask.png"
                # import_mask = "../compare/testing/ref3_masks/eddie_p0_2_pulid_gen_mask_new.png"
                import_mask = "../compare/testing/ref3_masks/eddie_p0_2_pulid_gen_mask_new2.png"
                # import_mask = "../compare/testing/ref3_masks/eddie_p0_2_pulid_gen_mask_new_easy.png"
                # import_mask = "../compare/testing/ref3_masks/mask_gen_black.png" 
                import_mask_ref = "../compare/testing/ref3_masks/eddie_mask_new.png"
                

                if use_v11_api:
                    # v1.1 style — use the SAME NS pipeline for both modes; toggle only the branched flag
                    try:
                        debug_dir = os.path.join(args.output_dir, "pulid_debug", ref_name)
                        imgs = ns.inference(
                            prompt=cur_prompt,
                            size=(1, args.height, args.width),
                            prompt_n=args.neg_prompt,
                            id_embedding=id_embedding,
                            uncond_id_embedding=uncond_id_embedding,
                            id_scale=args.id_scale,
                            guidance_scale=args.scale,
                            steps=args.steps,
                            seed=(int(args.seed) + TARGET_IMG_ID),
                            use_branched_attention=bool(args.use_branched_attention),
                            branched_attn_start_step=int(args.branched_attn_start_step),
                            pose_adapt_ratio=float(args.pose_adapt_ratio),
                            ca_mixing_for_face=bool(args.ca_mixing_for_face),
                            import_mask=import_mask if bool(args.use_branched_attention) else None,
                            import_mask_ref=import_mask_ref if bool(args.use_branched_attention) else None,
                            import_mask_folder=None,
                            use_mask_folder=False,
                            reference_pil=ref_pil,
                            debug_dir=debug_dir if bool(args.use_branched_attention) else None,
                        )
                        img = imgs[0]
                    except TypeError:
                        imgs = ns.inference(
                            prompt=cur_prompt,
                            size=(1, args.height, args.width),
                            prompt_n=args.neg_prompt,
                            id_embedding=id_embedding,
                            uncond_id_embedding=uncond_id_embedding,
                            id_scale=args.id_scale,
                            guidance_scale=args.scale,
                            steps=args.steps,
                            seed=(int(args.seed) + TARGET_IMG_ID),
                        )
                        img = imgs[0]
                        
                        
                else:
                    # v1.0 style
                    if not bool(args.use_branched_attention):
                        imgs = base.inference(
                            prompt=cur_prompt,
                            size=(1, args.height, args.width),
                            prompt_n=args.neg_prompt,
                            image_embedding=image_embedding,
                            id_scale=args.id_scale,
                            guidance_scale=args.scale,
                            steps=args.steps,
                        )
                        img = imgs[0]
                    else:
                        # Two-pass blend for v1.0 signature
                        def _reseed(s:int):
                            os.environ["PL_GLOBAL_SEED"]=str(s); torch.manual_seed(s)
                            if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
                            random.seed(s); np.random.seed(s)
                        _reseed(int(args.seed)+int(img_id))
                        img_emb = image_embedding if image_embedding is not None else base.get_id_embedding(np_ref)
                        imgs_base = base.inference(
                            prompt=cur_prompt,size=(1,args.height,args.width),prompt_n=args.neg_prompt,
                            image_embedding=img_emb,id_scale=args.id_scale,guidance_scale=args.scale,steps=args.steps,
                        )
                        img_base=imgs_base[0]

                        # temporary self-attn patch
                        import torch.nn as nn, torch.nn.functional as F
                        def _make_center_mask_latent(h,w,device,dtype):
                            H=h//8; W=w//8
                            yy,xx=torch.meshgrid(torch.arange(H,device=device),torch.arange(W,device=device),indexing='ij')
                            cy,cx=H/2.0,W/2.0; ry,rx=H*0.40,W*0.35
                            m=(((yy-cy)**2)/(ry**2)+((xx-cx)**2)/(rx**2))<=1.0
                            return m.to(dtype=dtype)[None,None]
                        class _MaskedSelfAttn(nn.Module):
                            def __init__(self,hidden,strength=0.7):
                                super().__init__(); self.strength=float(strength); self.mask=None
                                self.has_cross_attention_kwargs = True
                            def set_mask(self,m): self.mask=m
                            def __call__(self,attn,hs,encoder_hidden_states=None,attention_mask=None,temb=None,cross_attention_kwargs=None,id_embedding=None,id_scale:float=1.0,**kw):
                                res=hs; nd=hs.ndim
                                if attn.spatial_norm is not None: hs=attn.spatial_norm(hs,temb)
                                if nd==4:
                                    b,c,h,w=hs.shape; hs=hs.view(b,c,h*w).transpose(1,2)
                                b,L,_=hs.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                                am=None
                                if attention_mask is not None:
                                    am=attn.prepare_attention_mask(attention_mask,L,b); am=am.view(b,attn.heads,-1,am.shape[-1])
                                if attn.group_norm is not None: hs=attn.group_norm(hs.transpose(1,2)).transpose(1,2)
                                q=attn.to_q(hs)
                                enc=hs if encoder_hidden_states is None else (attn.norm_encoder_hidden_states(encoder_hidden_states) if attn.norm_cross else encoder_hidden_states)
                                k=attn.to_k(enc); v=attn.to_v(enc)
                                d=k.shape[-1]//attn.heads
                                q=q.view(b,-1,attn.heads,d).transpose(1,2); k=k.view(b,-1,attn.heads,d).transpose(1,2); v=v.view(b,-1,attn.heads,d).transpose(1,2)
                                H=int(L**0.5)
                                if self.mask is not None and H*H==L:
                                    m=self.mask; m=m[:,None] if m.dim()==3 else m
                                    m2d=F.interpolate(m.float(),size=(H,H),mode='bilinear',align_corners=False)
                                    m1d=m2d.flatten(2); scale=(1.0-self.strength)+self.strength*m1d
                                    scale=scale.to(device=q.device,dtype=q.dtype); q=q*scale.unsqueeze(-1)
                                out=F.scaled_dot_product_attention(q,k,v,attn_mask=am,dropout_p=0.0,is_causal=False)
                                out=out.transpose(1,2).reshape(b,-1,attn.heads*d).to(q.dtype)
                                out=attn.to_out[0](out); out=attn.to_out[1](out)
                                if nd==4: out=out.transpose(-1,-2).reshape(b,c,h,w)
                                if attn.residual_connection: out=out+res
                                return out/attn.rescale_output_factor
                        def _patch(unet,mask4,strength=0.7):
                            orig={}; procs={}; cnt=0
                            for n,p in unet.attn_processors.items():
                                if n.endswith('attn1.processor'):
                                    orig[n]=p
                                    if 'mid_block' in n: hidden=unet.config.block_out_channels[-1]
                                    elif n.startswith('up_blocks'): bid=int(n[len('up_blocks.'):].split('.')[0]); hidden=list(reversed(unet.config.block_out_channels))[bid]
                                    elif n.startswith('down_blocks'): bid=int(n[len('down_blocks.'):].split('.')[0]); hidden=unet.config.block_out_channels[bid]
                                    else: hidden=unet.config.block_out_channels[0]
                                    mproc=_MaskedSelfAttn(hidden,strength).to(unet.device,dtype=unet.dtype); mproc.set_mask(mask4); procs[n]=mproc; cnt+=1
                                else: procs[n]=p
                            unet.set_attn_processor(procs); print(f"[BrINF] patched self-attn: {cnt}"); return orig
                        def _restore(unet,orig):
                            procs=dict(unet.attn_processors)
                            for k,v in orig.items():
                                if k in procs: procs[k]=v
                            unet.set_attn_processor(procs)

                        mask_lat=_make_center_mask_latent(args.height,args.width,device=base.device,dtype=base.pipe.unet.dtype)
                        _orig=_patch(base.pipe.unet,mask_lat,strength=0.7)
                        _reseed(int(args.seed)+int(img_id))
                        img_emb = image_embedding if image_embedding is not None else base.get_id_embedding(np_ref)
                        imgs_face=base.inference(
                            prompt=cur_prompt,size=(1,args.height,args.width),prompt_n=args.neg_prompt,
                            image_embedding=img_emb,id_scale=args.id_scale,guidance_scale=args.scale,steps=args.steps,
                        )
                        _restore(base.pipe.unet,_orig)
                        img_face=imgs_face[0]

                        mask_pil=Image.new('L',(args.width,args.height),0); d=ImageDraw.Draw(mask_pil)
                        rx,ry=int(args.width*0.35),int(args.height*0.40); cx,cy=args.width//2,args.height//2
                        d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),fill=255)
                        base_np=np.array(img_base); face_np=np.array(img_face); mask_np=np.array(mask_pil)[:,:,None]/255.0
                        blend_np=(base_np*(1.0-mask_np)+face_np*mask_np).astype(np.uint8)
                        img=Image.fromarray(blend_np)
                # save

                out_path = os.path.join(args.output_dir, f"{ref_name}_p{p_idx}_{img_id}.jpg")
                img.save(out_path)
                print(f"[OK] saved {out_path}")

                # Debug: save mask overlays when branched and masks available
                if bool(args.use_branched_attention):
                    debug_dir = os.path.join(args.output_dir, "pulid_debug", ref_name)
                    os.makedirs(debug_dir, exist_ok=True)
                    gen_mask_path = import_mask if (import_mask and os.path.isfile(import_mask)) else None
                    ref_mask_path = import_mask_ref if (import_mask_ref and os.path.isfile(import_mask_ref)) else None
                    try:
                        if gen_mask_path:
                            mg = Image.open(gen_mask_path).convert('L').resize((args.width, args.height))
                            arr = np.array(img)
                            mb = np.array(mg) > 127
                            arr[mb, 0] = np.minimum(arr[mb, 0].astype(np.int32) + 80, 255).astype(np.uint8)
                            Image.fromarray(arr).save(os.path.join(debug_dir, f"overlay_gen_p{p_idx}_{img_id}.png"))
                            mg.save(os.path.join(debug_dir, f"mask_gen_p{p_idx}_{img_id}.png"))
                        if ref_mask_path:
                            mr = Image.open(ref_mask_path).convert('L').resize(ref_pil.size)
                            arr = np.array(ref_pil)
                            mb = np.array(mr) > 127
                            arr[mb, 1] = np.minimum(arr[mb, 1].astype(np.int32) + 80, 255).astype(np.uint8)
                            Image.fromarray(arr).save(os.path.join(debug_dir, f"overlay_ref.png"))
                            mr.save(os.path.join(debug_dir, f"mask_ref.png"))
                    except Exception as e:
                        print(f"[DbgMasks] Warning: failed to save debug overlays: {e}")


if __name__ == "__main__":
    main()
