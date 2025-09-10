# attn_hm_NS_nosm7.py

import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
# from photomaker import PhotoMakerStableDiffusionXLPipeline
from photomaker import PhotoMakerStableDiffusionXLPipeline2 as PhotoMakerStableDiffusionXLPipeline
from photomaker import FaceAnalysis2, analyze_faces
from transformers import CLIPTokenizer            # ← add (needed later if you keep tokenizer logic)


from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import types, math

import cv2

# `ADD_TO_PROMPT` forces the focus-word to be present in the generation prompt;
# if it is already there nothing is appended.
def create_heatmap(reference_image_path,
                   prompt,
                   TOKEN_FOCUS: str = "face",
                   ADD_TO_PROMPT: bool = False,
                   MASK_LAYERS: list[dict] | None = None,
                   MASK_JPG_NAME: str | None = None,    # ← NEW
                   MASK_PDF_NAME: str | None = None,     # ← NEW
                   SAVE_HEATMAPS: bool = True):          # ← NEW

    # ── global font object (about 3 % of the tile height) ───────────────────
    FONT_SIZE = 40                                  # tweak if tiles change
    try:                                            # try a TTF first
        ttf_path = next(
            p for p in (
                Path("/usr/share/fonts"), Path("/usr/local/share/fonts"))
            if p.is_dir()).rglob("DejaVuSans.ttf").__next__()
        font = ImageFont.truetype(str(ttf_path), FONT_SIZE)
    except StopIteration:
        font = ImageFont.load_default()             # fallback bitmap font




    # Device and dtype setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "mps":
        torch_dtype = torch.float16

    # Load base SDXL model and PhotoMaker V2 adapter:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", torch_dtype=torch_dtype
    ).to(device)
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")
    pipe.load_photomaker_adapter(os.path.dirname(ckpt_path), subfolder="", weight_name=os.path.basename(ckpt_path), trigger_word="img")
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)  # Euler sampler:contentReference[oaicite:6]{index=6}
    pipe.disable_xformers_memory_efficient_attention()  # disable for attention inspection

    # Reference image and identity embedding preparation
    # reference_image_path = "keanu.jpg"  # <--- set your reference image path
    # reference_image_path = "tom.jpg" 
    # reference_image_path = "eddie.webp" 
    # reference_image_path = "sydney.jpg" 
    
    reference_image_path = reference_image_path


    # Which token should the heat-map follow?
    #   "face" – fixed auxiliary prompt "a face" (sharper ID-agnostic face map)
    #   "img"  – the PhotoMaker trigger word inside *your* prompt
    #   "man"  – a normal word inside *your* prompt
    # TOKEN_FOCUS = "face"            # ← change to "img" or "man" when needed
    
    TOKEN_FOCUS = TOKEN_FOCUS
    # TOKEN_FOCUS = "man"  
    # prompt = "a man img with a beard in a space shuttle"
    # prompt = "a portrait of a man img with a beard playing football"
    # prompt = "a man img with enjoying pasta in a restaurant"
    # prompt = "a girl img in the arctic wearing a red jacket"

    # ── 0-bis. optionally inject the focus token into the prompt ────────────
    if ADD_TO_PROMPT and TOKEN_FOCUS not in prompt.split():
        prompt = f"{prompt} {TOKEN_FOCUS}"
    

    prompt = prompt

    # Load reference image
    ref_image = load_image(reference_image_path)
    # Detect face and get identity embedding:contentReference[oaicite:7]{index=7}
    face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    img_np = np.array(ref_image)[:, :, ::-1]  # convert PIL (RGB) to BGR NumPy for detector
    faces = analyze_faces(face_detector, img_np)
    if not faces:
        raise RuntimeError("No face detected in the reference image.")

    id_embed = torch.from_numpy(faces[0]["embedding"]).unsqueeze(0)  # identity embedding tensor


    tokenizer = getattr(pipe, "tokenizer", None) or CLIPTokenizer.from_pretrained(
        "SG161222/RealVisXL_V4.0", subfolder="tokenizer")

    # ── build K from **“a <word>”**  (matches legacy AUX_PROMPT behaviour) ──
    focus_prompt = f"a {TOKEN_FOCUS}"
    # focus_prompt = " " + TOKEN_FOCUS     # e.g. " face"
    # focus_prompt = TOKEN_FOCUS
    
    with torch.no_grad():
        focus_latents, *_ = pipe.encode_prompt(
            prompt=focus_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False)   # (1,T,2048)

    # locate the BPE span of “ face” inside “a face”
    prompt_ids = tokenizer(focus_prompt, add_special_tokens=False).input_ids
    word_ids   = tokenizer(" " + TOKEN_FOCUS, add_special_tokens=False).input_ids
    # word_ids   = tokenizer(TOKEN_FOCUS, add_special_tokens=False).input_ids

    def find_sub(seq, sub):
        for i in range(len(seq) - len(sub) + 1):
            if seq[i : i + len(sub)] == sub:
                return list(range(i, i + len(sub)))
        return []

    token_idx_global = find_sub(prompt_ids, word_ids)
    if not token_idx_global:
        raise RuntimeError(f"Could not locate '{TOKEN_FOCUS}' in auxiliary prompt")

    print(f"[TOKEN] IDs for '{TOKEN_FOCUS}': {token_idx_global}")
    if len(token_idx_global) > 1:
        print(f"[WARN] '{TOKEN_FOCUS}' splits into {len(token_idx_global)} BPE tokens – averaging will be used.")


    # Prepare lists to collect overlay images

    layer_names    = []     # to be filled after first callback
    heatmaps_cross = {}     # dict[layer] -> list[PIL.Image]
    heatmaps_mask  = []                    # frames for the mask montage
    final_image    = None

    # ---------- helper for JSON mask-layer spec --------------------------
    # each entry: {"name": str, "top_ratio": 0.1, "invert": false, "weight": 1.0}
    # fall-backs
    MASK_LAYERS = MASK_LAYERS or []
    for m in MASK_LAYERS:
        m.setdefault("top_ratio", 0.10)
        m.setdefault("invert", False)
        m.setdefault("weight", 1.0)


    # Seed for reproducibility
    seed = 42 # 56789
    generator = torch.Generator(device=device).manual_seed(seed)


    ###############################################################################
    # 🪄  Monkey‑patch every CrossAttention: Q = to_q(hidden_states),
    #      K = to_k(face_latents); heat‑map = softmax(Q·K_face)
    ###############################################################################
    from diffusers.models.attention_processor import Attention as CrossAttention

    attn_maps_current = {}                 # {layer_name: [head_maps]}
    layer_names    = []                    # will be filled later
    heatmaps_cross = {}
    final_image    = None
    


    def make_hook(layer_name, module):
        orig_forward = module.forward
        # scale = math.sqrt(module.head_dim)      # = √d  (matches SD-XL impl.)

        def forward_with_hook(hidden_states,
                            encoder_hidden_states=None,
                            attention_mask=None):

            # standard forward pass first ─ we still need the layer output
            out = orig_forward(hidden_states, encoder_hidden_states, attention_mask)
            if encoder_hidden_states is None:           # self-attention → skip
                return out

            # ── 0. keep only the “guided / conditional” half (CFG doubles batch) ──
            B_all = hidden_states.shape[0]
            hs_cond   = hidden_states[B_all // 2:]          # (B,L,C_img)
            enc_cond  = encoder_hidden_states[B_all // 2:]  # (B,T,C_txt)

            # ── 1. projections exactly like the real layer does ──────────────────
            proj_q = (module.to_q if hasattr(module,"to_q") else module.q_proj)(hs_cond)
            proj_k = (module.to_k if hasattr(module,"to_k") else module.k_proj)(
# +                        focus_latents.to(hs_cond.dtype).repeat(hs_cond.shape[0],1,1))
                        focus_latents.to(hs_cond.dtype).repeat(hs_cond.shape[0],1,1))


            B, L, C = proj_q.shape
            h, d    = module.heads, C // module.heads

            Q = proj_q.view(B, L, h, d).permute(0,2,1,3)        # (B,h,L,d)
            K = proj_k.view(B, -1, h, d).permute(0,2,1,3)       # (B,h,T,d)

            # ── 2. find token indices that spell TOKEN_FOCUS ─────────────────────
            
            token_idx = token_idx_global        # same for every layer

            # ── 3. scaled dot-product *before* the soft-max (same as SD-XL) ──────
            #     module.scale == 1/√dim_head
            logits   = (Q @ K.transpose(-2, -1)) * module.scale   # (B,h,L,T)
            # weights  = logits.softmax(-1)                       # soft-max over *T*
            weights = logits

            # # pick our token(s) and mean-pool if the word split into several BPEs
            # att      = weights[..., token_idx].mean(-1).mean(1)[0]   # (L,)
            
            # ── 3b. pool over BPEs (warned once at start) ───────────────────
            if len(token_idx_global) > 1:
                att = weights[..., token_idx_global].mean(-1).mean(1)[0]   # (L,)
            else:
                att = weights[..., token_idx_global[0]].mean(1)[0]  

            # ── 4. reshape to (H,W) and store ────────────────────────────────────
            H = int(math.sqrt(att.numel()))
            W = att.numel() // H

            # NumPy has no native bfloat16 → cast first, then move to CPU
            att2d = (
            att.to(dtype=torch.float32)    # <─ NEW
                .view(H, W)
                .cpu()
                .numpy()
            )

            layer_buf = attn_maps_current.setdefault(layer_name, [])
            layer_buf.append(att2d)          # keep heads already averaged

            return out

        # assign plain function – works as instance attribute
        return forward_with_hook



    for lname, mod in pipe.unet.named_modules():
        if isinstance(mod, CrossAttention):
            mod.forward = make_hook(lname, mod)


    num_steps = 50
    STEP_INTERVAL = 5 # 10  # how often to save attention maps

    
    def callback(step, timestep, latents):
        nonlocal attn_maps_current, layer_names, heatmaps_cross, heatmaps_mask, final_image

        if step % STEP_INTERVAL == 0 or step == num_steps - 1:
            # ❶  Consolidate maps *layer by layer* into square grids
            snapshot = {}
            for layer, maps in attn_maps_current.items():
                flat  = np.stack(maps).mean(0)                 # mean over heads
                H     = int(math.sqrt(flat.size))
                snapshot[layer] = flat.reshape(H, H)
                
                
            # ── ❶·A  build layer-group averages ────────────────────────────
            max_H = max(m.shape[0] for m in snapshot.values())
            def _up(m):
                return m if m.shape[0] == max_H else \
                       np.array(Image.fromarray(m).resize((max_H, max_H),
                                                          Image.BILINEAR))

            all_layers   = list(snapshot)
            up_layers    = [ln for ln in all_layers if ln.startswith("up")]
            down_layers  = [ln for ln in all_layers if ln.startswith("down")]
            mid_layers   = [ln for ln in all_layers if ln.startswith("mid")]

            def _avg(layer_list):
                return None if not layer_list else \
                       np.stack([_up(snapshot[ln]) for ln in layer_list]).mean(0)

            group_maps = {
                "AVG_ALL" : _avg(all_layers),
                "AVG_UP"  : _avg(up_layers),
                "AVG_DOWN": _avg(down_layers),
                "AVG_MID" : _avg(mid_layers),
            }
            # drop empty groups, then inject into `snapshot`
            snapshot.update({k: v for k, v in group_maps.items() if v is not None})

                

            if step == 0:
                k, v = next(iter(snapshot.items()))
                print(f"[DEBUG] first-snapshot  layer={k}  max={v.max():.4f}  mean={v.mean():.4f}")
                
                # NEW: print where the absolute max lives and some refs
                flat_idx = v.argmax()
                r, c     = divmod(flat_idx, v.shape[1])
                print(f"[VAL0] max@({r},{c})={v.max():.3f}  "
                    f"centre={v[v.shape[0]//2, v.shape[1]//2]:.3f}  "
                    f"corner={v[0,0]:.3f}")


            
            # ── DEBUG: check numeric values before colouring ────────────────
            if step == 0:
                first_layer = next(iter(snapshot))
                H0, W0      = snapshot[first_layer].shape
                centre_val  = snapshot[first_layer][H0 // 2, W0 // 2]
                corner_val  = snapshot[first_layer][0, 0]
                print(f"[VAL] centre={centre_val:.3f}  corner={corner_val:.3f}  "
                    f"max={snapshot[first_layer].max():.3f}")

            
            with torch.no_grad():
                
                vae_dev = next(pipe.vae.parameters()).device
                img = pipe.vae.decode(
                    (latents / 0.18215).to(device=vae_dev, dtype=pipe.vae.dtype)
                ).sample[0]

            img_np = ((img.float() / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # FIRST snapshot: decide which layers we will keep
            if not layer_names:                                  # first snapshot
                preferred = ["AVG_ALL", "AVG_UP", "AVG_DOWN", "AVG_MID"]
                layer_names = [p for p in preferred if p in snapshot] + \
                              [ln for ln in snapshot if ln not in preferred]
                heatmaps_cross = {ln: [] for ln in layer_names}

                
                # sanity-check requested mask layers
                bad = [m["name"] for m in MASK_LAYERS
                       if m["name"] not in snapshot]
                if bad:
                    print("[WARNING] mask layer(s) not found:",
                          ", ".join(bad))
    

            # pick colour-map: "jet" (default) or "Greys" for a monotone ramp
            cmap_name = "jet"          # ← change to "Greys" if you like
            INVERT    = False        # ← flip to True to check if colours are inverted
            colormap  = cm.get_cmap(cmap_name)

            # save the colour-bar once (step-0 of first layer)
            if step == 0:
                import matplotlib.pyplot as plt, numpy as _np
                plt.figure(figsize=(4, .4))
                plt.axis("off")
                plt.imshow(_np.linspace(0, 1, 256)[None, :],
                        cmap=colormap, aspect="auto")
                plt.savefig("colourbar.png", bbox_inches="tight")
                plt.close()        
            
            
            
            for ln in layer_names:
                amap        = snapshot[ln]
                amap_norm   = (amap / amap.max()) if amap.max() > 0 else amap
                amap_disp   = 1.0 - amap_norm if INVERT else amap_norm

                # ── build heat-map overlay (always) ───────────────────────
                hmap = (colormap(amap_disp)[..., :3] * 255).astype(np.uint8)
                hmap = np.array(Image.fromarray(hmap).resize((1024,1024),
                                                            Image.BILINEAR))
                heat_np = (0.5 * img_np + 0.5 * hmap).astype(np.uint8)
                heat_im = Image.fromarray(heat_np)

                # add 5 × 5 numeric grid on the heat-map
                draw = ImageDraw.Draw(heat_im)
                H_blk, W_blk = amap.shape[0] // 5, amap.shape[1] // 5
                vis_h, vis_w = 1024 // 5, 1024 // 5
                for bi in range(5):
                    for bj in range(5):
                        block = amap[bi*H_blk:(bi+1)*H_blk,
                                     bj*W_blk:(bj+1)*W_blk]
                        mv   = block.mean()
                        cx   = bj * vis_w + vis_w // 2
                        cy   = bi * vis_h + vis_h // 2
                        txt  = f"{mv:.2f}"
                        tw, th = draw.textbbox((0,0), txt, font=font)[2:]
                        draw.text((cx-tw//2, cy-th//2), txt,
                                  font=font, fill="white",
                                  stroke_width=2, stroke_fill="black")

                heatmaps_cross.setdefault(ln, []).append(heat_im)



            # ---- build *combined* mask per STEP ------------------
            # <<<<<<<<<<  after the for-loop over ln  >>>>>>>>>>
            if MASK_LAYERS:
                # ── pick a *common* resolution (largest among selected layers) ──
                base_H = max(
                    snapshot[sp["name"]].shape[0]
                    for sp in MASK_LAYERS if sp["name"] in snapshot
                )
                combined_bin = np.zeros((base_H, base_H), dtype=np.float32)

                total_w = 0.0
                for spec in MASK_LAYERS:
                    lname   = spec["name"]
                    if lname not in snapshot:
                        continue
                    amap    = snapshot[lname]

                    # up-/down-sample to the *base* resolution
                    if amap.shape[0] != base_H:
                        amap = np.array(
                            Image.fromarray(amap).resize((base_H, base_H),
                                                         Image.BILINEAR)
                        )

                    amap_n  = amap / amap.max() if amap.max() > 0 else amap
                    if spec["invert"]:
                        # amap_n = 1.0 - amap_n
                        thr = np.quantile(amap_n, spec["top_ratio"])
                        sel = (amap_n < thr)
                    else:
                        thr = np.quantile(amap_n, 1.0 - spec["top_ratio"])
                        sel = (amap_n > thr)

                    # ensure `sel` matches target size
                    if sel.shape[0] != base_H:
                        sel = np.array(
                            Image.fromarray(sel.astype(np.uint8)).resize(
                                (base_H, base_H), Image.NEAREST)
                        ).astype(bool)
                    
                    combined_bin += spec["weight"] * sel.astype(np.float32)
                    total_w += spec["weight"]

                # normalise & binarise union
                mask = (combined_bin / max(total_w, 1e-6)) > 0.0

                # keep largest blob, fill holes, etc. (same as before) ----
                n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                if n_lbl > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask = (lbl == largest)


                # Convex-hull closure
                n,lbl,stats,_ = cv2.connectedComponentsWithStats(
                                    mask.astype(np.uint8), 8)
                main = (lbl == 1 + np.argmax(stats[1:,cv2.CC_STAT_AREA])
                        ).astype(np.uint8)
                cnt, _ = cv2.findContours(main, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.convexHull(cnt[0])
                mask = np.zeros_like(main)
                # cv2.drawContours(mask, [hull], -1, 1, -1)
                # mask = mask.astype(bool)

                cv2.drawContours(mask, [hull], -1, 1, -1)

                # ── limit over-fill ──────────────────────────────
                HULL_MAX_AREA_FACTOR = 1 # 1.1   # allow up-to 40 % growth
                area_orig = main.sum()
                k_erode   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

                while mask.sum() > HULL_MAX_AREA_FACTOR * area_orig:
                    mask = cv2.erode(mask, k_erode, 1)     # shrink by 1 px
                    if not mask.any():                      # fallback safety
                        mask = main.copy()
                        break

                mask = mask.astype(bool)
                                
                
                
                # ── scale to 0/255 for alpha-blend ─────────────────────────────                
                mask_u8 = (mask * 255).astype(np.uint8)
                mask_u8 = cv2.resize(mask_u8, (1024, 1024),
                                     cv2.INTER_NEAREST)
                red     = np.zeros_like(img_np); red[..., 0] = 255
                alpha   = mask_u8[..., None] / 255.0
                mask_np = (img_np * (1 - alpha) + red * alpha).astype(np.uint8)
                heatmaps_mask.append(Image.fromarray(mask_np))



            # keep latest clean image for the “Final” panel
            final_image = Image.fromarray(img_np)

        # attn_maps_current = {}      # reset per step
        attn_maps_current.clear()   # reset per step without rebinding

    # Run the diffusion process with the callback (50 steps):contentReference[oaicite:8]{index=8}

    print(f'prompt: {prompt}')

    _ = pipe(
        prompt=prompt,
        negative_prompt="(asymmetry, worst quality, low quality, illustration, 3d, cartoon, sketch)", 
        input_id_images=[ref_image], id_embeds=id_embed, 
        num_inference_steps=num_steps, 
        start_merge_step=10,
        # start_merge_step=0, 
        generator=generator, callback=callback, callback_steps=1
    )


    # ─────────────────────────────────────────────────────────────
    #  Build and save montage from the overlays collected in‑callback
    # ─────────────────────────────────────────────────────────────
    import re
    header_h = 30

    for ln in layer_names:                          # set during callback
        cols = []
        for i, attn_img in enumerate(heatmaps_cross[ln]):
            step_num = i * STEP_INTERVAL                       # 0,10,20,30,40,50
            cols.append((f"S{step_num}", attn_img))

        cols.append(("Final", final_image))         # last clean frame

        img_w, img_h = cols[0][1].width, cols[0][1].height
        strip = Image.new("RGB", (img_w * len(cols), img_h + header_h),
                        color=(0, 0, 0))
        draw  = ImageDraw.Draw(strip)
        font  = ImageFont.load_default()

        for idx, (label, img) in enumerate(cols):
            x_off = idx * img_w
            strip.paste(img, (x_off, header_h))
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.text((x_off + (img_w - tw)//2, (header_h - th)//2),
                    label, font=font, fill=(255, 255, 255))


        safe = re.sub(r"[^\w\-]+", "_", ln)

        # --- save HEAT-MAP strip for *every* layer -----------------
        if SAVE_HEATMAPS:
            hm_dir = Path("heatmaps"); hm_dir.mkdir(parents=True, exist_ok=True)
            strip.save(hm_dir / f"{safe}_attn_hm.jpg")


    
    # ───────────────────────── save MASK montage ─────────────────────────
    if MASK_LAYERS and heatmaps_mask:
        cols = [(f"S{i*STEP_INTERVAL}", im) for i, im in enumerate(heatmaps_mask)]
        cols.append(("Final", final_image))

        img_w, img_h = cols[0][1].width, cols[0][1].height
        strip = Image.new("RGB", (img_w * len(cols), img_h + header_h),
                          color=(0, 0, 0))
        draw  = ImageDraw.Draw(strip)
        font  = ImageFont.load_default()

        for idx, (label, img) in enumerate(cols):
            x_off = idx * img_w
            strip.paste(img, (x_off, header_h))
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.text((x_off + (img_w - tw)//2, (header_h - th)//2),
                      label, font=font, fill=(255, 255, 255))


        # --- decide final file-name ---------------------------------
        if MASK_JPG_NAME:                                # user-supplied
            fname_out = MASK_JPG_NAME if MASK_JPG_NAME.lower().endswith(".jpg") \
                         else f"{MASK_JPG_NAME}.jpg"
        else:                                            # legacy auto name
            safe = "_".join([re.sub(r"[^\w\-]+", "_", m["name"])
                             for m in MASK_LAYERS if m["name"]])
            # fname_out = f"{safe}_mask.jpg"
            fname_out = f"final_mask.jpg"
        
        
        out_dir = Path("hm_results"); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname_out
        strip.save(out_path)
        print(f"[INFO] Mask montage saved to {out_path}")

        # ------- optional single-page PDF with header -----------------
        if MASK_PDF_NAME:

            HEADER_FNT_SZ = 60          # keep the size you like
            line_gap      = 8            # vertical gap between lines
            
            # text to display ------------------------------------------------
            info = [
                reference_image_path.rsplit(".",1)[0],       # file stem
                prompt,
                " / ".join(f"{sp['top_ratio']:.2f}|{sp['weight']:.2f}|{'inv' if sp['invert'] else 'norm'}"
                           for sp in MASK_LAYERS)
            ]
            
            # --- calculate exact header height ---
            num_lines = len(info)
            head_h    = HEADER_FNT_SZ * num_lines + line_gap * (num_lines-1) + 20

            # pick bigger font just for header
            header_font = ImageFont.truetype(str(ttf_path), HEADER_FNT_SZ) \
                          if 'ttf_path' in locals() else font

            page          = Image.new("RGB", (strip.width, strip.height+head_h),"white")
            draw          = ImageDraw.Draw(page)
            y_txt         = 10

            info     = [
                f"ref: {Path(reference_image_path).stem}",
                f"prompt: {prompt}",
                "layers: " + "; ".join(
                    f"r={m['top_ratio']},w={m['weight']},inv={m['invert']}"
                    for m in MASK_LAYERS)
            ]

            
            for ln in info:
                draw.text((10, y_txt), ln, fill="black", font=header_font)
                y_txt += HEADER_FNT_SZ + line_gap
            
            page.paste(strip, (0, head_h))

            pdf_name = MASK_PDF_NAME if MASK_PDF_NAME.lower().endswith(".pdf") \
                       else f"{MASK_PDF_NAME}.pdf"
            pdf_path = out_dir / pdf_name
            page.save(pdf_path, "PDF")
            print(f"[INFO] Mask PDF saved to {pdf_path}")


    # save final image
    if final_image is not None:
        final_image.save("final_image.jpg")
        
        

def create_pdf(output_pdf_file):
    # ─────────────────────────────────────────────────────────────
    #  QUICK PIL‑ONLY PDF MAKER  (≤10 rows per portrait A4 page)
    # ─────────────────────────────────────────────────────────────
    from PIL import Image, ImageDraw, ImageFont
    import os

    # --- config --------------------------------------------------
    DPI         = 150                       # output resolution
    PAGE_W_PX   = int(8.27 * DPI)           # A4 portrait 8.27×11.69 in
    PAGE_H_PX   = int(11.69 * DPI)
    ROWS_PER_PG = 10
    ROW_H_PX    = PAGE_H_PX // ROWS_PER_PG
    LABEL_W_PX  = int(PAGE_W_PX * 0.15)     # 15 % gutter for filename
    RIGHT_PAD   = 20                        # px margin on right
    FONT        = ImageFont.load_default()

    dir = Path("heatmaps")
    if not dir.is_dir():
        raise RuntimeError(f"Directory {dir} not found. "
                        "Run the attention evolution script first.")

    # gather montage strips
    montage_files = sorted(
        f for f in os.listdir(dir) if f.endswith('_attn_hm.jpg')
    )

    pages, y = [], 0
    page = Image.new('RGB', (PAGE_W_PX, PAGE_H_PX), 'white')
    draw = ImageDraw.Draw(page)

    def wrapped_label(draw, text, x, y_top, row_h, max_w):
        """Draw *any* filename (no spaces needed) within max_w pixels."""
        line_h  = FONT.getbbox("A")[3]
        max_lin = row_h // line_h
        lines, cur = [], ""

        for ch in text:
            trial = cur + ch
            if draw.textlength(trial, font=FONT) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)

        if len(lines) > max_lin:                 # truncate vertically
            lines = lines[:max_lin]
            if len(lines[-1]) > 1:
                while draw.textlength(lines[-1] + "…", font=FONT) > max_w:
                    lines[-1] = lines[-1][:-1]
                lines[-1] += "…"

        y_txt = y_top + (row_h - line_h * len(lines)) // 2
        for ln in lines:
            draw.text((x, y_txt), ln, fill="black", font=FONT)
            y_txt += line_h


    for fname in montage_files:
        # --- scale montage to fit row height *and* available width ----
        file_location = dir / fname
        strip = Image.open(file_location)
        max_w = PAGE_W_PX - LABEL_W_PX - RIGHT_PAD
        scale = min(ROW_H_PX / strip.height, max_w / strip.width)
        strip = strip.resize((int(strip.width * scale),
                            int(strip.height * scale)),
                            Image.LANCZOS)

        # --- new page if needed ---------------------------------------
        if y + ROW_H_PX > PAGE_H_PX:
            pages.append(page)
            page = Image.new('RGB', (PAGE_W_PX, PAGE_H_PX), 'white')
            draw = ImageDraw.Draw(page)
            y = 0

        # --- filename (wrapped) ---------------------------------------
        wrapped_label(draw, fname, 10, y, ROW_H_PX, LABEL_W_PX - 20)

        # --- paste montage strip --------------------------------------
        x_strip = LABEL_W_PX
        y_strip = y + (ROW_H_PX - strip.height) // 2
        page.paste(strip, (x_strip, y_strip))

        y += ROW_H_PX

    # final page
    pages.append(page)

    # --- save multipage PDF -------------------------------------------
    dir = Path("hm_results")
    dir.mkdir(exist_ok=True)
    
    out_pdf = dir / output_pdf_file
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])
    print(f'PDF saved to {out_pdf}')




import argparse, json

def main(cfg_path: str) -> None:
    # ---------- load config ----------
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # ---------- unpack & run ----------
    ref_img  = cfg["reference_image_path"]
    prompt   = cfg["prompt"]
    focus    = cfg["TOKEN_FOCUS"]
    add2prompt   = cfg.get("ADD_TO_PROMPT", False)   # <── NEW line
    pdf_file = cfg.get("output_pdf_file")
    mask_jpg = cfg.get("output_mask_file")    
    mask_pdf = cfg.get("output_mask_pdf_file")      # ← NEW
    run_grid    = cfg.get("run_mask_grid", False)
    # mask_top_ratio = cfg.get("mask_top_ratio", 0.10) 
    mask_layers_cfg = cfg.get("MASK_LAYERS", [])     # ← list of dicts



    out_paths = []                       # collect *.jpg for grid-mode

    def _single_run(layers_spec, fname_stub):
        out_name = f"{fname_stub}.jpg"
        create_heatmap(
            ref_img, prompt, focus,
            ADD_TO_PROMPT=add2prompt,
            MASK_LAYERS=layers_spec,
            MASK_JPG_NAME=out_name)
        out_paths.append(Path("hm_results")/out_name)

    if run_grid and len(mask_layers_cfg)==2:
        L1, L2 = mask_layers_cfg          # only pair-wise grid supported
        for w1, w2 in zip(L1["weight"], L2["weight"]):
            for tr1 in L1["top_ratio"]:
                for tr2 in L2["top_ratio"]:
                    spec = [
                        {"name":L1["name"],"weight":w1,
                         "top_ratio":tr1,
                         "invert": L1.get("invert",False)},
                        {"name":L2["name"],"weight":w2,
                         "top_ratio":tr2,
                         "invert": L2.get("invert",False)}
                    ]

                    tag = (f"w1={w1:.2f},w2={w2:.2f},"
                           f"tr1={tr1:.2f},tr2={tr2:.2f}")
                    print("[GRID]", tag, spec)            # debug print
                    _single_run(spec, f"grid_{tag}")
    else:
        _single_run(mask_layers_cfg, mask_jpg or "mask")

    # ---------- optional PDF outputs -----------------------------
    if mask_pdf and out_paths:
        from PIL import Image, ImageDraw, ImageFont



        # ---------- PAGE LAYOUT ----------------------------------
        BIG  = ImageFont.truetype(str(next(Path("/usr/share/fonts").rglob("DejaVuSans-Bold.ttf"))), 28)
        SMALL= ImageFont.truetype(str(next(Path("/usr/share/fonts").rglob("DejaVuSans.ttf"))), 18)
        A4_W, A4_H = 1240, 1754                     # ≈ A4 @150 dpi
        COLS, ROWS = 1, 10                           # 10 thumbs / page
        thumbs     = [Image.open(p) for p in out_paths]
        t_w, t_h   = thumbs[0].size
        pad_y      = BIG.getbbox("A")[3]*2+20       # header height




        BIG   = ImageFont.truetype(str(next(Path("/usr/share/fonts").rglob(
                        "DejaVuSans-Bold.ttf"))), 36)
        SMALL = ImageFont.truetype(str(next(Path("/usr/share/fonts").rglob(
                        "DejaVuSans.ttf"))), 24)

        A4_W, A4_H   = 1240, 1754                   # ≈ A4 @150 dpi



        thumbs       = [Image.open(p) for p in out_paths]
        if not thumbs:
            raise RuntimeError("no thumbnails produced – nothing to put in PDF")
        t_w, t_h     = thumbs[0].size
        MARGIN       = 30

        # ---------- build wrapped layer header *early* -----------------
        dummy_draw   = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        raw_hdr      = "; ".join(
            [f"{m['name']}  invert={m.get('invert', False)}"
             for m in mask_layers_cfg])
        layer_lines, cur = [], ""
        for token in raw_hdr.split("; "):
            nxt = (cur + "; " if cur else "") + token
            if dummy_draw.textlength(nxt, font=SMALL) <= A4_W - 2*MARGIN:
                cur = nxt
            else:
                if cur:
                    layer_lines.append(cur)
                cur = token
        if cur:
            layer_lines.append(cur)

        # two BIG lines + the wrapped SMALL lines
        hdr_h = (BIG.getbbox("A")[3]*2 +
                 len(layer_lines)*SMALL.getbbox("A")[3] + 3*MARGIN)


        GRID_TOP     = hdr_h
        CELL_W       = (A4_W - 2*MARGIN)
        CELL_H       = (A4_H - GRID_TOP - MARGIN) // ROWS


        pages, page = [], Image.new("RGB",(A4_W,A4_H),"white")
        draw        = ImageDraw.Draw(page)


        hdr_y = MARGIN

        draw.text((10,hdr_y), ref_img.rsplit(".",1)[0], font=BIG, fill="black")
        hdr_y+=BIG.getbbox("A")[3]+4
        draw.text((10,hdr_y), prompt, font=BIG, fill="black")

        hdr_y+=BIG.getbbox("A")[3]+4
        for ln in layer_lines:
            draw.text((10,hdr_y), ln, font=SMALL, fill="black")
            hdr_y += SMALL.getbbox("A")[3]

        # cell_w, cell_h = t_w, t_h+SMALL.getbbox("A")[3]+8
        idx = 0
        for im,path in zip(thumbs,out_paths):
            r,c = divmod(idx, COLS)
            if r==ROWS:                             # new A4 page
                pages.append(page)
                page  = Image.new("RGB",(A4_W,A4_H),"white")
                draw  = ImageDraw.Draw(page)
                hdr_y=MARGIN
                draw.text((10,hdr_y), ref_img.rsplit(".",1)[0], font=BIG, fill="black")
                hdr_y+=BIG.getbbox("A")[3]+4
                draw.text((10,hdr_y), prompt, font=BIG, fill="black")
                hdr_y+=BIG.getbbox("A")[3]+4
                for ln in layer_lines:
                    draw.text((10,hdr_y), ln, font=SMALL, fill="black")
                    hdr_y += SMALL.getbbox("A")[3]
                r,c,idx = 0,0,0                     # reset grid


            # ----- resize thumb to fit its cell ------------------
            scale  = min(CELL_W / im.width, CELL_H / (im.height+SMALL.getbbox("A")[3]+6))
            t_res  = im.resize((int(im.width*scale), int(im.height*scale)), Image.LANCZOS)

            x = MARGIN + c*CELL_W + (CELL_W - t_res.width)//2
            y = GRID_TOP + r*CELL_H + (CELL_H - (t_res.height+SMALL.getbbox("A")[3]+6))//2
            page.paste(t_res,(x,y))

            spec_txt = path.stem.split("grid_")[-1]   # weight|ratio|inv…
            tx = x + (t_res.width - draw.textlength(spec_txt,font=SMALL))//2
            draw.text((tx, y+t_res.height+4), spec_txt, font=SMALL, fill="black")

            idx +=1
        pages.append(page)
        pdf_path = Path("hm_results")/mask_pdf
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f"[INFO] mask-grid PDF saved to {pdf_path}")

        # -------- clean up individual *.jpg thumbnails -----------
        for p in out_paths:
            try: p.unlink()
            except: pass

    if pdf_file:  
        create_pdf(pdf_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention heat-map PDF")
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()

    main(args.config)