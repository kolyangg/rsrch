# +++ pulid/pipeline_v1_1_NS2.py

import torch, os
import numpy as np
from types import SimpleNamespace

from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0

from pulid.pipeline_v1_1 import PuLIDPipeline   # reuse base
from pulid.attention_processor_NS2 import make_pulid_branched_processors

import photomaker.branched_new as bn
import inspect

# Reuse your PM branched + masking stack
from photomaker.branched_new import (
    two_branch_predict, prepare_reference_latents, patch_unet_attention_processors
)
from photomaker.branch_helpers import (
    aggregate_heatmaps_to_mask, prepare_mask4, save_branch_previews,
    debug_reference_latents_once, save_debug_ref_mask_overlay
)
from photomaker.add_masking import DynamicMaskGenerator, get_default_mask_config
from photomaker.mask_utils import MASK_LAYERS_CONFIG
from photomaker.create_mask_ref import compute_face_mask_from_pil


class PuLIDPipelineNS2(PuLIDPipeline):
    """
    PuLID with PhotoMaker-style Branched Attention integration.
    We operate on `self.pipe` (the inner SDXL pipeline).
    """
    def enable_branched_attention(self,
        use_branched_attention: bool,
        face_embed_strategy: str = "face",
        # mask sources
        import_mask: Optional[str] = None,
        import_mask_ref: Optional[str] = None,
        auto_mask_ref: bool = True,
        use_dynamic_mask: bool = False,
        mask_start: int = 10, mask_end: int = 20,
        save_heatmaps_dynamic: bool = True,
        heatmap_mode: str = "identity",
        token_focus: str = "face",
        add_token_to_prompt: bool = False,
        mask_layers_config: Optional[List[Dict]] = None,
        debug_dir: str = "hm_debug",
        # branched runtime knobs
        pose_adapt_ratio: float = 0.25,
        ca_mixing_for_face: bool = True,
    ):
        self._ns2_enabled = bool(use_branched_attention)
        pm = self.pipe
        
        # === Vanilla PuLID mode: restore original ID processors and exit ===
        if not self._ns2_enabled:
            # Put back PuLID's IDAttnProcessor on attn2 and AttnProcessor on attn1
            self.hack_unet_attn_layers(pm.unet)
            return
        

        # If branched attention is OFF, restore PuLID's original ID processors and return.
        if not self._ns2_enabled:
            # try:
            #     # pm.unet.set_attn_processor(None)  # revert to default processors if previously patched
            #     # Revert to standard Diffusers attention processors (avoid None)
            #     pm.unet.set_attn_processor(AttnProcessor2_0())
            # except Exception:
            #     pass
            # # drop any leftover branched state if present
            # for attr in ("_dyn","_face_mask","_face_mask_t","_face_mask_ref",
            #              "_face_mask_highres_ref","_face_mask_t_ref",
            #              "_heatmaps","_hm_layers","_step_tags","_orig_forwards"):
            #     if hasattr(pm, attr):
            #         try:
            #             delattr(pm, attr)
            #         except Exception:
            #             pass
            # Reinstall PuLID's IDAttnProcessor on attn2, AttnProcessor on attn1
            self.hack_unet_attn_layers(pm.unet)
            return
        
        # expose runtime flags to processors (same as PM)
        pm.pose_adapt_ratio   = float(pose_adapt_ratio)
        pm.ca_mixing_for_face = bool(ca_mixing_for_face)


        # (0) Provide a fallback feature_extractor expected by PM helpers
        if not hasattr(pm, "feature_extractor") or pm.feature_extractor is None:
            def _ns2_feature_extractor(image, return_tensors="pt"):
                # Prefer SDXL's image_processor if present (returns BCHW in [-1, 1])
                try:
                    if hasattr(pm, "image_processor") and pm.image_processor is not None:
                        pixel_values = pm.image_processor.preprocess(image)
                    else:
                        raise AttributeError
                except Exception:
                    # Safe PIL → tensor → [-1, 1]
                    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
                    pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                return SimpleNamespace(pixel_values=pixel_values)
            pm.feature_extractor = _ns2_feature_extractor


        # (0.5) Ensure heatmap buffers exist even if we import masks directly.
        # branch_helpers.aggregate_heatmaps_to_mask() clears `_heatmaps`.
        # Match PhotoMaker’s pipeline_br init so `.clear()` never crashes.
        if not hasattr(pm, "_heatmaps"):
            pm._heatmaps      = {}
            pm._hm_layers     = [s["name"] for s in MASK_LAYERS_CONFIG]
            pm._step_tags     = []
            pm._orig_forwards = {}


        # (1) prep dynamic mask generator on the *inner* SDXL pipeline if requested
        self._dyn = None
        if use_dynamic_mask:
            self._dyn = DynamicMaskGenerator(
                pipeline=pm, use_dynamic_mask=True,
                mask_start=mask_start, mask_end=mask_end,
                save_heatmaps=save_heatmaps_dynamic,
                token_focus=token_focus, add_to_prompt=add_token_to_prompt,
                mask_layers_config=mask_layers_config or MASK_LAYERS_CONFIG,
                debug_dir=debug_dir, num_inference_steps=50,
                heatmap_mode=heatmap_mode
            )

        # (2) reference mask (auto or provided)
        # if auto_mask_ref and getattr(pm, "_ref_img", None) is not None:
        #     # dynamic ref mask lives in branch_helpers.aggregate_heatmaps_to_mask with suffix "_ref"
        #     aggregate_heatmaps_to_mask(pm, mask_mode="spec", import_mask=import_mask_ref, suffix="_ref")
        #     save_debug_ref_mask_overlay(pm, debug_dir=debug_dir)
        # elif import_mask_ref:
        if import_mask_ref:
            aggregate_heatmaps_to_mask(pm, mask_mode="spec", import_mask=import_mask_ref, suffix="_ref")

        elif auto_mask_ref and getattr(pm, "_ref_img", None) is not None:
            # Build a high-res ref mask from the reference image (no heatmaps needed)
            try:
                ref_t = pm._ref_img[0] if pm._ref_img.dim() == 4 else pm._ref_img  # [3,H,W], in [-1,1]
                ref_np = ((ref_t.detach().float().cpu().permute(1, 2, 0) + 1) * 0.5 * 255).clamp(0, 255).numpy().astype("uint8")
                ref_pil = Image.fromarray(ref_np)
                mask_u8 = compute_face_mask_from_pil(ref_pil)              # H×W uint8 (0/255)
                mask_bool = (mask_u8 > 0)
                # Store both numpy (for fallback) and "highres" mask so prepare_mask4 can downscale precisely
                setattr(pm, "_face_mask_ref", mask_bool)
                setattr(pm, "_face_mask_highres_ref", mask_bool)
                setattr(pm, "_face_mask_t_ref", None)  # force rebuild at latent size
            except Exception as e:
                print(f"[NS2] auto_mask_ref failed: {e}")


        # (3) static generation mask (optional)
        if import_mask:
            aggregate_heatmaps_to_mask(pm, mask_mode="spec", import_mask=import_mask, suffix="")

        # (4) ensure default masks exist so prepare_mask4() never crashes on step 0
        import numpy as _np
        if getattr(pm, "_face_mask", None) is None:
            setattr(pm, "_face_mask", _np.zeros((8, 8), dtype=bool))  # tiny blank; will be upsampled
            setattr(pm, "_face_mask_t", None)
        # for reference mask: either we created _face_mask_highres_ref above or fall back to zeros
        if not hasattr(pm, "_face_mask_highres_ref") and getattr(pm, "_face_mask_ref", None) is None:
            setattr(pm, "_face_mask_ref", _np.zeros((8, 8), dtype=bool))
            setattr(pm, "_face_mask_t_ref", None)

        # # # (4) patch UNet processors (compose Branched + PuLID ID)
        # if self._ns2_enabled:
        #     cad = pm.unet.config.cross_attention_dim
        #     procs = make_pulid_branched_processors(pm.unet, cad, num_tokens=getattr(pm, "num_tokens", 77), scale=1.0)
        #     pm.unet.set_attn_processor(procs)

        # (4) Do NOT patch UNet here.
        # Branched processors are installed on-demand inside two_branch_predict(...)
        # together with the correct per-step masks (matching PhotoMaker behavior).


        # (4) Install our processors now (attn1 + attn2 with safe fallbacks)
        if self._ns2_enabled:
            cad = pm.unet.config.cross_attention_dim
            procs = make_pulid_branched_processors(pm.unet, cad, num_tokens=getattr(pm, "num_tokens", 77), scale=1.0)
            pm.unet.set_attn_processor(procs)

        # # (6) Torch compat: some versions lack 'generator' in torch.randn_like(...)
        # try:
        #     if "generator" not in inspect.signature(torch.randn_like).parameters:
        #         def _randn_like_compat(x, generator=None, **kwargs):
        #             return torch.randn(x.shape, device=x.device, dtype=x.dtype)
        #         # Patch the 'torch' module that branched_new uses
        #         bn.torch.randn_like = _randn_like_compat
        # except Exception as e:
        #     print(f"[NS2] randn_like compat shim skipped: {e}")
        

        # (6) Torch compat: force randn_like that ignores `generator` kwarg (older torch builds)
        def _randn_like_compat(x, *args, **kwargs):
            return torch.randn(x.shape, device=x.device, dtype=x.dtype)
        # Patch the 'torch' module that branched_new uses
        bn.torch.randn_like = _randn_like_compat

        # (5) Monkeypatch PM's patcher so two_branch_predict uses our NS2 processors and manual masks
        def _ns2_patch_unet_attention_processors(pipeline, mask, mask_ref, scale: float = 1.0,
                                                 id_embeds=None, class_tokens_mask=None):
            cad = pipeline.unet.config.cross_attention_dim
            procs = make_pulid_branched_processors(pipeline.unet, cad,
                                                   num_tokens=getattr(pipeline, "num_tokens", 77),
                                                   scale=scale)
            # set processors
            pipeline.unet.set_attn_processor(procs)
            # feed masks & optional id embeds
            for p in pipeline.unet.attn_processors.values():
                if hasattr(p, "set_masks"):
                    p.set_masks(mask, mask_ref)
                elif hasattr(p, "mask"):
                    setattr(p, "mask", mask); setattr(p, "mask_ref", mask_ref)
                if id_embeds is not None and hasattr(p, "id_embeds"):
                    p.id_embeds = id_embeds.to(pipeline.device, dtype=pipeline.unet.dtype)
                # propagate runtime flags
                for k in ("pose_adapt_ratio", "ca_mixing_for_face"):
                    if hasattr(pipeline, k):
                        setattr(p, k, getattr(pipeline, k))
        bn.patch_unet_attention_processors = _ns2_patch_unet_attention_processors


    @torch.no_grad()
    def generate_branched(self,
        prompt: str,
        id_image_np: "np.ndarray",
        height: int = 1024, width: int = 1024,
        steps: int = 30, guidance_scale: float = 5.0,
        seed: int = 42,
        # schedule switches (names mirror your PM CLI for parity)
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        branched_attn_start_step: int = 15,
        branched_start_mode: str = "both",
        face_embed_strategy: str = "face",
        # masks / flags same as enable_branched_attention
        **kw,
    ):
        """
        Minimal denoising loop that mirrors your PM flow but drives `self.pipe`
        and preserves PuLID ID adapter via cross_attention_kwargs.
        """
        # torch.manual_seed(int(seed))
        # pm = self.pipe
        # device = pm.device

        # # 0) id embedding from PuLID (original path)
        # # get_id_embedding may return Tensor or (Tensor, aux...) – normalize to Tensor
        # _id_out = self.get_id_embedding([id_image_np])
        # if isinstance(_id_out, (tuple, list)):
        #     id_embedding = _id_out[0]
        # elif isinstance(_id_out, dict):
        #     id_embedding = _id_out.get("id_embedding") or _id_out.get("id_embeds") or next(iter(_id_out.values()))
        # else:
        #     id_embedding = _id_out
        # if not torch.is_tensor(id_embedding):
        #     raise TypeError(f"Expected Tensor from get_id_embedding, got {type(id_embedding)}")
        # # id_embedding = id_embedding.to(device=pm.unet.device, dtype=pm.unet.dtype)

        # # Match UNet weight dtype to avoid Half vs Float matmul inside IDAttnProcessor
        # unet_dtype = next(pm.unet.parameters()).dtype
        # id_embedding = id_embedding.to(device=pm.unet.device, dtype=unet_dtype)

        torch.manual_seed(int(seed))
        pm = self.pipe
        device = pm.device

        # === Pure PuLID path when branched is OFF =============================
        if not self._ns2_enabled:
            # Use PuLID's own ID embedding + sampler (matches original behavior)
            uncond_id, pos_id = self.get_id_embedding([id_image_np])
            return self.inference(
                prompt=prompt,
                size=(1, height, width),
                prompt_n="",  # keep default negative empty (same as many examples)
                id_embedding=pos_id,
                uncond_id_embedding=uncond_id,
                id_scale=float(getattr(self, "id_scale", 0.8)),
                guidance_scale=float(guidance_scale),
                steps=int(steps),
                seed=int(seed),
           )[0]

        # === Branched path below =============================================
        # 0) id embedding for branched (dtype/device aligned later in processor)
        _id_out = self.get_id_embedding([id_image_np])
        if isinstance(_id_out, (tuple, list)):   # (uncond, pos)
            id_embedding = _id_out[1]
        elif isinstance(_id_out, dict):
            id_embedding = _id_out.get("id_embedding") or _id_out.get("id_embeds") or next(iter(_id_out.values()))
        else:
            id_embedding = _id_out
        if not torch.is_tensor(id_embedding):
            raise TypeError(f"Expected Tensor from get_id_embedding, got {type(id_embedding)}")
        unet_dtype = next(pm.unet.parameters()).dtype
        id_embedding = id_embedding.to(device=pm.unet.device, dtype=unet_dtype)


        # # 1) encode main prompt (SDXL native)
        # prompt_embeds, neg_embeds, pooled, neg_pooled, _ = pm.encode_prompt(
        #     prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
        # )
        # added_cond = {"text_embeds": pooled, "time_ids": pm._get_add_time_ids((height, width), (height, width), 0, 0, device)}


        # 1) encode main prompt (SDXL native) -> 4-tuple
        # prompt_embeds, neg_embeds, pooled, neg_pooled = pm.encode_prompt(
        prompt_pos, prompt_neg, pooled_pos, pooled_neg = pm.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
        )
        # # CFG: concat [neg, pos]
        # prompt_cat = torch.cat([neg_embeds, prompt_embeds], dim=0)
        # pooled_cat = torch.cat([neg_pooled, pooled], dim=0)
        # add_time_ids = pm._get_add_time_ids((height, width), (height, width), 0, 0, device)
        # add_time_ids = add_time_ids.to(dtype=prompt_cat.dtype)
        # add_time_ids = add_time_ids.repeat(2, 1)  # batch=2 for CFG

        # SDXL helper: some diffusers versions require text_encoder_projection_dim (None would crash).
        proj_dim = 0
        try:
            if getattr(pm, "text_encoder_2", None) is not None:
                proj_dim = int(getattr(pm.text_encoder_2.config, "projection_dim", 0) or 0)
        except Exception:
            proj_dim = 0
        try:
            add_time_ids_pos = pm._get_add_time_ids(
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(height, width),
                dtype=prompt_pos.dtype,
                text_encoder_projection_dim=proj_dim,
            )
        except TypeError:
            # Fallback for older signature without text_encoder_projection_dim
            add_time_ids = pm._get_add_time_ids(
               (height, width), (0, 0), (height, width), prompt_cat.dtype
            )

        # add_time_ids = add_time_ids.repeat(2, 1)  # batch=2 for CFG
        # # batch=2 for CFG and move to same device/dtype as text embeddings
        # add_time_ids = add_time_ids.repeat(2, 1).to(device=prompt_cat.device, dtype=prompt_cat.dtype)
        
        # # added_cond = {"text_embeds": pooled_cat, "time_ids": add_time_ids}
        
        # # # Branched helpers rely on this flag being present on the pipeline
        # # pm.do_classifier_free_guidance = True

        # added_cond = {"text_embeds": pooled_cat, "time_ids": add_time_ids}
        
        # Prepare conds for two paths:
        #  (A) vanilla CFG path → concat [neg,pos]
        prompt_cat = torch.cat([prompt_neg, prompt_pos], dim=0)
        pooled_cat = torch.cat([pooled_neg, pooled_pos], dim=0)
        add_time_ids_cat = add_time_ids_pos.repeat(2, 1).to(device=prompt_cat.device, dtype=prompt_cat.dtype)
        added_cond_cat = {"text_embeds": pooled_cat, "time_ids": add_time_ids_cat}
        #  (B) branched path → single-pos (two_branch_predict doubles internally)
        add_time_ids_pos = add_time_ids_pos.to(device=prompt_pos.device, dtype=prompt_pos.dtype)
        added_cond_pos = {"text_embeds": pooled_pos, "time_ids": add_time_ids_pos}
            
        # Use a proxy so branched helpers see do_classifier_free_guidance=True
        class _PMProxy:
            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "do_classifier_free_guidance", True)
            def __getattr__(self, name):
                return getattr(self._inner, name)
            def __setattr__(self, name, value):
                if name in ("_inner", "do_classifier_free_guidance"):
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._inner, name, value)
        pm_proxy = _PMProxy(pm)


        # # 2) init latents & timesteps
        # pm.scheduler.set_timesteps(steps, device=device)
        # latents = torch.randn( (1, pm.unet.in_channels, height // pm.vae_scale_factor, width // pm.vae_scale_factor),
        #                        device=device, dtype=pm.unet.dtype )
        # # 3) reference latents for the two-branch predictor (use your PM util)
        # # ref_latents = prepare_reference_latents(pm, Image.fromarray(id_image_np))

        # #    PM variant expects (pipeline, ref_pil, height, width, dtype)
        # ref_latents = prepare_reference_latents(
        #     pm,
        #     Image.fromarray(id_image_np),
        #     height,
        #     width,
        #     pm.unet.dtype,
        # )

        # # 2) init latents & timesteps
        # pm.scheduler.set_timesteps(steps, device=device)
        # latents = torch.randn(
        #     (1, pm.unet.in_channels, height // pm.vae_scale_factor, width // pm.vae_scale_factor),
        #     device=device, dtype=pm.unet.dtype
        # )

        # === Vanilla path when branched is OFF =================================
        if not self._ns2_enabled:
        #    for t in pm.scheduler.timesteps:
        #         latent_in = pm.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        #         noise_pred = pm.unet(
        #             latent_in, t,
        #             encoder_hidden_states=prompt_cat,
        #             added_cond_kwargs=added_cond_cat,
        #             cross_attention_kwargs=getattr(pm, "_cross_attention_kwargs", None),
        #         ).sample
        #         # CFG combine
        #         noise_uncond, noise_text = noise_pred.chunk(2)
        #         noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        #         latents = pm.scheduler.step(noise_pred, t, latents).prev_sample

        #    image = pm.vae.decode((latents / pm.vae.config.scaling_factor).to(dtype=pm.vae.dtype)).sample
        #    image = (image / 2 + 0.5).clamp(0, 1)
        #    image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        #    return Image.fromarray(image)

            # Use PuLID's original sampler & ID path
            uncond_id, pos_id = self.get_id_embedding([id_image_np])
            pil = self.inference(
                prompt=prompt,
                size=(1, height, width),
                prompt_n="",                             # keep default negs empty here
                id_embedding=pos_id,
                uncond_id_embedding=uncond_id,
                id_scale=float(getattr(self, "id_scale", 0.8)),
                guidance_scale=float(guidance_scale),
                steps=int(steps),
                seed=int(seed),
             )[0]
            return pil
 
        # 2) init latents & timesteps (branched path only)
        pm.scheduler.set_timesteps(steps, device=device)
        latents = torch.randn(
            (1, pm.unet.in_channels, height // pm.vae_scale_factor, width // pm.vae_scale_factor),
            device=device, dtype=pm.unet.dtype
        )


        # 3) reference latents for the two-branch predictor (branched only)
        ref_latents = prepare_reference_latents(
            pm, Image.fromarray(id_image_np), height, width, pm.unet.dtype
        )



        # optional: dynamic mask hooks
        if self._dyn is not None:
            self._dyn.setup_hooks(prompt, class_tokens_mask=None, num_tokens=getattr(pm, "num_tokens", 77))

        for i, t in enumerate(pm.scheduler.timesteps):
            step_idx = i
            # keep heatmaps updated if dynamic mask
            if self._dyn is not None:
                self._dyn.update_mask(step_idx, latents)
                if self._dyn.current_mask is not None:
                    pm._face_mask = self._dyn.current_mask  # numpy bool H×W
                    pm._face_mask_t = None                  # force rebuild

            # Prepare masks at the current latent resolution
            mask_gen   = prepare_mask4(pm, latents, suffix="")
            mask_ref   = prepare_mask4(pm, latents, suffix="_ref")

            # Build cross_attention_kwargs so our cross-attn shim can fetch PuLID’s ID
            pm._cross_attention_kwargs = {"id_embedding": id_embedding, "id_scale": float(getattr(self, "id_scale", 0.8))}

            if self._ns2_enabled and step_idx >= branched_attn_start_step:
                # Use your two-branch predictor (handles processor patching + mask passing)
                noise_pred, noise_face, noise_bg = two_branch_predict(
                    # pipeline=pm,
                    # pipeline=pm_proxy,
                    # latent_model_input=pm.scheduler.scale_model_input(latents, t),
                    # t=t,
                    # # prompt_embeds=prompt_embeds,
                    # prompt_embeds=prompt_cat,
                    # added_cond_kwargs=added_cond,
                    pipeline=pm_proxy,
                    latent_model_input=pm.scheduler.scale_model_input(latents, t),
                    t=t,
                    prompt_embeds=prompt_pos,          # single-pos
                    added_cond_kwargs=added_cond_pos,  # single-pos (will be doubled inside)
                    mask4=mask_gen, mask4_ref=mask_ref,
                    reference_latents=ref_latents,
                    face_embed_strategy=face_embed_strategy,
                    id_embeds=None,
                    step_idx=step_idx, scale=1.0,
                )
            # else:
            #     # vanilla unet call (CFG)
            #     latent_in = pm.scheduler.scale_model_input(torch.cat([latents]*2), t)
            #     # noise_pred = pm.unet(latent_in, t, encoder_hidden_states=prompt_embeds,
            #     noise_pred = pm.unet(latent_in, t, encoder_hidden_states=prompt_cat,
            #                          added_cond_kwargs=added_cond_cat,
            #                          cross_attention_kwargs=getattr(pm, "_cross_attention_kwargs", None)).sample
            # # CFG combine (same as SDXL)
            # noise_uncond, noise_text = noise_pred.chunk(2)
            # noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            else:
                # vanilla unet call (CFG)
                latent_in = pm.scheduler.scale_model_input(torch.cat([latents]*2), t)
                noise_pred = pm.unet(
                    latent_in, t,
                    encoder_hidden_states=prompt_cat,
                    added_cond_kwargs=added_cond_cat,
                    cross_attention_kwargs=getattr(pm, "_cross_attention_kwargs", None),
                ).sample
                # CFG combine only for vanilla path
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = pm.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        image = pm.vae.decode( (latents / pm.vae.config.scaling_factor).to(dtype=pm.vae.dtype) ).sample
        image = (image / 2 + 0.5).clamp(0,1)
        image = (image[0].permute(1,2,0).float().cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(image)
