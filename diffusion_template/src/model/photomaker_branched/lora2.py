from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import CLIPImageProcessor

import os
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from src.model.photomaker_path import resolve_photomaker_path
from src.model.sdxl.original import SDXL

##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
"""Import branched-attention forward/patch helpers and PMv2 face-ID dependencies used by training."""
from .insightface_package import create_face_analyzer
from .lora2_helpers import (
    install_branched_processors_for_training,
    prepare_branched_training_inputs,
    run_branched_forward_pass,
    ensure_branched_after_eval as ensure_branched_after_eval_helper,
)
from .model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .e13_contract import (
    assert_trainable_contract as assert_e13_trainable_contract,
    get_state_dict as get_e13_state_dict,
    initialise_e13_contract,
    load_state_dict as load_e13_state_dict,
    optimizer_groups as e13_optimizer_groups,
)
##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####

### PhotomakerLora upgraged for BA ###
class PhotomakerBranchedLora(SDXL):
    """
    PhotoMaker LoRA model that trains with the branched-attention modifications.
    The implementation mirrors ``PhotomakerLora`` but swaps in the upgraded ID
    encoder and routes the UNet forward through the branched predictor so that
    LoRA weights observe the same architecture used at inference time.
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        photomaker_path,
        rank,
        weight_dtype,
        device,
        init_lora_weights,
        lora_modules,
        target_size: int = 1024,
        trigger_word: str = "img",
        ##### BRANCHED ATTENTION - NEW PARAMS 1 #####
        photomaker_lora_rank: int = 64,
        pose_adapt_ratio: float = 0.25, 
        ca_mixing_for_face: bool = True,  
        face_embed_strategy: str = "face", 
        train_branch_mode: str = "both",   # 'both' or 'ref_only' for BranchedAttnProcessor
        train_ba_only: bool = False,       # 28 Nov: optionally train only branched-attn layers
        ba_weights_split: bool = False,    # optionally enable per-branch BA-specific adapters
        use_attn_v2: bool = True,          # toggle between attn_processor2 (v2) and attn_processor (legacy)
        branched_attn_weight_mode: str = "shared",
        branched_attn_new_weight_kind: str = "full",
        train_branched_ca_lora: bool = True,
        ba_train_top_k: float = 1.0,
        ba_patch_top_k: float = 1.0,
        non_ba_train: bool = False,
        train_ba_all_steps: bool = False,
        id_alpha: float = 0.3,             # strength of ID embedding injection in BranchedAttnProcessor
        use_id_embeds: bool = True,        # toggle ID embedding injection (controls id_to_hidden usage)
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        branched_attn_start_step: int = 15,
        num_inference_steps: int = 50,
        # 10 Aug 2026 - E13C-CORE-01: One explicit switch activates the clean,
        # fail-closed E13 contract; June configurations retain their defaults.
        e13_family_contract: bool = False,
        ba_hard_v1_lora_rank: int = 128,
        generic_adapter_train_scope: str = "effective_all",
        photomaker_default_train_scope: str = "effective_all",
        strict_branched_install: bool = True,
        strict_trainable_contract: bool = True,
        branched_state_dict_mode: str = "trainable_unet_v2",
        ba_training_mask_feather: int = 0,
        conditioning_cache_enabled: bool = False,
        skip_unused_text_conditioning: bool = False,
        batched_conditioning_preparation: bool = True,
        cache_prepared_masks: bool = True,
        compute_branch_debug_outputs: bool = False,
        ba_hardcase_mode: str = "off",
        ba_hardcase_groups: Optional[Sequence[str]] = None,
        ba_hardcase_transition_cells: int = 2,
        ba_crossview_consistency_enabled: bool = False,
        ba_crossview_consistency_probability: float = 0.25,
        ba_crossview_consistency_weight: float = 0.05,
        ##### BRANCHED ATTENTION - NEW PARAMS 1 #####
    ):
        """NEW PARAMS 1: define BA training controls (strategy, processor variant, ID mixing, and BA-only toggles)."""
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_dtype=weight_dtype,
            device=device,
        )
        self.lora_rank = rank
        self.branched_attn_lora_rank = int(rank)
        self.init_lora_weights = init_lora_weights
        self.lora_modules = lora_modules

        self.id_image_processor = CLIPImageProcessor()
        
        
        
        
        ####  PhotoMaker v2 integration START: upgraded ID encoder & face embeddings ---
        
        # Mirror the PhotoMaker v2 ID encoder configuration (512-d InsightFace input).
        self.id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        self.target_size = target_size


        ##### BRANCHED ATTENTION - NEW PARAMS 2 #####
        """NEW PARAMS 2: initialize BA sizing helpers used for mask resolution and reference preprocessing."""
        # self.feature_extractor = self.id_image_processor  # --- MODIFIED For training integration ---
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae.config, "block_out_channels") else 8
        ##### BRANCHED ATTENTION - NEW PARAMS 2 #####
        

        ### FIX FOR OOM ERROR ###
        # Pin ONNXRuntime CUDA provider to the per-rank GPU; otherwise multiple ranks may load on GPU:0 and OOM.
        _device_id = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else 0
        faceanalysis_default = "0" if e13_family_contract else "1"
        FACEANALYSIS_CPU = os.environ.get(
            "FACEANALYSIS_CPU", faceanalysis_default
        ).lower() not in {"0", "false", "no"}
        if e13_family_contract and FACEANALYSIS_CPU:
            raise RuntimeError("E13C-PERF-03 requires FACEANALYSIS_CPU=0")
        
        # Instantiate FaceAnalysis once for extracting 512-D identity embeddings.
        if FACEANALYSIS_CPU:
            self.face_analyzer = create_face_analyzer(
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
                ctx_id=-1,
                det_size=(640, 640),
                fallback_ctx_id=-1,
                quiet=True,
            )
        else:
            ctx_id = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else -1
            self.face_analyzer = create_face_analyzer(
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"device_id": _device_id}, {}],
                allowed_modules=["detection", "recognition"],
                ctx_id=ctx_id,
                det_size=(640, 640),
                fallback_ctx_id=-1,
                quiet=True,
            )
        if e13_family_contract:
            # 10 Aug 2026 - E13C-PERF-03: Cosmic conditioning regressed from
            # ~2 s/step to 5-7 s/step when ORT fell back to CPU. Check every
            # loaded InsightFace session and fail before training.
            import onnxruntime as ort

            if ort.__version__ != "1.20.1":
                raise RuntimeError(
                    "Clean E13 runs require onnxruntime-gpu==1.20.1; "
                    f"found {ort.__version__}"
                )
            sessions = [
                getattr(component, "session", None)
                for component in getattr(self.face_analyzer, "models", {}).values()
            ]
            providers = [
                session.get_providers() for session in sessions if session is not None
            ]
            if not providers or any(
                "CUDAExecutionProvider" not in active for active in providers
            ):
                raise RuntimeError(
                    "InsightFace ONNX sessions did not activate CUDAExecutionProvider: "
                    f"{providers}"
                )
        ### FIX FOR OOM ERROR ###
         
        ####  PhotoMaker v2 integration END: upgraded ID encoder & face embeddings ---
        
        self.trigger_word = trigger_word ### "img" hardcoded by default
        self.num_tokens = self.id_encoder.num_tokens ### 1 hardcoded by default
        added_tokens_1 = self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        added_tokens_2 = self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)
        if added_tokens_1:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        if added_tokens_2:
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))

        ##### BRANCHED ATTENTION - NEW PARAMS 3 #####
        """NEW PARAMS 3: persist runtime BA knobs on the model so patched processors can read them."""
        # Branched helpers expect ``scheduler`` attribute – alias it once.
        self.scheduler = self.noise_scheduler
        self.pose_adapt_ratio = float(pose_adapt_ratio) # --- ADDED For training integration
        self.ca_mixing_for_face = bool(ca_mixing_for_face) # --- ADDED For training integration
        self.face_embed_strategy = (face_embed_strategy or "face").lower() # --- ADDED For training integration
        self.train_branch_mode = (train_branch_mode or "both").lower()
        self.photomaker_start_step = int(photomaker_start_step)
        self.merge_start_step = int(merge_start_step)
        self.branched_attn_start_step = int(branched_attn_start_step)
        self.num_inference_steps = int(num_inference_steps)
        # ID embedding mixing strength for branched self-attention
        self.id_alpha = float(id_alpha)
        # Global on/off switch for BranchedAttnProcessor.id_to_hidden usage
        self.use_id_embeds = bool(use_id_embeds)
        self.train_ba_only = bool(train_ba_only)
        ### 28 Nov: train only BA layers ###
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        self.ba_weights_split = bool(ba_weights_split) # Clean separataion of BA-specific parameters
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        # Select which branched attention processor implementation to use at train time.
        self.use_attn_v2 = bool(use_attn_v2)
        self.branched_attn_weight_mode = (branched_attn_weight_mode or "shared").lower()
        self.branched_attn_new_weight_kind = (branched_attn_new_weight_kind or "full").lower()
        self.train_branched_ca_lora = bool(train_branched_ca_lora)
        self.ba_train_top_k = float(ba_train_top_k)
        self.ba_patch_top_k = float(ba_patch_top_k)
        self.non_ba_train = bool(non_ba_train)
        self.train_ba_all_steps = bool(train_ba_all_steps)
        self.ba_hardcase_mode = str(ba_hardcase_mode or "off").lower()
        self.ba_hardcase_groups = tuple(
            str(group) for group in (ba_hardcase_groups or ())
        )
        self.ba_hardcase_transition_cells = int(ba_hardcase_transition_cells)
        self.ba_crossview_consistency_enabled = bool(
            ba_crossview_consistency_enabled
        )
        self.ba_crossview_consistency_probability = float(
            ba_crossview_consistency_probability
        )
        self.ba_crossview_consistency_weight = float(
            ba_crossview_consistency_weight
        )
        self.e13_family_contract = bool(e13_family_contract)
        if self.e13_family_contract:
            required = {
                "train_ba_only": self.train_ba_only,
                "noise_and_ref": self.branched_attn_weight_mode == "noise_and_ref",
                "lora_weights": self.branched_attn_new_weight_kind == "lora",
                "self_attention_only": not self.train_branched_ca_lora,
                "all_steps": self.train_ba_all_steps,
                "pose_reference_only": self.pose_adapt_ratio == 0.0,
                "no_face_ca_mix": not self.ca_mixing_for_face,
            }
            failed = [name for name, valid in required.items() if not valid]
            if failed:
                raise ValueError(f"Invalid clean E13 contract settings: {failed}")
            initialise_e13_contract(
                self,
                ba_hard_v1_lora_rank=ba_hard_v1_lora_rank,
                generic_adapter_train_scope=generic_adapter_train_scope,
                photomaker_default_train_scope=photomaker_default_train_scope,
                strict_branched_install=strict_branched_install,
                strict_trainable_contract=strict_trainable_contract,
                branched_state_dict_mode=branched_state_dict_mode,
                ba_training_mask_feather=ba_training_mask_feather,
                conditioning_cache_enabled=conditioning_cache_enabled,
                skip_unused_text_conditioning=skip_unused_text_conditioning,
                batched_conditioning_preparation=batched_conditioning_preparation,
                cache_prepared_masks=cache_prepared_masks,
                compute_branch_debug_outputs=compute_branch_debug_outputs,
                ba_hardcase_mode=ba_hardcase_mode,
                ba_hardcase_groups=ba_hardcase_groups,
                ba_hardcase_transition_cells=ba_hardcase_transition_cells,
                ba_crossview_consistency_enabled=ba_crossview_consistency_enabled,
                ba_crossview_consistency_probability=(
                    ba_crossview_consistency_probability
                ),
                ba_crossview_consistency_weight=ba_crossview_consistency_weight,
            )
        ##### BRANCHED ATTENTION - NEW PARAMS 3 #####

        photomaker_lora_config = LoraConfig(
            r=photomaker_lora_rank, ### 64 by default
            lora_alpha=photomaker_lora_rank, ### 64 by default
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(photomaker_lora_config)

        photomaker_path = resolve_photomaker_path(photomaker_path, version="v2")
        photomaker_state_dict = torch.load(photomaker_path, map_location="cpu")
        self.load_photomaker_state_dict_(photomaker_state_dict)

    def prepare_for_training(self):
        super().prepare_for_training()
        self.unet.requires_grad_(False)
        self.id_encoder.to(dtype=self.weight_dtype)
        self.id_encoder.requires_grad_(False)

        adapter_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.lora_modules,
        )
        self.unet.add_adapter(adapter_lora_config, adapter_name="lora_adapter")
        self.unet.set_adapter(["lora_adapter", "default"])


        ##### BRANCHED ATTENTION - NEW BLOCK 1 #####
        """NEW BLOCK 1: pre-install branched attention processors before optimizer creation and mark their params trainable."""
        install_branched_processors_for_training(self)

        ##### BRANCHED ATTENTION - NEW BLOCK 1 #####


    def load_photomaker_state_dict_(self, state_dict):
        # load lora
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="default")
        
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert not unexpected_keys, unexpected_keys

        # load id_encoder
        self.id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)

    def get_trainable_params(self, config):

        if self.e13_family_contract:
            # 10 Aug 2026 - E13C-CORE-03: Return named, disjoint optimizer
            # groups so the audited ownership split is visible in logs.
            return e13_optimizer_groups(self, config)

        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####
        """NEW BLOCK 2: optional custom optimizer grouping for branched processor parameters and BA-related LoRA params."""
        # ### TRAIN_BA_ONLY - CHECK ###
        # if getattr(self, "train_ba_only", False):
        #     # Train branched attention processors + LoRA weights on attention projections.
        #     proc_params = []
        #     lora_params = []
        #     for name, p in self.unet.named_parameters():
        #         if not p.requires_grad:
        #             continue
        #         if ".attn1.processor." in name or ".attn2.processor." in name:
        #             proc_params.append(p)
        #         elif "lora_A" in name or "lora_B" in name:
        #             lora_params.append(p)

        #     param_groups = []
        #     if proc_params:
        #         param_groups.append(
        #             {"params": proc_params, "lr": config.lr_for_lora, "name": "branched_processors"}
        #         )
        #     if lora_params:
        #         param_groups.append(
        #             {"params": lora_params, "lr": config.lr_for_lora, "name": "branched_lora"}
        #         )
        #     return param_groups
        # ### TRAIN_BA_ONLY - CHECK ###
        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####

        # Default behavior: train all UNet parameters with requires_grad=True (LoRA + processors).
        lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {"params": lora_params, "lr": config.lr_for_lora, "name": "lora_params"},
        ]
        return trainable_params

    def get_state_dict(self):
        if self.e13_family_contract:
            return get_e13_state_dict(self)
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        state = {
            'lora_weights': lora_weights,
        }
        if hasattr(self.unet, "attn_processors"):
            proc_sd = {}
            patched_proc_names = set(getattr(self, "_ba_patched_processor_names", ()))
            for name, proc in self.unet.attn_processors.items():
                if patched_proc_names and name not in patched_proc_names:
                    continue
                if not isinstance(proc, torch.nn.Module):
                    continue
                trainable = tuple(n for n, p in proc.named_parameters() if p.requires_grad)
                if not trainable:
                    continue
                full_sd = proc.state_dict()
                sd = {
                    k: v for k, v in full_sd.items()
                    if any(k == n or k.startswith(n + ".") for n in trainable)
                }
                if sd:
                    proc_sd[name] = sd
            if proc_sd:
                state["attn_processors"] = proc_sd
        return state

    def load_state_dict_(self, state_dict):
        if self.e13_family_contract and int(state_dict.get("schema_version", 1)) == 2:
            load_e13_state_dict(self, state_dict)
            return
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # In newer peft versions this is an empty list when there are no unexpected keys
            assert not unexpected_keys, unexpected_keys
        for name, sd in state_dict.get("attn_processors", {}).items():
            proc = self.unet.attn_processors.get(name)
            if proc is not None and hasattr(proc, "load_state_dict"):
                proc.load_state_dict(sd, strict=False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
        face_bbox_ref: Sequence[Sequence[float]] | None = None,
        spatial_ref_images_alt: Sequence[Sequence[Image.Image]] | None = None,
        face_bbox_ref_alt: Sequence[Sequence[float]] | None = None,
        do_cfg: bool = False,
        *args,
        **kwargs,
    ):
        del do_cfg  # classifier-free guidance is not used during training

        pixel_values = pixel_values.to(self.device, self.vae.dtype)
        with torch.no_grad(): ### TO CHECK - torch.no_grad() only here now
            latents = self.vae.encode(pixel_values).latent_dist.sample() ### latents are caled model_input in lora v1
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Match the current inference schedule at batch level:
        # NO_ID (0-9), PHOTOMAKER (10-14), BOTH (15-49)
        t_scalar = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (1,),
            device=latents.device,
        ).long()
        timesteps = t_scalar.repeat(batch_size)
        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(orig_size, crop) for orig_size, crop in zip(original_sizes, crop_top_lefts)]
        )

        ##### BRANCHED ATTENTION - NEW BLOCK 4 #####
        """NEW BLOCK 4: delegate BA sample preparation (masks, refs, embeddings) to helper utilities."""
        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
            face_prompt_embeds,
            id_features,
            mask4,
            mask4_ref,
            reference_latents,
        ) = prepare_branched_training_inputs(
            self,
            prompts=prompts,
            ref_images=ref_images,
            face_bbox=face_bbox,
            face_bbox_ref=face_bbox_ref,
            pixel_values=pixel_values,
            noisy_latents=noisy_latents,
        )
        ##### BRANCHED ATTENTION - NEW BLOCK 4 #####

        ##### BRANCHED ATTENTION - NEW BLOCK 5 #####
        """NEW BLOCK 5 is implemented in `lora2_helpers.prepare_branched_training_inputs`."""
        ##### BRANCHED ATTENTION - NEW BLOCK 5 #####

        ##### BRANCHED ATTENTION - NEW BLOCK 6 #####
        """NEW BLOCK 6 is implemented in `lora2_helpers.prepare_branched_training_inputs`."""
        ##### BRANCHED ATTENTION - NEW BLOCK 6 #####

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids.to(device=self.device, dtype=self.unet.dtype),
        }

        # 10 Aug 2026 - E13C-PERF-02: E13 routes every sampled timestep through
        # BA, so text-only embeddings and the scalar GPU-to-host timestep sync
        # are unreachable work. June mode keeps the original path unchanged.
        skip_text_only = self.train_ba_all_steps and bool(
            getattr(self, "skip_unused_text_conditioning", False)
        )
        if not skip_text_only:
            num_inference_steps = max(1, self.num_inference_steps)
            photomaker_start_ratio = float(self.photomaker_start_step) / float(
                num_inference_steps
            )
            branched_start_ratio = float(self.branched_attn_start_step) / float(
                num_inference_steps
            )
            denoise_progress = 1.0 - (
                float(t_scalar.item())
                / float(self.noise_scheduler.config.num_train_timesteps - 1)
            )

            text_only_prompts = []
            trigger_word_pattern = re.compile(
                rf"\b{re.escape(self.trigger_word)}\b", flags=re.IGNORECASE
            )
            for prompt in prompts:
                text_only_prompt = trigger_word_pattern.sub(" ", prompt)
                text_only_prompts.append(" ".join(text_only_prompt.split()))

            (
                prompt_embeds_text_only,
                pooled_prompt_embeds_text_only,
            ) = self.encode_prompt(prompt=text_only_prompts, do_cfg=False)
            prompt_embeds_text_only = prompt_embeds_text_only.to(
                device=self.device, dtype=self.unet.dtype
            )
            pooled_prompt_embeds_text_only = pooled_prompt_embeds_text_only.to(
                device=self.device, dtype=self.unet.dtype
            )

        ### MEMO: INITIAL LORA UNet pass ###
        # model_pred = self.unet(
        #     noisy_model_input,
        #     timesteps,
        #     encoder_hidden_states=prompt_embeds,
        #     added_cond_kwargs=added_cond_kwargs,
        #     return_dict=False,
        # )[0]
        ### MEMO: INITIAL LORA UNet pass ###

        if self.train_ba_all_steps:
            noise_pred = run_branched_forward_pass(
                self,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                mask4=mask4,
                mask4_ref=mask4_ref,
                reference_latents=reference_latents,
                face_prompt_embeds=face_prompt_embeds,
                class_tokens_mask=class_tokens_mask,
                id_features=id_features,
            )
        elif denoise_progress < photomaker_start_ratio:
            text_only_kwargs = {
                "text_embeds": pooled_prompt_embeds_text_only,
                "time_ids": add_time_ids.to(device=self.device, dtype=self.unet.dtype),
            }
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds_text_only,
                added_cond_kwargs=text_only_kwargs,
                return_dict=False,
            )[0]
        elif denoise_progress < branched_start_ratio:
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        else:
            ##### BRANCHED ATTENTION - FORWARD PASS #####
            """FORWARD PASS: run branched prediction via helper wrapper around `two_branch_predict`."""
            noise_pred = run_branched_forward_pass(
                self,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                mask4=mask4,
                mask4_ref=mask4_ref,
                reference_latents=reference_latents,
                face_prompt_embeds=face_prompt_embeds,
                class_tokens_mask=class_tokens_mask,
                id_features=id_features,
            )
            ##### BRANCHED ATTENTION - FORWARD PASS #####

        crossview_loss = noise_pred.float().new_tensor(0.0)
        if (
            self.training
            and self.ba_crossview_consistency_enabled
            and spatial_ref_images_alt is not None
            and face_bbox_ref_alt is not None
            and torch.rand((), device=latents.device).item()
            < self.ba_crossview_consistency_probability
        ):
            # 12 Aug 2026 - CL18 changes only the spatial reference view. The
            # target/noise, prompt tokens and paired reference noise stay fixed.
            alternate_refs = []
            alternate_masks = []
            for refs, bbox in zip(spatial_ref_images_alt, face_bbox_ref_alt):
                refs = refs if isinstance(refs, (list, tuple)) else [refs]
                if not refs:
                    raise RuntimeError("Cross-view consistency received no alternate ref")
                ref = refs[0]
                alternate_refs.append(ref)
                ref_size = (
                    tuple(ref.shape[-2:])
                    if isinstance(ref, torch.Tensor)
                    else (ref.height, ref.width)
                )
                alternate_masks.append(
                    self._bbox_to_ref_mask(
                        bbox,
                        noisy_latents.shape[-2:],
                        ref_size,
                    )
                )
            alternate_latents = self._encode_reference_latents(
                alternate_refs,
                target_shape=noisy_latents.shape[-2:],
            ).to(dtype=noisy_latents.dtype)
            alternate_mask4 = torch.cat(alternate_masks, dim=0).to(
                device=self.device,
                dtype=noisy_latents.dtype,
            )
            paired_reference_noise = getattr(self, "_ref_noise", None)
            if paired_reference_noise is None:
                raise RuntimeError("Cross-view consistency lost paired reference noise")
            student_pred = run_branched_forward_pass(
                self,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                mask4=mask4,
                mask4_ref=alternate_mask4,
                reference_latents=alternate_latents,
                face_prompt_embeds=face_prompt_embeds,
                class_tokens_mask=class_tokens_mask,
                id_features=id_features,
                reference_noise=paired_reference_noise,
            )
            face = mask4.float()
            if face.shape[-2:] != noise_pred.shape[-2:]:
                face = F.interpolate(face, size=noise_pred.shape[-2:], mode="nearest")
            teacher_face = noise_pred.detach().float() * face
            student_face = student_pred.float() * face
            smooth_map = F.smooth_l1_loss(
                student_face,
                teacher_face,
                reduction="none",
            )
            smooth = (smooth_map * face).sum() / (
                face.sum() * student_face.shape[1]
            ).clamp_min(1.0)
            cosine = F.cosine_similarity(
                student_face.flatten(1),
                teacher_face.flatten(1),
                dim=1,
            ).mean()
            crossview_loss = smooth + 0.10 * (1.0 - cosine)

        ba_aux_loss = self.ba_crossview_consistency_weight * crossview_loss
        return {
            'model_pred': noise_pred,
            'target': noise,
            'ba_aux_loss': ba_aux_loss,
            'ba_ownership_loss': noise_pred.float().new_tensor(0.0),
            'ba_crossview_loss': crossview_loss.detach(),
        }

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        do_cfg: bool = False,
    ):
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt = prompt if not do_cfg else ""

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                image_token_id = tokenizer.convert_tokens_to_ids(self.trigger_word)
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                self._validate_text_input_ids(
                    text_input_ids,
                    text_encoder,
                    prompt,
                    f"{text_encoder.__class__.__name__} (trigger path)",
                )

                if not do_cfg:
                    clean_index = 0
                    clean_input_ids = []
                    class_token_index = []
                    # Find out the corresponding class word token based on the newly added trigger word token
                    for i, token_id in enumerate(text_input_ids.tolist()[0]):
                        if token_id == image_token_id:
                            class_token_index.append(clean_index - 1)
                        else:
                            clean_input_ids.append(token_id)
                            clean_index += 1

                    if len(class_token_index) != 1:
                        raise ValueError(
                            f"PhotoMaker currently does not support multiple trigger words in a single prompt. "
                            f"Trigger word: {self.trigger_word}, Prompt: {prompt}."
                        )
                    class_token_index = class_token_index[0]

                    # Expand the class word token and corresponding mask
                    class_token = clean_input_ids[class_token_index]
                    clean_input_ids = (
                        clean_input_ids[:class_token_index]
                        + [class_token] * num_id_images * self.num_tokens
                        + clean_input_ids[class_token_index + 1 :]
                    )

                    # Truncation or padding
                    max_len = tokenizer.model_max_length
                    if len(clean_input_ids) > max_len:
                        clean_input_ids = clean_input_ids[:max_len]
                    else:
                        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (max_len - len(clean_input_ids))

                    class_tokens_mask = [
                        class_token_index <= i < class_token_index + (num_id_images * self.num_tokens)
                        for i in range(len(clean_input_ids))
                    ]

                    text_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                    class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)
                    class_tokens_mask = class_tokens_mask.to(self.device)

                prompt_embeds_curr = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
                
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds_curr[0]
                prompt_embeds_curr = prompt_embeds_curr.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds_curr)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(self.device)

        bs_embed, _, _ = prompt_embeds.shape
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    ##### BRANCHED ATTENTION - HELPER UTILS #####
    """HELPER UTILS: utilities for bbox-to-latent masks, reference-latent encoding, and branched re-patching after eval."""
    # 10 Aug 2026 - E13C-PERF-01: Encode the frozen prompt batch in one pass;
    # token cleanup and trigger masks are identical to the scalar path.
    def encode_prompts_with_trigger_word(
        self,
        prompts: Sequence[str],
        *,
        num_id_images: int = 1,
    ):
        """Encode a one-trigger PhotoMaker prompt batch in one encoder pass."""
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts must not be empty")

        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)
        tokenizers = (
            [self.tokenizer, self.tokenizer_2]
            if self.tokenizer is not None
            else [self.tokenizer_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )

        prompt_embeds_list = []
        class_tokens_mask = None
        pooled_prompt_embeds = None
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            cleaned_rows = []
            mask_rows = []
            for prompt, token_ids in zip(prompts, text_inputs.input_ids.tolist()):
                clean_input_ids = []
                class_token_indices = []
                for token_id in token_ids:
                    if token_id == image_token_id:
                        class_token_indices.append(len(clean_input_ids) - 1)
                    else:
                        clean_input_ids.append(token_id)

                if len(class_token_indices) != 1:
                    raise ValueError(
                        "PhotoMaker currently requires exactly one trigger word "
                        f"per prompt. Trigger word: {self.trigger_word}, "
                        f"Prompt: {prompt}."
                    )
                class_token_index = class_token_indices[0]
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = (
                    clean_input_ids[:class_token_index]
                    + [class_token] * num_id_images * self.num_tokens
                    + clean_input_ids[class_token_index + 1 :]
                )

                max_len = tokenizer.model_max_length
                clean_input_ids = clean_input_ids[:max_len]
                clean_input_ids += [tokenizer.pad_token_id] * (
                    max_len - len(clean_input_ids)
                )
                cleaned_rows.append(clean_input_ids)
                mask_rows.append(
                    [
                        class_token_index
                        <= index
                        < class_token_index + (num_id_images * self.num_tokens)
                        for index in range(max_len)
                    ]
                )

            text_input_ids = torch.tensor(
                cleaned_rows, dtype=torch.long, device=self.device
            )
            class_tokens_mask = torch.tensor(
                mask_rows, dtype=torch.bool, device=self.device
            )
            prompt_embeds_curr = text_encoder(
                text_input_ids, output_hidden_states=True
            )
            pooled_prompt_embeds = prompt_embeds_curr[0]
            prompt_embeds_list.append(prompt_embeds_curr.hidden_states[-2])

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1).to(self.device)
        pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompts), -1)
        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    ##### BRANCHED ATTENTION - HELPER UTILS #####
    """HELPER UTILS: utilities for bbox-to-latent masks, reference-latent encoding, and branched re-patching after eval."""

    def _bbox_to_ref_mask(
        self,
        bbox: Optional[Sequence[float]],
        latent_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        mask = torch.zeros(1, 1, self.target_size, self.target_size, device=self.device)
        if bbox is None or len(bbox) < 4:
            mask.fill_(1.0)
        else:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            if x1 <= x0 or y1 <= y0:
                mask.fill_(1.0)
            else:
                image_h, image_w = image_shape
                scale = min(
                    self.target_size / max(image_w, 1),
                    self.target_size / max(image_h, 1),
                )
                resized_w = max(8, int(round(image_w * scale)) // 8 * 8)
                resized_h = max(8, int(round(image_h * scale)) // 8 * 8)
                pad_left = (self.target_size - resized_w) // 2
                pad_top = (self.target_size - resized_h) // 2

                scale_w = resized_w / max(image_w, 1)
                scale_h = resized_h / max(image_h, 1)

                x_start = max(0, min(self.target_size, int(round(x0 * scale_w + pad_left))))
                x_end = max(0, min(self.target_size, int(round(x1 * scale_w + pad_left))))
                y_start = max(0, min(self.target_size, int(round(y0 * scale_h + pad_top))))
                y_end = max(0, min(self.target_size, int(round(y1 * scale_h + pad_top))))

                if x_end <= x_start or y_end <= y_start:
                    mask.fill_(1.0)
                else:
                    mask[:, :, y_start:y_end, x_start:x_end] = 1.0

        if mask.shape[-2:] != latent_shape:
            mask = F.interpolate(mask, size=latent_shape, mode="nearest")
        return mask

    def _bbox_to_mask(
        self,
        bbox: Optional[Sequence[float]],
        latent_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        mask = torch.zeros(1, 1, latent_shape[0], latent_shape[1], device=self.device)
        if bbox is None or len(bbox) < 4:
            mask.fill_(1.0)
            return mask

        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            mask.fill_(1.0)
            return mask

        scale_w = latent_shape[1] / max(image_shape[1], 1)
        scale_h = latent_shape[0] / max(image_shape[0], 1)

        x_start = max(0, min(latent_shape[1], int(round(x0 * scale_w))))
        x_end = max(0, min(latent_shape[1], int(round(x1 * scale_w))))
        y_start = max(0, min(latent_shape[0], int(round(y0 * scale_h))))
        y_end = max(0, min(latent_shape[0], int(round(y1 * scale_h))))

        if x_end <= x_start or y_end <= y_start:
            mask.fill_(1.0)
            return mask

        mask[:, :, y_start:y_end, x_start:x_end] = 1.0
        feather = int(getattr(self, "ba_training_mask_feather", 0))
        if feather > 0:
            # 10 Aug 2026 - E13C-CORE-06: CL14 feathers only the target mask
            # used by training. Reference masks and inference masks remain
            # unchanged, preserving the historical CL14 generation path.
            for step in range(1, feather + 1):
                weight = step / float(feather + 1)
                ys, ye = y_start + step - 1, y_end - step + 1
                xs, xe = x_start + step - 1, x_end - step + 1
                if ye <= ys or xe <= xs:
                    break
                mask[:, :, ys, xs:xe] = weight
                mask[:, :, ye - 1, xs:xe] = weight
                mask[:, :, ys:ye, xs] = weight
                mask[:, :, ys:ye, xe - 1] = weight
        return mask

    def _encode_reference_latent(
        self,
        ref_image,
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        if isinstance(ref_image, torch.Tensor):
            ref_tensor = ref_image.clone().detach()
            if ref_tensor.dim() == 3:
                ref_tensor = ref_tensor.unsqueeze(0)
            if ref_tensor.shape[-2:] != target_shape:
                ref_tensor = F.interpolate(ref_tensor, size=target_shape, mode="bilinear", align_corners=False)
            ref_tensor = ref_tensor.to(device=self.device, dtype=self.vae.dtype)
        else:
            if not isinstance(ref_image, Image.Image):
                raise TypeError(f"Unsupported reference image type: {type(ref_image)}")
            ow, oh = ref_image.size
            scale = min(self.target_size / ow, self.target_size / oh)
            rw = max(8, int(round(ow * scale)) // 8 * 8)
            rh = max(8, int(round(oh * scale)) // 8 * 8)
            pl = (self.target_size - rw) // 2
            pr = self.target_size - rw - pl
            pt = (self.target_size - rh) // 2
            pb = self.target_size - rh - pt
            ref_resized = ref_image.resize((rw, rh), Image.BILINEAR)
            ref_np = np.array(ref_resized).astype(np.float32) / 255.0
            ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
            ref_tensor = (ref_tensor - 0.5) / 0.5
            ref_tensor = F.pad(ref_tensor, (pl, pr, pt, pb), value=0.0)
            ref_tensor = ref_tensor.to(device=self.device, dtype=self.vae.dtype)

        with torch.no_grad():
            latents = self.vae.encode(ref_tensor).latent_dist.mode() 
        latents = latents * self.vae.config.scaling_factor

        if latents.shape[-2:] != target_shape:
            latents = F.interpolate(latents, size=target_shape, mode="bilinear", align_corners=False)

        return latents

    # 10 Aug 2026 - E13C-PERF-01: Encode equal-sized references in one frozen
    # VAE pass; image resize, padding, dtype and latent scaling remain unchanged.
    def _encode_reference_latents(
        self,
        ref_images: Sequence[Image.Image],
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Encode an equal-sized reference batch with the frozen VAE."""
        ref_tensors = []
        for ref_image in ref_images:
            if not isinstance(ref_image, Image.Image):
                raise TypeError(
                    "Batched conditioning currently requires PIL references, "
                    f"got {type(ref_image)}"
                )
            ow, oh = ref_image.size
            scale = min(self.target_size / ow, self.target_size / oh)
            rw = max(8, int(round(ow * scale)) // 8 * 8)
            rh = max(8, int(round(oh * scale)) // 8 * 8)
            pl = (self.target_size - rw) // 2
            pr = self.target_size - rw - pl
            pt = (self.target_size - rh) // 2
            pb = self.target_size - rh - pt
            ref_resized = ref_image.resize((rw, rh), Image.BILINEAR)
            ref_np = np.array(ref_resized).astype(np.float32) / 255.0
            ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1)
            ref_tensor = (ref_tensor - 0.5) / 0.5
            ref_tensors.append(F.pad(ref_tensor, (pl, pr, pt, pb), value=0.0))

        ref_batch = torch.stack(ref_tensors).to(
            device=self.device, dtype=self.vae.dtype
        )
        with torch.no_grad():
            latents = self.vae.encode(ref_batch).latent_dist.mode()
        latents = latents * self.vae.config.scaling_factor
        if latents.shape[-2:] != target_shape:
            latents = F.interpolate(
                latents,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
        return latents


    def ensure_branched_after_eval(self):
        ensure_branched_after_eval_helper(self)

    def assert_trainable_contract(self, optimizer=None):
        if not self.e13_family_contract:
            return {}
        return assert_e13_trainable_contract(self, optimizer=optimizer)
    ##### BRANCHED ATTENTION - HELPER UTILS #####
