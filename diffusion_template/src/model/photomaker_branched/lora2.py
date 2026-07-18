from __future__ import annotations

import hashlib
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
        ba_correctness_guards: bool = False,
        ba_invalid_sample_policy: str = "legacy",
        ba_strict_processor_restore: bool = False,
        ba_train_timestep_mode: str = "all",
        ba_face_prompt_attention_mask: bool = False,
        ba_sa_train_mode: str = "all",
        ba_sa_ref_token_mode: str = "full_grid",
        ba_sa_face_mode: str = "reference",
        ba_sa_ref_layer_scope: str = "all",
        ba_sa_roi_grid_size: int = 8,
        ba_sa_core_ratio: float = 0.7,
        ba_sa_mix_init: float = 0.25,
        id_alpha: float = 0.3,             # strength of ID embedding injection in BranchedAttnProcessor
        use_id_embeds: bool = True,        # toggle ID embedding injection (controls id_to_hidden usage)
        ba_uncond_face_fix: bool = False,  # F1: keep plain negative prompt for the uncond face branch under CFG
        ba_face_prompt_mode: str = "id_only",  # B1: face-branch prompt: id_only (legacy) | full_boosted
        use_id_loss: bool = False,
        id_loss_weight: float = 0.1,
        id_loss_max_timestep: int = 400,
        id_loss_face_size: int = 160,
        id_loss_identity_source: str = "reference",
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        branched_attn_start_step: int = 15,
        num_inference_steps: int = 50,
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
        FACEANALYSIS_CPU = os.environ.get("FACEANALYSIS_CPU", "1").lower() not in {"0", "false", "no"}
        
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
        # F1: uncond face prompt handling under CFG (see branched_runtime.two_branch_predict)
        self.ba_uncond_face_fix = bool(ba_uncond_face_fix)
        # B1: face-branch prompt construction mode (see branched_runtime.two_branch_predict)
        self.ba_face_prompt_mode = str(ba_face_prompt_mode or "id_only").lower()
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
        self.ba_correctness_guards = bool(ba_correctness_guards)
        self.ba_invalid_sample_policy = str(ba_invalid_sample_policy or "legacy").lower()
        self.ba_strict_processor_restore = bool(ba_strict_processor_restore)
        self.ba_train_timestep_mode = str(ba_train_timestep_mode or "all").lower()
        self.ba_face_prompt_attention_mask = bool(ba_face_prompt_attention_mask)
        self.ba_sa_train_mode = str(ba_sa_train_mode or "all").lower()
        self.ba_sa_ref_token_mode = str(ba_sa_ref_token_mode or "full_grid").lower()
        self.ba_sa_face_mode = str(ba_sa_face_mode or "reference").lower()
        self.ba_sa_ref_layer_scope = str(ba_sa_ref_layer_scope or "all").lower()
        self.ba_sa_roi_grid_size = int(ba_sa_roi_grid_size)
        self.ba_sa_core_ratio = float(ba_sa_core_ratio)
        self.ba_sa_mix_init = float(ba_sa_mix_init)
        if self.ba_invalid_sample_policy not in {"legacy", "error", "skip_batch"}:
            raise ValueError(f"Unknown ba_invalid_sample_policy: {self.ba_invalid_sample_policy}")
        if self.ba_train_timestep_mode not in {"all", "inference_ba_region"}:
            raise ValueError(f"Unknown ba_train_timestep_mode: {self.ba_train_timestep_mode}")
        if self.ba_sa_train_mode not in {"all", "ref_kv_only"}:
            raise ValueError(f"Unknown ba_sa_train_mode: {self.ba_sa_train_mode}")
        if self.ba_sa_ref_token_mode not in {"full_grid", "roi"}:
            raise ValueError(f"Unknown ba_sa_ref_token_mode: {self.ba_sa_ref_token_mode}")
        if self.ba_sa_face_mode not in {
            "reference",
            "dual",
            "core_ring",
            "confidence_residual",
        }:
            raise ValueError(f"Unknown ba_sa_face_mode: {self.ba_sa_face_mode}")
        if self.ba_sa_ref_layer_scope not in {"all", "up"}:
            raise ValueError(f"Unknown ba_sa_ref_layer_scope: {self.ba_sa_ref_layer_scope}")
        if self.ba_sa_roi_grid_size <= 0:
            raise ValueError("ba_sa_roi_grid_size must be positive")
        if not 0.0 < self.ba_sa_core_ratio <= 1.0:
            raise ValueError("ba_sa_core_ratio must be in (0, 1]")
        if not 0.0 < self.ba_sa_mix_init < 1.0:
            raise ValueError("ba_sa_mix_init must be in (0, 1)")
        self.use_id_loss = bool(use_id_loss)
        self.id_loss_weight = float(id_loss_weight)
        self.id_loss_max_timestep = int(id_loss_max_timestep)
        self.id_loss_face_size = int(id_loss_face_size)
        self.id_loss_identity_source = str(id_loss_identity_source or "reference").lower()
        if self.id_loss_identity_source not in {"ground_truth_target", "reference"}:
            raise ValueError(f"Unknown id_loss_identity_source: {self.id_loss_identity_source}")
        self._id_loss_net = None
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

        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####
        """NEW BLOCK 2: optional per-branch optimizer grouping (ref vs noise processor groups).

        ba_noise_lr_scale < 1.0 slows the noise_to_* processor clones relative to everything
        else — the anti-drift damper from debug_04Jul/04Jul_findings.md §4.2/§5: the face loss
        reaches the noise CA group with no background anchor, so its LR is bounded separately
        while both pathways stay trainable (noise_and_ref). ba_noise_weight_decay optionally
        overrides the optimizer's weight_decay for that group only (lora_B → 0 = base weights,
        so wd on the noise group is a pull toward base behaviour).
        Defaults (1.0 / unset) reproduce the previous single-group behaviour exactly.
        """
        noise_lr_scale = float(config.get("ba_noise_lr_scale", 1.0))
        noise_wd = config.get("ba_noise_weight_decay", None)

        named = [(n, p) for n, p in self.unet.named_parameters() if p.requires_grad]

        def _is_noise_clone(name: str) -> bool:
            return ".processor." in name and ".noise_to_" in name

        if (noise_lr_scale == 1.0 and noise_wd is None) or not any(
            _is_noise_clone(n) for n, _ in named
        ):
            # Single group — bit-identical to the previous behaviour.
            return [
                {"params": [p for _, p in named], "lr": config.lr_for_lora, "name": "lora_params"},
            ]

        main_params = [p for n, p in named if not _is_noise_clone(n)]
        noise_params = [p for n, p in named if _is_noise_clone(n)]
        noise_group = {
            "params": noise_params,
            "lr": config.lr_for_lora * noise_lr_scale,
            "name": "ba_noise_params",
        }
        if noise_wd is not None:
            noise_group["weight_decay"] = float(noise_wd)
        return [
            {"params": main_params, "lr": config.lr_for_lora, "name": "lora_params"},
            noise_group,
        ]
        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####

    def get_state_dict(self):
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        state = {
            'lora_weights': lora_weights,
        }
        if hasattr(self.unet, "attn_processors"):
            proc_sd = {}
            trainable_by_processor = {}
            patched_proc_names = set(getattr(self, "_ba_patched_processor_names", ()))
            for name, proc in self.unet.attn_processors.items():
                if patched_proc_names and name not in patched_proc_names:
                    continue
                if not isinstance(proc, torch.nn.Module):
                    continue
                trainable = tuple(n for n, p in proc.named_parameters() if p.requires_grad)
                if not trainable:
                    continue
                trainable_by_processor[name] = sorted(trainable)
                full_sd = proc.state_dict()
                sd = {
                    k: v for k, v in full_sd.items()
                    if any(k == n or k.startswith(n + ".") for n in trainable)
                }
                if sd:
                    proc_sd[name] = sd
            if proc_sd:
                state["attn_processors"] = proc_sd
            if self.ba_strict_processor_restore:
                state["ba_processor_manifest"] = {
                    "installed_processor_names": sorted(patched_proc_names),
                    "state_processor_names": sorted(proc_sd),
                    "trainable_keys_by_processor": trainable_by_processor,
                    "architecture": {
                        "ba_sa_ref_token_mode": self.ba_sa_ref_token_mode,
                        "ba_sa_face_mode": self.ba_sa_face_mode,
                        "ba_sa_ref_layer_scope": self.ba_sa_ref_layer_scope,
                        "ba_sa_roi_grid_size": self.ba_sa_roi_grid_size,
                        "ba_sa_core_ratio": self.ba_sa_core_ratio,
                        "ba_sa_mix_init": self.ba_sa_mix_init,
                    },
                    "processor_classes": {
                        name: self.unet.attn_processors[name].__class__.__name__
                        for name in sorted(patched_proc_names)
                    },
                }
        return state

    def load_state_dict_(self, state_dict):
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # In newer peft versions this is an empty list when there are no unexpected keys
            assert not unexpected_keys, unexpected_keys
        processor_state = state_dict.get("attn_processors", {})
        if self.ba_strict_processor_restore:
            manifest = state_dict.get("ba_processor_manifest")
            if not isinstance(manifest, dict):
                raise RuntimeError("Strict BA restore requires ba_processor_manifest in the checkpoint")
            current_names = set(getattr(self, "_ba_patched_processor_names", ()))
            saved_names = set(manifest.get("installed_processor_names", ()))
            if current_names != saved_names:
                raise RuntimeError(
                    "Strict BA restore processor-name mismatch: "
                    f"missing={sorted(saved_names - current_names)[:5]}, "
                    f"unexpected={sorted(current_names - saved_names)[:5]}"
                )
            saved_architecture = manifest.get("architecture")
            if saved_architecture is not None:
                current_architecture = {
                    "ba_sa_ref_token_mode": self.ba_sa_ref_token_mode,
                    "ba_sa_face_mode": self.ba_sa_face_mode,
                    "ba_sa_ref_layer_scope": self.ba_sa_ref_layer_scope,
                    "ba_sa_roi_grid_size": self.ba_sa_roi_grid_size,
                    "ba_sa_core_ratio": self.ba_sa_core_ratio,
                    "ba_sa_mix_init": self.ba_sa_mix_init,
                }
                if saved_architecture != current_architecture:
                    raise RuntimeError(
                        "Strict BA restore architecture mismatch: "
                        f"checkpoint={saved_architecture}, current={current_architecture}"
                    )
            expected_state_names = set(manifest.get("state_processor_names", ()))
            actual_state_names = set(processor_state)
            if actual_state_names != expected_state_names:
                raise RuntimeError(
                    "Strict BA restore processor-state mismatch: "
                    f"missing={sorted(expected_state_names - actual_state_names)[:5]}, "
                    f"unexpected={sorted(actual_state_names - expected_state_names)[:5]}"
                )
            saved_trainable = manifest.get("trainable_keys_by_processor", {})
            for name in sorted(saved_names):
                proc = self.unet.attn_processors.get(name)
                if proc is None:
                    raise RuntimeError(f"Strict BA restore cannot find processor {name}")
                expected_class = manifest.get("processor_classes", {}).get(name)
                if expected_class and proc.__class__.__name__ != expected_class:
                    raise RuntimeError(
                        f"Strict BA restore class mismatch for {name}: "
                        f"checkpoint={expected_class}, current={proc.__class__.__name__}"
                    )
                current_trainable = sorted(
                    key for key, parameter in proc.named_parameters() if parameter.requires_grad
                )
                checkpoint_trainable = sorted(saved_trainable.get(name, ()))
                if current_trainable != checkpoint_trainable:
                    raise RuntimeError(
                        f"Strict BA restore trainable-key mismatch for {name}: "
                        f"checkpoint={checkpoint_trainable}, current={current_trainable}"
                    )
                if name in processor_state and set(processor_state[name]) != set(checkpoint_trainable):
                    raise RuntimeError(
                        f"Strict BA restore state-key mismatch for {name}: "
                        f"checkpoint={sorted(processor_state[name])}, expected={checkpoint_trainable}"
                    )

        for name, sd in processor_state.items():
            proc = self.unet.attn_processors.get(name)
            if proc is None or not hasattr(proc, "load_state_dict"):
                if self.ba_strict_processor_restore:
                    raise RuntimeError(f"Strict BA restore cannot load missing processor {name}")
                continue
            incompatible = proc.load_state_dict(sd, strict=False)
            if self.ba_strict_processor_restore and getattr(incompatible, "unexpected_keys", ()):
                raise RuntimeError(
                    f"Strict BA restore found unexpected keys for {name}: "
                    f"{incompatible.unexpected_keys}"
                )

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
        face_bbox_ref: Sequence[Sequence[float]] | None = None,
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
        max_timestep_exclusive = int(self.noise_scheduler.config.num_train_timesteps)
        if self.ba_train_timestep_mode == "inference_ba_region":
            ba_start_ratio = float(self.branched_attn_start_step) / float(max(1, self.num_inference_steps))
            max_timestep_inclusive = int(
                (1.0 - ba_start_ratio)
                * float(self.noise_scheduler.config.num_train_timesteps - 1)
            )
            max_timestep_exclusive = max(1, min(max_timestep_exclusive, max_timestep_inclusive + 1))
            if not hasattr(self, "_ba_logged_timestep_region"):
                print(
                    "[BA timestep sampler] inference_ba_region: "
                    f"sampling t in [0, {max_timestep_exclusive - 1}]"
                )
                self._ba_logged_timestep_region = True
        t_scalar = torch.randint(
            0,
            max_timestep_exclusive,
            (1,),
            device=latents.device,
        ).long()
        timesteps = t_scalar.repeat(batch_size)
        denoise_progress = 1.0 - (
            float(t_scalar.item()) / float(self.noise_scheduler.config.num_train_timesteps - 1)
        )

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

        num_inference_steps = max(1, self.num_inference_steps)
        photomaker_start_ratio = float(self.photomaker_start_step) / float(num_inference_steps)
        branched_start_ratio = float(self.branched_attn_start_step) / float(num_inference_steps)

        text_only_prompts = []
        trigger_word_pattern = re.compile(rf"\b{re.escape(self.trigger_word)}\b", flags=re.IGNORECASE)
        for prompt in prompts:
            text_only_prompt = trigger_word_pattern.sub(" ", prompt)
            text_only_prompt = " ".join(text_only_prompt.split())
            text_only_prompts.append(text_only_prompt)

        prompt_embeds_text_only, pooled_prompt_embeds_text_only = self.encode_prompt(
            prompt=text_only_prompts,
            do_cfg=False,
        )

        prompt_embeds_text_only = prompt_embeds_text_only.to(device=self.device, dtype=self.unet.dtype)
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

        if self.ba_correctness_guards and not hasattr(self, "_ba_logged_first_prediction"):
            prediction_fp32 = noise_pred.detach().float().cpu().contiguous()
            digest = hashlib.sha256(prediction_fp32.numpy().tobytes()).hexdigest()[:16]
            print(
                "[BA first-prediction fingerprint] "
                f"sha256={digest} mean={prediction_fp32.mean().item():.8f} "
                f"std={prediction_fp32.std().item():.8f}"
            )
            self._ba_logged_first_prediction = True

        output = {
            'model_pred': noise_pred,
            'target': noise,
        }
        if self.use_id_loss:
            apply_id_loss = int(t_scalar.item()) <= self.id_loss_max_timestep
            output["id_loss_applied"] = noise_pred.new_tensor(float(apply_id_loss))
            output["id_loss"] = (
                self._compute_id_loss(
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    pixel_values=pixel_values,
                    face_bbox=face_bbox,
                    ref_images=ref_images,
                    face_bbox_ref=face_bbox_ref,
                )
                if apply_id_loss
                else noise_pred.new_zeros(())
            )
        return output

    def _compute_id_loss(
        self,
        *,
        noise_pred,
        noisy_latents,
        timesteps,
        pixel_values,
        face_bbox,
        ref_images,
        face_bbox_ref,
    ):
        """Decode predicted x0 and compare its target face to the trusted reference identity."""
        if self._id_loss_net is None:
            from src.loss.id_loss import IdentityLoss

            self._id_loss_net = IdentityLoss(
                face_size=self.id_loss_face_size,
                device=self.device,
            )

        alpha_bar = self.noise_scheduler.alphas_cumprod.to(noise_pred.device)[timesteps].float()
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        x0 = (
            noisy_latents.float()
            - (1.0 - alpha_bar).sqrt() * noise_pred.float()
        ) / alpha_bar.sqrt().clamp_min(1e-4)

        tiling = hasattr(self.vae, "enable_tiling") and hasattr(self.vae, "disable_tiling")
        slicing = hasattr(self.vae, "enable_slicing") and hasattr(self.vae, "disable_slicing")
        if tiling:
            self.vae.enable_tiling()
        if slicing:
            self.vae.enable_slicing()
        try:
            generated = self.vae.decode(
                (x0 / self.vae.config.scaling_factor).to(self.vae.dtype)
            ).sample
        finally:
            if tiling:
                self.vae.disable_tiling()
            if slicing:
                self.vae.disable_slicing()

        targets = pixel_values.to(device=generated.device, dtype=generated.dtype)
        bboxes = face_bbox if isinstance(face_bbox, (list, tuple)) else [face_bbox]
        device_type = self.device.type if hasattr(self.device, "type") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            if self.id_loss_identity_source == "reference":
                return self._id_loss_net(
                    generated.float(),
                    targets.float(),
                    bboxes,
                    reference_images=ref_images,
                    reference_bboxes=face_bbox_ref,
                )
            return self._id_loss_net(generated.float(), targets.float(), bboxes)

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
    def _bbox_to_ref_mask(
        self,
        bbox: Optional[Sequence[float]],
        latent_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        mask = torch.zeros(1, 1, self.target_size, self.target_size, device=self.device)
        if bbox is None or len(bbox) < 4:
            if self.ba_correctness_guards:
                raise ValueError("Strict BA reference bbox is missing or incomplete")
            mask.fill_(1.0)
        else:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            if x1 <= x0 or y1 <= y0:
                if self.ba_correctness_guards:
                    raise ValueError(f"Strict BA reference bbox is inverted or empty: {bbox}")
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
                    if self.ba_correctness_guards:
                        raise ValueError(
                            f"Strict BA reference bbox is empty after resize: {bbox}"
                        )
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
            if self.ba_correctness_guards:
                raise ValueError("Strict BA target bbox is missing or incomplete")
            mask.fill_(1.0)
            return mask

        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            if self.ba_correctness_guards:
                raise ValueError(f"Strict BA target bbox is inverted or empty: {bbox}")
            mask.fill_(1.0)
            return mask

        scale_w = latent_shape[1] / max(image_shape[1], 1)
        scale_h = latent_shape[0] / max(image_shape[0], 1)

        x_start = max(0, min(latent_shape[1], int(round(x0 * scale_w))))
        x_end = max(0, min(latent_shape[1], int(round(x1 * scale_w))))
        y_start = max(0, min(latent_shape[0], int(round(y0 * scale_h))))
        y_end = max(0, min(latent_shape[0], int(round(y1 * scale_h))))

        if x_end <= x_start or y_end <= y_start:
            if self.ba_correctness_guards:
                raise ValueError(f"Strict BA target bbox is empty after resize: {bbox}")
            mask.fill_(1.0)
            return mask

        mask[:, :, y_start:y_end, x_start:x_end] = 1.0
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

    def ensure_branched_after_eval(self):
        ensure_branched_after_eval_helper(self)
    ##### BRANCHED ATTENTION - HELPER UTILS #####
