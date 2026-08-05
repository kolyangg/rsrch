from __future__ import annotations

import hashlib
import time
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
from diffusers import DDIMScheduler
from src.model.photomaker_path import resolve_photomaker_path
from src.model.sdxl.original import SDXL

##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
"""Import branched-attention forward/patch helpers and PMv2 face-ID dependencies used by training."""
from .insightface_package import create_face_analyzer
from .lora2_helpers import (
    assert_branched_trainable_contract,
    branched_trainable_role_groups,
    collect_branched_telemetry,
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
        id_alpha: float = 0.3,             # strength of ID embedding injection in BranchedAttnProcessor
        use_id_embeds: bool = True,        # toggle ID embedding injection (controls id_to_hidden usage)
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        branched_attn_start_step: int = 15,
        num_inference_steps: int = 50,
        skip_unused_text_conditioning: bool = False,
        conditioning_cache_enabled: bool = False,
        conditioning_cache_max_entries: int = 512,
        batched_conditioning_preparation: bool = False,
        cache_prepared_masks: bool = False,
        compute_branch_debug_outputs: bool = True,
        strict_branched_install: bool = False,
        strict_trainable_contract: bool = False,
        branched_state_dict_mode: str = "legacy",
        ba_architecture_version: str = "hard_replace_v1",
        branched_trainable_dtype: str = "inherit",
        ba_ref_kv_rank: Optional[int] = None,
        ba_output_rank: Optional[int] = None,
        ba_branch_q_rank: int = 16,
        ba_face_fusion_mode: str = "hard_reference_replace",
        ba_face_branch_scale: float = 1.0,
        ba_gate_init: float = 0.10,
        ba_gate_max: float = 1.0,
        ba_gate_timestep: bool = True,
        ba_gate_face_area: bool = True,
        ba_mix_init: float = 0.50,
        ba_mix_floor: float = 0.0,
        ba_mix_max: float = 1.0,
        ba_mix_timestep: bool = True,
        ba_mix_face_area: bool = True,
        ba_reference_rms_match: bool = False,
        ba_reference_rms_clip_min: float = 0.50,
        ba_reference_rms_clip_max: float = 2.00,
        ba_mix_override: Optional[float] = None,
        ba_telemetry_enabled: bool = False,
        ba_telemetry_interval: int = 50,
        ba_reference_loss_mode: str = "detached_diagnostic",
        ba_require_denoise_progress: bool = True,
        ba_self_attention_groups: Optional[Sequence[str]] = None,
        ba_training_timestep_policy: str = "uniform_all",
        ba_spatial_reference_shuffle_probability: float = 0.0,
        ba_install_on_device: bool = False,
        ba_enforce_reference_only_hard_route: bool = False,
        ba_hard_v1_true_reference_key_mask: bool = False,
        ba_hard_v1_branch_output_rank: Optional[int] = None,
        ba_hard_v1_reference_roi_warp: bool = False,
        generic_adapter_train_scope: str = "none",
        photomaker_default_train_scope: str = "none",
        ba_hard_v1_lora_rank: Optional[int] = None,
        ba_identity_ca_v2_enabled: bool = False,
        ba_identity_ca_v2_groups: Optional[Sequence[str]] = None,
        ba_identity_ca_v2_rank: int = 16,
        ba_residual_identity_ca_v3_enabled: bool = False,
        ba_residual_identity_ca_v3_groups: Optional[Sequence[str]] = None,
        ba_residual_identity_ca_v3_rank: int = 64,
        ba_residual_identity_ca_v3_gate_init: float = 0.02,
        ba_residual_identity_ca_v3_gate_max: float = 0.20,
        identity_aux_enabled: bool = False,
        identity_aux_cadence: int = 4,
        identity_aux_max_timestep: int = 400,
        identity_aux_ramp_start_step: int = 2000,
        identity_aux_ramp_end_step: int = 6000,
        identity_aux_max_weight: float = 0.05,
        identity_aux_crop_padding: float = 0.25,
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
        FACEANALYSIS_CPU = False  # Set to True to force CPU provider (debug / low-VRAM mode)
        
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
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

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
        self.skip_unused_text_conditioning = bool(skip_unused_text_conditioning)
        self.conditioning_cache_enabled = bool(conditioning_cache_enabled)
        self.conditioning_cache_max_entries = max(0, int(conditioning_cache_max_entries))
        self.batched_conditioning_preparation = bool(
            batched_conditioning_preparation
        )
        self.cache_prepared_masks = bool(cache_prepared_masks)
        self.compute_branch_debug_outputs = bool(compute_branch_debug_outputs)
        self.strict_branched_install = bool(strict_branched_install)
        self.strict_trainable_contract = bool(strict_trainable_contract)
        self.branched_state_dict_mode = (branched_state_dict_mode or "legacy").lower()
        if self.branched_state_dict_mode not in {"legacy", "trainable_v2"}:
            raise ValueError(
                "branched_state_dict_mode must be 'legacy' or 'trainable_v2', "
                f"got {self.branched_state_dict_mode!r}"
            )
        self.generic_adapter_train_scope = (
            generic_adapter_train_scope or "none"
        ).lower()
        if self.generic_adapter_train_scope not in {
            "none",
            "effective_all",
            "cross_attention",
            "self_attention_output",
        }:
            raise ValueError(
                "generic_adapter_train_scope must be 'none', 'effective_all', "
                "'cross_attention', or 'self_attention_output'; got "
                f"{self.generic_adapter_train_scope!r}"
            )
        self.photomaker_default_train_scope = (
            photomaker_default_train_scope or "none"
        ).lower()
        if self.photomaker_default_train_scope not in {
            "none",
            "effective_all",
        }:
            raise ValueError(
                "photomaker_default_train_scope must be 'none' or "
                f"'effective_all'; got {self.photomaker_default_train_scope!r}"
            )
        self.ba_architecture_version = (
            ba_architecture_version or "hard_replace_v1"
        ).lower()
        if self.ba_architecture_version not in {
            "hard_replace_v1",
            "residual_sa_v2",
            "anchored_mix_sa_v3",
            "query_adaptive_hard_sa_v4",
        }:
            raise ValueError(
                "ba_architecture_version must be 'hard_replace_v1', "
                "'residual_sa_v2', 'anchored_mix_sa_v3', or "
                "'query_adaptive_hard_sa_v4'; got "
                f"{self.ba_architecture_version!r}"
            )
        adapter_scope_enabled = any(
            scope != "none"
            for scope in (
                self.generic_adapter_train_scope,
                self.photomaker_default_train_scope,
            )
        )
        if adapter_scope_enabled and not all(
            (
                self.train_ba_only,
                self.strict_trainable_contract,
                self.branched_state_dict_mode == "trainable_v2",
                self.ba_architecture_version == "hard_replace_v1",
                self.branched_attn_weight_mode == "noise_and_ref",
                not self.train_branched_ca_lora,
            )
        ):
            raise ValueError(
                "Audited outer-adapter scopes require train_ba_only=true, "
                "strict_trainable_contract=true, trainable-v2 checkpoints, "
                "hard_replace_v1 noise_and_ref, and branched CA disabled"
            )
        self.branched_trainable_dtype = (
            branched_trainable_dtype or "inherit"
        ).lower()
        if self.branched_trainable_dtype not in {
            "inherit",
            "fp32",
            "float32",
        }:
            raise ValueError(
                "branched_trainable_dtype must be 'inherit' or 'fp32', "
                f"got {self.branched_trainable_dtype!r}"
            )
        if (
            self.ba_architecture_version in {
                "residual_sa_v2",
                "anchored_mix_sa_v3",
                "query_adaptive_hard_sa_v4",
            }
            and self.branched_trainable_dtype not in {"fp32", "float32"}
        ):
            raise ValueError(
                f"{self.ba_architecture_version} requires branched_trainable_dtype=fp32"
            )
        self.ba_ref_kv_rank = int(
            ba_ref_kv_rank if ba_ref_kv_rank is not None else rank
        )
        self.ba_output_rank = int(
            ba_output_rank if ba_output_rank is not None else rank
        )
        self.ba_branch_q_rank = int(ba_branch_q_rank)
        self.ba_face_fusion_mode = str(ba_face_fusion_mode).lower()
        self.ba_face_branch_scale = float(ba_face_branch_scale)
        if min(
            self.ba_ref_kv_rank,
            self.ba_output_rank,
            self.ba_branch_q_rank,
        ) <= 0:
            raise ValueError("Versioned BA ranks must be positive")
        self.ba_gate_init = float(ba_gate_init)
        self.ba_gate_max = float(ba_gate_max)
        self.ba_gate_timestep = bool(ba_gate_timestep)
        self.ba_gate_face_area = bool(ba_gate_face_area)
        self.ba_mix_init = float(ba_mix_init)
        self.ba_mix_floor = float(ba_mix_floor)
        self.ba_mix_max = float(ba_mix_max)
        self.ba_mix_timestep = bool(ba_mix_timestep)
        self.ba_mix_face_area = bool(ba_mix_face_area)
        self.ba_reference_rms_match = bool(ba_reference_rms_match)
        self.ba_reference_rms_clip_min = float(ba_reference_rms_clip_min)
        self.ba_reference_rms_clip_max = float(ba_reference_rms_clip_max)
        self.ba_mix_override = (
            None if ba_mix_override is None else float(ba_mix_override)
        )
        if self.ba_mix_override is not None and not 0.0 <= self.ba_mix_override <= 1.0:
            raise ValueError("ba_mix_override must be in [0, 1]")
        if self.ba_architecture_version == "query_adaptive_hard_sa_v4":
            if self.ba_face_fusion_mode != "hard_reference_replace":
                raise ValueError(
                    "Hard BA-v4 requires ba_face_fusion_mode="
                    "hard_reference_replace"
                )
            if self.ba_face_branch_scale != 1.0:
                raise ValueError("Hard BA-v4 requires ba_face_branch_scale=1.0")
            if self.ba_mix_override is not None:
                raise ValueError("Hard BA-v4 does not accept ba_mix_override")
        self.ba_telemetry_enabled = bool(ba_telemetry_enabled)
        self.ba_telemetry_interval = int(ba_telemetry_interval)
        if self.ba_telemetry_interval <= 0:
            raise ValueError("ba_telemetry_interval must be positive")
        self.ba_reference_loss_mode = (
            ba_reference_loss_mode or "detached_diagnostic"
        ).lower()
        if self.ba_reference_loss_mode not in {
            "detached_diagnostic",
            "differentiable_rank",
        }:
            raise ValueError(
                "ba_reference_loss_mode must be 'detached_diagnostic' or "
                f"'differentiable_rank', got {self.ba_reference_loss_mode!r}"
            )
        if self.ba_architecture_version == "anchored_mix_sa_v3":
            if not 0.0 <= self.ba_mix_floor < self.ba_mix_max <= 1.0:
                raise ValueError(
                    "BA-v3 mix bounds require 0 <= floor < max <= 1"
                )
            if not self.ba_mix_floor < self.ba_mix_init < self.ba_mix_max:
                raise ValueError("BA-v3 mix_init must be strictly inside its bounds")
            if self.ba_reference_rms_clip_min <= 0.0:
                raise ValueError("BA-v3 reference RMS clip minimum must be positive")
            if self.ba_reference_rms_clip_max < self.ba_reference_rms_clip_min:
                raise ValueError("BA-v3 reference RMS clip bounds are reversed")
        self.ba_require_denoise_progress = bool(ba_require_denoise_progress)
        self.ba_self_attention_groups = (
            None
            if ba_self_attention_groups is None
            else tuple(str(group) for group in ba_self_attention_groups)
        )
        self.ba_training_timestep_policy = (
            ba_training_timestep_policy or "uniform_all"
        ).lower()
        if self.ba_training_timestep_policy not in {
            "uniform_all",
            "inference_active",
        }:
            raise ValueError(
                "ba_training_timestep_policy must be 'uniform_all' or "
                f"'inference_active', got {self.ba_training_timestep_policy!r}"
            )
        if (
            self.ba_training_timestep_policy == "inference_active"
            and not self.train_ba_all_steps
        ):
            raise ValueError(
                "inference_active timestep sampling requires train_ba_all_steps=true"
            )
        self.ba_spatial_reference_shuffle_probability = float(
            ba_spatial_reference_shuffle_probability
        )
        if not 0.0 <= self.ba_spatial_reference_shuffle_probability <= 1.0:
            raise ValueError(
                "ba_spatial_reference_shuffle_probability must be in [0, 1], "
                f"got {self.ba_spatial_reference_shuffle_probability}"
            )
        self.ba_install_on_device = bool(ba_install_on_device)
        self.ba_enforce_reference_only_hard_route = bool(
            ba_enforce_reference_only_hard_route
        )
        self.ba_hard_v1_true_reference_key_mask = bool(
            ba_hard_v1_true_reference_key_mask
        )
        self.ba_hard_v1_branch_output_rank = (
            None
            if ba_hard_v1_branch_output_rank is None
            else int(ba_hard_v1_branch_output_rank)
        )
        if (
            self.ba_hard_v1_branch_output_rank is not None
            and self.ba_hard_v1_branch_output_rank <= 0
        ):
            raise ValueError(
                "ba_hard_v1_branch_output_rank must be positive when enabled"
            )
        self.ba_hard_v1_reference_roi_warp = bool(
            ba_hard_v1_reference_roi_warp
        )
        self.ba_hard_v1_lora_rank = (
            None
            if ba_hard_v1_lora_rank is None
            else int(ba_hard_v1_lora_rank)
        )
        if (
            self.ba_hard_v1_lora_rank is not None
            and self.ba_hard_v1_lora_rank <= 0
        ):
            raise ValueError("ba_hard_v1_lora_rank must be positive when enabled")
        if (
            self.ba_hard_v1_lora_rank is not None
            and self.branched_attn_new_weight_kind != "lora"
        ):
            raise ValueError(
                "ba_hard_v1_lora_rank requires branched_attn_new_weight_kind=lora"
            )
        self.ba_identity_ca_v2_enabled = bool(ba_identity_ca_v2_enabled)
        self.ba_identity_ca_v2_groups = (
            None
            if ba_identity_ca_v2_groups is None
            else tuple(str(group) for group in ba_identity_ca_v2_groups)
        )
        self.ba_identity_ca_v2_rank = int(ba_identity_ca_v2_rank)
        if self.ba_identity_ca_v2_rank <= 0:
            raise ValueError("ba_identity_ca_v2_rank must be positive")
        if self.ba_identity_ca_v2_enabled and not self.ba_identity_ca_v2_groups:
            raise ValueError(
                "ba_identity_ca_v2_enabled requires non-empty block groups"
            )
        if self.ba_identity_ca_v2_enabled and not all(
            (
                self.train_ba_only,
                self.strict_trainable_contract,
                self.branched_state_dict_mode == "trainable_v2",
                self.ba_architecture_version == "hard_replace_v1",
                self.branched_attn_weight_mode == "noise_and_ref",
                not self.train_branched_ca_lora,
                self.pose_adapt_ratio == 0.0,
                not self.ca_mixing_for_face,
            )
        ):
            raise ValueError(
                "Corrected identity CA requires strict trainable-v2 hard-v1 "
                "BA-only noise_and_ref, legacy branched CA off, "
                "pose_adapt_ratio=0, and ca_mixing_for_face=false"
            )
        self.ba_residual_identity_ca_v3_enabled = bool(
            ba_residual_identity_ca_v3_enabled
        )
        self.ba_residual_identity_ca_v3_groups = (
            None
            if ba_residual_identity_ca_v3_groups is None
            else tuple(str(group) for group in ba_residual_identity_ca_v3_groups)
        )
        self.ba_residual_identity_ca_v3_rank = int(
            ba_residual_identity_ca_v3_rank
        )
        self.ba_residual_identity_ca_v3_gate_init = float(
            ba_residual_identity_ca_v3_gate_init
        )
        self.ba_residual_identity_ca_v3_gate_max = float(
            ba_residual_identity_ca_v3_gate_max
        )
        if self.ba_identity_ca_v2_enabled and self.ba_residual_identity_ca_v3_enabled:
            raise ValueError("Hard and residual identity CA are mutually exclusive")
        if self.ba_residual_identity_ca_v3_enabled and not all(
            (
                self.ba_residual_identity_ca_v3_groups,
                self.ba_residual_identity_ca_v3_rank > 0,
                0.0
                < self.ba_residual_identity_ca_v3_gate_init
                < self.ba_residual_identity_ca_v3_gate_max
                <= 1.0,
                self.train_ba_only,
                self.strict_trainable_contract,
                self.branched_state_dict_mode == "trainable_v2",
                self.ba_architecture_version == "hard_replace_v1",
                self.branched_attn_weight_mode == "noise_and_ref",
                not self.train_branched_ca_lora,
                self.pose_adapt_ratio == 0.0,
                not self.ca_mixing_for_face,
            )
        ):
            raise ValueError(
                "Residual identity CA requires groups, valid rank/gate bounds, "
                "strict trainable-v2 hard-v1 BA-only noise_and_ref, legacy "
                "branched CA off, pose_adapt_ratio=0, and ca_mixing_for_face=false"
            )
        self.identity_aux_enabled = bool(identity_aux_enabled)
        self.identity_aux_cadence = int(identity_aux_cadence)
        self.identity_aux_max_timestep = int(identity_aux_max_timestep)
        self.identity_aux_ramp_start_step = int(identity_aux_ramp_start_step)
        self.identity_aux_ramp_end_step = int(identity_aux_ramp_end_step)
        self.identity_aux_max_weight = float(identity_aux_max_weight)
        self.identity_aux_crop_padding = float(identity_aux_crop_padding)
        if self.identity_aux_enabled and not all(
            (
                self.identity_aux_cadence > 0,
                self.identity_aux_max_timestep >= 0,
                0 <= self.identity_aux_ramp_start_step
                < self.identity_aux_ramp_end_step,
                self.identity_aux_max_weight > 0.0,
                self.identity_aux_crop_padding >= 0.0,
            )
        ):
            raise ValueError("Invalid predicted-x0 identity auxiliary configuration")
        if self.ba_architecture_version != "hard_replace_v1" and any(
            (
                self.ba_enforce_reference_only_hard_route,
                self.ba_hard_v1_true_reference_key_mask,
                self.ba_hard_v1_branch_output_rank is not None,
                self.ba_hard_v1_reference_roi_warp,
                self.ba_hard_v1_lora_rank is not None,
                self.ba_identity_ca_v2_enabled,
                self.ba_residual_identity_ca_v3_enabled,
            )
        ):
            raise ValueError("Hard-v1 controls require ba_architecture_version=hard_replace_v1")
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
        prepare_started = time.perf_counter()
        if (
            self.ba_architecture_version in {
                "residual_sa_v2",
                "anchored_mix_sa_v3",
                "query_adaptive_hard_sa_v4",
            }
            and self.ba_install_on_device
        ):
            # 2 Aug 2026 - AICODE-NOTE: Effective PEFT K/V materialization is
            # prohibitively slow in BF16 on CPU. Opt-in v2 runs stage only the
            # U-Net on its assigned GPU before dtype conversion and processor
            # installation; historical hard-replacement behavior is unchanged.
            self.unet.to(self.device)
            print(
                "[BA Init Timing] staged_unet_on_device "
                f"seconds={time.perf_counter() - prepare_started:.3f} "
                f"device={self.device}"
            )
        super().prepare_for_training()
        print(
            "[BA Init Timing] base_prepare_complete "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )
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
        print(
            "[BA Init Timing] adapters_ready "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )


        ##### BRANCHED ATTENTION - NEW BLOCK 1 #####
        """NEW BLOCK 1: pre-install branched attention processors before optimizer creation and mark their params trainable."""
        install_branched_processors_for_training(self)
        print(
            "[BA Init Timing] processors_ready "
            f"seconds={time.perf_counter() - prepare_started:.3f}"
        )

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

        if (
            self.ba_architecture_version == "hard_replace_v1"
            and self.train_ba_only
            and (
                self.generic_adapter_train_scope != "none"
                or self.photomaker_default_train_scope != "none"
            )
        ):
            # 5 Aug 2026 - AICODE-NOTE: E13+ jointly trains three explicitly
            # owned paths. Separate groups make the persistent PhotoMaker
            # default LR independently controllable without changing ownership.
            role_parameters = {
                "ba": [],
                "generic_adapter": [],
                "photomaker_default": [],
            }
            for name, parameter in self.unet.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".lora_adapter." in name:
                    role = "generic_adapter"
                elif ".default." in name:
                    role = "photomaker_default"
                elif ".attn1.processor." in name or ".attn2.processor." in name:
                    role = "ba"
                else:
                    raise RuntimeError(
                        f"Joint hard-v1 optimizer cannot classify {name!r}"
                    )
                role_parameters[role].append(parameter)
            lr_by_role = {
                "ba": float(getattr(config, "ba_lr", config.lr_for_lora)),
                "generic_adapter": float(
                    getattr(config, "generic_adapter_lr", config.lr_for_lora)
                ),
                "photomaker_default": float(
                    getattr(config, "photomaker_default_lr", config.lr_for_lora)
                ),
            }
            groups = [
                {
                    "params": role_parameters[role],
                    "lr": lr_by_role[role],
                    "name": role,
                }
                for role in ("ba", "generic_adapter", "photomaker_default")
                if role_parameters[role]
            ]
            if not groups:
                raise RuntimeError("Joint hard-v1 optimizer resolved no parameters")
            return groups

        if self.ba_architecture_version in {
            "residual_sa_v2",
            "anchored_mix_sa_v3",
            "query_adaptive_hard_sa_v4",
        } and self.train_ba_only:
            role_groups = branched_trainable_role_groups(self)
            lr_by_role = {
                "ref_kv": float(getattr(config, "ba_ref_kv_lr", config.lr_for_lora)),
                "ref_output": float(
                    getattr(config, "ba_ref_output_lr", config.lr_for_lora)
                ),
                "gate": float(getattr(config, "ba_gate_lr", config.lr_for_lora)),
                "mix": float(getattr(config, "ba_mix_lr", config.lr_for_lora)),
                "ref_query": float(
                    getattr(config, "ba_ref_query_lr", config.lr_for_lora)
                ),
            }
            unknown_roles = set(role_groups) - set(lr_by_role)
            if unknown_roles:
                raise RuntimeError(
                    f"Unknown {self.ba_architecture_version} optimizer roles: "
                    f"{unknown_roles}"
                )
            if self.ba_architecture_version == "residual_sa_v2":
                role_order = ("ref_kv", "ref_output", "gate")
            elif self.ba_architecture_version == "anchored_mix_sa_v3":
                role_order = ("ref_kv", "ref_output", "mix")
            else:
                role_order = ("ref_query", "ref_kv", "ref_output")
            return [
                {
                    "params": role_groups[role],
                    "lr": lr_by_role[role],
                    "name": f"ba_{role}",
                }
                for role in role_order
                if role_groups.get(role)
            ]

        # Default behavior: train all UNet parameters with requires_grad=True (LoRA + processors).
        lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {"params": lora_params, "lr": config.lr_for_lora, "name": "lora_params"},
        ]
        return trainable_params

    def assert_trainable_contract(self, optimizer=None) -> dict:
        if not self.strict_trainable_contract:
            return {}
        return assert_branched_trainable_contract(self, optimizer=optimizer)

    def _branched_architecture_manifest(self) -> dict:
        named_parameters = dict(self.unet.named_parameters())
        trainable_names = tuple(
            sorted(
                name for name, parameter in named_parameters.items()
                if parameter.requires_grad
            )
        )
        processor_names = list(getattr(self, "_ba_patched_processor_names", ()))
        semantic_names = list(getattr(self, "_ba_semantic_processor_names", ()))
        semantic_names_sha256 = hashlib.sha256(
            "\n".join(semantic_names).encode("utf-8")
        ).hexdigest()
        hard_v1_extended = bool(
            self.ba_architecture_version == "hard_replace_v1"
            and (
                self.ba_hard_v1_true_reference_key_mask
                or self.ba_hard_v1_branch_output_rank is not None
                or self.ba_hard_v1_reference_roi_warp
                or self.ba_hard_v1_lora_rank is not None
                or self.ba_identity_ca_v2_enabled
                or self.ba_residual_identity_ca_v3_enabled
            )
        )
        processor_code_version = {
            "hard_replace_v1": (
                4
                if self.ba_residual_identity_ca_v3_enabled
                else (
                    3
                    if self.ba_identity_ca_v2_enabled
                    else (2 if hard_v1_extended else 1)
                )
            ),
            "residual_sa_v2": 2,
            "anchored_mix_sa_v3": 3,
            "query_adaptive_hard_sa_v4": 4,
        }[self.ba_architecture_version]
        manifest = {
            "format": "photomaker_branched_trainable_unet_v2",
            "ba_architecture_version": self.ba_architecture_version,
            "processor_code_version": processor_code_version,
            "branched_attn_lora_rank": int(self.branched_attn_lora_rank),
            "branched_attn_weight_mode": self.branched_attn_weight_mode,
            "branched_attn_new_weight_kind": self.branched_attn_new_weight_kind,
            "train_ba_only": bool(self.train_ba_only),
            "train_branched_ca_lora": bool(self.train_branched_ca_lora),
            "ba_patch_top_k": float(self.ba_patch_top_k),
            "ba_train_top_k": float(self.ba_train_top_k),
            "non_ba_train": bool(self.non_ba_train),
            "disable_branched_sa": bool(getattr(self, "disable_branched_sa", False)),
            "disable_branched_ca": bool(getattr(self, "disable_branched_ca", False)),
            "branched_trainable_dtype": self.branched_trainable_dtype,
            "ba_ref_kv_rank": int(self.ba_ref_kv_rank),
            "ba_output_rank": int(self.ba_output_rank),
            "ba_branch_q_rank": int(self.ba_branch_q_rank),
            "ba_face_fusion_mode": self.ba_face_fusion_mode,
            "ba_face_branch_scale": float(self.ba_face_branch_scale),
            "ba_gate_init": float(self.ba_gate_init),
            "ba_gate_max": float(self.ba_gate_max),
            "ba_gate_timestep": bool(self.ba_gate_timestep),
            "ba_gate_face_area": bool(self.ba_gate_face_area),
            "ba_require_denoise_progress": bool(self.ba_require_denoise_progress),
            "ba_training_timestep_policy": self.ba_training_timestep_policy,
            "ba_spatial_reference_shuffle_probability": float(
                self.ba_spatial_reference_shuffle_probability
            ),
            "ba_install_on_device": bool(self.ba_install_on_device),
            "ba_self_attention_groups": (
                None
                if self.ba_self_attention_groups is None
                else list(self.ba_self_attention_groups)
            ),
            "semantic_processor_names": semantic_names,
            "semantic_processor_names_sha256": semantic_names_sha256,
            "merge_kind": (
                "reference_residual"
                if self.ba_architecture_version == "residual_sa_v2"
                else (
                    "anchored_reference_interpolation"
                    if self.ba_architecture_version == "anchored_mix_sa_v3"
                    else (
                        "query_adaptive_hard_reference_replacement"
                        if self.ba_architecture_version
                        == "query_adaptive_hard_sa_v4"
                        else "hard_face_replacement"
                    )
                )
            ),
            "target_query_source": (
                "frozen_target"
                if self.ba_architecture_version in {
                    "residual_sa_v2",
                    "anchored_mix_sa_v3",
                }
                else (
                    "branch_adapted_target"
                    if self.ba_architecture_version
                    == "query_adaptive_hard_sa_v4"
                    else "configured_noise_projection"
                )
            ),
            "reference_key_mask": (
                self.ba_hard_v1_true_reference_key_mask
                if self.ba_architecture_version == "hard_replace_v1"
                else self.ba_architecture_version in {
                    "residual_sa_v2",
                    "anchored_mix_sa_v3",
                    "query_adaptive_hard_sa_v4",
                }
            ),
            "strict_face_routing": bool(getattr(self, "strict_face_routing", False)),
            "pose_adapt_ratio": float(self.pose_adapt_ratio),
            "ca_mixing_for_face": bool(self.ca_mixing_for_face),
            "photomaker_start_step": int(self.photomaker_start_step),
            "branched_attn_start_step": int(self.branched_attn_start_step),
            "num_inference_steps": int(self.num_inference_steps),
            "patched_processor_names": processor_names,
            "trainable_processor_names": list(
                getattr(self, "_ba_trainable_processor_names", ())
            ),
            "trainable_names": list(trainable_names),
            "trainable_shapes": {
                name: list(named_parameters[name].shape) for name in trainable_names
            },
            "trainable_dtypes": {
                name: str(named_parameters[name].dtype).replace("torch.", "")
                for name in trainable_names
            },
        }
        if (
            self.generic_adapter_train_scope != "none"
            or self.photomaker_default_train_scope != "none"
        ):
            # 4 Aug 2026 - AICODE-NOTE: Only new explicit adapter-scope runs
            # extend the manifest. Defaults-off manifests stay byte-compatible
            # with existing E0-E6 schema-v2 checkpoints.
            manifest["generic_adapter_train_scope"] = (
                self.generic_adapter_train_scope
            )
            manifest["photomaker_default_train_scope"] = (
                self.photomaker_default_train_scope
            )
        if hard_v1_extended:
            hard_v1_extensions = {
                "true_reference_key_mask": bool(
                    self.ba_hard_v1_true_reference_key_mask
                ),
                "branch_output_rank": self.ba_hard_v1_branch_output_rank,
                "reference_roi_warp": bool(
                    self.ba_hard_v1_reference_roi_warp
                ),
                "face_fusion_mode": "hard_reference_replace",
            }
            if self.ba_hard_v1_lora_rank is not None:
                hard_v1_extensions["lora_rank"] = int(
                    self.ba_hard_v1_lora_rank
                )
            if self.ba_identity_ca_v2_enabled:
                identity_names = list(
                    getattr(self, "_ba_identity_ca_processor_names", ())
                )
                hard_v1_extensions["identity_ca_v2"] = {
                    "enabled": True,
                    "groups": list(self.ba_identity_ca_v2_groups or ()),
                    "rank": int(self.ba_identity_ca_v2_rank),
                    "processor_names": identity_names,
                    "routing": "target_q_active_photomaker_id_kv",
                    "merge": "native_outside_face_id_only_inside_face",
                    "legacy_branched_ca": False,
                }
            if self.ba_residual_identity_ca_v3_enabled:
                identity_names = list(
                    getattr(self, "_ba_identity_ca_processor_names", ())
                )
                hard_v1_extensions["residual_identity_ca_v3"] = {
                    "enabled": True,
                    "groups": list(
                        self.ba_residual_identity_ca_v3_groups or ()
                    ),
                    "rank": int(self.ba_residual_identity_ca_v3_rank),
                    "gate_init": float(
                        self.ba_residual_identity_ca_v3_gate_init
                    ),
                    "gate_max": float(
                        self.ba_residual_identity_ca_v3_gate_max
                    ),
                    "processor_names": identity_names,
                    "routing": "target_q_active_photomaker_id_kv",
                    "merge": "native_plus_face_mask_times_bounded_gate_times_rms_id_delta",
                    "delta_output_zero_init": True,
                    "legacy_branched_ca": False,
                }
            manifest["hard_v1_extensions"] = hard_v1_extensions
        if self.identity_aux_enabled:
            manifest["identity_auxiliary"] = {
                "kind": "predicted_x0_photomaker_clip_cosine",
                "cadence": self.identity_aux_cadence,
                "max_timestep": self.identity_aux_max_timestep,
                "ramp_steps": [
                    self.identity_aux_ramp_start_step,
                    self.identity_aux_ramp_end_step,
                ],
                "max_weight": self.identity_aux_max_weight,
                "crop_padding": self.identity_aux_crop_padding,
            }
        if self.ba_architecture_version == "anchored_mix_sa_v3":
            manifest.update(
                {
                    "routing": "target_q_reference_kv_true_key_mask",
                    "merge_equation": (
                        "native_plus_target_mask_times_alpha_times_"
                        "reference_minus_native"
                    ),
                    "reference_output_base": "frozen_native_to_out",
                    "ba_mix_init": float(self.ba_mix_init),
                    "ba_mix_floor": float(self.ba_mix_floor),
                    "ba_mix_max": float(self.ba_mix_max),
                    "ba_mix_timestep": bool(self.ba_mix_timestep),
                    "ba_mix_face_area": bool(self.ba_mix_face_area),
                    "ba_reference_rms_match": bool(
                        self.ba_reference_rms_match
                    ),
                    "ba_reference_rms_clip": [
                        float(self.ba_reference_rms_clip_min),
                        float(self.ba_reference_rms_clip_max),
                    ],
                    "ba_telemetry_enabled": bool(self.ba_telemetry_enabled),
                    "ba_telemetry_interval": int(self.ba_telemetry_interval),
                    "ba_reference_loss_mode": self.ba_reference_loss_mode,
                }
            )
        elif self.ba_architecture_version == "query_adaptive_hard_sa_v4":
            manifest.update(
                {
                    "routing": "branch_target_q_reference_kv_true_key_mask",
                    "merge_equation": (
                        "native_outside_target_mask_reference_inside_target_mask"
                    ),
                    "face_fusion_mode": "hard_reference_replace",
                    "face_branch_scale": 1.0,
                    "reference_output_base": "frozen_native_to_out",
                    "ba_reference_loss_mode": self.ba_reference_loss_mode,
                    "ba_telemetry_enabled": bool(self.ba_telemetry_enabled),
                    "ba_telemetry_interval": int(self.ba_telemetry_interval),
                }
            )
        return manifest

    def _get_trainable_state_dict_v2(self) -> dict:
        self.assert_trainable_contract()
        named_parameters = dict(self.unet.named_parameters())
        trainable_names = tuple(
            sorted(
                name for name, parameter in named_parameters.items()
                if parameter.requires_grad
            )
        )
        if not trainable_names:
            raise RuntimeError("Refusing to save an empty trainable U-Net state")
        return {
            "schema_version": 2,
            "state_format": "trainable_unet_v2",
            "architecture": self._branched_architecture_manifest(),
            "trainable_unet": {
                name: named_parameters[name].detach().cpu().clone()
                for name in trainable_names
            },
        }

    def get_state_dict(self):
        if self.branched_state_dict_mode == "trainable_v2":
            # 1 Aug 2026 - Save the exact requires-grad allowlist so checkpoint
            # fidelity cannot depend on a hand-maintained adapter-name subset.
            return self._get_trainable_state_dict_v2()

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

    def _load_trainable_state_dict_v2(self, state_dict: dict) -> None:
        if state_dict.get("state_format") != "trainable_unet_v2":
            raise RuntimeError(
                f"Unknown schema-v2 state format: {state_dict.get('state_format')!r}"
            )
        saved_manifest = state_dict.get("architecture")
        current_manifest = self._branched_architecture_manifest()
        if saved_manifest != current_manifest:
            raise RuntimeError(
                "Branched checkpoint architecture mismatch: "
                f"saved={saved_manifest!r}, current={current_manifest!r}"
            )

        received = state_dict.get("trainable_unet")
        if not isinstance(received, dict):
            raise RuntimeError("Schema-v2 checkpoint is missing trainable_unet")
        named_parameters = dict(self.unet.named_parameters())
        expected = set(current_manifest["trainable_names"])
        received_names = set(received)
        if expected != received_names:
            raise RuntimeError(
                "Schema-v2 trainable state mismatch: "
                f"missing={sorted(expected - received_names)}, "
                f"unexpected={sorted(received_names - expected)}"
            )

        with torch.no_grad():
            for name in sorted(expected):
                value = received[name]
                parameter = named_parameters[name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Checkpoint value for {name} is not a tensor")
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {name}: "
                        f"saved={tuple(value.shape)}, current={tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.assert_trainable_contract()

    def load_state_dict_(self, state_dict):
        if int(state_dict.get("schema_version", 1)) == 2:
            self._load_trainable_state_dict_v2(state_dict)
            return

        # Historical schema-v1 checkpoints retain their original loader even
        # when the current run writes schema v2.
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

    def _sample_training_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_train_timesteps = int(self.noise_scheduler.config.num_train_timesteps)
        if self.ba_training_timestep_policy == "uniform_all":
            # Preserve the exact historical scalar-per-batch behavior.
            scalar = torch.randint(
                0,
                num_train_timesteps,
                (1,),
                device=device,
            ).long()
            return scalar.repeat(batch_size)

        if self.ba_training_timestep_policy != "inference_active":
            raise RuntimeError(
                f"Unhandled timestep policy {self.ba_training_timestep_policy!r}"
            )
        if not 0 <= self.branched_attn_start_step < self.num_inference_steps:
            raise ValueError(
                "branched_attn_start_step must be within the inference schedule: "
                f"start={self.branched_attn_start_step}, "
                f"steps={self.num_inference_steps}"
            )

        # 2 Aug 2026 - AICODE-NOTE: BA-v2 samples only DDIM timesteps at which
        # the fixed validation protocol actually enables branched attention.
        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps, device=device)
        active = scheduler.timesteps[self.branched_attn_start_step :]
        if active.numel() == 0:
            raise RuntimeError("Inference-active BA timestep set is empty")
        indices = torch.randint(
            0,
            active.numel(),
            (batch_size,),
            device=device,
        )
        return active.index_select(0, indices).long()

    @staticmethod
    def _reference_prediction_delta_ratio(
        correct: torch.Tensor,
        wrong: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.detach().float()
        if mask.shape[-2:] != correct.shape[-2:]:
            mask = F.interpolate(mask, size=correct.shape[-2:], mode="nearest")
        if mask.shape[0] != correct.shape[0]:
            if correct.shape[0] % mask.shape[0] != 0:
                raise RuntimeError("BA prediction-delta mask batch mismatch")
            mask = mask.repeat(correct.shape[0] // mask.shape[0], 1, 1, 1)
        denom = (
            mask.sum(dim=(1, 2, 3)) * correct.shape[1]
        ).clamp_min(1.0)
        correct_energy = (
            correct.detach().float().square() * mask
        ).sum(dim=(1, 2, 3)) / denom
        delta_energy = (
            (correct.detach().float() - wrong.detach().float()).square() * mask
        ).sum(dim=(1, 2, 3)) / denom
        return (
            delta_energy.clamp_min(0.0).sqrt()
            / correct_energy.clamp_min(1.0e-12).sqrt()
        ).mean()

    def _identity_aux_weight(self, global_step: int) -> float:
        if global_step < self.identity_aux_ramp_start_step:
            return 0.0
        progress = min(
            1.0,
            (global_step - self.identity_aux_ramp_start_step)
            / (
                self.identity_aux_ramp_end_step
                - self.identity_aux_ramp_start_step
            ),
        )
        return self.identity_aux_max_weight * progress

    def _face_crop_for_identity_proxy(
        self,
        image: torch.Tensor,
        bbox: Sequence[float],
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        pad_x = (x1 - x0) * self.identity_aux_crop_padding
        pad_y = (y1 - y0) * self.identity_aux_crop_padding
        x0 = max(0, min(width - 1, int(np.floor(x0 - pad_x))))
        y0 = max(0, min(height - 1, int(np.floor(y0 - pad_y))))
        x1 = max(x0 + 1, min(width, int(np.ceil(x1 + pad_x))))
        y1 = max(y0 + 1, min(height, int(np.ceil(y1 + pad_y))))
        crop = image[..., y0:y1, x0:x1]
        return F.interpolate(
            crop,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _predicted_x0_identity_auxiliary(
        self,
        *,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        pixel_values: torch.Tensor,
        face_bbox: Sequence[Sequence[float]],
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = noise_pred.float().new_tensor(0.0)
        weight = self._identity_aux_weight(global_step)
        if (
            not self.identity_aux_enabled
            or weight <= 0.0
            or global_step % self.identity_aux_cadence != 0
        ):
            return zero, zero, zero
        eligible = torch.nonzero(
            timesteps <= self.identity_aux_max_timestep,
            as_tuple=False,
        ).flatten()
        if eligible.numel() == 0:
            return zero, zero, zero
        index = int(eligible[0].item())
        timestep = int(timesteps[index].item())
        alpha = self.noise_scheduler.alphas_cumprod[timestep].to(
            device=noise_pred.device,
            dtype=torch.float32,
        )
        predicted_x0 = (
            noisy_latents[index : index + 1].float()
            - (1.0 - alpha).sqrt() * noise_pred[index : index + 1].float()
        ) / alpha.sqrt().clamp_min(1.0e-6)
        decoded = self.vae.decode(
            (
                predicted_x0
                / float(self.vae.config.scaling_factor)
            ).to(dtype=self.vae.dtype),
            return_dict=False,
        )[0]
        predicted_face = self._face_crop_for_identity_proxy(
            decoded,
            face_bbox[index],
        )
        target_face = self._face_crop_for_identity_proxy(
            pixel_values[index : index + 1],
            face_bbox[index],
        )

        mean = torch.tensor(
            self.id_image_processor.image_mean,
            device=predicted_face.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.id_image_processor.image_std,
            device=predicted_face.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        def normalize(image: torch.Tensor) -> torch.Tensor:
            image = (image.float().clamp(-1.0, 1.0) + 1.0) * 0.5
            return ((image - mean) / std).to(self.id_encoder.dtype)

        predicted_embedding = self.id_encoder.vision_model(
            normalize(predicted_face)
        )[1].float()
        with torch.no_grad():
            target_embedding = self.id_encoder.vision_model(
                normalize(target_face)
            )[1].float()
        identity_loss = 1.0 - F.cosine_similarity(
            predicted_embedding,
            target_embedding,
            dim=-1,
        ).mean()
        return (
            identity_loss,
            zero.new_tensor(weight),
            zero.new_tensor(1.0),
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
        reference_cache_key: Sequence[str] | None = None,
        identity_id: Sequence[str] | None = None,
        global_step: int = 0,
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

        timesteps = self._sample_training_timesteps(
            batch_size=batch_size,
            device=latents.device,
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
            reference_cache_keys=reference_cache_key,
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

        ### MEMO: INITIAL LORA UNet pass ###
        # model_pred = self.unet(
        #     noisy_model_input,
        #     timesteps,
        #     encoder_hidden_states=prompt_embeds,
        #     added_cond_kwargs=added_cond_kwargs,
        #     return_dict=False,
        # )[0]
        ### MEMO: INITIAL LORA UNet pass ###

        skip_text_only = self.train_ba_all_steps and self.skip_unused_text_conditioning
        if not skip_text_only:
            num_inference_steps = max(1, self.num_inference_steps)
            photomaker_start_ratio = float(self.photomaker_start_step) / float(
                num_inference_steps
            )
            branched_start_ratio = float(self.branched_attn_start_step) / float(
                num_inference_steps
            )
            if not self.train_ba_all_steps:
                if timesteps.unique().numel() != 1:
                    raise RuntimeError(
                        "Mixed per-sample timesteps require train_ba_all_steps=true"
                    )
                denoise_progress = 1.0 - (
                    float(timesteps[0].item())
                    / float(self.noise_scheduler.config.num_train_timesteps - 1)
                )

            text_only_prompts = []
            trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
            for prompt in prompts:
                tokens_text_only = self.tokenizer.encode(prompt, add_special_tokens=False)
                if trigger_word_token in tokens_text_only:
                    tokens_text_only.remove(trigger_word_token)
                text_only_prompts.append(
                    self.tokenizer.decode(tokens_text_only, add_special_tokens=False)
                )

            prompt_embeds_text_only, pooled_prompt_embeds_text_only = self.encode_prompt(
                prompt=text_only_prompts,
                do_cfg=False,
            )
            prompt_embeds_text_only = prompt_embeds_text_only.to(
                device=self.device, dtype=self.unet.dtype
            )
            pooled_prompt_embeds_text_only = pooled_prompt_embeds_text_only.to(
                device=self.device, dtype=self.unet.dtype
            )

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

        # Read matched-forward telemetry before an optional counterfactual
        # pass. The latter is deliberately suppressed so it cannot overwrite
        # the actual production-path sample.
        ba_telemetry = collect_branched_telemetry(self)
        wrong_spatial_reference_pred = None
        reference_shuffle_applied = noise_pred.new_tensor(0.0)
        reference_prediction_delta_ratio = noise_pred.new_tensor(0.0)
        shuffle_probability = self.ba_spatial_reference_shuffle_probability
        if (
            self.training
            and shuffle_probability > 0.0
            and batch_size > 1
            and torch.rand((), device=latents.device).item() < shuffle_probability
        ):
            if identity_id is None:
                identities = None
            elif isinstance(identity_id, str):
                if batch_size != 1:
                    raise ValueError(
                        "identity_id must contain one value per training sample"
                    )
                identities = [identity_id]
            else:
                identities = [str(value) for value in identity_id]
                if len(identities) != batch_size:
                    raise ValueError(
                        "identity_id batch mismatch: "
                        f"received={len(identities)}, expected={batch_size}"
                    )
            permutation = None
            for shift in range(1, batch_size):
                candidate = torch.roll(
                    torch.arange(batch_size, device=latents.device),
                    shifts=shift,
                )
                if identities is None or all(
                    identities[index] != identities[int(candidate[index].item())]
                    for index in range(batch_size)
                ):
                    permutation = candidate
                    break
            if permutation is not None:
                # Keep PhotoMaker prompt/ID conditioning, target noise, and
                # timestep fixed; shuffle only spatial reference latents/masks.
                paired_reference_noise = getattr(self, "_ref_noise", None)
                previous_suppression = bool(
                    getattr(self, "_ba_suppress_telemetry", False)
                )
                self._ba_suppress_telemetry = True

                def _wrong_reference_forward():
                    return run_branched_forward_pass(
                        self,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        mask4=mask4,
                        mask4_ref=mask4_ref.index_select(0, permutation),
                        reference_latents=reference_latents.index_select(
                            0, permutation
                        ),
                        face_prompt_embeds=face_prompt_embeds,
                        class_tokens_mask=class_tokens_mask,
                        id_features=id_features,
                        reference_noise=paired_reference_noise,
                    )

                try:
                    if self.ba_reference_loss_mode == "differentiable_rank":
                        wrong_spatial_reference_pred = _wrong_reference_forward()
                    else:
                        # Preserve residual-v2's exact detached diagnostic path.
                        with torch.no_grad():
                            wrong_spatial_reference_pred = _wrong_reference_forward()
                finally:
                    self._ba_suppress_telemetry = previous_suppression

                if paired_reference_noise is not None and getattr(
                    self, "_ref_noise", None
                ) is not paired_reference_noise:
                    raise RuntimeError(
                        "Correct/wrong BA forwards did not reuse reference noise"
                    )
                reference_prediction_delta_ratio = (
                    self._reference_prediction_delta_ratio(
                        noise_pred,
                        wrong_spatial_reference_pred,
                        mask4,
                    ).to(device=noise_pred.device)
                )
                reference_shuffle_applied = noise_pred.new_tensor(1.0)

        identity_aux_loss, identity_aux_weight, identity_aux_applied = (
            self._predicted_x0_identity_auxiliary(
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                timesteps=timesteps,
                pixel_values=pixel_values,
                face_bbox=face_bbox,
                global_step=int(global_step),
            )
        )

        return {
            'model_pred': noise_pred,
            'target': noise,
            'pred_wrong_spatial_ref': wrong_spatial_reference_pred,
            'reference_shuffle_applied': reference_shuffle_applied,
            'reference_prediction_delta_ratio': reference_prediction_delta_ratio,
            'ba_telemetry': ba_telemetry,
            'identity_aux_loss': identity_aux_loss,
            'identity_aux_weight': identity_aux_weight,
            'identity_aux_applied': identity_aux_applied,
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
        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

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
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids

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

    def clear_runtime_caches(self):
        cache = getattr(self, "_conditioning_cache", None)
        if cache is not None:
            cache.clear()
    ##### BRANCHED ATTENTION - HELPER UTILS #####
