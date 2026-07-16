from __future__ import annotations

import math
import re
from typing import Mapping, Optional, Sequence

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
    set_branched_training_mode,
    attach_inactive_branched_params,
    ensure_branched_after_eval as ensure_branched_after_eval_helper,
    select_wrong_identity_features,
)
from .identity_memory import CanonicalFacePartResampler, FacePatchIdentityResampler
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
        ba_ca_train_mode: str = "all",
        ba_train_top_k: float = 1.0,
        ba_patch_top_k: float = 1.0,
        non_ba_train: bool = False,
        train_ba_all_steps: bool = False,
        ba_enable_runtime_sa_knobs: bool = False,
        ba_train_sa_id_embed_proj: bool = False,
        ba_face_fusion_mode: str = "legacy",
        ba_face_fusion_gate_init: float = 0.25,
        ba_face_fusion_gate_max: float = 1.0,
        id_alpha: float = 0.3,             # strength of ID embedding injection in BranchedAttnProcessor
        use_id_embeds: bool = True,        # toggle ID embedding injection (controls id_to_hidden usage)
        ba_uncond_face_fix: bool = False,  # F1: keep plain negative prompt for the uncond face branch under CFG
        ba_face_prompt_mode: str = "id_only",  # B1: face-branch prompt: id_only (legacy) | full_boosted
        use_id_loss: bool = False,         # ID loss: cosine-distance identity loss on the decoded face (off by default)
        id_loss_weight: float = 0.5,       # weight of the ID loss added to the diffusion loss
        id_loss_max_timestep: int = 500,   # only apply ID loss when the sampled timestep <= this (x0 is meaningful)
        id_loss_face_size: int = 160,      # face crop size fed to the recognizer
        id_loss_identity_source: str = "ground_truth_target",
        ba_sa_mode: str = "legacy",
        ba_face_kv_mode: str = "zero_masked_full",
        ba_face_roi_size: int = 4,
        ba_ca_mode: str = "legacy_ref_branch",
        ba_identity_token_count: int = 4,
        ba_identity_memory_mode: str = "mean_plus_basis",
        ba_identity_image_mode: str = "full_reference",
        ba_identity_crop_padding: float = 0.10,
        ba_identity_patch_padding: float = 0.0,
        ba_identity_resampler_hidden_dim: int = 256,
        ba_identity_dependence_mode: str = "none",
        ba_identity_dependence_weight: float = 0.25,
        ba_identity_dependence_margin: float = 0.02,
        ba_identity_dependence_global_negatives: bool = True,
        ba_negative_strategy: str = "least_similar",
        ba_negative_target_similarity: float = 0.30,
        ba_causal_identity_weight: float = 0.25,
        ba_causal_margin: float = 0.02,
        ba_causal_direct_weight: float = 0.25,
        ba_causal_wrong_weight: float = 1.0,
        ba_causal_cross_weight: float = 0.5,
        ba_causal_preservation_weight: float = 0.1,
        ba_causal_structure_weight: float = 0.1,
        ba_causal_max_timestep: int = 300,
        ba_causal_every_n_steps: int = 1,
        ba_causal_require_landmarks: bool = False,
        ba_pm_preservation_mode: str = "none",
        ba_hard_mask_resize: str = "legacy_threshold",
        ba_target_mask_fail_closed: bool = False,
        disable_reference_spatial_branch: bool = False,
        ba_skip_inactive_optimizer_decay: bool = False,
        ba_fix_tensor_ref_resolution: bool = False,
        ba_ca_layer_allowlist: Optional[Sequence[str]] = None,
        ba_trainable_dtype: str = "model",
        ba_face_gate_mode: str = "legacy_scalar",
        ba_face_gate_init: float = 1.0,
        ba_face_gate_max: float = 1.0,
        ba_face_gate_init_overrides: Optional[Mapping[str, float]] = None,
        ba_pm_identity_context_scale: float = 1.0,
        ba_pm_identity_context_scale_overrides: Optional[Mapping[str, float]] = None,
        ba_cfg_composition: str = "legacy_guided",
        ba_residual_scale: float = 1.0,
        ba_post_cfg_guidance_scale: bool = False,
        ba_sync_timestep: bool = False,
        ba_require_reference_face: bool = False,
        ba_identity_canonical_size: int = 224,
        ba_strict_checkpoint_restore: bool = False,
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
        self.ba_ca_train_mode = str(ba_ca_train_mode or "all").lower()
        self.ba_train_top_k = float(ba_train_top_k)
        self.ba_patch_top_k = float(ba_patch_top_k)
        self.non_ba_train = bool(non_ba_train)
        self.train_ba_all_steps = bool(train_ba_all_steps)
        self.ba_enable_runtime_sa_knobs = bool(ba_enable_runtime_sa_knobs)
        self.ba_train_sa_id_embed_proj = bool(ba_train_sa_id_embed_proj)
        self.ba_face_fusion_mode = str(ba_face_fusion_mode or "legacy").lower()
        self.ba_face_fusion_gate_init = float(ba_face_fusion_gate_init)
        self.ba_face_fusion_gate_max = float(ba_face_fusion_gate_max)
        # ID loss (identity-supervised training). Off by default -> zero overhead / behaviour change.
        self.use_id_loss = bool(use_id_loss)
        self.id_loss_weight = float(id_loss_weight)
        self.id_loss_max_timestep = int(id_loss_max_timestep)
        self.id_loss_face_size = int(id_loss_face_size)
        self.id_loss_identity_source = str(id_loss_identity_source or "ground_truth_target").lower()
        if self.id_loss_identity_source not in {"ground_truth_target", "reference"}:
            raise ValueError(f"Unknown id_loss_identity_source: {self.id_loss_identity_source}")
        self._id_loss_net = None  # lazily built on first use
        self.ba_sa_mode = str(ba_sa_mode or "legacy").lower()
        self.ba_face_kv_mode = str(ba_face_kv_mode or "zero_masked_full").lower()
        self.ba_face_roi_size = max(1, int(ba_face_roi_size))
        self.ba_ca_mode = str(ba_ca_mode or "legacy_ref_branch").lower()
        self.ba_identity_token_count = max(1, int(ba_identity_token_count))
        self.ba_identity_memory_mode = str(ba_identity_memory_mode or "mean_plus_basis").lower()
        self.ba_identity_image_mode = str(ba_identity_image_mode or "full_reference").lower()
        self.ba_identity_crop_padding = float(ba_identity_crop_padding)
        self.ba_identity_patch_padding = float(ba_identity_patch_padding)
        self.ba_identity_resampler_hidden_dim = int(ba_identity_resampler_hidden_dim)
        self.ba_identity_dependence_mode = str(ba_identity_dependence_mode or "none").lower()
        self.ba_identity_dependence_weight = float(ba_identity_dependence_weight)
        self.ba_identity_dependence_margin = float(ba_identity_dependence_margin)
        self.ba_identity_dependence_global_negatives = bool(ba_identity_dependence_global_negatives)
        self.ba_negative_strategy = str(ba_negative_strategy or "least_similar").lower()
        self.ba_negative_target_similarity = float(ba_negative_target_similarity)
        self.ba_causal_identity_weight = float(ba_causal_identity_weight)
        self.ba_causal_margin = float(ba_causal_margin)
        self.ba_causal_direct_weight = float(ba_causal_direct_weight)
        self.ba_causal_wrong_weight = float(ba_causal_wrong_weight)
        self.ba_causal_cross_weight = float(ba_causal_cross_weight)
        self.ba_causal_preservation_weight = float(ba_causal_preservation_weight)
        self.ba_causal_structure_weight = float(ba_causal_structure_weight)
        self.ba_causal_max_timestep = int(ba_causal_max_timestep)
        self.ba_causal_every_n_steps = max(1, int(ba_causal_every_n_steps))
        self.ba_causal_require_landmarks = bool(ba_causal_require_landmarks)
        self.ba_pm_preservation_mode = str(ba_pm_preservation_mode or "none").lower()
        self.ba_hard_mask_resize = str(ba_hard_mask_resize or "legacy_threshold").lower()
        self.ba_target_mask_fail_closed = bool(ba_target_mask_fail_closed)
        self.disable_reference_spatial_branch = bool(disable_reference_spatial_branch)
        self.ba_skip_inactive_optimizer_decay = bool(ba_skip_inactive_optimizer_decay)
        self.ba_fix_tensor_ref_resolution = bool(ba_fix_tensor_ref_resolution)
        self.ba_ca_layer_allowlist = (
            None if ba_ca_layer_allowlist is None
            else tuple(str(item) for item in ba_ca_layer_allowlist)
        )
        self.ba_trainable_dtype = str(ba_trainable_dtype or "model").lower()
        self.ba_face_gate_mode = str(ba_face_gate_mode or "legacy_scalar").lower()
        self.ba_face_gate_init = float(ba_face_gate_init)
        self.ba_face_gate_max = float(ba_face_gate_max)
        self.ba_face_gate_init_overrides = {
            str(pattern): float(value)
            for pattern, value in (ba_face_gate_init_overrides or {}).items()
        }
        self.ba_pm_identity_context_scale = float(ba_pm_identity_context_scale)
        self.ba_pm_identity_context_scale_overrides = {
            str(pattern): float(value)
            for pattern, value in (ba_pm_identity_context_scale_overrides or {}).items()
        }
        self.ba_cfg_composition = str(ba_cfg_composition or "legacy_guided").lower()
        self.ba_residual_scale = float(ba_residual_scale)
        self.ba_post_cfg_guidance_scale = bool(ba_post_cfg_guidance_scale)
        self.ba_sync_timestep = bool(ba_sync_timestep)
        self.ba_require_reference_face = bool(ba_require_reference_face)
        self.ba_identity_canonical_size = int(ba_identity_canonical_size)
        self.ba_strict_checkpoint_restore = bool(ba_strict_checkpoint_restore)
        if self.ba_pm_preservation_mode not in {"none", "hard_epsilon_merge"}:
            raise ValueError(f"Unknown ba_pm_preservation_mode: {self.ba_pm_preservation_mode}")
        if self.ba_identity_memory_mode not in {
            "mean_plus_basis",
            "qformer_tokens",
            "face_patch_resampler",
            "canonical_face_parts",
            "qformer_plus_canonical_parts",
        }:
            raise ValueError(f"Unknown ba_identity_memory_mode: {self.ba_identity_memory_mode}")
        if self.ba_identity_image_mode not in {"full_reference", "bbox_normalized"}:
            raise ValueError(f"Unknown ba_identity_image_mode: {self.ba_identity_image_mode}")
        if (
            self.ba_identity_memory_mode == "face_patch_resampler"
            and self.ba_identity_image_mode != "full_reference"
        ):
            raise ValueError("face_patch_resampler selects patches from the full reference image")
        if self.ba_identity_dependence_mode not in {
            "none", "paired_wrong_reference", "decoded_causal"
        }:
            raise ValueError(f"Unknown ba_identity_dependence_mode: {self.ba_identity_dependence_mode}")
        if self.ba_identity_dependence_weight < 0 or self.ba_identity_dependence_margin < 0:
            raise ValueError("Identity-dependence weight and margin must be non-negative")
        if self.ba_negative_strategy not in {"least_similar", "semi_hard"}:
            raise ValueError(f"Unknown ba_negative_strategy: {self.ba_negative_strategy}")
        if self.ba_trainable_dtype not in {"model", "fp32"}:
            raise ValueError(f"Unknown ba_trainable_dtype: {self.ba_trainable_dtype}")
        if self.ba_face_gate_mode not in {"legacy_scalar", "bounded_sigmoid"}:
            raise ValueError(f"Unknown ba_face_gate_mode: {self.ba_face_gate_mode}")
        if not 0.0 <= self.ba_pm_identity_context_scale <= 1.0:
            raise ValueError("ba_pm_identity_context_scale must be in [0, 1]")
        if any(
            not 0.0 <= value <= 1.0
            for value in self.ba_pm_identity_context_scale_overrides.values()
        ):
            raise ValueError(
                "ba_pm_identity_context_scale_overrides values must be in [0, 1]"
            )
        if self.ba_cfg_composition not in {"legacy_guided", "post_cfg_delta"}:
            raise ValueError(f"Unknown ba_cfg_composition: {self.ba_cfg_composition}")
        if (
            self.ba_cfg_composition == "post_cfg_delta"
            and self.ba_pm_preservation_mode != "hard_epsilon_merge"
        ):
            raise ValueError(
                "post_cfg_delta requires ba_pm_preservation_mode=hard_epsilon_merge"
            )
        self.ba_identity_resampler = None
        if self.ba_identity_memory_mode == "face_patch_resampler":
            self.ba_identity_resampler = FacePatchIdentityResampler(
                num_tokens=self.ba_identity_token_count,
                hidden_dim=self.ba_identity_resampler_hidden_dim,
            )
        elif self.ba_identity_memory_mode in {
            "canonical_face_parts", "qformer_plus_canonical_parts"
        }:
            self.ba_identity_resampler = CanonicalFacePartResampler(
                num_tokens=8,
                hidden_dim=self.ba_identity_resampler_hidden_dim,
            )
            expected_tokens = 8 if self.ba_identity_memory_mode == "canonical_face_parts" else 10
            if self.ba_identity_token_count != expected_tokens:
                raise ValueError(
                    f"{self.ba_identity_memory_mode} requires "
                    f"ba_identity_token_count={expected_tokens}"
                )
        if self.ba_identity_resampler is not None:
            memory_dtype = (
                torch.float32 if self.ba_trainable_dtype == "fp32" else self.weight_dtype
            )
            self.ba_identity_resampler.to(device=self.device, dtype=memory_dtype)
        self._causal_id_loss_net = None
        self._ba_active_this_batch = False
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
        if self.ba_identity_resampler is not None:
            memory_dtype = (
                torch.float32 if self.ba_trainable_dtype == "fp32" else self.weight_dtype
            )
            self.ba_identity_resampler.to(device=self.device, dtype=memory_dtype)
            self.ba_identity_resampler.requires_grad_(True)

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
        ca_lr_scale = float(config.get("ba_ca_lr_scale", 1.0))
        noise_wd = config.get("ba_noise_weight_decay", None)

        named = [(n, p) for n, p in self.unet.named_parameters() if p.requires_grad]
        identity_resampler_params = (
            [p for p in self.ba_identity_resampler.parameters() if p.requires_grad]
            if self.ba_identity_resampler is not None
            else []
        )

        def _with_identity_resampler(groups):
            if identity_resampler_params:
                groups.append({
                    "params": identity_resampler_params,
                    "lr": config.lr_for_lora,
                    "name": "ba_identity_resampler_params",
                })
            return groups

        def _is_noise_clone(name: str) -> bool:
            return ".processor." in name and ".noise_to_" in name

        def _is_ca_processor(name: str) -> bool:
            return ".attn2.processor." in name

        if (
            noise_lr_scale == 1.0
            and ca_lr_scale == 1.0
            and noise_wd is None
        ) or not any(_is_noise_clone(n) or _is_ca_processor(n) for n, _ in named):
            # Single group — bit-identical to the previous behaviour.
            return _with_identity_resampler([
                {"params": [p for _, p in named], "lr": config.lr_for_lora, "name": "lora_params"},
            ])

        groups = {}
        for name, param in named:
            is_noise = _is_noise_clone(name)
            is_ca = _is_ca_processor(name)
            if is_ca and is_noise:
                group_name = "ba_ca_noise_params"
            elif is_ca:
                group_name = "ba_ca_params"
            elif is_noise:
                group_name = "ba_noise_params"
            else:
                group_name = "lora_params"
            lr = float(config.lr_for_lora)
            if is_noise:
                lr *= noise_lr_scale
            if is_ca:
                lr *= ca_lr_scale
            if group_name not in groups:
                groups[group_name] = {"params": [], "lr": lr, "name": group_name}
                if is_noise and noise_wd is not None:
                    groups[group_name]["weight_decay"] = float(noise_wd)
            groups[group_name]["params"].append(param)
        return _with_identity_resampler([group for group in groups.values() if group["params"]])
        ##### BRANCHED ATTENTION - NEW BLOCK 2 #####

    def _ba_architecture_manifest(self):
        return {
            "ba_sa_mode": self.ba_sa_mode,
            "ba_ca_mode": self.ba_ca_mode,
            "ba_ca_train_mode": self.ba_ca_train_mode,
            "branched_attn_weight_mode": self.branched_attn_weight_mode,
            "branched_attn_new_weight_kind": self.branched_attn_new_weight_kind,
            "train_branched_ca_lora": self.train_branched_ca_lora,
            "face_embed_strategy": self.face_embed_strategy,
            "use_id_embeds": self.use_id_embeds,
            "ba_uncond_face_fix": self.ba_uncond_face_fix,
            "ba_face_prompt_mode": self.ba_face_prompt_mode,
            "ba_identity_memory_mode": self.ba_identity_memory_mode,
            "ba_identity_token_count": self.ba_identity_token_count,
            "ba_identity_image_mode": self.ba_identity_image_mode,
            "ba_identity_crop_padding": self.ba_identity_crop_padding,
            "ba_identity_patch_padding": self.ba_identity_patch_padding,
            "ba_identity_resampler_hidden_dim": self.ba_identity_resampler_hidden_dim,
            "ba_identity_canonical_size": self.ba_identity_canonical_size,
            "ba_ca_layer_allowlist": list(self.ba_ca_layer_allowlist or []),
            "ba_trainable_dtype": self.ba_trainable_dtype,
            "ba_face_gate_mode": self.ba_face_gate_mode,
            "ba_face_gate_init": self.ba_face_gate_init,
            "ba_face_gate_max": self.ba_face_gate_max,
            "ba_face_gate_init_overrides": dict(self.ba_face_gate_init_overrides),
            "ba_pm_identity_context_scale": self.ba_pm_identity_context_scale,
            "ba_pm_identity_context_scale_overrides": dict(
                self.ba_pm_identity_context_scale_overrides
            ),
            "ba_cfg_composition": self.ba_cfg_composition,
            "ba_residual_scale": self.ba_residual_scale,
            "ba_pm_preservation_mode": self.ba_pm_preservation_mode,
            "ba_hard_mask_resize": self.ba_hard_mask_resize,
            "ba_target_mask_fail_closed": self.ba_target_mask_fail_closed,
            "ba_require_reference_face": self.ba_require_reference_face,
            "disable_reference_spatial_branch": self.disable_reference_spatial_branch,
        }

    def get_state_dict(self):
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        state = {
            'lora_weights': lora_weights,
            'ba_architecture': self._ba_architecture_manifest(),
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
        if self.ba_identity_resampler is not None:
            state["ba_identity_resampler"] = self.ba_identity_resampler.state_dict()
        return state

    def load_state_dict_(self, state_dict):
        saved_architecture = state_dict.get("ba_architecture")
        if self.ba_strict_checkpoint_restore and saved_architecture is None:
            raise ValueError("Strict BA checkpoint restore requires ba_architecture")
        if self.ba_strict_checkpoint_restore:
            current = self._ba_architecture_manifest()
            mismatches = {
                key: (saved_architecture.get(key), current.get(key))
                for key in current
                if saved_architecture.get(key) != current.get(key)
            }
            if mismatches:
                raise ValueError(f"BA checkpoint architecture mismatch: {mismatches}")
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # In newer peft versions this is an empty list when there are no unexpected keys
            assert not unexpected_keys, unexpected_keys
        saved_processors = state_dict.get("attn_processors", {})
        if self.ba_strict_checkpoint_restore:
            expected_names = {
                name
                for name, proc in self.unet.attn_processors.items()
                if isinstance(proc, torch.nn.Module)
                and any(param.requires_grad for param in proc.parameters())
            }
            if set(saved_processors) != expected_names:
                raise ValueError(
                    "BA checkpoint processor-name mismatch: "
                    f"saved={sorted(saved_processors)}, expected={sorted(expected_names)}"
                )
        for name, sd in saved_processors.items():
            proc = self.unet.attn_processors.get(name)
            if proc is None or not hasattr(proc, "load_state_dict"):
                if self.ba_strict_checkpoint_restore:
                    raise ValueError(f"Checkpoint processor is unavailable: {name}")
                continue
            if self.ba_strict_checkpoint_restore:
                trainable = tuple(n for n, p in proc.named_parameters() if p.requires_grad)
                expected_keys = {
                    key for key in proc.state_dict()
                    if any(key == n or key.startswith(n + ".") for n in trainable)
                }
                if set(sd) != expected_keys:
                    raise ValueError(
                        f"Checkpoint tensor mismatch for {name}: "
                        f"saved={sorted(sd)}, expected={sorted(expected_keys)}"
                    )
            proc.load_state_dict(sd, strict=False)
        resampler_state = state_dict.get("ba_identity_resampler")
        if (
            self.ba_strict_checkpoint_restore
            and (resampler_state is None) != (self.ba_identity_resampler is None)
        ):
            raise ValueError(
                "BA checkpoint resampler mismatch: "
                f"saved={resampler_state is not None}, "
                f"expected={self.ba_identity_resampler is not None}"
            )
        if resampler_state is not None:
            if self.ba_identity_resampler is None:
                raise ValueError(
                    "Checkpoint contains a BA identity resampler, but the current config does not enable it"
                )
            self.ba_identity_resampler.load_state_dict(resampler_state, strict=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
        face_bbox_ref: Sequence[Sequence[float]] | None = None,
        identity_id: Sequence[str] | None = None,
        do_cfg: bool = False,
        *args,
        **kwargs,
    ):
        del do_cfg  # classifier-free guidance is not used during training
        self._ba_active_this_batch = False

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
        if (
            self.ba_sync_timestep
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.broadcast(t_scalar, src=0)
        timesteps = t_scalar.repeat(batch_size)
        denoise_progress = 1.0 - (
            float(t_scalar.item()) / float(self.noise_scheduler.config.num_train_timesteps - 1)
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        batch_idx = int(kwargs.get("batch_idx", 0))
        causal_step = (
            self.ba_identity_dependence_mode == "decoded_causal"
            and int(t_scalar.item()) <= self.ba_causal_max_timestep
            and batch_idx % self.ba_causal_every_n_steps == 0
        )

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
            collect_identity_metadata=causal_step,
        )
        ##### BRANCHED ATTENTION - NEW BLOCK 4 #####

        wrong_id_features = None
        wrong_identity_indices = None
        prepared_reference_embeddings = None
        if self.ba_identity_dependence_mode == "paired_wrong_reference":
            if id_features is None:
                raise ValueError("paired_wrong_reference requires identity memory features")
            wrong_id_features = select_wrong_identity_features(
                id_features,
                global_negatives=self.ba_identity_dependence_global_negatives,
            )
        elif causal_step:
            if id_features is None:
                raise ValueError("decoded_causal requires identity memory features")
            wrong_id_features, wrong_identity_indices = select_wrong_identity_features(
                id_features,
                selector_features=getattr(self, "_ba_identity_selector_features", None),
                identity_ids=identity_id,
                global_negatives=self.ba_identity_dependence_global_negatives,
                strategy=self.ba_negative_strategy,
                target_similarity=self.ba_negative_target_similarity,
                return_indices=True,
            )
            prepared_reference_embeddings = self._prepare_causal_reference_embeddings(
                ref_images,
                face_bbox_ref,
            )

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
        self._ba_text_only_prompt_embeds = prompt_embeds_text_only

        def _run_active_predictions():
            common = dict(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                mask4=mask4,
                mask4_ref=mask4_ref,
                reference_latents=reference_latents,
                face_prompt_embeds=face_prompt_embeds,
                class_tokens_mask=class_tokens_mask,
            )
            if wrong_id_features is None:
                return (
                    run_branched_forward_pass(self, id_features=id_features, **common),
                    None,
                    None,
                )
            correct_pred, photomaker_pred = run_branched_forward_pass(
                self,
                id_features=id_features,
                return_photomaker_pred=True,
                **common,
            )
            wrong_pred = run_branched_forward_pass(
                self,
                id_features=wrong_id_features,
                photomaker_pred=photomaker_pred,
                **common,
            )
            return correct_pred, wrong_pred, photomaker_pred

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
            noise_pred, wrong_identity_pred, null_identity_pred = _run_active_predictions()
        elif denoise_progress < photomaker_start_ratio:
            text_only_kwargs = {
                "text_embeds": pooled_prompt_embeds_text_only,
                "time_ids": add_time_ids.to(device=self.device, dtype=self.unet.dtype),
            }
            set_branched_training_mode(self, branched_active=False)
            try:
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds_text_only,
                    added_cond_kwargs=text_only_kwargs,
                    return_dict=False,
                )[0]
            finally:
                set_branched_training_mode(self, branched_active=True)
            noise_pred = attach_inactive_branched_params(self, noise_pred)
        elif denoise_progress < branched_start_ratio:
            set_branched_training_mode(self, branched_active=False)
            try:
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            finally:
                set_branched_training_mode(self, branched_active=True)
            noise_pred = attach_inactive_branched_params(self, noise_pred)
        else:
            ##### BRANCHED ATTENTION - FORWARD PASS #####
            """FORWARD PASS: run branched prediction via helper wrapper around `two_branch_predict`."""
            noise_pred, wrong_identity_pred, null_identity_pred = _run_active_predictions()
            ##### BRANCHED ATTENTION - FORWARD PASS #####

        out = {
            'model_pred': noise_pred,
            'target': noise,
        }
        if (
            self.ba_identity_dependence_mode == "paired_wrong_reference"
            and 'wrong_identity_pred' in locals()
            and wrong_identity_pred is not None
        ):
            out['wrong_identity_pred'] = wrong_identity_pred
        if (
            self.ba_identity_dependence_mode == "decoded_causal"
            and 'wrong_identity_pred' in locals()
            and wrong_identity_pred is not None
            and null_identity_pred is not None
            and wrong_identity_indices is not None
        ):
            causal = self._compute_causal_identity_loss(
                correct_pred=noise_pred,
                null_pred=null_identity_pred,
                wrong_pred=wrong_identity_pred,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                face_bbox=face_bbox,
                ref_images=ref_images,
                face_bbox_ref=face_bbox_ref,
                wrong_identity_indices=wrong_identity_indices,
                prepared_reference_embeddings=prepared_reference_embeddings,
            )
            out["causal_identity_loss"] = causal.pop("loss")
            for key, value in causal.items():
                out[f"causal_identity_{key}"] = value

        # ID loss (identity-supervised): only when enabled AND the sampled timestep is low enough
        # that the predicted x0 is meaningful. t is shared across the batch (see t_scalar above),
        # so this gates the whole step -> no VAE decode on high-noise steps.
        if self.use_id_loss and int(t_scalar.item()) <= self.id_loss_max_timestep:
            out['id_loss'] = self._compute_id_loss(
                noise_pred=noise_pred,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                pixel_values=pixel_values,
                face_bbox=face_bbox,
                ref_images=ref_images,
                face_bbox_ref=face_bbox_ref,
            )
        return out

    def _prediction_to_x0(self, prediction, noisy_latents, timesteps):
        abar = self.noise_scheduler.alphas_cumprod.to(prediction.device)[timesteps].float()
        abar = abar.view(-1, 1, 1, 1)
        return (
            noisy_latents.float() - (1.0 - abar).sqrt() * prediction.float()
        ) / abar.sqrt().clamp_min(1e-4)

    def _decode_x0(self, x0):
        return self.vae.decode(
            (x0 / self.vae.config.scaling_factor).to(self.vae.dtype)
        ).sample

    def _causal_identity_loss_module(self):
        if self._causal_id_loss_net is None:
            from src.loss.id_loss import CausalIdentityLoss

            self._causal_id_loss_net = CausalIdentityLoss(
                face_size=self.id_loss_face_size,
                device=self.device,
            )
        return self._causal_id_loss_net

    def _prepare_causal_reference_embeddings(self, ref_images, face_bbox_ref):
        ref_bboxes = (
            face_bbox_ref if isinstance(face_bbox_ref, (list, tuple)) else [face_bbox_ref]
        )
        with torch.autocast(
            device_type=self.device.type if hasattr(self.device, "type") else "cuda",
            enabled=False,
        ):
            return self._causal_identity_loss_module().prepare_reference_embeddings(
                ref_images,
                getattr(self, "_ba_reference_landmarks", None),
                ref_bboxes,
                device=self.device,
                global_negatives=self.ba_identity_dependence_global_negatives,
            )

    def _compute_causal_identity_loss(
        self,
        *,
        correct_pred,
        null_pred,
        wrong_pred,
        noisy_latents,
        timesteps,
        face_bbox,
        ref_images,
        face_bbox_ref,
        wrong_identity_indices,
        prepared_reference_embeddings,
    ):
        causal_id_loss = self._causal_identity_loss_module()

        correct_x0 = self._prediction_to_x0(correct_pred, noisy_latents, timesteps)
        wrong_x0 = self._prediction_to_x0(wrong_pred, noisy_latents, timesteps)
        with torch.no_grad():
            null_x0 = self._prediction_to_x0(null_pred, noisy_latents, timesteps)

        tile = hasattr(self.vae, "enable_tiling") and hasattr(self.vae, "disable_tiling")
        slicing = hasattr(self.vae, "enable_slicing") and hasattr(self.vae, "disable_slicing")
        if tile:
            self.vae.enable_tiling()
        if slicing:
            self.vae.enable_slicing()
        try:
            correct_images = self._decode_x0(correct_x0)
            wrong_images = self._decode_x0(wrong_x0)
            with torch.no_grad():
                null_images = self._decode_x0(null_x0)
        finally:
            if tile:
                self.vae.disable_tiling()
            if slicing:
                self.vae.disable_slicing()

        bboxes = face_bbox if isinstance(face_bbox, (list, tuple)) else [face_bbox]
        ref_bboxes = (
            face_bbox_ref if isinstance(face_bbox_ref, (list, tuple)) else [face_bbox_ref]
        )
        with torch.autocast(
            device_type=self.device.type if hasattr(self.device, "type") else "cuda",
            enabled=False,
        ):
            return causal_id_loss(
                correct_images.float(),
                null_images.float(),
                wrong_images.float(),
                target_landmarks=getattr(self, "_ba_target_landmarks", None),
                target_bboxes=bboxes,
                reference_images=ref_images,
                reference_landmarks=getattr(self, "_ba_reference_landmarks", None),
                reference_bboxes=ref_bboxes,
                wrong_indices=wrong_identity_indices,
                global_negatives=self.ba_identity_dependence_global_negatives,
                margin=self.ba_causal_margin,
                direct_weight=self.ba_causal_direct_weight,
                wrong_weight=self.ba_causal_wrong_weight,
                cross_weight=self.ba_causal_cross_weight,
                preservation_weight=self.ba_causal_preservation_weight,
                structure_weight=self.ba_causal_structure_weight,
                prepared_reference_embeddings=prepared_reference_embeddings,
            )

    def _compute_id_loss(
        self,
        noise_pred,
        noisy_latents,
        timesteps,
        pixel_values,
        face_bbox,
        ref_images,
        face_bbox_ref,
    ):
        """Cosine-distance identity loss between the generated face (decoded from the predicted
        x0) and the ground-truth face (from pixel_values), both cropped at face_bbox.
        Differentiable through the VAE decode + recognizer, so it trains the BA weights."""
        if self._id_loss_net is None:
            from src.loss.id_loss import IdentityLoss
            self._id_loss_net = IdentityLoss(face_size=self.id_loss_face_size, device=self.device)

        # x0 from epsilon-prediction: x0 = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)
        abar = self.noise_scheduler.alphas_cumprod.to(noise_pred.device)[timesteps].float()
        abar = abar.view(-1, 1, 1, 1)
        x0 = (noisy_latents.float() - (1.0 - abar).sqrt() * noise_pred.float()) / abar.sqrt().clamp_min(1e-4)

        # Decode to pixels in [-1, 1] (differentiable; VAE is frozen but the graph flows to x0).
        # Decode under VAE tiling+slicing, scoped to this call, so a full-res decode does not spike
        # VRAM (branched training already runs near the card limit). Tiling processes the image in
        # spatial tiles and slicing one batch item at a time -> much lower peak activation memory.
        # Restored afterwards so the training-forward vae.encode is unaffected.
        _tile = hasattr(self.vae, "enable_tiling") and hasattr(self.vae, "disable_tiling")
        _slice = hasattr(self.vae, "enable_slicing") and hasattr(self.vae, "disable_slicing")
        if _tile:
            self.vae.enable_tiling()
        if _slice:
            self.vae.enable_slicing()
        try:
            gen_images = self.vae.decode(
                (x0 / self.vae.config.scaling_factor).to(self.vae.dtype)
            ).sample
        finally:
            if _tile:
                self.vae.disable_tiling()
            if _slice:
                self.vae.disable_slicing()
        gt_images = pixel_values.to(device=gen_images.device, dtype=gen_images.dtype)

        # Normalize bbox to a per-sample list in pixel coords.
        bboxes = face_bbox if isinstance(face_bbox, (list, tuple)) else [face_bbox]

        # FaceNet runs in fp32 outside autocast for numerical stability.
        with torch.autocast(device_type=self.device.type if hasattr(self.device, "type") else "cuda", enabled=False):
            if self.id_loss_identity_source == "reference":
                id_loss = self._id_loss_net(
                    gen_images.float(),
                    gt_images.float(),
                    bboxes,
                    reference_images=ref_images,
                    reference_bboxes=face_bbox_ref,
                )
            else:
                id_loss = self._id_loss_net(gen_images.float(), gt_images.float(), bboxes)
        return id_loss

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

                use_coverage = self.ba_hard_mask_resize == "area_preserving"
                start_fn = math.floor if use_coverage else round
                end_fn = math.ceil if use_coverage else round
                x_start = max(0, min(self.target_size, int(start_fn(x0 * scale_w + pad_left))))
                x_end = max(0, min(self.target_size, int(end_fn(x1 * scale_w + pad_left))))
                y_start = max(0, min(self.target_size, int(start_fn(y0 * scale_h + pad_top))))
                y_end = max(0, min(self.target_size, int(end_fn(y1 * scale_h + pad_top))))

                if x_end <= x_start or y_end <= y_start:
                    mask.fill_(1.0)
                else:
                    mask[:, :, y_start:y_end, x_start:x_end] = 1.0

        if mask.shape[-2:] != latent_shape:
            if self.ba_hard_mask_resize == "area_preserving":
                mask = F.adaptive_max_pool2d(mask, output_size=latent_shape)
            else:
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
            if self.ba_target_mask_fail_closed:
                raise ValueError(f"Missing target face bbox: {bbox}")
            mask.fill_(1.0)
            return mask

        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            if self.ba_target_mask_fail_closed:
                raise ValueError(f"Invalid target face bbox: {bbox}")
            mask.fill_(1.0)
            return mask

        scale_w = latent_shape[1] / max(image_shape[1], 1)
        scale_h = latent_shape[0] / max(image_shape[0], 1)

        use_coverage = self.ba_hard_mask_resize == "area_preserving"
        x_start_fn = math.floor if use_coverage else round
        x_end_fn = math.ceil if use_coverage else round
        x_start = max(0, min(latent_shape[1], int(x_start_fn(x0 * scale_w))))
        x_end = max(0, min(latent_shape[1], int(x_end_fn(x1 * scale_w))))
        y_start = max(0, min(latent_shape[0], int(x_start_fn(y0 * scale_h))))
        y_end = max(0, min(latent_shape[0], int(x_end_fn(y1 * scale_h))))

        if x_end <= x_start or y_end <= y_start:
            if self.ba_target_mask_fail_closed:
                raise ValueError(
                    f"Target face bbox becomes empty at latent shape {latent_shape}: {bbox}"
                )
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
            if self.ba_fix_tensor_ref_resolution:
                oh, ow = ref_tensor.shape[-2:]
                scale = min(self.target_size / max(ow, 1), self.target_size / max(oh, 1))
                rw = max(8, int(round(ow * scale)) // 8 * 8)
                rh = max(8, int(round(oh * scale)) // 8 * 8)
                ref_tensor = F.interpolate(
                    ref_tensor.float(), size=(rh, rw), mode="bilinear", align_corners=False
                )
                pl = (self.target_size - rw) // 2
                pr = self.target_size - rw - pl
                pt = (self.target_size - rh) // 2
                pb = self.target_size - rh - pt
                ref_tensor = F.pad(ref_tensor, (pl, pr, pt, pb), value=0.0)
            elif ref_tensor.shape[-2:] != target_shape:
                ref_tensor = F.interpolate(
                    ref_tensor, size=target_shape, mode="bilinear", align_corners=False
                )
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
