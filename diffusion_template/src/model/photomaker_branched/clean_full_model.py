"""Audited hard-v1 model used by clean_full; June `lora2.py` stays intact."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from transformers import CLIPImageProcessor

import os
import random
from diffusers import DDIMScheduler
from src.model.photomaker_path import resolve_photomaker_path
from src.model.sdxl.original import SDXL

##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
"""Import branched-attention forward/patch helpers and PMv2 face-ID dependencies used by training."""
from .insightface_package import create_face_analyzer
from .clean_full_model_helpers import (
    collect_branched_telemetry,
    clear_lowband_contrastive_state,
    collect_frequency_schedule_anchor_loss,
    collect_frequency_surface_aux_loss,
    collect_visibility_ownership_v2_loss,
    collect_hardcase_aux_loss,
    collect_attention_ownership_loss,
    collect_lowband_contrastive_loss,
    collect_lowband_positive_loss,
    collect_roi_teacher_loss,
    prepare_branched_training_inputs,
    run_branched_forward_pass,
    ensure_branched_after_eval as ensure_branched_after_eval_helper,
)
from .model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .clean_full_excluded_objectives import ExcludedObjectiveCompatibilityMixin
from .clean_full_model_config import apply_clean_full_model_config
from .clean_full_model_contract import CleanFullModelContractMixin
##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####

### PhotomakerLora upgraged for BA ###
class PhotomakerBranchedLora(
    CleanFullModelContractMixin,
    ExcludedObjectiveCompatibilityMixin,
    SDXL,
):
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
        clean_full_config: Optional[dict] = None,
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

        # 22 Aug 2026 - AICODE-NOTE: the supported constructor exposes one
        # closed-schema mapping. Historical defaults stay quarantined in the
        # resolver so adding an experiment cannot silently regrow this method.
        self.scheduler = self.noise_scheduler
        self.pose_adapt_ratio = float(pose_adapt_ratio)
        self.ca_mixing_for_face = bool(ca_mixing_for_face)
        self.face_embed_strategy = (face_embed_strategy or "face").lower()
        self.train_branch_mode = (train_branch_mode or "both").lower()
        self.photomaker_start_step = int(photomaker_start_step)
        self.merge_start_step = int(merge_start_step)
        self.branched_attn_start_step = int(branched_attn_start_step)
        self.num_inference_steps = int(num_inference_steps)
        self.id_alpha = float(id_alpha)
        self.use_id_embeds = bool(use_id_embeds)
        self.train_ba_only = bool(train_ba_only)
        self.ba_weights_split = bool(ba_weights_split)
        self.use_attn_v2 = bool(use_attn_v2)
        self.branched_attn_weight_mode = (
            branched_attn_weight_mode or "shared"
        ).lower()
        self.branched_attn_new_weight_kind = (
            branched_attn_new_weight_kind or "full"
        ).lower()
        self.train_branched_ca_lora = bool(train_branched_ca_lora)
        self.ba_train_top_k = float(ba_train_top_k)
        self.ba_patch_top_k = float(ba_patch_top_k)
        self.non_ba_train = bool(non_ba_train)
        self.train_ba_all_steps = bool(train_ba_all_steps)
        apply_clean_full_model_config(self, clean_full_config)

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

    def _sample_training_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_train_timesteps = int(self.noise_scheduler.config.num_train_timesteps)
        if (
            self.ba_low_noise_id_reward_enabled
            and int(getattr(self, "_ba_current_global_step", -1))
            % self.identity_aux_cadence == 0
        ):
            scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
            scheduler.set_timesteps(self.num_inference_steps, device=device)
            active = scheduler.timesteps[-self.ba_low_noise_id_reward_last_steps :]
            if active.numel() != self.ba_low_noise_id_reward_last_steps:
                raise RuntimeError("Low-noise reward DDIM suffix is incomplete")
            indices = torch.randint(0, active.numel(), (batch_size,), device=device)
            return active.index_select(0, indices).long()
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
        face_bbox_ref: Sequence[Sequence[float]] | None = None,
        identity_face_bboxes_ref: Sequence[Sequence[Sequence[float]]] | None = None,
        reference_cache_key: Sequence[str] | None = None,
        identity_id: Sequence[str] | None = None,
        ba_occluder_mask: Sequence[torch.Tensor] | None = None,
        spatial_ref_images_alt: Sequence[Sequence[Image.Image]] | None = None,
        face_bbox_ref_alt: Sequence[Sequence[float]] | None = None,
        global_step: int = 0,
        do_cfg: bool = False,
        *args,
        **kwargs,
    ):
        del do_cfg  # classifier-free guidance is not used during training

        self._ba_current_global_step = int(global_step)
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

        self._ba_ownership_target_mask = None
        if ba_occluder_mask is not None:
            prepared_occluders = []
            for value in ba_occluder_mask:
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.ndim != 3 or tensor.shape[0] != 1:
                    raise ValueError(
                        "ba_occluder_mask items must have shape HxW or 1xHxW"
                    )
                prepared_occluders.append(tensor)
            ownership_mask = torch.stack(prepared_occluders).to(self.device)
            if ownership_mask.shape[-2:] != mask4.shape[-2:]:
                ownership_mask = F.interpolate(
                    ownership_mask,
                    size=mask4.shape[-2:],
                    mode="nearest",
                )
            self._ba_ownership_target_mask = ownership_mask.to(
                dtype=noisy_latents.dtype
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

        lowband_permutation = None
        lowband_sampled = False
        lowband_skipped_same_identity = False
        positive_sameid_sampled = False
        # 14 Aug 2026 - AICODE-NOTE: alternate-base validation deliberately
        # keeps modules in train mode under no_grad; CL29's sampled auxiliary
        # path must remain training-only and must never require dual refs there.
        if (
            self.training
            and torch.is_grad_enabled()
            and self.ba_frequency_lowband_contrastive_enabled
        ):
            if spatial_ref_images_alt is None or face_bbox_ref_alt is None:
                raise RuntimeError("CL29 requires distinct same-ID alternate references")
            sample_device = "cpu" if self.ba_frequency_lowband_sample_on_cpu else latents.device
            # 16 Aug 2026 - CPU sampling is opt-in for throughput runs and
            # avoids synchronizing the active CUDA stream before every step.
            lowband_sampled = torch.rand((), device=sample_device).item() < (
                self.ba_frequency_lowband_contrastive_probability
            )
            if lowband_sampled and batch_size > 1:
                if identity_id is None:
                    raise RuntimeError("CL29 requires identity_id for wrong-ID negatives")
                identities = (
                    [str(identity_id)]
                    if isinstance(identity_id, str)
                    else [str(value) for value in identity_id]
                )
                if len(identities) != batch_size:
                    raise RuntimeError("CL29 identity_id batch mismatch")
                for shift in range(1, batch_size):
                    candidate_indices = [
                        (index - shift) % batch_size for index in range(batch_size)
                    ]
                    if all(
                        identities[index] != identities[candidate_indices[index]]
                        for index in range(batch_size)
                    ):
                        lowband_permutation = torch.tensor(
                            candidate_indices,
                            device=latents.device,
                            dtype=torch.long,
                        )
                        break
                lowband_skipped_same_identity = lowband_permutation is None
            elif lowband_sampled:
                lowband_skipped_same_identity = True
        if (
            self.training
            and torch.is_grad_enabled()
            and self.ba_frequency_positive_sameid_enabled
        ):
            if spatial_ref_images_alt is None or face_bbox_ref_alt is None:
                raise RuntimeError("CL30 requires distinct same-ID alternate references")
            sample_device = (
                "cpu" if self.ba_frequency_positive_sameid_sample_on_cpu else latents.device
            )
            positive_sameid_sampled = torch.rand((), device=sample_device).item() < (
                self.ba_frequency_positive_sameid_probability
            )
        self._ba_lowband_capture_mode = (
            "anchor"
            if lowband_permutation is not None or positive_sameid_sampled
            else "off"
        )
        self._ba_lowband_negative_permutation = None
        patch_identity_due = (
            self.ba_patch_identity_enabled
            and int(global_step) % self.ba_patch_identity_cadence == 0
            and bool((timesteps <= self.ba_patch_identity_max_timestep).any().item())
        )
        ownership_sampled = False
        if (
            self.training
            and torch.is_grad_enabled()
            and self.ba_attention_ownership_loss_enabled
        ):
            sample_device = (
                "cpu" if self.ba_attention_ownership_sample_on_cpu else latents.device
            )
            ownership_sampled = torch.rand((), device=sample_device).item() < (
                self.ba_attention_ownership_probability
            )
        self._ba_attention_ownership_capture = bool(
            ownership_sampled or patch_identity_due
        )
        roi_teacher_sampled = False
        if (
            self.training
            and torch.is_grad_enabled()
            and self.ba_roi_teacher_distill_enabled
        ):
            sample_device = "cpu" if self.ba_roi_teacher_sample_on_cpu else latents.device
            roi_teacher_sampled = torch.rand((), device=sample_device).item() < (
                self.ba_roi_teacher_probability
            )
        self._ba_roi_teacher_capture = roi_teacher_sampled

        # 09 Aug 2026 - CL13: occasionally take the plain PhotoMaker path so the
        # native route stays a coherent fallback. The branch has never been
        # trained to defer, which is why it paints a full face onto goggles and
        # hands. Training only - `_reference_dropout_active` is never set during
        # validation, so inference always uses the branch.
        drop_reference = False
        if self.training and self.ba_reference_dropout_probability > 0.0:
            drop_reference = random.random() < self.ba_reference_dropout_probability
        self._reference_dropout_active = bool(drop_reference)

        if drop_reference:
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        elif self.train_ba_all_steps:
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
        ownership_loss = collect_hardcase_aux_loss(self)
        if ownership_loss is None:
            ownership_loss = noise_pred.float().new_tensor(0.0)
        hardcase_aux_loss = (
            self.ba_semantic_ownership_loss_weight * ownership_loss
        )

        attention_visible_mass = noise_pred.float().new_tensor(0.0)
        if self.ba_attention_ownership_loss_enabled:
            # 17 Aug 2026 - AICODE-NOTE: CL31 samples this auxiliary. Keep its
            # declared scalar contract present on unsampled steps so logging
            # cannot turn a valid no-op into a training failure.
            ba_telemetry.update(
                {
                    "loss_ba_attention_ownership": attention_visible_mass,
                    "ba/attention_visible_ref_mass/up0": attention_visible_mass,
                    "ba/attention_visible_ref_mass/up1": attention_visible_mass,
                    "ba/attention_top_ref_mass/up0": attention_visible_mass,
                    "ba/attention_top_ref_mass/up1": attention_visible_mass,
                    "ba/attention_ownership_applied_fraction": attention_visible_mass,
                }
            )

        if self.ba_visibility_ownership_v2_enabled:
            ownership = collect_visibility_ownership_v2_loss(self)
            if ownership is None:
                raise RuntimeError("CL38 did not capture its top/contact anchor")
            top_native, contact_native, top_area, applied = ownership
            start = self.ba_visibility_ownership_v2_ramp_start_step
            end = self.ba_visibility_ownership_v2_ramp_end_step
            ramp = max(0.0, min(1.0, (int(global_step) - start) / max(end - start, 1)))
            ownership_loss = ramp * (
                self.ba_visibility_ownership_v2_top_native_weight * top_native
                + self.ba_visibility_ownership_v2_contact_native_weight * contact_native
            )
            hardcase_aux_loss = hardcase_aux_loss + ownership_loss
            ba_telemetry.update(
                {
                    "loss_ba_visibility_ownership_v2": ownership_loss.detach(),
                    "ba/visibility_ownership_v2/top_native_l1": top_native.detach(),
                    "ba/visibility_ownership_v2/contact_native_l1": contact_native.detach(),
                    "ba/visibility_ownership_v2/top_area": top_area.detach(),
                    "ba/visibility_ownership_v2/applied_fraction": applied.detach(),
                }
            )
        if self._ba_attention_ownership_capture:
            attention_loss, attention_visible_mass, attention_top_mass = (
                collect_attention_ownership_loss(self)
            )
            if ownership_sampled:
                hardcase_aux_loss = hardcase_aux_loss + (
                    self.ba_attention_ownership_weight * attention_loss
                )
            ba_telemetry.update(
                {
                    "loss_ba_attention_ownership": attention_loss.detach(),
                    "ba/attention_visible_ref_mass/up0": attention_visible_mass.detach(),
                    "ba/attention_visible_ref_mass/up1": attention_visible_mass.detach(),
                    "ba/attention_top_ref_mass/up0": attention_top_mass.detach(),
                    "ba/attention_top_ref_mass/up1": attention_top_mass.detach(),
                    "ba/attention_ownership_applied_fraction": noise_pred.new_tensor(
                        float(ownership_sampled)
                    ),
                }
            )

        if self.ba_frequency_shared_schedule_enabled:
            shared_raw = self.unet.ba_frequency_shared_schedule_raw.float()
            shared_anchor = shared_raw.square().mean()
            hardcase_aux_loss = hardcase_aux_loss + (
                self.ba_frequency_shared_anchor_weight * shared_anchor
            )
            bounded = torch.tanh(shared_raw.detach())
            ba_telemetry.update(
                {
                    "loss_ba_shared_schedule_anchor": (
                        self.ba_frequency_shared_anchor_weight * shared_anchor
                    ).detach(),
                    "ba/shared_frequency_low_late": bounded[0] * self.ba_frequency_shared_low_late_half_range + self.ba_frequency_shared_low_late_center,
                    "ba/shared_frequency_high_early": bounded[1] * self.ba_frequency_shared_high_early_half_range + self.ba_frequency_shared_high_early_center,
                    "ba/shared_frequency_high_late": bounded[2] * self.ba_frequency_shared_high_late_half_range + self.ba_frequency_shared_high_late_center,
                }
            )

        if roi_teacher_sampled:
            roi_loss, roi_cosine, roi_eligible = collect_roi_teacher_loss(self)
            hardcase_aux_loss = hardcase_aux_loss + self.ba_roi_teacher_weight * roi_loss
            ba_telemetry.update(
                {
                    "loss_ba_roi_teacher": roi_loss.detach(),
                    "ba/roi_teacher_eligible_fraction": roi_eligible.detach(),
                    "ba/roi_teacher_applied_fraction": noise_pred.new_tensor(1.0),
                    "ba/roi_teacher_student_cosine/up0": roi_cosine.detach(),
                    "ba/roi_teacher_student_cosine/up1": roi_cosine.detach(),
                }
            )
        elif self.ba_roi_teacher_distill_enabled:
            zero = noise_pred.float().new_tensor(0.0)
            ba_telemetry.update(
                {
                    "loss_ba_roi_teacher": zero,
                    "ba/roi_teacher_eligible_fraction": zero,
                    "ba/roi_teacher_applied_fraction": zero,
                    "ba/roi_teacher_student_cosine/up0": zero,
                    "ba/roi_teacher_student_cosine/up1": zero,
                }
            )

        patch_identity_weighted, patch_identity_metrics = (
            self._predicted_x0_patch_identity_auxiliary(
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                timesteps=timesteps,
                face_bbox=face_bbox,
                ref_images=ref_images,
                face_bbox_ref=face_bbox_ref,
                identity_face_bboxes_ref=identity_face_bboxes_ref,
                global_step=int(global_step),
                visible_gate_mass=attention_visible_mass,
            )
        )
        hardcase_aux_loss = hardcase_aux_loss + patch_identity_weighted
        if self.ba_patch_identity_enabled:
            ba_telemetry.update(patch_identity_metrics)

        if self.ba_frequency_surface_loss_enabled:
            surface = collect_frequency_surface_aux_loss(self)
            surface_loss = noise_pred.float().new_tensor(0.0)
            surface_applied = surface_loss
            if surface is not None:
                top_loss, floor_loss, surface_applied = surface
                surface_loss = (
                    self.ba_frequency_surface_top_weight * top_loss
                    + self.ba_frequency_surface_visible_floor_weight * floor_loss
                )
                hardcase_aux_loss = hardcase_aux_loss + surface_loss
            ba_telemetry.update(
                {
                    "loss_ba_frequency_surface": surface_loss.detach(),
                    "ba/frequency_surface_applied_fraction": (
                        surface_applied.detach()
                    ),
                }
            )

        if self.ba_frequency_learnable_schedule_enabled:
            schedule_anchor = collect_frequency_schedule_anchor_loss(self)
            if schedule_anchor is None:
                raise RuntimeError("CL28 schedule parameters were not installed")
            schedule_weighted = (
                self.ba_frequency_schedule_anchor_weight * schedule_anchor
            )
            hardcase_aux_loss = hardcase_aux_loss + schedule_weighted
            ba_telemetry["loss_ba_frequency_schedule_anchor"] = (
                schedule_weighted.detach()
            )

        lowband_loss = noise_pred.float().new_tensor(0.0)
        lowband_metrics = {
            "ba/lowband_positive_cosine/all": lowband_loss,
            "ba/lowband_wrong_cosine/all": lowband_loss,
            "ba/lowband_correct_wrong_margin/all": lowband_loss,
        }
        if lowband_permutation is not None or positive_sameid_sampled:
            alternate_refs = []
            alternate_masks = []
            for refs, bbox in zip(spatial_ref_images_alt, face_bbox_ref_alt):
                refs = refs if isinstance(refs, (list, tuple)) else [refs]
                if len(refs) != 1:
                    raise RuntimeError("Low-band auxiliary requires one alternate spatial reference")
                ref = refs[0]
                alternate_refs.append(ref)
                ref_size = (
                    tuple(ref.shape[-2:])
                    if isinstance(ref, torch.Tensor)
                    else (ref.height, ref.width)
                )
                alternate_masks.append(
                    self._bbox_to_ref_mask(
                        bbox, noisy_latents.shape[-2:], ref_size
                    )
                )
            alternate_latents = self._encode_reference_latents(
                alternate_refs, target_shape=noisy_latents.shape[-2:]
            ).to(dtype=noisy_latents.dtype)
            alternate_mask4 = torch.cat(alternate_masks, dim=0).to(
                device=self.device, dtype=noisy_latents.dtype
            )
            paired_reference_noise = getattr(self, "_ref_noise", None)
            if paired_reference_noise is None:
                raise RuntimeError("CL29 lost paired reference noise")
            previous_suppression = bool(
                getattr(self, "_ba_suppress_telemetry", False)
            )
            self._ba_suppress_telemetry = True
            self._ba_lowband_capture_mode = (
                "contrast" if lowband_permutation is not None else "positive"
            )
            self._ba_lowband_negative_permutation = lowband_permutation
            try:
                run_branched_forward_pass(
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
                if lowband_permutation is not None:
                    lowband_loss, lowband_metrics = collect_lowband_contrastive_loss(
                        self,
                        temperature=self.ba_frequency_lowband_contrastive_temperature,
                    )
                else:
                    lowband_loss, positive_cosine = collect_lowband_positive_loss(self)
            finally:
                self._ba_suppress_telemetry = previous_suppression
                self._ba_lowband_capture_mode = "off"
                self._ba_lowband_negative_permutation = None
                clear_lowband_contrastive_state(self)
            ramp_start = (
                self.ba_frequency_lowband_contrastive_ramp_start_step
                if lowband_permutation is not None
                else self.ba_frequency_positive_sameid_ramp_start_step
            )
            ramp_end = (
                self.ba_frequency_lowband_contrastive_ramp_end_step
                if lowband_permutation is not None
                else self.ba_frequency_positive_sameid_ramp_end_step
            )
            aux_weight = (
                self.ba_frequency_lowband_contrastive_weight
                if lowband_permutation is not None
                else self.ba_frequency_positive_sameid_weight
            )
            ramp = max(
                0.0,
                min(
                    1.0,
                    (
                        int(global_step)
                        - ramp_start
                    )
                    / float(
                        ramp_end - ramp_start
                    ),
                ),
            )
            hardcase_aux_loss = hardcase_aux_loss + (
                ramp * aux_weight * lowband_loss
            )
        elif self.ba_frequency_lowband_contrastive_enabled:
            self._ba_lowband_capture_mode = "off"
            self._ba_lowband_negative_permutation = None
            clear_lowband_contrastive_state(self)
        if self.ba_frequency_lowband_contrastive_enabled:
            ba_telemetry.update(lowband_metrics)
            ba_telemetry.update(
                {
                    "loss_ba_lowband_contrastive": lowband_loss.detach(),
                    "ba/lowband_contrastive_applied_fraction": (
                        lowband_loss.new_tensor(float(lowband_permutation is not None))
                    ),
                    "ba/lowband_skipped_same_identity_fraction": (
                        lowband_loss.new_tensor(
                            float(lowband_sampled and lowband_skipped_same_identity)
                        )
                    ),
                }
            )
        if self.ba_frequency_positive_sameid_enabled:
            ba_telemetry.update(
                {
                    "loss_ba_positive_sameid": lowband_loss.detach(),
                    "ba/positive_sameid_cosine/all": (
                        positive_cosine.detach()
                        if positive_sameid_sampled
                        else lowband_loss.new_tensor(0.0)
                    ),
                    "ba/positive_sameid_applied_fraction": lowband_loss.new_tensor(
                        float(positive_sameid_sampled)
                    ),
                }
            )

        crossview_loss = noise_pred.float().new_tensor(0.0)
        if (
            self.training
            and self.ba_crossview_consistency_enabled
            and spatial_ref_images_alt is not None
            and face_bbox_ref_alt is not None
            and torch.rand((), device=latents.device).item()
            < self.ba_crossview_consistency_probability
        ):
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
                face = F.interpolate(
                    face,
                    size=noise_pred.shape[-2:],
                    mode="nearest",
                )
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
            hardcase_aux_loss = hardcase_aux_loss + (
                self.ba_crossview_consistency_weight * crossview_loss
            )
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

        (
            identity_aux_loss,
            identity_aux_weight,
            identity_aux_applied,
            identity_aux_telemetry,
        ) = (
            self._predicted_x0_identity_auxiliary(
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                timesteps=timesteps,
                pixel_values=pixel_values,
                face_bbox=face_bbox,
                ref_images=ref_images,
                face_bbox_ref=face_bbox_ref,
                identity_face_bboxes_ref=identity_face_bboxes_ref,
                global_step=int(global_step),
            )
        )

        (
            boundary_weighted,
            boundary_raw,
            top_raw,
            boundary_fraction,
            boundary_teacher_rms,
        ) = self._pm_boundary_distillation(
            student=noise_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            face_mask=mask4,
        )
        hardcase_aux_loss = hardcase_aux_loss + boundary_weighted
        if self.ba_pm_boundary_distill_enabled:
            ba_telemetry.update(
                {
                    "loss_ba_pm_boundary": boundary_raw.detach(),
                    "loss_ba_pm_top_object": top_raw.detach(),
                    "ba/pm_boundary_fraction": boundary_fraction.detach(),
                    "ba/pm_boundary_teacher_student_rms": boundary_teacher_rms.detach(),
                }
            )

        anchor_loss = noise_pred.float().new_tensor(0.0)
        if self.ba_low_noise_id_reward_enabled:
            ba_telemetry["ba/id_reward_trajectory_divergence"] = anchor_loss.detach()
        if (
            self.training
            and self.ba_low_noise_id_reward_enabled
            and float(identity_aux_applied.detach()) > 0.0
        ):
            frozen_pred = self._frozen_cl19_prediction(
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
            anchor_loss = F.mse_loss(noise_pred.float(), frozen_pred.float())
            ba_telemetry["ba/id_reward_trajectory_divergence"] = (
                anchor_loss.detach().sqrt()
            )

        result = {
            'model_pred': noise_pred,
            'target': noise,
            'pred_wrong_spatial_ref': wrong_spatial_reference_pred,
            'reference_shuffle_applied': reference_shuffle_applied,
            'reference_prediction_delta_ratio': reference_prediction_delta_ratio,
            'ba_telemetry': ba_telemetry,
            'identity_aux_loss': identity_aux_loss,
            'identity_aux_weight': identity_aux_weight,
            'identity_aux_applied': identity_aux_applied,
            'ba_anchor_loss': anchor_loss,
            'ba_anchor_weight': noise_pred.new_tensor(
                self.ba_low_noise_id_reward_kl_weight
                if self.ba_low_noise_id_reward_enabled
                else 0.0
            ),
            **identity_aux_telemetry,
        }
        # 12 Aug 2026 - Training optimization: defaults-off hard-case plumbing
        # must leave CL14's backward graph exact; expose aux edges only when used.
        if self.ba_hardcase_mode != "off" or self.ba_crossview_consistency_enabled:
            result.update(
                ba_aux_loss=hardcase_aux_loss,
                ba_ownership_loss=ownership_loss.detach(),
                ba_crossview_loss=crossview_loss.detach(),
            )
        return result

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
        feather = int(getattr(self, "ba_training_mask_feather", 0))
        if feather > 0:
            # 09 Aug 2026 - CL14: a hard binary box teaches a discontinuous
            # handover at the edge, which shows up as seams and unblended faces.
            # Ramp inward so the branch learns a gradual transition. Training
            # only: the inference pipeline builds its own mask and never reads
            # this flag, so validation stays byte-comparable.
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
