from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Sequence

import torch
from PIL import Image
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPImageProcessor

from diffusers.utils import convert_unet_state_dict_to_peft
from src.model.photomaker_path import resolve_photomaker_path
from src.model.sdxl.original import SDXL

from .e13_objectives import compute_e13_objectives, prepare_frequency_surface_mask
from .insightface_package import assert_cuda_face_analyzer, create_face_analyzer
from .e13_training_helpers import (
    ensure_branched_after_eval as ensure_branched_after_eval_helper,
    install_branched_processors_for_training,
    prepare_branched_training_inputs,
    run_branched_forward_pass,
)
from .model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .e13_contract import (
    assert_trainable_contract as assert_e13_trainable_contract,
    get_state_dict as get_e13_state_dict,
    initialise_e13_contract,
    load_state_dict as load_e13_state_dict,
    optimizer_groups as e13_optimizer_groups,
)


class PhotomakerBranchedLora(SDXL):
    """PhotoMaker-v2 model using the fixed E13 branched-attention route."""

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
        photomaker_lora_rank: int = 64,
        e13_settings: Mapping | None = None,
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_dtype=weight_dtype,
            device=device,
        )
        self.lora_rank = rank
        self.init_lora_weights = init_lora_weights
        self.lora_modules = lora_modules

        self.id_image_processor = CLIPImageProcessor()
        self.id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        self.target_size = target_size
        block_channels = getattr(self.vae.config, "block_out_channels", None)
        self.vae_scale_factor = 2 ** (len(block_channels) - 1) if block_channels else 8

        # Pin ONNX Runtime to the local rank and reject silent CPU fallback.
        device_id = int(os.environ.get("LOCAL_RANK", "0"))
        faceanalysis_cpu = os.environ.get(
            "FACEANALYSIS_CPU", "0"
        ).lower() not in {"0", "false", "no"}
        if faceanalysis_cpu:
            raise RuntimeError("E13C-PERF-03 requires FACEANALYSIS_CPU=0")
        self.face_analyzer = create_face_analyzer(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": device_id}, {}],
            allowed_modules=["detection", "recognition"],
            ctx_id=device_id,
            det_size=(640, 640),
            fallback_ctx_id=-1,
            quiet=True,
        )
        assert_cuda_face_analyzer(self.face_analyzer)

        self.trigger_word = trigger_word
        self.num_tokens = self.id_encoder.num_tokens
        added_tokens_1 = self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        added_tokens_2 = self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)
        if added_tokens_1:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        if added_tokens_2:
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))

        # 22 Aug 2026 - The clean branch has one fixed E13 route. Only genuine
        # leaf deltas live in e13_settings; removed selectors had one value in
        # every supported recipe and duplicated the contract in three places.
        self.scheduler = self.noise_scheduler
        self.pose_adapt_ratio = 0.0
        self.ca_mixing_for_face = False
        self.face_embed_strategy = "id"
        initialise_e13_contract(self, e13_settings)

        photomaker_lora_config = LoraConfig(
            r=photomaker_lora_rank,
            lora_alpha=photomaker_lora_rank,
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
        install_branched_processors_for_training(self)

    def load_photomaker_state_dict_(self, state_dict):
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {
            key.replace("unet.", ""): value
            for key, value in lora_state_dict.items()
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            self.unet, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert not unexpected_keys, unexpected_keys
        self.id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)

    def get_trainable_params(self, config):
        return e13_optimizer_groups(self, config)

    def get_state_dict(self):
        return get_e13_state_dict(self)

    def load_state_dict_(self, state_dict):
        load_e13_state_dict(self, state_dict)

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
        face_bbox_ref: Sequence[Sequence[float]] | None = None,
        ba_occluder_mask: Sequence[torch.Tensor] | None = None,
        spatial_ref_images_alt: Sequence[Sequence[Image.Image]] | None = None,
        face_bbox_ref_alt: Sequence[Sequence[float]] | None = None,
        do_cfg: bool = False,
        *args,
        **kwargs,
    ):
        del do_cfg  # classifier-free guidance is not used during training

        pixel_values = pixel_values.to(self.device, self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        t_scalar = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (1,),
            device=latents.device,
        ).long()
        timesteps = t_scalar.repeat(batch_size)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = torch.cat(
            [
                self.compute_time_ids(original_size, crop)
                for original_size, crop in zip(original_sizes, crop_top_lefts)
            ]
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
            face_prompt_embeds,
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
        prepare_frequency_surface_mask(
            self, ba_occluder_mask, mask4, noisy_latents.dtype
        )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids.to(device=self.device, dtype=self.unet.dtype),
        }

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
        )
        objectives = compute_e13_objectives(
            self,
            noise_pred=noise_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            mask4=mask4,
            face_prompt_embeds=face_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
            spatial_ref_images_alt=spatial_ref_images_alt,
            face_bbox_ref_alt=face_bbox_ref_alt,
        )
        return {"model_pred": noise_pred, "target": noise, **objectives}

    def ensure_branched_after_eval(self):
        ensure_branched_after_eval_helper(self)

    def assert_trainable_contract(self, optimizer=None):
        return assert_e13_trainable_contract(self, optimizer=optimizer)
