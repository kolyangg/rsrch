from __future__ import annotations

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
from src.model.sdxl.original import SDXL

# --- Branched-attention specific import ---
### 24 Nov: fixing oneid training issues
from .branched_new2 import (
    two_branch_predict,
    prepare_reference_latents,
    patch_unet_attention_processors,
)  # Use branched v2 logic end-to-end
### 24 Nov: fixing oneid training issues

# --- PhotoMaker v2 upgraded ID encoder + InsightFace integration START ---
from .insightface_package import FaceAnalysis2, analyze_faces
from .model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
# --- PhotoMaker v2 upgraded ID encoder + InsightFace integration END ---


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
        photomaker_lora_rank: int = 64,
        pose_adapt_ratio: float = 0.25, # --- ADDED For training integration
        ca_mixing_for_face: bool = True,  # --- ADDED For training integration
        face_embed_strategy: str = "face", # --- ADDED For training integration
        train_branch_mode: str = "both",   # 'both' or 'ref_only' for BranchedAttnProcessor
        train_ba_only: bool = False,       # 28 Nov: optionally train only branched-attn layers
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        ba_weights_split: bool = False,    # optionally enable per-branch BA-specific adapters
        ### 29 Nov - Clean separataion of BA-specific parameters ###
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_dtype=weight_dtype,
            device=device,
        )
        self.lora_rank = rank
        self.init_lora_weights = init_lora_weights
        self.lora_modules = lora_modules
        self.target_size = target_size

        self.id_image_processor = CLIPImageProcessor()
        self.feature_extractor = self.id_image_processor  # --- MODIFIED For training integration ---
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae.config, "block_out_channels") else 8  # --- MODIFIED For training integration ---

        # --- PhotoMaker v2 integration START: upgraded ID encoder & face embeddings ---
        # Mirror the PhotoMaker v2 ID encoder configuration (512-d InsightFace input).
        self.id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()

        # Instantiate FaceAnalysis once for extracting 512-D identity embeddings.
        self.face_analyzer = FaceAnalysis2(providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                                           allowed_modules=["detection", "recognition"])
        ctx_id = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else -1  # --- MODIFIED For training integration ---
        try:
            self.face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))  
        except Exception:
            self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))  # --- MODIFIED For training integration ---
        # --- PhotoMaker v2 integration END ---

        self.trigger_word = trigger_word
        self.num_tokens = self.id_encoder.num_tokens
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

        # --- Branched-attention integration START: runtime knobs used by branched processors ---
        # Branched helpers expect ``scheduler`` attribute – alias it once.
        self.scheduler = self.noise_scheduler
        self.pose_adapt_ratio = float(pose_adapt_ratio) # --- ADDED For training integration
        self.ca_mixing_for_face = bool(ca_mixing_for_face) # --- ADDED For training integration
        self.face_embed_strategy = (face_embed_strategy or "face").lower() # --- ADDED For training integration
        self.train_branch_mode = (train_branch_mode or "both").lower()
        ### 28 Nov: train only BA layers ###
        self.train_ba_only = bool(train_ba_only)
        ### 28 Nov: train only BA layers ###
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        self.ba_weights_split = bool(ba_weights_split)
        ### 29 Nov - Clean separataion of BA-specific parameters ###
        # --- Branched-attention integration END ---

        photomaker_lora_config = LoraConfig(
            r=photomaker_lora_rank,
            lora_alpha=photomaker_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(photomaker_lora_config)

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
        
        ### Modified to make attention processors train ###

        # --- enable training of branched attention processors (minimal) ---
        # Install processors *once* so their parameters are registered before the optimizer is built.
        try:
            # Safe zero masks at latent resolution (prevents "requires a mask" errors during validation startup)
            H = self.target_size // int(self.vae_scale_factor)
            W = self.target_size // int(self.vae_scale_factor)
            zero_ctx = torch.zeros(1, 1, H, W, device=self.unet.device, dtype=self.unet.dtype)

            patch_unet_attention_processors(
                pipeline=self,
                mask=zero_ctx,
                mask_ref=zero_ctx,
                scale=1.0,
                id_embeds=None,
                class_tokens_mask=None,
            )
            # Ensure processor params are trainable (will be included in your existing param filter)
            if hasattr(self.unet, "attn_processors"):
                for proc in self.unet.attn_processors.values():
                    for p in proc.parameters():
                        p.requires_grad_(True)

            ### 28 Nov: train only BA layers ###
            # Optionally restrict training to branched processors + LoRA on attention projections.
            if getattr(self, "train_ba_only", False):
                # First freeze everything in UNet.
                for name, p in self.unet.named_parameters():
                    p.requires_grad_(False)

                # Re-enable branched attention processors.
                if hasattr(self.unet, "attn_processors"):
                    for proc in self.unet.attn_processors.values():
                        for p in proc.parameters():
                            p.requires_grad_(True)

                # Re-enable LoRA weights on attention projections (lora_A / lora_B).
                for name, p in self.unet.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        p.requires_grad_(True)
            ### 28 Nov: train only BA layers ###
        except Exception as e:
            print(f"[PhotomakerBranchedLora] warning while installing branched processors: {e}")
        # --- end minimal change ---

        ### Modified to make attention processors train ###


    def load_photomaker_state_dict_(self, state_dict):
        # load lora
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert not unexpected_keys, unexpected_keys

        # load id_encoder
        self.id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)

    def get_trainable_params(self, config):
        ### 28 Nov: train only BA layers ###
        if getattr(self, "train_ba_only", False):
            # Train branched attention processors + LoRA weights on attention projections.
            proc_params = []
            lora_params = []
            for name, p in self.unet.named_parameters():
                if not p.requires_grad:
                    continue
                if ".attn1.processor." in name or ".attn2.processor." in name:
                    proc_params.append(p)
                elif "lora_A" in name or "lora_B" in name:
                    lora_params.append(p)

            param_groups = []
            if proc_params:
                param_groups.append(
                    {"params": proc_params, "lr": config.lr_for_lora, "name": "branched_processors"}
                )
            if lora_params:
                param_groups.append(
                    {"params": lora_params, "lr": config.lr_for_lora, "name": "branched_lora"}
                )
            return param_groups
        ### 28 Nov: train only BA layers ###

        # Default behavior: train all UNet parameters with requires_grad=True (LoRA + processors).
        lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {"params": lora_params, "lr": config.lr_for_lora, "name": "lora_params"},
        ]
        return trainable_params

    def get_state_dict(self):
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        return {
            'lora_weights': lora_weights,
        }

    def load_state_dict_(self, state_dict):
        lora_state_dict = state_dict["lora_weights"]
        unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            # In newer peft versions this is an empty list when there are no unexpected keys
            assert not unexpected_keys, unexpected_keys

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str],
        ref_images: Sequence[Sequence[Image.Image]],
        original_sizes: Sequence[Sequence[int]],
        crop_top_lefts: Sequence[Sequence[int]],
        face_bbox: Sequence[Sequence[float]],
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

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(orig_size, crop) for orig_size, crop in zip(original_sizes, crop_top_lefts)]
        )

        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        class_tokens_mask_list = []
        mask_list = []
        ref_latents_list = []
        pm_feature_list = []  # --- MODIFIED For training integration ---

        image_h, image_w = pixel_values.shape[-2:]
        latent_h, latent_w = noisy_latents.shape[-2:]

        for prompt, refs, bbox in zip(prompts, ref_images, face_bbox):
            refs = refs if isinstance(refs, (list, tuple)) else [refs]

            prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.encode_prompt_with_trigger_word(
                prompt=prompt,
                num_id_images=len(refs),
                do_cfg=False,
            )

            with torch.no_grad():
                # --- PhotoMaker v2 integration START: derive InsightFace embeddings for v2 ID encoder ---
                id_pixel_values = self.id_image_processor(refs, return_tensors="pt").pixel_values.unsqueeze(0)
                id_pixel_values = id_pixel_values.to(self.device, dtype=self.id_encoder.dtype)

                prompt_for_id = prompt_embeds.to(dtype=self.id_encoder.dtype)
                id_embed_list = []
                for ref in refs:
                    img_np = np.array(ref.convert("RGB"))[:, :, ::-1]
                    faces = analyze_faces(self.face_analyzer, img_np)
                    if faces:
                        embedding = torch.from_numpy(faces[0]["embedding"]).float()
                    else:
                        embedding = torch.zeros(512, dtype=torch.float32)
                    id_embed_list.append(embedding)

                id_embeds = torch.stack(id_embed_list, dim=0).unsqueeze(0)
                id_embeds = id_embeds.to(device=self.device, dtype=self.id_encoder.dtype)

                prompt_embeds = self.id_encoder(
                    id_pixel_values,
                    prompt_for_id,
                    class_tokens_mask,
                    id_embeds,
                )
                # --- PhotoMaker v2 integration END ---

                # --- Branched-attention integration START: prepare reference latents for branch mixing ---
                # --- MODIFIED For training integration ---
                reference_latent = prepare_reference_latents(
                    self,  
                    refs[0],
                    image_h,
                    image_w,
                    noisy_latents.dtype,
                )
                # --- MODIFIED For training integration ---
                # --- Branched-attention integration END ---

                # --- MODIFIED For training integration ---
                if self.face_embed_strategy == "id_embeds":  
                    pm_features = self.id_encoder.extract_id_features(
                        id_pixel_values.to(device=self.device, dtype=self.id_encoder.dtype),
                        class_tokens_mask=class_tokens_mask,
                    )
                    pm_feature_list.append(pm_features.to(device=self.device, dtype=self.unet.dtype))
                # --- MODIFIED For training integration ---

            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            class_tokens_mask_list.append(class_tokens_mask)
            ref_latents_list.append(reference_latent)
            mask_list.append(
                self._bbox_to_mask(
                    bbox,
                    latent_shape=(latent_h, latent_w),
                    image_shape=(image_h, image_w),
                )
            )

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(device=self.device, dtype=self.unet.dtype)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0).to(device=self.device, dtype=self.unet.dtype)
        class_tokens_mask = torch.cat(class_tokens_mask_list, dim=0).to(device=self.device)

        # --- MODIFIED For training integration ---
        ### 24 Nov: fixing oneid training issues
        id_embeds_proc = None
        if self.face_embed_strategy == "face":  
            face_prompt_text = ["a close-up human face laughing hard"] * batch_size 
            face_prompt_embeds, _ = self.encode_prompt(face_prompt_text, do_cfg=False)
            face_prompt_embeds = face_prompt_embeds.to(device=self.device, dtype=self.unet.dtype)
        elif self.face_embed_strategy == "id_embeds" and pm_feature_list:
            id_features = torch.cat(pm_feature_list, dim=0)
            seq_len = prompt_embeds.shape[1]
            dim = prompt_embeds.shape[2]
            face_prompt_embeds = id_features.unsqueeze(1).expand(-1, seq_len, dim).contiguous()
            id_embeds_proc = id_features
        else:
            face_prompt_embeds = prompt_embeds  
        # --- MODIFIED For training integration ---
        ### 24 Nov: fixing oneid training issues

        # --- Branched-attention integration START: cache masks, reference latents, CFG state ---
        mask4 = torch.cat(mask_list, dim=0).to(device=self.device, dtype=noisy_latents.dtype)
        mask4_ref = mask4.clone()

        reference_latents = torch.cat(ref_latents_list, dim=0).to(device=self.device, dtype=noisy_latents.dtype)
        self._ref_latents_all = reference_latents
        self._face_prompt_embeds = prompt_embeds
        self.do_classifier_free_guidance = False
        # --- Branched-attention integration END ---

        # Re-sample reference noise every forward pass for training stability.
        if hasattr(self, "_ref_noise"):
            delattr(self, "_ref_noise")

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids.to(device=self.device, dtype=self.unet.dtype),
        }

        # --- Branched-attention integration START: run dual-branch UNet pass ---
        noise_pred, _, _ = two_branch_predict(
            pipeline=self,
            latent_model_input=noisy_latents,
            t=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            mask4=mask4,
            mask4_ref=mask4_ref,
            reference_latents=reference_latents,
            face_prompt_embeds=face_prompt_embeds,  # --- MODIFIED For training integration ---
            class_tokens_mask=class_tokens_mask,
            face_embed_strategy=self.face_embed_strategy,
            id_embeds=id_embeds_proc,
            step_idx=0,
            scale=1.0,
            timestep_cond=None,
        )
        # --- Branched-attention integration END ---

        return {
            'model_pred': noise_pred,
            'target': noise,
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
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt = prompt if not do_cfg else ""

        if prompt_embeds is None:
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

                    class_token = clean_input_ids[class_token_index]
                    clean_input_ids = (
                        clean_input_ids[:class_token_index]
                        + [class_token] * num_id_images * self.num_tokens
                        + clean_input_ids[class_token_index + 1 :]
                    )

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

    # --- Branched-attention helper utilities START ---
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
            ref_resized = ref_image.resize((self.target_size, self.target_size), Image.BILINEAR)
            ref_np = np.array(ref_resized).astype(np.float32) / 255.0
            ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
            ref_tensor = (ref_tensor - 0.5) / 0.5
            ref_tensor = ref_tensor.to(device=self.device, dtype=self.vae.dtype)

        with torch.no_grad():
            latents = self.vae.encode(ref_tensor).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        if latents.shape[-2:] != target_shape:
            latents = F.interpolate(latents, size=target_shape, mode="bilinear", align_corners=False)

        return latents
    # --- Branched-attention helper utilities END ---

    ### Modified to make attention processors train ###
    def ensure_branched_after_eval(self):
        # Re-install (no-op if already installed); pass safe masks/embeds
        ### 24 Nov: fixing oneid training issues
        from .branched_new2 import patch_unet_attention_processors
        ### 24 Nov: fixing oneid training issues
        
        # patcher expects `pipeline.device`; ensure it's present on this module
        dev = getattr(self, "device", None) or self.unet.device
        if not hasattr(self, "device"):
            self.device = dev
        dt = self.unet.dtype

        # 1x1 dummy masks just to initialize; real masks are set during forward
        z = torch.zeros(1, 1, 1, 1, device=dev, dtype=dt)
        idem = torch.zeros(1, 2048, device=dev, dtype=dt)
        patch_unet_attention_processors(
            self, z, z, scale=1.0,
            id_embeds=idem, class_tokens_mask=None
        )
    ### Modified to make attention processors train ###
