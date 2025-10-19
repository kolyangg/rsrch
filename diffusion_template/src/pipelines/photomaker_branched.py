# photomaker/pipeline.py

#####
# Modified from https://github.com/huggingface/diffusers/blob/v0.29.1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
# PhotoMaker v2 @ TencentARC and MCG-NKU 
# Author: Zhen Li
#####

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path  # --- MODIFIED For training integration ---
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.branched_new import (
    two_branch_predict,
    prepare_reference_latents,
    encode_face_prompt,
    patch_unet_attention_processors,
    restore_original_processors,
    save_debug_images,
)

# # Keep existing imports:
# from .branched_v4 import (
#     MASK_LAYERS_CONFIG,
#     compute_binary_face_mask,
#     simple_threshold_mask,
#     encode_face_latents,
# )


# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.branch_helpers import (
    aggregate_heatmaps_to_mask,
    prepare_mask4,
    save_branch_previews,
    debug_reference_latents_once,
    save_debug_ref_latents,
    save_debug_ref_mask_overlay,
    collect_attention_hooks,
)

# Only import what's actually needed
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.mask_utils import compute_binary_face_mask, simple_threshold_mask
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.mask_utils import MASK_LAYERS_CONFIG

# Import dynamic mask generation
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.add_masking import DynamicMaskGenerator, get_default_mask_config


import os
import numpy as np
from PIL import Image
import torch.nn.functional as F


import PIL

import torch
from transformers import CLIPImageProcessor

from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
# --- ADDED For training integration (FOLDER STUCTURE) ---
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.callbacks import (
    MultiPipelineCallbacks,
    PipelineCallback,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    _get_model_file,
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    scale_lora_layers,
    unscale_lora_layers,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.model import PhotoMakerIDEncoder  # PhotoMaker v1
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken  # PhotoMaker v2
# --- ADDED For training integration (FOLDER STUCTURE) ---
from src.model.photomaker_branched.insightface_package import FaceAnalysis2, analyze_faces

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
    

class PhotoMakerStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @validate_hf_hub_args
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        
        
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        pm_version: str = 'v2',
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            weight_name (`str`):
                The weight name NOT the path to the weight.

            subfolder (`str`, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

            trigger_word (`str`, *optional*, defaults to `"img"`):
                The trigger word is used to identify the position of class word in the text prompt, 
                and it is recommended not to set it as a common word. 
                This trigger word must be placed after the class word when used, otherwise, it will affect the performance of the personalized generation.           
        """

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        # self.num_tokens =2
        self.num_tokens = 2 if pm_version == 'v2' else 1
        self.pm_version = pm_version
        self.trigger_word = trigger_word
        # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
        print(f"Loading PhotoMaker {pm_version} components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
        self.id_image_processor = CLIPImageProcessor()
        if pm_version == "v1": # PhotoMaker v1 
            id_encoder = PhotoMakerIDEncoder()
        elif pm_version == "v2": # PhotoMaker v2
            id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        else:
            raise NotImplementedError(f"The PhotoMaker version [{pm_version}] does not support")

        id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
        id_encoder = id_encoder.to(self.device, dtype=self.unet.dtype)    
        self.id_encoder = id_encoder
        self._ensure_face_analyzer()

        # load lora into models
        print(f"Loading PhotoMaker {pm_version} components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word token
        if self.tokenizer is not None: 
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)
        

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids 
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

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
                        f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                            Trigger word: {self.trigger_word}, Prompt: {prompt}."
                    )
                class_token_index = class_token_index[0]

                # Expand the class word token and corresponding mask
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images * self.num_tokens + \
                    clean_input_ids[class_token_index+1:]                
                    
                # Truncation or padding
                max_len = tokenizer.model_max_length
                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                        max_len - len(clean_input_ids)
                    )

                class_tokens_mask = [True if class_token_index <= i < class_token_index+(num_id_images * self.num_tokens) else False \
                     for i in range(len(clean_input_ids))]
                
                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                prompt_embeds = text_encoder(clean_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case
        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, class_tokens_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Added parameters (for PhotoMaker)
        input_id_images: PipelineImageInput = None,
        # start_merge_step kept for back-compat. If provided, it will populate both new knobs.
        start_merge_step: int = 10, # TODO: change to `style_strength_ratio` in the future
        # NEW: split the semantics
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        id_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,

        # ───────────────────  Branched-attention switches  ───────────────────
        use_branched_attention: bool = False,
        photomaker_scale: float = 1.0,  # Add scale parameter for attention
        save_heatmaps: bool = True,
        branched_attn_start_step: int = 10,
        heatmap_mode: str = "identity", # "focus_token", # "identity",  # "identity" or "focus_token" - choose heatmap generation approach
        # heatmap_mode: str = "focus_token", # 
        # focus_token: str = "",
        focus_token: str = "face",
        mask_mode: str = "spec",                 # or "simple"
        face_embed_strategy: str = "face", # "face", #  "face" or "id_embeds"
        # import_mask: Optional[str] = "hm_debug/keanu_gen_mask.png",
        import_mask: Optional[str] = "../compare/testing/ref5_masks/marion_gen_mask.png",
        # import_mask: Optional[str] = "../compare/testing/ref5_masks/marion_gen_mask_simple.png",
        # import_mask: Optional[str] = "hm_debug/keanu_gen_mask_white_new.png",
        # import_mask_ref: Optional[str] = "hm_debug/keanu_ref_mask.png",
        
        import_mask_ref: Optional[str] = None, # to debug auto_mask_ref

        # New: folder-based lookup (ignored if use_dynamic_mask=True)
        import_mask_folder: Optional[str] = "../compare/testing/gen_masks",
        use_mask_folder: bool = True,
        
        # import_mask_ref: Optional[str] = "../compare/testing/ref5_masks/marion_ref_mask.png",
        
        # auto_mask_ref: bool = False,
        auto_mask_ref: bool = True,

        # import_mask: Optional[str] = "../compare/testing/ref3_masks/eddie_pm_mask_new.jpg",
        # import_mask: Optional[str] = "../compare/testing/ref3_masks/eddie_pm_mask_new_easy.png",
        # import_mask: Optional[str] = "../compare/testing/ref3_masks/eddie_pm_mask_white_new.png",
        # import_mask_ref: Optional[str] = "../compare/testing/ref3_masks/eddie_mask_new.png",
        
        # import_mask: Optional[str] = "../compare/testing/ref3_masks/eddie_pm_mask_white.jpg",
        # import_mask_ref: Optional[str] = "../compare/testing/ref3_masks/eddie_mask_white.jpg",
        # ───────── Debug / branch-preview switches ─────────

        # ───────── Dynamic mask generation parameters ─────────────
        # use_dynamic_mask_ref: bool = False,

        # generation mask
        use_dynamic_mask: bool = True,
        # use_dynamic_mask: bool = False,
        
        mask_start: int = 10,
        mask_end: int = 20,
        save_heatmaps_dynamic: bool = True,
        token_focus: str = "face",
        add_token_to_prompt: bool = False,
        mask_layers_config: Optional[List[Dict]] = None,
        # save_heatmap_pdf: bool = False,
        # save_hm_pdf: bool = True,
        save_hm_pdf: bool = False,
        heatmap_interval: int = 5,
 

        debug_save_face_branch: bool = True,
        debug_save_bg_branch: bool = True,
        debug_dir: str = "hm_debug",
        force_par_before_pm: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Only the parameters introduced by PhotoMaker are discussed here. 
        For explanations of the previous parameters in StableDiffusionXLPipeline, please refer to https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py

        Args:
            input_id_images (`PipelineImageInput`, *optional*): 
                Input ID Image to work with PhotoMaker.
            class_tokens_mask (`torch.LongTensor`, *optional*):
                Pre-generated class token. When the `prompt_embeds` parameter is provided in advance, it is necessary to prepare the `class_tokens_mask` beforehand for marking out the position of class word.
            prompt_embeds_text_only (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds_text_only (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        # --- Back-compat for older callers that still pass start_merge_step ---
        if start_merge_step is not None:
            # If explicit new knobs are not set differently by caller, mirror legacy.
            photomaker_start_step = photomaker_start_step if photomaker_start_step is not None else start_merge_step
            merge_start_step      = merge_start_step      if merge_start_step      is not None else start_merge_step
        # --- Back-compat for older callers that still pass start_merge_step ---


        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        
        # self._guidance_scale = 0 # TEMP WTF!

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        #        
        if prompt_embeds is not None and class_tokens_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed. Make sure to generate `class_tokens_mask` from the same tokenizer that was used to generate `prompt_embeds`."
            )
        # check the input id images
        if input_id_images is None:
            raise ValueError(
                "Provide `input_id_images`. Cannot leave `input_id_images` undefined for PhotoMaker pipeline."
            )
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        
        num_id_images = len(input_id_images)
        (
            prompt_embeds, 
            _,
            pooled_prompt_embeds,
            _,
            class_tokens_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            class_tokens_mask=class_tokens_mask,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        # encode, remove trigger word token, then decode
        tokens_text_only = self.tokenizer.encode(prompt, add_special_tokens=False)
        trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        tokens_text_only.remove(trigger_word_token)
        prompt_text_only = self.tokenizer.decode(tokens_text_only, add_special_tokens=False)
        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 6. Prepare the input ID images
        dtype = next(self.id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts
        

        # ================================================================
        #  Branched-attention one-off preparation
        # ================================================================
        if use_branched_attention or save_heatmaps or save_hm_pdf:

            self._heatmaps      = {}
            self._hm_layers     = [s["name"] for s in MASK_LAYERS_CONFIG]
            self._step_tags     = []              # for pretty frame labels
            self._orig_forwards = {}

        # ================================================================
        #  Initialize dynamic mask generator
        # ================================================================
        # if save_heatmaps or save_hm_pdf:
        if use_dynamic_mask or save_heatmaps or save_hm_pdf:
            self.mask_generator = DynamicMaskGenerator(
                pipeline=self,
                use_dynamic_mask=use_dynamic_mask,
                mask_start=mask_start,
                mask_end=mask_end,
                save_heatmaps=save_heatmaps_dynamic,
                token_focus=token_focus,
                add_to_prompt=add_token_to_prompt,
                mask_layers_config=mask_layers_config or get_default_mask_config(),
                debug_dir=debug_dir,
                save_hm_pdf=save_hm_pdf,
                heatmap_interval=heatmap_interval,
                num_inference_steps=num_inference_steps,
                heatmap_mode=heatmap_mode,
            )

            # Setup hooks before denoising loop
            self.mask_generator.setup_hooks(prompt, class_tokens_mask)


        if use_branched_attention or save_heatmaps or save_hm_pdf:
            collect_attention_hooks(
                self,
                heatmap_mode,
                focus_token,
                class_tokens_mask,
                self.do_classifier_free_guidance,
                self._heatmaps,
                self._orig_forwards,
            )


        # 7. Get the update text embedding with the stacked ID embedding
        self._ensure_face_analyzer()

        if id_embeds is not None:
            id_embeds = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            # --- ADDED For training integration ---
            embeddings = []
            for ref in input_id_images:
                if isinstance(ref, torch.Tensor):
                    ref_img = ref.detach().cpu()
                    if ref_img.dim() == 3:
                        ref_img = ref_img.unsqueeze(0)
                    ref_img = ref_img[0]
                    ref_img = (ref_img * 0.5 + 0.5).clamp(0, 1)
                    ref_img = (ref_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_np = ref_img[:, :, ::-1]
                else:
                    img_np = np.array(ref.convert("RGB"))[:, :, ::-1]

                faces = analyze_faces(self._face_analyzer, img_np)
                if faces:
                    embedding = torch.from_numpy(faces[0]["embedding"]).float()
                else:
                    embedding = torch.zeros(512, dtype=torch.float32)
                embeddings.append(embedding)
            # --- ADDED For training integration ---

            id_embeds = torch.stack(embeddings, dim=0).unsqueeze(0).to(device=device, dtype=dtype)

        prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # 8. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        
        
        ##### NEW BRANCHED ATTENTION LOGIC #####

        if use_branched_attention:
            # Encode reference latents with AR-preserving resize + letterbox to (H,W)
            if input_id_images is not None and len(input_id_images) > 0:
                with torch.no_grad():
                    ref_pixels = self.image_processor.preprocess(
                        input_id_images, height=height, width=width
                    )  # (B,3,H,W) in [-1,1]
                    ref_pixels = ref_pixels.to(device=self._execution_device, dtype=latents.dtype)
                    ref_latents = self.vae.encode(ref_pixels).latent_dist.sample()
                    ref_latents = ref_latents * self.vae.config.scaling_factor
                # If multiple refs, average or pick first; keep shape [1,4,h_lat,w_lat]
                if ref_latents.shape[0] > 1:
                    ref_latents = ref_latents.mean(dim=0, keepdim=True)
                self._ref_latents_all = ref_latents
                
                from math import sqrt
                from PIL import Image
                import torch.nn.functional as F
                pil = input_id_images[0] if isinstance(input_id_images, (list, tuple)) else input_id_images
                ow, oh = pil.size
                self._ref_orig_size = (oh, ow)
                # scale to fit inside (height,width) while keeping AR; make divisible by 8
                s = min(width / ow, height / oh)
                rw = max(8, int(round(ow * s)) // 8 * 8)
                rh = max(8, int(round(oh * s)) // 8 * 8)
                pl = (width  - rw) // 2; pr = width  - rw - pl
                pt = (height - rh) // 2; pb = height - rh - pt
                self._ref_pad = (pl, pr, pt, pb)
                self._ref_scaled_size = (rh, rw)
                with torch.no_grad():
                    ref_pixels = self.image_processor.preprocess(pil, height=rh, width=rw)  # (1,3,rh,rw) in [-1,1]
                    ref_pixels = F.pad(ref_pixels, (pl, pr, pt, pb), value=0.0)            # letterbox to (H,W)
                    ref_pixels = ref_pixels.to(device=self._execution_device, dtype=latents.dtype)
                    ref_latents = self.vae.encode(ref_pixels).latent_dist.sample() * self.vae.config.scaling_factor
                self._ref_latents_all = ref_latents  # shape (1,4,H/8,W/8)
                
                
                # # --- optional: auto-generate mask for the reference image ---
                # if auto_mask_ref:
                #     try:
                #         # create a binary face mask (0/255) for the *original* reference image
                #         from .create_mask_ref import compute_face_mask_from_pil
                #         os.makedirs(debug_dir, exist_ok=True)
                #         _dst = os.path.join(debug_dir, "ref_mask_auto.png")
                #         m = compute_face_mask_from_pil(pil)  # uint8 HxW in {0,255}
                #         from PIL import Image as _PILImage
                #         _PILImage.fromarray(m).save(_dst)    # keep as 8-bit grayscale
                #         import_mask_ref = _dst               # used by aggregate_heatmaps_to_mask(..., suffix="_ref")
                #         print(f"[AutoMaskRef] Using generated ref mask at {_dst}")
                #     except Exception as e:
                #         print(f"[AutoMaskRef] Failed to generate ref mask (fallback to manual file): {e}")

                # ── Auto-generate reference face mask *early* so downstream uses it
                if auto_mask_ref:                    
                    # --- ADDED For training integration (FOLDER STUCTURE) ---
                    from src.model.photomaker_branched.create_mask_ref import compute_face_mask_from_pil
                    os.makedirs(debug_dir, exist_ok=True)
                    auto_ref_path = os.path.join(debug_dir, "auto_ref_mask.png")
                    mask_array = compute_face_mask_from_pil(pil)
                    Image.fromarray(mask_array).save(auto_ref_path)
                    import_mask_ref = auto_ref_path
                    print(f"[AutoMaskRef] Generated ref mask → {auto_ref_path}")
                else:
                    print(f"[AutoMaskRef] Using existing ref mask at {import_mask_ref}")

        

                # if id_embeds is not None:
                #     self._face_prompt_embeds = id_embeds.to(device=device, dtype=dtype)
                # else:

                # Do not stash 512-D id_embeds into cross-attn face prompt.
                # _face_prompt_embeds is only used in 'face' strategy (text-shaped).
                if id_embeds is None:
                    # Use the face-specific part of prompt_embeds after ID encoder
                    if class_tokens_mask is not None:
                        # Extract only the face tokens
                        face_indices = class_tokens_mask[0].nonzero(as_tuple=True)[0]
                        if len(face_indices) > 0:
                            self._face_prompt_embeds = prompt_embeds[:, face_indices, :]
                        else:
                            # Fallback: encode "face" text
                            face_text_embeds, _ = self.encode_prompt(
                                prompt="a face",
                                prompt_2="a face",
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                do_classifier_free_guidance=self.do_classifier_free_guidance,
                            )[:2]
                            self._face_prompt_embeds = face_text_embeds
                    else:
                        # No mask available, use full prompt embeds
                        self._face_prompt_embeds = prompt_embeds
                            
                # Set face embedding strategy (can be controlled via parameter)
                self.face_embed_strategy = face_embed_strategy  # 'id_embeds' or 'face'
                
                            
                # Also store as _reference_latents for the new approach
                self._reference_latents = self._ref_latents_all
                
                # Store the original RGB for debug
                self._ref_img = id_pixel_values[0] if id_pixel_values.dim() == 5 else id_pixel_values
                
                 # IMPORTANT: Create ref_noise with same generator as latents for consistency
                if not hasattr(self, '_ref_noise'):
                    # --- ADDED For training integration ---
                    gen = None
                    if generator is not None:
                        cand = generator[0] if isinstance(generator, (list, tuple)) and len(generator) > 0 else generator
                        if isinstance(cand, torch.Generator):
                            if hasattr(cand, "device") and cand.device.type == device.type:
                                gen = cand
                            else:
                                try:
                                    gen = torch.Generator(device=device)
                                    gen.set_state(cand.get_state())
                                except Exception:
                                    gen = None
                    # --- ADDED For training integration ---
                    self._ref_noise = torch.randn(
                        self._ref_latents_all.shape,
                        generator=gen,
                        device=device,
                        dtype=self._ref_latents_all.dtype
                    )
                
            else:
                # Fallback: use initial latents as reference
                self._ref_latents_all = latents.clone()
                self._reference_latents = self._ref_latents_all
                self._ref_img = None
            
            # Canonicalize strategy & keep for per-step call
            fes = (face_embed_strategy or "face").lower()
            if fes in {"faceanalysis"}:   # old CLI synonyms
                fes = "face"
            self.face_embed_strategy = fes
            # Precompute “face text” once if needed
            if self.face_embed_strategy == "face":
                self._face_prompt_embeds = encode_face_prompt(
                    self, device=device, batch_size=batch_size,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                ).to(device)
            # Cache ID token positions for masking (used in id_embeds mode)
            self._id_token_idx = class_tokens_mask[0].nonzero(as_tuple=True)[0]

            # --- NEW: cache raw 2048-D PhotoMaker ID features (not fused text) ---
            if id_pixel_values is not None and hasattr(self, "id_encoder"):
                pm_feats = self.id_encoder.extract_id_features(
                    id_pixel_values.to(device=self.device, dtype=prompt_embeds.dtype),
                    class_tokens_mask=class_tokens_mask
                )  # [B,2048]
                self._pm_id_embeds_2048 = pm_feats.to(device=self.device, dtype=self.unet.dtype)


        ##### END NEW BRANCHED ATTENTION LOGIC #####

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
            
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        
        #### NEW BRANCHED ATTENTION LOGIC ####
        # Store original processors once before denoising loop
        # Prepare reference latents once
        if use_branched_attention:
            #  # Store face prompt embeds for cross-attention
            # Do NOT overwrite with id_embeds (512-D) — keep text embeds (2048-D)
            #  self._face_prompt_embeds = id_embeds
             
             # Prepare reference latents if not already done
             if not hasattr(self, '_ref_latents_all'):
                 if id_pixel_values is not None:
                     self._ref_latents_all = prepare_reference_latents(
                         self,
                         id_pixel_values[0, 0] if id_pixel_values.dim() == 5 else id_pixel_values[0],
                         height,
                         width,
                         latents.dtype,
                         generator
                     )
                 else:
                     # Fallback: use initial latents as reference
                     self._ref_latents_all = latents.clone()
               

        #### NEW BRANCHED ATTENTION LOGIC ####
        
        

        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # PoseAdapt: remember the user-provided ratio once per call + logging guards
        self._pose_user_ratio = float(getattr(self, "pose_adapt_ratio", 0.25))
        _pose_forced_logged = False
        _pose_relaxed_logged = False



        # 11.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 12. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            prev_mode = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # ----------------------------------------------------------------
                #  Initialise per-step branch tensors (may stay None this step)
                # ----------------------------------------------------------------
                noise_face: torch.Tensor | None = None
                noise_bg:   torch.Tensor | None = None
                
  
                
                # ───── choose prompt-conditioning for this step (must come FIRST) ─────
                # Before PhotoMaker merge: use text-only to avoid early leakage.
                # After that (and for branched attention), keep the ID-enhanced embeddings.
                
                # Initial defaults; will be overridden by mode below
                # use_text_only = (i <= start_merge_step)
                use_text_only = (i <= photomaker_start_step)
                base_prompt   = prompt_embeds_text_only if use_text_only else prompt_embeds
                base_pooled   = pooled_prompt_embeds_text_only if use_text_only else pooled_prompt_embeds
                
                # ── PoseAdapt pre-PhotoMaker: force ratio=1.0, relax back after ──
                if force_par_before_pm:
                    _desired_par = 1.0 if use_text_only else self._pose_user_ratio
                else:
                    _desired_par = self._pose_user_ratio
                    
                if getattr(self, "pose_adapt_ratio", None) != _desired_par:
                    self.pose_adapt_ratio = _desired_par
                    if _desired_par == 1.0 and not _pose_forced_logged:
                        print(f"[PoseAdapt] Forcing POSE_ADAPT_RATIO=1.0 until photomaker_start_step={photomaker_start_step}")
                        _pose_forced_logged = True
                    elif _desired_par != 1.0 and not _pose_relaxed_logged:
                        print(f"[PoseAdapt] Relaxing POSE_ADAPT_RATIO to user value {self._pose_user_ratio:.2f} at step {i}")
                        _pose_relaxed_logged = True
                
                
                # # --- print when switching approach (half-open intervals) ---
                # # [0, min) => NO_ID
                # # [start_merge, branched_start) => PHOTOMAKER
                # # [branched_start, start_merge) => BRANCHED
                # # afterwards => whichever applies
                # bs = branched_attn_start_step
                # if bs is None:
                #     bs = 10**9  # sentinel if not using branched
                # m1 = min(start_merge_step, bs)
                # if i < m1: mode = "NO_ID"
                # elif start_merge_step <= i < bs: mode = "PHOTOMAKER"
                # elif bs <= i < start_merge_step: mode = "BRANCHED"
                # else: mode = ("BRANCHED" if i >= bs else "PHOTOMAKER")

                # --- unified schedule (half-open intervals) ---
                # [0, a) → NO_ID; [a, b) → early; [b, ∞) → late
                # bs = branched_attn_start_step
                # a  = min(start_merge_step, bs)
                # b  = max(start_merge_step, bs)
                # early = "PHOTOMAKER" if start_merge_step < bs else "BRANCHED"
                # late  = "BRANCHED"   if start_merge_step < bs else "PHOTOMAKER"
                # if i < a:        mode = "NO_ID"
                # elif i < b:      mode = early
                # else:            mode = late

                # early = "PHOTOMAKER" if start_merge_step < bs else "BRANCHED"
                # # New 4-mode schedule:
                # # [0, a) → NO_ID; [a, b) → early; [b, ∞) → BOTH (PM + Branched)
                # if i < a:
                #     mode = "NO_ID"
                # elif i < b:
                #     mode = early
                # else:
                #     mode = "BOTH"

                bs = branched_attn_start_step
                # sm = start_merge_step
                sm = photomaker_start_step
                a  = min(sm, bs)
                b  = max(sm, bs)
                bsm = getattr(self, "branched_start_mode", "both").lower()  # "both" or "branched"
                # Case A: sm < bs  → NO_ID → PHOTOMAKER → (BOTH/BRANCHED)
                # Case B: bs <= sm → NO_ID → (BOTH/BRANCHED) → PHOTOMAKER
                if i < a:
                    mode = "NO_ID"
                elif sm < bs:
                    mode = "PHOTOMAKER" if i < b else ("BOTH" if bsm == "both" else "BRANCHED")
                else:
                    mode = ("BOTH" if bsm == "both" else "BRANCHED") if i < b else "PHOTOMAKER"


                if mode != prev_mode:
                    # print(f"[Switch] step {int(i)} → {mode}  (start_merge_step={int(start_merge_step)}, branched_attn_start_step={int(bs)})")
                    print(f"[Switch] step {int(i)} → {mode}  (photomaker_start_step={int(photomaker_start_step)}, branched_attn_start_step={int(bs)})")
                    prev_mode = mode

                # # Enforce prompts by mode (PHOTOMAKER/BOTH → ID; NO_ID/BRANCHED → text-only)
                # if mode in ("PHOTOMAKER", "BOTH"):
                #     use_text_only = False
                #     base_prompt   = prompt_embeds
                #     base_pooled   = pooled_prompt_embeds
                # else:
                #     use_text_only = True
                #     base_prompt   = prompt_embeds_text_only
                #     base_pooled   = pooled_prompt_embeds_text_only

                # Prompts by mode: PHOTOMAKER/BOTH → ID-enhanced, else text-only
                if mode in ("PHOTOMAKER", "BOTH"):
                    base_prompt = prompt_embeds
                    base_pooled = pooled_prompt_embeds


                current_prompt_embeds = (
                    torch.cat([negative_prompt_embeds, base_prompt], dim=0)
                    if self.do_classifier_free_guidance else base_prompt
                )
                add_text_embeds = (
                    torch.cat([negative_pooled_prompt_embeds, base_pooled], dim=0)
                    if self.do_classifier_free_guidance else base_pooled
                )

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                

                ##### NEW BRANCHED ATTENTION LOGIC #####
                
                # ------------------------------------------------------------
                #  Activate branched-attention after <branched_attn_start_step>
                # ------------------------------------------------------------
                mask4 = None
                mask4_ref = None

                # ============================================================
                #  Update dynamic mask if enabled
                # ============================================================
                # if use_dynamic_mask:
                if hasattr(self, 'mask_generator'):
                    # self.mask_generator.update_mask(i)
                    self.mask_generator.update_mask(i, latents)
                                
                
                # if use_branched_attention and i >= branched_attn_start_step:
                # # Only run branched block when the schedule says so
                # branched_active = use_branched_attention and (mode == "BRANCHED")
                
                # Only run branched block when the schedule says so (BRANCHED or BOTH)
                branched_active = use_branched_attention and (mode in ("BRANCHED", "BOTH"))

                
                if branched_active:
                    # Build noise mask - use dynamic if available, otherwise use import_mask
                    if use_dynamic_mask:
                        mask_np, mask_tensor = self.mask_generator.get_mask_for_pipeline()
                        if mask_np is not None:
                            self._face_mask = mask_np
                            self._face_mask_t = mask_tensor
                    
                    if not hasattr(self, '_face_mask') or self._face_mask is None:
                        aggregate_heatmaps_to_mask(self, mask_mode, import_mask, suffix="")

                    mask4     = prepare_mask4(self, latent_model_input, suffix="")
                    
                    # Build ref mask from import_mask_ref (unchanged)
                    aggregate_heatmaps_to_mask(self, mask_mode, import_mask_ref, suffix="_ref")


                    mask4_ref = prepare_mask4(self, latent_model_input, suffix="_ref")
                    
                    # Quick checks & one-off debug (moved here)
                    if mask4 is not None and mask4_ref is not None and (i == branched_attn_start_step or i % 10 == 0):
                        print(f"[PL] step={i}  mask_gen>0.5={(mask4>0.5).float().mean().item():.4f}  mask_ref>0.5={(mask4_ref>0.5).float().mean().item():.4f}")
                        md = (mask4 - mask4_ref).abs().mean().item()
                        if md < 0.01:
                            print(f"[Warning] Noise and ref masks are nearly identical (diff={md:.4f})")
                    debug_reference_latents_once(self, mask4_ref, debug_dir)
                    if i == branched_attn_start_step:
                        # --- MODIFIED For training integration (hm_debug) ---
                        base_debug_dir = Path(debug_dir) if debug_dir is not None else None  
                        if base_debug_dir is not None:
                            ref_masks = mask4_ref
                            total_outputs = latents.shape[0]
                            if ref_masks.dim() == 4 and ref_masks.shape[0] == total_outputs:
                                ref_masks_iter = [ref_masks[idx:idx+1] for idx in range(total_outputs)]
                            else:
                                ref_masks_iter = [ref_masks] * total_outputs
                            for idx, mask_ref_single in enumerate(ref_masks_iter):
                                per_image_dir = base_debug_dir if total_outputs == 1 else base_debug_dir / f"{idx:02d}"
                                per_image_dir.mkdir(parents=True, exist_ok=True)
                                save_debug_ref_latents(self, str(per_image_dir))
                                save_debug_ref_mask_overlay(self, mask_ref_single, str(per_image_dir))
                        # --- MODIFIED For training integration (hm_debug) ---
                        else:
                            save_debug_ref_latents(self, debug_dir)
                            save_debug_ref_mask_overlay(self, mask4_ref, debug_dir)
                        print(f"[Debug] Step {i}: Ref mask overlay saved.")

                        
                    # # # NEW: if branched starts before PhotoMaker, keep face branch text-only until start_merge_step
                    # # fes_step = self.face_embed_strategy
                    # # if fes_step in {"id", "id_embeds"} and i < start_merge_step:
                    # #     fes_step = "face"
                    # fes_step = self.face_embed_strategy

                    # If branched runs before PhotoMaker starts, suppress ID in face branch until merge starts
                    # fes_step = self.face_embed_strategy
                    # if fes_step in {"id", "id_embeds"} and i < start_merge_step:
                    #     fes_step = "face"

                    fes_step = self.face_embed_strategy
                    # In BOTH mode we allow ID/id_embeds even before start_merge_step
                    # In BOTH mode we allow ID/id_embeds even before photomaker_start_step
                    # if fes_step in {"id", "id_embeds"} and i < start_merge_step and mode != "BOTH":
                    if fes_step in {"id", "id_embeds"} and i < photomaker_start_step and mode != "BOTH":
                        fes_step = "face"
                        
                    # pick face embeddings per strategy
                    face_ehs = (
                        current_prompt_embeds
                        # if self.face_embed_strategy in {"id", "id_embeds"}
                        if fes_step in {"id", "id_embeds"}
                        else self._face_prompt_embeds
                    )

                    id_embeds_2048 = getattr(self, "_pm_id_embeds_2048", None) if fes_step == "id_embeds" else None
                    print('[DEBUG] id_embeds_2048:', id_embeds_2048)



                    # Call the new two_branch_predict function
                    # noise_pred, noise_face, noise_bg = two_branch_predict(
                    # Apply mask-merge only from merge_start_step onwards
                    _mask4     = mask4     if i >= merge_start_step else None
                    _mask4_ref = mask4_ref if i >= merge_start_step else None


                    # Build face-branch encoder_hidden_states from 2048-D PM ID features
                    id_face_ehs = None
                    if fes_step == "id_embeds":
                        pm = getattr(self, "_pm_id_embeds_2048", None)  # [B, 2048]
                        if pm is not None:
                            # match [B or 2B, seq_len, dim] of current_prompt_embeds
                            seq_len = current_prompt_embeds.shape[1]
                            dim = current_prompt_embeds.shape[2]
                            B = pm.shape[0]
                            pos = pm.unsqueeze(1).expand(B, seq_len, dim)          # [B, L, D]
                            if self.do_classifier_free_guidance:
                                neg = torch.zeros_like(pos)                          # [B, L, D]
                                id_face_ehs = torch.cat([neg, pos], dim=0)           # [2B, L, D]
                            else:
                                id_face_ehs = pos                                     # [B, L, D]
                            id_face_ehs = id_face_ehs.to(device=current_prompt_embeds.device,
                                                         dtype=current_prompt_embeds.dtype)

                    noise_pred, noise_face, noise_bg = two_branch_predict(
                        self,  # pipeline
                        latent_model_input,  # latent_model_input
                        t=t,
                        prompt_embeds=current_prompt_embeds, 
                        added_cond_kwargs=added_cond_kwargs,
                        mask4=mask4,
                        mask4_ref=mask4_ref,
                        reference_latents=self._ref_latents_all,
                        # For "face" → use text; for "id_embeds" → use PM ID features as pseudo-tokens
                        face_prompt_embeds=(self._face_prompt_embeds if fes_step == "face" else id_face_ehs),
                        class_tokens_mask=class_tokens_mask,
                        # face_embed_strategy=self.face_embed_strategy,
                        face_embed_strategy=fes_step,
                        # We no longer inject ID via processors for this mode; Cross-Attn uses id_face_ehs.
                        id_embeds=None,
                        step_idx=i,
                        scale=photomaker_scale,  # Use the new parameter
                        timestep_cond=timestep_cond,
                    )

                    # Debug: check if noise_pred has expected values
                    if i < (branched_attn_start_step + 3):
                        print(f"[Debug] Step {i}: noise_pred stats - "
                              f"mean={noise_pred.mean().item():.4f}, "
                              f"std={noise_pred.std().item():.4f}, "
                              f"min={noise_pred.min().item():.4f}, "
                              f"max={noise_pred.max().item():.4f}")                    
                    
                    # Clear any temporary state
                    if hasattr(self, '_kv_override'):
                        self._kv_override = None
                        
                else:
                    # Standard single-branch prediction
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=current_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
            
                
                ##### NEW BRANCHED ATTENTION LOGIC ##### 
    
                # ## TODO: need to add mask_ref here?
                            
                # optional PNG previews of two branches
                if i % 10 == 0 or i == num_inference_steps - 1: # every 10 steps
                    if "mask4" in locals() and noise_face is not None:
                        base_debug_dir = Path(debug_dir) if debug_dir is not None else None
                        if base_debug_dir is not None:
                            total_outputs = latents.shape[0]
                            for idx, latent_sample in enumerate(latents):
                                per_image_dir = base_debug_dir if total_outputs == 1 else base_debug_dir / f"{idx:02d}"
                                per_image_dir.mkdir(parents=True, exist_ok=True)
                                mask_slice = mask4[idx:idx+1] if mask4.shape[0] > idx else mask4
                                save_branch_previews(
                                    self,
                                    latent_sample.unsqueeze(0),
                                    noise_pred,
                                    mask_slice,
                                    t,
                                    i,
                                    str(per_image_dir),
                                    extra_step_kwargs,
                                )
                        else:
                            save_branch_previews(
                                self,
                                latents,
                                noise_pred,
                                mask4,
                                t,
                                i,
                                debug_dir,
                                extra_step_kwargs,
                            )


                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)             

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
                
        ### NEW BRANCHED ATTENTION LOGIC ###
        ## CLEANUP ##
        if use_branched_attention:
            # Restore original processors
             restore_original_processors(self)
             
             # Clean up temporary attributes
             for attr in ['_reference_latents', '_face_prompt_embeds', '_ref_latents_all']:
                 if hasattr(self, attr):
                     delattr(self, attr)

            #  # Save heatmap PDF after inference completes
            #  if hasattr(self, 'mask_generator') and self.mask_generator.save_hm_pdf:
        # Save heatmap PDF and cleanup (moved outside branched attention block)
        if hasattr(self, 'mask_generator'):
             if self.mask_generator.save_hm_pdf:

                 # Get final image from last latents
                 final_image = None
                 if latents is not None:
                     with torch.no_grad():
                         lat_scaled = (latents[0:1] / self.vae.config.scaling_factor).to(self.vae.dtype)
                         img = self.vae.decode(lat_scaled).sample[0]
                         final_image = ((img.float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                 self.mask_generator.save_heatmap_pdf(final_image)

             # Cleanup dynamic mask generator
            #  if hasattr(self, 'mask_generator'):
            #      self.mask_generator.cleanup()
             self.mask_generator.cleanup()

            
        ### NEW BRANCHED ATTENTION LOGIC ###

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        # if self.watermark is not None:
        #     image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
    
    ### NEW BRANCHED ATTENTION LOGIC ###
    @property
    def cross_attention_kwargs(self):
        """Get cross attention kwargs if they exist."""
        return getattr(self, '_cross_attention_kwargs', None)

    @cross_attention_kwargs.setter
    def cross_attention_kwargs(self, value):
        """Set cross attention kwargs."""
        self._cross_attention_kwargs = value
    ### NEW BRANCHED ATTENTION LOGIC ###
    
    # --- ADDED For training integration ---
    def _ensure_face_analyzer(self):
        if hasattr(self, "_face_analyzer"):
            return
        self._face_analyzer = FaceAnalysis2(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        try:
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            self._face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    # --- ADDED For training integration ---


# --- ADDED For training integration (FOLDER STUCTURE) ---
class PhotomakerBranchedPipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        kwargs = dict(kwargs)
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])

        unwrapped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        scheduler = DDIMScheduler.from_pretrained(
            kwargs["pretrained_model_name_or_path"],
            subfolder="scheduler",
        )

        # --- ADDED For training integration (CONFIG DEFAULTS) ---
        photomaker_start_step_cfg = kwargs.pop("photomaker_start_step", 10)
        merge_start_step_cfg = kwargs.pop("merge_start_step", 10)
        branched_attn_start_step_cfg = kwargs.pop("branched_attn_start_step", 10)
        branched_start_mode_cfg = kwargs.pop("branched_start_mode", "both")
        pose_adapt_ratio_cfg = kwargs.pop(
            "pose_adapt_ratio",
            getattr(unwrapped_model, "pose_adapt_ratio", 0.25),
        )
        ca_mixing_for_face_cfg = kwargs.pop(
            "ca_mixing_for_face",
            getattr(unwrapped_model, "ca_mixing_for_face", True),
        )
        face_embed_strategy_cfg = kwargs.pop(
            "face_embed_strategy",
            getattr(unwrapped_model, "face_embed_strategy", "face"),
        )
        # --- ADDED For training integration (CONFIG DEFAULTS) ---

        pipeline = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            scheduler=scheduler,
            tokenizer=unwrapped_model.tokenizer,
            tokenizer_2=unwrapped_model.tokenizer_2,
            text_encoder=unwrapped_model.text_encoder,
            text_encoder_2=unwrapped_model.text_encoder_2,
            unet=unwrapped_model.unet,
            vae=unwrapped_model.vae,
            *args,
            **kwargs,
        )
        pipeline.set_progress_bar_config(disable=True)

        pipeline.num_tokens = getattr(unwrapped_model, "num_tokens", 2)
        pipeline.pm_version = "v2"
        pipeline.trigger_word = unwrapped_model.trigger_word

        pipeline.id_image_processor = CLIPImageProcessor()
        pipeline.id_encoder = unwrapped_model.id_encoder

        pipeline.pose_adapt_ratio = pose_adapt_ratio_cfg
        pipeline.ca_mixing_for_face = ca_mixing_for_face_cfg
        pipeline.face_embed_strategy = face_embed_strategy_cfg

        pipeline.tokenizer.add_tokens([pipeline.trigger_word], special_tokens=True)
        pipeline.tokenizer_2.add_tokens([pipeline.trigger_word], special_tokens=True)

        # --- ADDED For training integration (CONFIG DEFAULTS) ---
        pipeline._config_photomaker_start_step = photomaker_start_step_cfg
        pipeline._config_merge_start_step = merge_start_step_cfg
        pipeline._config_branched_attn_start_step = branched_attn_start_step_cfg
        pipeline._config_branched_start_mode = branched_start_mode_cfg
        pipeline._config_pose_adapt_ratio = pose_adapt_ratio_cfg
        pipeline._config_ca_mixing_for_face = ca_mixing_for_face_cfg
        pipeline._config_face_embed_strategy = face_embed_strategy_cfg

        return pipeline
