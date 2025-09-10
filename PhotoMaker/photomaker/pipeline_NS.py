# photomaker/pipeline_NS.py

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
import PIL
from PIL import Image   

import torch
from transformers import CLIPImageProcessor

from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from diffusers import StableDiffusionXLPipeline
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

# NEW IMPORTS 30 JUL
import warnings, math, types
import matplotlib.cm as cm  
from pathlib import Path
from PIL import ImageDraw, ImageFont   
import cv2
import os
# mask helper cloned from `attn_hm_NS_nosm7.py`
# ──────────────────────────────────────────────────────────────────────────────
#  Shared utils
# ──────────────────────────────────────────────────────────────────────────────
from .mask_utils import (
    MASK_LAYERS_CONFIG,
    compute_binary_face_mask,         # weighted-union (legacy / spec)
    _resize_map,
    simple_threshold_mask,            # NEW helper ❶ (see mask_utils.py patch)
)


# `identity` will now use the inline hook copied from the known-good version;
# token-based maps still rely on the helper from heatmap_utils.
from .heatmap_utils import build_hook_focus_token, build_hook_identity


import numpy as np

# import nn
import torch.nn as nn
import torch.nn.functional as F



from . import (
    PhotoMakerIDEncoder, # PhotoMaker v1
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken, # PhotoMaker v2
)


# --------------------------------------------------------------------------- #
#  Debug helpers
# --------------------------------------------------------------------------- #

DEBUG_DIR = os.getenv("PM_DEBUG_DIR", "./branched_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

def _save_gray(arr: np.ndarray, fp: str):
    """
    Save a 2-D numpy array as an 8-bit grayscale PNG.
    """
    if arr.ndim != 2:
        return
    arr = arr.astype(np.float32)
    arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-8)).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(fp)




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
    

class PhotoMakerStableDiffusionXLPipeline2(StableDiffusionXLPipeline):
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

        # load lora into models
        print(f"Loading PhotoMaker {pm_version} components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")
        print(f'[DEBUG] Using upgraded pipeline_NS.py') # ADD 30 JUL

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
        start_merge_step: int = 10, # TODO: change to `style_strength_ratio` in the future
        class_tokens_mask: Optional[torch.LongTensor] = None,
        id_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        # **kwargs,
        
        # Added parameters (for branched attention) # NEW 30 JUL
        use_branched_attention: bool = False,
        face_embed_strategy: str = "faceanalysis",  # "faceanalysis" or "heatmap"
        save_heatmaps: bool = False,                # ← NEW flag
        # ---------------- NEW SWITCHES ----------------
        heatmap_mode: str =  "identity", # "token",  # "identity", # "token", # "identity",             # "identity" | "token"
        focus_token: str = "face",
        mask_mode: str = "spec", # "simple", # "spec",                    # "spec" | "simple"
        branched_attn_start_step: int = 10,
        # ────── DEBUG MASK STRIP ───────────────────────────────
        debug_save_masks: bool = False,
        mask_save_dir: str = "hm_debug",
        mask_interval: int = 5,
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

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

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
        
        if class_tokens_mask is None:
            print("[WARN] class_tokens_mask is None – will fallback to last "
                f"{self.num_tokens} tokens")
        else:
            idx = class_tokens_mask[0].nonzero(as_tuple=True)[1].tolist()
            print(f"[DBG] class_tokens_mask indices = {idx}")
                
        
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


        # NEW 30 JUL
        # Decide identity embedding usage based on strategy
        if use_branched_attention and face_embed_strategy.lower() == "heatmap":
            # Skip using external face recognition embedding (ArcFace); rely on image features only
            id_embeds = None
        if id_embeds is not None:
            id_embeds = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
            prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
        else:
            prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
        # NEW 30 JUL


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

        # 11. Denoising loop ──────────────────────────────────────────────────
        # Prepare branched-attention bookkeeping

        # ─── DEBUG ①: echo run-time flags ────────────────────────────────────
        print(
            f"[DEBUG] branched={use_branched_attention}  "
            f"save_heatmaps={save_heatmaps}  "
            f"start_step={branched_attn_start_step}  "
            f"strategy={face_embed_strategy}"
        )
        
        
        # ------------ validate new switches ------------
        heatmap_mode  = heatmap_mode.lower()
        if heatmap_mode not in ("identity", "token"):
            raise ValueError("heatmap_mode must be 'identity' or 'token'")

        mask_mode = mask_mode.lower()
        if mask_mode not in ("spec", "simple"):
            raise ValueError("mask_mode must be 'spec' or 'simple'")

        face_embed_strategy   = face_embed_strategy.lower()
        use_branched_attention = bool(use_branched_attention)
        collect_hm = use_branched_attention or save_heatmaps  # ← NEW
        branched_attn_start_step = int(branched_attn_start_step)

        # collect maps across several steps so that we have something to
        # aggregate when we reach `branched_attn_start_step`
        attn_maps_current: Dict[str, List[np.ndarray]] = {}
        orig_attn_forwards: Dict[str, Callable] = {}

        # if use_branched_attention:
        if collect_hm:            # allow --save_heatmaps to capture maps
            # make sure we run *regular* PyTorch attention
            if hasattr(self.unet, "attn_processors"):          # diffusers ≥ 0.25
                self.unet.set_attn_processor(dict(self.unet.attn_processors))
            else:                                              # legacy (< 0.25)
                self.unet.set_attn_processor(self.unet.attn_processor)
            
            from diffusers.models.attention_processor import Attention as CrossAttention
            wanted_layers = {spec["name"] for spec in MASK_LAYERS_CONFIG}

            # # ❶ choose hook builder -----------------------------------------
            # if heatmap_mode == "identity":
            #     hook_builder = lambda ln, mod: build_hook_identity(
            #         ln, mod, wanted_layers, class_tokens_mask,
            #         self.num_tokens, attn_maps_current, orig_attn_forwards,
            #         self.do_classifier_free_guidance)
            
            if heatmap_mode == "identity":
                # use the rock-solid implementation that already lives in heatmap_utils
                hook_builder = lambda ln, mod: build_hook_identity(
                    ln,                                  # layer name
                    mod,                                 # module itself
                    wanted_layers,                       # set of layers we care about
                    class_tokens_mask,                   # marks the duplicated ID tokens
                    self.num_tokens,                     # e.g. 2 for PM-v2
                    attn_maps_current,                   # dict that collects numpy maps
                    orig_attn_forwards,                  # backup dict for clean-up
                    self.do_classifier_free_guidance,    # CFG flag – same as old code
                )
                        
                
            
            else:  # "token" (auxiliary prompt)
                # build `focus_latents` & token indices ONCE
                aux_prompt   = f"a {focus_token}"
                with torch.no_grad():
                    focus_lat, *_ = self.encode_prompt(
                        prompt      = aux_prompt,
                        device      = device,
                        num_images_per_prompt = 1,
                        do_classifier_free_guidance = False,
                    )
                tok  = self.tokenizer or self.tokenizer_2
                idsA = tok(aux_prompt        , add_special_tokens=False).input_ids
                idsW = tok(" " + focus_token , add_special_tokens=False).input_ids
                def _find_sub(seq, sub):
                    for i in range(len(seq)-len(sub)+1):
                        if seq[i:i+len(sub)] == sub:
                            return list(range(i,i+len(sub)))
                    return []
                token_idx_global = _find_sub(idsA, idsW)
                if not token_idx_global:
                    raise RuntimeError(f"focus token '{focus_token}' not found")

                hook_builder = lambda ln, mod: build_hook_focus_token(
                    ln, mod, wanted_layers, focus_lat, token_idx_global,
                    attn_maps_current, orig_attn_forwards,
                    self.do_classifier_free_guidance)

            
            
            hook_count = 0
            for n, m in self.unet.named_modules():          # ← **keep this**
                if isinstance(m, CrossAttention) and n in wanted_layers:
                    m.forward = hook_builder(n, m)
                    hook_count += 1

            print(f"[DEBUG] wanted_layers={len(wanted_layers)}  "
                  f"hooks_loaded={hook_count}")

        # now the scheduler warm-up calculation
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

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

        # ── prepare containers for DEBUG strip ──────────────────────────
        if debug_save_masks:
            Path(mask_save_dir).mkdir(parents=True, exist_ok=True)
            mask_frames: list[Image.Image] = []
            step_labels: list[str] = []


        with self.progress_bar(total=num_inference_steps) as progress_bar:

            
            # Start denoising loop
            for i, t in enumerate(timesteps):

                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    ) if self.do_classifier_free_guidance else prompt_embeds_text_only
                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0
                        ) if self.do_classifier_free_guidance else pooled_prompt_embeds_text_only
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    ) if self.do_classifier_free_guidance else prompt_embeds
                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                        ) if self.do_classifier_free_guidance else pooled_prompt_embeds
                    
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                    
                    
                # ---------------------------------------------------------- #
                #  diagnostics
                # ---------------------------------------------------------- #
                # if i % 1 == 0:  # every step
                #     print(
                #         f"[STEP {i:03d}]  σ={t.item():>6}  "
                #         f"latents={tuple(latents.shape)}[{latents.dtype}]"
                #     )
                #     print(f"        prompt emb norm={current_prompt_embeds.norm():.4f}")
                
                # print(f"[DBG] hook alive?  any('{n}' in orig_attn_forwards for n in wanted_layers) = "
                #     f"{any(n in orig_attn_forwards for n in wanted_layers)}")
                
                # # NEW ─── how many maps do we have for every layer right now?
                # if collect_hm:
                #     sizes = {k: len(v) for k, v in attn_maps_current.items()}
                #     print(f"[DBG] attn_maps_current sizes: {sizes}")

                
                # ───────────────────────────────────────────────────────────
                #  ❶ Drop the maps gathered **before** the ID-tokens are
                #     merged in – they are pure noise.
                # ───────────────────────────────────────────────────────────
                if collect_hm and i == start_merge_step - 1:
                    attn_maps_current.clear()            # flush noise

                # ───────────────────────────────────────────────────────────
                #  ❷ Save a visual overlay only after the merge step
                # ───────────────────────────────────────────────────────────
                if (
                    collect_hm
                    and i >= start_merge_step            # NEW guard
                    and (i % mask_interval == 0 or i == len(timesteps) - 1)
                    and attn_maps_current                # anything captured yet?
                ):
                    print(f"[HM] step={i:3d}   layers with maps={len(attn_maps_current)}")
    
                    print(f"[DEBUG] saving attention maps for step {i:03d}...")
                    for ln, maps in attn_maps_current.items():
                        if not maps:                 # safety guard
                            print(f"[DEBUG] no maps for layer {ln} at step {i:03d}")
                            continue
                        amap = maps[-1]              # most-recent map this step
                        safe_ln = ln.replace("/", "_").replace(".", "_")
  
                        fname_png = f"{DEBUG_DIR}/heat_{i:03d}_{safe_ln}.png"
                        _save_gray(amap, fname_png)          # keep PNG as before


                    # ── 1. consolidate -> snapshot (mean over heads) ─────────
                    # snapshot = {ln: np.stack(maps).mean(0)
                    #             for ln, maps in attn_maps_current.items() if maps}
                    
                    # ── 1. consolidate -> snapshot (mean over heads) ─────────
                    snapshot = {}
                    for ln, lst in attn_maps_current.items():
                        # keep only proper 2-D maps
                        lst2 = [m for m in lst if m.ndim == 2]
                        if not lst2:
                            continue
                        # bring every map to the largest H×H in that list
                        max_H = max(m.shape[0] for m in lst2)
                        aligned = [
                            m if m.shape[0] == max_H else _resize_map(m, max_H)
                            for m in lst2
                        ]
                        snapshot[ln] = np.stack(aligned, axis=0).mean(0)
            
                    # --- DEBUG: quality of the consolidated map -----------------
                    print("[SNAP]", {ln: (m.max(), m.mean()) for ln, m in snapshot.items()})
                    
                    if not snapshot:
                        pass
                    else:
                        # ── allocate bookkeeping on first hit ───────────────
                        if not hasattr(self, "_hm_layers"):
                            self._hm_layers  = list(snapshot)          # keep *all* layers
                            self._heatmaps   = {ln: [] for ln in self._hm_layers}
                            self._step_tags  = []

                        # ── 2. decode current latent → base RGB 1024² ──────
                        vae_dev = next(self.vae.parameters()).device
                        img = self.vae.decode(
                            (latents / self.vae.config.scaling_factor)
                            .to(device=vae_dev, dtype=self.vae.dtype)
                        ).sample[0]
                        img_np = ((img.float() / 2 + 0.5)
                                  .clamp_(0, 1)
                                  .permute(1, 2, 0)
                                  .cpu()
                                  .numpy() * 255).astype(np.uint8)

                        cmap  = cm.get_cmap("jet")
                        font  = ImageFont.load_default()

                        for ln in self._hm_layers:
                            if ln not in snapshot:
                                continue
                            amap = snapshot[ln]
                            amap_n = amap / amap.max() if amap.max() > 0 else amap

                            hmap = (cmap(amap_n)[..., :3] * 255).astype(np.uint8)
                            hmap = np.array(Image.fromarray(hmap)
                                            .resize((1024, 1024), Image.BILINEAR))
                            heat_np = (0.5 * img_np + 0.5 * hmap).astype(np.uint8)
                            heat_im = Image.fromarray(heat_np)

                            # ----- numeric 5×5 grid ------------------------
                            draw   = ImageDraw.Draw(heat_im)
                            H_blk  = amap.shape[0] // 5
                            W_blk  = amap.shape[1] // 5
                            vis_h  = 1024 // 5
                            vis_w  = 1024 // 5
                            for bi in range(5):
                                for bj in range(5):
                                    blk = amap[bi*H_blk:(bi+1)*H_blk,
                                                bj*W_blk:(bj+1)*W_blk]
                                    mv  = blk.mean()
                                    cx  = bj*vis_w + vis_w//2
                                    cy  = bi*vis_h + vis_h//2
                                    txt = f"{mv:.2f}"
                                    tw, th = draw.textbbox((0,0), txt, font=font)[2:]
                                    draw.text((cx-tw//2, cy-th//2), txt,
                                              font=font, fill="white",
                                              stroke_width=1, stroke_fill="black")

                            self._heatmaps[ln].append(heat_im)

                        self._step_tags.append(i)

                        self.__dict__.setdefault("_heat_pngs", []).append(fname_png)
                        
                        # reset cache so the next step starts fresh
                        attn_maps_current.clear()


                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

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

                
                # ───────────────────────── update mask BEFORE user callback ──
                # if (
                #     collect_hm
                #     and i >= start_merge_step
                #     and (i % mask_interval == 0 or i == len(timesteps) - 1)
                #     and attn_maps_current
                # ):

                if (
                    collect_hm
                    # start drawing from step 0 (no ≥ start_merge_step gate)
                    and (i % mask_interval == 0 or i == len(timesteps) - 1)
                    and attn_maps_current
                ):
                    from .mask_utils import _resize_map
                    snapshot = {}
                    for ln, lst in attn_maps_current.items():
                        maps2d = [m for m in lst if m.ndim == 2]
                        if not maps2d:
                            continue
                        max_H = max(m.shape[0] for m in maps2d)
                        aligned = [
                            m if m.shape[0] == max_H else _resize_map(m, max_H)
                            for m in maps2d
                        ]
                        snapshot[ln] = np.stack(aligned, 0).mean(0)

                    if mask_mode == "spec":
                        self._face_mask = compute_binary_face_mask(
                            snapshot, MASK_LAYERS_CONFIG)
                    else:
                        self._face_mask = simple_threshold_mask(snapshot)

                    attn_maps_current.clear()   # reset → one-step snapshot

                # call the callback, if provided (now sees fresh mask)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                
                # NEW 30 JUL                    
                
                
                # # ───────────── build mask right AFTER the merge step ─────────────
                # # (not one iteration before).  Only needed when branched-attention
                # # is enabled – otherwise keep the hooks alive so we can keep
                # # collecting heat-maps for the whole diffusion trajectory.
                # if use_branched_attention and i == branched_attn_start_step:
                    
                    
                # ───────────── build mask right AFTER the merge step ─────────────
                # (not one iteration before). We *always* build the mask once we
                # have enough maps (`collect_hm`), but we only detach hooks and
                # patch the UNet when branched-attention is actually requested.
                if collect_hm and i == branched_attn_start_step:
                    
                    if not attn_maps_current:
                        warnings.warn("[BranchedAttn] no attention maps captured – mask disabled.")
                    else:

                        # Average all collected maps per layer.  Normalise each
                        # map to the *largest* H×H in its list to avoid
                        # “all input arrays must have the same shape”.
                        from .mask_utils import _resize_map

                        snapshot = {}
                        for ln, lst in attn_maps_current.items():
                                
                            # ── drop any malformed (e.g. 1-D) maps ───────────
                            lst2 = [m for m in lst if m.ndim == 2]
                            if not lst2:
                                continue
                            
                            # bring every map to a common resolution ― keep tile
                            # alignment with the helper used in attn_hm_NS_nosm7.py
                            max_H = max(m.shape[0] for m in lst2)

                            normed = []
                            for m in lst2:
                                if m.shape[0] != max_H:
                                    m = _resize_map(m, max_H)    # ← grid-aware
                                normed.append(m)
                            
                            
                            
                            snapshot[ln] = np.stack(normed, axis=0).mean(0)
                            
                        # self._face_mask = compute_binary_face_mask(snapshot, MASK_LAYERS_CONFIG)
                        
                        if mask_mode == "spec":          # weighted-union (default)
                            self._face_mask = compute_binary_face_mask(
                                snapshot, MASK_LAYERS_CONFIG)
                        else:                           # "simple" → single-layer top-k
                            self._face_mask = simple_threshold_mask(snapshot)
                    
                        # save the binary mask for visual inspection
                        _save_gray(
                            self._face_mask.astype(np.uint8) * 255,
                            f"{DEBUG_DIR}/mask_final.png",
                        )
                        
                        # ─── DEBUG ③: mask statistics
                        print(
                            f"[DEBUG] mask built ✓  size={self._face_mask.shape}  "
                            f"face_pixels={(self._face_mask>0).sum()}"
                        )


                    # # Remove hooks (restore original forwards)
                    # for name, module in self.unet.named_modules():
                    #     if name in orig_attn_forwards:
                    #         module.forward = orig_attn_forwards[name]
                    # attn_maps_current.clear()
                    
                    
                    
                    
                    # orig_attn_forwards.clear()
                    # # Monkey-patch UNet self-attention layers for branched mechanism
                    # if use_branched_attention:
                        
                    # -------------------------------------------------------------
                    # ❶ Build the mask exactly *once* (or whenever you like)
                    # -------------------------------------------------------------
                    if (use_branched_attention and
                        i == branched_attn_start_step):
                        # Average heads collected so far
                        snapshot = {ln: np.stack(m).mean(0)
                                    for ln, m in attn_maps_current.items()
                                    if len(m)}
        
                        if snapshot:                         # we have something!
                            if mask_mode == "simple":
                                mask_np = simple_threshold_mask(snapshot)
                            else:                            # "spec"
                                mask_np = compute_binary_face_mask(
                                            snapshot, MASK_LAYERS_CONFIG)
        
                            # save for optional callbacks / debugging -------------
                            self._face_mask = torch.from_numpy(mask_np) \
                                                .to(device=latents.device,
                                                    dtype=latents.dtype) \
                                                .unsqueeze(0)            # (1,H,W)
                            
                            # keep a plain H×W NumPy array – the forward hook will turn it into
                            # a tensor on-the-fly and cache the result, avoiding dtype mix-ups
                            self._face_mask = mask_np.astype(bool)        # ← 2-D numpy                    
                            
                        else:
                            warnings.warn("[BranchedAttn] no attention maps "
                                          "captured – mask disabled.")
        
                        # maps no longer needed – free memory
                        attn_maps_current.clear()
        
                    # -------------------------------------------------------------
                    # ❷ DO NOT clear the buffer at the *start* of every step.
                    #    Only wipe it *after* we are done using it (see above).
                    # -------------------------------------------------------------    
                        
                    
                    # ───────────── activate branched attention (optional) ────────
                    # if use_branched_attention:
                    #     # Remove hooks (restore original forwards) – we won't
                    #     # collect any more heat-maps once we fork the branches.
                    #     for name, module in self.unet.named_modules():
                    #         if name in orig_attn_forwards:
                    #             module.forward = orig_attn_forwards[name]
                    #     attn_maps_current.clear()
                    #     orig_attn_forwards.clear()

                    #     # Monkey-patch UNet self-attention layers for the
                    #     # face / background split
                    
                    # 🔸 **Do *NOT* restore the cross-attention forwards.**
                        #     Their tiny hooks must stay in place so we can
                        #     keep harvesting attention grids every
                        #     `mask_interval` steps and refine the mask.
                        #
                        #     We *only* monkey-patch the **self-attention**
                        #     modules ('.attn1') below; their original call is
                        #     stored in `module._orig_forward`, while the
                        #     cross-attention ('.attn2') layers – where the
                        #     heat-map hooks live – remain untouched.

                        # Monkey-patch UNet self-attention layers for the
                        # face / background split
                    
                        
                        from diffusers.models.attention_processor import Attention as CrossAttention
                        def custom_attn_forward(module, hidden_states, encoder_hidden_states=None, attention_mask=None):
                            # Use branched attention for self-attn (no encoder_hidden_states)
                            if encoder_hidden_states is None and hasattr(self, "_face_mask") and self._face_mask is not None:
                                # separate unconditional and conditional parts if CFG
                                B, L, C = hidden_states.shape
                                if self.do_classifier_free_guidance and B % 2 == 0:
                                    B_half = B // 2
                                    hidden_uncond = hidden_states[:B_half]
                                    hidden_cond = hidden_states[B_half:]
                                else:
                                    B_half = 0
                                    hidden_uncond = None
                                    hidden_cond = hidden_states
                                # Do standard self-attn for unconditional part (no face injection)
                                out_uncond = None
                                if hidden_uncond is not None:
                                    out_uncond = module._orig_forward(hidden_uncond, None, attention_mask)
                                # Prepare for conditional branch attention
                                hs = hidden_cond
                                B_cond = hs.shape[0]
                                # Project Q, K, V for current latent
                                q_proj = module.to_q(hs) if hasattr(module, "to_q") else module.q_proj(hs)  # (B_cond, L, inner_dim)
                                k_proj = module.to_k(hs) if hasattr(module, "to_k") else module.k_proj(hs)  # (B_cond, L, inner_dim)
                                v_proj = module.to_v(hs) if hasattr(module, "to_v") else module.v_proj(hs)  # (B_cond, L, inner_dim)
                                # Reshape for multi-head
                                Bc, Lc, Ci = q_proj.shape
                                h = module.heads
                                d = Ci // h
                                Q = q_proj.view(Bc, Lc, h, d).permute(0, 2, 1, 3)  # (B_cond, h, L, d)
                                Kc = k_proj.view(Bc, Lc, h, d).permute(0, 2, 1, 3)  # (B_cond, h, L, d)
                                Vc = v_proj.view(Bc, Lc, h, d).permute(0, 2, 1, 3)  # (B_cond, h, L, d)
                                # Determine face vs background query indices for this resolution
                                L_curr = Lc
                                H_curr = int(math.sqrt(L_curr))
                                # mask = torch.from_numpy(self._face_mask)  # shape (H_map, W_map)
                                # ── ① get / cache the mask as a (H,W) torch.bool on *this* device ──
                                if isinstance(self._face_mask, np.ndarray):
                                    mask_np   = self._face_mask
                                else:                           # just in case someone changes it later
                                    mask_np   = self._face_mask.squeeze().cpu().numpy()
                                mask = torch.from_numpy(mask_np).to(device=hidden_states.device)
                                H_map = mask.shape[0]
                                if H_map != H_curr:
                                    # Downsample or upsample mask to current resolution
                                    # Compute factor (assuming integer scale)
                                    if H_map % H_curr == 0:
                                        factor = H_map // H_curr
                                        # Downsample: group factor x factor blocks
                                        mask = mask.view(H_curr, factor, H_curr, factor).any(dim=(1, 3))
                                    elif H_curr % H_map == 0:
                                        factor = H_curr // H_map
                                        mask = mask.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)
                                    else:
                                        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0),
                                                            size=(H_curr, H_curr), mode="nearest")[0,0] > 0.5
                                mask_flat = mask.view(-1).to(device=hidden_states.device)
                                face_idx = mask_flat.nonzero(as_tuple=True)[0]  # indices of face region queries
                                bg_idx = (~mask_flat).nonzero(as_tuple=True)[0]  # indices of background queries
                                L_face = face_idx.shape[0]
                                L_bg = bg_idx.shape[0]
                                # Compute standard self-attention output for all queries (we will override face part)
                                # shape for bmm: flatten batch and heads
                                Q_flat = Q.reshape(Bc * h, Lc, d)
                                Kc_flat = Kc.reshape(Bc * h, Lc, d)
                                Vc_flat = Vc.reshape(Bc * h, Lc, d)
                                attn_scores = torch.baddbmm(
                                    torch.empty(Bc * h, Lc, Lc, device=Q_flat.device, dtype=Q_flat.dtype),
                                    Q_flat,
                                    Kc_flat.transpose(-1, -2),
                                    beta=0,
                                    alpha=module.scale,
                                )  # (B_cond*heads, L, L)
                                attn_probs = attn_scores.softmax(dim=-1)
                                context_all = torch.bmm(attn_probs, Vc_flat)  # (B_cond*heads, L, d)
                                context_all = context_all.view(Bc, h, Lc, d)
                                # Compute face-branch attention for face query positions
                                if L_face > 0:
                                    # Prepare reference image keys/values
                                    # Use CLIP patch features for reference face region if available, else fall back to id_embeds
                                    if face_embed_strategy.lower() == "heatmap":
                                        # Use CLIP vision patch features directly for reference
                                        # (Already processed via id_encoder in else branch, which uses CLIP global. 
                                        # For patch-level features, we reuse internal CLIP features)
                                        last_hidden = self.id_encoder.vision_model(id_pixel_values.to(device=device))[0]  # (1, n_patches+1, 1024)
                                        # Identify face region patches (approx): use same mask scaled to CLIP patch grid
                                        # CLIP ViT-L/14 yields 16x16 patches for 224x224 input
                                        clip_patches = last_hidden[:, 1:, :]  # exclude CLS
                                        n_patches = clip_patches.shape[1]
                                        H_patch = W_patch = int(math.sqrt(n_patches))
                                        # If face_mask at output resolution, roughly assume face occupies similar area in CLIP input
                                        face_mask_patch = F.interpolate(mask.float().view(1,1,H_curr,H_curr), size=(H_patch, W_patch), mode="nearest")
                                        face_mask_patch = face_mask_patch.view(-1) >= 0.5
                                        
                                        
                                        
                                        # ref_keys = clip_patches[0, face_mask_patch, :].to(device=hidden_states.device)  # shape (M, 1024)
                                        # if ref_keys.numel() == 0:
                                        #     ref_keys = clip_patches[0:1, 0:1, :].to(device=hidden_states.device)  # fallback to CLS if no face patch
                                        # # Project CLIP patch features to UNet latent dim via a linear
                                        # proj_dim = hs.shape[-1]
                                        # # k_proj_layer = nn.Linear(ref_keys.shape[-1], d).to(device=hidden_states.device)
                                        # # v_proj_layer = nn.Linear(ref_keys.shape[-1], d).to(device=hidden_states.device)
                                        
                                        # # ── ② *initialise these layers only once* – re-use afterwards ──
                                        # if not hasattr(module, "_ref_proj"):
                                        #     module._ref_proj = nn.Linear(ref_keys.shape[-1], 2 * d, bias=False)
                                        #     nn.init.xavier_uniform_(module._ref_proj.weight, gain=0.2)
                                        # kv_ref = module._ref_proj(ref_keys)           # (M, 2d)
                                        # K_ref, V_ref = kv_ref.chunk(2, dim=-1)        # each (M, d)
                                        
                                        # # make dtype match UNet attention tensors (bfloat16/fp16)
                                        # # K_ref = k_proj_layer(ref_keys).to(dtype=Q.dtype)  # (M, d)
                                        # # V_ref = v_proj_layer(ref_keys).to(dtype=Q.dtype)  # 
                                        
                                        # K_ref = K_ref.to(dtype=Q.dtype)
                                        # V_ref = V_ref.to(dtype=Q.dtype)
                                        
                                        ref_feats = clip_patches[0, face_mask_patch, :].to(device=hidden_states.device)  # (M, 1024)
                                        if ref_feats.numel() == 0:                     # safety fallback
                                            ref_feats = clip_patches[0:1, 0:1, :].to(device=hidden_states.device)

                                        # ── project *with the SAME matrices as the UNet* ──
                                        with torch.no_grad():
                                            K_ref = module.to_k(ref_feats)             # (M, inner_dim)
                                            V_ref = module.to_v(ref_feats)             # (M, inner_dim)

                                        # split heads → (h, M, d) and cast
                                        K_ref = K_ref.view(-1, h, d).transpose(0, 1).to(dtype=Q.dtype, device=Q.device)
                                        V_ref = V_ref.view(-1, h, d).transpose(0, 1).to(dtype=Q.dtype, device=Q.device)
                                        
                                        
                                    else:
                                        ref_vec = id_embeds.squeeze().to(hidden_states.device).float()
                                        if ref_vec.dim() == 1:                       # (512,) → (1, 512)
                                            ref_vec = ref_vec.unsqueeze(0)

                                        in_feats = ref_vec.shape[-1]                 # usually 512

                                        # project to the head dimension `d`
                                        k_proj_layer = torch.nn.Linear(
                                            in_feats, d, bias=False
                                        ).to(hidden_states.device)
                                        v_proj_layer = torch.nn.Linear(
                                            in_feats, d, bias=False
                                        ).to(hidden_states.device)

                                        # K_ref = k_proj_layer(ref_vec).to(dtype=Q.dtype)       # (1, d) – dtype aligned
                                        # V_ref = v_proj_layer(ref_vec).to(dtype=Q.dtype)       # (1, d) – dtype aligned
                                        
                                        # project once …
                                        K_ref = k_proj_layer(ref_vec).to(dtype=Q.dtype)       # (1, d)
                                        V_ref = v_proj_layer(ref_vec).to(dtype=Q.dtype)       # (1, d)
                                        # … then replicate for every head → (h, 1, d)
                                        K_ref = K_ref.repeat(h, 1, 1)
                                        V_ref = V_ref.repeat(h, 1, 1)    
                                        
                                    # Compute face-query attention
                                    Q_face = Q[:, :, face_idx, :]  # (B_cond, h, L_face, d)
                                    # Expand K_ref, V_ref to include head dimension
                                    # K_ref_h = K_ref.unsqueeze(0).expand(h, -1, -1)  # (h, M, d)
                                    # V_ref_h = V_ref.unsqueeze(0).expand(h, -1, -1)  # (h, M, d)
                                    
                                    
                                    # # Already per-head
                                    # K_ref_h = K_ref                                  # (h, M, d)
                                    # V_ref_h = V_ref                                  # (h, M, d)
                                    
                                    
                                    # ──────────────────────────────────────────────────────────────────
                                    #  (1)  reference tensors per head
                                    # ──────────────────────────────────────────────────────────────────
                                    K_ref_h = K_ref          # (h, M_ref, d)
                                    V_ref_h = V_ref          # (h, M_ref, d)
                                    
                                    # fresh – *local* – reference-length (don’t reuse the old `M`)
                                    M_ref = K_ref_h.shape[1]
                                    
                                    # -------  DEBUG  --------------------------------------------------
                                    # print once per step so we know what shapes we are feeding to
                                    # the broadcast / reshape logic.
                                    # print(
                                    #         f"[BRANCHED] step={i:03d}  Bc={Bc}  h={h}  d={d}  "
                                    #         f"M_ref={M_ref}  "
                                    #         f"K_ref_h={tuple(K_ref_h.shape)}  "
                                    #         f"Q_face_flat={tuple(Q_face.reshape(Bc * h, L_face, d).shape)}"
                                    # )
                                    
                                    
                                    print(
                                            f"[BRANCHED] step={i:03d}  Bc={Bc}  h={h}  d={d}  "
                                            f"M_ref={M_ref}  "
                                            f"K_ref_h={tuple(K_ref_h.shape)}"
                                    )
                                        # ------------------------------------------------------------------
                                    
                                    # ──────────────────────────────────────────────────────────────────
                                    #  (2)  make K/V reference batch-wise
                                    #       use **repeat** so the tensor owns real memory
                                    # ──────────────────────────────────────────────────────────────────
                                    K_ref_flat = (
                                            K_ref_h.unsqueeze(0)               # (1, h, M_ref, d)
                                                   .repeat(Bc, 1, 1, 1)        # (Bc, h, M_ref, d)
                                                   .view(Bc * h, M_ref, d)     # (Bc·h, M_ref, d)
                                    )
                                    
                                    V_ref_flat = (
                                            V_ref_h.unsqueeze(0)
                                                   .repeat(Bc, 1, 1, 1)
                                                   .view(Bc * h, M_ref, d)
                                    )
                                    # sanity check
                                    if K_ref_flat.numel() != Bc * h * M_ref * d:
                                        print("[ERR] broadcast produced unexpected element count!")
                                    
                                    
 
                                    
                                    # NOTE: repeat *allocates* real memory, so the element count matches the
                                    # upcoming reshape.  clone() would work too.
                                    # K_ref_flat = (
                                    #         K_ref_h.unsqueeze(0)                 # (1, h, M, d)
                                    #                .repeat(Bc, 1, 1, 1)          # (Bc, h, M, d)
                                    #               .reshape(Bc * h, -1, d)       # (Bc·h, M, d)
                                    # )
                                    
                                    # V_ref_flat = (
                                    #         V_ref_h.unsqueeze(0)
                                    #                .repeat(Bc, 1, 1, 1)
                                    #                .reshape(Bc * h, -1, d)
                                    # )
                                    
                                    
                                    # # Keep `M` explicit so the product of dimensions is always correct
                                    # M = K_ref_h.shape[1]                     # number of reference tokens
                                    # K_ref_flat = (
                                    #     K_ref_h.unsqueeze(0)                 # (1, h, M, d)
                                    #            .repeat(Bc, 1, 1, 1)          # (Bc, h, M, d)
                                    #            .reshape(Bc * h, M, d)        # (Bc·h, M, d)   ✓
                                    # )
                                
                                    # V_ref_flat = (
                                    #     V_ref_h.unsqueeze(0)
                                    #            .repeat(Bc, 1, 1, 1)
                                    #            .reshape(Bc * h, M, d)        # (Bc·h, M, d)   ✓
                                    # )
                                    
                                    # ── make the reference batch-wise ─────────────────────────────────
                                    M_ref = K_ref_h.shape[1]                    # number of reference tokens
                                    
                                    K_ref_flat = (
                                            K_ref_h.unsqueeze(0)                    # (1, h, M_ref, d)
                                                   .expand(Bc, -1, -1, -1)          # (Bc, h, M_ref, d)  – broadcast
                                                   .contiguous()                    # materialise memory
                                                   .view(Bc * h, M_ref, d)          # (Bc·h, M_ref, d)
                                        )
                                    
                                    V_ref_flat = (
                                            V_ref_h.unsqueeze(0)
                                                   .expand(Bc, -1, -1, -1)
                                                   .contiguous()
                                                   .view(Bc * h, M_ref, d)
                                        )
                                    
                                    
                                    # Compute dot-product attention for face queries
                                    # Reshape Q_face to (B_cond*h, L_face, d)
                                    Q_face_flat = Q_face.permute(0,2,1,3).reshape(Bc * L_face, h, d)  # Actually, easier to loop per head than flatten differently
                                    Q_face_flat = Q_face.reshape(Bc * h, L_face, d)  # (B_cond* h, L_face, d)
                                                                       
                                    attn_scores_face = torch.baddbmm(
                                        torch.empty(Bc * h, L_face, K_ref_flat.shape[1], device=Q_face_flat.device, dtype=Q_face_flat.dtype),
                                        Q_face_flat,
                                        K_ref_flat.transpose(-1, -2),
                                        beta=0,
                                        alpha=module.scale,
                                    )
                                    attn_probs_face = attn_scores_face.softmax(dim=-1)
                                    context_face = torch.bmm(attn_probs_face, V_ref_flat)  # (B_cond*h, L_face, d)
                                    context_face = context_face.view(Bc, h, L_face, d)
                                    if i >= branched_attn_start_step:
                                        print("mean|std context_face:", context_face.mean().item(),
                                                                        context_face.std().item())
                                        
                                    # Replace corresponding outputs in context_all
                                    context_all[:, :, face_idx, :] = context_face
                                # Merge heads back and project out
                                context_flat = context_all.permute(0, 2, 1, 3).reshape(Bc, Lc, Ci)
                                hidden_cond_out = module.to_out[0](context_flat)
                                hidden_cond_out = module.to_out[1](hidden_cond_out)
                                # Combine with unconditional (if exists)
                                if out_uncond is not None:
                                    hidden_out = torch.cat([out_uncond, hidden_cond_out], dim=0)
                                else:
                                    hidden_out = hidden_cond_out
                                # Add residual connection if applicable
                                if getattr(module, "residual_connection", False):
                                    hidden_out = hidden_out + hidden_states
                                return hidden_out
                            else:
                                # For cross-attention or if branched mask not set, use original forward
                                return module._orig_forward(hidden_states, encoder_hidden_states, attention_mask)
                            
                            
                            
                        # end custom_attn_forward
                        # Patch all Attention modules that are self-attn
                        for name, module in self.unet.named_modules():
                            if isinstance(module, CrossAttention):
                                # Only patch if not cross-only (module.is_cross_attention == False)
                                
                                # Patch self-attention only – leave cross-attn
                                # (where hooks sit) unchanged.
                                if not getattr(module, "is_cross_attention", False):
                                    module._orig_forward = module.forward  # backup original
                                    module.forward = types.MethodType(custom_attn_forward, module)

                if XLA_AVAILABLE:
                    xm.mark_step()
                    


        # ── AFTER denoising: build coloured montage strips per layer ──────
        if hasattr(self, "_heatmaps"):
            header_h    = 30
            final_clean = self._heatmaps[self._hm_layers[0]][-1] \
                          if self._heatmaps[self._hm_layers[0]] else None

            for ln, frames in self._heatmaps.items():
                if not frames:
                    continue
                cols = frames + ([final_clean] if final_clean else [])

                img_w, img_h = cols[0].width, cols[0].height
                strip = Image.new("RGB",
                                  (img_w * len(cols), img_h + header_h),
                                  "black")
                draw  = ImageDraw.Draw(strip)
                font  = ImageFont.load_default()

                for idx, frm in enumerate(cols):
                    x = idx * img_w
                    strip.paste(frm, (x, header_h))
                    label = ("Final" if idx >= len(self._step_tags)
                             else f"S{self._step_tags[idx]}")
                    tw, th = draw.textbbox((0,0), label, font=font)[2:]
                    draw.text((x + (img_w-tw)//2, (header_h-th)//2),
                              label, font=font, fill="white")

                safe_ln = ln.replace("/", "_").replace(".", "_")
                out_jpg = Path(DEBUG_DIR) / f"{safe_ln}_attn_hm.jpg"
                strip.save(out_jpg, quality=95)
                print(f"[DEBUG] heat-map strip saved → {out_jpg}")

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

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)