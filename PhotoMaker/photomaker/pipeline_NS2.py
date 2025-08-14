# photomaker/pipeline_NS2.py

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
from PIL import Image  # ← needed for mask I/O

# # ─── tiny helper for export_mask ──────────────────────────────────────────
# def _save_gray(arr: np.ndarray, path: str) -> None:
#     """Save a H×W uint8 array as an 8-bit grayscale PNG."""
#     Image.fromarray(arr, mode="L").save(path)

# ─── tiny helper for export_mask ──────────────────────────────────────────
def _save_gray(
    arr:  np.ndarray,
    path: str,
    size: tuple[int, int] | None = None,   # (W,H)
) -> None:
    """
    Save a H×W uint8 array as an 8-bit grayscale PNG.  
    If *size* is given the array is **nearest-neighbour**-resized first
    so the binary mask keeps hard edges.
    """
    img = Image.fromarray(arr, mode="L")
    if size and img.size != size:
        img = img.resize(size, Image.NEAREST)
    img.save(path)


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

FULL_DEBUG  = False
# FULL_DEBUG  = True

# def _save_gray(arr: np.ndarray, fp: str):
#     """
#     Save a 2-D numpy array as an 8-bit grayscale PNG.
#     """
#     if arr.ndim != 2:
#         return
#     arr = arr.astype(np.float32)
#     arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-8)).clip(0, 255).astype(np.uint8)
#     Image.fromarray(arr, mode="L").save(fp)




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


    # ──────────────────────────────────────────────────────────────
    #  Utility: robust, normalised face-region latents from 1st ID-image
    # ──────────────────────────────────────────────────────────────
    def _encode_face_latents(
        self,
        id_pixel_values: torch.Tensor,
        target_hw: Tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return normalised VAE latents for the **first** ID image
        and resize them so that `latents.shape[-2:] == target_hw`.
        Always encodes in fp32 for numerical stability, then casts
        to the requested *dtype* on the caller’s device.
        """
        # ref_img = id_pixel_values[0, 0].unsqueeze(0).to(dtype=torch.float32)

        # ------------------------------------------------------------------
        # Encode with **the same dtype / device** as the VAE’s weights so
        # Conv2d does not complain (fp32 / fp16 / bf16 – whatever the model
        # is in).  We still up-cast to float32 *afterwards* for the normal-
        # isation & interpolation steps so numerical behaviour stays stable.
        # ------------------------------------------------------------------
        vae_weight   = next(self.vae.parameters())
        vae_dtype    = vae_weight.dtype
        vae_device   = vae_weight.device

        ref_img = (
            id_pixel_values[0, 0]          # (3,H,W)
            .unsqueeze(0)                  # (1,3,H,W)
            .to(device=vae_device, dtype=vae_dtype)
        )

        with torch.no_grad():
            z = self.vae.encode(ref_img).latent_dist.mode() * self.vae.config.scaling_factor

        z = F.interpolate(z.float(), size=target_hw, mode="bilinear")
        # z = z.clamp_(-1.0, 1.0)
        # z = z.clamp_(-3.0, 3.0)   # or even remove clamp altogether
        z = z.clamp_(-5.0, 5.0)   # or even remove clamp altogether
        z = (z - z.mean()) / z.std().clamp(min=1e-4)
        return z.to(device=self.device, dtype=dtype).detach()


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
        num_inference_steps: int = 50, # 100, # 50,
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
        heatmap_mode: str = "identity", # "token",  # "identity", # "token", # "identity",             # "identity" | "token"
        focus_token: str = "face",
        mask_mode: str = "spec", # "simple", # "spec",                    # "spec" | "simple"
        branched_attn_start_step: int = 10,
        # ────── DEBUG MASK STRIP ───────────────────────────────
        debug_save_masks: bool = False,
        mask_save_dir: str = "hm_debug",
        mask_interval: int = 5,
        # ─── debug: capture standalone branch images ───────────────
        debug_save_face_branch: bool = True,
        debug_save_bg_branch:   bool = True,   # NEW: background previews
        face_branch_interval: int = 10,        
        # ---------------- mask I/O ----------------
        export_mask: bool = True, # False, # True, # False,              # save final face-mask to disk
        # import_mask: Optional[str] = "hm_debug/mask_export.png",      # path to an existing mask
        # import_mask: Optional[str] = "hm_debug/mask_export2.png",      # path to an existing mask
        import_mask: Optional[str] = None,      # path to an existing mask
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
        
        face_embed_strategy = "heatmap" # TEMP WTF!!!
        # face_embed_strategy = "faceanalysis" # TEMP WTF!!!

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
        


        # ------------------------------------------------------------------ #
        # 6. Prepare the input-ID images  (builds `id_pixel_values`)
        # ------------------------------------------------------------------ #

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

        
        # ------------------------------------------------------------------ #
        # 6.a  ⟶  Robust reference-latents for branched-attention
        # ------------------------------------------------------------------ #
        # if use_branched_attention:
        #     # vae_dtype = next(self.vae.parameters()).dtype        # fp32 / fp16 / bf16
        #     # # ref_img_for_lat = id_pixel_values[0:1].to(dtype=vae_dtype)

        #     # # id_pixel_values has shape (B, N, 3, H, W); we need (1, 3, H, W)
        #     # ref_img_for_lat = id_pixel_values[0, 0].unsqueeze(0).to(dtype=vae_dtype)

        #     # always go through fp32 for a stable VAE encode
        #     ref_img_for_lat = id_pixel_values[0, 0].unsqueeze(0).to(dtype=torch.float32)

        #     def _encode(img):
        #         with torch.no_grad():
        #             return self.vae.encode(img).latent_dist.mode() * self.vae.config.scaling_factor

        #     try:
        #         _ref_latents_all = _encode(ref_img_for_lat)      # first try (preferred dtype)

        #         # bring stats to a predictable range
        #         _ref_latents_all = _ref_latents_all.clamp(-1.0, 1.0)
        #         _ref_latents_all = (
        #             _ref_latents_all - _ref_latents_all.mean()
        #         ) / _ref_latents_all.std().clamp(min=1e-4)

        #         self._ref_latents_all = _ref_latents_all.to(device=device, dtype=latents.dtype).detach()
            
        #     except Exception as e1:
        #         try:                                             # fallback: fp32
        #             _ref_latents_all = _encode(ref_img_for_lat.float())
        #             print("[WARN] VAE encode retried in fp32")
        #         except Exception as e2:
        #             _ref_latents_all = None
        #             print(f"[WARN] VAE encode failed twice: {e1} / {e2} "
        #                   "→ will fall back to noisy latents")

        #     if _ref_latents_all is not None:
        #         print(f"[DBG] VAE ref-latents ready  shape={_ref_latents_all.shape}  "
        #               f"dtype={_ref_latents_all.dtype}")
        #         # ── keep a copy on the pipeline so the inner loop sees it
        #         self._ref_latents_all = _ref_latents_all

        if use_branched_attention:
            # _ref_latents_all = self._encode_face_latents(
            #     id_pixel_values,
            #     target_hw=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            #     dtype=latents.dtype,
            # )

            dtype_lat = next(self.unet.parameters()).dtype  # safe early dtype
            _ref_latents_all = self._encode_face_latents(
                id_pixel_values,
                target_hw=(height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=dtype_lat,
            )

            self._ref_latents_all = _ref_latents_all          # cache for inner loop
            print(f"[DBG] VAE ref-latents ready  shape={_ref_latents_all.shape}  "
                  f"dtype={_ref_latents_all.dtype}")
            
            
        # # NEW 30 JUL
        # # Decide identity embedding usage based on strategy
        # if use_branched_attention and face_embed_strategy.lower() == "heatmap":
        #     # Skip using external face recognition embedding (ArcFace); rely on image features only
        #     id_embeds = None
        # if id_embeds is not None:
        #     id_embeds = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
        #     prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
        #     # ─── DEBUG ────────────────────────────────────────────────────
        #     # raw, *unnormalised* embedding – just for inspection
        #     print(f"[ID-DEBUG pre-norm] ArcFace raw  shape={id_embeds.shape} "
        #           f"dtype={id_embeds.dtype}  norm={id_embeds.norm():.4f}")
        # else:
        #     prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, None) # check WTF
        #     print("[ID-DEBUG] ArcFace path disabled (face_embed_strategy=='heatmap')")
        
        # ──────────────────────────────────────────────────────────────
        # 7.  Feed identity branch into PhotoMaker’s ID-encoder
        #     • ArcFace path  → real embedding
        #     • Heat-map path → zero placeholder (ID-encoder still needs a tensor)
        # ──────────────────────────────────────────────────────────────
        # if face_embed_strategy.lower() == "heatmap":
        if use_branched_attention and face_embed_strategy.lower() == "heatmap":
            emb_dim = 512                                           # InsightFace length
            id_embeds_t = torch.zeros((1, num_id_images, emb_dim),  # (B,N,512)
                                    device=device, dtype=dtype)
            print("[ID-DEBUG] Using zero-placeholder id_embeds for heat-map strategy")
        elif id_embeds is not None:
            id_embeds_t = id_embeds.unsqueeze(0).to(device=device, dtype=dtype)
            print(f"[ID-DEBUG pre-norm] ArcFace raw  shape={id_embeds_t.shape} "
                f"dtype={id_embeds_t.dtype}  ‖⋅‖={id_embeds_t.norm():.4f}")
        else:
            raise ValueError("`id_embeds` must be supplied when "
                            "face_embed_strategy is not 'heatmap'")

        # Always call the four-argument forward so the encoder API is satisfied
        prompt_embeds = self.id_encoder(
            id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds_t
        )

        # ─── keep a unit-norm copy for the fallback KV path ───────────
        self._id_embed_vec = None
        if face_embed_strategy.lower() != "heatmap":
            self._id_embed_vec = F.normalize(
                id_embeds_t.squeeze(0).float(), p=2, dim=-1
            ).detach().to(self.device)

        
        # ─── store a copy of the identity embedding for branched-attn ─────
        # self._id_embed_vec = None
        # if id_embeds is not None:
        #     # (512,)  or  (1,512)
        #     self._id_embed_vec = id_embeds.squeeze().detach().to(self.device)  # (512,)
        
        # keep a unit-norm copy for the fallback path used by patched
        # self-attention when no _kv_override is available
        self._id_embed_vec = None
        if id_embeds is not None:
            self._id_embed_vec = F.normalize(
                id_embeds.squeeze(0).float(), p=2, dim=-1
            ).detach().to(self.device)                                  # (512,)
        
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
        if use_branched_attention:
            _branched_ready  = False                  # init after mask exists
            _ref_latents_clean     = None                   # face-region K/V
            _ref_latents_all = None                   # full reference latents

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


        # # ─── Import a pre-computed mask, if requested ─────────────────────
        # if import_mask:
        #     if not os.path.isfile(import_mask):
        #         raise FileNotFoundError(f"import_mask file '{import_mask}' not found")
        #     _imp = Image.open(import_mask).convert("L")
        #     self._face_mask = (np.array(_imp) > 127).astype(np.uint8)
        #     print(f"[INFO] imported face mask from {import_mask}  "
        #           f"shape={self._face_mask.shape}")
        #     # Skip internal heat-map collection / mask building
        #     collect_hm = False
            
        # ─── (1) optional MERGE mask – used **only** for bg/face blending ──
        self._merge_mask = None
        if import_mask:
            if not os.path.isfile(import_mask):
                raise FileNotFoundError(import_mask)
            # m = Image.open(import_mask).convert("L")
            # # self._merge_mask = (np.array(m) > 127).astype(np.uint8)  # H×W, {0,1}
            # # print(f"[INFO] imported merge-mask  {import_mask}  {self._merge_mask.shape}")
            # # # also use it as the face-region mask for branched K/V
            # # self._face_mask = self._merge_mask
            # # self._freeze_mask = True # keep unchanged during the run

            # self._merge_mask = (np.array(m) > 127).astype(np.uint8)  # H×W, {0,1}


            m = Image.open(import_mask).convert("L")
            # ensure mask matches latent resolution before binarising
            lat_hw = (height // self.vae_scale_factor,
                      width  // self.vae_scale_factor)          # (H,W)
            if m.size != (lat_hw[1], lat_hw[0]):                 # PIL uses (W,H)
                m = m.resize((lat_hw[1], lat_hw[0]), Image.NEAREST)
            self._merge_mask = (np.array(m) > 127).astype(np.uint8)  # H×W, {0,1}
            

            # ─── make the dispatcher see the mask ──────────────────────────────
            # branched_now checks for `self._face_mask`, so point it to the same
            # binary map we just imported.
            self._face_mask = self._merge_mask
  
            # # Invert the mask: in the imported mask, white=face, black=background
            # self._merge_mask = (np.array(m) <= 127).astype(np.uint8)  # H×W, {0,1} where 1=face
  
            print(f"[INFO] imported merge-mask  {import_mask}  {self._merge_mask.shape}")
            # only background/face *blending* uses this mask;
            # the branched-attention face-mask will still be built
            # from heat-maps / InsightFace.
            self._freeze_mask = True            # do NOT over-write merge-mask
            
            # We already have a mask → no need to harvest attention maps.
            # Turn the collector off so “no attention maps captured” never fires.
            collect_hm = False


            print(f"[DEBUG] Mask stats - min: {self._merge_mask.min()}, max: {self._merge_mask.max()}")
            print(f"[DEBUG] Face pixels (>0): {(self._merge_mask > 0).sum()}")
            print(f"[DEBUG] Face ratio: {self._merge_mask.mean()}")

            # Save the mask for visual inspection:
            Image.fromarray((self._merge_mask * 255).astype(np.uint8)).save("debug_loaded_mask.png")


            # ----------------------------------------------------------------
            # Branched self-attention needs a *face* mask (1 = face).  Use the
            # inverse of the imported background mask and freeze it for the run.
            # ----------------------------------------------------------------
            # self._face_mask_static = 1 - self._merge_mask                  # numpy (H,W)

            self._face_mask_static = self._merge_mask
            print(f"[INFO] face-mask fixed from imported PNG – "
                  f"face-pixels = {self._face_mask_static.sum()}  "
                  f"ratio = {self._face_mask_static.mean():.3f}")
        
            # _mask_4ch will be created lazily the first time we know the UNet grid

        else:
            # No external mask supplied – let the internal mask evolve.
            self._freeze_mask = False # allow mask to change during the run
            print("[INFO] no merge-mask imported; will build one from heat-maps")
            
        # # ------------------------------------------------------------------
        # #  One-shot caches for a *fixed* face-mask (2-D) and its 4-channel
        # #  broadcast version that is used for blending.
        # # ------------------------------------------------------------------
        # self._face_mask_static: Optional[np.ndarray] = None
        # self._mask_4ch:        Optional[torch.Tensor] = None
        

        # ------------------------------------------------------------------
        #  One-shot caches for a *fixed* face-mask (2-D) and its 4-channel
        #  broadcast clone used for masking latents.
        #  KEEP any values already set by `import_mask` above.
        # ------------------------------------------------------------------
        if not hasattr(self, "_face_mask_static"):
            self._face_mask_static = None
        if not hasattr(self, "_mask_4ch"):
            self._mask_4ch = None


        # # heat-map collection not needed when user supplied merge-mask
        # if self._merge_mask is not None:
        #     collect_hm = False
        
        # Keep `collect_hm` ON so we can still harvest attention maps and
        # hit the debug-print / PNG-saving code-paths even when the user has
        # provided a merge-mask.  We only skip the *spec-mask construction*
        # later on.
        # build_spec_mask = False if self._merge_mask is not None else True
        # build_spec_mask = True                   # always build face-mask
        build_spec_mask = self._merge_mask is None  # still build face-mask


        # ─────────────────────────────────────────────────────────────────
        #  Make sure self-attention is patched for branched mode
        #  (needed when we load a mask instead of building one)
        # ─────────────────────────────────────────────────────────────────
        from diffusers.models.attention_processor import Attention as _CrossAttn
        if use_branched_attention:

            def _patch_self_attn(pipeline_self):
                import math, types, torch.nn.functional as F

                def custom_attn_forward(mod, hidden_states,
                                         encoder_hidden_states=None,
                                         attention_mask=None):
                    

                    # keep default route for cross-attention or if no override
                    if encoder_hidden_states is not None or \
                       getattr(pipeline_self, "_kv_override", None) is None:
                       return mod._orig_forward(hidden_states,
                                                  encoder_hidden_states,
                                                  attention_mask)


                    # ------------------------------------------------------
                    #  ❶ reference latents (preferred)  or  ❷ id-vector
                    # ------------------------------------------------------
                    C_tgt = hidden_states.shape[-1]

                    if pipeline_self._kv_override is not None:          # ← preferred
                        ref = pipeline_self._kv_override                # (B,C,H,W)
                        if ref.dim() == 4:
                            ref = ref.permute(0, 2, 3, 1)              # (B,H,W,C)
                            ref = ref.reshape(1, -1, ref.shape[3])     # (1,L,Csrc)
                        C_src = ref.shape[-1]
                        if C_src < C_tgt:
                            ref = F.pad(ref, (0, C_tgt - C_src))
                        elif C_src > C_tgt:
                            ref = ref[..., :C_tgt]
                        if not getattr(pipeline_self, "_dbg_attn_once", False):
                            print(f"[DBG] {mod.__class__.__name__} uses REF latents  "
                                  f"L={ref.shape[1]}  C={C_tgt}")
                            pipeline_self._dbg_attn_once = True
                    elif pipeline_self._id_embed_vec is not None:
                        ref_vec = pipeline_self._id_embed_vec.to(hidden_states.dtype)
                        if ref_vec.numel() < C_tgt:
                            repeat = (C_tgt + ref_vec.numel() - 1) // ref_vec.numel()
                            ref_vec = ref_vec.repeat(repeat)[:C_tgt]
                        else:
                            ref_vec = ref_vec[:C_tgt]
                        ref = ref_vec.unsqueeze(0).unsqueeze(0)         # (1,1,C_tgt)
                        if not hasattr(mod, "_dbg_id"):
                            print(f"[DBG] {mod.__class__.__name__} uses ID vector  C={C_tgt}")
                            mod._dbg_id = True
                    else:
                        raise RuntimeError("No reference source for branched attention")

                    
                    # ----------------------------------------------------
                    # 1) project current latent → original K/V (k_orig/v_orig)
                    # 2) later replace conditional half with reference K/V
                    # ----------------------------------------------------
                    q       = (mod.to_q if hasattr(mod,"to_q") else mod.q_proj)(hidden_states)
                    k_orig  = (mod.to_k if hasattr(mod,"to_k") else mod.k_proj)(hidden_states)
                    v_orig  = (mod.to_v if hasattr(mod,"to_v") else mod.v_proj)(hidden_states)
                    
                
                    

                    # ---------- CFG-aware K/V replacement -------------------------------
                    # expected embedding size for this attention layer
                    K_IN = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj).weight.shape[1]

                    if pipeline_self.do_classifier_free_guidance and k_orig.shape[0] % 2 == 0:
                        B_half = k_orig.shape[0] // 2
            
                        # conditional half gets K/V projected from reference — but we first
                        # reshape the 4-ch latent grid to (B_half, seq_len, embed_dim) by
                        # *borrowing the embed shape* from the existing conditional slice.
                        _, L_cond, C_embed = k_orig[B_half:].shape            # (?, L, 640/1280)
                        
                        ref_flat = (
                            pipeline_self._kv_override.permute(0, 2, 3, 1)     # (B, H, W, 4)
                            .reshape(B_half, L_cond, -1)
                        )
                        # pad / clip so last-dim == K_IN
                        if ref_flat.shape[-1] < K_IN:
                            ref_flat = F.pad(ref_flat, (0, K_IN - ref_flat.shape[-1]))
                        elif ref_flat.shape[-1] > K_IN:
                            ref_flat = ref_flat[..., :K_IN]
                        
                        
                        
            
                        k_cond = (mod.to_k if hasattr(mod, "to_k") else mod.k_proj)(ref_flat)
                        v_cond = (mod.to_v if hasattr(mod, "to_v") else mod.v_proj)(ref_flat)
            
                        k = torch.cat([k_orig[:B_half], k_cond], 0)
                        v = torch.cat([v_orig[:B_half], v_cond], 0)

                    else:
                        # no CFG ⇒ project reference for the whole batch
                        B_lat = k_orig.shape[0]
                        L_lat = k_orig.shape[1]

                        
                        # B_lat, L_lat = k_orig.shape[:2]
                        ref_flat = (
                            pipeline_self._kv_override.permute(0, 2, 3, 1)
                            .reshape(B_lat, L_lat, -1)
                        )
                        if ref_flat.shape[-1] < K_IN:
                            ref_flat = F.pad(ref_flat, (0, K_IN - ref_flat.shape[-1]))
                        elif ref_flat.shape[-1] > K_IN:
                            ref_flat = ref_flat[..., :K_IN]

                        k = (mod.to_k if hasattr(mod,"to_k") else mod.k_proj)(ref_flat)
                        v = (mod.to_v if hasattr(mod,"to_v") else mod.v_proj)(ref_flat)
                                                        
                                            
                    B, L, C = hidden_states.shape
                    h   = mod.heads
                    d   = C // h
                    scl = 1 / math.sqrt(d)


                    q = q.view(B, L, h, d).permute(0, 2, 1, 3)          # B  h L  d

                    Bk, Lk, _ = k.shape                                 # actual batch for K/V
                    k = k.view(Bk, Lk, h, d).permute(0, 2, 1, 3)        # Bk h Lk d
                    v = v.view(Bk, Lk, h, d).permute(0, 2, 1, 3)        # Bk h Lk d

                    attn = (q @ k.transpose(-1, -2)) * scl                    
                    attn = attn.softmax(dim=-1)
                    
                    if FULL_DEBUG:
                        # ─── DEBUG (once per layer) ─────────────────────────────────────
                        if not hasattr(mod, "_once"):
                            layer_name = getattr(mod, "_orig_name", "self_attn")
                            print(f"[{layer_name}] mean-P = {attn.mean().item():.4f}")
                            mod._once = True
                        # ────────────────────────────────────────────────────────────────


                    out  = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)

                    out  = mod.to_out[0](out)
                    out  = mod.to_out[1](out)

                    # # one-time per layer debug
                    # if not hasattr(mod, "_dbg_once"):
                    #     print(f"[DBG] {mod.__class__.__name__}  "
                    #           f"tok_dim={C}  ref_dim={ref.shape[-1]}  "
                    #           f"heads={h}")
                    #     mod._dbg_once = True


                    return out

                patched = 0
                for n, m in pipeline_self.unet.named_modules():
                    if isinstance(m, _CrossAttn) and \
                       not getattr(m, "is_cross_attention", False) and \
                       not hasattr(m, "_orig_forward"):
                        m._orig_forward = m.forward
                        m.forward = types.MethodType(custom_attn_forward, m)
                        patched += 1
                print(f"[DEBUG] self-attention patched count = {patched}")

            _patch_self_attn(self)


        branched_attn_start_step = int(branched_attn_start_step)

        # collect maps across several steps so that we have something to
        # aggregate when we reach `branched_attn_start_step`
        attn_maps_current: Dict[str, List[np.ndarray]] = {}
        orig_attn_forwards: Dict[str, Callable] = {}
        
        # ------------------------------------------------------------------
        #  Prepare reference K / V **once** here so every Attention module
        #  can simply *reuse* them during the denoising loop.
        # ------------------------------------------------------------------
        self._ref_feats = None           # set below
        # if use_branched_attention:
        #     if face_embed_strategy == "heatmap":
        #     #     with torch.no_grad():
        #     #         # ViT-L/14 → (1, 257, 1024); drop CLS
        #     #         self._ref_feats = (
        #     #             self.id_encoder.vision_model(id_pixel_values.to(device))[0][:, 1:, :]
        #     #             .squeeze(0)                     # (n_patches, 1024)
        #     #             .detach()
        #     #         )
                
        #         # Replace above CLIP embeddings extraction by using VAE latents directly:
        #         with torch.no_grad():
        #             vae_dtype = next(self.vae.parameters()).dtype
        #             # ref_img = id_pixel_values[0, 0].unsqueeze(0).to(dtype=vae_dtype)
        #             ref_img = id_pixel_values[0, 0].unsqueeze(0).to(dtype=torch.float32)
        #             _ref_latents_all = self.vae.encode(ref_img).latent_dist.mode() * self.vae.config.scaling_factor
                    

        #             _ref_latents_all = F.interpolate(
        #                 _ref_latents_all.float(), size=latents.shape[-2:], mode="bilinear"
        #             ).clamp(-1.0, 1.0)
        #             _ref_latents_all = (_ref_latents_all - _ref_latents_all.mean()) / (_ref_latents_all.std().clamp(min=1e-4))

        #             self._ref_latents_all = _ref_latents_all.to(device=device, dtype=latents.dtype).detach()

        #             print("[Debug] Fixed ref_latents_all stats after clamp+norm:",
        #               self._ref_latents_all.mean().item(),
        #               self._ref_latents_all.std().item(),
        #               self._ref_latents_all.min().item(),
        #               self._ref_latents_all.max().item())
                            
        #             # self._ref_latents_all = F.interpolate(
        #             #     _ref_latents_all, size=latents.shape[-2:], mode="bilinear"
        #             # ).to(device=device, dtype=latents.dtype)
        #             self._ref_feats = self._ref_latents_all.detach()

        #         print(f"[BranchedAttn] CLIP patch bank {self._ref_feats.shape}")
            

            
        #     else:  # 'faceanalysis'
        #         # Instead of a single ArcFace vector, pull out the per-patch CLIP features
        #         with torch.no_grad():
        #             # id_pixel_values: (B, N, 3, H, W) → take the *first* ID image
        #             clip_img = id_pixel_values[:, 0].to(device)         # (B, 3, H, W)
        #             clip_out = self.id_encoder.vision_model(clip_img)[0]
        #             # clip_out = self.id_encoder.vision_model(id_pixel_values.to(device))[0]
        #             # drop the [CLS] token, keep only patch features
        #             patches = clip_out[:, 1:, :].squeeze(0)  # (n_patches, feature_dim)
        #         self._ref_feats = patches.detach()
        #         print(f"[BranchedAttn] CLIP patch features for face {self._ref_feats.shape}")

        if use_branched_attention:
            # one common, normalised latent bank for every strategy
            # self._ref_latents_all = getattr(self, "_ref_latents_all", None) or self._encode_face_latents(
            #     id_pixel_values,
            #     target_hw=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            #     # dtype=latents.dtype,
            #     dtype=dtype_lat,  # use UNet dtype for consistency
            # )

            # --------------------------------------------------------------
            # Build the reference latents **once**; on later iterations just
            # re-use the cached tensor.  Avoid “tensor in boolean context”
            # by checking against None explicitly.
            # --------------------------------------------------------------
            if getattr(self, "_ref_latents_all", None) is None:
                self._ref_latents_all = self._encode_face_latents(
                    id_pixel_values[0, 0],               # RGB (3,H,W)
                    latents.shape[-2:],                  # target (H,W)
                    latents.dtype,                       # match UNet
                )

            if face_embed_strategy == "heatmap":
                # use latent tiles directly as reference K/V
                self._ref_feats = self._ref_latents_all.detach()
                print(f"[BranchedAttn] latent patch bank {self._ref_feats.shape}")
            else:   # 'faceanalysis'
                with torch.no_grad():
                    clip_img = id_pixel_values[:, 0].to(device)
                    clip_out = self.id_encoder.vision_model(clip_img)[0][:, 1:, :]  # drop CLS
                self._ref_feats = clip_out.squeeze(0).detach()
                print(f"[BranchedAttn] CLIP patch features {self._ref_feats.shape}")


        # ------------------------------------------------------------------
        #  HOOKS for attention-map harvesting (unchanged below)
        # ------------------------------------------------------------------
        

        if collect_hm:            # allow --save_heatmaps to capture maps
            # make sure we run *regular* PyTorch attention
            if hasattr(self.unet, "attn_processors"):          # diffusers ≥ 0.25
                self.unet.set_attn_processor(dict(self.unet.attn_processors))
            else:                                              # legacy (< 0.25)
                self.unet.set_attn_processor(self.unet.attn_processor)
            
            from diffusers.models.attention_processor import Attention as CrossAttention
            wanted_layers = {spec["name"] for spec in MASK_LAYERS_CONFIG}


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
                    
                    

                # show whether any attention hooks are still alive
                if FULL_DEBUG:
                    if "orig_attn_forwards" in locals():                    
                        _hook_alive = any(
                            hasattr(m, "_orig_forward") for m in self.unet.modules()
                        )
                        print(f"[DBG] custom self-attention active? {_hook_alive}")

                
                    # NEW ─── how many maps do we have for every layer right now?
                    if collect_hm:
                        sizes = {k: len(v) for k, v in attn_maps_current.items()}
                        print(f"[DBG] attn_maps_current sizes: {sizes}")

                
                # ───────────────────────────────────────────────────────────
                #  ❶ Drop the maps gathered **before** the ID-tokens are
                #     merged in – they are pure noise.
                # ───────────────────────────────────────────────────────────
                if collect_hm and i == start_merge_step - 1:
                    attn_maps_current.clear()            # flush noise

                # ───────────────────────────────────────────────────────────
                #  ❷ Save a visual overlay only after the merge step
                # ───────────────────────────────────────────────────────────
                
                # print(f"[WTF_DEBUG] step {i:03d}  ")
                # print(f"[WTF_DEBUG]  collect_hm={collect_hm}  "
                #         f"i={i}  start_merge_step={start_merge_step}  "
                #         f"mask_interval={mask_interval}  "
                #         f"attn_maps_current={len(attn_maps_current)}")
                
                if (
                    collect_hm
                    and i >= start_merge_step            # NEW guard
                    and (i % mask_interval == 0 or i == len(timesteps) - 1)
                    and attn_maps_current                # anything captured yet?
                ):
                    print(f"[HM] step={i:3d}   layers with maps={len(attn_maps_current)}")
    
                    if FULL_DEBUG:
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
                    if FULL_DEBUG:
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


                # ----------------------------------------------------------
                #  Branched-attention dispatcher
                # ----------------------------------------------------------
                branched_now = (
                    use_branched_attention
                    and i >= branched_attn_start_step
                    and hasattr(self, "_face_mask")
                    and self._face_mask is not None
                )

                if branched_now:
                    # print(f"[WTF_DEBUG] branched attention active at step {i:03d} ")

                    if not _branched_ready:
                        
                        # ─── ensure we *have* reference latents ───────────
                        # if _ref_latents_all is None and hasattr(self, "_ref_latents_all"):
                        #     _ref_latents_all = self._ref_latents_all        # retrieve persisted copy
                        
                        _ref_latents_all = self._ref_latents_all   # always available now

                        print("[Debug] ref_latents_all stats:", _ref_latents_all.mean().item(), _ref_latents_all.std().item())


                        if _ref_latents_all is None:
                            print("[WARN] VAE latents missing → using noisy copy")
                            _ref_latents_all = latents.detach().clone()


                        # keep a copy of the raw reference RGB (needed for the
                        # optional debug crop a few lines below)
                        ref_img_for_lat = id_pixel_values[0, 0]  # (3,H,W)

                        # align dtype / device with UNet latents
                        _ref_latents_all = _ref_latents_all.to(
                            device=latents.device, dtype=latents.dtype
                        )

                        # resize ref-latents to UNet grid if needed
                        if _ref_latents_all.shape[-2:] != latents.shape[-2:]:
                            _ref_latents_all = F.interpolate(
                                _ref_latents_all,
                                size=latents.shape[-2:],
                                mode="bilinear",
                            )

                        print("[Debug] _ref_latents_all stats", 
                            _ref_latents_all.mean().item(), 
                            _ref_latents_all.std().item(), 
                            _ref_latents_all.min().item(), 
                            _ref_latents_all.max().item())

                        # # resize 0/1 mask to UNet spatial size
                        # mask_float = torch.from_numpy(self._face_mask)[None,None].float().to(device)


                        # mask_bool = F.interpolate(
                        #     mask_float, size=latents.shape[-2:], mode="nearest"
                        # ).bool()                                                  # (1,1,H,W)
                        
                        # ------------------------------------------------------------------
                        # # Resize the binary face-mask **and broadcast it to the full CFG batch**
                        # # ------------------------------------------------------------------
                        # B_total   = 2 if self.do_classifier_free_guidance else 1   # uncond + cond
                        
                        # # mask_float = torch.from_numpy(self._face_mask)[None, None].float().to(device)
                        # # mask_bool  = F.interpolate(
                        # # mask_float, size=latents.shape[-2:], mode="nearest").bool() # (1,1,H,W)
                        
                        # # # (B,1,H,W)  →  (B,4,H,W)  so every latent channel is gated
                        # # mask_bool  = mask_bool.repeat(B_total, 1, 1, 1)
                        # # mask_4ch   = mask_bool.repeat(1,       4, 1, 1)            # (B,4,H,W)
                        
                        
                        # # 0/1 mask at UNet resolution (float → bool)
                        # mask_float = torch.from_numpy(self._face_mask)[None, None].float().to(device)
                        # mask_1     = F.interpolate(mask_float, size=latents.shape[-2:], mode="nearest")  # (1,1,H,W)
                        
                        
                        # ------------------------------------------------------------------
                        # (2) reference-mask – detect face on the ORIGINAL RGB
                        #     once with InsightFace, then down-sample to UNet grid
                        # ------------------------------------------------------------------
                        if not hasattr(self, "_ref_mask"):
                            from .insightface_package import FaceAnalysis2, analyze_faces

                            _fa = getattr(self, "_fa_model", None)
                            if _fa is None:                       # load once per pipeline
                                self._fa_model = _fa = FaceAnalysis2(name="buffalo_l")
                                _fa.prepare(ctx_id=0, det_size=(640,640))

                            # id_pixel_values : (1, N, 3, H, W)  – take the first RGB
                            rgb = (id_pixel_values[0,0]             # [3,H,W]  in −1…1
                                   .mul_(127.5).add_(127.5)         # → 0…255
                                   .byte().permute(1,2,0)           # HWC uint8
                                   .cpu().numpy())
                            faces = analyze_faces(_fa, rgb)
                            if not faces:
                                raise RuntimeError("InsightFace found no face!")

                            x0,y0,x1,y1 = faces[0].bbox.astype(int)  # first face
                            mask_h, mask_w = rgb.shape[:2]
                            ref_mask = np.zeros((mask_h,mask_w), dtype=np.uint8)
                            ref_mask[y0:y1, x0:x1] = 1               # crude box

                            # bring to UNet resolution & Tensor
                            mask_1 = torch.from_numpy(ref_mask)[None,None].float().to(device)
                            mask_1 = F.interpolate(mask_1,
                                                    size=latents.shape[-2:],
                                                    mode="nearest").bool()
                            self._ref_mask = mask_1.cpu().clone()    # keep a copy
                        else:
                            mask_1 = self._ref_mask.to(device)
                       
                        
                        
                        

                        # # broadcast to full (B,1,H,W) batch for CFG
                        # mask_bool = mask_bool.repeat(B_total, 1, 1, 1)    

                        # mask_4ch = mask_bool.repeat(1, 4, 1, 1)                 # (B,4,H,W)

                        # # expand mask across the 4 latent channels → (1,4,H,W)
                        # mask_4ch = mask_bool.repeat(1, 4, 1, 1)  # → (1,4,H,W)

                        # _ref_latents_clean = _ref_latents_all.clone()
                        # _ref_latents_clean[~mask_4ch] = 0.0
                        

                        # ─────────────────────────────────────────────────────────────
                        # (NEW)  keep *only* face-region latents and z-score them
                        # ─────────────────────────────────────────────────────────────
                        #
                        # ① build a 4-channel boolean mask, then duplicate for CFG (B=2)
                        #
                        B_total   = 2 if self.do_classifier_free_guidance else 1   # uncond+cond
                        
                        
                        # mask_bool  = mask_1.bool()
                        # mask_4ch  = mask_bool.repeat(1, 4, 1, 1)                   # (1,4,H,W)
                        
                        
                        mask_bool = mask_1.bool()

                        # --------------------------------------------------
                        # Cache the mask on its FIRST build; thereafter keep
                        # re-using the same tensor so the silhouette never
                        # shifts between steps.
                        # --------------------------------------------------
                        if self._face_mask_static is None:
                            self._face_mask_static = mask_bool[0,0].cpu().numpy()
                        else:
                            mask_bool = torch.from_numpy(self._face_mask_static) \
                                           .to(mask_bool.device).bool().unsqueeze(0).unsqueeze(0)

                        # if self._mask_4ch is None:
                        #     self._mask_4ch = mask_bool.repeat(1, 4, 1, 1).clone()
                        # mask_4ch = self._mask_4ch        # immutable thereafter
                        
                        # ---------------- fixed-size broadcast mask ----------------
                        if self._mask_4ch is None:
                            # up-sample the 1×1 mask to UNet spatial size
                            mask_up = F.interpolate(
                                mask_bool.float(),
                                size=latents.shape[-2:],     # e.g. 128×128
                                mode="nearest",
                            ).bool()
                            self._mask_4ch = mask_up.repeat(1, 4, 1, 1).clone()

                        # if the UNet grid changes (rare with SD-XL, but happens
                        # when height/width are non-square) resize on-the-fly
                        if self._mask_4ch.shape[-2:] != latents.shape[-2:]:
                            mask_up = F.interpolate(
                                self._mask_4ch[:, :1].float(),
                                size=latents.shape[-2:],
                                mode="nearest",
                            ).bool()
                            self._mask_4ch = mask_up.repeat(1, 4, 1, 1)

                        mask_4ch = self._mask_4ch        # always correct size
                        
                        mask_4ch  = mask_4ch.expand(B_total, -1, -1, -1).clone()   # (B,4,H,W)
                
                        # ② compute mean / std *only* over face voxels
                        #
                        # face_vals = _ref_latents_all[mask_4ch[:1]].float()         # first batch
                        face_vals = _ref_latents_all[0][mask_4ch[0]].float()       # face voxels only
                        face_mean = face_vals.mean()
                        face_std  = face_vals.std().clamp(min=1e-4)
                        
                        # ③ build a clean tensor: background = 0, face = (z-scored)
                       #
                        # _ref_latents_clean = _ref_latents_all.new_zeros(
                        #                         (B_total, *_ref_latents_all.shape[1:]))   # (B,4,H,W)
                
                        # # normalised (1,4,H,W) → broadcast to (B,4,H,W)
                        # norm_lat = ((_ref_latents_all.float() - face_mean) / face_std) \
                        #              .expand(B_total, -1, -1, -1)
                
                        # # assign *only* the face indices to avoid shape mismatch
                        # _ref_latents_clean[mask_4ch] = norm_lat[mask_4ch].to(_ref_latents_all.dtype)
                        

                        _ref_latents_clean = torch.zeros_like(_ref_latents_all) \
                                                 .expand(B_total, -1, -1, -1).clone()
                        # norm_lat = ((_ref_latents_all.float() - face_mean) / face_std) \
                        #              .expand(B_total, -1, -1, -1)
                                     
                        norm_lat = ((_ref_latents_all.float() - face_mean) / face_std) \
                            .expand(B_total, -1, -1, -1) \
                            .to(_ref_latents_all.dtype)          # keep dtypes in sync
                            
                        _ref_latents_clean[mask_4ch] = norm_lat[mask_4ch]           # bg stays 0
                
                        # # one-off sanity check
                        # if not hasattr(self, "_dbg_mask_once"):
                        #     fσ = _ref_latents_clean[mask_4ch].std().item()
                        #     bσ = _ref_latents_clean[~mask_4ch].std().item()
                        #     print(f"[DBG mask] σ_face={fσ:.3f}  σ_bg={bσ:.3f}")
                        #     from torchvision.utils import save_image
                        #     vis = _ref_latents_clean[0].abs().mean(0, keepdim=True)
                        #     save_image(vis / vis.max(), f"{DEBUG_DIR}/ref_latents_masked.png")
                        #     self._dbg_mask_once = True
                        #     print("[DBG] saved masked reference latents to "
                        #           f"{DEBUG_DIR}/ref_latents_masked.png")
                        
                        # ---- DEBUG   (always write once per run) ----------------
                        if not getattr(self, "_dbg_mask_once", False):
                            fσ = _ref_latents_clean[mask_4ch].std().item()
                            bσ = _ref_latents_clean[~mask_4ch].std().item()
                            print(f"[DBG mask] σ_face={fσ:.3f}  σ_bg={bσ:.3f}")

                            # ① masked-latents preview
                            from torchvision.utils import save_image
                            vis = _ref_latents_clean[0].abs().mean(0, keepdim=True)
                            save_image(vis / vis.max().clamp(min=1e-8),
                                       f"{DEBUG_DIR}/ref_latents_masked.png")

                            # ② RGB crop that seeded the reference latents
                            if not hasattr(self, "_saved_ref_face"):
                                up_mask = F.interpolate(mask_1.float(),
                                                        size=ref_img_for_lat.shape[-2:],
                                                        mode="nearest")[0,0] > 0.5
                                ys, xs = up_mask.nonzero(as_tuple=True)
                                y0,y1,x0,x1 = 0, *ref_img_for_lat.shape[-2:], 0
                                if ys.numel():
                                    y0, y1 = ys.min().item(), ys.max().item()+1
                                    x0, x1 = xs.min().item(), xs.max().item()+1
                                # ref_img_for_lat is bfloat16 – convert first or PIL fails
                                # img_np = ((ref_img_for_lat.to(dtype=torch.float32) * 0.5 + 0.5)
                                img_np = ((ref_img_for_lat.float().add(1).mul(0.5))   
                                # img_np = ((ref_img_for_lat * 0.5 + 0.5)
                                          .permute(1,2,0).clamp(0,1)
                                          .cpu().numpy()*255).astype("uint8")
                                Image.fromarray(img_np[y0:y1, x0:x1]).save(
                                    f"{DEBUG_DIR}/reference_face_crop.png")
                                self._saved_ref_face = True

                            self._dbg_mask_once = True
                        
                                        
                                
                
                        # ④ tiny sanity report
                        print("[DBG norm] face_mean={:.4f}  face_std={:.4f}".format(
                              face_mean.item(), face_std.item()))

                        # ── DEBUG: once, save the RGB patch that will feed the face branch ──
                        if not hasattr(self, "_saved_ref_face"):
                            # 1. up-sample binary mask to reference-image resolution
                            up_mask = F.interpolate(
                                mask_float,                     # (1,1,h,w) – 0/1
                                size=ref_img_for_lat.shape[-2:],  # (H_ref,W_ref)
                                mode="nearest",
                            )[0, 0] > 0.5                      # bool (H,W)

                            # 2. get reference RGB in [0‥255] uint8
                            ref_np = (
                                (ref_img_for_lat.squeeze(0)
                                 .to(dtype=torch.float32) * 0.5 + 0.5)     # de-norm
                                .permute(1, 2, 0)
                                .clamp_(0, 1)
                                .cpu()
                                .numpy()
                            )
                            ref_np = (ref_np * 255).astype("uint8")

                            # 3. tight crop around the mask (fallback to full img)
                            ys, xs = np.where(up_mask.cpu().numpy())
                            if ys.size:
                                y0, y1 = ys.min(), ys.max() + 1
                                x0, x1 = xs.min(), xs.max() + 1
                                ref_np = ref_np[y0:y1, x0:x1]

                            # 4. write to DEBUG_DIR/reference_face_crop.png
                            Image.fromarray(ref_np).save(
                                os.path.join(DEBUG_DIR, "reference_face_crop.png")
                            )
                            self._saved_ref_face = True
                            print(f"[DBG] saved reference face crop to {DEBUG_DIR}/reference_face_crop.png")

                        # create one random tensor we will recycle every step
                        self._step_noise = torch.randn_like(_ref_latents_clean)


                        print(f"[DBG] reference latents prepared  "
                              f"mean={_ref_latents_clean.mean():.4f}")
                        _branched_ready = True
                        
                        #### WTF DIFF                        
                        # Prepare the merge mask at latent resolution
                        if not hasattr(self, "_latent_mask_4ch"):
                            lat_mask = F.interpolate(mask_1.float(), 
                                                   size=latents.shape[-2:], 
                                                   mode="nearest").bool()
                            self._latent_mask_4ch = lat_mask.repeat(1, 4, 1, 1)  # (1,4,H,W)
                        #### WTF DIFF 

                    # ── background branch (regular UNet) ──────────────────
                    # background branch – clear override
                    # self._branched_active = False
                    
                    
                    # self._kv_override = None

                    # noise_bg = self.unet(
                    #     latent_model_input,
                    #     t,
                    #     encoder_hidden_states=current_prompt_embeds,
                    #     timestep_cond=timestep_cond,
                    #     cross_attention_kwargs=self.cross_attention_kwargs,
                    #     added_cond_kwargs=added_cond_kwargs,
                    #     return_dict=False,
                    # )[0]
                    
                    mask_bg  = (~mask_4ch).to(dtype_lat)       # (B,4,H,W) 1 = background
                    mask_face=   mask_4ch.to(dtype_lat)         #            1 = face
                    
                    # -------------- BRANCH 1 : background ----------------------
                    #   Q  = current-noise **background** only
                    #   K,V= current-noise **face** only
                    bg_q  = latent_model_input * mask_bg
                    bg_kv = latent_model_input * mask_face
                    self._kv_override = bg_kv
                    
                    noise_bg = self.unet(
                        bg_q,
                        t,
                        encoder_hidden_states=current_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                   
                    # # ① keep the reference latents *clean & frozen*
                    # kv_noised = _ref_latents_clean          # no add_noise()

                    # ① add the *correct* noise so that face-branch K/V
                    #    lives on the same σₜ manifold as the UNet latents
                    # kv_noised = self.scheduler.add_noise(
                    #     _ref_latents_clean,
                    #     self._step_noise,
                    #     t.expand(_ref_latents_clean.shape[0])
                    # )

                    # # Use the same noise schedule as the main denoising process
                    # if not hasattr(self, "_ref_noise"):
                    #     self._ref_noise = torch.randn_like(_ref_latents_clean)
                        
                #     # Create noise tensor if needed
                #     if not hasattr(self, "_ref_noise"):
                #         # Use a fixed random noise that we'll scale by timestep
                #         self._ref_noise = torch.randn_like(_ref_latents_all)
                    
                #    # Add noise to clean reference latents to match current timestep
                #     kv_noised = self.scheduler.add_noise(
                #         # _ref_latents_clean,
                #         _ref_latents_all,  # Use full reference, not masked
                #         self._ref_noise,
                #         t.expand(_ref_latents_clean.shape[0])
                #     )

                    # Use **clean** reference latents – we want a crisp identity signal,
                    # not a noisy one.  This single line fixes the low-contrast / “grey
                    # mush” artefact that survived the previous tweaks.
                    
                    kv_noised = _ref_latents_all.clone()
                    
                    

                    # ------------- FACE  BRANCH --------------------------------
                    # 1) expose reference KV to the patched self-attention
                    # self._kv_override = _ref_latents_clean    # only face region, z-scored
                    
                    ## NEW FIX 04 AUG ###
                    # For face branch, we need to use the full reference latents, not just the masked version
                    # The mask will be applied during the attention computation in the custom_attn_forward
                    
                    #### WTF DIFF 
                    
                    # # Ensure reference latents match the batch size
                    # if _ref_latents_all.shape[0] == 1 and latent_model_input.shape[0] > 1:
                    #     ref_for_face = _ref_latents_all.expand(latent_model_input.shape[0], -1, -1, -1)
                    # else:
                    #     ref_for_face = _ref_latents_all
                    
                    # Create a hybrid latent for face branch:
                    # - Keep background from current latent_model_input
                    # - Replace face region with reference latents
                    if hasattr(self, "_latent_mask_4ch"):
                        # Expand mask to match batch size
                        mask_4ch_batch = self._latent_mask_4ch
                        if self.do_classifier_free_guidance and mask_4ch_batch.shape[0] == 1:
                            mask_4ch_batch = mask_4ch_batch.repeat(2, 1, 1, 1)
                        
                        # Create hybrid latent: background from current, face from reference
                        ref_expanded = _ref_latents_all
                        if ref_expanded.shape[0] < latent_model_input.shape[0]:
                            ref_expanded = ref_expanded.expand(latent_model_input.shape[0], -1, -1, -1)
                        
                        # For face branch: use reference in face region, current in background
                        # latent_face_input = torch.where(mask_4ch_batch, ref_expanded, latent_model_input)
                        
                        # else:
                        #     latent_face_input = latent_model_input
                        
                        
                        # --- build per-branch latent ---------------------------------
                        #   • unconditional half (first B/2) → leave as-is
                        #   • conditional half             → replace face pixels only
                        if self.do_classifier_free_guidance and latent_model_input.shape[0] % 2 == 0:
                            B_half = latent_model_input.shape[0] // 2
    
                            lat_cond = torch.where(
                                mask_4ch_batch[B_half:],                  # face = True
                                # expand reference if necessary
                                (ref_expanded if ref_expanded.shape[0] == 1
                                               else ref_expanded[B_half:]),
                                latent_model_input[B_half:],              # background
                            )
                            latent_face_input = torch.cat(
                                [latent_model_input[:B_half], lat_cond],  # (uncond | cond)
                                dim=0,
                            )
                        else:
                            # no CFG → single slice
                            latent_face_input = torch.where(
                                mask_4ch_batch, ref_expanded, latent_model_input
                            )

                    
                    #
                    
                    # # Scale reference to match current noise level
                    # ref_std = ref_for_face.std(dim=(1,2,3), keepdim=True).clamp(min=1e-4)
                    # q_std = latent_model_input.std(dim=(1,2,3), keepdim=True)
                    # ref_scaled = ref_for_face * (q_std / ref_std)
                    
                    
                    
                    # Set the FULL reference for K/V override in attention
                    # Using _ref_latents_all instead of _ref_latents_clean ensures
                    # that background regions have valid K/V to attend to
                    self._kv_override = _ref_latents_all
                    if self._kv_override.shape[0] < latent_model_input.shape[0]:
                        self._kv_override = self._kv_override.expand(latent_model_input.shape[0], -1, -1, -1)
                    
                   
                    # Also set a flag to indicate we're in face branch
                    self._face_branch_active = True
                    
                    # self._kv_override = ref_scaled
                    
                    ### WTF DIFF 
                    
                    
                    ## NEW FIX 04 AUG ###

                    noise_face = self.unet(
                        # latent_model_input,
                        latent_face_input,
                        t,
                        encoder_hidden_states=current_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    self._face_branch_active = False ### WTF DIFF 
                    
                    
                    # ─────────────────────────────────────────────────────────
                    #  PREVIEW DUMPS  (face + background)
                    #  – first branched step
                    #  – every <interval>
                    #  – very last step
                    # ─────────────────────────────────────────────────────────

                    do_dump = (
                        i % face_branch_interval == 0
                        or i == branched_attn_start_step
                        or i == len(timesteps) - 1          # final step
                    )
                    
                    
                    if do_dump and (debug_save_face_branch or debug_save_bg_branch):

                        # keep scheduler state intact
                        _saved_idx = getattr(self.scheduler, "_step_index", None)

                        def _step_and_decode(noise):
                            lat = self.scheduler.step(
                                noise, t, latents.detach().clone(),
                                **extra_step_kwargs, return_dict=False
                            )[0]
                            if _saved_idx is not None:
                                self.scheduler._step_index = _saved_idx
                            vae_dev = next(self.vae.parameters()).device
                            img = self.vae.decode(
                                (lat / self.vae.config.scaling_factor)
                                .to(device=vae_dev, dtype=self.vae.dtype)
                            ).sample[0]
                            img_np = (
                                (img.float() / 2 + 0.5)
                                .clamp_(0, 1)
                                .permute(1, 2, 0)
                                .cpu()
                                .numpy() * 255
                            ).astype("uint8")
                            return img_np

                        # binary face mask @ full RGB size for visual masking
                        # m_np = None
                        # if hasattr(self, "_merge_mask_t"):
                        #     m_np = self._merge_mask_t[0, 0].cpu().numpy().astype(bool)
                            
                        # latent-resolution face mask; will be up-sampled on demand
                        m_np_lat = None
                        if hasattr(self, "_merge_mask_t"):
                            m_np_lat = self._merge_mask_t[0, 0].cpu().numpy().astype(bool)

                        # ---------------- FACE branch -------------------
                        if debug_save_face_branch:
                            # img_np = _step_and_decode(noise_face)
                            # if m_np is not None:            # zero background
                            #     img_np = img_np.copy()
                            #     img_np[~m_np] = 127
                            img_np = _step_and_decode(noise_face)
                            if m_np_lat is not None:        # zero background
                                # resize latent-grid mask → RGB resolution
                                H, W = img_np.shape[:2]
                                m_big = np.array(
                                    Image.fromarray(m_np_lat.astype(np.uint8)*255)
                                         .resize((W, H), Image.NEAREST)
                                ).astype(bool)
                                img_np = img_np.copy()
                                # img_np[~m_big] = 127
                            out_fb = os.path.join(
                                DEBUG_DIR, f"face_branch_{i:03d}.png"
                            )
                            Image.fromarray(img_np).save(out_fb)
                            print(f"[DBG] face-branch preview saved → {out_fb}")

                        # ---------------- BG branch ---------------------
                        if debug_save_bg_branch:
                            # img_np = _step_and_decode(noise_bg)
                            # if m_np is not None:            # zero face area
                            #     img_np = img_np.copy()
                            #     img_np[m_np] = 127

                            img_np = _step_and_decode(noise_bg)
                            if m_np_lat is not None:        # zero face area
                                H, W = img_np.shape[:2]
                                m_big = np.array(
                                    Image.fromarray(m_np_lat.astype(np.uint8)*255)
                                         .resize((W, H), Image.NEAREST)
                                ).astype(bool)
                                img_np = img_np.copy()
                                # img_np[m_big] = 127

                            out_bg = os.path.join(
                                DEBUG_DIR, f"background_branch_{i:03d}.png"
                            )
                            Image.fromarray(img_np).save(out_bg)
                            print(f"[DBG] background-branch preview saved → {out_bg}")

                        # # run *only* the face-branch prediction through one
                        # # scheduler step, then decode to RGB and save
                        # lat_face = self.scheduler.step(
                        #     noise_face, t, latents.detach().clone(),
                        #     **extra_step_kwargs, return_dict=False
                        # )[0]
                        
                        # run *only* the face-branch prediction through one
                        # scheduler step **without** altering the global
                        # scheduler state used by the main loop
                        
                        
                        # _saved_idx = getattr(self.scheduler, "_step_index", None)
                        
                        # lat_face   = self.scheduler.step(
                        #     noise_face, t, latents.detach().clone(),
                        #     **extra_step_kwargs, return_dict=False
                        # )[0]
                        # if _saved_idx is not None:
                        #     self.scheduler._step_index = _saved_idx

                        # vae_dev = next(self.vae.parameters()).device
                        # img_fb = self.vae.decode(
                        #     (lat_face / self.vae.config.scaling_factor)
                        #     .to(device=vae_dev, dtype=self.vae.dtype)
                        # ).sample[0]
                        # img_np = (
                        #     (img_fb.float() / 2 + 0.5)
                        #     .clamp_(0, 1)
                        #     .permute(1, 2, 0)
                        #     .cpu()
                        #     .numpy() * 255
                        # ).astype("uint8")

                        # out_fb = os.path.join(
                        #     DEBUG_DIR, f"face_branch_{i:03d}.png"
                        # )
                        # Image.fromarray(img_np).save(out_fb)
                        # print(f"[DBG] face-branch preview saved → {out_fb}")

                    # 2) clear override for the rest of the pipeline
                    self._kv_override = None

                    # 3) blend the two predictions with the 4-channel mask
                    
                    mask_f = mask_4ch.to(noise_bg.dtype)      # (B,4,H,W)
                    noise_pred = noise_bg * (1.0 - mask_f) + noise_face * mask_f

                # -------- continue standard SD-XL loop -----------------

                # Scheduler step just below already expects `noise_pred`


                    
                    # ─── DEBUG ──────────────────────────────────────────────────────
                    # Compare *raw* σ of the reference K/V bank with the current query
                    # latents **before** any variance-matching rescale.
                    if i == branched_attn_start_step:
                        print(
                            "[DBG σ-raw] kv_std = {:.4f}   q_std = {:.4f}".format(
                                kv_noised.std().item(),
                                latent_model_input.std().item()
                            )
                        )
                                    
                    # if not hasattr(self, "_ref_noise"):
                    #      self._ref_noise = torch.randn_like(_ref_latents_all)  # fixed seed per run
                    
                    # kv_noised = self.scheduler.add_noise(
                    #      _ref_latents_all,
                    #      self._ref_noise,            # any noise - we scale it anyway
                    #      t.expand(_ref_latents_all.shape[0])
                    #  )

                    # keep the reference latents **clean** – adding extra noise
                    # wipes out identity detail and produces a dull, grey face
                    kv_noised = _ref_latents_all      # no `add_noise` here
                    
                    # # Create noise tensor if needed
                    # if not hasattr(self, "_ref_noise"):
                    #     self._ref_noise = torch.randn_like(_ref_latents_all)
                    
                    # # Add noise to match current timestep - this is crucial for proper denoising
                    # kv_noised = self.scheduler.add_noise(
                    #     _ref_latents_all,
                    #     self._ref_noise,
                    #     t.expand(_ref_latents_all.shape[0])
                    # )
                    
                    # # Expand to match batch size for CFG
                    # if self.do_classifier_free_guidance and kv_noised.shape[0] == 1:
                    #     kv_noised = kv_noised.repeat(2, 1, 1, 1)
                    
                    # # Ensure we have the right batch size
                    # B_total = 2 if self.do_classifier_free_guidance else 1

                    # # ② scale **before** masking so zeros don’t skew σ
                    # q_std  = latent_model_input.std(dim=(1,2,3), keepdim=True)
                    # kv_std = kv_noised.std(dim=(1,2,3), keepdim=True).clamp(min=1e-4)
                    # ref_kv_t = kv_noised * (q_std / kv_std)
                    
                    # # Watch raw σ before scaling
                    # if i == branched_attn_start_step:
                    #     print("raw  kv_std =", kv_noised.std().item(),
                    #         "   raw  q_std =", latent_model_input.std().item())

                                    
                                        
                    # ------------------------------------------------------------------
                    # (1) bring overall σ of reference K/V to query σ …
                    # ------------------------------------------------------------------
                    q_std  = latent_model_input.std(dim=(1,2,3), keepdim=True)
                    kv_std = kv_noised.std(dim=(1,2,3), keepdim=True).clamp(min=1e-4)
                    kv_noised = kv_noised * (q_std / kv_std)
                    

                    # ------------------------------------------------------------------
                    # (2) …then fine-tune so that the **face region only** has the same
                    #     variance as the queries.  This removes the residual overshoot
                    #     you saw in the debug line (KV(face)=1.023 vs Q≈0.988).
                    # ------------------------------------------------------------------
                    
                    # `kv_noised` has *2* batches when CFG is on
                    # (uncond | cond) whereas `mask_4ch` is (1,4,H,W).
                    # Expand the mask so the shapes line-up, then
                    # measure σ on the conditional half only.

                    # if kv_noised.shape[0] != mask_4ch.shape[0]:
                    #     mask_exp = mask_4ch.expand(kv_noised.shape[0], -1, -1, -1)
                    # else:
                    #     mask_exp = mask_4ch
                    
                    mask_exp = mask_4ch  
                                 
                    # # Ensure mask_exp matches the batch size of kv_noised
                    # mask_exp = mask_exp.expand(kv_noised.shape[0], -1, -1, -1)

                    # Face pixels reside in the *conditional* slice (2nd half)
                    start = kv_noised.shape[0] // 2 if self.do_classifier_free_guidance else 0
                    face_std = kv_noised[start:][mask_exp[start:]].std().clamp(min=1e-4)

                    kv_noised = kv_noised * (q_std / face_std)


                    # # ③ finally zero-out the background

                    # # ref_kv_t = ref_kv_t * mask_4ch  # Don't mask K/V here - we'll mask the output

                    # # ─── now force the binary mask to the UNet grid every time ───
                    # #     so there’s never an off-by-a-factor
                    # mask_bool = F.interpolate(
                    #     mask_float, size=ref_kv_t.shape[-2:], mode="bilinear", align_corners=False
                    # ).bool()
                    # # ref_kv_t = ref_kv_t * mask_bool.repeat(1, ref_kv_t.shape[1], 1, 1)  # Don't mask K/V
                    

                    # # ④ re-scale *after* masking so face-region σ ≈ Q σ
                    # mask_exp  = mask_4ch.expand(ref_kv_t.shape[0], -1, -1, -1)   # (B,4,H,W)
                    # face_std  = ref_kv_t[mask_exp].std(dim=None, keepdim=True).clamp(min=1e-4)
                    # ref_kv_t  = ref_kv_t * (q_std / face_std)

                    # Store the noised reference for K/V computation
                    # ref_kv_t = kv_noised

                    # use the *masked & normalised* face-only tensor as K/V
                    # (background zeros are fine – the branch only serves face queries)
                    ref_kv_t = _ref_latents_clean
                    

                    # Debug statistics - define q_std before using it
                    mask_exp  = mask_4ch.expand(ref_kv_t.shape[0], -1, -1, -1)   # (B,4,H,W)
                    q_std = latent_model_input.std(dim=(1,2,3), keepdim=True)
 


                    # After preparing ref_kv_t, save a debug image
                    if i == branched_attn_start_step:
                        # debug_img = self.vae.decode(ref_kv_t / self.vae.config.scaling_factor).sample[0]
                        # debug_img = (debug_img / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

                        # Ensure VAE gets the right dtype
                        vae_device = next(self.vae.parameters()).device
                        vae_dtype = next(self.vae.parameters()).dtype
                        # ref_for_decode = (ref_kv_t / self.vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)
                        ref_for_decode = (_ref_latents_all / self.vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)
                        debug_img = self.vae.decode(ref_for_decode).sample[0]
                        debug_img = (debug_img.float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray((debug_img * 255).astype(np.uint8)).save("debug_ref_latents.png")
                        print("[Debug] saved reference latents image to debug_ref_latents.png")
                                        


                    if not hasattr(self, "_dbg_scale_once"):
                    

                        # _mask  = mask_4ch.expand(ref_kv_t.shape[0], -1, -1, -1)
                        q_val  = q_std.mean().item()
                        _mask  = mask_exp
                        face_v = ref_kv_t[_mask].std().item()
                        back_v = ref_kv_t[~_mask].std().item()
                        glob_v = ref_kv_t.std().item()

                        print(f"[DBG σ] t={t.item():>4} | "
                              f"Q={q_val:.3f} | "
                              f"KV(face)={face_v:.3f} | "
                              f"KV(bg)={back_v:.3f} | "
                              f"KV(global)={glob_v:.3f}")
                        
                        self._dbg_scale_once = True

                    # face branch – set override
                    # self._kv_override = ref_kv_t
                    # # self._branched_active = True

                    # # run face branch with reference K/V
                    # noise_face = self.unet(
                    #     # latent_model_input,
                    #     latent_face_input,
                    #     t,
                    #     encoder_hidden_states=current_prompt_embeds,
                    #     timestep_cond=timestep_cond,
                    #     cross_attention_kwargs=self.cross_attention_kwargs,
                    #     # added_cond_kwargs=added_cond_face,
                    #     added_cond_kwargs={**added_cond_kwargs,
                    #                         "kv_override": ref_kv_t},
                    #     return_dict=False,
                    # )[0]
                    
                    #   Q  = current-noise **face** only
                    #   K,V= **reference-face** only
                    face_q  = latent_model_input * mask_face
                    face_kv = ref_kv_t.to(dtype_lat) * mask_face        # ref_kv_t was already face-only & z-scored
                    self._kv_override = face_kv
                    noise_face = self.unet(
                        face_q,
                        t,
                        encoder_hidden_states=current_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        # added_cond_kwargs=added_cond_face,
                        added_cond_kwargs={**added_cond_kwargs,
                                            "kv_override": ref_kv_t},
                        return_dict=False,
                    )[0]

                    # ── merge two predictions by resized mask ─────────────

                    # mask_float = torch.from_numpy(self._face_mask)[None,None].float().to(latents.device)
                    # mask_bool  = F.interpolate(
                    #     mask_float, size=noise_bg.shape[-2:], mode="nearest"
                    # ).bool()

                    # noise_pred = torch.where(mask_bool, noise_face, noise_bg)
                    
                    
                    # #   *ref_mask*   – already used for K/V – stays internal
                    # #   *merge_mask* – drives the bg/face merge here
                    # if self._merge_mask is None:
                    #     merge_mask = mask_1          # fallback: use ref-mask
                    # else:
                    #     merge_mask = torch.from_numpy(self._merge_mask)[None,None] \
                    #                    .float().to(latents.device)
                    #     merge_mask = F.interpolate(merge_mask,
                    #                                 size=noise_bg.shape[-2:],
                    #                                 mode="nearest").bool()
                        
                    # prepare resized torch mask **once** (outside the loop)
                    if not hasattr(self, "_merge_mask_t"):
                        if self._merge_mask is None:
                            self._merge_mask_t = mask_1          # fallback
                        else:
                            _m = torch.from_numpy(self._merge_mask)[None,None] \
                                    .float().to(latents.device)
                            self._merge_mask_t = F.interpolate(
                                    _m, size=noise_bg.shape[-2:], mode="nearest"
                                ).bool()
                    # merge_mask = self._merge_mask_t
                        
                  
                    # # # Ensure merge_mask matches batch size
                    # # if merge_mask.shape[0] == 1 and noise_bg.shape[0] > 1:
                    # #     merge_mask = merge_mask.expand(noise_bg.shape[0], -1, -1, -1)

                    # noise_pred = torch.where(merge_mask, noise_face, noise_bg)

                    merge_mask = self._merge_mask_t            # bool tensor

                    if not hasattr(self, "_dbg_overlay"):
                        from torchvision.utils import save_image
                        vis = merge_mask.float().expand(-1,3,-1,-1)  # make it 3-ch
                        save_image(vis, f"{DEBUG_DIR}/merge_mask_vis.png")
                        self._dbg_overlay = True


                    # # user-supplied mask → white = background
                    # if self._merge_mask is not None:
                    #     # inside mask  ⇒ background   ;  outside ⇒ face
                    #     noise_pred = torch.where(merge_mask, noise_bg, noise_face)
                    # else:
                    #     # internal mask (heat-map / InsightFace) → white = face
                    #     noise_pred = torch.where(merge_mask, noise_face, noise_bg)

                    # merge_mask: 1 = face, 0 = background
                    # So we use: face pixels from noise_face, background pixels from noise_bg

                    if i == branched_attn_start_step:
                            # how many pixels go to the background branch?
                            print(f"[DBG merge] mask-mean={merge_mask.float().mean():.3f}  "
                                  f"noise_bg σ={noise_bg.std():.4f}  "
                                  f"noise_face σ={noise_face.std():.4f}")
                        
                            # verify that the mask has the same spatial size as noise_pred slices
                            print(f"[DBG merge] mask shape {merge_mask.shape}  "
                                  f"noise branch shape {noise_bg.shape[-2:]}")


                    # noise_pred = torch.where(merge_mask, noise_face, noise_bg)
                    # clear for the rest of the pipeline
                    self._kv_override = None
                    
                    # blend by mask (1 = face)
                    noise_pred = torch.where(merge_mask, noise_face, noise_bg)

                    
                    if i == branched_attn_start_step:
                        nz = (ref_kv_t.abs() > 0).float().mean()
                        print(f"[DBG] mask-true ratio = {merge_mask.float().mean():.3f}  "
                              f"face-K/V non-zero = {nz:.3f}")
                
                        

                    # reset flag for next iteration
                    # self._branched_active = False
                    self._kv_override = None

                else:
                    # single-branch (original) call
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

                    # if mask_mode == "spec":
                    #     self._face_mask = compute_binary_face_mask(
                    #         snapshot, MASK_LAYERS_CONFIG)

                    #     if build_spec_mask and mask_mode == "spec":
                    #         self._face_mask = compute_binary_face_mask(
                    #             snapshot, MASK_LAYERS_CONFIG)

                    # else:
                    #     self._face_mask = simple_threshold_mask(snapshot)

                        # if self._merge_mask is None:        # keep imported mask
                        #     if build_spec_mask and mask_mode == "spec":
                        #         self._face_mask = compute_binary_face_mask(
                        #             snapshot, MASK_LAYERS_CONFIG)
                        #     else:
                        #         self._face_mask = simple_threshold_mask(snapshot)


                        # if build_spec_mask and mask_mode == "spec":
                        #     self._face_mask = compute_binary_face_mask(
                        #         snapshot, MASK_LAYERS_CONFIG)
                        # else:
                        #     self._face_mask = simple_threshold_mask(snapshot)
                            
                            
                        # --------------------------------------------------
                        # Build the face-mask only once.  Subsequent passes
                        # just re-use the cached `_face_mask_static`.
                        # --------------------------------------------------
                        # if self._face_mask_static is None:
                        #     # if build_spec_mask and mask_mode == "spec":
                        #     if build_spec_mask and mask_mode == "spec" and self._merge_mask is None:
                        #        self._face_mask_static = compute_binary_face_mask(
                        #             snapshot, MASK_LAYERS_CONFIG)
                        #     else:
                        #         # self._face_mask_static = simple_threshold_mask(snapshot)
                        #         self._face_mask_static = simple_threshold_mask(snapshot) if self._merge_mask is None else self._merge_mask
                        # self._face_mask = self._face_mask_static


                        if not self._freeze_mask and self._face_mask_static is None:
                            # Build mask from attention maps
                            if mask_mode == "spec":
                                self._face_mask = compute_binary_face_mask(snapshot, MASK_LAYERS_CONFIG)
                            else:
                                self._face_mask = simple_threshold_mask(snapshot)
                            
                            # Don't make it static yet for no_branched_attention case
                            if not use_branched_attention:
                                # Allow mask to update each step
                                print(f"[Debug] Updated face mask: pixels={self._face_mask.sum()}")
                            else:
                                # For branched attention, freeze after first build
                                self._face_mask_static = self._face_mask
                                        
                    if FULL_DEBUG:
                        print("[Debug] face mask:", self._face_mask.shape, 
                            self._face_mask.sum(), self._face_mask.dtype)

                    attn_maps_current.clear()   # reset → one-step snapshot

                # call the callback, if provided (now sees fresh mask)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                
                # NEW 30 JUL                    
                
            
                    
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
                        
                        # if mask_mode == "spec":          # weighted-union (default)
                        #     self._face_mask = compute_binary_face_mask(
                        #         snapshot, MASK_LAYERS_CONFIG)
                        # else:                           # "simple" → single-layer top-k
                        #     self._face_mask = simple_threshold_mask(snapshot)

                        # if self._merge_mask is None:      # keep imported mask
                        #     if mask_mode == "spec":          # weighted-union
                        #         self._face_mask = compute_binary_face_mask(
                        #             snapshot, MASK_LAYERS_CONFIG)
                        #     else:                           # "simple"
                        #         self._face_mask = simple_threshold_mask(snapshot)
                        
                        if mask_mode == "spec":              # weighted-union
                            self._face_mask = compute_binary_face_mask(
                                snapshot, MASK_LAYERS_CONFIG)
                        else:                               # "simple"
                            self._face_mask = simple_threshold_mask(snapshot)
                        
                        print("[Debug] final face mask stats:", self._face_mask.shape, self._face_mask.sum())
                    
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



                    # -------------------------------------------------------------
                    # ❶ Build the mask exactly *once* (or whenever you like)
                    # -------------------------------------------------------------
                    if (use_branched_attention and
                        i == branched_attn_start_step):
                        # Average heads collected so far
                        snapshot = {ln: np.stack(m).mean(0)
                                    for ln, m in attn_maps_current.items()
                                    if len(m)}
        
                        # if snapshot:                         # we have something!
                        if snapshot and self._merge_mask is None:  # keep imported mask
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
                        else:
                            warnings.warn("[BranchedAttn] no attention maps "
                                          "captured – mask disabled.")
        
                        # maps no longer needed – free memory
                        attn_maps_current.clear()
        
                        
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
                                    
                                # --- unconditional path MUST ignore face K/V ---
                                out_uncond = None
                                if hidden_uncond is not None:
                                    prev_kv = self._kv_override           # save current override
                                    self._kv_override = None              # disable for uncond
                                    out_uncond = module._orig_forward(
                                        hidden_uncond, None, attention_mask
                                    )
                                    self._kv_override = prev_kv           # restore for face branch
                                    
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
                                mask = torch.from_numpy(self._face_mask)  # shape (H_map, W_map)
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
                                    # ------------------------------------------------------------------
                                    # build the face-query slice *first* so we can reuse it below
                                    # ------------------------------------------------------------------
                                    Q_face = Q[:, :, face_idx, :]          # (B_cond, h, L_face, d)

      
                                    
                                    # ---------- reference K/V (prepared once) ----------
                                    if not hasattr(module, "_ref_kv"):
                                        ref = self._ref_feats            # may be (M,F), (512,) or (B,4,H,W)
                                        if ref is None:
                                            raise RuntimeError("No reference features available for face branch!")

                                        proj_k = module.to_k if hasattr(module,"to_k") else module.k_proj
                                        proj_v = module.to_v if hasattr(module,"to_v") else module.v_proj
                                        in_dim = proj_k.weight.shape[1]

                                        if   ref.dim() == 4:                             # (B,4,H,W)
                                            B,C,H,W   = ref.shape
                                            ref_flat  = ref.permute(0,2,3,1).reshape(B, -1, C)  # (B,L,C)
                                            if C < in_dim:
                                                ref_flat = F.pad(ref_flat,(0,in_dim-C))
                                            elif C > in_dim:
                                                ref_flat = ref_flat[...,:in_dim]
                                            K_ref = proj_k(ref_flat.to(Q.dtype))
                                            V_ref = proj_v(ref_flat.to(Q.dtype))

                                        elif ref.dim() == 2:                             # (M,F)
                                            # feats  = ref.unsqueeze(0).to(Q.dtype)         # (1,M,F)
                                            # K_ref  = proj_k(feats)
                                            # V_ref  = proj_v(feats)

                                            M, Fdim = ref.shape
                                            if Fdim < in_dim:                      # pad channels
                                                ref_adj = F.pad(ref, (0, in_dim - Fdim))
                                            elif Fdim > in_dim:                    # or clip
                                                ref_adj = ref[:, :in_dim]
                                            else:
                                                ref_adj = ref

                                            feats = ref_adj.unsqueeze(0).to(Q.dtype)   # (1,M,in_dim)
                                            K_ref = proj_k(feats)                     # (1,M,C)
                                            V_ref = proj_v(feats)
                                            
                                            # # CLIP features: (256, 1024) but attention expects in_dim (640)
                                            # feats = ref.to(Q.dtype)  # (M, F)
                                            # M, N = feats.shape
                                            
                                            # if N > in_dim:
                                            #     # Use only first in_dim features
                                            #     feats = feats[:, :in_dim]
                                            # elif N < in_dim:
                                            #     # Pad with zeros
                                            #     feats = F.pad(feats, (0, in_dim - N))
                                            
                                            # # Ensure we have the right shape: (1, M, in_dim)
                                            # feats = feats.unsqueeze(0)  # (1, M, in_dim)
                                            
                                            # # Project through K/V projections
                                            # K_ref = proj_k(feats)  # (1, M, inner_dim)
                                            # V_ref = proj_v(feats)  # (1, M, inner_dim)
 
                                        elif ref.dim() == 1:                             # (512,)
                                            vec = ref.to(Q.dtype)
                                            if vec.shape[0] < in_dim:
                                                rep = (in_dim + vec.shape[0] - 1)//vec.shape[0]
                                                vec = vec.repeat(rep)[:in_dim]
                                            else:
                                                vec = vec[:in_dim]
                                            K_ref = vec.unsqueeze(0).unsqueeze(0)         # (1,1,d)
                                            V_ref = K_ref.clone()
                                        else:
                                            raise RuntimeError(f"Unsupported ref_feats shape {ref.shape}")

                                        module._ref_kv = (K_ref.detach(), V_ref.detach())

                                        # if not hasattr(module, "_debug_kv_once"):
                                        #     print(f"[BranchedAttn] cached KV  shape={K_ref.shape} ‖K‖={K_ref.norm():.3f}")
                                        #     module._debug_kv_once = True


    
                                    K_ref, V_ref = module._ref_kv
                                        

                                   
                                    # # --- idea #2: scale reference K/V to Q ---
                                    # # Compute global σ of current face-query vectors (Q_face) and
                                    # # reference keys (K_ref).  Then scale K_ref and V_ref so that
                                    # # their magnitude matches the query magnitude for this
                                    # # timestep / layer.  This keeps the soft-max logits in a
                                    # # numerically comparable range and largely removes the “white
                                    # # noise face” pathology.
                                    
                                    # --- match ref KV variance to current queries -----------------
                                    q_std  = Q_face.detach().float().std()
                                    kv_std = K_ref.detach().float().std().clamp(min=1e-4)
                                    scale  = (q_std / kv_std).to(K_ref.dtype)
                                    K_ref  = K_ref * scale
                                    V_ref  = V_ref * scale
                                    
                                    # No additional scaling to K_ref/V_ref needed here, as 
                                    # _ref_latents_all already scaled correctly by VAE & scheduler.

                                    # # Explicit scaling to match Q distribution
                                    # q_std  = Q_face.detach().float().std().clamp(min=1e-6)
                                    # kv_std = K_ref.detach().float().std().clamp(min=1e-6)
                                    # scale  = (q_std / kv_std).to(K_ref.dtype)
                                    # K_ref  = K_ref * scale
                                    # V_ref  = V_ref * scale

                                    # print("[Debug] Scaling K_ref/V_ref explicitly with factor:", scale.item())

                                    # # Debug stats
                                    # print("[Debug] K_ref stats post-scale:", K_ref.mean().item(), K_ref.std().item())
                                    # print("[Debug] Q_face stats:", Q_face.mean().item(), Q_face.std().item())
                                    

                                    # Don't scale K_ref/V_ref - they should already be normalized
                                    # from the clean reference latents. The queries Q_face come from
                                    # the noised latents at timestep t, so they're already on the
                                    # correct scale.

                                    # -------------- debug (runs once per layer) --------------
                                    
                                    if FULL_DEBUG:                                                                        
                                        if not hasattr(module, "_dbg_face_once"):
                                            print(f"[DBG] branched-attn layer_dim={d}  "
                                                f"L_face={L_face}  K_ref={tuple(K_ref.shape)}")
                                            module._dbg_face_once = True


                                    # Compute face-query attention
                                    # # (Q_face already defined)
                                    # # Expand K_ref, V_ref to include head dimension
                                    # K_ref_h = K_ref.unsqueeze(0).expand(h, -1, -1)  # (h, M, d)
                                    # V_ref_h = V_ref.unsqueeze(0).expand(h, -1, -1)  # (h, M, d)
                                    # # Compute dot-product attention for face queries
                                    # # Reshape Q_face to (B_cond*h, L_face, d)
                                    # # Q_face_flat = Q_face.permute(0,2,1,3).reshape(Bc * L_face, h, d)  # Actually, easier to loop per head than flatten differently
                                    # Q_face_flat = Q_face.reshape(Bc * h, L_face, d)  # (B_cond* h, L_face, d)

                                    # # Verify shape alignment
                                    # if K_ref_h.shape[1] != L_face:
                                    #     K_ref_h = F.interpolate(K_ref_h.permute(0,2,1), size=L_face, mode="linear").permute(0,2,1)
                                    #     V_ref_h = K_ref_h.clone()

                                    # # Prepare K_ref repeated for batch
                                    # K_ref_flat = K_ref_h.unsqueeze(0).expand(Bc, -1, -1, -1).reshape(Bc * h, -1, d)
                                    # V_ref_flat = V_ref_h.unsqueeze(0).expand(Bc, -1, -1, -1).reshape(Bc * h, -1, d)


                                    # # Reshape reference K/V for multi-head attention
                                    # if face_embed_strategy == "heatmap":
                                    #     # K_ref, V_ref are (B, L_ref, inner_dim)
                                    #     B_kv, L_ref, C_kv = K_ref.shape
                                    #     K_ref_heads = K_ref.view(B_kv, L_ref, h, d).permute(0, 2, 1, 3)  # (B, h, L_ref, d)
                                    #     V_ref_heads = V_ref.view(B_kv, L_ref, h, d).permute(0, 2, 1, 3)  # (B, h, L_ref, d)
                                        
                                    #     # If batch sizes don't match, expand
                                    #     if B_kv < Bc:
                                    #         K_ref_heads = K_ref_heads.expand(Bc, -1, -1, -1)
                                    #         V_ref_heads = V_ref_heads.expand(Bc, -1, -1, -1)
                                    # else:
                                    #     # K_ref, V_ref are (1, 1, inner_dim) for ArcFace
                                    #     K_ref_heads = K_ref.view(1, 1, h, d).permute(0, 2, 1, 3).expand(Bc, -1, -1, -1)
                                    #     V_ref_heads = V_ref.view(1, 1, h, d).permute(0, 2, 1, 3).expand(Bc, -1, -1, -1)

                                    # Ensure K_ref and V_ref have correct batch dimension
                                    if K_ref.shape[0] < Bc:
                                        K_ref = K_ref.expand(Bc, -1, -1)
                                        V_ref = V_ref.expand(Bc, -1, -1)
                                    
                                    # Reshape for multi-head attention
                                    B_kv, L_kv, C_kv = K_ref.shape
                                    K_ref_heads = K_ref.view(B_kv, L_kv, h, d).permute(0, 2, 1, 3)  # (B, h, L_kv, d)
                                    V_ref_heads = V_ref.view(B_kv, L_kv, h, d).permute(0, 2, 1, 3)  # (B, h, L_kv, d)
                                    
                                    # Flatten for batched matrix multiplication
                                    Q_face_flat = Q_face.reshape(Bc * h, L_face, d)
                                    K_ref_flat = K_ref_heads.reshape(Bc * h, -1, d)
                                    V_ref_flat = V_ref_heads.reshape(Bc * h, -1, d)


                                    # print("[Debug] K_ref stats:", K_ref.mean().item(), K_ref.std().item(), K_ref.min().item(), K_ref.max().item())
                                    # print("[Debug] Q_face stats:", Q_face.mean().item(), Q_face.std().item(), Q_face.min().item(), Q_face.max().item())

                                    
                                    # Compute attention scores
                                    boost = 12.0 # give the reference branch a stronger pull
                                    
                                    attn_scores_face = torch.baddbmm(
                                        torch.empty(Bc * h, L_face, K_ref_flat.shape[1], device=Q_face_flat.device, dtype=Q_face_flat.dtype),
                                        Q_face_flat,
                                        K_ref_flat.transpose(-1, -2),
                                        beta=0,
                                        alpha=module.scale * boost,  # Boost attention scores
                                    )
                                    attn_probs_face = attn_scores_face.softmax(dim=-1)
                                    context_face = torch.bmm(attn_probs_face, V_ref_flat)  # (B_cond*h, L_face, d)
                                    context_face = context_face.view(Bc, h, L_face, d)
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

                        for name, module in self.unet.named_modules():
                            if not isinstance(module, CrossAttention):
                                continue
                            if getattr(module, "is_cross_attention", False):
                                continue                                          # skip X-attn
                
                            # Already patched once? – skip
                            if getattr(module, "_face_patch", False):
                                continue
                
                            # keep the *current* forward as base and tag the module
                            module._base_forward = module.forward
                            module.forward       = types.MethodType(custom_attn_forward, module)
                            module._face_patch   = True
                            module._orig_name    = name  

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
        
        # # ─── Export the mask from the final step, if requested ────────────
        # if export_mask and hasattr(self, "_face_mask") and self._face_mask is not None:
        #     Path(mask_save_dir).mkdir(parents=True, exist_ok=True)
        #     out_path = os.path.join(mask_save_dir, "mask_export2.png")
        #     _save_gray(self._face_mask.astype(np.uint8) * 255, out_path)

        # ------------------------------------------ export final mask
        if export_mask and hasattr(self, "_face_mask") and self._face_mask is not None:
            # latent grid = H//8×W//8  (1024→128,  768→96 …)
            out_path = os.path.join(mask_save_dir, "mask_export2.png")
            out_wh = (latents.shape[-1], latents.shape[-2])   # (W,H)
            _save_gray((self._face_mask.astype(np.uint8))*255,
                       out_path,
                       size=out_wh)


            print(f"[INFO] exported face mask → {out_path}")

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)