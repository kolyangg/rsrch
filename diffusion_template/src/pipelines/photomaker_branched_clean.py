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

##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
"""Import compact BA helper entrypoints used by pipeline runtime and wrapper methods."""
from src.pipelines.br_pipeline_helpers import (
    build_pipeline_from_pretrained as build_pipeline_from_pretrained_helper,
    cleanup_branched_runtime as cleanup_branched_runtime_helper,
    ensure_face_analyzer as ensure_face_analyzer_helper,
    ensure_id_embeds as ensure_id_embeds_helper,
    ensure_ref_latents_ready as ensure_ref_latents_ready_helper,
    prepare_gen_mask as prepare_gen_mask_helper,
    prepare_id_features as prepare_id_features_helper,
    prepare_ref_latents as prepare_ref_latents_helper,
    prepare_ref_mask as prepare_ref_mask_helper,
    run_branched_setup as run_branched_setup_helper,
    run_branched_step as run_branched_step_helper,
    run_denoising_step as run_denoising_step_helper,
    save_step_previews as save_step_previews_helper,
    select_mode_and_prompts as select_mode_and_prompts_helper,
)

OLD_HEATMAP_MASKING = False

##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####


import PIL

import torch
from transformers import CLIPImageProcessor

from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
# --- ADDED For training integration (FOLDER STUCTURE) ---
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


from src.model.photomaker_branched.model import PhotoMakerIDEncoder  # PhotoMaker v1
from src.model.photomaker_branched.model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken  # PhotoMaker v2

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
                #### 08 MAR - FIX BATCHED VALIDATION ####
                clean_input_ids_batch = []
                class_tokens_mask_batch = []
                for row_ids in text_input_ids.tolist():
                    clean_index = 0
                    clean_input_ids = []
                    class_token_index = []
                    # Find out the corresponding class word token based on the newly added trigger word token
                    for token_id in row_ids:
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
                        clean_input_ids[class_token_index + 1:]

                    # Truncation or padding
                    max_len = tokenizer.model_max_length
                    if len(clean_input_ids) > max_len:
                        clean_input_ids = clean_input_ids[:max_len]
                    else:
                        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (max_len - len(clean_input_ids))

                    class_tokens_mask = [
                        True if class_token_index <= i < class_token_index + (num_id_images * self.num_tokens) else False
                        for i in range(len(clean_input_ids))
                    ]
                    clean_input_ids_batch.append(clean_input_ids)
                    class_tokens_mask_batch.append(class_tokens_mask)

                clean_input_ids = torch.tensor(clean_input_ids_batch, dtype=torch.long)
                class_tokens_mask = torch.tensor(class_tokens_mask_batch, dtype=torch.bool)
                #### 08 MAR - FIX BATCHED VALIDATION ####

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
        # Diagnostic-only spatial reference override. PhotoMaker ID prompt
        # conditioning still comes exclusively from input_id_images.
        ppr_reference_image: PipelineImageInput = None,
        ppr_face_bbox_ref: Optional[List[float]] = None,
        # start_merge_step kept for back-compat. If provided, it will populate both new knobs.
        start_merge_step: int = 10, # TODO: change to `style_strength_ratio` in the future
        # NEW: split the semantics
        class_tokens_mask: Optional[torch.LongTensor] = None,
        id_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,

        ##### BRANCHED ATTENTION - ADDITIONAL SWITCHES #####
        photomaker_start_step: int = 10,
        merge_start_step: int = 10,
        use_branched_attention: bool = False,
        photomaker_scale: float = 1.0,  # Add scale parameter for attention
        branched_attn_start_step: int = 10,
        branched_attn_end_step: Optional[int] = None,
        face_embed_strategy: str = "face", # "face", #  "face" or "id_embeds"
        use_bbox_mask_ref: bool = False, # BBox-driven masking toggles (validation convenience)
        use_bbox_mask_gen: bool = False, # BBox-driven masking toggles (validation convenience)
        face_bbox_ref: Optional[List[float]] = None, # Optional per-sample face boxes (x0,y0,x1,y1) in pixel space
        face_bbox_gen: Optional[List[float]] = None, # Optional per-sample face boxes (x0,y0,x1,y1) in pixel space
        mask_expansion_ratio: float = 1.0,
        mask_softness: float = 0.0,
        import_mask: Optional[str] = "../compare/testing/ref2_masks/keanu_gen_mask.png",        
        import_mask_ref: Optional[str] = None, # to debug auto_mask_ref

               
        auto_mask_ref: bool = True,
        use_dynamic_mask: bool = True, # generation mask

        # C6 gen-bbox re-tracking: instead of freezing the generation face box from the
        # pre-branch PhotoMaker preview, re-detect it on the *branched* trajectory (decode the
        # in-loop latents, run the face detector, rebuild the gen mask) so the mask follows the
        # face the branched pass actually produces. OFF by default => byte-identical behaviour.
        # See debug_planning_03Jul/ba_gen_bbox_retrack_04Jul.md.
        gen_bbox_retrack: bool = False,
        gen_bbox_retrack_every: int = 6,        # re-detect every N steps inside the branched window
        gen_bbox_retrack_min_frac: float = 0.5, # only start once >= this fraction of steps (clean enough to detect)
        gen_bbox_retrack_detector: str = "yolo",
        gen_bbox_retrack_model: str = "bbox_utils/yolov8n-face.pt",
        gen_bbox_retrack_conf: float = 0.3,
        gen_bbox_retrack_padding: float = 0.08,
        gen_bbox_retrack_debug_dir: Optional[str] = None,

        debug_dir: Optional[str] = None,
        debug_idx: Optional[int] = None,
        val_debug: bool = True,
        force_par_before_pm: bool = False,
        ##### BRANCHED ATTENTION - ADDITIONAL SWITCHES #####
        
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
        """BRANCHED ATTENTION - ADDITIONAL SWITCHES: runtime knobs for scheduling, masking, and debug behavior."""

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        photomaker_use_lora_adapter = kwargs.pop("photomaker_use_lora_adapter", None)
        if photomaker_use_lora_adapter is not None:
            self.photomaker_use_lora_adapter = bool(photomaker_use_lora_adapter)
        

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
        self._val_debug = bool(val_debug)

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

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        #### 08 MAR - FIX BATCHED VALIDATION ####
        per_prompt_id_images = False
        if isinstance(input_id_images, list) and batch_size > 1 and len(input_id_images) == batch_size:
            per_prompt_id_images = True
            normalized = []
            for refs in input_id_images:
                refs_list = list(refs) if isinstance(refs, (list, tuple)) else [refs]
                if len(refs_list) == 0:
                    raise ValueError("Each prompt must provide at least one reference image.")
                normalized.append(refs_list)
            input_id_images = normalized
            input_id_images_first = [refs[0] for refs in input_id_images]
            num_id_images = 1  # keep validation batched with first ref image per prompt
        else:
            if not isinstance(input_id_images, list):
                input_id_images = [input_id_images]
            input_id_images_first = input_id_images
            num_id_images = len(input_id_images)
        #### 08 MAR - FIX BATCHED VALIDATION ####

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
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
        #### 08 MAR - FIX BATCHED VALIDATION ####
        trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        if isinstance(prompt, list):
            prompt_text_only = []
            for single_prompt in prompt:
                tokens_text_only = self.tokenizer.encode(single_prompt, add_special_tokens=False)
                if trigger_word_token in tokens_text_only:
                    tokens_text_only.remove(trigger_word_token)
                prompt_text_only.append(self.tokenizer.decode(tokens_text_only, add_special_tokens=False))
        else:
            tokens_text_only = self.tokenizer.encode(prompt, add_special_tokens=False)
            if trigger_word_token in tokens_text_only:
                tokens_text_only.remove(trigger_word_token)
            prompt_text_only = self.tokenizer.decode(tokens_text_only, add_special_tokens=False)
        #### 08 MAR - FIX BATCHED VALIDATION ####
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
        #### 08 MAR - FIX BATCHED VALIDATION ####
        if per_prompt_id_images:
            if not isinstance(input_id_images_first[0], torch.Tensor):
                id_pixel_values = self.id_image_processor(input_id_images_first, return_tensors="pt").pixel_values
            else:
                id_pixel_values = torch.stack(
                    [x if x.dim() == 3 else x[0] for x in input_id_images_first],
                    dim=0,
                )
            id_pixel_values = id_pixel_values.unsqueeze(1).to(device=device, dtype=dtype)
            id_images_for_embeds = input_id_images
            input_id_images_for_setup = input_id_images_first
        else:
            if not isinstance(input_id_images[0], torch.Tensor):
                id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values
            else:
                id_pixel_values = torch.stack(
                    [x if x.dim() == 3 else x[0] for x in input_id_images],
                    dim=0,
                )
            id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype)
            id_images_for_embeds = input_id_images
            input_id_images_for_setup = input_id_images
        #### 08 MAR - FIX BATCHED VALIDATION ####
        

        # 7. Get the update text embedding with the stacked ID embedding

        ##### BRANCHED ATTENTION - ALWAYS NEED ID EMBEDS #####
        """Resolve ID embeddings from inputs (or use provided tensor) before PMv2 ID encoder fusion."""
        id_embeds = ensure_id_embeds_helper(
            self,
            id_embeds=id_embeds,
            input_id_images=id_images_for_embeds,
            device=device,
            dtype=dtype,
        )
        prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
        ##### BRANCHED ATTENTION - ALWAYS NEED ID EMBEDS #####
        
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
        
        
        
        ##### BRANCHED ATTENTION - BIG BA BLOCK #####
        """Prepare branched runtime state once (reference latents/masks, strategy cache, and ID features)."""
        self.mask_softness = float(mask_softness)
        self.force_binary_masks = bool(float(mask_softness) <= 0.0)
        # ### 05 APR - FIX VALIDATION REF BATCHING ISSUE ###
        spatial_reference_images = (
            ppr_reference_image
            if ppr_reference_image is not None
            else input_id_images_for_setup
        )
        if ppr_reference_image is not None and not isinstance(
            spatial_reference_images,
            (list, tuple),
        ):
            spatial_reference_images = [spatial_reference_images]
        spatial_reference_bbox = (
            ppr_face_bbox_ref
            if ppr_reference_image is not None
            else face_bbox_ref
        )
        run_branched_setup_helper(
            self,
            use_branched_attention=use_branched_attention,
            input_id_images=spatial_reference_images,
            height=height,
            width=width,
            latents=latents,
            id_pixel_values=id_pixel_values,
            auto_mask_ref=auto_mask_ref,
            use_bbox_mask_ref=use_bbox_mask_ref,
            face_bbox_ref=spatial_reference_bbox,
            mask_expansion_ratio=mask_expansion_ratio,
            mask_softness=mask_softness,
            import_mask_ref=import_mask_ref,
            debug_dir=debug_dir,
            use_dynamic_mask=use_dynamic_mask,
            use_bbox_mask_gen=use_bbox_mask_gen,
            face_bbox_gen=face_bbox_gen,
            generator=generator,
            device=device,
            face_embed_strategy=face_embed_strategy,
            batch_size=batch_size,
            prompt_embeds=prompt_embeds,
            id_embeds=id_embeds,
            class_tokens_mask=class_tokens_mask,
        )
        ##### BRANCHED ATTENTION - BIG BA BLOCK #####

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

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            
            
        ##### BRANCHED ATTENTION - BLOCK 1 #####
        """Pre-loop validation that branched prerequisites (e.g., cached reference latents) are available."""
        ensure_ref_latents_ready_helper(
            self,
            use_branched_attention=use_branched_attention,
            id_pixel_values=id_pixel_values,
        )
        ##### BRANCHED ATTENTION - BLOCK 1 #####
        
        

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
                
                
                ##### BRANCHED ATTENTION - BLOCK 2 #####
                """Per-step branched flow delegated to helper (mode select, UNet/BA forward, preview dumps)."""
                noise_pred, add_text_embeds, prev_mode, _pose_forced_logged, _pose_relaxed_logged = (
                    run_denoising_step_helper(
                        self,
                        i=i,
                        t=t,
                        prev_mode=prev_mode,
                        photomaker_start_step=photomaker_start_step,
                        branched_attn_start_step=branched_attn_start_step,
                        branched_attn_end_step=branched_attn_end_step,
                        prompt_embeds_text_only=prompt_embeds_text_only,
                        pooled_prompt_embeds_text_only=pooled_prompt_embeds_text_only,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        force_par_before_pm=force_par_before_pm,
                        pose_forced_logged=_pose_forced_logged,
                        pose_relaxed_logged=_pose_relaxed_logged,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        add_time_ids=add_time_ids,
                        ip_adapter_image=ip_adapter_image,
                        ip_adapter_image_embeds=ip_adapter_image_embeds,
                        image_embeds=image_embeds,
                        use_branched_attention=use_branched_attention,
                        latent_model_input=latent_model_input,
                        timestep_cond=timestep_cond,
                        class_tokens_mask=class_tokens_mask,
                        photomaker_scale=photomaker_scale,
                        merge_start_step=merge_start_step,
                        debug_dir=debug_dir,
                        latents=latents,
                        extra_step_kwargs=extra_step_kwargs,
                        num_inference_steps=num_inference_steps,
                    )
                )
                ##### BRANCHED ATTENTION - BLOCK 2 #####

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                _sched_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents = _sched_out.prev_sample
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # C6: re-track the generation face bbox on the branched trajectory (toggle).
                if (
                    use_branched_attention
                    and gen_bbox_retrack
                    and use_bbox_mask_gen
                    and (not use_dynamic_mask)
                    and gen_bbox_retrack_every > 0
                    and i >= branched_attn_start_step
                    and i >= int(gen_bbox_retrack_min_frac * len(timesteps))
                    and (i % gen_bbox_retrack_every == 0)
                    and i < len(timesteps) - 1
                ):
                    # Detect on the model's x0 estimate (recognizable mid-diffusion), not the
                    # still-noisy running latent; fall back to the latent if unavailable.
                    _x0 = getattr(_sched_out, "pred_original_sample", None)
                    self._retrack_gen_bbox(
                        latents=(_x0 if _x0 is not None else latents),
                        step=i,
                        mask_expansion_ratio=mask_expansion_ratio,
                        mask_softness=mask_softness,
                        height=height,
                        width=width,
                        detector_backend=gen_bbox_retrack_detector,
                        detector_model=gen_bbox_retrack_model,
                        conf=gen_bbox_retrack_conf,
                        padding=gen_bbox_retrack_padding,
                        debug_dir=gen_bbox_retrack_debug_dir,
                    )

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
        
        ##### BRANCHED ATTENTION - BLOCK 3 #####
        """Post-loop cleanup of temporary branched state kept on the pipeline instance."""
        cleanup_branched_runtime_helper(self, use_branched_attention=use_branched_attention)
        ##### BRANCHED ATTENTION - BLOCK 3 #####

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


        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
    
    
    ##### BRANCHED ATTENTION - BLOCK 4 #####
    """Helper API used by branched inference (state prep, mode selection, branched forward pass, debug previews)."""
    @property
    def cross_attention_kwargs(self):
        """Get cross attention kwargs if they exist."""
        return getattr(self, '_cross_attention_kwargs', None)

    @cross_attention_kwargs.setter
    def cross_attention_kwargs(self, value):
        """Set cross attention kwargs."""
        self._cross_attention_kwargs = value
    
    def _ensure_face_analyzer(self):
        ensure_face_analyzer_helper(self)
    
    def _prepare_ref_latents(
        self,
        *,
        pil: PIL.Image.Image,
        height: int,
        width: int,
        latents_dtype: torch.dtype,
    ) -> torch.Tensor:
        return prepare_ref_latents_helper(
            self,
            pil=pil,
            height=height,
            width=width,
            latents_dtype=latents_dtype,
        )

    def _prepare_ref_mask(
        self,
        *,
        pil: PIL.Image.Image,
        auto_mask_ref: bool,
        use_bbox_mask_ref: bool,
        face_bbox_ref: Optional[List[float]],
        mask_expansion_ratio: float,
        mask_softness: float,
        import_mask_ref: Optional[str],
        debug_dir: Optional[str],
        height: int,
        width: int,
    ) -> Optional[str]:
        return prepare_ref_mask_helper(
            self,
            pil=pil,
            auto_mask_ref=auto_mask_ref,
            use_bbox_mask_ref=use_bbox_mask_ref,
            face_bbox_ref=face_bbox_ref,
            mask_expansion_ratio=mask_expansion_ratio,
            mask_softness=mask_softness,
            import_mask_ref=import_mask_ref,
            debug_dir=debug_dir,
            height=height,
            width=width,
        )

    def _prepare_gen_mask(
        self,
        *,
        use_dynamic_mask: bool,
        use_bbox_mask_gen: bool,
        face_bbox_gen: Optional[List[float]],
        mask_expansion_ratio: float,
        mask_softness: float,
        height: int,
        width: int,
    ) -> None:
        prepare_gen_mask_helper(
            self,
            use_dynamic_mask=use_dynamic_mask,
            use_bbox_mask_gen=use_bbox_mask_gen,
            face_bbox_gen=face_bbox_gen,
            mask_expansion_ratio=mask_expansion_ratio,
            mask_softness=mask_softness,
            height=height,
            width=width,
        )

    @torch.no_grad()
    def _decode_latents_to_pil(self, latents: torch.Tensor) -> List[PIL.Image.Image]:
        """Decode in-loop latents to PIL WITHOUT disturbing the denoising state (clone).
        Mirrors the final VAE-decode block; used only by C6 gen-bbox re-tracking."""
        vae = self.vae
        lat = latents.detach().clone()
        needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()
        lat = lat.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        has_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
        has_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
        if has_mean and has_std:
            lm = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(lat.device, lat.dtype)
            ls = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(lat.device, lat.dtype)
            lat = lat * ls / vae.config.scaling_factor + lm
        else:
            lat = lat / vae.config.scaling_factor
        image = vae.decode(lat, return_dict=False)[0]
        if needs_upcasting:
            vae.to(dtype=torch.float16)
        pil = self.image_processor.postprocess(image, output_type="pil")
        if not isinstance(pil, list):
            pil = [pil]
        return pil

    @torch.no_grad()
    def _retrack_gen_bbox(
        self,
        *,
        latents: torch.Tensor,
        step: int,
        mask_expansion_ratio: float,
        mask_softness: float,
        height: int,
        width: int,
        detector_backend: str = "yolo",
        detector_model: str = "bbox_utils/yolov8n-face.pt",
        conf: float = 0.3,
        padding: float = 0.08,
        debug_dir: Optional[str] = None,
    ) -> None:
        """C6: decode the current branched latents, re-detect the face box(es), and rebuild the
        generation mask in place so the merge follows the branched face (not the frozen preview)."""
        from bbox_utils.generate_bboxes import detect_face_box, clamp_bbox

        # Lazy-init the detector on CPU to avoid extra VRAM during the branched pass.
        if getattr(self, "_retrack_detector", None) is None:
            from bbox_utils.generate_bboxes import load_face_detector
            self._retrack_detector, self._retrack_backend = load_face_detector(
                backend=detector_backend, model_name=detector_model, device="cpu"
            )

        pil_list = self._decode_latents_to_pil(latents)
        B = len(pil_list)

        # Per-sample fallback boxes (keep the old box where detection fails).
        cur = getattr(self, "_face_bbox_gen_original", None)
        if cur is None:
            base_boxes = [None] * B
        elif isinstance(cur, (list, tuple)) and len(cur) > 0 and isinstance(cur[0], (list, tuple)):
            base_boxes = [list(b) for b in cur]
            if len(base_boxes) < B:
                base_boxes += [base_boxes[-1]] * (B - len(base_boxes))
        else:
            base_boxes = [list(cur) for _ in range(B)]

        new_boxes, n_hit = [], 0
        for bi, pil in enumerate(pil_list):
            box = detect_face_box(self._retrack_detector, self._retrack_backend, None, pil, conf, "cpu")
            if box is not None:
                box = [float(x) for x in clamp_bbox(box, pil.size[0], pil.size[1])]
                new_boxes.append(box)
                n_hit += 1
            else:
                new_boxes.append(base_boxes[bi])
            if debug_dir and new_boxes[bi] is not None:
                try:
                    from pathlib import Path as _P
                    from bbox_utils.visualize_bboxes import save_annotated_pil
                    _P(debug_dir).mkdir(parents=True, exist_ok=True)
                    save_annotated_pil(
                        pil, {"face_crop_new": new_boxes[bi]},
                        _P(debug_dir) / f"retrack_s{step:02d}_b{bi}.png", line_width=4,
                    )
                except Exception:
                    pass

        if n_hit == 0 or any(b is None for b in new_boxes):
            return  # nothing reliable to update -> keep the existing mask

        prepare_gen_mask_helper(
            self,
            use_dynamic_mask=False,
            use_bbox_mask_gen=True,
            face_bbox_gen=(new_boxes if B > 1 else new_boxes[0]),
            mask_expansion_ratio=mask_expansion_ratio,
            mask_softness=mask_softness,
            height=height,
            width=width,
            batch_size=B,
        )
        if not getattr(self, "_ba_retrack_logged", False):
            print(f"[C6 retrack] step {step}: updated {n_hit}/{B} gen bbox(es) -> {new_boxes[0]}")
            self._ba_retrack_logged = True

    def _prepare_id_features(
        self,
        *,
        id_pixel_values: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        id_embeds: Optional[torch.Tensor],
        class_tokens_mask: torch.LongTensor,
    ) -> None:
        prepare_id_features_helper(
            self,
            id_pixel_values=id_pixel_values,
            prompt_embeds=prompt_embeds,
            id_embeds=id_embeds,
            class_tokens_mask=class_tokens_mask,
        )

    def _select_mode_and_prompts(
        self,
        *,
        i: int,
        photomaker_start_step: int,
        branched_attn_start_step: int,
        branched_attn_end_step: Optional[int],
        prompt_embeds_text_only: torch.Tensor,
        pooled_prompt_embeds_text_only: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        force_par_before_pm: bool,
        pose_forced_logged: bool,
        pose_relaxed_logged: bool,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, bool, bool]:
        return select_mode_and_prompts_helper(
            self,
            i=i,
            photomaker_start_step=photomaker_start_step,
            branched_attn_start_step=branched_attn_start_step,
            branched_attn_end_step=branched_attn_end_step,
            prompt_embeds_text_only=prompt_embeds_text_only,
            pooled_prompt_embeds_text_only=pooled_prompt_embeds_text_only,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            force_par_before_pm=force_par_before_pm,
            pose_forced_logged=pose_forced_logged,
            pose_relaxed_logged=pose_relaxed_logged,
        )

    def _run_branched_step(
        self,
        *,
        i: int,
        t: torch.Tensor,
        mode: str,
        latent_model_input: torch.Tensor,
        current_prompt_embeds: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        class_tokens_mask: Optional[torch.LongTensor],
        timestep_cond: Optional[torch.Tensor],
        photomaker_scale: float,
        merge_start_step: int,
        photomaker_start_step: int,
        branched_attn_start_step: int,
        debug_dir: Optional[str],
        num_outputs: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        return run_branched_step_helper(
            self,
            i=i,
            t=t,
            mode=mode,
            latent_model_input=latent_model_input,
            current_prompt_embeds=current_prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            class_tokens_mask=class_tokens_mask,
            timestep_cond=timestep_cond,
            photomaker_scale=photomaker_scale,
            merge_start_step=merge_start_step,
            photomaker_start_step=photomaker_start_step,
            branched_attn_start_step=branched_attn_start_step,
            debug_dir=debug_dir,
            num_outputs=num_outputs,
        )

    def _save_step_previews(
        self,
        *,
        i: int,
        t: torch.Tensor,
        num_inference_steps: int,
        debug_dir: Optional[str],
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        mask4: Optional[torch.Tensor],
        noise_face: Optional[torch.Tensor],
        extra_step_kwargs: Dict[str, Any],
    ) -> None:
        save_step_previews_helper(
            self,
            i=i,
            t=t,
            num_inference_steps=num_inference_steps,
            debug_dir=debug_dir,
            latents=latents,
            noise_pred=noise_pred,
            mask4=mask4,
            noise_face=noise_face,
            extra_step_kwargs=extra_step_kwargs,
        )
    ##### BRANCHED ATTENTION - BLOCK 4 #####
    

##### BRANCHED ATTENTION - BLOCK 5 #####
"""Training-facing factory that builds the pipeline and wires branched-attention runtime knobs."""
class PhotomakerBranchedPipeline:
    @staticmethod
    def from_pretrained(model, accelerator, *args, **kwargs):
        """Build the training pipeline via helper so BA config wiring stays out of this file."""
        return build_pipeline_from_pretrained_helper(
            PhotoMakerStableDiffusionXLPipeline,
            model=model,
            accelerator=accelerator,
            args=args,
            kwargs=kwargs,
        )
##### BRANCHED ATTENTION - BLOCK 5 #####
