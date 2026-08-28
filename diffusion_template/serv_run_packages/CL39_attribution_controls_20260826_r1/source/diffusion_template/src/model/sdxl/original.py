import torch
from torch import nn
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from src.utils.model_utils import import_model_class_from_model_name_or_path


class SDXL(nn.Module):
    def __init__(self, pretrained_model_name_or_path, weight_dtype, device, target_size=1024):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if weight_dtype == 'fp16':
            self.weight_dtype = torch.float16
        elif weight_dtype == 'fp32':
            self.weight_dtype = torch.float32
        elif weight_dtype == 'bf16':
            self.weight_dtype = torch.bfloat16
        self.target_size = target_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )

        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None, subfolder="text_encoder_2"
        )

        self.text_encoder = text_encoder_cls_one.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.text_encoder_2 = text_encoder_cls_two.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder_2"
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="unet",
        )
        
    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(True)
        

        self.unet.to(dtype=self.weight_dtype)
        self.vae.to(dtype=self.weight_dtype)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.text_encoder_2.to(dtype=self.weight_dtype)

    def get_trainable_params(self, config):
        unet_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {'params': unet_params, 'lr': config.lr_for_unet, 'name': 'unet_params'},
        ]
        return trainable_params

    def get_state_dict(self):
        unet_weights = self.unet.state_dict()
        return {
            'unet_weights': unet_weights,
        }

    def load_state_dict_(self, state_dict):
        self.unet.load_state_dict(state_dict['unet_weights'])

    def forward(self, pixel_values, prompt, original_sizes, crop_top_lefts, do_cfg=False, *args, **kwargs):
        pixel_values = pixel_values.to(self.device, self.vae.dtype)
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device 
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(s, c) for s, c in zip(original_sizes, crop_top_lefts)]
        )

        #*********************************************************
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            do_cfg=do_cfg
        )

        #*********************************************************
        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1).to(dtype=self.unet.dtype)
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds = add_text_embeds.to(self.device, dtype=self.unet.dtype)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        target = noise
                
        return {
            'model_pred': model_pred,
            'target': target,
        }

    def compute_time_ids(self, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = [self.target_size, self.target_size]
        add_time_ids = list(original_size) + list(crops_coords_top_left) + target_size
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.weight_dtype)
        return add_time_ids
        
    def encode_prompt(self, prompt, do_cfg=False):
        prompt = prompt if not do_cfg else [""] * len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), 
            output_hidden_states=True, 
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids_2.to(self.text_encoder_2.device), 
            output_hidden_states=True, 
        )

        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]


        prompt_embeds = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)
        bs_embed = prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds_2.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds