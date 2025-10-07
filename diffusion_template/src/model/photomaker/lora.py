from src.model.sdxl.original import SDXL
import torch

from typing import  Optional

from peft.utils import get_peft_model_state_dict
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPImageProcessor
from src.model.photomaker.id_encoder import PhotoMakerIDEncoder

from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)



class PhotomakerLora(SDXL):
    def __init__(self, pretrained_model_name_or_path, photomaker_path, rank, weight_dtype, device, init_lora_weights, lora_modules, target_size=1024, trigger_word="img"):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_dtype=weight_dtype,
            device=device
        )
        self.lora_rank = rank
        self.init_lora_weights = init_lora_weights
        self.lora_modules = lora_modules
        
        self.id_image_processor = CLIPImageProcessor()
        self.id_encoder = PhotoMakerIDEncoder()

        self.trigger_word = "img"
        self.num_tokens = 1
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

        photomaker_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(photomaker_lora_config)

        photomaker_state_dict = torch.load(photomaker_path)
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

    def load_photomaker_state_dict_(self, state_dict):
        # load lora
        lora_state_dict = state_dict['lora_weights']
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert not unexpected_keys, unexpected_keys

        # load id_encoder
        self.id_encoder.load_state_dict(state_dict['id_encoder'], strict=True)
        
    def get_trainable_params(self, config):
        lora_params = filter(lambda p: p.requires_grad, self.unet.parameters())
        trainable_params = [
            {'params': lora_params, 'lr': config.lr_for_lora, 'name': 'lora_params'},
        ]
        return trainable_params

    def get_state_dict(self):
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet, adapter_name="lora_adapter"))
        return {
            'lora_weights': lora_weights,
        }

    def load_state_dict_(self, state_dict):
        lora_state_dict = state_dict['lora_weights']
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="lora_adapter")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            assert unexpected_keys is None, unexpected_keys

    def forward(self, pixel_values, prompts, ref_images, original_sizes, crop_top_lefts, face_bbox, do_cfg=False, *args, **kwargs):
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device, self.vae.dtype)
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
            
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
    
            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device  # num_train_timesteps ???
            )
            timesteps = timesteps.long()
    
            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
    
            add_time_ids = torch.cat(
                [self.compute_time_ids(s, c) for s, c in zip(original_sizes, crop_top_lefts)]
            )
        
            #*********************************************************
            prompt_embeds_list, pooled_prompt_embeds_list, class_tokens_mask_list = [], [], []
            for prompt, refs in zip(prompts, ref_images):
                prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.encode_prompt_with_trigger_word(
                    prompt=prompt,
                    num_id_images=len(refs),
                    do_cfg=do_cfg,
                )
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
                class_tokens_mask_list.append(class_tokens_mask)
                #*********************************************************
                if not do_cfg:
                    id_pixel_values = self.id_image_processor(refs, return_tensors="pt").pixel_values.unsqueeze(0)
                    id_pixel_values = id_pixel_values.to(self.device, dtype=self.id_encoder.dtype)
                    prompt_embeds = prompt_embeds.to(dtype=self.id_encoder.dtype)
                    prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
                prompt_embeds_list.append(prompt_embeds)
        
            prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
            pooled_prompt_embeds = torch.concat(pooled_prompt_embeds_list, dim=0)
        
            #*********************************************************
            
        bs_embed, seq_len, _ = prompt_embeds.shape
        assert seq_len == 77 

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
                # untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                #     text_input_ids, untruncated_ids
                # ):
                #     removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

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
                    
                    text_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                    class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)
                    class_tokens_mask = class_tokens_mask.to(self.device)                    

                prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
            
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(self.device)
        
        bs_embed, _, _ = prompt_embeds.shape
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    
        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask
          


        