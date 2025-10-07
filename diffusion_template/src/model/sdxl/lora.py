from src.model.sdxl.original import SDXL
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig, set_peft_model_state_dict

from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)

class SDXLLora(SDXL):
    def __init__(self, rank, lora_modules, init_lora_weights, *args, **kwargs):
        self.lora_rank = rank
        self.lora_modules = lora_modules
        self.init_lora_weights = init_lora_weights
        super().__init__(*args, **kwargs)

    def prepare_for_training(self):
        super().prepare_for_training()
        self.unet.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights=self.init_lora_weights,
            target_modules=self.lora_modules,
        )
        self.unet.add_adapter(unet_lora_config)

    def get_state_dict(self):
        return {
            'lora_weights': convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet)),
        }

    def load_state_dict_(self, state_dict):
        lora_state_dict = state_dict['lora_weights']
        lora_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, lora_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys is not None:
               raise ValueError(f"Loading lora weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}")