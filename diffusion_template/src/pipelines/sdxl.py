import torch
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers import DDIMScheduler

class SDXLPipeline:
    def from_pretrained(model, accelerator, *args, **kwargs):
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = getattr(torch,  kwargs["torch_dtype"])
        unwraped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        scheduler = DDIMScheduler.from_pretrained(
            kwargs["pretrained_model_name_or_path"], 
            subfolder="scheduler"
        )
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            scheduler=scheduler,
            tokenizer=unwraped_model.tokenizer,
            tokenizer_2=unwraped_model.tokenizer_2,
            text_encoder=unwraped_model.text_encoder,
            text_encoder_2=unwraped_model.text_encoder_2,
            unet=unwraped_model.unet,
            vae=unwraped_model.vae,
            *args, 
            **kwargs
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline
        