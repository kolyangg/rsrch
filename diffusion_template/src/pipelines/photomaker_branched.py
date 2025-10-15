import torch
from diffusers import DDIMScheduler
from transformers import CLIPImageProcessor

from src.model.photomaker_branched.pipeline_br import PhotoMakerStableDiffusionXLPipeline


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

        pipeline.pose_adapt_ratio = getattr(unwrapped_model, "pose_adapt_ratio", 0.25)
        pipeline.ca_mixing_for_face = getattr(unwrapped_model, "ca_mixing_for_face", True)
        pipeline.face_embed_strategy = getattr(unwrapped_model, "face_embed_strategy", "face")

        pipeline.tokenizer.add_tokens([pipeline.trigger_word], special_tokens=True)
        pipeline.tokenizer_2.add_tokens([pipeline.trigger_word], special_tokens=True)

        return pipeline
