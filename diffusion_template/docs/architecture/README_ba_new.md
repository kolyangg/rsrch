# Diffusion Template - Branched Attention Integration

## Training command (one-id attn1 config)

```bash
ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error \
PYTHONWARNINGS="ignore::FutureWarning" COMET_DISABLE_AUTO_LOGGING=1 \
COMET_LOGGING_CONSOLE=ERROR CUDA_VISIBLE_DEVICES=0 COMET_API_KEY=XXX \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py \
  --config-name=one_id_br_attn1_local \
  trainer.epoch_len=2 \
  dataloaders.train.batch_size=2 \
  dataloaders.train.num_workers=12 \
  model.rank=16 \
  validation_args.num_images_per_prompt=1 \
  lr_scheduler.warmup_steps=200 \
  writer=cometml_local writer.run_name=photomaker_bf16 \
  model.weight_dtype=bf16 \
  datasets.val.one_id_val.limit=4
```

## How branched-attention changes are labeled

- In files adapted from non-branched originals, branched edits are grouped with comments like `##### BRANCHED ATTENTION - ... #####`.

## Key files and roles

- [`src/model/photomaker_branched/_old3/attn_processor_clean.py`](../../src/model/photomaker_branched/_old3/attn_processor_clean.py) - historical core branched self-attention and cross-attention processor described in this document. It splits doubled batch into noise/ref halves, applies face/background masking, and merges with optional ID-feature mixing.
  ```python
  class BranchedAttnProcessor(nn.Module):
      self.pose_adapt_ratio: float = 0.0
      self.ca_mixing_for_face: bool = True
      self.use_id_embeds: bool = True

  face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden
  if USE_ID_EMBEDS:
      id_features = self.id_to_hidden(self.id_embeds)
      id_alpha = getattr(self, "id_alpha", 0.3)
      face_hidden_mixed = face_hidden_mixed * (1 - id_alpha) + id_features * id_alpha
  ```

- [`src/pipelines/photomaker_branched_clean.py`](../../src/pipelines/photomaker_branched_clean.py) - branched-enabled PhotoMaker pipeline wrapper. This is the main inference/validation integration point; BA changes are explicitly tagged with `##### BRANCHED ATTENTION - ... #####` blocks.
  ```python
  ##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
  from src.pipelines.br_pipeline_helpers import (
      build_pipeline_from_pretrained as build_pipeline_from_pretrained_helper,
      run_branched_setup as run_branched_setup_helper,
      run_denoising_step as run_denoising_step_helper,
  )

  ##### BRANCHED ATTENTION - BIG BA BLOCK #####
  run_branched_setup_helper(self, ...)

  ##### BRANCHED ATTENTION - BLOCK 2 #####
  noise_pred, add_text_embeds, prev_mode, _, _ = run_denoising_step_helper(self, ...)

  class PhotomakerBranchedPipeline:
      @staticmethod
      def from_pretrained(model, accelerator, *args, **kwargs):
          return build_pipeline_from_pretrained_helper(...)
  ```

- [`src/pipelines/br_pipeline_helpers.py`](../../src/pipelines/br_pipeline_helpers.py) - helper layer for `photomaker_branched_clean.py`; centralizes setup, per-step branched execution, cleanup, and builder wiring so the pipeline file stays readable.
  ```python
  def run_branched_step(...):
      mask4 = prepare_mask4(pipeline, latent_model_input, suffix="")
      mask4_ref = prepare_mask4(pipeline, latent_model_input, suffix="_ref")
      noise_pred, noise_face, _ = two_branch_predict(
          pipeline,
          latent_model_input,
          mask4=mask4,
          mask4_ref=mask4_ref,
          reference_latents=pipeline._ref_latents_all,
          ...,
      )

  def build_pipeline_from_pretrained(...):
      pipeline = pipeline_cls.from_pretrained(..., unet=unwrapped_model.unet, vae=unwrapped_model.vae, ...)
      pipeline.id_encoder = unwrapped_model.id_encoder
      pipeline.face_embed_strategy = face_embed_strategy_cfg
      pipeline.use_id_embeds = bool(use_id_embeds_cfg)
      pipeline.id_alpha = float(id_alpha_cfg)
  ```

- [`src/model/photomaker_branched/lora2.py`](../../src/model/photomaker_branched/lora2.py) - branched training model (PhotoMaker v2 + LoRA + branched runtime hooks). BA edits vs original are marked by `##### BRANCHED ATTENTION - ... #####` blocks.
  ```python
  ##### BRANCHED ATTENTION - ADDITIONAL IMPORTS #####
  from .lora2_helpers import (
      install_branched_processors_for_training,
      prepare_branched_training_inputs,
      run_branched_forward_pass,
  )

  ##### BRANCHED ATTENTION - NEW PARAMS 3 #####
  self.pose_adapt_ratio = float(pose_adapt_ratio)
  self.face_embed_strategy = (face_embed_strategy or "face").lower()
  self.id_alpha = float(id_alpha)
  self.use_id_embeds = bool(use_id_embeds)

  ##### BRANCHED ATTENTION - NEW BLOCK 1 #####
  install_branched_processors_for_training(self)

  ##### BRANCHED ATTENTION - FORWARD PASS #####
  noise_pred = run_branched_forward_pass(self, ...)
  ```

- [`src/model/photomaker_branched/lora2_helpers.py`](../../src/model/photomaker_branched/lora2_helpers.py) - training-side helper functions used by `lora2.py` for processor installation, batch preparation (prompts/masks/ref latents/id features), branched forward call, and post-eval re-patching.
  ```python
  from .branched_runtime import patch_unet_attention_processors, two_branch_predict

  def install_branched_processors_for_training(model):
      patch_unet_attention_processors(pipeline=model, mask=zero_ctx, mask_ref=zero_ctx, ...)

  def run_branched_forward_pass(model, ...):
      noise_pred, _, _ = two_branch_predict(pipeline=model, latent_model_input=noisy_latents, ...)
      return noise_pred
  ```

- [`src/model/photomaker_branched/branched_runtime.py`](../../src/model/photomaker_branched/branched_runtime.py) - runtime orchestrator for branched attention: patches UNet attention processors, prepares face-branch conditioning, executes the two-branch denoising forward, and restores original processors.
  ```python
  def patch_unet_attention_processors(pipeline, mask, mask_ref, scale=1.0, id_embeds=None, class_tokens_mask=None):
      use_attn_v2 = bool(getattr(pipeline, "use_attn_v2", False))
      if use_attn_v2:
          from ._old2.attn_processor2 import BranchedAttnProcessor, BranchedCrossAttnProcessor
      else:
          from .attn_processor_clean import BranchedAttnProcessor, BranchedCrossAttnProcessor
      ...
      pipeline.unet.set_attn_processor(new_procs)

  def two_branch_predict(...):
      batched_latents = torch.cat([latent_model_input, ref_noised], dim=0)
      patch_unet_attention_processors(pipeline, mask4, mask4_ref, ...)
      encoder_hidden_states = torch.cat([prompt_embeds, face_prompt_embeds], dim=0)
      noise_pred = pipeline.unet(batched_latents, t_batched, encoder_hidden_states=encoder_hidden_states, ...)[0]
  ```

- [`src/model/photomaker_branched/branch_helpers.py`](../../src/model/photomaker_branched/branch_helpers.py) - small mask utility module; currently provides `prepare_mask4(...)` to normalize/resize face masks to current latent resolution for branched steps.


These files together define how branched attention is integrated into both training and inference: processor-level branching (`attn_processor_clean.py`), runtime patch/forward orchestration (`branched_runtime.py`), pipeline wiring (`photomaker_branched_clean.py` + `br_pipeline_helpers.py`), and training wiring (`lora2.py` + `lora2_helpers.py`).
