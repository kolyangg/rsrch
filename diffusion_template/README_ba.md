# Diffusion Template – One‑ID Branched Attention Training

## Training command (one‑ID attn1 config)

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=XXX \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py \
  --config-name=one_id_br_attn1_local \
  trainer.epoch_len=200 \
  dataloaders.train.batch_size=1 \
  dataloaders.train.num_workers=12 \
  model.rank=16 \
  validation_args.num_images_per_prompt=1 \
  lr_scheduler.warmup_steps=200 \
  writer=console writer.run_name=photomaker_bf16 model.weight_dtype=bf16
```

## Key files and roles

- [`train.py`](train.py) – entrypoint, Hydra setup, trainer wiring  
  ```python
  @hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
  def main(config):
      ...
      model = instantiate(config.model, device=device, **ba_kwargs)
      model.prepare_for_training()
      trainable_params = model.get_trainable_params(config)
      optimizer = instantiate(config.optimizer, params=trainable_params)
      ...
      trainer = instantiate(config.trainer, model=model, pipe=pipeline, ...)
      trainer.train()
  ```

- [`src/configs/one_id_br_attn1_local.yaml`](src/configs/one_id_br_attn1_local.yaml) – local one‑ID config (branched attn v1/v2 switch, attn1 focus)  
  ```yaml
  defaults:
    - trainer: photomaker_lora
    - model: photomaker_branched_lora2
    - pipeline: photomaker_branched2_ref1
    - metrics: all_metrics_oneid
    - datasets: all_datasets_local
    - dataloaders: all_dataloaders
  train_ba_only: true
  disable_branched_sa: false
  disable_branched_ca: false
  model:
    pretrained_model_name_or_path: SG161222/RealVisXL_V4.0
    id_alpha: 0.3        # ID injection strength (BranchedAttnProcessor)
    use_attn_v2: false   # attn1 configs use legacy `attn_processor.py`
    use_id_embeds: false # disable BranchedAttnProcessor.id_to_hidden (ID embedding branch)
  ```

- [`src/configs/model/photomaker_branched_lora2.yaml`](src/configs/model/photomaker_branched_lora2.yaml) – model type and LoRA settings  
  ```yaml
  _target_: src.model.photomaker_branched.lora2.PhotomakerBranchedLora
  pretrained_model_name_or_path: stabilityai/stable-diffusion-xl-base-1.0
  photomaker_path: ${oc.env:HOME}/.cache/.../photomaker-v2.bin
  rank: 16
  photomaker_lora_rank: 64
  lora_modules: [to_q, to_k, to_v, to_out.0]
  ```

- [`src/model/photomaker_branched/lora2.py`](src/model/photomaker_branched/lora2.py) – main training model (PhotoMaker + branched LoRA)  
  ```python
  class PhotomakerBranchedLora(SDXL):
      def __init__(..., pose_adapt_ratio=0.25, ca_mixing_for_face=True,
                   train_branch_mode="both", train_ba_only=False, ba_weights_split=False,
                   use_attn_v2=True, id_alpha: float = 0.3):
          ...
          self.pose_adapt_ratio = float(pose_adapt_ratio)
          self.ca_mixing_for_face = bool(ca_mixing_for_face)
          self.train_branch_mode = (train_branch_mode or "both").lower()
          self.train_ba_only = bool(train_ba_only)
          self.ba_weights_split = bool(ba_weights_split)
          # ID embedding mixing strength for BranchedAttnProcessor
          self.id_alpha = float(id_alpha)
      def prepare_for_training(self):
          ...
          patch_unet_attention_processors(pipeline=self, mask=zero_ctx, mask_ref=zero_ctx, ...)
      def get_trainable_params(self, config):
          # returns branched_processors + LoRA param groups when train_ba_only is True
  ```

- [`src/model/photomaker_branched/branched_new2.py`](src/model/photomaker_branched/branched_new2.py) – training‑time patcher that installs branched attention on the UNet  
  ```python
  def patch_unet_attention_processors(pipeline, mask, mask_ref, scale=1.0,
                                      id_embeds=None, class_tokens_mask=None):
      disable_sa = bool(getattr(pipeline, "disable_branched_sa", False))
      disable_ca = bool(getattr(pipeline, "disable_branched_ca", False))
      # Default to legacy processors unless use_attn_v2 is true
      use_attn_v2 = bool(getattr(pipeline, "use_attn_v2", False))
      if use_attn_v2:
          from ._old2.attn_processor2 import BranchedAttnProcessor, BranchedCrossAttnProcessor
      else:
          from .attn_processor import BranchedAttnProcessor, BranchedCrossAttnProcessor
      ...
      for name in pipeline.unet.attn_processors.keys():
          if name.endswith("attn1.processor"):
              proc = BranchedAttnProcessor(...)
          elif name.endswith("attn2.processor"):
              proc = BranchedCrossAttnProcessor(...)
      pipeline.unet.set_attn_processor(new_procs)
  ```
  For all current `*_attn1*` configs (`use_attn_v2: false`), this patcher is active in both training and validation but always resolves to **`attn_processor.py`**; `_old2/attn_processor2.py` is only used if you explicitly enable `use_attn_v2: true`.

- [`src/model/photomaker_branched/attn_processor.py`](src/model/photomaker_branched/attn_processor.py) – branched self‑/cross‑attention actually used for attn1 configs (`use_attn_v2: false`)  
  ```python
  class BranchedAttnProcessor(nn.Module):
      def __init__(..., scale=1.0, ...):
          self.id_embeds = None
          self.id_to_hidden = None
          self.use_id_embeds: bool = True  # can be overridden from config
      def __call__(..., hidden_states, encoder_hidden_states=None, ...):
          # splits [noise, reference] batch and applies background + face branches
          use_id_flag = getattr(self, "use_id_embeds", True)
          USE_ID_EMBEDS = bool(use_id_flag) and (self.id_embeds is not None)
          if USE_ID_EMBEDS:
              if self.id_to_hidden is None:
                  self.id_to_hidden = nn.Linear(
                      self.id_embeds.shape[-1],
                      face_hidden_mixed.shape[-1],
                      bias=False,
                  ).to(face_hidden_mixed.device, face_hidden_mixed.dtype)
              id_features = self.id_to_hidden(self.id_embeds)
              id_alpha = getattr(self, "id_alpha", 0.3)
              face_hidden_mixed = face_hidden_mixed * (1 - id_alpha) + id_features * id_alpha
  ```

  For the current attn1 configs (`use_id_embeds: false`), `USE_ID_EMBEDS` is always false, so `id_to_hidden` is never created or trained. The only trainable parts are the LoRA adapters on the UNet’s attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`) grouped under `"branched_lora"` in `PhotomakerBranchedLora.get_trainable_params`.

- [`src/model/photomaker_branched/_old2/attn_processor2.py`](src/model/photomaker_branched/_old2/attn_processor2.py) – alternate v2 processors (not used when `use_attn_v2: false`)  
  They pre‑register `id_to_hidden` and optional branch‑specific adapters in `__init__` and are selected only if you set `use_attn_v2: true` in the config.

- [`src/configs/pipeline/photomaker_branched2_ref1.yaml`](src/configs/pipeline/photomaker_branched2_ref1.yaml) – validation pipeline config (branched PhotoMaker)  
  ```yaml
  _target_: src.pipelines.photomaker_branched_orig_fixed.PhotomakerBranchedPipeline.from_pretrained
  pretrained_model_name_or_path: stabilityai/stable-diffusion-xl-base-1.0
  photomaker_start_step: 10
  merge_start_step: 10
  branched_attn_start_step: 15
  branched_start_mode: both
  train_branch_mode: both
  pose_adapt_ratio: 0
  ca_mixing_for_face: false
  face_embed_strategy: id_embeds
  ```

- [`src/pipelines/photomaker_branched_orig_fixed.py`](src/pipelines/photomaker_branched_orig_fixed.py) – SDXL pipeline wrapper used for validation (imports `branched_new2`, which in our attn1 configs still resolves to `attn_processor.py`)  
  ```python
  class PhotoMakerStableDiffusionXLPipeline(StableDiffusionXLPipeline):
      ...  # modified SDXL pipeline with PhotoMaker v2 + branched logic

  class PhotomakerBranchedPipeline:
      @staticmethod
      def from_pretrained(model, accelerator, *args, **kwargs):
          photomaker_start_step_cfg = kwargs.pop("photomaker_start_step", 10)
          ...
          id_alpha_cfg = kwargs.pop("id_alpha", 0.3)
          pipeline = PhotoMakerStableDiffusionXLPipeline.from_pretrained(..., unet=unwrapped_model.unet, ...)
          pipeline.id_alpha = float(id_alpha_cfg)
          return pipeline
  ```

- [`src/trainer/sdxl_trainers.py`](src/trainer/sdxl_trainers.py) – training loop logic for SDXL / PhotoMaker  
  ```python
  class PhotomakerLoraTrainer(SDXLTrainer):
      def process_batch(self, batch, train_metrics):
          self.optimizer.zero_grad()
          output = self.model(**batch, do_cfg=do_cfg)
          all_losses = self.criterion(**batch)
          self.accelerator.backward(batch["loss"])
          self._clip_grad_norm()
          self.optimizer.step()
  ```

- [`src/configs/datasets/all_datasets_local.yaml`](src/configs/datasets/all_datasets_local.yaml) – train/val dataset wiring  
  ```yaml
  train:
    one_id:
      _target_: src.datasets.cosmic.OneIDTrain
      cosmic_json_pth: ../dataset_full/one_id/nm0005092_adj_train.json
      images_path: ../dataset_full/one_id/nm0005092_adj
  val:
    one_id_val:
      _target_: src.datasets.manual_val.ManualPhotoMakerValDataset
      images_dir: ../dataset_full/one_id/ref
  ```

- [`src/datasets/cosmic.py`](src/datasets/cosmic.py) – training dataset (OneID / Cosmic)  
  ```python
  class OneIDTrain(Dataset):
      def __init__(self, cosmic_json_pth, images_path, num_refs, instance_transforms):
          ...
      def __getitem__(self, idx):
          # returns pixel_values, prompts, ref_images, face_bbox, etc.
  ```

- [`src/datasets/manual_val.py`](src/datasets/manual_val.py) – manual validation dataset  
  ```python
  class ManualPhotoMakerValDataset(Dataset):
      def __getitem__(self, idx):
          return {
              "ref_images": [ref_img],
              "prompt": sample["prompt"],
              "seed": sample["seed"],
              "id": sample["id"],
          }
  ```

- [`src/configs/metrics/all_metrics_oneid.yaml`](src/configs/metrics/all_metrics_oneid.yaml) and metric implementations  
  ```yaml
  clip_ts:
    _target_: src.metrics.text_sim.TextSimMetric
    model_name: ViT-L/14@336px
  id_sim_best:
    _target_: src.metrics.id_sim_metric.IDSimBest
    id_embeds_pth: ../dataset_full/one_id/id_embeds_one_id.pth
  ```

These files together define the training loop, model, branched attention, datasets, pipeline, and metrics required to train with `one_id_br_test_local`.  
