# Diffusion Template – One‑ID Branched Attention Training

## Training command (one‑ID attn1 config)

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=XXX \
accelerate launch --config_file=src/configs/ddp/accelerate.yaml train.py \
  --config-name=one_id_br_attn1 \
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

- [`src/configs/one_id_br_attn1.yaml`](src/configs/one_id_br_attn1.yaml) – one‑ID attn1 training config (branched attn v1/v2 switch)  
  ```yaml
  defaults:
    - trainer: photomaker_lora
    - model: photomaker_branched_lora2
    - pipeline: photomaker_branched2_ref1
    - metrics: all_metrics_oneid
    - datasets: all_datasets
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

- [`src/model/photomaker_branched/lora2.py`](src/model/photomaker_branched/lora2.py) – main training model (PhotoMaker + branched LoRA, wired by `id_alpha`, `train_ba_only`, `pose_adapt_ratio`, `face_embed_strategy`, `use_id_embeds`)  
  ```python
  class PhotomakerBranchedLora(SDXL):
      def __init__(..., pose_adapt_ratio: float = 0.25,
                   ca_mixing_for_face: bool = True,
                   face_embed_strategy: str = "face",
                   train_branch_mode: str = "both",
                   train_ba_only: bool = False,
                   ba_weights_split: bool = False,
                   use_attn_v2: bool = True,
                   id_alpha: float = 0.3,
                   use_id_embeds: bool = True):
          ...
          self.pose_adapt_ratio = float(pose_adapt_ratio)
          self.ca_mixing_for_face = bool(ca_mixing_for_face)
          self.train_branch_mode = (train_branch_mode or "both").lower()
          # Branched-attn knobs coming from configs
          self.face_embed_strategy = (face_embed_strategy or "face").lower()
          self.train_ba_only = bool(train_ba_only)   # controls which params are trainable
          self.ba_weights_split = bool(ba_weights_split)
          self.id_alpha = float(id_alpha)            # strength of ID mixing in BranchedAttnProcessor
          self.use_id_embeds = bool(use_id_embeds)   # global on/off for ID projection branch
      def prepare_for_training(self):
          ...
          patch_unet_attention_processors(
              pipeline=self,
              mask=zero_ctx,
              mask_ref=zero_ctx,
              scale=1.0,
              id_embeds=None,
              class_tokens_mask=None,
          )
      def get_trainable_params(self, config):
          if getattr(self, "train_ba_only", False):
              # Train only branched processors + LoRA on attention projections
              proc_params, lora_params = [], []
              for name, p in self.unet.named_parameters():
                  if not p.requires_grad:
                      continue
                  if ".attn1.processor." in name or ".attn2.processor." in name:
                      proc_params.append(p)
                  elif "lora_A" in name or "lora_B" in name:
                      lora_params.append(p)
              param_groups = []
              if proc_params:
                  param_groups.append({"params": proc_params, "lr": config.lr_for_lora, "name": "branched_processors"})
              if lora_params:
                  param_groups.append({"params": lora_params, "lr": config.lr_for_lora, "name": "branched_lora"})
              return param_groups
          # otherwise: train all LoRA parameters with requires_grad=True
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
      def _apply_runtime_flags(proc, pipe):
          for k in ("pose_adapt_ratio", "ca_mixing_for_face", "train_branch_mode", "id_alpha", "use_id_embeds"):
              if hasattr(pipe, k):
                  setattr(proc, k, getattr(pipe, k))
      ...
      for name in pipeline.unet.attn_processors.keys():
          ...
          if name.endswith("attn1.processor"):
              proc = BranchedAttnProcessor(
                  hidden_size=hidden_size,
                  cross_attention_dim=hidden_size,
                  scale=scale,
              ).to(pipeline.device, dtype=pipeline.unet.dtype)
              proc.set_masks(_mask, _mref)
              _apply_runtime_flags(proc, pipeline)
              proc.id_embeds = _idem  # zeros if missing; gated by use_id_embeds
              new_procs[name] = proc
          elif name.endswith("attn2.processor"):
              proc = BranchedCrossAttnProcessor(...)
              proc.set_masks(_mask, _mref)
              proc.id_embeds = _idem
              proc.class_tokens_mask = class_tokens_mask
              new_procs[name] = proc
      pipeline.unet.set_attn_processor(new_procs)
  ```
  For all current `*_attn1*` configs (`use_attn_v2: false`), this patcher is active in both training and validation but always resolves to **`attn_processor.py`**; `_old2/attn_processor2.py` is only used if you explicitly enable `use_attn_v2: true`.

- [`src/model/photomaker_branched/attn_processor.py`](src/model/photomaker_branched/attn_processor.py) – branched self‑/cross‑attention actually used for attn1 configs (`use_attn_v2: false`) and controlled by `pose_adapt_ratio`, `id_alpha`, `use_id_embeds`  
  ```python
  class BranchedAttnProcessor(nn.Module):
      def __init__(..., scale=1.0, ...):
          self.id_embeds = None
          self.id_to_hidden = None
          self.use_id_embeds: bool = True  # can be overridden from config
      def __call__(..., hidden_states, encoder_hidden_states=None, ...):
          # splits [noise, reference] batch and applies background + face branches
          POSE_ADAPT_RATIO   = getattr(self, "pose_adapt_ratio", 0.25)
          CA_MIXING_FOR_FACE = getattr(self, "ca_mixing_for_face", True)
          ...
          # Blend reference and current-noise face features; higher POSE_ADAPT_RATIO → more pose flexibility
          face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + POSE_ADAPT_RATIO * noise_face_hidden

          # ID injection branch (disabled in attn1 configs via use_id_embeds: false)
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

- [`src/configs/pipeline/photomaker_branched2_ref1.yaml`](src/configs/pipeline/photomaker_branched2_ref1.yaml) – validation pipeline config (branched PhotoMaker; sets `branched_attn_start_step`, `pose_adapt_ratio`, `face_embed_strategy`)  
  ```yaml
  _target_: src.pipelines.photomaker_branched_orig_fixed.PhotomakerBranchedPipeline.from_pretrained
  pretrained_model_name_or_path: stabilityai/stable-diffusion-xl-base-1.0
  photomaker_start_step: 10
  merge_start_step: 10
  branched_attn_start_step: 15
  branched_start_mode: both
  train_branch_mode: both
  pose_adapt_ratio: 0.25
  ca_mixing_for_face: false
  face_embed_strategy: id_embeds
  ```

- [`src/pipelines/photomaker_branched_orig_fixed.py`](src/pipelines/photomaker_branched_orig_fixed.py) – SDXL pipeline wrapper used for validation (imports `branched_new2`, and exposes `pose_adapt_ratio`, `face_embed_strategy`, `use_id_embeds`, `id_alpha` to the processors)  
  ```python
  class PhotoMakerStableDiffusionXLPipeline(StableDiffusionXLPipeline):
      ...  # modified SDXL pipeline with PhotoMaker v2 + branched logic

  class PhotomakerBranchedPipeline:
      @staticmethod
      def from_pretrained(model, accelerator, *args, **kwargs):
          photomaker_start_step_cfg = kwargs.pop("photomaker_start_step", 10)
          merge_start_step_cfg = kwargs.pop("merge_start_step", 10)
          branched_attn_start_step_cfg = kwargs.pop("branched_attn_start_step", 10)
          ...
          face_embed_strategy_cfg = kwargs.pop(
              "face_embed_strategy",
              getattr(unwrapped_model, "face_embed_strategy", "face"),
          )
          use_id_embeds_cfg = kwargs.pop(
              "use_id_embeds",
              getattr(unwrapped_model, "use_id_embeds", True),
          )
          id_alpha_cfg = kwargs.pop(
              "id_alpha",
              getattr(unwrapped_model, "id_alpha", 0.3),
          )
          pipeline = PhotoMakerStableDiffusionXLPipeline.from_pretrained(..., unet=unwrapped_model.unet, ...)
          pipeline.pose_adapt_ratio = pose_adapt_ratio_cfg
          pipeline.face_embed_strategy = face_embed_strategy_cfg
          pipeline.use_id_embeds = bool(use_id_embeds_cfg)
          pipeline.id_alpha = float(id_alpha_cfg)
          # cached copies for debugging / serialization
          pipeline._config_branched_attn_start_step = branched_attn_start_step_cfg
          ...
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
