# Quality-neutral runtime optimizations

Date: 2026-07-24

## Scope

The April RHCA replay configurations opt into a small runtime patch intended
to remove repeated work without changing the model, loss, sampled target
latents, optimizer updates, or RealVis validation generation.

All new model/trainer switches default to the historical behavior. Other
configs are unaffected unless they explicitly enable the switches.

## Enabled for the April replay

- `model.skip_unused_text_conditioning=true`: when
  `train_ba_all_steps=true`, do not encode the text-only prompt used solely by
  the disabled timestep-routing branches. This also avoids the associated
  host synchronization from `timestep.item()`.
- `model.conditioning_cache_enabled=true`: cache frozen deterministic
  conditioning by an explicit dataset-provided `reference_cache_key`. Cached
  values include PhotoMaker prompt conditioning, InsightFace identity input,
  deterministic reference VAE mode, and face masks. The target image VAE
  posterior is still sampled on every training step.
- `model.cache_prepared_masks=true`: memoize each mask resolution only within
  the current doubled U-Net forward. The cache is attached to that forward's
  mask tensor and cannot cross into another batch.
- `model.compute_branch_debug_outputs=false`: do not construct `noise_face`
  and `noise_bg` tensors after the merged training prediction has already been
  computed. These tensors are diagnostic only and do not participate in the
  loss.
- `trainer.post_backward_parameter_touch=false`: disable a historical
  zero-valued parameter scan that runs after backward and therefore cannot
  affect that step's gradient reduction.
- `trainer.grad_norm_log_only=true`: compute gradient norms only on steps
  where they are logged. This changes only gradient-norm telemetry from an
  interval aggregate to the value at the logging step; optimizer clipping and
  updates are unchanged.

Persistent workers are deliberately not enabled. Keeping workers alive would
change their augmentation RNG sequence across epochs. The one-record Cosmic
dataset instead uses its existing virtual length, which avoids per-step worker
recreation without changing epoch-boundary behavior.

## Cache safety

Conditioning is cached only when the dataset supplies
`reference_cache_key`. The key describes both the file and its reference-side
transformation (`raw` or horizontal-flip state), preventing augmented
references from sharing entries accidentally. If the key is absent, the
original computation runs on every sample.

The cache is cleared before every validation epoch. This releases its GPU
tensors before the training model is offloaded and the RealVis validation
pipeline is created.

## Validation guarantee

The RealVis validation pipeline, scheduler, seeds, masks, image count, and
generation loop are unchanged. Runtime flags that remove training-only debug
work are not copied onto the validation pipeline. Validation therefore keeps
the historical execution path.

## Reverting

Set the following values to restore historical runtime behavior:

```yaml
model:
  skip_unused_text_conditioning: false
  conditioning_cache_enabled: false
  cache_prepared_masks: false
  compute_branch_debug_outputs: true
trainer:
  post_backward_parameter_touch: true
  grad_norm_log_only: false
```
