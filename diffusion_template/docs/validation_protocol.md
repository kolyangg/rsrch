# Default validation protocol

The standard training protocol validates at step 0 and then every 2,000
optimizer steps on the fixed 96-image `manual_val` panel. It generates one
image per validation item. Historical one-identity and controlled-factorial
experiments may retain an explicitly documented 12-image panel; those results
must not be compared directly with full-96 aggregates.

The controlling Hydra values are:

```yaml
trainer:
  validation_interval_steps: 2000
  face_quality:
    enabled: true
    expected_images: 96

datasets:
  val:
    manual_val:
      limit: 96

validation_args:
  num_images_per_prompt: 1
```

`validation_interval_steps` must be an exact multiple of `trainer.epoch_len`.
The trainer fails before launch if the requested cadence cannot be represented
exactly. Set it to `null` only for the legacy every-epoch behavior, or to `0`
to retain step-0 validation while disabling periodic training validation.
Validation-only checkpoint schedules always run every explicitly requested
checkpoint.

For new training launchers, `trainer.epoch_len` defaults to 2,000 optimizer
steps, so validation, checkpointing, and epoch boundaries coincide. Express
the total budget as `trainer.n_epochs = total_steps / 2000` (for example,
2 epochs for 4k, 10 for 20k, and 20 for 40k). Historical continuation and
validation scripts that address immutable 500-step checkpoint epoch numbers
must explicitly set `TRAIN_EPOCH_LEN=500`; they are not new-run defaults.

## Face-quality metrics

Every actual validation event runs
`tools/inference/calculate_face_quality_metrics.py` by default and logs these
seven Comet curves in the separate `face_quality/` namespace:

- `face_detection_rate`
- `topiq_face_mean`
- `topiq_face_p10`
- `topiq_face_coverage`
- `topiq_mean`
- `musiq_mean`
- `maniqa_mean`

The definitions are the same as the July 2026 historical backfill: InsightFace
largest-face detection, a 25% padded square 512-pixel crop, and PyIQA 0.1.15
TOPIQ-Face, TOPIQ, MUSIQ, and MANIQA-PIPAL. **All four models receive that same
face crop; none receives the whole generated image.** TOPIQ-Face additionally
runs its own internal face alignment, which is why its coverage can be lower
than InsightFace detection coverage. Each step also uploads one
`face_quality_details__<partition>__step_<step>.csv` Comet asset containing
per-image values. It is an API asset with hidden-by-default metadata, not a
report table.

Disable the calculation explicitly with:

```yaml
trainer:
  face_quality:
    enabled: false
```

The scoring interpreter defaults to the active Conda environment's `python`.
If PyIQA is supplied by a separate machine-local environment, set
`FACE_QUALITY_SCORER_PYTHON` to that interpreter and, when required, set
`PYTHONPATH` in the launcher. Do not put machine paths or credentials in a
committed config. The active RHCA launcher also discovers the established Neb
`metric_envs/pyiqa-0.1.15` interpreter and the Serv
`python_overlays/pyiqa-0.1.15` overlay, and fails before training if neither
PyIQA 0.1.15 nor an explicit `trainer.face_quality.enabled=false` override is
available.

With one validation process, `device: auto` uses CUDA and temporarily offloads
the generation pipeline before scoring. With multi-process DDP it uses CPU on
rank 0 because moving only one rank's wrapped model is unsafe.
