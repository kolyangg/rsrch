# PPR 8k residual-scale sweep

## Goal

Determine whether the 8k checkpoint learned a useful correction that is simply too weak at the current `1×` inference strength. Do **not** retrain until this test is complete.

## Use the existing checkpoint

Run the same fixed RealVis validation set from the 8k checkpoint at:

- `scale=0`: ordinary PhotoMaker baseline;
- `scale=1`: current PPR behavior;
- `scale=2`;
- `scale=3`;
- `scale=4`;
- optionally `scale=6` if `4×` remains stable.

Keep everything else identical: checkpoint, seed, identity image, spatial reference, prompt, negative prompt, masks, resolution, scheduler, CFG, denoising steps and PhotoMaker start step.

## Required implementation detail

Apply `runtime_scale` to the **final bounded PPR residual**, immediately before it is spatially gated and added to the ordinary target self-attention output:

```python
applied_delta = runtime_scale * bounded_delta
output = base_output + spatial_gate * applied_delta
```

Do not implement the sweep only by scaling the residual before a hard cap; clipping could suppress the requested strength. Do not change model weights or save modified checkpoints.

For every generated image, record:

```text
checkpoint
runtime_scale
seed
prompt_id
reference_id
PPR mode
active processor count
mean gate
RMS(applied_delta) / RMS(base_output)
cap fraction
```

## Comparisons

Use identical filenames and make paired contact sheets containing `0×`, `1×`, `2×`, `3×`, `4×` and, if generated, `6×`.

Assess:

- identity similarity to the identity reference;
- face/head alignment with neck and body;
- facial realism and landmark stability;
- prompt and pose adherence;
- seams, duplicated features, texture corruption and body drift;
- face-region LPIPS/MAE relative to `0×`;
- reference-swap sensitivity at the most promising scale.

The reference-swap test must change only the PPR spatial-reference input. Keep PhotoMaker identity conditioning fixed.

## Decision rules

| Result | Interpretation | Action |
|---|---|---|
| `2–4×` improves identity/alignment without meaningful artifacts | The learned residual is useful but under-scaled | Use the best runtime scale with this checkpoint; retraining is not immediately necessary |
| Quality peaks above `1×` but the desired deployment interface requires `1×` | Branch calibration is too weak | In the next run, calibrate the gate/output magnitude toward the measured optimum; change one training variable at a time |
| Higher scales change pixels but do not improve identity | The residual direction is not useful | Do not increase gain or LR; revise supervision, loss or reference conditioning |
| Higher scales improve identity but create seams/body drift | The correction is useful but spatial application is poor | Tune mask dilation/feathering, cap, sites or timestep schedule |
| Higher scales differ, but reference swapping has little effect | The branch may be learning a generic correction | Audit reference K/V routing and reference-token use before retraining |
| Even `4–6×` is numerically and visually unchanged | The residual is not reaching the output | Debug checkpoint loading, active processors, gates, masks and residual placement |

## If another training run is needed

Only modify training after selecting the relevant failure class above.

If the direction is useful but weak, first prefer a calibrated learned/fixed output gain or gate schedule. Consider a larger PPR-branch learning-rate multiplier only if gradients and parameter deltas show that the branch is genuinely under-updating. Preserve ordinary PhotoMaker behavior as the zero-residual base.

Do not simultaneously change learning rate, masks, loss and inference scale: that would make the cause of any improvement impossible to identify.

## Recommended output

```text
ppr_8k_scale_sweep/
  metadata.csv
  metrics_summary.csv
  contact_sheets/
  scale_0/
  scale_1/
  scale_2/
  scale_3/
  scale_4/
  scale_6/          # optional
  conclusion.md
```

`conclusion.md` should identify the best scale, report identity/alignment and artifact trade-offs, state whether reference swapping affects the correction, and choose exactly one next action: retain the checkpoint with inference scaling, tune spatial application, audit routing, or retrain the objective.
