# One-ID validation-reference holdout control

Date: 2026-07-24

## Purpose

The historical April RHCA replay validates with
`../dataset_full/one_id/ref/51.jpg`. That file is byte-identical to
`../dataset_full/one_id/nm0005092_adj/51.jpg`, which is present in the
19-image training manifest. The historical launcher remains unchanged for
exact replay comparability.

This companion control holds `51.jpg` out of training while preserving the
historical architecture, optimizer, loss cadence, inference schedule,
validation prompts, seed, reference image, bounding boxes, and metrics.

## Split invariant

Training dataset `one_id_holdout51` wraps the historical `OneIDTrain` dataset
and removes `51.jpg` from both aligned collections used by that loader:

- `_index`, which supplies diffusion targets; and
- `ids`, which supplies different-image same-identity reference candidates.

The resulting training pool contains 18 images. Validation continues to use
the single held-out reference `ref/51.jpg` across the same 12 prompts at seed
0.

## Launcher

Run from `diffusion_template/`:

```bash
bash launchers/active/run_rhca_apr2026_one_id_holdout51_1gpu.sh
```

Default run name:

```text
rhca_apr2026_one_id_holdout51_4k
```

This is a new controlled comparison, not an exact reproduction of the
historical training dataset. Compare it against the historical replay at
matched steps and keep the two run labels distinct.
