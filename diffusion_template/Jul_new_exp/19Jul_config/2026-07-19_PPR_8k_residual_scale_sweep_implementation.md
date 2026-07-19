# PPR 8k residual-scale sweep implementation

Date: 19 July 2026

## Launcher

```text
jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_scale_sweep_1gpu.sh
```

The launcher uses one GPU (GPU 0 by default), loads
`checkpoint-epoch4.pth` strictly, and runs the fixed 96-image RealVis
validation set at residual scales `0, 1, 2, 3, 4`. Scale zero returns the exact
ordinary single-target PhotoMaker prediction. Nonzero scales multiply the
already bounded residual immediately before its spatial gate and addition.
No weights are changed or saved.

Run:

```bash
cd /home/niko/rsrch/diffusion_template

CHECKPOINT_PATH=/home/niko/rsrch/diffusion_template/saved/ba_NN2_ppr1_realvis_1gpu/checkpoint-epoch4.pth \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_scale_sweep_1gpu.sh
```

The default run also spatially swaps 12 representative references at scale
`3`, while retaining the original PhotoMaker identity conditioning. Select a
different candidate with `SWAP_SCALE=2` or `SWAP_SCALE=4`; use
`SWAP_SCALE=none` to omit that diagnostic.

To include the optional `6x` column:

```bash
INCLUDE_SCALE6=true \
SWAP_SCALE=4 \
CHECKPOINT_PATH=/path/to/checkpoint-epoch4.pth \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_8k_scale_sweep_1gpu.sh
```

The launcher runs detached by default. Use `RUN_FOREGROUND=1` for foreground
execution, `CUDA_VISIBLE_DEVICES=<gpu>` to select the GPU, and
`OVERWRITE_OUTPUT=true` only for an intentional clean rerun.

## Outputs

The default output is `ppr_8k_scale_sweep/`:

```text
ppr_8k_scale_sweep/
  metadata.csv
  metrics_summary.csv
  processor_diagnostics.jsonl
  manifest.json
  conclusion.md
  contact_sheets/
  scale_0/
  scale_1/
  scale_2/
  scale_3/
  scale_4/
  reference_swap_scale_3/
```

`metadata.csv` records the checkpoint, runtime scale, seed, stable prompt ID,
reference identity, PPR mode, active processor count, mean gate, applied
residual/base RMS ratio, cap fraction, image hash, whole/face MAE, available
face LPIPS, identity similarity, and text similarity for every image.
Reference-swap rows also record direct whole/face MAE against the non-swapped
image at the same runtime scale.

The runner verifies that initial target latents and reference noise are
identical across scales. Contact sheets use paired columns and identical
filenames.

## LPIPS availability

The current PhotoMaker environment does not include the optional `lpips`
package. The sweep still runs and records `NaN` for face LPIPS, with the reason
in `manifest.json` and `conclusion.md`; face-core MAE remains available. If
`lpips` is installed in the same environment, AlexNet LPIPS is computed
automatically.

`conclusion.md` reports the automatic identity-similarity leader but leaves the
final action pending visual inspection, because alignment, seams, artifacts and
body drift cannot be selected safely from scalar metrics alone.
