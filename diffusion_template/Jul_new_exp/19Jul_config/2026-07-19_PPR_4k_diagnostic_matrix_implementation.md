# PPR 4k diagnostic matrix implementation

Date: 19 July 2026

## Purpose

This checkpoint-only test distinguishes five explanations for weak NN2-PPR
validation effects without taking an optimizer step:

- A: exact ordinary PhotoMaker epsilon at every BA-active step;
- B: the unchanged 4k PPR checkpoint;
- C: PPR with the final `base_outside_core` anchor disabled at runtime;
- D: the applied processor residual multiplied by four after gate and cap;
- E: original PhotoMaker identity conditioning with only the PPR spatial
  reference latent and reference mask swapped to another identity.

The checkpoint is strictly restored before any runtime override. The temporary
RealVis validation model and pipeline are constructed once, and A–E use fresh
pipeline calls with the same sample seeds. The runner loads the checkpoint once;
the epoch-2 assertion, strict architecture check, exact processor-tensor check,
and nonzero connector check all operate on that in-memory load.

## Launcher

```text
jul_serv_runs/start_ba_NN2_ppr1_realvis_4k_diagnostic_1gpu.sh
```

Run on the NN2-PPR server:

```bash
cd /home/niko/rsrch/diffusion_template

CHECKPOINT_PATH=/home/niko/rsrch/diffusion_template/saved/ba_NN2_ppr1_realvis_1gpu/checkpoint-epoch2.pth \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_4k_diagnostic_1gpu.sh
```

GPU 0 is the default. Use `CUDA_VISIBLE_DEVICES=<gpu>` to select another GPU.
The script runs detached unless `RUN_FOREGROUND=1` is set.

The launcher refuses to replace an existing nonempty output directory. For an
intentional clean rerun:

```bash
OVERWRITE_OUTPUT=true \
CHECKPOINT_PATH=/path/to/checkpoint-epoch2.pth \
bash jul_serv_runs/start_ba_NN2_ppr1_realvis_4k_diagnostic_1gpu.sh
```

## Outputs

The default output directory is `ppr_4k_diagnostic/`:

```text
ppr_4k_diagnostic/
  A_exact_pm/
  B_current_ppr/
  C_no_anchor/
  D_ppr_x4/
  E_reference_swap/
  metrics.csv
  epsilon_diagnostics.jsonl
  manifest.json
  contact_sheets/
```

A–D contain all 96 fixed validation samples with identical filenames. E
contains 12 samples selected across lower, middle and upper generation-face
area thirds.

`metrics.csv` contains PNG SHA-256, normalized whole-image and face-core pixel
MAE versus A, ID similarity, text similarity, identity/prompt/seed metadata,
and E's swapped spatial identity.

`epsilon_diagnostics.jsonl` contains:

- per-option hashes of initial latents and reference noise; finalization fails
  if A–D or a selected E sample does not match A;
- pre- and post-output-control epsilon/base RMS ratios inside the face core and
  outside the generation bbox at denoising steps 15, 25, 35 and 49;
- per selected PPR processor applied residual ratios after core mask, gate,
  runtime scale and RMS cap.

## Runtime isolation

All new behavior is disabled by default:

```yaml
ppr_diagnostic_matrix: false
model:
  ba_output_anchor_mode: base_outside_core
```

`ba_ppr_runtime_scale` defaults to `1.0`, and the spatial-reference override is
used only when `ppr_reference_image` is explicitly passed. Existing training
and validation calls retain their previous architecture and behavior.
