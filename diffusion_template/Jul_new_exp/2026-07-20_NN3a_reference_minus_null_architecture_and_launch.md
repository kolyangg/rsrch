# NN3a reference-minus-null architecture and launch

Date: 20 July 2026

## Goal

NN3a is the next single-GPU experiment after the NN2-PPR epoch-4 diagnostics.
It keeps the parts of packed spatial branched attention that worked:

- the reference image remains a full noised VAE stream;
- target queries retrieve K/V from a packed reference-face ROI;
- target and reference coordinates are not directly substituted;
- the reference stream continues through frozen ordinary self-attention;
- the target face receives a bounded, gated additive residual;
- an ordinary PhotoMaker prediction remains exact outside the feathered face
  core;
- split target/reference cross-attention remains active but frozen.

It changes the trainable connector input so that the residual cannot use the
stable target-attention term as a shortcut.

## Why NN2-PPR failed

NN2-PPR computed:

```text
C_ref = Attention(Q_target, K_reference_ROI, V_reference_ROI)
x_NN2 = C_ref - A_target
delta = Connector(x_NN2)
```

The 8k reference/noise tests established that:

- the branch was active;
- R1 and R2 produced different internal reference representations;
- the connector retained a measurable R1/R2 signal;
- changing R1 to R2 did not move identity toward R2;
- the visible effect was dominated by expression, mouth, eyes, and texture;
- original identity degraded while prompt similarity increased.

`-A_target` allowed the connector to produce a useful target/prompt correction
without relying on reference identity. Neutralizing reference-half
cross-attention did not fix this.

## NN3a architecture

NN3a uses a fixed no-person null memory with zero K/V:

```text
C_ref  = Attention(Q_target, K_reference_ROI, V_reference_ROI)
C_null = Attention(Q_target, K_zero, V_zero) = 0
x_NN3  = C_ref - C_null
delta  = Connector(x_NN3)
```

The same target query is used conceptually for both candidates. Because
`V_zero = 0`, the null candidate is exactly zero. Both connector projections
are bias-free, so zero reference evidence maps to an exact zero residual.

At each selected up-block self-attention site:

```text
raw_delta     = Up(Down(C_ref - C_null))
bounded_delta = RMSCap(raw_delta, relative_to=A_target, maximum=0.25)
gate          = 0.5 * sigmoid(gate_logit)
A_NN3         = A_target + inner_face_core * gate * bounded_delta
```

The second/reference batch half still returns its frozen ordinary
self-attention continuation. At the U-Net output:

```text
epsilon =
    inner_face_core * epsilon_NN3
  + (1 - inner_face_core) * epsilon_PhotoMaker
```

This preserves the core branched-attention principle:

```text
Q = target coordinates
K,V = spatial reference-face evidence
```

The architecture does not introduce compact QFormer identity tokens, replace
branched attention with ordinary PhotoMaker attention, or allow the reference
grid to own target coordinates.

## Conservative identity supervision

The diagnostic showed reference-content sensitivity without reference-identity
direction. NN3a therefore enables the existing decoded reference-ID objective,
but more conservatively than the failed NN1e run:

| Setting | NN1e | NN3a |
|---|---:|---:|
| ID-loss weight | 0.10 | 0.05 |
| Maximum training timestep | 400 | 300 |
| Spatial composition | Absolute/destructive full BA | Bounded residual + PM anchor |

The objective is:

```text
0.8 * full diffusion MSE
+ 0.2 * target-face diffusion MSE
+ 0.05 * decoded reference-ID loss, only at t <= 300
```

This loss is intentionally small. Its role is to align the reference-derived
residual with identity rather than allow another high-strength metric
shortcut.

## Trainable and frozen parameters

Trainable at the selected up-block `attn1` sites:

- reference K LoRA;
- reference V LoRA;
- connector down projection;
- zero-initialized connector up projection;
- scalar gate logit.

Frozen:

- target self-attention base;
- reference continuation base;
- all split cross-attention projections;
- PhotoMaker and base U-Net weights;
- VAE, text encoders, and PhotoMaker ID encoder.

## Reversible toggles

The default remains NN2 behavior:

```yaml
model:
  ba_connector_input_mode: reference_minus_target
```

NN3a opts in with:

```yaml
model:
  ba_connector_input_mode: reference_minus_null
```

To reproduce NN2 connector behavior while keeping the NN3 config:

```bash
model.ba_connector_input_mode=reference_minus_target
```

To disable the additional identity supervision:

```bash
model.use_id_loss=false
```

No existing NN1, NN2, PPR diagnostic, or checkpoint behavior changes unless
the new connector mode is explicitly selected.

## Training protocol

- GPUs: one;
- default physical GPU: 0;
- dataset: `cosmic_large_neb`;
- physical batch: 2;
- effective batch: 2;
- maximum: 20k optimizer steps;
- validation: fixed 96-image RealVis set at step 0 and every 2k;
- validation batch: 12;
- inference schedule: text 0–9, PhotoMaker 10–14, BA 15–49;
- training timesteps: BA-active inference region;
- precision: BF16;
- optimizer LR: `5e-5`;
- warmup: 2k optimizer steps.

The `cosmic_large_neb` dataset paths are:

```text
/home/niko/datasets/gathered_data_cosmic_large_filtered.json
/home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang
```

## Launch

```bash
cd /home/niko/rsrch/diffusion_template

bash jul_serv_runs/start_ba_NN3a_reference_null_realvis_1gpu.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
PM_PATH=/home/niko/models/PhotoMaker-V2/photomaker-v2.bin \
bash jul_serv_runs/start_ba_NN3a_reference_null_realvis_1gpu.sh
```

The launcher starts detached by default and prints the PID and log path.
Use `RUN_FOREGROUND=1` for an interactive run.

## Validation decision rules

Do not judge NN3a only by face MAE or the ordinary original-ID metric.

At 2k, 4k, and 6k, inspect:

1. whether faces visibly depart from PM0 without expression-only
   amplification;
2. R1/R2 directional identity gain using the same reference/noise test;
3. original and swapped-reference identity similarities;
4. body, pose, hands, clothing, and background stability;
5. face seams, eye/mouth distortion, and landmark displacement;
6. connector, gate, cap, and per-site gradient/norm diagnostics.

Promising behavior:

- R2 replacement moves a clear majority of faces toward R2;
- mean directional gain is positive with a bootstrap interval above zero;
- the null/reference-free residual remains zero;
- body and scene stay PhotoMaker-aligned;
- the late up-block route no longer acts mainly as an expression amplifier.

Stop early if:

- the branch again changes expression but not R2 identity;
- original identity falls without a compensating gain toward R2;
- face artifacts increase monotonically across validations;
- ID loss improves its metric while visually smoothing or collapsing facial
  landmarks.

## Files

Architecture/configuration:

```text
src/model/photomaker_branched/packed_residual_attn_processor.py
src/model/photomaker_branched/lora2.py
src/model/photomaker_branched/branched_runtime.py
src/pipelines/br_pipeline_helpers.py
src/configs/model/photomaker_branched_lora2.yaml
src/configs/one_id_ba_NN3a_reference_null.yaml
```

Launcher:

```text
jul_serv_runs/start_ba_NN3a_reference_null_realvis_1gpu.sh
```

Visualizer:

```text
ba_architecture_explorer/index.html
ba_architecture_explorer/app.js
```
