# Training handoff: `n3a_fullgrid_up_core_ring_anchor`

## Purpose and provenance

This document describes how to reproduce and train the repaired N3a architecture
called `n3a_fullgrid_up_core_ring_anchor` using the current main repository,
without replacing or specializing the existing attention implementation.

The step-zero experiment was run from:

- branch: `main_clean`
- commit: `1e88825dc4a325ea1e146be2fa519801f048a73e`
- base Hydra config: `src/configs/one_id_ba_NN1a_n3a_replay.yaml`
- model class: `src.model.photomaker_branched.lora2.PhotomakerBranchedLora`
- legacy SA processor: `src/model/photomaker_branched/attn_processor_cleanest.py`
- doubled-branch runtime and final epsilon anchor:
  `src/model/photomaker_branched/branched_runtime.py`

The current worktree contains unrelated research artifacts. An implementation
agent should start from a clean `main_clean` checkout at the commit above (or a
newer commit after checking the relevant files), then add a new opt-in config.
Do not copy the notebook's experiment-local model construction shim into the
training path.

## Architecture in one sentence

Keep N3a's doubled target/reference U-Net and full spatial reference grid, but
allow reference K/V to control only an inner target-face core in self-attention
up blocks; use target self-attention in the surrounding face ring, disable
branched cross-attention, and replace the final predicted epsilon outside a
slightly eroded target-face core with the ordinary PhotoMaker prediction.

## Exact toggle set

The experiment used these values:

```yaml
disable_branched_sa: false
disable_branched_ca: true
strict_face_routing: false

train_ba_all_steps: true
train_ba_only: true
branched_attn_weight_mode: noise_and_ref
branched_attn_new_weight_kind: lora
train_branched_ca_lora: false
ba_patch_top_k: 1.0
ba_train_top_k: 1.0
non_ba_train: false

model:
  ba_processor_variant: legacy
  ba_site_policy: all
  ba_sa_train_mode: all
  ba_sa_ref_token_mode: full_grid
  ba_sa_face_mode: core_ring
  ba_sa_core_ratio: 0.68
  ba_sa_ref_layer_scope: up
  ba_target_core_erode_frac: 0.10
  ba_output_anchor_mode: base_outside_core
```

Step-zero evaluation additionally used:

```yaml
model:
  photomaker_start_step: 4
  branched_attn_start_step: 6
  num_inference_steps: 20

validation_args:
  photomaker_start_step: 4
  branched_attn_start_step: 6
  num_inference_steps: 20
  guidance_scale: 5
```

The research runner used RealVisXL V4.0, PhotoMaker V2, BF16, and rank 32. The
resolved step-zero model had 70 legacy self-attention processors, zero branched
cross-attention processors, and approximately 31.95 million trainable processor
parameters. The ordinary N3a launcher already supplies rank 32 and BF16.

## Recommended Hydra config

Create a new config; do not alter the defaults of N3a, NN2, NN4, NN5, NN6, or
NN7. A suitable config body is:

```yaml
# src/configs/one_id_ba_N3a_fullgrid_up_core_ring_anchor.yaml
defaults:
  - one_id_ba_NN1a_n3a_replay
  - _self_

# Reference information is carried only through branched self-attention.
disable_branched_sa: false
disable_branched_ca: true
train_branched_ca_lora: false
strict_face_routing: false

# Preserve the guarded N3a optimizer recipe.
train_ba_all_steps: true
train_ba_only: true
branched_attn_weight_mode: noise_and_ref
branched_attn_new_weight_kind: lora
ba_patch_top_k: 1.0
ba_train_top_k: 1.0
non_ba_train: false
ba_noise_lr_scale: 0.25
loss_kind: masked_alternating

model:
  # Explicit values make checkpoint manifests and reviews unambiguous.
  train_branched_ca_lora: false
  ba_processor_variant: legacy
  ba_site_policy: all
  ba_sa_train_mode: all

  # Exact promising architecture.
  ba_sa_ref_token_mode: full_grid
  ba_sa_face_mode: core_ring
  ba_sa_core_ratio: 0.68
  ba_sa_ref_layer_scope: up
  ba_target_core_erode_frac: 0.10
  ba_output_anchor_mode: base_outside_core

  # Retain the NN1a correctness guards.
  ba_correctness_guards: true
  ba_invalid_sample_policy: skip_batch
  ba_strict_processor_restore: true
  ba_train_timestep_mode: all
  ba_face_prompt_attention_mask: false
  ba_uncond_face_fix: true
  ba_face_prompt_mode: id_only
  use_id_loss: false
```

For a literal reproduction of the 20-step screen, override both `model.*` and
`validation_args.*` schedule fields to `4/6/20`. For a normal 50-step training
validation, the established `10/15/50` schedule is the proportional equivalent.
Do not silently mix `model` and `validation_args` schedules.

## What each architecture toggle does

### `ba_sa_ref_token_mode: full_grid`

The reference branch remains a full latent spatial grid. The reference face
mask zeros content outside the reference face, but tokens are not cropped and
resampled into a normalized ROI. This preserves the closest possible path to
N3a and also preserves its spatial-coordinate sensitivity.

In `attn_processor_cleanest.py`, the non-ROI branch constructs reference K/V
from:

```python
reference_source = ref_hidden * ref_mask_flat
```

This differs from `roi`, which extracts the face rectangle and normalizes it to
a fixed token grid.

### `ba_sa_face_mode: core_ring`

The target query attends twice:

1. target K/V produce `target_face_heads`;
2. reference K/V produce `reference_face_heads`.

An inner elliptical gate then selects reference attention in the core and
target attention in the surrounding face ring:

```python
core_gate = self._inner_core_mask(mask_gate)
hidden_face_heads = (
    target_face_heads * (1.0 - core_gate)
    + reference_face_heads * core_gate
)
```

`ba_sa_core_ratio: 0.68` controls this processor-local inner ellipse. It is not
the same control as the final output-anchor erosion.

### `ba_sa_ref_layer_scope: up`

All legacy SA processors are installed, but reference K/V are enabled only when
the processor name starts with `up_blocks.`. At down and mid sites the target
face attention candidate is used. This preserves pose/layout formation before
reference detail is introduced during decoding.

Important training nuance: with the current legacy installer, processors are
still present at all 70 self-attention sites. `train_ba_only` plus
`noise_and_ref` can therefore leave target/noise projections trainable outside
the active reference sites. That is the exact step-zero architecture. Freezing
all non-up processor parameters would be a separate training ablation, not an
exact reproduction, and should receive its own config/experiment name.

### `disable_branched_ca: true`

No branched attn2 processors should be installed and no CA processor parameters
should enter the optimizer. Set both top-level `train_branched_ca_lora: false`
and `model.train_branched_ca_lora: false`; otherwise strict processor manifests
may expect CA weights that were never installed.

### `strict_face_routing: false`

Keep the historical non-strict N3a merge behavior. The final epsilon anchor is
the primary protection outside the trusted core; enabling strict routing would
change boundary behavior and is a distinct ablation.

### `ba_output_anchor_mode: base_outside_core`

At every active branched prediction, the runtime performs an additional
ordinary single-target U-Net pass using the original processors. It returns:

```python
anchored = base_prediction + core_mask * (
    branched_prediction - base_prediction
)
```

Thus the prediction is exactly ordinary PhotoMaker outside the target core and
branched inside it. `ba_target_core_erode_frac: 0.10` builds this output-space
core from the generated-face mask. This control is separate from the
self-attention `ba_sa_core_ratio: 0.68`.

The base pass is intentionally under `torch.no_grad()`. BA gradients are
therefore localized through the branched prediction inside the core. The extra
base pass also increases compute and memory traffic; budget training throughput
accordingly.

## One current compatibility gap

The runtime already implements `base_outside_core` generically and the notebook
proved it works with the legacy processor. However, the production model
constructor currently rejects any non-`none` output anchor when
`ba_processor_variant != packed_residual_v1` in
`src/model/photomaker_branched/lora2.py`.

The notebook only bypassed that constructor guard for disposable inference:
it instantiated legacy N3a with `none`, restored the model attribute to
`base_outside_core`, and then built the validation pipeline. That shim is not a
training solution.

For an exact trainable config, the implementation agent should make a small,
opt-in compatibility change:

1. Continue validating the allowed anchor values as `none` or
   `base_outside_core`.
2. Allow `base_outside_core` for the legacy processor when the original
   processor registry and target face mask are available. Do not change the
   default (`none`).
3. Keep all packed-residual-specific invariants restricted to
   `packed_residual_v1`; do not make legacy pretend to be packed residual.
4. Record `ba_output_anchor_mode` and `ba_target_core_erode_frac` in the strict
   checkpoint architecture manifest for this legacy opt-in case. Currently
   those fields are added only inside the non-legacy manifest branch.
5. Ensure `br_pipeline_helpers.py` continues copying both fields from model to
   validation pipeline. That propagation already exists.

This is preferable to creating a second processor class or changing global
defaults. Every other configuration should remain behaviorally identical when
its anchor stays `none`.

## Training launch pattern

The existing launcher
`jul_serv_runs/start_ba_NN1a_n3a_replay_1gpu.sh` and shared
`jul_serv_runs/_run_ba_NN1_common_1gpu.sh` provide the right dataset, optimizer,
BF16/rank-32, full-96 validation, and guarded one-GPU structure. Add a sibling
launcher that changes only:

```bash
NN1_CONFIG_NAME=one_id_ba_N3a_fullgrid_up_core_ring_anchor
NN1_RUN_NAME_DEFAULT=ba_N3a_fullgrid_up_core_ring_anchor_1gpu
NN1_DESCRIPTION="N3a full-grid up-only core-ring 0.68 with protected output"
```

Reuse the shared runner instead of copying its full command. The shared runner
currently defaults to 10,000 optimizer steps, effective batch 2, rank 32,
BF16, learning rate `5e-5`, noise-clone LR scale `0.25`, and fixed 96-image
validation. Confirm these values in the launch log because command-line Hydra
overrides take precedence over the new config.

## Required preflight checks

Before starting a long run, execute a one-batch construction/forward smoke test
and verify:

- model class is `PhotomakerBranchedLora`;
- processor variant is `legacy`;
- 70 branched self-attention processors and 0 branched cross-attention
  processors are installed for the RealVisXL/SDXL U-Net;
- every processor reports `full_grid`, `core_ring`, ratio `0.68`, scope `up`;
- model and validation pipeline both report
  `ba_output_anchor_mode=base_outside_core` and erosion `0.10`;
- no `.attn2.processor.*` tensors are trainable;
- processor parameters are present in the optimizer and receive finite,
  nonzero gradients on a valid face batch;
- the protected output path can access `_original_attn_processors`;
- one forward/backward step is finite in BF16;
- the core mask is nonempty for every accepted training sample;
- strict checkpoint save/load reproduces the architecture manifest and exact
  trainable parameter keys.

## Regression tests the implementation agent should add

1. Legacy N3a with `ba_output_anchor_mode: none` remains unchanged.
2. Existing packed-residual configs retain their current validation behavior.
3. Legacy plus `base_outside_core` constructs successfully only with a valid
   mode and erosion in `[0, 0.5)`.
4. With a deterministic input, anchored and ordinary predictions are exactly
   equal outside the output core.
5. Inside the core, the anchored prediction equals the branched prediction.
6. At down/mid processors, `scope=up` returns the target-attention candidate;
   at up processors, it enables reference attention.
7. `disable_branched_ca=true` yields no branched CA processors or trainable CA
   tensors.
8. Checkpoint round-trip rejects a different anchor mode, core ratio, scope, or
   erosion under strict restore.
9. Existing configuration tests pass without changing their snapshots.

## Monitoring during training

At step 0 and every validation interval, retain ordinary PhotoMaker and BA
outputs using identical prompts/seeds. Track at minimum:

- face MAE and outside-face MAE versus ordinary PhotoMaker;
- face detection rate;
- landmark displacement and face-box IoU versus PhotoMaker;
- identity similarity to the matched reference and its gain over PhotoMaker;
- positive identity-gain fraction, both overall and per identity;
- mask/core area, especially empty or oversized cores;
- per-site gradient norms for ref and noise Q/K/V LoRA tensors;
- wrong-reference causality controls;
- exact outside-core epsilon equality on a diagnostic batch.

Do not select checkpoints from aggregate identity gain alone. This architecture
looked unusually strong on the original four Eddie cases, while other N3a
variants became identity-dependent on a broader set. Use the fixed 96-image
manual validation grid and stratify metrics by identity.

## Known limitations and suggested ablations

- Full-grid masked K/V retains absolute spatial layout and many zeroed tokens;
  it can still pull reference pose or facial geometry into the target.
- The 0.68 reference core is visually strong. If training amplifies alignment
  drift, use separate configs for core `0.50` and `0.35`; do not alter the
  canonical config in place.
- A later BA start or larger target-core erosion may reduce geometry movement,
  but each changes the architecture/schedule and needs its own run name.
- The ordinary base pass makes the output safe outside the core but does not
  guarantee correct identity direction inside it.
- `train_ba_all_steps=true` with `ba_train_timestep_mode=all` trains across the
  full diffusion timestep range. An `inference_ba_region` sampler is a valid
  follow-up ablation, not part of the exact recipe.

## Exact step-zero artifact references

Within `Jul_new_exp/22Jul_debug`, the original four-case immutable bundle is:

```text
experiments/20260722T220046_716752Z__quickB__n3a_fullgrid_up_core_ring_anchor/
```

The all-96 rerun uses experiment ID:

```text
matrix96_n3a_fullgrid_up_core_ring_anchor
```

The 96-case run completed after this handoff was started. Its step-zero result
shows a strong, active initialization: median face MAE `0.09124`, median matched
reference gain `-0.15307`, only 10/96 positive, median landmark movement
`0.02872`, bbox IoU `0.93418`, and outside MAE `0.01422`. Eddie was the only
identity with a positive median (`+0.01541`, 8/12); all seven other identity
medians were negative. Because this is an untrained step-zero architecture,
identity direction is a diagnostic baseline rather than a reason to reject it:
faces remain coherent/detectable in 96/96 cases, are broadly aligned, and show
substantial identity-bearing changes versus PhotoMaker. This makes it the
leading training candidate. Require the fixed 96-image validation plus
matched/wrong-reference controls throughout training and measure whether
optimization turns the recorded identity baseline positive without increasing
alignment drift or losing the visible branch effect.
