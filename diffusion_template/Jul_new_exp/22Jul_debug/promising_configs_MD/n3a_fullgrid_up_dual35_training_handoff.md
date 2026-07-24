# Training handoff: `n3a_fullgrid_up_dual35`

## Status and recommendation

This is the leading step-zero training architecture from the 22/23 July search.
It keeps the useful spatial capacity of N3a, but gives target self-attention
65% ownership and reference self-attention 35% ownership at every head in the
active up-block face route.

Validated at 20 inference steps (`PhotoMaker=4`, `BA=6`):

| set | faces | face MAE vs PM | landmark shift | bbox IoU | outside MAE |
|---|---:|---:|---:|---:|---:|
| diverse 8 | 8/8 | 0.08051 | 0.01219 | 0.97379 | 0.01076 |
| canonical 24 | 24/24 | 0.08121 | 0.01223 | 0.96120 | 0.01199 |
| all 96 | 96/96 | 0.07723 | 0.01134 | 0.95526 | 0.01358 |

For comparison, canonical `n3a_fullgrid_up_core_ring_anchor` on the same 24
cases has face MAE `0.08588`, landmark shift `0.02411`, and bbox IoU `0.94407`.
Thus dual-0.35 retains nearly all visible activity while roughly halving median
landmark movement on 24. On all 96 it reduces landmark movement by about 61%
relative to canonical core-ring N3a (`0.02872` to `0.01134`) and improves bbox
IoU (`0.93418` to `0.95526`), with 96/96 detected faces.

The safety fallback `ba_sa_mix_init: 0.25` also completed 96/96: face MAE
`0.05787`, landmark `0.00732`, bbox `0.96643`, outside `0.01232`. Train it as
the second arm if resources allow, or use it if 0.35 authority grows too fast.

Identity similarity is not a step-zero rejection criterion here. The training
question is whether identity direction improves from this active, coherent
initialization without losing its alignment advantage.

## Provenance

- repository branch: `main_clean`
- commit: `1e88825dc4a325ea1e146be2fa519801f048a73e`
- base config: `src/configs/one_id_ba_NN1a_n3a_replay.yaml`
- model: `src.model.photomaker_branched.lora2.PhotomakerBranchedLora`
- legacy SA processor: `src/model/photomaker_branched/attn_processor_cleanest.py`
- doubled runtime/output anchor:
  `src/model/photomaker_branched/branched_runtime.py`
- experiment spec:
  `Jul_new_exp/22Jul_debug/n3a_fullgrid_dual_specs.json`

The experiment used an inference-only constructor shim for the legacy output
anchor. Do not copy that shim into training. Implement the narrow compatibility
change described below.

## Exact architecture toggles

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
  ba_sa_face_mode: dual
  ba_sa_mix_init: 0.35
  ba_sa_ref_layer_scope: up
  ba_target_core_erode_frac: 0.10
  ba_output_anchor_mode: base_outside_core
```

Step-zero schedule:

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

For a normal 50-step validation, `10/15/50` is the proportional schedule. Keep
the model and validation schedule fields synchronized.

## Recommended opt-in Hydra config

Create a new config rather than changing any existing N3a/NN defaults:

```yaml
# src/configs/one_id_ba_N3a_fullgrid_up_dual35_anchor.yaml
defaults:
  - one_id_ba_NN1a_n3a_replay
  - _self_

disable_branched_sa: false
disable_branched_ca: true
train_branched_ca_lora: false
strict_face_routing: false

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
  train_branched_ca_lora: false
  ba_processor_variant: legacy
  ba_site_policy: all
  ba_sa_train_mode: all

  ba_sa_ref_token_mode: full_grid
  ba_sa_face_mode: dual
  ba_sa_mix_init: 0.35
  ba_sa_ref_layer_scope: up
  ba_target_core_erode_frac: 0.10
  ba_output_anchor_mode: base_outside_core

  ba_correctness_guards: true
  ba_invalid_sample_policy: skip_batch
  ba_strict_processor_restore: true
  ba_train_timestep_mode: all
  ba_face_prompt_attention_mask: false
  ba_uncond_face_fix: true
  ba_face_prompt_mode: id_only
  use_id_loss: false
```

## What is different from canonical core-ring N3a

The reference memory and global safeguards are unchanged:

- full masked reference latent grid;
- target-coordinate face queries;
- reference K/V enabled only in `up_blocks.*`;
- branched cross-attention disabled;
- ordinary PhotoMaker epsilon restored outside the eroded target face core.

Only the face arbitration changes. `core_ring` gives the inner ellipse 100%
reference attention and its ring 100% target attention. `dual` computes both
target and reference face attention for every target face query, then blends
them per attention head:

```python
ref_weight = sigmoid(face_mix_logits)
face = target_face * (1 - ref_weight) + reference_face * ref_weight
```

`ba_sa_mix_init: 0.35` initializes every head to 35% reference ownership. The
processor stores trainable `face_mix_logits`, so training can specialize the
mix by head and layer. Freezing those logits would be a separate ablation and
would not reproduce this recommended trainable architecture.

## Current production compatibility gap

The runtime supports `base_outside_core` for the legacy doubled path, but the
model constructor currently rejects any non-`none` output anchor unless the
processor variant is `packed_residual_v1`. The experiment instantiated legacy
N3a with the anchor temporarily disabled, restored the attribute, and built the
inference pipeline. That proves behavior but is not a training solution.

Make a narrow opt-in compatibility change:

1. Continue accepting only `none` and `base_outside_core`.
2. Allow `base_outside_core` for `legacy` only when the original processor
   registry and target face mask needed by the ordinary PhotoMaker base pass
   are available.
3. Do not relax packed-residual-specific invariants or change any default.
4. Include `ba_output_anchor_mode` and `ba_target_core_erode_frac` in the strict
   legacy architecture/checkpoint manifest.
5. Verify that validation pipeline propagation keeps both values.

The same gap and fix are documented in
`n3a_fullgrid_up_core_ring_anchor_training_handoff.md`; implement it once as a
generic, default-off legacy-anchor capability.

## Training and validation checks

Before a long run:

1. Assert 70 legacy SA processors and zero branched CA processors.
2. Assert `face_mix_logits` exist at every installed SA processor and start at
   `sigmoid(logit) = 0.35` within tolerance.
3. Assert reference attention is enabled only at `up_blocks.*` sites.
4. Assert final epsilon is bit-exact to ordinary PhotoMaker outside the eroded
   target core.
5. Save the resolved architecture and processor manifest with each checkpoint.
6. Reproduce the 24-case step-zero metrics and adjacent-column PDF before
   training.

During training, log per-layer/head mix statistics (min/median/max), face-local
delta strength, outside-core exactness, landmark displacement, bbox IoU, face
detection, and identity diagnostics. A rising identity score is not sufficient
if geometry or face/body attachment degrades.

Validate intermediate checkpoints on all 96 cases. Compare against:

- step-zero dual-0.35;
- step-zero and trained canonical core-ring N3a;
- ordinary PhotoMaker;
- dual-0.25 as the safer fallback.

Stop or reduce reference authority if landmark/bbox drift rises materially,
faces detach from heads/bodies, occluders are overwritten, or mix logits rapidly
saturate toward full reference ownership.

## Reproduction artifacts

- 24-case cross-config PDF:
  `visual_reports/20260723_n3a_canonical_vs_dual25_dual35_24.pdf`
- all96 dual-0.35 PDF:
  `visual_reports/20260723_n3a_fullgrid_dual35_all96.pdf`
- all96 dual-0.25 PDF:
  `visual_reports/20260723_n3a_fullgrid_dual25_all96.pdf`
- diverse-eight spec: `n3a_fullgrid_dual_specs.json`
- progress log: `expanded_study_progress.md`
- result audit: `2026-07-23_recent_run_idea_audit.md`

No production code was changed while creating this handoff.
