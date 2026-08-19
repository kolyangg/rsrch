# CL38-CL45: eight single-delta successors to CL27 for identity and hard-case quality

**Date:** 19 August 2026  
**Branch:** `test`  
**Evidence commit:** `aaa4e0c17b1e801d0b5cb4a63c13fa14e65b8d26`  
**Control:** CL27 frequency-surface SA-only BA, CL27 r3 at 16k (`ID_SIM=0.547260`)  
**Status:** implementation handoff; no training launched

## Decision

Keep CL27-16k as the base. No CL30-CL37 arm improves aggregate ID against the correct CL27 control. CL33 is closest (`0.546311`, delta `-0.000949`) and raises Skiing, but sometimes does so by deleting identity-owned ordinary glasses. CL35 and CL36 did not sustain their intended identity-reward activity, and CL37’s teacher was too sparsely eligible. The remaining targets are visibility ordering without deletion, Marion, small/action/profile faces, and PhotoMaker’s remaining ~`0.0093` ID advantage.

The suite below preserves CL27 and changes one mechanism per arm. Recommended order: **CL38 → CL42 → CL39 → CL44 → CL43 → CL40 → CL41 → CL45**.

| Priority | Arm | One change | Main target |
|---:|---|---|---|
| 1 | CL38 | visibility objective v2 with owned-region anti-deletion | Skiing/glasses/Marion |
| 2 | CL42 | multi-scale reference K/V pyramid | small/action faces |
| 3 | CL39 | identity-adaptive frequency modulation | whole-panel ID/Marion |
| 4 | CL44 | local Sinkhorn correspondence lane | pose/occlusion correspondence |
| 5 | CL43 | query-adaptive frequency controller | local under/over-injection |
| 6 | CL40 | semantic face-part reference banks | eyes/mouth/shape/Marion |
| 7 | CL41 | canonical landmark transport | profile/action pose mismatch |
| 8 | CL45 | sparse three-expert residual | heterogeneous hard-case tail |

## Common CL27 contract

All YAMLs inherit `CL27_cosmic_frequency_surface_energy_24k` and retain its target-query/reference-KV SA route, Gaussian low/high split, fixed temporal schedule, soft face router, frequency-surface objective in `up_blocks.0/1`, 25% deterministic semantic occlusion, Cosmic distribution, optimizer/LR, 24k horizon, fixed 96-cell validation, `pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. Treat 16k and 18k as mandatory selection gates.

For parameter-bearing arms, use a zero final projection or zero outer gate so step-zero output is exactly CL27. The `-1` trainable-contract values are deliberate fail-closed sentinels: replace them with measured post-installation counts before launch.

## Mandatory implementation/launch gates

1. Add defaults-off parameters in `lora2.py`; install only in declared groups in `branched_runtime.py`/`lora2_helpers.py`.
2. Extend the exact optimizer allowlist, state-dict save/load and strict trainable contract; frozen teachers/landmarks are not optimizer-owned.
3. Prove exact step-zero parity, save/reload parity and disabled-mechanism CL27 parity.
4. Run a 500-step smoke; require non-zero new loss/module activity and non-zero gradient on intended BA/new parameters.
5. Log per-group `RMS(new_delta)/RMS(native)`, activity fraction, gate/router statistics and auxiliary-gradient ratio. Fail before 24k if activity is below the arm-specific floor.
6. Keep fixed validation prompts, seeds, references, bboxes, scheduler, DDIM50, CFG5 and subject-v2 metric. Add object-presence/ordinary-eyewear topology review and face-quality p10.
7. Promotion: no whole-panel ID loss at both 16k and 18k; improvement at two adjacent gates; no object deletion, duplicate face, mask-ownership or prompt-following regression.

## Research basis

The proposals transfer narrow mechanisms from CRAFT (attention-localized identity supervision), AnyPhoto (identity-adaptive modulation and spatial identity control), Beyond Facial Consistency (region/time coordination), MaSC (masked patch identity evaluation), DynamicID (query-level activation), UniversalBooth (hierarchical attention and optimal transport), PersonaCraft (geometry/occlusion), ConsistentID (fine facial localization), FlashFace (spatial identity maps), and InfiniteYou (bounded residual identity injection). They do not import whole pipelines.

## 1. CL38: Visibility-balanced v2 with anti-deletion ownership

**Hypothesis / single delta.** Replace only the occluded-batch reconstruction objective with one region-normalized visibility objective that includes an owned-region presence term. This directly repairs CL33’s useful Skiing signal while closing its ordinary-glasses deletion shortcut.

**Implementation.** No inference module. Extend `visibility_balanced_loss.py` (or add `visibility_balanced_v2_loss.py`), return top/contact/visible/ordinary-eyewear masks from `cosmic_large_adapted.py`, and add a frozen DINOv2 target-presence comparator. The frozen encoder is never optimizer-owned.

**Activity and promotion gates.** Require 20–30% visibility-v2 application, >5% owned-presence application, non-zero BA gradient from the new objective, zero ordinary-eyewear deletions, Skiing >=7/8 topology passes, and aggregate/Marion/Crying no worse than CL27 at 16k and 18k.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl38_*.py` plus shared contract tests.

### Training YAML: `CL38_cosmic_visibility_v2_antideletion_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: visibility-balanced reconstruction v2
# with an explicit owned-object/ordinary-eyewear anti-deletion anchor.
model:
  ba_visibility_v2_enabled: true
  ba_visibility_v2_probability: 0.25
  ba_visibility_v2_visible_weight: 0.75
  ba_visibility_v2_top_weight: 0.25
  ba_visibility_v2_contact_weight: 0.05
  ba_visibility_v2_full_weight: 0.05
  ba_visibility_v2_contact_width: 1
  ba_visibility_v2_sample_on_cpu: true
  ba_owned_presence_enabled: true
  ba_owned_presence_backend: dinov2_vits14
  ba_owned_presence_weight: 0.02
  ba_owned_presence_margin: 0.85
  ba_owned_presence_cadence: 4
  ba_owned_presence_max_timestep: 300
  ba_owned_presence_max_samples_per_step: 1
  ba_owned_presence_stopgrad_target: true
  ba_owned_presence_require_top_or_eyewear: true

# New masks are deterministic labels for the same Cosmic samples; the base
# distribution, semantic-occlusion probability and seed remain unchanged.
datasets:
  train:
    cosmic_large_adapted:
      semantic_occlusion_probability: 0.25
      semantic_occlusion_seed: 150017
      semantic_occlusion_return_region_masks: true
      ordinary_eyewear_mask_enabled: true
      ordinary_eyewear_mask_backend: insightface_face_parsing
      ordinary_eyewear_min_latent_pixels: 2

# CL38 adds frozen teachers/losses only; trainable ownership is exactly CL27.
expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128: {name_substring: ".attn1.processor.", tensors: 840, parameters: 127795200}
    generic_effective_adapter_r32: {name_substring: ".lora_adapter.", tensors: 700, parameters: 30474240}
    photomaker_default_effective_adapter_r64: {name_substring: ".default.", tensors: 700, parameters: 60948480}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_visibility_v2
    - loss_ba_owned_presence
    - ba/visibility_v2_applied_fraction
    - ba/visibility_v2_visible_fraction
    - ba/visibility_v2_top_fraction
    - ba/visibility_v2_contact_fraction
    - ba/owned_presence_applied_fraction
    - ba/owned_presence_target_patch_cosine
    - ba/owned_presence_ordinary_eyewear_fraction
  experiment_comment: >-
    CL38 preserves the exact CL27 inference route and adds only visibility-balanced v2 plus a masked anti-deletion target-presence anchor.
```

## 2. CL42: Multi-scale reference K/V pyramid

**Hypothesis / single delta.** CL27 has only one spatial reference scale. Small/action faces likely need high-frequency reference tokens while larger faces benefit from pooled context. Add one hierarchical K/V mechanism, not a second identity loss.

**Implementation.** In `attn_processor_cleanest.py`, build native, 2x-pooled and 4x-pooled reference face tokens for `up0/up1/up2`; apply scale embeddings and a target-query scale router; merge a zero-gated residual into CL27’s reference lane. Keep native CL27 as exact initialization/fallback.

**Activity and promotion gates.** All three scales must receive >=5% router mass and non-zero gradients; no scale may collapse >90%. Require Jumping and Dancing improvement, TOPIQ-Face p10 no worse, and no aggregate ID loss at 8k/16k.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl42_*.py` plus shared contract tests.

### Training YAML: `CL42_cosmic_multiscale_reference_kv_pyramid_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: hierarchical reference K/V tokens at
# native, 2x-pooled and 4x-pooled scales, blended into the existing reference lane.
model:
  ba_reference_kv_pyramid_enabled: true
  ba_reference_kv_pyramid_groups: [up_blocks.0, up_blocks.1, up_blocks.2]
  ba_reference_kv_pyramid_scales: [1, 2, 4]
  ba_reference_kv_pyramid_max_tokens_per_scale: [256, 64, 16]
  ba_reference_kv_pyramid_projection_rank: 16
  ba_reference_kv_pyramid_use_scale_embeddings: true
  ba_reference_kv_pyramid_mass_normalization: per_scale
  ba_reference_kv_pyramid_scale_priors: [0.60, 0.25, 0.15]
  ba_reference_kv_pyramid_gate_max: 0.25
  ba_reference_kv_pyramid_gate_zero_init: true
  ba_reference_kv_pyramid_min_face_side_px: 0
  ba_reference_kv_pyramid_face_area_conditioned_gate: true

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    reference_kv_pyramid: {name_substring: "ba_reference_kv_pyramid", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - ba/kv_pyramid_gate_mean/up0
    - ba/kv_pyramid_gate_mean/up1
    - ba/kv_pyramid_gate_mean/up2
    - ba/kv_pyramid_mass_scale1/all
    - ba/kv_pyramid_mass_scale2/all
    - ba/kv_pyramid_mass_scale4/all
    - ba/kv_pyramid_small_face_delta_ratio
    - ba/kv_pyramid_parameter_grad_ratio
  experiment_comment: >-
    CL42 adds only a multi-scale reference K/V pyramid to CL27, preserving the same references, face masks, loss and denoising schedule.
```

## 3. CL39: Identity-adaptive frequency modulation

**Hypothesis / single delta.** The fixed CL27 frequency route is strong globally but not identity-specific. Add one shared identity-conditioned modulation network so different identities/channels can receive bounded low/high corrections without changing spatial routing.

**Implementation.** Feed detached PhotoMaker fused-ID embedding, denoise-progress Fourier features and log face area into one shared low-rank hypernetwork. Produce channel-wise low/high gains in `mid/up0/up1`, capped at ±0.20. Zero the final projection so step zero is exactly CL27.

**Activity and promotion gates.** Gain telemetry must leave zero, all selected groups must have non-zero parameter gradients, no correction may saturate, and disabling the module at checkpoint must return CL27. Promote only on whole-panel and Marion gains at adjacent 16k/18k gates.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl39_*.py` plus shared contract tests.

### Training YAML: `CL39_cosmic_identity_adaptive_frequency_modulation_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: zero-initialized identity-conditioned,
# feature-wise modulation of CL27 low/high routed frequency components.
model:
  ba_identity_frequency_modulation_enabled: true
  ba_identity_frequency_modulation_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_identity_frequency_modulation_embedding_source: photomaker_fused_id
  ba_identity_frequency_modulation_detach_id_embedding: true
  ba_identity_frequency_modulation_rank: 64
  ba_identity_frequency_modulation_hidden_dim: 256
  ba_identity_frequency_modulation_progress_fourier_dim: 16
  ba_identity_frequency_modulation_use_face_area: true
  ba_identity_frequency_modulation_max_channel_gain: 0.20
  ba_identity_frequency_modulation_shared_hypernetwork: true
  ba_identity_frequency_modulation_zero_init: true
  ba_identity_frequency_modulation_anchor_weight: 0.0005

# FAIL-CLOSED BLUEPRINT SENTINELS. Replace every -1 with exact counts from the
# mandatory processor-installation smoke before any scientific launch.
expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    identity_frequency_modulation: {name_substring: "ba_identity_frequency_modulation", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_identity_frequency_modulation_anchor
    - ba/id_mod_low_gain_abs_mean/all
    - ba/id_mod_high_gain_abs_mean/all
    - ba/id_mod_gain_abs_max/all
    - ba/id_mod_routed_delta_ratio/all
    - ba/id_mod_parameter_grad_ratio
  experiment_comment: >-
    CL39 adds only a shared identity-conditioned modulation hypernetwork on CL27 frequency components; the final projection is zero-initialized for exact step-zero parity.
```

## 4. CL44: Sinkhorn local correspondence refiner

**Hypothesis / single delta.** Pose and occlusion can make direct spatial reference attention mis-correspond even when masks are correct. Add one local entropic-transport lane to allocate target face queries to reference tokens with balanced marginals.

**Implementation.** At `up0/up1`, project target/reference face tokens to a small transport space; compute log-domain Sinkhorn in fp32 with masked marginals; attend through the transport plan; add a zero-gated residual capped at 0.20. CL27 remains the primary lane.

**Activity and promotion gates.** Synthetic permutation tests must recover known maps; row/column marginal error <1e-3; NaN fallback must be exact CL27; gradient and transport entropy must be in range. Require Marion/profile/action gains without texture-copy or prompt regressions.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl44_*.py` plus shared contract tests.

### Training YAML: `CL44_cosmic_sinkhorn_correspondence_refiner_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: a local entropic-transport correspondence
# lane between target face queries and reference face K/V tokens.
model:
  ba_ot_correspondence_enabled: true
  ba_ot_correspondence_groups: [up_blocks.0, up_blocks.1]
  ba_ot_correspondence_projection_rank: 32
  ba_ot_correspondence_sinkhorn_iterations: 3
  ba_ot_correspondence_epsilon: 0.07
  ba_ot_correspondence_max_face_tokens: 256
  ba_ot_correspondence_column_mass_cap: 2.0
  ba_ot_correspondence_gate_max: 0.25
  ba_ot_correspondence_gate_zero_init: true
  ba_ot_correspondence_entropy_floor: 0.20
  ba_ot_correspondence_entropy_weight: 0.001
  ba_ot_correspondence_deterministic_pooling: true

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    ot_correspondence: {name_substring: "ba_ot_correspondence", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_ot_entropy_floor
    - ba/ot_gate_mean/up0
    - ba/ot_gate_mean/up1
    - ba/ot_transport_entropy/all
    - ba/ot_column_mass_max/all
    - ba/ot_effective_matches/all
    - ba/ot_delta_ratio/all
    - ba/ot_parameter_grad_ratio
  experiment_comment: >-
    CL44 adds only a zero-gated Sinkhorn correspondence lane to CL27 reference attention, with bounded face-token compute and no new dataset or identity reward.
```

## 5. CL43: Query-adaptive closed-loop frequency controller

**Hypothesis / single delta.** CL34 showed that another shared scalar schedule is not enough. Add one token-wise controller that adjusts the existing CL27 low/high scales only where query uncertainty and reference/native disagreement justify it.

**Implementation.** Use disagreement RMS, reduced reference-attention entropy, denoise progress and log face area. A tiny shared MLP outputs bounded additive corrections (low ±0.10, high ±0.15), zero-initialized. It changes the gain field, not the CL27 route or masks.

**Activity and promotion gates.** Corrections must move but not saturate; entropy/disagreement perturbation tests must change outputs monotonically; background correction is exactly zero. Require aggregate ID >=CL27 and no Skiing/quality-tail loss.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl43_*.py` plus shared contract tests.

### Training YAML: `CL43_cosmic_query_adaptive_frequency_controller_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: token-wise bounded corrections to the
# fixed CL27 low/high schedule from query confidence and denoising state.
model:
  ba_query_frequency_controller_enabled: true
  ba_query_frequency_controller_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_query_frequency_controller_hidden_dim: 64
  ba_query_frequency_controller_input_features: [target_norm, reference_native_disagreement, reference_attention_entropy, denoise_progress, log_face_area]
  ba_query_frequency_controller_max_low_correction: 0.20
  ba_query_frequency_controller_max_high_correction: 0.25
  ba_query_frequency_controller_spatial_smoothing_kernel: 3
  ba_query_frequency_controller_zero_init: true
  ba_query_frequency_controller_mean_anchor_weight: 0.001
  ba_query_frequency_controller_total_variation_weight: 0.0005

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    query_frequency_controller: {name_substring: "ba_query_frequency_controller", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_query_frequency_anchor
    - loss_ba_query_frequency_tv
    - ba/query_freq_low_correction_mean/all
    - ba/query_freq_high_correction_mean/all
    - ba/query_freq_correction_abs_max/all
    - ba/query_freq_entropy_correlation/all
    - ba/query_freq_small_face_gain/all
    - ba/query_freq_parameter_grad_ratio
  experiment_comment: >-
    CL43 adds only a query-adaptive bounded controller around CL27 fixed frequency scales; a zero final layer gives exact CL27 at step zero.
```

## 6. CL40: Semantic face-part reference bank

**Hypothesis / single delta.** Global reference-face attention can average away identity-defining local parts. Add one semantic part-bank mechanism to preserve eye/brow, nose/cheek, mouth/chin and contour/forehead details.

**Implementation.** Use cached InsightFace-106 landmarks to partition reference tokens into five banks. A target-query part router mixes part-specific reference messages in `up0/up1`; its outer gate is zero-initialized. Target landmark labels supervise the router only in training; inference needs reference landmarks only.

**Activity and promotion gates.** Landmark coverage >95%; part-router synthetic accuracy >90%; each part gets >=5% mass and non-zero gradient; no eye-bank collapse. Require Marion and low-ID-tail gains with object topology intact.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl40_*.py` plus shared contract tests.

### Training YAML: `CL40_cosmic_semantic_face_part_bank_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: part-balanced reference attention using
# landmark-derived reference banks and a target-query part router.
model:
  ba_face_part_bank_enabled: true
  ba_face_part_bank_groups: [up_blocks.0, up_blocks.1]
  ba_face_part_bank_parts: [left_eye_brow, right_eye_brow, nose_cheeks, mouth_chin, contour_forehead]
  ba_face_part_bank_reference_landmark_backend: insightface_106
  ba_face_part_bank_router_hidden_dim: 128
  ba_face_part_bank_router_rank: 32
  ba_face_part_bank_attention_temperature: 0.70
  ba_face_part_bank_min_reference_mass_per_part: 0.05
  ba_face_part_bank_gate_max: 0.25
  ba_face_part_bank_gate_zero_init: true
  ba_face_part_bank_alignment_probability: 0.25
  ba_face_part_bank_alignment_weight: 0.02
  ba_face_part_bank_sample_on_cpu: true

# Target landmark labels supervise only the router during training. Inference
# uses the learned query router and requires landmarks only for the reference.
datasets:
  train:
    cosmic_large_adapted:
      return_reference_face_landmarks_106: true
      return_target_face_landmarks_106: true
      face_landmark_cache_enabled: true

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    face_part_bank: {name_substring: "ba_face_part_bank", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_face_part_alignment
    - ba/face_part_gate_mean/up0
    - ba/face_part_gate_mean/up1
    - ba/face_part_router_accuracy
    - ba/face_part_min_mass/all
    - ba/face_part_entropy/all
    - ba/face_part_parameter_grad_ratio
  experiment_comment: >-
    CL40 adds only a semantic face-part reference bank and query router on top of CL27; ordinary CL27 attention remains the zero-gate initialization.
```

## 7. CL41: Canonical landmark transport

**Hypothesis / single delta.** Large reference/target pose mismatch can force attention to solve geometry and identity simultaneously. Add one pose-normalization/transport lane before reference K/V.

**Implementation.** Cache 106-point landmarks and deterministic PnP pseudo-labels. Predict a bounded affine+TPS warp from reference to target/canonical coordinates, sample reference features, run reference K/V, and blend with a zero gate. Add pose/grid/validity losses; invalid grids fall back exactly to CL27.

**Activity and promotion gates.** Synthetic warp recovery <0.5 latent cell, valid-grid fraction >99%, no folding, non-zero gate/pose gradients. Require profile/action gains and no prompt-pose copying.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl41_*.py` plus shared contract tests.

### Training YAML: `CL41_cosmic_canonical_landmark_transport_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: a canonical landmark transport lane that
# pose-aligns reference features before K/V attention and blends it at zero gate.
model:
  ba_landmark_transport_enabled: true
  ba_landmark_transport_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_landmark_transport_reference_backend: insightface_106_pnp
  ba_landmark_transport_warp_kind: affine_tps
  ba_landmark_transport_control_points: 9
  ba_landmark_transport_pose_head_hidden_dim: 128
  ba_landmark_transport_projection_rank: 32
  ba_landmark_transport_gate_max: 0.20
  ba_landmark_transport_gate_zero_init: true
  ba_landmark_transport_pose_loss_weight: 0.01
  ba_landmark_transport_grid_loss_weight: 0.01
  ba_landmark_transport_validity_weight: 0.002
  ba_landmark_transport_supervision_probability: 0.50

# 106-point landmarks plus a deterministic PnP fit provide reference/target
# pseudo-labels; no new data distribution is introduced.
datasets:
  train:
    cosmic_large_adapted:
      return_reference_face_landmarks_106: true
      return_target_face_landmarks_106: true
      return_face_pose_pnp: true
      face_landmark_cache_enabled: true

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    landmark_transport: {name_substring: "ba_landmark_transport", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_landmark_transport_pose
    - loss_ba_landmark_transport_grid
    - loss_ba_landmark_transport_validity
    - ba/landmark_transport_gate_mean/all
    - ba/landmark_transport_pose_mae/all
    - ba/landmark_transport_grid_valid_fraction/all
    - ba/landmark_transport_delta_ratio/all
    - ba/landmark_transport_parameter_grad_ratio
  experiment_comment: >-
    CL41 adds only a zero-gated canonical landmark transport lane; CL27 remains the exact initial and fallback route.
```

## 8. CL45: Sparse hard-case mixture-of-experts residual

**Hypothesis / single delta.** Crossed slice results may indicate that one transform cannot serve visibility, geometry and identity-detail failures. Add one sparse conditional-capacity mechanism around the existing CL27 message.

**Implementation.** Create three low-rank residual experts (identity detail, geometry low band, visibility safe). A top-2 router uses target state, disagreement, entropy, time and face area. All expert output projections and the outer gate are zero-initialized; outputs remain face-mask bounded.

**Activity and promotion gates.** Every expert must receive 10–70% mass, non-zero gradients and distinct outputs; dropped tokens <1%; expert ablations must attribute gains. Stop on expert collapse, duplicates or whole-panel loss.

**Primary files/tests.** `attn_processor_cleanest.py`, `lora2.py`, `lora2_helpers.py`, `branched_runtime.py` as applicable; add `tests/test_cl45_*.py` plus shared contract tests.

### Training YAML: `CL45_cosmic_sparse_hardcase_moe_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Single scientific delta versus CL27: a sparse three-expert residual adapter on
# the existing CL27 routed message, with a query-conditioned top-2 router.
model:
  ba_hardcase_moe_enabled: true
  ba_hardcase_moe_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_hardcase_moe_experts: [identity_detail, geometry_lowband, visibility_safe]
  ba_hardcase_moe_num_experts: 3
  ba_hardcase_moe_top_k: 2
  ba_hardcase_moe_expert_rank: 16
  ba_hardcase_moe_router_hidden_dim: 64
  ba_hardcase_moe_router_features: [target_norm, reference_native_disagreement, reference_attention_entropy, denoise_progress, log_face_area]
  ba_hardcase_moe_gate_max: 0.20
  ba_hardcase_moe_zero_init: true
  ba_hardcase_moe_load_balance_weight: 0.005
  ba_hardcase_moe_router_z_loss_weight: 0.001
  ba_hardcase_moe_expert_orthogonality_weight: 0.001
  ba_hardcase_moe_capacity_factor: 1.25

expected_trainable_contract:
  enabled: true
  total_tensors: -1
  total_parameters: -1
  optimizer_tensors: -1
  optimizer_parameters: -1
  categories:
    hardcase_moe: {name_substring: "ba_hardcase_moe", tensors: -1, parameters: -1}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/frequency_surface_top_high_rms/up0
    - ba/frequency_surface_top_high_rms/up1
    - ba/frequency_surface_top_low_rms/up0
    - ba/frequency_surface_top_low_rms/up1
    - ba/frequency_surface_visible_ratio/up0
    - ba/frequency_surface_visible_ratio/up1
    - ba/frequency_surface_applied_fraction
    - loss_ba_moe_load_balance
    - loss_ba_moe_router_z
    - loss_ba_moe_expert_orthogonality
    - ba/moe_gate_mean/all
    - ba/moe_expert_identity_fraction/all
    - ba/moe_expert_geometry_fraction/all
    - ba/moe_expert_visibility_fraction/all
    - ba/moe_router_entropy/all
    - ba/moe_dropped_token_fraction/all
    - ba/moe_parameter_grad_ratio
  experiment_comment: >-
    CL45 adds only a zero-initialized sparse hard-case expert adapter around the CL27 message; all experts are mask-bounded and the native CL27 path remains intact.
```

## Execution sequence

- Phase A: shared validator, then CL38; implement/profile CL42 and CL39 in parallel; run each as an independent cold start.
- Phase B: CL44 and CL43. Run CL40 only after landmark cache/router tests, and CL41 only after synthetic warp tests.
- Phase C: CL45 only if earlier arms remain crossed by hard-case family.
- Do not combine winners during this screen. The first composition test, after isolated promotion, is CL38 plus one architecture winner.

## Source record

Internal: `docs/handoffs/LATEST.md`; `analysis/2026-08-19_cl30_cl37_completed_results_and_base_decision.md`; `analysis/2026-08-17_cl27_cl29_vs_cl23_visual_results_and_next_experiments.md`; CL27/CL23/CL19 configs; `attn_processor_cleanest.py`; `lora2.py`; `lora2_helpers.py`.

Primary papers: CRAFT (arXiv:2608.14403); AnyPhoto (arXiv:2603.14770); Beyond Facial Consistency (arXiv:2607.25622); MaSC (arXiv:2605.22469); DynamicID (arXiv:2503.06505); UniversalBooth (ICCV 2025); PersonaCraft (ICCV 2025); InfiniteYou (arXiv:2503.16418); ConsistentID (arXiv:2404.16771); FlashFace (arXiv:2403.17008).

## Final recommendation

Implement **CL38 first**, then **CL42 and CL39**. CL38 is the best evidence-led repair; CL42 has the strongest small-face causal rationale; CL39 has the largest plausible whole-panel identity upside at low overhead. CL44 is the next high-upside correspondence experiment. Keep CL40/41/45 behind strict mechanism gates.
