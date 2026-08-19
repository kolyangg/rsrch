---
title: "CL27 additional architecture experiments CL46-CL53"
date: "19 August 2026"
status: "Implementation handoff; no training jobs launched"
branch: "test"
baseline: "exact CL27 r3 at 16k, Comet dbfbf40c3bdd4f70bedc58bda3dfb9cd"
---

# CL27 additional architecture experiments: CL46-CL53

> Branch `test` advanced during this analysis to commit `ae95f390739ba2601baa9ded9ce3f88598a54236`, which already assigns CL38-CL45 to a separate proposal. This addendum therefore uses CL46-CL53 rather than silently reusing experiment IDs. Every arm below is still one scientific delta from exact CL27 r3 at 16k.

## Decision

Keep CL27 r3 16k as the source and control. CL27 scores `0.547260` ID_SIM on fixed-96 versus `0.556580` for controlled PhotoMaker and `0.546311` for the closest CL30-CL37 arm, CL33. CL27 has stronger ownership than PhotoMaker (reported mask IoU `0.9211` vs `0.8652`) but lower face quality (TOPIQ-Face `0.7142` vs `0.7532`). The next experiments should therefore add identity representation or transport capacity rather than another small loss/schedule variant.

Recommended order: **CL47, CL46, CL48, CL49** first; then **CL51, CL50, CL53**; run **CL52** only after its inference-only factorization diagnostic passes.

| Rank | Run | One change from CL27 | Primary target |
|---:|---|---|---|
| 1 | CL47 | Seven reliability-weighted facial-part identity tokens and target-query router | Marion and part-level identity |
| 2 | CL46 | Query-conditioned deformable sampling of reference K/V | Pose/occlusion correspondence |
| 3 | CL48 | Face-local ArcFace-conditioned AdaGN/FiLM in selected ResNet blocks | Persistent identity outside attention |
| 4 | CL49 | Target-conditioned identity semantic completion token memory | Severe occlusion |
| 5 | CL51 | Multi-depth aligned face-feature pyramid with face-size router | Small/action faces |
| 6 | CL50 | Target-only latent landmark geometry tokens | Pose/expression disentanglement |
| 7 | CL53 | Late face-interior Haar high-band reference residual | Face sharpness and TOPIQ-Face |
| 8 | CL52 | Text/identity/interaction-factorized classifier-free guidance | Text-ID conflict |

## Shared contract

1. Resume exact CL27 r3 16k; do not reconstruct it from config and do not use CL27 24k.
2. New flags default false. Every new output projection is exactly zero-initialized; enabled continuation step 0 must match CL27.
3. Freeze all pre-existing CL27 parameters for the first 2k continuation steps. Train only the new module at LR `1e-4`, weight decay `0`.
4. Evaluate continuation `0/250/500/1000/2000/4000` (absolute `16k/16.25k/16.5k/17k/18k/20k`). Select at 18k; do not default to 24k.
5. Run one matched no-change CL27 continuation with the same optimizer reset and data order.
6. A passing arm may unfreeze CL27 BA parameters for the second 2k at `0.10x` new-module LR.
7. Before GPU training require FP32/BF16 step-zero parity, nonzero new-path gradients, state-dict round trip, validation transfer, CFG/replication shape tests, and a two-GPU DDP smoke.
8. Report paired fixed-96 ID_SIM and bootstrap CI, TOPIQ-Face, CLIP/text similarity, mask IoU, face count/no-face rate, Skiing/Crying/Jumping/Dancing/Marion, ordinary-eyewear controls, and face-size quartiles.
9. A negative score is not interpretable until gate mass, residual/native RMS, condition deltas, and gradients prove that the new path is active.

## CL47 - Facial-Part Identity Token Bank

**Delta.** Add residual target-query cross-attention to seven tokens: global, left eye, right eye, nose, mouth, midface, and contour. Build them from aligned PhotoMaker CLIP patches plus the fixed 512-D ArcFace embedding, part position, and reliability. Keep native PhotoMaker CA and CL27 SA unchanged.

**Files.** Add `part_identity_encoder.py` and `part_identity_ca_processor_v1.py`; expose patch features in `model_v2_NS.py`; wire/cache/checkpoint through `branched_runtime.py`, `lora2.py`, and `lora2_helpers.py`; add part-router diagnostics and tests.

**Training.** Frozen target CLIP part features are a cadence-8 stop-gradient teacher at `t<=400`, weight `0.01`; 10% part-token dropout. Promote for positive aggregate ID with Marion and Skiing non-negative; kill/repair on router collapse or ordinary-glasses deletion.

```yaml
model:
  ba_part_identity_tokens_enabled: true
  ba_part_identity_token_groups: [up_blocks.0, up_blocks.1]
  ba_part_identity_parts: [global, left_eye, right_eye, nose, mouth, midface, contour]
  ba_part_identity_token_dim: 2048
  ba_part_identity_rank: 64
  ba_part_identity_router_hidden_dim: 128
  ba_part_identity_gate_init: 0.10
  ba_part_identity_gate_max: 0.20
  ba_part_identity_output_zero_init: true
  ba_part_identity_alignment_weight: 0.01
  ba_part_identity_alignment_cadence: 8
  ba_part_identity_alignment_max_timestep: 400
```

## CL46 - Deformable Reference Correspondence

**Delta.** In mid/up0/up1, map each target face query to a deterministic face-normalized reference point, then predict `K=8` bounded offsets and weights. Sample projected reference K/V with `grid_sample` and inject only a zero-init correction to the dense CL27 message.

**Files.** Add `deformable_reference_processor_v1.py`; compose it in `attn_processor_cleanest.py`; pass masks/bboxes/state in `branched_runtime.py`; register ownership and checkpoint roles in `lora2.py`/`lora2_helpers.py`; add correspondence tests.

**Training.** New module only; retain CL27 losses and add offset smoothness `1e-4`. Promote for `>=+0.003` fixed-96 ID or `>=+0.008` hard-case mean with aggregate `>=-0.001`; kill/repair if offsets remain dead, invalid samples exceed 25%, or residual/native RMS stays below 0.5%.

```yaml
model:
  ba_deformable_reference_enabled: true
  ba_deformable_reference_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_deformable_reference_rank: 64
  ba_deformable_reference_points: 8
  ba_deformable_reference_max_offset_ratio: 0.25
  ba_deformable_reference_gate_init: 0.10
  ba_deformable_reference_gate_max: 0.25
  ba_deformable_reference_progress_start: 0.15
  ba_deformable_reference_output_zero_init: true
  ba_deformable_reference_offset_smoothness_weight: 0.0001
```

## CL48 - Face-Local Identity Adaptive GroupNorm

**Delta.** Add ArcFace-conditioned scale/shift adapters after selected GroupNorms in mid/up0/up1 ResNet blocks. Apply only inside an eroded target-face mask, bound gamma/beta, and RMS-cap the residual. This creates a persistent non-attention identity path.

**Files.** Add `id_adagn.py`; register adapters and identity state in `lora2.py`; extend state/telemetry/checkpoint handling in `lora2_helpers.py` and validation runtime; add parity/leakage tests.

**Training.** New adapters only, no identity reward in the first screen, 10% identity-condition dropout. Promote on positive ID with TOPIQ-Face no worse than `-0.005`; kill on outside-face leakage, bound saturation, or text-similarity loss.

```yaml
model:
  ba_id_adagn_enabled: true
  ba_id_adagn_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_id_adagn_source: arcface_plus_stopgrad_pm_global
  ba_id_adagn_rank: 64
  ba_id_adagn_gate_init: 0.10
  ba_id_adagn_gate_max: 0.12
  ba_id_adagn_gamma_limit: 0.15
  ba_id_adagn_beta_native_rms_cap: 0.10
  ba_id_adagn_face_mask_erosion: 1
  ba_id_adagn_progress_start: 0.20
  ba_id_adagn_output_zero_init: true
  ba_id_adagn_condition_dropout_probability: 0.10
```

## CL49 - Identity-Conditioned Semantic Completion Memory

**Delta.** Predict eight target-conditioned semantic tokens (four geometry, four texture) from frozen reference identity context plus visible target-native face context. Inject them through a separate zero-init residual CA in mid/up0/up1; do not append them to text tokens.

**Files.** Add `identity_semantic_completion.py`; expose clean patch tokens in `model_v2_NS.py`; add teacher extraction/loss in `lora2.py` and `lora2_helpers.py`; cache semantic state in `branched_runtime.py`; add collapse/round-trip tests.

**Training.** Frozen unoccluded target PhotoMaker patches are the stop-gradient teacher, weight `0.05`, cadence 4. Promote for hard-case mean `>=+0.008` with Skiing and Crying both positive; kill on token collapse or generic over-smoothed faces.

```yaml
model:
  ba_identity_semantic_completion_enabled: true
  ba_identity_semantic_completion_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_identity_semantic_num_tokens: 8
  ba_identity_semantic_geometry_tokens: 4
  ba_identity_semantic_texture_tokens: 4
  ba_identity_semantic_decoder_depth: 2
  ba_identity_semantic_heads: 8
  ba_identity_semantic_rank: 64
  ba_identity_semantic_gate_init: 0.10
  ba_identity_semantic_gate_max: 0.15
  ba_identity_semantic_output_zero_init: true
  ba_identity_semantic_teacher_weight: 0.05
  ba_identity_semantic_teacher_cadence: 4
```

## CL51 - Scale-Adaptive Aligned Face Feature Pyramid

**Delta.** Return frozen PhotoMaker vision hidden states from layers 8/16/24 in the existing reference-encoder pass, project each to a fixed 64-token 2-D memory, and route retrieval by target face size, denoising progress, and pooled target-native features.

**Files.** Add `scale_adaptive_face_pyramid.py`; expose selected hidden layers in `model_v2_NS.py`; wire/cache/checkpoint via `branched_runtime.py`, `lora2.py`, and `lora2_helpers.py`; add router and face-size tests.

**Training.** CL27 objectives only; ensure at least 30% of updates have target faces below 256 px. Promote for bottom face-size-quartile ID gain `>=+0.010` with aggregate non-negative; kill if the router ignores size or one level dominates over 90%.

```yaml
model:
  ba_scale_adaptive_face_pyramid_enabled: true
  ba_scale_adaptive_face_pyramid_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_scale_adaptive_face_pyramid_encoder_layers: [8, 16, 24]
  ba_scale_adaptive_face_pyramid_tokens_per_level: 64
  ba_scale_adaptive_face_pyramid_rank: 64
  ba_scale_adaptive_face_pyramid_router_hidden_dim: 128
  ba_scale_adaptive_face_pyramid_gate_init: 0.10
  ba_scale_adaptive_face_pyramid_gate_max: 0.15
  ba_scale_adaptive_face_pyramid_output_zero_init: true
  ba_scale_adaptive_face_pyramid_small_face_threshold_px: 256
```

## CL50 - Latent Landmark Geometry Adapter

**Delta.** Predict 12 grouped landmark heatmaps from native mid-block target features, convert them to confidence-weighted location-encoded geometry tokens, and adapt only target queries in up0/up1. The path never consumes reference appearance.

**Files.** Add `latent_landmark_adapter.py`; pass cached target FaceMesh labels through dataset/collate and `lora2_helpers.py`; wire target-native state through `branched_runtime.py`; add predictor and pose-lock tests.

**Training.** Heatmap focal plus coordinate L1, total weight `0.02`; no identity loss. Promote only when Crying/Jumping/Dancing structure improves, aggregate ID is non-negative, and TOPIQ-Face gains `>=0.005`; kill if the predictor does not beat a canonical prior or faces become pose-locked.

```yaml
model:
  ba_latent_landmark_adapter_enabled: true
  ba_latent_landmark_predictor_group: mid_block
  ba_latent_landmark_injection_groups: [up_blocks.0, up_blocks.1]
  ba_latent_landmark_groups: 12
  ba_latent_landmark_heatmap_size: 32
  ba_latent_landmark_token_dim: 256
  ba_latent_landmark_rank: 64
  ba_latent_landmark_location_pe: fourier
  ba_latent_landmark_gate_init: 0.10
  ba_latent_landmark_gate_max: 0.15
  ba_latent_landmark_progress_start: 0.20
  ba_latent_landmark_output_zero_init: true
  ba_latent_landmark_loss_weight: 0.02
```

## CL53 - Wavelet Identity Detail Residual

**Delta.** Apply fixed Haar filters to reference-face hidden maps, project LH/HL/HH bands into dedicated K/V tokens, and inject a late (`progress>=0.45`) zero-init detail residual only inside an eroded face mask in up0/up1.

**Files.** Add `wavelet_detail_residual.py`; compose it in the attention processor/runtime; register trainables/checkpoint/telemetry in `lora2.py` and `lora2_helpers.py`; add ringing and boundary tests.

**Training.** CL27 losses only in the first screen; no quality reward until clean activation is proven. Promote for TOPIQ-Face gain `>=+0.010` with aggregate ID non-negative; kill on ringing, pose-incompatible texture copying, or a larger identity/quality trade-off.

```yaml
model:
  ba_wavelet_detail_residual_enabled: true
  ba_wavelet_detail_residual_groups: [up_blocks.0, up_blocks.1]
  ba_wavelet_detail_residual_wavelet: haar
  ba_wavelet_detail_residual_bands: [lh, hl, hh]
  ba_wavelet_detail_residual_rank: 64
  ba_wavelet_detail_residual_gate_init: 0.10
  ba_wavelet_detail_residual_gate_max: 0.12
  ba_wavelet_detail_residual_progress_start: 0.45
  ba_wavelet_detail_residual_face_mask_erosion: 1
  ba_wavelet_detail_residual_native_rms_cap: 0.12
  ba_wavelet_detail_residual_output_zero_init: true
```

## CL52 - Modality-Factorized Identity Guidance

**Delta.** Evaluate four conditions per denoising step: neither (`eps00`), text-only (`eps10`), ID-only (`eps01`), and joint (`eps11`). Compose separate text, face-local ID, and interaction deltas. Initialize all scales to standard CFG so step zero is algebraically identical to CL27 guidance.

**Files.** Add `factorized_guidance.py`; implement four-condition batch packing in `branched_runtime.py`; add condition dropout in `lora2.py`; update inference/validation CFG composition; add algebra/parity/leakage tests.

**Training.** Before training, run an inference-only CL27 diagnostic and require a nonzero face-local identity delta. Then use 15% text-only, 15% ID-only, 5% neither dropout; train the tiny schedule MLP first. Promote only for positive aggregate ID with no CLIP/text loss because compute is approximately doubled.

```yaml
model:
  ba_factorized_identity_guidance_enabled: true
  ba_factorized_identity_guidance_condition_dropout_text: 0.15
  ba_factorized_identity_guidance_condition_dropout_id: 0.15
  ba_factorized_identity_guidance_condition_dropout_both: 0.05
  ba_factorized_identity_guidance_schedule_hidden_dim: 32
  ba_factorized_identity_guidance_scale_delta_max: 0.75
  ba_factorized_identity_guidance_face_local_id_delta: true
  ba_factorized_identity_guidance_init_matches_standard_cfg: true
  ba_factorized_identity_guidance_output_zero_init: true
```

## Implementation artifacts and caution

The companion handoff contains eight Hydra configs, eight detailed implementation blueprints, eight two-GPU launch templates, a manifest, and README. Before launch, the coding agent must implement the new schema, fill exact `expected_trainable_contract` counts from a CPU audit, enable the contract, and replace the launch-template resume-key placeholder with the checkpoint override proven in the current training stack. The YAML files are implementation specifications, not executable against the unchanged code.

Do not combine modules in the first screen. Best later pairs are CL46+CL47, CL48+CL53, CL49+CL50, and CL51 plus the best of CL46/CL47.

## Sources

Repository: [LATEST](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/docs/handoffs/LATEST.md), [CL30-CL37 report](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-19_cl30_cl37_completed_results_and_base_decision.md), [prior CL27 plan](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-17_cl27_cl29_vs_cl23_visual_results_and_next_experiments.md), [CL27 config](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml), and [reviewed code commit](https://github.com/kolyangg/rsrch/commit/aaa4e0c17b1e801d0b5cb4a63c13fa14e65b8d26).

Research: [ReSem-Face](https://arxiv.org/abs/2608.04820), [LaTo](https://arxiv.org/abs/2509.25731), [FaceCrafter](https://arxiv.org/abs/2505.15313), [AnyPhoto](https://arxiv.org/abs/2603.14770), [Magic Mirror](https://arxiv.org/abs/2501.03931), [Deformable DETR](https://arxiv.org/abs/2010.04159), [WaveFace](https://arxiv.org/abs/2403.12760), and [Diff-ID](https://arxiv.org/abs/2607.25078).
