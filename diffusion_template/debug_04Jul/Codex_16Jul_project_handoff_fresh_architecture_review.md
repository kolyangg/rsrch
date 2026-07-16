# Branched Attention PhotoMaker: Project Handoff

Date: 16 July 2026

## Purpose

This project modifies PhotoMaker V2 with branched attention (BA) to improve the
generated face's similarity to the reference identity.

The intended outcome is narrow:

- Everything outside the face should remain as close as possible to ordinary
  PhotoMaker.
- The face should become more faithful to the reference identity, visually and
  eventually in identity metrics.
- Pose, face placement, expression compatibility, lighting, color, and image
  quality must remain coherent.

Do not optimize identity score at the expense of visibly broken images. Metrics
are supporting evidence; comparisons of the fixed 96-image validation set are
the primary evidence, unless a metric difference is large and visually credible.

## Current Constraints

- Keep hard target face bounding boxes for now.
- Do not make `POSE_ADAPT_RATIO` the main solution. Prior tests mostly blended
  the original PhotoMaker face back into the BA face and obscured the core issue.
- Preserve an exact or near-exact PhotoMaker path outside the target face.
- Prefer architectural tests over small hyperparameter sweeps.
- Any new behavior must be controlled by config flags so old runs remain
  reproducible.
- Recent validation uses the same 96 images, fixed prompts/seeds, batch size 12,
  validation every 2,000 steps, and an optional 24-image step-zero smoke test.

## Current Architecture

The most successful safe family starts with N29:

1. Run the normal PhotoMaker target path.
2. Build a compact identity memory from the reference.
3. Let target-face queries attend that memory in selected cross-attention layers.
4. Add a zero-initialized BA face residual to the PhotoMaker result.
5. Use a hard epsilon merge so pixels outside the face follow PhotoMaker.

N29 uses two distinct frozen PhotoMaker QFormer identity tokens. This avoided the
pose copying, displaced faces, and severe artifacts of older spatial-reference
BA, but PhotoMaker still appears to dominate and the BA identity contribution is
not yet causally demonstrated.

## Experiment Summary

| Run | Main change | Main visual result | Interpretation |
|---|---|---|---|
| PhotoMaker baseline | No BA | Stable, aligned baseline | Target to beat |
| N3a / early spatial BA | Hard use of spatial reference features | Misalignment, pose/scene leakage, artifacts | Raw spatial reference features contain too much nuisance information |
| N17 | Longer early BA training | Some identity change but stuck/displaced faces | More training preserved architectural errors |
| N24 | Learned dual-gate blending | Safer than hard BA but still inconsistent | Blending incompatible absolute predictions is weak arbitration |
| N27/N28 | Target-aligned residual plus hard PM merge | Good alignment and few artifacts; PM-like identities | Established the current safe residual direction |
| N29 | Two QFormer ID tokens | Best safe BA result so far; subtly different from PM | Compact ID memory works, but is information-limited and weakly supervised |
| N30 | Bbox-normalized QFormer variant | No clear improvement over N29 | Coordinate normalization alone did not solve identity transfer |
| N31 | N29 plus wrong-reference epsilon ranking, 4 GPUs | Stable geometry but later faces desaturate or become nearly grayscale | BA became influential through a shortcut; epsilon distance is not identity |
| N32 | Trainable face-patch resampler | Aligned and artifact-free; face keeps changing through 16k but oscillates | More information reaches BA, but it mixes identity with pose/expression/lighting |
| N33 | Continue N29 to 40k total target | Clean but mostly plateaued through available checkpoints | More steps do not fix N29's memory/supervision bottleneck |

Approximate mean identity scores, for orientation only:

| Run/checkpoint | Mean ID similarity |
|---|---:|
| PhotoMaker | 0.4886 |
| N3a | 0.1709 |
| N24 | 0.3899 |
| N29 10k | 0.4706 |
| N31 12k | 0.4480 |
| N32 16k | 0.4453 |
| N33 24k | 0.4731 |

N32's 16k checkpoint is now complete and does not change the broad conclusion:
the branch remains active and visually safe, but additional training has not
produced consistent movement toward the reference identity.

## Latest Result Locations

- N31: `full_validation_results/ba_identity_dependence_4gpu_N31`
  - Complete checkpoints: 2k, 6k, 10k, 12k.
- N32: `full_validation_results/ba_facepatch_resampler_N32`
  - Complete checkpoints: 2k, 6k, 10k, 16k.
- N33: `full_validation_results/ba_qformer_continue40k_N33`
  - Complete checkpoints: 14k, 20k, 24k, 26k.
- PhotoMaker: `full_validation_results/photomaker_baseline`
- Older failure reference: `full_validation_results/ba_nr_alt_N3a`
- N24: `full_validation_results/ba_dualgate_train_N24_steps`
- N29: `full_validation_results/ba_qformer_idtokens_N29`

Latest report and diagnostic images:

- `full_validation_results/ba_n31_n32_n33_16Jul/full_val_report_N31_N32_N33_vs_key.pdf`
- `full_validation_results/ba_n31_n32_n33_16Jul/N31_N32_N33_closeup_faces_vs_key.png`
- `full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_face_progression.png`
- `full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_full_images.png`
- `full_validation_results/ba_n31_n32_n33_16Jul/visual_statistics.json`

Read first:

- `debug_04Jul/Codex_16Jul_N31_N32_N33_visual_architecture_analysis.md`
- `debug_04Jul/Codex_15Jul_N29_N30_visual_attribution_and_next_architectures.md`
- `debug_04Jul/Codex_15Jul_N27_N28_visual_architecture_analysis.md`
- `debug_04Jul/Codex13_Jul_N25_N26_architecture_analysis.md`

## Relevant Code

- `src/model/photomaker_branched/lora2.py`
  - Model flags, BA input preparation, forward pass, and current ID loss.
- `src/model/photomaker_branched/lora2_helpers.py`
  - Trainable processor installation, BA training inputs, branch forwarding,
    and N31 wrong-reference selection.
- `src/model/photomaker_branched/attn_processor_cleanest.py`
  - Cross-attention implementations and target-face residual path.
- `src/model/photomaker_branched/branched_runtime.py`
  - Hard epsilon merge and runtime BA behavior.
- `src/model/photomaker_branched/identity_memory.py`
  - QFormer and face-patch identity memory construction.
- `src/loss/diffusion_loss.py`
  - Diffusion, masked, and identity-dependence ranking losses.
- `src/trainer/sdxl_trainers.py`
  - Loss composition, validation, and BA diagnostics.
- `src/pipelines/photomaker_branched_clean.py`
  - Inference-time BA schedule and pipeline integration.
- `src/configs/one_id_ba_qformer_idtokens_N29.yaml`
- `src/configs/one_id_ba_identity_dependence_N31.yaml`
- `src/configs/one_id_ba_facepatch_resampler_N32.yaml`
- `src/configs/one_id_ba_qformer_continue20k_N33.yaml`
- `serv_new_runs/start_ba_*`

Evaluation utilities:

- `infer_tools/pdf_full_val.py`
- `infer_tools/full_val_n31_n32_n33_16jul_report.yaml`
- `comet_utils/download_full_validation.py`
- `comet_utils/comet_full_validation_N31_N32_N33.json`

The `photomaker` Conda environment is normally used.

## Findings That Need Independent Verification

Treat these as hypotheses to audit, not settled facts.

1. **N31 optimizes the wrong causal signal.** The correct/wrong ranking compares
   face epsilon residual distances, not decoded identity. The network can satisfy
   it through color, contrast, or expression changes. Its negative selection is
   not clearly identity-label-aware.

2. **The current branch gate may not be useful arbitration.** The scalar
   `face_residual_gate` appears to remain exactly `1.0` in logs. Check parameter
   dtype, gradient magnitude, optimizer inclusion, checkpointing, and whether a
   single scalar can represent layer/head/timestep-dependent trust.

3. **BA may operate at overly coarse UNet resolutions.** Low-resolution
   cross-attention can alter face placement, shape, expression, and color.
   Identity detail may be better restricted to mid/high-resolution up-blocks.

4. **Hard outside-face merging is necessary but insufficient.** It protects the
   background, yet BA remains free to change all face attributes. There is no
   strong separation between identity detail and pose/expression/chroma.

5. **N32 memory is nuisance-rich.** Raw CLIP face patches plus one identity
   embedding may carry pose, crop, illumination, expression, and background
   along with identity. Audit the CLIP resize/crop and bbox mapping carefully,
   especially because reference and target resolutions differ.

6. **N29's two-token memory may be too compressed.** N33 suggests that longer
   training cannot recover information absent from the conditioning memory.

7. **PhotoMaker/BA attribution is unresolved.** Because both paths use
   PhotoMaker-derived identity features, a BA output that differs from baseline
   is not automatically evidence of reference-causal BA identity transfer.

8. **Training and inference schedules may differ subtly.** Reconstruct exactly
   which denoising timesteps and UNet layers use PhotoMaker, BA, or both in
   training and inference.

9. **Mask geometry needs a fresh dimensional audit.** Verify hard bbox scaling,
   latent-space masks, attention-resolution interpolation, non-square inputs,
   reference crops, and off-by-one behavior at every selected UNet resolution.

10. **Current decoded ID loss needs inspection.** Verify crop/alignment,
    recognizer preprocessing, reference target, timestep gating, gradient flow,
    and whether it rewards adversarial texture or conflicts with diffusion loss.

## Recommended Fresh Audit

Before implementing another run:

1. Trace one training batch and one inference sample end to end, documenting
   tensor shapes, masks, identity features, branch injection points, trainable
   parameters, and denoising schedules.
2. Confirm that outside-face predictions are identical to PhotoMaker at the
   merge point and measure whether iterative denoising still leaks changes
   outside the mask.
3. Run same-seed attribution checks on existing checkpoints:
   BA enabled, BA disabled, null identity memory, wrong/swapped reference, and
   correct reference. Compare close-up faces rather than only aggregate scores.
4. Inspect gradients and update magnitudes for memory modules, K/V projections,
   residual gates, and each enabled UNet resolution.
5. Review the official TencentARC PhotoMaker implementation where local behavior
   is ambiguous, especially identity token injection and training loss.
6. Rank concrete bugs separately from architectural limitations, with file and
   line references.

## Promising Directions To Evaluate

These are starting points, not mandatory designs.

### 1. Identity-Causal High-Resolution Residual

Keep N29's target-only residual and hard PhotoMaker merge, but:

- supervise correct, wrong, and null references in decoded identity space;
- select label-aware, same-domain semi-hard negative identities;
- preserve target landmarks, expression, and chroma explicitly;
- inject only at selected mid/high-resolution up-block attention layers;
- use bounded FP32 per-layer or per-head residual gates.

The decisive question is whether swapping the reference changes the face toward
the swapped identity while preserving pose, expression, and color.

### 2. Canonical Face-Part Identity Memory

Replace nuisance-rich spatial patches with landmark-aligned identity tokens:

- canonical eyes, nose, mouth, contour, and global identity tokens;
- multi-scale recognition features rather than raw reference UNet grids;
- initialize as a residual extension of N29's stable QFormer memory;
- combine with the same causal reference supervision and safe target-only merge.

This aims to add identity capacity without reintroducing reference pose and scene
leakage.

The next proposal should define two experiments that are distinguishable within
10k steps, including expected visual signatures and failure criteria. Avoid two
runs that differ only by a small loss weight.

## Expected Next-Session Deliverable

Produce a fresh, code-grounded report containing:

- ranked implementation bugs or inconsistencies;
- an independently reasoned architecture assessment;
- attribution evidence for whether BA carries reference identity;
- two decisive next-run designs with controls and expected outcomes;
- exact files requiring changes, with all new behavior behind flags.

Do not immediately assume that the latest proposed directions are correct. The
main value of the next session is an independent reconstruction of what the code
actually does and why recent runs behave as they do.
