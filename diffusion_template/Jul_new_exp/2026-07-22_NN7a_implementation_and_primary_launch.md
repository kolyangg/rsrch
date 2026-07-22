# NN7a implementation and primary-server launch

**Date:** 22 July 2026  
**Run:** `ba_NN7a_clean_patch_takeover_up1_1gpu`  
**Approval horizon:** 4,000 optimizer steps

## Scope

This implements **NN7a**, the first ablation in the NN7 proposal:

```text
target self-attention candidate
  + face-cropped clean PhotoMaker-V2/CLIP patch memory
  + bbox-relative local 5x5 reference attention
  + direct, bounded candidate takeover
  + up_blocks.1.attn1 only
```

It deliberately does **not** claim to implement NN7c's landmark/UV registration,
semantic face-part parser, visibility model, or occluder mask. Those components
need a frozen geometry/parsing stack that is not currently present in this
repository. NN7a is the proposal's prescribed test of whether rich clean spatial
memory can produce stronger identity ownership before adding those dependencies.

## Architecture

- The clean reference face is cropped using the existing reference bbox.
- The PhotoMaker-V2 CLIP vision encoder supplies its square patch grid before
  QFormer compression; the CLS token is discarded.
- At each selected `up_blocks.1.attn1` site, target queries attend to a local
  `5 x 5` window in the clean reference grid at the same bbox-normalized
  coordinate.
- The ordinary target self-attention candidate remains the explicit fallback.
- The full-dimensional difference between reference and target candidates is
  used directly. The old rank-16 connector is not on this path.
- A scalar gate starts near `0.02`, is bounded at `0.80`, and the spatial and
  total deltas are RMS-capped at `0.45`.
- Branched cross-attention, pose adaptation, and CA face mixing remain disabled.
- The older reference-U-Net memory and connector behavior remain available via
  the default toggles:

```yaml
model.ba_spatial_memory_mode: reference_unet
model.ba_spatial_mix_mode: connector_residual
```

NN7a enables:

```yaml
model.ba_spatial_memory_mode: clean_clip_patches
model.ba_spatial_mix_mode: direct_candidate_takeover
```

## RealVis and mask guarantees

Validation explicitly uses `SG161222/RealVisXL_V4.0`. It continues to use the
existing manual validation files:

```text
dataset_full/val_dataset/ref_bboxes.json
dataset_full/val_dataset/pm96_bboxes_new.json
```

The masks have three roles:

1. the reference bbox selects the clean face crop and defines the reference
   spatial memory;
2. the target generation bbox creates the feathered/eroded core used by the
   attention takeover;
3. the final combined epsilon is exactly
   `epsilon_PM + M_core * (epsilon_NN7 - epsilon_PM)`, preserving PhotoMaker
   outside the target core.

Thus the combined result remains mask-controlled; NN7a does not apply the clean
reference candidate to the body or scene.

## Validation protocol

- All ordinary in-training validations use the full fixed set of 96 RealVis
  examples.
- After step 4,000, the launcher runs the complete five-condition causal matrix
  (`PM0`, `R1N1`, `R2N1`, `R1N2`, `R2N2`) on a deterministic subset of 24/96.
- Subset seed: `20260722`.
- Original validation indices are retained for bbox lookup and output names:

```text
5, 6, 8, 10, 14, 17, 18, 22, 31, 35, 36, 47,
51, 52, 53, 64, 70, 72, 74, 77, 81, 84, 89, 94
```

This subset is repeatable across future runs as long as the 96-example source
ordering and seed remain unchanged.

## Primary-server launch

```bash
cd /home/niko/rsrch/diffusion_template
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_train_then_diagnose_1gpu.sh
```

The post-training diagnostic defaults to batch size 12. It can be reduced if
memory is tight:

```bash
DIAGNOSTIC_BATCH_SIZE=6 CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_train_then_diagnose_1gpu.sh
```

The standalone checkpoint diagnostic is:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash jul_serv_runs/start_ba_NN7a_checkpoint_reference_vs_noise_24_1gpu.sh \
  saved/ba_NN7a_clean_patch_takeover_up1_1gpu/checkpoint-epoch2.pth
```

## Expected outputs

- Checkpoint: `saved/ba_NN7a_clean_patch_takeover_up1_1gpu/checkpoint-epoch2.pth`
- Diagnostic directory:
  `ppr_NN7a_4000step_realvis_scale1_reference_vs_noise_subset24_seed20260722`
- The diagnostic directory contains images, metrics, tensor/epsilon diagnostics,
  contact sheets, and `manifest.json`.

## Approval decision after 4k

Proceed to NN7b only if R1 to R2 changes the clean spatial candidate and final
inner-face epsilon consistently while N1 to N2 does not change the clean patch
memory. If the branch is causal but creates part crossing or boundary artifacts,
that is the intended evidence for NN7b semantic-part/occlusion restrictions. If
the clean candidate itself remains non-causal or negligible, do not add the
geometry stack yet; first diagnose the gate, cap fraction, candidate direction,
and optimizer manifest.
