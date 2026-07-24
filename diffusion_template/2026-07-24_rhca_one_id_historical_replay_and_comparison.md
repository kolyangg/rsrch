# April 2026 RHCA one-ID replay: provenance, architecture, and comparison

**Prepared:** 24 July 2026
**Historical Comet run:** `rhca_1e-4_ml_step2_allst_trref_diff`
**Replay branch:** `test`
**Historical base commit:** `6782e9d62345fe910633cc8ceec0e2fda6ec2fd1`

## 1. Provenance decision

The best reproducibility base is commit `6782e9d` from 4 April 2026
00:35 BST.

The relevant history is:

| Commit | Time | Relevance |
|---|---:|---|
| `c60e712` | 3 Apr 13:50 | Introduced `train_ba_all_steps` in the model, training entry point, and one-ID config. |
| `aede146` | 3 Apr 13:55 | Removed duplicate nested config keys; no model-behaviour change. |
| `12f900f` | 3 Apr 20:04 | Added the matching RHCA inference path and validation/pipeline support. |
| `6782e9d` | 4 Apr 00:35 | Completed the inference denoising-window support. This is the last model-relevant commit before/while the run was being tested. |
| `40ffcfa` | 4 Apr 01:19 | Only changed Cosmic dataset download scripts. |
| `cadcd7d` | 5 Apr 17:42 | Added the exact `one_id_rhca_diff.yaml` inference manifest after the training run; it confirms checkpoint/config details but is too late to be the training-code base. |

The exact shell command used to start the Comet experiment was not committed.
The replay config is therefore reconstructed from three mutually consistent
sources:

1. the Comet run name;
2. the exact inference manifest added immediately after the run; and
3. the contemporary `one_id_09Feb_testing` config and launchers.

High-confidence settings are:

- `lr_for_lora=1e-4`;
- alternating masked/full diffusion loss with `masked_loss_step=2`;
- `train_ba_all_steps=true`;
- `train_on_separate_image=true`;
- `branched_attn_weight_mode=noise_and_ref`;
- `branched_attn_new_weight_kind=lora`;
- trainable branched cross-attention LoRA;
- all BA sites patched and trained;
- BA-only optimization;
- rank 32;
- PhotoMaker at denoising step 10 and spatial BA from step 15 at inference;
- RealVisXL V4 for the saved-run inference/validation path;
- checkpoint 34 as the later inspected checkpoint.

The 200-step historical epoch length, batch size 4, and 2,000-step warmup are
the most likely contemporary launcher values. For the requested control run,
the replay YAML deliberately uses eight 500-step epochs: exactly 4,000
optimizer steps, with validation and checkpointing every 500 steps. This
schedule change does not touch historical model code.

## 2. Historical approach

### 2.1 Two spatial streams in one doubled U-Net call

Training constructs:

```text
first half:  noised target latent
second half: independently noised reference-image latent
```

The reference noise is held fixed through an inference trajectory. The doubled
batch runs through a U-Net whose self- and cross-attention processors are
replaced by `BranchedAttnProcessor` and `BranchedCrossAttnProcessor`.

With `train_on_separate_image=true`, `OneIDTrain` selects a different image
index as reference. Since the small dataset contains one identity, target and
reference have the same identity but can differ in pose, expression, framing,
and appearance.

### 2.2 Branched self-attention

For the target half:

```text
background:
    Q_target outside target face
    K/V from the target stream

face:
    Q_target inside target face
    K/V from the masked reference-face stream
```

The two outputs are merged by the target face mask. The reference half also
computes ordinary reference self-attention so that a complete reference stream
continues through the U-Net.

The run uses separate rank-32 LoRA projection copies for both target/noise and
reference Q/K/V (`noise_and_ref`). Their base projections are cloned from the
active attention layer, so step zero is intended to start near the inherited
operator while allowing the two streams to specialize.

The old processor hard-codes:

```text
pose_adapt_ratio = 0
ca_mixing_for_face = false
```

Consequently, face K/V come entirely from the reference face; target-face K/V
are not retained as a fallback candidate.

### 2.3 Branched cross-attention

The target and reference halves receive different text-conditioning lanes.
`face_embed_strategy=id` keeps only the PhotoMaker identity-token positions for
the reference/face prompt lane. Branched CA has trainable branch-specific LoRA
in this run.

### 2.4 Temporal and loss recipe

`train_ba_all_steps=true` routes every sampled training timestep through the
doubled branched forward. At inference, the intended 50-step schedule is:

```text
steps 0–9:   text/base path
steps 10–14: PhotoMaker
steps 15–49: spatial branched attention
```

Every second optimizer batch uses face-cropped diffusion MSE; intervening
batches use full-latent diffusion MSE. There is no explicit identity,
counterfactual, null-reference, pose, boundary, or preservation loss.

## 3. Differences from recent configurations

| Configuration | Reference evidence and operator | Reference authority | Main difference from April RHCA |
|---|---|---|---|
| **NN3a** (`reference_minus_null`) | Packed reference ROI enters a low-rank residual connector; reference-minus-null removes target-base leakage. | Bounded gate and RMS cap over an ordinary target/PhotoMaker baseline. | Much safer and causally cleaner, but no absolute reference-face K/V takeover; can remain visually close to PhotoMaker. |
| **N3a_new1** | Restores full-grid legacy spatial BA, disables branched CA, uses reference only in up-block face cores, and anchors output to the base outside the core. | Reference owns the eroded core; target/base owns the exterior. | Closest modern relative. It removes risky trainable split CA and all-layer reference ownership, adds correctness guards and an output anchor. |
| **N3a_new2** | Same full-grid source, but computes target and reference face candidates with a learned per-head dual blend initialized to 35% reference. | Learned fallback between target and reference in up blocks. | April RHCA has no target-face fallback; N3a_new2 explicitly trades identity authority for pose/alignment safety. |
| **NN6a** | No spatial reference lane. Target Q attends two clean PMv2 identity tokens through a low-rank connector at `up_blocks.0.attn1`. | Zero-initialized, gated, RMS-capped identity residual; exact PhotoMaker outside core. | This is an identity adapter inside BA, not the April noised spatial-reference mechanism. It removes pose leakage but strongly compresses identity evidence. |
| **NN7a_init_v2** | Clean PMv2/CLIP spatial patches; sibling-attn2 Q/K/V initialization; local 5×5 candidate attention at `up_blocks.1.attn1`. | Direct target/reference candidate arbitration, warm-started to about 10% reference, capped and anchored outside core. | Preserves spatial evidence but replaces April's unaligned noised full grid with clean local correspondence and a target candidate. |

The important diagnosis is that April RHCA and the recent safe residual runs
answer different questions. April RHCA asks whether the reference stream can
own the face; NN3a/NN6a ask whether a bounded correction can improve an already
dominant PhotoMaker result. Stronger visual change in April is therefore
architecturally expected, not by itself evidence of a training bug in July.

## 4. Obvious issues to keep unchanged for the first replay, then test

The first replay should not silently repair these because that would destroy
the historical control. They should be measured explicitly.

1. **No pose/correspondence model.** Target face queries attend an unregistered
   reference grid. Different reference pose can transfer geometry, hair,
   lighting, expression, or occluders together with identity.
2. **No target-face fallback.** Inside the face mask, reference K/V have
   absolute authority. A poor match cannot defer to target self-attention.
3. **Hard-coded runtime knobs.** The processor ignores configured
   `pose_adapt_ratio` and `ca_mixing_for_face`; both are hard-coded to pure
   reference/no CA mixing.
4. **All-layer intervention.** Every patched self-attention layer receives the
   same bbox-level routing. Early/down/mid layers can overwrite geometry, while
   repeated up-layer intervention can compound seams.
5. **Train/inference timestep mismatch.** Training uses BA at every sampled
   timestep, while inference enables it only after step 15 of 50.
6. **Split branched CA is a confounder.** Both spatial SA and trainable split CA
   change together, so a good or bad result cannot be attributed cleanly.
7. **Loss is not identity-causal.** Diffusion MSE can be minimized without
   proving that swapping the reference changes identity in the intended
   direction.
8. **Mask boundary risk.** Processor masks are binary; a later output-space
   blur does not fully prevent feature-space discontinuities or face/body
   attachment problems.
9. **Historical base ambiguity.** The training model config defaults to SDXL
   base while the saved inference and validation path uses RealVisXL V4.
   Reproduce this first, but record the resolved Hydra config and model IDs.
10. **Different-reference sampling is one-ID-specific.** `OneIDTrain` can pick
    any other sample because all images share one identity. Applying the same
    logic naively to a multi-identity dataset would mix identities.

## 5. Adapting the replay to the NN3a_new1 training dataset

Do this only after the one-ID replay establishes that the historical behavior
is recovered. No model-code change is required for the basic data migration,
but the dataset contract must be preserved.

1. Select the `cosmic_large_neb` dataset used by N3a_new1 instead of `one_id`.
2. Use the same JSON metadata, caption JSON, image root, transformations,
   crop-margin jitter, downscale jitter, and `num_refs=1` as N3a_new1.
3. Keep `train_on_separate_image=true`, but provide a verified
   `same_id_ref_map_json_pth`. `CosmicDoubledTrain` deliberately raises an
   error if a target has no different same-ID candidate.
4. Audit the map: exclude the target path, require identity equality, verify
   paths exist, and report candidate-count distribution. Do not fall back to a
   random global image.
5. Preserve separate `face_bbox` and `face_bbox_ref`; check horizontal flips
   and crop transforms against both coordinate systems.
6. Keep effective batch size and optimizer-step accounting explicit. A
   multi-GPU launch must not accidentally multiply the intended effective
   batch or alter masked-loss cadence.
7. Continue to validate on RealVisXL with the same deterministic manual
   validation set and its matching fixed generation bboxes. Do not use fixed
   RealVis bboxes with SDXL-generated baseline images.
8. Save the fully resolved Hydra config, Git commit, dataset paths/map hash,
   model IDs, and first-batch target/reference filenames.
9. Start with a short causal smoke test: same target/noise with matched,
   swapped, and null reference. Confirm that reference swaps affect the face
   more than the exterior before committing to a long run.

## 6. Tiny follow-up improvements worth isolating

After the exact replay, change one variable per run:

### A. Disable branched CA

Keep April self-attention, masks, all-step training, and optimizer unchanged,
but set `disable_branched_ca=true` and `train_branched_ca_lora=false`.
N3a_new1 suggests this is the smallest way to determine whether split CA caused
color/collage artifacts.

### B. Restrict reference authority to up blocks

Keep full-grid reference K/V and the historical loss, but apply reference face
routing only in up blocks. This imports the least invasive N3a_new1 safety
idea while preserving the core BA equation.

### C. Add a small target candidate

Initialize a fixed or learned target/reference face blend at 20–35% target,
following N3a_new2. This directly tests whether a modest pose fallback removes
misalignment without eliminating the strong April identity effect.

### D. Replace noised reference memory only

At the same sites and with the same ownership rule, compare the historical
noised reference grid with clean PMv2 spatial patches initialized from sibling
attention as in NN7a_init_v2. This is a larger change than A–C, but it isolates
whether reference noise and coordinate entanglement are the main failure
source.

Do not begin with NN6a's two-token identity-only lane: it removes the spatial
mechanism that made the April control scientifically informative.

## 7. Branch contents and use

The branch is an isolated worktree at:

```text
/home/kolyangg/rsrch_apr_test
```

No July model code, analysis folders, downloaded results, saved checkpoints, or
new diagnostic framework has been copied into it. The historical framework is
retained because those modules are runtime dependencies of the run.

Replay files added on top of `6782e9d`:

```text
diffusion_template/src/configs/one_id_rhca_apr2026_replay.yaml
diffusion_template/run_rhca_apr2026_one_id_1gpu.sh
diffusion_template/2026-07-24_rhca_one_id_historical_replay_and_comparison.md
```

Launch from the historical worktree:

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
conda activate photomaker_NS
export COMET_API_KEY=...
bash run_rhca_apr2026_one_id_1gpu.sh
```

This performs step-zero validation and then validation/checkpointing at steps
500, 1,000, ..., 4,000. On the primary server, set its known PhotoMaker path
without editing the YAML:

```bash
PM_PATH=/home/niko/models/PhotoMaker-V2/photomaker-v2.bin \
  bash run_rhca_apr2026_one_id_1gpu.sh
```

For a configuration smoke test without Comet:

```bash
WRITER=console bash run_rhca_apr2026_one_id_1gpu.sh trainer.n_epochs=1 trainer.epoch_len=1
```
