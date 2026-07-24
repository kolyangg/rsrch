# Test-branch RHCA one-ID overfit handoff

Date: 2026-07-24  
Working tree: `/home/kolyangg/rsrch_apr_test`  
Git branch: `test`  
Project directory: `/home/kolyangg/rsrch_apr_test/diffusion_template`

## Purpose of this branch

This branch is a deliberately narrow diagnostic environment for the older April 2026 branched-attention implementation. The immediate goal is not to develop another large architecture variant. It is to establish a trustworthy positive control:

> Can the historical RHCA/branched-attention model overfit one identity, visibly transfer reference identity into the generated face, and avoid unnecessarily changing the pose, clothes, and background?

The wider research goal is to improve PhotoMaker identity similarity using branched attention while retaining PhotoMaker's composition, pose alignment, visual quality, and low artifact rate.

Recent `main_clean` experiments became architecturally complicated and often produced images very close to the original PhotoMaker output. The `test` branch therefore replays the older configuration that had produced visibly different faces, then changes one factor at a time.

## How to work in this repository

- Use the `test` branch only from the dedicated worktree:

  ```bash
  cd /home/kolyangg/rsrch_apr_test/diffusion_template
  git branch --show-current
  ```

  The branch should print `test`.

- Continue unrelated/current architecture work on `main_clean` from `/home/kolyangg/rsrch`. A Git branch can only be checked out in one worktree at a time; this is why the separate `rsrch_apr_test` directory exists.

- Run training commands from the `diffusion_template` directory.

- Active launchers are in `launchers/active/`. Older scripts and research material are grouped elsewhere and should not be treated as active configurations.

- Machine-specific paths and credentials belong in `.env`. New Comet experiments should use the `rsrch-jul` project.

- Do not merge newer architecture code into this branch casually. The historical replay launcher verifies hashes for important architecture files so an “exact replay” cannot silently become a different model.

- Do not commit or push unless explicitly requested. Before changing anything, inspect:

  ```bash
  git status --short
  git diff
  git diff --cached
  ```

## Historical positive control

The target historical run is:

`rhca_1e-4_ml_step2_allst_trref_diff`

The launch-time source was traced to commit:

`aede146e2e2a2dae1cb3d14a0ea5daed25ae9604`

The current replay launcher is:

`launchers/active/run_rhca_apr2026_one_id_1gpu.sh`

The comparable recent replay is named:

`rhca_apr2026_one_id_4k_exact`

Its central configuration is:

- doubled U-Net batch containing a noisy target and independently noised reference;
- branched self-attention at all selected U-Net sites;
- target background uses target/noise features;
- target face queries attend to reference-face keys and values;
- spatial merging uses the target face mask;
- `noise_and_ref` trainable LoRA projection copies;
- branched cross-attention remains enabled;
- BA-only optimization;
- `lr_for_lora=1e-4`;
- rank 32;
- batch size 2;
- warmup 20 steps;
- `trainer.masked_loss_step=2`, alternating face-masked and full-image diffusion loss;
- training on a separate same-identity reference image;
- inference schedule: base/text at steps 0–9, PhotoMaker at 10–14, branched attention at 15–49;
- RealVisXL V4 for validation.

The old one-ID dataset contains 19 target images with different prompts, poses, and backgrounds. A different same-identity image can be sampled as the reference. This run is the present positive control: it demonstrates that the historical branch can visibly affect faces. It is not a perfect final method—face alignment, artifacts, and mask-boundary behavior can still be poor—but its behavior proves that the BA path is active and trainable.

## New Cosmic one-ID test

Dataset:

`../dataset_full/cosmic_large_one_id`

Original launcher:

`launchers/active/run_rhca_apr2026_cosmic_large_one_id_1gpu.sh`

This dataset contains:

- one repeated training target image;
- eight training reference images of the same identity;
- a validation holdout and a separate final-only holdout;
- prompts for a woman rather than the man used in the old one-ID test.

The resolved model, optimizer, loss, and inference architecture of this launcher were checked against the old one-ID launcher. Apart from the selected datasets, run metadata, and endpoint, they use the same training approach.

However, the observed Cosmic run changes the background much more than expected. That does not currently look like proof of an implementation mismatch. It is plausibly caused by a major dataset/objective difference:

- the old dataset spreads full-image loss over 19 varied target images;
- Cosmic repeats one target, including the same clothes, pose, and background;
- with `masked_loss_step=2`, half of the optimizer steps use full-image loss;
- `noise_and_ref` adapters and branched cross-attention can influence locations outside the face.

Therefore the model receives a strong and repeated incentive to memorize the single target's whole image. Branched attention is spatially merged around the face, but the overall training configuration is not mathematically equivalent to freezing all non-face output.

The recently added faster conditioning/cache path is not the leading suspect: its outputs were checked for equivalence, and validation still uses the original generation path.

## Current controlled experiment

Launcher:

`launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_8k_1gpu.sh`

Run name:

`rhca_apr2026_cosmic_large_one_id_faceonly_8k`

This starts from scratch for 8,000 steps and changes one main variable:

```text
trainer.masked_loss_step=1
```

This makes every optimizer step use face-masked diffusion loss instead of alternating face and full-image losses. It tests whether direct full-image supervision is the main reason for the Cosmic background drift.

This is an isolation experiment, not a guaranteed solution. Self-attention is global, and trainable adapters outside the face can still change the exterior even when the loss is measured only inside the face.

## Central unresolved questions

1. Does the April RHCA model reliably overfit identity, or did the original result depend on the structure of the old 19-image dataset?
2. Does the Cosmic setup fail because it has only one repeated target rather than because its identity/reference images are unsuitable?
3. Is alternating full-image loss the main source of Cosmic background drift?
4. With face-only loss, does reference identity improve while pose, body, clothes, and background remain stable?
5. If exterior drift remains, is it caused mainly by global `noise_and_ref` self-attention adapters, branched cross-attention, mask boundaries, or another pathway?
6. Are comparisons fair? Old and Cosmic runs currently use different validation identities, references, and bounding-box files even though both use RealVis and related prompt templates.

## Recommended evaluation sequence

Compare the original Cosmic run and the face-only Cosmic run at matched steps: step 0, 500, 1k, 2k, 4k, and 8k where available.

For each checkpoint:

- use fixed prompts, seeds, references, and manual bounding boxes;
- compare against that run's own step-0 PhotoMaker baseline;
- inspect full images and aligned face crops;
- measure identity similarity inside the face;
- measure LPIPS or MAE outside the target mask;
- inspect the face-mask boundary for seams, duplicated facial parts, glasses/hair corruption, and pose mismatch;
- run reference-swap and null-reference controls with target noise held fixed;
- log first-batch target/reference filenames, prompt, boxes, and mask coverage;
- verify that `masked_loss_step=1` really produces face-only loss on every training batch.

Do not infer too much from raw identity similarity alone. The desired result must also preserve composition and reduce artifacts.

## Decision tree after the face-only run

- If face identity changes appropriately and exterior drift is much lower, keep the historical architecture and investigate the face-only objective on a larger, diverse target set.

- If identity changes but the exterior still drifts, isolate one architectural path at a time. The first candidates are freezing the target/noise background adapters or disabling branched cross-attention. These are architecture changes and should not be mixed into the current control run.

- If the face barely changes, verify gradients, trained parameter names, masks, checkpoint deltas, and reference sensitivity before proposing a new architecture.

- If the old dataset works but both Cosmic objectives fail, build a Cosmic-format one-ID dataset with multiple full target images. Each target should sample another training image of the same identity as its reference, while validation holdouts remain excluded. This is the cleanest dataset-level match to the old positive control.

## Runtime and checkpoint notes

- The one-ID epoch length is 500 optimizer steps.
- Validation and checkpointing occur every 500 steps.
- Checkpoint epoch `N` corresponds to approximately `N × 500` steps.
- `TRAIN_EPOCHS` is the total training endpoint, not the number of extra epochs after resume.
- Full trainer checkpoints require `torch.load(..., weights_only=False)` under newer PyTorch versions because they include optimizer state and Hydra/OmegaConf objects. The compatibility fix is in `src/trainer/base_trainer.py`.
- The quality-neutral speedups avoid redundant frozen conditioning work but intentionally do not change target VAE sampling, augmentation RNG, or validation generation.

## Relevant documentation

- `docs/experiments/2026-07-24_rhca_one_id_historical_replay_and_comparison.md`
- `docs/experiments/2026-07-24_quality_neutral_runtime_optimizations.md`
- `../dataset_full/cosmic_large_one_id/README.md`
- `README.md`

## Handoff priority

The next agent should first finish and compare the scratch face-only Cosmic run against both the alternating-loss Cosmic run and the old one-ID positive control. Do not introduce another broad architecture variant until this comparison establishes whether the current problem is primarily dataset diversity, the full-image objective, or an actual failure of the BA/reference pathway.
