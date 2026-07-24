# CosmicLarge next steps: implementation and experiment handoff

**Date:** 24 July 2026

**Branch:** `test`

**Local commit at handoff:** `718bead`

**Scope of this file:** implementation plan only; no model, dataset, loss,
configuration, or launcher code was changed while preparing it

**Detailed diagnosis:** [2026-07-24_cosmic_large_pipeline_performance_analysis.md](2026-07-24_cosmic_large_pipeline_performance_analysis.md)

**Historical replay context:** [2026-07-24_test_branch_one_id_overfit_handoff.md](../2026-07-24_test_branch_one_id_overfit_handoff.md)

## 1. Mission and updated conclusion

The next agent should determine the smallest change that makes the April RHCA
pipeline train stably on CosmicLarge-style references, then turn that result
into a safe full-dataset recipe.

The new leak-free control changes the interpretation of the earlier report:

- `rhca_apr2026_one_id_holdout51_4k` excludes `51.jpg` from both target and
  training-reference sampling while retaining it as the validation reference.
- The user reports that it looks good and has metrics similar to the original
  `one_id` run.
- A direct inspection of its step-4000 images confirms coherent human anatomy.
- Therefore, `51.jpg` overlap is **ruled out as the explanation for the
  one-ID/Cosmic split**. Do not spend more compute repeating the leakage A/B.

The completed Cosmic all-face-loss endpoint also closes the first loss
hypothesis:

- `rhca_apr2026_cosmic_large_one_id_faceonly_8k` has an epoch-8 checkpoint,
  which is exactly 4,000 optimizer steps.
- Its resolved config has `trainer.masked_loss_step=1`.
- Step-4000 images still contain missing, displaced, or merged eyes, noses, and
  mouths. The Reading and Angry prompts are unambiguous failures.
- Thus changing alternating masked/full MSE to face-only MSE is **not
  sufficient**. Do not resume it toward 8k as the default next action.

The immediate next run should isolate branched cross-attention:

```text
Cosmic one-ID
+ face-only loss
+ branched CA disabled
+ CA LoRA not trained
+ every other setting unchanged
+ fresh initialization
+ 4,000 optimizer steps
```

If that reduces global corruption but leaves reference-fragment facial
anatomy, the next one-variable run should freeze the target/noise Q/K/V copies
with `branched_attn_weight_mode=ref_only`. In parallel or immediately after,
use one identity from `large_dataset_adj` to separate the effects of
multi-target diversity and tight 256×256 references.

## 2. Evidence already available on Neb

Neb was inspected read-only with `ssh neb`. At the time of the audit no RHCA
training process or tmux session was active.

The remote checkout was on branch `test` at `bf09b45`, two commits behind the
local `test` branch at `718bead`. Before launching anything new, reconcile the
checkout with a normal fast-forward workflow and preserve the two untracked
automatic-bbox files. Do not delete them:

```text
../dataset_full/cosmic_large_one_id/photomaker_generated_bboxes_auto.json
../dataset_full/cosmic_large_one_id/photomaker_generated_bboxes_auto.json.lock
```

### 2.1 Usable checkpoints

All paths below are on Neb under
`/home/niko/rsrch/diffusion_template`.

| Purpose | Checkpoint | Steps | SHA-256 |
|---|---|---:|---|
| Leak-free successful one-ID control | `saved/rhca_apr2026_one_id_holdout51_4k/checkpoint-epoch8.pth` | 4,000 | `bff146619407d6c3302b2cdeda5b6123eb2a39388e31804c5c0f8a5a0c7f14df` |
| Cosmic face-only failure | `saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/checkpoint-epoch8.pth` | 4,000 | `197aff6f82f898c4f671852c3e780fb3046678e0f420843ab467fa082b9fbf4e` |
| Cosmic alternating-loss failure | `saved/rhca_apr2026_cosmic_large_one_id_1gpu/checkpoint-epoch4.pth` | 2,000 | `526ced97dff0141f93721ca790ba3d10895809bb3859ebfabd6c2627cf1118a3` |
| Original one-ID comparison | `saved/rhca_apr2026_one_id_4k_exact/checkpoint-epoch8.pth` | 4,000 | verify again before use |

Each full checkpoint is approximately 501 MB. Weight-only companions also
exist in the same directories.

Despite the Cosmic face-only run name ending in `8k`, only epochs 1–8 exist.
Its saved config requests 16 epochs, but epoch 8 is the completed 4k decision
point. Do not describe it as an 8k result.

### 2.2 Fixed validation outputs

Use these folders for the baseline visual comparison:

```text
saved/rhca_apr2026_one_id_holdout51_4k/val_images/one_id_val/step_4000_batch_0
saved/rhca_apr2026_cosmic_large_one_id_faceonly_8k/val_images/cosmic_large_one_id_val/step_4000_batch_0
saved/rhca_apr2026_cosmic_large_one_id_1gpu/val_images/cosmic_large_one_id_val/step_2000_batch_0
```

The fixed 12 prompts, seed, validation references, PhotoMaker bboxes, RealVis
validation base, DDIM scheduler, 50 inference steps, PhotoMaker start step 10,
and BA start step 15 are part of the experiment contract.

### 2.3 Dataset locations confirmed during the audit

The full Cosmic files currently visible on Neb are:

```text
metadata:
  /home/niko/datasets/gathered_data_cosmic_large_filtered.json

1024 target root:
  /home/niko/datasets/LAION-5B-Filtered-Large/laion1B-nolang

256 face-reference root:
  /home/niko/datasets/LAION-5B-Filtered-Large-Faces/laion1B-nolang
```

The metadata has 59,143 raw records. Its keys start with
`LAION-5B-Filtered-Large/...`, while `face_paths` start with
`LAION-5B-Filtered-Large-Faces/...`; resolve both against
`/home/niko/datasets` or strip their top-level prefixes deliberately. Add a
preflight that opens examples from both roots rather than inferring path
semantics from strings.

The `large_dataset_adj` comparison files were confirmed locally, but not under
the old expected mount paths on Neb:

```text
/home/kolyangg/rsrch/dataset_full/filtered_ids3_adj.json
/home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset
```

Build the small controlled-identity artifact locally, hash it, and transfer
only that immutable subset to Neb. Do not copy the full large dataset unless
it is actually needed and authorized.

The `test` branch contains the dedicated Cosmic one-ID loader, not the newer
full `CosmicLargeTrain` implementation used in the clean sibling worktree.
Treat a later full-data port as a separate reviewed change; do not copy the
entire sibling `cosmic.py` over the historical replay file.

## 3. Non-negotiable comparability rules

Apply these rules to every arm:

1. Run from `diffusion_template/` so relative Hydra paths resolve.
2. Use the existing `photomaker_NS` environment on Neb.
3. Use one GPU, training batch size 2, rank 32, BF16, LR `1e-4`, 20-step
   warmup, seed 0, and exactly 500 optimizer steps per epoch.
4. Stop and compare at epoch 8 / step 4,000.
5. Validate and checkpoint every 500 steps, including the same step-0
   validation.
6. Do not change prompts, validation reference, bboxes, validation base,
   scheduler, inference steps, or guidance while testing a training variable.
7. Start every experimental arm from fresh initialization:

   ```text
   continue_run=false
   saved_checkpoint=null
   trainer.resume_from=null
   ```

8. Give every arm a new run name. Never write into an existing saved folder.
9. Save the fully resolved Hydra config and the Git commit with the run.
10. Record the first 20 target/reference path pairs. For all controlled
    datasets, assert target path differs from reference path.
11. Treat visual facial anatomy as a hard gate. Identity similarity alone can
    reward a pasted or duplicated facial fragment.
12. Preserve the historical replay launcher and old default behavior. New
    architectural behavior must have explicit config switches and a separate
    launcher.

## 4. Ordered execution plan

```text
Completed face-only Cosmic 4k still malformed
                |
                v
Run face-only + CA-off from scratch for 4k
        |                       |
        | clean/coherent        | still malformed
        v                       v
Confirm causal ref use       Did global drift fall?
on 3 identities                |              |
        |                      yes             no
        v                       |              |
Move to full-data          Run ref-only    Run controlled
pilot + safety fixes       CA-off 4k       data factorial
                                \             /
                                 v           v
                         Controlled reference-format
                         and target-diversity factorial
```

An inference-only checkpoint sweep is useful and should be implemented, but it
must not delay the simple CA-off training arm if only one task can run first.

## 5. Task A — add and run the CA-off 4k launcher

### 5.1 Files

Add:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh
```

Do not edit:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_8k_1gpu.sh
launchers/active/run_rhca_apr2026_cosmic_large_one_id_1gpu.sh
launchers/active/run_rhca_apr2026_one_id_1gpu.sh
```

The new launcher should be a thin wrapper around the current face-only
launcher:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_cosmic_large_one_id_faceonly_8k_1gpu.sh" \
  "disable_branched_ca=true" \
  "train_branched_ca_lora=false" \
  "model.train_branched_ca_lora=false" \
  "$@"
```

Keep `branched_attn_weight_mode=noise_and_ref` in this arm. The purpose is to
change only CA installation/training, not the SA target/reference projection
scheme.

### 5.2 Why all three CA settings are explicit

- `disable_branched_ca=true` prevents `BranchedCrossAttnProcessor` from being
  installed.
- `train_branched_ca_lora=false` prevents CA sites from being selected as
  trainable.
- `model.train_branched_ca_lora=false` makes the nested model setting explicit
  even if Hydra interpolation behavior changes later.

At startup, require the trainable summary to show self-attention processor LoRA
parameters and **zero** `attn2.processor` trainables.

### 5.3 Preflight

Run in an existing `photomaker_NS` environment before the real launch. On a
machine with Conda available, for example:

```bash
bash -n launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh

conda run -n photomaker_NS python train.py \
  --config-name=cosmic_large_one_id_rhca_apr2026_replay \
  trainer.masked_loss_step=1 \
  trainer.n_epochs=8 \
  disable_branched_ca=true \
  train_branched_ca_lora=false \
  model.train_branched_ca_lora=false \
  --cfg job
```

Inspect the composed output and verify:

```text
train_dataset_name: cosmic_large_one_id
trainer.epoch_len: 500
trainer.n_epochs: 8
trainer.masked_loss_step: 1
disable_branched_ca: true
train_branched_ca_lora: false
branched_attn_weight_mode: noise_and_ref
lr_for_lora: 0.0001
lr_scheduler.warmup_steps: 20
dataloaders.train.batch_size: 2
model.rank: 32
```

Also run the smallest dataset smoke check:

- instantiate `cosmic_large_one_id`;
- fetch at least 20 samples;
- assert target is 1024×1024;
- assert reference is 256×256;
- assert all target and reference bboxes are in bounds;
- assert target and reference paths differ;
- print the target/reference/cache keys without printing credentials.

Do not add a permanent test suite solely for this experiment unless requested.
A small read-only verification script or inline preflight is sufficient.

### 5.4 Neb launch

First verify the remote state:

```bash
ssh neb
cd /home/niko/rsrch/diffusion_template
git branch --show-current
git status --short
git pull --ff-only
```

Only fast-forward after confirming that the untracked bbox files do not
conflict with incoming tracked paths. Then launch through the new wrapper from
`diffusion_template/`. Use the normal `.env`-loading path; never echo it.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k \
TRAIN_EPOCHS=8 \
bash launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh
```

### 5.5 Checkpoints and stopping

Inspect step 0, 500, and 1,000 before committing to the full run.

- If all 12 faces are already catastrophically malformed by 1k and the result
  is not visibly better than the CA-on face-only run at matched steps, it is
  acceptable to stop early after saving evidence.
- Otherwise continue through step 4k.
- Do not extend beyond 4k until the 4k comparison is reviewed.

Compare against the CA-on face-only run at every matched step. The conclusive
question is whether removing CA restores anatomy and reduces exterior drift,
not whether the average training MSE is lower.

## 6. Task B — implement an inference-only checkpoint diagnostic

### 6.1 Goal

Existing training code has no clean evaluation-only entry point. Implement a
small tool that loads a saved weight-only or full RHCA checkpoint and runs the
fixed validation set without taking an optimizer step.

Add:

```text
tools/inference/evaluate_rhca_checkpoint.py
```

Prefer reusing the repository's model, pipeline, dataset, and metric
constructors. Do not implement a second denoising pipeline.

### 6.2 Required command-line interface

At minimum:

```text
--config
--checkpoint
--output-dir
--validation-dataset
--guidance-scale
--disable-branched-ca
--validation-base
--processor-base-mode
--reference-condition
--limit
```

Recommended values:

```text
validation-dataset = cosmic_large_one_id_val | one_id_val
processor-base-mode = validation_native | legacy_full_copy
reference-condition = matched | wrong | null
```

Write a `resolved_config.yaml`, command manifest, checkpoint SHA-256, Git
commit, and per-image JSON into each output directory.

### 6.3 Loading order is an invariant

For `validation_native`:

1. Instantiate the model on the requested validation base.
2. Set `disable_branched_ca` **before**
   `prepare_for_training()`/processor installation.
3. Initialize branched processors from that validation model's own Q/K/V.
4. Load only saved trainable LoRA/processor deltas with
   `PhotomakerBranchedLora.load_state_dict_`.
5. Build the pipeline from that model.

Do not copy `BranchLoRALinear.base_weight` or any full processor state from the
SDXL training model into a RealVis validation processor in this mode.

For `legacy_full_copy`, deliberately reproduce the current
`BaseTrainer._evaluation_epoch` behavior and label the output as legacy. This
mode exists only to measure the hybrid-base effect.

The current `patch_unet_attention_processors()` checks `disable_branched_ca`
only while processors are first installed. Flipping the flag after a branched
CA processor is already present does not remove it. The evaluation tool must
set the flag before installation or explicitly rebuild the processor map.

### 6.4 Minimum matrix

Use these checkpoint endpoints:

- Cosmic face-only, epoch 8 / 4k;
- leak-free one-ID, epoch 8 / 4k;
- the new Cosmic CA-off arm, epoch 8 / 4k when available.

Run at least:

| Row | CFG | CA | Processor base | Ref | Purpose |
|---:|---:|---|---|---|---|
| 1 | 5 | on | legacy full copy | matched | reproduce saved validation |
| 2 | 5 | on | validation native | matched | isolate cross-base full-copy effect |
| 3 | 1 | on | validation native | matched | test CFG amplification |
| 4 | 5 | off | validation native | matched | discard trained CA at inference |
| 5 | 5 | off | validation native | wrong | causal reference check |
| 6 | 5 | off | validation native | null | causal reference check |

For the CA-off-trained checkpoint, CA should remain off; do not invent
untrained CA weights and call that a fair CA-on comparison.

Initially, wrong/null may replace both PhotoMaker and spatial-reference input.
Label that as an **end-to-end conditioning intervention**. A stronger later
diagnostic should accept separate PhotoMaker and BA reference images so that
`matched PM + wrong/null spatial BA` can isolate the spatial lane.

### 6.5 Reproduction gate

Before trusting the matrix, row 1 must reproduce the saved validation images
to normal deterministic tolerance under the same CUDA/software stack. At a
minimum verify:

- filenames and prompt mapping match;
- image count is 12;
- seed and generator device match;
- the images are pixel-identical, or document why exact equality is not
  available and show that metrics/visuals match.

## 7. Task C — run CA-off with reference-only trainable projections

Run this only if Task A removes exterior/global corruption but faces still
look like pasted, duplicated, or misregistered reference fragments.

Add:

```text
launchers/active/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k_1gpu.sh
```

It should wrap the Task A launcher and change only:

```text
branched_attn_weight_mode=ref_only
model.branched_attn_weight_mode=ref_only
```

Use:

```text
RUN_NAME=rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k
TRAIN_EPOCHS=8
```

The startup summary must show trainable `attn1.processor.ref_to_*` LoRA
parameters and no `noise_to_*` or `attn2.processor` trainables.

Interpretation:

- improvement over Task A implicates target/noise Q/K/V drift;
- no improvement leaves the raw spatial reference grid, all-layer ownership,
  and lack of target fallback as the leading causes.

Do not combine this run with LR, mask, reference crop, or layer-scope changes.

## 8. Task D — implement the controlled data factorial

### 8.1 Purpose

This factorial holds identity and validation fixed while changing:

1. multiple target views versus one repeated target; and
2. full-scene 1024 references versus tight Cosmic-style 256 references.

It is more informative than another comparison between two different people.

The local comparison dataset is:

```text
metadata: /home/kolyangg/rsrch/dataset_full/filtered_ids3_adj.json
images:   /home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset
```

It contains 2,561 identities. `nm0004960` is a reasonable first candidate:
19 image records exist locally and its median annotated face area is 8.01%.
This is only a candidate; run face-embedding consistency, duplicate, bbox, and
visual checks before freezing the manifest.

### 8.2 Files to add

Suggested layout:

```text
src/datasets/controlled_identity_factorial.py
tools/datasets/build_controlled_identity_factorial.py
src/configs/controlled_identity_factorial_rhca.yaml
launchers/active/run_rhca_controlled_identity_factorial_4k_1gpu.sh
```

Register the dataset in both server/local dataset registries only if the
project still requires separate registries. Keep absolute machine paths out of
committed configs; place machine-local roots in `.env` or use the existing
repository-relative `../dataset_full/...` layout.

### 8.3 Manifest, not implicit sampling

The builder should emit a versioned JSON manifest containing:

```text
identity
source metadata path + SHA-256
source image root
8 training image IDs
1 recurring validation image ID
1 final untouched holdout image ID
target mode
reference mode
per-image face bbox
derived-reference path and bbox
all source/derived file SHA-256 values
selection seed
reject reasons/counts
```

The recurring and final holdouts must not appear as targets or training
references. No target may be its own reference.

Do not let each arm select a new random split. All arms consume the same
checked-in manifest or the same immutable generated artifact.

### 8.4 Dataset contract

The dataset must return the same fields as `CosmicLargeOneID`:

```text
pixel_values
face_bbox
bbox
ref_images
face_bbox_ref
prompts
prompt
original_sizes
crop_top_lefts
target_sizes
identity_id
target_path
reference_path
reference_cache_key
```

Required modes:

```text
target_mode = multi | single
reference_mode = full_scene | cosmic_256
virtual_length = 1000
random_horizontal_flip = identical across arms
```

For `single`, repeat one fixed target through `virtual_length`; still sample
from seven distinct training reference images.

For `multi`, sample the same eight target images uniformly. The reference must
be a different image from the same identity.

For `full_scene`, return the normal 1024×1024 scene and its scene-coordinate
face bbox.

For `cosmic_256`, deterministically:

1. expand the source face bbox by the same fixed 20% margin used by the current
   full Cosmic loader;
2. clamp to image bounds;
3. form a deterministic square crop without changing identity;
4. resize to 256×256 with the explicitly recorded interpolation;
5. save as a derived artifact with a stable filename and quality setting;
6. transform the face bbox into the 256 coordinate system;
7. use a cache key that includes the source hash, crop coordinates,
   interpolation, and output hash.

Do not randomly vary crop margin or JPEG quality in this factorial. The point
is a controlled format intervention.

### 8.5 Three 4k arms

Run with the Task A CA-off, face-only architecture unless Task A itself fails
to improve anything. Keep all other training settings fixed.

| Run suffix | Targets | References | Isolated question |
|---|---|---|---|
| `multi_full` | 8 independent full scenes | another full scene | clean control |
| `single_full` | one repeated full scene | 7 distinct full scenes | cost of losing target diversity |
| `multi_cosref` | same 8 full scenes | deterministic tight 256 crops | cost of Cosmic reference format |

If only two arms can be afforded initially, run `multi_full` and
`multi_cosref`.

Use the same recurring validation reference, prompts, seed, PhotoMaker
baseline, and generation bboxes for every arm. Generate that validation
package once, inspect its face boxes, hash it, and reuse it.

### 8.6 Factorial interpretation

```text
multi_full clean, multi_cosref broken
    => reference crop/resolution/spatial-grid path is causal

multi_full clean, single_full broken
    => target-view diversity is causal

both degraded
    => architecture still unstable independently of Cosmic formatting

all clean under CA-off
    => branched CA was the dominant cause; replicate on more identities
```

## 9. Task E — evaluation and report generation

Add or extend a reporting tool so every 4k arm produces:

```text
report.json
per_image.json
resolved_config.yaml
run_manifest.json
comparison_montage.png
```

Report both per-image values and aggregates. Never hide detector failures by
dropping rows.

### 9.1 Required measurements

- visual anatomy review for all 12 fixed prompts;
- face detector success/failure;
- detected-face count and duplicate-face count;
- identity similarity to the held-out reference;
- identity gain versus the exact PhotoMaker output;
- prompt CLIP similarity;
- generated face bbox size and center;
- landmark geometry/displacement when landmarks are detected;
- outside-face pixel MAE and, if the existing environment supports it, LPIPS
  versus the same-seed PhotoMaker output;
- a narrow ring error around the face-mask boundary;
- matched/wrong/null reference sensitivity, split inside versus outside the
  face.

Use the same mask expansion and bbox source for all arms. Store the mask used
for every measurement.

### 9.2 Promotion gate

Do not promote a run unless all of the following hold:

1. none of the fixed prompts has repeated, displaced, or missing primary
   facial features;
2. face/body attachment is visually plausible;
3. detector success does not regress from PhotoMaker;
4. held-out identity gain versus PhotoMaker is positive;
5. wrong/null interventions show causal reference use;
6. the reference effect is materially stronger inside the face than outside;
7. prompt/scene quality does not materially regress;
8. the result reproduces on at least two additional identities.

Metrics similar to the successful one-ID run do not override an obvious visual
anatomy failure.

## 10. Production fixes after the decisive experiments

Do not implement all of these before Tasks A–D establish causality. When a fix
is implemented, keep the old behavior behind explicit config toggles.

### 10.1 Stop copying training-base processor buffers into RealVis

**Files:**

```text
src/trainer/base_trainer.py
src/model/photomaker_branched/lora2.py
```

Current alternate-base validation first loads the filtered trainable state,
then loops over processors and calls:

```python
v_proc.load_state_dict(t_proc.state_dict(), strict=False)
```

That second copy includes non-trainable `BranchLoRALinear.base_weight` buffers,
so SDXL-initialized Q/K/V can overwrite validation-native RealVis bases.

Implement:

```text
validation_processor_base_mode = legacy_full_copy | validation_native
```

In `validation_native`, initialize processors from RealVis and load only the
saved trainable keys from `model.get_state_dict()["attn_processors"]`. Never
copy full processor state. Preserve `legacy_full_copy` for historical replay.

Verification:

- at step zero, every processor base buffer equals its sibling RealVis
  attention projection;
- saved LoRA A/B deltas load;
- no base buffer changes during delta loading;
- legacy mode reproduces prior validation.

### 10.2 Replace fractional layer order with semantic site selection

**Files:**

```text
src/model/photomaker_branched/branched_runtime.py
src/model/photomaker_branched/lora2_helpers.py
relevant model/config YAML
```

`ba_patch_top_k` and `ba_train_top_k` currently take the first fraction of
processor names. Add an explicit allowlist/pattern selector, for example:

```text
ba_patch_patterns:
  - up_blocks.0.*.attn1.processor
  - up_blocks.1.*.attn1.processor
```

Use exact resolved processor names in the run manifest. Start with up-block
self-attention only; keep CA disabled. The patched and trainable name sets must
be identical unless a config explicitly says otherwise.

Keep the old fractional selector as the backward-compatible default.

### 10.3 Add a target-face fallback inside branched self-attention

**File:**

```text
src/model/photomaker_branched/attn_processor_cleanest.py
```

The current face branch uses target queries with pure reference K/V. Add an
explicit target candidate:

```text
target_face = attention(Q_target_face, K_target_face, V_target_face)
ref_face    = attention(Q_target_face, K_ref_face, V_ref_face)
face_out    = (1 - alpha_ref) * target_face + alpha_ref * ref_face
```

Start a controlled arm with `alpha_ref=0.65` to retain strong reference
authority while giving 35% target geometry fallback. A later learned gate must
be bounded, logged per layer/head, and initialized to the fixed value.

Do not replace branched attention with a generic identity adapter. The target
query and explicit reference K/V path must remain inspectable.

### 10.4 Make cross-attention face-local if it is reintroduced

Current branched CA computes separate global target/reference prompt paths but
does not use the spatial face mask to protect the exterior. If CA is restored:

- compute ordinary CA as the baseline everywhere;
- compute the face-conditioned CA candidate separately;
- merge it only inside the target face mask;
- leave the unconditional CFG lane on ordinary CA by default;
- log the mask and per-lane norms.

Never re-enable trainable global CA in the same run that first tests target
fallback or layer restriction.

### 10.5 Remove the train/inference CFG dead path

**Files:**

```text
src/model/photomaker_branched/lora2.py
src/trainer/sdxl_trainers.py
src/pipelines/photomaker_branched_clean.py
```

`PhotomakerLoraTrainer` computes `do_cfg`, but
`PhotomakerBranchedLora.forward()` immediately deletes it. Decide explicitly:

- either implement conditional dropout/CFG-aware training and document it; or
- remove the misleading trainer cadence and treat training as conditional-only.

At inference add a switch, defaulting off for new work, that prevents
reference/face routing from contaminating the unconditional lane. Verify CFG
1 and CFG 5 with the same seed. Preserve legacy behavior for replay.

### 10.6 Fix mask and face-analysis failure policy

**Files:**

```text
src/model/photomaker_branched/lora2.py
src/model/photomaker_branched/lora2_helpers.py
src/model/photomaker_branched/insightface_package.py
```

For new modes:

- invalid/missing bboxes must not silently become all-ones masks;
- a missing intended face must skip/reject the sample or disable the reference
  branch for that sample, not grant the reference full-image authority;
- when multiple faces are detected, select by bbox overlap or identity
  similarity rather than `faces[0]`;
- never substitute a zero identity embedding without recording the failure;
- keep separate target and reference bboxes through all transforms;
- add a configurable soft boundary and log the final latent-space masks.

### 10.7 Replace raw 4× pixel upsampling as the long-term reference interface

**Files:**

```text
src/model/photomaker_branched/lora2.py
src/model/photomaker_branched/lora2_helpers.py
dataset preprocessing utilities
```

The current `_encode_reference_latent()` bilinearly enlarges a 256×256 tight
crop to 1024×1024 before VAE encoding. The preferred long-term interface is:

1. landmark-align/canonicalize the face crop;
2. encode it at native or modest resolution;
3. expose face-local reference tokens or a local spatial grid;
4. allow target-face geometry as a fallback;
5. restrict reference transfer to explicit face ROI sites.

Before that larger change, use the Task D factorial to prove that tight
256-reference formatting is causal. A target-scale padded-canvas variant can
then be a cheap bridge experiment, but it must carry the correct transformed
reference bbox and should not be confused with true full-scene evidence.

### 10.8 Shorten Cosmic prompts without losing scene supervision

Full Cosmic captions exceed the 77-token CLIP limit far more often than
`large_dataset_adj`. Add a tokenizer-aware prompt builder for future
full-dataset runs:

```text
class + trigger token
essential identity phrase
pose/action
clothing
background/composition
```

Record pre/post token counts and truncation rates. Do not alter prompts in the
one-ID CA/loss controls.

### 10.9 Repair or replace inert full-Cosmic loader options

The newer full `CosmicLargeTrain` in the sibling clean worktree accepts several
legacy flags and then deletes them in `__init__`, including reference and
same-identity controls. Do not assume a Hydra override is active merely
because it composes.

For every future loader knob:

- either implement it and expose the resolved behavior in a sample manifest;
- or remove/reject it with a clear error;
- never silently accept and ignore it.

## 11. Full-dataset pilot only after single-identity promotion

Do not launch the full CosmicLarge job immediately after one good image set.

First:

1. reproduce the promoted recipe on three identities;
2. confirm matched/wrong/null causality;
3. confirm detector and bbox failure policy;
4. record prompt truncation;
5. record target/reference identity consistency and candidate counts;
6. verify reference paths do not cross identity boundaries;
7. run a 100-step overfit/smoke job and inspect first-batch paths/masks;
8. then run a 4k pilot with fixed validation.

Because full CosmicLarge supplies only one target per inferred pseudo-identity,
consider mixing it with `large_dataset_adj` or another multi-view identity
source. Use explicit dataset weights and identity-balanced sampling. Do not
silently let the much larger Cosmic record count dominate every batch.

## 12. Verification checklist for the implementing agent

Before handing the work back:

- [ ] Current branch is `test`; unrelated user changes are preserved.
- [ ] Existing replay launchers are unchanged.
- [ ] New launchers pass `bash -n`.
- [ ] Hydra composition shows the intended one-variable overrides.
- [ ] CA-off startup has zero trainable branched CA parameters.
- [ ] Ref-only startup has no trainable target/noise or CA processor params.
- [ ] Target/reference path inequality is checked at sampling time.
- [ ] Step mapping is recorded: epoch 8 equals 4,000 optimizer steps.
- [ ] Validation still uses the fixed RealVis/DDIM/seed/prompt/bbox contract.
- [ ] Evaluation-only reproduction matches saved validation before sweeps.
- [ ] Processor-native mode does not copy base Q/K/V buffers across bases.
- [ ] Every run has a unique name and fresh initialization.
- [ ] Per-image metrics retain detector failures.
- [ ] All 12 step-4k images receive a visual anatomy review.
- [ ] Conclusions distinguish direct observations, metrics, and hypotheses.
- [ ] No `.env`, token, API key, or machine-specific credential is committed.

## 13. Recommended deliverables

The next agent should return:

1. the CA-off launcher and its preflight evidence;
2. the 4k CA-off saved run and visual/metric comparison;
3. the evaluation-only checkpoint tool and reproduction check;
4. the ref-only run only if the CA-off result meets its trigger condition;
5. controlled-factorial manifests, dataset/config/launcher, and at least the
   `multi_full` versus `multi_cosref` comparison;
6. a dated report that states which hypothesis each result supports or rules
   out;
7. exact run names, commits, configs, checkpoint hashes, and output paths.

The first decision to report is simple:

> Does disabling branched CA make the Cosmic one-ID face anatomically coherent
> by 4,000 steps while leaving the historical self-attention mechanism intact?

That answer determines whether to stabilize the remaining spatial reference
path or revisit the data representation first.
