# NN3a_new1 one-ID training improvement plan

Date: 2026-07-23  
Status: **approved and active — identity 1 selected**  
Production training entrypoint: `jul_serv_runs/start_ba_N3a_new1_1gpu.sh`  
Working boundary: all new configs, code, datasets, logs, checkpoints, metrics,
images, and reports will be written only under `Jul_new_exp/23Jul_debug`.

## 1. Objective

Use `NN3a_new1` as the immutable architectural starting point and find a
training recipe that:

1. preserves its useful step-zero branched-attention face;
2. learns reference identity faster than the present training run;
3. does not drift back toward ordinary PhotoMaker identity at later
   checkpoints;
4. retains face geometry, face/body attachment, and the PhotoMaker scene;
5. demonstrates causal reference control rather than merely creating a
   different texture or expression.

The fast screen will deliberately overfit one well-curated identity for 600
optimizer steps. This is an optimization and architecture-routing experiment,
not a generalization claim.

## 2. Exact starting architecture

The control is the current `one_id_ba_N3a_new1` configuration:

```yaml
disable_branched_sa: false
disable_branched_ca: true
strict_face_routing: false

train_ba_all_steps: true
train_ba_only: true
branched_attn_weight_mode: noise_and_ref
branched_attn_new_weight_kind: lora
ba_noise_lr_scale: 0.25
loss_kind: masked_alternating

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

The core logic will remain fixed unless an experiment explicitly says
otherwise:

- doubled target/reference U-Net execution;
- target-coordinate queries;
- full-grid, face-masked reference memory;
- reference K/V ownership only in the inner `0.68` face core;
- target attention in the surrounding face ring;
- reference injection only in up blocks;
- no branched cross-attention;
- ordinary PhotoMaker epsilon restored outside the eroded output core.

No proposal below re-enables branched cross-attention, replaces PhotoMaker as
the global scene generator, or introduces a broad full-image BA residual.

## 3. Diagnosis motivating the search

The current live run provides several concrete clues:

- It installs 70 legacy self-attention processors and optimizes 840 tensors,
  approximately 31.95M parameters.
- Only 36 processors are in `up_blocks` (30 in up0 and 6 in up1), where
  reference ownership is actually enabled. The remaining 34 down/mid
  processors still carry trainable target/reference clones and participate in
  reference-branch propagation even though they cannot directly select
  reference attention for the target face.
- Reference and noise projections are both trained. The reference group uses
  LR `5e-5`; the noise group uses `1.25e-5`.
- `masked_alternating` produces highly heterogeneous scalar losses because
  every second step is a face-only crop and the intervening step is a
  full-image mean. Earlier July work showed that face-only steps can push
  trainable branch pathways without a preservation anchor.
- The output anchor confines useful gradients to a small face core. A
  full-image mean therefore dilutes the optimization signal by the core area,
  which is a plausible cause of slow early learning.
- Historical evidence says the untrained N3a route can be better than trained
  checkpoints. Plain denoising MSE does not explicitly reward retaining the
  initial reference-specific face.

The current live timing is also important:

- training: about 35.6 GB VRAM and approximately 1.4 seconds/step;
- 2,000 training steps: approximately 47 minutes;
- full 96-image validation at batch 12: approximately 76 GB and 15 minutes.

These values make concurrent execution possible only after a measured
small-validation memory calibration. Free VRAM during the main training phase
is not sufficient evidence that a second run will remain safe when either
process enters validation.

## 4. Controlled 600-step protocol

Every experiment will start from the same step-zero PhotoMaker/NN3a_new1
initialization, not from a later checkpoint of the long-running job.

### Training budget

- optimizer steps: `600`;
- checkpoint and validation cadence: steps `0, 200, 400, 600`;
- physical/effective batch: begin with `1/1` for the memory calibration, then
  use `2/2` only if safe;
- rank: 32 unless a later explicitly named rank ablation is needed;
- precision: BF16;
- seed: fixed across all experiments;
- base model: SDXL base for training, matching the current run;
- validation base: RealVisXL V4.0, matching the current run;
- Comet: one distinct experiment per training arm;
- scalar logging: every 25 steps; optimizer/layer norm summaries every 25
  steps.

Setting `trainer.epoch_len=200` and `trainer.n_epochs=3` gives a
checkpoint/validation boundary every 200 optimizer steps. The warmup will be
shortened proportionally; the initial control will test 50 warmup steps.

### One-ID data split

After the user selects an identity:

- copy its single JSON record into an experiment-local training JSON;
- keep eight of its ten generated face references for training;
- reserve two references as fixed holdouts;
- use holdout A for all recurring four-prompt validations;
- use holdout B only at step 0 and step 600 as a small reference-generalization
  check;
- retain the target image and bbox unchanged;
- retain the current reference crop-margin and sharpness jitter unless a
  later data ablation is explicitly named.

The split and exact source paths will be written into every experiment
manifest. This prevents accidentally training on the validation reference.

### Initialization parity gate

Before any 600-step run:

- render the four validation cases at step 0;
- compare them with exact `NN3a_new1`;
- for changes intended to be training-only, require identical architecture
  signatures and pixel hashes;
- for new residual adapters, require zero-initialized functional parity;
- reject the configuration before training if step-zero faces or routing
  differ unintentionally.

## 5. Four recurring validation prompts

The prompts are taken directly from the current 12-prompt validation file:

1. `Reading paper <class>, park bench, calm face, grey overcoat`
2. `Rushing <class> portrait, subway platform, anxious face, swinging briefcase`
3. `Kickboxing <class>, gym ring, fierce roar face, sweatband`
4. `Dancing <class>, neon club, euphoric face, silver jumpsuit`

`<class>` will resolve to `man img` or `woman img` from the selected identity.
This set includes a calm control, a motion/attachment case, a hard
occluder/expression case, and a difficult lighting/expression case.

The prompt text, reference, seed, and target bbox will be fixed across every
checkpoint and experiment.

## 6. Three validation configurations per training run

Each checkpoint will be evaluated with three separately named image streams:

### V1 — `canonical50`

- 50 inference steps;
- PhotoMaker starts at step 10;
- BA starts at step 15;
- guidance scale 5;
- exact trained architecture and `0.68/0.10` core controls.

This is the deployment-comparable view and the primary selection stream.

### V2 — `earlyBA50`

- 50 inference steps;
- PhotoMaker starts at step 10;
- BA starts at step 12;
- guidance scale 5;
- all architecture settings otherwise identical.

This exposes whether a checkpoint has learned useful branch authority that the
canonical schedule introduces too late. A gain only in this stream is useful
diagnostic evidence, not sufficient for promotion.

### V3 — `pmControl50`

- exact same reference, prompt, seed, base, and 50-step schedule;
- ordinary PhotoMaker output with branched self-attention disabled.

PhotoMaker weights are frozen, so this image can be generated once per
reference/prompt and reused in every checkpoint grid. It is still logged under
the checkpoint namespace for easy Comet comparison.

At steps 0 and 600 only, an additional `wrongRef50` causal audit may replace BA
memory with a different shortlisted identity while keeping the PhotoMaker
reference fixed. This is not part of the recurring 12-image validation budget.

### Collision-proof image names

Local files and Comet image names will include every relevant key, for example:

```text
canonical50__step0100__p00_reading__seed0.png
earlyBA50__step0100__p00_reading__seed0.png
pmControl50__step0100__p00_reading__seed0.png
```

Comet panel namespaces will be:

```text
val/canonical50
val/earlyBA50
val/pmControl50
val/wrongRef50_final
```

This prevents multiple validation modes from overwriting each other or
appearing under ambiguous image names.

## 7. First ten modifications to test

`E00` is the unmodified 600-step control. The following are the first ten
planned changes. They will not all be combined at once; early arms isolate one
mechanism, and only demonstrated winners enter a combined arm.

### E01 — active-up optimizer pruning

Keep all processors installed for exact forward compatibility, but freeze all
down/mid processor clones. Train only the 36 up-block processors.

Hypothesis: gradients and Adam capacity are currently spread over 70 sites
although only up blocks can select reference attention for the target face.
Pruning should accelerate useful learning and reduce target/background drift
without changing the step-zero forward.

Primary readout: faster `ΔIS` improvement per 100 steps with equal or lower
outside-face change.

### E02 — up1-only detail training

Within the exact forward, train only the six `up_blocks.1` processors; leave
up0 and all earlier processors at their step-zero values.

Hypothesis: the highest-resolution route may learn eyes, mouth, skin, and
identity detail without allowing coarse face geometry to move toward the
training target.

Risk: identity change may remain too weak or become texture-only.

### E03 — staged up1 → up0 unfreezing

Train up1 alone for steps 0–100. At step 100, unfreeze up0 with a reduced LR
(`0.25–0.35×` the up1 LR) while keeping up1 at full LR.

Hypothesis: learn detail first, then allow limited lower-resolution shape
adaptation. This directly transfers the useful staged-layer finding from the
recent step-zero experiments.

The unfreeze event and new optimizer groups must be checkpoint-resumable and
logged explicitly.

### E04 — projection-specific ref/noise learning rates

Keep both branches trainable, but separate:

- reference K/V: `1.0–2.0×` base LR;
- reference Q: `0.5×` base LR;
- noise Q/K/V: `0.10–0.20×` base LR;
- up0 groups: an additional `0.35×` multiplier when enabled.

Hypothesis: target-coordinate reference K/V are the direct identity route and
need faster adaptation; the noise route is needed for face/body integration
but should not become the identity owner.

This supersedes treating every one of the 840 tensors as two broad optimizer
groups.

### E05 — always-anchored blended loss

Replace `masked_alternating` with:

```text
L = 0.80 * full_image_MSE + 0.20 * face_MSE
```

on every step. Test `lambda_face=0.20` first.

Hypothesis: a stable full-image term on every update prevents the alternating
face-only objective from encouraging branch drift, while the face term avoids
losing the already small core gradient.

This is configuration-only and is a high-priority clean A/B against E00.

### E06 — core-normalized loss with ring preservation

Expose the legacy output core mask to an experiment-local loss and optimize:

```text
L = 0.70 * normalized_core_MSE
  + 0.20 * feathered_face_ring_MSE
  + 0.10 * full_image_MSE
```

The core term is normalized by core area rather than the full latent area.

Hypothesis: the present PhotoMaker output anchor leaves useful gradients in a
small core, so a full-image mean dilutes them. Core normalization should make
600 steps informative without increasing global LR.

The ring/full terms are preservation guards. A pure core-only loss will not be
promoted.

### E07 — inference-relevant timestep curriculum

Keep the same diffusion objective but change timestep sampling:

- steps 0–100: 70% from lower-noise/detail timesteps, 30% uniform;
- steps 100–600: 50% lower-noise/detail, 50% uniform.

An alternative implementation is SNR weighting rather than hard sampling; the
first version will use the simpler auditable sampler.

Hypothesis: uniformly spending updates on very noisy layout timesteps is
inefficient for an up-only face-detail mechanism and can reinforce PhotoMaker
structure without strengthening reference identity.

### E08 — low-timestep decoded identity loss

Add a small InsightFace identity loss on decoded predicted x0 only for
low-noise timesteps, with weight ramped from 0 to at most `0.05–0.10` during
the first 100 steps.

The identity target is the held-out/reference identity embedding, not ordinary
PhotoMaker output.

Hypothesis: denoising MSE alone does not distinguish a good BA identity from a
PhotoMaker-like face. A bounded decoded identity term supplies the missing
semantic direction.

Promotion requires no geometry or chroma regression. This arm is later than
the optimizer/loss screens because decoded identity losses are expensive and
historically easy to misuse.

### E09 — paired correct/wrong-reference directional loss

On a controlled fraction of updates, run a paired forward with identical
target noise, timestep, prompt, and PhotoMaker conditioning, changing only BA
reference memory. Require the correct BA reference to move decoded identity
toward the selected ID relative to the wrong-reference branch.

Include:

- correct-vs-wrong identity margin;
- ring consistency penalty;
- exact paired-noise invariants;
- no inherited epsilon-ranking loss unless explicitly enabled.

Hypothesis: this is the strongest test that BA, rather than PhotoMaker, owns
the learned identity direction.

This is a higher-cost/high-risk arm and will run only after a stable E01–E07
winner exists.

### E10 — zero-initialized reference-face residual adapter

Add a small bottleneck residual adapter only after the reference-attention
candidate in up blocks and before the unchanged core-ring merge:

```text
reference_face = reference_face + gate * Adapter(LN(reference_face))
```

- zero-initialize the adapter output projection or gate for exact step-zero
  parity;
- use one adapter per resolution block rather than per transformer layer at
  first;
- keep the original reference K/V route and core gate intact;
- cap/log the residual RMS.

Hypothesis: the current optimizer must reshape many cloned Q/K/V matrices to
improve identity. A small localized training path may learn the needed
reference correction faster while preserving the strong initial attention
geometry.

The adapter is rejected if it merely increases face MAE without improving
held-out identity similarity.

## 8. Experiment ordering

### Wave A — establish the trajectory

1. `E00_baseline`
2. `E01_active_up_only`
3. `E02_up1_only`
4. `E05_blended020`

These are the cleanest causal A/B tests and require the least novel code.

### Wave B — accelerate the best stable route

5. `E03_staged_up1_up0`
6. `E04_projection_lr`
7. `E06_core_ring_loss`
8. `E07_timestep_curriculum`

Each starts from step zero. One combined arm may then use the best layer scope,
optimizer split, and loss, but only after the individual effects are known.

### Wave C — make identity direction explicit

9. `E08_decoded_id`
10. `E09_correct_wrong_direction`
11. `E10_ref_face_adapter`

These are promoted only if Wave A/B still drift toward PhotoMaker or improve
too slowly.

## 9. Metrics and promotion criteria

Metrics will be recorded separately for every checkpoint and validation mode.

### Primary identity metrics

- `IS_BA`: generated-face InsightFace cosine to the held-out identity template;
- `IS_PM`: identical metric for ordinary PhotoMaker;
- `ΔIS = IS_BA - IS_PM`;
- identity consistency across the four prompts;
- correct-reference versus wrong-reference directional gain at the final
  causal audit.

Desired 600-step signal:

- median `ΔIS >= +0.03`, or a clearly improving trajectory that exceeds E00;
- no fall of more than `0.02` from the best prior checkpoint;
- correct-reference output closer to the selected identity than the
  wrong-reference output.

These thresholds are screen gates, not final production claims.

### Distinctness and geometry

- face MAE and LPIPS between BA and same-seed PhotoMaker;
- face detection: 4/4;
- landmark displacement versus PhotoMaker;
- bbox IoU versus PhotoMaker;
- outside-face MAE and boundary-ring MAE;
- face/body attachment and occluder preservation by visual review.

Desired safety region:

- visible nontrivial face difference from PhotoMaker;
- median landmark displacement `<= 0.04`;
- median bbox IoU `>= 0.90`;
- outside-face MAE `<= 0.015`;
- no duplicate face, detached head, severe color seam, or face melt.

### Training diagnostics

- loss components separately, never only total loss;
- per-layer and per-projection LoRA A/B norms;
- gradient RMS for up0 and up1;
- reference/noise optimizer-group LR and update norms;
- branch output face RMS versus PhotoMaker;
- selected core area and invalid-sample counters;
- adapter/gain values where applicable;
- GPU memory and seconds per optimizer step.

The best checkpoint may be step 200–400. Step 600 is not automatically the
winner.

## 10. GPU scheduling while the main run is active

The existing job must not be interrupted or exposed to validation OOM.

### Safety policy

1. Never launch while the main log is between `Validation start` and
   `Validation end`, or while used GPU memory is above 60 GB.
2. First run a one-step training plus one-prompt validation calibration with
   validation batch 1.
3. Allow co-resident execution only if observed aggregate peak memory remains
   below 78 GB, leaving at least approximately 3.5 GB emergency headroom.
4. Start only immediately after the main job finishes validation and returns
   to a new 2,000-step training epoch.
5. Require a predicted experiment wall time below the remaining main-training
   window with a 10-minute safety margin.
6. If a run cannot finish, stop only at a saved 100-step boundary, terminate
   it to release VRAM, and resume from that checkpoint after the next main
   validation. `SIGSTOP` is not useful because it retains VRAM.
7. If the calibration is marginal or training slows the main run enough to
   erase the window, wait for the long run to finish instead of gambling on an
   OOM.

A local watcher will parse the main log state and `nvidia-smi`; it will not
alter, signal, or checkpoint the main process.

## 11. Comet and artifact organization

Distinct Comet names will follow:

```text
23Jul_N3a1_<candidate_id>_E00_baseline_600_s0
23Jul_N3a1_<candidate_id>_E01_activeUp_600_s0
23Jul_N3a1_<candidate_id>_E05_blend020_600_s0
```

Every run will have an immutable local bundle:

```text
23Jul_debug/
  data/
    <candidate_id>/
      train_8refs.json
      holdout_manifest.json
      validation_prompts_4.txt
  configs/
  nn3a_training_lab/
  experiments/
    <timestamp>__<experiment_id>/
      command.txt
      resolved_config.yaml
      architecture_signature.json
      data_manifest.json
      comet.json
      stdout.log
      checkpoints/
      metrics/
        checkpoint_metrics.csv
        training_diagnostics.csv
      images/
        canonical50/
        earlyBA50/
        pmControl50/
        wrongRef50_final/
      contact_sheets/
      summary.md
  progress.md
  leaderboard.md
```

Production files remain read-only. If model or loss changes are required, an
experiment-local Python package under `nn3a_training_lab/` will subclass or
wrap the current implementation. Hydra will point to that local class through
an experiment-local launcher and `PYTHONPATH`; the main `src/` model, losses,
configs, and launchers will not be edited.

## 12. Top five identity candidates

All candidates have ten reference images, a target face larger than 400
pixels, good detector scores, and successful CPU InsightFace recognition for
the target and all ten references.

| rank | candidate | target↔ref cosine mean/min | ref↔ref cosine mean/min | target face | recommendation |
|---:|---|---:|---:|---:|---|
| 1 | `id_00081_1017318003459` | `0.8452 / 0.8233` | `0.8948 / 0.8405` | 432 px | Best identity consistency; clean near-frontal target |
| 2 | `id_00125_1150962006461` | `0.8436 / 0.8058` | `0.8901 / 0.8511` | 428 px | Very stable mature identity with distinctive facial hair |
| 3 | `id_00020_4037853000645` | `0.8037 / 0.7624` | `0.8684 / 0.8017` | 498 px | Sharpest target and highly distinctive facial structure |
| 4 | `id_00096_3540969004221` | `0.8076 / 0.7898` | `0.8569 / 0.8074` | 432 px | Strong worst-case target↔ref consistency and useful lighting variation |
| 5 | `id_00119_1313266018184` | `0.8080 / 0.7789` | `0.8705 / 0.8056` | 468 px | Distinctive mature female identity and stable reference set |

The accompanying PDF shows the full target, target face crop, and all ten
references for each candidate:

`one_id_candidate_reference_target_review.pdf`

The exact paths, bboxes, captions, and measurements are in:

`one_id_candidate_manifest.json`

### Recommended initial choice

`id_00081_1017318003459` is the metric-first recommendation because it has the
best target-to-reference and reference-to-reference consistency. If a less
recognizable or more structurally distinctive identity is preferred,
`id_00020_4037853000645` is the best alternative due to its sharp target and
strong brows, face shape, moustache, and beard.

## 13. Approval needed

Before execution, the user should:

1. choose one of the five identity candidates;
2. approve the Wave A ordering, or request a different first arm;
3. confirm that `canonical50`, `earlyBA50`, and `pmControl50` are the desired
   recurring validation streams.

No experimental training run will be launched until that approval.
