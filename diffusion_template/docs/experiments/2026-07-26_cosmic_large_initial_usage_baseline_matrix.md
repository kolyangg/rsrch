# Cosmic Large initial-usage baseline and dataset-policy matrix

**Date:** 26 July 2026

**Status:** active; all five trainers passed startup and monitoring stopped

## Purpose

This matrix establishes a fresh baseline for the way the initial `test` branch
used the Cosmic Large portion of `CosmicDoubledTrain`, then changes one
dataset-policy field per comparison while holding the current eligible
branched-attention model fixed.

The initial branch point is immutable commit
`6782e9d62345fe910633cc8ceec0e2fda6ec2fd1`. Its Cosmic Large behavior was:

- `cosmic_large_alldata.json` plus the matching generated-caption JSON;
- `LAION-5B` target paths remapped to `LAION-5B-Filtered-Large`;
- no minimum target-face size beyond the historical bbox bounds check;
- appearance-first `facial, pose, background` captions;
- horizontal target flipping with the bbox flipped identically; and
- the transformed target image itself used as the reference.

The baseline isolates that Cosmic Large portion. It intentionally does not mix
the separate 121k legacy Cosmic set that `CosmicDoubledTrain` also loaded.

## Fixed model and training contract

Every arm uses exactly the current eligible SA-only reference-conditioned BA
model:

```text
use_branched_attention = true
disable_branched_ca = true
branched_attn_weight_mode = noise_and_ref
pipeline.pose_adapt_ratio = 0.0
pipeline.ca_mixing_for_face = false
reference_face_kv_weight = 1.0
rank = 32
lr_for_lora = 1e-4
masked_loss_step = 1
```

The model, optimizer, loss, augmentations, training seed, 500-step epoch
length, 4,000 optimizer-step budget, checkpoint format, validation base,
scheduler, prompts, seeds, reference images, bboxes, inference steps, CFG, and
metrics are fixed across the matrix.

Each machine job is a train-then-evaluate chain. After training, the same job
opens a separate immutable Comet experiment and evaluates checkpoints at
steps `0`, `1,000`, `2,000`, `3,000`, and `4,000` on all 96 sealed
`cosmic_full96_auto_v1` samples. There are 96 images per step, batch size 12.

## Arms

| Role | Run | Machine | Only changed dataset field | Rationale |
|---|---|---|---|---|
| Baseline | `rhca_cosmic_initial_selfref_4k_baseline_r2` | Neb | None: `reference_mode=self`, `min_face_res=0` | Reproduce initial test-branch Cosmic Large usage |
| Improvement 1 | `rhca_cosmic_initial_distinct_uniform_4k` | Serv | `reference_mode: self -> uniform` | Remove target/reference leakage while retaining candidate diversity |
| Improvement 2 | `rhca_cosmic_initial_distinct_highest_4k` | Serv | `reference_mode: self -> highest_score` | Prefer maximum ArcFace match quality |
| Improvement 3 | `rhca_cosmic_initial_distinct_top3softmax_4k_r2` | Serv | `reference_mode: self -> top3_softmax` | Balance match quality and reference-view diversity; temperature `0.05` |
| Improvement 4 | `rhca_cosmic_initial_selfref_minface256_4k` | Serv | `min_face_res: 0 -> 256` | Test cleaner, larger target-face supervision without changing reference semantics |

The old metadata has 76,045 input rows. The newer reference-candidate package
covers 59,143 of those target paths. To keep the old target population fixed,
the three distinct-reference arms use the historical self-reference fallback
only for old rows absent from the candidate package. Startup preflight records
the exact accepted, candidate-covered, fallback, and filtered counts.

This fallback makes the reference interventions conservative: after the
historical caption and bbox gates, they change reference usage for 59,143
accepted targets and retain self-reference for the remaining 15,611, rather
than changing the target population. Results must be interpreted with that
measured intervention fraction.

## Why these four improvements

Reference choice is the highest-priority executable dataset question. The
current adapted loader samples uniformly despite having per-reference ArcFace
scores, while the initial branch leaked the target directly as its own
reference. Uniform, highest-score, and top-three score-weighted policies test
the principal quality/diversity choices.

Target scale is the next executable priority. Small/action faces remain a
repeatable failure mode, so the 256px threshold tests whether cleaner identity
supervision helps without changing reference selection.

Stable multi-target identity grouping and native full-scene reference context
remain higher-level priorities, but they are not honest one-variable jobs yet:
the package has no stable identity IDs joining targets, and its current
reference sources are already 256px face-focused assets. Those experiments
remain blocked on audited data artifacts rather than being approximated here.

## Runtime and provenance

Training and full-96 evaluation have separate local JSON specifications under
[`experiments/cosmic_large_dataset_usage`](../../experiments/cosmic_large_dataset_usage/).
At startup each Comet writer must create
`saved/<run_name>/comet_experiment.json`; the immutable key is the only
experiment identity used later.

| Training run | Scheduler/process ID | Training Comet key | Startup state |
|---|---|---|---|
| `rhca_cosmic_initial_selfref_4k_baseline_r2` | Neb PID `196928`, PGID `196733` | `aa982105aad148bf9b2a30d3fc2149f1` | Completed; all eight 500-step checkpoints present |
| `rhca_cosmic_initial_distinct_uniform_4k` | `lm-mpi-job-8f161a20-3303-40e2-8884-8c137348d9bb` | `288ebfe3ccf74d5ea328a55b3abe31cb` | Running; 840/840 processor tensors and optimizer loop verified |
| `rhca_cosmic_initial_distinct_highest_4k` | `lm-mpi-job-acd898ba-b09a-46e4-a8b5-4becae1b1280` | `fc3dec2223e84d49aa7c711fda968135` | Running; 840/840 processor tensors and optimizer loop verified |
| `rhca_cosmic_initial_distinct_top3softmax_4k_r2` | `lm-mpi-job-f2a4b83f-ab44-4717-82b8-cd085307db3f` | `b7821337e24e49f388450c103553a9da` | Running; 840/840 processor tensors and optimizer loop verified |
| `rhca_cosmic_initial_selfref_minface256_4k` | `lm-mpi-job-ca0acbd0-7433-42da-bcc1-39ab72a38272` | `c6979abd46754e4ca43fae87df77eeff` | Running; 840/840 processor tensors and optimizer loop verified |

The first Neb launch, `rhca_cosmic_initial_selfref_4k_baseline`
(`a42206ee6fd241a4914aabdb436eca7f`), was stopped before optimizer step 1
because ONNX Runtime could not load `libcudnn_adv.so.9`. Its record is retained
as `failed_start`. The launcher now loads the packaged CUDA libraries and
requires the CUDA provider explicitly; the clean run has a new output
directory and Comet identity.

The first top-three Serv process,
`rhca_cosmic_initial_distinct_top3softmax_4k`
(`lm-mpi-job-5295c0a9-49b9-43b0-8013-feabeeebe687`,
`ec43ee00375f4563b353bf701720c9eb`), stopped emitting output during model
initialization and was deleted before branched-processor installation or
optimizer step 1. Its files and immutable identities are retained. The clean
`_r2` replacement has the same experiment semantics and additionally disables
the optional C++ `addr2line` stack symbolizer; this does not alter the model or
training computation.

Startup dataset audits recorded 74,754 accepted examples for the baseline and
all three distinct-reference arms. The 256px target-face arm recorded 16,168.
All five deterministic decode preflights passed 64/64. A composed-config diff
against the current adapted Cosmic Large run found no model, pipeline,
optimizer, loss, or BA-flag differences.

Startup monitoring ended after every clean run completed initial validation
and entered its optimizer loop. No completion or validation-phase monitoring
is active.

The baseline's automatic validation chain stopped before Comet creation
because its historical bbox source did not match the sealed SHA-256. On
27 July it was restarted on idle Neb using the existing sealed protocol after
reproducing the original 12-image endpoint. The tracked full-96 run is active
at step 0 with eight batches of 12:
`rhca_cosmic_initial_selfref_4k_baseline_r2_full96_steps0_1k_2k_3k_4k`,
Comet `658d22341cf24accb5a3890869e76c28`.

Other full-96 Comet keys remain pending until their chained validation phases
start. Each job must fail closed if a required checkpoint, source Comet key,
CUDA ONNX Runtime provider, sealed bbox protocol, exact image count, or Comet
export is missing.

## Decision rule

Compare every checkpoint against the baseline at the same optimizer step.
Selection is visual-first and per identity:

1. reject malformed, detached, pasted, duplicated, or mask-like faces;
2. require improvement on Jisoo and no regression on small/action prompts;
3. use identity and text similarity only after anatomy passes;
4. reject recognizable target/reference fragment copying even if ID rises;
5. treat step 3k as the current candidate, but retain the full trajectory.
