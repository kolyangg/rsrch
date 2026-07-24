# NN3a 4,000-step experiment schedule

Date: 2026-07-23 UTC  
Status: approved by user; implementation and launch active

## Goal

Run a longer controlled screen to distinguish temporary 600-step behavior
from durable training dynamics. Every arm trains for 4,000 optimizer steps,
saves every 500 steps, and validates at:

`0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000`.

The recurring validation is four fixed prompts, one fixed held-out reference,
seed 0, canonical 50-step inference, plus a single PhotoMaker control set.

## Non-leakage requirement

- Native OneID arms use `train_on_separate_image=true`; each reference is
  sampled from the seven subset images other than the target.
- CosmicLarge arms use target `1017318003459.jpg` and references from its
  separate `face_paths` files.
- Every run must pass a preflight pairing audit before a GPU process starts.
- The validation reference is excluded from training for both profiles.

## Complete priority-ordered factorial

The first three pairs are already running through `schedule_4k_queue.sh`.
`schedule_4k_all_after_current.sh` is armed behind that live parent and then
runs every remaining valid 600-step recipe. The invalid E15/E16 same-image
leakage controls are deliberately replaced by distinct-reference OneID arms.

| priority | OneIDTrain / `nm0005092` | CosmicLarge single ID / `id_00081` | controlled question |
|---:|---|---|---|
| 1 | `L4_O1_oneid_projection_alt` | `L4_C1_large_projection_alt` | dataset/loader effect under current projection split |
| 2 | `L4_O2_oneid_projection_blend20` | `L4_C2_large_projection_blend20` | whether an 80/20 preservation anchor prevents late PM drift |
| 3 | `L4_O3_oneid_ref_value_blend20` | `L4_C3_large_ref_value_blend20` | whether freezing Q/K/noise protects coordinates over long training |
| 4 | `L4_O4_oneid_projection_schedule` | `L4_C4_large_projection_schedule` | projection split plus train/inference timestep matching |
| 5 | `L4_O5_oneid_active_up_blend20` | `L4_C5_large_active_up_blend20` | active-up pruning plus the preservation loss |
| 6 | `L4_O6_oneid_active_up_schedule` | `L4_C6_large_active_up_schedule` | active-up pruning plus timestep matching |
| 7 | `L4_O7_oneid_noise_damped` | `L4_C7_large_noise_damped` | all-scope control with stronger noise-clone damping |
| 8 | `L4_O8_oneid_all_blend20` | `L4_C8_large_all_blend20` | full trainable scope with the preservation loss |
| 9 | `L4_O9_oneid_projection_teacher20` | `L4_C9_large_projection_teacher20` | longer test of the matched PhotoMaker teacher |
| 10 | `L4_O10_oneid_control` | `L4_C10_large_control` | exact NN3a_new1 training control |
| 11 | `L4_O11_oneid_active_up` | `L4_C11_large_active_up` | active-up-only scope control |
| 12 | `L4_O12_oneid_up1_detail` | `L4_C12_large_up1_detail` | high-resolution up1-only scope control |
| 13 | `L4_O13_oneid_staged_up1_up0` | `L4_C13_large_staged_up1_up0` | staged up1 then reduced-rate up0 control |

The two arms in each row run concurrently. OneID uses GPU port slot 0 and
CosmicLarge uses slot 1. A run can train while its prior checkpoint is
validated; expected worst-case residency for two trainers plus two batch-1
validators remains below the 80 GB device limit.

This is 26 total 4k runs, including the six already in the live queue. The
ordering front-loads mechanisms that could plausibly protect coordinates or
correct the train/inference mismatch. Previously rejected scope-only and
teacher controls remain in the schedule at lower priority because 4,000 steps
can reveal recovery or late degradation that a 600-step screen cannot.

## Comet invariant

Each arm creates exactly one training experiment. Validation subprocesses use
the console writer and upload images with `ExistingExperiment` to the verified
training experiment key. Names include the stream, optimizer step, and prompt:

`canonical50__step1500__p02__...png`

No validation subprocess may use `writer=cometml`. After each arm, a local and
API audit requires:

1. exactly one Comet experiment with the run name;
2. one unique key across every validation manifest;
3. that key equals the training experiment key;
4. four canonical images at all nine stages and four PM controls.

## Artifact layout

All new run artifacts are isolated under:

```text
23Jul_debug/
  experiments_4k/
  architectures_4k/
  scheduler_4k/
  visual_reports_4k/
  EXPERIMENT_LOG_4K.md
```

Each run keeps the resolved command, source/config/data snapshots, pairing
audit, checkpoints, validation manifests and images, metrics, PDF, bbox debug
sheet, and Comet unity audit.

## Selection

Step zero is the identity-preserving reference point. At each 500-step stage
we track reference similarity, gain versus PhotoMaker, face similarity to
PhotoMaker, landmark displacement, bbox IoU, face/outside MAE, and visual
anatomy. A later checkpoint is useful only if it retains positive identity
gain without duplicated landmarks or face/body detachment.
