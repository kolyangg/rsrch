# Cosmic Large training analysis, adaptation plan, and prepared experiments

**Date:** 25 July 2026

**Branch:** `test`

**Status:** analysis and experiment code are prepared locally; no new run has
been launched, deployed, committed, or pushed by this work.

**Primary evidence:** [Cosmic Large one-ID Tasks A–D results
handoff](2026-07-25_cosmic_large_tasks_a_d_results_handoff.md)

**Operational companion:** [Cosmic Large experiment launch
plan](2026-07-25_cosmic_large_experiment_launch_plan.md)

## Executive decision

Cosmic Large should remain in the project, but it should not be fed through
the historical `cosmic` loader or treated as a conventional collection of
interchangeable full-scene target/reference pairs.

The best evidence-backed provisional policy is:

```text
training target:
    diverse 1024px full scenes with valid target-face boxes

training reference:
    a different same-identity image
    -> deterministic square face crop
    -> 20% margin per side
    -> bicubic resize to 256px

validation/inference reference:
    a full 1024px scene with its true face box

architecture:
    branched self-attention enabled
    branched cross-attention disabled
    masked loss every optimizer step
```

This is a research candidate, not a promoted recipe. Task D supports cropped
references during optimization, while Task B shows that the same tight
reference is unsafe when injected into the current spatial branch at
inference. The best Task D arm still failed Drumming and Jumping, and only one
identity was tested.

Before a long full-dataset run, the prepared 4k controls must answer three
questions:

1. Does slightly wider crop context repair the remaining small/dynamic-face
   failures without losing the identity gain?
2. Does explicit reference canvas occupancy help, or is crop content alone
   responsible for Task D's benefit?
3. Does the full Cosmic manifest preserve the Task D advantage once identity,
   pose, caption, and face-scale diversity increase?

## What Tasks A–D establish

### Observed evidence

| Task | Controlled change | Result | Implication |
|---|---|---|---|
| A | Disable branched CA | Text `23.7565 -> 24.7982`, ID `0.0351 -> 0.1418`; scenes/bodies improved, but about 9/12 faces still failed | CA contributes to global corruption but is not the main face-local failure |
| B | Fixed-checkpoint reference/CFG/processor interventions | Tight 256px Cosmic reference recreated malformed fragments on a healthy checkpoint; full-scene Larry King reference changed identity but produced mostly attached faces; CFG 1 collapsed; null removed the face | The reference path is causal, and inference reference geometry is the leading trigger |
| C | Train only the reference-side SA copies with CA off | Text `24.4779`, ID `0.1484`, about 9/12 failures | Target/noise projection drift is not the primary cause |
| D | Separate target diversity and training-reference format | `multi_cosref` was best: endpoint text `26.9297`, ID `0.3375`, roughly 10/12 plausible; `multi_full` and `single_full` were worse | Cropped references help during training; target-view diversity also helps |

Task D's strongest checkpoint was `multi_cosref` step 2,500, with text
`26.6471` and ID `0.3591`. It is a diagnostic checkpoint, not a deployment
candidate, because the hard anatomy gate still failed.

### Stage-dependent interpretation

Tasks B and D are not contradictory:

```text
crop during training
    reduces nuisance background/style and focuses the learned identity signal

crop during inference through the current spatial reference branch
    can be interpreted as literal spatial geometry and pasted into the target
```

That distinction is the central design constraint for Cosmic Large. A single
global “always crop” or “never crop” rule is not supported.

### Hypotheses, clearly separated from observations

- Cropped training references act as an identity-focused regularizer.
- The remaining failures are sensitive to target face scale, pose, and
  reference-to-target spatial correspondence, especially for Jumping and
  Drumming.
- Full-layer branched SA may overwrite target-native geometry too strongly
  when the target face is small or rotated.
- Long appearance-first captions may reduce pose/background conditioning
  because SDXL token capacity is finite.

The prepared data experiments test the first, second, and caption hypotheses.
They do not yet implement a new attention architecture.

## Recommended use of Cosmic Large

### 1. Preserve diverse target views

Do not repeat one target image as an identity-training shortcut. In Task D,
`single_full` lost `0.0504` ID similarity and `0.7266` text similarity versus
`multi_full`, and its anatomy was worse.

For the full dataset:

- keep the 1024px body/scene target;
- preserve its true target face box;
- stratify analysis by target face-area and pose, rather than reporting only a
  global mean;
- reject target faces whose shorter side is below 192px for the first
  controlled runs.

The 192px threshold is deliberately conservative. Lower thresholds can be
reintroduced later as a curriculum after the mechanism works reliably.

### 2. Use a different same-identity reference

Self-reference lets the model copy the target instead of learning identity
transfer. The new loader requires a distinct reference path and fails loudly
if target and reference are equal.

The current full manifest supplies candidate reference faces for each target.
After filtering, the real Neb package has:

- 59,143 input records;
- 22,140 targets passing the 192px target-face threshold;
- 2–10 valid reference candidates per accepted target, mean `8.158`;
- 137 invalid reference-bbox entries removed;
- zero accepted targets left without a reference.

There are 22,140 unique fallback reference-parent groups for 22,140 accepted
targets. In other words, this package does not demonstrate multiple target
views per explicit identity in the way Task D did. Its references may be
same-identity views, but the training targets are effectively one target per
pseudo-ID. This is the largest remaining dataset-structure risk.

Recommended follow-up data work:

- obtain or construct a stable identity ID for every target;
- group multiple full-scene targets per identity where possible;
- require at least two target views for a high-confidence training tier;
- retain single-target identities only in a lower-weight tier or mix them with
  a multi-view anchor dataset;
- visually audit random groups before trusting directory-parent identity
  inference.

### 3. Crop reference content, but do not cosmetically upscale it

For the first full run:

- square crop around the reference face;
- add 20% of the face side on every side;
- resize the resulting square once to 256×256 with bicubic interpolation;
- transform the face box by the exact same crop/resize operation;
- include the policy and flip state in the reference-conditioning cache key.

Upscaling the crop to 1024×1024 before feature extraction adds pixels, not
information. The prepared `canvas1024` intervention instead keeps the exact
256px content and centers it on a neutral 1024px canvas. This isolates spatial
occupancy from detail and should be treated as a diagnostic, not an assumed
quality enhancement.

The prepared `margin40` one-ID arm widens context at fixed 256px output. If it
repairs Jumping/Drumming while retaining Task D's metrics, use its crop margin
for the full-dataset candidate.

### 4. Keep full-scene references for validation and inference

Task B is direct causal evidence against tight inference references in the
current spatial branch. Every comparison must therefore keep:

- the same full-scene reference image;
- the same reference and generated face boxes;
- the same 12 prompts and seeds;
- the same scheduler, steps, validation model, and metrics.

Do not silently make training and inference reference formatting identical.
Their intentional difference must be recorded in each run manifest.

### 5. Make pose and scene tokens early enough to survive truncation

The current legacy caption concatenation is:

```text
facial appearance, pose, background
```

The prepared `pose_first` control uses:

```text
<class> img, pose, background, remaining appearance
```

and caps the prompt at 55 whitespace-delimited words. This is an isolated
caption-order experiment; it must be compared against the exact crop20 legacy
baseline before use in a long run.

### 6. Keep CA off and evaluate anatomy before scalar metrics

CA-off is the safer architecture based on Task A. Identity similarity can
reward a recognizable but malformed fragment, so promotion must be decided in
this order:

1. attached and complete face anatomy;
2. body/scene integrity outside the face;
3. pose and prompt adherence;
4. identity similarity;
5. text similarity.

## Code audit: does the pipeline work for this dataset?

### Historical full-Cosmic path: no

The existing `cosmic` dataset entry points to
`src.datasets.cosmic.CosmicDoubledTrain`. It:

- combines older Cosmic and Cosmic Large metadata rather than consuming
  `gathered_data_cosmic_large_filtered.json`;
- defaults to the target image itself as the reference unless a separate
  same-ID map is explicitly supplied;
- cannot read the manifest's `face_paths`, per-reference boxes, and scores;
- does not make crop/canvas policy part of the conditioning-cache key.

It remains untouched because it is historical replay evidence. It should not
be used for the new full-Cosmic experiments.

### New isolated full-Cosmic path: mechanically correct for the prepared runs

The new `CosmicLargeAdaptedTrain` path:

- reads the real `face_paths` manifest and configurable dataset root;
- validates target and reference boxes;
- filters target face size and optional reference score;
- samples a distinct reference and rejects target/reference leakage;
- preserves the 1024px target and its exact box;
- applies explicit crop, resize, and optional canvas transforms to references;
- transforms the reference box with the image;
- updates cache keys for policy and horizontal-flip state;
- exposes target/reference paths and identity IDs for audits;
- implements backward-compatible `legacy` and optional `pose_first` captions.

A deterministic preflight is mandatory in the launcher. It decodes samples
and verifies target/reference inequality, dimensions, bboxes, trigger word,
face-area fractions, prompts, and cache keys before training creates a run.

### Validation performed

Against the actual Neb package:

- the manifest scan accepted 22,140 records and filtered 137 invalid reference
  boxes;
- a deterministic 64-sample decode preflight passed 64/64;
- target face-area fraction ranged from `0.0441` to `0.6247` in that sample;
- cropped reference face-area fraction ranged from `0.3379` to `0.4967`;
- Hydra composition resolved the new loader, CA-off, masked-loss step 1, and
  the requested crop/canvas/caption policies;
- a real two-sample dataset instantiate and collate produced targets shaped
  `(2, 3, 1024, 1024)`, distinct references, valid boxes, pose-first prompts,
  and distinct cache keys;
- the canvas arm produced 1024px references while preserving 256px content.

These checks establish data and configuration correctness. They do not prove
that the attention architecture will generate valid anatomy.

### Remaining pipeline risks and safeguards

| Risk | Code evidence | Current safeguard | Required future fix |
|---|---|---|---|
| Heterogeneous batched spatial validation uses the first reference box/setup for the batch | `photomaker_branched_clean.py` reduces `face_bbox_ref` to entry 0 during branched setup | Prepared 12-image validation uses one shared identity/reference; use batch size 1 for heterogeneous identities | Make spatial reference latents/masks truly per-sample before multi-identity batched validation |
| `pose_adapt_ratio` is configured but the active attention processor hard-codes it to `0.0` | `attn_processor_cleanest.py` lines 292–331 | No prepared run claims to vary pose adaptation | Implement a backward-compatible runtime toggle, then test it separately |
| Branched SA is installed over a positional “first fraction” of processor names | `branched_runtime.py` returns `candidate_names[:keep_count]` | Prepared runs keep the historical `1.0` setting | Replace future architecture experiments with explicit semantic block allowlists |
| Face embedding selection uses the first detected face | `br_pipeline_helpers.py` uses `faces[0]` | Curated references and checked boxes contain a primary face | Select the detected face by overlap with the supplied reference bbox |
| A missing face can fall back to a zero identity embedding | current embedding helper behavior | Preflight checks boxes, but a box is not a detector-success guarantee | Fail or log/quarantine samples with no usable reference face |
| Alternate validation base previously installed the wrong processors | fixed in commit `5e55450b...` | Architecture flags are now set before processor installation and on the temporary pipeline | Keep the fix and audit processor counts in run logs |
| One-GPU alternate-base validation peaks near 79.3GB | observed on Neb | Never overlap GPU jobs; use one GPU/process for prepared runs | Redesign/offload validation before attempting two-GPU training |

### Pipeline conclusion

The newly prepared data path is suitable for controlled 4k experiments on
this package. The historical full-Cosmic path is not. The broader generation
pipeline still contains spatial-routing limitations, so a successful data
preflight must not be mistaken for a model-quality pass.

## Implemented backward-compatible experiment support

No historical dataset, launcher, or attention behavior was replaced.

| File | Purpose and compatibility |
|---|---|
| [`src/datasets/reference_policy.py`](../src/datasets/reference_policy.py) | Shared deterministic crop/resize/canvas transform with exact bbox propagation and cache descriptor |
| [`src/datasets/cosmic_large_adapted.py`](../src/datasets/cosmic_large_adapted.py) | Isolated loader for the real full-Cosmic manifest; legacy `cosmic.py` remains unchanged |
| [`src/datasets/controlled_identity_factorial.py`](../src/datasets/controlled_identity_factorial.py) | Adds optional reference transforms; all new options default to `null`, preserving Task D behavior |
| [`src/configs/cosmic_large_adapted_rhca.yaml`](../src/configs/cosmic_large_adapted_rhca.yaml) | CA-off full-Cosmic config using the new loader |
| [`src/configs/controlled_identity_reference_policy_rhca.yaml`](../src/configs/controlled_identity_reference_policy_rhca.yaml) | Post-Task-D one-ID reference-policy config |
| [`tools/datasets/preflight_cosmic_large_adapted.py`](../tools/datasets/preflight_cosmic_large_adapted.py) | Mandatory deterministic full-dataset preflight |
| [`launchers/active/run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh`](../launchers/active/run_rhca_cosmic_one_id_reference_policy_4k_1gpu.sh) | Exact `margin40` and `canvas1024` one-ID arms |
| [`launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh`](../launchers/active/run_rhca_cosmic_large_adapted_1gpu.sh) | Exact full-Cosmic arms and step budgets |
| [`launchers/neb/start_rhca_cosmic_experiment.sh`](../launchers/neb/start_rhca_cosmic_experiment.sh) | Neb environment/dataset wrapper |
| [`launchers/lib/prepare_comet_record.sh`](../launchers/lib/prepare_comet_record.sh) | Seeds and validates the canonical per-run JSON before Comet registration |
| [`src/logger/cometml.py`](../src/logger/cometml.py) | Atomically fills the immutable Comet key while preserving the experiment plan |
| [`tools/comet/comet_experiment.py`](../tools/comet/comet_experiment.py) | Retrieves metrics and images by immutable ID from local, Neb, or Serv runs |

The historical replay launcher continues to hash-lock architecture/runtime
files. New functionality is confined to new loaders/configs/launchers and
null-default options in the controlled loader.

## Prepared experiment matrix

### Code-ready training runs

| Global order | Run | Machine | Question | Direct control | Status |
|---:|---|---|---|---|---|
| 1 | `rhca_cosmic_oneid_margin40_4k` | Neb, 1 GPU | Does 40% crop context fix the remaining Task D small/dynamic-face failures at fixed 256px resolution? | Task D `multi_cosref` | Ready |
| 2 | `rhca_cosmic_full_crop20_legacy_4k` | Serv, 1 GPU | Does the clean full-manifest path reproduce a viable Task D-like crop benefit at dataset scale? | Task D bridge; control for all full arms | Ready |
| 3 | `rhca_cosmic_full_crop20_posefirst_4k` | Serv, 1 GPU | Does pose-first/capped captioning improve prompt and anatomy at the same reference policy? | Full crop20 legacy 4k | Ready after baseline starts cleanly |
| 4 | `rhca_cosmic_oneid_canvas1024_4k` | Neb, 1 GPU | Is spatial occupancy, rather than crop detail, driving the remaining failure? | Task D `multi_cosref` | Conditional |
| 5 | `rhca_cosmic_full_canvas1024_posefirst_4k` | Serv, 1 GPU | Does the one-ID canvas result transfer to full Cosmic? | Full crop20 pose-first 4k | Conditional |
| 6 | `rhca_cosmic_full_crop20_posefirst_20k` | Serv, 1 GPU | Does the winning 4k policy remain stable over a longer run? | Winning full 4k run | Gated; do not submit yet |

Every run has a source JSON under
[`experiments/cosmic_large_adaptation/`](../experiments/cosmic_large_adaptation/).
The launcher copies that JSON to:

```text
saved/<run_name>/comet_experiment.json
```

`CometMLWriter` then fills `comet.experiment_key`, workspace, and URL in the
same file. Run-name reuse, a non-empty saved directory, or an already
registered key is a hard failure.

### Recommended non-training diagnostics before architecture work

These do not require new optimization and should use the Task D
`multi_cosref` step-2,500 checkpoint:

1. Full-scene matched reference.
2. Original tight 256px reference.
3. The same 256px pixels centered on a 1024px neutral canvas.
4. Wider 40%-margin crop at 256px.
5. Wrong-identity full scene.
6. Null identity/reference.

This matrix separates training-reference benefit from inference-reference
occupancy on the best checkpoint. It should reuse the exact 12 prompts,
seeds, bboxes, validation base, CFG, and scheduler. The existing fixed
checkpoint evaluator supports matched/wrong/null conditions; crop/canvas
packages must be sealed with exact transformed bboxes before claiming a
deterministic comparison.

### Architecture experiments to prepare only after the data gates

If wider context/canvas does not reach the hard anatomy gate, add these behind
explicit defaults-off toggles:

1. **Target-native face fallback:** blend the branched face output with target
   SA output, beginning with branch weight `0.65`.
2. **Semantic site allowlist:** patch only named mid/up self-attention blocks
   instead of positional `top_k`.
3. **BBox-selected identity face:** choose the detected face that overlaps the
   supplied reference box rather than `faces[0]`.
4. **Per-sample spatial setup:** remove the first-reference assumption so
   multi-identity validation can run correctly in one batch.

Do not combine these in one first run; each needs a one-ID 4k control before a
full-dataset trial.

## Gates and decision rules

### One-ID 4k gate

Inspect matched step 500 and step 1,000 against Task D `multi_cosref` at the
same steps. Stop early only if both gates show at least 10/12 catastrophic
faces and no visible improvement.

Promotion requires:

- at least 11/12 anatomically coherent endpoint images;
- both Jumping and Drumming pass;
- no new body/scene corruption;
- text similarity at least `26.0`;
- ID similarity at least `0.32`;
- the result repeats on two additional identities before a full recommendation.

### Full-Cosmic 4k gate

Before training:

- preflight passes 64/64 deterministic samples;
- loader audit and policy are saved;
- the canonical Comet JSON exists and receives a live key;
- no target/reference path collision is observed.

During training:

- inspect step 500 and step 1,000;
- stop on OOM, corrupted masks/bboxes, systemic outside-face corruption, or
  the same catastrophic rule as above;
- compare legacy and pose-first arms at matched steps, not only their
  endpoints.

Endpoint promotion requires a fixed evaluation package covering at least
three held-out identities and face-scale/pose strata. Each identity must
reach at least 11/12 valid faces, with no outside-mask regression and stable
text/ID trends.

### Long-run gate

Do not submit the 20k job because it exists. Submit it only after:

- one full 4k policy wins the matched visual gate;
- at least three held-out identities pass;
- Comet metrics and images were retrieved by immutable experiment key;
- the winning policy JSON is sealed and its code commit is recorded.

## Machine allocation and safe parallelism

### Neb

Neb has one 80GB GPU. A 12-image validation pass reached about 79.3GB, so no
training or evaluation jobs may overlap, even when training itself appears to
use only 36–45GB.

Recommended Neb sequence:

1. `rhca_cosmic_oneid_margin40_4k`
2. `rhca_cosmic_oneid_canvas1024_4k` only if the margin/fixed-checkpoint
   evidence supports an occupancy test

The full baseline is assigned to Serv so it can start in parallel with the
one-ID margin control on Neb. Once its step-500 output and startup audit are
clean, the pose-first Serv arm may be submitted as a separate queued job.

### Serv

Prepared Serv packages use one `a100.1gpu.8C.243G` worker:

- [`serv_run_packages/rhca_cosmic_full_crop20_legacy_4k/`](../serv_run_packages/rhca_cosmic_full_crop20_legacy_4k/)
- [`serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k/`](../serv_run_packages/rhca_cosmic_full_crop20_posefirst_4k/)
- [`serv_run_packages/rhca_cosmic_full_canvas1024_posefirst_4k/`](../serv_run_packages/rhca_cosmic_full_canvas1024_posefirst_4k/)
- [`serv_run_packages/rhca_cosmic_full_crop20_posefirst_20k/`](../serv_run_packages/rhca_cosmic_full_crop20_posefirst_20k/)

They are built locally but not deployed or submitted. The source plan JSON is
sealed into each package.

Two-GPU packages were intentionally not created. The base launcher requests
one Accelerate process, changing process count changes global-batch semantics,
and rank 0 can retain the DDP training model while instantiating the alternate
validation model. That path has not passed an 80GB validation smoke test.
Prefer independent one-GPU experiments on Neb and Serv over one unverified
two-GPU run.

### Parallel schedule

Safe cross-machine parallelism:

```text
Neb:  one-ID margin40
      -> conditional one-ID canvas

Serv: full crop20 legacy
      -> full crop20 pose-first after the baseline step-500 gate
      -> conditional full canvas
      -> gated 20k winner
```

Neb and Serv jobs may overlap each other. Jobs on the same GPU may not.

## Runtime estimate

Measured Task D 4k runs took about 1h49m–1h54m on Neb. Allow approximately:

- one-ID 4k on Neb: 2 hours;
- full-Cosmic 4k on Neb: 2–3 hours, to be recalibrated after step 500;
- full-Cosmic 4k on a Serv A100: 3–4 hours plus queue;
- 20k on a Serv A100: roughly 15–20 hours plus queue.

These are planning estimates, not guarantees. Full-dataset decode and Serv
hardware/queue effects have not yet been measured by a completed run.

## Recommended execution decision

1. Review and commit the prepared code on `test`; record that commit in every
   JSON/run manifest.
2. Sync the exact commit to Neb and Serv.
3. Run one-ID `margin40` on Neb.
4. Submit full `crop20_legacy_4k` to Serv in parallel.
5. Once the full baseline passes its step-500 gate, submit
   `crop20_posefirst_4k` to Serv and let it run in parallel.
6. Run canvas arms only if the earlier evidence makes spatial occupancy a
   live hypothesis.
7. Evaluate every endpoint on sealed, multi-identity, full-scene validation
   references.
8. Submit a 20k Serv run only after a 4k candidate passes the documented
   visual and multi-identity gates.

The key near-term goal is not to prove that Cosmic Large “works” through one
average score. It is to identify a stable reference policy that retains the
dataset's identity coverage without converting reference face geometry into
malformed target anatomy.
