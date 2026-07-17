# Fresh pre-N34 proposal for N39 and N40

Date: 17 July 2026

## Recommendation

Use commit `3c06eed7bb11744d87e2b816dc3a889808f051ba` as the behavioral base and run
one restoration anchor plus one isolated identity-memory improvement:

- **N39: clean N32 architecture anchor.** Re-establish the last visibly active,
  artifact-safe target-face residual path without any post-N33 architecture.
- **N40: N39 plus canonical-aligned input to the same eight-token face-patch
  resampler.** Change only the information presented to N32's memory module;
  preserve every downstream path, gate, merge, schedule, and loss.

Do not carry the N34-N38 layer allowlist, PhotoMaker context attenuation,
bounded gate, post-CFG composition, decoded-causal objective, FP32 master
conversion, or identity-owner naming into these runs. Some of those mechanisms
may eventually be useful, but their combined experiment is not interpretable
and produced a residual about half as visible as N32.

No training code or launch configuration is changed by this document.

## Why start at `3c06eed`

At this commit:

- N31, N32, and N33 configs/scripts already exist.
- N32's safe target-only residual implementation is present.
- the target residual is installed at all 70 SDXL cross-attention sites;
- `face_residual_gate` is the simple unit scalar;
- PM preservation is the legacy pre-CFG hard epsilon merge;
- there is no CA layer allowlist;
- there is no local PM identity-context attenuation;
- there is no post-CFG delta composition;
- there is no decoded-causal objective or canonical-memory code.

The current N31/N32/N33 YAML files are unchanged from this commit. The important
post-commit differences are in shared runtime/model code, not those leaf config
files.

The historical target residual is:

1. ordinary PhotoMaker attention output;
2. target hidden-state query to compact identity-token K/V;
3. zero-initialized low-rank identity residual;
4. unit residual gate;
5. hard target-face mask;
6. additive result at every selected cross-attention site;
7. hard PM epsilon restoration outside the face before CFG.

This path produced N32 face MAE versus PhotoMaker of:

- 2k: `0.06784`
- 6k: `0.07494`
- 10k: `0.07351`
- 16k: `0.07763`

N36/N38 remain around `0.038-0.039`, so restoring this path is a measurable
first requirement.

## Do not silently reuse all current shared code

Creating only a new YAML on current `HEAD` is not equivalent to a pre-N34 run.
Most new options default to legacy values, but current shared code also includes
mixed-precision conversions, validation/checkpoint fixes, new loss plumbing,
and other behavior.

The clean experimental choices are:

1. create a dedicated worktree at `3c06eed` and apply only explicitly selected
   correctness fixes; or
2. create an explicit compatibility mode on current code and prove with
   step-zero/one-batch tests that it reproduces the historical path.

For interpretability, prefer a worktree. Port only correctness fixes that are
needed to run on the current machines, and list each one. Do not port N34-N38
architecture features into N39.

## N39: N32 restoration anchor

### Question

Can the clean pre-N34 code reproduce N32-level branch activity under the current
dataset, validation set, and effective batch?

### Architecture

Use `one_id_ba_facepatch_resampler_N32` unchanged as the parent:

```yaml
defaults:
  - one_id_ba_facepatch_resampler_N32
  - _self_
```

Keep the resolved behavior:

| Component | N39 |
|---|---|
| identity memory | eight trainable InsightFace-conditioned face-patch tokens |
| reference preprocessing | N32 full reference + hard-bbox CLIP patch mask |
| spatial reference UNet | disabled |
| self-attention BA | disabled / standard |
| cross-attention BA | target-face residual |
| CA sites | all 70 |
| residual gate | legacy scalar, initialized `1.0` at every site |
| PM identity context | unchanged/full |
| PM preservation | hard epsilon merge |
| CFG composition | legacy pre-CFG |
| schedule | PhotoMaker at step 10; BOTH at step 15 |
| objective | exactly the inherited N32 objective |
| new architecture code | none |

### Training shape

Use the same effective batch for N39 and N40. Recommended:

- two DDP processes;
- local batch `2`;
- effective global batch `4`;
- no extra gradient accumulation;
- optimizer LR `1e-4`, warmup `200`, rank `32`;
- full 96-image validation at step 0, 2k, 4k, 6k, 8k, and 10k.

This is not a bit-for-bit replay of historical one-GPU N32 (global batch 2).
It is an architecture restoration under the intended current effective batch.
If exact historical reproduction is more important than machine utilization,
run an additional one-GPU/local-batch-2 canary to 2k.

### Expected signature

- step 0 matches ordinary PhotoMaker;
- by 2k, target-face MAE versus PM should be near N32's `0.0678`, not the
  N36/N38 range around `0.038`;
- changes remain localized and aligned;
- no N31-style systematic desaturation;
- identity score may remain below PM: N39 first tests branch restoration, not
  final identity superiority.

### Stop/fail criteria

Stop and debug before 4k if:

- selected target-face CA processor count is not 70;
- validation does not load all 70 trained processors;
- 2k face MAE versus PM is below `0.050`;
- BA-off, null-memory, and correct-memory outputs are indistinguishable;
- full-image/background differences become materially larger than N32.

N39 is the control. Do not alter its memory, objective, layer count, or merge to
make its metrics look better.

## N40: canonical-aligned N32 face-patch memory

### Question

Can the already-active N32 route become more identity-specific if its eight
tokens see a canonical face rather than an unaligned reference crop, while all
downstream behavior remains identical?

### Architecture

Start from N39 and change only reference preprocessing for the existing
`FacePatchIdentityResampler`:

1. align the detected reference face to the standard five-landmark canonical
   crop;
2. use bbox-centered crop fallback when landmarks are unavailable;
3. resize through the same CLIP image processor;
4. treat the canonical crop as the face region (all valid canonical CLIP patch
   tokens), or use a fixed inner-face mask defined in canonical coordinates;
5. feed those patches and the same 512-D identity embedding into the unchanged
   eight-query N32 resampler.

Everything after the resampler must remain N39:

| Component | N40 |
|---|---|
| token count/module | same eight-token `FacePatchIdentityResampler` |
| CA sites | all 70 |
| residual/gates | historical N32 unit-gate residual |
| PM context | full |
| PM merge | legacy pre-CFG hard merge |
| schedule | PM 10, BOTH 15 |
| loss | exactly N39/N32 |
| decoded causal loss | off |
| wrong-reference epsilon rank | off |

### Minimal future code surface

N40 should require only:

- one new `ba_identity_image_mode`, for example
  `canonical_face_patch_resampler`;
- a small canonical crop helper;
- the same preprocessing branch in training and inference;
- a strict token-shape/equality test.

Likely files:

- `src/model/photomaker_branched/identity_memory.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/pipelines/br_pipeline_helpers.py`
- one new N40 YAML and launch script
- focused unit tests for train/inference preprocessing parity

Do not import the complete post-N34 canonical/causal/identity-owner stack. If
using current code as reference, copy only the canonical crop operation needed
for this mode.

### Why this remains promising despite N37

N37 does not answer this question. It changed all of the following together:

- memory type and token count;
- 70 sites to 16;
- gates from 70×1 to 6×1 + 10×0.5;
- local PM identity attenuation;
- pre-CFG merge to post-CFG residual;
- training objective to decoded causal plus accidental epsilon ranking.

Its richer memory was evaluated through a route with only 11 unit-gate site
equivalents. N40 tests canonical memory through N32's known-active 70-site
route, making the result attributable.

### Expected signature

Relative to N39 at the same step:

- face MAE should remain in the active N32 range, not collapse toward N38;
- pose, expression compatibility, and background should remain as stable as
  N39;
- identity changes should be more consistent across reference pose/crop;
- gains should be strongest in eyes, nose, mouth, and contour rather than
  global color or contrast;
- wrong-reference inference should move identity in the swapped direction
  without moving the face box.

### Stop/fail criteria

Stop if:

- step-zero output differs from N39/PhotoMaker;
- face MAE is below `0.050` at 2k;
- canonical alignment causes repeated crop/landmark failures;
- changes correlate with expression/illumination rather than identity;
- artifacts, face displacement, or boundary seams exceed N39.

## Parallel allocation

For the cleanest comparison:

- run N39 and N40 concurrently with two GPUs each and effective batch 4;
- on the four-GPU machine, use GPUs `0,1` for one run and leave `2,3` for
  attribution probes, or run both experiments there if the second machine is
  occupied;
- keep seeds, data order policy, validation images, inference steps, CFG, and
  optimizer-step count identical.

The spare two GPUs are more valuable for short attribution jobs than a third
long architecture:

- correct reference;
- wrong/swapped reference;
- null identity memory;
- BA disabled;
- PhotoMaker baseline.

Run these at step 0 and 2k before committing to 10k.

## Mandatory diagnostics

At startup and every validation, log/assert:

- patched target-face CA processors: `70`;
- copied trained validation processors: `70/70`;
- identity memory shape: `[B, 8, 2048]`;
- residual gate min/mean/max;
- face-delta RMS relative to PM epsilon inside the bbox;
- outside-bbox epsilon max difference from PM;
- correct/null/wrong output difference for a fixed canary sample.

For each full validation, compute:

- fixed ArcFace/InsightFace mean ID score;
- same-seed target-face MAE versus PM;
- full-image MAE versus PM;
- landmark/expression displacement versus PM;
- chroma/saturation drift;
- blind enlarged face sheet.

## Decision rule after 2k and 6k

1. If N39 does not restore N32-level face movement, stop both runs and debug
   environment/behavioral parity; N40 cannot be interpreted.
2. If N39 restores activity but N40 does not, the canonical preprocessing patch
   is suppressing or corrupting memory.
3. If both are active and equally safe, prefer N40 only when reference-swap and
   fixed validation show more identity-specific movement.
4. If neither improves identity direction, retain N39 as the verified base and
   design the next experiment around a single, explicit decoded identity
   objective—without restricting layers or changing CFG composition at the
   same time.

## Explicit exclusions

Do not include in N39/N40:

- N34/N35 checkpoints;
- N36/N37/N38 checkpoints;
- `ba_ca_layer_allowlist`;
- `ba_pm_identity_context_scale < 1`;
- `ba_cfg_composition=post_cfg_delta`;
- bounded per-layer gates;
- decoded-causal loss;
- implicit simultaneous epsilon ranking;
- full spatial reference latents;
- N24-style interpolation of absolute target/reference outputs;
- residual-scale sweeps before the 70-site anchor is verified.

