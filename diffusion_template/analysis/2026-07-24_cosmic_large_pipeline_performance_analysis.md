# Why the current pipeline performs worse on CosmicLarge

**Date:** 24 July 2026  
**Working branch:** `test` at `a913fc1`  
**Scope:** read-only analysis; no training, model, dataset, launcher, or
configuration code was changed  
**Primary handoff:** [2026-07-24_test_branch_one_id_overfit_handoff.md](../2026-07-24_test_branch_one_id_overfit_handoff.md)

## Executive conclusion

CosmicLarge is not failing because its diffusion targets are simply the wrong
size. The active full-dataset filter (`min_face_res=192`) leaves 22,140
1024×1024 targets whose median face size is very close to
`large_dataset_adj`. The dominant differences are on the **reference and
identity-grouping side**:

1. CosmicLarge has one 1024 target per inferred pseudo-identity and a set of
   tightly cropped 256×256 JPEG references. `one_id` and
   `large_dataset_adj` have multiple independent 1024×1024 scene images per
   identity, and another full scene image is sampled as the reference.
2. The pipeline enlarges each Cosmic reference about 4× with bilinear
   interpolation, VAE-encodes it as a full spatial grid, noises it, and gives
   its unregistered face K/V absolute authority over the target face at every
   patched U-Net self-attention layer. There is no landmark alignment, pose
   warp, correspondence model, or target-face fallback.
3. The current Cosmic one-ID diagnostic compounds this with one target repeated
   for every sample. Its training face occupies 25.13% of the image, while the
   fixed validation face boxes have a median area of only 4.58%. The successful
   `one_id` training faces have a median area of 8.80%, much closer to their
   5.96% validation median.
4. The current `one_id` comparison is not a clean holdout comparison:
   validation reference `ref/51.jpg` is byte-identical to
   `nm0005092_adj/51.jpg`, which is in the 19-image training JSON. Cosmic
   holdout A is correctly excluded from training. This gives `one_id` a real
   validation advantage, although prior leak-free 4k experiments show that
   leakage does not explain the whole Cosmic geometry failure.
5. Branched cross-attention and the trainable target/noise projections provide
   global drift paths. Exact current checkpoints show that by 2k steps the
   mean effective LoRA delta is 1.54× the `one_id` value in self-attention and
   1.36× in cross-attention for the alternating Cosmic run. The all-face-loss
   Cosmic run is still at 1.48× and 1.45× respectively. The masked-loss change
   therefore does not restrain the unstable branch dynamics.
6. Training never uses CFG, while validation uses guidance 5 and applies the
   ID-position mask to both conditional and unconditional prompt lanes. Prior
   controlled debugging on the closely related full-Cosmic pipeline showed
   that this amplifies trained branch errors, and that disabling branched CA
   removes most catastrophic corruption.

The latest `masked_loss_step=1` launcher is a good loss ablation: it removes
direct full-image reconstruction pressure from the single repeated target.
It does **not** alter reference resolution/alignment, pure-reference ownership,
all-layer routing, branched CA, CFG behavior, or the target/validation scale
gap. Manual inspection through step 3,000 shows that it has not repaired the
displaced/duplicated facial anatomy.

The highest-value next training experiment is therefore **Cosmic one-ID with
branched CA disabled**, keeping the current face-only loss and everything else
fixed. In parallel, a small same-identity data factorial should independently
test (a) multiple targets versus one repeated target and (b) full-resolution
scene references versus Cosmic-style tight 256 references.

## 1. Evidence and provenance

### 1.1 Sources inspected

The conclusions below combine:

- the `test` branch handoff, launchers, resolved configs, dataset classes,
  loss, trainer, model, attention processors, validation path, and current
  local datasets;
- read-only inspection over `ssh neb` of the full Cosmic metadata/images and
  current saved runs;
- the clean sibling `main_clean` worktree at `af20640` for the full
  `CosmicLargeTrain` and `LargeDatasetTrain` implementations, because the
  `test` branch contains only the new Cosmic one-ID loader;
- prior controlled reports under `main_clean`, especially:
  - `Jul_new_exp/23Jul_debug/EXPERIMENT_LOG_4K.md`;
  - `Jul_new_exp/23Jul_debug/results_4k_latest.md`;
  - `debug_planning_03Jul/ba_debug_results_v1.md`; and
  - `debug_04Jul/Codex13_Jul_N25_N26_architecture_analysis.md`.

Findings from the exact current RHCA replay are marked as current evidence.
Findings from later NN3a/N25/N26 variants are corroborating evidence, not
silently treated as results from the exact April architecture.

### 1.2 Terminology correction

`one_id` is **one identity with 19 different images**, not a multiple-identity
dataset. The additional large multi-identity dataset is
`large_dataset_adj`:

- metadata:
  `/home/kolyangg/rsrch/dataset_full/filtered_ids3_adj.json`;
- images:
  `/home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset`;
- loader in the sibling worktree:
  `/home/kolyangg/rsrch/diffusion_template/src/datasets/cosmic.py`,
  `LargeDatasetTrain`.

This distinction matters. The useful property shared by `one_id` and
`large_dataset_adj` is not “multiple IDs”; it is **multiple independent target
views per identity**.

## 2. Current observed behavior

### 2.1 Exact RHCA visual comparison

I inspected the fixed validation PNGs on Neb for:

- `rhca_apr2026_one_id_4k_exact`;
- `rhca_apr2026_cosmic_large_one_id_1gpu`; and
- `rhca_apr2026_cosmic_large_one_id_faceonly_8k`.

For the common Crying, Dancing, Kickboxing, and Reading prompts:

- `one_id` has coherent, recognizably human faces by step 4,000, although some
  images retain a locally composited/stylized face;
- alternating-loss Cosmic still has face-local smearing, displaced features,
  and missing or duplicated eyes/mouth at step 2,000;
- all-face-loss Cosmic still has severe deformation at steps 2,500 and 3,000.
  Crying, Kickboxing, and Reading are especially clear failures. Dancing is
  less severe but does not establish stable anatomy.

This is direct current-run evidence that `masked_loss_step=1` is not sufficient
to fix the main failure. It may still improve exterior preservation, which
should be measured at the planned 4k endpoint.

### 2.2 Exact checkpoint-delta comparison

I measured the effective rank-32 branch delta
`B @ A` in the current weight-only checkpoints. The reported value is mean
per-element RMS across Q/K/V, reference/noise lanes, and all corresponding
processor sites. It is a drift diagnostic, not a quality metric.

| Current run at 2k | self-attention delta RMS | versus `one_id` | cross-attention delta RMS | versus `one_id` |
|---|---:|---:|---:|---:|
| `one_id` exact | 0.0007307 | 1.00× | 0.0006184 | 1.00× |
| Cosmic alternating | 0.0011238 | **1.54×** | 0.0008408 | **1.36×** |
| Cosmic all-face loss | 0.0010811 | **1.48×** | 0.0008938 | **1.45×** |

The largest individual difference is in self-attention target/noise K:
0.0017134 mean for Cosmic alternating versus 0.0008439 for `one_id` at 2k,
about 2.03×.

This supports two conclusions:

- Cosmic produces substantially larger learned projection changes under the
  same optimizer schedule;
- making every step face-only does not prevent that growth, because the
  trainable Q/K/V and CA paths remain global operators even when the final MSE
  crop is local.

### 2.3 Corroborating leak-free 4k evidence from NN3a

The later controlled 4k matrix used distinct target/reference files and
excluded the validation reference from training. It is a different
architecture, but it reproduced the same dataset-dependent geometry split:

- at step 500, all four Cosmic prompts had horizontally displaced or
  duplicated eyes, nose, and mouth;
- the leak-free `one_id` arm was imperfect at 500 but recovered substantially
  by 1,000;
- Cosmic remained severely malformed at 1,000;
- at 4,000, Cosmic projection/alternating still had reference similarity
  0.3121 and gain versus PhotoMaker -0.2061, while the matched `one_id` arm had
  0.3373 and +0.0302;
- matching the train/inference schedule did not rescue Cosmic: its 4k gain
  versus PhotoMaker was -0.2787.

The Cosmic scalar identity score rose at early malformed checkpoints. This is
important: a face detector/encoder can lock onto a pasted or duplicated
reference fragment. Visual anatomy must remain a promotion gate.

## 3. Dataset comparison

### 3.1 Quantitative summary

| Property | Full CosmicLarge | Current Cosmic one-ID | `one_id` | `large_dataset_adj` |
|---|---|---|---|---|
| Identity structure | 59,143 inferred pseudo-identities; each reference-parent directory is unique to one target record | one identity | one identity | 2,561 explicit identities |
| Targets per identity | **1** | **1**, virtually repeated to 1,000 samples/epoch | 19 | 5–30, median 18 |
| Total target records | 59,143 raw; 22,140 pass current `min_face_res=192` | 1 | 19 | 47,500 |
| Target image | 1024×1024 training view; source mix of JPG/WebP/PNG | 1024×1024 JPEG | 1024×1024 JPEG | 1024×1024 JPEG |
| References per target/identity | 3–10, median 9 | 8 train refs + recurring holdout A | up to 18 other target images | 4–29 other target images |
| Reference image | **256×256 JPEG tight face crop** | **256×256 JPEG tight face crop** | another full 1024×1024 scene | another full 1024×1024 scene |
| Median target face area | raw 3.25%; **9.49% after current filter** | **25.13%** | 8.80% | 8.56% |
| Median reference face area | about **41.6%** of the 256 image | 41.0–43.9% | approximately the target distribution | approximately the target distribution |
| Median target face minimum side | raw 159 px; **272 px after current filter** | 432 px | 254 px | 255 px |
| Prompt length | median 64 whitespace words | 62 words / 83 CLIP tokens | median 54 words | median 51 words |
| Target/reference source distribution | different: scene target versus face-crop reference | different | matched | matched |
| Recurring validation ref seen in training? | depends on split | **no** | **yes: `51.jpg`** | should be explicitly split; not established by the loader alone |

Full Cosmic file types are:

- 50,998 JPG targets (86.2%);
- 7,123 WebP targets (12.0%);
- 1,022 PNG targets (1.7%);
- 484,409 references, all represented as 256×256 `.jpg` paths.

The active face-size filter retains 22,140/59,143 records (37.4%) and 180,760
reference paths. Its retained target median—272 px minimum side and 9.49% face
area—is slightly larger than `large_dataset_adj`, not smaller. A stale analysis
based on the raw 159 px Cosmic median therefore does not describe the current
filtered training set.

### 3.2 Image/content differences

Visual review of full Cosmic target/reference pairs showed:

- targets are varied web photographs, from portraits to upper-body/action
  scenes, with varied lighting, backgrounds, clothing, occlusion, and
  occasional other people;
- references are centered close-up faces or selfies, often with much less
  scene context;
- some sampled refs are blurred, tightly clipped, distorted, occluded, or
  include fragments of another person/background;
- reference pose, expression, hair, crop, and scale frequently differ from the
  target.

Visual review of `large_dataset_adj` and `one_id` showed that both the target
and sampled reference are full 1024 scene images. They still contain pose,
lighting, occlusion, and multi-person noise, but the two lanes have much closer
resolution, sharpness, face occupancy, and image-domain statistics.

The selected Cosmic one-ID itself is not a poor identity cluster. Prior
InsightFace screening ranked it first:

- target↔reference cosine mean/min: 0.8452/0.8233;
- reference↔reference cosine mean/min: 0.8948/0.8405;
- every selected reference had a successful face detection.

The current failure on this identity therefore cannot be dismissed as a
wrong-person or grossly inconsistent-reference problem.

### 3.3 Prompt truncation is much more common in Cosmic

[`encode_prompt_with_trigger_word`](../src/model/photomaker_branched/lora2.py)
tokenizes with `max_length=77` and `truncation=True` (lines 518–523).

Using the actual SDXL CLIP tokenizer on seeded 5,000-record samples:

| Dataset | median tokens before truncation | p75 | records over 77 tokens |
|---|---:|---:|---:|
| filtered CosmicLarge | 86 | 93 | **79.28%** |
| `large_dataset_adj` | 64 | 68 | **1.52%** |

The current Cosmic one-ID prompt is 83 tokens, while most `one_id` prompts fit
or are only slightly over. The trigger occurs early, so identity conditioning
is retained; later pose/background information is what is preferentially
lost. This is especially harmful on full-image-loss steps: the optimizer is
asked to explain target scene details that are absent from the truncated text,
making target/noise and CA adapters a tempting memorization path.

### 3.4 The current `one_id` validation comparison leaks

The `one_id` training JSON contains `51.jpg`. The recurring validation
directory also contains `51.jpg`, and the files have the identical SHA-256:

```text
5fdc48f461b0629b75113cfb2a7e078cc94d6a4f1b61eb1121452d87b4d77ed9
```

With uniform 19-image sampling, that image is expected to appear as the
diffusion target on about 1/19 of samples and as the distinct reference on
about 1/19. Across 4,000 steps at batch size 2, that is roughly 421 exposures
in each role. Cosmic validation holdout A is never sampled in training.

This does not invalidate the April historical replay, whose purpose is exact
reproduction. It does mean its validation quality must not be used as a fair
dataset comparison without an 18-image, leak-free `one_id` control.

The later leak-free NN3a result still favored `one_id`, so leakage is an
inflating factor rather than a complete explanation.

### 3.5 Current Cosmic one-ID is a stress test, not a single-factor A/B

Relative to `one_id`, it simultaneously changes:

- identity and class (`woman` versus `man`);
- one target versus 19 targets;
- eight training references versus an 18-image candidate pool;
- 256 face crops versus 1024 scenes;
- target face area (25.13% versus 8.80% median);
- train/validation target-mask scale;
- prompt/token length;
- validation holdout policy.

It is useful for reproducing a Cosmic-style failure, but it cannot by itself
identify which data property caused the difference.

One additional integrity issue: `split_manifest.json` and the README declare a
final-only `holdouts/holdout_B.jpg`, but that file is absent in the checked
`test` worktree. Recurring holdout A is present and correctly separated.

## 4. Pipeline issues that interact badly with Cosmic

### 4.1 Critical: 256 reference is enlarged and treated as a full spatial memory

[`CosmicLargeOneID.__getitem__`](../src/datasets/cosmic_large_one_id.py)
returns the raw 256 reference and its raw bbox (lines 105–137).
[`_encode_reference_latent`](../src/model/photomaker_branched/lora2.py)
then:

1. resizes the reference to fill/fit 1024 using `Image.BILINEAR`
   (lines 676–688);
2. VAE-encodes the enlarged image (lines 691–698);
3. noises it to the target timestep in
   [`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py)
   (lines 367–386).

A 256 source has only about a 32×32 native VAE-resolution information grid,
even though it is expanded to a 128×128 latent. Interpolation cannot recreate
skin/eye/mouth detail. The high-resolution U-Net sites receive oversampled,
correlated, JPEG-affected reference features.

More importantly, Cosmic's reference face occupies about 42% of that grid,
while a typical filtered target face occupies about 9.5% and current
validation faces about 4.6%. `large_dataset_adj` supplies target and reference
faces at approximately the same scale.

**Improvement:** use original full-resolution references where available. If
only 256 crops exist, encode a clean, landmark-aligned face ROI with an
explicit native-resolution reference encoder or compact tokens. Do not pretend
that a bilinear 4× enlargement is a coordinate-compatible 1024 scene.

### 4.2 Critical: raw reference geometry gets pure authority

In
[`BranchedAttnProcessor`](../src/model/photomaker_branched/attn_processor_cleanest.py):

- the target face supplies Q;
- the masked reference grid supplies K/V;
- `POSE_ADAPT_RATIO` is hard-coded to `0.0` (lines 296–331);
- the face and background absolute outputs are selected by the target mask
  (lines 374–384).

The reference and target latent cells do not represent corresponding anatomy.
There are no landmarks, canonical coordinates, pose warp, or local
correspondence constraints. Hair, expression, occlusion, and face orientation
can therefore be transferred together with identity. The observed displaced
eyes/mouth and pasted face fragments are the expected failure mode.

**Improvement:** preserve target pose/geometry and use reference information as
a bounded correction. A conservative short-term version is a 20–35% target
face K/V fallback. The better long-term interface is:

```text
h_out = h_photomaker
      + soft_face_mask * bounded_zero_init_gate * delta_from_aligned_reference
```

### 4.3 Critical/high: all layers are patched

The replay sets `ba_patch_top_k=1.0` and `ba_train_top_k=1.0`, so all selected
self- and cross-attention sites participate. Down/mid blocks can change coarse
face/head geometry; repeated up-block replacement can compound seams.

Using a fractional `top_k` is not a reliable “up-block only” control:
[`select_branched_processor_names`](../src/model/photomaker_branched/branched_runtime.py)
keeps the first `candidate_names[:keep_count]` (lines 15–40), which follows
module iteration order rather than an explicit semantic layer policy.

**Improvement:** use an explicit allowlist for late/up-block face sites,
starting with the high-resolution/detail block. Keep ordinary PhotoMaker
attention elsewhere.

### 4.4 High: branched CA is global and not target-face CA

The active
[`BranchedCrossAttnProcessor`](../src/model/photomaker_branched/attn_processor_cleanest.py)
does:

| Lane | Query | K/V |
|---|---|---|
| target/noise | every target token | generation prompt |
| reference | every reference token | sparse face/ID prompt |

It does not use the target/reference spatial masks in `__call__`, and there is
no target-face query attending the face prompt. Its influence on the generated
face is indirect through the continuing reference stream and later
self-attention.

With `noise_and_ref`, both target/noise and reference CA Q/K/V LoRAs are
trainable. A face-cropped final loss can still update a global target CA
operator whose inference effect is not spatially restricted.

Prior full-Cosmic debugging on the related 4k checkpoint found:

- trained guidance-5 output: catastrophic collage;
- same checkpoint with branched CA disabled: best clean trained result,
  identity score 0.414 versus 0.446 for untrained clones;
- guidance 1 removed the paint explosion but retained reference-fragment
  collage.

**Improvement:** first set `disable_branched_ca=true` and
`train_branched_ca_lora=false` as a clean A/B. If CA is later restored, make it
actual target-face CA with masked target queries and compact identity tokens.

### 4.5 High: target/noise adapters can memorize a one-target dataset

`noise_and_ref` trains independent target/noise Q/K/V as well as reference
Q/K/V at every site. On Cosmic one-ID:

- every sample has the same target scene and face;
- a batch of two contains two noisy versions/flips of the same content;
- one scalar timestep is sampled and repeated across the batch
  ([`lora2.py`](../src/model/photomaker_branched/lora2.py), lines 336–344);
- alternating loss directly supervises that same full scene on half the
  steps.

This is a much more correlated optimization problem than 19 independent
`one_id` targets. The measured 1.5× branch deltas are consistent with this
pressure.

All-face loss removes direct exterior reconstruction, but it still repeatedly
supervises one target face pose/expression and updates global Q and CA paths.
It also supplies no positive “preserve PhotoMaker outside the face” objective.

**Improvement:** after the CA-off test, use `ref_only` so target/noise
projections stay at their inherited values. For a production loss, combine
face learning with an explicit outside-mask PhotoMaker teacher/preservation
term and a narrow boundary-ring term.

### 4.6 High: CFG behavior is untrained and the unconditional face lane is unsafe

Training explicitly discards `do_cfg`
([`lora2.py`](../src/model/photomaker_branched/lora2.py), line 325).
Validation uses guidance scale 5.

For `face_embed_strategy=id`,
[`two_branch_predict`](../src/model/photomaker_branched/branched_runtime.py)
clones the prompt embeddings, repeats the class-token mask to match the CFG
batch, zeros non-ID positions, and scales ID positions by 2.5 (lines 426–468).
This also applies the ID-position mask to the unconditional/negative lane. CFG
then extrapolates the conditional-minus-unconditional branch difference by
5×.

Prior controlled evidence identifies this as the main amplifier, though not
the source of the corruption.

**Improvement:** make the unconditional face lane zero or correctly
unconditional, and compose a bounded BA correction after CFG rather than
allowing guidance to multiply an untrained branch delta. Add conditioning
dropout or CFG-consistent training if the branched path must exist inside CFG.

### 4.7 High: hard, fixed rectangular masks amplify drift

`mask_softness=0` makes masks binary. The processor resizes and thresholds
them at each attention resolution. Multiplying reference hidden states by a
mask does not remove non-face tokens from softmax; zero K/V tokens remain
attention sinks.

The target bbox is generated by a preliminary PhotoMaker pass and then held
fixed. If BA moves or enlarges the face, the write region becomes stale. Prior
debugging observed a ghost face painted into the old bbox while the real face
moved outside.

The Gaussian blur in `two_branch_predict` lines 581–598 only decomposes the
already merged prediction for debug outputs; it does not undo hard
feature-space replacement.

**Improvement:** use an additive attention mask or compact face-token sequence,
soft semantic masks, and fractional coverage at coarse scales. Separately test
re-detecting/tracking the target mask at BA start. This changes validation
semantics and must be a named experiment, not a silent default change.

### 4.8 High evaluation issue: SDXL branch bases are copied into RealVis

Training initializes on SDXL base; validation instantiates RealVisXL V4.
`BranchLoRALinear.base_weight` and `base_bias` are registered buffers
([`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py),
lines 12–36).

The validation model is correctly initialized from RealVis, but
[`base_trainer.py`](../src/trainer/base_trainer.py) then calls
`v_proc.load_state_dict(t_proc.state_dict(), strict=False)` for every
processor (lines 463–480). A full module `state_dict()` includes the SDXL
`base_weight` buffers, not only trainable LoRA A/B. The result is a hybrid:
RealVis U-Net plus SDXL-derived branch Q/K/V bases.

This affects both datasets and is not the sole Cosmic cause. It can magnify
Cosmic's larger reference-driven correction and means validation is not a
clean measurement of either the training base or native RealVis branch.

**Improvement:** either train and validate on the same base, or copy only the
trainable LoRA A/B keys and leave the validation processor's base buffers
initialized from RealVis. Test this as its own validation A/B.

### 4.9 Medium: training/inference timestep mismatch

`train_ba_all_steps=true` routes every uniformly sampled training timestep
through BA. Inference uses:

```text
steps 0–9:   base/text
steps 10–14: PhotoMaker
steps 15–49: BA + PhotoMaker
```

At high training noise, the bilinear-upscaled 256 reference is itself heavily
noised and carries little reliable local identity, yet the face branch still
has pure-reference authority. Matching the inference window is sensible, but
the prior NN3a 4k schedule arm did not rescue Cosmic. Treat this as an
amplifier, not the leading root cause.

### 4.10 Medium/robustness issues

- Face detection failure silently substitutes a zero 512-D identity embedding
  (`lora2_helpers.py`, lines 227–245). It should reject/log the sample.
- When multiple faces are detected, `faces[0]` is used without matching it to
  `face_bbox_ref`. Use bbox IoU/center and confidence to select the intended
  face.
- Invalid target/reference bboxes fail open to a full mask in
  `_bbox_to_mask`/`_bbox_to_ref_mask`. Invalid masked-loss boxes also fall back
  to full-image MSE. Fail closed and report counts.
- Loss boxes use floor division through `astype(int32)`, while routing boxes
  use rounding. The supervised crop and routed region can differ by a latent
  cell. Use one shared mask builder.
- The full Cosmic loader uniformly samples refs and does not use `face_scores`
  to reject or down-weight weak refs.
- The model uses only `refs[0]` for the spatial reference latent. Increasing
  `num_refs` would not create spatial multi-reference aggregation without
  further implementation.

## 5. Full-Cosmic loader/config observations

These points apply to the clean `main_clean` worktree at `af20640`, not to the
new `test`-branch one-ID loader, and should be checked against the exact commit
of any full-Cosmic run.

The current `CosmicLargeTrain.__init__` explicitly deletes these incoming
arguments:

```text
require_nested_identity_subdir
upscale_to_1024
const_ref
crop_ref
ref_similar
origtarget_genref
train_on_separate_image
same_id_ref_map_json_pth
```

The config still exposes several of them. Experiments that believe those knobs
changed the active full loader may therefore be no-ops.

The active loader always:

- filters targets by `min_face_res`;
- samples one path from `face_paths`;
- applies a square bbox crop with a default fixed +20% margin;
- optionally mirrors the cropped ref;
- returns that cropped reference and bbox.

Because the raw 256 reference already has a median 41.6% face area, the +20%
crop is usually close to the whole 256 image and does not make it
scene-distribution-compatible with a 1024 target.

## 6. Recommended Cosmic preprocessing and use

### 6.1 Rebuild identities around multiple independent targets

Highest priority:

1. assign a stable explicit `identity_id`;
2. cluster/deduplicate target records belonging to the same person;
3. require at least 4–5 independent full-resolution target images per
   identity;
4. reserve image-disjoint recurring and final holdouts;
5. rotate eligible images through target and reference roles, always excluding
   the current target;
6. log target/reference paths, identity, bboxes, crop/flip, and mask coverage
   for initial batches.

If Cosmic cannot supply multiple full-resolution targets per person, mix it
with `large_dataset_adj` identities or use Cosmic at a lower sampling weight
after a clean multi-view curriculum. Repeating a one-target pseudo-identity
thousands of times should remain a stress test, not the production sampling
unit.

### 6.2 Normalize reference representation, not merely file dimensions

Preferred order:

1. recover the original full-resolution reference images, if available;
2. detect the intended face by matching the supplied bbox;
3. landmark-align an inner-face ROI to canonical coordinates;
4. preserve a separate hair/context representation if desired;
5. encode the canonical ROI at its native useful resolution;
6. match train and validation preprocessing exactly.

If only 256 JPEG crops are available, do not improve them by a cosmetic
Lanczos/bicubic upscale alone. The architectural consumer must know it is
receiving a low-resolution face ROI rather than a full 1024 spatial scene.

### 6.3 Make target face scale representative

The chosen one-ID target was selected for a face larger than 400 px, but this
creates a 25.13% train-mask area versus 4.58% median validation area.

For the diagnostic:

- select a high-consistency identity with a 220–320 px target face; or
- add deterministic/random scale-and-pad augmentation that covers the
  validation face-area distribution while updating the bbox exactly.

For full Cosmic, keep the current `min_face_res=192` baseline, but stratify
sampling by face area so large close-ups do not dominate face-token counts.

### 6.4 Shorten captions with token-aware ordering

Build prompts to fit before the 77-token limit, preferably with margin for
PhotoMaker token expansion:

```text
class + img, action/pose, essential background, short appearance summary
```

The current order spends many early tokens on facial appearance that the
reference/PhotoMaker path already supplies and truncates later scene
information. Record pre/post token counts and require a near-zero truncation
rate.

### 6.5 Offline quality filters

For every retained identity:

- validate paths, RGB decode, dimensions, bbox bounds, and EXIF orientation;
- require successful intended-face detection;
- use target↔ref and ref↔ref identity consistency thresholds;
- filter or weight blur, JPEG damage, severe crop, occlusion, watermark, and
  multi-face ambiguity;
- deduplicate near-identical refs;
- maintain pose/expression diversity after identity/quality filtering;
- record all reject counts and thresholds.

The selected one-ID already passes identity consistency, so these filters will
improve the full dataset but will not by themselves solve the present
architecture failure.

## 7. Quick experiments that are decisive by 4k

All training arms should use the same optimizer-step count, effective batch,
seed, validation model, prompts, reference, fixed bboxes, scheduler, inference
steps, and checkpoint cadence. Validate at:

```text
0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000
```

### Priority 0: inference-only diagnosis on existing checkpoints

No new training is needed. On the current `one_id`, Cosmic alternating, and
Cosmic face-only checkpoints, run the same four or twelve prompts with:

| Toggle | Values | Question |
|---|---|---|
| CFG | 1 and 5 | How much is trained branch corruption amplified by CFG? |
| Branched CA | on and off | Is CA the dominant global destroyer in the exact RHCA replay? |
| Reference | matched, wrong identity, null | Is the output causally using the ref, and is the effect face-local? |
| Processor base | current copied SDXL buffers versus validation-native RealVis buffers | How much does the hybrid-base validation copy hurt? |

This matrix can eliminate bad 4k directions within hours.

### Priority 1: finish the current all-face-loss run at 4k

This experiment is already running:

```text
rhca_apr2026_cosmic_large_one_id_faceonly_8k
trainer.masked_loss_step=1
```

Use its 4k checkpoint as the decision point even though the launcher permits
8k. Compare it with the alternating run at matched steps.

**Conclusive result:** if anatomy remains malformed while outside drift is
reduced, full-image loss was an exterior-drift contributor but not the root
cause. Step 3k already strongly points in this direction.

### Priority 2: one new 4k CA-off Cosmic arm

Keep the Cosmic one-ID data and all-face loss fixed:

```text
disable_branched_ca=true
train_branched_ca_lora=false
branched_attn_weight_mode=noise_and_ref
```

Do not also change LR, layer scope, masks, or reference preprocessing.

**Conclusive result:** clean geometry with much lower exterior drift would
identify branched CA as the main unstable learned path. Persistent
reference-fragment anatomy would isolate self-attention/raw-grid transfer as
the remaining blocker.

### Priority 3: same-identity data factorial using `large_dataset_adj`

Choose one clean identity with at least ten 1024 images. Reserve one recurring
holdout and one final holdout. Use the same identity, target prompts, and
validation in three 4k arms:

| Arm | Targets | References | Isolated question |
|---|---|---|---|
| `L-multi-full` | 8 independent 1024 images | another one of the 8 full images | clean baseline |
| `L-single-full` | one repeated target | 7 distinct full 1024 refs | cost of losing target diversity |
| `L-multi-cosref` | same 8-target sampling as baseline | deterministic tight face crop downscaled to 256 | cost of Cosmic reference domain/resolution |

No training or validation image may overlap, and target must never equal ref.

These three arms are more informative than comparing different people in
`one_id` and Cosmic. They hold identity/content fixed and change one data
property at a time. If resources permit only two arms initially, run
`L-multi-full` and `L-multi-cosref`; the reference-format hypothesis is the
most directly tied to the spatial BA failure.

### Priority 4: ref-only CA-off arm if Priority 2 is only partially clean

Starting from the CA-off recipe, change only:

```text
branched_attn_weight_mode=ref_only
```

This freezes target/noise branch projections and asks whether reference-side
specialization alone can improve identity without rewriting the generated
scene.

If this is still malformed, do not spend the next runs sweeping LR or loss
weights. Move to an explicit up-block allowlist plus target-face fallback or a
canonical reference-token interface.

### Optional targeted scale test

On the same current Cosmic identity, deterministically scale/pad the training
target so its face-area distribution matches the 4–10% validation range,
updating its bbox. Keep refs and architecture unchanged.

This is a useful one-factor test of the 25.13%→4.58% mask-scale gap, but it
should follow CA-off because it is less likely to fix pure-reference geometry
transfer on its own.

## 8. Evaluation and stopping rules

For every 4k arm, report:

- visual face anatomy and face/body attachment;
- detector success and duplicate-face count;
- identity similarity to the held-out reference;
- gain versus the exact PhotoMaker control;
- landmark displacement and bbox IoU;
- prompt CLIP score;
- outside-mask LPIPS/MAE to PhotoMaker;
- narrow boundary-ring error;
- matched/wrong/null reference sensitivity inside and outside the face.

Promote only if:

1. no repeated/displaced eyes, nose, or mouth occur in the fixed prompt set;
2. identity gain versus PhotoMaker is positive on the held-out ref;
3. wrong/null reference tests prove causal reference use;
4. the reference effect is materially larger inside the face than outside;
5. prompt/scene quality and face/body alignment do not regress.

The existing evidence shows that the Cosmic pathology is obvious by
500–1,000 steps and can persist through 4,000. Four thousand steps are enough
for these rejection decisions. A single-identity success is mechanistically
useful but should be replicated on at least two additional identities before a
full-dataset claim.

## 9. Ranked diagnosis

| Rank | Diagnosis | Confidence |
|---:|---|---|
| 1 | Unaligned, bilinear-upscaled 256 face crops are used as pure full-grid spatial K/V | **High; code + visuals + prior controlled results** |
| 2 | One target per identity/repeated one-target training removes view diversity and encourages target/scene memorization | **High as a dataset difference; causal size needs factorial** |
| 3 | All-layer pure-reference ownership with no target fallback destroys target geometry | **High; code + characteristic artifacts** |
| 4 | Global branched CA and target/noise adapters are major drift channels | **High; current deltas + prior CA-off A/B** |
| 5 | CFG/unconditional face-lane mismatch amplifies learned corruption | **High as amplifier; prior exact-style A/B** |
| 6 | Hard fixed masks and stale PhotoMaker bboxes turn drift into collage/ghost faces | **Medium/high; code + prior smoke evidence** |
| 7 | Current one-ID comparison is biased by holdout leakage and face-scale mismatch | **Certain; hashes + metadata** |
| 8 | Cosmic prompt truncation removes pose/background supervision much more often | **High for full Cosmic; tokenizer audit** |
| 9 | SDXL branch base buffers are copied into RealVis validation processors | **Certain in current code; effect size unmeasured** |
| 10 | General ref quality/detection failures hurt the full dataset | **Medium; visual/data audit, but selected identity is clean** |

The main practical implication is that continuing to tune only
`masked_loss_step`, face-loss weight, or training duration is unlikely to make
Cosmic robust. The next experiments should first remove the global CA
confounder and then isolate target diversity and 256-reference preprocessing
under a leak-free same-identity design.
