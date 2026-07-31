# Cosmic Full dataset: issues, attempted fixes, and next experiments

**Date:** 26 July 2026

**Branch / local commit:** `test` / `c04970f342a186d1092f07f9a08d7d8a797383e8`

**Scope:** data selection, target/reference pairing, filtering, and reference
formatting for the reference-conditioned branched-self-attention protocol

**Status:** analysis and recommendations only; no training job was launched

## Executive recommendation

Keep Cosmic Full, but stop treating its current `face_paths` package as a
conventional multi-view identity dataset.

The current package is better described as:

```text
one 1024px target
    + 2–10 target-specific 256px face-reference candidates
    + target/reference ArcFace similarity scores
    + no stable explicit identity ID joining targets
```

The adapted loader has made this package mechanically safe to train, but it
has not repaired that identity structure. The next useful experiments should:

1. select references by an explicit, audited policy rather than uniformly from
   every candidate;
2. construct a high-confidence multi-target identity tier and split validation
   by identity, not file;
3. filter or curriculum-sample target face scale and image quality; and
4. test real context only after obtaining native full-scene references.

Do **not** run another 40%-versus-60% margin or 256-versus-512 content arm on
the current 256px reference assets. A fresh manifest-geometry audit found:

- 40% and 60% produce the same crop box for `99.9922%` of the `180,623`
  valid reference candidates;
- 40% already returns the entire 256px source for `99.9922%` of candidates;
- 60% returns the entire source for `100%`;
- 512px content is bicubic upsampling from an at-most-256px source, so it adds
  no reference information.

The recent 40/60/512 matrix therefore establishes that the **full existing
256px reference asset** is a workable training input. It does not establish
an optimal real-context margin or source resolution.

## Non-negotiable experiment invariants

All proposed BA experiments retain:

```text
use_branched_attention = true
pipeline.pose_adapt_ratio = 0.0
pipeline.ca_mixing_for_face = false
reference_face_kv_weight = 1.0
disable_branched_ca = true
```

Keep the canonical full-96 prompts, seeds, references, bboxes, RealVisXL
validation base, scheduler, inference steps, CFG, and metric definitions
unchanged. Add dataset-proximate holdouts; do not replace the canonical
protocol.

Train the initial comparisons to the same 4,000-step budget and inspect the
existing 1k/2k/3k/4k checkpoints. The current matrix peaks at 3k, but retaining
the 4k checkpoint preserves direct comparability.

## What actually helped in previous experiments

The useful changes fall into three different categories. Only the first group
is evidence that the learned model improved; the second made training and
evaluation reliable; the third produced attractive numbers for the wrong
reason and must not be treated as progress on reference-conditioned BA.

### Evidence-backed model and data improvements

| Change | Controlled evidence | What improved | Decision |
|---|---|---|---|
| Use diverse target views instead of repeating one target | Task D `multi_full` versus `single_full`: text/ID `25.7448 / 0.2357` versus `25.0182 / 0.1853` | `+0.7266` text, `+0.0504` ID, and visibly better anatomy | Build explicit multi-target identity groups; do not simulate an identity dataset by repeating one image |
| Use an identity-focused crop during training, while retaining a full-scene reference at inference | Task D `multi_cosref` versus `multi_full`: `26.9297 / 0.3375` versus `25.7448 / 0.2357`; roughly 10/12 versus 6–7/12 plausible outputs | `+1.1849` text and `+0.1018` ID, with the clearest controlled visual improvement | Preserve the train/inference stage distinction; do not infer that tight inference crops are safe |
| Retain real visual context rather than blank padding | One-identity 40%-context run: `26.7409 / 0.3076`, mostly coherent; the same tight content on a neutral 1024px canvas: `26.2995 / 0.1377`, catastrophic in about 10/12 | Attached anatomy and much higher identity similarity | Test context only with real source pixels; reject blank canvases as a substitute |
| Disable branched cross-attention while retaining reference-conditioned branched self-attention | Task A CA-off versus the matched historical CA-on baseline: `24.7982 / 0.1418` versus `23.7565 / 0.0351` | Better scenes and bodies, `+1.0417` text and `+0.1067` ID | Keep branched CA disabled; this reduced global corruption but did not solve the face-local failure |
| Put pose and background early in the caption | Matched full-Cosmic step-3k arms: pose-first `26.6846 / 0.3606`; legacy `27.0054 / 0.3457` | `+0.0149` ID at a `-0.3208` text-score trade-off | Keep pose-first fixed for data experiments; it is a modest identity improvement, not an anatomy cure |
| Select the 3k checkpoint instead of assuming longer training is better | Every final full-Cosmic arm peaked in ID at 3k and regressed at 4k; the leading arm fell from `0.3606` to `0.3422` | Better identity without changing the training recipe | Train to 4k for comparison, but use 2k/3k gates and treat 3k as the current selection point |

The strongest clean data lesson is the combination of **diverse target views**
and **identity-focused training references**. Task D isolated both effects with
one verified identity and one full-scene validation reference. Its best
`multi_cosref` checkpoint was step 2,500 at text/ID
`26.6471 / 0.3591`, although Drumming and Jumping still failed. The result is
promising rather than a solved recipe.

The reference policy is stage-dependent. Task B showed that feeding a tight
256px crop at inference could recreate pasted or displaced faces even on a
healthy checkpoint, whereas a full-scene reference mostly restored attached
anatomy. Thus the evidence supports a crop as a **training regularizer**, not
as the canonical inference input.

For the current full-Cosmic package, “40% crop” should now be read as “use the
complete existing 256px asset.” The live audit shows that nearly every 40% and
60% crop is pixel-equivalent because the source is already 256px. The final
matrix therefore confirms that the complete asset is viable; it does not prove
that 40% is an optimal context margin or that native 512px detail is unhelpful.

### Pipeline and evaluation changes that helped

These changes did not isolate a model-quality gain, but they made the
full-Cosmic result trainable and auditable:

- the adapted loader consumes the real manifest, requires different
  target/reference paths, validates boxes and decoded geometry, propagates
  transforms to reference bboxes, and keys the conditioning cache by reference
  policy and flip state;
- the conservative 192px target-face filter reduced 59,143 raw rows to 22,140
  mechanically safer targets, while invalid reference boxes were removed;
- a 64/64 decode-and-bbox preflight and fail-closed checks prevented malformed
  samples or silent processor installation failures from contaminating runs;
- asynchronous CUDA, CUDA ONNX Runtime, and two loader workers reduced runtime
  from roughly 5–7 to 2.06–2.10 seconds/step; and
- multistep full-96 validation exposed identity-specific failures hidden by the
  earlier Eddie-only panel and showed that aggregate identity similarity can
  reward face fragments.

These are infrastructure and decision-quality improvements. In particular,
the 192px filter is a safe starting point, not yet evidence that 192px is the
optimal target threshold.

### Apparent improvements that should not be credited

- `pose_adapt_ratio=1.0` reached full-96 ID `0.5136` with 12/12 coherent
  outputs, but it zeroed the spatial reference-face K/V path. Plain
  PhotoMaker also slightly beat it in the matched panel. This is architectural
  drift, not an improvement to reference-conditioned BA.
- Numeric 40%-versus-60% margin and 256-versus-512 output comparisons did not
  add context or native detail to the already-256px references.
- Reference-only training, neutral-canvas padding, repeated single targets,
  CFG 1, and the historical CA-on path all failed their visual or controlled
  comparison gates.
- Running to 4k did not improve the promoted result; identity regressed after
  3k in every final arm.

The best evidence-backed recipe to carry forward is therefore: verified
multi-view targets for each identity; a different same-identity,
identity-focused training reference; a full-scene reference at inference;
pose-first captions; branched CA off; `pose_adapt_ratio=0`; and checkpoint
selection around 3k using full-96 visual anatomy gates. The current manifest
cannot yet realize the first requirement reliably because it has no stable
identity ID joining multiple targets.

## What was wrong with Cosmic Full

### 1. The historical loader represented the wrong data contract

`CosmicDoubledTrain` combines older Cosmic metadata and does not consume the
new manifest's `face_paths`, per-reference boxes, or reference scores. Unless
a separate mapping is supplied, it can also use the target itself as the
reference.

This made it unsafe for the current full-Cosmic package and conditioning-cache
semantics.

**Attempted fix:** the isolated
[`CosmicLargeAdaptedTrain`](../src/datasets/cosmic_large_adapted.py) loader now
reads the actual manifest, validates boxes, samples a distinct reference,
propagates transforms to the bbox, exposes paths for audits, and includes the
reference policy and flip state in the cache key. Historical behavior remains
unchanged for replay.

### 2. Most raw targets have faces below the initial safe scale

The live manifest contains `59,143` targets. The current 192px minimum
face-side filter removes `37,003` (`62.57%`) and retains `22,140`.

Target counts available at stricter thresholds are:

| Minimum target-face short side | Targets |
|---:|---:|
| 192px | 22,140 |
| 224px | 16,662 |
| 256px | 12,669 |
| 320px | 7,313 |
| 384px | 3,808 |

Small and high-motion faces were among the hardest historical prompts, so
simply restoring all small faces would likely reintroduce instability.

**Attempted fix:** start conservatively at 192px. This made training
mechanically reliable, but no controlled target-scale curriculum or
scale-balanced sampler has been tested.

### 3. “Same identity” exists only within a target-specific reference package

For the accepted set:

- there are `180,623` valid reference paths;
- every reference path occurs once;
- the first-reference parent directory is unique for all `22,140` targets;
- the manifest has no `identity_id`, `person_id`, or `id` field.

The fallback `identity_id` is therefore a per-target pseudo-ID. A target has
multiple candidate references, but the package does not join multiple 1024px
targets into one verified identity. This is materially weaker than Task D,
where eight diverse targets belonged to one sealed identity.

**Attempted fix:** target/reference paths are required to differ, and Task D
demonstrated that diverse targets help. The full package still lacks the
stable grouping needed to reproduce that advantage at scale.

### 4. Reference scores are available but ignored by the current sampling rule

The upstream `face_scores` were generated as target/reference ArcFace cosine
similarities, not face-detector confidence scores. The filtered manifest has
already removed scores below `0.70`.

For accepted targets:

- candidate score median: `0.7651`;
- median per-target mean score, equivalent to the expected uniform-random
  reference score: `0.7589`;
- median best-reference score: `0.7971`;
- `19,362` targets have at least one valid candidate at `>=0.75`;
- `10,474` have at least one at `>=0.80`;
- `1,585` have at least one at `>=0.85`.

The loader currently chooses uniformly from all surviving candidates. That
preserves reference diversity but does not distinguish a strong match from a
borderline `0.70` match. Conversely, selecting only the highest score could
favor near-duplicate pose or appearance and remove useful diversity. This is
an unresolved, directly testable trade-off.

**Attempted fix:** an optional `min_reference_score` exists, but it has been
left `null`; no top-k, temperature, or diversity-aware policy has been tested.

### 5. Path inequality does not exclude content-level leakage

The new loader prevents exact path equality. It does not reject:

- byte duplicates under different paths;
- perceptual near-duplicates;
- the same photograph at a different crop or resolution; or
- the same validation identity appearing elsewhere in Cosmic Full.

The canonical validation identities include public figures likely to occur in
web-scale imagery. No target/reference embedding audit has yet established
that the full-96 identities are held out from Cosmic Full. Until that audit is
done, the full-96 result measures a consistent benchmark but cannot support a
strong held-out-identity generalization claim.

**Attempted fix:** the earlier one-ID exact image overlap was removed and the
controlled Task D artifact rejects perceptual duplicates. Equivalent checks
have not yet been applied to the full manifest.

### 6. Current references are already face-focused 256px images

The reference bboxes occupy a median `41.60%` of the original 256×256 reference
area. The median reference-face short side is `142px`. These are not native
full-scene reference images.

The crop implementation caps the crop at the source dimensions. On the live
manifest:

| Requested policy | Fraction returning the full 256×256 source |
|---|---:|
| 20% margin | 77.2177% |
| 40% margin | 99.9922% |
| 60% margin | 100.0000% |

The 40% and 60% crop boxes are identical for all but 14 of 180,623 candidates.
This explains why their final trajectories were nearly indistinguishable: the
named intervention changed almost no reference pixels.

The one-ID margin experiment was different: it cropped native 1024px
full-scene references. Its favorable result therefore did not transfer the
same image transformation to the full package, even though both configurations
used the label “40% margin.”

**Attempted fixes and results:**

- tight crops at inference causally recreated pasted/misregistered faces;
- full-scene inference references were much safer;
- a neutral 1024px canvas around 256px content was catastrophic, showing that
  padding is not real context;
- using the entire existing 256px reference during training was stable enough
  to remove the widespread global failure;
- 512px resizing did not help because it only interpolated existing pixels.

### 7. Target filtering checks geometry, not content quality

The accepted-target gate validates the bbox and its size. It does not
currently filter or stratify:

- blur, compression, watermarking, or non-photographic content;
- collage/product layouts;
- severe occlusion or profile pose;
- mismatch between the annotated bbox and the face selected by the embedding
  model;
- multiple-face ambiguity;
- body-mask quality or foreground coverage.

A small visual spot audit found at least one accepted collage/product target;
this is illustrative, not a population estimate. The manifest supplies a body
mask path, but the adapted loader does not use it as a quality signal.

The newer `gathered_data_cosmic_large_filtered2.json` is identical to the
current manifest on all common fields and adds only `has_simple_back`.
Among accepted targets, `938` are flagged `true` and `21,202` `false`. This
flag is useful as a stratification variable, but its construction and accuracy
need auditing before it becomes a hard filter.

### 8. Face selection can disagree with the supplied bbox

The data path now propagates the correct bbox through crop/resize/flip
operations. However, downstream face embedding can still choose the first or
largest detected face rather than the detection with maximum overlap with the
supplied bbox. A valid box also does not guarantee a successful recognition
embedding.

**Attempted fix:** preflight validates boxes and decoded geometry. A future
data tier should additionally require detector/bbox overlap and fail closed on
missing identity embeddings.

### 9. Captions are long and appearance-first in the legacy mode

The combined legacy prompts range from 36 to 97 whitespace-delimited words,
with a median of 64. Pose and background appear after facial appearance and
can be weakened by truncation.

**Attempted fix:** pose-first captions move class/trigger, pose, and background
ahead of remaining appearance and cap at 55 words. In the final matrix they
improved the identity/text trade-off relative to legacy captions. This policy
should remain fixed while testing data changes.

### 10. Runtime problems initially obscured data behavior

`CUDA_LAUNCH_BLOCKING=1`, CPU ONNX Runtime fallback, and worker settings made
early Serv runs take 5–7 seconds/step.

**Attempted fix:** asynchronous CUDA, CUDA ONNX Runtime, and two loader workers
reduced training to roughly 2.06–2.10 seconds/step. This is solved operationally
and should not be varied in dataset experiments.

## What the experiments establish

### Observed

- Task D: eight distinct targets plus cropped training references beat both
  full-scene training references and a repeated single target. At 4k,
  `multi_cosref` reached text/ID `26.9297 / 0.3375`.
- Task B: a tight 256px reference at inference recreated malformed faces on a
  healthy checkpoint; a full-scene wrong-identity reference changed identity
  but mostly restored attached anatomy.
- Full Cosmic: all four final arms trained cleanly on 22,140 accepted records.
- The strongest final checkpoint is the 40%-labelled/full-256, pose-first
  step-3k model at text/ID `26.6846 / 0.3606`.
- Every arm peaked on identity at 3k and declined by 4k.
- Jisoo retains a repeated malformed-face cluster; Marion and small/action
  faces remain weaker.

### Revised interpretation after the manifest audit

- The full-Cosmic result supports **using the complete existing 256px
  reference asset**, not a specifically optimal 40% crop.
- The 40%-versus-60% result is not evidence about real-context width because
  the transformed references are almost always identical.
- The 256-versus-512 result is not evidence that native high-resolution
  reference detail is useless; it only shows that upsampling 256px content is
  useless.
- The remaining improvement from better target/reference organization may be
  larger than the improvement available from more numeric margin sweeps.

## Recommended experiment sequence

### Priority 0 — sealed dataset audit before another training run

Create a versioned, immutable audit manifest without changing the source
package. Record every inclusion/exclusion reason.

Required fields:

1. target detector/bbox IoU, face count, face short side and area;
2. blur/compression and basic image-quality scores;
3. body-mask foreground fraction and target-face/mask consistency;
4. reference detector/bbox IoU, embedding success, ArcFace score, bbox area,
   and source resolution;
5. byte hash and perceptual hash for target/reference near-duplicate checks;
6. candidate identity-cluster ID and split assignment;
7. overlap against every canonical validation reference/identity.

Manually audit stratified target/reference pairs from at least these ArcFace
bands:

```text
[0.70, 0.75), [0.75, 0.80), [0.80, 0.85), >=0.85
```

The score must be calibrated as a same-identity probability on this package;
it should not be treated as ground truth merely because it is available.

Add a pre-launch intervention audit:

- report the percentage of transformed reference pixels/crop boxes that differ
  from the control;
- reject an experimental arm that changes almost no inputs unless it is
  explicitly registered as a no-op/reproducibility control;
- report the native source size and resize factor, distinguishing upsampling
  from added source detail.

### Priority 1 — reference-selection factorial on a fixed target subset

**Question:** does a more deliberate reference choice improve identity without
removing useful cross-view diversity?

Use one fixed subset with at least three audited reference candidates per
target. A practical first tier is candidates at `>=0.75`; approximately 15.6k
targets have at least three before the final bbox/duplicate audit.

Hold target order, sample count, augmentations, captions, architecture, and
validation fixed. Use the full 256px reference asset explicitly:

```text
reference_crop_margin = null
reference_content_size = 256
reference_canvas_size = null
```

Arms:

| Arm | Reference policy | Purpose |
|---|---|---|
| A | Uniform among audited candidates | Exact current-selection control on the fixed subset |
| B | Highest ArcFace score, excluding near-duplicates | Test maximum match quality |
| C | Sample among top 3 with a fixed softmax temperature | Retain view diversity while favoring quality |

Precompute the candidate schedule by target and epoch so worker timing cannot
change which reference each arm sees.

Promote only if the candidate improves or matches the current step-3k full-96
anatomy result, removes the Jisoo cluster, and does not trade identity gain for
recognizable copied fragments. Review results per reference-score band.

### Priority 2 — high-confidence multi-target identity tier

**Question:** can Cosmic Full reproduce Task D's target-diversity benefit when
identities are grouped correctly?

Use the existing target/reference embeddings to create mutual-nearest-neighbor
identity components, then manually audit a sample of every acceptance band.
Requirements for the first tier:

- at least three distinct 1024px target images per identity;
- target/reference byte and perceptual inequality;
- a detector that overlaps the supplied bbox;
- multiple poses or scenes rather than near-duplicate crops;
- an identity-level train/validation split.

Compare:

| Arm | Targets | References |
|---|---|---|
| A | Existing one-target pseudo-ID sampling | Current audited 256px candidates |
| B | Multiple 1024px targets from a verified identity group | Same 256px candidate policy as A |
| C | Multiple 1024px targets | A different 1024px scene from the same group, formatted by normalized face occupancy |

Match optimizer steps and sampled target count across arms. Arm B isolates
target grouping; Arm C then tests whether native scene context adds value.

If too few reliable groups can be recovered, use an audited stable-ID anchor
dataset as a mixture rather than pretending the pseudo-IDs are equivalent.
The adjacent `filtered_ids3.json` artifact contains 3,866 named groups and
127,283 images, with at least five images per group, but it requires its own
provenance and loader audit before mixing.

### Priority 3 — target face-scale and quality curriculum

**Question:** do large, clean faces teach stable identity routing before
small/action faces are introduced?

Keep the winning reference-selection policy fixed and compare:

| Arm | Target sampling |
|---|---|
| A | Current uniform sampling, face side >=192 |
| B | High-quality tier, face side >=256; 12,669 targets before other filters |
| C | Curriculum: >=320 through 1k, >=256 through 2k, then scale-balanced >=192 through 4k |

For all arms, report results in face-scale bins rather than only as one mean:

```text
192–255, 256–319, >=320
```

Add separate tags for frontal/profile, occlusion, action, multiple faces, and
simple/complex background. Do not silently remove hard cases from validation.

If the quality audit finds many bad targets, add a second controlled comparison
between:

- a strict tier requiring detector/bbox agreement, usable embedding, and no
  severe blur/collage; and
- the same tier plus lower-quality records sampled at reduced weight.

This distinguishes “bad data should be removed” from “hard data should be
introduced later.”

### Priority 4 — real-context and occupancy experiment

**Question:** how much genuine head/shoulder/scene context should a training
reference contain?

This experiment is blocked until native full-resolution reference scenes are
available. Recover them from the upstream source or use a different 1024px
target from a verified identity group as the reference.

Define reference formatting by **post-transform face occupancy**, not a margin
whose crop saturates:

| Arm | Native source | Target face occupancy in reference | Output |
|---|---|---:|---:|
| A | Current 256px asset | Existing distribution; median area ~41.6% | 256px |
| B | Native full scene | Standardized face area around 20–25% with real context | 256px |
| C | Native full scene | Full scene or face area around 10–15% | 256px |
| D, conditional | Native crop with at least 512px real content | Same occupancy as B | 512px |

Using 256px output for A–C isolates context from tensor resolution. Run D only
when the source crop contains at least 512 native pixels; do not upscale.

A later augmentation arm may jitter real-context occupancy and face position
within safe bounds, with exact bbox propagation. It should follow the
deterministic occupancy comparison, not replace it.

### Priority 5 — simple-background curriculum, lower priority

After auditing the `has_simple_back` flag, test whether its 938 accepted simple
background targets are useful for a short geometry-first warm-up, followed by
the diverse target set. This is lower priority because Task D already shows
that target diversity matters, and 938 examples may overemphasize portraits.

## Validation and decision gates

Every experiment should include:

1. canonical full-96 at steps 0/1k/2k/3k/4k;
2. a dataset-proximate holdout split by verified identity;
3. per-identity and per-face-scale visual anatomy scores;
4. ID and text similarity, clearly secondary to coherent face anatomy;
5. exact source/reference hashes and the realized reference candidate schedule;
6. counts of detector failures, bbox mismatches, near-duplicates, and filtered
   samples;
7. a matched plain PhotoMaker control for any candidate considered for
   promotion.

Required data gates:

- no target/reference exact or perceptual duplicate;
- no train/validation identity overlap in the new dataset-proximate holdout;
- detector-selected identity face overlaps the supplied bbox;
- every experimental transform changes the intended fraction of inputs;
- step-0 outputs remain identical across matched training arms.

## What not to spend compute on now

- another 40%-versus-60% crop on the current 256px references;
- another 512px resize of a 256px source;
- neutral-canvas padding;
- repeated single-target training;
- more legacy-versus-pose-first caption runs before data changes;
- `pose_adapt_ratio > 0` or CA mixing as a purported BA fix;
- a 20k run before a data-policy arm passes identity-level visual gates.

## Recommended immediate action

The most efficient next action is the no-GPU audit followed by the three-arm
reference-selection factorial. It uses fields already present in the manifest,
requires only localized backward-compatible loader toggles, and tests a real
data change.

In parallel, begin the identity-clustering and validation-overlap audit. That
work determines whether Cosmic Full can become a genuine multi-view identity
dataset or should instead remain a lower-weight source of diverse target
scenes paired with a separate stable-ID reference dataset.

## Evidence sources

- [Current project handoff](../docs/handoffs/LATEST.md)
- [Tasks A–D results](2026-07-25_cosmic_large_tasks_a_d_results_handoff.md)
- [Earlier Cosmic training recommendations](2026-07-25_cosmic_large_training_recommendations_and_experiments.md)
- [Full-Cosmic 4k/full-96 results](../docs/experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
- [Final four-arm multistep result](../docs/experiments/2026-07-26_current_four_full_cosmic_4k_runs_handoff.md)
- [`CosmicLargeAdaptedTrain`](../src/datasets/cosmic_large_adapted.py)
- [Reference transform implementation](../src/datasets/reference_policy.py)
