---
title: "E13 vs BC_E13: BigCelebs dataset root-cause analysis"
subtitle: "Fixed-96 results, per-identity/per-prompt interactions, dataset audit, and BC_E13_ds1–ds3 plan"
date: "9 August 2026"
---

# Executive conclusion

BC_E13 did **not** fail to learn identity. Its best fixed-96 identity score,
`0.399010 @16k`, is essentially tied with E13's best/final `0.399799 @24k`
(`−0.000789`). The disappointment is that a 7.35-times larger manifest did not
produce the expected gain, and BC_E13 then drifted down to `0.389430 @24k`
while E13 remained at `0.399799`. At the matched 24k endpoint the difference is
`−0.010369` ID_SIM and `−0.4766` text similarity.

The main cause is **dataset usage under a fixed 48,000-target budget**, followed
by a strong **portrait-versus-scene domain shift**:

1. E13 makes one full pass over 47,500 images and repeatedly exposes all 2,561
   identities. BC_E13 consumes only 13.74% of 349,348 BigCelebs images. It is
   expected to see about 31,480 of 68,648 identities; about 21,382 of those are
   seen exactly once and only about 10,097 at least twice. The extra data mostly
   becomes unseen breadth, not repeated evidence that can refine identity.
2. BigCelebs is an identity-rich portrait corpus, not a scaled version of the
   E13 corpus: 83.97% of its captions say portrait/close-up, versus 0.324% for
   Large Dataset. Its median face side is 410 px versus 255 px. Standing,
   hands/holding, and multi-person context are much rarer.
3. The validation panel confirms this interaction. BigCelebs strongly improves
   Skiing (`+0.07929` at 24k; 8/8 identities win) and Jensen (`+0.06016`), but
   consistently regresses expressive, occluded, or dynamic cells such as
   Jisoo/Crying, Eddie/Night ride, and Lex/Jumping.
4. The current loader samples shuffled **images**, not balanced identities,
   selects an unrestricted random distinct reference, and horizontally flips
   directional targets without changing left/right caption text. That creates
   an estimated wrong-direction rate of 6.36% for BigCelebs versus 2.52% for
   Large Dataset—about 1,846 additional mismatched rows in 48,000 draws.

There is no evidence of broad release corruption: the sealed release passes
path/decode/bbox/trigger/validation-overlap checks, and a deterministic 16-image
caption audit found descriptions aligned with images. There is, however, a
proven provenance defect in the auxiliary `caption_changes.jsonl`: all 71,321
rows carry the same `path`, so it cannot be used for row-level caption-change
auditing. This file is not consumed by training.

The three recommended dataset-only runs, in priority order, are:

1. `BC_E13_ds1_repeatdepth_balanced_24k_full96_r1`  
   Test whether Large-like identity repeat depth recovers the late ID_SIM loss.
2. `BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1`  
   Test whether scene-rich targets plus clean identity references fix the
   prompt/identity interactions.
3. `BC_E13_ds3_large_anchor_2to1_24k_full96_r1`  
   Test whether BigCelebs adds identity diversity without replacing E13's
   scene-rich training distribution.

All three must inherit the complete E13/BC_E13 architecture, optimizer,
scheduler, losses, seeds, fixed-96 validation, and BA settings. Only training
dataset selection and scheduling may change.

# Controlled comparison and evidence

**E13** — `E13_large_ds_joint_shadow_sa128_24k_full96_r4`  
Immutable Comet key: `1cc0a02371094b24a6a02a4cc649f10c`  
Training data: Large Dataset, 47,500 images / 2,561 identities.

**BC_E13** — `BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1`  
Immutable Comet key: `c138db7c41ae435c8a7560f40cf5f58d`  
Training data: BigCelebs v2, 349,348 images / 68,648 identities.

Both runs use the same 24,000 optimizer steps, batch size 2, hard spatial BA
rank 128, generic rank 32, PhotoMaker-default rank 64, and joint optimization.
They use the same fixed-96 validation at step 0 and every 2k, including prompts,
identities, references, seeds, boxes, scheduler, inference settings, and metric
definitions. Both pin `pose_adapt_ratio=0` and
`ca_mixing_for_face=false`.

The comparison uses the immutable Comet records, the 13 per-image ID_SIM CSVs
for each run, matched 24k generated images, training metrics, loader/config code,
the exact manifests, and the sealed BigCelebs release metadata. Derived files
are reproducible from
[`build_analysis_assets.py`](assets/bc_e13_vs_e13_20260809/build_analysis_assets.py),
with paired endpoint values in
[`paired_24k.csv`](assets/bc_e13_vs_e13_20260809/paired_24k.csv) and numerical
summaries in
[`derived_summary.json`](assets/bc_e13_vs_e13_20260809/derived_summary.json).

The report distinguishes observation from inference. Caption regexes and face
box geometry are useful domain proxies, not semantic ground truth. Repeated
celebrity names are candidates for identity fragmentation, not proof that two
manifest IDs should be merged.

**Report-freeze caveat.** BC_E13's training and fixed-96 ID/text validation are
complete, but its deferred final face-quality scorer was still running on Serv
when this report was frozen. No BC_E13 face-quality values are inferred or
silently substituted. The compact seven face-quality curves remain a required
promotion check once the finalizer publishes them; none of the dataset findings
below depend on those pending values.

# Run-level results

## Headline metrics

| Comparison | E13 | BC_E13 | BC − E13 |
|---|---:|---:|---:|
| Best fixed-96 ID_SIM | 0.399799 @24k | 0.399010 @16k | −0.000789 |
| Matched 24k ID_SIM | 0.399799 | 0.389430 | −0.010369 |
| Matched 24k text similarity | 27.0347 | 26.5581 | −0.4766 |
| 24k paired cell wins | 56/96 | 40/96 | — |
| 24k paired-cell bootstrap 95% CI for BC − E13 | — | — | [−0.02488, +0.00441] |
| 24k median paired-cell delta | — | — | −0.01733 |

The 24k paired interval crosses zero and the exact two-sided sign test is
`p=0.1253`. The data therefore support a **modest endpoint regression with large
cell interactions**, not a statistically clean claim that BigCelebs is worse
for every validation case. Averaging each cell across 8k–24k gives a smaller
mean delta of `−0.00248` (41/96 BC wins; 95% CI
`[−0.01255,+0.00784]`).

![E13 and BC_E13 identity/text validation trajectories](assets/bc_e13_vs_e13_20260809/metric_trajectories.png)

Two temporal facts matter:

- BC_E13 reaches parity at 16k, then loses `0.00958` ID_SIM by 24k. E13 gains
  `0.00417` over the same interval.
- BC_E13 text similarity peaks early and trends down after 4k, while E13 stays
  materially higher late in training. This is consistent with a portrait-heavy
  target distribution learning identity appearance without enough scene/action
  supervision. It is an inference, not a direct causal measurement.

BC_E13 also has consistently lower logged loss and gradient norms. Across
16k–24k, mean loss is `0.12784` versus `0.13590`, and mean total gradient norm is
`0.01670` versus `0.02565`. Because architecture and loss definitions are
identical, BigCelebs supplies an easier/lower-gradient training distribution;
the close-up bias is a plausible explanation. Lower training loss here is not
evidence of better validation generalization.

![Training loss and gradient dynamics](assets/bc_e13_vs_e13_20260809/training_dynamics.png)

## Identity and prompt variation

At 24k, seven of eight validation identities are lower under BC_E13; Jensen is
the clear exception.

| Identity | E13 ID_SIM | BC_E13 ID_SIM | Delta | BC wins across 12 prompts |
|---|---:|---:|---:|---:|
| Eddie | 0.17422 | 0.15595 | −0.01826 | 5/12 |
| Elon | 0.50293 | 0.45869 | −0.04424 | 1/12 |
| Jennie | 0.40643 | 0.39781 | −0.00862 | 5/12 |
| Jensen | 0.47265 | 0.53281 | **+0.06016** | 10/12 |
| Jisoo | 0.45436 | 0.42183 | −0.03253 | 3/12 |
| Keanu | 0.46665 | 0.44765 | −0.01899 | 6/12 |
| Lex | 0.38031 | 0.36742 | −0.01289 | 4/12 |
| Marion | 0.34085 | 0.33328 | −0.00757 | 6/12 |

The prompt result is even more structured: Skiing wins for all eight identities,
Kickboxing also improves, while Reading, Laughing, Crying, Jumping, and Night
ride regress most.

| Prompt family | BC − E13 at 24k | Late mean, 8k–24k |
|---|---:|---:|
| Reading | −0.03782 | −0.01624 |
| Laughing | −0.03626 | −0.00098 |
| Crying | −0.03250 | −0.02169 |
| Jumping | −0.03171 | −0.02739 |
| Night ride | −0.02987 | −0.02188 |
| Chef | −0.02344 | −0.00787 |
| Angry | −0.01672 | −0.00676 |
| Rushing | −0.01381 | −0.00442 |
| Drumming | −0.00626 | −0.00615 |
| Dancing | +0.00383 | +0.00690 |
| Kickboxing | +0.02086 | +0.01356 |
| Skiing | **+0.07929** | **+0.06318** |

![Final and late ID_SIM deltas by identity and prompt](assets/bc_e13_vs_e13_20260809/paired_deltas.png)

The identity×prompt heatmap shows why a single headline mean is insufficient.
Jisoo/Crying is lower at all nine validation gates from 8k through 24k
(`−0.10574` late mean); Eddie/Night ride is also lower at all nine
(`−0.09980`). Jennie/Skiing gains `+0.19833` on the same late average. Gates are
temporally correlated and must not be treated as nine independent samples; the
consistency is descriptive evidence of stable interactions.

![Late identity by prompt interaction heatmap](assets/bc_e13_vs_e13_20260809/late_cell_delta_heatmap.png)

## Matched 24k visual examples

| Validation cell | E13 | BC_E13 | Delta | What the images show |
|---|---:|---:|---:|---|
| Jisoo / Crying, idx 55 | 0.474100 | 0.360445 | −0.113655 | Hand/eye occlusion exposes a facial-morphology change. |
| Eddie / Night ride, idx 10 | 0.167974 | 0.053471 | −0.114503 | BC has a more exaggerated open mouth and facial creasing. |
| Lex / Jumping, idx 81 | 0.297323 | 0.245375 | −0.051948 | A small action face remains harder under BC. |
| Jennie / Skiing, idx 26 | 0.230119 | 0.485343 | +0.255223 | BC is much more recognizable; goggles remain malformed. |
| Jensen / Kickboxing, idx 40 | 0.419518 | 0.497481 | +0.077963 | Centered face identity improves. |
| Jensen / Reading, idx 36 | 0.563326 | 0.644709 | +0.081383 | Static portrait-like rendering improves. |

![Matched full images and per-image metrics at 24k](assets/bc_e13_vs_e13_20260809/matched_24k_examples_full.jpg)

![Matched face crops and per-image metrics at 24k](assets/bc_e13_vs_e13_20260809/matched_24k_examples_faces.jpg)

The visual evidence agrees with the metrics: BigCelebs improves some centered,
portrait-like identity renderings, but performs less reliably under expression,
occlusion, and small/dynamic faces. The Skiing gain is real in ID_SIM but is not
artifact-free, so it is not an unconditional visual promotion.

# BigCelebs dataset and loader audit

## 1. The optimizer budget converts size into sparse breadth

![Manifest scale, identity depth, exposure and caption-domain audit](assets/bc_e13_vs_e13_20260809/dataset_audit_comparison.png)

| Property | Large Dataset | BigCelebs v2 |
|---|---:|---:|
| Images | 47,500 | 349,348 |
| Manifest identities | 2,561 | 68,648 |
| Mean / median images per identity | 18.55 / 18 | 5.09 / 4 |
| 2-image identities | 0 | 19,454 |
| 3-image identities | 0 | 13,913 |
| IDs with at least 8 images | 2,092 | 11,000 |
| Portion of images consumed in 48k target draws | 101.1% | 13.74% |
| Expected unique IDs seen | 2,561 | 31,480 |
| Expected IDs seen once | 0 | 21,382 |
| Expected IDs seen at least twice | 2,561 | 10,097 |

The current shuffled DataLoader is image-proportional: an identity with 30
images receives about 15 times the target probability of a two-image identity.
For BigCelebs, 48.6% of identity groups have only two or three images. At equal
optimizer steps, the run sees far fewer repeated observations per identity than
E13 and gets fewer consistent updates from distinct target/reference pairs.
This is the highest-confidence root cause.

Simply training longer would eventually expose more images, but matching E13's
one-pass coverage would require about 174,674 optimizer steps at batch 2—over
seven times the current budget. More steps are not the first recommended test;
the experiments below make the 24k budget informative.

## 2. Target-domain mismatch

| Metadata signal | Large Dataset | BigCelebs v2 |
|---|---:|---:|
| Portrait/close-up caption | 0.324% | 83.97% |
| Median face-box side | 255 px | 410 px |
| Median face-box area | 8.56% | 21.68% |
| Standing | 45.73% | 9.77% |
| Sitting | 10.95% | 4.42% |
| Hands/holding | 37.36% | 15.10% |
| Multiple people | 26.09% | 0.298% |
| Directional caption | 5.032% | 12.726% |

BigCelebs offers excellent identity-reference material, but its target images
are much less representative of the full-body, contextual and dynamic prompt
panel. Raising the minimum face side from 192 to the existing 256 variant would
not address this; it would keep 295,867 images and 62,673 identities while
likely increasing the close-up bias.

## 3. Target and reference roles are not separated

`BigCelebsTrain` inherits the Large Dataset behavior: each manifest image is a
target, and a distinct same-ID reference is selected with unrestricted runtime
`random.choice`. A reference may therefore contain occlusion, accessories,
hands, a second person, a small face, or an action pose. Conversely, a clean
large-face portrait is used as a target even though it may be more valuable as
an identity reference.

The manifest has enough metadata for a better role split without unavailable
Neb embeddings:

- 26,535 identities have at least one target whose face area is no larger than
  Large Dataset's 75th percentile and at least two same-ID references with face
  side at least 384 px.
- Among the deepest 2,561 BigCelebs identities, 2,313 satisfy the strict rule.
  Across the whole release, 13,848 identities satisfy it; choosing the deepest
  2,561 of that eligible pool yields 60,087 images, 23.46 images per identity
  on average, and a minimum depth of 14. The same strict cohort can therefore
  be used in ds1 and ds2 with no fallback.

This does not prove reference quality—caption and bbox rules cannot measure
blur or identity cohesion—but it is a stronger and reproducible baseline than
unrestricted random choice.

## 4. Horizontal-flip label noise

The loader mirrors the target image and face bbox with probability 0.5 but
leaves the caption unchanged. For captions containing left/right, half of
flipped rows are mislabeled. Expected wrong-direction rates are 2.516% for
Large Dataset and 6.363% for BigCelebs. Over 48,000 target draws, the BigCelebs
run therefore receives about 3,054 mismatched rows versus 1,208 under E13—about
1,846 extra mismatches.

This is not large enough to explain every metric gap, but it is a concrete data
bug and should be corrected in every new schedule by suppressing flips for
directional captions. Non-directional targets retain the same deterministic
50% flip rate.

## 5. Possible identity fragmentation, not yet a filtering rule

Splitting BigCelebs identity keys at the final `__` reveals 5,430 repeated base
names spanning 11,021 identity groups and 63,244 images (18.10% of the release).
Examples include Jessica Chastain, Emma Stone, George Clooney and Ryan Gosling.
This is consistent with one celebrity being split into multiple source IDs,
which further reduces repeat depth, but names can collide. Do not merge or
exclude these groups without ArcFace/visual verification. The proposed runs
retain exact manifest identity boundaries and log repeated-name membership for
later audit.

## 6. Release health and auditability

Observed healthy properties:

- exact manifest SHA-256 is pinned;
- all selected paths exist and fully decode as 1024×1024 RGB JPEGs;
- bboxes are valid and meet the 192 px minimum;
- captions contain exactly one trigger and fit both tokenizers;
- singletons are excluded, so distinct references exist;
- the fixed validation identities/images are disjoint.

A deterministic hash-selected 16-image audit found the manifest captions
visually aligned with their images and also illustrates the portrait dominance:

![Deterministic 16-image BigCelebs manifest audit](assets/bc_e13_vs_e13_20260809/bigcelebs_caption_audit_contact_sheet.jpg)

The auxiliary release file `caption_changes.jsonl` has a separate provenance
defect: all 71,321 lines report the same path while containing different
captions. Training reads final manifest captions rather than this log, so the
defect does not explain BC_E13's behavior; it should nevertheless be fixed
before the file is used for traceability.

## 7. Historical scheduled-policy evidence

The prior BigCelebs policy-v1 used broad identity weighting, scale scheduling,
top-three centroid references and flip safety, yet reached `0.3727 @12k` versus
the old uniform BigCelebs peak `0.3817 @18k`. That comparison used an older
contaminated architecture, so it cannot reject scheduling under E13, but it
does show that merely recreating broad sqrt-weight sampling is unlikely to be
enough. The new experiments specifically test repeat depth and target/reference
role separation.

# Recommended experiments

## Shared immutable contract

Every run below must inherit `BC_E13_big_celebs_joint_shadow_sa128_24k.yaml`
and change only `train_dataset_name` plus the corresponding schedule path.
Preserve:

- 24,000 optimizer steps, batch 2, one A100, `epoch_len=2000`;
- hard spatial BA rank 128, generic effective rank 32, PhotoMaker-default
  effective rank 64, all at the existing learning rate and schedule;
- checkpoint schema and validation shadow/restore behavior;
- fixed 96-image validation at step 0/every 2k, same seeds/prompts/references/
  boxes/scheduler/inference/metrics;
- `pipeline.pose_adapt_ratio=0` and
  `pipeline.ca_mixing_for_face=false` in training and validation;
- deferred canonical face-quality scoring.

Use a new opt-in sequential schedule loader. Do not change the existing
`big_celebs` loader or its defaults.

## BC_E13_ds1 — repeat-depth-balanced BigCelebs

**Run:** `BC_E13_ds1_repeatdepth_balanced_24k_full96_r1`

**Hypothesis:** BC_E13's principal loss comes from spending the 48k-row budget
on too many identities with too little repetition.

**Schedule:**

1. Require at least two scene-eligible targets and two canonical-reference
   candidates per exact manifest identity, then rank eligible groups by
   descending usable image count with a stable SHA-256 tie-break and take 2,561
   identities—the same count as E13. The measured cohort contains 60,087
   images, averages 23.46 images per ID, and has a minimum of 14.
2. Emit exactly 48,000 rows in identity round-robin order: every identity gets
   18 or 19 target visits.
3. Rotate target paths before reuse. Select a deterministic distinct same-ID
   reference from all other paths; do not add target/reference quality gates.
4. Retain a deterministic 50% target flip only when the caption has no
   left/right token.

This arm tests fixed-budget identity exposure while keeping raw BigCelebs
target/reference usage inside a cohort that can be held identical for ds2.

**Required Comet comment:**

> E13 base; dataset-only BC_E13_ds1 change: deterministic BigCelebs repeat-depth balancing over a 2,561-ID deep cohort with flip-safe captions. Architecture, optimization, loss, seeds, and fixed-96 validation are unchanged from E13/BC_E13.

## BC_E13_ds2 — scene targets with canonical references

**Run:** `BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1`

**Hypothesis:** BigCelebs portraits are useful identity references, but using
them predominantly as targets—and allowing arbitrary references—causes the
prompt/ID interaction.

**Schedule:** use the exact ds1 identity cohort and 18/19-row round-robin, then:

1. Allocate two of every three rows to scene-rich targets. Primary scene rule:
   target face-box area `<= 0.17154455184936523`, the measured Large Dataset
   75th percentile. The third row is unrestricted to retain portrait coverage.
2. Rank reference candidates by: face side at least 384 px; portrait/headshot
   caption; no multi-person, action, hands/holding, glasses/goggles, or hat/cap
   term; then larger face side and stable hash. Select among the top three
   deterministically and require `target != reference`.
3. Require at least two strict scene targets and two strict canonical-reference
   candidates for every cohort identity; the audited shared cohort satisfies
   this by construction. Fail instead of falling back if the sealed manifest
   does not reproduce those counts.
4. Apply the same directional-caption flip safety as ds1.

This is a metadata-only role split. It intentionally does not depend on the old
Neb ArcFace sidecars. If it wins, a later embedding audit can improve reference
ranking and test identity cohesion.

**Required Comet comment:**

> E13 base; dataset-only BC_E13_ds2 change: ds1 repeat-depth balancing plus a 2:1 scene-rich target quota and deterministic canonical same-ID references with flip-safe captions. Architecture, optimization, loss, seeds, and fixed-96 validation are unchanged from E13/BC_E13.

## BC_E13_ds3 — Large-anchored 2:1 mixture

**Run:** `BC_E13_ds3_large_anchor_2to1_24k_full96_r1`

**Hypothesis:** BigCelebs can be additive if it supplements rather than replaces
the E13 target domain.

**Schedule:**

1. Emit exactly 32,000 Large Dataset rows and 16,000 ds2-policy BigCelebs rows,
   interleaved deterministically in a 2:1 source ratio.
2. For the Large source, use a seeded image permutation matching the current
   image-proportional semantics, a deterministic distinct same-ID reference,
   and the same directional-caption flip rule. Do not filter the Large
   manifest.
3. For the BigCelebs source, use ds2 target/reference role selection and the
   same 2,561-ID cohort. This gives approximately six or seven BigCelebs rows
   per core identity while retaining the scene-rich E13 anchor.
4. Log per-source and per-role counts globally and in every 2,000-step window.

This is the safest arm for exceeding E13: it preserves most of the source
distribution that produced E13 while testing whether curated BigCelebs identity
diversity adds value.

**Required Comet comment:**

> E13 base; dataset-only BC_E13_ds3 change: deterministic 2:1 Large-to-BigCelebs mixture, with the BigCelebs share using ds2 scene-target/canonical-reference sampling and flip-safe captions. Architecture, optimization, loss, seeds, and fixed-96 validation are unchanged from E13/BC_E13.

# Implementation plan through Serv YAMLs

The implementing agent should produce the following localized, backward-
compatible artifacts.

## 1. Sealed schedule builder

Add `tools/datasets/build_bc_e13_dataset_schedule.py` with explicit modes
`ds1`, `ds2`, and `ds3`. Inputs must include both source manifest paths, image
roots, source SHA-256 values, schedule seed, row count, cohort size and policy
thresholds. Output one JSONL per experiment plus a summary JSON containing:

- schema/policy version and complete CLI arguments;
- source paths and SHA-256 values;
- schedule SHA-256;
- exact counts by source, identity, role, flip and fallback reason;
- target/reference distinctness and same-ID checks;
- per-2k-step-window exposure counts;
- cohort statistics, repeated-name flags, directional-caption audit;
- first/last row fingerprints for resume diagnostics.

Each JSONL row should carry at least: `schedule_index`, `source`, `phase`,
`identity_id`, `target_path`, `reference_path`, `target_role`, `reference_tier`,
`flip_target`, source manifest hash, target bbox, reference bbox and prompt.
The builder must be deterministic byte-for-byte and fail rather than silently
relax a rule; documented ds2 fallbacks are explicit policy tiers, not silent
relaxation.

## 2. Opt-in sequential dataset

Add a new dataset class such as
`src/datasets/big_celebs_e13_scheduled.py`. It should:

- load the sealed schedule in order and return exactly the chosen target,
  reference and flip decision;
- revalidate source/schedule hashes, path existence, same identity,
  target/reference inequality, bbox bounds, one trigger and row index;
- reuse existing image loading/preprocessing behavior;
- expose the existing sample keys and cache semantics unchanged;
- support deterministic resume/worker behavior;
- leave `BigCelebsTrain` and `LargeDatasetTrain` behavior untouched.

Add a sparse `AICODE-NOTE` at the schedule/hash invariant. No BA, pipeline,
trainer, loss, validation or metric code should change.

## 3. Hydra datasets and experiment configs

Add three opt-in entries to `src/configs/datasets/all_datasets.yaml`, for
example `bc_e13_ds1`, `bc_e13_ds2`, `bc_e13_ds3`, using environment-resolved
schedule paths. Add:

- `src/configs/BC_E13_ds1_repeatdepth_balanced_24k.yaml`
- `src/configs/BC_E13_ds2_scene_target_canonical_ref_24k.yaml`
- `src/configs/BC_E13_ds3_large_anchor_2to1_24k.yaml`

Each config defaults to `BC_E13_big_celebs_joint_shadow_sa128_24k`, overrides
only `train_dataset_name`, and contains its exact Comet comment above.

## 4. Preflight and experiment records

Create one immutable JSON record under `experiments/big_celebs/` for each exact
run name. Record the E13 baseline, controlled dataset delta, manifest/schedule
hashes, counts, config, launcher, Serv YAML, one-GPU request, validation
contract and required startup checks.

Extend or add a focused preflight that composes each Hydra config and verifies:

- all architecture/trainable tensor counts equal BC_E13;
- total steps, batch size, LR schedule, losses and validation are equal;
- only the training dataset selector/schedule differs;
- every schedule row satisfies its declared policy;
- old `big_celebs` composition remains unchanged.

Do not add broad vanity tests. Use config composition, import/compile, complete
schedule scan, sampled decode, old/new loader smoke, launcher shell syntax and
the existing architecture validator.

## 5. Launchers and Serv packages

Add one active launcher per run, rejecting ad-hoc Hydra overrides and requiring
the exact run/config/spec/schedule values. Use the existing deferred face-
quality finalizer and Comet-record preparation.

Build one standard one-A100 Serv package and YAML per run with the repository
package builder. The start scripts must pin the clean runtime commit/overlay,
BigCelebs v2 manifest SHA, Large manifest SHA for ds3, schedule SHA, image
roots, completion marker, validation bbox file and `photomaker_NS` environment.
They must execute preflight before Comet creation.

Before each submission, inspect all project-owned Running and Pending MLS jobs
and count actual one/two-GPU requests. These three named submissions have the
user-authorized temporary ceiling of eight project A100 GPUs; do not submit a
job that would exceed it. If MLS rejects a request for allocation/request
limits, do not retry without a new user request.

After acceptance, verify startup and require
`saved/<run_name>/comet_experiment.json` with a non-empty immutable key before
considering the launch valid. Record the MLS job ID, Comet key, package hash,
schedule hash and observed startup batch in each experiment JSON and the stable
handoff.

# Evaluation and decision rules

Primary endpoint is matched 24k fixed-96 ID_SIM; best checkpoint is a secondary
diagnostic, not a replacement for the pre-registered endpoint. For each run:

1. Compare headline ID_SIM/text similarity to both BC_E13 and E13 at every 2k.
2. Compute paired 96-cell bootstrap intervals and win counts at 24k.
3. Report the eight identity and twelve prompt means, the identity×prompt
   heatmap, and the specific persistent cells in this report.
4. Inspect matched images for face identity, prompt adherence, artifacts,
   face/body alignment and accessories; do not promote solely on ID_SIM.
5. Include the compact seven face-quality curves after deferred finalization.

A strong promotion should exceed E13's `0.399799` final ID_SIM without the
current `−0.477` text-similarity loss, avoid concentrated regressions on
Jumping/Night ride/Crying, and preserve or improve visual face quality. A run
that only increases Skiing/Jensen while losing the other groups is a subgroup
specialization, not a general winner.

# Bottom line

BigCelebs contains more identity material, but BC_E13's fixed budget turns that
material into shallow, portrait-heavy exposure. The correct next step is not a
blanket image deletion or the existing min-face-256 variant. It is to control
**which identities repeat, which images serve as targets versus references,
and how much of E13's scene-rich distribution remains**. The ds1–ds3 sequence
tests those mechanisms with the model and validation contract held fixed.
