# E13-E18 identity-similarity and visual analysis

**Date:** 6 August 2026  
**Scope:** successful E13-E18 runs listed in the
[`E13-E18 successful-run index`](2026-08-06_e13_e18_successful_run_index.md),
with `E0_large_ds_base_historical_r4_20k_full96_r1` as the fixed-panel
historical comparator  
**Primary decision signal:** `manual_val/id_sim` / per-image `IDSimBest` on
the fixed 96-image `manual_val` panel, interpreted together with full-image
and enlarged face-crop inspection  
**Status:** analysis and next-experiment design only; no job was launched

## Executive conclusion

E13 is the new fixed-panel leader: **0.39980 at 24k**. E14 reaches **0.39185
at 20k**. Both beat the prior E0 comparator's **0.37083 at 14k**:

- E13 minus E0 is `+0.02897`, with 67/96 paired wins and a panel-bootstrap
  95% interval of `+0.01756` to `+0.04029`.
- E14 minus E0 is `+0.02102`, with 65/96 paired wins and an interval of
  `+0.00856` to `+0.03306`.
- This is not just best-checkpoint selection. At the common 20k checkpoint,
  E13 and E14 beat E0 by `+0.03021` and `+0.02296`, respectively, each with
  68/96 wins and a positive paired interval. Both were also already ahead at
  8k.

The breakthrough is the **training-only PhotoMaker-default coadapter**, or
"shadow co-adaptation." E13/E14 jointly optimize hard spatial BA, the generic
effective adapter, and the PhotoMaker-default adapter; validation then
restores the pretrained PhotoMaker-default path. Persisting that trained
default path at inference is the dominant failure in E15: even with a tenfold
lower default-path LR and protected reconstruction, E15 falls `0.07398` below
E14 at their best checkpoints. Its images retain the intended body position,
but acquire a strong exaggerated/open-mouth expression bias and lose facial
morphology.

The average still hides important discrepancies:

- E13's gains over E0 are concentrated in Elon `+0.071`, Marion `+0.051`,
  Keanu `+0.041`, and Jisoo `+0.040`. Jennie, Jensen, and Lex change by less
  than `0.009` in their identity means. Eddie improves only `+0.021` and
  remains the weakest identity at `0.174`.
- E13 improves 11/12 prompt means, but **skiing regresses `-0.021`**. E14 also
  regresses skiing, dancing, and night ride. Ski-goggle/eyeglass duplication
  is the clearest remaining systematic artifact.
- The most run-sensitive cells are identity-by-prompt interactions, not whole
  identities or prompts. Keanu laughing ranges from `.169` to `.531`, while
  Jennie skiing ranges from `.123` to `.436` and is much worse in E13/E14
  despite their higher overall means.

Among the persistent-path experiments, E18 provides the only strong positive
element. Its identity-balanced, multi-reference training package improves
E15 by **`+0.03735`**, wins 65/96 matched cells, and improves 7/8 identity
means and 11/12 prompt means. It remains below E13 because the bad persisted
default path is still present. The highest-probability next step is therefore
to put E18's data/reference package on E13's shadow route.

The plateau remains an objective/optimization problem rather than something
likely to disappear with more data or unchecked longer training. E13 gains
only `+0.01181` from 8k to 24k; E14 gains `+0.00718` from 8k to its 20k peak.
The much larger BigCelebs history plateaued similarly. The next six parallel
one-GPU arms should test the strongest transferable data component, the
previously safe branch-local output component, their combination, a
metric-aligned differentiable identity objective, earlier LR decay, and exact
alternation of masked and full losses.

## Evidence, cutoff, and interpretation boundaries

The analysis uses the immutable Comet IDs recorded in the
[`successful-run index`](2026-08-06_e13_e18_successful_run_index.md). Local
downloads include complete metric histories, output logs, per-image ID tables,
and generated images. The fixed panel is eight identities crossed with 12
prompt families, one generated image per cell.

Scientific comparisons use the latest **complete** per-image table available
for trajectories and each run's best complete post-training checkpoint for
subgroup and visual comparisons:

| Run | Best reviewed checkpoint | Latest complete table at analysis cutoff |
|---|---:|---:|
| E0 historical | 14k | 20k |
| E13 | 24k | 24k |
| E14 | 20k | 24k |
| E15 | 8k | 24k |
| E16 | 8k | 24k |
| E17 | 10k | 16k |
| E18 | 12k | 22k |

The live data were refreshed at **2026-08-06 16:42 UTC**. E18's 22k
validation, which was initially in progress during analysis, subsequently
completed with 96/96 images and `ID_sim=.35401`; its complete per-image table
is included in the trajectory. It does not replace E18's 12k best checkpoint.
E17 still had no image step newer than its complete 16k panel at that cutoff.

Three boundaries matter when reading the numbers:

1. `IDSimBest` selects the best-matching detected face anywhere in an image;
   it does not prove that the face belongs to the intended body. Full-image
   attachment review is therefore a separate gate.
2. The paired bootstrap resamples the 96 fixed-panel cells. It measures panel
   stability, not new-identity, new-prompt, or training-seed generalization.
   E13-E18 each contribute one successful training trajectory.
3. Selecting each run's best checkpoint is optimistic. Common-step
   comparisons and full trajectories are included so the conclusion does not
   depend on peak selection.

The derived numerical data and figures can be regenerated with
[`build_analysis_assets.py`](assets/e13_e18_20260806/build_analysis_assets.py).
The normalized best-checkpoint cell table is
[`best_per_image.csv`](assets/e13_e18_20260806/best_per_image.csv), and the
machine-readable summary is
[`derived_summary.json`](assets/e13_e18_20260806/derived_summary.json).

## Aggregate `ID_sim`

![ID similarity trajectories](assets/e13_e18_20260806/id_trajectories.png)

| Arm | Scientific change | Step 0 | Best complete checkpoint | Latest complete checkpoint | Best minus E0 best |
|---|---|---:|---:|---:|---:|
| **E0 historical** | Historical shadow-coadapted comparator | .30187 | **.37083 @14k** | .36889 @20k | -- |
| **E13** | Shadow PM-default co-adaptation; face-only diffusion loss | .30212 | **.39980 @24k** | **.39980 @24k** | **+.02897** |
| **E14** | E13 + face/full/boundary protected loss | .30212 | **.39185 @20k** | .39074 @24k | **+.02102** |
| E15 | E14, but persist trained default path with differential LRs | .30212 | .31787 @8k | .31031 @24k | -.05296 |
| E16 | E15 + predicted-x0 PhotoMaker-CLIP identity proxy | .30212 | .31031 @8k | .30362 @24k | -.06052 |
| E17* | E15 + bounded residual identity-token CA | .30471 | .31188 @10k | .30773 @16k | -.05895 |
| E18* | E15 + identity-balanced multi-reference package | .30212 | .35522 @12k | .35401 @22k | -.01561 |

`*` E17 and E18 were still running when their last complete panels were
captured. Their available histories are sufficient for the component-level
conclusions below, but not completion claims.

### Paired best-checkpoint comparisons

| Comparison | Mean delta | Median delta | 95% paired panel interval | Left-run wins |
|---|---:|---:|---:|---:|
| **E13@24k - E0@14k** | **+.02897** | +.02442 | **+.01756 to +.04029** | **67/96** |
| **E14@20k - E0@14k** | **+.02102** | +.02687 | **+.00856 to +.03306** | **65/96** |
| E13@24k - E14@20k | +.00795 | +.00642 | +.00013 to +.01582 | 56/96 |
| E15@8k - E14@20k | **-.07398** | -.06577 | -.08946 to -.05835 | 11/96 |
| E16@8k - E15@8k | -.00755 | -.00372 | -.01927 to +.00405 | 45/96 |
| E17@10k - E15@8k | -.00599 | +.00258 | -.01875 to +.00625 | 51/96 |
| **E18@12k - E15@8k** | **+.03735** | +.03842 | **+.02168 to +.05253** | **65/96** |
| E18@12k - E13@24k | -.04458 | -.03611 | -.05902 to -.03011 | 25/96 |

E13's small best-checkpoint advantage over E14 is borderline in practical
size and depends partly on comparing 24k with 20k. The safe conclusion is that
both shadow arms form a new high-performing tier; E13 is the current numerical
base, while E14 is not evidence that protected reconstruction improves the
mean.

### Common-checkpoint comparisons against E0

| Step | E13 - E0 | E13 wins | E14 - E0 | E14 wins |
|---:|---:|---:|---:|---:|
| 8k | **+.02792** (`+.01551`, `+.04017`) | 67/96 | **+.02460** (`+.01387`, `+.03522`) | 67/96 |
| 14k | +.01225 (`+.00009`, `+.02413`) | 58/96 | +.00151 (`-.00994`, `+.01280`) | 49/96 |
| 20k | **+.03021** (`+.02104`, `+.03952`) | 68/96 | **+.02296** (`+.01317`, `+.03288`) | 68/96 |

Parentheses show the paired 95% panel interval. The temporary 14k dip is
shared by both shadow runs; both recover while the LR decays. E13 also exceeds
the older `0.39039 @24k` Large Dataset headline, while E14 narrowly exceeds
it. That older number came from a different historical trajectory, so it is
an aspirational cross-history reference rather than a paired comparison.

## Identity-level structure

![Best-checkpoint ID similarity by identity](assets/e13_e18_20260806/id_by_identity_best.png)

| Identity | E0 | E13 | E13-E0 | E14 | E14-E0 | E15 | E16 | E17 | E18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eddie | .153 | .174 | +.021 | .177 | +.023 | .130 | .130 | .121 | .153 |
| Elon | .432 | **.503** | **+.071** | .487 | **+.055** | .374 | .355 | .356 | .369 |
| Jennie | .408 | .406 | -.002 | .408 | -.000 | .302 | .302 | .336 | .376 |
| Jensen | .465 | **.473** | +.008 | .464 | -.001 | .412 | .403 | .387 | .439 |
| Jisoo | .414 | **.454** | **+.040** | .445 | **+.031** | .422 | .411 | .426 | .444 |
| Keanu | .426 | **.467** | **+.041** | .447 | +.021 | .356 | .325 | .344 | .383 |
| Lex | .379 | .380 | +.001 | **.392** | +.013 | .318 | .344 | .335 | .376 |
| Marion | .290 | **.341** | **+.051** | .315 | +.025 | .230 | .212 | .191 | .302 |

What the identity averages show:

- **Elon and Marion are the clearest shadow-coadaptation wins.** E13's
  improvement is not a uniform `+0.029`; those two identity means gain `.071`
  and `.051`.
- **Keanu and Jisoo also improve materially**, while Jennie, Jensen, and Lex
  are essentially flat at their identity means. Their individual cells still
  move substantially in opposite directions.
- **Eddie remains the hard tail.** E13/E14 faces are generally coherent and
  attached, but the metric remains only `.174-.177`. This looks more like a
  reference/evaluation/training-tail problem than a gross artifact problem.
- Across all seven reviewed best checkpoints, the largest identity-level
  ranges are Marion `.150`, Elon `.148`, and Keanu `.142`. Jisoo is the most
  stable at only `.044`.
- E18 versus E15 improves 7/8 identities: Jennie `+.074`, Marion `+.072`, Lex
  `+.058`, Jensen `+.028`, Keanu `+.027`, Eddie `+.023`, and Jisoo `+.022`.
  Elon is the only exception at `-.004`. That breadth is the strongest reason
  to transfer E18's data/reference package to the shadow base.

## Prompt-level structure

![Best-checkpoint ID similarity by prompt](assets/e13_e18_20260806/id_by_prompt_best.png)

| Prompt family | E0 | E13 | E13-E0 | E14 | E14-E0 | E15 | E16 | E17 | E18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reading | .428 | **.474** | **+.046** | .458 | +.031 | .350 | .353 | .350 | .416 |
| Rushing | .423 | **.464** | **+.042** | .445 | +.022 | .411 | .403 | .410 | .446 |
| Skiing | **.290** | .269 | **-.021** | .281 | -.009 | .212 | .202 | .172 | .196 |
| Drumming | .407 | **.433** | +.026 | .412 | +.005 | .348 | .357 | .343 | .400 |
| Kickboxing | .373 | .396 | +.023 | **.414** | **+.041** | .285 | .312 | .294 | .323 |
| Dancing | .310 | **.323** | +.013 | .299 | -.011 | .288 | .260 | .305 | .321 |
| Angry | .366 | **.429** | **+.062** | .417 | **+.050** | .352 | .318 | .303 | .387 |
| Laughing | .383 | .413 | +.029 | **.418** | **+.034** | .299 | .289 | .305 | .353 |
| Crying | .377 | **.428** | **+.051** | .416 | **+.039** | .315 | .297 | .324 | .361 |
| Chef | .389 | **.438** | **+.049** | .425 | **+.036** | .344 | .347 | .352 | .374 |
| Night ride | .416 | **.425** | +.009 | .411 | -.005 | .345 | .339 | .308 | .384 |
| Jumping | .289 | .306 | +.018 | **.307** | +.018 | .265 | .245 | .276 | .303 |

What the prompt averages show:

- E13 improves 11/12 prompt means. Its largest gains are angry `+.062`,
  crying `+.051`, chef `+.049`, reading `+.046`, and rushing `+.042`.
- **Skiing is the only E13 mean regression and remains a systematic failure.**
  E14 recovers about half of E13's prompt-level loss but does not fix the
  visual cause. E18's multi-reference package makes skiing `-.015` worse than
  E15 even while improving every other prompt.
- E14's protected loss redistributes performance rather than raising the
  average: relative to E13 it helps kickboxing `+.018`, skiing `+.012`, and
  laughing `+.005`, but hurts dancing `-.024`, drumming `-.021`, rushing
  `-.019`, reading `-.016`, and night ride `-.014`.
- Across all reviewed best checkpoints, the largest prompt-level ranges are
  crying `.130`, kickboxing `.129`, laughing `.128`, angry `.125`, reading
  `.125`, skiing `.117`, and night ride `.117`. Prompt means again understate
  the larger identity-by-prompt swings.

## Highest-discrepancy identity-by-prompt cells

![Largest cross-run face differences](assets/e13_e18_20260806/largest_cross_run_ranges_faces.jpg)

| Fixed-panel cell | Lowest best run | Highest best run | Range | E0 | E13 | E14 | E18 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Keanu laughing** | E16 .169 | E13 .531 | **.363** | .335 | **.531** | .469 | .356 |
| **Jennie skiing** | E18 .123 | E0 .436 | **.314** | **.436** | .230 | .226 | .123 |
| **Jennie angry** | E16 .194 | E14 .498 | **.304** | .378 | .459 | **.498** | .406 |
| Keanu crying | E16 .279 | E14 .537 | .258 | .416 | .493 | **.537** | .417 |
| Jennie reading | E16 .245 | E13 .501 | .256 | .455 | **.501** | .478 | .286 |
| Marion crying | E16 .132 | E18 .385 | .253 | .301 | .385 | .384 | **.385** |
| Jennie night ride | E16 .239 | E13 .490 | .252 | .434 | **.490** | .428 | .423 |
| Jennie kickboxing | E15 .218 | E14 .457 | .239 | .437 | .454 | **.457** | .333 |
| Jensen angry | E17 .285 | E13 .518 | .233 | .501 | **.518** | .466 | .485 |
| Lex skiing | E15 .139 | E14 .369 | .230 | .341 | .296 | **.369** | .260 |

The largest joint E13/E14 gains over E0 are also highly localized:

- E13: Keanu laughing `+.196`, Elon dancing `+.133`, Marion chef `+.128`,
  Elon reading `+.120`, Marion rushing `+.119`, Marion skiing `+.113`, Keanu
  angry `+.107`, and Elon night ride `+.105`.
- E14: Marion kickboxing `+.151`, Keanu laughing `+.134`, Jensen dancing
  `+.126`, Lex jumping `+.123`, Elon jumping `+.122`, Keanu crying `+.121`,
  Jennie angry `+.120`, and Elon night ride `+.118`.

The largest regressions must not be hidden by the average:

- E13: Jennie skiing `-.206`, Jensen night ride `-.080`, Keanu dancing
  `-.076`, Jisoo jumping `-.066`, Lex night ride `-.057`, and Keanu skiing
  `-.054`.
- E14: Jennie skiing `-.210`, Keanu dancing `-.166`, Marion dancing `-.095`,
  Marion drumming `-.085`, Jensen rushing `-.078`, and Lex night ride
  `-.075`.

The complete focused sheets are
[`largest joint gains`](assets/e13_e18_20260806/largest_joint_gains_over_e0_faces.jpg)
and
[`largest joint losses`](assets/e13_e18_20260806/largest_joint_losses_vs_e0_faces.jpg).

## Visual inspection: body alignment and artifacts

Every best-checkpoint panel was reviewed both as a full-image contact sheet
and as an enlarged crop around the fixed validation face box. The sheets are
available under
[`review_contacts_full/`](assets/e13_e18_20260806/review_contacts_full/) and
[`review_contacts_faces/`](assets/e13_e18_20260806/review_contacts_faces/).

### E13 and E14

The numerical advantage is visually credible. In both runs, the scene,
subject position, and body pose are effectively fixed while facial morphology
changes. Across the reviewed best panels, no E10-like subject relocation or
extra-person failure and no E12-like detached face plate/mask seam was
observed. The intended face remains attached to the intended body.

That does not mean the images are artifact-free:

- **Skiing:** the recurring defect is doubled or nested eyewear--ordinary
  glasses or facial features appear underneath/inside ski goggles. Jennie is
  the clearest metric-and-visual regression: E0 leaves enough visible facial
  structure to identify her, while E13/E14 occlude and warp it. E18 is often
  worse, especially for Jennie and Jisoo.
- **Crying:** faces remain attached, but hands, eyelids, cheeks, and tears can
  merge unnaturally. Jisoo is a recurring hard case. E18 sometimes puts
  fingers more directly through the eye region.
- **Jumping:** body association is good, but faces are very small. A few face
  pixels dominate the metric, so gains are inconsistent by identity even
  when the full image looks coherent.
- **Laughing:** E13's Keanu and Jensen gains correspond to plausible,
  attached faces rather than metric gaming. Mouth exaggeration remains a
  risk, but there is no detached identity fragment.
- **Chef:** several source scenes contain a background person. The foreground
  intended face appears attached in the reviewed panels, but `IDSimBest` could
  in principle select the other face; intended-box ID telemetry would make
  this auditable.

Focused hard-prompt sheets:

- [`skiing faces`](assets/e13_e18_20260806/hard_skiing_faces.jpg) and
  [`full images`](assets/e13_e18_20260806/hard_skiing_full.jpg)
- [`crying faces`](assets/e13_e18_20260806/hard_crying_faces.jpg) and
  [`full images`](assets/e13_e18_20260806/hard_crying_full.jpg)
- [`jumping faces`](assets/e13_e18_20260806/hard_jumping_faces.jpg) and
  [`full images`](assets/e13_e18_20260806/hard_jumping_full.jpg)
- [`laughing faces`](assets/e13_e18_20260806/hard_laughing_faces.jpg) and
  [`full images`](assets/e13_e18_20260806/hard_laughing_full.jpg)

### E15-E18

E15-E17 mostly preserve body layout because of the protected loss and lower
default-path LR, but they show an obvious inference-visible expression bias:
reading, drumming, kickboxing, night ride, and jumping often acquire an
unprompted wide-open mouth or scream-like expression, sometimes with red or
oversaturated facial color. This is not the catastrophic position drift seen
in E10, but it explains a large part of the identity loss.

E18 recovers facial morphology broadly relative to E15, particularly for
Jennie, Marion, and Lex. It does not remove the persisted-path expression bias
and it worsens accessory interactions in skiing. This makes E18's package a
good transferable data/conditioning component, not a deployable replacement
for E13.

## What E13-E18 suggest

### What works

1. **Intentional shadow co-adaptation is the main result.** Training the
   PhotoMaker-default and generic effective paths changes the optimization
   trajectory of BA; restoring the pretrained default for validation avoids
   expressing the damaging learned default-path appearance/expression drift.
   E13/E14 beat E0 at common 8k and 20k checkpoints, so this is not a lucky
   endpoint.
2. **Wider spatial BA is useful in the right interaction.** E11 showed rank128
   alone was insufficient, while E13 shows rank128 plus the historical
   co-adaptation interaction reaches a new tier.
3. **Identity-balanced multi-reference conditioning is the strongest clean
   transferable component in E15-E18.** E18's `+0.03735` over E15 is broad
   across cells and groups. E18 bundles balanced sampling and extra
   PhotoMaker references, so this suite does not isolate which half supplies
   the gain; the bundle should be transferred first, then ablated if it wins.
4. **Branch-local output rank remains a low-risk candidate from E2.** It was
   visually stable and modestly positive in the prior suite. It should now be
   retested on the much stronger E13 base.

### What does not work, or is not yet supported

1. **Persisting the trained PhotoMaker-default path is the principal failure.**
   E15 is `-.07398` versus E14 despite lower LRs and protected reconstruction.
   Do not use E15 as the base for performance-seeking arms.
2. **The E14 protected objective does not raise the overall mean.** It helps
   selected cells and may protect boundaries, but E13 is `+.00795` higher.
   Full-image supervision remains worth testing in a temporally separated
   alternating form, not assuming E14 proved it beneficial.
3. **E16's PhotoMaker-CLIP proxy does not improve identity.** E16 is
   `-.00755` versus E15 with an interval crossing zero and is worse at 24k.
   After 6k, the auxiliary is applied on only about 10.63% of logged batches;
   its window-mean raw loss is `.0310`, implying an applied-only raw loss near
   `.292`, and its approximate mean weighted contribution is only `.00155`
   versus total loss `.1587`. More importantly, PhotoMaker CLIP is not the
   ArcFace-like validation recognizer and may reward appearance or expression
   rather than identity. Simply increasing E16's weight is not justified.
4. **E17's residual identity-token CA is visually safe but has no mean gain.**
   E17 versus E15 is `-.00599` with a wide interval crossing zero. Its gate,
   residual-norm, and saturation telemetry were implemented in the processor
   but omitted from the configured writer metrics, so Comet cannot establish
   how strongly the path was used. Fix observability before changing rank or
   gate limits; do not prioritize a stronger E17 variant now.

### Why the plateau remains

E13 reaches `.38799` by 8k and adds only `.01181` through 24k. E14 reaches
`.38467` by 8k and adds `.00718` through its 20k peak. E15 and E16 actually
lose `.00756` and `.00670` from 8k to 24k. E18 makes a useful jump from 8k to
12k (`+.04247`) and then oscillates around the new level.

The shared scheduler holds `1e-4` through 14k and then decays to `1e-5` by
24k. Both shadow runs recover from their 14k dip after decay begins. Combined
with the similar BigCelebs plateau, this points to three ceilings:

- the face-noise objective is only indirectly related to recognition
  identity;
- a fixed single reference provides incomplete evidence under expression,
  pose, hand occlusion, and accessories;
- sustained high LR moves within the same basin rather than producing a new
  identity solution.

Therefore, more steps or more images alone are low-priority. Better reference
evidence, a safe identity-aligned objective, branch-local capacity, and
schedule changes are more plausible routes beyond `.400`.

## Recommended next six parallel one-GPU experiments

### Shared contract for all six

Use the exact E13 performance route unless an arm explicitly overrides one
field:

- hard spatial BA rank128, generic effective rank32, and training-time
  PhotoMaker-default effective rank64;
- `ba_lr=generic_adapter_lr=photomaker_default_lr=1e-4`;
- `validation_shadow_photomaker_default=true`, with the pretrained default
  restored for every validation/inference panel;
- 24k optimizer steps, batch size 2, one A100, step 0 and every-2k fixed
  full-96 validation/checkpointing;
- identical seeds, prompts, reference images, cached boxes, scheduler,
  inference steps, and `IDSimBest` definition;
- `pipeline.pose_adapt_ratio=0` and
  `pipeline.ca_mixing_for_face=false`;
- exact trainable-ownership manifest and schema-v2 checkpoint round-trip.

The shadow default should be described explicitly as a **training-only
auxiliary path** in configs, experiment JSON, checkpoint metadata, and
inference policy. A checkpoint must not silently choose between trained and
pretrained default adapters.

| Priority | Proposed arm | Single question | Expected chance / risk |
|---:|---|---|---|
| **1** | **E19: E13 + E18 balanced multi-reference package** | Does the strongest E18 component transfer to the winning shadow route? | Highest evidence-backed chance |
| **2** | **E20: E13 + branch-local output r32** | Does safe branch-specific output capacity add to shadow co-adaptation? | Moderate chance, low architectural risk |
| **3** | **E21: E19 + branch-local output r32** | Are multi-reference evidence and output capacity additive? | Highest upside; interpretable as a parallel 2x2 |
| **4** | **E22: E13 + verified ArcFace-like predicted-x0 loss** | Can direct recognition supervision break the objective ceiling safely? | High upside, highest implementation risk |
| **5** | **E23: E13 + LR decay beginning at 8k** | Does leaving the high-LR basin earlier improve/stabilize the peak? | Moderate chance, very low code risk |
| **6** | **E24: E13 + exact masked/full alternation** | Is temporal separation better than face-only or simultaneous protected loss? | Useful loss test; weaker prior from E14 |

These arms can run simultaneously. E13 supplies the `no multi-ref / no branch
output` corner, E19 and E20 supply the two single-component corners, and E21
supplies the combined corner. No sequential result is needed to interpret the
2x2.

### 1. E19: shadow co-adaptation plus balanced multi-reference conditioning

Suggested config name:
`E19_large_ds_joint_shadow_sa128_multiref_24k.yaml`.

Implementation:

1. Inherit from `E13_large_ds_joint_shadow_sa128_24k`, not E18/E15, so face-only
   loss and shadow validation remain the base.
2. Copy only E18's dataset/conditioning delta:
   `train_dataset_name=large_dataset_balanced_multiref`,
   `train_dataloader_shuffle=false`, `schedule_rows=48000`,
   `schedule_seed=130018`, `schedule_start_row=0`, and
   `num_identity_refs=3`.
3. Preserve the reference routing invariant: reference 0 is the **only**
   spatial latent/KV lane; up to three additional distinct same-ID references
   contribute PhotoMaker ID tokens only. Do not average or concatenate
   multiple spatial BA lanes in this arm.
4. Reuse E18's quality-ranked, pose-diverse selection and deterministic row
   schedule. Log identity frequency, number of usable refs, selected-reference
   filenames/quality tier, and ID-token norm so coverage can be audited.
5. Validate step-zero parity against E13 before training. The only intended
   learned-data delta is the schedule/reference package.

Why first: E18 improves E15 by `+.03735` and 65/96 wins despite the damaging
persisted path. If even part of that gain transfers, E19 is the most plausible
single arm to exceed `.400`.

### 2. E20: branch-local self-attention output rank32

Suggested config name:
`E20_large_ds_joint_shadow_sa128_branchout_r32_24k.yaml`.

Implementation:

1. Inherit E13 and set only
   `model.ba_hard_v1_branch_output_rank: 32`.
2. Keep the ordinary/shared self-attention output unchanged. The new output
   basis must affect only the hard reference-face branch, matching E2.
3. Preserve zero-init exact parity at step zero. Add a focused processor
   installation check showing that every intended hard-BA processor owns the
   branch output and no generic layer does.
4. Update the expected trainable contract. Based on E2, the delta is 140
   tensors / 5,324,800 parameters, giving a provisional E13 total of **2,380
   tensors / 224,542,720 parameters**. Treat Hydra composition and the runtime
   ownership audit as authoritative before launch.
5. Verify save/load parity with both shadow validation and full schema-v2
   checkpoint loading.

Why second: E2 was a small but visually stable positive on a much weaker base;
this is localized identity capacity without reintroducing generic U-Net drift.

### 3. E21: balanced multi-reference plus branch-local output

Suggested config name:
`E21_large_ds_joint_shadow_sa128_multiref_branchout_r32_24k.yaml`.

Implementation:

1. Compose exactly E19's dataset/reference overrides and E20's single model
   flag on E13.
2. Do not add protected loss, identity auxiliary loss, residual ID-CA, or a
   scheduler change. Those would destroy the parallel 2x2 interpretation.
3. Use exactly the same 48,000-row schedule and start offset as E19; use the
   exact expected trainable contract from E20.
4. Record both component names in the experiment JSON and declare E13, E19,
   and E20 as factorial comparators.

Why third: multi-reference evidence and branch-local output capacity address
different bottlenecks and may be additive. Running it in parallel avoids
waiting for E19/E20 while preserving attribution.

### 4. E22: verified ArcFace-like identity loss on predicted x0

Suggested config name:
`E22_large_ds_joint_shadow_sa128_arcfaceaux_24k.yaml`.

Do **not** implement this by turning up E16. E16 uses the PhotoMaker CLIP vision
tower, which is not the validation recognizer. The implementation should:

1. Refactor the existing `_predicted_x0_identity_auxiliary` behind a backend
   enum whose current/default option preserves E16 byte-for-byte. Add a new
   `arcface_torch_v2` backend in a small dedicated module.
2. Use a frozen differentiable PyTorch recognition network loaded from the
   same ArcFace/InsightFace recognition weights used by the validation stack,
   or a rigorously matched equivalent. First establish ONNX-versus-PyTorch
   embedding parity on already aligned 112x112 crops (cosine near 1 and
   matching pairwise similarities). Record the weight hash and preprocessing
   contract in the experiment manifest.
3. Never use the fixed validation identities' stored embeddings as training
   targets. Build a detached target embedding from the current training
   image and distinct same-ID references, preferably a normalized centroid
   after quality filtering.
4. Compute predicted x0 using the scheduler's declared prediction type
   (`epsilon`, `v_prediction`, or sample) rather than assuming epsilon. Decode
   through the frozen VAE while retaining gradients to the trainable paths.
5. Crop the **intended training face box** with differentiable ROIAlign and
   match the recognizer's alignment/preprocessing as closely as possible.
   Do not optimize a best face anywhere in the image; that would encourage
   duplicate/detached identity fragments.
6. Start after 4k, ramp through 6k, evaluate one eligible sample every two
   optimizer steps, and initially restrict to `t <= 300`, where decoded faces
   are meaningful. Before the full launch, use a short local gradient
   calibration to choose a maximum weight whose identity-gradient norm is
   roughly 5-10% of the diffusion-gradient norm on BA parameters.
7. Log auxiliary applied fraction, timestep, raw cosine/loss, weight, weighted
   contribution, predicted/target embedding norms, and gradient-norm ratio by
   optimizer role. Fail startup if the recognizer is trainable or the parity
   check/weight hash is absent.
8. Preserve the canonical `IDSimBest` evaluation unchanged. Add an intended-box
   recognition metric only as secondary telemetry so improved score cannot
   hide face duplication.

Why fourth: the current objective is the most plausible ceiling, so a truly
metric-aligned loss has high upside. It is below the proven components because
recognizer alignment, noisy predicted-x0 crops, and gradient scale are real
failure modes.

### 5. E23: earlier LR decay

Suggested config name:
`E23_large_ds_joint_shadow_sa128_earlydecay_24k.yaml`.

Implementation:

1. Inherit E13 and change only `lr_scheduler.hold_steps` from 14,000 to
   **8,000**. Keep warmup 20, total steps 24,000, and minimum factor 0.1.
2. Apply the same multiplier to BA, generic, and training-only default
   optimizer groups; do not change their base LRs.
3. Log exact per-role LRs and validate at every 2k as usual.
4. Keep raw weights as the canonical comparison. Do not add EMA to this arm;
   EMA plus scheduler would be two scientific changes.

Why fifth: E13/E14 are already strong at 8k and recover after the current
14k hold ends. Earlier decay is a cheap test of whether sustained `1e-4`
causes the mid-run oscillation.

### 6. E24: exact every-other-step masked/full loss

Suggested config name:
`E24_large_ds_joint_shadow_sa128_alternating_24k.yaml`.

The current code already supports the core behavior:
`PhotomakerLoraTrainer` sets `is_masked_loss` when
`batch_idx % masked_loss_step == 0`, and `MaskedDiffusionLoss` selects face or
full MSE. Therefore:

1. Inherit E13, keep `MaskedDiffusionLoss`, set
   `loss_kind: masked_alternating`, and set
   `trainer.masked_loss_step: 2`.
2. With `epoch_len=2000` (even), this produces a stable 50/50 face/full
   sequence across epoch boundaries. Do not alter batch sampling or alternate
   references at the same parity.
3. Add an audited wrapper or trainer telemetry that logs `loss_face`,
   `loss_full`, `is_masked_loss`, realized masked fraction, and per-mode
   gradient norms. Only the selected component contributes gradients on a
   given step. Keep historical loss classes/defaults unchanged.
4. Do not add E14's simultaneous `0.1` full plus `0.05` boundary terms. This
   arm asks whether temporal separation works better than E13 face-only and
   E14 simultaneous protected reconstruction.
5. Confirm in a short smoke run that alternating parity survives gradient
   accumulation and distributed batch indexing, and that checkpoint/resume
   resumes the global alternation rather than silently restarting it.

Why sixth: it directly tests the proposed loss hypothesis with little code,
but E14's lower mean makes additional full-image supervision less likely than
the first five changes to beat E13. It remains scientifically valuable because
alternation can yield different gradients from a weighted simultaneous sum.

## Implementation and decision gates for the next suite

For another agent implementing these arms:

1. Add six localized configs under `src/configs/`, all inheriting E13. Reuse
   E18/E2 settings through explicit overrides; do not refactor historical
   configs.
2. Extend the active one-GPU launcher with an explicit allow-list for the six
   configs and immutable experiment JSON records. Keep one run per GPU and
   the normal Serv allocation gate.
3. For every arm, compose Hydra config from `diffusion_template/`, run the
   trainable-ownership audit, verify `pose_adapt_ratio=0` and
   `ca_mixing_for_face=false`, and check step-zero validation parity where the
   architecture should be exact-parity.
4. For E20/E21, verify processor installation and schema-v2 checkpoint
   round-trip. For E22, verify frozen recognizer parity/hash and gradient flow.
   For E24, verify exact global alternating parity and telemetry.
5. During startup, require `saved/<run_name>/comet_experiment.json` with its
   immutable experiment key before accepting the job as live.
6. Keep the fixed-96 images and per-image CSV at every 2k. Add intended-box ID
   telemetry without replacing `IDSimBest`.

Rank a new arm above E13 only if all of the following hold:

- it exceeds `.39980` and remains at or above `.400` for at least two adjacent
  validation gates, rather than one noisy peak;
- at a common checkpoint (preferably 20k), its matched per-image mean delta
  versus E13 is positive with a paired panel interval that does not cross
  zero, or a later independent-seed replication confirms a smaller gain;
- identity and prompt tables do not hide a material tail regression,
  especially Eddie/Marion and skiing/jumping;
- full-panel visual review finds the identity on the intended body, with no
  increase in duplicate faces, face plates/seams, hand-eye fusion, or nested
  goggles;
- checkpoint reload produces the same processor routing and shadow-default
  inference policy.

After the parallel suite, combine only demonstrated positives. Before calling
the resulting setup a new general leader, replicate the winning configuration
with a new training seed; the current panel-bootstrap intervals do not replace
seed replication.

## Local evidence paths

- Run logs, metric histories, and downloaded Comet images:
  [`comet_data/e13_e18_20260806/`](../comet_data/e13_e18_20260806/)
- Per-run path/index details:
  [`2026-08-06_e13_e18_successful_run_index.md`](2026-08-06_e13_e18_successful_run_index.md)
- Reproducible derived analysis assets:
  [`analysis/assets/e13_e18_20260806/`](assets/e13_e18_20260806/)
- Prior E0-E12 analysis used as the methodological template:
  [`ANALYSIS.md`](../comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md)
