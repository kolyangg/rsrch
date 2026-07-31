# Cosmic Large initial-usage policies through 20k

**Date:** 27 July 2026
**Status:** complete; five training trajectories and full-96 validation audited

## Decision summary

- None of the five policies is ready for promotion. Identity similarity peaks
  early and finishes lower in every arm, while action and occluded-face prompts
  still contain conspicuous stretched, pasted, duplicated, or detached facial
  features at 20k.
- If a longer-training probe is required, **top-three score-weighted is the
  clear first choice** and highest-score is an optional second. Give them a 50k
  budget, but gate at 22/24/28/32k and stop early unless identity and visual
  face coherence both improve. This is a bounded test of the late recovery,
  not evidence that uninterrupted training to 50k will help.
- Do not continue the self-reference baseline or the 256px self-reference arm.
  Their identity signal is contaminated by target/reference leakage. The
  256px arm has the cleanest IQA curves but loses `0.0820` identity similarity
  from its 4k peak to 20k and trains on only 16,168 accepted images.
- The highest-value dataset experiment is a clean factorial over
  distinct-reference selection and target-face scale, with no self-reference
  fallback. The highest-value architecture experiment is a bounded,
  layer/timestep-aware gate on the reference branch residual while retaining
  target queries and explicit reference K/V.

The complete pixel-level comparison is in the
[97-page full-96 PDF](assets/2026-07-27_cosmic_large_initial_usage_20k/cosmic_large_initial_usage_20k_full96_comparison.pdf).
It contains every one of the 96 sealed samples for all five runs at
0/4k/8k/12k/16k/20k, annotated per image with identity similarity, face
detection, TOPIQ-Face, TOPIQ, MUSIQ, and MANIQA.

## Question and controlled setup

The experiment asked whether the old `test`-branch use of Cosmic Large could
be improved by changing one dataset-policy field while keeping the current
eligible branched-attention model fixed. The baseline reproduces the old
target-as-reference behavior. Three arms replace that leakage with a
distinct reference chosen uniformly, by highest ArcFace score, or by a
top-three score-weighted draw. The fifth arm retains self-reference but removes
targets with faces smaller than 256px.

All arms use the same SA-only branched-attention model and training contract:

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

Validation uses the sealed 96-image set. Steps 0/1k/2k/3k/4k and then every
2k through 20k are present under one immutable validation key per arm. The
PDF selects 0/4k/8k/12k/16k/20k. All five training runs completed 20k, and
each validation key has exactly 96 images plus one identity/text metric and
seven compact face-quality metrics at all 13 checkpoints.

The four Serv arms have byte-identical step-0 validation images and metrics.
The Neb baseline's step-0 render differs slightly (`id_sim=0.3042` versus
`0.2999` for the Serv arms), so baseline interpretation should emphasize
within-run change and should not imply a perfectly paired five-way
initialization.

## Runs and immutable Comet links

| Arm | Dataset-policy change | Accepted training images | Validation | Training |
|---|---|---:|---|---|
| Old baseline | Self-reference; no minimum face size | 74,754 | [658d22341cf24accb5a3890869e76c28](https://www.comet.com/nikolay-2104/jul-comet-large-testing/658d22341cf24accb5a3890869e76c28) | [aa982105aad148bf9b2a30d3fc2149f1](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/aa982105aad148bf9b2a30d3fc2149f1) |
| Uniform | Distinct reference sampled uniformly; self fallback for 15,611 uncovered rows | 74,754 | [ced6658b5b12484a9e003fe47cd0c2bf](https://www.comet.com/nikolay-2104/jul-comet-large-testing/ced6658b5b12484a9e003fe47cd0c2bf) | [288ebfe3ccf74d5ea328a55b3abe31cb](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/288ebfe3ccf74d5ea328a55b3abe31cb) |
| Highest | Distinct reference with highest ArcFace score; same fallback | 74,754 | [ddaeb234353b45a1ae6763f5d8a1c81f](https://www.comet.com/nikolay-2104/jul-comet-large-testing/ddaeb234353b45a1ae6763f5d8a1c81f) | [fc3dec2223e84d49aa7c711fda968135](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/fc3dec2223e84d49aa7c711fda968135) |
| Top-three | Distinct top-three score-weighted reference, temperature `0.05`; same fallback | 74,754 | [b9751dc78c3b460c9b2ebc50d61b2036](https://www.comet.com/nikolay-2104/jul-comet-large-testing/b9751dc78c3b460c9b2ebc50d61b2036) | [b7821337e24e49f388450c103553a9da](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/b7821337e24e49f388450c103553a9da) |
| Face ≥256px | Self-reference; minimum target-face resolution 256px | 16,168 | [e44bd0b7434348fa868844e96d704fca](https://www.comet.com/nikolay-2104/jul-comet-large-testing/e44bd0b7434348fa868844e96d704fca) | [c6979abd46754e4ca43fae87df77eeff](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/c6979abd46754e4ca43fae87df77eeff) |

## Quantitative results

### Identity trajectory

| Arm | 0 | 4k | 8k | 12k | 16k | 20k | Peak over all 13 gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | .3042 | .3078 | .2802 | .2723 | .2801 | .2821 | .3078 at 4k |
| Uniform | .2999 | .2866 | .2513 | .2466 | .2421 | .2428 | .2999 at 0 |
| Highest | .2999 | .2945 | .2704 | .2445 | .2444 | .2646 | .2999 at 0 |
| Top-three | .2999 | .2886 | .2927 | .2833 | .2744 | .2703 | .2999 at 0 |
| Face ≥256px | .2999 | **.3467** | .2877 | .2785 | .2841 | .2647 | **.3467 at 4k** |

Every arm finishes below its best identity checkpoint. The distinct-reference
arms initially underperform the leaking baseline, but top-three preserves
identity best at 8–20k. Highest is the only arm with a meaningful late rebound:
`.2324` at 14k, `.2444` at 16k, `.2490` at 18k, and `.2646` at 20k. Top-three
also rebounds at 20k after `.2626` at 18k, but remains below its 8k value.

The self-reference 256px arm demonstrates that larger-face supervision can
produce a strong early identity gain, but not that the gain generalizes: it
falls from `.3467` at 4k to `.2647` at 20k. Its much smaller 16,168-image pool
and target/reference leakage make further continuation especially unattractive.

Text similarity moves in the opposite direction for the uniform arm: it rises
from `26.3205` to the experiment-best `27.1631` while identity falls to
`.2428`. Highest and top-three finish at `26.5549` and `26.6947`;
the baseline and 256px arm finish at `26.2324` and `26.1571`. Uniform therefore
improves prompt alignment at the expense of the experiment's primary identity
objective.

### Face quality at 20k

| Arm | Face detect | TOPIQ-Face mean | p10 | coverage | TOPIQ | MUSIQ | MANIQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | .979 | .639 | .526 | .854 | **.550** | 69.40 | .593 |
| Uniform | **1.000** | .633 | .512 | .865 | .533 | 69.50 | .599 |
| Highest | .990 | .642 | .528 | **.875** | .548 | **70.91** | **.609** |
| Top-three | .979 | .653 | .530 | .865 | .542 | 70.09 | .595 |
| Face ≥256px | .979 | **.673** | **.575** | .844 | **.565** | 69.82 | .590 |

The 256px arm has the best face-IQA mean and lower tail, but that does not
rescue its falling identity or the visibly malformed hard cases. Among
eligible distinct-reference arms, highest has the strongest broad 20k quality
profile (coverage, MUSIQ, MANIQA), while top-three has the best identity and
TOPIQ-Face mean.

Training loss is not a useful selection signal here. Approximate late
1,000-step means are `.0887` baseline, `.1216` uniform, `.1219` highest,
`.1220` top-three, and `.0893` face≥256px. Loss continues to decline while
validation identity declines.

## Visual assessment

The visual review uses matched samples rather than isolated attractive images.
The PDF is ordered by sample at each checkpoint, so each row compares the same
prompt/identity across all five arms.

At 20k, matched review of all 480 images ranks **top-three first, highest a
close second, and uniform last among the distinct-reference arms**. Easy
large/frontal faces can look coherent in every policy. For example,
`Chef_man_i_lex` has a well-attached, plausible foreground face in both
highest and top-three. This explains why face-crop IQA means can rise with
training.

The hard set tells a different story:

- `Jumping_ma_elon` remains catastrophically malformed across the policies:
  facial regions are stretched or pasted into the head, with missing or
  displaced eyes and duplicated mouth/nose structure.
- `Drumming_m_eddie`, `Jumping_ma_eddie`, and `Reading_pa_eddie` show a real
  benefit from distinct references: highest/top-three, and often uniform,
  attach a coherent face where baseline and face≥256px produce displaced eyes,
  mouths, or stretched face planes. The identity can still drift away from the
  Eddie reference, so this is an anatomy gain rather than a complete success.
- `Reading_pa_elon` and `Rushing_ma_elon` are cleanest under highest/top-three.
  Top-three is also the most consistent on `Chef_woman_jennie`,
  `Reading_pa_jennie`, and `Rushing_wo_jennie`; competing arms show blank
  mouths, pasted eye strips, or mask-like faces.
- `Skiing_wom_jisoo` contains repeated internal facial fragments and
  mask-like boundaries despite a sharp, attractive ski scene.
- `Reading_pa_marion` often has duplicated eye/face regions under hair and a
  collapsed lower face. TOPIQ-Face fails to produce a score for this sample in
  every arm at 20k, which is more informative than the surviving-image mean.
- Shared unsolved failures include `Dancing_ma_elon`,
  `Dancing_ma_keanu`, `Jumping_ma_keanu`, `Jumping_ma_lex`,
  `Dancing_wo_jennie`, and `Skiing_man_keanu`. Crying prompts often merge
  fingers into eyes or facial planes.
- Jumping and dancing are consistently the weakest prompt groups. At 20k,
  TOPIQ-Face coverage across all runs is only about `.60` for dancing men,
  `.67` for dancing women, `.60` for jumping men, and `.53` for jumping women.

The failure is therefore not simply blur or low fidelity. It is spatial
coherence: the reference-conditioned facial content is sharp but incorrectly
attached, repeated, or warped inside the target pose. Longer training improves
face-crop sharpness and naturalness more reliably than it fixes this
routing/alignment error.

### Selected matched panels

Distinct references repair Eddie's attachment in the jumping example, but the
next row shows that every policy still fails on jumping Elon. Each column is
one run and each label is per image.

![20k drumming and jumping comparison](assets/2026-07-27_cosmic_large_initial_usage_20k/step_020000_page_087.png)

The reading panel makes the ranking clearer: highest/top-three repair Elon,
and top-three is the only consistently clean Jennie result. The strong
Jensen/Keanu rows also show why averages can look healthy while identity-
specific failures remain.

![20k reading comparison](assets/2026-07-27_cosmic_large_initial_usage_20k/step_020000_page_093.png)

The ski panel is the clearest metric counterexample. `Skiing_wom_jisoo` is
visibly malformed in every column despite high face-crop IQA and successful
InsightFace detection.

![20k skiing comparison](assets/2026-07-27_cosmic_large_initial_usage_20k/step_020000_page_096.png)

### Which metric best matches perceived coherence?

No current scalar is sufficient. Across the 65 run/checkpoint aggregate
points, descriptive Pearson correlations with identity similarity are:
TOPIQ-Face coverage `.61`, MANIQA `.57`, TOPIQ-Face p10 `.34`, TOPIQ-Face
mean `.28`, and face-detection rate `-.09`. These correlations do not prove
causality, but they agree with the visual audit in two useful ways:

1. **TOPIQ-Face coverage is the safest first coherence guard.** It penalizes
   faces rejected by the stricter internal alignment stage instead of hiding
   them from the quality mean.
2. **TOPIQ-Face p10 is more useful than its mean** for catching a malformed
   tail. The mean rewards crisp surviving faces and can look good while several
   action faces are unusable.

Face-detection rate is almost saturated at `.97–1.00` and is too permissive.
For example, all five detectors accept `Skiing_wom_jisoo`; the baseline image
even receives TOPIQ-Face `.714`, crop TOPIQ `.737`, and crop MUSIQ `78.2`
despite grossly duplicated goggles/eyes/mouth geometry. All four IQA models
were evaluated on the same 25%-padded, square 512×512 crop around the largest
InsightFace bbox; none used the whole image. TOPIQ, MUSIQ, and MANIQA are
generic IQA models applied to that crop, so they emphasize local sharpness and
naturalness rather than facial anatomy or attachment. They can score a crisp,
geometrically wrong face highly. The per-image `multi_face` flag, although
intentionally not one of the seven headline Comet curves, is useful through
the API as an artifact diagnostic; its aggregate rate is negatively associated
with identity (`r≈-.39`).

For decisions, use: coverage as a fail gate, p10 as the malformed-tail
indicator, per-image multi-face/failure inspection, then identity and text.
Do not rank arms by TOPIQ-Face mean alone.

## What worked and what did not

### Worked

- Filtering to target faces ≥256px yields the clearest short-run result:
  identity `.3467` at 4k plus substantially stronger face-IQA means. This is
  strong evidence for a scale/quality curriculum, but not for self-reference.
- Top-three score-weighted distinct references preserve identity better than
  uniform or highest selection over most of 8–20k, suggesting that some
  reference-view diversity is useful.
- Highest-score selection recovers on identity late and has the strongest
  broad distinct-reference quality metrics at 20k.
- The sealed full-96 protocol exposes failures that the original 12-image
  panels and aggregate means would miss.

### Did not work

- Uniform distinct selection trades away identity for text adherence and is
  not a continuation candidate.
- Highest-score selection alone does not prevent the mid-training identity
  collapse or the hard-pose facial artifacts.
- The 256px self-reference arm does not sustain its 4k identity gain. A small,
  repeatedly sampled subset plus leakage is not an acceptable final data
  strategy.
- More optimization does not monotonically improve identity or attachment.
  All endpoints are below their peak; hard-pose artifacts remain at 20k.
- Mean face IQA and training loss are both capable of improving while the
  actual research target worsens.

## Recommended next experiments

### A. Bounded longer-training probe

Continue **highest-score distinct** and **top-three score-weighted distinct**
from their exact 20k checkpoints, preserving their existing training and
validation Comet IDs if continuation semantics require it.

Use the same sealed 96 images at 22k, 24k, 28k, 32k, 40k, and 50k. Treat 50k
as a maximum budget, not a requirement. Stop if two consecutive gates fail
all of:

- identity exceeds the arm's 20k value;
- TOPIQ-Face coverage and p10 do not regress;
- a fixed hard subset (`Jumping_*`, `Dancing_*`, `Skiing_wom_jisoo`,
  `Reading_pa_marion`) shows fewer attachment/duplication failures.

Top-three is the first choice because it has the best 20k identity and
TOPIQ-Face mean among distinct-reference arms and the most consistent visual
coherence across identities. Highest is the lower-confidence second because
of its unique 14–20k identity recovery, strong late broad quality, and good
Elon/Eddie images, but it has no clear overall visual win over top-three.
Neither currently provides evidence for an automatic full 50k run. If only one
continuation is affordable, choose top-three.

### B. Improve Cosmic Large usage

Run a clean `2 × 2` factorial with one controlled target population:

1. reference selection: highest-score versus top-three score-weighted;
2. target scale: fixed face≥256 versus a scale-balanced curriculum.

Remove self-reference fallback. If a row lacks a valid distinct candidate,
exclude it and apply the same accepted-target manifest to all arms. For the
curriculum, oversample faces≥256 for the first 4–6k steps, then introduce
192–255px faces in balanced bins; keep the loss mask/bbox face-weighted.
This tests the promising early scale result without reducing the entire run to
the 16,168-image subset.

Before launching, audit reference candidates jointly on ArcFace score, pose
difference, occlusion, blur, and native resolution. Highest ArcFace alone may
select a near-duplicate view that encourages literal spatial copying. The
longer-term data fix is stable multi-target identity grouping and native
full-scene references. Do not repeat numeric 40%/60% margin or 512px-upscale
arms on the current 256px face assets; they do not add source information.

### C. Concise branched-attention improvements

Keep target queries and explicit reference K/V, with
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and branched CA disabled:

1. **Reference-residual gate:** add a bounded learned gate per SA
   layer/resolution and timestep on the reference-branch residual merge.
   Initialize it to the current behavior and regularize against collapse to
   zero. This can suppress reference injection where it produces duplicated
   facial structure without substituting target K/V.
2. **Spatially aligned reference K/V:** remap reference-face coordinates into
   the target bbox frame before the branch mask/merge. The dominant visual
   failure is sharp identity content in the wrong target location; an explicit
   bbox-relative mapping addresses that while preserving BA.
3. **Fixed-checkpoint routing ablation:** before training, evaluate
   layer/resolution and denoising-timestep windows for branched SA on one
   checkpoint. Promote only a window that reduces hard-set duplication without
   weakening identity, then train it as a one-variable architecture job.

An optional low-weight identity loss on the predicted-x0 face crop can be
tested after the routing fixes, but it must remain auxiliary; the principal
identity mechanism must stay reference-conditioned branched attention.

## Evidence and reproducibility

- Observed metrics: immutable Comet API histories for the five validation and
  five training keys above.
- Observed images: exact Comet image assets, file-size/PIL verified, 96 samples
  at each of six selected checkpoints.
- Per-image annotations: identity recomputed from the downloaded pixels with
  the same InsightFace/ref-image contract; face quality joined by immutable
  Comet asset ID from the uploaded per-image CSVs. Re-instantiating
  InsightFace offline produces small hardware/runtime differences from the
  historical aggregate Comet values (up to roughly `0.008` in a step mean);
  the tables above use the canonical logged Comet aggregates, while the PDF
  uses the per-image offline scores.
- PDF builder:
  `tools/comet/build_full96_longitudinal_pdf.py`.
- Download manifests and generated intermediates remain under the ignored
  `comet_data/cosmic5_report/` workspace and the Neb staging directory
  `/home/niko/rsrch/report_staging/2026-07-27_cosmic5/`.
- Prior setup/provenance:
  [baseline matrix](2026-07-26_cosmic_large_initial_usage_baseline_matrix.md)
  and [20k continuation report](2026-07-27_cosmic_large_initial_usage_20k_continuations.md).
