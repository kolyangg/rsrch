# Expanded 24-case and architecture study progress

## Scope

- Canonical deterministic subset seed: `20260722`.
- Canonical manual-validation indices:
  `5, 6, 8, 10, 14, 17, 18, 22, 31, 35, 36, 47, 51, 52, 53, 64, 70, 72, 74, 77, 81, 84, 89, 94`.
- Continue writing only below `Jul_new_exp/22Jul_debug/`.
- Preserve the protected-source fingerprint guard and immutable experiment
  directories.

## Wave 1

- Run the two strongest existing repaired-N3a candidates on all 24 canonical
  cases at the 20-step discovery schedule.
- In parallel, screen native step-zero NN4, NN5a, NN5b, and NN6a on the original
  four cases to verify construction, activity, and safety before 24-case use.
- Build an incremental visual PDF from completed immutable experiment bundles.

## Alignment repair introduced during Wave 1

- Native NN4/NN5/NN6 use zero-initialized output connectors, so initial logs
  show exact-zero step-zero bypass. Their completed metrics will quantify this.
- Added an experiment-local landmark-alignment option. It detects the five
  reference landmarks, maps them by similarity transform to the normalized
  PhotoMaker target landmarks, and supplies the aligned face-only crop to the
  existing spatial K/V route.
- This is a two-pass step-zero architecture: ordinary PhotoMaker establishes
  target pose; BA uses reference identity evidence expressed in that pose.
- Planned four-case comparison: aligned confidence, dual, ROI core-ring, and
  face-only full-grid core-ring. Production code remains unchanged.

- Alignment smoke v1 on sample 0 remained anatomically coherent but had
  negative identity gain (`-0.04897`) and exposed reflected duplicate-face
  content at the warped boundary. Before judging the idea, v2 expands the
  source crop by 12 percent and uses replicated rather than reflected borders.

- Alignment v2 removed duplicated boundary content but still failed on the
  smoke case: identity gain `-0.01661`, face MAE `0.07745`, landmark movement
  `0.04726`, and outside MAE `0.01520`. Global image warping is rejected.
- Added a non-warping alternative for NN7 clean patch memory. Target queries
  stay at target coordinates; an inverse-distance blend of five landmark
  displacements chooses the center of a local reference K/V window. Planned
  diverse-identity screen compares 1x1/3x3/5x5 windows and effective gates
  `0.05/0.10/0.20`, each against an in-architecture BA0 baseline.

## User-requested full 96-case visual matrix

- Queued `matrix96_n3a_fullgrid_up_core_ring_anchor`, matching the original
  repaired full-grid core-ring recipe (`core_ratio=0.68`, up-only, CA off,
  protected output) on manual-validation indices `0..95` at 20 steps.
- The notebook contact sheet is capped at the first 24 rows for large runs to
  avoid an unreliable giant bitmap. A dedicated paginated PDF will include all
  96 target references, PM images, BA images, face crops, and metrics.

## Branched-zero causal baseline correction

- NN4, NN5a, NN5b, and NN6a produced identical four-case outputs and identical
  apparent metrics versus ordinary PM, while runtime diagnostics reported zero
  residual and exact-zero output-anchor bypass.
- This reproduces the documented difference between the ordinary PM execution
  path and the doubled branched path; it is not architecture activity.
- Added optional `compare_branched_zero` generation. It generates the identical
  architecture at runtime scale zero and logs face/scene/identity metrics as
  BA versus BA0. Historical later-family controls will be rerun with this
  causal baseline before any 24-case promotion.

## Wave 1 results

- `matrix24_confidence50_step5`: median face MAE `0.03664`, reference gain
  `-0.00168`, 10/24 positive, landmark movement `0.00479`, bbox IoU `0.97895`,
  outside MAE `0.01039`, face detection 24/24. Geometrically safe overall but
  not identity-causal across identities.
- `matrix24_dual25_step6`: median face MAE `0.06958`, reference gain `-0.01723`,
  10/24 positive, landmark movement `0.01013`, bbox IoU `0.96518`, outside MAE
  `0.01103`, face detection 24/24. Stronger change but worse identity direction.
- The original four-case Eddie-only screen was not representative. Confidence
  was positive for Jennie and Keanu but negative for Eddie, Elon, Jensen, Lex,
  and Marion identity medians; dual showed the same broad inconsistency.
- NN4, NN5a, NN5b, and NN6a are exactly inactive at native step zero when
  measured against BA0: face/outside MAE `0`, identity gain `0`, landmark
  movement `0`, bbox IoU `1`. Their apparent `+0.00603` versus ordinary PM is
  entirely doubled-path drift.

## Diverse NN7 correspondence wave — interim

- Eight identities: indices `5,14,31,36,51,64,72,84`.
- Native NN7v2 local5 gate `0.10`: BA0-relative face MAE `0.01252`, identity
  gain `+0.00009`, 4/8 positive.
- Landmark-IDW local3 gate `0.10`: face MAE `0.01281`, identity gain `+0.00066`,
  4/8 positive.
- Landmark-IDW local3 gate `0.05`: face MAE `0.00843`, identity gain `+0.00219`,
  6/8 positive, landmark movement `0.00081`, bbox IoU `0.99428`, outside MAE
  `0.00582`. This is below the visible-change threshold but is the first broad
  positive BA0-relative identity signal and motivates narrow authority tuning.
- Landmark-IDW local3 gate `0.20` is too strong/non-causal: identity gain
  `-0.01425`, only 2/8 positive.
- Window sweep at gate `0.10`: local3 is best (`+0.00066`); local1 is
  `-0.00087`; local5 is `-0.00193`. Narrow refinement will compare native
  local3 gate `0.05`, landmark gates `0.065/0.075/0.085`, and landmark
  displacement sigmas `0.12/0.35` against the current sigma `0.22`.

## Full96 execution note

- The first `matrix96_n3a_fullgrid_up_core_ring_anchor` attempt completed 96 PM
  baselines but hit CUDA OOM before its first BA image when four processes grew
  to about 20 GB each. The error bundle is retained.
- Retrying with three concurrent model processes and expandable CUDA allocation.
- Retry completed all 96 outputs. Aggregate result: face MAE `0.09124`, median
  reference gain `-0.15307`, only 10/96 positive, landmark movement `0.02872`,
  bbox IoU `0.93418`, outside MAE `0.01422`, face detection 96/96.
- The four-case identity metric was Eddie-specific. Eddie is the only
  positive identity median (`+0.01541`, 8/12); Elon, Jennie, Jensen, Jisoo,
  Keanu, Lex, and Marion have negative medians, and six of those identities
  have 0/12 positive cases. This records the untrained identity baseline, but
  is not a step-zero rejection criterion: the architecture is visibly active,
  changes identity-bearing facial content, detects coherent faces in 96/96
  cases, and remains broadly aligned (landmarks `0.02872`, bbox IoU `0.93418`).
  It remains the leading training candidate; training should be judged by
  whether identity direction improves from this baseline without losing its
  strong, bounded face-local effect.

## NN7 landmark-local refinement result

- The native bbox-relative local3 gate `0.05` control was negative against BA0
  (`-0.00213`, 3/8 positive), confirming that the gain comes from the
  landmark-conditioned correspondence rather than merely reducing authority.
- Landmark sigma `0.22`, gate `0.05` remains the consistency leader:
  `+0.00219`, 6/8 positive, face MAE `0.00843`, landmark movement `0.00081`.
- Gate `0.065` produced the strongest median gain (`+0.00695`) but only 4/8
  positive; gate `0.075` fell to `+0.00112` and 4/8, while gate `0.085` was
  nearly neutral (`+0.00018`, 4/8). This is a narrow optimum, not monotonic
  improvement from more reference authority.
- Sigma `0.12` was negative (`-0.00065`, 3/8); sigma `0.35` was weakly positive
  (`+0.00091`, 5/8). Sigma `0.22` is retained.
- Promote both gate `0.05` (consistency) and gate `0.065` (stronger median) to
  the canonical 24-case set, always evaluated causally versus BA0.
- Canonical 24 result, gate `0.05`: BA0-relative face MAE `0.01027`, matched
  reference gain `+0.00143`, 13/24 positive, landmark movement `0.00106`, bbox
  IoU `0.99517`, outside MAE `0.00476`, face detection 24/24.
- Canonical 24 result, gate `0.065`: face MAE `0.01145`, reference gain
  `+0.00337`, 14/24 positive, landmark movement `0.00111`, bbox IoU `0.99314`,
  outside MAE `0.00466`, face detection 24/24. This is the stronger modern
  candidate, though it remains just below the `0.012` visible threshold and
  below the desired 75 percent per-case identity consistency.
- Gate `0.065` identity medians are positive for Elon, Jennie, Jensen, Jisoo,
  Keanu, and Lex, but negative for Eddie and Marion. This is substantially
  broader than N3a core-ring, yet not a solved architecture.
- Add a cyclic wrong-reference control for the diverse eight at gate `0.065`.
  The harness now supports a per-sample branch-reference mapping and logs
  branch-reference similarity gain against the identical BA0 architecture,
  avoiding the ambiguity of a single fixed wrong identity for all targets.

## Progressive N3a alignment-repair wave

- In addition to the full 96-case core `0.68` matrix, compare a staged N3a
  safety ladder on the diverse eight identities: core ratios `0.50` and
  `0.35`, a core `0.50` late-start variant (step 8/20), and a core `0.50`
  target-erosion `0.20` variant.
- Every rung retains the later-family safety elements already shown to matter:
  up-only injection, cross-attention disabled, and exact base anchoring outside
  the target core. This isolates how spatial authority and timing affect the
  N3a alignment failure before promoting a rung to 24 cases.
- The core `0.50/0.35` diverse screen is an alignment/authority ablation: test
  whether geometry can tighten without losing the strong visible step-zero
  activity of the canonical core `0.68` training candidate.

## Recent-run idea mining — next modern wave

- The NN7 proposal and later audits consistently identify three useful pieces
  not yet combined in this search: semantic part ownership, low-authority up0
  plus stronger up1 specialization, and a tighter trusted output core.
- Added experiment-local semantic eligibility around the five target landmarks;
  queries outside the selected eye/nose/mouth neighborhoods fall back exactly
  to target attention. Compare radii `0.18` and `0.25` at gate `0.065`.
- Added per-site effective-gate initialization without production edits. Test
  up0/up1 gates `0.02/0.065` and `0.03/0.075` with landmark-local memory.
- Add gate `0.075` with target-core erosion `0.22`, plus gate `0.065` delayed to
  step 8/20. All use BA0 causal baselines, CA off, and exact PM outside-core
  anchoring.
- The NN5/NN6 reports support a clean identity fallback lane, but its current
  connector is exact-zero at native initialization; it is not a meaningful
  step-zero ablation without inventing a new warm start. Defer that component
  to training rather than misclassifying an inactive screen.
- Because visual review favors the canonical N3a core `0.68`, add a focused
  refinement around it rather than replacing it: intermediate core `0.60`,
  BA starts 7/20 and 8/20, and output erosions `0.15/0.20`. Screen on the
  diverse eight, then promote the best activity/alignment tradeoff to 24 cases.
- Add one-change-at-a-time newer-runtime controls to canonical N3a: paired CFG
  reference noise, zero reference token text, and zero reference pooled text.
  Only after the isolated arms, test paired noise plus zero token text. These
  retain the exact full-grid/core-ring/up-only/anchor topology.

## Recent-run idea wave — diverse-eight results

- Semantic eligibility radius `0.18` was extremely conservative: BA0-relative
  face MAE `0.00829`, reference gain `-0.00367` (3/8 positive), landmark shift
  `0.00113`, bbox IoU `0.99629`, and 8/8 faces detected. Radius `0.25` remained
  very safe but weak: face MAE `0.00924`, reference gain `+0.00037` (4/8),
  landmark shift `0.00108`, bbox IoU `0.99493`.
- Delaying landmark-local gate `0.065` to step 8 also suppressed useful branch
  activity: face MAE `0.00897`, reference gain `-0.00947` (1/8), landmark shift
  `0.00070`, bbox IoU `0.99390`.
- Gate `0.075` plus target-core erosion `0.22` is the best of these completed
  modern-local safety variants: face MAE `0.00965`, reference gain `+0.00176`
  (5/8), landmark shift `0.00076`, bbox IoU `0.99286`, outside MAE `0.00579`,
  and 8/8 faces detected. It is a safe secondary design but too weak to replace
  the visibly active canonical N3a training candidate.
- The staged `up0=0.02/up1=0.065` arm produced face MAE `0.01056`, reference
  gain `-0.00538` (2/8), landmark shift `0.00084`, bbox IoU `0.99250`.
  The stronger `up0=0.03/up1=0.075` arm stopped before metrics and is being
  rerun unchanged.
- Current focused execution: canonical N3a plus exactly one of paired CFG
  reference noise, neutral reference token text, or neutral pooled text; a
  combination arm is included only after the isolated controls. The experiment
  environment requires preloading the system `libstdc++` for the current
  InsightFace binary; failed pre-model launches wrote no result bundles.

## N3a plus one newer-run addition — diverse-eight results

- Matched canonical full96 subset for indices `5,6,8,10,14,17,18,22`:
  face MAE `0.08556`, landmark shift `0.02709`, bbox IoU `0.94432`, outside
  MAE `0.01107`, and 8/8 faces detected. This is the direct comparison anchor.
- Paired CFG reference noise: face MAE `0.08126`, landmark shift `0.02266`,
  bbox IoU `0.94610`, outside MAE `0.01112`, 8/8 faces. This slightly tightens
  geometry while retaining strong activity.
- Zero reference pooled-text conditioning: face MAE `0.08589`, landmark shift
  `0.02000`, bbox IoU `0.94656`, outside MAE `0.01099`, 8/8 faces. This is the
  best isolated alignment improvement with essentially unchanged face-change
  strength and is the leading one-addition refinement for a 24-case check.
- Zero reference token text: face MAE `0.08672`, landmark shift `0.02521`, bbox
  IoU `0.94177`, outside MAE `0.01142`, 8/8 faces. It does not improve the
  geometric balance.
- Paired noise plus zero token text: face MAE `0.08816`, landmark shift
  `0.02849`, bbox IoU `0.93732`, outside MAE `0.01114`, 8/8 faces. Combining
  the controls loses the isolated alignment benefit and is not promoted.
- Reference-similarity signs are logged but not used to reject these untrained
  initializations; the decision prioritizes visible branch activity, coherent
  faces, and alignment as requested.

## Stronger staged landmark-local result

- `up0=0.03/up1=0.075` is the first modern landmark-local arm to cross the
  visible-change threshold: BA0-relative face MAE `0.01597`, landmark shift
  `0.00148`, bbox IoU `0.99235`, outside MAE `0.00471`, and 8/8 faces. Its
  reference gain is `-0.00225` (4/8 positive), retained as a diagnostic only.
- Visual inspection shows coherent expressions, poses, head placement, and
  scene structure across the diverse eight. It is a much safer but much gentler
  training alternative than N3a, so it remains the secondary modern candidate.

## N3a local refinements and target-fallback hybrid — interim

- Core ratio `0.60` retains strong activity (face MAE `0.07468`) and improves
  alignment over canonical core `0.68` on the matched subset: landmark shift
  `0.01978`, bbox IoU `0.95126`, 8/8 faces. Zero pooled-text at core `0.68`
  remains stronger (`0.08589`) at nearly the same landmark shift (`0.02000`).
- Core `0.68` with output erosion `0.15`: face MAE `0.08680`, landmark shift
  `0.02593`, bbox IoU `0.94813`. Erosion `0.20`: face MAE `0.07910`, landmark
  shift `0.02233`, bbox IoU `0.94649`. Both are valid but dominated by the
  zero-pooled conditioning arm for the desired activity/alignment tradeoff.
- Delaying canonical N3a to step 7 lowers activity to face MAE `0.08048` but
  does not materially improve geometry (landmark `0.02536`, bbox `0.94752`).
- Full-grid N3a memory plus fixed target/reference dual attention is the most
  useful historical combination so far. Dual `0.25`: face MAE `0.05876`,
  landmark shift `0.00682`, bbox IoU `0.97817`, outside MAE `0.00995`, 8/8
  faces. Dual `0.35`: face MAE `0.08051`, landmark shift `0.01219`, bbox IoU
  `0.97379`, outside MAE `0.01076`, 8/8 faces.
- Visual inspection confirms that dual `0.25` removes most canonical N3a
  eye/mouth warping while visibly changing the face; dual `0.35` restores
  nearly canonical activity with substantially better alignment. Promote
  dual `0.25` to the canonical 24 immediately and promote dual `0.35` when a
  worker slot frees; use their 24-case visual consistency to select between
  the safer and stronger settings.

## Full-grid dual 24-case validation

- Matched canonical N3a core `0.68` on these 24 cases: face MAE `0.08588`,
  landmark shift `0.02411`, bbox IoU `0.94407`, outside MAE `0.01192`, and
  24/24 faces.
- Dual `0.25`: face MAE `0.05830`, landmark shift `0.00902`, bbox IoU
  `0.96873`, outside MAE `0.01093`, 24/24 faces. Relative to canonical, it
  reduces median landmark movement by about 63 percent while remaining clearly
  active. This is the safest strong training candidate.
- Dual `0.35`: face MAE `0.08121`, landmark shift `0.01223`, bbox IoU
  `0.96120`, outside MAE `0.01199`, 24/24 faces. It retains about 95 percent of
  canonical face-change magnitude while roughly halving landmark movement.
- Visual inspection across all eight identities shows coherent expressions,
  head placement, face/body attachment, and scene structure for both dual
  variants. Dual `0.35` is the leading balance for training; dual `0.25` is the
  conservative fallback.
- Created adjacent-column 24-case visual PDF:
  `visual_reports/20260723_n3a_canonical_vs_dual25_dual35_24.pdf`.

## Confidence-residual sweep correction

- The first confidence sweep accidentally placed `legacy_confidence_gain` in
  config overrides instead of runtime mutations; gains `0.25` and `0.50`
  therefore had identical hashes. Those bundles are retained but excluded.
- The corrected sweep records the gain in `experiment_spec.runtime_mutations`.
  Corrected gain `0.25` gives face MAE `0.02566`, landmark shift `0.00266`,
  bbox IoU `0.98463`, outside MAE `0.00878`, and 8/8 faces. It is safer and
  weaker than fixed dual `0.25`.
- Corrected gain `0.50`: face MAE `0.03524`, landmark shift `0.00380`, bbox IoU
  `0.97845`, outside MAE `0.00996`, 8/8 faces. Gain `0.75`: face MAE `0.04386`,
  landmark shift `0.00440`, bbox IoU `0.98003`, outside MAE `0.01022`, 8/8.
  These form a clean ultra-safe ladder but remain visually gentler than fixed
  dual `0.25/0.35`; they do not displace the leading candidates.

## One-addition 24-case follow-up

- Canonical core `0.68` plus zero reference pooled text completed 24/24 faces:
  face MAE `0.08389`, landmark shift `0.02147`, bbox IoU `0.94362`, outside MAE
  `0.01273`. The eight-case alignment improvement did not generalize strongly;
  it is dominated by dual `0.25/0.35`.
- Combining core `0.60` with zero pooled text also failed to compound the two
  isolated gains: face MAE `0.07428`, landmark shift `0.02318`, bbox IoU
  `0.95579`. Core `0.60` plus paired noise is similarly neutral (`0.07042`,
  landmark `0.01988`, bbox `0.95437`).
- Because dual `0.35` won the 8- and 24-case screens, start its full 96-case
  validation in parallel with isolated dual `0.35` refinements.

## Dual authority boundary and fine sweep

- Dual `0.50` did not hold its diverse-eight alignment advantage on 24 cases:
  face MAE `0.10353`, landmark shift `0.01929`, bbox IoU `0.94207`, outside MAE
  `0.01246`, 24/24 faces. It is more active than canonical but adds avoidable
  shape/eye drift and is rejected as the training default.
- Fine sweep on the diverse eight confirms a smooth authority curve:

| reference mix | face MAE | landmark | bbox IoU | outside |
|---:|---:|---:|---:|---:|
| 0.25 | 0.05876 | 0.00682 | 0.97817 | 0.00995 |
| 0.30 | 0.06836 | 0.00952 | 0.97317 | 0.01028 |
| 0.35 | 0.08051 | 0.01219 | 0.97379 | 0.01076 |
| 0.40 | 0.08417 | 0.01730 | 0.96004 | 0.01160 |
| 0.45 | 0.09716 | 0.01751 | 0.95509 | 0.01223 |
| 0.50 | 0.10386 | 0.01572 | 0.95603 | 0.01213 |

- The knee is between `0.35` and `0.40`: activity changes little from 0.35 to
  0.40 while landmark/bbox alignment worsens sharply. Retain `0.35` as the
  leading balance and `0.25` as the safety setting.
- Dual `0.35` plus zero pooled text is worse than plain dual `0.35`. Paired CFG
  noise has mixed small changes and no visual advantage. A step-7 delay simply
  lands between the existing `0.25/0.35` authority settings. None is promoted.

## Dual-0.35 full96 validation

- Completed 96/96 detected faces with face MAE `0.07723`, landmark shift
  `0.01134`, bbox IoU `0.95526`, outside MAE `0.01358`.
- Canonical core-ring N3a all96 is face `0.09124`, landmark `0.02872`, bbox
  `0.93418`, outside `0.01422`. Dual-0.35 therefore cuts landmark movement by
  about 61%, improves bbox alignment, and slightly reduces outside change while
  retaining a strong active face-local initialization.
- Matched-reference gain is near neutral (`-0.00270`, 46/96 positive) rather
  than canonical's strongly negative diagnostic, but identity remains a
  secondary step-zero measure.
- Generated `visual_reports/20260723_n3a_fullgrid_dual35_all96.pdf` for full
  visual inspection. The dual-0.25 comparison completed afterward.
- Per-identity geometry improves strongly for Eddie, Elon, Jennie, Jensen,
  Jisoo, Keanu, and Lex. Marion remains the main alignment weakness: median
  landmark shift `0.03357` and bbox IoU `0.90068` versus canonical `0.03176`
  and `0.93000`. The dual-0.25 results below show that lower authority repairs
  most of this identity-specific tail.

## Dual-0.25 full96 validation

- Completed 96/96 detected faces with face MAE `0.05787`, landmark shift
  `0.00732`, bbox IoU `0.96643`, outside MAE `0.01232`.
- This is safer than both canonical and dual-0.35 while remaining visibly
  active. The matched-reference diagnostic is slightly positive (`+0.00255`,
  53/96 positive), though it remains secondary at step zero.
- Marion landmark shift improves to `0.01472` from dual-0.35 `0.03357` and
  canonical `0.03176`; bbox IoU `0.90783` remains the main tail but improves
  over dual-0.35 `0.90068`.
- Generated `visual_reports/20260723_n3a_fullgrid_dual25_all96.pdf` (25 pages).
  Final training recommendation: dual-0.35 primary, dual-0.25 safety arm.
