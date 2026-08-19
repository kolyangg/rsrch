# CL27 at 16k is the only clear CL23 successor; it improves identity and key occlusion cells but does not yet solve Skiing

**Date:** 17 August 2026  
**Evidence cutoff:** 06:42 UTC / 07:42 BST, 17 August 2026  
**Scope:** completed CL23, CL27, CL28, and CL29 runs; fixed 96-image validation;
matched per-image identity analysis; face/body alignment; visual review; code,
prior-report, and primary-paper inspection; design-only YAML handoff for eight
successors. No training code was changed and no job was launched.  
**Primary metric:** mask-owned subject-v2 `manual_val/id_sim`. It remains the
historical decision metric, but it is insufficient on its own for goggles,
ordinary glasses, hands, hair, face/body attachment, and duplicate faces.  
**Reproducible assets:**
[`assets/cl27_cl29_vs_cl23_20260817/`](assets/cl27_cl29_vs_cl23_20260817/)  
**Design blueprints:**
[`blueprints/2026-08-17_cl27_next_eight/`](blueprints/2026-08-17_cl27_next_eight/)  
**New source archive:**
[`sources/2026-08-17_cl27_cl29_followup/`](sources/2026-08-17_cl27_cl29_followup/)

| Arm | Immutable Comet key | Selected checkpoint | ID_SIM | Status |
|---|---|---:|---:|---|
| CL23 temporal-frequency | `a9ec9c59d1624c68acb98737dcd65298` | 24k | `0.539085` | complete |
| **CL27 frequency-surface** | `dbfbf40c3bdd4f70bedc58bda3dfb9cd` | **16k** | **`0.547260`** | complete |
| CL27 frequency-surface | same key | 24k | `0.543081` | complete |
| CL28 learnable schedules | `3d8aca3b4cbb4ddc9338f14952c5bd0e` | 24k | `0.539631` | complete |
| CL29 low-band contrastive | `2981820837564d01b1cefbf52c4dabd0` | 24k | `0.537603` | complete |
| PhotoMaker step 0 | `74efd227d3f8488a98e83d815c77c07c` | 0 | `0.556580` | controlled baseline |

## Executive conclusion

**CL27 at 16k is the only result worth promoting.** It reaches **`0.547260`
ID_SIM**, beating matched CL23 by **`+0.013436`** with `61/96` per-image wins
and a paired 95% bootstrap interval of **`[+0.005021,+0.023271]`**. The
step-zero-adjusted difference-in-differences is nearly identical at
`+0.013798`, interval `[+0.005205,+0.023691]`. No CL28 or CL29 checkpoint has a
statistically established aggregate advantage over CL23. `[measured][paired]`

CL27 is also the best hard-case base. At 16k it improves Skiing from
`0.374274` to **`0.433680`**, Crying from `0.559548` to **`0.585529`**, and
Marion from `0.448000` to **`0.493482`** versus matched CL23. Its 24k Jisoo
Skiing image is the cleanest of the four reviewed runs: both eyes remain below
the large goggle layer and ID rises to `0.489`, versus `0.441` for CL23,
`0.117` for CL28, and `0.428` for CL29. `[measured][visual]`

This is **not a general Skiing solution**. On the fixed 24k single-seed panel,
CL23 and CL27 both score `5 pass / 1 minor / 2 fail`; Lex retains nested eyewear
fragments and Marion's goggles still cover the eye region. Crying is much
healthier: all eight CL23, CL27, and CL28 rows pass the defined hand/face
topology rubric. `[visual]`

The apparent closeness of the aggregate scores hides meaningful regressions.
CL28 produces severe Jisoo goggle/face fusion and an opaque half-lens for Elon;
CL29 deletes Jensen's identity-defining ordinary glasses in Crying and curls a
goggle edge into Jisoo's eye. Conversely, CL29 genuinely helps some small/action
cells, especially Keanu Jumping (`0.414` versus CL23 `0.277` at 24k), and gives
the best aggregate face-quality mean. This makes CL29 useful mechanistic
evidence, not the next base. `[measured][visual]`

CL27 still trails PhotoMaker by `0.009320` at its peak. The next suite therefore
starts from CL27, removes CL29's harmful wrong-ID repulsion, teaches attention
ownership and contact topology more directly, tests a visibility-normalized
weighted loss, and includes two carefully constrained identity objectives.
Literal every-other-step masked/full loss is not repeated: E24 already tested
that exact route and was strongly negative. `[report]`

# 1. Evidence integrity and measurement protocol

## 1.1 Fixed validation contract

All reported checkpoints use the same 96-item `manual_val` panel, one image per
item, fixed prompt/reference/seed, fixed face ownership box, DDIM50, CFG 5, and
subject-v2 mask-owned identity metric. Every selected export contained exactly
`96` images and one `96`-row per-image table; export warnings and errors were
zero. The runtime face-box map was copied read-only from Serv and hashes to
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.
`[measured]`

The analysis normalized spaces and underscores before joining per-image table
keys, image files, and face boxes. This avoids the repository's silent partial
join. Paired intervals use `50,000` bootstrap resamples of the fixed 96 cells.
The visual rubric is deliberately separate from ID_SIM:

- **pass:** prompted object and any identity-defining ordinary eyewear are
  present, layer order is readable, the intended face remains attached;
- **minor:** readable face and ordering with a localized asymmetry/artifact;
- **fail:** fused/duplicated layers, important object deletion, unreadable
  face, or wrong face/body association.

The visual counts come from one unblinded reviewer on one fixed seed. They are
useful for locating mechanisms, not a population estimate. A promotion panel
should repeat the hard prompts with four seeds and two blinded reviewers.
`[limitation]`

## 1.2 Step-zero comparability caveat

CL27, CL28, and CL29 have `96/96` byte-identical step-zero images. CL23 differs
from that common initialization in `96/96` files despite nominal exact
inheritance. The discrepancy is small: median pixel MAE `1.595/255`, mean MAE
`1.811/255`, mean RMSE `3.849/255`, and mean PSNR `36.42 dB`; CL23 step-zero ID
is `0.465001` versus `0.464640` for the successors. `[measured]`

This does not explain CL27's result: the difference-in-differences estimate
remains `+0.013798` with a positive interval, and CL27/CL28/CL29 share exact
initial pixels while only CL27 improves. It does mean that a strict
byte-identical CL23-to-successor claim is false. The exact source-revision or
numerical cause is not established, so all new release packages must enforce a
step-zero byte-parity gate against their implemented CL27 base. `[measured]
[not established]`

# 2. Quantitative result

## 2.1 Identity trajectories

![](assets/cl27_cl29_vs_cl23_20260817/id_sim_trajectories.png){ width=92% }

*Figure 1. Subject-v2 identity trajectories. The dashed line is controlled
PhotoMaker step zero.*

| Arm | ID at 16k | ID at 24k | Best checkpoint | Gap to PM0 |
|---|---:|---:|---:|---:|
| CL23 | `0.533824` | `0.539085` | `0.539085` at 24k | `-0.017495` |
| **CL27** | **`0.547260`** | `0.543081` | **`0.547260` at 16k** | **`-0.009320`** |
| CL28 | `0.536965` | `0.539631` | `0.539631` at 24k | `-0.016949` |
| CL29 | `0.531784` | `0.537603` | `0.537603` at 24k | `-0.018978` |

CL27 loses `0.004179` from 16k to 24k. Its later endpoint still beats CL23 by
`+0.003996`, but the paired interval `[-0.001812,+0.009922]` crosses zero. The
16k checkpoint is therefore the selected model and the later 8k should not be
treated as useful training by default. `[measured][paired]`

| Arm/checkpoint | Mean delta vs matched CL23 | Wins | Paired 95% interval | Decision |
|---|---:|---:|---:|---|
| **CL27 16k** | **`+0.013436`** | **`61/96`** | **`[+0.005021,+0.023271]`** | promote |
| CL27 24k | `+0.003996` | `56/96` | `[-0.001812,+0.009922]` | not established |
| CL28 16k | `+0.003141` | `56/96` | `[-0.002866,+0.008972]` | neutral |
| CL28 24k | `+0.000546` | `53/96` | `[-0.008748,+0.007909]` | neutral |
| CL29 16k | `-0.002040` | `51/96` | `[-0.012602,+0.009282]` | neutral |
| CL29 24k | `-0.001482` | `44/96` | `[-0.009115,+0.006203]` | neutral/negative |

## 2.2 Hard-case slices

| Slice | CL23 16k | CL27 16k | CL28 16k | CL29 16k | Best |
|---|---:|---:|---:|---:|---|
| Skiing | `0.374274` | **`0.433680`** | `0.353618` | `0.375369` | CL27 |
| Crying | `0.559548` | **`0.585529`** | `0.566040` | `0.558348` | CL27 |
| Jumping | `0.385140` | `0.394555` | `0.395457` | **`0.405386`** | CL29 |
| Dancing | `0.434389` | `0.442227` | **`0.457477`** | `0.439755` | CL28 |
| Marion | `0.448000` | **`0.493482`** | `0.471217` | `0.469001` | CL27 |
| Jisoo | `0.532052` | `0.562542` | `0.524577` | **`0.573155`** | CL29 |

CL27's gains align with the intended intervention: the strongest improvement is
the goggle/hair/hand family rather than only a global average. CL29's better
Jumping and some Jisoo rows show that low-band same-ID supervision contains a
useful small/action-face signal, but its aggregate and Skiing results show that
the wrong-ID term or its balance is harmful. `[measured][hypothesis]`

At 24k, CL27 still leads Skiing (`0.431679`) and Crying (`0.581205`). CL29 leads
Jumping (`0.418478`) and ties CL28 on Dancing (`0.454317`), but falls to
`0.375812` on Skiing. These crossed effects are why the successor suite isolates
positive attraction from negative repulsion rather than stacking CL27 and CL29
unchanged. `[measured]`

## 2.3 Quality and ownership diagnostics

| Arm 24k | Text | Mask IoU | Mean faces | TOPIQ-Face mean | TOPIQ-Face p10 |
|---|---:|---:|---:|---:|---:|
| CL23 | `26.2189` | `0.9237` | `1.156` | `0.7090` | `0.5840` |
| CL27 | `26.2134` | `0.9231` | **`1.104`** | `0.7115` | **`0.5971`** |
| CL28 | **`26.3156`** | `0.9239` | `1.135` | `0.7114` | `0.5883` |
| CL29 | `26.0863` | **`0.9275`** | `1.135` | **`0.7157`** | `0.5926` |

There is no broad quality collapse. CL27 has the best face-quality tail and the
lowest mean detected-face count; CL29 has the best mean face quality and mask
IoU. Those aggregates nevertheless miss CL29's deletion of Jensen's glasses and
CL28's malformed goggle layers. `[measured][visual]`

# 3. Visual comparison: the differentiating generations

## 3.1 Selected CL27 checkpoint at 16k

![](assets/cl27_cl29_vs_cl23_20260817/skiing_16k_face.jpg){ width=100% }

*Figure 2. The complete 16k Skiing face panel for the selected CL27 checkpoint
and all matched comparators.*

This panel is the strongest visual justification for selecting CL27-16k. CL27
beats matched CL23 on six of eight Skiing identities: Eddie `+0.019`, Elon
`+0.052`, Jennie `+0.049`, Jensen `+0.030`, Jisoo `+0.313`, and Keanu `+0.063`.
Jisoo changes from an unreadable goggle/nose composite at `0.180` to a readable
face with both eyes below the goggle layer at `0.493`; CL28 collapses the same
cell to `0.034`, while CL29 reaches `0.474` with a less clean lower rim.
`[measured][visual]`

The sheet also prevents overclaiming. CL27 lowers Lex from `0.359` to `0.316`
and Marion from `0.266` to `0.258`; Lex retains nested ordinary-eyewear
fragments and Marion's opaque goggles still replace the readable eye region.
The companion
[`16k Crying face sheet`](assets/cl27_cl29_vs_cl23_20260817/crying_16k_face.jpg)
keeps hand/face topology readable across all four runs, with CL27 improving six
of eight identities, tying one, and losing only Lex. `[measured][visual]`

## 3.2 Critical cells, part 1

![](assets/cl27_cl29_vs_cl23_20260817/critical_differentiators_24k_part1.jpg){ width=100% }

*Figure 3. Critical 24k differentiators. Each run shows the full generation
and an enlarged intended-face crop.*

| Cell | What differentiates the runs |
|---|---|
| Jisoo / Skiing | CL27 is cleanest (`0.489`): both eyes remain below the goggle layer. CL28's `0.117` corresponds to severe eye/nose erasure. CL29 reaches `0.428` but curls the lower goggle rim into the right eye. |
| Jensen / Crying | CL23/27/28 retain ordinary glasses and coherent hands. CL29 deletes the ordinary glasses; its lower `0.615` is consistent with the visual identity loss. |
| Keanu / Jumping | CL29 is a genuine success (`0.414` versus CL23 `0.277`): the small face remains attached and substantially more recognizable. |
| Eddie / Dancing | CL28 gets the highest ID (`0.515`) despite poor face/body alignment. CL29 improves detected-box IoU to `0.675` but turns the head and lowers ID to `0.397`. This is a direct example of one metric rewarding only one part of the desired result. |
| Marion / Night-ride | CL27 is visibly strongest and moves ID from CL23 `0.465` to `0.582`; it is the clearest Marion gain in the sheet. |
| Jennie / Crying | CL27 preserves coherent hands and gives the best ID (`0.617` versus CL23 `0.568`). |

## 3.3 Critical cells, part 2

![](assets/cl27_cl29_vs_cl23_20260817/critical_differentiators_24k_part2.jpg){ width=100% }

*Figure 4. Additional 24k differentiators, including misleading Skiing scores
and successor regressions.*

- **Keanu / Skiing:** CL27 is the best coherent result (`0.439`).
- **Elon / Skiing:** CL28 scores `0.487`, the highest automatic ID in the row,
  but one opaque orange lens remains while the normal eyewear layer is broken.
  This is the most important metric false positive in the comparison.
- **Eddie / Skiing:** CL29 enlarges/reinforces regular glasses and falls to
  `0.376`, versus CL23 `0.472`.
- **Jisoo / Night-ride:** CL29 is a genuine coherent gain (`0.636` versus CL23
  `0.563`), showing that the mechanism is not globally defective.
- **Jennie / Drumming:** CL27 is visually coherent but loses identity
  (`0.573` versus CL23 `0.643`), so CL27 is not uniformly better.
- **Keanu / Night-ride:** every successor is worse than CL23 (`0.586`); CL29 is
  lowest at `0.504`.

## 3.4 Fixed-seed hard-case rubric

Pass/minor/fail counts at 24k:

- CL23 - Skiing `5 / 1 / 2`; Crying `8 / 0 / 0`.
- **CL27 - Skiing `5 / 1 / 2`; Crying `8 / 0 / 0`.**
- CL28 - Skiing `3 / 1 / 4`; Crying `8 / 0 / 0`.
- CL29 - Skiing `5 / 0 / 3`; Crying `7 / 1 / 0`.

CL27 does not increase the number of 24k Skiing passes over CL23, but it repairs
the historically catastrophic Jisoo cell and has the best Skiing mean at both
16k and 24k. CL28 and CL29 trade isolated wins for more severe layer-order
failures. `[measured][visual]`

## 3.5 Face/body alignment

| Arm 24k | Center offset median | Center p90 | Offset > `0.25` | Size ratio median | Detected IoU median |
|---|---:|---:|---:|---:|---:|
| CL23 | `0.0208` | `0.0491` | `1` | `1.0003` | **`0.9382`** |
| CL27 | `0.0221` | `0.0490` | `1` | `1.0010` | `0.9362` |
| CL28 | **`0.0205`** | `0.0508` | `1` | `0.9978` | `0.9341` |
| CL29 | `0.0211` | **`0.0490`** | **`0`** | `1.0051` | `0.9324` |

All `96/96` intended boxes have a detected face and no row has IoU below `0.3`.
Aggregate alignment is effectively tied. Eddie/Dancing is the outlier:
CL23/27/28 have offsets around `0.38-0.40` and IoU around `0.44-0.45`; CL29
reduces the offset to `0.233` and raises IoU to `0.683`, but changes the head
pose and lowers identity. Face alignment is therefore a cell-specific visual
gate, not the explanation for the aggregate CL27 gain. `[measured][visual]`

# 4. Mechanistic interpretation

## 4.1 Why CL27 worked

CL27 leaves inference architecture and trainable ownership unchanged from
CL23. It adds a training-only objective on the already-routed temporal-frequency
message: suppress reference-message energy on synthetic top-object pixels while
enforcing a nonzero visible-face reference floor. `[code]`

The last logged training sample shows a non-null path: applied fraction `0.165`,
up0/up1 top high-band RMS `0.02155/0.03235`, visible routed/native ratios
`0.1262/0.1596`, and auxiliary loss `0.000138`. Across training phases the
auxiliary declines while the route remains active. Combined with the positive
16k paired result and the Jisoo repair, this is evidence that shaping existing
reference bandwidth is more useful than adding another dense ownership head.
`[measured][code]`

The later ID decline suggests the fixed strength eventually over-regularizes or
the ordinary diffusion objective drifts after the hard-case benefit has been
learned. It does not prove which term is responsible. CL27 should be checkpoint
selected at 16k while the next arms test better-localized or better-balanced
objectives. `[hypothesis]`

## 4.2 Why CL28 is not the base

CL28 adds three bounded endpoint corrections independently to each of `70`
processors. At 24k their raw absolute mean/max are only `0.02052/0.02312`, yet
Jisoo Skiing is repeatedly unstable and ends at `0.117`. Small parameters do
not imply a small functional change when every layer has its own correction.
The most plausible failure is incoherent layer-by-layer schedule specialization,
not an excessive global scale. `[measured][code][hypothesis]`

CL34 therefore retains the idea only as one shared three-scalar correction,
narrows each range from `+/-0.15` to `+/-0.05`, keeps low-early fixed, enforces
monotonicity, and uses a `10x` stronger zero anchor. It is a new test, not a
retry of CL28.

## 4.3 Why CL29 is useful but not promotable

CL29's positive same-ID cosine stays roughly flat near `0.12`, while wrong-ID
cosine falls and the correct-minus-wrong margin grows from near zero to about
`0.044`. At 24k the last values are positive `0.1170`, wrong `0.07268`, margin
`0.04432`. The objective has primarily learned to repel wrong identities rather
than draw two correct views together. `[measured]`

That behavior matches the result pattern: some small/action cells improve, but
global ID and Skiing do not; identity-defining eyewear can be removed. The
inference is that negative repulsion provides an easy loss shortcut and pushes
the low-band representation away from useful shared face/eyewear structure.
This is not causally proven, so CL30 removes only the negative term while keeping
the same detached target query, paired noise/timestep, low-band representation,
and CPU sampling. `[hypothesis]`

## 4.4 What is not the cause

- **Not PhotoMaker leakage unique to a successor.** CL23 and CL27-29 all use the
  same pretrained PhotoMaker identity tokens/default adapter and explicit BA
  route. No run loads a trained PhotoMaker experiment checkpoint, and none
  optimizes subject-v2. The absolute model is PhotoMaker+BA; the matched delta
  remains the scientific comparison. `[code][report]`
- **Not a validation detector collapse.** Every run has `96/96` intended-box
  detections and high median IoU. `[measured]`
- **Not broad text/quality failure.** Text and compact face-quality metrics stay
  close, although CL29 loses `0.127` text points versus CL27 at 24k. `[measured]`
- **Not literal training alternation.** CL27-29 inherit face-only masked MSE on
  every ordinary update; their differences come from the declared auxiliaries
  or schedule. `[code]`
- **Not established:** the exact source-level reason for CL23's small step-zero
  pixel drift; whether CL27's 16k peak replicates under another training seed;
  whether a four-seed visual ranking changes the fixed-seed ordering.

## 4.5 Confidence

| Claim | Confidence | Basis |
|---|---|---|
| CL27-16k improves aggregate ID over matched CL23 | High | `+0.013436`, 61/96 wins, fully positive paired and difference-in-differences intervals |
| CL27 is the best current hard-case base | High | Best Skiing/Crying means at 16k and cleanest Jisoo Skiing; no Crying regression |
| CL27 solves Skiing generally | Low / rejected | Same `5/1/2` fixed-seed rubric as CL23; Lex and Marion still fail |
| CL28's per-processor freedom is unsafe | Medium-high | Neutral aggregate, severe repeated Jisoo failure, 70 independent schedule vectors |
| CL29 improves mainly by wrong-ID repulsion | High for telemetry, medium for causal interpretation | Wrong cosine drops while positive cosine stays flat; result pattern is consistent but not an ablation |
| Aggregate face alignment explains score differences | Low / rejected | Medians and p90s are nearly tied; important differences are cell-specific |
| CL27 can beat PhotoMaker with one more well-chosen change | Medium | Only `0.009320` remains, but no existing arm has crossed the baseline and visual gates constrain reward strength |

# 5. Prior experiments and new paper transfer

## 5.1 Ideas that should not be repeated literally

| Earlier evidence | Result | Constraint on the new suite |
|---|---|---|
| E24 exact every-other-step face/full loss | Best about `0.37491` on its controlled Large-data substrate; visual anatomy remained poor | Do not spend an arm on literal alternation; test spatially normalized simultaneous weighting instead |
| E22 naive predicted-x0 ArcFace | About `0.42938` at 20k and `0.42663` at 24k with image degradation | No constant/global/high-noise identity pull |
| CL25 low-noise four-step ArcFace + frozen trajectory anchor | Warm start `0.506823`, fell to `0.484385`, ended `0.493873` | Direct ID remains low priority and must be hinge-gated, weaker, and BA-gradient-only |
| CL22 dense three-state visibility router | Brief early gain, then neutral with large text loss and artifacts | Do not replace CL27 routing with another learned all-token blend head |
| CL24 sparse PhotoMaker boundary distillation | Neutral/negative; Skiing remained broken | Supervise attention/contact ownership directly, not teacher epsilon differences |
| CL26 anchored high-resolution ROI | Matched-neutral in the reviewed gates | Do not add another persistent ROI residual; CL37 uses a training-only teacher and keeps CL27 inference |
| CL9 four-seed 18-step ROI suffix | Mean small-face ID about `+0.097`, `43/56` wins, exact outside ROI | Local bandwidth remains real causal evidence and justifies CL37's lower-priority distillation arm |

## 5.2 New primary-paper analysis

| Paper | Most relevant evidence | Transfer, not wholesale adoption |
|---|---|---|
| [CRAFT](https://arxiv.org/abs/2608.14403), 14 Aug 2026 | Identity reward alone improves modestly; attention localization plus identity reward gives a much larger gain and reduces split/misplaced routing | CL31 trains where the reference should be used; CL35 gates patch identity reward with that learned attention region |
| [DivRL](https://arxiv.org/abs/2606.23950) | A quadratic hinge treats identity as a feasibility constraint and avoids unbounded reward competition | CL36 rewards only below-margin faces instead of maximizing ArcFace everywhere |
| [Beyond Facial Consistency](https://arxiv.org/abs/2607.25622) | Bounded stage-aware branch scaling and region-aware objectives improve coherent person identity; highest face score is not always best overall | CL33 normalizes visible/top/contact regions; CL34 uses one shared bounded schedule; visual coherence remains a gate |
| [Intermediate Structural Prediction](https://arxiv.org/abs/2605.20807) | Explicit sparse structure protects high-frequency subject details better than undifferentiated injection | CL32 concentrates frequency-surface supervision on the contact ring where layer order is decided |
| [MaSC](https://arxiv.org/abs/2605.22469) | Masked reference-patch max-cosine correlates better with human concept preservation than global pooling on the same encoder | CL35 uses masked patch aggregation; it remains an auxiliary and does not replace subject-v2 evaluation |

The earlier archives already cover PuLID, DreamCache, DynamicID, UniPortrait,
InfiniteYou, SpatialID, Diff-PC, AnyPhoto, ReSem-Face, GroupPortrait, and
LatentIdentityTuning. The CL27 result narrows their useful transfer:

- from PuLID/identity-reward work, take accurate low-noise supervision and
  semantic protection, not another global ArcFace objective;
- from DynamicID/SpatialID, take query-specific spatial/time relevance, not a
  dense unconstrained ownership head;
- from AnyPhoto/InfiniteYou, take local identity isolation and bounded residual
  coordination, while keeping PhotoMaker plus BA as the base;
- from ReSem-Face, take explicit visible/occluded semantic separation, but do
  not import an inpainting cascade into full-body generation.

## 5.3 Dataset decision

Keep Cosmic as the base distribution for all eight arms. Broad BigCelebs and
the earlier curated variants did not beat CL14 overall; they are portrait-heavy,
shallower per identity, and weak on action/face-object interactions. CL27 also
shows that deterministic synthetic labels can help without changing the base
corpus. `[report][measured]`

Do not add BigCelebs to this suite. If CL31 or CL32 passes, the next data test
should mine a **Cosmic-only real occluder subset** with audited goggles, ordinary
glasses, hands, and hair, then use at most 20% auxiliary sampling followed by a
Cosmic-only re-anchor. That is deferred because it would otherwise confound the
mechanism tests.

# 6. Eight priority-ordered successor experiments

## 6.1 Summary

| Priority | Config | Base | Single critical change | Main target |
|---:|---|---|---|---|
| 1 | CL30 positive low-band same-ID | CL27 cold start | Remove CL29 wrong-ID repulsion; explicit positive-only low-band pull | ID and small/action faces |
| 2 | CL31 attention ownership | CL27 cold start | Synthetic-mask-supervised reference attention mass | Goggles, hands, hair |
| 3 | CL32 contact surface | CL27 cold start | Concentrate CL27 surface penalty at the object/face contact ring | Layer order |
| 4 | CL33 visibility-balanced reconstruction | CL27 cold start | Region-normalized weighted diffusion loss on occluded batches | Hard-case topology and attachment |
| 5 | CL34 shared frequency calibration | CL27 cold start | One shared bounded three-scalar schedule correction | Aggregate ID without CL28 instability |
| 6 | CL35 attention-gated patch identity | CL31 cold start | Sparse masked-patch DINO reward gated by CL31 attention | Visual identity plus object preservation |
| 7 | CL36 BA ArcFace hinge | CL27-16k, 4k continuation | Weak below-margin ArcFace reward with BA-only identity gradients | Close the PM ID gap |
| 8 | CL37 ROI-teacher distillation | CL27 cold start | Training-only ROI32 teacher, no inference residual | Small faces |

## Priority 1 - CL30: positive-only low-band same-ID attraction

**Hypothesis.** CL29 contains a useful same-ID low-band signal, but its InfoNCE
objective improves the margin by pushing wrong identities away. An explicit
`1-cos(z_ref1, stopgrad(z_ref2))` pull should retain Keanu/Jumping-style gains
without the Skiing and eyewear regressions. `[hypothesis]`

**Implementation.** Reuse CL29's distinct alternate reference, identical
target/noise/timestep, detached target Q, and branch-local low-band pooling at
`mid`, `up0`, and `up1`. Remove all wrong-reference logits from the training
loss. Sample 12.5% of eligible batches on CPU, ramp weight to `0.01` over 2k-6k,
and keep wrong/zero reference probes diagnostic only. Trainables remain
`2,240 / 219,217,920`.

**Prediction.** ID should exceed CL27 at two adjacent gates; Jumping should
retain CL29's gain while Skiing remains at least CL27-level.

**Risk and gate.** Positive consistency can be satisfied by a generic or weak
message. Kill if correct-versus-wrong branch sensitivity falls more than 10%,
if positive cosine rises while ID does not, or if Skiing falls by `>0.02`.

Blueprint:
[`01_CL30_positive_lowband_sameid.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/01_CL30_positive_lowband_sameid.blueprint.yaml)  
Serv YAML:
[`run_CL30...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL30_cosmic_positive_lowband_sameid_24k_full96_r1_1gpu.yaml)

## Priority 2 - CL31: attention ownership alignment

**Hypothesis.** CL27 constrains the energy of the final routed message but not
the source attention that produced it. A query-level "where to look" loss can
teach visible facial tokens to retrieve reference-face K/V while top-object and
contact tokens use the zero/native alternative. `[hypothesis]`

**Implementation.** On 25% of synthetic-occlusion batches and only at
`up_blocks.0/1`, compute reduced QK attention mass for face-query tokens. Do not
materialize full all-layer attention telemetry. Require visible-face reference
mass at least `0.55` and top/contact reference mass at most `0.10`; weight
`0.02`. The runtime route and trainables are unchanged. This is an attention
loss, not CL22's inference router.

**Prediction.** Jisoo remains repaired while Lex/Marion Skiing layer order
improves; Crying stays `8/8` pass.

**Risk and gate.** The model may send top-object queries to zero without
preserving the object. Require ordinary-object presence, text within `0.10` of
CL27, and nonzero visible reference mass. Kill if the loss falls only by
uniformly suppressing reference attention.

Blueprint:
[`02_CL31_attention_ownership_alignment.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/02_CL31_attention_ownership_alignment.blueprint.yaml)  
Serv YAML:
[`run_CL31...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL31_cosmic_attention_ownership_alignment_24k_full96_r1_1gpu.yaml)

## Priority 3 - CL32: contact-ring-partitioned frequency surface

**Hypothesis.** A goggle/hair/hand failure is decided at the narrow contact
boundary. CL27 spreads its top-object penalty across the full synthetic object;
concentrating the same normalized weight at a one-token contact ring should
improve layer order without erasing the object interior. `[hypothesis]`

**Implementation.** Erode the top mask by one latent token; define interior and
contact ring. Keep CL27's total top-object weight `0.02`, but use factors `0.5`
interior and `2.0` contact, normalized by active region and total expected
weight. Preserve the visible reference floor. No new modules or parameters.

**Prediction.** Fewer nested eyewear fragments than CL27 with neutral aggregate
ID; specifically Lex and Marion should improve without recreating CL28's
half-lens failure.

**Risk and gate.** Over-focusing on crude synthetic edges can produce halos.
Reject on new ringing, TOPIQ-Face p10 below `0.58`, or no visual gain at 8k.

Blueprint:
[`03_CL32_contact_frequency_surface.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/03_CL32_contact_frequency_surface.blueprint.yaml)  
Serv YAML:
[`run_CL32...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL32_cosmic_contact_frequency_surface_24k_full96_r1_1gpu.yaml)

## Priority 4 - CL33: visibility-normalized weighted reconstruction

**Hypothesis.** CL27's ordinary face MSE averages every in-box pixel together,
so a narrow hair strand, goggle rim, or hand contact can contribute little even
when semantically decisive. Separately normalizing visible-face and top-object
errors should give the object meaningful gradient without sacrificing face ID.
`[hypothesis][code]`

**Implementation.** Only on sampled occlusion batches use `0.75 L_visible +
0.25 L_top + 0.05 L_contact + 0.05 L_full`, with every region normalized by its
own pixel/channel count. Non-occluded batches retain exact face-only MSE. The
criterion must add CL27's `ba_aux_loss` exactly once. `train.py` must explicitly
map `loss_kind: visibility_balanced_ba`; testing only `_target_` is insufficient.

**Prediction.** Better object presence/contact and face attachment, with a
smaller ID upside than CL30 but lower reward-hacking risk.

**Risk and gate.** Synthetic primitives may dominate rare real structures.
Reject if non-occluded prompts lose ID, text drops `>0.10`, or the applied
fraction differs from the deterministic 25% contract.

Blueprint:
[`04_CL33_visibility_balanced_reconstruction.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/04_CL33_visibility_balanced_reconstruction.blueprint.yaml)  
Serv YAML:
[`run_CL33...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL33_cosmic_visibility_balanced_reconstruction_24k_full96_r1_1gpu.yaml)

## Priority 5 - CL34: shared narrow frequency calibration

**Hypothesis.** CL23's fixed schedule is good but unlikely exact; CL28 failed
because each processor could move independently. One coherent correction shared
by all layers may recover a small ID gain without layer-specific topology
conflict. `[hypothesis]`

**Implementation.** Register one three-value parameter on the model, shared by
all temporal-frequency processors. Keep low-early at `0.50`; bound low-late,
high-early, and high-late to `+/-0.05` around CL27, enforce monotonicity, and use
anchor weight `0.001`. The trainable contract becomes `2,241 tensors /
219,217,923 parameters`.

**Prediction.** A modest broad ID improvement with none of CL28's Jisoo/Elon
instability.

**Risk and gate.** Even a shared scale can over-strengthen reference features.
Kill if any correction hits its bound, Jisoo Skiing falls below `0.40`, or ID
does not beat CL27 by 12k.

Blueprint:
[`05_CL34_shared_frequency_calibration.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/05_CL34_shared_frequency_calibration.blueprint.yaml)  
Serv YAML:
[`run_CL34...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL34_cosmic_shared_frequency_calibration_24k_full96_r1_1gpu.yaml)

## Priority 6 - CL35: attention-gated masked-patch identity reward

**Base:** CL31, not bare CL27. The single addition versus CL31 is the reward.

**Hypothesis.** CRAFT's ablation and MaSC's same-backbone result indicate that
identity supervision is more useful when attention first localizes the owned
subject and when aggregation is patch-local rather than global. This should
reward facial detail without treating goggles/hands/background as identity.
`[literature][hypothesis]`

**Implementation.** Every 16th update, only for timestep `<=200`, decode at
most one predicted-x0 face. Extract frozen DINOv2 patch features and compute
reference-side masked max-cosine against three distinct same-ID references. Gate
the loss with CL31's attention-visible region and require gate mass `>=0.55`.
Weight `0.01`, ramp 2k-6k. The encoder is frozen and absent from the optimizer.

**Prediction.** Visual identity, Marion, and ID_SIM improve without CL29-style
eyewear deletion. The gain may be smaller in ArcFace than in patch similarity.

**Risk and gate.** Patch reward can copy texture/hair or add substantial cost.
Require intended-box subject-v2 gain, no prompt/layout loss, bounded cadence,
and no reference pose/background copying. Profile before launch.

Blueprint:
[`06_CL35_attention_gated_patch_identity.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/06_CL35_attention_gated_patch_identity.blueprint.yaml)  
Serv YAML:
[`run_CL35...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL35_cosmic_attention_gated_patch_identity_24k_full96_r1_1gpu.yaml)

## Priority 7 - CL36: BA-only ArcFace hinge continuation

**Hypothesis.** Direct biometric reward may close the remaining `0.009320`, but
only if it stops rewarding already-good faces and cannot steer generic/default
PhotoMaker adapters. This is materially different from E22 and CL25.
`[hypothesis]`

**Implementation.** Load the exact immutable CL27 r3 16k checkpoint and train
4k local steps. Every 16th update at timestep `<=200`, use the existing frozen
PyTorch ArcFace backend on the intended padded crop and a three-reference
same-ID centroid. Apply `relu(0.55-cos)^2`, maximum weight `0.01`; project this
auxiliary's gradient only to the `840 / 127,795,200` BA tensors and cap it at
2.5% of BA diffusion-gradient norm. Ordinary diffusion continues to update all
CL27 trainables. Pin checkpoint and ArcFace hashes in the experiment JSON.

**Prediction.** Low-ID tail cells rise while already-good/occluded faces receive
little or no biometric gradient; a successful arm can cross PM0.

**Risk and gate.** This remains the highest reward-hacking risk. The default
`0.55` margin must be frozen after a pre-run low-noise calibration, not tuned
after validation. Stop at the first gate if ID falls, expression/eyewear is
suppressed, duplicates rise, or the BA gradient cap is violated.

Blueprint:
[`07_CL36_ba_arcface_hinge_4k.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/07_CL36_ba_arcface_hinge_4k.blueprint.yaml)  
Serv YAML:
[`run_CL36...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL36_cosmic_ba_arcface_hinge_4k_full96_r1_1gpu.yaml)

## Priority 8 - CL37: small-face ROI-teacher distillation

**Hypothesis.** The proven CL9 ROI intervention supplies useful local bandwidth,
but CL26's persistent ROI residual did not transfer. A stop-gradient local
teacher can instead train the ordinary CL27 message and leave inference exact.
`[report][hypothesis]`

**Implementation.** On 12.5% of eligible late-denoising batches whose face side
is `<=256 px`, compute ROI32 target-Q/reference-KV messages at `up0/up1`, detach
them, and apply `0.02` SmoothL1+cosine distillation to the normal CL27 routed
message inside the face. No ROI module or parameter is saved. CPU-sample the
gate; preflight must prove nonzero eligibility.

**Prediction.** Jumping/Dancing improve without the persistent high-resolution
sticker risk of CL26.

**Risk and gate.** A trainable student can ignore the teacher or sharpen the
wrong composite. Kill if teacher/student cosine improves without small-face ID,
if face size grows artificially, or if outside-face output changes in a fixed
teacher smoke.

Blueprint:
[`08_CL37_smallface_roi_teacher_distill.blueprint.yaml`](blueprints/2026-08-17_cl27_next_eight/08_CL37_smallface_roi_teacher_distill.blueprint.yaml)  
Serv YAML:
[`run_CL37...yaml`](blueprints/2026-08-17_cl27_next_eight/serv/run_CL37_cosmic_smallface_roi_teacher_distill_24k_full96_r1_1gpu.yaml)

# 7. Implementation and Serv handoff

## 7.1 Code map

1. Add all fields as defaults-off arguments in
   `src/model/photomaker_branched/lora2.py` and propagate them through
   `branched_runtime.py`.
2. Implement CL30-CL32/CL34/CL37 processor-local collection in
   `attn_processor_cleanest.py`; aggregate each collector in
   `lora2_helpers.py` with one cached `unet.attn_processors` lookup.
3. Implement CL33 in a new concise loss module. Add the exact `loss_kind`
   mapping in `train.py`; include `ba_aux_loss` once.
4. Reuse `arcface_identity_aux.py` for CL36. Do not run InsightFace ONNX inside
   the training graph and do not reproduce best-face-anywhere selection.
5. CL35 may add a frozen DINO loader, but it must be lazy and enabled only for
   the declared arm. Record model revision/hash and peak memory.
6. Extend `cosmic_large_adapted.py` only if an additional returned mask or
   reference tuple is necessary. Disabled modes must avoid mask allocation and
   extra decoding.
7. Create `src/configs/CL30...CL37`, one immutable experiment JSON per run, a
   narrow validator, and a new active launcher. Do not widen the historical
   CL27-CL29 allowlist.

Before editing these subsystems, preserve/update existing `AICODE-NOTE` anchors
around processor-map caching, target-Q/reference-KV routing, low-band capture,
and alternate-base validation. Add one dated invariant comment only where a new
critical route or loss enters.

## 7.2 Required optimized pipeline

Every implemented config must contain or inherit:

```yaml
pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
trainer:
  epoch_len: 2000
  validation_interval_steps: 2000
  skip_initial_validation: false
  active_grad_norm_mode: requested_only
model:
  ba_hardcase_telemetry_enabled: false
```

Additionally:

- cache Diffusers processor maps once per collector;
- return immediately from disabled collectors;
- request active-gradient norms only for CL36's actual BA calibration;
- CPU-sample CL30's low-band gate and all other optional sampled gates;
- do not re-enable full-activation BA telemetry to obtain loss scalars;
- retain step-zero plus every-2k validation. A throughput smoke may omit step
  zero only as an explicitly named non-scientific qualification.

## 7.3 Fail-closed verification

For each arm, the implementation agent should require:

1. Hydra composition and `train.py` loss-kind resolution.
2. Old mode disabled output parity and new mode deterministic fixed-input
   forward/backward.
3. Exact trainable tensor/parameter inventory and optimizer role ownership.
4. Nonzero finite gradient for the declared mechanism, zero gradient to frozen
   encoders, and no duplicated auxiliary addition.
5. Processor installation parity between training and validation.
6. Checkpoint save/reload equality for any new parameter or EMA buffer.
7. 64-sample dataset preflight, deterministic occluder mask hash, dual-reference
   path inequality, and zero target/reference path leakage.
8. Step-zero 96-image byte parity against the exact packaged base for all
   cold-start arms; CL36 must instead equal the source CL27-16k validation at
   its continuation step zero.
9. One-batch and 100-step smoke with finite losses and no unexpected memory or
   throughput regression.
10. At startup, verify `saved/<run_name>/comet_experiment.json` contains the
    immutable 32-character key before reporting submission success.

The eight Serv YAMLs are syntactically valid but intentionally point to planned
packages. They are **not launchable** until every check above passes.

# 8. Evaluation and promotion ladder

## 8.1 Common gates

The standard panel remains primary. Evaluate at every 2k gate; do not select a
winner from training loss or one aggregate point.

**Research pass:**

- positive paired ID delta versus matched CL27 at two adjacent validation gates;
- at least one interval fully above zero or mean ID `>=0.5523`;
- Skiing at least `6 pass / at most 1 fail` in the blinded fixed panel and no
  regression from `8/8` Crying passes;
- text no more than `0.10` below matched CL27;
- TOPIQ-Face p10 `>=0.58`, `96/96` detection, no duplicate/ownership increase;
- alignment center p90 `<=0.06`, no more than one offset `>0.25`;
- correct-reference branch remains causally stronger than wrong/zero reference.

**Headline pass:** subject-v2 ID `>=0.556580` (controlled PhotoMaker step zero)
with all research-pass visual/quality gates. The PM threshold alone cannot
promote a run.

## 8.2 Checkpoint policy

Select per run by the earliest gate satisfying both ID and visual gates. The
CL27 result shows why final checkpoint is not automatically best. Cold-start
arms may stop after 16k if two consecutive later gates decline and hard-case
review does not improve. CL36 has only 0/2k/4k local gates and should stop at
the first regression.

Before broad training, run a no-training four-seed diagnostic on CL27-16k for
the 16 Skiing/Crying rows. It will estimate fixed-seed fragility and provides a
better visual baseline for the successors without changing the standard 96
contract.

# 9. Reproducing

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

# Validate immutable local records.
python tools/comet/comet_experiment.py show \
  --record comet_records/CL23_cosmic_temporal_frequency_router_24k_full96_r1.json
python tools/comet/comet_experiment.py show \
  --record comet_records/CL27_cosmic_frequency_surface_energy_24k_full96_r3.json
python tools/comet/comet_experiment.py show \
  --record comet_records/CL28_cosmic_learnable_frequency_schedule_24k_full96_r4.json
python tools/comet/comet_experiment.py show \
  --record comet_records/CL29_cosmic_lowband_causal_contrastive_24k_full96_r3.json

# Re-fetch each selected step into a distinct directory before rebuilding the
# visual assets. The builder expects comet_data/cl27_cl29_vs_cl23_20260817/.
python tools/comet/comet_experiment.py fetch \
  --record comet_records/CL27_cosmic_frequency_surface_energy_24k_full96_r3.json \
  --step-number 16000 \
  --output-dir comet_data/cl27_cl29_vs_cl23_20260817/CL27/step_016000

# Parse every design and Serv YAML.
python - <<'PY'
from pathlib import Path
import yaml
files = sorted(Path('analysis/blueprints/2026-08-17_cl27_next_eight').rglob('*.yaml'))
for path in files:
    assert isinstance(yaml.safe_load(path.read_text()), dict), path
print('parsed', len(files), 'YAML files')
PY

cd analysis/assets/cl27_cl29_vs_cl23_20260817
sha256sum -c SHA256SUMS.txt
cd ../../../analysis/sources/2026-08-17_cl27_cl29_followup
sha256sum -c SHA256SUMS.txt
```

The full transient Comet image/export cache was deleted after figures and
compact tables were verified because the local volume reached capacity. The
immutable keys, all 52 per-step ID tables, derived CSV/JSON files, figures,
builder/downloader scripts, and checksums remain local; the raw images are
reconstructible from Comet. `[reproduction note]`

# 10. References

## Project reports

1. [`2026-08-14_cl21_cl26_current_results_cl23_fairness_and_next_experiments.md`](2026-08-14_cl21_cl26_current_results_cl23_fairness_and_next_experiments.md)
2. [`2026-08-13_cl15_cl20_results_cl19_next_architecture.md`](2026-08-13_cl15_cl20_results_cl19_next_architecture.md)
3. [`2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md`](2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md)
4. [`2026-08-11_cl9_validation_interventions_results.md`](2026-08-11_cl9_validation_interventions_results.md)
5. [`2026-08-05_loss_objective_and_identity_supervision_advice.md`](2026-08-05_loss_objective_and_identity_supervision_advice.md)
6. [`2026-08-16_training_pipeline_processor_lookup_fix.md`](2026-08-16_training_pipeline_processor_lookup_fix.md)

## Primary papers

1. Park et al., [CRAFT: Constrained Reward via Attention Fine-Tuning for Subject Personalization without Composed Targets](https://arxiv.org/abs/2608.14403), 2026.
2. Wang et al., [DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation](https://arxiv.org/abs/2606.23950), 2026.
3. Li et al., [Beyond Facial Consistency: Personalized Person Image Generation with Holistic Identity Preservation](https://arxiv.org/abs/2607.25622), 2026.
4. Zhang et al., [Decomposing Subject-Driven Image Generation via Intermediate Structural Prediction](https://arxiv.org/abs/2605.20807), 2026.
5. [MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation](https://arxiv.org/abs/2605.22469), 2026.
6. Guo et al., [PuLID](https://arxiv.org/abs/2404.16022), NeurIPS 2024.
7. Hu et al., [DynamicID](https://arxiv.org/abs/2503.06505), ICCV 2025.
8. Jiang et al., [InfiniteYou](https://arxiv.org/abs/2503.16418), ICCV 2025.
9. Li, [SpatialID](https://arxiv.org/abs/2602.13994), 2026.
10. Yuan, [AnyPhoto](https://arxiv.org/abs/2603.14770), 2026.
11. Ding et al., [ReSem-Face](https://arxiv.org/abs/2608.04820), 2026.
12. Rizwan et al., [Diff-ID](https://arxiv.org/abs/2607.25078), 2026.

## Final recommendation

Use **CL27 r3 at 16k** as the sole base and implement CL30 first, followed by
CL31 and CL32. These three best match the observed evidence: keep CL27's useful
frequency-surface behavior, remove CL29's likely negative shortcut, and teach
source ownership/contact geometry without adding an inference-time dense
router. CL33/CL34 are controlled loss/schedule alternatives. Do not run CL35 or
CL36 until their reward gates and visual review pipeline are proven in smoke;
direct identity supervision remains a high-upside, high-risk option rather than
the default.
