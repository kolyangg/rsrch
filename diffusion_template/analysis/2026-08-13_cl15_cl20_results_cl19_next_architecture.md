---
title: "CL15-CL20 results and the next CL19 architecture experiments"
subtitle: "Visibility order for face-overlapping objects, the remaining PhotoMaker identity gap, prior evidence, new literature, and six implementation-ready blueprints"
date: "13 August 2026"
---

# Executive decision

**CL19 is decisively the best CL15-CL20 training run, but it has not yet beaten
the controlled original PhotoMaker baseline.** `[measured]` At the common final
step 24,000, CL19 reaches **0.506823 subject-v2 ID similarity**. It is
`+0.050707` above CL14, wins `74/96` paired cells, and its cell-bootstrap 95%
interval is `[+0.03730,+0.06464]`. The next-best CL15-CL20 endpoint is CL16 at
`0.453810`; no other arm has a clear positive ID delta over CL14.

The remaining target is material. Controlled PhotoMaker reaches **0.556580**
on the same fixed panel. CL19 is `-0.049757`, wins only `18/96` paired cells,
and the paired interval is `[-0.06483,-0.03531]`. PhotoMaker also leads CL19
in TOPIQ-Face mean by `0.06454`, generic TOPIQ by `0.04384`, MUSIQ by `2.14`,
and MANIQA by `0.02374`. CL19 does lead PhotoMaker in text similarity by
`0.36914` and mask IoU by `0.04903`. `[measured]`

The hard-case result is more specific than "Crying and Skiing are both still
broken." `[measured: visual]` In a one-reviewer, unblinded topology review of
the eight fixed identities:

- controlled PhotoMaker had `8 pass / 0 minor / 0 fail` for both Skiing and
  Crying;
- CL19 Skiing had `4 pass / 1 minor / 3 fail`; the failures were fragmented or
  nested goggles/eyewear crossing facial features;
- CL19 Crying had `6 pass / 1 minor / 1 fail`; it is largely repaired, with the
  clearest residual failure being Eddie's finger/eyelid/cheek merge.

Thus the primary topology target is now **Skiing-style visible-order control**:
the model must keep goggles, glasses, hair, or hands as a coherent top layer
while reconstructing identity only in the facial surface that remains visible.
Removing the requested object is not a solution. CL17 demonstrates why this
guard matters: its Skiing ID mean is higher than CL19's, but several outputs
achieve that by deleting goggles and exposing more face. `[measured: visual]`

The six next experiments are independent CL19 ablations, in this priority
order:

| Priority | Proposed arm | One scientific delta | Primary goal | Confidence |
|---:|---|---|---|---|
| **1** | **CL21 residual identity CA v3** | Bounded target-Q -> active PhotoMaker-ID-token K/V residual at up0/up1 | Broad ID gain | Medium-high; code path already exists |
| **2** | **CL22 visibility-order router** | Three-state top-object / visible-face / background ownership inside CL19 lanes | Skiing/Crying topology | Medium; best causal fit, more implementation work |
| **3** | **CL23 temporal-frequency router** | Native early structure; reference-weighted late/high-frequency detail | Topology plus ID | Medium; deterministic and low-risk |
| **4** | **CL24 PhotoMaker boundary distillation** | Training-only teacher loss on sparse top-object/face boundaries | Hard-case topology | Medium; direct teacher evidence, extra compute |
| **5** | **CL25 low-noise identity reward** | Four-step pathwise ArcFace reward with frozen-CL19 trajectory anchoring | Close the PM ID gap | Medium-low; high upside, reward-hacking risk |
| **6** | **CL26 anchored high-resolution ROI BA** | CL15 ROI residual with a guaranteed `0.05-0.25` active scale | Small/local faces and ID | Medium-low; positive diagnostic, training transfer unproved |

An interim result obtained after the main CL15-CL20 export strengthens priority
1. `[measured: interim]` Corrected `CL14_CA_optimized_r11`, immutable key
`fafd7a61b06c4114b9dec2c21d29ca38`, has a complete 96-row panel at 10k and
scores `0.445238`, versus matched CL14 `0.423661`: `+0.021577`, `58/96` wins,
cell-bootstrap interval `[+0.00800,+0.03521]`. Text is `-0.05013` and mask IoU
is `-0.00996` versus CL14 at that step. This is not a final 24k result, its
face-quality aggregates were not yet present at export, and it does not
establish the effect on CL19; it is nevertheless the first positive controlled
signal from the corrected residual-CA mechanism.

The 11 August plan was wrong to rank CL15, CL16, CL17, and CL18 ahead of CL19.
`[measured] [checkpoint]` Their ideas were plausible, but their trained routes
were neutral, harmful, or effectively inactive; CL19's corrected main routing
equation was the decisive intervention. The useful parts of those ideas are
retained below only after fixing the observed activation or supervision flaw.

The exact design YAMLs are in the
[CL19 next-six blueprint folder](blueprints/2026-08-13_cl19_next_six/README.md).
They are deliberately marked `design_only_non_runnable`: another agent must
implement and validate one defaults-off path at a time before producing active
Hydra and Serv YAMLs. No training job was submitted as part of this report.

Generic BigCelebs mixing is **not** one of the six. `[measured]` CL20's curated
80/20 BigCelebs curriculum ends at `0.450064`, `-0.006052` versus CL14 and
`-0.056759` versus CL19. Historical BigCelebs arms tell the same story. Use
Cosmic as the base distribution; use hard-case images only as high-quality,
explicitly labeled supervision for a mechanism that has first shown causal
activity.

![Final-step metrics and controlled PhotoMaker context.](assets/cl15_cl20_20260813/endpoint_overview.png){ width=98% }

## Scope, evidence tags, and fixed contract

This report combines immutable Comet metrics and images, per-image paired
analysis, read-only final-checkpoint inspection, current code/config
inspection, the completed CL9 intervention study, relevant reports from both
`rsrch_apr_test/` and the previous `rsrch/` project, and a new primary-paper
search. It does not change weights, validation inputs, active configs, jobs, or
metric definitions.

- **`[measured]`**: immutable Comet asset or deterministic calculation on it.
- **`[measured: visual]`**: inspected fixed images/crops using the stated
  rubric; not a numeric metric.
- **`[checkpoint]`**: read-only inspection of a saved final checkpoint and its
  live parameterization.
- **`[measured: interim]`**: a complete per-image validation panel from an
  active run before its final scientific endpoint.
- **`[code]`**: current source or composed-config inspection.
- **`[report]`**: established in a linked prior project report.
- **`[literature]`**: a result or design from a primary paper; transfer to CL19
  remains unmeasured.
- **`[hypothesis]`**: proposed mechanism not trained here.
- **`[not established]`**: uncertainty that should not be converted into a
  conclusion.

The comparison contract is the fixed 96-image `manual_val` panel, one image
per item, seed 0, unchanged prompts, references and bboxes, RealVisXL
validation base, DDIM 50, CFG 5, PhotoMaker start step 10, and the existing
subject-v2 identity, text, mask, detection, and face-quality definitions. CL14
through CL20 are compared at the common 24,000-step endpoint and across all
2,000-step validation gates. PhotoMaker is an external step-0 control on the
same inference contract, not a matched training-step arm.

The **primary promotion metric is `manual_val/id_sim` subject-v2** at matched
steps. For the hard-case goal it is necessary but not sufficient: Skiing ID can
increase when a model deletes goggles and exposes more face. Therefore
prompted-object retention and the stated visibility-topology rubric are joint
decision gates, while text similarity, mask IoU, and the seven face-quality
curves are safeguards.

Every per-image join uses numeric `image_index` before consulting filenames.
This avoids the space/underscore and truncated-output-key traps documented by
the research-report skill. All eight exports contain exactly 96 images and one
table-sealing 96-row identity CSV at the requested step; the exporter reported
no warnings or errors.

# 1. Immutable experiments and measurement method

## 1.1 Run registry

| Label | Immutable Comet key | Requested/evaluated step | Scientific delta |
|---|---|---:|---|
| PM0 | `74efd227d3f8488a98e83d815c77c07c` | 0 / 0 | Original PhotoMaker V2; BA disabled |
| CL14 | `6fe0028be92242c38056b3d36665fdd6` | 24k / 24k | Thresholded feather/double-mask BA control |
| CL15 | `d57604dc77334e0f9874ddd049e85a17` | 24k / 24k | Shared high-resolution ROI residual |
| CL16 | `7b71eb3dbb3a492e8fa9bb0d87343c28` | 24k / 24k | Clean multi-scale reference memory |
| CL17 | `e05ce586c9364678a8370a774773341c` | 24k / 24k | Semantic visibility ownership |
| CL18 | `f6530436bf22472c9fb7731d1696c5ab` | 24k / 24k | Same-ID cross-view prediction consistency |
| CL19 | `cfeda7b55c174b3c83e8d40537ebb6dd` | 24k / 24k | True-soft, full-target-query, single-blend router |
| CL20 | `b05488e2cce94476acc92bcaa21d7362` | 24k / 24k | 80/20 Cosmic/curated BigCelebs then 4k Cosmic anchor |

The local export manifest is
[`tools/comet/comet_runs_13Aug_CL15_CL20_PM0_CL14.json`](../tools/comet/comet_runs_13Aug_CL15_CL20_PM0_CL14.json).
Downloaded Comet payloads are in the gitignored
`comet_data/13Aug_CL15_CL20_PM0_CL14_latest/` cache. Derived, hash-sealed tables
and figures are in
[`analysis/assets/cl15_cl20_20260813/`](assets/cl15_cl20_20260813/).

## 1.2 Uncertainty convention

For paired ID deltas, this report uses 100,000 bootstrap resamples of the 96
fixed cells with seed `20260813`. These intervals describe stability across
this panel, not generalization to new identities, prompts, reference sets, or
seeds. Identity- and prompt-cluster uncertainty would be wider; there are only
eight identities and twelve prompts. No multiple-comparison correction is
claimed.

The visual topology rubric has three labels:

- **pass**: prompted occluder remains present and its top/bottom relation to the
  face is coherent;
- **minor**: relation remains readable but has a local asymmetry/crowding;
- **fail**: fragmented, nested, duplicated, deleted, or intersecting surfaces
  make the layer order wrong.

The review was performed by one unblinded reviewer on one deterministic panel.
It is directional evidence, not an inter-rater benchmark. The row-level record
is [`visual_review.csv`](assets/cl15_cl20_20260813/visual_review.csv).

## 1.3 Interim CL14_CA context

The immutable keys are `6fe0028be92242c38056b3d36665fdd6` for CL14 and
`fafd7a61b06c4114b9dec2c21d29ca38` for the corrected CL14_CA candidate.

| Arm | Step | ID | Text | Mask IoU |
|---|---:|---:|---:|---:|
| CL14 | 10k | 0.423661 | 26.3071 | 0.89457 |
| CL14_CA optimized r11 | 10k | **0.445238** | 26.2570 | 0.88461 |
| Delta | matched | **+0.021577** | -0.0501 | -0.00996 |

`[measured: interim]` The candidate has exactly 96 images and a table-sealing
96-row identity CSV at 10k. Its full-96 face-quality aggregates were not yet
available, so it cannot pass a promotion gate. The older `CL14_CA_r7` panel is
excluded because it used the pre-correction Eddie validation path. The
reproducible summary is
[`cl14_ca_interim_10k.csv`](assets/cl15_cl20_20260813/cl14_ca_interim_10k.csv).

# 2. CL15-CL20 results: what worked and what did not

## 2.1 Endpoint metrics

| Run | ID @24k | Delta vs CL14 | Text | Mask IoU | ID peak (step) |
|---|---:|---:|---:|---:|---:|
| PM0 | **0.556580** | +0.100464 | 26.0015 | 0.86515 | 0.556580 (0) |
| CL14 | 0.456116 | - | 26.2769 | 0.89794 | 0.457096 (22k) |
| CL15 | 0.451067 | -0.005049 | 26.1379 | 0.89074 | 0.451067 (24k) |
| CL16 | 0.453810 | -0.002306 | 26.5216 | 0.88517 | 0.455661 (20k) |
| CL17 | **0.439219** | **-0.016898** | 25.9727 | 0.88086 | 0.440330 (22k) |
| CL18 | 0.451972 | -0.004144 | 26.2363 | 0.88290 | 0.454040 (20k) |
| **CL19** | **0.506823** | **+0.050707** | 26.3706 | **0.91419** | **0.507105 (22k)** |
| CL20 | 0.450064 | -0.006052 | **26.5933** | 0.89341 | 0.452692 (22k) |

`[measured]` All runs have `96/96` face-detection coverage and zero no-face or
unowned subject-v2 rows. CL19 is not exploiting detector failure. Its final
value is only `0.000282` below its 22k peak, so the result is not a transient
checkpoint-selection artifact.

| Run | TOPIQ-Face mean | TOPIQ-Face p10 | TOPIQ | MUSIQ | MANIQA |
|---|---:|---:|---:|---:|---:|
| PM0 | **0.75319** | **0.59185** | **0.61471** | **73.099** | **0.64373** |
| CL14 | 0.68638 | 0.57801 | 0.57754 | 70.817 | 0.61313 |
| CL15 | 0.68462 | 0.58586 | 0.58140 | 71.308 | 0.61535 |
| CL16 | 0.69298 | **0.59444** | 0.58850 | 71.567 | 0.61650 |
| CL17 | **0.66336** | **0.55984** | **0.55432** | 69.810 | **0.60430** |
| CL18 | 0.69100 | 0.57633 | 0.57200 | **69.727** | 0.60915 |
| CL19 | 0.68865 | 0.57472 | 0.57087 | 70.959 | **0.61999** |
| CL20 | 0.68838 | 0.57490 | 0.57055 | 70.266 | 0.61236 |

CL19's gain is therefore principally identity and spatial alignment, not a
face-quality gain over PhotoMaker. Against CL14 it is approximately neutral on
TOPIQ-Face (`+0.00227` mean, `-0.00329` p10), slightly lower on generic TOPIQ
(`-0.00667`), and higher on MANIQA (`+0.00686`). `[measured]`

## 2.2 Paired ID evidence

| Comparison @24k | Mean delta | Median delta | Wins | Cell-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| CL15 - CL14 | -0.005049 | -0.008331 | 38/96 | [-0.01270,+0.00240] |
| CL16 - CL14 | -0.002306 | -0.002491 | 47/96 | [-0.01472,+0.01011] |
| **CL17 - CL14** | **-0.016898** | -0.015019 | 38/96 | **[-0.03084,-0.00337]** |
| CL18 - CL14 | -0.004144 | -0.003690 | 44/96 | [-0.01583,+0.00700] |
| **CL19 - CL14** | **+0.050707** | **+0.044695** | **74/96** | **[+0.03730,+0.06464]** |
| CL20 - CL14 | -0.006052 | -0.012179 | 38/96 | [-0.02027,+0.00845] |
| **CL19 - PM0** | **-0.049757** | **-0.044843** | **18/96** | **[-0.06483,-0.03531]** |

![Paired full-panel ID differences.](assets/cl15_cl20_20260813/paired_id_forest.png){ width=88% }

## 2.3 Training trajectories

`[measured]` CL19 is distinct from the first validation event. Its step-0 ID is
`0.437661`, versus `0.316329` for CL15-CL18/CL20 and `0.301867` for CL14.
It then rises to `0.507105` at 22k. This is consistent with the soft-router
equation having a large immediate effect and training co-adapting around it.
Because the step-0 records belong to separate run initializations, this is not
a substitute for a fixed-weight processor swap.

CL15 and CL20 rise smoothly but converge to the CL14 band. CL16 and CL18 have
larger mid-run oscillations but no endpoint advantage. CL17 learns more slowly
and stays lower after 4k. None shows a hidden early checkpoint that changes the
main conclusion.

![Complete identity trajectories at the fixed 2k validation cadence.](assets/cl15_cl20_20260813/id_trajectories.png){ width=96% }

## 2.4 Per-identity result

| Identity | CL14 | CL19 | PM0 | CL19-CL14 | CL19-PM0 |
|---|---:|---:|---:|---:|---:|
| Eddie | 0.4124 | 0.4918 | 0.4934 | +0.0794 | **-0.0016** |
| Elon | 0.4955 | 0.5714 | 0.5877 | +0.0759 | -0.0163 |
| Jennie | 0.4689 | 0.5442 | 0.6408 | +0.0753 | **-0.0966** |
| Jensen | 0.5453 | 0.5517 | 0.5911 | +0.0064 | -0.0395 |
| Jisoo | 0.4604 | 0.4998 | 0.6286 | +0.0394 | **-0.1288** |
| Keanu | 0.5003 | 0.4911 | 0.5336 | **-0.0092** | -0.0425 |
| Lex | 0.4163 | 0.4525 | 0.4744 | +0.0362 | -0.0219 |
| Marion | 0.3500 | 0.4521 | 0.5029 | **+0.1021** | -0.0508 |

CL19 largely resolves the previous Marion-specific aggregate collapse: Marion
has the largest improvement over CL14. Eddie is essentially tied with
PhotoMaker and Elon is close. The remaining PhotoMaker gap is concentrated in
Jisoo and Jennie, then Marion. Keanu is the only identity that regresses against
CL14. `[measured]` This argues against a single global scale increase: it could
overcondition identities already near PhotoMaker while failing view/appearance
specific identities.

## 2.5 Per-prompt result

| Prompt | CL14 | CL19 | PM0 | CL19-CL14 | CL19-PM0 |
|---|---:|---:|---:|---:|---:|
| Reading | 0.5213 | 0.5629 | 0.6099 | +0.0416 | -0.0470 |
| Rushing | 0.5167 | 0.5777 | 0.6234 | +0.0610 | -0.0457 |
| Skiing | 0.3548 | 0.3793 | 0.4640 | +0.0245 | **-0.0847** |
| Drumming | 0.5111 | 0.5546 | 0.6018 | +0.0434 | -0.0472 |
| Kickboxing | 0.4905 | 0.5142 | 0.5930 | +0.0237 | **-0.0789** |
| Dancing | 0.3574 | 0.4337 | 0.4487 | +0.0763 | **-0.0150** |
| Angry | 0.5119 | 0.5623 | 0.6210 | +0.0504 | -0.0587 |
| Crying | 0.4626 | 0.5562 | 0.6000 | **+0.0935** | -0.0438 |
| Laughing | 0.4468 | 0.4788 | 0.5493 | +0.0320 | -0.0705 |
| Jumping | 0.3324 | 0.3778 | 0.4173 | +0.0454 | -0.0395 |
| Night ride | 0.4758 | 0.5513 | 0.5851 | +0.0755 | -0.0338 |
| Chef | 0.4921 | 0.5331 | 0.5654 | +0.0410 | -0.0323 |

CL19 improves all twelve prompt means over CL14. Crying is the largest gain,
followed by Dancing and Night ride. Skiing remains the lowest topology-related
prompt and has CL19's largest gap to PhotoMaker; Kickboxing and Laughing are
the next-largest gaps. `[measured]`

# 3. Hard cases: what PhotoMaker gets right

## 3.1 Full-context comparison

The red rectangles below are the unchanged subject-v2 target boxes. The sheet
preserves the full generated context so an apparent face improvement cannot be
credited if the prompted occluder disappeared.

![Controlled PhotoMaker versus CL19 for all eight Skiing and Crying rows.](assets/cl15_cl20_20260813/photomaker_vs_cl19_hardcase_context.jpg){ width=92% }

`[measured: visual]` PhotoMaker's Skiing outputs consistently preserve two
distinct semantic surfaces: large ski goggles sit above/over the upper face;
when ordinary glasses are also present, they remain smaller and structurally
separate. CL19 often succeeds, but three cases have a different failure mode:

- **Elon:** fragmented/nested eyewear competes with the face;
- **Jisoo:** a goggle boundary intersects facial features and creates an
  eye-like lens/facial hybrid;
- **Lex:** nested/fragmented eyewear crosses the eye region.

Eddie is readable but asymmetric. Jennie, Jensen, Keanu, and Marion preserve a
coherent top layer. Marion's lower face remains soft, but this is not the old
catastrophic identity collapse.

For Crying, CL19 preserves a readable hand/face boundary in six cases and a
minor but readable one for Jisoo. Eddie remains a clear merge: finger/hand
texture enters the lower eyelid and cheek. The qualitative task is therefore
not to add a global "occlusion strength." It is to decide **which surface owns
each overlap cell** and to stop identity reference detail at the top-object
boundary.

## 3.2 Why raw ID can reward the wrong fix

![All Skiing face crops across PM0 and CL14-CL20.](assets/cl15_cl20_20260813/skiing_all_runs_face_crops.jpg){ width=100% }

CL17 has Skiing ID `0.39058`, higher than CL19's `0.37929`, despite being much
worse overall. `[measured]` In several CL17 rows, goggles are missing or
reduced, exposing more recognizable face. `[measured: visual]` An ArcFace-like
metric naturally rewards that change. Therefore every future hard-case arm
must pair identity with an object-retention/topology review; otherwise the
experiment can "solve" Skiing by violating the prompt.

![All Crying face crops across PM0 and CL14-CL20.](assets/cl15_cl20_20260813/crying_all_runs_face_crops.jpg){ width=100% }

## 3.3 Architecture interpretation

CL19 computes full native and reference messages, then blends once:

```text
N = Attention(Q_target, K_target, V_target)
R = Attention(Q_target, K_reference_face, V_reference_face)
Y = (1 - r_geometry) * N + r_geometry * R
```

`[code]` Target Q remains full. Native target K/V remains full. The reference
lane uses reference-face-masked K/V; masked zero tokens remain softmax sinks.
The two-cell cosine router is approximately `0.25` outside, `0.75` at the
transition, and `1.0` in the face interior. This corrects CL14's thresholded
double mask and is the only CL15-CL20 intervention that changes the main route
at material scale without asking a new gate to grow from zero.

The limitation is equally explicit. `r_geometry` knows only distance from the
face mask. It cannot represent "goggles own this pixel, visible skin owns that
pixel," and it is unchanged across denoising time and spatial frequency.
Inside a goggle/hand/hair overlap, the native message carries the prompt's top
object while the reference message carries a clean reference face. A geometric
average can create a hybrid surface. `[hypothesis]`

PhotoMaker supplies a useful teacher because its native attention has no
spatial-reference face K/V competing at that boundary. This does not mean BA
should be removed: CL19's text and mask alignment are stronger, and explicit
reference-conditioned BA is the project's invariant. It means the native path
should retain ownership of the top object while BA supplies identity only to
the visible face and late detail.

## 3.4 What is not the cause

- **Not face-detection failure.** `[measured]` CL19 detects an owned face in
  `96/96` rows; no-face and unowned counts are zero.
- **Not a broad bbox-placement failure.** `[measured]` CL19 has the best mean
  mask IoU of the compared trained arms (`0.91419`) and still has the visual
  layer-order errors.
- **Not principally the old Marion collapse.** `[measured]` Marion improves by
  `+0.10211` over CL14. Jisoo and Jennie now account for larger average gaps to
  PhotoMaker.
- **Not solved by more reference amplitude or rank alone.** `[report]` Earlier
  scale/rank and memory experiments plateaued or harmed output, while CL16's
  added memory was almost unused.
- **Not solved by a hard occluder polygon.** `[report]` The CL9 oracle-style
  polygons improved ID but still duplicated or relocated goggles because a
  binary region does not define top/bottom ownership.
- **Not a generic data-volume problem established by BigCelebs.** `[measured]
  [report]` CL20 and historical BigCelebs arms improve neither identity nor the
  hard-case architecture.

# 4. Why CL15, CL16, CL17, CL18, and CL20 did not improve CL14

## 4.1 Final-checkpoint path audit

| Arm | Added trainables | Effective routed scale | Auxiliary evidence | Diagnosis |
|---|---:|---:|---|---|
| CL15 ROI | 36 gate tensors; total `2,276 / 219,217,956` | mean `0.00218`, max `0.00308` | - | Near-null residual |
| CL16 clean memory | 322 added tensors; total `2,562 / 240,353,326` | mean `0.00167`, max `0.00320` | - | Near-null residual |
| CL17 ownership | 180 added tensors; total `2,420 / 224,643,144` | mean `0.00452`, max `0.00848` | BCE `0.693->0.0255`; head RMS `0.01264` | Labels learned; denoiser barely changed |
| CL18 cross-view | no inference trainables; `2,240 / 219,217,920` | none | final consistency loss `0.000142` | Tiny training-only objective |
| CL19 router | no added trainables; `2,240 / 219,217,920` | material fixed blend | n/a | Main route is causally active |

`[checkpoint]` CL15's raw gates average `0.00870`, but the actual formula
`0.25*tanh(raw)` reduces them to approximately `0.00218`. CL16 behaves the same
way with a `0.20` maximum. CL17 successfully classifies its synthetic masks,
but the learned ownership contribution stays below `0.85%` of native scale.
The detailed record is
[`checkpoint_path_audit.csv`](assets/cl15_cl20_20260813/checkpoint_path_audit.csv).

This changes the interpretation:

- **CL15 does not reject a high-resolution ROI mechanism.** It rejects a
  zero-start path that training can ignore.
- **CL16 does not reject clean reference memory.** It shows that adding
  capacity/aligned memory is insufficient when its residual is effectively
  absent. Earlier N37 evidence nevertheless makes this lower priority than an
  explicit ID route.
- **CL17 rejects its complete implementation as a useful intervention**, and
  its binary label is semantically insufficient. It does not prove that
  visibility order is useless; it proves that a label head can win its own loss
  without controlling generation.
- **CL18 provides no useful cross-view gain.** Its objective becoming tiny is
  consistent with encouraging reference-indifferent predictions, the opposite
  of the desired causal spatial reference path.

## 4.2 Per-arm decision

### CL15 - shared high-resolution ROI BA

`[measured]` Endpoint ID is `0.451067`, with a paired interval spanning zero.
Face-quality p10 is slightly higher than CL14, but identity, text, and mask IoU
are lower. `[checkpoint]` The effective gate is approximately `0.2%`, so this
run is best treated as a **failed activation design**, not a strong negative on
ROI detail. The CL9 fixed-checkpoint ROI suffix remains positive evidence; it
is revisited only after replacing the gate with an anchored floor in CL26.

### CL16 - clean multi-scale reference memory

`[measured]` Endpoint ID is `0.453810`; its 20k peak `0.455661` does not beat
CL14's peak. It has the best non-PM TOPIQ-Face p10, but lower mask IoU.
`[checkpoint]` Its residual scale is also near zero. DreamCache/ReF-LDM support
clean aligned K/V in other tasks, but previous project N37 found abundant
reference memory without identity causality. This mechanism is not promoted
again until a direct route or objective proves it needs more memory.

### CL17 - semantic visibility ownership

`[measured]` CL17 is the only statistically clear regression: `-0.016898` ID,
lower text, lower IoU, and the worst face-quality metrics. The ownership BCE
falls sharply, yet its actual denoiser contribution remains sub-percent.
`[measured: visual]` Some apparent Skiing gains are object deletion. A future
router must encode top/visible/background separately, directly own the convex
blend, and log causal output amplitude.

### CL18 - same-ID cross-view consistency

`[measured]` CL18 is neutral/negative overall and especially weak on Crying.
The inference architecture is unchanged. Its final consistency loss is very
small. A loss that asks two different spatial references to produce the same
prediction can teach the model to ignore spatial-reference variation; it does
not distinguish identity-stable features from useful pose/view evidence.
`[hypothesis]` Multi-view data remains useful as a reference pool for a direct
identity reward, not as prediction invariance.

### CL19 - true-soft full-query router

`[measured] [code]` This is the clear success. It removes CL14's thresholded
feather and double output mask, preserves target Q, computes complete native
and reference messages, and blends once. It improves every prompt family and
seven of eight identities over CL14. The next architecture should preserve
this exact base and change only one orthogonal mechanism.

### CL20 - BigCelebs curriculum

`[measured]` CL20 has the highest text score, but its identity is CL14-level and
far below CL19. The final 4k Cosmic re-anchoring does not recover the gap. This
is a valid negative on **generic dataset mixing with the CL14 path**. It does
not establish that labeled hard-case data is useless; it establishes that more
depth/scene diversity without a causally effective ownership or ID mechanism
does not fix the architecture.

# 5. Prior-project evidence that constrains the next experiments

The current project and `/home/kolyangg/rsrch` contain many attractive ideas
that should not be rediscovered without their failure conditions.

| Prior experiment/result | Observed evidence | Constraint carried forward |
|---|---|---|
| CL9 late ROI suffix | Four-seed, fixed-checkpoint ROI step-18 intervention: mean ID about `+0.097`, `43/56` wins, exact outside ROI | Local high-resolution work can help, but training must guarantee path activation. `[report]` |
| CL9 precise occluder polygons | Mean ID about `+0.038`, but only `4/7` Skiing topology successes; goggles duplicated/relocated | Geometry masks alone do not encode visibility order. `[report]` |
| Marion eye-line roll / similarity | Four-seed roll mean about `+0.001`; five-point alignment neutral/unstable | Do not spend another arm on 2D reference normalization. `[report]` |
| Larger rank / stronger reference scale | Paths were active but stronger scale harmed or plateaued | Capacity and amplitude are not the missing semantic control. `[report]` |
| N31 ranking shortcut | Raised a proxy while desaturating/shortcutting outputs | Auxiliary score success is not image success. `[report]` |
| N32 unaligned patch capacity | More patch capacity did not solve identity | Raw capacity without aligned routing is insufficient. `[report]` |
| N36-N38 causal probes | Reference memory existed; causal identity effect remained attenuated and loss nearly zero | Add direct identity causality, not another memory bank. `[report]` |
| E12 corrected hard identity CA | Target-Q/ID-KV hard face replacement was catastrophic | Keep native CA intact; identity CA must be a bounded residual. `[report]` |
| E17 residual identity CA | Safe/null on weaker substrate, branch telemetry missing historically | Retest corrected v3 on CL19 with gate and RMS telemetry. `[report]` |
| E22 naive one-step x0 ArcFace | Failed and degraded images | Any ID objective must be low-noise, multi-step, quality/trajectory anchored. `[report]` |
| E13 joint shadow co-adaptation | Joint BA plus adapters worked better than persistent/pretrained-only variants | Train new routing with the main model; do not bolt it on only at inference. `[report]` |
| Multi-reference PhotoMaker tokens | Weak/neutral on the strong base | More ID tokens do not solve spatial ownership. `[report]` |
| BigCelebs E13 and ds1/ds2/ds3 | BigCelebs endpoints `0.4149-0.4296`, generally below/near Large Dataset E13 `0.4304` | Broad dataset substitution is not a high-probability ID fix. `[report]` |

Three reusable principles emerge.

1. **A new module must affect the denoiser, not merely learn its auxiliary
   label.** Use a centered/floored route and log output/native RMS.
2. **The target/native path owns composition and top objects.** Reference
   identity is residual, visible-region, and late-detail information.
3. **Identity supervision must see a sufficiently denoised face and must be
   anchored against reward hacking.** A noisy one-step x0 estimate is not a
   reliable ArcFace input.

# 6. Dataset decision

## 6.1 What CL20 establishes

CL20 trains on an 80/20 Cosmic/curated depth-at-least-six BigCelebs schedule
through 20k, then uses 4k Cosmic-only re-anchoring. `[code]` Its endpoint is:

```text
ID       0.450064  (CL14 delta -0.006052; CL19 delta -0.056759)
Text    26.593262  (best of CL14-CL20)
IoU      0.893410
```

The curriculum plausibly broadens caption/scene semantics, as reflected in
text similarity, but it does not improve identity or topology. `[measured]`
Historical BigCelebs training was also worse than Cosmic/Large Dataset on the
same families. This is enough evidence to remove generic BigCelebs mixing from
the six highest-probability arms.

## 6.2 Recommended data use

Keep Cosmic as the training distribution for CL21-CL24 and CL26. For CL22 and
CL24, use the existing deterministic CL17 synthetic occluder family only after
versioning its labels into:

```text
top-object overlap | visible face | background | contact boundary | object present
```

This is supervision for one mechanism, not a dataset replacement. Preserve
the index seed and balance eyewear/goggles/hair/hand/tear types. Add loss and
object-retention telemetry by type.

For CL25, use at least three **distinct same-ID Cosmic faces** to construct an
identity centroid/reward pool. Do not reward similarity to only the conditioning
crop; that encourages copy-pose and expression collapse.

BigCelebs can become useful later as a **mined, quality-filtered annotation
source** if CL22 or CL24 first proves the mechanism on Cosmic. Required filters
would include one owned face, trustworthy same-ID grouping, face size, explicit
occluder caption or segmentation, top/bottom order, and no conflict with the
fixed validation identities. `[hypothesis]` A second-stage data transfer is not
one of the six independent architecture tests.

# 7. External research: deeper transfer analysis

## 7.1 Search scope

The 11 August archive already contained nineteen requested/comparable papers:
PuLID, DreamCache, DynamicID, UniPortrait, InfiniteYou, SpatialID, Diff-PC,
AnyPhoto, PersonaHOI, RealisID, SerialGen, Face2Diffusion, ConsistentID,
MasterWeaver, ID-Patch, Leffa, ReSem-Face, MagicMakeup, and UniversalBooth.

This review adds eighteen primary PDFs and searchable text, including 2026
work available by 13 August. The new archive and hashes are at
[`analysis/sources/2026-08-13_cl19_architecture_review/`](sources/2026-08-13_cl19_architecture_review/SOURCES.md).
The newest directly relevant papers found were ReSem-Face (5 August 2026,
prior archive), Holistic Identity / DBS and Diff-ID (late July 2026), BioDDM
(CVPRW 2026), and LayerBind (March 2026).

External results use different backbones, tasks, datasets, and metrics. They
support mechanisms, not predicted CL19 effect sizes.

## 7.2 Requested approaches revisited after CL15-CL20

| Approach | Most useful idea | What CL15-CL20 changes | Decision for CL19 |
|---|---|---|---|
| [PuLID](https://arxiv.org/abs/2404.16022) | Accurate low-step generated-image ID supervision plus contrastive semantic/layout protection | E22 rejects the naive one-step version; CL19 now supplies a stronger route | Transfer the low-noise/anchored objective, not PuLID's full architecture: CL25. |
| [DreamCache](https://arxiv.org/abs/2411.17786) | Clean null-text, low-noise U-Net features cached at middle/decoder scales | CL16's clean-memory scale collapsed; earlier memory probes found capacity without causality | Do not repeat now; revisit only if an active ID route becomes memory-limited. |
| [DynamicID](https://arxiv.org/abs/2503.06505) | Semantic-Activated Attention and identity/motion reconfiguration | CL17 shows an unconstrained semantic gate can learn labels while routing almost nothing | Use explicit states, floors, and routed/native telemetry: CL22. |
| [UniPortrait](https://arxiv.org/abs/2408.05939) | Separate intrinsic identity from structural features; route each identity spatially | CL19 already supplies target/reference separation, but no top-object state | Keep PhotoMaker/BA; add visibility ownership rather than replacing the encoder. |
| [InfiniteYou](https://arxiv.org/abs/2503.16418) | Residual personalization and same-person/different-image training reduce copy-paste | CL18's prediction consistency likely encouraged reference indifference | Use multi-view faces as a reward pool or residual ID K/V, not equality of predictions. |
| [SpatialID](https://arxiv.org/abs/2602.13994) | Identity relevance varies across space and denoising time | CL19 varies only across geometric mask distance | Add time/frequency relevance while retaining a nonzero reference floor: CL23. |
| [Diff-PC](https://arxiv.org/abs/2602.00639) | Explicit identity versus pose disentanglement and same-ID cross-view training | CL18 demonstrates that raw view invariance is insufficient | Preserve target Q/native structure and supervise identity directly; no 3D injector now. |
| [AnyPhoto](https://arxiv.org/abs/2603.14770) | Identity-isolated attention tied to location and replacement curriculum | CL19 validates isolated lanes; CL17/CL20 expose gate/data failure modes | Borrow location/ownership discipline, not aligned modulation that risks copy-paste. |

## 7.3 New occlusion and visibility-order evidence

### VODiff: visibility is ordered, not binary

[VODiff](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_VODiff_Controlling_Object_Visibility_Order_in_Text-to-Image_Generation_CVPR_2025_paper.pdf)
shows that layout boxes alone put overlapping objects on the same conceptual
layer. Its visibility-order-aware objective separates overlap, visible, and
background attention terms; ablations show the combined terms improve
occlusion accuracy. `[literature]` This maps directly onto CL19's failure:
inside the face box, "occluded" is not enough. The model needs to know that the
goggle/hand/hair is the top object, visible skin is underneath, and background
is neither.

Transfer: CL22 uses three logits and directly owns the native/reference convex
blend. Unlike CL17, it cannot report success based only on a binary BCE, and it
keeps an explicit object-retention term.

### PersonaCraft: correct the contact boundary, not the whole object interior

[PersonaCraft](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_PersonaCraft_Personalized_and_Controllable_Full-Body_Multi-Human_Scene_Generation_Using_Occlusion-Aware_ICCV_2025_paper.pdf)
uses depth-edge signals inside occlusion masks and localized conditioning in
occluded regions. Its ablations indicate that an occlusion-only path can lose
global pose, while global geometry plus a local boundary enhancer works better.
`[literature]` The transferable insight is sparse: keep CL19/native global
structure and apply a teacher or ownership correction only around the
top-object/face contact ring. This motivates CL24.

### LayerBind: layout first, semantic detail later

[LayerBind](https://arxiv.org/abs/2603.05769) separates early layer binding from
later semantic nursing. Early denoising establishes regional and occlusion
order; later stages reinforce object detail without re-solving the whole
layout. `[literature]` The model is a training-free DiT controller, so its
implementation is not portable, but its timing is. CL23 lets the native lane
retain early/low-frequency structure and strengthens reference identity in
late/high-frequency detail.

### FaceDancer and Parallel Visual Attention: target attributes must remain a
first-class path

[FaceDancer](https://arxiv.org/abs/2210.10473) adaptively blends target skip
features and identity-conditioned decoder features; its analysis associates
earlier face-recognition features with pose, expression, and occlusion, while
later features are more identity-specific. `[literature]`
[Parallel Visual Attention](https://openaccess.thecvf.com/content/WACV2024/papers/Xu_Personalized_Face_Inpainting_With_Diffusion_Models_by_Parallel_Visual_Attention_WACV_2024_paper.pdf)
keeps text and visual attention separate and evaluates explicit eye/lower-face
semantic occlusions. `[literature]` Both support keeping the native target path
intact. PVA receives an inpainting mask, so it does not solve CL19's need to
infer layer ownership from a prompt and evolving latent.

## 7.4 New identity-supervision evidence

### Centered temporal gates avoid the CL15-CL17 failure mode

[Holistic Identity / DBS](https://arxiv.org/abs/2607.25622) uses adaptive
temporal gating around the existing contribution, rather than requiring a new
branch to escape an exact zero. It also separates face, appearance, and global
objectives. `[literature]` The direct lesson is parameterization: future
learned paths need a centered or floored contribution and region-specific
telemetry. CL21 begins with a `0.02` identity-CA gate; CL26 cannot fall below
`0.05`; CL22's probabilities directly own the blend.

### Time-aware frequency routing separates structure from identity detail

[TFCustom](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_TFCustom_Customized_Image_Generation_with_Time-Aware_Frequency_Feature_Guidance_CVPR_2025_paper.pdf)
adapts reference injection by timestep and frequency and reports that this
improves subject detail. It also restricts its reward signal to more reliable
low-noise behavior. `[literature]` CL23 applies the minimal transferable part
to the **reference-minus-native BA message**, not to Q/K/V separately. This
preserves CL19 exactly when all scales equal one and avoids importing a new
ReferenceNet.

### Low-noise, multi-step reward is a different experiment from E22

[ReF-LDM](https://arxiv.org/abs/2412.05043) reports that a naive ArcFace loss at
noisy timesteps damages quality, while timestep scaling improves identity with
less degradation. `[literature]`
[Reference-Guided Identity Preserving Face Restoration](https://arxiv.org/abs/2505.21905)
adds reference/HQ hard examples because the ordinary identity loss rapidly
becomes too small; its severe-restoration ablations show a large identity
benefit, with a reference-versus-target tradeoff that must be balanced.
`[literature]`

[Portrait reward optimization](https://openaccess.thecvf.com/content/ICCV2025W/HiGen/papers/Liu_Enhancing_Identity_Preservation_in_Portrait_Generation_via_Reward_Optimization_ICCVW_2025_paper.pdf)
uses a pathwise ArcFace reward and KL regularization; pathwise gradients were
more stable than high-variance policy-gradient alternatives in its small-batch
setting. `[literature]`
[IPRO](https://arxiv.org/abs/2510.14255) backpropagates only through the final
four sampling steps, uses a face pool rather than one frame, and constrains the
reverse trajectory to a frozen model. Its ablations favor low-noise steps and
show reward hacking without the anchor. `[literature]`

Together these are a materially different design from E22: CL25 starts from
the exact CL19 endpoint, differentiates only the final four DDIM steps, rewards
a same-ID centroid, and penalizes deviation from frozen CL19 at each of those
steps.

### Patch-local and multi-region objectives

[PatchDPO](https://arxiv.org/abs/2412.03177) finds that image-level preference
can reward bad local patches together with good ones; patch-weighted feedback
performs better in its customization benchmark. `[literature]`
[FairHuman](https://arxiv.org/abs/2507.02714) separates face, hand, and global
objectives and allocates gradients to reduce regional conflict. `[literature]`
These support sparse boundary and object-presence terms in CL22/CL24. A full
patch-DPO dataset or multi-objective gradient solver is not promoted now; both
would add a second major change and substantial annotation burden.

### Identity subspaces are promising but not yet the shortest route

[BioDDM](https://openaccess.thecvf.com/content/CVPR2026W/AIMS/papers/Dosi_From_Pixels_to_Identity_Biometric_Subspace_Guidance_for_Diffusion-Based_Face_CVPRW_2026_paper.pdf)
projects corrections into identity-bearing embedding directions while leaving
orthogonal appearance directions freer. `[literature]` This is conceptually
attractive for glasses/hair/expression separation, but its restoration-time
embedding inversion is not a direct U-Net BA layer. It remains a reserve idea
after CL21/CL25 establish whether explicit ID tokens or reward already close
the gap.

# 8. Six independent CL19 experiments

These arms are ordered by expected value, combining probability of a real
effect, implementation risk, and relevance to the two goals. They are not a
stack. Promote one only from matched, immutable full-96 evidence.

Exact proposed Hydra config names, in priority order:

1. `CL21_cosmic_true_soft_router_resididca_v3_24k`
2. `CL22_cosmic_visibility_order_router_24k`
3. `CL23_cosmic_temporal_frequency_router_24k`
4. `CL24_cosmic_pm_boundary_distill_24k`
5. `CL25_cosmic_low_noise_id_reward_4k`
6. `CL26_cosmic_anchored_highres_roi_ba_24k`

| Priority | Arm | Directional prediction | Main risk |
|---:|---|---|---|
| 1 | Residual identity CA | Broad ID increase, especially the Jennie/Jisoo gap, with nearly unchanged layout | Residual CA could over-strengthen identity; the active CL14_CA result may change the premise |
| 2 | Visibility-order router | Fewer fragmented/nested Skiing layers with occluders retained; ID nonnegative | Synthetic labels may not transfer; the head may shortcut unless route amplitude is enforced |
| 3 | Temporal-frequency router | Better boundary topology and fine ID detail with stable early composition | Frequency splitting can ring or suppress valid low-frequency identity shape |
| 4 | PM boundary distillation | PhotoMaker-like contact boundaries without reducing visible-face CL19 identity | Teacher leakage may pull the whole output toward PhotoMaker and increases compute |
| 5 | Low-noise ID reward | Largest direct ID upside; possible approach to or passage of PM0 | ArcFace reward hacking, rigid/copy-pose faces, and high memory/step cost |
| 6 | Anchored high-res ROI | Higher ID on small/local faces if the CL9 ROI effect transfers | Active ROI may sharpen an incorrect face-object composite or copy reference attributes |

## Priority 1 - CL21: corrected residual identity-token CA v3

**Hypothesis.** `[hypothesis]` CL19's spatial reference SA has strong geometry
and identity, but it remains `0.04976` below PhotoMaker. A small, late,
face-local target-Q -> active PhotoMaker identity-token K/V residual can add
identity without replacing native text/PhotoMaker CA or changing the soft
router.

**Single delta.** Enable the existing corrected v3 processor only in
`up_blocks.0/1`, rank 64, gate init `0.02`, maximum `0.20`. Native CA is added
unchanged; legacy branched CA, hard identity CA v2, pose adaptation, and
`ca_mixing_for_face` remain off.

```text
Y = NativeCA(Q_target, text + PhotoMaker context)
    + g * FaceMask * ResidualCA(Q_target, active_ID_tokens)
```

**Implementation.** Reuse
`residual_identity_ca_processor_v3.py` and CL14_CA's runtime/telemetry. The
CL19 child should pin `2,348` trainable tensors and `224,624,676` parameters.
Startup must observe exactly two active ID tokens, gate `0.02`, finite
native/residual RMS, and zero initial residual output from the zero-initialized
projection. `[code]`

**Why first.** This is the most implementation-ready direct ID intervention.
Historical E12 rejects hard replacement, whereas corrected residual CA has
been safe. `[measured: interim]` The corrected CL14_CA optimized r11 arm is
already `+0.02158` over CL14 at the matched 10k panel, which strengthens the
mechanism. Its final endpoint and face-quality safeguards are still absent, so
they must be read before CL21 launch. `[not established]` A positive CL14
interaction does not guarantee a positive CL19 interaction.

**Accept.** Positive paired ID interval over CL19, no Skiing/Crying topology or
object-retention regression, and no material text/IoU/TOPIQ-Face-p10 loss.
Stretch target: `ID >= 0.556580`.

**Blueprint:**
[`CL21_residual_identity_ca_v3.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL21_residual_identity_ca_v3.blueprint.yaml)

## Priority 2 - CL22: explicit visibility-order router

**Hypothesis.** `[hypothesis]` CL19's remaining goggle/hand failures arise
because a geometric face router cannot decide which semantic surface owns an
overlap cell. A three-state router can preserve the native top object, route
reference identity to visible face, and leave background native.

**Single delta.** Version CL17's one-logit head into three logits:

```text
p_top, p_visible, p_background
    = softmax(Head(target_hidden, native-reference disagreement, timestep))

top object   -> native weight >= 0.95
visible face -> reference weight >= max(CL19 router, 0.50)
background   -> native
```

Use class-balanced top/visible/background supervision and a two-cell contact
boundary. The existing deterministic synthetic family becomes
`visibility_order_v2` labels; object presence is supervised so goggle deletion
cannot win. The probabilities directly own the convex blend-there is no
near-zero learned output multiplier.

**Implementation.** Modify the dataset label output, attention processor,
runtime, `lora2.py`, and diffusion loss. Keep CL19 target Q, native message,
reference K/V support, and one final blend. Pin the new ownership counts after
implementation; the blueprint deliberately leaves them `TO_COMPUTE`.

**Accept.** Reduce CL19's three clear Skiing failures and one Crying failure in
a blinded review, retain all prompted occluders, and do not regress full-96 ID.
Reject even with higher Skiing ID if goggles disappear or if routed/native RMS
is below one percent.

**Blueprint:**
[`CL22_visibility_order_router.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL22_visibility_order_router.blueprint.yaml)

## Priority 3 - CL23: temporal-frequency CL19 delta routing

**Hypothesis.** `[hypothesis]` The native lane should determine early,
low-frequency layout and visibility order, while the reference lane should be
strongest for late, high-frequency identity detail. CL19 currently uses one
time-independent blend.

**Single delta.** Split only the reference-minus-native message:

```text
d      = R - N
d_low  = GaussianBlur(d)
d_high = d - d_low
Y      = N + r_CL19 * (g_low(t) d_low + g_high(t) d_high)
```

Use a deterministic cosine/SNR schedule:

```text
g_low:  0.50 early -> 0.85 late
g_high: 0.75 early -> 1.25 late
```

Reference contribution never falls below `0.50`, so this remains explicit BA
rather than a native-only ablation. All-one scales must reproduce CL19
numerically. No trainable parameters are added; ownership remains
`2,240 / 219,217,920`.

**Implementation.** Add an FP32-stable split/merge after CL19 forms its two
messages. Do not filter Q/K/V independently. Log low/high delta RMS, scales,
and merged/native RMS.

**Accept.** Positive paired ID delta and fewer boundary failures with intact
reference causality. Reject if wrong-ID/reference-shuffle probes show that the
schedule simply suppresses the reference path.

**Blueprint:**
[`CL23_temporal_frequency_router.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL23_temporal_frequency_router.blueprint.yaml)

## Priority 4 - CL24: sparse PhotoMaker boundary distillation

**Hypothesis.** `[hypothesis]` Controlled PhotoMaker's native denoiser carries
a better top-object/face boundary prior. Matching it only at contact boundaries
can improve topology without pulling visible-face identity back toward
PhotoMaker.

**Single delta.** On the sealed 25% hard-case subset, run a frozen original
PhotoMaker teacher at the same latent, timestep, prompt, and reference. Apply:

```text
L_boundary = mean(B_contact * Charbonnier(eps_CL19 - stopgrad(eps_PM)))
L_top      = mean(M_top * Charbonnier(eps_CL19 - stopgrad(eps_PM)))
L_total    = L_diffusion + 0.05 L_boundary + 0.02 L_top
```

`B_contact` is a sparse two-cell ring around the top-object/face intersection.
Visible nonboundary face pixels receive no teacher loss. Inference remains
exact CL19 and trainable ownership stays `2,240 / 219,217,920`.

**Implementation.** Reuse the versioned CL22/CL17 label geometry, freeze and
exclude all teacher tensors from optimizer/checkpoints, and run the teacher
only on selected rows. Measure memory/throughput before packaging the full job.

**Accept.** Fewer topology failures with no full-96 ID regression or global
convergence toward PhotoMaker. Reject if gains exist only on synthetic shapes,
or text/composition falls because the teacher loss leaks outside the boundary.

**Blueprint:**
[`CL24_photomaker_boundary_distillation.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL24_photomaker_boundary_distillation.blueprint.yaml)

## Priority 5 - CL25: low-noise multi-step identity reward continuation

**Hypothesis.** `[hypothesis]` Directly optimizing a sufficiently denoised face
can close the ID gap that diffusion MSE leaves, provided the path is constrained
against ArcFace reward hacking and reference-pose copying.

**Single delta and schedule.** Start from the exact hash-pinned CL19 24k
checkpoint at local step 0, reset optimizer/scheduler, and train 4,000 local
steps. On `1/16` batches, generate the sampling prefix under no-grad and retain
the graph through only the final four DDIM steps:

```text
R_id = cosine(ArcFace(differentiable_crop(x0)), same_ID_reference_centroid)
L_KL = sum_last4 ||eps_candidate - stopgrad(eps_frozen_CL19)||^2
L    = L_standard + 0.05 (1 - R_id) + 1.0 L_KL
```

The centroid uses at least three distinct same-ID faces. A fixed differentiable
bbox transform feeds ArcFace; a nondifferentiable detector cannot sit in the
gradient path. Log valid-face fraction, reward gradient norm, and trajectory
divergence. Validation at local 0/2k/4k remains the unchanged full96 panel.

**Why fifth despite high upside.** This is the arm most directly aligned with
beating PhotoMaker ID, but it has the highest compute and metric-overfitting
risk. An independent face encoder diagnostic is required even though the
official decision metric remains unchanged.

**Accept.** Positive paired subject-v2 ID at 2k and 4k, corroborated by the
independent encoder, with no rigid/copied expressions, face-quality loss,
topology regression, or runaway KL. Stretch target: `ID >= 0.556580`.

**Blueprint:**
[`CL25_low_noise_identity_reward.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL25_low_noise_identity_reward.blueprint.yaml)

## Priority 6 - CL26: anchored high-resolution ROI BA

**Hypothesis.** `[hypothesis]` CL15 failed because its output gate collapsed,
not because the shared-projection ROI path was harmful. A late ROI residual
with a guaranteed active range can reproduce part of the CL9 fixed-checkpoint
gain and help smaller/local faces.

**Single delta.** Version CL15's 32x32 target-ROI-Q/reference-ROI-KV residual in
up0/up1. Replace `0.25*tanh(raw)` with an anchored sigmoid in `[0.05,0.25]`,
initialized at `0.10`. Apply only in the final 40% of denoising and cap
residual/native ROI RMS at `0.25`.

**Implementation.** Reuse CL15's shared projections; do not edit the historical
mode. A wrong-reference smoke must change only the intended face ROI before
downstream convolution mixing. Pin actual parameter ownership after
implementation.

**Accept.** Positive full-96 ID, concentrated in smaller faces, without copied
reference eyewear/hair, outside-face drift, or harder Skiing/Crying topology.
Reject if the routed/native ratio again becomes effectively null.

**Blueprint:**
[`CL26_anchored_highres_roi.blueprint.yaml`](blueprints/2026-08-13_cl19_next_six/CL26_anchored_highres_roi.blueprint.yaml)

# 9. Launch and decision protocol for the implementing agent

For each arm:

1. Read the current handoff and verify live Serv Running/Pending allocations;
   the normal project ceiling is six requested A100s.
2. Implement one defaults-off path. Do not change CL19, CL14, the fixed panel,
   scheduler, prompt/ref/bbox files, metric definitions, or checkpoint schema.
3. Compose old and new modes. Verify the off mode reproduces CL19 and pin exact
   optimizer/trainable ownership.
4. Run the smallest forward/backward/startup smoke that proves route activation,
   finite telemetry, and checkpoint round-trip. Do not promote an auxiliary
   loss without output/native RMS.
5. Create a fresh immutable experiment JSON, launcher, and one-A100 Serv
   package. Submit only after the resource gate.
6. During startup, verify `saved/<run>/comet_experiment.json` exists and record
   its immutable key. Monitor through validation and training startup.
7. Evaluate step 0 plus every 2k. Use the same 96-row completion seal before
   reading a step as complete.
8. Promote only from the matched full-96 endpoint/peak analysis, seven
   face-quality curves, per-image paired table, and hard-case topology/object
   review.

Do not stack arms during these six tests. If two independent arms pass, a later
combination should be a separate experiment with its own interaction risk. The
likely route to actually exceed PhotoMaker is one successful **identity** arm
(CL21 or CL25) plus one successful **ownership/timing** arm (CL22, CL23, or
CL24), but that combination is not established and is outside this six-arm
single-delta plan.

# 10. Confidence and unresolved questions

| Conclusion | Confidence | Basis / limitation |
|---|---|---|
| CL19 is the best CL15-CL20 run | High | Exact 24k, full96, immutable keys; large paired delta |
| CL19 still trails PhotoMaker ID | High | Same controlled panel; negative paired interval and 18/96 wins |
| CL19 largely repairs Crying but not Skiing | Medium | Complete fixed panel, but one unblinded reviewer and one seed |
| CL15/CL16 paths were near-null | High | Final checkpoint formulas and effective scales |
| CL17 learned labels without controlling generation materially | High | BCE/head activation plus sub-percent routed scale and negative endpoint |
| Generic BigCelebs mixing is low priority | Medium-high | CL20 plus multiple historical negative/neutral arms; not every possible curation tested |
| Visibility order is the main residual topology mechanism | Medium | Strong visual/code fit and external evidence; untrained on CL19 |
| Low-noise reward can close the PM gap | Medium-low | Several external positives, but E22 negative and backbone/task mismatch |
| Any one proposed arm will beat PhotoMaker | Low | Required improvement is about `+0.050`; no proposed arm is trained |

`[not established]` The following remain open:

- whether PhotoMaker's `8/8` hard-case topology generalizes beyond the sixteen
  reviewed images;
- whether CL19's Jisoo/Jennie gap is a representation problem, a reference-view
  problem, or subject-v2 scorer sensitivity;
- whether active CL14_CA ultimately improves identity; it had no completed
  scientific endpoint at this evidence cutoff;
- whether the CL9 ROI gain transfers from a fixed-checkpoint intervention to a
  jointly trained CL19 path;
- population performance beyond eight identities, twelve prompts, and seed 0;
- an automated topology metric that distinguishes coherent occlusion from
  object deletion. Until one is validated, visual object-retention review is
  mandatory.

# 11. Reproduction and artifacts

Run from `diffusion_template/` in the existing `photomaker`/`photomaker_NS`
environment.

```bash
python tools/comet/export_comet_runs.py \
  --manifest tools/comet/comet_runs_13Aug_CL15_CL20_PM0_CL14.json \
  --output-dir comet_data/13Aug_CL15_CL20_PM0_CL14_latest

python analysis/assets/cl15_cl20_20260813/build_assets.py
```

The exporter obtains `COMET_API_KEY` from the gitignored environment; it must
not be printed or committed. Reproduction assets:

- [`run_metrics.csv`](assets/cl15_cl20_20260813/run_metrics.csv)
- [`paired_id_comparisons.csv`](assets/cl15_cl20_20260813/paired_id_comparisons.csv)
- [`per_image_id.csv`](assets/cl15_cl20_20260813/per_image_id.csv)
- [`identity_means.csv`](assets/cl15_cl20_20260813/identity_means.csv)
- [`action_means.csv`](assets/cl15_cl20_20260813/action_means.csv)
- [`visual_review.csv`](assets/cl15_cl20_20260813/visual_review.csv)
- [`checkpoint_path_audit.csv`](assets/cl15_cl20_20260813/checkpoint_path_audit.csv)
- [`cl14_ca_interim_10k.csv`](assets/cl15_cl20_20260813/cl14_ca_interim_10k.csv)
- [`derived_summary.json`](assets/cl15_cl20_20260813/derived_summary.json)
- [`SHA256SUMS.txt`](assets/cl15_cl20_20260813/SHA256SUMS.txt)

The six YAMLs parse as valid YAML but are not Hydra configs. Their
`design_only_non_runnable` status and unresolved ownership/hash fields are
intentional safety gates.

# References and source archives

Project evidence:

- [CL9 validation interventions](2026-08-11_cl9_validation_interventions_results.md)
- [CL14 hard-case architecture review](2026-08-11_cl14_hard_cases_architecture_research_and_experiment_plan.md)
- [CL19 soft-router architecture versus CL14](2026-08-11_cl19_soft_router_architecture_vs_cl14_pose_adapt_ca_mixing.md)
- [Branched cross-attention history and corrected CL19 reintroduction](2026-08-12_branched_cross_attention_disable_history_and_cl19_reintroduction.md)
- [Loss objective and identity supervision advice](2026-08-05_loss_objective_and_identity_supervision_advice.md)
- [E13-E18 results and next experiments](2026-08-06_e13_e18_results_and_next_experiments.md)
- [CL8-CL11 hard cases and CL12-CL14](2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md)

Primary-paper archives:

- [11 August requested/comparator archive](sources/2026-08-11_cl14_architecture_review/SOURCES.md)
- [13 August extended archive, transfer notes, and hashes](sources/2026-08-13_cl19_architecture_review/SOURCES.md)

All literature conclusions in this report are paraphrases from the linked
primary papers. No external paper establishes performance on this exact
PhotoMaker + CL19 branched-attention implementation.
