# CL39 beats PhotoMaker on identity and hard cases, but much of its gain follows a PhotoMaker-native retreat

**Date:** 21 August 2026  
**Evidence cutoff:** 18:53 BST / 17:53 UTC, 21 August 2026  
**Scope:** completed CL38-CL45 scientific runs; immutable Comet histories and
table-sealed fixed-96 panels; paired per-image identity analysis; hard-case
visual review; decoded-pixel and perceptual comparison with controlled
PhotoMaker; source/configuration shortcut audit. Failed retries, CL38 r3, and
bounded smokes are operational evidence only and are excluded.  
**Primary metric:** mask-owned subject-v2 `manual_val/id_sim` on fixed
`manual_val`-96.  
**Reproducible assets:**
[`assets/cl38_cl45_20260821/`](assets/cl38_cl45_20260821/)

## Executive conclusion

**CL39 at 16k is the first result in this series that beats controlled original
PhotoMaker on aggregate identity and also cleanly resolves the fixed-panel
Skiing topology case.** Its subject-v2 ID_SIM is `0.570124`, versus CL27-16k
`0.547260` and PhotoMaker `0.556580`. The paired gain is `+0.022864` over
matched CL27 with `70/96` wins and a 95% cell-bootstrap interval of
`[+0.012937,+0.032875]`; versus PhotoMaker it is `+0.013544`, `57/96` wins,
`[+0.004501,+0.023137]`. The result is not merely a selected-checkpoint spike:
CL39-24k remains at `0.566342`, `+0.009762` over PhotoMaker with interval
`[+0.000636,+0.019052]`. `[measured][paired]`

The visual result is equally important. CL39-16k is `8 pass / 0 minor / 0
fail` on the eight Skiing cells under the existing object-order rubric,
compared with PhotoMaker `8/0/0` and CL27-16k `6/1/1`. Large goggles stay above
readable faces, the male identities retain their ordinary eyewear, and Marion's
eyes remain readable. Crying remains `8/0/0`; the hands, hair, and faces are
not fused. CL39 also raises Jumping from `0.3946` to `0.4437` and Marion's
all-prompt mean from `0.4935` to `0.5161`. `[measured][visual]`

**The generated images are different from PhotoMaker's; there is no literal
output leak or replay.** For every selected CL38-CL45 panel, `0/96` decoded RGB
images exactly match the same-cell PhotoMaker image. For CL39, mean pixel MAE
is `0.04088` on `[0,1]`, SSIM is `0.83814`, face-crop SSIM is `0.79950`, and
the difference images contain structure across the person and background.
`[measured]`

The user's shortcut concern is nevertheless **partly correct**. CL39 is much
closer to PhotoMaker than CL27 is: same-cell CLIP image cosine is `0.96363`
versus `0.93393`, SSIM is `0.83814` versus `0.82095`, and face SSIM is
`0.79950` versus `0.76601`. Paired CL39-minus-CL27-to-PhotoMaker closeness is
positive in `81/96`, `80/96`, and `86/96` cells, respectively, with all three
bootstrap intervals above zero. This pull already occurs at step zero, before
training. `[measured][paired]`

The code explains why. CL39 does not load PhotoMaker generations or optimize
ID_SIM. It keeps target queries and reference K/V, but multiplies CL27's
explicit reference-minus-native residual by an entropy confidence whose
training median is only `0.315588`; about `68.4%` of that residual is therefore
suppressed on average. The fallback is the native target self-attention path,
whose branch projections were initialized from effective PhotoMaker weights,
while the generic and PhotoMaker-default adapters are also co-trained. CL39 is
still an SA-only BA model with a nonzero reference lane, but the present data
do **not** establish that BA—rather than the trained PhotoMaker/native path—is
what causes the surplus over original PhotoMaker. `[code][measured][caveat]`

The practical decision is therefore two-track:

- **Use CL39-16k as the provisional best end-to-end checkpoint** for identity,
  face quality, and the fixed hard cases.
- **Keep CL27-16k as the causal BA research control** until the same CL39
  checkpoint is evaluated with BA disabled, the null router disabled, and the
  spatial reference shuffled while identity tokens remain fixed. Do not call
  CL39 a proven BA-over-PhotoMaker result before those evaluation-only
  counterfactuals.

CL44 is the only secondary positive arm. It reaches `0.550846` at 22k and
improves over matched CL27 by `+0.008952`, interval
`[+0.002135,+0.015999]`, but it remains below PhotoMaker and still fails Lex
and Marion Skiing topology. CL38, CL41, and CL45 are clearly negative; CL40,
CL42, and CL43 are neutral configurations with no promotion evidence.
`[measured][decision]`

# 1. Evidence integrity

## 1.1 Immutable completed scientific records

Each CL38-CL45 key below has all 13 expected per-image Comet tables at steps
0, 2k, ..., 24k. Every selected export contains exactly 96 images and one
unique 96-row table at the exact requested step, with no nearest-step fallback
or exporter warning. `[measured]`

| Arm | Single change versus CL27 | Immutable Comet key | Selected step | Final step |
|---|---|---|---:|---:|
| CL38 r4 | corrected delta-only native ownership anchor | `368d6fe8caec43fab8be374a8926d6ed` | 18k | 24k |
| CL39 r4 | entropy/null-key confidence fallback | `b1ca0b3da679401c85b991f1bbdf0b2a` | 16k | 24k |
| CL40 r5 | rank-32 identity-motion projector | `1c2e0ac2fcae433db18f55de663b59ef` | 20k | 24k |
| CL41 r4 | five-landmark canonical reference K/V | `b40179ef6a9d4dd6954f6d06d148069c` | 16k | 24k |
| CL42 r5 | five-token facial component memory | `9613ca23f49f469b9bc0fda89055483d` | 16k | 24k |
| CL43 r5 | 512-D ID-adaptive residual modulation | `d29cbfa7927547c9ac71a8da0b583e33` | 22k | 24k |
| CL44 r5 | semantic/time high-frequency window | `42928f13f7ee41448d3d715231f8bb32` | 22k | 24k |
| CL45 r3 | asymmetric BA-only PCGrad | `bfb129031773494f881ea629ced3fe60` | 18k | 24k |

Controls are CL27 r3, key `dbfbf40c3bdd4f70bedc58bda3dfb9cd`,
selected at 16k, and original PhotoMaker, key
`74efd227d3f8488a98e83d815c77c07c`, at step zero. CL38 r3 key
`296f91454faa4b3cbd5b8cb98cfed5d6` is excluded because its incorrect
auxiliary gradient collapsed the model. `[measured][code]`

## 1.2 Fixed protocol and joins

The comparison preserves the fixed 96-image `manual_val` panel: the same
prompts, identities, reference images, subject-selection boxes, generated-face
boxes, seed 0, RealVisXL V4.0 validation base, DDIM50, CFG 5, and one generated
image per cell. The primary score is subject-v2 mask-owned ID_SIM; legacy
best-face ID is reported only as a diagnostic. Output keys containing spaces
were normalized to PNG filenames containing underscores before table/image
joins. `[measured]`

Selected checkpoints are each arm's maximum subject-v2 mean over completed
sealed panels. That selection is descriptive and can be optimistic. The
separate 24k endpoint analysis is therefore the robustness check. Paired
intervals use 50,000 bootstrap resamples of the 96 fixed cells. They quantify
variation across this panel, not across training seeds or identities sampled
from a population. `[method][limitation]`

The image-distance audit decodes RGB pixels before hashing, then computes
same-cell 256-pixel SSIM, normalized RGB MAE, 64-bit DCT perceptual hash,
owned-face-crop SSIM, and OpenAI CLIP ViT-L/14-336 image cosine. CL45 is missing
two Comet image assets at step zero only; its 96-row step-zero metric table and
all 96 selected/final images are present. No missing cell enters a selected
comparison. `[measured][caveat]`

## 1.3 Step-zero behavior is mechanism evidence

| Run | Step-zero ID | Delta vs CL27-0 | Exact ID-vector cells | Interpretation |
|---|---:|---:|---:|---|
| CL27 | `0.464640` | — | 96/96 | base |
| CL38 | `0.464640` | `0` | 96/96 | training-only change |
| **CL39** | **`0.503182`** | **`+0.038542`** | **0/96** | router changes inference immediately |
| CL40 | `0.464640` | `0` | 96/96 | zero-start learned projector |
| CL41 | `0.440728` | `-0.023911` | 12/96 | canonicalization changes eligible cells |
| CL42 | `0.464742` | `+0.000102` | 12/96 | component memory is active but nearly neutral initially |
| CL43 | `0.464640` | `0` | 96/96 | zero-start learned modulation |
| CL44 | `0.471748` | `+0.007109` | 0/96 | window changes inference immediately |
| CL45 | `0.464640` | `0` | 96/96 | optimization-only change |

CL39's favorable initialization is part of its parameter-free architecture,
not a checkpoint or panel leak. It means the later CL39-versus-CL27 difference
is an architecture-plus-training result rather than a pure training
difference-in-differences result. `[measured][caveat]`

# 2. Quantitative results

## 2.1 Identity trajectories and matched decisions

![](assets/cl38_cl45_20260821/id_sim_results.png){ width=96% }

*Figure 1. Left: complete subject-v2 trajectories. PhotoMaker is the dashed
line. Right: each arm's selected-checkpoint paired difference against CL27 at
the same optimizer step; bars are fixed-cell 95% bootstrap intervals.*

| Arm | Selected ID @ step | Final 24k ID | Delta vs CL27 | Wins | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| CL38 | `0.538993 @18k` | `0.538303` | `-0.006328` | 38/96 | `[-0.012017,-0.000695]` |
| **CL39** | **`0.570124 @16k`** | **`0.566342`** | **`+0.022864`** | **70/96** | **`[+0.012937,+0.032875]`** |
| CL40 | `0.541975 @20k` | `0.540369` | `+0.001662` | 44/96 | `[-0.005346,+0.009085]` |
| CL41 | `0.534795 @16k` | `0.529705` | `-0.012465` | 30/96 | `[-0.018653,-0.006268]` |
| CL42 | `0.544544 @16k` | `0.544262` | `-0.002716` | 50/96 | `[-0.007468,+0.001864]` |
| CL43 | `0.541810 @22k` | `0.540837` | `-0.000084` | 46/96 | `[-0.008730,+0.008666]` |
| **CL44** | **`0.550846 @22k`** | **`0.550284`** | **`+0.008952`** | **61/96** | **`[+0.002135,+0.015999]`** |
| CL45 | `0.537525 @18k` | `0.535199` | `-0.007796` | 36/96 | `[-0.012579,-0.003168]` |

Decision labels are: CL39 provisional winner; CL44 secondary positive;
CL40/CL42/CL43 neutral; and CL38/CL41/CL45 reject.

CL39 is the only selected arm with a controlled PhotoMaker win:
`+0.013544`, interval `[+0.004501,+0.023137]`. CL44's selected point estimate
remains `-0.005734` versus PhotoMaker and its interval crosses zero. CL38,
CL40, CL41, CL43, and CL45 are clearly below PhotoMaker; CL42's interval just
crosses zero. `[measured][paired]`

At the non-selected 24k endpoint, CL39 still beats both controls:

| Endpoint | ID_SIM | Paired delta | Wins | 95% interval |
|---|---:|---:|---:|---:|
| CL39-24k vs PhotoMaker | `0.566342` | `+0.009762` | 59/96 | `[+0.000636,+0.019052]` |
| CL39-24k vs CL27-24k | `0.566342` vs `0.543081` | `+0.023261` | 68/96 | `[+0.015020,+0.031526]` |
| CL44-24k vs CL27-24k | `0.550284` vs `0.543081` | `+0.007203` | 63/96 | `[+0.000497,+0.014022]` |

This endpoint result makes random checkpoint selection an implausible
explanation for CL39's lead, although a second training seed is still absent.
`[measured][limitation]`

## 2.2 Hard-case and identity slices

| Run | Skiing | Crying | Jumping | Dancing | Marion, all prompts |
|---|---:|---:|---:|---:|---:|
| PhotoMaker | `0.4640` | `0.6000` | `0.4173` | `0.4487` | `0.5029` |
| CL27 | `0.4337` | `0.5855` | `0.3946` | `0.4422` | `0.4935` |
| CL38 | `0.4074` | `0.5679` | `0.3929` | `0.4162` | `0.4841` |
| **CL39** | **`0.4911`** | **`0.5995`** | **`0.4437`** | **`0.4617`** | **`0.5161`** |
| CL40 | `0.4407` | `0.5674` | `0.4101` | `0.4538` | `0.4847` |
| CL41 | `0.4145` | `0.5607` | `0.4162` | `0.4353` | `0.4674` |
| CL42 | `0.4302` | `0.5700` | `0.4106` | `0.4373` | `0.4945` |
| CL43 | `0.4479` | `0.5768` | `0.4071` | `0.4390` | `0.4280` |
| CL44 | `0.4534` | `0.5829` | `0.4017` | `0.4499` | `0.5010` |
| CL45 | `0.4108` | `0.5700` | `0.4106` | `0.4405` | `0.4672` |

CL39 is the only arm that improves every displayed slice over CL27. It exceeds
PhotoMaker on Skiing, Jumping, Dancing, and Marion while essentially matching
Crying. CL44 improves Skiing and Marion means relative to CL27 but does not
resolve the topology cells. CL43's `0.4280` Marion mean is a material
identity-specific regression despite a neutral aggregate matched comparison.
`[measured]`

## 2.3 Ownership, prompt adherence, and quality

| Metric | PhotoMaker | CL27-16k | CL39-16k | CL44-22k |
|---|---:|---:|---:|---:|
| subject-v2 ID_SIM | `0.556580` | `0.547260` | **`0.570124`** | `0.550846` |
| legacy best-face ID | `0.501431` | `0.480837` | **`0.503554`** | `0.490022` |
| text similarity | `26.0015` | **`26.2432`** | `26.0371` | `26.2178` |
| mask IoU | `0.8652` | `0.9211` | **`0.9269`** | `0.9249` |
| mean detected faces | `1.135` | `1.125` | **`1.094`** | `1.104` |
| TOPIQ-Face mean | **`0.7532`** | `0.7142` | `0.7399` | `0.7148` |
| MUSIQ mean | **`73.0988`** | `71.9271` | `73.0433` | `71.9961` |
| MANIQA mean | **`0.6437`** | `0.6277` | `0.6412` | `0.6269` |

Every selected panel has face-detection rate `1.0` and zero no-face, unowned,
or ambiguous subject-v2 cells. CL39's gain is not created by selecting an
unintended extra face: it has the lowest mean face count, the best mask IoU,
and also edges PhotoMaker on the independent legacy-best diagnostic. Its text
score is `0.206` below CL27 but remains close to PhotoMaker; without per-cell
text scores, the significance of that small difference is not established.
`[measured][caveat]`

CL39 also recovers most of PhotoMaker's quality advantage over CL27. Its
TOPIQ-Face remains `0.0133` below PhotoMaker, but MUSIQ differs by only
`0.0555` and MANIQA by `0.0025`. `[measured]`

\newpage

# 3. Visual inspection

The fixed rubric is:

- **pass:** prompted top object and identity-defining ordinary eyewear are
  present, object/face order is readable, and the intended face is attached;
- **minor:** readable ownership and order with a localized asymmetry;
- **fail:** fused or duplicated layers, important object deletion, unreadable
  face, or wrong face/body association.

Counts are one unblinded review of one fixed seed. They diagnose cells rather
than estimate a population rate. `[method][limitation]`

## 3.1 Skiing: CL39 fixes the panel without deleting eyewear

![](assets/cl38_cl45_20260821/skiing_face_a.jpg){ width=92% }

*Figure 2. Skiing owned-face crops: PhotoMaker, CL27, CL38, CL39, and CL40.
Values are subject-v2 ID and mask IoU.*

![](assets/cl38_cl45_20260821/skiing_face_b.jpg){ width=92% }

*Figure 3. Skiing owned-face crops: CL27 and CL41-CL45.*

| Run | Pass | Minor | Fail |
|---|---:|---:|---:|
| PhotoMaker | 8 | 0 | 0 |
| CL27-16k | 6 | 1 | 1 |
| **CL39-16k** | **8** | **0** | **0** |
| CL44-22k | 6 | 0 | 2 |

CL39 preserves the intended two-layer eyewear structure for Eddie, Elon,
Jensen, Keanu, and Lex: large ski goggles remain above distinct ordinary
glasses. Jennie and Jisoo retain clean large-goggle/face boundaries. Most
importantly, Marion's goggles sit above readable eyes rather than replacing
the eye region. This differs from CL33's earlier score shortcut, where Elon's
ordinary glasses disappeared. `[visual]`

CL44's ID gains do not imply a full visual solution. Lex's ordinary eyewear is
not readable as a distinct layer, and Marion's large goggles obscure the eye
region. CL38's direct anti-deletion objective performs worse than CL27 in both
mean Skiing ID and the reviewed cells. `[measured][visual]`

## 3.2 Crying remains solved

![](assets/cl38_cl45_20260821/crying_face_a.jpg){ width=92% }

*Figure 4. Crying comparison for PhotoMaker, CL27, CL38, CL39, and CL40.*

![](assets/cl38_cl45_20260821/crying_face_b.jpg){ width=92% }

*Figure 5. Crying comparison for CL27 and CL41-CL45.*

PhotoMaker, CL27, CL39, and CL44 are all `8/0/0`: hands and hair remain
separate from readable faces. CL39's `0.5995` mean is almost exactly
PhotoMaker's `0.6000`. The largest CL39 regression within this slice is Marion
(`0.543` versus CL27 `0.591`), but it is an identity-strength regression, not
a hand/face topology failure. `[measured][visual]`

## 3.3 Small/action faces improve under CL39

![](assets/cl38_cl45_20260821/jumping_face_a.jpg){ width=92% }

*Figure 6. Jumping crops for PhotoMaker, CL27, CL38, CL39, and CL40. The full
generated faces are small; crops use the fixed owned-face box.*

CL39 gives the strongest Jumping mean (`0.4437`) and clear fixed-cell gains for
Jennie (`0.509` versus CL27 `0.374`), Jisoo (`0.513` versus `0.393`), and Lex
(`0.373` versus `0.302`). The body poses and global compositions remain
prompt-consistent; facial details change. Keanu is neutral (`0.346` in both
CL39 and CL27), so the gain is not universal. `[measured][visual]`

![](assets/cl38_cl45_20260821/hardcase_differentiators.jpg){ width=100% }

*Figure 7. Full image plus owned-face crop for the most important Skiing,
Jumping, and Marion/Crying cells across PhotoMaker, CL27, CL39, CL44, and
CL42.*

## 3.4 CL39 still has meaningful regressions

![](assets/cl38_cl45_20260821/largest_gains_regressions.jpg){ width=100% }

*Figure 8. Largest useful gains and regressions. CL39 improves Lex/Chef,
Eddie/Crying, and Keanu/Dancing, but regresses Lex/Dancing and Eddie/Reading.*

CL39 wins `70/96` cells against CL27, not all 96. Its largest losses are
Eddie/Reading (`-0.0991`), Lex/Dancing (`-0.0943`), Jensen/Dancing (`-0.0723`),
and Eddie/Skiing (`-0.0619`). Lex/Dancing is visibly less identity-specific.
These regressions argue for retaining per-cell promotion gates rather than
using only the mean. `[measured][visual]`

\newpage

# 4. Are these PhotoMaker generations? Shortcut and fairness audit

## 4.1 Pixel answer: different images, shared deterministic composition

![](assets/cl38_cl45_20260821/photomaker_pixel_difference_examples.jpg){ width=82% }

*Figure 9. Same fixed cells from PhotoMaker, CL27, and CL39, plus amplified
absolute CL39-minus-PhotoMaker RGB differences. Differences cover the face,
clothing, object boundaries, and background.*

![](assets/cl38_cl45_20260821/photomaker_image_distance.png){ width=94% }

*Figure 10. Same-cell selected-image similarity to PhotoMaker. CL39 is closer
than every other arm but remains below exact identity (`1.0`).*

| Selected run | Exact RGB matches | RGB MAE | SSIM to PM | Face SSIM to PM | CLIP image cosine to PM |
|---|---:|---:|---:|---:|---:|
| CL27 | 0/96 | `0.04504` | `0.82095` | `0.76601` | `0.93393` |
| CL38 | 0/96 | `0.04546` | `0.81982` | `0.76610` | `0.93643` |
| **CL39** | **0/96** | **`0.04088`** | **`0.83814`** | **`0.79950`** | **`0.96363`** |
| CL40 | 0/96 | `0.04453` | `0.82206` | `0.76780` | `0.93889` |
| CL41 | 0/96 | `0.04608` | `0.81661` | `0.76403` | `0.93046` |
| CL42 | 0/96 | `0.04493` | `0.82213` | `0.76674` | `0.93371` |
| CL43 | 0/96 | `0.04547` | `0.81929` | `0.76531` | `0.92966` |
| CL44 | 0/96 | `0.04431` | `0.82616` | `0.77301` | `0.94171` |
| CL45 | 0/96 | `0.04418` | `0.82617` | `0.77200` | `0.93688` |

All models share the same SDXL/RealVis base, prompts, references, scheduler,
and seed. Similar global composition is therefore expected and is visible even
for CL27. CL39 is not pixel-close to exact replay: its average per-channel
difference is about `10.4/255`, its SSIM is far from `1`, and every decoded
hash differs. It is also structurally closer to CL27 than to PhotoMaker
(`0.89005` versus `0.83814` SSIM), despite being semantically closer to
PhotoMaker in CLIP space (`0.96363` versus CL27 `0.95379`). `[measured]`

## 4.2 The PhotoMaker pull is real and causal at step zero

Relative to CL27's own distance from PhotoMaker, CL39 is closer by:

| Metric | Mean paired closeness gain | Cells closer | 95% interval |
|---|---:|---:|---:|
| full-image SSIM | `+0.017185` | 80/96 | `[+0.011397,+0.022390]` |
| CLIP image cosine | `+0.029700` | 81/96 | `[+0.023257,+0.036481]` |
| face-crop SSIM | `+0.033484` | 86/96 | `[+0.025319,+0.041220]` |

At step zero, before any CL39 optimizer update, the corresponding gains are
`+0.012680` SSIM in 90/96 cells, `+0.048941` CLIP cosine in 91/96, and
`+0.038885` face SSIM in 95/96. This is direct interventional evidence that
the null router pulls the CL27 initialization toward the PhotoMaker-like
native path. Training is not needed for that direction of movement.
`[measured][paired]`

This does not mean perceptual PhotoMaker closeness mechanically causes a high
ID score. CL45 is also modestly closer than CL27 in SSIM but is significantly
worse in ID. CL39 combines the retreat with a strong trained result and clean
hard-case topology. `[measured][inference]`

## 4.3 Code-path audit

The current source and sealed run configuration establish the following:

1. **No stored PhotoMaker generation enters CL39.** The CL39 YAML enables only
   the parameter-free null-key router on top of CL27. No PhotoMaker output
   image path, ID_SIM loss, ArcFace loss, or native-PhotoMaker distillation is
   enabled. The separate PhotoMaker boundary teacher is defaults-off and was
   enabled only by historical CL24. `[code]`
2. **The explicit BA topology remains present.** `_full_target_lanes()` builds
   target queries from target features, a native target-K/V message, and a
   target-Q/reference-KV message. `pose_adapt_ratio=0`,
   `ca_mixing_for_face=false`, and branched CA is disabled; no target K/V is
   substituted into the reference lane. `[code]`
3. **CL39 attenuates rather than replaces BA.** It computes normalized
   reference-attention entropy, maps it to a confidence bounded to
   `[0.25,1.0]`, multiplies both low- and high-frequency reference-minus-native
   components by that confidence, and returns `native_out + routed_delta`.
   The recorded median reference fraction is `0.315588` and the last value is
   `0.318337`. `[code][measured]`
4. **The fallback is PhotoMaker-effective, not an unrelated frozen UNet.** The
   branch Q/K/V linears are cloned from effective attention projections, which
   include the loaded PhotoMaker `default` adapter at initialization. The
   configuration also co-trains 127.8M BA parameters, 30.5M generic-adapter
   parameters, and 60.9M PhotoMaker-default parameters. `[code]`
5. **The controlled PhotoMaker comparison is an end-to-end benchmark, not a
   BA-only causal ablation.** PhotoMaker step zero uses the same panel with BA
   disabled and the pretrained default adapter; CL39 has 24k steps of joint
   BA/generic/default-adapter training. The fixed protocol makes the generated
   systems comparable, but the parameter ownership prevents attributing the
   whole surplus to BA. `[code][caveat]`

## 4.4 Verdict on fairness

| Question | Answer | Evidence |
|---|---|---|
| Are CL39 images literally PhotoMaker images? | **No.** | 0/96 exact decoded matches; material RGB/SSIM differences. |
| Is the validation protocol controlled? | **Yes.** | Same fixed panel, references, seeds, boxes, base, DDIM50, CFG, and metrics. |
| Is BA switched off or replaced by target K/V? | **No.** | Target-Q/reference-KV remains; confidence has a 0.25 floor; pose/CA mixing are off. |
| Does CL39 lean strongly on the native/PhotoMaker path? | **Yes.** | Median reference fraction `0.3156`; step-zero output and perceptual shift toward PhotoMaker. |
| Is the high ID score a face-selection artifact? | **No evidence of one.** | Zero unowned/no-face cells, best mask IoU, fewer extra faces, legacy ID also improves. |
| Does current evidence prove BA causes the win over PhotoMaker? | **No; not established.** | Generic/default adapters are co-trained and no same-checkpoint BA-off counterfactual exists. |

CL39 is fair as an **end-to-end architecture experiment** and does not leak
PhotoMaker outputs. It is not yet a clean demonstration that stronger explicit
reference-conditioned BA beats PhotoMaker, because its successful change is
precisely to abstain from most of the explicit BA residual. `[decision]`

# 5. What worked and what did not

## 5.1 What worked

### CL39: high-confidence abstention

The router is strongly active (`null_mass` median `0.91255`, reference
fraction median `0.31559`). It improves aggregate ID, the 24k endpoint,
PhotoMaker-relative ID, face quality, mask ownership, Skiing topology, Marion,
and small/action faces. Its likely benefit is selective avoidance of
low-correspondence reference transfer; the cost is reduced causal reliance on
the BA lane. `[measured][code][inference]`

### CL44: bounded high-frequency window

CL44 is a smaller but controlled positive result. The semantic-window high
scale has median `0.94733` and range `0.80458-1.12760`; the low band is
unchanged. It improves selected and 24k ID against matched CL27, suggesting
that high-frequency reference transfer benefits from mild time/agreement
calibration. It does not beat PhotoMaker or solve all Skiing topology, so it is
an architectural lead rather than a new base. `[measured][code]`

## 5.2 What did not work

- **CL38 corrected ownership anchor:** the auxiliary is active and finite, but
  selected ID is `-0.006328` versus matched CL27 and Skiing falls to `0.4074`.
  The corrected configuration is negative; the earlier collapsed r3 says
  nothing scientific about the intended loss. `[measured]`
- **CL40 identity-motion projector:** the gate reaches `0.35` and the
  correction is nonzero, yet the matched delta is neutral and TOPIQ-Face is
  the suite low (`0.6992`). This configuration supplies no benefit.
  `[measured]`
- **CL41 canonical K/V:** it is applied to about `79%` of logged samples and
  hurts ID immediately at step zero and after training. The fixed
  canonicalization appears to distort useful reference geometry.
  `[measured][inference]`
- **CL42 component memory:** it is also applied about `79%` of the time, but
  the global token receives roughly `72%` of memory mass. Its matched result
  is neutral and it does not improve the target hard slices over CL39.
  `[measured][inference]`
- **CL43 ID-adaptive modulation:** gamma/beta and output corrections are
  nonzero after the ramp, but aggregate change is neutral and Marion falls to
  `0.4280`. Do not promote this configuration. `[measured]`
- **CL45 PCGrad:** conflicts occur in a median `24%` of logged windows, so the
  arm is not a no-op, but the projection norm is tiny and selected/final ID are
  significantly worse than CL27. Asymmetric projection in this form does not
  prevent late drift. `[measured]`

These are conclusions about the tested configurations and single training
seed, not proofs that the broader mechanism families are impossible.
`[limitation]`

# 6. Confidence and remaining unknowns

| Claim | Confidence | Basis | Main limitation |
|---|---|---|---|
| CL39 is the fixed-panel ID winner | High | selected and 24k paired intervals vs both controls | one training seed |
| CL39 fixes fixed-panel Skiing topology | Moderate-high | 8/0/0 unblinded visual rubric and higher slice mean | one seed; 16k visual panel only |
| CL39 images are not PhotoMaker replays | High | decoded hashes, RGB differences, SSIM, CLIP, visual difference maps | metrics do not define philosophical novelty |
| CL39 moves toward PhotoMaker/native behavior | High | step-zero intervention, distance intervals, median 0.316 reference fraction, code path | exact contribution of each trained adapter unknown |
| CL39's BA lane causes the surplus over PhotoMaker | **Not established** | no same-checkpoint BA-off or shuffled-spatial-reference counterfactual | co-trained default/generic paths |
| CL39 generalizes to unseen identities/seeds/datasets | **Not established** | fixed celebrity panel only | no held-out identity suite or replicate |
| CL44 is a useful secondary mechanism | Moderate | selected and endpoint matched CL27 gains | below PhotoMaker; hard-case failures remain |

## What is not established

- The fraction of CL39's `+0.013544` PhotoMaker-relative gain caused by the
  explicit spatial BA lane.
- Whether a second CL39 training seed reproduces the mean and the `8/0/0`
  Skiing result.
- Whether CL39-24k retains the 16k visual topology; only its sealed per-image
  identity endpoint is analyzed here.
- Generalization to identities not represented by this training/validation
  regime, different seeds, different schedulers, or another dataset.
- Whether the `0.206` CL39-versus-CL27 text-score difference is meaningful;
  per-image text scores are unavailable.

# 7. Decision and required causal qualification

**Provisional deployment/base candidate:** CL39 r4 at 16k, immutable Comet key
`b1ca0b3da679401c85b991f1bbdf0b2a`.

**Research control:** retain CL27 r3 at 16k, key
`dbfbf40c3bdd4f70bedc58bda3dfb9cd`, until CL39 passes the following
evaluation-only matrix on the same checkpoint and panel:

1. **Actual CL39:** current confidence router.
2. **Router disabled:** force reference fraction 1.0 while loading the exact
   CL39 checkpoint; isolates the router's attenuation.
3. **BA disabled, trained adapters retained:** preserve CL39's trained
   PhotoMaker-default and generic adapters but bypass branched SA; measures the
   trained PhotoMaker/native shortcut directly.
4. **Spatial-reference shuffle:** keep correct PhotoMaker identity tokens and
   reference embedding, but shuffle only spatial BA reference latents; tests
   whether CL39 outputs causally use the explicit spatial reference lane.

Report ID_SIM, text similarity, face quality, the same pixel-distance table,
and Skiing/Crying topology for all four. A successful BA claim requires actual
CL39 to beat the trained-adapter BA-off control and to degrade meaningfully
under spatial-reference shuffle without object deletion. Until then, describe
CL39 as the best end-to-end result with **provisional BA attribution**.
`[decision]`

# 8. Reproduction

Run from `diffusion_template/` in the existing `photomaker` environment:

```bash
set -a
source .env
set +a

/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  tools/comet/export_comet_runs.py \
  --manifest analysis/assets/cl38_cl45_20260821/comet_manifest_step0.json \
  --output-dir /tmp/cl38_cl45_20260821/step0 \
  --output-json /tmp/cl38_cl45_20260821/step0/comet_runs_export.json \
  --timeout 120

/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  tools/comet/export_comet_runs.py \
  --manifest analysis/assets/cl38_cl45_20260821/comet_manifest_selected.json \
  --output-dir /tmp/cl38_cl45_20260821/selected \
  --output-json /tmp/cl38_cl45_20260821/selected/comet_runs_export.json \
  --timeout 120

/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  analysis/assets/cl38_cl45_20260821/audit_comet_completion.py

/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  analysis/assets/cl38_cl45_20260821/build_analysis_assets.py

/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  analysis/assets/cl38_cl45_20260821/compute_image_distances.py
```

Render, verify, and upload under the repository report contract:

```bash
/home/kolyangg/anaconda3/envs/photomaker/bin/python \
  tools/reports/publish_report.py \
  analysis/2026-08-21_cl38_cl45_completed_results_and_photomaker_shortcut_audit.md \
  --upload
```

The raw multi-gigabyte Comet image cache is reconstructible from the immutable
keys and manifests and may be deleted after the checked figures, tables, PDF,
and SHA-256 inventory are sealed.

# 9. Internal evidence

- `docs/handoffs/LATEST.md`
- `analysis/2026-08-20_cl38_visibility_ownership_gradient_failure_and_recovery.md`
- `analysis/2026-08-19_cl27_next_eight_architecture_experiments.md`
- `analysis/2026-08-19_cl30_cl37_completed_results_and_base_decision.md`
- `analysis/2026-08-17_cl27_cl29_vs_cl23_visual_results_and_next_experiments.md`
- `src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml`
- `src/configs/CL38_cosmic_visibility_ownership_v2_24k.yaml` through
  `src/configs/CL45_cosmic_ba_pcgrad_24k.yaml`
- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2.py`
- `src/configs/PM0_original_photomaker_CL19_full96.yaml`
