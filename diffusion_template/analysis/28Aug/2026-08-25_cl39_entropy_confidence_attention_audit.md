---
title: "CL39's entropy gate attenuates—but does not remove—the branched-attention correction"
subtitle: "A 16-sample, same-checkpoint audit of confidence, reference attention, frequency routing, and image-space counterfactuals"
date: "25 August 2026"
---

# CL39's entropy gate attenuates—but does not remove—the branched-attention correction

**Date:** 25 August 2026  
**Scope:** evaluation-only instrumentation of the canonical CL39 r4 endpoint
through the historical trainer/YAML validation path on Serv A100s. The actual,
`C=1`, correction-zero, and routed-raw-`R` arms each ran the complete fixed
96-image panel at the original batch size 12; compact telemetry was retained
for 16 deterministic cells. No training, checkpoint, validation prompt, seed,
reference, face box, scheduler, inference-step, or metric definition was
changed.  
**Evidence cutoff:** canonical Serv/Comet artifacts and the exact Serv replay
available on 25 August 2026. Neb was not contacted.  
**Primary mechanism measurements:** routed-face confidence `C`, effective
low/high weights, applied correction magnitude relative to native target
self-attention, and same-checkpoint image changes when forcing `C=1` or setting
the explicit BA correction to zero. Identity similarity is secondary because it
does not by itself establish whether the spatial BA route was used.

## Executive conclusion

CL39 is neither an inactive BA route nor an unattenuated reference route. On
the routed target face, mean confidence is `0.495`; after the learned target
router and temporal schedules, the effective low/high scalar weights are
`0.248 / 0.366`. The final explicit correction has mean pointwise magnitude
`0.221×` native target self-attention. [measured]

Setting that correction to zero changes `60.7%` of pixels by more than `1/255`
and lowers subject-v2 identity similarity from `0.5575` to `0.5193`.
Actual CL39 wins `15/16` cells and loses one; the paired mean gain is `+0.0383`
with bootstrap 95% interval `[+0.0209,+0.0577]`. The explicit `(R-N)` lane
therefore causally contributes to identity on this Serv panel—it is not merely
computed and then ignored. [measured]

Forcing `C=1` gives mean identity `0.4998`, `0.0577` below actual. Actual wins
`14/16` cells and loses two; the paired bootstrap interval
`[+0.0343,+0.0795]` excludes zero. Entropy attenuation therefore improves this
selected Serv panel strongly, although two samples still prefer `C=1`.
[measured]

Both frequency bands are live: `D_low + D_high` reconstructs `R-N` to `0.124%`
relative error, the high band contributes `37.2%` of the summed applied-band
magnitudes, and its magnitude map has `3.82×` the normalized spatial variation
of the low band. [measured] The instrumented actual arm is a strict replay of
the sealed Serv panel: all 96 outputs pass the image gate, with zero RGB error.
The counterfactual comparisons therefore share the canonical A100 execution
path rather than relying on the earlier MHZ trajectory. [measured]

A direct branch-face intervention also makes the visual role of `R` and `N`
concrete. Replacing `N` by raw `R` inside CL39's existing soft face router
changes `76.6%` of all pixels and `95.2%` of fixed-face-crop pixels relative to
the N-only arm. Raw routed-`R` frequently duplicates or warps eyes, noses,
glasses, and expressions; actual CL39 remains close to the stable `N` anchor
because it applies only the confidence-, schedule-, and router-weighted
`R-N` correction. [measured] [visual]

| Question | Result | Evidence |
|---|---|---|
| What is `C`? | A detached inverse-entropy reference fraction, not a learned correctness probability or literal null-key probability. | [code] |
| Is the entropy gate active? | Yes. Routed-face `C` is continuous (`0.397 / 0.477 / 0.542` mean p10/p50/p90), not pinned to either endpoint. | [measured] |
| Is `(R-N)` actually used? | Yes. Correction/native is `0.221`; BA-off changes `60.7%` of pixels and lowers ID by `0.0383`. | [measured] |
| Is CL39 in practice an exact PhotoMaker fallback? | No. The native/PhotoMaker-effective lane remains the anchor, but zeroing BA materially changes outputs and identity. BA-off is also not the separately trained PM0 model. | [measured] [code] |
| How do the `N` and `R` face branches look? | The N-only trajectory is stable; raw `R` routed onto the face often introduces large structural distortions. Four overview grids show all 16 cells, with signed `R-on-face-N` panels. | [measured] [visual] |
| Do `D_low` and `D_high` both operate? | Yes. Both are nonzero; high supplies `37.2%` of summed band magnitude and is `3.82×` rougher after magnitude normalization. | [measured] [code] |
| Is the instrumented run a pixel replay of sealed Serv? | Yes. All 96 actual outputs pass the exact replay gate; RGB MAE and max error are zero. | [measured] |

## 1. Exact run and comparability boundary

| Property | Sealed value |
|---|---|
| Run | `CL39_cosmic_null_key_confidence_router_24k_full96_r4` |
| Immutable Comet key | `b1ca0b3da679401c85b991f1bbdf0b2a` |
| Endpoint | epoch 12 / optimizer step `24,000` |
| Checkpoint SHA-256 | `74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07` |
| Checkpoint bytes | `1,318,771,270` |
| Training base | `stabilityai/stable-diffusion-xl-base-1.0` |
| Validation base | `SG161222/RealVisXL_V4.0` |
| Processor transfer | `legacy_full_copy`, strict state copied for 70 processors |
| CL39 processors | 36 null-key-router processors in `up_blocks.0/1` |
| Validation | `1024²`, DDIM 50, CFG 5, seed 0, batch 12, one image per item |
| Identity schedule | PhotoMaker from step 10; BA from step 15 |
| Reference ownership | `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, branched CA disabled |
| Trainable contract | schema-v2, 2,240 tensors / 219,217,920 parameters |

| Serv validation-only job | MLS job ID |
|---|---|
| 12-image sealed replay smoke | `lm-mpi-job-d24ac1da-8cef-47ea-b1b9-acbcfe28b06b` |
| full-96 actual + telemetry | `lm-mpi-job-79af3dd2-f662-48bc-ac83-c18adc33d490` |
| full-96 forced `C=1` | `lm-mpi-job-ba15e767-18cb-4000-aa8a-8b1613c683a1` |
| full-96 correction `=0` | `lm-mpi-job-01f46a9a-d2cb-4f4b-b279-0842bf1a2718` |
| assemble, metrics, and final render | `lm-mpi-job-20808f03-0104-44dd-a0ae-b21901ccdc49` |
| full-96 raw `R` routed on face + branch render | `lm-mpi-job-1d94aa46-9197-466a-8f22-1abcce4e4312` |

The checkpoint, sealed `config.yaml`, immutable Comet record, and historical
runtime source were read from Serv. The checkpoint is also preserved locally
under
`artifacts/checkpoints/CL39_cosmic_null_key_confidence_router_24k_full96_r4/`;
its size and SHA-256 match Serv exactly. [measured]

The publication replay used the ordinary `train.py` validation-only path with
the original batch size 12. A fail-closed 12-image smoke first matched the
sealed PNGs byte-for-byte. The final actual arm then generated all 96 cells and
was compared with the sealed panel before the counterfactual evidence was
accepted. The endpoint's authoritative fixed-96 subject-v2 score remains
`0.566342`; the selected 16 actual cells are drawn directly from that exact
trajectory. [measured]

## 2. What CL39 confidence actually is

For a target query, CL39 computes reference-lane attention probabilities over
the complete reference token grid, averages normalized entropy across heads,
and applies a fixed monotone mapping:

```text
N       = target-Q / target-KV attention output
R       = target-Q / reference-KV attention output
D       = R - N
D_low   = fixed 5×5 binomial-Gaussian filter(D)
D_high  = D - D_low

H       = -sum_k p_k log(p_k) / log(number of reference tokens)
null    = sigmoid((H - 0.75) / 0.08)
C       = clamp(1 - 0.75 * null, 0.25, 1.00)

Y       = N + router * C * (s_low(p) D_low + s_high(p) D_high)
s_low   = 0.50 + p(0.85 - 0.50)
s_high  = 0.75 + p(1.25 - 0.75)
```

`C` is detached, parameter-free, and query-local. It is best interpreted as a
bounded inverse-entropy **reference fraction**, not a calibrated probability
that identity transfer is correct. Despite the internal `null_mass` name, the
implementation does not append a learned null key: `null` is the sigmoid of
entropy itself. [code]

| Normalized entropy `H` | `C` from the fixed mapping |
|---:|---:|
| `0.00` | `0.9999` |
| `0.50` | `0.9684` |
| `0.65` | `0.8328` |
| `0.75` | `0.6250` |
| `0.85` | `0.4172` |
| `1.00` | `0.2816` |

Because normalized entropy is mathematically in `[0,1]`, the nominal
`clamp(..., 0.25, 1.00)` is not reached at either end under finite inputs. The
actual formula range is approximately `[0.2816, 0.9999]`; the configured
`0.25` minimum is a safety bound rather than the attainable operating floor.
[code]

There is an important interpretation limit. CL39 zeros reference hidden states
outside the sealed reference-face mask before projecting reference K/V, but it
does **not** exclude those locations from the softmax support or from entropy.
Masked positions remain in the denominator, and projection bias can make their
K/V nonzero. Thus current `H` mixes correspondence ambiguity with reference
mask area and invalid-key dilution. [code]

## 3. Audit design

### 3.1 Deterministic 16-cell panel

The analysis selects two items per identity from the fixed 96-image
`manual_val` panel using Python RNG seed `390024`. The resulting validation
indices are:

```text
01, 07, 13, 17, 33, 35, 38, 40, 51, 55, 63, 69, 78, 80, 87, 93
```

This is a deterministic stratified-random subset, not a hand-picked best/worst
gallery. It covers all eight identities and ten prompt/action cells.
[measured]

Generated face boxes are joined with the validation dataset's literal
space-bearing `prompt[:10]_{identity}.png` key. Underscore normalization is
used only for exported Comet PNG/table joins. Keeping those two joins separate
avoids silently dropping or mismatching validation boxes. [code]

### 3.2 Four same-checkpoint arms

| Arm | Single inference intervention | Meaning |
|---|---|---|
| Actual | none | ordinary CL39 |
| `C=1` | replace only applied `C` with one | isolates entropy attenuation while retaining router, `R-N`, schedules, and trained weights |
| correction `=0` | multiply the final explicit routed BA correction by zero | native/PhotoMaker-effective fallback with CL39's trained generic/default adapters retained |
| raw `R` on face | set target attention to `N + router·(R-N)` at every shipped CL39 processor | replaces `N` by the unattenuated reference-attention output inside the existing soft face router; `N` remains outside |

Every arm runs all 96 items in the same 12-item batches and uses the same
prompt, reference, face boxes, seed, RealVisXL base, DDIM schedule, 50 steps,
and CFG. The zero-correction arm is not the separately trained PM0 experiment;
it is the causal same-checkpoint fallback requested by the earlier shortcut
audit. The raw-`R` arm is an intentionally extreme evaluation intervention,
not a trained operating point or a proposed replacement for CL39. [code]

### 3.3 Capture semantics

The capture is opt-in and attaches dynamic analysis attributes only; it adds no
checkpoint key or trainable. Full `L×L` attention tensors are never persisted.
Attention probabilities are reduced across heads and query/key axes inside each
processor, channel magnitudes are reduced to per-query RMS, and maps are resized
to `64×64` before CPU aggregation by denoising progress and `up0`/`up1`.
[code]

Capture receives the complete batch so classifier-free conditional rows are
split correctly, but it retains only the 16 declared sample indices. The
instrumented actual images are replay-gated against sealed Serv outputs, making
the non-perturbing nature of the hooks directly testable rather than assumed.
[measured] [code]

The reported face values are weighted by CL39's actual soft target router.
There are `20,160` instrumented processor calls across the 16 samples.

## 4. The confidence route is active

| Routed-face quantity, all layers/active steps | Panel mean |
|---|---:|
| normalized entropy | `0.8213` |
| confidence `C`, all target queries | `0.3251` |
| confidence `C`, router-weighted face | `0.4949` |
| mean target-router mass over the full grid | `3.039%` |
| mean layerwise C p10 / p50 / p90 | `0.3967 / 0.4773 / 0.5416` |
| queries at `C≈0.25` floor | `0.000%` |
| queries at `C≈1` | `0.000%` |
| effective low weight `router·C·s_low` | `0.2479` |
| effective high weight `router·C·s_high` | `0.3663` |

The gate operates in its smooth interior. It does not make a binary
reference/native decision: the measured routed-face values largely occupy
about `0.3–0.7`. Face queries have higher
confidence than the all-query mean (`0.495` versus `0.325`), which reconciles
this audit with the lower all-grid reference fractions reported in historical
Comet telemetry. [measured]

Confidence rises mildly through the BA-active denoising interval while
correction/native grows from roughly `0.17` to `0.26`. The effective band
weights remain far below their configured schedules because the target router
and `C` both attenuate them. [measured] [visual]

![Measured entropy-confidence calibration and routed-cell distribution](assets/cl39_attention_24k_serv_a100/cl39_entropy_confidence_calibration.png)

The cyan curve is the exact per-query mapping. The hexbin points are compact
maps averaged over processors within each progress/group cell; because `C(H)`
is nonlinear, averaging `H` and `C` separately does not leave those aggregate
points exactly on the curve. [code]

![Confidence and correction over real denoising progress](assets/cl39_attention_24k_serv_a100/cl39_temporal_mechanism.png)

The schedule plot distinguishes configured band scales from weights actually
applied after the target router and confidence. `p` is real denoising progress,
not the ordinal count among BA-active steps. [measured] [code]

![Layer-group and denoising-step heatmaps](assets/cl39_attention_24k_serv_a100/cl39_layer_step_heatmaps.png)

## 5. `(R-N)` materially affects CL39 outputs

### 5.1 Activation-space evidence

| Routed-face magnitude | Panel mean |
|---|---:|
| raw `|R-N|` | `0.4534` |
| applied correction | `0.1194` |
| applied correction / native `|N|` | `0.2210` |
| applied `D_low` | `0.0927` |
| applied `D_high` | `0.0599` |

The raw reference-minus-native message is substantial, but CL39 applies only a
bounded, face-routed fraction. Native target self-attention therefore remains
the dominant anchor while the explicit reference lane supplies a nontrivial
correction. This is attenuation, not route deletion. [measured] [code]

### 5.2 Image-space counterfactuals

| Same-checkpoint comparison, `n=16` | RGB MAE | face-box RGB MAE | SSIM | pixels changed >1/255 |
|---|---:|---:|---:|---:|
| actual vs `C=1` | `0.01433` | `0.07027` | `0.93914` | `64.86%` |
| actual vs correction `=0` | `0.01157` | `0.04568` | `0.95019` | `60.71%` |
| raw `R` on face vs N-only | `0.02163` | `0.07253` | — | `76.60%` |

Neither counterfactual is an exact or negligible perturbation of actual CL39. Forced `C=1` changes
slightly more than zero correction on average, consistent with moving from
mean face `C≈0.49` upward by about `0.51` versus downward by about `0.49`.
Diffusion makes this relationship nonlinear, so those image distances should
not be expected to scale exactly with gate distance. [measured] [inference]

![Distribution of same-checkpoint counterfactual effects](assets/cl39_attention_24k_serv_a100/cl39_counterfactual_distributions.png)

The difference maps are face-centered but not confined to the face rectangle:
changing hidden states during denoising propagates through later U-Net layers
and steps. This is expected causal propagation, not evidence that the router
itself is spatially global. [visual] [code]

### 5.3 Direct branch-face comparison: `N`, routed `R`, and their difference

The following grids use the fixed validation face crop for every arm. `N-only`
is the existing correction-zero trajectory. `R-on-face` is a new Serv arm that
sets each shipped CL39 processor to
`N + soft_face_router·(R-N)`. The final column is the signed **image-space**
`R-on-face − N-only` RGB difference at fixed `4×` gain, where neutral gray is
zero. It is not an attention magnitude map. [code]

![Direct N, routed-R, and signed difference faces, cells 01–17](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_1.png)

![Direct N, routed-R, and signed difference faces, cells 33–40](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_2.png)

![Direct N, routed-R, and signed difference faces, cells 51–69](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_3.png)

![Direct N, routed-R, and signed difference faces, cells 78–93](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_4.png)

Across the 16 cells, raw routed-`R` versus N-only has mean full-image RGB MAE
`0.02163` and mean fixed-face-crop RGB MAE `0.07253`. Pixels changed above
`1/255` average `76.60%` globally and `95.24%` within the face crop. Every cell
changes at least `88.10%` of face-crop pixels; face MAE ranges from `0.03815`
(index 35) to `0.11643` (index 40). [measured]

Visually, `R` is carrying identity- and face-structured information: the signed
residual traces eyes, brows, nose, mouth, hairline, eyewear, and face outline.
But raw `R` is not a clean decoded identity face. It frequently creates doubled
or misregistered facial parts, most visibly for Jensen indices 38/40 and for
occluded/expressive cells 07, 55, and 80. This directly supports CL39's design
choice to keep `N` as the anchor and use a bounded `R-N` correction rather than
substituting raw reference attention. [visual] [inference]

![Detailed N versus routed-R branch view, Eddie index 01](assets/cl39_attention_24k_serv_a100_branch_faces/branch_samples/01_eddie.png)

![Detailed N versus routed-R branch view, Jensen index 38](assets/cl39_attention_24k_serv_a100_branch_faces/branch_samples/38_jensen.png)

Interpretation limit: `R` and `N` are intermediate attention outputs, not RGB
latents. The `R-on-face` image is therefore a controlled whole-denoising
intervention applied at all CL39 layers and active steps; it is not a direct VAE
decode of a single `R` tensor. The signed final-image panel includes downstream
diffusion propagation and must not be read as the raw feature tensor itself.
[code] [caveat]

### 5.4 Subject-v2 identity similarity

| 16-cell output set | Mean subject-v2 ID similarity |
|---|---:|
| sealed Serv actual | `0.557538` |
| replayed Serv actual | `0.557538` |
| Serv forced `C=1` | `0.499840` |
| Serv correction `=0` | `0.519252` |

Actual-minus-BA-off is `+0.03829` (median `+0.03091`; `15 wins / 0 ties /
1 loss`; deterministic paired-bootstrap 95% interval
`[+0.02093,+0.05774]`). This is strong Serv evidence that the explicit BA lane
contributes useful identity rather than merely perturbing pixels.
[measured]

Actual-minus-`C=1` is `+0.05770` (`14 / 0 / 2`) and its paired interval
`[+0.03432,+0.07949]` excludes zero. The entropy gate improves the selected
Serv mean decisively, driven by broad gains rather than one outlier, although
two cells still prefer the unattenuated route. [measured]

All 16 selected cells are owned under all three Serv arms. No missing/unowned
face was dropped from these means. [measured]

Identity deltas are secondary causal evidence. A route can materially alter the
image without improving this metric, and the 16-cell counterfactual subset is
not a replacement for a scored fixed-96 counterfactual panel. [caveat]

| Index | Identity / action | C | correction / N | ID actual | ID BA-off | Δ actual−off | Δ actual−C=1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | eddie / Rushing | 0.468 | 0.225 | 0.632 | 0.573 | +0.059 | +0.002 |
| 07 | eddie / Crying | 0.451 | 0.228 | 0.656 | 0.557 | +0.099 | +0.130 |
| 13 | elon / Rushing | 0.501 | 0.231 | 0.681 | 0.625 | +0.055 | +0.049 |
| 17 | elon / Dancing | 0.455 | 0.138 | 0.536 | 0.494 | +0.042 | +0.080 |
| 33 | jennie / Jumping | 0.455 | 0.164 | 0.485 | 0.436 | +0.049 | +0.087 |
| 35 | jennie / Chef | 0.488 | 0.229 | 0.661 | 0.641 | +0.020 | +0.054 |
| 38 | jensen / Skiing | 0.510 | 0.295 | 0.495 | 0.445 | +0.050 | +0.060 |
| 40 | jensen / Kickboxing | 0.505 | 0.285 | 0.544 | 0.527 | +0.016 | +0.112 |
| 51 | jisoo / Drumming | 0.551 | 0.221 | 0.591 | 0.583 | +0.008 | +0.014 |
| 55 | jisoo / Crying | 0.547 | 0.260 | 0.617 | 0.613 | +0.003 | +0.112 |
| 63 | keanu / Drumming | 0.558 | 0.228 | 0.634 | 0.574 | +0.060 | +0.069 |
| 69 | keanu / Jumping | 0.522 | 0.145 | 0.385 | 0.371 | +0.014 | +0.082 |
| 78 | lex / Angry | 0.491 | 0.231 | 0.526 | 0.548 | -0.022 | -0.043 |
| 80 | lex / Laughing | 0.489 | 0.255 | 0.568 | 0.555 | +0.013 | +0.052 |
| 87 | marion / Drumming | 0.480 | 0.226 | 0.522 | 0.510 | +0.012 | +0.076 |
| 93 | marion / Jumping | 0.448 | 0.176 | 0.388 | 0.255 | +0.134 | -0.013 |

The per-cell table shows why aggregate mean alone is insufficient. BA-off
loses in 15 cells, while confidence attenuation is sample-dependent: two cells
prefer `C=1`, even though actual has the substantially higher panel mean.
[measured]

## 6. The low/high mechanism is numerically live

The fixed split reconstructs `D` as `D_low + D_high` by construction. The
captured relative reconstruction error is `0.001240` (`0.124%`), consistent with
bf16 rounding rather than a missing band. [measured] [code]

| Band diagnostic | Result |
|---|---:|
| scheduled low scale | `0.50 → 0.85` |
| scheduled high scale | `0.75 → 1.25` |
| applied low magnitude | `0.0927` |
| applied high magnitude | `0.0599` |
| high share of `low+high` applied magnitudes | `37.15%` |
| high/low absolute spatial total-variation ratio | `1.349×` |
| high/low magnitude-normalized total-variation ratio | `3.817×` |

`D_low` is the larger applied component, while `D_high` is distinctly rougher
and supplies a substantial minority of correction magnitude. Both applied
components grow through denoising as their schedules rise. This is the expected
behavior of a working low/high split, not two duplicate maps or a dead high
branch. [measured] [visual]

The band magnitudes are RMS summaries of signed feature vectors. They are not
additive scalar energy: low and high vectors can reinforce or cancel when
merged, which is why the merged correction is measured separately. [code]

## 7. Reference attention and target routing are spatially meaningful

Across the panel, `51.44%` of face-query reference-key mass falls inside
the sealed reference face box, whose mean area is `16.68%` of the padded
reference grid. Mean per-sample enrichment is `4.643`× uniform area; `48.56%`
of the mass remains outside the declared face box. [measured]

The route is spatially meaningful—the reference face is strongly enriched and
the target router/correction overlays track the declared target face. It is not
face-exclusive. The substantial mass outside the reference box is consistent with the
known full-sequence support: masked-out reference locations remain softmax keys
and entropy candidates. The maps do not establish that every sharp
correspondence is the correct facial part. [measured] [code] [caveat]

The four overview pages show reference, actual output, `C`, applied correction,
and actual-minus-BA-off for every selected cell. Heatmaps are independently
scaled where their units are magnitudes; confidence always uses the fixed
`[0.25,1]` color scale. [visual]

![Overview 01–17](assets/cl39_attention_24k_serv_a100/cl39_overview_01_17.png)

\clearpage

![Overview 33–40](assets/cl39_attention_24k_serv_a100/cl39_overview_33_40.png)

\clearpage

![Overview 51–69](assets/cl39_attention_24k_serv_a100/cl39_overview_51_69.png)

\clearpage

![Overview 78–93](assets/cl39_attention_24k_serv_a100/cl39_overview_78_93.png)

## 8. Detailed examples

Each detailed panel overlays the exact maps on the validation output and shows
the reference-key mass on the same letterboxed reference geometry used to
construct reference latents and masks.

### 8.1 Eddie / Rushing, index 01

The reference-key mass peaks on Eddie's face, while the target router confines
the strongest correction to the generated face. Mean routed-face `C=0.468`
and correction/native `=0.225`. Actual identity is `0.632`, versus `0.630` for
`C=1` and `0.573` for BA-off. The gate is nearly neutral for identity here,
while the explicit BA route still provides a clear gain over correction-zero.
The image-space difference maps extend into the body and subway because a
face-local hidden-state intervention propagates through later denoising.
[measured] [visual]

![Detailed CL39 mechanism panel, sample 01](assets/cl39_attention_24k_serv_a100/samples/01_eddie.png)

### 8.2 Jensen / Skiing, index 38: highest route activity

This cell has the panel's highest correction/native ratio (`0.295`) with
`C=0.510`. `D_low` is visibly smoother, `D_high` concentrates on eyewear and
facial edges, and both applied maps are nonzero. Actual identity is `0.495`,
versus `0.435` under `C=1` and `0.445` under BA-off. The gate therefore selects an
intermediate correction that beats both endpoints for this difficult eyewear
case. [measured] [visual]

![Detailed CL39 mechanism panel, Jensen/Skiing](assets/cl39_attention_24k_serv_a100/samples/38_jensen.png)

## 9. Serv trainer replay gate

The instrumented actual arm reproduces the sealed A100 trajectory exactly:

| Actual replay vs sealed Serv actual | Result |
|---|---:|
| complete outputs | `96 / 96` |
| mean RGB MAE | `0.00000` |
| maximum per-image RGB MAE | `0.00000` |
| maximum absolute channel error | `0.00000` |
| pixels changed by more than `1/255` | `0.000%` |

![Serv trainer/YAML replay versus sealed Serv A100](assets/cl39_attention_24k_serv_a100/cl39_serv_trainer_vs_sealed_replay.png)

This closes the main qualification of the earlier MHZ draft. The source,
checkpoint, batch-12 grouping, CUDA generator list, validation-base swap, and
legacy processor-copy path jointly reproduce the canonical pixels. The maps
and counterfactuals in this report therefore belong to the same A100 execution
path as the authoritative CL39 panel. [measured]

## 10. Confidence in the conclusions

| Claim | Confidence | Basis | Main limitation |
|---|---|---|---|
| `C` is active and attenuates most routed corrections | High | 16-cell, all-layer telemetry; code equation | one checkpoint and one validation seed |
| The explicit `(R-N)` lane causally affects output | High for this Serv panel | same-checkpoint correction-zero arm; paired ID result; activation ratios; difference maps | counterfactual statistics are reported for 16 deterministic cells |
| Both frequency bands are computed and applied | High | reconstruction, band magnitudes, schedules, spatial roughness | magnitude maps discard feature sign/direction |
| Raw routed `R` and N-only produce visibly different faces | High for this Serv panel | full-96 batch-12 routed-R arm; fixed-crop 16-cell grids and image differences | whole-denoising intervention, not direct feature decoding |
| Reference attention is semantically correct identity correspondence | **Not established** | attention concentrates spatially but no shuffled/wrong spatial-reference arm | sharp attention can still be confidently wrong |
| BA alone causes CL39's gain over independently trained PhotoMaker | **Not established** | same-checkpoint BA-off isolates route use, not training ownership | default/generic adapters were co-trained |
| Results generalize beyond this panel/checkpoint | **Not established** | no second seed, held-out identities, or alternate dataset | fixed celebrity validation panel |

## 11. What is not established

- A low-entropy match is not necessarily a correct semantic facial-part match.
  CL39 measures sharpness, not cycle consistency or identity correctness.
- This audit does not exclude masked reference keys from the softmax. It
  diagnoses the shipped CL39 mechanism rather than silently replacing it with
  the proposed valid-key variant.
- The correction-zero arm proves that the explicit route affects Serv outputs;
  it does not decompose how CL39's jointly trained default, generic, and BA
  parameter groups created its final advantage over PM0.
- No spatial-reference shuffle was run, so dependence on the **correct spatial
  identity content**—as distinct from a generic route perturbation—remains to be
  tested.
- The 16 cells are deterministic and identity-stratified but are not a second
  independent validation set.
- The routed-`R` arm is deliberately unattenuated and outside CL39's trained
  operating policy. Its artifacts do not establish that `R` is intrinsically
  bad; they establish that raw branch substitution is not equivalent to CL39's
  bounded correction.

## 12. Recommended next experiments

### 12.1 Evaluation-only spatial-reference shuffle

**Config/arm:** exact CL39 r4 checkpoint; preserve the correct PhotoMaker ID
tokens and 512-D ID embedding, but permute only spatial reference latents and
their reference boxes within the 16-cell panel.  
**Single change:** spatial reference identity.  
**Hypothesis:** if CL39 uses identity-specific spatial correspondence, actual
outputs and owned-face ID should degrade under the shuffle.  
**Prediction:** the largest change should occur in cells with higher `C` and
higher applied correction/native ratio.  
**Risk:** the current pipeline couples PhotoMaker and spatial reference inputs;
the intervention must split them explicitly without changing the correct ID
token path.  
**Gate:** require an exact no-shuffle parity arm before interpreting results.

### 12.2 Valid-reference-key entropy ablation

**Config/arm:** exact 24k CL39 r4 checkpoint on the replay-gated Serv path;
evaluation only.  
**Single change:** exclude masked reference positions from the confidence
softmax and entropy normalization while leaving `R`, target routing, band
schedules, and PhotoMaker inputs unchanged.  
**Hypothesis:** current `C` partly measures invalid-key dilution and reference
face-mask area rather than only correspondence ambiguity.  
**Prediction:** valid-key entropy will be lower and less correlated with mask
area; improvements should concentrate in samples whose reference face occupies
a small fraction of the padded grid.  
**Risk:** changing softmax support also changes the meaning of the shipped
checkpoint's fixed threshold `0.75`; a threshold-calibrated companion arm may
be required.  
**Gate:** preserve an untouched actual arm that replays all 96 sealed images
exactly before interpreting the ablation.

These are evaluation runs, not new training experiments. If a new training arm
is launched later, it must retain the optimized pipeline and the standard
step-zero/every-2,000 fixed-96 contract.

## 13. Reproduction and artifacts

The publication generations run through `train.py` with these validation-only
Hydra configs and their one-A100 YAML launchers:

```bash
src/configs/CL39_attention_audit_serv_actual.yaml
src/configs/CL39_attention_audit_serv_c1.yaml
src/configs/CL39_attention_audit_serv_ba_off.yaml
src/configs/CL39_attention_audit_serv_reference_face.yaml
serv_run_packages/CL39_attention_audit_serv_final/run_actual_1gpu.yaml
serv_run_packages/CL39_attention_audit_serv_final/run_c1_1gpu.yaml
serv_run_packages/CL39_attention_audit_serv_final/run_ba_off_1gpu.yaml
serv_run_packages/CL39_attention_audit_serv_branch_faces_r1/run_CL39_attention_audit_serv_branch_faces_r1_1gpu.yaml
```

After all three fail-closed arm gates pass, the report job runs:

```bash
python tools/analysis/assemble_cl39_serv_audit.py --help
python tools/analysis/analyze_cl39_attention.py render \
  --output-root artifacts/cl39_attention_24k_serv_a100 \
  --figure-dir analysis/assets/cl39_attention_24k_serv_a100
python tools/reports/publish_report.py \
  analysis/2026-08-25_cl39_entropy_confidence_attention_audit.md --upload
```

The CLI's one-item `generate` stage remains available for interactive probing,
but it is not publication-comparable because changing the sealed batch size
changes the diffusion trajectory.

| Artifact | Path |
|---|---|
| analysis CLI | `tools/analysis/analyze_cl39_attention.py` |
| compact capture/aggregation | `tools/analysis/cl39_attention_capture.py` |
| Serv artifact assembler | `tools/analysis/assemble_cl39_serv_audit.py` |
| direct N/R face renderer | `tools/analysis/render_cl39_branch_faces.py` |
| usage note | `tools/analysis/README_CL39_ATTENTION.md` |
| notebook playground | `notebooks/CL39_attention_analysis.ipynb` |
| copied checkpoint and sealed records | `artifacts/checkpoints/CL39_cosmic_null_key_confidence_router_24k_full96_r4/` |
| sample manifest | `artifacts/cl39_attention_24k_serv_a100/sample_manifest.json` |
| layer telemetry and maps | `artifacts/cl39_attention_24k_serv_a100/telemetry/` |
| combined layer table | `artifacts/cl39_attention_24k_serv_a100/all_layer_calls.csv` |
| counterfactual metrics | `artifacts/cl39_attention_24k_serv_a100/counterfactual_metrics.csv` |
| joined per-sample audit table | `artifacts/cl39_attention_24k_serv_a100/per_sample_summary.csv` |
| machine-readable summary | `artifacts/cl39_attention_24k_serv_a100/summary.json` |
| Serv submission and replay gates | `artifacts/cl39_attention_24k_serv_a100/serv_jobs/` |
| N/routed-R images and metrics | `artifacts/cl39_attention_24k_serv_a100_branch_faces/` |
| N/routed-R overview and detailed figures | `analysis/assets/cl39_attention_24k_serv_a100_branch_faces/` |

The report's conclusions use exact records and generated artifacts above, not
display names alone.
