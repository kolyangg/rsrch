# Query-adaptive hard BA-v4 results through 12k

**Run:** `rhca_big_celebs_scheduled_v1_hard_ba_v4_q16_r32_20k_full96_r1`  
**Comet:** [`408606871a5b40c6b75d2da855b83a44`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/408606871a5b40c6b75d2da855b83a44)  
**Local evidence:** `comet_data/rhca_big_celebs_scheduled_v1_hard_ba_v4_q16_r32_20k_full96_r1`  
**Historical comparison:** `rhca_big_celebs_sameid_40k_full96_r1`, Comet [`569cc685ff9144f5a9b42bf70e14e040`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/569cc685ff9144f5a9b42bf70e14e040)  
**Evidence cutoff:** completed fixed-96 validation through step 12,000; training telemetry through step 13,300

## Executive verdict

This run is a **mechanical and partial learning success, but not a quality or
identity promotion through 12k**.

The clean hard branched-attention mechanism is installed correctly, receives
stable gradients, changes its branch target queries, uses a substantial
reference-derived message, and never leaks native face attention back through
a mixer. Identity recovers from the expected early 2k dip and rises from
`.1488` at initialization to `.2054` at 12k, with a peak of `.2213` at 8k.
Text similarity, face detection, generic image quality, and most face-quality
scores all improve substantially from initialization.

The important positive result is the **within-run identity movement**. From
step 0 to 12k, v4 gains `.0566`; the historical same-ID run gained `.0638`
over the same interval. V4 therefore does not look frozen, PhotoMaker-anchored,
or incapable of learning. It reproduces much of the old run's *relative*
multi-thousand-step identity trajectory with only the exact 12.329M BA-owned
parameters.

The failure is the **absolute operating point and visual usability**. V4 starts
`.1575` below the historical run and remains `.1647` below it at 12k. The
different `validation_native` versus `legacy_full_copy` construction prevents
treating this as a clean scalar architecture comparison, but full-panel visual
review confirms that the low v4 score is not merely metric semantics. Many
faces begin blank, stretched, or textureless; training reconstructs recognizable
face structure, but at 12k there are still colored vertical patches, duplicated
features/accessories, mask-like face inserts, extreme mouths/expressions, and
poor integration under hands, hair, goggles, or strong lighting.

The telemetry explains the split result. The reference branch becomes stronger
relative to native attention and its query adapts, while its conditional
correct-versus-shuffled advantage stays positive but small and declines from
the first phase. V4 is learning a strong, face-producing reference route, but
not enough of that capacity is aligned with the *correct person's identity*.
It improves face completion and visual quality more reliably than identity.

No next experiment is proposed in this report, as requested.

## Evidence integrity and scope

| Check | Observed result |
|---|---|
| Immutable run identity | Every export resolves to `408606871a5b40c6b75d2da855b83a44` in `jul-comet-large-testing-tr`. |
| Requested image steps | Exact matches at 0, 2,000, and 12,000; no fallback. |
| Available Comet image steps | 0, 2k, 4k, 6k, 8k, 10k, and 12k. |
| Locally reviewed image panels | All 96 images at 0, all 96 at 2k, and all 96 at 12k. |
| Image integrity | 288/288 images decode as RGB 1024×1024; filenames match across panels. |
| Duplicate-output audit | 0/96 byte-identical pairs for 0→2k, 0→12k, and 2k→12k. |
| Validation metrics | Complete at every 2k gate from 0 through 12k. |
| Training telemetry | 267 samples per BA/loss/gradient curve through step 13,300. |
| Export warnings/errors | Zero warnings and zero errors. |
| Runtime log | No traceback, CUDA OOM, non-finite diagnostic, or integrity failure found. |
| Last completed checkpoint evidenced by log | `checkpoint-epoch6.pth` and `weights-epoch6.pth`, corresponding to step 12,000. |
| Planned versus evidenced endpoint | Config requests 20k; this package ends during epoch 7 at batch 1,332, before a completed 14k validation. The reason/final live machine state is not established by the local package. |

The package does not contain per-image identity scores. Aggregate identity
deltas are exact for the deterministic panel; identity-specific visual
observations below come from reviewing all 288 local images, not from a
per-image metric table.

## Configuration actually evaluated

| Component | Resolved value |
|---|---|
| Architecture | `query_adaptive_hard_sa_v4` |
| Target-face merge | Hard reference replacement at unit scale; no native/reference face interpolation |
| Target query | Branch-only rank-16 effective-Q clone |
| Reference K/V | Rank 32 |
| Reference output residual | Rank 32, zero initialized |
| Patched processors | 46 self-attention sites in `mid`, `up0`, and `up1` |
| Exact trainable contract | 368 tensors / 12,328,960 FP32 parameters |
| Optimizer roles | `ref_query`: 1.76128M; `ref_kv`: 7.04512M; `ref_output`: 3.52256M |
| Learning rates | `5e-5 / 5e-5 / 1e-4` for query / K/V / output |
| Timestep policy | `inference_active` |
| Wrong-reference path | 25% spatial shuffle, detached diagnostic, zero causal/rank-loss weight |
| Loss | Full + face + boundary, weights `1.0 / 1.0 / 0.1` |
| Validation processor base | `validation_native` on RealVisXL V4.0 |
| Required BA controls | `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, branched CA disabled |
| Training data | Pinned BigCelebs v2 policy-v1 schedule, batch size 2 |

## 1. Validation trajectory

| Step | Identity | Text | TOPIQ-Face mean | TOPIQ-Face p10 | TOPIQ | MUSIQ | MANIQA | Face detected | TOPIQ-Face coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .148798 | 27.4134 | .543050 | .403622 | .478471 | 62.7253 | .570457 | 92/96 | 63/96 |
| 2k | **.114847** | 27.6102 | .603129 | .494931 | .531247 | 66.8843 | .564270 | 95/96 | 69/96 |
| 4k | .162517 | 27.7419 | .607103 | .498498 | .533455 | 66.9097 | .568226 | 96/96 | 78/96 |
| 6k | .191403 | 27.9609 | .642466 | .519998 | **.552927** | 68.7150 | .581028 | 95/96 | 87/96 |
| 8k | **.221332** | 27.5848 | .627806 | .500201 | .550024 | **69.4518** | **.584726** | 96/96 | **94/96** |
| 10k | .186476 | **28.0998** | **.642579** | **.523738** | .548464 | 68.3539 | .581802 | 96/96 | 91/96 |
| 12k | .205400 | 27.9370 | .630272 | .491640 | .548100 | 68.5763 | .580458 | 96/96 | 89/96 |

### 1.1 What improves

| Metric | Step 0 | Step 12k | Change | Interpretation |
|---|---:|---:|---:|---|
| Identity | .148798 | .205400 | **+.056602** | Real identity learning occurs despite the low absolute level. |
| Text | 27.4134 | 27.9370 | **+.5236** | Frozen native/background attention plus full-image loss preserves and improves prompt alignment. |
| TOPIQ-Face mean | .543050 | .630272 | **+.087222** | Faces become sharper and more complete on average. |
| TOPIQ-Face p10 | .403622 | .491640 | **+.088018** | The weakest tail improves substantially from the severely broken initialization. |
| TOPIQ | .478471 | .548100 | **+.069629** | Overall rendering quality improves while the scene structure remains stable. |
| MUSIQ | 62.7253 | 68.5763 | **+5.8510** | Strong generic perceptual-quality gain. |
| MANIQA | .570457 | .580458 | **+.010001** | Smaller but positive generic-quality gain. |
| Face detection | 95.83% | 100% | **+4.17 pp** | Training converts blank/failed faces into detector-recognizable faces. |
| TOPIQ-Face coverage | 65.63% | 92.71% | **+27.08 pp** | Far more images yield scorable face crops. |

### 1.2 What does not improve cleanly

Identity is not monotonic. It peaks at `.2213` at 8k, falls sharply to `.1865`
at 10k, and only partly recovers to `.2054` at 12k. The 12k score is `7.2%`
below the 8k peak. This is not a simple early plateau: the model has learned
meaningfully, but the identity direction oscillates while other quality
metrics remain strong.

The weakest-face tail also regresses late. From its 10k peak to 12k,
TOPIQ-Face mean falls `.0123`, p10 falls `.0321`, and coverage falls from
`91/96` to `89/96`. Face detection stays 96/96, so these are not missing-face
failures; they are quality/integration failures among detectable faces.

Generic quality metrics must not be read as proof that the faces are usable.
The 12k panel is sharper and easier to detect than step 0, but many of the
sharp details are pathological: harsh teeth, extreme mouths, colored mask
stripes, repeated goggles, and abrupt texture changes can raise sharpness or
face-detection metrics while reducing realism and identity fidelity.

## 2. Comparison with the historical same-ID run

Absolute cross-run values are confounded by processor construction:
historical validation used `legacy_full_copy`, while v4 uses
`validation_native`. The models also differ in patched sites, loss, data order,
precision, and trainable ownership. The table is useful evidence, but not a
one-variable ablation.

| Step | Historical identity | V4 identity | Historical Δ from step 0 | V4 Δ from step 0 |
|---:|---:|---:|---:|---:|
| 0 | .306327 | .148798 | — | — |
| 2k | .284121 | .114847 | −.022206 | −.033951 |
| 4k | .313756 | .162517 | +.007429 | +.013719 |
| 6k | .309550 | .191403 | +.003223 | +.042605 |
| 8k | .360949 | **.221332** | +.054622 | **+.072534** |
| 10k | .372307 | .186476 | +.065980 | +.037678 |
| 12k | .370134 | .205400 | **+.063807** | **+.056602** |

Two conclusions coexist:

1. **The clean BA path learns almost as much within its own coordinate
   system.** Its 0→12k identity gain is only `.0072` smaller than the old
   run's, and its 0→8k gain is larger.
2. **It remains much worse as an actual generator.** The absolute gap is
   `.1575` at initialization and `.1647` at 12k; the 12k full panel visibly
   retains more severe face failures than the historical setup.

The first point is evidence that the old identity trajectory was not produced
*only* by its accidental 139.35M broad adapter parameters. The explicit hard
BA route itself has material learning capacity. The second point shows that
the clean v4 initialization/representation does not provide a sufficiently
good face operating point by 12k.

Prompt preservation is a genuine v4 strength. V4 text increases by `.524`
from 0 to 12k and reaches `28.10` at 10k. The historical run's text peaked
early and then declined toward `26.75` at 12k. This is consistent with v4
freezing the native/background and ordinary cross-attention paths instead of
training broad generic/PhotoMaker adapters.

## 3. Branched-attention telemetry

The values below are phase means over sampled training diagnostics. The final
row is partial, covering 12k–13.3k rather than a completed 14k phase.

| Phase | Query Δ / native Q | Ref/native RMS | Ref/native cosine | Conditional error gap | Conditional relative gap | Conditional prediction Δ | Native-face leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–2k | .03078 | 1.1810 | .4137 | .00413 | .02046 | .09638 | 0 |
| 2–4k | .07612 | 1.2716 | .4253 | .00344 | .01768 | .07861 | 0 |
| 4–6k | .08834 | 1.3317 | .4140 | .00316 | .01583 | .07958 | 0 |
| 6–8k | .08489 | 1.3964 | .4113 | .00278 | .01520 | .07352 | 0 |
| 8–10k | .08721 | 1.4417 | .4133 | .00277 | .01561 | .07471 | 0 |
| 10–12k | .08808 | 1.4476 | .4156 | .00289 | .01491 | .07743 | 0 |
| 12–13.3k | .08809 | 1.4551 | .4143 | .00301 | .01599 | .07858 | 0 |

### 3.1 What works mechanically

- **Hard routing is real.** `hard_face_native_leakage` is exactly zero for all
  267 logged samples. There is no learned retreat toward PhotoMaker/native
  face attention and no hidden interpolation.
- **The target query learns.** Its relative delta rises rapidly from `.031`
  in the first phase to about `.088`, then remains active. V4 is not behaving
  as its frozen initialization.
- **The reference message does not collapse.** Its RMS grows from `1.18×`
  native to about `1.45×`; the reference branch is at least as strong as the
  native message inside the measured face region.
- **The branch remains distinct.** Cosine stays near `.41`, so the reference
  message is not simply copying native attention.
- **Spatial reference content matters.** Conditional wrong-reference gaps and
  prediction deltas remain positive on average.

### 3.2 What the telemetry says is insufficient

The correct-versus-shuffled advantage is small and weakens while branch
magnitude grows. Conditional error gap declines from `.00413` to roughly
`.0028–.0030`; relative gap declines from `.0205` to about `.015`; conditional
prediction delta declines from `.0964` to roughly `.074–.079`. Meanwhile the
reference/native RMS ratio climbs steadily to `1.45`.

This separates **branch strength** from **identity usefulness**. V4 learns to
make a strong face-local change, and correct spatial reference information is
not ignored, but a growing share of the branch output appears to encode generic
face reconstruction, expression, texture, or canonical-reference appearance
rather than uniquely correct identity. That matches the validation pattern:
large gains in face coverage and perceptual quality, much smaller and unstable
identity gains.

Layerwise query adaptation is also informative:

| Phase | Mid query Δ | Up0 query Δ | Up1 query Δ |
|---|---:|---:|---:|
| 0–2k | .0214 | .0362 | .0193 |
| 2–4k | .0502 | .0892 | .0540 |
| 4–6k | .0598 | **.1020** | .0677 |
| 6–8k | .0632 | .0948 | .0717 |
| 8–10k | .0675 | .0954 | .0790 |
| 10–12k | .0699 | .0933 | **.0926** |
| 12–13.3k | .0715 | .0938 | .0870 |

Up0 adapts fastest and saturates early; mid grows gradually; up1 catches up
late. Every selected layer group participates. There is no evidence here of a
completely dead group, but the query-delta plateau after roughly 4k–6k is
consistent with later identity oscillation rather than continued acceleration.

## 4. Optimization health

| Phase | Total loss | Full | Face | Boundary | Query grad | K/V grad | Output grad | Total grad | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–2k | .4226 | .1922 | .2095 | .2090 | .00428 | .02056 | .02941 | .03630 | .4970 |
| 2–4k | .4045 | .1847 | .1996 | .2019 | .00645 | .01790 | .02675 | .03287 | .4890 |
| 4–6k | .4168 | .1908 | .2052 | .2079 | .00595 | .01559 | .02172 | .02743 | .4888 |
| 6–8k | .4094 | .1860 | .2029 | .2049 | .00699 | .01817 | .02360 | .03063 | .4884 |
| 8–10k | .4141 | .1887 | .2046 | .2075 | .00685 | .01733 | .02240 | .02929 | .4907 |
| 10–12k | .4145 | .1884 | .2053 | .2081 | .00746 | .01830 | .02255 | .03010 | .4887 |
| 12–13.3k | .4074 | .1854 | .2014 | .2048 | .00844 | .02017 | .02127 | .03061 | .5138 |

All logged losses and gradient norms are finite. Every optimizer role receives
nonzero gradients; none vanishes, explodes, or is consistently clipped near
the max-norm threshold. Average throughput is stable around `.49 steps/s`, or
approximately `2.04 s/step`.

The loss does not show a sustained downward phase trend after the first 4k;
it fluctuates around `.41`. Because images, masks, and inference-active noise
levels vary, this is not by itself an optimization failure. It does show that
the later validation changes are not accompanied by an obvious improving
training objective. The full, face, and boundary components all share the
same flat behavior.

## 5. Full-panel visual review

### 5.1 What works visually

1. **Scene structure is exceptionally stable.** Background, body, pose,
   clothing, camera, and prompt composition remain nearly fixed from 0 to 12k
   across the panel. The visible changes are concentrated in the face. This is
   direct evidence that freezing native target/background attention works as
   intended.
2. **Blank faces become faces.** Many step-0 outputs have smooth, stretched,
   or missing facial features. By 2k and especially 12k, eyes, noses, mouths,
   beards, glasses, and expressions emerge. This matches the large detection
   and TOPIQ-Face coverage gains.
3. **Several identities become recognizable.** Keanu's hair/beard silhouette,
   Marion's face shape, Lex's compact morphology, Jensen's hair/glasses, and
   some Eddie/Elon examples become more identifiable by 12k.
4. **Prompt expressions are learned strongly.** Angry, crying, laughing, and
   action prompts visibly affect mouths and expressions instead of leaving a
   featureless face insert.
5. **The result is not plain PhotoMaker anchoring.** Faces change materially at
   every reviewed checkpoint and are clearly controlled by the hard branch.

### 5.2 What does not work visually

1. **Hard-mask feature discontinuities persist.** Many 12k faces contain a
   vertical or central colored strip, abrupt contrast boundary, different skin
   tone, or a pasted-mask appearance. This is particularly visible for Jisoo
   and Jennie and under neon/night lighting.
2. **Occlusions are not integrated.** Hands over faces, loose hair, goggles,
   chef hats, and boxing headbands frequently merge into or erase eyes, noses,
   and mouths. The branch reconstructs a face without consistently respecting
   target occlusion geometry.
3. **Reference accessories leak or duplicate.** Jensen frequently acquires
   sunglasses or multiple glasses/goggle structures even when the target scene
   does not call for them. Skiing examples can contain doubled lenses or eyes
   inside goggles.
4. **Expressions are overdriven.** Laughing, crying, angry, kickboxing, and
   sometimes neutral/action prompts produce oversized open mouths, excessive
   teeth, harsh wrinkles, or horror-like faces. Strong prompt response is not
   the same as natural expression control.
5. **Identity quality is uneven by person.** Marion and many Keanu/Lex cases
   are comparatively coherent; Jisoo remains the most consistently broken;
   Jennie varies from clean to central-strip artifacts; Jensen is dominated by
   accessories; Eddie and Elon often gain expression at the expense of a
   natural likeness.
6. **The 12k panel is sharper but not uniformly better than 8k–10k metrics
   imply.** The late p10 and coverage regression is visible in a persistent
   tail of severe cases even though the average face is more complete.

### 5.3 Prompt-family behavior

| Prompt family | What improves | Persistent failure |
|---|---|---|
| Angry / crying / laughing | Faces become complete and expressions become unmistakable. | Mouths, teeth, wrinkles, and hand-face intersections are exaggerated or broken. |
| Chef / reading | Calmer compositions often yield cleaner morphology and better identity cues. | Hats/hair and face masks can still stretch features or create central seams. |
| Dancing / jumping / drumming | Body pose and background are preserved almost perfectly while faces gain detail. | Dynamic hair and small faces still produce mask-like insertions and colored center patches. |
| Kickboxing | Headbands and target pose remain stable; facial detail increases strongly. | High rate of open-mouth distortion, missing eyes, duplicated cavities, and mask/headband conflicts. |
| Night ride | Global neon scene remains stable and text adherence is strong. | Lighting colors bleed into a central face strip; mouths/teeth become unnaturally bright. |
| Skiing | Clothing, snow scene, and goggles remain prompt-correct. | Reference eyewear leaks and duplicates; goggles, eyes, skin, and teeth overlap incorrectly. |

## 6. What works and what does not

| Area | Verdict | Evidence |
|---|---|---|
| Exact trainable ownership | **Works** | 368/368 tensors, 12.329M parameters, only three intended roles. |
| Branched-attention installation | **Works** | 46 intended processors; strict startup contract passes. |
| Hard no-mix invariant | **Works** | Native-face leakage is zero for every telemetry sample. |
| Target-query learning | **Works** | Query delta grows from `.031` to about `.088`; every layer group participates. |
| Branch activity | **Works** | Reference message grows to about `1.45×` native RMS and remains distinct. |
| Gradient/precision health | **Works** | All roles have stable finite FP32 gradients; no numerical failure. |
| Native structure preservation | **Works strongly** | Scenes, poses, bodies, and backgrounds remain stable; text improves. |
| Face detectability/completion | **Works strongly** | Detection reaches 96/96; coverage rises by 27 points; face-quality means rise. |
| Multi-thousand-step identity learning | **Works partially** | +.0566 from 0→12k and +.0906 from the 2k trough; comparable relative movement to the old run. |
| Absolute identity | **Does not work well enough** | `.2054` at 12k versus `.3701` historical; full panel confirms weak likeness. |
| Identity stability | **Does not work** | Peak at 8k, sharp 10k regression, only partial 12k recovery. |
| Correct-reference specificity | **Present but weak** | Positive conditional gaps, but they are small and decline as branch magnitude increases. |
| Face realism/integration | **Does not work reliably** | Persistent seams, color patches, duplicated accessories/features, occlusion failures, and extreme expressions. |
| Worst-case face quality | **Still inadequate** | TOPIQ-Face p10 and coverage regress from their 8k–10k peaks; severe visual tail remains. |
| Promotion over historical same-ID setup | **No** | V4 does not close the absolute identity or visual-quality gap through 12k. |

## 7. Overall interpretation

V4 answers the main mechanistic question more cleanly than any earlier run:
**branched attention can learn without a native-face mixer and without broad
accidental trainables.** The hard path is active, query-adaptive, stable, and
capable of producing an identity gain close to the historical run's within-run
gain over the same horizon.

It also reveals the next-level limitation with unusual clarity. The problem is
no longer “PhotoMaker takes over” or “the BA branch is dead.” PhotoMaker/native
face attention cannot take over in this architecture, leakage is zero, the
branch grows stronger, and its query learns. The remaining failure is that a
strong hard branch is not automatically a *well-aligned identity branch*.
It learns face completeness, expression, sharpness, and some reference traits,
but its correct-reference advantage is weak and its face features are not
consistently reconciled with target geometry, occlusions, lighting, or the
surrounding residual stream.

Therefore the fairest result label is:

> **Clean BA mechanism validated; relative identity learning demonstrated;
> absolute identity and face integration remain insufficient through 12k.**

This report deliberately stops at analysis and does not specify a successor
architecture or experiment.
