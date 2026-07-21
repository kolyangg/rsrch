# NN4 results analysis and next architecture recommendation

**Date:** 21 July 2026  
**Run:** `ba_NN4_causal_null_up0_nfs_1gpu`  
**Evidence:** normal validation through 8k, training metrics through step 9,800,
and four 96-image causal checkpoint tests at 2k/4k.

## Executive verdict

NN4 should be stopped rather than trained to 20k. It passes the engineering and
geometric-safety checks, but fails its decisive scientific objective:
**changing the spatial reference does not move the generated face toward the
changed reference identity**.

This is not another silent-branch or validation-loading failure. At the causal
test's deliberately amplified residual scale of 4, the branch measurably changes
the face, receives nonzero gradients, and reacts more to a changed reference
image than to changed reference noise. The problem is that the response is
mostly generic face/expression/texture variation and is not identity-directed.
At normal scale 1, that weak and misdirected response is visually almost hidden
by PhotoMaker, explaining the nearly identical step-0/2k/4k/8k validations.

The strongest positive result is preservation: pose, body, clothes, background,
occluders, and face attachment remain stable, without the gross pasted-face and
anatomical failures of the old absolute branched-attention route. The protected
residual PPR scaffold should therefore be retained. The next run must change
the identity representation and supervision, not merely branch strength or
training duration.

## Evidence and integrity

The four fixed-target tests were:

- 2k checkpoint on RealVisXL validation;
- 2k checkpoint on the same SDXL base as training;
- 4k checkpoint on RealVisXL validation;
- 4k checkpoint on the same SDXL base as training.

Each test contains 96 target samples and five conditions:
`PM0`, `R1N1`, `R2N1`, `R1N2`, and `R2N2`. The PPR conditions use residual
scale 4 so that a weak learned effect is visible. This yields 192 reference-swap
direction measurements per test. All integrity checks passed, the observed
batch size was 12, CFG reference-noise pairing was active, LPIPS was available,
and the effective reference CA/text mode was correctly `zero`.

The same-SDXL tests are the primary causal evidence because they remove the
RealVis cross-backbone transfer confound. RealVis is useful as a secondary
deployment-style check. Both lead to the same conclusion.

## Causal checkpoint results

Positive directional gain means replacing R1 with R2 moved the output identity
toward R2. NN4 produces no positive result:

| Step | Validation base | Mean gain toward R2 | 95% bootstrap CI | Positive fraction | Reference/noise core effect | Reference/noise LPIPS |
|---:|---|---:|---:|---:|---:|---:|
| 2k | RealVisXL | -0.00512 | [-0.00888, -0.00155] | 47.9% | 1.390 | 1.716 |
| 2k | same SDXL | -0.00638 | [-0.00971, -0.00318] | 36.5% | 1.228 | 1.443 |
| 4k | RealVisXL | -0.00609 | [-0.00959, -0.00266] | 43.2% | 1.476 | 1.912 |
| 4k | same SDXL | -0.00174 | [-0.00464, +0.00133] | 45.3% | 1.205 | 1.382 |

The 4k same-SDXL confidence interval overlaps zero, but that is only an
improvement from significantly wrong direction to statistically unresolved
direction. It is not evidence of useful control: the mean remains negative and
fewer than half the samples move in the desired direction. RealVis remains
significantly negative and becomes slightly worse from 2k to 4k.

### The branch is active, but its changes are not useful identity changes

At scale 4, the full PPR perturbation relative to PM0 is substantial inside the
face core:

| Step/base | PPR core MAE | Reference-image core effect | Reference-noise core effect |
|---|---:|---:|---:|
| 2k RealVisXL | 0.05491 | 0.01254 | 0.00903 |
| 2k same SDXL | 0.03506 | 0.00913 | 0.00743 |
| 4k RealVisXL | 0.06047 | 0.01358 | 0.00920 |
| 4k same SDXL | 0.03679 | 0.00879 | 0.00729 |

Reference content therefore reaches the output, especially perceptually: its
LPIPS effect is 1.38–1.91 times the reference-noise effect. But the margin is
not large enough, and—more importantly—the extra content-dependent change has
the wrong semantic direction. At full-image pixel level the image/noise effect
ratio is only 1.00–1.05 because the protected face-core route leaves almost all
of the image unchanged.

Amplification also reduces similarity to the original PhotoMaker identity:

| Step/base | PM0 original-ID similarity | Mean scale-4 PPR original-ID similarity |
|---|---:|---:|
| 2k RealVisXL | 0.52313 | 0.44450 |
| 4k RealVisXL | 0.52313 | 0.43381 |
| 2k same SDXL | 0.41524 | 0.37246 |
| 4k same SDXL | 0.41537 | 0.37123 |

Thus “make the branch stronger” is not the solution: scale 4 already reveals
that stronger NN4 mostly damages or drifts identity rather than replacing it
with the requested identity.

## Visual findings

I reviewed representative contact sheets across all identities and all four
tests, including panels from the beginning, middle, and end of each 96-image
set.

What works:

- Body pose, clothing, hands, background, framing, and scene structure are
  effectively unchanged between PM0 and PPR variants.
- Faces remain attached to the head and aligned with the body. I did not find
  systematic duplicated features, displaced face plates, jaw/neck seams, or
  the gross folding artifacts seen in the older absolute-replacement BA runs.
- Changes are spatially concentrated in the intended face region. This agrees
  with the exact outside-core output anchor and low full-image effects.

What does not work:

- R1 and R2 columns are usually visually almost indistinguishable at contact-
  sheet scale and both remain recognizably dominated by the PM target identity.
- Enlarged faces show changes in blinking, eye opening, mouth/expression, jaw
  shading, skin smoothness, and sharpness, but not consistent transfer of the
  swapped person's identity.
- The pattern is identity-dependent: a minority of identities/samples show a
  positive direction, while others are strongly negative. For example, at 4k
  same-SDXL the per-identity mean ranges from modestly positive to about -0.010;
  this is not a robust controller.
- RealVis produces larger visible face changes than same-SDXL, but also more
  negative directional evidence. This suggests that cross-backbone transfer
  amplifies the residual's generic appearance effect; it is not the root cause,
  because same-SDXL also fails.

The visual conclusion matches the metrics: NN4 is anatomically safe but not
identity-causal.

## Normal validation over training

| Step | ID similarity | Text similarity |
|---:|---:|---:|
| 0 | 0.522534 | 26.4113 |
| 2k | 0.518964 | 26.4658 |
| 4k | 0.518393 | 26.5298 |
| 6k | 0.519547 | 26.5308 |
| 8k | 0.518131 | 26.5296 |

Normal ID similarity ends 0.00440 below step 0 (about -0.84%) and never exceeds
the baseline. Text similarity gains only 0.12 points and plateaus by 4k. The
user-observed near-identical images are therefore expected from the recorded
metrics, not a download or Comet display issue.

There is no evidence of a late improvement trend. The causal 2k→4k movement is
inconsistent across validation backbones, and normal validation stays flat
through 8k. Continuing the same objective to 20k is unlikely to discover the
missing identity direction.

## Training health and what the diagnostics mean

There is no indication of OOM, NaN, dead gradients, failed checkpoint loading,
or an inactive processor topology. The exported training metrics reach step
9,800 and normal validation reaches 8k. The log contains temporary Comet 503/
read-timeout upload errors, but training continued and checkpoints were saved;
these are logging-service issues rather than model failures.

Important signals:

- Mean diffusion loss is noisy and essentially flat (about 0.183–0.190 across
  2k windows). This objective does not report reference identity causality.
- Total and connector gradients remain finite and nonzero. Connector-up carries
  most of the trainable gradient, so the branch is being optimized.
- `sa_ref` norm grows from 2.58 in the first window to 7.75 after 8k. Parameters
  are moving substantially; the branch is not frozen.
- PhotoMaker identity attenuation is exactly 50% as configured.
- The low-timestep identity loss is applied to roughly 41–44% of logged batches,
  but its mean does not improve consistently.
- Null residual is very small (roughly 0.6–1.4e-6). The learned-null suppression
  objective is functioning numerically.
- `match_null_margin` falls to essentially zero after 2k. This metric is the
  **hinge penalty**, so zero means the small magnitude margin is already
  satisfied. It does not mean matched and null collapsed; it also does not say
  the matched residual carries the correct identity. The objective becomes
  nearly inactive once any nontrivial difference exists.
- `cap_excess` rises from 0.00024 in 0–2k to about 0.0035 after 8k. Site logs
  repeatedly show many of the 30 `up_blocks.0.attn1` processors at the 0.15 RMS
  cap. The cap is preventing a growing raw residual from taking over; increasing
  it would expose more of the already misdirected signal.
- The per-site scalar gate remains close to its 0.25 initialization. Its tiny
  movement and the cap saturation reinforce that the missing signal is not
  simply insufficient scalar authority.

### Why the existing losses can reach these numbers without learning identity

The main reconstruction target, PhotoMaker target identity, and matched
reference all describe the same person. A generic face correction can lower
diffusion/identity loss without encoding which reference identity supplied it.
The candidate-level matched/null margin only requires a nonzero connector
response; it does not require that changing identity changes the response in
the correct identity direction. Once its 0.02 margin is met, it contributes
almost no gradient. The reference identity loss is also confounded by the
same-person target and by full PhotoMaker conditioning in half the batch.

This explains the full pattern:

1. branch parameters and norms grow;
2. the branch reacts to image content;
3. stronger inference changes the face;
4. swapped-reference direction remains absent or negative;
5. normal PhotoMaker validation barely changes.

## Do more checkpoints need to be downloaded?

No additional checkpoint is required to decide whether to stop NN4. The 2k and
4k checkpoints were directly loaded into both same-base and alternate-base
causal tests, all integrity checks passed, and training/validation metrics show
the same failure through 8k.

Download the 8k checkpoint only if it is cheap and useful as a research archive
or if one final same-SDXL scale-4 matrix is desired to document that no late
turn occurred. It is not necessary before designing the next run.

## Recommendation for the next run: NN5 causal identity PPR

The next experiment should preserve NN4's safe PPR operator but introduce a
**true fixed-target, wrong-identity counterfactual training path**. This is the
smallest change that directly addresses the failed causal metric.

### Keep unchanged

- ordinary target self-attention as the base;
- target-Q to packed reference-K/V branched self-attention;
- zero-initialized additive connector and learned null through the same K/V
  projections;
- soft face-core routing and exact PhotoMaker output outside the core;
- `up_blocks.0` only, branched CA disabled, reference token/pooled text neutral;
- CFG reference-noise pairing, pose adaptation off, and CA face mixing off;
- residual cap at 0.15 or lower and runtime scale 1 for the approval metric.

### Replace magnitude-only causality with paired identity-direction training

For a target identity A, construct matched and swapped-reference examples with
the **same target latent, target noise, timestep, prompt, masks, and geometry**:

```text
matched path:      target A + reference A
counterfactual:    target A + reference B
```

Process the pair in one physical batch where possible. Use ordinary diffusion
reconstruction on the matched path. On the counterfactual path, do not apply an
inside-face pixel/diffusion target for identity A; that would explicitly teach
the model to ignore B. Instead:

- decode predicted clean faces only at safe low timesteps;
- drive face-recognition identity toward B;
- preserve target A's pose, expression, landmarks, visibility, and face/body
  boundary with frozen geometry/validity networks;
- preserve the ordinary target prediction outside the face core exactly;
- gate all decoded losses on successful face detection and landmark confidence;
- include a null path whose connector residual is explicitly suppressed.

Train across a continuum of target PhotoMaker-ID strengths rather than only a
hard 50% on/off split. The counterfactual subbatch must include PM-ID=0 cases so
the branch owns identity, but should also include PM-ID=1 cases so it learns to
modify the real inference condition instead of succeeding only when PhotoMaker
is absent.

### Make reference memory identity-focused and noise-stable

The current output-level reference/noise ratios of only 1.2–1.9 show that the
same-timestep noised ROI remains a significant nuisance source. For the first
NN5 implementation, add a two-noise consistency loss for the same reference:

```text
Delta(A, N1) approximately equals Delta(A, N2)
```

The more promising architectural follow-up is to derive branched K/V from a
clean, frozen face encoder/cache rather than an evolving noised reference U-Net
stream. Use identity-focused global and local facial-part tokens, with part ID,
visibility, and landmark/3D-relative position. Target queries still perform
branched attention over reference K/V, so the core BA hypothesis is preserved,
but eyes retrieve eye memory and mouth retrieves mouth memory instead of an
unstructured bbox token set.

Do not combine a full new reference encoder with the first counterfactual run
unless implementation cost forces a single attempt. The clean scientific order
is:

1. **NN5a:** current packed ROI memory + true matched/wrong identity objective +
   two-noise consistency;
2. **NN5b:** replace the evolving reference stream with clean semantic/part K/V
   after NN5a demonstrates positive identity direction;
3. only then consider query/head/timestep gates or a low-cap `up_blocks.1`
   detail route.

### NN5 approval gate

Run the same 96-image matrix on the same-SDXL base at 2k and 4k, at normal
residual scale 1; scale 2 may be a secondary sensitivity check. Continue beyond
4k only if all of the following hold:

- mean directional gain toward R2 is positive and its 95% bootstrap interval
  excludes zero;
- positive fraction is materially above 50% and not driven by one identity;
- reference-image effect clearly exceeds reference-noise effect in both identity
  and LPIPS spaces;
- pose/landmark and outside-core differences remain close to PM0;
- visual panels show identity movement rather than blinking, expression, age,
  smoothness, or sharpness changes.

Use RealVis only after the same-base gate passes. Two GPUs may reduce wall-clock
time for the paired paths, but changing GPU count or effective batch alone will
not fix NN4's objective.

## Changes that are not recommended

- Do not continue NN4 merely because the 20k budget remains.
- Do not increase runtime scale, gate maximum, RMS cap, or number of attention
  sites before identity direction is positive.
- Do not re-enable absolute reference ownership, branched CA, pose adaptation,
  or CA face mixing as a shortcut.
- Do not train wrong references against target-A face reconstruction loss.
- Do not interpret lower diffusion loss, larger face MAE, or a satisfied
  matched/null magnitude margin as proof of reference identity control.

## Preserved evidence

The compact evidence bundle is in `21Jul_NN4_results/`. It contains aggregate
CSVs, full causal metric tables and tensor diagnostics, selected contact sheets,
the Comet metric history, and the training log. Full images, crops, and heatmaps
remain in `rsrch_21Jul_test/`.
