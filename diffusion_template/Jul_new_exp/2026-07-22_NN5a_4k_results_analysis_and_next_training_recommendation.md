# NN5a 4k results analysis and next training recommendation

**Date:** 22 July 2026  
**Run:** `ba_NN5a_counterfactual_directional_ppr_1gpu`  
**Checkpoint tested:** epoch 2 / optimizer step 4,000  
**Primary evidence:** RealVis normal validation through 4k and the 96-sample,
five-condition, RealVis residual-scale-1 causal test

## Executive verdict

NN5a is a healthy run and a useful negative experiment, but it does **not**
pass the 4k causal approval gate. The new counterfactual objective makes the
branch more visibly active than NN4 at normal inference strength, yet changing
the spatial reference still does not move the generated face reliably toward
the changed identity.

The decisive 4k result is a mean identity-direction gain of `-0.000684`, with a
95% bootstrap interval of `[-0.002816, +0.001291]` and only `48.96%` positive
samples. This is statistically unresolved and visually indistinguishable from
chance. The result also changes sign with reference-noise seed. NN5a should be
stopped or held at 4k rather than continued unchanged to 20k.

This is not a silent-branch, checkpoint-loading, or validation failure. All
integrity assertions pass, all PPR parameter groups receive finite gradients,
the branch produces a substantial face-core perturbation, and internal branch
tensors react much more strongly to reference content than to reference noise.
The failure is semantic: the PPR residual is learned and spatially safe, but its
output is mostly a generic face/gaze/expression correction rather than a
reference-identity controller.

The next training priority should be **NN5b**, which retains the protected PPR
operator and counterfactual supervision but adds clean PhotoMaker-V2 identity
tokens to the reference route. It should still be judged at 2k/4k by the same
scale-1 causal matrix; a nominal 30k job budget should not imply training it
blindly to 30k if the early causal gate fails.

## Evidence and test integrity

The controlled test contains 96 target samples and five conditions:

```text
PM0   ordinary PhotoMaker, PPR scale 0
R1N1  matched spatial reference, reference noise N1
R2N1  cyclic wrong-identity reference, the same N1
R1N2  matched spatial reference, reference noise N2
R2N2  wrong-identity reference, the same N2
```

The manifest confirms:

- checkpoint `checkpoint-epoch2.pth`, corresponding to step 4,000;
- `SG161222/RealVisXL_V4.0` validation base;
- residual scale `1.0`;
- 96 samples and observed batch size 12;
- fixed target seed and paired reference-noise seeds 918273/271828;
- reference-half cross-attention mode `zero`;
- identity-token lane disabled, as required for NN5a;
- LPIPS available and every integrity assertion passed.

The generated `conclusion.md` has a stale “PPR 8k” heading. This is a report-
template label only: the manifest, source directory, checkpoint path, and Comet
history consistently identify the test as 4k.

Normal-validation images were not downloaded locally, so the temporal visual
assessment uses the user's observation. The detailed visual findings below are
from the complete 4k causal panel, face crops, and contact sheets.

## Normal validation: more movement, not better identity

| Step | ID similarity | Text similarity |
|---:|---:|---:|
| 0 | 0.523129 | 26.3659 |
| 2k | 0.508999 | 26.4383 |
| 4k | 0.507018 | 26.4660 |

From step 0 to 4k, identity similarity falls by `0.016111` (about `3.08%` of
the baseline), while text similarity rises by only `0.1001`. This agrees with
the observation that 4k looks more different than the earlier flat runs, but
it does not indicate improvement: the standard identity metric moves in the
wrong direction and most of the change occurs by 2k.

The curve gives no evidence of an emerging late identity gain. Between 2k and
4k, ID similarity declines by another `0.001981`; the visible effect is opening
without becoming more identity-correct.

## Decisive reference-versus-noise result

### Swapping the reference does not transfer identity

| Noise condition | Mean gain toward R2 | 95% bootstrap CI | Positive fraction |
|---|---:|---:|---:|
| N1 | +0.000842 | [-0.002448, +0.004132] | 53.13% |
| N2 | -0.002209 | [-0.004556, +0.000196] | 44.79% |
| Combined | **-0.000684** | **[-0.002816, +0.001291]** | **48.96%** |

The opposing N1/N2 results are especially important. A learned identity
controller should preserve the sign when only reference noise changes. Here,
individual samples can also reverse sign across N1 and N2—for example samples
005 and 062. The apparent small positive cases are therefore not robust
evidence of identity transfer.

No target identity carries the overall result. Per-identity mean gains range
from approximately `-0.00644` to `+0.00272`, and positive fractions range from
37.5% to 58.3%. None reaches the predeclared 60% continuation threshold.

### The branch changes the face, but most of the change is not reference-specific

Relative to PM0, the mean scale-1 PPR effect is substantial:

| Effect versus PM0 | Full-image MAE | Face-core MAE | Face-core LPIPS |
|---|---:|---:|---:|
| Mean PPR effect | 0.011148 | 0.030543 | 0.069051 |

Holding noise fixed and changing only R1 to R2 produces:

| Pairwise effect | Full-image MAE | Face-core MAE | Face-core LPIPS |
|---|---:|---:|---:|
| Reference-image swap | 0.007112 | 0.009498 | 0.013039 |
| Reference-noise swap | 0.007043 | 0.008429 | 0.010447 |
| Reference/noise ratio | **1.010×** | **1.127×** | **1.248×** |

There is real reference-content sensitivity, particularly in LPIPS, but it is
only modestly larger than nuisance sensitivity to reference noise at the final
image. The reference swap accounts for a face-core displacement equivalent to
31.1% of the full PM0→PPR displacement; the noise swap is already 27.6%.
These pairwise magnitudes are not additive, but their near equality shows why
R1 and R2 columns look almost identical.

Mean original-ID similarity falls from `0.523129` in PM0 to `0.506707` across
the four PPR variants (`-0.016422`). Meanwhile, using R2 does not produce a
consistent rise in similarity to R2. The branch is moving away from the
PhotoMaker output without landing on the supplied replacement identity.

## Tensor-path diagnosis

The exact captured tensors rule out complete reference blindness:

| Captured stage | Reference/noise sensitivity ratio |
|---|---:|
| Reference hidden state | 3.17× |
| Reference candidate | 3.36× |
| Connector down | 3.65× |
| Raw residual | 4.80× |
| Bounded residual | 4.99× |
| Applied residual | 4.35× |
| Target epsilon before output anchor | **1.018×** |
| Target epsilon after output anchor | **1.014×** |

The spatial reference is distinguishable inside the PPR branch. By the target
epsilon, however, changing content and changing reference noise have almost the
same aggregate effect. This is the key architectural result: the current
spatial candidate carries image-specific information internally, but the
iterative U-Net trajectory does not preserve it as a clean identity direction.
Adding more residual scale or sites would amplify a contaminated output rather
than solve this bottleneck.

## Visual assessment

### What works

- Pose, clothing, hands outside the face core, background, framing, and scene
  structure are essentially unchanged across PM0 and all four PPR variants.
- Faces remain attached and aligned with the body. There is no systematic
  pasted face plate, duplicated face, displaced jaw/neck, or broad anatomical
  failure.
- I did not find a systematic hard bbox seam. The exact outside-core anchor and
  soft core routing are still providing a useful safety envelope.
- Face detection is 100% in every condition, and mean face confidence remains
  close to PM0 (`0.8340` for PM0 versus roughly `0.8320` for PPR).
- Text/scene adherence is preserved and slightly higher than PM0 on average.

### What does not work

- At contact-sheet scale, R1N1, R2N1, R1N2, and R2N2 are usually
  indistinguishable. Enlarged crops reveal edits, but not the intended person
  swap.
- The recurring edits concern eye opening, gaze, mouth tension, skin smoothing,
  sharpness, and local expression. These are generic face-rendering changes.
- Sample 007 is the clearest structural case: PM0 shows the eyes around the
  hands, while all four PPR variants similarly rewrite the hand/eye occlusion.
  R2 does not create a distinct identity effect.
- Sample 090 similarly changes gaze/eye orientation in all PPR variants. This
  explains the elevated landmark metric for that sample without indicating
  identity transfer.
- A small number of difficult/occluded samples have sizable face-core LPIPS and
  landmark displacement, but the same shift appears under both references and
  both noise seeds. These are generic branch artifacts, not controlled swaps.

The geometry safety result remains good at the body/attachment level, but face-
internal pose and occlusion are not perfectly protected.

## Training health

The run completes both epochs and saves epoch-1 and epoch-2 checkpoints. The
log contains no traceback, OOM, invalid-sample skip, NaN/non-finite event, or
runtime error. Installation reports 30 branched self-attention processors,
zero branched cross-attention processors, and 6.45M trainable parameters.

The counterfactual construction is also behaving as intended:

- `reference_noise_equal` is exactly 1.0 throughout;
- mean A/B reference-identity cosine is about 0.031, so B is genuinely distinct;
- counterfactual supervision is applied to about 42–44% of logged samples;
- connector-up, connector-down, gate, null memory, and reference K/V all receive
  finite nonzero gradients after the zero-initialized connector opens.

Selected windowed means show limited optimization progress:

| Metric | 0–2k | 2k–4k | Interpretation |
|---|---:|---:|---|
| Total loss | 0.2273 | 0.2315 | no downward trend |
| Counterfactual absolute-ID loss | 0.9579 | 0.9385 | slight improvement |
| Counterfactual directional loss | 0.5750 | 0.5410 | slight improvement |
| Directional gain, B minus A | -0.7033 | -0.6775 | still strongly A-dominant |
| Similarity to matched A | 0.7454 | 0.7389 | remains high |
| Similarity to wrong/reference B | 0.0421 | 0.0615 | remains very low |
| `sa_ref` norm | 4.46 | 9.94 | parameters are moving strongly |
| Cap excess | 0.0183 | 0.0548 | raw residual increasingly exceeds cap |
| Null residual | 1.29e-5 | 7.10e-6 | null suppression improves |

The supervision is not dead: the training directional gain becomes about 0.026
less negative, and B similarity rises modestly. But the decoded wrong-reference
row remains overwhelmingly closer to A than B, and this modest training change
does not produce positive inference causality.

At 4k the validation gate is only about `0.252` (near its `0.25`
initialization), while the causal test reports mean cap fractions near 0.58–0.60
for scale-1 PPR variants and many individual sites hit the 0.15 cap. The branch
is therefore not simply too weak: substantial raw authority is already being
clipped for safety while its semantic direction remains unresolved.

## Comparison with NN4

NN5a is directionally less bad than NN4's 4k RealVis scale-4 result:

- NN4: mean gain `-0.00609`, CI entirely below zero, 43.2% positive;
- NN5a: mean gain `-0.000684`, CI crosses zero, 49.0% positive.

This suggests that fixed-target counterfactual supervision is the right type of
objective. It has moved the result from significantly wrong direction toward
zero. However, the runs use different diagnostic scales, so the magnitude is
not a strict apples-to-apples comparison. NN5a still fails every continuation
criterion that matters: positive mean, positive lower confidence bound, at
least 60% positive samples, cross-noise stability, and visible identity motion.

NN5a's normal validation is also more changed than NN4's, but its 4k identity
similarity is lower. Thus the counterfactual branch gained output influence
without yet gaining useful identity ownership.

## Should NN5a train longer or use more GPUs?

No unchanged continuation is recommended. More GPUs would improve throughput
or batch statistics but would not change the missing representation. The
normal metric has already worsened by 2k, the 4k causal confidence interval
still contains zero, noise can reverse the direction, and raw residual growth
is being absorbed by the cap.

Keep the 4k checkpoint as evidence. No additional checkpoint download is
needed to diagnose branch liveness or training health. An offline scale-2 test
could document sensitivity, but it should not be used to rescue or approve
NN5a: stronger NN4 residuals already demonstrated that magnitude can expose
generic edits without creating identity transfer.

## Recommended next training run

### Priority: NN5b clean identity-token lane

Run the already specified NN5b architecture next. It keeps all successful
parts of NN5a:

- ordinary target self-attention plus target-Q/reference-KV branched attention;
- counterfactual matched-A/wrong-B supervision with exact target/noise pairing;
- learned-null subtraction and zero-initialized additive connector;
- `up_blocks.0.attn1` only, no branched CA;
- soft face-core routing, 0.15 cap, and exact PhotoMaker output outside the core;
- neutral reference text, paired CFG noise, no pose adaptation, and no CA mix.

Its one essential representation change is to inject the two clean unpooled
PhotoMaker-V2 identity tokens derived from the spatial reference. The tensor
diagnostic explains why this is appropriate: the spatial route sees content,
but the output lacks a stable semantic identity direction. The clean token lane
adds identity semantics without abandoning the branched-attention mechanism or
the safe residual envelope.

Even if the NN5b job is configured for 30k, evaluate checkpoints at 2k and 4k
with the same RealVis, residual-scale-1, five-condition test. Continue only if:

1. mean directional gain toward R2 is positive;
2. the bootstrap lower bound is above zero;
3. at least 60% of samples are positive, without one identity dominating;
4. N1 and N2 have the same positive sign;
5. reference-image effects clearly exceed reference-noise effects;
6. matched R1 preserves A near PM0 while R2 visibly moves toward B;
7. face detection remains 100% and attachment/seam/landmark safety remains near
   the current level.

Do not approve NN5b based only on more visible differences or a falling training
loss.

### Contingency if NN5b also fails

The next architectural experiment should separate identity and spatial
candidates instead of averaging them before one connector:

```text
delta_id      = D_id(C_identity_tokens - C_identity_null)
delta_spatial = D_spatial(C_packed_reference - C_spatial_null)
delta         = gate_id * delta_id + gate_spatial * delta_spatial
```

Use independent zero-initialized connector-up projections, gates, diagnostics,
and conservative caps. Apply counterfactual identity supervision directly to
the identity lane; keep the spatial lane low-authority for local texture and
geometry. This would expose whether the current 50/50 pre-connector fusion lets
generic spatial variation dilute the clean identity signal. It is a better
follow-up than increasing scale, adding U-Net sites, or re-enabling cross-
attention.

## Final decision

NN5a answers its intended question: a correct counterfactual objective alone is
not sufficient for the current packed spatial representation. It improves the
directional failure from clearly negative toward chance, while preserving the
safe branched-attention geometry, but it does not establish reference identity
control. Stop NN5a at 4k and use NN5b's clean identity-token representation as
the next gated training experiment.
