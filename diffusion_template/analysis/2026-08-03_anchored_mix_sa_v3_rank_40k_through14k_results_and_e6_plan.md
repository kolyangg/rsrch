# Anchored mix SA-v3 rank long run through 14k: no identity promotion and a hard-routed BA-v4 plan

**Date:** 3 August 2026  
**Run:** `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1`  
**Immutable Comet ID:** `f5b5a7054e854137abe53c47f34ebae0`  
**Comet project:** `jul-comet-large-testing-tr`  
**Local result:**
[`comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1/`](../comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1/)  
**Historical comparison requested:** `rhca_big_celebs_sameid_40k_full96_r1`,
Comet `569cc685ff9144f5a9b42bf70e14e040`  
**Clean hard-routing comparison:**
`rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1`, Comet
`700240d8f90b48cfa2cc16f8ff2886b6`  
**Evidence cutoff:** complete aggregate validation metrics through optimizer
step 14,000; complete fixed-96 images at steps 0, 2,000, and 14,000; training
telemetry through step 15,450 in the downloaded console/metric record

The run name has a 40k ceiling, but the supplied local evidence is not a
complete 40k result. This report deliberately calls it the **through-14k
result**. Later checkpoints must not be folded into the conclusion until their
immutable metrics and full-96 images are exported and audited.

## Executive verdict

There is **no convincing identity improvement** in the current result. The
only identity value above initialization is step 6k, and the difference is
only `0.002826`:

```text
step 0 identity       0.494456
best identity, 6k     0.497282  (+0.57% relative; +0.002826 absolute)
step 14k identity     0.447258  (-0.047198 from initialization)
```

That tiny 6k excursion is not a promotion, especially because TOPIQ-Face p10
is `0.584245` there versus `0.622961` at initialization. Identity then falls
at every measured gate from 6k through 14k. Meanwhile text similarity, generic
TOPIQ, TOPIQ-Face mean, and MANIQA improve or remain healthy. The model is
becoming a better-looking, more prompt-responsive generator while becoming
less similar to the requested identities.

The run is not broken and the BA route is not dead:

- exact trainable ownership remains `414 tensors / 10,567,818 FP32 parameters`;
- optimizer membership is `414/414`;
- the fixed-96 panel completes with `96/96` face detection and coverage at
  every reported gate;
- the correct spatial reference has a positive training-time advantage over a
  shuffled spatial reference;
- actual BA contribution remains roughly `0.41x` native attention RMS late in
  training;
- all reviewed step-14k images differ visibly from step zero while retaining
  coherent pose and composition.

The important failure is therefore **alignment and route commitment**, not
installation or raw rank. The learned mean reference mix falls approximately
from `.50` to `.35`, while the native coefficient rises from `.50` to about
`.65`. The model is discovering a lower-reference solution even though the
spatial branch remains causally active. The differentiable rank objective does
not prevent that: its margin signal remains small, and it can improve by
making the wrong-reference prediction worse rather than making the production
prediction better.

The concern about PhotoMaker anchoring is consequently well founded, with one
qualification:

```text
identical to plain PhotoMaker                      no
strongly protected by a frozen PhotoMaker/native path  yes
learning to shift weight back toward that path         yes
explicit reference BA route active                    yes
explicit reference BA route improving identity         no evidence
```

### Should the next experiment start from the old `sameid` setup?

**Use its architectural lesson, but do not replay or resume it literally.**

Verified from the resolved run record and historical processor: there was no
native/reference face-attention mix in
`rhca_big_celebs_sameid_40k_full96_r1`. Its resolved
`branched_attn_weight_mode` was `noise_and_ref`, `pose_adapt_ratio` was `0.0`,
and its face merge was:

```python
merged = hidden_bg * (1 - mask) + hidden_face * mask * 1.0
```

Thus the target face used target queries with reference-only K/V; native
attention supplied the non-face region. The standard transformer residual and
PhotoMaker token conditioning still existed, but there was no alpha blending
of native and reference face self-attention messages.

The old run has the clearest positive identity-learning trajectory:
`0.3063 -> 0.3751` by 14k and a peak of `0.3817` at 18k. It demonstrates that
a reference-dominant/hard face route with more target-side adaptation can
learn identity over several thousand steps. That is valuable evidence.

However, it is not a safe or clean starting checkpoint:

- a swallowed processor-installation exception left about `171.29M`
  requires-grad parameters, not the advertised BA-only state;
- the pretrained PhotoMaker adapter and a generic U-Net adapter trained in
  addition to BA;
- the saved checkpoint omitted part of the live trained PhotoMaker state;
- validation used `legacy_full_copy`, not the current `validation_native`
  contract;
- routing, data order, timestep policy, precision, and trainable ownership all
  differ from the current run.

Resuming that checkpoint would restore neither the exact live model nor a
scientifically attributable BA mechanism. Re-enabling the same broad
trainables might improve the aggregate identity score again, but it would not
show that branched attention caused the improvement.

The recommended successor is therefore a **modern clean reconstruction of the
useful old inductive bias**:

1. remove native/reference interpolation from the target face route;
2. use native target attention only outside the target-face mask;
3. add a branch-only target-query LoRA so target queries can learn how to
   retrieve from reference K/V;
4. disable the current two-sided rank loss for the first architecture arm;
5. retain explicit target-Q/reference-KV branched attention, true reference
   key masking, face-local merging, CA-off, `pose_adapt_ratio=0`, and
   `ca_mixing_for_face=false`;
6. keep every old v1/v2/v3 behavior available behind its existing selector,
   but give v4 no mix parameter, floor, schedule, or alpha override.

This is called **query-adaptive hard-routed BA-v4** below. It follows the same
principle as the project's `pose_adapt_ratio=0` rule: do not add a soft path
that can hide whether the intended reference-conditioned mechanism works.

## Evidence integrity and limits

The supplied package contains:

- exactly 96 images for step 0, 96 for step 2k, and 96 for step 14k;
- complete identity, text, and seven compact face-quality curves at every 2k
  gate from 0 through 14k;
- complete sampled routing, causal, gradient, and loss diagnostics through
  step 15,450;
- exact requested/resolved/manifest step agreement at the three exported
  image gates;
- no export warning, export error, traceback, CUDA OOM, or non-finite loss;
- immutable key `f5b5a7054e854137abe53c47f34ebae0` in each export;
- 96/96 byte-identical step-zero images versus the short E4 control;
- 0/96 byte-identical step-2k images versus the short E4 run, but nearly
  identical aggregate behavior (`0.464927` versus `0.463905` identity), which
  is normal run-level numerical variation rather than a configuration change.

The local export does not include per-image identity scores. Aggregate identity
deltas are exact for the deterministic panel, but no per-image confidence
interval can be calculated from this package. Visual conclusions below come
from reviewing all 96 images at steps 0, 2k, and 14k. Only aggregate metrics,
not full image panels, are locally present at steps 4k through 12k.

No conclusion here assumes the eventual 16k–40k trajectory. The monotonic
identity decline over four consecutive gates from 6k to 14k is already enough
to reject the current checkpoint family as an identity promotion through this
evidence cutoff; later recovery would be new evidence.

## 1. What this training behaviour uncovers

### 1.1 The long run resolves the 2k ambiguity: the early dip briefly recovers, then fails

The complete fixed-96 curve is:

| Step | Identity | Text | TOPIQ-Face mean | TOPIQ-Face p10 | TOPIQ | MUSIQ | MANIQA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.494456 | 25.8003 | 0.717769 | **0.622961** | 0.587789 | 73.2048 | 0.633840 |
| 2k | 0.464927 | 27.0057 | 0.717318 | 0.569454 | 0.603523 | 72.9566 | 0.634492 |
| 4k | 0.493786 | 26.9105 | 0.710884 | 0.581575 | 0.596785 | 72.3937 | 0.629211 |
| 6k | **0.497282** | 26.8703 | 0.724689 | 0.584245 | 0.609202 | 73.2542 | 0.634949 |
| 8k | 0.481687 | 27.0579 | 0.718692 | 0.589309 | 0.603668 | 73.0248 | 0.634642 |
| 10k | 0.474720 | 27.0923 | 0.714803 | 0.580304 | 0.605124 | 73.0191 | 0.635176 |
| 12k | 0.460783 | **27.3924** | 0.722715 | 0.582357 | 0.612119 | **73.3409** | 0.637072 |
| 14k | **0.447258** | 27.2728 | **0.724949** | 0.590998 | **0.612688** | 73.2262 | **0.637192** |

Face detection and TOPIQ-Face coverage are `1.0` at every gate.

The earlier report correctly avoided rejecting E4 at only 2k because both
hard-routing histories recovered after an early dip. E5-L40 now supplies the
missing long-horizon evidence:

- 2k to 4k: strong identity recovery;
- 4k to 6k: a very small new high;
- 6k to 14k: four consecutive declines;
- 10k to 14k: quality/text remain healthy while identity continues down.

Thus, “2k was too early” was true, but “more of the same training will make
v3+rank a better identity model” is not supported. The current trajectory has
already shown its turn and is moving in the wrong direction.

### 1.2 The apparent gain is a quality/prompt trade, not a BA identity gain

From initialization to 14k:

| Metric | Change |
|---|---:|
| Identity similarity | **-0.047198** |
| Text similarity | **+1.47249** |
| TOPIQ-Face mean | +0.007180 |
| TOPIQ-Face p10 | **-0.031963** |
| Generic TOPIQ | +0.024899 |
| MUSIQ | +0.0214 |
| MANIQA | +0.003352 |

The model produces clean, detected faces and better prompt/rendering scores,
but identity similarity and the weak-face tail are worse. A developer should
not describe the 6k point as an improvement without stating that its identity
gain is only `.0028` and its p10 is `.0387` below initialization.

This separation is important because aggregate face IQA can reward sharpness,
contrast, expression, and artifact-free rendering without rewarding the
correct person's morphology. The step-14k panels visibly show exactly that:
faces are usually polished and structurally attached, yet the changes are not
consistently toward the named identity.

### 1.3 The branch is active, but the optimizer is learning route evasion

The phase-averaged routing telemetry is:

| Training window | Mix | Ref/native RMS | Contribution/native RMS | Ref/native cosine | Merged/native RMS |
|---|---:|---:|---:|---:|---:|
| 0–2k | 0.478 | 1.001 | 0.436 | 0.561 | 0.883 |
| 2–4k | 0.444 | 1.044 | 0.463 | 0.476 | 0.877 |
| 4–6k | 0.417 | 1.058 | 0.456 | 0.444 | 0.877 |
| 6–8k | 0.389 | 1.055 | 0.434 | 0.424 | 0.874 |
| 8–10k | 0.374 | 1.052 | 0.412 | 0.431 | 0.877 |
| 10–12k | 0.362 | 1.074 | 0.412 | 0.418 | 0.883 |
| 12–14k | **0.353** | 1.077 | 0.408 | 0.407 | 0.882 |
| 14–15.45k | **0.345** | 1.117 | 0.419 | 0.396 | 0.897 |

This rules out a dead reference branch: a `0.41x` contribution is substantial,
reference/native cosine has rotated far from one, and the correct/shuffled
prediction difference remains measurable. But the learned interpolation
weight retreats steadily. With

```text
output_face = (1 - alpha) * native + alpha * reference
```

the average native coefficient grows from approximately `.50` to `.65`. The
`.25` floor permits this behavior. The current loss has no production-quality
term that says a minimum amount of correct spatial reference information must
remain useful. It only asks the matched diffusion prediction to fit the target
and, on sampled batches, asks matched and shuffled errors to separate.

The simplest interpretation is an optimization shortcut: PhotoMaker identity
tokens and frozen native target attention already solve much of the task, so
the model improves text/rendering while reducing exposure to the harder
spatial-reference route.

### 1.4 Causal separation is not the same as useful identity conditioning

The shuffle-conditional diagnostics remain positive, but they do not track
validation identity:

| Window | Conditional relative gap | Prediction delta / prediction RMS | Rank loss | Total loss |
|---|---:|---:|---:|---:|
| 0–2k | 1.300% | 8.09% | 0.00453 | 0.4114 |
| 2–4k | 0.879% | 6.04% | 0.00556 | 0.3977 |
| 4–6k | 0.944% | 5.94% | 0.00607 | 0.4144 |
| 6–8k | 0.866% | 5.37% | 0.00573 | 0.3990 |
| 8–10k | 1.171% | 5.62% | 0.00529 | 0.4034 |
| 10–12k | 1.128% | 5.70% | 0.00509 | 0.4074 |
| 12–14k | 1.341% | 5.72% | 0.00499 | 0.3908 |
| 14–15.45k | 1.471% | 5.89% | 0.00467 | 0.4234 |

The desired relative margin is 2%, so the phase means never satisfy it. More
importantly, the conditional gap recovers from about `.87%` to `1.47%` while
fixed-panel identity falls. The current objective can therefore make the
network more sensitive to which spatial reference is present without making
the correct production image look more like the identity.

This is direct evidence against increasing the existing rank-loss weight. It
is also evidence that “branched attention is working” must be split into two
claims:

1. **mechanical/causal use:** yes;
2. **useful identity improvement:** no.

### 1.5 The mix parameters are saturating toward a low-reference solution

The phase-mean mix-gradient norm falls from about `0.00404` to `0.00047`, an
approximately nine-fold reduction. Reference-K/V and output gradients remain
finite, but the route-control gradients are approaching a stable low-mix
solution. Waiting for the same learned gate to spontaneously return to a
reference-dominant regime is not a strong plan.

Increasing K/V or output rank would not address this. A rank-64 branch behind
an alpha that the model is learning to suppress is simply a larger route that
the optimizer can continue to avoid.

### 1.6 Linear interpolation is still attenuating the attention message

RMS matching constrains the endpoint magnitudes, not their directions. As the
reference message rotates away from the native message, their interpolation
partially cancels. The logged merged/native RMS stays around `.87–.90`, meaning
the face attention message is commonly 10–13% weaker than native even though
the reference endpoint itself is RMS-matched.

This is a plausible contributor to the persistent p10 deficit. It is not the
main explanation for declining identity—the mix retreat is more direct—but it
is a quality-risk that becomes more important if the next run commits to a
higher alpha.

### 1.7 Full-panel visual findings

Review of all 96 step-0/2k/14k triplets finds:

- pose, body placement, background, clothing, and the main prompt composition
  remain very stable;
- step 2k and especially step 14k change mouth shape, eye opening, wrinkles,
  skin rendering, hair, and expression intensity;
- angry, crying, laughing, night-ride, kickboxing, and skiing groups often
  become more emphatic and visually polished;
- there is no broad pasted-face, detached-face, double-face, or body-collapse
  failure;
- identity changes are mixed: some individual faces gain a recognizable trait,
  while others become more generic, older/younger, or expression-dominated;
- no identity or prompt family shows a repeated enough morphology improvement
  to contradict the aggregate identity decline;
- the old and clean hard-routing 20k images move facial morphology more
  forcefully, but also show more strained expressions and local face/eye
  integration problems.

These images support the architectural interpretation. V3's frozen native
path is doing valuable structural work, but it also makes small, safe,
PhotoMaker-like refinements an easier optimization target than learning a
strong spatial identity correspondence.

### 1.8 Comparison with the old and clean hard routes

Identity trajectories through the comparable early horizon are:

| Run | Step 0 | 2k | 4k | 6k | 8k | 10k | 12k | 14k | Best observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current anchored v3 + rank | 0.4943 | 0.4649 | 0.4938 | **0.4973** | 0.4817 | 0.4747 | 0.4608 | **0.4473** | 0.4973 at 6k |
| Historical hard + broad unintended trainables | 0.3063 | 0.2841 | 0.3138 | 0.3095 | 0.3609 | 0.3723 | 0.3701 | **0.3751** | 0.3817 at 18k |
| Clean hard BA32 | 0.3063 | 0.1519 | 0.2744 | 0.2668 | 0.3192 | 0.3039 | 0.3258 | **0.3290** | 0.3347 at 18k |

Within-run step-0-to-14k changes are more informative than absolute levels:

```text
current anchored v3 + rank    -0.0472
historical hard/broad         +0.0687
clean hard BA32               +0.0227
```

The current model's much higher absolute initialization mainly reflects its
native/PhotoMaker anchor and different validation processor semantics. It does
not prove that current BA is better. Conversely, the old run's relative gain
does not prove hard BA caused all of it because its generic and PhotoMaker
adapters also trained.

The clean hard arm is particularly useful: despite a severe 2k dip, it gained
about `.0227` by 14k without the 171M ownership bug. This supports retaining a
clean, reference-dominant arm in the next ladder. Its `legacy_full_copy`
validation still prevents treating it as a direct absolute control.

## 2. High-priority issues in the current code

### P0 — The trainable mix is an unregularized escape route

In
[`anchored_mix_sa_processor_v3.py`](../src/model/photomaker_branched/anchored_mix_sa_processor_v3.py),
alpha is learned independently at every processor from a global logit plus
timestep and log-face-area terms:

```python
logits = mix_logit + mix_t * (2 * progress - 1)
logits = logits + mix_area * log(face_area)
mix = floor + (max - floor) * sigmoid(logits)
```

Nothing penalizes a low reference budget. The `.25` floor was designed to
prevent complete collapse, but the long run shows it is low enough for the
optimizer to re-anchor substantially to native attention.

Required repair:

- preserve v3 unchanged for exact replay;
- give v4 a hard target-face route with no trainable or fixed interpolation;
- use native target attention outside the face and reference BA inside it;
- remove `mix_logit`, `mix_t`, and `mix_area` from the v4 processor, optimizer,
  checkpoint, and telemetry contract;
- record `face_fusion_mode=hard_reference_replace` and `face_branch_scale=1.0`
  explicitly in the v4 manifest.

The historical processor already used this form:

```python
merged = hidden_bg * (1 - target_mask) + hidden_face * target_mask * 1.0
```

There was no alpha interpolation between native and reference face-attention
messages. With `pose_adapt_ratio=0`, the face K/V source was reference-only.
The new v4 should restore that invariant cleanly rather than attempting to
regularize a mechanism whose main function is to soften or hide it.

### P0 — The rank loss can improve by degrading the wrong branch

The current `differentiable_rank` mode keeps both correct and shuffled
predictions in the graph:

```python
relative_gap = (wrong_face - correct_face) / stopgrad(correct_face)
loss_rank = relu(margin - relative_gap)
```

Inside the hinge, gradients can lower `correct_face`, raise `wrong_face`, or
do both. Only the first direction improves the production output. The observed
late increase in reference gap together with falling validation identity is
consistent with this loophole.

Implement a reversible `correct_only_relative_rank` mode, but leave its weight
zero in the first v4 architecture run:

```diff
 if mode == "correct_only_relative_rank":
+    correct_i = _masked_face_mse_per_sample(model_pred, target, face_bbox)
+    with torch.no_grad():
+        wrong_i = _masked_face_mse_per_sample(
+            pred_wrong_spatial_ref, target, face_bbox
+        )
+    relative_gap_i = (
+        (wrong_i - correct_i)
+        / correct_i.detach().clamp_min(1.0e-6)
+    )
+    reference_causal = F.relu(
+        reference_relative_margin - relative_gap_i
+    ).mean()
```

The wrong-reference U-Net forward must also run under `torch.no_grad()` for
this mode. Do not merely detach `wrong_face` after building a full graph.

### P0 — The rank hinge is batch-aggregated

`_masked_face_mse` reduces across the entire batch before the hinge. With
batch size two, one easy example can hide a failure on the other example.
Implement the per-sample helper used above and reduce after applying the
hinge:

```diff
-correct = _masked_face_mse(model_pred, target, face_bbox)
-wrong = _masked_face_mse(pred_wrong, target, face_bbox)
-loss_rank = relu(margin - (wrong - correct) / correct.detach())
+correct_i = _masked_face_mse_per_sample(model_pred, target, face_bbox)
+wrong_i = _masked_face_mse_per_sample(pred_wrong, target, face_bbox)
+gap_i = (wrong_i - correct_i) / correct_i.detach().clamp_min(1e-6)
+loss_rank = relu(margin - gap_i).mean()
```

Log conditional `gap_mean`, `gap_p10`, `margin_satisfied_fraction`, and
`correct_improvement` separately. Preserve the current aggregate mode for
exact E4/E5 replay.

### P0 — The reference branch cannot adapt its target query

V3 uses the same frozen target query for native target attention and reference
retrieval:

```python
q_target = attn.to_q(target_hidden)       # frozen
native = Attn(q_target, target_k, target_v)
reference = Attn(q_target, ref_k, ref_v)
```

Reference K/V and output projections can rotate, but the branch cannot learn
a query space that maps target pose/location features to identity-bearing
reference features. This is the highest-value bounded capacity missing from
the current architecture.

The old `noise_and_ref` design had target/noise-side adaptation, although it
also had many confounds. Recover only the relevant degree: a branch-local
target-Q LoRA. The native Q/K/V path must remain frozen.

### P1 — No held-out hard-route causality diagnostic was completed before long training

Training-time shuffled error is not enough to establish held-out identity
value. The current v3 checkpoint can still be evaluated diagnostically at
alpha one to estimate whether its learned reference projections are
structurally usable without native face mixing:

| Arm | Spatial reference | Alpha | Purpose |
|---|---|---:|---|
| `alpha0` | matched | 0 | exact native/PhotoMaker endpoint |
| `learned` | matched | checkpoint value | production checkpoint |
| `alpha100` | matched | 1.0 | modern hard-route upper bound |
| `shuffle100` | wrong spatial identity, correct PM identity | 1.0 | held-out spatial causality |
| `zero100` | zero spatial reference, correct PM identity | 1.0 | content dependence |

Each arm needs 96 images, identity/text/seven IQA metrics, per-image results,
one immutable Comet key, and an explicit checkpoint SHA-256. If the 6k
checkpoint is unavailable, use 14k first; do not silently substitute a
different checkpoint.

This diagnostic is useful but is no longer a prerequisite for v4: the user has
selected the no-mix architecture on mechanistic grounds. A v3 checkpoint
trained while alpha was retreating is also not a fair substitute for training
a hard route from initialization.

### P2 — Post-mix attenuation has no guard in v3

Add a defaults-off, detached, clipped norm guard:

```diff
 mixed_face = native_out + mix * (reference_out - native_out)
+if self.post_mix_rms_preserve:
+    native_rms = self._masked_rms(native_out, target_mask)
+    mixed_rms = self._masked_rms(mixed_face, target_mask)
+    correction = (native_rms / mixed_rms.clamp_min(1e-6)).clamp(
+        self.post_mix_rms_clip_min,
+        self.post_mix_rms_clip_max,
+    ).detach()
+    mixed_face = mixed_face * correction[:, None, None].to(mixed_face.dtype)
 target_out = native_out + target_mask * (mixed_face - native_out)
```

Use `[0.90, 1.10]` only for a future v3 ablation. V4 has no interpolation, so
there is no post-mix cancellation to normalize and this guard must not be part
of the first hard-route experiment.

### P2 — The uncentered face-area term is hard to audit in v3

`mix_area * log(area)` is always driven by a negative log-area value and its
scale depends on the face-size distribution. Current telemetry logs only the
final mix, so a developer cannot determine whether the global logit, timestep
slope, or face-area slope caused the retreat.

If v3 is studied again, either keep `mix_face_area=false` or center the feature
using an explicitly recorded reference area:

```diff
-area_feature = torch.log(area.clamp_min(1e-4))
+area_feature = torch.log(area.clamp_min(1e-4)) - self.mix_log_area_center
 logits = logits + self.mix_area * area_feature
```

V4 contains no mix terms, so this entire ambiguity disappears from the new
experiment.

### P1 — Current telemetry cannot tell whether the correct output improved

The logger exposes correct-versus-wrong separation, not the two production
directions independently. Add:

```text
correct_face_error_conditional
wrong_face_error_conditional
correct_face_error_delta_vs_unshuffled_baseline
gap_p10_conditional
margin_satisfied_fraction_conditional
branch_query_delta_rms_ratio
```

Keep `mix_near_floor_fraction` only as a v3 diagnostic; v4 must have no mix
series at all.

This makes a later correct-only objective auditable and prevents another run
from being declared successful solely because its wrong branch deteriorated.

## 3. Architectural improvements and next experiments in priority order

## Priority 0 — E6-H: query-adaptive hard-routed BA-v4

### Scientific question

Can the old run's identity-learning behavior be reproduced with an explicit
hard reference face route, branch-local capacity, strict ownership, and
complete checkpoints—without its accidental global/PhotoMaker trainables?

### Routing invariant

```text
background path:
    Qbg = frozen_to_q(target background)
    Kbg, Vbg = frozen full-target K/V
    B = frozen_to_out(Attn(Qbg, Kbg, Vbg))

target-face reference BA path:
    Qr = frozen_to_q(target) + LoRA_Q_branch(target)
    Kr = frozen_to_k(reference) + LoRA_K_ref(reference)
    Vr = frozen_to_v(reference) + LoRA_V_ref(reference)
    F = frozen_to_out(Attn(Qr, Kr, Vr; true_reference_key_mask))
        + LoRA_out_ref(...)

hard spatial merge:
    Y = (1 - target_face_mask) * B + target_face_mask * F
```

There is no alpha, gate, native/reference face interpolation, schedule, floor,
or override. Target queries consume explicit reference K/V inside the face.
No target K/V is inserted into the face reference path.

The standard transformer residual connection still exists, and PhotoMaker
identity-token cross-attention remains present. “No mix” here means no native
face **self-attention message** is blended into the BA face message. It does
not pretend the entire U-Net has ceased to be PhotoMaker-conditioned.

### Key processor diff

Create `query_adaptive_hard_sa_processor_v4.py` as a separate versioned
processor. Reuse mask/projection helpers, but do not inherit v3 mix parameters:

```diff
+class QueryAdaptiveHardBranchedSelfAttnProcessorV4(nn.Module):
+    architecture_version = "query_adaptive_hard_sa_v4"
+
+    def __init__(self, ..., branch_q_rank=16, ref_kv_rank=32,
+                 output_rank=32, ...):
+        self.branch_q_rank = int(branch_q_rank)
+        self.branch_to_q = None
+        self.ref_to_k = None
+        self.ref_to_v = None
+        self.ref_out = ResidualLoRALinear(..., rank=output_rank)
+
+    def init_from_attention(self, attn):
+        self.branch_to_q = _clone_effective_linear(
+            attn.to_q,
+            kind="lora",
+            rank=self.branch_q_rank,
+            trainable_dtype=self.trainable_dtype,
+        )
+        self.ref_to_k = _clone_effective_linear(attn.to_k, kind="lora", ...)
+        self.ref_to_v = _clone_effective_linear(attn.to_v, kind="lora", ...)
+
+    def named_ba_trainables(self):
+        for name, parameter in self.branch_to_q.named_parameters():
+            yield f"branch_to_q.{name}", parameter, "ref_query"
+        # yield ref_to_k/ref_to_v as ref_kv and ref_out as ref_output

+q_reference_branch = self._reshape_heads(
+    self.branch_to_q(target_hidden), heads
+)
 face_message = F.scaled_dot_product_attention(
     q_reference_branch,
     k_reference,
     v_reference,
     attn_mask=key_bias,
 )
+face_out = frozen_output_projection(face_message) + self.ref_out(face_message)
+target_out = native_background_out * (1.0 - target_mask)
+target_out = target_out + face_out * target_mask
```

The cloned base equals effective frozen target Q and its LoRA-B matrix starts
at zero. Therefore v4 begins with the effective pretrained target query, but
the face still uses reference-only K/V from update zero. Step-zero images are
expected to differ from anchored v3 and should resemble a clean hard-route
initialization. They must not be forced to match PhotoMaker/v3.

Start with query rank 16. Across 46 existing sites it adds 92 trainable tensors
and approximately 1.76M parameters relative to reference-K/V/output-only
capacity. Removing v3's 138 mix tensors gives an expected v4 ownership near
368 tensors and 12.33M parameters. The implementation must derive and assert
the exact count rather than hard-coding either estimate.

### Hard-route config

The first training arm should use:

```yaml
model:
  ba_architecture_version: query_adaptive_hard_sa_v4
  ba_branch_q_rank: 16
  ba_ref_kv_rank: 32
  ba_output_rank: 32
  ba_face_fusion_mode: hard_reference_replace
  ba_face_branch_scale: 1.0
  ba_reference_rms_match: false
  ba_reference_loss_mode: detached_diagnostic
  ba_spatial_reference_shuffle_probability: 0.25

loss_function:
  reference_mode: detached_diagnostic
  reference_weight: 0.0
  reference_margin: 0.0
  reference_relative_margin: 0.0

ba_ref_query_lr: 5.0e-5
ba_ref_kv_lr: 5.0e-5
ba_ref_output_lr: 1.0e-4

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false

trainer:
  epoch_len: 2000
  n_epochs: 10
  validation_interval_steps: 2000
  save_period: 1

weights_only_save_period: 1
```

This is a 20k first run, with scientific decisions at 6k, 8k, 12k, and 20k.
It is long enough to cover both historical recovery curves without spending a
full 40k on another demonstrably falling route. If an unattended 40k ceiling
is operationally preferred, preserve every 2k checkpoint and treat 20k as a
mandatory review boundary; do not select the last checkpoint automatically.

### Why disable the rank objective in E6-H?

The current two-sided objective has now failed the long-horizon alignment test.
Leaving it on would confound whether branch-Q and hard routing work. The
25% shuffled pass remains as a detached diagnostic, so causal use is still
measured without allowing the negative branch to shape the model.

### Promotion criteria

At startup:

- step-zero is generated and preserved as the new hard-route baseline; it is
  not required or expected to match v3/PhotoMaker;
- native target Q/K/V and native output projection are absent from the
  optimizer;
- exact allowlist equals exact optimizer membership;
- every query/K/V/output trainable is FP32 and checkpointed;
- no parameter or config key with role `mix` is active in v4;
- processor map, architecture version, ranks, sites, masks, and hard fusion are
  identical in training and validation.

At 2k:

- use as a safety/reproducibility gate, not a final score gate;
- all first three batches and all validation metrics finite;
- no broad anatomy regression;
- branch-query delta has nonzero gradient and a finite nonzero RMS after
  training starts;
- matched correct-reference gap remains positive.

At 6k/8k:

- identity must turn upward from the 2k trough;
- by 8k, prefer recovery to at least the new hard-route step-zero identity;
- TOPIQ-Face p10 should recover toward at least `.61`;
- detection and coverage remain `96/96`;
- matched fixed-checkpoint output beats shuffled/zero spatial references and a
  separately generated validation-native/PhotoMaker control.

At 12k/20k:

- require identity at least `.03` above the new hard-route initialization;
- require no four-gate monotonic identity decline;
- require p10 and TOPIQ-Face mean not below the new hard-route initialization;
- select the best coherent intermediate checkpoint, not necessarily 20k.

Those are architecture-promotion gates, not final project success. A final
model promotion still requires the matched hard-BA checkpoint to beat the
fixed validation-native/PhotoMaker control on identity without losing the
face-quality tail.

Failure is: route remains causal but identity stays at/below the native control
or declines while prompt/IQA rises. That would show the missing factor is not
query capacity alone.

## Priority 1 — D3-H: optional current-checkpoint hard-route diagnostic

For diagnosis only, validate E5's 6k or 14k checkpoint with alpha one,
matched/shuffled/zero spatial references, and the native alpha-zero control.
This can quantify what the already-trained v3 projections do when forced, but
it must not gate or replace E6-H. Those projections learned under a retreating
mix and are not representative of hard-route training from initialization.

## Priority 2 — E7-R: per-sample correct-only ranking

Only after E6-H demonstrates useful identity learning, test the corrected
objective:

```yaml
model:
  ba_reference_loss_mode: correct_only_relative_rank
  ba_spatial_reference_shuffle_probability: 0.50

loss_function:
  reference_mode: correct_only_relative_rank
  reference_weight: 0.05
  reference_relative_margin: 0.02
```

Use a lower `.05` weight initially. Require the correct prediction error and
validation identity to improve together; a larger correct/wrong gap alone is
not a promotion.

## Priority 2 — E8-D: bounded PhotoMaker identity-conditioning dropout

If E6-H remains causally active but the native control is as good as matched
hard-BA inference, PhotoMaker token conditioning is probably still the
dominant identity shortcut. Add a training-only 10–15% dropout of PhotoMaker
identity-token contribution while keeping:

- prompt text;
- target/noise/timestep;
- correct spatial reference latent and mask;
- target-Q/reference-KV BA;
- full PhotoMaker conditioning at validation/inference.

Log normal and dropout batches separately and require the dropout batches to
prefer the correct spatial reference. This should be its own experiment. Do
not combine it with the first query-adapter run.

## Priority 3 — add one coarser block, then test branch-local rank

Only after a lower-capacity v4 arm improves identity:

1. add `down_blocks.2` to the existing `mid/up0/up1` sites;
2. test branch-Q rank 32;
3. test reference K/V and output rank 64;
4. keep all native/global/PhotoMaker adapters frozen and checkpoint-complete.

The current run proves rank 32 is enough to produce a strong branch output and
causal sensitivity. It does not prove rank is the current bottleneck. More
rank before fixing query alignment and route commitment is low priority.

## 4. Required model/optimizer/checkpoint integration for v4

The implementation must remain defaults-off and reversible.

### Model and processor factory

Add fields without changing shared defaults:

```yaml
ba_branch_q_rank: 16
ba_face_fusion_mode: hard_reference_replace
ba_face_branch_scale: 1.0
```

Then add `query_adaptive_hard_sa_v4` to:

- `PhotomakerBranchedLora.__init__` architecture validation;
- the strict processor factory in `branched_runtime.py`;
- `expected_branched_trainable_names`;
- `branched_trainable_role_groups`;
- telemetry collection;
- training-to-validation flag propagation;
- the standalone checkpoint evaluator.

Old `hard_replace_v1`, `residual_sa_v2`, and `anchored_mix_sa_v3` paths must
remain unchanged.

### Optimizer ownership

Add a dedicated role:

```diff
 role_lrs = {
     "ref_kv": config.ba_ref_kv_lr,
     "ref_output": config.ba_ref_output_lr,
+    "ref_query": config.ba_ref_query_lr,
 }
```

At startup, assert:

```text
requires_grad parameter IDs == processor-declared IDs == optimizer IDs
```

No native `attn.to_q/to_k/to_v/to_out`, generic U-Net LoRA, or PhotoMaker
adapter parameter may enter the optimizer.

### Schema-v2 checkpoint contract

Record and validate:

```text
architecture_version
branch_q_rank
ref_kv_rank
output_rank
face_fusion_mode = hard_reference_replace
face_branch_scale = 1.0
reference RMS mode
exact processor names
routing and mask semantics
pose_adapt_ratio and CA flags
trainable names/shapes/dtypes
```

Loading a v3 checkpoint as v4, a v4 rank-16 checkpoint as rank 32, or a mixed
fusion checkpoint as hard-routed v4 must fail closed unless an explicit
migration is implemented and recorded.

### Validation parity

The validation model must install v4 before loading its processor tensors and
must copy every architecture field from training. Required assertions:

- v4 update-zero processor output matches its independently implemented hard
  routing equation;
- no target native face-attention message enters the output inside the face
  mask;
- outside-target-mask output is unchanged;
- invalid reference keys receive no attention probability;
- target native Q/K/V remain frozen;
- `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, branched CA off.

## 5. Controlled experiment ladder

| Stage | Intervention | Horizon | Main question | Advance condition |
|---|---|---:|---|---|
| E6-H | Hard target-face reference BA + branch-Q rank16; no mix/rank loss | 20k, first decision 8k | Can the useful old hard-route behavior be reproduced cleanly? | Upward post-2k trajectory; ID >= own initialization at 8k and +.03 by 12–20k; matched > native/shuffle/zero |
| D3-H | Current v3 checkpoint forced alpha1 + native/shuffle/zero | optional validation | What do already-trained v3 projections do without the learned mix? | Diagnostic only; does not gate E6-H |
| E7-R | Per-sample correct-only rank | 8k | Can causal separation improve the correct output? | Correct error, ID, and margin improve together |
| E8-D | 10–15% PM identity dropout | 8k | Is PM conditioning shortcutting spatial identity? | Matched spatial reference becomes necessary and matched inference improves |
| E9 | Add down2, then branch-local rank | 8–20k | Is coarse geometry or low rank finally limiting? | Only after an aligned smaller v4 succeeds |

## 6. Verification checklist for implementation

For optional D3-H:

- exact checkpoint SHA-256 and schema-v2 manifest recorded;
- 96/96 fixed-panel inputs in every arm;
- PhotoMaker identity conditioning unchanged in shuffle/zero spatial arms;
- forced alpha one explicit in resolved config, output manifest, and Comet
  parameters;
- one fresh immutable Comet key per arm;
- all aggregate and per-image metrics complete before interpretation.

For E6-H:

- shared v1/v2/v3 defaults compose exactly as before;
- v4 selector is explicit and defaults off;
- step-zero fixed-96 hard-route baseline preserved; v3 parity is neither
  required nor expected;
- exact optimizer ownership and FP32 trainable audit;
- finite nonzero branch-query gradient and later query-delta RMS;
- no mix parameter, optimizer role, interpolation, floor, or hidden override;
- full-96 validation at step 0 and every 2k;
- immutable Comet record created at startup;
- all query/K/V/output tensors round-trip exactly;
- evaluator rejects architecture/rank/routing mismatches;
- matched/shuffle/zero hard-route validation and a separate native control work;
- no validation seed, prompt, reference, bbox, scheduler, CFG, inference-step,
  or metric drift.

## 7. What not to do next

- Do not call the `.0028` identity increase at 6k a meaningful improvement.
- Do not continue interpreting text/IQA gains as identity gains.
- Do not say the model is plain PhotoMaker; the branch is active and causal.
- Do not say the anchoring concern is false; the learned alpha directly shows
  increasing native reliance.
- Do not resume the historical `569cc...` checkpoint as though it contains the
  complete live model.
- Do not recreate its swallowed installation exception or 171M broad
  trainables.
- Do not increase current K/V/output rank while alpha is retreating.
- Do not increase the two-sided rank-loss weight.
- Do not reintroduce mix as a safety mechanism in v4; structural failures must
  remain visible and be fixed in the branch itself.
- Do not combine query adaptation, norm manipulation, corrected
  rank loss, PhotoMaker dropout, new layers, and rank 64 in one run.
- Do not select the final process step automatically; preserve and compare
  every 2k checkpoint.

## Final recommendation

The current run gives a clear answer through 14k: **it has not improved
identity**. It briefly returns to initialization at 4–6k, then sacrifices
identity while continuing to improve prompt response and image-quality means.
The BA route is real, but its learned mixing retreats from `.50` toward `.35`,
and the current rank objective produces causal separation that is not aligned
with identity quality.

Starting from the old `rhca_big_celebs_sameid_40k_full96_r1` setup makes sense
only at the level of inductive bias: stronger reference commitment and
target-query adaptation did show a real multi-thousand-step learning curve.
It does **not** make sense to resume or exactly replay its incomplete,
171M-parameter state.

Implement query-adaptive hard-routed BA-v4 directly: branch-only target-Q rank
16, frozen native target projections, reference K/V rank 32, hard reference
replacement inside the target-face self-attention route, no mix parameters,
no current rank loss, and the same strict scheduled/full-96 protocol. Native
attention remains outside the face, but the face BA path cannot silently
retreat toward it. This is the explicit modern test of the useful old setup.

The current-checkpoint alpha-one/shuffle/zero matrix remains useful as an
optional diagnostic, but it should not delay or determine the no-mix v4 design.
Any structural weakness exposed by hard routing must be fixed through query,
reference K/V, output, masks, or layer selection—not hidden by interpolating
PhotoMaker/native face attention back in.
