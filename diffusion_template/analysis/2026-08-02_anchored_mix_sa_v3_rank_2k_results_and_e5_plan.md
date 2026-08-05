# Anchored mix SA-v3 rank objective at 2k: informative early phase, no promotion, and E5 plan

**Date:** 2 August 2026  
**Run:** `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2`  
**Immutable Comet ID:** `f72ea55eb0af44828cd6511a15ba5933`  
**Comet project:** `jul-comet-large-testing-tr`  
**Local result:**
[`comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2/`](../comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2/)  
**Direct 2k control:** `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2`,
Comet `de23193eeac9433fa090bc009f10e752`  
**Historical long-run comparison:** `rhca_big_celebs_sameid_40k_full96_r1`,
Comet `569cc685ff9144f5a9b42bf70e14e040`  
**Evidence cutoff:** fixed-96 validation at optimizer steps 0 and 2,000;
training diagnostics through step 1,950

This report reviews E4, revises the earlier 2k decision rule using the actual
long-run evidence, and specifies the next controlled experiment. It separates
three questions that should not be conflated:

1. is the explicit spatial BA route active;
2. is the new ranking objective changing that route in the intended way; and
3. has the model had enough optimization steps to improve the fixed-96
   endpoint.

## Executive verdict

E4 is a **healthy, causally active BA run but not a 2k promotion**. The
differentiable matched-versus-shuffled objective produced a small improvement
in training-time reference separation, but it did not reach its 2% margin and
did not improve the validation endpoint. From step 0 to step 2,000:

- identity similarity fell `0.494456 -> 0.463905`;
- text similarity rose `25.8003 -> 27.0073`;
- TOPIQ-Face mean was essentially flat, `0.717769 -> 0.717843`;
- TOPIQ-Face p10 fell sharply, `0.622961 -> 0.568263`;
- all 96 outputs changed and remained structurally renderable;
- face detection and TOPIQ-Face coverage remained `96/96`.

Against the otherwise identical E3 run at step 2,000, E4 is lower on identity
by `0.01401`, lower on TOPIQ-Face p10 by `0.01994`, and lower on every logged
IQA mean; only text is higher, by `0.16162`. The rank objective therefore has
no observed model-quality benefit at this horizon.

That is informative, but **2k is not a valid hard stopping point for this
family**. The two available hard-routing BA trajectories both lost identity
at 2k and recovered later:

| Run | Step 0 | 2k | 4k | 8k | 18k |
|---|---:|---:|---:|---:|---:|
| Historical hard BA + unintended broad trainables | 0.3063 | 0.2841 | 0.3138 | 0.3609 | **0.3817** |
| Clean hard BA32 | 0.3063 | 0.1519 | 0.2744 | 0.3192 | **0.3347** |
| E3 anchored v3, no rank loss | 0.4945 | 0.4779 | — | — | — |
| E4 anchored v3, rank loss | 0.4945 | 0.4639 | — | — | — |

The first historical run also changed from lower identity/higher text at 2k
to higher identity/lower text later. E4 currently has the same qualitative
early-phase signature. These historical curves do not prove E4 will recover,
because their routing, trainable ownership, sampling, and validation processor
semantics differ. They do prove that “below step zero at 2k” must not, by
itself, reject an otherwise healthy run.

The concern about a return to plain PhotoMaker is also only partly correct:

- E4 is **more structurally anchored** to frozen native PhotoMaker/SDXL than
  the old hard-replacement runs by design;
- it is **not functionally collapsed** to PhotoMaker. Its step-zero images
  differ from the matched native/PhotoMaker v2 anchor, its branch contribution
  grows to about `48%` of native attention RMS, and its reference/native
  cosine falls to about `0.46`;
- the old run is not a clean template to restore wholesale. It forced
  reference routing at many more sites and unintentionally trained `171.29M`
  parameters, including generic and PhotoMaker adapters that were not fully
  checkpointed.

The immediate next training experiment should therefore be **E5-L40: a fresh,
E4-exact run with a 40k ceiling and an 8k first decision point**, with fixed-96
validation every 2k. This is the only clean way to answer whether E4 is merely
in the same early dip seen before while making unattended GPU time useful.
Before interpreting or increasing reference strength, run the already enabled
five-arm fixed-checkpoint matrix on E4's 2k weights: matched, shuffled spatial,
zero spatial, alpha 0, and alpha 1.

If E5-L40 has an upward identity trajectory by 4–8k, keep the architecture and
use guarded 10k/12k and later checkpoints to find the peak. If identity is flat
or falling through 8k while the causal gap remains positive, the next
architecture should be a **query-adaptive anchored BA-v4**: preserve the frozen
native path but add a zero-initialized, branch-only target-query LoRA. That
borrows the useful adaptation degree from the old hard route without unfreezing
or replacing the native PhotoMaker path. A post-mix norm-preservation toggle
and a corrected per-sample, correct-only ranking objective should be
implemented separately, not silently stacked into E5-L40.

## Evidence integrity and limits

The downloaded E4 result contains:

- exactly 96 step-zero images and the same 96 filenames at step 2,000;
- identity and text aggregates at both validation steps;
- the compact seven face-quality metrics at both validation steps;
- 40 training windows at 50-step cadence, through step 1,950;
- direct shuffle-conditional metrics for 39 active windows;
- no export warning, export error, traceback, CUDA OOM, or non-finite loss;
- requested, resolved, and manifest steps all equal to the selected endpoint;
- the immutable key `f72ea55eb0af44828cd6511a15ba5933` in both exports.

Image-hash checks establish:

- all `96/96` E4 step-zero PNGs are byte-identical to E3 step zero;
- all `96/96` E4 step-2k images differ from E4 step zero;
- all `96/96` E4 step-2k images differ from E3 step 2k.

The local export does not include a per-image identity-score table. The
identity deltas below are therefore exact deterministic panel aggregates, but
there is no per-image identity confidence interval in this evidence package.
Visual conclusions are based on all 96 downloaded images, not formal anatomy
labels.

No validation seed, prompt, reference image, bbox, scheduler, CFG, inference
step count, validation base, or metric definition changed between E3 and E4.

## 1. What this training behaviour uncovers

### 1.1 E4 is mechanically healthy

The startup and training evidence rules out the failure modes seen in earlier
runs:

```text
architecture                         anchored_mix_sa_v3
branched self-attention sites         46: mid, up0, up1
branched cross-attention              disabled
target native Q/K/V                   frozen
reference K/V rank                    32
branch output rank                    32
trainable ownership                   414 tensors / 10,567,818 parameters
trainable precision                   FP32
validation processor base             validation_native
pose_adapt_ratio                      0
ca_mixing_for_face                    false
step-zero images                      96
step-2k images                        96
face detections at both gates          96/96
```

The exact optimizer groups are present:

| Role | Tensors | Approximate parameters |
|---|---:|---:|
| Reference K/V | 184 | 7.05M |
| Reference output | 92 | 3.52M |
| Bounded mix | 138 | scalars |

The run completed all 2,000 updates, both full-96 validations, face-quality
scoring, and checkpoint saves. This result is not explained by CPU model
construction, missing optimizer parameters, missing checkpoint state, or an
inactive inference processor.

### 1.2 The endpoint is worse than initialization and worse than E3 at 2k

The complete fixed-96 comparison is:

| Metric | E4 step 0 | E4 step 2k | E4 change | E3 step 2k | E4 − E3 at 2k |
|---|---:|---:|---:|---:|---:|
| Identity similarity | 0.494456 | 0.463905 | **-0.030551** | 0.477912 | **-0.014007** |
| Text similarity | 25.8003 | 27.0073 | **+1.2070** | 26.8457 | +0.1616 |
| Face detection rate | 1.0000 | 1.0000 | 0 | 1.0000 | 0 |
| TOPIQ-Face mean | 0.717769 | 0.717843 | +0.000074 | 0.722528 | -0.004684 |
| TOPIQ-Face p10 | 0.622961 | 0.568263 | **-0.054698** | 0.588201 | **-0.019938** |
| TOPIQ-Face coverage | 1.0000 | 1.0000 | 0 | 1.0000 | 0 |
| Generic TOPIQ mean | 0.587789 | 0.602684 | +0.014895 | 0.606483 | -0.003799 |
| MUSIQ mean | 73.2048 | 72.9664 | -0.2385 | 73.1228 | -0.1565 |
| MANIQA mean | 0.633840 | 0.634641 | +0.000801 | 0.637325 | -0.002685 |

The rank objective produces a consistent direction: slightly more prompt/text
adaptation and weaker identity/face-quality behavior than E3. It is too early
to know whether that direction later reverses, but there is no 2k promotion
signal in the validation metrics.

E4 also remains below the residual-v2/native-PhotoMaker control at the same
endpoint: v2 reached identity `0.50861` and TOPIQ-Face mean `0.73590` at 2k.
That comparison establishes the current performance gap to the structurally
strong anchor; it does not mean v2 has useful spatial-reference causality.
V2 previously failed that functional test.

### 1.3 Two thousand steps are informative but not decisive

The old BigCelebs same-ID run is useful for trajectory shape, not as a clean
absolute control:

| Metric | Step 0 | 2k | 4k | 8k | 10k | 18k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical hard-BA identity | 0.3063 | 0.2841 | 0.3138 | 0.3609 | 0.3723 | **0.3817** | 0.3628 |
| Historical hard-BA text | 26.4229 | **27.8118** | 27.2061 | 27.0166 | 26.8761 | 26.6243 | 26.5828 |
| Clean hard-BA32 identity | 0.3063 | 0.1519 | 0.2744 | 0.3192 | 0.3039 | **0.3347** | 0.3273 |

Both hard-BA runs first optimized an easier text/rendering direction. Identity
then recovered after several validation intervals. The clean run did not even
cross its own step-zero identity until 8k. E4's `identity down / text up`
signature is therefore compatible with an early optimization phase rather
than a terminal plateau.

The caveats are material:

- the historical run trained 171.29M parameters after the swallowed
  installation exception, rather than clean BA-only state;
- it used hard face replacement, trainable target and reference projections,
  and more processor sites;
- its checkpoint omitted some live trainables;
- it used `legacy_full_copy` validation rather than E4's
  `validation_native` semantics;
- it used a different BigCelebs sampling path rather than E4's pinned policy-v1
  schedule.

Therefore the old curve justifies a longer controlled E4 horizon. It does not
justify claiming that E4 will recover, nor does it justify restoring the old
model wholesale.

### 1.4 The differentiable rank objective is active, but its effect is small

E4 uses:

```text
shuffle probability                    0.50
relative margin                        0.02
reference loss weight                  0.10
wrong-reference forward                differentiable
PhotoMaker identity inputs             always correct and unchanged
spatial reference in wrong forward     different identity
target/noise/timestep/reference noise  paired and unchanged
```

Its direct shuffle-conditional diagnostics are:

| Interval | Conditional relative gap | Absolute face-error gap | Prediction delta / prediction RMS |
|---|---:|---:|---:|
| 50–500 | 1.740% | 0.002921 | 9.696% |
| 550–1,000 | 1.291% | 0.002299 | 8.154% |
| 1,050–1,500 | 1.064% | 0.001977 | 7.374% |
| 1,550–1,950 | **1.081%** | **0.001913** | **7.052%** |
| Whole run | **1.299%** | **0.002287** | **8.095%** |

All `39/39` active windows have a positive correct-reference advantage, but
only `3/39` meet the requested 2% margin and only `8/39` reach 1.5%. The gap
declines after the first 500 steps and then stabilizes near 1.1%.

Compared with E3's zero-dilution-corrected diagnostic, E4 improves the later
relative gap by only about `0.13–0.22` percentage points:

| Interval | E3 conditional estimate | E4 direct conditional | E4 − E3 |
|---|---:|---:|---:|
| 50–500 | 1.767% | 1.740% | -0.027 pp |
| 550–1,000 | 1.071% | 1.291% | +0.220 pp |
| 1,050–1,500 | 0.936% | 1.064% | +0.127 pp |
| 1,550–1,950 | 0.929% | 1.081% | +0.152 pp |

The reference loss is numerically small. Its zero-diluted mean is `0.004533`;
after weight `0.10`, its mean contribution to total loss is about `0.000453`
against total loss `0.4114`, roughly `0.11%` of the scalar objective. Gradient
scale is not identical to scalar-loss share, so this is not proof that its
effect is negligible. It does explain why E4's branch trajectory remains very
close to E3.

### 1.5 E4 is anchored, but it is not plain PhotoMaker

The correct distinction is:

```text
structurally anchored to a frozen native path       yes
functionally identical to plain PhotoMaker          no
explicit spatial reference route active             yes
reference route dominant or hard-replacing          no
```

At initialization, v3 mixes a live reference message at alpha `0.50`; it is
not the zero-start v2 residual. Previous exact comparison found a mean
normalized pixel difference of about `0.0246` from the native/PhotoMaker v2
anchor. At 2k, the learned branch telemetry is:

| Telemetry | Step 0 | Step 1,950 |
|---|---:|---:|
| Mean mix | 0.5000 | 0.4670 |
| Reference/native RMS | 0.9998 | 1.0092 |
| Contribution/native RMS | 0.3022 | **0.4827** |
| Reference/native cosine | 0.8088 | **0.4566** |
| Merged/native RMS | 0.9507 | **0.8565** |

All three semantic groups show the same active behavior:

| Group | Mix at 1,950 | Contribution/native RMS | Reference/native cosine | Merged/native RMS |
|---|---:|---:|---:|---:|
| Mid | 0.4707 | 0.4964 | 0.4339 | 0.8462 |
| Up 0 | 0.4649 | 0.4839 | 0.4486 | 0.8548 |
| Up 1 | 0.4711 | 0.4535 | 0.5349 | 0.8818 |

The branch is not shrinking toward its `.25` floor. It is rotating away from
the native message while retaining about half-strength mixing. The validation
images also move substantially from step zero. On 64px deterministic
thumbnails, mean normalized absolute displacement is `0.01934`; the E3-versus-
E4 endpoint difference is only `0.00466`. Thus training changes the image
about four times more than the objective toggle changes it.

The model nevertheless has a stronger native anchor than the old hard route:

- native target Q/K/V and native attention remain frozen;
- only 46 mid/up sites receive BA, rather than the old broad site set;
- face output is a bounded interpolation, not hard replacement;
- PhotoMaker's pretrained adapter remains frozen;
- only 10.57M exact BA parameters train, versus the historical live 171.29M.

This anchor is currently protecting structure, but it may limit identity
adaptation. The correct response is a bounded branch-specific degree of
freedom, not global unfreezing.

### 1.6 Linear interpolation is attenuating the face message

E4 exposes a concrete architectural effect that was not visible before direct
cosine telemetry was added. The current face fusion is

```text
merged = (1 - alpha) * native + alpha * reference
```

RMS matching makes `||reference||` approximately equal to `||native||`, but
it does not preserve the norm of their interpolation. For equal endpoint
norms, reference/native cosine `c`, and mix `a`, the expected norm ratio is

```text
sqrt((1-a)^2 + a^2 + 2*a*(1-a)*c)
```

At the final E4 window, `a ~= 0.467` and `c ~= 0.457`, which predicts a ratio
near `0.856`—exactly the logged `merged/native RMS = 0.8565`. The branch is
not only adding reference information; it is cancelling part of the native
message as its direction rotates.

This is a plausible contributor to the worse weak-face tail. It is not yet a
proved cause because no norm-preserving fixed-checkpoint intervention has
been run. A post-mix RMS guard should be implemented behind a toggle and
tested as a fixed-checkpoint arm before it is included in training.

### 1.7 Visual changes are coherent but do not show systematic identity gain

The all-96 side-by-side review finds:

- no broad return of pasted, displaced, doubled, or detached faces;
- action and small-face compositions remain much more coherent than the old
  hard-replacement outputs;
- E4 and E3 at 2k are visually very close;
- their differences are mainly mouth, eyes, expression intensity, local skin
  rendering, and small pose changes;
- E4 does not show a repeated identity-morphology improvement across people or
  prompt groups;
- the weaker TOPIQ-Face p10 is consistent with a subset of more strained or
  less clean faces, even though detection remains saturated.

The old hard-BA 20k/32k images move farther from initialization and sometimes
show stronger recognizable traits, but they also show more facial distortion,
goggle/eye integration problems, over-strong expressions, and lower aggregate
quality. This is the trade-off a new architecture should avoid.

### 1.8 What E4 does and does not establish

E4 establishes:

1. the spatial BA branch is installed, optimized, and causally sensitive;
2. the differentiable objective slightly strengthens late correct-versus-wrong
   separation;
3. the objective does not improve the fixed-96 endpoint by 2k;
4. the branch is not collapsing to PhotoMaker;
5. the interpolation is increasingly attenuating its face message;
6. a 2k hard promotion rule is inconsistent with the observed historical
   learning curves.

E4 does not establish:

1. that the 8k identity trajectory will recover;
2. that more reference strength would improve identity;
3. that matched spatial BA beats alpha-zero PhotoMaker on held-out images;
4. that the rank loss improves the correct output rather than mainly making
   the wrong-reference output worse;
5. that the old hard route's gain came from hard BA rather than its broader,
   unintended trainables;
6. that rank, layer count, or dataset size is the current bottleneck.

## 2. High-priority issues in the current code and experiment design

### P0. The ranking loss is batch-aggregated rather than per sample

[`_masked_face_mse`](../src/loss/diffusion_loss.py) averages face error across
the batch before [`BranchedReferenceLoss`](../src/loss/branched_reference_loss.py)
applies the hinge. With batch size two, E4 optimizes one aggregate margin:

```python
correct_face = _masked_face_mse(model_pred, target, face_bbox)
wrong_face = _masked_face_mse(pred_wrong_spatial_ref, target, face_bbox)
relative_gap = (wrong_face - correct_face) / stopgrad(correct_face)
loss_rank = relu(margin - relative_gap)
```

One easy sample can compensate for one hard sample. The logged 1.3% mean also
does not reveal whether both members consistently prefer the correct spatial
reference.

Implement a per-example helper and reduce only after the hinge:

```diff
+def _masked_face_mse_per_sample(model_pred, target, face_bbox):
+    errors = []
+    for index, box in enumerate(face_bbox):
+        # same validated latent crop as the historical scalar helper
+        errors.append(F.mse_loss(pred_crop.float(), target_crop.float()))
+    return torch.stack(errors)  # [B]

-correct = _masked_face_mse(model_pred, target, face_bbox)
-wrong = _masked_face_mse(pred_wrong_spatial_ref, target, face_bbox)
-gap = (wrong - correct) / correct.detach().clamp_min(1e-6)
-rank = F.relu(relative_margin - gap)
+correct_i = _masked_face_mse_per_sample(model_pred, target, face_bbox)
+wrong_i = _masked_face_mse_per_sample(pred_wrong_spatial_ref, target, face_bbox)
+gap_i = (wrong_i - correct_i) / correct_i.detach().clamp_min(1e-6)
+rank = F.relu(relative_margin - gap_i).mean()
```

Add conditional `gap_mean`, `gap_p10`, `margin_satisfied_fraction`, and
correct/wrong face errors. Keep the current scalar mode available for exact E4
replay.

### P0. The differentiable negative branch can satisfy the loss by becoming worse

In E4's active hinge region, gradients both lower correct-reference error and
raise wrong-reference error. The latter creates reference discrimination, but
it does not necessarily improve the production output. The 2k result—slightly
larger separation with worse identity and quality—is compatible with this
failure mode, although it does not prove it.

Add a reversible `correct_only_relative_rank` mode:

```diff
+elif self.reference_mode == "correct_only_relative_rank":
+    correct_i = _masked_face_mse_per_sample(model_pred, target, face_bbox)
+    with torch.no_grad():
+        wrong_i = _masked_face_mse_per_sample(
+            pred_wrong_spatial_ref, target, face_bbox
+        )
+    relative_gap_i = (
+        (wrong_i - correct_i) / correct_i.detach().clamp_min(1.0e-6)
+    )
+    reference_causal = F.relu(
+        self.reference_relative_margin - relative_gap_i
+    ).mean()
```

This mode cannot win by deliberately degrading the negative. Do not switch
E5-L40 to it: E5-L40 must be an exact long-horizon E4 control. Use it only in a
later objective ablation after E5-L40.

### P0. No held-out fixed-checkpoint causal/alpha matrix has been run

Training-time diffusion errors prove causal use on training batches. They do
not answer the user's central inference question: whether the 2k checkpoint's
held-out images are still mostly PhotoMaker and whether stronger spatial BA
would help.

The evaluator now supports v3 and `--ba-mix-override`, but no five-arm E4
matrix or Neb shell wrapper exists. This matrix is required before increasing
alpha, reducing the native path, or restoring hard routing.

Required arms, each with 96 images and its own immutable Comet key:

| Arm | Spatial BA input | PhotoMaker identity | Alpha | Question |
|---|---|---|---:|---|
| `matched` | correct reference | correct | learned | Current production endpoint |
| `shuffle_spatial` | different identity | correct | learned | Held-out spatial causality |
| `zero_spatial` | zero latent/noise | correct | learned | Dependence on spatial content |
| `alpha0` | correct | correct | 0 | Exact native/PhotoMaker endpoint |
| `alpha1` | correct | correct | 1 | Strong-reference upper bound and artifact test |

Interpretation:

- `matched > alpha0` on identity with stable p10: useful incremental BA;
- `alpha1 > matched` with stable structure: current native anchor may be too
  strong;
- `alpha0 >= matched`: the branch is active but currently harmful;
- matched approximately equal to shuffled/zero: training-time causality is
  not transferring to generation.

### P0. Linear fusion has no post-mix norm guard

[`anchored_mix_sa_processor_v3.py`](../src/model/photomaker_branched/anchored_mix_sa_processor_v3.py)
RMS-matches the reference endpoint and then linearly interpolates. E4 directly
shows the merged message falling to `0.856x` native RMS. Add a defaults-off
post-mix guard:

```diff
 mixed_face = native_out + mix * (reference_out - native_out)
+if self.post_mix_rms_preserve:
+    native_rms = self._masked_rms(native_out, target_mask)
+    mixed_rms = self._masked_rms(mixed_face, target_mask)
+    scale = (native_rms / mixed_rms.clamp_min(1e-6)).clamp(
+        self.post_mix_rms_clip_min,
+        self.post_mix_rms_clip_max,
+    ).detach()
+    mixed_face = mixed_face * scale[:, None, None].to(mixed_face.dtype)
 target_out = native_out + target_mask * (mixed_face - native_out)
```

Use a conservative scale clip such as `[0.90, 1.10]` for the first
fixed-checkpoint intervention. Record the toggle and clip in schema-v2. Do not
enable it in E5-L40 because that would alter the control's step-zero pixels.

### P1. Frozen target Q limits branch-specific correspondence capacity

The native path should remain frozen, but the reference branch currently uses
the same frozen target query. Reference K/V and output can rotate, yet the
branch cannot learn a separate query space for matching target pose/location
to identity-bearing reference features.

The old hard processor had trainable target/noise Q in addition to reference
K/V. Its result is confounded, but this is the most relevant architectural
degree to recover in a bounded form. The proposed v4 adds a query adapter only
inside the reference branch; it never changes native target self-attention.

### P1. The 2k experiment design had an over-strict endpoint gate

The E4 spec required identity not below its own step zero at 2k. That was a
reasonable safety gate before the long histories were compared closely, but
it is not a valid terminal criterion now. It would have rejected both hard-BA
trajectories before their improvement phase.

The revised rule is:

- 2k: reproducibility, causality, finite training, and structural safety;
- 4k: look for an identity turn, but do not require promotion;
- 6k/8k: require a sustained upward trajectory and stable weak-face tail;
- only 8k is the first promotion decision for this model family.

### P1. Conditional logging is still incomplete

E4 correctly logs conditional gap and prediction delta, but
`loss_reference_causal` and `loss_wrong_reference_face` remain zero-diluted.
Add conditional companions and the per-example satisfaction fraction. Preserve
the historical series for replay.

### P1. Cross-run absolute comparisons can hide validation semantics

E4 step zero (`0.4945` identity) is much higher than the old hard run step zero
(`0.3063`), but this is not evidence that E4's trainable BA is already better.
The architectures generate different step-zero images, and the old run uses
`legacy_full_copy` processor-base behavior while E4 uses `validation_native`.
Use within-run changes and the fixed checkpoint alpha matrix for mechanistic
claims.

## 3. Improvements and next experiments in priority order

## Priority 0 — D2: run E4's fixed-checkpoint causal/alpha matrix

This is the fastest way to answer whether the checkpoint is over-anchored.
It uses no training and keeps the exact fixed-96 contract.

Implementation artifacts to add:

```text
launchers/neb/run_anchored_mix_sa_v3_rank_2k_d2_validation_matrix.sh
experiments/big_celebs/d2_e4_rank_2k_matched.json
experiments/big_celebs/d2_e4_rank_2k_shuffle_spatial.json
experiments/big_celebs/d2_e4_rank_2k_zero_spatial.json
experiments/big_celebs/d2_e4_rank_2k_alpha0.json
experiments/big_celebs/d2_e4_rank_2k_alpha1.json
```

The launcher must:

1. refuse a busy Neb GPU;
2. verify the E4 weights checkpoint by exact SHA-256;
3. verify the architecture manifest is v3, 46 sites, rank 32, alpha bounds
   `.25/.90`, ratio zero, CA off, and `validation_native`;
4. run one arm at a time;
5. require 96 images, `per_image.json`, `run_manifest.json`, and all aggregate
   metrics before publishing completion;
6. create one fresh Comet experiment per arm in
   `jul-comet-large-testing-tr`;
7. record alpha explicitly in both the command and manifest.

No result should be inferred from the arm name alone; audit each immutable key.

## Priority 0 — E5-L40: fresh exact E4 run with a 40k ceiling

### Scientific question

Does the causally active v3+rank model recover identity after the same early
text-first phase seen in prior BA runs, or is its trajectory genuinely
misaligned?

### Why a fresh continuous run rather than an architectural jump or blind chain

A fresh run preserves the standard step-zero plus every-2k validation contract
in one immutable experiment. It also reproduces E4 at 2k before asking the new
4k/6k/8k question. Resuming or chaining into new Comet runs would require
special boundary validation, duplicate initialization work, and split the
canonical trajectory across keys. The first scientific decision remains 8k;
the 40k ceiling simply preserves later checkpoints during unattended running.

### Exact configuration

E5-L40 changes only the horizon:

```yaml
defaults:
  - big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_2k
  - _self_

trainer:
  epoch_len: 2000
  n_epochs: 20
  validation_interval_steps: 2000
  save_period: 1

weights_only_save_period: 1

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
```

Everything else remains E4-exact:

```text
architecture                  anchored_mix_sa_v3
sites                         mid, up0, up1 (46)
reference K/V rank            32
reference output rank         32
mix                           init .50 / floor .25 / max .90
RMS matching                  reference endpoint only
rank mode                     differentiable_rank
rank weight                   .10
relative margin               .02
shuffle probability           .50
timestep policy               inference_active
trainable dtype               FP32
dataset                       pinned BigCelebs policy-v1
batch size                    2
validation                    fixed manual_val 96
validation base               RealVisXL V4.0
validation processor mode     validation_native
branched CA                   off
pose_adapt_ratio              0
ca_mixing_for_face            false
```

The run consumes schedule rows `[0, 80000)`—40,000 optimizer batches at batch
size two—without resetting the iterator at validation boundaries.

Implemented artifacts:

```text
src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_40k.yaml
launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_40k.sh
experiments/big_celebs/
  rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1.json
```

Run name:

```text
rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1
```

### Required gates

At startup:

- all 96 step-zero hashes must match E4 step zero;
- exact ownership must remain `414 / 10,567,818`, optimizer `414/414`;
- resolved rank mode, probability, weight, and margin must match E4;
- validation must log identity, text, seven face-quality metrics, and 96 images;
- first three training batches must be finite;
- conditional and BA telemetry series must appear in Comet.

At 2k:

- treat identity below step zero as an early-phase observation, not a stop;
- require no widespread structural regression;
- require positive conditional reference gap and non-collapsed branch
  contribution;
- compare against E4 2k for reproducibility.

At 4k:

- prefer identity above the 2k value;
- require TOPIQ-Face p10 to stop deteriorating;
- require merged/native RMS to remain above `0.80`;
- stop only for continued sharp degradation, non-finite behavior, or clear
  anatomy failure—not merely for remaining below step zero.

At 6k and 8k:

- require a sustained positive identity slope from the 2k trough;
- promotion target: identity above own step zero `0.49446`;
- TOPIQ-Face p10 should recover to at least `0.613` (within `.01` of step zero);
- TOPIQ-Face mean should be no worse than step zero;
- text should not collapse below its step-zero value;
- conditional correct-reference advantage should remain positive and preferably
  at or above `1.5%`;
- matched fixed-checkpoint output must beat shuffled/zero and must add value
  over alpha zero;
- all fixed hard prompt groups must remain anatomically coherent.

If identity is still below step zero at 8k but has risen monotonically at
4/6/8 and quality has recovered, 10k/12k remain the next interpretation gates.
For the user-selected unattended 40k-capable run, preserve all later
checkpoints but do not assume the process should select its final state. If
identity is flat or falling from 4k through 8k, v4 remains the next
architecture even if the current process is allowed to finish collecting its
long-horizon curve.

### Expected runtime

E4 trained at approximately `2.0–2.6 s/it` because about half of batches run
a differentiable second U-Net forward. Eight thousand steps therefore require
roughly five training hours plus five full-96 validations. Exact runtime will
depend on validation and face-quality scoring. Do not reduce the panel or
silently change batch semantics to save time.

### Overnight 40k-capable adaptation

The user subsequently selected an unattended ceiling of 40,000 steps. The
scientific decision horizon remains 8k, but the execution artifact is now a
single fresh E4-exact run through at most 40k:

```text
src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_40k.yaml
launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_40k.sh
rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1
```

This is preferable to blindly chaining several models. The continuous run
retains optimizer and deterministic dataset position, while full checkpoints,
weights-only checkpoints, fixed-96 images, and metrics are preserved every
2,000 steps. Consequently, reaching 40k cannot erase an earlier 8k, 12k, or
18k candidate. **The selected model must be the best validated intermediate
checkpoint, not automatically the final checkpoint.**

The longer ceiling is an operational choice, not permission to reinterpret a
bad curve as progress. Use 8k as the first promotion decision, treat roughly
10–20k as the likely selection window from the historical curves, and treat
22–40k primarily as plateau/overtraining evidence unless metrics and images
continue improving. The launcher still aborts on runtime/integrity failures;
visual promotion remains a post-run decision because it cannot be made safely
from aggregate metrics alone.

At E4's observed `2.0–2.6 s/it`, plus approximately 12 minutes for each of 21
fixed-96 validations, the complete ceiling is roughly a 27–34 hour job. A
normal 9–12 hour overnight window is expected to reach approximately 8–14k,
which covers the first defensible decision region even if the full process
continues into the following day.

## Priority 1 — E6-Q: query-adaptive anchored BA-v4 if E5-L40 fails

### Design goal

Give the explicit reference branch its own target-query adaptation while
keeping the native PhotoMaker path completely frozen. This is the smallest
architectural change that meaningfully reduces the native query anchor and
borrows a useful capacity from the old hard-replacement processor.

The routing remains explicit BA:

```text
native path:
    Q_native = frozen target Q
    K_native, V_native = frozen target K/V
    N = Attn(Q_native, K_native, V_native)

reference branch:
    Q_branch = frozen target Q + branch-only LoRA-Q(target)
    K_ref, V_ref = true-key-masked reference K/V + branch LoRA
    R = Attn(Q_branch, K_ref, V_ref)

face merge:
    output = N + target_face_mask * alpha * (R - N)
```

Target queries still consume explicit reference K/V. No target K/V is
substituted into the reference branch; `pose_adapt_ratio=0` and CA mixing stay
off.

### Processor diff

Create `query_adaptive_mix_sa_processor_v4.py` as a versioned subclass or
small extension of v3:

```diff
 class QueryAdaptiveMixBranchedSelfAttnProcessorV4(
     AnchoredMixBranchedSelfAttnProcessorV3
 ):
     architecture_version = "query_adaptive_mix_sa_v4"

     def __init__(..., branch_q_rank=16, ...):
         super().__init__(...)
         self.branch_q_rank = int(branch_q_rank)
         self.branch_to_q = None

     def init_from_attention(self, attn):
         super().init_from_attention(attn)
         self.branch_to_q = _clone_effective_linear(
             attn.to_q,
             kind="lora",
             rank=self.branch_q_rank,
             trainable_dtype=self.trainable_dtype,
         )

     def named_ba_trainables(self):
         yield from super().named_ba_trainables()
         for name, parameter in self.branch_to_q.named_parameters():
             yield f"branch_to_q.{name}", parameter, "ref_query"

-    q_reference_branch = q_target
+    q_reference_branch = self._reshape_heads(
+        self.branch_to_q(target_hidden), heads
+    )
     reference_message = F.scaled_dot_product_attention(
         q_reference_branch,
         k_reference,
         v_reference,
         attn_mask=key_bias,
         dropout_p=0.0,
         is_causal=False,
     )
```

Because the cloned base equals the effective frozen target projection and the
LoRA B matrix starts at zero, v4 must be exactly pixel-identical to v3 at step
zero. That is a mandatory parity gate.

Start with branch-Q rank 16. Across the existing 46 sites this adds 92 tensors
and approximately 1.76M parameters, for an expected total near 506 tensors and
12.33M parameters. Derive and assert the exact number in the implementation;
do not encode the approximation as a contract.

### Model, optimizer, manifest, and evaluator changes

Add defaults-off model fields:

```yaml
ba_branch_q_rank: 16
ba_post_mix_rms_preserve: false
ba_post_mix_rms_clip_min: 0.90
ba_post_mix_rms_clip_max: 1.10
```

Then:

- add v4 to the strict versioned architecture allowlists in
  [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py);
- instantiate v4 in the processor factory without changing v1/v2/v3 defaults;
- add optimizer role `ref_query` with LR `5e-5`, matching reference K/V;
- propagate all new fields into the validation pipeline in
  [`train.py`](../train.py);
- record query rank, fusion mode, norm toggle, clip, routing, site names, and
  exact tensors/shapes/dtypes in schema-v2;
- make checkpoint load fail on v3/v4 or rank mismatch;
- teach the standalone evaluator to recognize v4 and allow the same audited
  alpha override;
- add `branch_query_delta_rms_ratio` telemetry by all/mid/up0/up1.

### First v4 experiment

Do not combine v4 with the unproven E4 negative-gradient objective. The first
v4 arm should compare against E3 and change only branch-Q capacity while
running long enough to observe recovery:

```yaml
model:
  ba_architecture_version: query_adaptive_mix_sa_v4
  ba_branch_q_rank: 16
  ba_reference_loss_mode: detached_diagnostic
  ba_spatial_reference_shuffle_probability: 0.25
  ba_post_mix_rms_preserve: false

loss_function:
  reference_mode: detached_diagnostic
  reference_weight: 0.0
  reference_margin: 0.0
  reference_relative_margin: 0.0

trainer:
  epoch_len: 2000
  n_epochs: 4
  validation_interval_steps: 2000
```

This gives a clean E3-versus-v4 comparison at 2k and a first 4/6/8k
trajectory. Only after query adaptation demonstrates identity value should it
be combined with a corrected per-sample rank loss.

## Priority 1 — test norm-preserving fusion as a separate ablation

The post-mix norm guard has no trainable parameters and directly targets the
observed `0.856x` attenuation. It should be tested first on the fixed E4
checkpoint, then as an otherwise identical short training arm if it improves
the weak-face tail without eliminating reference causality.

Do not use normalization to rotate the message back toward native. Preserve
the mixed direction; constrain only its RMS with a detached, clipped scalar.

## Priority 2 — replace E4's rank loss with per-sample correct-only ranking

If E5-L40 or v4 shows useful causal routing but identity remains weak, test:

```yaml
model:
  ba_reference_loss_mode: correct_only_relative_rank
  ba_spatial_reference_shuffle_probability: 0.50

loss_function:
  reference_mode: correct_only_relative_rank
  reference_weight: 0.05
  reference_relative_margin: 0.02
```

Use weight `.05` first because E4's `.10` arm worsened every endpoint metric
except text. Ramp the weight from zero only if a warmup implementation is made
explicit and logged. Do not silently reinterpret E4's mode.

## Priority 2 — small PhotoMaker identity-conditioning dropout if the alpha matrix confirms over-anchoring

If alpha-zero and matched remain nearly identical while the branch is healthy,
add a low-probability training-only conditioning dropout:

- on at most 10–20% of batches, remove the PhotoMaker identity-token
  contribution from the target conditioning;
- keep text, target latent, correct spatial reference latent/mask, and all BA
  routes unchanged;
- retain full matched PhotoMaker conditioning at validation and inference;
- log the dropout fraction and compare spatial shuffle on dropout versus normal
  batches.

This forces the explicit spatial branch to carry identity on a bounded subset
without globally unfreezing PhotoMaker. It is a later experiment, not part of
E5-L40 or first v4, because incorrect per-sample token masking would change the
conditioning contract broadly.

## Priority 3 — add `down_blocks.2`, then increase branch-local rank

Only after a query-adaptive or corrected-objective arm improves identity:

1. add `down_blocks.2` while keeping rank 16/32 and all other sites fixed;
2. inspect whether coarse identity geometry improves without attachment
   failures;
3. test rank 32 for branch Q, then rank 64 only for reference K/V/output;
4. never globally increase the historical generic U-Net or PhotoMaker adapter
   as a substitute for useful BA.

The current route is active and has millions of trainable parameters. More
rank before alignment would mainly make the same direction easier to express.

## 4. Controlled experiment ladder

| Stage | Intervention | Horizon | Main question | Advance condition |
|---|---|---:|---|---|
| D2 | E4 checkpoint matched/shuffle/zero/alpha0/alpha1 | validation only | How much held-out output comes from spatial BA, and is alpha too low? | Matched adds identity over alpha0 and beats shuffle/zero without structural loss |
| E5-L40 | Exact E4, fresh continuous run | 40k ceiling; first decision at 8k | Is 2k only the early dip, and where does the long curve peak? | Positive 2k→4k→6k→8k identity slope, recovered p10, persistent causality; select the best intermediate checkpoint |
| E6-Q | Add branch-only target Q rank 16; E3 objective | 8k | Does bounded query adaptation improve identity without losing native structure? | Step-zero parity; better ID/p10 than E3 at matched gates |
| E6-N | Post-mix norm guard | fixed checkpoint, then 4k | Does preventing cancellation recover the weak-face tail? | Higher p10/coverage with unchanged causal ordering |
| E7-R | Per-sample correct-only rank | 4k/8k | Can causal use be aligned without degrading negatives? | Correct output improves; margin and ID rise together |
| E8-D | 10–20% PM identity dropout | 4k/8k | Is PhotoMaker conditioning shortcutting spatial identity? | Dropout batches use correct spatial ref and matched inference improves |
| E9 | Add down2, then branch-local rank | 4k/8k | Is coarse-layer or low-rank capacity limiting? | Only after an aligned lower-capacity arm succeeds |

## 5. Implementation verification checklist

For D2:

- immutable E4 checkpoint SHA-256 pinned;
- 96/96 exact fixed-panel inputs per arm;
- PhotoMaker identity inputs unchanged in spatial shuffle/zero arms;
- alpha recorded in resolved config and run manifest;
- one Comet key per arm;
- per-image and aggregate metrics complete before declaring success.

For E5-L40:

- Hydra diff against E4 changes only the horizon/checkpoint cadence:
  `trainer.n_epochs: 1 -> 20`, with save and validation every epoch;
- step-zero image hashes are 96/96 identical to E4;
- trainable contract remains exactly 414 / 10,567,818;
- schedule preflight confirms rows `[0, 80000)` and no target/reference leakage;
- fixed-96 validation exists at 0 and every 2k through the reached endpoint;
- all conditional metrics and all/mid/up telemetry are present;
- no automatic early stop merely because 2k identity is below step zero.

For v4:

- v3 mode remains byte-identical when v4 selector is off;
- v4 step zero is byte-identical to v3 with zero query delta;
- native target Q/K/V receive no gradient and remain absent from optimizer;
- branch Q receives a finite nonzero gradient after its B matrix becomes
  active;
- exact allowlist equals exact optimizer membership;
- schema-v2 round-trip restores every v4 tensor and rejects v3/v4 mismatch;
- training and validation install the same v4 processor map;
- alpha zero is exact native attention even with branch-Q parameters present;
- outside-target-mask output is unchanged;
- invalid reference keys remain excluded;
- `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and branched CA-off remain
  asserted.

## 6. What not to do next

- Do not reject E4 solely because identity is lower at 2k.
- Do not claim E4 is plain PhotoMaker; telemetry and pixels contradict that.
- Do not claim the high step-zero score proves useful BA; it partly reflects
  the stronger native anchor.
- Do not restore the old 171M-parameter behavior or its incomplete checkpoint
  semantics.
- Do not force alpha one in training before running the fixed-checkpoint alpha
  arm.
- Do not increase the current differentiable-rank weight while it can improve
  by degrading the wrong branch.
- Do not combine branch-Q, post-mix normalization, objective rewrite, new
  layers, and rank 64 in one run.
- Do not change validation inputs or relax the fixed-96/p10 visual gates.
- Do not equate a 40k process ceiling with selecting the 40k checkpoint. Use
  the preserved 2k validation gates and select the best coherent intermediate
  state.

## Final recommendation

E4 shows something important: the branch is real and the objective nudges
causal separation, but 2k validation quality is worse and the nudge is small.
The existing long-run evidence makes 8k—not 2k—the first defensible decision
horizon.

Run the E4 checkpoint causal/alpha matrix, then launch E5-L40 as one fresh,
E4-exact, 40k-capable trajectory. Interpret 8k as the first decision point and
select the best fixed-96 intermediate checkpoint rather than the final state.
This preserves the branched-attention hypothesis and answers the user's
early-dip concern without confounding it with another architecture or splitting
one curve across chained runs. If that curve does not turn, implement
query-adaptive v4: a branch-only target-Q adapter with a frozen native
PhotoMaker path. That is the highest-priority architectural way to reduce
anchoring while retaining explicit target-Q/reference-KV branched attention
and avoiding the old hard route's quality and checkpoint failures.
