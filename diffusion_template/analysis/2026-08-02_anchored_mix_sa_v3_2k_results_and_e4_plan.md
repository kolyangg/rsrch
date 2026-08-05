# Anchored mix SA-v3 at 2k: causal success, identity regression, and E4 plan

**Date:** 2 August 2026  
**Run:** `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2`  
**Immutable Comet ID:** `de23193eeac9433fa090bc009f10e752`  
**Comet project:** `jul-comet-large-testing-tr`  
**Evidence cutoff:** fixed-96 validation at optimizer steps 0 and 2,000;
training diagnostics through step 1,950  
**Local evidence:**
[`comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2/`](../comet_data/rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2/)  
**Compared control:**
`rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6`, fixed-96
steps 0 and 2,000  
**Status:** E3 analysis complete. The planned E4 changes were subsequently
implemented and fresh Neb run
`rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2`
was launched with immutable Comet key `f72ea55eb0af44828cd6511a15ba5933`.

Implementation and launch details are in
[`2026-08-02_anchored_mix_sa_v3_e4_rank_implementation.md`](../docs/experiments/2026-08-02_anchored_mix_sa_v3_e4_rank_implementation.md).

This report is the result review for E3 in
[`2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md`](2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md).
E3 was intentionally an architecture-only gate: rank 32, the same 46
mid/decoder sites, the same data order and validation contract, a 25% detached
wrong-spatial-reference diagnostic, and zero reference-ranking weight.

## Executive verdict

Anchored mix SA-v3 fixed the central residual-v2 failure: the explicit spatial
reference route is now live at initialization, visibly affects the generated
images, receives gradients, and is causally sensitive to which reference is
supplied. This is a real architectural success, not a logging illusion.

It is **not yet a model-quality promotion**. From step 0 to step 2,000:

- identity similarity fell from `0.49446` to `0.47791`;
- TOPIQ-Face p10 fell sharply from `0.62296` to `0.58820`;
- text similarity improved from `25.8003` to `26.8457`;
- generic TOPIQ and mean face quality improved modestly;
- visual changes were large and structurally coherent, but predominantly
  expression, action, and generic facial rendering changes rather than clear
  identity-specific morphology improvements.

The training-time counterfactual is nevertheless decisive. After correcting
for the current logger's zero dilution on unshuffled batches, every one of the
39 shuffled diagnostic windows had positive
`wrong-reference face error - correct-reference face error`. The approximate
conditional relative advantage was `1.21%` over the run and `0.88%` over the
last 500 steps. The corresponding correct/wrong prediction difference was
about `8.06%` and `6.84%` of prediction RMS. Residual v2, restricted to the
same step range, had a conditional absolute gap of about `-1.0e-5` and only
20/39 positive windows.

The correct conclusion is therefore:

> V3 repaired reference reachability and causal use, but ordinary diffusion
> reconstruction teaches the now-strong branch to carry information that is
> more useful for prompt/expression rendering than for the held-out identity
> objective.

The next experiment should be the already isolated **E4 differentiable
matched-versus-shuffled ranking arm**, not more rank, more layers, or a 40k
continuation. Keep the architecture and all controls identical to E3 and
change only the reference objective:

```yaml
model:
  ba_reference_loss_mode: differentiable_rank
  ba_spatial_reference_shuffle_probability: 0.50

loss_function:
  reference_mode: differentiable_rank
  reference_weight: 0.10
  reference_relative_margin: 0.02
```

Before launching E4, make three small correctness/measurement changes:

1. teach the standalone checkpoint evaluator to recognize v3 and expose an
   audited `alpha` override;
2. log shuffled-reference metrics conditionally rather than averaging zeros
   from unshuffled batches;
3. log native/reference cosine and post-merge contribution so a learned output
   rotation cannot hide behind an approximately unit RMS ratio.

Then run the five-arm fixed-checkpoint matrix on the existing E3 checkpoint
and a fresh 2k E4 run. E4 earns a 4k/8k promotion only if identity no longer
declines relative to its own identical step-zero anchor and the causal
advantage strengthens without a weak-face-tail or structural regression.

## Evidence boundary and integrity

The downloaded result contains:

- exactly 96 images at step 0 and the same 96 filenames at step 2,000;
- one identity and text aggregate at both validation steps;
- the compact seven face-quality metrics at both validation steps;
- training losses, role gradient norms, mix/reference telemetry, and the
  matched-versus-shuffled diagnostic through step 1,950;
- no export warnings or errors;
- requested, resolved, and manifest validation step all equal to 2,000.

The immutable checkpoint on Neb was also inspected read-only:

```text
weights-epoch1.pth     42,483,180 bytes
sha256                 83b4137dff04ab3e4ff6a980fcd4255a3c7f1c9aaf1a69c7375762f9d1fe8021

checkpoint-epoch1.pth 127,534,810 bytes
sha256                 9fec1a35e0ff0a8c7677487294022f407408a075dfdd457f08a27e9a795af50c
```

Its schema-v2 architecture record says:

```text
architecture             anchored_mix_sa_v3
processor code version   3
routing                  frozen target Q -> true-key-masked reference K/V
merge                    native + mask * alpha * (reference - native)
reference output         frozen native to_out + trainable rank-32 delta
sites                    46: mid, up0, up1
trainables               414 tensors / 10,567,818 FP32 parameters
pose_adapt_ratio         0
ca_mixing_for_face       false
branched CA              disabled
```

The result folder does **not** contain a fixed-checkpoint matched/shuffled/
zero/branch-off/forced-reference image matrix. Training-time U-Net
counterfactual evidence is strong, but held-out image-level incremental
spatial causality still needs that matrix. Visual observations below are a
manual audit of contact sheets, not formal per-image anatomy labels.

No validation seeds, prompts, references, bboxes, model base, scheduler,
inference steps, CFG, or metric definitions were changed for this comparison.

## 1. What this training behaviour uncovers

### 1.1 The run was healthy; this is not CPU construction, optimizer, or checkpoint failure

The preserved log shows:

```text
processor construction                    5.935 s
strict trainable contract                  414 / 414 tensors
reference K/V optimizer group              184 tensors / 7.05M params
reference output optimizer group            92 tensors / 3.52M params
mix optimizer group                         138 scalars
validation processor base                  validation_native
step-0 validation                           96 images
step-2000 validation                        96 images
weights and full checkpoint                 saved successfully
```

Construction happened on `cuda:0`; the U-Net's frozen bulk used bfloat16 and
all BA trainables used FP32. The complete run took about 1 h 32 min including
two roughly 12-minute full-96 validations. Training settled near
`0.50-0.53 steps/s`, or about `1.9-2.0 s/it`. There is no traceback, OOM, NaN,
failed validation load, or missing checkpoint in the evidence.

The checkpoint independently rules out a dead optimizer path. Every one of
the 46 B matrices in the reference-K, reference-V, and reference-output
families is nonzero. Aggregate B-matrix RMS values are:

| Trainable family | Checkpoint RMS |
|---|---:|
| Reference K LoRA B | 0.005197 |
| Reference V LoRA B | 0.003709 |
| Reference output LoRA B | 0.006654 |

The 46 learned base mix logits, timestep terms, and face-area terms are also
all finite and nonzero:

| Parameter | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| `mix_logit` | -0.59409 | -0.67117 | -0.48181 |
| `mix_t` | -0.07815 | -0.15180 | +0.02248 |
| `mix_area` | +0.10785 | +0.00470 | +0.18368 |

### 1.2 V3 successfully escaped the plain-PhotoMaker anchor

Residual v2's zero-initialized output route made its step-zero validation an
exact native/PhotoMaker attention anchor. V3 instead starts with a frozen
native projection of target-Q/reference-KV attention and `alpha=.50`, so its
step-zero images are expected to differ.

They do. Relative to the v2/plain-PhotoMaker anchor at step zero, v3 changes
the deterministic RGB thumbnail by a mean normalized absolute error of
`0.02458`. Within v3, step 0 to step 2k changes by `0.02253`; the analogous v2
change is only `0.00799`. Thus the visible v3 training displacement is about
`2.82x` the v2 displacement. None of the 96 v3 step-0/step-2k PNG pairs is
byte-identical.

This directly answers the earlier concern that validation remained ordinary
PhotoMaker: the new architecture no longer has that failure mode.

It also exposes a cost. Merely enabling the untrained halfway interpolation at
step zero lowers identity by `0.02911` and mean face quality by `0.02954`
relative to the same-protocol v2/plain-PhotoMaker anchor. Guaranteed branch
use is necessary for diagnosis, but nonzero use is not automatically useful.

### 1.3 The 2k direction improves prompt rendering, not identity

The fixed-96 metrics are:

| Metric | V3 step 0 | V3 step 2k | Change |
|---|---:|---:|---:|
| Identity similarity | 0.494456 | 0.477912 | **-0.016544** |
| Text similarity | 25.8003 | 26.8457 | **+1.0454** |
| Face detection rate | 1.0000 | 1.0000 | 0 |
| TOPIQ-Face mean | 0.717769 | 0.722528 | +0.004758 |
| TOPIQ-Face p10 | 0.622961 | 0.588201 | **-0.034761** |
| TOPIQ-Face coverage | 1.0000 | 1.0000 | 0 |
| Generic TOPIQ mean | 0.587789 | 0.606483 | +0.018694 |
| MUSIQ mean | 73.2048 | 73.1228 | -0.0820 |
| MANIQA mean | 0.633840 | 0.637325 | +0.003485 |

The mean and tail tell different stories. Mean face quality recovers slightly,
but the weakest 10% of detected faces deteriorate substantially. A model that
raises mean polish while damaging its weakest faces is not ready to scale.

Against residual v2 at the same checkpoints:

| Metric | V3 - v2 at step 0 | V3 - v2 at step 2k |
|---|---:|---:|
| Identity similarity | -0.029105 | **-0.030700** |
| Text similarity | -0.5332 | +0.1239 |
| TOPIQ-Face mean | -0.029537 | -0.013377 |
| TOPIQ-Face p10 | +0.029411 | -0.003399 |
| Generic TOPIQ mean | -0.022696 | -0.008139 |
| MUSIQ mean | +0.2755 | -0.1353 |
| MANIQA mean | -0.011532 | -0.006902 |

V3 therefore learns faster and ends slightly better on text than v2, but does
not recover the identity or quality price of its active reference route by 2k.

### 1.4 The explicit spatial reference is now causally used

On diagnostic batches, the model keeps the target noisy latent, noise,
timestep, prompt, correct PhotoMaker identity tokens, and target face mask
fixed. It permutes only the spatial reference latent and its reference-face
mask to another identity and reuses the same reference-noise realization.

The current logger averages a zero into the metric on every unshuffled batch.
Dividing each equal-size logging window's numerator by its observed shuffle
fraction gives the following approximate conditional result:

| Training-time causal diagnostic | Whole run, shuffled only | Last 500 steps, shuffled only |
|---|---:|---:|
| Effective shuffle fraction | 0.2646 | 0.2440 |
| `wrong_face_mse - correct_face_mse` | **+0.002115** | **+0.001615** |
| Relative face-error gap | **+1.211%** | **+0.877%** |
| Correct/wrong prediction delta / correct RMS | **8.06%** | **6.84%** |
| Positive relative-gap windows | **39 / 39** | **10 / 10** |

At the final logged window, the corresponding conditional estimates are
`+0.001626`, `+0.869%`, and `6.44%`.

For comparison, residual v2 through the same step-1,950 cutoff had an
approximate conditional absolute gap of `-1.03e-5`, with only 20/39 windows
positive. V3 is not merely noisier: its sign consistency and two-orders-larger
mean gap establish a real matched-reference advantage in the training
objective.

The advantage is still modest and weakened over the final 500 steps. The
planned E4 margin of 2% is therefore well targeted: it asks the model to turn a
real but sub-margin signal into a stronger one instead of trying to create
causality from zero.

### 1.5 The branch is strong; it is not collapsing toward PhotoMaker

Selected telemetry windows show:

| Metric | Step 0 | 500 | 1,000 | 1,500 | 1,950 |
|---|---:|---:|---:|---:|---:|
| Mix mean | 0.5000 | 0.4844 | 0.4765 | 0.4622 | 0.4644 |
| Reference/native RMS | 0.9998 | 1.0003 | 0.9999 | 1.0000 | 1.0159 |
| Contribution/native RMS | 0.3022 | 0.4433 | 0.4285 | 0.4583 | **0.4892** |
| Reference K/V grad norm | 0.00302 | 0.07090 | 0.03193 | 0.00780 | 0.00439 |
| Reference output grad norm | 0.00226 | 0.08623 | 0.04137 | 0.01058 | 0.00506 |
| Mix grad norm | 0.00435 | 0.02156 | 0.01067 | 0.00199 | 0.00064 |

All three optimizer roles remain active. Mix decreases only about 7% and stays
far above its `.25` floor, while the actual contribution grows about 62%.
RMS matching holds the reference vector's magnitude near the native vector's
magnitude, so the growth is not a simple amplitude explosion.

Instead, the trainable K/V and output projection make the reference vector
increasingly different in direction from the native vector. As a rough
geometric inference from the aggregate ratios—not a directly logged cosine—
native/reference cosine falls from approximately `0.82` at step zero to
approximately `0.45` at step 1,950. This is how contribution can rise while
mix falls and both endpoint RMS values remain near one.

That observation is important for the next implementation. Monitoring only
`alpha` and reference/native RMS is insufficient; an output adapter can make
the branch much more disruptive by rotating the reference message. Direct
masked cosine and post-merge RMS need to be logged.

The learned average `mix_t < 0` and `mix_area > 0` imply slightly stronger
reference mixing earlier in the active denoising window and for larger face
masks, and weaker mixing later and for smaller faces. This is an inference
from the parameterization. It is consistent with the observed gain in coarse
prompt/expression behavior and lack of a fine identity gain, but it is not by
itself proof of cause.

### 1.6 Visuals remain structurally coherent, but changes are generic

The contact-sheet audit did not find a broad return of the old pasted,
displaced, doubled, or detached-face failure. Small/action faces generally
remain attached to their bodies, and all 96 outputs remain renderable with
100% face detection and coverage.

The strongest step-0 to step-2k changes repeatedly follow prompt/action
semantics:

- angry faces become more overtly angry;
- laughing and crying examples acquire stronger open-mouth expressions and
  hand-to-face gestures;
- kickboxing, drumming, skiing, and night-ride examples often acquire a broad
  smile or open mouth;
- some occlusions and local facial rendering become cleaner.

Those are useful changes, but the audit does not show consistent
identity-specific changes to facial proportions, eyes, nose, jaw, or other
persistent morphology. This agrees with higher text similarity, lower identity
similarity, and a worse TOPIQ-Face tail.

Visual evidence:

- [V3 step 0 versus step 2k, prompt groups 1-4](assets/2026-08-02_anchored_mix_sa_v3_r2/v3_step0_vs_step2k_prompts_1_4.jpg)
- [V3 step 0 versus step 2k, prompt groups 5-8](assets/2026-08-02_anchored_mix_sa_v3_r2/v3_step0_vs_step2k_prompts_5_8.jpg)
- [V3 step 0 versus step 2k, prompt groups 9-12](assets/2026-08-02_anchored_mix_sa_v3_r2/v3_step0_vs_step2k_prompts_9_12.jpg)
- [Residual v2 versus anchored v3 endpoints on difficult prompts](assets/2026-08-02_anchored_mix_sa_v3_r2/v2_v3_endpoints_hard_prompts.jpg)

### 1.7 The bottleneck has moved from architecture reachability to objective alignment

Residual v2 could minimize the ordinary objective while leaving its optional,
zero-start branch almost reference-independent. V3 removes that escape: a
correct reference now helps predict the training target more than a shuffled
reference.

However, face-region diffusion MSE rewards every predictable facial property,
including expression, illumination, makeup, texture, pose cues, and local
rendering. It does not explicitly say that the incremental spatial branch
should improve the identity metric while PhotoMaker preserves prompt and
structure. Once V3 supplies a large 46-site route, ordinary reconstruction can
spend it on the easiest shared/predictive signal.

This explains the apparently contradictory evidence:

```text
branch installed and optimized            yes
branch visibly changes validation          yes
correct spatial reference beats shuffled   yes, modestly and consistently
identity improves                          no
prompt/expression rendering improves       yes
```

E3 therefore passed its intended causal architecture gate and failed its model
promotion gate. It should be retained as a positive causal control.

### 1.8 Increasing rank or adding layers now would amplify an unaligned route

The model already has 10.57M trainable parameters, a roughly 49%-of-native
face-local contribution at the final logged window, nonzero gradients in every
role, and a visible 2.8x larger training displacement than v2. There is no
evidence that rank-32 capacity or the number of sites is the immediate limit.

Rank 64, `down_blocks.2`, or more output capacity could make the same
expression/prompt direction stronger while worsening identity and the weak
face tail. Capacity expansion becomes meaningful only after an objective-only
arm shows that the existing route can be aligned with identity usefulness.

## 2. High-priority issues in the current code and experiment tooling

### P0. Shuffled-reference metrics are diluted by unshuffled zeros

[`branched_reference_loss.py`](../src/loss/branched_reference_loss.py) returns
zero for the reference gap and prediction delta when no wrong-reference pass
was sampled. [`sdxl_trainers.py`](../src/trainer/sdxl_trainers.py) then averages
every batch uniformly:

```python
for loss_name in self.config.writer.loss_names:
    batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
    train_metrics.update(loss_name, batch[loss_name].item())
```

Consequences:

- a reported `0.003` relative gap at 25% shuffling actually means roughly
  `0.012` on the batches where the counterfactual ran;
- changing shuffle probability from 25% to 50% changes the plotted magnitude
  even if conditional behavior is identical;
- E3 and E4 cannot be compared directly without offline correction;
- a low effective shuffle rate can make a useful branch look weak.

Keep the existing unconditional series for historical continuity, but add
conditional series based only on ranks/batches where the intervention ran:

```diff
--- a/src/trainer/sdxl_trainers.py
+++ b/src/trainer/sdxl_trainers.py
@@
+conditional_reference_metrics = {
+    "reference_error_gap": "reference_error_gap_conditional",
+    "reference_error_relative_gap":
+        "reference_error_relative_gap_conditional",
+    "reference_prediction_delta_ratio":
+        "reference_prediction_delta_ratio_conditional",
+}
+shuffle_by_rank = self.accelerator.gather(
+    batch["reference_shuffle_applied"].detach().reshape(1)
+).float()
+active_rank = shuffle_by_rank > 0.5
 
 for loss_name in self.config.writer.loss_names:
-    batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
+    gathered = self.accelerator.gather(
+        batch[loss_name].detach().reshape(1)
+    ).float()
+    batch[loss_name] = gathered.mean()
     train_metrics.update(loss_name, batch[loss_name].item())
+    conditional_name = conditional_reference_metrics.get(loss_name)
+    if conditional_name is not None and torch.any(active_rank):
+        train_metrics.update(
+            conditional_name,
+            gathered[active_rank].mean().item(),
+        )
+
+train_metrics.update(
+    "reference_shuffle_rank_fraction",
+    active_rank.float().mean().item(),
+)
```

The implementation should preserve gradient-bearing tensors in `batch`; only
detached gathered copies are used for logging. On one GPU this becomes the
obvious conditional batch average; on multiple ranks it remains correct when
different ranks make different shuffle decisions.

### P0. The fixed-checkpoint evaluator currently rejects v3 as “no branched self-attention”

[`evaluate_rhca_checkpoint.py`](../tools/inference/evaluate_rhca_checkpoint.py)
recognizes only these processor class names:

```python
if type_name in {
    "BranchedAttnProcessor",
    "ResidualBranchedSelfAttnProcessorV2",
}:
    branched_self.append(name)
```

`AnchoredMixBranchedSelfAttnProcessorV3` is missing. The evaluator therefore
fails its own nonzero-branched-SA audit before it can run a trustworthy v3
matrix. This did not affect in-training validation; it blocks the separate
checkpoint intervention tool.

Required fail-closed fix:

```diff
@@ def processor_type_audit(model):
     if type_name in {
         "BranchedAttnProcessor",
         "ResidualBranchedSelfAttnProcessorV2",
+        "AnchoredMixBranchedSelfAttnProcessorV3",
     }:
```

A more future-proof follow-up can audit a known `architecture_version`
attribute, but the localized v3 addition is the safest experiment fix.

### P0. The evaluator has no audited alpha override

The processor supports `ba_mix_override=0` and `=1`, but the standalone tool
has no CLI argument and its pipeline runtime refresh would reset an override
unless the value is also copied to the pipeline. Add:

```diff
@@ def main():
+parser.add_argument(
+    "--ba-mix-override",
+    type=float,
+    help="V3 diagnostic alpha override in [0,1]; omitted uses checkpoint mix.",
+)
@@ def run(args):
+if args.ba_mix_override is not None:
+    if not 0.0 <= args.ba_mix_override <= 1.0:
+        raise ValueError("--ba-mix-override must be in [0,1]")
+    if str(config.model.ba_architecture_version) != "anchored_mix_sa_v3":
+        raise ValueError("--ba-mix-override is restricted to anchored v3")
+    config.model.ba_mix_override = float(args.ba_mix_override)
@@ after build_pipeline(...):
+setattr(pipeline, "ba_mix_override", args.ba_mix_override)
```

Record the override in `run_manifest.json` and in the Comet hyperparameters.
Do not infer branch-off from a run name. `alpha=0` must remain explicit and
auditable; it is a diagnostic native endpoint, not a new training setting.

### P1. RMS telemetry hides reference/native directional divergence

Current v3 telemetry logs:

```text
reference RMS / native RMS
contribution RMS / native RMS
alpha
```

It does not log the angle between native and reference messages or the actual
post-merge magnitude. Because interpolation uses
`alpha * (reference - native)`, unit endpoint RMS does not bound the
difference. The observed contribution growth while alpha falls demonstrates
this blind spot.

Add masked cosine and merged/native RMS inside the existing sampled,
detached telemetry block:

```diff
--- a/src/model/photomaker_branched/anchored_mix_sa_processor_v3.py
+++ b/src/model/photomaker_branched/anchored_mix_sa_processor_v3.py
@@ def _record_telemetry(...):
+native_norm = self._masked_rms(native, target_mask).clamp_min(1.0e-8)
+reference_norm = self._masked_rms(reference, target_mask).clamp_min(1.0e-8)
+masked_dot = self._masked_mean_dot(reference, native, target_mask)
+reference_native_cosine = (
+    masked_dot / (reference_norm * native_norm).clamp_min(1.0e-8)
+).mean()
+merged = native + contribution
+merged_native_rms_ratio = (
+    self._masked_rms(merged, target_mask) / native_norm
+).mean()
@@
+"reference_native_cosine": reference_native_cosine.detach(),
+"merged_native_rms_ratio": merged_native_rms_ratio.detach(),
```

Also log `ref_out_delta/reference_base RMS` if it can be computed without
retaining extra activations. It will distinguish reference K/V learning from a
generic high-capacity output rotation.

### P1. Telemetry values are sampled and then carried forward

Each processor updates `_latest_ba_telemetry` every configured processor
interval. `collect_branched_telemetry` returns that last value on intervening
steps, and the trainer averages the carried-forward snapshots. The curves are
useful, but they are not fresh per-step measurements or true averages over all
forwards.

For E4, add a monotonically increasing sample ID or `telemetry_fresh` flag and
only add a conditional snapshot when the ID changes. Preserve the old series
names for comparison, but label new exact samples with a suffix such as
`_sampled`. This is P1 rather than P0 because it does not change training and
the broad E3 trend is too large to depend on this detail.

### P1. The ranking objective is implemented but presently disabled

The model and loss already have matched `differentiable_rank` toggles. In that
mode the wrong-reference forward remains in the graph and the loss is:

```text
E_correct = face MSE(correct spatial reference)
E_wrong   = face MSE(shuffled spatial reference)

relative_gap = (E_wrong - E_correct) / stopgrad(E_correct)
L_rank = relu(relative_margin - relative_gap)
```

E3 deliberately used:

```yaml
reference_mode: detached_diagnostic
reference_weight: 0
```

That was correct for isolating architecture. It is now the most important
remaining experimental limitation. E4 should enable the existing mode before
adding architectural capacity.

The hinge stops contributing once the requested margin is achieved, and the
correct path retains full, face, and boundary reconstruction anchors. Even so,
monitor for the trivial solution of making wrong references destructive:

- correct-path full and face losses must not rise;
- contribution/native RMS must stay bounded;
- matched validation must improve, not merely shuffled validation worsen;
- fixed-checkpoint matched/shuffled outputs must remain structurally valid.

### P1. A nonzero floor at every site is now a quality constraint, not only a safety feature

The `.25` floor successfully prevented branch collapse. It also forces every
one of the 46 selected sites to retain a reference contribution even if a
specific layer/timestep is harmful to identity. E3's learned alpha moved down,
not up, while its output path learned to become more different from native.

Do **not** change the floor in E4; doing so would confound the objective test.
If E4 strengthens causality but still loses identity, the next isolated arm
should replace a universal per-site floor with a small global/group-level
reference-use budget and allow individual harmful sites to approach zero.
That preserves the BA core and causal constraint without forcing all sites to
be equally active.

### Verified paths that should not be changed in E4

The following contracts are supported by code, logs, and checkpoint evidence:

- target Q remains frozen and explicit;
- reference K/V use a true reference-face key mask;
- reference noise is identical between correct and shuffled forwards;
- only spatial reference latent/mask is shuffled; PhotoMaker identity inputs
  remain matched;
- RMS matching is detached and clipped;
- mix is bounded and has explicit diagnostic overrides;
- exactly 414 intended FP32 tensors are trainable and in the optimizer;
- all three roles receive gradients;
- schema-v2 saves and reloads the versioned v3 state;
- training and in-training validation install the same v3 processors;
- validation uses `validation_native` on RealVisXL;
- `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and branched CA-off hold.

## 3. Architectural and experimental improvements in priority order

## Priority 0 — run E3's fixed-checkpoint causal matrix

Before interpreting a new training objective, quantify the completed E3
checkpoint at the image level. Use the existing
`weights-epoch1.pth`, fixed 96 panel, fixed seeds, matched PhotoMaker identity
tokens, RealVis validation base, 50 inference steps, the existing scheduler and
CFG, and one image per item.

Required arms:

| Arm | Spatial reference | PhotoMaker ID inputs | Alpha | Purpose |
|---|---|---|---:|---|
| `matched` | correct | correct | learned | production E3 behavior |
| `shuffle_spatial` | different identity | correct | learned | incremental spatial causality |
| `zero_spatial` | zero latent/noise | correct | learned | dependence on spatial content |
| `branch_off` | correct | correct | 0 | exact native/PhotoMaker endpoint |
| `force_reference` | correct | correct | 1 | forced reference endpoint and wiring bound |

Each arm needs a unique run name, spec, immutable Comet ID, full metrics, all
96 images, `run_manifest.json`, and per-image metrics. A single shell launcher
should fail on a busy GPU, verify checkpoint hash, run one arm at a time, and
publish only after image count, manifest, and face-quality outputs pass.

The matrix is not a replacement for E4. Its purpose is to answer:

1. Does matched beat shuffled/zero on held-out identity similarity?
2. How much of v3's step-2k output is incremental to branch-off?
3. Is the learned mix near the best point on the alpha continuum?
4. Does alpha one expose structural damage that the learned mix currently
   suppresses?

## Priority 0 — E4: 2k differentiable reference-ranking arm

### 3.1 Scientific question

E4 asks one narrow question:

> Given an already live and causally sensitive rank-32 reference route, can a
> small differentiable correct-versus-shuffled margin redirect learning toward
> reference-specific usefulness without sacrificing PhotoMaker structure?

It must not simultaneously test more layers, more rank, a different mix
floor, a new dataset order, or identity-token dropout.

### 3.2 Exact controlled configuration

Create
`src/configs/big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_2k.yaml`:

```yaml
defaults:
  - big_celebs_scheduled_rhca_anchored_mix_sa_v3_2k
  - _self_

# E4 changes only the counterfactual objective and its sampling rate.
model:
  ba_reference_loss_mode: differentiable_rank
  ba_spatial_reference_shuffle_probability: 0.50

loss_function:
  reference_mode: differentiable_rank
  reference_weight: 0.10
  reference_margin: 0.0
  reference_relative_margin: 0.02

trainer:
  epoch_len: 2000
  n_epochs: 1
  validation_interval_steps: 2000

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
```

Everything inherited from E3 must remain unchanged:

```text
dataset plan              big_celebs_v2_policy_v1/train_40k_bs2.jsonl
schedule rows             first 4,000 rows
batch size                2
optimizer steps           2,000
seed/data order           identical
architecture              anchored_mix_sa_v3
reference/output rank     32 / 32
sites                     46: mid, up0, up1
mix init/floor/max        .50 / .25 / .90
RMS match/clip            on / [.5, 2.0]
role learning rates       5e-5 / 1e-4 / 2e-4
training timestep policy  inference-active
PhotoMaker start          10 / 50
BA start                  15 / 50
validation                fixed 96 at step 0 and 2,000
validation base           SG161222/RealVisXL_V4.0, validation_native
pose adaptation           0
branched CA               off
```

### 3.3 Why `weight=.10`, `margin=.02`, and probability `.50` are appropriate

E3's conditional relative gap is already about `0.009-0.012`. A 2% margin is
large enough to produce a gradient on most shuffled batches, but close enough
that it does not demand an arbitrary order-of-magnitude separation.

At the late E3 gap of roughly `0.009`, the hinge is about `0.011`; multiplying
by `.10` adds roughly `0.0011` to a total training loss around `0.4`. This is a
small redirecting term, not a replacement objective. A 50% probability yields
enough paired evidence in 2k steps while leaving half the steps at E3's normal
cost.

Do not tune these values from step-zero validation. Predeclare them in the
experiment JSON and change them only in a separately named retry if the first
three batches show a numerical or memory failure.

### 3.4 Planned launcher and experiment record

Create:

```text
launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_2k.sh
experiments/big_celebs/
  rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r1.json
```

Recommended run name:

```text
rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r1
```

The launcher should reuse the sealed BigCelebs preflights and differ from the
E3 launcher only in `CONFIG_NAME`, default `RUN_NAME`, and spec path:

```diff
--- a/launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_2k.sh
+++ b/launchers/neb/start_rhca_big_celebs_scheduled_anchored_mix_sa_v3_rank_2k.sh
@@
-export CONFIG_NAME="big_celebs_scheduled_rhca_anchored_mix_sa_v3_2k"
-export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r1}"
+export CONFIG_NAME="big_celebs_scheduled_rhca_anchored_mix_sa_v3_rank_2k"
+export RUN_NAME="${RUN_NAME:-rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r1}"
 export TRAIN_EPOCH_LEN=2000
 export TRAIN_EPOCHS=1
```

The experiment JSON must identify ladder step `E4` and explicitly record the
four changed objective values. It must retain the exact E3 fixed-control block
and expected 414 / 10,567,818 ownership contract. The saved directory must
write `comet_experiment.json` before validation is treated as accepted.

### 3.5 Expected runtime cost

E3 ran a detached second U-Net forward on about 25% of steps. E4 retains both
graphs on about 50% of steps. A material slowdown and higher training memory
are expected; this is not model construction or CPU execution. Profile the
first three training batches and record peak allocated/reserved CUDA memory.

Do not overlap validation with another GPU process. If the sequential
differentiable pair OOMs on the 80 GB GPU, stop the new run and implement
activation checkpointing/recomputation for the wrong path under a separate
retry name. Do not silently reduce batch size or validation panel, because
that changes schedule semantics and comparability.

### 3.6 Startup and step-zero gates

Before allowing training to continue:

1. GPU is idle and the intended process is on `cuda:0`.
2. Comet immutable record exists and names the E4 project/run.
3. Runtime hashes and sealed dataset schedule pass.
4. Exact optimizer contract remains 414 tensors / 10,567,818 parameters.
5. Model/loss reference modes both resolve to `differentiable_rank`.
6. Effective configured shuffle probability is `.50`.
7. Full-96 step-zero validation completes.
8. E4 step-zero images are byte-identical to E3 step zero, or a deterministic
   image audit explains any difference. Loss mode and training-only shuffle
   probability must not alter inference initialization.
9. The first three training batches are finite. An explicit forced-shuffle
   smoke batch, or a naturally shuffled one among those batches, shows a
   nonzero rank loss and gradients through both correct and wrong forwards;
   do not fail randomly just because three Bernoulli draws are unshuffled.
10. Reference K/V, reference output, and mix role gradients remain finite.

### 3.7 E4 decision gates at 2k

Hard gates for promotion:

| Category | Required result |
|---|---|
| Identity | Step-2k identity does not fall below E4's own step-zero anchor (`~0.49446`) |
| Causal gap | Last-500 conditional relative gap exceeds E3's `0.00877`; target `>=0.015` |
| Sign consistency | No repeated negative conditional windows in the final 500 steps |
| Weak-face tail | TOPIQ-Face p10 falls by no more than `0.01` from its own step zero |
| Text | No large reversal; target at least `26.5` under the fixed protocol |
| Contribution | Nontrivial but bounded; engineering range `0.10-0.65` of native RMS |
| Structure | No repeated pasted, displaced, doubled, malformed, or detached face cluster |
| Change type | Visual improvements include persistent identity morphology, not only expression/polish |
| Matrix | Matched spatial reference is better than shuffled/zero on paired identity evidence |

Stretch targets are identity above the v2 2k control (`0.50861`) and recovery
of the E3 step-zero TOPIQ-Face tail without losing the E3 text gain.

If identity still declines while the causal margin grows, E4 has answered its
question negatively: reference specificity in face-noise MSE is insufficient
to align the branch with identity. Do not extend that run to 40k.

## Priority 1 — add a contribution budget only if E4 becomes more causal but remains harmful

E3 proves that alpha alone does not control disruption; the output adapter can
rotate the reference vector. If E4 reaches the margin but still harms identity,
add a defaults-off contribution regularizer in a separately named E4b arm:

```text
r = RMS(mask * alpha * (reference - native)) / stopgrad(RMS(native))
L_budget = relu(r - r_max)^2 + relu(r_min - r)^2
```

Suggested initial bounds are `r_min=.10`, `r_max=.50`, with a small weight such
as `.01`. The lower bound prevents collapse; the upper bound prevents the
reference output delta from bypassing a falling alpha through vector rotation.

This requires exposing the sampled contribution statistic to the loss without
detaching only when the explicit budget toggle is enabled. Keep current
telemetry detached. Do not combine this with the first E4 arm, because E4 must
isolate whether ranking alone fixes direction.

## Priority 1 — allow site selectivity after causal use is secured

If E4/E4b retains positive causal use but some groups are harmful, replace the
universal `.25` per-site floor with:

- a lower per-site floor, e.g. `.05`;
- a group/global mean-use target or contribution lower bound;
- explicit per-group telemetry and checkpoint manifest fields.

This lets the model suppress a harmful decoder site while preserving a
nonzero reference-conditioned BA route overall. Compare one change at a time:

```text
E4 control:  floor .25 at every site
E4c:         floor .05 per site + global mean contribution lower bound
```

Do not use an unrestricted zero floor without the differentiable ranking loss
and a global use contract; that would recreate residual v2's shortcut.

## Priority 2 — add identity-specific supervision or memory if ranking MSE is insufficient

If E4 strengthens the correct/wrong diffusion-error gap but identity still
falls, the loss is distinguishing references using non-identity visual cues.
The next high-value architecture should append compact frozen
PhotoMaker-derived identity-memory tokens to the existing spatial K/V memory:

```text
K_memory = concat(K_reference_face_spatial, K_identity_tokens)
V_memory = concat(V_reference_face_spatial, V_identity_tokens)

Y_reference = Attention(Q_target, K_memory, V_memory)
```

Requirements:

- preserve target queries and the spatial reference K/V path;
- use two to four compact identity tokens rather than repeating a global vector
  at every spatial position;
- train only small `id_to_k`, `id_to_v`, and optional output-delta adapters;
- use a token-type embedding/bias so spatial and identity memory are explicit;
- log spatial versus ID-token attention mass and contributions separately;
- keep ordinary PhotoMaker conditioning unchanged in the first arm;
- add a new architecture/version or explicit defaults-off memory selector and
  checkpoint all token/projection semantics.

This remains branched attention: target Q explicitly consumes reference-derived
K/V memory. It is preferable to broadly unfreezing the target U-Net because it
adds identity-aligned reference capacity rather than another shortcut.

An alternative objective-only probe is a low-frequency frozen face-recognition
loss on predicted `x0` face crops. It is more expensive and introduces decoder
and face-alignment complexity, so it should be a separate ablation after E4,
not part of the first ranking run.

## Priority 3 — layer expansion, then rank

Only after an arm passes identity, causal, and structural gates:

1. add `down_blocks.2` at rank 32 while holding the objective and mix fixed;
2. compare output rank 32 versus 64;
3. compare K/V rank 32 versus 64.

Do not change layer coverage and rank in the same experiment. The present E3
result shows the route is already strong enough to hurt; capacity is not the
first-order problem.

## 4. Controlled implementation sequence

### D1 — existing E3 checkpoint matrix

**Training:** none.  
**Checkpoint:** immutable E3 r2 `weights-epoch1.pth`, hash recorded above.  
**Arms:** matched, shuffled spatial, zero spatial, alpha zero, alpha one.  
**Code changes:** evaluator v3 audit, alpha CLI/manifest, matrix launcher/specs.  
**Decision:** quantify held-out spatial causality and the learned alpha's
position between native and forced-reference endpoints.

### E4 — 2k objective-only arm

**Training:** fresh from the same base, not resumed from E3.  
**Changed:** differentiable ranking mode, `.10` weight, `.02` relative margin,
`.50` shuffle probability.  
**Unchanged:** every architectural, data, optimizer, schedule, and validation
control listed above.  
**Decision:** identity must stop declining while conditional causal separation
strengthens.

### E5 — 4k/8k promotion

Only if E4 passes. Reproduce the same objective from initialization with 2k
validation/checkpoint boundaries and extend to 4k, then 8k. Run the causal
matrix at each promoted checkpoint. Stop after two consecutive identity,
tail-quality, or causal regressions.

### E4b/E4c — bounded/selective branch fallback

Only if E4 is more causal but remains harmful. Test contribution budget first,
then site-selective floor, separately. Neither arm should add layers or rank.

### E6 — identity memory or direct identity supervision

Only if diffusion-error ranking remains non-identity-aligned. Preserve spatial
BA and add one identity-specific mechanism at a time.

### E7 — layer and rank capacity

Only after a preceding arm improves identity and passes the fixed matrix.

## 5. Verification checklist for the implementation

Use the existing local `photomaker` environment and the smallest checks that
can catch a real regression:

1. compile changed Python files;
2. compose E3 and E4 Hydra configs and diff resolved values;
3. assert only the four declared objective fields differ;
4. shell-syntax check both new launchers;
5. load the E3 checkpoint into `validation_native` v3 and assert 46 processors;
6. run alpha-zero native parity and alpha-one matched/shuffle separation on a
   small fixed batch;
7. verify the evaluator manifest records processor version, spatial condition,
   and alpha override;
8. run one detached E3 backward and one differentiable E4 backward, confirming
   the old/new toggle behavior and finite role gradients;
9. assert the exact 414-tensor optimizer contract in both modes;
10. verify v1 and residual-v2 config composition remains unchanged;
11. verify `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and CA-off in
    training and every validation arm;
12. on Neb, verify checkpoint hash, idle GPU, immutable Comet record, step-zero
    validation, and the first three batches before leaving E4 unattended.

No broad refactor or permanent new test suite is required for this experiment.
The changes should remain localized and defaults-off.

## 6. What not to do next

- Do not call E3 a failed branch implementation; it is the first run with
  strong evidence of causal spatial-reference use.
- Do not promote E3 on text or generic quality while identity and p10 regress.
- Do not continue E3 to 40k hoping that ordinary reconstruction will become
  identity-specific spontaneously.
- Do not increase rank or add encoder layers before the objective-only E4 arm.
- Do not lower the mix floor in E4; isolate the ranking objective first.
- Do not judge causality from matched images alone; run the intervention
  matrix.
- Do not compare raw E3 and E4 gap curves without correcting for 25% versus
  50% shuffle probability.
- Do not set `pose_adapt_ratio>0` or `ca_mixing_for_face=true`.
- Do not reactivate branched cross-attention in the same experiment.
- Do not change the fixed-96 validation contract, model base, prompts, seeds,
  references, bboxes, scheduler, inference steps, or metric definitions.

## Final recommendation

Keep anchored mix SA-v3 as the architecture base. It solved the exact problem
it was designed to solve: the target-query/reference-KV route is live,
trainable, visible, and consistently sensitive to reference identity. Reverting
to zero-start residual v2 would discard that progress, while reverting to
permanent hard replacement would reintroduce its structural risk.

Treat E3 r2 as a **positive causal control but a negative quality/identity
candidate**. Implement the evaluator and conditional-telemetry corrections,
run the existing checkpoint's five-arm matrix, then launch one fresh 2k E4 arm
with `.10` differentiable ranking weight, `.02` relative margin, and 50%
paired shuffling. Do not change rank, layers, mix bounds, data, or validation.

If E4 makes the causal advantage stronger and prevents the identity decline,
promote it cautiously to 4k/8k. If it becomes more causal but identity still
falls, the next problem is identity alignment—not capacity—and the appropriate
follow-up is bounded/site-selective contribution or explicit compact identity
memory inside the same branched-attention K/V route.
