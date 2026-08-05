# Residual SA-v2 at 2k: plain-PhotoMaker failure analysis and anchored BA-v3 design

**Date:** 2 August 2026  
**Run:** `rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6`  
**Immutable Comet ID:** `4d6186f56cd24ba3a907fa35406c284e`  
**Comet project:** `jul-comet-large-testing-tr`  
**Evidence cutoff:** fixed-96 validation at optimizer step 2,000; training
diagnostics through step 2,000 unless explicitly stated otherwise  
**Local evidence:**
[`comet_data/rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6/`](../comet_data/rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6/)  
**Status:** investigation and design only. No model code was changed and the
live Neb process was not stopped by this analysis.

This report updates the residual-branch recommendation in
[`2026-08-02_clean_ba32_32k_architecture_recommendations.md`](2026-08-02_clean_ba32_32k_architecture_recommendations.md).
The earlier report correctly fixed key masking, trainable ownership, precision,
checkpointing, and validation-base semantics. The first live residual-SA-v2
result now exposes a new, more fundamental problem: the branch is mechanically
present but is initialized and trained as an optional near-zero correction to
an already strong PhotoMaker solution.

## Executive verdict

The run is **not failing because the residual processor was omitted, its
parameters were left on CPU, the optimizer ignored it, the checkpoint failed
to save, or validation failed to load it**. Those paths all have direct
evidence of working.

It is also not correct to say that useful branched attention is working. The
precise conclusion is:

> At step zero the explicit spatial-reference branch contributes exactly zero
> by construction. By step 2k its weights are nonzero and it perturbs every
> validation image slightly, but there is no measurable evidence that the
> output depends beneficially on which spatial reference is supplied.

The core causal diagnostic is decisive. Keeping the target, noise, timestep,
prompt, and correct PhotoMaker identity tokens fixed while shuffling only the
spatial reference produces an average face-error gap centered essentially at
zero. The branch is learning a small generic face correction, not a reliable
target-query/reference-KV identity route.

The architecture makes this outcome easy:

```text
current v2 target output
    = complete frozen PhotoMaker target self-attention
    + face mask
      * approximately 0.10 gate
      * zero-initialized low-rank output(reference attention)
```

The ordinary denoising loss can be reduced through the already correct
PhotoMaker identity-conditioning path. It never requires the redundant
spatial branch to become reference-causal. The prepared wrong-reference loss
has weight zero, and its negative forward is detached.

The recommended repair is **not** simply rank 64. It is a versioned,
defaults-off **anchored interpolation BA-v3** that connects two already known
endpoints:

```text
alpha = 0   -> exact frozen PhotoMaker target attention
alpha = 1   -> target Q consumes reference K/V as the face attention message

target output
    = native target output
    + face mask * alpha * (reference output - native target output)
```

Unlike v2, `reference output` must contain a nonzero frozen native output
projection from initialization; a trainable low-rank output delta is added on
top. `alpha` must have a nonzero floor and should start around `0.50` in the
first controlled arm. This retains PhotoMaker structure while making the
reference route visible and trainable from update one. A differentiable
matched-versus-shuffled reference-ranking loss and runtime branch-contribution
telemetry should be implemented behind separate toggles.

The old hard-replacement architecture is useful as an endpoint and parity
test, but should not simply be restored as the final model. It forced reference
use, yet it erased the native target self-attention message inside the face and
previously produced a severe early regression and a later identity ceiling.

## Evidence boundary

The downloaded folder contains 96 matched validation images at step 0 and the
same 96 items at step 2,000, plus a Comet export. It does **not** contain a
fixed-checkpoint step-2k branch-off/zero-reference/shuffled-reference image
matrix. Therefore:

- the processor, optimizer, checkpoint, gradients, matched images, scalar
  metrics, and training-time shuffled-reference diagnostic can be audited now;
- a full validation-level causal intervention remains a required experiment;
- the absence of reference dependence is already high-confidence, but the
  report does not pretend that the missing intervention was run;
- no validation controls, seeds, prompts, references, bboxes, scheduler,
  inference steps, CFG, or metric definitions were changed for this analysis.

## 1. What this training behaviour uncovers

### 1.1 Step zero is plain PhotoMaker by mathematical construction

The active processor in
[`residual_sa_processor_v2.py`](../src/model/photomaker_branched/residual_sa_processor_v2.py)
computes the complete native target message first:

```python
base_message = scaled_dot_product_attention(q_target, k_target, v_target)
base_out = attn.to_out(base_message)
```

It then computes target-query/reference-KV attention, but sends it through
`ResidualLoRALinear`:

```python
reference_message = scaled_dot_product_attention(
    q_target, k_reference, v_reference, attn_mask=reference_key_bias
)
reference_delta = self.ref_out(reference_message)
target_out = base_out + target_mask * gate * reference_delta
```

`ResidualLoRALinear.lora_B` is initialized to all zeros and has no frozen/base
linear term. Consequently:

```text
reference_delta(step 0) = 0 exactly
target_out(step 0)       = base_out exactly
```

This was a deliberate safe initialization, not a runtime accident. The local
processor smoke test also verified equality to frozen native target
self-attention at zero initialization. The user's observation that the images
look exactly like ordinary PhotoMaker is therefore expected for v2.

The problem is not that step zero is stable. The problem is that the model is
given no architectural or objective pressure to leave that stable solution in
a reference-specific direction.

### 1.2 Step 2k is not pixel-identical, but the change is modest and preserves the same solution

An exact filename-paired pixel audit across all 96 images found:

| Statistic, step 0 versus step 2k | Result |
|---|---:|
| Paired images | 96 |
| Pixel-identical pairs | 0 |
| RGB absolute-error mean, average over images | 3.371 / 255 |
| RGB absolute-error median | 3.186 / 255 |
| RGB absolute-error range | 1.619–7.112 / 255 |
| Mean PSNR | 30.71 dB |
| Median PSNR | 30.63 dB |
| Mean fraction of channels changing by more than 2 levels | 31.9% |

The largest pixel changes occur in several laughing and crying examples; the
smallest occur in jumping examples. Direct visual inspection agrees with the
user: facial details and local texture move, but scene, body, pose, face shape,
and overall PhotoMaker interpretation remain essentially the same.

This distinction matters. The checkpoint has an effect, so “the weights were
not loaded” is not a viable explanation. But nonzero pixels do not establish
that the correct reference caused the change.

### 1.3 The direction of the 2k change is not an identity improvement

The fixed-96 aggregate metrics are:

| Metric | Step 0 | Step 2k | Change |
|---|---:|---:|---:|
| Identity similarity | 0.523562 | 0.508613 | **-0.014949** |
| Text similarity | 26.3335 | 26.7218 | +0.3883 |
| TOPIQ-Face mean | 0.747306 | 0.735905 | **-0.011401** |
| TOPIQ-Face p10 | 0.593550 | 0.591600 | -0.001950 |
| Face detection/coverage | 1.0000 | 1.0000 | 0 |
| Generic TOPIQ mean | 0.610485 | 0.614622 | +0.004137 |
| MUSIQ mean | 72.9293 | 73.2582 | +0.3288 |
| MANIQA mean | 0.645372 | 0.644227 | -0.001145 |

The learned perturbation slightly improves prompt and generic image-quality
scores while reducing both identity similarity and crop-based face quality.
This is consistent with a generic denoising/style correction around the
PhotoMaker basin, not successful extraction of extra identity information from
the spatial reference.

The step-0 identity score should not be compared naively to every old hard-BA
score because historical processor-base validation modes differ. It does,
however, make one practical point clear: “looking unlike PhotoMaker at step
zero” is only a causal-routing signal, not automatically an improvement.

### 1.4 The spatial-reference causal signal is effectively zero

The run uses a 25% probability diagnostic. On selected batch-size-two updates,
the code keeps all of these fixed:

- target noisy latent;
- target noise and timestep;
- generation prompt;
- correct PhotoMaker-fused identity tokens;
- target face mask.

It shifts only the spatial reference latent and reference bbox mask to a
different identity. That is the correct intervention for asking whether the
explicit spatial BA route matters in addition to ordinary PhotoMaker
conditioning.

Through step 2k:

| Diagnostic | Value |
|---|---:|
| Mean logged shuffle application rate | 0.2517 |
| Mean `wrong_face_mse - correct_face_mse`, including unshuffled zeros | **-2.60e-6** |
| Minimum logged gap | -3.39e-4 |
| Maximum logged gap | +2.14e-4 |
| Mean diagnostic causal hinge | 3.18e-5 |
| Configured causal-loss weight | **0.0** |
| Configured margin | **0.0** |

A useful correct-reference route should make the wrong reference consistently
worse, yielding a positive gap. Instead the sign oscillates around zero. Even
rescaling the window average approximately by the 25% application rate leaves
the effect extremely small.

This is the strongest available answer to “is branched attention working at
all?”:

- **graph/optimizer level:** yes;
- **visible nonzero perturbation level:** weakly yes by 2k;
- **causal identity/reference-conditioning level:** no evidence of useful
  operation, and the measured evidence is consistent with near-independence.

### 1.5 Gradients and checkpoint contents rule out a dead optimizer path

The first saved 2k weights file on Neb contains the exact schema-v2 contract:

```text
processor sites                   46
trainable tensors                414
trainable parameters      10,567,818
trainable dtype                 FP32
selected groups        mid, up0, up1
```

All 46 LoRA-B matrices in each of `ref_to_k`, `ref_to_v`, and `ref_out` are
nonzero at 2k. Their checkpoint RMS summaries are:

| Tensor family | Mean B-matrix RMS | Min | Max | Exactly zero sites |
|---|---:|---:|---:|---:|
| Reference K LoRA B | 0.004365 | 0.000336 | 0.011171 | 0/46 |
| Reference V LoRA B | 0.006563 | 0.000555 | 0.014973 | 0/46 |
| Reference output LoRA B | 0.008352 | 0.004914 | 0.014193 | 0/46 |

Comet gradient norms through step 2k also show all optimizer roles receiving
gradients:

| Role | First log | Mean | Maximum | Last log at/before 2k |
|---|---:|---:|---:|---:|
| Gate | 0 | 0.005643 | 0.07477 | 0.000183 |
| Reference K/V | 0 | 0.003922 | 0.04071 | 0.000549 |
| Reference output | 0.000359 | 0.006622 | 0.04926 | 0.001924 |
| Total | 0.000359 | 0.009937 | 0.09836 | 0.002010 |

The zero first-update gradients for gates and reference K/V are themselves
explained by the architecture. With `ref_out.B = 0`, the reference branch
output is zero, so update one can reach `ref_out.B` but cannot yet provide a
gradient to the gate, `ref_out.A`, or reference K/V. Those components only
start learning after the output B matrix becomes nonzero.

This creates an unnecessary staged bottleneck at exactly the point where the
branch is competing with a complete pretrained shortcut.

### 1.6 The learned gate remains near its weak initialization

At 2k, the 46 checkpoint gates have:

```text
sigmoid(gate_logit): min 0.101733, mean 0.103366, max 0.104751
gate_t:              min -0.022697, mean 0.038195, max 0.066721
gate_area:           min -0.059733, mean -0.041884, max -0.018069
```

The base gate has moved only from 0.100 to roughly 0.103. The timestep and
face-area terms modulate it, but they do not change the central diagnosis: a
small learned low-rank output is multiplied by about one tenth before being
added to a full native PhotoMaker message.

The gate did not numerically collapse to zero. It did something almost as bad
for this experiment: it remained weak enough that ordinary reconstruction did
not need to make the branch identity-causal.

### 1.7 The old structure forced causality but had the opposite failure mode

The historical processor in
[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py)
computes, inside the target face:

```text
Q = target/noise face queries
K,V = reference-face features
face attention message = Attention(Q_target, K_reference, V_reference)
```

It then replaces the target self-attention message inside the face mask with
that reference message at full scale. There is no complete native target
self-attention message inside the face to fall back to. Therefore its step-zero
images must differ from PhotoMaker, which matches the user's recollection.

That property is valuable: a shuffled or zero reference cannot be silently
ignored. But permanent hard replacement creates a difficult structure/identity
trade-off. The earlier clean BA32 run fell sharply at 2k, recovered later, and
then plateaued. The correct next step is to bridge the old and new endpoints,
not to choose either extreme permanently.

### 1.8 More training is unlikely to repair this objective by itself

Nothing prevents v2 from continuing to change. The active run may yield a
useful long negative control. But its objective still rewards correct denoising
without requiring any correct-versus-wrong spatial-reference separation, and
its strongest conditioning path is still ordinary PhotoMaker.

Additional steps can increase a generic residual just as easily as a
reference-specific residual. The negative identity change and near-zero
shuffle gap at 2k give no basis for expecting a spontaneous late transition
to strong causal reference use. Rank 64 would enlarge the same optional path;
it would not remove the shortcut or the zero-start bottleneck.

## 2. High-priority issues in the current code

### P0. The reference output is exactly zero at initialization

Current implementation:

```python
self.ref_out = ResidualLoRALinear(
    hidden_size,
    rank=output_rank,
    zero_init_output=True,
)

# ResidualLoRALinear has no base output term.
reference_delta = self.ref_out(reference_message)
target_out = base_out + target_mask * gate * reference_delta
```

Consequences:

1. step zero is exactly PhotoMaker;
2. the gate and K/V adapters receive zero gradient on the first update;
3. the model must first learn an output basis before it can learn which
   reference keys and values should be useful;
4. the full PhotoMaker path receives an immediate, well-scaled output while
   the new path starts with no output at all;
5. a low gate multiplies the already-small learned output.

Zero-initializing only the **trainable delta** is sensible. Zero-initializing
the entire reference message is the mistake. The branch needs a frozen native
output projection from the start.

### P0. The merge parameterization makes PhotoMaker the unconstrained optimum

The current additive form is:

```text
Y = Y_native + M_target * gate * delta_reference
```

`delta_reference = 0` is always a valid solution. There is no lower bound on
functional reference contribution and no requirement that a different
reference produce a different result. The architecture protects PhotoMaker
structure so completely that it also protects PhotoMaker from having to use
the new connection.

The next merge must represent an explicit continuum between native target
attention and reference attention, with a configurable nonzero floor.

### P0. The training objective does not reward spatial-reference causality

The run configuration explicitly has:

```yaml
loss_function:
  reference_weight: 0.0
  reference_margin: 0.0
model:
  ba_spatial_reference_shuffle_probability: 0.25
```

The wrong-reference forward in
[`lora2.py`](../src/model/photomaker_branched/lora2.py) is under
`torch.no_grad()`. The loss in
[`branched_reference_loss.py`](../src/loss/branched_reference_loss.py) also
detaches it:

```python
wrong_face = masked_face_mse(pred_wrong_spatial_ref.detach(), target)
gap = wrong_face.detach() - correct_face
causal = relu(margin - gap)
loss += reference_weight * causal
```

With weight zero, this is diagnostic only. Merely setting the current weight
nonzero is not the preferred final fix: because the wrong branch is detached,
an equal correct/wrong pair behaves primarily like another correct-face
reconstruction term. It does not apply a direct counterfactual gradient that
pushes correct and wrong references apart.

The negative forward should remain detached only in diagnostic mode. In causal
training mode both correct and wrong predictions must participate in the
ranking objective.

### P0. Ordinary PhotoMaker identity conditioning is a strong redundant shortcut

During the spatial shuffle, the correct PhotoMaker-fused prompt/ID tokens are
kept fixed intentionally. This is a clean diagnostic, and it reveals the
shortcut: the target can retain the right identity without consulting the
spatial lane.

This is not a reason to remove PhotoMaker from the production model. The
project goal is to improve it. It is a reason to add causal supervision and,
if necessary, a small defaults-off training-only PhotoMaker-ID-token dropout
arm while leaving validation unchanged.

### P1. Runtime observability measures weights and gradients, not branch use

The current logs establish optimizer activity but do not record the quantities
that decide whether BA is functionally active:

- RMS of native target attention output;
- RMS of the ungated reference output;
- RMS of the gated face-local contribution;
- `reference/native` contribution ratio;
- matched-versus-shuffled U-Net output difference;
- gate distribution by U-Net group and denoising-time band;
- target/reference mask coverage at each token resolution;
- reference attention entropy and valid-key count.

Without these, a 10.6M-parameter checkpoint can look healthy while being
causally irrelevant. This is a high-priority code gap, not optional debugging
polish.

### P1. The branch output has no scale contract with the native output

`ref_out(reference_message)` is a learned low-rank vector with no frozen basis
and no normalization relative to `base_out`. The gate must simultaneously
learn whether to use the branch and compensate for an arbitrary output scale.

The new direct reference output should use the same frozen native output
projection as target attention, then apply a trainable delta. A detached,
clipped per-sample RMS match can keep interpolation numerically meaningful.

### P1. Standard matched validation cannot detect an ignored reference branch

Step-0 and step-2k matched images answer whether the complete model changed.
They do not answer whether the supplied spatial reference caused the change.
Every promoted checkpoint needs a fixed-seed intervention matrix:

1. matched spatial reference;
2. shuffled spatial reference;
3. zero spatial reference;
4. branch disabled / `alpha=0` PhotoMaker;
5. forced-reference diagnostic / `alpha=1`.

PhotoMaker identity tokens must remain matched in arms 1–4 when measuring the
incremental spatial route. Each arm must log the same metrics and all 96 images
to its own immutable Comet experiment.

### P1. Only mid and decoder attention sites receive the new branch

The v2 arm patches 46 sites in `mid_block`, `up_blocks.0`, and `up_blocks.1`.
The old forced architecture also included 24 encoder-side sites in
`down_blocks.1` and `down_blocks.2`.

This is not the reason step zero is PhotoMaker—the zero output is. It may become
a capacity ceiling after causal use is repaired. The first v3 experiment
should keep the same 46 sites for isolation; the next layer-expansion arm
should add `down_blocks.2` before increasing rank globally.

### P2. The spatial reference is noised at the target timestep

[`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py)
adds noise to the reference latent at the same scheduler timestep as the
target. This keeps target/reference feature statistics compatible, but it also
weakens fine identity detail at noisier active timesteps. A small optional
persistent low-noise reference-memory lane is a later architectural candidate.
It should not be implemented until the simpler v3 route passes causal gates.

### P2. Several historical ID-embedding knobs are wired but inactive here

The run uses `pipeline.face_embed_strategy=id` and `model.use_id_embeds=false`.
The v2 processor does not consume `proc.id_embeds`; identity still reaches the
target through normal PhotoMaker prompt embeddings and reaches the reference
lane through its prompt handling. This is not a hidden reason for the current
zero-start behaviour.

For a later stronger branch, PhotoMaker's extracted 2048-D identity features
can be projected into a small set of explicit reference-memory K/V tokens. That
must be a new documented connection, not an assumption that the current
`id_embeds` assignment already has an effect.

### Verified paths that should not be “fixed” again

The following parts passed this audit:

- residual-SA-v2 processors are installed at all expected 46 sites;
- the exact 414-tensor optimizer ownership contract passes;
- all trainables are FP32;
- reference invalid tokens are excluded from the attention softmax with a true
  key bias;
- target Q remains the explicit query source;
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false` are preserved;
- branched CA is disabled;
- training samples inference-active DDIM timesteps;
- schema-v2 saves and reloads the exact trainable state;
- alternate-base validation reconstructs the same v2 architecture and loads
  its live state;
- every step-2k image differs from step zero, independently confirming that
  loaded trainables affect inference.

## 3. Architectural improvements in priority order

## Priority 0 — implement anchored interpolation BA-v3

### 3.1 Required attention equations

For each selected self-attention site, keep the doubled batch layout
`[target, reference]` and compute:

```text
Q_t = Wq_native(H_target)
K_t = Wk_native(H_target)
V_t = Wv_native(H_target)

K_r = Wk_native(H_reference) + LoRA_k(H_reference)
V_r = Wv_native(H_reference) + LoRA_v(H_reference)

A_t = Attention(Q_t, K_t, V_t)
A_r = Attention(Q_t, K_r, V_r, valid_reference_face_keys_only)

Y_t = Wo_native(A_t)
Y_r = Wo_native(A_r) + DeltaWo_r(A_r)

alpha = alpha_floor
      + (alpha_max - alpha_floor) * sigmoid(g_layer + g_timestep + g_area)

Y_target = Y_t + M_target * alpha * (RMSMatch(Y_r, Y_t) - Y_t)
```

Key invariants:

- query is always the target query;
- reference K/V are explicit and reference-face-key-masked;
- `pose_adapt_ratio` remains zero;
- native target attention is exact when `alpha=0`;
- old hard face-message replacement is approached when `alpha=1`;
- outside the target face mask, the native target path is unchanged;
- the trainable output delta may remain zero-initialized because `Wo_native(A_r)`
  supplies a nonzero reference message from update one;
- reference K/V LoRA-B, output LoRA-B, and alpha all receive gradients on the
  first update;
- the branch cannot become functionally absent while `alpha_floor > 0`.

### 3.2 Why interpolation is preferable to another additive residual

An additive direct reference output,
`Y_t + alpha * Y_r`, can double attention-message energy and force the gate to
learn amplitude correction. Interpolation,
`Y_t + alpha * (Y_r - Y_t)`, gives alpha a clear meaning and connects two known
architectures:

| Alpha | Meaning |
|---:|---|
| 0.00 | exact PhotoMaker/native target self-attention |
| 0.25 | native-dominant but guaranteed reference use |
| 0.50 | equal native/reference interpolation inside face |
| 0.75 | reference-dominant, close to old behaviour |
| 1.00 | forced reference face-message diagnostic |

The first controlled training candidate should use:

```yaml
ba_mix_init: 0.50
ba_mix_floor: 0.25
ba_mix_max: 0.90
ba_reference_rms_match: true
ba_reference_rms_clip_min: 0.50
ba_reference_rms_clip_max: 2.00
```

Before training, run the same untrained model at alpha
`0, 0.25, 0.50, 0.75, 1.0` on a small fixed subset. If `0.50` is not visibly
different from `0`, that is a wiring failure. If `0.50` already causes severe
face/body damage, select `0.35` for the first arm but keep a floor of at least
`0.20`.

### 3.3 Processor implementation diff

Create a new processor rather than silently changing v2:

```diff
--- a/src/model/photomaker_branched/residual_sa_processor_v2.py
+++ b/src/model/photomaker_branched/anchored_mix_sa_processor_v3.py
@@
-class ResidualBranchedSelfAttnProcessorV2(nn.Module):
-    architecture_version = "residual_sa_v2"
+class AnchoredMixBranchedSelfAttnProcessorV3(nn.Module):
+    architecture_version = "anchored_mix_sa_v3"
@@
-    gate_init: float = 0.10,
-    gate_max: float = 1.0,
+    mix_init: float = 0.50,
+    mix_floor: float = 0.25,
+    mix_max: float = 0.90,
+    reference_rms_match: bool = True,
+    reference_rms_clip_min: float = 0.50,
+    reference_rms_clip_max: float = 2.00,
@@
-    return gate_max * torch.sigmoid(logits)
+    unit = torch.sigmoid(logits)
+    return mix_floor + (mix_max - mix_floor) * unit
@@
     reference_message = F.scaled_dot_product_attention(
         q_target,
         k_reference,
         v_reference,
         attn_mask=key_bias,
         dropout_p=0.0,
         is_causal=False,
     )
-    reference_delta = self.ref_out(self._merge_heads(reference_message))
+    reference_message = self._merge_heads(reference_message)
+    # Frozen native projection gives a nonzero reference path at initialization.
+    reference_base = self._apply_output_projection(attn, reference_message)
+    reference_out = reference_base + self.ref_out(reference_message)
+
+    if self.reference_rms_match:
+        reference_out = self._rms_match_reference(
+            reference_out,
+            base_out,
+            target_mask,
+        )
@@
-    target_out = base_out + target_mask * gate * reference_delta * self.scale
+    alpha = self._bounded_mix(...).to(dtype=base_out.dtype)
+    target_out = base_out + target_mask * alpha * (
+        reference_out - base_out
+    ) * self.scale
```

Use a detached, clipped scale for RMS matching so the model cannot optimize the
normalizer itself:

```python
def _rms_match_reference(self, reference, native, target_mask):
    mask = target_mask.float()
    denom = (mask.sum(dim=(1, 2), keepdim=True) * native.shape[-1]).clamp_min(1.0)
    native_rms = ((native.float().square() * mask).sum((1, 2), keepdim=True) / denom).sqrt()
    ref_rms = ((reference.float().square() * mask).sum((1, 2), keepdim=True) / denom).sqrt()
    ratio = (native_rms / ref_rms.clamp_min(1.0e-6)).clamp(
        self.reference_rms_clip_min,
        self.reference_rms_clip_max,
    ).detach()
    return reference * ratio.to(reference.dtype)
```

Compute `target_mask` before RMS matching. Preserve all v2 fail-closed checks
for square tokens, nonempty reference key masks, doubled-batch layout,
denoising-progress shape, and dtype.

### 3.4 Correct initialization helper

The current `_logit` assumes a `[0, 1]` gate. V3 needs to initialize a bounded
interval correctly:

```python
def _bounded_logit(value: float, lower: float, upper: float) -> float:
    if not lower < value < upper:
        raise ValueError(
            f"mix_init must be strictly inside ({lower}, {upper}), got {value}"
        )
    probability = (value - lower) / (upper - lower)
    return math.log(probability / (1.0 - probability))
```

The architecture manifest must store `mix_init`, `mix_floor`, `mix_max`, RMS
settings, merge equation version, and reference-output-base mode. Loading a v2
checkpoint into v3 must fail unless an explicit conversion utility is invoked.

### 3.5 Keep every old behaviour reversible

Top-level selection should remain:

```yaml
model:
  ba_architecture_version: anchored_mix_sa_v3
```

Supported version meanings:

```text
hard_replace_v1      historical forced face-message replacement
residual_sa_v2       current exact-PhotoMaker zero-delta residual
anchored_mix_sa_v3   new nonzero reference/native interpolation
```

Do not change the default in the shared model config. Existing launchers and
checkpoints must replay their historical processor class. A new v3 config and
launcher should opt in explicitly.

## Priority 0 — add branch-use telemetry and fail-closed causal gates

### 3.6 Runtime metrics

Each v3 processor should accumulate detached FP32 summaries during a bounded
sample of forwards, not retain activations:

```python
with torch.no_grad():
    native_face_rms = masked_rms(base_out, target_mask)
    reference_face_rms = masked_rms(reference_out, target_mask)
    contribution = target_mask * alpha * (reference_out - base_out)
    contribution_rms = masked_rms(contribution, target_mask)
    contribution_ratio = contribution_rms / native_face_rms.clamp_min(1e-8)
```

Log at least these Comet series, aggregated by `mid`, `up0`, `up1` and by
early/middle/late active denoising timestep:

```text
train/ba/alpha_mean
train/ba/alpha_min
train/ba/alpha_max
train/ba/reference_native_rms_ratio
train/ba/contribution_native_rms_ratio
train/ba/reference_valid_key_fraction
train/ba/reference_attention_entropy
train/ba/correct_wrong_pred_delta_ratio
train/ba/correct_wrong_face_error_relative_gap
```

Add configuration for a sampling interval so this does not materially slow
training. One detailed batch every 50 optimizer steps is enough initially.

### 3.7 Startup and early-stop assertions

The launcher should run a pretraining causal smoke on one fixed batch:

1. `alpha=0` equals the native PhotoMaker processor within numerical tolerance;
2. `alpha=1` differs from `alpha=0` inside the target face;
3. matched and shuffled references differ at `alpha=1`;
4. pixels/tensors outside the target mask remain native within tolerance;
5. invalid reference keys have exactly zero influence;
6. all v3 trainables receive finite gradients on the first backward;
7. trainable and optimizer manifests match exactly.

At step 100 or 250, fail the short smoke run if either condition holds:

```text
mean contribution/native RMS ratio < 0.05
mean absolute matched-vs-shuffled prediction delta ratio < 0.005
```

These thresholds are initial engineering gates, not scientific claims. Record
the observed distributions and adjust once, before launching 40k. The current
v2 result should fail them, which is the intended discrimination.

## Priority 1 — make matched-versus-wrong reference training differentiable

### 3.8 Loss definition

On a configured fraction of updates, compute correct and cross-identity
spatial-reference predictions while keeping target and PhotoMaker conditioning
fixed:

```text
E_correct = face_mse(pred_correct, target_noise)
E_wrong   = face_mse(pred_wrong,   target_noise)

relative_gap = (E_wrong - E_correct) / stopgrad(E_correct + eps)
L_reference_rank = relu(relative_margin - relative_gap)
```

Recommended first values:

```yaml
model:
  ba_spatial_reference_shuffle_probability: 0.50
loss_function:
  reference_mode: differentiable_rank
  reference_weight: 0.10
  reference_relative_margin: 0.02
```

Both predictions must remain differentiable when `reference_weight > 0`.
Retain the current no-grad path when `reference_mode=detached_diagnostic` or
weight is zero.

### 3.9 Loss implementation diff

```diff
--- a/src/loss/branched_reference_loss.py
+++ b/src/loss/branched_reference_loss.py
@@
-wrong_face = _masked_face_mse(
-    pred_wrong_spatial_ref.detach(), target, face_bbox
-)
-gap_for_loss = wrong_face.detach() - face
-reference_causal = F.relu(self.reference_margin - gap_for_loss)
+wrong_face = _masked_face_mse(
+    pred_wrong_spatial_ref, target, face_bbox
+)
+gap_for_log = wrong_face.detach() - face.detach()
+relative_gap = (wrong_face - face) / face.detach().clamp_min(1.0e-6)
+reference_causal = F.relu(
+    self.reference_relative_margin - relative_gap
+)
 loss = loss + self.reference_weight * reference_causal
@@
-"reference_error_gap": gap_for_loss.detach(),
+"reference_error_gap": gap_for_log,
+"reference_error_relative_gap": relative_gap.detach(),
+"loss_wrong_reference_face": wrong_face.detach(),
```

And in the model forward:

```diff
--- a/src/model/photomaker_branched/lora2.py
+++ b/src/model/photomaker_branched/lora2.py
@@
-with torch.no_grad():
-    wrong_spatial_reference_pred = run_branched_forward_pass(...)
+if self.ba_reference_loss_mode == "differentiable_rank":
+    wrong_spatial_reference_pred = run_branched_forward_pass(...)
+else:
+    with torch.no_grad():
+        wrong_spatial_reference_pred = run_branched_forward_pass(...)
```

Keep the same per-target reference noise realization between correct and wrong
forwards. Only reference content and its bbox mask should change; otherwise the
prediction delta is confounded by random reference noise.

For efficiency, an optional later implementation may concatenate correct and
wrong pairs into one larger doubled-lane U-Net call. First implement the
sequential version because its invariants are easier to audit. Profile memory
before selecting the production path.

### 3.10 Prevent trivial negative-reference sabotage

A ranking loss can be gamed by making every mismatched reference arbitrarily
bad. Keep these anchors:

- full-image reconstruction on the correct path;
- face reconstruction on the correct path;
- boundary-ring reconstruction;
- small reference-ranking weight initially;
- wrong-reference loss applied only to the target face;
- contribution-ratio upper monitoring;
- matched fixed-96 visual quality and text metrics.

Do not add an objective that merely maximizes `pred_correct - pred_wrong`
without tying the correct prediction to the diffusion target.

## Priority 1 — add a fixed-checkpoint v2/v3 causal validation matrix

### 3.11 Required arms

For the existing 2k v2 checkpoint, and later for every v3 gate, evaluate:

| Arm | Spatial reference | PhotoMaker ID tokens | Merge override | Purpose |
|---|---|---|---|---|
| `matched` | correct | correct | trained | production behaviour |
| `shuffle_spatial` | wrong identity | correct | trained | incremental spatial causality |
| `zero_spatial` | zero latent, valid lane | correct | trained | spatial input dependence |
| `branch_off` | correct | correct | alpha 0 / branch disabled | exact PM anchor |
| `force_reference` | correct | correct | alpha 1 | wiring/upper-bound diagnostic |

Use the same 96 validation items, prompts, seeds, reference assets, generated
and reference bboxes, RealVis validation base, scheduler, inference steps, CFG,
and metric definitions. Log each arm to a distinct immutable Comet experiment
with all images.

Add paired image/latent statistics, not only aggregate ID score:

```text
matched vs branch_off pixel MAE and LPIPS
matched vs shuffled pixel MAE and LPIPS
matched vs zero pixel MAE and LPIPS
face-region versions of the same
per-identity ID-sim deltas
```

The v2 prediction is that `matched`, `shuffle_spatial`, `zero_spatial`, and
`branch_off` will be structurally very close, while `force_reference` will
finally expose the latent reference path. This experiment is a diagnosis, not
a reason to continue v2 to 40k.

## Priority 2 — remove the PhotoMaker shortcut on a small training fraction

After v3 and the ranking loss work, add a defaults-off training-only ablation:

```yaml
model:
  ba_photomaker_id_dropout_probability: 0.15
```

On selected training samples:

- preserve text prompt tokens;
- remove or replace only PhotoMaker ID-enhanced tokens in the target lane;
- retain the correct spatial reference and v3 BA path;
- keep standard validation completely unchanged.

This tests whether the spatial branch can carry identity when the redundant
shortcut is temporarily unavailable. Start at 10–15%, not 50%. It is a
regularizer and causal probe, not a new validation protocol.

Promote it only if matched validation improves and ordinary PhotoMaker prompt
adherence is retained. Do not combine it with the first v3 merge-only arm;
isolate the architectural repair first.

## Priority 2 — expand selected layers before increasing rank

Once the 46-site v3 model passes the causal gate, compare:

```yaml
# Controlled v3 base
ba_self_attention_groups:
  - mid_block
  - up_blocks.0
  - up_blocks.1

# Encoder-augmented arm
ba_self_attention_groups:
  - down_blocks.2
  - mid_block
  - up_blocks.0
  - up_blocks.1
```

Adding `down_blocks.2` restores 20 high-width encoder-side connections where
identity can influence features before the bottleneck and skip path. Keep
rank, loss, mix floor, schedule, data order, and validation fixed. Add
`down_blocks.1` only if the down2 arm helps without high-resolution seams or
pose damage.

Do not change layer coverage and rank in the same first comparison.

## Priority 3 — append compact identity-memory tokens to reference K/V

Raw spatial reference features are valuable for geometry and local appearance,
but a noised VAE/UNet lane may lose fine identity information. A stronger BA
connection can append a few identity-memory tokens derived from PhotoMaker's
frozen extracted features:

```text
S_id = frozen PhotoMaker 2048-D identity features
K_id = Linear_or_LoRA_k_id(S_id)
V_id = Linear_or_LoRA_v_id(S_id)

K_memory = concat(K_reference_face_spatial, K_id)
V_memory = concat(V_reference_face_spatial, V_id)

Y_reference = Attention(Q_target, K_memory, V_memory)
```

Requirements:

- keep spatial reference face keys; ID tokens augment rather than replace them;
- use two to four ID tokens, not one repeated global vector at every spatial
  position;
- tag/token-type encode spatial versus identity memory;
- include all ID tokens in the key mask while masking invalid spatial keys;
- keep target Q frozen;
- give the ID-token output a separately logged gate or contribution ratio;
- keep ordinary PhotoMaker cross-attention unchanged in the first arm;
- train `id_to_k`, `id_to_v`, and a small output adapter in FP32;
- checkpoint the exact token count, projection shapes, and token-type mode.

This remains explicit branched attention: target queries consume a
reference-derived memory. It should be tested only after the spatial v3 path
itself is causal; otherwise it can hide another broken spatial route.

## Priority 3 — add persistent low-noise reference memory

If identity remains limited at noisy active timesteps, add a third lane or
cached reference feature bank:

```text
target lane:            target latent at timestep t
matched reference lane: reference latent at timestep t
anchor reference lane:  same reference at capped low-noise timestep t_anchor
```

Concatenate matched-timestep and anchor reference K/V, with token-type
embeddings and separate gates. Suggested first cap is the latest 20–30% noise
level among active inference timesteps. This is more expensive and must remain
defaults-off.

Do not feed a completely clean reference through a module normalized for a
very noisy target without measuring feature norms; the RMS/entropy telemetry
is mandatory for this arm.

## Priority 4 — targeted rank and output-capacity sweep

Only after a rank-32 v3 checkpoint demonstrates positive reference causality,
compare:

| Arm | Ref K/V rank | Ref output rank | Other trainables |
|---|---:|---:|---|
| Capacity control | 32 | 32 | fixed |
| Output-heavy | 32 | 64 | fixed |
| K/V + output | 64 | 64 | fixed |

Do not restore trainable target Q/K/V in this sweep. The old clean checkpoint
already showed that large target-path capacity can learn a generic adaptation
without solving reference specificity. Rank is valuable only after the model
is forced to spend it on a causal reference route.

## Priority 4 — bbox-relative positional bias

The current attention can match any target face query to any valid reference
face key. Add normalized bbox-relative coordinates `(u, v)` and a small learned
per-head bias:

```text
bias(q, k) = MLP([u_target(q), v_target(q), u_reference(k), v_reference(k)])
attention_logits += bias(q, k)
```

This can encourage eyes-to-eyes and mouth-to-mouth correspondences while still
allowing pose changes. Keep it low-rank/small, initialize its output to zero,
and apply it only after v3 causal use is proven. It addresses correspondence,
not branch collapse.

## Priority 5 — corrected target-query identity cross-attention

Branched CA is currently disabled and should remain disabled for the first v3
experiments. A later corrected CA branch may use:

```text
Q = target hidden states inside target face
K,V = gathered active PhotoMaker ID tokens only
output = native target CA + bounded face-local identity residual
```

It must have an actual ID-token key mask, a branch-local output projection,
separate gate/telemetry, and an exact trainable/checkpoint contract. Do not
reactivate the historical branched CA processor wholesale; previous CA-on
paths mix prompt and face-branch semantics too loosely for a controlled test.

## 4. Detailed integration checklist

### 4.1 Configuration and model constructor

Add defaults without changing historical values:

```yaml
# Shared defaults: old runs remain old.
model:
  ba_architecture_version: hard_replace_v1
  ba_merge_mode: hard_replace
  ba_mix_init: 0.50
  ba_mix_floor: 0.00
  ba_mix_max: 1.00
  ba_reference_base_output: zero_delta
  ba_reference_rms_match: false
  ba_reference_loss_mode: detached_diagnostic
  ba_photomaker_id_dropout_probability: 0.0
```

The new experiment config overrides:

```yaml
model:
  ba_architecture_version: anchored_mix_sa_v3
  ba_merge_mode: anchored_interpolate
  ba_mix_init: 0.50
  ba_mix_floor: 0.25
  ba_mix_max: 0.90
  ba_reference_base_output: frozen_native
  ba_reference_rms_match: true
  ba_reference_rms_clip_min: 0.50
  ba_reference_rms_clip_max: 2.00
  ba_reference_loss_mode: detached_diagnostic
```

Keep for all eligible arms:

```yaml
pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
disable_branched_ca: true
train_branched_ca_lora: false
```

### 4.2 Runtime processor installation

In
[`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py):

```diff
 if architecture_version == "residual_sa_v2":
     proc = ResidualBranchedSelfAttnProcessorV2(...)
+elif architecture_version == "anchored_mix_sa_v3":
+    proc = AnchoredMixBranchedSelfAttnProcessorV3(
+        hidden_size=hidden_size,
+        ref_kv_rank=pipeline.ba_ref_kv_rank,
+        output_rank=pipeline.ba_output_rank,
+        mix_init=pipeline.ba_mix_init,
+        mix_floor=pipeline.ba_mix_floor,
+        mix_max=pipeline.ba_mix_max,
+        reference_rms_match=pipeline.ba_reference_rms_match,
+        reference_rms_clip_min=pipeline.ba_reference_rms_clip_min,
+        reference_rms_clip_max=pipeline.ba_reference_rms_clip_max,
+        trainable_dtype=torch.float32,
+        require_denoise_progress=True,
+    )
 else:
     proc = BranchedAttnProcessor(...)
```

Update every architecture-version allowlist in
[`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py),
including expected trainables, optimizer role groups, strict ownership, and
error messages. Do not route v3 through a broad “non-v1 means v2” condition.

### 4.3 Checkpoint manifest

Add at least:

```json
{
  "ba_architecture_version": "anchored_mix_sa_v3",
  "routing": "target_q_reference_kv_true_key_mask",
  "merge": "native_plus_mask_alpha_reference_minus_native",
  "reference_output_base": "frozen_native_to_out",
  "mix_floor": 0.25,
  "mix_max": 0.90,
  "reference_rms_match": true,
  "reference_rms_clip": [0.5, 2.0],
  "reference_loss_mode": "detached_diagnostic"
}
```

Schema-v2 exact tensor-name/shape/dtype checks should remain. Bump the
processor code version or state format if needed; never silently interpret a
v2 `gate_logit` as a v3 bounded mix logit.

### 4.4 Validation propagation

The alternate-base validation constructor must receive every v3 constructor
invariant before processors are installed. After loading, assert:

```text
architecture version equals training
processor count and exact names equal manifest
mix floor/max and output-base mode equal manifest
all 414-or-new expected tensors loaded
pose_adapt_ratio == 0
ca_mixing_for_face == false
branched CA count == 0
```

Expose validation-only overrides for `alpha=0` and `alpha=1` without mutating
checkpoint parameters. Log the override prominently into Comet hyperparameters
and an audit JSON.

### 4.5 First-backward verification

The v3 smoke must assert nonzero finite gradients on update one for:

- every reference-K LoRA-B site;
- every reference-V LoRA-B site;
- every reference-output LoRA-B site;
- every mix logit;
- timestep and area mix terms if enabled.

LoRA-A gradients may still be zero on update one when LoRA-B starts at zero;
that is standard. The important repair is that K/V B and the gate no longer
wait for the output adapter to become nonzero.

## 5. Controlled implementation and experiment ladder

Do not launch another unchecked 40k architecture arm. Use this ladder:

### E0 — existing v2 2k fixed-checkpoint causal matrix

**Training:** none.  
**Checkpoint:** `weights-epoch1.pth` from immutable run
`4d6186f56cd24ba3a907fa35406c284e`.  
**Arms:** matched, shuffled spatial, zero spatial, branch off, forced reference.  
**Decision:** formally quantify how close v2 is to PM and whether any reference
dependence is visible at inference.

### E1 — v3 processor algebra and zero-step alpha sweep

**Training:** none.  
**Panel:** fixed subset first, then full 96 for selected alpha.  
**Alpha:** `0, .25, .50, .75, 1`.  
**Required checks:** alpha 0 native parity; alpha 1 matched/shuffle separation;
outside-mask parity; first-backward role gradients.  
**Decision:** choose the strongest structurally acceptable nonzero start.

### E2 — 250-step small-set causal overfit

**Architecture:** v3, rank 32, same 46 sites.  
**Objective:** standard reconstruction first; log matched/wrong diagnostics.  
**Data:** small multi-identity subset, never a one-identity-only overfit.  
**Decision:** reference contribution remains nonzero and correct references
beat shuffled references. If not, do not scale data or rank.

### E3 — 2k scheduled v3 merge-only arm

**Changed versus v2:** v3 direct reference output + anchored interpolation.  
**Unchanged:** dataset schedule, 46 sites, rank 32, optimizer precision,
training timestep policy, ordinary objective, validation contract.  
**Reference loss:** detached diagnostic only.  
**Decision:** isolate whether architecture alone creates causal use.

### E4 — 2k scheduled v3 + differentiable ranking arm

**Changed versus E3:** reference loss mode, weight, margin, and 50% paired
probability only.  
**Decision:** require a sustained positive correct-versus-wrong relative gap
and no face-structure regression.

### E5 — 4k/8k promotion arm

Promote only the better of E3/E4. Keep fixed-96 validation every 2k and run
the five-arm causal matrix at each promotion checkpoint. Stop if identity,
causal gap, or face quality regresses twice.

### E6 — layer expansion

Add `down_blocks.2` at rank 32. Keep everything else equal to E5.

### E7 — capacity sweep

Compare ref/output ranks 32/32, 32/64, and 64/64 only after E5 or E6 proves
causal reference use.

### E8 — stronger memory

Test identity-memory tokens, then low-noise persistent reference memory, one at
a time. Bbox-relative positional bias follows only if correspondence remains a
visible failure.

## 6. Promotion criteria

A model should not be called working BA merely because its weights, gradients,
or images changed. Require all of the following:

1. `alpha=0` reproduces the native PM anchor and `alpha=1` responds strongly to
   spatial-reference intervention.
2. Trained matched and shuffled/zero spatial arms are measurably different at
   the U-Net and image levels.
3. Wrong-reference face MSE is consistently higher than correct-reference face
   MSE on held-out batches.
4. Identity similarity improves relative to the candidate's own step zero and
   does not merely trade away prompt adherence or face quality.
5. Full-96 visuals show identity-specific facial changes, not only color,
   texture, expression, or generic polishing.
6. Reference contribution ratios are nontrivial but bounded; no layer group
   dominates through an amplitude explosion.
7. Exact trainable/checkpoint/validation manifests pass.
8. `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and explicit target-Q /
   reference-KV routing remain true.

## 7. What not to do next

- Do not wait for 40k and interpret any late pixel change as reference use.
- Do not increase v2 rank from 32 to 64 before fixing its zero-output and
  causal-objective problems.
- Do not set `pose_adapt_ratio>0`; that substitutes target features for
  reference K/V and is not an eligible fix.
- Do not broadly unfreeze target Q/K/V or the complete U-Net as the first
  response. That creates more shortcut capacity.
- Do not reactivate historical branched CA in the same experiment.
- Do not remove PhotoMaker identity conditioning from validation.
- Do not silently replace the fixed-96 protocol with a smaller or easier
  panel.
- Do not overwrite v1/v2 checkpoint semantics; add an explicit v3 version.

## Final recommendation

Treat residual-SA-v2 r6 as a **negative architectural control**: its software
path works, but its reference path is functionally too optional. The current
2k result does not support spending the remaining budget on the expectation
that it will escape PhotoMaker by itself.

Implement anchored interpolation BA-v3 first, retaining the current correct
key masking, FP32 trainables, exact ownership, checkpoint manifest, and
validation-native loading. Start with rank 32 and the same 46 sites, a
nonzero frozen reference output, `alpha_init=.50`, and `alpha_floor=.25`.
Instrument actual contribution and matched/wrong dependence. Then isolate the
differentiable reference-ranking objective in the next 2k arm.

This takes the useful part of the old structure—reference use cannot be
silently absent—and the useful part of v2—native PhotoMaker structure is
retained—while avoiding both permanent hard replacement and an exact
PhotoMaker no-op initialization.
