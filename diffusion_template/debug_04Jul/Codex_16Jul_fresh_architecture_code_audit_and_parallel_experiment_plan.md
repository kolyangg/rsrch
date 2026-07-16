# Fresh architecture and implementation audit for PhotoMaker branched attention

Date: 16 July 2026

Status: analysis and recommendations only. No training or inference code was changed.

## Executive conclusion

The current target-face residual family is the right base topology, but N29-N33
are limited by three issues that should be fixed before another long run:

1. **The BA trainable modules are explicitly converted to BF16.** In particular,
   every `face_residual_gate` is a BF16 scalar initialized to `1.0`. At the
   current `1e-4` learning rate, an AdamW update is far smaller than the BF16
   spacing around one, so the gate remains exactly `1.0`. This is a confirmed
   implementation defect, not merely a weak architectural choice.
2. **Inference amplifies the BA residual through classifier-free guidance.**
   Training sees a positive-prompt BA correction at scale one, but inference
   gives zero BA memory to the unconditional half and correct memory to the
   conditional half, then applies guidance scale 5. The BA correction is
   therefore approximately multiplied by five. This is a major train/inference
   mismatch and a plausible contributor to late-run over-intervention.
3. **The identity residual is installed in all 70 SDXL cross-attention sites.**
   Most of those sites operate at coarse 32x32 latent attention resolution.
   Coarse down, mid, and early-up injection can control face shape, expression,
   lighting, and color rather than only identity detail. N31's desaturation and
   expression drift are consistent with that authority.

The latest results do not support continuing N29, N31, N32, or N33 unchanged:

- N31 proves that BA can become influential, but its epsilon-ranking objective
  rewards non-identity shortcuts.
- N32 proves that richer memory can remain geometrically safe, but raw,
  unaligned CLIP face patches are still nuisance-rich and do not produce
  monotonic identity improvement.
- N33 proves that longer training cannot overcome N29's redundant two-token
  memory and weak supervision.

My recommended next step is therefore not a loss-weight sweep. It is a corrected
identity-residual architecture with:

- FP32 trainable BA parameters and bounded per-layer gates;
- explicit high-resolution layer routing, initially only the six
  `up_blocks.1` cross-attention sites;
- BA correction added once after PhotoMaker CFG rather than multiplied by CFG;
- correct, wrong, and null memory interventions on the same noisy latent;
- decoded, landmark-aligned, identity-causal supervision;
- label-aware semi-hard negatives;
- explicit preservation of target pose, expression, and chroma.

Two parallel runs should then isolate the remaining memory question:

- **N34:** corrected high-resolution residual using N29's two QFormer tokens;
- **N35:** the same corrected residual and objective, with a zero-initialized
  canonical face-part memory extension.

The 4-GPU machine should run N34; the 2-GPU machine should run N35. Before either
long run, use the existing N29/N31/N32 checkpoints for a short CFG, memory-swap,
and layer-ablation attribution matrix.

## Scope and evidence reviewed

The audit covered:

- the project handoff and prior visual analysis;
- N29-N33 training configurations and launch scripts;
- BA processor creation, training, loss composition, optimizer behavior,
  checkpoint saving/loading, and inference;
- the complete 96-image validation summaries for N31, N32, and N33;
- the close-up comparison and N31 desaturation progression;
- the active `CosmicLargeTrain` implementation;
- the local official PhotoMaker V2 identity-token implementation.

Primary project context:

- [Project handoff](Codex_16Jul_project_handoff_fresh_architecture_review.md)
- [N31/N32/N33 visual analysis](Codex_16Jul_N31_N32_N33_visual_architecture_analysis.md)
- [Close-up comparison](../full_validation_results/ba_n31_n32_n33_16Jul/N31_N32_N33_closeup_faces_vs_key.png)
- [N31 desaturation progression](../full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_face_progression.png)
- [Visual statistics](../full_validation_results/ba_n31_n32_n33_16Jul/visual_statistics.json)
- [Validation metrics/configuration](../comet_utils/comet_full_validation_N31_N32_N33.json)

## Reconstructed current architecture

### N29/N31/N33

For each target cross-attention layer:

1. Run ordinary PhotoMaker cross-attention and obtain `pm_out`.
2. Use target hidden states as queries.
3. Use the two frozen PhotoMaker QFormer identity tokens as BA keys and values.
4. Project the attended identity features through a zero-initialized low-rank
   output projection.
5. Multiply by one scalar `face_residual_gate`.
6. Mask the correction to the target face rectangle.
7. Add it to `pm_out`.

The implementation is in
[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L857).

N31 adds a second pass with a selected wrong memory and asks the correct-memory
epsilon prediction to have lower target-face MSE than the wrong-memory
prediction.

### N32

N32 replaces the two frozen QFormer memory tokens with eight trainable tokens:

1. The global InsightFace embedding creates eight queries.
2. Those queries attend the CLIP vision patches whose centers lie in the
   reference face bbox.
3. The resampled tokens are projected to the 2048-dimensional PhotoMaker
   conditioning space.
4. The same target-face residual path consumes those tokens in every
   cross-attention site.

The resampler is in
[`identity_memory.py`](../src/model/photomaker_branched/identity_memory.py#L100).

### Hard PhotoMaker preservation

Training and inference separately run a frozen PhotoMaker prediction, then
replace the BA prediction outside the target latent bbox:

```text
epsilon_merged = mask * epsilon_BA + (1 - mask) * epsilon_PM
```

See
[`hard_epsilon_merge`](../src/model/photomaker_branched/branched_runtime.py#L274).

This makes the epsilon prediction exactly PhotoMaker outside the bbox at that
particular UNet call. It does not make the complete denoising trajectory or
final pixels exactly PhotoMaker outside the bbox, because later UNet operations
receive a latent that was modified inside the face.

## Latest-result interpretation

| Run/checkpoint | Mean ID similarity | Architecture-level interpretation |
|---|---:|---|
| PhotoMaker | **0.4886** | Stable baseline and current identity score to beat |
| N29 2k / 6k / 10k | 0.4634 / 0.4692 / **0.4706** | Safe, but plateaus below PM |
| N31 2k / 6k / 10k / 12k | 0.4640 / 0.4611 / 0.4544 / **0.4480** | Dependence objective increasingly harms useful identity |
| N32 2k / 6k / 10k / 16k | 0.4394 / 0.4427 / **0.4482** / 0.4453 | Active and safe, but oscillatory rather than convergent |
| N33 14k / 20k / 24k / 26k | 0.4697 / 0.4709 / **0.4731** / 0.4663 | Longer N29 training does not remove the bottleneck |

All 96 generated faces were detected in every latest validation checkpoint, so
the main problem is not catastrophic face loss. It is semantic: the branch
changes the wrong face attributes or makes changes that are not reference-causal.

### N31 is the most informative failure

N31's face chroma falls from `84.35` at 2k to `48.42` at 12k, saturation falls
from `0.438` to `0.277`, and face MAE relative to PhotoMaker rises from `0.0668`
to `0.1114`. At the same time, its dependence loss approaches zero.

This combination is strong evidence that:

- the branch is active;
- the optimizer can distinguish correct and wrong memories;
- the current objective does not distinguish identity from color, contrast,
  expression, age, gender, lighting, or reference style;
- increasing branch strength without changing its semantics is unsafe.

### N32 is not frozen

N32 continues to change faces between 10k and 16k, and its target-ID K/V and
face-delta norms are nonzero. Its failure is therefore not simply that the
resampler never participates. The more likely problem is that its memory mixes
identity with pose, expression, crop, illumination, makeup, hair, and local
reference texture, while the loss gives no reason to isolate identity.

### N33 rules out “just train N29 longer”

N33's trainable norms continue growing, but its identity score and visual
identity do not improve monotonically. The two QFormer tokens already condition
the ordinary PhotoMaker prompt. Reusing the same tokens as a second attention
memory is mostly a second route for already-present information, not an
independent source of facial evidence.

## Ranked confirmed implementation and architecture problems

### P0. BA trainable parameters are stored and optimized in BF16

The processor creates its scalar gate in the dtype of the cloned UNet
projection:

- [`face_residual_gate`](../src/model/photomaker_branched/attn_processor_cleanest.py#L868)

The processors and N32 resampler are then explicitly converted to the UNet
weight dtype:

- [`prepare_for_training`](../src/model/photomaker_branched/lora2.py#L290)
- [`patch_unet_attention_processors`](../src/model/photomaker_branched/branched_runtime.py#L223)

All recent launch scripts set:

```text
model.weight_dtype=bf16
lr_for_lora=1e-4
```

An isolated AdamW reproduction in the project's `photomaker` environment gave:

```text
BF16 scalar initialized at 1.0 after 1000 unit-gradient steps, lr=1e-4: 1.0
next lower BF16 value: 0.99609375
next higher BF16 value: 1.0078125
AdamW exp_avg dtype: torch.bfloat16
AdamW exp_avg_sq dtype: torch.bfloat16
```

The gate would need an update of roughly `0.0039` just to cross half of the
nearest BF16 interval around one. A normal Adam update near `1e-4` is about 39
times too small and is rounded back to `1.0` every step. The telemetry showing
all gates exactly `1.0` is therefore expected.

Consequences:

- the supposed learned arbitration gate is effectively fixed;
- N32 LayerNorm scales initialized at one are similarly unable to make small
  updates;
- K/V adapters and resampler weights can move, but their updates and optimizer
  moments are quantized more heavily than intended;
- interpreting a constant gate as evidence that “the model prefers scale one”
  would be incorrect.

Recommendation:

- retain FP32 master parameters and FP32 optimizer states for all trainable BA
  processors, gates, and memory modules;
- let autocast use BF16 for matrix operations where appropriate;
- cast only outputs/activations back to the frozen UNet dtype;
- add update telemetry: gradient norm, parameter delta norm, and fraction of
  elements changed per optimizer step;
- use bounded FP32 per-layer gates rather than one unbounded scalar per site.

The extra memory for approximately 20 million BA parameters in FP32 is modest
relative to SDXL and is justified by the current failure.

### P0. Inference multiplies the BA correction by CFG

Training explicitly discards `do_cfg`:

- [`lora2.py`](../src/model/photomaker_branched/lora2.py#L454)

Inference supplies zero BA memory to the unconditional half and correct memory
to the conditional half:

- [`br_pipeline_helpers.py`](../src/pipelines/br_pipeline_helpers.py#L908)

The merged prediction is then guided:

- [`photomaker_branched_clean.py`](../src/pipelines/photomaker_branched_clean.py#L1047)

For the bias-free SDXL K/V projections, zero memory gives zero BA delta. If:

```text
epsilon_BA_cond = epsilon_PM_cond + delta
epsilon_BA_uncond = epsilon_PM_uncond
```

then current inference computes:

```text
epsilon_guided
  = epsilon_PM_uncond
    + cfg * (epsilon_PM_cond + delta - epsilon_PM_uncond)
  = epsilon_PM_guided + cfg * delta
```

At the current guidance scale of 5, the branch correction evaluated at inference
is about five times the correction seen during training.

The existing hard-PM pass already provides everything needed to correct this
without another UNet evaluation. Before CFG:

```text
delta_cond = epsilon_BA_cond - epsilon_PM_cond
epsilon_PM_guided =
    epsilon_PM_uncond + cfg * (epsilon_PM_cond - epsilon_PM_uncond)
epsilon_final = epsilon_PM_guided + ba_scale * delta_cond
```

Use `ba_scale=1` initially. This makes the BA authority explicit and independent
of text CFG while preserving exact PM epsilon outside the bbox.

Required tests:

- with a fixed checkpoint and seed, BA delta norm should be almost invariant
  across CFG 1, 3, and 5;
- `ba_scale=0` must exactly reproduce the PM-guided prediction;
- null memory must exactly reproduce PM at the merge point;
- no special rescaling should be silently inferred from CFG.

### P0. The target residual is trained at every SDXL cross-attention site

Cross-attention patching has no semantic layer filter:

- [`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py#L213)

For the standard SDXL UNet used here, the target-face CA processor is installed
at 70 sites:

| Region | Approx. attention grid at 1024px | Sites | Hidden size |
|---|---:|---:|---:|
| `down_blocks.1` | 64x64 | 4 | 640 |
| `down_blocks.2` | 32x32 | 20 | 1280 |
| `up_blocks.0` | 32x32 | 30 | 1280 |
| `up_blocks.1` | 64x64 | 6 | 640 |
| `mid_block` | 32x32 | 10 | 1280 |
| **Total** |  | **70** |  |

At rank 32, this creates approximately `19,824,710` trainable processor
parameters:

- 10 sites with 212,993 trainable parameters each;
- 60 sites with 294,913 each.

N32 adds another `2,375,168` resampler parameters.

Most trainable capacity is therefore attached to coarse 32x32 features. Those
features have authority over:

- face geometry and placement;
- mouth/eye expression;
- broad skin shading and chroma;
- illumination and low-frequency appearance;
- compatibility with hair, hands, goggles, and other occlusions.

This is too broad for an “identity detail only” residual.

Recommendation:

- add an explicit name/block allowlist shared by patching, trainable selection,
  checkpoint metadata, restoration, and validation;
- start with only the six `up_blocks.1.*.attn2` sites;
- consider selected late `up_blocks.0` sites only after an attribution ablation
  shows that they add identity without geometry/chroma drift;
- do not train down-block or mid-block identity residuals in the first corrected
  run.

### P0. `top_k` is an ordering prefix, not a meaningful resolution selector

[`select_branched_processor_names`](../src/model/photomaker_branched/branched_runtime.py#L15)
returns the first fraction of raw processor names:

```python
return candidate_names[:keep_count]
```

The relevant order starts with down blocks, then up blocks, then the mid block.
Consequently, lowering `ba_train_top_k` preferentially retains early/down
processors and can exclude the desired high-resolution up processors.

In addition, `ba_patch_top_k` currently applies only to self-attention. It does
not restrict the target-face cross-attention processors at all.

Recommendation:

- deprecate fractional `top_k` for architecture experiments;
- use explicit semantic selectors such as:

```text
ba_ca_layer_allowlist:
  - up_blocks.1
```

- log the exact selected names, site count, spatial grid, hidden size, and
  trainable parameter count at startup;
- save that resolved list in every checkpoint.

### P0. Target bboxes fail open to a full-image BA mask

[`_bbox_to_mask`](../src/model/photomaker_branched/lora2.py#L872) fills the mask
with ones when:

- the bbox is missing;
- fewer than four values are present;
- the bbox is degenerate;
- the scaled bbox becomes empty.

For the target-face residual, this converts malformed supervision into
full-image BA authority. It violates the central safety invariant.

The diffusion losses have a similar fallback:

- [`_masked_face_mse`](../src/loss/diffusion_loss.py#L7) falls back to full-image
  MSE if no valid bbox remains;
- [`identity_dependence_ranking_loss`](../src/loss/diffusion_loss.py#L35) uses
  the full tensor for a degenerate bbox.

The losses also use integer truncation of `bbox / 8`, while the branch mask uses
floor/ceil area-preserving coverage. The optimized loss region can therefore
differ from the region the architecture is allowed to modify.

Recommendation:

- fail closed for target masks: invalid bbox means zero BA authority or a hard
  data error, never a full-image mask;
- use one shared bbox-to-mask implementation for routing, PM merge, diffusion
  losses, decoded losses, and diagnostics;
- clip coordinates and use documented floor/ceil coverage consistently;
- emit sample identifiers and bbox values for every rejected sample.

## High-priority objective and representation problems

### P1. N31's dependence loss is not identity-causal

[`identity_dependence_ranking_loss`](../src/loss/diffusion_loss.py#L35) compares
epsilon MSE in the face rectangle. It does not decode an image and does not
measure identity.

[`select_wrong_identity_features`](../src/model/photomaker_branched/lora2_helpers.py#L493)
selects the least cosine-similar flattened memory. It has no person labels,
identity-cluster check, or same-domain/semi-hard criterion.

This objective can be satisfied by making the correct and wrong branches differ
in any predictable reference-correlated attribute:

- skin tone or saturation;
- lighting and contrast;
- expression;
- age or gender;
- crop and pose;
- makeup, hair, or photography style.

It can also satisfy the ranking by degrading only the wrong branch. That does
not prove the correct branch transfers identity.

N31's decreasing dependence loss alongside grayscale faces is a direct empirical
demonstration of this shortcut.

Recommendation:

- keep the ordinary PM reference and prompt fixed;
- change only the BA memory among correct, null, and wrong interventions;
- compare decoded, aligned identity embeddings;
- require correct memory to improve similarity to the correct identity relative
  to null/PM;
- require wrong memory to move the face toward the selected wrong identity,
  rather than merely making it worse;
- preserve target geometry, expression, and chroma in all interventions;
- choose a confirmed different-person, same-domain, semi-hard wrong identity.

### P1. The existing decoded ID loss is too confounded

The current loss:

- reconstructs one predicted `x0` from epsilon;
- decodes it with the VAE;
- takes a hard rectangular crop;
- bilinearly resizes it to 160x160;
- compares one frozen VGGFace2 FaceNet embedding.

See:

- [`_compute_id_loss`](../src/model/photomaker_branched/lora2.py#L664)
- [`IdentityLoss`](../src/loss/id_loss.py#L21)

Limitations:

- no landmark alignment;
- no inner-face mask;
- one recognizer can reward pose, expression, crop, color, or adversarial
  texture;
- `x0` reconstructed at moderately noisy timesteps can have extreme values that
  are later clamped by recognizer preprocessing;
- no correct-vs-null causal comparison;
- no wrong-identity directional test;
- no geometry, expression, or chroma preservation term.

Recommendation:

- apply a fixed differentiable affine crop derived from ground-truth target
  landmarks to all generated branches;
- use an aligned inner-face region for identity, while retaining the hard bbox
  only as the allowed write envelope;
- restrict decoded supervision to lower-noise timesteps initially;
- compare correct, null, and wrong interventions on the same latent;
- train with one recognizer and validate with at least one independent identity
  metric to detect recognizer exploitation;
- add explicit low-frequency chroma and landmark/expression preservation.

### P1. N32 memory is richer but still nuisance-rich

N32's global InsightFace embedding creates queries that attend an unaligned
subset of CLIP reference patches. Those patches can encode much more than
identity:

- head pose;
- expression;
- illumination;
- makeup and skin texture;
- hair and nearby background;
- crop scale and camera style.

The current CLIP resize/center-crop bbox mapping was checked against the
installed processor behavior and is broadly consistent. I did not find a
high-confidence coordinate-transform bug there. Patch-center inclusion and
rounding still deserve an overlay unit test, but they are not the best
explanation for N32's behavior.

The architectural problem is semantic, not simply geometric: the resampler is
free to encode any face-patch information that helps denoising.

Recommendation:

- canonicalize the reference with landmarks before extracting part features;
- use frozen recognition features rather than raw CLIP appearance patches for
  the new identity-detail tokens;
- construct ordered tokens for global identity, left eye, right eye, nose,
  mouth, and contour/cheeks;
- introduce these as a zero-initialized residual extension of the stable
  QFormer route;
- use the same correct/null/wrong causal objective as N34 so memory is the only
  major difference.

### P1. N29's BA memory is redundant with ordinary PhotoMaker conditioning

PhotoMaker already fuses the two QFormer tokens into the prompt at the trigger
token. N29 then reuses those same frozen tokens as a second CA memory.

This has two limitations:

1. It does not supply independent facial evidence beyond what PhotoMaker already
   sees.
2. With two memory tokens, each attention head can only mix two global value
   vectors. It cannot establish robust eye/nose/mouth-specific correspondence.

N33's continued parameter movement without useful identity gain is consistent
with an information and objective bottleneck, not undertraining.

N34 should retain these tokens as the cleanest control. N35 should test whether
canonical part tokens add useful information after routing, precision, CFG, and
supervision are corrected.

## Additional implementation issues

### P1. Inactive optimizer protection omits the N32 resampler

[`attach_inactive_branched_params`](../src/model/photomaker_branched/lora2_helpers.py#L468)
attaches both processor and resampler parameters to inactive forwards with
exactly zero gradients.

When `ba_skip_inactive_optimizer_decay=true`, the trainer clears gradients only
for processor parameters:

- [`sdxl_trainers.py`](../src/trainer/sdxl_trainers.py#L423)

The resampler's zero gradients remain, so AdamW can still apply weight decay and
update optimizer moments during the approximately 30% of schedule stages where
BA is intentionally inactive.

Recommendation:

- define explicit optimizer groups for all BA components;
- skip every BA group consistently when no accumulated microbatch used BA;
- include the resampler and future memory modules;
- unit-test that inactive optimizer steps leave every BA parameter bitwise
  unchanged.

Distributed correctness should not rely on each rank independently sampling the
same regime. Broadcast the regime/timestep bucket from rank zero or all-reduce
the BA-active flag before deciding whether an optimizer group steps.

### P1. Checkpoint restoration is incomplete and permissive

[`_apply_saved_ba_architecture`](../infer.py#L27) restores several BA switches
but omits architecture-relevant values including:

- `ba_identity_patch_padding`;
- `ba_identity_resampler_hidden_dim`;
- any future semantic layer allowlist;
- any future gate type or CFG-combination mode.

[`infer.py`](../infer.py#L151) also swallows every exception from
`prepare_for_training()`. A missing adapter or failed processor installation can
therefore be hidden until later.

Processor state loading uses `strict=False`:

- [`lora2.py`](../src/model/photomaker_branched/lora2.py#L433)

Current N32 defaults happen to match its saved architecture, so this is not a
proven explanation for the existing validation. It is nevertheless a serious
future reproducibility risk.

Recommendation:

- save one resolved architecture manifest in the checkpoint;
- restore every shape and behavior key before module construction;
- require exact processor-name and tensor-key matches for BA checkpoints;
- fail loudly on processor installation or architecture mismatch;
- print a concise checkpoint compatibility report.

### P1. Active dataset flags are silently ignored

The active `CosmicLargeTrain` constructor deletes these arguments:

- `require_nested_identity_subdir`;
- `upscale_to_1024`;
- `const_ref`;
- `crop_ref`;
- `ref_similar`;
- `origtarget_genref`;
- `train_on_separate_image`;
- `same_id_ref_map_json_pth`.

See [`cosmic.py`](../src/datasets/cosmic.py#L922).

This means a config can claim a data ablation that the active dataset never
performs. N29-N33 all use the same active behavior, so it does not explain their
differences. It can, however, invalidate future experiments silently.

The currently sampled references are synthetic face variants tied to the same
target entry and can differ in pose, expression, lighting, and background.
There is no explicit stable person ID in the batch used by N31's negative
selection.

Recommendation:

- remove unsupported options or implement them and log the resolved behavior;
- add an explicit `identity_id` or offline identity-cluster ID to each sample;
- ensure wrong negatives have a different identity ID;
- record target and reference paths plus IDs in attribution manifests;
- if multiple target poses per identity are available, use them to separate
  identity from target-entry-specific attributes.

### P2. Failed reference-face detection becomes a zero identity embedding

If InsightFace finds no reference face during training, the code substitutes a
zero 512-D embedding:

- [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L224)

The sample is not rejected or tagged. PhotoMaker and BA memory construction then
continue with a semantically invalid identity signal.

Recommendation:

- validate face detection offline where possible;
- attach a detection-validity/confidence field;
- skip invalid identity supervision or replace the sample deterministically;
- log invalid rates by dataset source.

### P2. Hard epsilon merge is not an exact PhotoMaker trajectory

The merge guarantees PM epsilon outside the bbox for the current latent, but the
current latent was already changed inside the bbox. Later convolutions and
global attention can propagate that change outside the face.

This is a lower priority because N29-N33 already preserve the scene well.
If stricter preservation becomes necessary:

- maintain a parallel PM latent trajectory;
- after each scheduler step, copy the PM latent outside the target mask;
- consider a narrow transition band to avoid boundary seams;
- report both epsilon-level and final-pixel outside-mask differences.

## Items checked that are not primary bugs

### Training and inference timestep schedules are close

At 50 DDIM steps the inference timesteps are approximately:

```text
981, 961, ..., 1
```

PhotoMaker begins at step 10, around timestep 781, and BA begins at step 15,
around timestep 681.

Training maps:

- text-only to approximately the highest 20% of timesteps;
- PhotoMaker-only to the next approximately 10%;
- PM+BA to the lowest approximately 70%.

This corresponds roughly to PM at `t <= 799` and BA at `t <= 699`. The mismatch
is small and is not a convincing explanation for N31/N32/N33.

The shared scalar timestep per batch is less diverse than per-sample timesteps,
but it preserves one regime per batch. A future implementation can sample the
regime synchronously and then sample per-example timesteps within that regime.

### Current target transforms preserve the active 1024px bbox coordinates

The active dataset loads/crops the target to 1024x1024 before the configured
tensor transform. I did not find a current random target flip or resize that
silently invalidates the target bbox in the N29-N33 path.

### DDP does not multiply the gradient by world size

N31's four ranks average gradients; they do not make one optimizer update four
times larger. N31 does, however, see four times as many samples per optimizer
step as a one-GPU local-batch-2 run:

- N29/N32: effective global batch about 2;
- N31: effective global batch about 8.

Therefore comparisons by raw optimizer step are not comparisons by samples
seen. N31's shortcut was exposed to many more examples and became visible
faster. Future runs should use matched effective global batch and report both
optimizer steps and samples.

## Required pre-run attribution matrix

These tests should be performed on existing checkpoints before spending another
10k training steps. They are inference interventions, not new training runs.

Use fixed seeds, prompts, target bboxes, and ordinary PhotoMaker reference
conditioning. Change only the specified BA intervention.

### Memory causality

For N29 10k, N31 12k, N32 10k/16k, and N33 24k:

1. PhotoMaker / BA scale 0.
2. Correct BA memory.
3. Null BA memory.
4. Swapped BA memory while the PhotoMaker reference remains correct.

Measure:

- similarity to the correct identity;
- similarity to the swapped identity;
- aligned face difference from PM;
- landmarks/expression;
- face chroma and saturation;
- outside-mask difference.

Interpretation:

- if correct and swapped memory produce nearly the same edit, BA is learning a
  generic face operator;
- if swapped memory only reduces quality, the branch is dependence-sensitive
  but not identity-directional;
- if swapped memory moves the generated embedding toward the swapped identity
  while preserving target geometry, BA carries causal identity information.

### CFG sweep

For each memory intervention, evaluate guidance scales 1, 3, and 5.

Under the current implementation, the face delta should grow approximately with
CFG. After the proposed correction, PM prompt strength may change with CFG but
the isolated BA delta should remain controlled by `ba_scale`.

### Layer-site ablation

On N29 and N32, disable BA output by region without retraining:

- down blocks only;
- mid block only;
- `up_blocks.0` only;
- `up_blocks.1` only;
- all sites.

The key comparison is:

```text
up_blocks.1 versus all 70 sites
```

Expected useful signature:

- `up_blocks.1` retains local identity detail;
- disabling down/mid reduces expression, geometry, and chroma drift;
- outside-mask preservation remains unchanged.

If `up_blocks.1` carries no measurable identity signal, test a small explicitly
named late subset of `up_blocks.0`; do not revert directly to all sites.

## Proposed corrected architecture shared by N34 and N35

### 1. PhotoMaker remains the global generator

Keep:

- frozen RealVis/SDXL base;
- frozen PhotoMaker identity encoder;
- ordinary PhotoMaker prompt-token fusion;
- target-coordinate queries;
- hard target bbox as the maximum write envelope;
- PM preservation outside the bbox.

### 2. Add the BA correction after CFG exactly once

For the existing hard-PM two-prediction structure:

```text
pm_u, pm_c = PhotoMaker unconditional and conditional predictions
ba_c = conditional prediction with BA
delta_c = hard_mask * (ba_c - pm_c)

pm_cfg = pm_u + cfg * (pm_c - pm_u)
final = pm_cfg + ba_scale * delta_c
```

Do not route `delta_c` through text CFG.

### 3. Restrict trainable routing to high-resolution up-block CA

Initial allowlist:

```text
up_blocks.1.*.attentions.*.transformer_blocks.*.attn2.processor
```

Expected site count: 6.

Keep all down, mid, and `up_blocks.0` BA residuals absent, not merely frozen.
This reduces both semantic authority and trainable capacity dramatically.

### 4. Use bounded FP32 per-layer gates

For each selected site:

- keep the gate parameter in FP32;
- use a bounded parameterization, for example
  `max_scale * sigmoid(logit)` or `max_scale * tanh(logit)`;
- retain a zero-initialized output projection so step zero is exactly PM;
- log effective scale per layer and timestep bucket.

A single per-layer scalar is sufficient for the first corrected run. Per-head
or query-dependent gating should be a later ablation, not combined immediately
with every other change.

### 5. Separate write region from identity-supervision region

- Hard target bbox: maximum region BA may modify.
- Landmark-aligned inner face: region used for recognition identity loss.
- Target/PM face features: source for geometry, expression, and chroma
  preservation.

The face bbox can include hair, goggles, hands, and background. Those are valid
context for rendering but should not define the identity objective.

### 6. Correct/null/wrong causal branches

Use the same:

- target image;
- noisy latent;
- timestep;
- text prompt;
- PhotoMaker reference and prompt tokens;
- target bbox.

Change only BA memory:

```text
correct memory: memory from correct reference
null memory: zero/disabled BA memory
wrong memory: confirmed different identity
```

The null branch is the causal baseline. It should be exactly PM at the BA merge
point.

### 7. Decoded identity-direction objective

At low-noise timesteps:

1. Reconstruct and decode `x0` for correct, null, and selected wrong branches.
2. Apply the same target-landmark-derived differentiable alignment to each.
3. Compute normalized frozen recognition embeddings.
4. Optimize directional margins:

```text
correct output is closer to correct ID than null output
correct output is closer to correct ID than wrong ID
wrong output moves toward wrong ID relative to null
```

The third condition prevents the model from satisfying the objective by merely
damaging the wrong branch.

Keep ordinary diffusion reconstruction as the main image prior. Ramp the causal
loss in gradually and, if compute is limiting, evaluate the decoded objective
on a deterministic fraction of steps rather than weakening its semantics.

### 8. Preserve non-identity face attributes

Add lightweight explicit constraints:

- landmark/geometry consistency with target or PM;
- expression-feature consistency with target or PM;
- low-frequency Lab chroma consistency;
- optional low-frequency luminance consistency;
- outside-mask PM consistency diagnostic.

These losses should use the correct branch and should not force high-frequency
identity details back to PM.

### 9. Label-aware semi-hard negatives

Do not select the least similar item in the global batch.

Prefer:

- a different `identity_id`;
- same broad data domain and image type;
- moderate recognition similarity, excluding likely duplicates and extremely
  dissimilar easy negatives;
- deterministic selection given seed and sample IDs.

If reliable labels do not exist, create offline recognition clusters and audit
cluster collisions before training.

## Experiment N34: identity-causal high-resolution QFormer residual

### Question

Can the existing two PhotoMaker QFormer tokens improve identity once the branch
precision, CFG semantics, layer routing, and causal supervision are corrected?

### Architecture

- Memory: N29's two distinct frozen QFormer tokens.
- BA sites: only the six `up_blocks.1` cross-attention processors.
- Trainable BA parameters: FP32 K/V adapters, zero-init output adapters, bounded
  FP32 per-layer gates.
- Composition: BA delta added once after PM CFG.
- Masking: fail-closed target bbox; aligned inner-face identity crop.
- Objective: diffusion reconstruction plus correct/null/wrong decoded identity
  direction and geometry/expression/chroma preservation.
- Initialization: fresh BA residual from zero over the released/frozen PM path,
  not a continuation of N29/N31 weights.

Starting fresh avoids inheriting weights trained across all 70 layers and
implicitly calibrated for CFG amplification.

### Machine allocation

Use the 4-GPU machine.

Suggested effective batch:

- local batch 1 if three branch passes plus decode are memory-heavy;
- gradient accumulation chosen to reach a fixed effective global batch, ideally
  8;
- synchronize the regime across ranks;
- report examples seen as well as optimizer steps.

The correct, null, and wrong passes can run sequentially on each rank while
sharing frozen features and the PM prediction.

### Checkpoints

Validate at:

```text
step 0, 1k, 3k, 6k, 10k
```

The 1k and 3k checkpoints are essential. A semantic architecture should show
directional memory causality before a long run.

### Expected success signature

- Step zero is exactly PM.
- Correct memory improves aligned identity relative to null/PM.
- Swapped memory moves the result toward the swapped identity.
- Pose and expression remain target-consistent.
- Face chroma remains within approximately 10% of PM/target aggregate values.
- No N31-like monotonic desaturation.
- `up_blocks.1` gates and parameter deltas move measurably in FP32.
- Outside-face differences remain no worse than the N29 safe family.

### Failure signature and stopping rule

Stop by 3k-6k if any of the following persists across two validations:

- correct-vs-null identity direction is absent;
- swapped memory only worsens quality rather than moving identity;
- mean ID falls materially while branch norm grows;
- face chroma drops by more than about 15%;
- landmark or expression deviation grows systematically;
- gates saturate at their bounds;
- the branch becomes a generic sharpening/smoothing/color operator.

If N34 fails while the attribution matrix shows current N29 has some
memory-causal identity signal, test a small named late-`up_blocks.0` extension
before changing memory.

## Experiment N35: canonical face-part identity memory extension

### Question

After the same routing, precision, CFG, and objective corrections, does
canonical part-level identity evidence outperform the two-token QFormer memory
without reintroducing reference pose and lighting leakage?

### Architecture

Use the full N34 corrected route and objective. Change only identity memory.

Recommended memory:

1. Detect reference landmarks and warp the face to a canonical crop.
2. Extract frozen multi-scale features from a face-recognition backbone.
3. Pool ordered regions for:
   - global identity;
   - left eye;
   - right eye;
   - nose;
   - mouth;
   - left/right cheek or contour.
4. Project these tokens into the 2048-D BA memory space.
5. Add them through a separate zero-initialized part-memory residual path while
   retaining the two stable QFormer tokens.

Using a separate zero-initialized output for part memory ensures that adding
tokens does not alter softmax normalization at step zero. N35 can reproduce the
N34/PM starting behavior exactly, then learn whether the extra evidence is
useful.

Avoid raw unaligned CLIP patches in this experiment. The point is to test
identity-specific canonical information, not merely more appearance capacity.

### Machine allocation

Use the 2-GPU machine.

Suggested effective batch:

- local batch 1;
- gradient accumulation sufficient to match N34's effective global batch;
- if the wrong branch plus decode is too expensive, compute correct/null every
  applicable step and the symmetric wrong-direction term on a deterministic
  cadence, while keeping samples-seen accounting matched.

### Checkpoints and stop rules

Use the same `0, 1k, 3k, 6k, 10k` validation schedule and the same stop rules as
N34.

### Interpretation against N34

| Result | Interpretation |
|---|---|
| N34 succeeds; N35 adds no gain | Correct routing/objective was the main issue; two QFormer tokens are sufficient |
| N34 succeeds; N35 improves swap direction and ID | Canonical part memory adds useful identity information |
| N34 fails; N35 succeeds | N29's two-token representation was the main remaining bottleneck |
| Both fail with no swap causality | Dataset/recognizer/objective still lacks usable identity supervision |
| Both improve ID but drift in expression/chroma | Preservation and/or selected layer authority remains too weak |

## Optional short architecture control

If the 4-GPU machine can be split without making N34 impractically slow, run a
short 2-GPU control for at most 3k steps:

- same FP32, post-CFG, causal objective, and QFormer memory as N34;
- all 70 cross-attention sites enabled.

This is not a candidate for a long run. It is a direct control for the layer
routing hypothesis. Stop immediately if the all-layer run develops larger
chroma/expression changes without stronger swap-direction identity.

If resource pressure is high, the existing-checkpoint layer-ablation matrix is
preferable to this extra training run.

## Evaluation protocol

### Fixed primary validation

Keep the current 96 images, prompts, seeds, and target bboxes. Add a deterministic
wrong-reference mapping so every checkpoint receives the same intervention.

For each sample save:

- PM/null output;
- correct-memory output;
- wrong-memory output;
- reference images for correct and wrong identities;
- aligned face crops;
- difference maps inside and outside the bbox.

### Primary metrics

Report distributions and paired deltas, not only means:

- correct-ID similarity: correct versus null;
- wrong-ID direction: wrong output versus null;
- percentage of samples with positive correct-ID delta;
- percentage with positive wrong-ID directional delta;
- landmark displacement normalized by interocular distance;
- expression-feature distance;
- face Lab chroma and saturation;
- face LPIPS/MAE versus PM;
- outside-mask LPIPS/MAE versus PM;
- face-detection rate.

Validate identity with an encoder different from the one used in the training
loss.

### Minimum useful success criteria

A checkpoint is promising only if:

- median correct-vs-null identity delta is positive;
- a clear majority of samples improve rather than a few outliers driving the
  mean;
- swapped memory produces directional movement toward the swapped identity;
- aggregate chroma remains within about 10% of PM;
- normalized landmark/expression drift remains small;
- visual review confirms identity changes rather than color or expression
  shortcuts.

An aggregate ID increase without swap causality is not sufficient evidence.

## Preflight tests required before launching

1. **Zero-init equivalence:** fresh N34/N35 checkpoint exactly matches PM at the
   merged epsilon prediction.
2. **CFG independence:** isolated BA delta has stable magnitude at CFG 1, 3, 5.
3. **Allowlist:** exactly six N34 CA processors are installed and trainable.
4. **Precision:** every trainable BA parameter and Adam state is FP32; frozen
   UNet/PM remains BF16.
5. **Update test:** a `1e-4` optimizer step changes the gate and representative
   LayerNorm weights.
6. **Mask safety:** missing/invalid target bbox raises or produces zero BA mask.
7. **Mask consistency:** routing and every face loss use the same covered latent
   cells.
8. **Null identity:** null memory produces zero delta at every selected layer.
9. **Inactive step:** every BA parameter remains bitwise unchanged when its
   optimizer groups are skipped.
10. **DDP regime:** all ranks use the same branch-active regime.
11. **Checkpoint round trip:** architecture manifest, allowlist, memory shape,
    gates, and optimizer-relevant module structure restore exactly.
12. **Reference validity:** missing face detections are rejected or explicitly
    skipped, never silently represented as a valid zero identity.

## Recommended implementation order for a future coding session

No code was changed in this audit. The safest future implementation order is:

1. Add architecture manifest and semantic layer allowlist.
2. Keep BA trainables and optimizer states in FP32.
3. Implement post-CFG BA composition and its exact-equivalence tests.
4. Make bbox handling fail closed and unify masks.
5. Correct inactive optimizer-group skipping and synchronized regime sampling.
6. Add attribution controls: BA scale, null memory, swapped memory, and
   per-region layer disable.
7. Implement aligned correct/null/wrong identity-direction loss and preservation
   terms.
8. Add identity IDs or offline clusters and semi-hard negative selection.
9. Launch N34.
10. Implement canonical part memory as a zero-init extension and launch N35.

## Files likely to require changes later

Core residual and precision:

- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2_helpers.py`
- `src/model/photomaker_branched/lora2.py`

CFG composition:

- `src/pipelines/br_pipeline_helpers.py`
- `src/pipelines/photomaker_branched_clean.py`

Losses and supervision:

- `src/loss/diffusion_loss.py`
- `src/loss/id_loss.py`
- `src/trainer/sdxl_trainers.py`

Memory and data:

- `src/model/photomaker_branched/identity_memory.py`
- `src/datasets/cosmic.py`

Checkpoint/inference reproducibility:

- `infer.py`
- future N34/N35 model and inference configs
- future N34/N35 launch scripts

## Final recommendation

Do not continue N31: it is successfully optimizing a harmful shortcut.

Do not continue N32 unchanged: the branch is active, but additional raw
appearance capacity has not become identity-specific.

Do not continue N33 unchanged: the long-run evidence already shows saturation.

First confirm the CFG amplification and layer attribution on existing
checkpoints. Then run N34 and N35 in parallel with matched effective batch,
samples-seen accounting, early 1k/3k validation, and hard stop rules.

The decisive criterion is not merely “different from PhotoMaker” or a small
mean-ID increase. It is:

> Holding the PhotoMaker path, target latent, prompt, pose, and expression fixed,
> changing only BA memory should move the generated face toward the identity
> encoded by that memory, while preserving chroma, geometry, and everything
> outside the target face.

