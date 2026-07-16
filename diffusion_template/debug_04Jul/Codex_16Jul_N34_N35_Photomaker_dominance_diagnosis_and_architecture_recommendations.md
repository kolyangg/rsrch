# N34/N35 PhotoMaker dominance: diagnosis and architecture recommendations

Date: 16 July 2026

Scope: investigation and recommendations only. No training or model code was
changed.

## Executive conclusion

N34 and N35 can eventually produce faces that differ from ordinary PhotoMaker.
The BA path is not frozen, and its output is not mathematically capped to a tiny
correction: the learned K/V and output projections can continue growing.

However, the current architecture is much more PhotoMaker-dominant than the
older N29/N31/N32 family. This is mostly intentional, but several conservative
choices multiply together:

1. PhotoMaker is the complete baseline generator and remains active throughout
   the face trajectory.
2. BA starts from exactly zero.
3. BA is installed at only six high-resolution cross-attention sites, versus
   approximately 70 eligible sites in the old runs.
4. Every BA residual is multiplied by a gate initialized at `0.25`.
5. The BA correction is added once after CFG with scale `1`, while the old
   conditional BA difference was effectively amplified by guidance scale `5`.
6. BA begins only at inference step 15, after five PhotoMaker-only steps.
7. The decoded causal objective asks BA to improve over an already
   correct-identity PhotoMaker baseline by a small cosine margin of `0.02`.
8. N34/N35 first full validation is at 1,000 optimizer steps, while N31/N32
   first full validation was at 2,000 steps and N33 began from an already
   trained 10k N29 checkpoint.

The first PM-like validations therefore do not indicate that microbatching has
silently disabled training. They indicate that the new route has much less
early authority.

The important distinction is:

- **Global/image dominance:** PhotoMaker should retain this, to preserve scene,
  pose, composition, expression, and general rendering quality.
- **Identity causality inside the face:** BA should dominate this. Ordinary
  correct-reference images do not measure it well because PhotoMaker and BA are
  both given the same identity.

The decisive test is not “does the correct-reference image look dramatically
different from PhotoMaker?” It is “with the PhotoMaker reference held fixed,
does changing only BA memory from null to correct to wrong move the generated
identity in the corresponding direction?”

## What the implementation actually computes

At each selected cross-attention site, the target-face processor computes:

```text
site_output = ordinary_PM_attention_output
            + hard_face_mask * gate * learned_identity_delta
```

See
[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L961).

At inference with CFG, N34/N35 then compute:

```text
pm_cfg = pm_uncond + guidance * (pm_cond - pm_uncond)
final  = pm_cfg + ba_residual_scale * (ba_cond - pm_cond)
```

inside the hard face mask. See
[`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py#L328).

This is a sound composition rule. It prevents BA from being accidentally
multiplied by text guidance, which was one source of excessive old-run
authority and artifacts. But it also means that replacing the old route with
post-CFG scale `1` removes an approximately fivefold amplification at
validation guidance `5`.

The new architecture was explicitly designed with “PhotoMaker remains the
global generator” as its first principle. See the original proposal in
[`Codex_16Jul_fresh_architecture_code_audit_and_parallel_experiment_plan.md`](Codex_16Jul_fresh_architecture_code_audit_and_parallel_experiment_plan.md#L775).
The design goal was a safe, causal identity correction, not a BA face
replacement.

## Evidence that N34 is learning

The downloaded N34 epoch-1 checkpoint represents 1,000 optimizer windows and
contains:

- exactly six `up_blocks.1` BA processors;
- all 42 intended trainable processor tensors in the optimizer;
- Adam state for every one of those tensors;
- 915 real Adam updates for every tensor, consistent with accumulation and the
  approximately 70% BA-active timestep regime;
- output-adapter-up norm `1.3837`;
- target identity K/V LoRA-B norm `1.4990`;
- effective gates between `0.2512` and `0.2531`.

The branch is therefore neither disconnected nor frozen.

At the same logged 1k point:

| Run | Face output-adapter norm | Target-ID K/V norm | Gate |
|---|---:|---:|---:|
| N29 | 4.788 | 6.096 | 1.0 |
| N31 | 4.837 | 6.409 | 1.0 |
| N32 | 4.203 | 4.162 | 1.0 |
| N34 | 1.384 | 1.499 | about 0.252 |

The aggregate N34 face-adapter norm is smaller largely because it contains six
sites rather than approximately 70. Dividing by the square root of site count
gives:

```text
N34 per-site proxy: 1.3837 / sqrt(6)  = 0.565
N29 per-site proxy: 4.7882 / sqrt(70) = 0.572
```

Thus N34 is learning almost exactly as much output-adapter magnitude per active
site as N29 did. The visible weakness comes from total route authority, not from
failure to update each site.

A rough, non-literal authority proxy at 1k is:

```text
N34 / N29
≈ (1.3837 * 0.252 * post_CFG_scale_1)
  / (4.7882 * 1.0 * legacy_CFG_scale_5)
≈ 0.0146
≈ 1 / 69
```

This is not an exact epsilon-output ratio, but it explains the visual result:
the same-order per-site learning is being expressed through far fewer sites, a
quarter-scale gate, and no CFG amplification.

## Why the gate initialization is more conservative than necessary

Each `face_delta_out.up.weight` is initialized to exactly zero:

[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L17).

Therefore the branch output is exactly zero at step 0 regardless of whether the
gate starts at `0.25`, `1.0`, or another finite value. Initializing the gate at
`0.25` is not required for PhotoMaker equivalence.

It does have two suppressive effects:

1. The initial gradient into `face_delta_out.up.weight` is multiplied by
   `0.25`, reducing it fourfold relative to a gate of `1`.
2. After the output adapter becomes nonzero, the forward residual and upstream
   gradients into K/V and the N35 resampler remain multiplied by the small
   gate.

The N34 checkpoint gates have moved only from `0.2500` to approximately
`0.252`. The branch can still grow through its unbounded output and K/V
projections, but the gate is not currently providing rapid authority
calibration.

For a future run, exact PM initialization should come from the zero output
adapter, while the gate can start at a useful learning scale such as `1.0`.

## Why the causal objective permits only a subtle correction

The null branch is not an identity-free generator. It is ordinary PhotoMaker
with the correct PhotoMaker reference and prompt identity tokens. BA is asked to
improve over an already identity-conditioned baseline.

The correct-direction rank term becomes zero after BA improves correct identity
similarity over null by only `0.02`. In addition:

- the whole causal loss has outer weight `0.25`;
- its internal direct correct-identity term has weight `0.25`;
- the resulting effective direct coefficient is only `0.0625`;
- decoded causal supervision is used only for timesteps `t <= 300`, about 30%
  of sampled microbatches.

This is a reasonable safety-oriented objective, but it does not strongly demand
that BA replace PhotoMaker's face identity. It demands a small, measurable
identity improvement while preserving PhotoMaker-like chroma and structure.

If the scientific goal is “BA controls identity while PhotoMaker controls
pose,” future objectives should use a larger causal margin and/or a ramped
identity coefficient, while retaining preservation constraints.

## Microbatching assessment

Microbatching is not the primary cause of PhotoMaker dominance.

The implementation correctly divides every microbatch loss by the accumulation
count and steps the optimizer only at the end of the accumulation window:

[`sdxl_trainers.py`](../src/trainer/sdxl_trainers.py#L349).

N35 accumulation does change branch-specific Adam semantics:

- with accumulation 4, an optimizer update averages BA-active and BA-inactive
  microbatches;
- approximately 99.2% of optimizer windows contain at least one BA-active
  microbatch;
- an average window contains about 2.8 BA-active microbatches out of 4;
- approximately 76% of windows contain at least one decoded-causal
  microbatch, averaging about 1.2 causal microbatches per window.

Old no-accumulation runs received full branch gradients on active updates and
skipped inactive BA updates. Accumulation produces smoother, more mixed Adam
updates. That can modestly slow early visible authority, but it cannot explain
the roughly 69-fold authority proxy above.

For future branch-specific optimization, consider either:

- keeping one timestep/regime fixed across each accumulation window; or
- normalizing BA-specific losses by the number of BA-active microbatches in the
  window rather than by all microbatches.

This should be tested as an optimization ablation, not treated as the main
architectural fix.

## N35-specific implementation divergence

The original N35 proposal recommended:

- retaining N34's two stable QFormer tokens;
- adding canonical face-part memory through a separate zero-initialized
  residual path.

That would make N35 a strict additive extension of N34.

The implemented N35 instead constructs a new trainable
`CanonicalFacePartResampler` that fuses:

- eight canonical CLIP part features;
- the global InsightFace embedding;
- the mean of the two QFormer tokens;

and replaces N34's two-token BA memory with the resulting eight tokens. See
[`identity_memory.py`](../src/model/photomaker_branched/identity_memory.py#L238)
and
[`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L381).

The final BA output is still exactly zero at initialization, but N35 no longer
has a stable N34 memory path plus an additive part-memory extension. The entire
BA memory must co-adapt through a new random resampler. This can make N35 slower
than N34 in early validations and makes the N34/N35 comparison less clean than
originally intended.

A future N35b should restore the planned decomposition:

```text
stable QFormer BA residual
+ separately gated, zero-initialized canonical-part residual
```

That would reveal whether canonical parts add useful identity information
without forcing them to replace the known QFormer representation.

## Validation-path checks

The PM-like images are not explained by validation dropping the trained BA
weights:

- resolved N34 config has `update_proc_weights_val: true`;
- the alternate RealVis validation model loads the saved processor tensors;
- strict architecture restoration is enabled.

The log message `selected_processors=0` while constructing the temporary
validation model is misleading but does not mean zero processors are installed.
The temporary model is instantiated without the top-level `train_ba_only`
runtime argument, so the trainable-selection list is not populated; the six
processors are still created and their states are loaded.

There is nevertheless a secondary cross-base concern. Training uses
`stabilityai/stable-diffusion-xl-base-1.0`, while validation creates a
`SG161222/RealVisXL_V4.0` model. After loading trainable processor tensors, the
validation path also copies the full training processor state. This can copy
non-trainable cloned K/V base buffers from the SDXL training model into BA
processors attached to the RealVis validation model. It does not disable BA,
but it mixes a RealVis PhotoMaker path with an SDXL-calibrated BA path.

Before drawing a final architecture conclusion, validate the same N34
checkpoint once on:

1. the original SDXL training base, using the training model directly;
2. the usual RealVis validation base.

If BA effects are substantially stronger on the training base, the alternate
base transfer/copy path is part of the problem.

The N34 checkpoint also shows scheduler `last_epoch=4000` after 1,000 optimizer
steps, because Accelerate advances the wrapped scheduler per distributed rank.
N35 will likely advance it by a factor of two. Since `CustomLinearLR` becomes
constant after warmup, this mainly shortens N34 warmup to about 50 optimizer
steps and N35 warmup to about 100 instead of the requested 200. It does not
explain PM dominance after 1k, but N34 and N35 are not perfectly scheduler
matched.

## What to do with the current N34 and N35 runs

Do not stop either run solely because the 1k validation looks similar to
PhotoMaker.

For N34:

- the checkpoint is healthy and all intended parameters are updating;
- continue through at least 3k optimizer steps;
- judge 1k, 2k, and 3k by correct/null/wrong causality, not only visual distance
  from PM.

For N35:

- continue to at least 2k-3k if losses, gates, and resampler norms remain finite;
- expect slower early emergence because the new resampler must co-adapt;
- download the first checkpoint and log for the same optimizer-state and norm
  audit performed on N34.

Stop or redesign by approximately 3k-6k if:

- `causal_identity/correct_gain` remains near zero;
- `causal_identity/wrong_gain` remains near zero or negative;
- changing BA memory does not change identity direction;
- increasing inference residual scale only creates generic color, sharpness, or
  texture changes;
- branch norms grow but correct-vs-null identity does not.

## Highest-value diagnostics before new training

These diagnostics can be run from existing checkpoints and do not require
retraining.

### 1. Correct/null/wrong BA-memory validation

Hold fixed:

- seed;
- text prompt;
- PhotoMaker reference and prompt tokens;
- target bbox;
- all inference settings.

Change only BA memory:

```text
null BA memory
correct BA memory
wrong BA memory
```

Measure:

- similarity to correct identity;
- similarity to wrong identity;
- correct-minus-null gain;
- wrong-minus-null gain;
- landmark, expression, chroma, and face-MAE changes.

This is the primary test of BA identity dominance.

### 2. Explicit residual-scale sweep

For each checkpoint, evaluate:

```text
ba_residual_scale = 0, 1, 2, 4
```

Interpretation:

| Result | Meaning |
|---|---|
| Scale 2-4 improves identity and preserves pose | Training learned a useful BA direction; default authority is too conservative |
| Scale 2-4 changes color/texture but not identity | BA learned a generic face edit |
| Scale 2-4 remains almost unchanged | Route or memory is genuinely underpowered/disconnected |
| Scale 2-4 changes identity but breaks pose/expression | Authority exists, but layer routing or preservation needs refinement |

Using an explicit BA scale is preferable to returning the residual to text CFG.

### 3. BA start-step sweep

Compare BA starting at inference step 10 versus 15. The current schedule gives
PhotoMaker five identity-conditioned denoising steps before BA begins. Earlier
BA may provide more identity authority while the hard mask still protects the
scene.

### 4. Same-base versus RealVis validation

This isolates whether cross-base processor transfer is attenuating the learned
residual.

### 5. Direct epsilon-delta telemetry

Log by denoising-step bucket:

```text
||delta_BA|| / ||pm_cfg||
||delta_BA|| inside the aligned face
correct-memory delta versus wrong-memory delta
```

Parameter norms alone do not show how strongly the branch affects the actual
denoising prediction.

## Recommended next architecture direction

The best successor should preserve post-CFG composition and hard spatial
masking, but explicitly calibrate BA authority.

### Priority 1: remove redundant early attenuation

For a future N34 authority-control run:

- retain zero-initialized `face_delta_out.up`;
- initialize effective gate at `1.0`, not `0.25`;
- keep the gate in FP32 and bounded;
- retain post-CFG composition;
- test explicit inference residual scales `1` and `2`.

This preserves exact PhotoMaker output at initialization while avoiding a
fourfold reduction in the first useful output-adapter gradient.

### Priority 2: add limited multi-resolution identity authority

Six `up_blocks.1` sites may be sufficient for skin, eye, mouth, and contour
detail but insufficient for stronger face-shape identity changes.

Do not immediately return to all 70 sites. Add a small named subset of late
`up_blocks.0` cross-attention sites with a separate, lower gate. The desired
factorization is:

```text
selected up_blocks.0: low-strength identity geometry
up_blocks.1: stronger local identity detail
PhotoMaker: pose, expression, scene, and global rendering
```

### Priority 3: strengthen causal identity demand gradually

Potential future settings to ablate:

- increase causal margin from `0.02` toward `0.05`;
- ramp outer causal weight from `0.25` to `0.5` after the output adapter has
  opened;
- raise the direct correct-identity term while retaining wrong-direction and
  chroma/structure constraints;
- normalize causal supervision by the number of causal-active accumulated
  microbatches.

Use a ramp rather than a large constant from step 0.

### Priority 4: make BA the explicit identity source

If the true objective is that BA, rather than PhotoMaker, controls identity,
the architecture should separate PhotoMaker geometry from PhotoMaker identity.

The current baseline still gives PhotoMaker the correct identity globally, and
then asks BA for a small improvement. A stronger factorized design would:

- use PhotoMaker/text features to establish pose, expression, clothing, and
  scene;
- attenuate or neutralize PhotoMaker identity authority inside the target face
  during the BA-active phase;
- let BA memory be the principal identity-specific input inside the face;
- preserve the null PM path as a safety and regularization target.

This is a larger architecture change, but it aligns the implementation with the
stated goal “BA-based face identity with PhotoMaker posture.”

### Priority 5: correct N35 into an additive memory experiment

Keep N34 QFormer memory and add canonical parts through a distinct
zero-initialized residual with its own gate. This makes it possible to measure
whether canonical parts add value, instead of asking a new resampler to replace
the complete memory path.

## Recommended experiment order

1. Continue current N34/N35 to 3k unless causal metrics fail badly.
2. Run checkpoint-only null/correct/wrong, residual-scale, start-step, and
   same-base/RealVis diagnostics.
3. If scale 2-4 reveals good identity direction, run an authority-calibrated
   N34 successor with gate init `1.0`.
4. If identity direction is present but face shape remains PM-like, add a small
   gated `up_blocks.0` subset.
5. If identity direction is absent even under a scale sweep, strengthen the
   causal objective and identity-source factorization.
6. Rebuild N35 as an additive canonical-part residual after the N34 authority
   route is calibrated.

## Bottom line

N34/N35 are capable of changing faces, but the current runs were configured to
make BA a small, safe correction to PhotoMaker. The architecture does not yet
ensure that BA dominates identity.

The most likely explanation for the first-validation similarity is the stacked
authority reduction, not a disconnected optimizer:

```text
six sites
× gate 0.25
× no CFG amplification
× BA starts after PM
× small improvement-over-PM causal margin
× first validation at only 1k optimizer steps
```

The next decision should be based on identity-memory causality and explicit
residual-scale sweeps. If a larger explicit scale reveals useful identity
movement while preserving pose, the core learned direction is sound and only
authority calibration needs correction.
