# NN3a at 4k: metrics, training diagnosis, and stop decision

Date: 20 July 2026

## Executive decision

**Pause/stop the current NN3a run after preserving the 4k checkpoint. Do not
spend the full 20k budget on the unchanged configuration yet.**

The 0–4k evidence does **not** show a failed optimizer, bypassed branch,
incorrect validation checkpoint, OOM, NaN, or wrong architecture selection.
NN3a is training and its validation pipeline is using the learned processor
weights.

The problem is architectural/objective-level:

1. the visible output remains very close to step-zero PhotoMaker;
2. validation identity similarity moves slightly in the wrong direction;
3. the gate barely changes;
4. the raw connector residual is already being clipped at most training
   sites, so more raw-weight growth will not automatically create more visible
   authority;
5. the implemented “reference-minus-null” operation is mathematically
   `C_ref - 0 = C_ref`, not a trained matched-versus-null contrast;
6. there is no null-reference forward, null-residual loss, reference-dependence
   objective, or target-PhotoMaker identity attenuation.

The saved 4k checkpoint is still valuable. Before discarding it, run a short
inference-only scale and reference-swap diagnostic. If a `2×–4×` residual
reveals correct, clean reference-identity direction, the architecture may only
be under-calibrated at `1×`. If stronger scaling produces generic expression
changes or no directional identity transfer, terminate NN3a and start the
dependence-enforced configuration described below.

## Data reviewed

Export:

```text
comet_data/metrics_only_NN3a/ba_NN3a_reference_null_1gpu/
```

Files used:

- `metrics_history.json`;
- `metrics_summary.json`;
- `comet_run_export.json`;
- `comet_output.log`.

Run:

```text
Comet ID: d208b88f0ef646549930003bded059cb
Original name: ba_NN3a_reference_null_realvis_1gpu
Config: one_id_ba_NN3a_reference_null
Dataset: cosmic_large_neb
Training base: stabilityai/stable-diffusion-xl-base-1.0
Validation base: SG161222/RealVisXL_V4.0
Physical/effective batch: 2/2
Training timesteps: t in [0, 699]
Warmup: 2k optimizer steps
```

The export contains metrics and logs but not validation images or local
face-region metrics. The visual conclusion in this report therefore uses the
user's observation that the 4k images differ only minimally from step zero.
Face MAE, LPIPS, detection rate, and per-image progression cannot be
independently measured from this export.

## 1. Validation metrics

| Step | Original-ID similarity | Text similarity |
|---:|---:|---:|
| 0 | 0.523129 | 26.36589 |
| 2k | 0.514099 | 26.35775 |
| 4k | 0.512338 | 26.38216 |
| 0→4k change | **-0.010791** | **+0.01628** |

Interpretation:

- original-ID similarity falls by about `2.1%` relative to its step-zero
  value;
- most of that fall occurs by 2k;
- text similarity is essentially flat;
- neither metric provides evidence of useful progress by 4k;
- the ordinary ID metric tests similarity to the original identity, not
  whether spatial-reference identity controls the branch.

The small ID decrease is consistent with a weak learned face residual, but it
does not indicate that the residual is moving toward the reference in a useful
way.

## 2. The correct NN3a architecture is running

The startup log confirms:

```text
variant=packed_residual_v1
site_policy=up_blocks_attn1
SA=36
CA=70
connector_input=reference_minus_null
```

Trainability is also correct:

```text
252 trainable tensors
6.76M trainable parameters
36 reference-K LoRA pairs
36 reference-V LoRA pairs
36 connector-down matrices
36 connector-up matrices
36 scalar gates
```

All 70 split cross-attention processors execute but remain frozen. All 36
packed self-attention processors are installed. The optimizer has the intended
252 trainable tensors.

There are no logged:

- Python exceptions or tracebacks;
- OOM skips;
- NaNs or infinities;
- invalid-sample skips;
- processor-restore mismatches.

## 3. Validation is not stuck on step-zero weights

Step-zero validation reports the expected exact bypass:

```text
[BA output anchor] state=exact-zero-bypass
pre/post residual ratio=0
```

At 2k and 4k, validation instead reports:

```text
[BA output anchor] state=base-outside-core
nonzero pre/post residual ratios
gate≈0.252
```

The 4k validation pipeline therefore contains learned processor weights. The
small image change is not explained by accidentally validating an
uninitialized model.

The validation schedule is also correct:

```text
steps 0–9: text only
steps 10–14: PhotoMaker
steps 15–49: PhotoMaker + branched attention
```

The log saves:

```text
saved/ba_NN3a_reference_null_realvis_1gpu/checkpoint-epoch2.pth
saved/ba_NN3a_reference_null_realvis_1gpu/weights-epoch2.pth
```

immediately after the 4k validation. The exported log then starts epoch 3.

## 4. Gradients and parameter movement

The zero-initialized connector-up projection creates intentional staged
learning:

1. connector-up receives gradients immediately;
2. connector-down, gate, and reference K/V initially receive zero or nearly
   zero gradients;
3. once connector-up becomes nonzero, gradients reach every component.

This sequence appears correctly in the log.

At the exported 4k point:

| Group | Gradient norm |
|---|---:|
| connector up | 0.003845 |
| connector down | 0.000422 |
| reference V | 0.000221 |
| gate | 0.0000753 |
| reference K | 0.0000384 |
| total | 0.003875 |

The reference LoRA-B norm canary grows:

```text
step 0:    0
step 1k:  1.068
step 2k:  5.632
step 3k:  7.048
step 4k:  7.872
```

This is strong evidence that training is live. Growth becomes sublinear after
2k rather than explosively doubling, so the log does not show the old runaway
branch-weight failure.

## 5. Residual magnitude and RMS-cap saturation

The branch uses:

```text
raw_delta     = Connector(C_ref)
bounded_delta = RMSCap(raw_delta, maximum=0.25 relative to target attention)
gate          = 0.5 * sigmoid(gate_logit)
applied_delta = face_core * gate * bounded_delta
```

The mean gate changes only from `0.250` to approximately `0.252`: less than a
one-percent relative increase.

Representative final training diagnostics at 4k:

| Sites | Number | Mean pre-cap ratio | Mean post-cap ratio | Fully capped |
|---|---:|---:|---:|---:|
| `up_blocks.0` | 30 | 0.453 | 0.241 | **27/30** |
| `up_blocks.1` | 6 | 0.174 | 0.144 | 2/6 |
| all | 36 | 0.407 | 0.225 | **29/36** |

The mean reported cap fraction across sites is about `0.81` in that final
training diagnostic batch.

A representative 4k validation diagnostic is milder:

| Sites | Mean pre-cap ratio | Mean post-cap ratio | Mean cap fraction |
|---|---:|---:|---:|
| `up_blocks.0` | 0.218 | 0.190 | 0.325 |
| `up_blocks.1` | 0.063 | 0.063 | 0 |
| all | 0.192 | 0.169 | 0.271 |

After the approximately `0.252` gate, the representative validation
applied/base ratio is only about:

```text
0.252 * 0.169 ≈ 0.043
```

inside the face core at one site before downstream propagation. The
independent PhotoMaker epsilon remains exact outside that core.

This explains why standard `1×` validation can remain close to PhotoMaker.
More importantly, many training sites are already using capacity to increase a
raw magnitude that the cap removes. Continuing longer may mostly increase
discarded pre-cap norm unless the residual direction or gate changes.

This is not evidence that the cap should simply be raised. The NN2 diagnostics
showed that amplifying a semantically wrong residual increases expression and
texture artifacts without producing reference-identity control.

## 6. Loss behavior

Approximate 1k-window means:

| Steps | Total loss | Logged ID loss | ID applied fraction | Weighted ID term |
|---:|---:|---:|---:|---:|
| 0–1k | 0.186 | 0.106 | 0.442 | 0.00531 |
| 1–2k | 0.175 | 0.090 | 0.379 | 0.00450 |
| 2–3k | 0.175 | 0.091 | 0.394 | 0.00454 |
| 3–4k | 0.181 | 0.097 | 0.442 | 0.00485 |

The identity loss is active at approximately the expected frequency:

```text
P(t <= 300 | t sampled uniformly from 0..699) ≈ 43%
```

Its weighted contribution is only about `2–3%` of total loss. After early
improvement it plateaus rather than continuing downward. This matches the
flat/worsening validation identity metric.

The conservative weight prevents the severe NN1e identity-loss behavior, but
it also does not supply a strong reference-dependence signal.

## 7. Main architectural shortfall

The implemented NN3 path is:

```python
null_candidate = torch.zeros_like(reference_candidate)
connector_input = reference_candidate - null_candidate
```

Therefore:

```text
connector_input = C_ref - 0 = C_ref
```

This is useful because it removes NN2's explicit `-A_target` shortcut. It is
not, however, a true matched-versus-null contrast:

- no null reference is encoded;
- no null K/V candidate is produced through the same reference projections;
- no second forward shares target/noise while changing only the reference;
- no loss penalizes a null-reference residual;
- no loss requires matched-reference output to differ from null output;
- no target-side PhotoMaker-ID dropout forces identity ownership into BA.

A zero candidate also does not cancel generic components in `C_ref`. It only
subtracts zero. The connector can still learn an average face, rendering,
expression, or prompt correction from reference features without learning
which identity-specific dimensions matter.

The low-timestep ID loss compares the generated face with a same-identity
reference, but ordinary target PhotoMaker conditioning already contains strong
identity evidence. The frozen PM path can solve much of the identity task
without the spatial branch. A small residual is therefore a low-loss solution.

This is the strongest explanation for “training is active, but images hardly
move.”

## 8. Immediate checkpoint diagnostic

Before starting another long run, use the 4k epoch-2 checkpoint for a fixed
inference matrix:

```text
A: branch off / runtime scale 0
B: matched reference, scale 1
C: matched reference, scale 2
D: matched reference, scale 4
E: cyclic swapped reference R2, scale 4
F: second reference-noise seed, matched reference, scale 4
```

Hold fixed:

- target latent and seed;
- prompt and target PhotoMaker embedding;
- scheduler and CFG;
- target/reference masks;
- batch size;
- validation base.

Calculate:

- face-core MAE and LPIPS against branch-off;
- original-ID similarity;
- similarity to matched and swapped references;
- directional identity gain toward R2;
- face-detection rate and landmark displacement;
- outside-face MAE/LPIPS;
- per-site applied residual ratio and cap fraction.

Decision:

| Result | Action |
|---|---|
| `2×–4×` is clean and moves identity toward the supplied reference | Keep checkpoint; the primary problem is `1×` calibration. Resume only after choosing the useful scale. |
| Stronger scale changes expression/texture but not reference identity | Terminate NN3a; the learned semantics remain wrong. |
| Scale 4 is still nearly identical | Terminate NN3a; branch authority/optimization is too weak despite nonzero internal residuals. |
| Swapped reference has little effect relative to noise | Terminate NN3a; reference dependence is not learned. |

This test is more informative than waiting for another ordinary scale-1
validation.

## 9. Recommended fixed next configuration: NN3b

Keep:

- doubled target/reference streams;
- target Q with packed spatial reference K/V;
- frozen split cross-attention;
- bounded additive face residual;
- independent PhotoMaker output outside the face core;
- `cosmic_large_neb`;
- RealVis fixed validation.

Change the architecture/objective:

### A. Use a real matched-versus-null candidate

Build both candidates through the reference route with the same target query:

```text
C_ref  = Attn(Q_target, K_matched, V_matched)
C_null = Attn(Q_target, K_null,    V_null)
delta  = Connector(C_ref - C_null)
```

`K_null/V_null` should represent an explicit no-person/null memory, not simply
`zeros_like(C_ref)` after attention.

### B. Add paired null-reference training

For the same target latent, timestep, prompt, and noise:

- matched reference: ordinary diffusion/face objective plus reference-ID
  objective;
- null reference: explicitly require applied branch residual to approach zero;
- optionally require matched residual/features to remain separated from null.

This turns “reference minus null” into an actual training constraint.

### C. Add controlled target PhotoMaker-ID attenuation

On a controlled subset of matched-reference BA-active batches, attenuate or
drop target-side PhotoMaker identity conditioning. Keep full-PhotoMaker batches
in the mixture.

A reasonable first screen is:

```text
50% full target PhotoMaker ID
50% attenuated target PhotoMaker ID with matched spatial reference
```

This forces the spatial route to carry identity information while preserving
the deployment regime in half of training.

Do not use wrong references with the ordinary matched diffusion target. If
wrong references are introduced, give them an explicit contrastive identity
semantics or a branch-off target; otherwise the optimal solution is to ignore
the reference.

### D. Keep gate and cap conservative initially

Do not simultaneously increase:

- `ba_gate_max`;
- `ba_delta_rms_cap`;
- identity-loss weight.

First establish directional reference control. If the 4k scale diagnostic
shows correct but under-strength behavior, calibrate runtime scale or gate as
one isolated change.

## 10. Stop/continue recommendation

The current run has already answered its main question:

> Removing `-A_target` prevents that direct shortcut, but `C_ref - 0` plus a
> small ordinary identity loss does not produce clear reference-controlled
> movement by 4k.

There is no evidence that a concise bug fix to checkpoint loading, validation,
optimizer registration, or processor routing would rescue the current
trajectory. The appropriate action is:

1. preserve `checkpoint-epoch2.pth` and `weights-epoch2.pth`;
2. pause/stop the running process;
3. run the 4k inference diagnostic;
4. resume only if stronger inference reveals clean directional identity;
5. otherwise implement NN3b and start fresh.

The epoch-2 checkpoint is sufficient for this decision. Downloading it locally
would additionally allow exact per-site gate, connector, and LoRA norm
inspection, but the current logs already establish that the branch is live and
that the main issue is not a training bypass.

