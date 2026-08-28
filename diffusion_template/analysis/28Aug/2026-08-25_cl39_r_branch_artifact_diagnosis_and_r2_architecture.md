# CL39 reference-branch artifacts: diagnosis and a buildable R2 architecture

**Date:** 25 August 2026  
**Evidence cutoff:** 18:30 BST, 25 August 2026  
**Scope:** canonical CL39 r4 at 24k; exact Serv A100 attention audit and
same-checkpoint interventions; clean-new CL14/E13 mechanism; immutable
CL14/CL19/CL27/CL39 fixed-panel records; completed CL40-CL44 architecture
results. No new training result is claimed in this report.  
**Primary checkpoint:** CL39 r4, epoch 12 / 24k, immutable Comet key
`b1ca0b3da679401c85b991f1bbdf0b2a`, checkpoint SHA-256
`74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07`.  
**Reproducible assets:** exact audit, branch, and report-figure locations are
listed in Section 9.

## Executive answer

### 1. Is there an issue with R in CL39?

**Yes, but it is a branch-conditioning and robustness issue, not evidence that
ordinary CL39 generations are generally bad.** The raw `R-on-face` images are
not CL39's trained operating point. That intervention replaces the native
attention message `N` by the full reference message `R` inside the face router
at the 36 CL39 confidence-enabled `up_blocks.0/1` processors throughout
denoising. The other 34 transferred BA processors retain their ordinary
CL23/27 temporal-frequency route. Actual CL39 instead keeps `N` as the anchor
in those 36 processors and applies an average face correction whose magnitude
is only `0.22099` of `N`. The raw intervention is therefore a deliberately
severe but group-scoped stress test. `[code][measured]`

The artifacts are nevertheless meaningful. Raw `R` is not self-sufficient:
it often duplicates or misregisters eyes, noses, glasses, hands, and expression
geometry. More importantly, a less extreme same-checkpoint intervention that
keeps the full CL39 combined-denoising equation but forces confidence `C=1`
reduces subject-v2 ID from `0.55754` to `0.49984` on the selected 16 cells.
Actual CL39 beats forced `C=1` by `+0.05770`, with `14/16` paired wins and a
fixed-cell bootstrap interval `[+0.03432,+0.07949]`. Thus unattenuated
reference correction can harm the final image, not merely the visualization.
`[measured][paired]`

At the same time, the audited correction is useful when controlled. Zeroing
the correction in those same 36 processors lowers ID from `0.55754` to
`0.51925`; actual CL39 gains `+0.03829`, wins `15/16`, and has interval
`[+0.02093,+0.05774]`. This is not a global 70-processor BA-off arm. On the
full fixed-96 scientific panel, CL39 reaches `0.570124` at 16k and `0.566342`
at 24k, with near-PhotoMaker face quality and clean fixed Skiing/Crying
topology. The correct conclusion is: **R contains useful identity structure,
but the audited raw lane is underconstrained and must remain anchored,
attenuated, and better calibrated.** `[measured][visual][caveat]`

### 2. Can the R part be improved architecturally?

**Probably, and CL14 points to the most direct improvement.** CL14 made its
reference face lane own the face output during training, so that lane had to
be a usable denoising path. CL39 only needs `R-N` to be useful after spatial,
frequency, and confidence scaling. It never asks raw `R` to produce a coherent
face by itself. The recommended CL39-R2 therefore borrows CL14's *training
obligation* without reverting to CL14's hard inference handover:

1. retain CL39's full target-query `N` and explicit target-query/reference-KV
   `R` lanes;
2. on a small, globally selected fraction of training batches, route the whole
   U-Net through `N + S(R-N)` with a warm ramp, forcing `R` to learn a coherent
   face route under the ordinary diffusion target;
3. keep the normal inference route anchored at `N`;
4. replace the single entropy-only scalar with bounded, zero-initialized
   low/high reliability corrections that can see valid-reference mass,
   conditional entropy, `N/R` agreement, residual/native norm, and denoising
   progress; and
5. add separately switchable RMS tail caps for `D_low` and `D_high`.

This should be built as an ablation ladder—route training first, gate/caps
second, combination only if both pass—not as one bundled run. The first stage
adds no inference parameters or extra U-Net forward and is the easiest causal
test. `[proposal][decision]`

# 1. Evidence integrity and exact intervention

## 1.1 What the branch figure actually renders

At one self-attention processor, CL39 computes two output-projected messages:

$$N=O(A(Q_t(T),K_t(T),V_t(T)))$$

$$R=O(A(Q_t(T),K_r(H_rM_r),V_r(H_rM_r)))$$

where `T` is the target state, `H_r` is the evolving reference state, `M_r`
is its face mask, and `O` is the attention output projection. Target queries
remain in target coordinates; `pose_adapt_ratio=0`, face cross-attention mixing
is off, and the explicit identity lane reads reference K/V. `[code]`

The branch audit used the exact checkpoint and ordinary 50-step validation
loop, but replaced the target attention output at every shipped CL39 processor
with:

$$Y_{R\text{-on-face}}=N+S(R-N),$$

where `S` is the existing two-cell soft face router. This is a coherent
whole-denoising intervention. It is **not** a VAE decode of an attention tensor,
but it is also **not** the trained CL39 equation. `[code][method]`

Actual CL39 first forms `D=R-N`, splits it with the fixed 5x5 Gaussian into
`L+H`, applies progress-dependent gains, and—in `up_blocks.0/1`—multiplies both
bands by detached entropy confidence:

$$Y_{CL39}=N+S\,C(q)\,[g_L(p)L+g_H(p)H].$$

The other five block groups retain the CL23/27 temporal-frequency route without
the CL39 confidence extension. The sealed model transfers 70 BA processors in
total; the raw branch intervention removes every frequency and confidence
restraint at the 36 null-key processors and leaves the other 34 unchanged.
`[code]`

## 1.2 Exact Serv evidence

The final audit used the sealed Serv configuration, RealVisXL V4.0 validation
base, DDIM50, CFG5, seed0, fixed prompts/references/boxes, and exact checkpoint.
The instrumented actual arm matched all 96 sealed PNGs byte-for-byte. The
branch job was Serv MLS job
`lm-mpi-job-1d94aa46-9197-466a-8f22-1abcce4e4312`; it completed 96 images and
rendered the deterministic 16-cell diagnostic set. `[measured][record]`

| Quantity on selected 16 | Result | Interpretation |
|---|---:|---|
| Raw `R-on-face` vs N-only, global RGB MAE | `0.02163` | large whole-image propagation from a face-local attention change |
| Raw `R-on-face` vs N-only, face RGB MAE | `0.07253` | strong final face intervention |
| Face pixels changed above `1/255` | `95.24%` | raw R is not a small cosmetic perturbation |
| Mean raw `|R-N|` feature magnitude on face | `0.45339` | unscaled source residual |
| Mean actual routed correction magnitude | `0.11939` | `3.80x` smaller than raw residual magnitude |
| Actual correction/native magnitude | `0.22099` | actual CL39 remains N-anchored |
| Face confidence | `0.49485` | mean query confidence in confidence-enabled layers |
| Effective low/high face weights | `0.24793 / 0.36627` | neither band is close to raw unit replacement on average |
| High-band applied-magnitude share | `37.15%` | high band is material, not a no-op |
| High/low normalized spatial-TV ratio | `3.817x` | high band is much more edge/detail sensitive |

![](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_2.png){ width=100% }

*Figure 1. Jennie and Jensen examples. `R-on-face` is a whole-denoising stress
test. Jensen/Skiing and Jensen/Kickboxing show the clearest duplicated glasses,
eyes, nose, and mouth geometry; actual CL39 remains coherent.* `[visual]`

![](assets/cl39_attention_24k_serv_a100_branch_faces/cl39_branch_faces_overview_3.png){ width=100% }

*Figure 2. Jisoo and Keanu examples. The crying cell shows hand/face transfer
and the other raw-R cells show milder pose/expression drift. The signed panel
confirms that `R-N` carries structured face signal rather than random noise.*
`[visual]`

# 2. What is a visualization artifact, and what is a real problem?

## 2.1 Reasons not to judge ordinary CL39 by raw R

1. **The intervention changes the optimization contract at inference.** CL39
   was trained to use scaled bands of `R-N`, not to replace `N` by `R`.
2. **It is repeated across the U-Net.** A locally plausible attention error can
   be reinforced over 35 active denoising steps and 36 late/up-block
   processors. The stress test does not expose raw R in the other 34 BA
   processors.
3. **`R` is an attention update, not a full image hypothesis.** The processor
   still has the U-Net residual connection, but raw `R` is not independently
   decoded or supervised as a complete face.
4. **The magnitude is far outside the operating point.** Mean raw residual is
   `3.80x` the actual routed residual; actual correction is about `22.1%` of
   native attention magnitude.

Therefore a bad `R-on-face` image does not imply that the corresponding actual
CL39 image is bad. `[code][measured][inference]`

## 2.2 Reasons the concern remains valid

### Forced C=1 fails in the real combined denoising path

Both counterfactuals below act only on the 36 CL39 null-key processors; the
other 34 BA processors remain at their ordinary temporal-frequency route.

| Same CL39 checkpoint, selected 16 | Subject-v2 ID | Paired result vs actual |
|---|---:|---:|
| Actual CL39 | `0.557538` | — |
| CL39 up0/up1 correction off, trained adapters retained | `0.519252` | actual `+0.038286`, `15/0/1`, CI `[+0.020927,+0.057738]` |
| Confidence forced to 1 | `0.499840` | actual `+0.057699`, `14/0/2`, CI `[+0.034316,+0.079494]` |

This is the strongest evidence that raw-reference errors can matter to final
denoising. The confidence is not merely hiding an ugly diagnostic; it prevents
a measurable failure when the same residual is over-applied. `[measured][paired]`

### Current entropy is not correspondence correctness

CL39 zeroes reference hidden states outside the reference face box but leaves
those token positions in the softmax. Face queries place `51.44%` of reference
attention mass inside boxes that occupy `16.68%` of the grid, while `48.56%`
still goes to zero-sink positions. Entropy is normalized over all `L` positions,
so it mixes at least three effects:

- genuine ambiguity among reference-face locations;
- the number and area of zero-sink positions; and
- attention-logit scale.

A sharply concentrated but wrong eye-to-glasses match can have low entropy and
high confidence. Conversely, a useful distributed match over a large face can
have high entropy. CL39's `C` is therefore an abstention heuristic, not a
probability that `R` is geometrically correct. `[code][measured][inference]`

On the selected 16, mean entropy confidence has only `r=+0.11` Pearson and
`rho=+0.15` rank correlation with the final face-pixel size of the raw-R
intervention. Raw `|R-N|` magnitude correlates more strongly (`r=+0.56`,
`rho=+0.62`). Face MAE is not a quality score, and 16 cells are too few for a
general result; this is a diagnostic that confidence does not track even the
size of the raw intervention well. `[measured][limitation]`

![](assets/cl39_r2_architecture_20260825/fig_r_intervention_diagnostic.png){ width=100% }

*Figure 3. Selected-cell diagnostic. The plot does not label images good or
bad; it shows why entropy alone is insufficient for residual calibration.*

## 2.3 End-to-end CL39 does not show a general artifact collapse

The fixed-96 evidence goes in the opposite direction. CL39-16k is the current
end-to-end winner at `0.570124`, versus CL27-16k `0.547260` and PhotoMaker
`0.556580`. It is `8 pass / 0 minor / 0 fail` on the fixed Skiing topology
rubric and `8/0/0` on Crying. Its TOPIQ-Face is `0.7399` versus PhotoMaker
`0.7532`; MUSIQ is `73.0433` versus `73.0988`; MANIQA is `0.6412` versus
`0.6437`. `[measured][visual]`

CL39 still has cell-level regressions, and only one training seed is available.
The evidence supports improving R for robustness and possible identity
headroom; it does not support rejecting CL39 because the raw stress test looks
bad. `[decision][limitation]`

# 3. Why CL14 can look good when its route is more R-like

## 3.1 Exact mechanism difference

The clean-new CL14/E13 processor computes:

$$B=A(Q_t(T)(1-M),K_t(T),V_t(T)),$$

$$F=A(Q_t(T)M,K_r(H_rM_r),V_r(H_rM_r)),$$

$$Y_{CL14}=O((1-M)B+MF).$$

Inside the binary face core, `F` owns the attention update. The ordinary
diffusion loss therefore trains the reference lane as the only face path at
every installed processor. CL14's nominal two-cell training feather is
thresholded at `>0.5` by the historical processor; it acts as a narrowed hard
support, not a continuous blend. `[code]`

CL39 instead computes two complete projected messages and treats `R` mainly as
the source of a correction. Its training gradient to the reference lane is
scaled by the spatial router, low/high gains, and—only in up0/up1—the detached
confidence. The two designs therefore impose different obligations:

| Property | CL14 hard face route | CL39 temporal/confidence route |
|---|---|---|
| Reference lane target | usable face attention output | useful residual source around N |
| Face-core ownership | `F` is the attention update | `N` remains anchor; scaled `R-N` is added |
| Target queries | query-masked before attention | full target queries in both lanes |
| Merge | hard binary, before shared output projection | soft spatial, after separate message projection |
| Frequency control | none | Gaussian low/high schedule |
| Confidence | none | entropy confidence in up0/up1 |
| Invalid reference positions | zero sinks remain in softmax | same historical zero-sink behavior |
| Main failure pressure | seams and object/face ownership | underconstrained or over-trusted residual |

The shared zero-sink behavior is important: it cannot by itself explain why
raw CL39 R looks worse. The strongest architectural explanation is that CL14
*must* make its face lane independently useful, while CL39 can minimize loss
with a good native path and a small, selectively useful correction.
`[code][inference]`

![](assets/fig_cl14_marion_face_crops.png){ width=96% }

*Figure 4. CL14 Marion face crops from the sealed fixed panel. Many results are
visually coherent because the hard reference face route is trained as the
face owner. The panel also shows identity/detail variability; good-looking
examples do not establish stronger aggregate performance.* `[visual]`

## 3.2 CL14 is a clue, not the architecture to restore

| Fixed-panel result | Subject-v2 ID |
|---|---:|
| CL14 24k, Comet `6fe0028be92242c38056b3d36665fdd6` | `0.456116` |
| CL19 24k, Comet `cfeda7b55c174b3c83e8d40537ebb6dd` | `0.506823` |
| CL39 24k, Comet `b1ca0b3da679401c85b991f1bbdf0b2a` | `0.566342` |
| CL39 selected 16k | `0.570124` |
| PhotoMaker fixed reference | `0.556580` |

CL19's full-query, single soft merge improved CL14 by `+0.050707` at 24k,
with `74/96` paired wins and a fully positive cell-bootstrap interval. CL14
also retained structured Skiing/goggle and small-face weaknesses. A direct
reversion to hard CL14 would discard the largest proven routing improvement
in the lineage. `[measured][paired][decision]`

The useful lesson is narrower: **periodically require the CL39 reference lane
to carry the face during training, but keep the better N-anchored CL39 route at
inference.** `[proposal]`

# 4. Proposed CL39-R2 architecture

## 4.1 Design invariants

The proposal preserves the project's BA definition:

- target queries remain in target coordinates;
- the identity lane reads explicit reference K/V;
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false` in training and validation;
- native `N` is the inference anchor;
- old checkpoints and CL39 behavior remain available behind defaults-off
  toggles; and
- sampling, panel, scheduler, seeds, prompts, boxes, metrics, and validation
  cadence remain unchanged.

It deliberately does **not** use fixed landmark canonicalization, component
tokens, or a generic identity-motion projector. Those close alternatives were
already tested: CL41 canonical K/V was negative, while CL40/CL42/CL43 were
neutral. CL44's modestly positive time/agreement result motivates
band-specific reliability, but not a wholesale switch to CL44. `[measured][decision]`

![](assets/cl39_r2_architecture_20260825/fig_cl39_r2_architecture.png){ width=100% }

*Figure 5. Target architecture. Dashed orange is training-only. The normal
inference route remains N-anchored and retains explicit target-Q/reference-KV
BA.*

## 4.2 Component A: coherent R-route dropout

Sample one route mode for the entire U-Net forward, on CPU and shared across
the selected processors/ranks. On ordinary batches, keep exact CL39. In the
first arm, select the same 36 `up_blocks.0/1` processors exposed by the audit;
the other 34 retain their ordinary route. On a proposed initial `12.5%` of
training batches after a warm start, interpolate the selected group toward:

$$Y_{R\text{-train}}=N+S(R-N).$$

Use a route-strength ramp `lambda_R` from 0 to 1 over a bounded interval, for
example optimizer steps 2k-6k:

$$Y_{train}=N+S[(1-\lambda_R)C(g_LL+g_HH)+\lambda_R(R-N)].$$

The normal diffusion target supervises this route. No second U-Net pass,
teacher model, ArcFace loss, or inference parameter is required. The route
decision must be global; independently sampling processors would create a
mixture that is neither CL39 nor a coherent R path. `[proposal]`

This is the direct CL14-inspired change. It gives the audited `R` processors
enough full-ownership training exposure to learn facial topology while
retaining normal CL39 on `87.5%` of batches and at all validation/inference
calls. Extending route training to all 70 BA processors is a later, separate
scope experiment. The probability and ramp are experiment settings, not
established optimums. `[hypothesis][limitation]`

## 4.3 Component B: bounded low/high reliability correction

Keep current CL39 confidence `C_0(q)` as the baseline. Add detached diagnostic
features for each query/layer:

- mass assigned to valid reference-face key positions;
- entropy conditional on those valid positions;
- cosine agreement between normalized `N` and `R`;
- low/native and high/native RMS ratios; and
- denoising progress.

A tiny zero-initialized MLP emits two bounded logit corrections, one per band:

$$C_b=\mathrm{clip}(\sigma(\mathrm{logit}(C_0)+
\delta_{max}\tanh(f_b(x)))), C_{min},1),\quad b\in\{L,H\}.$$

Zero final weights make `C_L=C_H=C_0` at initialization. Inputs are detached,
so the reference projections cannot learn to fake a confidence statistic.
Separate bands let the model be conservative with high-frequency edges while
retaining low-frequency identity structure. Bound `delta_max`, retain
`C_min=0.25`, and initially keep the CL39 `up_blocks.0/1` scope. Extending the
gate to all seven groups is a later independent experiment. `[proposal]`

This learned correction is intentionally small. A free gate could collapse
toward native PhotoMaker and obtain a deceptively clean result. Promotion must
therefore require both group-scoped correction-zero and global BA-off controls,
plus the spatial-reference-shuffle causal gate in Section 6.
`[risk][decision]`

## 4.4 Component C: separately switchable residual tail caps

For each sample/layer, compute masked face RMS and cap the low/high bands before
their gains:

$$\hat b=b\,\min\left(1,\frac{k_b\,\mathrm{RMS}_{face}(N)}
{\mathrm{RMS}_{face}(b)+\epsilon}\right).$$

The 20,160 recorded layer calls give audit starting points rather than tuned
constants:

| Ratio before CL39 confidence | p50 | p95 | maximum |
|---|---:|---:|---:|
| low/native face RMS | `0.767` | `0.910` | `1.424` |
| high/native face RMS | `0.301` | `0.425` | `0.631` |
| raw `(R-N)`/native face RMS | `0.880` | `1.021` | `1.499` |
| actual routed/native face RMS | `0.200` | `0.415` | `0.691` |

An initial center sweep can use `k_L=0.90` and `k_H=0.45`, with lower/higher
neighbors. These values touch only the observed tails, but they are not proven
quality optima. Caps must be an independent toggle so their causal effect is
measurable and CL39 parity is exact when off. `[measured][proposal][limitation]`

The final normal route becomes:

$$Y_{R2}=N+S[C_Lg_L\hat L+C_Hg_H\hat H].$$

# 5. Implementation plan

## 5.1 Localized code changes

| File | Change | Invariant |
|---|---|---|
| `src/model/photomaker_branched/attn_processor_cleanest.py` | add defaults-off R2 flags, valid-mass/conditional-entropy diagnostics, per-band bounded gate, optional RMS caps, and externally supplied global route mode/strength | existing CL39 forward remains bit-identical with flags off; no per-layer RNG |
| `src/model/photomaker_branched/lora2.py` | declare/configure R2 fields; sample or derive one coherent training route per forward; attach it to all selected processors; expose telemetry and architecture manifest | inference always normal route; DDP ranks agree; route is never sampled independently per processor |
| `src/model/photomaker_branched/branched_runtime.py` | pass R2 controls only to declared processor groups; install tiny gate modules when enabled | no lookup of `unet.attn_processors` inside a per-layer loop |
| `src/model/photomaker_branched/lora2_helpers.py` | collect compact gate/cap/route telemetry only when requested | no full-activation telemetry by default |
| `src/configs/model/photomaker_branched_lora2.yaml` | add defaults-off schema | historical configs compose unchanged |
| new `src/configs/CL39_R2_*.yaml` leaves | one YAML per ablation, inheriting CL39 | no silent dataset, seed, scheduler, mask, box, or metric change |
| `tools/validate_CL39_R2_config.py` and run registry | validate group scopes, probability/ramp, exact trainable manifest, `pose_adapt_ratio=0`, CA mixing off | fail closed before Serv submission |
| existing active Serv packaging launcher | render immutable YAML package and submit after MLS GPU audit | optimized pipeline and standard step0/every2k fixed-96 validation |

## 5.2 Suggested implementation order

1. **Parity foundation.** Add fields and plumbing with all flags off. Load the
   CL39 checkpoint and reproduce the existing 12-cell byte-exact smoke.
2. **R2-A: route dropout only.** No gate parameters or caps. Verify one route
   decision is shared by all 36 selected processors, ranks, and
   gradient-accumulation replicas; verify the other 34 processors are
   unchanged. Confirm inference is exactly CL39.
3. **R2-B: bounded band gate only.** Zero-init the final gate layer; compute the
   exact trainable tensor/parameter contract at startup and seal it in the YAML.
4. **R2-C: caps only.** Run a bounded cap sweep without the learned gate.
5. **R2-D: combine only passing components.** Do not infer interaction benefit
   from single-arm results.
6. **Later option: true reference-key mask.** Current zero sinks are an old
   defect, but they are shared with CL14 and changing them alters R at step zero.
   Test this only after R2-A/B, as its own checkpointed arm. Do not bundle it
   into the first R2 run.

## 5.3 Verification before a scientific launch

- Hydra composition and fail-closed config validator.
- Python compile/import check and shell syntax check.
- Old CL39 checkpoint load with no unexpected/missing keys when R2 is off.
- Exact 12-image parity against sealed CL39 when every R2 flag is off.
- A two-step train smoke proving finite loss and nonzero gradients in R
  projections on R-route batches.
- A route-coherence assertion showing every selected processor received the
  same route mode and strength for one forward and every unselected processor
  retained the ordinary route.
- Optimized-pipeline audit: no per-layer processor-dictionary lookup; disabled
  collectors skipped; active-gradient norms requested only if consumed; no
  full-activation telemetry unless declared.
- Exact startup `comet_experiment.json` record and immutable Comet key.
- Standard validation at step 0 and every 2,000 optimizer steps on all 96 fixed
  cells.

# 6. Experiment ladder and decision gates

| Arm | Single change vs CL39 | Hypothesis | Prediction if correct | Main risk | Promotion gate |
|---|---|---|---|---|---|
| **E0 lambda sweep** | evaluation-only `N+S lambda(R-N)`, `lambda={0,.25,.5,.75,1}` in the audited up0/up1 group on the exact CL39 checkpoint | harm grows after a safe residual scale | topology/quality breakpoints align with residual magnitude | no training adaptation | establish a monotonic or interpretable safe region; otherwise do not tune caps from lambda alone |
| **R2-A** | 12.5% coherent R-route training with warm ramp; inference unchanged | CL14-like ownership makes R self-sufficient | fewer raw-R landmark artifacts while actual CL39 ID/quality stays neutral or improves | early instability or loss conflict | at least 50% reduction in predeclared severe raw-R cells; actual fixed-96 ID no worse than `-0.005`; TOPIQ/MUSIQ/MANIQA and topology non-inferior |
| **R2-B** | zero-init bounded low/high gate in up0/up1 | entropy misses wrong-but-sharp matches and high-frequency risk | retains BA gain while reducing forced-C/high-band failures | gate learns native shortcut | beats matched CL39 or improves hard-case/quality without reducing correction-zero, global BA-off, and shuffle sensitivity |
| **R2-C** | RMS caps only | rare large bands seed duplicate facial structure | tail artifacts fall with minimal mean ID change | useful identity detail clipped | choose cap only if fixed-96 ID loss is `<0.005` and face quality/topology improve |
| **R2-D** | combine passing R2-A/B/C | self-sufficient R plus calibrated residual gives additive benefit | best actual output and cleaner raw route | interactions erase single-arm gains | must beat matched best single arm, not merely CL39 |
| **R2-E later** | true invalid-key exclusion | zero sinks dilute R and corrupt entropy semantics | clearer valid correspondence and less area dependence | step-zero distribution shift; over-strong R | separate matched run; reject on any repeat of CL41-like immediate ID loss |

Every trained arm should use the same 24k budget and fixed protocol. Report both
the selected checkpoint and the non-selected 24k endpoint. Paired intervals
are across the 96 fixed cells; a second training seed is required before a
population-level claim. `[method][decision]`

## Required causal evaluation matrix

For CL39 and every promoted R2 candidate, generate:

1. actual route;
2. CL39 `up_blocks.0/1` correction off with trained adapters retained;
3. all 70 BA processors off with trained PhotoMaker/generic adapters retained;
4. confidence/gate forced fully open;
5. raw R-on-face stress route, first group-scoped and then all-processor only
   if the group-scoped arm is stable;
6. correct PhotoMaker identity tokens with spatial reference latents shuffled;
7. low-only and high-only residual routes; and
8. the fixed lambda sweep.

Promotion requires actual R2 to beat or match CL39 on ID, text adherence, face
quality, masks, and visual topology **and** to remain causally sensitive to the
correct spatial reference. A clean image obtained by collapsing toward `N` is
not an R improvement. `[decision]`

# 7. Risks and what is not established

| Claim | Confidence | Basis | Main limitation |
|---|---|---|---|
| Raw CL39 R is not a self-sufficient face route | High | repeated visual structure errors; 95.24% face pixels changed; raw intervention definition | no standardized raw-route quality score |
| Raw R images overstate ordinary CL39 failure | High | actual equation and `3.80x` raw/actual residual magnitude difference | nonlinear denoising prevents a single exact “overdrive factor” in RGB space |
| Unattenuated R can harm combined denoising | High | same-checkpoint forced-C=1 paired loss | selected 16 cells, not all 96 |
| Actual CL39 uses useful explicit correction in up0/up1 | Moderate-high | same-checkpoint group-scoped correction-zero loss and structured R-N maps | selected 16; not a global BA-off; spatial-reference shuffle still missing |
| Entropy alone identifies bad correspondence | Low / unsupported | zero-sink semantics and weak intervention-size correlation | no ground-truth correspondence labels |
| CL14 proves hard inference replacement is better | Rejected | CL19 and CL39 fixed-panel gains over CL14 | single seed per trained arm |
| R-route dropout will improve CL39 | Moderate hypothesis | mechanism fit to CL14/CL39 difference; no extra-forward design | not yet trained |
| Bounded band gates/caps will improve final quality | Moderate-low hypothesis | CL44 high-band lead and measured tails | thresholds and interaction untested |

### What is not established

- A poor raw R image does not quantify how many actual CL39 artifacts came from
  R; the actual route is nonlinear and much weaker.
- The raw branch stress images do not have a full-96 standardized TOPIQ/MUSIQ/
  MANIQA/ID table in the retained local artifact set.
- The raw intervention covers the 36 confidence-enabled up0/up1 processors;
  standalone R quality in the other 34 BA processors has not been visualized.
- The 16-cell audit is deterministic and intentionally covers identities and
  hard cases; it is not a random population sample.
- The current confidence has not been calibrated against true facial
  correspondence or human artifact labels.
- No proposed R2 arm has been trained. The implementation plan is evidence-led,
  not a result.
- Generalization to unseen identities, other seeds, schedulers, or datasets is
  unknown.

# 8. Decision

Keep CL39-16k as the provisional best end-to-end checkpoint. Treat raw R
artifacts as a valid architecture diagnostic and a robustness/headroom problem,
not as a reason to discard CL39. `[decision]`

Build **R2-A first**: coherent, training-only R-route dropout with a warm ramp
and no inference change. It is the smallest change that directly tests the
CL14 lesson, adds no inference parameters, avoids an extra forward, and gives a
clear success criterion. Build the bounded band gate and RMS caps as independent
arms only after R2-A parity and stability are proven. `[decision]`

Do not restore CL14's hard query mask/hard inference replacement, do not bundle
true key masking or landmark warping into the first run, and do not promote an
R2 candidate unless group-scoped correction-zero, global BA-off, and
spatial-reference-shuffle tests show that its gain still comes from the
explicit reference lane. `[decision]`

# 9. Reproduction and source ledger

Render the two report-specific figures from repository-root-relative evidence:

```bash
AUDIT_CSV=artifacts/cl39_attention_24k_serv_a100/\
per_sample_summary.csv
BRANCH_CSV=artifacts/cl39_attention_24k_serv_a100_branch_faces/\
branch_face_metrics.csv
R2_FIGURES=analysis/assets/cl39_r2_architecture_20260825

python3 tools/analysis/render_cl39_r2_report_figures.py \
  --audit-csv "$AUDIT_CSV" \
  --branch-csv "$BRANCH_CSV" \
  --output-dir "$R2_FIGURES"
```

Primary evidence sources:

- [CL39 attention audit](2026-08-25_cl39_entropy_confidence_attention_audit.md)
- [completed CL38-CL45 results](2026-08-21_cl38_cl45_completed_results_and_photomaker_shortcut_audit.md)
- [CL14/CL19/CL23/CL27/CL39 code inventory](2026-08-22_serv_training_code_inventory_cl14_cl19_cl23_cl27_cl39.md)
- PM0/CL14/CL19/CL23/CL27/CL39 Comet report pages under `tools/comet/`
- clean-new branch E13-family architecture reference, dated 13 August, under
  `diffusion_template/docs/architecture/`
- clean-new branch June2-to-E13 architecture lineage, dated 18 August, in the
  same directory
- current processor directory: `src/model/photomaker_branched/`; file
  `attn_processor_cleanest.py`
- current configuration directory: `src/configs/`; CL14, CL19, CL23, CL27,
  and CL39 YAML leaves named in Sections 1-3.

Exact data directories: [attention audit records](../artifacts/cl39_attention_24k_serv_a100/),
[branch-face records](../artifacts/cl39_attention_24k_serv_a100_branch_faces/),
and [report figures](assets/cl39_r2_architecture_20260825/).

The clean-new mechanism description explains source semantics; historical run
results remain tied to their immutable sealed Comet records. No historical
checkpoint is represented as retrained on the current worktree. `[provenance]`
