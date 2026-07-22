# NN5a 4k independent results audit and NN6 architecture specification

**Date:** 22 July 2026  
**Repository:** `kolyangg/rsrch`  
**Branch:** `main_clean`  
**Audited head:** `6447cdd491a0c2c4aefd330d862f574aa6016743`  
**NN5a checkpoint:** `checkpoint-epoch2.pth` / 4,000 optimizer steps  
**Primary evaluation:** 96 targets, RealVisXL V4.0, PPR residual scale 1, five-condition reference/noise matrix  
**Repository mutation:** none

## 1. Executive decision

NN5a should remain stopped at 4k.

The run establishes three useful facts:

1. The protected packed-reference residual remains geometrically safe.
2. Counterfactual A/B supervision makes the branch materially more active than NN4 at normal inference scale.
3. The current noised spatial-reference representation still does not provide robust identity ownership.

The decisive failure is not merely that the aggregate directional score is statistically unresolved. Independent reanalysis of the 96 target cases shows:

- `44 / 96` targets reverse directional sign when only the reference-noise seed changes;
- only `25 / 96` targets are positive under **both** reference-noise seeds;
- after averaging N1 and N2 per target, mean similarity to B changes by `-0.000320`;
- after averaging N1 and N2 per target, mean similarity to A changes by `+0.000364`;
- only `4 / 8` identity groups have positive mean directional gain;
- several apparently positive identity groups are positive mainly because the output moves away from A, not because it approaches B.

NN5a is therefore not a weakly successful identity controller. It is an active, safe, reference-sensitive face editor whose final semantic direction remains dominated by target PhotoMaker identity and reference-noise nuisance.

The already running NN5b is the correct immediate comparison, but its current 50/50 pre-connector fusion has an important limitation: the clean identity candidate and noised spatial candidate share one connector, one spatial-null baseline, one gate, and one cap budget. If NN5b fails, that result will not prove that clean identity tokens are useless; they may have been diluted or clipped by the spatial lane.

The recommended implementation target is a **factorized identity/spatial processor** with a first approval run in **identity-only mode**. The identity lane must have its own null memory, connector, gate, cap, site policy, diagnostics, and checkpoint state. Spatial reference K/V should be disabled in the first run. This creates a clean causal test and makes reference-noise dependence an architectural invariant rather than another learned objective.

---

## 2. Evidence reviewed

### Result bundle

- `diffusion_template/Jul_new_exp/22Jul_NN5a_results/README.md`
- `NN5a_normal_validation_curve.csv`
- `NN5a_causal_summary.csv`
- `NN5a_training_window_summary.csv`
- `NN5a_tensor_stage_summary.csv`
- `causal_test/manifest.json`
- `causal_test/metrics_per_image.csv`
- `causal_test/paired_effects.csv`
- `causal_test/metrics_summary.csv`
- `causal_test/effect_decomposition.csv`
- `causal_test/identity_direction_per_image.csv`
- `causal_test/identity_direction_summary.csv`
- `causal_test/integrity_hashes.json`
- `causal_test/tensor_diagnostics.jsonl`
- `causal_test/neutral_reference_ca_summary.md`
- `causal_test/conclusion.md`
- `comet_training/metrics_summary.json`
- `comet_training/metrics_history.json`
- selected contact sheets under `causal_test/contact_sheets/`

### Reference analysis

- `diffusion_template/Jul_new_exp/2026-07-22_NN5a_4k_results_analysis_and_next_training_recommendation.md`

The numerical conclusions below were recomputed independently from the raw directional CSV and checked against the aggregate files. The six contact-sheet JPGs are committed, but the available GitHub connector exposed their binary content as base64 rather than a directly renderable image. Consequently, image-derived metrics and the repository's visual note are treated as supporting evidence; the architecture decision below is based primarily on the raw numeric and tensor evidence.

---

## 3. Test integrity

The causal test is valid for the RealVis deployment path.

The manifest confirms:

- RealVisXL V4.0 validation;
- residual scale 1;
- 96 targets;
- actual batch size 12;
- fixed target prompts and target seeds;
- matched A and cyclic wrong-identity B references;
- exactly paired A/B reference noise;
- neutral reference-half token and pooled-text conditioning;
- identity-token lane disabled, as required for NN5a;
- fixed RealVis-derived target bboxes;
- every integrity assertion passed;
- LPIPS available.

This removes the main implementation confounds found in NN2–NN4:

- the wrong row does not receive target-A diffusion reconstruction;
- reference noise is exactly paired across A and B;
- target PhotoMaker conditioning remains A in both rows;
- reference text does not leak target semantics;
- scale-zero output is ordinary PhotoMaker;
- the checkpoint and processor topology are restored strictly.

The stale “PPR 8k” heading in `causal_test/conclusion.md` is a report-template label. The manifest and checkpoint identify the test as 4k.

---

## 4. Independent quantitative analysis

## 4.1 Normal validation: branch activity increases while original-ID similarity falls

| Step | ID similarity | Text similarity |
|---:|---:|---:|
| 0 | 0.523129 | 26.365885 |
| 2k | 0.508999 | 26.438314 |
| 4k | 0.507018 | 26.465983 |

Changes from step 0 to 4k:

- ID similarity: `-0.016111`, or approximately `-3.08%` of the baseline score;
- text similarity: `+0.100098`.

Most of the ID decline occurs by 2k. The additional 2k steps increase visible branch influence without producing a recovery in identity similarity.

This does not by itself prove identity failure because normal validation tests matched A, not A→B causality. It does show that the extra face movement is not a Pareto improvement over ordinary PhotoMaker.

## 4.2 Reported causal result

| Noise | Mean directional gain | 95% bootstrap interval | Positive fraction |
|---|---:|---:|---:|
| N1 | +0.000842 | [-0.002448, +0.004132] | 53.13% |
| N2 | -0.002209 | [-0.004556, +0.000196] | 44.79% |
| Pooled rows | **-0.000684** | **[-0.002816, +0.001291]** | **48.96%** |

The opposing N1/N2 signs already fail the causal gate.

## 4.3 Target-clustered reanalysis

The repository's `all` row pools 192 N1/N2 observations. N1 and N2 from the same target are repeated measurements, so I also averaged the two noise rows per target and bootstrapped the 96 independent target means.

| Statistic | Value |
|---|---:|
| Mean target-averaged directional gain | -0.000684 |
| Target-clustered 95% bootstrap interval | [-0.002859, +0.001530] |
| Median target-averaged gain | -0.000252 |
| Target-positive fraction | 48.96% |

The clustered interval is slightly wider but does not change the conclusion.

Bootstrapping the eight identity-group means gives approximately:

```text
mean = -0.000684
95% identity-cluster interval = [-0.003025, +0.001422]
positive identity groups = 4 / 8
```

The result is chance-level at both the target and identity-group levels.

## 4.4 Reference-noise robustness is poor

Across the 96 targets:

| Cross-noise result | Count | Fraction |
|---|---:|---:|
| Direction positive under both N1 and N2 | 25 | 26.04% |
| Direction negative under both N1 and N2 | 27 | 28.13% |
| Direction changes sign between N1 and N2 | 44 | 45.83% |

Additional diagnostics:

```text
Pearson correlation between N1 and N2 directional gains:  0.202
Spearman correlation between N1 and N2 gains:               0.153
Mean absolute N1–N2 gain difference:                        0.01327
```

Nearly half of all targets reverse the conclusion when only the reference-noise realization changes. This is not compatible with a stable reference-identity controller.

## 4.5 The aggregate directional metric is not hiding real movement toward B

A relative directional score can look positive when an output merely moves away from A. Therefore the two components must be inspected separately.

After averaging N1 and N2 per target:

| Component | Mean | 95% target-cluster bootstrap interval | Positive fraction |
|---|---:|---:|---:|
| Similarity change toward B, R2 minus R1 | **-0.000320** | [-0.001982, +0.001264] | 58.33% |
| Similarity change toward A, R2 minus R1 | **+0.000364** | [-0.001360, +0.001911] | 53.13% |
| Directional gain = B change − A change | **-0.000684** | [-0.002859, +0.001530] | 48.96% |

On average, changing the reference from A to B makes the result very slightly **less B-like and more A-like**. Both changes are tiny and statistically unresolved, but their signs are the opposite of the intended behavior.

Robust movement toward B is also uncommon:

```text
B similarity increases under both N1 and N2: 32 / 96
B similarity decreases under both N1 and N2: 21 / 96
B-similarity sign changes across noise:       43 / 96
```

## 4.6 Identity-group decomposition

Values below average N1 and N2 per target.

| Identity | Targets | Mean direction | Median direction | Positive fraction | Mean B change | Mean A change |
|---|---:|---:|---:|---:|---:|---:|
| eddie | 12 | +0.001280 | -0.000857 | 50.0% | +0.000961 | -0.000319 |
| elon | 12 | +0.001997 | +0.000935 | 58.3% | +0.001370 | -0.000628 |
| jennie | 12 | -0.004583 | -0.004511 | 25.0% | -0.001862 | +0.002721 |
| jensen | 12 | +0.002415 | -0.000012 | 50.0% | -0.000078 | -0.002494 |
| jisoo | 12 | -0.002282 | -0.000754 | 41.7% | +0.000397 | +0.002679 |
| keanu | 12 | +0.002721 | +0.002487 | 58.3% | +0.000007 | -0.002714 |
| lex | 12 | -0.000579 | +0.001481 | 58.3% | -0.000618 | -0.000039 |
| marion | 12 | -0.006440 | -0.000402 | 50.0% | -0.002737 | +0.003703 |

The decomposition is important:

- Jensen and Keanu have positive relative direction mainly because A similarity falls; B similarity is approximately flat.
- Jennie and Marion move in the wrong direction in both components.
- No identity group reaches the predeclared 60% positive threshold.
- The overall result is not a single outlier: 5% and 10% trimmed means remain negative.

## 4.7 The branch is content-sensitive internally, but this advantage is lost by target epsilon

| Captured stage | Reference-content difference | Reference-noise difference | Content/noise ratio |
|---|---:|---:|---:|
| Reference hidden | 1.188410 | 0.374686 | 3.172× |
| Reference candidate | 1.051110 | 0.313017 | 3.358× |
| Connector down | 0.710597 | 0.194418 | 3.655× |
| Raw delta | 0.319127 | 0.066485 | 4.800× |
| Bounded delta | 0.327277 | 0.065533 | 4.994× |
| Applied delta | 0.202167 | 0.046477 | 4.350× |
| Target epsilon before anchor | 0.091152 | 0.089515 | **1.018×** |
| Target epsilon after anchor | 0.087916 | 0.086684 | **1.014×** |

These are deterministic-sketch relative differences rather than full-tensor norms, but the trend is clear.

The packed reference candidate is not blind. The connector actually increases content/noise separation. The identity-specific advantage then disappears between the processor-local applied residual and the final target epsilon.

This supports two nonexclusive causes:

1. **Semantic contamination:** the spatial candidate distinguishes images, but the distinguishing information is pose, expression, illumination, crop, texture, occlusion, or reference-noise structure rather than a stable identity direction.
2. **Trajectory/layer washout:** the residual is injected only at `up_blocks.0.attn1`; later U-Net blocks can rewrite or dilute the identity-specific component before epsilon is emitted.

Increasing scale, cap, or spatial sites would not distinguish these causes and is therefore a poor next experiment.

## 4.8 Training objective strength is not the primary problem

Window means:

| Metric | 0–2k | 2k–4k |
|---|---:|---:|
| Total loss | 0.227344 | 0.231520 |
| Counterfactual absolute-ID loss | 0.957938 | 0.938537 |
| Counterfactual directional loss | 0.575047 | 0.541007 |
| Training directional gain, B minus A | -0.703292 | -0.677467 |
| Similarity to A | 0.745354 | 0.738930 |
| Similarity to B | 0.042062 | 0.061463 |
| Counterfactual active fraction | 0.4195 | 0.4425 |
| Cap excess | 0.018285 | 0.054809 |
| Null residual | 1.29e-5 | 7.10e-6 |
| Reference-LoRA norm | 4.464 | 9.935 |

Using configured weights, the two counterfactual identity terms contribute approximately:

```text
0.05 * L_abs + 0.10 * L_direction
≈ 0.0469 + 0.0541
≈ 0.1010
```

during the 2k–4k window, or roughly 44% of the mean total loss. The objective is not numerically negligible.

The problem is that increased optimization pressure mainly produces:

- growing reference projection norms;
- growing raw residuals;
- increasing cap pressure;
- only a small rise in B similarity;
- no positive inference causality.

The current Comet summary also shows connector-up gradients dominating the trainable gradient budget. In the 2k–4k window:

```text
connector-up gradient norm / total gradient norm ≈ 0.983
reference-K gradient norm / total norm            ≈ 0.020
reference-V gradient norm / total norm            ≈ 0.096
```

Group norms are not additive, but the imbalance shows that learning is concentrated in output remapping rather than in constructing a cleaner reference retrieval representation.

The counterfactual path is active only about 42–44% of sampled timesteps because training samples uniformly from the BA inference region while decoded identity losses are limited to `t <= 300`. This is a secondary efficiency issue. It does not explain why reference-noise sign reversals remain so frequent after the branch has opened.

## 4.9 Safety remains the main positive result

Image-derived metrics show:

- face detection rate `1.0`;
- reference and noise perturbations remain localized;
- seam and landmark pairwise effects are small;
- text effects remain modest;
- exact ordinary PhotoMaker output remains protected outside the core.

The output anchor and bounded core residual should be retained.

---

## 5. Root-cause ranking

## 5.1 Primary: noised spatial memory is not an identity representation

The spatial branch sees reference-specific content, but that content is not identity-pure. Counterfactual supervision improves the relative direction from clearly wrong toward chance, yet the branch still edits gaze, expression, smoothness, and local rendering in a way that is almost as sensitive to reference noise as to reference identity at the final image.

This is a representation problem, not simply a weak branch.

## 5.2 Primary: the useful signal is injected too early and then washed out

All selected NN5 sites are in `up_blocks.0.attn1`. Processor-local content/noise separation is strong; final epsilon separation is approximately one.

A clean identity residual should be separately routable to later up-block attention sites rather than being permanently tied to the spatial lane's site policy.

## 5.3 Important NN5b limitation: pre-connector fusion is not truly independent

Current NN5b computes:

```text
C_fused = 0.5 * C_spatial + 0.5 * C_identity
delta   = D(C_fused - C_spatial_null)
```

Consequences:

- identity and spatial candidates are not normalized before blending;
- a fixed numeric 50/50 mixture is not guaranteed to be a semantic 50/50 mixture;
- the identity candidate has no identity-specific null passed through the same identity K/V route;
- both lanes share one connector;
- both lanes share one scalar gate;
- both lanes share one RMS cap budget;
- spatial nuisance can consume the cap and suppress the clean identity contribution;
- counterfactual gradients cannot be attributed to a specific lane.

NN5b is still worth running because it may succeed despite this. A failure, however, should lead to factorization—not to the conclusion that PMv2 identity tokens are ineffective.

## 5.4 Secondary: target PhotoMaker A is a strong competing identity source

Both counterfactual rows retain full PhotoMaker identity A. This is the correct deployment stress test, but training metrics show that the B row remains far closer to A than B.

Do not introduce PM attenuation in the first factorized identity run. First determine whether a clean, dedicated identity lane can move the output under the real full-A condition. A later curriculum can ramp A conditioning only if the dedicated lane receives healthy gradients but remains completely unable to move.

## 5.5 Secondary: training and evaluation recognizers differ

Training counterfactual supervision uses frozen VGGFace2 InceptionResnetV1. The principal validation identity metric uses the repository's InsightFace-based embedding path.

This mismatch can reduce correlation between training improvement and validation improvement. It is worth addressing in a later loss ablation, but it is not the first architectural change: NN5a training direction is still strongly A-dominant even under its own FaceNet objective.

## 5.6 Unresolved confound: SDXL training versus RealVis validation

The operational result on RealVis is valid: NN5a does not solve the intended deployment problem.

For architectural diagnosis, however, the branch is trained on SDXL-base hidden distributions and evaluated after transfer to RealVis. The PPR reference projections, connectors, and gates are trained in one U-Net feature space and applied in another.

Before treating trajectory washout as entirely backbone-independent, run a same-SDXL 4k causal screen using **SDXL-specific fixed PM0 bboxes**. Do not reuse RealVis bboxes.

Decision:

- if same-SDXL is also near chance, proceed with NN6 factorization;
- if same-SDXL becomes clearly positive while RealVis remains near chance, perform a RealVis-native training ablation before changing architecture;
- do not change backbone and architecture in the same first run.

---

## 6. Decision for NN5b

Evaluate NN5b at 2k and 4k; do not let the nominal 30k job budget determine the scientific budget.

### Required NN5b metrics

In addition to the existing causal outputs, inspect the newly separated tensor diagnostics:

```text
spatial_reference_candidate
identity_candidate
reference_candidate              # their fused result
connector_down
raw_delta
bounded_delta
applied_delta
target_epsilon_pre_anchor
target_epsilon_post_anchor
```

The important questions are:

1. Does R1→R2 strongly change `identity_candidate`?
2. Is `identity_candidate` reference-noise invariant?
3. Is its RMS much smaller or larger than the spatial candidate before the 50/50 blend?
4. Does its swap direction survive the shared connector?
5. Is the fused residual frequently capped?
6. Does final B similarity actually rise?
7. Are N1 and N2 directions consistent?

### NN5b continue gate

Continue beyond 4k only when all are true:

- target-averaged directional mean is positive;
- target-clustered lower confidence bound is above zero;
- mean B-similarity change is positive;
- N1 and N2 means are both positive;
- at least 60% of target-averaged gains are positive;
- at least 50% of targets are positive under both N1 and N2;
- matched A remains near PM0;
- cap pressure does not rise while semantic direction stalls;
- face detection and geometry safety remain near NN5a.

If NN5b is near zero or negative at 4k, stop it regardless of the 30k launcher budget.

---

## 7. Recommended next architecture: NN6 factorized identity residual

## 7.1 Design goal

Create a branch in which:

- clean reference identity has a dedicated causal path;
- spatial reference content cannot dilute or consume the identity path;
- identity and spatial lanes can use different U-Net sites;
- reference noise cannot affect the identity-only target prediction;
- ordinary PhotoMaker remains the exact protected baseline;
- each lane is independently measurable and ablatable.

Implement one modular processor update, then run the first configuration in identity-only mode.

## 7.2 Core equations

At selected target self-attention site \(l\):

```text
A_target = Attention(Q_target, K_target, V_target)
```

Clean PMv2 identity tokens:

```text
T_id(ref) ∈ R[B, 2, 2048]
T_id(null) ∈ R[1, 2, 2048]        # learned identity-null tokens
```

Identity candidate and matched null:

```text
C_id     = Attention(Q_target, K_id(T_id(ref)),  V_id(T_id(ref)))
C_idnull = Attention(Q_target, K_id(T_id(null)), V_id(T_id(null)))
```

Dedicated identity residual:

```text
R_id_raw     = D_id(C_id - C_idnull)
R_id_bounded = RMSCap(R_id_raw, base=A_target, ratio=cap_id)
g_id         = gate_max_id * sigmoid(gate_logit_id)
```

NN6a output:

```text
A_out = A_target + M_core * g_id * R_id_bounded
```

Everything outside `M_core` remains ordinary target attention, and final epsilon remains anchored to ordinary PhotoMaker outside the same core.

### Required identity-null property

`T_id(null)` must pass through the **same** `identity_to_k` and `identity_to_v` projections as real identity tokens. Do not subtract the existing spatial hidden-state null from an identity candidate.

## 7.3 NN6a first run: identity-only, same sites

Run the cleanest control first:

```text
identity lane: enabled
identity site policy: up_blocks0_attn1
spatial lane: disabled
branched CA: disabled
pose adaptation: disabled
CA face mixing: disabled
output anchor: base_outside_core
```

This isolates the effect of:

- removing the 50/50 fusion;
- providing an identity-specific null;
- providing a dedicated connector/gate/cap;
- removing noised spatial K/V from the target path.

Keep the same `up_blocks.0` site scope initially so that NN5b versus NN6a differs only in lane factorization and spatial removal.

### Hard causal invariant

In identity-only mode, changing reference noise while keeping identity tokens fixed must not change:

```text
identity_candidate
identity connector input
identity raw/bounded/applied delta
target epsilon
final image
```

Within deterministic BF16 tolerance, `R1N1 == R1N2` and `R2N1 == R2N2`.

If this invariant fails, the implementation is still leaking the reference half into the target path.

## 7.4 NN6b second run: late identity injection

Run only if NN6a produces a reference-sensitive identity candidate but the final epsilon or image remains weak.

Add a separate identity site policy:

```text
spatial site policy: up_blocks0_attn1
identity site policy: up_blocks1_attn1
```

For a first late-site experiment, keep the spatial lane off:

```text
identity lane: up_blocks.1
spatial lane: disabled
```

Rationale:

- NN5a has strong processor-local content sensitivity but loses it by final epsilon;
- identity tokens are semantic and safer than spatial K/V at a later resolution;
- later injection reduces the opportunity for downstream blocks to wash out identity;
- `up_blocks.0` remains available later for low-authority spatial shape/detail.

Do not activate all up-block sites in the first late-site experiment. Use an explicit block policy so authority remains interpretable.

## 7.5 NN6c optional final form: factorized dual lane

Enable only after an identity-only run passes the causal gate.

Spatial candidate:

```text
C_spatial     = Attention(Q_target, K_spatial(H_ref_roi), V_spatial(H_ref_roi))
C_spatialnull = Attention(Q_target, K_spatial(H_spatial_null), V_spatial(H_spatial_null))

R_spatial_raw     = D_spatial(C_spatial - C_spatialnull)
R_spatial_bounded = RMSCap(R_spatial_raw, ratio=cap_spatial)
```

Combined residual:

```text
R_total_pre = g_id * R_id_bounded + g_spatial * R_spatial_bounded
R_total     = RMSCap(R_total_pre, ratio=cap_total)

A_out = A_target + M_core * R_total
```

Recommended initial authority:

```yaml
identity cap:       0.12
spatial cap:        0.03
total cap:          0.15
identity gate max:  0.50
spatial gate max:   0.15
```

These are starting values, not universal constants. The important property is independent per-lane authority plus a final total cap.

### Training order

1. Train identity lane alone.
2. Freeze the accepted identity lane.
3. Enable the spatial lane at low authority.
4. Train the spatial lane on matched reconstruction/local-detail objectives.
5. Do not allow counterfactual identity loss to be satisfied through the spatial lane.
6. Add same-reference/two-noise consistency to the spatial applied residual.

Spatial noise consistency:

```text
L_spatial_noise =
    mean_l RMS(
        M_core * (
            R_spatial_applied(ref, N1)
          - R_spatial_applied(ref, N2)
        )
    )^2
```

Do not add this loss to NN6a; identity-only mode should be noise-independent by construction.

---

## 8. Backward-compatible implementation plan

## 8.1 Configuration surface

Extend the existing packed-residual processor rather than replacing all NN5 code.

Recommended fields:

```python
ba_identity_fusion_mode: str = "blend"
# blend | identity_only | factorized_dual

ba_identity_site_policy: str = "inherit"
# inherit | up_blocks0_attn1 | up_blocks1_attn1 | up_blocks_attn1

ba_spatial_site_policy: str = "inherit"

ba_spatial_lane_enabled: bool = True

ba_identity_null_tokens: int = 2
ba_identity_connector_rank: int = 16
ba_identity_gate_max: float = 0.5
ba_identity_gate_init_logit: float = 0.0
ba_identity_delta_rms_cap: float = 0.15

ba_spatial_gate_max: float = 0.15
ba_spatial_delta_rms_cap: float = 0.03
ba_total_delta_rms_cap: float = 0.15
```

Backward compatibility:

```text
blend + spatial enabled + inherited site policy
```

must reproduce NN5b exactly.

## 8.2 `packed_residual_attn_processor.py`

Add modules:

```python
self.identity_null_memory     # [T_id, 2048]
self.identity_connector_down
self.identity_connector_up    # zero initialized
self.identity_gate_logit
```

Keep existing modules as the spatial lane:

```python
self.null_memory
self.connector_down
self.connector_up
self.gate_logit
self.ref_to_k
self.ref_to_v
```

Refactor forward into helpers:

```python
_compute_spatial_candidate(...)
_compute_identity_candidate(...)
_compute_spatial_residual(...)
_compute_identity_residual(...)
_combine_factorized_residuals(...)
```

Important details:

- compute spatial candidate only when the processor's spatial lane is enabled;
- do not calculate and then multiply an unwanted lane by zero;
- identity-only target output must not depend on `reference_hidden`;
- use the same target query for real and null identity candidates;
- use bias-free connectors;
- zero-initialize every connector-up;
- cap each lane before summation;
- optionally cap the sum;
- keep final core multiplication and target residual behavior unchanged.

## 8.3 `branched_runtime.py`

Extend processor-site selection:

```python
identity_names = select_names(ba_identity_site_policy)
spatial_names  = select_names(ba_spatial_site_policy)
patched_names  = identity_names | spatial_names
```

Instantiate one processor on the union and pass booleans:

```python
enable_identity = name in identity_names
enable_spatial  = name in spatial_names
```

Add selector support for:

```text
up_blocks1_attn1
```

Propagate all new architecture fields.

NN6a still uses the existing doubled forward for minimal implementation risk. Once it passes, a later optimization may remove the unused reference half in identity-only mode.

## 8.4 `lora2.py`

Add and validate new fields.

Update `_ba_architecture_state()` so strict checkpoints record:

- fusion mode;
- per-lane site policies;
- enabled lanes;
- identity-null token count;
- connector ranks;
- gate/cap values.

Update optimizer grouping:

```text
ba_ppr_identity_k
ba_ppr_identity_v
ba_ppr_identity_null
ba_ppr_identity_connector_down
ba_ppr_identity_connector_up
ba_ppr_identity_gate
```

For NN6a, the optimizer must contain no spatial/ref-KV group.

Keep NN5 counterfactual paired forward and decoded losses unchanged for the first run. This isolates architecture.

Add counters:

```text
ba_cf_active_updates
ba_cf_inactive_updates
```

The existing fractional metric can hide how many semantic updates a checkpoint has actually received.

## 8.5 `lora2_helpers.py`

Extend strict trainability checks for each fusion mode.

For `identity_only`, require exactly:

```text
identity_to_k.{0,2}.weight
identity_to_v.{0,2}.weight
identity_null_memory
identity_connector_down.weight
identity_connector_up.weight
identity_gate_logit
```

Reject trainable:

```text
ref_to_k
ref_to_v
spatial null
spatial connector
spatial gate
branched CA
```

## 8.6 `br_pipeline_helpers.py`

Continue deriving identity tokens from the spatial reference image, including R2 diagnostic swaps.

In identity-only mode:

- identity token extraction remains mandatory;
- invalid recognition embeddings remain fatal;
- reference latent preparation can remain temporarily for doubled-batch compatibility;
- target routing must not consume it.

Persist identity-token hashes in every diagnostic.

## 8.7 `ppr_reference_noise.py`

Add per-lane tensor fields:

```text
identity_candidate
identity_null_candidate
identity_connector_input
identity_raw_delta
identity_bounded_delta
identity_applied_delta

spatial_candidate
spatial_null_candidate
spatial_connector_input
spatial_raw_delta
spatial_bounded_delta
spatial_applied_delta

combined_applied_delta
```

In identity-only mode, assert:

```text
R1N1 identity tensors == R1N2 identity tensors
R2N1 identity tensors == R2N2 identity tensors
R1N1 target epsilon   == R1N2 target epsilon
R2N1 target epsilon   == R2N2 target epsilon
```

Record whether equality is exact or within an explicitly configured tolerance.

Update the summary to bootstrap targets, not the 192 repeated noise rows, as the primary `all` confidence interval.

Also report:

```text
both-noise-positive fraction
noise-sign-flip fraction
mean B-similarity change
mean A-similarity change
per-identity means
```

These statistics exposed the NN5a failure more clearly than pooled direction alone.

## 8.8 Tests

Add tests before launch.

### Exact parity

- zero-initialized identity connector reproduces ordinary attention exactly;
- scale zero reproduces ordinary PhotoMaker;
- outside-core output remains exact PhotoMaker.

### Identity route

- changing identity tokens changes output after opening identity connector;
- changing spatial reference latent does not change identity-only target output;
- changing reference noise does not change identity-only target output;
- identity-null and real tokens traverse the same identity K/V modules;
- invalid or zero identity embeddings fail closed.

### Gradients

- counterfactual B loss reaches identity connector-up;
- after connector-up opens, gradients reach connector-down and identity K/V;
- no gradients reach spatial K/V or spatial connector in identity-only mode;
- base U-Net and PhotoMaker remain frozen.

### Checkpointing

- strict manifest includes every new field;
- blend-mode NN5b checkpoint restores unchanged;
- identity-only checkpoint rejects blend-mode restore;
- optimizer groups are complete and nonoverlapping.

### CFG and batching

- CFG unconditional/conditional copies receive identical identity tokens;
- target and reference batch mapping remains correct;
- batch 1 training and batch 12 validation both pass.

---

## 9. NN6a configuration

Create:

```text
src/configs/one_id_ba_NN6a_factorized_identity_only_up0.yaml
```

Suggested configuration:

```yaml
defaults:
  - one_id_ba_NN5a_counterfactual_directional_ppr
  - _self_

model:
  ba_identity_token_lane: true
  ba_identity_fusion_mode: identity_only

  ba_identity_site_policy: up_blocks0_attn1
  ba_spatial_site_policy: up_blocks0_attn1
  ba_spatial_lane_enabled: false

  ba_identity_null_tokens: 2
  ba_identity_token_dim: 2048
  ba_identity_token_rank: 32
  ba_identity_connector_rank: 16

  ba_identity_gate_max: 0.50
  ba_identity_gate_init_logit: 0.0
  ba_identity_delta_rms_cap: 0.15
  ba_total_delta_rms_cap: 0.15

  # Keep the NN5 causal objective unchanged for clean attribution.
  ba_counterfactual_enabled: true
  ba_counterfactual_max_timestep: 300
  ba_counterfactual_abs_id_weight: 0.05
  ba_counterfactual_direction_weight: 0.10
  ba_counterfactual_direction_margin: 0.03
  ba_counterfactual_ring_weight: 0.05

  use_id_loss: true
  id_loss_weight: 0.025
  id_loss_max_timestep: 300
  id_loss_identity_source: reference

  # Do not add a PM attenuation curriculum in this first run.
  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0
```

Keep:

```text
up_blocks.0 only
no branched CA
pose adaptation off
CA face mixing off
neutral reference text
paired CFG reference noise
base_outside_core
RealVis validation with RealVis PM0 bboxes
```

Initial budget:

```text
2k checkpoint screen
4k maximum approval-stage budget
no automatic continuation to 20k/30k
```

---

## 10. Evaluation protocol

## 10.1 Before NN6 architecture training

Run the NN5a 4k checkpoint once on the SDXL training base using an SDXL-specific PM0 bbox file.

A 24-case stratified screen can identify a large same-base recovery; use the full 96 if the result is promising.

This determines whether RealVis-native training deserves priority.

## 10.2 NN6a approval

RealVis scale-1 matrix at 2k and 4k.

Primary criteria:

1. target-clustered mean directional gain > 0;
2. target-clustered lower 95% bound > 0 at 4k;
3. mean B-similarity change > 0;
4. both N1 and N2 means positive;
5. noise-sign-flip fraction below 10% in identity-only mode;
6. at least 60% target-averaged positive;
7. at least 50% positive under both N1 and N2;
8. R1 original-ID preservation within approximately `-0.005` of PM0;
9. 100% face detection;
10. no systematic gaze, expression, occlusion, or boundary artifacts;
11. cap fraction not broadly saturated.

Identity-only reference-noise effect should be effectively zero. If it is not, treat that as an implementation failure before interpreting identity metrics.

## 10.3 NN6b late-site gate

Run only if:

- identity candidate changes strongly under R1→R2;
- it is noise-invariant;
- up0 identity-only final direction remains weak.

Compare:

```text
NN6a identity-only up0
NN6b identity-only up1
```

with identical losses, seeds, data, cap, and checkpoint budget.

## 10.4 RealVis-native training decision

Train on RealVis only when:

```text
same-SDXL causal result is clearly better than RealVis transfer result
```

Then run the **same architecture** on RealVis without other changes.

Do not combine:

```text
new factorized architecture
+ RealVis training
+ new loss backbone
+ new timestep sampler
```

in one experiment.

---

## 11. Secondary follow-ups, not part of NN6a

### Low-noise sampling

Current counterfactual supervision is active on approximately 43% of steps. After architecture isolation, add an explicit mixture sampler rather than increasing loss weights:

```text
p_low = 0.70: sample t uniformly from [0, 300]
1-p_low:       sample t uniformly from [301, BA_max]
```

This increases semantic updates without applying decoded identity loss at unstable high-noise timesteps.

### Identity-loss backbone alignment

Add a differentiable ArcFace/AdaFace-family loss or an ensemble with FaceNet only after the clean lane is validated. Continue reporting InsightFace causal evaluation independently.

### Single-stream optimization

If identity-only succeeds, remove the unused reference U-Net half for identity-only inference/training. The clean token lane does not require a noised reference latent stream.

This is a compute optimization, not part of the first causal test.

### Semantic spatial details

After identity-only success, add semantic facial-part or canonical patch tokens instead of restoring the full noised ROI at high authority.

---

## 12. Changes not recommended

- Do not continue NN5a unchanged beyond 4k.
- Do not approve NN5b because its loss falls or faces change more.
- Do not allow the 30k launcher budget to bypass 2k/4k causal gates.
- Do not raise PPR scale, gate maximum, RMS cap, or spatial site count to rescue direction.
- Do not re-enable branched cross-attention.
- Do not re-enable pose adaptation or CA face mixing.
- Do not train the counterfactual B row against target-A diffusion reconstruction.
- Do not interpret movement away from A as movement toward B.
- Do not pool N1/N2 rows as independent evidence without a target-clustered summary.
- Do not reuse RealVis target bboxes for an SDXL same-base diagnostic.
- Do not change backbone and architecture simultaneously.
- Do not give the identity candidate a spatial-null baseline.

---

## 13. Agent handoff checklist

An implementation agent should deliver:

1. Backward-compatible `blend`, `identity_only`, and `factorized_dual` modes.
2. Identity-specific learned null tokens through identity K/V.
3. Dedicated identity connector, gate, and cap.
4. Independent identity and spatial site policies.
5. Identity-only trainability manifest and optimizer groups.
6. Strict checkpoint architecture fields.
7. Per-lane diagnostics and target-clustered causal summaries.
8. Hard reference-noise independence tests for identity-only mode.
9. NN6a config and one-GPU launcher.
10. 2k/4k RealVis scale-1 checkpoint diagnostic wrapper.
11. No Git changes outside the listed files and no silent fallback to NN5b blend behavior.

Expected files:

```text
src/model/photomaker_branched/packed_residual_attn_processor.py
src/model/photomaker_branched/branched_runtime.py
src/model/photomaker_branched/lora2.py
src/model/photomaker_branched/lora2_helpers.py
src/pipelines/br_pipeline_helpers.py
src/trainer/ppr_reference_noise.py
src/trainer/sdxl_trainers.py
tests/test_packed_residual_processor.py
tests/test_nn5_components.py
src/configs/one_id_ba_NN6a_factorized_identity_only_up0.yaml
jul_serv_runs/start_ba_NN6a_factorized_identity_only_up0_1gpu.sh
jul_serv_runs/start_ba_NN6a_checkpoint_reference_vs_noise_1gpu.sh
```

---

## 14. Final recommendation

Treat NN5a as a completed negative result:

> Counterfactual supervision can open the protected PPR branch and reduce a clearly wrong identity direction toward chance, but it cannot make the current noised spatial candidate a reliable identity controller.

Let NN5b answer whether clean PMv2 tokens help despite fused routing.

In parallel, implement the factorized processor but do not start a long run automatically. If NN5b does not pass its 2k/4k causal gate, launch NN6a in identity-only mode. Its central scientific question is precise:

> Can clean reference identity tokens, with their own matched null, connector, gate, cap, and unchanged protected PhotoMaker baseline, produce reference-causal identity without any noised spatial-reference contribution?

Only after that answer is positive should spatial detail be reintroduced as a separately bounded, separately trained, noise-consistent lane.
