# CL39-R2: three independent high-potential experiments

**Date:** 26 August 2026  
**Status:** implementation plan only; no R2 code, configuration, or training run
has been created or launched  
**Source branch:** `clean_full`  
**Direct baseline:** `CL39_cosmic_null_key_confidence_router_24k`  
**Baseline run:** `CL39_cosmic_null_key_confidence_router_24k_full96_r4`  
**Baseline Comet key:** `b1ca0b3da679401c85b991f1bbdf0b2a`  
**Primary metric:** fixed-96 subject-v2 identity similarity, matched to the
intended generated face. Face quality, topology, prompt adherence, object
retention, and causal BA sensitivity are mandatory co-gates. RGB difference is
an intervention-strength diagnostic, not a quality metric.

## Executive decision

Build three **independent direct children of CL39**, in this order:

1. **CL39-R2-A — coherent reference-face ownership exposure.** On a
   deterministic 12.5% of training steps, gradually require the 36 audited
   `up_blocks.0/1` processors to let raw `R` own the routed face. Validation and
   inference remain exact CL39. This is the highest-priority experiment because
   it directly tests whether CL39's fragile `R` can learn coherent facial
   topology when the diffusion loss occasionally depends on it. `[decision]`
2. **CL39-R2-B — bounded learned low/high reliability.** Keep CL39 entropy
   confidence as the baseline and add a tiny, zero-initialized, per-processor
   MLP that can adjust low- and high-band confidence by at most `0.20`. This
   tests whether entropy alone misses sharp but geometrically wrong reference
   matches. `[hypothesis]`
3. **CL39-R2-C — fixed per-band RMS tail caps.** Cap only unusually large
   low/high residuals using audit-derived `k_L=0.90`, `k_H=0.45`. This tests
   whether rare residual tails seed duplicate or warped facial structure.
   `[hypothesis]`

Each arm changes one scientific mechanism only. R2-B does **not** include R2-A;
R2-C includes neither R2-A nor R2-B. A later composition run is allowed only
after individual arms pass their gates.

Do not create a literal `q_face=q*M` training arm. In the current geometry,
the CL39 soft face router has `S>0 => M=1`; multiplying the query by the binary
face mask therefore changes mainly the transition boundary, not the face core.
The useful CL14 lesson is that its reference lane is trained to **own** the
face. R2-A tests that obligation directly without adding a syntactically
different but mostly equivalent query path. `[code][decision]`

## 1. Evidence motivating the ladder

The completed Serv lineage audit used 18 arms and 1,728 fixed-96 images. On the
predeclared 16-cell diagnostic subset: `[measured]`

| CL39 arm | Subject-v2 ID | Meaning |
|---|---:|---|
| actual CL39 | **`0.55754`** | successful operating point |
| native `N` only | `0.52184` | PhotoMaker/native target self-attention anchor |
| raw `R` on routed face | `0.42241` | standalone reference-route stress |
| low only | `0.52793` | low-band correction only |
| high only | `0.54047` | high-band correction only |
| confidence forced to `1` | `0.49984` | same bands without entropy attenuation |

Actual CL39 beat native by `+0.03570`, raw R by `+0.13513` on `16/16` cells,
and forced-`C=1` by `+0.05770`. Raw R had major face-part/object-fusion failures
in about `8/16` inspected cells, whereas CL19's trained reference-owned route
had `0/16` such failures on the same visual audit. `[measured][visual]`

The frequency audit recorded 20,160 layer calls and found these ratios before
CL39 confidence: `[measured]`

| Face RMS ratio | p50 | p95 | maximum |
|---|---:|---:|---:|
| low / native | `0.767` | `0.910` | `1.424` |
| high / native | `0.301` | `0.425` | `0.631` |
| raw `(R-N)` / native | `0.880` | `1.021` | `1.499` |
| actual routed correction / native | `0.200` | `0.415` | `0.691` |

These measurements support three separate questions:

- Can training make `R` coherent? R2-A.
- Can confidence distinguish reliable low/high content better than entropy
  alone? R2-B.
- Are rare residual magnitudes, rather than average routing, responsible for a
  useful part of the artifact tail? R2-C.

They do **not** establish that any proposal will beat CL39. `[limitation]`

## 2. Baseline equation and frozen contract

At an audited processor, define:

- `N`: target-query/target-KV native message after the shared output projection;
- `R`: target-query/reference-face-KV message after the same output projection;
- `S`: existing two-cell soft target-face router;
- `D=R-N`, with Gaussian split `D=L+H`;
- `g_L(p)` and `g_H(p)`: current CL23 denoising-progress gains;
- `C_0(q)`: current CL39 detached entropy confidence in `[0.25,1]`.

CL39 produces:

```text
Y_CL39 = N + S * C_0 * (g_L * L + g_H * H)
```

All three runs must preserve the following unless the row is the arm's one
declared intervention:

| Contract | Required value |
|---|---|
| Parent YAML | `CL39_cosmic_null_key_confidence_router_24k` |
| Dataset | `cosmic_large_adapted`, unchanged ordering/augmentation |
| Training budget | 24,000 optimizer steps; `epoch_len=2000`, `n_epochs=12` |
| Validation | step 0 and every 2,000 steps; fixed `manual_val` 96; one image/item |
| Inference | RealVisXL baseline, DDIM50, CFG 5, sealed prompts/references/boxes/seeds |
| BA geometry | `hard_replace_v1`, target Q, explicit reference-only K/V |
| Forbidden pose ablation | `model.pose_adapt_ratio=0`, `pipeline.pose_adapt_ratio=0` |
| Forbidden CA mixing | `model.ca_mixing_for_face=false`, `pipeline.ca_mixing_for_face=false` |
| CL27 objective | unchanged frequency-surface loss and semantic-occlusion policy |
| Existing CL39 router | enabled in `up_blocks.0/1` with unchanged thresholds/floor |
| Loss selection | `loss_kind=masked_alternating`; do not rely on `loss_function` alone |
| Optimized pipeline | cached processor maps, disabled collectors skipped, requested-only grad norms, full-activation telemetry off |
| Historical behavior | every new flag defaults off; old CL39 loads and runs identically |

All arms start from the same frozen initialization and training seed as a
matched CL39 reproduction. Do not fine-tune the completed CL39 checkpoint and
compare it with from-scratch siblings.

## 3. Experiment A — coherent reference-face ownership exposure

### 3.1 Single scientific change

On ordinary training steps, use exact CL39. On a stateless, deterministic
`12.5%` of optimizer steps, interpolate the selected processors toward raw
reference ownership:

```text
Y_R = N + S * (R - N)
alpha(step) = selected(step) * clip((step - 2000) / 4000, 0, 1)
Y_train = (1 - alpha) * Y_CL39 + alpha * Y_R
```

Only processors whose names begin with `up_blocks.0.` or `up_blocks.1.` are
eligible. The other 34 hardcase processors retain ordinary CL39 on every step.
The normal diffusion loss supervises `Y_train`; the existing CL27 auxiliary
continues to read the normal low/high components so its definition does not
change. There is no second U-Net forward, teacher, added loss, or new trainable
parameter.

At validation and inference, force `alpha=0` regardless of global step. The
saved architecture remains normal CL39; the intervention is a training
obligation, not a deployed route.

Implement the merge as an explicit `if alpha > 0` branch. When alpha is zero,
return the existing CL39 expression without performing an algebraically
equivalent `lerp`; this preserves the historical floating-point path exactly.

### 3.2 Coherent deterministic selection

Compute selection once in `PhotomakerBranchedLora.forward`, on CPU, from a
specified 32-bit integer hash of `(seed, global_step)`. Do not call
`torch.rand()` and do not sample separately by layer or DDP rank. A suitable
portable hash is:

```python
def stateless_u32(seed: int, step: int) -> int:
    value = (int(seed) ^ (int(step) * 0x9E3779B1)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF

selected = stateless_u32(seed, global_step) < round(probability * 2**32)
```

The model stores one scalar runtime strength. `branched_runtime.py` copies that
scalar to selected processors before the U-Net forward. An assertion in the
smoke path must prove that all 36 selected processors see the same value and
all unselected processors see zero.

### 3.3 Hypothesis, prediction, and risk

- **Hypothesis:** raw R is fragile partly because CL39 training only needs it
  as a confidence-attenuated correction around N. `[hypothesis]`
- **Prediction:** the R2-A checkpoint's raw-R counterfactual has higher ID and
  fewer duplicate/warped facial parts, while its normal CL39 inference remains
  non-inferior. `[prediction]`
- **Risk:** full ownership creates conflicting gradients early or weakens the
  normal N-anchored route. The warm ramp and 12.5% duty cycle bound this risk.

### 3.4 Proposed training YAML

File: `src/configs/CL39R2A_cosmic_reference_face_ownership_24k.yaml`

```yaml
defaults:
  - CL39_cosmic_null_key_confidence_router_24k
  - _self_

model:
  ba_reference_face_ownership_enabled: true
  ba_reference_face_ownership_groups: [up_blocks.0, up_blocks.1]
  ba_reference_face_ownership_probability: 0.125
  ba_reference_face_ownership_seed: 390024
  ba_reference_face_ownership_ramp_start_step: 2000
  ba_reference_face_ownership_ramp_end_step: 6000
  ba_reference_face_ownership_max_strength: 1.0

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/null_key/null_mass/all
    - ba/null_key/reference_fraction/all
    - ba/r2/ownership/selected_fraction/all
    - ba/r2/ownership/strength/all
    - ba/r2/ownership/raw_delta_native_ratio/all
  experiment_comment: >-
    CL39-R2-A vs CL39 adds only deterministic training-time reference-face
    ownership exposure in up0/up1; validation and inference remain exact CL39.
```

Trainable contract remains exactly `2240` tensors and `219217920` parameters.

### 3.5 Promotion gate

Promote A only if all hold:

1. predeclared severe raw-R failures fall from about `8/16` to at most `4/16`;
2. raw-R subject-v2 ID improves by at least `+0.03` on the fixed 16 and does
   not regress on the full 96;
3. normal fixed-96 subject-v2 ID is no worse than CL39 by `0.005`;
4. paired face quality, topology, prompt adherence, and object retention are
   non-inferior;
5. normal output still changes materially under group correction-zero, global
   BA-off, and spatial-reference shuffle controls.

## 4. Experiment B — bounded low/high reliability correction

### 4.1 Single scientific change

Keep the current entropy confidence `C_0`. In each selected processor, add one
small MLP with detached query-level features:

```text
x = [valid_reference_mass,
     conditional_valid_entropy,
     cosine(N, R),
     log1p(rms_channel(L) / rms_channel(N)),
     log1p(rms_channel(H) / rms_channel(N)),
     denoising_progress]

z = MLP_6_16_2(x_detached)
C_L = clip(C_0 + 0.20 * tanh(z_L), 0.25, 1.0)
C_H = clip(C_0 + 0.20 * tanh(z_H), 0.25, 1.0)
Y_B = N + S * (C_L * g_L * L + C_H * g_H * H)
```

`valid_reference_mass` and conditional entropy are computed inside the existing
chunked, no-grad CL39 attention-probability loop. No `LxL` tensor is retained.
For conditional entropy, renormalize probability over binary valid reference
face keys; samples with fewer than two valid keys must fail closed rather than
divide by `log(1)`.

The MLP inputs are detached so Q/K/V projections cannot manipulate the
reliability statistic to open the gate. Only the MLP parameters receive the
new gradient path. Do not add an auxiliary gate loss in this arm.

### 4.2 Initialization and exact trainable contract

Each of the 36 selected processors owns `Linear(6,16) -> SiLU -> Linear(16,2)`.
Initialize the final linear weights and bias to zero. This gives
`C_L=C_H=C_0` at construction. Instantiate the MLP inside
`torch.random.fork_rng(devices=[])` with a fixed local seed so adding the module
does not shift CL39's global initialization or training-noise stream.

Per processor:

```text
6*16 + 16 + 16*2 + 2 = 146 parameters, 4 tensors
```

Across 36 processors:

```text
5,256 parameters, 144 tensors
```

Expected total contract:

```text
2,384 trainable tensors
219,223,176 trainable parameters
```

The implementation must measure and assert these totals; if the selected
processor count differs from 36, stop and update the plan rather than silently
changing the YAML.

### 4.3 Hypothesis, prediction, and risk

- **Hypothesis:** entropy detects diffuse correspondence but misses confidently
  wrong, high-frequency, or abnormally large residuals. `[hypothesis]`
- **Prediction:** learned `C_H` becomes more conservative on bad geometry while
  `C_L` retains useful identity structure; actual quality improves without
  eliminating BA sensitivity. `[prediction]`
- **Risk:** the gate learns the clean-looking native shortcut by suppressing
  both bands. This is why BA-off, correction-zero, shuffle, confidence
  distributions, and correction/native magnitude are promotion gates.

### 4.4 Proposed training YAML

File: `src/configs/CL39R2B_cosmic_band_reliability_gate_24k.yaml`

```yaml
defaults:
  - CL39_cosmic_null_key_confidence_router_24k
  - _self_

model:
  ba_band_reliability_gate_enabled: true
  ba_band_reliability_gate_groups: [up_blocks.0, up_blocks.1]
  ba_band_reliability_gate_feature_version: detached_v1
  ba_band_reliability_gate_hidden_dim: 16
  ba_band_reliability_gate_max_delta: 0.20
  ba_band_reliability_gate_init_seed: 390216

expected_trainable_contract:
  enabled: true
  total_tensors: 2384
  total_parameters: 219223176
  optimizer_tensors: 2384
  optimizer_parameters: 219223176
  categories:
    branched_sa_r128: {name_substring: ".attn1.processor.", tensors: 984, parameters: 127800456}
    generic_effective_adapter_r32: {name_substring: ".lora_adapter.", tensors: 700, parameters: 30474240}
    photomaker_default_effective_adapter_r64: {name_substring: ".default.", tensors: 700, parameters: 60948480}

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/null_key/null_mass/all
    - ba/null_key/reference_fraction/all
    - ba/r2/reliability/valid_mass/all
    - ba/r2/reliability/conditional_entropy/all
    - ba/r2/reliability/confidence_low/all
    - ba/r2/reliability/confidence_high/all
    - ba/r2/reliability/low_minus_base/all
    - ba/r2/reliability/high_minus_base/all
    - ba/r2/reliability/correction_native_ratio/all
  experiment_comment: >-
    CL39-R2-B vs CL39 adds only a zero-initialized bounded low/high reliability
    correction in up0/up1; the CL39 entropy confidence remains the base gate.
```

The `branched_sa_r128` category deliberately includes the new processor MLPs:
`840+144=984` tensors and `127795200+5256=127800456` parameters. If the current
contract checker requires disjoint substrings, add a fourth explicit
`band_reliability_mlp` category instead and keep the same totals; do not weaken
the checker.

### 4.5 Promotion gate

Promote B only if all hold:

1. fixed-96 subject-v2 ID improves by at least `+0.005`, or a predeclared
   hard-case/face-quality improvement is paired with ID non-inferiority within
   `-0.003`;
2. face topology and object retention improve or remain non-inferior;
3. neither band collapses to the `0.25` floor on most queries;
4. routed-correction/native RMS retains at least 70% of matched CL39's mean;
5. actual remains causally sensitive to reference shuffle, group
   correction-zero, and global BA-off;
6. gate outputs are finite and do not saturate at both bounds.

## 5. Experiment C — separately capped residual tails

### 5.1 Single scientific change

For each sample and selected processor, compute binary-face masked RMS and cap
the Gaussian bands **before** denoising-progress gains and CL39 confidence:

```text
scale_b = min(1, k_b * RMS_face(N) / (RMS_face(b) + 1e-6))
b_hat = stop_gradient(scale_b) * b

k_L = 0.90
k_H = 0.45

Y_C = N + S * C_0 * (g_L * L_hat + g_H * H_hat)
```

The constants start near the observed p95 ratios, so the mechanism should touch
tails rather than rescale the median layer call. The scale is detached and
adds no parameters. CL27's existing surface objective consumes the capped
components because those are the activations actually routed by this arm; its
weights, masks, and semantics otherwise remain unchanged.

### 5.2 Hypothesis, prediction, and risk

- **Hypothesis:** a small number of large band residuals initiate duplicate or
  warped facial structure during denoising. `[hypothesis]`
- **Prediction:** cap activation is sparse, actual face-quality/topology tails
  improve, and mean identity stays nearly unchanged. `[prediction]`
- **Risk:** useful identity detail is in the clipped tail. The arm fails if caps
  activate broadly or identity falls by `0.005` or more.

### 5.3 Proposed training YAML

File: `src/configs/CL39R2C_cosmic_band_rms_cap_24k.yaml`

```yaml
defaults:
  - CL39_cosmic_null_key_confidence_router_24k
  - _self_

model:
  ba_band_rms_cap_enabled: true
  ba_band_rms_cap_groups: [up_blocks.0, up_blocks.1]
  ba_band_rms_cap_low_ratio: 0.90
  ba_band_rms_cap_high_ratio: 0.45
  ba_band_rms_cap_epsilon: 1.0e-6

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/null_key/null_mass/all
    - ba/null_key/reference_fraction/all
    - ba/r2/rms_cap/low_scale/all
    - ba/r2/rms_cap/high_scale/all
    - ba/r2/rms_cap/low_active_fraction/all
    - ba/r2/rms_cap/high_active_fraction/all
    - ba/r2/rms_cap/correction_native_ratio/all
  experiment_comment: >-
    CL39-R2-C vs CL39 adds only fixed per-sample low/high face-RMS tail caps in
    up0/up1, before the existing gains and confidence.
```

Trainable contract remains exactly `2240` tensors and `219217920` parameters.

### 5.4 Promotion gate

Promote C only if all hold:

1. cap activation is sparse and interpretable: initially target roughly
   `1-15%` of eligible layer calls per band, not most calls;
2. actual fixed-96 subject-v2 ID loss is less than `0.005`;
3. predeclared topology/artifact and lower-tail face-quality measures improve;
4. BA sensitivity to correction-zero, BA-off, and shuffle is retained;
5. improvements are not explained only by a smaller RGB intervention.

If caps activate on more than 20% of calls, treat the constants as a failed
mechanism setting. Do not silently tune them inside the same run. A lower/higher
cap is a new registered arm.

## 6. Minimal implementation map

Keep changes localized to existing CL39 plumbing. Do not add a parallel model
or copy the processor.

### 6.1 `src/configs/model/photomaker_branched_lora2.yaml`

Add the following defaults-off schema near the CL39 fields:

```yaml
# CL39-R2 independent experiments. All defaults preserve historical CL39.
ba_reference_face_ownership_enabled: false
ba_reference_face_ownership_groups: null
ba_reference_face_ownership_probability: 0.125
ba_reference_face_ownership_seed: 390024
ba_reference_face_ownership_ramp_start_step: 2000
ba_reference_face_ownership_ramp_end_step: 6000
ba_reference_face_ownership_max_strength: 1.0

ba_band_reliability_gate_enabled: false
ba_band_reliability_gate_groups: null
ba_band_reliability_gate_feature_version: detached_v1
ba_band_reliability_gate_hidden_dim: 16
ba_band_reliability_gate_max_delta: 0.20
ba_band_reliability_gate_init_seed: 390216

ba_band_rms_cap_enabled: false
ba_band_rms_cap_groups: null
ba_band_rms_cap_low_ratio: 0.90
ba_band_rms_cap_high_ratio: 0.45
ba_band_rms_cap_epsilon: 1.0e-6
```

### 6.2 `src/model/photomaker_branched/lora2.py`

Make only these orchestration changes:

1. add matching constructor fields and persist normalized values;
2. require exactly zero or one R2 flag enabled;
3. when an R2 flag is enabled, require `hardcase_mode=temporal_frequency`,
   inherited CL39 null-key routing, exact groups `up_blocks.0/1`, strict
   BA-only ownership, pose ratio zero, and CA mixing false;
4. for A, calculate the one stateless route strength once per training forward;
   validation/eval always yields zero;
5. put the R2 name, groups, constants, feature version, and runtime/inference
   semantics in `_branched_architecture_manifest()`;
6. leave the existing optimizer path unchanged; its trainable-contract check
   must discover B's MLP parameters automatically.

Add a dated `AICODE-NOTE` at the single route-strength assignment explaining
that one decision must own the full selected processor set and must be zero in
evaluation. No per-layer RNG or processor-map lookup belongs here.

### 6.3 `src/model/photomaker_branched/branched_runtime.py`

1. Build one `r2_enabled`/`r2_groups` map beside the existing CL38-CL44 map.
2. Pass each constructor flag only when the processor name matches its declared
   group.
3. In `_apply_runtime_flags`, call a processor setter for A's already-computed
   scalar strength.
4. Keep one cached `unet.attn_processors` dictionary per collector/loop. Never
   resolve the Diffusers property inside a per-layer loop.

### 6.4 `src/model/photomaker_branched/attn_processor_cleanest.py`

Extend the existing `BranchedAttnProcessor`; do not create a duplicate class.

- Add defaults-off constructor fields and validate local bounds.
- Construct `band_reliability_mlp` only for B-enabled processors, in fp32 and
  under a local forked RNG.
- Preserve the current `_null_key_confidence()` implementation verbatim when B
  is off. Use a separate B-enabled branch to return compact detached features;
  this protects byte parity for historical CL39.
- Apply C's cap immediately after `_gaussian_split(raw_delta)` and before
  `low_component_before_confidence`/`high_component_before_confidence`.
- Apply B's `C_L/C_H` where current CL39 multiplies both bands by one confidence.
- Build ordinary `Y_CL39` first, then apply A's externally supplied interpolation
  immediately before `target_out` is finalized. Do not alter the CL27 auxiliary
  inputs for A.
- Record only scalar/reduced telemetry in `_latest_ba_telemetry`; do not retain
  full activations or an attention matrix.

### 6.5 Validation-pipeline plumbing

Add the new model attributes to the existing validation-pipeline attribute
copy in `src/trainer/base_trainer.py`. The validation pipeline needs B/C
architecture constants and learned processor state, but A's runtime strength
must still be forced to zero. `src/pipelines/br_pipeline_helpers.py` should keep
reusing the trained processor map and checkpoint state; no second
implementation of any R2 equation is allowed.

### 6.6 Compact telemetry collection

Use the existing `_latest_ba_telemetry` collector in
`src/model/photomaker_branched/lora2_helpers.py`. Add
`.attn1.processor.band_reliability_mlp.` to the hard-v1 trainable allowlist only
when R2-B is enabled; without this explicit ownership marker, strict BA-only
startup will correctly reject the new parameters. Register only the scalar
names shown in the YAMLs. Skip the collector entirely when no requested R2
metric is present. Do not enable `ba_hardcase_telemetry_enabled`; its
full-activation reductions are unnecessary for these scientific measurements.

## 7. Fail-closed allowlist and run records

### 7.1 `src/configs/clean_full_runs.json`

Add three records. Their `canonical_run`/`canonical_comet_key` identify the
matched historical CL39 reference until each new run gets its own immutable
startup record:

```json
"CL39R2A_cosmic_reference_face_ownership_24k": {
  "family": "cosmic_r2",
  "dataset": "cosmic_large_adapted",
  "dataset_target": "src.datasets.cosmic_large_adapted.CosmicLargeAdaptedTrain",
  "validation_only": false,
  "hardcase_mode": "temporal_frequency",
  "feature_path": "model.ba_reference_face_ownership_enabled",
  "feature_value": true,
  "active_extension_paths": [
    "model.ba_null_key_router_enabled",
    "model.ba_reference_face_ownership_enabled"
  ],
  "trainable_tensors": 2240,
  "trainable_parameters": 219217920,
  "canonical_run": "CL39_cosmic_null_key_confidence_router_24k_full96_r4",
  "canonical_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a"
},
"CL39R2B_cosmic_band_reliability_gate_24k": {
  "family": "cosmic_r2",
  "dataset": "cosmic_large_adapted",
  "dataset_target": "src.datasets.cosmic_large_adapted.CosmicLargeAdaptedTrain",
  "validation_only": false,
  "hardcase_mode": "temporal_frequency",
  "feature_path": "model.ba_band_reliability_gate_enabled",
  "feature_value": true,
  "active_extension_paths": [
    "model.ba_null_key_router_enabled",
    "model.ba_band_reliability_gate_enabled"
  ],
  "trainable_tensors": 2384,
  "trainable_parameters": 219223176,
  "canonical_run": "CL39_cosmic_null_key_confidence_router_24k_full96_r4",
  "canonical_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a"
},
"CL39R2C_cosmic_band_rms_cap_24k": {
  "family": "cosmic_r2",
  "dataset": "cosmic_large_adapted",
  "dataset_target": "src.datasets.cosmic_large_adapted.CosmicLargeAdaptedTrain",
  "validation_only": false,
  "hardcase_mode": "temporal_frequency",
  "feature_path": "model.ba_band_rms_cap_enabled",
  "feature_value": true,
  "active_extension_paths": [
    "model.ba_null_key_router_enabled",
    "model.ba_band_rms_cap_enabled"
  ],
  "trainable_tensors": 2240,
  "trainable_parameters": 219217920,
  "canonical_run": "CL39_cosmic_null_key_confidence_router_24k_full96_r4",
  "canonical_comet_key": "b1ca0b3da679401c85b991f1bbdf0b2a"
}
```

### 7.2 `tools/validate_clean_full_config.py`

Extend, do not bypass, the unified validator:

1. add the three R2 enabled paths to the extension path tuple;
2. let a manifest record optionally declare `active_extension_paths`; preserve
   the current fallback for all historical records;
3. require exact R2 group lists and exact constants from the YAML above;
4. require mutual exclusivity of A/B/C;
5. require inherited CL39 null routing and unchanged null-key settings;
6. require A/C historical trainable totals and B's measured expanded totals;
7. retain every `COMMON_VALUE`, forbidden-path, dataset, fixed-96, and PCGrad
   check already present.

The validator must reject ad-hoc Hydra overrides. Do not create a separate R2
launcher or a permissive R2 validator.

## 8. Implementation and verification sequence

### Phase 1 — defaults-off parity

1. Add schema, constructor plumbing, manifest fields, and validator support
   with every R2 flag off.
2. Run Python compile/import checks and Hydra composition for historical CL39.
3. Load the canonical CL39 checkpoint with no missing/unexpected processor keys.
4. Re-run the existing 12-image parity smoke: every PNG must be byte-identical
   with all R2 flags off.
5. Confirm trainable names/counts remain exactly `2240/219217920`.

### Phase 2 — focused mechanism checks

Use a small diagnostic harness or existing smoke path; do not add broad vanity
tests.

| Arm | Required focused check |
|---|---|
| A | forced strength 0 equals CL39; forced 1 equals `N+S(R-N)`; one runtime value across all selected processors; zero during eval |
| B | zero-init output equals CL39 within exact dtype tolerance; 36 MLPs; finite nonzero MLP gradients; detached features; measured `2384/219223176` contract |
| C | very large synthetic band is capped to requested ratio; sub-threshold band is unchanged; cap scale has no gradient; flag-off path exact |

Also verify:

- old CL39 checkpoint round-trip;
- new B checkpoint save/reload with every MLP key present;
- no processor-map lookup inside a per-layer loop;
- disabled auxiliary collectors do no work;
- no full-activation BA telemetry;
- `trainer.active_grad_norm_mode=requested_only`;
- shell syntax for the unified launcher.

### Phase 3 — bounded training smoke

Before a 24k scientific submission, run a bounded two-step Serv smoke through
the same packaged source and environment. This is an operational check only;
it may omit the full validation panel if explicitly named as a throughput/smoke
run. Confirm finite loss, optimizer ownership, GPU memory, expected telemetry,
and checkpoint serialization. Remove/finish the smoke before allocating the
scientific run name.

### Phase 4 — scientific runs

Run A first. B and C may be prepared in parallel but should not be interpreted
as an R2-A composition. If resources permit, the three independent 1-GPU jobs
may run concurrently after the Serv project-job audit. The normal project
ceiling is six requested A100s; the earlier ten-GPU exception applied to the
previous named experiment scope and must not be assumed here.

## 9. Serv launch handoff

Run every command from `diffusion_template/` in the packaged Serv source. The
three Hydra YAML leaves above are consumed by the existing unified launcher;
do not encode infrastructure behavior in them.

### 9.1 Local/pre-package gates

```bash
python tools/validate_clean_full_config.py --list

python tools/validate_clean_full_config.py \
  --config-name CL39R2A_cosmic_reference_face_ownership_24k
python tools/validate_clean_full_config.py \
  --config-name CL39R2B_cosmic_band_reliability_gate_24k
python tools/validate_clean_full_config.py \
  --config-name CL39R2C_cosmic_band_rms_cap_24k

bash -n launchers/active/run_clean_full_config_1gpu.sh
```

Package the exact source using the usual hash-manifest workflow, then verify it
on Serv with `tools/verify_serv_source_manifest.py` before submission.

### 9.2 Scientific commands inside each one-GPU Serv job

```bash
CONFIG_NAME=CL39R2A_cosmic_reference_face_ownership_24k \
RUN_NAME=CL39R2A_cosmic_reference_face_ownership_24k_full96_r1 \
  bash launchers/active/run_clean_full_config_1gpu.sh
```

```bash
CONFIG_NAME=CL39R2B_cosmic_band_reliability_gate_24k \
RUN_NAME=CL39R2B_cosmic_band_reliability_gate_24k_full96_r1 \
  bash launchers/active/run_clean_full_config_1gpu.sh
```

```bash
CONFIG_NAME=CL39R2C_cosmic_band_rms_cap_24k \
RUN_NAME=CL39R2C_cosmic_band_rms_cap_24k_full96_r1 \
  bash launchers/active/run_clean_full_config_1gpu.sh
```

The Serv job environment must provide the existing `.env` values, sealed
subject-v2 embedding path, Cosmic manifest/root, PhotoMaker checkpoint when
overridden, and face-quality scorer interpreter. Never put credentials in the
job YAML or source package.

### 9.3 Startup acceptance gate

For each run, require all of the following before reporting it as running:

1. MLS state is Running or valid Pending, with the expected one-A100 request;
2. Cosmic preflight completed successfully;
3. `saved/<run_name>/comet_experiment.json` exists;
4. it contains a 32-character immutable experiment key written by
   `CometMLWriter`;
5. the log shows the exact expected trainable contract;
6. step-zero validation starts/completes 96 images with no configuration drift.

Do not identify the experiment later only by its display name; copy each
immutable Comet key into the experiment ledger.

## 10. Evaluation required after training

For CL39 and each candidate, report both the selected checkpoint and 24k
endpoint on the same fixed 96. Generate:

- normal actual output;
- native `N` only;
- raw `R` on routed face;
- low only;
- high only;
- confidence forced to `1`;
- group-scoped correction-zero;
- global BA-off;
- correct-ID-token/spatial-reference shuffle.

Use the same predeclared 16-cell panel for detailed branch visuals and score all
96 where the intervention is available. Report paired mean, median, p10,
95% bootstrap interval, and wins/ties/losses for subject-v2 ID; report face
quality, face detection/subject assignment, prompt alignment, object retention,
and topology alongside it.

For B and C, a cleaner image produced by shutting off BA is a failure, not a
success. For A, improvement in raw R without preserving normal output is also a
failure.

## 11. Confidence and what is not established

| Claim | Confidence | Basis |
|---|---|---|
| R2-A is the best first test | High | direct match to measured raw-R weakness and CL19/CL14 ownership contrast |
| Literal binary `q_face` is the useful CL14 change | Low / rejected as primary | current mask/router geometry makes it mostly boundary-local |
| A 12.5% duty cycle and 2k-6k ramp are optimal | Not established | conservative starting design, not a completed sweep |
| Entropy confidence is helpful in current CL39 | High on audited subset | forced `C=1` was materially worse |
| Learned band reliability will improve CL39 | Moderate hypothesis | mechanism is motivated but untrained; shortcut risk remains |
| `k_L=0.90`, `k_H=0.45` target tails | Moderate-high | values align with measured p95 ratios |
| RMS caps improve perceptual quality | Not established | magnitude telemetry is not a quality metric |
| The three mechanisms combine additively | Not established | independent runs must pass before any composition |
| One seed establishes a population-level gain | No | a second training seed is required for promotion beyond a research lead |

## 12. Developer completion checklist

- [ ] Read `docs/handoffs/LATEST.md` and this plan before editing.
- [ ] Recheck branch/worktree and preserve unrelated changes.
- [ ] Add defaults-off schema and constructor/runtime plumbing.
- [ ] Implement R2-A only; verify historical parity and mechanism checks.
- [ ] Implement R2-B independently; measure, do not guess, trainable contract.
- [ ] Implement R2-C independently; verify sparse cap behavior.
- [ ] Add three direct-child YAML leaves exactly as specified.
- [ ] Extend `clean_full_runs.json` and the unified fail-closed validator.
- [ ] Verify optimized-pipeline invariants.
- [ ] Run bounded Serv smoke for each arm.
- [ ] Inspect running/pending project jobs before scientific submission.
- [ ] Launch with unique run names through the unified one-GPU launcher.
- [ ] Capture immutable Comet keys at startup.
- [ ] Monitor step-zero and every-2,000-step fixed-96 validation.
- [ ] Run the complete branch/counterfactual evaluation before combining arms.
- [ ] Update `docs/handoffs/LATEST.md` only when a material result or decision
      supersedes the current handoff.

## References

- `docs/handoffs/LATEST.md`
- `analysis/2026-08-26_ba_lineage_r_frequency_confidence_audit.md`
- `analysis/2026-08-25_cl39_r_branch_artifact_diagnosis_and_r2_architecture.md`
- `analysis/2026-08-25_cl39_cl14_qface_hypothesis_and_experiment_plan.md`
- `analysis/2026-08-16_training_pipeline_processor_lookup_fix.md`
- `src/model/photomaker_branched/attn_processor_cleanest.py`
- `src/model/photomaker_branched/branched_runtime.py`
- `src/model/photomaker_branched/lora2.py`
- `src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml`
- `src/configs/clean_full_runs.json`
- `tools/validate_clean_full_config.py`
- `launchers/active/run_clean_full_config_1gpu.sh`
