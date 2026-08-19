---
title: "CL27 next-eight architecture experiments"
subtitle: "Implementation-ready single-delta plans for higher ID_SIM, better visual quality, and harder occlusion/small-face cases"
date: "19 August 2026"
status: "Design and implementation handoff; no training jobs launched"
branch: "test"
baseline: "CL27 r3 at 16k, Comet dbfbf40c3bdd4f70bedc58bda3dfb9cd"
---

# Executive decision

**Keep CL27 at 16k as the scientific control and build the next suite as eight independent, cold-start, one-change ablations.** `[measured][decision]`

The completed evidence does not justify replacing CL27 with any CL30-CL37 checkpoint. CL27's fixed-panel subject-v2 identity score is **0.547260 at 16k**. CL33 is close at **0.546311**, but it is a diagnostic control rather than a successor: its Skiing gain is partly accompanied by deletion of ordinary glasses in the Elon case, and it still does not solve Marion's opaque goggles. Controlled PhotoMaker remains slightly higher at **0.556580**, while also retaining stronger face-quality aggregates. The next suite therefore needs to improve identity **without using object deletion, extra faces, or face-crop degradation as shortcuts**.

The eight recommended arms are:

| Priority | Run | Single critical change from CL27 | Main target | Risk |
|---:|---|---|---|---|
| **1** | **CL38 visibility-ownership v2** | Add a training-only native-target anchor on top-object/contact pixels | Stop glasses/goggles/hands/hair deletion while retaining CL27 identity | Low-medium |
| **2** | **CL39 null-key confidence router** | Let each target query abstain from unmatched reference K/V and fall back to native SA | Occlusion topology and broad visual stability | Medium |
| **3** | **CL41 landmark-canonical K/V** | Canonicalize the reference face by five landmarks before target-Q/reference-KV attention | Pose/expression mismatch, profile views, hard identities | Medium |
| **4** | **CL43 ID-adaptive modulation** | Modulate the routed spatial-reference delta from the existing 512-D face embedding | Broad ID_SIM uplift without more reference copying | Medium-high |
| **5** | **CL42 facial-component token memory** | Add five pooled reference component tokens queried by the existing target face queries | Small faces and fine identity detail | Medium |
| **6** | **CL44 semantic high-frequency window** | Sample-condition only the high-frequency reference lane by time and target/reference agreement | Avoid late texture artifacts and pose/occluder leakage | Low-medium |
| **7** | **CL40 identity-motion projector** | Remove a learned motion-aligned component from the routed reference delta | Identity/pose-expression disentanglement | High |
| **8** | **CL45 BA-only gradient surgery** | Project the CL27 auxiliary gradient only when it conflicts with main diffusion gradients | Prevent the post-16k identity/quality drift | Medium-high engineering risk |

The suite is deliberately **not** another clean multi-scale memory, dense ownership MLP, same-ID cross-view consistency loss, residual identity cross-attention branch, global ArcFace reward, static learned frequency schedule, low-band contrastive objective, or small-face teacher. Those literal ideas already have negative, inactive, or ambiguous project evidence in CL16-CL18, CL21-CL26, and CL28-CL37.

No arm should run automatically to 24k. The common funnel is:

```text
implementation + unit parity
          |
          v
1k activation/gradient gate
          |
          v
4k fixed-96 + causal intervention gate
          |
          v
16k primary promotion gate
          |
          +---- reject / archive
          |
          v
18k stability gate
          |
          +---- optional 24k only if still improving
```

# 1. Evidence base and constraints

## 1.1 Source-derived result ledger

The following values come from the latest handoff and the completed CL30-CL37 report on branch `test`.

| Arm | Evaluated checkpoint | ID_SIM | Delta vs CL27 | Decision |
|---|---:|---:|---:|---|
| Controlled PhotoMaker | 0 | **0.556580** | +0.009320 | External upper control; not BA |
| **CL27** | **16k** | **0.547260** | - | Retained base |
| CL30 | 16k | 0.537826 | -0.009435 | Reject |
| CL31 | 24k | 0.537079 | -0.010181 | Reject |
| CL32 | 18k | 0.542138 | -0.005122 | Diagnostic only |
| CL33 | 16k | 0.546311 | -0.000949 | Closest control; not successor |
| CL34 | 18k | 0.538839 | -0.008421 | Reject |
| CL35 | 24k | 0.537951 | -0.009309 | Intended identity reward inactive |
| CL36 | 4k | 0.528958 | -0.018302 | Early reject; identity gradients inactive |
| CL37 | 18k | 0.537343 | -0.009917 | Reject |

Key hard-slice evidence:

| Slice | PhotoMaker | CL27 | Best relevant new control | Interpretation |
|---|---:|---:|---:|---|
| Skiing | 0.4640 | 0.4337 | CL33 0.4620 | CL33 gains but has deletion risk |
| Crying | 0.6000 | 0.5855 | CL33 0.5710 | CL27 remains stronger |
| Jumping | - | 0.3946 | CL32 0.4105 | Contact partition can help locally |
| Dancing | - | 0.4422 | CL35 0.4619 | Local gain does not establish active patch-ID learning |
| Marion, all prompts | 0.5029 | 0.4935 | CL31 0.4885 | No CL30-CL37 successor solves Marion |

Quality/ownership context:

- CL27 has approximately **0.9211 mask IoU**, **1.125 mean faces**, and TOPIQ face/overall around **0.7142/0.5882**.
- PhotoMaker has stronger face TOPIQ, around **0.7532**, but lower mask ownership IoU, around **0.8652**.
- A candidate that raises ID_SIM while deleting prompted eyewear, producing no face, or producing a second face is a failure.

## 1.2 What CL27 actually contributes

CL27 inherits CL23's target-query/reference-KV self-attention and temporal-frequency routing. It adds only a training-time frequency-surface objective at `up_blocks.0` and `up_blocks.1`:

- suppress routed high-frequency energy on synthetic top-object pixels;
- down-weight low-band pressure on those top-object pixels;
- maintain a visible-face reference-energy floor;
- train with deterministic semantic occlusion at probability 0.25.

Conceptually, the current path is:

```text
target hidden states H_t ---------------------> native self-attention Y_native
          |                                               |
          | Q_target                                      |
          v                                               |
reference hidden states H_r -> K_ref,V_ref -> Y_reference |
                                                          |
D = Y_reference - Y_native                                |
R = temporal_frequency_router(D, timestep, block group)   |
                                                          v
Y_CL27 = Y_native + face_mask * R
```

The audited inference contract is:

```text
use_branched_attention = true
pose_adapt_ratio = 0.0
ca_mixing_for_face = false
reference_face_kv_weight = 1.0
branched cross-attention = disabled
```

Every proposed arm keeps those settings unless its named delta explicitly acts inside the self-attention reference lane. None re-enables branched cross-attention.

## 1.3 Failure map that the next eight must address

| Failure mechanism | Evidence | Design implication |
|---|---|---|
| Reference face detail overwrites a target-owned object | Ski goggles, glasses, hair, hand/tear contact; CL33 can delete glasses | Add a true native fallback or explicit native-ownership anchor |
| Global bbox alignment is too weak under pose/expression change | Persistent hard identities and profile/expressive cases | Canonicalize geometry or separate identity from motion |
| Small face has too few spatial tokens for stable reference retrieval | CL37 teacher did not transfer; small/action cases remain variable | Preserve part-level identity in a fixed-size token memory |
| Raw identity auxiliaries are easy to make inactive | CL35 gate below its intended floor; CL36 BA auxiliary gradient ratio zero | Require activity and gradient gates before long runs |
| Static learned frequency endpoints did not generalize | CL28 and CL34 underperform CL27 | Use sample-conditioned confidence, not another global schedule |
| CL27 peaks at 16k and degrades later | Current base decision | Protect main reconstruction gradients from conflicting auxiliary updates |
| Residual sidecar can be ignored | Historical residual-SA v2 and CL21 | Do not add another unconstrained optional residual as a top-priority arm |
| Wrong-reference contrast can damage requested accessories | CL29 and related controls | Do not use a new wrong-ID repulsion arm in this suite |

# 2. Literature-derived design principles

The papers below inform the mechanisms, but **their results do not establish that the proposed transfers will work in CL27**.

1. **DynamicID** uses query-level semantic activation to reduce disruption from identity injection and an identity-motion reconfiguration module to improve editability. This motivates CL39 and CL40.
2. **InfiniteYou** injects identity through a separate residual path, while staged paired-data training reduces copy-paste. The project already has negative evidence for a literal residual repetition, so this report keeps the principle of non-destructive injection but does not rank another generic residual branch.
3. **FaceCrafter** encourages identity features to remain orthogonal to pose/expression/emotion controls. This motivates CL40.
4. **LaTo** and **DiffSwap++** use landmark/3D facial structure to decouple geometry from identity. This motivates CL41.
5. **ConsistentID** uses local, fine-grained facial information and attention localization. This motivates CL42.
6. **AnyPhoto** uses identity-adaptive modulation from face-recognition embeddings and anti-copy training perturbations. This motivates CL43.
7. **Beyond Facial Consistency** uses adaptive temporal gating and region-aware optimization to coordinate identity branches. This motivates CL44.
8. **InsHuman** and **MagicMakeup** emphasize region-aware weighting and token-aligned region gating for identity-preserving edits. This supports CL38's ownership-first design.
9. **WithAnyone** shows why paired identity supervision must explicitly avoid copy-paste. It supports the evaluation controls, but prior project cross-view/contrastive arms make another literal paired contrastive experiment low priority.
10. **PCGrad** projects conflicting task gradients. This motivates CL45's asymmetric, BA-only optimization intervention.

# 3. Shared implementation contract

## 3.1 Common branch and checkpoint policy

Each arm is a **cold run from the same CL27 parent configuration**, not a continuation from the CL27 16k checkpoint. This keeps the scientific delta attributable to the proposed mechanism.

Suggested branches:

```text
exp/cl38-visibility-ownership-v2
exp/cl39-null-key-router
exp/cl40-id-motion-projector
exp/cl41-landmark-canonical-kv
exp/cl42-component-token-memory
exp/cl43-id-adaptive-modulation
exp/cl44-semantic-window-gate
exp/cl45-ba-pcgrad
```

The CL27 16k checkpoint remains the evaluation control. A candidate checkpoint is not promoted merely because it is the highest point on its own curve.

## 3.2 Files another agent should expect to touch

| File | Shared purpose |
|---|---|
| `src/model/photomaker_branched/attn_processor_cleanest.py` | New processor math, feature flags, per-layer telemetry/loss accumulators |
| `src/model/photomaker_branched/branched_runtime.py` | Runtime state propagation, reference landmarks and ID embeddings |
| `src/model/photomaker_branched/lora2.py` | Defaults-off config fields, processor wiring, loss popping |
| `src/model/photomaker_branched/lora2_helpers.py` | Trainable role classification, checkpoint and telemetry allowlists |
| `src/datasets/cosmic_large_adapted.py` | Reference landmarks for CL41/42; no dataset change for the other arms |
| `src/datasets/collate.py` | Collate/pad new landmark tensors and validity flags |
| `train.py` | Config mapping, candidate auxiliary loss integration, CL45 gradient handling |
| `src/configs/CL38...CL45...yaml` | One YAML per arm |
| `tests/photomaker_branched/` | Disabled-path parity, activity, gradients, checkpoint round trip, DDP smoke |
| `analysis_sidecars/` | Fixed-96 pairing, reference shuffle/zero-reference interventions, object-retention sheet |

## 3.3 Required fail-closed behavior

For every arm:

1. `enabled=false` must reproduce the current CL27 processor output within the existing dtype tolerance.
2. Missing optional metadata must either:
   - fall back to exact CL27 and increment a fallback counter; or
   - raise before training starts if the active arm scientifically requires that metadata.
3. No module may be created lazily inside `forward`.
4. Every trainable tensor must be registered before optimizer construction.
5. Checkpoint save/load must reproduce the active output.
6. Two-GPU DDP must exercise the active path without unused-parameter divergence.
7. The first 1k logs must prove nonzero path activity and, for trainable arms, nonzero gradients.

# 4. Common evaluation and promotion gates

## 4.1 Primary fixed-panel gates

Use the existing fixed 96-image validation panel, one image per item, unchanged references, prompts, seeds, scheduler, DDIM50, CFG5, RealVisXL validation base, and subject-v2 identity metric.

**16k promotion gate:**

- aggregate ID_SIM `>= 0.547260`; or, for a scientifically exceptional hard-case arm, paired delta no worse than `-0.002` with a bootstrap interval containing zero and a clear hard-case/topology win;
- no ordinary-glasses deletion in the fixed panel;
- Skiing topology at least `7 pass / 1 minor / 0 fail`;
- Crying ID delta versus CL27 no worse than `-0.005`;
- Marion-all delta versus CL27 no worse than `-0.005`;
- no increase in zero-face or two-face topology;
- mask IoU `>= 0.91`;
- TOPIQ-face and overall TOPIQ each no worse than CL27 by more than `0.01`.

**18k stability gate:**

- no ID_SIM fall greater than `0.003` from the candidate's 16k value;
- no new topology deletion;
- active mechanism telemetry remains in its intended range.

A 24k continuation is optional only after both gates pass.

## 4.2 Causal intervention panel

At 4k, 16k, and the selected final gate, render four matched variants for at least the full hard subset and preferably all 96 cells:

```text
A. normal CL27/candidate reference
B. spatial reference shuffled, PhotoMaker ID tokens kept correct
C. spatial reference zeroed, PhotoMaker ID tokens kept correct
D. candidate feature disabled at inference
```

Record:

- face-region pixel/latent difference;
- ID_SIM change;
- correct-versus-shuffled gap;
- object-retention topology;
- mechanism-specific telemetry.

A spatial-reference mechanism that produces no correct-versus-shuffled separation is not causally established even if aggregate images change.

## 4.3 Supplementary hard panel

Do not replace the fixed 96. Add a diagnostic panel with:

- thin ordinary spectacles;
- opaque ski goggles;
- hair crossing one eye;
- hand/tear contact with cheek or eyelid;
- profile and three-quarter pose;
- extreme smile/cry;
- small face occupying less than 8% of image area;
- low-contrast/poorly lit references.

# 5. Experiment CL38 - visibility-ownership v2 native anchor

## 5.1 Hypothesis

CL27 already identifies synthetic top-object and visible-face regions, but it only shapes frequency energy. It does not directly say: **on target-owned top-object pixels, the branched route must not replace the native target self-attention interpretation**.

CL38 adds one training-only loss family:

```text
L_top_native =
    mean_top |Y_candidate - stopgrad(Y_native)|

L_contact_native =
    mean_contact |Y_candidate - stopgrad(Y_native)|

L_CL38 =
    L_CL27
    + ramp(step) * (
          0.020 * L_top_native
        + 0.010 * L_contact_native
      )
```

`Y_native` is already computed inside the processor, so CL38 does not need a second U-Net forward. The top-object mask is the existing deterministic semantic occluder mask. The contact ring is a one-cell dilation intersection around the object/visible-face boundary.

This is different from CL33. CL33 reweighted reconstruction partitions and could improve Skiing by exposing more face. CL38 directly anchors target-owned pixels to the native target path, making deletion an explicit loss violation.

## 5.2 Exact code plan

1. In `attn_processor_cleanest.py`, immediately after both `Y_native` and the final candidate message exist:
   - resize `semantic_occluder_mask` to the token grid;
   - build `M_top` and `M_contact`;
   - accumulate detached area telemetry;
   - compute the two L1 terms without detaching the candidate;
   - detach only `Y_native`.
2. Add `pop_visibility_ownership_v2_loss()` and `pop_visibility_ownership_v2_metrics()` mirroring the current frequency-surface accumulator pattern.
3. In `lora2.py`, add defaults-off fields, propagate them to processors, collect the loss once per training forward, and add it to `loss_ba_aux`.
4. In `train.py`, expose the scalar and the two component metrics to the writer.
5. No inference code change. No new parameters. Keep the exact CL27 trainable contract.

## 5.3 Activation tests

- All-black top mask: loss exactly zero.
- Synthetic goggles crossing the eye region: top and contact losses nonzero.
- Candidate equal to native: loss exactly zero.
- Candidate differs only on visible face: ownership loss zero.
- Candidate differs only on top object: ownership loss positive and gradient reaches BA tensors.
- `enabled=false`: exact CL27 output/loss list.

## 5.4 YAML - `src/configs/CL38_cosmic_visibility_ownership_v2_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

# Design blueprint. Remove this guard only after code/tests/contract audit pass.
implementation_blueprint_do_not_launch: true

model:
  ba_visibility_ownership_v2_enabled: true
  ba_visibility_ownership_v2_groups: [up_blocks.0, up_blocks.1]
  ba_visibility_ownership_v2_top_native_weight: 0.020
  ba_visibility_ownership_v2_contact_native_weight: 0.010
  ba_visibility_ownership_v2_dilate_cells: 1
  ba_visibility_ownership_v2_min_top_area: 0.002
  ba_visibility_ownership_v2_stopgrad_native: true
  ba_visibility_ownership_v2_ramp_start_step: 1000
  ba_visibility_ownership_v2_ramp_end_step: 4000

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - loss_ba_visibility_ownership_v2
    - ba/visibility_ownership_v2/top_native_l1
    - ba/visibility_ownership_v2/contact_native_l1
    - ba/visibility_ownership_v2/top_area
    - ba/visibility_ownership_v2/applied_fraction
  experiment_comment: >-
    CL38 vs CL27 adds only a training-time native-ownership anchor on
    synthetic top-object and contact pixels. Inference remains exact CL27.
```

## 5.5 Arm-specific gate

At 4k, the mean top/contact native deviation must be at least 20% lower than matched CL27 on applied synthetic-occlusion batches, while visible-face routed energy remains at least 90% of CL27. Reject if the model obtains lower top deviation simply by collapsing the full reference lane.

# 6. Experiment CL39 - null-key confidence router

## 6.1 Hypothesis

CL27 forces a reference message inside the face lane even when a target query has no good reference match. This is structurally wrong for target-only objects such as glasses, goggles, hair, tears, or a hand. Instead of asking a dense MLP to classify ownership, add one **null match** to the reference attention.

For each target query and head:

```text
z_ref[j] = Q_target dot K_ref[j] / sqrt(d)
p_ref = softmax(z_ref)

normalized_entropy =
    -sum_j p_ref[j] * log(p_ref[j] + eps)
    / log(number_of_reference_tokens)

p_null =
    sigmoid(
      (normalized_entropy - entropy_threshold)
      / temperature
    )

Y_ref = ordinary_reference_attention(Q_target, K_ref, V_ref)

confidence = clamp(
    1 - max_abstention * p_null,
    min_reference_fraction,
    1
)

R_CL39 = confidence * R_CL27
Y_CL39 = Y_native + M_face * R_CL39
```

The null value is not emitted as a zero feature. Its probability becomes a **native fallback coefficient**, so abstention keeps `Y_native` rather than producing a blank face.

This is not CL17/CL22's learned dense ownership router. It adds no predictor and no region labels at inference. The attention matching itself produces confidence.

## 6.2 Exact code plan

1. Factor the reference attention logits in `attn_processor_cleanest.py` into a helper that can optionally return logits/probabilities.
2. Under `ba_null_key_router_enabled`:
   - compute normalized per-query reference-attention entropy;
   - compare it with the fixed virtual-null entropy threshold to obtain `p_null`;
   - keep the ordinary reference attention output unchanged;
   - multiply the already-routed CL27 delta by the bounded confidence.
3. Preserve the current PyTorch scaled-dot-product path when disabled. The enabled path may use explicit logits only at configured groups.
4. Log null mass by block group and, in training, by synthetic top-object versus visible-face query regions.
5. No new parameters and no data changes.

## 6.3 Activation tests

- A low-entropy reference match: null mass low.
- High-entropy/ambiguous reference logits: null mass high.
- `max_abstention=0`: exact CL27.
- `p_null=1`: output becomes native target message, not zero.
- Top-object query mass should exceed visible-face null mass on synthetic occlusion by 1k.
- Reject if visible-face median reference fraction falls below 0.55.

## 6.4 YAML - `src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true

model:
  ba_null_key_router_enabled: true
  ba_null_key_router_groups: [up_blocks.0, up_blocks.1]
  ba_null_key_logit_mode: normalized_entropy_virtual_null
  ba_null_key_entropy_threshold: 0.75
  ba_null_key_temperature: 0.08
  ba_null_key_max_abstention: 0.75
  ba_null_key_min_reference_fraction: 0.25
  ba_null_key_apply_low_band: true
  ba_null_key_apply_high_band: true
  ba_null_key_native_fallback: true

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/null_key/null_mass/all
    - ba/null_key/null_mass/up0
    - ba/null_key/null_mass/up1
    - ba/null_key/reference_fraction/all
    - ba/null_key/object_minus_visible_mass
  experiment_comment: >-
    CL39 vs CL27 adds only a parameter-free virtual-null abstention score:
    target queries may retain native target self-attention when reference K/V
    has no confident match.
```

## 6.5 Arm-specific gate

By 4k:

- median `p_null(top object) - p_null(visible face) >= 0.08`;
- visible-face reference fraction `>= 0.55`;
- shuffled-reference intervention changes the face more than CL27, but top-object retention is not worse;
- no detector/topology regression.

# 7. Experiment CL40 - low-rank identity-motion projector

## 7.1 Hypothesis

The routed reference difference contains both stable identity and mutable pose/expression/illumination. A lightweight reconfiguration block can remove the part aligned with target/reference motion disagreement before the message is merged.

At each selected processor:

```text
z_t = P_t(LN(H_target_face))
z_r = P_r(LN(H_reference_face_aligned))

z_common = 0.5 * (z_t + z_r)
z_motion = z_t - z_r

z_identity =
    z_common
    - proj(z_common, z_motion)

C = W_out(z_identity)

R_CL40 = R_CL27 + gate * C
```

`P_t`, `P_r`, and `W_out` are rank-32. `W_out` is zero-initialized, so the active configuration begins at exact CL27 output and the new path gets a direct gradient on the first update. The gate is bounded at 0.35.

This is a deliberately smaller, processor-local analogue of identity-motion reconfiguration, not a second denoiser or a new cross-attention branch.

## 7.2 Exact code plan

1. Add a small registered module class in a new file:
   `src/model/photomaker_branched/identity_motion_projector.py`.
2. Instantiate it in processor construction, never lazily in forward.
3. Use masked target/reference tokens at matching token counts. For the initial arm, use the current bbox-aligned reference representation; do not combine with CL41.
4. RMS-match the correction to the CL27 routed delta before the bounded gate.
5. Register trainable roles in `lora2_helpers.py`.
6. Generate and seal a new exact parameter contract after implementation.
7. Log cosine before/after projection, correction RMS, and gate value.

## 7.3 Activation tests

- Identical target/reference features: motion norm near zero; finite output.
- Pure synthetic motion direction: cosine after projection near zero.
- Zero-initialized output: exact CL27.
- First backward: `W_out` receives nonzero gradient.
- By 1k: at least 80% of selected processors have nonzero output RMS and gradient.
- Reject if the correction becomes a generic face residual with no matched-versus-shuffled separation.

## 7.4 YAML - `src/configs/CL40_cosmic_identity_motion_projector_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true
contract_generation_required: true

model:
  ba_identity_motion_projector_enabled: true
  ba_identity_motion_projector_groups: [up_blocks.0, up_blocks.1]
  ba_identity_motion_projector_rank: 32
  ba_identity_motion_projector_gate_max: 0.35
  ba_identity_motion_projector_zero_init_output: true
  ba_identity_motion_projector_eps: 1.0e-6
  ba_identity_motion_projector_ramp_start_step: 1000
  ba_identity_motion_projector_ramp_end_step: 6000

trainer:
  active_grad_norm_mode: requested_only

# The module adds trainable tensors. Generate and seal the exact contract after
# implementation; do not copy a guessed parameter count into an active config.
expected_trainable_contract:
  enabled: false

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/id_motion/cosine_before
    - ba/id_motion/cosine_after
    - ba/id_motion/correction_rms
    - ba/id_motion/gate
  experiment_comment: >-
    CL40 vs CL27 adds only a low-rank identity-motion reconfiguration block
    that removes motion-aligned components from the routed reference delta.
```

# 8. Experiment CL41 - five-landmark canonical reference K/V

## 8.1 Hypothesis

The current global bbox warp assumes that corresponding facial details occupy comparable positions after a single box transform. That is weak for head rotation, expression, crop, and camera geometry. Reference features should first be sampled into a canonical face coordinate system.

Use the standard five points - left eye, right eye, nose, left mouth corner, right mouth corner - to estimate a similarity transform from the reference image to an ArcFace-style canonical template. At every selected self-attention resolution:

```text
H_ref_canon = grid_sample(H_ref, T_ref_to_canonical)
Y_canon = Attention(Q_target, K(H_ref_canon), V(H_ref_canon))

Y_ref_mix =
    rms_match(
        (1 - mix) * Y_ref_original
        + mix * Y_canon,
        Y_ref_original
    )
```

The initial arm uses `mix=0.50`, not a learned scalar. If the landmarks are missing, low confidence, degenerate, or outside the reference crop, the processor uses exact CL27.

Target landmarks are not required at inference: target queries remain in their generated coordinate system and retrieve semantic content from the canonical reference memory.

## 8.2 Exact code plan

1. Extend the existing face-analysis result in `cosmic_large_adapted.py` to return normalized `reference_landmarks_5` and a validity/confidence scalar.
2. Add the fields to `collate.py`.
3. In inference/validation preparation, reuse the existing InsightFace detection and forward its five keypoints; do not run a second detector.
4. Add `landmark_canonicalizer.py`:
   - fixed normalized template;
   - robust least-squares/similarity transform;
   - grid builder cached by spatial resolution, device, and dtype where safe;
   - `grid_sample(..., mode="bilinear", padding_mode="zeros", align_corners=False)`.
5. In `attn_processor_cleanest.py`, compute the canonical reference attention only for the selected groups, then mix and RMS-match.
6. Log fallback rate, landmark confidence, native/canonical cosine, and correction RMS.
7. No trainable parameters; keep the CL27 contract.

## 8.3 Activation tests

- Identity transform landmarks reproduce the original crop within interpolation tolerance.
- Horizontal flip or invalid landmark order fails closed.
- Extreme degenerate points fail closed.
- Checkpoint/inference with missing metadata produces exact CL27 and a fallback counter.
- Applied fraction must exceed 0.95 on the training and fixed validation face-detected samples.

## 8.4 YAML - `src/configs/CL41_cosmic_landmark_canonical_kv_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true

model:
  ba_landmark_canonical_kv_enabled: true
  ba_landmark_canonical_kv_groups: [up_blocks.0, up_blocks.1]
  ba_landmark_canonical_kv_mix: 0.50
  ba_landmark_canonical_kv_template: arcface_112
  ba_landmark_canonical_kv_grid_cells: 24
  ba_landmark_canonical_kv_rms_match: true
  ba_landmark_canonical_kv_fallback: cl27
  ba_landmark_canonical_kv_min_confidence: 0.80

datasets:
  train:
    cosmic_large_adapted:
      return_reference_landmarks_5: true
      reference_landmark_detector: insightface_existing_detection

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/canonical_kv/applied_fraction
    - ba/canonical_kv/landmark_confidence
    - ba/canonical_kv/native_vs_canonical_cosine
    - ba/canonical_kv/correction_rms
  experiment_comment: >-
    CL41 vs CL27 adds only a parameter-free five-landmark canonicalization of
    reference K/V before target-query attention; invalid landmarks fail closed
    to exact CL27.
```

## 8.5 Arm-specific gate

At 4k, require a positive correct-versus-shuffled reference gap on profile/three-quarter and expression-heavy subsets. Reject if canonical K/V improves ID by copying the reference pose or if overall text/pose adherence visibly regresses.

# 9. Experiment CL42 - facial-component token memory

## 9.1 Hypothesis

At small target faces, a mouth or eye may occupy fewer than one or two useful spatial tokens. Full spatial reference K/V can therefore dilute the most discriminative details. Add a tiny fixed-size memory with one token each for:

```text
left eye | right eye | nose | mouth | global face
```

Reference landmarks define Gaussian pooling masks on the current reference feature grid. Each pooled token receives a fixed sinusoidal component-type code. The existing target face queries attend to these five tokens through a separate parameter-free attention operation:

```text
T_part[k] = weighted_pool(H_ref, component_mask[k])
Y_part = Attention(Q_target, K(T_part), V(T_part))
R_CL42 = R_CL27 + 0.15 * rms_match(Y_part, R_CL27)
```

The spatial reference lane remains untouched. The new memory adds fine-detail retrieval; it does not replace full-face context.

This is not CL16 clean multi-scale memory. CL16 stored broad clean spatial features at several U-Net scales and underperformed. CL42 stores only five semantically indexed component summaries at the same active CL27 layers.

## 9.2 Exact code plan

1. Reuse the CL41 landmark metadata path, but implement CL42 independently.
2. Add `component_token_memory.py` with:
   - landmark-to-Gaussian component masks;
   - weighted pooling with minimum-mass checks;
   - fixed sinusoidal type codes;
   - a five-token attention helper using existing Q/K/V projections.
3. Apply only at `up_blocks.0` and `up_blocks.1`.
4. Use a fixed scale of 0.15 and RMS matching in the first arm. Do not add a learned gate.
5. If landmarks are invalid, skip the component correction and use exact CL27.
6. Log attention mass per component and correction RMS.

## 9.3 Activation tests

- Moving only the mouth landmark changes only the mouth token pooling mask.
- Empty component mask does not produce NaNs.
- Small face synthetic test still returns five finite tokens.
- Swapping reference eyes changes eye-token attention but not global token construction.
- Component attention must not collapse entirely to the global token.

## 9.4 YAML - `src/configs/CL42_cosmic_component_token_memory_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true

model:
  ba_component_token_memory_enabled: true
  ba_component_token_memory_groups: [up_blocks.0, up_blocks.1]
  ba_component_token_memory_components:
    [left_eye, right_eye, nose, mouth, face_global]
  ba_component_token_memory_scale: 0.15
  ba_component_token_memory_sigma_cells: 1.75
  ba_component_token_memory_type_encoding: fixed_sincos
  ba_component_token_memory_rms_match: true
  ba_component_token_memory_fallback: cl27
  ba_component_token_memory_min_confidence: 0.80

datasets:
  train:
    cosmic_large_adapted:
      return_reference_landmarks_5: true
      reference_landmark_detector: insightface_existing_detection

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/component_memory/applied_fraction
    - ba/component_memory/left_eye_mass
    - ba/component_memory/right_eye_mass
    - ba/component_memory/nose_mass
    - ba/component_memory/mouth_mass
    - ba/component_memory/global_mass
    - ba/component_memory/correction_rms
  experiment_comment: >-
    CL42 vs CL27 adds only a five-token facial-component memory pooled from
    reference features and queried by the existing target face queries.
```

## 9.5 Arm-specific gate

On the supplementary small-face subset, require a positive median ID delta versus CL27 and no TOPIQ-face drop greater than 0.01. On the standard fixed 96, reject if mouth/eye tokens cause duplicated or asymmetrical facial parts.

# 10. Experiment CL43 - ID-adaptive modulation of the routed delta

## 10.1 Hypothesis

CL27 has a spatial reference route and PhotoMaker ID tokens, but the spatial message is not directly calibrated to the subject's face-recognition embedding. Use the existing normalized 512-D InsightFace vector to modulate the routed reference delta:

```text
h_id = SiLU(W1(normalize(e_id)))
gamma, beta = W2(h_id)

R_hat = LayerNorm(R_CL27)

R_CL43 =
    R_CL27
    + scale(t) * (
          gamma * R_hat
        + beta
      )
```

`W2` is exactly zero-initialized. The active arm starts at exact CL27, but the modulation output gets first-step gradients. The scale is bounded at 0.20 and ramps from 1k to 6k.

Unlike simply increasing PhotoMaker strength, this modulation acts only on the explicit spatial-reference message and is conditioned on a biometric identity representation.

## 10.2 Exact code plan

1. Ensure the existing dataset/inference 512-D `id_embeds` tensor is passed through `branched_runtime.py` to every active processor.
2. Add a registered `IDAdaptiveModulation` module, preferably rank/bottleneck 32.
3. Instantiate only in `up_blocks.0` and `up_blocks.1`.
4. Normalize the ID vector in FP32; cast the generated modulation to the processor dtype.
5. Zero-initialize the final projection, not the whole module.
6. Add role classification and exact contract generation.
7. Log gamma/beta/output RMS and per-site gradient norms.

## 10.3 Activation tests

- Same image/reference with a different ID vector changes the modulation after training.
- Zero final projection gives exact CL27.
- First update reaches final projection.
- By 1k, at least 80% of selected processors have nonzero modulation output and gradient.
- Shuffled raw ID vector must hurt ID more than correct ID, while a shuffled spatial reference remains separately measurable.

## 10.4 YAML - `src/configs/CL43_cosmic_id_adaptive_modulation_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true
contract_generation_required: true

model:
  ba_id_adaptive_modulation_enabled: true
  ba_id_adaptive_modulation_groups: [up_blocks.0, up_blocks.1]
  ba_id_adaptive_modulation_embedding_dim: 512
  ba_id_adaptive_modulation_bottleneck: 32
  ba_id_adaptive_modulation_target: routed_reference_delta
  ba_id_adaptive_modulation_scale_max: 0.20
  ba_id_adaptive_modulation_normalize_id: true
  ba_id_adaptive_modulation_zero_init_output: true
  ba_id_adaptive_modulation_ramp_start_step: 1000
  ba_id_adaptive_modulation_ramp_end_step: 6000

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: false

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/id_modulation/gamma_rms
    - ba/id_modulation/beta_rms
    - ba/id_modulation/output_rms
    - ba/id_modulation/active_fraction
  experiment_comment: >-
    CL43 vs CL27 adds only zero-initialized AdaLN-style modulation of the
    routed spatial-reference delta from the existing 512-D InsightFace ID
    embedding.
```

## 10.5 Arm-specific gate

Reject at 4k if the candidate behaves like a global face-strength increase: stronger identity with worse pose, repeated facial texture, reduced text alignment, or no separation between correct and shuffled spatial references.

# 11. Experiment CL44 - semantic high-frequency window gate

## 11.1 Hypothesis

CL28/CL34 show that another static learned schedule is unlikely to help. A useful gate should vary **by sample** and should target the risky lane: high-frequency reference detail.

For each selected layer:

```text
agreement =
    cosine(
      masked_mean(H_target_face),
      masked_mean(H_reference_face)
    )

w_time =
    smooth_window(denoising_progress; start=0.20, end=0.85)

w_agree =
    sigmoid((agreement - 0.15) / 0.08)

high_scale =
    clamp(
      min_scale
      + (max_scale - min_scale) * w_time * w_agree,
      0.60,
      1.15
    )

R_CL44 =
    R_low_CL27
    + high_scale * R_high_CL27
```

Low-band structure remains exactly CL27. High-frequency identity is reduced when the reference is semantically mismatched or the denoising step is outside the useful detail window, and can be modestly boosted when agreement is high.

This is parameter-free. Thresholds should be checked in a telemetry-only CL27 replay before launch; do not silently tune them on final validation outcomes.

## 11.2 Exact code plan

1. Add a helper beside `_apply_temporal_frequency_router`.
2. Use the already available target/reference face tokens and normalized denoising progress.
3. Stop-gradient the agreement statistic.
4. Apply only to the high-frequency routed component.
5. Log agreement, time weight, high scale, and top-object versus visible-face scale during synthetic occlusion.
6. No new parameters or data.

## 11.3 Activation tests

- At early and final denoising progress, scale approaches the configured minimum.
- In-window identical features approach the maximum.
- Orthogonal features remain near the minimum.
- Low-band output is bitwise/tolerance-equal to CL27.
- No NaNs when face masks are tiny; fail closed to scale 1.0 below minimum valid area.

## 11.4 YAML - `src/configs/CL44_cosmic_semantic_window_gate_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true

model:
  ba_semantic_window_gate_enabled: true
  ba_semantic_window_gate_groups: [up_blocks.0, up_blocks.1]
  ba_semantic_window_gate_apply_low_band: false
  ba_semantic_window_gate_apply_high_band: true
  ba_semantic_window_gate_progress_start: 0.20
  ba_semantic_window_gate_progress_end: 0.85
  ba_semantic_window_gate_progress_temperature: 0.08
  ba_semantic_window_gate_agreement_threshold: 0.15
  ba_semantic_window_gate_agreement_temperature: 0.08
  ba_semantic_window_gate_min_scale: 0.60
  ba_semantic_window_gate_max_scale: 1.15
  ba_semantic_window_gate_stopgrad_agreement: true

trainer:
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/semantic_window/agreement
    - ba/semantic_window/time_weight
    - ba/semantic_window/high_scale
    - ba/semantic_window/object_minus_visible_scale
  experiment_comment: >-
    CL44 vs CL27 adds only a parameter-free sample-conditioned gate on the
    high-frequency reference lane, using denoising progress and masked
    target/reference agreement.
```

## 11.5 Arm-specific gate

At 4k, require lower high-frequency energy on synthetic top objects than CL27, no reduction of visible-face high-frequency energy greater than 15%, and no aggregate ID loss greater than 0.005.

# 12. Experiment CL45 - asymmetric BA-only gradient surgery

## 12.1 Hypothesis

CL27's hard-case auxiliary objective helps at 16k but the run later degrades. The main diffusion reconstruction and the frequency-surface objective may sometimes push BA parameters in opposing directions. Instead of changing architecture or loss weights, project the auxiliary gradient only when it conflicts with the primary gradient.

For BA parameters `theta_BA`:

```text
g_main = grad(L_diffusion, theta_BA)
g_aux  = grad(L_frequency_surface, theta_BA)

if dot(g_main, g_aux) < 0:
    g_aux =
        g_aux
        - dot(g_aux, g_main)
          / (||g_main||^2 + eps)
          * g_main

g_total_BA = g_main + g_aux
```

The main gradient is never projected. Non-BA PhotoMaker/generic adapter parameters use the existing summed loss.

This arm has exact CL27 inference and the exact trainable parameter contract.

## 12.2 Exact code plan

1. Add `src/optimization/ba_pcgrad.py`.
2. Build the BA parameter list from the existing role/allowlist system; assert that it is nonempty and has the expected CL27 tensor count.
3. In `train.py`, retain separate `loss_diffusion` and `loss_ba_frequency_surface`.
4. Use `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` on BA parameters to obtain detached gradient vectors.
5. Implement an Accelerate/DDP-safe gradient correction. A practical pattern is:
   - compute the ordinary summed loss;
   - construct a linear surrogate whose derivative equals the desired detached correction on BA parameters;
   - call the existing single `accelerator.backward()` so DDP synchronization and gradient accumulation remain under the current framework.
6. Unit-test equality between the surrogate gradient and a direct manual PCGrad calculation on a toy model.
7. Log cosine, conflict fraction, projection norm, and main/aux norms.
8. Expect materially higher backward cost; run a 200-step throughput probe before submitting the full job.

## 12.3 Activation tests

- Positive gradient cosine: exact ordinary summed gradient.
- Negative cosine: projected auxiliary dot primary is approximately zero.
- Main gradient unchanged.
- Non-BA gradients unchanged.
- Gradient accumulation over two microbatches matches the non-accumulated reference within tolerance.
- Two-GPU DDP gradients match one-GPU effective-batch gradients within the established project tolerance.

## 12.4 YAML - `src/configs/CL45_cosmic_ba_pcgrad_24k.yaml`

```yaml
defaults:
  - CL27_cosmic_frequency_surface_energy_24k
  - _self_

implementation_blueprint_do_not_launch: true

trainer:
  ba_pcgrad_enabled: true
  ba_pcgrad_scope: branched_sa_only
  ba_pcgrad_primary_loss: loss_diffusion
  ba_pcgrad_aux_losses: [loss_ba_frequency_surface]
  ba_pcgrad_mode: asymmetric_project_aux_onto_primary_normal_plane
  ba_pcgrad_interval: 1
  ba_pcgrad_eps: 1.0e-12
  ba_pcgrad_accumulation_safe: true
  active_grad_norm_mode: requested_only

expected_trainable_contract:
  enabled: true
  total_tensors: 2240
  total_parameters: 219217920
  optimizer_tensors: 2240
  optimizer_parameters: 219217920
  categories:
    branched_sa_r128:
      name_substring: ".attn1.processor."
      tensors: 840
      parameters: 127795200
    generic_effective_adapter_r32:
      name_substring: ".lora_adapter."
      tensors: 700
      parameters: 30474240
    photomaker_default_effective_adapter_r64:
      name_substring: ".default."
      tensors: 700
      parameters: 60948480

writer:
  loss_names:
    - loss
    - loss_ba_aux
    - loss_ba_frequency_surface
    - ba/pcgrad/gradient_cosine
    - ba/pcgrad/conflict_fraction
    - ba/pcgrad/projection_norm
    - ba/pcgrad/main_norm
    - ba/pcgrad/aux_norm
  experiment_comment: >-
    CL45 vs CL27 changes only optimization: when the CL27 frequency-surface
    gradient conflicts with the main diffusion gradient on BA parameters,
    project the auxiliary gradient and leave the main gradient untouched.
```

## 12.5 Arm-specific gate

Before 1k, require a nontrivial but not dominant intervention rate, provisionally 5%-60% of logged windows. If conflicts are below 5%, CL45 is effectively a no-op and should be stopped. If above 60%, inspect whether the CL27 auxiliary is fundamentally misweighted before continuing.

# 13. Comparative architecture and loss table

| Arm | Inference change | New trainable params | Extra metadata | New training objective | Expected main benefit |
|---|---|---:|---|---|---|
| CL38 | No | 0 | Existing semantic occluder mask | Native top/contact anchor | Anti-deletion |
| CL39 | Yes | 0 | None | None | Query-level native fallback |
| CL40 | Yes | Yes, rank-32 | None | None beyond diffusion | Identity-motion separation |
| CL41 | Yes | 0 | Reference 5-point landmarks | None | Geometry normalization |
| CL42 | Yes | 0 | Reference 5-point landmarks | None | Small-face/local detail |
| CL43 | Yes | Yes, bottleneck-32 | Existing 512-D ID vector | None | Subject-specific calibration |
| CL44 | Yes | 0 | None | None | Sample/time-specific high-band control |
| CL45 | No | 0 | None | Gradient projection only | Longer-run stability |

Approximate engineering overhead, to be measured rather than assumed:

| Arm | Expected training overhead | Expected inference overhead |
|---|---:|---:|
| CL38 | Low | None |
| CL39 | Low-medium | Low |
| CL40 | Medium | Medium |
| CL41 | Medium | Medium |
| CL42 | Low-medium | Low |
| CL43 | Low | Low |
| CL44 | Very low | Very low |
| CL45 | High, potentially 1.6-2.2x backward | None |

# 14. Recommended build and run order

## Wave 1 - highest information per GPU-hour

1. **CL38** - lowest-risk direct response to CL33's deletion shortcut.
2. **CL39** - strongest architectural fit to target-only occluders.
3. **CL44** - low-cost sample-conditioned correction to the risky high band.
4. **CL41** - geometry intervention with clear hard-case causal tests.

Run each through unit tests, a 200-step smoke, 1k activation, and 4k fixed-panel/intervention evaluation before submitting more.

## Wave 2 - identity capacity

5. **CL43** - strongest broad ID capacity hypothesis, but requires exact contract and active-gradient audit.
6. **CL42** - local detail/small-face arm, ideally after the shared landmark metadata path is proven by CL41.

## Wave 3 - higher-risk mechanisms

7. **CL40** - only after a toy and fixed-checkpoint diagnostic shows that the projection removes motion without erasing identity.
8. **CL45** - only after logging ordinary CL27 main-versus-aux gradient cosine on a replay. If conflicts are rare, do not train it.

# 15. Implementation acceptance checklist

An implementation agent should not mark an arm ready until all relevant boxes are checked.

## Code and config

- [ ] Feature defaults off in the model constructor.
- [ ] YAML inherits exactly `CL27_cosmic_frequency_surface_energy_24k`.
- [ ] Only the named critical change is enabled.
- [ ] No accidental branched cross-attention.
- [ ] Runtime invariants remain `PAR=0`, `mix=false`, `ref_kv_weight=1`.
- [ ] No lazy trainable module creation.
- [ ] Exact trainable contract sealed, or explicit contract-generation guard remains.
- [ ] Writer loss/telemetry names exist and are finite.

## Correctness

- [ ] Disabled path equals CL27.
- [ ] Active path changes the intended tensor only.
- [ ] All mask resizing uses 2-D interpolation on the real token grid.
- [ ] Tiny/empty masks are finite and fail closed.
- [ ] FP32 control calculations cast back safely.
- [ ] Save/load round trip preserves output.
- [ ] One-GPU and two-GPU smoke pass.

## Scientific activity

- [ ] Correct reference differs from shuffled/zero reference.
- [ ] New trainable tensors receive sustained gradients.
- [ ] Gate/mass is in an interpretable range.
- [ ] No ordinary-glasses deletion.
- [ ] No detector/topology shortcut.
- [ ] Fixed-96 metrics joined by numeric image index.
- [ ] 16k and 18k gates applied before 24k.

# 16. Ideas deliberately excluded from this suite

| Idea | Why excluded now |
|---|---|
| Clean multi-scale reference memory | CL16 underperformed and did not close the identity gap |
| Dense learned semantic ownership | CL17/CL22 showed deletion and/or broad quality/text damage |
| Same-ID dual-reference consistency | CL18 and CL30 did not improve the strong base |
| Generic residual identity CA/InfuseNet clone | Historical residual-SA and CL21 could be ignored or were negative on the strong route |
| Boundary distillation | CL24 did not transfer topology |
| Global/pathwise ArcFace reward | CL25 and CL36 were negative or inactive |
| Anchored high-resolution ROI branch | CL26 was neutral/negative |
| Learned static frequency endpoints | CL28 and CL34 were negative |
| Wrong-ID/low-band contrastive objective | CL29 damaged eyewear and CL30 positive-only did not recover the base |
| Attention-gated DINO patch reward | CL35 gate was mostly inactive |
| Small-face ROI teacher | CL37 underperformed |
| Combining CL33 with another new module in the same arm | Would break one-change attribution; CL38 instead isolates the anti-deletion mechanism |

# 16.1 Suggested launch command after implementation approval

Use the repository's existing Accelerate configuration and change only the config name:

```bash
accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  train.py \
  --config-name=CL38_cosmic_visibility_ownership_v2_24k
```

Replace the config name for CL39-CL45. Preserve the existing cluster/job wrapper, environment, batch size, validation schedule, Comet project, and fixed manual-validation inputs. The `implementation_blueprint_do_not_launch` guard is intentional; the implementation agent should remove it only in the same commit that adds passing tests and, for CL40/CL43, a generated exact trainable contract.

# 17. Key references

## Repository evidence

- [LATEST handoff](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/docs/handoffs/LATEST.md)
- [CL30-CL37 completed results and base decision](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-19_cl30_cl37_completed_results_and_base_decision.md)
- [CL27/CL29 versus CL23 report](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-17_cl27_cl29_vs_cl23_visual_results_and_next_experiments.md)
- [CL15-CL20 results](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-13_cl15_cl20_results_cl19_next_architecture.md)
- [CL21-CL26 current results](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-14_cl21_cl26_current_results_cl23_fairness_and_next_experiments.md)
- [Residual SA-v2 failure analysis](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/analysis/2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md)
- [CL27 YAML](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/src/configs/CL27_cosmic_frequency_surface_energy_24k.yaml)
- [Current attention processor](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py)
- [Current branched runtime](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/src/model/photomaker_branched/branched_runtime.py)
- [Training integration](https://github.com/kolyangg/rsrch/blob/test/diffusion_template/src/model/photomaker_branched/lora2.py)

## Primary papers

- DynamicID: https://arxiv.org/abs/2503.06505
- InfiniteYou: https://arxiv.org/abs/2503.16418
- FaceCrafter: https://arxiv.org/abs/2505.15313
- LaTo: https://arxiv.org/abs/2509.25731
- DiffSwap++: https://arxiv.org/abs/2511.05575
- ConsistentID: https://arxiv.org/abs/2404.16771
- AnyPhoto: https://arxiv.org/abs/2603.14770
- InsHuman: https://arxiv.org/abs/2605.07402
- MagicMakeup: https://arxiv.org/abs/2607.20924
- Beyond Facial Consistency: https://arxiv.org/abs/2607.25622
- WithAnyone: https://arxiv.org/abs/2510.14975
- Gradient Surgery for Multi-Task Learning: https://arxiv.org/abs/2001.06782

# 18. Final recommendation

The **most likely near-term winner is CL38 or CL39**:

- CL38 is the cleanest response to the exact shortcut exposed by CL33 and has no inference or parameter risk.
- CL39 is the strongest architectural response to the deeper problem: a face query should be able to say that a target-owned object has no corresponding reference token and keep native target self-attention.

CL41 and CL43 have the highest upside for a broader identity gain beyond the hard occlusion cases. CL42 is the best targeted small-face/local-detail arm. CL44 is the cheapest visual-stability experiment. CL40 and CL45 are valuable but should be treated as higher-risk research arms, not launched in parallel before their activity diagnostics are proven.

The promotion rule remains strict: **CL27 at 16k stays the base until a candidate improves or matches aggregate identity, preserves ordinary glasses and prompted occluders, does not worsen Crying/Marion materially, and maintains face topology and quality.**
