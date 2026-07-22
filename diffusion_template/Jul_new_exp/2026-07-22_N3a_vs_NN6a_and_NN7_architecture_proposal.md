# N3a versus NN6a, and the proposed NN7 architecture

**Date:** 22 July 2026  
**Project:** PhotoMaker conditional generation with branched attention  
**Architectures compared:** N3a initial full-spatial branched attention and current NN6a factorized clean identity-only residual  
**Proposed next architecture:** **NN7 — Pose-Locked Correspondence Branch Takeover**  
**Repository mutation:** none

![NN7 architecture scheme](sandbox:/mnt/data/2026-07-22_NN7_pose_locked_correspondence_branch_takeover.png)

A vector version is also provided alongside the PNG:

```text
2026-07-22_NN7_pose_locked_correspondence_branch_takeover.svg
```

---

## 1. Executive assessment

N3a and NN6a occupy opposite ends of the same design space.

- **N3a gives the reference branch too much ownership without correspondence.** It proves that target-query/reference-KV attention can make the face come from the reference, but it copies reference pose, layout, expression, lighting, hair, and occlusion together with identity. Its strong face change is real, but its ownership rule is ill-posed.
- **NN6a protects target geometry so aggressively that it no longer tests the original spatial branched-attention hypothesis.** It is a safe, clean identity-token residual adapter implemented inside branched self-attention. It may improve identity, but the noised spatial reference branch—the defining N3a mechanism—is absent.

The original idea therefore has **not** been disproved. What has been disproved is the unsafe version:

```text
unaligned full reference grid
+ reference as the only target-face attention candidate
+ the same rule at all self-attention layers
+ no independent PhotoMaker baseline
+ no visibility, part, pose, or confidence control
```

The useful N3a idea is narrower:

> A target-coordinate face query should be able to retrieve rich spatial identity evidence from a reference and, where the match is valid, let that reference evidence own a substantial part of the face update.

NN7 restores that mechanism. It does **not** restore N3a's full-grid, all-layer, absolute replacement. The new design adds the missing conditions under which strong reference ownership can be safe:

1. lock the target geometry to an ordinary PhotoMaker teacher pass;
2. use clean spatial reference features rather than a noised evolving reference latent;
3. align reference features to target pose and visibility;
4. restrict each query to the corresponding semantic part and local neighborhood;
5. retain an explicit target candidate;
6. use a clean identity-token candidate when spatial correspondence is unreliable;
7. allow high reference ownership only inside trusted visible identity regions;
8. keep hairline, jaw boundary, neck, accessories, hands, and other occluders target-owned;
9. preserve ordinary PhotoMaker epsilon outside the trusted core;
10. train with causal A/B/null identity supervision and explicit geometry preservation.

This is intentionally braver than NN5/NN6: the branch is allowed to become the majority owner of visible inner-face identity details. The safety comes from **alignment and ownership policy**, not from making the residual too small to matter.

---

## 2. What N3a actually demonstrated

### 2.1 The positive result

N3a demonstrated something later runs often do not:

```text
the reference branch can materially own the generated face
```

Its target-face queries use reference-face K/V, and the target-face self-attention candidate is absent. The first half of the doubled U-Net is returned directly, without an independent PhotoMaker epsilon merge. Consequently, when the branch activates, the generated face can move far from the PhotoMaker baseline and visibly inherit the reference.

That is scientifically useful evidence. The branch is not merely perturbing texture; it can change face structure and apparent identity strongly.

### 2.2 The negative result

N3a's mechanism has no way to distinguish:

```text
identity evidence
from
reference pose, expression, crop layout, lighting, hair, accessories, or occlusion
```

For a target query corresponding to a profile eye, the reference memory may contain a frontal eye, mouth, hair, glasses, or a zero-valued outside-bbox position. The attention result nevertheless receives absolute ownership of the target face update.

The key failures follow directly:

- **pose overwrite:** target yaw and expression are replaced by reference geometry;
- **feature duplication:** target and reference eyes, mouth, glasses, or hair survive simultaneously;
- **occluder collision:** target hands, goggles, hats, or hair conflict with reference facial parts;
- **boundary mismatch:** reference face width, jaw, hairline, and neck do not fit the target head;
- **error compounding:** the same unsafe decision is repeated across all 70 self-attention layers and most denoising steps;
- **color/collage failure:** trainable split cross-attention amplifies the unstable reference stream;
- **no fallback:** a bad reference match cannot defer to target self-attention.

N3a therefore has high **reference authority**, but low **reference validity**.

---

## 3. What NN6a changes

NN6a removes the spatial lane and uses two clean PMv2 identity tokens:

```text
C_id     = Attention(Q_target, K_id(T_ref),  V_id(T_ref))
C_idnull = Attention(Q_target, K_id(T_null), V_id(T_null))

delta_id = Connector_id(C_id - C_idnull)
```

The residual is then:

- zero-initialized;
- gated;
- RMS capped;
- applied only in a feathered target-face core;
- installed only at `up_blocks.0.attn1`;
- protected by an independent ordinary PhotoMaker epsilon outside the core.

Split branched cross-attention, pose adaptation, and CA face mixing are disabled.

### 3.1 Why NN6a is much safer

NN6a removes nearly every mechanism that produced N3a's artifacts:

- no reference coordinate grid;
- no reference pose or crop layout in the target attention lane;
- no reference noise;
- no absolute face replacement;
- no all-layer intervention;
- no trainable split cross-attention;
- target self-attention is always present;
- the output is bounded and spatially anchored;
- the final epsilon is exactly ordinary PhotoMaker outside the core.

The expected result is strong target alignment, stable neck/body attachment, and few visible artifacts.

### 3.2 Why NN6a may remain visually weak

NN6a also removes the main source of fine-grained spatial identity evidence.

Two PMv2 tokens can express global identity, but they do not provide a direct spatial bank for:

- eye shape and spacing;
- brow contour;
- nose and nostril structure;
- mouth shape;
- asymmetric facial details;
- local proportions and texture.

More importantly, NN6a's identity signal passes through:

```text
2 tokens
→ low-rank K/V projections
→ a rank-16 connector
→ a scalar gate
→ an RMS cap
→ one up-block site family
```

This is a conservative correction path, not reference ownership comparable to N3a.

NN5a's results already showed the characteristic safe-but-weak regime: the face remained aligned and artifact-free, but reference swaps produced only minor, often non-identity changes. NN6a may improve semantic purity, but it is still architecturally designed as a bounded correction to PhotoMaker.

---

## 4. Side-by-side comparison

| Dimension | N3a | NN6a | Consequence |
|---|---|---|---|
| Reference memory | Full noised spatial VAE/U-Net reference stream | Two clean PMv2 identity tokens | N3a is rich but entangled; NN6a is clean but highly compressed |
| Core interaction | Target face Q attends reference face K/V | Target Q attends identity-token K/V | Only N3a preserves the original spatial branched-attention premise |
| Target-face fallback | Absent | Ordinary target self-attention is the baseline | N3a must use bad matches; NN6a can preserve target geometry |
| Reference authority | Absolute replacement of face attention | Bounded additive residual | N3a changes faces strongly; NN6a may barely move them |
| Alignment | Bbox only; separate target/reference coordinates | No spatial reference coordinates | N3a copies pose/layout; NN6a avoids that problem rather than solving it |
| Visibility / occlusion | None | Spatial lane absent | N3a collides with hands/hair/glasses; NN6a keeps target ownership |
| Layer scope | All 70 self-attention sites | `up_blocks.0.attn1` only | N3a compounds geometry errors; NN6a can be washed out downstream |
| Cross-attention | Split target/reference CA at all sites | Branched CA disabled | Disabling split CA is a clear safety improvement |
| Reference noise | Independent, fixed during trajectory | None in identity lane | N3a carries stochastic nuisance; NN6a is deterministic with respect to identity memory |
| Face mask | Hard bbox routing | Feathered inner core | NN6a handles boundaries much better |
| Output baseline | Direct target half from doubled U-Net | Independent ordinary PhotoMaker epsilon outside core | NN6a has explicit preservation |
| Training attribution | Diffusion MSE, optional absolute ID loss | Exact matched/wrong/null counterfactual identity supervision | NN6a asks a more causal question |
| Typical behavior | Strong reference-looking face, poor pose and artifacts | Target-aligned, safe, often close to PhotoMaker | Authority and validity are inverted |

---

## 5. Does the original branched-attention idea still hold?

### 5.1 Yes, at the level of the attention equation

The attractive idea remains:

```text
A_ref(i) = Attention(
    Q_target(i),
    K_reference,
    V_reference
)
```

Target queries preserve target-coordinate intent. Reference K/V retain richer identity evidence than a few global tokens.

This mechanism is worth preserving.

### 5.2 No, at the level of N3a's ownership policy

The following rule should not be restored:

```text
A_face = A_ref
```

unless the reference memory has already been registered to target geometry and the query has a valid visible correspondence.

A bbox cannot provide that guarantee.

### 5.3 NN6a is no longer a full test of spatial branched attention

NN6a is still a branched processor implementation, but its active lane is closer to a decoupled identity adapter:

```text
target Q
→ clean identity-token K/V
→ bounded residual
```

The evolving spatial reference branch is absent. Therefore:

> If NN6a fails, it does not prove that target-query/spatial-reference-KV attention is ineffective. It proves that a two-token, low-authority identity residual at the selected sites is insufficient.

The scientifically clean next question is not “return to N3a unchanged.” It is:

> Can a geometrically registered, visibility-aware spatial reference candidate take substantial ownership of the inner face while the target candidate owns pose, boundaries, and occluders?

NN7 is designed to answer exactly that.

---

## 6. NN7 — Pose-Locked Correspondence Branch Takeover

## 6.1 Design objective

NN7 aims for the missing middle:

```text
N3a-level reference ownership
+
NN6-level target geometry and artifact safety
```

The design is intentionally not a tiny residual. In trusted inner-face regions, the reference candidate may become the majority attention owner.

The architecture has four essential ideas:

1. **PhotoMaker geometry teacher:** ordinary PhotoMaker supplies target pose, scene, target self-attention, and the protected epsilon baseline.
2. **Clean aligned reference memory:** spatial reference features are extracted before diffusion noise and warped into target pose.
3. **Dual candidate with strong ownership:** target and reference candidates are computed separately; a learned, bounded gate may favor the reference strongly where correspondence is valid.
4. **Semantic ownership zones:** inner visible identity parts may be branch-owned; boundaries and occluders remain target-owned.

---

## 7. Pass 1: ordinary PhotoMaker as a geometry teacher

At every BA-active denoising step, run the ordinary target-only PhotoMaker U-Net first.

Capture at the selected self-attention sites:

```text
Q_PM,l
K_PM,l
V_PM,l
A_target,l = Attention(Q_PM,l, K_PM,l, V_PM,l)
```

Also retain:

```text
epsilon_PM
```

This is not an extra conceptual cost relative to the current protected-output implementation, which already performs an independent target-only pass for the epsilon anchor. NN7 reuses that pass more productively.

### Why capture the teacher attention tensors?

N3a's target queries can drift after earlier reference injections. NN7 instead lets the reference branch answer the **ordinary PhotoMaker target query**.

That locks the branch to the pose and layout that PhotoMaker intended:

```text
reference identity answers a target-owned geometric question
```

The teacher pass also supplies the exact target candidate, so bad correspondence can fall back without approximation.

### Target geometry map

Obtain target:

- face landmarks or a 3D face mesh;
- face parser regions;
- visibility;
- occluder masks for hands, glasses, hats, hair, and other foreground objects.

Two implementation options:

1. **Preferred inference implementation:** use the same-seed PhotoMaker baseline image or a low-noise decoded teacher prediction and a frozen 3D face/landmark/parser stack.
2. **Lower-dependency implementation:** train a small geometry head on cached `Q_PM`/hidden features, supervised by target-image landmarks and parsing during training.

The first option is easier to debug. The second is faster once validated.

---

## 8. Clean spatial reference memory

Do not restore the noised evolving reference latent.

Instead, extract a clean spatial patch grid from the existing PMv2/CLIP image encoder before the two-token QFormer compression:

```text
F_ref ∈ R[B, Hpatch, Wpatch, D]
```

Also retain the two PMv2 identity tokens:

```text
T_id ∈ R[B, 2, 2048]
```

This gives NN7 both:

- spatial facial evidence;
- a clean global identity fallback.

### Why use PMv2/CLIP patch features?

They are:

- deterministic;
- free of reference diffusion noise;
- already available in the PhotoMaker identity encoder;
- spatially organized;
- semantically cleaner than VAE noise-space features;
- cacheable per reference image.

If PMv2 patch features prove too coarse, DINOv2 face-crop features are a reasonable second encoder, but that should be a separate ablation rather than part of the first implementation.

---

## 9. Geometric registration and local correspondence

Estimate reference landmarks or a 3D face mesh. Align the clean reference feature grid to target geometry.

### Preferred registration

Use a 3D morphable face model or equivalent UV parameterization:

1. fit reference identity shape and reference pose;
2. use target pose and expression from the PhotoMaker teacher;
3. project the reference feature atlas into target coordinates;
4. use z-buffer visibility to mark visible target-reference correspondences.

This produces:

```text
F_ref→target,l
V_visible,l
Part_target,l
Part_reference,l
```

### Practical MVP

A piecewise-affine warp over 68 landmarks plus face parsing is simpler:

- triangulate the reference face;
- map triangles to target landmarks;
- warp feature patches rather than RGB pixels;
- suppress triangles that are back-facing or outside the visible target face;
- keep a conservative confidence for large yaw.

### Local deformable attention

Do not let every target query attend the whole face.

For target query \(i\), correspondence predicts reference coordinate \(u_i\). Attend only a local window or a small set of learned offsets around \(u_i\):

```text
A_spatial(i) =
    DeformableLocalAttention(
        Q_PM(i),
        K_ref→target(N(u_i)),
        V_ref→target(N(u_i))
    )
```

Restrict the candidate to the same semantic part, with limited adjacent-part access.

This directly prevents:

- target eye querying reference mouth;
- profile-side query selecting frontal far-side eye;
- face query selecting hair or background;
- full-grid zero-token softmax sinks.

---

## 10. Three attention candidates

At selected self-attention sites:

### Target candidate

```text
A_target = Attention(Q_PM, K_PM, V_PM)
```

This candidate owns target geometry, expression, pose, and scene consistency.

### Spatial reference candidate

```text
A_spatial =
    LocalDeformAttention(
        Q_PM,
        K_ref_aligned,
        V_ref_aligned
    )
```

This is the retained N3a core.

### Clean identity fallback

```text
A_id = Attention(Q_PM, K_id(T_id), V_id(T_id))
```

This is retained from NN6a.

The ID candidate is useful when:

- the corresponding reference side is invisible;
- reference pose is extreme;
- a semantic part is occluded;
- the alignment confidence is low;
- only one reference image is available.

---

## 11. Correspondence confidence

Define a confidence \(c_i\in[0,1]\) per target query and head.

Recommended inputs:

```text
3D / landmark warp validity
× z-buffer visibility
× target semantic-part validity
× reference semantic-part validity
× local feature similarity
× forward/backward correspondence cycle consistency
× target occluder suppression
```

Do not use attention entropy alone. A wrong eye-to-mouth match can be confidently sharp.

Combine the branch candidates:

```text
A_branch(i) =
    c_i · A_spatial(i)
  + (1 - c_i) · A_id(i)
```

High-confidence visible parts use spatial evidence. Low-confidence parts fall back to global identity rather than copying invalid geometry.

---

## 12. Strong but bounded branch ownership

NN7 should use a direct attention-candidate takeover rather than a tiny rank-16 output correction.

Let \(\alpha_{i,h,l,t}\) be a per-query, head, layer, and timestep ownership gate:

```text
alpha =
    alpha_schedule(layer, timestep)
    × correspondence_confidence
    × semantic_ownership_mask
    × sigmoid(gate_MLP)
```

Then:

```text
delta_attention = A_branch - A_target

A_face =
    A_target
    + clip_norm(
        alpha · delta_attention,
        max_ratio(layer, timestep)
      )
```

This is equivalent to a convex mixture when the norm cap is inactive:

```text
A_face = (1 - alpha) A_target + alpha A_branch
```

### Why this is more branch-driven than NN6a

- no rank-16 connector bottleneck is required for the main spatial takeover;
- the branch candidate is full-dimensional;
- `alpha` can approach `0.7–0.8` in trusted regions;
- the reference candidate can become the majority owner of the visible inner face;
- target fallback remains explicit.

### Initialization

Do not start with N3a-level authority. Recommended:

```text
up0 initial alpha ≈ 0.02
up1 initial alpha ≈ 0.05
```

Use a branch-coverage curriculum to increase ownership after alignment begins working.

A zero-initialized auxiliary correction connector may still be used for calibration, but it must not be the only path by which the reference candidate reaches the face.

---

## 13. Semantic ownership zones

A single rectangular bbox is too coarse.

### Reference-eligible inner regions

- visible eye and brow interiors;
- nose interior;
- lips and mouth interior;
- cheeks;
- central forehead;
- visible inner-face skin.

### Target-owned regions

- face boundary ring;
- hairline and hair;
- jaw-to-neck transition;
- ears unless correspondence is very reliable;
- glasses, hats, goggles;
- hands and other foreground occluders;
- clothing and background.

Recommended masks:

```text
M_identity_inner
M_boundary_ring
M_occluder
M_visibility
```

The effective branch mask is:

```text
M_branch =
    M_identity_inner
    × M_visibility
    × (1 - M_occluder)
```

Force:

```text
alpha = 0
```

in the boundary ring and target occluders.

This is the main artifact-control mechanism. The cap is secondary.

---

## 14. Layer and denoising specialization

Do not return to all 70 self-attention sites.

A strong first configuration is:

### `up_blocks.0.attn1`

Purpose:

- broad facial proportions;
- moderate identity shape;
- target pose must remain dominant.

Suggested maximum ownership:

```text
alpha_max_up0 = 0.25–0.35
```

### `up_blocks.1.attn1`

Purpose:

- eye, brow, nose, mouth, and local identity details;
- stronger reference ownership after geometry is established.

Suggested maximum ownership:

```text
alpha_max_up1 = 0.70–0.85
```

### Denoising schedule

For a 50-step schedule:

| Steps | Behavior |
|---:|---|
| 0–9 | text-only SDXL |
| 10–14 | ordinary PhotoMaker |
| 15–24 | geometry-safe branch warmup; up0 low authority |
| 25–44 | identity takeover; up1 high authority in trusted parts |
| 45–49 | reduced authority for final texture and edge stability |

The reference branch should not own coarse down-block or mid-block geometry.

Branched cross-attention remains disabled.

---

## 15. Output ownership

Keep the independent ordinary PhotoMaker epsilon:

```text
epsilon_PM
```

The branched pass yields:

```text
epsilon_BA
```

Final output:

```text
epsilon_out =
    epsilon_PM
    + M_trust · (epsilon_BA - epsilon_PM)
```

where `M_trust` is the feathered union of valid inner-face regions.

This gives:

- exact PhotoMaker outside the trusted core;
- smooth boundary transitions;
- full branch output inside trusted identity regions.

The branch can therefore own the face interior without owning the neck, hair, body, or scene.

---

## 16. Training objectives

NN7 needs supervision for both **identity ownership** and **geometry rejection**.

## 16.1 Exact causal identity pairing

Retain NN5's matched/wrong/null construction:

```text
same target latent
same target diffusion noise
same timestep
same target prompt and PhotoMaker A identity
same target mask
only reference A versus B changes
```

Use:

```text
L_abs_B
L_direction_B_over_A
L_null
```

The decisive result remains whether R1→R2 moves the generated identity toward R2.

## 16.2 Geometry preservation

Use a frozen differentiable geometry network or feature losses:

```text
L_landmark
L_head_pose
L_expression
L_face_parse
```

Compare the generated target to:

- the ground-truth target during training;
- the ordinary PhotoMaker teacher output where appropriate.

The branch must not improve identity by rotating the head or closing/opening the eyes.

## 16.3 Occluder and boundary preservation

Use:

```text
L_boundary
L_occluder
L_outside_core
```

Recommended targets:

- ordinary target/PhotoMaker features in the boundary ring;
- target segmentation and depth ordering;
- zero branch ownership on target occluder pixels.

## 16.4 Correspondence supervision

Use target/reference landmark and part labels to train:

```text
L_warp
L_cycle
L_part_match
L_locality
```

Penalize attention mass outside the predicted correspondence neighborhood.

## 16.5 Same-identity multi-pose invariance

For the same target, sample two references of the same person with different poses:

```text
A_branch(ref_pose_1)
A_branch(ref_pose_2)
```

After alignment, require identity residuals to agree:

```text
L_same_id_pose_invariance
```

This directly teaches the branch to reject reference pose, lighting, and crop.

## 16.6 Branch ownership coverage

A safe architecture can still learn to ignore the reference.

Add a curriculum target for valid inner-face queries:

```text
mean(alpha | trusted visible inner face)
```

Suggested schedule:

```text
steps 0–500:   no minimum
steps 500–2k:  target mean 0.20–0.35
steps 2k–4k:   target mean 0.40–0.60
```

Do not reward high ownership outside valid correspondence regions.

## 16.7 Controlled PhotoMaker-ID competition

The target PhotoMaker identity A is a strong competing signal.

A brave but controlled curriculum is:

1. keep a full-PM matched row;
2. on 30–50% of low-noise counterfactual rows, attenuate only the PhotoMaker identity delta to `0.5`;
3. require the reference-B direction to remain consistent between full and attenuated PM rows;
4. finish with a full-PM-only fine-tuning phase.

This forces the branch to learn identity ownership without changing deployment conditioning.

Do not apply PM attenuation in the first 500 alignment warmup steps.

---

## 17. Recommended first NN7 configuration

```yaml
model:
  ba_architecture: pose_locked_correspondence_takeover

  # Retained N3a core
  ba_spatial_reference_attention: true
  ba_reference_memory: pmv2_clean_patch_grid
  ba_reference_noise: false

  # Target geometry teacher
  ba_teacher_target_attention: true
  ba_teacher_capture_qkv: true
  ba_teacher_epsilon_anchor: true

  # Correspondence
  ba_alignment: uv_or_landmark_piecewise
  ba_local_attention_window: 5
  ba_semantic_part_masking: true
  ba_visibility_masking: true
  ba_occluder_suppression: true

  # Candidate lanes
  ba_target_candidate: true
  ba_identity_token_fallback: true
  ba_identity_null_tokens: 2

  # Strong bounded ownership
  ba_mix_mode: direct_candidate_takeover
  ba_site_policy: up_blocks0_and_1_attn1
  ba_up0_alpha_max: 0.30
  ba_up1_alpha_max: 0.80
  ba_late_alpha_max: 0.35
  ba_up0_delta_rms_cap: 0.25
  ba_up1_delta_rms_cap: 0.45

  # Ownership zones
  ba_target_core_erode_frac: 0.15
  ba_boundary_reference_authority: 0.0
  ba_occluder_reference_authority: 0.0

  # Unsafe N3a paths remain off
  train_branched_ca_lora: false
  disable_branched_ca: true
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false

  ba_output_anchor_mode: base_outside_trusted_core
```

The exact caps should be tuned from processor diagnostics. The important departure from NN6 is that the branch is permitted to be strong **after** alignment.

---

## 18. Implementation map

## 18.1 Reference feature extraction

Extend the PMv2 encoder API:

```python
extract_spatial_patch_tokens(...)
extract_id_tokens(...)
```

Return:

```text
patch grid
patch coordinates
two identity tokens
recognition embedding
```

Likely files:

```text
src/model/photomaker_branched/model_v2_NS.py
src/model/photomaker_branched/lora2_helpers.py
src/pipelines/br_pipeline_helpers.py
```

## 18.2 Geometry and registration module

Create:

```text
src/model/photomaker_branched/face_correspondence.py
```

Responsibilities:

- reference and target landmarks/mesh;
- semantic part maps;
- UV or piecewise-affine warp;
- visibility and occluder masks;
- local attention neighborhoods;
- confidence features.

Keep the first implementation deterministic and inspectable.

## 18.3 Teacher attention capture

Extend the ordinary target-only U-Net pass to capture selected-site:

```text
Q_PM
K_PM
V_PM
A_target
```

The current protected baseline pass should run before the branched pass.

Likely files:

```text
src/model/photomaker_branched/branched_runtime.py
src/pipelines/br_pipeline_helpers.py
```

## 18.4 New processor

Create a separate processor rather than overloading NN6 immediately:

```text
PoseLockedCorrespondenceBranchedAttnProcessor
```

Core methods:

```python
set_teacher_attention(...)
set_reference_memory(...)
set_correspondence(...)
compute_target_candidate(...)
compute_spatial_candidate(...)
compute_identity_fallback(...)
compute_correspondence_confidence(...)
compute_ownership_gate(...)
mix_and_bound(...)
```

Do not instantiate spatial reference modules outside selected up0/up1 sites.

## 18.5 Output merge

Generalize:

```text
base_outside_core
```

to:

```text
base_outside_trusted_core
```

The trusted mask should include visibility, semantic part, and occluder suppression rather than only bbox erosion.

## 18.6 Diagnostics

Capture separately:

```text
A_target
A_spatial
A_id
c_align
alpha
delta_before_cap
delta_after_cap
attention locality
part-crossing fraction
boundary authority
occluder authority
epsilon_pre_anchor
epsilon_post_anchor
```

Also report:

```text
mean alpha in trusted core
fraction alpha > 0.5
mean spatial-vs-ID fallback weight
reference/no-reference and A/B direction
landmark change versus PM
```

---

## 19. Ablation ladder

Do not implement every brave component in one opaque run. Use a staged ladder while keeping the final architecture in view.

### NN7a — dual candidate, no alignment

```text
A_target + packed clean patch reference candidate
up1 only
direct gate
```

Purpose: confirm that clean patch memory can produce stronger reference ownership than two ID tokens. This is not expected to solve hard pose.

### NN7b — semantic part restriction

Add same-part attention and boundary/occluder ownership masks.

Purpose: test whether most duplication and accessory collision disappear without full 3D alignment.

### NN7c — pose-locked local correspondence

Add landmark/UV warp, local deformable windows, and visibility.

This is the primary NN7 architecture.

### NN7d — strong ownership curriculum

Ramp up1 authority toward `0.8` and add branch-coverage loss.

Purpose: verify that the safe aligned branch can visibly own the face.

### NN7e — controlled PM-ID attenuation

Use only if NN7c/d remains directionally correct but too weak under full PhotoMaker A conditioning.

---

## 20. Evaluation requirements

The usual matched-reference validation is not sufficient.

Use the five-condition matrix:

```text
PM0
R1N1
R2N1
R1N2
R2N2
```

For NN7, the clean spatial memory should make N1/N2 reference-noise effects zero or irrelevant.

### Identity success

Require:

- positive target-averaged directional gain;
- bootstrap lower bound above zero;
- positive mean change in similarity to B;
- at least 60% target-positive;
- improvement distributed across identities;
- visible person-specific changes rather than expression-only changes.

### Geometry success

Require:

- head pose near PM0;
- landmark displacement near PM0/R1;
- expression preservation;
- no increased face-detection failures;
- no duplicated landmarks;
- occluders retain target depth ordering.

### Ownership success

Require:

- branch alpha meaningfully above zero in visible inner parts;
- a substantial fraction of trusted queries with `alpha > 0.5`;
- R1→R2 changes in `A_spatial` and final epsilon;
- target boundary authority exactly zero or near zero.

### Visual target

The correct qualitative result is not:

```text
“NN7 looks slightly different from PhotoMaker.”
```

It is:

```text
“The face is recognizably the supplied reference identity,
while head pose, expression, occlusion, hairline, jaw/neck attachment,
body, and scene remain the PhotoMaker target.”
```

---

## 21. Risks

### External geometry model errors

A failed target landmark/mesh fit could misroute the branch.

Mitigation:

- fail closed to target attention;
- expose confidence;
- never allow reference authority where correspondence is invalid.

### Overly conservative confidence

The model may remain close to PhotoMaker.

Mitigation:

- branch-coverage curriculum;
- high up1 authority;
- multi-pose alignment supervision;
- controlled PM-ID attenuation only after direction is correct.

### Overly strong branch

Even aligned reference features can transfer lighting or texture.

Mitigation:

- identity-token modulation;
- same-ID multi-reference invariance;
- target color/illumination preservation losses;
- boundary and occluder target ownership.

### Compute cost

Teacher attention capture plus a branched pass is expensive.

Mitigation:

- the current protected baseline already requires a target-only pass;
- cache clean reference memory once;
- capture only selected sites;
- remove the evolving reference U-Net half;
- after correctness, fuse teacher capture and branch execution more efficiently.

---

## 22. Final recommendation

N3a should not be viewed simply as a failed obsolete architecture. It contains the strongest evidence that spatial target-query/reference-KV interaction can make the face genuinely reference-owned.

NN6a is a valuable safety and causality experiment, but it has moved far enough toward a conventional token adapter that it no longer tests the rich spatial branch premise.

NN7 should restore the original idea under stricter conditions:

```text
retain:
  target query → spatial reference K/V
  explicit face-local branch
  strong reference ownership

replace:
  noised evolving reference grid
  bbox-only correspondence
  all-layer absolute replacement
  split branched cross-attention

add:
  PhotoMaker teacher Q/K/V and epsilon
  clean spatial patch memory
  target-pose registration
  semantic part and visibility masks
  local deformable correspondence
  target candidate and ID-token fallback
  per-query/head/layer/timestep ownership
  geometry, occlusion, and branch-coverage supervision
```

The central NN7 hypothesis is:

> Strong branched attention is not inherently incompatible with target pose. It becomes unsafe when reference authority is granted before correspondence and visibility are established.

NN7 therefore uses **alignment to earn authority**, rather than using a tiny residual to avoid exercising authority.
