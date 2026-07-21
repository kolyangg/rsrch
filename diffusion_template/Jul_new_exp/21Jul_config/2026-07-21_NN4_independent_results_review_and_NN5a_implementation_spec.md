# NN4 independent results review and NN5a implementation specification

**Date:** 21 July 2026  
**Repository branch reviewed:** `main_clean`  
**Repository head reviewed:** `13b56999e55226e216f4ec74d45b35dab63a28e2`  
**NN4 run:** `ba_NN4_causal_null_up0_nfs_1gpu`  
**Primary evidence:** `diffusion_template/Jul_new_exp/21Jul_NN4_results/`  
**Recommended next run:** `NN5a_counterfactual_directional_ppr`

---

## 1. Executive decision

**Stop NN4 in its current form. Do not spend the remaining 20k-step budget on the unchanged objective.**

NN4 is a successful **engineering and safety** experiment but an unsuccessful
**reference-identity control** experiment.

It proves that the protected packed-reference residual design can keep the body,
pose, scene, occluders, and face attachment stable while applying a measurable
face-local residual. It also proves that the learned route is live: reference
content reaches the output, trainable parameters move, gradients are finite, the
learned-null residual is suppressed, and the checkpoint/diagnostic machinery is
working.

It does **not** prove useful identity control. In all four controlled 2k/4k
reference-swap tests, the mean directional identity gain toward the swapped
reference is negative. Three of four confidence intervals exclude zero in the
wrong direction. The least negative result, 4k on the same SDXL base, remains
negative with only 45.3% positive samples. Increasing the residual to 4× makes
the branch visible, but mostly reduces similarity to the original PhotoMaker
identity without moving toward the supplied replacement identity.

The next run should therefore preserve NN4's safe operator and change one
scientific variable:

> Train the branch on a fixed target with both a matched reference and an
> explicitly different-identity reference, and optimize a directional identity
> objective on the different-identity output.

The first NN5 run should **not** simultaneously add a new reference encoder,
semantic face parts, extra U-Net sites, a larger cap, or branched cross-attention.
Those are useful follow-ups only after the current spatial route is given a
causal identity objective and tested cleanly.

---

## 2. Evidence reviewed

### 2.1 Aggregate result files

- `21Jul_NN4_results/NN4_causal_summary.csv`
- `21Jul_NN4_results/NN4_normal_validation_curve.csv`
- `21Jul_NN4_results/NN4_training_window_summary.csv`
- `21Jul_NN4_results/comet_training/metrics_history.json`
- `21Jul_NN4_results/comet_training/metrics_summary.json`
- `21Jul_NN4_results/comet_training/comet_output.log`

### 2.2 Controlled causal tests

Four 96-sample, five-condition tests were run:

1. 2k checkpoint, RealVisXL validation backbone;
2. 2k checkpoint, same SDXL backbone as training;
3. 4k checkpoint, RealVisXL validation backbone;
4. 4k checkpoint, same SDXL backbone as training.

Each test contains:

```text
PM0   = ordinary PhotoMaker / PPR scale 0
R1N1  = matched spatial reference, reference noise seed 1
R2N1  = cyclic wrong-identity spatial reference, same reference noise seed 1
R1N2  = matched spatial reference, reference noise seed 2
R2N2  = cyclic wrong-identity spatial reference, same reference noise seed 2
```

The PPR variants used runtime residual scale 4 to reveal a weak branch. The
target latent, target prompt, target PhotoMaker identity, scheduler, masks,
batch size, and target seed were controlled. The manifests report:

- 96 samples;
- actual batch size 12;
- CFG reference-noise pairing active;
- reference token/pooled-text conditioning neutralized;
- LPIPS available;
- all integrity assertions passed.

The same-SDXL tests are the primary causal evidence because they remove
cross-backbone transfer as an explanation. RealVisXL remains useful as a
deployment-style transfer test.

### 2.3 Visual evidence

The compact bundle includes selected contact sheets spanning all four 96-image
panels. The complete generated images, crops, and heatmaps are preserved under
`diffusion_template/rsrch_21Jul_test/`.

The selected panels are consistent with the quantitative localization evidence:

- pose, body, clothes, hands, framing, and background remain stable;
- faces remain attached to the head;
- no systematic duplicated face plate, displaced jaw, or broad boundary failure
  is apparent;
- the visible changes concentrate on eyes, blinking, mouth/expression, local
  shading, smoothness, and sharpness;
- R1 and R2 generally remain dominated by the target/PhotoMaker identity rather
  than acquiring the swapped identity.

The central failure is therefore semantic, not geometric.

---

## 3. What works

### 3.1 The protected PPR scaffold works as a safety envelope

NN4 retains ordinary target self-attention and adds a bounded residual only
inside a feathered face core. The final output is anchored to an independent
ordinary PhotoMaker prediction outside that core.

This is the strongest positive result in the experiment. The old absolute
reference-owned branch produced pasted geometry and severe face/body mismatch.
NN4 does not. The following design decisions should remain unchanged in NN5a:

- ordinary target self-attention as the base;
- target queries retrieving packed reference-face K/V;
- learned-null candidate through the same reference K/V projections;
- bias-free, zero-initialized connector-up projection;
- gated, RMS-capped additive residual;
- soft inner-core routing;
- exact ordinary PhotoMaker output outside the core;
- `up_blocks.0.attn1` sites only;
- branched cross-attention disabled;
- reference token and pooled-text conditioning neutralized;
- CFG unconditional/conditional reference-noise pairing;
- pose adaptation disabled;
- CA face mixing disabled.

### 3.2 The branch is live

This is not a silent checkpoint, optimizer, restore, or routing failure.

Training summaries show:

- `sa_ref` LoRA norm grows from about `2.58` in the first 2k window to `7.75`
  after 8k;
- connector-up carries most of the trainable gradient;
- total and per-group gradients remain finite and nonzero;
- the run reaches step 9,800 in the exported metrics;
- normal validation reaches 8k;
- the causal test loads the 2k and 4k checkpoints successfully;
- all causal-test integrity checks pass.

### 3.3 Reference content reaches the output

At residual scale 4, changing the reference image has a larger face-region
effect than changing only reference noise:

| Step | Base | Reference/noise core MAE | Reference/noise face LPIPS |
|---:|---|---:|---:|
| 2k | RealVisXL | 1.390× | 1.716× |
| 2k | same SDXL | 1.228× | 1.443× |
| 4k | RealVisXL | 1.476× | 1.912× |
| 4k | same SDXL | 1.205× | 1.382× |

This rules out the simple diagnosis that the reference image is ignored
completely. The spatial route contains real image-dependent signal.

### 3.4 Learned-null suppression works numerically

The null residual remains around `0.6e-6` to `1.4e-6` in the windowed summaries.
The learned-null memory and null-response penalty are active and stable.

### 3.5 Correctness fixes from the NN4 audit are present

The latest processor code correctly:

- treats `raw_delta` as `D(C_ref - C_null)`;
- uses its normalized pre-cap magnitude for the matched/null distance;
- does not subtract the null response twice;
- excludes rows with empty reference ROI or empty target core from auxiliary
  losses;
- rejects empty cores in the core-normalized diffusion loss.

These are not the cause of the failed result.

---

## 4. What does not work

### 4.1 Swapping the reference does not steer identity toward the new person

The decisive metric is the directional identity gain produced by replacing R1
with R2 while holding target conditions and reference noise fixed.

| Step | Base | Mean directional gain toward R2 | 95% bootstrap CI | Positive fraction |
|---:|---|---:|---:|---:|
| 2k | RealVisXL | **-0.00512** | [-0.00888, -0.00155] | 47.9% |
| 2k | same SDXL | **-0.00638** | [-0.00971, -0.00318] | 36.5% |
| 4k | RealVisXL | **-0.00609** | [-0.00959, -0.00266] | 43.2% |
| 4k | same SDXL | **-0.00174** | [-0.00464, +0.00133] | 45.3% |

Three tests are significantly wrong-direction. The fourth is statistically
unresolved, not positive. None clears even the weak criterion of a positive
mean and a majority of positive samples.

### 4.2 Stronger residual authority damages identity rather than transferring it

At scale 4, original-target identity similarity changes as follows:

| Step | Base | PM0 original-ID similarity | Mean PPR similarity | Change |
|---:|---|---:|---:|---:|
| 2k | RealVisXL | 0.52313 | 0.44450 | **-0.07863** |
| 2k | same SDXL | 0.41524 | 0.37246 | **-0.04278** |
| 4k | RealVisXL | 0.52313 | 0.43381 | **-0.08932** |
| 4k | same SDXL | 0.41537 | 0.37123 | **-0.04414** |

This is not a useful identity trade. The output moves away from A but does not
move reliably toward B. Raising the runtime scale, cap, gate, or number of sites
would expose more of the same misdirected residual.

### 4.3 Most of the PPR effect is generic rather than reference-specific

Only about 22–26% of the full scale-4 face-core displacement is associated with
changing the reference image:

| Step/base | Reference effect / total PPR core effect | Noise effect / total PPR core effect |
|---|---:|---:|
| 2k RealVisXL | 22.8% | 16.4% |
| 2k same SDXL | 26.0% | 21.2% |
| 4k RealVisXL | 22.5% | 15.2% |
| 4k same SDXL | 23.9% | 19.8% |

The remaining displacement is shared across reference/noise variants and is
consistent with generic target-query, rendering, expression, or connector
behavior.

### 4.4 Normal validation is flat

| Step | ID similarity | Text similarity |
|---:|---:|---:|
| 0 | 0.522534 | 26.4113 |
| 2k | 0.518964 | 26.4658 |
| 4k | 0.518393 | 26.5298 |
| 6k | 0.519547 | 26.5308 |
| 8k | 0.518131 | 26.5296 |

ID similarity never exceeds the baseline and ends about `0.00440` lower
(`-0.84%`). Text similarity gains about `0.118` and plateaus by 4k. This matches
the near-identical normal-scale validation images.

### 4.5 The matched/null margin is not an identity objective

The margin penalty falls effectively to zero after 2k. This does **not** mean
the matched and null paths collapsed. It means the branch quickly satisfies a
small magnitude-separation hinge.

Once any nontrivial connector response exceeds the `0.02` margin, the term
provides almost no gradient. It does not require:

- identity A and identity B to produce different residuals;
- the difference to align with a face-recognition identity direction;
- the output to move toward the supplied reference;
- pose/expression/lighting variation to be rejected.

For NN5a, set the magnitude-only matched/null margin weight to zero. Keep the
null residual and cap penalties.

### 4.6 The raw residual increasingly pushes against the cap

Windowed `cap_excess` grows from approximately `0.00024` before 2k to
`0.00351` after 8k. The processor logs show many selected sites reaching the
`0.15` cap.

The cap is doing useful safety work. The model is spending capacity increasing
a raw residual whose semantic direction remains wrong. Do not raise the cap.

### 4.7 Same-person reconstruction does not identify the causal source

In NN4, the target image, target PhotoMaker identity, spatial reference, and
decoded identity target describe the same person. A generic face correction can
improve or preserve reconstruction without depending on *which* identity is in
the spatial reference.

The 50% hard PhotoMaker-ID attenuation does not fix this. It increases the need
for some identity correction, but the ordinary objective still never asks:

```text
same target + reference A  -> identity A
same target + reference B  -> identity B
```

That missing counterfactual is the main objective-level failure.

### 4.8 The evolving noised reference remains a nuisance source

Reference-image effects exceed reference-noise effects, but only by 1.2–1.9×.
Noise is therefore still a material contributor to output variation.

This should be controlled in NN5a by using the **same explicit reference-noise
tensor for matched A and wrong B**. A separate two-noise consistency experiment
can follow later; it should not be combined with the first counterfactual run.

---

## 5. Root-cause diagnosis

NN4 has two useful candidates:

```text
C_ref  = Attention(Q_target, K_ref,  V_ref)
C_null = Attention(Q_target, K_null, V_null)
delta  = Connector(C_ref - C_null)
```

The problem is not that `C_ref` is invariant. The problem is that the training
loss can reward any same-person face correction without requiring the useful
part of `delta` to be identity-specific.

The current incentives permit the following shortcut:

1. target queries already encode target pose, prompt, and evolving target face;
2. reference K/V contains identity plus expression, lighting, crop, texture, and
   noising nuisance;
3. the connector learns a broadly useful face correction;
4. the small matched/null hinge is satisfied;
5. the cap keeps the correction safe;
6. ordinary same-person reconstruction never tests reference ownership.

This produces exactly the observed pattern:

- safe localization;
- live gradients;
- growing branch norms;
- measurable reference sensitivity;
- generic expression/texture changes;
- no reliable swapped-identity direction.

---

## 6. Recommended next run

# NN5a: counterfactual directional PPR

NN5a should test one question:

> Can the existing NN4 packed spatial route become identity-causal when trained
> with an explicit fixed-target, wrong-identity reference objective?

Do not add a clean reference encoder or facial-part tokens in this run. If NN5a
fails, NN5b should replace or augment the noised spatial memory with clean
PhotoMaker QFormer/InsightFace identity tokens.

---

## 7. NN5a invariant architecture

Keep the following exactly as in NN4:

```yaml
disable_branched_ca: true
loss_kind: core_normalized

model:
  ba_processor_variant: packed_residual_v1
  ba_site_policy: up_blocks0_attn1
  ba_sa_train_mode: packed_residual
  ba_connector_input_mode: reference_minus_learned_null
  ba_reference_token_mode: packed_bbox_roi
  ba_reference_continuation: frozen_base
  ba_output_anchor_mode: base_outside_core

  ba_cfg_reference_noise_pairing: true
  ba_reference_token_text_mode: zero
  ba_reference_pooled_text_mode: zero

  ba_delta_rms_cap: 0.15
  ba_gate_max: 0.50
  ba_target_core_erode_frac: 0.10

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
```

Keep the current zero initialization, LoRA reference K/V, connector rank, null
memory, strict processor restore, invalid-sample guards, and same-base training.

Remove only the obsolete magnitude-separation objective:

```yaml
model:
  ba_match_null_margin_weight: 0.0
```

Retain:

```yaml
model:
  ba_null_residual_loss_weight: 0.10
  ba_cap_loss_weight: 0.01
  ba_cap_loss_target: 0.12
```

---

## 8. Paired training batch design

Use **one target sample per physical microbatch** and gradient accumulation 2.
This preserves the effective two-target batch of NN4 while making room for a
matched/wrong pair at low timesteps.

For target identity A, the dataset returns:

```text
target image A
matched reference image A'
wrong-identity reference image B
target bbox
matched-reference bbox
wrong-reference bbox
prompt for target A
identity keys for A and B
```

The model samples one target latent, target noise, and timestep, then constructs:

```text
generation rows:
  row 0: target A, matched reference A'
  row 1: the exact same target A, wrong reference B

reference rows:
  row 0: A'
  row 1: B
```

The two generation rows must share:

- target VAE latent;
- target diffusion noise;
- timestep;
- target mask/core;
- prompt text;
- target PhotoMaker identity conditioning A;
- scheduler scaling;
- added SDXL time IDs.

The two reference rows must share the **same sampled reference-noise tensor**.
Only the clean reference latent and reference bbox differ.

At timesteps above the decoded-loss threshold, run the ordinary matched row only.
At low timesteps, run the paired rows.

Recommended threshold:

```yaml
model:
  ba_counterfactual_max_timestep: 300
```

With the current inference-region timestep sampler, this activates the
counterfactual path on roughly 43% of batches.

### Why this batch construction is preferable

With physical target batch 1:

- matched-only U-Net input is `[target A, reference A']` → batch 2;
- paired U-Net input is `[target A, target A, reference A', reference B]`
  → batch 4.

NN4 already used physical target batch 2, which also produced doubled U-Net
batch 4. The paired low-timestep path should therefore remain in the existing
memory envelope.

---

## 9. NN5a loss

Let:

- `eps_match` be row 0 output;
- `eps_swap` be row 1 output;
- `eps_gt` be the target diffusion noise;
- `x0_match` and `x0_swap` be decoded predicted clean images;
- `e_A`, `e_B`, and `e_swap` be normalized frozen face-recognition embeddings;
- `M_core` be the feathered identity core;
- `M_bbox` be the hard target face bbox mask;
- `M_ring = clamp(M_bbox - M_core, 0, 1)`.

### 9.1 Matched reconstruction

Apply the existing core-normalized diffusion MSE only to the matched row:

```text
L_diff =
  sum(M_core * (eps_match - eps_gt)^2)
  / sum(M_core)
```

**Do not apply target-A core diffusion loss to the wrong-reference row.**
Doing so would train the model to ignore B.

### 9.2 Counterfactual absolute identity loss

At `t <= 300`:

```text
L_cf_abs = 1 - cosine(e_swap, e_B)
```

This requires the swapped output to approach the supplied B identity.

### 9.3 Counterfactual directional margin

Use the same semantic direction as the causal evaluation:

```text
direction = cosine(e_swap, e_B) - cosine(e_swap, e_A)

L_cf_dir = relu(margin - direction)^2
```

Recommended margin:

```yaml
model:
  ba_counterfactual_direction_margin: 0.03
```

The absolute term prevents the model from satisfying the margin merely by
moving away from A.

### 9.4 Boundary-ring preservation

Preserve the target-compatible face edge, jaw/neck transition, hairline ring,
and nearby occluders:

```text
L_ring =
  sum(M_ring * (eps_swap - eps_match)^2)
  / sum(M_ring)
```

The ordinary output anchor already preserves everything outside the core
relative to each row's PhotoMaker baseline. The explicit ring loss protects the
transition region where face/body artifacts first appear.

### 9.5 Existing safety losses

Keep:

```text
L_null = connector response to learned null memory
L_cap  = excess raw residual above the pre-cap target
```

### 9.6 Initial weights

Use conservative weights:

```yaml
model:
  use_id_loss: true
  id_loss_weight: 0.025
  id_loss_max_timestep: 300
  id_loss_identity_source: reference

  ba_counterfactual_abs_id_weight: 0.05
  ba_counterfactual_direction_weight: 0.10
  ba_counterfactual_direction_margin: 0.03
  ba_counterfactual_ring_weight: 0.05
  ba_counterfactual_max_timestep: 300

  ba_null_residual_loss_weight: 0.10
  ba_match_null_margin_weight: 0.0
  ba_cap_loss_weight: 0.01
```

Total training objective:

```text
L =
    L_diff
  + 0.025 * L_id_match
  + 0.05  * L_cf_abs
  + 0.10  * L_cf_dir
  + 0.05  * L_ring
  + 0.10  * L_null
  + 0.01  * L_cap
```

These weights put the new semantic term above NN4's nearly inactive
matched/null margin while leaving the diffusion objective dominant.

### 9.7 Target PhotoMaker conditioning

For the first clean test, keep **full target PhotoMaker identity A** on both
matched and swapped rows:

```yaml
model:
  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0
```

This removes NN4's 50% train/inference conditioning mismatch and tests the real
deployment problem: can spatial reference B steer identity while the target
PhotoMaker route still supplies A?

If the branch remains completely unable to move under full PM despite a healthy
directional gradient, a later NN5a-curriculum run may introduce PM scales
`0 → 0.5 → 1`. Do not mix that curriculum into the first causal test.

---

## 10. Concrete code changes

### 10.1 `src/datasets/cosmic.py`

Extend the active `CosmicLargeTrain` implementation.

#### New constructor arguments

```python
return_counterfactual_ref: bool = False
counterfactual_same_class_probability: float = 0.8
counterfactual_max_resample_attempts: int = 20
```

#### Build identity records at initialization

When reading the dataset JSON, store a stable identity key with every record.

Preferred resolution order:

```python
identity_key = (
    img_data.get("identity_id")
    or img_data.get("person_id")
    or img_data.get("id")
    or str(Path((img_data.get("face_paths") or [img_path])[0]).parent)
    or str(img_path)
)
```

Build:

```python
self.identity_records: list[dict]
self.records_by_prompt_class: dict[str, list[dict]]
```

Use the existing `PROMPT_CLASS_RE` to derive a coarse class. Sample B from a
different identity, with 80% probability from the same coarse class and 20%
from the unrestricted pool.

#### New helper

```python
def get_counterfactual_ref_image(
    self,
    matched_img_data: dict,
    matched_identity_key: str,
) -> tuple[Image.Image, list[float], str]:
    ...
```

Requirements:

- `wrong_identity_key != matched_identity_key`;
- valid face image and bbox;
- same crop-margin, flip, and sharpness augmentation as matched references;
- bounded retry count;
- raise an explicit data error if no valid B is found.

#### New returned fields

```python
instance_data["matched_identity_key"] = matched_identity_key
instance_data["counterfactual_identity_key"] = wrong_identity_key
instance_data["counterfactual_ref_images"] = [wrong_ref_image]
instance_data["face_bbox_counterfactual_ref"] = wrong_ref_bbox
```

Add tests that:

- A and B keys are always different;
- bboxes remain in bounds after crop and flip;
- same-class sampling obeys the configured probability statistically;
- dataloader collation preserves the nested image lists.

### 10.2 `src/model/photomaker_branched/lora2_helpers.py`

Split spatial-reference preparation from target PhotoMaker prompt preparation.

Add:

```python
def prepare_spatial_reference_batch(
    model,
    *,
    ref_images,
    face_bbox_ref,
    latent_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
      reference_latents
      reference_masks
      normalized 512-D frozen recognition embeddings
    """
```

Refactor `prepare_branched_training_inputs()` to call this helper for the
matched A reference. Call it a second time for B when counterfactual training is
active.

Do **not** recompute target prompt/PhotoMaker embeddings from B. The target
conditioning must remain identity A.

Add strict checks:

```python
matched_identity_key != counterfactual_identity_key
all masks nonempty
all recognition embeddings finite and nonzero
```

Optionally log cosine similarity between reference A and B. Do not silently
accept identical embeddings.

### 10.3 `src/model/photomaker_branched/branched_runtime.py`

Add an explicit reference-noise override to `two_branch_predict()`:

```python
def two_branch_predict(
    ...,
    reference_noise_override: Optional[torch.Tensor] = None,
    ...
):
```

Behavior:

```python
if reference_noise_override is not None:
    validate shape/device/dtype
    reference_noise = reference_noise_override
else:
    use current cached sampling path
```

For the NN5a pair, create one `[1,C,H,W]` reference-noise tensor and repeat it:

```python
reference_noise_pair = base_reference_noise.repeat(2, 1, 1, 1)
```

Assert exact equality between A and B noise rows. Record hashes in diagnostics.

Do not key the paired path only through mutable `_ref_noise` cache state; the
training comparison should be explicit and testable.

### 10.4 `src/model/photomaker_branched/lora2.py`

Add model parameters:

```python
ba_counterfactual_enabled: bool = False
ba_counterfactual_max_timestep: int = 300
ba_counterfactual_abs_id_weight: float = 0.0
ba_counterfactual_direction_weight: float = 0.0
ba_counterfactual_direction_margin: float = 0.03
ba_counterfactual_ring_weight: float = 0.0
```

Validate that counterfactual training requires:

```text
packed_residual_v1
up_blocks0_attn1
reference_minus_learned_null
disable_branched_ca
reference token text = zero
reference pooled text = zero
base_outside_core
```

#### Forward path

After normal target/matched-reference preparation:

```python
counterfactual_active = (
    self.ba_counterfactual_enabled
    and int(t_scalar.item()) <= self.ba_counterfactual_max_timestep
)
```

If inactive, keep the current matched-only NN4 forward.

If active:

```python
# target tensors are exact duplicates
noisy_pair = torch.cat([noisy_latents, noisy_latents], dim=0)
t_pair = torch.cat([timesteps, timesteps], dim=0)
prompt_pair = torch.cat([prompt_embeds, prompt_embeds], dim=0)
mask_pair = torch.cat([mask4, mask4], dim=0)

# only reference identity changes
reference_pair = torch.cat(
    [matched_reference_latents, wrong_reference_latents],
    dim=0,
)
reference_mask_pair = torch.cat(
    [matched_reference_mask, wrong_reference_mask],
    dim=0,
)

# identical nuisance noise
reference_noise_pair = torch.cat(
    [base_reference_noise, base_reference_noise],
    dim=0,
)
```

Run one paired branched forward and split:

```python
eps_match, eps_swap = noise_pred_pair.chunk(2, dim=0)
```

Return the matched row as the ordinary criterion input:

```python
output["model_pred"] = eps_match
output["target"] = target_noise
```

Compute decoded losses from `eps_match` and `eps_swap` only at low timesteps.

#### Refactor x0 decoding

Extract the duplicated code from `_compute_id_loss()`:

```python
def _decode_predicted_x0(
    self,
    *,
    noise_pred,
    noisy_latents,
    timesteps,
) -> torch.Tensor:
    ...
```

Use it for matched and counterfactual outputs.

### 10.5 `src/loss/id_loss.py`

Keep `IdentityLoss` for matched supervision and add:

```python
class CounterfactualIdentityLoss(nn.Module):
    def forward(
        self,
        generated_images,
        target_bboxes,
        matched_reference_images,
        matched_reference_bboxes,
        wrong_reference_images,
        wrong_reference_bboxes,
        margin: float,
    ) -> dict[str, torch.Tensor]:
        """
        Return:
          absolute_loss
          directional_loss
          sim_to_matched
          sim_to_wrong
          directional_gain
        """
```

Use the same frozen InceptionResnetV1 preprocessing as the current loss.

Equations:

```python
sim_a = (generated_embedding * matched_embedding).sum(-1)
sim_b = (generated_embedding * wrong_embedding).sum(-1)

absolute = (1.0 - sim_b).mean()
directional = F.relu(margin - (sim_b - sim_a)).square().mean()
```

Do not detach `generated_embedding`.

Return per-batch diagnostics. Reject invalid reference crops. Log generated-face
embedding norms and finite checks.

The existing FaceNet loss can reward malformed identity-correlated structure.
NN5a relies on the conservative PPR cap/core and ring loss for safety; any face
detection failure in validation is an immediate stop condition.

### 10.6 `src/trainer/sdxl_trainers.py`

Add aggregation blocks analogous to the existing ID and BA auxiliary losses:

```python
counterfactual_losses = (
    ("ba_counterfactual_abs_id_loss",
     "ba_counterfactual_abs_id_weight",
     "ba_cf/absolute_id"),
    ("ba_counterfactual_direction_loss",
     "ba_counterfactual_direction_weight",
     "ba_cf/directional"),
    ("ba_counterfactual_ring_loss",
     "ba_counterfactual_ring_weight",
     "ba_cf/ring"),
)
```

Also log:

```text
ba_cf/applied_fraction
ba_cf/sim_to_matched
ba_cf/sim_to_wrong
ba_cf/directional_gain
ba_cf/reference_identity_cosine_A_B
ba_cf/reference_noise_equal
```

Assert:

- all losses finite;
- A/B identity keys differ;
- paired target latents/noise/timesteps are exact;
- paired reference noise is exact;
- wrong-reference row never enters the target-A core diffusion loss.

### 10.7 `src/trainer/ppr_reference_noise.py`

The current diagnostic hardcodes scale 4. Replace the global scale values with a
configurable function:

```python
def _variants(scale: float):
    return {
        "PM0": (0.0, "R1", "N1"),
        "R1N1": (scale, "R1", "N1"),
        "R2N1": (scale, "R2", "N1"),
        "R1N2": (scale, "R1", "N2"),
        "R2N2": (scale, "R2", "N2"),
    }
```

New config:

```yaml
ppr_reference_noise_scale: 1.0
```

Use scale 1 for the primary NN5 approval test. Scale 2 is a secondary
sensitivity run. Scale 4 should no longer be the primary acceptance metric.

Add the scale to `manifest.json`, output folder naming, and integrity metadata.

### 10.8 Configuration file

Create:

```text
src/configs/one_id_ba_NN5a_counterfactual_directional_ppr.yaml
```

Suggested content:

```yaml
defaults:
  - one_id_ba_NN4_causal_null_up0
  - _self_

# Approval-stage run, not a 20k commitment.
loss_kind: core_normalized

model:
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

  ba_pm_id_attenuation_probability: 0.0
  ba_pm_id_attenuation_scale: 1.0

  ba_null_residual_loss_weight: 0.10
  ba_match_null_margin_weight: 0.0
  ba_cap_loss_weight: 0.01
  ba_cap_loss_target: 0.12
  ba_delta_rms_cap: 0.15

datasets:
  train:
    cosmic_large:
      return_counterfactual_ref: true
      counterfactual_same_class_probability: 0.8
      counterfactual_max_resample_attempts: 20

dataloaders:
  train:
    batch_size: 1
    grad_accum_enabled: true
    batch_size_eff: 2
```

The launcher may override the server-specific dataset key
(`cosmic_large_neb`, `cosmic_large`, etc.) as today.

### 10.9 Launcher

Create:

```text
jul_serv_runs/start_ba_NN5a_counterfactual_directional_ppr_1gpu.sh
```

Use:

```text
physical batch = 1
effective batch = 2
gradient accumulation = 2
optimizer steps per epoch = 2000
maximum initial budget = 4000
same-SDXL causal validation at 2k and 4k
```

Remove NN4's hard assertion requiring exactly 20,000 steps.

The run should stop automatically or require explicit continuation after 4k.

---

## 11. Required tests before launch

### Dataset tests

1. wrong identity key is never equal to matched identity key;
2. wrong-reference image and bbox survive crop/flip;
3. coarse-class preference works;
4. invalid B candidates are retried and fail explicitly;
5. dataloader batch 1 collates all nested reference fields.

### Pair-construction tests

1. target latent rows are byte-identical;
2. target diffusion-noise rows are byte-identical;
3. timesteps are identical;
4. target prompt and PM-A embeddings are identical;
5. reference A/B latents differ;
6. reference-noise rows are byte-identical;
7. target masks are identical;
8. reference masks correspond to A and B.

### Loss routing tests

1. core diffusion loss receives only `eps_match`;
2. changing `eps_swap` cannot alter `L_diff`;
3. directional loss decreases when `sim(e_swap,B)` increases;
4. directional loss increases when the output moves toward A;
5. absolute term prevents a pure “move away from A” solution;
6. ring loss is zero for identical pair outputs;
7. all counterfactual losses are zero/not emitted above the max timestep.

### Gradient tests

With a nonzero initialized test connector:

1. counterfactual directional loss reaches connector-up;
2. gradients reach connector-down and reference K/V after connector-up opens;
3. wrong-reference latent affects the gradient;
4. target/base U-Net remains frozen;
5. no branched CA parameters are installed or trained.

### Checkpoint tests

1. strict restore includes all NN5 config fields;
2. validation model receives counterfactual-independent runtime architecture;
3. step-zero connector still produces exact PhotoMaker parity;
4. scale-0 diagnostic remains exact ordinary PhotoMaker.

---

## 12. Approval protocol

### 12.1 Checkpoints

Evaluate at:

```text
2k
4k
```

Do not approve continuation merely because training loss falls.

### 12.2 Primary evaluation

Run the 96-image five-way matrix on the **same SDXL base** at:

```text
residual scale 1
```

Then optionally run scale 2 for sensitivity.

Use RealVisXL only after the same-base scale-1 gate passes.

### 12.3 Required metrics

- mean directional gain toward R2;
- bootstrap 95% interval;
- positive fraction;
- per-identity directional means;
- similarity to original A and swapped B;
- matched R1 preservation versus PM0;
- reference-image versus reference-noise effects in identity, LPIPS, and core MAE;
- face-detection rate;
- landmark displacement;
- seam/boundary proxy;
- outside-core MAE/LPIPS;
- per-site applied residual and cap fraction.

### 12.4 Continue criteria

Continue beyond 4k only if all are true at scale 1 on same SDXL:

1. mean directional gain toward R2 is positive;
2. bootstrap lower bound is above zero;
3. positive fraction is at least 60%;
4. improvement is not driven by one identity;
5. matched-reference outputs preserve A approximately as well as PM0;
6. reference-image effect is clearly larger than reference-noise effect;
7. face detection and boundary/landmark metrics remain near PM0;
8. visual changes look like identity movement, not blinking, expression,
   smoothness, age, or sharpening.

### 12.5 Early stop criteria

Stop at 2k if:

- mean direction is negative and the upper confidence bound is below zero;
- face validity drops;
- the cap saturates broadly while direction remains wrong;
- swapped rows learn generic expression/texture changes.

Stop at 4k if the confidence interval still includes zero or the positive
fraction remains below 55%.

---

## 13. NN5b fallback if NN5a fails

If NN5a provides a correct directional gradient but the image result remains
weak or noisy, the next representation change should be a clean identity-token
lane.

The existing PhotoMaker V2 encoder already contains a QFormer/Perceiver that
produces two 2048-D identity tokens from:

- InsightFace recognition embedding;
- CLIP image patch tokens.

Add `extract_id_tokens()` returning `[B, 2, 2048]` without mean reduction.

At each selected PPR site, compute a separate clean identity candidate:

```text
C_spatial = Attention(Q_target, K_spatial, V_spatial)
C_id      = Attention(Q_target, K_id_tokens, V_id_tokens)
C_null    = Attention(Q_target, K_null, V_null)
```

Fuse with a zero-init bounded connector, while preserving the same target base
and output anchor.

Do not implement this in NN5a. First determine whether the current spatial
memory can learn identity causality when the objective actually requires it.

---

## 14. Changes not recommended

- Do not continue NN4 simply because its launcher budget is 20k.
- Do not increase runtime scale, gate maximum, RMS cap, or number of attention
  sites.
- Do not re-enable branched cross-attention.
- Do not re-enable pose adaptation or CA face mixing.
- Do not return to absolute reference-owned face attention.
- Do not train a wrong-reference row against target-A core diffusion MSE.
- Do not treat a satisfied matched/null magnitude margin as identity success.
- Do not use RealVis-only validation to approve the next architecture.
- Do not combine counterfactual supervision, clean identity tokens, semantic
  parts, new sites, and larger authority in one first run.

---

## 15. Final recommendation

NN4 answered its question:

> The protected PPR operator is safe and reference-sensitive, but same-person
> reconstruction plus null/magnitude supervision does not make it
> reference-identity-causal.

Finalize NN4 as the safety baseline and archive its 2k/4k causal results.

Implement NN5a as a paired fixed-target matched/wrong-reference run with:

- the same NN4 PPR operator;
- one target per microbatch;
- exact target/noise/timestep duplication;
- exact shared reference noise;
- full target PhotoMaker identity A;
- matched diffusion reconstruction only;
- absolute and directional identity loss toward B on the wrong-reference row;
- boundary-ring preservation;
- a 4k approval gate using same-SDXL scale-1 causal evaluation.

This is the smallest experiment that directly attacks the failure shown by the
NN4 data and produces an unambiguous answer for the next architectural decision.
