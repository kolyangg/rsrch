# NN4 causal-null / `up_blocks.0` code audit

**Date:** 20 July 2026  
**Repository:** `kolyangg/rsrch`  
**Branch:** `main_clean`  
**Audited head:** `1b2bc95e447f0cd8b79dd07046496e487bafa570`  
**Primary launcher:** `diffusion_template/jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh`

## Executive verdict

NN4 is **substantially implemented as the intended minimal correctness-and-causality screen**, rather than as the complete longer-term geometry-aligned architecture. The important runtime and preservation changes are wired correctly:

- one base reference-noise tensor is duplicated exactly across CFG unconditional/conditional copies;
- reference token embeddings and reference pooled text conditioning are neutralized;
- split branched cross-attention is disabled, including on the alternate-base validation model;
- spatial PPR authority is restricted to `up_blocks.0.attn1`;
- target self-attention remains the baseline and the branch is a zero-initialized, gated, RMS-bounded additive residual;
- the ordinary PhotoMaker epsilon prediction remains exact outside the feathered target core;
- the learned null memory uses the same target query and the same reference K/V projections as the matched reference;
- target-side PhotoMaker identity is attenuated on half of the physical training batch;
- low-timestep reference identity loss, strict processor restore, and invalid-sample checks remain enabled;
- pose adaptation and CA face mixing remain disabled, as intended.

There is, however, **one real objective-algebra bug that should be fixed before training**:

> The implemented matched/null margin measures `D_ref - 2·D_null`, not the intended `D_ref - D_null`.

The fix is one line and is given below. After applying it and adding the regression test, NN4 is suitable for a **2k/4k approval-stage run**. It should not be allowed to consume the complete 20k budget unless a fixed-seed R1/R2 reference-swap evaluation shows positive directional identity control.

Two additional safeguards are strongly recommended in the same patch:

1. exclude samples with an empty target core from all auxiliary-loss denominators, not merely samples with an empty reference ROI;
2. make the 4k counterfactual approval test an explicit part of the run protocol, because the normal matched-reference RealVis validation cannot establish causal reference identity control.

## Scope of this audit

The audit traced the current NN4 path through:

1. local and server launchers;
2. Hydra inheritance and resolved model controls;
3. model construction and processor installation;
4. trainable-parameter grouping;
5. target/reference latent preparation;
6. CFG reference-noise handling;
7. reference token and pooled-text conditioning;
8. packed residual self-attention;
9. learned-null candidate construction;
10. diffusion, null, separation, cap, and decoded identity losses;
11. output anchoring;
12. checkpoint manifests and strict restoration;
13. alternate-base RealVis validation;
14. existing NN4 regression tests.

This is a static, line-by-line audit of the current GitHub branch. I did not independently execute the GPU training job or the repository test suite in this environment.

## 1. Architecture actually launched

The launcher selects:

```text
one_id_ba_NN4_causal_null_up0
```

and inherits the NN2 → NN3a → NN3b chain. The effective NN4-specific controls are:

```yaml
disable_branched_ca: true
loss_kind: core_normalized

model:
  ba_site_policy: up_blocks0_attn1
  ba_cfg_reference_noise_pairing: true
  ba_reference_token_text_mode: zero
  ba_reference_pooled_text_mode: zero

  ba_null_residual_loss_weight: 0.10
  ba_match_null_margin_weight: 0.05
  ba_match_null_margin: 0.02
  ba_cap_loss_weight: 0.01
  ba_cap_loss_target: 0.12
  ba_delta_rms_cap: 0.15
```

Inherited controls include:

```yaml
train_ba_all_steps: true
train_ba_only: true
train_branched_ca_lora: false
branched_attn_weight_mode: ref_only
branched_attn_new_weight_kind: lora

model:
  ba_processor_variant: packed_residual_v1
  ba_sa_train_mode: packed_residual
  ba_train_timestep_mode: inference_ba_region
  ba_correctness_guards: true
  ba_invalid_sample_policy: skip_batch
  ba_strict_processor_restore: true
  ba_connector_input_mode: reference_minus_learned_null
  ba_null_memory_tokens: 8
  ba_pm_id_attenuation_probability: 0.50
  ba_pm_id_attenuation_scale: 0.0
  ba_gate_max: 0.50
  ba_gate_init_logit: 0.0
  ba_target_core_erode_frac: 0.10
  ba_reference_token_mode: packed_bbox_roi
  ba_reference_continuation: frozen_base
  ba_output_anchor_mode: base_outside_core
  use_id_loss: true
  id_loss_weight: 0.05
  id_loss_max_timestep: 300

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
```

The local launcher trains on SDXL base, validates on RealVisXL V4.0, uses rank-32 reference K/V LoRA, connector rank 16, batch 2, LR `5e-5`, 2k warmup, and a nominal maximum of 20k optimizer steps.

## 2. Recommendation-to-code compliance matrix

| Requested NN4 property | Current implementation | Verdict |
|---|---|---|
| Preserve ordinary target self-attention | `target_base` is computed by ordinary self-attention before adding PPR | Correct |
| Packed real reference-face tokens | reference bbox tokens are packed and padded positions receive an additive `-inf` mask | Correct |
| Target Q attends reference K/V | `target_query` retrieves packed reference K/V | Correct |
| Real learned null through same route | null tokens use the same `ref_to_k`, `ref_to_v`, K norm, target Q, and SDPA | Correct |
| Zero-init protected residual | `connector_up` starts at zero; output is additive | Correct |
| Bounded authority | per-site scalar gate plus relative RMS cap `0.15` | Correct |
| Protect PhotoMaker outside core | independent base prediction and `base + core·(branched-base)` | Correct |
| Restrict authority to `up_blocks.0` | explicit `up_blocks.0.*.attn1.processor` selection | Correct |
| Disable branched CA | original Diffusers `attn2` processors retained | Correct |
| Propagate CA disable to RealVis validation | copied to temporary alternate-base validation model and pipeline | Correct |
| Pair CFG reference noise | base noise/latents duplicated exactly and equality asserted after noising | Correct |
| Neutralize reference token text | reference `encoder_hidden_states` half is zero | Correct |
| Neutralize reference pooled text | reference `added_cond_kwargs['text_embeds']` half is zero | Correct |
| Keep target conditioning unchanged | only the reference halves are zeroed | Correct |
| Core-normalized face diffusion loss | per-sample weighted MSE divided by core area | Correct, with boundary caveat |
| Penalize null response | connector response to `C_null` is RMS-penalized | Correct as a proxy |
| Separate matched and null | intended margin exists | **Incorrect algebra; mandatory fix** |
| Penalize pre-cap excess | squared excess above `0.12` | Correct |
| Force some identity ownership into BA | target PhotoMaker ID delta removed in 50% of batch | Correct |
| Direct reference-ID supervision | low-timestep decoded FaceNet loss | Correct but not anatomy-gated |
| Strict checkpoint/topology restore | architecture manifest and exact processor-state checks | Correct |
| Counterfactual R1/R2/null approval | not part of ordinary launcher validation | **Protocol gap** |
| Semantic part/3D correspondence | not implemented | Expected scope gap; NN4 is not the full vNext architecture |
| Query/head/timestep-local gate | scalar per-site gate only | Expected scope gap |
| Dedicated stable reference encoder/cache | evolving noised U-Net reference stream retained | Expected scope gap |

## 3. Correctly implemented high-risk fixes

### 3.1 CFG reference-noise pairing

The previous inference bug was that the CFG-expanded reference batch could be assigned independently sampled noise in its unconditional and conditional halves. NN4 now:

1. determines the base output-image batch `B` from a CFG generation batch `2B`;
2. prepares/caches reference latents, masks, and noise at batch `B`;
3. constructs:

```text
reference latents = [R, R]
reference masks   = [M, M]
reference noise   = [eps, eps]
```

4. applies the scheduler with identical timesteps;
5. asserts exact equality of both the raw reference noise and the actual noised reference tensors.

The setup and cleanup paths clear both `_ref_noise` and `_ref_noise_base`, so the cache is scoped to a generation or training batch rather than leaking across samples.

**Verdict:** correct and materially better than the prior implementation.

### 3.2 Reference text isolation

NN4 zeroes the reference token sequence after any legacy face-prompt construction and independently zeroes only the reference half of SDXL pooled `text_embeds`:

```text
encoder_hidden_states = [target PhotoMaker tokens, zeros]
added text_embeds     = [target pooled text, zeros]
```

`time_ids` remain duplicated. This retains image-size/crop conditioning but removes the target prompt/identity semantics that were the primary confound.

Because split branched CA is disabled, all `attn2` sites use ordinary processors. Target rows attend the normal PhotoMaker prompt; reference rows attend the zero context. No trainable target/reference CA clone is active.

**Verdict:** correct for the intended ablation. Using a learned or empty-prompt neutral context may later be more in-distribution than literal zeros, but literal zero is a valid clean screen.

### 3.3 `up_blocks.0`-only authority

The selector matches only names beginning with:

```text
up_blocks.0.
```

and ending with:

```text
attn1.processor
```

All other self-attention processors remain ordinary. The strict installation test also verifies that no `attn2` processor is branched.

**Verdict:** correct.

### 3.4 Alternate-base validation

The RealVis validation path is not merely a pipeline label change. At each alternate-base validation, the trainer:

1. instantiates a separate model with `pretrained_model_name_or_path=RealVisXL_V4.0`;
2. copies `disable_branched_sa/ca`;
3. installs the same NN4 processor topology;
4. loads the training checkpoint's trainable processor tensors under strict architecture checks;
5. builds a pipeline bound to that RealVis validation model;
6. restores the original training model after validation.

**Verdict:** implementation is correct. Scientifically, it remains a cross-backbone transfer evaluation and should be accompanied by a same-SDXL-base validation before publication-quality conclusions.

## 4. Mandatory bug: matched/null margin uses the wrong tensor

### 4.1 Current equations

Let the bias-free connector be the linear composite:

```text
D(x) = connector_up(connector_down(x))
```

The main branch correctly computes:

```text
C_ref  = Attention(Q_target, K_ref,  V_ref)
C_null = Attention(Q_target, K_null, V_null)
raw_delta = D(C_ref - C_null)
```

Because `D` is linear and bias-free:

```text
raw_delta = D(C_ref) - D(C_null)
```

The code then separately computes:

```text
null_raw_delta = D(C_null)
```

but the margin distance currently uses:

```text
raw_delta - null_raw_delta
```

which equals:

```text
D(C_ref) - 2·D(C_null)
```

This is not the matched/null response difference. It gives the learned null candidate twice the intended coefficient and can train the null memory and connector in the wrong direction.

### 4.2 Mandatory production diff

```diff
diff --git a/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py b/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
@@
-            _, _, matched_null_distance, _ = self._masked_rms_cap(
-                raw_delta - null_raw_delta,
-                base=target_base,
-                mask=target_core,
-                max_ratio=self.delta_rms_cap,
-            )
+            # raw_delta is already D(C_ref - C_null). Since both connector
+            # projections are bias-free linear maps, pre_ratio is exactly the
+            # normalized magnitude of D(C_ref) - D(C_null). Subtracting
+            # null_raw_delta again would measure D(C_ref) - 2*D(C_null).
+            matched_null_distance = pre_ratio
```

This is both the mathematically correct quantity and avoids a redundant RMS calculation.

### 4.3 Mandatory regression test

Add this test beside the existing learned-null auxiliary-loss test:

```diff
diff --git a/diffusion_template/tests/test_packed_residual_attn_processor.py b/diffusion_template/tests/test_packed_residual_attn_processor.py
@@
 class PackedResidualProcessorTests(unittest.TestCase):
+    def test_match_null_margin_uses_main_difference_pre_ratio(self) -> None:
+        torch.manual_seed(13)
+        side = 8
+        attn = _attention()
+        processor = PackedResidualBranchedAttnProcessor(
+            16,
+            ref_kv_rank=4,
+            connector_rank=4,
+            connector_input_mode="reference_minus_learned_null",
+            collect_aux_losses=True,
+            match_null_margin=0.02,
+        )
+        processor.init_from_attention(attn)
+        mask, core = _masks(2, side)
+        processor.set_masks(mask, mask, core)
+
+        # The first cap call is for raw_delta=D(C_ref-C_null); the second is
+        # for D(C_null). The margin must use the first call's pre_ratio and
+        # must not invoke a third cap on raw_delta-D(C_null).
+        ratios = iter((0.01, 0.50))
+
+        def fake_cap(delta, *, base, mask, max_ratio):
+            del base, mask, max_ratio
+            ratio = delta.new_full((delta.shape[0],), next(ratios))
+            return delta, torch.ones_like(ratio), ratio, ratio
+
+        processor._masked_rms_cap = fake_cap
+        processor(attn, torch.randn(4, side * side, 16))
+
+        self.assertAlmostEqual(
+            float(processor.last_aux_losses["match_null_margin"]),
+            (0.02 - 0.01) ** 2,
+            places=7,
+        )
+
     def test_learned_null_uses_same_kv_route_and_receives_gradients(self) -> None:
```

The current implementation would attempt a third `next(ratios)` call and fail; the corrected implementation uses the first pre-ratio and passes.

## 5. Strongly recommended safeguard: auxiliary losses must require target-core support

The current auxiliary denominator uses only:

```text
sample_has_roi
```

which means “the reference bbox contributed at least one token.” It does not also require a nonempty target core. In an edge case where the target bbox survives strict validation but the eroded/cosine-feathered core becomes empty at a particular resolution:

- the applied PPR delta is zero;
- cap/null ratios are zero;
- the positive separation margin is nevertheless counted for that sample.

That sample can therefore train the margin despite having no target region where the residual could be applied.

Apply this straightforward diff:

```diff
diff --git a/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py b/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py
@@
         self.last_aux_losses = {}
         if self.collect_aux_losses:
-            valid_rows = sample_has_roi.float()
+            core_has_support = target_core.float().sum(dim=(1, 2)) > 0
+            valid_rows = (sample_has_roi & core_has_support).float()
             valid_count = valid_rows.sum().clamp_min(1.0)
```

The dataset filters make this edge case unlikely, but this change makes the objective internally correct and costs nothing.

For maximum strictness, also reject an empty core in `CoreNormalizedDiffusionLoss` rather than silently dropping it from a mixed batch:

```diff
diff --git a/diffusion_template/src/loss/diffusion_loss.py b/diffusion_template/src/loss/diffusion_loss.py
@@
         denominator = mask.flatten(1).sum(dim=1)
         valid = denominator > 0
-        if not bool(valid.any()):
-            return {
-                "loss": F.mse_loss(
-                    model_pred.float(),
-                    target.float(),
-                    reduction="mean",
-                )
-            }
-        per_sample = numerator[valid] / denominator[valid].clamp_min(1e-6)
+        if not bool(valid.all()):
+            bad = (~valid).nonzero(as_tuple=False).flatten().tolist()
+            raise ValueError(
+                f"CoreNormalizedDiffusionLoss received empty cores at rows {bad}"
+            )
+        per_sample = numerator / denominator.clamp_min(1e-6)
         return {"loss": per_sample.mean()}
```

If stopping the whole run on a rare empty core is undesirable, move this check into sample preparation and raise `InvalidBranchedSampleError` so the existing DDP-safe `skip_batch` path handles it.

## 6. What NN4 does and does not prove

### 6.1 Correct minimal screen

After the margin fix, NN4 is a sound test of this narrower hypothesis:

> Can a conservative, protected, `up_blocks.0` target-Q/reference-KV residual learn useful identity direction when CFG noise, target prompt leakage, target-base connector shortcuts, and late high-resolution authority are reduced?

That is a meaningful next experiment.

### 6.2 Not a full matched-versus-null training pair

NN4's null supervision is candidate-level inside one attention processor. It does **not** perform a second full U-Net forward in which the same target/noise/prompt is paired with a separately encoded null reference. Consequently:

- there is no trajectory-level `epsilon_null`;
- no full-output loss explicitly requires BA to vanish under a null reference;
- the null memory can be pushed into the connector's null space rather than learning a semantically rich “generic no-person” baseline;
- the separation margin only requires a nonzero reference-derived residual, not an identity-correct residual.

This is acceptable for the stated low-memory approval screen, but the run/documentation should call it **candidate-level learned-null regularization**, not a complete paired null-reference objective.

### 6.3 Not yet geometry aligned

NN4 still uses:

- an evolving same-timestep noised reference U-Net stream;
- a rectangular packed bbox ROI;
- no landmark, 3DMM, semantic-part, visibility, or occlusion correspondence;
- one scalar gate per attention site;
- the full BA-active denoising interval rather than a learned or explicit middle-window gate.

Restricting PPR to a coarse up block and bounding it should reduce artifacts, but it does not solve reference pose/expression entanglement by construction. If NN4 becomes reference-sensitive but still changes expression rather than identity, the next required step is semantic/3D-aligned part memory or a dedicated identity-feature bank—not additional residual scale.

## 7. Core-normalized loss: correct but double-feathered at the boundary

The current criterion computes:

```text
L = sum(core * MSE(output, target)) / sum(core)
```

The final output anchor already computes:

```text
output = base + core * (branched - base)
```

Therefore the gradient from the criterion to the pre-anchor branched prediction is proportional to approximately `core²` near the boundary. In the center, where `core=1`, nothing changes; in the feather ring, supervision is weaker than a single-feather interpretation might suggest.

This is not necessarily wrong. It is a defensible seam-protection choice. It should simply be understood as **double boundary attenuation**, not merely area normalization.

Do not change it before the first approval run unless the intent was explicitly to apply only one feather. If one-feather behavior is desired, use a hard support mask in the criterion while retaining the soft output anchor:

```python
support = (mask > 0).to(mask.dtype)
numerator = (per_pixel * support).flatten(1).sum(dim=1)
denominator = support.flatten(1).sum(dim=1)
```

## 8. Evaluation is the main remaining protocol gap

The ordinary 96-image RealVis validation uses matched same-identity references. It can report:

- original identity similarity;
- text similarity;
- face validity;
- visual quality.

It cannot determine whether the spatial branch causally follows the supplied reference identity, because target PhotoMaker and the spatial reference normally describe the same person.

The decisive NN4 gate remains the fixed-target five-way matrix already supported by the repository:

```text
PM0
R1N1
R2N1
R1N2
R2N2
```

where target latent, target prompt, target PhotoMaker identity, scheduler, masks, and batch size remain fixed.

### Recommended approval protocol

Run only 4k by default:

```bash
RUN_FOREGROUND=1 NUM_EPOCHS=2 \
  bash jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh
```

At 2k and 4k, run the reference/noise diagnostic on the saved checkpoint:

```bash
RUN_FOREGROUND=1 \
  bash jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh \
  validation_only=true \
  saved_checkpoint=/absolute/path/to/checkpoint-epoch2.pth \
  ppr_reference_noise_test=true \
  ppr_reference_noise_output_dir=ppr_NN4_4k_reference_vs_noise \
  ppr_reference_noise_overwrite=true \
  ppr_scale_sweep=false \
  ppr_diagnostic_matrix=false
```

The exact checkpoint epoch should match the repository's save cadence.

### Continue criteria

Resume toward 20k only if the 2k/4k panel shows all of the following:

1. replacing R1 with R2 produces positive mean directional identity gain toward R2;
2. the majority of samples move toward R2, not merely away from the original identity;
3. the R1/R2 effect is materially larger than the N1/N2 reference-noise effect in identity space;
4. body, hands, clothing, pose, occluders, and background remain stable;
5. face detection and landmark displacement remain acceptable;
6. `ba_aux/null_residual` remains small;
7. the matched/null margin does not collapse to zero;
8. the cap fraction is not saturating across most sites;
9. stronger residual scale is not required merely to reveal generic expression changes.

### Stop criteria

Stop at 2k/4k if:

- R2 changes expression/texture but not identity direction;
- original identity degrades without movement toward R2;
- matched and wrong reference effects are comparable to reference-noise effects;
- the cap or gate becomes the main limiter before identity direction appears;
- face geometry or occlusion artifacts return.

### Optional launcher safety diff

To make the approval stage the default rather than relying on manual stopping:

```diff
diff --git a/diffusion_template/jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh b/diffusion_template/jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh
@@
-export NUM_EPOCHS="${NUM_EPOCHS:-10}"
+# Default to the 4k approval stage. Explicitly pass NUM_EPOCHS=10 only after
+# the fixed-target R1/R2 directional-identity check passes.
+export NUM_EPOCHS="${NUM_EPOCHS:-2}"
```

The maximum 20k recipe remains available with `NUM_EPOCHS=10`.

## 9. Same-base validation should accompany RealVis

NN4 trains its branch on SDXL base and evaluates the main fixed panel on RealVisXL V4.0. The current code correctly transfers the processor state to the alternate base, but this answers a cross-backbone question.

For clean attribution, run a companion same-SDXL validation at each approval checkpoint. Otherwise a failure may come from either:

- the learned architecture/objective; or
- transfer of SDXL-trained connector/LoRA deltas into RealVis attention features.

Treat:

```text
SDXL -> SDXL
```

as the primary learning diagnosis and:

```text
SDXL -> RealVisXL
```

as a separate transfer/generalization test.

## 10. Identity-loss caveat

The inherited decoded FaceNet loss is correctly wired and restricted to low timesteps, but it still lacks explicit face-validity, landmark, visibility, or occlusion gating. Previous experiments showed that identity metrics can improve while anatomy becomes smoother or invalid.

For NN4's first approval run, keep its weight at `0.05` and do not raise it together with gate/cap changes. If identity loss begins to dominate, add one of:

- face detector confidence gating;
- landmark-validity gating;
- segmentation/occlusion preservation;
- a second identity encoder for agreement;
- a pose/expression invariance objective over same-identity references.

This is not a launch-blocking code defect, but it remains a scientific risk.

## 11. Low-priority cleanup

These items do not need to block the run:

1. `ba_reference_ca_preserve_full_pm=true` is inherited to permit target PM-ID attenuation, but NN4 subsequently zeros reference token text and disables branched CA. It is operationally harmless but semantically misleading.
2. Face-prompt embeddings are constructed and normalized before being zeroed. This is wasted compute, not incorrect behavior.
3. Auxiliary losses remain enabled on validation processor instances and are computed under `no_grad`; this adds small unnecessary work.
4. `time_ids` are retained on the reference half. This is reasonable because they carry size/crop geometry rather than target identity text, but it should be documented as intentional.
5. `up_blocks.0` still includes every self-attention site within that block. If one site dominates or artifacts return, select a smaller explicit site list rather than raising the cap.

## 12. Final patch set

### Required before launch

1. Fix the matched/null margin to use `pre_ratio` / `raw_delta`, not `raw_delta - null_raw_delta`.
2. Add the regression test that permits exactly two `_masked_rms_cap` calls and verifies the margin uses the main-difference ratio.

### Strongly recommended in the same commit

3. Define auxiliary valid rows as `reference ROI present AND target core present`.
4. Reject or DDP-safely skip empty target cores rather than silently excluding them from a mixed batch.
5. Make 4k the launcher default or clearly enforce a 2k/4k approval gate.
6. Run the existing R1/R2/reference-noise matrix at 2k and 4k.
7. Add a same-SDXL validation alongside RealVis.

### Do not change yet

- do not re-enable pose adaptation;
- do not re-enable CA face mixing;
- do not raise gate, cap, or runtime scale;
- do not add late high-resolution `up_blocks.1` authority;
- do not increase identity-loss weight;
- do not continue to 20k without positive directional R2 control.

## 13. Final assessment

**After the one-line margin correction, NN4 is coherent and faithful to the minimal architecture recommended for the next experiment.** It fixes the two main inference confounds, protects the base model, limits spatial authority, and introduces a real learned-null candidate through the correct projection route.

It is not yet the complete geometry-aligned causal identity architecture. Its remaining scientific question is whether the combination of coarse spatial evidence, PM-ID attenuation, low-timestep identity loss, and candidate-level null regularization is enough to make the residual identity-directed rather than expression/texture-directed.

The run should therefore be finalized as an **approval-stage causal screen**, not as a 20k production run. The decisive success criterion is positive matched-versus-wrong reference identity direction under fixed target conditions.

## Source map

All links below are pinned to the audited commit.

- [NN4 local launcher](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/jul_serv_runs/start_ba_NN4_causal_null_up0_realvis_1gpu.sh)
- [NN4 Hydra config](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/configs/one_id_ba_NN4_causal_null_up0.yaml)
- [Branched runtime](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/model/photomaker_branched/branched_runtime.py)
- [Packed residual processor](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/model/photomaker_branched/packed_residual_attn_processor.py)
- [Training model and objectives](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/model/photomaker_branched/lora2.py)
- [Training input/restore helpers](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/model/photomaker_branched/lora2_helpers.py)
- [Pipeline runtime propagation](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/pipelines/br_pipeline_helpers.py)
- [Diffusion losses](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/loss/diffusion_loss.py)
- [Trainer loss integration](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/trainer/sdxl_trainers.py)
- [Alternate-base validation](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/src/trainer/base_trainer.py)
- [NN4/PPR regression tests](https://github.com/kolyangg/rsrch/blob/1b2bc95e447f0cd8b79dd07046496e487bafa570/diffusion_template/tests/test_packed_residual_attn_processor.py)
