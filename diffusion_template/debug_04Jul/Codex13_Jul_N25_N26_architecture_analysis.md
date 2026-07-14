# N25/N26 full-validation analysis and architecture review

Date: 14 July 2026

## Executive conclusion

N25 and N26 do not improve the effective N17 architecture.

- N25 reaches mean ID similarity `0.3420` at 10k, essentially the same as step-matched N17 (`0.3431`). Staged training changes convergence speed, not the final behavior.
- N26 reaches `0.3381` at 10k. Its reference-only cross-attention (CA) training redistributes successes across identities and prompts, but does not create a net gain or resolve the visual failures.
- Both runs preserve PhotoMaker's global scene very closely, then alter the local face region. The recurring failure is not scene generation; it is transferring reference spatial content into a target face with different pose, expression, scale, hair, or occlusion.
- N26 is **not a clean test of direct trainable face CA**. Its `ref_only` CA parameters act on the reference branch (`ref_hidden -> face_prompt`), while the target branch still receives ordinary generation-prompt CA. The target can only benefit indirectly when later self-attention copies reference features.
- The next useful work should change the face-conditioning architecture: canonical reference face tokens, target-face queries, and a zero-initialized residual adapter on the PhotoMaker target stream. Another long run of the current raw reference-grid replacement is unlikely to move beyond the N17 ceiling.

## Artifacts

- Comparison PDF: [`../full_validation_results/ba_n25_n26_14Jul/full_val_report_N25_N26_vs_key.pdf`](../full_validation_results/ba_n25_n26_14Jul/full_val_report_N25_N26_vs_key.pdf)
- Reproducible PDF config: [`../infer_tools/full_val_n25_n26_14jul_report.yaml`](../infer_tools/full_val_n25_n26_14jul_report.yaml)
- N25 metrics: [`../full_validation_results/ba_staged_legacy_N25_steps/metrics_ba_staged_legacy_N25_steps.json`](../full_validation_results/ba_staged_legacy_N25_steps/metrics_ba_staged_legacy_N25_steps.json)
- N26 metrics: [`../full_validation_results/ba_staged_caref_N26_steps/metrics_ba_staged_caref_N26_steps.json`](../full_validation_results/ba_staged_caref_N26_steps/metrics_ba_staged_caref_N26_steps.json)

The PDF has a config/metric table followed by one image page per identity. Every included result directory matched all 96 expected filenames. The columns are PhotoMaker, N17 10k, N23 10k, N24 10k, N25 1k/5k/10k, and N26 1k/5k/10k. N17 10k is used instead of N17 26k for a step-matched architecture comparison; the N17 26k mean was only `0.3482` and its visual failure mode was already stable.

Recreate it with:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda run -n photomaker python infer_tools/pdf_full_val.py \
  --config infer_tools/full_val_n25_n26_14jul_report.yaml
```

## Quantitative comparison

All runs detected a face in `96/96` images. Detection rate therefore does not distinguish face quality.

| Run | Mean ID sim | Eddie | Elon | Jennie | Jensen | Jisoo | Keanu | Lex | Marion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PhotoMaker V2 | **0.4886** | 0.2053 | **0.5462** | **0.5898** | **0.5572** | **0.5768** | 0.4997 | 0.4727 | 0.4611 |
| N17 legacy 10k | 0.3431 | 0.1325 | 0.4930 | 0.4207 | 0.4242 | 0.2532 | 0.3986 | 0.3613 | 0.2616 |
| N23 CAMIX 10k | 0.4653 | 0.1881 | 0.5179 | 0.5614 | 0.5064 | 0.4961 | **0.5025** | **0.4752** | **0.4745** |
| N24 dual gate 10k | 0.3899 | 0.1507 | 0.4850 | 0.4556 | 0.4548 | 0.3726 | 0.4357 | 0.3889 | 0.3761 |
| N25 staged legacy 1k | 0.2223 | 0.1012 | 0.3211 | 0.2399 | 0.3745 | 0.1115 | 0.2331 | 0.2129 | 0.1841 |
| N25 staged legacy 5k | 0.3278 | 0.1152 | 0.4695 | 0.4252 | 0.4461 | 0.2021 | 0.3924 | 0.3429 | 0.2292 |
| N25 staged legacy 10k | 0.3420 | **0.1396** | **0.4997** | **0.4486** | **0.4514** | 0.1920 | 0.3902 | 0.3685 | **0.2463** |
| N26 staged + ref CA 1k | 0.2961 | 0.1096 | 0.3990 | 0.4318 | 0.3645 | **0.3392** | 0.2621 | 0.2761 | 0.1865 |
| N26 staged + ref CA 5k | 0.2935 | 0.1091 | 0.4522 | 0.3129 | 0.3959 | 0.2158 | 0.3494 | 0.3425 | 0.1703 |
| N26 staged + ref CA 10k | 0.3381 | 0.1247 | 0.4868 | 0.3878 | 0.4468 | **0.2319** | **0.4210** | **0.3801** | 0.2254 |

Bold in the N25/N26 block marks the better 10k result for that identity, not statistical significance.

### Checkpoint behavior

N25 is monotonic (`0.2223 -> 0.3278 -> 0.3420`) but converges to N17. Its 10k delta from N17 10k is only `-0.0011`. This strongly rejects the hypothesis that training BA at all noise levels was the main cause of N17's visual ceiling.

N26 is non-monotonic (`0.2961 -> 0.2935 -> 0.3381`). Its unusually strong 1k result, particularly for Jisoo and Jennie, disappears at 5k and only partly recovers. The CA parameters change optimization dynamics, but do not produce a stable improvement.

At 10k, N26 versus N25 changes identity means as follows:

| Identity | N26 - N25 |
|---|---:|
| Jisoo | +0.0399 |
| Keanu | +0.0308 |
| Lex | +0.0116 |
| Jensen | -0.0046 |
| Elon | -0.0129 |
| Eddie | -0.0149 |
| Marion | -0.0209 |
| Jennie | -0.0608 |

This is redistribution rather than broad improvement. Per-image variance is larger still: N26 improves `Jumping wo_jisoo.png` by `+0.2834` and `Kickboxing_marion.png` by `+0.2340`, but degrades `Laughing w_marion.png` by `-0.4002` and `Skiing wom_jennie.png` by `-0.3179`.

### Prompt sensitivity at 10k

| Prompt group | N25 | N26 | Observation |
|---|---:|---:|---|
| Dancing men | 0.2856 | 0.3428 | N26 helps |
| Dancing women | 0.2898 | 0.3529 | N26 helps |
| Jumping women | 0.1797 | 0.3036 | N26 helps |
| Kickboxing | 0.3539 | 0.3750 | small N26 gain |
| Laughing women | 0.3471 | 0.2158 | severe N26 regression |
| Rushing women | 0.3575 | 0.2909 | N26 regression |
| Skiing women | 0.2941 | 0.1664 | severe N26 regression |

The difficult cases combine non-frontal pose, strong expression, small faces, hair movement, hands, or goggles. This pattern is more informative than a `0.0039` mean-score difference.

## Visual analysis

### Shared N25/N26 behavior

1. **Global composition remains PhotoMaker-like.** Background, body pose, clothing, and object placement are usually nearly unchanged. A low-resolution same-seed image comparison gives PhotoMaker-to-N25 correlations of `0.9667-0.9678` and PhotoMaker-to-N26 correlations of `0.9666-0.9695`. These correlations are background dominated, so they are evidence of scene preservation, not identity quality.
2. **Training changes the local face, not the scene.** N25 5k-to-10k correlation is `0.9927`. N25 10k versus N26 10k is `0.9923` (mean absolute difference `0.0130`). The architectures produce almost the same image outside a small local region.
3. **Both return to the N17 family.** N17 10k versus N25 10k correlation is `0.9882`; N17 10k versus N26 10k is `0.9881`. The staged schedule and low-LR reference CA do not change the effective face-transfer mechanism.
4. **The face often inherits reference geometry instead of target geometry.** Hair, forehead extent, frontal orientation, and occluding content from the reference crop appear in a target head whose pose and scale differ. The result can be recognizable to a face encoder while still looking pasted, displaced, elongated, or anatomically incompatible.

### Identity-specific findings

- **Jisoo:** the clearest architecture failure. N25/N26 frequently transfer hair or an occluding shape into the face box. Skiing, laughing, angry, crying, and kickboxing remain warped or partially covered. N26 helps selected jumping/dancing samples but does not generalize; its 10k Jisoo mean (`0.2319`) remains far below N24 (`0.3726`) and PhotoMaker (`0.5768`).
- **Marion:** repeated face/hair boundary failures and weak identity. N26's laughing image is especially severe (`-0.0118` ID similarity versus N25's `0.3884`). A learned reference CA that genuinely represented identity should not create this prompt-specific collapse.
- **Keanu:** N26 improves some drumming, dancing, crying, and kickboxing images, but the known pose/location mismatch persists. More training changes texture and recognition score without reliably moving the face to target-compatible geometry.
- **Elon and Jensen:** the strongest N25 cases. Their reference and target face geometry is often closer to frontal and unobstructed, so raw reference spatial features are less harmful. Their success supports the geometry-mismatch diagnosis rather than a global inability to learn identity.
- **Jennie:** N25 is stronger than N26 overall. N26's large skiing and night-ride regressions show that reference CA does not supply a robust pose-invariant identity representation.
- **Eddie:** all BA variants remain low. N25/N26 do not solve weak identity when the reference signal itself is difficult for the current transfer path.
- **Lex:** N26 gives a modest gain, but remains below N23 and PhotoMaker. Improvements are local rather than a new stable regime.

### What N23 and N24 add to the diagnosis

- **N23 CAMIX (`0.4653`)** keeps target/noise face K/V available alongside reference K/V. It avoids many destructive replacements and is the strongest BA score, but images remain very close to PhotoMaker. This shows that target geometry is essential, while fixed concatenation lets the strong PhotoMaker path dominate instead of learning a useful identity correction.
- **N24 dual attention gate (`0.3899`)** separates target and reference attentions and learns a gate. It is better than N17/N25/N26, so separation is directionally correct. However, it blends two absolute attention outputs. Averaging incompatible target- and reference-coordinate representations is still not the same as learning a target-aligned residual.
- **N17/N25 (`~0.343`)** show that pure reference face K/V replacement converges to a stable but poor attractor.
- **N26 (`0.338`)** shows that training the current reference-side CA does not remove that attractor.

## Branched-attention implementation audit

The ranking below distinguishes architectural blockers from smaller implementation defects.

### 1. Critical: raw reference spatial tokens are used as identity memory

In [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py), the active self-attention processor:

- masks the full reference feature grid (`ref_face_hidden = ref_hidden * ref_mask`, around lines 399-406);
- uses it directly as face K/V (lines 443-470);
- replaces the target face attention output inside the target bbox (line 505).

The reference and target sequences have the same tensor dimensions, but their cells do **not** represent the same anatomy. Reference pose, crop location, hair, hands, and background context survive in the UNet features. Attention has no landmarks, canonical coordinates, ROI alignment, or pose warp to establish correspondence.

`POSE_ADAPT_RATIO > 0` is even more explicit: line 414 adds reference and target tokens at the same sequence indices before attention. At N25/N26's ratio `0`, that direct index addition is absent, but the reference remains a full spatial grid with no canonical alignment.

This is the highest-priority issue and matches the visual evidence.

### 2. Critical: N26's cross-attention is not target-face CA

The active `BranchedCrossAttnProcessor` (approximately lines 615-844) does this:

| Half | Query | K/V | Output |
|---|---|---|---|
| target/noise | `noise_hidden` | generation prompt | `hidden_bg` |
| reference | `ref_hidden` | face/ID prompt | `hidden_ref` |

It then concatenates the outputs. There is no target-face query attending a face prompt, no spatial target face/background merge in CA, and the masks set on this processor are not used by `__call__`.

[`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py) lines 78-92 make N26's `ba_ca_train_mode=ref_only` train only `.attn2.processor.ref_to_*`. Therefore N26 optimizes how the **reference branch** reads the face prompt. Its influence on the generated face is indirect through subsequent self-attention transfer.

Conclusion: N26 does not show that trainable face CA is unhelpful. It shows that this CA topology and trainable-path selection do not directly solve target-face conditioning.

### 3. High: masking K/V by multiplication leaves attention sinks

Self-attention multiplies non-face reference tokens by zero, but still passes the full sequence to softmax. Zero K/V tokens are not excluded by an attention mask. They consume probability mass and make behavior depend on face area and UNet resolution.

The `id_only` face prompt has the same issue. [`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py) lines 484-498 zeros roughly 75 of 77 prompt tokens. The source comment already notes that these become attention sinks. N26 trains reference CA against this sparse full-length sequence.

The CA processor also accepts `attention_mask` but does not pass it to either scaled-dot-product attention call. A compact token sequence or a real boolean/additive mask is required; multiplying hidden states by zero is not equivalent.

### 4. High: bbox masks are hard, rectangular, and fragile across scales

- `mask_softness=0` makes `force_binary_masks=True`.
- `_prepare_mask` bilinearly resizes each mask, then thresholds at `> 0.5` (processor lines 541-569).
- A small face can lose substantial area at coarse UNet attention grids.
- A rectangle includes hair, hands, goggles, and nearby background, exactly the content seen leaking into difficult images.
- The target mask is fixed from the preliminary PhotoMaker image unless optional re-tracking is enabled. Once BA moves or deforms the face, the write region does not follow it.

These issues amplify the raw-grid problem. They are not likely to fix it alone.

### 5. High: the face output is replacement-style, not a bounded residual

The face attention output is selected inside the mask and the background output outside it (`merged = bg*(1-mask) + face*mask`). Although the transformer residual is later added, the learned branch itself predicts an absolute replacement. N24 similarly blends two absolute attention outputs.

This gives the reference path enough authority to alter geometry and boundaries. It also creates a false choice between N17-like reference dominance and N23-like PhotoMaker dominance. A zero-initialized residual correction on top of the standard PhotoMaker target stream is a better-controlled interface.

### 6. Medium: the branches are not strictly isolated

`strict_face_routing` is false in these configs. Consequently:

- background K/V use the full target sequence, including target-face tokens;
- the standard residual is added across the entire target branch;
- face information can spread outside the intended route.

Some target residual is useful for preserving pose, so strict isolation is not automatically better. The problem is that the current behavior does not match a clean face/background routing abstraction and is not instrumented. The intended contribution of each path cannot be measured reliably.

### 7. Medium: the staged optimizer still updates inactive BA parameters through AdamW decay

On text-only and PhotoMaker-only training samples, `attach_inactive_branched_params()` connects every trainable BA parameter to the graph with an exactly zero gradient. N25/N26 use `optimizer.weight_decay=1e-3`. AdamW applies decoupled weight decay when a parameter has a gradient tensor, even when that tensor is zero.

Thus the nominally inactive BA parameters are decayed on approximately 30% of batches. This does not explain the visual failure, but it means N25 is not a perfectly clean "BA trains only in its inference window" experiment. Inactive parameters should have `grad=None`, or their optimizer groups should be skipped for those batches.

### 8. Medium: training/inference conditioning still differs

Training explicitly discards `do_cfg` and never trains null/negative prompt behavior (`lora2.py` lines 368-372). Inference uses classifier-free guidance. `ba_uncond_face_fix=true` avoids one known malformed unconditioned face prompt, but does not train the branched path under conditioning dropout.

The staged schedule is also approximate: a uniformly sampled training timestep is converted to linear progress ratios (`20%` text-only, `10%` PhotoMaker, `70%` BA). This matches the count of 50 inference steps, not necessarily the scheduler's actual timestep/noise distribution.

### 9. Medium: the current ID loss is target reconstruction supervision, not reference identity supervision

[`id_loss.py`](../src/loss/id_loss.py) compares the generated crop to the ground-truth **target** image crop, using the target bbox. The target and reference should share identity in this dataset, so the loss is not invalid. However, it does not explicitly require consistency with the selected reference, and it can reward target-specific pose/expression reconstruction without teaching reference-to-target correspondence.

This explains why changing ID-loss weight has limited value while the spatial transfer path remains wrong.

### 10. Low but concrete: some wired CA controls are inert

`branched_runtime.py` assigns `equalize_face_kv`, `class_tokens_mask`, and `id_embeds` to active CA processors, but `BranchedCrossAttnProcessor.__call__` does not consume them. These settings currently provide no CA behavior. The wiring should either be implemented and tested or removed from the claimed configuration surface.

### Dimension and resolution check

For the current CosmicLarge dataset, no obvious height/width swap or reference-bbox scaling bug was found:

- dataset references are PIL images;
- training and inference letterbox them to `1024 x 1024` while preserving aspect ratio;
- reference bbox coordinates are scaled and padded consistently;
- the VAE produces a `128 x 128` latent, matching the target latent;
- masks are resized in 2-D at each attention resolution.

This tensor-level correctness does **not** imply semantic alignment: a reference-grid coordinate and target-grid coordinate still depict different anatomy.

There is one dormant dimension bug: `_encode_reference_latent()` treats a tensor reference as if `target_shape` were an image size, resizing it to the latent size before VAE encoding and then upsampling the resulting latent. CosmicLarge returns PIL references, so this did not affect N25/N26, but it will break a future dataset that supplies tensor references.

## Architecture direction

### Recommended core design: target stream plus canonical identity memory

1. **Keep standard PhotoMaker as the target backbone.** Generate the scene and target pose through the normal target stream.
2. **Canonicalize the reference face.** Detect landmarks, crop and align the face to a canonical coordinate system, and encode that ROI into a compact multi-scale token set. Do not pass the full reference image grid as face K/V.
3. **Separate identity from pose.** Use PhotoMaker/InsightFace global identity tokens for identity. Preserve pose, expression, scale, and occlusion from target queries and optional target landmarks. Reference spatial tokens should provide local appearance only after alignment.
4. **Use target-face queries.** At selected UNet layers, target face ROI queries attend compact reference identity/appearance tokens. This is a genuine target-face cross-attention path, unlike N26's reference-branch CA.
5. **Inject a bounded residual.** Predict `delta_face` and apply

   ```text
   h_out = h_photomaker + soft_face_mask * gate(layer, timestep, confidence) * delta_face
   ```

   Initialize the gate to zero or near zero. This makes the starting model exactly PhotoMaker and lets training learn only corrections that improve reference identity.
6. **Use semantic soft masks.** Face parsing or landmark-derived masks should distinguish inner face, hair, and occluders. Preserve fractional mask coverage at coarse scales instead of thresholding every resized mask.

This design directly addresses both observed extremes: N17/N25 reference dominance and N23 PhotoMaker dominance.

### Should the doubled reference UNet branch remain?

Not by default. Encoding a clean reference through the full noised UNet branch is expensive and preserves unwanted spatial scene information. First test a compact reference encoder or ROI feature pyramid that can be computed once per reference. Keep a full reference branch only if an ablation demonstrates value beyond canonical identity tokens.

If the branch is retained, it should output canonical ROI memory rather than a full-grid tensor, and reference noise should not be allowed to define target pose.

### Training objectives for the new interface

- Reference-to-generated identity similarity, using the actual selected reference embedding.
- Standard diffusion loss on the target.
- Outside-mask consistency to a frozen PhotoMaker teacher, preventing scene drift.
- Landmark/pose consistency with the target or teacher face.
- Boundary consistency around a soft face mask.
- Optional feature/perceptual loss on an aligned inner-face crop, excluding hair and occluders.
- Conditioning dropout/null-prompt training so CFG behavior is learned rather than patched only at inference.

The gate and residual should be logged per layer and timestep. A useful model should show nonzero corrections in identity-bearing mid/high-resolution layers while keeping background residual norms near zero.

## Recommended research sequence before another long run

No new run scripts are created in this report, as requested.

1. **Write correctness tests first.** Verify mask area at every UNet resolution, compact-token attention exclusion, target/ref batch ordering, checkpoint round-trip, and that intended target-face CA parameters receive nonzero gradients.
2. **Fix the existing probe semantics.** If current CA is retained temporarily, add an opt-in target-face CA mode; use target-face queries, compact ID-token K/V, spatial merge, and a real attention mask. Do not call the existing `ref_only` path target-face CA.
3. **Prototype the zero-init residual adapter.** Use canonical aligned reference ROI tokens and soft target masks. Freeze PhotoMaker and train only the adapter/gates initially.
4. **Run short architecture probes.** Validate at 1k and 3k/5k on the same 96 images. Require improvements on the known geometry failures, not just mean ID similarity.
5. **Only then run 10k+.** A long run is justified if Jisoo/Marion/Keanu hard cases improve without collapsing N23/PhotoMaker-like scene and pose quality.

### Acceptance criteria for the next architecture

- It must beat N24 (`0.3899`) overall while improving, not merely redistributing, difficult identities.
- It must visibly fix several fixed hard cases: Jisoo skiing/laughing, Marion laughing/night-ride, and Keanu rushing/kickboxing.
- It must remain close to PhotoMaker outside the face, measured with an outside-mask perceptual or pixel metric.
- Reference-token attention mass must be measured and concentrated on real compact face tokens, with no zero-token sinks.
- Target pose/landmarks must remain closer to PhotoMaker/target geometry than to the reference geometry.

## Final interpretation

N25 falsifies schedule mismatch as the primary explanation for N17. N26 does not falsify trainable cross-attention; it exposes that the implemented CA trains the wrong side for direct target-face conditioning. Across N17 and N23-N26, the consistent lesson is:

> Identity should be injected as a target-aligned, bounded correction, not as a raw reference spatial replacement and not as an untrained fixed concatenation with the PhotoMaker path.

The next gain is more likely to come from changing that interface than from another loss-weight, learning-rate, or start-step sweep.
