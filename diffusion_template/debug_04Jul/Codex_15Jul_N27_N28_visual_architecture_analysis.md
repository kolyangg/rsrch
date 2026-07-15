# N27/N28 full-validation visual and architecture analysis

Date: 15 July 2026

## Scope and artifacts

This analysis covers all 96 same-seed validation images at steps 1k, 5k, and 10k for:

- N27: `ba_spatial_roi_residual_N27`
- N28: `ba_idtoken_ca_residual_N28`

The primary evidence is visual: face-to-body alignment, head geometry, local artifacts, whether the generated face differs meaningfully from PhotoMaker, and whether any difference appears identity-specific. ID similarity is used only as a secondary warning signal because the differences are small, the validation set has only 96 images, and 10k total steps expose the BA path on only part of the staged schedule.

Artifacts:

- [Full 10-column comparison PDF](../full_validation_results/ba_n27_n28_15Jul/full_val_report_N27_N28_vs_key.pdf)
- [Selected full-image comparisons](../full_validation_results/ba_n27_n28_15Jul/N27_N28_focus_full.png)
- [Selected enlarged face comparisons](../full_validation_results/ba_n27_n28_15Jul/N27_N28_focus_faces.png)
- [Report-generation config](../infer_tools/full_val_n27_n28_15jul_report.yaml)

The PDF contains PhotoMaker, N17 26k, N23 10k, N24 10k, and all three N27/N28 checkpoints. Every column has all 96 expected images.

![Selected face comparisons](../full_validation_results/ba_n27_n28_15Jul/N27_N28_focus_faces.png)

## Bottom line

1. **N27 and N28 solve the worst geometry failure class.** Faces generally stay on the PhotoMaker head, align with the body, retain the intended target pose, and no longer reproduce the N17/N24 hand, hair, goggles, or reference-layout corruption.
2. **PhotoMaker is still the dominant face prior.** N27/N28 are not exact PhotoMaker copies, but most outputs are PhotoMaker faces with small local corrections rather than visibly stronger reference identity.
3. **The safe residual architecture is the right base, but the identity interface is not yet strong enough.** The result is much safer than absolute face-attention replacement. It has also made bypassing the new branch an easy solution.
4. **N28 has a specific architectural weakness.** Its four tokens are formed by adding the same identity vector to four learned global basis vectors. The common identity component cancels from relative key logits, so this is not a genuinely identity-specific four-token memory.
5. **A blind 100k continuation is not the next best experiment.** A larger dataset may require far more than 10k steps, but the current 1k/5k/10k sequence does not show a reliable trend toward stronger identity. First make branch use identifiable and measurable; then long training is justified.
6. **Keep N24's idea of learned arbitration, not its absolute-output/full-grid implementation.** Gating should scale a bounded identity residual over PhotoMaker, not average two incompatible absolute attention outputs.

## Visual comparison

### High-level run comparison

| Run | Face/body alignment | Difference from PhotoMaker | Identity-specific gain | Main failure mode |
|---|---|---|---|---|
| PhotoMaker | Strong baseline | n/a | Baseline | Identity is sometimes generic or weak |
| N17 26k | Frequently wrong in difficult prompts | Very large | Unreliable | Reference layout, occluders, and face position overwrite target geometry |
| N23 10k | Usually better than N17 | Moderate | Inconsistent | Fixed PM/BA mixing suppresses BA or makes an averaged face |
| N24 10k | Better than N17, still visibly fragile | Large | Sometimes strong, often destructive | Learned gate blends two absolute full-grid attention outputs |
| N27 1k-10k | Consistently good | Small, slowly increasing | A few useful cases, no broad trend | BA remains a weak local perturbation; ROI memory still contains reference pose/context |
| N28 1k-10k | Consistently good | Small and tightly face-local | Uneven by identity and prompt | Redundant/low-rank ID memory; some local drift without consistent identity gain |

### Face alignment and image integrity

N27/N28 are clearly better than the legacy spatial runs on the most diagnostic cases:

- **Keanu rushing on the subway:** N17 places a narrow, displaced face above a long neck. N27 and N28 retain PhotoMaker's correct head position, neck length, suit alignment, and three-quarter orientation at every checkpoint.
- **Jisoo skiing:** N17 inserts reference-like content over the goggles/face and N23/N24 still show hand-like contamination. N27/N28 preserve the goggles, face opening, coat collar, and centered head.
- **Jisoo laughing and kickboxing:** N17/N24 can transfer hair or hand structures across the face. N27/N28 preserve the target expression and glove/face boundary.
- **Marion crying:** N17 transfers hair over the face and changes the hand/face interaction. N27/N28 keep the PhotoMaker hand placement and face orientation.
- **Lex dancing:** all N27/N28 faces remain attached to the small, moving body. Older spatial BA produces stronger but much less controlled head changes.

This is not merely a better score distribution. The new preservation contract visibly works: target composition, body, clothing, pose, and face placement are stable.

### Do N27/N28 mostly keep PhotoMaker IDs?

**Yes, mostly.** The precise interpretation is:

- They keep PhotoMaker's head geometry, expression, pose, hair boundary, and much of its apparent identity.
- They do make real face-local changes. Enlarged crops show changes to eye shape, cheek width, mouth, beard texture, and skin detail, and 1k and 10k are not identical.
- Those changes are usually too small or too inconsistent to claim that BA has replaced PhotoMaker's identity with a stronger reference-conditioned identity.

The same-seed pixel diagnostic supports this visual reading. It is not an identity-quality metric; it only measures how far the generated image moved from PhotoMaker.

| Run | Full-image normalized MAE vs PM | Expanded face-crop MAE vs PM | Outside-face MAE vs PM |
|---|---:|---:|---:|
| N23 10k | 0.0240 | 0.1017 | 0.0285 |
| N24 10k | 0.0297 | 0.1392 | 0.0328 |
| N27 1k | 0.0150 | 0.0423 | 0.0196 |
| N27 5k | 0.0152 | 0.0448 | 0.0197 |
| N27 10k | 0.0153 | 0.0465 | 0.0197 |
| N28 1k | 0.0120 | 0.0482 | 0.0148 |
| N28 5k | 0.0124 | 0.0569 | 0.0149 |
| N28 10k | 0.0123 | 0.0560 | 0.0149 |

N27/N28 are much closer to PhotoMaker than N23/N24. N28 is the most globally PM-like, while its face crop changes slightly more than N27's. This is consistent with a tightly localized CA residual rather than no branch activity.

### N27 observations

N27's compact 4x4 reference ROI is substantially safer than the old masked full grid.

What works:

- Correct face placement is maintained across all inspected prompts and identities.
- Background, body, clothes, pose, and occluders remain very close to PhotoMaker.
- The branch can change the face without moving the head. Marion laughing and some Lex/Keanu cases show visible cheek, eye, and mouth changes while retaining target geometry.
- The 1k-to-10k face-crop difference grows slightly, so the residual is not completely dead.

What does not yet work:

- Most 10k outputs still read first as the PhotoMaker face.
- The small changes are not consistently closer to the reference. Some are merely sharper, older, narrower, or more strained.
- Progress from 5k to 10k is weak. Full-image similarity is effectively flat, and face changes increase only slightly.
- The reference ROI is compact but not canonical. It still contains expression, pose, hair, hands, goggles, and UNet context inside the hard rectangle. More force can therefore revive the same semantic-transfer problem in a less extreme form.

### N28 observations

N28 is the cleanest identity-only topology tested so far because it has no spatial reference UNet branch.

What works:

- Face alignment and target pose are strong in all difficult examples.
- Changes stay tightly localized to the face; this is the closest run to the stated goal that everything else remain PhotoMaker.
- Some prompts do show visible face evolution, especially Marion laughing and several Lex/Keanu prompts.

What does not yet work:

- Jennie and Jisoo frequently remain almost indistinguishable from PhotoMaker even at 10k.
- Lex changes more strongly than several other identities, but the changes are prompt-dependent and are not consistently better identity matches.
- Keanu changes in drumming, kickboxing, and some close portraits, but the later face can be less natural or less reference-specific.
- N28 changes more between 1k and 10k than N27 in face crops, yet the change does not form a monotonic identity improvement. This argues against simply increasing branch speed.

### Checkpoint trajectory

| Comparison | Full-image MAE | Face-crop MAE | Interpretation |
|---|---:|---:|---|
| N27 1k vs 10k | 0.0040 | 0.0350 | Some face learning, very little scene movement |
| N28 1k vs 10k | 0.0049 | 0.0451 | More local face evolution than N27, still small globally |

The checkpoints prove that both branches are active. They do not prove that the branch is learning a broadly better identity correction. N28 changes more but becomes uneven; N27 changes less and appears to plateau.

## Metrics, kept secondary

All runs detected a face in 96/96 images. Mean ID similarity is:

| Run | Mean ID similarity |
|---|---:|
| PhotoMaker | 0.4886 |
| N23 10k | 0.4653 |
| N24 10k | 0.3899 |
| N27 1k / 5k / 10k | 0.4702 / 0.4661 / 0.4669 |
| N28 1k / 5k / 10k | 0.4713 / 0.4582 / 0.4580 |
| N17 26k | 0.3482 |

These values support the large visual conclusion, not fine ranking: N27/N28 avoid N17/N24's major degradation but do not clearly beat PhotoMaker. Differences of a few hundredths should not choose the next architecture. The more important observation is that later checkpoints do not show a broad, monotonic visual identity gain.

## Implementation review after seeing N27/N28

### 1. Critical N28 limitation: the token construction is not a rich identity memory

N28 constructs tokens as:

```python
id_tokens = id_embeds.unsqueeze(1) + id_token_basis.unsqueeze(0)
```

See [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L934-L960).

For token `i`, a linear key projection gives:

```text
k_i = Wk(id) + Wk(basis_i)
```

For one query, `q dot Wk(id)` is identical for every token and cancels under softmax. Therefore identity does not determine relative selection among the four keys. It enters mainly through the common value term `Wv(id)`, while query-dependent token selection acts on identity-independent learned basis vectors.

Consequences:

- Four tokens do not represent four identity-specific facial parts.
- The memory is close to one global PhotoMaker identity vector plus generic learned offsets.
- N28 is conditioned by the same PhotoMaker/InsightFace family already used by the base model, so PhotoMaker dominance is expected.

This is an architectural expressivity issue, not a tensor-shape error. The replacement should generate distinct tokens from identity, for example `reshape(MLP(id), [T,D])`, identity-modulated learned queries, or projected intermediate patch/part features from an aligned face encoder.

### 2. Critical optimization shortcut: the residual may be ignored

Both new modes start exactly at PhotoMaker because `face_delta_out.up` is zero-initialized and the scalar gate starts active; see [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L17-L29), [N27 residual](../src/model/photomaker_branched/attn_processor_cleanest.py#L383-L401), and [N28 residual](../src/model/photomaker_branched/attn_processor_cleanest.py#L951-L968).

That is the correct stability initialization. However, the standard diffusion loss and reference ID loss can already be reasonably low through PhotoMaker's existing identity conditioning. Nothing explicitly requires the generated face to depend on the new memory. A near-zero residual is therefore a valid shortcut.

The next training objective must measure and reward **reference dependence**, not only identity on an ordinary correctly paired batch. A correct-reference, wrong-reference, and null-reference counterfactual on the same prompt/latent can force the branch to carry identity information while preserving everything else.

### 3. High N27 limitation: compact is not canonical

N27 crops each reference UNet feature rectangle and adaptive-average-pools it to 4x4 tokens; see [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L326-L342).

This fixes absolute grid location and zero-token attention sinks, but it does not align eyes, nose, mouth, pose, or expression. It also uses a noised full reference UNet path. Thus the memory remains appearance plus reference geometry/context, not pure identity.

The correct next step is a landmark-aligned reference crop or identity-part encoder before tokenization. Keep the hard target bbox, but canonicalize what is stored in reference memory.

### 4. High preservation limitation: hard epsilon merge is not a strict PM background trajectory

The code computes PM and BA epsilon on the current latent, then uses PM epsilon outside the bbox; see [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L321-L388) and [`br_pipeline_helpers.py`](../src/pipelines/br_pipeline_helpers.py#L861-L915).

This is much stronger than masking an intermediate attention output, and the visual results show it works well. It is not mathematically identical to an independent PhotoMaker generation outside the face:

- BA changes the latent inside the bbox.
- On the next step, convolutions and attention make PM's outside prediction depend on that already-modified latent.
- VAE decoding also has a receptive field across the boundary.

This explains the small but nonzero outside-face difference. If strict PM preservation is the requirement, maintain a parallel PM latent trajectory and hard-copy its outside-bbox latent after every scheduler update. A small fixed context band may still be needed to avoid a visible hard boundary, but it should not be a learned whole-image blend.

### 5. Medium limitation: one scalar gate cannot arbitrate difficult cases

N27 and N28 each use one trainable scalar `face_residual_gate` per processor. It cannot respond to timestep, layer role, face size, pose, occlusion, prompt, or reference confidence.

This is where N24's learned arbitration idea remains useful. The improved version should be a bounded per-head or low-dimensional gate conditioned on target queries, timestep, and reference confidence. It must scale a residual over PM, not blend two absolute full-grid outputs.

### 6. Hard mask and dimension checks

No new target/reference resolution mismatch was found in the N27/N28 path:

- N27 reference images are encoded at image resolution and converted to the target latent shape.
- N28 disables the spatial reference branch, so differing source/target image resolutions do not enter its attention memory.
- `area_preserving` mask resize uses adaptive max-pooling when reducing resolution and nearest interpolation when increasing it; see [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L680-L715).
- The residual is hard-masked at each attention resolution and the final epsilon is hard-merged.

The remaining mask issue is semantic rather than dimensional: a hard rectangle necessarily includes hair, hands, goggles, or background. The user requirement is to keep hard bboxes, so the architecture should learn identity from canonical inner-face memory while using the rectangle only as the allowed write region.

## Reassessment of N24

N24's core idea was reasonable:

- preserve a target/PhotoMaker face source;
- keep a separate reference source;
- learn how much each attention head should use.

The implementation mixed **absolute outputs**:

```python
hidden_face = hidden_face_ref * (1 - gate) + hidden_face_noise * gate
```

See [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L577-L596) and [N24 config](../src/configs/one_id_ba_dualgate_train_N24.yaml#L15-L30).

Both attentions still operate on full, zero-masked spatial grids that are not anatomically aligned. The gate can choose how much destructive reference geometry to accept, but cannot turn it into a target-aligned identity correction. This explains why N24 changes identity more than N27/N28 while producing many more malformed or averaged faces.

**Conclusion:** reuse source separation and learned gating, but at the residual level:

```text
h = h_PM + hard_bbox * gate(q_target, timestep, confidence) * delta_identity
```

Do not return to N24's absolute face-output interpolation.

## Will 100k+ steps or faster BA adaptation solve PM dominance?

The large dataset argument is valid. Ten thousand total steps are not necessarily enough, particularly because staged training activates BA on only about 70% of sampled batches. A final architecture may indeed require 50k-100k+ steps.

Current evidence does not justify continuing the unchanged N27/N28 recipe to 100k yet:

- N27 changes only modestly from 1k to 10k and is close to flat after 5k.
- N28 changes more, but its later changes are identity- and prompt-dependent rather than consistently better.
- N28's token construction limits what more optimization can learn.
- Both objectives permit the PM-like shortcut.

Faster adaptation through a larger learning rate, larger initial gate, or removing the residual bound is more likely to recover N17/N24-like geometry corruption than to create missing identity correspondence.

Before a long continuation, run three same-seed interventions on several checkpoints:

1. correct reference;
2. deliberately shuffled identity reference;
3. residual disabled, which must reproduce PhotoMaker.

Log face residual norms and gate values by layer/timestep. If correct versus shuffled reference separation grows steadily from 1k to 10k while geometry remains stable, then a 100k continuation is justified. If it is flat, architecture/objective changes are required first.

## Recommended architecture direction

### Experiment A: genuine identity-token target-face residual

This is the clean successor to N28.

1. Keep standard PhotoMaker SA/CA and hard target bbox writes.
2. Replace `id + global_basis_i` with identity-specific tokens:

   ```text
   tokens = reshape(MLP(normalized_id_embedding), T, D)
   ```

   Better still, use a small resampler over aligned face-encoder patch/part features so eyes, nose, mouth, and face shape have distinct identity-conditioned tokens.
3. Use target face queries attending those tokens.
4. Add a bounded per-head/per-layer/timestep residual gate initialized near zero.
5. Add correct/wrong/null reference counterfactual training so the adapter cannot be ignored.
6. Preserve target landmarks/pose against a frozen PhotoMaker teacher while optimizing reference identity.
7. Use a parallel PM latent trajectory if strict non-face equivalence is required.

This architecture contains no raw reference spatial grid and is the lower-risk primary direction.

### Experiment B: canonical ROI memory plus N24-style residual arbitration

This is the higher-capacity successor to N27/N24.

1. Landmark-align the hard reference bbox to a canonical face crop before encoding.
2. Build compact multi-scale identity/appearance tokens from the aligned crop, not from a noised full-image UNet grid.
3. Keep PM as the absolute target stream.
4. Compute a reference-conditioned delta and use a learned bounded gate to arbitrate between zero correction and that delta.
5. Condition the gate on target face queries, timestep, and reference quality/pose mismatch.
6. Train with the same shuffled-reference dependence objective and PM pose/background teacher constraints.

This retains N27's ability to transfer details that a single global ID vector misses, while removing reference crop location and reducing pose transfer.

Do not combine Experiments A and B initially. Their first comparison should answer whether genuinely identity-specific global/part tokens are sufficient, or whether canonical local appearance memory adds value.

## Required diagnostics and acceptance criteria

Before spending 100k steps:

1. Zero residual must reproduce PhotoMaker numerically.
2. A shuffled reference must visibly change only the face; if it barely changes the face, the branch is bypassed.
3. Correct reference must beat shuffled reference under the recognizer and visual inspection.
4. Generated landmarks/head pose must remain close to the PhotoMaker target, not the reference pose.
5. Outside-bbox latent/output differences must be logged separately.
6. Gate and residual norms must be nonzero, stable, and concentrated in identity-bearing layers/timesteps.
7. Validate 1k, 5k, and 10k first. Continue to 30k/100k only if reference dependence is still improving without head-placement regressions.
8. Keep the same hard cases as visual canaries: Keanu rushing, Jisoo skiing/laughing/kickboxing, Marion crying/laughing/night ride, and Lex dancing/night ride.

## Final recommendation

N27/N28 establish the correct safety principle: **PhotoMaker should remain the generator, and BA should be a hard-bbox residual.** They do not yet establish a strong identity-conditioning mechanism.

The next work should fix N28's identity-token construction and add explicit branch-use/counterfactual supervision. In parallel, a canonical aligned-ROI residual can test whether local appearance tokens add identity details that global ID features miss. Reuse N24's learned arbitration only as a bounded residual gate. Once reference dependence is measurable and improves through 10k without geometry damage, a 50k-100k+ run becomes technically justified.
