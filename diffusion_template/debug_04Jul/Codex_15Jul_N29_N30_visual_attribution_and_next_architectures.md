# N29/N30 visual analysis, BA attribution, and next architectures

Date: 15 July 2026

## Scope and artifacts

This analysis covers the same-seed 96-image validation sets at steps 2k, 6k, and 10k for:

- N29: `ba_qformer_idtokens_N29`
- N30: `ba_bboxnorm_idtokens_N30`

The main criteria are visual identity, face/body alignment, target pose, local artifacts, and
whether the face changes meaningfully from PhotoMaker. Metrics are secondary unless the movement
is large or supports a clear visual trend.

Artifacts:

- [Full 10-column, 96-image comparison PDF](../full_validation_results/ba_n29_n30_15Jul/full_val_report_N29_N30_vs_key.pdf)
- [Enlarged faces versus key runs](../full_validation_results/ba_n29_n30_15Jul/N29_N30_closeup_faces_vs_key.png)
- [N29/N30 checkpoint and architecture differences](../full_validation_results/ba_n29_n30_15Jul/N29_N30_checkpoint_architecture_faces.png)
- [Largest metric-gain/loss cases as faces](../full_validation_results/ba_n29_n30_15Jul/N29_N30_metric_gain_loss_faces.png)
- [Selected full-image comparisons](../full_validation_results/ba_n29_n30_15Jul/N29_N30_selected_full_images.png)
- [Reproducible PDF config](../infer_tools/full_val_n29_n30_15jul_report.yaml)

![Enlarged N29/N30 face comparison](../full_validation_results/ba_n29_n30_15Jul/N29_N30_closeup_faces_vs_key.png)

## Bottom line

1. **N29 is the best safe BA result so far.** It preserves PhotoMaker's target pose, head
   position, body, clothing, occluders, and scene while making real face-local changes. It is
   materially safer than N17/N24 and more identity-capable than N28's mean-plus-basis memory.
2. **The N29 face changes are produced by trained BA parameters.** PhotoMaker and its ID encoder
   are frozen; only target-ID K/V, the face-delta projection, and its gate train. The output is
   explicitly `PhotoMaker attention + BA face delta`.
3. **That does not yet prove the changes are identity-causal.** PhotoMaker and BA see the same
   reference. BA may learn a partly generic face correction on top of a PhotoMaker identity. A
   branch-off and BA-only reference-swap test is needed to distinguish these cases.
4. **N30 does not justify bbox-normalized QFormer input.** It is visually safe, but less stable
   and no better overall than N29. A square crop is not canonical face alignment and shifts the
   frozen PhotoMaker QFormer's input distribution.
5. **More N29 training should produce more face evolution, but it is not currently on a reliable
   path to beat PhotoMaker.** The 2k-to-10k trend is mildly positive, while gains and losses remain
   prompt-dependent. A blind 50k run risks learning stronger generic edits rather than stronger
   reference dependence.
6. **The next priority is causal identity use, then richer face memory.** One 10k run should train
   correct-reference versus wrong-reference dependence. A separate 10k run should replace the
   two-token memory with identity-specific face-part tokens while retaining the safe hard-bbox
   residual.

## What the images show

### Geometry and image integrity

N29 and N30 preserve the target generation substantially better than the older spatial BA paths.

- **Keanu rushing:** the head stays on the body, with the PhotoMaker neck length, three-quarter
  orientation, hair boundary, suit, and subway composition. The N17 long-neck/displaced-face
  failure does not return.
- **Jisoo skiing and kickboxing:** goggles, gloves, face opening, hair, and body pose remain intact.
  N24's hand/hair contamination is absent.
- **Marion crying:** the hands remain correctly placed around the face; the branch changes facial
  detail without replacing the hand/face layout.
- **Small moving figures such as Lex dancing:** the face remains attached and correctly scaled.
- **Non-face content:** the full-image sheets show almost unchanged clothing, body, scene, camera,
  and pose. Differences are concentrated around the face even though iterative denoising allows a
  small downstream pixel difference outside the literal box.

This validates the current preservation topology: standard PhotoMaker is the base path, the BA
change is a hard-target-box residual, and the final epsilon merge protects the rest of the image.

### Are N29 faces just PhotoMaker faces?

No, but they are still best described as **PhotoMaker faces with a learned BA identity delta**, not
as independently generated BA faces.

Visible N29 changes include:

- different cheek and jaw width;
- eye aperture and brow shape changes;
- mouth and nasolabial changes under expression;
- beard, moustache, skin texture, and apparent age changes;
- substantial checkpoint evolution in dancing/reading Keanu, crying/dancing Eddie, dancing
  Jensen, and several Lex cases.

The strongest 2k-to-10k face evolution occurs in dancing Keanu, reading Keanu, crying Eddie,
dancing Jisoo/Eddie/Jensen, and jumping Lex. Thus the face is not stuck at the PhotoMaker output.

However, not every change is a better identity:

- Eddie often becomes older and greyer; added beard texture is not consistently supported by the
  clean-shaven/moustached reference.
- Chef Lex can gain a goatee even though the reference does not have one. Its metric improvement is
  therefore not convincing visual evidence.
- Male dynamic-expression prompts change more than many Jennie/Jisoo/Marion prompts. Several
  female outputs remain very close to PhotoMaker.
- Some large metric losses, notably rushing Keanu, remain visually plausible identity matches.
  This demonstrates why individual score movements should not drive architecture decisions.

N29 therefore clears the first requirement, "BA visibly affects the face without damaging the
image," but not yet the stronger requirement, "BA consistently supplies a better reference
identity than PhotoMaker."

### N29 versus N30

N30's crop does not produce a consistent visual identity advantage.

- N30 10k changes the face slightly more than N29 10k in aggregate, but the direction is not more
  consistently reference-correct.
- N30 is visibly unstable at 6k in several faces and recovers by 10k rather than improving
  monotonically.
- N29 is better for the current long-run candidate because it has the stronger overall trend and
  fewer crop-specific changes.
- A hard square crop removes some background, but it does not normalize yaw, expression, scale,
  lighting, or landmarks. It also feeds the frozen PhotoMaker vision/QFormer stack a distribution
  different from the one on which it was trained.

The result does not disprove reference normalization in general. It disproves this simple
`bbox crop -> frozen QFormer` form as a useful next default.

## Same-seed distance from PhotoMaker

Normalized pixel MAE is used only to establish whether and where the result moved. It is not an
identity-quality metric.

| Run | Full image | Target face crop | Outside target face crop |
|---|---:|---:|---:|
| N24 dual gate 10k | 0.0452 | 0.1381 | 0.0376 |
| N27 spatial ROI 10k | 0.0267 | 0.0517 | 0.0245 |
| N28 mean+basis 10k | 0.0223 | 0.0597 | 0.0191 |
| N29 QFormer 2k | 0.0219 | 0.0566 | 0.0189 |
| N29 QFormer 6k | 0.0221 | 0.0584 | 0.0191 |
| N29 QFormer 10k | 0.0223 | 0.0596 | 0.0191 |
| N30 crop 2k | 0.0214 | 0.0524 | 0.0189 |
| N30 crop 6k | 0.0222 | 0.0603 | 0.0190 |
| N30 crop 10k | 0.0224 | 0.0623 | 0.0191 |

N29 2k versus 10k has face-crop MAE 0.0426, so later training makes a real face change. N29 and
N30 at 10k differ by 0.0376 in the face crop. The near-constant distance from PhotoMaker does not
mean training stopped; it means the learned face moves within a similarly sized neighborhood of
the PhotoMaker solution.

## ID metrics, secondary evidence

All runs detect 96/96 faces.

| Run | Mean ID similarity |
|---|---:|
| PhotoMaker | **0.4886** |
| N27 10k | 0.4669 |
| N28 10k | 0.4580 |
| N29 2k / 6k / 10k | 0.4634 / 0.4692 / **0.4706** |
| N30 2k / 6k / 10k | 0.4605 / 0.4521 / **0.4655** |
| N24 10k | 0.3899 |

N29 is encouraging relative to recent BA controls, not yet relative to PhotoMaker:

- it improves from 25/96 images above PhotoMaker at 2k to 37/96 at 10k;
- 19/96 are more than 0.02 above PhotoMaker at 10k, while 44/96 are more than 0.02 below;
- its strongest identity means are Jennie 0.5958, Jisoo 0.5912, and Marion 0.4652, each around or
  slightly above PhotoMaker;
- Eddie, Jensen, Keanu, and Lex remain below PhotoMaker overall.

The slowing aggregate gain from 6k to 10k and mixed per-prompt direction argue against assuming
that duration alone will cross the PhotoMaker baseline.

## 1. Is the new face carried by branched attention?

### What can already be established

**Yes: the difference from PhotoMaker is generated by the BA residual.** The code path is direct:

1. PhotoMaker's UNet and ID encoder are frozen in
   [`lora2.py`](../src/model/photomaker_branched/lora2.py#L259).
2. With `train_ba_only`, all UNet parameters are frozen and only target-ID K/V,
   `face_delta_out`, and `face_residual_gate` are re-enabled for this mode in
   [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L60).
3. N29 preserves the two distinct frozen QFormer outputs in
   [`model_v2_NS.py`](../src/model/photomaker_branched/model_v2_NS.py#L160).
4. Target-face CA attends to those tokens, masks the delta, and returns `pm_out + face_delta` in
   [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L913).

Comet's trainable-weight diagnostics independently show a live branch:

| N29 diagnostic | step 0 | step 2k | step 6k | step 9.95k |
|---|---:|---:|---:|---:|
| target-ID K/V LoRA norm | 0.000 | 7.950 | 12.163 | 13.629 |
| face-delta output norm | 0.001 | 5.884 | 7.990 | 8.703 |

The branch is neither dead nor accidentally frozen.

### What cannot yet be established

The current evaluation cannot prove that the BA delta is **identity-causal**, because ordinary
PhotoMaker conditioning and BA memory are both derived from the same reference. The BA path could
learn an identity-conditioned correction, a generic face-improvement prior, or a mixture.

The decisive test is a four-way same-checkpoint intervention:

| Condition | PhotoMaker reference | BA memory reference | Expected interpretation |
|---|---|---|---|
| PM control | correct | residual scale 0 | must reproduce PhotoMaker |
| normal BA | correct | correct | current N29 output |
| wrong BA | correct | different identity | face should move toward wrong BA identity only inside bbox |
| null BA | correct | zero memory | should return the PM control |

Use identical prompt, seed, target bbox, and checkpoint. Measure face-crop similarity to both the
correct and swapped identities, and inspect enlarged crops blindly. If swapping only BA memory
changes identity in the expected direction while the PM scene/pose stays fixed, attribution is
causal. Branch weight norms and normal-versus-PM pixel differences alone cannot provide that proof.

## 2. Will more training make faces differ more from PhotoMaker?

**Probably somewhat; not necessarily in a better direction.**

Evidence for continued learning:

- N29's target-ID and face-delta norms continue growing through 10k.
- N29 2k versus 10k has a substantial 0.0426 target-face MAE.
- visual evolution is clear in multiple difficult prompts.
- mean ID similarity and count of PhotoMaker-beating images improve from 2k to 10k.

Evidence against relying on duration alone:

- face distance from PhotoMaker only rises from 0.0566 to 0.0596;
- mean ID gain slows after 6k;
- several faces oscillate rather than move monotonically toward the reference;
- the objective rewards correct generated identity, but does not explicitly require the BA memory
  to be responsible for that identity;
- the frozen PhotoMaker path is already a strong local optimum, so a residual can learn generic
  age, texture, beard, or expression corrections without solving reference dependence.

More N29 training is useful as a trajectory experiment. It is not yet justified as the primary
architecture strategy.

## 3. What recent runs show works and does not

### Works

- **Hard target bboxes and hard PM preservation:** solve the displaced-face/long-neck/reference
  layout failure class while keeping the rest of the image PhotoMaker-like.
- **Residual rather than absolute replacement:** N27-N30 are much safer than N17/N24.
- **No spatial reference UNet branch for identity-only correction:** N28-N30 avoid copying hands,
  goggles, hair layout, and reference pose.
- **Distinct identity-specific tokens:** N29 is better than N28's averaged identity plus global
  basis offsets.
- **Full-reference QFormer input:** currently more reliable than the N30 square crop.
- **Frozen PhotoMaker base:** gives a clear preservation contract and makes BA attribution of
  output differences possible.

### Does not work, or remains insufficient

- **N24 absolute dual-attention arbitration:** the learned-gating idea is useful, but blending two
  absolute outputs damaged geometry and identity. Arbitration should gate a bounded residual, not
  choose between incompatible full outputs.
- **N28 mean-plus-basis memory:** token selection is mostly identity-independent and too weak.
- **N30 naive bbox normalization:** removes context but creates a frozen-encoder distribution shift
  without actually canonicalizing the face.
- **Two QFormer tokens alone:** safer and better than N28, but still a very compact memory from the
  same encoder family already driving PhotoMaker.
- **Current scalar residual gate:** Comet reports exactly 1.0 throughout N29/N30. It supplies no
  spatial, head-wise, timestep, or identity-dependent arbitration.
- **Positive-only identity supervision:** it says what output identity should be, but does not test
  whether changing BA memory changes that identity.

## 4. Best next two 10k architecture experiments

These should be separate experiments so each answers one question.

### Run A: N29 plus counterfactual BA-memory dependence

Keep all N29 generation architecture unchanged:

- frozen PhotoMaker and frozen PhotoMaker ID encoder;
- two genuine QFormer tokens from the full reference;
- target-face CA residual only;
- hard target bbox and hard PM epsilon preservation;
- no spatial reference UNet, CAMIX, or pose-adapt blending.

Add a paired wrong-reference branch only on BA-active training samples. The normal PhotoMaker
conditioning remains correct; only BA memory is shuffled to another identity in the batch. Train a
ranking objective so correct BA memory predicts the target face better than wrong BA memory:

```text
L_depend = max(0, margin + L_face(correct_memory) - L_face(wrong_memory))
```

`L_face` can initially be the existing hard-box diffusion reconstruction loss, avoiding another
face metric in the core loop. The existing reference ID loss remains on the correct branch.

Required opt-in switches should preserve old behavior, for example:

```yaml
model:
  ba_identity_dependence_mode: paired_wrong_reference  # default: none
  ba_identity_dependence_region: hard_face_bbox
```

Decisive 10k result:

- normal BA improves reference identity without alignment regressions;
- wrong-memory inference moves identity toward the wrong person;
- null/scale-zero BA returns PhotoMaker;
- correct-versus-wrong ranking separates over training.

This run directly answers whether BA carries identity. It is higher priority than changing loss
weights or training speed.

### Run B: hard-reference-face part-token memory

Keep N29's target-side residual and preservation contract, but replace the two-token memory with a
richer identity-specific memory:

- extract frozen PhotoMaker/CLIP patch features from the normal full reference;
- select patch tokens using the existing hard reference bbox rather than cropping/resizing the
  whole input as N30 does;
- fuse the frozen InsightFace global embedding into each selected token;
- use a small trainable resampler to produce about 8 distinct face-part tokens;
- feed only those tokens to target-face CA.

This is not N27's spatial reference UNet branch. The memory contains compact face-encoder tokens,
not a full reference latent grid, so pose/layout copying remains constrained. Each token is
identity-specific, unlike N28's global basis offsets.

Required compatibility switch:

```yaml
model:
  ba_identity_memory_mode: face_patch_resampler  # old: qformer_tokens
  ba_identity_patch_mask: hard_reference_bbox
```

Decisive 10k result:

- larger and more consistently reference-correct face changes than N29;
- no return of reference pose, hair, hand, or background transfer;
- hard cases retain N29's face/body alignment;
- swapped memory changes identity, not target pose.

Do not combine Run A and Run B initially. If both work, combine the richer memory with the
counterfactual objective in a short confirmation run before committing to long training.

### Where N24's idea still fits

A later architecture can replace the current scalar gate with a bounded per-head/per-position gate
conditioned on target hidden state and pooled identity tokens:

```text
output = PM + hard_mask * bounded_gate(target, identity, timestep) * BA_delta
```

This preserves N24's learned arbitration idea without blending absolute PM and reference outputs.
It is not the first next run because N29's current problem is insufficient identity causality, not
excessive artifacts. A smarter gate can otherwise learn to suppress the already weak branch.

## 5. Should a 50k long run start now?

### Current recommendation

**Do not start an unconditional 50k N30 run.** N30 has no demonstrated advantage.

For N29, a guarded continuation is reasonable, but treat 20k as the first decision point:

1. Resume N29 10k rather than restart if the checkpoint is available.
2. Continue validation on the same 96 images every 2k through 20k.
3. At 14k and 20k, compare enlarged faces to PhotoMaker and N29 10k, with special attention to
   dancing/reading Keanu, crying/dancing Eddie, Lex dynamic prompts, Jisoo skiing/kickboxing, and
   Marion crying/laughing.
4. Continue toward 30k/50k only if identity-specific visual wins broaden while alignment and
   non-face preservation remain intact.
5. Stop if the branch mainly increases age, facial hair, sharpness, or expression intensity, or if
   correct-versus-wrong BA memory is not separable.

If a 50k allocation must be chosen before the next architecture tests, **N29 full-reference
QFormer tokens is the only defensible current setup**. It is the safest and strongest recent BA
base. However, the most promising eventual long run is Run A, or Run A combined with Run B after
both pass their 10k tests, because those setups explicitly make improved identity depend on BA
rather than merely allowing a BA residual to adjust a strong PhotoMaker face.

## Recommended decision order

1. Run the branch-off/null/wrong-reference intervention on N29 10k.
2. Train Run A for 10k to enforce causal BA identity use.
3. Train Run B for 10k to test richer identity-specific memory.
4. Choose by blind face crops and alignment first, metrics second.
5. Start the 50k run only from the architecture that passes both identity-causality and image
   preservation checks.

