# N31, N32, and N33 visual and architecture analysis

## Scope and artifacts

Analyzed runs:

- `ba_identity_dependence_4gpu_N31`: complete 96-image results at 2k, 6k, 10k, and 12k.
- `ba_facepatch_resampler_N32`: complete results at 2k, 6k, 10k, and 16k.
- `ba_qformer_continue40k_N33`: complete results at 14k, 20k, 24k, and 26k.

Artifacts:

- [Full 96-image comparison PDF](../full_validation_results/ba_n31_n32_n33_16Jul/full_val_report_N31_N32_N33_vs_key.pdf)
- [Enlarged face comparison](../full_validation_results/ba_n31_n32_n33_16Jul/N31_N32_N33_closeup_faces_vs_key.png)
- [N31 desaturation face progression](../full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_face_progression.png)
- [N31 desaturation full images](../full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_full_images.png)
- [Computed visual statistics](../full_validation_results/ba_n31_n32_n33_16Jul/visual_statistics.json)
- [PDF configuration](../infer_tools/full_val_n31_n32_n33_16jul_report.yaml)

The main evidence is visual: target pose, head/body alignment, expression, facial geometry,
local artifacts, color stability, and whether changes move toward the reference. ID metrics are
secondary because small changes do not reliably distinguish useful identity improvement from
expression, contrast, or color changes.

![Close-up comparison](../full_validation_results/ba_n31_n32_n33_16Jul/N31_N32_N33_closeup_faces_vs_key.png)

## Executive conclusion

1. **The target-only residual topology remains the right base architecture.** N31, N32, and N33
   keep the head attached to the body, retain the PhotoMaker pose, and avoid the copied hands,
   goggles, hair, displaced faces, and severe local corruption seen in N3a/N17/N24.
2. **N31 makes BA important, but for the wrong reason.** Its correct/wrong-reference objective is
   successfully optimized, yet it learns desaturation, contrast, and expression shortcuts rather
   than a clean identity correction. This is the strongest evidence so far that branch strength is
   not the missing ingredient by itself.
3. **N32 is safe but not yet a better identity representation.** The complete 16k checkpoint
   confirms that the face-patch resampler keeps changing faces without breaking pose, but the
   changes oscillate rather than becoming consistently closer to the reference.
4. **N33 shows that more training of unchanged N29 is not enough.** From 14k through 26k, faces
   receive small detail and expression changes but remain semantically almost the same PhotoMaker
   identity. There is no monotonic visual identity improvement.
5. **PhotoMaker should remain in control globally.** The next architecture should increase BA
   authority only over identity-relevant face features, not over target pose, expression,
   lighting, or scene structure.
6. **The next decisive change is an identity-causal decoded loss plus a safer adaptive residual.**
   Do not continue N31, remove the PM preservation path, or simply raise residual magnitude.

## Run overview

| Run | Main change | Visual result | Main diagnosis |
|---|---|---|---|
| PhotoMaker | released baseline | coherent and well aligned | identity ceiling to beat |
| N3a 10k | full spatial reference branch and hard face replacement | severe lighting, geometry, and pasted-face artifacts | raw reference spatial transfer is unsafe |
| N24 10k | learned gate between absolute target/reference outputs | stronger face changes, but frequent hair/hand/occlusion transfer | gating incompatible absolute outputs is not enough |
| N29 10k | two QFormer tokens, target-only CA residual, hard PM preservation | clean and slightly different from PM | safest previous BA path, but weak identity causality |
| N31 2k-12k | N29 plus correct/wrong epsilon ranking, global batch 8 | progresses from clean to pale/monochrome and expression-shifted | dependence loss learns nuisance shortcuts |
| N32 2k-16k | eight learned queries over hard-bbox CLIP face patches | clean, modestly more varied faces with no monotonic identity gain | memory has capacity but lacks canonical identity structure and direct causal supervision |
| N33 14k-26k | unchanged N29 continuation | clean but visually plateaued | more steps do not add information missing from the two PM-derived tokens |

## Metrics as secondary evidence

| Run/checkpoint | Mean ID similarity | Visual interpretation |
|---|---:|---|
| PhotoMaker | **0.4886** | current baseline |
| N3a | 0.1709 | consistent with major visible corruption |
| N24 10k | 0.3899 | stronger but unsafe face intervention |
| N29 10k | 0.4706 | best safe earlier BA result |
| N31 2k / 6k / 10k / 12k | 0.4640 / 0.4611 / 0.4544 / **0.4480** | declines as desaturation and expression drift increase |
| N32 2k / 6k / 10k / 16k | 0.4394 / 0.4427 / **0.4482** / 0.4453 | peaks at 10k, then slips slightly; still below N29 |
| N33 14k / 20k / 24k / 26k | 0.4697 / 0.4709 / **0.4731** / 0.4663 | no monotonic long-run gain |

The metric trends support the visual conclusions, but the architecture decision does not depend on
the small differences among N29, N32, and N33.

## Visual analysis

### N31: strong branch use, wrong visual direction

N31 is clean at 2k and remains geometrically aligned throughout training. By 6k, many large,
frontal male faces become cooler and paler. At 10k and 12k, several become almost monochrome:

- angry Jensen, Keanu, Lex, and Eddie;
- rushing Keanu, Elon, and Eddie;
- crying Keanu and Elon;
- reading Eddie and Lex.

This is not a minor global color shift. The face/person changes while colored scene details remain,
for example red vehicle lights and colored subway/background elements. The hard PM merge is still
protecting the non-face generation.

N31 also changes expressions in ways unrelated to identity. Rushing Eddie changes from the intended
anxious look toward a broad smile; several angry or crying faces acquire different mouth and eye
configurations. These changes make the branch visibly important, but not identity-specific.

Measured mean face color and difference from PhotoMaker:

| Checkpoint | Mean face chroma | Mean face saturation | Face MAE vs PM |
|---|---:|---:|---:|
| PhotoMaker | 83.34 | 0.446 | 0 |
| N31 2k | 84.35 | 0.438 | 0.0668 |
| N31 6k | 61.28 | 0.345 | 0.1063 |
| N31 10k | 47.97 | 0.275 | 0.1176 |
| N31 12k | 48.42 | 0.277 | 0.1114 |

The strongest individual chroma collapses at 12k versus 2k are:

| Image | Chroma ratio, 12k / 2k |
|---|---:|
| Angry Jensen | 0.03 |
| Angry Keanu | 0.05 |
| Angry Lex | 0.07 |
| Rushing Keanu | 0.09 |
| Angry Eddie | 0.10 |
| Rushing Elon | 0.10 |
| Rushing Eddie | 0.13 |

![N31 desaturation](../full_validation_results/ba_n31_n32_n33_16Jul/N31_desaturation_face_progression.png)

### Why N31 does this

The N31 loss is not an identity loss. It asks the correct memory to predict the target diffusion
noise more accurately than a selected wrong memory inside the face rectangle:

- [`identity_dependence_ranking_loss`](../src/loss/diffusion_loss.py#L35) compares face-region
  epsilon MSE.
- [`select_wrong_identity_features`](../src/model/photomaker_branched/lora2_helpers.py#L493)
  chooses the least cosine-similar flattened memory.
- The correct and wrong predictions share the same PhotoMaker prediction and differ only in BA
  memory in [`lora2.py`](../src/model/photomaker_branched/lora2.py#L566).

That objective can use anything correlated with the reference: skin tone, illumination, contrast,
expression, age, gender, or reference photography style. It never asks whether the decoded output
is closer to the correct person's identity.

There are two additional problems with the negative:

1. It is selected by feature distance without checking identity labels, so it is not guaranteed to
   be a different person.
2. The least-similar candidate is an easy, nuisance-rich negative. It encourages a large generic
   difference rather than a subtle identity discrimination.

Training telemetry confirms that N31 succeeds at its defined objective while image quality worsens:

| Training range | Mean dependence loss | Mean wrong-minus-correct face MSE |
|---|---:|---:|
| 0-2k | 0.0195 | approximately 0 |
| 2k-6k | 0.0112 | 0.0309 |
| 6k-10k | 0.00156 | 0.1196 |
| 10k-12.2k | 0.00086 | 0.1508 |

The target-ID K/V norm grows to `19.84` and the face-delta norm to `10.79`. The branch is active and
has learned to separate correct from wrong memory. The problem is the semantic meaning of that
separation.

N31 also sees four times as many samples per optimizer step as the one-GPU runs: local batch 2 on
four GPUs gives global batch 8. DDP averages gradients, so this does not multiply one update by
four, but 12k steps expose the objective to about 96k training examples. This accelerates the
shortcut. It does not explain the specific grayscale direction by itself.

### N32: safe extra capacity, insufficient identity structure

N32 retains the correct generation topology. Its eight compact tokens produce local face changes
without moving the head or copying reference geometry. Keanu rushing, Jisoo skiing/kickboxing,
Jennie night-ride, and the small dynamic Lex faces remain attached and coherent.

The changes are not consistently identity-improving:

- some eye, mouth, and cheek shapes move away from PhotoMaker, but not clearly toward the reference;
- faces often become smoother or simply different;
- there is no broad visual gain from 6k to 10k or from 10k to 16k;
- mean ID similarity remains below N29.

The complete 16k checkpoint is useful because it rules out the earlier possibility that N32 merely
needed a few more steps. Its mean target-face MAE versus PhotoMaker increases from `0.0735` at 10k
to `0.0776` at 16k, so the branch is not frozen. The mean face change from 10k to 16k is `0.0469`,
comparable to the `0.0446` change from 6k to 10k. Visually, the largest changes affect examples such
as Elon kickboxing, Eddie crying/dancing, and Keanu dancing/rushing, but they mainly alter
expression, mouth/eye shape, and texture. They do not form a consistent reference-identity trend.
Color and target alignment remain stable, unlike N31.

The resampler maps an unaligned hard-bbox subset of CLIP patch features using global InsightFace
queries in [`identity_memory.py`](../src/model/photomaker_branched/identity_memory.py#L100). This is
safer than a reference UNet grid, but it still entangles reference pose, expression, lighting, and
crop geometry. It has no landmark-canonical face parts and no loss that forces each token to carry
identity-specific information.

The logged total resampler norm (`77.60` to `77.67`) is not a useful learning diagnostic because it
is dominated by the full parameter norm. At 16k, the target-ID K/V norm is `12.47` and the
face-delta norm is `8.77`, so N32 is not frozen. The extra training changes the output but does not
resolve the identity-information bottleneck.

### N33: stable but saturated

N33 is the cleanest long trajectory. Across 14k, 20k, 24k, and 26k:

- body pose, head position, occlusions, and scene composition remain correct;
- no N3a/N24-style pasted content reappears;
- most faces retain the same semantic identity and geometry as N29/PhotoMaker;
- checkpoint differences are mostly wrinkles, mouth opening, eye shape, skin texture, and small
  expression changes.

The target-ID K/V norm continues from `13.64` at the N29 10k checkpoint to `15.98` at N33 26k, and
the face-delta norm grows from `8.71` to `10.07`. More parameter movement therefore does not produce
more useful identity movement.

N33 reuses the same two PhotoMaker QFormer tokens that already condition the PhotoMaker prompt.
It adds a second route for nearly the same identity representation, not a new identity signal.
Longer training can learn a stable local edit, but it cannot recover facial evidence absent from
that bottleneck.

## 1. Compared with older runs: what works and what does not

### What works

- **Target-coordinate queries:** the generated face queries attend identity memory, so target pose
  and head placement come from the generated image.
- **Compact identity memory:** N29/N31/N32/N33 do not run a full reference latent through the UNet.
- **Additive residual:** BA adds a zero-initialized face correction over the standard PM
  cross-attention output in
  [`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py#L915).
- **Hard target bbox:** the residual is masked at the target face tokens.
- **Hard PM epsilon preservation:** outside the target bbox, the prediction is exactly the PM
  prediction in [`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py#L274).
- **Frozen PhotoMaker base and ID encoder:** the scene prior remains stable and comparisons remain
  interpretable.

These decisions solve the old alignment and gross-artifact problem.

### What does not work

- **Full spatial reference memory and absolute replacement:** N3a/N17 transfer reference geometry
  and nuisance content.
- **Gating two absolute outputs:** N24 reduces but does not solve incompatible target/reference
  coordinates.
- **Two PM-derived QFormer tokens alone:** N29/N33 are safe but plateau near PhotoMaker.
- **Epsilon-space correct/wrong ranking:** N31 creates strong non-identity shortcuts.
- **Unaligned CLIP patch resampling without direct identity supervision:** N32 adds capacity but not
  reliable identity correspondence.
- **A single scalar residual gate:** it has no layer, head, timestep, target-query, or identity
  confidence dependence.

## 2. Key changes versus N3a and other hard-BA runs

N3a uses `noise_and_ref` branched weights and a full reference branch in
[`start_ba_nr_alt_vast_N3a.sh`](../serv_new_runs/start_ba_nr_alt_vast_N3a.sh).

The legacy self-attention path:

- uses same-index target/reference face-token mixing
  ([processor lines 541-552](../src/model/photomaker_branched/attn_processor_cleanest.py#L541));
- builds an absolute face attention output from reference spatial K/V;
- replaces the target face-region hidden output
  ([processor lines 636-646](../src/model/photomaker_branched/attn_processor_cleanest.py#L636)).

The legacy cross-attention path also processes a doubled target/reference batch and computes the
reference half from reference queries and a face prompt
([processor lines 1000-1102](../src/model/photomaker_branched/attn_processor_cleanest.py#L1000)).
The reference branch therefore carries spatial pose, crop, hair, hands, lighting, and background
information that is not aligned to the target head.

N29 and the latest runs changed the contract:

1. Run the ordinary PhotoMaker target path.
2. Build compact, non-spatial identity tokens.
3. Let target face queries attend those tokens.
4. pass the attention result through a zero-initialized residual projection.
5. apply it only inside the hard target bbox.
6. restore PhotoMaker epsilon outside the bbox.

The relevant current implementation is
[`_target_face_residual_forward`](../src/model/photomaker_branched/attn_processor_cleanest.py#L915)
and [`run_branched_forward_pass`](../src/model/photomaker_branched/lora2_helpers.py#L381).

This is why N31 can become very influential while the head remains correctly placed. Its failure
is now appearance/objective drift inside the face, not reference-coordinate transfer.

## 3. How to increase BA importance without returning to old artifacts

The answer is **not** to scale the current residual, train N31 longer, remove the hard PM merge, or
restore a spatial reference UNet.

Use the existing target-only residual as the safety envelope, then improve three components.

### A. Supervise decoded identity causality

For the same noisy target, run correct, wrong, and null/disabled BA memories. On low-noise
timesteps, decode the predicted `x0` face and use a frozen face recognizer:

- correct-memory output must be closer to the correct reference identity than PM/null output;
- correct-memory output must be closer to the correct identity than wrong-memory output;
- wrong-memory output should move toward the wrong identity rather than merely become worse;
- negatives must be selected by known person ID, with same-domain or semi-hard sampling.

Keep target-geometry safeguards:

- landmark/pose/expression consistency to the target or PhotoMaker output;
- low-frequency color/chroma consistency to prevent the N31 shortcut;
- outside-bbox hard PM preservation;
- optionally a small face-boundary consistency loss.

This changes the objective from “memory changes epsilon reconstruction” to “memory causally changes
who the person is.”

### B. Use canonical identity-part memory

Build a successor to N32:

1. Landmark-align the reference face to a canonical crop.
2. Extract multi-scale face-recognition features, not only final CLIP patches.
3. Produce ordered identity-part tokens for eyes, nose, mouth, face contour, and global identity.
4. Initialize or anchor them around N29's stable QFormer tokens, learning only additional part
   residuals initially.
5. Let target queries supply pose and expression; identity tokens should not carry target layout.

This provides genuinely new facial evidence beyond PhotoMaker's two-token bottleneck without
reintroducing a raw reference spatial grid.

### C. Make residual arbitration real and bounded

The current `face_residual_gate` is created at value `1` in the attention module and remains exactly
`1.0` in every N31/N32/N33 Comet record
([initialization](../src/model/photomaker_branched/attn_processor_cleanest.py#L857),
[logging](../src/trainer/sdxl_trainers.py#L290)). It is trainable in the optimizer selection, but it
is stored in the attention dtype. With BF16 training and a value near one, `1e-4`-scale updates can
quantize away. In practice it is not providing learned arbitration.

Replace it behind a new switch with:

- an FP32 logit initialized at zero;
- a bounded sigmoid/tanh scale;
- per-layer or per-head values;
- optional conditioning on timestep, target face query statistics, and identity-memory confidence.

Keep the output as `PM + gate * identity_delta`. Do not interpolate two absolute PM/reference
outputs as N24 did.

### D. Restrict identity intervention by UNet resolution

The current target-face residual is installed across all selected cross-attention layers. Low
resolution layers can affect coarse shape, expression, and color. A stronger branch should first
operate in mid/high-resolution up blocks, where it can alter eyes, nose, mouth, skin detail, and
face contour with less authority over head placement and scene geometry.

This is an architectural separation of responsibilities:

- PM and low-resolution UNet features control pose, composition, and coarse head geometry;
- BA mid/high-resolution adapters control identity-specific facial detail;
- the FP32 gate decides how much correction each layer/head should apply.

## 4. Why N31 faces become black and white

The grayscale effect is a learned shortcut, not a filename, PDF, inference-seed, or face-detection
problem.

Evidence:

- it appears progressively from 2k to 6k to 10k;
- it is strongest in the hard face/person region while colored background details remain;
- the correct/wrong dependence margin becomes strongly satisfied at the same time;
- target-ID K/V and residual norms grow rapidly;
- all 96 faces are still detected and target geometry remains coherent;
- unchanged N33 and N32 do not show the same systematic collapse.

The objective rewards any predictable correct-versus-wrong difference in face-region epsilon.
PhotoMaker QFormer tokens contain identity mixed with appearance and photographic nuisance.
InsightFace-style ID supervision is also largely insensitive to color. There is no explicit chroma
or target-appearance constraint inside the face box. The easiest stable solution is therefore
allowed to remove or remap color while changing expression/contrast enough to distinguish memory
conditions.

This reveals an additional flaw: **hard PM preservation protects location and background, but does
not define what BA is allowed to change inside the face.** A strong identity branch needs positive
identity supervision and explicit target-geometry/color safeguards.

## 5. Architectural improvements to try next

### Priority experiment A: causal high-resolution identity adapter

Keep N29's compact target-only residual and hard PM preservation. Change:

- identity-labelled correct/wrong/null counterfactual decoding;
- direct frozen-recognizer identity ranking on decoded faces;
- target landmark/expression and chroma preservation;
- BA residual only in mid/high-resolution up-block cross-attention;
- FP32 bounded per-head/per-layer residual gate.

This is the highest-priority experiment because it directly tests whether BA can become more
identity-important without receiving authority over pose and global appearance.

### Priority experiment B: canonical face-part token memory

Keep the same safe residual and objective from experiment A, but replace the two QFormer tokens
with landmark-aligned multi-scale face-part tokens. Initialize the new memory path as a residual
around the QFormer baseline so step 0 reproduces N29 behavior.

This tests the remaining information bottleneck. It is the better successor to N32 than simply
training the current resampler longer.

### Explicitly avoid

- continuing N31;
- a long continuation of unchanged N33;
- increasing the current unbounded residual scale;
- disabling the hard PM epsilon merge;
- reintroducing full reference latent grids;
- POSE_ADAPT_RATIO/CAMIX blending;
- N24-style interpolation of absolute outputs.

## Recommended decision rule

Do not start a 50k-100k run until a 10k architecture test passes all of these:

1. correct-reference BA beats null/wrong-reference BA on decoded identity similarity;
2. improvement is visible in blind enlarged face crops, not only metrics;
3. target landmarks, pose, expression, and face/body alignment remain stable;
4. no systematic chroma, contrast, or style drift;
5. BA-off output reproduces PhotoMaker and BA-on changes are localized to identity-relevant face
   details;
6. the gain continues from 2k to 6k to 10k rather than plateauing or changing direction.

The latest runs show that the safe topology has enough capacity. The missing piece is a correctly
defined identity-causal learning signal and an identity representation that contains more useful
face evidence than PhotoMaker's existing two-token bottleneck.
