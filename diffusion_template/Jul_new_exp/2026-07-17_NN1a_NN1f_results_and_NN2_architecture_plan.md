# NN1a–NN1f visual results and NN2 architecture plan

Date: 17 July 2026

Status: analysis and proposals only. The NN2 model changes, configs, and
launchers described below have **not** been implemented.

## Executive conclusion

All six NN1 runs are active and change the generated face substantially. The
problem is not a dead branch, flat checkpoint, or PhotoMaker dominance. The
problem is that full spatial BA learns an unsafe identity route:

- target-face queries receive unaligned reference-coordinate K/V at every
  self-attention layer;
- the resulting face hidden state has absolute authority inside a broad target
  bbox;
- reference pose, hair, glasses, lighting, hands, and face layout are therefore
  transferred as if they were aligned identity evidence.

NN1a–c progressively turn faces into hard orange/white/black masks and
collages. NN1d–f are more photorealistic because branched CA weights are
frozen, but they still fold, duplicate, displace, or erase facial features on
hard poses and occlusions. NN1e/f's higher identity scores are not visual
improvements: they coincide with smoothed-away landmarks and fewer detected
faces.

The best conclusion is:

1. **NN1d is the right inheritance point, not a run to continue.** Keep both
   processor classes active, keep all branched CA weights frozen, and remove
   the current decoded ID loss from the first next-round attribution.
2. **Do not train NN1a–f longer or merely rerun them on two GPUs.** The artifact
   is a deterministic spatial-correspondence failure already visible by 2k;
   a larger batch cannot repair it.
3. **Retain core BA but change its spatial arbitration.** The next work should
   preserve the doubled target/reference U-Net and target-Q/reference-KV
   mechanism while introducing real reference ROI tokens, an explicit
   target-face geometry lane, layer specialization, or confidence fallback.
4. **Use six one-GPU architecture screens first.** Run NN2a–f to 6k with full
   validation at 0/2k/4k/6k. Promote only a visually clean winner to a two-GPU
   longer run.

## Artifacts

- [Full 96-image PDF comparison](../full_validation_results/ba_NN1a_NN1f_17Jul/full_val_report_NN1a_NN1f_vs_PM.pdf)
- [2k/6k/10k face progression](../full_validation_results/ba_NN1a_NN1f_17Jul/NN1a_NN1f_closeup_faces_progression.png)
- [10k enlarged face comparison](../full_validation_results/ba_NN1a_NN1f_17Jul/NN1a_NN1f_closeup_faces_10k.png)
- [10k selected full images](../full_validation_results/ba_NN1a_NN1f_17Jul/NN1a_NN1f_full_images_10k.png)
- [Computed visual statistics](../full_validation_results/ba_NN1a_NN1f_17Jul/visual_statistics.json)
- [Reproducible PDF config](../infer_tools/full_val_NN1a_NN1f_17jul_report.yaml)
- [Interactive architecture explorer](../debug_04Jul/ba_architecture_explorer/index.html)

The PDF was generated with the repository's existing
`infer_tools/pdf_full_val.py` workflow. It has one summary page, one
configuration page, and eight identity pages. Every included run column
matched 96/96 images.

NN1a and NN1c step 2k were intentionally omitted from the complete PDF because
their downloaded folders contain only 94 images. The progression sheet retains
these checkpoints and marks the missing cells. The missing files are an upload
issue, not absent inference:

- NN1a 2k: `Chef woman_jisoo.png`, `Night-ride_jisoo.png`;
- NN1c 2k: `Chef woman_jennie.png`, `Night-ride_jennie.png`.

The close-up sheets use the fixed `face_crop_new` boxes from
`pm96_bboxes_new.json`, expanded for visual context. The numerical face MAE
uses the unexpanded fixed box.

![Face progression](../full_validation_results/ba_NN1a_NN1f_17Jul/NN1a_NN1f_closeup_faces_progression.png)

## Experimental matrix

| Run | Isolated change from NN1a | 10k visual result |
|---|---|---|
| NN1a | guarded N3a replay | reproduces destructive N3a spatial drift |
| NN1b | train only in BA-active inference timestep region | essentially the same mask/collage failure |
| NN1c | explicitly mask non-ID reference-prompt tokens | strongest and most binary collapse |
| NN1d | branched CA active but cloned CA weights frozen | cleanest NN1, yet repeated/folded features remain |
| NN1e | NN1d + decoded reference-ID loss | higher metric but faceless smoothing and detection loss |
| NN1f | NN1e, train only spatial reference K/V | selective ownership does not fix an unsafe K/V source |

All runs use one GPU, physical/effective training batch 2, validation batch 12,
10k optimizer steps, and full validation every 2k. All retain:

- one doubled `[target, reference]` U-Net call;
- 70 `BranchedAttnProcessor` self-attention sites;
- 70 `BranchedCrossAttnProcessor` cross-attention sites;
- target-face Q attending reference-face spatial K/V;
- direct return of the target epsilon half;
- text-only inference steps 0–9, PhotoMaker steps 10–14, and spatial BA steps
  15–49.

## Metrics as secondary evidence

The visuals determine the decision. Identity similarity can increase when a
face recognizer accepts a distorted, smoothed, or identity-correlated texture,
and can decrease on a coherent face with glasses, hands, or a difficult pose.

| Run | Mean ID at 2k | Mean ID at 6k | Mean ID at 10k | Faces detected at 10k |
|---|---:|---:|---:|---:|
| NN1a | not downloaded | 0.1797 | 0.1801 | 93/96 |
| NN1b | 0.1471 | 0.1765 | 0.1715 | 93/96 |
| NN1c | not downloaded | 0.1482 | 0.1460 | 93/96 |
| NN1d | 0.1818 | 0.2302 | 0.2272 | **96/96** |
| NN1e | 0.2097 | 0.2603 | **0.2701** | 94/96 |
| NN1f | 0.1992 | **0.2621** | 0.2472 | 95/96 |

Fixed-box RGB MAE from the same-seed PhotoMaker face establishes movement, not
quality:

| Run | Face MAE vs PM at 2k | 6k | 10k | Face change 2k→6k | Face change 6k→10k |
|---|---:|---:|---:|---:|---:|
| NN1a | 0.1656 | 0.1892 | 0.2000 | 0.1701 | 0.1186 |
| NN1b | 0.1747 | 0.2042 | 0.2082 | 0.1846 | 0.1226 |
| NN1c | 0.2486 | 0.2470 | **0.2533** | 0.1475 | 0.1065 |
| NN1d | 0.1619 | 0.1488 | 0.1511 | 0.1021 | 0.0820 |
| NN1e | 0.1645 | 0.1530 | 0.1600 | 0.1087 | 0.0931 |
| NN1f | 0.1670 | 0.1607 | 0.1641 | 0.1028 | 0.0896 |

NN1a's final `0.2000` face MAE and `0.1801` identity score closely reproduce
the historical N3a result (`0.20616`, `0.1710`). That is useful evidence that
the restored N3a-era path and correctness guards are behaving as intended.

The outside-fixed-face MAE stays in a narrow range of roughly `0.038–0.044`.
The full-image sheet confirms that clothing, body pose, and background remain
substantially stable. The critical failure is local to the head/face route,
although denoising lets local changes propagate slightly beyond the literal
bbox.

## Detailed visual analysis

### NN1a: guarded N3a replay

NN1a establishes that correctness guards 1–4 do not solve the model's
architectural weakness. At 2k many simple faces are still plausible. By 6k and
10k:

- skin becomes unnaturally orange or white;
- the face is flattened into a high-contrast mask;
- hair, mouth, and eyes split into incompatible regions;
- the reference frontal layout is imposed on target expressions and poses;
- face/head boundaries stretch into the neck or surrounding hair.

Keanu rushing develops an orange, elongated face/neck insert. Jensen skiing
loses coherent eyes and goggles. Jisoo skiing and laughing acquire large white
holes and displaced black/orange fragments. This is strong BA authority in the
wrong coordinate system.

### NN1b: schedule-matched BA training

NN1b answers audit issue 5 cleanly: sampling only the BA-active inference
region does not repair the failure. Its trajectory and 10k result are nearly
NN1a:

- similar white/orange face plates;
- similar long-neck and face-placement failures;
- similar missing facial features under goggles, hair, and hands;
- no meaningful advantage in final identity or detection.

The train/inference timestep mismatch was worth fixing for attribution, but it
is not the cause of N3a's destructive spatial transfer.

### NN1c: explicit reference prompt mask

NN1c is the clearest negative result. Removing the 75 zero-token attention
sinks strengthens reference-half conditioning, but the downstream route is
unsafe. It produces the largest face MAE and the most binary collapse:

- almost pure white/orange/black regions;
- cartoon-like face segmentation;
- hard vertical splits and holes;
- severe loss of local lighting and expression;
- face fragments that ignore goggles, hat openings, hair, and target yaw.

Issue 6 was real as an attention-strength issue, but stronger reference prompt
conditioning makes the wrong spatial behavior more dominant. Do not carry the
explicit token mask into the next round until spatial arbitration is safe.

### NN1d: active/frozen branched CA

NN1d is decisively the best NN1 variant:

- normal skin color is mostly retained;
- all 96 generated faces remain detectable through 10k;
- simple unobstructed examples can look coherent and more reference-like;
- it avoids NN1a–c's broad binary mask collapse.

It is still not usable:

- Jensen/Jisoo skiing show repeated eyes, glasses, and goggle components;
- Jisoo night-ride, chef, and laughing show hair/facial features folded across
  the face;
- kickboxing faces can contain a second small face or displaced feature patch;
- long or pinched neck transitions remain on several male examples;
- Marion with hair or hands shows face structure displaced behind the
  occluder.

This confirms the historical N11/N17 lesson: training branched CA is highly
destabilizing, but frozen CA cannot correct unaligned reference K/V inside
branched self-attention.

### NN1e: NN1d plus decoded identity loss

NN1e has the highest final identity metric, but visuals reject it. On several
Jennie examples, the face becomes a smooth skin-colored surface with missing
eyes/nose; on hard Jisoo and Marion examples, features fold or smear. Face
detection falls from 96/96 at 2k to 94/96 at 10k.

The loss is therefore finding recognizer/crop shortcuts rather than repairing
correspondence. It can reward identity-correlated color, texture, or partial
features even when the generated face is not anatomically coherent. This loss
should remain off in the next architecture screen. A later identity objective
must include structural/landmark validity or be applied only after the spatial
route is visually safe.

### NN1f: reference K/V-only updates

NN1f tests whether freezing target/noise projections protects target geometry.
It does not. The visual result is close to NN1e, with:

- erased or folded faces;
- repeated goggle/eye structures;
- displaced mouth and hair fragments;
- 95/96 detected faces at 10k;
- a metric peak at 6k followed by regression.

This is important: the unsafe source is not merely a trainable target query.
Full reference K/V itself contains unaligned pose, layout, and nuisance
content. Narrowing optimizer ownership cannot turn it into identity-only
evidence.

## Face/body alignment and artifact taxonomy

### What remains aligned

The selected full images show that BA usually preserves:

- target body pose and scale;
- clothing and major props outside the face;
- scene composition, camera, and background;
- the overall PhotoMaker head location.

The hard target region and staged schedule are therefore doing useful work.

### What is not aligned

The local head/face composition fails in four recurring ways:

1. **Reference-layout overwrite.** A frontal reference face is imposed on a
   target yaw, expression, or partial view.
2. **Occluder collision.** Goggles, hats, hair, and hands occupy the same broad
   bbox as identity features; reference and target evidence are both rendered.
3. **Boundary mismatch.** The BA face does not match PhotoMaker hair, jaw,
   neck, or head width, producing pasted faces and long/pinched necks.
4. **Feature duplication or erasure.** Separate source layouts create repeated
   eyes/mouths, while the ID loss can instead suppress landmarks into a smooth
   face-colored patch.

The broad target/reference rectangles localize the operation but do not create
semantic correspondence. Applying the same mechanism at all 70 self-attention
sites also lets reference evidence affect both coarse face geometry and late
appearance detail.

## Do images improve with training?

Not in a way that justifies continuation.

- **NN1a/b:** face movement and corruption increase through 10k. Later images
  are stronger BA images, not better faces.
- **NN1c:** already severely collapsed at 2k; later checkpoints change the
  pattern without repairing it.
- **NN1d:** the metric improves until 6k/8k and then plateaus, while hard-pose
  fold/duplication remains. The 6k checkpoint is not a hidden clean winner.
- **NN1e:** the identity metric rises, but visual anatomy and face-detection
  reliability worsen.
- **NN1f:** the metric peaks at 6k and falls; artifacts remain throughout.

Checkpoint-to-checkpoint face MAE remains substantial from 6k to 10k for every
run, so the models are not frozen. Continued parameter movement is reinforcing
or rearranging the same unsafe route.

## Training and validation log audit

### Healthy evidence

All six completed 10k:

- five checkpoints and five weights-only checkpoints were saved;
- six full 96-image validation passes were executed (step 0 plus every 2k);
- all startup logs report `SA=70 CA=70`;
- NN1a–c report the expected 1,680 trainable branch tensors;
- NN1d/e report 840 trainable SA tensors and frozen CA weights;
- NN1f reports only 280 `sa_ref_k`/`sa_ref_v` tensors;
- there is no CUDA OOM, NaN/Inf report, model exception, or truncated epoch.

The trajectories and large face MAE also independently show that processor
weights are training and reaching validation.

### Operational upload failures

NN1a and NN1c logs each contain six Python tracebacks from Comet visualization
uploads timing out after 900 seconds. Training continues and saves all later
checkpoints. These timeouts explain the two missing 2k files and absent 2k
downloaded metrics; they are not model-training failures.

Future runs should decouple local validation image saving from Comet upload so
network failure cannot make a checkpoint look incomplete.

### Older temporary-validation trainability metadata in NN1a–c

NN1a–c were launched before the validation-constructor parity fix documented
in the implementation guide. Their logs show:

```text
training model:   1680 tensors / 71,598,080 trainable BA parameters
temporary val:    1680 tensors / 210,944,000 trainable parameters
```

NN1d–f, launched after the fix, show identical training/validation counts.

The extra `requires_grad` flags in a `torch.no_grad()` temporary validation
model do not by themselves change its forward values, and strict processor
state transfer completed at every validation. The strong, progressive
checkpoint changes also rule out PhotoMaker-only validation. Therefore this
does not alter the architecture conclusion.

It is still a reproducibility blemish. If NN1a/b/c are needed as exact
numerical baselines, download one final checkpoint (NN1a epoch 5 is enough) and
rerun its 96-image validation under the current constructor-parity code. There
is no need to download all six checkpoints to decide the next architecture.
Downloading NN1d epoch 3/5 would be useful only for extra inference ablations
or attention-map inspection.

## Should any current run continue or move to two GPUs?

| Run | Continue? | Two-GPU rerun? | Reason |
|---|---|---|---|
| NN1a | No | No | destructive N3a behavior successfully reproduced |
| NN1b | No | No | schedule correction does not change failure class |
| NN1c | No | No | strongest prompt makes unsafe spatial path worse |
| NN1d | No, except diagnostics | Not unchanged | best baseline, but systematic geometry/occlusion artifacts |
| NN1e | No | No | identity metric is confounded by faceless smoothing |
| NN1f | No | No | unsafe reference K/V persists despite selective training |

A two-GPU run would double global batch from 2 to 4 and reduce gradient noise.
It would not create target/reference correspondence or prevent an absolute
reference face output from overwriting target geometry. Spend the six GPUs on
architectural attribution first.

## What to retain from previous experiments

Retain:

- NN1 correctness and strict restore guards;
- one doubled target/reference U-Net call;
- both branched processor classes;
- target-face Q → reference-face K/V as a real active route;
- PhotoMaker establishment at step 10 and BA at step 15;
- active but frozen branched CA from NN1d;
- fixed 96-image same-seed visual comparison;
- hard difficult cases: skiing, chef, night-ride, laughing, kickboxing, and
  crying with hair/hands.

Borrow concepts without copying the post-N3a architecture:

- N17's CAMIX ablation showed that access to current target-face K/V repairs
  face/head geometry;
- N24 showed why target and reference attention should use separate softmaxes
  with per-head/layer arbitration rather than concatenated K/V competition;
- N27's compact hard ROI idea removes outside-box zero-token sinks and absolute
  crop location;
- N29/N32 showed that target-coordinate identity residuals are much safer, but
  those runs removed `BranchedAttnProcessor`, so NN2 should reuse only their
  safety ideas inside the original spatial BA mechanism.

Do not carry:

- trainable branched CA;
- NN1c's prompt mask before spatial safety is fixed;
- the current NN1e/f decoded ID objective;
- a full-grid reference face as the sole absolute face authority at all layers;
- post-N34 restricted compact attn2-only residuals.

## Proposed NN2 runs

Every proposal keeps:

- the doubled `[target, reference]` U-Net;
- all 70 `BranchedAttnProcessor` objects installed;
- all 70 `BranchedCrossAttnProcessor` objects active;
- target-face Q attending reference-face K/V in the active reference lane;
- the split target-generation/reference-face prompt route;
- direct target-half epsilon output;
- branched CA cloned weights frozen;
- NN1 correctness guards.

### NN2a — packed ROI spatial BA

Isolate reference token selection while retaining N3a's absolute face merge.

At each self-attention resolution:

1. take the validated hard reference bbox;
2. ROI-normalize the reference hidden rectangle to a fixed per-layer grid;
3. pack only real ROI tokens;
4. use a real padding attention mask;
5. let target-face Q attend these packed reference K/V tokens.

There must be no zero-masked outside-grid tokens left in the softmax.

Purpose: determine how much collapse comes from zero-token sinks and absolute
reference crop location. Risk: an absolute packed-ROI face can still be
geometrically incompatible with the target.

Priority: medium, diagnostic.

### NN2b — separate target/reference attention with per-head arbitration

Isolate source arbitration while retaining the legacy full reference grid.

For the target face, compute:

```text
target_face = Attn(Qtarget, Ktarget, Vtarget)
ref_face    = Attn(Qtarget, Kreference, Vreference)
face_out    = (1 - gate[layer, head]) * target_face
              + gate[layer, head] * ref_face
```

The two sources must use separate attention softmaxes. Do not concatenate their
K/V sets. Use a bounded per-head/layer gate so target geometry cannot disappear
immediately.

Purpose: test the N17 CAMIX/N24 geometry lesson inside the original processor.

Priority: high, clean arbitration attribution.

### NN2c — packed ROI plus dual attention

Combine NN2a's clean reference evidence with NN2b's explicit target geometry
lane:

```text
target_face = Attn(Qtarget, Ktarget, Vtarget)
ref_face    = Attn(Qtarget, Kpacked_roi, Vpacked_roi)
face_out    = per_head_layer_blend(target_face, ref_face)
```

This is the best balanced proposal. It attacks both major failures:

- packed ROI removes reference location/outside-token leakage;
- target attention preserves pose, expression, occluders, and face/head
  geometry.

Priority: highest.

### NN2d — up-block identity specialization

Keep all 70 branched SA processors installed, but allow reference K/V only in
the 36 `up_blocks.*` self-attention sites. The 24 down-block and 10 mid-block
sites use target K/V for the target face. All 70 split CA processors remain
active/frozen.

Purpose: test whether reference BA is most damaging while coarse face geometry
is formed, while still allowing reference identity detail during reconstruction.

This is architectural layer specialization, not a top-k hyperparameter sweep.

Priority: high.

### NN2e — inner-face identity core with protected boundary ring

Split the hard target bbox into two target-coordinate regions:

- an inner elliptical/core region where target Q may use reference K/V;
- a surrounding ring where target K/V retains authority.

The protected ring should include face edges, jaw/neck transition, hairline,
and bbox-adjacent props. Both regions remain inside the validated hard bbox;
invalid bboxes still fail closed.

Purpose: directly reduce pasted-face, neck, hair, and hat boundary failures.
Risk: goggles, eyes, or hands inside the core can still collide with reference
features.

Priority: medium, boundary-specific.

### NN2f — confidence-gated packed-ROI BA residual

This is the brave safety design. Keep target self-attention as the absolute
face anchor and turn reference BA into a zero-initialized residual:

```text
target_face = Attn(Qtarget, Ktarget, Vtarget)
ref_face, P = Attn(Qtarget, Kpacked_roi, Vpacked_roi)
confidence  = 1 - entropy(P) / log(number_of_real_roi_tokens)
face_out    = target_face
              + gate[layer, head] * confidence[query, head]
                * Delta(ref_face - target_face)
```

Low-confidence/diffuse correspondence falls back to target geometry. The
reference lane remains genuine BA: target Q still attends reference K/V in
every active processor. `Delta` must be zero-initialized so step zero exactly
matches the target-attention face lane, while the gate itself must not also be
initialized to a gradient-blocking zero.

Purpose: make reference authority conditional on correspondence quality rather
than uniformly absolute.

Priority: highest upside, highest implementation risk.

## Recommended six-GPU allocation

Use all six GPUs for architecture breadth, not larger batches:

| Machine | GPU | Run | Main question |
|---|---:|---|---|
| 2-GPU | 0 | NN2a | Are real packed ROI tokens sufficient? |
| 2-GPU | 1 | NN2b | Is explicit target/ref arbitration sufficient? |
| 4-GPU | 0 | NN2c | Does the combined repair give the best balance? |
| 4-GPU | 1 | NN2d | Should coarse geometry stay target-owned? |
| 4-GPU | 2 | NN2e | Does boundary ownership fix seams/neck/hair? |
| 4-GPU | 3 | NN2f | Can confidence-gated residual BA stay clean and strong? |

Common first-screen protocol:

| Setting | Value |
|---|---|
| GPUs per run | 1 |
| physical/effective batch | 2 / 2 |
| maximum optimizer steps | 6,000 |
| validation | full 96 at step 0, 2k, 4k, 6k |
| base optimizer/objective | NN1d masked-alternating, frozen CA |
| current decoded ID loss | off |
| inference schedule | text 0–9, PM 10–14, BA 15–49 |
| checkpoint selection | visuals first, metrics secondary |

The first goal is not a final identity score. It is to find a topology that
changes identity evidence while preserving valid face anatomy and target
geometry.

### Stop gates

Stop a run at 2k or 4k if the same failure repeats systematically on:

- Jensen/Jisoo skiing;
- Jisoo/Jennie night-ride;
- Jisoo chef/laughing;
- Jensen/Marion kickboxing;
- Marion crying.

A promotable run should:

- retain 96/96 face detection;
- show no white/orange hard face plates;
- avoid repeated eyes/mouths and blank faces;
- preserve goggles, hats, hair, hands, jaw, and neck alignment;
- visibly respond to reference identity on simple and hard poses;
- stay stable or improve from 2k to 6k.

After screening, promote only the cleanest NN2c/NN2f-like result to a fresh
two-GPU run with global batch 4 and a longer 10k–20k budget. Do not resume from
an NN1 checkpoint because its learned projections already encode the unsafe
spatial solution.

## Checkpoints worth downloading

Checkpoint download is not required for the conclusion above. If additional
forensics are desired, the minimal useful set is:

1. **NN1a epoch 5** — rerun with current validation-constructor parity and
   confirm exact final images/metrics;
2. **NN1d epoch 3 and/or epoch 5** — inspect attention maps, run target-K/V
   runtime ablations, or initialize diagnostic-only comparisons;
3. **NN1e epoch 5** — only if investigating the current ID-loss shortcut with
   detector/landmark diagnostics.

Do not download all six merely to decide whether to continue training.

## Implementation boundary

No NN2 model code, Hydra config, or shell launcher was created in this task.
The interactive HTML contains proposal records so NN1d can be compared
visually with NN2a–f, and changed K/V, arbitration, layer, mask, and objective
elements are highlighted. Implementation should wait for approval.
