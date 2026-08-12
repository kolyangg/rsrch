---
title: "Shared hard prompts, face-ownership bugs, and contract-correct Eddie validation across E13, BC_E13, CL10 and CL11"
subtitle: "Final evidence: exact historical replay, corrected-reference intervention, fixed-mask visual audit, and implementation-ready next experiments"
date: "9 August 2026"
status: "FINAL: contract-v2 Serv replay completed and hash-verified 9 August 2026"
---

> **Validation correction completed.** The earlier standalone sidecar was
> invalid because it used the wrong processor-copy mode, omitted the shadow
> PhotoMaker-default restore, and batched one image instead of the training
> batch of 12. It has been replaced by a guarded Serv replay using RealVis,
> strict `legacy_full_copy`, shadow restore, the full-96 dataset context and
> batch 12. Before any corrected arm ran, E13, BC_E13 and CL11 each reproduced
> all 12 historical Eddie images RGB pixel-for-pixel: 36/36 exact, zero
> contract mismatches. All corrected-Eddie values and figures in this edition
> come only from that contract-v2 run. **[measured] [code]**

> **Generation-box ownership erratum, 9 August 2026.** The original Chef/Lex
> ownership audit below compared detections with the dataset's manual box
> `[590,413,694,543]`. In-training BA actually used the cached automatic box
> `[223,380,447,668]` (protocol SHA-256 `4db6344d...`). With the actual box,
> the historical ID winner has IoU `0.8987` and is the mask-owned face; with
> the unused manual box its score is `0.1391`. Chef/Lex is therefore not a
> proven live-generation ownership failure. It exposed a real analysis/metric
> hand-off hazard: scoring must receive the exact resolved BA box. The code now
> enforces that invariant and retains a mask-owned metric as a prospective
> guard. Figures and counts using the manual box are withdrawn as evidence
> about the historical BA mask. **[measured] [code]**

# Executive conclusion

The recurring validation failures are real, highly structured, and mostly
shared across datasets. They are **not** explained by one bad experiment. The
same identity/prompt cells remain hard through 41 checkpoints spanning
large_dataset (E13), BigCelebs (BC_E13), and cosmic_large (CL10/CL11):
Marion/Skiing is in the bottom ten at **41/41** checkpoints, Jisoo/Skiing at
**38/41**, Lex/Dancing at **35/41**, Marion/Jumping and Marion/Laughing at
**34/41**, and Lex/Jumping at **31/41**. Rank correlations between the four
models' 84 valid non-Eddie cells are `0.746-0.800` at the controlled 18k gate.
That persistence makes prompt/reference/architecture interaction the primary
cause; dataset choice changes the severity and which identities benefit, but
does not remove the failure set. **[measured]**

Two validation defects must be fixed before treating the headline identity
score as a reliable leaderboard:

1. **Eddie is scored and PhotoMaker-conditioned on the wrong person.** His
   reference contains Eddie in the foreground and a small background face.
   The embedding builder and conditioning path take detector result `faces[0]`,
   which is the background face. The stored embedding has cosine `-0.0078` to
   the intended/largest foreground face. All historical Eddie scores are
   invalid; they also hide that the generated faces match the intended Eddie
   poorly (`0.068-0.077` mean corrected diagnostic at 18k). **[measured] [code]**
2. **`IDSimBest` has no ownership constraint, and the scorer must receive the
   resolved BA box.** The metric takes the maximum ArcFace cosine over every
   detected face. The original Chef/Lex claim was a false positive caused by
   using the dataset's manual box after generation had overridden it with the
   cached automatic box. The actual E13 Chef/Lex winner is mask-owned (IoU
   `0.8987`). A versioned mask-owned metric remains necessary to detect future
   drift, but Chef/Lex is not evidence that historical BA put Lex on an
   off-mask body. **[measured] [code]**

The contract-correct rerun confirms that the Eddie selector bug is causal and
that the earlier apparent body/layout drift was an evaluator artifact. Using
the final E13 24k, BC_E13 24k and CL11 20k checkpoints, replacing the
historical bystander embedding with intended foreground Eddie raises mean
intended-Eddie similarity by `+0.360`, `+0.291` and `+0.289`; every one of the
36 prompt pairs improves. Median fixed-mask IoU stays close to the exact replay:
`0.896 -> 0.891`, `0.904 -> 0.875`, and `0.885 -> 0.880`. No corrected image
falls below `0.30` IoU; the minima are `0.733`, `0.500`, and `0.684`.
Kickboxing and Jumping retain their original body composition while their face
identity improves. Residual failures are visual face quality - especially
Skiing goggles, Laughing mouths, small Dancing faces, and extra Chef people -
not catastrophic ownership relocation. **[measured] [visual]**

After excluding Eddie, CL11 has the strongest 18k identity mean (`0.46094`),
CL10 is second (`0.45272`), E13 third (`0.42764`), and BC_E13 fourth
(`0.42331`). That is not a global CL11 promotion: CL11 has a worse alignment
tail than E13/CL10 and a major Marion regression (`0.28785` at 18k; `0.26201`
at 20k), while CL10 gives the best combined identity/alignment result at the
controlled gate. E13 remains the base because its final checkpoint is the most
stable geometry anchor and the proposed changes can be evaluated as narrow,
reversible extensions. **[measured] [inference]**

The recurring visual failures fall into three model-facing families:

- **Small, dynamic target faces** in Jumping and Dancing. Their mask is seated
  correctly and the face-size ratio is usually near `1.0`, but the face has only
  about 13-20 latent cells across its short side. ArcFace can still recognize a
  coarse identity-correlated texture even when eyes, teeth, or contours are
  visibly malformed. **[measured] [inference]**
- **Occlusion/expression conflicts** in Skiing, Crying, Laughing and sometimes
  Kickboxing. Goggles, hair, hands, tears, teeth, and gloves collide with a
  hard reference-only face replacement. These are not “unwinnable prompts,”
  but the present routing gives the target face insufficient native scene
  evidence to preserve the occluder or expression cleanly. **[visual]
  [code] [hypothesis]**
- **Reference-specific difficulty**, especially Marion. Multi-reference CL11
  helps Jisoo/Keanu but hurts Marion; this suggests that adding more reference
  evidence can amplify pose/hair incompatibility instead of averaging it away.
  **[measured] [hypothesis]**

The highest-priority work is therefore: (1) productionize subject selection and
add a mask-owned identity metric in a versioned validation namespace, (2)
evaluate a bounded mixture of E13's trained reference message with the native
target-face message for occlusion/expression quality, and (3) apply
target-scale-matched reference preparation to small dynamic faces. The
PhotoMaker-onset sweep is demoted: the contract-correct intervention no longer
shows the composition failure it was designed to explain. **[measured]
[inference]**

> **Final evidence cutoff.** This edition uses the complete controlled 18k
> panels, the historical final saved panels for E13 24k, BC_E13 24k and CL11
> 20k, and the completed contract-v2 12-prompt Eddie intervention at those
> checkpoints. Each historical arm first passed exact RGB replay. The
> intervention then preserved reference pixels/bbox, prompts, seeds, fixed
> generation masks, RealVis, strict processor/shadow behavior, scheduler, 50
> steps, CFG 5 and checkpoint; it changed only the ArcFace vector fused into
> PhotoMaker's global identity tokens. It is an explicit 12-image diagnostic
> exception, not a replacement for fixed-96 validation. **[measured]**

# Runs and comparison contract

**E13** - large_dataset; controlled 18k and final 24k.  
Exact run: `E13_large_ds_joint_shadow_sa128_24k_full96_r4`  
Immutable Comet key: `1cc0a02371094b24a6a02a4cc649f10c`

**BC_E13** - BigCelebs; controlled 18k and final 24k.  
Exact run: `BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1`  
Immutable Comet key: `c138db7c41ae435c8a7560f40cf5f58d`

**CL10** - cosmic_large; controlled 18k.  
Exact run: `CL10_cosmic_joint_shadow_sa128_refscale_fullbody_`  
`24k_full96_r2`  
Immutable Comet key: `eba0187806ec476996f5ea4af356361e`

**CL11** - cosmic_large; controlled 18k and requested saved 20k checkpoint.  
Exact run: `CL11_cosmic_joint_shadow_sa128_refscale_multiref_`  
`24k_full96_r1`  
Immutable Comet key: `32f4ba2a3b3a493f96a3a2345147e84c`

The causal comparison is the fixed 96-image `manual_val` panel at the common
18,000-step gate: same prompts, identities, references, seeds, target masks,
scheduler, inference steps, CFG, and historical metric definitions. Endpoint
24k/20k values are descriptive because their steps differ. All arms preserve
the eligible E13 BA contract `pose_adapt_ratio=0` and
`ca_mixing_for_face=false`; the audited checkpoints retain `2,240` trainable
tensors / `219,217,920` trainable parameters. **[code] [measured]**

The visual overlays use the **rectangular support of the fixed validation face
mask in red** and InsightFace detections in cyan. Reported alignment is the IoU
between the selected detected-face box and the fixed mask box; center offset is
normalized by mask dimensions; size ratio is detected-face short side divided
by mask short side. Red is therefore “where this body expects the conditioned
face,” not a detector prediction. This distinction exposes ownership errors
that ID similarity alone cannot see.

The recommended **primary identity metric** for all new decisions is
`IDSimMaskMatched`: ArcFace cosine for the generated detection with greatest
overlap to the fixed target mask, reported together with that overlap and face
count. Historical `IDSimBest` remains necessary for audit continuity but is
misleading whenever the correct identity appears on another body. Until P0 is
implemented in the main validation code, the sidecar uses the equivalent
per-image mask-selected calculation against the intended Eddie embedding.

# 1. Validation integrity: two independent ownership bugs

## 1.1 Eddie: the historical reference embedding is the background face

![](assets/problematic_validation_20260809/eddie_reference_metric_error.png)

*Figure 1. Eddie reference with the historical stored-face mask in red and the
intended foreground-face mask in green.*

The 400×300 Eddie reference has two detections. The historical builder takes
the first result, `[336, 136, 400, 257]`, a small blurred face clipped by the
right edge. The intended/largest face is `[104, 0, 303, 291]`. The stored vector
has cosine `1.0000` to the background detection and `-0.0078` to Eddie. The
same first-result assumption appears in:

- `tools/datasets/create_manual_val_id_embeds.py` for the validation target;
- `src/pipelines/br_pipeline_helpers.py` for inference conditioning; and
- both active training extraction paths in
  `src/model/photomaker_branched/lora2_helpers.py`.

The BA reference crop itself uses the intended foreground box. Eddie therefore
receives conflicting people: PhotoMaker identity tokens for the bystander and
BA reference pixels for Eddie, then is graded against the bystander. This is a
validation and conditioning defect, not evidence that Eddie is intrinsically
unrepresentable. **[code] [measured]**

Re-scoring the unchanged 18k generated images against the intended foreground
embedding makes the apparent result worse, as expected:

| Run | Historical Eddie mean (invalid) | Intended Eddie diagnostic | Delta |
|---|---:|---:|---:|
| E13 | `0.1715` | `0.0768` | `-0.0947` |
| BC_E13 | `0.1448` | `0.0680` | `-0.0767` |
| CL10 | `0.1599` | `0.0753` | `-0.0846` |
| CL11 | `0.1714` | `0.0775` | `-0.0939` |

The diagnostic fixes only the scoring target, not the historical conditioning,
so it is **not** a corrected model comparison. It establishes that the old
images match neither intended Eddie nor the bystander well. The completed
final-checkpoint intervention below fixes conditioning and scoring together
while holding the prompt, seed, mask and denoising contract fixed. **[measured]**

Consequence: every historical aggregate in this report is shown both as logged
and with Eddie's 12 rows excluded. Old metrics are preserved for auditability;
they are not overwritten.

## 1.2 Chef/Lex erratum: manual and active automatic boxes select different bodies

![](assets/problematic_validation_20260809/idsim_mask_leakage_chef_lex.png)

*Figure 2 (withdrawn as historical BA-ownership evidence). Red is the dataset
manual box, not the cached automatic box actually used by BA; cyan and magenta
therefore visualize the analysis hand-off error.*

At the 18k gate, 43/384 run-images contain more than one detected face. The
original audit scored them against `pm96_bboxes_new.json`, while automatic BA
generation used `pm96_bboxes_new_auto.json`. They differ materially for the
four Chef/Lex cells. In the original, now-withdrawn diagnostic:

- the fixed red mask follows the background chef;
- the identity-looking Lex face appears on the foreground chef;
- the magenta ID winner has mask IoU `0.000`;
- the cyan face within the fixed mask has IoU `0.723-0.814`, but much lower
  identity (`0.128-0.212` instead of reported `0.313-0.351`).

This is not evidence of a historical generation/body-ownership failure. It is
evidence that metric ownership must be coupled to the resolved generation box,
not a stale dataset field. The legacy maximum-over-faces score remains for
continuity; the new mask-owned score selects against the exact box passed to
BA and logs overlap, face count and ambiguity. **[measured] [code]**

# 2. Controlled 18k comparison

![](assets/problematic_validation_20260809/overview_metrics_18k.png)

*Figure 3. Controlled 18k run metrics and alignment distributions.*

| Run | Historical mean | Excluding invalid Eddie | Faces | Multi-face | TOPIQ-Face |
|---|---:|---:|---:|---:|---:|
| E13 | `0.39562` | `0.42764` | 96/96 | 11 | `0.7184` |
| BC_E13 | `0.38849` | `0.42331` | 96/96 | 8 | `0.6848` |
| CL10 | `0.41611` | `0.45272` | 96/96 | 11 | not fetched |
| CL11 | **`0.42475`** | **`0.46094`** | 96/96 | 13 | not fetched |

| Run | median IoU | p10 IoU | IoU <0.70 | median size ratio |
|---|---:|---:|---:|---:|
| E13 | `0.8498` | `0.7599` | 2 | `1.0249` |
| BC_E13 | `0.8391` | `0.7161` | 7 | `1.0473` |
| CL10 | **`0.8527`** | `0.7558` | 3 | `1.0184` |
| CL11 | `0.8431` | `0.7218` | 7 | `1.0204` |

All four detect a face in every image and all have zero faces undersized below
`0.80×` the fixed mask. Face detection and scale matching are therefore
necessary but weak quality gates: malformed goggles, eyes, teeth and contours
still pass. CL11 leads identity by `+0.00823` over CL10 excluding Eddie, but its
tail has seven boxes below IoU `0.70` versus three for CL10 and two for E13.
CL10 is the best controlled compromise. **[measured]**

The 18k identity means reveal interaction rather than dominance:

| Identity | E13 | BC_E13 | CL10 | CL11 | Reading |
|---|---:|---:|---:|---:|---|
| Elon | `0.497` | `0.459` | `0.507` | **`0.510`** | cosmic helps modestly |
| Jennie | `0.421` | `0.410` | `0.458` | **`0.494`** | multi-reference helps |
| Jensen | `0.463` | `0.527` | **`0.543`** | `0.532` | both alternative datasets help |
| Jisoo | `0.458` | `0.423` | `0.415` | **`0.506`** | CL11-specific gain |
| Keanu | `0.447` | `0.453` | `0.474` | **`0.518`** | CL11-specific gain |
| Lex | `0.382` | `0.361` | **`0.410`** | `0.379` | CL10 helps; Chef metric caveat |
| Marion | `0.326` | `0.329` | **`0.362`** | **`0.288`** | CL11 materially regresses |

CL11's multi-reference training is most useful for Jisoo and Keanu, but it is
not uniformly safer: Marion is `-0.074` below CL10. This is why “use more
references” should not be merged into E13 globally without a subject-aware
gate or a controlled fallback. **[measured] [inference]**

Prompt means show a stable action/occlusion split. At 18k, Jumping is only
`0.296-0.366`, Dancing `0.341-0.374`, and Skiing `0.302-0.349`. Reading is
`0.486-0.528`, Rushing `0.481-0.521`, and Drumming reaches `0.519` in CL11.
Skiing is the special case where BigCelebs later improves strongly at 24k,
showing that the dataset can change a hard prompt's severity, but not remove
the general hard family.

## Endpoint view (descriptive, not step-matched)

![](assets/problematic_validation_20260809/endpoint_metrics.png)

*Figure 4. Historical endpoint metrics for E13 24k, BC_E13 24k and the requested
CL11 20k checkpoint.*

| Saved checkpoint | Historical mean | Excluding Eddie | median / p10 IoU | IoU <0.70 | Marion | Jumping | Skiing |
|---|---:|---:|---:|---:|---:|---:|---:|
| E13 24k | `0.39980` | `0.43203` | `0.8495 / 0.7664` | 3 | `0.341` | `0.336` | `0.285` |
| BC_E13 24k | `0.38943` | `0.42278` | `0.8384 / 0.7287` | 7 | `0.333` | `0.294` | **`0.369`** |
| CL11 20k | `0.41656` | **`0.45141`** | `0.8432 / 0.7179` | 8 | **`0.262`** | `0.305` | `0.346` |

The endpoints reinforce the controlled result: CL11 retains an identity lead
but has the weakest Marion result and the largest alignment tail; BigCelebs
specializes toward Skiing but loses on Jumping and the aggregate; E13 is the
best geometry anchor. **[measured]**

# 3. Corrected Eddie final-checkpoint intervention

The final-checkpoint rerun is a paired intervention over Eddie's 12 prompts.
For each model, the historical image and corrected image are both scored
against the same intended foreground Eddie embedding. This is the valid paired
comparison; the historical logged Eddie number remains invalid because it uses
the background bystander's embedding. The unchanged arm first reproduced every
historical PNG exactly (12/12 for each model), proving that base, processors,
shadow restore, batch semantics and inference inputs match training validation.

![](assets/problematic_validation_20260809/corrected_eddie_final_checkpoint_metrics.png)

*Figure 5. Intended-Eddie identity deltas by prompt and corrected fixed-mask
geometry. This is a paired diagnostic, not uncertainty over model training.*

| Final checkpoint | Historical images, correct target | Corrected images, correct target | Paired delta | Wins | Corrected multi-face | Corrected median mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| E13 24k | `0.0653` | **`0.4254`** | **`+0.3600`** | **12/12** | 2 | **`0.8911`** |
| BC_E13 24k | `0.0626` | `0.3540` | `+0.2913` | **12/12** | **1** | `0.8747` |
| CL11 20k | `0.0741` | `0.3633` | `+0.2893` | **12/12** | **1** | `0.8798` |

All three generate a detected face in 12/12 corrected images. That result
proves the selector error suppressed the intended identity: E13 is strongest
on corrected Eddie and all models win every paired prompt. It does **not** promote E13
unconditionally. The endpoint steps differ, and the corrected 12-image set is
one identity rather than the fixed-96 panel. If the 84 unchanged non-Eddie rows
are held fixed and the 12 Eddie rows are replaced, the descriptive
counterfactual full-96 means are E13 `0.4312`, BC_E13 `0.4142`, and CL11
`0.4404`. They must not be joined to historical Comet curves as if the metric
namespace were unchanged. **[measured]**

![](assets/problematic_validation_20260809/corrected_eddie_before_after_masks.png)

*Figure 6. Historical versus corrected conditioning for representative Eddie
prompts. Red is the immutable generation mask; cyan is the selected generated
face detection. Labels report intended-Eddie ID, mask IoU and detected-face
count.*

The clean prompts show the benefit clearly. Corrected Drumming reaches
`0.548/0.365/0.459` for E13/BC_E13/CL11; Chef reaches
`0.517/0.380/0.504`, and Night ride reaches `0.457/0.381/0.385`. In Chef, the
intended Eddie face is on the mask-owned foreground person rather than being a
metric win on another body. E13 is strongest on Reading (`0.467`), Drumming
and Chef. The dataset arms therefore change which clean
context is easiest, but all benefit from the correct subject. **[measured]
[visual]**

The hard prompts remain hard for common, model-facing reasons:

- **Skiing:** intended-Eddie ID improves to `0.260-0.443` with mask IoU
  `0.757-0.804`, but goggles still distort or fragment the eye region. This is
  an occlusion/anatomy failure, not a placement failure; CL11 is strongest.
- **Laughing and Dancing:** Laughing remains only `0.136-0.225`; mouth and eye
  structure are weak. Small Dancing faces are aligned but CL11 ID is only
  `0.146`, with BC_E13's weakest corrected mask IoU (`0.500`).
- **Chef:** identity is strong, but all three retain two detected faces. A
  binary detector and even mask-owned ID do not penalize unnecessary
  background people.

![](assets/problematic_validation_20260809/corrected_eddie_alignment_failures.png)

*Figure 7. Contract-v2 Kickboxing and Jumping comparison with the same
immutable masks. Unlike the invalid sidecar, the corrected images retain the
historical body layout and mask ownership.*

Kickboxing and Jumping are the clearest confirmation that the evaluator fix
matters. E13 Kickboxing improves `0.115 -> 0.485` while mask IoU improves
`0.891 -> 0.922`; BC_E13 and CL11 remain aligned at `0.899` and `0.890`.
Jumping improves to `0.341/0.329/0.227`, with corrected IoUs
`0.875/0.741/0.744`. The base scene and body pose remain visually stable; the
meaningful change is concentrated in facial appearance. **[visual] [measured]**

The corrected embedding is still fused into global PhotoMaker prompt tokens,
so this is not a face-local BA-only edit and the images are not expected to be
pixel-identical outside the mask. Across models, mean absolute RGB change is
`22.0-25.0` inside the fixed face box versus `11.5-12.3` outside, but the
detected face remains owned by the same mask. The earlier gross layout changes
came from the invalid processor/shadow/batch contract, not from the Eddie
selector correction itself. **[code] [measured] [visual]**

The exact checkpoint SHA-256 values are:

- E13 24k: `4a9d95a3f957609fcf4eb77771f263dec8e71189dc72aae347233091de4249ab`;
- BC_E13 24k: `99b305bad425dd07073a4a54e0a978dea0d4a02456c8129eb1b12afbbf5a459e`;
- CL11 20k: `e65972c8c14b5031f879e1ee8b1e11a707823e0cfccdb80553219fc8069dbb83`.

E13 and BC_E13 used generation-bbox protocol SHA-256
`4db6344d0deb0af0ee7a25d839b774c9a4a0c5b8f6ff4cc00aaa9c0d6d85c099`;
CL11 used its historical runtime's exact cached protocol
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.
Exact pixel replay proves each is the bbox state used for its source images.
Per-image metrics, images and manifests are preserved in the
`final_checkpoint_sidecar_contract_v2/` subdirectory of the report asset root.

# 4. The recurring hard cells and what is common

![](assets/problematic_validation_20260809/persistent_hard_cells.png)

*Figure 8. Persistence of the worst identity-prompt cells across 41 validation
checkpoints.*

| Identity / prompt | Bottom-10 rate | Mean ID_SIM over 41 checkpoints | Primary visual pressure |
|---|---:|---:|---|
| Marion / Skiing | **100.0%** | `0.146` | goggles + reference view/hair |
| Jisoo / Skiing | **92.7%** | `0.174` | goggles intersect eyes; detector instability |
| Lex / Dancing | **85.4%** | `0.205` | small dynamic face, low feature bandwidth |
| Marion / Jumping | **82.9%** | `0.210` | small face, motion, hair crossing boundary |
| Marion / Laughing | **82.9%** | `0.223` | open mouth/expression + weak reference |
| Lex / Jumping | **75.6%** | `0.245` | small dynamic face |
| Keanu / Jumping | **65.9%** | `0.255` | small face, beard/mouth instability |

![](assets/problematic_validation_20260809/hard_cases_with_masks_18k.png)

*Figure 9. Hard cases at 18k; red is the fixed mask and cyan the largest
detection.*

The overlays separate “bad score” from “bad face”:

- **Jisoo/Skiing:** goggles cut through the eye region. CL11's detected face has
  IoU `0.624`, yet the detector still returns one face. CL10 and CL11 show the
  most visibly asymmetrical or fused eye/goggle geometry. The four-run mean
  identity is only `0.194`. **[visual] [measured]**
- **Marion/Skiing:** anatomy can look plausible while identity collapses because
  goggles conceal the ArcFace-sensitive eye region; run scores are
  `0.096-0.237`. This is a prompt/metric conflict compounded by the Marion
  reference, not a face-size error. **[visual] [measured]**
- **Marion/Crying:** hands, strands of hair and tear/expression cues compete at
  the mask boundary. CL10 scores `0.452`, CL11 only `0.187`, despite similar
  placement. This is evidence that multi-reference strength does not guarantee
  robust composition. **[visual] [measured]**
- **Marion/Jumping, Lex/Dancing, Keanu/Jumping:** the masks and detections align
  reasonably, but the face is a small portion of the 1024² canvas. Enlarged
  crops expose smeared eyes, teeth and identity-shape approximations that the
  full image hides. **[visual]**

The common architectural pressure is visible in the E13 routing. In
`hard_replace_v1`, background target queries use native target K/V, while face
target queries attend explicit reference-only K/V and the result is merged
through the hard face mask. With `pose_adapt_ratio=0`, native target-face K/V is
not available inside that reference message. The route is excellent at forcing
identity evidence, but it is asked to reconstruct goggles, hands, tears and an
extreme expression from reference content that often does not contain them.
At small face scales it must also encode identity and anatomy in very few latent
positions. This is the most coherent mechanism linking all three failure
families. **[code] [hypothesis]**

# 5. Good ID_SIM can still mean a visibly bad face

![](assets/problematic_validation_20260809/score_traps_with_masks_18k.png)

*Figure 10. High-scoring visual traps at 18k with fixed masks and detected-face
zooms.*

The clearest score traps are:

- Crying/Keanu in CL10 (`0.574`) and CL11 (`0.594`): strong identity signal,
  but the hand/temple/eye transition is visibly malformed.
- Kickboxing/Elon (`~0.448-0.564`): recognizable, yet caricatured with a
  body/glove collision.
- Laughing/Jisoo in CL11 (`0.613`): identity is high while mouth/teeth and the
  expression are implausible.
- Dancing/Jensen in CL11 (`0.572`, IoU `0.615`): good ArcFace score on a
  poorly seated, low-resolution face.
- Skiing/Jennie (up to `0.556` in the audited panels): floating or malformed
  goggles remain even when identity is high.

Face detection is 96/96 for every arm at 18k, so a binary detector cannot catch
these. TOPIQ-Face helps identify blur/low detail but can also reward a crisp,
malformed face. A promotion gate must combine: mask-owned identity, alignment,
face quality, artifact-oriented visual review, and prompt adherence. **[visual]
[measured] [inference]**

# 6. Dataset differences and curriculum

BC_E13 demonstrates why more identities are not automatically better target
training. BigCelebs has 349,348 images / 68,648 manifest identities versus
47,500 / 2,561 for Large Dataset, but a 48k-target budget consumes only 13.74%
of BigCelebs and about one pass of Large Dataset. About 48.6% of BigCelebs
identity groups have only 2-3 images. More importantly, 83.97% of its captions
are portrait/close-up versus 0.324% for Large Dataset, and median face side is
410 px versus 255 px. Dynamic full-body, hands and multi-person context are
rarer. **[report] [measured]**

That distribution explains the observed specialization: BC_E13 can improve
large, centered or goggle-covered faces such as Skiing and Jensen, but is less
reliable for Jumping, Crying and scene-level alignment. The loader also has an
estimated wrong-direction caption rate of 6.36% for BigCelebs versus 2.52% for
Large Dataset because horizontal flips do not rewrite left/right captions.
These facts make BigCelebs a poor replacement for E13's target corpus.
**[report] [inference]**

Recommended ordering:

1. **Train E13/Large Dataset first.** It supplies scene, pose and compositional
   grounding and remains the base.
2. If continuing on cosmic_large, use a short, curated full-body/reference-scale
   continuation with approximately **50% Large replay**, so the model gains
   cosmic's identity/reference diversity without forgetting the target domain.
3. Use curated BigCelebs images preferentially as **canonical references**, or
   in a Large:BigCelebs target schedule near **2:1**, rather than BigCelebs-only
   pretraining. There is no evidence supporting BigCelebs→Large as the first
   curriculum. **[hypothesis]**

The already launched BC_E13 dataset-policy arms (repeat-depth balancing,
scene-target/canonical-reference separation, and 2:1 Large anchoring) directly
test this conclusion. Their results should be read before authorizing another
dataset-only training run. **[report]**

# 7. What is worth recovering from the previous architecture work

Earlier project reports rejected broad hard reference replacement because it
transferred reference pose, layout, hair, glasses and hands as if they were
identity. They also found that decoded identity losses could raise identity
metrics while worsening face anatomy, and that weak zero-initialized residual
routes learned generic corrections rather than identity. Those negative results
still rule out simply increasing BA scale, adding an unconstrained identity
loss, or reviving a residual path that must relearn identity from scratch.
**[report]**

One older result is newly relevant: a step-zero **separate native/reference
attention mixture** preserved geometry much better than hard replacement.
Reference mix `0.25` retained 96/96 faces with face MAE `0.05787`, landmark
shift `0.00732`, and bbox IoU `0.96643`; mix `0.35` gave `0.07723`, `0.01134`,
and `0.95526`. Those were geometry diagnostics, not a trained identity win.
**[report]**

The proposal below differs materially from the failed residual/PPR and anchored
v3 directions: it starts from the **final trained E13 reference route**, keeps
its reference-only K/V explicit, computes the native target-face message as a
separate softmax, and mixes the two bounded outputs only inside the fixed face
region. At `alpha=1` it must reproduce E13 exactly. This tests whether a small
amount of native target evidence can preserve occluders and expression without
asking a new branch to relearn identity. **[hypothesis]**

Relevant prior writeups:

- [`2026-08-09_problem_prompts_identities_root_cause_and_e25_e27.md`](2026-08-09_problem_prompts_identities_root_cause_and_e25_e27.md)
- [`2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md`](2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md)
- [`2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md`](2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [`2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md`](2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md)
- previous-worktree `Jul_new_exp/22Jul_debug/2026-07-23_expanded_step0_architecture_study_report.md`
- previous-worktree `Jul_new_exp/22Jul_debug/2026-07-23_recent_run_idea_audit.md`

# 8. Priority decisions

| Priority | Change | Why it is first | GPU cost | Promotion signal |
|---|---|---|---:|---|
| **P0** | Subject-aware reference selection + resolved-mask ID metric | Removes Eddie's proven false target and prevents metric/BA box divergence | none / validation only | Eddie foreground selected; exact active boxes scored; non-Eddie legacy unchanged |
| **P1** | E13 dual native/reference face-message sweep | Directly targets goggles, mouth/expression and occlusion while retaining the trained E13 reference route | one inference sweep | hard-set anatomy improves with ≤`0.01` valid ID loss; `alpha=1` exact |
| **P2** | E13 target-scale-matched spatial reference sweep | Small Jumping/Dancing faces remain information-limited; CL9/CL10 show scale calibration is useful | one inference sweep | small-face quality improves without alignment or mask-owned-ID regression; baseline exact |

No global switch to CL11 and no BigCelebs-first curriculum is recommended.
CL11's strongest idea - multiple independent references - should later become a
subject-aware optional input to E13, not the new unconditional base.
The PhotoMaker-onset/isolation sweep is no longer a top-three item: after exact
replay, Kickboxing and Jumping preserve mask ownership and body layout.

# 9. Implementation plan for the top three experiments

## Experiment 1 - P0 subject ownership and metric repair

**Goal.** Make reference selection and generated-face scoring agree with the
declared subject/mask, while keeping historical metrics immutable.

**Config/namespace.** `manual_val_subject_v2` and
`E13_val_subject_v2_full96` (new; legacy namespace remains unchanged).
**Single scientific change:** subject ownership used for reference selection
and generated-face scoring. **Hypothesis:** the old leaderboard contains
Eddie false targets and is vulnerable to future off-mask wins. **Prediction:**
Eddie changes to the foreground person, Chef/Lex remains on the winner selected
by the actual automatic BA box, and all other reference embeddings are
numerically unchanged. **Risk:** a silent metric-series join could make the new
aggregate look like a continuation of the old curve; preserve the legacy score
explicitly during any requested historical replacement.

**Code changes.**

1. Add one shared selector, e.g. `src/face_subject_selector.py`, accepting
   detections plus an optional declared subject box. Rank first by overlap with
   that box; when no box exists use largest confident face; fail closed on an
   ambiguous tie. Return bbox, detection score, selection reason and ambiguity.
2. Replace direct `faces[0]` reads with that selector in:

   - `tools/datasets/create_manual_val_id_embeds.py`;
   - `src/pipelines/br_pipeline_helpers.py`; and
   - both active sites in
     `src/model/photomaker_branched/lora2_helpers.py`.
3. Materialize a **new versioned** manual-val embedding file/namespace. Never
   overwrite the legacy vectors. Store source-image hash, selected bbox,
   detector version and selector policy per identity.
4. Add `IDSimMaskMatched` beside `IDSimBest` in
   `src/metrics/id_sim_metric.py`. It selects the generated detection with
   maximum IoU to the fixed mask, then calculates ArcFace cosine. Log
   `face_count`, `selected_mask_iou`, no-face, and ambiguity. Preserve
   `IDSimBest` and its name unchanged.
5. Add a dataset preflight that rejects a reference when the declared box has
   no confident overlapping detection or multiple candidates are ambiguous.

**Verification gates.**

- Eddie selects the foreground bbox and its embedding matches the saved
  intended vector; the old background embedding remains loadable only under
  the explicit legacy namespace.
- Every other manual-val identity selects the same face and reproduces its
  legacy embedding within numerical tolerance.
- On Chef/Lex, `IDSimBest` remains bitwise/historically identical and
  `IDSimMaskMatched` selects the same foreground face when given the cached
  automatic BA box; a negative check with the unused manual box selects the
  background chef and proves the hand-off is observable.
- A full fixed-96 dry run reports zero unowned, ambiguous, or silently missing
  reference selections.
- Emit a migration note: old and new aggregate curves must never be joined as
  if they were the same metric.

## Experiment 2 - P1 E13 dual native/reference message diagnostic

**Goal.** Preserve target-native occluders, expression and anatomy without
removing explicit reference conditioning.

**Config.** `E13_subject_v2_dual_face_message_eval` (new, defaults-off).
**Single scientific change:** bounded output mixture of two separately
normalized native and reference face messages. **Hypothesis:** native target
evidence repairs goggles, hands, teeth and expression while explicit reference
K/V preserves identity. **Prediction:** an intermediate `alpha` improves blind
hard-set anatomy with at most `0.01` full-panel mask-owned ID loss. **Risk:**
native evidence can dilute identity or learn a shortcut if the diagnostic is
prematurely trained. Inference keeps the checkpoint's `2,240` tensors /
`219,217,920` parameters; any later continuation must document its new
trainable contract separately.

**Implementation.**

1. Add a defaults-off architecture/evaluation toggle such as
   `hard_replace_v1_dual_eval` in the active hard-SA processor. Do not alter the
   `hard_replace_v1` default.
2. For target queries inside the face mask, compute two independent attention
   messages with separate softmax normalization:

   `C_native = Attn(Q_target_face, K_target_face, V_target_face)`

   `C_ref = Attn(Q_target_face, K_ref_face, V_ref_face)`

   and merge `C_face = (1 - alpha) C_native + alpha C_ref`. Apply the existing
   output projection/residual and hard spatial merge exactly once. Background
   routing stays unchanged. This is not `pose_adapt_ratio`; the reference K/V
   route remains explicit and auditable.
3. Load the untouched E13 24k checkpoint. Sweep `alpha ∈ {1.00, 0.85, 0.70,
   0.55, 0.35}` with no training on the full fixed-96 panel and a named hard
   subset: both Skiing prompts, Crying, Laughing, Jumping and Dancing for
   Marion/Jisoo/Lex/Keanu, plus corrected Eddie Skiing, Crying, Laughing and
   Dancing.
4. Log versioned mask-owned ID, legacy ID, TOPIQ-Face, face detection,
   mask IoU/center/size, and per-image outputs. Perform blind visual ranking for
   goggles/eyes, hands/skin boundaries, teeth/mouth, and small-face anatomy.
5. Run reference-shuffle and reference-zero causal controls at the selected
   alpha to show that the identity effect still comes from reference K/V.

**Gates.** `alpha=1.00` must be pixel-exact or deterministically metric-exact to
E13. Promote a lower alpha only if corrected mask-owned ID falls by at most
`0.01` overall, no identity collapses, alignment does not regress, and the hard
set shows a clear blinded anatomy/occluder win. If a Pareto point exists, run a
short E13 continuation with a bounded per-layer or per-query gate initialized
to that alpha, a reference floor of `0.35`, and gate telemetry. If no Pareto
point exists, stop; do not train it.

## Experiment 3 - P2 E13 target-scale-matched spatial reference diagnostic

**Goal.** Give small dynamic target faces a spatial reference at a compatible
latent scale without changing the global PhotoMaker identity inputs.

**Config.** `E13_subject_v2_target_scale_eval` (new, defaults-off).
**Single scientific change:** spatial reference crop scale before the explicit
BA K/V path. **Hypothesis:** Jumping and Dancing faces have too few latent cells
for the current large canonical reference, so scale mismatch contributes to
eye, mouth and contour artifacts. **Prediction:** a target-matched crop improves
small-face visual quality while keeping face center and identity stable.
**Risk:** aggressive downscaling can erase identity detail or leak background.

**Implementation.**

1. Load the untouched E13 24k checkpoint and P0 subject-v2 references. Keep the
   PhotoMaker identity image/vector unchanged; modify only the spatial BA input.
2. Reuse the versioned reference-frame utilities developed for CL9/CL10. Build
   crops whose face short side is `{0.75, 1.00, 1.25}` times the requested
   target-mask short side, plus the exact E13 baseline. Preserve aspect ratio,
   use deterministic padding, and record the source bbox/crop transform.
3. Run fixed-96 for every setting with primary cells Marion/Jisoo/Lex Jumping
   and Dancing; include Eddie Jumping/Dancing and Skiing/Crying controls.
4. Log mask-owned and legacy ID, TOPIQ-Face, face count, mask IoU/center/size,
   crop scale and per-image outputs. Blind-rank eyes, mouth, contour and
   face/body integration at native resolution.
5. Keep `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, scheduler, seeds,
   masks, 50 steps and CFG 5 fixed. Do not combine multi-reference conditioning
   in this experiment.

**Gates.** The baseline must reproduce E13 exactly. Promote only a scale that
improves the blinded small-face set and TOPIQ-Face, loses at most `0.01`
full-panel mask-owned ID, introduces no face-center/ownership regression, and
does not worsen Skiing/Crying anatomy. If no setting is Pareto-improving, stop
without training.

# Confidence, limits, and reproduction

| Conclusion | Confidence | Basis |
|---|---|---|
| Eddie historical metric/conditioning selects the wrong face | **Very high** | detector boxes, embedding cosine, three direct `faces[0]` code paths **[measured] [code]** |
| Correct subject selection materially recovers Eddie identity | **Very high** | exact-replay paired intervention; `+0.289` to `+0.360`, 36/36 prompt-level wins **[measured]** |
| Contract-v2 correction preserves Eddie mask ownership | **Very high** | 36/36 historical pixels exact; corrected median IoU `0.875-0.891`, no IoU below `0.30` **[measured]** |
| Chef/Lex exposed a manual-vs-active bbox hand-off error, not a live BA ownership failure | **Very high** | E13 active-box winner IoU `0.8987`; unused-manual-box score `0.1391` **[measured] [code]** |
| Hard cells are shared across datasets/models | **High** | 41 checkpoints; rank correlations `0.746-0.800` **[measured]** |
| CL11 improves aggregate ID but worsens Marion/alignment tail | **High** | controlled 18k and saved 20k panel **[measured]** |
| Hard reference-only replacement causes occluder conflict | **Medium-high** | code routing + consistent visual failure; not yet isolated by dual sweep **[code] [visual] [hypothesis]** |
| E13→curated cosmic/BigCelebs mix is better than BigCelebs→E13 | **Medium** | dataset audit and endpoint interactions; curriculum not yet completed **[report] [hypothesis]** |

## What is ruled out

- **A single bad dataset is not the root cause.** The same hard cells recur
  across Large, BigCelebs and cosmic training over 41 checkpoints.
- **The Eddie BA reference crop is not the selector error.** Its spatial bbox
  already encloses foreground Eddie; the wrong person enters through the
  separate ArcFace/PhotoMaker embedding choice.
- **Face absence or simple undersizing is not sufficient.** Every controlled
  18k image and every corrected Eddie image has a detection, and median size is
  near the requested mask even when anatomy is visibly broken.
- **The earlier Kickboxing/Jumping relocation was not a model conclusion.** It
  disappears under strict processor copy, shadow restore and batch-12 replay;
  those invalid sidecar images must not guide architecture priorities.
- **Correcting the selector alone is not a complete visual-quality fix.** It
  recovers identity and ownership, but Skiing, Laughing, Dancing and Chef still
  expose occlusion, mouth/eye, small-face and extra-person failures.

These are measured exclusions; they do not prove the proposed dual-message or
target-scale mechanisms until those causal sweeps run.

Not yet established:

- Whether a dual native/reference mix has a usable identity/anatomy Pareto
  point in the trained E13 model.
- Whether target-scale matching improves small-face anatomy without erasing
  identity detail.
- Whether Marion's CL11 regression is caused by reference disagreement, pose,
  hair/occlusion, or training noise; the comparison establishes the regression,
  not its isolated cause.
- CL10's 24k endpoint; only its exact controlled 18k panel is used here.
- CL10/CL11 TOPIQ-Face at 18k; missing values are not imputed.

Reproduction assets are under
`analysis/assets/problematic_validation_20260809/`. Principal scripts:

- `build_analysis_assets.py`;
- `rescore_eddie.py`;
- `audit_multiface_idsim.py`;
- `build_endpoint_summary.py`; and
- `run_corrected_eddie_sidecar.py` for the final-checkpoint corrected-reference
  run;
- `analyze_final_corrected_sidecar.py` for paired identity, detection and fixed
  mask analysis.

Derived numerical tables are in the `data/` subdirectory:

- `run_summary_18k.csv`;
- `persistent_hard_cells_across_steps.csv`;
- `multiface_idsim_audit_18k.csv`; and
- `endpoint_summary.csv`;
- `corrected_eddie_final_checkpoint_rows.csv`; and
- `corrected_eddie_final_checkpoint_summary.json`.

The corrected images, `per_image.json`, resolved configs, input manifests and
command manifests are under `final_checkpoint_sidecar_contract_v2/` in
model-specific folders. The three replay gates passed 12/12 exact and the
local copy verifies all 105 files against Serv's output SHA-256 manifest.

## Exact reproduction commands

Generation ran as one guarded Serv job,
`lm-mpi-job-baea4903-7f8d-4785-a67d-f153df3299da`, using each experiment's
immutable runtime snapshot plus the patched evaluator as an external overlay.
The runtime trees were not edited. The package and exact launcher are under
`serv_run_packages/eddie_revalidation_contract_v2_serv_20260809_r1/`.

```bash
cd /home/kolyangg/rsrch_apr_test/diffusion_template
python analysis/assets/problematic_validation_20260809/analyze_final_corrected_sidecar.py
python analysis/assets/problematic_validation_20260809/build_eddie_pre_post_report_assets.py
python tools/reports/publish_report.py \
  analysis/2026-08-09_problematic_validation_e13_cl10_cl11_bc_e13.md --upload
```

`command_manifest.json` inside each model folder is the authoritative argv;
`run_manifest.json` is the authoritative checkpoint and generation contract.

# References

- [`docs/validation_protocol.md`](../docs/validation_protocol.md) - standard
  fixed-96 contract and labeled 12-image diagnostic exceptions.
- [`2026-08-09_problem_prompts_identities_root_cause_and_e25_e27.md`](2026-08-09_problem_prompts_identities_root_cause_and_e25_e27.md) - prior
  hard-prompt and architecture analysis.
- [`2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md`](2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md) - cosmic run
  comparisons and multi-reference effects.
- [`2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md`](2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md) - Large versus
  BigCelebs dataset audit.
- The report asset root's `data/` directory contains derived numerical
  evidence; `final_checkpoint_sidecar_contract_v2/` contains immutable corrected
  generation manifests and images.
