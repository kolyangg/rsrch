---
title: "CL14 hard cases: architecture diagnosis and six next experiments"
subtitle: "Marion, face-overlapping objects, small faces, prior-project evidence, external research, and Serv-ready implementation blueprints"
date: "11 August 2026"
---

# Executive decision

CL14 is the strongest corrected fixed-96 run in this audit, but it has not
solved the three hard-case mechanisms. `[measured]` At step 24,000 it reaches
**0.45612 subject-v2 identity**, `+0.00812` over CL9, and wins `63/96` paired
cells. Marion improves the most by identity (`+0.03879`) but remains the lowest
identity at `0.34998`; Skiing and Crying regress against CL9; and Jumping plus
Dancing remain the lowest-resolution prompt families.

The highest-probability next move is **not another global ID loss, a larger
rank, more PhotoMaker references, a hard occluder polygon, or broad BigCelebs
training**. Those directions already failed or traded away the required
behavior. The recommended sequence is:

| Priority | Proposed arm | Main target | Evidence / confidence |
|---:|---|---|---|
| **1** | **CL15 high-resolution ROI BA** | Small faces; Marion | Replicated ROI gain + RealisID; **high** |
| **2** | **CL16 clean reference memory** | Broad ID; Marion | Stable multi-scale K/V + DreamCache; medium-high |
| **3** | **CL17 semantic ownership gate** | Skiing/Crying topology | Query gating + PersonaHOI; medium |
| **4** | **CL18 cross-view BA consistency** | Marion; action/view robustness | Spatial-lane view invariance; medium |
| **5** | **CL19 true-soft full-query router** | Face/hair boundary | Known CL14 mask discrepancy; medium |
| **6** | **CL20 hard-case curriculum** | All three data tails | BigCelebs salvage, not replacement; medium-low |

These are **six independent ablations**, not one stacked architecture.
PhotoMaker and the explicit BA invariant remain in every architecture arm:
target queries stay in target coordinates and retrieve identity/reference
information through reference K/V. All arms keep
`pipeline.pose_adapt_ratio=0` and `pipeline.ca_mixing_for_face=false`.

One code finding changes how CL14 should be interpreted. `[code]` CL14 creates
a two-cell training feather with values `1/3` and `2/3`, but every installed BA
processor defaults to `force_binary_masks=true` and thresholds the resized mask
at `>0.5`. The nominal soft ramp therefore becomes **a hard one-cell erosion**.
CL14's gain cannot be attributed to continuous blending. Simply setting the
flag to false is also unsafe: the present processor soft-masks queries and then
soft-masks the output, which is not a single convex native/reference blend.
CL19 specifies the correct implementation.

The six YAMLs are in an implementation-gated
[blueprint folder](../experiments/designs/cl14_next_20260811/README.md).
They deliberately set `launchable: false`; no training job was submitted for
this analysis.

![CL14 base and the three highest-priority independent additions.](assets/cl14_architecture_20260811/fig_proposed_architecture.png){ width=100% }

## Scope, evidence tags, and comparison contract

This report combines immutable Comet metrics and images, local code/config
inspection, the completed CL9 fixed-checkpoint intervention study, current and
previous-project reports, corrected BigCelebs comparisons, and primary research
papers. It makes no change to weights, validation inputs, or metric definitions.

- **`[measured]`** means a value came from immutable experiment assets or a
  deterministic local measurement on those assets.
- **`[measured: visual]`** means the claim comes from inspected full images or
  fixed-box crops, not a numeric metric.
- **`[code]`** means it follows from composed CL14 configuration or executed code
  inspection/probes.
- **`[report]`** means it is established by a linked prior project report.
- **`[hypothesis]`** means the mechanism is plausible but untrained here.
- **`[not established]`** marks remaining uncertainty explicitly.

The standard decision contract remains 24,000 optimizer steps, batch size 2,
one A100, step 0 plus every 2,000 steps, fixed 96-image `manual_val`, one image
per item, corrected subject-v2 identity embeddings, DDIM50, and unchanged
prompts, seeds, references, boxes, scheduler, CFG, and face-quality definitions.
Derived tables join by canonical output key/numeric dataset row, avoiding the
space/underscore join trap documented by the report skill.

## Immutable experiment evidence

| Run | Immutable Comet key | Data | Corrected subject-v2 ID @24k |
|---|---|---|---:|
| CL14 | `6fe0028be92242c38056b3d36665fdd6` | Cosmic Large | **0.456116** |
| CL9 | `81bb311ed70545eda3281c64bc48be47` | Cosmic Large | 0.447997 |
| E13 | `1cc0a02371094b24a6a02a4cc649f10c` | Large Dataset | 0.430367 |
| BC_E13 | `c138db7c41ae435c8a7560f40cf5f58d` | BigCelebs | 0.415172 |
| BC ds1 | `b5b23b0ca4b449bc8f4703d6a7334be1` | BigCelebs repeat-depth | 0.414865 |
| BC ds2 | `5db54d7d4557487e94251656736843db` | BigCelebs scene/canonical | 0.424302 |
| BC ds3 | `43adf33cf7174e89b8fde1cdd640a052` | Large/BigCelebs anchor | 0.429634 |

`[measured]` CL14 peaks at `0.457096 @22k` and ends only `0.00098` lower, so
the conclusion does not depend on selecting a transient checkpoint. The CL14
record was backfilled from its immutable Comet key because the original local
plan JSON did not contain the written key; the generated images and metrics
were retrieved by the immutable key, not by display name.

# 1. What CL14 improves—and what remains

## 1.1 Run-level result

| Metric @24k | CL9 | CL14 | CL14 − CL9 |
|---|---:|---:|---:|
| Corrected subject-v2 ID | 0.447997 | **0.456116** | **+0.008120** |
| Paired wins | — | **63/96** | — |
| Detection rate | — | **96/96** | — |
| Mean mask IoU | — | **0.89794** | — |
| TOPIQ-Face mean | — | **0.68638** | — |
| TOPIQ-Face p10 | — | **0.57801** | — |
| Generic TOPIQ | — | 0.57754 | — |
| MUSIQ | — | 70.817 | — |
| MANIQA-PIPAL | — | 0.61313 | — |
| Text similarity | — | 26.27686 | — |

`[measured]` A naive cell bootstrap gives a 95% interval of
`[+0.00034,+0.01619]` for the paired CL14−CL9 mean. The more conservative
identity-cluster interval is `[-0.00150,+0.01914]`, and a two-way
identity/prompt cluster interval is `[-0.00556,+0.02312]`. CL14 is the best
observed run, but this one-seed fixed panel does not prove a population-wide
gain independent of identity and prompt composition.

![CL14 versus CL9 by validation identity and prompt family.](assets/cl14_architecture_20260811/fig_cl14_failure_profile.png){ width=100% }

## 1.2 The remaining floor is structured

| Identity | CL9 | CL14 | Delta |
|---|---:|---:|---:|
| Jensen | 0.55575 | **0.54527** | −0.01048 |
| Keanu | 0.49825 | 0.50027 | +0.00203 |
| Elon | 0.47669 | 0.49547 | +0.01879 |
| Jennie | 0.46336 | 0.46890 | +0.00554 |
| Jisoo | 0.44357 | 0.46038 | +0.01681 |
| Lex | 0.41972 | 0.41627 | −0.00345 |
| Eddie | 0.41545 | 0.41238 | −0.00307 |
| **Marion** | **0.31119** | **0.34998** | **+0.03879** |

| Hard prompt | CL9 | CL14 | Delta | Reading of the result |
|---|---:|---:|---:|---|
| Jumping | 0.32776 | **0.33244** | +0.00468 | Absolute/local-resolution floor remains |
| Skiing | 0.36222 | **0.35477** | −0.00745 | Object topology and visible identity remain unstable |
| Dancing | 0.35008 | **0.35736** | +0.00728 | Absolute/local-resolution floor remains |
| Crying | 0.47330 | **0.46263** | −0.01068 | Occlusion plus expression/metric ambiguity |
| Laughing | 0.42012 | 0.44679 | +0.02668 | Strong mean gain, but Marion remains low |

`[measured]` CL14 is not simply trading identity for geometry. Every face is
detected, the median detected-to-requested face-size ratio is `1.0145`, median
IoU is `0.9078`, and no row has IoU below `0.3`. The weak small-face cells fill
their requested boxes; their problem is insufficient absolute facial detail,
not systematic box underfill.

# 2. Failure mechanism 1: Marion

Marion is improved but not solved. Her CL14 mean is `0.34998`, `0.0624` below
the next-lowest identity, while individual prompts span `0.1094–0.4646`.

| Prompt | CL14 ID | Delta vs CL9 | Mechanism suggested by the image |
|---|---:|---:|---|
| Reading | 0.4333 | +0.0511 | Large, mostly clean face; recognizable but off-axis |
| Rushing | 0.3668 | +0.0540 | Hair and view change |
| Skiing | **0.1993** | −0.0220 | Goggles own most eye-region pixels |
| Drumming | 0.3824 | +0.0175 | Expression/view change |
| Kickboxing | 0.4246 | +0.0806 | Gloves near face but visible skin remains |
| Dancing | 0.3356 | −0.0556 | Small/action face and expression |
| Angry | 0.3659 | +0.0257 | Expression deformation |
| Crying | 0.4646 | +0.0950 | Hands/tears preserved; high ID despite occlusion |
| Laughing | 0.2972 | **+0.1918** | Large expression change; metric remains low |
| Jumping | **0.1094** | −0.0013 | Small blurred face and motion |
| Night ride | 0.4431 | +0.0236 | Large, illuminated face |
| Chef | 0.3775 | +0.0052 | Hat/hair boundary; clean central face |

![All twelve CL14 Marion face crops, with CL14 score and paired CL9 change.](assets/cl14_architecture_20260811/fig_cl14_marion_face_crops.png){ width=94% }

Three conclusions follow.

1. `[measured: visual]` Marion is not a uniform “wrong person” collapse. Several
   large clean outputs carry coherent facial structure, while the metric remains
   lower than for other identities. Optimizing a Marion-only ArcFace objective
   risks learning the scorer or frontalizing the face instead of improving
   prompt-conditioned likeness.
2. `[report]` Removing `−7.65°` eye-line roll did not replicate: four-seed mean
   identity change was `+0.001`, only `19/48` wins, with interval
   `[-0.008,+0.012]`. Five-point similarity was also neutral/unstable. The easy
   2D-normalization explanation is rejected by the completed
   [CL9 intervention report](2026-08-11_cl9_validation_interventions_results.md).
3. `[hypothesis]` The residual problem is a combination of view-specific
   reference features, inadequate local bandwidth under action/small faces, and
   semantic ownership under occlusion. This motivates P1/P2/P4 rather than a
   Marion-specific embedding or another same-image alignment.

`[not established]` A genuinely different, frontal, same-ID Marion photograph
has not been tested as the **spatial BA reference**. E19/CL11 only added images
to PhotoMaker identity-token preparation; their first image remained the sole
spatial reference. That is why P4 is not a repeat of the failed multi-reference
arm.

# 3. Failure mechanism 2: objects over the face

Skiing and Crying must not be optimized as the same problem.

| Identity | Skiing ID | Crying ID | Interpretation |
|---|---:|---:|---|
| Eddie | 0.4056 | 0.4208 | Goggles/tear expression both reduce clean-face evidence |
| Elon | 0.3444 | 0.4885 | Ski geometry is much harder than hands/tears |
| Jennie | **0.5579** | 0.3298 | Strong Ski identity but weak Crying expression |
| Jensen | 0.3171 | **0.6093** | Crying preserves identity exceptionally well |
| Jisoo | **0.2107** | **0.3206** | Both are severe; Skiing has topology failure |
| Keanu | 0.4273 | **0.6075** | Crying is visually/metric robust |
| Lex | 0.3758 | 0.4600 | Moderate degradation |
| Marion | **0.1993** | 0.4646 | Goggles catastrophic; hands/tears remain compatible |

\clearpage

![Matched CL9/CL14 hard cases and CL14 fixed-box ROIs.](assets/cl14_architecture_20260811/fig_cl14_hardcase_contact_sheet.png){ width=78% }

\clearpage

`[measured: visual]` Skiing often treats the reference face rectangle as one
semantic material: goggles can move to the forehead, duplicate, or reveal a
different facial surface. This is an **ownership/topology** error. Crying more
often preserves the correct person but changes how much face is visible through
hands, tears, and expression. A raw ID increase can therefore reward removal of
the requested hand or tear geometry.

The CL9 causal interventions constrain the design:

- `[report]` A family-wide static occluder mask was neutral on identity and
  shrank faces. It is rejected.
- `[report]` Reviewed per-image polygons improved non-Eddie identity by `+0.038`
  with `9/14` wins, but Skiing reached only `4/7`; goggles still duplicated or
  relocated. Better geometry alone is insufficient.
- `[hypothesis]` The missing variable is **what each query represents at the
  current timestep**. A hand/goggle/hair query inside the face rectangle should
  retain target-native K/V context, whereas a visible cheek/eye/nose query
  should use explicit reference K/V. The gate must be query-dependent, learned
  with visibility evidence, and prevented from retreating to the native path.

This is the reason P3 computes both complete lanes and blends once. It does not
delete reference keys or substitute target K/V into BA. It supervises an
occluder probability using deterministic real parsing plus exact synthetic
alpha, initializes to CL14 behavior, enforces a clean-skin reference floor, and
logs correct/wrong-reference causality.

# 4. Failure mechanism 3: small faces

Small faces are the strongest causal finding in the whole audit.

`[measured]` Jumping and Dancing average about `0.3449` subject-v2 ID in CL14.
Their TOPIQ-Face mean is `0.5458`, with minimum `0.4385`, versus `0.6864`
overall. Yet their boxes are filled and detected. The remaining bottleneck is
absolute spatial resolution.

For an approximately 121-pixel face in a 1024 image, the available width is:

| Representation | Grid width | Approximate face tokens across |
|---|---:|---:|
| Image | 1024 | 121 px |
| VAE/highest U-Net latent | 128 | 15.1 |
| 64×64 feature stage | 64 | 7.6 |
| 32×32 feature stage | 32 | **3.8** |

`[report]` The completed four-seed CL9 ROI suffix is the decisive intervention.
An 18-step local face refinement improved non-Eddie identity by **`+0.09684`**,
won `43/56` pairs, and had a target-row clustered interval of
`[+0.04354,+0.14659]`; each seed was positive. Every composite was exact outside
the ROI. Twenty steps also passed but moved the face/boundary more and gained
less identity. This is direct evidence that local denoising/detail capacity is
causal.

P1 turns that inference-time result into a shared training/inference mechanism:

1. Detect the declared target face short side from the unchanged fixed box.
2. At selected high-resolution decoder stages, crop target and reference face
   features and resample each to a fixed 32×32 lattice.
3. Reuse CL14's rank-128 BA projections: target ROI supplies Q, reference ROI
   supplies K/V.
4. RMS-bound and gate the result, inverse-grid-sample it into target coordinates,
   and apply a two-cell cosine scatter support.
5. Preserve the entire normal CL14 full-grid path. The auxiliary residual is
   exactly zero outside the target ROI.

`[hypothesis]` Sharing CL14's BA projections is important: a completely new
local identity network could learn a pasted face. A shared local view asks the
same identity routing to operate at adequate resolution. RealisID independently
supports a crop/upscale/local-process/scatter design, but P1 retains PhotoMaker
and explicit BA rather than importing RealisID's model.

# 5. CL14 architecture audit

## 5.1 What the composed config actually runs

`[code]` Hydra composition of
`src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml` shows:

- hard branched self-attention, target Q and reference K/V;
- BA LoRA rank 128, generic rank 32, PhotoMaker-default rank 64;
- joint training with shadow validation processors;
- uniform-all training timestep policy;
- `ba_hard_v1_true_reference_key_mask=false`;
- `ba_training_mask_feather=2`;
- `pose_adapt_ratio=0` and `ca_mixing_for_face=false`;
- no composed pipeline override for `force_binary_masks`.

The reference latent is encoded once, then noised at the current target
timestep and processed in parallel through the U-Net. Thus CL14 has a global
PhotoMaker identity-token lane and a spatial BA lane, but no stable clean
multi-scale feature memory.

## 5.2 The “soft mask” is binary erosion

The training mask builder in
`src/model/photomaker_branched/lora2.py` writes two inward rings:

```text
outer ring = 1 / 3
inner ring = 2 / 3
interior   = 1
```

Every relevant processor in
`src/model/photomaker_branched/attn_processor_cleanest.py` initializes
`force_binary_masks = True` and later executes the equivalent of:

```python
m2d = interpolate(mask, size=attention_resolution, mode="bilinear")
if force_binary_masks:
    m2d = (m2d > 0.5)
```

`[code]` An executed 8×8 probe preserved only `{0,1}`. A nominal 6×6 face mask
with the two-cell ramp became a hard 4×4 support: the `1/3` ring disappeared and
the `2/3` ring became one. At deeper resolution, already-small supports can
shrink again after interpolation/thresholding.

This does **not** make CL14 invalid. Its actual single change is a target-mask
erosion, and the corrected images show that this treatment performs well. It
does mean:

- the config comment and experiment interpretation overstate continuous
  handover;
- CL14 does not test the proposed soft blend;
- a true-soft arm remains scientifically distinct;
- `force_binary_masks=false` alone is not the correct treatment.

`[code]` In the current attention implementation, target Q is masked before
attention and the result is masked again. For continuous `m`, this changes the
softmax logits and then scales the output; it is not
`(1-m)A_native + mA_reference`. P5 therefore separates the target router from
the reference key mask, computes both full-query lanes, and performs exactly
one blend.

## 5.3 Why true reference-key masking is excluded

CL14 intentionally keeps invalid reference positions as zero K/V “sinks” in
the softmax. Removing those positions changes normalization, not only geometry.
`[report]` E1 true-key masking (`ce0c9b918d79449b92fa83ef970285c3`)
never recovered the successful branch; its best trained identity was about
`0.29664 @16k` under that suite's historical metric, used here only as a
within-suite negative. `[code]` A current-layer probe showed roughly 13×
reference amplification when zero sinks were removed. Consequently:

- no proposed arm enables `ba_hard_v1_true_reference_key_mask` by itself;
- compact/clean memories use RMS normalization, zero initialization, and a
  bounded residual gate;
- semantic ownership changes target queries' output ownership, not reference
  softmax normalization.

## 5.4 Architectural opportunity map

| Missing capability | Direct evidence | Proposed treatment |
|---|---|---|
| Adequate local tokens | Small faces fill boxes; ROI suffix is strongly positive | P1 high-resolution ROI BA |
| Stable identity detail at every timestep | Reference features are repeatedly noised | P2 clean multi-scale memory |
| Semantic ownership inside face box | Static/polygon masks fail on goggles/hands | P3 learned visibility ownership |
| View-invariant spatial identity | Marion roll fails; first reference owns spatial lane | P4 cross-view spatial consistency |
| Actual continuous handover | CL14 feather is thresholded | P5 true-soft single router |
| Useful hard-case data without domain replacement | BigCelebs breadth/portrait bias loses overall | P6 curated minority curriculum |

# 6. What prior experiments already rule out

The earlier project contains many relevant ideas, but their results must be
read through the now-corrected validation/architecture state. The most useful
reports are the recent-run idea audit, expanded step-zero architecture study,
4k results, NN4 analysis, and NN5a analysis under
`/home/kolyangg/rsrch/Jul_new_exp/`. Their durable architectural lessons are:
target-coordinate Q protects pose; broad CA or target/reference mixing can
over-control the scene; bounded residuals are safer than replacement; and a
learned branch needs causality telemetry because it can learn to ignore its
reference.

The current corrected experiment history is more decisive:

| Treatment | Corrected best/final ID | Decision for next suite |
|---|---|---|
| E19 extra PhotoMaker references | 0.43330 @16k / 0.42876 @24k | No repeat; no extra spatial reference |
| E20 branch-output rank32 | 0.43204 @24k | No standalone capacity repeat |
| E21 multi-ref + branch output | 0.43312 @22k / 0.42984 @24k | Combination does not recover CL14 |
| E22 naive generated-x0 ArcFace | 0.42938 @20k / 0.42663 @24k | Exclude another naive ID loss |
| E23 early BA decay | 0.38778 best | Exclude reference decay |
| E24 alternating route | 0.37491 best | Exclude route alternation |

Immutable Comet keys:

- E19 `3280232a45ef4ea2ae68c8deff3b81c1`
- E20 `4084c35600ae4ad3904446e5f4d2de92`
- E21 `3ef78907f60a4f5cbd7727fc5be7143e`
- E22 `5a91be0df76f4966be5c77eee26cfc29`
- E23 `9b6942c0ee6740c7aa4d3fe74effee93`
- E24 `5b64f84f134441b791e7c3ffbd6fe4f7`

Other completed evidence:

- `[report]` Reference-scale calibration fixed undersized **box fill**, but did
  not fix identity detail once the face was genuinely small.
- `[report]` Broad pose adaptation and face cross-attention mixing damaged the
  intended reference-conditioned BA invariant and/or prompt/layout behavior.
- `[report]` Static and precise occluder masks do not solve topology.
- `[report]` Marion roll/five-point normalization does not replicate.
- `[report]` The replicated 18-step local ROI intervention is strongly positive.

The six proposals therefore change **where identity bandwidth exists, how
stable the reference evidence is, who owns an occluding query, or how spatial
identity is regularized**. They do not repeat a known global-strength knob.

# 7. External research synthesis

Nineteen primary papers were downloaded, hashed, and text-extracted in the
[local source archive](sources/2026-08-11_cl14_architecture_review/SOURCES.md).
The archive includes every method requested by the user plus closely related
and recent work. Published benchmark numbers are not compared directly with the
project's corrected subject-v2 metric because datasets, crops, prompts, and
scorers differ.

## 7.1 Requested methods

| Method | What is useful here | What is not transplanted |
|---|---|---|
| [PuLID](https://arxiv.org/abs/2404.16022) | Four-step accurate-x0 ID supervision plus semantic/layout contrast between native and personalized paths shows how to protect editability | E22 already rejects naive x0 ID loss; a Lightning branch is expensive and not top-six until routing improves |
| [DreamCache](https://arxiv.org/abs/2411.17786) | Cache U-Net features at `t=1`, null text, middle plus decoder scales; target queries retrieve cached K/V | Cache-only personalization would replace rather than strengthen CL14 BA |
| [DynamicID](https://arxiv.org/abs/2503.06505) | Query-level Semantic-Activated Attention and multi-view identity/motion reconfiguration | An unconstrained activation gate can learn reference retreat |
| [UniPortrait](https://arxiv.org/abs/2408.05939) | Separate intrinsic identity and structure features; conditioning dropout prevents dependence on one cue | Its full encoder/router would displace PhotoMaker |
| [InfiniteYou](https://arxiv.org/abs/2503.16418) | Residual personalization and same-person/different-image training reduce copy-paste bias | InfuseNet is not adopted as a replacement architecture |
| [SpatialID](https://arxiv.org/abs/2602.13994) | Spatial/time relevance should be adaptive rather than one hard rectangle | Literal schedule is not adopted: its reported default FaceSim trade-off is large and evidence is small-scale |
| [Diff-PC](https://arxiv.org/abs/2602.00639) | Explicit identity-versus-pose disentanglement and same-ID cross-view training | No 3D injector or new raw ID loss; target Q stays native |
| [AnyPhoto](https://arxiv.org/abs/2603.14770) | Identity-isolated attention, explicit location, and same-ID replacement curriculum | Its own ablation warns that aligned modulation/location can copy-paste without balancing losses |

## 7.2 Closest external matches to the three hard cases

| Paper | Connection | Proposed use |
|---|---|---|
| [RealisID](https://arxiv.org/abs/2412.16832) | Local crop/upscale/process/scatter plus global harmony is the closest analogue to the successful ROI suffix | P1, but with shared PhotoMaker BA projections |
| [PersonaHOI](https://arxiv.org/abs/2501.05823) | Native and personalized branches with local residual ownership improve human-object interactions | P3 native/reference lanes |
| [Leffa](https://arxiv.org/abs/2412.08486) | High-resolution/low-noise attention flow regularization aligns correct reference correspondence | Reserve after P4; same-image references create copy-paste risk |
| [ID-Patch](https://arxiv.org/abs/2411.13632) | Couple global identity and spatial identity patches; verify subject association | P2/P4 causality telemetry |
| [ReSem-Face](https://arxiv.org/abs/2608.04820) | Very recent large-occlusion work separates identity attention and semantic reconstruction | Supports P3 conceptually; face inpainting evidence is not directly portable |
| [MagicMakeup](https://arxiv.org/abs/2607.20924) | Token-aligned query/key masks demonstrate semantic attention ownership | Supports attention-domain gating, not E1-style key renormalization |

## 7.3 New connections for PhotoMaker + BA

### Connection A: identity bandwidth is both spatial and temporal

`[hypothesis]` P1 and P2 solve complementary bandwidth losses. P1 prevents a
small target face from collapsing to four tokens; P2 prevents reference detail
from disappearing as the reference is repeatedly noised. PhotoMaker continues
to supply a global identity prior, CL14 BA continues to supply timestep-aligned
spatial evidence, and the clean memory supplies stable detail. This is a
three-band design, not a replacement encoder.

### Connection B: semantic ownership should be a query property

`[hypothesis]` The relevant mask is not “inside the face box”; it is “does this
target query represent visible identity-bearing facial surface at this
timestep?” DynamicID's query activation and PersonaHOI's dual branch point in
the same direction. The target prompt/object path should own goggles, hands,
hair, tears, and helmet edges even inside the fixed face rectangle.

### Connection C: cross-view invariance belongs in the spatial lane

`[hypothesis]` Existing multi-reference experiments only enriched global
PhotoMaker tokens. Holding those tokens and target Q fixed while swapping the
spatial BA reference isolates what the spatial lane must learn: identity
features that survive reference yaw/expression. This is more diagnostic than
feeding three images everywhere.

### Connection D: causality telemetry is a training requirement

`[report]` Older learned gates and weak residual branches could improve base
quality by ignoring the reference. Every new branch must therefore log, by
layer and timestep, correct-reference, wrong-reference, and zero-reference ROI
effects. A high validation score is insufficient if the new branch gate remains
near zero or if correct/wrong references become indistinguishable.

### Reserve idea: PuLID-style accurate ID loss only after routing improves

PuLID shows that accurate generated-image ID supervision can outperform naive
predicted-x0 ID loss, but its ablation also shows the identity/editability
trade-off and the need for semantic/layout alignment. If P1 or P2 creates a
causal, bounded branch but plateaus, a seventh experiment could attach a
four-step accurate-x0 ID objective **only to the local face branch**, paired
with frozen-native semantic/layout preservation. It is deliberately excluded
from the top six because E22 and earlier ID-loss arms already show high reward-
hacking risk.

# 8. Dataset use: BigCelebs should be a curated auxiliary, not the base

## 8.1 Distribution and exposure

| Property | Large Dataset audit comparator | BigCelebs v2 | Consequence at fixed 48k targets |
|---|---:|---:|---|
| Images | 47,500 | 349,348 | BigCelebs consumes only 13.74% of images |
| Identities | 2,561 | 68,648 | Breadth replaces repeated identity evidence |
| Median images/ID | 18 | 4 | Most BigCelebs IDs receive too few consistent updates |
| Portrait/close-up captions | 0.324% | 83.97% | Weak action/body/occlusion match |
| Median face side | 255 px | 410 px | Weak direct supervision for small target faces |
| Hands/holding captions | 37.36% | 15.10% | Less face-object interaction coverage |
| Multiple people | 26.09% | 0.298% | Less scene complexity |

These statistics come from the completed
[E13 versus BigCelebs audit](2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md).
The left column is E13's Large Dataset, not CL14's Cosmic manifest; it provides
the controlled corpus audit that explains why “more images” did not become more
useful identity learning at the unchanged optimizer budget. The corrected
fixed-96 table below supplies the direct BigCelebs-versus-CL14 evidence.

## 8.2 Corrected 24k hard-case results

| Run | Overall ID | Small: Jumping+Dancing | Skiing | Crying | Marion |
|---|---:|---:|---:|---:|---:|
| **CL14** | **0.4561** | 0.3449 | **0.3548** | 0.4626 | 0.3500 |
| Broad BC_E13 | 0.4152 | 0.3291 | 0.3625 | 0.4375 | 0.3382 |
| BC ds1 repeat-depth | 0.4149 | 0.3214 | 0.3330 | 0.4418 | 0.3275 |
| BC ds2 scene/canonical | 0.4243 | **0.3493** | **0.2472** | 0.4214 | **0.3604** |
| BC ds3 2:1 anchor | 0.4296 | 0.3474 | 0.2717 | **0.4729** | 0.3283 |

`[measured]` No BigCelebs arm beats CL14 overall. Ds2 buys `+0.0104` Marion
and `+0.0044` small-face ID but loses `−0.1076` on Skiing. Ds3 buys `+0.0103`
Crying and `+0.0025` small-face ID but loses overall and Marion. Broad corpus
replacement is therefore rejected.

P6 instead uses a deterministic auxiliary manifest with:

- at least six images per identity and distinct target/reference files;
- balanced pose-divergent same-ID, 64–180 px target faces, and visible
  eyewear/hair/hand/action-occlusion strata;
- strict quality and validation-overlap filters;
- exactly 20% auxiliary probability for steps 0–19,999;
- 100% Cosmic data for steps 20,000–23,999 to re-anchor composition.

`[hypothesis]` This preserves repeated same-ID and hard-case evidence while
limiting portrait-domain drift. P6 remains last because the existing dataset
evidence is negative overall and selection metadata cannot guarantee identity
cohesion or useful occlusion semantics.

# 9. Six experiment implementation plans

## Common non-negotiable implementation contract

Before any run becomes launchable:

1. Add every new mechanism behind a backward-compatible disabled default.
2. Verify old/new Hydra composition and the exact trainable tensor/parameter
   inventory; do not reuse CL14's `2,240 / 219,217,920` claim after adding
   modules.
3. Verify processor installation in training and validation, checkpoint
   save/load, deterministic forward/backward, and finite nonzero reference
   gradients.
4. Preserve target-Q/reference-KV routing, reference boxes, zero sinks,
   PhotoMaker tokens, scheduler, prompts, and standard validation.
5. Run a 100-step smoke for correctness/memory only. It is not a quality result.
6. At startup, verify `saved/<run_name>/comet_experiment.json` contains the
   immutable key written by `CometMLWriter`.
7. Apply the 8k kill rule and final gate in the corresponding YAML; do not tune
   thresholds after seeing the result.

## P1 — CL15 shared high-resolution ROI BA

Blueprint:
[`01_CL15_shared_highres_roi_ba_24k.yaml`](../experiments/designs/cl14_next_20260811/01_CL15_shared_highres_roi_ba_24k.yaml)

**Hypothesis.** `[hypothesis]` A fixed-resolution local view restores the
target/reference identity bandwidth that the global latent loses for small
faces. It should preserve pose because Q remains sampled from target features.

**Implementation.** Add ROI sampling/scattering around selected `up_blocks.0`
and `up_blocks.1` BA stages in
`attn_processor_cleanest.py`, with runtime coordinates in
`branched_runtime.py`/`lora2.py`. Trigger only when declared target face short
side is at most 160 px. Resample target and reference features to 32×32, reuse
CL14 r128 BA projections, RMS-match the residual to the native lane, cap the
learned gate at `0.25`, and inverse-scatter through a cosine two-cell support.
The complete normal CL14 path stays active.

**Correctness proof.** Disabled mode must be byte-equivalent to CL14. Synthetic
boxes must prove coordinate round-trip and zero residual outside ROI. Correct
reference must produce a nonzero ROI gradient; wrong reference must change the
ROI. Train/validation checkpoint reload must reproduce tensors.

**Promotion.** Overall ID at least `0.4511`, small-family ID at least `0.3849`
(`+0.040`), small TOPIQ-Face at least `0.600`, 96/96 detection, and at least
10/16 blind Jumping/Dancing pair wins without face enlargement or a pasted seam.

**Main failure mode.** Local face repainting could improve ID while changing
expression or creating a high-resolution sticker. Bounded sharing, scatter-only
support, text guard, and full-image review are mandatory.

## P2 — CL16 clean multi-scale reference memory

Blueprint:
[`02_CL16_clean_multiscale_ref_memory_24k.yaml`](../experiments/designs/cl14_next_20260811/02_CL16_clean_multiscale_ref_memory_24k.yaml)

**Hypothesis.** `[hypothesis]` A clean reference memory stabilizes identity
detail that the timestep-noised CL14 branch cannot expose reliably.

**Implementation.** Run a reference-only U-Net pass once at `t=1` with null
text, cache face-supported features from `mid_block`, `up_blocks.0`, and
`up_blocks.1`, and project them to rank-64 K/V. Target Q cross-attends the cache.
Merge as a zero-initialized, RMS-normalized residual after the existing CL14 BA
output, capped at `0.20`. Do not replace the current noised-reference lane and
do not change the reference key mask/zero sinks.

**Correctness proof.** One cache build per reference/denoising call; identical
disabled output; bounded RMS ratios; correct/wrong/zero-reference probes;
checkpoint restoration of projections/gates; identical training and validation
cache construction.

**Promotion.** Overall ID at least `0.4661` (`+0.010`), Marion at least `0.3800`,
TOPIQ-Face at least `0.680`, text at least `26.17`, and blind preference on at
least 7/12 Marion rows without reference pose, hair, or background leakage.

**Main failure mode.** Clean high-resolution features can dominate the target
and copy the reference. Zero initialization, bounded RMS merge, target Q, face
support, and wrong-reference telemetry are essential.

## P3 — CL17 semantic visibility ownership gate

Blueprint:
[`03_CL17_semantic_visibility_ownership_gate_24k.yaml`](../experiments/designs/cl14_next_20260811/03_CL17_semantic_visibility_ownership_gate_24k.yaml)

**Hypothesis.** `[hypothesis]` A query's semantic material, not its rectangle
membership, should determine whether native or reference attention owns it.

**Implementation.** Compute full native target and CL14 reference-BA outputs.
Predict an occluder probability from target Q, native/reference disagreement,
prompt cross-attention summary, and timestep. Parameterize the reference gate
as `face_mask × (1 − p_occluder)`, initialize `p_occluder=0` to reproduce CL14,
enforce a visible-skin reference floor, and blend the two outputs exactly once.
Supervise with deterministic real face/human parsing plus exact synthetic alpha
for eyewear, goggles, hair strands, hands, and tears on 25% of training targets.

**Correctness proof.** Hash the augmentation manifest; verify image/alpha/prompt
alignment; prove CL14 output at zero occluder probability and native/reference
limits; log gate histograms on skin/occluder/hair/outside; require
correct-reference sensitivity.

**Promotion.** Overall ID at least `0.4511`, Skiing ID at least `0.375`, no
regression from CL14's Skiing/Crying TOPIQ-Face means, text at least `26.17`, and
at least 50% fewer blind-reviewed Skiing topology/duplication errors. Crying
must preserve tears, expression, and hand topology; ID alone cannot pass.

**Main failure mode.** The learned gate can improve native quality by disabling
the reference. The clean-skin floor, correct/wrong reference probes, and an 8k
gate-use kill rule prevent this false win.

## P4 — CL18 cross-view spatial BA consistency

Blueprint:
[`04_CL18_crossview_spatial_ba_consistency_24k.yaml`](../experiments/designs/cl14_next_20260811/04_CL18_crossview_spatial_ba_consistency_24k.yaml)

**Hypothesis.** `[hypothesis]` Spatial BA should represent identity consistently
across reference yaw/expression while target Q determines output pose.

**Implementation.** On 25% of Cosmic batches whose target has at least three
accepted ArcFace-retrieved reference candidates, materialize a deterministic
pair: high-quality canonical teacher reference and 15–55° pose-divergent
student reference. Reuse the same target latent, timestep, noise, target Q, and
teacher-derived PhotoMaker embedding in both passes; change only spatial BA K/V.
At `up_blocks.0/1`, compare normalized BA outputs in target-face coordinates
with stop-gradient teacher and `0.05 × (SmoothL1 + cosine)` loss. Inference
remains single-reference.

**Correctness proof.** Hash and decontaminate the pair manifest, manually audit
candidate identity because the Cosmic groups are ArcFace retrievals rather than
curated named identities, assert that only spatial reference tensors differ,
prove teacher stop-gradient, and measure
correct/wrong-reference sensitivity so consistency cannot be achieved by
ignoring the spatial reference.

**Promotion.** Overall ID at least `0.4561`, Marion at least `0.3800`, Skiing at
least `0.3650`, text at least `26.17`, and at least 7/12 Marion blind wins with
unchanged pose/composition/face size.

**Main failure mode.** Consistency can collapse the spatial lane to a generic
output. Wrong-ID sensitivity must not shrink by more than 10%.

## P5 — CL19 true-soft full-query router

Blueprint:
[`05_CL19_true_soft_fullquery_router_24k.yaml`](../experiments/designs/cl14_next_20260811/05_CL19_true_soft_fullquery_router_24k.yaml)

**Hypothesis.** `[hypothesis]` A single continuous output handover can reduce
hairline/cheek seams and abrupt ownership without weakening the central face.

**Implementation.** Separate two concepts: a continuous signed-distance
**target routing mask** and CL14's unchanged binary **reference key mask**.
Compute full native target self-attention and full target-Q/reference-KV BA;
blend once with a two-cell cosine transition. Never multiply Q by the soft
router. Preserve legacy binary behavior behind the default mode.

**Correctness proof.** Reproduce the current 6×6→4×4 erosion; verify monotone
`1/3,2/3` values after resizing; prove exact lane limits at router 0/1; prove
legacy/new equivalence for a hard mask; verify identical training/validation
router construction.

**Promotion.** Overall ID at least `0.4511`, TOPIQ-Face at least `0.6864`, mean
mask IoU at least `0.905`, 96/96 detection, and blind evidence of fewer face
boundary seams without reference bleeding into hair, goggles, hands, or clothes.

**Main failure mode.** CL14 may benefit from erosion, not softness. The arm is
scientifically clean and relatively narrow, but it must be killed if boundary
quality does not compensate for an ID loss.

## P6 — CL20 hard-case/deep-ID curriculum

Blueprint:
[`06_CL20_hardcase_deepid_curriculum_24k.yaml`](../experiments/designs/cl14_next_20260811/06_CL20_hardcase_deepid_curriculum_24k.yaml)

**Hypothesis.** `[hypothesis]` A small amount of repeated same-ID hard-case data
can improve robustness without inheriting broad BigCelebs portrait drift.

**Implementation.** Keep the CL14 model/loss/optimizer exact. Create and hash a
BigCelebs manifest with depth at least six, different target/reference files,
quality/overlap filters, and balanced pose-divergent, 64–180 px face, and
visible face-object/action strata. Use 80% Cosmic/20% curated BigCelebs for
steps 0–19,999 and 100% Cosmic for the final 4k. Sampling decisions and manifest
hash are immutable and logged.

**Correctness proof.** Report stratum/depth/face-size distributions and contact
sheets; hash images and prove zero validation overlap; simulate exactly 24k
sampler decisions across the boundary; prove step-0 model state/trainables are
byte-equivalent to CL14.

**Promotion.** Overall ID at least `0.4611`, Marion at least `0.3700`, small
family at least `0.3650`, text at least `26.17`, and hard-case gains must survive
the final Cosmic-only phase without portrait-copy bias or lost action layout.

**Main failure mode.** Selection proxies may still choose easy portraits or
fragmented celebrity identities. Do not raise the BigCelebs fraction in-flight.

# 10. Measurement and Serv execution plan

## 10.1 Decision scorecard

The primary scalar remains corrected fixed-96 subject-v2 identity, but the hard
cases require a vector decision:

| Failure family | Primary | Mandatory guards |
|---|---|---|
| Overall | Mean subject-v2 ID and paired wins | Text, detection, TOPIQ-Face, full-image blind review |
| Marion | 12-row mean and wins | Per-prompt view/expression review; no frontal copy |
| Small | Jumping+Dancing mean | TOPIQ-Face, face size ratio, outside-ROI/scatter checks |
| Skiing | ID plus topology error count | Goggles/helmet/hair ownership, prompt adherence, face size |
| Crying | Visible-face identity retention | Hands/tears/expression topology; raw ID cannot promote alone |
| New branch | Correct-vs-wrong reference effect | Gate/RMS by stage and timestep; no branch retreat |

For occlusion, add **own-clean retention** as a derived diagnostic: each hard
cell's ID divided by that identity's mean over designated clean prompts. This
does not replace the metric; it distinguishes “same person survives reduced
visibility” from an identity whose baseline is generally low. The clean-prompt
set and formula must be frozen before training.

## 10.2 Priority and GPU use

Each blueprint requests one A100. Six simultaneous one-GPU requests would equal
the project's normal six-A100 ceiling, but that is not the recommended first
action. Implementation risk is different across arms.

1. Implement and smoke P1 first because it has replicated causal evidence.
2. Implement P2 and P5 next; both have clear disabled-mode equivalence tests.
3. Implement P3 only after the augmentation/gate telemetry is reviewed.
4. Implement P4 after a decontaminated same-ID pair manifest exists.
5. Build P6's manifest last, after architecture arms show which hard-case data
   is actually missing.

When launch is later authorized, inspect actual Running/Pending MLS jobs before
every submission and keep the project at or below six requested A100s. A pending
job counts. If MLS rejects a request for allocation/request limits, do not retry
without a later user request. No job in this report is currently authorized or
submitted.

## 10.3 Checkpoint selection

Use the predeclared final gates and 8k kill rules in the YAMLs. Report the full
step-0/2k trajectory; do not select a winner only from the endpoint. A model can
be promoted from an earlier checkpoint only if the selection rule is declared
before the run and all fixed-96 visual/quality guards pass. Preserve the exact
checkpoint semantics and immutable Comet identity.

# 11. Immediate low-risk action before training

The completed CL9 18-step ROI suffix is the only replicated causal
intervention. It should be replayed **unchanged on CL14 @24k as a fixed-
checkpoint confirmation**, using the same small-face rows and guards, before
the P1 implementation is finished. This is not a new training run and must not
change the default pipeline unless it replicates. Jensen should retain the
baseline fallback because it lost consistently in the CL9 study.

Do not spend another diagnostic cycle on Marion roll, five-point alignment, a
family-wide Skiing/Crying mask, or standalone true-key masking; those questions
are already answered negatively.

# 12. Confidence and what is not established

| Claim | Confidence | Basis |
|---|---|---|
| CL14 is the best observed corrected fixed-96 run in this set | High | Immutable metrics; 0.45612 @24k |
| CL14's nominal feather is re-binarized into erosion | **High** | Composed config, code path, executed mask probe |
| Small-face failure is local-resolution limited | **High** | Filled boxes plus replicated ROI causal gain |
| Static target geometry cannot solve face-overlapping objects | High | Static and reviewed-polygon interventions |
| Clean reference memory will improve CL14 | Medium | Strong external mechanism; no in-repo trained test |
| Semantic ownership will fix goggles/hands | Medium | Mechanistically aligned; gate retreat/topology remain risks |
| Marion needs cross-view spatial invariance | Medium-low | Heterogeneous prompt evidence; 2D alignment rejected |
| Curated 20% BigCelebs will help | Medium-low | Broad/ds arms are negative; proposed stratum is untested |

**Not established:**

- Whether CL14 itself replicates the 18-step ROI gain measured on CL9.
- Whether Marion's low metric is mainly model likeness, reference-view domain,
  or scorer sensitivity; the evidence supports a mixture, not a single cause.
- Whether a clean cache can avoid copying reference pose/hair/background under
  this exact PhotoMaker + BA stack.
- Whether prompt cross-attention reliably identifies glasses/hands/hair at the
  relevant U-Net stages.
- Whether current datasets contain enough high-quality same-ID pose-divergent
  pairs for P4 without identity fragmentation.
- Whether BigCelebs metadata proxies correspond to true semantic occlusion and
  identity consistency.
- Whether any two winning arms combine additively. Combination runs should only
  follow successful independent ablations.

# 13. Reproduction record and local artifacts

## Derived CL14 assets

- Script:
  [`analysis/assets/cl14_architecture_20260811/build_assets.py`](assets/cl14_architecture_20260811/build_assets.py)
- Paired 96-row table:
  [`cl14_vs_cl9_per_image.csv`](assets/cl14_architecture_20260811/cl14_vs_cl9_per_image.csv)
- Run/uncertainty summary:
  [`cl14_vs_cl9_summary.json`](assets/cl14_architecture_20260811/cl14_vs_cl9_summary.json)
- CL14 per-image face quality:
  [`face_quality_details__CL14__step_024000.csv`](assets/cl14_architecture_20260811/face_quality_details__CL14__step_024000.csv)
- Corrected BigCelebs endpoint tables:
  [`dataset_tables/`](assets/cl14_architecture_20260811/dataset_tables/)

## Source archive

- Manifest and takeaways:
  [`SOURCES.md`](sources/2026-08-11_cl14_architecture_review/SOURCES.md)
- PDF integrity manifest:
  [`SHA256SUMS.txt`](sources/2026-08-11_cl14_architecture_review/SHA256SUMS.txt)
- Nineteen PDFs:
  [`papers/`](sources/2026-08-11_cl14_architecture_review/papers/)
- Extracted text:
  [`text/`](sources/2026-08-11_cl14_architecture_review/text/)
- Eight immutable official-README snapshots:
  [`source_readmes/`](sources/2026-08-11_cl14_architecture_review/source_readmes/)

## Proposed configs

The [blueprint README](../experiments/designs/cl14_next_20260811/README.md)
freezes the common comparison contract and promotion procedure.
All six YAMLs parse successfully and have `status: design_only_not_runnable`
and `launchable: false`.

# 14. Primary references

1. Guo et al., [PuLID: Pure and Lightning ID Customization via Contrastive Alignment](https://arxiv.org/abs/2404.16022), NeurIPS 2024.
2. Aiello et al., [DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](https://arxiv.org/abs/2411.17786), CVPR 2025.
3. Hu et al., [DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](https://arxiv.org/abs/2503.06505), ICCV 2025.
4. He et al., [UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization](https://arxiv.org/abs/2408.05939).
5. Jiang et al., [InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity](https://arxiv.org/abs/2503.16418), ICCV 2025.
6. Li, [Inject Where It Matters: Training-Free Spatially-Adaptive Identity Preservation](https://arxiv.org/abs/2602.13994).
7. Xu et al., [Diff-PC: Identity-preserving and 3D-aware Controllable Diffusion](https://arxiv.org/abs/2602.00639).
8. Yuan, [AnyPhoto: Multi-Person Identity Preserving Image Generation with ID Adaptive Modulation on Location Canvas](https://arxiv.org/abs/2603.14770).
9. Hu et al., [PersonaHOI: Effortlessly Improving Personalized Face with Human-Object Interaction Generation](https://arxiv.org/abs/2501.05823).
10. Sun et al., [RealisID: Scale-Robust and Fine-Controllable Identity Customization via Local and Global Complementation](https://arxiv.org/abs/2412.16832).
11. Zhou et al., [Learning Flow Fields in Attention for Controllable Person Image Generation](https://arxiv.org/abs/2412.08486), CVPR 2025.
12. Ding et al., [When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions](https://arxiv.org/abs/2608.04820).
