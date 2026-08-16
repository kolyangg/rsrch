# CL21–CL26 current results: CL23 is the only clear early ID gain, but Skiing topology and BA causality remain open

**Date:** 14 August 2026  
**Evidence cutoff:** 11:34 UTC / 12:34 BST, 14 August 2026  
**Comet project:** `aug-large-ds`  
**Evaluation contract:** fixed 96-image `manual_val`, one image per item,
subject-v2 mask-owned ID similarity, fixed prompts/references/seeds, DDIM50,
CFG 5  
**Status:** current interim review; CL21, CL22, CL23, CL24, and CL26 were still
running at the cutoff; CL25 had completed 4k  
**Reproducible snapshot:**
[`assets/cl21_cl26_20260814_current/`](assets/cl21_cl26_20260814_current/)  
**Collector:**
[`collect_snapshot.py`](../analysis_sidecars/2026-08-14_cl21_cl26_current_review/collect_snapshot.py)

## Executive verdict

CL23 is the only CL21–CL26 arm with a clear, persistent matched-step identity
gain over CL19. Its current peak is **`0.525002` ID_SIM at 10k**. The newly
completed 12k gate softens to **`0.518674`**, but still beats matched CL19
(`0.480898`) by **`+0.037776`**, with 73/96 wins and a 95% bootstrap interval
of **`[+0.026193,+0.049563]`**. It also exceeds the completed CL19 24k
checkpoint by `+0.011851` with a fully positive paired interval.
**[measured][paired]**

This is not yet a PhotoMaker win. The controlled PhotoMaker step-zero baseline
is `0.556580`; CL23 12k remains `-0.037906` lower, with a fully negative paired
interval. PhotoMaker also remains visibly much better at preserving the layer
order “large ski goggles above the face, ordinary glasses on the face.”
**[measured][visual]**

CL23 is a fair **PhotoMaker + explicit branched-self-attention** experiment.
It is a cold start: Comet records `trainer.from_pretrained=null`,
`trainer.resume_from=null`, and `saved_checkpoint=null`. Its nontrivial
step-zero score is an expected architecture-initialization effect, not a prior
CL19 checkpoint leak. The branch LoRA residuals start at zero, but their base
Q/K/V/output weights clone the effective PhotoMaker/U-Net projections; CL23
then applies a different, nonzero temporal-frequency blend to the same
reference-minus-native message that CL19 already computes. **[config][code]**

The absolute CL23 score is not “BA-only.” PhotoMaker identity tokens, the
generic adapter, and the PhotoMaker default adapter are deliberately part of
the shared base and also train. This is fair for the project's stated
PhotoMaker+BA question because CL19 and CL23 share those components and differ
only in routing. It does mean that causal attribution of the *learned* 12k gain
to the spatial BA lane is not complete: no held-out BA-native, zero-spatial-ref,
or shuffled-spatial-ref panel has yet been logged. That diagnostic is mandatory
before promoting CL23 as a causally proven BA improvement. **[code][limitation]**

The hard-case result is asymmetric. CL23 12k materially improves Crying to a
provisional 8 pass / 0 minor / 0 fail. Skiing improves modestly to 5 pass / 1
minor / 2 fail, but is not solved: Jisoo and Lex retain severe fused/nested
eyewear. Its current Skiing mean is `0.367513`, below CL19 24k (`0.379294`)
and far below PhotoMaker (`0.464005`). **[measured][visual]**

The three recommended successors are therefore:

| Priority | Run | Only critical change from CL23 | Main objective | Initial output |
|---:|---|---|---|---|
| **1** | **CL27 frequency-surface energy shaping** | Training-only top-object suppression plus visible-face floor on CL23's routed frequency message | Fix Skiing/hair/hand ownership without changing inference | Exact CL23 |
| **2** | **CL28 bounded learnable frequency endpoints** | Zero-initialized per-processor corrections around three CL23 schedule endpoints | Improve aggregate ID while allowing different U-Net layers to specialize | Exact CL23 |
| **3** | **CL29 low-band causal contrastive loss** | Same-ID positive versus wrong-ID negative supervision on branch-local low-band messages | Strengthen spatial-reference causality and stable identity without ArcFace reward | Exact CL23 |

These are deliberately not another global ArcFace reward, another direct
three-state router, or another boundary-distillation arm. CL25, CL22, and CL24
provide current negative evidence for those literal repetitions.

# 1. Evidence integrity and current run ledger

## 1.1 Immutable runs and live status

The table step is the latest complete 96-row per-image asset at the cutoff.
“Train log” is only a progress indicator and is not used as a validation
result. **[measured]**

| Run | Complete validation | Latest train log | State |
|---|---:|---:|---|
| CL21 r2 | 10k | 11.05k | Running |
| CL22 r2 | 10k | 11.60k | Running |
| CL23 r1 | **12k** | 12.05k | Running |
| CL24 r1 | 14k | 15.90k | Running |
| CL25 r2 | 4k | 3.95k | Completed |
| CL26 r3 | 10k | 10.55k | Running |
| CL19 | 24k | 24k | Completed |
| PM0 | 0 | 0 | Completed |

Immutable record map:

- CL21 r2 — Comet `6670db89c44a489388b8f09b91423b0d`; Serv
  `lm-mpi-job-fba7a7ca-ce8f-4b65-a7e5-f139cb3187af`.
- CL22 r2 — Comet `b181feb6c54644e69fb7e8709a59f32e`; Serv
  `lm-mpi-job-84855e01-da1a-4066-b2b3-e71d4904f66e`.
- CL23 r1 — Comet `a9ec9c59d1624c68acb98737dcd65298`; Serv
  `lm-mpi-job-f9160c9d-2b18-401d-844c-1e1116f17c3e`.
- CL24 r1 — Comet `a18e22ae9f0e4a24b6252f6b392fab62`; Serv
  `lm-mpi-job-caae3dad-99ab-40ac-80f2-6ebb106f813a`.
- CL25 r2 — Comet `120b72df8134474ca094e6162d085eb0`; Serv
  `lm-mpi-job-893096da-e633-40cc-9a28-cde68fd4e813`.
- CL26 r3 — Comet `e9c0a9b505f041a68a183ca3cb4ca0af`; Serv
  `lm-mpi-job-e07a2b02-6f5b-4ad8-bf80-e1f36c24cd4b`.
- CL19 — Comet `cfeda7b55c174b3c83e8d40537ebb6dd`; completed control.
- PM0 — Comet `74efd227d3f8488a98e83d815c77c07c`; controlled inference baseline.

No running job was stopped or modified for this review.

## 1.2 Joining and measurement checks

Each selected validation table contains exactly 96 unique rows with
`image_index=0..95`. Tables were joined only by `image_index`; display names
were not used as keys. Space/underscore normalization was used only to find
the corresponding image assets. The bootstrap uses 100,000 resamples with
seed `20260814`. **[method]**

CL23 12k has zero `no_face`, zero `unowned`, and zero `ambiguous` subject-v2
rows. Its higher ID_SIM therefore does not come from a detector fallback or a
change in subject ownership. **[measured]**

The ongoing runs have not yet executed their deferred, finalized PyIQA face-
quality pass. Face-quality comparisons are therefore not fabricated here.
CL25 and the completed baselines have finalized curves; other quality columns
remain pending by design. **[limitation]**

The sealed numerical files, per-image tables, visual sheets, manifest, and
hashes are in the
[`sealed snapshot directory`](assets/cl21_cl26_20260814_current/).

# 2. Current quantitative result

## 2.1 Latest complete endpoint per run

These endpoints occur at different steps and are a status overview, not a
matched-step ranking. **[measured]**

| Run | Step | ID_SIM | Legacy ID | Mask IoU | Face count | Text sim |
|---|---:|---:|---:|---:|---:|---:|
| PhotoMaker | 0 | **0.556580** | **0.501431** | 0.86515 | 1.135 | 26.0015 |
| CL19 | 24k | 0.506823 | 0.450748 | 0.91419 | 1.156 | 26.3706 |
| CL21 | 10k | 0.498801 | 0.448147 | 0.91104 | 1.125 | 26.5343 |
| CL22 | 10k | 0.489460 | 0.438465 | 0.87913 | 1.083 | **25.0934** |
| **CL23** | **12k** | **0.518674** | **0.457160** | **0.92197** | 1.115 | 26.2713 |
| CL24 | 14k | 0.489801 | 0.436630 | 0.90905 | 1.156 | 26.3696 |
| CL25 | 4k | 0.493873 | 0.438612 | 0.91737 | 1.125 | 26.4814 |
| CL26 | 10k | 0.479096 | 0.424117 | 0.91363 | 1.156 | 26.4925 |

CL22's raw Skiing score should not be read without its `-1.42` to `-1.44`
text-sim deficit versus the strong controls and visible Jisoo/Marion failures.
It is not a promotion candidate. **[measured][visual]**

## 2.2 Matched CL23 versus CL19

| Step | CL23 ID | CL19 ID | Paired delta | Wins | 95% bootstrap interval |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.465001 | 0.437661 | +0.027340 | 68/96 | [+0.011250,+0.042895] |
| 2k | 0.488368 | 0.420160 | **+0.068208** | 79/96 | [+0.053304,+0.083246] |
| 4k | 0.516760 | 0.469410 | +0.047350 | 80/96 | [+0.036829,+0.057949] |
| 6k | 0.515798 | 0.471274 | +0.044524 | 80/96 | [+0.034234,+0.054766] |
| 8k | 0.513434 | 0.459939 | +0.053495 | 81/96 | [+0.041703,+0.065786] |
| 10k | **0.525002** | 0.479404 | **+0.045598** | 76/96 | **[+0.034060,+0.057365]** |
| 12k | **0.518674** | 0.480898 | **+0.037776** | 73/96 | **[+0.026193,+0.049563]** |

The difference-in-differences from initialization to 12k is approximately
`+0.010436`: CL23 itself gains `+0.053673`, while CL19 gains `+0.043237` over
the same interval. This still separates a learned advantage from the already
strong CL23 initialization effect, although 12k is below CL23's 10k peak.
CL23 text also falls from `26.5112` at 10k to `26.2713` at 12k, about `0.200`
below matched CL19; this is a real tradeoff to monitor. **[measured]**

CL23 12k also beats CL19 24k by `+0.011851`, 58/96 wins, interval
`[+0.001651,+0.022047]`. It remains below PhotoMaker by `-0.037906`, only
27/96 wins, interval `[-0.054756,-0.022044]`. **[paired]**

![](assets/cl21_cl26_20260814_current/id_trajectories_current.png){ width=78% }

*Figure 1. Current ID trajectories.*

## 2.3 What the other arms currently say

| Arm | Current evidence | Decision at this cutoff |
|---|---|---|
| CL21 residual identity-token CA | `0.498801` at 10k; matched delta already `-0.015070` at 4k with a negative interval | Corrected residual CA is mechanically valid but does not add to the strong CL19/CL23 SA route |
| CL22 direct visibility router | Brief `+0.023790` at 2k, neutral by 4k; text `25.0934`; hard-case artifacts remain | Do not repeat a dense head that directly owns all top/visible/background blending |
| CL23 temporal-frequency route | Positive at every matched gate; current best non-PM endpoint | Promote as the architectural base, subject to causal probes |
| CL24 PM boundary distillation | `0.489801` at 14k; Skiing remains broken | Sparse epsilon-space teacher correction did not transfer topology reliably |
| CL25 low-noise ArcFace continuation | Warm starts at CL19 final `0.506823`, falls to `0.484385`, ends `0.493873` | Do not add another raw/global ArcFace reward; modest face-quality gains do not offset ID loss |
| CL26 anchored high-resolution ROI | `0.479096` at 10k; matched-neutral | More guaranteed high-resolution reference capacity alone is insufficient |

CL25's step zero is intentionally **not** a cold-start comparison: it loads the
pinned CL19 24k checkpoint (SHA-256 beginning `707cff...`). That is why its
step-zero ID is exactly CL19's final value. CL23 has no such continuation.
**[config]**

# 3. Hard cases

## 3.1 Quantitative slices

| Prompt | PhotoMaker 0 | CL19 24k | CL23 12k | CL23 − CL19 final | CL23 − PM0 |
|---|---:|---:|---:|---:|---:|
| Skiing | **0.464005** | 0.379294 | 0.367513 | -0.011781 | **-0.096492** |
| Crying | **0.599982** | 0.556172 | 0.546830 | -0.009342 | -0.053152 |

At the matched 12k gate, CL23 is better than CL19 on both slices: Skiing
`0.367513` versus `0.319808`, and Crying `0.546830` versus `0.512194`.
Nevertheless, CL19's later training recovers some Skiing ID and the visual
topology failure remains. **[measured]**

The CL23 gain is broad but not universal. Versus CL19 final it improves
Kickboxing `+0.054336`, Laughing `+0.042624`, Rushing `+0.028362`, Reading
`+0.024392`, and several other prompts; Skiing, Drumming, Dancing, and Crying
are negative at the current gate.
Across identities it improves seven of eight, but Marion declines
`-0.030401`. CL23 currently beats PhotoMaker only for Eddie; the largest
remaining identity gaps to PhotoMaker are Jisoo (`-0.098437`), Jennie
(`-0.096119`), and Marion (`-0.081220`). **[measured]**

## 3.2 Visual topology

The montage below uses fixed validation crops and prints each subject-v2 score
inside the tile. It is not cherry-picked: it includes all eight identities for
both requested hard prompts. **[visual]**

| Run | Skiing pass / minor / fail | Crying pass / minor / fail |
|---|---:|---:|
| PhotoMaker | 8 / 0 / 0 | 8 / 0 / 0 |
| CL19 24k | 4 / 1 / 3 | 6 / 1 / 1 |
| CL23 12k | **5 / 1 / 2** | **8 / 0 / 0** |

The rubric checks object presence, layer order, and whether the intended face
remains readable; it does not reward deleting goggles or hands. The labels are
one-reviewer, unblinded, and one-seed, so they are provisional rather than a
human-study estimate. Exact labels are in
[`visual_review.csv`](assets/cl21_cl26_20260814_current/visual_review.csv).

PhotoMaker gives the desired Skiing structure: a large ski-goggle layer is
clearly above the forehead/eyes, while ordinary orange glasses can remain on
the face. CL23 12k now solves that ordering for Elon and retains it for several
other identities, but Jisoo remains catastrophic and Lex retains nested
fragments; Eddie is asymmetric. CL23's current Crying set is clean across all
eight identities. **[visual]**

No CL21, CL22, CL24, CL25, or CL26 sheet shows a convincing general Skiing
topology breakthrough. CL22 produces a few cleaner single-goggle examples and
the highest new-run Skiing ID mean, but Jisoo and Marion remain broken and the
global prompt score collapses. **[visual][measured]**

![](assets/cl21_cl26_20260814_current/hardcases_pm0_cl19_cl23_face_comparison.jpg){ width=100% }

*Figure 2. PhotoMaker, CL19, and CL23 hard-case comparison.*

# 4. Is CL23 fair, or is step-zero ID biased by PhotoMaker/leakage?

## 4.1 Short answer

**Fair for the intended PhotoMaker+BA comparison: yes. Proven BA-only causal
gain: not yet.**

CL23 does not load a trained experiment checkpoint, does not optimize ID_SIM,
does not change the validation references or metric, and does not feed the
ground-truth target image during validation. Its step-zero output is different
because the architecture intentionally routes a nonzero spatial-reference
message differently before any optimizer update. **[config][code]**

## 4.2 What is and is not shared with PhotoMaker

| Component | CL19 | CL23 | Interpretation |
|---|---|---|---|
| RealVisXL/SDXL base | Same | Same | Controlled |
| Pretrained PhotoMaker V2 identity tokens/default adapter | Same | Same | Required project base, not a CL23-only shortcut |
| Generic effective adapter scope | `effective_all` | `effective_all` | Controlled |
| PhotoMaker-default adapter scope | `effective_all` | `effective_all` | Controlled |
| Explicit spatial BA self-attention | Enabled | Enabled | Core project mechanism |
| Branched cross-attention | Disabled | Disabled | No hidden CA-only identity injector |
| BA Q/K/V ownership | target Q; native target K/V; reference-face K/V | Same | Controlled |
| Face merge | CL19 soft spatial router | temporal low/high frequency schedule | **Only architectural delta** |
| Pose adaptation | 0 | 0 | Reference K/V are not replaced by target features |
| `ca_mixing_for_face` | false | false | Required invariant |
| Experiment checkpoint | none | none | Cold-start control |

The live CL23 trainable contract is 2,240 tensors / 219,217,920 parameters:
840 branched-SA tensors / 127,795,200 parameters, 700 generic-adapter tensors /
30,474,240 parameters, and 700 PhotoMaker-default tensors / 60,948,480
parameters. BA therefore holds about 58.3% of the trainable parameters, but it
is not the only trainable subsystem. **[config]**

## 4.3 Why step zero is already high

The branch projection constructor copies the effective base projection,
including the active adapter delta, into a frozen base buffer. The new branch
LoRA has random `A` but zero `B`, so its trainable residual is exactly zero at
initialization. See
[`attn_processor_cleanest.py`](../src/model/photomaker_branched/attn_processor_cleanest.py),
lines 12–45 and 48–91. **[code]**

The processor then computes:

```text
q          = target-query projection(target)
native     = Attention(q, target K, target V)
reference  = Attention(q, masked reference-face K, masked reference-face V)
delta      = reference - native
low, high  = GaussianSplit(delta)
output     = native + soft_face_router * (s_low(p)*low + s_high(p)*high)
```

The target/reference ownership is explicit in lines 459–487; the frequency
split and scheduled merge are in lines 608–629 and 840–865. The runtime forms
the doubled `[target, reference]` batch and computes denoising progress from
the actual scheduler timestep in
[`branched_runtime.py`](../src/model/photomaker_branched/branched_runtime.py),
lines 1101–1128. **[code]**

CL19 and CL23 have the same cloned projections. CL23's higher step zero comes
from applying scales `(low .50→.85, high .75→1.25)` to all seven U-Net groups,
not from loading later weights. Its paired step-zero advantage is therefore a
real initialization-level effect of the routing hypothesis. **[code][paired]**

## 4.4 Leakage taxonomy

| Possible leakage/bias | Finding | Consequence |
|---|---|---|
| Prior CL19/CL14 checkpoint | **Absent in CL23** | No checkpoint leakage |
| Ground-truth target image at validation | **Absent** | No target-pixel leakage |
| Direct subject-v2/ArcFace loss in CL23 | **Absent** | No direct optimization of the reported metric |
| PhotoMaker tokens and spatial reference depict the scored identity | **Present by design in every personalization arm** | ID_SIM measures fidelity to the supplied identity, not unseen-exemplar generalization |
| Same reference family conditions generation and defines the identity centroid | **Present and shared** | Copying reference appearance/accessories can inflate ID while hurting editability; hard-case visuals and cross-view controls remain necessary |
| Co-trained generic and PhotoMaker adapters | **Present and shared with CL19** | Matched CL23−CL19 isolates the route, but absolute learned gain is not BA-only |
| Missing held-out spatial-reference ablation | **Present limitation** | BA causality remains provisional despite active telemetry |

The current evidence argues against a trivial metric exploit: PhotoMaker still
scores significantly higher, CL23 fails the most reference-copy-sensitive
Skiing topology, and subject-v2 ownership has no fallback rows. It does not
eliminate subtler reference-copy bias. **[inference]**

Training telemetry also shows that the branch is active rather than numerically
dead: both low- and high-band reference-minus-native RMS are nonzero, the
merged/native RMS ratio differs from one, and BA gradients are nonzero. That
is necessary evidence, not a held-out causal counterfactual. **[measured]**

## 4.5 Mandatory no-training causal panel

Before launching or promoting successors, evaluate one exact CL23 checkpoint
(10k and 12k are already available; repeat at its final selected checkpoint) on the
same 96 items under four inference-only arms:

| Arm | PhotoMaker ID tokens | Spatial reference K/V | CL23 routed delta | Purpose |
|---|---|---|---|---|
| `normal` | correct | correct | normal | Production output |
| `native_endpoint` | correct | correct | forced exactly zero | What the co-trained PhotoMaker+generic path can do without spatial BA contribution |
| `zero_spatial` | correct | zero/masked | normal code path | Dependence on spatial-reference content |
| `shuffled_spatial` | correct | deterministic different identity | normal code path | Identity specificity of the spatial branch |

Record checkpoint SHA-256, output hashes, the full per-image subject-v2 table,
text/face-quality metrics, and Skiing/Crying sheets. `normal` should beat
`native_endpoint` and `shuffled_spatial` in paired ID with no material text
loss; `zero_spatial` and `shuffled_spatial` must not be pixel-identical to
normal. Keep PhotoMaker tokens correct in all four arms so only the spatial BA
lane changes. **[design]**

If normal does not beat both spatial controls, CL23 remains a fair architecture
result but cannot be claimed as a causally proven learned BA gain. Do not hide
that failure by freezing or removing PhotoMaker in a new training run.

# 5. Prior experiments and literature: constraints on the next step

## 5.1 Project evidence that should not be repeated literally

- **CL9 precise occluder masks:** some ID improvement, but only 4/7 Skiing
  topology successes. Binary geometry alone does not encode semantic layer
  order.
- **CL17 semantic gate:** label BCE converged while routed contribution stayed
  below about 0.85% of native and quality regressed. A head can win its own
  label loss while the denoiser ignores it.
- **CL18 cross-view prediction consistency:** neutral/negative and the
  consistency loss became tiny. Equality of final predictions can teach
  reference indifference.
- **CL22 direct three-state router:** active route, transient ID, large text
  regression, and remaining topology failures. Do not let a dense learned head
  replace all CL23 routing.
- **CL24 PM boundary distillation:** neutral/negative endpoint and Skiing not
  fixed. Epsilon-space teacher loss at sparse boundaries is insufficient.
- **E22 / older decoded-x0 ArcFace:** ID declined despite direct reward. Raw
  identity reward can conflict with denoising and semantics.
- **CL25 low-noise multi-step ArcFace + anchor:** warm-start ID
  `0.506823→0.493873`. Even improved low-noise reward is not currently a
  high-probability repeat.
- **CL26 anchored ROI:** guaranteed high-resolution contribution without a
  current ID gain. Capacity/resolution without ownership or objective alignment
  is not enough.
- **CL20 BigCelebs curriculum:** high text, CL14-level identity, and no recovery
  after Cosmic re-anchoring. Do not mix BigCelebs into these three mechanism
  tests.

## 5.2 Primary-paper transfer

The local source archives are
[`2026-08-11_cl14_architecture_review`](sources/2026-08-11_cl14_architecture_review/),
[`2026-08-13_cl19_architecture_review`](sources/2026-08-13_cl19_architecture_review/),
and the 14 August refresh
[`2026-08-14_cl23_followup`](sources/2026-08-14_cl23_followup/). The refresh
searched arXiv and official CVF proceedings through 14 August 2026 and saved
two additional primary PDFs locally. The newest directly relevant large-
occlusion paper remains ReSem-Face (5 August 2026). **[external]**

| Source | Useful idea | Transfer to this plan | What is not copied |
|---|---|---|---|
| [TFCustom](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_TFCustom_Customized_Image_Generation_with_Time-Aware_Frequency_Feature_Guidance_CVPR_2025_paper.pdf) | Time/frequency-specific reference control | CL23 validates the principle; CL28 lets layers tune bounded endpoints | A new ReferenceNet or unconstrained frequency injection |
| [SpatialID](https://arxiv.org/abs/2602.13994) | Identity strength should be spatially and temporally query-relevant | CL27 shapes the existing target-query message on top versus visible pixels | A training-free mask extractor replacing BA |
| [ReSem-Face](https://arxiv.org/abs/2608.04820) | Separate identity semantic prior from the occluded scene stream | Native target owns top objects; identity message remains strong on visible face | Cascaded face inpainting or explicit hole masks at inference |
| [PositionIC](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_PositionIC_Unified_Position_and_Identity_Consistency_for_Image_Customization_CVPR_2026_paper.html) | Visibility-aware attention decouples identity from layout | Top-object versus visible-surface ownership objective | Multi-subject NeRF-style controller |
| [AnyPhoto](https://arxiv.org/abs/2603.14770) | Identity-isolated attention plus reference degradation reduces copy-paste | Synthetic occluders and branch-local objectives | Strong aligned modulation or another global ArcFace loss |
| [GroupPortrait](https://openaccess.thecvf.com/content/WACV2026/papers/Huang_GroupPortrait_Multi-ID_Portrait_Generation_with_High_Identity_Preservation_and_Fine-Grained_WACV_2026_paper.pdf) | Latent region-aware identity feedback can be cheaper than decoded-image reward | CL29 supervises an internal identity-bearing branch representation | Its recognizer head or multi-person architecture |
| [Latent-Identity Tuning](https://arxiv.org/abs/2607.11885) | Identity tokens/subspaces have localized semantic roles | CL28 uses layer/band-specific strengths instead of one global scalar | Test-time identity editing |
| [PatchDPO](https://arxiv.org/abs/2412.03177) | Local preference is safer than whole-image reward | CL27's objective is local to top/visible face regions | A costly generated preference dataset |
| [FairHuman](https://arxiv.org/abs/2507.02714) | Regional gradients can conflict | Visible reference floor prevents the local top loss from collapsing identity everywhere | A multi-task human generator or broad gradient rewrite |
| [PuLID](https://arxiv.org/abs/2404.16022) | Identity improvement needs semantic/layout preservation | Every arm retains the native path, text gates, and fixed validation | Replacing PhotoMaker/BA with PuLID |

The synthesis is narrower than “add a stronger ID loss.” CL23 already shows
that *when and where* a spatial identity message is injected matters. The next
step should make that successful message more selective, then more causal,
without granting a recognizer direct control over the whole image.

# 6. Three successor experiments

All arms are cold-start 24k runs on the unchanged Cosmic manifest. They inherit
CL23, retain its fixed schedule unless the experiment explicitly changes that
schedule, and keep `disable_branched_ca=true`, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`. Validation stays exactly every 2,000 optimizer
steps on the fixed 96 images. **[design]**

## Priority 1 — CL27: frequency-surface energy shaping

**Run:** `CL27_cosmic_frequency_surface_energy_24k_full96_r1`  
**Config:** `CL27_cosmic_frequency_surface_energy_24k`  
**Only critical change:** add a training-only objective on CL23's existing
frequency message; inference architecture and parameters are unchanged.

### Hypothesis

CL23's `reference_out-native_out` is useful on visible facial surface but
harmful where a prompt-created object lies above that surface. Instead of
predicting a new router (CL22), teach the existing target query and reference
K/V projections to produce low reference energy for top-object queries while
maintaining a nonzero identity message on visible face queries. **[hypothesis]**

For synthetic-occlusion samples and only `up_blocks.0`/`up_blocks.1`, downsample
the existing `ba_occluder_mask` to the attention grid. Let `M_top` be top-object
pixels and `M_vis = face*(1-M_top)`. With CL23's low/high components:

```text
L_top = RMS(M_top * high_delta)^2
        + 0.25 * RMS(M_top * low_delta)^2

r_vis = RMS(M_vis * routed_delta) / stopgrad(RMS(M_vis * native_out) + eps)
L_floor = relu(0.35 - r_vis)^2

L_surface = 0.02 * L_top + 0.005 * L_floor
```

Compute per sample, exclude zero-mask samples from both denominators, then
average across eligible processors. The visible floor is essential: it blocks
the CL17/CL18 escape route of reducing reference dependence everywhere.

### Why this is higher probability than CL22/CL24

- It starts and validates exactly as CL23; no new inference head can collapse
  text or take over background routing.
- It uses CL22's already versioned deterministic synthetic labels but changes
  what is supervised: the actual reference message, not an auxiliary class
  predictor.
- It targets the observed failure at the level where it occurs—query-specific
  reference energy—without forcing a PhotoMaker epsilon teacher to match the
  whole denoising target.

### Implementation

1. In `attn_processor_cleanest.py`, expose the graph-bearing `low`, `high`,
   `routed_delta`, `native_out`, face mask, and ownership mask only when the
   CL27 flag is enabled. Do not retain graphs for telemetry-only steps.
2. Add a `frequency_surface_aux_loss()` getter and aggregate selected
   processors in `lora2_helpers.py` analogously to
   `collect_hardcase_aux_loss`, but with per-sample eligible-mask handling.
3. Add the weighted term once in `lora2.py`; do not sum the same region once
   per telemetry read.
4. Reuse `semantic_occlusion_probability=0.25`, seed `150017`, and the existing
   `ba_occluder_mask` plumbing. Do not change real validation inputs.
5. In inference or when disabled, do not create the loss tensors or change
   `target_out`.

Expected ownership remains exactly `2240 / 219217920`.

**Hydra blueprint:**
[`01_CL27_frequency_surface_energy.blueprint.yaml`](blueprints/2026-08-14_cl23_next_three/01_CL27_frequency_surface_energy.blueprint.yaml)

```yaml
defaults:
  - CL23_cosmic_temporal_frequency_router_24k
  - _self_
model:
  ba_frequency_surface_loss_enabled: true
  ba_frequency_surface_loss_groups: [up_blocks.0, up_blocks.1]
  ba_frequency_surface_top_weight: 0.02
  ba_frequency_surface_top_low_band_factor: 0.25
  ba_frequency_surface_visible_floor_weight: 0.005
  ba_frequency_surface_visible_floor_ratio: 0.35
datasets:
  train:
    cosmic_large_adapted:
      semantic_occlusion_probability: 0.25
      semantic_occlusion_seed: 150017
```

### Success gate

At 4k, ID must be noninferior to CL23 within `-0.005`, text within `-0.15`, and
Skiing must improve to at least 6 pass / at most 1 fail without deleting the
goggles. Promotion at 10k requires overall ID at least CL23 matched and a
positive paired Skiing delta. Crying must remain at least 7 pass / 1 minor.

## Priority 2 — CL28: bounded learnable frequency endpoints

**Run:** `CL28_cosmic_learnable_frequency_schedule_24k_full96_r1`  
**Config:** `CL28_cosmic_learnable_frequency_schedule_24k`  
**Only critical change:** replace three fixed CL23 schedule endpoints with
zero-initialized, tightly bounded per-processor corrections.

### Hypothesis

CL23 proves that the fixed frequency schedule is directionally right, but the
same four scalars are imposed on all 70 self-attention sites. Early/down blocks
should emphasize structure differently from late/up blocks; the current Skiing
failure may also benefit from reducing late high-frequency identity injection
at selected up sites while retaining it elsewhere. **[hypothesis]**

For each temporal-frequency processor, add one trainable vector
`schedule_raw[3]`, initialized to zero:

```text
low_early = 0.50                                  # fixed
low_late  = 0.85 + 0.15*tanh(schedule_raw[0])     # [0.70,1.00]
high_early= 0.75 + 0.15*tanh(schedule_raw[1])     # [0.60,0.90]
high_late = 1.25 + 0.15*tanh(schedule_raw[2])     # [1.10,1.40]
L_anchor  = 1e-4 * mean(schedule_raw^2)
```

The zero vector reproduces CL23 exactly, while the ranges prevent a learned
native-only escape or an uncontrolled high-frequency blow-up.

### Implementation

1. Register `schedule_raw` only on CL28 processors and interpolate the derived
   endpoints using CL23's existing real denoising progress.
2. Include each vector once in trainable/checkpoint ownership and add the small
   anchor loss through the existing auxiliary-loss aggregation.
3. Log actual low/high scales by semantic U-Net group, plus raw-vector p10,
   mean, and p90. Fail if any configured group lacks a parameter.
4. Preserve the old fixed-scalar code path byte-for-byte behind the default
   `false` toggle.

There are 70 CL23 SA processors, so the predicted contract is 2,310 tensors /
219,218,130 parameters: one three-scalar tensor per processor. The
implementation validator must derive and pin the actual count rather than
trusting this prediction.

**Hydra blueprint:**
[`02_CL28_learnable_frequency_schedule.blueprint.yaml`](blueprints/2026-08-14_cl23_next_three/02_CL28_learnable_frequency_schedule.blueprint.yaml)

```yaml
defaults:
  - CL23_cosmic_temporal_frequency_router_24k
  - _self_
model:
  ba_frequency_learnable_schedule_enabled: true
  ba_frequency_learnable_low_early: false
  ba_frequency_low_late_center: 0.85
  ba_frequency_low_late_half_range: 0.15
  ba_frequency_high_early_center: 0.75
  ba_frequency_high_early_half_range: 0.15
  ba_frequency_high_late_center: 1.25
  ba_frequency_high_late_half_range: 0.15
  ba_frequency_schedule_anchor_weight: 0.0001
```

### Success gate

Step-zero images must be byte-identical to CL23 under the same source/runtime.
At 4k and 10k the paired overall ID interval versus CL23 should be positive, or
the mean gain should exceed `+0.005` with no text/mask-IoU regression while a
later gate confirms it. The learned schedule must not sit at every bound.
Inspect whether up0/up1 high-late scales fall specifically on hard examples;
do not claim causality from average scalars alone.

## Priority 3 — CL29: low-band causal contrastive loss

**Run:** `CL29_cosmic_lowband_causal_contrastive_24k_full96_r1`  
**Config:** `CL29_cosmic_lowband_causal_contrastive_24k`  
**Only critical change:** add a branch-local contrastive loss using two
same-identity references as positives and an in-batch different-identity
reference as a negative.

### Hypothesis

CL18 failed because it asked two references to produce the same final denoiser
prediction; ignoring spatial-reference variation is an easy solution. CL29
instead supervises only the *low-frequency reference-minus-native message*,
where stable face structure should live, and includes an explicit wrong-ID
negative. High-frequency view/accessory variation and the final prediction are
left free. **[hypothesis]**

On 12.5% of eligible batches, reuse CL18's alternate same-ID reference
plumbing with the same noisy target, timestep, PhotoMaker tokens, and target
query. At `mid`, `up0`, and `up1`, pool the visible-face low-band message and
L2-normalize it. Use a detached target query for this auxiliary path so the
loss updates reference K/V/output projections rather than encoding the target
image in Q. For each sample, compute a wrong-reference message by permuting
reference K/V only among different `identity_id` values under the same query.

```text
z1 = norm(pool_visible(low_delta(target_q_detached, same_id_ref_1)))
z2 = norm(pool_visible(low_delta(target_q_detached, same_id_ref_2)))
zw = norm(pool_visible(low_delta(target_q_detached, wrong_id_ref)))

L_contrast = -log exp(cos(z1,z2)/0.10)
                  / (exp(cos(z1,z2)/0.10) + exp(cos(z1,zw)/0.10))
```

Ramp the total weight from zero at 2k to `0.02` at 6k. Skip, do not silently
self-negative, when a batch lacks a different identity.

### Implementation

1. Reuse `same_identity_dual_reference=true` and
   `min_reference_candidates_for_target=3`; assert distinct reference paths.
2. Add graph-bearing low-band capture to only the selected processors. The
   normal forward and alternate forward must use the same target/noise/timestep.
3. Compute wrong-reference K/V at the processor under the same detached query;
   do not use a second target or shuffle PhotoMaker tokens.
4. Average per valid sample, then per processor/group. Log positive cosine,
   wrong cosine, margin, application fraction, and same-identity skip fraction.
5. Do not add ArcFace, subject-v2, final-prediction consistency, or new
   trainable parameters.

Expected ownership remains `2240 / 219217920`. The sampled second full forward
adds roughly 12.5% model-forward work before attention overhead; profile the
real cost during smoke testing.

**Hydra blueprint:**
[`03_CL29_lowband_causal_contrastive.blueprint.yaml`](blueprints/2026-08-14_cl23_next_three/03_CL29_lowband_causal_contrastive.blueprint.yaml)

```yaml
defaults:
  - CL23_cosmic_temporal_frequency_router_24k
  - _self_
model:
  ba_frequency_lowband_contrastive_enabled: true
  ba_frequency_lowband_contrastive_groups: [mid_block, up_blocks.0, up_blocks.1]
  ba_frequency_lowband_contrastive_probability: 0.125
  ba_frequency_lowband_contrastive_weight: 0.02
  ba_frequency_lowband_contrastive_temperature: 0.10
  ba_frequency_lowband_contrastive_ramp_start_step: 2000
  ba_frequency_lowband_contrastive_ramp_end_step: 6000
  ba_frequency_lowband_contrastive_detach_target_query: true
  ba_frequency_lowband_contrastive_negative_mode: in_batch_different_identity
datasets:
  train:
    cosmic_large_adapted:
      same_identity_dual_reference: true
      min_reference_candidates_for_target: 3
```

### Success gate

Require positive correct-versus-wrong message margin, noncollapsed embedding
variance, and a positive or at least noninferior paired ID result versus CL23
at 4k. At 10k, promote only if aggregate ID improves and Marion/Jisoo do not
regress. A larger contrastive margin without output ID improvement is a failed
surrogate, not a success.

# 7. Implementation and Serv handoff

## 7.1 Code map

The implementation agent should make localized, toggle-gated changes:

- `attn_processor_cleanest.py`: CL27 graph loss; CL28 bounded endpoint vector;
  CL29 low-band auxiliary capture/wrong-KV message; defaults preserve CL23.
- `branched_runtime.py`: reuse masks and real denoising progress; do not change
  doubled target/reference ownership.
- `lora2.py`: register and validate new flags, aggregate each auxiliary term
  exactly once, and reuse the CL18 alternate-reference forward for CL29.
- `lora2_helpers.py`: collect graph-bearing auxiliary losses separately from
  detached telemetry and preserve exact optimizer ownership.
- `cosmic_large_adapted.py`: reuse existing synthetic-mask and dual-reference
  fields; add only fail-closed distinct-ID/path assertions if needed.
- `src/configs/model/photomaker_branched_lora2.yaml`: backward-compatible
  defaults with all new toggles false.
- `src/configs/CL27_*.yaml` through `CL29_*.yaml`: copy the reviewed blueprints
  and pin final ownership contracts.
- `tools/validate_CL27_CL29_config.py`: compose each config and compare its
  resolved diff to CL23 allowlists.
- `experiments/cosmic_large/*.json`: three immutable cold-start 24k specs on
  the exact fixed validation contract.
- `launchers/active/run_CL27_CL29_cl23_followups_1gpu.sh`: reject Hydra
  overrides; verify sealed hashes and the Comet record; finalize face quality.
- `serv_run_packages/...`: per-run source snapshot, manifest, start wrapper,
  and MLS YAML.

Add one dated architecture comment and a sparse `AICODE-NOTE` where graph
capture could accidentally retain all U-Net activations. Do not refactor the
working CL23 path.

## 7.2 Fail-closed preflight

Before any Serv submission:

1. Compose CL23 and each successor; permit only the fields named in its
   blueprint. Assert identical training dataset path/manifest, optimizer,
   scheduler, prompts, validation assets, seeds, 24k steps, and `epoch_len=2000`.
2. Assert `trainer.from_pretrained=null`, `trainer.resume_from=null`, and no
   saved checkpoint for all three.
3. Assert branched SA enabled, branched CA disabled, all seven CL23 groups,
   `pose_adapt_ratio=0`, and `ca_mixing_for_face=false` in training and
   validation.
4. Run an old-mode/CL23 forward and a new-mode forward with all toggles off;
   require exact equality. Require exact CL23 step-zero output for CL27/CL29
   and zero-initialized CL28.
5. Verify CL27 loss is finite/nonzero only on eligible synthetic masks and
   receives gradients in BA reference projections; verify visible-floor
   gradients oppose global collapse.
6. Verify CL28 has exactly one `[3]` vector per selected processor, all zeros,
   correct bounds, checkpoint round-trip, and actual ownership pinned.
7. Verify CL29 positives use distinct same-ID files, negatives use a different
   `identity_id`, target Q is detached only for the auxiliary path, and the
   production forward remains unchanged.
8. Run `py_compile`, Hydra composition, shell syntax, source-manifest verify,
   dataset preflight, one-update forward/backward, optimizer membership, and
   schema-v2 checkpoint round-trip.
9. During startup, require
   `saved/<run_name>/comet_experiment.json` with a 32-character immutable key
   before allowing the process to continue unattended.

## 7.3 Experiment JSON and launcher contract

Copy the CL23 experiment spec once per run. Change only run/config/comment and
the expected trainable contract. Preserve the exact Cosmic manifest SHA,
PhotoMaker V2 checkpoint, subject-v2 embedding SHA, validation file hashes,
batch size, seed, optimizer, and 12 epochs × 2,000 steps.

Use a new launcher instead of widening the old CL21–CL26 allowlist. The start
wrapper should mirror the verified CL23 wrapper, point to
`runtime_sources_cl27_cl29_v1/<RUN_ID>`, and map exactly the three run IDs to
their config names. Do not include the CL25 checkpoint/ArcFace environment
branch.

## 7.4 Serv MLS YAMLs

The complete design-only one-A100 YAMLs are:

- [`run_CL27_cosmic_frequency_surface_energy_24k_full96_r1_1gpu.yaml`](blueprints/2026-08-14_cl23_next_three/serv/run_CL27_cosmic_frequency_surface_energy_24k_full96_r1_1gpu.yaml)
- [`run_CL28_cosmic_learnable_frequency_schedule_24k_full96_r1_1gpu.yaml`](blueprints/2026-08-14_cl23_next_three/serv/run_CL28_cosmic_learnable_frequency_schedule_24k_full96_r1_1gpu.yaml)
- [`run_CL29_cosmic_lowband_causal_contrastive_24k_full96_r1_1gpu.yaml`](blueprints/2026-08-14_cl23_next_three/serv/run_CL29_cosmic_lowband_causal_contrastive_24k_full96_r1_1gpu.yaml)

Each uses the established image and resource block:

```yaml
job:
  environment:
    image: cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36
  resource:
    instance_type: a100.1gpu.8C.243G
    processes: 1
    workers: 1
  type: binary
```

Their scripts point to separate `serv_run_packages/<RUN_ID>/start_*_1gpu.sh`
files and separate stdout/stderr roots. They are intentionally marked design-
only: submission is invalid until the packages and source manifests exist.
Three one-GPU jobs request three A100s total, within the normal project ceiling
of six after the current CL21–CL26 allocations finish or are otherwise freed.
Inspect live running/pending jobs immediately before submission.

# 8. Evaluation and promotion ladder

## 8.1 Fixed gates

Evaluate 0, 2k, 4k, then every 2k through 24k. At every gate require:

- complete 96-row subject-v2 per-image table;
- aggregate and paired ID versus CL23 at the same step;
- text, mask IoU, face count, no-face/unowned/ambiguous counts;
- Skiing and Crying full/crop sheets with the topology rubric;
- finalized seven face-quality curves when the run completes;
- experiment key, source manifest, config digest, and checkpoint SHA.

The primary target is to exceed CL23 with a positive paired interval while
preserving text/geometry. The stretch target remains PhotoMaker `0.556580` on
this exact panel. Do not declare “beats PhotoMaker” from another protocol,
metric version, or selected subset.

## 8.2 Early stop / promotion rules

- Stop promoting an arm if 4k ID is below CL23 by more than `0.01`, text drops
  more than `0.15`, mask IoU drops more than `0.01`, or Skiing object presence
  worsens—even if its auxiliary loss looks good.
- A topology-only CL27 gain with ID within `-0.005` may be retained for a later
  factorial, but is not itself the requested aggregate-ID winner.
- Do not combine arms during their first runs. If CL27 and CL28 independently
  pass, their mechanisms are orthogonal enough for a later controlled
  combination. Combine CL29 only after its output ID, not merely its message
  margin, passes.
- Compare the final selected checkpoint, not necessarily 24k, using a
  predeclared rule: highest complete ID gate with text no worse than CL23−0.15
  and no hard-case visual regression.

## 8.3 Dataset decision

Keep all three on Cosmic. CL20 is direct negative evidence for generic
BigCelebs mixing on a weaker route, and changing architecture plus dataset
would destroy attribution. If one mechanism first passes on Cosmic, a later
fourth data arm can add a small, quality-filtered BigCelebs hard-case stratum
with explicit goggles/hair/hand masks and same-ID distinct references. That is
not one of these three experiments.

# 9. Limitations

- Five runs are incomplete. CL23 had just completed its 12k validation at the
  cutoff; later complete gates are new evidence and must update this report's
  conclusions rather than being silently folded in.
- Visual labels are one reviewer, unblinded, one seed. They are suitable for
  rejecting obvious topology failures, not estimating human preference.
- Ongoing face-quality metrics are deferred, so current CL23 quality is judged
  from fixed images, text/geometry metrics, and completed controls.
- The matched CL23 advantage isolates its route configuration relative to
  CL19, but co-trained PhotoMaker/generic adapters prevent a claim that every
  learned gain resides in BA weights. The proposed causal panel addresses this
  at inference, not with a full training factorial.
- Paper transfers are hypotheses from different backbones/tasks. No cited
  source validates these exact SDXL/PhotoMaker/BA losses.
- CL29 has the greatest compute and implementation risk; its priority is lower
  for that reason.

# 10. Final recommendation

Treat CL23 as the new provisional base. Its gain is statistically clear,
broad across prompts/identities, and mechanically consistent with an active
temporal-frequency target-Q/reference-KV branch. Do not dismiss its high step
zero as checkpoint leakage; do not present its absolute score as BA-only
either.

First run the four-arm no-training causal panel. In parallel, implement CL27,
CL28, and CL29 behind default-off toggles and validate exact CL23 replay. Train
them as isolated cold-start arms in the stated order. CL27 is the highest-
probability direct fix for Skiing topology, CL28 is the cleanest opportunity
for another broad ID gain, and CL29 is the most principled loss experiment
left after the failures of global ArcFace reward and prediction consistency.

## Reproducibility artifacts

Sealed data: [`snapshot manifest`](assets/cl21_cl26_20260814_current/snapshot_manifest.json),
[`selected endpoints`](assets/cl21_cl26_20260814_current/selected_endpoints.csv),
[`metric history`](assets/cl21_cl26_20260814_current/metric_history.csv),
[`paired comparisons`](assets/cl21_cl26_20260814_current/paired_comparisons.csv),
[`slice means`](assets/cl21_cl26_20260814_current/slice_means_selected.csv),
[`per-image table`](assets/cl21_cl26_20260814_current/per_image_selected.csv),
[`hard-case rows`](assets/cl21_cl26_20260814_current/hardcase_rows_selected.csv),
[`visual review`](assets/cl21_cl26_20260814_current/visual_review.csv), and
[`SHA-256 ledger`](assets/cl21_cl26_20260814_current/SHA256SUMS.txt).

Handoff materials: [`CL27–CL29 design blueprints`](blueprints/2026-08-14_cl23_next_three/)
and the [`14 August literature refresh`](sources/2026-08-14_cl23_followup/SOURCES.md).
