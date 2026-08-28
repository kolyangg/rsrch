---
title: "CL39 Branched-Attention Investigation"
subtitle: "Why it resembles PhotoMaker, what branched attention adds, why raw R is fragile, and the highest-potential next experiments"
date: "26 August 2026"
author: "Research report prepared from clean_new code, sealed run evidence, attached audits, and recent literature"
---

# CL39 Branched-Attention Investigation

**Code revision inspected:** `kolyangg/rsrch`, branch `clean_new`, HEAD `728fd28f7c2c73127fa300cd04a62b29a96be918` (22 August 2026, `docs: document branched training execution flow`).

**Primary comparison:** PhotoMaker V2, CL14, CL19, CL23, CL27 and CL39 under the fixed 96-image validation contract, plus the sealed CL39 24k same-checkpoint branch interventions.

**Evidence labels used throughout:**

- **[MEASURED]** comes directly from the attached fixed-panel reports or the same-checkpoint whole-denoising audits.
- **[CODE]** is supported by the inspected `clean_new` implementation at the revision above.
- **[INFERENCE]** is a reasoned interpretation of code and measurements, not a completed experiment.
- **[PROPOSAL]** is a new experiment or implementation design whose result is not yet known.

> **Bottom line.** CL39's similarity to PhotoMaker is largely expected, not a sign that the branch is inactive. The entire system is deliberately PhotoMaker-anchored: the same base model, seed, prompt and schedule are used; PhotoMaker identity conditioning begins before branched attention; branch Q/K/V start from effective PhotoMaker projections; and CL39 always preserves the native target message `N` while confidence suppresses most of the explicit reference correction. The branch nevertheless has a measurable causal effect on the audited subset. The main headroom is not to increase `R` blindly, but to make the reference lane independently coherent during training, reduce its PhotoMaker shortcut, improve correspondence semantics, and calibrate low/high-frequency corrections more intelligently.

![Why CL39 stays close to PhotoMaker](CL39_branched_attention_report_assets_2026-08-26/why_cl39_similar.png)

## Executive answers

| Question | Answer | Confidence |
|---|---|---|
| Why are CL39 images so similar to PhotoMaker? | Because PhotoMaker remains the dominant conditioning and initialization path, BA starts later, `N` remains the anchor, and CL39 retains only a modest confidence-gated residual. There is no loss encouraging distance from PhotoMaker. | High [CODE + MEASURED] |
| Should the images differ more? | They should differ where the explicit spatial reference adds correct identity, hard-case topology or controllability. Global pixel difference is not itself a quality target. A better objective is a Pareto improvement in ID, prompt adherence, topology and copy-paste freedom. | High [INFERENCE] |
| Does BA add anything over PhotoMaker/native attention? | Yes on the selected 16-cell same-checkpoint audit: CL39 is `+0.03570` ID above `N`-only with a positive bootstrap interval, and 93.8% of face-crop pixels change above 1/255. However, the full fixed-96 surplus over original PhotoMaker is not yet cleanly attributable to spatial BA because the trained adapters and PM path are co-adapted. | High for selected-16; moderate for full-96 attribution |
| Is raw `R` problematic? | Yes. Raw `R` is underconstrained and fragile as a standalone route, with duplicated/misregistered eyes, noses, glasses, hands and expression geometry. But raw `R` is an off-operating-point stress test roughly 3.8 times stronger than the actual routed correction. | High [MEASURED] |
| Should `R` be improved? | Yes, primarily to create robustness and identity headroom. The success target is a better actual combined image with positive spatial-reference causality, not merely a prettier raw-`R` panel. | High [INFERENCE] |
| What is the key CL14 lesson? | Not binary query masking and not hard inference replacement. The key lesson is **training ownership**: CL14 forced the reference face path to be usable because it owned the face update. CL39 lets `R` survive as a small residual around `N`. | High [CODE], moderate-high causal hypothesis |
| Highest-priority training experiment | Training-only, globally coherent reference-face ownership (`FOWN-1` / `R2-A`) in up-blocks 0/1, 12.5% of steps, warm-ramped from 2k to 6k, while normal validation/inference remains exact CL39. | Highest priority |
| Highest-priority attribution experiment | Full-96 all-70 BA-off plus a 2 x 2 crossing of correct/wrong PhotoMaker identity tokens and correct/shuffled spatial reference, with trained adapters retained. | Mandatory before strong BA claims |

# 1. Evidence basis and limits

The attached architecture map establishes the lineage: E13/CL14 use hard face replacement; CL19 preserves a full native message and routes `R-N`; CL23 adds low/high frequency shaping; CL27 changes training only; and CL39 adds per-query entropy-based confidence in `up_blocks.0/1` [P1, pp.1, 5-10]. The Comet report provides the fixed-96 metrics and image grids [P2]. The query-mask memo proves that binary `q_face=q*M` is almost entirely a boundary ablation in CL39, not a face-core fix [P3]. The R-branch report and lineage audit provide the same-checkpoint causal interventions [P4, P6].

Important limits:

1. The strongest branch audit has 16 deterministic cells, not a random population sample. Its bootstrap intervals describe those fixed cells only.
2. Only one main training seed is available for CL39. A second seed is required before claiming a population-level architecture improvement.
3. The raw-R intervention covers the 36 confidence-enabled `up_blocks.0/1` processors; raw standalone behavior of the other 34 BA processors has not been rendered equivalently.
4. A whole-denoising low-only or high-only arm is nonlinear. RGB outputs cannot be added to reconstruct the actual output.
5. The current fixed panel uses a single seed. It is excellent for paired causality but insufficient for diversity/copy-paste conclusions.
6. CL39, its generic LoRA, PhotoMaker's default LoRA and the branch projections are co-trained. A simple CL39-versus-PhotoMaker output comparison cannot isolate which component earned the gain.

# 2. Exact current CL39 contract

## 2.1 One self-attention site

At an audited CL39 self-attention processor, the target/noisy row is `T`, the evolving spatial reference row is `H_r`, the target face mask is `M`, the reference support mask is `M_r`, and `O` is `attn.to_out[0]`.

The two projected candidates are:

```text
N = O Attn(Q_n(T), K_n(T), V_n(T))
R = O Attn(Q_n(T), K_r(H_r * M_r), V_r(H_r * M_r))
D = R - N
L = G5(D)
H = D - L
Y_CL39 = N + S * C(q) * [g_L(p) L + g_H(p) H]
```

`S` is the two-cell cosine target-face router. `G5` is the fixed separable binomial 5 x 5 depthwise low-pass. `g_L` rises from 0.50 to 0.85 and `g_H` from 0.75 to 1.25 over denoising progress. In the CL39 confidence groups, `C(q)` is derived from detached normalized attention entropy and is clipped to `[0.25, 1]` [P1, pp.7, 9-10; P2, p.3].

The critical architectural property is that **`N` is never removed**. CL39 does not ask the reference path to generate a face independently; it asks it to provide a controlled correction around the native path.

## 2.2 Where PhotoMaker enters, even inside the branch

The latest training path has several PhotoMaker dependencies [CODE]:

1. The PhotoMaker V2 ID encoder fuses the reference image into the target prompt embedding before the U-Net forward.
2. During the historical `face_embed_strategy="id"` path, the reference half is conditioned by the same prompt embedding with non-ID token positions zeroed and the ID-token positions scaled.
3. The branch Q/K/V modules are initialized by cloning effective attention linears that include the PhotoMaker `default` LoRA delta.
4. The native target message is therefore a PhotoMaker-derived target-attention path, not an independent non-PM baseline.
5. The shared output projection remains part of the active adapted U-Net.
6. The generic `lora_adapter` and PhotoMaker `default` adapter are co-trained with the branch parameters.

The inspected branch cloning helper adds the `default` PhotoMaker delta to the cloned base weight. It does not fold the generic `lora_adapter` delta into the branch Q/K/V base buffer. This asymmetry is useful to understand: branch Q/K/V are separately trainable, but their starting point is explicitly PhotoMaker-shaped, while the common output projection still participates in the jointly adapted model [C2].

## 2.3 Schedule and dominance

The shared validation schedule uses:

```text
steps 0-9:   NO_ID
steps 10-14: PHOTOMAKER
steps 15-49: BOTH (PhotoMaker + branched attention)
```

Thus PhotoMaker begins five denoising steps before BA. By the time BA activates, coarse composition and much of the identity trajectory are already established. BA then acts predominantly as a spatial face correction rather than a new global generator.

## 2.4 Trainable ownership

The clean E13-family contract reports approximately:

| Parameter role | Trainable parameters | Tensors |
|---|---:|---:|
| Branched self-attention rank-128 Q/K/V | 127.80M | 840 |
| Generic U-Net LoRA rank 32 | 30.47M | 700 |
| PhotoMaker default LoRA rank 64 | 60.95M | 700 |
| **Total** | **219.22M** | **2,240** |

This is not a small frozen PhotoMaker with a tiny independent BA head. It is a large co-adapted system. Similarity to PhotoMaker can coexist with material BA learning because all three parameter roles optimize the same denoising objective.

# 3. Why CL39 resembles PhotoMaker

## 3.1 The similarity is architecturally expected

Eight mechanisms point in the same direction:

1. **Same base generative trajectory.** RealVisXL, prompt, negative prompt, seed, scheduler, CFG and number of steps are matched.
2. **PhotoMaker starts first.** PM has a five-step head start before BA.
3. **PhotoMaker conditions both halves.** The target prompt contains PM identity tokens; the reference half also receives PM-ID-conditioned text states.
4. **Branch Q/K/V start from PM-effective projections.** The explicit reference lane is not initialized orthogonally to PM.
5. **Native `N` is always retained.** Even inside the face, CL39 is residual correction, not replacement.
6. **Confidence suppresses most of the residual.** The report estimates a median retained reference fraction around 0.3156; roughly 68.4% of CL27's explicit routed residual is suppressed in the confidence-enabled groups [P2, p.47].
7. **The correction is local.** The target face router protects the rest of the image, so overall composition should remain close.
8. **No training objective rewards distance from PM.** Diffusion MSE, the CL27 surface objective and CL39 confidence calibration reward reconstruction/denoising and hard-case control, not novelty relative to PM output.

## 3.2 “More different” is not the right scalar objective

A model can differ from PhotoMaker for good reasons or bad reasons. Increasing full-image LPIPS from the PM output can mean better pose compliance, stronger identity transfer and less copy-paste; it can also mean artifacts, altered background, worse text adherence or loss of identity.

The recommended target is a **three-axis frontier**:

1. higher identity similarity to the intended person;
2. preserved or improved prompt/pose/expression compliance and face quality;
3. reduced copy-paste/over-similarity to the reference and reduced dependence on the PM-only path.

For each candidate, report both:

```text
identity/control quality: ID_sim, face quality, topology, text, mask IoU
causal contribution: actual - BA-off, correct spatial ref - shuffled ref
variation: generated-to-reference copy-paste metric and distance from PM output
```

A candidate is better when it moves the quality frontier, not merely when it is farther from PM pixels.

## 3.3 Current gains are heterogeneous

The fixed-panel means show that CL39's advantage is not uniform. Relative to PhotoMaker, the largest prompt gains are approximately Chef man `+0.060`, Laughing woman `+0.040`, Skiing man `+0.034`, Dancing man `+0.030`, Jumping man `+0.028`, Rushing man `+0.024` and Kickboxing `+0.023`. The largest regressions are Skiing woman `-0.068`, Drumming woman `-0.047`, Crying woman `-0.036`, Dancing woman `-0.028`, Chef woman `-0.020` and Rushing woman `-0.017` [P2, p.2].

![Prompt-level delta](CL39_branched_attention_report_assets_2026-08-26/prompt_delta_cl39_vs_pm.png)

This is not enough to infer a broad demographic bias: there are only eight fixed identities and one seed. It is enough to justify a predeclared stratification by identity, face size, occlusion, expression and prompt class. A single mean can hide exactly the hard cases the architecture is meant to solve.

# 4. Does branched attention add value?

## 4.1 End-to-end lineage

The fixed-96 lineage is:

| Model | Subject-v2 ID |
|---|---:|
| PhotoMaker V2 | 0.556580 |
| CL14 24k | 0.456116 |
| CL19 24k | 0.506823 |
| CL23 24k | 0.539085 |
| CL27 24k | about 0.543 at endpoint; 0.547 max |
| CL39 16k | **0.570124** |
| CL39 24k | **0.566342** |

![Fixed-96 lineage](CL39_branched_attention_report_assets_2026-08-26/fixed96_lineage_id.png)

CL39-16k is the current provisional end-to-end winner. It is `+0.013544` above the fixed PhotoMaker reference and `+0.022864` above matched CL27. CL39-24k remains `+0.009762` above PhotoMaker [P2, pp.1, 47; P4, p.6]. It also matches PhotoMaker's clean Skiing/Crying topology on the fixed rubric.

The lineage alone is not a clean BA causal proof because each checkpoint includes accumulated training and co-adapted adapters. The same-checkpoint interventions are more informative.

## 4.2 Same-checkpoint causal evidence

On the selected 16 cells:

| Whole-denoising arm | ID similarity | Interpretation |
|---|---:|---|
| Actual CL39 | **0.55754** | trained operating point |
| N-only | 0.52184 | explicit BA correction removed |
| Raw R-on-face | 0.42241 | unscaled standalone reference-route stress |
| Low-only | 0.52793 | only confidence-scaled low band |
| High-only | 0.54047 | only confidence-scaled high band |
| C = 1 | 0.49984 | normal bands, no confidence attenuation |

![Selected-16 branch interventions](CL39_branched_attention_report_assets_2026-08-26/selected16_interventions.png)

Actual CL39 beats N-only by `+0.03570`, wins 11/16 cells, and has a fixed-cell bootstrap interval `[+0.01204, +0.06613]`. It changes 93.80% of face-crop pixels above 1/255. This is strong evidence that the explicit correction is active and useful on this subset [P6, pp.1-3].

The additional group-scoped correction-zero arm in the 36 confidence processors gives actual `0.55754` versus `0.51925`, a `+0.03829` gain with 15/16 wins and a positive interval [P4, pp.1, 5].

## 4.3 What remains unproven

The claim “CL39 beats PhotoMaker because of spatial BA” still needs three controls:

1. **Global all-70 BA-off with trained adapters retained.** This removes the explicit correction everywhere without reverting to an untrained PM checkpoint.
2. **Spatial-reference shuffle with correct PM ID tokens.** If identity/structure does not degrade, the model may be using PM tokens rather than the spatial reference.
3. **Crossed PM-token/spatial-reference conditions.** This separates the two correlated identity sources.

Until these are run on all 96 cells and multiple seeds, the correct conclusion is:

> BA has a measurable incremental effect in the audited layers and cells; CL39 is the best complete system; the exact share of the full-96 surplus over original PhotoMaker attributable to the explicit spatial lane remains provisional.

# 5. The CL39 R branch: real issue, correct interpretation

## 5.1 Raw R is fragile

Raw R-on-face reaches `0.42241`, below CL23 raw R `0.44318`, CL27 raw R `0.46396`, and CL19's trained reference-owned route `0.48975`. Actual CL39 beats raw R on all 16 cells by `+0.13513`, with a positive fixed-cell interval `[+0.09178, +0.18101]` [P6, pp.1-4].

The visual failures are structured rather than random:

- duplicated or shifted eyes;
- nose/mouth misregistration;
- glasses/goggles fused with facial parts;
- hand-to-face transfer in crying cells;
- expression and accessory geometry drift.

This supports a correspondence/ownership diagnosis.

## 5.2 Raw R overstates normal failure

The raw arm routes `N + S(R-N)` in the 36 confidence groups and bypasses both the frequency shaping and confidence suppression. Measured raw residual magnitude is about 3.80 times the actual routed residual; the actual correction is about 22.1% of native attention magnitude [P4, pp.1-4].

Therefore:

- a poor raw-R image is not the normal CL39 output;
- raw R remains a valuable stress test because it reveals what the source residual contains;
- the actual system is successful partly because it attenuates that fragile source.

## 5.3 Confidence is essential but semantically weak

Forcing `C=1` reduces ID from `0.55754` to `0.49984`, with actual winning 14/16 and a positive interval. Several raw-R-like corruptions reappear. Confidence is therefore functional, not cosmetic [P4, p.5; P6, pp.1, 6].

However, current normalized entropy is not correspondence correctness:

1. Invalid reference positions remain in the softmax axis after reference hidden states are zeroed. They behave as zero or constant sinks depending on projection bias.
2. The reference face boxes occupy about 16.68% of the grid, but 48.56% of face-query attention mass remains on invalid/sink positions.
3. Entropy mixes genuine match ambiguity, valid-face area, invalid-key count and logit scale.
4. A sharply concentrated wrong eye-to-glasses match can have low entropy and high confidence.
5. A useful distributed match across a large face can have high entropy and low confidence.
6. The selected-16 correlation between confidence and final intervention size is weak (`r=+0.11`, rank `rho=+0.15`), while raw `|R-N|` magnitude correlates more strongly (`r=+0.56`, `rho=+0.62`) [P4, p.5].

The right conclusion is not to remove confidence. It is to replace the single entropy heuristic with valid-support semantics and richer, bounded low/high reliability.

## 5.4 D-low/D-high are not the cause of raw-R artifacts

The raw-R arm bypasses the Gaussian split. Its artifacts therefore exist before `D_low` and `D_high`. The bands attenuate and reshape the residual:

- raw R face MAE versus N-only: 0.08350;
- low-only: 0.04460;
- high-only: 0.03001;
- actual: 0.04777.

High-only is more ID-efficient than low-only on CL23/27/39, but actual CL39 beats each isolated mean. Both bands are active and likely complementary [P6, pp.1-3, 6].

## 5.5 Binary q-face is not the fix

For every routed target token, `S>0 => M=1`; therefore `q*M=q` in the face core. Binary query masking changes CL39 only indirectly: it changes `R` outside the face, then the pre-router 5 x 5 Gaussian can leak that change up to two cells into the boundary. Face-core duplicated eyes, noses, glasses, hands and expression errors are unaffected at the attention call itself [P3, pp.1-3].

A q-face ablation is worth running as a cheap boundary diagnostic, not as a 24k primary experiment.

# 6. The CL14 lesson and how to rely less on PhotoMaker

## 6.1 Do not restore CL14 hard inference

CL14's reference face path owns the face update:

```text
Y_CL14 = O[(1-M) * B + M * F]
```

Inside `M`, the native face message is absent. This forces `F` to become a usable face denoising path. But aggregate performance is much lower: CL14 24k is `0.456116`; CL19's full-query soft residual route improves it by about `+0.050707` with 74/96 paired wins [P4, pp.6-7]. Reverting to hard replacement would discard the largest proven routing improvement in the lineage.

## 6.2 Borrow the obligation, not the inference equation

The high-value CL14 property is **gradient ownership**:

| Property | CL14 | CL39 |
|---|---|---|
| Face owner during training | reference path | native `N` plus small correction |
| Native fallback inside face | absent | always present |
| Gradient to reference message | full face loss | scaled by router, bands and confidence |
| Required raw route quality | high | low |

The recommended design is to keep CL39 inference unchanged but, on a small coherent fraction of training steps, make the reference route carry the face under the ordinary diffusion target.

## 6.3 Reduce the PM shortcut gradually

There are three distinct PhotoMaker dependencies and they should be ablated separately:

1. **Target PM ID-token conditioning.** Remove only this on selected training forwards while preserving the spatial reference and reference-half conditioning.
2. **Reference-half PM ID conditioning.** Remove only after target dropout proves stable; otherwise both identity sources vanish at once.
3. **PM-derived branch initialization and PM LoRA learning rate.** Change only after the reference path has an ownership curriculum, or the branch may simply become weaker.

The correct sequence is: make R usable -> remove one shortcut -> test causal spatial dependence -> only then reduce PM initialization/learning rate further.

# 7. Mandatory attribution work before new architecture claims

![Experiment roadmap](CL39_branched_attention_report_assets_2026-08-26/experiment_roadmap.png)

## 7.1 Full-96 all-70 controls

For sealed CL39-16k and 24k, generate:

1. actual route;
2. all 70 BA corrections zero, branch/generic/PM trained adapters retained;
3. branch processors removed but all compatible trained LoRAs retained;
4. PM `default` adapter only;
5. generic `lora_adapter` only;
6. both adapters with BA off;
7. confidence forced open;
8. raw R, low-only and high-only for the same declared processor scope.

The all-70 BA-off arm is the minimum causal control missing from the current full panel.

## 7.2 The 2 x 2 identity-source crossing

![2 x 2 crossing](CL39_branched_attention_report_assets_2026-08-26/causal_2x2.png)

For each fixed cell, hold seed, prompt, boxes and weights constant and cross:

| | Correct spatial reference | Shuffled/wrong spatial reference |
|---|---|---|
| Correct PM ID tokens | A: normal | B: PM-only attribution test |
| Wrong/zero PM ID tokens | C: spatial BA isolation | D: negative control |

Strong evidence for spatial BA is `A > B` and `C > D` on ID/topology, with B remaining a plausible PM-conditioned image. Report effect sizes by face size, prompt and identity.

## 7.3 Route-strength, start-time and seed sweeps

Evaluation-only sweeps on the sealed checkpoint:

```yaml
lambda_reference: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
photomaker_start_step: [8, 10, 12, 15]
branched_attn_start_step: [8, 10, 12, 15, 18]
seeds: [0, 1, 2, 3]
```

Do not tune from ID alone. Plot a frontier of ID, text, topology, face quality, PM-output distance and reference copy-paste.

## 7.4 Copy-paste benchmark

WithAnyone identifies a common failure of reconstruction-heavy identity training: the generated face reproduces the reference pose/expression/lighting rather than the identity under a new condition [L1]. Add:

- generated-to-reference ArcFace/SigLIP/DINO similarity;
- generated-to-ground-truth same-ID target similarity where available;
- landmark/expression distance from reference versus target;
- the WithAnyone angular copy-paste metric or a compatible implementation;
- same identity with deliberately different pose/expression reference-target pairs.

This distinguishes “looks like PM” from “copies the reference” and from “correctly preserves identity.”

# 8. Ranked experiment programme

The ranking below considers expected ID headroom, likelihood of improving actual output rather than diagnostics, implementation risk and scientific clarity.

| Priority | Experiment | One key change | Expected outcome | Main risk |
|---:|---|---|---|---|
| 1 | **E1 FOWN-1 / R2-A** | occasional training-only reference-face ownership | coherent R, fewer landmark artifacts, neutral/positive actual ID | loss conflict or early instability |
| 2 | **E2 PM target-condition dropout** | remove target PM ID fusion on selected training forwards | stronger unique spatial-BA contribution and less PM imitation | branch cannot compensate; text/ID regression |
| 3 | **E3 Valid-key reference attention** | truly exclude invalid reference keys and normalize confidence over valid support | cleaner correspondence, meaningful entropy, less face-area dependence | step-zero distribution shift |
| 4 | **E4 Learned band-specific reliability** | bounded learned corrections to low/high confidence | keep useful high-frequency ID while rejecting wrong matches | gate collapses toward N |
| 5 | **E5 Low/high RMS tail caps** | deterministic per-band residual caps | remove rare high-amplitude topology failures cheaply | clip useful identity detail |
| 6 | **E6 DreamMatcher-lite appearance matching** | preserve target QK structure and align only reference values | better pose/expression topology and less object fusion | extra compute; unstable matching |
| 7 | **E7 GT-aligned identity supervision** | low-noise predicted-x0 face identity loss | direct ID_sim lift, especially weak identities | ArcFace gaming or quality loss |
| 8 | **E8 PM-decoupled branch initialization** | reduce PM-default delta folded into branch Q/K/V | more independent spatial expert | weaker start; needs E1 curriculum |
| 9 | E9 Semantic core/halo support | different reference support for low/high bands | less hand/glasses/hair fusion | parser errors and lost accessories |
| 10 | E10 Resolution-normalized frequency | scale low-pass support by token resolution | cleaner coarse/detail allocation | subtle effect; broader sweep needed |
| Later | E11 TIDE-style preference fine-tuning | preference loss over balanced versus distorted outputs | hard-case polish after architecture stabilizes | hides causal mechanism; reward hacking |

Each trained arm should inherit the same data, base, optimizer budget, mask, scheduler, validation cadence and fixed panel unless the stated experiment explicitly changes one of them. Do not bundle E1-E5 into one run.

# 9. Shared implementation foundation

Before any scientific arm, add a small defaults-off control layer. Prefer new helper modules and minimal changes to key model files.

## 9.1 Proposed files

```text
diffusion_template/src/model/photomaker_branched/
  ba_experiment_state.py       # immutable per-forward route/condition state
  reference_reliability.py     # learned reliability and RMS helpers
  reference_correspondence.py  # DreamMatcher-lite utilities
  id_objectives.py             # predicted-x0 ID losses

Existing files with concise changes:
  hardcase_attn_processor.py   # consume state; route/gate/mask hooks
  branched_runtime.py          # install controls by declared group
  lora2.py                     # choose one global mode before U-Net forward
  lora2_helpers.py             # return text-only and PM-fused prompt states
  e13_contract.py              # exact trainable/checkpoint manifest
  configs/model/photomaker_branched_lora2.yaml
```

## 9.2 One global state per U-Net forward

Never sample route modes inside individual processors. Use a frozen state object installed before the U-Net call and cleared in `finally`.

```python
# ba_experiment_state.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BAForwardState:
    ownership_active: bool = False
    ownership_strength: float = 0.0
    target_pm_dropout: bool = False
    reference_query_mode: str = "full"


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def deterministic_event(seed: int, step: int, probability: float) -> bool:
    if probability <= 0:
        return False
    u = splitmix64(seed ^ (step * 0x9E3779B1)) / float(2**64)
    return u < probability
```

Because `seed` and optimizer `global_step` are identical on all DDP ranks, the decision is coherent without per-rank RNG. If the framework exposes a gradient-accumulation microstep index, include it in the hash only when every rank receives the same value.

```python
# lora2.py - schematic
state = self.ba_controller.state_for_step(
    global_step=self.global_step,
    training=self.training,
)
for proc in self._ba_selected_processors:
    proc.set_forward_state(state)
try:
    noise_pred = run_branched_forward_pass(...)
finally:
    for proc in self._ba_selected_processors:
        proc.clear_forward_state()
```

## 9.3 Required parity gates

Before launching any new training job:

1. historical configs compose without adding checkpoint keys;
2. every new field defaults off;
3. sealed CL39 loads with no missing/unexpected keys;
4. a 12-image smoke is byte-identical with all new controls off;
5. no processor dictionary is resolved inside a per-layer forward loop;
6. collectors allocate no full activations unless explicitly enabled;
7. startup records exact trainable tensors, parameters, group scopes and effective config.

# 10. E1 - FOWN-1 / R2-A: training-only reference ownership

## 10.1 Hypothesis

Raw R is fragile because the main loss never requires it to be a complete face path. On a small coherent fraction of training forwards, moving the selected processors toward `N + S(R-N)` will make R geometrically usable while normal CL39 inference remains unchanged.

This is the most direct test of the CL14 training-ownership hypothesis and adds no inference parameters or second U-Net forward [P3, pp.3-6; P4, pp.8-13].

## 10.2 Exact forward change

Normal CL39:

```text
delta_normal = C * (g_L * L + g_H * H)
Y = N + S * delta_normal
```

Ownership forward:

```text
delta_raw = R - N
Y_train = N + S * [(1-lambda) * delta_normal + lambda * delta_raw]
```

Use one global mode across the 36 `up_blocks.0/1` processors. The other 34 BA processors remain at ordinary CL39 in the first arm.

## 10.3 Key processor code

```python
# hardcase_attn_processor.py

def _apply_reference_ownership(
    self,
    *,
    native_out: torch.Tensor,
    reference_out: torch.Tensor,
    normal_delta: torch.Tensor,
) -> torch.Tensor:
    state = getattr(self, "_forward_state", None)
    if (
        not self.training
        or state is None
        or not state.ownership_active
        or state.ownership_strength <= 0.0
    ):
        return normal_delta

    raw_delta = reference_out - native_out
    lam = float(state.ownership_strength)
    return torch.lerp(normal_delta, raw_delta, lam)

# In _call_temporal_frequency / confidence route:
normal_delta = confidence * (gain_low * delta_low + gain_high * delta_high)
routed_delta = self._apply_reference_ownership(
    native_out=native_out,
    reference_out=reference_out,
    normal_delta=normal_delta,
)
target_out = native_out + router * routed_delta
```

The route should use the already computed full target-query/reference-KV message. Writing a binary `q*M` version may make the CL14 relationship explicit, but it is not the scientific change in the face core.

## 10.4 Controller schedule

```python
def ownership_strength(step: int, start: int = 2000, end: int = 6000) -> float:
    if step <= start:
        return 0.0
    return min(1.0, max(0.0, (step - start) / float(end - start)))

active = (
    self.training
    and deterministic_event(experiment_seed, global_step, 0.125)
)
state = BAForwardState(
    ownership_active=active,
    ownership_strength=ownership_strength(global_step) if active else 0.0,
)
```

## 10.5 YAML

```yaml
# CL39F_cosmic_reference_face_ownership_24k.yaml
defaults:
  - CL39_cosmic_null_key_confidence_router_24k
  - _self_

model:
  ba_reference_face_ownership_enabled: true
  ba_reference_face_ownership_groups:
    - up_blocks.0
    - up_blocks.1
  ba_reference_face_ownership_probability: 0.125
  ba_reference_face_ownership_ramp_start_step: 2000
  ba_reference_face_ownership_ramp_end_step: 6000
  ba_reference_face_ownership_query_mode: full

pipeline:
  pose_adapt_ratio: 0.0
  ca_mixing_for_face: false
```

## 10.6 Verification

- one state value across all selected processors and ranks;
- zero ownership during validation/inference;
- finite two-step smoke;
- nonzero gradients in reference Q/K/V on ownership steps;
- no extra trainables;
- ordinary-step gradients match historical CL39 numerically within tolerance;
- exact CL39 output when ownership is off.

## 10.7 Expected result and gate

Expected: raw-R severe artifacts fall materially, raw-R ID increases, actual fixed-96 ID stays neutral or improves, and actual remains better than BA-off.

Promote only if:

- severe raw-R failures fall by at least 50% on predeclared cells;
- actual ID is no worse than CL39 by more than 0.005;
- TOPIQ-Face, MUSIQ, MANIQA, mask IoU, text and Skiing/Crying topology are non-inferior;
- actual remains causally sensitive to the correct spatial reference.

# 11. E2 - PM target-condition dropout

## 11.1 Hypothesis

The target half can satisfy identity loss through PhotoMaker tokens and a strong native path, reducing pressure to use spatial BA. On selected training forwards, replacing the PM-fused target prompt with its text-only pre-fusion version will force the explicit reference lane to carry more unique identity information. Normal validation/inference remains unchanged.

This is modality/shortcut dropout, not permanent removal of PhotoMaker.

## 11.2 Preserve both prompt representations

Current input preparation overwrites the text prompt with the ID-encoder output. Return both:

```python
# lora2_helpers.py - within prepare_branched_training_inputs
text_prompt_embeds, pooled, class_mask = _encode_prompts_with_trigger_word(...)

with torch.no_grad():
    pm_prompt_embeds = model.id_encoder(
        id_pixel_values,
        text_prompt_embeds.to(dtype=model.id_encoder.dtype),
        class_mask,
        id_embeds,
    )

return {
    "text_prompt_embeds": text_prompt_embeds.to(model.unet.dtype),
    "pm_prompt_embeds": pm_prompt_embeds.to(model.unet.dtype),
    "pooled_prompt_embeds": pooled.to(model.unet.dtype),
    ...
}
```

On a dropout forward, use text-only states for the target half but keep the historical PM-ID-only reference-half conditioning. This changes one shortcut at a time.

```python
# branched_runtime.py / two_branch_predict
if state.target_pm_dropout:
    target_encoder_states = text_prompt_embeds
else:
    target_encoder_states = pm_prompt_embeds

reference_encoder_states = build_id_only_reference_states(
    pm_prompt_embeds, class_tokens_mask, id_token_scale
)
encoder_hidden_states = torch.cat(
    [target_encoder_states, reference_encoder_states], dim=0
)
```

## 11.3 Scope and schedule

Start only after E1 demonstrates a stable R path. Recommended first arm:

```yaml
model:
  ba_pm_condition_dropout_enabled: true
  ba_pm_condition_dropout_probability: 0.25
  ba_pm_condition_dropout_scope: target_prompt_only
  ba_pm_condition_dropout_start_step: 2000
  ba_pm_condition_dropout_end_step: 16000
```

No change to optimizer LRs in the first arm. A separate follow-up may lower the PM-default LR or freeze it late; do not bundle that with dropout.

## 11.4 Expected result and gate

Expected: stronger A-versus-B separation in the 2 x 2 causal crossing, higher C-versus-D spatial-BA isolation, more distance from PM output on identity-relevant face pixels, and equal or better ID.

Risks: target ID collapses on dropout batches, generic text conditioning shifts class semantics, or the reference branch learns a PM-token shortcut via the reference half.

Promotion requires:

- actual fixed-96 ID no worse than E1 parent by 0.005;
- correct spatial reference beats shuffled reference more strongly than in E1;
- PM-off/spatial-on arm improves relative to the negative control;
- no background or prompt-adherence regression.

# 12. E3 - true valid-key reference attention

## 12.1 Problem

CL39 multiplies reference hidden states by the reference face mask, then projects K/V, but invalid positions remain on the softmax axis. This creates zero/constant sinks and makes entropy depend on mask area. A valid-key attention contract should exclude invalid positions rather than asking the model to learn around them.

## 12.2 Masked SDPA helper

```python
# reference_reliability.py
import torch
import torch.nn.functional as F


def make_key_bias(valid_ref: torch.Tensor, query_len: int, dtype) -> torch.Tensor:
    # valid_ref: [B, L] bool; output broadcastable to [B, H, Q, K]
    if valid_ref.ndim != 2:
        raise ValueError(f"valid_ref must be [B,L], got {tuple(valid_ref.shape)}")
    if not valid_ref.any(dim=-1).all():
        raise RuntimeError("reference face mask contains a sample with no valid keys")
    bias = torch.zeros(
        valid_ref.shape[0], 1, query_len, valid_ref.shape[1],
        device=valid_ref.device, dtype=dtype,
    )
    return bias.masked_fill(~valid_ref[:, None, None, :], torch.finfo(dtype).min)


def masked_reference_attention(q, k_ref, v_ref, valid_ref):
    bias = make_key_bias(valid_ref, q.shape[-2], q.dtype)
    return F.scaled_dot_product_attention(
        q, k_ref, v_ref,
        attn_mask=bias,
        dropout_p=0.0,
        is_causal=False,
    )
```

Project the unmasked reference hidden states, then exclude invalid keys:

```python
k_ref = self._reshape_heads(self._k_ref(attn, reference), heads)
v_ref = self._reshape_heads(self._v_ref(attn, reference), heads)
valid_ref = ref_mask.squeeze(-1).bool()
reference_message = masked_reference_attention(query, k_ref, v_ref, valid_ref)
```

## 12.3 Conditional entropy

Use the exact same valid set for confidence:

```python
logits = torch.matmul(q.float(), k_ref.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
valid = valid_ref[:, None, None, :]
logits = logits.masked_fill(~valid, -torch.inf)
prob = logits.softmax(dim=-1)

entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
num_valid = valid_ref.sum(dim=-1).clamp_min(2).float()
normalized_entropy = entropy / num_valid.log()[:, None, None]
normalized_entropy = normalized_entropy.mean(dim=1)  # heads -> per query
```

## 12.4 Warm transition

A sealed CL39 checkpoint was trained with sink positions, so immediate inference-only replacement may cause a distribution shift. First run the same-checkpoint diagnostic. For training, warm the mask bias from zero to full over 2k steps:

```python
mask_strength = ramp(global_step, start=0, end=2000)
bias = invalid_mask * (mask_strength * torch.finfo(dtype).min)
```

In practice, multiplying the most negative finite value is unsafe at intermediate strengths. Use a stable finite schedule, for example `-20 * mask_strength` in fp32 logits, reaching full `-inf` only after the ramp.

## 12.5 YAML

```yaml
model:
  ba_reference_key_mask_mode: additive_logit_mask
  ba_reference_key_mask_groups: [up_blocks.0, up_blocks.1]
  ba_reference_key_mask_warmup_steps: 2000
  ba_confidence_entropy_support: valid_keys_only
  ba_confidence_entropy_normalization: log_valid_count
```

## 12.6 Expected result and gate

Expected: valid mass rises to 1 by construction, entropy becomes comparable across face sizes, wrong but sharp matches remain a separate problem, and R requires less suppression.

Reject if step-zero or early-training ID drops sharply, topology repeats the negative canonical-K/V pattern seen in CL41, or the gain disappears under spatial-reference shuffle.

# 13. E4 - bounded low/high learned reliability

## 13.1 Hypothesis

One scalar entropy confidence cannot distinguish coarse structural reliability from fine identity-detail reliability. A tiny zero-initialized MLP can learn bounded corrections for low and high bands from detached diagnostic features while preserving exact CL39 at initialization.

## 13.2 Features

For every query, use detached, compact features:

1. attention mass on valid reference positions;
2. conditional entropy over valid positions;
3. top-1 minus top-2 probability margin;
4. cosine agreement between `N` and `R`;
5. `log(RMS(D)/RMS(N))`;
6. denoising progress.

These features directly address the measured weaknesses of entropy-only confidence and the modestly positive time/agreement result in CL44 [P4, p.8].

## 13.3 Module

```python
# reference_reliability.py
import torch
import torch.nn as nn

class BandReliability(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 16, max_delta: float = 0.20):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base_conf: torch.Tensor):
        z = self.net(features.detach())
        correction = self.max_delta * torch.tanh(z)
        c_low = (base_conf + correction[..., 0]).clamp(0.25, 1.0)
        c_high = (base_conf + correction[..., 1]).clamp(0.25, 1.0)
        return c_low, c_high
```

Parameter count is only 146 per processor (`6x16+16 + 16x2+2`), about 5,256 parameters across 36 processors.

## 13.4 Feature construction

```python
valid_mass = (prob_all * valid.float()).sum(dim=-1).mean(dim=1)
cond_prob = prob_all * valid.float()
cond_prob = cond_prob / cond_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
cond_entropy = -(cond_prob * cond_prob.clamp_min(1e-8).log()).sum(-1)
cond_entropy = cond_entropy.mean(dim=1) / valid_count.log()

top2 = cond_prob.topk(k=2, dim=-1).values
margin = (top2[..., 0] - top2[..., 1]).mean(dim=1)
nr_cos = F.cosine_similarity(native_out.float(), reference_out.float(), dim=-1)
log_ratio = torch.log(
    reference_delta.float().pow(2).mean(-1).sqrt()
    / native_out.float().pow(2).mean(-1).sqrt().clamp_min(1e-6)
    + 1e-6
)
progress = torch.full_like(valid_mass, float(self.ba_denoise_progress))
features = torch.stack(
    [valid_mass, cond_entropy, margin, nr_cos, log_ratio, progress], dim=-1
)
```

## 13.5 Route

```python
c_low, c_high = self.reliability(features, base_confidence)
target_out = native_out + router * (
    c_low.unsqueeze(-1) * gain_low * delta_low
    + c_high.unsqueeze(-1) * gain_high * delta_high
)
```

## 13.6 YAML and trainable manifest

```yaml
model:
  ba_reliability_mode: learned_band
  ba_reliability_groups: [up_blocks.0, up_blocks.1]
  ba_reliability_hidden_dim: 16
  ba_reliability_max_delta: 0.20
  ba_reliability_features:
    - valid_mass
    - conditional_entropy
    - top2_margin
    - native_reference_cosine
    - log_residual_native_rms
    - denoise_progress
```

Fail closed if the exact added tensor/parameter count differs from the startup manifest.

## 13.7 Expected result and gate

Expected: high-band confidence stays open on useful identity detail but closes on wrong-but-sharp matches; low-band confidence remains smoother and more conservative. Actual should approach or exceed CL39 with fewer C=1/raw-R-like failures.

Reject if correction-zero and spatial-shuffle sensitivity shrink materially: that would indicate the gate learned a native shortcut rather than better reliability.

# 14. E5 - deterministic low/high RMS tail caps

## 14.1 Hypothesis

Rare residual tails create duplicated or fused facial structure. The audit records the following face RMS ratios before confidence:

| Ratio | p50 | p95 | maximum |
|---|---:|---:|---:|
| low / native | 0.767 | 0.910 | 1.424 |
| high / native | 0.301 | 0.425 | 0.631 |
| raw `(R-N)` / native | 0.880 | 1.021 | 1.499 |
| actual routed / native | 0.200 | 0.415 | 0.691 |

A cap at approximately `k_L=0.90`, `k_H=0.45` touches the observed tails rather than the median [P4, p.9].

## 14.2 Code

```python
# reference_reliability.py

def masked_rms(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [B,L,C], mask: [B,L,1]
    w = mask.to(dtype=x.dtype)
    denom = (w.sum(dim=1) * x.shape[-1]).clamp_min(1.0)
    return ((x.float().pow(2) * w).sum(dim=(1, 2)) / denom.squeeze(-1)).sqrt()


def cap_band(
    band: torch.Tensor,
    native: torch.Tensor,
    face_mask: torch.Tensor,
    cap_ratio: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    band_rms = masked_rms(band, face_mask)
    native_rms = masked_rms(native, face_mask)
    scale = (float(cap_ratio) * native_rms / (band_rms + eps)).clamp(max=1.0)
    return band * scale[:, None, None].to(band.dtype), scale
```

Apply caps before gains and confidence:

```python
low_capped, low_scale = cap_band(delta_low, native_out, binary_face, self.low_cap)
high_capped, high_scale = cap_band(delta_high, native_out, binary_face, self.high_cap)
routed_delta = confidence * (
    gain_low * low_capped + gain_high * high_capped
)
```

The binary face core, not the soft router, should define RMS statistics; otherwise transition-ring area changes the cap.

## 14.3 Experiment design

Run same-checkpoint evaluation first:

```yaml
caps:
  - {low: null, high: null}  # exact CL39
  - {low: 0.80, high: 0.40}
  - {low: 0.90, high: 0.45}
  - {low: 1.00, high: 0.50}
```

Then train only the best non-regressive setting as a separate 24k arm. No learned reliability in this run.

## 14.4 Expected result and gate

Expected: fewer severe topology failures with nearly unchanged mean ID. Reject any cap with ID loss above 0.005 or visible loss of fine identity features. Record the fraction of layer calls actually capped; a cap that fires almost everywhere is not a tail cap.

# 15. E6 - DreamMatcher-lite appearance matching

## 15.1 Motivation

Current `R` changes both the structure path and appearance path because it uses target queries with reference keys and reference values. DreamMatcher's central insight is to leave target query-key structure intact and inject semantically aligned reference values [L2]. This directly addresses pose/expression and glasses/hand fusion.

## 15.2 Proposed message

At one processor:

```text
A_target = softmax(Q_n(T) K_n(T)^T)
P_tr     = semantic correspondence from target tokens to valid reference tokens
V_match  = P_tr V_r
V_mix    = cycle_conf * V_match + (1-cycle_conf) * V_n
R_AMA    = O(A_target V_mix)
D_AMA    = R_AMA - N
```

Target QK determines structure; reference V contributes aligned appearance.

## 15.3 Minimal implementation

```python
# reference_correspondence.py
import torch
import torch.nn.functional as F


def normalized_similarity(target_desc, ref_desc, valid_ref, temperature=0.07):
    t = F.normalize(target_desc.float(), dim=-1)
    r = F.normalize(ref_desc.float(), dim=-1)
    sim = torch.matmul(t, r.transpose(-1, -2)) / float(temperature)
    sim = sim.masked_fill(~valid_ref[:, None, :], -torch.inf)
    return sim


def cycle_confidence(sim_tr, target_side: int, threshold=0.08, temperature=0.02):
    # Hard indices are diagnostic/control only; value transport can remain soft.
    j = sim_tr.argmax(dim=-1)                     # target -> reference
    sim_rt = sim_tr.transpose(-1, -2)
    i_back_all = sim_rt.argmax(dim=-1)            # reference -> target
    i_back = torch.gather(i_back_all, 1, j)

    length = sim_tr.shape[1]
    yy, xx = torch.meshgrid(
        torch.arange(target_side, device=sim_tr.device),
        torch.arange(target_side, device=sim_tr.device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=-1).reshape(length, 2).float()
    coords = coords / max(target_side - 1, 1)
    source = coords[None].expand(sim_tr.shape[0], -1, -1)
    returned = coords[i_back]
    error = (returned - source).pow(2).sum(-1).sqrt()
    return torch.sigmoid((float(threshold) - error) / float(temperature))
```

Processor sketch:

```python
# Reuse target structure attention.
attn_target = torch.softmax(
    torch.matmul(q_n.float(), k_n.float().transpose(-1, -2)) * scale,
    dim=-1,
).to(v_n.dtype)

# Build correspondence once per selected processor/head-averaged descriptors.
target_desc = k_n.mean(dim=1)     # [B,L,D]
ref_desc = k_ref.mean(dim=1)
sim_tr = normalized_similarity(target_desc, ref_desc, valid_ref)
p_tr = sim_tr.softmax(dim=-1).to(v_ref.dtype)

# Align each reference value token to the target token grid.
v_ref_flat = v_ref.mean(dim=1)    # first implementation; later keep head-specific
v_match = torch.matmul(p_tr, v_ref_flat)
cycle = cycle_confidence(sim_tr, target_side=int(length**0.5))
v_native_flat = v_n.mean(dim=1)
v_mix = cycle[..., None] * v_match + (1.0 - cycle[..., None]) * v_native_flat

# Expand to heads for a minimal smoke, then implement head-specific transport.
v_mix_h = v_mix[:, None].expand(-1, heads, -1, -1)
r_ama = torch.matmul(attn_target, v_mix_h)
r_ama = attn.to_out[0](self._merge_heads(r_ama))
```

The production implementation should preserve head-specific values; the head-averaged version is only a bounded smoke to validate geometry and runtime.

## 15.4 Scope

- `up_blocks.0/1` only;
- token grids 16 x 16 and 32 x 32 initially;
- activate only in middle/late denoising where descriptors are meaningful;
- cache correspondence within a processor call; never across samples/steps;
- no full correlation tensor telemetry unless requested.

## 15.5 YAML

```yaml
model:
  ba_reference_message_mode: appearance_matching
  ba_appearance_matching_groups: [up_blocks.0, up_blocks.1]
  ba_appearance_matching_max_side: 32
  ba_appearance_matching_progress_start: 0.35
  ba_appearance_matching_temperature: 0.07
  ba_cycle_threshold: 0.08
  ba_cycle_temperature: 0.02
```

## 15.6 Expected result and gate

Expected: better pose/expression consistency, fewer glasses/hand/eye fusions, and greater identity gain in hard cases without forcing the target to copy reference geometry.

Risk: descriptors are too noisy or branch-specific K spaces are not semantically aligned. If so, add a **zero-initialized shared rank-32 descriptor projection** as a separate paired arm rather than silently bundling it.

# 16. E7 - GT-aligned predicted-x0 identity supervision

## 16.1 Motivation

CL39 optimizes diffusion MSE plus the CL27 surface objective; it has no direct identity loss. WithAnyone uses paired same-ID data, target-aligned face supervision and contrastive identity learning to increase ID while reducing copy-paste [L1]. The current dataset already provides a distinct same-ID target and reference, target face boxes and corrected geometry, so a low-cost target-aligned identity objective is feasible.

## 16.2 First arm: target-aligned ID loss only

Restrict the loss to a deterministic subset of low-noise training samples, where predicted `x0` is face-like enough for a frozen identity encoder.

For epsilon prediction:

```python
# id_objectives.py

def predict_x0(noisy_latents, noise_pred, timesteps, scheduler):
    alpha_bar = scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps]
    while alpha_bar.ndim < noisy_latents.ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)
    return (
        noisy_latents - (1.0 - alpha_bar).sqrt() * noise_pred
    ) / alpha_bar.sqrt().clamp_min(1e-6)
```

Decode with gradients through the frozen VAE decoder:

```python
x0 = predict_x0(noisy_latents, noise_pred, timesteps, self.noise_scheduler)
decoded = self.vae.decode(x0 / self.vae.config.scaling_factor).sample
```

Crop/align the predicted and ground-truth face using **ground-truth target landmarks or the target bbox**, not a detector run on the noisy prediction. This avoids unstable face matching.

```python
pred_face = differentiable_crop_align(decoded, target_landmarks, output_size=112)
with torch.no_grad():
    gt_face = crop_align(pixel_values, target_landmarks, output_size=112)
    gt_id = F.normalize(self.frozen_id_encoder(gt_face), dim=-1)
pred_id = F.normalize(self.frozen_id_encoder(pred_face), dim=-1)
loss_id = (1.0 - (pred_id * gt_id).sum(dim=-1)).mean()
```

## 16.3 Selection and weight

```yaml
model:
  ba_id_loss_enabled: true
  ba_id_loss_probability: 0.125
  ba_id_loss_max_timestep: 250
  ba_id_loss_weight: 0.01
  ba_id_loss_alignment: target_gt_landmarks
```

The loss should be globally selected by stateless hash, computed only where a valid target face/landmark set exists, and skipped cleanly otherwise.

## 16.4 Follow-up: contrastive identity centroid

Only after the direct ID loss passes, add a separate arm with InfoNCE against other identities in the batch or a memory queue:

```python
positive = gt_id                       # distinct same-ID target
negatives = memory_bank.sample_except(identity_id)
logits = torch.cat([
    (pred_id * positive).sum(-1, keepdim=True),
    pred_id @ negatives.T,
], dim=-1) / temperature
labels = torch.zeros(pred_id.shape[0], dtype=torch.long, device=pred_id.device)
loss_contrastive = F.cross_entropy(logits, labels)
```

The positive should be the target identity or an identity centroid across diverse same-ID images, not only the exact reference image; otherwise the objective can worsen copy-paste.

## 16.5 Expected result and gate

Expected: a direct ID_sim lift, especially for Jisoo/Marion and the weak female hard-case prompts, while paired diverse positives preserve editability.

Risks: metric gaming, over-sharpened faces, loss of prompt expression, and excessive VAE/ArcFace memory. Keep the first arm low-weight, low-noise and sparse. Promotion requires no copy-paste regression and non-inferior face quality/text.

# 17. E8 - PM-decoupled branch initialization

## 17.1 Hypothesis

Branch Q/K/V currently begin from effective PhotoMaker-default projections. This is a strong reason for architectural convergence toward PM. After E1 makes the reference route self-sufficient, reducing the PM delta folded into branch initialization can create a more independent spatial expert.

## 17.2 Code

```python
# attn_processor_cleanest.py

def _clone_effective_linear(attn_linear, *, rank=128, pm_init_scale=1.0):
    base = attn_linear.get_base_layer() if hasattr(attn_linear, "get_base_layer") else attn_linear
    cloned = BranchLoRALinear(...)
    with torch.no_grad():
        weight = base.weight.detach().clone()
        if hasattr(attn_linear, "lora_A") and "default" in attn_linear.lora_A:
            pm_delta = attn_linear.get_delta_weight("default").detach()
            weight.add_(float(pm_init_scale) * pm_delta.to(weight.device, weight.dtype))
        cloned.base_weight.copy_(weight)
        ...
    return cloned
```

Pass the scale through processor construction:

```yaml
model:
  ba_branch_pm_init_scale: 0.5
```

## 17.3 Critical checkpoint caveat

`base_weight` is a registered branch buffer. Loading an existing CL39 checkpoint may overwrite the scaled initialization. Therefore this experiment must either:

1. train a new matched CL39 architecture from the same PhotoMaker/E13 parent with `alpha in {1.0, 0.5, 0.0}`; or
2. implement and verify an explicit rebase operation that changes the effective branch weight after checkpoint load.

Do not claim an initialization experiment if the checkpoint silently restores `alpha=1` buffers.

## 17.4 Design

Use E1 as the training curriculum for every alpha arm, but compare `alpha=0.5` against an E1 parent with `alpha=1.0`. This isolates initialization while keeping ownership identical.

Expected: more distinct spatial-reference behavior and stronger shuffled-reference sensitivity. Risk: lower step-zero ID and slower convergence. Do not promote a visually different branch that merely becomes weaker.

# 18. E9 - semantic core/halo reference support

## 18.1 Hypothesis

A single rectangular reference support sends skin, facial components, hair, hands, glasses, hats and background through the same K/V path. High-frequency identity detail should come primarily from visible facial core regions; low-frequency shape can use a broader halo.

## 18.2 Offline masks

Run a frozen face parser during dataset preprocessing and store masks in the conditioning descriptor:

```text
core: skin, brows, eyes, nose, lips, facial hair
halo: core + ears + hair + neck
object/occluder: glasses, hats, hands and unknown foreground
```

A conservative first arm can treat glasses as object/occluder because the fixed Skiing failures show eyewear-to-eye fusion. A later identity-accessory arm can reintroduce reference glasses only when the prompt/reference contract requires them.

## 18.3 Processor

```python
reference_core = reference * core_mask
reference_halo = reference * halo_mask

r_core = project_attn(query, K_ref(reference_core), V_ref(reference_core))
r_halo = project_attn(query, K_ref(reference_halo), V_ref(reference_halo))

# High-frequency identity from core; low-frequency form from broader support.
delta_high_source = r_core - native_out
delta_low_source = r_halo - native_out
low = gaussian_lowpass(delta_low_source)
high = delta_high_source - gaussian_lowpass(delta_high_source)
```

This is one support-factorization change. Do not simultaneously add learned reliability.

## 18.4 Expected result and gate

Expected: fewer hand/eyewear/hair transfers and better high-frequency face topology. Risk: parser errors, loss of legitimate accessories, or mismatch between reference and target semantics. Require parser-mask overlays in every smoke and fail closed on empty core masks.

# 19. E10 - resolution-normalized frequency split

## 19.1 Problem

A fixed 5 x 5 kernel has very different physical support at 16 x 16, 32 x 32 and 64 x 64 token grids. At low resolution it can blur a large part of the face; at high resolution it is a narrow local filter. The same `g_L/g_H` schedule therefore does not represent the same coarse/detail decomposition across blocks.

## 19.2 Minimal block-normalized arm

```python
KERNEL_BY_SIDE = {
    16: (3, [1, 2, 1]),
    32: (5, [1, 4, 6, 4, 1]),
    64: (9, [1, 8, 28, 56, 70, 56, 28, 8, 1]),
}
```

Normalize each 1D kernel, take the outer product and apply depthwise with padding. For unlisted sizes, select the nearest supported side.

A more adaptive follow-up uses face diameter:

```python
face_radius_cells = (binary_face.sum(dim=1).sqrt() / 2.0).clamp_min(1.0)
sigma_cells = (0.12 * face_radius_cells).clamp(0.8, 3.0)
```

The dynamic version is more expensive and should not be the first arm.

## 19.3 Expected result and gate

Expected: cleaner low/high semantics, less boundary leakage at low-resolution blocks and better fine identity at high-resolution blocks. Gate on band telemetry, actual ID/quality and hard-case topology, not only raw band visualizations.

# 20. E11 - later TIDE-style preference tuning

TIDE trains with winner/loser targets to balance subject preservation and instruction compliance [L8]. Once a causal architecture has passed, create preference pairs from:

- actual CL39 or promoted candidate;
- N-only;
- C=1;
- raw R;
- low-only/high-only;
- shuffled spatial reference;
- outputs with known topology failures.

A DPO-like noise-space loss can train only the reliability/router parameters while regularizing to the promoted base. This is a polishing stage, not a substitute for causal architecture. It should not be used to hide a branch that contributes no unique spatial evidence.

# 21. Secondary code and training observations

## 21.1 One timestep per whole batch

The inspected `lora2.py` samples one scalar timestep and repeats it across the batch. This preserves target/reference alignment but reduces timestep diversity per optimizer step. A low-cost follow-up can sample one timestep per target-reference pair:

```python
timesteps = torch.randint(
    0, self.noise_scheduler.config.num_train_timesteps,
    (batch_size,), device=latents.device,
).long()
```

The doubled U-Net input must use the corresponding repeated vector `[t_target, t_reference]`, with the same timestep for each pair. Treat this as an optimization/training-coverage arm, not an architecture claim.

## 21.2 Fixed 16k versus 24k selection

CL39 peaks at 16k and declines modestly by 24k. New runs should report both the selected checkpoint and the endpoint, and selection should consider ID, prompt, quality and topology jointly. A later checkpoint with a slightly lower mean may still improve robustness; conversely, selecting solely on fixed-96 ID risks panel overfitting.

## 21.3 Role-specific learning rates

The three trainable roles are very different but currently share a broad optimizer contract. After PM-dropout attribution, test a separate arm such as:

```yaml
optimizer:
  branch_qkv_lr: 1.0e-4
  generic_lora_lr: 5.0e-5
  photomaker_default_lr: 1.0e-5
```

Do not combine this with the first PM-dropout run. Its purpose is to prevent the already strong PM path from absorbing most training while allowing branch adaptation.

# 22. Literature-to-architecture mapping

| Paper | Relevant mechanism | Direct transfer to CL39 | Priority |
|---|---|---|---:|
| WithAnyone (2025) [L1] | paired same-ID data, target-aligned ID loss, contrastive identity objective, copy-paste benchmark | E7 and the new variation/copy-paste evaluation | High |
| DreamMatcher (2024) [L2] | preserve target structure path; semantically align reference values; cycle-consistency confidence | E6 appearance-only value transfer and cycle gate | High |
| IC-Portrait (2025) [L3] | fast dense correspondence learning and synthetic view-consistent profiles | supports a lightweight learned descriptor if E6's frozen descriptors fail | Medium-high |
| DynamicID (2025) [L4] | query-level semantic activation and identity-motion disentanglement | supports richer query reliability and expression/pose-aware gating | Medium |
| InfiniteYou (2025) [L5] | residual identity injection and staged paired SPMS training to reduce copy-paste | validates residual architecture plus paired-stage curriculum | Medium-high |
| InstantCharacter (2025) [L6] | paired/unpaired staged training for consistency, controllability and quality | supports curriculum: ownership -> paired controllability -> quality | Medium |
| Omni-ID (2024) [L7] | generatively trained holistic identity representation, few-to-many reconstruction | later replacement/augmentation of discriminative PM identity tokens | Medium, larger project |
| TIDE (2025) [L8] | target supervision and preference learning over balanced/distorted outputs | E11 hard-case polishing | Later |
| WithEveryone (2026) [L9] | layout-grounded ID loss and ID representation forcing | reinforces GT-region identity supervision and explicit forcing; group planner is out of scope | Medium for later loss design |

A common theme across the strongest recent work is that identity preservation and editability improve when training uses **paired, diverse same-ID targets** and explicitly separates identity evidence from pose/structure, rather than simply increasing reference feature strength.

# 23. Unified evaluation protocol

## 23.1 Fixed controls

Every scientific comparison should retain:

```text
base: RealVisXL V4.0
sampler: historical matched DDIM path
inference steps: 50
CFG: 5
fixed references/prompts/boxes/masks
pose_adapt_ratio: 0
ca_mixing_for_face: false
validation: step 0 and every 2k; selected checkpoint plus 24k endpoint
```

Add four seeds for promoted checkpoints while keeping seed 0 as the immutable paired panel.

## 23.2 Metrics

### Primary

- subject-v2 mask-matched identity similarity;
- paired fixed-cell ID delta and bootstrap interval;
- severe topology failure count under a predeclared rubric;
- correct-spatial-reference versus shuffled-reference delta.

### Quality and control

- TOPIQ-Face;
- MUSIQ;
- MANIQA;
- prompt/text similarity;
- mask IoU and face-box validity;
- face landmark plausibility/symmetry where reliable;
- Skiing and Crying topology pass/minor/fail.

### Variation and dependence

- face and global SSIM/LPIPS versus PhotoMaker output;
- CLIP/DINO/SigLIP image similarity versus PhotoMaker output;
- generated-to-reference copy-paste score;
- generated-to-same-ID target similarity;
- A/B/C/D causal crossing effects;
- actual versus all-70 BA-off;
- correct versus shuffled spatial reference;
- correct versus wrong PM ID tokens.

### R health

- raw-R ID and severe artifacts;
- raw residual/native RMS by band;
- actual routed/native RMS;
- cap firing rate;
- valid-reference attention mass;
- conditional entropy, top-2 margin and gate statistics;
- low-only/high-only outputs on the declared scope.

## 23.3 Statistical reporting

For the fixed 96 cells:

1. report mean and median paired delta;
2. report wins/ties/losses;
3. bootstrap cells with a fixed seed and publish the interval;
4. stratify by identity, prompt, face size and occlusion;
5. do not call a one-seed training comparison a population result;
6. run a second training seed before final promotion.

# 24. Promotion gates by experiment

| Experiment | Minimum technical gate | Scientific promotion gate |
|---|---|---|
| E1 FOWN | exact CL39 inference parity; coherent state across processors/ranks | >=50% fewer severe raw-R failures; actual ID >= parent -0.005; BA/shuffle causality retained |
| E2 PM dropout | PM-fused and text-only states verified; no accidental class-token deletion | stronger spatial-reference causal gap; actual quality/ID non-inferior |
| E3 valid keys | no empty supports; stable fp16/bf16 SDPA; exact off parity | less mask-area dependence and better actual ID/topology after adaptation |
| E4 reliability | zero-init exact CL39; exact parameter manifest | actual beats parent or hard cases improve without collapsing toward N |
| E5 caps | cap only tails; no NaNs; off parity | topology improves with ID loss <0.005 |
| E6 correspondence | bounded memory/runtime; cycle maps visually plausible | hard-case topology/pose improves; correct ref sensitivity positive |
| E7 ID loss | valid low-noise selection and differentiable crop; finite gradients | ID lift without copy-paste, expression or quality regression |
| E8 PM init | verified effective weights differ at step zero | more unique spatial causality and equal/better final ID |
| E9 semantic support | parser overlays and nonempty core/halo | fewer object/face fusions and no systematic accessory loss |
| E10 frequency | normalized kernels sum to one; exact shapes/padding | clearer band complementarity and actual improvement |

# 25. Decision tree

```text
1. Run all-70 BA-off and the 2 x 2 identity-source crossing.
   |
   +-- Spatial reference has little causal effect
   |     -> Prioritize E1, then E2.
   |     -> Do not tune confidence/caps first: there is too little unique signal to calibrate.
   |
   +-- Spatial reference has useful effect but raw R remains fragile
         -> Run E1.
         |
         +-- E1 improves raw R and keeps actual quality
         |     -> Run E2 and E3 as separate arms.
         |
         +-- E1 does not help
               -> Run E3 valid-key semantics before adding learned gates.

2. Once R is coherent enough:
   |
   +-- Errors are rare, high-amplitude tails
   |     -> E5 RMS caps.
   |
   +-- Errors are wrong-but-confident or band-dependent
   |     -> E4 learned reliability.
   |
   +-- Errors are pose/expression/object correspondence failures
         -> E6 DreamMatcher-lite, then E9 semantic support.

3. Once architecture is causally useful and stable:
   |
   +-- ID still plateaus
   |     -> E7 target-aligned ID loss.
   |
   +-- Outputs remain too PM-like and spatial causality is weak
   |     -> E8 PM-decoupled initialization and separate PM LR arm.
   |
   +-- Remaining problems are perceptual preferences
         -> E11 preference tuning.
```

# 26. Recommended run order and compute plan

## Stage 0 - no-training audits

1. all-70 BA-off and processor-group correction-zero;
2. 2 x 2 PM-token/spatial-reference crossing;
3. route lambda sweep;
4. PM/BA start-step sweep;
5. q-face full/binary/soft-router boundary ablation;
6. four-seed copy-paste frontier for PM, CL39-16k and CL39-24k.

These runs answer whether the main premise is correct before spending a 24k budget.

## Stage 1 - reference-route competence

1. **E1 FOWN-1** - full 24k, standard validation;
2. one confirmation seed if promoted;
3. only then **E2 PM target-condition dropout**.

## Stage 2 - reference semantics and calibration

Run independently:

1. **E3 valid-key reference attention**;
2. **E4 learned band reliability**;
3. **E5 RMS caps**.

Combine only passing components. A combined R2-D must beat the best matched single arm, not merely CL39.

## Stage 3 - correspondence and identity headroom

1. **E6 DreamMatcher-lite** at selected groups/resolutions;
2. **E7 GT-aligned ID loss** on the best architecture;
3. E9/E10 if the residual failure taxonomy points to support/frequency issues.

# 27. What not to do

1. **Do not force `C=1`.** It is already measured to reduce ID strongly and reintroduce artifacts.
2. **Do not route raw R at unit strength as the normal model.** Raw R is well outside the trained operating point.
3. **Do not train binary q-face as the main fix.** It is face-core equivalent and only boundary-local after the Gaussian split.
4. **Do not revert to CL14 hard inference replacement.** CL19 and later architectures demonstrate the value of retaining `N`.
5. **Do not remove frequency shaping because raw R looks bad.** Raw-R failures occur before the split; the bands reduce intervention strength.
6. **Do not optimize full-image distance from PhotoMaker.** This can reward background drift and artifacts.
7. **Do not bundle ownership, learned gates, caps and key masking.** The result would be uninterpretable.
8. **Do not call a cleaner raw-R panel a success if actual output collapses toward N.** Require BA-off and shuffled-reference causality.
9. **Do not load CL14 branch weights into CL39 and interpret the result as an architecture ablation.** The checkpoints co-adapted under different equations.
10. **Do not rely only on the fixed seed-0 mean.** Add seed and failure-category stratification before promotion.

# 28. Final conclusions

## 28.1 Why CL39 and PhotoMaker look similar

They are designed to. CL39 is not a replacement for PhotoMaker; it is a PhotoMaker-conditioned, PhotoMaker-initialized, native-anchored residual spatial expert. PhotoMaker begins first and remains active. The reference half is also PM-ID-conditioned. Confidence suppresses most of the correction. Therefore similar composition and many similar face decisions are the expected default.

The right question is not “why are the pixels not more different?” but “does the explicit spatial reference cause a better identity/control result than the trained native/PM path?” The selected-16 audit says yes; the full-96 attribution matrix is still required.

## 28.2 Does branched attention provide incremental improvement?

Yes, within the audited scope. Actual CL39 materially beats N-only and correction-zero, changes most face pixels and depends on confidence. CL39 also leads the fixed-96 end-to-end lineage. What remains unresolved is how much of the surplus over the original PhotoMaker checkpoint comes from explicit spatial BA rather than co-adapted PM/generic LoRAs. The all-70 and crossed-reference controls will resolve this.

## 28.3 Is R defective?

R is underconstrained, not useless. It contains structured identity signal but is not a self-sufficient face route. Current CL39 succeeds by anchoring to N, shaping frequency bands and abstaining. Improving R is worthwhile because a coherent source residual should allow stronger useful identity transfer with less suppression.

## 28.4 Best path beyond CL14 and CL39

The highest-probability path is:

```text
CL39 inference equation
+ CL14-like training ownership
+ selected PM shortcut dropout
+ true valid-key semantics
+ bounded band-specific reliability
+ correspondence-aware value transfer
+ paired target-aligned identity supervision
```

The first implementation should be **FOWN-1 / R2-A**. It is the cleanest causal test, requires no extra inference parameters or second U-Net pass, and directly targets the strongest diagnosis. The most important parallel task is the full causal attribution matrix; without it, a higher ID score alone will not prove that spatial BA became more significant.

# 29. Source ledger

## Project evidence supplied with this task

- **[P1]** `branched_attention_short_combined_E13_CL14_Hardcase_CL19_CL23_CL39(1).pdf`. Architecture map, exact processor diagrams, CL19/23/39 equations and code walkthroughs.
- **[P2]** `comet_report_PM0_CL14_CL19_CL23_CL27_CL39_reordered_appendix_23Aug2026.pdf`. Fixed-96 curves, cell grids, identity/prompt means, architecture comparison and appendix.
- **[P3]** `2026-08-25_cl39_cl14_qface_hypothesis_and_experiment_plan.pdf`. Exact q-mask algebra and FOWN plan.
- **[P4]** `2026-08-25_cl39_r_branch_artifact_diagnosis_and_r2_architecture.pdf`. Raw-R diagnosis, selected-16 interventions, R2 design and implementation ladder.
- **[P5]** `2026-08-26_CL39_R2_architecture_schemes.pdf`. CL39/R2-A/R2-B/R2-C architecture figures.
- **[P6]** `2026-08-26_ba_lineage_r_frequency_confidence_audit.pdf`. Whole-denoising lineage interventions, band analysis and confidence evidence.

## Inspected clean_new source at immutable revision

- **[C1]** [clean_new revision 728fd28f7c2c73127fa300cd04a62b29a96be918](https://github.com/kolyangg/rsrch/tree/728fd28f7c2c73127fa300cd04a62b29a96be918)
- **[C2]** [attn_processor_cleanest.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/attn_processor_cleanest.py)
- **[C3]** [hardcase_attn_processor.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/hardcase_attn_processor.py)
- **[C4]** [branched_runtime.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/branched_runtime.py)
- **[C5]** [lora2.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/lora2.py)
- **[C6]** [lora2_helpers.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/lora2_helpers.py)
- **[C7]** [e13_contract.py](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/model/photomaker_branched/e13_contract.py)
- **[C8]** [CL39 config](https://github.com/kolyangg/rsrch/blob/728fd28f7c2c73127fa300cd04a62b29a96be918/diffusion_template/src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml)

## Papers and official sources

- **[L1]** [WithAnyone: Towards Controllable and ID Consistent Image Generation](https://arxiv.org/abs/2510.14975), 2025. Official code: [Doby-Xu/WithAnyone](https://github.com/Doby-Xu/WithAnyone).
- **[L2]** [DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization](https://arxiv.org/abs/2402.09812), CVPR 2024.
- **[L3]** [IC-Portrait: In-Context Matching for View-Consistent Personalized Portrait](https://arxiv.org/abs/2501.17159), 2025.
- **[L4]** [DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](https://arxiv.org/abs/2503.06505), 2025.
- **[L5]** [InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity](https://arxiv.org/abs/2503.16418), 2025.
- **[L6]** [InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework](https://arxiv.org/abs/2504.12395), 2025.
- **[L7]** [Omni-ID: Holistic Identity Representation Designed for Generative Tasks](https://arxiv.org/abs/2412.09694), 2024.
- **[L8]** [TIDE: Achieving Balanced Subject-Driven Image Generation via Target-Instructed Diffusion Enhancement](https://arxiv.org/abs/2509.06499), 2025.
- **[L9]** [WithEveryone: Unified Planning and Identity Grounding for Group Image Generation](https://arxiv.org/abs/2608.20336), 2026.

