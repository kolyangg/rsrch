# CL39 spatial BA is causally useful, but PhotoMaker tokens remain the dominant identity source

**Date:** 27 August 2026  
**Evidence cutoff:** 07:56 BST, 27 August 2026  
**Scope:** sealed CL39 checkpoints at 16k and 24k; fixed 96-image
`manual_val` panel; one seed and one image per cell; full-96 global all-70
BA-off intervention at 16k; full-96 2 × 2 PhotoMaker-token/spatial-reference
crossing at 24k. No training state, prompt, seed, reference, box, mask,
scheduler, CFG, inference-step count, or metric definition was changed.  
**Primary metric:** subject-v2 identity cosine for the face matched to the fixed
generated-face box. Intended-versus-next-identity margin is the primary
directional diagnostic for the crossed conditions. RGB MAE and SSIM measure
intervention size, **not quality**.  
**Immutable experiment:** all validation-only arms inherit CL39 Comet key
`b1ca0b3da679401c85b991f1bbdf0b2a`; they did not create new Comet experiments.

| Arm | Condition | Key | Step | Intended ID ↑ | Next/wrong ID |
|---|---|---|---:|---:|---:|
| 16k actual | trained CL39 equation | `b1ca0b3da679401c85b991f1bbdf0b2a` | `16000` | **`0.56732`** | `-0.00088` |
| 16k BA-off | all 70 BA corrections disabled; trained adapters retained | `b1ca0b3da679401c85b991f1bbdf0b2a` | `16000` | `0.53402` | `-0.00126` |
| A | correct PM tokens / correct spatial reference | `b1ca0b3da679401c85b991f1bbdf0b2a` | `24000` | **`0.56396`** | `0.00176` |
| B | correct PM tokens / next-identity spatial reference | `b1ca0b3da679401c85b991f1bbdf0b2a` | `24000` | `0.53445` | `0.01180` |
| C | next-identity PM tokens / correct spatial reference | `b1ca0b3da679401c85b991f1bbdf0b2a` | `24000` | `0.01684` | `0.43970` |
| D | next-identity PM tokens / next-identity spatial reference | `b1ca0b3da679401c85b991f1bbdf0b2a` | `24000` | `-0.00046` | `0.46330` |

## Executive conclusion

The missing causal controls now resolve the main CL39 attribution question.
**Explicit spatial branched attention contributes real, positive identity
signal on the complete fixed panel**, while **PhotoMaker identity tokens remain
the dominant source of which person is generated**. `[measured]`

At 16k, normal CL39 beats the same checkpoint with all 70 BA corrections
disabled by `+0.03330` intended-ID, wins `77/96` paired cells, and has a
fixed-cell bootstrap interval `[+0.02424,+0.04258]`. Because the PM and generic
LoRAs remain loaded, this is a clean causal estimate of the explicit BA
correction at the successful 16k operating point. `[measured]`

At 24k, replacing only the spatial reference with the next identity reduces
intended-ID by `0.02951` when PM tokens are correct (`A−B`) and by `0.01731`
when PM tokens are wrong (`C−D`). Both intervals are strictly positive, and
the effects win `76/96` and `75/96` cells respectively. The spatial reference
also improves intended-minus-wrong identity margin by approximately `0.040`
under either PM condition. This is direct evidence that the explicit spatial
lane is read rather than ignored. `[measured]`

However, changing PM tokens from correct to the next identity has a far larger
effect: `A−C=+0.54712` and `B−D=+0.53491` intended-ID, winning all `96/96`
cells in both comparisons. With wrong PM tokens, outputs score much closer to
the wrong identity (`0.43970–0.46330`) than the intended identity
(`-0.00046–0.01684`). Representative images visibly follow the adversarial PM
identity even when the spatial reference remains correct. `[measured][visual]`

The earlier report's caution—that CL39 was the best full system but its
full-96 spatial-BA share was not yet isolated—is now superseded by measured
evidence: the share is positive and repeatable across this fixed panel, but it
is incremental around a PM-dominated identity trajectory. This supports the
current E2 target-PM-dropout training experiment; it does **not** support
removing PM conditioning or replacing CL39's native anchor. `[decision]`

![Aggregate intended and wrong identity scores](assets/cl39_attribution_controls_20260827/identity_summary.png){ width=94% }

*Figure 1. Correct PM tokens determine the generated identity; the spatial
reference adds a smaller but measurable shift toward its identity.*

## 1. Exact experiment and integrity gates

### 1.1 Fixed controls

Both jobs used RealVisXL V4.0, DDIM50, CFG 5, seed 0, the fixed 96 prompts,
references, generated/reference boxes and masks, one image per cell,
`pose_adapt_ratio=0`, and `ca_mixing_for_face=false`. The trained generic and
PhotoMaker LoRAs remained loaded. All arms are whole-denoising interventions.
`[record]`

The 16k checkpoint SHA-256 is
`a598b929e4fbfab7eac0f9474c9c96d1713dbac6224e1de6ffbca4b43ae29e86`;
the 24k checkpoint SHA-256 is
`74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07`.
The final MLS jobs were:

- 16k actual plus all-70 BA-off:
  `lm-mpi-job-8a1e80fe-4ae5-4ae1-b3a9-f7ab2c8e945f`;
- 24k 2 × 2 identity crossing:
  `lm-mpi-job-b890b097-0eab-40c0-854e-999e54617119`.

Each actual arm reproduced the sealed checkpoint output exactly on `96/96`
images: mean RGB MAE `0` and maximum per-image RGB MAE `0`. Only after this
gate passed did its counterfactual arms run. Final state is six accepted arms
× 96 images = `576` images and six gate JSONs. Earlier unified-runtime drift
and hook-placement attempts are archived failures, not inputs to this report.
`[measured][record]`

### 1.2 The interventions

The all-70 arm sets every explicit BA correction to native mode while
retaining all compatible trained adapters. The 24k crossing holds prompt,
seed, model weights, boxes and denoising controls fixed, then uses the next
identity in sorted fixed-panel order as the deliberately wrong condition:

| | Correct spatial reference | Next-identity spatial reference |
|---|---|---|
| Correct PM identity tokens | A | B |
| Next-identity PM tokens | C | D |

The analysis joined generated PNGs to bbox metadata after normalizing spaces
to underscores on **both** sides. Every arm joined `96/96`; a literal join
would have silently dropped most cells. `[record]`

## 2. The all-70 control proves the explicit BA correction helps at 16k

| Comparison | Mean Δ ID | Median Δ | W/T/L | Fixed-cell bootstrap 95% interval |
|---|---:|---:|---:|---:|
| 16k actual − all-70 BA-off | **`+0.03330`** | `+0.02784` | `77/0/19` | **`[+0.02424,+0.04258]`** |

The actual aggregate is `0.56732`, versus `0.53402` with all 70 corrections
disabled. The identity-margin effect is similarly positive: `+0.03293`, with
interval `[+0.02174,+0.04404]` and `70/96` wins. Next-identity score itself is
unchanged within uncertainty (`+0.00037`, interval
`[-0.00669,+0.00758]`), so the gain is an intended-identity improvement rather
than a generic embedding-score shift. `[measured]`

The intervention is not pixel-trivial: BA-off differs from actual over
`94.31%` of fixed face-crop pixels above `1/255`, with face RGB MAE `0.04857`
and global SSIM `0.90017`. Those numbers prove output influence, not visual
quality. `[measured][limitation]`

![All-70 paired identity effects](assets/cl39_attribution_controls_20260827/all70_scatter.png){ width=94% }

*Figure 2. Most fixed cells lie above the equality line; the negative tail
shows that BA is helpful on average rather than universally per prompt.*

The mean gain is positive in seven of eight identity strata. It is strongest
for Marion (`+0.06575`) and Eddie (`+0.06394`), and effectively neutral for
Jisoo (`-0.00030`). This heterogeneity is why the complete paired panel is more
informative than a single aggregate. `[measured]`

## 3. The 2 × 2 crossing separates spatial BA from PhotoMaker identity

### 3.1 Causal effects on intended identity

| Contrast | Meaning | Mean Δ ID | W/T/L | Fixed-cell bootstrap 95% interval |
|---|---|---:|---:|---:|
| A − B | correct versus wrong spatial, PM correct | **`+0.02951`** | `76/0/20` | **`[+0.02222,+0.03673]`** |
| C − D | correct versus wrong spatial, PM wrong | **`+0.01731`** | `75/0/21` | **`[+0.01226,+0.02244]`** |
| A − C | correct versus wrong PM, spatial correct | **`+0.54712`** | `96/0/0` | **`[+0.52722,+0.56606]`** |
| B − D | correct versus wrong PM, spatial wrong | **`+0.53491`** | `96/0/0` | **`[+0.51610,+0.55277]`** |

![Paired causal effects](assets/cl39_attribution_controls_20260827/paired_effects.png){ width=94% }

*Figure 3. Both spatial-reference effects are positive, but the adversarial PM
identity switch is roughly an order of magnitude larger.*

The identity margin makes the source direction especially clear:

| Contrast | Mean Δ intended−wrong margin | W/T/L | Bootstrap 95% interval |
|---|---:|---:|---:|
| A − B | `+0.03955` | `79/0/17` | `[+0.03069,+0.04837]` |
| C − D | `+0.04091` | `80/0/16` | `[+0.03281,+0.04918]` |
| A − C | `+0.98505` | `96/0/0` | `[+0.93673,+1.03156]` |
| B − D | `+0.98641` | `96/0/0` | `[+0.93850,+1.03210]` |

Correct spatial conditioning therefore pushes the output toward the intended
identity by almost the same margin under correct and wrong PM tokens. Its
factorial interaction on identity margin is only `-0.00135`. This is evidence
of a stable spatial signal. Yet the adversarial PM switch changes the margin
by about `0.985`, roughly 25 times the `0.040` spatial effect in this protocol.
That ratio describes this next-identity intervention; it is not a general
percentage decomposition of CL39. `[measured][limitation]`

### 3.2 Spatial BA both raises intended ID and resists the wrong identity

With correct PM tokens, replacing the correct spatial reference lowers
intended-ID by `0.02951` and raises wrong-ID attraction by `0.01004`. With
wrong PM tokens, restoring the correct spatial reference raises intended-ID by
`0.01731` and lowers wrong-ID attraction by `0.02360`. Thus the spatial-margin
gain is not merely metric noise on the target embedding: it moves both sides
of the intended-versus-wrong contrast in the expected direction.
`[measured]`

All six arms detect a face in the target box for `96/96` cells, with zero
unowned and zero ambiguous selections. Mean fixed-box overlap is
`0.8523/0.8487` for A/B and `0.7809/0.7832` for C/D. The wrong-PM intervention
degrades alignment somewhat, but it does not create missing-face failures that
could explain the identity collapse. `[measured]`

### 3.3 Heterogeneity

| Identity | 16k actual−BA-off | 24k A−B spatial | 24k C−D spatial | 24k A−C PM |
|---|---:|---:|---:|---:|
| Eddie | `+0.06394` | `+0.05038` | `+0.01331` | `+0.54258` |
| Elon | `+0.04378` | `+0.02171` | `+0.01725` | `+0.55854` |
| Jennie | `+0.00605` | `+0.03887` | `+0.02160` | `+0.60356` |
| Jensen | `+0.03238` | `+0.02317` | `+0.00911` | `+0.53318` |
| Jisoo | `-0.00030` | `+0.01976` | `+0.01179` | `+0.58581` |
| Keanu | `+0.03309` | `+0.04742` | `+0.01957` | `+0.57578` |
| Lex | `+0.02171` | `+0.01835` | `+0.02955` | `+0.46178` |
| Marion | `+0.06575` | `+0.01646` | `+0.01627` | `+0.51574` |

The correct-spatial effect is positive for every identity mean in both PM
conditions. It also remains positive across small, medium and large fixed-box
terciles: A−B is `+0.02356/+0.03387/+0.03111`; C−D is
`+0.01324/+0.01500/+0.02367`. `[measured]`

Prompt means are less uniform. A−B is negative for Skiing woman (`-0.01700`)
and Crying woman (`-0.00806`), while it is strongest for Dancing woman
(`+0.06185`), Laughing woman (`+0.05374`), Skiing man (`+0.05302`) and Crying
man (`+0.05111`). C−D is negative for Angry woman (`-0.01503`) and Drumming
woman (`-0.01496`). These are fixed-cell diagnostics, not demographic claims;
there are only eight identities and one seed. `[measured][limitation]`

![Identity-stratified effects](assets/cl39_attribution_controls_20260827/identity_strata.png){ width=94% }

### 3.4 Matched visual inspection

![Representative 2 × 2 crossings](assets/cl39_attribution_controls_20260827/representative_crossing_grid.png){ width=98% }

*Figure 4. Rows include the largest A−B, largest C−D, largest A−C, and lowest
A−B cells. Red boxes are the immutable target-face boxes; labels show intended
and next/wrong identity scores.*

In these preselected metric-extreme cells, A and B retain the target person and
mostly preserve composition; changing the spatial reference changes facial
details without replacing the subject. C and D visibly follow the wrong PM
identity even when C retains the correct spatial reference. The lowest A−B
cell demonstrates the counterexample: the correct spatial reference does not
improve every individual image. This visual inspection supports the causal
direction and the PM-dominance conclusion, but is not a complete topology or
quality rubric over all 96 cells. `[visual][limitation]`

## 4. Root cause and interpretation

CL39 is architecturally PM-anchored. PM identity is fused into the target text
condition before the U-Net; the explicit spatial lane supplies a routed
reference correction around the native target message rather than replacing
that native trajectory. PM conditioning also begins before BA in the sealed
schedule. The measured ordering—large PM effect plus smaller positive spatial
effect—is therefore consistent with the implemented equation rather than a
surprise failure. `[code][report]`

The controls rule out three simpler explanations:

1. **The branch is not inactive.** Disabling all 70 corrections lowers ID by
   `0.03330`, while changing the spatial identity lowers ID under both PM
   conditions. `[measured]`
2. **The effect is not an inference replay artifact.** Both actual arms match
   their sealed endpoints at exactly zero RGB error. `[measured]`
3. **The wrong-PM collapse is not caused by missing or off-mask faces.** Every
   cell has an owned, unambiguous detected face. `[measured]`

The correct interpretation is not “CL39 is only PhotoMaker” and not “spatial
BA owns identity.” CL39 uses the explicit spatial route for an incremental,
quality-relevant identity correction, while PM tokens set the dominant
identity trajectory. `[decision]`

### Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Explicit all-70 BA correction improves 16k intended identity | High | matched `96` cells, `77` wins, positive fixed-cell interval, exact replay gate |
| Spatial-reference identity is causally read at 24k | High for this panel | A−B and C−D are positive with `76/96` and `75/96` wins; intervals exclude zero |
| PM tokens dominate adversarial identity choice | High for this protocol | A−C and B−D win `96/96`; wrong-PM arms score near the wrong identity and look wrong in selected grids |
| Spatial effect generalizes over identities and face sizes | Moderate-high | positive in all eight identity means and all three box-size terciles, but one seed |
| Spatial BA improves prompt adherence or face quality | Not established | text and compact face-quality metrics were disabled for these jobs |
| The measured ratio generalizes to ordinary non-adversarial use | Not established | “wrong PM” is a strong next-identity intervention, not PM absence or natural noise |

## 5. What is not established

- The 16k and 24k aggregate means must not be compared as if they were the
  same checkpoint. Only within-step contrasts are causal.
- One fixed seed supports paired causality on these 96 cells, not population or
  training-seed uncertainty.
- C is **not** “spatial BA alone”: it keeps a deliberately wrong PM identity
  active. It measures whether correct spatial evidence can resist an
  adversarial PM condition.
- RGB distance and SSIM do not establish better faces, topology, prompt
  adherence, or aesthetics.
- The controls do not measure copy-paste, expression transfer, or diversity.
- The experiment does not justify removing PM, the native N anchor, frequency
  shaping, or confidence. It instead motivates training the spatial lane to
  carry more unique identity information without losing the successful A
  operating point.

## 6. Proposed experiments

### Priority 1 — evaluate the already-running E2 checkpoint

- **Config/evaluation:** `CL39R2D_cosmic_pm_target_condition_dropout_24k`
  plus `CL39R2D_attribution_crossing_16k_24k`.
- **Single scientific change:** evaluate the E2 checkpoint trained with
  target-only PM-condition dropout; keep this causal protocol identical.
- **Hypothesis:** E2 increases A−B and C−D identity-margin effects beyond the
  CL39 baselines `0.03955/0.04091` while preserving the A operating point.
- **Risk:** reducing the PM shortcut may lower actual ID, text adherence, or
  face quality.
- **Decision gate:** promote only if A intended-ID is no worse than matched-step
  CL39 by more than `0.01`, both spatial-margin effects increase with positive
  intervals, ownership failures remain zero, and compact face quality, text
  adherence, and topology are non-inferior.

### Priority 2 — repeat the crossing across inference seeds

- **Config/evaluation:** `CL39_24k_identity_crossing_multiseed_s1_s3`.
- **Single scientific change:** repeat the sealed 24k 2 × 2 crossing at seeds
  `1`, `2`, and `3`.
- **Hypothesis:** spatial effects remain positive and PM dominance remains
  qualitatively stable across seeds.
- **Risk:** extra inference cost and wider variance may weaken the fixed-seed
  conclusion.
- **Decision gate:** require positive pooled intervals for A−B and C−D,
  consistent direction in every seed aggregate, and no new face-ownership
  failure tail.

No new training arm should be added before E2 reaches the declared checkpoint.
E1/E3/E4/E5 should later receive the same causal panel so promotion compares
actual quality and spatial dependence, not only ordinary endpoint ID.
`[decision]`

## 7. Implementation plan

1. Preserve this baseline package, gate JSONs, per-image CSV and bootstrap seed.
2. At E2 step 16k, load the immutable checkpoint and run actual/all-70 plus the
   same A/B/C/D conditions with all validation controls unchanged.
3. Fail closed unless actual reproduces its ordinary validation output within
   the same `0.002` maximum per-image RGB-MAE threshold; prefer exact zero.
4. Score intended ID, next-identity attraction, identity margin, mask ownership,
   compact face quality, text similarity, topology rubric, and copy-paste.
5. Compare E2 contrasts to this CL39 baseline at matched steps. Do not compare
   E2 16k against CL39 24k.
6. If E2 passes, repeat with seeds `1–3`; otherwise retain CL39's PM-anchored
   balance and use E1/E3 route-health evidence before changing architecture.

## 8. Reproducing

From the repository root, the accepted Serv images were copied without
modifying the server task and scored locally:

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

PYTHONPATH=. python tools/analysis/analyze_cl39_attribution_controls.py \
  --task-root artifacts/cl39_attribution_controls_20260827/serv_task \
  --bbox-json ../dataset_full/val_dataset/pm96_bboxes_new.json \
  --reference-root ../dataset_full/val_dataset/references \
  --subject-v2-embeds ../dataset_full/val_dataset/id_embeds_manual_val_subject_v2.pth \
  --output-root artifacts/cl39_attribution_controls_20260827/scored \
  --device cpu
```

The scorer writes per-image and aggregate CSVs, paired effects with `100,000`
fixed-cell bootstrap draws (seed `390027`), identity/prompt/face-size strata,
summary JSON and the figures used here. The selected-16 A-arm mean is
`0.557538`, reproducing the prior audit's `0.55754` and independently checking
the metric contract. `[measured][record]`

Important trap: bbox keys contain spaces while PNG names contain underscores.
The scorer normalizes `name.replace(" ", "_")` on both sides and fails unless
all `96` cells exist in every arm. The main machine-readable outputs are:

- `artifacts/cl39_attribution_controls_20260827/scored/summary.json`
  
  SHA-256: `e041f90f4e60806a10bafcd404e3b3c285ac05a7d182ceacb0dcbd8b75059cc0`
- `artifacts/cl39_attribution_controls_20260827/scored/per_image.csv`
  
  SHA-256: `aefdd592e6cb53cbbba6f673050aa142bb3c620b7fa5715da4c3bbde17f1b768`
- `serv_run_packages/CL39_attribution_controls_20260826_r1/submission.json`;
- `artifacts/cl39_attribution_controls_20260827/serv_task/gates/`.

## 9. References

1. `analysis/blueprints/26Aug/CL39_branched_attention_investigation_2026-08-26.md` — preregistered all-70 and 2 × 2 attribution design.
2. `analysis/2026-08-26_ba_lineage_r_frequency_confidence_audit.md` — prior selected-16 causal audit and subject-v2 measurement contract.
3. `docs/handoffs/LATEST.md` — immutable run IDs, source-replay history and final job state.
4. `src/model/photomaker_branched/attn_processor_cleanest.py` — CL39 spatial correction, frequency and confidence equation.
5. `src/pipelines/photomaker_branched_clean.py` and `src/trainer/sdxl_trainers.py` — validation-only PM/spatial source crossing.

## Bottom line

CL39's explicit spatial BA is **causally active and beneficial**: it contributes
about `+0.03` intended-ID at the successful operating point and pushes identity
margin toward the spatial reference under both PM conditions. PhotoMaker tokens
still decide the person much more strongly. The correct next question is no
longer whether BA is used; it is whether E2 can increase the spatial lane's
unique contribution without sacrificing CL39's actual face quality, prompt
control, topology, or identity.
