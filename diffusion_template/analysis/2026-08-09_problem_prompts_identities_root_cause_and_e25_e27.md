# Problematic prompts and identities: one bug now fixed, two prompt-level limits, and one retraction

**Date:** 9 August 2026 · **revised 10 August 2026**
**Scope:** every problematic prompt and identity on the fixed 96-image `manual_val`
panel, across three training datasets — large_dataset (E13), BigCelebs (BC_E13)
and cosmic_large (CL4/CL8/CL9/CL10/CL11). Per-image identity, no-reference face
quality, face detection, mask geometry, mask-boundary continuity, and a
per-identity consistency decomposition over `670` re-embedded generations.
No running job was touched; no training code was changed for this analysis.
**Evidence cutoff:** E13, BC_E13, CL4, CL8, CL9 complete at 24k; CL10 (14k) and
CL11 (16k) still running, compared at the matched step 14,000. Prompt-family,
face-quality and seam measurements (§3-§4) use the **pre-fix** panel export.

> **Revision note (10 Aug).** Two things in the 9 August version were wrong and
> are corrected here.
> **(1)** The eddie reference bug has since been **fixed and verified** — the
> conditioning hypothesis in §2 is now confirmed by intervention rather than
> inferred, and eddie is no longer a problem identity.
> **(2)** The claim that jennie and jisoo were "merging into a shared face" is
> **retracted** — §1.1. Rank-1 identification is `99.1%` and no jennie image is
> closer to jisoo's reference than to her own. The proposed experiment built on
> it (hard-negative identity batching) is **withdrawn**, on two independent
> grounds set out in §1.1.

| Arm | Comet key | Dataset | panel `id_sim` @24k | eddie @24k |
|---|---|---|---:|---:|
| **E13** (base) | `1cc0a02371094b24a6a02a4cc649f10c` | large_dataset | `.39980` → **`.43040`** | `.176` → **`.4254`** |
| BC_E13 | `c138db7c41ae435c8a7560f40cf5f58d` | BigCelebs | `.38943` → **`.41520`** | `.158` → **`.3540`** |
| CL9 | `81bb311ed70545eda3281c64bc48be47` | cosmic_large | `.41513` → **`.44800`** | `.159` → **`.4155`** |
| CL4 | `0dd86b436b224f939efa3887ad6acbe2` | cosmic_large | `.40387` *(pre-fix)* | `.157` |
| CL8 | `a6b5970aa1a24d3490ad08e7994b5f1e` | cosmic_large | `.40171` *(pre-fix)* | `.163` |
| CL10 *(running)* | `eba0187806ec476996f5ea4af356361e` | cosmic_large | `.39900` @14k | — |
| CL11 *(running)* | `32f4ba2a3b3a493f96a3a2345147e84c` | cosmic_large | `.40594` @16k | — |

Arrows are pre-fix → post-fix, where the fix is the corrected subject-face
identity vector plus regenerated eddie validation images (§2). The remaining
arms are being backfilled the same way.

**Metrics used, and why more than one.** `id_sim` (ArcFace cosine) is the project
default and is blind to artefacts. **TOPIQ-Face** (already logged per image) is
blind to *who* it is. **`identity_lock = sim_self - sim_ref`** is introduced
here: `sim_ref` is the mean cosine between an identity's generations and its
reference, `sim_self` the mean pairwise cosine among that identity's own
generations. A large positive gap means the model renders one consistent person
who is not the reference — which is exactly what eddie showed before the fix, and
what nobody else shows after it.

---

## Executive conclusion

**The one genuine bug found in this analysis has been fixed, and the fix is
confirmed by intervention.** `references_actual/eddie.webp` contains two
detectable faces; InsightFace returns them in descending detection score and
every consumer in this repo took `faces[0]` — the blurred bystander at the right
border, not Eddie. That vector was used both to **score** and to **condition**.
After re-selecting the subject face by maximum IoU against the declared box in
`ref_bboxes.json` and regenerating eddie's validation images, at E13 @24k:

| eddie, E13 @24k | vs subject | vs bystander | `sim_self` | `identity_lock` |
|---|---:|---:|---:|---:|
| before (bystander vector) | `0.065` | **`0.176`** | `0.341` | **`+0.257`** |
| after (subject vector) | **`0.425`** | `0.061` | `0.415` | **`-0.010`** |

The relationship inverts cleanly. Before, the generations matched the bystander
better than Eddie; after, the reverse, and eddie moves from the worst identity in
the panel to mid-pack, with an `identity_lock` gap of `-0.010` — the "renders the
subject" regime occupied by elon, keanu and jensen. Panel means rise `+.026` to
`+.033`.

**What remains is not an identity problem.** With eddie fixed, rank-1
identification across the panel is `99.1%` (581/586 generated faces closest to
their own reference out of eight; chance `14.3%`). The identity mechanism works.
Three things remain, in descending order of how much they should worry us:

1. **The small-face family — a resolution floor that costs quality *and*
   identity.** Jumping and Dancing require a face of `~108 px` in a 1024² output
   — **13.5 latent cells**. Below `110 px` TOPIQ-Face is `0.478` against `0.694`
   above (`r = +0.82` with pixel size), and the same images lose **`-0.292`** of
   their identity relative to that identity's own clean renders — worst for lex at
   `-0.473`. The pooled `corr(face_px, id_sim) = +0.065` in the 9 August version
   understates this: the relationship is non-monotonic, and the large-face Skiing
   images at the other end cancel it (§4). `Jumping_ma_jensen` still scores
   `id_sim .502` with TOPIQ-Face `.546`, so the metric under-reports the family
   either way. The deficit is **inherited from the base model** — at step 0, BA
   effectively off, it already sits at `0.490`.

2. **Apparent age is wrong for two of eight identities.** jennie (reference ~22)
   renders as a **child** in E13, BC_E13, CL8 and CL9; lex (~32) renders roughly
   twenty years older everywhere. Both at `id_sim` `.32-.53`, so the metric is
   satisfied. The other six identities are age-faithful. This is a visual finding,
   not yet quantified — see §1.2 for why the available age estimator cannot be
   trusted for it.

3. **Occlusion splits into two regimes, and only one is unwinnable.** Comparing
   each generation with that identity's *own clean renders* in the same run
   separates them. **Crying** barely touches the underlying face (`-0.124`): the
   model draws the right person and `id_sim` cannot read them through hands and
   closed eyes — a measurement problem, fix the protocol. **Skiing** is different
   (`-0.322`, and `-0.54`/`-0.71` for marion and jisoo): those renders are *not*
   the same person the model draws on clean prompts. The mechanism is visible and
   almost binary — six of eight identities get goggles pushed onto the forehead
   with eyes visible and identity intact at `0.51-0.82`; marion and jisoo get them
   pulled down over the eyes and collapse to `0.24` and `0.19`. That is a
   composition decision, and it is worth an intervention. §3 revises the
   9 August conclusion that no architecture change could help here.

**Corrections to the 9 August CL8-CL11 report also stand:** it grouped the
residual hard cases under "occlusion and extreme expression", which is right for
Skiing/Crying and wrong for Jumping/Dancing — those are the worst-*quality*
images in the panel and are not occluded at all.

**What helps, measured.** On the small-face family CL8 (full-body targets) is the
only clear winner (`.3804` vs E13's `.3429`) and the only arm still improving at
24k. On occluded and clean families CL9 leads (`.4336` / `.4973`). No arm moves
the mask-boundary seam (`4.2-4.8` ΔE, identical across all five).

---

## 1. Identity: the mechanism works

Every generated image's largest detected face was re-embedded (`670` images
across 7 arm/steps) and compared with its own identity's reference and with that
identity's other generations.

### Rank-1 identification

| identity | n | rank-1 correct | mean margin over the best wrong reference | min margin |
|---|---:|---:|---:|---:|
| elon | 84 | 84/84 | `+0.411` | `+0.250` |
| jensen | 84 | 84/84 | `+0.471` | `+0.191` |
| keanu | 84 | 84/84 | `+0.374` | `+0.110` |
| jennie | 84 | 84/84 | `+0.351` | `+0.096` |
| lex | 84 | 83/84 | `+0.308` | `-0.001` |
| marion | 84 | 83/84 | `+0.242` | `-0.009` |
| jisoo | 82 | 79/82 | `+0.330` | `-0.067` |
| **overall** | **586** | **`99.1%`** | | |

The five misses are all images where the identity signal is ~zero to begin with
(own cosine `0.025`-`0.123`): three occluded jisoo, one small-face lex, one
small-face marion. Those are ties between two near-zero numbers, not confusions.

### `identity_lock` by identity

| identity | `sim_ref` | `sim_self` | gap | reading |
|---|---:|---:|---:|---|
| **eddie**, after fix (E13) | **`0.425`** | `0.415` | **`-0.010`** | renders the subject |
| elon | `0.491` | `0.503` | `+0.013` | renders the subject |
| keanu | `0.460` | `0.466` | `+0.005` | renders the subject |
| jensen | `0.506` | `0.559` | `+0.053` | renders the subject |
| marion | `0.308` | `0.403` | `+0.096` | weakest signal (§1.3) |
| lex | `0.379` | `0.493` | `+0.114` | age drift (§1.2) |
| jisoo | `0.418` | `0.617` | `+0.199` | see §1.1 |
| jennie | `0.433` | `0.692` | `+0.258` | age drift (§1.2) |
| *eddie, before fix* | *`0.072`* | *`0.329`* | *`+0.257`* | *wrong vector conditioned in* |

### 1.1 Retracted: jennie and jisoo are **not** merging

The 9 August version reported that jennie and jisoo were converging on a shared
face, on the basis that the mean pairwise cosine between their generations is
`+0.265` against a cross-identity mean of `+0.041`, while their references are
only `+0.055` apart. That framing does not survive testing:

| test | result |
|---|---|
| jennie images closer to jisoo's reference than to jennie's | **0 / 84** |
| jisoo images closer to jennie's reference than to jisoo's | **0 / 82** |
| jennie's mean cosine — own reference vs jisoo's | `0.433` vs `0.051` |
| jennie×jennie (same person) vs jennie×jisoo | `0.661` vs `0.265` |
| jennie×jisoo pairs above a same-person threshold of `0.40` | `6.8%` |
| above `0.50` | `0.3%` |

Two reasons the original claim was unsound. **First, the baseline was wrong.**
ArcFace embeddings encode ethnicity, apparent age, hair and lighting alongside
identity. jennie and jisoo share all of those, plus identical prompts and
identical seeds — every comparison is the *same scene* with a different
reference. The control that would have justified the claim is the ArcFace cosine
between two *real* young Korean women photographed similarly, which I never
measured. **Second, part of the elevation has a simpler explanation**: both
render younger than their references (§1.2), and child-like face geometry is
itself a shared attribute.

The proposal built on this, hard-negative identity batching, is **withdrawn**.
Independently of the premise, the mechanism would not have worked:
`_masked_face_mse` sums per-sample MSE and divides by the count
([`diffusion_loss.py:7-32`](../src/loss/diffusion_loss.py#L7-L32)), the UNet uses
GroupNorm rather than BatchNorm, and attention is within-sample — so `∇L₁` is
completely independent of which identity occupies the other batch slot. Changing
*who is batched together* changes gradient noise, not the expected update.

What does survive is narrower: **the model does broadly preserve the geometry
between identities.** Excluding the jennie-jisoo pair, `corr(reference distance,
generated distance)` across the other 20 pairs is `+0.508` and mean inflation is
`+0.036`.

### 1.2 Apparent age is wrong for jennie and lex

![Reference against generation for all eight identities](assets/hardcase_20260809/fig8_age_drift.png)

Same prompt, same seed, E13 @24k. Six of eight identities are age-faithful.
Two are not:

- **jennie**, reference ~22, renders as a child of roughly ten — in **E13,
  BC_E13, CL8 and CL9**, i.e. every dataset. `id_sim` `.350`-`.529`.
- **lex**, reference ~32, renders as a gaunt man of roughly fifty with a receding
  hairline and deep nasolabial folds — in every arm. `id_sim` `.317`-`.451`.

This is the same *class* of failure as the small-face artefacts in §4: the face
is the right person by ArcFace and wrong to a human. It is also consistent with
lex's `identity_lock` of `+0.114` and jennie's `+0.258` — self-consistent renders
carrying a systematic offset from the reference.

**Not quantified, deliberately.** InsightFace's `genderage` model scores jennie's
renders at `+2.1` years' drift, which is plainly wrong given the images, so it
cannot be used for this. It does report `lex +18.9` and pre-fix `eddie +25.1`,
both of which match the visuals — but an estimator that fails on the clearest
case is not a measurement instrument. §8 proposes quantifying this properly
before designing any arm around it.

One lead worth recording: **CL11, the three-reference arm, is the only one that
renders an adult jennie.** That is a single cell, not a result, but more
reference photos plausibly supply more evidence about age and build.

### 1.3 marion: normalising the same reference does not help

marion is lowest on both axes (`sim_ref .308`, `sim_self .403`) and lowest on
every prompt family, including clean portraits (`.375` against `.434-.537` for
everyone else). She is not confused with anyone (best competing reference
`+0.036`), and her renders are recognisably her.

The only property that singles her out is head pose. A 5-point landmark yaw proxy
gives marion `0.368` against `≤0.093` for all seven others — the only
substantially off-axis reference, at `4x` the next highest, with `-7.6°` of roll.

#### The normalisation experiment — a negative result

Six identity vectors were built **from marion's existing reference file** and
CL9's twelve marion generations at 24k re-scored against each. Nothing was
regenerated, so this isolates "is the target vector a poor summary of her" from
"is the model drawing her badly". The control is the same six variants applied to
the other seven identities.

| identity | raw | hflip | roll-corrected | multi-crop | flip-average | full TTA | best gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| **marion** | `0.3013` | `0.2975` | **`0.3103`** | `0.3084` | `0.3025` | `0.3068` | **`+0.0091`** |
| keanu | `0.4995` | `0.4828` | `0.5060` | **`0.5160`** | `0.4964` | `0.5053` | `+0.0165` |
| jisoo | `0.4394` | `0.4272` | `0.4399` | **`0.4504`** | `0.4387` | `0.4434` | `+0.0110` |
| jennie | `0.4616` | `0.4392` | `0.4697` | **`0.4699`** | `0.4544` | `0.4622` | `+0.0083` |
| eddie | `0.4155` | `0.4170` | `0.4168` | `0.4175` | `0.4194` | **`0.4195`** | `+0.0041` |
| lex | `0.4205` | `0.4141` | **`0.4227`** | `0.4148` | `0.4206` | `0.4207` | `+0.0022` |
| jensen | **`0.5585`** | `0.5383` | `0.5521` | `0.5581` | `0.5527` | `0.5554` | `-0.0004` |
| elon | **`0.4800`** | `0.4697` | `0.4760` | `0.4735` | `0.4784` | `0.4781` | `-0.0016` |
| *mean gain, excluding marion* | | `-0.0124` | `+0.0012` | `+0.0036` | `-0.0020` | `+0.0014` | |

**Roll correction is marion's best variant and buys `+0.0091`** — `0.3013 →
`0.3103`. keanu, whose reference is already frontal, gains almost twice as much
(`+0.0165`) from plain multi-crop averaging. Every gain here is generic
test-time augmentation, at the scale of `+0.00` to `+0.02` for everyone; none of
it is pose repair, and none of it moves marion out of last place.

**Conclusion: there is no easy scoring-side normalisation of this file.** Her
embedding is not degenerate — it is stable to `0.9575` under h-flip and reproduces
her reference exactly (cosine `1.0000` in the v2 rebuild). The target vector is a
faithful summary of *that photo*; the photo is simply a 3/4 view of her.

#### What is still worth testing, and why

The experiment above only normalises the vector used for **scoring**. The
conditioning side is untouched and is where an off-axis reference plausibly
costs more: PhotoMaker's CLIP encoder and the BA reference crop both receive a
3/4 face and must synthesise the frontal view the prompts ask for.

That reference is also near the tail of what the model is trained to consume.
On the locally available large_dataset sample, `|yaw|` over images that can be
drawn as a reference:

| p10 | p25 | p50 | p75 | p90 | p95 | p99 | share ≥ marion's `0.368` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.012` | `0.031` | `0.127` | `0.196` | `0.253` | `0.305` | `0.441` | **`2.6%`** (1/38) |

marion's reference sits at roughly the **97th percentile** of training reference
pose. `n = 38` is underpowered — the full corpus is on Serv — but it is the only
measured property that makes her an edge case, and it is the one the user wants
the model to handle.

**Recommended next step is a generation test, not a scoring test:** regenerate
marion's twelve panel images with a roll-corrected, tightly-aligned crop of the
same file supplied as the *conditioning* reference, leaving the scoring vector
untouched. If `id_sim` moves, the deficit is in how the branch consumes an
off-axis reference and the fix is reference preprocessing. If it does not, the
reference photo itself is the limit and only a second photo will help — which is
the comparison already planned. Cost is 12 images of inference on one GPU.

### 1.4 Per-identity, per-family

Mean `id_sim` over the five completed arms at 24k (pre-fix export):

| identity | small-face | occluded | clean | all |
|---|---:|---:|---:|---:|
| jensen | **`.488`** | `.489` | `.537` | **`.513`** |
| elon | `.398` | `.489` | `.533` | `.496` |
| keanu | `.318` | `.465` | `.517` | `.466` |
| jennie | `.342` | `.440` | `.486` | `.447` |
| jisoo | `.341` | **`.331`** | `.473` | `.404` |
| lex | **`.264`** | `.408` | `.434` | `.397` |
| marion | `.283` | `.294` | **`.375`** | `.333` |

Each weak identity is weak differently: jisoo collapses on **occlusion** (Skiing
woman `.120`, the worst cell in the panel); lex on **small faces** (`.264`);
marion uniformly. jensen's `.488` on small-face is a striking outlier — see §5.

---

## 2. The eddie reference bug: confirmed, and fixed

![eddie before and after the identity-vector fix](assets/hardcase_20260809/fig7_eddie_before_after.png)

Same prompts, same seeds; only the identity vector changed.

**What was wrong.** `references_actual/eddie.webp` (400×300) contains two
detected faces: the subject at det `0.545`, ‖emb‖ `21.56`, and a blurred
bystander clipped by the right border at det `0.626`, ‖emb‖ `13.15`. InsightFace
returns faces in descending detection score, so `faces[0]` was the bystander —
and `faces[0]` was what every consumer took:

| consumer | file | which face, before |
|---|---|---|
| stored panel embedding | [`create_manual_val_id_embeds.py:56`](../tools/datasets/create_manual_val_id_embeds.py#L56) | bystander |
| PhotoMaker ID vector, inference | [`br_pipeline_helpers.py:209-214`](../src/pipelines/br_pipeline_helpers.py#L209-L214) | bystander |
| PhotoMaker ID vector, training (per-sample) | [`lora2_helpers.py:623-628`](../src/model/photomaker_branched/lora2_helpers.py#L623-L628) | bystander |
| PhotoMaker ID vector, training (batched) | [`lora2_helpers.py:796-803`](../src/model/photomaker_branched/lora2_helpers.py#L796-L803) | bystander |
| BA reference crop | `ref_bboxes.json` `face_crop_new = [98, 0, 313, 281]` | **the subject** |

So eddie was conditioned on one person's ArcFace vector, cropped to another
person's face for the branch, and graded against the first.

**The fix, and why it is the right one.** Selection is now by **maximum IoU
against the declared box** in `ref_bboxes.json` rather than by detection score or
size — the ground-truth box already existed and was unused. The rebuilt file is
`id_embeds_manual_val_subject_v2.pth`, written **alongside** the original rather
than over it, with an audit at `manual_val_subject_v2_preflight.json`:
eddie `face_count: 2`, `index: 1`, `declared_bbox_iou: 0.896`,
`selection_reason: "declared_bbox_max_iou"`, `ambiguous_count: 0` across all 12
identities. Every other identity's vector is bit-identical to the original
(cosine `1.0000`); only eddie changed (`-0.0078`).

**Verified by intervention, not inference.** Eddie's validation images were
regenerated at all twelve steps — `subject_v2_validation_replacement__2000_24000.json`
records `changed_images: 144`, `verified_asset_count: 168` — and re-scored:

| arm | eddie before | eddie after | vs bystander, after | panel before | panel after |
|---|---:|---:|---:|---:|---:|
| E13 | `.176` | **`.4254`** | `.061` | `.3998` | **`.4304`** |
| BC_E13 | `.158` | **`.3540`** | `.023` | `.3894` | **`.4152`** |
| CL9 | `.159` | **`.4155`** | `.024` | `.4151` | **`.4480`** |

The pre-fix images matched the bystander better than the subject (`.176` vs
`.065`); the regenerated images match the subject and not the bystander. That
inversion is the proof that the conditioning path — not only the metric — was
carrying the wrong identity, which was the open question in the 9 August version.

**One piece of this is not yet done.** The three `faces[0]` sites in the *model*
code above still select by detection score, so the **training** conditioning path
retains the defect for any reference image with more than one detectable face. On
the 38 locally available large_dataset images, `28.9%` have more than one detected
face and `13.2%` have `faces[0]` != the largest. `n = 38` is underpowered and the
full corpora are on Serv, but `filtered_ids3.json` already stores
`new_face_crop` for all `127,283` records, so the same declared-box rule can be
applied at zero preprocessing cost. That is E25 in §8.

---

## 3. Occlusion: two regimes, and only one of them is unwinnable

The 9 August version treated Skiing, Crying, Kickboxing and Laughing as one
"occluded" family and concluded that no architecture change could help. Measured
properly on CL9 at 24k, **that is right for one regime and wrong for the other.**

### 3.1 Is the identity still there underneath the occluder?

`id_sim` compares a generation with the reference, so an occluder that hides the
periocular region will depress it whether or not the model drew the right person.
The question `id_sim` cannot answer is whether the person is there at all. To get
at it, each generation is compared with **that identity's own clean-prompt
generations in the same run** — same model, same style, no occluder.

| identity | clean baseline | Crying | Skiing | small-face |
|---|---:|---:|---:|---:|
| jennie | `0.919` | `0.845` | **`0.823`** | `0.784` |
| jensen | `0.854` | `0.796` | `0.651` | `0.613` |
| keanu | `0.828` | `0.768` | `0.645` | `0.482` |
| lex | `0.842` | `0.720` | `0.583` | `0.369` |
| elon | `0.847` | `0.775` | `0.547` | `0.468` |
| eddie | `0.801` | `0.717` | `0.514` | `0.483` |
| **marion** | `0.783` | `0.563` | **`0.243`** | `0.450` |
| **jisoo** | `0.899` | `0.600` | **`0.191`** | `0.786` |
| **mean drop vs clean** | — | **`-0.124`** | **`-0.322`** | `-0.292` |

**Crying barely touches the underlying face** (`-0.124`; every identity stays
above `0.56`). The model draws the right person and `id_sim` cannot read them
through hands and closed eyes. That is a **measurement** problem.

**Skiing is different.** The mean drop is `-0.322` and for marion and jisoo it is
`-0.540` and `-0.707` — their Skiing renders are not the same person as their own
clean renders. That is a **generation** problem, and it is fixable in principle.

### 3.2 The mechanism: where the model puts the goggles

![CL9 Skiing and Crying for all eight identities](assets/hardcase_20260809/fig9_cl9_occlusion.png)

The split is almost binary, and it is a **composition** decision rather than a
rendering-quality one:

- **Goggles pushed onto the forehead** — jennie, keanu, jensen, elon, lex, eddie.
  Eyes fully visible, identity retained at `0.51-0.82`, `id_sim` `.175-.518`.
- **Goggles pulled down over the eyes** — marion and jisoo only. Identity
  collapses to `0.24` and `0.19`. jisoo's render additionally breaks down
  structurally (duplicated goggle geometry, malformed face) and is the image
  whose TOPIQ-Face returns NaN.

So the model *can* satisfy "snow goggles" while keeping the face legible — it does
so six times out of eight. It fails on the two identities that were already the
weakest, which is the interaction to attack.

A colour/landmark signature separates the two occluder types cleanly, and
distinguishes them from everything else in the panel:

| family | n | eye-patch saturation excess | eye-patch contrast ratio | det score |
|---|---:|---:|---:|---:|
| clean | 48 | `0.038` | `1.085` | `0.82` |
| small-face | 16 | `0.017` | `1.119` | `0.85` |
| Kickboxing / Laughing | 16 | `0.061` | `1.157` | `0.82` |
| **Crying** | 8 | `0.023` | **`0.964`** | `0.82` |
| **Skiing** | 8 | **`0.194`** | **`1.385`** | `0.86` |

Skiing is strongly *saturated* and high-contrast (mirrored orange lenses with
specular highlights); Crying is *desaturated* and the flattest region in the panel
(skin-coloured hands, closed lids). Panel-wide these signatures do not predict
`id_sim` (`-0.054` and `+0.285`), which is the point — the damage is specific to
the Skiing prompt on weak identities, not a general function of eye coverage.

### 3.3 What follows

- **Crying, Kickboxing, Laughing: change the protocol, not the model.** Report
  `id_sim` for these separately. The face is correct; the metric cannot see it.
- **Skiing: worth an intervention.** The failure is that the branch is asked to
  paint identity into a region the scene has given to an opaque object, and on
  weak identities it loses the whole face rather than degrading gracefully. The
  closest existing idea is §16.3 of the old project — *zero branch ownership on
  target occluder pixels*: let the base model own the goggles, and confine the
  branch to the visible cheeks, mouth and jaw instead of competing for the eye
  region. CL13's reference dropout, already running, is a weaker probe of the same
  instinct (teach the branch to defer) and its Skiing behaviour should be read
  with this in mind.
- **Neither will rescue `id_sim` when the eyes are genuinely covered** — ArcFace
  needs the periocular region. The target is graceful degradation: keep marion's
  and jisoo's Skiing renders at the `0.5-0.8` retention the other six achieve,
  rather than `0.19-0.24`.

### 3.4 The original observation still stands

![Hard cases with the detected face box and a 1:1 crop](assets/hardcase_20260809/fig3_hard_cases.png)

`Skiing_man_eddie` and `Skiing_wom_marion` render large, sharp, well-lit faces —
TOPIQ-Face `.82` and `.83`, among the best in the panel — with `id_sim` `.155`
and `.186`. The two Skiing prompts have the **largest** required faces in the
panel (`11.8%` and `14.5%` of frame) and among the worst identity scores.

This family also produces the only detection failures worth noting:

| Arm | rows (13 steps × 96) | no face detected | at 24k | dominant image |
|---|---:|---:|---:|---|
| E13 | 1248 | 1 | 0 | `Jumping_wo_marion` |
| BC_E13 | 1248 | 1 | 0 | `Jumping_wo_marion` |
| CL4 | 1248 | 7 | 1 | `Skiing_wom_jisoo` ×6 |
| CL8 | 1248 | 7 | 1 | `Skiing_wom_jisoo` ×6 |
| CL9 | 1248 | 1 | 0 | `Jumping_wo_marion` |

In CL9 at 24k that jisoo image is detected but **TOPIQ-Face returns NaN** on the
crop — the same failure one notch milder. A non-detection scores `id_sim = 0` and
still counts in the denominator
([`id_sim_metric.py:26-31`](../src/metrics/id_sim_metric.py#L26-L31)).

Separately, `Chef man/woman` yields a second detected face in **5/5 arms for all
8 identities** (busy kitchen background). `IDSimBest` takes the **maximum** over
detected faces, so a bystander can only inflate a score.

**Recommendation: stop treating this family as a defect.** Report `id_sim`
primarily on the 12 unoccluded prompts; keep the occluded 12 for prompt adherence
and face quality.

---

## 4. The small-face family: 13 latent cells

![Face pixel size drives quality and not identity](assets/hardcase_20260809/fig1_two_axes.png)

| detected face short side | n | latent cells | TOPIQ-Face | MUSIQ | `id_sim` |
|---|---:|---:|---:|---:|---:|
| `80-110 px` | 30 | 12.6 | **`0.478`** | 56.4 | `0.307` |
| `110-150 px` | 48 | 16.0 | `0.574` | 60.7 | `0.330` |
| `150-200 px` | 150 | 22.7 | `0.669` | 72.4 | `0.432` |
| `200-300 px` | 221 | 28.9 | `0.725` | 73.7 | `0.427` |
| `300-1024 px` | 28 | 40.4 | **`0.786`** | 76.6 | **`0.311`** |

`corr(face_px, TOPIQ-Face) = +0.824`; `corr(face_px, id_sim) = +0.065`.

**A refinement of how that second number should be read.** It does *not* mean
small faces keep their identity. It means the relationship between size and
identity is **non-monotonic**, and pooling the whole range hides it: the last row
above — the `300 px+` bin, which is the Skiing prompts — has the best quality and
the worst identity, and it cancels the small end when a single correlation is
fitted. Within the panel the small-face family does score lower on identity too
(`.315` against `.444` for clean prompts at 24k in E13).

The CL9 retention measure from §3.1 puts a number on it. Compared with each
identity's own clean-prompt renders, the small-face family loses **`-0.292`** —
essentially the same as Skiing's `-0.322`, and more than twice Crying's `-0.124`:

| identity | clean baseline | small-face | drop |
|---|---:|---:|---:|
| **lex** | `0.842` | **`0.369`** | **`-0.473`** |
| keanu | `0.828` | `0.482` | `-0.346` |
| elon | `0.847` | `0.468` | `-0.379` |
| eddie | `0.801` | `0.483` | `-0.318` |
| marion | `0.783` | `0.450` | `-0.333` |
| jensen | `0.854` | `0.613` | `-0.241` |
| jennie | `0.919` | `0.784` | `-0.135` |
| jisoo | `0.899` | `0.786` | `-0.113` |

So the small-face family is not only an artefact problem — at `13-15` latent
cells the model also stops rendering *that specific person*, worst of all for
lex (`-0.473`, consistent with his panel-worst small-face `id_sim` of `.264`).
Quality and identity degrade together here; what breaks the pooled correlation is
the opposite behaviour at the large end.

### The discordance, quantified

Within-arm z-scores, `z(id_sim) - z(TOPIQ-Face)`. Positive = **scores well, looks
wrong**:

| identity | prompt | mean discordance | worst single arm |
|---|---|---:|---:|
| jensen | Jumping man | **`+2.57`** | `+2.28` |
| elon | Dancing man | **`+2.51`** | `+1.73` |
| elon | Jumping man | `+2.22` | `+1.41` |
| keanu | Dancing man | `+2.12` | `+1.57` |
| keanu | Jumping man | `+1.94` | `+1.58` |
| jensen | Dancing man | `+1.90` | `+1.04` |
| lex | Jumping man | `+1.67` | `+1.13` |

**All seven are Jumping or Dancing.** `Dancing_ma_elon` in BC_E13 scores
`id_sim .469` — above that arm's median — with TOPIQ-Face `.463`, the
second-worst face in the panel.

### Two candidate causes ruled out, one confirmed

**Not the loss.** E13 runs `loss_kind: masked_alternating` with
`masked_loss_step: 1`, and `_masked_face_mse` uses
`F.mse_loss(..., reduction="mean")` **inside the box**
([`diffusion_loss.py:26`](../src/loss/diffusion_loss.py#L26)). The face-region
loss is already area-normalised — a 13-cell face gets the same gradient magnitude
as a 40-cell one. Inverse-area re-weighting would be a no-op. **[code]**

**Not data scarcity.** The large_dataset manifest has `127,283` usable target
face boxes; `48.6%` are below 3% of frame area and `65.4%` below 5%. The panel is
only `17.7%` below 3%. Training already sees **more** small faces than validation
asks for. **[measured]**

**It is the base model's spatial resolution.** At step 0 — pretrained PhotoMaker
default, BA effectively off — the family already sits at `0.490`:

| family | step 0 TOPIQ | 24k TOPIQ | Δ | step 0 `id_sim` | 24k `id_sim` | Δ |
|---|---:|---:|---:|---:|---:|---:|
| small-face | `0.490` | `0.559` | `+0.069` | `.181` | `.315` | `+0.133` |
| occluded | `0.664` | `0.761` | `+0.097` | `.327` | `.376` | `+0.049` |
| clean | `0.628` | `0.744` | `+0.116` | `.330` | `.444` | `+0.114` |

The rendered face is **not** undersized — mean rendered short side `116 px`
against a required `108 px` (ratio `1.08`). The mask is small because the prompt
puts the person far away.

### The seam

Mean CIE-Lab step across the fixed mask border, 8 px bands inside versus outside,
672 images:

| family | n | ring ΔE | jaw ΔE | p90 jaw ΔE | mask px |
|---|---:|---:|---:|---:|---:|
| **small-face** | 112 | **`7.29`** | **`8.29`** | **`13.80`** | 108 |
| occluded | 224 | `4.18` | `4.61` | `9.58` | 249 |
| clean | 336 | `5.09` | `3.63` | `6.92` | 202 |

The six worst individual seams are all Jumping prompts (`Jumping_ma_keanu` at
`20.17`). *Caveat:* for a distant subject the band below the jaw is often sky
rather than neck, so part of the gap is background contrast — the ranking is
reliable, the absolute value is an upper bound. **No arm differs**: at step
14,000, jaw ΔE is `4.66 / 4.19 / 4.77 / 4.74 / 4.73` for E13 / CL8 / CL9 / CL10 /
CL11. CL14's feathered training mask is the first arm aimed at it.

---

## 5. Which experiments do better, and why

![Identity and face quality by prompt family and arm](assets/hardcase_20260809/fig2_family_by_arm.png)

Step 24,000, eddie excluded (pre-fix export):

| Arm | dataset | small-face | occluded | clean |
|---|---|---:|---:|---:|
| E13 | large_dataset | `.3429` | `.4051` | `.4797` |
| BC_E13 | BigCelebs | `.3206` | `.4173` | `.4605` |
| CL4 | cosmic | `.3412` | `.4150` | `.4879` |
| **CL8** | cosmic | **`.3804`** | `.4129` | `.4708` |
| **CL9** | cosmic | `.3527` | **`.4336`** | **`.4973`** |

Matched step 14,000, with CL10 and CL11:

| Arm | small-face | occluded | clean | all |
|---|---:|---:|---:|---:|
| E13 | `.330` | `.389` | `.460` | `.415` |
| CL4 | `.355` | `.413` | `.491` | **`.443`** |
| **CL8** | **`.369`** | `.399` | `.449` | `.419` |
| CL9 | `.344` | `.389` | `.490` | `.432` |
| CL10 | `.348` | `.412` | `.475` | `.433` |
| CL11 | `.327` | `.412` | `.479` | `.431` |

- **CL8 owns the small-face family** — `+.037` over E13 at 24k and the only arm
  still rising there. It was written off for leaving 10 undersized faces; that
  verdict was right about face *scale* and wrong as a judgement on the arm.
- **CL9 owns everything else.** CL10 combines both and at 14k has **not**
  inherited CL8's small-face gain (`.348` vs `.369`) — re-check at 24k.

### Why jensen is immune on small faces

jensen scores `.488` on the small-face family against a next-best `.398`. His
reference also has the **smallest face** — `102 px` short side, `7.1%` of a
`600×337` image, against `184-501 px` for everyone else: a low-resolution
reference matched to a low-resolution target.

Across all 84 (identity, prompt) cells with identity and prompt main effects
removed, the partial correlation between `log2(reference face px / target mask
px)` and `id_sim` is **`-0.399`**. The raw correlation is only `-0.071` and the
binned means are flat except at the extreme (`>2.8x` → `.394`). **Real but
modest** — enough for one arm, not enough to lead with.

### Datasets and the curriculum question

| identity | E13 | BC_E13 | CL4 | CL8 | CL9 | best |
|---|---:|---:|---:|---:|---:|---|
| marion | `.341` | `.333` | **`.352`** | `.322` | `.314` | CL4 |
| lex | `.380` | `.367` | `.396` | **`.427`** | `.414` | CL8 |
| jisoo | **`.454`** | `.422` | `.378` | `.320` | `.444` | E13 |
| jennie | `.406` | `.398` | `.473` | **`.496`** | `.462` | CL8 |
| keanu | `.467` | `.448` | `.464` | `.458` | **`.496`** | CL9 |
| elon | `.503` | `.459` | `.511` | **`.531`** | `.475` | CL8 |
| jensen | `.473` | `.533` | `.499` | `.500` | **`.559`** | CL9 |

**BigCelebs wins nothing** except a narrow lead on the occluded family.
**large_dataset wins only jisoo.** cosmic takes 6 of 7.

Against a cross-dataset curriculum: all three corpora fail on the same prompts,
and the age drift in §1.2 appears in all of them. Sequencing corpora cannot
remove a failure they share. The one curriculum the data motivates is **within
cosmic** — CL8's target distribution first, then CL9's reference-scale
calibration — and it ranks behind the items in §8.

---

## 6. Root cause summary, and what is *not* the cause

| Failure | Affected | Cause | Status |
|---|---|---|---|
| eddie identity vector | 12 panel images, plus training references | `faces[0]` selects a bystander | **Fixed for validation (§2); training path outstanding** |
| small-face | Jumping, Dancing (16 images) | `13-15` latent cells: quality **and** identity degrade (`-0.292` retention) | Partly fixable — resolution-side change |
| **Skiing** | 8 images, severe on 2 | goggles placed **over** the eyes rather than on the forehead; branch competes with the occluder and loses the whole face | **Fixable — occluder-aware branch ownership (§3.3)** |
| Crying / Kickboxing / Laughing | 24 images | occluder hides the periocular region; the rendered face is still correct (`-0.124`) | **Not a model defect — change the protocol** |
| apparent age | jennie, lex (24 images) | not established | Diagnose before acting |
| marion | 12 images | off-axis reference at the `97th` percentile of training pose; scoring-side normalisation ruled out | **Generation test next (§8)** |

### What is NOT the cause

- **Not identity confusion.** Rank-1 is `99.1%`; no jennie image is closer to
  jisoo's reference than to her own, and vice versa. *(Retraction — §1.1.)*
- **Not face scale.** Every reference-scaled arm has zero undersized faces at
  14k; rendered/required ratio in the small-face family is `1.08`.
- **Not face detection.** `≤7` non-detections per 1,248 rows; `0-1` at 24k.
- **Not the loss weighting.** The face-region MSE is already area-normalised.
- **Not training-data scarcity of small faces.** `48.6%` of targets are below 3%
  face area against `17.7%` of the panel.
- **Not the dataset.** Same prompts, same identities, same age drift in all three.
- **Not reference image resolution per se.** jensen's reference is `600×337` with
  a `102 px` face and he is the **best** identity.
- **Not reference head pose, as a demonstrated mechanism.** The within-identity
  test came out with the wrong sign for marion and inconsistent signs elsewhere.
- **Not a broken embedding for anyone but eddie.** The other eleven reproduce
  their reference at cosine `1.0000` in the v2 rebuild.

### Confidence

| Claim | Confidence | Basis |
|---|---|---|
| The eddie vector was the bystander, in scoring **and** conditioning | **High** | cosine `1.0000` to it; regeneration inverts the scores **[measured, interventional]** |
| Fixing it recovers eddie | **High** | `.176 → .4254`, `identity_lock +0.257 → -0.010` **[measured]** |
| Identity is preserved across the panel | **High** | rank-1 `99.1%`, 586 images **[measured]** |
| jennie/jisoo are **not** merging | **High** | 0/84 and 0/82 misassignments **[measured]** — retraction |
| Face pixel size drives quality, not identity | **High** | `r = +0.82` vs `+0.07`, n=477 **[measured]** |
| Small-face deficit is inherited from the base model | **High** | step-0 TOPIQ `0.490` with BA off **[measured]** |
| Loss weighting is not the small-face cause | **High** | `reduction="mean"` inside the box **[code]** |
| jennie and lex have wrong apparent age | **Medium-high** | visual, reproduced across 4 arms and 3 datasets; not quantified **[visual]** |
| Crying preserves the underlying identity | **High** | `-0.124` against own clean renders, every identity above `0.56` **[measured]** |
| Skiing genuinely destroys it for marion and jisoo | **High** | `0.243` and `0.191` against own clean renders vs `0.78-0.90` baseline **[measured]** |
| The determinant is goggles-on-forehead vs over-the-eyes | **Medium-high** | 6/8 vs 2/8 split matches the retention split exactly; visual, n=8 |
| Small faces lose identity as well as quality | **High** | `-0.292` retention, worst lex `-0.473` **[measured]** |
| No scoring-side normalisation repairs marion | **High** | 6 variants from the same file, best `+0.0091`, below keanu's `+0.0165` **[measured]** |
| marion's reference is out of the training pose distribution | **Low-medium** | `97th` percentile on an n=38 local sample **[underpowered]** |
| CL8's full-body targets help small faces | **Medium-high** | `+.037` at 24k, monotone; single arm |
| Reference/target scale mismatch costs identity | **Medium** | partial `r = -0.399` over 84 cells; raw `r = -0.071` |
| `~13%` of training references pick a non-subject face | **Low-medium** | n=38 local sample **[underpowered]** |
| The seam is a true head-paste artefact | **Medium** | `8.29` vs `3.63` ΔE, background-contaminated |
| marion's weakness | **Not established** | no measured property explains it |

### Not established

- Why jennie and lex specifically drift in apparent age.
- The true multi-face rate in cosmic_large / BigCelebs / full large_dataset.
- Whether CL8 and CL9's mechanisms compose (CL10 at 24k answers it).
- Whether the `id_sim` gains from the eddie fix change any arm *ranking* — all
  three backfilled arms moved by a similar `+.026` to `+.033`, so probably not,
  but CL4/CL8/CL10/CL11 are still pending.

---

## 7. Ideas from the previous project worth revisiting

From `rsrch/diffusion_template/Jul_new_exp/`, written before the reference-scale
and validation fixes landed and never run:

- **§15A, dual target/reference attention lanes with a bounded per-head gate**
  (`A_face = (1-g)·A_target + g·A_reference`). Shelved when BA training was
  degrading identity outright; that confound is gone — every arm now improves
  identity over step 0 by `+.11` to `+.13`. A bounded gate is a plausible
  mechanism for the age drift in §1.2, since it lets the model weight reference
  evidence per head rather than replacing K/V wholesale. Worth a design pass.
- **§16.3, occluder and boundary preservation** (`L_boundary`, `L_occluder`,
  zero branch ownership on occluder pixels). Closest prior idea to §3 and to the
  seam in §4. It cannot rescue `id_sim` on Skiing, but it is the right shape for
  the seam.
- **§16.4, correspondence supervision.** Expensive, needs landmark/part labels on
  both sides, and nothing measured here motivates it. Not recommended.

From this project: **E16 already tried an identity auxiliary loss** (predicted-x0
PhotoMaker-CLIP proxy) and came out `-.06` against its own base. Any new
identity-space objective must differ in mechanism, not just in weight.

---

## 8. Proposed work

All arms keep the E13 contract: **24k steps, batch 2, one A100, fixed full-96,
`2,240 tensors / 219,217,920 parameters`**, `use_branched_attention=true`,
`pipeline.pose_adapt_ratio=0`, `pipeline.ca_mixing_for_face=false`, unchanged
seeds, prompts, references, scheduler and inference steps. **E13 is the base.**

### Diagnostics first — no GPU, and they gate the arms

1. **marion, conditioning-side normalisation (12 images of inference).**
   Regenerate her twelve CL9 panel images with a roll-corrected, tightly-aligned
   crop of the **same reference file** supplied as the conditioning reference,
   leaving the scoring vector untouched. §1.3 has already ruled out the
   scoring-side fix (`+0.0091` at best, less than keanu gains from plain
   multi-crop TTA), so this is the remaining hypothesis and it is cheap. Decision
   rule: `id_sim` up by more than `+0.03` → reference preprocessing is the fix and
   belongs in the pipeline; unchanged → the photo is the limit and only the
   planned second reference will help.
2. **Quantify the age drift.** The InsightFace estimator is unusable here (§1.2).
   Score reference-versus-generation apparent age for all 8 identities × 5 arms
   with a model reliable on stylised faces, or with a small blind manual rating.
   Until this is a number, no arm should be designed around it.
3. **Finish the eddie backfill** across CL4, CL8, CL10, CL11 and confirm no arm
   ranking changes (all three done so far moved `+.026` to `+.033`).
4. **Add the retention measure to validation.** `cos(generation, own clean-prompt
   centroid)` is what separated Crying from Skiing and what exposed the small-face
   identity loss. It costs nothing beyond embeddings the loop already extracts,
   and it is the only number here that distinguishes "the metric cannot see the
   person" from "the model did not draw the person".

### E25 — subject-face selection in the **training** conditioning path (priority 1)

`E25_large_ds_joint_shadow_sa128_subjectface_24k.yaml`

**Single change:** the three `faces[0]` sites in model code adopt the same rule
the validation rebuild already uses — select by maximum IoU against the declared
face box, falling back to the largest face above a detection floor when no box is
declared.

```yaml
model:
  ba_reference_face_selection: declared_bbox_iou   # NEW; default "first" = today
  ba_reference_face_min_det: 0.30                  # NEW; fallback floor
```

**Why:** §2 proved the mechanism on the validation path by intervention — the
same defect is still live in training. `filtered_ids3.json` already carries
`new_face_crop` for all `127,283` records, so the declared box is available
at zero preprocessing cost.

**Prediction:** references with a single face are unaffected (this is the gate,
not the goal). If the `13.2%` local rate holds, expect a small broad gain rather
than a large narrow one.
**Risk:** low. Defaulting to `first` preserves byte-identical behaviour. The real
risk is a null result if the true rate is much lower than 13% — still worth
knowing, and cheap.

**Gates:** 4k — identities whose references have one face within `±.005` of E13;
8k — no regression on the panel; 24k — panel `id_sim` at or above E13's post-fix
`.43040`.

### E26 — reference face scale matched to the target mask (priority 2)

`E26_large_ds_joint_shadow_sa128_refscalematch_24k.yaml`
*(this was E27 in the 9 August version; the former E26 is withdrawn — §1.1)*

**Single change:** composite the reference so its face occupies the fraction of
the reference canvas that the target's face box occupies of the target canvas,
per sample, keeping CL9's position jitter.

```yaml
datasets:
  train:
    large_dataset:
      reference_frame_mode: target_face_frame
      reference_scale_match: true          # NEW; replaces a uniform scale draw
      reference_scale_match_jitter: 0.15
      reference_position_jitter: 0.15      # unchanged from CL9
```

**Why:** the partial correlation of `-0.399` across 84 cells, plus jensen — the
identity with the smallest reference face and by far the best small-face score.

**This revisits CL2, which failed.** CL2 locked the reference face to exactly the
target's scale **and** position and collapsed on `id_sim` because the branch
learned an in-place copy. CL9 then showed position jitter removes the degeneracy.
E26 keeps the jitter and changes only *how the scale is chosen*. That distinction
is the whole experiment, and it is only testable because CL9 landed.

**Prediction:** small-face TOPIQ-Face above `0.60` (from `0.559`), jaw ΔE below
`6.5` (from `8.29`), overall `id_sim` within `.01` of E13.
**Risk:** the CL2 collapse recurs. Kill rule: `id_sim` at 8k more than `.02`
below E13's 8k value → stop.

**Gates:** 4k — undersized count stays 0; 8k — small-face TOPIQ-Face above
`0.58`; 14k — jaw ΔE below `7.0`; 24k — small-face `id_sim` at or above CL8's
`.3804`.

### E27 — occluder-aware branch ownership (priority 3)

`E27_large_ds_joint_shadow_sa128_occluder_24k.yaml`

**Single change:** during training, exclude occluder pixels inside the target
face box from the branch's ownership, so the branch is supervised only on the
visible face surface.

```yaml
model:
  ba_occluder_aware_mask: true       # NEW, defaults-off
  ba_occluder_skin_threshold: 0.35   # face-region pixels this far from the
                                     # sample's own median skin colour are
                                     # treated as occluder and dropped from the
                                     # branch's target mask
```

**Why:** §3.2 shows the failure is the branch competing with an opaque object for
the same pixels and, on weak identities, losing the entire face rather than
degrading. Six of eight identities already resolve this correctly by pushing the
goggles onto the forehead, so the behaviour is learnable — it just isn't
supervised. Training targets are real photos in which the face is rarely occluded,
so the branch has never had to render an identity into a partially covered region.
This is §16.3 of the July architecture proposal (*zero branch ownership on target
occluder pixels*), which was written before the reference-scale and validation
fixes and never run.

**Prediction:** marion and jisoo Skiing retention rises from `0.24`/`0.19`
towards the `0.5-0.8` the other six already achieve; overall `id_sim` within
`.01` of E13; Crying unchanged (there is no occluder to exclude — hands are
skin-coloured, which the threshold will not catch, and §3.1 shows Crying does not
need fixing).
**Risk:** a colour-threshold occluder proxy is crude and will misfire on strong
shadow or heavy makeup. Mitigation: log the fraction of face-box pixels dropped
per batch; if it exceeds `15%` on average the threshold is wrong. Kill rule:
`id_sim` at 8k more than `.02` below E13.

**Gates:** 4k — mean dropped-pixel fraction between `2%` and `15%`; 8k — no
regression on clean prompts; 14k — Skiing retention for marion and jisoo above
`0.40`; 24k — panel `id_sim` at or above E13's post-fix `.43040`.

**Note on what it cannot do.** It will not rescue `id_sim` when the eyes are
genuinely covered — ArcFace needs the periocular region. The target is graceful
degradation, measured by retention, not by `id_sim`.

### Not proposed

- **Hard-negative identity batching.** Withdrawn — §1.1, on two independent
  grounds.
- **Another identity auxiliary loss.** E16 came out `-.06`.
- **Inverse-area face loss weighting.** Already area-normalised (§4).
- **Small-face oversampling on large_dataset.** Training already has more small
  faces than validation asks for (§4).
- **Anything that chases `id_sim` on Crying, Kickboxing or Laughing.** The face is
  correct; the metric cannot see it (§3.1). Change the protocol.
- **Scoring-side reference normalisation for marion.** Ruled out by measurement
  (§1.3).
- **A cross-dataset curriculum.** All three fail identically (§5).
- **Any age-specific arm before the diagnostic above returns.**

---

## 9. Implementation plan

### Step 1 — E25 model change (defaults-off)

Add to `PhotomakerBranchedLora.__init__` in
[`lora2.py`](../src/model/photomaker_branched/lora2.py), beside the CL13/CL14
flags:

```python
# 10 Aug 2026 - E25: reference ArcFace selection. InsightFace returns faces in
# descending det_score, so faces[0] can be a bystander (see the eddie panel bug).
# "first" reproduces every run before E25 byte-for-byte and stays the default.
ba_reference_face_selection: str = "first",   # "first" | "declared_bbox_iou"
ba_reference_face_min_det: float = 0.30,
```

Add one shared helper, mirroring the rule already used by the validation rebuild,
and call it from every site:

```python
def select_reference_face(faces, *, mode="first", declared_bbox=None, min_det=0.30):
    if not faces:
        return None
    if mode == "first":
        return faces[0]
    if declared_bbox is not None:
        return max(faces, key=lambda f: _iou(f["bbox"], declared_bbox))
    ok = [f for f in faces if float(f.get("det_score", 0.0)) >= min_det] or faces
    return max(ok, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
```

| file | line | path | declared box available? |
|---|---:|---|---|
| [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L623) | 623-628 | training, per-sample | yes — `face_bbox_ref` in the batch |
| [`lora2_helpers.py`](../src/model/photomaker_branched/lora2_helpers.py#L796) | 796-803 | training, batched | yes — `face_bbox_ref` in the batch |
| [`br_pipeline_helpers.py`](../src/pipelines/br_pipeline_helpers.py#L209) | 209-214 | inference | yes — `ref_bboxes.json` via `face_bbox_ref` |

`LargeDatasetTrain.__getitem__` already emits `face_bbox_ref`
([`large_dataset.py:150`](../src/datasets/large_dataset.py#L150)), so the box is
in the batch and needs threading through to the ArcFace call rather than
recomputing. This flag **must** reach the inference pipeline — add it to the
attribute-propagation list in [`train.py:462-491`](../train.py#L462-L491).

### Step 2 — E26 reference compositor

`reference_frame_mode: target_face_frame` already exists and is what CL9 uses.
Add `reference_scale_match` / `reference_scale_match_jitter` to the compositor in
`src/datasets/reference_frame.py` (reached via `reference_policy.py`). When
enabled: compute the target face-area fraction from the sample's own face box,
multiply by `1 ± jitter`, clamp into the existing `[0.06, 0.30]` range, use that
instead of the uniform draw. **Do not touch `_bbox_to_ref_mask`** and do not
change the inference-side reference crop.

### Step 3 — E27 occluder-aware mask (training only)

Extend `_bbox_to_mask` in
[`lora2.py`](../src/model/photomaker_branched/lora2.py), the same function
CL14's feathering touches, behind `ba_occluder_aware_mask`. Inside the target
face box: take the median Lab colour of the nose and cheek region as that
sample's skin reference, and zero the mask on pixels further than
`ba_occluder_skin_threshold` from it. Guard the whole thing with `self.training`
and keep it out of the inference attribute-propagation list in `train.py`, so
validation is provably unchanged — the same pattern CL13/CL14 use.

Two things to get right, both of which have bitten this repo before:

- The mask is built in **latent** space (`bbox / 8`), so the skin statistics must
  be computed on the decoded target at pixel resolution and then downsampled, not
  computed on latents.
- **Do not touch `_bbox_to_ref_mask`.** Only the target-side mask changes; the
  reference is not occluded and must keep full coverage.

Log the dropped-pixel fraction per batch as a training scalar — it is the gate in
§8 and the only way to notice the threshold misfiring.

### Step 4 — registry, validator, launcher

- Validator (`tools/validate_aug_large_ds_config.py`): add `E25`, `E26`, `E27` to
  `ARMS`; assert E25 changes only the two selection fields; assert E26 sets
  `reference_scale_match: true` **and** keeps `reference_position_jitter: 0.15`;
  assert E27 sets only the two occluder fields and leaves the reference policy
  alone. Use **exact arm tokens** (`name.split("_", 1)[0].upper()`) —
  `startswith("E2")` collides, which is the bug that killed CL10 r1 after 40
  seconds.
- Launcher: add all three config names to the `case` gate in
  `launchers/active/run_E_large_ds_hard_v1_20k_1gpu.sh`.
- Records: `experiments/large_dataset/E2{5,6,7}_..._r1.json`.
- Serv packages cloned from E13's, each with a sealed hash-verified snapshot.

### Step 5 — pre-launch gates

1. Composition: 24,000 steps, `2,240 / 219,217,920`.
2. Only the intended field differs from E13.
3. **E25 no-op proof:** with `ba_reference_face_selection: first`, step-0 output
   byte-identical to E13's. With `declared_bbox_iou`, identities whose reference
   has a single detected face must still be byte-identical.
4. **E25 coverage log:** record how many training references had `>1` detected
   face and how often the selection changed — this turns the `n=38` estimate into
   a real number on the first epoch.
5. **E26:** assert the realised reference face-fraction distribution matches the
   target mask-fraction distribution (KS below `0.15`) **and** that position
   jitter is still active — matched scale with zero jitter is CL2.
6. Allow ~10 min model construction plus ~25 min of silent step-0 generation
   before treating quiet as a hang.

### Step 6 — add `identity_lock` and retention to validation

`sim_self` and `sim_ref` come from embeddings the validation loop already
extracts:

```text
sim_ref[id]   = mean cos(generation_i, reference_embedding[id])
sim_self[id]  = mean pairwise cos(generation_i, generation_j), i < j
identity_lock = sim_self - sim_ref
```

Twelve extra scalars per validation, no extra model calls. It moved from `+0.257`
to `-0.010` for eddie across the fix, so it tracks exactly the failure `id_sim`
under-reports.

### Step 7 — report six numbers per arm

| number | why |
|---|---|
| `id_sim`, full 96 | continuity — and state which embedding file was used |
| `id_sim` on the 12 unoccluded prompts | the identity signal that can move |
| `identity_lock` per identity | catches a stable wrong face |
| mean TOPIQ-Face on the small-face family | the artefact axis `id_sim` cannot see |
| undersized count + jaw ΔE | geometry and blending |

`id_sim` alone ranked CL8 above CL9 at 8k, called CL8 a failure at 14k, and hid
that CL8 is the best arm in the project on the small-face family.

---

## 10. Reproducing

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh && conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

# per-image face quality is a Comet asset: one CSV per run, all 13 steps
python - <<'PY'
import comet_ml, os
api = comet_ml.API(api_key=os.environ["COMET_API_KEY"])
ex = api.get_experiment_by_key("1cc0a02371094b24a6a02a4cc649f10c")
a = [x for x in ex.get_asset_list() if x["fileName"].startswith("face_quality_details")][0]
open("E13_fq.csv", "wb").write(ex.get_asset(a["assetId"], return_type="binary"))
PY

# per-image id_sim, per step
python tools/comet/comet_experiment.py fetch \
  --record comet_records/E13_large_ds_joint_shadow_sa128_24k_full96_r4.json \
  --step-number 24000 --output-dir comet_data/<batch>/E13_24000
```

**Post-fix images carry a version suffix.** After the eddie backfill, Comet holds
multiple assets with the same `fileName` (`Angry_man__eddie (12).png`). Strip
`\s*\(\d+\)` and keep the entry with the **largest `createdAt`** per base name,
or you will silently analyse the pre-fix images.

**Join key.** The face-quality CSV's `file_name` is
`0000__eddie__Reading_paper_man_….png`; the `id_sim` CSV's `output_key` is
`Reading pa_eddie.png`; the exported PNG is `Reading_pa_eddie.png`. Join on
`(step, image_index)` from the filename prefix — joining on the name string
silently drops most rows. All 6,240 rows joined here matched, with the identity
field asserted equal on both sides.

**Harness validation.** Re-scoring generations against the reference reproduces
the logged `id_sim` for every single-face identity (E13 @24k: elon `.502` vs
logged `.503`, jensen `.473` vs `.473`, marion `.336` vs `.341`).

**Other traps.** `pyiqa` is not installed locally, so TOPIQ-Face must come from
the Comet asset — and CL10/CL11 have no such asset yet because it is written at
run completion. Local `onnxruntime` is the CPU build; `analyze_faces` runs at
about one second per 1024² image. `python` does not exist outside the conda env.
InsightFace `genderage` is unreliable on stylised faces (§1.2).

**Data written by this analysis**, under `comet_data/hardcase_20260809/`:
`joined_per_image.json` (7,872 rows), `identity_consistency.json`,
`gen_embeds.npz` (670 face embeddings), `reference_audit.json`,
`reference_stability.json`, `reference_repair.json`, `eddie_rescore.json`,
`seam.json`, `eddiefix/` (regenerated eddie panels plus post-fix CSVs), and five
`*__face_quality_per_image.csv`. Figures under
`analysis/assets/hardcase_20260809/`.

## 11. References

- [CL8-CL11 results and CL12-CL14, 9 Aug](2026-08-09_cl8_cl11_results_hard_cases_and_cl12_cl14.md) — corrected here on the small-face family
- [E13 vs BC_E13 dataset analysis, 9 Aug](2026-08-09_e13_vs_bc_e13_bigcelebs_dataset_analysis.md)
- [CL8/CL9 face-scale results, 9 Aug](2026-08-09_cl8_cl9_face_scale_results_and_cl10_cl11.md)
- [Face-scale root cause, 8 Aug](2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md)
- [E13-E18 results, 6 Aug](2026-08-06_e13_e18_results_and_next_experiments.md) — E16's identity-loss failure
- `../dataset_full/val_dataset/manual_val_subject_v2_preflight.json` — the
  subject-face selection audit for the fix in §2
- `rsrch/diffusion_template/Jul_new_exp/2026-07-22_N3a_vs_NN6a_and_NN7_architecture_proposal.md`
  §15A, §16.3, §16.4 — see §7
