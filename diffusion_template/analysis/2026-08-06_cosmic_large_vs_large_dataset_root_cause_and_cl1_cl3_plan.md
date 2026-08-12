# Why cosmic_large fails where large_dataset works, and how to run E13 on it

**Date:** 6 August 2026
**Branch / worktree:** `test`, `/home/kolyangg/rsrch_apr_test`
**Scope:** analysis plus the CL1-CL3 implementation. No experiment was launched
and no live checkout was mutated. See §9 for the implemented file inventory and
its verification status.
**Requested by:** compare `cosmic_large` against `large_dataset`, find pipeline
or dataset-usage defects that only bite `cosmic_large`, and design experiments
that run **exactly E13** on `cosmic_large`.

`BigCelebs` is explicitly out of scope and was not analyzed.

**Experiment naming:** the proposed arms are **CL1 / CL2 / CL3** (Cosmic-Large),
kept separate from the `E##` series so the `large_dataset` ladder stays
unambiguous.

---

## Executive conclusion

`cosmic_large` is not a worse dataset than `large_dataset`. It is a dataset
whose **reference asset is a 256×256 tight face crop**, and our branched
attention (BA) route consumes the reference in a way that silently assumes the
reference is a **native-resolution full scene at the same face scale as the
target**. `large_dataset` satisfies that assumption by construction; every
other dataset property is secondary.

Three numbers carry most of the argument. All were measured locally in this
session from the real artifacts, not quoted from prior reports:

| Quantity | `large_dataset` | `cosmic_large` |
|---|---:|---:|
| Target face area (median, of a 1024² target) | `7.32%` | `9.49%` (≥192px filter) |
| **Reference face area (median, of the reference image)** | **`7.32%`** | **`42.6%`** |
| **Reference→1024 resize factor before the VAE** | **`1.00×`** | **`4.00×` bilinear upscale** |

So on `large_dataset` the reference face and the target face land on the
**same latent grid at the same scale** (≈35×35 latent cells each, ratio `1.00`).
On `cosmic_large` the reference face lands at ≈83×83 latent cells against a
target face of ≈39×39 — a **2.12× linear / 4.5× area misregistration** — and the
pixels backing those cells are a 4× bilinear enlargement of a 144px JPEG face.

The hard BA route (`hard_replace_v1`, which E13 uses) takes target-face queries
and gives them **unwarped, unregistered reference-face K/V with absolute
authority** over the target face. On `large_dataset` this is nearly a
"copy the same thing at the same scale" problem, which is learnable. On
`cosmic_large` it is a "copy a 2× oversized, 4× blurred thing" problem, which is
exactly the pasted / oversized / displaced-face pathology that every Cosmic run
has reported since 24 July.

**§4 answers the natural objection** — "BA only uses the face part of the
reference, so a 256×256 face crop should be ideal" — by tracing what the
reference lane actually is: the reference is noised and denoised **through the
entire frozen U-Net as a second batch half**, so its features depend on its
global scale and composition, and the bbox mask can only select *which* tokens
are used, never *what they encode* or *how many there are*.

Four smaller, independently real, `cosmic_large`-only defects compound it:
independent random **mirroring of the reference**, **79% caption truncation**,
**per-target pseudo-identities** (no multi-view identity, so E18/E19-style
identity packages are impossible), and a **train/validation reference-domain
shift** that `large_dataset` does not have.

Recommended next work is a **three-arm design that holds Cosmic content, the
E13 model contract, and the caption/flip policy fixed, and varies only how the
reference reaches the BA lane**:

| Arm | Reference lane | Model change | Blocked by |
|---|---|---|---|
| **CL1 `sceneref`** | a native 1024² **same-identity Cosmic target**, exactly like `LargeDatasetTrain` | none | offline identity grouping |
| **CL2 `facecanon`** | the same 256px crop, **composited into the target's face frame at the target's scale before the VAE** | **none** | nothing |
| **CL3 `fmtfix`** | the 256px crop as-is + feature-space `ba_hard_v1_reference_roi_warp` | one existing defaults-off flag | nothing |

CL1 and CL2 are the recommended pair: CL1 asks *"does Cosmic train when the
reference is a real scene?"*, CL2 asks *"is the 256 crop enough if we present it
at the right frame and scale?"*. CL3 is a contingency whose mechanism is a
strictly weaker, post-encoder version of CL2's, and it carries a negative prior
from E3.

Every change is either inside a `cosmic_large` dataset file or behind an
existing defaults-off model flag, so **no `large_dataset` run changes
behavior**.

---

## 0. Evidence and provenance

Read completely: `AGENTS.md`, `docs/handoffs/LATEST.md`, and the five Cosmic
reports the user supplied. Code read directly at the current worktree:

| Path | What was verified |
|---|---|
| [large_dataset.py](../src/datasets/large_dataset.py) | target/reference sampling, flip policy, bbox handling |
| [cosmic_large_adapted.py](../src/datasets/cosmic_large_adapted.py) | manifest filters, reference candidate selection, flip policy, prompt modes |
| [cosmic_large_initial_usage.py](../src/datasets/cosmic_large_initial_usage.py) | `self` / `uniform` / `highest_score` / top-3 reference modes |
| [reference_policy.py](../src/datasets/reference_policy.py) | crop / resize / canvas transforms and cache descriptor |
| [lora2.py](../src/model/photomaker_branched/lora2.py) | `_encode_reference_latent(s)`, `_bbox_to_mask`, `_bbox_to_ref_mask`, `encode_prompt_with_trigger_word`, training forward |
| [lora2_helpers.py](../src/model/photomaker_branched/lora2_helpers.py) | per-sample and batched conditioning preparation |
| [branched_runtime.py](../src/model/photomaker_branched/branched_runtime.py) | reference noising and the doubled `[target; reference]` batch |
| [attn_processor_cleanest.py](../src/model/photomaker_branched/attn_processor_cleanest.py) | hard face branch, reference mask use, ROI warp, true-key mask |
| [br_pipeline_helpers.py](../src/pipelines/br_pipeline_helpers.py) | inference-side reference latents and reference mask |
| `src/configs/E13_*`, `large_dataset_joint_r128_24k`, `large_dataset_rhca_hard_v1_audited_20k`, `large_dataset_rhca_40k`, `cosmic_large_initial_usage_rhca`, `cosmic_large_adapted_rhca` | the exact E13 config chain |

Measured locally in this session (new evidence, reproducible offline):

- `dataset_full/large_dataset_sample_3ids/` — 38 images, all `1024×1024`;
  median target-face area `7.32%`, median face short side `239px`.
- `dataset_full/cosmic_large_one_id/` — all references `256×256` JPEG;
  reference-face area `42.6%`, face short side `144px`; target `1024×1024`.
- `dataset_full/val_dataset/references` + `ref_bboxes.json` — the 12 canonical
  validation references are whole photographs from `400×300` to `4763×2679`,
  face area `6.22%`–`50.35%`, median ≈ `19.9%`.

Prior-report numbers reused (not re-measured here, flagged as such): full-Cosmic
median reference-face area `41.60%` / short side `142px`, target-face medians
after the `min_face_res=192` filter, `22,140` accepted records, the
`79.28%` vs `1.52%` CLIP-token truncation audit, and the `180,623` unique
reference paths.

Statements below are labelled **[code]** (verified by reading the current
source), **[measured]** (computed in this session), **[report]** (from a prior
audit), or **[hypothesis]**.

---

## 1. Structural comparison

### 1.1 What each dataset actually hands to the model

```text
large_dataset  target : 1024x1024 body_crop scene   face ~7-9% of area, ~239px
               ref    : ANOTHER 1024x1024 body_crop scene of the SAME named ID
                        face ~7-9% of area, native resolution, real background
               pairing: 2,561 curated IMDb-style identities x 5-30 images

cosmic_large   target : 1024x1024 body_crop scene   face ~9.5% of area, ~272px
               ref    : one of 2-10 256x256 tight face JPEG crops
                        face ~42% of area, 144px of real detail, no scene
               pairing: 22,140 pseudo-identities, ONE target each;
                        refs are ArcFace retrievals at cosine >= 0.70
```

### 1.2 Field-by-field

| Property | `large_dataset` | `cosmic_large` (adapted loader) | Same? |
|---|---|---|---|
| Target image | 1024² body crop **[measured]** | 1024² body crop **[code]** | ✅ |
| Target face area (median) | `7.32%` **[measured]** | `9.49%` **[report]** | ✅ |
| Target face short side (median) | `239px` **[measured]** | `272px` **[report]** | ✅ |
| **Reference image** | another 1024² scene **[code]** | **256² tight face crop** **[measured]** | ❌ |
| **Reference face area (median)** | `7.32%` **[measured]** | **`42.6%`** **[measured]** | ❌ |
| **Reference native detail** | full **[code]** | **144px face, 4× upscaled** **[code]** | ❌ |
| Reference has real context | yes | essentially none | ❌ |
| Identity definition | curated named ID **[code]** | ArcFace retrieval ≥ 0.70 **[report]** | ❌ |
| Targets per identity | 5–30, median 18 **[report]** | **1** **[report]** | ❌ |
| Reference reuse across targets | yes (shared ID pool) | no (each ref path used once) **[report]** | ❌ |
| **Reference random mirroring** | **never** **[code]** | **independent 50% mirror** **[code]** | ❌ |
| Target random mirroring | 50%, bbox propagated **[code]** | 50%, bbox propagated **[code]** | ✅ |
| Captions > 77 CLIP tokens | `1.52%` **[report]** | **`79.28%`** **[report]** | ❌ |
| Accepted records | 47,500 **[report]** | 22,140 of 59,143 **[report]** | — |
| Epochs at E13's 24k×bs2 = 48k samples | ≈ 1.01 | ≈ 2.17 | — |

`large_dataset.py:114-117` flips only the target;
`cosmic_large_adapted.py:281-286` flips target and reference **independently**.
`BigCelebsScheduledTrain` also never flips the reference. `cosmic_large` is the
only training path in the repository that mirrors the reference.

---

## 2. Exactly what the pipeline does with a reference

Both training and inference route the reference through a **letterbox fit into a
`target_size = 1024` square, then a frozen VAE encode**, then treat the result
as a full 128×128 spatial memory.

**Training** — `lora2.py:2198-2235` (`_encode_reference_latent`) and
`lora2.py:2237-2277` (`_encode_reference_latents`, the batched variant E13
uses because `batched_conditioning_preparation: true` is inherited from
`cosmic_large_initial_usage_rhca`):

```python
ow, oh = ref_image.size
scale = min(self.target_size / ow, self.target_size / oh)   # target_size = 1024
rw = max(8, int(round(ow * scale)) // 8 * 8); rh = ...
ref_resized = ref_image.resize((rw, rh), Image.BILINEAR)
ref_tensor  = F.pad(normalized(ref_resized), (pl, pr, pt, pb), value=0.0)  # mid-gray
latents = self.vae.encode(ref_tensor).latent_dist.mode() * scaling_factor
```

**Inference** — identical geometry in
`br_pipeline_helpers.py:225-251` (`prepare_ref_latents`). Verified: no
train/inference asymmetry in the *transform*; the asymmetry is in the *input*.

**Reference mask** — `lora2.py:2126-2165` (`_bbox_to_ref_mask`) reproduces the
same letterbox mapping and then `F.interpolate(..., mode="nearest")` to the
latent shape. The mask is therefore **correctly aligned** with the reference
latent. Misalignment is not the bug; **scale mismatch against the target** is.

**Consumption** — `attn_processor_cleanest.py:365-430`:

```python
ref_face_hidden  = ref_hidden * ref_mask_flat          # reference face, its own frame
face_hidden_mixed = (1 - POSE_ADAPT_RATIO) * ref_face_hidden + ...   # ratio = 0
key_face, value_face = self._k_ref(...), self._v_ref(...)
q_face  = q * mask_gate                                 # target face queries
hidden_face = F.scaled_dot_product_attention(q_face, key_face, value_face, ...)
merged = hidden_bg * (1 - mask_flat) + hidden_face * mask_flat * self.scale
```

There is **no warp, no landmark alignment, no correspondence model and no
target-face fallback** (`pose_adapt_ratio = 0` is a required project
invariant). The target face is written entirely from reference-face K/V.

### 2.1 The consequence, in latent cells

At 1024 input the latent grid is 128×128. Linear face extent = `sqrt(area) × 128`:

| | reference face | target face | **ratio** |
|---|---:|---:|---:|
| `large_dataset` | `sqrt(.0732)×128 ≈ 34.6` cells | `≈ 34.6` cells | **`1.00`** |
| `cosmic_large` | `sqrt(.426)×128 ≈ 83.5` cells | `sqrt(.0949)×128 ≈ 39.4` cells | **`2.12`** |

**[hypothesis, strongly supported]** On `large_dataset` the branch can learn a
near-diagonal spatial correspondence — an easy function that preserves target
geometry while swapping identity. On `cosmic_large` no such correspondence
exists; whatever the branch learns must be scale-invariant matching, and the
easiest local optimum is still approximate identity-mapping of positions, which
copies a face that is ~2× too large and offset. That is precisely the reported
failure signature (Task B: "a tight 256px Cosmic reference can be copied into
the target as an oversized, displaced, or incomplete face").

### 2.2 The effective conditioning strength also differs by ~5×

E13 keeps `ba_hard_v1_true_reference_key_mask: false`, so non-face reference
positions are **zeroed but still supplied as keys**. A zeroed position yields
`K = W_k·0 + b` — one constant key repeated many times, absorbing softmax mass
proportional to its token count **[code]**. The "sink" fraction is
`1 - reference_face_fraction`:

| Reference format | sink fraction | outcome |
|---|---:|---|
| `large_dataset` 1024 scene | `≈ 92.7%` | E13's calibrated operating point |
| `cosmic_large` 256 crop | `≈ 57.4%` | ~5× stronger effective reference push |
| 256 crop on blank 1024 canvas | `≈ 97.3%` | catastrophic (ID `0.1377`) **[report]** |

**[hypothesis]** This explains why the blank-canvas arm collapsed and why the
same LR/rank/schedule is not transferable between the two datasets: identical
hyperparameters produce materially different branch authority. It is also the
reason CL2 is expected to be safe where the historical canvas arm was not —
see §4.4.

`ba_hard_v1_reference_roi_warp` also normalizes this fraction, but **not**
through `face_key_mask_flat` (`attn_processor_cleanest.py:388-391`), which is
inert while `ba_hard_v1_true_reference_key_mask` stays `false` as E13 requires.
The active mechanism is `warped = warped * target_mask_2d` at
`attn_processor_cleanest.py:574`: the warped tensor is zeroed outside the
**target** face box, so the nonzero key positions become target-face-sized.
A direct CPU test of the real `_warp_reference_roi_to_target` with the measured
Cosmic geometry confirms it at both SDXL `attn1` resolutions **[measured]**:

| Site | reference tokens | target tokens | sink before | sink after |
|---|---:|---:|---:|---:|
| 64×64 | 1,764 | 361 | `56.9%` | **`91.2%`** |
| 32×32 | 400 | 90 | `60.9%` | **`91.2%`** |

`91.2%` is `large_dataset`'s `92.7%` operating point. The warp output is exactly
zero outside the target mask, and the empty-mask guard raises as intended;
neither resolution produces an empty mask under Cosmic geometry.

---

## 3. Ranked defect list

Only defects that are **cosmic-specific or cosmic-amplified** are ranked. Shared
defects (CFG behavior, `legacy_full_copy` validation bases, hard rectangular
masks) affect both datasets equally and are not part of this diagnosis.

| # | Defect | Where | Evidence | Cosmic-only? |
|---:|---|---|---|---|
| **P1** | **Reference face is 2.12× oversized relative to the target face on a shared latent grid, and is a 4× bilinear upscale of 144px of real detail** | `lora2.py:2198-2277` + dataset asset | **[measured]** + **[code]** | Yes — `large_dataset` ratio is `1.00`, resize `1.00×` |
| **P2** | **Reference is independently mirrored 50% of the time**, doubling pose misregistration for a spatially routed branch | `cosmic_large_adapted.py:281-286` | **[code]** | Yes — no other loader flips references |
| **P3** | **79.28% of captions exceed 77 CLIP tokens** and are truncated before pose/background are seen; the tokenizer truncates, then PhotoMaker's class-token expansion consumes more budget | `lora2.py:1963-2010` | **[report]** + **[code]** | Yes — `large_dataset` is `1.52%` |
| **P4** | **Per-target pseudo-identities**: 22,140 targets → 22,140 identity groups; `_identity_id` falls back to the reference parent directory; "same person" is an ArcFace retrieval at ≥0.70, not curation | `cosmic_large_adapted.py:171-181` | **[code]** + **[report]** | Yes |
| **P5** | **Train/validation reference-domain shift on both conditioning lanes.** `manual_val` supplies whole photographs (face `6–50%`, median `≈20%`, native detail). Cosmic training supplies a 42%-face upscaled crop. The same tensor feeds the VAE spatial lane, the CLIP `id_image_processor`, and the ArcFace embedding | `manual_val.py`, `lora2_helpers.py:611-643` / `:786-815` | **[measured]** + **[code]** | Yes — `large_dataset` training references are the same domain as validation |
| **P6** | Effective branch authority differs ~5× between datasets under identical hyperparameters (§2.2) | `attn_processor_cleanest.py:365-430` | **[code]** + **[hypothesis]** | Yes |
| **P7** | **Silent zero identity embedding**: a failed detection substitutes `torch.zeros(512)`, and `faces[0]` is never matched against `face_bbox_ref` | `lora2_helpers.py:621-630`, `:795-805` | **[code]** | Amplified — tight crops clip faces and can contain a second person's fragment. `analyze_faces` retries down to `det_size=(256,256)`, so this is a *robustness* item, not a headline; **no Cosmic run has ever measured the rate** |
| **P8** | 24k×bs2 = 48k samples over 22,140 accepted records ≈ **2.17 epochs** vs ≈ `1.01` on `large_dataset` | config arithmetic | **[code]** | Yes — raises memorization pressure at fixed step budget |

### 3.1 What is *not* the problem

Explicitly ruling these out prevents another round of the same experiments:

- **Target resolution or target face size.** After `min_face_res=192`, Cosmic
  targets are *slightly larger-faced* than `large_dataset` (`272px` vs `239px`
  median short side). Targets are 1024² body crops in both.
- **Loader mechanics.** `CosmicLargeAdaptedTrain` is audited: 22,140 accepted
  records, box validation, exact bbox propagation through crop/resize/flip,
  path-inequality enforcement, policy-aware cache keys, 64/64 decode preflight.
- **Reference crop margin (40% vs 60%) or 512px content.** `40%` and `60%`
  produce identical crops for `99.9922%` of `180,623` candidates and `512px` is
  an upscale of a ≤256px source **[report]**. Do not repeat these.
- **Blank-canvas padding as previously run.** Falsified (ID `0.1377`, ~10/12
  catastrophic). §2.2 and §4.4 give the mechanism and explain why CL2 is a
  different proposition.
- **Runtime speed.** Solved: async CUDA + ONNX Runtime CUDA + 2 workers →
  `2.06–2.10 s/step`.

---

## 4. Why the BA route is a *second consumer* of the reference

> *"In BA the face part of the reference is what matters, so I thought 256×256
> would work well."*

This is the right question and the intuition is half correct: BA does use only
the face **region**. The problem is that "region" is selected by a mask *after*
the reference has already been turned into features, and the mask cannot change
what those features encode. This section traces both consumers explicitly.

### 4.1 Consumer 1 — the PhotoMaker identity lane (crop-friendly)

```text
ref PIL ──► id_image_processor (CLIP)  ─► 224x224, normalized ─┐
        └─► InsightFace analyze_faces  ─► aligned 112x112 ArcFace ─► 512-d ─┤
                                                                            ▼
                                        id_encoder ──► ID token embeddings
                                        fused at the expanded class-token slots
```

`lora2_helpers.py:611-643` (per-sample) and `:786-815` (batched) **[code]**.

Properties that matter:

- **the reference is normalized internally.** CLIP resizes to 224 regardless of
  input size; ArcFace detects the face and warps it to a canonical 112×112 by
  landmarks.
- **the output is a global vector.** There is no spatial correspondence to the
  target at all — the ID tokens enter through ordinary cross-attention.
- therefore **the tighter the crop, the better**: a 256 crop puts ~144px of face
  into the 224 CLIP input, while a 1024 scene shrinks the same face to ~65px.

For this consumer, Cosmic's asset is genuinely *better* than `large_dataset`'s.

### 4.2 Consumer 2 — the branched spatial lane (scene-shaped)

```text
ref PIL ─► letterbox to 1024x1024 ─► frozen VAE ─► 128x128x4 latent
        ─► scheduler.add_noise(t)  ─► scale_model_input(t)
        ─► torch.cat([target_latents, ref_noised], dim=0)      # batch is DOUBLED
        ─► the ENTIRE frozen U-Net runs on both halves
             at every patched attn1 site:
                 ref_hidden  = hidden_states[batch:]           # reference half
                 ref_face    = ref_hidden * mask_ref           # mask applied HERE
                 K,V         = W_k_ref(ref_face), W_v_ref(ref_face)
                 q_face      = q_target * target_mask
                 out_face    = softmax(q_face K^T) V
```

`branched_runtime.py:932-951` **[code]** (noising + doubled batch),
`lora2.py:1762-1776` **[code]** (training entry),
`attn_processor_cleanest.py:365-430` **[code]** (masking + attention).

The decisive structural fact: **the reference is not passed to BA as a face
crop. It is denoised through the whole U-Net as its own 1024×1024 image**, and
BA reads its intermediate feature maps. The bbox mask is applied at each layer,
*to features that were already computed from the full canvas*.

### 4.3 Why masking the face region is not enough

Five independent reasons, each verifiable in the code above:

1. **The mask selects tokens; it does not rescale them.** The U-Net's receptive
   field at a given block is fixed in *latent cells*, not in "face units". At
   `up_blocks.0` (32×32 grid) one token covers 1/32 of the image width. On a
   `large_dataset` reference the face spans ≈9 tokens, so one token ≈ an eye or
   a mouth corner. On a 4×-upscaled Cosmic crop the face spans ≈21 tokens, so
   one token ≈ part of an iris. The target's queries at that same layer are
   "eye-sized". **Query and key granularity do not match.**
2. **Token count scales with area.** Cosmic supplies ≈4.5× more face keys, and
   they compete against the constant sink keys differently (§2.2). Same
   hyperparameters, different branch authority.
3. **SDXL features are scale-tuned.** The frozen U-Net was trained on ~1024
   photographs where faces occupy a typical photographic fraction. A face
   rendered at 2.12× that linear scale is off-distribution for the frozen
   encoder — the same reason img2img at the wrong resolution duplicates faces
   and limbs. The reference lane must survive as a *plausible photograph*, not
   just as a face.
4. **4× bilinear upscale destroys the signal identity lives in.** A 256 source
   carries ≈32×32 latent cells of true information but is written into a 128×128
   latent. The VAE encodes a low-passed image; high-frequency skin, iris, and
   hair detail is simply absent from the K/V, independent of geometry.
5. **The merge is positional.** `merged = hidden_bg*(1-mask) + hidden_face*mask`
   writes the attention output back into the target's face rectangle. Any
   residual mismatch in *where inside the face* the identity content came from
   is a literal geometric copy error, not a soft blend.

**Consumer 1 normalizes the reference for us (resize to 224, landmark-align to
112). Consumer 2 does not — it inherits whatever geometry the 1024 canvas
gives it.** That asymmetry, not the crop itself, is the defect.

### 4.4 What would make consumer 2 behave like consumer 1

Supply the normalization ourselves: **render the reference face into the
target's face frame, at the target's scale, before the VAE and the U-Net.**
The dataset already knows both boxes inside `__getitem__`, so this is a pure
dataset-side transform. That is **CL2 `facecanon`** (§5.3).

Why this is not the failed blank-canvas experiment:

| | historical canvas arm **[report]** | CL2 `facecanon` |
|---|---:|---:|
| Reference face size in the 1024 canvas | native 256 crop → face ≈ `167px` | rescaled to the **target's** face box, ≈ `315px` |
| Reference face area of canvas | `2.66%` | **`≈9.5%`, i.e. the target's own fraction** |
| Reference/target face scale ratio | `0.53` (2× too small) | **`1.00`** |
| Sink fraction (§2.2) | `97.3%` — collapse | **`≈90.5%`, matching `large_dataset`'s `92.7%`** |
| Face position vs target | centered, unrelated to target | **same center as the target box** |
| Non-face fill | flat mid-gray | reflect-padded real pixels (gray is a logged ablation) |

The old arm changed position and *shrank* the face into a near-empty canvas.
CL2 changes position **and scale to match the target**, landing on
`large_dataset`'s operating point on every axis in the table. **[hypothesis]** —
the mechanism in §2.2 predicts this difference, and CL2 is precisely the test of
that prediction.

Two related options that are **not** proposed here, and why:

- **`ba_hard_v1_true_reference_key_mask: true`** would make the non-face fill
  provably irrelevant by excluding those keys from softmax entirely. But E1 ran
  it on `large_dataset` and moved step-0 ID from `.30187` to `.10310`, because
  removing the sinks multiplies the reference contribution by ~13×. Combining it
  with a geometry change would be a confounded two-factor arm. Keep it as a
  follow-up if CL2's fill turns out to matter.
- **A compact face-token interface** — encode a landmark-aligned face ROI into a
  small set of tokens with *no* spatial correspondence to the target grid, and
  merge as a bounded residual. This is the true "consumer-1-shaped" BA and was
  already suggested on 24 July ("canonical reference-token interface"). It is an
  architecture change, not a dataset change, so it is out of scope for a
  "run E13 on cosmic_large" comparison; revisit it if CL1 *and* CL2 both fail.

### 4.5 Why `cosmic_large` works well for other groups

**[hypothesis]** IP-Adapter / PuLID / plain-PhotoMaker-style methods have
**only consumer 1**. For them a 256×256 tight crop is the ideal input and the
dataset is excellent. Our BA route adds consumer 2, whose contract the asset
does not satisfy. This is a compatibility statement about our architecture, not
a defect in the dataset, and it is consistent with both their result and ours.

---

## 5. Recommended experiments

### 5.1 Shared contract (identical to E13 — do not vary)

Everything below is inherited unchanged from
`E13_large_ds_joint_shadow_sa128_24k` → `large_dataset_joint_r128_24k` →
`large_dataset_rhca_hard_v1_audited_20k`:

- hard spatial BA `hard_replace_v1` rank 128, generic effective rank 32,
  training-time PhotoMaker-default effective rank 64;
- `ba_lr = generic_adapter_lr = photomaker_default_lr = 1e-4`;
  `WarmupHoldCosineLR(warmup 20, hold 14000, total 24000, min_factor 0.1)`;
- `validation_shadow_photomaker_default: true`;
- 24,000 optimizer steps, batch size 2, one A100,
  `epoch_len=2000`, `n_epochs=12`, validation + checkpoint at step 0 and every
  2,000 steps;
- `masked_loss_step: 1` (face-only loss every step);
- fixed 96-image `manual_val` panel, unchanged seeds/prompts/references/cached
  boxes/scheduler/steps/CFG/`IDSimBest`;
- `pipeline.pose_adapt_ratio: 0.0`, `pipeline.ca_mixing_for_face: false`,
  branched CA disabled;
- expected trainable contract **2,240 tensors / 219,217,920 parameters**
  (unchanged: all three arms add zero parameters);
- Comet project `aug-large-ds`, so the E13 curve is the in-project control.

**Shared Cosmic controls — fixed across CL1/CL2/CL3, so the only variable is the
reference lane:**

| Control | Value | Removes |
|---|---|---|
| Target filter | `min_face_res: 192` | — |
| Captions | `prompt_mode: pose_first`, `prompt_max_words: 55` | P3 |
| Target mirroring | on, bbox propagated | — |
| **Reference mirroring** | **off** | P2 |
| Reference identity embedding | fail closed, bbox-IoU face selection | P7 |
| Self-reference fallback | forbidden | — |

**Free correctness gate:** the training dataset does not touch step-0
inference, and no arm changes a step-0-visible model flag. All 96 step-0 PNGs
of CL1, CL2 and CL3 **must be byte-identical to E13 r4's step-0 panel**
(immutable Comet key `1cc0a02371094b24a6a02a4cc649f10c`, step-0 ID `.30212`).
If they differ, the validation contract has drifted and the run must be stopped.

### 5.2 CL1 — `sceneref` (primary, decisive)

**Config:** `src/configs/CL1_cosmic_joint_shadow_sa128_sceneref_24k.yaml`
**Run:** `CL1_cosmic_joint_shadow_sa128_sceneref_24k_full96_r1`

**Question:** when `cosmic_large` supplies the *same kind of reference*
`large_dataset` supplies — a native 1024² same-identity scene — does E13 train
on it?

**Delta:** the reference is **another accepted 1024² Cosmic target of the same
identity**, with `LargeDatasetTrain` conventions exactly:
`reference_crop_margin=null`, `reference_content_size=null`,
`reference_canvas_size=null`, raw `face_crop_new` bbox, and a hard error when no
distinct same-ID target exists.

This eliminates **P1, P5, P6 and P8 simultaneously** and, if grouping succeeds,
largely **P4**. It leaves the Cosmic *content* — target scenes, captions, and
identity-retrieval noise — intact. That is what makes it decisive:

- if CL1 approaches E13's trajectory → the problem was **entirely** the
  reference asset format, and the fix generalizes to any dataset;
- if CL1 still plateaus below `.32` → the problem is Cosmic **content**
  (identity noise, target quality), and no reference reformatting will rescue it.

**Prerequisite (Phase 0 gate, no GPU):** Cosmic must be joined into multi-target
identity groups. Two routes, in preference order:

1. **Manifest-native.** Re-audit the live manifest keys on Serv. The local
   one-ID extract carries `"identity_id": "id_00081_1017318003459"`
   (`dataset_full/cosmic_large_one_id/train.json`), while the 26 July audit
   reported no `identity_id` / `person_id` / `id` field in
   `gathered_data_cosmic_large_filtered.json`. **Resolve this contradiction
   first** — if a real identity field or a reference *source-image* pointer
   exists, CL1 becomes trivial and also unlocks native full-scene references.
2. **Offline ArcFace grouping.** One pass over the 22,140 accepted targets:
   detect the face at `face_crop_new`, embed with the pinned Buffalo-L
   `w600k_r50.onnx` (SHA-256
   `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`, already
   used by E22), then build mutual-nearest-neighbour components at a
   **conservative** cosine threshold (start `0.75`, report `0.70/0.75/0.80`).
   Write an immutable, SHA-256-sealed `identity_groups.json`.

**Launch gate:** require **≥ 3,000 targets in groups of ≥ 2**. Below that, CL1 is
under-powered — report the count and stop rather than launching a small run.

**Risk:** grouping recall may be low; grouping errors inject wrong-identity
references. Mitigate with the conservative threshold, a manual audit of 100
sampled pairs per acceptance band, and by logging the target↔reference ArcFace
cosine of every emitted pair.

### 5.3 CL2 — `facecanon` (co-primary; answers §4, zero model change)

**Config:** `src/configs/CL2_cosmic_joint_shadow_sa128_facecanon_24k.yaml`
**Run:** `CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1`

**Question:** is the 256×256 crop sufficient once its face is presented to the
frozen encoder **at the target's frame and scale** — i.e. once we do for
consumer 2 what CLIP/ArcFace already do for consumer 1?

**Delta:** a new dataset-side reference policy,
`reference_frame_mode: target_face_frame`. In `__getitem__`, after the target
flip and with both boxes known:

```text
1. s = target_face_short_side / reference_face_short_side       # ≈ 315/167 ≈ 1.89
2. resize the 256 crop and its bbox by s                        # ≈ 483x483
3. paste into a 1024x1024 canvas so the reference face bbox
   CENTER coincides with the target face bbox center
4. fill the remainder by reflect-padding the pasted content     # no flat regions
5. clamp: if the pasted patch would exceed the canvas, scale down
   to fit and record the realized ratio
6. emit the propagated reference bbox; NO reference flip
```

Result, by construction: reference/target face scale ratio `1.00`, same center,
sink fraction ≈ the target's own — i.e. **`large_dataset`'s operating point on
every axis of the §4.4 table**, using Cosmic's existing asset.

**No model flag changes**, so the trainable contract and the step-0 panel are
untouched. This is the cheapest arm to implement and the only one that is
unblocked today.

**Risks and controls.** The reference lane still carries a 4× upscale of 144px
of detail (P1's *resolution* half is not fixed — only its *geometry* half), and
the reflect-padded surround is synthetic. Log the realized scale ratio, the
clamp rate, and the fraction of canvas that is real vs reflected. Run the
flat-gray fill only as a later ablation if the surround is implicated.

**Interpretation:** CL2 vs CL1 cleanly separates *geometry* from
*native resolution and real context*, because they share every other control.

### 5.4 CL3 — `fmtfix` (contingency)

**Config:** `src/configs/CL3_cosmic_joint_shadow_sa128_fmtfix_24k.yaml`
**Run:** `CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1`

**Question:** can the scale mismatch be corrected **after** the encoder, in
feature space, instead of before it?

**Delta:** keep the 256 crop unchanged and set
`model.ba_hard_v1_reference_roi_warp: true`.
`attn_processor_cleanest.py:507+` (`_warp_reference_roi_to_target`) bilinearly
re-expresses the masked reference ROI in the target bbox frame before the
reference K/V projection, and zeroes it outside the target face box
(`:574`), which also normalizes the sink fraction to `91.2%` (§2.2).
The `face_key_mask_flat` switch at `:388-391` is inert here because
`ba_hard_v1_true_reference_key_mask` stays `false` under the E13 contract.

**Honest prior.** E3 ran exactly this flag on `large_dataset` and *regressed*
(`.32788` step-0 → `.30082 @12k` → `.28126 @20k`). Two reasons this arm is
ranked third:

1. On `large_dataset` the ratio is already `1.00`, so the warp corrected nothing
   and only resampled features — the regression there is uninformative about
   Cosmic, but it is not encouraging either;
2. more fundamentally, **it warps features that were already computed at the
   wrong granularity** (§4.3 point 1). Reason 3 of §4.3 — the frozen U-Net's
   scale tuning — is *not* addressed by a post-hoc warp, whereas CL2 addresses it
   by construction.

Because `ba_hard_v1_reference_roi_warp` changes the step-0 route, CL3 is the one
arm exempt from the byte-identical step-0 gate; declare its step-0 value
explicitly, as E1/E3/E4/E12 did.

**Run CL3 only if** CL2 shows a clear but incomplete gain (which would suggest
alignment helps and is worth attacking from both ends), or if CL2 is blocked.

### 5.5 What not to run

Do not re-run: uniform 256 references with legacy captions and reference flip
on (that is the arm that already failed); 40%/60% margin arms; 512px upscales;
the historical blank-canvas policy; `pose_adapt_ratio > 0`; CA mixing; or any
arm that changes the E13 step budget, LR schedule, or validation panel.

---

## 6. Implementation plan

Written for an implementing agent. Follow the order; each phase has a gate.
No GPU is needed before Phase 5. **CL2 is unblocked and can be implemented
first**; CL1 waits on Phase 0.

### 6.0 Blast radius: no pipeline code changes are required

**Verified by reading the current source.** None of the three arms needs any
change to the model, trainer, attention processors, pipelines, loss, or
`train.py`.

| Subsystem | CL1 | CL2 | CL3 | Why |
|---|:--:|:--:|:--:|---|
| `src/model/**` (lora2, helpers, branched_runtime, attn processors) | — | — | — | `_encode_reference_latent(s)` letterboxes **any** input size to 1024 and `_bbox_to_ref_mask` derives its mapping from the actual `(ref_h, ref_w)`, so 1024² and 256² references are both already handled **[code]** |
| `src/trainer/**` | — | — | — | see the flag-propagation row below |
| `src/pipelines/**` | — | — | — | inference-side reference geometry is unchanged; only training references differ |
| `src/loss/**`, `train.py` | — | — | — | `train.py` never references a dataset name; it only calls `get_dataloaders(config, ...)` (`train.py:239`) **[code]** |
| `src/datasets/data_utils.py` | — | — | — | the dataset is instantiated generically by name (`data_utils.py:70-71`); the only special case is the duck-typed `requires_sequential_sampling` attribute, which only `large_dataset_balanced_multiref` and `big_celebs_scheduled` set. All three CL arms are ordinary shuffled datasets **[code]** |
| `src/datasets/<cosmic loader>` | ✅ new file | ✅ defaults-off option | ✅ shared control only | the entire scientific delta lives here |
| `src/configs/datasets/all_datasets.yaml` | ✅ | ✅ | ✅ | registry entry / new constructor arguments — configuration, not pipeline code |
| `src/configs/CL#_*.yaml` | ✅ | ✅ | ✅ | leaf configs inheriting E13 |
| `tools/`, `launchers/`, `serv_run_packages/` | ✅ | ✅ | ✅ | new standalone files; the same scaffolding every `E##` arm needs |

Two points that could plausibly have required pipeline work, and do not:

- **CL3's `ba_hard_v1_reference_roi_warp` is already fully plumbed.** It is
  implemented in `attn_processor_cleanest.py:375-391` and `:507+`, and — the
  part that matters, given E17's failure mode where a model flag never reached
  the temporary validation pipeline — it is already in **both** propagation
  lists: `train.py:490` and `base_trainer.py:931` **[code]**. E3 exercised it
  end-to-end on Serv. CL3 is a pure config flip.
- **CL1/CL2 emit 1024² references where the adapted loader emits 256².** No
  consumer needs updating: `_encode_reference_latents` pads every reference to
  the same 1024 square before stacking, so `batched_conditioning_preparation`
  keeps working and the VAE cost is **unchanged** (it always encodes 1024²
  regardless of source size) **[code]**.

Two genuine, non-blocking consequences to record rather than fix:

- **CL1 increases loader I/O**: it reads a second 1024² JPEG per sample instead
  of a 256² one. This restores `large_dataset`'s I/O profile, which is the
  baseline the `2.06–2.10 s/step` figure was measured against; re-measure and
  report step time at startup.
- **`require_reference_face_embedding` must be resolved offline.** The in-loader
  form would need a per-worker InsightFace/ONNX session, because
  `analyze_faces` is called with the *model's* analyzer
  (`lora2_helpers.py:621-630`), which a DataLoader worker cannot reach. Phase 0
  step 6 therefore produces a JSON accept-list and the loader simply filters
  against it — a data artifact, zero runtime change. Record the resulting
  accepted-record count, since it changes the epoch arithmetic in §1.2.

### Phase 0 — Manifest and identity audit (no GPU, blocking for CL1 only)

1. On Serv, inspect the live manifest
   `/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data/gathered_data_cosmic_large_filtered.json`.
   Dump the **complete key set** of 20 random records. Explicitly answer:
   does any record carry `identity_id` / `person_id` / `id`, or a pointer to
   the **source image** each `face_paths` crop was cut from?
2. Record the manifest SHA-256 and the accepted-record count under the E13-side
   filters (`min_face_res=192`).
3. If no identity field exists, write
   `tools/datasets/build_cosmic_identity_groups.py`:
   - inputs: manifest, dataset root, ArcFace ONNX path + expected SHA-256,
     cosine threshold, output path;
   - for each accepted target: crop `face_crop_new` from the 1024 body crop,
     embed with Buffalo-L, store L2-normalized vectors;
   - mutual-nearest-neighbour graph at the threshold, connected components,
     drop components of size 1;
   - output an immutable JSON `{identity_id: [target_path, ...]}` plus an audit
     block (threshold, counts, group-size histogram, detection-failure count,
     embedding model SHA-256) and its own SHA-256;
   - default is a dry run; writing requires an explicit flag.
4. **Gate:** report `targets_in_groups_ge_2`. Require `≥ 3,000` to proceed with
   CL1. Report the number regardless; do not silently shrink the experiment.
5. Manually audit 100 sampled same-group pairs per band
   `[0.70,0.75) [0.75,0.80) [0.80,0.85) ≥0.85` and record the observed
   wrong-identity rate.
6. Independently, measure the **reference detection-failure and bbox-mismatch
   rate** on a 5,000-candidate sample of the 256px crops (sizes P7 and produces
   the accept-list used by the shared fail-closed control).

### Phase 1 — Dataset code

All changes live in `src/datasets/`. **Do not modify
`src/datasets/large_dataset.py` or `src/datasets/reference_policy.py`.**

**CL1 loader** — add `CosmicLargeSceneRefTrain` in a new
`src/datasets/cosmic_large_sceneref.py` (a new file is cleaner than more modes
in `cosmic_large_adapted.py`, and must not change the adapted loader):

- constructor: `manifest_path`, `dataset_root`, `identity_groups_path`,
  `expected_identity_groups_sha256`, `min_face_res=192`,
  `random_horizontal_flip=True`, `random_reference_flip=False` (assert it is
  `False`), `prompt_mode`, `prompt_max_words`, `instance_transforms`;
- fail closed on: manifest/groups SHA mismatch, an identity with `< 2` accepted
  targets, target == reference path, invalid `face_crop_new`;
- `__getitem__` mirrors `LargeDatasetTrain.__getitem__`
  ([large_dataset.py:107-166](../src/datasets/large_dataset.py)) exactly:
  load the 1024 target via `body_crop`, flip target + bbox together, sample a
  **distinct** same-group target as the reference, load it the same way, emit
  its raw `face_crop_new`, **no reference transform and no reference flip**;
- emit the same keys as `LargeDatasetTrain`, plus `identity_id`, `target_path`,
  `reference_path`, `reference_cache_key = f"{reference_path}::raw"`;
- validate both bboxes against `(1024, 1024)` after `preprocess_data`.

**CL2 policy** — add a **defaults-off** `reference_frame_mode` to
`CosmicLargeAdaptedTrain`, implemented as a new helper in a *new* module
`src/datasets/reference_frame.py` (leave `reference_policy.py` untouched):

- `reference_frame_mode: str = "native"` — `"native"` reproduces today's
  behavior byte-for-byte; `"target_face_frame"` performs the §5.3 transform;
- signature `compose_target_frame_reference(reference, reference_bbox,
  target_bbox, canvas=1024, fill="reflect") -> (image, bbox, descriptor)`;
- the returned descriptor must enter `reference_cache_key`, exactly as
  `apply_reference_policy` does today
  (`reference_policy.py:152-155` explains why);
- deterministic given `(reference, reference_bbox, target_bbox)`; assert the
  emitted reference-face/target-face short-side ratio is within `[0.95, 1.05]`
  or record the clamp;
- emit telemetry: realized ratio, clamp flag, real-vs-reflected canvas
  fraction.

**CL2/CL3 shared control** — add defaults-off
`require_reference_face_embedding: bool = False` to `CosmicLargeAdaptedTrain`.
Prefer resolving it **offline** into an accept-list from Phase 0 step 6 so no
InsightFace call is added to the hot loop; if done in the loader, select the
detection with maximum IoU against `face_bbox_ref` (not `faces[0]`), raise
below IoU `0.3`, and log the rejection rate.

**Registry** — add `cosmic_large_sceneref` to
`src/configs/datasets/all_datasets.yaml` under `train:`, following the existing
`cosmic_large_adapted` pattern with inert `${oc.env:...,/nonexistent/...}`
defaults so unrelated configs still compose. Add the new
`cosmic_large_adapted` arguments with their defaults-off values.

### Phase 2 — Configs

All three inherit **E13**, never the Cosmic configs, so the model/optimizer/
validation contract is provably identical.

```yaml
# src/configs/CL1_cosmic_joint_shadow_sa128_sceneref_24k.yaml
defaults:
  - E13_large_ds_joint_shadow_sa128_24k
  - _self_

train_dataset_name: cosmic_large_sceneref
val_datasets_names: [manual_val]

datasets:
  train:
    cosmic_large_sceneref:
      min_face_res: 192
      random_horizontal_flip: true
      random_reference_flip: false
      prompt_mode: pose_first
      prompt_max_words: 55
  val:
    manual_val:
      limit: 96

writer:
  experiment_comment: >-
    CL1: exact E13 route on cosmic_large with native 1024 same-identity scene
    references built from an offline ArcFace identity grouping. Isolates the
    reference-asset format from Cosmic content; no model or optimizer change.
```

```yaml
# src/configs/CL2_cosmic_joint_shadow_sa128_facecanon_24k.yaml
defaults:
  - E13_large_ds_joint_shadow_sa128_24k
  - _self_

train_dataset_name: cosmic_large_adapted
val_datasets_names: [manual_val]

datasets:
  train:
    cosmic_large_adapted:
      min_face_res: 192
      reference_crop_margin: null          # complete existing 256px asset
      reference_content_size: null         # scale is set by the frame policy
      reference_canvas_size: null
      reference_frame_mode: target_face_frame   # NEW, defaults-off elsewhere
      reference_frame_fill: reflect
      random_horizontal_flip: true
      random_reference_flip: false
      prompt_mode: pose_first
      prompt_max_words: 55
      require_reference_face_embedding: true
  val:
    manual_val:
      limit: 96

writer:
  experiment_comment: >-
    CL2: exact E13 route on cosmic_large; the 256px reference face is composited
    into the target face frame at the target scale BEFORE the VAE, so the frozen
    encoder sees the reference at the same granularity as the target. Tests
    whether a tight crop suffices once consumer-2 geometry is normalized.
```

```yaml
# src/configs/CL3_cosmic_joint_shadow_sa128_fmtfix_24k.yaml
defaults:
  - E13_large_ds_joint_shadow_sa128_24k
  - _self_

train_dataset_name: cosmic_large_adapted
val_datasets_names: [manual_val]

model:
  # Defaults-off in large_dataset_rhca_hard_v1_audited_20k; E1-E24 unaffected.
  ba_hard_v1_reference_roi_warp: true

datasets:
  train:
    cosmic_large_adapted:
      min_face_res: 192
      reference_crop_margin: null
      reference_content_size: 256
      reference_canvas_size: null
      reference_frame_mode: native
      random_horizontal_flip: true
      random_reference_flip: false
      prompt_mode: pose_first
      prompt_max_words: 55
      require_reference_face_embedding: true
  val:
    manual_val:
      limit: 96

writer:
  experiment_comment: >-
    CL3: exact E13 route on cosmic_large keeping the native 256px reference and
    correcting the measured 2.12x reference/target face-scale mismatch in
    feature space via bbox-frame reference ROI warping. Contingency arm; changes
    the step-zero route, unlike CL1 and CL2.
```

**Gate:** `python train.py --config-name=CL#_... --cfg job` must resolve to
24,000 steps, the 2,240 / 219,217,920 contract, `pose_adapt_ratio=0`,
`ca_mixing_for_face=false`, and exactly the intended deltas. Diff each resolved
config against E13's and attach the diff to the experiment JSON.

### Phase 3 — Preflight and validators

- `tools/datasets/preflight_cosmic_sceneref.py` (CL1) and
  `preflight_cosmic_facecanon.py` (CL2), modelled on
  `preflight_cosmic_large_adapted.py`. Both decode 64 pairs, assert
  `target_path != reference_path`, assert in-bounds bboxes, and — the direct
  check that P1 is fixed — **report the realized
  reference-face / target-face short-side ratio distribution**:
  - CL1: reference-face area must fall in the `large_dataset` band (≈`4–20%`),
    not the `≈42%` Cosmic band;
  - CL2: ratio must be within `[0.95, 1.05]` for `≥95%` of samples, with the
    clamp rate reported.
- Add a caption-token check to all three preflights: on a 2,000-sample draw,
  `< 5%` of emitted prompts may exceed 77 CLIP tokens after `pose_first` +
  `prompt_max_words: 55`.
- `tools/validate_CL1_CL3_config.py`, modelled on
  `tools/validate_E19_E24_config.py`: pin the allowed config names, assert the
  E13-inherited fields, and assert the intended per-arm deltas and nothing else.

### Phase 4 — Experiment records

Create `experiments/cosmic_large/CL1_..._r1.json`, `CL2_..._r1.json` and
`CL3_..._r1.json` following `experiments/large_dataset/E19_*.json`: hypothesis,
fixed controls, the single changed variable, machine, launcher/package, dataset
manifest and identity-groups SHA-256, decision gates, expected trainable
contract, and a `null` Comet key to be filled at startup.

### Phase 5 — Launcher and Serv package

- `launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh`, copied from
  `run_E19_E24_large_ds_24k_1gpu.sh` with:
  - required env: `RUN_NAME`, `CONFIG_NAME`, `EXPERIMENT_SPEC_PATH`,
    `COSMIC_LARGE_MANIFEST`, `COSMIC_LARGE_ROOT`, `COMET_API_KEY`,
    `FACE_QUALITY_SCORER_PYTHON`, plus `COSMIC_IDENTITY_GROUPS` +
    `COSMIC_IDENTITY_GROUPS_SHA256` for CL1;
  - the allowed-`CONFIG_NAME` case statement;
  - `tools/validate_CL1_CL3_config.py`, then the arm's preflight;
  - `prepare_comet_record`, then the same `accelerate launch` block with
    `writer=cometml writer.project_name=aug-large-ds`;
  - no ad-hoc Hydra overrides accepted.
- `serv_run_packages/<RUN_NAME>/start_<RUN_NAME>_1gpu.sh`, copied from the E19
  start script, replacing the `LARGE_DATASET_*` exports with:

  ```bash
  export COSMIC_LARGE_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/bobkov/cosmic_data"
  export COSMIC_LARGE_MANIFEST="${COSMIC_LARGE_ROOT}/gathered_data_cosmic_large_filtered.json"
  export COSMIC_IDENTITY_GROUPS="${OWNER_ROOT}/datasets/cosmic_identity_groups_v1.json"   # CL1
  export COSMIC_IDENTITY_GROUPS_SHA256="<sealed>"                                          # CL1
  ```

  Keep unchanged: the per-run immutable snapshot + `verify_serv_source_manifest.py`
  gate, `PM_PATH`, `FULL96_BBOX_MANUAL`, the `libstdc++`/`GLIBCXX_3.4.32` check,
  the ONNX Runtime `1.20.1` + `CUDAExecutionProvider` + PyIQA `0.1.15` probe,
  `EXPERIMENT_SPEC_PATH` (pointing at `experiments/cosmic_large/`),
  `COMET_PROJECT=aug-large-ds`, and the deferred face-quality contract.
  Add `test -s "${COSMIC_LARGE_MANIFEST}"`, `test -d "${COSMIC_LARGE_ROOT}"`,
  and for CL1 a SHA-256 check on the identity-groups file.
- `serv_run_packages/<RUN_NAME>/run_<RUN_NAME>_1gpu.yaml`, copied from the E19
  MLS YAML: same image
  `cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36`, same
  `a100.1gpu.8C.243G`, `processes: 1`, `workers: 1`, `type: binary`, a new
  description ending in `#nasilaev`, and log/script paths for the new run.

### Phase 6 — Submission

1. Verify the E19–E24 / E17 / E18 jobs' live state and count this project's
   Running + Pending A100 requests. The normal ceiling is **six**. Additional
   one-GPU jobs may need an explicit user-authorized eight-GPU exception —
   **obtain it before submitting**.
2. Use a **separate hash-verified runtime tree per run**, as E19–E24 did. Do
   not mutate any checkout a live job reads.
3. Submit CL2 first (unblocked, zero model change), then CL1 once its Phase-0
   gate passes. Hold CL3 per §5.4.
4. Confirm for each: `saved/<run_name>/comet_experiment.json` exists, the
   immutable key is retrievable by API, the ownership gate reports exactly
   `2,240 / 219,217,920` with `840/840` BA tensors in the optimizer, and the
   step-0 panel is 96 images that are **byte-identical to E13 r4's step 0**
   (CL1 and CL2; CL3 declares its own step-0 value).

---

## 7. Decision gates

Compare against E13 on the same panel and project. E13 reference points:
step 0 `.30212`, best `.39980 @24k`; E11 (best previously-clean arm) `.32704 @8k`.

| Gate | Step | Rule |
|---|---:|---|
| Contract | 0 | CL1/CL2: 96 step-0 PNGs byte-identical to E13 r4. All: ownership `2,240 / 219,217,920`. Otherwise **stop**. |
| Data sanity | 0 | CL1: reference-face-area median in the `large_dataset` band. CL2: face-scale ratio in `[0.95, 1.05]` for ≥95% of samples. All: `<5%` prompts over 77 tokens; reference rejection rate logged. Otherwise **stop**. |
| Early life | 8k | ID `≥ .30212` (E13 step 0). Below it **and falling** → record as a negative result and stop; do not extend. |
| Scientific | 12k | ID `≥ .32704` → the arm is at least as good as the best previously-clean `large_dataset` arm; continue to 24k. |
| Promotion-relevant | 24k | ID within `.02` of E13's `.39980`, **and** the fixed 96-image visual review shows no pasted/oversized/displaced faces, no E10-style subject relocation, and no E12-style face plates. |
| Interpretation | — | Rank by ID **and** per-image visual anatomy. `IDSimBest` can reward a detached identity fragment; body association remains a separate required gate. |

Also log, for every arm: the target↔reference ArcFace cosine of each emitted
pair, the reference-face-area and face-scale-ratio distributions, the
caption-token distribution, the identity-coverage histogram, and (CL2) the
clamp rate and real-vs-reflected canvas fraction. These turn a negative result
into an attributable one.

**Reading the outcome:**

| CL1 `sceneref` | CL2 `facecanon` | Conclusion |
|---|---|---|
| ✅ | ✅ | **Geometry was the whole problem.** A tight 256 crop is fine for BA provided it is framed to the target before the encoder. Adopt `target_face_frame` as the standard policy for crop-only datasets. |
| ✅ | ❌ | Geometry is necessary but **not sufficient**: native resolution and real surrounding context also matter. Cosmic needs source scenes (Phase 0 route 1) or a mixture with a stable-ID scene dataset. |
| ❌ | ✅ | Unlikely; would mean the reframed crop beats a real same-ID scene, implying the identity **grouping** is noisy rather than the asset. Re-audit grouping before concluding anything. |
| ❌ | ❌ | The blocker is Cosmic **content** — identity-retrieval noise, caption quality, or target quality — not reference geometry. Next step is the Priority-0 sealed content audit, not another training arm. |

If CL2 is clearly positive but short of E13, run CL3 to test whether
post-encoder alignment adds anything on top; if CL2 and CL3 are both flat while
CL1 works, the conclusion is that **native reference resolution/context is
irreducible for this route**, which is an architecture finding, not a data one.

---

## 8. Open questions to resolve first

1. **Does the live manifest carry an identity field or a reference source-image
   pointer?** The local one-ID extract has `identity_id`; the 26 July audit says
   the full manifest does not. This single answer decides whether CL1 needs the
   offline grouping tool at all — and if source images exist, it also unlocks
   the native full-scene reference experiment that has been blocked since
   26 July.
2. **How many accepted targets group into identities of ≥2?** Unknown until
   Phase 0 runs. This is CL1's launch gate.
3. **What is the actual reference detection-failure and wrong-face rate on the
   256px crops?** Never measured. Cheap to obtain in Phase 0 and it sizes P7.
4. **Are the canonical full-96 identities present in Cosmic Full?** Still
   unaudited **[report]**. It does not affect CL1/CL2/CL3 comparability (all are
   measured on the same fixed panel) but it does bound any generalization claim.

---

## 9. Implemented files and verification status

Implemented on 6 August 2026. **Two existing files changed, both in the dataset
layer; every change to them is additive and defaults-off.** No model, trainer,
pipeline, loss, or `train.py` change was needed, as predicted in §6.0.

| File | Status | Note |
|---|---|---|
| `src/datasets/cosmic_large_adapted.py` | **modified** | behaviour-preserving extraction of `build_cosmic_prompt` / `load_cosmic_target` for reuse, plus defaults-off `reference_frame_mode`, `reference_frame_fill`, `reference_accept_list_path` |
| `src/configs/datasets/all_datasets.yaml` | **modified** | new `cosmic_large_sceneref` entry; three new `cosmic_large_adapted` keys at historical values |
| `src/datasets/reference_frame.py` | new | CL2 target-frame compositing |
| `src/datasets/cosmic_large_sceneref.py` | new | CL1 loader |
| `src/configs/CL{1,2,3}_*.yaml` | new | leaf configs inheriting E13 |
| `tools/datasets/build_cosmic_identity_assets.py` | new | `groups` (CL1 prerequisite, with the ≥3,000 launch gate) and `accept-list` subcommands; `--write` is explicit |
| `tools/datasets/preflight_cosmic_cl.py` | new | per-arm face-scale/area/caption gates |
| `tools/validate_CL1_CL3_config.py` | new | asserts 40 inherited E13 fields plus the per-arm delta |
| `launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh` | new | validator → preflight → Comet record → train → deferred face quality |
| `serv_run_packages/CL*_full96_r1/{start_*.sh,run_*_1gpu.yaml}` | new | one-A100 MLS packages |
| `experiments/cosmic_large/CL*_full96_r1.json` | new | immutable pre-registration records |

**Rollback:** delete the new files and revert the two modified ones. Because
every new option defaults to the historical value, reverting is not required to
restore old behaviour — no existing config selects any of them.

Verified locally in the `photomaker` env:

- Hydra composition for CL1/CL2/CL3 resolves to 24,000 steps, contract
  `2,240 / 219,217,920`, `pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and
  `roi_warp` `False/False/True` respectively;
- **E13, E19, E3 and `cosmic_large_adapted_rhca` compose unchanged**, and the
  legacy `cosmic_large_adapted` block resolves the new keys to
  `native` / `edge` / `null`;
- a real-fixture smoke test over the local Cosmic one-ID assets passes 24/24
  checks: the native path still emits a 256² reference with a ~42% face area and
  an uncorrected scale mismatch; CL2 emits a 1024² canvas with scale ratio
  `1.0000`, coincident face centres, no flat region, and a target-dependent cache
  key; CL1 emits a native 1024² same-identity scene with ratio `1.000`; and the
  accept-list, groups-SHA, reference-mirroring and conflicting-geometry guards
  all fail closed;
- the validator fails closed on an unapproved config, a mismatched run name, a
  mismatched spec, and a simulated `trainer.n_epochs` drift;
- `bash -n`, `py_compile`, and YAML parsing pass for all new files.

Note: the fill default is `edge` (border replication), not the `reflect` named
earlier in §5.3. Edge replication produces smooth low-frequency surround with no
fabricated face structure; `symmetric` and `gray` remain available as ablations.

### CL2 Serv staging and rejected submission — 6 August 2026

CL2 is fully staged, verified against the **real** 59,143-record Cosmic manifest,
and **not running**. Its single submission was rejected by MLS before job
creation.

- Isolated runtime
  `runtime_sources_cl1_cl3_v1/CL2_cosmic_joint_shadow_sa128_facecanon_24k_full96_r1`,
  built read-only from the verified E19 base; **no live checkout was read or
  mutated**. Sealed manifest: 861 files, revision
  `d903b2c9e92ce1a6f3db7a1f8fccf82c0d1ab21f+cl1-cl3-snapshot-v1-20260806`,
  verify passes. The MLS YAML, start script and code all live inside that
  hash-verified tree, so the job is self-contained.
- Config/spec validator: ok (34 inherited E13 fields, 24,000 steps,
  `target_face_frame`).
- **Real-data preflight: ok.** 22,140 accepted records from 59,143 (37,003
  filtered on target face size, 137 on reference bbox) — exactly the documented
  counts. Reference size `(1024, 1024)`; **face-scale ratio `1.0000` at p10,
  median and p90**; max centre offset `0.50px`; **reference face area median
  `9.23%` against a target median `8.71%`** (the historical figures are `42.6%`
  and `9.49%`). P1's geometry half is therefore measurably fixed on real data.
- Submission at `2026-08-06T23:19:45+01:00` under the user-authorized
  eight-A100 exception scoped to CL2 returned
  `{'error_code': 1, 'error_message': 'WORKSPACE_GPU_LIMIT_REACHED_ONLY_0_FREE'}`.
  This is a **workspace-wide** capacity limit, not this project's ceiling; the
  project remained at 7 A100s. No job, no log directory and no Comet experiment
  were created, so the `r1` run name is uncontaminated and reusable as-is.
  Per `AGENTS.md` this request must not be retried unless the user asks.
  Audit record:
  `local_scripts/serv_job_records/unparsed-submit-2026-08-06T23-19-45+01-00.json`.

### CL3 Serv staging — verified, deliberately not submitted, 6 August 2026

CL3 is staged and verified to the same standard as CL2 in
`runtime_sources_cl1_cl3_v1/CL3_cosmic_joint_shadow_sa128_fmtfix_24k_full96_r1`
(sealed manifest 863 files, same revision; validator ok with
`roi_warp=true`, `frame=native`, 24,000 steps; preflight ok).

**It was not submitted.** The project is at 7 Running A100s, CL2 is authorized as
the 8th, so CL3 would be the **9th — above the eight-A100 hard maximum** in
`AGENTS.md`, which no exception can raise. Submit it when the project drops to
six or fewer, or under a fresh authorization once capacity allows. Both packages
need no further changes.

CL3's real-data preflight is also the cleanest independent confirmation of the
whole diagnosis. On the shared 1024 latent frame it measures a reference/target
face short-side ratio of **p10 `1.300`, median `2.157`, p90 `2.880`**, against
the `2.12` predicted in §2.1 from population medians, with a reference face
occupying `41.78%` of its own frame versus a target median of `8.71%`. CL2, from
the identical source data, measures `1.000` flat. The defect and its removal are
therefore both confirmed on the real 22,140-record dataset.

### Preflight metric corrected — raw bbox ratio → shared-frame ratio

The first CL3 preflight would have wrongly **failed**. The gate compared raw
bbox short sides in each image's own frame, which for a 256px reference gives
`142/272 = 0.52` and understates the mismatch by exactly the 4× letterbox factor
`_encode_reference_latent` applies before the VAE. The metric now multiplies the
reference bbox by `letterbox_scale(reference.size)` so both faces are measured
in the shared 1024 frame, which is the geometry the branch actually sees.
CL1/CL2 references are already 1024, so their scale factor is `1.0` and CL2's
earlier result is unchanged — re-verified after the fix.

### Caption cap corrected 55 → 50 — measured, not assumed

The first real-data preflight **failed**: at `prompt_max_words: 55`, the value
inherited from the historical pose-first policy, `16.5%` of Cosmic captions
still exceeded 77 CLIP tokens. A 3,000-record tokenizer sweep gives:

| `prompt_max_words` | median tokens | p90 | fraction > 77 |
|---|---:|---:|---:|
| none (pose-first) | 89 | 102 | **87.2%** |
| none (legacy) | 88 | — | **86.4%** |
| 55 | 74 | 79 | 17.6% |
| **50** | **67** | **72** | **0.7%** |
| 45 | 61 | 65 | 0.0% |

**Pose-first ordering does essentially nothing for truncation** (`87.2%` versus
legacy's `86.4%`); the word cap alone controls it. Prior reports credited
pose-first with mitigating truncation and never measured it at the token level.
The shared control is now `prompt_max_words: 50` across CL1/CL2/CL3, the
loosest cap meeting the `<5%` gate. §5.1 and the validator were updated.

### Phase 0 question 1 — answered, negatively

A direct audit of the live 59,143-record manifest on Serv shows the record
fields are exactly `face_crop_old`, `face_crop_new`, `body_crop`,
`facial_caption`, `pose_caption`, `background_caption`, `is_simp`,
`body_mask_path`, `face_paths`, `face_bboxes`, `face_scores`.

- There is **no `identity_id`, `person_id` or `id` field**. The 26 July audit was
  right; the `identity_id` in the local one-ID extract was synthesised during
  that extraction.
- There is **no reference source-image pointer**. Reference crops live in a
  directory named after the *target*
  (`LAION-5B-Filtered-Large-Faces/.../<target_stem>_jpg/0-9.jpg`), so the full
  images they were cut from are not recoverable from this manifest.
- The 10 crops per target are distinct files scoring `0.72-0.83` against the
  target, so they are genuinely different photographs of the same person, not
  duplicates of the target's own face.

**Consequences:** CL1 must use the offline ArcFace grouping route (route 2), and
the native full-scene reference experiment (§5.2 route 1, and the 26 July
Priority-4 item) stays blocked on this package. Open question 1 in §8 is closed.

Still outstanding for CL1: run the grouping tool, clear the ≥3,000-target launch
gate, and pin the real `COSMIC_IDENTITY_GROUPS_SHA256` in its start script,
which currently holds `REPLACE_WITH_SEALED_SHA256` and **fails closed**.

## 10. References

- [Current handoff](../docs/handoffs/LATEST.md)
- [Cosmic pipeline performance analysis, 24 Jul](2026-07-24_cosmic_large_pipeline_performance_analysis.md)
- [Cosmic training recommendations, 25 Jul](2026-07-25_cosmic_large_training_recommendations_and_experiments.md)
- [Cosmic full dataset usage recommendations, 26 Jul](2026-07-26_cosmic_full_dataset_usage_recommendations.md)
- [Initial-usage baseline matrix, 26 Jul](../docs/experiments/2026-07-26_cosmic_large_initial_usage_baseline_matrix.md)
- [Full-Cosmic 4k/full-96 results, 26 Jul](../docs/experiments/2026-07-26_cosmic_large_adaptation_4k_full96_results.md)
- [E0–E12 analysis and E3 ROI-warp result](../comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md)
- [E13–E18 results and next experiments, 6 Aug](2026-08-06_e13_e18_results_and_next_experiments.md)
- [E19–E24 implementation and launch, 6 Aug](2026-08-06_e19_e24_implementation_and_launch.md)
