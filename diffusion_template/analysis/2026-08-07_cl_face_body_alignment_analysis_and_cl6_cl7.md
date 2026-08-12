# Face/body misalignment in the CL cosmic_large runs

**Date:** 7 August 2026
**Scope:** measurement and diagnosis of the reported face/body misalignment
across CL0-CL5 versus E13, plus two proposed experiments. No running job was
touched; no training code was changed.
**Evidence cutoff:** each run's latest complete validation. **CL4 and CL5 are
only at 4k of 24k**, so their numbers are early and not comparable as endpoints.

| Run | Comet key | Latest step | `id_sim` |
|---|---|---:|---:|
| CL0 `asis` | `511f40bc339e425d87869f3a81eef0b2` | 6,000 | `.37666` |
| CL2 `facecanon` | `be7b7a2acf174b69b5e361490926140e` | 10,000 | `.29096` |
| CL3 `fmtfix` | `488ec4fdee5b4560a77f3924af3e0b6e` | 10,000 | `.37646` |
| CL4 `hygiene` | `0dd86b436b224f939efa3887ad6acbe2` | 4,000 | `.37576` |
| CL5 `roiwarp+multiref` | `2851395f018e4613b39a6565a92a89c6` | 4,000 | `.34936` |
| E13 (large_dataset) | `1cc0a02371094b24a6a02a4cc649f10c` | 24,000 | `.39980` |

Export: `comet_data/cl_alignment_20260807/`. Measurement tool:
`tools/datasets/measure_face_body_alignment.py`.

---

## Executive summary

The reported failure is real, measurable, and **present in every cosmic arm and
absent from E13**. It is invisible to `id_sim`, which is why it survived four
rounds of analysis.

- E13's face-centre offset from the fixed mask box has p90 `0.095` and **zero**
  images beyond `0.25`.
- Every CL arm has **3-4x more images outside E13's own envelope**: CL4 28,
  CL3 32, CL2 32, CL5 35, CL0 37 (of ~95).
- **17-25 of those still score `id_sim > 0.3`.** A well-formed face in the wrong
  place scores well, because `IDSimBest` takes the best-matching face anywhere in
  the image and never checks it sits on the body.

**The mask contract is identical across E13 and all CL arms**, so a stale mask
cannot be the differentiator. The asymmetry is that `masked_loss_step: 1` makes
the loss `_masked_face_mse(...)` at *every* step, so **nothing outside the face
box is ever supervised** — while only the cosmic arms feed the branch a face at
2.12x the target's scale. The in-mask optimum is therefore a face that cannot
meet the shoulders, and no gradient ever objects. That is precisely the "anchored
early, never corrected" dynamic you described.

Both proposed experiments restore a gradient outside the face box. Both are
**config-only**, reusing loss code already verified by E14 and E24.

---

## 1. Which images are problematic

Measured per image: detected-face centre offset from the cached mask box
(in units of mask width), signed vertical component, size ratio, and IoU.

### Alignment versus E13's envelope (p90 = `0.095`)

| Run | Step | Outside E13 envelope | of those `id_sim>0.3` | No face | Offset median | Offset p90 |
|---|---:|---:|---:|---:|---:|---:|
| **E13** | 24,000 | **9/96** (by definition ~10%) | 4 | 0 | `0.055` | `0.093` |
| CL4 | 4,000 | **28/95** | 18 | 1 | `0.059` | `0.163` |
| CL3 | 10,000 | 32/95 | 17 | 1 | `0.059` | `0.172` |
| CL2 | 10,000 | 32/95 | n/a¹ | 1 | `0.068` | `0.159` |
| CL5 | 4,000 | 35/94 | 20 | 2 | `0.074` | `0.192` |
| **CL0** | 6,000 | **37/96** | 25 | 0 | `0.067` | `0.195` |

¹ CL2's per-image `id_sim` table did not join (its export lacked matching keys);
its offset numbers are valid, the `id_sim` cross-tab is unavailable.

Median offsets are all similar — **the problem is entirely in the tail.** E13's
p90 is `0.093`; every CL arm sits at `0.16-0.20`, roughly double.

### The recurring set

Eleven images fall outside E13's envelope in **all five** CL arms:

```text
Angry_man__jensen      Crying_man_jensen     Dancing_ma_eddie
Dancing_ma_elon        Jumping_ma_eddie      Jumping_ma_keanu
Laughing_m_jensen      Night-ride_marion     Rushing_ma_jensen
Skiing_man_jensen      Skiing_wom_marion
```

**`Rushing_ma_jensen` — one of your two screenshots — is in this universal set.**
The other, `Rushing_ma_keanu`, is affected in CL0/CL3/CL5.

**jensen accounts for 5 of the 11.** Its validation reference is the most extreme
of the twelve: `600x337` (widest aspect) with the **smallest face at `6.22%` of
area, `99px` short side**. Letterboxed to 1024 it gets heavy top/bottom padding
and the face lands on very few latent cells.

Worst individual cases (offset, `id_sim`):

| Run | Worst images |
|---|---|
| CL5 | `Kickboxing_jensen` 0.51 / .04 · `Crying_man_jensen` 0.32 / .38 · `Skiing_man_jensen` 0.32 / .02 |
| CL3 | `Kickboxing_jensen` 0.39 / .09 · `Crying_man_jensen` 0.28 / .10 · `Kickboxing_keanu` 0.26 / .12 |
| CL0 | `Skiing_wom_jisoo` 0.32 / .06 · `Dancing_ma_eddie` 0.30 / .11 · `Kickboxing_jisoo` 0.29 / **.35** |
| CL4 | `Dancing_ma_eddie` 0.28 / .01 · `Crying_man_jensen` 0.26 / **.55** · `Rushing_ma_jensen` 0.22 / **.56** |

The bolded rows are your exact complaint: **badly misaligned, healthy `id_sim`**.
`Rushing_ma_jensen` at offset `0.22` with `id_sim .56` in CL4 is the screenshot.

### Direction

Large offsets are overwhelmingly **negative in `dy`** — the detected face sits
*above* the mask centre (CL5 `Kickboxing_jensen` dy `-0.41`, CL3
`Kickboxing_jensen` dy `-0.32`, CL0 `Skiing_wom_jisoo` dy `-0.25`). This is the
floating-head signature in the screenshots: head high, neck stretched or absent.

Prompts cluster too: **Kickboxing, Skiing, Jumping, Dancing, Crying, Rushing** —
all extreme pose, occlusion or motion, where the body composition departs most
from a plain portrait.

---

## 2. Which runs are best and worst

Ranked by images outside E13's envelope (lower is better):

```text
E13   9/96   <- reference, no tail at all
CL4  28/95   best cosmic arm  (4k, early)
CL3  32/95
CL2  32/95   (but id_sim collapsed to .29 - failing for other reasons)
CL5  35/94   (4k, early)
CL0  37/96   worst - and the highest id_sim
```

Two cautions:

- **CL4 and CL5 are at 4k against CL0's 6k and CL3's 10k.** Ranking across
  different steps is indicative only. The honest same-step comparison available
  today is CL0 (6k) versus CL3 (10k) — and CL3 is better on alignment despite
  more training.
- **CL0 is the worst on alignment while leading on `id_sim`** (`.38780 @4k`,
  above E13's `.36454 @4k`). That inversion is the whole point: the unimproved
  baseline wins the metric partly *by* pasting well-formed faces that do not
  belong to the body. Its 25 misaligned-but-high-`id_sim` images are the most of
  any arm.

**Do not promote on `id_sim` alone.** On this evidence CL0 would win and it is
the least visually correct.

---

## 3. Why this happens, and why not in E13

### What is NOT the cause

I checked the obvious suspect first and ruled it out. The validation mask
contract is **identical** for E13 and every CL arm:

```text
automatic_bboxes=True   automatic_bboxes_every_val=False
use_bbox_mask_gen=True  use_dynamic_mask=False
mask_expansion_ratio=1.0   branched_attn_start_step=15/50
```

All 96 generation boxes are computed once from a BA-off PhotoMaker pass and
reused at every validation step, for every run. A stale mask is a **shared
condition**, so it cannot explain a difference between E13 and the CL arms. It
is the mechanism through which damage becomes visible, not the cause.

### The actual asymmetry: an unsupervised region meets a mis-scaled reference

Two facts combine.

**Fact 1 — nothing outside the face box is ever supervised.** Every arm here uses
`masked_loss_step: 1`, so `is_masked_loss` is always true and the loss is:

```python
loss = _masked_face_mse(model_pred, target, face_bbox)   # face box only
```

The neck, jaw, shoulders and their junction with the head receive **zero
gradient at every step of training**.

**Fact 2 — only the cosmic arms feed the branch a mis-scaled face.** On the
shared 1024 latent frame the reference/target face short-side ratio is:

| | reference : target |
|---|---:|
| E13 (large_dataset) | **`1.00`** |
| CL0 / CL4 (native 256px crop) | **`2.18`** measured |
| CL3 / CL5 | `2.14` in pixels, corrected in feature space by the ROI warp |
| CL2 | `1.00` in pixels, but ~78% of its canvas is fabricated |

On `large_dataset` the reference face already arrives at the target's scale, so
whatever satisfies the in-mask objective is *automatically* body-consistent —
E13 gets alignment for free and never needs a gradient outside the box. On
cosmic the branch is handed a face at ~2.2x the target scale; the in-mask
optimum is a large face that physically cannot meet the shoulders, and **Fact 1
guarantees nothing ever penalises the mismatch.**

That is exactly your intuition about anchoring. The branch takes over the masked
region within the first couple of thousand steps, and from then on the
disconnection is invisible to the objective. It is not corrected later because
there is no term that could correct it.

### Supporting observations

- **CL3/CL5 correct the scale in feature space and still misalign.** The warp
  fixes what the branch *reads*; it does nothing about what the model *writes*
  outside the box. Consistent with Fact 1 being the binding constraint.
- **CL2 corrects scale in pixel space and is the worst on `id_sim`** while
  mid-pack on alignment — its fabricated surround is a separate problem.
- **Failures concentrate on extreme-pose prompts** (Kickboxing, Skiing, Jumping),
  where the body departs furthest from a portrait and the unsupervised junction
  is hardest to get right by luck.
- **Composition statistics differ between the datasets.** Vertical face-centre
  position in the 1024 training target: cosmic median `0.349`, large_dataset
  median `0.288` — cosmic faces sit **62px lower** in frame. This is a real
  distribution shift the model learns, and it is plausibly why the cached
  base-composition mask fits cosmic-trained output less well. **It does not by
  itself explain the observed upward drift** (the sign is opposite), so I record
  it as a contributing factor, not the mechanism.

### Confidence

| Claim | Confidence |
|---|---|
| Misalignment is real, measurable, CL-only | **High** — measured, E13 has zero images past 0.25 |
| `id_sim` does not detect it | **High** — 17-25 images per arm are misaligned with `id_sim>0.3` |
| Face-only loss leaves the junction unsupervised | **High** — read from the loss code |
| Reference scale mismatch makes the in-mask optimum body-inconsistent | **Medium-high** — mechanism is sound and fits the E13 contrast, not yet isolated |
| Vertical composition shift contributes | **Low-medium** — direction does not match the observed drift |

---

## 4. What to test

The fix must put a gradient back outside the face box. Both experiments do that
and change **nothing else**, so they are clean single-factor tests against CL3.

Base: **CL3**, the best-aligned arm with a healthy `id_sim` trajectory
(`.37646 @10k` and still climbing). CL0 is a worse base despite higher `id_sim`
because it is the worst on alignment and carries mirroring plus uncapped
captions.

### CL6 — `CL6_cosmic_joint_shadow_sa128_boundary_24k` (priority 1)

CL3 plus E14's protected reconstruction loss:

```yaml
loss_kind: branched_reference
loss_function:
  _target_: src.loss.branched_reference_loss.BranchedReferenceLoss
  face_weight: 1.0
  full_weight: 0.1
  boundary_weight: 0.05
  boundary_ring_width: 2
  reference_weight: 0.0
```

The `boundary_ring_width: 2` term supervises a two-latent-cell ring **around the
face box** — precisely the neck/jaw junction that is currently unsupervised —
and `full_weight: 0.1` adds weak whole-image supervision so the body cannot
drift freely. This is the most directly targeted intervention available, and the
code is already verified: E14 ran it on large_dataset and reached `.39185`
against E13's `.39980`, i.e. it costs little identity.

**Prediction:** images outside E13's envelope drop from 32 toward E13's ~9, with
`id_sim` within ~0.01 of CL3. If alignment does not improve, the unsupervised
region is *not* the binding constraint and hypothesis 1 is wrong.

### CL7 — `CL7_cosmic_joint_shadow_sa128_altloss_24k` (priority 2)

CL3 plus E24's exact alternating loss:

```yaml
loss_kind: masked_alternating_audited
trainer:
  masked_loss_step: 2
```

Odd batches use full-latent MSE, supervising the entire image including the
head/shoulder junction; even batches stay face-only. This is a blunter but
stronger signal than CL6's weighted ring, and it tests whether the junction
needs *equal* supervision rather than a small weighted term.

**CL6 versus CL7 is informative either way:** if the narrow ring suffices, CL6
wins on identity; if the junction needs full supervision, CL7 wins on alignment.

### Deliberately not proposed

- **Dynamic validation masks** (`automatic_bboxes_every_val=true`). This would
  re-detect the box each validation and make the *measurement* fairer, but it
  changes the validation contract and breaks comparability with every prior
  arm — and it fixes nothing in training. Worth doing as a **separately named
  diagnostic sidecar**, as E10V did, never as a change to these runs.
- **Widening `mask_expansion_ratio`.** Same objection: it alters validation for
  all arms.
- **More geometry work.** CL3 already corrects scale and still misaligns; the
  evidence points at the objective, not the reference.

---

## 5. Implementation plan

Both arms are **config-only**. No dataset change, no model change, no new
parameters — the trainable contract stays `2,240 / 219,217,920`.

### Step 1 — configs

`src/configs/CL6_cosmic_joint_shadow_sa128_boundary_24k.yaml` and
`CL7_cosmic_joint_shadow_sa128_altloss_24k.yaml`, both `defaults: [CL3_..., _self_]`
so every cosmic control (native 256px asset, ROI warp on, mirroring off,
pose-first captions capped at 50) is inherited unchanged. Written already — see
the files beside this report.

The one thing to get right is the writer's `loss_names`, which must list the new
components or they will not be logged (E17 lost its telemetry to exactly this):

```yaml
# CL6
writer:
  loss_names: [loss, loss_face, loss_full, loss_boundary]
# CL7
writer:
  loss_names: [loss, loss_face, loss_full, loss_mode_face,
               active_grad_norm_ba, active_grad_norm_generic_adapter,
               active_grad_norm_photomaker_default]
```

### Step 2 — validator

Add both names to `ARMS` in `tools/validate_CL1_CL3_config.py` as
`("cosmic_large_adapted", True, "native")`. **Remove `trainer.masked_loss_step`
from the `INHERITED` tuple**, or CL7 will fail the E13-parity check — it
deliberately changes that field. Add an assertion instead that only CL7 may
differ, so the parity check stays fail-closed for the others.

### Step 3 — preflight

No change. Both arms use the CL3 dataset configuration, which the preflight
already covers (`CL0/CL3/CL4/CL5` branch, native 256px, ratio > 1.5).

### Step 4 — launcher

Add both config names to the `case` gate in
`launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh`.

### Step 5 — records and packages

One `experiments/cosmic_large/CL{6,7}_..._r1.json` each, then a Serv package
cloned from CL3's — isolated runtime under
`runtime_sources_cl1_cl3_v1/<run>`, sealed source manifest, self-contained MLS
YAML pointing inside that runtime.

### Step 6 — gates before launch

1. Hydra composition resolves to 24,000 steps and `2,240 / 219,217,920`.
2. The **only** diffs versus CL3 are `loss_kind`, `loss_function` /
   `masked_loss_step`, and `writer.loss_names`.
3. `loss_kind` and the loss target instantiate in the Serv environment — the
   E14/E18 r1 failures were caused by a missing `loss_kind` silently falling
   back to `MaskedDiffusionLoss` while keeping the protected kwargs.
4. Step-0 panel identical to CL3's (`id_sim .32788`) — neither arm changes the
   inference route, so a difference means drift.
5. First three batches log finite `loss_face` **and** the new component
   (`loss_boundary` for CL6, `loss_full` for CL7). A zero or missing component
   means the loss silently degraded.

### Step 7 — decision gates

Primary metric is **alignment**, per your instruction, with `id_sim` secondary:

| Gate | Step | Rule |
|---|---:|---|
| Telemetry | 3 batches | new loss component present and non-zero |
| Alignment | 4,000 | images outside E13's `0.095` envelope **below CL3's 32**; re-measure with `measure_face_body_alignment.py` at matched steps |
| Recurring set | 4,000 | the 11 universally-affected images improve, especially the five jensen cells |
| Identity | 4,000 | `id_sim` within `0.02` of CL3's `.34294 @4k` — a large drop means the boundary term is over-weighted |
| Promotion | 12,000+ | alignment approaching E13 with `id_sim` at or above CL3 |

**Always compare alignment at matched steps.** CL4/CL5 at 4k versus CL0 at 6k
and CL3 at 10k is the main weakness of the table in §2 and should not be
repeated.

### Capacity

The project is at **11 A100** under two stacked one-off exceptions, already
above the eight-A100 maximum in `AGENTS.md`. **CL6/CL7 should not be submitted
until the E19-E24 suite finishes** and usage returns to normal.

---

## 6. Reproducing this analysis

```bash
python tools/datasets/measure_face_body_alignment.py \
  --images-dir <canonical export dir> \
  --mask-boxes ../dataset_full/val_dataset/pm96_bboxes_new.json \
  --id-sim-csv <id_sim_step_XXXXXX.csv> --label CL3 --output alignment.json
```

Two joining traps, both of which fail silently and both of which bit this
analysis before being fixed:

- **bbox keys contain spaces** (`"Rushing ma_jensen.png"`, built from
  `prompt[:10]`), while exported PNGs use underscores. Matching literally
  silently drops ~83% of images. `normalize_key()` handles it; the same trap
  applies to the per-image `id_sim` CSV `output_key` column.
- The Comet **API** returns figure names for image assets, not output keys. Use
  `tools/comet/comet_experiment.py fetch` for anything that must join to bbox or
  `id_sim` tables.

## 7. References

- [CL2/CL3 early results, 7 Aug](2026-08-07_cl2_cl3_early_results_and_next_cosmic_experiments.md)
- [Root cause and CL1-CL3 plan, 6 Aug](2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md)
- [E10 fixed-mask drift precedent](2026-08-04_e10_face_position_and_static_mask_drift.md)
- Export and per-image data: `comet_data/cl_alignment_20260807/`
