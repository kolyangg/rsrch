# Undersized faces in the CL cosmic_large runs: root cause and fix

**Date:** 8 August 2026
**Scope:** diagnosis of the reported "face too small for its required area,
not aligned to the body" failure across CL0-CL7 versus E13, and two proposed
experiments. No running job was touched; no training code was changed.
**Evidence cutoff:** CL0/CL2/CL3/CL4 complete at 24k; CL5 at 22k; CL6/CL7 at
16k (still running); E13 complete at 24k.

---

## Executive conclusion

The failure is real, measurable, and has a **single sufficient cause**:

> The model renders a face at the scale its *training references* had. Cosmic
> training references are 256px tight crops whose face fills **~42%** of the
> frame. large_dataset references are 1024px scenes whose face fills **~8.6%**.
> The validation references span **6.2%-50.4%**. A model calibrated at 42% is
> badly miscalibrated for the low end of that range, so it renders an undersized
> face for exactly those identities — and only those.

The prediction is exact, with no overlap:

| Validation reference face area | Identities | Undersized faces across 6 CL arms |
|---|---|---:|
| **< 8%** | jensen `6.22%`, jisoo `7.11%`, keanu `7.84%` | **52** |
| **≥ 10%** | tom, lex, elon, michael, jennie, marion, sydney, robert, eddie | **0** |

Those are precisely the three identities you named. **E13 has zero undersized
faces anywhere** because its training references (`8.6%`) sit inside the
validation range rather than far above it.

**Your hypothesis about full bodies is right in effect, but the cause is ours,
not the dataset's.** cosmic_large contains *more* full-body imagery than
large_dataset — `64.6%` of its raw targets have a face under 5% of frame versus
large_dataset's `27.1%`. Our own `min_face_res=192` filter then deletes **96% of
it**. Both defects — the reference scale and the discarded full-body targets —
are preprocessing choices, and both are one-line fixes.

Two experiments follow: **CL8** restores the discarded full-body targets, and
**CL9** calibrates the reference face scale to the inference range without the
copy shortcut that broke CL2.

---

## 1. Which images are problematic

Metric: detected-face short side divided by the **fixed mask box** short side.
The mask *is* the required face area — it comes from a BA-off PhotoMaker pass and
is identical for every run, so the ratio is directly comparable. `1.0` means the
face fills its required area; `0.8` means it is a fifth too small.

| Run | Step | ratio p10 | p25 | median | **< 0.8** | < 0.9 | no face |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E13** | 24k | **`0.941`** | `0.985` | `1.028` | **0** | 3 | 0 |
| CL2 | 24k | `0.908` | `0.974` | `1.034` | **1** | 8 | 1 |
| CL6 | 16k | `0.834` | `0.929` | `1.007` | 8 | 20 | 2 |
| CL4 | 24k | `0.824` | `0.901` | `1.001` | 6 | 23 | 1 |
| CL7 | 16k | `0.812` | `0.926` | `1.007` | 8 | 18 | 3 |
| CL5 | 22k | `0.810` | `0.919` | `1.005` | 8 | 20 | 1 |
| CL3 | 24k | `0.787` | `0.908` | `0.988` | **12** | 23 | 1 |
| CL0 | 24k | `0.784` | `0.879` | `0.972` | **10** | 32 | 1 |

**Medians are all near 1.0 — the failure is entirely in the lower tail.** E13's
p10 is `0.941`; the CL arms sit at `0.78-0.83`. E13 never produces a face below
`0.8`; the CL arms produce 6-12 each.

### Concentration by identity

Undersized (`< 0.8`) counts:

| Run | eddie | elon | jennie | **jensen** | **jisoo** | **keanu** | lex | marion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E13 | 0 | 0 | 0 | **0** | **0** | **0** | 0 | 0 |
| CL0 | 0 | 0 | 0 | 2 | 6 | 2 | 0 | 0 |
| CL3 | 0 | 0 | 0 | 5 | 4 | 3 | 0 | 0 |
| CL4 | 0 | 0 | 0 | 1 | 3 | 2 | 0 | 0 |
| CL5 | 0 | 0 | 0 | 1 | 5 | 2 | 0 | 0 |
| CL6 | 0 | 0 | 0 | 4 | 3 | 1 | 0 | 0 |
| CL7 | 0 | 0 | 0 | 1 | 5 | 2 | 0 | 0 |
| CL2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

**Every single undersized face in every CL arm belongs to jensen, jisoo or
keanu** (one jennie case in CL2 aside). Five of the eight identities are never
affected.

### The failure is invisible to `id_sim`

The worst CL0 cases, with E13's ratio on the same image for reference:

| Image | CL0 ratio | E13 ratio | CL0 `id_sim` |
|---|---:|---:|---:|
| `Kickboxing_jisoo` | `0.70` | `0.97` | **`0.36`** |
| `Night-ride_jisoo` | `0.73` | `0.94` | **`0.41`** |
| `Reading_pa_jisoo` | `0.74` | `0.93` | **`0.47`** |
| `Kickboxing_keanu` | `0.74` | `0.97` | **`0.40`** |
| `Skiing_man_keanu` | `0.74` | `0.98` | **`0.38`** |
| `Laughing_m_jensen` | `0.78` | `1.00` | **`0.53`** |

These are badly undersized and score **well**. `IDSimBest` takes the
best-matching detected face anywhere in the image and never checks that it fits
the body, so a small, well-formed, correctly-identified face is rewarded.

This is why the metric now actively misleads: at 24k, **CL0 reaches `.40190`
and CL4 `.40387`, both above E13's `.39980`** — while being visibly worse.

Visual comparison (red box = the identical fixed mask in every column):

![Face size versus required area: E13 fills the mask box, the CL arms sit inside it](assets/problem_grid.png)

E13's face fills the box; the CL faces sit noticeably inside it, with the
surplus becoming forehead, hair or a stretched neck.

---

## 2. Which runs are better, and why

Ranked by undersized count (lower is better):

```text
E13   0    reference face fraction ~8.6%   <- inside the validation range
CL2   1    composited to target scale ~9.5% <- best CL arm on this metric
CL4   6
CL5   8   CL6  8   CL7  8
CL0  10
CL3  12
```

**CL2 is the standout and it is the key piece of evidence.** CL2 is the only arm
that rescales the reference so its face matches the target's scale (~9.5%),
and it is the only arm that essentially eliminates undersized faces (1 versus
6-12). It is otherwise the *worst* arm — `id_sim .28255` at 24k against CL0's
`.40190` — so it is not a candidate for promotion, but it is a clean natural
experiment: **change the training reference face fraction from 42% to ~9.5% and
the undersized-face problem disappears.**

CL3/CL5's ROI warp does *not* help (12 and 8). That is consistent: the warp
corrects geometry *inside the network*, after the frozen encoder has already
formed features from a 42%-face image. It changes what the branch reads, not
what scale the model learns to render.

CL6/CL7's loss changes do not help either (8 and 8). Restoring supervision
outside the face box addresses the *junction*, not the *scale*. Both are early
(16k) so their final numbers may move, but neither shows a scale effect.

---

## 3. Root cause

### 3.1 The reference scale mismatch (sufficient cause)

| Lane | Face fraction of the reference frame |
|---|---:|
| cosmic 256px crop — **CL0, CL3-CL7** | **~42%** |
| large_dataset 1024 scene — **E13** | **~8.6%** |
| CL2 composited to target frame | ~9.5% |
| **Validation references (what inference supplies)** | **`6.22%` - `50.35%`, median ~`19.9%`** |

The branch learns a mapping from reference-face appearance to rendered-face
appearance at whatever scale it is trained on. Trained at 42%, it is
extrapolating badly when handed a 6-8% reference. E13, trained at 8.6%, is
interpolating across the whole validation range.

The per-identity separation is exact — every identity below 8% fails, every
identity at or above 10.32% is clean — and CL2 provides the interventional
confirmation.

### 3.2 The discarded full-body targets (compounding cause)

Your instinct about full bodies is correct, but inverted in origin:

| | targets | face < 5% of frame | median face area |
|---|---:|---:|---:|
| cosmic **raw** | 59,143 | **`64.6%`** | `3.25%` |
| cosmic **after `min_face_res=192`** | 22,140 | **`6.1%`** | `9.49%` |
| large_dataset (no such filter) | 47,500 | `27.1%` | `8.56%` |

The filter drops 37,003 targets, and **99.6% of those are full-body framing.**
It removes **96% of all the full-body imagery cosmic has** (36,843 of 38,197).

So every CL arm trains on an almost purely portrait-framed corpus and never
learns to place a small face on a full body. The failing prompts are exactly the
wide-framing ones — Kickboxing, Skiing, Jumping, Rushing, Night-ride. The filter
was adopted on 26 July as a deliberately conservative starting point and
explicitly never validated: *"no controlled target-scale curriculum or
scale-balanced sampler has been tested."*

This also costs 2.7x the data and pushes CL runs to ~2.17 epochs at 24k versus
E13's ~1.0, adding memorisation pressure.

### 3.3 Why every previous CL arm missed it

CL1-CL7 varied the **reference geometry lane** (crop policy, framing, ROI warp,
multi-reference) and the **loss** (boundary ring, alternating). Every one of
them inherited the same `min_face_res=192` filter and, except CL2, the same 42%
reference face fraction. **The two variables that actually matter were held
constant across the entire suite.**

### 3.4 Confidence

| Claim | Confidence |
|---|---|
| Undersized faces are real, CL-only, confined to jensen/jisoo/keanu | **High** — measured; E13 has zero |
| `id_sim` does not detect it | **High** — undersized images score `.36-.53` |
| Reference face fraction is the cause | **High** — exact per-identity separation plus CL2's interventional result |
| The 192px filter removes cosmic's full-body distribution | **High** — counted directly from the manifest |
| Restoring full-body targets will improve alignment | **Medium** — strong mechanism, not yet tested |

---

## 4. Proposed experiments

Both are **dataset-side**, both build on **CL4** (best clean arm: `id_sim
.40387` at 24k, only 6 undersized), and both keep the E13 contract:
24k steps, batch 2, one A100, fixed full-96, `2,240 / 219,217,920` parameters.

### CL8 — restore the full-body target distribution (priority 1)

`CL8_cosmic_joint_shadow_sa128_fullbody_24k.yaml`

**Change:** `min_face_res: 192 → 64`, plus a scale-balanced sampler so the
restored small-face targets are not swamped.

```yaml
datasets:
  train:
    cosmic_large_adapted:
      min_face_res: 64                 # was 192; restores ~37k full-body targets
      target_scale_balance: true       # NEW, defaults-off
      target_scale_bins: [0.0, 2.0, 5.0, 10.0, 20.0, 100.0]   # face area %
```

**Why first:** it is the largest single defect, it is a one-line data change, it
recovers 2.7x the training data, and it directly supplies the wide-framing
compositions the failing prompts need.

**Risk:** the 26 July note warns small/high-motion faces were historically hard.
The scale-balanced sampler is the mitigation — it keeps large faces represented
rather than letting the 64.6% small-face majority dominate. If instability
appears, `min_face_res: 128` is the fallback.

**Prediction:** undersized faces drop below CL4's 6; the wide-framing prompts
(Kickboxing, Skiing, Jumping) improve most.

### CL9 — calibrate reference face scale to the inference range (priority 2)

`CL9_cosmic_joint_shadow_sa128_refscale_24k.yaml`

**Change:** reuse CL2's compositing, but fix the two things that made CL2 fail.

```yaml
datasets:
  train:
    cosmic_large_adapted:
      reference_frame_mode: target_face_frame
      reference_frame_fill: edge
      reference_scale_jitter: [0.06, 0.30]   # NEW: sample face fraction from
                                             # the inference range, not the
                                             # target's exact scale
      reference_position_jitter: 0.35        # NEW: break the positional
                                             # copy shortcut
```

**Why:** CL2 proved scale control eliminates undersized faces (1 vs 6-12). Its
`id_sim` collapse is attributable to two things it also did: it placed the
reference face at *exactly* the target's position, letting the branch learn a
trivial positional copy that does not transfer to validation; and it locked to a
single scale rather than covering the inference range. Jittering both keeps the
scale calibration while removing the shortcut.

**Prediction:** undersized faces near CL2's 1, with `id_sim` near CL4's `.40`.
If `id_sim` still collapses, the fabricated surround is the culprit and the
`gray`/`symmetric` fill ablation isolates it.

**CL8 and CL9 are independent and can run in parallel.** If both work, a
combined arm is the natural follow-up.

### Not proposed

- **More ROI-warp or loss variants.** CL3/CL5/CL6/CL7 show neither addresses
  scale.
- **Changing the validation protocol** (dynamic masks, wider `mask_expansion_ratio`).
  It would improve the *measurement* but breaks comparability with every prior
  arm and fixes nothing in training. Worth a separately named diagnostic sidecar.
- **Reviving CL1.** Identity grouping failed its gate: 1,876 targets against a
  3,000 floor, ~half of those pairs being the same photograph twice.

---

## 5. Implementation plan

### Step 1 — dataset code (`src/datasets/cosmic_large_adapted.py`)

Three new **defaults-off** constructor arguments; defaults reproduce today's
behaviour exactly.

```python
target_scale_balance: bool = False,
target_scale_bins: Sequence[float] | None = None,
reference_scale_jitter: Sequence[float] | None = None,   # (min, max) face fraction
reference_position_jitter: float = 0.0,                  # fraction of canvas
```

**CL8 — scale-balanced sampling.** In `__init__`, after the accept loop, bucket
records by target face-area percent using `target_scale_bins`. When
`target_scale_balance` is true, build the index by round-robin over non-empty
buckets so each scale band contributes comparably. Log realised bucket counts in
`self.audit`. Keep it deterministic: no RNG in `__init__`, order derived from the
existing manifest order.

**CL9 — reference scale/position jitter.** In `compose_target_frame_reference`
(`src/datasets/reference_frame.py`), add two optional parameters:

- `target_face_fraction`: when given, size the reference face to that fraction of
  the canvas instead of matching the target's short side.
- `position_jitter`: offset the paste centre by up to that fraction of the
  canvas, clamped so the reference face stays fully inside.

In `__getitem__`, when `reference_scale_jitter` is set, draw the fraction
uniformly from the configured range and pass it through, and draw the position
offset from `reference_position_jitter`. Both must enter
`reference_cache_key` via the policy descriptor — the composed reference is
already target-dependent and must not be cached across samples.

Keep the existing `[0.95, 1.05]` realised-ratio assertion for the non-jitter
path; under jitter, assert the realised fraction is within 10% of the requested
one instead.

### Step 2 — registry

Add the four arguments to `cosmic_large_adapted` in
`src/configs/datasets/all_datasets.yaml` with defaults
`false / null / null / 0.0`.

### Step 3 — configs

`CL8_...yaml` and `CL9_...yaml`, both `defaults: [CL4_cosmic_joint_shadow_sa128_hygiene_24k, _self_]`
so every shared control is inherited (native 256px asset, mirroring off,
pose-first captions capped at 50, no ROI warp). Only the fields above differ.

### Step 4 — validator (`tools/validate_CL1_CL3_config.py`)

Add both to `ARMS`. Assert per arm:

- CL8: `min_face_res == 64`, `target_scale_balance is True`, non-empty bins,
  `reference_frame_mode == "native"`.
- CL9: `reference_frame_mode == "target_face_frame"`, jitter range inside
  `[0.03, 0.60]`, `reference_position_jitter > 0`, and — as CL2 requires —
  `reference_crop_margin/content_size/canvas_size` all null.

CL8 changes `min_face_res`, which the shared-controls check currently pins to
192; exempt CL8 there explicitly rather than loosening it for everyone.

### Step 5 — preflight (`tools/datasets/preflight_cosmic_cl.py`)

Add `CL8`/`CL9` to the arm branch and add two gates that would have caught this
whole problem earlier:

- **report the realised reference face-fraction distribution** (p10/median/p90)
  and require the median to fall within `[0.06, 0.30]` for CL9 — the inference
  range. For CL8 report it for information.
- **report the realised target face-area distribution**, and for CL8 require at
  least 20% of sampled targets below 5% face area, confirming full-body framing
  actually returned.

Also print `accepted_records`; CL8 should land near 59,143 rather than 22,140.

### Step 6 — launcher, records, packages

Add both config names to the `case` gate in
`launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh`; write
`experiments/cosmic_large/CL{8,9}_..._r1.json`; clone the Serv package from
CL4's — isolated runtime under `runtime_sources_cl1_cl3_v1/<run>`, sealed
manifest, self-contained MLS YAML.

### Step 7 — pre-launch gates

1. Hydra composition: 24,000 steps, `2,240 / 219,217,920`.
2. Only the intended fields differ from CL4.
3. Real-data preflight passes, including the two new distribution gates.
4. A dataset smoke test showing `target_scale_balance=false` and
   `reference_scale_jitter=null` reproduce current behaviour byte-for-byte.
5. Step-0 panel identical to CL4's (`id_sim .30187`) — neither arm changes the
   inference route.

### Step 8 — decision gates

**Primary metric is face scale, not `id_sim`.**

| Gate | Step | Rule |
|---|---:|---|
| Data | preflight | CL8: ≥20% of targets below 5% face area, ~59k accepted. CL9: reference face fraction median in `[0.06, 0.30]` |
| Scale | 4k | undersized (`ratio<0.8`) below CL4's 6; measure with `measure_face_body_alignment.py` at **matched steps** |
| Identity trio | 8k | jensen/jisoo/keanu undersized counts trending to zero |
| Identity | 8k | `id_sim` within `0.02` of CL4's trajectory |
| Promotion | 24k | zero or near-zero undersized, `id_sim` ≥ CL4 |

Report **undersized count alongside `id_sim` in every future comparison.**
On current evidence `id_sim` alone selects CL0/CL4 over E13, which is wrong.

---

## 6. Reproducing

```bash
# face size vs required area, per run
python tools/datasets/measure_face_body_alignment.py \
  --images-dir <canonical export dir> \
  --mask-boxes ../dataset_full/val_dataset/pm96_bboxes_new.json \
  --id-sim-csv <id_sim_step_XXXXXX.csv> --label CL4 --output alignment.json
```

Two joining traps, both silent:

- bbox keys and `id_sim` `output_key` contain **spaces** (built from
  `prompt[:10]`); exported PNGs use **underscores**. Matching literally drops
  ~83% of images while appearing to succeed.
- The Comet **API** returns figure names for image assets. Use
  `tools/comet/comet_experiment.py fetch` for anything joining to bbox or
  `id_sim` tables.

Data: `comet_data/cl_facesize_20260808/` — per-image rows, the eight
96-image panels, and `problem_grid.png`.

## 7. References

- [Alignment analysis, 7 Aug](2026-08-07_cl_face_body_alignment_analysis_and_cl6_cl7.md)
- [CL2/CL3 early results, 7 Aug](2026-08-07_cl2_cl3_early_results_and_next_cosmic_experiments.md)
- [Root cause and CL1-CL3 plan, 6 Aug](2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md)
- [Cosmic full dataset usage recommendations, 26 Jul](2026-07-26_cosmic_full_dataset_usage_recommendations.md) — origin of the 192px filter
