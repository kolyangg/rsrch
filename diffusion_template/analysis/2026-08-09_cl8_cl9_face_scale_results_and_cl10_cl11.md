# CL8/CL9 results: reference scale fixes the small-face failure, target framing does not

**Date:** 9 August 2026
**Scope:** analysis of CL8 and CL9 against CL4/CL0/E13 at the matched step 8,000,
and two follow-up experiments. No running job was touched; no training code was
changed for this analysis.
**Evidence cutoff:** CL8 and CL9 at step 8,000 of 24,000 — both still running.
CL0/CL4/E13 are complete but are compared **at 8,000** so every number is
step-matched.

| Run | Comet key | Change vs CL4 | `id_sim` @8k |
|---|---|---|---:|
| E13 | `1cc0a02371094b24a6a02a4cc649f10c` | large_dataset (reference control) | `.38799` |
| CL0 `asis` | `511f40bc339e425d87869f3a81eef0b2` | pre-CL loader | `.39215` |
| CL4 `hygiene` | `0dd86b436b224f939efa3887ad6acbe2` | base for CL8/CL9 | `.39391` |
| **CL8 `fullbody`** | `a6b5970aa1a24d3490ad08e7994b5f1e` | `min_face_res` 192→64 | **`.39640`** |
| **CL9 `refscale`** | `81bb311ed70545eda3281c64bc48be47` | reference face scale/position jitter | `.38619` |

---

## Executive conclusion

**CL9 solves the small-face failure. CL8 does not.**

| Arm | undersized faces (`ratio < 0.8`) | ratio p10 |
|---|---:|---:|
| E13 | **0** | `0.937` |
| **CL9** | **1** | **`0.933`** |
| CL8 | **12** | `0.779` |
| CL4 | 11 | `0.791` |
| CL0 | 11 | `0.781` |

CL9 reduces undersized faces from CL4's 11 to **1**, and its p10 (`0.933`)
essentially matches E13 (`0.937`). **jensen, jisoo and keanu — the three
identities that failed in every previous CL arm — are at zero.** It costs
`.0077` of `id_sim` against CL4, which is negligible next to CL2's `.11`
collapse from the same compositing without jitter.

CL8 changes nothing: 12 undersized against CL4's 11 and CL0's 11, p10
indistinguishable. **Restoring the full-body targets does not affect face
scale.** The 8 August finding that our filter discards 96% of cosmic's
full-body imagery remains true and worth fixing, but it is not the cause of
this failure.

This confirms the 8 August root cause exactly: the model renders a face at the
scale its *training references* had. Fix the reference scale and the problem
disappears; change the target framing and it does not move.

---

## 1. Results at matched step 8,000

Metric: detected-face short side ÷ the fixed mask box short side. The mask is
the required face area and is identical across runs.

| Run | `id_sim` | p10 | p25 | median | **< 0.8** | < 0.9 | no face |
|---|---:|---:|---:|---:|---:|---:|---:|
| E13 | `.3880` | `0.937` | `0.975` | `1.027` | **0** | 3 | 0 |
| **CL9** | `.3862` | **`0.933`** | `0.960` | `1.025` | **1** | **5** | 0 |
| CL8 | `.3964` | `0.779` | `0.887` | `0.963` | **12** | 25 | 0 |
| CL4 | `.3939` | `0.791` | `0.883` | `0.983` | 11 | 26 | 0 |
| CL0 | `.3921` | `0.781` | `0.896` | `0.973` | 11 | 25 | 1 |

CL9's `<0.9` count (5) is also close to E13's (3), against 25-26 for the others,
so the improvement is the whole lower tail rather than a handful of outliers.

### By identity

| Run | eddie | elon | jennie | **jensen** | **jisoo** | **keanu** | lex | marion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E13 | 0 | 0 | 0 | **0** | **0** | **0** | 0 | 0 |
| **CL9** | 1 | 0 | 0 | **0** | **0** | **0** | 0 | 0 |
| CL8 | 0 | 0 | 0 | 2 | 7 | 3 | 0 | 0 |
| CL4 | 0 | 0 | 0 | 2 | 5 | 4 | 0 | 0 |
| CL0 | 0 | 0 | 0 | 2 | 6 | 3 | 0 | 0 |

CL9 eliminates the entire jensen/jisoo/keanu cluster — the three identities
whose validation reference face is below 8% of frame. Its single remaining case
is **eddie**, whose reference face is `50.35%`, the *largest* of the twelve.
That is a coherent and expected side effect: CL9 samples training reference
faces from `[0.06, 0.30]`, so it now under-serves the extreme high end that used
to be well covered. One image, and directly addressable (§4).

Visual comparison at step 8,000, red box = the identical required face area:

![CL9 fills the required area like E13; CL8 and CL4 render a small face inside it](assets/cl89_facesize_grid.png)

`Kickboxing_jisoo`: E13 `0.97`, CL9 `0.94`, CL8 `0.66`, CL4 `0.73`.
`Skiing_man_keanu`: E13 `0.93`, CL9 `0.93`, CL8 `0.78`, CL4 `0.75`.
CL8/CL4 additionally show mouth and eye artifacts inside the shrunken face;
CL9's faces are clean and correctly proportioned.

---

## 2. What worked, and what did not

### CL9 worked — and it fixed CL2's failure mode

CL2 established that scale control removes undersized faces (1 case) but
collapsed `id_sim` to `.28255`. CL9 keeps the scale control and adds two
changes, and the collapse is gone:

| | CL2 | **CL9** |
|---|---|---|
| Reference face scale | locked to the target's exact scale | sampled from `[0.06, 0.30]` |
| Reference face position | locked to the target's exact position | jittered up to `0.15` of canvas |
| Undersized faces | 1 | **1** |
| `id_sim` | `.28255` @24k | **`.38619`** @8k |

**The identity collapse was caused by the degenerate fixed scale-and-position,
not by the fabricated surround.** Both arms composite onto the same
edge-filled canvas, so the surround cannot explain a `.10` difference. With the
copy shortcut removed, the branch has to learn an actual scale-and-position
invariant mapping — which is what generalises to validation.

### CL8 did not work for face scale

12 undersized versus CL4's 11 — no effect. The mechanism is now clear:
target framing determines what compositions the model can *compose*, while
reference framing determines what scale it *renders the face at*. The failure
was always the latter.

CL8 is not worthless: it leads `id_sim` at 8k (`.39640`, the highest of any arm
at that step) with 2.6× the data, and its full-body distribution is genuinely
restored (58,227 records, 64.0% below 5% face area, verified in-job). It is a
data-scale win, not a face-scale fix.

**Caveat on CL8:** its scale-balanced sampler is **inert**. The DataLoader
shuffles (`train_dataloader_shuffle=None`, dataset does not declare
`requires_sequential_sampling`), so the round-robin reordering is destroyed
every epoch. CL8 is therefore an *unbalanced* full-body restoration. Correct
balancing requires oversampling by duplicating under-represented buckets, which
survives shuffling. This does not invalidate CL8 — the restoration itself is the
primary variable and it is correctly implemented — but the arm is not what its
config comment claims.

### `id_sim` remains the wrong primary metric

At 8k, `id_sim` ranks CL8 `.3964` > CL4 `.3939` > CL0 `.3921` > E13 `.3880` >
CL9 `.3862`. It puts the **best** arm last and three broken arms above the
reference. `IDSimBest` takes the best-matching face anywhere in the image and
never checks that it fits the body, so a small well-formed face scores well.
Report undersized count alongside it, always.

---

## 3. Confidence

| Claim | Confidence |
|---|---|
| CL9 fixes the small-face failure | **High** — 1 vs 11 undersized at a matched step; p10 matches E13; the jensen/jisoo/keanu cluster is at zero **[measured]** |
| CL8 does not affect face scale | **High** — 12 vs CL4's 11, p10 identical **[measured]** |
| Reference scale is the causal variable | **High** — CL9 changes only it and the effect is complete; CL8 changes target framing and nothing moves **[measured]** |
| CL2's collapse came from fixed scale+position, not the fill | **Medium-high** — CL9 shares the fill and does not collapse; not isolated by a fill ablation |
| CL9's eddie case comes from the `[0.06, 0.30]` upper bound | **Medium** — consistent with eddie's 50.35% reference, single image |
| CL8's inert balancing | **High** — read from `data_utils.py` **[code]** |

### Not established

- That CL9 holds to 24k. It is at 8k; CL2's `id_sim` degraded over training,
  though for a different reason.
- That CL9's `.0077` `id_sim` deficit versus CL4 is real rather than noise.
- Whether CL8's data-scale gain compounds with CL9's fix — untested, and the
  motivation for CL10.
- Whether the fabricated surround costs anything at all; no fill ablation ran.

---

## 4. Proposed experiments

Both build on **CL9**, the proven fix, and keep the E13 contract: 24k steps,
batch 2, one A100, fixed full-96, `2,240 / 219,217,920` parameters.

### CL10 — CL9 reference scaling on the restored full-body dataset (priority 1)

`CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k.yaml`

**Change:** CL9's reference scale/position jitter **plus** CL8's
`min_face_res: 64`, this time with **working** scale balancing.

```yaml
datasets:
  train:
    cosmic_large_adapted:
      # CL9 half - the proven face-scale fix
      reference_crop_margin: null
      reference_content_size: null
      reference_canvas_size: null
      reference_frame_mode: target_face_frame
      reference_frame_fill: edge
      reference_scale_jitter: [0.06, 0.40]   # widened; see below
      reference_position_jitter: 0.15
      # CL8 half - 2.6x the data, full-body framing restored
      min_face_res: 64
      target_scale_balance: true
      target_scale_bins: [2.0, 5.0, 10.0, 20.0]
      target_scale_balance_mode: oversample   # NEW - must survive shuffling
```

**Why:** the two arms are independent and both positive on their own axis —
CL9 on face scale, CL8 on data volume and `id_sim`. Nothing observed suggests
they interfere: CL8 leaves face scale untouched, so CL9's fix should carry
over intact, while CL9 does not touch target selection.

Two corrections folded in:

1. **Jitter widened to `[0.06, 0.40]`.** CL9's single failure is eddie at a
   `50.35%` reference; `0.30` under-serves the top of the inference range. `0.40`
   covers the realistic band without wasting capacity on the 50% extreme.
2. **Balancing must oversample, not reorder.** Add
   `target_scale_balance_mode: oversample`, duplicating entries from
   under-represented buckets so the index itself is balanced. `reorder` keeps
   the current (inert) behaviour for exact reproduction of CL8.

**Prediction:** undersized ≤ 2, `id_sim` ≥ CL8's `.39640` at matched steps —
i.e. E13-level face geometry with the best identity score of any arm.

**Risk:** oversampling small-face targets is the variant the 26 July note warned
could destabilise. Mitigation: bins already cap the duplication factor; if
training destabilises, fall back to `min_face_res: 128`.

### CL11 — CL9 plus multi-reference identity conditioning (priority 2)

`CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k.yaml`

**Change:** CL9 plus `num_identity_refs: 3` (the CL5/E18 mechanism).

```yaml
datasets:
  train:
    cosmic_large_adapted:
      # ... CL9 block unchanged ...
      num_identity_refs: 3
model:
  batched_conditioning_preparation: false   # the batched path asserts one ref
```

**Why:** CL9 is `.0077` below CL4 on `id_sim`. Every cosmic target carries 2-10
same-person crops, and a tight crop is the ideal input for the PhotoMaker ID
lane — the consumer that does not care about spatial scale. E18 gave `+.03735`
from this mechanism on large_dataset. This is the cheapest way to recover the
identity gap without touching the geometry that CL9 just fixed.

**Prediction:** undersized stays ≤ 2 (the spatial lane is unchanged — reference 0
remains the only spatial lane), `id_sim` recovers to ≥ CL4's `.39391`.

**CL10 and CL11 are independent and can run in parallel.** If both succeed, the
three-way combination is the obvious follow-up.

### Not proposed

- **More CL8-style target work.** It demonstrably does not move face scale.
- **A fill ablation** (`gray`/`symmetric`). CL9 shows the fill costs little; not
  worth an A100 while CL10/CL11 are untested.
- **Changing the validation protocol.** Would improve measurement but breaks
  comparability and fixes nothing in training.

---

## 5. Implementation plan

### Step 1 — dataset code (`src/datasets/cosmic_large_adapted.py`)

One new **defaults-off** argument:

```python
target_scale_balance_mode: str = "reorder",   # "reorder" | "oversample"
```

In the balancing block, keep the current round-robin for `reorder` (so CL8
reproduces exactly) and add `oversample`: compute the largest bucket size, then
extend each smaller bucket by cycling its own entries until all buckets are
equal, and concatenate. Record realised duplication factors in `self.audit`
(`scale_bucket_oversample_factors`) so the preflight can report them.

Validate: `oversample` requires `target_scale_balance: true`; cap the
duplication factor at, say, 4x and fail closed above it, so a pathological bin
cannot explode the index.

Nothing else changes — CL9's jitter parameters already exist and are proven.

### Step 2 — registry

Add `target_scale_balance_mode: reorder` to `cosmic_large_adapted` in
`src/configs/datasets/all_datasets.yaml`.

### Step 3 — configs

Both `defaults: [CL9_cosmic_joint_shadow_sa128_refscale_24k, _self_]` so the
proven CL9 block is inherited rather than restated. CL10 adds the CL8 target
fields plus the widened jitter; CL11 adds `num_identity_refs` and the batched
conditioning toggle.

### Step 4 — validator (`tools/validate_CL1_CL3_config.py`)

Add both to `ARMS` as `("cosmic_large_adapted", False, "target_face_frame")`.
Assert:

- CL10: `min_face_res < 192`, `target_scale_balance is True`,
  `target_scale_balance_mode == "oversample"`, jitter inside `[0.03, 0.60]`,
  positive position jitter. Extend the existing CL8 `min_face_res` exemption to
  cover CL10.
- CL11: jitter and position jitter as CL9, `num_identity_refs == 3`, and the
  existing rule that `num_identity_refs > 1` requires
  `batched_conditioning_preparation: false`.

### Step 5 — preflight (`tools/datasets/preflight_cosmic_cl.py`)

Add `CL10`/`CL11` to the CL9 branch (1024² canvases, reference face fraction
median inside the jitter range). For CL10 additionally require the CL8 gate —
≥20% of sampled targets below 5% face area — **and** report the realised bucket
counts after oversampling, so an inert balancer cannot pass unnoticed a second
time.

### Step 6 — launcher, records, packages

Add both names to the `case` gate in
`launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh`; write
`experiments/cosmic_large/CL{10,11}_..._r1.json`; clone Serv packages from
CL9's — isolated runtime under `runtime_sources_cl1_cl3_v1/<run>`, sealed
manifest, self-contained MLS YAML.

### Step 7 — pre-launch gates

1. Composition resolves to 24,000 steps and `2,240 / 219,217,920`.
2. Only the intended fields differ from CL9.
3. **Smoke test that `oversample` actually changes the sampled distribution** —
   the specific failure CL8 hit. Assert bucket counts are equalised in the index
   itself, not merely reordered.
4. Real-data preflight passes both the face-fraction and full-body gates.
5. Step-0 panel identical to CL9's (`id_sim .30187`).
6. Allow ~10 minutes for model construction before treating silence as a hang;
   step-0 generation then emits no log lines for ~25 minutes.

### Step 8 — decision gates

Primary metric is **undersized-face count at matched steps**.

| Gate | Step | Rule |
|---|---:|---|
| Data | preflight | CL10: ≥20% targets below 5% face area, buckets equalised, ~58k records |
| Face scale | 4k | undersized ≤ 2 (CL9 achieved 1; CL4/CL0/CL8 are 11-12) |
| Identity trio | 8k | jensen/jisoo/keanu at zero |
| Identity | 8k | CL10 `id_sim` ≥ CL8's `.39640`; CL11 ≥ CL4's `.39391` |
| Promotion | 24k | undersized ≤ 2 sustained with `id_sim` ≥ `.40` |

Also re-measure **CL9 at 24k** when it completes, to confirm the fix holds; that
is the single most informative datapoint pending.

---

## 6. Reproducing

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh && conda activate photomaker
cd /home/kolyangg/rsrch_apr_test/diffusion_template

# fetch a matched step for each arm into its OWN directory
python tools/comet/comet_experiment.py fetch --record comet_records/<run>.json \
  --step-number 8000 --output-dir comet_data/<batch>/<arm>

# measure face size against the required area
python tools/datasets/measure_face_body_alignment.py \
  --images-dir <canonical export dir> \
  --mask-boxes ../dataset_full/val_dataset/pm96_bboxes_new.json \
  --id-sim-csv <id_sim_step_008000.csv> --label CL9 --output align8k.json
```

Traps, all silent: bbox keys and `id_sim` `output_key` contain spaces while
exported PNGs use underscores (normalise both sides); the Comet API returns
figure names for image assets, so use `comet_experiment.py fetch`; give each
step its own `--output-dir`.

Data: `comet_data/cl89_20260808/` — per-image rows at 8k for all five arms.

## 7. References

- [Face-scale root cause, 8 Aug](2026-08-08_cl_face_scale_root_cause_and_cl8_cl9.md)
- [Alignment analysis, 7 Aug](2026-08-07_cl_face_body_alignment_analysis_and_cl6_cl7.md)
- [CL2/CL3 early results, 7 Aug](2026-08-07_cl2_cl3_early_results_and_next_cosmic_experiments.md)
