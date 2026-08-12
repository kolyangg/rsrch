# CL2/CL3 early results on cosmic_large, and what to run next

**Date:** 7 August 2026
**Scope:** analysis of the first 4,000 steps of CL2 and CL3, plus implementation
of CL0, CL4 and CL5. No running job was touched.
**Evidence cutoff:** step 4,000 of a 24,000-step contract. Both runs are live.
Every number below is from the immutable Comet keys, not display names.

| Run | Comet key | State |
|---|---|---|
| CL2 `facecanon` | `be7b7a2acf174b69b5e361490926140e` | Running, past 4,950 |
| CL3 `fmtfix` | `488ec4fdee5b4560a77f3924af3e0b6e` | Running, past 4,450 |
| E13 (large_dataset reference) | `1cc0a02371094b24a6a02a4cc649f10c` | complete, 24k |

Local export: `comet_data/cl2_cl3_early_20260807/` — full metric histories,
per-image `id_sim` tables at 0/2k/4k, all 96 images per step per run, and the
comparison sheets.

---

## Executive conclusion

**CL3 is the promising arm and CL2 is actively regressing — the opposite of what
I predicted.** I argued in the 6 August analysis that CL2 should win because it
fixes reference geometry *before* the frozen encoder while CL3 only warps
already-computed features. The data contradicts that, and the reason appears to
be the risk I flagged but under-weighted: CL2 fabricates ~78% of its reference
canvas, and that costs more than the geometry gain is worth.

The more important finding is what the per-image data shows about *where* CL3's
remaining deficit lives:

```text
CL3 at 4k versus E13 at 4k
    mean          .34294  vs  .36454      -.0216   (looks like a clear loss)
    median        .3755   vs  .3901       -.0146
    images > 0.5  21      vs  14          +7       (CL3 has MORE strong images)
    paired wins   51/96                             (CL3 wins the majority!)
    catastrophic  6       vs  0           +6       (id_sim < 0.05)
```

CL3's *typical* image is already competitive with E13-on-large_dataset. Its
entire aggregate deficit is a **six-image catastrophic tail**. That is a far more
tractable target than "make cosmic_large work", and it redirects the next
experiments away from geometry and toward tail stabilisation.

---

## 1. Aggregate results

`manual_val/id_sim`, fixed 96-image panel, identical seeds/prompts/references:

| Step | CL2 `facecanon` | CL3 `fmtfix` | E13 (large_dataset) |
|---:|---:|---:|---:|
| 0 | `0.30187` | `0.32788` | `0.30212` |
| 2,000 | `0.24273` | `0.32087` | `0.29681` |
| 4,000 | `0.27219` | **`0.34294`** | `0.36454` |

`manual_val/text_sim`: all three sit in `26.3 → 27.0–27.4`; no arm trades text
for identity. Throughput is `0.457 steps/s` (CL2) and `0.438` (CL3), i.e.
`2.19` and `2.28` s/step — consistent with the historical `2.06–2.10` s/step
Cosmic figure, so neither arm is paying a hidden data cost.

Three observations:

1. **CL2 falls below its own initialization and stays there.** `.30187` →
   `.24273` → `.27219`. Training is actively destroying identity.
2. **CL3 rises above its initialization**: `.32788` → `.32087` → `.34294`,
   `+.0151` over step zero and still climbing at the evidence cutoff.
3. **CL3's step-zero is `.32788`, exactly E3's step-zero on large_dataset.**
   That is a strong internal consistency check: step-zero validation uses the
   fixed `manual_val` references, so the ROI warp must produce the identical
   untrained route regardless of which dataset the run trains on. It does, to
   five decimals, despite E3 being rank-32 and CL3 rank-128 (LoRA-B is zero at
   step zero, so rank cannot matter). The warp is installed and behaving.

### A contract-gate deviation worth recording

I specified that CL2's step-zero must be **byte-identical to E13 r4**. It is
not: `.30187` versus `.30212`. This is not a CL2 defect — `.30187` is the
step-zero shared by E0-fixed, E5, E6 and E11, and CL2/CL3 were built from the
E19-E24 snapshot base (`d903b2c`) whereas E13 r4 ran from `ebf1ac8`. The
codebase moved between them. **The gate was mis-specified**: the correct
comparator for CL2 is an E19-E24 arm on the same snapshot, not E13 r4. The gate
should be restated that way before it is used to stop a future run.

---

## 2. The per-image structure is the real result

Distribution of per-image `id_sim` across the 96-image panel:

| Run | Step | Mean | Median | `<0.05` | `<0.15` | `>0.5` |
|---|---:|---:|---:|---:|---:|---:|
| CL2 | 0 | `.3019` | `.3203` | 7 | 24 | 13 |
| CL2 | 2,000 | `.2427` | `.2215` | 17 | 35 | 10 |
| CL2 | 4,000 | `.2722` | `.2532` | **11** | 35 | 17 |
| CL3 | 0 | `.3279` | `.3436` | 5 | 20 | 13 |
| CL3 | 2,000 | `.3209` | `.3624` | 6 | 18 | 9 |
| CL3 | 4,000 | `.3429` | `.3755` | **6** | 16 | **21** |
| E13 | 4,000 | `.3645` | `.3901` | **0** | 8 | 14 |

**E13 on large_dataset has zero catastrophic images at 4k. Both Cosmic arms have
several.** That single line is the clearest statement of what cosmic_large costs
us today, and it is a stability problem rather than a general quality problem —
CL3 actually produces *more* strong images (`>0.5`) than E13.

### The two arms fail on disjoint sets

| Arm | Catastrophic images at 4k (`id_sim < 0.05`) |
|---|---|
| CL2 | 11 — concentrated on **jennie** (4), **eddie** (3), marion |
| CL3 | 6 — concentrated on **jisoo** (4), plus jensen and keanu on **kickboxing** |
| **Overlap** | **0** |

Per-identity means at 4k:

| Identity | CL2 | CL3 | E13 |
|---|---:|---:|---:|
| jennie | `.130` | **`.488`** | `.355` |
| elon | `.264` | **`.486`** | `.400` |
| marion | `.119` | `.314` | `.283` |
| lex | `.218` | `.385` | `.370` |
| keanu | `.443` | `.364` | `.425` |
| jensen | **`.494`** | `.315` | `.495` |
| jisoo | **`.420`** | `.229` | `.428` |
| eddie | `.090` | `.162` | `.161` |

CL3 beats E13 outright on jennie, elon, marion and lex. It loses badly on jisoo
and jensen — and **CL2 handles exactly those two well**. Zero failure overlap
between two arms that share every control except the geometry mechanism is a
strong signal that these failures are a property of the reference-geometry
pathway, not of Cosmic's content.

Visual confirmation (`comet_data/cl2_cl3_early_20260807/compare_4k_CL2_vs_CL3.png`):

- *jensen, kickboxing*: CL2 `.564` renders a coherent face; CL3 `.000` renders a
  melted, mis-aged face.
- *elon, night-ride*: CL2 `.130` renders a tongue/mouth artifact; CL3 `.399`
  renders a clean, recognisable subject.

Jisoo dominating CL3's failures is notable: the **Jisoo cluster is the recurring
Cosmic pathology** documented since 26 July across four unrelated arms. It
survives the geometry fix, so it is not caused by reference scale.

---

## 3. What this tells us

**Established by this evidence:**

- Correcting reference/target face-scale registration helps on cosmic_large:
  CL3 is `+.07074` mean over CL2 and wins 62/96 paired images.
- Correcting it **in feature space, leaving the reference image natural**, beats
  correcting it **in pixel space at the cost of a fabricated surround**. This is
  the CL2-vs-CL3 contrast and it is the session's main new fact.
- The residual gap to large_dataset is a small catastrophic tail, not a broad
  quality deficit. CL3's median and its count of strong images already match or
  beat E13.
- Neither arm trades text adherence for identity.

**Not established:**

- That CL3 beats CL2 *because* of the fabricated surround specifically. CL2 and
  CL3 differ in two ways at once — where the correction is applied, and whether
  pixels are fabricated. Separating them needs an arm that does pre-encoder
  framing without fabrication, which the 256px assets cannot supply.
- That CL3's 4k trajectory continues. Historical Cosmic arms peaked at 3k and
  regressed; E13-family arms kept climbing to 24k. CL3 could do either.
- Any part of the CL2/CL3 delta attributable to the shared hygiene changes
  (mirroring off, capped captions) rather than the geometry mechanism — **CL4
  exists to answer this**.
- That the catastrophic tail is caused by the reference and not by the target
  or caption for those specific identities.

---

## 4. Is there potential to improve cosmic_large further?

Yes, and the target is now specific. Three observations set the direction:

1. Fixing the tail is worth roughly `+.02` mean on its own. Six images at
   `~.02` lifted to CL3's median `~.375` would move the mean by
   `6 × .355 / 96 ≈ +.022` — enough to close essentially the whole gap to E13's
   `.36454` without improving a single typical image.
2. Cosmic's genuine strength is unused. Every accepted target carries **2–10
   same-person 256px crops** (mean 8.16). A tight crop is the *ideal* input for
   the PhotoMaker ID lane, which is exactly the consumer that does not care
   about spatial geometry. We currently use one crop and discard the rest.
3. The failures are identity-specific, which is what more independent identity
   evidence is best placed to fix.

---

## 5. Next experiments

All five arms share the exact E13 contract: 24k steps, batch 2, one A100, fixed
full-96 at step 0 and every 2k, shadow pretrained-default validation,
`pose_adapt_ratio=0`, `ca_mixing_for_face=false`, and
**2,240 tensors / 219,217,920 parameters** — none of the new arms adds a
parameter.

### The ladder

```text
CL0  as-is           E13 on cosmic_large, pre-CL loader          <- baseline
 |
CL4  hygiene         + no reference mirroring, capped captions   <- control
 |\
 | CL2 facecanon     + pre-encoder target-face-frame compositing  (running)
 |
CL3  fmtfix          + feature-space ROI warp                     (running)
 |
CL5  roiwarp+multiref  + 3 PhotoMaker identity references        <- new
```

### CL0 — `CL0_cosmic_joint_shadow_sa128_asis_24k` (priority 1)

The baseline we never ran. E13 exactly, on cosmic_large with the loader as it
behaved before any CL work: `reference_crop_margin=0.2`, `content_size=256`,
**reference mirroring on**, **uncapped legacy captions**, native frame, no warp.
Values are written out explicitly in the config so the baseline is
self-documenting and cannot drift with a future default change.

Without CL0 we cannot say whether CL3's `.34294` is an improvement on
cosmic_large at all — only that it beats CL2. **This is the highest-priority
run** despite being the least interesting, because every other number depends
on it.

### CL4 — `CL4_cosmic_joint_shadow_sa128_hygiene_24k` (priority 2)

CL2 and CL3 each bundle three deltas versus CL0. CL4 applies only the two cheap
hygiene ones — mirroring off, pose-first captions capped at 50 words — keeping
the native frame and no warp. This makes the ladder decompose:

```text
CL0 -> CL4   cost of reference mirroring + caption truncation
CL4 -> CL3   value of feature-space ROI warping
CL4 -> CL2   value of pre-encoder framing (with fabricated surround)
```

Without CL4, a CL3 gain cannot be separated from simply having fixed captions —
and the caption fix is large: at the inherited 55-word cap, 16.5% of Cosmic
captions still exceeded 77 CLIP tokens, and pose-first ordering alone leaves
87.2% over (legacy 86.4%). The cap, not the ordering, does the work.

### CL5 — `CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k` (priority 3)

CL3 plus `num_identity_refs=3`: two additional same-target 256px crops feed the
PhotoMaker ID encoder, while **`ref_images[0]` remains the sole spatial
latent/KV lane** — the E18/E19 invariant. E18 gave `+.03735` from this mechanism
on large_dataset while carrying a damaging persisted path; here it sits on the
better route.

This is the only arm that targets the actual deficit. It plays to cosmic's
strength (many same-person crops) on the lane where crops are ideal, having
already repaired the lane where they are not.

**Implementation cost:** one defaults-off dataset argument plus one config
toggle. `_prepare_branched_training_inputs_batched` asserts exactly one
reference per sample, so CL5 sets `model.batched_conditioning_preparation:
false` — exactly as E19 does for `large_dataset_balanced_multiref`. That changes
execution grouping only, not the maths.

### Deliberately not proposed

- **A CL2 fill ablation** (`symmetric`/`gray`). The knob exists, but CL2's
  trajectory is bad enough that tuning its surround is not the best use of an
  A100 while CL0/CL4 are missing.
- **More CL1 work.** The identity grouping fails its gate decisively — 1,876
  targets against a 3,000 floor, and ~half of those pairs are the same
  photograph twice. Recorded separately; not revived by this result.
- **Longer horizons before CL0 lands.** Extending CL3 past 24k is premature
  while the baseline is unknown.

---

## 6. Implementation status

All code is written and verified. **No non-dataset behaviour changed**: the one
new model-level setting used by CL5 (`batched_conditioning_preparation: false`)
is an existing config field that E19 already uses, and every new dataset
argument defaults to the historical value.

| File | Status |
|---|---|
| `src/datasets/cosmic_large_adapted.py` | modified — added defaults-off `num_identity_refs: int = 1` |
| `src/configs/datasets/all_datasets.yaml` | modified — registry default `num_identity_refs: 1` |
| `src/configs/CL0_cosmic_joint_shadow_sa128_asis_24k.yaml` | new |
| `src/configs/CL4_cosmic_joint_shadow_sa128_hygiene_24k.yaml` | new |
| `src/configs/CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k.yaml` | new |
| `tools/validate_CL1_CL3_config.py` | extended — CL0 baseline exemption, multi-ref/batched guard |
| `launchers/active/run_CL1_CL3_cosmic_24k_1gpu.sh` | extended allowed-config gate |
| `experiments/cosmic_large/CL{0,4,5}_*_r1.json` | new immutable records |
| `serv_run_packages/CL{0,4,5}_*_r1/` | new one-A100 MLS YAML + start script |

Verified locally in the `photomaker` env:

- all five CL arms compose to 24,000 steps and `2,240 / 219,217,920`, with the
  intended per-arm deltas and nothing else;
- the validator passes all five and enforces the new rule that
  `num_identity_refs>1` requires `batched_conditioning_preparation=false`;
- **E13, E19, E3 and `cosmic_large_adapted_rhca` compose unchanged**, with
  `num_identity_refs` resolving to the historical `1`;
- a real-fixture smoke test confirms `num_identity_refs=1` yields one reference
  (historical behaviour byte-for-byte), `=3` yields three distinct 256px crops
  with the spatial-lane bbox untouched, and the `[1,4]` range guard fails closed;
- `bash -n` and `py_compile` pass on all new and modified files.

**Not done:** the three runtimes are not staged on Serv and nothing is
submitted. Each needs the same treatment CL2/CL3 got — isolated runtime under
`runtime_sources_cl1_cl3_v1/<run>`, sealed source manifest, then validator and
preflight against the real manifest before submission. Project usage is
currently 8/8 A100 (E19-E24 plus CL2/CL3), so there is no room until several
finish.

### Suggested order when capacity frees

1. **CL0** — everything else is uninterpretable without it.
2. **CL4** — makes CL2/CL3 decomposable.
3. **CL5** — the actual improvement attempt.

If only one slot opens, run CL0.

---

## 7. References

- [Root cause and CL1-CL3 plan, 6 Aug](2026-08-06_cosmic_large_vs_large_dataset_root_cause_and_cl1_cl3_plan.md)
- [E13-E18 results, 6 Aug](2026-08-06_e13_e18_results_and_next_experiments.md)
- [E0-E12 analysis incl. E3 ROI warp](../comet_data/aug-large-ds_E0-E12_20260805/ANALYSIS.md)
- Local export: `comet_data/cl2_cl3_early_20260807/`
- Identity-grouping artifacts: `artifacts/cosmic_identity/`
