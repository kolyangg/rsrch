# Seed-specific PhotoMaker masks confirm CL39's spatial identity signal

**Date:** 27 August 2026  
**Evidence cutoff:** 16:46 BST, 27 August 2026  
**Scope:** sealed CL39 24k checkpoint; fixed 96-image `manual_val` panel;
full A/B/C/D identity-source crossing at inference seeds 0, 1, 2 and 3;
1,536 scored images. These are inference seeds on one trained checkpoint, not
independent training runs.  
**Primary metric:** subject-v2 intended-identity cosine and
intended-minus-next-identity margin, evaluated on seed 0's accepted box and on
each later seed's accepted PhotoMaker-only automatic face box.  
**Immutable parent Comet key:** `b1ca0b3da679401c85b991f1bbdf0b2a`.

> **Correction of record.** This version supersedes the earlier report at this
> same path. The earlier seeds 1–3 used a seed-0 automatic-box cache during BA
> generation as well as scoring. Their images and estimates are invalid as
> multiseed evidence and are retained only as failure evidence. The corrected
> runs regenerated seed-specific boxes from PhotoMaker-only images before
> generating A/B/C/D. `[record][decision]`

## Executive conclusion

Correct spatial BA reference adds a small but repeatable identity signal in
every tested inference seed; PhotoMaker tokens remain the dominant identity
source. The correction strengthens rather than weakens the causal result.
`[measured][decision]`

Across seeds 0–3, correct spatial reference raises intended identity by
`+0.03055` for A−B (PhotoMaker identity correct) and `+0.01422` for C−D
(PhotoMaker identity wrong). Two-way seed-and-cell bootstrap 95% intervals are
`[+0.02393,+0.03740]` and `[+0.00896,+0.01917]`. Identity-margin gains are
`+0.04326` and `+0.04015`, with intervals `[+0.03549,+0.05136]` and
`[+0.02990,+0.04991]`. Every per-seed interval is above zero for both spatial
contrasts. `[measured]`

PhotoMaker identity remains decisive. Pooled intended-ID effects are
`A−C=+0.53712` and `B−D=+0.52080`; pooled identity-margin effects are
`+0.96225` and `+0.95915`. The target identity wins in all `96/96` mean cells
for A−C and B−D on intended identity and margin. `[measured]`

![Per-seed and pooled corrected identity-source effects](assets/cl39_identity_crossing_multiseed_20260827/paired_effects_multiseed_dynamic.png){ width=98% }

*Figure 1. Red points are four inference-seed means. Blue diamonds and bars
are pooled means and two-way seed-and-cell bootstrap 95% intervals. Positive
A−B/C−D values favor correct spatial reference; positive A−C/B−D values favor
correct PhotoMaker identity.*

## 1. Completion and integrity

The three canonical one-A100 Serv jobs completed successfully with platform
error code zero:

| Seed | Immutable MLS job | Canonical output |
|---:|---|---:|
| 1 | `lm-mpi-job-ee43b350-de5c-44e3-9cab-d694e9f5806e` | 4 arms × 96 PNGs |
| 2 | `lm-mpi-job-9c599a15-1d97-49a9-8609-81f38d03ca85` | 4 arms × 96 PNGs |
| 3 | `lm-mpi-job-f04b6ebf-aded-4da2-ad5d-206a65534f15` | 4 arms × 96 PNGs |

The corresponding dynamic-box SHA-256 prefixes are `ea76ee575431`,
`c367dd4f8d8b`, and `468c95ff788f`; full values are in `summary.json` and the
audit comment in this Markdown source.

<!-- Dynamic-box SHA-256 seed1=ea76ee575431f51210a2754db9434ed6a8633fc6eb76bfe1e7cc1e2dd809c27b seed2=c367dd4f8d8b5646fb6a14c817c13bc7e23e198d416ee612606f7da67490c90a seed3=468c95ff788f5c4774357007a980364ef4e97e1e9228862fa403465dd8014999 -->

All 12 canonical arm gates record the same immutable parent Comet key, sealed
checkpoint and source manifest, plus the appropriate seed-specific box SHA.
All 1,152 new PNGs open at 1024×1024; the join contains exactly
`3 × 4 × 96` new rows, no duplicates and no dropped cells. The accepted seed-0
384 rows are byte-for-value identical to the preceding attribution analysis.
`[record]`

The sealed checkpoint SHA-256 is
`74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07`;
the corrected source-manifest SHA-256 is
`e1022d515296892ab6c46a36a51db37bcf9dead4798c30502660138b5b0d7643`.
The stale shared seed-0 box cache had SHA-256
`b33cf02665cd875a738fba8f20c2ea95fcb0585358436e32691af0013a2f1c7d`.
Model weights, prompts, reference mapping, scheduler, inference steps, CFG,
PM/spatial interventions, `pose_adapt_ratio=0`, and CA mixing off are unchanged.
`[record]`

An acceleration job,
`lm-mpi-job-d2a306cb-e84d-4eff-af6a-c374f59f12c5`, completed a redundant
seed-1 D arm but later failed its sealed source-manifest gate after detecting
eight generated debug PNGs. No gate was weakened and no accelerator output is
used here. The original seed-1/2/3 jobs completed every canonical arm, so the
failure does not reduce the accepted panel. `[record]`

## 2. Corrected factorial results

The four interventions are:

| | Correct spatial reference | Next-identity spatial reference |
|---|---|---|
| Correct PhotoMaker identity tokens | A | B |
| Next-identity PhotoMaker tokens | C | D |

### 2.1 Intended identity by inference seed

| Seed | A−B spatial, PM correct | 95% interval | C−D spatial, PM wrong | 95% interval |
|---:|---:|---:|---:|---:|
| 0 | `+0.02951` | `[+0.02225,+0.03670]` | `+0.01731` | `[+0.01220,+0.02243]` |
| 1 | `+0.03396` | `[+0.02621,+0.04208]` | `+0.01408` | `[+0.00896,+0.01916]` |
| 2 | `+0.03216` | `[+0.02438,+0.04016]` | `+0.01672` | `[+0.01114,+0.02243]` |
| 3 | `+0.02655` | `[+0.01907,+0.03406]` | `+0.00880` | `[+0.00289,+0.01461]` |
| **Pooled** | **`+0.03055`** | **`[+0.02393,+0.03740]`** | **`+0.01422`** | **`[+0.00896,+0.01917]`** |

Correct spatial reference also reduces next-identity attraction: pooled
wrong-identity changes are `−0.01272` for A−B and `−0.02593` for C−D, with
intervals `[-0.01721,-0.00823]` and `[-0.03405,-0.01846]`. Thus the intended
gain and wrong-identity suppression point in the same favorable direction.
`[measured]`

### 2.2 Identity margin by inference seed

| Seed | A−B margin | 95% interval | C−D margin | 95% interval |
|---:|---:|---:|---:|---:|
| 0 | `+0.03955` | `[+0.03069,+0.04843]` | `+0.04091` | `[+0.03275,+0.04915]` |
| 1 | `+0.04664` | `[+0.03736,+0.05630]` | `+0.04800` | `[+0.03795,+0.05805]` |
| 2 | `+0.04705` | `[+0.03723,+0.05693]` | `+0.04376` | `[+0.03458,+0.05313]` |
| 3 | `+0.03980` | `[+0.03097,+0.04867]` | `+0.02794` | `[+0.01815,+0.03788]` |
| **Pooled** | **`+0.04326`** | **`[+0.03549,+0.05136]`** | **`+0.04015`** | **`[+0.02990,+0.04991]`** |

The direction is stable and the corrected magnitude is no longer anomalously
small at seeds 1–3. Seed 3 is the weakest C−D case, but its interval remains
strictly positive. `[measured]`

### 2.3 PhotoMaker remains the dominant identity source

| Pooled contrast | Intended-ID Δ | Two-way 95% interval | Margin Δ | Two-way 95% interval |
|---|---:|---:|---:|---:|
| A−C, PM correct vs wrong with spatial correct | `+0.53712` | `[+0.51451,+0.55738]` | `+0.96225` | `[+0.91176,+1.01022]` |
| B−D, PM correct vs wrong with spatial wrong | `+0.52080` | `[+0.49835,+0.54048]` | `+0.95915` | `[+0.90660,+1.00796]` |

The intended-ID PM effects are about 18 times A−B and 37 times C−D; margin
effects are about 22 and 24 times the corresponding spatial effects. These
ratios belong to this adversarial next-identity crossing and are not a general
variance decomposition. `[measured][limitation]`

## 3. Why the earlier result was invalid

The earlier launcher silently reused 96 seed-0 records for seeds 1–3. Because
that box drives BA generated-face masking, the defect changed generation; it
was not merely a scoring crop problem. The visual failure can place the red
mask on grass, a bench or another background region. I was wrong to interpret
those images as evidence that the spatial effect attenuated at later seeds.
`[code][record][decision]`

![Corrected versus invalid pooled estimates](assets/cl39_identity_crossing_multiseed_20260827/corrected_vs_flawed_effects.png){ width=98% }

*Figure 2. The gray estimates are not alternative valid measurements: their
generation masks were wrong. They are shown only to quantify the impact of
the defect. Corrected-minus-invalid pooled means are +0.01467/+0.00576 for
intended-ID A−B/C−D and +0.02281/+0.01754 for margin.*

\newpage

![Largest correction example](assets/cl39_identity_crossing_multiseed_20260827/representative_largest_correction.png){ width=92% }

*Figure 3. `Reading pa_jensen.png`, selected by the largest mean absolute
corrected-versus-invalid margin change. Invalid masks visibly target
background; corrected masks track the seed-specific face. Images also differ
because the box participates in generation.*

## 4. Face ownership, alignment and visuals

Before A/B/C/D generation, each new seed produced exactly 96 PhotoMaker-only
images, 96 automatic boxes and 96 overlays. All three pre-generation gates
passed with no face misses or unowned boxes:

| Seed | PM-only boxes/images | No face | Unowned | Mean best IoU | Gate |
|---:|---:|---:|---:|---:|---:|
| 1 | 96/96 | 0 | 0 | `0.9311` | pass |
| 2 | 96/96 | 0 | 0 | `0.9232` | pass |
| 3 | 96/96 | 0 | 0 | `0.9206` | pass |

The same accepted box is reused across A/B/C/D within a seed, preserving a
paired spatial intervention. On the final generated images, every image has a
detected face and none is ambiguous. Four images are unowned: the C and D
versions of `Crying man_eddie.png` at seed 2 and `Angry man _jensen.png` at
seed 3. This is `4/1,536=0.26%` of the full analysis, or two unique cells where
wrong-PM generation moved the face away from the PhotoMaker-derived box.
`[measured][limitation]`

Restricting each contrast to cells owned in both arms changes no conclusion.
For the affected C−D comparisons, corrected intended-ID/margin effects are
`+0.01690/+0.04422` at seed 2 and `+0.00889/+0.02823` at seed 3. This
post-generation sensitivity is supporting evidence, not the primary estimate.
`[measured][limitation]`

\newpage

![Robust corrected example](assets/cl39_identity_crossing_multiseed_20260827/representative_robust_corrected.png){ width=84% }

*Figure 4. `Chef man i_jensen.png`, selected by the largest minimum spatial
margin gain across A−B and C−D over all seeds. The face box follows each
seed-specific composition; the PM switch remains visually dominant.*

\newpage

![Ownership stress example](assets/cl39_identity_crossing_multiseed_20260827/representative_ownership_stress_corrected.png){ width=84% }

*Figure 5. `Angry man _jensen.png`, selected by the declared lowest corrected
ownership/alignment rule. Seed-3 C/D move the wrong-PM face away from the
PhotoMaker-only box, producing the two zero scores shown. The other seeds and
A/B are correctly owned.*

The figures were selected by predeclared score rules and inspected at full
resolution. They establish face ownership/alignment and make the identity
switch visible, but do not quantify prompt adherence or overall image quality.
`[visual][limitation]`

## 5. Uncertainty, limitations and confidence

Per-seed intervals use 100,000 fixed-cell bootstrap draws over 96 paired cells
with published seeds `390100–390103`. The primary pooled interval independently
resamples four inference seeds and 96 cells with seed `390427`; W/T/L counts
operate on each cell's four-seed mean. A fixed-cell bootstrap of those cell
means with seed `390426` is retained in the audit table. `[record]`

| Claim | Confidence | Evidence and boundary |
|---|---|---|
| Correct spatial BA carries identity information | **High** | Positive A−B and C−D intended-ID and margin intervals in every seed; matched visual grids |
| Pooled spatial magnitude for this checkpoint/panel | **Moderate–high** | Four inference seeds and 96 fixed cells; no training-seed replication |
| PhotoMaker dominates identity in this crossing | **High** | Large PM effects with 96/96 pooled cell wins |
| General prompt adherence or aesthetic quality | **Not established** | This report did not score those outcomes |
| Generalization to other checkpoints/training seeds | **Not established** | One sealed checkpoint only |

Boxes are derived from matched-seed PhotoMaker-only generations, then held
fixed across A/B/C/D. This is the requested causal contract, but it does not
guarantee ownership when wrong-PM interventions move the face. Subject-v2
cosine is an identity proxy, not a human preference score. `[limitation]`

## 6. Decision gates and next experiments

| Gate | Criterion | Result | Decision |
|---|---|---|---|
| Dynamic-box integrity | Three distinct seed-specific SHAs; 96 PM images, boxes and overlays per seed; all pre-gates pass | **Pass** | Accept corrected seeds 1–3 |
| Spatial replication | A−B and C−D margin positive in all seeds; every interval above zero | **Pass** | Retain explicit spatial BA as causally useful |
| PM dominance | PM effects much larger and 96/96 mean-cell wins | **Pass** | Preserve PM conditioning/native anchor |
| Generated-image ownership | No face=0, ambiguous=0; unowned ≤2 per seed | **Pass with limitation** | Keep owned-pair sensitivity in multiseed reports |
| Earlier static-box evidence | Later-seed generation used seed-0 masks | **Fail** | Never use those estimates for scientific conclusions |

Two bounded follow-ups are justified:

1. **CL39 seeds 4–7 dynamic-box replication.** Single change: inference seeds.
   Hypothesis: both spatial margin contrasts remain positive. Accept if all
   seed gates pass, pooled two-way intervals exclude zero, and no seed reverses
   both A−B and C−D; risk is additional composition-dependent ownership.
2. **E2 checkpoint crossing under this exact mask contract.** Single change:
   checkpoint/config under test. Hypothesis: target-PM dropout reduces PM
   dominance while retaining or increasing spatial margin. Accept only with
   the same fixed-96 cells, per-seed PM-only boxes, immutable checkpoint hash,
   positive spatial intervals, and no material face-quality regression.

These are proposals, not evidence that either outcome will occur.
`[hypothesis][decision]`

## 7. Reproduction and audit files

From `diffusion_template/`:

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker

dynamic_root=artifacts/cl39_identity_crossing_dynamic_masks_20260827
flawed_root=artifacts/cl39_identity_crossing_multiseed_20260827

PYTHONPATH=. python \
  tools/analysis/analyze_cl39_identity_crossing_dynamic_masks.py \
  --seed0-csv artifacts/cl39_attribution_controls_20260827/scored/per_image.csv \
  --seed0-task-root artifacts/cl39_attribution_controls_20260827/serv_task \
  --new-task-root "$dynamic_root/serv_task_complete" \
  --flawed-csv "$flawed_root/scored/per_image.csv" \
  --flawed-task-root "$flawed_root/serv_task" \
  --reference-root ../dataset_full/val_dataset/references \
  --subject-v2-embeds ../dataset_full/val_dataset/id_embeds_manual_val_subject_v2.pth \
  --output-root "$dynamic_root/scored" \
  --device cpu
```

The join normalizes spaces to underscores on metadata and PNG keys and fails
unless every seed/arm has exactly 96 unique cells. Recompute exact audit hashes
with:

```bash
sha256sum "$dynamic_root/scored"/{summary.json,per_image.csv,paired_effects.csv}
sha256sum "$dynamic_root/scored"/{aggregate_by_seed.csv,owned_pair_sensitivity.csv}
sha256sum "$dynamic_root/scored/corrected_vs_flawed_effects.csv"
sha256sum tools/analysis/analyze_cl39_identity_crossing_dynamic_masks.py
```

Recorded SHA-256 prefixes in that order are `dd950498d44e`, `363203dc419c`,
`1d663f33cc97`, `f02565d4c4c6`, `631c5210f0bb`, `a6bcc2d80d98`, and
`e640fb42e768`. Full values are in the audit comment below.

<!-- Audit SHA-256 summary.json=dd950498d44edc0331a9eeec177b48b9740ed9b9d656b5d0578e228e6c0444ed per_image.csv=363203dc419c863f20442868a5b09c056de028d124ae53db02231517dc78f71e paired_effects.csv=1d663f33cc9725cc615eebae88f1dd35ea5c55f3950a48480b47520bd6324713 aggregate_by_seed.csv=f02565d4c4c6d618b3a863f69897a5cbc64841c57569069a08f621e87297df31 owned_pair_sensitivity.csv=631c5210f0bbb3482b3c44bf266c96f6443b36984de045bda816747d9825b59a corrected_vs_flawed_effects.csv=a6bcc2d80d9815f3073d4d720fe654a193f8a3404fe7546448d1f35210362625 scorer=e640fb42e768de439569977843dcec58c6cfabce82b8a8199a518ca24a0f96a5 -->

Audit roots:

- package: `serv_run_packages/CL39_identity_crossing_dynamic_masks_20260827_r1/`;
- outputs: `$dynamic_root/serv_task_complete/`.

## 8. References

1. `analysis/2026-08-27_cl39_spatial_ba_attribution_controls.md` — accepted
   seed-0 causal crossing and subject-v2 scoring contract.
2. `analysis/blueprints/26Aug/CL39_branched_attention_investigation_2026-08-26.md`
   — preregistered investigation sequence.
3. `docs/handoffs/LATEST.md` — current immutable run state and architecture
   history.
