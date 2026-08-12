# CL9 update: small-face ROI replicates; Marion roll does not; precise occluder routing remains insufficient

**Date:** 11 August 2026  
**Scope:** Fixed-checkpoint validation at CL9 step `24,000`. No model weights,
training data, standard validation inputs, Comet experiment, or inference
default changed. This revision incorporates the complete `r4` chain and
supersedes this report's earlier single-seed conclusions.  
**Evidence cutoff:** Serv completed at `2026-08-11 09:14:44 Europe/London`;
local audit and visual review completed on 11 August 2026.  
**Primary metric:** Paired target-row change in mask-owned subject-v2 `id_sim`.
TOPIQ-Face, mask IoU, face size, RGB sentinels, and complete visual grids are
guards. ROI and occluder promotion summaries exclude Eddie because the exact
historical replay predates Eddie's subject-v2 generation repair.

| Arm | Immutable source Comet key | Step | Evidence | Headline |
|---|---|---:|---:|---|
| Historical replay | `81bb311ed70545eda3281c64bc48be47` | `24,000` | `96` | **`96/96` RGB-exact** |
| ROI `14` | same | `24,000` | `56` pairs | ID `+0.051`, `34/56` wins; fail |
| ROI `16` | same | `24,000` | `56` pairs | ID `+0.096`, `41/56`; fail by one win |
| ROI `18` | same | `24,000` | `56` pairs | **ID `+0.097`, `43/56` wins; pass** |
| ROI `20` | same | `24,000` | `56` pairs | **ID `+0.085`, `45/56` wins; pass** |
| Precise occluder | same | `24,000` | `14` pairs | ID `+0.038`, `9/14` wins; fail |
| Marion roll, seeds `0-3` | same | `24,000` | `48` pairs | **ID `+0.001`, `19/48` wins; fail** |

All arms reused the source Comet identity and deliberately created no new
Comet experiment. Serv job
`lm-mpi-job-09cb6478-b936-4ecb-b69b-a082742641c2` ran the full chain on one
A100 from `02:15:27` to `09:14:44` UK time and exited at `final_hashes` with
code `0`.

The source is `weights-epoch12.pth`, SHA-256
`5396993b16ace89908501bfddb2e412e755a3f6478a6449c502062d6ca7357c3`.
Every arm retained RealVis, `legacy_full_copy`, batch `12`, CFG `5`, DDIM `50`,
the active bbox map, `pose_adapt_ratio=0`, and
`ca_mixing_for_face=false`.

## Executive conclusion

The small-face intervention is the only mechanism that survives replication.

1. **Promote the `18`-step ROI suffix as an opt-in intervention and unlock its
   training analogue.** `[measured]` Non-Eddie identity improves **`+0.097`**
   across four seeds, with `43/56` wins and a target-row clustered bootstrap
   interval of `[+0.044,+0.147]`. Every seed is positive (`+0.094` to
   `+0.100`), every composite is pixel-exact outside its ROI, and quality/
   alignment guards pass. The `20`-step arm also passes, but has lower mean ID
   and slightly more face-size and boundary movement.

2. **Withdraw the earlier recommendation to retain Marion roll.** `[measured]`
   The seed-0 result (`+0.023`, `9/12` wins) does not replicate: seeds `1-3`
   change ID by `-0.012`, `+0.006`, and `-0.012`. Across `48` pairs the gain
   is **`+0.001`**, only `19/48` win, and the interval is
   `[-0.008,+0.012]`. Roll raises finite-pair TOPIQ by `+0.018`, but this is
   an image-quality effect, not a reliable identity fix.

3. **Precise occluder geometry is better than the rejected static mask, but
   does not unlock a learned gate.** `[measured]` ID rises `+0.038`, TOPIQ
   `+0.027`, and mask IoU changes only `-0.003`. Crying reaches `5/7` wins;
   Skiing reaches only `4/7`. `[measured: visual]` Several Skiing rows still
   duplicate or relocate goggles, and the largest wins expose or shrink faces.

The earlier ROI conclusion is confirmed and strengthened. The earlier Marion
roll conclusion was seed-specific and is retracted. The static occluder
rejection remains correct; sharper masks help, but ownership routing alone
does not solve occlusion.

## 1. Contract and evidence integrity

| Gate | Arms | Checked per arm | Exact per arm | Mismatches |
|---|---:|---:|---:|---:|
| Historical replay | `1` | `96` | **`96`** | `0` |
| ROI untouched rows | `16` | `80` | **`80`** | `0` |
| ROI outside declared ROI | `16` | `16` | **`16`** | `0` |
| ROI `18` seed `0` versus `r3` | `1` | `96` | **`96`** | `0` |
| Precise-occluder untouched rows | `1` | `80` | **`80`** | `0` |
| New Marion untouched rows | `6` | `84` | **`84`** | `0` |

`[measured]` All `26` stages wrote completion seals. The remote root contains
`COMPLETE`, `LAST_EXIT_CODE=0`, and a `2,974`-entry SHA-256 manifest. The
report retrieved `1,110` selected/metadata artifacts and matched all `1,110`
against that seal. These checks rule out a checkpoint, scheduler, batch-order,
reference-box, or general evaluator mismatch.

- `[code]` ROI seeds `1-3` change only local refinement noise on `16` target
  rows; the other `80` remain exact seed-0 historical pixels. This is a
  diagnostic exception, not a new standard validation protocol.
- `[report]` All joins use numeric `dataset_index`, avoiding the known
  space/underscore key trap.
- `[report]` Marion seed `0` reuses the completed `r3` pair; seeds `1-3` are
  newly generated original-versus-roll pairs, paired within prompt and seed.
- `[measured]` TOPIQ produced `47/48` finite Marion pairs. For Skiing seed `3`,
  the detector found one face in both images but TOPIQ rejected both crops.
  Subject-v2 ID remains finite for all `48` pairs.
- `[report]` Prompt adherence has no numeric sidecar here. Full-image grids are
  mandatory because TOPIQ/ID can improve when an occluder is weakened.

## 2. Small faces: the `18`-step gain is stable

The intervention crops a square `2x` region around the fixed face box, uses
`1.5x` bbox expansion, restarts from a suffix of DDIM50, and feather-composites
with fraction `0.12`.

| Suffix | n | Base ID | Arm ID | ID delta | Clustered 95% interval | Wins | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| `14` | `56` | `0.350` | `0.401` | `+0.051` | `[-0.004,+0.106]` | `34/56` | Fail |
| `16` | `56` | `0.350` | `0.447` | `+0.096` | `[+0.039,+0.153]` | `41/56` | Fail |
| `18` | `56` | `0.350` | **`0.447`** | **`+0.097`** | **`[+0.044,+0.147]`** | **`43/56`** | **Pass** |
| `20` | `56` | `0.350` | `0.435` | `+0.085` | `[+0.030,+0.137]` | **`45/56`** | **Pass** |

| Suffix | TOPIQ delta | Mask-IoU delta | Face-side delta | Outside-ROI exact |
|---:|---:|---:|---:|---:|
| `14` | `+0.015` | `-0.002` | `-0.2 px` | `64/64` |
| `16` | **`+0.022`** | `-0.003` | `-0.3 px` | `64/64` |
| `18` | `+0.008` | `-0.000` | `-1.1 px` | `64/64` |
| `20` | `+0.008` | `+0.002` | `-1.8 px` | `64/64` |

![Four-seed ROI response and guards. Error bars are target-row clustered bootstrap intervals.](assets/2026-08-11_cl9v_r4_roi_multiseed_summary.png){ width=100% }

| Suffix | Seed `0` | Seed `1` | Seed `2` | Seed `3` |
|---:|---:|---:|---:|---:|
| `14` delta / wins | `+0.049`, `8/14` | `+0.063`, `8/14` | `+0.026`, `8/14` | `+0.065`, `10/14` |
| `16` delta / wins | `+0.103`, `11/14` | `+0.091`, `10/14` | `+0.080`, `10/14` | `+0.112`, `10/14` |
| `18` delta / wins | **`+0.097`, `11/14`** | **`+0.097`, `11/14`** | **`+0.094`, `11/14`** | **`+0.100`, `10/14`** |
| `20` delta / wins | `+0.084`, `11/14` | `+0.088`, `12/14` | `+0.080`, `11/14` | `+0.089`, `11/14` |

`[measured]` The `18`-step mean is essentially invariant across seeds. Jumping
gains `+0.118` (`22/28` wins) and Dancing `+0.075` (`21/28`). Elon, Jennie,
Jisoo, and Marion win all `8/8` prompt-seed pairs, while Jensen loses all
`8/8` with mean `-0.074`. Jisoo and Marion gain identity but lose mean TOPIQ
(`-0.031`, `-0.022`), so the pass needs a fallback policy.

\clearpage

![All Dancing rows at seed 0: exact baseline and selected 18-step arm.](assets/2026-08-11_cl9v_r4_roi_best_dancing.png){ width=78% }

\clearpage

![All Jumping rows at seed 0.](assets/2026-08-11_cl9v_r4_roi_best_jumping.png){ width=78% }

\clearpage

`[measured: visual]` Facial detail changes without body/background movement or
a square seam. All `64/64` target composites per setting are exact outside the
ROI. Boundary RGB MAD rises from median `0.92` at `14` steps to `1.05` at `18`
and `1.13` at `20`; the `18`-step maximum is `1.89`. Jensen is a consistent
identity counterexample rather than a compositing failure.

**Interpretation.** `[measured]` Small-face identity is limited by local
denoising/detail capacity, not box underfill. `[hypothesis]` Earlier local
denoising lets target queries rebuild coherent identity before the global
latent is frozen. `[not established]` The sidecar does not prove that a second
pass is the most efficient deployment or that auxiliary training will retain
the gain.

## 3. Marion: roll improves quality, not identity

The same photograph is used in both conditions. Roll removes `-7.65` degrees
of eye-line rotation while retaining the canvas and scoring embedding.

| Seed | n | Original ID | Roll ID | ID delta | Wins | TOPIQ delta | Finite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `12` | `0.311` | `0.334` | **`+0.023`** | `9/12` | `+0.015` | `12/12` |
| `1` | `12` | `0.290` | `0.278` | `-0.012` | `3/12` | `+0.018` | `12/12` |
| `2` | `12` | `0.295` | `0.301` | `+0.006` | `4/12` | `+0.011` | `12/12` |
| `3` | `12` | `0.259` | `0.247` | `-0.012` | `3/12` | `+0.026` | `11/12` |
| **All** | **`48`** | `0.289` | `0.290` | **`+0.001`** | **`19/48`** | `+0.018` | `47/48` |

The all-seed ID interval is `[-0.008,+0.012]`. Only Laughing is directionally
consistent (`+0.043`, `3/4` wins); Night-ride loses `4/4`. Skiing averages
`-0.018` and remains occlusion-dominated.

![Per-prompt, per-seed paired changes. Seed-3 Skiing TOPIQ is unavailable for both conditions.](assets/2026-08-11_cl9v_r4_marion_heatmap.png){ width=88% }

\clearpage

![Marion prompts 84-89, original and roll face crops for seeds 0-3.](assets/2026-08-11_cl9v_r4_marion_visual_1.png){ width=100% }

![Marion prompts 90-95.](assets/2026-08-11_cl9v_r4_marion_visual_2.png){ width=100% }

\clearpage

`[measured: visual]` Gains and losses reverse across seeds. The source and
outputs remain off-axis; 2D rotation cannot remove yaw or reveal unseen facial
structure. `[report]` Five-point similarity was already neutral at seed `0`
(`+0.003`) and unstable. Same-image 2D canonicalization is therefore ruled
out as the easy fix. `[not established]` A genuinely frontal second photo,
multi-reference conditioning, or pose-balanced training remains untested.

## 4. Occluders: precise masks help but still miss the gate

`[code]` A reviewed per-image polygon only removes reference ownership from
target queries. Excluded queries fall through to the native lane; reference
K/V, reference mask, and weights remain unchanged. This oracle is stronger
than the rejected family-wide shape but is not deployable on a first pass.

| Non-Eddie arm/family | n | Base ID | Arm ID | Delta | Wins | TOPIQ | IoU | Face-side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static `r3`, combined | `14` | `0.409` | `0.409` | `+0.000` | `6/14` | `+0.009` | `-0.052` | `-12.7 px` |
| Precise, Skiing | `7` | `0.352` | `0.400` | `+0.048` | **`4/7`** | `+0.011` | `-0.023` | `-14.3 px` |
| Precise, Crying | `7` | `0.467` | `0.495` | `+0.028` | **`5/7`** | `+0.044` | `+0.017` | `-2.7 px` |
| **Precise, combined** | **`14`** | `0.409` | **`0.448`** | **`+0.038`** | **`9/14`** | **`+0.027`** | `-0.003` | `-8.5 px` |

The precise combined interval is `[-0.016,+0.094]`. Skiing misses the
predeclared `5/7` continuation gate.

![Reviewed Skiing polygons and blocked regions.](assets/2026-08-11_cl9v_r4_skiing_geometry_review.png){ width=88% }

![Reviewed Crying polygons.](assets/2026-08-11_cl9v_r4_crying_geometry_review.png){ width=88% }

\clearpage

![All Skiing rows: baseline, static mask, and precise geometry.](assets/2026-08-11_cl9v_r4_occluder_skiing.png){ width=86% }

\clearpage

![All Crying rows.](assets/2026-08-11_cl9v_r4_occluder_crying.png){ width=86% }

\clearpage

`[measured: visual]` Precise polygons preserve Crying hand placement better and
improve Elon, Jennie, Jisoo, and Marion; Keanu still loses `-0.133`. Skiing is
heterogeneous: Jisoo and Marion gain `+0.244` and `+0.192`, but their faces
shrink `43` and `47` pixels, while stacked/doubled eyewear remains. Localization
was part of the static failure, but fixed image-space ownership is insufficient.

## 5. Root cause and confidence

| Claim | Confidence | Basis |
|---|---|---|
| Local detail capacity is causal for small faces | **High** | `+0.097`, `43/56`, positive interval, four stable seeds, exact outside ROI |
| `18` steps is preferred on this panel | **High** | Passes all gates; higher mean ID and less movement than `20` |
| ROI should be unconditional | Medium-low | Jensen loses `8/8`; quality tails remain |
| Marion roll is the main cause | **Ruled out** | `+0.001`, `19/48`; two new seeds negative |
| Same-image 2D normalization solves Marion | **Ruled out** | Roll fails replication; five-point was neutral/unstable |
| Precise masks beat static masks | Medium-high | ID `+0.038` versus `+0.000`; alignment recovers |
| Precise ownership solves occlusion | **Ruled out** | Skiing `4/7`; interval crosses zero; visual topology failures |
| Results are config artifacts | **Ruled out** | Replay, sentinel, reproduction, contract, and hash gates pass |

### What is not established

- No training was performed; these are fixed-checkpoint interventions.
- Diagnostic seeds do not replace the fixed seed-0 full96 protocol.
- Latency/memory of production ROI inference was not measured.
- Prompt adherence was visually guarded, not numerically scored.
- A genuinely frontal Marion reference remains untested.
- Oracle polygons are baseline-informed and unavailable at first-pass inference.

## 6. Proposed experiments

### Priority 1 - training: internalize the replicated ROI gain

**Config:** `CL9T_shared_ba_highres_face_aux_24k_r1`  
**Single scientific change:** Add a `2x` square target-face crop denoising
example and auxiliary loss through the same reference-conditioned BA
processors. Share existing weights; do not add a separate identity adapter.
Keep `2,240` trainable tensors / `219,217,920` parameters.  
**Hypothesis:** High-resolution target queries during training will internalize
the local identity gain and reduce or remove the second inference pass.  
**Prediction:** Jumping/Dancing improve by at least `+0.05` ID from the first
matched `2k` checkpoint, including Jensen, without lowering clean prompts.  
**Risk:** Crop-heavy loss can overfit faces, weaken expression/hair consistency,
or alter effective batch behavior through extra memory.  
**Decision gate:** Non-Eddie Jumping/Dancing ID `>= CL9 +0.05`; full-panel ID
within `0.01` of CL9; TOPIQ mean/p10 within `0.01`; detection `1.0`; median
mask IoU within `0.02`; no clean-row visual regression. Keep
`pose_adapt_ratio=0` and `ca_mixing_for_face=false` in training and validation.

### Priority 2 - validation/deployment: selective ROI fallback

**Config:** `CL9V_smallface_roi18_selective_24k_r5`  
**Single scientific change:** Apply the validated `18`-step pass only when
detected face short side is below a fixed threshold; sweep thresholds without
changing denoising parameters.  
**Hypothesis:** A scale trigger captures the gain while avoiding unnecessary
repainting and reducing cost.  
**Prediction:** Full-panel ID improves, target gains remain near `+0.09`, and
untriggered rows remain exact.  
**Risk:** Face size alone may not identify Jensen-like negative cases.  
**Decision gate:** Same target ID gate as `r4`; all untriggered rows RGB-exact;
no triggered clean row loses more than `0.05` ID; measured runtime and peak
memory reported. If Jensen remains, add a general fallback score rather than
identity-specific rules.

### Priority 3 - training: pose-robust references, not pixel rotation

**Config:** `CL9T_offaxis_reference_pair_consistency_24k_r1`  
**Single scientific change:** For verified multi-view identities, sample
frontal/off-axis conditioning views for the same target and add consistency
supervision on identity-conditioned BA output. Do not frontalize pixels. The
base model and trainable inventory stay unchanged.  
**Hypothesis:** View-diverse conditioning teaches yaw robustness without
destroying details through synthetic warps.  
**Prediction:** Marion-like off-axis references improve while frontal-reference
and clean-prompt performance remain stable.  
**Risk:** Noisy identity groups or view imbalance can teach appearance
averaging; single-view identities cannot participate safely.  
**Decision gate:** On a held-out view-swap panel, off-axis ID `>= CL9 +0.03`,
frontal-reference ID within `0.01`, TOPIQ within `0.01`, and no prompt family
loses more than `0.03`. A future genuine frontal Marion photo should be held
out for validation, not used as training data.

### Deferred - occluder training

Do **not** launch `CL9T_query_visibility_gate_24k_r1` from this result. The
precise oracle missed its Skiing and visual gates. First build a controlled set
with explicit occluder segmentation and score goggle/hand location, topology,
and severity separately from face identity. A later training arm should combine
real/synthetic occluder augmentation with a retention loss; both identity and
occlusion-severity gates must pass. TOPIQ or ID alone is insufficient.

## 7. Implementation plan

1. **High-resolution BA training:** add defaults-off dataset outputs for the
   square `2x` target crop and coordinates from the active bbox. Route target
   queries through the same BA processors/reference K/V, log the auxiliary
   denoising loss separately, and assert unchanged trainable and optimizer
   membership. Retain step `0` plus every-`2k` full96 validation; add, but do
   not substitute, Jumping/Dancing curves.

2. **Selective inference:** reuse the exact `r4` ROI code and square-canvas/
   scheduler assertions. Add a defaults-off short-side threshold, log trigger
   indices plus first/second-pass hashes, keep outside-ROI exactness, and
   benchmark wall time and peak memory.

3. **Off-axis data contract:** require verified same-identity multi-view
   groups, record yaw/roll strata, and keep target/reference splits
   deterministic. Preserve explicit target Q and reference K/V ownership and
   report per-view results so aggregates cannot hide frontal regressions.

4. **Occlusion evidence before model code:** create a reviewed label set for
   goggles, hands, glasses, and hair; define retention metrics and blinded
   review. Only then design a time/query-adaptive BA gate. It may remove
   target-query reference ownership but must not replace reference K/V or
   enable pose adaptation.

5. **Prelaunch checks:** compose old/new configs, run shell syntax and Python
   compile checks, load the checkpoint in both toggle states, and verify Q/K/V
   routing, square sequences, processor counts, trainable inventory, optimizer
   roles, and checkpoint round-trip before the first update.

## 8. Reproducing

From `diffusion_template/`:

```bash
source /home/kolyangg/anaconda3/etc/profile.d/conda.sh
conda activate photomaker

python analysis/assets/cl9v_results_20260811_data/build_assets.py
python analysis/assets/cl9v_r4_results_20260811_data/build_assets.py
python tools/reports/publish_report.py \
  analysis/2026-08-11_cl9_validation_interventions_results.md
```

The `r4` builder expects `tmp/cl9v_r4_report_20260811/`. Authoritative Serv
root:

```text
/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/analysis_sidecars/cl9v_validation_chain_20260811_r4
```

Auditable derived data:

- [summary and gates](assets/cl9v_r4_results_20260811_data/summary.json),
  SHA-256 `b1e89fac5a2413739a64c2eb4555256082273589a9b128b5b39b1613e07d53ed`;
- [ROI pairs](assets/cl9v_r4_results_20260811_data/roi_paired_rows.csv),
  SHA-256 `ed7fe52aca32b598507b0639f745dfd3bf1206cf6c29950d50c35b798f35fdc8`;
- [occluder pairs](assets/cl9v_r4_results_20260811_data/occluder_paired_rows.csv),
  SHA-256 `dfc1a7f9c281a3f905f0949d61cd263666eba237082bcdef8e8cb383d034d8c8`;
- [Marion pairs](assets/cl9v_r4_results_20260811_data/marion_paired_rows.csv),
  SHA-256 `6a3448d4aa5d74c84405e493e785d58066018c5ddbaf1f1f270b9d16e281c736`;
- [retrieval hash audit](assets/cl9v_r4_results_20260811_data/retrieval_hash_audit.json),
  SHA-256 `742019d0a44855ba2d957bf5f1598772ab418ab582a499ed5d5b75d28087c9fb`;
- [asset builder](assets/cl9v_r4_results_20260811_data/build_assets.py).

The builder asserts checkpoint/config controls, exact replay gates, every ROI
outside-composite gate, and retrieved hashes before writing figures. Joins use
numeric `dataset_index`; baselines come only from exact-sentinel arms.

## 9. References

- [Pre-experiment edge-case analysis](2026-08-10_cl9_marion_occlusion_small_faces.md)
- [Baseline replay status](2026-08-10_cl9_baseline_replay_and_validation_status.md)
- [Completed `r4` record](../experiments/diagnostics/CL9V_validation_chain_24k_20260811_r4.json)
- [Small-face `r3` record](../experiments/diagnostics/CL9V_smallface_roi_refine_24k_20260810_r3.json)
- [Marion/occlusion `r3` record](../experiments/diagnostics/CL9V_marion_occlusion_validation_24k_20260810_r3.json)
- [Validation protocol](../docs/validation_protocol.md)
- [CL9 configuration](../src/configs/CL9_cosmic_joint_shadow_sa128_refscale_24k.yaml)

## Recommended next action

Move the replicated mechanism into training with
`CL9T_shared_ba_highres_face_aux_24k_r1`, while exposing the validated
`18`-step pass behind an opt-in small-face trigger. Retire Marion roll and
five-point normalization as identity fixes. Defer a learned occluder gate until
retention is measured explicitly and a sharper validation clears both identity
and topology gates.
