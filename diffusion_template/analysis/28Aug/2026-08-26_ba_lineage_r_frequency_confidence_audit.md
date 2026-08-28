# Branched-attention lineage audit: what CL39 R, D-low/D-high, and confidence actually do

**Date:** 26 August 2026  
**Evidence cutoff:** 11:45 BST, 26 August 2026  
**Scope:** sealed 24k checkpoints for CL19, CL23, CL27, and CL39; fixed-96
Serv A100 whole-denoising interventions; deterministic stratified 16-cell
diagnostic subset; subject-v2 mask-matched identity and matched visual review.  
**Primary metric:** subject-v2 identity similarity, matched to the fixed
generated-face box. RGB MAE and SSIM measure intervention size, not quality.  
**Central limitation:** all counterfactual images are complete 50-step
denoising runs. Low-only and high-only images are nonlinear interventions and
cannot be added in RGB space to reconstruct the actual image.

## Executive answer

The concern about CL39's raw reference branch is valid, but it does **not** mean
that normal CL39 behaves like the raw-R images.

1. **CL39 R is a fragile standalone route.** Its raw `R-on-face` intervention
   has subject-v2 ID `0.42241`, below CL23 raw R (`0.44318`), CL27 raw R
   (`0.46396`), and CL19's trained reference-owned route (`0.48975`). Actual
   CL39 beats raw R on `16/16` cells by `+0.13513`, with fixed-cell bootstrap
   interval `[+0.09178,+0.18101]`. Visual review finds unmistakable major
   face-part/object fusion in about `8/16` CL39 raw-R cells, versus `0/16` for
   CL19's trained route. `[measured][visual]`
2. **Normal CL39 is not merely PhotoMaker/native attention.** Actual CL39
   reaches `0.55754`, while N-only reaches `0.52184`; actual gains `+0.03570`,
   wins `11/16`, and has interval `[+0.01204,+0.06613]`. Actual differs from
   N-only over `93.80%` of face-crop pixels above `1/255`. The correction is
   active and, on this subset, beneficial. `[measured]`
3. **D-low/D-high do not create the raw-R artifacts.** The raw-R arm bypasses
   the Gaussian frequency split and routes `N+S(R-N)` directly. Its failures
   therefore exist before `D_low/D_high` are applied. The band mechanism instead
   attenuates and reshapes that residual: CL39 face MAE versus N-only falls from
   `0.08350` for raw R to `0.04460` for low-only, `0.03001` for high-only, and
   `0.04777` for actual CL39. `[code][measured]`
4. **Both frequency bands are active; high is the more ID-efficient isolated
   intervention.** Across CL23/27/39, high-only changes fewer pixels than
   low-only yet has higher mean ID. CL39 actual nevertheless beats low-only by
   `+0.02961` (CI `[+0.01258,+0.04730]`) and high-only by `+0.01707` (CI
   `[-0.00003,+0.03826]`). This supports complementary use, but does not prove
   pixel-additive synergy. `[measured][limitation]`
5. **CL39 confidence is doing important work.** Forcing `C=1` keeps the normal
   low/high equation but reduces ID from `0.55754` to `0.49984`; actual gains
   `+0.05770`, wins `14/16`, and has interval `[+0.03443,+0.07939]`. It also
   raises face MAE versus N-only from `0.04777` to `0.06610` and visibly
   reintroduces several raw-R-like corruptions. `[measured][visual]`

The best interpretation is: **CL39 learned to extract useful identity signal
from an underconstrained R lane by anchoring it to N, splitting/scaling its
residual, and abstaining through confidence.** Improving R remains a sensible
architecture direction because a more coherent source residual may preserve
CL39's gains with less reliance on suppression. The present audit does not show
that removing the frequency or confidence mechanisms would improve CL39.

## 1. Exact experiment

### 1.1 Fixed controls

Every arm used the sealed Serv validation path: RealVisXL V4.0, DDIM50, CFG 5,
seed 0, step 24k, fixed 96 prompts, identity references, generated/reference
face boxes, masks, and one image per item. `pose_adapt_ratio=0` and
`ca_mixing_for_face=false` remained fixed. The diagnostic set selected two
cells per identity with seed `390024`: `1, 7, 13, 17, 33, 35, 38, 40, 51, 55,
63, 69, 78, 80, 87, 93`. `[record]`

All four instrumented actual arms reproduced their sealed 96 PNGs exactly:
mean RGB MAE `0`, maximum image MAE `0`, maximum absolute difference `0`, and
no pixels changed. This is the inference-parity gate for interpreting the
counterfactuals. `[measured]`

Immutable records, each with exact actual replay on 96/96 images:

- CL19 checkpoint:
  `07aefcb03e432e84f31556429e0bfe221c23703cbe2164e09fe988f984cd2bd9`  
  Comet: `cfeda7b55c174b3c83e8d40537ebb6dd`.
- CL23 checkpoint:
  `70201f0e82c9cb24aeb5adc27ad660e5e11aea8d29b6969c449ec39e3c8b379c`  
  Comet: `a9ec9c59d1624c68acb98737dcd65298`.
- CL27 checkpoint:
  `100072242ca34b2056f512f41a32e7aa8e7b98e4b10146043fd258a410ca8a50`  
  Comet: `dbfbf40c3bdd4f70bedc58bda3dfb9cd`.
- CL39 checkpoint:
  `74f61d03ccb94cae9569c158d2f9369eb3dd5274070ef74ee254b926656fbd07`  
  Comet: `b1ca0b3da679401c85b991f1bbdf0b2a`.

### 1.2 Arms and their meanings

At an analysed processor, let `N` be native target self-attention, `R` be the
target-query/reference-face-KV message, `S` be the spatial face router, and
`D=R-N`. CL23/27/39 split `D` with a fixed Gaussian into `D_low+D_high` and
apply progress-dependent gains. CL39 additionally applies entropy confidence
in its configured up-block groups. `[code]`

| Arm | Whole-denoising target message | What it tests |
|---|---|---|
| Actual | sealed trained equation | operating point |
| N-only | `N` | whether BA correction matters |
| Raw R-on-face | `N+S(R-N)` | standalone reference-route stress test |
| Low-only | `N+S*C*g_low*D_low` | low-band intervention |
| High-only | `N+S*C*g_high*D_high` | high-band intervention |
| CL39 C=1 | normal CL39 bands with `C` forced to one | confidence causal ablation |

CL19's actual path is its trained reference-owned face route, so its `actual`
and `raw R-on-face` columns are the same by construction. This makes CL19 the
right qualitative comparator for whether a reference lane can be trained to
stand on its own, but **not** a controlled same-checkpoint comparison against
CL39 raw R. `[code][limitation]`

## 2. Quantitative results

### 2.1 Identity and intervention strength

| Lineage / arm | Subject-v2 ID ↑ | Face RGB MAE vs N-only | Face pixels changed >1/255 |
|---|---:|---:|---:|
| CL19 actual / trained R route | `0.48975` | `0.07648` | `97.22%` |
| CL19 N-only | `0.47244` | `0` | `0%` |
| CL23 actual | `0.51368` | `0.06617` | `95.83%` |
| CL23 N-only | `0.48998` | `0` | `0%` |
| CL23 raw R | `0.44318` | `0.08166` | `96.61%` |
| CL23 low-only | `0.50078` | `0.06026` | `95.27%` |
| CL23 high-only | `0.52027` | `0.03697` | `89.91%` |
| CL27 actual | `0.52132` | `0.06743` | `96.03%` |
| CL27 N-only | `0.49972` | `0` | `0%` |
| CL27 raw R | `0.46396` | `0.08258` | `96.85%` |
| CL27 low-only | `0.51091` | `0.06514` | `96.00%` |
| CL27 high-only | `0.52977` | `0.03833` | `90.20%` |
| **CL39 actual** | **`0.55754`** | `0.04777` | `93.80%` |
| CL39 N-only | `0.52184` | `0` | `0%` |
| CL39 raw R | `0.42241` | `0.08350` | `96.59%` |
| CL39 low-only | `0.52793` | `0.04460` | `92.99%` |
| CL39 high-only | `0.54047` | `0.03001` | `86.96%` |
| CL39 C=1 | `0.49984` | `0.06610` | `95.32%` |

![](assets/ba_lineage_branch_audit_20260826/lineage_metric_heatmaps.png){ width=94% }

*Figure 1. Identity is a quality-relevant primary metric; face RGB MAE is only
the strength of the intervention relative to N-only. Missing arms are not part
of that lineage.*

### 2.2 Paired causal comparisons

| Lineage | Comparison, actual minus arm | Mean Δ ID | W/T/L | Fixed-cell bootstrap 95% interval |
|---|---|---:|---:|---:|
| CL19 | actual − N-only | `+0.01731` | `10/0/6` | `[-0.01343,+0.05121]` |
| CL23 | actual − N-only | `+0.02371` | `9/0/7` | `[-0.00913,+0.05969]` |
| CL23 | actual − raw R | `+0.07050` | `14/0/2` | `[+0.04380,+0.09883]` |
| CL23 | actual − low-only | `+0.01290` | `9/0/7` | `[-0.00817,+0.03407]` |
| CL23 | actual − high-only | `-0.00659` | `7/0/9` | `[-0.03189,+0.01835]` |
| CL27 | actual − N-only | `+0.02160` | `9/0/7` | `[-0.00565,+0.05252]` |
| CL27 | actual − raw R | `+0.05736` | `14/0/2` | `[+0.02926,+0.08809]` |
| CL27 | actual − low-only | `+0.01041` | `9/0/7` | `[-0.00897,+0.03049]` |
| CL27 | actual − high-only | `-0.00846` | `8/0/8` | `[-0.02959,+0.01365]` |
| **CL39** | **actual − N-only** | **`+0.03570`** | **`11/0/5`** | **`[+0.01204,+0.06613]`** |
| **CL39** | **actual − raw R** | **`+0.13513`** | **`16/0/0`** | **`[+0.09178,+0.18101]`** |
| **CL39** | **actual − low-only** | **`+0.02961`** | **`13/0/3`** | **`[+0.01258,+0.04730]`** |
| CL39 | actual − high-only | `+0.01707` | `9/0/7` | `[-0.00003,+0.03826]` |
| **CL39** | **actual − C=1** | **`+0.05770`** | **`14/0/2`** | **`[+0.03443,+0.07939]`** |

Intervals resample only the fixed 16 cells (`100,000` draws, seed `390026`).
They describe this diagnostic panel, not training-seed or population
uncertainty. No multiple-comparison claim is made. `[measured][limitation]`

## 3. What the lineage says about R

### 3.1 CL19 is the useful comparator the question anticipated

CL19's reference lane is the face owner in its trained equation. On the same
16 prompts it remains visually coherent, while N-only changes identity,
expression, pose, and accessories. This is direct evidence that a reference
lane based on the same broad BA idea can be trained to act as a plausible
denoising path. `[visual][code]`

The ID ordering of the raw/reference-owned routes is:

`CL19 0.48975 > CL27 0.46396 > CL23 0.44318 > CL39 0.42241`.

That ordering supports the concern that CL39's R is less self-sufficient. It
does not prove that CL39's reference projection is intrinsically worse,
because CL19 actual is trained and evaluated at its operating point, whereas
CL23/27/39 raw R removes the frequency/gain/confidence context under which
those checkpoints were optimized. `[measured][limitation]`

### 3.2 The issue becomes larger at CL39 even as the complete model improves

Actual ID rises monotonically across the selected checkpoints:
`0.48975 → 0.51368 → 0.52132 → 0.55754`. N-only also rises:
`0.47244 → 0.48998 → 0.49972 → 0.52184`. Raw R does not follow that trend and
falls to `0.42241` at CL39. `[measured]`

This divergence is the key architectural signal. CL39 is the best complete
system while its raw reference stress route is the weakest. The later model
has not merely learned a better R; it has learned a better **controlled
correction around N**. A plausible hypothesis is that detached confidence and
scaled residual training reduce the gradient pressure for R to remain
standalone-coherent. The current interventions reveal that possibility but do
not identify its training-time cause. `[inference][hypothesis]`

## 4. Visual evidence

The exploratory visual rubric calls a cell a major structural failure only
when duplicated/missing facial parts or hand/object-to-face fusion are
unmistakable at overview scale. One unblinded reviewer found about `0/16` major
failures for CL19's trained R route, `3/16` for CL23 raw R, `2/16` for CL27 raw
R, `8/16` for CL39 raw R, and `4/16` for CL39 C=1. These counts are descriptive,
not a blinded human-study result. `[visual][exploratory]`

![](assets/ba_lineage_branch_audit_20260826/07_eddie.png){ width=100% }

*Figure 2. Cell 07. Rows are CL19, CL23, CL27, CL39. CL19's trained reference
route carries the crying face coherently. CL23/27 raw R pull the hand toward
the eye; CL39 raw R breaks the face more severely. CL39 low/high remain
coherent, while C=1 reintroduces hand/face fusion.*

![](assets/ba_lineage_branch_audit_20260826/38_jensen.png){ width=100% }

*Figure 3. Cell 38. CL39 raw R catastrophically mixes goggles, glasses, eyes,
and mouth. Normal CL39 is coherent; C=1 substantially recreates the failure.
This is evidence about the combined denoising path, not just an attention-map
rendering artifact.*

![](assets/ba_lineage_branch_audit_20260826/55_jisoo.png){ width=100% }

*Figure 4. Cell 55. The hand/object crossing exposes correspondence and
ownership errors in raw R. Frequency-separated routes retain topology;
unattenuated confidence again fails.*

![](assets/ba_lineage_branch_audit_20260826/80_lex.png){ width=100% }

*Figure 5. Cell 80. CL23 raw R already shows eye-region corruption; CL27 is
more stable; CL39 raw R again collapses. Actual CL39 preserves pose and laugh,
showing why raw R must not be treated as the normal model output.*

## 5. D-low/D-high and confidence interpretation

### 5.1 D-low/D-high are working, but their image ablations are nonlinear

Both isolated bands change most face pixels and produce distinct images. High
only is consistently the smaller intervention (`0.03697`, `0.03833`, `0.03001`
face MAE for CL23/27/39) yet has higher ID than low only (`0.52027`, `0.52977`,
`0.54047` versus `0.50078`, `0.51091`, `0.52793`). The high band therefore
appears more ID-efficient on this selected set. `[measured]`

For CL23/27, high-only slightly exceeds actual mean ID, but paired intervals
cross zero. For CL39, actual exceeds each isolated band and decisively exceeds
low-only. A reasonable interpretation is that low-frequency structure and
high-frequency identity detail become complementary in CL39. Because each arm
changes the entire denoising trajectory, the experiment does not establish a
linear decomposition or prove that low-band content itself improves perceptual
quality. `[measured][inference][limitation]`

### 5.2 Confidence is not cosmetic

The C=1 arm uses the trained CL39 low/high route and changes only the entropy
attenuation. Its large ID loss, positive paired interval, greater distance from
N, and visible return of raw-R-like failures jointly show that confidence is a
functional safety/quality mechanism. `[measured][visual]`

This does not establish that normalized attention entropy is the best
confidence estimator. It establishes that the checkpoint depends on the
attenuation supplied by the current estimator. The earlier code/telemetry
audit still applies: entropy mixes match ambiguity, zero-sink mass, and logit
scale, so a learned reliability estimator with valid-reference mass and N/R
agreement remains a plausible improvement. `[code][hypothesis]`

## 6. Confidence table and what is not established

| Claim | Confidence | Basis |
|---|---|---|
| CL39 correction is used in practice, not just PhotoMaker/N | High on selected 16 | actual − N positive interval; 93.8% changed face pixels |
| Raw CL39 R is fragile as a standalone route | High on selected 16 | 16/16 ID losses; large visual failures |
| D-low/D-high cause the raw-R artifacts | Rejected | raw-R arm bypasses the split |
| Both bands are active | High | distinct full denoising outputs and large pixel changes |
| High-only is more ID-efficient than low-only | Medium | consistent across three lineages; only 16 cells/one seed |
| Both bands are jointly necessary for CL39 quality | Medium | actual beats both means; high-only interval nearly touches zero; quality panel is limited |
| Current CL39 confidence helps | High on selected 16 | C=1 loses 14/16 with positive interval and visual regressions |
| Entropy is an optimal confidence estimator | Not established | no alternative confidence estimator was tested |
| CL39 R became worse because of one specific training feature | Not established | checkpoint lineage is not a one-factor controlled training ablation |
| Ordinary CL39 has a general artifact problem | Not supported | actual is coherent here and best on identity; raw R is off-operating-point |

## 7. Decision and next experiment

The experiment strengthens, rather than replaces, the earlier architecture
recommendation:

1. Keep CL39's N anchor, frequency bands, and confidence in normal inference.
2. Test a training-only CL14-inspired reference-face ownership obligation on a
   small deterministic fraction of steps, so R must sometimes carry a coherent
   face instead of only supplying an attenuated residual.
3. Evaluate that run with this exact audit. The primary causal success gate is
   fewer major raw-R failures and higher raw-R ID **without** losing actual
   fixed-96 ID, prompt adherence, face quality, or topology.
4. Only after route training is isolated should a richer confidence estimator
   or separate low/high reliability/caps be added.

The present audit is not a license to remove `D_low/D_high` or set `C=1`.
Those changes move the checkpoint away from its successful operating point.
`[decision]`

## 8. Reproduction and recovery audit trail

The user-facing package is
`serv_run_packages/BA_lineage_branch_audit_serv_r1/`; its entry YAML is
`run_BA_lineage_branch_audit_serv_r1_1gpu.yaml`. It used a four-GPU parallel
continuation and a five-GPU CL23 recovery under the explicit ten-GPU exception.
Generation jobs were:

- sequential CL19/CL23 launcher:
  `lm-mpi-job-a64df24d-350d-4cae-bdf5-8b31d2a5af29`;
- parallel CL27/CL39 launcher:
  `lm-mpi-job-46454c8f-2967-4d12-9a28-a75d7232cf86`;
- rejected clean-branch CL23 replay:
  `lm-mpi-job-80541ec2-6f30-4292-9ca3-506255088b30`;
- historical-runtime lifecycle diagnostic:
  `lm-mpi-job-4d9dcf45-0823-4e73-a5ca-ba6bd144ce54`;
- historical-runtime symbolizer hang:
  `lm-mpi-job-2858db1e-5c33-4845-862d-d600a071ce8a`;
- final five-worker CL23 recovery:
  `lm-mpi-job-c7456c95-415d-4ce4-9d3e-5026b5196440`.

The first CL23 clean-branch replay differed from the sealed model (mean RGB
MAE `0.00712`, maximum image MAE `0.01549`, maximum absolute difference `1.0`),
so it was rejected. Replaying through the immutable historical CL23 runtime
restored exact 96/96 parity. Two operational fixes were then required: attach
the analysis mode through the stable private processor map after the public
Diffusers map disappeared on batch two, and disable C++ addr2line symbolization
after a platform hang. Neither changes the model equation or validation
controls. `[record][measured]`

Final generated state: `18` arms × `96` images = `1,728` Serv images, with 18
gate JSONs, 16 matched sample panels, 16 lineage overview pages, per-sample CSV,
aggregate CSV/JSON, and metric heatmaps. The selected scoring archive SHA-256
is `7d0aaea3569b7397e6f9cabfa5d5664984aecc1e8cfcfe786e1825fbae6fcaf4`.

Key reproducible local assets:

- `artifacts/ba_lineage_branch_audit_20260826/AUDIT_COMPLETE.json`;
- `artifacts/ba_lineage_branch_audit_20260826/gates/`;
- `artifacts/ba_lineage_branch_audit_20260826/summary.json`;
- `artifacts/ba_lineage_branch_audit_20260826/branch_metrics.csv`;
- `artifacts/ba_lineage_branch_audit_20260826/scored/analysis_summary.json`;
- `artifacts/ba_lineage_branch_audit_20260826/scored/identity_and_distance_per_sample.csv`;
- `artifacts/ba_lineage_branch_audit_20260826/samples/` and `overviews/`;
- `tools/analysis/render_ba_lineage_branch_audit.py`;
- `tools/analysis/analyze_ba_lineage_branch_audit.py`.

## Bottom line

Investigating CL19's R was the right control. It shows that a reference-owned
route can produce coherent faces, whereas CL39's raw R is materially less
robust. But CL39's actual result is not raw R: it is a successful, N-anchored,
frequency-shaped, confidence-attenuated correction. The audit provides strong
evidence that the correction and confidence are both used, that raw-R failures
precede the D-low/D-high split, and that CL39 currently depends on those
controls to turn a fragile R lane into its best final result.
