# Consolidated Large Dataset and BigCelebs branched-attention run comparison

**Date:** 3 August 2026

**Scope:** the August Large Dataset/BigCelebs architecture-result reports in
`analysis/`; July Cosmic dataset/data-pipeline studies are outside this
comparison because they answer a different dataset question.

**Validation:** fixed 96-image `manual_val` panel, one generated image per item

**Requested reference cases:**

- **Large Dataset base:** `rhca_large_dataset_sameid_40k_full96_r4`, immutable
  Comet ID [`a99db1fb953d4511827672380e6c1645`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/a99db1fb953d4511827672380e6c1645)
- **BigCelebs base:** `rhca_big_celebs_sameid_40k_full96_r1`, immutable Comet
  ID [`569cc685ff9144f5a9b42bf70e14e040`](https://www.comet.com/nikolay-2104/jul-comet-large-testing-tr/569cc685ff9144f5a9b42bf70e14e040)

## Executive comparison

No single scalar ranking across all nine runs is scientifically valid because
there are two validation processor constructions. The four historical/clean
hard-route runs use `legacy_full_copy` and start at identity `.3063`; residual
v2 and both anchored-mix families use `validation_native` and start near plain
PhotoMaker at `.4945–.5236`; query-adaptive hard v4 also uses
`validation_native` but deliberately starts with an untrained hard reference
face route and therefore begins at `.1488`. Comparisons within each group and
within-run changes are the strongest evidence.

The main conclusions are:

1. **The Large Dataset base remains the strongest historical long-run identity
   result.** It peaks at `.3904` at 24k and is `.3797` at the last complete 34k
   validation. It is the required Large Dataset base case.
2. **The BigCelebs base learns similarly but does not exploit its much larger
   dataset.** It peaks at `.3817` at 18k and falls to `.3628` at 32k after
   consuming only a minority of the dataset. It is the required BigCelebs base
   case.
3. **The historical identity gains are not clean BA attribution.** The old
   processor-install path failed open and trained about 171.29M broad U-Net,
   generic adapter, PhotoMaker-adapter, and BA parameters; part of the live
   PhotoMaker state was omitted from checkpoints.
4. **Clean hard BA32 is the best face-quality/prompt-preservation result in the
   legacy group, but not the best identity result.** It reaches `.7353`
   TOPIQ-Face mean, `.6281` p10, and `27.876` text at 32k, but only `.3273`
   identity.
5. **Residual v2 is closest to plain PhotoMaker because the spatial branch is
   effectively optional.** It keeps excellent absolute identity/face metrics,
   changes the images only modestly, and fails the spatial-reference causality
   test.
6. **Anchored mix v3 makes the spatial branch active while preserving coherent
   PhotoMaker structure, but the long rank run learns to reduce reference
   commitment.** Identity falls from `.4945` to `.4473` by 14k even while text
   and generic quality remain healthy.
7. **Hard v4 proves that a clean, query-adaptive, no-mix BA branch can learn.**
   Its identity rises `.0566` from initialization to 12k, close to the
   historical BigCelebs base's `.0638` gain over the same interval. However,
   its absolute face operating point remains poor and visual seams, colored
   face patches, occlusion failures, and duplicated accessories remain severe.
8. **Distance from PhotoMaker is not a success metric.** Residual v2 stays very
   close to PhotoMaker and does not learn useful spatial causality; hard v4
   moves farthest in face space but remains visually broken; the historical
   bases occupy a middle distance and achieve the strongest measured identity.

## 1. Evidence and comparability boundary

### 1.1 Dataset scale

| Dataset | Accepted images | Identities | Reference policy | Runs in this report |
|---|---:|---:|---|---|
| **Large Dataset** | 47,500 | 2,561 | Uniform distinct image from the same explicit identity | Large Dataset base only |
| **BigCelebs v2** | 349,348 | 68,648 | Base: uniform distinct same-ID image; scheduled arms: pinned centroid-ranked/top-three same-ID policy | All other runs |

BigCelebs is `7.35×` larger by accepted-image count and `26.8×` larger by
identity count. The fact that its base peaks at 18k—after seeing at most about
10% of its images—means data exhaustion cannot explain its plateau.

### 1.2 Fixed controls and important confound

The validation panel, prompts, seeds, reference images, generated/reference
bboxes, RealVisXL V4 base, scheduler, 50 inference steps, CFG, and metric
definitions are controlled. Branched cross-attention is disabled and every run
keeps `pose_adapt_ratio=0` and `ca_mixing_for_face=false`.

| Validation family | Runs | Step-zero identity | Meaning |
|---|---|---:|---|
| `legacy_full_copy` | Large base, BigCelebs base, scheduled broad, clean hard BA32 | .3063 | Historical full processor-state copy from the training base into the RealVis validation U-Net |
| `validation_native` PhotoMaker/native route | Residual v2, anchored mix v3, mix+rank | .4945–.5236 | RealVis keeps its native processors; residual v2 step zero is the explicit plain-PhotoMaker reference used below |
| `validation_native` hard reference route | Hard v4 | .1488 | Native background, but the target face must use an initially untrained target-Q/reference-KV route |

Absolute values must therefore be compared inside a validation family or
interpreted together with step-zero and visual evidence. A high native-group
step-zero score is not evidence that its spatial BA branch works.

### 1.3 Run index and source reports

| Label used below | Dataset | Run and immutable Comet ID | Last complete result used | Source analysis |
|---|---|---|---:|---|
| **Large base** | Large Dataset | `rhca_large_dataset_sameid_40k_full96_r4` · `a99db1f…` | 34k | `2026-08-01_large_dataset_big_celebs_ba_architecture_recommendations.md` |
| **BigCelebs base** | BigCelebs | `rhca_big_celebs_sameid_40k_full96_r1` · `569cc68…` | 32k | same 1 August report |
| Scheduled broad | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_40k_full96_r1` · `7c8b047…` | 14k | clean-BA32 report's controlled comparison |
| Clean hard BA32 | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_clean_ba32_40k_full96_r1` · `700240d…` | 32k | `2026-08-02_clean_ba32_32k_architecture_recommendations.md` |
| Residual v2 | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_residual_sa_v2_r32_40k_full96_r6` · `4d6186f…` | 2k | `2026-08-02_residual_sa_v2_2k_plain_photomaker_failure_analysis.md` |
| Anchored mix v3 | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_r32_2k_full96_r2` · `de23193…` | 2k | `2026-08-02_anchored_mix_sa_v3_2k_results_and_e4_plan.md` |
| Mix v3 + rank short | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_2k_full96_r2` · `f72ea55…` | 2k | `2026-08-02_anchored_mix_sa_v3_rank_2k_results_and_e5_plan.md` |
| Mix v3 + rank long | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_anchored_mix_sa_v3_rank_r32_40k_full96_r1` · `f5b5a70…` | 14k | `2026-08-03_anchored_mix_sa_v3_rank_40k_through14k_results_and_e6_plan.md` |
| Hard v4 | BigCelebs scheduled | `rhca_big_celebs_scheduled_v1_hard_ba_v4_q16_r32_20k_full96_r1` · `4086068…` | 12k | `2026-08-03_query_adaptive_hard_sa_v4_through12k_results.md` |

The 2k-only runs remain early causal/architecture diagnostics, not long-run
quality candidates. For runs that reached at least 8k, this report compares
both the last complete result and the best identity point at or after 8k, as
requested.

## 2. Side-by-side configuration comparison

### 2.1 Data and run design

| Run | Training data/order | Horizon evidenced | Main controlled question |
|---|---|---:|---|
| **Large base** | Large Dataset; uniform distinct same-ID reference | 34k | Historical base behavior on the smaller dataset |
| **BigCelebs base** | BigCelebs; uniform distinct same-ID reference | 32k | Whether much larger identity/data diversity breaks the historical plateau |
| Scheduled broad | Pinned BigCelebs policy-v1; top-three centroid-ranked reference and face-scale schedule | 14k | Deterministic data scheduling with the historical broad trainable state |
| Clean hard BA32 | Same pinned schedule | 32k | Historical attention math with exact BA-only ownership/checkpointing |
| Residual v2 | Same pinned schedule | 2k | Whether a weak additive spatial-reference residual can improve plain PhotoMaker |
| Anchored mix v3 | Same schedule, first 4,000 rows | 2k | Whether nonzero bounded native/reference interpolation makes the route causal |
| Mix v3 + rank short | Same first 4,000 rows | 2k | Objective-only test of differentiable matched-vs-shuffled ranking |
| Mix v3 + rank long | Same full 80,000-row plan | 14k | Whether the 2k dip recovers with a long horizon |
| Hard v4 | Same first 40,000 rows | 12k complete | Whether clean no-mix hard routing plus branch-specific target Q recovers historical identity learning |

### 2.2 Architecture, trainables, and objective

| Run | Target-face self-attention route | Sites / ranks | Actual trainable state | Objective / important behavior | Validation mode |
|---|---|---|---|---|---|
| **Large base** | Hard target-Q/reference-KV replacement; native target attention only outside face | 70 SA sites; rank 32 | **Fail-open ~171.29M BF16** broad U-Net/adapter + BA state, despite `train_ba_only`; incomplete live-state checkpoint | Face epsilon MSE; constant `1e-4`; no explicit reference-causal objective | `legacy_full_copy` |
| **BigCelebs base** | Same historical hard route | Same | Same fail-open broad state | Same | `legacy_full_copy` |
| Scheduled broad | Same historical hard route | Same | Same broad unintended state | Same; only data schedule controlled | `legacy_full_copy` |
| Clean hard BA32 | Same hard face replacement | 70 sites; rank 32 | **840 tensors / 31.949M**, exact BA-only state; observed BF16 trainables | Same face epsilon MSE, but correct ownership and schema-v2 checkpoint | `legacy_full_copy` |
| Residual v2 | `native + mask × gate × reference_residual`; frozen target Q; true reference key mask | 46 sites (`mid`, `up0`, `up1`); K/V32 + output32 | **414 tensors / 10.568M FP32** (`ref_kv`, `ref_output`, `gate`) | Gate starts `.10`; branch output exactly zero at initialization; full + face + `.1×` boundary loss; shuffled reference diagnostic detached and weight 0 | `validation_native` |
| Anchored mix v3 | `(1−α) native + α reference`, face-local; frozen target Q; reference endpoint RMS-matched | Same 46 sites; K/V32 + output32 | **414 / 10.568M FP32** (`ref_kv`, `ref_output`, `mix`) | `α` init `.50`, floor `.25`, max `.90`; full/face/boundary; detached 25% shuffle diagnostic | `validation_native` |
| Mix v3 + rank short | Same anchored interpolation | Same | Same | Only change: differentiable rank loss weight `.10`, relative margin `.02`, 50% shuffle | `validation_native` |
| Mix v3 + rank long | Exact same architecture/objective as short rank arm | Same | Same | Continuous long-run reproduction; learned mean mix falls from about `.50` toward `.35` | `validation_native` |
| Hard v4 | **Hard reference replacement with no native/reference face mix**; branch-only adaptive target Q | 46 sites; Q16 + K/V32 + output32 | **368 tensors / 12.329M FP32** (`ref_query`, `ref_kv`, `ref_output`) | Full/face/boundary; detached 25% shuffle diagnostic; native-face leakage measured as zero | `validation_native` |

This table separates two things that older names obscured: “branched attention
is present” and “only branched-attention parameters train.” The historical
bases satisfy the first but not the second.

## 3. Metric progression

### 3.1 Identity trajectories by validation family

![Legacy validation identity progression](assets/2026-08-03_ba_run_comparison/identity_progression_legacy.png)

The Large and BigCelebs bases both dip at 2k, recover by 8–14k, then oscillate
or regress. The scheduled broad run recovers faster after a deeper 2k dip but
does not exceed the Large base. Clean hard BA32 suffers the largest early
identity collapse, recovers above initialization only after several gates, and
then remains on a lower identity plateau despite better face quality.

![Native validation identity progression](assets/2026-08-03_ba_run_comparison/identity_progression_native.png)

The high purple/pink/black starting scores come from keeping the native
PhotoMaker face path, not from stronger spatial BA. Residual v2 barely moves
and is non-causal; anchored mix is causally active but declines; hard v4 starts
far lower because no native face route is available, then learns the largest
positive within-run change in this family.

### 3.2 Compact endpoint and post-8k comparison

| Run | ID 0 | ID 2k | ID final | Δ ID | Best ID ≥8k | Final text | Final face | Final p10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Large base** | .3063 | .2983 | **.3797 @34k** | **+.0734** | **.3904 @24k** | 27.160 | .7005 | .5721 | 100.0% |
| **BigCelebs base** | .3063 | .2841 | **.3628 @32k** | **+.0564** | **.3817 @18k** | 26.583 | .6816 | .5512 | 97.9% |
| Scheduled broad | .3063 | .2403 | .3684 @14k | +.0621 | .3727 @12k | 26.882 | .6688 | .5295 | 96.9% |
| Clean hard BA32 | .3063 | .1519 | .3273 @32k | +.0209 | .3347 @18k | **27.876** | **.7353** | **.6281** | 100.0% |
| Residual v2 | .5236 | .5086 | .5086 @2k | −.0149 | — | 26.722 | .7359 | .5916 | 100.0% |
| Anchored mix v3 | .4945 | .4779 | .4779 @2k | −.0165 | — | 26.846 | .7225 | .5882 | 100.0% |
| Mix v3 + rank short | .4945 | .4639 | .4639 @2k | −.0306 | — | 27.007 | .7178 | .5683 | 100.0% |
| Mix v3 + rank long | .4945 | .4649 | .4473 @14k | **−.0472** | .4817 @8k | 27.273 | .7249 | .5910 | 100.0% |
| Hard v4 | .1488 | .1148 | .2054 @12k | **+.0566** | .2213 @8k | **27.937** | .6303 | .4916 | 92.7% |

Bold does not mean a global winner across validation modes; it highlights the
most decision-relevant values inside a row or comparable family.

### 3.3 Step-zero, 2k, and final metric stages

| Run | Stage | ID | Text | Face mean | p10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| **Large base** | 0 | .3063 | 26.423 | .6225 | .5118 | 90.6% |
|  | 2k | .2983 | 27.357 | .6960 | .5712 | 100.0% |
|  | 34k final | .3797 | 27.160 | .7005 | .5721 | 100.0% |
| **BigCelebs base** | 0 | .3063 | 26.423 | .6225 | .5118 | 90.6% |
|  | 2k | .2841 | 27.812 | .6683 | .5109 | 93.8% |
|  | 32k final | .3628 | 26.583 | .6816 | .5512 | 97.9% |
| Scheduled broad | 0 | .3063 | 26.423 | .6225 | .5118 | 90.6% |
|  | 2k | .2403 | 27.380 | .6318 | .4703 | 88.5% |
|  | 14k final | .3684 | 26.882 | .6688 | .5295 | 96.9% |
| Clean hard BA32 | 0 | .3063 | 26.423 | .6225 | .5118 | 90.6% |
|  | 2k | .1519 | 27.929 | .6700 | .5189 | 74.0% |
|  | 32k final | .3273 | 27.876 | .7353 | .6281 | 100.0% |
| Residual v2 | 0 | .5236 | 26.333 | .7473 | .5935 | 100.0% |
|  | 2k/final | .5086 | 26.722 | .7359 | .5916 | 100.0% |
| Anchored mix v3 | 0 | .4945 | 25.800 | .7178 | .6230 | 100.0% |
|  | 2k/final | .4779 | 26.846 | .7225 | .5882 | 100.0% |
| Mix v3 + rank short | 0 | .4945 | 25.800 | .7178 | .6230 | 100.0% |
|  | 2k/final | .4639 | 27.007 | .7178 | .5683 | 100.0% |
| Mix v3 + rank long | 0 | .4945 | 25.800 | .7178 | .6230 | 100.0% |
|  | 2k | .4649 | 27.006 | .7173 | .5695 | 100.0% |
|  | 14k final | .4473 | 27.273 | .7249 | .5910 | 100.0% |
| Hard v4 | 0 | .1488 | 27.413 | .5430 | .4036 | 65.6% |
|  | 2k | .1148 | 27.610 | .6031 | .4949 | 71.9% |
|  | 12k final | .2054 | 27.937 | .6303 | .4916 | 92.7% |

### 3.4 What the progression establishes

- Every long hard-route history has an early identity dip. A 2k drop alone is
  not enough to reject a hard-route architecture.
- The historical bases make their useful identity gains between roughly 4k
  and 18–24k, but neither benefits indefinitely from later training.
- BigCelebs does not lift the ceiling despite much greater unused diversity.
  This points to architecture/objective/optimization, not insufficient data.
- Clean ownership reallocates the result: identity falls relative to the broad
  historical state while text, mean face quality, and weak-tail face quality
  improve substantially.
- The rank objective does not rescue anchored mixing. The independent 2k rank
  run is worse than no-rank v3, and the long run's temporary 4–6k recovery
  reverses into four consecutive identity declines through 14k.
- Hard v4 is neither dead nor PhotoMaker-anchored: it rises from the 2k trough
  and produces a within-run gain comparable to the BigCelebs base. Its failure
  is the low and artifact-heavy operating point, not absence of learning.

## 4. Matched image progression: step 0, 2k, and final

Each row below is the same fixed validation item. The right column is the last
locally evidenced panel: 34k, 32k, 14k, 12k, or 2k depending on the run. For
the three 2k-only experiments, the 2k and final columns are intentionally the
same image.

### 4.1 Angry / Keanu

![Angry Keanu progression](assets/2026-08-03_ba_run_comparison/progression_Angry_man__keanu.jpg)

The historical bases and scheduled broad run move from the common legacy
initial face toward a recognizable but increasingly generic/smoothed Keanu.
Clean BA32 changes expression more aggressively. Native residual/mix variants
preserve the original PhotoMaker composition and face far more closely. Hard
v4 never resolves its central face-mask/feature-collapse artifact for this
sample.

### 4.2 Kickboxing / Jisoo

![Kickboxing Jisoo progression](assets/2026-08-03_ba_run_comparison/progression_Kickboxing_jisoo.jpg)

This is a useful worst-case geometry test. The native PhotoMaker/mix rows keep
the face attached and coherent, although identity does not improve in the
aggregate. The hard legacy rows reconstruct the face forcefully and sometimes
improve recognition, but produce mouth/skin discontinuities. Hard v4 remains
dominated by a vertical mask/lighting patch at 12k.

### 4.3 Night ride / Jennie

![Night-ride Jennie progression](assets/2026-08-03_ba_run_comparison/progression_Night-ride_jennie.jpg)

All architectures preserve the body, vest, camera, and neon street. This
shows that most architectural differences are face-local. The clean hard
route strongly changes expression and face brightness; anchored mixing stays
close to the native face; hard v4 converts the face region into a colored
patch and still has severe light/skin integration failure at 12k.

### 4.4 Skiing / Jensen

![Skiing Jensen progression](assets/2026-08-03_ba_run_comparison/progression_Skiing_man_jensen.jpg)

The scene remains stable, but eyewear exposes the routing difference. The
legacy and clean hard routes introduce or duplicate reference glasses beneath
the target goggles. Mix variants are more coherent but increasingly resemble
safe PhotoMaker refinements. Hard v4 reconstructs a face but duplicates
goggles/glasses and misplaces eyes and lenses.

## 5. Final images versus plain PhotoMaker

### 5.1 Direct face-crop comparison

The first column is residual-v2 step zero, which is plain PhotoMaker by the
processor equation and is the `validation_native` PhotoMaker reference. The
remaining columns are each run's last locally evidenced image.

![Final face comparison against plain PhotoMaker](assets/2026-08-03_ba_run_comparison/final_face_comparison.jpg)

### 5.2 Quantified image-distance diagnostic

To make “looks like PhotoMaker” less subjective, a stratified 12/96 set was
used: one example from every prompt family and varied identities. Full images
were decoded at 256×256; fixed validation bbox crops were resized to 96×96.
The table reports RGB SSIM and face RMSE. Higher SSIM means more similar;
higher RMSE means more different. These are **change diagnostics, not quality
or identity metrics**.

| Run | Own step 0 → final full SSIM | Own step 0 → final face SSIM | Plain PM → final full SSIM | Plain PM → final face SSIM | Plain PM → final face RMSE |
|---|---:|---:|---:|---:|---:|
| **Large base** | .735 | .278 | .764 | .493 | .147 |
| **BigCelebs base** | .731 | .302 | .753 | .450 | .164 |
| Scheduled broad | .725 | .297 | .748 | .451 | .164 |
| Clean hard BA32 | .689 | .248 | .717 | .509 | .158 |
| Residual v2 | **.972** | **.869** | **.972** | **.869** | **.055** |
| Anchored mix v3 | .901 | .418 | .923 | .642 | .116 |
| Mix v3 + rank short | .902 | .422 | .920 | .626 | .120 |
| Mix v3 + rank long | .889 | .380 | **.940** | **.738** | **.094** |
| Hard v4 | .850 | .281 | .890 | **.446** | **.176** |

The comparison supports four specific conclusions:

1. **Residual v2 is effectively a PhotoMaker-preserving correction.** Its
   final face SSIM to PhotoMaker is `.869`, by far the closest endpoint, but it
   has no useful spatial-reference causal signal and loses identity.
2. **The long anchored-mix model moves toward the safe PhotoMaker basin.** It
   is closer to plain PhotoMaker at the endpoint than the short mix models
   (`.738` face SSIM versus `.642/.626`) while identity declines. This agrees
   with the learned mix coefficient retreating toward the native path.
3. **Hard v4 produces the largest face-region deviation by RMSE (`.176`).**
   That confirms it escaped plain PhotoMaker, but the direct crops show that
   much of the deviation is malformed structure, color patches, or accessory
   duplication rather than identity improvement.
4. **The historical bases also move far from their own step zero and from
   plain PhotoMaker, but with a better identity/structure trade-off than hard
   v4.** Their identity advantage cannot be assigned solely to BA because of
   the broad unintended trainables.

## 6. What each architectural stage contributes

| Stage | What works | What does not work |
|---|---|---|
| Historical hard route on Large Dataset | Strongest observed long-run identity; clear multi-thousand-step improvement | Smaller dataset; broad unintended trainables; incomplete checkpoint attribution; eventual plateau/oscillation |
| Historical hard route on BigCelebs | Similar positive trajectory on a much larger identity pool | More data does not raise the ceiling; late identity/text/face-quality regression; same attribution bugs |
| Deterministic scheduled broad data | Reproduces recovery and reaches `.3727` by 12k | Scheduling alone does not exceed either base and retains broad-trainable bugs |
| Clean hard BA32 | Exact BA ownership; complete checkpoint; best legacy-group prompt and face quality | Severe 2k identity/coverage collapse; lower identity ceiling; target-face replacement remains intrusive |
| Residual v2 | Best PhotoMaker structure and highest native absolute metrics | Reference branch starts at zero, remains optional, and does not become reference-causal |
| Anchored mix v3 | Nonzero causal branch with coherent structure and reversible native/reference interpolation | Native path is an optimization escape route; identity falls; mix reduces reference commitment |
| Mix v3 + rank | Correct/shuffled separation becomes measurable | Ranking can improve by damaging the wrong branch; does not improve production identity; long run retreats toward PhotoMaker |
| Query-adaptive hard v4 | Clean no-mix BA attribution; zero native-face leakage; adaptive target Q; real relative identity learning | Very poor initialization, weak absolute identity, hard-mask seams, lighting/occlusion failure, accessory duplication, unstable late identity |

## 7. Consolidated interpretation

Across the complete series, branched attention has passed three progressively
harder tests:

1. **Can a reference route change the output?** Historical hard routing and
   v4: yes. Residual v2: barely.
2. **Does the output depend on the spatial reference?** Anchored mix and v4:
   yes. Residual v2: effectively no.
3. **Does correct spatial-reference use reliably improve identity while
   preserving structure?** No evaluated clean architecture has yet passed.

The two required base cases remain the clearest empirical anchors:

- **Large Dataset base:** strongest observed identity peak and endpoint, but
  scientifically contaminated by broad trainables.
- **BigCelebs base:** clearest demonstration that much more unseen data does
  not break the plateau under the same architecture/objective.

Clean BA32 shows that strict BA-only training can yield excellent prompt and
face-quality metrics, but loses identity capacity. Residual and mix designs
show that protecting PhotoMaker too strongly makes the safe native solution
easier than learning spatial identity correspondence. Hard v4 removes that
escape and learns, but its hard face interface is not sufficiently integrated
with target geometry, lighting, occlusions, and the surrounding residual
stream.

The result series therefore does not support either extreme:

- staying near PhotoMaker preserves quality but does not prove useful BA;
- moving far from PhotoMaker proves branch influence but can produce severe
  non-identity artifacts.

The unresolved target is a causally reference-dependent branch that changes
the face in the correct identity direction while retaining the native model's
geometry and face integration. This document is a consolidation of observed
results and deliberately does not propose the next experiment.

## 8. Artifact and provenance notes

- Missing exact step-zero/2k/final panels were fetched locally using the
  immutable Comet keys above. Every new export resolved to the requested step,
  downloaded 96 images, and reported zero warnings/errors.
- `saved/` was not present in this local checkout at analysis time; all image
  grids therefore use exact local `comet_data/` exports rather than an
  unverified substitute.
- The raw image-distance record is
  `analysis/assets/2026-08-03_ba_run_comparison/image_distance_summary.json`.
- The source metric values are the immutable Comet scalar histories; no
  validation settings or metric definitions were recomputed or changed.
