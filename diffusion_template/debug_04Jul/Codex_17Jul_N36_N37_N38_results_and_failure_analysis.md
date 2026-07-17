# N36, N37, and N38 result analysis

Date: 17 July 2026

## Executive conclusion

N36, N37, and N38 are not completely disabled, but none is a successful
identity-owner run. Their branch weights update, validation receives the trained
processors, and the output changes slightly. The actual BA correction is too
weak to displace PhotoMaker identity, however, and it does not become stronger
from 4k to 8k.

The primary remaining problem is architectural attenuation:

- N32 applied its target-face residual at all 70 SDXL cross-attention sites with
  unit residual gates.
- N36-N38 apply it at only 16 sites. Six have gate `1.0`; ten have gate `0.5`.
  This is only 11 unit-gate site equivalents, or about one sixth of N32's
  injection opportunity.
- The other 54 cross-attention sites retain full PhotoMaker conditioning. Even
  within the 16 selected sites, ten retain half of PhotoMaker identity context.
  In the BA pass, this leaves approximately `54 + 10 * 0.5 = 59` of 70
  PhotoMaker identity-context site equivalents.
- The final prediction also starts from a fully identity-conditioned PhotoMaker
  prediction and adds BA only as a post-CFG face-box correction. Calling BA the
  "identity owner" therefore overstates what the code implements.

The decoded causal objective also failed its central test. At the end of all
three runs, `correct_gain` is approximately zero; correct and wrong memories do
not causally move decoded identity toward their corresponding references.

N38 is the least-bad member of this family because it has the best 8k ID score
and the simplest memory, but its advantage over N36 is negligible. N37's extra
canonical-part memory did not survive the downstream 16-site bottleneck and
performed worst.

## Scope

Analyzed:

- `full_validation_results/ba_identity_owner_qformer_2gpu_N36`
- `full_validation_results/ba_identity_owner_hybrid_2gpu_N37`
- `full_validation_results/ba_identity_owner_cropped_qformer_2gpu_N38`
- their saved training logs, Comet exports, and fixed full-validation metrics
- the fixed 96-image PhotoMaker baseline
- N31/N32/N33 only where needed to quantify the regression

N34 and N35 are known-problematic precursor runs. They had earlier schedule,
validation-transfer, and CFG-strength issues and should not be used as result
anchors. Per the requested scope, they were noted but not reanalyzed here.

## Results

### Fixed full-validation identity metric

PhotoMaker baseline mean ID similarity is `0.4886`.

| Run | 4k | 6k | 8k | 4k -> 8k |
|---|---:|---:|---:|---:|
| N36: full-reference QFormer | 0.4561 | 0.4552 | 0.4517 | -0.0044 |
| N37: QFormer + canonical parts | 0.4523 | unavailable | 0.4472 | -0.0051 |
| N38: cropped QFormer | 0.4562 | 0.4524 | 0.4530 | -0.0032 |

All scored checkpoints have a face-detection rate of `1.0`. The N37 6k image
folder contains 95 rather than 96 images and its metrics JSON correctly omits
that incomplete checkpoint. The missing file is `Dancing ma_eddie.png`.

The metric does not merely plateau; it slips slightly in every run. More
training is not moving the result toward the reference identity.

### Same-seed pixel deviation from PhotoMaker

For each generated image, mean absolute RGB difference was measured against the
same-seed PhotoMaker image. "Face MAE" uses the fixed
`pm96_bboxes_new.json` target face crop.

| Run/checkpoint | Images | Face MAE vs PM | Full-image MAE vs PM |
|---|---:|---:|---:|
| N36 4k | 96 | 0.03834 | 0.01949 |
| N36 6k | 96 | 0.03841 | 0.01945 |
| N36 8k | 96 | 0.03856 | 0.01945 |
| N37 4k | 96 | 0.04163 | 0.01965 |
| N37 6k | 95 | 0.04294 | 0.01960 |
| N37 8k | 96 | 0.04620 | 0.02009 |
| N38 4k | 96 | 0.03842 | 0.01946 |
| N38 6k | 96 | 0.03942 | 0.01952 |
| N38 8k | 96 | 0.03877 | 0.01946 |

For scale:

| Older run/checkpoint | Face MAE vs PM | Full-image MAE vs PM |
|---|---:|---:|
| N31 2k | 0.06683 | 0.02203 |
| N32 2k | 0.06784 | 0.02236 |
| N32 6k | 0.07494 | 0.02301 |
| N32 10k | 0.07351 | 0.02278 |
| N32 16k | 0.07763 | 0.02329 |
| N33 24k | 0.06653 | 0.02224 |

N36 and N38 produce only about half N32's face-region displacement. N37 is
slightly more active, but its 8k ID score is the worst. The latest runs are not
bitwise copies of PhotoMaker; their meaningful face intervention is simply much
smaller.

Checkpoint-to-checkpoint face MAE is around `0.023` for N36/N38 and `0.027` for
N37. This confirms that weights are changing without increasing the branch's
net authority over PhotoMaker.

## Runtime and training-path verification

The downloaded logs are from the corrected path:

- every validation reports
  `[BA Architecture] selected_processors=16 ...`;
- every validation reports
  `[BA Validation] copied trained processors 16/16`;
- the schedule is `NO_ID` at 0, `PHOTOMAKER` at 10, and `BOTH` at 15;
- runtime diagnostics report `applied_gain=5`, so the post-CFG correction
  includes the restored guidance multiplier;
- no evidence indicates an OOM or a completely detached optimizer in these
  downloaded trajectories.

The first validation-loader, equal-start schedule, and missing-guidance bugs
explain earlier N34-N38 observations, but they do not explain the current 4k-8k
results. These checkpoints still fail after those fixes.

Typical validation-time conditional corrections are only about
`0.0002-0.0006` mean absolute epsilon before the configured gain, with localized
maxima around `0.05-0.13`. This is consistent with a real but small residual.

## Why "identity owner" remains PhotoMaker-dominated

### 1. The layer allowlist removes most BA injection sites

N32's pre-N34 target residual is installed in all 70 SDXL `attn2` processors.
N36-N38 select:

- six `up_blocks.1` sites at gate `1.0`;
- ten `up_blocks.0.attentions.2` sites at gate `0.5`.

The branch therefore has 16 sites and 11 unit-gate equivalents. The remaining
54 sites are ordinary PhotoMaker cross-attention processors.

Relevant code:

- processor allowlist filtering:
  `src/model/photomaker_branched/branched_runtime.py:16`
- selected processor installation:
  `src/model/photomaker_branched/branched_runtime.py:214`
- trainable processor selection:
  `src/model/photomaker_branched/lora2_helpers.py:192`
- N36 allowlist/gates:
  `src/configs/one_id_ba_identity_owner_qformer_N36.yaml:15`

### 2. PhotoMaker is still the external prediction owner

The BA pass does attenuate PhotoMaker identity context at selected processors,
but the code separately computes a full PhotoMaker prediction. The final output
is:

`guided_PM + residual_scale * guidance_scale * bbox(BA_cond - PM_cond)`

This is a safe residual architecture, not exclusive BA identity ownership.
PhotoMaker remains the absolute baseline at every denoising step.

Relevant code:

- PM context attenuation inside selected processors:
  `src/model/photomaker_branched/attn_processor_cleanest.py:983`
- selected-site identity residual:
  `src/model/photomaker_branched/attn_processor_cleanest.py:1021`
- post-CFG composition:
  `src/model/photomaker_branched/branched_runtime.py:375`
- inference call site:
  `src/pipelines/br_pipeline_helpers.py:1113`

### 3. The gates are healthy but do not compensate

Final logged gate means:

| Run | min | mean | max |
|---|---:|---:|---:|
| N36 | 0.5037 | 0.6910 | 1.0060 |
| N37 | 0.4921 | 0.6812 | 0.9992 |
| N38 | 0.5049 | 0.6919 | 1.0047 |

These values are exactly what the six-at-1/ten-at-0.5 initialization predicts.
The gate implementation is working, but training does not learn substantially
more authority.

### 4. Memory capacity is not the active bottleneck

N36 and N38 use two QFormer tokens; N37 appends eight canonical tokens. N37
still performs worst. The downstream target K/V norms grow to `5.86-6.35`, so
the memories and K/V adapters are not frozen. The 16-site residual bottleneck
and ineffective identity objective dominate before the richer memory can help.

The cropped reference in N38 also gives no meaningful advantage over N36. This
rules out full-reference background leakage as the main cause of the flat
behavior.

## Causal-loss diagnosis

Final Comet values:

| Run | causal loss | correct gain | wrong gain | correct sim | wrong sim |
|---|---:|---:|---:|---:|---:|
| N36 | 0.2637 | -0.000036 | 0.000447 | 0.7655 | 0.8230 |
| N37 | 0.2529 | 0.000065 | 0.000369 | 0.7811 | 0.7690 |
| N38 | 0.2634 | -0.000066 | 0.000731 | 0.7655 | 0.8233 |

The intended margin is `0.05`, but measured gains remain approximately zero.
The loss remains active rather than converging. It does not demonstrate that
correct memory changes the generated identity toward the correct reference.

Two implementation/design details make this setup less clean than intended:

1. The decoded causal path runs only when sampled training timestep is at most
   300, about 30% of uniformly sampled batches. Its configured weight `0.5`
   therefore has a much smaller average influence over all optimizer steps.
2. `wrong_identity_pred` is also consumed unconditionally by the old
   epsilon-space `identity_dependence_ranking_loss` in
   `src/trainer/sdxl_trainers.py:477`. For decoded-causal runs this silently
   combines the new decoded objective with N31's old nuisance-prone epsilon
   ranking, using inherited `ba_identity_dependence_weight=0.25`. That is an
   unintended objective coupling and should be removed or explicitly toggled in
   a future implementation.

The decoded objective is not numerically dead: BA gradient norms are nonzero
and weights grow. It is semantically ineffective under this architecture.

## Per-run verdicts

### N36

- Clean, stable, and properly wired after the fixes.
- Face intervention is essentially flat from 4k to 8k.
- ID similarity declines by `0.0044`.
- Full-reference QFormer memory is not sufficient through the restricted route.

Verdict: failed identity-owner architecture; do not continue.

### N37

- Most visibly active of the three, but still substantially weaker than N32.
- Lowest 8k ID similarity (`0.4472`).
- Extra canonical tokens increase complexity without producing identity gain.
- The 6k artifact is incomplete (95 images), so use 4k/8k for comparisons.

Verdict: richer memory is not justified while the downstream route is weak; do
not continue.

### N38

- Best 8k score among the three (`0.4530`), but only marginally.
- Face MAE is effectively the same as N36.
- Bbox-normalized QFormer preprocessing does not solve the problem.

Verdict: best comparison representative of N36-N38, not a successful run.

## Decision

Do not tune or extend N36-N38. Return to the pre-N34 N32 generation topology to
re-establish an active branch:

- all 70 target-face cross-attention residual sites;
- unit legacy residual gates;
- legacy pre-CFG hard epsilon merge;
- no local PhotoMaker identity-context attenuation;
- no post-CFG composition;
- no decoded-causal/epsilon-ranking objective coupling.

Only after that anchor reproduces N32-level face movement should one isolated
identity-memory improvement be tested. The accompanying N39/N40 proposal uses
exactly this strategy.

