# NN7 step-zero branched-attention architecture search report

**Date:** 2026-07-22  
**Repository commit:** `1e88825dc4a325ea1e146be2fa519801f048a73e` (`main_clean`)  
**Environment:** `photomaker_NS`, NVIDIA H100 80 GB, RealVisXL V4.0, BF16  
**Result:** no untrained configuration passed the complete 50-step promotion gate

## Executive conclusion

The search found a useful middle region at 20 inference steps, but it did not
survive the required 50-step confirmation.

- At 20 steps, ROI-normalized, up-block-only, CA-off branched self-attention
  with a PhotoMaker output anchor was substantially safer than exact N3a.
- The best safety configuration was confidence-residual gain `0.50`, starting
  immediately after PhotoMaker at step 5. It improved reference similarity on
  3/4 cases while keeping geometry close to PhotoMaker.
- Dual reference ownership `0.25` was stronger and improved identity on 4/4
  cases, but transferred more reference pose/expression.
- At 50 steps, both promoted configurations changed the scene too much. The
  confidence candidate also lost identity direction; dual retained some gain
  but exceeded the outside-face safety threshold.
- Lower confidence gains and a later BA start did not rescue the 50-step
  behavior. A wrong-reference Eddie-to-Elon test did not show causal movement
  toward Elon.

Therefore no configuration was advanced to the deterministic 24-case matrix
or training YAML. Doing so would violate the prescribed promotion gate.

## What was run

The automated harness executed 34 distinct four-case configurations, plus
pre-generation smoke checks. It regenerated exact per-worker PhotoMaker
baselines with the same prompts, seeds, bboxes, precision, and schedule.

The search covered:

- historical controls: exact N3a, N3a without CA, NN7a init v1/v2;
- reference token layout: full-grid versus normalized ROI grids `6/8/12`;
- layer scope: all self-attention versus up blocks;
- face ownership: reference-only, core-ring, dual, confidence residual;
- dual ownership: `0.20/0.25/0.30/0.35/0.50/0.65/0.75`;
- confidence gain: `0.20/0.25/0.40/0.50/0.55/0.75`;
- core ratios: `0.35/0.50/0.68`;
- 20-step BA starts: steps `5/6/8`;
- 50-step BA starts: steps `15/20`;
- 50-step same-reference confirmation and an Eddie-to-Elon branch-reference
  control.

All generated data and metrics are indexed by the deterministic leaderboard:

- `experiments/leaderboard_latest_four_case.csv` — latest immutable result for
  every distinct four-case experiment;
- `experiments/leaderboard_all.csv` — every completed summary, including
  repeated attempts;
- `experiments/registry.jsonl` — append-only notebook registry.

## Top 20-step screens

| Experiment | Face MAE | Reference gain | Positive | Landmark | Bbox IoU | Outside MAE | Visual assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| `n3a_roi_up_confidence50_early25_anchor` | 0.03268 | +0.03011 | 3/4 | 0.00328 | 0.97124 | 0.01058 | Best safety/identity balance; coherent, target pose preserved |
| `n3a_roi_up_dual25_anchor` | 0.06440 | +0.03508 | 4/4 | 0.01487 | 0.92669 | Stronger identity; visible pose/expression pull |
| `n3a_roi_up_dual30_anchor` | 0.07973 | +0.03933 | 4/4 | 0.02037 | 0.90677 | Strongest of the three, but more geometry transfer |

All three detected a face in 4/4 cases and passed the numerical 20-step screen.
Only the first two were promoted because dual `0.30` was already a strictly
less safe point on the same ownership axis.

Relevant artifacts:

- confidence step-5:
  `experiments/20260722T222014_264834Z__refineB__n3a_roi_up_confidence50_early25_anchor/`
- dual `0.25`:
  `experiments/20260722T222014_289597Z__refineC__n3a_roi_up_dual25_anchor/`
- dual `0.30`:
  `experiments/20260722T222014_289597Z__refineC__n3a_roi_up_dual30_anchor/`

## Important rejected 20-step controls

- Exact N3a was the intended unsafe positive control: severe displaced eyes,
  lower-face loss, face MAE `0.13467`, landmark displacement `0.10911`, and
  outside MAE `0.03235`.
- Repaired full-grid core-ring produced strong automatic scores but repeatedly
  pasted frontal reference geometry, crossed/displaced eyes, or erased mouths.
- Full-grid core `0.50` kept anatomy intact but still transferred gaze,
  expression, and face layout. Core `0.35` reduced activity without making
  identity direction consistent.
- Dual `0.75` produced strong recognizer gain but severe anatomy failure.
  Reducing dual ownership monotonically improved safety; `0.25` was the useful
  boundary, not a complete solution.
- NN7a init v1/v2 remained close to PhotoMaker and did not improve identity.
  Increasing its initial strength changed the face while moving identity in
  the wrong direction.
- Confidence gain was not monotonic: `0.25` was mostly too weak, `0.75` was
  inconsistent, and `0.40/0.55` did not improve on `0.50`.
- A requested BA start at step 4 collided with the PhotoMaker switch in the
  pipeline's mutually exclusive switch logic and produced exact PM output.
  The valid early test was step 5.

## 50-step promotion gate

Fixed schedule: 50 inference steps, PhotoMaker step 10, BA step 15.

| Experiment | Face MAE | Reference gain | Positive | Landmark | Bbox IoU | Outside MAE | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `promote50_confidence50_standard` | 0.06698 | -0.00395 | 1/4 | 0.00834 | 0.94332 | 0.01725 | Fail: identity and outside-face |
| `promote50_dual25_standard` | 0.09628 | +0.02024 | 3/4 | 0.02295 | 0.89682 | 0.01846 | Fail: outside-face and visual pose transfer |
| `rescue50_confidence20_standard` | 0.07540 | -0.03062 | 1/4 | 0.01636 | 0.94932 | 0.01749 | Fail |
| `rescue50_confidence25_standard` | 0.07589 | -0.01945 | 1/4 | 0.01612 | 0.94545 | 0.01840 | Fail |
| `rescue50_confidence50_late20` | 0.05003 | -0.00090 | 2/4 | 0.00853 | 0.95492 | 0.01632 | Fail |

Face detection was 4/4 for every row, but all exceeded the required outside
MAE `0.015`. Contact sheets show that the excess is meaningful, not metric
noise: subject orientation, gaze, upper body, and high-contrast scene edges
move. Sample 1 is the clearest geometry failure; sample 3 has widespread
outside-face differences.

Relevant artifacts:

- confidence50 confirmation:
  `experiments/20260722T222903_792409Z__promote_conf__promote50_confidence50_standard/`
- dual25 confirmation:
  `experiments/20260722T222903_791169Z__promote_dual__promote50_dual25_standard/`
- best safety rescue (late step 20):
  `experiments/20260722T223525_087788Z__rescueLate__rescue50_confidence50_late20/`

## Wrong-reference causality

`wrongref50_confidence50_elon12` kept the normal PhotoMaker input, prompt,
seed, and target bboxes from Eddie samples 0–3. Only branched attention received
the Elon reference from manual-validation sample 12.

- Median similarity gain toward Eddie: `+0.00433`.
- Median similarity gain toward the actual BA reference, Elon: `-0.00667`.
- Per-sample Elon gains: `-0.01872`, `+0.01221`, `-0.07162`, `+0.00538`.
- Outside MAE: `0.01780`; sample 1 landmark displacement: `0.13260`.

This is not evidence of reliable reference-causal identity transfer. The dual
wrong-reference run was cancelled because dual had already failed its
same-reference 50-step safety gate.

Artifacts:

`experiments/20260722T222903_790164Z__wrong_conf__wrongref50_confidence50_elon12/`

## Technical interpretation

The experiments support four conclusions.

1. Normalized ROI, up-only routing, CA-off operation, target fallback, and a
   protected output are all valuable repairs to N3a. They eliminate the worst
   step-zero anatomy collapse at 20 steps.
2. Untrained target-Q to spatial-reference-K/V attention still lacks reliable
   correspondence. Increasing ownership transfers reference layout as well as
   identity; decreasing ownership can make the perturbation non-causal rather
   than safely identity-directed.
3. Authority is exposure-dependent and nonlinear. A configuration selected at
   20 steps cannot be assumed to extrapolate to 50 steps, and simply scaling
   down the residual gain did not reproduce the 20-step safe region.
4. Recognizer gain alone is insufficient. Several full-grid and dual settings
   scored strongly while visibly changing gaze, expression, anatomy, or scene.

## Recommended next direction

Do not train any configuration from this search unchanged. The highest-value
next experiment is a correspondence-aware, exposure-normalized variant that
keeps the successful repair envelope:

```text
target-coordinate Q -> normalized reference-face K/V
up blocks only
branched CA off
PhotoMaker fallback / protected output
```

Add two controls before another broad sweep:

1. Normalize accumulated branch authority by the number or integral of active
   denoising steps, rather than using the same per-step gain at 20 and 50
   steps. Verify that matched effective exposure produces similar outputs.
2. Gate or align reference tokens with explicit local correspondence (for
   example landmark-relative/local-window matching) and require an Eddie-to-
   Elon wrong-reference directional test at the four-case stage, before
   promotion.

The 20-step confidence step-5 recipe is useful as a diagnostic starting point,
not as a training recommendation. Dual `0.25` is a useful strong comparison.

## Reproducibility and scope audit

- `run_search_notebook.py` provides unattended cell execution and experiment
  sharding through environment variables.
- `adaptive_specs.json`, `refinement_specs.json`, `promotion_specs.json`, and
  `rescue50_specs.json` contain all added recipes.
- `rebuild_leaderboard.py` deterministically rebuilds the aggregate tables from
  immutable `metrics_summary.json` files.
- The notebook records compact configs, architecture signatures, hashes,
  per-sample metrics, contact sheets, and self-contained image bundles.
- The wrong-reference extension stores target and BA reference images
  separately and records similarity movement toward each.
- Every completed notebook run re-hashed 220 protected repository files. No
  protected source, training file, or repository config changed.
- All authored files and generated artifacts stayed under
  `Jul_new_exp/22Jul_debug/`.

## Complete experiment inventory

```text
n3a_exact
n3a_no_ca
n3a_roi_all_reference
n3a_roi_up_reference_anchor
n3a_roi_up_core_ring_anchor
n3a_roi_up_dual75_anchor
n3a_roi_up_confidence50_anchor
n3a_fullgrid_up_core_ring_anchor
nn7a_init_v1_default
nn7a_init_v2_default
nn7a_init_v2_strong25
n3a_roi_up_confidence25_anchor
n3a_roi_up_confidence75_anchor
n3a_roi_up_confidence50_early20_anchor
n3a_roi_up_confidence50_late40_anchor
n3a_roi_up_dual35_anchor
n3a_roi_up_dual50_anchor
n3a_roi_up_dual65_anchor
n3a_fullgrid_up_core50_anchor
n3a_roi6_up_confidence50_anchor
n3a_roi12_up_confidence50_anchor
n3a_roi_up_confidence40_anchor
n3a_roi_up_confidence55_anchor
n3a_roi_up_confidence50_early25_anchor
n3a_roi_up_dual20_anchor
n3a_roi_up_dual25_anchor
n3a_roi_up_dual30_anchor
n3a_fullgrid_up_core35_anchor
promote50_confidence50_standard
promote50_dual25_standard
wrongref50_confidence50_elon12
rescue50_confidence20_standard
rescue50_confidence25_standard
rescue50_confidence50_late20
```
