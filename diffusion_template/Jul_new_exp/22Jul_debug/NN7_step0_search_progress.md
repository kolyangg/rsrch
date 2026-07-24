# NN7 step-zero search progress

## 2026-07-22 — execution started

- Approved scope: autonomous step-zero experimentation, writing only below
  `Jul_new_exp/22Jul_debug/`.
- Environment preflight: `photomaker_NS`, NVIDIA H100 80 GB, approximately
  81 GB initially free.
- Notebook preflight: Python syntax valid; added the missing `json` import.
- Added `run_search_notebook.py` to run selected notebook experiments with
  environment-controlled sample IDs, experiment shards, schedules, and unique
  run IDs.
- Next gate: one-sample N3a/core-ring/dual smoke test, followed by the complete
  four-sample quick ladder if artifact and metric integrity checks pass.

### Smoke attempt 1

- Stopped before image generation because the notebook passed the stale
  constructor key `model.ba_collect_aux_losses`.
- The active model derives auxiliary-loss collection from its three auxiliary
  loss weights and has no constructor argument by that name.
- Removed only the stale notebook override; core code remains unchanged.

### Smoke attempt 2

- PM0 and exact N3a completed. Exact N3a behaved as the intended unsafe
  positive control on sample 0: face MAE `0.1363`, outside MAE `0.0367`,
  landmark displacement `0.1681`, and reference gain `-0.0609`. The contact
  sheet shows severe eye/face displacement and loss of lower-face anatomy.
- Repaired legacy candidates were rejected during construction because the
  production model limits `base_outside_core` to packed-residual variants,
  although the shared branched runtime implements the anchor generically.
- Added an experiment-local instantiation shim: construct legacy N3a with
  `anchor=none`, then apply the requested runtime anchor in memory before
  pipeline creation. No production guard or source file was changed.

## Initial four-case quick ladder

- Completed all 11 planned controls in two parallel H100 workers; core-code
  fingerprints remained unchanged.
- `n3a_roi_up_confidence50_anchor` is the only numerically promising candidate
  that also looks anatomically coherent in all four contact-sheet rows:
  face MAE `0.03337`, reference gain `+0.01594` on 3/4 samples, landmark
  displacement `0.00744`, bbox IoU `0.91825`, outside MAE `0.01012`.
- `n3a_fullgrid_up_core_ring_anchor` passed the automatic screen but is
  visually rejected: repeated missing mouths, crossed/displaced eyes, and
  pasted frontal geometry.
- `n3a_roi_up_dual75_anchor` had strong positive recognizer gain (`+0.10798`,
  4/4 positive) but the same severe anatomy failures and slightly excessive
  outside-face change.
- NN7a_init v1/v2 remained close to PhotoMaker and had negative identity gain;
  the strong v2 control increased face change while making identity direction
  worse (`-0.03043`, 0/4 positive).
- Next batch is adaptive and single-axis: confidence gain/timing/ROI size,
  lower dual ownership, and a smaller full-grid core.

## Adaptive four-case ablations

- Completed 10 single-axis ablations in three parallel workers; protected core
  fingerprints remained unchanged.
- Confidence gain `0.50` remains the best safety/identity balance. Gain `0.25`
  was too weak (`+0.00074` mean reference gain), while gain `0.75` and ROI-grid
  changes were less consistent (only 2/4 or fewer positive cases).
- Starting BA at step 8 weakened the effect. The attempted step-4 start was an
  invalid schedule collision with PhotoMaker's step-4 switch and produced the
  exact PM baseline; a true early test must start at step 5.
- Dual reference ownership at `0.35` was much safer than `0.75` and produced
  strong identity metrics (`+0.05572`, 4/4 positive), but visual review still
  found systematic frontalization of head pose and expression.
- Full-grid core ratio `0.50` preserved anatomy better than the original
  repaired full-grid candidate and scored `+0.05340` mean reference gain, but
  visual review still shows gaze/expression and facial-geometry transfer.
- Final refinement is deliberately narrow: confidence gains `0.40/0.55`, true
  early step 5, dual ownership `0.20/0.25/0.30`, and full-grid core `0.35`.

## Narrow refinement and promotion decision

- The true step-5 confidence schedule is the safety winner: median reference
  gain `+0.03011` (3/4 positive), median face MAE `0.03268`, landmark movement
  `0.00328`, bbox IoU `0.97124`, and outside MAE `0.01058`.
- Dual ownership `0.25` is the stronger finalist: median reference gain
  `+0.03508` (4/4 positive), face MAE `0.06440`, landmark movement `0.01487`,
  bbox IoU `0.92669`, and outside MAE `0.01183`. It is anatomically coherent,
  but visually transfers more facial pose/expression than confidence residual.
- Confidence gains `0.40/0.55`, dual `0.20`, and full-grid core `0.35` did not
  improve consistency. Dual `0.30` was stronger but moved geometry further.
- Promoted exactly two architectures to 50 steps. The fixed confirmation
  schedule is PhotoMaker step 10 and BA step 15. Each is also tested with
  PhotoMaker conditioned on Eddie but the BA reference replaced by Elon, so
  direction toward the branch reference can be measured directly.

## Initial 50-step gate

- Neither promoted architecture passed. Confidence gain `0.50` lost identity
  improvement (median `-0.00395`, 1/4 positive) and exceeded outside-face MAE
  (`0.01725`). Dual `0.25` retained some gain (`+0.02024`, 3/4 positive) but
  was still more globally destructive (`0.01846` outside MAE).
- Contact sheets confirm the numerical failure: especially on sample 1, gaze,
  head orientation, and scene structure shift substantially at 50 steps.
- The confidence wrong-reference test did not move causally toward Elon:
  median Elon gain was `-0.00667`; one case also had severe landmark movement.
- Because dual failed the same-reference safety gate, its wrong-reference run
  is cancelled. The only remaining rescue directly compensates for the longer
  BA-active window: confidence gains `0.20/0.25`, or gain `0.50` starting at
  step 20. No broader architecture search is reopened.

## Search closed

- All three 50-step rescues failed. Gain `0.20` had median reference gain
  `-0.03062` and outside MAE `0.01749`; gain `0.25` had `-0.01945` and
  `0.01840`; gain `0.50` delayed to step 20 had `-0.00090` and `0.01632`.
- Per-sample review confirmed that the aggregate failures are real: sample 1
  remains sensitive to geometry changes, while sample 3 shows broad scene and
  subject differences.
- No candidate passed the 50-step same-reference and wrong-reference gates, so
  the 24-case matrix and production training config were correctly not run.
- Final report: `2026-07-22_NN7_step0_architecture_search_report.md`.
