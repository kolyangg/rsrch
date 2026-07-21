# NN4 compact results bundle

This directory preserves the key evidence used in
`../2026-07-21_NN4_results_analysis_and_next_architecture.md` without copying
the approximately 4.8 GB of full generated images, crops, and heatmaps.

## Top-level summaries

- `NN4_causal_summary.csv`: four-way comparison of the 2k/4k RealVis and
  same-SDXL causal tests.
- `NN4_normal_validation_curve.csv`: Comet validation metrics through 8k.
- `NN4_training_window_summary.csv`: windowed training diagnostics through the
  last exported point at step 9,800.

## `causal_tests/`

Each of the four test directories contains:

- the run manifest and integrity hashes;
- directional identity, effect-decomposition, per-image, and paired metrics;
- exact tensor diagnostics;
- four representative 6-sample contact sheets spanning the 96-image panel.

The complete visual outputs remain under `../../rsrch_21Jul_test/`.

## `comet_training/`

- `metrics_history.json`: full exported scalar histories;
- `metrics_summary.json`: Comet extrema/current-value summary;
- `comet_output.log`: training and validation log, including PPR site
  diagnostics and checkpoint messages.

The source Comet export remains under `../../comet_data/metrics_only_NN4/`.
