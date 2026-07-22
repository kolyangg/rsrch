# NN5a 4k compact results bundle

This directory preserves the key evidence used in
`../2026-07-22_NN5a_4k_results_analysis_and_next_training_recommendation.md`
without duplicating the full generated-image, face-crop, and heatmap tree.

## Top-level summaries

- `NN5a_normal_validation_curve.csv`: RealVis validation metrics at step 0,
  2k, and 4k.
- `NN5a_causal_summary.csv`: the decisive 4k, RealVis, residual-scale-1
  reference-versus-noise result.
- `NN5a_training_window_summary.csv`: selected Comet training diagnostics over
  the 0–2k and 2k–4k windows.
- `NN5a_tensor_stage_summary.csv`: reference-content and reference-noise
  sensitivities along the captured PPR path.

## `causal_test/`

This contains the 4k test manifest and integrity hashes; aggregate, per-image,
paired, and directional metrics; exact tensor diagnostics; and six selected
contact sheets spanning all eight validation identities. The complete visual
outputs remain at:

`../../rsrch_21Jul_test/ppr_NN5a_4000step_realvis_scale1_reference_vs_noise/`

The copied `conclusion.md` says “PPR 8k” in its generated title. That title is
stale metadata: the source directory, loaded checkpoint in `manifest.json`, and
Comet history all identify this as checkpoint epoch 2 / step 4,000.

## `comet_training/`

- `metrics_history.json`: complete exported scalar histories through 4k;
- `metrics_summary.json`: Comet extrema/current-value summary;
- `comet_output.log`: training, validation, processor, and checkpoint log;
- `comet_run_export.json`: run metadata and export manifest.

The source export remains under `../../comet_data/metrics_only_NN5a/`.
