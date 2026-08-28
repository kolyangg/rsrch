# clean_full tools

Run commands from `diffusion_template/` so Hydra and dataset-relative paths
resolve consistently.

## Training

- `tools/validate_clean_full_config.py` is the fail-closed allowlist and
  composition gate. Use `--list` to enumerate supported configs.
- `launchers/active/run_clean_full_config_1gpu.sh` is the only supported Serv
  training/validation entry point. The config selects scientific behavior;
  the run name selects only the output and Comet label.
- `tools/verify_serv_source_manifest.py` verifies a packaged source snapshot
  before a remote launch.
- `tools/datasets/preflight_cosmic_cl.py`, `preflight_large_dataset.py`,
  `preflight_big_celebs.py`, and `preflight_bc_e13_dataset_schedule.py` are
  selected by the unified launcher from the resolved dataset name.

Every Comet run must create `saved/<run_name>/comet_experiment.json` and the
launcher must observe a 32-character immutable experiment key before treating
startup as successful.

## Comet and reports

- `tools/comet/comet_experiment.py` retrieves runs by immutable key.
- `tools/comet/export_comet_runs.py` and
  `tools/comet/build_comet_report_pdf.py` build comparison exports/reports.
- `tools/comet/finalize_deferred_face_quality.py` scores the staged fixed-96
  panel after training; it calls `tools/inference/calculate_face_quality_metrics.py`
  and `tools/comet/backfill_face_quality_metrics.py`.
- `tools/reports/publish_report.py <report.md> --upload` renders a report under
  `analysis/assets/` and uploads it through
  `tools/dropbox/upload_to_dropbox.py`.

Use the repository `research-report` skill for analyses and PDFs. Credentials
belong only in the gitignored `.env` file.

Neb remains unavailable; do not use it for launches, retrieval, or proxying.
