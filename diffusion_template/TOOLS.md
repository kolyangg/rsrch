# Project tools

This is the entry point for operational and experiment utilities. Run project
commands from `diffusion_template/` unless a linked guide says otherwise.

## Experiment tracking and reports

- [Comet experiment records and retrieval](tools/comet/README.md) —
  automatically records every new Comet experiment key beside its saved run,
  then retrieves metrics and validation images by that immutable key.
- [General tool directory](tools/README.md) — Comet report generation,
  checkpoint inference/evaluation, and dataset preparation.

For every new Comet experiment, verify
`saved/<run_name>/comet_experiment.json` during startup. Use
`tools/comet/comet_experiment.py` and the recorded key for later retrieval;
never select a run only by its display name.

## Server operations

- [Neb operations](LOCAL_NEB_SERVER_OPERATIONS.md) — SSH, synchronization,
  process/GPU checks, and safe job operation on Neb.
- [Serv/MLS operations](../local_scripts/serv_instructions.MD) — MLS job
  submission, inspection, stopping, and per-job JSON audit records.
- [Serv run-package builder](../local_scripts/serv_run_builder/README.MD) —
  standardized launcher/YAML packaging for Serv.

## Common experiment utilities

- `tools/inference/evaluate_rhca_checkpoint.py` — deterministic fixed-checkpoint
  diagnostic evaluation.
- `tools/inference/calc_metrics.py` — local validation metrics.
- `tools/datasets/` — controlled datasets, bounding boxes, and PhotoMaker
  validation preparation.
- `tools/datasets/preflight_cosmic_large_adapted.py` — decodes a deterministic
  sample of full-Cosmic target/reference pairs and records path, bbox, prompt,
  face-area, and cache-key integrity before a run starts.
