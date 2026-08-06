# Project tools

This is the entry point for operational and experiment utilities. Run project
commands from `diffusion_template/` unless a linked guide says otherwise.

## New-session entry point

- [Current project handoff](docs/handoffs/LATEST.md) — required experiment
  history, current conclusions, architectural boundaries, machine caveats,
  immutable Comet IDs, and recommended next work.

## Neb outage — 3 August 2026

Neb is unavailable and must not be accessed or used, including as a Comet
download proxy. Use local credentials/tools or Serv when appropriate. If a
user requests Neb, obtain separate explicit confirmation that the machine is
working again before even testing connectivity. The linked Neb operations
guide is dormant historical guidance while this notice is in force.

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

## New experiment workflow

Every experiment needs one non-secret local JSON before submission. For
Cosmic Large, place durable plans under `experiments/cosmic_large_adaptation/`
or `experiment_specs/`; Serv packages may also carry a sealed copy. Record the
hypothesis, fixed controls, changed variables, dataset/reference policy,
architecture flags, launcher/package, machine, gates, and status.

1. Create a unique launcher/run name and its local JSON. Never reuse a
   contaminated saved directory or Comet identity.
2. Do not submit to Neb while the outage notice above is in force. A future
   Neb request requires separate user confirmation before a read-only
   connectivity check.
3. For Serv, build/deploy a package with
   `../local_scripts/serv_run_builder/create_serv_run.py`, then submit its YAML
   through `../local_scripts/serv_job.py submit`. This creates a local
   per-submission audit JSON under `../local_scripts/serv_job_records/`.
4. During startup, require `saved/<run_name>/comet_experiment.json`, then copy
   the live immutable Comet key and URL into the experiment JSON if the
   launcher did not already preserve the plan in the canonical runtime record.
5. Inspect Serv jobs with `serv_job.py status|inspect`; preserve failed
   records. Do not attempt Neb job inspection during the outage.
6. Retrieve metrics and images by immutable key with
   `tools/comet/comet_experiment.py fetch`, then verify requested steps, image
   counts, warnings, hashes, and visual gates.

Typical checks from `diffusion_template/`:

```bash
test -s saved/<run_name>/comet_experiment.json
python tools/comet/comet_experiment.py show \
  saved/<run_name>/comet_experiment.json

python tools/comet/comet_experiment.py fetch \
  --record saved/<run_name>/comet_experiment.json \
  --step-number 4000

python3 ../local_scripts/serv_job.py submit <REMOTE_YAML> \
  --comment "<experiment objective>"
python3 ../local_scripts/serv_job.py inspect <MLS_JOB_ID> --lines 40
```

For Serv retrieval, pass `--host serv`, the absolute remote
`diffusion_template` path, and the Serv `photomaker_NS` Python path as shown
in [the Comet tool guide](tools/comet/README.md). Credentials remain in
machine-local `.env` files and must never be printed or copied into JSON.

## Server operations

- [Neb operations](LOCAL_NEB_SERVER_OPERATIONS.md) — dormant historical
  guidance; do not use during the current outage.
- [Serv/MLS operations](../local_scripts/serv_instructions.MD) — MLS job
  submission, inspection, stopping, and per-job JSON audit records.
- [Serv run-package builder](../local_scripts/serv_run_builder/README.MD) —
  standardized launcher/YAML packaging for Serv.

### August Large Dataset hard-BA suite

- [Six-arm design](docs/experiments/2026-08-03_large_dataset_hard_ba_six_arm_design.md)
  — historical evidence, single-delta hypotheses, fixed controls, and decision
  gates.
- [r4/Serv recovery and E0 pair](docs/experiments/2026-08-04_large_dataset_r4_serv2gpu_recovery_and_e0.md)
  — recovered historical sources, the two-GPU causal audit, and the separate
  historical-observed/fixed-BA-only E0 packages.
- [Historical-E0 adapter analysis and E7-E10](analysis/2026-08-04_e0_historical_global_adapter_id_gain_and_next_experiments.md)
  — matched 8k identity evidence and four explicit generic/default adapter
  decompositions, with 20k one-GPU configs and Serv packages.
- [E11/E12 BA-capacity plan](docs/experiments/2026-08-04_e11_e12_large_ds_ba_capacity_plan.md)
  — implemented parameter-matched spatial-SA rank-128 and corrected hard
  identity-CA rank-256 experiments, with exact ownership and live Serv IDs.
- `tools/validate_aug_large_ds_config.py` — fail-closed composition/spec gate
  for E0 and the six single-delta configs.
- `launchers/active/run_E_large_ds_hard_v1_20k_1gpu.sh` — shared
  one-GPU launcher; generated MLS YAMLs are under `serv_run_packages/`.

E0-E12 have been submitted as documented in the linked reports. E11/E12 use a
named temporary eight-A100 exception and an isolated runtime; do not submit
duplicates. The ceiling returns to six after E11/E12 finish or are removed.

## Common experiment utilities

- [Dropbox uploader](tools/dropbox/upload_to_dropbox.py) — uploads one or more
  local files to `/rsrch/YYYY-MM-DD/<filename>`, verifies Dropbox's content
  hash, and requires a temporary direct-download link for every uploaded file.
  The caller must include each printed link in the user-facing reply and note
  its approximately four-hour expiry. Credentials are read from the gitignored
  `diffusion_template/.env`. Run from `diffusion_template/`:

  ```bash
  python tools/dropbox/upload_to_dropbox.py /path/to/file
  ```

- [Default validation protocol](docs/validation_protocol.md) — full-96,
  2,000-step cadence, default face-quality scoring, toggles, and Comet layout.
- `tools/inference/evaluate_rhca_checkpoint.py` — deterministic fixed-checkpoint
  diagnostic evaluation.
- `tools/inference/calc_metrics.py` — local validation metrics.
- `tools/inference/calculate_face_quality_metrics.py` — no-reference
  face-crop IQA scoring with TOPIQ-Face, TOPIQ, MUSIQ, and MANIQA-PIPAL.
- `tools/comet/backfill_face_quality_metrics.py` — exact-step Comet image
  download, offline face-quality scoring, compact seven-curve backfill,
  one API-only per-image CSV asset, fail-closed legacy-layout cleanup, and
  post-write verification.
- `tools/comet/download_face_quality_images.py` — download-only staging of
  exact Comet image steps with file-size and PIL verification.
- `tools/comet/build_full96_longitudinal_pdf.py` — fail-closed PDF comparison
  of the same full-96 samples across runs and steps, with per-image identity
  and face-quality annotations. Its YAML template is beside the script.
- `tools/comet/rebase_verify_face_quality_manifest.py` — rebase a transferred
  staging manifest and verify every image by size, SHA-256, and PIL decode.
- `tools/datasets/` — controlled datasets, bounding boxes, and PhotoMaker
  validation preparation.
- `tools/datasets/preflight_cosmic_large_adapted.py` — decodes a deterministic
  sample of full-Cosmic target/reference pairs and records path, bbox, prompt,
  face-area, and cache-key integrity before a run starts.
