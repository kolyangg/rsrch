# Project tools

This is the entry point for operational and experiment utilities. Run project
commands from `diffusion_template/` unless a linked guide says otherwise.

## New-session entry point

- [Current project handoff](docs/handoffs/LATEST.md) — required experiment
  history, current conclusions, architectural boundaries, machine caveats,
  immutable Comet IDs, and recommended next work.

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
2. For Neb, follow `LOCAL_NEB_SERVER_OPERATIONS.md` and launch the active
   shell script from the remote `diffusion_template/` checkout.
3. For Serv, build/deploy a package with
   `../local_scripts/serv_run_builder/create_serv_run.py`, then submit its YAML
   through `../local_scripts/serv_job.py submit`. This creates a local
   per-submission audit JSON under `../local_scripts/serv_job_records/`.
4. During startup, require `saved/<run_name>/comet_experiment.json`, then copy
   the live immutable Comet key and URL into the experiment JSON if the
   launcher did not already preserve the plan in the canonical runtime record.
5. Inspect jobs through the documented Neb process/GPU checks or
   `serv_job.py status|inspect`; preserve failed records.
6. Retrieve metrics and images by immutable key with
   `tools/comet/comet_experiment.py fetch`, then verify requested steps, image
   counts, warnings, hashes, and visual gates.

Typical checks from `diffusion_template/`:

```bash
test -s saved/<run_name>/comet_experiment.json
python tools/comet/comet_experiment.py show \
  saved/<run_name>/comet_experiment.json

python tools/comet/comet_experiment.py fetch \
  --host neb \
  --run-name <run_name> \
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

- [Neb operations](LOCAL_NEB_SERVER_OPERATIONS.md) — SSH, synchronization,
  process/GPU checks, and safe job operation on Neb.
- [Serv/MLS operations](../local_scripts/serv_instructions.MD) — MLS job
  submission, inspection, stopping, and per-job JSON audit records.
- [Serv run-package builder](../local_scripts/serv_run_builder/README.MD) —
  standardized launcher/YAML packaging for Serv.

## Common experiment utilities

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
