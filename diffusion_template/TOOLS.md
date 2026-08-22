# Clean E13-family tool index

Run commands from `diffusion_template/`. These utilities are deliberately
separate from the model and pipeline; retaining them does not add experimental
architecture switches.

## Session and implementation entry points

- `docs/handoffs/LATEST.md` — required new-session handoff.
- `docs/architecture/2026-08-10_e13_family_clean_implementation.md` — complete
  implementation ledger, verification evidence, limitations, and Serv runbook.
- `docs/architecture/2026-08-12_cl18_cl19_cl20_clean_extension.md` — isolated
  CL18-CL20 deltas, source provenance, verification and server instructions.
- `docs/validation_protocol.md` — fixed-96 validation contract.
- `../.claude/skills/research-report/SKILL.md` — required house style for
  experiment analysis and reports.

## Pre-launch correctness gates

```bash
python tools/validate_e13_family_config.py
python tools/verify_cl14_generation_parity.py
python tools/validate_cl14_ca_config.py
python tools/validate_cl18_cl20_config.py --config-name <CL18|CL19|CL20_config>
python tools/validate_cl23_cl27_config.py --config-name <CL23|CL27|CL39_config>
bash -n launchers/active/run_e13_family_24k_1gpu.sh
bash -n launchers/serv/*.sh
```

- `validate_e13_family_config.py` composes all three leaves, validates the
  schedule/ownership/runtime contract, proves their shared output-affecting
  projection is identical, and checks each dataset leaf.
- `verify_cl14_generation_parity.py` compares the pipeline and denoising source
  against sealed CL14 hashes and byte-seals every fixed-96 prompt, identity,
  reference, and bbox input.
- `validate_cl14_ca_config.py` checks the isolated residual identity-CA delta.
- `validate_cl18_cl20_config.py` checks each named arm against the sealed CL14
  schedule, validation and trainable contract.
- `validate_cl23_cl27_config.py` checks each named frequency/null-key arm and
  the optimized processor-lookup contract.
- `launchers/active/run_e13_family_24k_1gpu.sh` is the only training launcher
  for all ten supported recipes.
- `serv_run_packages/README.md` is the concise architecture matrix and points
  to the exact ten one-A100 Serv YAMLs.
- `serv_run_packages/e13_family_1gpu.yaml.example` remains the generic template
  for a deliberately new run identity.

## Dataset preflights and measurement

- `tools/datasets/preflight_large_dataset.py` — manifest structure, distinct
  same-ID pairing, bbox/decode, and sample audit for E13.
- `tools/datasets/preflight_big_celebs.py` — sealed release hash/readiness,
  identity cardinality, strict fields, trigger, bbox, and decode checks.
- `tools/datasets/preflight_cosmic_large_adapted.py` — Cosmic filtering,
  prompts, target/reference geometry, cache-key, and decode sample audit.
- `tools/datasets/preflight_cosmic_cl.py` — authoritative configured Cosmic
  recipe decode gate for the 1024 canvas, face-area band, caption budget, and
  CL18 distinct alternate reference.
- `tools/datasets/build_cl20_hardcase_schedule.py` — deterministic sealed
  48k-row Cosmic/BigCelebs curriculum builder.
- `tools/datasets/preflight_cl20_curriculum.py` — verifies CL20 phase counts
  and decodes schedule boundary rows through the configured loader.
- `tools/datasets/measure_face_body_alignment.py` — detected face versus the
  fixed generated bbox: center offset, size ratio, and IoU. A size ratio below
  0.8 is the historical undersized-face threshold.

Read each tool's `--help` before use. Dataset paths and expected hashes belong
in `.env` or explicit command arguments, never in committed files.

## Immutable Comet records and retrieval

Every online run must create `saved/<run_name>/comet_experiment.json` before
training proceeds. Never retrieve a run by mutable display name alone.

```bash
python tools/comet/comet_experiment.py show \
  saved/<run_name>/comet_experiment.json

python tools/comet/comet_experiment.py fetch \
  --record saved/<run_name>/comet_experiment.json \
  --step-number 4000 \
  --output-dir /path/to/step_4000
```

- `tools/comet/comet_experiment.py` — show/pull/fetch by immutable key.
- `tools/comet/export_comet_runs.py` — fail-closed multi-run export.
- `tools/comet/backfill_face_quality_metrics.py` — exact-step image retrieval,
  scoring, compact metric backfill, and per-image CSV asset.
- `tools/comet/finalize_deferred_face_quality.py` — validates all staged steps
  and runs scoring only after successful training.

Give each fetched step its own output directory; otherwise successive fetches
can overwrite images.

## Face-quality scoring

- `tools/inference/calculate_face_quality_metrics.py` — canonical no-reference
  face-crop scorer for TOPIQ-Face, TOPIQ, MUSIQ, and MANIQA-PIPAL.
- `src/metrics/face_quality_validation.py` — stages validation images during
  training without importing PyIQA when execution mode is `deferred`.

The scorer environment must use PyIQA 0.1.15. The E13-family launcher expects
`FACE_QUALITY_SCORER_PYTHON` in the ignored `.env` and runs the finalizer only
after Accelerate exits successfully.

## Research reports and Dropbox

The `research-report` skill is mandatory for experiment analyses. Publish a
Markdown report with:

```bash
python tools/reports/publish_report.py \
  analysis/<YYYY-MM-DD>_<slug>.md --upload
```

- `tools/reports/publish_report.py` renders the PDF into `analysis/assets/` and
  delegates upload.
- `tools/dropbox/upload_to_dropbox.py` uploads to
  `/rsrch/YYYY-MM-DD/<filename>`, verifies Dropbox's content hash, and requires
  a temporary direct-download link.

An upload is incomplete without the exact temporary link printed by the tool;
the link expires in approximately four hours. Credentials remain in `.env`.

## Operational boundary

Neb is unavailable and must not be accessed. For Serv, follow the implementation
runbook, inspect Running and Pending MLS jobs before submission, respect the
normal six-A100 project ceiling, use the existing `photomaker_NS` environment,
and do not launch with ad-hoc Hydra overrides.
