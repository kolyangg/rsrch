# Comet tools retained by clean_full

The unified launcher creates `saved/<run_name>/comet_experiment.json` before
training. `CometMLWriter` preserves its `plan` and atomically adds the live
immutable experiment key, URL, host, and Git metadata. A display name is never
accepted as a substitute for the key.

Verify or retrieve a run with:

```bash
python tools/comet/comet_experiment.py show saved/<run_name>/comet_experiment.json
python tools/comet/comet_experiment.py fetch \
  --record saved/<run_name>/comet_experiment.json --step-number 24000
```

The training launcher invokes `finalize_deferred_face_quality.py` only after
Accelerate exits successfully. That tool joins the exact staged validation
manifest, calls `calculate_face_quality_metrics.py`, and writes the compact
seven-curve metrics plus the per-image CSV through
`backfill_face_quality_metrics.py`.

For analysis, `export_comet_runs.py` exports immutable-key-selected data and
`build_comet_report_pdf.py` renders configured image/metric comparisons.
`comet_pdf_config_template.json` documents the report schema, including the
optional fixed-bbox face-closeup pages.

Neb is unavailable. Retrieve locally or through an approved Serv checkout;
credentials remain in `.env` and must never appear in records or logs.
