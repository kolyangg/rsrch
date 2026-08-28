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

`metric_reference_lines` can render a selected baseline as a horizontal line
instead of a one-point series. Each metric entry accepts `run_id`, plus optional
`label`, `color`, `linestyle`, `linewidth`, `decimals`, or an explicit `value`.
Without `value`, the selected run's last metric value is used.

`page_order` controls the complete top-level report sequence. Built-in section
names are `metric_charts`, `group_average_tables`, `hyperparameters`,
`full_images`, and `face_closeups`. Use `markdown` for every page in one Markdown
source, or `markdown:<name>` for pages marked with
`<!-- report-group: <name> -->`. The builder rejects duplicate or omitted
Markdown pages, so a misspelled group cannot silently disappear from a report.
The optional `flysheet` Markdown layout renders a section-divider page.

## Rebuild the PM0/CL14/CL19/CL23/CL27/CL39 report

The reusable JSON is
`comet_pdf_config_23Aug_PM0_CL14_CL19_CL23_CL27_CL39_faces.json`. Change its
`runs`, `max_columns`, and `face_closeups.enabled` fields for later comparisons;
the architecture narrative and code excerpts are maintained separately in
`comet_report_pages_PM0_CL14_CL19_CL23_CL27_CL39.md`.

From `diffusion_template/` with the `photomaker` environment active:

```bash
python tools/comet/build_comet_report_pdf.py \
  --config tools/comet/comet_pdf_config_23Aug_PM0_CL14_CL19_CL23_CL27_CL39_faces.json \
  --output output/pdf/comet_report_PM0_CL14_CL19_CL23_CL27_CL39_reordered_appendix_23Aug2026.pdf \
  --dpi 200 \
  --image-max-side 768
```

This config renders results and comparisons first, the fixed references/prompts
page fourth, both image-comparison blocks next, and all architecture/formula/code
pages after an Appendix flysheet. Architecture table rows and run labels are
plain JSON overrides; the equations and source excerpts remain in the Markdown
file so they can be audited independently.

The config reuses the immutable-key export cache under
`comet_data/23Aug_PM0_CL14_CL19_CL23_CL27_CL39_faces/`; rerun the exporter only
when the selected runs or steps change.

Neb is unavailable. Retrieve locally or through an approved Serv checkout;
credentials remain in `.env` and must never appear in records or logs.
