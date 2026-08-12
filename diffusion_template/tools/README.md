# Utilities

- `reports/publish_report.py` — render an analysis Markdown report to PDF and
  optionally upload it to Dropbox, in one call. This is the standard pipeline
  for every findings report; pair it with the `research-report` skill at
  [`.claude/skills/research-report/SKILL.md`](../../.claude/skills/research-report/SKILL.md),
  which defines the report structure the PDF is expected to follow.
- `comet/` — automatically record Comet experiment IDs, retrieve metrics and
  images by ID, selectively repair subject-v2 validation from saved
  checkpoints, export runs, and build PDF reports. See
  [`comet/README.md`](comet/README.md).
- `inference/` — calculate metrics and build image/metric PDFs.
- `datasets/` — prepare validation embeddings and the Cosmic Large one-ID
  dataset.

Run these tools from `diffusion_template` unless a tool's help text says
otherwise.
