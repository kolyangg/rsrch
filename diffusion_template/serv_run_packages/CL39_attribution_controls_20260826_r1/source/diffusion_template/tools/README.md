# Utilities retained by clean_full

- `validate_clean_full_config.py` — supported-config manifest gate and run
  record generator.
- `datasets/` — dataset preflights invoked by the unified launcher.
- `comet/` — immutable-key retrieval, deferred face-quality finalization, and
  report export/rendering.
- `inference/calculate_face_quality_metrics.py` — canonical face-crop IQA
  scorer.
- `reports/publish_report.py` and `dropbox/upload_to_dropbox.py` — standard
  Markdown-to-PDF and Dropbox publishing path.
- `verify_serv_source_manifest.py` — immutable Serv source verification.

Run these tools from `diffusion_template/`.
