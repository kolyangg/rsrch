# PhotoMaker branched-attention training

This is the primary project in the `test` worktree. It preserves the April
2026 RHCA model/runtime while keeping current one-identity replay configs and
launchers easy to find.

## Main entry points

- `train.py` — Hydra training entry point.
- `infer.py` — inference entry point.
- `src/` — model, pipeline, trainer, datasets, metrics, and Hydra configs.
- `bbox_utils/` — active face-box utilities and detector checkpoint.
- `setup/` — reproducible environment snapshots and helpers.

Run Python and Hydra commands from this directory so relative dataset and
output paths resolve consistently.

## Active launchers

```bash
conda activate photomaker

# One-time machine-local setup (the resulting .env is ignored by Git).
cp .env.example .env
chmod 600 .env
# Edit .env and set COMET_API_KEY, PM_PATH, and any optional overrides.

# Historical April one-ID replay: 4k steps, validation every 500.
bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh

# Same RHCA architecture on cosmic_large_one_id.
bash launchers/active/run_rhca_apr2026_cosmic_large_one_id_1gpu.sh
```

Both active launchers automatically load and export values from
`diffusion_template/.env`; shell-level `export` commands are not required.
Set `ENV_FILE=/another/path/.env` only when a server needs a different file.

The default Comet project is `rsrch-jul`. Override names without editing a
launcher. Values passed on the command line take precedence unless the same
variable is explicitly assigned in `.env`:

```bash
RUN_NAME=my_run COMET_PROJECT=rsrch-jul \
  bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh
```

## Supporting material

- [`docs/`](docs/) — architecture and experiment documentation.
- [`tools/`](tools/) — Comet, inference/reporting, and dataset utilities.
- [`requirements/`](requirements/) — supplemental environment requirements.
- [`artifacts/reference_debug/`](artifacts/reference_debug/) — retained debug
  examples, not runtime output.
- [`archive/`](archive/) — historical entry points and integration examples.
- [`launchers/archive/`](launchers/archive/) — historical SLURM launchers.

Generated checkpoints, logs, Hydra outputs, and new debug images should remain
untracked.
