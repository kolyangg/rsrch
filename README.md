# Branched-attention research workspace

This checkout is the `test` worktree used for the April 2026 RHCA
reproduction and closely related one-identity experiments.

## Repository layout

- [`diffusion_template/`](diffusion_template/) — primary training and
  validation project.
- [`dataset_full/`](dataset_full/) — datasets and validation metadata. This
  stays beside `diffusion_template` because Hydra configs use
  `../dataset_full/...`.
- [`_other_models/`](_other_models/) — PhotoMaker, PuLID, PersonaGen, and the
  legacy cross-model comparison harness.

The `compare` symlink is retained only for compatibility with historical
runtime defaults. New commands and documentation should use
`_other_models/compare`.

## Active April RHCA experiments

Run commands from `diffusion_template`:

```bash
cd diffusion_template
conda activate photomaker

cp .env.example .env
chmod 600 .env
# Edit .env once with COMET_API_KEY and PM_PATH.

bash launchers/active/run_rhca_apr2026_one_id_1gpu.sh
bash launchers/active/run_rhca_apr2026_cosmic_large_one_id_1gpu.sh
```

Both launchers load credentials and machine-local paths automatically from
`diffusion_template/.env` and log to the `rsrch-jul` Comet project by default.
See the
[`diffusion_template` README](diffusion_template/README.md) for details.

## Worktrees

- `~/rsrch` normally checks out `main_clean`.
- `~/rsrch_apr_test` checks out `test`.

Pull and commit each branch from its own worktree. Do not pull `test` into the
`main_clean` worktree.
