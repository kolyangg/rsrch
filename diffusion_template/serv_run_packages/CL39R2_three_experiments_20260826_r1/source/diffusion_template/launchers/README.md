# clean_full launcher

`active/run_clean_full_config_1gpu.sh` is the only supported training entry
point on this branch. `CONFIG_NAME` selects the model, dataset, routing,
objective, validation, and trainable ownership. `RUN_NAME` is only an output
and Comet label; command-line Hydra overrides are rejected.

List and validate supported configs before submission:

```bash
python tools/validate_clean_full_config.py --list
python tools/validate_clean_full_config.py --config-name CL39_cosmic_null_key_confidence_router_24k
```

Launch from any directory after loading machine-local paths and credentials in
`diffusion_template/.env`:

```bash
CONFIG_NAME=CL39_cosmic_null_key_confidence_router_24k \
RUN_NAME=CL39_clean_full_replay_r1 \
bash diffusion_template/launchers/active/run_clean_full_config_1gpu.sh
```

The launcher composes the allowlisted config, runs the matching dataset
preflight, creates `saved/<run_name>/comet_experiment.json`, verifies that
Comet wrote a 32-character immutable key, and finalizes the fixed-96 face
quality panel after a successful run. Historical launchers are available in
git history and are intentionally absent from `clean_full`.
