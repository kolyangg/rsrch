# PhotoMaker branched-attention: clean_full

This branch is the unified training and validation code base for PM0,
CL14/19/23/27/39, CL40-CL45, E13, and the four supported BigCelebs/E13 dataset
arms. Every scientific choice is made by a reviewed Hydra config.

## Entry points

- `train.py` — shared Hydra/Accelerate entry point.
- `src/configs/clean_full_runs.json` — supported run manifest and immutable
  historical Comet references.
- `tools/validate_clean_full_config.py` — fail-closed composition gate.
- `launchers/active/run_clean_full_config_1gpu.sh` — sole supported one-GPU
  Serv launcher.
- `analysis/2026-08-22_clean_full_code_structure_and_run_inventory.md` —
  file/class/function ownership and excluded-code inventory.

Run from this directory:

```bash
conda activate photomaker
cp .env.example .env
chmod 600 .env
python tools/validate_clean_full_config.py --list

CONFIG_NAME=CL39_cosmic_null_key_confidence_router_24k \
RUN_NAME=CL39_clean_full_replay_r1 \
bash launchers/active/run_clean_full_config_1gpu.sh
```

The launcher rejects arbitrary Hydra overrides, resolves the selected dataset
from the config, runs its sealed preflight, creates the canonical run record,
verifies Comet registration, and finalizes face quality after successful
training. Machine paths and credentials live only in `.env`.

Run comparison/report commands through the tools documented in `TOOLS.md`.
Historical implementations and generated artifacts can be recovered from the
`test` branch or earlier commits; they are not runtime dependencies here.
