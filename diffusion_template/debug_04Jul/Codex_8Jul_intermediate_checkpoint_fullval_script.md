# Codex 8 Jul — intermediate checkpoint full-validation script

Added:

```text
serv_new_runs/run_full_validation_steps.sh
```

Purpose: run `inference/full_val` on selected intermediate checkpoints from one saved run, using
step numbers rather than manually finding checkpoint epoch numbers.

## Usage

From repo root on the machine that has the checkpoints:

```bash
cd /home/kolyangg/rsrch/diffusion_template
BATCH_SIZE=4 bash serv_new_runs/run_full_validation_steps.sh ba_longrun_N17 8000 10000 12000 14000 16000
```

The first argument is the run folder under `saved/`. Remaining arguments are training steps.

The script reads `trainer.epoch_len` from:

```text
saved/<run_name>/config.yaml
```

Then it maps:

```text
epoch = step / epoch_len
checkpoint = saved/<run_name>/weights-epoch<epoch>.pth
```

If `weights-epoch<epoch>.pth` is missing, it falls back to `checkpoint-epoch<epoch>.pth`.

## Outputs

For each requested step, images go to a separate report-compatible folder:

```text
full_validation_results/<run_name>_step<step>/
```

Example:

```text
full_validation_results/ba_longrun_N17_step16000/
```

Metrics are appended to:

```text
full_validation_results/metrics_<run_name>_steps.json
```

Example:

```text
full_validation_results/metrics_ba_longrun_N17_steps.json
```

A timestamped log is written to:

```text
full_validation_results/run_full_validation_steps_<run_name>_<timestamp>.log
```

## Useful overrides

```bash
BATCH_SIZE=2                       # reduce GPU memory
CUDA_VISIBLE_DEVICES=1             # choose GPU
METRICS_JSON=full_validation_results/metrics_N17_intermediate.json
RESULTS_DIR=full_validation_results
PYTHON_BIN=python                  # or python3
```

## Notes

- Requested steps must be divisible by `trainer.epoch_len`; otherwise the script skips them.
- Existing output folders with at least 96 PNG images are not regenerated; metrics are recomputed.
- The output folder names can be added directly to `infer_tools/full_val_report.yaml` as runs if
  they should appear as columns in `full_val_report.pdf`.
