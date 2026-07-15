# Comet full-validation downloader

`download_full_validation.py` downloads selected validation checkpoints from one or more Comet runs and organizes them for the existing full-validation report and metric tools.

## Config

Copy `comet_full_validation_download_template.json` and fill in the experiment keys. `run_name` may be `null`; in that case the current Comet experiment name is used. Set it only when a different local name is required.

```json
{
  "output_root": "../full_validation_results",
  "refs_dir": "../../dataset_full/val_dataset/references",
  "expected_names_json": "../../dataset_full/val_dataset/pm96_bboxes_new.json",
  "expected_images": 96,
  "default_steps": [2000, 6000, 10000],
  "strict_steps": true,
  "compute_metrics": true,
  "clean_step_dirs": false,
  "runs": [
    {
      "run_id": "N29_COMET_EXPERIMENT_KEY",
      "run_name": "ba_qformer_idtokens_N29"
    },
    {
      "run_id": "N30_COMET_EXPERIMENT_KEY",
      "run_name": "ba_bboxnorm_idtokens_N30"
    }
  ]
}
```

Each run may provide its own `steps`, `run_name`, `epoch_len`, or `compute_metrics`. If `steps` is absent, `default_steps` is used. Paths in the JSON are resolved relative to the JSON file.

Keep `strict_steps=true` for training validation. It prevents a missing full-validation step from silently falling back to step 0, which contains only the 24-image smoke set.

## Run

Use the PhotoMaker environment when local ID metrics are enabled:

```bash
cd /home/kolyangg/rsrch/diffusion_template
conda activate photomaker
export COMET_API_KEY="<your-key>"

cp comet_utils/comet_full_validation_download_template.json \
  comet_utils/comet_full_validation_N29_N30.json
# Edit comet_full_validation_N29_N30.json and insert both Comet experiment keys.

python comet_utils/download_full_validation.py \
  --config comet_utils/comet_full_validation_N29_N30.json
```

The command shows progress bars and ETA for Comet metric-history requests,
each run and requested step, image downloads, reference-face processing, and
local identity metrics. Add `--no-progress` for plain non-interactive logs.

Useful overrides:

```bash
# Select different steps for every configured run.
python comet_utils/download_full_validation.py --config <config.json> --steps 2000 10000

# Re-download existing valid images or recompute existing complete metric records.
python comet_utils/download_full_validation.py --config <config.json> --force-download
python comet_utils/download_full_validation.py --config <config.json> --force-metrics

# Download images and Comet metrics without initializing InsightFace locally.
python comet_utils/download_full_validation.py --config <config.json> --skip-local-metrics
```

Rerunning is resumable by default: valid existing images and complete per-image metric records are reused. `clean_step_dirs=true` is an explicit destructive option for configured step folders.

## Output

For a Comet run named `ba_qformer_idtokens_N29`:

```text
full_validation_results/
  ba_qformer_idtokens_N29/
    ba_qformer_idtokens_N29_step2000/   # 96 canonical PNG names
    ba_qformer_idtokens_N29_step6000/
    ba_qformer_idtokens_N29_step10000/
    metrics_ba_qformer_idtokens_N29_steps.json
    comet_export.json
  comet_full_validation_export.json
```

`comet_export.json` stores run metadata, flattened hyperparameters, complete Comet metric histories, exact requested-step metric snapshots, and per-asset download records. The separate `metrics_<run>_steps.json` uses the established local schema with aggregate, per-identity, and per-image InsightFace scores.

Comet replaces spaces in logged image names with underscores. The downloader restores exact canonical names such as `Angry man _elon.png` using `pm96_bboxes_new.json`, deduplicates repeated assets by newest creation time, rejects incomplete local sets, and computes local scores only for an exact 96-image folder.
