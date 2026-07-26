#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/niko/rsrch/diffusion_template"
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
RUN_ID="rhca_cosmic_full_crop20_legacy_4k_batched_r1_full96_eval"

# shellcheck disable=SC1090
source "${CONDA_INIT}"
conda activate photomaker_NS
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a
export ENV_FILE=/dev/null
export PM_PATH="/home/niko/models/PhotoMaker-V2/photomaker-v2.bin"
export CUDA_VISIBLE_DEVICES=0
# 26 Jul 2026 - AICODE-NOTE: Hydra still instantiates the configured training
# dataset during an evaluation-only run, so the source full-Cosmic paths must
# be explicit instead of inheriting the fail-closed /nonexistent defaults.
export COSMIC_LARGE_MANIFEST="/home/niko/datasets/gathered_data_cosmic_large_filtered.json"
export COSMIC_LARGE_ROOT="/home/niko/datasets"
export RUN_NAME="${RUN_ID}"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop20_legacy_4k_batched_r1"
export VALIDATION_SOURCE_COMET_KEY="7ec45fae85684aac97b2266967adbe2a"
export FULL96_BBOX_MANUAL="/home/niko/rsrch/dataset_full/val_dataset/protocols/cosmic_full96_auto_v1/pm96_bboxes_new.json"
export EXPERIMENT_SPEC_PATH="${PROJECT_ROOT}/experiments/cosmic_large_adaptation/${RUN_ID}.json"

exec bash launchers/active/run_rhca_cosmic_full96_eval_1gpu.sh
