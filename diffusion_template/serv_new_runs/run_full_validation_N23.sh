#!/usr/bin/env bash
# Generate the 96-image full validation set for N23 at 1k, 5k, and 10k steps.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

N23_INFER_OVERRIDES="pipeline.ba_enable_runtime_sa_knobs=true model.ba_enable_runtime_sa_knobs=true pipeline.pose_adapt_ratio=0 pipeline.ca_mixing_for_face=true pipeline.ba_face_fusion_mode=legacy model.ba_face_fusion_mode=legacy"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/full_validation_results/ba_camix_train_N23_steps}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
EXTRA_INFER_OVERRIDES="${N23_INFER_OVERRIDES}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash serv_new_runs/run_full_validation_steps.sh \
    ba_camix_train_N23 1000 5000 10000
