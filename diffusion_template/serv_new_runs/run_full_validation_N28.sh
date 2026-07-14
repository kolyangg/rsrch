#!/usr/bin/env bash
# Generate the 96-image full validation set and metrics for N28 at 1k, 5k, and 10k steps.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/full_validation_results/ba_idtoken_ca_residual_N28_steps}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
bash serv_new_runs/run_full_validation_steps.sh \
    ba_idtoken_ca_residual_N28 1000 5000 10000
