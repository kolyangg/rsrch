#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/full_validation_results/ba_qformer_idtokens_N29_steps}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
BATCH_SIZE="${BATCH_SIZE:-12}" \
bash serv_new_runs/run_full_validation_steps.sh \
    ba_qformer_idtokens_N29 2000 4000 6000 8000 10000

