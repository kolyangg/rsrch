#!/usr/bin/env bash
# Wait for N27 training, then generate 96 images plus metrics at 1k, 5k, and 10k.

set -euo pipefail

RUN_NAME="ba_spatial_roi_residual_N27"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/full_validation_results/${RUN_NAME}_steps}"
LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/run_after_training_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${DETACHED_RUN:-0}" != "1" ]]; then
    mkdir -p "${RESULTS_DIR}"
    echo "Queueing ${RUN_NAME} validation on GPU ${CUDA_VISIBLE_DEVICES:-0}"
    echo "Log: ${LOG_FILE}"
    DETACHED_RUN=1 RESULTS_DIR="${RESULTS_DIR}" LOG_FILE="${LOG_FILE}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        nohup bash "${SCRIPT_PATH}" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "PID: $!"
    echo "Follow with: tail -f ${LOG_FILE}"
    exit 0
fi

cd "${PROJECT_DIR}"
TRAIN_PATTERN='train.py.*--config-name=one_id_ba_spatial_roi_residual_N27'

while pgrep -f "${TRAIN_PATTERN}" >/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] N27 training still active; checking again in 60s"
    sleep 60
done

FINAL_CHECKPOINT="${PROJECT_DIR}/saved/${RUN_NAME}/weights-epoch10.pth"
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
    FINAL_CHECKPOINT="${PROJECT_DIR}/saved/${RUN_NAME}/checkpoint-epoch10.pth"
fi
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
    echo "N27 training stopped without an epoch-10 checkpoint; validation not started." >&2
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] N27 training complete; starting full validation on GPU ${CUDA_VISIBLE_DEVICES:-0}"
RESULTS_DIR="${RESULTS_DIR}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash serv_new_runs/run_full_validation_steps.sh \
    "${RUN_NAME}" 1000 5000 10000
