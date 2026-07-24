#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:?usage: run_4k_matrix_arm.sh RUN_ID BASE_ARCH DATASET_PROFILE PORT_SLOT}"
BASE_ARCH="${2:?usage: run_4k_matrix_arm.sh RUN_ID BASE_ARCH DATASET_PROFILE PORT_SLOT}"
DATASET_PROFILE="${3:?usage: run_4k_matrix_arm.sh RUN_ID BASE_ARCH DATASET_PROFILE PORT_SLOT}"
PORT_SLOT="${4:?usage: run_4k_matrix_arm.sh RUN_ID BASE_ARCH DATASET_PROFILE PORT_SLOT}"
source "${HERE}/comet_credentials.sh"

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_DIR="${HERE}/experiments_4k/${STAMP}__${RUN_ID}"
mkdir -p "${HERE}/experiments_4k"

"${HERE}/run_architecture.sh" "${BASE_ARCH}" \
    --run-id "${RUN_ID}" \
    --protocol-id 4k \
    --dataset-profile "${DATASET_PROFILE}" \
    --port-slot "${PORT_SLOT}" \
    --run-dir "${RUN_DIR}" &
TRAIN_PID=$!
echo "run_id=${RUN_ID} base=${BASE_ARCH} dataset=${DATASET_PROFILE} run_dir=${RUN_DIR} train_pid=${TRAIN_PID}"

set +e
"${HERE}/watch_validate_4k.sh" "${RUN_DIR}" "${TRAIN_PID}"
WATCH_STATUS=$?
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
if (( TRAIN_STATUS != 0 || WATCH_STATUS != 0 )); then
    echo "4k matrix arm failed: ${RUN_ID} train=${TRAIN_STATUS} watcher=${WATCH_STATUS}" >&2
    exit 2
fi
echo "4k matrix arm complete: ${RUN_ID} ${RUN_DIR}"
