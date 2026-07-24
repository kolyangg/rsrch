#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHITECTURE="${1:?usage: run_4k_arm.sh ARCHITECTURE}"
source "${HERE}/comet_credentials.sh"

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_DIR="${HERE}/experiments_4k/${STAMP}__${ARCHITECTURE}"
mkdir -p "${HERE}/experiments_4k"

"${HERE}/run_architecture.sh" "${ARCHITECTURE}" --run-dir "${RUN_DIR}" &
TRAIN_PID=$!
echo "run_dir=${RUN_DIR} train_pid=${TRAIN_PID}"

set +e
"${HERE}/watch_validate_4k.sh" "${RUN_DIR}" "${TRAIN_PID}"
WATCH_STATUS=$?
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
if (( TRAIN_STATUS != 0 || WATCH_STATUS != 0 )); then
    echo "4k arm failed: ${ARCHITECTURE} train=${TRAIN_STATUS} watcher=${WATCH_STATUS}" >&2
    exit 2
fi
echo "4k arm complete: ${ARCHITECTURE} ${RUN_DIR}"
