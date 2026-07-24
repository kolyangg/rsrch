#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${HERE}/../.." && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
RUN_DIR="$(
    cd -- "${HERE}/experiments_4k/20260724T085838Z__L4_OF1_oneid_full18_projection_alt" \
        && pwd
)"
STATUS="${HERE}/scheduler_4k/FULL_ONEID_ONLY_RESUME_STATUS.md"
source "${HERE}/comet_credentials.sh"
export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${PROJECT}/setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${PROJECT}:${PYTHONPATH:-}"

{
    echo "# Full-OneID-only detached resume"
    echo
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "Run directory: \`${RUN_DIR}\`"
    echo
    echo "Resume source: step-1000 / epoch-2 checkpoint."
    echo
    echo "All training and validation logging must reuse Comet key"
    echo "\`e0ac0450df424c9a9de854d1715abcb5\`."
} >"${STATUS}"

"${HERE}/launch_training.py" L4_O1_oneid_projection_alt \
    --run-id L4_OF1_oneid_full18_projection_alt \
    --protocol-id 4k \
    --dataset-profile one_id_nm0005092_full18_heldout_distinct \
    --port-slot 0 \
    --resume-run-dir "${RUN_DIR}" &
TRAIN_PID=$!
echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): trainer PID ${TRAIN_PID}" >>"${STATUS}"

set +e
"${HERE}/watch_validate_4k.sh" "${RUN_DIR}" "${TRAIN_PID}"
WATCH_STATUS=$?
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e

{
    echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): trainer/watcher statuses ${TRAIN_STATUS}/${WATCH_STATUS}"
} >>"${STATUS}"
if (( TRAIN_STATUS != 0 || WATCH_STATUS != 0 )); then
    exit 2
fi
echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${STATUS}"
