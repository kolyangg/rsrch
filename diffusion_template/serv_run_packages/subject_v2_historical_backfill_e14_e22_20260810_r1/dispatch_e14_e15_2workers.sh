#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="subject_v2_historical_backfill_e14_e15_2gpu_20260810_r1"
PACKAGE_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package"
WORKER_SCRIPT="${PACKAGE_ROOT}/start_subject_v2_historical_backfill_11runs_5workers.sh"

job_prefix="${HOSTNAME%-mpimaster-0}"
[[ "${job_prefix}" != "${HOSTNAME}" ]]
test -s "${WORKER_SCRIPT}"

run_worker() {
  BACKFILL_TASK_NAME="${TASK_NAME}" \
  BACKFILL_PACKAGE_ROOT="${PACKAGE_ROOT}" \
  BACKFILL_WAVE=e14_e15 \
  BACKFILL_WORKER_COUNT=2 \
  bash "${WORKER_SCRIPT}"
}

run_worker &
master_pid="$!"
ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=20 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${job_prefix}-mpiworker-0" \
  "BACKFILL_TASK_NAME='${TASK_NAME}' BACKFILL_PACKAGE_ROOT='${PACKAGE_ROOT}' BACKFILL_WAVE=e14_e15 BACKFILL_WORKER_COUNT=2 bash '${WORKER_SCRIPT}'" &
worker_pid="$!"

failed=0
wait "${master_pid}" || failed=1
wait "${worker_pid}" || failed=1
exit "${failed}"
