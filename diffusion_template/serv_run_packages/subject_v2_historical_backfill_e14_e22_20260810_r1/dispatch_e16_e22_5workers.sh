#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="subject_v2_historical_backfill_cl10_cl14_e14_e22_5gpu_20260810_r1"
PACKAGE_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package"
WORKER_SCRIPT="${PACKAGE_ROOT}/start_subject_v2_historical_backfill_11runs_5workers.sh"

job_prefix="${HOSTNAME%-mpimaster-0}"
[[ "${job_prefix}" != "${HOSTNAME}" ]]
test -s "${WORKER_SCRIPT}"

BACKFILL_TASK_NAME="${TASK_NAME}" BACKFILL_PACKAGE_ROOT="${PACKAGE_ROOT}" \
  BACKFILL_WAVE=cl10_cl14_then_e14_e22 BACKFILL_WORKER_COUNT=5 bash "${WORKER_SCRIPT}" &
pids=("$!")
for worker_index in 0 1 2 3; do
  ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${job_prefix}-mpiworker-${worker_index}" \
    "BACKFILL_TASK_NAME='${TASK_NAME}' BACKFILL_PACKAGE_ROOT='${PACKAGE_ROOT}' BACKFILL_WAVE=cl10_cl14_then_e14_e22 BACKFILL_WORKER_COUNT=5 bash '${WORKER_SCRIPT}'" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
