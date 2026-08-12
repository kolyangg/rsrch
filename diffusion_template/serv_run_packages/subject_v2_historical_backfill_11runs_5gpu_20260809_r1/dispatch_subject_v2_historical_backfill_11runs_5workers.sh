#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="subject_v2_historical_backfill_11runs_5gpu_20260809_r1"
WORKER_SCRIPT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}/package/start_subject_v2_historical_backfill_11runs_5workers.sh"

# MLS binary multi-worker jobs execute their YAML script on mpimaster.  Fan the
# same NFS-backed entry point to the four allocated mpiworker nodes and run the
# fifth one locally on the one-GPU mpimaster; each process atomically claims a
# distinct chain.
job_prefix="${HOSTNAME%-mpimaster-0}"
if [[ "${job_prefix}" == "${HOSTNAME}" ]]; then
  echo "Dispatcher expected an MLS mpimaster hostname, found ${HOSTNAME}." >&2
  exit 80
fi
test -s "${WORKER_SCRIPT}"

declare -a child_pids=()
declare -a child_names=()
bash "${WORKER_SCRIPT}" &
child_pids+=("$!")
child_names+=("mpimaster-0")
for worker_index in 0 1 2 3; do
  worker_host="${job_prefix}-mpiworker-${worker_index}"
  ssh \
    -p 2222 \
    -o BatchMode=yes \
    -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${worker_host}" \
    "bash '${WORKER_SCRIPT}'" &
  child_pids+=("$!")
  child_names+=("mpiworker-${worker_index}")
done

failed=0
for index in "${!child_pids[@]}"; do
  if wait "${child_pids[${index}]}"; then
    echo "BACKFILL_NODE_COMPLETE node=${child_names[${index}]}"
  else
    status="$?"
    echo "BACKFILL_NODE_FAILED node=${child_names[${index}]} exit=${status}" >&2
    failed=1
  fi
done
exit "${failed}"
