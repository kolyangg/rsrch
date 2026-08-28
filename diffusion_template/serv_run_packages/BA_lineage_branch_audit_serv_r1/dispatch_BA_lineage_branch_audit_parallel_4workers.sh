#!/usr/bin/env bash
# Fan four deterministic CL27/CL39 audit shards across one MLS 4-worker job.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/BA_lineage_branch_audit_serv_r1"
WORKER_SCRIPT="${TASK_ROOT}/package/start_BA_lineage_branch_audit_parallel_worker.sh"

cd "${TASK_ROOT}"
sha256sum -c parallel_package_manifest.sha256
job_prefix="${HOSTNAME%-mpimaster-0}"
if [[ "${job_prefix}" == "${HOSTNAME}" ]]; then
  echo "Parallel dispatcher expected an MLS mpimaster hostname, found ${HOSTNAME}." >&2
  exit 80
fi
test -s "${WORKER_SCRIPT}"

declare -a child_pids=()
declare -a child_names=()
bash "${WORKER_SCRIPT}" &
child_pids+=("$!")
child_names+=("mpimaster-0")
for worker_index in 0 1 2; do
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
    echo "BA_PARALLEL_NODE_COMPLETE node=${child_names[${index}]}"
  else
    status="$?"
    echo "BA_PARALLEL_NODE_FAILED node=${child_names[${index}]} exit=${status}" >&2
    failed=1
  fi
done
if [[ "${failed}" -eq 0 ]]; then
  printf 'complete\n' > "${TASK_ROOT}/PARALLEL_AUDIT_COMPLETE"
fi
exit "${failed}"
