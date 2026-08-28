#!/usr/bin/env bash
# Fan the replay-gated CL23 recovery across five one-GPU MLS nodes.
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/BA_lineage_branch_audit_serv_r1"
WORKER_SCRIPT="${TASK_ROOT}/package/start_BA_lineage_CL23_recovery_worker.sh"

cd "${TASK_ROOT}"
sha256sum -c cl23_recovery_package_manifest.sha256
job_prefix="${HOSTNAME%-mpimaster-0}"
if [[ "${job_prefix}" == "${HOSTNAME}" ]]; then
  echo "CL23 recovery dispatcher expected an MLS mpimaster hostname, found ${HOSTNAME}." >&2
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
    echo "BA_CL23_RECOVERY_NODE_COMPLETE node=${child_names[${index}]}"
  else
    status="$?"
    echo "BA_CL23_RECOVERY_NODE_FAILED node=${child_names[${index}]} exit=${status}" >&2
    failed=1
  fi
done
if [[ "${failed}" -eq 0 ]]; then
  printf 'complete\n' > "${TASK_ROOT}/CL23_RECOVERY_COMPLETE"
fi
exit "${failed}"
