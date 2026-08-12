#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
CONDA_ENV="${OWNER_ROOT}/conda_env/photomaker_NS"
TASK_NAME="subject_v2_dynamic_remaining_1gpu_20260810_r2"
TASK_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}"
PACKAGE_ROOT="${TASK_ROOT}/package"
CLAIM_ROOT="${TASK_ROOT}/dynamic_run_claims"
SCHEDULER_ROOT="${TASK_ROOT}/scheduler"
LOG_FILE="${SCHEDULER_ROOT}/monitor.log"
OBSERVED_CLUSTER_CAPACITY=8
MAX_PROJECT_JOBS=8
TOTAL_RUNS=13
POLL_SECONDS=30

mkdir -p "${SCHEDULER_ROOT}"
exec >> "${LOG_FILE}" 2>&1

if [[ "$(readlink -f "${CONDA_PREFIX:-/nonexistent}")" != "$(readlink -f "${CONDA_ENV}")" ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR wrong Conda environment: ${CONDA_PREFIX:-<none>}"
  exit 70
fi

next_slot=5
while true; do
  running_json="${SCHEDULER_ROOT}/running.json"
  pending_json="${SCHEDULER_ROOT}/pending.json"
  mls job list --status Running --limit 100 --output json > "${running_json}"
  mls job list --status Pending --limit 100 --output json > "${pending_json}"

  read -r global_gpus project_jobs < <(
    "${CONDA_ENV}/bin/python" - "${running_json}" "${pending_json}" <<'PY'
import json
import sys

jobs = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        jobs.extend(json.load(handle).get("jobs", []))
global_gpus = sum(int(job.get("gpu_count", 0)) for job in jobs)
project_jobs = sum(
    1
    for job in jobs
    if str(job.get("job_desc", "")).startswith(
        "Subject-v2 dynamic recovery worker"
    )
)
print(global_gpus, project_jobs)
PY
  )

  claim_count="$(find "${CLAIM_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  unclaimed=$((TOTAL_RUNS - claim_count))
  (( unclaimed < 0 )) && unclaimed=0
  nonproject_gpus=$((global_gpus - project_jobs))
  available_for_project=$((OBSERVED_CLUSTER_CAPACITY - nonproject_gpus))
  (( available_for_project < 0 )) && available_for_project=0
  desired_project_jobs=$((project_jobs + unclaimed))
  (( desired_project_jobs > available_for_project )) && desired_project_jobs="${available_for_project}"
  (( desired_project_jobs > MAX_PROJECT_JOBS )) && desired_project_jobs="${MAX_PROJECT_JOBS}"

  printf '%s global_gpus=%s nonproject_gpus=%s project_jobs=%s claims=%s unclaimed=%s desired_project_jobs=%s next_slot=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${global_gpus}" "${nonproject_gpus}" \
    "${project_jobs}" "${claim_count}" "${unclaimed}" "${desired_project_jobs}" "${next_slot}"

  if (( project_jobs == 0 && unclaimed == 0 )); then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) MONITOR_COMPLETE all runs claimed and no worker remains"
    exit 0
  fi

  if (( project_jobs < desired_project_jobs && next_slot <= 8 )); then
    yaml="${PACKAGE_ROOT}/run_worker_$(printf '%02d' "${next_slot}").yaml"
    output="${SCHEDULER_ROOT}/submission_$(printf '%02d' "${next_slot}").out"
    test -s "${yaml}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMIT slot=${next_slot} yaml=${yaml}"
    if ! mls job submit --config "${yaml}" > "${output}" 2>&1; then
      cat "${output}"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMIT_REJECTED_OR_FAILED slot=${next_slot}; monitor exits without retry"
      exit 80
    fi
    cat "${output}"
    if ! grep -Eq "['\"]job_name['\"]: ['\"]lm-mpi-job-" "${output}"; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SUBMIT_UNPARSED slot=${next_slot}; monitor exits without retry"
      exit 81
    fi
    next_slot=$((next_slot + 1))
  fi

  sleep "${POLL_SECONDS}"
done
