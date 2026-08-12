#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
JOB_ID="lm-mpi-job-7df3819d-7fdc-4ca0-a50a-b058bb254f03"
TASK_NAME="subject_v2_historical_backfill_recovery_priority_8gpu_20260810_r1"
MLS="${OWNER_ROOT}/conda_env/nasilaev/bin/mls"
PYTHON="${OWNER_ROOT}/conda_env/nasilaev/bin/python"
STATE_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}/monitor"
STAGING_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/staging"
LOG_ROOT="${OWNER_ROOT}/logs/${TASK_NAME}/${JOB_ID}"
EXPECTED_VERIFIED_RUNS=25
STALE_SECONDS=2700

mkdir -p "${STATE_ROOT}"
while true; do
  status_json="$(${MLS} job status "${JOB_ID}" -O json)"
  status="$(printf '%s' "${status_json}" | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["status"].lower())')"
  verified="$(find "${STAGING_ROOT}" -type f -name replacement_verified.json | wc -l)"
  staged="$(find "${STAGING_ROOT}" -type f -name step_manifest.json | wc -l)"
  running_workers="$({ grep -l '^running$' "${OWNER_ROOT}/analysis_jobs/${TASK_NAME}/status/${JOB_ID}"/worker_*.status 2>/dev/null || true; } | wc -l)"
  failed_workers="$({ grep -l '^failed$' "${OWNER_ROOT}/analysis_jobs/${TASK_NAME}/status/${JOB_ID}"/worker_*.status 2>/dev/null || true; } | wc -l)"
  newest_log_epoch="$(find "${LOG_ROOT}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)"
  newest_log_epoch="${newest_log_epoch:-0}"
  stale_seconds=$(($(date +%s) - newest_log_epoch))
  printf '%s status=%s verified=%s/%s staged_steps=%s workers_running=%s workers_failed=%s log_stale_seconds=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${verified}" \
    "${EXPECTED_VERIFIED_RUNS}" "${staged}" "${running_workers}" \
    "${failed_workers}" "${stale_seconds}" | tee -a "${STATE_ROOT}/monitor.log"

  if (( stale_seconds > STALE_SECONDS )) && [[ "${status}" == "running" ]]; then
    printf '%s ALERT no worker log progress for %s seconds\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stale_seconds}" | tee -a "${STATE_ROOT}/alerts.log"
  fi
  case "${status}" in
    completed|succeeded)
      if [[ "${verified}" -ne "${EXPECTED_VERIFIED_RUNS}" ]]; then
        echo "Job completed without all expected replacement audits." | tee -a "${STATE_ROOT}/alerts.log"
        exit 3
      fi
      date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE_ROOT}/complete_at.txt"
      exit 0
      ;;
    failed|stopped|deleted|terminated)
      echo "Recovery job ended unsuccessfully with status ${status}." | tee -a "${STATE_ROOT}/alerts.log"
      exit 2
      ;;
  esac
  sleep 60
done
