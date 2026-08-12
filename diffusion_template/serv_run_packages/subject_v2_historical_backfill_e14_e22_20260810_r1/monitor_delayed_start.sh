#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
MLS="${OWNER_ROOT}/conda_env/nasilaev/bin/mls"
PYTHON="${OWNER_ROOT}/conda_env/nasilaev/bin/python"
STATE="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_e14_e22_20260810_r1/scheduler"

while [[ ! -s "${STATE}/submit_output.txt" ]]; do sleep 60; done
job_name="$(${PYTHON} - "${STATE}/submit_output.txt" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(r'"job_name"\s*:\s*"([^"]+)"', text)
if not matches:
    raise SystemExit("Delayed submission output has no job_name")
print(matches[-1])
PY
)"
printf '%s\n' "${job_name}" > "${STATE}/delayed_job_name.txt"

while true; do
  status_json="$(${MLS} job status "${job_name}" -O json)"
  status="$(printf '%s' "${status_json}" | ${PYTHON} -c 'import json,sys; print(json.load(sys.stdin)["status"].lower())')"
  printf '%s delayed_job=%s status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job_name}" "${status}" | tee -a "${STATE}/start_monitor.log"
  case "${status}" in
    running)
      date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE}/all_starts_confirmed_at.txt"
      exit 0
      ;;
    failed|stopped|deleted|terminated)
      echo "Delayed job failed before Running." > "${STATE}/start_monitor_failed.txt"
      exit 2
      ;;
  esac
  sleep 60
done
