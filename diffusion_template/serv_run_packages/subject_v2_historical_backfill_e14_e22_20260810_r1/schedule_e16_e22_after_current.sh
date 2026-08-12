#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
MLS="${OWNER_ROOT}/conda_env/nasilaev/bin/mls"
CURRENT_JOB="lm-mpi-job-44b99a20-a6ad-4023-b3c6-f249b1abe83d"
DELAY_SECONDS=1200
REQUESTED_GPUS=5
GPU_CEILING=10
YAML="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_e14_e22_20260810_r1/package/run_e16_e22_5workers.yaml"
STATE="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_e14_e22_20260810_r1/scheduler"
mkdir -p "${STATE}"
test -s "${YAML}"

while true; do
  status_json="$(${MLS} job status "${CURRENT_JOB}" -O json)"
  status="$(printf '%s' "${status_json}" | "${OWNER_ROOT}/conda_env/nasilaev/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["status"].lower())')"
  printf '%s status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" | tee -a "${STATE}/scheduler.log"
  case "${status}" in
    succeeded|completed)
      break
      ;;
    failed|stopped|deleted|terminated)
      echo "Current backfill ended unsuccessfully; delayed submission blocked." | tee -a "${STATE}/scheduler.log"
      exit 2
      ;;
  esac
  sleep 60
done

ready_at=$(($(date +%s) + DELAY_SECONDS))
printf '%s\n' "${ready_at}" > "${STATE}/ready_at_epoch.txt"
while (( $(date +%s) < ready_at )); do sleep 60; done

while true; do
  running="$(${MLS} job list -s Running -l 100 -O json)"
  pending="$(${MLS} job list -s Pending -l 100 -O json)"
  project_gpus="$(printf '%s\n%s' "${running}" "${pending}" | "${OWNER_ROOT}/conda_env/nasilaev/bin/python" -c '
import json,sys
text=sys.stdin.read(); dec=json.JSONDecoder(); pos=0; total=0
while pos < len(text):
    while pos < len(text) and text[pos].isspace(): pos += 1
    if pos >= len(text): break
    obj,pos = dec.raw_decode(text,pos)
    total += sum(int(j.get("gpu_count",0)) for j in obj.get("jobs",[]) if "#nasilaev" in str(j.get("job_desc","")))
print(total)')"
  printf '%s project_gpus=%s requested=%s ceiling=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${project_gpus}" "${REQUESTED_GPUS}" "${GPU_CEILING}" | tee -a "${STATE}/scheduler.log"
  if (( project_gpus + REQUESTED_GPUS <= GPU_CEILING )); then
    break
  fi
  sleep 60
done

# One submission attempt only: an MLS limit rejection is not retried automatically.
${MLS} job submit --config "${YAML}" -O json | tee "${STATE}/submit_output.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE}/submitted_at.txt"
