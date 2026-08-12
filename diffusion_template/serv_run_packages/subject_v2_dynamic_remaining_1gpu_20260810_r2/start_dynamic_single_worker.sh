#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="subject_v2_dynamic_remaining_1gpu_20260810_r2"
PACKAGE_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package"
WORKER_SCRIPT="${PACKAGE_ROOT}/start_subject_v2_historical_backfill_11runs_5workers.sh"
STAGING_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/staging"
DYNAMIC_CLAIM_ROOT="${OWNER_ROOT}/analysis_jobs/${TASK_NAME}/dynamic_run_claims"

test -s "${WORKER_SCRIPT}"
test -d "${STAGING_ROOT}"
mkdir -p "${DYNAMIC_CLAIM_ROOT}"

BACKFILL_TASK_NAME="${TASK_NAME}" \
  BACKFILL_PACKAGE_ROOT="${PACKAGE_ROOT}" \
  BACKFILL_STAGING_ROOT="${STAGING_ROOT}" \
  BACKFILL_DYNAMIC_CLAIM_ROOT="${DYNAMIC_CLAIM_ROOT}" \
  BACKFILL_WAVE=dynamic_remaining_8gpu \
  BACKFILL_WORKER_COUNT=8 \
  bash "${WORKER_SCRIPT}"
