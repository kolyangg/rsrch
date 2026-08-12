#!/usr/bin/env bash
set -euo pipefail

OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_NAME="cl6_metric_cache_recovery_1gpu_20260810_r1"
PACKAGE_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/package"
STAGING_ROOT="${OWNER_ROOT}/analysis_jobs/subject_v2_historical_backfill_11runs_5gpu_20260809_r1/staging"
WORKER_SCRIPT="${PACKAGE_ROOT}/start_subject_v2_historical_backfill_11runs_5workers.sh"

test -s "${WORKER_SCRIPT}"
test -d "${STAGING_ROOT}"
BACKFILL_TASK_NAME="${TASK_NAME}" BACKFILL_PACKAGE_ROOT="${PACKAGE_ROOT}" \
  BACKFILL_STAGING_ROOT="${STAGING_ROOT}" \
  BACKFILL_WAVE=recovery_and_priority_8gpu BACKFILL_WORKER_COUNT=1 \
  bash "${WORKER_SCRIPT}"
