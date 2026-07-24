#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${HERE}/../.." && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
LOG="${HERE}/scheduler_4k/REPAIR_COMPLETED_STATUS.md"

source "${HERE}/comet_credentials.sh"
export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${PROJECT}/setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${PROJECT}:${PYTHONPATH:-}"

RUNS=(
    "${HERE}/experiments_4k/20260723T230116Z__L4_C1_large_projection_alt"
    "${HERE}/experiments_4k/20260723T230116Z__L4_O1_oneid_projection_alt"
    "${HERE}/experiments_4k/20260724T011752Z__L4_C4_large_projection_schedule"
    "${HERE}/experiments_4k/20260724T011752Z__L4_O4_oneid_projection_schedule"
)

{
    echo "# Completed-run report repair"
    echo
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >"${LOG}"

for run_dir in "${RUNS[@]}"; do
    run_name="$(jq -r '.run_name' "${run_dir}/run_manifest.json")"
    {
        echo
        echo "## ${run_name}"
        echo
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): repair started"
    } >>"${LOG}"

    if [[ ! -f "${run_dir}/report/metrics_summary.json" ]]; then
        "${ENV_BIN}/python" "${HERE}/checkpoint_diagnostics.py" "${run_dir}"
        "${ENV_BIN}/python" "${HERE}/summarize_run.py" "${run_dir}"
        "${ENV_BIN}/python" "${HERE}/visualize_pm_masks.py" \
            --run-dir "${run_dir}" \
            --output-dir "${run_dir}/report/pm_bbox_debug"
    fi
    "${ENV_BIN}/python" "${HERE}/audit_comet_unity.py" "${run_dir}"
    "${ENV_BIN}/python" "${HERE}/upload_report_to_comet.py" "${run_dir}"
    echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): audit and report upload complete" \
        >>"${LOG}"
done

"${ENV_BIN}/python" "${HERE}/export_4k_results.py" >>"${LOG}"
echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${LOG}"
