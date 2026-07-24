#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${HERE}/../.." && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
RUN_DIR="${1:?usage: watch_validate_4k.sh RUN_DIR TRAIN_PID}"
TRAIN_PID="${2:?usage: watch_validate_4k.sh RUN_DIR TRAIN_PID}"
source "${HERE}/comet_credentials.sh"

export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${PROJECT}/setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${PROJECT}:${PYTHONPATH:-}"

while [[ ! -f "${RUN_DIR}/run_manifest.json" ]]; do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "Trainer exited before creating a manifest: ${RUN_DIR}" >&2
        exit 2
    fi
    sleep 2
done

RUN_NAME="$(jq -r '.run_name' "${RUN_DIR}/run_manifest.json")"
CHECKPOINT_EVERY="$(jq -r '.protocol.checkpoint_every' "${RUN_DIR}/run_manifest.json")"
mapfile -t STAGES < <(
    jq -r '.protocol.validation_steps[] | select(. > 0)' \
        "${RUN_DIR}/run_manifest.json"
)
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${RUN_NAME}"
PROGRESS="${RUN_DIR}/VALIDATION_PROGRESS.md"

wait_for_stable_checkpoint() {
    local checkpoint="$1"
    local previous_size=-1
    local stable_count=0
    while true; do
        if [[ -f "${checkpoint}" ]]; then
            current_size="$(stat -c '%s' "${checkpoint}")"
            if [[ "${current_size}" -gt 1000000 && "${current_size}" -eq "${previous_size}" ]]; then
                stable_count=$((stable_count + 1))
                if (( stable_count >= 2 )); then
                    return 0
                fi
            else
                stable_count=0
            fi
            previous_size="${current_size}"
        elif ! kill -0 "${TRAIN_PID}" 2>/dev/null || [[ "$(ps -o stat= -p "${TRAIN_PID}" 2>/dev/null)" == Z* ]]; then
            echo "Trainer exited before checkpoint appeared: ${checkpoint}" >&2
            return 2
        fi
        sleep 5
    done
}

{
    echo "# 4k validation progress"
    echo
    echo "Run: \`${RUN_NAME}\`"
    echo
    echo "All validation uses writer=console and direct upload to the training Comet key."
} >"${PROGRESS}"

for stage in "${STAGES[@]}"; do
    epoch=$((stage / CHECKPOINT_EVERY))
    checkpoint="${CHECKPOINT_DIR}/checkpoint-epoch${epoch}.pth"
    wait_for_stable_checkpoint "${checkpoint}"
    validation_manifest="$(
        printf '%s/validation/canonical50/step_%04d/validation_manifest.json' \
            "${RUN_DIR}" "${stage}"
    )"
    metric_receipt="$(
        printf '%s/report/incremental_metrics/step_%04d.comet_uploaded.json' \
            "${RUN_DIR}" "${stage}"
    )"
    comet_key="$(jq -r '.comet_experiment_key // empty' "${RUN_DIR}/run_manifest.json")"
    validation_complete=false
    metric_complete=false
    if [[ -f "${validation_manifest}" ]] \
        && [[ "$(jq -r '.status // empty' "${validation_manifest}")" == "completed" ]] \
        && [[ "$(jq -r '.comet_upload_status // empty' "${validation_manifest}")" == "completed" ]] \
        && [[ "$(jq -r '.comet_experiment_key // empty' "${validation_manifest}")" == "${comet_key}" ]]; then
        validation_complete=true
    fi
    if [[ -f "${metric_receipt}" ]] \
        && [[ "$(jq -r '.status // empty' "${metric_receipt}")" == "completed" ]] \
        && [[ "$(jq -r '.comet_experiment_key // empty' "${metric_receipt}")" == "${comet_key}" ]]; then
        metric_complete=true
    fi
    if [[ "${validation_complete}" == true && "${metric_complete}" == true ]]; then
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): checkpoint ${stage} already uploaded to original Comet key; skipped" \
            >>"${PROGRESS}"
        continue
    fi
    {
        echo
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): checkpoint ${stage} stable; validation started"
    } >>"${PROGRESS}"
    if [[ "${validation_complete}" != true ]]; then
        if [[ "${stage}" -eq "${CHECKPOINT_EVERY}" ]]; then
            "${HERE}/run_validation_suite.sh" "${RUN_DIR}" \
                --steps "0,${stage}" --modes "canonical50,pmControl50"
            "${ENV_BIN}/python" "${HERE}/log_validation_step_metrics.py" \
                "${RUN_DIR}" --step 0
        else
            "${HERE}/run_validation_suite.sh" "${RUN_DIR}" \
                --steps "${stage}" --modes "canonical50"
        fi
    else
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): checkpoint ${stage} images already uploaded; backfilling metrics only" \
            >>"${PROGRESS}"
    fi
    if [[ "${metric_complete}" != true ]]; then
        "${ENV_BIN}/python" "${HERE}/log_validation_step_metrics.py" \
            "${RUN_DIR}" --step "${stage}"
    fi
    "${ENV_BIN}/python" "${HERE}/export_live_4k_results.py"
    echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): checkpoint ${stage} validation uploaded" \
        >>"${PROGRESS}"
done

while [[ "$(jq -r '.status' "${RUN_DIR}/run_manifest.json")" == "running" ]]; do
    sleep 5
done
if [[ "$(jq -r '.status' "${RUN_DIR}/run_manifest.json")" != "completed" ]]; then
    echo "Training did not complete cleanly: ${RUN_DIR}" >&2
    exit 2
fi

"${ENV_BIN}/python" "${HERE}/checkpoint_diagnostics.py" "${RUN_DIR}"
"${ENV_BIN}/python" "${HERE}/summarize_run.py" "${RUN_DIR}"
"${ENV_BIN}/python" "${HERE}/visualize_pm_masks.py" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${RUN_DIR}/report/pm_bbox_debug"
"${ENV_BIN}/python" "${HERE}/audit_comet_unity.py" "${RUN_DIR}"
"${ENV_BIN}/python" "${HERE}/upload_report_to_comet.py" "${RUN_DIR}"
echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): report, scalar metrics, and Comet audit complete" \
    >>"${PROGRESS}"
