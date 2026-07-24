#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
source "${HERE}/comet_credentials.sh"

export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${HERE}/../../setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${HERE}/../..:${PYTHONPATH:-}"

count_images() {
    local root="$1"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return 0
    fi
    find "${root}" -type f -name '*.png' ! -name '*_mask.png' | wc -l
}

watch_run() {
    local run_dir="$1"
    local step receipt canonical_count pm_count status
    mapfile -t steps < <(
        jq -r '.protocol.validation_steps[]' "${run_dir}/run_manifest.json"
    )
    for step in "${steps[@]}"; do
        receipt="${run_dir}/report/incremental_metrics/step_$(printf '%04d' "${step}").comet_uploaded.json"
        if [[ -f "${receipt}" ]]; then
            continue
        fi
        while true; do
            canonical_count="$(
                count_images \
                    "${run_dir}/validation/canonical50/step_$(printf '%04d' "${step}")"
            )"
            pm_count="$(
                count_images "${run_dir}/validation/pmControl50/step_0000"
            )"
            if (( canonical_count == 4 && pm_count == 4 )); then
                break
            fi
            status="$(jq -r '.status' "${run_dir}/run_manifest.json")"
            if [[ "${status}" == "interrupted" ]]; then
                echo "Run interrupted before metric stage ${step}: ${run_dir}" >&2
                return 2
            fi
            sleep 15
        done
        "${ENV_BIN}/python" "${HERE}/log_validation_step_metrics.py" \
            "${run_dir}" --step "${step}"
        "${ENV_BIN}/python" "${HERE}/export_live_4k_results.py"
    done
}

if (( $# == 0 )); then
    echo "usage: backfill_live_4k_metrics.sh RUN_DIR [RUN_DIR ...]" >&2
    exit 2
fi

pids=()
for run_dir in "$@"; do
    watch_run "$(realpath "${run_dir}")" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=2
    fi
done
exit "${status}"
