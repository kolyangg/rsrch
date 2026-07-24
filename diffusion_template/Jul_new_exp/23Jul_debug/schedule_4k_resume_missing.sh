#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULER_DIR="${HERE}/scheduler_4k"
mkdir -p "${SCHEDULER_DIR}"
STATUS="${SCHEDULER_DIR}/QUEUE_RESUME_STATUS.md"

# Priority 1 and priority 4 have completed training.  This queue contains
# every missing pair, in original priority order.  Native entries use the
# frozen 4k registry definitions; matrix entries derive a 4k run from the
# named 600-step architecture.
PAIRS=(
    "2|native|L4_O2_oneid_projection_blend20|||native|L4_C2_large_projection_blend20||"
    "3|native|L4_O3_oneid_ref_value_blend20|||native|L4_C3_large_ref_value_blend20||"
    "5|matrix|L4_O5_oneid_active_up_blend20|E11_active_up_blended20|one_id_nm0005092_subset8_distinct|matrix|L4_C5_large_active_up_blend20|E11_active_up_blended20|cosmic_large_id00081"
    "6|matrix|L4_O6_oneid_active_up_schedule|E07_schedule_matched_up|one_id_nm0005092_subset8_distinct|matrix|L4_C6_large_active_up_schedule|E07_schedule_matched_up|cosmic_large_id00081"
    "7|matrix|L4_O7_oneid_noise_damped|E04_noise_damped|one_id_nm0005092_subset8_distinct|matrix|L4_C7_large_noise_damped|E04_noise_damped|cosmic_large_id00081"
    "8|matrix|L4_O8_oneid_all_blend20|E05_blended20|one_id_nm0005092_subset8_distinct|matrix|L4_C8_large_all_blend20|E05_blended20|cosmic_large_id00081"
    "9|matrix|L4_O9_oneid_projection_teacher20|E14_projection_split_pm_teacher20|one_id_nm0005092_subset8_distinct|matrix|L4_C9_large_projection_teacher20|E14_projection_split_pm_teacher20|cosmic_large_id00081"
    "10|matrix|L4_O10_oneid_control|E00_control|one_id_nm0005092_subset8_distinct|matrix|L4_C10_large_control|E00_control|cosmic_large_id00081"
    "11|matrix|L4_O11_oneid_active_up|E01_active_up|one_id_nm0005092_subset8_distinct|matrix|L4_C11_large_active_up|E01_active_up|cosmic_large_id00081"
    "12|matrix|L4_O12_oneid_up1_detail|E02_up1_detail|one_id_nm0005092_subset8_distinct|matrix|L4_C12_large_up1_detail|E02_up1_detail|cosmic_large_id00081"
    "13|matrix|L4_O13_oneid_staged_up1_up0|E03_staged_up1_up0|one_id_nm0005092_subset8_distinct|matrix|L4_C13_large_staged_up1_up0|E03_staged_up1_up0|cosmic_large_id00081"
)

completed_training_exists() {
    local run_id="$1"
    local manifest
    while IFS= read -r manifest; do
        if [[ "$(jq -r '.architecture_id // empty' "${manifest}")" == "${run_id}" ]] \
            && [[ "$(jq -r '.status // empty' "${manifest}")" == "completed" ]] \
            && [[ "$(find "$(dirname -- "${manifest}")/checkpoints" -type f \
                -name 'checkpoint-epoch8.pth' -print -quit 2>/dev/null)" != "" ]]; then
            return 0
        fi
    done < <(
        find "${HERE}/experiments_4k" -mindepth 2 -maxdepth 2 \
            -name run_manifest.json -type f 2>/dev/null
    )
    return 1
}

launch_arm() {
    local kind="$1"
    local run_id="$2"
    local base="$3"
    local dataset="$4"
    local slot="$5"
    if [[ "${kind}" == "native" ]]; then
        "${HERE}/run_4k_arm.sh" "${run_id}"
    else
        "${HERE}/run_4k_matrix_arm.sh" \
            "${run_id}" "${base}" "${dataset}" "${slot}"
    fi
}

{
    echo "# Resumed 4k queue status"
    echo
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "Completed priority pairs 1 and 4 are intentionally absent."
} >"${STATUS}"

for pair in "${PAIRS[@]}"; do
    IFS='|' read -r priority left_kind left_id left_base left_dataset \
        right_kind right_id right_base right_dataset <<<"${pair}"

    if completed_training_exists "${left_id}" \
        && completed_training_exists "${right_id}"; then
        {
            echo
            echo "## Priority pair ${priority}"
            echo
            echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): skipped; both trainings already complete"
        } >>"${STATUS}"
        continue
    fi

    left_log="${SCHEDULER_DIR}/${left_id}.log"
    right_log="${SCHEDULER_DIR}/${right_id}.log"
    launch_arm "${left_kind}" "${left_id}" "${left_base}" \
        "${left_dataset}" 0 >"${left_log}" 2>&1 &
    left_pid=$!
    launch_arm "${right_kind}" "${right_id}" "${right_base}" \
        "${right_dataset}" 1 >"${right_log}" 2>&1 &
    right_pid=$!
    {
        echo
        echo "## Priority pair ${priority}"
        echo
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${left_id}\` PID ${left_pid}"
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${right_id}\` PID ${right_pid}"
    } >>"${STATUS}"

    set +e
    wait "${left_pid}"
    left_status=$?
    wait "${right_pid}"
    right_status=$?
    set -e
    echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): pair finished; statuses ${left_status}/${right_status}" \
        >>"${STATUS}"
    if (( left_status != 0 || right_status != 0 )); then
        echo "Priority pair ${priority} failed; see ${left_log} and ${right_log}" >&2
        exit 2
    fi
    /home/niko/miniconda3/envs/photomaker_NS/bin/python \
        "${HERE}/export_4k_results.py" >>"${STATUS}"
done

echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${STATUS}"
