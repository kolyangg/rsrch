#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_PID="${1:?usage: schedule_full_oneid_priority_after_pair2.sh QUEUE_PID}"
STATUS="${HERE}/scheduler_4k/FULL_ONEID_PRIORITY_STATUS.md"
RUN_ID="L4_OF1_oneid_full18_projection_alt"
BASE_ARCH="L4_O1_oneid_projection_alt"
PROFILE="one_id_nm0005092_full18_heldout_distinct"

resume_main_queue() {
    if kill -0 "${QUEUE_PID}" 2>/dev/null; then
        kill -CONT "${QUEUE_PID}" 2>/dev/null || true
    fi
}
trap resume_main_queue EXIT

{
    echo "# Full-OneID priority run"
    echo
    echo "Armed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "The main queue PID \`${QUEUE_PID}\` is paused at its current pair."
    echo "Its child training/watcher processes continue normally."
} >"${STATUS}"

while true; do
    complete=0
    for architecture in \
        L4_O2_oneid_projection_blend20 \
        L4_C2_large_projection_blend20; do
        audit="$(
            find "${HERE}/experiments_4k" -mindepth 3 -maxdepth 3 \
                -path "*/report/comet_unity_audit.json" -type f \
                -print 2>/dev/null \
                | while IFS= read -r candidate; do
                    manifest="$(dirname -- "$(dirname -- "${candidate}")")/run_manifest.json"
                    if [[ "$(jq -r '.architecture_id // empty' "${manifest}")" == "${architecture}" ]] \
                        && [[ "$(jq -r '.status // empty' "${manifest}")" == "completed" ]] \
                        && [[ "$(jq -r '.status // empty' "${candidate}")" == "PASS" ]]; then
                        echo "${candidate}"
                        break
                    fi
                done
        )"
        if [[ -n "${audit}" ]]; then
            complete=$((complete + 1))
        fi
    done
    if (( complete == 2 )); then
        break
    fi
    sleep 20
done

echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): active pair 2 completed and audited" \
    >>"${STATUS}"

set +e
"${HERE}/run_4k_matrix_arm.sh" \
    "${RUN_ID}" "${BASE_ARCH}" "${PROFILE}" 0 \
    >>"${HERE}/scheduler_4k/${RUN_ID}.log" 2>&1
run_status=$?
set -e
echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): \`${RUN_ID}\` finished with status ${run_status}" \
    >>"${STATUS}"

resume_main_queue
trap - EXIT
echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): main queue resumed" >>"${STATUS}"
exit "${run_status}"
