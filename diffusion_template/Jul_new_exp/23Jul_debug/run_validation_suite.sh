#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${HERE}/../.." && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
source "${HERE}/comet_credentials.sh"

if [[ -z "${COMET_API_KEY:-}" ]]; then
    MAIN_PID="$(pgrep -f '/home/niko/miniconda3/envs/photomaker_NS/bin/python -u train.py.*ba_N3a_new1' | head -n 1 || true)"
    if [[ -n "${MAIN_PID}" && -r "/proc/${MAIN_PID}/environ" ]]; then
        while IFS= read -r ENV_LINE; do
            case "${ENV_LINE}" in
                COMET_API_KEY=*) export "${ENV_LINE}" ;;
            esac
        done < <(tr '\0' '\n' <"/proc/${MAIN_PID}/environ")
    fi
fi

if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "COMET_API_KEY unavailable; export it or keep the production run alive." >&2
    exit 2
fi

export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${PROJECT}/setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${PROJECT}:${PYTHONPATH:-}"
exec "${ENV_BIN}/python" "${HERE}/launch_validation.py" "$@"
