#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="cosmic_large_one_id_rhca_apr2026_replay"
export RUN_NAME="${RUN_NAME:-rhca_apr2026_cosmic_large_one_id_4k}"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

exec bash "${ROOT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" "$@"
