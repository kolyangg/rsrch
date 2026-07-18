#!/usr/bin/env bash
set -euo pipefail

# 2-GPU machine, physical GPU 0: normalized reference ROI, absolute BA merge.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2a_packed_roi"
export NN2_RUN_NAME_DEFAULT="ba_NN2a_packed_roi_1gpu"
export NN2_DEFAULT_GPU="0"
export NN2_DEFAULT_PORT="29721"
export NN2_DESCRIPTION="NN2a: normalized 8x8 reference ROI with absolute spatial BA"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
