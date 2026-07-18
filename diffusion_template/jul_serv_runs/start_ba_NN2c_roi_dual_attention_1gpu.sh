#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 0: normalized ROI + dual target/reference attention.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2c_roi_dual_attention"
export NN2_RUN_NAME_DEFAULT="ba_NN2c_roi_dual_attention_1gpu"
export NN2_DEFAULT_GPU="0"
export NN2_DEFAULT_PORT="29723"
export NN2_DESCRIPTION="NN2c: normalized reference ROI plus dual per-head attention arbitration"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
