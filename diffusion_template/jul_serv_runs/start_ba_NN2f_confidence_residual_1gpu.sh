#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 3: confidence-gated, zero-init packed-ROI BA residual.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2f_confidence_residual"
export NN2_RUN_NAME_DEFAULT="ba_NN2f_confidence_residual_1gpu"
export NN2_DEFAULT_GPU="3"
export NN2_DEFAULT_PORT="29726"
export NN2_DESCRIPTION="NN2f: target anchor plus confidence-gated zero-init packed-ROI BA residual"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
