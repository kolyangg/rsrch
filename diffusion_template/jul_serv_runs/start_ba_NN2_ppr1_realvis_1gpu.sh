#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_RUN_NAME_DEFAULT="ba_NN2_ppr1_realvis_1gpu"
export NN1_DEFAULT_GPU="0"
export NN1_DEFAULT_PORT="29621"
export NN1_DESCRIPTION="NN2-PPR1: up-block packed-reference residual; RealVis validation"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
export NN1_VALIDATION_MODEL="SG161222/RealVisXL_V4.0"

source "${SCRIPT_DIR}/start_ba_NN2_ppr1_1gpu.sh" "$@"
