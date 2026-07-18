#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 1: target geometry down/mid, reference BA in up.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2d_upblock_reference"
export NN2_RUN_NAME_DEFAULT="ba_NN2d_upblock_reference_1gpu"
export NN2_DEFAULT_GPU="1"
export NN2_DEFAULT_PORT="29724"
export NN2_DESCRIPTION="NN2d: target-owned down/mid geometry with reference BA in up blocks"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
