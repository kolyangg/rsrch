#!/usr/bin/env bash
set -euo pipefail

# 2-GPU machine, physical GPU 1: separate target/reference attention + head gates.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2b_dual_attention"
export NN2_RUN_NAME_DEFAULT="ba_NN2b_dual_attention_1gpu"
export NN2_DEFAULT_GPU="1"
export NN2_DEFAULT_PORT="29722"
export NN2_DESCRIPTION="NN2b: dual target/reference attention with bounded per-head arbitration"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
