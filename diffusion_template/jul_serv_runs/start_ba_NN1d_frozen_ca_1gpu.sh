#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 1: all split CA active, CA weights frozen.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN1_CONFIG_NAME="one_id_ba_NN1d_frozen_ca"
export NN1_RUN_NAME_DEFAULT="ba_NN1d_frozen_ca_1gpu"
export NN1_DEFAULT_GPU="1"
export NN1_DEFAULT_PORT="29614"
export NN1_DESCRIPTION="NN1d: full spatial BA with active/frozen branched cross-attention"
export NN1_REQUIRE_ID_LOSS="0"
export NN1_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN1_common_1gpu.sh" "$@"
