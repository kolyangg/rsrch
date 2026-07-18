#!/usr/bin/env bash
set -euo pipefail

# 4-GPU machine, physical GPU 2: reference core, target-owned boundary ring.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NN2_CONFIG_NAME="one_id_ba_NN2e_core_ring"
export NN2_RUN_NAME_DEFAULT="ba_NN2e_core_ring_1gpu"
export NN2_DEFAULT_GPU="2"
export NN2_DEFAULT_PORT="29725"
export NN2_DESCRIPTION="NN2e: inner reference identity core with target-owned face boundary ring"
export NN2_LAUNCHER_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_run_ba_NN2_common_1gpu.sh" "$@"
